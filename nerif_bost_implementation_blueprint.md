# NeRIF / BOST 毕设实施蓝图

用途：把“我准备做 BOST/NeRIF”落成一个可维护的代码项目。它连接 `starter_code_spec.md`、`data_templates/`、M0-M3B demo 和后续真实数据接入。

核心原则：先让 synthetic pipeline、baseline、neural field、metrics 和 report 自动跑通，再追求论文级 NeRIF 复现。

---

## 1. 最小可维护仓库

推荐单独建一个代码仓库，例如 `bost-nerif-bishe/`。当前工作台负责调研和材料，真正实验代码可以放新仓库。

```text
bost-nerif-bishe/
  configs/
    synthetic_m0_2d.yaml
    synthetic_m1_3d.yaml
    nerif_ablation_view_noise.yaml
    piv_bost_compensation.yaml
    fourd_lowrank.yaml
  data/
    synthetic/
    open_bos/
    oerf_sample/
  src/
    geometry/
      views.py
      rays.py
      camera_model.py
    dataio/
      manifest.py
      loaders.py
      units.py
    fields/
      phantom.py
      gradients.py
      gladstone_dale.py
    forward/
      bos_projection.py
      displacement.py
      reprojection.py
    baselines/
      voxel_tikhonov.py
      landweber.py
      sirt.py
      ubost_proxy.py
    nerif/
      encoding.py
      model.py
      losses.py
      train.py
    piv_bost/
      velocity_error.py
      compensation.py
    fourd/
      lowrank.py
      temporal_metrics.py
    eval/
      metrics.py
      plots.py
      report.py
  experiments/
    2026-07-xx_m1_view_noise/
      config.yaml
      metrics.csv
      figures/
      logs/
      notes.md
  notebooks/
    01_forward_model_sanity.ipynb
    02_baseline_vs_nerif.ipynb
    03_real_data_manifest_check.ipynb
  tests/
    test_shapes.py
    test_units.py
    test_reprojection.py
  README.md
```

本科阶段不一定要一次建满，但目录思路要稳定：**数据读取、几何、前向模型、baseline、神经场、指标、报告**不要混在一个脚本里。

---

## 2. 数据对象和形状约定

统一约定能避免后期真实数据接入时崩掉。

| 对象 | 建议变量名 | 形状 | 单位 | 备注 |
| --- | --- | --- | --- | --- |
| 折射率场 | `n_field` | `(Nz, Ny, Nx)` | dimensionless | 若只重构增量，可用 `delta_n` |
| 折射率梯度 | `grad_n` | `(Nz, Ny, Nx, 3)` | 1/m | 分量顺序 `(dz, dy, dx)` 或明确写入 config |
| 密度场 | `rho_field` | `(Nz, Ny, Nx)` | kg/m^3 | 通过 Gladstone-Dale 与 `n` 关联 |
| 温度场 | `T_field` | `(Nz, Ny, Nx)` | K | 若没有真实标定，暂不承诺定量温度 |
| BOS 位移场 | `disp_uv` | `(V, H, W, 2)` | pixel 或 mm | 分量顺序 `(u, v)`，必须写单位 |
| 视角参数 | `views` | `V` 个 dict | rad/mm/pixel | 每个 view 包含角度、投影方向、裁剪信息 |
| mask | `mask` | `(V, H, W)` 或 `(Nz, Ny, Nx)` | boolean | 视角 mask 与体 mask 分开 |
| PIV 速度场 | `velocity_uv` | `(T, H, W, 2)` | m/s | 2D PIV 先做 `(u, v)` |
| 4D 折射率场 | `n_time` | `(T, Nz, Ny, Nx)` | dimensionless | 先小尺寸，再谈真实 4D |

强制规则：

- 所有数组保存时必须配一个 `metadata.json` 或 manifest 字段。
- 坐标轴顺序不能靠记忆，必须写进 config。
- 所有位移、长度、速度都必须标单位。

---

## 3. 模块接口

### 3.1 phantom / field generation

```python
def make_phantom(config) -> dict:
    return {
        "n_field": n_field,
        "grad_n": grad_n,
        "spacing": (dz, dy, dx),
        "metadata": {...},
    }
```

验收：

- 能画中心切片。
- 能输出梯度幅值图。
- 能换 2-3 种 phantom：Gaussian blob、flame sheet、moving plume。

### 3.2 view geometry

```python
def make_views(config) -> list[dict]:
    return [
        {"view_id": "view_01", "angle_deg": 0.0, "projection": "..."},
        ...
    ]
```

验收：

- 3/5/7/9 视角可配置。
- 能和真实九视角 BOST manifest 对齐。

### 3.3 forward BOST

```python
def project_to_displacement(n_field, grad_n, views, config) -> dict:
    return {
        "disp_uv": disp_uv,
        "disp_clean": disp_clean,
        "noise": noise,
    }
```

验收：

- 同一 phantom 改 view count 后，输出 shape 正确。
- noise seed 固定时结果可复现。
- 能计算 leave-one-view-out reprojection。

### 3.4 baseline reconstruction

```python
def reconstruct_baseline(disp_uv, views, config) -> dict:
    return {
        "n_recon": n_recon,
        "history": history,
        "method": "tikhonov|landweber|sirt|fbp_proxy",
    }
```

验收：

- 至少一个传统 baseline。
- 保存迭代曲线或正则化参数。
- baseline 失败也要保存图，不能只展示 neural 成功图。

### 3.5 NeRIF-style neural field

```python
class CoordinateField(nn.Module):
    def forward(self, coords):
        return delta_n
```

建议从三档实现，不要一开始追论文级完整复现：

1. `plain MLP`：最小坐标网络，验证 pipeline、loss、metric。
2. `Fourier / positional encoding`：对齐 NeRIF 和 neural-field 基础文献，观察高频结构与噪声敏感性。
3. `hash-like multiresolution encoding`：只作为 NRIP-style 轻量对照，先用可控 toy 或 tiny-cuda-nn/torch-grid 思路评估速度、显存和高频重建，不承诺完整复现 NRIP。

训练输出：

```python
{
  "checkpoint": "...",
  "loss_history": "...",
  "n_recon": "...",
  "grad_est": "...",
  "config": "...",
}
```

验收：

- 先能拟合 ground-truth phantom，再接 displacement loss。
- 必须保存训练 loss 和重投影误差。
- 不把 neural loss 当唯一指标。
- 如果引入 NRIP-style mask、automatic/discrete gradient loss 或 hash encoding，必须作为消融变量单独开关，不能和真实数据、baseline、loss 权重同时改变。

### 3.6 metrics and report

```python
def evaluate(gt, recon, measurements, config) -> dict:
    return {
        "rel_l2": ...,
        "rmse": ...,
        "cc": ...,
        "reprojection_rmse": ...,
        "runtime_s": ...,
        "memory_mb": ...,
    }
```

每次实验必须自动保存：

- `metrics.csv`
- `config_resolved.yaml/json`
- `summary.png`
- `error_slice.png`
- `reprojection.png`
- `notes.md`

---

## 4. 指标优先级

### 必须有

| 指标 | 为什么 |
| --- | --- |
| relative L2 / RMSE | 基础误差，方便比较 |
| correlation coefficient | 避免只看幅值误差，看结构是否对 |
| reprojection error | 物理一致性，比普通网络 loss 更能说服老师 |
| runtime | 本科毕设要报告可用性 |
| view/noise sensitivity | 把“复现”变成“研究” |

### 有条件再加

| 指标 | 用在什么场景 |
| --- | --- |
| SSIM/PSNR | 切片图像比较 |
| leave-one-view-out error | 真实数据没有 ground truth 时 |
| temporal smoothness | 4D BOST |
| centroid trajectory error | 4D moving phantom |
| PIV RMSE / p95 error | PIV-BOST compensation |
| uncertainty map | 未来数据同化接口 |

---

## 5. 实验命名规范

推荐格式：

```text
YYYYMMDD_{modality}_{dataset}_{method}_{scan}
```

例子：

```text
20260710_bost_synthetic_m1_baseline_vs_coord_view_noise
20260715_bost_openbos_sirt_vs_nerif_9view
20260720_pivbost_synthetic_velocity_compensation_noise
20260725_4dbost_synthetic_lowrank_rank_scan
```

每个实验目录必须包含：

```text
config.yaml/json
metrics.csv
figures/
logs/
notes.md
```

`notes.md` 必须回答：

1. 这次实验想验证什么？
2. 哪个图最重要？
3. 失败或不确定的地方是什么？
4. 下次要改哪个参数？
5. 这个结果能不能写进开题或论文？

---

## 6. First 30 Days Coding Plan

| 天数 | 目标 | 交付 |
| --- | --- | --- |
| Day 1-3 | 建仓库、配置模板、phantom 生成 | `phantom_slice.png`, `metadata.json` |
| Day 4-6 | forward BOST 合成位移 | `displacement_views.png`, `disp_uv.npy` |
| Day 7-10 | 传统 baseline | `baseline_recon.png`, `metrics.csv` |
| Day 11-14 | 坐标 MLP 拟合 phantom | `mlp_fit.png`, `loss_curve.png` |
| Day 15-18 | NeRIF-style displacement loss | `nerif_recon.png`, `reprojection.png` |
| Day 19-22 | 视角数/噪声扫描 | `view_noise_heatmap.png` |
| Day 23-25 | 自动报告 | `report.html` 或 `summary.md` |
| Day 26-28 | 数据 manifest 接口 | 能读取 `data_templates/bost_sample_manifest.json` |
| Day 29-30 | 会前汇报整理 | 三张图 + 一页问题清单 |

30 天后，哪怕真实数据还没到，也应该有一个能汇报的闭环。

---

## 7. 代码质量底线

- 所有实验必须固定 random seed。
- 所有图必须在文件名或标题里写清 dataset/method/view/noise。
- 不覆盖旧结果；每次实验新建目录。
- 不手工改 CSV；指标由脚本生成。
- 不只保存 notebook；核心流程必须能用命令行跑。
- 不把真实 OERF 数据路径硬编码进脚本。
- 不把未获许可的真实图像直接放进公开材料。

---

## 8. 与何远哲沟通时的技术问题

1. 真实 BOST 数据里，组里标准的数组轴顺序是什么？
2. displacement field 的单位是 pixel、mm，还是归一化坐标？
3. 真实九视角几何可否先抽象成平行投影？
4. baseline 是用 SIRT、Landweber、UBOST，还是组内自定义方法？
5. NeRIF 输出是 `n`、`grad n`，还是同时输出？
6. 外部 NRIP 路线里的 hash encoding、3D mask 和 gradient loss，组内认为值得本科做轻量消融吗？
7. 论文/组内最看重哪些指标：reprojection error、L2、MRE、速度误差还是可视化？
8. 如果我做自动报告，哪些图对师兄最有用？

---

## 9. 与现有工作台的对应关系

| 本蓝图模块 | 已有材料 |
| --- | --- |
| 数据 manifest | `data_templates/`、`data_request_checklist.md` |
| 最小 demo | `demo_m0/` 到 `demo_m3b/` |
| 12 周学习 | `undergrad_foundation_bootcamp.md` |
| 第一周代码规格 | `starter_code_spec.md` |
| 论文映射 | `paper_to_demo_map.md` |
| 开题材料 | `opening_report_draft.md`、`opening_ppt_outline.md` |

---

## 10. 最现实的第一版题目产物

如果按照这个蓝图走，最稳的本科毕设产物是：

> 一个面向少视角 BOST 的可复现神经隐式折射率场重构与鲁棒性分析工具包。

它包含：

- 合成/开源/真实数据统一接口。
- 传统 baseline。
- 简化 NeRIF。
- 参数扫描。
- 物理一致性指标。
- 自动图表报告。
- PIV-BOST 或 4D BOST 的一个升级 toy。

这个产物比“我复现一篇论文”更稳，也更贴 OERF 长期需要。
