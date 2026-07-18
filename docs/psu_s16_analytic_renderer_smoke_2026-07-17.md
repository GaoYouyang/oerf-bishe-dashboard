# PSU-S16 独立解析 renderer：第一条不共享逆算子离散链的三维基线

日期：2026-07-17

证据等级：**E1 synthetic interface smoke**

判决：**接口通过；算法优越性未测试，不能写成实验重建或论文成功。**

## 1. 为什么要重做真值生成器

旧 synthetic 闭环用同一个体素场、有限差分和三线性插值生成观测，再让逆算法解回同一个网格。即使 QMC 数不同，这仍保留轻度 inverse crime：出题与答题共享关键离散原语，可能低估真实 forward mismatch。

本轮新链路明确拆开：

```text
连续解析形态 q(x)
  -> 手写解析梯度 grad q(x)
  -> QMC-64 有限孔径积分生成观测
  -> QMC-8 体素有限差分 + 三线性逆算子
  -> 固定 12 步 weighted CGLS
```

解析 renderer 源码不导入 `PSUB0VoxelGradientOperator` 或 `finite_difference_gradient`。这只移除一种数值自洽偏差，不能消除 synthetic-to-experiment gap。

## 2. 四类反应流形态代理

1. `smooth_plume`：平滑热产物标量亏损代理，无界面指标。
2. `wrinkled_density_interface`：`tanh(s/delta)` 褶皱密度界面，只称 flame-related density interface，不称化学反应面。
3. `oblique_compression_sheet`：受来流方向约束的曲面压缩层代理。
4. `shock_expansion_pair`：压缩/膨胀双层 morphology-OOD 代理，不把解析函数称为 Euler/NS 解。

每个 phantom 都返回 `q(x)`、解析 `grad q(x)`、适用时的 level set 与法向。外边界使用二次平滑窗使标量和梯度同时归零。标量合同是 `normalized_scalar_perturbation_not_physical_density`；没有 Gladstone-Dale 单位标定前不得写成定量密度。

## 3. 空间指标已经换掉什么

旧 `front_top10_f1` 让预测与真值各取最高 10% 梯度体素并要求精确重合。它对一体素平移过敏，也可能奖励面积相近但弥散的伪前沿；现在只保留为辅助诊断。

新主指标模块提供：

- field relative-L2、RMSE、dynamic-range NRMSE、mean bias；
- 相对解析真值梯度的 H1-seminorm error；
- 物理距离单位的 ASSD、HD95；
- `1*dx`、`2*dx` 容差下的双向 surface precision / recall / F1；
- signed 与 orientation-invariant 法向角 median / p95。

实验 PSU/OERF 数据没有三维真值时，field、H1 与 surface 指标全部禁止计算，只能报告 held-out reprojection 和物理一致性。

## 4. 冻结配置

| 项目 | 值 |
|---|---:|
| 几何 | 真实 PSU 9 个 support views |
| 每视角光线 | 64 |
| 逆网格 | 16 cubed |
| 真值 renderer | 解析梯度 + QMC-64 |
| renderer 收敛审计 | QMC-32 vs QMC-64 |
| 逆算子 | 体素梯度 + QMC-8 |
| 噪声 | 每视角 clean RMS 的 1% 独立高斯噪声 |
| 求解器 | weighted CGLS-12 |
| 调用预算 | 12 F + 13 A-transpose；评估 forward 另记 |

配置：[psu_s16_analytic_renderer_smoke_v1.json](../demo_t16_operator/configs/psu_s16_analytic_renderer_smoke_v1.json)

## 5. 实测结果

| 量 | 结果 |
|---|---:|
| QMC-32/64 最大相对观测差 | 0.001185 |
| 四场平均 field relative-L2 | 0.708671 |
| 四场平均 H1-seminorm error | 1.428984 |
| support reprojection relative-L2 范围 | 0.0620 - 0.0697 |
| 褶皱界面 surface-F1 at 1 dx / 2 dx | 0.5435 / 0.6706 |
| 压缩层 surface-F1 at 1 dx / 2 dx | 0.5272 / 0.6513 |
| checksum 文件 | 5 / 5 通过 |
| 相关核心测试 | 55 / 55 通过 |

最重要的解释不是“CGLS 很差”，而是：**解析积分已较稳定，低重投影残差仍没有给出准确三维场和法向。** 新算法必须同时改善 field/H1/front，并守住重投影、调用预算和 OOD harm，不能只把 support residual 做得更漂亮。

## 6. 非绑定 headroom 探针

在同一个已经打开的四场和随后 48 个开发 phantom 上做了不进入正式结论的快速扫描：

- Sobolev strength 2-3 把四场平均 field-L2 从 CGLS 的 0.7087 降到约 0.39，但重投影从 0.0653 变差到 0.159-0.227；
- 48 场标量强度 catalog 的逐样本 truth oracle 仅比最佳固定强度多 1.711% joint-loss headroom；
- 加入各向异性轴权重后 oracle headroom 也只有 2.690%。

因此“用小网络预测一个 Sobolev 强度/轴权重”暂不够格作为论文主创新。它可作为可解释 baseline；主算法应正面处理 data consistency 与不可观测空间先验之间的矛盾，并避免重复仓库里已经 NO-GO 的 free/nullspace corrector、旧 positive spectral direction 和简单 residual router。

## 7. 下一算法的最低合同

候选暂不命名为成功模型。进入训练前必须满足：

1. 真值继续由独立解析 renderer 生成，逆算法不得调用其 family、seed、level set 或真值梯度。
2. fallback 固定为 validation-selected Sobolev / PCGLS，而不是弱 Landweber。
3. learned branch 只能读取 `y`、`sigma`、view mask、几何摘要和共享的初始 normal residual。
4. 同时优化 field、H1、surface-F1/ASSD 与 held-out reprojection；单指标胜出无效。
5. 完整 geometry/phantom/noise 父簇是统计单位，光线、体素和 frame 不能伪装成独立样本。
6. support gate 的方法选择、split-conformal 阈值、风险校准和 fresh 必须分开。
7. 常数 feature、单位、grid、renderer、sampling manifest 不匹配时 exact fallback。
8. 先在 16 cubed development 证伪；通过后才运行 32 cubed 和神经对手，不因 Mac 空闲就盲目扩大。

## 8. 本机复现

```bash
cd /path/to/oerf-bishe-dashboard

PYTHONPATH=. .venv/bin/python site_tools/run_psu_s16_analytic_renderer_smoke.py \
  --config demo_t16_operator/configs/psu_s16_analytic_renderer_smoke_v1.json \
  --view-root private_library/external_datasets/psu_bost_flight_body/all_view_geometry_audit \
  --output-dir demo_t16_operator/results/psu_s16_analytic_renderer_smoke_public

PYTHONPATH=. .venv/bin/python site_tools/validate_psu_s16_analytic_renderer_smoke.py \
  --config demo_t16_operator/configs/psu_s16_analytic_renderer_smoke_v1.json \
  --output-dir demo_t16_operator/results/psu_s16_analytic_renderer_smoke_public

(cd demo_t16_operator/results/psu_s16_analytic_renderer_smoke_public && \
  shasum -a 256 -c checksums.sha256)
```

公开结果包：[README](../demo_t16_operator/results/psu_s16_analytic_renderer_smoke_public/README.md) · [summary JSON](../demo_t16_operator/results/psu_s16_analytic_renderer_smoke_public/summary.json) · [metric CSV](../demo_t16_operator/results/psu_s16_analytic_renderer_smoke_public/metric_rows.csv) · [diagnostic PDF](../demo_t16_operator/results/psu_s16_analytic_renderer_smoke_public/diagnostic.pdf)
