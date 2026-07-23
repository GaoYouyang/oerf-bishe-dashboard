# DG-CovGate：BOST flow-off 重复帧采集与 detector covariance 闸门

日期：2026-07-17

## 结论先行

本轮没有训练新的三维重建网络，而是先解决一个更靠近真实实验的前置问题：

> 要估计每台 BOST 相机的空间相关、方向耦合、异方差和低频漂移，至少需要多少张固定实验条件下的 flow-off 重复帧？

在 PSU 公开九相机真实 detector 坐标上，使用三类合成 covariance truth、8 个随机种子、9 台相机和 160 张封闭测试帧进行采集规划，得到：

- `R=4/8/12/20/32` 均未通过全部预设闸门；
- `R=50` 首次通过；
- `R=32` 的最坏 truth-family coverage p90 误差为 **12.44 个百分点**，高于 8 个百分点门槛；
- `R=50` 降至 **5.625 个百分点**；
- graph-heat truth 在 `R=50` 的 held-out NLL 改善中位数为 **0.03448 nat / dimension**，p10 为 **0.03291**；
- IID truth 的 graph false activation 为 **0%**，p90 harm 为 **0**；
- nonstationary-drift truth 在 `R=50` 的 rank-1 drift 启用率为 **90.28%**，而 IID truth 为 **0%**。

因此，给师兄的当前最小请求应从模糊的“20-50 帧”收紧为：

```text
每台相机、每个固定光学/背景/工况条件至少 50 张 flow-off repeats；
其中 25% 在任何模型拟合和超参数选择之外永久留作 validation。
```

这只是 **真实 detector geometry + 合成 covariance family** 的采集规划。它不证明真实 OERF 数据一定 50 张足够，也不授权三维重建、算法优越性或实验 covariance 已估计。

## 1. 为什么先做这一步

VD0-B 已证明真实 detector 邻域含有局部压力信号，但 validation/calibration 的 field harm 和 front safety 失败。随后真实 PSU 位移域审计发现：

- 只有 1 个真实物理场；
- 130 个相机子集全部越出当前 synthetic 95% feature envelope；
- informative feature 平均越界 23.994%。

这意味着继续增加 DeepSets、FNO 或 attention 容量，会更高效地拟合一个尚未覆盖真实 detector statistics 的生成器。应先建立可审计的 measurement data model。

## 2. 公开 PSU 数据为什么不能直接估 temporal covariance

PSU 论文说明每次试验采集了 2000 张 flow-off 和 2000 张 flow-on 图像；但公开归档索引暴露的是平均/复合产物，而不是固定条件下的独立时间帧：

- 12 个 ZIP、1838 个文件条目；
- 70 个 flow-on deflected TIFF，恰好每个 camera-rotation condition 1 张；
- 每个 camera-rotation condition 的独立 temporal flow-off/frame count 为 0；
- `withoutCylinder` calibration TIFF 对应不同标定靶角度，不是同一条件的时间重复；
- 因而公开包不授权 temporal covariance estimation。

可复核清单：

- `docs/psu_flowoff_repeat_inventory_public_summary.json`
- `site_tools/audit_psu_flowoff_repeat_inventory.py`
- `data_templates/open_bos_zip_file_content.txt`

## 3. 模型

每台相机的位移残差写成 `x_t in R^(N x 2)`，`N=256` 个真实 detector rays，两个分量为 `u/v`。

从真实 detector pixel coordinates 构造 8-NN 图及对称归一化 Laplacian `L`。基础 graph covariance 为：

```text
K_graph = (1 - alpha) I + alpha exp(-tau L)
Sigma_graph = sigma^2 K_graph tensor B_uv
```

其中 `B_uv` 表示两个位移分量的方差比和相关系数。

异方差扩展为：

```text
Sigma_amp = A Sigma_graph A
```

`A` 是由 node energy 估计并在 graph spectral basis 中平滑的正对角幅值。

低频同步漂移在基础模型白化坐标中写成 rank-1 更新：

```text
Sigma_drift = Sigma_amp^(1/2) (I + beta v v^T) Sigma_amp^(1/2)
```

评价使用 Woodbury 二次型和 matrix-determinant lemma，不构造 `512 x 512` 稠密逆矩阵。

## 4. 为什么不能直接拟合稠密经验 covariance

中心化经验 covariance 的秩最多为 `R-1`。每台相机的 measurement dimension 为 512：

| repeats | 最大经验秩 | 占 512 维比例 |
|---:|---:|---:|
| 4 | 3 | 0.59% |
| 8 | 7 | 1.37% |
| 12 | 11 | 2.15% |
| 20 | 19 | 3.71% |
| 32 | 31 | 6.05% |
| 50 | 49 | 9.57% |

即使 `R=50`，稠密经验 covariance 仍严重秩亏。因此本轮只授权低参数结构模型，并要求独立留出帧决定是否启用 graph/amplitude/drift。

## 5. 两级拟合与拒答

每个 calibration block 按 75%/25% 分为 fit/validation：

1. fit block 拟合 component-IID、graph、graph+amplitude 和 graph+amplitude+rank-1 drift；
2. validation block 比较 held-out NLL；
3. 候选相对 component-IID 的每维 NLL 改善必须至少为 0.0025；
4. 未达到门槛时回退到 component-IID；
5. 选型结束后才用全部 calibration repeats 重拟合被选模型；
6. 最终指标只读独立的 160 张 sealed synthetic test repeats。

这种 gate 的目的不是保证真实安全，而是避免小样本下无条件启用复杂 covariance。

## 6. 预设采集闸门

一个 repeat count 必须同时满足：

- graph-heat NLL gain median >= 0.005；
- graph-heat NLL gain p10 >= 0；
- IID p90 NLL harm <= 0.003；
- graph-heat activation >= 75%；
- IID false activation <= 25%；
- 三类 truth 的最大 p90 95%-coverage error <= 8 个百分点。

只有 coverage 门区分了 `R=32` 和 `R=50`。这说明仅看 likelihood gain 会过早宣布成功；不确定度校准是采集量判决的必要部分。

## 7. 结果解释

### 7.1 可保留的正结果

- graph structure 在 in-class graph truth 上稳定提高 held-out likelihood；
- validation gate 基本不在 IID truth 上误启用 graph；
- rank-1 drift 对 nonstationary truth 的作用随 repeats 增多而变得可识别；
- `R=50` 在本轮合成压力族上同时通过 likelihood、harm、activation 和 coverage。

### 7.2 不能说的话

不能说：

- 真实 OERF flow-off covariance 已经估计；
- 50 张对所有实验都足够；
- DG-CovGate 已改善三维折射率场；
- 已击败 DeepONet、FNO、NeRIF、TDBOST 或组内方法；
- graph heat、amplitude 或 low-rank covariance 本身构成新颖算法。

Graph heat/Matérn covariance 和低秩漂移已有成熟谱系。可能保留的研究贡献只能来自：

1. BOST detector geometry 与 optical-flow residual 的专门参数化；
2. flow-off acquisition gate 与 sealed validation 协议；
3. covariance 对 whitening、PCGLS、front reconstruction 和 held-out camera 的闭环影响；
4. 真实组内 session 的复现与失败边界。

## 8. 给何远哲师兄的最小数据问题

建议直接问：

1. 是否能给每台相机同一固定条件下至少 50 张未经平均的 flow-off/reference frames？
2. 这些帧是否保留时间顺序、曝光、f-number、focus、背景距离和同步 timestamp？
3. 是否能同时给 raw image、位移 `u/v`、confidence、ROI/mask 和 bad-pixel map？
4. 是否有至少一台不参与 covariance 拟合或重建的 held-out camera？
5. reference 是否一直固定；是否有不同 background realization 或重复 reference？
6. flow-on 是否也有未经平均的短序列，用来检查 covariance 是否随流动/亮度变化？

若只能提供 20-32 张，可以做 exploratory fit，但应保留“coverage gate 未通过”的状态，不把它写成 calibrated uncertainty。

## 9. 下一轮实验

拿到组内 repeats 后按顺序执行：

1. 固定预处理，不在 flow-on/reconstruction truth 上选参数；
2. 以时间 block 而不是随机 frame 打乱做留出，避免慢漂移泄漏；
3. 检查 registration、background-dependent bias、坏点、空间相关和 `u-v` 耦合；
4. 与 component-IID、diagonal shrinkage、stationary graph、amplitude graph、low-rank drift 同时比较；
5. 先过 held-out likelihood、coverage、residual autocorrelation 和 camera holdout；
6. 再把 whitening 放入同调用 PCGLS/TV/NeRIF-compatible inverse；
7. 最后才检查 field/front/reprojection，不回头调 covariance。

## 10. 可复现入口

```bash
PYTHONPATH=. .venv/bin/python \
  site_tools/run_psu_b0_detector_graph_covariance_gate.py \
  --view-root "$PSU_VIEW_ROOT"

PYTHONPATH=. .venv/bin/python \
  site_tools/plot_psu_b0_detector_graph_covariance_gate.py \
  --report demo_t16_operator/results/psu_b0_detector_graph_covariance_gate/report.json \
  --output demo_t16_operator/results/psu_b0_detector_graph_covariance_gate/psu_b0_detector_graph_covariance_gate_figure.png
```

固定结果：

```text
trial rows: 7776
wall time: about 42 s on this Mac; runtime metadata varies
peak RSS: 314.8 MB
minimum synthetic-gate repeat count: 50
real temporal repeats used: 0
3D reconstruction performed: false
algorithm superiority claimed: false
```

产物：

- `demo_t16_operator/detector_graph_covariance.py`
- `demo_t16_operator/configs/psu_b0_detector_graph_covariance_gate_v1.json`
- `demo_t16_operator/results/psu_b0_detector_graph_covariance_gate/report.json`
- `demo_t16_operator/results/psu_b0_detector_graph_covariance_gate/trial_rows.csv`
- `demo_t16_operator/results/psu_b0_detector_graph_covariance_gate/summary_rows.csv`
- `demo_t16_operator/results/psu_b0_detector_graph_covariance_gate/psu_b0_detector_graph_covariance_gate_figure.png`

两次完整确定性复跑的 `trial_rows.csv` 与 `summary_rows.csv` SHA-256
逐字一致。`report.json` 只因实际 wall time 元数据不同而改变 hash，采集判决和
科学指标一致。

## 11. 相关主文献

- PSU open-source BOST dataset: `https://arxiv.org/html/2508.17120v2`
- Raffel et al., BOS techniques review: `https://link.springer.com/article/10.1007/s00348-015-1927-5`
- Dynamic backgrounds for BOS systematic error: `https://link.springer.com/article/10.1007/s00348-021-03285-6`
- Optical-flow BOS assessment: `https://link.springer.com/article/10.1007/s00348-022-03553-z`
- BOS density uncertainty quantification: `https://www.osti.gov/pages/biblio/1598804`
- Matérn Gaussian processes on graphs: `https://proceedings.mlr.press/v130/borovitskiy21a.html`
- Non-separable spatio-temporal graph kernels: `https://proceedings.mlr.press/v151/nikitin22a.html`
