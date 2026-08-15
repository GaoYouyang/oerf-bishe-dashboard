# v153：坐标规范化没有修复跨轨迹支持，当前预测路线关闭

## 这次真正检验了什么

v152 已经说明：增加一条同功率、不同尺寸的公开训练轨迹能改善覆盖，但原 p33 条件在 5 相机下仍只有 `83.68%` 支持率。v153 没有训练更大的模型，而是直接检验一个更基础的解释：轨迹间的差异是否主要来自可由观测识别的中心、尺度或单调坐标变形。

结果前固定了两条完全不读目标的路径：

- 便宜仿射 control：根据 K1 residual 的多视角中心与尺度做坐标规范化；
- primary 单调输运：用 residual 幅值质量的固定逆累积分布映射，同时变换 observation、residual、dual 与对应几何 token。

两者都只读 deployment-visible observation、exact K1 residual、exact K1 detector dual、reported geometry 和 active-camera mask。六条公开 train 轨迹继续做 complete-trajectory leave-one-out；标准化与支持阈值只由 fold 内轨迹决定。共审计 `4,440` 个样本、`36,630` 个 active camera rows，validation/test 未打开，不读取 CFD truth 或 Krylov target，不拟合 predictor，也不做物理重建 replay。新增在线精确调用为 `0A+0A^T`。

## 独立确认的结果

原 p33-s01 的支持率如下：

| active cameras | v152 raw | v153 affine | v153 monotone | 90% 门 |
|---:|---:|---:|---:|---:|
| 5 | 83.68% | 65.73% | 71.14% | 未通过 |
| 7 | 91.81% | 79.77% | 82.93% | 未通过 |
| 9 | 97.24% | 88.41% | 89.07% | 未通过 |
| 12 | 97.79% | 93.47% | 94.77% | 通过 |

primary 不但没有把 5 相机推过门，还把它从 `83.68%` 降到 `71.14%`；原本已经通过的 7/9 相机分层也分别降到 `82.93% / 89.07%`。因此“保持所有已通过分层”这一冻结门同样失败。

新增 p33-s03 在单调输运下仍为 `98.27% / 96.99% / 97.48% / 99.05%`，四档都通过。但完整轨迹汇总暴露出更强的工况差异：p14、p22、p33-s01、p45、p58、p33-s03 分别为 `99.92% / 99.56% / 87.13% / 7.60% / 93.86% / 98.07%`。尤其 p45 的 `7.60%` 说明固定坐标变换无法把当前跨功率/尺寸状态放入同一个可靠支持域。

正式程序得到：

`FAIL_TARGET_FREE_MONOTONE_COORDINATE_SUPPORT_V153`

独立第二实现重新构造仿射与单调坐标、geometry update、fold-only normalization、最近邻支持和全部判决。`15/15` 项检查全部通过：浮点数组最大差 `3.29e-14`，汇总最大差 `4.00e-15`，全部整数/布尔数组与科学判决一致；正式结果树和输入保持不变。

## 这次失败怎样改变路线

这不是“模型还不够大”。当前更直接的证据是：六条训练轨迹的 deployment-visible 状态覆盖不足，而且一个固定的 target-free 仿射或单调坐标规范化既不能修复缺口，还会伤害原本通过的分层。

因此当前**坐标规范化 + 跨轨迹系数预测**路线关闭。不会用 CNN、FNO、UNO、DeepONet 或租 GPU 来补偿缺失的工况覆盖。下一步只允许先扩展公开 train 工况覆盖，并在 validation/test 继续封存的条件下重新审计 deployment-visible 跨轨迹支持；支持门通过前不训练 predictor 或神经算子。

当前边界：

- `predictor_training_authorized=false`；
- `physical_replay=false`；
- `gpu_rental_authorized=false`；
- `algorithm_breakthrough=false`；
- `resource_speedup=false`；
- `external_generalization=false`；
- `curved_ray_validated=false`；
- `real_bost=false`；
- `paper_success=false`。

公开证据：

- `docs/poolfire_k1_coordinate_canonicalization_v153_public_summary.json`
- `assets/figures/poolfire_k1_coordinate_canonicalization_v153.png`

## English checkpoint

v153 tests whether cross-trajectory support is limited mainly by observation-visible center, scale, or monotone coordinate changes. It freezes two target-free mechanisms before results: a cheap affine centroid/scale control and one fixed monotone inverse-CDF warp derived from K1-residual magnitude. The same map is applied consistently to observations, residuals, detector duals, and reported geometry tokens.

The audit uses six public PoolFire training trajectories under complete-trajectory leave-one-out, covering `4,440` samples and `36,630` active camera rows. It reads no CFD truth or Krylov target, fits no predictor, opens no validation/test truth, performs no reconstruction replay, and adds `0A+0A^T` exact calls.

For p33-s01 under `5/7/9/12` cameras, the v152 raw support is `83.68% / 91.81% / 97.24% / 97.79%`. The affine control falls to `65.73% / 79.77% / 88.41% / 93.47%`; the monotone primary reaches only `71.14% / 82.93% / 89.07% / 94.77%`. The primary therefore fails to rescue five cameras and harms the previously passing seven- and nine-camera strata. Aggregate monotone support for p45-size05 is only `7.60%`.

An independent second implementation rebuilds both coordinate maps, geometry updates, fold-only normalization, nearest-neighbor support, and all decisions. All `15/15` checks pass; the largest floating-array and summary differences are `3.29e-14` and `4.00e-15`, with exact agreement in discrete decisions.

The scientific decision is `FAIL_TARGET_FREE_MONOTONE_COORDINATE_SUPPORT_V153`. This closes the current coordinate-canonicalization plus cross-trajectory coefficient-prediction route. It is not a reconstruction, learned-model, exact-call-saving, wall/RSS, external-generalization, curved-ray, real-BOST, or paper-success result. Larger neural models and GPU rental are not authorized. The next gate expands public training-condition coverage while keeping validation and test sealed.
