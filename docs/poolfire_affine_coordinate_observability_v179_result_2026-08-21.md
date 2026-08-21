# v179：五相机观测能确定近满秩仿射坐标，但精确逆仍不是部署算法

更新：2026-08-21

## 结论

v178 已证明 `1,010` 个已开封 PoolFire 训练场形成的稳定秩 `1,009` 仿射空间具有真值可见容量，但没有回答一个更关键的问题：这 `1,009` 个坐标是否能从部署时真正可见的二维观测和报告相机几何中确定。

v179 冻结同一个仿射空间、同一个已开封四帧评估、`13` 套标定、v176 的五相机选择和九相机对照。主方案不读取目标三维真值，只把测量矩阵 `A U^T` 做一次精确缓存 SVD，并由当前观测残差求最小范数坐标。便宜对照分别是静态训练场均值，以及只做一次坐标空间 CGLS / 最速下降。

完全独立第二实现复算后，正式科学判决为：

`PASS_AFFINE_MEASUREMENT_INVERSE_HEADROOM_V179`

五相机主方案在 K0 与未修改 CGLS K1 后都通过 `52/52` 单元、`13/13` 标定和 `4/4` 帧；九相机两个主臂也都通过 `52/52`。测量秩在全部设置中都是 `1,009/1,009`。这说明 v178 的近满秩表示并没有被五相机观测隐藏：在当前已开封代理上，观测和几何确实足以确定其全部坐标。

## 五相机结果

精确测量伪逆 K0 的 field / gradient / observation p90 为：

`0.253087 / 0.406712 / 0.098374`

对应 worst 为：

`0.258014 / 0.428365 / 0.104380`

再接一轮未修改 CGLS K1 后，p90 为：

`0.250113 / 0.396906 / 0.067119`

对应 worst 为：

`0.255267 / 0.420202 / 0.072764`

K0 与 K1 的逻辑物理重放账分别为 `1A+0A^T` 和 `2A+1A^T`。但这不能单独写成 exact-call 节省，因为测量缓存并不便宜。

## 便宜对照为什么重要

一次坐标迭代并不能解释主结果。它在五相机 K0 下的 p90 是 `0.556114 / 0.769582 / 0.465577`，K1 下是 `0.533539 / 0.703929 / 0.290899`，两者都只有 `0/52` 严格安全单元。

静态均值 K0 和 K1 同样为 `0/52`。因此，正结果不是“均值场已经够好”，也不是“一次廉价坐标更新就足够”；完整测量逆所利用的信息确实更丰富。

## 独立复算

独立实现不导入正式 v179 runner，采用独立的场变换、射线与稀疏算子、SVD 路径和指标重建。`36/36` 项检查全部通过：

- 候选场最大相对差 `8.40e-12`；
- 坐标最大相对差 `1.35e-11`；
- 指标最大绝对差 `4.48e-12`；
- 最大 stationarity `5.92e-15`；
- 测量缓存与直接 forward 最大相对误差 `3.10e-16`；
- 相机乱序后的坐标最大相对差 `1.22e-14`；
- 固定观测后突变留出真值，坐标和候选变化为 `0`；
- 全部连续数值门、离散通过/失败和最终判决一致。

## 最重要的成本边界

这个结果不是低成本 warm initializer。每个传感器与标定都要把均值和 `1,009` 个仿射基方向投影到测量空间；冻结审计账合计 `26,260` 次 forward-equivalent 几何缓存投影。即使这些缓存理论上可离线复用，它仍然只是一个高成本解析可观测性见证，没有 fresh wall、峰值 RSS 或整体部署资源证据。

所以 v179 改变的判断是：当前瓶颈不再是“五相机观测缺少足够信息”，而是“怎样把一个 1,009 维精确逆压缩成共享、稳定、低成本的 observation + geometry-only 近似”。下一门只能另行结果前冻结一个紧凑 CPU 近似诊断，并保留完整轨迹隔离与相同便宜对照。

v179 不是学习算法、神经算子、部署方案、exact-call 减少、资源加速、外部泛化、curved ray、真实 BOST、论文成功或算法突破：

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`broad_external_generalization=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v179: five-camera observations identify the near-full-rank affine coordinates, but the exact inverse is not a deployed algorithm

Updated: 2026-08-21

v178 establishes truth-aware capacity in a stable-rank-`1,009` affine span built from `1,010` opened PoolFire training fields. It does not establish whether those `1,009` coordinates can be identified from deployment-visible 2D observations and reported camera geometry.

v179 freezes the same affine span, four opened evaluation frames, thirteen calibrations, the v176 five-camera selections, and an all-nine control. The primary reads no target 3D truth. It performs a cached SVD of the measurement matrix `A U^T` and computes the minimum-norm affine coordinates from the current observation residual. Cheap controls are the static fit mean and one coordinate-space CGLS / steepest-descent step.

After fully independent recomputation, the decision is `PASS_AFFINE_MEASUREMENT_INVERSE_HEADROOM_V179`.

Under five cameras, both exact-measurement-pseudoinverse K0 and unchanged CGLS K1 pass `52/52` cells, `13/13` calibrations, and `4/4` frames. Both primary arms also pass `52/52` under all nine cameras. Measurement rank is `1,009/1,009` for every setup. The near-full-rank v178 representation is therefore observable from the frozen five-camera measurement and reported geometry on this opened proxy.

For five-camera K0, field / gradient / observation p90 values are `0.253087 / 0.406712 / 0.098374`, with worst values `0.258014 / 0.428365 / 0.104380`. After one unchanged CGLS K1 step, p90 values are `0.250113 / 0.396906 / 0.067119`, with worst values `0.255267 / 0.420202 / 0.072764`.

The one-step coordinate controls do not explain the result. Their five-camera K0 and K1 p90 values are `0.556114 / 0.769582 / 0.465577` and `0.533539 / 0.703929 / 0.290899`; both have `0/52` strict-safe cells. Static-mean K0 and K1 also remain `0/52`.

A fully independent implementation uses separate field transforms, ray and sparse-operator construction, SVD, and metric reconstruction. All `36/36` checks pass. Maximum candidate-field relative, coordinate relative, and metric absolute differences are `8.40e-12`, `1.35e-11`, and `4.48e-12`. Maximum stationarity is `5.92e-15`; camera permutation changes coordinates by at most `1.22e-14`; fixed-observation held-out-truth mutation changes the coordinates and candidates by exactly zero. Every continuous gate and discrete decision agrees.

The decisive limitation is cost. Building the measurement caches requires the fit mean plus `1,009` affine-basis projections for every sensor and calibration, totaling `26,260` forward-equivalent setup projections. This exact inverse is a high-cost observability witness, not a low-cost warm initializer. There is no fresh wall, peak-RSS, or whole-pipeline resource evidence.

v179 changes the diagnosis: the immediate bottleneck is no longer missing five-camera information, but compactly approximating a 1,009-dimensional exact inverse using observation and geometry only. The next gate must be a separately preregistered compact CPU approximation with complete-trajectory isolation and the same cheap controls.

v179 is not a learned algorithm, neural operator, deployment method, exact-call reduction, resource speedup, external generalization, curved-ray validation, real BOST, paper success, or an algorithmic breakthrough: `algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `broad_external_generalization=false`, `curved_ray_validated=false`, `real_bost=false`.
