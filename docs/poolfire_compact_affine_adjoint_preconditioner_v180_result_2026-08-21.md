# v180：精确逆可观测，但共享紧凑线性近似没有通过

更新：2026-08-21

## 结论

v179 已经证明，在当前已开封的 PoolFire 代理上，冻结五相机观测和报告几何可以通过逐几何精确逆确定全部 `1,009` 个仿射坐标。v180 检验下一步最小假设：能否把这些不同几何下的精确逆压缩为一个共享的、只读部署可见输入的线性预条件器。

主候选先用一次精确伴随把观测残差提升到 `1,009` 维坐标特征，再应用共享 diagonal + rank-16 残差映射；它有 `34,322` 个拟合系数，不读取留出三维真值。K0 的逻辑在线账是 `0A+1A^T`，接一轮完全未修改的 CGLS K1 后是 `2A+2A^T`。

在正式运行、独立第二实现和一次只修正近零均值比较规则的封存审计完成后，科学判决为：

`FAIL_SHARED_COMPACT_ADJOINT_PRECONDITIONER_V180`

这不是“观测里没有信息”。相反，v179 已排除了这个解释。v180 说明的是：**一个固定共享的 diagonal + rank-16 线性映射，仍不足以表达不同相机几何对应的逆结构。**

## 主候选结果

五相机 K1 的 field / gradient / observation p90 为：

`0.344248 / 0.485871 / 0.311000`

对应 worst 为：

`0.370752 / 0.527929 / 0.354429`

全九相机 K1 的 p90 为：

`0.332602 / 0.466727 / 0.363093`

对应 worst 为：

`0.376542 / 0.509568 / 0.405783`

冻结绝对门要求 observation p90 不超过 `0.20`、worst 不超过 `0.35`。因此两档相机的 field 与 gradient 全局尾部都在门内，但 observation 持续越线。最终严格安全单元只有五相机 `4/52`、全九相机 `7/52`；两者都只有 `1/13` 套完整标定通过，四个时刻分层均为 `0/4`。

K0 更弱：五相机与九相机都是 `0/52`。更便宜的共享 diagonal、一次坐标 CGLS 和静态训练均值也全部失败。这个结果不能包装成调用节省，因为完整 matched-accuracy 门没有通过。

## 独立复算与近零均值审计

原独立 validator 的 `42` 项检查中有 `41` 项通过，唯一失败是把数值上接近零的 feature mean 与 target mean 用相对误差比较。两组均值的独立绝对差分别只有 `7.85e-16` 与 `3.17e-16`，但分母同样接近零，使相对量失去意义。原始 `INCONCLUSIVE` 记录被原样保留，没有删除或覆盖。

结果后另行冻结的 v180.1 审计不重跑拟合、预测、物理重放或评分，只对封存数组使用预注册的近零绝对误差门；非零数组仍沿用原相对误差门。`24/24` 项审计全部通过：

- 候选场最大相对差 `8.40e-12`；
- deployment feature 最大相对差 `1.45e-11`；
- 指标最大绝对差 `7.85e-12`；
- 各臂汇总最大绝对差 `7.21e-12`；
- 所有连续数值门和离散通过/失败判决一致；
- formal 与原 validator 都没有重跑。

因此修正的是验证尺度，而不是科学门或候选结果。

## 科学判断改变在哪里

v178 说明训练场仿射空间有容量；v179 说明五相机观测确实包含确定全部坐标的信息；v180 进一步排除了一个最小、便宜而直接的压缩方式。当前瓶颈被收窄为**几何依赖的逆结构**：它能被逐几何精确求解，但不能由一个固定的低秩共享线性映射稳定近似。

因此关闭当前 shared linear adjoint-preconditioner 家族，不追加 rank、不改 ridge、不放宽 observation 门，也不用更大 CNN、FNO、UNO、DeepONet 或 GPU 挽救。后续只有两种合理入口：结果前冻结一个物理上真正不同、显式几何条件化的因子分解；或获得工况匹配的实验二维 BOS 位移与完整映射后，从新物理信息重新判断。

v180 不是部署算法、exact-call 减少、wall/RSS 加速、外部泛化、curved ray、真实 BOST、论文成功或算法突破：

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`broad_external_generalization=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v180: the exact inverse is observable, but the shared compact linear approximation fails

Updated: 2026-08-21

v179 establishes that, on the opened PoolFire proxy, the frozen five-camera observation and reported geometry identify all `1,009` affine coordinates through a geometry-specific exact inverse. v180 tests the next minimal hypothesis: can those geometry-specific inverses be compressed into one shared linear preconditioner that reads only deployment-visible inputs?

The primary applies one exact adjoint to the observation residual, producing a `1,009`-coordinate feature, then uses a shared diagonal plus rank-16 residual map. It has `34,322` fitted coefficients and never reads held-out 3D truth. Its logical online ledger is `0A+1A^T` at K0 and `2A+2A^T` after one unchanged CGLS K1 step.

After formal execution, an independent second implementation, and a sealed audit that only corrects the comparison of numerically zero means, the decision is `FAIL_SHARED_COMPACT_ADJOINT_PRECONDITIONER_V180`.

For five-camera K1, field / gradient / observation p90 values are `0.344248 / 0.485871 / 0.311000`, with worst values `0.370752 / 0.527929 / 0.354429`. For all-nine K1, p90 values are `0.332602 / 0.466727 / 0.363093`, with worst values `0.376542 / 0.509568 / 0.405783`.

The frozen observation limits are `0.20` at p90 and `0.35` at worst. Field and gradient global tails stay within their limits under both sensor arms, but observation error does not. Only `4/52` five-camera cells and `7/52` all-nine cells are jointly strict-safe; each sensor arm passes only `1/13` complete calibrations and `0/4` time strata. K0 is `0/52` under both sensors. The shared-diagonal, one-step coordinate-CGLS, and static-mean controls also fail.

The original validator passes `41/42` checks. Its only false check applies a relative comparison to feature and target means whose norms are themselves near numerical zero. Their maximum independent absolute differences are only `7.85e-16` and `3.17e-16`. The original inconclusive record remains preserved.

The separately frozen v180.1 audit does not rerun fitting, prediction, physical replay, or scoring. It applies preregistered absolute checks only to the near-zero means while retaining the original relative checks for every nonzero array. All `24/24` checks pass. Maximum candidate-field relative, deployment-feature relative, metric absolute, and arm-summary absolute differences are `8.40e-12`, `1.45e-11`, `7.85e-12`, and `7.21e-12`. Every discrete decision agrees, and neither the formal run nor the original validator is rerun.

The scientific diagnosis is now narrower. v178 establishes affine-span capacity, v179 establishes measurement observability, and v180 rejects one simple compact approximation. The remaining bottleneck is the geometry-dependent inverse structure: it is recoverable with a geometry-specific exact inverse but not with this fixed shared low-rank linear map.

This is not an impossibility result for every compact mechanism.

Close the current shared linear adjoint-preconditioner family without increasing rank, retuning ridge, relaxing the observation gate, or using a larger CNN, FNO, UNO, DeepONet, or GPU as rescue. A next attempt requires either a preregistered physically distinct geometry-conditioned factorization or new condition-matched experimental 2D BOS displacement and mapping information.

v180 is not a deployed algorithm, exact-call reduction, wall/RSS speedup, external generalization, curved-ray validation, real BOST, paper success, or an algorithmic breakthrough: `algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `broad_external_generalization=false`, `curved_ray_validated=false`, `real_bost=false`.
