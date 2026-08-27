# v273：直接三维 Hodge 提升等价于泊松预条件伴随，不是新物理方向

## 做了什么

v273 在读取任何 Case 19 密度真值、观测、残差、重建场或评分数组之前，先审计冻结 straight-ray 离散算子的代数结构。当前算子可精确分解为 `A = M D`：`D` 是固定零边界有限差分梯度，`M` 是三线性射线路径积分与相机平面投影。因此，把观测残差先提升到三维梯度场、再做 curl-free Hodge/Poisson 恢复，得到

`(D^T D)^-1 D^T M^T r = (D^T D)^-1 A^T r`。

也就是说，这条看似“梯度优先”的三维路线，实际上只是既有标量伴随的固定 Dirichlet-Poisson 预条件写法。审计覆盖 13 套 reported geometry，新增科学调用为 `0A+0A^T`，可训练参数为 0。

## 独立复算与判决

正式实现用稀疏 Kronecker 差分、显式三线性射线矩阵和直接 Poisson 求解；完全独立的第二实现用逐项循环装配差分、独立 COO 射线矩阵和迭代 Poisson 求解。两条实现的 16/16 有效性检查均通过。

正式实现的最大 `A = M D` 相对误差为 `3.55e-16`，Hodge 提升等价误差为 `1.33e-15`。独立实现的对应误差为 `3.51e-16` 和 `1.59e-12`，Poisson 残差为 `9.17e-13`；相机乱序误差为 `0`。正式与独立公共指标的最大绝对差为 `1.75e-12`，低于冻结的 `1e-9` 一致性界。

权威判决为 `FAIL_CASE19_DIRECT_VOLUME_HODGE_IS_POISSON_REPARAMETERIZATION_V273`：直接线性三维 Hodge 提升没有产生物理上不同的新方向，因此这条机制关闭，不再调 Hodge 边界、差分模板、projector 或 Laplacian。

## 证据边界

这是一次避免无效大实验的代数 no-go 结果，不是重建性能结果。它没有读取 Case 19 真值或评分，没有生成 warm initializer，也没有证明 matched-accuracy、有效减调用、wall/RSS、外部泛化、curved ray 或真实 BOST。它只关闭当前 frozen linear straight-ray 离散下的直接线性 volume-Hodge lift，不关闭非线性物理、非二次数据先验、噪声感知方法、curved rays 或整条 C 路线。`algorithm_breakthrough=false`。

# v273: Direct volumetric Hodge lifting is a Poisson reparameterization, not a new physical direction

## What was tested

Before reading any Case 19 density truth, observation, residual, reconstructed field, or score array, v273 audits the algebraic structure of the frozen straight-ray discretization. The operator factorizes exactly as `A = M D`, where `D` is the fixed zero-boundary finite-difference gradient and `M` performs trilinear path integration followed by camera-plane projection. Lifting an observation residual into a volumetric gradient field and then applying a curl-free Hodge/Poisson recovery therefore gives

`(D^T D)^-1 D^T M^T r = (D^T D)^-1 A^T r`.

The apparently gradient-first route is thus the existing scalar adjoint under a fixed Dirichlet-Poisson preconditioner. The audit covers thirteen reported geometries, adds `0A+0AT` scientific calls, and has zero trainable parameters.

## Independent recomputation and verdict

The formal implementation uses sparse Kronecker differences, an explicit trilinear ray matrix, and a direct Poisson solve. A separate implementation assembles differences in loops, constructs its own COO ray matrix, and uses an iterative Poisson solve. Both paths pass all 16/16 validity checks.

The formal maximum relative errors are `3.55e-16` for `A = M D` and `1.33e-15` for the Hodge-lift equivalence. Independent values are `3.51e-16` and `1.59e-12`, with a `9.17e-13` Poisson residual and zero camera-permutation error. The maximum absolute difference across common formal and independent metrics is `1.75e-12`, below the frozen `1e-9` agreement limit.

The authoritative verdict is `FAIL_CASE19_DIRECT_VOLUME_HODGE_IS_POISSON_REPARAMETERIZATION_V273`. Direct linear volumetric Hodge lifting creates no physically distinct direction, so this mechanism closes without tuning its boundary, stencil, projector, or Laplacian.

## Evidence boundary

This is an algebraic no-go result that avoids an unnecessary large experiment, not a reconstruction-performance result. It reads no Case 19 truth or score, creates no warm initializer, and establishes no matched accuracy, effective call reduction, wall/RSS result, external generalization, curved-ray validity, or real BOST result. It closes only the direct linear volume-Hodge lift under the frozen linear straight-ray discretization; nonlinear physics, nonquadratic data priors, noise-aware methods, curved rays, and the wider C route remain open. `algorithm_breakthrough=false`.
