# v183：相机×分量分块 Galerkin 明显改善观测，但仍未通过完整门

## 做了什么，为什么这样做

v182 只沿一条全局残差方向更新，无法表达不同相机、不同探测器分量之间的响应差异。v183 因此冻结了一个物理上不同、仍然完全结果不可见的机制：把当前中心化二维观测残差按 **camera ID × detector component** 拆成物理块，每个块生成一条 Jacobi 预条件仿射坐标方向，再联合求解一次最小范数可观测 Galerkin 最小二乘。

五相机对应 `10` 条方向，九相机对应 `18` 条方向。系数只由当前观测、报告几何、fit-only 仿射空间和封存的 Jacobi 对角量决定。相对 SVD cutoff 固定为 `1e-12`；没有 ridge、阻尼、裁剪、回退、真值输入、超参数搜索或可训练参数。得到 warm field 后，再运行一轮完全未修改的物理 CGLS K1。

## 正式结果

分块确实比 v182 的全局单方向更有效，但仍不足以通过冻结门。

- 五相机 K1 的 field / gradient / observation p90 为 `0.445694 / 0.612373 / 0.226659`，严格通过 `1/52`，完整标定 `0/13`，完整帧 `0/4`。
- 九相机 K1 为 `0.371621 / 0.508927 / 0.207224`，严格通过 `37/52`，完整标定 `3/13`，完整帧 `1/4`。
- 冻结 p90 门是 field `<=0.50`、gradient `<=0.75`、observation `<=0.20`。两档 field 与 gradient 都通过，但 observation 仍越线。
- 相比 v182，五相机 observation p90 从 `0.244595` 降到 `0.226659`，九相机从 `0.266826` 降到 `0.207224`。

因此，**相机与分量之间的低阶异质性是真实存在且有用的**，尤其九相机已有 `37/52` 个单元严格安全；但当前每块一个固定系数的精确 Galerkin 表示仍不能实现完整 matched accuracy。正式科学判决是 `FAIL_OBSERVATION_BLOCK_GALERKIN_V183`。

## 独立复算

完全独立第二实现使用不同的 SVD 驱动路径，重新构造物理分块、方向、联合系数、候选场、观测、未修改 CGLS K1、全部指标、调用账和相机换序审计。`46/46` 项检查全真，全部离散判决一致。

候选场最大相对差为 `5.85e-12`，系数最大绝对差 `1.11e-10`，逐单元指标最大绝对差 `9.46e-13`，最大正规方程 stationarity 为 `4.28e-15`。固定观测输入在真值突变审计中的输出差为精确 `0`，相机换序后的候选场相对差为 `7.03e-16`。

## 成本与证据边界

逻辑在线账为 K0 `1A+1A^T`、K1 `3A+2A^T`，直接 CGLS K4 为 `4A+4A^T`；同时完整披露继承几何缓存构建的 `26,260` 次 forward-equivalent setup projection。因为 matched accuracy 失败，不能声称 exact-call 减少，也没有启动 wall/RSS 门。

v183 只关闭当前**每个相机分量一条方向、联合最小范数 Galerkin 系数**的精确机制。不得事后调 SVD cutoff、增加 ridge/阻尼、继续细分块或放宽门，也不能用更大 CNN/FNO/UNO/DeepONet 或 GPU 挽救。它没有关闭完整 C 路线，也不是数学不可能性证明。

`algorithm_breakthrough=false`、`paper_success=false`、`exact_call_reduction=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

# v183: camera-by-component block Galerkin improves observation, but still misses the complete gate

## What was tested and why

v182 updates along one global residual direction, so it cannot represent response differences between cameras and detector components. v183 freezes a physically distinct but still strictly deployment-visible mechanism: split the centered observation residual by **camera ID × detector component**, form one Jacobi-preconditioned affine-coordinate direction per physical block, and jointly solve a minimum-norm observable Galerkin least-squares problem.

Five cameras provide `10` directions and all nine provide `18`. Coefficients use only the current observation, reported geometry, the fit-only affine space, and a sealed Jacobi diagonal. The relative SVD cutoff is fixed at `1e-12`; there is no ridge, damping, clipping, fallback, target truth, hyperparameter search, or trainable parameter. One unchanged physical CGLS K1 step follows the warm field.

## Formal result

Blocking is materially better than the v182 global line, but still insufficient.

- Five-camera K1 field / gradient / observation p90 values are `0.445694 / 0.612373 / 0.226659`, with `1/52` strict-safe cells, `0/13` complete calibrations, and `0/4` complete frames.
- All-nine K1 values are `0.371621 / 0.508927 / 0.207224`, with `37/52` strict-safe cells, `3/13` complete calibrations, and `1/4` complete frames.
- Frozen p90 gates are field `<=0.50`, gradient `<=0.75`, and observation `<=0.20`. Field and gradient pass under both sensor arms, while observation still fails.
- Relative to v182, five-camera observation p90 falls from `0.244595` to `0.226659`, and all-nine falls from `0.266826` to `0.207224`.

Camera-and-component low-order heterogeneity is therefore real and useful, with `37/52` all-nine cells now strict-safe. Yet one fixed coefficient per block still cannot deliver complete matched accuracy. Decision: `FAIL_OBSERVATION_BLOCK_GALERKIN_V183`.

## Independent recomputation

A fully independent second implementation uses a different SVD driver and rebuilds physical blocks, directions, joint coefficients, candidate fields, observations, unchanged CGLS K1, every metric, call ledgers, and camera-permutation audits. All `46/46` checks pass and every discrete decision agrees.

Maximum candidate-field relative difference is `5.85e-12`, maximum coefficient absolute difference is `1.11e-10`, maximum per-cell metric absolute difference is `9.46e-13`, and maximum normal-equation stationarity is `4.28e-15`. Fixed-observation output changes by exactly `0` under truth mutation, and the camera-permuted candidate field differs by only `7.03e-16` relatively.

## Cost and claim boundary

Logical online ledgers are `1A+1A^T` for K0 and `3A+2A^T` for K1, versus `4A+4A^T` for direct CGLS K4. The inherited geometry-cache construction cost of `26,260` forward-equivalent setup projections is disclosed separately. Since matched accuracy fails, no exact-call reduction is established and wall/RSS testing is not authorized.

v183 closes only the exact **one-direction-per-camera-component, joint minimum-norm Galerkin coefficient** mechanism. Do not tune the SVD cutoff, add ridge or damping, split more blocks, relax gates, or rescue it with a larger CNN/FNO/UNO/DeepONet or GPU. It does not close the full C route and is not an impossibility result.

`algorithm_breakthrough=false`, `paper_success=false`, `exact_call_reduction=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
