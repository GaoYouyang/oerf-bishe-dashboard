# v167：四分区局部连续性流没有救回晚时刻五相机梯度尾部

更新：2026-08-20

## 先说结论

v166 已经排除了质量守恒的**全局**仿射输运。v167 改成物理上不同的局部机制：用四个固定、平滑且四面体对称的空间分区，每个分区拟合三维平移速度，共十二个只由当前二维观测、报告几何和上一时刻部署重建决定的系数。密度仍按连续性方程输运。

正式运行与完全独立第二实现得到相同判决：`FAIL_OBSERVATION_LOCAL_CONTINUITY_FLOW_V167`。

主策略只通过 `10/12` 个冻结时间×相机分层。`t=0.75`、五相机的 gradient p90 / worst 为 `0.813123 / 1.145759`；`t=1.0` 为 `0.757524 / 1.244834`。两个分层都越过 `0.75 / 1.00` 梯度门。

它没有修复 v166 的失败，反而把同预算 v166 的两组 gradient 值 `0.795556 / 1.059791` 与 `0.730257 / 1.087987` 进一步变差，也明显差于只需 `1A+1A^T` 的冻结 H1。

因此关闭这个精确定义的四分区、静止速度、局部连续性流家族。它不关闭所有局部流或整个 C 路线，但不再通过增加分区、修改插值、调 SVD / cap / H1 或扩大网络来挽救。

## 机制与成本

四个软分区在物理域内固定，权重处处为正且和为 1。每个分区有三个平移方向，形成十二个局部速度方向。对密度 `rho`，一阶方向为：

`-div(rho * phi_r * e_j)`

十二个系数通过当前观测残差中的投影作用做列归一化最小二乘，SVD cutoff 固定为 `1e-10`。速度统一受一个网格尺度 cap 约束，再用 16 步 RK4 反向积分流，并沿路径累计连续性方程的 log-density 因子。

初始时刻逻辑在线账为 `1A+1A^T`；其余时刻为 `13A+1A^T`，与 v166 相同。冻结 H1 control 为 `1A+1A^T`，CGLS K16 为 `16A+16A^T`。几何缓存、方向审计和离线真值评分均单独披露，不能算作免费部署成本。

## 运行前机械纠错

在读取正式科学数组前，合成一阶变分检查发现，普通三线性插值在网格节点不可微，且零流在上边界不稳定，无法满足本轮连续性切向与有限流的一致性合同。

实现因此在结果前改为 C1 Catmull-Rom 三次卷积、显式零延拓、边界数值吸附和与零延拓一致的中心导数。全部十二个一阶方向、零流恒等、独立分区/切向/拟合/RK4/warp 对照随后通过。这个修复只说明执行链机械正确，不是算法成果。

第一次正式执行在完整计算后因 JSON 布尔量序列化失败，没有生成 formal report 或 READY；其科学数组没有被读取或复用。仅把布尔量显式转成原生类型后，用新 run ID 完整重跑。

## 十二个冻结分层

下表为主策略的 gradient p90 / worst；绝对门分别为 `0.75 / 1.00`。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.708532 / 0.791316 PASS | 0.573003 / 0.610096 PASS | 0.494193 / 0.529490 PASS |
| 0.25 | 0.728404 / 0.978659 PASS | 0.608198 / 0.716863 PASS | 0.528809 / 0.548719 PASS |
| 0.75 | 0.813123 / 1.145759 FAIL | 0.624086 / 0.824340 PASS | 0.564703 / 0.604462 PASS |
| 1.00 | 0.757524 / 1.244834 FAIL | 0.632306 / 0.829116 PASS | 0.542512 / 0.580882 PASS |

两个失败分层的 field p90 为 `0.333662 / 0.331701`，observation p90 为 `0.117309 / 0.119180`，都守住各自绝对门。失败仍集中在稀疏视角下的三维梯度尾部。

## 独立复算

第二实现独立重建 `39` 个算子设置、`1,404` 个 cells 和四臂共 `5,616` 条记录，`58/58` 项检查全部通过。

局部流参数、主系数、输运 prior、逐 cell 指标、汇总指标和相机乱序主结果的最大相对/绝对差分别为 `7.39e-11`、`1.80e-10`、`1.66e-10`、`5.79e-11`、`7.13e-12` 和 `7.71e-13`。全部局部拟合秩为 `12`；分区和误差为 `3.33e-16`，流往返误差相对最小网格间距为 `2.10e-14`，密度因子范围为 `[0.908621, 1.044928]`。

## 证据边界

这关闭的是固定四分区、固定静止局部速度、固定 RK4 / SVD / cap / H1 下的局部连续性流，不证明所有局部、时变或更一般流都不可能。二维位移仍由可执行三维密度场和相机几何仿真，不是逐工况配对的真实 BOST。

没有 predictor、fresh wall/RSS、独立外部工况或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v167: four-region local continuity flow does not repair late five-camera gradient tails

Updated: 2026-08-20

v167 tests a physically different mechanism after v166 closed global affine continuity transport. Four fixed, smooth, tetrahedrally symmetric spatial partitions each carry a three-component translation velocity, producing twelve local directions. Their coefficients are fitted only from current simulated two-component observations, reported geometry, and the preceding deployed reconstruction. Density is transported through the continuity equation.

The formal run and fully independent second implementation agree on `FAIL_OBSERVATION_LOCAL_CONTINUITY_FLOW_V167`. The primary clears `10/12` frozen time-by-camera strata. Five-camera gradient p90 / worst are `0.813123 / 1.145759` at `t=0.75` and `0.757524 / 1.244834` at `t=1.0`, exceeding the `0.75 / 1.00` limits in both strata.

The local family is not an improvement over the same-budget v166 global affine continuity transport, whose corresponding values were `0.795556 / 1.059791` and `0.730257 / 1.087987`. It is also worse than the cheaper frozen H1 control. The non-anchor logical cost remains `13A+1A^T`, versus `1A+1A^T` for H1.

Each infinitesimal direction is `-div(rho * phi_r * e_j)`. The twelve coefficients use column-normalized least squares with a fixed `1e-10` SVD cutoff. The capped stationary velocity is integrated by a fixed 16-step inverse RK4 flow while accumulating the continuity-equation log-density factor. The transported density centers the fixed `0.03` H1 solve.

Before formal scientific arrays were read, synthetic first-variation checks exposed that ordinary trilinear interpolation is non-differentiable at grid nodes and unstable for zero flow at the upper boundary. The frozen implementation was corrected to C1 Catmull-Rom cubic convolution with explicit zero extension and a matching derivative. All twelve first variations and independent mechanical checks then passed. This is engineering validity, not scientific progress. A first full calculation later failed only while serializing a non-native JSON boolean, producing neither a formal report nor READY; its scientific arrays were not read or reused. A fresh run followed the serialization-only fix.

The independent implementation rebuilds `39` operator setups, `1,404` cells, and `5,616` rows across four arms. All `58/58` checks pass. Maximum local-parameter, primary-coefficient, transported-prior, per-cell, summary, and camera-permutation differences are `7.39e-11`, `1.80e-10`, `1.66e-10`, `5.79e-11`, `7.13e-12`, and `7.71e-13`. Every local fit has rank `12`; the partition-sum error is `3.33e-16`, the flow round-trip error relative to minimum grid spacing is `2.10e-14`, and density factors stay in `[0.908621, 1.044928]`.

This closes the exact fixed four-region stationary local-continuity family without partition-count, interpolation, SVD, cap, H1, larger-model, or GPU rescue. It does not close every local or time-varying flow, nor the full C route. Observations remain simulated from executable reconstructed 3D density fields and camera geometry rather than condition-matched experimental BOST. No predictor, fresh wall/RSS, external-condition, real-BOST, paper-success, or algorithm-breakthrough claim is established.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
