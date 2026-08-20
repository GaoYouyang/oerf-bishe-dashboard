# v168：局部无散旋涡仍未救回晚时刻五相机梯度尾部

更新：2026-08-21

## 先说结论

v167 已经排除了固定四分区、静止平移速度的局部连续性流。v168 改成物理上不同的局部机制：四个固定四面体位置的标量包络各自产生三个旋度速度场，共十二个解析无散的局部旋涡方向。速度在物理边界处严格为零，密度沿流线只做体积保持输运，不再使用 v167 的局部压缩或膨胀因子。

正式运行与完全独立第二实现得到相同判决：`FAIL_OBSERVATION_LOCAL_DIVFREE_VORTEX_V168`。

主策略只通过 `10/12` 个冻结时间×相机分层。`t=0.75`、五相机的 gradient p90 / worst 为 `0.817990 / 1.158071`；`t=1.0` 为 `0.759393 / 1.271909`。两个分层都越过 `0.75 / 1.00` 梯度门。

它没有修复 v167，反而把同预算 v167 的两组值 `0.813123 / 1.145759` 与 `0.757524 / 1.244834` 分别继续推高。只需 `1A+1A^T` 的冻结 H1 在同两层为 `0.758639 / 0.835752` 与 `0.712033 / 0.789085`，明显更好。

因此关闭这个精确定义的四包络、边界渐消、静止无散旋涡家族。它不关闭所有局部、时变或更一般的无散流，也不关闭整个 C 路线；但不再通过改包络、宽度、cap、SVD、RK4、插值、H1 或更大网络挽救。

## 机制与成本

每个标量包络 `psi_r` 由四面体对称中心、固定宽度和边界渐消项构成。三个单位轴分别作为向量势方向：

`A_rj = h_min^2 * psi_r * e_j`

对应速度为：

`u_rj = curl(A_rj) = h_min^2 * grad(psi_r) x e_j`

所以解析上 `div(u_rj)=0`，边界包络、梯度与速度都为零。密度的一阶方向是 `-u_rj · grad(rho)`。十二个系数通过当前观测残差中的投影作用做列归一化最小二乘，SVD cutoff 固定为 `1e-10`；速度使用解析保守 cap，再以固定 16 步 RK4 反向积分。

初始时刻逻辑在线账为 `1A+1A^T`；其余时刻为 `13A+1A^T`，与 v167 完全相同。冻结 H1 control 为 `1A+1A^T`，CGLS K16 为 `16A+16A^T`。几何缓存、方向审计和离线真值评分均单独披露，不能算作免费部署成本。

## 十二个冻结分层

下表为主策略的 gradient p90 / worst；绝对门分别为 `0.75 / 1.00`。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.708532 / 0.791316 PASS | 0.573003 / 0.610096 PASS | 0.494193 / 0.529490 PASS |
| 0.25 | 0.730398 / 0.983582 PASS | 0.609130 / 0.720017 PASS | 0.529789 / 0.548498 PASS |
| 0.75 | 0.817990 / 1.158071 FAIL | 0.626078 / 0.829029 PASS | 0.566442 / 0.605418 PASS |
| 1.00 | 0.759393 / 1.271909 FAIL | 0.636816 / 0.848172 PASS | 0.546341 / 0.586579 PASS |

两个失败分层的 field p90 为 `0.327235 / 0.322356`，observation p90 为 `0.117324 / 0.119172`，都守住各自绝对门。失败仍集中在稀疏视角下的三维梯度尾部。

## 独立复算

第二实现独立重建 `39` 个算子设置、`1,404` 个 cells 和四臂共 `5,616` 条记录，`60/60` 项检查全部通过。

局部参数、主系数、输运 prior、逐 cell 指标和汇总指标的最大相对/绝对差分别为 `2.49e-10`、`1.89e-10`、`1.77e-10`、`5.79e-11` 和 `7.13e-12`。相机换序后的参数、主结果与 prior 最大相对差均不超过 `1.28e-12`。全部局部拟合秩为 `12`；边界包络、边界梯度、边界速度和解析散度均为 `0`；流往返误差相对最小网格间距为 `7.87e-16`，密度因子恒为 `1`。

第一次独立验证尝试在首个非初始时刻的数值容差计算处因缺失机器精度属性而中止，没有产生验证 rows、汇总或科学判决，也没有复用其输出。只修正这一处数值属性访问后，第二次从头独立复算并形成上述判决。这个过程属于执行完整性，不是算法成果。

## 证据边界

这关闭的是固定四包络、固定宽度、边界渐消、静止无散旋涡，以及冻结 cap / SVD / RK4 / 插值 / H1 的精确家族。它不证明所有局部或时变无散输运不可能。二维位移仍由可执行三维密度场和相机几何仿真，不是逐工况配对的真实 BOST。

没有 predictor、fresh wall/RSS、独立外部工况或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v168: local divergence-free vortices still do not repair late five-camera gradient tails

Updated: 2026-08-21

v168 tests a mechanism physically different from v167's locally compressible continuity flow. Four scalar envelopes at fixed tetrahedral locations each generate three curl-based velocity fields, producing twelve analytically divergence-free local vortex directions. The envelopes, their gradients, and the velocities vanish on the physical boundary. Density therefore undergoes volume-preserving advection with a factor exactly equal to one.

The formal run and fully independent second implementation agree on `FAIL_OBSERVATION_LOCAL_DIVFREE_VORTEX_V168`. The primary clears `10/12` frozen time-by-camera strata. Five-camera gradient p90 / worst are `0.817990 / 1.158071` at `t=0.75` and `0.759393 / 1.271909` at `t=1.0`, exceeding the `0.75 / 1.00` limits in both strata.

The vortex family does not improve on same-budget v167, whose corresponding values are `0.813123 / 1.145759` and `0.757524 / 1.244834`. It remains substantially worse than the cheaper frozen H1 control at `0.758639 / 0.835752` and `0.712033 / 0.789085`. The non-anchor logical cost remains `13A+1A^T`, versus `1A+1A^T` for H1.

Each vector potential is `A_rj = h_min^2 * psi_r * e_j`, with velocity `u_rj = curl(A_rj)`. The signed density tangent is `-u_rj · grad(rho)`. Twelve coefficients are fitted from projected current observation residuals using column-normalized least squares and a fixed `1e-10` SVD cutoff. A conservative analytic speed cap and fixed 16-step inverse RK4 flow complete the transport.

The independent implementation rebuilds `39` operator setups, `1,404` cells, and `5,616` rows across four arms. All `60/60` checks pass. Maximum local-parameter, primary-coefficient, transported-prior, per-cell, and summary differences are `2.49e-10`, `1.89e-10`, `1.77e-10`, `5.79e-11`, and `7.13e-12`. Camera-reordering differences stay below `1.28e-12`. Every local fit has rank `12`; boundary envelope, boundary gradient, boundary velocity, and analytic divergence are all zero; the flow round-trip error is `7.87e-16` relative to minimum spacing, and the density factor is exactly one.

The first independent-validation attempt stopped during its first non-anchor numeric-tolerance calculation because a machine-epsilon attribute was omitted. It produced no validation rows, summaries, or scientific decision, and none of its outputs were reused. After correcting only that attribute access, a fresh independent recomputation produced the verdict above. This is execution integrity, not an algorithmic result.

This closes the exact fixed four-envelope, boundary-tapered, stationary divergence-free vortex family without envelope, width, cap, SVD, RK4, interpolation, H1, larger-model, or GPU rescue. It does not close every local or time-varying divergence-free flow, nor the full C route. Observations remain simulated from executable reconstructed 3D density fields and camera geometry rather than condition-matched experimental BOST. No predictor, fresh wall/RSS, external-condition, real-BOST, paper-success, or algorithm-breakthrough claim is established.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
