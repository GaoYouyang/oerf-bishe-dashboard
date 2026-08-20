# v166：质量守恒全局仿射输运仍未稳住五相机梯度尾部

更新：2026-08-20

## 先说结论

v164.1 和 v165 都把三维场当成被动标量做坐标拉回，但师兄已明确模型第 0 通道是密度。v166 因此不再调整坐标模式，而是修正物理作用：一阶切向用 `-div(rho*u)`，精确仿射输运用逆映射采样并乘以 `1/det(A)`，从而守住密度的质量守恒关系。

正式运行与完全独立第二实现得到相同判决：`FAIL_OBSERVATION_CONTINUITY_AFFINE_TRANSPORT_V166`。

主策略仍只通过 `10/12` 个冻结时间×相机分层。`t=0.75`、五相机的 gradient p90 / worst 为 `0.795556 / 1.059791`；`t=1.0` 为 `0.730257 / 1.087987`。前者同时越过 `0.75 / 1.00` 两道门，后者虽让 p90 回到门内，worst 仍明显越线。

因此关闭这个精确定义的质量守恒全局仿射家族，不事后调 determinant 幂、cap、SVD cutoff、仿射生成元或 H1 乘数，也不用大网络或 GPU 挽救。

## 为什么这次与 v164.1 不同

v164.1 的仿射速度场是 `u(x)`，但切向只用 `-u·grad(rho)`，精确 warp 也只采样 `rho(A^-1(x-b))`。这适合被动染料，不是可压缩密度的守恒律。

v166 对同一十二生成元使用：

`-div(rho*u) = -u·grad(rho) - rho*div(u)`

并用：

`rho_current(x) = rho_previous(A^-1(x-b)) / det(A)`

平移项的散度为零；三个对角线性项有常数散度。这个 divergence 项和精确 Jacobian 密度因子都在结果前冻结，没有从得分中选取。

## 十二个冻结分层

下表为主策略的 gradient p90 / worst；绝对门分别为 `0.75 / 1.00`。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.708532 / 0.791316 PASS | 0.573003 / 0.610096 PASS | 0.494193 / 0.529490 PASS |
| 0.25 | 0.723599 / 0.940327 PASS | 0.608429 / 0.684131 PASS | 0.533604 / 0.569645 PASS |
| 0.75 | 0.795556 / 1.059791 FAIL | 0.626477 / 0.788318 PASS | 0.561726 / 0.683705 PASS |
| 1.00 | 0.730257 / 1.087987 FAIL | 0.622453 / 0.742742 PASS | 0.536706 / 0.574866 PASS |

两个失败分层的 field p90 为 `0.329900 / 0.314500`，observation p90 为 `0.117538 / 0.119237`，都守住各自绝对门。问题仍是稀疏视角下的三维梯度尾部，不是观测拟合本身。

## 同预算与便宜 control

非初始 cell 的逻辑在线账与 v164.1、v165 相同，都是 `13A+1A^T`；冻结 H1 为 `1A+1A^T`。

- `t=0.75`、五相机：v166 gradient p90 从 v165 的 `0.801162` 改善到 `0.795556`，worst 从 `1.080751` 改善到 `1.059791`，但仍同时失败，而便宜 H1 是 `0.758639 / 0.835752`。
- `t=1.0`、五相机：v166 从 v165 的 `0.759218 / 1.148727` 改善到 `0.730257 / 1.087987`；p90 过门，但 worst 仍失败，且便宜 H1 是 `0.712033 / 0.789085`。

这说明质量守恒修正确实改变了尾部，但没有把其变成合格策略，更没有超过成本便宜得多的 H1。

## 独立复算

第二实现独立重建 `39` 个算子设置、`1,404` 个 cells 和四臂共 `5,616` 条记录，`53/53` 项检查全部通过。

仿射参数、主系数、输运 prior、逐 cell 指标、汇总指标和相机乱序的最大相对/绝对差分别为 `7.04e-11`、`1.61e-10`、`1.46e-10`、`5.79e-11`、`1.36e-11` 和 `7.15e-13`。所有拟合秩都为 `12`，最小仿射行列式为 `0.988968`，`density_factor * det(A)` 相对 1 的最大绝对差为 `1.11e-16`。

## 证据边界

这关闭的是固定十二生成元、固定 cap、固定 SVD 和固定 H1 下的**质量守恒全局仿射输运**，不是整个 C 路线，也不证明所有局部或非仿射连续性流都不可行。二维位移仍由可执行三维密度场和相机几何仿真，不是逐工况配对的真实 BOST。

没有 predictor、fresh wall/RSS、独立外部工况或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v166: mass-conserving global affine transport still does not stabilize five-camera gradient tails

Updated: 2026-08-20

v166 corrects a physical mismatch left by v164.1 and v165. The model's first output is density, yet both predecessors transported it as a passive scalar. v166 keeps the same twelve global affine generators and the same non-anchor cost, but changes the infinitesimal action to `-div(rho*u)` and the exact push-forward to `rho(A^-1(x-b))/det(A)`.

The formal run and fully independent second implementation agree on `FAIL_OBSERVATION_CONTINUITY_AFFINE_TRANSPORT_V166`. The primary clears `10/12` frozen time-by-camera strata. Five-camera gradient p90 / worst are `0.795556 / 1.059791` at `t=0.75` and `0.730257 / 1.087987` at `t=1.0`, against limits `0.75 / 1.00`. Field and observation p90 stay within their gates in both failed strata.

At each non-anchor time, the twelve coefficients are fitted only from current simulated two-component observations, reported geometry, and the previous deployed reconstruction. The column-normalized least-squares fit uses a fixed `1e-10` SVD cutoff. The exact centered affine exponential is capped at one minimum-grid cell per `0.25` normalized-time step, inverse-sampled with trilinear interpolation and zero outside, and multiplied by the exact reciprocal determinant. The transported density then centers the fixed `0.03` H1 solve. No truth, alternate determinant power, generator, cap, SVD cutoff, or H1 multiplier is selected from results.

The non-anchor logical cost remains `13A+1A^T`; frozen H1 costs `1A+1A^T`. At `t=0.75`, v166 improves v165 from `0.801162 / 1.080751` to `0.795556 / 1.059791`, but still fails both gradient limits and remains worse than H1 `0.758639 / 0.835752`. At `t=1.0`, it improves v165 from `0.759218 / 1.148727` to `0.730257 / 1.087987`; p90 passes, but worst still fails and H1 is substantially better at `0.712033 / 0.789085`.

The independent implementation rebuilds `39` operator setups, `1,404` cells, and `5,616` rows across four arms. All `53/53` checks pass. Maximum affine-parameter, primary-coefficient, transported-prior, per-cell, summary, and camera-permutation differences are `7.04e-11`, `1.61e-10`, `1.46e-10`, `5.79e-11`, `1.36e-11`, and `7.15e-13`. Every fit has rank `12`; the minimum affine determinant is `0.988968`, and the maximum error in `density_factor * det(A) = 1` is `1.11e-16`.

This closes the exact mass-conserving global affine family without determinant, cap, SVD, H1, larger-model, or GPU rescue. It does not close the C route or prove that every local/non-affine continuity flow is impossible. Observations remain simulated from executable reconstructed 3D density fields and camera geometry rather than condition-matched experimental BOST. No predictor, fresh wall/RSS, external-condition, real-BOST, paper-success, or algorithm-breakthrough claim is established.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
