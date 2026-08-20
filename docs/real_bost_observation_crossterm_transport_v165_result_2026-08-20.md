# v165：纯交叉项输运仍未救回五相机梯度尾部

更新：2026-08-20

## 先说结论

v164.1 排除了十二参数全局仿射输运，但仍留下一个具体问题：失败是否来自仿射族只能表示整体平移、旋转、剪切和尺度变化，无法表达非仿射空间耦合。v165 因此保持相同参数数和相同非初始在线账，去掉全部常数与线性项，只保留 `sx*sy`、`sx*sz`、`sy*sz`、`sx*sy*sz` 四类交叉模式，并分别作用到三个位移分量。

正式运行与完全独立第二实现得到相同判决：`FAIL_OBSERVATION_CROSSTERM_TRANSPORT_V165`。

主策略仍只通过 `10/12` 个冻结时间×相机分层，在 `t=0.75` 和 `t=1.0` 的五相机梯度门失败。它既没有救回 frozen H1 已失败的尾部，也没有在同预算下稳定优于 v164.1 仿射输运。因此关闭这个精确定义的纯交叉项家族，不追加次数、不混入仿射项、不调 cap 或 H1，也不以更大模型或 GPU 挽救。

## 冻结机制

每个位移分量使用四个无量纲模式：`sx*sy`、`sx*sz`、`sy*sz` 和 `sx*sy*sz`，共十二个参数。这里没有平移、常数项或任何线性仿射项。每个非初始时刻只用上一部署重建的空间梯度、当前/上一仿真二维双分量观测与报告几何构造切向观测，并以列归一化最小二乘和固定 `1e-10` SVD cutoff 拟合当前观测创新。

位移场先受每 `0.25` 归一化时间一个最小网格单元的位移 cap，再受 `0.5` 的位移 Jacobian 谱范数 cap。随后用逆映射、三线性插值和域外置零输运上一部署场，并以该场为中心做固定 `0.03` H1 更新。模式、符号、尺度、边界、两个 cap、tie-break、controls、四个时间、`5/7/9` 相机、门与成本账都在结果前固定；当前三维真值没有进入拟合、方向生成、回退或停止。

## 十二个冻结分层

下表为主策略的 gradient p90 / worst；绝对门分别为 `0.75 / 1.00`。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.708532 / 0.791316 PASS | 0.573003 / 0.610096 PASS | 0.494193 / 0.529490 PASS |
| 0.25 | 0.720562 / 0.949122 PASS | 0.602224 / 0.707711 PASS | 0.533387 / 0.552850 PASS |
| 0.75 | 0.801162 / 1.080751 FAIL | 0.626887 / 0.784896 PASS | 0.561417 / 0.637569 PASS |
| 1.00 | 0.759218 / 1.148727 FAIL | 0.636427 / 0.757109 PASS | 0.550045 / 0.581194 PASS |

两处失败的 field p90 为 `0.333054 / 0.326976`，observation p90 为 `0.117583 / 0.119543`，都守住各自绝对门；失败仍集中在稀疏视角梯度尾部。

## 同预算与便宜 control

非初始 cell 的逻辑在线账与 v164.1 完全相同，均为 `13A+1A^T`；不输运 centered-H1 与 frozen H1 都为 `1A+1A^T`。

- 在 `t=0.75` 五相机，v165 gradient p90 为 `0.801162`，比同预算仿射的 `0.788531` 更差，也比 frozen H1 的 `0.758639` 更差。
- 在 `t=1.0` 五相机，v165 的 `0.759218` 比仿射的 `0.765210` 小 `0.005992`，但仍高于绝对门，并明显差于 frozen H1 的 `0.712033`；gradient worst `1.148727` 也仍越线。

因此不能把 t=1.0 的一处局部改善挑出来写成成功。相同的十二次额外 forward 没有换来完整精度通过。

## 独立复算

第二实现独立重建 `39` 个算子设置、`1,404` 个 cells 和四臂共 `5,616` 条记录，`48/48` 项检查通过。交叉参数、主系数、逐 cell 指标、汇总指标与相机乱序审计的最大相对/绝对差分别为 `1.21e-10 / 1.58e-10 / 5.79e-11 / 2.74e-11 / 7.27e-13`。全部拟合秩均为 `12`；最大位移 Jacobian 谱范数为 `0.202748`，最小行列式下界为 `0.506742`，两个冻结 cap 均守住。

两个较早尝试分别在导入项目模块和导入 Torch 阶段失败，均发生在读取模型和生成科学数组之前；原始失败证据已保留，且没有被解释为科学结果。唯一有效正式运行与独立验证使用同一冻结机制。

## 证据边界

这关闭的是精确定义的纯 `xy/xz/yz/xyz` 交叉项输运家族，不是整个 C 路线，也不证明所有局部或非刚性输运都不可能。二维位移仍由可执行三维重建场和相机几何仿真，不是逐工况配对的真实实验 BOST。没有 predictor、fresh wall/RSS、独立外部工况或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v165: pure cross-term transport still does not repair five-camera gradient tails

Updated: 2026-08-20

v165 asks whether v164.1 failed because a global affine family cannot express non-affine spatial coupling. It preserves the same twelve parameters and the same non-anchor online cost, removes every constant and linear-affine term, and retains only `sx*sy`, `sx*sz`, `sy*sz`, and `sx*sy*sz` for each of the three displacement components.

The formal run and fully independent second implementation agree on `FAIL_OBSERVATION_CROSSTERM_TRANSPORT_V165`. The primary clears `10/12` frozen time-by-camera strata but again fails five cameras at `t=0.75` and `t=1.0`. Gradient p90 / worst are `0.801162 / 1.080751` and `0.759218 / 1.148727`, against limits `0.75 / 1.00`. Field and observation p90 remain within their gates in both strata.

At each non-anchor time, the twelve coefficients are fitted only from current/previous simulated two-component observations, reported geometry, and the previous deployed reconstruction. Fitting uses column normalization and a fixed `1e-10` SVD cutoff. The frozen inverse-map warp uses trilinear interpolation, zero outside the domain, a one-minimum-cell displacement cap per `0.25` time step, and a displacement-Jacobian spectral-norm cap of `0.5`; the centered H1 multiplier remains `0.03`. No current truth, affine term, alternate mode, cap, or multiplier is selected from results.

The non-anchor logical cost is `13A+1A^T`, identical to v164.1 affine transport. At `t=0.75`, cross-term gradient p90 `0.801162` is worse than affine `0.788531` and frozen H1 `0.758639`. At `t=1.0`, it improves affine by only `0.005992` to `0.759218`, still fails the absolute gate, and remains worse than frozen H1 `0.712033`. This isolated improvement is therefore not a passing result.

The independent implementation rebuilds `39` operator setups, `1,404` cells, and `5,616` rows across four arms. All `48/48` checks pass. Maximum cross-parameter, primary-coefficient, per-cell, summary, and camera-permutation differences are `1.21e-10`, `1.58e-10`, `5.79e-11`, `2.74e-11`, and `7.27e-13`. Every fit has rank `12`; the maximum displacement-Jacobian spectral norm is `0.202748`, and the minimum determinant lower bound is `0.506742`.

Two earlier launch attempts failed during project-module and Torch import, before model reading or scientific-array generation. Their evidence is preserved and is not treated as science. The only interpreted run is the frozen formal execution followed by the independent recomputation.

This closes the exact pure `xy/xz/yz/xyz` cross-term family without mode, cap, H1, larger-model, or GPU rescue. It does not close the C route or prove that all local/nonrigid transport is impossible. Observations remain simulated from executable reconstructed 3D fields and camera geometry rather than condition-matched experimental BOST. No predictor, fresh wall/RSS, external-condition, or real-BOST claim is established.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
