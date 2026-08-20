# v164.1：全局仿射输运没有救回五相机梯度尾部

更新：2026-08-20

## 先说结论

v163 已证明直接持续上一时刻系数会放大稀疏视角梯度尾部。v164.1 改用一个物理上不同的因果机制：只从当前/上一时刻二维观测、报告相机几何和上一时刻部署重建中拟合一个三维全局仿射流，先输运旧场，再做固定 H1 更新。

正式运行与完全独立第二实现得到相同判决：`FAIL_OBSERVATION_AFFINE_TRANSPORT_V164_1`。

主策略通过 `10/12` 个冻结时间×相机分层，但仍在 `t=0.75` 和 `t=1.0` 的五相机梯度门失败。无输运 centered-H1 control 也恰好失败这两层；仿射输运虽然局部减轻了误差，却没有把它们救回，而且非初始时刻从 `1A+1A^T` 增加到 `13A+1A^T`。因此关闭当前全局仿射输运机制，不调 warp、不换半径，也不用大网络或 GPU 挽救。

## 冻结机制

仿射族包含 `12` 个生成元：三轴平移与九个线性坐标项。每个非初始时刻从上一部署场的空间梯度构造这些方向，用当前与上一时刻的观测创新做列归一化最小二乘；SVD 截断固定为 `1e-10`。拟合矩阵经齐次矩阵指数得到可逆变换，位移上限固定为每 `0.25` 归一化时间一个最小网格单元，再用逆映射三线性插值输运上一场。最后以输运场为中心做固定倍数 `0.03` 的 H1 更新。

这不是学习模型。没有读取当前真值来生成方向、选参数、回退或停止，也没有结果后搜索替代 warp、半径、截断或 H1 倍数。

## 十二个冻结分层

下表为主策略的 gradient p90 / worst；绝对门分别为 `0.75 / 1.00`。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.708532 / 0.791316 PASS | 0.573003 / 0.610096 PASS | 0.494193 / 0.529490 PASS |
| 0.25 | 0.723637 / 0.954076 PASS | 0.606784 / 0.688111 PASS | 0.531897 / 0.569822 PASS |
| 0.75 | 0.788531 / 1.078302 FAIL | 0.631906 / 0.798318 PASS | 0.565899 / 0.688588 PASS |
| 1.00 | 0.765210 / 1.157668 FAIL | 0.656006 / 0.787558 PASS | 0.561305 / 0.582594 PASS |

两处失败的 field p90 为 `0.329650 / 0.361259`，observation p90 为 `0.117589 / 0.119712`，都守住各自绝对门；失败集中在稀疏视角梯度尾部。预注册目标层 `t=0.75` 五相机的 frozen H1 gradient p90 为 `0.758639`，仿射输运反而变为 `0.788531`，没有达到“严格改善”条件。

## Control 与成本

不做输运、只以上一部署场为中心的 H1 control 也通过 `10/12`，失败层同样是 `t=0.75` 和 `t=1.0` 五相机。它每个 cell 的逻辑在线账为 `1A+1A^T`；仿射主策略在非初始时刻要用十二个方向拟合观测创新，账为 `13A+1A^T`。主策略没有用额外十二次 forward 换来完整分层通过，因此不能进入 predictor、wall/RSS 或外部门。

## 独立复算

第二实现独立重建 `39` 个算子设置、`1,404` 个 cells 和四臂共 `5,616` 条记录，`39/39` 项检查通过。仿射参数、主系数、逐 cell 指标、汇总指标与相机乱序审计的最大相对/绝对差分别为 `5.84e-11 / 1.49e-10 / 5.79e-11 / 7.13e-12 / 7.09e-13`。全部仿射拟合秩均为 `12`，最小变换行列式为 `0.930688`，位移上限严格守住。

早先 v164 执行因历史运行时的逐位重放不稳定而记为 inconclusive，且没有解释其科学数组。v164.1 没有修改机制、数据、控制、阈值或判决顺序，只改为绑定正确输入根，并要求 formal 与 independent 对同一次当前运行逐项一致。

## 证据边界

这关闭的是当前全局仿射输运表示，不是整个 C 路线，也不证明所有局部或非刚性输运都无效。二维位移仍由可执行三维重建场和相机几何仿真，不是逐工况配对的真实实验 BOST。没有 predictor、fresh wall/RSS、独立外部工况或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v164.1: global affine transport does not repair five-camera gradient tails

Updated: 2026-08-20

v164.1 tests a causal mechanism physically distinct from v163 coefficient persistence. A twelve-generator 3D affine flow is fitted only from current/previous simulated observations, reported camera geometry, and the previous deployed reconstruction. The previous field is inverse-map transported and then used as the center of the frozen H1 update.

The formal run and fully independent second implementation agree on `FAIL_OBSERVATION_AFFINE_TRANSPORT_V164_1`. The primary clears `10/12` frozen time-by-camera strata but fails five cameras at both `t=0.75` and `t=1.0`. Gradient p90 / worst are `0.788531 / 1.078302` and `0.765210 / 1.157668`, against limits `0.75 / 1.00`. Field and observation p90 remain within their gates in both strata.

The affine family contains three translations and nine linear coordinate terms. Observation-innovation fitting uses column normalization and a fixed SVD cutoff of `1e-10`; an exact homogeneous matrix exponential and a one-minimum-cell displacement cap preserve a small invertible map. No current truth, alternate warp, radius, cutoff, or H1 multiplier is selected from results.

The no-transport centered-H1 control also clears `10/12` and fails the same two five-camera strata. At the preregistered `t=0.75` target, frozen H1 has gradient p90 `0.758639`, while affine transport worsens it to `0.788531`. More importantly, the no-transport control costs `1A+1A^T` per cell, whereas each non-anchor affine update costs `13A+1A^T`. The extra twelve forward calls do not produce complete matched accuracy.

The independent implementation rebuilds `39` operator setups, `1,404` cells, and `5,616` rows across four arms. All `39/39` checks pass. Maximum affine-parameter, primary-coefficient, per-cell, summary, and camera-permutation differences are `5.84e-11`, `1.49e-10`, `5.79e-11`, `7.13e-12`, and `7.09e-13`. Every affine fit has rank `12`, the minimum determinant is `0.930688`, and the displacement cap is respected.

An earlier v164 execution was inconclusive because historical runtime bit replay was not stable, and its scientific arrays were not interpreted. v164.1 changes no scientific mechanism, threshold, control, or decision order; it binds the corrected input roots and requires formal-versus-independent current-run equality.

This closes the frozen global affine-transport representation, not the C route and not every possible local or nonrigid transport mechanism. The observations remain simulated from executable 3D reconstructed fields and camera geometry rather than condition-matched experimental BOST. No predictor, fresh wall/RSS, external-condition, or real-BOST claim is established.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
