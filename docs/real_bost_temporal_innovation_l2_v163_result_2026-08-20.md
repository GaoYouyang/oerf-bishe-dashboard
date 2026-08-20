# v163：单向时序系数持续在稀疏视角放大梯度尾部

更新：2026-08-20

## 先说结论

师兄确认历史二维对应数据已找不到后，当前数据链按“三维重建场 + 相机位置生成二维双分量位移”继续做受控虚拟 BOST。v163 检验一个与 v162 全局几何二次型不同的时序假设：当前时刻除了自己的二维观测和相机几何，还使用上一时刻已经部署得到的重建系数作为单向先验。

正式运行与独立第二实现得到相同判决：`FAIL_TEMPORAL_INNOVATION_L2_V163`。

主策略只通过 `7/12` 个冻结时间×相机分层。它没有改善稀疏视角的梯度尾部，反而把多个五相机和七相机分层推过绝对门。因此关闭当前“上一时刻系数直接持续到下一时刻”的机制，不调 alpha、不做结果后时间外推，也不用更大网络挽救。

## 冻结机制

`t=0` 使用已冻结的各向同性 H1、固定倍数 `0.03`。之后每个时刻只用当前二维观测、reported active-camera geometry 和上一时刻部署重建，求解：

`(G + alpha I)c = q + alpha c_previous`

其中 `alpha = 0.03 × median_positive_eigenvalue(G) × 0.25 / delta_t`。同尺度 static L2 零先验作为便宜确定性 control；另保留 H1 和 CGLS K16 父对照。没有搜索替代 alpha、双向平滑、外推或候选集合。

## 十二个冻结分层

下表为主策略的 gradient p90 / worst；绝对门分别为 `0.75 / 1.00`。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.708532 / 0.791316 PASS | 0.573003 / 0.610096 PASS | 0.494193 / 0.529490 PASS |
| 0.25 | 0.800719 / 1.081812 FAIL | 0.651135 / 0.799590 PASS | 0.555408 / 0.599917 PASS |
| 0.75 | 0.939342 / 1.358706 FAIL | 0.711873 / 1.015023 FAIL | 0.613098 / 0.676629 PASS |
| 1.00 | 0.864791 / 1.433536 FAIL | 0.711017 / 1.064363 FAIL | 0.583914 / 0.663787 PASS |

field 和 observation p90 大多仍好看，尤其 observation 全部低于 `0.2`；问题集中在 gradient worst 和五相机 gradient p90。这说明观测拟合良好并不能保证时序先验没有把旧的空间结构带入当前稀疏视角重建。

## 独立复算

第二实现重建 `39` 个算子设置、`1,404` 个 cells 和四臂共 `5,616` 条记录。`28/28` 项检查通过。主系数最大相对差 `1.89e-10`，逐 cell 指标最大差 `6.62e-11`，汇总最大差 `7.23e-12`，相机乱序后的主系数最大相对差 `6.05e-13`。

第一次独立运行曾因审计代码要求两种数学等价 RMS 归约在 JSON 中逐位相等而记为 inconclusive；二者只差 `5.55e-17`，其余 `27` 项检查全部通过。修复只把连续审计量改为 float64 舍入界，身份与离散字段仍严格相等；正式数组、算法、阈值和判决规则均未改变。

## 证据边界

这关闭的是当前单向系数持续机制，不是整个 C 路线，也不能证明所有时序信息都无效。当前二维位移仍由三维重建场和相机位置仿真，不是逐工况配对的真实实验 BOST。没有 predictor、wall/RSS、独立外部工况或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v163: one-sided temporal coefficient persistence amplifies sparse-view gradient tails

Updated: 2026-08-20

After the historical paired 2D data could not be recovered, the controlled virtual-BOST chain continues by generating two-component observations from executable 3D reconstructed fields and camera geometry. v163 tests a mechanism physically distinct from v162's global geometry quadratic: each current reconstruction uses its own observation and geometry plus the immediately preceding deployed coefficient vector as a one-sided prior.

The formal run and independent second implementation agree on `FAIL_TEMPORAL_INNOVATION_L2_V163`. The primary clears only `7/12` frozen time-by-camera strata. It does not repair sparse-view gradient tails and instead pushes several five- and seven-camera strata beyond the absolute limits.

At `t=0`, the method uses the frozen isotropic H1 reference with multiplier `0.03`. Later times solve `(G + alpha I)c = q + alpha c_previous`, where `alpha = 0.03 × median_positive_eigenvalue(G) × 0.25 / delta_t`. Inputs are limited to the current simulated observation, reported active-camera geometry, and the previous deployed reconstruction. No alternate alpha, bidirectional smoother, extrapolation, or candidate set is selected from results.

The strongest failures occur at five cameras: gradient p90 / worst are `0.800719 / 1.081812` at `t=0.25`, `0.939342 / 1.358706` at `t=0.75`, and `0.864791 / 1.433536` at `t=1.0`, against frozen limits `0.75 / 1.00`. Seven-camera worst values also fail at `t=0.75` and `t=1.0`. Observation p90 remains below `0.2` throughout, demonstrating that good observation fit does not guarantee a physically safe temporal prior.

The independent implementation rebuilds `39` operator setups, `1,404` cells, and `5,616` rows across four arms. All `28/28` checks pass. Maximum relative primary-coefficient, per-cell metric, summary, and camera-permutation differences are `1.89e-10`, `6.62e-11`, `7.23e-12`, and `6.05e-13` respectively.

An initial independent attempt was inconclusive only because the audit required bitwise JSON equality for two mathematically equivalent RMS reductions differing by `5.55e-17`; all other `27` checks passed. The repair kept exact identity/discrete checks and introduced only a float64 roundoff bound. Formal arrays, the mechanism, thresholds, and the scientific decision rule were unchanged.

This closes the frozen one-sided coefficient-persistence mechanism, not the C route and not all possible uses of temporal information. The observations remain simulated from reconstructed 3D fields and camera geometry rather than condition-matched experimental BOST. No predictor, wall/RSS, external-condition, or real-BOST claim is established.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
