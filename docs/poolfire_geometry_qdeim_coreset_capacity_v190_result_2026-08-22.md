# v190：固定 1280 列几何 QDEIM 压缩未保住完整 DCT 容量

## 为什么做

v189 已确认：每台相机保留完整 `24x24` 非 DC detector-potential DCT 后，五/九相机的未修改 CGLS K1 都能达到 `52/52`，并复现稠密 v185 容量。但完整表示在五相机下有 `2875` 个坐标，九相机有 `5175` 个坐标，不够紧凑。

v190 检验一个直接且可证伪的问题：能否只依据报告几何，从完整 DCT 中固定选择 `1280` 列，同时保住 v189 的物理容量？这一步不训练模型，也不测试速度。

## 怎么检验

保持同一条已开封 PoolFire p22 轨迹、四帧、13 套标定、五/九相机、1009 维仿射场空间、六个绝对误差门、K0/K1 和调用账不变。唯一候选先在任何 held-out 三维真值加载前完成全部 26 个几何设置的选择：

1. 从每个设置的完整逐相机 DCT 响应矩阵提取秩 `1009` 的响应子空间；
2. 用 QDEIM 固定选出 `1009` 个锚点；
3. 再按杠杆分数固定补入 `271` 列，总计 `1280`；
4. 不加 ridge、阻尼、回退、搜索、真值调参或可训练参数；
5. 选择完成并封存后，才加载四帧真值、构造候选并执行未修改的物理 K1。

五相机坐标数由 `2875` 降到 `1280`，减少 `55.48%`；九相机由 `5175` 降到 `1280`，减少 `75.27%`。这只是坐标数变化，不等于 exact-call、wall time 或内存收益。

## 独立复算后的结果

| arm | field p90 | gradient p90 | observation p90 | 严格通过 | 完整标定 | 完整时间层 |
|---|---:|---:|---:|---:|---:|---:|
| 五相机 K0 | 0.485969 | 0.870961 | 0.312125 | 0/52 | 0/13 | 0/4 |
| 九相机 K0 | 0.419797 | 0.703270 | 0.371062 | 0/52 | 0/13 | 0/4 |
| 五相机 K1 | 0.475126 | 0.844848 | 0.197362 | 35/52 | 2/13 | 0/4 |
| 九相机 K1 | 0.385466 | 0.633956 | 0.225603 | 30/52 | 1/13 | 0/4 |

一轮未修改 K1 确实救回了一部分单元，但没有恢复 v189 的完整容量。五相机主要败在 gradient p90：`0.844848 > 0.75`；九相机主要败在 observation p90：`0.225603 > 0.20`。两臂都没有一个完整时间层通过，而 v189 的完整 DCT 在两臂均为 `52/52`、`13/13`、`4/4`。

值得注意的是，所选子集仍保留 `1009/1009` 的响应秩，并包含 `818–920` 个 DCT12 之外的高频坐标，但这没有保证稳定物理逆。选后系统条件数范围约为 `262.44–652.18`，比完整 DCT 的 `48.93–187.41` 更差；相对完整 DCT，候选场逐单元最大相对差达到 `0.3996`，指标最大绝对差达到 `0.4251`。

完全独立第二实现采用不同 SVD driver，独立重建完整 DCT、响应矩阵、QDEIM 锚点、杠杆补列、候选、物理 K1、指标和分层，`59/59` 项检查全真。两条实现选出的锚点、补列和最终列集合完全一致；候选场最大相对差为 `3.11e-11`，指标最大绝对差为 `6.15e-12`。相机换序不改变选择，固定观测下修改真值也不改变候选。

## 科学结论

正式判决为 `FAIL_GEOMETRY_QDEIM1280_CORESET_CAPACITY_V190`。

在这条冻结、已开封的四帧容量诊断下，固定 `1280` 列的 geometry-only QDEIM + leverage 子集虽然保留了代数响应秩，却没有保住 held-out 物理逆和 matched-accuracy 尾部。因此当前固定 `1280` 列家族关闭：不提高预算、不事后调阈值或加权，也不用更大网络挽救。

这不证明所有紧凑表示都不可能，也没有否定 v189 的完整基容量。v190 没有训练预测器，没有完成完整轨迹、exact-call 减少、fresh wall/RSS、外部工况、曲线光路或真实 BOST 检验。下一条合格问题只能先利用已封存结果区分“固定子集的条件性损失”与“需要 observation-adaptive 坐标”，再结果前冻结一个物理上不同的表示。

`algorithm_breakthrough=false`，`paper_success=false`，GPU 与神经训练仍未授权。

# v190: a fixed 1280-column geometry QDEIM coreset does not preserve full-DCT capacity

## Why this test was needed

v189 established that the complete non-DC `24x24` detector-potential DCT in every camera gives `52/52` unchanged-CGLS-K1 cells under both five and nine cameras and reproduces dense v185 capacity. The complete representation, however, contains `2875` coordinates under five cameras and `5175` under nine.

v190 asks whether a fixed `1280`-column subset chosen from reported geometry alone can preserve that physical capacity. It trains no model and evaluates no speed claim.

## Frozen diagnostic

The opened PoolFire p22 trajectory, four frames, 13 calibrations, five/all-nine sensor arms, 1009-dimensional affine field space, six absolute gates, K0/K1 replay, and call accounting remain unchanged. Before any held-out 3D truth is loaded, each of the 26 geometry setups is processed as follows:

1. obtain the rank-`1009` response subspace of the complete per-camera DCT response;
2. select `1009` fixed QDEIM anchors;
3. add `271` fixed leverage-ranked supplement columns for a total of `1280`;
4. use no ridge, damping, fallback, search, truth-based tuning, or trainable parameter;
5. seal every selection before loading the four-frame truth and running candidate construction plus unchanged physical K1.

Coordinate count falls by `55.48%` under five cameras and `75.27%` under nine. This is only coordinate-count compression, not exact-call, wall-time, or memory evidence.

## Independently recomputed result

| arm | field p90 | gradient p90 | observation p90 | strict-safe | complete calibrations | complete times |
|---|---:|---:|---:|---:|---:|---:|
| Five-camera K0 | 0.485969 | 0.870961 | 0.312125 | 0/52 | 0/13 | 0/4 |
| All-nine K0 | 0.419797 | 0.703270 | 0.371062 | 0/52 | 0/13 | 0/4 |
| Five-camera K1 | 0.475126 | 0.844848 | 0.197362 | 35/52 | 2/13 | 0/4 |
| All-nine K1 | 0.385466 | 0.633956 | 0.225603 | 30/52 | 1/13 | 0/4 |

One unchanged K1 step repairs some cells but does not recover v189 capacity. Five-camera is limited mainly by gradient p90 (`0.844848 > 0.75`), while all-nine is limited mainly by observation p90 (`0.225603 > 0.20`). Neither arm passes any complete time stratum, compared with `52/52` cells, `13/13` calibrations, and `4/4` times for the v189 complete DCT.

The selected response retains rank `1009/1009` and contains `818–920` coordinates outside DCT12, yet algebraic rank does not ensure a stable physical inverse. Selected-system condition numbers range from about `262.44` to `652.18`, worse than the full-DCT range of `48.93–187.41`. Relative to full DCT, maximum cellwise candidate-field relative and metric absolute differences reach `0.3996` and `0.4251`.

A fully independent second implementation uses a different SVD driver and independently rebuilds the complete DCT, response matrices, QDEIM anchors, leverage supplements, candidates, physical K1 replay, metrics, and strata. All `59/59` checks pass. Anchor, supplement, and final selected-column rosters are exact between implementations. Maximum candidate-field relative and metric absolute differences are `3.11e-11` and `6.15e-12`. Camera reordering leaves the selection unchanged, and fixed-observation truth mutation does not change candidates.

## Scientific conclusion

The decision is `FAIL_GEOMETRY_QDEIM1280_CORESET_CAPACITY_V190`.

Under this frozen opened-data four-frame capacity diagnostic, the fixed geometry-only `1280`-column QDEIM plus leverage subset preserves algebraic response rank but not the held-out physical inverse or matched-accuracy tails. Close this fixed `1280`-column family without increasing its budget, retuning thresholds, adding post-result weights, or rescuing it with a larger model.

This does not prove every compact representation impossible and does not invalidate v189's full-basis capacity. v190 establishes no trained predictor, complete-trajectory accuracy, exact-call reduction, fresh wall/RSS benefit, external condition, curved-ray validation, or real-BOST evidence. The next eligible question is to use the sealed result to distinguish fixed-subset conditioning loss from a need for observation-adaptive coordinates before preregistering a physically different representation.

`algorithm_breakthrough=false`, `paper_success=false`, and neither GPU rental nor neural training is authorized.
