# v186.1：DCT12 + Plucker 共享线性近似未通过完整轨迹门

## 做了什么

v185 证明，如果对当前观测和全部 1,009 个仿射响应列使用同一 detector-potential 变换，稠密势域逆可以保留全部可观测信息。v186.1 检验的不是再做一次稠密逆，而是它能否被一个部署时只看当前观测与报告几何的共享线性 set map 紧凑近似。

冻结表示为：

1. 对每个有效相机的零均值 detector potential 做正交 DCT，保留固定 `12x12` 低频方块中的 143 个非 DC 系数；
2. 用常数、相机数倒数、报告射线方向和 Plucker 线矩组成 8 维相机描述；
3. 对相机集合求和，得到 1,144 维、相机换序不变的特征；
4. 只用 10 条已打开 fit 轨迹的 1,010 个三维场闭式拟合共享线性权重，不使用 held-out 真值调参、回退或停止；
5. 在已开封 p22 的 4 个时间层、13 套标定上，比较直接 K0 和一次未修改物理 CGLS K1，同时保留五相机与九相机两臂。

## 独立复算后的结果

初始完整空间 stationarity 检查与已冻结的截断伪逆求解定义不一致，v186.1 在查看 held-out 数字前把检查修正为保留特征子空间上的投影 stationarity，科学数据、表示、权重、阈值和决策规则均未改变。投影 stationarity 为 `6.60e-14`；被截断的 3 个特征方向仅携带 `1.88e-6` 的 target cross fraction。

但科学结果是负的：

| primary arm | field p90 | gradient p90 | observation p90 | 严格通过 | 完整标定 | 完整时间层 |
|---|---:|---:|---:|---:|---:|---:|
| 五相机 K0 | 0.334784 | 0.552997 | 0.415311 | 0/52 | 0/13 | 0/4 |
| 九相机 K0 | 0.337024 | 0.547091 | 0.417372 | 0/52 | 0/13 | 0/4 |
| 五相机 K1 | 0.305891 | 0.484862 | 0.215971 | 39/52 | 7/13 | 0/4 |
| 九相机 K1 | 0.284107 | 0.449993 | 0.235876 | 25/52 | 1/13 | 0/4 |

一次 K1 明显改善了结果，且 field 与 gradient 尾部已在冻结门内；但完整轨迹门要求每个时间层和每套标定都安全，不能只看合并 p90。五相机 K1 的四个 observation p90 为 `0.232724 / 0.200616 / 0.216058 / 0.199495`；九相机为 `0.235876 / 0.213836 / 0.253397 / 0.210684`，冻结门是 `0.20`。因此两臂均为 `0/4` 完整时间层。

几何盲 DCT12、一方向 potential-coordinate CGLS1 和 fit mean 三组更便宜 control 在 K1 下的两臂仍全部为 `0/52`；它们没有解释出一个可通过的简单替代，但也不能把失败的 primary 变成正结果。

完全独立第二实现重建 fit-only 仿射基、势场、DCT/Plucker 特征、共享线性求解、held-out 预测、物理 K1、指标、调用账和相机换序审计。`44/44` 项检查全真；候选场、坐标与逐单元指标最大差约为 `5.65e-10 / 9.04e-10 / 5.89e-11`，离散判决完全一致。相机换序和 held-out truth mutation 对 primary 预测的影响均为 `0`。

## 科学结论

正式判决是 `FAIL_POTENTIAL_SET_LINEAR_V186_1_1`。

这个结果说明：**v185 证明的稠密势域信息容量，不能由当前固定 DCT12 + 报告射线 Plucker pooling + 共享线性映射稳定复现。** 它能学到有用的 field/gradient 结构，但 observation 尾部仍不兼容；因此不允许用合并平均数包装成通过。

当前 DCT12 + Plucker 共享线性表示就此关闭，不扩大特征、不换更大 CNN/FNO/UNO 挽救，不租 GPU。这不是对整条 C 路线的否定，也不是数学不可能性证明。K1 逻辑在线账为 `2A+1A^T`，但因为 accuracy 门失败，不能声称 exact-call 减少，也不授权 wall/RSS、外部工况或真实 BOST 结论。`algorithm_breakthrough=false`。

# v186.1: the shared DCT12 + Plucker linear approximation fails the complete-trajectory gate

## What was tested

v185 showed that a dense potential-domain inverse can preserve all observable information when the current observation and all 1,009 affine-response columns undergo the same detector-potential transform. v186.1 tests whether that dense action can be approximated by one shared linear set map that reads only the current observation and reported geometry at deployment.

The frozen representation:

1. applies an orthonormal DCT to each active camera's zero-mean detector potential and retains the 143 non-DC modes in a fixed `12x12` low-frequency square;
2. forms an eight-value camera descriptor from constants, inverse cardinality, reported-ray direction, and Plucker line moment;
3. sums over the camera set to obtain a 1,144-dimensional camera-permutation-invariant feature;
4. fits shared linear weights in closed form using only 1,010 3D fields from ten opened fit trajectories, with no heldout-truth tuning, fallback, or stopping;
5. evaluates direct K0 and one unchanged physical CGLS K1 step on four opened p22 times and 13 calibrations under five- and nine-camera arms.

## Independently recomputed result

The initial full-space stationarity check was inconsistent with the already frozen truncated pseudoinverse. Before any heldout values were inspected, v186.1 corrected the diagnostic to projected stationarity on the retained eigensubspace without changing the data, representation, weights, thresholds, or decision rule. Projected stationarity is `6.60e-14`; the three discarded directions carry only `1.88e-6` of the target cross fraction.

The scientific result is nevertheless negative:

| primary arm | field p90 | gradient p90 | observation p90 | strict-safe | complete calibrations | complete times |
|---|---:|---:|---:|---:|---:|---:|
| Five-camera K0 | 0.334784 | 0.552997 | 0.415311 | 0/52 | 0/13 | 0/4 |
| All-nine K0 | 0.337024 | 0.547091 | 0.417372 | 0/52 | 0/13 | 0/4 |
| Five-camera K1 | 0.305891 | 0.484862 | 0.215971 | 39/52 | 7/13 | 0/4 |
| All-nine K1 | 0.284107 | 0.449993 | 0.235876 | 25/52 | 1/13 | 0/4 |

One K1 step improves the candidates substantially, and their field and gradient tails lie inside the frozen limits. The complete-trajectory contract, however, requires every time and calibration stratum to remain safe rather than accepting only pooled p90 values. Five-camera K1 observation p90 by time is `0.232724 / 0.200616 / 0.216058 / 0.199495`; all-nine is `0.235876 / 0.213836 / 0.253397 / 0.210684`, against a frozen limit of `0.20`. Neither arm therefore clears any complete time stratum.

The cheaper geometry-blind DCT12, one-direction potential-coordinate CGLS1, and fit-mean controls all remain at `0/52` in both K1 arms. They do not provide a passing simpler explanation, but they also cannot turn a failing primary into a positive result.

A fully independent second implementation rebuilds the fit-only affine basis, potentials, DCT/Plucker features, shared linear solve, heldout predictions, physical K1, metrics, call ledgers, and camera-permutation audit. All `44/44` checks pass. Maximum candidate-field, coordinate, and per-cell metric differences are about `5.65e-10`, `9.04e-10`, and `5.89e-11`, with identical discrete decisions. Camera reordering and heldout-truth mutation change the primary prediction by exactly `0`.

## Scientific conclusion

The formal verdict is `FAIL_POTENTIAL_SET_LINEAR_V186_1_1`.

The dense potential-domain capacity established by v185 is **not stably reproduced by the current fixed DCT12 + reported-ray Plucker pooling + shared linear map**. The map recovers useful field and gradient structure, but its observation tails remain incompatible with the frozen gate, so pooled averages cannot be presented as a pass.

The current DCT12 + Plucker shared-linear representation is closed. It will not be enlarged or replaced by a larger CNN/FNO/UNO as a rescue, and GPU rental remains unauthorized. This does not close the overall C route and is not a mathematical impossibility result. Although the logical online K1 ledger is `2A+1A^T`, the failed accuracy gate means no exact-call reduction, wall/RSS, external-condition, or real-BOST claim is established. `algorithm_breakthrough=false`.
