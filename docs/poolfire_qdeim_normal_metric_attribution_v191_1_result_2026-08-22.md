# v191.1：固定 QDEIM 子集改变了观测激活的正规度量

## 为什么继续查

v190 已经确认：固定 `1280` 列的 geometry-only QDEIM + 杠杆子集虽然保留 `1009/1009` 响应秩，却没有保住完整 DCT 的物理容量。五/九相机的未修改 K1 只达到 `35/52` 和 `30/52`，两臂完整时间层均为 `0/4`。

但 v190 还没有回答一个关键机制问题：失败是否只是某些相机标定整体条件差，还是同一标定下，不同观测会激活不同的最小二乘方向，使固定子集在某些帧有效、另一些帧失效？v191 只做归因，不构造新算法，也不训练模型。

## 怎么检验

沿用 v190 已开封且封存的四帧、13 套标定以及五/九相机两臂，共 `26` 个固定 sensor/calibration setup。对每个 setup 比较完整 DCT 与固定子集的正规方程响应，并检查：

1. 同一 setup 的四帧是否同时通过、同时失败，或出现混合成败；
2. 子集解与完整 DCT 解的坐标差异；
3. 子集解对完整正规方程的缺陷；
4. 坐标误差经完整观测响应后，有多少能量落在被固定子集丢弃的列上；
5. 两种度量的广义特征值、方向权重和条件数膨胀。

这是一条 post-open retrospective diagnostic：所有 v190 物理候选和成败已知，只允许定位机制，不能把它写成未见数据泛化、部署算法或速度结果。

## 结果

固定几何并不能固定成败。五相机 `13` 个 setup 中有 `10` 个在四帧内混合成败，九相机有 `11` 个；总计 `21/26`。五相机各 setup 的通过帧数为：

`3, 3, 1, 4, 3, 2, 0, 3, 4, 3, 3, 3, 3`

九相机为：

`3, 3, 2, 2, 3, 2, 1, 3, 0, 4, 3, 3, 1`

因此一个只依赖报告几何的全局条件数或单一难度分数，不足以解释同一 setup 内为什么有的帧过门、有的帧失败。

对 v190 的全部失败单元，子集与完整 DCT 的坐标差异都高于 `1e-8`，完整正规方程缺陷也都高于 `1e-8`。坐标差异中位数在五/九相机下为 `45.93% / 42.96%`；其中落在被固定子集丢弃列上的观测响应能量中位数高达 `90.82% / 93.50%`。完整度量相对所选子集的 trace-normalized 方向权重中位数只有 `10.66% / 8.70%`，条件数膨胀中位数为 `3.69x / 4.95x`。广义度量特征值跨越约 `0.0053–1.3910`。

同时，所选子集解对自己的目标已经充分收敛：formal 和独立实现的最大 stationarity residual 分别约为 `1.39e-14` 与 `3.73e-12`。这说明问题不是优化器没跑稳，而是固定子集本身改变了观测激活的最小二乘度量。

## 独立复算与 v191.1 修复边界

独立程序采用不同的 SVD driver 和广义特征值形式，完成全部 `13/13` 标定重建；父响应差约为 `1e-12`，所有离散归因谓词完全一致。原始 v191 验证仍然被保留为 `INCONCLUSIVE`，原因是它对所有数组统一使用相对误差门，把已经接近数值零的 stationarity 与能量恒等式残差也做了除法放大。

原始 formal/independent stationarity 数组的最大绝对差只有 `3.73e-12`，能量恒等式最大绝对差只有 `2.32e-15`；其他普通数组的最大相对差为 `1.67e-11`。v191.1 在看到该失败后透明冻结了一次静态比较器修复：普通数组仍保留 `1e-8` 相对门，近零残差改用预先写明的绝对数值门；不重跑物理、不修改数组、不重置一次性验证。修复后 `15/15` 检查通过。

这项修复属于数值审计完整性，不是算法成果；原始 inconclusive 证据没有被删除或改写。

## 科学结论

正式判决为 `PASS_OBSERVATION_ACTIVATED_NORMAL_METRIC_DISTORTION_ATTRIBUTION_V191_1`。

在这条已开封四帧诊断中，v190 固定子集的失败可以归因到 **observation-activated normal-metric distortion**：同一报告几何下，不同帧激活的最小二乘方向不同；固定选列对自身目标已经收敛，却没有保持完整 DCT 的正规方程与方向权重。约九成坐标误差响应能量位于被固定子集丢弃的列中。

这只授权下一条结果前冻结的最小 observation-adaptive 坐标**容量诊断**。它没有构造或验证 observation-adaptive 表示，没有训练 predictor，没有证明 exact-call 减少、wall/RSS、外部泛化、曲线光路或真实 BOST，也不授权 GPU。

`algorithm_breakthrough=false`，`paper_success=false`。

# v191.1: the fixed QDEIM subset distorts the observation-activated normal metric

## Why this attribution was needed

v190 established that the fixed `1280`-column geometry-only QDEIM-plus-leverage subset retains response rank `1009/1009` but not the physical capacity of the complete DCT. Unchanged K1 reaches only `35/52` and `30/52` cells under five and all-nine cameras, with `0/4` complete time strata in both arms.

v191 asks a narrower mechanism question: is failure explained by a uniformly bad camera calibration, or do different observations under the same reported geometry activate different least-squares directions, making one fixed subset succeed on some frames and fail on others? This is attribution only; it constructs no algorithm and trains no model.

## Frozen diagnostic

The sealed v190 roster is retained: four opened frames, 13 calibrations, and five/all-nine sensor arms, for `26` fixed sensor/calibration setups. Each setup compares complete-DCT and fixed-subset normal-equation responses, coordinate discrepancy, full-normal defect, discarded-column response energy, generalized metric eigenvalues, directional weighting, and condition inflation.

This is a post-open retrospective diagnostic. It may localize the failure mechanism, but it cannot establish unseen-data generalization, deployment, or speed.

## Result

Pass/fail is not fixed by geometry. Ten of 13 five-camera setups and 11 of 13 all-nine setups contain both passing and failing frames: `21/26` mixed setups overall. Every failed v190 cell has coordinate discrepancy and full-normal defect above `1e-8`, and every mixed setup has within-setup coordinate-discrepancy spread above `1e-8`.

Median coordinate discrepancy is `45.93% / 42.96%` under five/all-nine cameras. Median response energy in discarded columns is `90.82% / 93.50%`; median trace-normalized directional weighting is only `10.66% / 8.70%`; median condition inflation is `3.69x / 4.95x`. Generalized metric eigenvalues span approximately `0.0053–1.3910`.

The selected solves are nevertheless stationary for their own reduced objective. Maximum formal and independent stationarity residuals are approximately `1.39e-14` and `3.73e-12`. The failure is therefore not an unconverged optimizer; the fixed subset changes the observation-activated least-squares metric.

## Independent recomputation and the v191.1 repair boundary

An independent implementation uses a different SVD driver and generalized-eigenvalue formulation and completes all `13/13` calibration reconstructions. Parent-response differences are approximately `1e-12`, and every discrete attribution predicate agrees exactly.

The original v191 validation remains preserved as `INCONCLUSIVE`. Its only failed check applied one relative comparator to every array, including stationarity and energy-identity residuals already near numerical zero. Their maximum absolute differences are only `3.73e-12` and `2.32e-15`; ordinary arrays have maximum relative difference `1.67e-11`.

v191.1 transparently freezes one static comparator repair after that failure: ordinary arrays retain the `1e-8` relative gate, while near-zero residual arrays use stated absolute numerical gates. It reruns no physics, changes no array, and does not reset the single-use validation. The repair passes `15/15` checks. This is numerical audit integrity, not an algorithmic gain, and the original inconclusive evidence is not erased.

## Scientific decision

Decision: `PASS_OBSERVATION_ACTIVATED_NORMAL_METRIC_DISTORTION_ATTRIBUTION_V191_1`.

On this opened four-frame diagnostic, v190 failure is attributable to observation-activated normal-metric distortion. Under identical reported geometry, frames activate different least-squares directions; the fixed subset converges for its own objective but does not preserve the complete-DCT normal equations or directional weighting. Roughly nine tenths of coordinate-error response energy lies in discarded columns.

This authorizes only one separately preregistered minimal observation-adaptive coordinate **capacity diagnostic**. No observation-adaptive representation, predictor, exact-call reduction, wall/RSS gain, external generalization, curved-ray validation, real-BOST result, or GPU authorization has been established.

`algorithm_breakthrough=false`, `paper_success=false`.
