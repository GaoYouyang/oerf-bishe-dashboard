# v264：固定半射线分层修正通过 Case 19 首帧完整门

## 为什么做

v263.1 已经证明，九个既有低成本候选即使让真值逐套相机挑最好，完整门仍是 `0/13`。v264 因此不再组合旧候选，而是结果前冻结一个物理上不同的确定性修正：从已封存的 v258 场与部署可见残差出发，在九台相机、两个位移分量中固定选择探测器棋盘偶相位的一半射线，用一次所选射线的精确伴随、geometry-Jacobi 逆和一次所选射线的精确前向做标量最小二乘修正。

射线选择、相位、比例、归一化和步长公式都在结果前固定；不读取真值、时间或 rig 标签，不搜索奇相位、射线比例、权重、阻尼或 fallback。相机换序不改变选中物理射线和结果。

## 独立结果

完全独立第二实现从原始已开封三维场、几何和 v258 封存状态重建所选射线算子、修正方向、物理观测和全部指标，`32/32` 项检查全真。在正式与独立两条实现的保守包络下，v264 同时通过绝对门与 K16-matched 门 `13/13`；v258 和 Zero-PCGLS K15 的完整门均为 `0/13`。

v264 的 field / full-gradient / interior-gradient / observation 指标 p90-higher 为 `0.13418 / 0.25441 / 0.34669 / 0.04924`，对应 K16-matched p90-higher 比为 `0.43560 / 0.45167 / 0.47261 / 0.96580`。四项 matched worst 比为 `0.45581 / 0.45966 / 0.47398 / 1.04717`，仍守住冻结的 worst 容差。

正式与独立候选场最大相对差为 `1.04e-9`，残差按观测归一化后的最大差为 `2.41e-10`，逐指标最大绝对差为 `4.76e-11`。所选射线算子的伴随恒等式误差不超过 `2.79e-15`，完整物理 replay 误差不超过 `9.56e-15`；相机换序后的场和内部状态差均为 `0`。

## 成本与判决边界

候选继承 v258 的 `15A+14A^T`，再增加一对各覆盖一半探测器射线的精确前向/伴随，因此逻辑射线等价账为 `15.5A+14.5A^T`，低于 K16 reference 的 `16A+16A^T`。离线完整观测 replay 不计入部署账。

封存判决是 `POST_OPEN_CASE19_STRATIFIED_RAY_CORRECTION_HEADROOM_V264`。这在 Case 19 首帧上给出一个部署可见、无训练、相机换序等变的确定性候选，并在保守双实现包络中达到 `13/13` 完整门，因此授权下一步只做一次完全相同机制、禁止调参的 Case 19 完整序列门。

它仍不是算法突破或论文成功：Case 19 已经打开，目前只有首帧；没有完整序列、fresh wall/RSS、未打开外部工况、curved ray 或真实 BOST 证据。K16 自身有一套相机未过绝对 interior-gradient 门，因此其角色仅是冻结的 matched 分母；v264 的绝对门与 matched 门均由自身独立满足。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v264: a fixed half-ray stratified correction clears the complete Case 19 frame-zero gate

## Why this experiment

v263.1 established that even truth-aware per-rig selection over nine existing cheap arms reaches only `0/13` joint passes. v264 therefore does not recombine that pool. It preregisters one physically distinct deterministic correction: starting from the sealed v258 field and deployment-visible residual, it selects the even detector-checkerboard phase across all nine cameras and both displacement components, then applies one exact selected-ray adjoint, the geometry-Jacobi inverse, and one exact selected-ray forward with a scalar least-squares step.

Ray selection, phase, fraction, normalization, and step formula are fixed before results. The mechanism reads no truth, time, or rig label and does not search the odd phase, ray fraction, weighting, damping, or fallback. Camera reordering preserves the selected physical rays and the result.

## Independent result

A fully independent second implementation rebuilds the selected-ray operator, correction direction, physical observations, and every metric from the opened raw 3D fields, geometry, and sealed v258 state. All `32/32` checks pass. Under the conservative envelope of the formal and independent implementations, v264 clears both the absolute and K16-matched gates on `13/13` rigs; v258 and Zero-PCGLS K15 each remain at `0/13` joint passes.

The v264 field / full-gradient / interior-gradient / observation p90-higher metrics are `0.13418 / 0.25441 / 0.34669 / 0.04924`. Their K16-matched p90-higher ratios are `0.43560 / 0.45167 / 0.47261 / 0.96580`; matched worst ratios are `0.45581 / 0.45966 / 0.47398 / 1.04717`, all within the frozen worst-case tolerance.

The maximum formal-independent candidate-field relative difference is `1.04e-9`, the maximum observation-normalized residual difference is `2.41e-10`, and the maximum cell-metric absolute difference is `4.76e-11`. Selected-ray adjoint identity error is at most `2.79e-15`, full physical replay error is at most `9.56e-15`, and camera reordering changes neither the field nor the internal state.

## Cost and decision boundary

The candidate inherits v258's `15A+14AT` and adds one exact forward/adjoint pair over half the detector rays. Its logical ray-equivalent ledger is therefore `15.5A+14.5AT`, below the K16 reference at `16A+16AT`. Offline full-observation replay is excluded from the deployment ledger.

The sealed decision is `POST_OPEN_CASE19_STRATIFIED_RAY_CORRECTION_HEADROOM_V264`. This deployment-visible, training-free, camera-permutation-equivariant candidate reaches `13/13` joint passes on Case 19 frame zero under a conservative two-implementation envelope. It authorizes exactly one full-sequence Case 19 gate using the unchanged mechanism with no retuning.

It is not an algorithm breakthrough or paper result. Case 19 is already opened and only frame zero has been tested. There is no full-sequence, fresh wall/RSS, unopened external-condition, curved-ray, or real-BOST evidence. K16 itself misses one absolute interior-gradient gate, so it serves only as the frozen matched denominator; v264 independently satisfies both its own absolute and matched gates.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
