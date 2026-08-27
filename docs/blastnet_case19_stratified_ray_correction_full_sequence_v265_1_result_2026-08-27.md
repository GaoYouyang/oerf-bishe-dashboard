# v265.1：完整序列否定固定半射线修正路线

## 为什么做

v264 在已经打开的 BLASTNet Case 19 首帧上给出正机制证据：固定偶相位半射线修正在十三套相机上都通过绝对与 K16-matched 完整门。v265.1 不改变相位、射线比例、权重、阻尼、阈值或 fallback，把同一部署可见、无训练机制一次性扩展到 `13` 套相机的 `33` 帧完整序列，共 `429` 个单元。

候选仍只读取封存父状态、当前观测残差和报告几何；不读取真值、时间或 rig 标签。逻辑射线等价账保持 `15.5A+14.5A^T`，冻结 K16 reference 为 `16A+16A^T`。

## 独立结果

完全独立第二实现通过 `35/35` 项检查，另有一个不导入正式或验证器实现的数组复算程序确认形状、有限性、计数、汇总和判决。正式与独立两条实现的保守包络中，候选绝对门为 `429/429` 单元、`13/13` 完整轨迹，但 K16-matched 门只有 `200/429` 单元、`0/13` 完整轨迹。

全部 `229` 个 matched 失败都只来自 observation；field、full-gradient 和 interior-gradient 的 matched 失败数均为 `0`。四项 matched p90-higher 比为 `0.49241 / 0.47497 / 0.60289 / 1.23503`，worst 比为 `0.62231 / 0.55141 / 0.75588 / 1.85517`。前三项有明显余量，但 observation 同时越过冻结的 p90 与 worst 门。

作为对照，封存父候选虽然绝对门为 `429/429`，matched 仅 `4/429`、完整轨迹 `0/13`；Zero K15 为绝对 `383/429`、matched `0/429`、完整轨迹 `0/13`。K16 reference 的绝对门为 `417/429`、`9/13`，并作为全部 `429` 个单元的冻结 matched 分母。

正式与独立候选场最大相对差为 `4.86e-9`，观测归一化残差最大差为 `1.10e-9`，逐指标最大绝对差为 `1.47e-10`；算子/局部伴随与完整物理 replay 误差分别不超过 `9.85e-15` 和 `4.21e-15`，相机换序观测差为 `0`。底层几何和密度核仍共享，因此没有证明端到端物理独立。

## 判决边界

封存判决是 `FAIL_CASE19_STRATIFIED_RAY_CORRECTION_FULL_SEQUENCE_V265_1`。首帧的 v264 正机制证据仍是有效历史结果，但它没有延伸到完整序列。固定偶相位半射线修正路线现在关闭，不再调整奇偶相位、比例、深度、权重、阻尼、阈值或 fallback，也不用 CNN、FNO 或 GPU 挽救。

这不是整条 C 路线不可能，也不是算法突破或论文结果。没有 fresh wall/RSS、未打开外部工况、curved ray 或真实 BOST 证据。下一步只接受物理上真正不同、仍只读取部署可见信息的机制，或工况配对的真实二维 BOST 数据。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v265.1: the full sequence rejects the fixed half-ray correction route

## Why this experiment

v264 provided positive mechanism evidence on frame zero of the already-opened BLASTNet Case 19: the fixed even-quincunx half-ray correction cleared both absolute and K16-matched gates on all thirteen rigs. v265.1 changes no phase, ray fraction, weighting, damping, threshold, or fallback. It applies exactly the same deployment-visible, training-free mechanism once to all `33` frames of the `13` rigs, for `429` cells.

The candidate still reads only the sealed parent state, current observation residual, and reported geometry. It reads no truth, time, or rig label. Its logical ray-equivalent ledger remains `15.5A+14.5AT`, against the frozen K16 reference at `16A+16AT`.

## Independent result

A fully independent second implementation passes all `35/35` checks. A separate array-only program importing neither the formal runner nor the validator also confirms shapes, finiteness, counts, summaries, and the decision. Under the conservative envelope of the two implementations, the candidate clears the absolute gate on `429/429` cells and `13/13` complete trajectories, but clears the K16-matched gate on only `200/429` cells and `0/13` complete trajectories.

All `229` matched failures are observation-only; field, full-gradient, and interior-gradient each have zero matched failures. The four matched p90-higher ratios are `0.49241 / 0.47497 / 0.60289 / 1.23503`, and the worst ratios are `0.62231 / 0.55141 / 0.75588 / 1.85517`. The first three metrics retain clear margin, while observation exceeds both frozen p90 and worst limits.

The sealed parent control reaches `429/429` absolute cells but only `4/429` matched cells and `0/13` matched trajectories. Zero K15 reaches `383/429` absolute cells, `0/429` matched cells, and `0/13` matched trajectories. K16 reaches `417/429` absolute cells and `9/13` absolute trajectories, and remains the frozen matched denominator for all `429` cells.

The maximum formal-independent candidate-field relative difference is `4.86e-9`, the maximum observation-normalized residual difference is `1.10e-9`, and the maximum cell-metric absolute difference is `1.47e-10`. Operator/partial-adjoint and full physical-replay errors are at most `9.85e-15` and `4.21e-15`, with zero camera-reordering observation difference. Low-level geometry and density kernels remain shared, so end-to-end physics independence is not established.

## Decision boundary

The sealed decision is `FAIL_CASE19_STRATIFIED_RAY_CORRECTION_FULL_SEQUENCE_V265_1`. The v264 frame-zero result remains valid historical mechanism evidence, but it does not extend to the full sequence. The fixed even-quincunx half-ray route now closes. No parity, fraction, depth, weighting, damping, threshold, or fallback retuning is authorized, and no CNN, FNO, or GPU rescue follows.

This does not prove the entire C route impossible, and it is not an algorithm or paper result. There is no fresh wall/RSS, unopened external-condition, curved-ray, or real-BOST evidence. The next admissible step requires either a genuinely physically distinct deployment-visible mechanism or condition-matched real two-component BOST data.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
