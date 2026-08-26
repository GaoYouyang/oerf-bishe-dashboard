# v258：Krylov 正交补热修正绝对门 13/13，但 K16-matched 为 0/13

## 为什么做

v257 的独立数值合同未闭合后，v258 不再使用离散 support 或局部 mask，而检验一个物理上不同的 solver-native 机制。它保留 zero-start geometry-Jacobi PCGLS K13 的场方向与测量方向，将冻结的线性热修正分解为“已有 Krylov 测量空间可以解释的部分”和其正交补；只把正交补送入最后一轮未修改的 geometry-Jacobi PCGLS K1。

结果前固定了 13 个全秩方向、单位对角缩放、无 rank 截断、无 ridge、无系数裁剪、无搜索、无混合、无回退，以及 `15A+14A^T` 调用账。等成本线性热 control、raw K14 和 K16 reference 使用同一批 13 个已开封 Case 19 九相机首帧。候选不读取 CFD 真值、时间、rig 或失败标签，训练参数为 0。

## 独立闭环有效

正式实现使用对称特征分解，完全独立的第二实现使用逐项 Gram 构造与 Cholesky 求解；两套实现先各自封存物理 replay，再按逐指标较差值形成保守 envelope。独立验证通过 `47/47` 项冻结检查，状态为 `PASS_INDEPENDENT_RECOMPUTATION_CASE19_KRYLOV_COMPLEMENT_HEAT_FRAME_ZERO_V258`。

final field 相对差、normalized residual 差、逐单元指标差和汇总差的最大值分别为 `1.02e-9 / 2.41e-10 / 4.76e-11 / 2.24e-11`。正交补 correction、projection 与审计量的最大相对差为 `2.12e-12 / 1.38e-11 / 1.20e-10`。因此本次是可以解释的科学负结果，不是数值合同不确定。

## 精确负结果

primary 在冻结绝对门上达到 `13/13`，但 K16-matched 只有 `0/13`。field、full-gradient 与 interior-gradient 的 matched p90 比分别为 `0.452 / 0.466 / 0.495`，都低于 `1.05`；唯一系统性阻塞是 observation，matched p90 为 `1.226`，worst 为 `1.367`，均高于冻结 `1.05` 门。

这项正交补确实把等成本线性热 control 的 observation matched p90 从 `3.222` 降到 `1.226`，但“明显改善”不等于“通过”。raw K14 也只有 `7/13` 绝对通过、`0/13` matched。K16 reference 自身为 `13/13` matched；其一个 interior-gradient 绝对尾部使绝对计数为 `12/13`，不改变 matched reference 的定义。

科学判决为 `FAIL_CASE19_KRYLOV_COMPLEMENT_HEAT_FRAME_ZERO_V258`。当前 Krylov-complement heat 路线关闭；不修改投影秩、floor、ridge、热扩散日程、深度、混合或回退，不运行完整 429 单元序列，也不训练模型或租 GPU。

primary 的名义单帧账为 `15A+14A^T`，K16 为 `16A+16A^T`。由于 matched-accuracy 失败且只运行首帧，这不是有效 exact-call 减少，不授权 wall/RSS 或外部门。这个负结果不关闭整条 C 路线。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v258: Krylov-complement heat correction clears 13/13 absolute gates but 0/13 K16-matched gates

## Motivation

After v257 fails to close its independent numerical contract, v258 leaves discrete support selection and local masks behind and tests a physically distinct solver-native mechanism. It retains the field and measurement directions from zero-start geometry-Jacobi PCGLS K13, decomposes the frozen linear-heat correction into the component explained by the existing Krylov measurement space and its orthogonal complement, and sends only that complement into one final unchanged geometry-Jacobi PCGLS K1 step.

The 13 full-rank directions, unit-diagonal scaling, absence of rank truncation, ridge, coefficient clipping, search, blending, or fallback, and the `15A+14A^T` ledger were fixed before results. An equal-call linear-heat control, raw K14, and the K16 reference use the same 13 opened Case 19 nine-camera frame-zero cells. The candidate reads no CFD truth, time, rig, or failure label and has zero trainable parameters.

## Valid independent closure

The formal implementation uses a symmetric eigendecomposition, while the fully independent implementation builds the Gram system entry by entry and solves it through Cholesky. Each implementation seals its physical replay before a conservative per-metric worst envelope is formed. Independent validation passes all `47/47` frozen checks with status `PASS_INDEPENDENT_RECOMPUTATION_CASE19_KRYLOV_COMPLEMENT_HEAT_FRAME_ZERO_V258`.

Maximum disagreements for final field, normalized residual, cell metric, and summary are `1.02e-9 / 2.41e-10 / 4.76e-11 / 2.24e-11`. Maximum relative disagreements for the complement correction, projection, and audit quantity are `2.12e-12 / 1.38e-11 / 1.20e-10`. This makes the outcome an interpretable scientific negative rather than an inconclusive numerical contract.

## Exact negative result

The primary clears the frozen absolute gate on `13/13` cells but reaches only `0/13` under K16-matched accuracy. Matched p90 ratios for field, full gradient, and interior gradient are `0.452 / 0.466 / 0.495`, all below `1.05`. Observation is the sole systematic blocker: its matched p90 is `1.226` and its worst ratio is `1.367`, both above the frozen `1.05` limit.

The complement does reduce the equal-call linear-heat control's observation matched p90 from `3.222` to `1.226`, but a clear improvement is not a pass. Raw K14 reaches only `7/13` absolute and `0/13` matched cells. The K16 reference is `13/13` matched; one absolute interior-gradient tail leaves its absolute count at `12/13` without changing the matched-reference definition.

The scientific decision is `FAIL_CASE19_KRYLOV_COMPLEMENT_HEAT_FRAME_ZERO_V258`. The Krylov-complement heat route closes without changing projection rank, floor, ridge, heat schedule, depth, blending, or fallback. The full 429-cell sequence is not run, and neither model training nor GPU rental is authorized.

The nominal primary frame ledger is `15A+14A^T`, versus `16A+16A^T` for K16. Because matched accuracy fails and only frame zero is covered, this is not effective exact-call reduction and does not authorize wall/RSS or an external gate. This negative result does not close the broader C route.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.
