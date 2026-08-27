# v270：完整几何 normal-SGS 控制因 K16 reference 不充分而保持不确定

## 讲人话结论

v270 检验了一个和学习模型、低秩修补、局部块修正都不同的经典控制：直接由九相机几何构造 active 区域完整稀疏 normal 矩阵 `A^T A`，按固定体素顺序做对称 Gauss-Seidel 预条件，再运行 Zero-PCGLS K14。它不读真值、不训练参数，也不是 warm initializer。

正式实现通过 `21/21` 项有效性门，完全独立第二实现通过 `32/32` 项。两套实现的场、残差、指标和汇总最大差分别为 `1.31e-10 / 2.41e-10 / 4.76e-11 / 7.91e-10`；二维观测差为 `8.38e-17`，物理重放误差为 `3.52e-16`。因此执行与独立复算本身是成立的。

但结果前冻结的裁决要求 K16 reference 先在 `13/13` 套相机上通过绝对门。它实际只通过 `12/13`：唯一失败的 interior-gradient 为 `0.758223`，高于冻结的 `0.750000` 门 `0.008223`。所以权威判决是 `INCONCLUSIVE_REFERENCE_INADEQUATE_CASE19_GEOMETRY_NORMAL_SGS_FRAME_ZERO_V270`，不能用一个自身不合格的 reference 判定 SGS 是否匹配。

独立复算后的诊断数字仍然很明确：normal-SGS K14 的绝对门和 matched 门均为 `0/13`，field / full-gradient / interior-gradient p90 为 `2.09477 / 3.51715 / 7.48569`；同价 geometry-Jacobi K14 绝对门为 `7/13`，未预条件 CGLS K14 为 `0/13`。这些数字说明当前自然顺序 exact-normal SGS 没有值得推进的迹象，但在冻结判决顺序下只能作为诊断，不能包装成正式算法负判决。

主候选逻辑账为 `14A+14A^T`，reference 为 `16A+16A^T`。完整 normal 每套相机约有 `864-897` 万个非零元，封存状态约 `198-205 MiB`；这些是 setup 披露，不是 whole-pipeline RSS。由于 reference 不充分且 matched-accuracy 未成立，没有有效减调用、wall/RSS、完整序列、外部泛化或真实 BOST 结论。当前执行封存，不调体素顺序、不加松弛/载荷、不加深同一 reference，也不用训练或 GPU 挽救。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v270: Full-geometry normal-SGS control remains inconclusive because the K16 reference is inadequate

## Plain-language conclusion

v270 tests a classical control that is physically distinct from learned models, low-rank repair, and local block corrections. It builds the full sparse active normal matrix `A^T A` from the nine-camera geometry, applies a symmetric Gauss-Seidel preconditioner in a fixed voxel order, and runs zero-start PCGLS K14. It reads no truth for prediction, trains no parameter, and is not a warm initializer.

The formal implementation passes `21/21` validity checks and a fully independent second implementation passes `32/32`. Maximum field, residual, metric, and summary differences are `1.31e-10 / 2.41e-10 / 4.76e-11 / 7.91e-10`; the normalized observation difference is `8.38e-17` and physical-replay error is `3.52e-16`. Execution and independent recomputation are therefore valid.

The preregistered adjudication requires the K16 reference to clear the absolute gate on all `13/13` rigs first. It reaches only `12/13`: one interior-gradient value is `0.758223`, exceeding the frozen `0.750000` limit by `0.008223`. The authoritative verdict is therefore `INCONCLUSIVE_REFERENCE_INADEQUATE_CASE19_GEOMETRY_NORMAL_SGS_FRAME_ZERO_V270`; an inadequate reference cannot adjudicate SGS matched accuracy.

Independently recomputed diagnostics remain informative. Normal-SGS K14 reaches `0/13` absolute and `0/13` matched rigs, with field / full-gradient / interior-gradient p90 values of `2.09477 / 3.51715 / 7.48569`. The equal-cost geometry-Jacobi K14 control reaches `7/13` absolute rigs, while unpreconditioned CGLS K14 reaches `0/13`. These figures give no reason to advance natural-order exact-normal SGS, but the frozen decision order keeps them diagnostic rather than a formal algorithmic failure.

The primary logical ledger is `14A+14AT`, versus `16A+16AT` for the reference. Each full normal matrix contains roughly `8.64-8.97` million nonzeros and stores about `198-205 MiB` of sealed state; this is setup disclosure, not whole-pipeline RSS. With inadequate reference and no matched-accuracy result, there is no effective call reduction, wall/RSS, full-sequence, external-generalization, or real-BOST claim. The execution is sealed without ordering changes, relaxation/loading, deeper reuse of the same reference, training, or GPU rescue.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
