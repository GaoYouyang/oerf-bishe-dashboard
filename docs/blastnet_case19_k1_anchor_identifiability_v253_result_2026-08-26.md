# v253：K1 残差收缩选锚未识别出独立信号，当前锚点假设关闭

## 为什么做

v252 的事后诊断显示，观测图遍历的两个绝对门缺口都发生在图 medoid 锚点本身。v253 因此只问一个更窄、可证伪的问题：在已经开封的 Case 19 中，能否只用部署可见的二维观测与 reported geometry，对每帧执行一次零起点 geometry-Jacobi PCGLS K1，并用归一化残差收缩选择一个安全锚点？

结果前合同固定了 13 套 rig × 33 帧、唯一 K1 分数、确定性 tie-break、四指标绝对门和 K16 双实现稳健安全标签。便宜对照是最小观测范数与余弦 medoid；固定 frame 16 只作为不合格的时间索引诊断。锚点封存前不读取真值、时间或 rig 标签。只审计选锚，不运行完整遍历，也不训练模型。

## 独立复算通过，科学结论为负

独立第二实现通过 `16/16` 项检查。formal 与 independent 的分数最大相对差为 `6.10e-16`，相机乱序后的最大相对差为 `3.07e-16`；选中锚点、稳健安全标签和调用账完全一致。因此这次负结论不是工程不确定，而是可复算的科学 FAIL。

K1 残差收缩 primary 在 13 套 rig 中全部选择 frame 3，只得到 `9/13` 条安全 rig；不安全的是 rig 0、5、9、12。更关键的是，零精确调用的最小观测范数 control 也在每套 rig 选择 frame 3，并得到完全相同的 `9/13` 安全 roster。K1 分数没有隔离出 solver-specific 的锚点信息。

余弦 medoid control 达到 `11/13`，仍未通过完整 rig 门。固定中点 frame 16 达到 `13/13`，但它依赖时间索引，不是 camera-permutation-equivariant、variable-cardinality 部署条件下可接受的 observation/geometry-only 选择器，不能用于宣称成功或事后替换 primary。

## 成本与路线动作

K1 primary 的选锚筛查每套 rig 需要 `33A+33A^T`；两个便宜 control 都是 `0A+0A^T`。即使假设选中的 K1 状态可在未来完整序列中复用，名义账也只是 `528A+496A^T`，相对 K16 的 `528A+528A^T` 仅有 `3.0303%` 算术差。由于 primary 未通过 13/13、又被零调用 control 完全解释，完整遍历、wall/RSS 与资源门均不授权，这个百分比不是有效减调用或速度结果。

正式判决是 `FAIL_CASE19_K1_RESIDUAL_CONTRACTION_ANCHOR_V253`。当前 K1 残差收缩选锚假设关闭，不再在 Case 19 上增加其他 anchor heuristic、学习型排序、大模型或 GPU。它不关闭整条 C 路线，也不证明所有 solver-state、geometry-local 或 nonlinear 机制不可能；下一步只能来自物理上真正不同、结果前唯一冻结的机制，或新的配对真实 BOST 信息。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v253: K1 residual-contraction anchoring does not isolate an independent signal

## Motivation

The post-open v252 diagnostic places both absolute graph-traversal misses at the medoid anchors themselves. v253 therefore asks a narrower falsifiable question on the already-opened Case 19: can one zero-start geometry-Jacobi PCGLS K1 step per frame, using only deployment-visible 2D observations and reported geometry, identify a safe batch anchor through normalized residual contraction?

Before execution, the contract fixes 13 rigs by 33 frames, one K1 score, deterministic tie-breaking, the four absolute metric limits, and a robust K16 safety label formed from two sealed implementations. Cheap controls are minimum observation norm and cosine medoid. Fixed frame 16 is only a time-index-dependent inadmissible diagnostic. Truth, time, and rig labels are unavailable before the anchor barrier. The audit selects anchors only; it does not execute a complete traversal or train a model.

## Independent recomputation passes; the scientific result is negative

The independent second implementation passes `16/16` checks. Maximum formal-independent score disagreement is `6.10e-16`, and maximum camera-permutation disagreement is `3.07e-16`; selected anchors, robust safety labels, and call ledgers agree exactly. This is therefore a reproducible negative scientific result rather than an engineering inconclusive result.

The K1 residual-contraction primary selects frame 3 in every rig and is safe on only `9/13` rigs, missing rigs 0, 5, 9, and 12. More importantly, the zero-exact-call minimum-observation-norm control also selects frame 3 in every rig and produces exactly the same `9/13` safety roster. The K1 score does not isolate solver-specific anchor information.

The cosine-medoid control reaches `11/13` and still misses the complete-rig gate. Fixed midpoint frame 16 reaches `13/13`, but it depends on the time index and is not an admissible observation/geometry-only selector under camera-permutation-equivariant, variable-cardinality deployment. It cannot be used to claim success or replace the primary after observing results.

## Cost and route action

Screening the K1 primary costs `33A+33A^T` per rig; both cheap controls cost `0A+0A^T`. Even if the selected K1 state were reused in a hypothetical future sequence, the nominal ledger would be `528A+496A^T` versus `528A+528A^T` for K16, only a `3.0303%` arithmetic difference. Because the primary misses 13/13 and is completely explained by a zero-call control, complete traversal, wall/RSS, and resource gates are not authorized. This percentage is not effective call reduction or a speed result.

The formal decision is `FAIL_CASE19_K1_RESIDUAL_CONTRACTION_ANCHOR_V253`. The K1 residual-contraction anchor hypothesis closes, with no further Case 19 anchor heuristics, learned ordering, larger model, or GPU run. This does not close the C route or prove that all solver-state, geometry-local, or nonlinear mechanisms are impossible. Any next step must be physically distinct and uniquely preregistered, or use new paired real-BOST information.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.
