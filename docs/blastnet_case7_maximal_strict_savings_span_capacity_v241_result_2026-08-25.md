# v241：K14 当前 Krylov 空间恢复 Case 7 的必要容量，但还不是可部署算法

## 为什么做

v240 已证明，更新前 FIFO16 cache 加当前 K1 方向的完整 rank-17 空间，在已开封 BLASTNet Case 7 的 533 个后续帧上没有一个满足必要容量门。这个结果关闭了 K1 空间，却没有回答：在仍然严格少于 K16 的 `A` 和 `A^T` 调用时，加入当前样本自身更多 Krylov 方向能否补回缺失结构。

v241 选择 K14，因为它是同时严格节省两类精确调用的最大深度。每个后续帧 K14 为 `15A+14A^T`，K16 为 `16A+16A^T`；K15 已是 `16A+15A^T`，不再严格节省 `A`。

## 实际做了什么

对同一 13 条 rig、每条 42 帧和同一冻结 FIFO16 cache，后续每帧运行当前 geometry-Jacobi PCGLS 到 K14，构造 `[U,p1,...,p14]` 的 rank-30 空间。和 v240 一样，field、完整梯度、内部梯度和 observation 分别使用自己的真值可见最优系数。

这是一项故意偏宽松的必要容量审计。四个指标可以使用四组不同系数，所以结果不代表存在一个同时通过四门的三维场，更不是部署时可获得的系数规则。

## 结果

K14 空间达到 **533/533** 个必要安全后续帧和 **13/13** 条完整 rig。四个指标的失败数均为 **0/533**。

| 指标 | p50 | p90-higher | worst | 冻结绝对门 |
| --- | ---: | ---: | ---: | ---: |
| Field | 0.213796 | 0.259846 | 0.283549 | 0.500000 |
| Full gradient | 0.375984 | 0.407378 | 0.443281 | 0.750000 |
| Interior gradient | 0.437416 | 0.483680 | 0.527713 | 0.750000 |
| Observation | 0.029733 | 0.036932 | 0.043547 | 0.200000 |

全部 **2,132** 个逐指标设计在正式与独立实现中数值秩都恰为 **30**。K14 对 K1 的逐指标最小值没有任何正向嵌套违例。

## 第一次独立验证为什么保持 inconclusive

第一次独立实现用 pivoted QR 重算 K14，而正式实现用 SVD。两者的 K14 最小指标和汇总最大差仅为 **3.71e-11 / 1.26e-12**；K1 最小指标最大差为 **1.78e-15**，正式和独立 K1 对封存 v240 的最大差分别为 **2.05e-14 / 1.93e-14**。方向投影直接重放、调用账、cache 物理检查、相机换序和伴随检查也都通过。

但第一次验证仍诚实保留为 `INCONCLUSIVE`，因为两个额外数值诊断使用了不合适的门：一个把通过逆 Jacobi 对角重建的 K1 场当成核心科学数组比较，另一个把有限精度投影方向要求到接近精确算术的正交阈值。二者都没有参与四指标最小值、秩、嵌套或调用账判决。

v241.1 只重新打开封存数组，不重放物理、不读新工况、不写新科学数组，也不改变任何精度、秩或正式-独立容差。它独立重算全部 533 个单元、13 条 rig、K1 父对照、秩、嵌套、调用账与物理检查，最终 **35/35** 项通过。第一次 inconclusive 记录继续保留。

## 调用账与科学边界

每条完整序列中，K14 的逻辑账为 `631A+590A^T`，K16 为 `672A+672A^T`，总调用名义少 **9.1518%**。这是容量假设下的理论账，不是有效调用节省：当前还没有一个联合系数向量，更没有 observation-only 预测器、fresh wall time 或 whole-pipeline RSS 结果。

精确判决是 `POST_OPEN_CASE7_MAXIMAL_STRICT_SAVINGS_SPAN_NECESSARY_HEADROOM_V241`。它推翻了“同一 cache 下所有严格节省调用的当前 Krylov 空间都缺少必要方向”这个悲观解释，并把下一门收缩为实际未修改 K14 PCGLS 系数及同价或更便宜 controls。它不是算法突破、论文成功、外部泛化、curved ray 或真实 BOST 证据。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v241: the current K14 Krylov span restores necessary Case 7 capacity, but is not yet a deployable algorithm

## Why this audit was run

v240 shows that the complete rank-17 span of the pre-update FIFO16 cache plus the current K1 direction cannot satisfy the necessary capacity gate on any of the 533 later frames in opened BLASTNet Case 7. That closes the K1 span, but leaves a sharper question: can more current-sample Krylov directions recover the missing structure while still using strictly fewer exact `A` and `A^T` calls than K16?

v241 selects K14 because it is the maximal depth that strictly saves both operator types. A later K14 frame uses `15A+14A^T`, versus `16A+16A^T` for K16. K15 already uses `16A+15A^T`, so it no longer strictly saves `A`.

## What was executed

For the same 13 rigs, 42 frames per rig, and frozen FIFO16 cache, each later frame runs current geometry-Jacobi PCGLS through K14 and forms the rank-30 span `[U,p1,...,p14]`. As in v240, field, full-gradient, interior-gradient, and observation each receive their own truth-aware optimal coefficients.

This is deliberately an optimistic necessary-capacity audit. The four metrics may use four different coefficient vectors, so the result does not show that one 3D field passes all four gates, and it does not supply coefficients available at deployment.

## Result

The K14 span reaches **533/533** necessary-safe later frames and **13/13** complete rigs. All four metric-specific failure counts are **0/533**.

| Metric | p50 | p90-higher | worst | Frozen absolute limit |
| --- | ---: | ---: | ---: | ---: |
| Field | 0.213796 | 0.259846 | 0.283549 | 0.500000 |
| Full gradient | 0.375984 | 0.407378 | 0.443281 | 0.750000 |
| Interior gradient | 0.437416 | 0.483680 | 0.527713 | 0.750000 |
| Observation | 0.029733 | 0.036932 | 0.043547 | 0.200000 |

All **2,132** metric designs have numerical rank **30** in both implementations. Neither implementation shows a positive K14-versus-K1 nesting violation.

## Why the first independent validation remained inconclusive

The independent implementation uses pivoted QR while the formal implementation uses SVD. Their maximum K14 metric-minimum and summary differences are only **3.71e-11 / 1.26e-12**. The K1 metric-minimum difference is **1.78e-15**, and formal / independent K1 differ from sealed v240 by at most **2.05e-14 / 1.93e-14**. Direct projection replay, call ledgers, cache physics, camera permutation, and adjoint checks also pass.

The first validation nevertheless remains recorded as `INCONCLUSIVE` because two additional numerical diagnostics used unsuitable thresholds: one treated a K1 field reconstructed through the inverse Jacobi diagonal as a core science-array comparison, and one required finite-precision projected directions to meet a near-exact-arithmetic orthogonality bound. Neither diagnostic enters the four metric minima, ranks, nesting, or call ledger.

v241.1 reopens only sealed arrays. It does not replay physics, open a new condition, write new science arrays, or alter accuracy, rank, or formal-independent tolerances. It independently recomputes all 533 cells, all 13 rigs, the K1 parent, ranks, nesting, call ledgers, and physical checks, and passes **35/35** checks. The original inconclusive record remains preserved.

## Call ledger and scientific boundary

Across a complete sequence, K14 has a logical ledger of `631A+590A^T`, versus `672A+672A^T` for K16, a nominal **9.1518%** reduction in total exact calls. This is a capacity-conditional theoretical ledger, not an effective saving. There is still no jointly feasible coefficient vector, observation-only predictor, fresh-process wall-time result, or whole-pipeline RSS result.

The exact decision is `POST_OPEN_CASE7_MAXIMAL_STRICT_SAVINGS_SPAN_NECESSARY_HEADROOM_V241`. It refutes the pessimistic claim that every same-cache current Krylov span with strict savings lacks necessary directions. The next gate is therefore the actual unchanged K14 PCGLS coefficients against equal-or-cheaper controls. This is not an algorithm breakthrough, paper success, external generalization, curved-ray validation, or real-BOST evidence.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
