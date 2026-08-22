# v192：观测自适应选列有实质改善，但固定 1280 坐标容量仍未过门

## 讲人话结论

v191.1 已经查明：固定 geometry-only 子集失败，是因为不同帧观测会激活不同的正规方程方向。v192 随即做了最小、结果不可见的验证：保留固定 `1009` 个 QDEIM 锚点，再从当前部署可见观测与报告几何中，按每列对**完整正规方程缺陷的贡献**补选 `271` 列，总预算仍为 `1280`。选列不读取目标真值，不搜索预算、评分、ridge 或回退；之后只运行一轮未修改的精确 CGLS。

这个评分确实比旧固定子集更好。五相机严格安全单元从 v190 的 `35/52` 提高到 `40/52`，九相机从 `30/52` 提高到 `40/52`。只按观测幅值补列的便宜控制仅为 `32/52` 与 `26/52`，因此改善不能用“挑幅值最大的列”解释。

但冻结通过门是两档相机都必须 `52/52`。primary 在五相机和九相机下仍各有 `12` 个失败单元，四个时间层没有一个完整通过。五相机失败主要来自梯度：`12` 个失败中有 `10` 个梯度越线、`5` 个观测越线，部分单元同时越线；九相机的 `12` 个失败全部只失 observation。两档 field 都没有单元失败。

独立第二实现重新构造选列评分、排序、候选场、未修改 K1、逐单元指标、调用账和相机换序审计，`17/17` 检查全真。普通数组最大相对差为 `1.74e-10`，近零数组最大绝对差为 `1.96e-14`，相机换序后的特征、响应和离散选择完全一致。

正式判决为 `FAIL_NORMAL_CONTRIBUTION_OBSERVATION_ADAPTIVE_QDEIM_CAPACITY_V192`。

这条结果同时包含一个正增量和一个负结论：

- 正增量：v191.1 的归因得到机制验证，观测自适应正规缺陷评分确实能救回一部分固定子集失败。
- 负结论：当前 `1009 + 271` 固定预算和这一条评分仍不足以保住完整物理容量，不能进入预测器、资源或外部门。

因此关闭这一精确机制，不提高预算、不事后添加或调整评分、不用 CNN/FNO/UNO/DeepONet 或 GPU 挽救。后续只有物理上真正不同的结果不可见机制，或新的成对真实二维双分量 BOST 位移数据，才值得继续。

`algorithm_breakthrough=false`、`paper_success=false`、`exact_call_reduction=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

# v192: Observation-adaptive selection helps materially, but the fixed 1280-coordinate capacity still fails

## Plain-language conclusion

v191.1 establishes that a fixed geometry-only subset fails because different frame observations activate different normal-equation directions. v192 tests the smallest result-blind response to that diagnosis. It keeps the fixed `1009` QDEIM anchors and supplements exactly `271` columns according to each column's contribution to the **full normal-equation defect**, computed only from the current deployment-visible observation and reported geometry. The total budget remains `1280`. Selection reads no target truth and performs no budget, score, ridge, or fallback search; one unchanged exact CGLS step follows.

The score is genuinely better than the fixed subset. Strict-safe cells rise from v190's `35/52` to `40/52` under five cameras and from `30/52` to `40/52` under all nine. A cheap observation-magnitude supplement reaches only `32/52` and `26/52`, so the gain is not explained by selecting the largest-amplitude columns.

The frozen gate, however, requires `52/52` in both sensor arms. The primary still leaves `12` failed cells under five cameras and `12` under all nine, with no complete time stratum. Five-camera failure is gradient-dominated: among the 12 failed cells, 10 violate gradient and 5 violate observation, with overlap. All 12 all-nine failures are observation-only. No field cell fails in either arm.

A fully independent second implementation rebuilds the score, ranking, candidate fields, unchanged K1 replay, cell metrics, call ledger, and camera-permutation audit. All `17/17` checks pass. The maximum ordinary-array relative difference is `1.74e-10`; the maximum near-zero-array absolute difference is `1.96e-14`; camera reordering leaves features, responses, and discrete selection unchanged.

Decision: `FAIL_NORMAL_CONTRIBUTION_OBSERVATION_ADAPTIVE_QDEIM_CAPACITY_V192`.

The result contains both a positive increment and a negative gate:

- Positive increment: the v191.1 attribution receives mechanism-level support; an observation-adaptive normal-defect score rescues part of the fixed-subset failure.
- Negative gate: the current `1009 + 271` budget and exact score remain insufficient to preserve complete physical capacity, so no predictor, resource, or external gate is authorized.

Close this exact mechanism without increasing the budget, adding or tuning scores post hoc, or using CNN/FNO/UNO/DeepONet or GPU scale as rescue. Only a physically distinct result-blind mechanism or genuinely new paired real two-component BOS displacement data can justify continuing.

`algorithm_breakthrough=false`, `paper_success=false`, `exact_call_reduction=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
