# v235/v235.1：Case 7 前瞻外门否定固定 Direct Low64 K11 迁移

## 结论

v234 在已开封 Case 12 上发现：固定 Direct Low64 warm + 未修改 geometry-Jacobi PCGLS K11 全部通过，而额外 fallback 制造了策略失败。v235 因此在读取新数值前，把这一固定方法单独冻结，并在当时全局未打开的 BLASTNet Case 7 上做一次前瞻验证。13 个固定 rig、每 rig 42 帧，共 546 个单元；结果相关的替换工况没有使用。

正式运行完成全部 546 个单元，`18/18` 项执行有效性检查通过。完全独立第二实现也重算了全部场、观测、四项指标、逐 rig 尾部和调用账。第一次独立验证中，28 项检查有 26 项为真；两项失败来自两套数学等价的 Jacobi 重建没有逐字节同 hash，而不是场、指标、观测、离散判决、输入或调用账不一致。该次结果保持为 `INCONCLUSIVE`，没有被覆盖。

在读取 13 个 Jacobi 数值差之前，v235.1 另行冻结了仅针对这项验证缺陷的数值等价门。它不读取 Case 7 密度或科学数组、不重跑算法、不开新候选，只重建几何 Jacobi。最大相对 L2、尺度化绝对差和 floor 相对差分别为 `2.18036e-16 / 2.18036e-16 / 1.56298e-16`，远低于结果前冻结的 `1e-12 / 1e-12 / 1e-15`。17 个检查均满足预期极性，因此恢复最终独立判决，而不改写第一次验证记录。

固定 K16 reference 本身合格：`546/546` 严格安全单元、`13/13` 完整 rig。固定 Direct Low64 K11 也通过绝对门：`546/546、13/13`。但论文目标要求它在减少调用时仍与 K16 达到匹配精度；这一门只通过 `330/546` 单元，完整 rig 为 `0/13`。

| 方法 | 绝对安全单元 | 完整 rig | 相对 K16 匹配单元 |
| --- | ---: | ---: | ---: |
| Zero geometry-Jacobi PCGLS K16 reference | `546/546` | `13/13` | `546/546` |
| Direct Low64 warm + PCGLS K11 | `546/546` | `13/13` | `330/546` |
| Zero geometry-Jacobi PCGLS K11 | `257/546` | `0/13` | `0/546` |
| Zero CGLS K11 | `209/546` | `0/13` | `0/546` |
| BP + geometry-Jacobi PCGLS K10 | `259/546` | `0/13` | `0/546` |
| BP + CGLS K10 | `209/546` | `0/13` | `0/546` |

216 个 primary 失败单元中，field / full-gradient / interior-gradient / observation 分别有 `209 / 204 / 196 / 216` 个越线，`196` 个同时越过四项匹配门。primary/reference 比值尾部为：

| 汇总 | Field | Full gradient | Interior gradient | Observation |
| --- | ---: | ---: | ---: | ---: |
| p50 | `0.872729` | `0.920516` | `0.826986` | `0.944291` |
| p90-higher | `1.446797` | `1.263931` | `1.209751` | `1.878106` |
| worst | `1.626182` | `1.383454` | `1.334930` | `2.087383` |

固定 primary 的逻辑账是每单元 `12A+11A^T`，reference 是 `16A+16A^T`，名义总调用少 `28.125%`。但匹配精度失败，所以这不是有效调用节省，更不是 wall-time、RSS 或资源加速证据；正式资源门没有启动。

**讲人话：** 这次考试不是“重建结果够不够像”，而是“少算几步后，是否仍与认真算到 K16 的答案等价”。Direct K11 自己看起来安全，却在 Case 7 的 13 套 rig 上都没有守住与 K16 的逐轨迹匹配精度。简单便宜对照也没有通过，因此失败不能归因成“一个更简单方法已经解释了它”；但固定 Direct Low64 K11 的前瞻迁移仍然失败，不能再用事后调深度、换 basis、改门或加 fallback 挽救。

最终科学判决是 `FAIL_CASE7_LOW64_K11_PROSPECTIVE_CONFIRMATION_V235`。关闭的是当前固定 Direct Low64 K11 路线，不是整个 C 路线，也不是数学不可能性证明。保留的未打开替换工况不为同一失败假设继续消耗；下一步必须等待另行结果前冻结、物理上真正不同的机制，或新的真实配对 BOST 数据。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v235/v235.1: prospective Case 7 evidence rejects fixed Direct Low64 K11 transfer

## Conclusion

v234 found on opened Case 12 that fixed Direct Low64 warm plus unchanged geometry-Jacobi PCGLS K11 passed completely, while the added fallback created the policy failures. Before reading any new numerical values, v235 therefore froze that fixed method by itself and tested it prospectively on then-unopened BLASTNet Case 7. The test contains 13 fixed rigs, 42 frames per rig, and 546 cells. No result-dependent replacement condition was used.

The formal run completes all 546 cells and passes all `18/18` execution-validity checks. A fully independent second implementation also recomputes every field, observation, four-metric gate, rig tail, and call ledger. In the first independent validation, 26 of 28 checks are true. The two failures come from mathematically equivalent Jacobi reconstructions not having byte-identical hashes, rather than any disagreement in fields, metrics, observations, discrete decisions, inputs, or call accounting. That first result remains preserved as `INCONCLUSIVE`.

Before reading the 13 Jacobi numerical differences, v235.1 separately freezes a numerical-equivalence erratum for this validation defect only. It does not read Case 7 density or scientific arrays, rerun the algorithm, or introduce a candidate. It rebuilds geometry-only Jacobi states. Maximum relative L2, scaled absolute, and floor-relative differences are `2.18036e-16 / 2.18036e-16 / 1.56298e-16`, well below the preregistered `1e-12 / 1e-12 / 1e-15` limits. All 17 checks satisfy their expected polarity, recovering the final independent decision without rewriting the first validation record.

The fixed K16 reference is adequate at `546/546` strict-safe cells and `13/13` complete rigs. Fixed Direct Low64 K11 also passes the absolute gate at `546/546 and 13/13`. The thesis target, however, requires matched accuracy to K16 while using fewer calls. That gate passes only `330/546` cells and `0/13` complete rigs.

| Method | Absolute-safe cells | Complete rigs | Matched-to-K16 cells |
| --- | ---: | ---: | ---: |
| Zero geometry-Jacobi PCGLS K16 reference | `546/546` | `13/13` | `546/546` |
| Direct Low64 warm + PCGLS K11 | `546/546` | `13/13` | `330/546` |
| Zero geometry-Jacobi PCGLS K11 | `257/546` | `0/13` | `0/546` |
| Zero CGLS K11 | `209/546` | `0/13` | `0/546` |
| BP + geometry-Jacobi PCGLS K10 | `259/546` | `0/13` | `0/546` |
| BP + CGLS K10 | `209/546` | `0/13` | `0/546` |

Among the 216 primary failures, field / full-gradient / interior-gradient / observation violations occur in `209 / 204 / 196 / 216` cells, with `196` cells violating all four matched gates. Primary/reference ratio tails are:

| Summary | Field | Full gradient | Interior gradient | Observation |
| --- | ---: | ---: | ---: | ---: |
| p50 | `0.872729` | `0.920516` | `0.826986` | `0.944291` |
| p90-higher | `1.446797` | `1.263931` | `1.209751` | `1.878106` |
| worst | `1.626182` | `1.383454` | `1.334930` | `2.087383` |

The fixed primary ledger is `12A+11A^T` per cell versus `16A+16A^T` for the reference, a nominal `28.125%` reduction in total exact calls. Matched accuracy fails, so this is not an effective call saving and is not wall-time, RSS, or resource-speedup evidence. The formal resource gate was not run.

In plain language, this test does not ask whether the reconstruction looks acceptable in isolation. It asks whether taking fewer steps remains equivalent to the careful K16 answer. Direct K11 is absolute-safe, yet none of the 13 Case 7 rigs preserves matched accuracy to K16. No simple equal-or-cheaper control passes either, so a cheaper control does not explain the failure; nevertheless, prospective transfer of fixed Direct Low64 K11 fails and cannot be rescued by post-open depth, basis, gate, or fallback changes.

The final scientific decision is `FAIL_CASE7_LOW64_K11_PROSPECTIVE_CONFIRMATION_V235`. This closes the current fixed Direct Low64 K11 route, not the full C route, and it is not a proof of mathematical impossibility. The unopened replacement conditions remain unused for this failed hypothesis. Any continuation requires a separately preregistered, physically different mechanism or new paired real-BOST data.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
