# v234：Case 12 的失败来自 fallback，不是固定 Direct K11

## 结论

v230.1 已经公开了 Case 12 的完整结果，v229 的 dual-PRESS 策略在这批数据上也已经被看过。v234 因此不是新外门，而是一项结果已开的机制归因：逐单元重放固定 Direct Low64 warm + 未修改 PCGLS K11、Zero geometry-Jacobi PCGLS K16，以及 v229 固定 dual-PRESS 接受/回退策略，问清楚策略失败究竟来自 direct 臂还是 fallback。

正式程序与完全独立第二实现均重建了 13 个 rig、每 rig 46 帧、共 598 个单元的场、观测、四项精度门、逐 rig 尾部与逻辑调用账。独立复算 `14/14` 项检查全真，正式与独立的归因行逐项一致，K1-K16 深度表逐项一致，汇总最大数值差为 `1.47097e-10`。

结果非常明确：

| 固定方法/策略 | 严格安全单元 | 完整 rig |
| --- | ---: | ---: |
| Direct Low64 warm + PCGLS K11 | `598/598` | `13/13` |
| Zero geometry-Jacobi PCGLS K16 | `594/598` | `11/13` |
| v229 dual-PRESS 策略 | `595/598` | `11/13` |

dual-PRESS 接受 437 个单元、拒绝 161 个单元。被拒绝的 161 个 direct 结果全部本来就严格安全；其中 3 个单元在回退到 K16 后才变得不安全，恰好等于策略的全部 3 个失败。与此同时，接受 direct 还救回了 1 个 K16 本来不安全的单元。因果划分因此闭合：当前固定 fallback 没有保护 Case 12，反而制造了策略的全部失败。

固定 Direct K11 的 field / full-gradient / interior-gradient / observation 相对误差为：

| 汇总 | Field | Full gradient | Interior gradient | Observation |
| --- | ---: | ---: | ---: | ---: |
| p50 | `0.245623` | `0.498138` | `0.510966` | `0.046925` |
| p90-higher | `0.256611` | `0.521771` | `0.548982` | `0.050743` |
| worst | `0.262944` | `0.536194` | `0.585944` | `0.054065` |

逻辑在线账也更低：固定 Direct K11 每单元为 `12A + 11A^T`，旧策略平均为 `13.076923A + 12.346154A^T`。在 598 个单元上，去掉 fallback 共少 `644A + 805A^T`。这是封存调用 receipt 的逻辑差，不是 fresh wall 或 RSS 结果。

**讲人话：** 旧策略不是因为“直接做 K11 不够准”而失败，恰好相反，直接做 K11 在 Case 12 全部通过；加上的风险 fallback 把三个本来正确的结果换坏了。正式科学判决是：

`POST_OPEN_CASE12_DIRECT_LOW64_K11_CONTRACT_DOMINATES_FIXED_DUAL_PRESS_FALLBACK_V234`

因此关闭的是当前 v229 固定 dual-PRESS fallback 壳，不是整个 C 路线。下一步只能另行结果前冻结固定 Direct Low64 warm + 未修改 PCGLS K11，并在下一个全局未打开的合格条件上做一次前瞻验证；不能把 Case 12 回顾归因写成外部泛化、速度或算法突破。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v234: the Case 12 failures come from fallback, not fixed Direct K11

## Conclusion

Case 12 was already opened by v230.1, and the v229 dual-PRESS policy outcomes had already been inspected. v234 is therefore not a new external gate. It is a post-open mechanism attribution that replays three fixed alternatives in every cell: Direct Low64 warm plus unchanged PCGLS K11, Zero geometry-Jacobi PCGLS K16, and the fixed v229 dual-PRESS accept/fallback policy. The question is whether the policy failures originate in the direct arm or in fallback.

The formal program and a fully independent second implementation rebuild the fields, observations, four accuracy gates, per-rig tails, and logical call ledger for 13 rigs, 46 frames per rig, and 598 cells. All `14/14` independent checks pass. Formal and independent attribution rows agree exactly, the K1-K16 depth tables agree exactly, and the maximum numerical summary difference is `1.47097e-10`.

The result is unambiguous:

| Fixed method/policy | Strict-safe cells | Complete rigs |
| --- | ---: | ---: |
| Direct Low64 warm + PCGLS K11 | `598/598` | `13/13` |
| Zero geometry-Jacobi PCGLS K16 | `594/598` | `11/13` |
| v229 dual-PRESS policy | `595/598` | `11/13` |

The policy accepts 437 cells and rejects 161. All 161 rejected direct results are already strict-safe. Three of them become unsafe only after K16 fallback, exactly accounting for all three policy failures. Accepting direct also rescues one cell that is unsafe under K16. The causal partition therefore closes: the current fixed fallback does not protect Case 12; it creates every policy failure.

The fixed Direct K11 field / full-gradient / interior-gradient / observation relative errors are:

| Summary | Field | Full gradient | Interior gradient | Observation |
| --- | ---: | ---: | ---: | ---: |
| p50 | `0.245623` | `0.498138` | `0.510966` | `0.046925` |
| p90-higher | `0.256611` | `0.521771` | `0.548982` | `0.050743` |
| worst | `0.262944` | `0.536194` | `0.585944` | `0.054065` |

The logical online ledger is also lower: fixed Direct K11 uses `12A + 11A^T` per cell, while the old policy averages `13.076923A + 12.346154A^T`. Removing fallback saves `644A + 805A^T` over 598 cells. This is a sealed logical-call difference, not a fresh wall-time or RSS result.

In plain language, the old policy does not fail because direct K11 is insufficient. Direct K11 passes every Case 12 cell; the added risk fallback replaces three correct results with failures. The scientific decision is `POST_OPEN_CASE12_DIRECT_LOW64_K11_CONTRACT_DOMINATES_FIXED_DUAL_PRESS_FALLBACK_V234`.

This closes the current fixed v229 dual-PRESS fallback shell, not the C route. The only valid next step is to separately preregister fixed Direct Low64 warm plus unchanged PCGLS K11 and test it prospectively on the next globally unopened eligible condition. The retrospective Case 12 attribution is not external generalization, speedup, or an algorithm breakthrough.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
