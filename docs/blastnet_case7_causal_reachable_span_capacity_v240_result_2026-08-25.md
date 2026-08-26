# v240: the full frozen causal recycling span lacks necessary Case 7 capacity

## Conclusion

v237 showed that the frozen causal FIFO16-plus-K1 rule improves absolute error but matches the adequate K16 reference only at the thirteen frame-zero anchors. v240 asks the sharper question: is the failure caused merely by v237's observable coefficient rule, or do the directions available before each causal update already lack enough capacity?

For each of the **533 later frames** across **13 Case 7 rigs**, v240 independently rebuilds the exact pre-update FIFO16 cache and adds the current raw K1 direction. This rank-17 design is the complete linear span reachable by the frozen mechanism before updating its cache. A truth-aware oracle then minimizes field, full-gradient, interior-gradient, and observation error **separately** within that same span.

This is deliberately optimistic. The four minima may use four different coefficient vectors, so passing would prove only necessary headroom, not one jointly feasible reconstruction. Failure is stronger: if even these separate lower bounds miss the gates, no single coefficient vector in the span can pass all four.

The necessary-capacity result is **`0/533` later cells** and **`0/13` complete rigs**. Metric-specific failures are:

| Metric-specific oracle minimum | Failed later cells | p50 | p90-higher | Worst | Absolute limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Field | `377/533` | `0.332735` | `0.475642` | `0.555630` | `0.500000` |
| Full gradient | `251/533` | `0.554218` | `0.653669` | `0.757015` | `0.750000` |
| Interior gradient | `309/533` | `0.647855` | `0.736045` | `0.801098` | `0.750000` |
| Observation | **`533/533`** | `0.199373` | **`0.301636`** | `0.364315` | `0.200000` |

Observation is decisive: even the observation-specific oracle lower bound fails on every later frame. At complete-rig level, the matched-observation p90 lower bound lies between `6.3474` and `7.1059`, versus the frozen `1.02` limit.

The independent implementation does not import the formal solver or recycling helper. It rebuilds the causal caches and directions, then uses pivoted QR instead of formal SVD. All **`20/20`** checks pass. Across `2,132` metric designs, both implementations obtain numerical rank 17 exactly; the maximum metric-minimum difference is `2.22e-16`, the maximum summary difference is `3.55e-15`, and camera permutation changes a minimum by at most `2.22e-16`.

The validated decision is `FAIL_CASE7_CAUSAL_REACHABLE_SPAN_NECESSARY_CAPACITY_V240`. It closes the entire frozen pre-update FIFO16-plus-current-K1 **linear span**, not only v237's `c = Q^T y` coefficient rule. Increasing predictor capacity cannot recover directions absent from this span, so a CNN, FNO, or GPU run is not an authorized rescue.

This does not close the full C route and does not prove mathematical impossibility outside the audited span. It is a post-open, truth-aware capacity diagnostic, not a deployment algorithm, joint-feasibility result, effective exact-call reduction, wall/RSS result, external generalization, curved-ray validation, or real BOST.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.

---

# v240：冻结的因果回收完整可达空间缺少 Case 7 必要容量

## 结论

v237 已经证明，冻结的因果 FIFO16+K1 规则虽然改善绝对误差，但相对充分 K16 reference 只有十三个首帧锚点达到同精度。v240 进一步追问：失败只是 v237 的可观测系数规则没选好，还是每帧更新前已有的方向本身就不够？

对 **13 条 Case 7 rig 的 533 个后续帧**，v240 独立重建当帧更新前的完整 FIFO16 cache，并加入当前原始 K1 方向。这个 rank-17 设计就是冻结机制在更新前能够到达的完整线性空间。随后用真值可见 oracle，在同一个空间内分别最小化 field、完整梯度、内部梯度和 observation 误差。

这个门故意对候选非常宽松：四个最小值允许使用四组不同系数。因此通过也只能证明“有必要容量”，不能证明存在一组同时满足四门的场；但如果连这些分别最优的下界都失败，那么同一空间里任何单一系数向量都不可能同时通过。

结果为 **`0/533` 个后续单元具备必要安全容量，`0/13` 条完整 rig 通过**。逐指标失败如下：

| 逐指标 oracle 最小值 | 失败后续单元 | p50 | p90-higher | worst | 绝对门 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Field | `377/533` | `0.332735` | `0.475642` | `0.555630` | `0.500000` |
| 完整梯度 | `251/533` | `0.554218` | `0.653669` | `0.757015` | `0.750000` |
| 内部梯度 | `309/533` | `0.647855` | `0.736045` | `0.801098` | `0.750000` |
| Observation | **`533/533`** | `0.199373` | **`0.301636`** | `0.364315` | `0.200000` |

其中 observation 已经给出决定性否证：即使逐帧只为 observation 单独挑最优系数，533 个后续帧仍全部失败。完整 rig 的 matched-observation p90 下界为 `6.3474-7.1059`，远高于冻结的 `1.02` 门。

独立实现不导入正式求解器或回收 helper，自行重建因果 cache 与方向，并用 pivoted QR 替代正式 SVD。最终 **`20/20`** 项检查全真。`2,132` 个逐指标设计在两种实现中数值秩都恰为 17；最小指标最大差 `2.22e-16`，汇总最大差 `3.55e-15`，相机换序带来的最小值差不超过 `2.22e-16`。

最终判决为 `FAIL_CASE7_CAUSAL_REACHABLE_SPAN_NECESSARY_CAPACITY_V240`。它关闭的不只是 v237 的 `c = Q^T y` 系数规则，而是冻结的“更新前 FIFO16 + 当前 K1”**完整可达线性空间**。更大的预测器无法补出这个空间里根本不存在的方向，因此不授权用 CNN、FNO 或 GPU 挽救。

这不关闭完整 C 路线，也不证明审计空间之外数学上不可能。它只是已开封工况上的 truth-aware 容量诊断，不是部署算法、联合可行性、有效 exact-call 减少、wall/RSS、外部泛化、curved ray 或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。
