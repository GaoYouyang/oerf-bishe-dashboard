# v237/v237.2: causal Krylov recycling improves absolute safety but matches K16 only at the anchors

## Conclusion

v237 tests a solver-native mechanism that is physically different from the fixed cross-rig rank-64 spaces later rejected by v236, v238, and v239. Each Case 7 rig starts with one zero-start geometry-Jacobi PCGLS-K16 anchor. Its sixteen field and projected directions form a measurement-orthonormal FIFO cache. Every later frame uses the current observation to project into that cache, runs exactly one unchanged PCGLS iteration, and then causally appends the new K1 direction while dropping the oldest direction.

The mechanism has a real but insufficient effect. The dynamic FIFO16 primary clears the absolute per-cell gate in **`148/546`** cells, versus **`14/546`** for the fixed frame-zero cache, **`13/546`** for previous-field carry, and **`0/546`** for zero-start PCGLS-K2. All **`533/533`** non-anchor cache updates are accepted, with maximum forward-consistency error `2.30e-16` and measurement-orthogonality error `4.88e-15`.

It nevertheless matches the adequate K16 reference in only **`13/546`** cells. Those thirteen cells are exactly frame zero, one anchor per rig; the later-frame result is **`0/533`**. Complete-rig results are `0/13` on both absolute and matched-accuracy contracts.

| Method | Absolute safe cells | K16-matched cells | Complete rigs |
| --- | ---: | ---: | ---: |
| Causal FIFO16 + PCGLS-K1 | `148/546` | `13/546` | `0/13` |
| Fixed frame-zero cache + K1 | `14/546` | `13/546` | `0/13` |
| Previous candidate field + K1 | `13/546` | `13/546` | `0/13` |
| Zero-start geometry-Jacobi PCGLS-K2 | `0/546` | `0/546` | `0/13` |

For the primary, global absolute p90 values for field, full gradient, interior gradient, and observation are `0.520900 / 0.694950 / 0.850791 / 0.350822`. The corresponding p90 ratios to K16 are `1.809275 / 1.699084 / 1.577655 / 7.848719`, all above the matched limit `1.02` at complete-rig level and far above the per-cell limit `1.05` in the tail.

The sequence ledger is `98A + 57A^T`, versus `672A + 672A^T` for per-frame K16, a nominal total-call reduction of `88.47%`. Because matched accuracy fails after every anchor, this is **not** an effective exact-call saving and does not authorize wall/RSS measurement.

## Validation erratum

The first independent implementation replays all 546 cells and passes 20 of 21 checks. Its only failed check compares the independently rebuilt frame-zero anchor with a `1e-10` tolerance, even though the same v237 contract permits `2e-9` for every formal-independent field comparison. The reported anchor difference is `7.72e-10`; the maximum difference over all fields is `1.58e-9`.

v237.2 is an explicitly post-result, validation-only correction. It inherits the already registered `2e-9` independent-field tolerance rather than selecting a new threshold from the anchor result. It does not rebuild Case 7 physics, rerun any candidate, or rewrite any formal or independent array. A separate sealed-array adjudicator directly recomputes 2,184 field comparisons, thirteen anchors, metric/cache differences, exact call ledgers, and the decision without importing the v237 decision helper.

All **`26/26`** v237.2 checks pass. The direct anchor maximum remains `7.72e-10`, the full-field maximum remains `1.58e-9`, the metric maximum is `1.75e-9`, the cache maximum is `7.77e-16`, and call ledgers are exact. The original inconclusive tree remains preserved.

The validated scientific decision is `FAIL_CASE7_CAUSAL_KRYLOV_RECYCLING_V237`. It closes only the frozen FIFO16-plus-K1 update mechanism. It does not justify a rank, depth, reset, update-rule, CNN, FNO, or GPU rescue, and it does not close the full C route.

This is a post-open mechanism negative on controlled Case 7, not an external result, a learned algorithm, wall/RSS evidence, curved-ray validation, or real BOST.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.

---

# v237/v237.2：因果 Krylov 回收提高绝对安全性，但只有锚点达到 K16 同精度

## 结论

v237 检验的是一种求解器原生机制，与 v236、v238、v239 后来否定的固定跨 rig rank-64 空间物理上不同。每条 Case 7 rig 先用一次零初值 geometry-Jacobi PCGLS-K16 建立锚点，将十六个场方向及其投影方向整理成观测空间正交的 FIFO 缓存。之后每帧只用当前观测投影进缓存，运行一次未修改 PCGLS，再把新的 K1 方向因果加入缓存并删除最老方向。

这个机制确实有作用，但远远不够。动态 FIFO16 主候选有 **`148/546`** 个单元通过绝对门；固定首帧缓存、上一候选场传递和零初值 PCGLS-K2 分别只有 **`14/546`**、**`13/546`** 和 **`0/546`**。全部 **`533/533`** 个非锚点更新都成功接纳，最大 forward consistency 误差为 `2.30e-16`，观测空间正交误差为 `4.88e-15`。

但是，它相对充分 K16 reference 只有 **`13/546`** 个 matched 单元，而且这十三个恰好全是每条 rig 的第零帧锚点；后续为 **`0/533`**。绝对门和 matched 门的完整 rig 都是 `0/13`。

| 方法 | 绝对安全单元 | K16 同精度单元 | 完整 rig |
| --- | ---: | ---: | ---: |
| 因果 FIFO16 + PCGLS-K1 | `148/546` | `13/546` | `0/13` |
| 固定首帧缓存 + K1 | `14/546` | `13/546` | `0/13` |
| 上一候选场 + K1 | `13/546` | `13/546` | `0/13` |
| 零初值 geometry-Jacobi PCGLS-K2 | `0/546` | `0/546` | `0/13` |

主候选的 field、完整梯度、内部梯度、observation 全局绝对 p90 为 `0.520900 / 0.694950 / 0.850791 / 0.350822`；相对 K16 的 p90 比值为 `1.809275 / 1.699084 / 1.577655 / 7.848719`，远高于 matched 门。

整条序列的逻辑账为 `98A + 57A^T`，逐帧 K16 为 `672A + 672A^T`，名义总调用减少 `88.47%`。但由于所有非锚点都没有达到同精度，这不能称为有效 exact-call 减少，也不授权 wall/RSS 测量。

## 验证勘误

第一次独立实现重放全部 546 个单元，21 项检查中通过 20 项。唯一失败项把独立重建的首帧锚点限制在 `1e-10`，而同一 v237 合同对所有 formal-independent 场差已经允许 `2e-9`。实际 anchor 最大差为 `7.72e-10`，全部场的最大差为 `1.58e-9`。

v237.2 明确是结果后的纯验证勘误。它继承原来已经注册的 `2e-9` 独立场门，没有根据 anchor 数字另选新阈值；也没有重建 Case 7 物理、重跑候选或改写任何数组。独立封存数组再裁决器不导入 v237 判决 helper，直接重算 2,184 个场差、十三个 anchor、指标/缓存差、调用账和最终判决。

最终 **`26/26`** 项检查全真：anchor 最大差仍为 `7.72e-10`，全场最大差 `1.58e-9`，指标最大差 `1.75e-9`，缓存最大差 `7.77e-16`，调用账逐值一致；旧 inconclusive 证据树原样保留。

最终科学判决为 `FAIL_CASE7_CAUSAL_KRYLOV_RECYCLING_V237`。它只关闭冻结的 FIFO16 + K1 更新机制，不授权调整 rank、深度、reset、更新规则、CNN、FNO 或 GPU，也不关闭完整 C 路线。

这是已开封 Case 7 上的机制负结果，不是外部门、学习算法、wall/RSS、curved ray 或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。
