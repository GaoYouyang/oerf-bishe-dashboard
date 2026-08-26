# v217.1：K16 仍是最低充分的全局固定 PCGLS 深度

## 结论

v216 已确认 geometry-Jacobi PCGLS K16 是合格 reference。v217 继续问一个更严格的问题：在同一批已开封的 `42` 帧、`13` 套虚拟九相机几何和 `546` 个单元上，K8 到 K16 之间是否存在一个更浅、但仍能同时守住绝对精度与 K16 matched-accuracy 的全局固定深度。

v217 首次执行没有通过独立审计：相机包被反转后，观测块没有按相机 ID 恢复到冻结顺序，导致相机乱序数值门失败。它被原样保留为 `INCONCLUSIVE_INVALID_GLOBAL_PCGLS_DEPTH_QUALIFICATION_V217`，不能用于科学结论。

v217.1 只修复这一执行漏项，没有修改数据、PCGLS、深度列表、门槛、指标或成本账。正式与独立实现都确认：

`PASS_K16_REMAINS_MINIMAL_ADEQUATE_GLOBAL_PCGLS_DEPTH_V217_1`

K16 仍是最低可靠的全局固定 deterministic reference。

## 严格结果

| 全局固定深度 | `A/A^T` | 绝对门单元 | 绝对门完整几何 | Matched 单元 | Matched 完整几何 |
|---|---:|---:|---:|---:|---:|
| K8 | `8/8` | `0/546` | `0/13` | `0/546` | `0/13` |
| K9 | `9/9` | `0/546` | `0/13` | `0/546` | `0/13` |
| K10 | `10/10` | `0/546` | `0/13` | `0/546` | `0/13` |
| K11 | `11/11` | `96/546` | `0/13` | `0/546` | `0/13` |
| K12 | `12/12` | `318/546` | `0/13` | `0/546` | `0/13` |
| K13 | `13/13` | `467/546` | `5/13` | `0/546` | `0/13` |
| K14 | `14/14` | `526/546` | `8/13` | `0/546` | `0/13` |
| K15 | `15/15` | `544/546` | `11/13` | `0/546` | `0/13` |
| K16 | `16/16` | `546/546` | `13/13` | `546/546` | `13/13` |

K15 是最接近的低成本深度。它只在两个单元上未过绝对门，但这不等于与 K16 精度等价。K15 相对 K16 的 field / 完整梯度 / 内部梯度 / observation 逐单元比值中位数为 `1.05462 / 1.04711 / 1.03622 / 1.11759`，最大 matched ratio 为 `1.13700`。每个 K15 单元都至少有一个指标超过 `1.05` matched 上限，因此 matched 仍是 `0/546`。

这排除了“把 reference 从 K16 降成 K15 就能无损省一次 A 和 A^T”的解释。

## 独立复算与执行修复

v217.1 把带标签的相机观测包按 camera ID 恢复到冻结顺序后，相机乱序的 K16 场差降为 `0`。正式侧与不可用 v217 的科学数组逐字节一致，证明修复没有改变深度扫描本身；独立侧从输入重新构造相机包、预条件、K8-K16 场、观测、四项指标、逐几何尾部和调用账。

独立 `14/14` 项检查全真：

- 科学判决与调用账完全一致
- 相机 ID 与观测块恢复完全一致
- 相机乱序 K16 场相对差为 `0`
- 正式与独立场最大相对差 `3.22e-9`
- 正式与独立逐单元指标最大差 `1.15e-10`
- 正式与独立汇总最大差 `1.89e-10`
- 输入与父证据保持不变

## 路线动作

K16 继续作为 Case 5 上最低可靠的全局固定 deterministic reference。未来候选不能通过降低 reference 来制造调用优势；它必须用部署可见的二维观测与已知几何生成物理上不同的 warm initializer，在最终 field、完整梯度、内部梯度与 observation 匹配 K16 时，严格减少 A 和 A^T。

本结果只资格化已开封 Case 5 的 deterministic reference。它不是 learned initializer、exact-call 减少、wall/RSS 加速、外部泛化、曲线光路验证或真实 BOST。

`algorithm_breakthrough=false`

---

# v217.1: K16 Remains the Lowest Adequate Globally Fixed PCGLS Depth

## Conclusion

v216 establishes geometry-Jacobi PCGLS K16 as an adequate reference. v217 asks a stricter question on the same `42` opened frames, `13` virtual nine-camera geometries, and `546` cells: is there a shallower globally fixed depth between K8 and K16 that preserves both absolute accuracy and K16-matched accuracy?

The first v217 execution does not pass independent audit. After reversing the labeled camera packet, its observation blocks are not restored by camera ID, so the camera-permutation numerical gate fails. That execution remains preserved as `INCONCLUSIVE_INVALID_GLOBAL_PCGLS_DEPTH_QUALIFICATION_V217` and is not used for a scientific conclusion.

v217.1 repairs only that execution omission. It does not change the data, PCGLS solver, depth roster, thresholds, metrics, or cost ledger. The formal and independent implementations both conclude:

`PASS_K16_REMAINS_MINIMAL_ADEQUATE_GLOBAL_PCGLS_DEPTH_V217_1`

K16 remains the lowest reliable globally fixed deterministic reference.

## Strict Result

| Globally fixed depth | `A/A^T` | Absolute cells | Complete absolute rigs | Matched cells | Complete matched rigs |
|---|---:|---:|---:|---:|---:|
| K8 | `8/8` | `0/546` | `0/13` | `0/546` | `0/13` |
| K9 | `9/9` | `0/546` | `0/13` | `0/546` | `0/13` |
| K10 | `10/10` | `0/546` | `0/13` | `0/546` | `0/13` |
| K11 | `11/11` | `96/546` | `0/13` | `0/546` | `0/13` |
| K12 | `12/12` | `318/546` | `0/13` | `0/546` | `0/13` |
| K13 | `13/13` | `467/546` | `5/13` | `0/546` | `0/13` |
| K14 | `14/14` | `526/546` | `8/13` | `0/546` | `0/13` |
| K15 | `15/15` | `544/546` | `11/13` | `0/546` | `0/13` |
| K16 | `16/16` | `546/546` | `13/13` | `546/546` | `13/13` |

K15 is the closest lower-cost depth. It misses the absolute gates in only two cells, but that does not make it accuracy-equivalent to K16. Its median field / full-gradient / interior-gradient / observation ratios to K16 are `1.05462 / 1.04711 / 1.03622 / 1.11759`, and its largest matched ratio is `1.13700`. Every K15 cell exceeds the `1.05` matched limit in at least one metric, so it remains at `0/546` matched cells.

This rejects the explanation that lowering the reference from K16 to K15 saves one A and one A^T without accuracy loss.

## Independent Recomputation and Execution Repair

After v217.1 restores the labeled observation packet by camera ID, the K16 camera-permutation field difference becomes `0`. The formal scientific arrays are bit-identical to the invalid v217 execution, confirming that the repair does not change the depth scan itself. The independent implementation rebuilds camera packets, preconditioning, K8-K16 fields, observations, all four metrics, per-geometry tails, and call ledgers from the inputs.

All `14/14` independent checks pass:

- scientific decision and call ledgers agree exactly
- camera IDs and observation blocks are restored exactly
- camera-permutation K16 field relative difference is `0`
- maximum formal-independent field relative difference is `3.22e-9`
- maximum formal-independent cell-metric difference is `1.15e-10`
- maximum formal-independent summary difference is `1.89e-10`
- inputs and parent evidence remain unchanged

## Route Action

K16 remains the lowest reliable globally fixed deterministic reference on Case 5. A future candidate cannot manufacture a call advantage by weakening that reference. It must use deployment-visible 2D observations and known geometry to construct a physically distinct warm initializer, match K16 final field, full-gradient, interior-gradient, and observation accuracy, and strictly reduce both A and A^T.

This result only qualifies a deterministic reference on opened Case 5 data. It is not a learned initializer, exact-call reduction, wall/RSS speedup, external generalization, curved-ray validation, or real BOST.

`algorithm_breakthrough=false`
