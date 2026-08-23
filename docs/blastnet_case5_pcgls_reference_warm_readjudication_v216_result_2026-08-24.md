# v216：合格 PCGLS 参考下，固定 Low-64 Warm Start 正式失败

## 结论

v215 因 Zero-CGLS K16 参考不足而不能裁决 warm start，但同一批预注册控制中，精确几何 Jacobi-PCGLS K16 已由正式与独立实现共同重放，并在 `546/546` 个单元、`13/13` 套完整几何上通过四项绝对门。

v216 在读取任何新的 proxy-versus-PCGLS matched 数值前，把这个充分参考固定为唯一裁判，并只对 v215 已封存的正式与独立指标数组分别做再审裁。判决是：

`FAIL_FIXED_LOW64_PROXY_WARM_START_AGAINST_ADEQUATE_PCGLS_REFERENCE_V216`

这次不是 reference 不足造成的不可判定，而是固定 low-64 observation/geometry-only proxy warm start 在合格参考下确实没有通过。

## 严格结果

五个全局固定 proxy checkpoint 均未通过完整的绝对门与 matched-accuracy 门：

| Proxy arm | `A/A^T` | 绝对门单元 | 绝对门完整几何 | Matched 单元 | Matched 完整几何 |
|---|---:|---:|---:|---:|---:|
| low-64 K0 | `1/0` | `0/546` | `0/13` | `0/546` | `0/13` |
| low-64 K1 | `2/1` | `0/546` | `0/13` | `0/546` | `0/13` |
| low-64 K2 | `3/2` | `0/546` | `0/13` | `0/546` | `0/13` |
| low-64 K4 | `5/4` | `390/546` | `0/13` | `0/546` | `0/13` |
| low-64 K8 | `9/8` | `546/546` | `13/13` | `0/546` | `0/13` |

K8 已经通过原有绝对精度门，但它仍没有与合格的 PCGLS-K16 reference 等价。按逐单元 `1.05` matched 上限计，K8 的 field / 完整梯度 / 内部梯度 / observation 越线数分别为 `545 / 546 / 23 / 546`。其逐单元 matched ratio 的中位数为 `1.14343 / 1.16705 / 0.99619 / 1.71339`。

因此问题不是只剩一个边缘单元，也不是仅由内部梯度造成。当前代理经过八步 CGLS 后，observation 和完整梯度在全部单元仍比充分参考差逾 5%。

## 为什么这次可以下负结论

v216 没有事后换 reference 或挑 proxy 深度。唯一 reference、五个 checkpoint 的固定顺序、四项绝对门、matched ratio、调用账和 deterministic-control 支配规则都在读取新的 matched 结果前冻结。正式侧只读 v215 formal metrics/calls，独立侧只读 v215 独立重放的 metrics/calls；没有新真值读取、forward、adjoint 或训练。

独立再审裁 `18/18` 项检查全真：

- v215 正式与独立逐单元指标最大差：`1.43e-10`
- v216 正式与独立汇总最大差：`1.86e-10`
- 调用账、reference 充分性、checkpoint 选择与最终判决完全一致
- 封存输入与源码闭包保持不变

## 路线动作

关闭当前固定 low-64 observation proxy warm-start 机制，不用更大 CNN、FNO、UNO 或 GPU 挽救。下一步先用结果前冻结的合同确定最低仍充分的全局 PCGLS 深度，收紧 deterministic baseline；任何新的 observation-only initializer 都必须同时达到 matched accuracy、严格少于该基线的 `A/A^T`，并排除同价或更便宜控制。

本结果只属于已开封 Case 5 的 post-open 机制裁决。它不是外部泛化、wall/RSS、曲线光路或真实 BOST 结果。

`algorithm_breakthrough=false`

---

# v216: The Fixed Low-64 Warm Start Fails Against an Adequate PCGLS Reference

## Conclusion

v215 cannot adjudicate the warm start because its designated Zero-CGLS K16 reference is inadequate. In the same preregistered replay, however, exact-geometry Jacobi-PCGLS K16 is independently reproduced and clears all four absolute gates in `546/546` cells and `13/13` complete geometries.

Before reading any new proxy-versus-PCGLS matched value, v216 fixes that adequate arm as the sole reference and separately re-adjudicates the sealed formal and independent v215 arrays. The decision is:

`FAIL_FIXED_LOW64_PROXY_WARM_START_AGAINST_ADEQUATE_PCGLS_REFERENCE_V216`

This is no longer an inconclusive result caused by an inadequate reference. The fixed low-64 observation/geometry-only proxy warm start genuinely fails against an adequate reference.

## Strict Result

None of the five globally fixed proxy checkpoints clears the complete absolute-plus-matched contract:

| Proxy arm | `A/A^T` | Absolute cells | Complete absolute rigs | Matched cells | Complete matched rigs |
|---|---:|---:|---:|---:|---:|
| low-64 K0 | `1/0` | `0/546` | `0/13` | `0/546` | `0/13` |
| low-64 K1 | `2/1` | `0/546` | `0/13` | `0/546` | `0/13` |
| low-64 K2 | `3/2` | `0/546` | `0/13` | `0/546` | `0/13` |
| low-64 K4 | `5/4` | `390/546` | `0/13` | `0/546` | `0/13` |
| low-64 K8 | `9/8` | `546/546` | `13/13` | `0/546` | `0/13` |

K8 clears the original absolute gates but still does not match the adequate PCGLS-K16 reference. Under the per-cell `1.05` matched limit, its field / full-gradient / interior-gradient / observation violation counts are `545 / 546 / 23 / 546`. The corresponding median per-cell ratios are `1.14343 / 1.16705 / 0.99619 / 1.71339`.

The failure is therefore neither a single marginal cell nor an interior-gradient-only issue. After eight CGLS steps, observation and full-gradient errors remain more than five percent worse than the adequate reference in every cell.

## Why the Negative Decision Is Valid

v216 does not switch references or select a proxy depth after seeing the result. The unique reference, fixed checkpoint order, four absolute gates, matched ratios, call ledgers, and deterministic-control dominance rule are frozen before the new matched values are read. The formal side reads only sealed v215 formal metrics/calls, while the independent side reads only independently replayed v215 metrics/calls. No new truth read, forward, adjoint, or training occurs.

All `18/18` independent checks pass:

- maximum v215 formal-versus-independent cell-metric difference: `1.43e-10`
- maximum v216 formal-versus-independent summary difference: `1.86e-10`
- call ledgers, reference adequacy, checkpoint selection, and the final decision agree exactly
- sealed inputs and source closure remain unchanged

## Route Action

Close the fixed low-64 observation-proxy warm-start mechanism. Do not rescue it with a larger CNN, FNO, UNO, or GPU. The next gate first preregisters the lowest globally adequate PCGLS depth to tighten the deterministic baseline. Any new observation-only initializer must then achieve matched accuracy, use strictly fewer `A/A^T` calls, and survive equal-or-cheaper control attribution.

This is a post-open Case 5 mechanism decision only. It is not external generalization, wall/RSS evidence, curved-ray validation, or real BOST.

`algorithm_breakthrough=false`
