# v223：一维可观测调和风险分数存在重叠，安全回退关闭

## 结论

v222.1 已经排除 Low-64 起点中 `null(A)` 成分导致跨工况伤害的解释，并把问题定位到可观测行空间中的谱作用。v223 因此不再改场、不再训练模型，只问一个更窄、可证伪的问题：**能否用部署时可见的当前二维观测与 reported geometry，构造一个一维风险分数，事先识别 direct Low-64 PCGLS K11 在哪些单元会伤害 Zero-PCGLS K16？**

正式程序和独立第二实现得到相同判决：

`FAIL_LOW64_HARMONIC_RISK_OVERLAP_V223`

在 `1261` 个已开封单元中，按冻结的全部绝对门和相对 K16 的 `1.05` 门，有 `1064` 个安全、`197` 个不安全。主分数使用 v214 已独立验证的 observation-only 调和可观测性，方向在看结果前固定为“越高越安全”；便宜 control 使用 Low-64 可观测拟合残差，方向固定为“越低越安全”。

## 结果

| 一维可观测分数 | 安全区间 | 不安全区间 | 严格分离 margin | 判决 |
|---|---:|---:|---:|---|
| 调和可观测性 `h`，高更安全 | `0.884743 - 1.241535` | `0.605149 - 1.118135` | `-0.233392` | 重叠 |
| Low-64 拟合残差 `q0`，低更安全 | `0.437879 - 0.710523` | `0.516021 - 0.741132` | `-0.194502` | 重叠 |

不安全单元整体确实偏向更低的 `h` 和更高的 `q0`，说明这两个量含有风险方向性信息；但区间重叠意味着任何一维阈值都会接受至少一个已知不安全单元，或者拒绝至少一个已知安全单元。结果前冻结的门要求对全部 `1261` 个单元严格分开，所以没有阈值、没有回退策略，也没有 exact-call 节省可供评分。

## 独立复算

独立程序自行重建 Low-64 响应、调和分数、拟合残差、安全标签、区间和分离门。相机换序最大差为 `2.02e-14`，正式与独立特征最大差为 `1.25e-14`，分离 margin 最大差为 `1.14e-14`，离散策略差为 `0`。独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_HARMONIC_RISK_V223`

共享冻结的底层 physics kernels 仍存在，所以 `end_to_end_physics_independence_proven=false`。这不改变本次一维分数重叠的判决，但禁止声称完全独立的端到端物理实现。

## 证据边界

- v223 是已开封 Case 2/5 上的 post-open 机制容量诊断，不是部署算法或 fresh 外部门；
- 它只关闭“一维调和风险或一维拟合残差可以 fail-closed 回退”这一条路线，不证明更高维、物理上不同的可观测机制不可能；
- 不允许看到结果后反转方向、调阈值、换 Low-64 基、改迭代深度或用大模型/GPU 挽救；
- 没有训练、物理候选重放、exact-call 减少、wall/RSS、外部工况、曲线光路或真实 BOST 结果。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v223: Overlapping One-Dimensional Observable Harmonic Risk Closes the Safe Fallback

## Conclusion

v222.1 rules out the Low-64 initializer component in `null(A)` as the cause of cross-condition harm and localizes the issue to spectral action inside the observable row space. v223 therefore changes no field and trains no model. It asks a narrower falsifiable question: **can a one-dimensional score computed only from the current deployment-visible 2D observation and reported geometry identify, in advance, where direct Low-64 PCGLS K11 will harm Zero-PCGLS K16?**

The formal and independent implementations reach the same decision:

`FAIL_LOW64_HARMONIC_RISK_OVERLAP_V223`

Of `1,261` opened cells, `1,064` are safe and `197` are unsafe under every frozen absolute gate and the `1.05` ratios to K16. The primary reuses the independently validated v214 observation-only harmonic observability with a preregistered higher-is-safer orientation. The cheap control is the Low-64 observable fit residual with a preregistered lower-is-safer orientation.

## Results

| Observable scalar | Safe range | Unsafe range | Strict margin | Decision |
|---|---:|---:|---:|---|
| Harmonic observability `h`, higher is safer | `0.884743 - 1.241535` | `0.605149 - 1.118135` | `-0.233392` | overlap |
| Low-64 fit residual `q0`, lower is safer | `0.437879 - 0.710523` | `0.516021 - 0.741132` | `-0.194502` | overlap |

Unsafe cells are directionally shifted toward lower `h` and higher `q0`, so both scores contain risk information. Their ranges nevertheless overlap. Any one-dimensional threshold would accept at least one known unsafe cell or reject at least one known safe cell. Because the frozen gate requires strict separation across all `1,261` cells, no threshold or fallback policy is established and there is no exact-call saving to score.

## Independent recomputation

The independent implementation rebuilds the Low-64 response, harmonic score, fit residual, safety labels, ranges, and separation gates. Maximum camera-permutation difference is `2.02e-14`, maximum formal-independent feature difference is `1.25e-14`, maximum separation-margin difference is `1.14e-14`, and the discrete policy difference is `0`. The independent status is:

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_HARMONIC_RISK_V223`

Frozen low-level physics kernels remain shared, so `end_to_end_physics_independence_proven=false`. This does not alter the scalar-overlap verdict, but it prevents a claim of fully independent end-to-end physics.

## Evidence boundary

- v223 is a post-open mechanism-capacity diagnostic on opened Cases 2 and 5, not a deployment algorithm or fresh external gate;
- it closes only the claim that one-dimensional harmonic risk or one-dimensional fit residual supports fail-closed fallback; it does not prove that every higher-dimensional, physically distinct observable mechanism is impossible;
- orientation, threshold, Low-64 basis, and iteration depth may not be changed after results, and a larger model or GPU may not rescue this route;
- there is no training, physical candidate replay, exact-call reduction, wall/RSS, external-condition, curved-ray, or real-BOST result.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
