# v246：双实现最坏包络否决继续加深固定 reference

## 先说结论

Case 19 的 K14 → K16 reference 收敛诊断已经由 formal 与完全独立的循环式第二实现共同封存。独立验证 **16/16** 项全真，两套 v246 算法的最大差为 **0**。正式科学判决是：

`FAIL_CASE19_TWO_IMPLEMENTATION_ENVELOPE_NOT_MONOTONE_V246`

因此不运行 K20，也不搜索 K18、K22、K24、K32。当前固定深度的 global geometry-Jacobi PCGLS reference 加深路线关闭。

## 为什么先有 v245，再有 v246

v245 原本要求分别读取 formal 与 independent 两份父指标，再以 `1e-10` 比较两套诊断数值。formal 成功封存，但 independent 在写出任何验证结果前发现数值差超过这个结果前界并 fail-closed。v245 因此保持：

`INCONCLUSIVE_INVALID_CASE19_REFERENCE_CONVERGENCE_V245`

没有放宽 v245 容差，也没有第二次运行同一个 validator。

v246 问的是一个更严格而不同的问题：既然两份父指标已有约 `1.0219e-8` 的差异，就不把它们强行当成同一个数，而是逐分量取下界与上界。后续所有安全集合、增益、剩余差距和 rig 尾部都用最不利组合判定。换句话说，父差异被完整计入，不是被容差忽略。

## 最坏包络结果

从 K14 到 K16，确定安全单元从 **313/429** 增至 **417/429**：

- 新增确定安全单元：**104**；
- 丢失确定安全单元：**0**；
- 所有 rig × 指标的 p90 与 worst 最坏包络都没有恶化；
- 最小 p90 安全余量：**0.0127086**；
- 最小 worst 安全余量：**0.0136481**。

这些是明确改善，但不足以通过完整合同。K16 仍有 **12** 个可能越过绝对门的指标分量，其中 **11** 个具有正的最坏情形 K14 → K16 增益，且它们的最大“剩余差距 / 两步增益”只有 **0.366174**。

唯一阻断项位于第 0 条 rig、第 0 帧的内部梯度：

| 项目 | 数值 |
| --- | ---: |
| 绝对门 | 0.750000 |
| K14 包络 | 0.758223464859 – 0.758223464862 |
| K16 包络 | 0.758223464859 – 0.758223464862 |
| 最坏增益 | -2.97e-12 |
| 剩余差距 | 0.00822346 |

它从 K14 到 K16 在数值精度内没有改善。结果前规则要求每个仍可能失败的分量都必须有严格正增益，因此不能用其余 11 个分量的良好趋势替它投票，也不能据此假定 K20 会修复首帧。

## 独立复算和成本边界

formal 使用 NumPy 向量化下界/上界、掩码和 `higher` 分位数；独立侧不导入 formal 实现，改用逐 arm、rig、frame、metric 的标量 `min/max`、显式集合与排序索引分位数。独立验证 **16/16** 项通过，所有数值、集合、条件与最终判决最大差均为 **0**。

v246 只读取两份已封存指标数组，新增调用账是 `0A+0A^T`。它没有读取新原始密度、没有运行 forward/adjoint/solver，也没有测 wall 或 RSS。

## 路线动作与证据边界

这项结果关闭的是“继续加深同一个固定 global geometry-Jacobi PCGLS reference”这一条解释，不是整个 C 路线。下一步只能来自新的物理信息，或一个与固定深度、全局低秩和既有 global quadratic 机制真正不同、结果前唯一冻结且可证伪的机制。

Case 19 已经开封，所以 v246 仍是 post-open 机制诊断，不是外部泛化。当前不训练 CNN/FNO/UNO，不租 GPU，不启动 wall/RSS，也不把页面、测试或封存完整性写成算法成果。

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`

---

# v246: a worst-case two-implementation envelope rejects deeper fixed-reference iteration

## Bottom line

The Case 19 K14-to-K16 reference-convergence diagnostic is now sealed by a formal implementation and a fully independent loop-based implementation. Independent validation passes **16/16** checks, with maximum disagreement **0** between the two v246 algorithms. The scientific decision is:

`FAIL_CASE19_TWO_IMPLEMENTATION_ENVELOPE_NOT_MONOTONE_V246`

K20 will therefore not run, and K18, K22, K24, and K32 will not be searched. The current fixed-depth global geometry-Jacobi PCGLS reference-deepening route is closed.

## Why v245 precedes v246

v245 separately read the formal and independent parent metrics and required the two resulting diagnostics to agree within the preregistered `1e-10` bound. Its formal output sealed successfully, but its independent attempt failed closed before writing validation output because the numerical comparison exceeded that bound. v245 remains:

`INCONCLUSIVE_INVALID_CASE19_REFERENCE_CONVERGENCE_V245`

Its tolerance was not relaxed, and its validator was not rerun.

v246 asks a stricter, different question. Because the two parent metric replicas already differ by about `1.0219e-8`, each pair is retained as a componentwise lower/upper envelope. Safety, gain, remaining gap, and complete-rig tails are all judged under the adverse combination. Parent disagreement is therefore included, not tolerance-filtered away.

## Worst-case envelope result

The definitely-safe set grows from **313/429** cells at K14 to **417/429** at K16:

- **104** definitely-safe cells are gained;
- **0** definitely-safe cells are lost;
- every rig-by-metric p90 and worst tail is non-worsening under the envelope;
- the minimum robust p90 margin is **0.0127086**;
- the minimum robust worst margin is **0.0136481**.

This is real improvement, but it does not satisfy the full contract. K16 retains **12** possibly unsafe metric components. Eleven have positive worst-case K14-to-K16 gain, and their maximum remaining-gap/two-step-gain ratio is only **0.366174**.

The sole blocker is the interior-gradient component at rig 0, frame 0:

| Item | Value |
| --- | ---: |
| Absolute limit | 0.750000 |
| K14 envelope | 0.758223464859 – 0.758223464862 |
| K16 envelope | 0.758223464859 – 0.758223464862 |
| Worst-case gain | -2.97e-12 |
| Remaining gap | 0.00822346 |

It does not improve from K14 to K16 within numerical precision. The preregistered rule requires strictly positive gain for every possibly failing component, so the favorable behavior of the other eleven components cannot outvote it, nor can it justify assuming that K20 would repair the first frame.

## Independent recomputation and cost boundary

Formal uses vectorized NumPy envelopes, masks, and `higher` quantiles. The independent side imports no formal implementation and instead uses scalar `min/max`, explicit arm/rig/frame/metric loops, set construction, and sorted-index quantiles. It passes **16/16** checks with exact agreement in values, sets, conditions, and the final decision.

v246 reads only two already sealed metric arrays and adds `0A+0A^T`. It reads no new raw density, executes no forward, adjoint, or solver, and measures no wall time or RSS.

## Route action and evidence boundary

This result closes only the explanation that the same fixed global geometry-Jacobi PCGLS reference should be deepened. It does not close the entire C route. Further work requires new physical information or one physically distinct, uniquely preregistered, falsifiable mechanism beyond fixed depth, global low rank, and the existing global-quadratic family.

Case 19 is already open, so v246 remains a post-open mechanism diagnostic rather than external generalization. No CNN/FNO/UNO training, GPU rental, wall/RSS gate, or algorithmic claim is authorized.

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`
