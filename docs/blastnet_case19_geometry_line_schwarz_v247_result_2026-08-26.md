# v247：精确坐标线 Schwarz 独立复算越过残差界，判决保持不确定

## 先说结论

v247 在已经开封的 Case 19 上检验了一个 solver-native、非学习机制：由报告几何直接构造三个世界坐标轴上的精确线块，并把它们等权组合成 PCGLS 预条件器。formal 完成全部 **429** 个单元并通过 **21/21** 项有效性检查；完全独立的第二实现完成全部复算，但只通过 **25/27** 项。正式科学判决是：

`INCONCLUSIVE_INVALID_CASE19_GEOMETRY_LINE_SCHWARZ_V247`

因此不能把 formal 的通过/失败数字当作有效科学结果，也不能声称该机制减少了有效精确调用。

## 检验了什么

每条坐标线都使用报告几何下的精确正规算子子块。三个轴的线块逆采用同一个结果前固定的全局特征值 floor，再等权相加。没有搜索轴组合、分块、floor、缩放、深度、rank 或阈值；模型参数数为 **0**。

四个冻结 arm 分别是 warm + line-PCGLS K14 主候选、Zero + line-PCGLS K14、归一化 BP + line-PCGLS K13，以及 Zero + line-PCGLS K16 reference。场、完整梯度、内部梯度和 observation 的逐单元与完整 rig 门均保持不变。

## 哪一项没有闭合

两套实现对机制本身的复算非常接近：

| 比较项 | 最大差 | 冻结界 | 判定 |
| --- | ---: | ---: | --- |
| 精确线块，相对差 | 2.95e-16 | 1e-11 | 通过 |
| 线块逆，相对差 | 8.78e-16 | 1e-9 | 通过 |
| 候选场，相对差 | 4.50e-10 | 1e-8 | 通过 |
| 残差数组，相对差 | **3.24e-8** | **1e-8** | **失败** |
| 逐单元指标，绝对差 | 2.09e-10 | 1e-9 | 通过 |
| 汇总，绝对差 | 7.70e-12 | 1e-9 | 通过 |

残差数组的最大绝对差为 **2.38e-9**。虽然两套实现的离散逐 arm 汇总标志完全一致，但离散一致不能替代结果前冻结的浮点闭环。第二个失败检查 `formal_independent_decision_exact` 是这个验证缺口的派生结果：formal 的 pending 判决不能覆盖独立侧的 fail-closed 状态。

## 为什么不重跑或放宽阈值

`1e-8` 残差界在看到结果前已经冻结。看到 **3.24e-8** 后把它改成更宽的门，会让同一数据同时参与定规则和证明规则。v247 因而不重跑、不改目录、不放宽容差，也不建立 v247.1 来包装一个更好看的结论。

开封后的诊断数字只能帮助判断是否继续投入，不能作为已验证性能结论。它们显示四个 arm 都是 **0/13** 条完整 rig；主候选、K16 reference 与两个 controls 的逐单元严格安全数分别为 **73/429、14/429、9/429、9/429**。这支持停止继续投资当前精确线 Schwarz 机制，但不构成“数学上不可能”的证明。

## 路线动作与证据边界

当前精确坐标线 Schwarz 机制不再授权继续运行。关闭的是这个固定实现，不是整个 C 路线。下一步只能来自新的物理信息，或另一个物理上真正不同、结果前唯一冻结且可证伪的机制。

Case 19 已经开封，所以 v247 只是 post-open 机制诊断；它不是外部确认。没有 fresh wall/RSS、真实 BOST、curved-ray、预测器训练或 GPU 结论。

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`

---

# v247: exact coordinate-line Schwarz remains inconclusive after residual disagreement

## Bottom line

v247 evaluates a solver-native, non-learned mechanism on already-opened Case 19. Exact normal-operator blocks are constructed along the three reported world-coordinate axes and averaged into a PCGLS preconditioner. Formal completes all **429** cells and passes **21/21** validity checks. A fully independent second implementation completes the recomputation but passes only **25/27** checks. The scientific decision is:

`INCONCLUSIVE_INVALID_CASE19_GEOMETRY_LINE_SCHWARZ_V247`

Formal pass/fail counts are therefore not an admissible scientific result, and no effective exact-call reduction can be claimed.

## What was tested

Each coordinate line uses an exact reported-geometry normal-operator sub-block. The inverse blocks share one preregistered global eigenvalue floor and the three axes are averaged equally. No axis subset, partition, floor, scale, depth, rank, or threshold is searched; the trainable-parameter count is **0**.

The four frozen arms are warm + line-PCGLS K14, Zero + line-PCGLS K14, normalized BP + line-PCGLS K13, and Zero + line-PCGLS K16 reference. Cellwise and complete-rig gates for field, full gradient, interior gradient, and observation remain unchanged.

## What did not close

The two implementations agree closely on the mechanism itself:

| Comparison | Maximum difference | Frozen limit | Result |
| --- | ---: | ---: | --- |
| Exact line blocks, relative | 2.95e-16 | 1e-11 | Pass |
| Line-block inverses, relative | 8.78e-16 | 1e-9 | Pass |
| Candidate fields, relative | 4.50e-10 | 1e-8 | Pass |
| Residual arrays, relative | **3.24e-8** | **1e-8** | **Fail** |
| Cell metrics, absolute | 2.09e-10 | 1e-9 | Pass |
| Summaries, absolute | 7.70e-12 | 1e-9 | Pass |

The maximum absolute residual difference is **2.38e-9**. Discrete arm-level summary flags agree exactly, but discrete agreement cannot replace the preregistered floating-point closure. The second failed check, `formal_independent_decision_exact`, follows from this validation gap: the formal pending decision cannot override the independent fail-closed state.

## Why there is no rerun or tolerance relaxation

The `1e-8` residual limit was frozen before the result was seen. Widening it after observing **3.24e-8** would use the same data both to define and to prove the rule. v247 is therefore not rerun, redirected, tolerance-relaxed, or repackaged as v247.1.

Post-open diagnostic counts may guide whether to keep investing, but they are not validated performance evidence. All four arms reach **0/13** complete rigs; strict-safe cell counts for the primary, K16 reference, and the two controls are **73/429, 14/429, 9/429, and 9/429**. This supports retiring the current exact-line Schwarz mechanism operationally, but it does not prove mathematical impossibility.

## Route action and evidence boundary

The current exact coordinate-line Schwarz mechanism is no longer authorized for continued execution. This closes that fixed implementation, not the whole C route. Further work requires new physical information or another physically distinct, uniquely preregistered, falsifiable mechanism.

Case 19 is already open, so v247 is a post-open mechanism diagnostic rather than external confirmation. It provides no fresh wall/RSS, real-BOST, curved-ray, predictor-training, or GPU result.

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`
