# v230.1：Case 12 没有进入策略胜负，K16 参考解先失去资格

## 结论

v230 原本要把 v229 冻结的嵌套双 PRESS 规则放到 Case 12 上做同族新工况检验。第一次正式与独立复算在相机换序和第二求解器比较中出现了近零残差向量的相对差异，因此没有解释任何策略数字。v230.1 在不读取真值指标、策略汇总或接受结果之前，先冻结一套结果盲数值审裁：比较完整物理场、标量观测误差、残差方程闭合、双分数与离散决策，不再把近零残差向量的相对差当作物理等价门。

结果盲审裁的 `18/18` 个数值检查和 `7/7` 个科学释放检查全部通过。正式与完全独立第二实现的场相对差最多为 `4.85e-9`，标量观测误差绝对差最多为 `3.18e-14`，分数差最多为 `2.22e-15`，离散决策差为 `0`。相机换序后的场差约 `5.68e-9`，标量观测误差差约 `5.08e-15`；保存残差满足观测方程到 `2.04e-16`。这说明原来的告警来自近零残差向量的病态相对比较，不是场、观测误差、分数或决策发生了物理变化。

数值门通过后，正式与独立实现共同得到唯一科学判决：

`INCONCLUSIVE_INADEQUATE_CASE12_K16_REFERENCE_V230`

冻结的未修改 PCGLS K16 参考解只通过 `594/598` 个严格单元与 `11/13` 个完整 rig。四个失败都只越过内部梯度逐单元门 `0.75`：

| rig | frame | field | gradient | interior-gradient | observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 11 | `0.312705` | `0.582761` | `0.753166` | `0.056451` |
| 0 | 42 | `0.313491` | `0.583293` | `0.751727` | `0.058459` |
| 12 | 11 | `0.309986` | `0.580134` | `0.754621` | `0.054764` |
| 12 | 42 | `0.307734` | `0.577743` | `0.752865` | `0.056514` |

这四个单元只高出冻结门 `0.0017-0.0046`。rig 0 与 rig 12 的内部梯度 p90 分别仍为 `0.741774` 与 `0.741684`，但合同要求每个参考单元都合格，所以不能事后放宽成“差不多通过”。

## 为什么不能继续解释 dual-PRESS

策略比较的前提是 K16 参考解自身合格。现在参考门先失败，因此候选接受数、安全性、逻辑调用节省、wall time 和 RSS 都没有可解释性。它既不是 dual-PRESS 的外部成功，也不是 dual-PRESS 的算法失败；它精确说明的是 **Case 12 的冻结 K16 reference 不足以承担这次裁决**。

Case 12 已经打开，v230.1 又是看到数值告警后冻结的审裁，所以这也不再是纯净的一次性外部结果。下一步只能另行结果前冻结一个 post-open reference-depth qualification，使用固定深度名单寻找最小合格的未修改 PCGLS 深度。将来若再做新的未开封工况，必须在打开前绑定已经合格的 reference，不能回头用 Case 12 调策略或放宽精度门。

`resource_gate_authorized=false`、`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`real_bost=false`。

---

# v230.1: Case 12 never reaches a policy verdict because the K16 reference fails first

## Conclusion

v230 was intended to evaluate the fixed v229 nested dual-PRESS rule on Case 12 as a new same-family condition. The first formal and independent runs produced a discrepancy in camera-permutation and second-solver comparisons of a near-zero residual vector, so no policy number was interpreted. Before reading truth metrics, policy summaries, or acceptance outcomes, v230.1 froze a result-blind numerical adjudication based on complete physical fields, scalar observation errors, residual-equation closure, both scores, and discrete decisions. Relative disagreement between near-zero residual vectors was retained only as a diagnostic.

All `18/18` numerical checks and `7/7` science-release checks pass. The maximum formal-independent field difference is `4.85e-9`, the maximum scalar observation-error difference is `3.18e-14`, the maximum score difference is `2.22e-15`, and the discrete-decision mismatch count is `0`. Under camera reordering, the field difference is about `5.68e-9`, the scalar observation-error difference is about `5.08e-15`, and the saved residual satisfies the observation equation to `2.04e-16`. The earlier warning therefore came from an ill-conditioned relative comparison of a near-zero residual vector, not from a physical change in fields, scalar errors, scores, or decisions.

After that numerical barrier, both implementations return the same scientific decision: `INCONCLUSIVE_INADEQUATE_CASE12_K16_REFERENCE_V230`.

The frozen unchanged PCGLS K16 reference passes only `594/598` strict cells and `11/13` complete rigs. All four failures occur only at the strict `0.75` interior-gradient cell gate: frames 11 and 42 in rigs 0 and 12. Their interior-gradient values range from `0.751727` to `0.754621`. Although the two rigs retain p90 values below `0.75`, the preregistered reference contract requires every cell to pass and cannot be relaxed after seeing the result.

## Evidence boundary

Because reference adequacy fails before policy adjudication, candidate acceptance, safety, logical call reduction, wall time, and RSS are not interpretable. This is neither an external confirmation nor an algorithm failure of dual-PRESS. It establishes only that the frozen Case 12 K16 reference is inadequate for this adjudication.

Case 12 is now opened, and v230.1 was frozen after the numerical warning, so it is not a pristine one-shot external result. The next valid step is a separately preregistered post-open reference-depth qualification that identifies the minimum adequate unchanged PCGLS depth from a fixed roster. Any future unopened target must bind an already qualified reference before opening; Case 12 cannot be used to retune the policy or relax the accuracy gate.

`resource_gate_authorized=false`, `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `real_bost=false`.
