# v252：独立数值一致性门未通过，观测图遍历保持 INCONCLUSIVE 并关闭

## 为什么做

v251 说明直接拼接首帧线性解与后续因果暖启动不能守住 matched-accuracy。v252 改问一个物理上不同的问题：如果只用当前批次的二维观测构造帧间余弦距离图，从观测 medoid 出发按确定性 Prim 规则遍历，再沿遍历顺序使用未修改的 FIFO16 + geometry-Jacobi PCGLS K14，能否避开时间顺序中的困难冷启动？

正式合同在运行前固定了唯一观测图、medoid 与 tie-break、13 套 rig × 33 帧、同一四指标门、K16 reference、等成本固定中点遍历对照和调用账。formal 与 independent 各自在 prediction barrier 前只读取封存观测与 reported geometry；真值和父指标只在遍历顺序封存后用于评分。两套程序独立重建图和求解链，没有放宽容差或重复运行。

## 为什么权威结论只能是不确定

独立程序与 formal 的图边权最大绝对差为 `5.55e-16`，离散 medoid、遍历顺序和父节点全部一致；完整汇总差也通过冻结界。但三项预注册的数值一致性检查未通过：

- field 相对差 `1.1990e-7`，冻结上限 `2e-8`；
- metric 绝对差 `8.6125e-8`，冻结上限 `2e-8`；
- residual 相对差 `8.1713e-6`，冻结上限 `2e-7`。

因此独立状态是 `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_OBSERVATION_GRAPH_TRAVERSAL_V252`，科学判决是 `INCONCLUSIVE_INVALID_CASE19_OBSERVATION_GRAPH_TRAVERSAL_V252`。不能用离散判决一致替代结果前冻结的浮点闭环，也不能在看到结果后放宽容差。

## 事后离散诊断与路线动作

以下数字只用于决定是否继续投资这条机制，**不是通过独立验证的科学性能结果**。两套实现分别计算后，观测图 primary 都是 `427/429` 个绝对单元、`11/13` 条完整绝对 rig；两个缺口同为 rig 0 与 rig 5 的 frame 6，且都只越过 interior-gradient 门。两帧恰好是各自观测图的 medoid anchor，因此 anchor 自身的 K16 解已未过绝对门。

更关键的是，完全同价、但不读取观测相似性的固定中点遍历 control 在两套实现里都是 `429/429` 个绝对单元与 `13/13` 条完整 rig。封存时间顺序 control 为 `428/429` 与 `12/13`，K16 reference 只有 `417/429` 与 `9/13`。这说明当前观测图规则既没有通过自身绝对门，也没有排除更便宜解释。事后审裁据此关闭这条观测图遍历机制，但不把 v252 升格为科学 FAIL，也不证明所有批次遍历或暖启动都不可能。

观测图与固定中点遍历的名义账均为每套 rig `496A+464A^T`，K16 为 `528A+528A^T`，算术差 `9.0909%`。由于独立验证不闭合、primary 绝对门未过且 reference 不充分，这不是有效 exact-call 减少，也不是 wall time 或 RSS 结果。

不重跑、不放宽容差，也不扩展 alternate anchor、learned ordering、beam search、Case 13/18 补考、大模型或 GPU。`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v252: independent numeric agreement fails; observation-graph traversal remains INCONCLUSIVE and closes

## Motivation

v251 shows that directly composing a frame-zero linear solution with later causal warm starts does not preserve matched accuracy. v252 asks a physically different question: can a deterministic observation graph avoid difficult cold starts by computing framewise cosine distances from the current batch's 2D observations, selecting the observation medoid, traversing with a fixed Prim rule, and applying unchanged FIFO16 plus geometry-Jacobi PCGLS K14 along that order?

Before execution, the formal contract fixes the unique graph, medoid and tie-break rules, 13 rigs by 33 frames, the same four metric gates, the K16 reference, an equal-cost fixed-midpoint traversal control, and the call ledger. Formal and independent implementations read only sealed observations and reported geometry before their prediction barriers. Truth and parent metrics open only after traversal orders are sealed for scoring. The two programs independently rebuild the graph and solver chain, with no tolerance relaxation or repeat run.

## Why the authoritative result must remain inconclusive

The independent graph edge weights differ from formal by at most `5.55e-16`; medoids, traversal orders, and parent nodes agree exactly, and the complete summary difference remains within its frozen limit. Three preregistered numeric agreement checks nevertheless fail:

- field relative difference is `1.1990e-7` against a `2e-8` limit;
- metric absolute difference is `8.6125e-8` against a `2e-8` limit;
- residual relative difference is `8.1713e-6` against a `2e-7` limit.

The independent status is therefore `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_OBSERVATION_GRAPH_TRAVERSAL_V252`, and the scientific decision is `INCONCLUSIVE_INVALID_CASE19_OBSERVATION_GRAPH_TRAVERSAL_V252`. Discrete agreement cannot replace the preregistered floating-point closure, and the tolerance cannot be relaxed after observing the result.

## Post-open discrete diagnostic and route action

The following counts are used only to decide whether further investment in this mechanism is justified; they are **not independently validated scientific performance results**. Both implementations report `427/429` absolute cells and `11/13` complete absolute rigs for the observation-graph primary. The same two misses occur at frame 6 in rigs 0 and 5, only on the interior-gradient gate. Those frames are the graph medoid anchors, so their anchor K16 fields already fail the absolute gate.

More importantly, the equal-cost fixed-midpoint control, which does not use observation similarity, reaches `429/429` absolute cells and `13/13` complete rigs in both implementations. The sealed chronological control reaches `428/429` and `12/13`, while the K16 reference reaches only `417/429` and `9/13`. The current graph rule therefore neither clears its own absolute gate nor excludes an equal-cost explanation. A post-open adjudication closes this observation-graph mechanism without upgrading v252 to a scientific FAIL or claiming that all batch traversals or warm starts are impossible.

The nominal ledger is `496A+464A^T` per rig for both graph and midpoint traversal, versus `528A+528A^T` for K16, an arithmetic difference of `9.0909%`. Because independent validation does not close, the primary misses the absolute gate, and the reference is inadequate, this is not effective exact-call reduction and not a wall-time or RSS result.

There is no rerun, tolerance relaxation, alternate-anchor search, learned ordering, beam search, Case 13/18 rescue, larger model, or GPU run. `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.
