# v275：固定体平流暖启动完成，但独立数值闭环与 reference 充分性均未通过

日期：2026-08-28

## 1. 问了什么

Case 19 的 33 个时间片带有官方入口速度和时间戳。v275 结果前只冻结一个物理假设：把上一帧自己的重建场沿 source-x 方向做固定的因果半拉格朗日平流，再把这个场交给未修改的 geometry-Jacobi PCGLS K14。平流只使用部署时可见的上一帧重建、官方速度、时间间隔和报告几何；不读取当前真值，不搜索速度、方向、插值、边界、深度或阈值。

同样的平流加 PCGLS K16 是 matched reference。对照包括不平流的 previous-self、FIFO、Zero-PCGLS/CGLS、BP-PCGLS/CGLS 和 Zero-PCGLS K16。Case 19 已经开封，所以这只能是 post-open 机制诊断，不能称前瞻外门。

## 2. 执行边界

formal 的全部有效性检查为真，预测进程没有读取密度真值。第一次独立尝试完成 416/416 个预测后，在真值评分前主动停止：预测进程安装禁读守卫后，又试图为完整性检查重新哈希真值文件。该尝试没有计算 truth metric 或科学判决，原样保留为评分前工程失效。

v275.1 只修这个隔离边界：准备进程额外封存一份只含 observation 与 source grid 的 deployment-visible release；预测进程不再读取真值文件或真值清单，完整真值树只由退出后的新评分进程核验。科学候选、数据、门、调用账和 formal 输出均未改变，失败预测也没有复用。

## 3. 独立复算结果

修复后的独立第二实现通过 **26/31** 项检查。内部有效性、相机乱序、物理 replay、调用账、离散判决、control roster 和汇总均一致；但五项结果前冻结的连续数值门失败：

| 检查 | 观察差 | 冻结上限 |
| --- | ---: | ---: |
| transport 相对差 | `3.3268e-9` | `1e-12` |
| transport audit 绝对差 | `1.2154e-10` | `1e-12` |
| field 相对差 | `1.5192e-8` | `1e-8` |
| residual 相对差 | `1.5673e-6` | `1e-8` |
| metric 绝对差 | `1.0219e-8` | `1e-8` |

汇总最大差 `2.1292e-9` 低于 `1e-8`，离散判决也一致，但它们不能替代连续数值闭环。因此权威独立状态是 `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_BULK_ADVECTION_WARM_V275`，科学判决是 `INCONCLUSIVE_INVALID_CASE19_BULK_ADVECTION_WARM_V275`。

## 4. 只能作诊断的数字

formal 与 independent 的离散判决一致：主候选绝对安全 **428/429** 单元、**12/13** 完整 rig；平流 K16 reference 也只有 **428/429** 与 **12/13**，自身不充分。主候选相对该 reference 的 matched 计数只有 **13/429** 单元与 **0/13** 完整 rig；没有同价或更便宜 control 通过完整合同。

主候选每条序列的逻辑账为 `496A+464A^T`，reference 为 `560A+528A^T`。由于独立数值门、reference 充分性和 matched-accuracy 都未成立，这个算术差不是有效 exact-call 减少，也不是 wall、RSS 或速度证据。

## 5. 结论与下一步

固定的“官方速度整体平流 + previous-self + PCGLS K14”机制关闭：不事后放宽容差，不改速度、方向、插值、边界或迭代深度，也不以 GPU、CNN、FNO、UNO 或 DeepONet 挽救。这个结果不证明局部速度场、非刚性输运或整条 C 路线数学上不可能。

下一步只接受两类新信息：工况配对的真实二维双分量 BOST 数据及标定/噪声/基线，或一个与固定整体平流、全局二次正则和旧低秩路线物理上真正不同、结果前唯一冻结且可证伪的机制。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v275: fixed bulk-advection warm starting completes, but independent numerical closure and reference adequacy both fail

Date: 2026-08-28

## Question and contract

Case 19 provides an official inlet speed and timestamps for 33 snapshots. v275 preregisters one physical hypothesis: advect the previous frame's own reconstruction causally along source-x with a fixed semi-Lagrangian step, then pass that field to unchanged geometry-Jacobi PCGLS K14. The transport reads only the deployment-visible previous reconstruction, official speed, time interval, and reported geometry. It does not read current truth or search speed, sign, interpolation, boundary rule, depth, or thresholds.

The same transport followed by PCGLS K16 is the matched reference. Controls include previous-self without transport, FIFO, zero-start PCGLS/CGLS, normalized-BP PCGLS/CGLS, and zero-start PCGLS K16. Because Case 19 is already opened, this is a post-open mechanism diagnostic rather than a prospective external gate.

## Execution boundary

All formal validity checks are true and the formal predictor excludes density truth. The first independent attempt computes all 416 predictions but fails closed before truth scoring: after installing its truth-open guard, the predictor tries to re-hash the truth file for a complete-tree integrity check. It computes no truth metric or scientific decision and is retained as a pre-scoring engineering failure.

v275.1 repairs only this isolation boundary. Preparation now emits a separate deployment-visible release containing observations and the source grid; prediction reads neither truth values nor truth inventory, while a fresh scoring process checks the complete truth tree after prediction exits. The scientific mechanism, data, gates, call ledger, and formal output are unchanged, and failed predictions are not reused.

## Independent recomputation

The repaired independent implementation passes **26/31** checks. Internal validity, camera permutation, physical replay, call accounting, discrete decision, control roster, and summaries agree. Five preregistered continuous numerical gates fail: transport relative disagreement is `3.3268e-9` against `1e-12`; transport-audit absolute disagreement `1.2154e-10` against `1e-12`; field relative disagreement `1.5192e-8` against `1e-8`; residual relative disagreement `1.5673e-6` against `1e-8`; and metric absolute disagreement `1.0219e-8` against `1e-8`. Summary disagreement of `2.1292e-9` passes its `1e-8` limit, and discrete decisions agree, but neither substitutes for continuous closure.

The authoritative independent status is `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_BULK_ADVECTION_WARM_V275`, and the scientific decision is `INCONCLUSIVE_INVALID_CASE19_BULK_ADVECTION_WARM_V275`.

## Diagnostic-only counts

Formal and independent discrete decisions agree: the primary is absolute-safe in **428/429** cells and **12/13** complete rigs. The transported K16 reference also reaches only **428/429** and **12/13**, so it is inadequate. The primary matches that reference in only **13/429** cells and **0/13** complete rigs. No equal-or-cheaper control passes the full contract.

The primary's logical sequence ledger is `496A+464A^T`, versus `560A+528A^T` for the reference. Because independent numerical closure, reference adequacy, and matched accuracy all fail, this arithmetic difference is not effective exact-call reduction and provides no wall, RSS, or speed evidence.

## Decision

The fixed official-speed bulk-advection previous-self K14 mechanism closes. There is no post-hoc tolerance relaxation or adjustment of speed, sign, interpolation, boundary rule, or depth, and no GPU, CNN, FNO, UNO, or DeepONet rescue. This does not prove local velocity fields, nonrigid transport, or the wider C route mathematically impossible.

Continuation requires either condition-matched real two-component BOST data with calibration, noise, and an accepted baseline, or one physically distinct, uniquely preregistered, falsifiable mechanism.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `curved_ray_validated=false`, and `real_bost=false`.
