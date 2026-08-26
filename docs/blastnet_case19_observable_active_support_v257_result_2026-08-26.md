# v257：可观测 active-support correction 因 residual 独立一致性越界而保持不确定

## 为什么做

v256 关闭后，v257 检验一个物理上不同、只读取部署可见观测与报告几何的局部非线性机制。它先用 zero-start geometry-Jacobi PCGLS K1 得到种子场，再以 `|x1| / sqrt(inverse_diagonal)` 定义 geometry-whitened 分数；按固定平坦索引打破并列，选择刚好覆盖 95% 分数平方能量的最小 active support，并只做一次六邻域膨胀。随后在该 support 上运行带自伴随 masked mean-zero projection 的 CGLS K11 correction，最后接一轮未修改的 full-field geometry-Jacobi PCGLS K1。

结果前固定了 support 阈值、并列规则、膨胀次数、mask、均值 gauge、K1→K11→K1 深度、四个同数据经典 control、K16 reference、四指标门、调用账和独立数值容差。候选不读取 CFD 真值、时间、rig 或失败标签，训练参数为 0。

## 唯一独立失败门

正式运行完成 13 个已开封 Case 19 九相机首帧并封存物理 replay。完全独立的第二实现通过 `23/24` 项冻结检查；唯一失败项是 `observation_normalized_residuals_agree`。最大差为 `3.08360e-8`，高于冻结的 `2.00000e-8`，即 `1.54180×`。

support mask 与 selected count 逐项完全一致，captured energy 与 score energy 差分别只有 `3.22e-15` 和 `5.78e-15`。final field 相对差为 `7.68e-9`，逐单元指标与汇总最大差为 `5.38e-10 / 1.43e-9`，均在各自 `2e-8` 冻结界内。可是这些通过项不能覆盖唯一明确越界的连续 residual 门。

因此独立状态是 `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_OBSERVABLE_ACTIVE_SUPPORT_FRAME_ZERO_V257`，科学判决是 `INCONCLUSIVE_INVALID_CASE19_OBSERVABLE_ACTIVE_SUPPORT_FRAME_ZERO_V257`。正式与独立科学数组都不可用于性能解释；本页不发布也不解释无效合同下的通过计数。

## 成本与路线动作

primary 的名义单帧账为 `14A+13A^T`，K16 reference 为 `16A+16A^T`，算术合计差为 `15.625%`。由于独立合同无效且只运行了首帧，这不是有效 exact-call 减少，也不授权完整 429 单元序列、wall time、RSS、外部门、训练或 GPU。

当前 observable active-support correction 关闭；不放宽容差、不重跑、不搜索 95% 阈值、膨胀次数、mask 或深度。这个结论不关闭整条 C 路线。下一步只能来自新的配对真实 BOST 信息，或一个结果前唯一冻结、部署可见、可独立证伪且物理上真正不同的机制。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v257: observable active-support correction remains inconclusive after residual agreement fails

## Motivation

After v256 closes, v257 tests a physically distinct local nonlinear mechanism that reads only deployment-visible observations and reported geometry. A zero-start geometry-Jacobi PCGLS K1 seed defines the geometry-whitened score `|x1| / sqrt(inverse_diagonal)`. With flat-index tie-breaking fixed in advance, the method takes the smallest active support that captures 95% of score-squared energy and applies exactly one six-neighbor dilation. It then runs a CGLS K11 correction under a self-adjoint masked mean-zero projection, followed by one unchanged full-field geometry-Jacobi PCGLS K1 step.

The support threshold, tie-break, dilation count, mask, mean gauge, K1-to-K11-to-K1 schedule, four same-data classical controls, K16 reference, four metric gates, call ledger, and independent numerical tolerances were fixed before results. The candidate reads no CFD truth, time, rig, or failure label and has zero trainable parameters.

## The sole independent failure

Formal execution completes and seals physical replay for the 13 opened Case 19 nine-camera frame-zero cells. A fully independent implementation passes `23/24` frozen checks. The sole failure is `observation_normalized_residuals_agree`: maximum disagreement is `3.08360e-8` against a frozen `2.00000e-8` limit, or `1.54180x`.

Support masks and selected counts agree exactly. Captured-energy and score-energy differences are only `3.22e-15` and `5.78e-15`. Final-field relative disagreement is `7.68e-9`, while maximum cell-metric and summary differences are `5.38e-10 / 1.43e-9`; all remain within their `2e-8` limits. These passing checks cannot override the sole explicit continuous residual failure.

The independent status is therefore `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_OBSERVABLE_ACTIVE_SUPPORT_FRAME_ZERO_V257`, with scientific decision `INCONCLUSIVE_INVALID_CASE19_OBSERVABLE_ACTIVE_SUPPORT_FRAME_ZERO_V257`. Neither the formal nor independent scientific arrays are admissible for performance interpretation, so this page does not publish or interpret pass counts from the invalid contract.

## Cost and route action

The nominal primary frame ledger is `14A+13A^T`, versus `16A+16A^T` for K16, an arithmetic combined-call difference of `15.625%`. Because the independent contract is invalid and covers frame zero only, this is not effective exact-call reduction and does not authorize the full 429-cell sequence, wall time, RSS, an external gate, training, or GPU use.

The observable active-support correction closes without relaxing tolerance, rerunning, or searching the 95% threshold, dilation count, mask, or depth. This does not close the C route. Any next step requires new paired real-BOST information or one uniquely preregistered, deployment-visible, independently falsifiable, and physically distinct mechanism.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.
