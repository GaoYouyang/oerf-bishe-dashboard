# v299: Exact-Discrete-Observation Counterfactual / 精确离散观测反事实

**Date / 日期:** 2026-09-30  
**Evidence grade / 证据等级:** independently recomputed post-open mechanism diagnostic / 独立复算的已开封机制诊断

## 中文

### 问题与设置

v298 在同一批已开封样本上发现：连续方式生成的二维观测与实际迭代使用的离散算子有明显差异；在总预算 `17A+17A^T` 时，learned warm 的场和全梯度中位误差高于 Zero，但观测残差略低。v299 只把这 25 行的观测替换为 `y=A x_truth`，其中 `A` 是冻结的离散算子；数据行、已训练好的预测器、BP 起点公式、CGLS 和检查点均不变。真值只用于构造这项已开封反事实观测与最终评分，没有训练、调参或打开新数据。

### 结果

- 反事实观测相对原连续合成观测的差异 p50/p90/worst 为 `0.239253 / 0.290883 / 0.305524`。
- 总预算 `2A+2A^T` 时，learned warm 的四项误差（场、全梯度、内部梯度、观测残差）都在 `25/25` 行低于 Zero。
- 总预算 `17A+17A^T` 时，learned warm 相对 Zero 的误差中位差依次为 `-0.004108 / -0.008820 / -0.003524 / -0.003397`；对应逐行更低的数量为 `20/25 / 20/25 / 19/25 / 20/25`。其中场误差中位数为 `0.129984` 对 `0.134092`，全梯度 `0.173699` 对 `0.181691`，内部梯度 `0.227713` 对 `0.231151`，观测残差 `0.019692` 对 `0.022736`（learned 对 Zero）。
- 对照 v298 原连续观测、相同预算的中位数差：场和全梯度分别为 `+0.005983`、`+0.014646`（learned 更高），内部梯度 `+0.000653`，观测残差 `-0.000694`。因此该离散一致反事实下的四指标中位方向均转为 learned 较低。
- formal 与独立第二实现的最大绝对差：算子 `1.11e-15`、观测 `4.00e-15`、初始场 `6.22e-15`、CGLS 场 `5.44e-15`、指标 `2.33e-15`；输入与 formal 输出树复核未变。

### 解释与边界

这支持“观测生成与离散反演算子不一致，可能参与了 v298 的深层指标分歧”这一解释；它不是因果证明，因为这是在已打开样本上用真值合成的新观测反事实。它没有建立新轨迹泛化、真实 BOST、可部署停止规则、matched-accuracy 下稳定减少 `A/A^T` 调用、端到端时间或 RSS 优势。它不撤销 v284 完整轨迹严格成本门失败，也不是算法突破或论文成功证据。

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.

## English

### Question and setup

On the same already-open rows, v298 found a visible difference between continuously generated 2D observations and the discrete operator used by the solver. At total budget `17A+17A^T`, the learned warm start had higher median field and full-gradient errors than Zero, despite a slightly lower observation residual. v299 changes only the observations on those 25 rows to `y=A x_truth`, using the frozen discrete operator. Rows, trained predictor, BP formula, CGLS, and checkpoints remain fixed. Truth is used only to construct this post-open counterfactual observation and score the result; there is no training, tuning, or newly opened data.

### Results

- The counterfactual observations differ from the original continuous synthetic observations by p50/p90/worst `0.239253 / 0.290883 / 0.305524`.
- At total budget `2A+2A^T`, learned warm has lower field, full-gradient, interior-gradient, and observation-residual errors than Zero on `25/25` rows for each metric.
- At `17A+17A^T`, the learned-minus-Zero median error differences are `-0.004108 / -0.008820 / -0.003524 / -0.003397`, respectively, with learned lower on `20/25 / 20/25 / 19/25 / 20/25` rows. Median field error is `0.129984` vs `0.134092`, full gradient `0.173699` vs `0.181691`, interior gradient `0.227713` vs `0.231151`, and observation residual `0.019692` vs `0.022736` (learned vs Zero).
- Under v298's original continuous observations at the same budget, the median differences were `+0.005983` for field, `+0.014646` for full gradient, `+0.000653` for interior gradient, and `-0.000694` for observation residual. Thus all four median directions are lower for learned warm under this exact-discrete-observation counterfactual.
- Maximum absolute differences between formal and independent implementations are `1.11e-15` for the operator, `4.00e-15` for observations, `6.22e-15` for initial fields, `5.44e-15` for CGLS fields, and `2.33e-15` for metrics. Inputs and the formal output tree were rechecked unchanged.

### Interpretation and limits

This supports the explanation that inconsistency between observation generation and the discrete inverse operator may contribute to v298's late-budget metric split. It is not causal proof: the counterfactual is synthesized with truth on already-open rows. It establishes no new-trajectory generalization, real BOST result, deployable stopping rule, stable reduction in `A/A^T` calls at matched accuracy, end-to-end time, or RSS benefit. It does not reverse v284's full-trajectory strict-cost failure and is not evidence of an algorithmic breakthrough or paper success.

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.
