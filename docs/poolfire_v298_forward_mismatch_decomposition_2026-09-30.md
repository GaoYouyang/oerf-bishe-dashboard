# v298: Forward-Mismatch Decomposition / 前向不匹配分解

**Date / 日期:** 2026-09-30  
**Evidence grade / 证据等级:** post-open mechanism attribution on already-open synthetic data / 已开封合成数据上的机制归因

## 中文

### 问题与设置

连续方式生成的二维观测与离散反演算子并不完全一致。v298 在五条已开封 PoolFire 轨迹的 25 个样本行上，分解这项不匹配对 CGLS 检查点的影响。比较使用已封存的 learned warm start 与 Zero，并按总 `A/A^T` 调用预算配对。三维真值仅用于事后误差归因；本结果不产生新模型，也不是前瞻性测试。

### 结果

- 将真值场代入离散算子后，预测投影与连续合成观测的相对差 p50/p90/worst 为 `0.239253 / 0.290883 / 0.305524`。
- 总预算 `2A+2A^T` 时，learned warm 相比 Zero 在场、全梯度、内部梯度和观测残差四项真值误差上，均逐行较低：每项 `25/25`。
- 总预算 `17A+17A^T` 时，learned warm 的场误差中位数为 `0.243313`，略高于 Zero 的 `0.236866`；全梯度中位数 `0.445191`，也高于 Zero 的 `0.429261`。内部梯度中位数为 `0.268684` 对 `0.266110`。与此同时，观测残差中位数略低：`0.1245` 对 `0.1254`；观测残差在 `21/25` 行较低、`4/25` 行较高。
- 形式分解与独立实现最大绝对差 `3.82e-14`；重算观测残差与既有封存汇总最大差 `1.11e-16`，输入保持不变。

### 解释与边界

结果提示，在这个合成设置中，CGLS 继续迭代可以让离散模型更贴合二维观测，却不保证三维场和梯度误差同步下降。它支持把前向模型/观测定义作为后续核查重点，但本身不是因果证明。样本与轨迹均已开封，且使用真值评分；不构成新轨迹泛化、真实 BOST、可部署停止规则、稳定算子调用节省、速度或内存优势。它不改变此前完整轨迹严格成本门未通过的结论。

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.

## English

### Question and setup

The continuously generated 2D observations do not exactly match the discrete inverse operator. On 25 rows from five already-open PoolFire trajectories, v298 attributes this mismatch across sealed CGLS checkpoints. It compares the sealed learned warm start with Zero at matched total `A/A^T` budgets. CFD truth is used only for post-open error attribution; this is neither a new model nor a prospective test.

### Results

- After applying the discrete operator to the truth field, the relative difference between its projection and the continuous synthetic observation has p50/p90/worst `0.239253 / 0.290883 / 0.305524`.
- At total budget `2A+2A^T`, learned warm has lower truth-scored field, full-gradient, interior-gradient, and observation-residual errors than Zero on every row: `25/25` for each metric.
- At `17A+17A^T`, learned warm has a slightly higher median field error (`0.243313` vs `0.236866`) and full-gradient error (`0.445191` vs `0.429261`). Median interior-gradient errors are `0.268684` vs `0.266110`. Meanwhile, median observation residual is slightly lower (`0.1245` vs `0.1254`), lower on `21/25` rows and higher on `4/25`.
- Formal and independent decompositions differ by at most `3.82e-14`; recomputed observation residuals differ from the sealed prior summary by at most `1.11e-16`, with inputs unchanged.

### Interpretation and limits

In this synthetic setting, continued CGLS iteration can improve fit to the discrete model's 2D observations without guaranteeing a simultaneous reduction in 3D field and gradient error. This motivates checking the forward/observation definition, but it is not a causal proof. The trajectories and rows are already open and truth-scored. The result establishes no new-trajectory generalization, real BOST performance, deployable stopping rule, robust operator-call savings, wall-time or memory benefit. It does not change the prior failure of the full-trajectory strict-cost gate.

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.
