# v112.4 Temporal CNN 父对照：当前 CNN 路线被同价父控制拒绝

> **证据状态：** `PASS_INDEPENDENT_RECOMPUTATION_CNN_PARENT_FAMILY_V112_4`  
> **科学判决：** `FAIL_V112_3_CNN_PARENT_FAMILY_REJECTS_LEARNED_ADVANTAGE`  
> **边界：** 这是已开封 PoolFire、已知九视角 straight-ray 代理上的父对照负结果；不是外部泛化、真实 BOST、资源结果或论文结论。

## 中文摘要

v111 的坐标条件 warm initializer 在冻结的五条 PoolFire 轨迹、三套几何和三个随机种子上曾通过 Formal Stage A。v112.4 随后提出更严格的问题：一个删去显式 pose、map 与 camera token、只保留时间邻帧输入的最小 CNN，是否已经能在相同的 `2A + 2A^T` 在线预算下解释该信号？

答案是：**当前候选不能稳定胜过这个 temporal CNN 父对照。** 正式评分和独立复算覆盖三个同种子配对、各 `2970` 个 candidate/control 单元与 `270` 个 deployment-only predictions。seed `1103` 没有被父对照拒绝；但 seed `2203` 和 `3301` 都在五条轨迹的 p90/worst 尾部比较失败。因此 family-level 规则拒绝“当前 learned advantage”主张。

| 同种子配对 | paired field 更优比例 | paired interior-gradient 更优比例 | p90/worst 尾部 | 判决 |
|---|---:|---:|---:|---|
| 1103 | 96.26% | 91.21% | 通过，最坏差 -0.00136 | 未拒绝 |
| 2203 | 84.24% | 78.18% | 失败，最坏差 +0.02766 | 拒绝 |
| 3301 | 89.19% | 77.27% | 失败，最坏差 +0.01145 | 拒绝 |

这里的“最坏差”是候选减 temporal parent 的冻结 trajectory-level p90/worst 汇总分量；正值表示候选在至少一个关键尾部指标上更差。三个 temporal controls 自身都通过五条 Stage-A trajectory gate，故不能把拒绝归因于一个失效基线。

## 独立复算

独立程序没有导入正式 CNN 预测编排器、预测 worker 或 scorer。它从封存 checkpoint 与 deployment-visible bundle 重新实现 CNN 推理、warm K1、指标、尾部、同种子比较与 family 判决。结果为：

- `270` 个 prediction、`2970` 个 candidate cell 与 `2970` 个 temporal-control cell 均被重放；
- 正式数值、prediction field 与 prediction residual 的最大绝对差均为 `0`；
- 封存 prediction tree 和 formal-score tree 在验证前后保持不变；
- artifact/API 级 truth-mutation noninterference 已验证；process-level never-read 与端到端 physics independence 未证明。

## 这改变什么

这不是“神经网络没有用”的结论，也没有证明不存在更好的 BOST warm start。它只关闭了**当前坐标条件 CNN 表示及其 FNO 延伸路线**：既然更普通的 temporal CNN 已在两个种子下否定稳定优势，就不能再靠训练 pose、reference 或更大的 FNO 来挽救同一主张。

下一次允许的科学尝试必须是结果前冻结、物理上不同且先经便宜控制检验的机制，而不是更大 CNN。fresh wall/RSS、独立公开反应流外门、曲折光线、真实 BOST 和论文成功仍全部未证。

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `real_bost=false`

---

# v112.4 Temporal CNN parent: the current CNN route is rejected by an equal-cost parent control

The v112.4 test asks whether a minimal temporal CNN, with explicit pose, map, and camera-token inputs removed, can already account for the v111 coordinate-conditioned warm-start signal under the same online `2A + 2A^T` budget.

It can reject the current learned-advantage claim at the family level. Across three same-seed comparisons, seed 1103 is not rejected, but seeds 2203 and 3301 fail the frozen p90/worst trajectory-tail criterion. The temporal controls themselves pass every Stage-A trajectory gate, so the failure cannot be dismissed as a broken baseline.

An independent implementation rebuilds inference, warm K1, metrics, tails, pairing, and the family decision from sealed artifacts. It replays 270 deployment-only predictions and 2970 candidate/control cells; the maximum formal numeric, field, and residual differences are all zero, and the sealed prediction and score trees remain unchanged.

This closes the current coordinate-conditioned CNN path and cancels the planned FNO continuation. It does not establish that all warm starts fail, nor does it provide external generalization, resource, curved-ray, real-BOST, or paper-success evidence.
