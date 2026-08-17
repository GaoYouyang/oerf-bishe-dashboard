# v155：三条失败轨迹不是同一种问题

更新：2026-08-17

## 先说结论

v154 已经确认：把当前可用的十条完整公开训练轨迹全部纳入后，跨轨迹支持仍只有 87.07%，其中 `p45-s05`、`p58-s03`、`p58-s05` 未过门。v155 没有再训练模型，而是把这三条失败轨迹的最近邻距离拆成四块：观测、K1 residual、K1 dual state 和报告几何。

结果表明，三条轨迹不是同一个原因：

- `p45-s05` 的不支持距离中，观测 + residual + dual state 合计占 **71.91%**，报告几何占 **28.09%**；
- `p58-s03` 的 state / geometry 占比为 **61.34% / 38.66%**；
- `p58-s05` 的 state / geometry 占比为 **61.41% / 38.59%**。

因此准确判决是 `ROOT_CAUSE_MIXED_SUPPORT_GAP_V155`：`p45-s05` 更偏向状态或形态差异，两个 p58 失败则同时包含明显的几何与状态差异。不能把当前缺口简单写成“相机位置没对齐”，也不能写成“只需要时间模型”。

## 为什么做这一步

v154 之后继续扩大同一模型没有依据，但直接停在“数据不够”也过于粗糙。v155 的价值是回答下一份新信息到底要补什么：只补几何、只补时间，还是必须同时获得真实三维场与对应投影。

审计沿用 v154 的十条 post-open 公开训练轨迹、61,050 个 active-camera rows、complete-trajectory leave-one-out 和 fold-train-only normalization。它不读取 Krylov target 或 CFD truth，不拟合 predictor，不做物理 replay，也不打开 validation/test；新增调用账为 `0A+0A^T`。

## 数字怎样读

三条失败轨迹分别有 `5,080 / 1,366 / 786` 个不支持行。将每个查询与最近训练邻居之间的 45 维标准化距离按四个冻结特征块分解后，可以看到：

| 失败轨迹 | observation | K1 residual | K1 dual | reported geometry | state 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p45-s05 | 23.66% | 24.59% | 23.66% | 28.09% | 71.91% |
| p58-s03 | 21.71% | 17.91% | 21.71% | 38.66% | 61.34% |
| p58-s05 | 20.17% | 21.07% | 20.17% | 38.59% | 61.41% |

选取帧 0 / 25 / 50 / 75 / 100 后，`p45-s05` 的支持率为 **25.55% / 18.35% / 16.05% / 22.93% / 1.06%**；`p58-s03` 在中段下降后又恢复；`p58-s05` 除首帧外大多接近或高于 90%。这些时间变化只是描述性证据，不能证明存在可直接利用的时间输运规律。

## 独立复算

独立第二实现重新计算所有距离、四块平方距离、占比和科学判决。六项科学检查全部通过：

- 重建总距离最大差：`1.78e-15`；
- 分块平方距离最大差：`1.42e-14`；
- 分块占比最大差：`2.22e-16`；
- 科学判决逐项一致。

有 171 个“最大贡献块”文字标签不完全一致，但这些位置都是 observation 与 K1 dual 在 `1e-12` 内并列，属于浮点求和次序造成的同值 tie；连续数值和科学结论均一致。这里把它明确披露，而不是把标签差异藏掉。

## 路线动作

当前公开跨轨迹系数预测路线继续关闭。不授权：

- 只做 geometry warp；
- 只做 temporal model；
- 重复已经失败的 residual joint least-squares control；
- 用更大的 CNN / FNO / UNO / DeepONet 挽救；
- 租 GPU。

真正能改变判断的下一份信息，是可被精确解码的实验三维场与其对应二维位移投影，或者真正更广的公开工况。只有这些信息到位，才能把当前代理问题推进到真实 BOST 的 forward、matched-accuracy 和迁移验证。

证据边界：这是一项 post-open、target-free 的失败归因审计，不是重建、学习算法、调用减少、速度、外部泛化或真实 BOST 成功。`algorithm_breakthrough=false`。

---

# v155: the three failed trajectories do not share one cause

Updated: 2026-08-17

## Verdict

v154 established that support remains only 87.07% after all ten available full public training trajectories are included, with `p45-s05`, `p58-s03`, and `p58-s05` still below the frozen gate. v155 fits no new model. It decomposes each failed nearest-neighbour distance into four frozen blocks: observation, K1 residual, K1 dual state, and reported geometry.

The three failures have different mixtures:

- state accounts for **71.91%** of the unsupported distance in `p45-s05`, versus **28.09%** for reported geometry;
- state / geometry shares are **61.34% / 38.66%** for `p58-s03`;
- state / geometry shares are **61.41% / 38.59%** for `p58-s05`.

The exact decision is `ROOT_CAUSE_MIXED_SUPPORT_GAP_V155`. The p45 failure is more state- or morphology-dominated, while both p58 failures contain substantial geometry and state mismatch. The evidence does not support a geometry-only or temporal-only explanation.

## Method and boundary

The audit preserves the v154 ten-trajectory post-open roster, 61,050 active-camera rows, complete-trajectory leave-one-out evaluation, and fold-only normalization. It reads neither a Krylov target nor CFD truth, fits no predictor, performs no physical replay, and opens no validation or test data. The new exact-call ledger is `0A+0A^T`.

An independent second implementation recomputes every distance, block contribution, share, and decision. All six scientific checks pass. Maximum total-distance, block-squared-distance, and block-share differences are `1.78e-15`, `1.42e-14`, and `2.22e-16`. The 171 exact dominant-label mismatches are all ties within `1e-12`; continuous values and the scientific decision agree.

## Route action

The public cross-trajectory coefficient-prediction route remains closed. A geometry-only warp, temporal-only model, repeated residual least squares, larger neural model, or GPU rental is not authorized as a rescue. Progress now requires physically different information, especially an exactly decodable experimental three-dimensional field with corresponding two-dimensional displacement projections, or genuinely broader public operating conditions.

This is a post-open target-free failure-attribution audit, not a reconstruction, learned algorithm, exact-call saving, speed, external-generalization, or real-BOST result. `algorithm_breakthrough=false`.
