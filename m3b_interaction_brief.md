# M3B 4D BOST 交互实验一页沟通 brief

生成日期：2026-07-10

用途：给何远哲师兄快速判断“低秩时序子问题是否值得继续做，以及下一步该接真实数据还是做自适应 rank 选择”。这不是 TDBOST 复现声明。

> 状态更新：本页提出的无真值 rank selector 已在 `m3b_family_selector_dashboard.html` 完成跨 4 个 synthetic morphology 的 nested-LOFO 实验。最新给师兄的材料请优先读 `m3b_family_selector_brief.md`；本页保留为单 phantom 交叉工作域的前序证据。

## 1. 这轮到底做了什么

- 交叉因素：5 ranks × 5 noise levels × 4 view counts × 4 dynamics。
- 重复：8 个同 seed 配对观测噪声 realization。
- 环境单元：80；观测单元（含 seed）：640。
- 记录：640 条 framewise、3,200 条 low-rank、3,200 条配对比较，共 3,840 条方法记录。
- 统计：每格均值、标准差、胜率和 Student-t 95% CI，n=8 时 `t(7)=2.365`。
- 复现：快速网格独立运行两次，除 reconstruction/postprocess timing 外，CSV 和共享报告数值一致。

## 2. 最重要的四个结果

### A. rank 3 是稳健默认值，不是逐格最优

- 80 个环境的 best rank：rank 2 为 27 格，rank 3 为 20 格，rank 5 为 24 格，rank 8 为 9 格，rank 1 为 0 格。
- rank 3 的全局平均 field L2 最低：0.3691。
- 相对逐格 oracle best：rank 3 mean regret 0.79%，P95 regret 2.21%，max 2.52%，均为五个固定 rank 中最低。
- 因此应写成“rank 3 是当前 synthetic design 的稳健固定默认值”，不能写成“rank 3 在所有工况最优”。

### B. noise 与 views 会共同改变最优 rank

- noise=0：16 格中 9 格选择 rank 8；充分视角、低噪时过度压缩会伤害结果。
- noise=0.07：12/16 格选择 rank 5。
- noise=0.28：12/16 格选择 rank 3。
- noise=0.42：11/16 格选择 rank 2。
- 3 views 的 20 个环境全部选择 rank 2，说明强欠定区需要独立 rank 规则。

### C. rank 3 有明确的负收益区

- Student-t 区间下，rank 3 的 field L2：61/80 格显著正收益，19/80 格显著负收益，0 格跨零。
- 负收益主要集中于 noise=0 的 5/7/9 views，以及 noise=0.07 的部分 7/9 views。
- 原来的“无噪声时低秩伤精度”应修正为：“无噪声且视角较充分时 rank 3 伤精度；3-view 强欠定时仍可能略有 field 收益。”

### D. field 变好不等于物理量更可信

- 400 个环境×rank 单元中，116 个出现 field L2 正收益但 mass-trace 负收益，占 29.0%。
- 对 rank 3：61 个 field 正收益格中有 37 个 mass-trace 变差，占 60.7%。
- rank 3 的 mass trace 多数要在 5-9 views、noise≈0.28 后才转正；3 views 在本扫描中始终未转正。

## 3. 对毕设题目的含义

最合适的本科子问题不再只是“试几个 rank”，而是：

> 面向少视角 4D BOST 的可观测自适应时序 rank 选择与物理一致性评价

最小版本：

1. 继续使用 synthetic 交叉设计建立 operating map。
2. 设计不依赖真值的 rank selector：held-out view residual、奇异谱 energy/elbow、centroid/mass trace 联合评分。
3. 与固定 rank 2/3/5、framewise 和 low-rank+sparse baseline 对照。
4. 接一小段真实 4D BOST 数据，只验证 failure signature 和选择稳定性，不承诺完整复现 TDBOST。

### 与 T16 神经算子的关系

- M3B 给 T16 提供 view/noise/dynamics 的 condition-level 实验设计、失败格和多指标反例；T16 不能只随机拆 seed 或只报告 pooled mean。
- M3B 的 temporal SVD rank 不等于 FNO Fourier modes、latent width 或 neural-operator rank，不能拿 rank 3 直接指定网络超参数。
- 静态 T16 应先比较 SIRT、U-Net、Residual FNO 与 NeRIF；只有静态模型跑稳后，M3B fixed-rank SVD 才进入 temporal operator baseline。
- M3B 噪声值是合成乘子，真实 T16/OERF 数据必须用 displacement residual 或重复测量重新标定。

## 4. 请师兄直接判断的 6 件事

1. 组里是否认可把“自适应 rank/capacity 选择”作为本科 4D BOST 子问题？
2. 真实 TDBOST 更看 test-view、field、频谱、事件峰值，还是积分量？
3. 是否存在可作为 physical trace 的 integrated density/temperature、centroid 或同步外部测量？
4. 能否给 18-30 帧、5/7/9 views 的小片段，并保留一个 withheld/test view？
5. 本页的 noise multiplier 应如何用真实 displacement residual 标定？
6. 下一轮优先做 bias magnitude、sampling rate/exposure blur，还是先做 synthetic-to-real failure audit？

## 5. 解释边界

- 24×24×10、18 帧、单一 phantom 族、简化直线投影、temporal SVD。
- 8 个 seed 只覆盖观测噪声，不覆盖不同 phantom、几何和实验工况。
- global affine alignment 和 oracle best rank 使用真值，不能直接部署。
- 该实验用于形成可检验问题和数据请求，不等价于 TDBOST 论文方法或真实 OERF 结果。

## 6. 入口

- 渲染页：`m3b_interaction_dashboard.html`
- 脚本：`demo_m3b/run_m3b_interaction_sweep.py`
- 报告：`demo_m3b/results/interaction_report.json`
- 400 格统计：`demo_m3b/results/interaction_cell_summary.csv`
- 80 格 rank 选择：`demo_m3b/results/interaction_rank_selection.csv`
- 3,200 配对结果：`demo_m3b/results/interaction_paired.csv`
