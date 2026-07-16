# T16 三维重建 x 算子学习：师兄审核 brief

更新日期：2026-07-10

> 本页保留第一 checkpoint 的单次运行证据。后续三随机种子 residual/loss/capacity 消融已经完成，最新方向判断请先读 `t16_operator_ablation_review_brief.md`；两份材料分别回答“模型能否跑通”和“收益来自哪里”。

## 一句话结论

已经不只是提出“用 FNO 做三维重建”：目前完成了一个可重复的 synthetic BOST closure test，使用同一物理提升输入比较传统重建、3D U-Net 和官方 NeuralOperator 2.0 的 residual 3D FNO，并将 IID、少视角、强噪声、联合偏移和全新 phantom family 分开验收。

当前建议题目仍是：

> 面向少视角背景纹影层析的物理提升残差神经算子与跨工况三维折射率场重建。

但需要师兄确认：组里所说的“算子学习”究竟是多视角观测到三维场的 inverse operator，还是三维/四维场之间的 evolution operator。

## 已完成的最小证据

- 168 个三维 paired samples；体素为 `8 x 16 x 16`。
- Gaussian/flame 为训练族；thin-front 整族只在 OOD 测试出现。
- 输入包含 physics lift、support、视角数、噪声和三维坐标共 7 个 channel。
- 物理提升标定只在训练集拟合一次，之后锁定；没有逐样本真值对齐。
- FNO 与 U-Net 都预测 lift 的 residual，并使用 field、gradient、BOST reprojection 和 boundary loss。
- 每个样本同时记录 field、gradient、observed/held-out reprojection、mass 和 centroid。
- 固定种子独立复跑后，逐样本 CSV 字节级一致。

## 核心结果

| 方法 | 参数量 | IID field rel. L2 | IID held-out rel. L2 |
| --- | ---: | ---: | ---: |
| physics lift | 0 | 0.4573 | 0.5203 |
| residual 3D U-Net | 86,633 | 0.2706 | 0.3191 |
| residual 3D FNO | 43,363 | 0.2321 | 0.2766 |

在五个测试域里，FNO 的平均 field error 都低于 U-Net。逐样本比较显示：

- IID：FNO 相对 U-Net 的 field error 平均相对改善 18.2%，20 个样本中胜 18 个。
- noise OOD：平均相对改善 15.5%，16 个样本全部获胜。
- view OOD：field 平均相对改善 13.4%，16 个样本中胜 13 个。
- joint OOD：field 平均相对改善 11.1%，16 个样本中胜 15 个。
- family OOD：平均相对改善仅 3.9%，20 个样本中胜 19 个，但绝对误差仍高达 0.6789。

## 最重要的反例

三视角 OOD 中，FNO 的 field error 是 0.5081，优于 U-Net 的 0.5824；held-out reprojection 均值只从 0.5466 降到 0.5420。配对均值差为 `-0.0046 +/- 0.0218`，95% 区间跨零；joint OOD 也只有 `-0.0024 +/- 0.0280`。而且部分样本会出现 field 更好、held-out 更差，因此当前不能声称 FNO 在少视角下物理一致性更强。

这说明：

1. field-space 监督与未观测视角一致性可能分叉；
2. synthetic truth 上的排名不能替代真实数据模型选择；
3. 少视角不是再调几个 Fourier modes 就能解决，可能需要 geometry-aware encoding、data consistency layer 或 test-time NeRIF refinement。

## 当前不能声称的事情

- 不能声称已经复现 NeRIF、TDBOST 或组内真实几何。
- 不能声称 FNO 已证明跨分辨率；当前没有独立高分辨率生成器测试。
- 不能把 synthetic noise multiplier 当实验像素误差或偏折角。
- 不能根据 168 个 toy samples 宣称 FNO 普遍优于 U-Net。
- 不能把 field truth 参与的训练/选择直接迁到无真值真实数据。

## 希望师兄审核的七个决策

1. 组内目标是 `projection/displacement -> 3D field`，还是 `3D history -> 4D evolution`？
2. 网络应吃 raw BOS image、位移场、sinogram，还是 SIRT/NeRIF 初值？
3. 当前“physics lift + residual FNO”是否符合组内希望的模型形态？
4. 是否能提供批量 paired simulation 或历史重建结果？样本、视角、分辨率、几何有多大？
5. 真实数据最可信的无真值指标是什么：test camera、重投影、边界、积分量还是其他诊断？
6. 更希望把算子作为最终快速重建器，还是作为 NeRIF/TDBOST 的 initialization/amortization？
7. 本科阶段应该优先做几何/数据一致性，还是比较 FNO、DeepONet、operator transformer 架构？

## 收到回答后的分支

### A. 师兄要 inverse operator

保留本题，下一步做 direct-vs-lift、projection-loss ablation、分辨率迁移和组内几何适配；DeepONet 只做第二模型。

### B. 师兄要 evolution operator

把当前静态 FNO 作为每帧空间编码 baseline，输入改成历史三维场或投影序列；与 framewise、M3B temporal SVD、蔡组 JFM 2019 和 TDBOST 比较。必须按完整时间段/工况划分，不能随机拆帧。

### C. 师兄要“重建 + 算子”结合

推荐两阶段：operator 一次前向给三维初值，NeRIF 用 BOST forward loss 做少量 test-time refinement。论文问题变成“amortized initialization 是否减少时间、失败率并保持 held-out 物理一致性”。

## 建议最低论文成果

1. 一个能接组内 geometry/data 的 paired-data manifest。
2. 传统重建、3D U-Net、residual FNO 三个公平 baseline。
3. 至少 direct/lift 与 with/without reprojection 两组 ablation。
4. IID、view/noise OOD、family OOD 和 resolution transfer。
5. field 与 held-out physical metric 的分叉分析。
6. 一个真实样例或清楚标注的 synthetic-to-real failure audit。
