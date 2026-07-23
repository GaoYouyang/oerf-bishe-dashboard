# T16 三维重建与神经算子学习路径

更新日期：2026-07-11

这条路径服务于一个具体任务：从多视角 BOST 位移或其物理提升，重建三维折射率/密度场。它不把“学会 FNO API”误当成学会逆问题。

> 基础不系统时，请从 [独立算子学习主页](operator-learning/index.html) 的 16 周关卡开始。本文保留为技术大纲，不是零基础第一页。文中“已完成”只表示仓库已有证据，不表示学习者已掌握。

## Level 0：先能解释任务，不急着写网络

### 要回答

- BOS/BOST 测到的是图像位移、折射率梯度的线积分，还是折射率本身？
- forward operator `A_g`、伴随 `A_g*`、逆算子和正则化分别是什么？
- 少视角为什么非唯一？噪声为什么会在反演中放大？
- NeRIF 的 per-instance coordinate field 与 shared neural operator 有何区别？

### 补课模块

1. 向量微积分：梯度、散度、方向导数、线积分。
2. 线性代数：SVD、条件数、正规方程、伪逆、正则化。
3. 概率统计：噪声模型、训练/测试分布、置信区间、配对比较。
4. 流体基础：连续方程、动量方程、密度/温度与折射率关系、可压缩流的基本量纲。
5. 成像基础：针孔/光线几何、Radon 型投影、filtered backprojection、迭代重建。

### 过关产物

用一页纸画出 `field -> deflection -> displacement -> lift -> reconstruction -> reprojection`，并标明每一步的单位、可观测量和误差源。

## Level 1：PyTorch 与三维张量工程

### 必会代码

- `Dataset/DataLoader`、固定随机种子、train/val/test split。
- `Conv3d`、FFT、autograd、混合损失、checkpoint。
- tensor shape：`[batch, channel, depth, height, width]`。
- 训练曲线、梯度爆炸检查、参数量、显存和推理计时。
- 不在 test 上调超参数，不用逐样本 GT 对齐预测。

### 练习顺序

1. 让小网络 overfit 8 个三维 phantom。
2. 用 identity mapping 检查 loader 和 shape。
3. 用固定 physics lift 预测 residual。
4. 只开 field loss；再逐个加入 gradient、reprojection、boundary。
5. 每加一项都做 ablation，而不是只看总 loss 下降。

### 过关产物

同一 seed 两次运行的 sample-level metrics 完全一致；训练集能 overfit；validation 与 test 代码不共享优化状态。

## Level 2：真正理解 FNO 和 DeepONet

### FNO

要会解释：

- lifting layer 为什么把物理 channel 映到 hidden width；
- spectral convolution 为什么保留有限 Fourier modes；
- local linear path 为什么与全局 spectral path 相加；
- `n_modes`、hidden width、层数和网格尺寸分别控制什么；
- discretization invariance 是模型定义和离散实现的性质，不等于你的数据必然跨分辨率成功。

代码练习：先实现一个 1D spectral layer，再使用官方 `neuralop.models.FNO` 做 2D Darcy tutorial，最后进入 3D residual reconstruction。

### DeepONet

要会解释：branch 编码离散观测函数，trunk 编码 query coordinate，内积输出连续场。固定传感器数/顺序是标准 DeepONet 在 BOST 可变视角下的主要限制。

代码练习：固定 5 视角，branch 输入 flattened lift 或 projection features，trunk 输入 `(x,y,z)`；先只查询规则网格，再讨论任意坐标。

### 过关产物

用相同 data split、训练预算和指标比较 3D U-Net 与 FNO；能解释负结果来自优化、容量、频谱偏置还是分布偏移，而不是只说“模型不够大”。

## Level 3：逆问题与物理一致性

### 必做设计

1. residual skip vs absolute output；注意 absolute 若仍使用 lift 输入，就不是 raw-measurement direct operator。
2. field loss vs field + BOST reprojection。
3. observed-view vs held-out-view reprojection。
4. one global train calibration vs forbidden per-sample GT calibration。
5. view/noise/family condition cells，而不是只随机打散。
6. field、gradient、mass/centroid 或组内真正关心的物理量并列。

### 关键判断

- 重投影低不一定场正确：病态逆问题可能有多个近似观测解。
- field truth 低不等于真实数据可选模型：真实实验没有 GT。
- physics loss 可能只让模型拟合 forward-model bias。
- synthetic-to-real gap 包含 phantom、noise、geometry、ray model 和 displacement estimator 五层。

### 过关产物

能给出一个失败案例，其中 field 与 held-out metric 排名不同，并说明下一步用什么实验区分原因。

仓库当前证据：三种子消融、router collapse、independent support-fit、exact null 上界、learned-null 负结果与 adaptive-query v2c 均保留。v2d 淘汰 QC-SNCO；v3a 建立 ridge-FNO 强基线；v3b 又让 U-Net/FNO/DeepONet/自有 ray-set 候选得到相同 42-channel ray/mask/angle/ridge 输入。三种子下，自有候选相对 FNO 在 K=4/6/8 平均 +1.77%/+5.36%/+1.09%，只有 K=6 通过当前开发门槛；K=4/K=8 各有一个 seed 转负，thin-front OOD 未胜出。详见 `own_algorithm_lab.html` 与 `own_algorithm_review_brief.md`。这些结果仍是已查看的线性小尺寸 synthetic development evidence，不是论文最终验证。

## Level 4：进入 OERF/何远哲主线

### 路线 A：camera-budgeted direct inverse operator

`measurements + acquisition geometry -> strong numerical reconstruction -> fixed-epoch FNO baseline -> cross-architecture compute accounting -> matched PEFT controls -> independent audit`。240-epoch 审计锁定 carry-continuation-Adam/restart-cosine 为 validation 冠军，同时保留 long-cosine plateau control。geometry 为 Go 后可写功能 pilot；FLOPs/内存/time-to-target、blind final 与确认性 superiority 继续关闭。

### 路线 B：operator-initialized NeRIF

把 ridge-FNO 场插值到 NeRIF 坐标网络，并用零初始化 residual head 保留初始场；固定 K、NeRIF 架构、optimizer、stop rule 和硬件，对比 random、ridge、ridge-FNO 与 oracle initialization。主终点是达到预锁定 Q_audit 质量所需的总墙钟时间，次终点是最终质量非劣、失败率、p10 和跨工况 harm rate。只有同时“更快且不更差”，才把它写成方法贡献。

### 路线 C：QC-SNCO 机制性负结果

当前路径已触发停止条件，不再调 router/corrector。只在 direct/warm-start 主线稳定、训练 mask 匹配并有全新锁定 final fields 后，决定是否做一次复核；否则作为“null consistency 不保证 field correctness”的完整负结果章。

### 路线 D：4D/evolution operator

输入历史投影或三维场，输出未来/完整四维场；与蔡组 JFM 2019、TDBOST、framewise 和 M3B temporal SVD 对照。静态重建未跑稳前不进入。

## 仓库研究续作表（不是零基础课表）

| 周 | 代码任务 | 研究验收 |
| --- | --- | --- |
| 1 | 跑通官方 2D FNO 与 PyTorch 3D toy | 能画 shape/loss/参数流程 |
| 2 | 读并复写 T16 forward/lift | forward 与 matrix 版本数值一致 |
| 3 | 批量 phantom、manifest、split | family 与 condition 不泄漏 |
| 4 | 3D U-Net baseline | 先 overfit，再验证 |
| 5 | residual 3D FNO（已完成） | 与 U-Net 同预算比较 |
| 6 | held-out reprojection（已完成） | 无逐样本 GT 标定 |
| 7 | residual/absolute/loss/capacity 消融（已完成） | 三种子稳定性与输出方式翻转 |
| 8 | fixed/metadata/observable gate 筛查（已完成） | 保留负对照、oracle regret 与可复跑证据 |
| 9 | shared dual v1 + support-fit（已完成） | 互补、闭式强基线、router collapse 与 feature alignment |
| 10 | independent dual + exact/matched null + adaptive query pilot（已完成审计） | 正/负结果、support leakage、统计单元与红队问题 |
| 11 | v3a training-matched direct rerun（已完成） | ridge-FNO 三预算 3/3；FBP-lift FNO 0/3；结果、配置和 validator 可追溯 |
| 12 | v3b/v3c 负结果 → v3d optimizer → v3e compute | 锁定 FNO epoch 冠军、plateaued control 与五架构成本 schema；geometry 为 Go 后写机制 pilot，并补 matched error–compute curves |

## 阅读顺序

### 第一层：定义和根架构

1. Kovachki et al., JMLR 2023, *Neural Operator*：函数空间、离散不变性和架构谱系。
2. Li et al., ICLR 2021, *Fourier Neural Operator*：spectral kernel、modes 和 resolution experiment。
3. Lu et al., Nature Machine Intelligence 2021, *DeepONet*：branch/trunk 与 operator generalization。

### 第二层：逆问题和物理约束

4. Molinaro et al., ICML 2023, *Neural Inverse Operators*。
5. Dai et al., MIDL 2023, *Neural Operator Learning for Ultrasound Tomography Inversion*。
6. Adler and Oktem, IEEE TMI 2018, *Learned Primal-Dual Reconstruction*。
7. Liu et al., IEEE TMI 2025, *Geometry-Aware Attenuation Learning*。

### 第三层：无真值与跨视角可靠性

8. Hendriksen et al., IEEE TCI 2020, *Noise2Inverse*：测量拆分与无 clean target 训练。
9. Sechaud et al., ICLR 2026, *Equivariant Splitting*：incomplete observation + splitting + equivariance。
10. Mo and Magri, DCE 2026：无完整 3D truth 的稀疏/noisy 三维流场重建。
11. Dryden and Hoefler, NeurIPS 2022, *Spatial Mixture-of-Experts*：self-supervised routing 与 collapse 控制。

### 第四层：流场重建与少视角邻域

12. Zhao et al., IJTS 2024, *RecFNO*：MLP/mask/Voronoi sparse embedding。
13. Zhang et al., JCP 2025, *Energy Transformer*：Schlieren 与 3D turbulent jet 稀疏重建。
14. Lin et al., CVPR 2024, *C2RV*：不同视角不等权、cross-view attention。
15. Reed et al., ICCV 2021, dynamic CT INR：template/motion field 与 4D 自监督。

### 第五层：BOST/OERF 物理主线

16. He et al., Physics of Fluids 2025, NeRIF。
17. Bo et al., Optics Express 2023, GRU-BOST。
18. OERF 的 NTAS CNN 2018、JFM 3D flame evolution 2019、limited-projection VT 2020。
19. He et al., ACM TOG 2026, TDBOST；只有静态算子稳定后再进入。

### 第六层：只有失败驱动才读的升级架构

20. CNO / U-NO：local/multiscale matched operator baseline。
21. Multiwavelet / Riesz NO：thin-front 高频和局部非平稳失败后再测。
22. GINO / Geo-FNO / OT geometry：真实相机几何跨样本变化时再引入。
23. LUNO / Approximate Bayesian NO / PNO：branch disagreement 已证明能识别失败后再升级 UQ。

### 第七层：当前最优先的强底座与 INR 初始化链

24. Schwab et al., *Deep Null Space Learning*：hard-null projector 与 data consistency。
25. Quan et al., *Siamese Cooperative Learning*：range/null 分解与不完整测量自监督。
26. Bhat et al., *Neural Correction Operator*：短物理重建 + learned correction 的直接邻域。
27. Prasthofer et al., *VIDON*；Hao et al., *GNOT*：相机数/位置变化与异构几何输入。
28. Serrano et al., *CORAL*；Pal et al., *O-INR*：operator prior 与 coordinate field/NeRIF 的连接；不要把 O-INR 越界写成“直接输出 NeRIF 权重”。
29. Tancik et al., CVPR 2021, *Learned Initializations for Optimizing Coordinate-Based Neural Representations*：初始化可加速逐实例坐标网络优化的直接先例。
30. Google Research, NeurIPS 2024, *STRAINER*：共享 prior/初始化如何帮助多个 INR 实例；用于设计对照，不等同 BOST 物理贡献。
31. Liu-Schiaffini et al., *Localized Kernels*；Tran et al., *DuFal*：thin-front 高频失败后的 local/global 对照。
32. Ma et al., calibrated operator UQ；Moya et al., Conformalized-DeepONet；SelectiveNet：独立校准、risk-coverage 与拒答系统评估。

完整角色化列表见 `operator_3d_reconstruction_literature_atlas.md`，交互筛选见 `operator_3d_innovation_lab.html`。

## 每篇论文的笔记模板

不要只写摘要。每篇固定回答：

1. 输入/输出函数与离散 shape 是什么？
2. forward physics 是否显式出现？
3. paired data 从哪里来，样本数是多少？
4. train/test 是 seed、parameter、family 还是 condition OOD？
5. baseline 是否公平，参数和预算是否接近？
6. 报了哪些 field/physics/runtime 指标？
7. 最严重 failure case 是什么？
8. 哪一部分能迁到 BOST，哪一部分物理不相同？
9. 复现最小依赖是什么？
10. 它会改变你的哪一个实验，而不是只增加引言引用？
