# 本科毕业设计开题报告正文初稿

题目建议：

面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

备选题目：

- 面向 BOST/NeRIF 的可复现重构、误差分析与真实数据迁移工具
- 基于背景纹影层析折射率场的 PIV 速度测量误差补偿方法初探
- 四维背景纹影层析中的低秩时序先验及其合成数据验证

## 1. 研究背景与意义

极端反应流、燃烧流场和高速流动中通常伴随强烈的温度、密度和折射率非均匀分布。传统接触式测量方法容易扰动流场，且难以获得三维瞬态结构。背景纹影技术（Background-Oriented Schlieren, BOS）利用背景图像在折射率梯度场中的表观位移来反推出流场信息，具有非接触、系统相对简单、适合大视场测量等优点。进一步地，背景纹影层析（Background-Oriented Schlieren Tomography, BOST）通过多视角背景位移反演三维折射率场，并可结合 Gladstone-Dale 关系进一步关联密度和温度场，因此是反应流非接触诊断的重要方向。

上海交通大学 OERF Lab 的公开研究方向包括极端反应流光学诊断、计算成像、机器学习辅助流动显示以及数据融合实验流体力学。蔡伟伟老师团队近年来在 BOST、NeRIF、PIV-BOST 和 4D BOST 等方向上形成了连续研究脉络：从背景图像位移测量，到三维折射率场重构，再到速度测量折射补偿和四维时序流场重构。对本科毕业设计而言，一个合理切入点不是泛泛使用深度学习，而是围绕 BOST 这一具体实验诊断问题，建立可复现的重构、误差分析和数据接口流程。

传统 BOST 常将三维折射率场离散成体素网格，并构造线性或近似线性的层析反演问题。该类方法在少视角、含噪和高分辨率条件下面临病态反演、离散化误差、内存成本高和伪影明显等问题。NeRIF 等神经隐式场方法用坐标网络连续表示折射率场，结合可微光线积分和重投影损失，有潜力改善空间连续性和少视角重构质量。但神经隐式表示并不天然优于传统方法，其实际优势与视角数、噪声水平、坐标编码、正则化强度、采样策略和真实几何误差密切相关。因此，本课题拟系统分析神经隐式折射率场方法在 BOST 重构中的适用边界，并根据课题组数据条件向 PIV-BOST 或 4D BOST 子问题升级。

## 2. 国内外研究现状

### 2.1 BOS/BOST 与反应流三维诊断

Raffel 对 BOS 技术的基本原理、图像位移估计、实验限制和误差来源进行了系统综述，为理解背景图像位移与折射率梯度之间的关系提供了基础。Grauer 等人提出瞬态三维火焰 BOST 成像方法，将多视角背景位移用于火焰折射率场重构。蔡伟伟团队在单相机内窥 BOST、火焰折射率/密度/温度体成像等方面开展了系列研究，为 OERF 的九视角/内窥式 BOST 系统和后续 NeRIF 工作奠定了基础。

### 2.2 传统体素重构与统一 BOST 方法

传统 BOST 通常基于体素网格离散化，将未知折射率场或其梯度场转化为有限维反演问题。UBOST 等方法提高了体折射率测量的速度和鲁棒性，但仍受到网格分辨率、正则化参数和少视角病态性的限制。对于本科阶段的实现而言，传统体素法、简化 FBP、Tikhonov 正则化或逐层 stack reconstruction 可作为 baseline，用于与 coordinate-field 或 neural-field 方法比较。

### 2.3 NeRIF 与神经隐式折射率场

He 等人的 NeRIF 工作将三维折射率场表示为连续神经隐式场，输入空间坐标并输出折射率及其梯度，再通过可微光线积分预测多视角背景位移。该方法将传统体素离散问题转化为连续函数拟合问题，并使用重投影误差、梯度一致性、坐标编码和随机采样提升重构质量。NeRIF 是本课题最直接的方法来源。但本科阶段不宜一开始完整复现其所有实验系统和网络细节，应先从合成数据、少视角扫描和简化坐标场表示入手，建立可控闭环。

### 2.4 PIV-BOST 折射补偿

在火焰和热流场中，PIV 粒子图像在传播至相机过程中会受到折射率非均匀性的影响，导致粒子图像位置偏移，从而引入速度测量误差。Zheng、He 等人的 PIV-BOST 工作使用同步 BOST 重构折射率场，并用于补偿 PIV 速度测量中的折射误差。后续 stereo-velocity compensation 工作进一步说明该方向已从二维速度补偿扩展到更复杂的立体速度测量。本课题若获得同步 PIV-BOST 数据，可将其作为第二阶段升级；若没有真实数据，则可先完成 2D 向量场或粒子图像 toy，分析误差传播链路。

### 2.5 4D BOST 与时序先验

对于高速流场，逐帧独立三维重构会带来较高计算成本和明显帧间抖动。He 等人的 4D BOST 工作通过张量分解和轻量网络对 `X-Y-Z-T` 时空场进行建模，以提高高速、高保真流场重构效率。完整复现该工作超出本科毕设风险边界，但其核心思想可以拆成低秩时序先验子问题：构造随时间变化的三维折射率 phantom，比较逐帧重构与低秩时序表示在误差、平滑度和运动轨迹上的差异。

## 3. 研究目标

本课题的总体目标是建立一个面向 BOST/NeRIF 的可复现实验与误差分析流程，系统评估神经隐式/坐标先验在少视角、含噪背景纹影层析中的适用边界，并为后续接入 OERF 真实 BOST、PIV-BOST 或 4D BOST 数据提供基础工具。

具体目标包括：

1. 建立合成 BOST 数据生成流程，包括二维/三维折射率 phantom、多视角 BOS-like deflection、噪声扰动和评价指标。
2. 实现传统 baseline，包括二维简化层析、三维逐层 stack reconstruction 和基础正则化/坐标正则化方法。
3. 实现简化的 coordinate-field / neural-field 思想验证，分析视角数、噪声和表示容量对重构误差的影响。
4. 形成可用于真实数据迁移的 manifest、数据接口、可视化和校验流程。
5. 根据何远哲师兄和课题组数据条件，选择 PIV-BOST 速度补偿或 4D BOST 低秩时序先验作为升级子问题。

## 4. 研究内容与技术路线

### 4.1 BOST 物理模型与合成数据

首先构造二维和三维折射率场 phantom，包括 Gaussian blob、火焰薄层和随时间变化的三维结构。基于小角度近似，使用折射率梯度沿视线方向积分得到 BOS-like deflection。该阶段的重点不是完全复刻真实光路，而是建立从物理场到观测位移的可控 forward model。

### 4.2 Baseline 与 coordinate-field 重构

在二维场景中，先实现 M0 toy：合成折射率场、多视角偏折观测、传统 baseline 和 coordinate-field inverse。随后扩展到三维栈重构 M1：对每个 z 切片生成多视角偏折，并逐层重构，再用三维 coordinate regularizer 抑制少视角伪影。M2 进一步进行视角数、噪声和表示容量扫描，量化坐标先验在何种条件下优于传统 baseline。

### 4.3 鲁棒性分析

鲁棒性分析是本课题的核心研究内容。拟扫描以下变量：

- 视角数：3、5、7、9 或更多。
- deflection noise：无噪声、低噪声、中噪声。
- 表示容量：Fourier feature 数量、网络宽度或低秩 rank。
- 评价指标：relative L2、CC、SSIM proxy、PSNR、重投影误差、temporal smoothness。

已有 M2 结果显示：在当前 synthetic proxy 中，3/5 视角下 coordinate regularizer 更稳，7/9 视角下传统 stack baseline 更强；表示容量过低会过度平滑。这说明本课题的重点不是证明神经方法永远更好，而是给出其适用边界。

### 4.4 PIV-BOST 速度补偿升级

若课题组更需要 PIV-BOST 方向，本课题将基于 M3A toy 分析折射率场对 PIV 速度测量误差的影响。当前 M3A 已在速度向量场层面模拟了“真实速度 + 折射位移变化 = 观测 PIV 位移”的过程，并用 BOST-style 折射位移估计进行补偿。后续可升级到粒子图像互相关、OpenPIV/PIVlab 参数扫描或真实 PIV-BOST 数据接口。

### 4.5 4D BOST 低秩时序升级

若课题组更需要 4D BOST 子模块，本课题将基于 M3B toy 分析低秩时序先验对逐帧重构抖动的抑制作用。当前 M3B 已构造时序三维折射率体序列，并比较逐帧 sparse-view stack reconstruction 与 SVD 低秩时序表示。结果显示 low-rank rank 3 能降低平均误差和 temporal smoothness 指标。后续可进一步扫描 rank、帧数、噪声和视角数，并与 4D BOST 论文中的张量分解思想进行概念对照。

## 5. 已有预研基础

目前已完成以下本地可运行材料：

| 编号 | 内容 | 当前结论 |
| --- | --- | --- |
| M0 | 2D BOST / coordinate-field toy | 9 视角下 coordinate-field inverse relative L2 约 0.108，baseline 约 0.113 |
| M1 | 3D-stack sparse-view BOST toy | 5 视角下 coordinate regularizer relative L2 约 0.234，baseline 约 0.257 |
| M2 | noise-view-capacity robustness scan | 3/5 视角下 coordinate regularizer 更稳，7/9 视角下 baseline 更强 |
| M3A | PIV-BOST velocity compensation toy | 速度 RMSE 从约 0.0101 降到 0.0067，95 分位误差从约 0.0240 降到 0.0072 |
| M3B | 4D BOST low-rank temporal toy | mean relative L2 从约 0.366 降到 0.347，temporal smoothness 从约 0.279 降到 0.177 |

此外，已整理 OERF 方向图谱、何远哲论文深挖、核心参考文献 BibTeX、论文到 demo 映射表、真实数据 manifest 模板、会前决策包和 12 周推进验收板。这些材料共同构成后续开题和毕业设计执行基础。

## 6. 拟解决的关键问题

1. 少视角条件下，coordinate-field / neural-field prior 是否能比传统 stack baseline 更稳定？
2. 噪声和表示容量如何影响 BOST 重构误差？
3. 当真实数据可得时，合成数据 pipeline 如何迁移到 OERF 九视角 BOST 数据？
4. 若选择 PIV-BOST，BOST 重构误差如何传播到 PIV 速度误差补偿？
5. 若选择 4D BOST，低秩时序先验能否减少逐帧重构抖动，rank 应如何选择？

## 7. 预期成果

预期完成以下成果：

- 一套可复现的 BOST/NeRIF 合成数据与重构实验脚本。
- 传统 baseline 与 coordinate-field / neural-field 方法的参数扫描结果。
- 视角数、噪声、表示容量、低秩 rank 等变量的误差图谱。
- 面向 OERF 真实数据的 manifest、数据读取和可视化接口。
- 一份开题报告、答辩 PPT、毕业论文图表和核心文献引用库。
- 若数据条件允许，完成 PIV-BOST 补偿或 4D BOST 低秩时序子问题的初步验证。

## 8. 进度安排

| 阶段 | 时间 | 主要任务 | 交付物 |
| --- | --- | --- | --- |
| 第 1 阶段 | 第 1-4 周 | 补 BOST 物理模型、反问题基础、运行 M0/M1 | 变量链条图、2D/3D toy 结果 |
| 第 2 阶段 | 第 5-8 周 | 完成 M2 鲁棒性扫描，整理指标和图表 | noise-view heatmap、capacity scan、实验表 |
| 第 3 阶段 | 第 9-12 周 | 与何远哲确认 A/B/C 定题路径，接入数据或升级 toy | 会前决策包、数据 manifest、开题 PPT |
| 第 4 阶段 | 大四上 | 接真实数据或完成 PIV/4D 子问题，完善方法章节 | 真实数据接口、补偿/时序结果 |
| 第 5 阶段 | 大四下 | 完成论文写作、答辩图和代码整理 | 毕业论文、答辩 PPT、代码仓库 |

## 9. 风险分析与应对

| 风险 | 影响 | 应对策略 |
| --- | --- | --- |
| 暂时拿不到真实 BOST 数据 | 难以做真实实验验证 | 使用 synthetic + open-source BOS/TBOS 数据保底，保留真实数据接口 |
| 完整 NeRIF 复现过重 | 代码和算力风险高 | 先做 coordinate-field / simplified NeRIF，再逐步加入 PyTorch neural field |
| PIV-BOST 数据复杂 | 速度补偿难以真实验证 | 先做向量场和粒子图像 toy，只在数据允许时接真实 PIV |
| 4D BOST 超出本科范围 | 任务不可控 | 只做低秩时序先验、rank scan 和时序指标，不承诺完整 TOG 复现 |
| 结果显示神经方法不总是更好 | 可能影响创新叙事 | 将创新点写成“适用边界和误差图谱”，而不是“神经方法绝对优越” |

## 10. 主要参考文献

本报告主线参考文献以 `references.bib`、`paper_to_demo_map.md` 和 `opening_reference_shortlist.md` 为准。核心文献包括：

1. Raffel, Background-oriented schlieren techniques, Experiments in Fluids, 2015.
2. Grauer et al., Instantaneous 3D flame imaging by BOST, Combustion and Flame, 2018.
3. Liu, Shui and Cai, time-resolved endoscopic BOST, Aerospace Science and Technology, 2020.
4. Grauer and Steinberg, UBOST, Experiments in Fluids, 2020.
5. He et al., Neural refractive index field, Physics of Fluids, 2025.
6. Zheng et al., simultaneous PIV-BOST compensation, Experiments in Fluids, 2025.
7. Zheng et al., stereo-velocity refractive index compensation, Proceedings of the Combustion Institute, 2026.
8. He et al., tensor decomposition-based 4D BOST, ACM Transactions on Graphics, 2026.
9. Huang et al., Computational flow visualization, Cell Reports Physical Science, 2024.
10. Molnar et al., open-source BOS tomography dataset, Experiments in Fluids, 2026.

## 11. 当前开题结论

当前最建议采用的主线仍是：

面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析。

理由是该题目同时满足四个条件：贴合 OERF 和何远哲主线、无真实数据也能用合成/开源数据启动、已有 M0-M2 预研基础、后续可自然升级到 PIV-BOST 或 4D BOST。真正开题前，应通过 `he_meeting_decision_pack.md` 与何远哲确认 A/B/C 路线，并以师兄能提供的数据和组内需求作为最终定题依据。
