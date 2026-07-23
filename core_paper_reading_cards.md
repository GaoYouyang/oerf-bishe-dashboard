# 核心论文精读卡片集

用途：这份文件不是论文摘要库，而是“读完每篇论文后要产出什么”的行动卡。每张卡都把论文拆成：为什么读、必须抽取的公式/图/参数、对应本地 demo、能写进毕业论文哪一章、要问何远哲什么。

最后核验：2026-07-07。公开入口优先使用出版社、DOI、arXiv 和 OERF 工作台已审计来源。

---

## 总读法

读 P0/P1 论文时，不要从头到尾抄摘要。按下面顺序读：

1. 先看 measurement：观测量是什么，是背景位移、PIV 粒子位移、光谱吸收、发射强度还是三维体数据？
2. 再看 unknown：反演对象是什么，是折射率、密度、温度、速度、碳烟还是颗粒位置？
3. 再看 forward model：作者如何把物理场变成图像/投影/位移？
4. 再看 inverse model：作者如何从观测反推物理场，baseline 是什么？
5. 最后看 evaluation：作者用什么指标证明结果可信，有没有重投影或真实物理验证？

每读完一篇，必须产出四件东西中的至少一件：一个公式块、一个参数表、一个 demo 改动、一个给何远哲的问题。

---

## Card 1：NeRIF

### 正式信息

- 论文：Neural refractive index field: Unlocking the potential of background-oriented Schlieren tomography in volumetric flow visualization
- 作者：Yuanzhe He, Yutao Zheng, Shijie Xu, Chang Liu, Di Peng, Yingzheng Liu, Weiwei Cai
- 期刊：Physics of Fluids, 2025
- DOI：`10.1063/5.0250899`
- 可读入口：`https://arxiv.org/html/2409.14722v2`

### 为什么必须读

这是你毕设主线的核心论文。它把 BOST 从体素离散重构推向连续神经隐式折射率场，正好对应“少视角、含噪、有限内存条件下如何重构三维折射率场”这个本科可做的问题。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 物理链条 | `n(x,y,z)`、`grad n`、ray deflection、image displacement | 一页 BOST forward model 图 |
| 网络表示 | 坐标输入、频率编码、MLP 输出、梯度 head | `CoordinateField` 伪代码 |
| 损失函数 | displacement consistency、gradient consistency、regularization | loss 公式草图 |
| 实验设置 | 视角数、投影分辨率、噪声、采样点数、真实火焰系统 | 参数表 |
| 评价指标 | L2、SSIM、CC、PSNR、reprojection consistency | `metrics.py` 必需项 |

### 对应本地 demo

- `demo_m0/`：二维最小闭环。
- `demo_m1/`：三维 stack proxy。
- `demo_m2/`：view-noise-capacity robustness scan。
- `nerif_bost_implementation_blueprint.md`：正式代码项目蓝图。

### 一周内能做的动作

- 把 M1/M2 的 coordinate regularizer 改写成 PyTorch 坐标 MLP。
- 先让 MLP 拟合 ground-truth phantom，再接 displacement loss。
- 输出 `baseline_vs_nerif.png`、`reprojection_error.png` 和 `loss_curve.png`。

### 写进论文

- 第 2 章：BOST 物理模型与反问题。
- 第 3 章：神经隐式折射率场方法。
- 第 4 章：少视角/噪声/容量鲁棒性实验。

### 问何远哲

1. 真实 NeRIF 数据里，网络是输出 `n`、`grad n`，还是多 head？
2. 真实九视角数据能否提供 displacement field，而不是原始背景图？
3. 本科阶段 baseline 应该对齐 UBOST、体素迭代法，还是组内已有代码？

---

## Card 2：PIV-BOST

### 正式信息

- 论文：Instantaneous refractive index compensation on the velocity measurement using simultaneous PIV-BOST
- 期刊：Experiments in Fluids, 2025
- DOI：`10.1007/s00348-025-04093-y`
- 公开入口：`https://link.springer.com/article/10.1007/s00348-025-04093-y`
- 后续扩展：Instantaneous refractive index compensation on stereo-velocity measurement in turbulent combustion, Proceedings of the Combustion Institute 2026, DOI `10.1016/j.proci.2026.106175`
- 支持前史：Particle image velocimetry in refractive index fields of combustion flows, Experiments in Fluids 2019, DOI `10.1007/s00348-019-2795-1`

### 为什么必须读

它是何远哲方向里最贴真实实验测量的一条升级线：BOST 重构折射率场，PIV 测速度，折射率梯度会污染粒子图像位置和速度估计，因此需要瞬时补偿。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 同步系统 | BOST 视角、PIV 平面、相机/激光同步 | 数据 manifest 字段 |
| 误差来源 | 折射率梯度如何造成粒子像素偏移 | 速度误差传播公式草图 |
| 前人量化 | particle position error、systematic/random velocity error、ray-tracing verification | PIV-BOST 不是凭空需求的证据 |
| 补偿层级 | 原始图像校正、互相关窗口校正、速度场校正 | 三种策略对比表 |
| 数据格式 | PIV image pairs、velocity field、BOST reconstruction、timestamps | `piv_bost_manifest.json` 扩展 |
| 指标 | velocity RMSE、局部误差、补偿前后差异 | M3A 指标表 |
| 后续边界 | stereo-velocity PIV-BOST 为什么比 planar PIV 难 | 展望段与“不承诺完整复现”的风险说明 |

### 对应本地 demo

- `demo_m3a/`：速度向量场层面的 PIV-BOST compensation toy。
- `data_templates/piv_bost_manifest.json`：同步数据模板。

### 一周内能做的动作

- 用 M3A 扫描 deflection noise 和 PIV displacement magnitude。
- 增加一个“补偿策略”对比：不补偿、速度场层补偿、理想位移补偿。
- 加一个 particle-position-error toy：先不做完整互相关，只模拟折射率导致的粒子像素位置偏差，再传播到速度误差。
- 若 OpenPIV 环境可用，生成一组合成粒子图像 pair 做初步互相关。

### 写进论文

- 第 5 章 B 路线：BOST 折射率场对 PIV 速度测量误差的传播与补偿。
- 作为主线 NeRIF 的实验升级，而不是一开始单独开大题。
- 2026 stereo-velocity 论文只作为冲刺/展望引用，用来说明 PIV-BOST 方向的真实需求和复杂度。

### 问何远哲

1. 真实数据是 planar PIV 还是 stereo-PIV？
2. 补偿更希望做在粒子图像层、互相关层，还是速度场层？
3. 组里是否关心 particle-position error，还是只看最终 velocity-field error？
4. 能否给一小份同步时间戳、PIV 参数和补偿前后速度场？

---

## Card 3：4D BOST

### 正式信息

- 论文：Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction
- 期刊：ACM Transactions on Graphics, 2026
- DOI：`10.1145/3809488`
- 公开入口：`https://dl.acm.org/doi/10.1145/3809488`

### 为什么读

它是何远哲当前最前沿的一条挑战线。它把三维 BOST 推到 `X-Y-Z-T` 时空重构，用张量分解和轻量网络处理高速动态流场。对本科阶段最有价值的不是完整复现，而是拆出低秩时序先验、rank 扫描、时序一致性指标和可视化子问题。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 数据组织 | `X-Y-Z-T` 四维体如何组织 | 4D array shape 约定 |
| 分解思想 | 空间-时间低秩或张量分解如何降低成本 | M3B rank scan 解释 |
| 时序指标 | temporal smoothness、轨迹误差、framewise error | `temporal_metrics.py` 草图 |
| 工程瓶颈 | 内存、帧数、视角同步、ray tracing | 风险登记表 |
| 子问题边界 | 哪些适合本科，哪些不能承诺 | 开题答辩 Q&A |

### 对应本地 demo

- `demo_m3b/`：4D low-rank temporal toy。
- `minimum_demo_protocol.md`：M3B 说明。

### 一周内能做的动作

- 修改 M3B 的 rank、noise、frame count，生成 rank trade-off 曲线。
- 增加 leave-one-time-window validation。
- 做一张“逐帧 baseline vs low-rank temporal prior”的时序抖动对比图。

### 写进论文

- 第 5 章 C 路线：四维 BOST 的低秩时序先验简化研究。
- 只能写“子问题”和“toy”，不要写“完整复现 ACM TOG”。

### 问何远哲

1. 4D BOST 最适合本科生做的是 rank 扫描、可视化、指标，还是数据清洗？
2. 能否给 synthetic 参数或一小段不涉密时序数据？
3. 真实 4D 数据最主要瓶颈是内存、速度、噪声、几何，还是时序同步？

---

## Card 4：Raffel 2015 BOS Techniques

### 正式信息

- 论文：Background-oriented schlieren (BOS) techniques
- 期刊：Experiments in Fluids, 2015
- DOI：`10.1007/s00348-015-1927-5`
- 公开入口：`https://link.springer.com/article/10.1007/s00348-015-1927-5`

### 为什么读

这是 BOS 原理综述。你需要用它补“背景纹影到底是什么、和传统 schlieren/shadowgraph/interferometry 有什么关系、为什么要用背景图案相关估计位移”。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 技术定义 | BOS 属于折射率/密度可视化方法 | 绪论定义 |
| 观测量 | 背景点阵或纹理的 apparent displacement | 变量表 |
| 图像处理 | cross-correlation / optical flow | 位移估计流程图 |
| 误差来源 | 背景图案、相机、距离、噪声、折射模型 | 风险清单 |
| 和 BOST 关系 | 多视角 BOS -> tomography | 第 2 章连接段 |

### 对应本地 demo

- `foundation_bridge.md`。
- `demo_m0/` forward model 说明。

### 一周内能做的动作

- 写一页“BOS vs Schlieren vs Shadowgraph vs BOST”表。
- 在 M0 文档中补清楚 `displacement` 是观测量，不是直接物理场。

### 问何远哲

- OERF 实验里位移估计主要用 cross-correlation、optical flow，还是 DeepFlow/其他方法？

---

## Card 5：Grauer 2018 BOST Flame Imaging

### 正式信息

- 论文：Instantaneous 3D flame imaging by background-oriented schlieren tomography
- 期刊：Combustion and Flame, 2018
- DOI：`10.1016/j.combustflame.2018.06.022`

### 为什么读

这是 BOST 用于三维火焰成像的重要前史，能帮你理解 NeRIF 为什么要改进传统体素/层析路线，也能补“火焰折射率、密度、温度”之间的实验背景。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 实验对象 | flame / reacting flow | 研究背景段 |
| 重构对象 | 3D refractive index / density / temperature 相关量 | 物理量表 |
| baseline | 传统 BOST 层析方法 | baseline 章节 |
| 图表 | 3D 重构切片、火焰结构、误差解释 | PPT 图示灵感 |
| 局限 | 体素、视角、噪声、真实实验复杂度 | NeRIF 动机 |

### 对应本地 demo

- `demo_m1/`。
- `figures/bost_physical_chain.png`。

### 一周内能做的动作

- 把 M1 的 phantom 改成 flame-sheet / plume-like 形状，而不是纯 Gaussian blob。
- 输出 RI / density / temperature 的概念链条图，不一定做定量温度。

### 问何远哲

- 现在组内 NeRIF 的真实对象和早期 BOST flame imaging 对象差别在哪里？

---

## Card 6：UBOST

### 正式信息

- 论文：Fast and robust volumetric refractive index measurement by unified background-oriented schlieren tomography
- 期刊：Experiments in Fluids, 2020
- DOI：`10.1007/s00348-020-2912-1`

### 为什么读

NeRIF 必须和传统强 baseline 对话。UBOST 是传统 BOST 路线中很重要的“fast and robust”参照，可以帮助你设计公平 baseline 和鲁棒性指标。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| baseline 思路 | unified / robust volumetric reconstruction | baseline 设计表 |
| 速度和鲁棒性 | 为什么它被称为 fast/robust | 指标对照 |
| 正则化 | 如何稳定少视角或噪声 | `voxel_baseline.py` TODO |
| 和 NeRIF 对比 | 体素/显式 vs 连续/隐式 | 方法对比表 |

### 对应本地 demo

- `demo_m1/`。
- `demo_m2/`。
- `nerif_bost_implementation_blueprint.md`。

### 一周内能做的动作

- 在 M2 结果解释里把 coordinate regularizer 与传统 baseline 的适用区间写清楚。
- 为后续代码项目留出 `ubost_proxy.py` 或 `sirt.py` baseline 接口。

### 问何远哲

- 本科论文里的 baseline 是否需要正式对齐 UBOST，还是用简化 SIRT/Landweber 即可？

---

## Card 7：Computational Flow Visualization

### 正式信息

- 论文：Computational flow visualization to reveal hidden properties of complex flow with optical and computational methods
- 期刊：Cell Reports Physical Science, 2024
- DOI：`10.1016/j.xcrp.2024.102282`

### 为什么读

这篇不是你要复现的方法论文，而是开题定位论文。它能帮你把题目从“我做一个 AI 重构”提升为“用 optical + computational methods 揭示复杂流动中的 hidden properties”。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 大框架 | optical method + computational method | 开题第 1 页站位 |
| hidden properties | 折射率、密度、温度、速度误差、时序结构 | 研究意义句子 |
| OERF 语言 | computational flow visualization | 题目包装 |
| 与本题连接 | BOST/NeRIF 是其中一条具体路线 | 绪论过渡段 |

### 对应本地材料

- `one_page_proposal.md`。
- `opening_report_draft.md`。
- `opening_ppt_outline.md`。

### 一周内能做的动作

- 把开题报告第一段改成“复杂流动物理量不可直接观测，需要光学测量和计算反演共同揭示”。
- 定义你的 hidden property：三维折射率场，以及由此引出的密度/速度误差。

### 问何远哲

- 我的题目中 hidden property 最好写成折射率场、密度场、速度误差，还是统一写成三维/四维流动物理场？

---

## Card 8：Open-Source BOS Tomography Dataset

### 正式信息

- 论文：Open-source BOS tomography dataset of high-speed flow over a flight body
- 期刊：Experiments in Fluids, 2026
- DOI：`10.1007/s00348-026-04189-z`

### 为什么读

如果 OERF 真实数据暂时不能给，这个数据集是最适合用来预演 pipeline 的外部保底。它可以让你练 calibration、mask、deflection estimates、3D reconstructions 和有限视角重构，不至于卡在“没有数据”。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 数据结构 | raw images、calibration、mask、deflection、reconstruction | open BOS manifest |
| 视角设置 | 多视角/有限视角如何组织 | data loader |
| benchmark | 论文给了哪些基线结果 | baseline 对照 |
| 与 OERF 区别 | 高速飞行体 vs 火焰/反应流 | 迁移限制说明 |

### 对应本地材料

- `data_request_checklist.md`。
- `data_templates/`。
- `nerif_bost_implementation_blueprint.md`。

### 一周内能做的动作

- 写一个 `open_bos_manifest.json` 草案。
- 用它检查你的 data loader 是否不依赖 OERF 私有数据格式。
- 在开题报告中把它列为“无组内数据时的保底公开 benchmark”。

### 问何远哲

- 如果暂时不能给组内数据，是否认可我先用 open-source BOS dataset 预演 pipeline，再迁移回 OERF 数据？

---

## Card 9：PIV/BOS Synthetic Image Generation

### 正式信息

- 论文：PIV/BOS synthetic image generation in variable density environments for error analysis and experiment design
- 期刊：Measurement Science and Technology, 2019
- DOI：`10.1088/1361-6501/ab1ca8`
- 代码线索：`lalitkrajendran/photon`，GPL-3.0

### 为什么读

这是“没有组内数据也能推进”的关键方法论文。它不是简单生成 phantom，而是用 ray tracing 生成带密度/折射率梯度、光学畸变和相机成像效果的 PIV/BOS 图像，正好能把 M0/M3A 从 toy 推向更接近实验的 synthetic benchmark。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 数据流 | density field -> ray tracing -> BOS/PIV image -> displacement/velocity | 一张 pipeline 图 |
| 可控误差 | optical aberration、perspective、noise、dot/particle blur | 参数表 |
| 代码接口 | 输入场、相机参数、输出图像和位移 | `data_templates/` 扩展 |
| 和 PIV-BOST 关系 | 折射率梯度如何影响粒子图像和速度估计 | M3A 升级说明 |

### 对应本地材料

- `demo_m0/`：把 simple forward model 升级成可替换接口。
- `demo_m3a/`：从向量场 toy 走向 synthetic particle image toy。
- `open_data_code_benchmark_map.md`：公开代码/数据路线。

### 一周内能做的动作

- 不完整移植 photon，只先画出输入输出接口。
- 在 `data_templates/` 增加 synthetic generator 字段：density source、camera/background、noise、ray model。
- 把 M3A 的“速度场补偿”拆成两层：图像层误差、速度矢量层误差。

### 问何远哲

- 师兄更希望我做一个可替换的 synthetic BOS/PIV generator，还是优先把现有 M0-M3A 指标做扎实？

---

## Card 10：BOS Displacement / Uncertainty Chain

### 正式信息

核心阅读组合：

- Dot tracking methodology for background-oriented schlieren (BOS), DOI `10.1007/s00348-019-2793-3`
- Uncertainty quantification in density estimation from BOS measurements, DOI `10.1088/1361-6501/ab60c8`
- Uncertainty amplification due to density/refractive index gradients, DOI `10.1007/s00348-020-02978-8`
- Wavelet-Based Optical Flow Analysis for BOS image processing, DOI `10.2514/1.J060218`

### 为什么读

NeRIF/4D BOST 的输入质量很大程度由 BOS 位移场决定。真实实验里，网络结构可能不是最大瓶颈，反而是背景图案、位移估计、强折射梯度导致的 blur、密度积分和 uncertainty propagation。这一组文献可以把 T15 “误差图谱”从泛泛鲁棒性升级成可信重构问题。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 位移估计 | cross-correlation、dot tracking、optical flow、wavelet OFA | baseline 表 |
| 不确定度 | dot blur、amplification ratio、CRLB、density uncertainty | uncertainty map 设计 |
| loss weighting | 哪些区域观测更不可信 | NeRIF confidence weighting 草图 |
| 实验设计 | 背景图案、f-number、视角方向、噪声 | 参数扫描列表 |

### 对应本地材料

- `demo_m2/`：从 noise scan 升级为 displacement-quality scan。
- future `demo_m3c/`：time offset / missing view / uncertainty map。
- `topic_decision_matrix.md`：T15 误差图谱选题。

### 一周内能做的动作

- 给 M2 增加一个 synthetic confidence map，不改变主模型。
- 输出 RMSE + uncertainty-weighted RMSE 两套指标。
- 写一页“为什么真实 BOST 的 uncertainty 比网络结构更先验”的答辩说明。

### 问何远哲

- 组里真实 BOST 数据是否已有位移场置信度、bad view 标记、mask 或 uncertainty map？如果没有，本科生做这个工具是否有用？

---

## Card 11：4D / Time-Resolved Neural Flame Reconstruction Neighbors

### 正式信息

核心阅读组合：

- FlameRF: A fast time-resolved reconstruction technique for turbulent flame using Tensorial Radiance Fields, DOI `10.1016/j.egyai.2026.100758`
- Neural dynamic fluid reconstruction technique for four-dimensional imaging of combustion flame based on deep learning, DOI `10.1016/j.engappai.2025.112288`
- Neural network-based 3D reconstruction of temperature and velocity for turbulent flames from 2D measurements, DOI `10.1016/j.combustflame.2025.114454`
- High-resolution reconstruction of turbulent flames from sparse data with physics-informed neural networks, DOI `10.1016/j.combustflame.2023.113275`
- OERF internal precursor: Limited-projection volumetric tomography for time-resolved turbulent combustion diagnostics via deep learning, DOI `10.1016/j.ast.2020.106123`
- OERF internal precursor: Volumetric reconstruction for combustion diagnostics via transfer learning and semi-supervised learning with limited labels, DOI `10.1016/j.ast.2020.106487`

### 为什么读

这一组不是 BOST 主线，也不全是何远哲论文；它们的价值在于把 4D BOST 放进更宽的“时空神经流场重构”方法族。尤其 FlameRF 的 Tensorial Radiance Fields、NDFRT 的四维动态重构、二维测量到三维温度/速度的生成式路线、稀疏数据 PINN 的物理约束，以及蔡组 2020/2021 limited-projection / limited-labels 过渡论文，都能帮助 M3B/M3C 不只停留在“做了一个 SVD toy”，而是明确为什么要扫 rank、时间插值、投影数量、少标注迁移和 sparse-measurement 误差。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 时空表示 | TensoRF / low-rank / coordinate representation | M3B rank-memory-time 表 |
| 观测模型 | 2D projection supervision、sparse measurements、PINN constraints | related work 对照表 |
| 数据约束 | limited projection、limited labels、transfer/semi-supervised learning | 真实数据迁移风险表 |
| 时间指标 | temporal interpolation、time consistency、per-frame error | M3B 新指标 |
| 边界声明 | 为什么它们不能直接当 BOST baseline | 开题答辩风险说明 |

### 对应本地材料

- `demo_m3b/`：低秩时序 toy 的 rank、内存、时间和 temporal consistency。
- future `demo_m3c/`：缺失视角/坏视角/稀疏测量和 projection completion。
- `paper_to_demo_map.md`：P2 方法邻居定位。

### 一周内能做的动作

- 不复现模型，只把 M3B 指标补成：RMSE、temporal jitter、interpolation RMSE、rank-memory-time。
- 写一页 related work 表格：4D BOST vs FlameRF vs NDFRT vs sparse PINN flames。
- 给何远哲发一个问题：4D BOST 开题 related work 是否要写 TensoRF/NDFRT 类方法，还是只保留 BOST/tomography 文献？

### 问何远哲

- 如果我做 4D BOST 子问题，师兄更希望我对齐 TOG 论文里的 tensor decomposition 指标，还是用外部 TensoRF/NDFRT 文献扩展 related work？

---

## Card 12：BOST Refinement / Displacement / Experiment Design / Physical Metrics

### 正式信息

核心阅读组合：

- The background oriented schlieren technique: sensitivity, accuracy, resolution and application to a three-dimensional density field, DOI `10.1007/s00348-007-0331-1`
- Practical aspects of designing background-oriented schlieren (BOS) experiments for vortex measurements, DOI `10.1007/s00348-023-03602-1`
- A lightweight convolutional neural network to reconstruct deformation in BOS recordings, DOI `10.1007/s00348-023-03618-7`
- Neural optical flow for planar and stereo PIV, DOI `10.1007/s00348-025-04058-1`
- Particle image velocimetry - Classical operating rules from today's perspective, DOI `10.1016/j.optlaseng.2020.106185`
- A method for automatic estimation of instantaneous local uncertainty in particle image velocimetry measurements, DOI `10.1007/s00348-012-1341-1`
- Deep recurrent optical flow learning for particle image velocimetry data / RAFT-PIV dataset, DOI `10.1038/s42256-021-00369-0`
- Evaluation of aero-optical distortion effects in PIV, DOI `10.1007/s00348-005-1002-8`
- Stereoscopic particle image velocimetry in inhomogeneous refractive index fields of combustion flows, DOI `10.1364/AO.431977`
- A novel method for correction of temporally- and spatially-variant optical distortion in planar PIV, DOI `10.1088/0957-0233/27/8/085201`
- Flow Field Estimation with Distortion Correction Based on Multiple Input Deep Convolutional Neural Networks and Hartmann-Shack Wavefront Sensing, DOI `10.3390/photonics11050452`
- Particle Image Velocimetry for MATLAB: Accuracy and enhanced algorithms in PIVlab, DOI `10.5334/jors.334`
- OpenPIV-Python CPU / piv-image-generator / SynthPix, DOI `10.3390/fluids8110285`、`10.1016/j.softx.2020.100537`、`10.1016/j.softx.2026.102642`
- A pyramid approach for background-oriented schlieren tomography, DOI `10.1007/s00348-025-04153-3`
- Subpixel-accurate real-time BOS measurement via GPU-optimized optical flow algorithms, DOI `10.1007/s00348-026-04277-0`
- Quantifying numerical uncertainty in background-oriented schlieren, DOI `10.1007/s00348-023-03734-4`
- Benchmark evaluation of tomographic algorithms for simultaneous reconstruction of temperature and volume fraction fields of soot and metal-oxide nanoparticles in non-uniform flames, DOI `10.1007/s11431-019-1507-6`
- A quantitative evaluation method of 3D flame curvature from reconstructed flame structure, DOI `10.1007/s00348-020-2905-0`

### 为什么读

这一组回答一个很现实的问题：本科毕业设计不一定要完整复现 NeRIF 或 4D BOST，仍然可以做出对课题组有用的工具链。Goldhahn 和 Schwarz 这类实验设计论文告诉你 sensitivity、blur、resolution、geometry 这些硬约束从哪里来；PIV operating rules、Timmins uncertainty、Elsinga、Vanselow 和 Zha 让 M3A 不止是速度场 toy，而能谈真实 PIV 图像互相关、逐点置信度、stereo-PIV 折射误差，以及 raw image dewarping vs vector-field correction 的工程选择；PIVlab、OpenPIV-Python、piv-image-generator、SynthPix、AOPIV-Net、LIMA、GPU optical flow、neural optical flow 和 RAFT-PIV 告诉你前处理/位移估计怎样变成可做的小项目；pyramid BOST 说明传统重构如何处理大密度梯度和多分辨率 refine；蔡组两篇旧谱系说明重构结果最终要服务于温度、体积分数、curvature、topology 等物理量，而不只是漂亮切片。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 实验设计 | sensitivity、geometric blur、circle of confusion、effective sensitivity | 真实实验参数风险表 |
| 位移估计 | cross-correlation、dot tracking、CNN deformation、optical flow、neural optical flow、RAFT-PIV、PIVlab/OpenPIV 参数 | 位移估计速度-误差-输入需求表 |
| PIV 图像层 | raw particle image dewarping、correlation/displacement correction、velocity-vector correction、synthetic particle image generation | M3A image-pair benchmark 设计 |
| refinement | coarse-to-fine、projection correction、warping | M2/M3B refinement 变量表 |
| 不确定度 | random uncertainty、numerical uncertainty、PIV local uncertainty、bias、sharp gradient effect | M3A/M3C uncertainty map 设计 |
| 物理量 | temperature、volume fraction、curvature、topology | “重构后能算什么”列表 |
| 边界 | 前处理/传统 refine 与 NeRIF/4D BOST 的区别 | 开题答辩风险说明 |

### 一周内能做的动作

- 给 M2 加一个 displacement-quality report，不改主模型；报告至少含位移噪声、窗口大小/网格间距、bad view mask 和 uncertainty proxy。
- 给 M3A 加一个 image-layer 升级草图：piv-image-generator/SynthPix synthetic particle images -> PIVlab/OpenPIV / neural optical flow -> raw image dewarping 或 velocity-field correction -> uncorrected/corrected velocity error。
- 把 M3B 的输出从 RMSE 扩成 rank-memory-time + temporal smoothness。
- 在开题 PPT 加一页“重构不是终点：折射率/密度/温度/速度/curvature 的物理量链路”。

### 问何远哲

- 师兄更希望我把时间花在 neural reconstruction 本身，还是做一个能帮组里检查位移场/重构质量/物理量导出的工具？
- 如果做 PIV-BOST，师兄真实流程更需要我处理原始粒子图像、互相关位移场，还是 PIVlab/OpenPIV/DaVis 导出的速度矢量场？

---

## Card 13：OERF CTC / Endoscopic Tomography Bridge

### 正式信息

核心阅读组合：

- Numerical and experimental validation of a three-dimensional combustion diagnostic based on tomographic chemiluminescence, DOI `10.1364/OE.21.007050`
- Practical aspects of implementing three-dimensional tomography inversion for volumetric flame imaging, DOI `10.1364/AO.52.008106`
- Demonstration of 3D computed tomography of chemiluminescence with a restricted field of view, DOI `10.1364/AO.56.007107`
- On the quantification of spatial resolution for three-dimensional computed tomography of chemiluminescence, DOI `10.1364/OE.25.024093`
- High spatial resolution computed tomography of chemiluminescence with densely sampled parallel projections, DOI `10.1364/OE.27.021050`
- Time-resolved measurements of a swirl flame at 4 kHz via computed tomography of chemiluminescence, DOI `10.1364/AO.57.005962`
- Development of an absorption-corrected method for 3D computed tomography of chemiluminescence, DOI `10.1088/1361-6501/ab01c1`
- Numerical and experimental validation of a single-camera 3D velocimetry based on endoscopic tomography, DOI `10.1364/AO.58.001363`
- Reconstruction and analysis of non-premixed turbulent swirl flames based on kHz-rate multi-angular endoscopic volumetric tomography, DOI `10.1016/j.ast.2019.05.025`
- Measurement of the 3D Rayleigh index field via time-resolved CH* computed tomography, DOI `10.1016/j.ast.2019.105487`
- Optimization of camera arrangement for volumetric tomography with constrained optical access, DOI `10.1364/JOSAB.385291`
- Tomographic reconstruction of an azimuthally forced flame in an annular chamber, DOI `10.1016/j.proci.2022.08.051`

### 为什么读

这一组是蔡老师课题组从 CTC / endoscopic tomography 走向 BOST / NeRIF 的“桥”。它不是何远哲主线，但能把开题报告里最关键的逻辑补齐：真实反应流三维测量长期受限于视场、视角数、相机布置、吸收/折射误差和空间分辨率；NeRIF / PIV-BOST / 4D BOST 本质上是在这个实验限制下引入更强的计算重构。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 实验验证 | hybrid reconstruction、view orientation、random vs restricted projections | M3C view-orientation / validation toy 变量表 |
| 实现细节 | calibration、projection number、reconstruction parameters、practical constraints | data_request_checklist 和 baseline 公平性 checklist |
| 受限视场 | restricted FOV、blocked projection、缺失投影怎么处理 | M3C bad-view / mask toy 变量表 |
| 分辨率指标 | resolution phantom、体素尺寸、投影数、dense projection sampling | 论文指标页：L2 之外加 resolution/edge sharpness |
| 系统误差 | absorption correction、projection mismatch、calibration error | forward model error list |
| 视角选择 | camera arrangement objective、projection correlation | view-selection 小实验问题定义 |
| 物理接口 | Rayleigh index、3D velocity、flame topology、annular-combustor phase averaging | “重构场服务什么物理量”段落 |

### 一周内能做的动作

- 在 `demo_m2/` 或 future `demo_m3c/` 里加一个 mask/bad-view 子实验：随机移除 1-3 个视角，看重构误差如何变化。
- 写一页“为什么本科先做系统误差/指标工具而不是完整复现 TOG 4D BOST”：这会让选题更稳。
- 把 `data_request_checklist.md` 增补一个问题：真实 BOST/NeRIF 数据是否有视角 mask、裁剪框、标定矩阵和坏视角记录。

### 问何远哲

- 师兄这里最痛的是 neural field 重构精度，还是前处理/视角/标定/坏视角导致的系统误差？
- 如果只能做一个本科可交付工具，M3C 应优先做 bad-view robustness、view-selection，还是 spatial-resolution metric？

---

## Card 14：TAS / NTAS Computational Reconstruction

### 正式信息

核心阅读组合：

- A tomographic technique for simultaneous imaging of temperature, chemical species, and pressure, DOI `10.1063/1.4862754`
- Development of a beam optimization method for absorption-based tomography, DOI `10.1364/OE.25.005982`
- Benchmark evaluation of inversion algorithms for tomographic absorption spectroscopy, DOI `10.1364/AO.56.002183`
- On the regularization for nonlinear tomographic absorption spectroscopy, DOI `10.1016/j.jqsrt.2017.11.016`
- Reconstruction for limited-data nonlinear tomographic absorption spectroscopy via deep learning, DOI `10.1016/j.jqsrt.2018.07.011`
- Compressing convolutional neural networks using POD for NTAS, DOI `10.1016/j.cpc.2019.03.020`
- Deep learning algorithms for temperature field reconstruction of NTAS, DOI `10.1016/j.measen.2020.100024`
- Tomographic absorption spectroscopy based on dictionary learning, DOI `10.1364/OE.440709`

### 为什么读

TAS / NTAS 不是你的主线，但它是蔡老师“光学测量 + 逆问题 + 计算重构”谱系里最清楚的一条。它比 BOST 更早系统讨论了有限光路、反演算法 benchmark、非线性 forward model、正则化、deep learning 加速和 learned prior。读它的目的不是转向吸收光谱，而是给 NeRIF/BOST 选题补一个更成熟的反问题语言。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 光路/视角设计 | beam optimization、weight matrix orthogonality | view-selection 指标候选 |
| baseline 公平性 | inversion algorithm benchmark | M1/M2 baseline 对比表结构 |
| 正则化 | nonlinear tomography regularization | NeRIF/BOST loss terms 风险说明 |
| deep learning | CNN/RNN/DBN、limited data、毫秒级重构 | AI for diagnostics 课题组前史段落 |
| learned prior | POD compression、dictionary learning | 4D low-rank 与 neural prior 的概念桥 |

### 一周内能做的动作

- 不复现 TAS，只从 beam optimization 提取一个 view/projection correlation 指标，放入 M3C 的 view-selection 方案。
- 写一个 related-work 小表：TAS 光束优化 vs CTC 相机布置 vs BOST 视角选择 vs NeRIF 神经隐式场。
- 给开题报告第 2 章加一句：蔡组已有 TAS/CTC/BOST 多条 tomography 谱系，本课题选择 BOST 是因为它直接对应何远哲当前数据和论文。

### 问何远哲

- 如果我做 view-selection / bad-view robustness，小实验是否可以借鉴 TAS beam optimization 的投影相关性指标？
- 师兄更希望 related work 强调 BOST/NeRIF 文献，还是也保留 TAS/NTAS 作为蔡老师课题组反问题谱系？

---

## Card 15：Tomography Geometry / Mesh / Data Assimilation Bridge

### 正式信息

核心阅读组合：

- Assessment of imaging models for volumetric tomography of fluid flows, DOI `10.1016/j.measurement.2022.112174`
- A reconstruction method for volumetric tomography within two parallel transparent plates, DOI `10.1016/j.optlaseng.2023.107699`
- Laser absorption tomography based on unstructured meshing, DOI `10.1088/1361-6501/ad068f`
- Correction procedure for a tomographic optical setup employing imaging fiber bundles and intensified cameras, DOI `10.1364/AO.507266`
- CTIS-GAN, DOI `10.1364/AO.478230`
- Data assimilation using ensemble Kalman filter and low-dimensional manifolds for reacting flow, DOI `10.1063/5.0255969`

### 为什么读

这组不是何远哲主线论文，但它正好补上 NeRIF/BOST 毕设最容易被忽略的“真实实验层”：投影模型是否足够准、光学窗口是否改变光路、规则体素是否适合边界、光纤束/增强相机数据是否需要专门校正，以及重构出来的折射率/密度场未来如何进入数据同化。

如果你只写“用神经网络重构三维场”，题目会显得泛。如果你能写清楚“在少视角、真实几何、光路误差和数据接口约束下，评估神经隐式表示的适用边界”，就明显更贴 OERF 的需求。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| imaging model | 不同投影模型的精度、速度、适用范围 | M3C 投影模型选择表 |
| transparent window | 平行透明板/光窗对 volumetric tomography 的影响 | 真实实验误差来源清单 |
| mesh | 非结构网格相对规则体素的优势和代价 | voxel / mesh / neural field 对照表 |
| correction | fiber bundle 与 intensified camera 的校正流程 | data loader 前处理 checklist |
| AI spectrum | CTIS-GAN 里的数据立方重构和 artifact reduction | OERF AI reconstruction 横向背景 |
| assimilation | EnKF + low-dimensional manifold 如何用观测校正反应流 | 第 6 章未来展望，不开局承诺复现 |

### 一周内能做的动作

- 在 `demo_m3c/` 草拟一个 geometry sanity check：同一 phantom 用简化 parallel-ray 和带 blur/shift 的 forward model 生成投影，比较重构误差。
- 做一个 `voxel_mesh_neural_field.md` 小表：规则体素、非结构网格、NeRIF 隐式场各自适合什么实验限制。
- 给 `data_request_checklist.md` 加问题：真实数据是否有 camera matrix、view angles、mask、window geometry、background-dot displacement confidence、fiber-bundle correction 结果。

### 问何远哲

- 当前 NeRIF/BOST 数据的 forward model 是平行光近似、pinhole camera，还是组内已有 ray-tracing / calibration matrix？
- 真实实验里更痛的是视角少、位移噪声、光窗折射、mask/遮挡，还是相机/光纤束校正？
- 本科毕设做 M3C 几何误差小 benchmark，对师兄后续论文/代码是否有实际帮助？

---

## Card 16：M3C Real Forward Model / Neural Tomography Sanity Checks

### 正式信息

核心阅读组合：

- Forward and Inverse Modeling of Depth-of-Field Effects in Background-Oriented Schlieren, AIAA Journal 2024, DOI `10.2514/1.J064095`，arXiv `2402.15954`。
- Localized gradient-index field reconstruction using background-oriented schlieren, Applied Optics 2019, DOI `10.1364/AO.58.007795`。
- Investigation of a neural implicit representation tomography method for flow diagnostics, Measurement Science and Technology 2024, DOI `10.1088/1361-6501/ad296a`。

### 为什么读

这一组直接服务 M3C：真实 BOST 数据不只难在视角少，也难在 forward model 是否真的匹配相机、背景图案、光圈、景深、光窗和位移估计。DoF-BOS 论文提醒你 thin-ray / pinhole 近似可能在有限光圈和景深 blur 下失配；localized gradient-index 论文提醒你 BOS 的物理语言经常是 deflection angle 与 `grad n`，不只是重构一个漂亮的 `n(x,y,z)`；NIRT 论文帮助你把 NeRIF 放进更宽的 coordinate-field tomography 家族，说明 NeRIF 的价值在于把神经隐式表示和 BOS 折射率物理绑在一起。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| forward model | thin-ray、pinhole、cone-ray、finite aperture、f-number、depth-of-field blur | M3C 投影/成像模型对照表 |
| 观测变量 | displacement、deflection angle、localized gradient-index、refractive index | 变量链条图：图像 -> 位移 -> `grad n` -> `n` |
| 神经表示 | voxel / SART / coordinate MLP / NeRIF / NIRT 的区别 | related-work 小表 |
| 误差指标 | blur 半径、f-number、reprojection error、weighted RMSE、bad-view sensitivity | `demo_m3c` 指标草案 |
| 真实数据问题 | camera geometry、aperture、DOF、mask、MTF、位移置信度 | 给何远哲的数据问题清单 |

### 一周内能做的动作

- 不完整复现 cone-ray，只先在 M2/M3C toy 里加入一个 blur / shift proxy：同一 phantom 生成 clean displacement 与 blurred displacement，比较重构偏差。
- 写一页 `forward_model_sanity_check.md`：哪些误差可由公开数据模拟，哪些必须等组内相机/光路参数。
- 把 `data_request_checklist.md` 里的“视角几何”问题扩成：相机内参、光圈/f-number、背景距离、测量体距离、景深、mask、bad view、位移置信度、是否有光窗。

### 问何远哲

- 组内 NeRIF/BOST 当前 forward model 使用 thin ray、pinhole，还是已有 ray tracing / cone-ray / calibration matrix？
- 九视角或内窥系统里，景深、光圈、背景/测量体距离和光窗折射是否是明显误差源？
- 如果本科阶段做一个 geometry / depth-of-field / bad-view sanity-check report，会不会比单纯调 neural field 结构更有用？
- 师兄希望我在相关工作里主动比较 NeRIF 与通用 NIRT / FluidNeRF，还是只保留 BOS/BOST 文献？

---

## Card 17：CTIS Computational Spectrometry as an Inverse-Problem Neighbor

### 正式信息

核心阅读组合：

- Computed tomography imaging spectrometry based on superiorization and guided image filtering, DOI `10.1364/OL.418355`
- Super-resolution computed tomography imaging spectrometry, DOI `10.1364/PRJ.472072`
- CTIS-GAN, DOI `10.1364/AO.478230`

### 为什么读

CTIS 不是何远哲 BOST 主线，但它和 BOST/NeRIF 共享同一个研究母题：有限观测下恢复高维物理/光谱数据立方。CTIS 的“少投影 + guided filtering / RGB guidance / GAN prior”可以帮你把 NeRIF 写得更稳：神经隐式场不是凭空套 AI，而是 OERF 长期做的 computational reconstruction 逻辑在 BOST 方向上的延伸。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 有限投影 | CTIS 为什么天然 ill-posed | 和 sparse-view BOST 的类比段 |
| 引导先验 | guided filtering、RGB guidance、zero-order diffraction | “外部先验如何约束重构”的小表 |
| AI prior | CTIS-GAN 的 artifact reduction 和速度动机 | AI reconstruction 谱系一句话 |
| 边界 | 光谱数据立方和折射率场的物理差异 | 避免开题跑题的限制段 |

### 一周内能做的动作

- 不复现 CTIS，只读摘要、图 1/系统图、方法动机和实验指标。
- 写一个类比表：CTIS hyperspectral cube vs BOST refractive-index volume vs 4D BOST tensor field。
- 在开题报告中只放 1-2 句：OERF 在计算光谱和流场可视化中都使用“有限观测 + 计算重构”的范式，本课题选择 BOST 是因为它最贴近何远哲当前工作。

---

## Card 18：OERF Holography / Metal-Particle Diagnostics Backup Line

### 正式信息

核心阅读组合：

- Quantification of the size, 3D location and velocity of burning iron particles in premixed methane flames using high-speed digital in-line holography, DOI `10.1016/j.combustflame.2021.111430`
- Clustering-based particle detection method for digital holography to detect the three-dimensional location and in-plane size of particles, DOI `10.1088/1361-6501/abd7aa`
- 3D particle sizing, thermometry and velocimetry of combusting aluminized propellants, DOI `10.1016/j.combustflame.2022.112500`
- Investigation on heat exchange of iron particle combustion based on simultaneous multi-physics field measurements, DOI `10.1016/j.combustflame.2025.114309`
- Robust 3D tracking of dynamic reacting particles based on holographic spatio-temporal similarity, DOI `10.1016/j.combustflame.2026.114899`
- Insights into the micro-explosion of burning iron particle under ammonia co-firing conditions, DOI `10.1016/j.proci.2026.106209`

### 为什么读

这条线不是何远哲 BOST/NeRIF 主线，但它是 OERF “全息成像 + 颗粒燃烧 + 多物理诊断”的强旁支。如果师兄后续给的不是 BOST 数据，而是金属颗粒图像、全息重构结果或轨迹数据，这组论文能快速把题目改写成一个仍然贴课题组需求的本科题：颗粒检测、三维定位、轨迹质量评估、粒径/速度/温度同步数据整理。

### 必须抽取

| 类型 | 要抽取什么 | 产物 |
| --- | --- | --- |
| 数据形态 | 原始 hologram、重建体、粒子中心/尺寸/速度/温度表 | 数据 manifest 备选字段 |
| 图像算法 | clustering-based detection、3D tracking、spatio-temporal similarity | 备选算法题清单 |
| 多物理场 | particle size、temperature、velocity、radiation / heat exchange | 若转颗粒方向的结果图模板 |
| 风险边界 | 燃烧机理依赖强、数据依赖强、和 He 主线距离较远 | 不把它和 NeRIF 混成一个大题 |

### 一周内能做的动作

- 不转主线，只写一个“若师兄给颗粒/全息数据时的备选方案”。
- 在 `data_request_checklist.md` 里增加可选问题：是否有 hologram、particle centerline、track ID、diameter、temperature、velocity 和 frame time。
- 如果真的拿到数据，优先做数据质量报告和 3D tracking 可视化，不先写燃烧机理。

### 问何远哲

- 组里近期是否有颗粒/全息数据需要本科生做后处理，还是仍以 BOST/NeRIF 为优先？
- 如果转颗粒方向，预期贡献是 detection/tracking/visualization，还是热交换/微爆机理分析？
- 数据是否已有标注或可信 reference track？如果没有，本科阶段如何评价算法结果？

---

## 读完后的周报模板

每周结束时，把读到的论文压成下面 7 行：

```text
本周论文：
最重要公式/模型：
我改了哪个 demo：
生成的图：
一个失败点：
一个能写进开题的句子：
要问何远哲的问题：
```

如果一篇论文读完后无法填这 7 行，说明阅读还没有转化成毕业设计推进。

---

## 本文件和其他文件的关系

- 选论文顺序看 `paper_to_demo_map.md`。
- 何远哲三篇主线的深挖看 `he_paper_deep_dive.md`。
- 每周阅读登记看 `reading_log_template.md`。
- 代码接口看 `nerif_bost_implementation_blueprint.md`。
- 开题写法看 `opening_report_draft.md` 和 `opening_ppt_outline.md`。
