# 高优先级毕设选题 Proposal 卡

用途：和何远哲沟通时，不要只说“我想做 BOST/NeRIF”，而是拿出可以被师兄选择、修改或否决的具体方案。这里按“可毕业性 + 贴何远哲 + 对课题组有用 + 风险可控”整理，并新增 T16 算子学习与三维重建结合方案。

---

## 选题 A：NeRIF/BOST 鲁棒性主线

### 推荐题目

面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

### 对齐论文

- He et al., Neural Refractive Index Field, Physics of Fluids, 2025.
- Lu et al., Neural Refractive Index Primitives, Combustion and Flame, 2026.
- Raffel, Background-oriented schlieren techniques, Experiments in Fluids, 2015.
- Grauer et al., Instantaneous 3D flame imaging by BOST, Combustion and Flame, 2018.
- Liu et al., Volumetric imaging of flame refractive index/density/temperature using BOST, Science China Technological Sciences, 2020.
- Grauer and Steinberg, UBOST, Experiments in Fluids, 2020.

### 核心问题

少视角、含噪、几何不完整条件下，神经隐式折射率场相对传统体素/正则化 baseline 到底在哪些区域更稳，在哪些区域不值得用？

### 最小可毕业版本

- 自建 3D synthetic refractive-index phantom。
- 生成多视角 BOS-like displacement field。
- 做传统 baseline：FBP/SIRT/Landweber/Tikhonov 中至少一种。
- 做简化 NeRIF：坐标 MLP 或 random Fourier feature 表示。
- 扫描 view count、noise level、field capacity、sampling strategy。
- 输出误差切片、重投影误差、视角数/噪声曲线、运行时间。

### 进阶版本

- 接入 OERF 真实九视角 BOST 数据或开源 BOS tomography dataset。
- 对比 NeRIF、NeDF、NRIP、UBOST 风格方法的表示差异。
- 增加 view orientation、spatial resolution phantom、mask/ROI、相机几何误差、背景位移提取误差、Fourier/hash-like encoding 的敏感性分析。

### 数据/代码需求

- 最低：自己生成 synthetic 数据。
- 希望：真实 displacement field、相机几何、mask、参考重构。
- 代码：Python + NumPy/SciPy/scikit-image；进阶用 PyTorch。

### 要问何远哲

1. 真实 BOST 数据的主要失败模式是视角少、噪声大、几何误差，还是重建速度慢？
2. 组里已有 baseline 是 voxel/SIRT/UBOST 风格，还是 NeRIF 论文代码？
3. 真实数据是否已经有 view geometry、mask、resolution/MTF 指标或 bad-view 记录？
4. 是否值得把 NRIP 的 hash encoding、automatic/discrete gradient loss 和 3D mask 写成外部对照，还是只作为 related work？
5. 毕设结果更需要“精度提升”还是“参数扫描和自动报告工具”？

### 风险评级

- 可毕业性：高。
- 新颖性：中高。
- 数据依赖：低到中。
- 最大风险：完整 NeRIF/NRIP 论文级复现可能过重，应先做简化版；NRIP 只做 encoding/mask/loss 消融线索，不承诺完全复现。

---

## 选题 B：PIV-BOST 折射补偿升级线

### 推荐题目

基于 BOST 折射率场的 PIV 速度测量折射误差补偿与误差传播分析

### 对齐论文

- Zheng et al., Instantaneous refractive index compensation on the velocity measurement using simultaneous PIV-BOST, Experiments in Fluids, 2025.
- Zheng et al., Instantaneous refractive index compensation on stereo-velocity measurement in turbulent combustion, Proceedings of the Combustion Institute, 2026.
- He et al., NeRIF, Physics of Fluids, 2025.
- Vanselow and Fischer 2018 / Vanselow et al. 2021：折射率场导致 standard / stereo PIV 误差。
- Elsinga et al. 2005 和 Zha et al. 2016：BOS/PIV 光学畸变评估、raw image dewarping 和 velocity-field correction。
- Thielicke and Sonntag 2021 PIVlab；OpenPIV-MATLAB/Python；piv-image-generator / SynthPix。

### 核心问题

热流/火焰折射率梯度会让粒子图像位置产生偏移，从而污染 PIV 速度。BOST 重构出的折射率场如何转化为速度补偿量？补偿误差对折射率重构误差有多敏感？更具体地说，补偿应该发生在 raw particle image、correlation/displacement map，还是 velocity-vector field？

### 最小可毕业版本

- 保留现有 M3A velocity-vector toy：真实速度 + 折射位移变化量 = 观测速度。
- 生成最小 synthetic particle image pair，先不依赖真实 PIV-BOST 原始数据。
- 用 PIVlab/OpenPIV 跑传统互相关，记录 window size、overlap、validation 和 outlier replacement。
- 插入简化折射畸变，比较 raw image dewarping、displacement correction 和 velocity-field correction。
- 扫描 deflection noise、PIV displacement magnitude、time interval、gradient strength。
- 输出补偿前后 RMSE、p95 error、局部残余误差图和三层接口图。

### 进阶版本

- 使用 piv-image-generator / SynthPix 风格生成合成粒子图像，做真实互相关级别补偿。
- 接入何远哲或郑聿韬相关真实 PIV-BOST 数据。
- 扩展到 stereo-PIV-BOST 的误差传播框架，但不承诺完整 stereo 几何。

### 数据/代码需求

- 最低：合成速度场 + 合成折射位移场 + synthetic particle image pair。
- 希望：同步 PIV 图像、BOST displacement/reconstruction、PIV 参数、相机标定。
- 工具：PIVlab/OpenPIV、piv-image-generator/SynthPix、NumPy/SciPy、Matplotlib；进阶用真实 PIV 处理脚本。

### 要问何远哲

1. PIV-BOST 真实项目里，当前最缺的是补偿算法、误差传播、数据可视化，还是实验数据处理自动化？
2. 组里是否允许本科生接触同步 PIV-BOST 样例数据？
3. 补偿应该发生在原始粒子图像层、互相关/位移层，还是速度矢量场层？
4. 组内能否给一对 PIV raw image、BOST 位移/折射率场，以及 PIVlab/OpenPIV/DaVis 导出的速度场作接口样例？

### 风险评级

- 可毕业性：中。
- 新颖性：高。
- 数据依赖：中高。
- 最大风险：没有真实 PIV raw image 时容易停留在 synthetic image-pair benchmark；所以应作为 NeRIF 主线后的升级章节，或和师兄明确它就是工具型支线。

---

## 选题 C：4D BOST 跨形态无真值容量选择

### 推荐题目

面向少视角四维背景纹影层析的可观测时序容量选择与可靠性评价

### 对齐论文

- He et al., Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography, ACM Transactions on Graphics, 2026.
- Molnar and Grauer, Algorithm for Time-Resolved BOST Applied to High-Speed Flows, AIAA SCITECH, 2025.
- Huang et al., Online in situ prediction of 3-D flame evolution from 2-D projections via deep learning, Journal of Fluid Mechanics, 2019.
- Huang et al., Limited-projection volumetric tomography for time-resolved turbulent combustion diagnostics via deep learning, Aerospace Science and Technology, 2020.
- Cai et al., Volumetric reconstruction via transfer learning and semi-supervised learning with limited labels, Aerospace Science and Technology, 2021.

### 核心问题

高速流场的三维结构随时间演化。固定低秩先验会因 morphology、noise、views 和 dynamics 不同而过度平滑或保留噪声。能否只依据 support data consistency、factor spectrum 和时序物理统计，在没有 field truth 的情况下选择表示容量，并在不确定时回退 full rank 或 NeRIF？

### 最小可毕业版本

- 当前已完成 4 morphology × 3 dynamics × 4 noise × 3 views × 6 seeds 的 864 个观测格。
- 同一 framewise 重建共享奇异分解，比较 rank 1/2/3/5/8/12/full 共 6,048 个候选。
- 外层 leave-one-phantom-family-out；测试时禁止使用 family、noise label、field truth 和 oracle rank。
- 组合 spectrum、support residual、temporal derivatives 和 integral traces 选择容量。
- 报告 mean/p95/worst oracle regret、1%-oracle coverage、full-rank 拒绝和平滑失败格。

### 进阶版本

- 将 ridge baseline 升级为带 uncertainty、risk calibration 和 abstention 的容量控制头。
- 从 leave-one-family-out 升级到 leave-one-geometry-out，并加入 bias/sync/exposure blur。
- 接一小段真实 4D BOST 数据，把容量映射到 TDBOST plane factors、feature width 或 temporal operator。
- 和 T16 support-consistent 3D neural operator 结合，低置信时触发 full-rank/NeRIF refinement。

### 数据/代码需求

- 最低：现有 864 格 synthetic benchmark、配置、CSV、六张图和 validator 已完成。
- 希望：18-30 帧真实 deflection/reconstruction、小型 camera geometry、support/test view、mask 和必须保真的物理量。
- 工具：NumPy/SciPy、tensorly 可选、PyTorch 可选。

### 要问何远哲

1. TDBOST 中哪一个 rank/plane factor/feature width 是组里真正需要按 case 调的容量？
2. 能否给小型 geometry、support/test view 和真实 4D 样例；held-out camera 是否现实？
3. 下一轮最有价值的是 geometry transfer、UQ/拒答、bias/sync/exposure blur，还是直接接真实数据？

### 风险评级

- 可毕业性：中。
- 新颖性：高。
- 数据依赖：中。
- 最大风险：synthetic morphology transfer 不等于真实泛化；创新点不能写成“ridge 选 rank”，必须落在 BOST 可观测容量控制、geometry transfer 和可靠性机制。

---

## 选题 D：BOST/NeRIF 自动报告与数据接口工具

### 推荐题目

面向 OERF BOST/NeRIF 数据的可复现实验管理、参数扫描与自动报告工具

### 适用场景

如果师兄最缺的是工程工具，而不是一个新的重构模型，这个题目反而可能最有组内价值。

### 核心问题

如何把不同来源的 BOST、PIV-BOST、4D BOST 数据统一成 manifest + baseline + metrics + visualization + report 的可复现流程？

### 最小可毕业版本

- 制定 BOST/PIV-BOST 数据 manifest。
- 自动检查缺失图像、几何参数、mask、单位、帧数。
- 批量运行 baseline 或已有重构结果评估。
- 自动生成误差图、重投影图、参数扫描表和简短 HTML 报告。

### 要问何远哲

1. 组里真实数据是否经常因为格式、路径、参数和图表不统一而难以复现实验？
2. 这个工具是否能服务师兄正在写的论文或后续学生？
3. 是否可以把它作为主线 A 的工程交付，而不是单独毕设题？

### 风险评级

- 可毕业性：高。
- 新颖性：中。
- 数据依赖：中。
- 最大风险：如果没有真实数据接口，容易变成空架子；最好作为 A/B/C/E 的配套工具。

---

## 选题 E：BOST 系统误差与少视角投影补全副线

### 推荐题目

面向火焰 BOST 三维重构的时间不同步、少视角缺失与神经投影补全误差分析

### 适用场景

如果师兄觉得“只复现 NeRIF”不够像研究，或者真实数据暂时不能给，这个题目可以作为 A 的增强版：不换主线，只把变量从 view count / noise 扩展到 time asynchrony、camera missing views、projection synthesis 和 uncertainty。

### 对齐论文

- Gao et al., Reconstruction Method of 3D Turbulent Flames by BOST and Analysis of Time Asynchrony, Fire, 2023.
- Jin et al., PENTAGON, Optics Express, 2024.
- Zhu et al., projection map synthesis / PIPEN, Optics Express, 2025.
- Zhang et al., NVRT / Fast-NFRT, Aerospace Science and Technology, 2023.
- Kelly and Thurow, FluidNeRF, AIAA SCITECH, 2023.
- He et al., NeRIF, Physics of Fluids, 2025.

### 核心问题

真实 BOST/4D BOST 不只受视角数和噪声影响，还会受相机触发时间差、视角方向、视角缺失、mask/ROI、ray tracing 计算近似和投影补全误差影响。哪些误差最值得组里优先建模和自动报告？

### 最小可毕业版本

- 在现有 M1/M3B synthetic phantom 上加入 time offset：不同视角看到略不同时刻的场。
- 模拟 missing views：随机丢视角、固定遮挡视角、两正交视角。
- 模拟 view orientation 和 mask/ROI：比较随机/均匀/受限视角、不同有效视场裁剪对分辨率和重投影误差的影响。
- 做三个 baseline：直接重构、简单插值补视角、轻量 projection synthesis 或低秩补全。
- 输出 time offset / missing view / noise 三维误差热图。
- 给出“真实实验同步误差容忍范围”的估计，而不是只报一个平均误差。

### 进阶版本

- 接入开源 BOS dataset 或 OERF 真实 BOST 序列。
- 把 PENTAGON/PIPEN 的思想简化成可解释的 projection-map synthesis baseline。
- 与 4D BOST 的低秩时序模块合并，形成“时间同步误差 + 低秩时序先验”实验。

### 要问何远哲

1. 真实九视角/4D BOST 中，相机或视角同步误差是否是已知痛点？
2. 如果某些视角质量差或缺失，组里现在是丢帧、插值、重构时加 mask，还是手动筛掉？
3. 师兄更需要“算法补全视角”，还是“自动判断这组数据能不能可信重构”的质量控制工具？

### 风险评级

- 可毕业性：高。
- 新颖性：中高。
- 数据依赖：低到中。
- 最大风险：如果做成复杂深度网络会跑偏；本科版本应先做误差图谱和轻量补全 baseline。

---

## 选题 F：BOS/PIV-BOST 位移估计与质量控制工具线

### 推荐题目

面向 PIV-BOST 与背景纹影层析的图像位移估计 benchmark、误差传播与数据质量控制

### 适用场景

如果师兄更需要你帮忙处理真实图像、清洗数据、判断哪些视角/区域可信，而不是马上提出一个新的三维重构网络，这条线非常适合作为 A/B/E 的工程支撑。它不抢 NeRIF 主线，而是解决 BOST/NeRIF/PIV-BOST 共同的入口问题：输入位移场是否可信。

### 对齐论文

- Zheng et al., real-time GPU-accelerated BOS with optical flow, Experiments in Fluids, 2026, DOI `10.1007/s00348-026-04277-0`。
- Wolf et al., wavefront aberration measurements based on BOS, Measurement: Sensors, 2025, DOI `10.1016/j.measen.2024.101509`。
- Liu et al., atmospheric turbulence effect on shock detection using BOS, Optics Express, 2025, DOI `10.1364/OE.567012`。
- Zheng/He et al., simultaneous PIV-BOST, Experiments in Fluids, 2025, DOI `10.1007/s00348-025-04093-y`。
- OpenPIV / PIVlab / PIV operating rules / PIV uncertainty / RAFT-PIV / classic optical flow 文献链。

### 核心问题

BOST 和 PIV-BOST 的第一步通常不是神经网络，而是从背景图案或粒子图像中提取位移。不同光流、互相关、DIC 或神经光流方法会给出不同的位移、置信度和坏区域。对于真实火焰或高速流图像，应该怎样判断某个视角、某个区域、某一帧是否足够可信，能不能进入 NeRIF/BOST 重构或 PIV-BOST 补偿？

### 最小可毕业版本

- 生成 synthetic background image pair 和 synthetic particle image pair。
- 比较 Farneback、DIS/TV-L1 或 OpenPIV 互相关；如果 GPU 和数据允许，再加 RAFT-lite/RAFT-PIV 作为概念对照。
- 扫描图像噪声、背景纹理、位移幅值、局部 blur、wavefront-like distortion 和 bad region。
- 输出 endpoint error、angular error、bad-pixel mask、局部置信度图、运行时间和失败案例。
- 把结果接入 `data_request_checklist.md` / BOST manifest：每个 view/frame 给出 health score、mask、推荐是否进入重构。
- 详细执行拆解见 `displacement_qc_route_f.md`。

### 进阶版本

- 接入 OERF 一小段真实 BOST/PIV-BOST 原始图像。
- 与 NeRIF 或 M3A/M3C demo 对接：看“位移场质量差异”如何传播到折射率重构或速度补偿误差。
- 加入 realtime GPU BOS 论文的处理速度指标，把工具做成“离线精度版 + 快速筛查版”。

### 数据/代码需求

- 最低：自己生成 background dot / random texture / Gaussian particle image pair。
- 希望：真实 reference/deformed background、PIV raw image pair、已有 displacement field、mask、相机视角编号、火焰区域 ROI。
- 工具：OpenCV optical flow、OpenPIV、NumPy/SciPy、Matplotlib；可选 PyTorch/RAFT-PIV。

### 要问何远哲

1. 组内 BOST/PIV-BOST 当前位移提取用的是互相关、光流、DIC、商用软件，还是论文自写代码？
2. 师兄现在更缺“更准的位移算法”，还是“自动标出坏视角、坏区域、坏帧”的质量控制？
3. 真实数据里最常见的失败是背景图案模糊、火焰强梯度、相机标定、wavefront aberration、外界湍流扰动，还是同步问题？
4. 如果我先做 synthetic benchmark，师兄希望输出哪些字段才能接进 NeRIF/BOST：位移场、置信度、mask、重投影误差、还是 health score？

### 风险评级

- 可毕业性：高。
- 新颖性：中。
- 数据依赖：低到中。
- 最大风险：如果只做纯 CV optical-flow 排行榜，会偏离 OERF；必须把输出接回 BOST/NeRIF/PIV-BOST 的 forward model、mask 和误差传播。

---

## 新增重点选题 T16：物理约束神经算子 × BOST 三维重建

### 推荐题目

面向少视角背景纹影层析三维重建的物理约束神经算子

### 对齐论文

- Lu et al., DeepONet, Nature Machine Intelligence, 2021.
- Li et al., Fourier Neural Operator, ICLR, 2021.
- Kovachki et al., Neural Operator, JMLR, 2023.
- Li et al., Physics-Informed Neural Operator, ICLR, 2022.
- Molinaro et al., Neural Inverse Operators, ICML, 2023.
- Zhao et al., RecFNO, International Journal of Thermal Sciences, 2024.
- Zhang et al., Operator Learning for Reconstructing Flow Fields from Sparse Measurements, Journal of Computational Physics, 2025.
- He et al., Neural Refractive Index Field, Physics of Fluids, 2025.

### 核心问题

一组共享模型能否把不同 phantom、视角、噪声和分辨率下的 BOST 观测快速映射到三维折射率场，并且在未见过的场型上仍保持重投影一致性？physics lift、operator architecture 和 reprojection loss 各自在哪些条件下真正有用？

### 最推荐模型

1. 用已知几何把多视角位移做伴随反投影或固定预算 SIRT，得到规则三维粗解 `z0`。
2. 把 `z0 + sampling density + valid mask + metadata` 输入 residual 3D FNO，预测修正量。
3. 用 field、gradient、boundary 和 BOST forward reprojection 联合训练。
4. DeepONet 作为第二架构：branch 编码固定视角观测，trunk 查询三维坐标。
5. 静态 operator 跑通后，才考虑 temporal operator 连接 4D BOST。

### 最小可毕业版本

- 批量生成 2D 与 24^3/32^3 BOST paired dataset。
- 对比 SIRT/Landweber、3D U-Net、direct FNO、physics-lift residual FNO。
- 按 phantom family 切分训练/测试，避免随机近邻样本泄漏。
- 扫描 3/5/7/9 views、noise、mask 和 resolution。
- 输出 unseen-family、held-out reprojection、精度/速度/显存 Pareto 和失败案例。

### 进阶版本

- 加入可变视角 mask/set encoder 或最小 DeepONet。
- 用 operator 输出初始化 NeRIF，比较“一次推理 + per-instance refinement”。
- 接入一小份 OERF 真实 BOST 数据，做 synthetic-to-real failure audit。
- 静态模型可靠后，再做 3D+T history-to-field operator。

### 数据/代码需求

- 最低：现有 M1/M2 forward model 批量生成 synthetic pair。
- 希望：组内相机几何、位移场、mask、参考重构和认可的传统 baseline。
- 软件：PyTorch、NeuralOperator、NumPy/SciPy；DeepXDE 只用于 DeepONet 最小实验。
- 硬件：2D/24^3 可在普通 GPU 起步；32^3/48^3 3D FNO 建议 12-24 GB 显存。无 GPU 时先完成 2D operator 与数据协议。

### 要问何远哲

1. “算子学习”指 projection-to-volume inverse operator，还是 3D/4D evolution operator？
2. 输入应是 raw image、位移场、投影，还是 SIRT/adjoint 粗解？
3. 输出应是 refractive index、density、temperature，还是 NeRIF/TDBOST 初值？
4. 组里是否已有 paired dataset、推荐 FNO/DeepONet 代码或固定 train/test split？
5. 是否接受 `physics lift + residual FNO`，以及最看重 field error、held-out view、速度还是跨工况泛化？

### 风险评级

- 可毕业性：中高，前提是从 2D/小 3D 起步。
- 新颖性：高，能把三维重建与师兄建议的模型设计结合。
- 数据依赖：中；synthetic 可启动，真实价值依赖组内数据确认。
- 最大风险：把“模型跑起来”误当论文结论。必须有 U-Net/传统法对照、OOD 切分、物理重投影和显存记录。

完整执行页：`operator_learning_bost_bridge.html`
实验规格：`operator_learning_bost_experiment_spec.md`

---

## 最终建议

### 旁支 G：TAS / NTAS 非线性反问题 toy

推荐题目：

吸收光谱层析非线性反问题的迭代重构与神经正则化 toy 研究

适用场景：

只有当蔡老师或师兄明确希望你转向 TAS / NTAS / 光谱层析时考虑。它和 NeRIF 共享“forward model + inverse problem + regularization”的数学语言，但实验对象不是 BOST。

对齐论文：

- Cai and Kaminski, Tomographic absorption spectroscopy for gas dynamics and reactive flows, PECS, 2017.
- Aragon-Artacho et al., iterative algorithms for nonlinear equations with TAS application, Communications in Optimization Theory, 2025.
- Deng/Cai NTAS deep algorithms、untrained neural network for LTAS、POD / regularization parameter TAS 文献链。

最小可毕业版本：

- 建一个 2D 温度/浓度 phantom。
- 用少量 beam / wavelength 生成吸收观测。
- 比较 Landweber / Gauss-Newton-like / fixed-point / Tikhonov toy。
- 只做算法与误差图，不碰真实光谱仪硬件。

放弃条件：

- 如果需要自己搭光谱实验或理解详细谱线数据库，立即降级为背景阅读。
- 如果何远哲仍带 BOST/NeRIF，不要把它抢成主线。

### 旁支 H：单相机/空间复用体测温

推荐题目：

单相机空间复用体测温的重构误差与 BOST 少相机化接口分析

适用场景：

如果师兄提到单相机、内窥、limited-view 或硬件编码，这条可作为 BOST/NeRIF 的“少硬件成本”旁支。它和 2020 单相机内窥 BOST、2019 kHz single-camera volumetric flame imaging、2022 single-camera volumetric thermometry 相关。

最小可毕业版本：

- 不做真实硬件，只做有限视角/空间复用 forward model toy。
- 比较视角复用数量、噪声、视场裁剪对体场重构误差的影响。
- 输出“少相机条件下哪些指标最容易崩”的误差图谱。

放弃条件：

- 如果需要真实内窥镜/空间复用光路或标定板参数才能开始，不作为主线。
- 如果没有真实数据，最多作为 A/E 的一个实验轴。

### 旁支 I：数字全息 / 铁颗粒诊断工具

推荐题目：

面向铁颗粒燃烧图像诊断的三维颗粒检测、可视化与误差报告工具

适用场景：

只有在课题组给出颗粒/全息/铁燃烧图像数据时考虑。它贴 OERF 颗粒燃烧和低碳燃料方向，也和何远哲参与的铁颗粒多物理场测量论文相邻，但不贴 BOST/NeRIF 主线。

最小可毕业版本：

- 读取一小段颗粒图像或合成 holography-like particle data。
- 做粒子检测、轨迹连接、浓度/粒径统计和可视化。
- 输出标定字段清单：像素尺寸、放大倍率、曝光、阈值、误检/漏检、浓度单位。

放弃条件：

- 没有组内数据就不要做。
- 不要试图同时研究铁颗粒燃烧机理、全息重构、温度测量和 BOST；本科题只能截一个工程子问题。

优先向师兄提出 A 作为稳健主线，同时把 T16 作为“模型设计 + 三维重建”的重点替代方案；B/C/F 作为二阶段分支，D 作为所有方向都需要的工程底座。具体话术：

> 我建议先以 NeRIF/BOST 鲁棒性分析为主线，保证合成数据、baseline、神经场和参数扫描能毕业；如果组里当前更需要 PIV-BOST、位移质量控制或 4D BOST，我可以把第二阶段分别收束成误差传播、图像位移/光流 benchmark 或低秩时序子问题，同时保留统一数据接口和自动报告工具。

若师兄明确希望做算子学习，可改为：

> 我想把算子学习和少视角 BOST 三维重建合在一起：先用相机几何做 adjoint/SIRT physics lift，再用 residual FNO 学跨样本修正，并用 BOST held-out reprojection 验收；DeepONet 作为第二模型。第一阶段只做 2D 和 24^3/32^3，和 SIRT、U-Net 公平比较，静态模型可靠后再考虑 4D evolution operator。

更新后的更具体说法：

> A 是主线，E 和 F 是最适合马上升级出“研究味”和“组内工具价值”的副线。也就是说，我先做 NeRIF/BOST 的合成闭环和鲁棒性，再把实验轴扩展到时间不同步、缺失视角、投影补全、位移场质量和坏视角筛查；如果师兄给真实数据，就把 E/F 做成数据质量控制、同步误差和图像位移报告工具。
