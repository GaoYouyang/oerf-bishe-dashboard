# 物理本科到 BOST/NeRIF 的 12 周基础训练营

用途：这份训练营把流体、燃烧、几何光学、反问题、PIV、PyTorch 和神经隐式场压缩成 12 周可执行任务。它不是完整课程替代品，而是为了让 Gao Youyang 能读懂 NeRIF / PIV-BOST / 4D BOST，并把学习转成毕业设计产物。

最后核验：2026-07-09。

---

## 0. 训练营原则

1. 每周必须留下一个可检查产物：图、代码、表格、公式推导或一页 memo。
2. 学概念时只追问一个问题：它如何进入 `T -> rho -> n -> ray deflection -> image displacement -> reconstruction` 链条。
3. 不把“学流体力学”写成开放任务；每周只学能服务 BOST/NeRIF 的最小知识块。
4. 每周至少准备一个要问何远哲的问题，把自学和组内需求连起来。
5. 第 4 周之后不能只读材料，必须持续跑 demo 或改 demo。

---

## 1. 资源核验表

| 模块 | 推荐资源 | 官方入口 | 为什么用它 | 只抽取什么 |
| --- | --- | --- | --- | --- |
| 流体力学 | MIT OCW 2.06 Fluid Dynamics | [课程主页](https://ocw.mit.edu/courses/2-06-fluid-dynamics-spring-2013/)；[Readings](https://ocw.mit.edu/courses/2-06-fluid-dynamics-spring-2013/pages/readings/)；[Download course](https://ocw.mit.edu/courses/2-06-fluid-dynamics-spring-2013/download/) | 官方课程页标注 undergraduate，覆盖 pressure、control volume、mass/momentum conservation、viscous flows、dimensional analysis、boundary layers、lift/drag | 连续介质、守恒方程、无量纲数、边界层/湍流直觉 |
| 燃烧/反应流 | Princeton CEFRC Combustion Summer School lecture notes | [当前 lecture notes](https://cefrc.princeton.edu/combustion-summer-school/lecture-notes)；[2019 laser diagnostics notes](https://cefrc.princeton.edu/combustion-summer-school/archived-programs/2019-session/lecture-notes)；[2018 quantitative laser diagnostics notes](https://cefrc.princeton.edu/combustion-summer-school/archived-programs/2018-session/lecture-notes) | 官方页收录 combustion chemistry、combustion physics、turbulent combustion、laser diagnostics 等 lecture notes | 火焰类型、温度/密度/折射率、为什么要非接触诊断 |
| 反问题/层析 | Kak & Slaney, Principles of Computerized Tomographic Imaging | [Slaney official chapter PDFs](https://www.slaney.org/pct/pct-toc.html)；[Purdue mirror / usage note](https://engineering.purdue.edu/~malcolm/pct/) | 官方电子版页面提供章节 PDF；Purdue 页面说明电子版可个人使用，不能商业再分发 | forward/inverse、Radon、FBP、ART/SART、病态性 |
| 图像层析代码 | scikit-image Radon transform example | [Radon transform example](https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html) | 官方示例解释 projection、sinogram、inverse Radon、FBP/SART，并有可运行代码 | 2D projection toy、FBP/SART baseline |
| PIV | OpenPIV docs + PIVlab 2021 + piv-image-generator/SynthPix | [OpenPIV docs](https://openpiv.readthedocs.io/en/latest/)；论文入口见 `paper_library/index.html` | OpenPIV 官方文档用于 PIV image analysis，PIVlab 论文给 MATLAB/GUI baseline，piv-image-generator/SynthPix 给 synthetic particle image pair | 粒子图像对、互相关窗口、位移到速度、raw image dewarping vs velocity-field correction |
| PyTorch | PyTorch Learn the Basics | [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)；[PyTorch homepage](https://pytorch.org/) | 官方教程入口，覆盖 tensor、dataset、model、autograd、optimization | MLP、autograd、optimizer、loss、训练曲线 |
| BOST 主线论文 | Raffel 2015; Grauer 2018; Liu/Cai 2020; He 2025 NeRIF | [`paper_library/index.html`](./paper_library/index.html) | 直接支撑毕业设计 forward model 和重构方法 | BOS 位移、BOST 反演、NeRIF loss、指标 |
| BOS 实验长读 | Cakir 2024 Lund doctoral thesis | [Lund Research Portal](https://portal.research.lu.se/en/publications/optical-diagnostics-with-background-oriented-schlieren-a-practica/)；[official PDF](https://lucris.lub.lu.se/ws/portalfiles/portal/195450596/e-nailing_ex_Bora.pdf) | 90 页长文把 BOS 实验搭建、background pattern、optical-flow displacement、非反应/反应流案例连起来 | 背景图案、光流位移、Bunsen flame、laminar burning velocity、swirl flame 案例 |
| BOS 旁支长读 | Gojani 2013 EPJ review; Bichal 2015 Auburn dissertation | [`paper_library/index.html`](./paper_library/index.html) | EPJ review 补 ballistic-range / shock BOS 工程背景；Auburn dissertation 补 plenoptic/light-field BOS 单相机三维测量 | 高速流应用、shock visualization、plenoptic views、depth perception |
| 升级论文 | Zheng/He 2025 PIV-BOST; He 2026 4D BOST | [`paper_library/index.html`](./paper_library/index.html)；私有全文状态见 [`xmu_vpn_private_library_protocol.html`](./xmu_vpn_private_library_protocol.html) | 直接对应何远哲同步方向 | 折射补偿误差传播、时序低秩/张量先验 |
| 旁支补课 | TAS fixed-point / single-camera thermometry / digital holography | [`paper_library/index.html`](./paper_library/index.html)；新增 COT 2025 TAS 反问题开放 PDF | 只用于理解 OERF 全方向，不替代 BOST/NeRIF 主线 | TAS 只学反问题语言；单相机只学 limited-view / hardware multiplexing；全息只学颗粒检测和三维诊断接口 |

## 1.1 资料获取状态

| 资源 | 当前处理 | 原因 |
| --- | --- | --- |
| MIT OCW 2.06 | 先放官方课程与整包下载入口，不把整包塞进 GitHub Pages。 | 官方整包可下载，但课程材料较多；学习日志里按周引用更清楚。 |
| Princeton CEFRC | 先放当前与 2018/2019 laser diagnostics 官方入口。 | PDF 很多，按周只读燃烧物理、激光诊断和 turbulent combustion 相关部分。 |
| Kak & Slaney | 放官方章节 PDF 入口，不缓存整书 PDF。 | 官方页面允许个人使用，但不适合把全书章节重新发布到本站。 |
| PyTorch / scikit-image / OpenPIV | 放官方文档入口，并把示例转成本地 demo 任务。 | 文档不断更新，保持在线入口比复制静态页面更稳。 |
| Cakir 2024 BOS thesis | 放 Lund 官方页面和 PDF 直链，不缓存 57 MB PDF。 | 很适合补实验直觉，但文件较大；按章节阅读比整本下载到 Pages 更合适。 |
| Gojani 2013 / Bichal 2015 | EPJ review 已缓存 8 页开放 PDF；Auburn plenoptic dissertation 只放官方入口。 | 一个补传统高速/ballistic BOS，一个补 light-field/plenoptic BOS；都作为旁支，不抢主线时间。 |
| 何远哲主线全文 | 4D BOST 已有 ACM OA / CC BY 4.0 HTML/eReader 和本地结构化精读；PIV-BOST 等订阅全文只进 `private_library/`。 | 区分在线开放全文、可缓存开放 PDF 与学校订阅 PDF；后者只能本机阅读，不上传 GitHub Pages。 |

---

## 2. 12 周路线总览

| 周 | 核心能力 | 学习输入 | 必交产物 |
| --- | --- | --- | --- |
| 1 | 看懂流体变量 | MIT OCW 2.06 syllabus/readings + NeRIF intro | 一页 `T-rho-n-displacement` 变量图 |
| 2 | 火焰为什么影响折射率 | CEFRC combustion physics / laser diagnostics notes | 300 字说明火焰、密度、折射率和 BOS 的关系 |
| 3 | 画出 BOS/BOST 前向链条 | Raffel 2015 + Grauer 2018 introduction | 相机-背景-火焰-位移示意图 |
| 4 | 2D Radon / tomography toy | Kak & Slaney + scikit-image Radon example | 2D phantom、sinogram、FBP/SART 重构图 |
| 5 | 病态反问题和正则化 | Kak & Slaney + UBOST intro | 少角度/含噪重构失败图和一句解释 |
| 6 | PyTorch 坐标 MLP | PyTorch Learn the Basics | MLP 拟合 `sin(x)+cos(y)+Gaussian(z)`，输出切片和 loss |
| 7 | NeRIF 最小数学框架 | He 2025 NeRIF method | forward model/loss 草图 + 伪代码 |
| 8 | 合成 BOST 数据闭环 | M0/M1 demo + NeRIF paper | 3/5/7/9 视角 synthetic displacement 和重构对比 |
| 9 | 鲁棒性扫描 | M2 demo + `paper_to_demo_map.md` | view/noise/capacity 曲线和结果解释 |
| 10 | PIV-BOST 入门 | OpenPIV/PIVlab + PIV-BOST + PIV image-pair 工具链 | 合成粒子图像 pair、互相关速度场、折射畸变/补偿图 |
| 11 | 4D BOST 入门 | He 2026 4D BOST + M3B demo | rank/temporal smoothness 曲线 |
| 12 | 开题整合 | 本工作台所有材料 | 6-8 页开题 memo + 10 页 PPT 骨架 + 数据请求清单 |

## 2.1 旁支补课的边界

这三类材料只在师兄或导师明确要求时深入。平时只读到“能说明课题组全景”和“能问出好问题”的程度。

| 旁支 | 读到什么程度 | 不要陷进去的部分 | 什么时候转成题目 |
| --- | --- | --- | --- |
| TAS / NTAS / 光谱层析 | forward model、非线性方程、迭代重构、正则化、superiorization | 详细谱线数据库、真实吸收光谱仪硬件、燃烧组分化学 | 师兄明确给 TAS 数据或要求你做反问题算法 toy |
| 单相机/空间复用/内窥体成像 | limited-view、space division multiplexing、single-camera calibration、视场裁剪误差 | 自己搭光路、复杂内窥镜标定、完整实验复现 | 真实 BOST 数据受相机数量/视角限制，需要少相机误差图谱 |
| BOS 位移/光流和质量控制 | OpenPIV/Farneback/DIS/RAFT-PIV 的基本差异、位移误差、confidence、bad-view/bad-region mask、运行时间 | 纯 CV 排行榜、只追求深度光流 SOTA、不接 BOST/NeRIF/PIV-BOST forward model | 师兄说当前痛点是原始图像前处理、位移场质量、坏视角筛查或 PIV-BOST raw image/correlation 层接口 |
| 外部 BOS 应用 | 氢喷流、微火箭、冷却通道、燃烧速度、shock detection 等应用如何定义观测量和误差 | 把应用场景当主线、深入具体发动机/喷注/冷却结构设计 | 只有当它提供可复用数据、位移前处理、系统误差模型或质量控制指标时才转成题目 |
| 数字全息/铁颗粒诊断 | 粒子检测、三维位置/浓度/粒径、误差报告、可视化 | 铁燃烧机理、完整 hologram reconstruction、温度辐射标定 | 课题组给真实颗粒/全息数据，并明确只做工具子问题 |

---

## 3. 每周详细任务

### Week 1：流体变量只学够用

目标：能说清 `rho`, `T`, `n`, `u`, `p` 是什么，不急着求解 Navier-Stokes。

任务：

- 看 MIT OCW 2.06 的 course description 和 readings 入口。
- 只整理连续介质、控制体、质量守恒、动量守恒和无量纲数。
- 画一张变量链条：`temperature -> density -> refractive index -> deflection -> displacement`。

验收：

- 300 字解释：为什么火焰会让背景图案移动。
- 一张变量图，必须出现单位或量纲。

问何远哲：

- 组里 BOST/NeRIF 数据中，折射率、密度、温度是否都会输出？单位如何定义？

### Week 2：燃烧只学火焰结构和诊断需求

目标：知道 Bunsen flame、premixed flame、diffusion flame 是什么，以及为什么反应流需要非接触光学诊断。

任务：

- 浏览 CEFRC lecture notes 里的 combustion physics 和 laser diagnostics 相关材料。
- 整理 5 个词：equivalence ratio、premixed flame、diffusion flame、OH/CH、PLIF/BOS/PIV。
- 写出 Gladstone-Dale 关系在 BOST 中的位置。

验收：

- 画一张 Bunsen flame 的温度/密度/折射率定性剖面。
- 写 5 条“我不需要深入学的燃烧内容”，例如详细化学机理和湍流燃烧闭合模型。

问何远哲：

- NeRIF/PIV-BOST 真实实验对象是哪类火焰或高速流？Bunsen flame 是否足够作为 toy 场景？

### Week 3：BOS/BOST 几何光学

目标：把 image displacement 看成观测量，把 refractive index field 看成未知量。

任务：

- 读 Raffel 2015 的 BOS 基本原理。
- 读 Grauer 2018 的 BOST introduction 和实验系统图。
- 翻 Cakir 2024 thesis 的目录和 BOS setup / background pattern / optical-flow processing 相关小节，只读能解释实验链路的部分。
- 快速扫 Gojani 2013 review 的 ballistic-range / shock application 例子；若对单相机或光场路线感兴趣，再看 Bichal 2015 的 plenoptic camera 思路。
- 画相机、背景板、流场、光线偏折、位移估计的示意图。

验收：

- 用一句话区分 BOS 和 BOST。
- 写出“为什么单视角通常不足以三维重构”。
- 画出观测量和未知量：`d(x,y,view)` vs `n(x,y,z)`。

问何远哲：

- 组内九视角 BOST 的几何能否先近似为平行投影，还是必须保留相机模型？

### Week 4：2D tomography toy

目标：用最简单 CT toy 理解 forward projection 和 inverse reconstruction。

任务：

- 读 Kak & Slaney 的目录和 Radon/FBP 相关章节。
- 跑 scikit-image Radon transform 示例。
- 改成 3/5/9/30 个投影视角，比较重构差异。

验收：

- 输出 phantom、sinogram、FBP reconstruction、error map。
- 写一句结论：视角数为什么会改变重构误差。

问何远哲：

- BOST baseline 更接近 FBP、SART、Tikhonov、Landweber 还是 UBOST？

### Week 5：病态性和正则化

目标：能解释“为什么少视角重构不是普通监督学习问题”。

任务：

- 在 Week 4 代码里加噪声和少角度。
- 对比无正则、平滑正则和迭代重构。
- 读 UBOST introduction，理解 robust/fast 的意义。

验收：

- 一张少视角/噪声对误差的热图。
- 一段解释：为什么必须有传统 baseline。

问何远哲：

- 真实 BOST 主要误差来自噪声、视角少、背景位移提取，还是几何标定？

### Week 6：PyTorch 坐标 MLP

目标：先训练一个坐标网络拟合已知场，不急着接 BOST forward model。

任务：

- 跑 PyTorch Learn the Basics 中 tensor、autograd、optimization 部分。
- 写 MLP：输入 `(x,y,z)`，输出 scalar field。
- 拟合 `sin(x)+cos(y)+Gaussian(z)` 或三维 phantom。

验收：

- loss curve。
- 三张切片：ground truth / MLP / error。
- 用 autograd 求一次输出对输入的梯度。

问何远哲：

- NeRIF 中是否更推荐网络输出 `n`、`grad n`，还是两者都输出？

### Week 7：NeRIF 数学框架

目标：把论文中的方法拆成自己能写的伪代码。

任务：

- 精读 NeRIF method。
- 写出：coordinate sampling、network output、ray integration、displacement loss、regularization。
- 对照 M0/M1 demo，标出差距。

验收：

- 一页公式草图。
- 一段伪代码。
- 一个“完整 NeRIF vs 我的简化版”差异表。

问何远哲：

- 本科阶段是否需要严格复现 NeRIF 的所有 loss，还是先做简化 forward-model loss？

### Week 8：合成 BOST 闭环

目标：把 Week 6 的坐标 MLP 和 Week 4-5 的 tomography toy 接到 M0/M1/M2。

任务：

- 跑 M0/M1。
- 改一个参数：视角数、噪声或 phantom 形状。
- 保存重构图和指标 CSV。

验收：

- 一张 2D 或 3D reconstruction summary。
- 一张 view-count curve。
- 一段诚实解释：neural/coordinate 方法什么时候不一定赢。

问何远哲：

- 真实 OERF 数据的 view count 和 noise level 大概落在哪个区间？

### Week 9：鲁棒性扫描

目标：从“我复现了”升级为“我知道方法边界”。

任务：

- 跑 M2。
- 改 scan grid，例如 noise 或 feature count。
- 整理结果成“什么时候 coordinate regularizer 有用”的判断。

验收：

- `m2_improvement_heatmap.png` 或自己的新热图。
- 一个参数建议表：推荐视角数、噪声范围、feature count。

问何远哲：

- 组里更看重最终重构精度、重投影误差、训练速度，还是真实数据稳定性？

### Week 10：PIV-BOST 升级

目标：理解折射率场误差如何污染速度测量，并判断补偿应该发生在 raw image、correlation/displacement 还是 velocity-vector 层。

任务：

- 读 OpenPIV docs 中 algorithm basics 和 tutorial，同时快速浏览 PIVlab 2021 的 enhanced algorithms。
- 快速读 realtime GPU BOS、wavefront aberration BOS、atmospheric-turbulence shock detection 三类新文献，只摘取位移估计、光流/前处理速度、坏区域和系统误差语言。
- 跑 M3A，保留现有 velocity-vector toy 作为基线。
- 用 piv-image-generator 或 SynthPix 思路生成一对最小 synthetic particle image；如果暂时不装工具，就先用自写 Gaussian particle image toy。
- 用 OpenPIV 或 PIVlab 处理图像 pair，记录 window size、overlap、validation、outlier replacement 和 bad-vector mask。
- 用 OpenCV 跑 Farneback / DIS 或 TV-L1 风格 optical flow，和 OpenPIV 互相关输出做一个小表。
- 加入一个简化 refractive distortion / image shift / wavefront-like blur，比较 raw image dewarping、displacement correction 和 velocity-field correction 的差别。

验收：

- 补偿前后速度误差图。
- 一张三层接口图：raw particle image -> correlation/displacement -> velocity field。
- 一张 quality-control 报告：位移误差、坏区域 mask、置信度、运行时间、是否建议进入 BOST/NeRIF 重构。
- 一段说明：补偿发生在粒子图像层、互相关层还是速度矢量层，各有什么差异。

问何远哲：

- 如果做 PIV-BOST，您更希望我做原始粒子图像校正、互相关位移场校正，还是速度场误差传播？
- 组内是否能给一对 PIV raw image、对应 BOST 位移/折射场，以及 PIVlab/OpenPIV/DaVis 导出的速度场？
- 组内 BOST/PIV-BOST 当前有没有 per-view/per-frame 的位移质量分数、bad-view 记录或手动筛帧规则？

### Week 11：4D BOST 升级

目标：只拆 4D BOST 的一个本科可控子问题：低秩时序先验和时间一致性。

任务：

- 读 4D BOST abstract、introduction、method overview。
- 跑 M3B。
- 改 rank 或 noise，观察 temporal smoothness 和 L2 的 trade-off。

验收：

- rank scan。
- temporal trace。
- 一段说明：低秩先验能减少抖动，但不能修正系统几何误差。

问何远哲：

- 4D BOST 最适合本科生帮忙的是 rank 扫描、时序指标、可视化工具，还是数据清洗？

### Week 12：开题整合

目标：把学习、demo、论文和问题清单收束成可汇报材料。

任务：

- 更新 `opening_report_draft.md`。
- 更新 `opening_ppt_outline.md`。
- 用 `advisor_three_page_brief.md` 准备 8-10 分钟会前汇报。
- 用 `data_request_checklist.md` 准备要数据。

验收：

- 6-8 页开题 memo。
- 10 页 PPT 骨架。
- 一页题目 A/B/C 选择表。
- 会后 7 天行动清单。

问何远哲：

- 题目是否定为 NeRIF/BOST 鲁棒性主线？PIV-BOST 或 4D BOST 哪个作为第二阶段？

---

## 4. 每周固定检查表

每周结束时填写：

| 项目 | 是否完成 | 证据 |
| --- | --- | --- |
| 读了 1 篇论文或 1 个课程模块 |  | 论文卡/笔记 |
| 生成了 1 张图 |  | 图片路径 |
| 改了或跑了 1 个脚本 |  | 脚本路径/命令 |
| 写了 1 个给何远哲的问题 |  | 问题文本 |
| 更新了工作台 |  | 文件路径 |

如果一周没有图，也没有代码，说明这周只是“看起来在学”，没有推进毕设。

---

## 5. 只需要掌握到什么程度

| 模块 | 够用标准 | 暂时不用学深 |
| --- | --- | --- |
| 流体 | 能解释变量、守恒、Re/Ma 和密度/温度关系 | 湍流闭合、LES、边界层精确解 |
| 燃烧 | 能解释火焰温度/密度/折射率和非接触诊断 | 详细化学机理、反应速率数据库 |
| 光学 | 能解释折射率梯度导致光线偏折 | 完整光机设计、像差理论 |
| 层析 | 能跑 2D Radon/FBP/SART toy | 严格 CT 理论证明 |
| PIV | 能处理合成粒子图像或速度场误差 | stereo-PIV 完整标定 |
| PyTorch | 能写坐标 MLP 和 autograd | 大模型训练、复杂分布式训练 |
| 神经场 | 能理解 coordinate network 和 positional encoding | 复杂 neural rendering 系统 |

---

## 6. 最小日程建议

每天 2-3 小时即可，不要把自己拖成全天低效率。

| 时间块 | 做什么 |
| --- | --- |
| 45 分钟 | 读课程/论文，只摘和本周目标有关的公式或图 |
| 60 分钟 | 跑或改一个最小代码实验 |
| 30 分钟 | 画图、写结论、记录失败 |
| 15 分钟 | 写一个要问师兄的问题 |

周末多加 1-2 小时，把本周材料整理进工作台。

---

## 7. 本训练营对应的本地材料

- `foundation_bridge.md`：概念版基础桥梁。
- `weekly_checkpoint_board.md`：12 周验收板。
- `minimum_demo_protocol.md`：M0-M3B demo 协议。
- `he_paper_deep_dive.md`：何远哲主线论文深挖。
- `paper_to_demo_map.md`：论文到 demo 的对应关系。
- `data_request_checklist.md`：向师兄要数据的清单。
- `advisor_three_page_brief.md`：会前 8-10 分钟沟通材料。

---

## 8. 公开资源入口

- MIT OCW 2.06 Fluid Dynamics: https://ocw.mit.edu/courses/2-06-fluid-dynamics-spring-2013/
- Princeton CEFRC Combustion Summer School lecture notes: https://cefrc.princeton.edu/combustion-summer-school/lecture-notes
- Kak & Slaney CT textbook: https://engineering.purdue.edu/~malcolm/pct/
- scikit-image Radon transform example: https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html
- OpenPIV documentation: https://openpiv.readthedocs.io/
- Cakir 2024 BOS practical thesis: https://portal.research.lu.se/en/publications/optical-diagnostics-with-background-oriented-schlieren-a-practica/
- Gojani 2013 BOS ballistic-range review: https://doi.org/10.1051/epjconf/20134501034
- Bichal 2015 plenoptic BOS dissertation: https://etd.auburn.edu/handle/10415/4827
- PIVlab paper: https://openresearchsoftware.metajnl.com/articles/10.5334/jors.334
- piv-image-generator: https://doi.org/10.1016/j.softx.2020.100537
- SynthPix: https://arxiv.org/abs/2512.09664
- PyTorch Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
