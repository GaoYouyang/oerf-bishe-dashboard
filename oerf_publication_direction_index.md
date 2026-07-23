# OERF 公开论文-方向-毕设切入索引

用途：把 OERF 官网/GitHub Pages 源码中公开的研究方向和代表性论文，整理成 Gao Youyang 做毕业设计时可使用的方向地图。它不是正式参考文献表；正式引用仍以出版社页面、DOI/Crossref 和 `references.bib` 为准。

最后核验：2026-07-09。

主要来源：

- OERF `src/data/research.ts`：公开四大研究方向。
- OERF `src/data/publications.ts`：公开代表性论文列表，源码注释说明只收录部分期刊和成员范围，不等于全部论文。
- OERF `src/data/members.ts`：蔡伟伟、郑聿韬、宁刘、徐世杰等成员公开研究兴趣。
- `source_audit.md`：本工作台记录的 DOI/题名冲突与修正。

---

## 1. 为什么还要单独做这个索引

OERF 的公开论文列表很有用，但有三个边界：

1. `publications.ts` 只收录代表作，且有 journal/member filter；何远哲的 NeRIF、PIV-BOST、4D BOST 并没有全部出现在这个源码文件里。
2. 部分 DOI 是早期占位、打字错误或跨条目错配；开题引用不能直接照抄。
3. 对本科毕设来说，最重要的不是“论文多”，而是知道每篇论文能给你公式、baseline、数据接口、图表、背景还是展望。

2026-07-09 二次扫描结论：OERF `publications.ts` 的 20 条代表作中，15 条已在本地库按正式 DOI/题名覆盖或纠正；5 条仍存在源码 DOI/题名冲突，主要集中在 Nature Chemical Engineering / Nature Communications 占位 DOI，以及 Combustion and Flame 的三维火焰、LES/flash-boiling 与铁颗粒条目错配。正式引用前必须看 `oerf_source_gap_scan.md` 和 `source_audit.md`。

所以本索引把公开论文按毕设价值重新分层：**主线必读、方法背景、组内旁支、了解即可、暂不碰**。

---

## 2. OERF 四大公开方向到毕设的翻译

| OERF 公开方向 | 官方关键词 | 对你的翻译 | 毕设优先级 |
| --- | --- | --- | --- |
| Extreme Reaction Flow Diagnostics | PLIF, tomographic absorption spectroscopy, BOS, holographic imaging | 用光学方法测反应流的温度、组分、折射率、密度、速度和动态结构 | 最高，尤其是 BOST/NeRIF |
| Computational Imaging & Miniature Spectrometers | light-field, tomography, holography, miniature spectrometers, AI reconstruction | 硬件编码 + 计算反演，把有限观测变成高维物理信息 | 高，但光谱仪支线暂不主攻 |
| Ultrafast Laser Spectroscopy & Plasma Energy Conversion | femtosecond spectroscopy, plasma, ammonia, extreme-temperature synthesis | 极端环境能量转换和超快/等离子体诊断 | 背景了解，除非导师转向 |
| Reacting Flow Simulation & Data Assimilation | LES, ML combustion modeling, data assimilation, ammonia/hydrogen combustion | 把实验测量场接到数值模拟和预测 | 展望/接口，不建议第一主线 |

最适合你的定位：

> 在 Extreme Reaction Flow Diagnostics 和 Computational Imaging 的交叉处，做 BOST/NeRIF 的可复现重构、误差分析和真实数据迁移。

---

## 3. 何远哲主线为什么比官网 publications.ts 更重要

OERF `publications.ts` 的筛选范围偏向 Nature/Science/eLight/PECS/Combustion and Flame/JFM 等代表性期刊。因此何远哲最贴你毕设的几篇文章并不一定出现在这个文件中：

| 论文 | 期刊/来源 | 对你价值 | 处理方式 |
| --- | --- | --- | --- |
| Neural refractive index field | Physics of Fluids 2025 | 主线核心，定义 NeRIF 问题和神经场表示 | 必读，放入正式 BibTeX |
| simultaneous PIV-BOST | Experiments in Fluids 2025 | 第二阶段升级，折射率场补偿速度测量误差 | 必读，作为 PIV-BOST 路线 |
| Tensor decomposition-based 4D BOST | ACM TOG 2026 | 挑战线，X-Y-Z-T 时空重构与张量/轻量网络 | 读思想，拆低秩 toy |
| stereo-velocity PIV-BOST compensation | Proceedings of the Combustion Institute 2026 | PIV-BOST 的进一步升级 | 只作展望，不直接承诺 |

结论：官网论文列表用于理解 OERF 全图谱；毕设定题必须以何远哲论文线为主轴。

---

## 4. OERF 代表作按方向分层

### A. 光学层析与流动可视化：主线背景

| 年份 | 论文 | OERF 源码期刊 | 毕设作用 | 引用状态 |
| --- | --- | --- | --- | --- |
| 2013 | Numerical and experimental validation of a three-dimensional combustion diagnostic based on tomographic chemiluminescence | Optics Express | 早期 CTC 数值/实验验证，补 view orientation、resolution、validation | DOI/PDF 已核验 |
| 2013 | Practical aspects of implementing three-dimensional tomography inversion for volumetric flame imaging | Applied Optics | 真实 tomography implementation 细节，补标定、投影数、重构参数 | DOI/PDF 已核验 |
| 2014 | Nonlinear Tomography: A New Imaging Theory for Combustion Diagnostics | HORIBA Readout | 非线性层析概念入口，解释温度/压力/组分耦合导致的多参数反问题 | 公开 PDF 已缓存；无 DOI |
| 2014 | Nonlinear Tomography with Multiplexed Wavelength Modulation Spectroscopy in Harsh Combustion Environments | OSA Technical Digest | WMS + nonlinear tomography，TAS/NTAS 早期概念根 | Optica 官方入口已核；PDF 自动下载返回 HTML，不缓存 |
| 2014 | Development of a Pressure Imaging Technique with Nonlinear Tomography for Flow Diagnostics | OSA Technical Digest | pressure imaging 与多参数流动诊断前史 | Optica 官方入口已核；PDF 自动下载返回 HTML，不缓存 |
| 2017 | Tomographic absorption spectroscopy for the study of gas dynamics and reactive flows | PECS | 学 optical tomography 和反问题背景 | DOI 已核验 |
| 2019 | Online in situ prediction of 3-D flame evolution from history 2-D projections via deep learning | JFM | 蔡组 AI + flame projection 的前史 | OERF DOI 曾误写，正式用 `10.1017/jfm.2019.545` |
| 2020 | 3D tomography integrating view registration and its application in highly turbulent flames | Combustion and Flame | 视角注册、湍流火焰层析背景 | OERF DOI 曾误写，正式用 `10.1016/j.combustflame.2020.08.025` |
| 2018 | Hybrid diagnostic for optimizing domain size and resolution of 3D measurements | Optics Letters | SVLIF 高速体成像；说明有限光学通道下 domain size、spatial resolution 与 kHz 体测量的折中 | DOI `10.1364/OL.43.003842` 已核验；Optica 自动 PDF 访问返回验证页，暂不缓存 PDF |
| 2023 | Volumetric emission tomography for combustion processes | PECS | emission tomography 综述，写绪论很有用 | OERF DOI 曾误写，正式用 `10.1016/j.pecs.2022.101024` |
| 2023 | Multi-kHz three-dimensional flame imaging | Combustion and Flame | 高速三维火焰成像背景线索 | OERF 源码 DOI `10.1016/j.combustflame.2023.112731` 实际指向 2,3-dimethyl-2-butene 氧化动力学论文；不要按此条正式引用。当前入库采用 Yutao Zheng 2023 3D flame surface measurements, DOI `10.1016/j.combustflame.2023.113103` |
| 2025 | Three-dimensional reconstruction of turbulent flame dynamics using high spatiotemporal resolution laser diagnostics | Combustion and Flame | OERF 源码题名提示三维火焰动态重构方向，但 DOI 冲突 | DOI `10.1016/j.combustflame.2025.114309` 对应铁颗粒燃烧 heat exchange 多物理测量；当前只作为颗粒/多物理旁支，不写成三维火焰重构 |

对你的使用方式：

- 第 1 章绪论：用 TAS/VET/JFM 说明蔡组长期做 optical tomography 和 AI flow visualization。
- 第 2 章基础：用 BOST/NeRIF/Raffel/Grauer/UBOST 补专门的 BOS/BOST 理论。
- 第 4 章实验：不要硬复现这些火焰实验，只借鉴它们的图表指标和物理解释方式。

### B. BOST/NeRIF/PIV-BOST：毕业设计主线

这部分不完全来自 OERF `publications.ts`，但与何远哲同步最相关。

| 方向 | 核心论文 | 你该提取什么 | 最小产物 |
| --- | --- | --- | --- |
| BOST 基础 | Raffel 2015; Grauer 2018; Liu/Cai 2020; UBOST 2020 | BOS 位移、折射率梯度、层析反演、baseline | 传统 baseline 和物理链条图 |
| NeRIF | He et al. 2025 | 坐标网络、forward model、loss、少视角优势 | M0-M2 + PyTorch 版简化 NeRIF |
| PIV-BOST | Zheng/He et al. 2025 | 折射率场如何补偿 PIV 速度误差 | M3A + 误差传播图 |
| 4D BOST | He et al. 2026 | tensor decomposition、时序低秩、轻量网络 | M3B + rank/temporal consistency 扫描 |

对你的使用方式：

- 这是开题主线，不是旁支。
- 读论文时要直接提取公式、数据结构、指标和可复现实验。
- 如果师兄给数据，优先把真实数据接入这条线。

### C. Light-field / single-camera tomography：主线背景桥梁

这部分来自 OERF computational imaging 方向。它不应替代 BOST/NeRIF，但能解释蔡组为什么长期关心少相机、有限光学通道、硬件编码、标定和计算反演。

| 年份 | 论文 | 对你价值 | 使用边界 |
| --- | --- | --- | --- |
| 2019 | Assessment of plenoptic imaging for reconstruction of 3D discrete and continuous luminous fields | 用单相机记录空间和角度信息，评估 3D luminous field 重构；对应 OERF light-field imaging 关键词 | 只作为“少相机硬件编码 + 计算重构”背景，不转成光场相机毕设 |
| 2019 | kHz-rate volumetric flame imaging using a single camera | 单相机高速体火焰成像，说明硬件压缩如何转成反问题 | 用于理解 kHz / volumetric imaging，不替代 BOST 折射率场 |
| 2020 | Parametric study on single-camera endoscopic tomography | 官方摘要指出优化 fiber bundle 输入端和镜头焦距，九投影注册到单相机表现最好 | 可借鉴为 view-count / view-mapping / calibration 小实验 |
| 2020 | Improved calibration model for single-camera endoscopic tomographic systems | 单相机内窥系统标定模型，说明真实几何误差的重要性 | 可接到 M3C geometry sanity check |
| 2020 | Three-dimensional concentration field imaging in a swirling flame via endoscopic volumetric laser-induced fluorescence at 10-kHz-rate | 10 kHz 体 LIF，展示 OERF 高速三维诊断能力 | 物理量是浓度场，不要和 BOST 折射率场混写 |
| 2020 | Reconstruction of kHz-rate 3-D flame image sequences from a low-rate 2-D recording via a data-driven approach | 低帧率 2D 到 kHz 3D 序列重建，和 4D BOST 的时序先验相邻 | 适合作 related work，不建议开局复现 |

对你的使用方式：

- 在绪论里说明 OERF 的 computational imaging 不只靠算法，也靠光路编码和系统设计。
- 在 M3C 中把 view count、view mapping、calibration、spatial resolution 写成真实实验问题。
- 和何远哲沟通时问清九视角/内窥/多输入系统的数据接口，而不是只问能不能给神经网络代码。

详细说明见 `light_field_single_camera_bridge.md`。

### D. 计算光谱与微型光谱仪：强旁支

| 年份 | 论文 | 期刊 | 对你价值 | 引用状态 |
| --- | --- | --- | --- | --- |
| 2021 | Miniaturization of optical spectrometers | Science | 理解“硬件编码 + 计算重构”的思想 | DOI 已核验 |
| 2021 | Computed tomography imaging spectrometry based on superiorization and guided image filtering | Optics Letters | CTIS 少投影光谱数据立方重构，和 BOST 少视角问题同构 | DOI 已核验 |
| 2022 | Miniaturized spectrometers with a tunable van der Waals junction | Science | 计算光谱仪代表作 | OERF DOI 曾误写，正式用 `10.1126/science.add8544` |
| 2023 | Super-resolution computed tomography imaging spectrometry | Photonics Research | CTIS + RGB guidance / zero-order diffraction 的超分辨重构 | DOI/PDF 已核验 |
| 2024 | Simple yet powerful | Nature Photonics | 综述/观点类背景 | OERF 旧题名/DOI 已修正，正式用 `10.1038/s41566-024-01470-7` |
| 2025 | Reconstructive spectrometers: hardware miniaturization and computational reconstruction | eLight | 计算光谱综述，很适合理解课题组另一大支线 | DOI 已核验 |
| 2025 | Miniaturized spectral sensing with a tunable optoelectronic interface | Science Advances | 谱传感硬件+算法 | DOI 已核验 |
| 2026 | In situ spectral reconstruction based on a memristor chip for energy-efficient computational spectrometry | Nature Electronics | memristor computational spectrometry | OERF 源码 DOI 已修正，正式用 `10.1038/s41928-026-01571-x` |

对你的使用方式：

- 不建议主动转主线。
- 可以在“课题组研究图谱”一页中说明 OERF 的计算成像不止 BOST。
- 如果未来转光谱仪，选题会变成 response matrix、spectral reconstruction、calibration、noise robustness。

### E. 等离子体、超快光谱与极端材料：了解背景

| 年份 | 论文 | 期刊 | 对你价值 | 引用状态 |
| --- | --- | --- | --- | --- |
| 2023 | A stable atmospheric-pressure plasma for extreme-temperature synthesis | Nature | OERF/宁刘等离子体极端合成代表方向 | OERF DOI 曾误写，正式用 `10.1038/s41586-023-06694-1`；OSTI accepted manuscript 可在线读 |
| 2024 | Enhancements of electric field and afterglow of non-equilibrium plasma by Pb(ZrxTi1-x)O3 ferroelectric electrode | Nature Communications | 等离子体增强机制 | 正式 DOI `10.1038/s41467-024-47230-7`；开放 PDF 已缓存 |
| 2025 | Enhanced production of active species and NH3 using non-equilibrium ferroelectric barrier discharge | Nature Communications | 氨合成等离子体方向 | OERF 旧题名/DOI 已修正，正式 DOI `10.1038/s41467-025-66403-6`；开放 PDF 已缓存 |
| 2026 | Electrified vapour deposition at ultrahigh temperature and atmospheric pressure for nanomaterials synthesis | Nature Synthesis | 极端温度材料合成 | OERF 源码 DOI 已修正为 `10.1038/s44160-025-00914-4`；DOE PAGES accepted manuscript 可在线读 |

对你的使用方式：

- 写“课题组全貌”时可提，不放入主线参考文献。
- 不建议在本科毕设里同时切入等离子体化学和 BOST/NeRIF。
- 可借鉴的是极端环境诊断中的不确定性意识。

### F. 金属颗粒燃烧、多物理诊断与全息/图像跟踪：备选应用

| 年份 | 论文 | 期刊 | 对你价值 | 引用状态 |
| --- | --- | --- | --- | --- |
| 2021 | Quantification of the size, 3D location and velocity of burning iron particles using high-speed digital in-line holography | Combustion and Flame | 数字全息同时测粒径、三维位置和速度，是全息/颗粒测量备选线的早期核心 | DOI `10.1016/j.combustflame.2021.111430` 已核验 |
| 2021 | Clustering-based particle detection method for digital holography | Measurement Science and Technology | 最像本科可做算法旁支：粒子检测、聚类、三维定位和尺寸估计 | DOI `10.1088/1361-6501/abd7aa` 已核验；IOP PDF 自动访问触发验证页 |
| 2023 | Tomographic reconstruction of an azimuthally forced flame in an annular chamber | Proceedings of the Combustion Institute | CTC 进入真实环形燃烧室/方位模态，补 OERF 真实燃烧器几何背景 | DOI `10.1016/j.proci.2022.08.051` 已核验；CC BY OA 页面可读 |
| 2024 | Phase change and combustion of iron particles in premixed CH4/O2/N2 flames | Combustion and Flame | 铁颗粒相变/燃烧建模旁支；可补金属燃料方向的实验-模型接口 | DOI `10.1016/j.combustflame.2023.113171` 已按 Crossref/ScienceDirect 纠正题名 |
| 2025 | Micro-explosion of burning iron particles with carbon impurity | Combustion and Flame | 金属颗粒燃烧应用背景 | OERF DOI 已按题名校正为 `10.1016/j.combustflame.2025.113974` |
| 2025 | Investigation on heat exchange of iron particle combustion based on simultaneous multi-physics field measurements | Combustion and Flame | 何远哲参与；多物理测量旁支 | DOI `10.1016/j.combustflame.2025.114309` 已核验 |
| 2026 | Insights into the micro-explosion of burning iron particle under ammonia co-firing conditions | Proceedings of the Combustion Institute | 蔡老师主页 POCI accepted 颗粒/氨共燃新线索 | DOI `10.1016/j.proci.2026.106209` 已核验 |
| 2026 | Combustion characteristics of single iron particles under ammonia co-firing conditions | Combustion and Flame | 氨共燃/铁颗粒方向 | OERF 源码 DOI 年份已修正，正式用 `10.1016/j.combustflame.2026.114780` |
| 2026 | Robust 3D tracking of dynamic reacting particles based on holographic spatio-temporal similarity | Combustion and Flame | 若转图像跟踪/全息方向可读 | 正式 DOI `10.1016/j.combustflame.2026.114899` |

对你的使用方式：

- 只有何远哲/郑聿韬明确给你颗粒或全息数据时才转向。
- 如果转向，算法题可做 3D tracking、particle segmentation、trajectory smoothing、multi-physics field registration。
- 不建议和 NeRIF 主线混成一个大题。

### G. 反应流仿真、LES 与数据同化：论文展望

| 年份 | 论文 | 期刊 | 对你价值 | 引用状态 |
| --- | --- | --- | --- | --- |
| 2025 | Online data assimilation of flamelet model based on ensemble Kalman filter for syngas combustion with XH2/CO = 50% | Combustion and Flame | 数据同化方向线索，可写成远期“BOST/NeRIF 重构场作为观测”的展望 | DOI `10.1016/j.combustflame.2024.113881` 已核验 |
| 2025 | A flamelet-based Eulerian transported PDF method for the modeling and simulation of supersonic combustion | Combustion and Flame | 反应流仿真/超燃建模旁支；可用于说明 OERF 不只做诊断，也做高保真燃烧模型 | DOI `10.1016/j.combustflame.2024.113864` 已核验；注意交大主页曾把该 DOI 误链到 micro-explosion 题名 |
| 2026 | Modeling of flame evolution using low-dimensional manifold learned from imperfectly-informed simulations and experimental data | Physics of Fluids | 低维模型/实验数据融合方向线索，适合放展望，不建议本科主线直接做 | DOI `10.1063/5.0255969` 已核验 |

对你的使用方式：

- 你可以把 BOST/NeRIF 重构结果组织成可被 CFD/data assimilation 使用的数据结构。
- 不建议本科阶段同时做 LES 和 NeRIF。
- 适合毕业论文最后一章写“未来可与数据同化结合”。

---

## 5. 读文献的优先级

### Tier 0：现在就读，直接决定题目

1. He et al. 2025 NeRIF。
2. Zheng/He et al. 2025 PIV-BOST。
3. He et al. 2026 4D BOST。
4. Raffel 2015 BOS techniques。
5. Grauer 2018 BOST 和 Grauer 2020 UBOST。

目标：提取 forward model、baseline、loss、指标和数据接口。

### Tier 1：开题绪论和课题组谱系

1. Cai and Kaminski 2017 TAS。
2. Grauer et al. 2023 VET。
3. Huang/Liu/Cai 2019 JFM deep learning flame evolution。
4. Huang et al. 2024 Computational Flow Visualization。

目标：说明蔡组长期的 optical tomography、computational flow visualization 和 AI flow diagnostics 背景。

### Tier 2：如果师兄给数据再读深

1. Multi-kHz 3D flame imaging：不要照抄 OERF 源码 DOI，应先向师兄确认具体正式论文。
2. iron particle multi-physics measurements。
3. holographic reacting particle tracking。
4. Open-source BOS tomography dataset。

目标：和真实数据、实验设置、可视化/追踪任务对接。

### Tier 3：了解 OERF 全貌，不放入主线

1. Science/Nature 系列 miniaturized spectrometer。
2. eLight reconstructive spectrometers。
3. Nature/Nature Communications/Nature Synthesis plasma and ultrahigh-temperature synthesis。
4. LES/flash-boiling simulation papers。

目标：了解课题组能力边界，避免面谈时只知道 BOST。

---

## 6. 可以从 OERF 全谱系里挖出的额外选题

这些不是首选，但如果导师希望换题，可以作为储备。

| 备选方向 | 可能题目 | 数据需求 | 风险 |
| --- | --- | --- | --- |
| 计算光谱仪 | 微型光谱仪响应矩阵反演与噪声鲁棒性分析 | 光谱响应矩阵、标定数据、真实/合成光谱 | 偏离何远哲，需要硬件数据 |
| 发射/吸收层析 | 神经隐式场在 flame emission/absorption tomography 中的迁移 | 投影图、几何、参考场 | 物理量不同，需重学 spectroscopy |
| 全息颗粒追踪 | 燃烧颗粒三维轨迹重构与时序平滑 | hologram、标注/参考轨迹 | 图像处理工作量大 |
| 数据同化接口 | BOST 重构场到 CFD/data assimilation 的数据接口与不确定度输出 | 重构体、网格、CFD 输入格式 | 数值模拟基础不足时易发散 |
| Agent for Science | BOST/NeRIF 参数扫描、失败案例索引和自动报告智能体 | 现有 demo + 数据 manifest | 新颖但要避免变成空壳工具 |

我的建议：这些都可以写进“拓展方向”，但主线仍以 NeRIF/BOST 为锚。

---

## 7. 下一次问何远哲的问题

1. OERF 公开论文里，哪一条线和您现在最急的项目最贴：NeRIF、PIV-BOST、4D BOST、三维火焰、多物理颗粒，还是工具化报告？
2. 如果我做 NeRIF/BOST 鲁棒性，组里最希望我交付代码、参数扫描、真实数据接口、可视化，还是论文图表？
3. 真实 BOST 数据能否给我一份最小样例？如果不能，是否可以给相机几何和一组匿名 displacement field？
4. OERF 源码论文列表里有些 DOI/题名冲突，我正式写开题时是否以您给的论文 PDF/DOI 为准？
5. 如果第二阶段升级，PIV-BOST 和 4D BOST 哪条更符合您对本科毕设的预期？

---

## 8. 这份索引如何进入网页工作台

- 完整调研看 `index.html`。
- 正式引用看 `references.bib` 和 `source_audit.md`。
- 会前沟通看 `advisor_three_page_brief.md`。
- 定题选择看 `top_topic_proposal_cards.md`。
- 课题组全貌看本文件和 `cai_oerf_research_genealogy.md`。
