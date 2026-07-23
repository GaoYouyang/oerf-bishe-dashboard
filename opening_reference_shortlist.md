# 开题参考文献短表与引用位置

用途：把 `references.bib` 和文献卡片转成开题报告真正能用的参考文献顺序。它不是最终排版版参考文献；最终提交前仍需按学院模板、出版社页面和 DOI 再统一格式。

生成日期：2026-07-06。

---

## 1. 推荐开题报告只放 12-15 篇

开题报告不宜把所有 OERF 方向都塞进参考文献。建议主线只保留：

- BOS/BOST 基础：3-4 篇。
- OERF/蔡组 tomography 和 computational flow visualization 背景：3-4 篇。
- 何远哲主线：NeRIF、PIV-BOST、4D BOST：3 篇；如果开题想强调 PIV-BOST 升级，可额外放 stereo-velocity PIV-BOST 作为 [11b]。
- 方法邻居/保底公开数据：2-3 篇。

光谱仪、等离子体、金属颗粒、LES/data assimilation 只在“课题组全貌”里提，不进入主线参考文献，除非导师明确要求转向。

---

## 2. 推荐引用顺序

| 建议编号 | 文献 | 用在开题哪一节 | 支撑哪句话 | 状态 |
| --- | --- | --- | --- | --- |
| [1] | Raffel 2015 BOS techniques | 1. 背景；2.1 BOS/BOST | BOS 通过背景图像位移表征折射率/密度变化，是非接触流场诊断方法 | DOI 已核验，Springer CC BY PDF 已缓存 |
| [2] | Grauer et al. 2018 BOST flame imaging | 2.1 BOS/BOST | BOST 已用于瞬态三维火焰折射率/密度相关成像 | DOI 已核验 |
| [3] | Liu, Shui and Cai 2020 endoscopic BOST | 2.1 BOS/BOST；真实数据接口 | 蔡组已有单相机/内窥 BOST 三维火焰折射率成像前史 | DOI 已核验 |
| [4] | Liu, Huang, Li and Cai 2020/2021 volumetric BOST | 1. 背景；2.1 BOS/BOST | BOST 可连接折射率、密度和温度三维场 | DOI 已核验 |
| [5] | Grauer and Steinberg 2020 UBOST | 2.2 baseline | 传统 BOST 中已有快速鲁棒的统一体折射率重构方法，可作为 baseline 参照 | DOI 已核验 |
| [6] | Cai and Kaminski 2017 TAS | 1. OERF 背景；2. 国内外现状 | OERF/蔡组长期研究光学层析诊断与反应流测量 | DOI 已核验 |
| [7] | Grauer et al. 2023 VET | 2. 国内外现状 | 体发射层析是燃烧三维诊断的重要相关方向 | DOI 已核验，注意 OERF 源码 DOI 曾误写 |
| [8] | Huang, Liu and Cai 2019 JFM | 2. AI for flow visualization 前史 | 蔡组较早探索用深度学习从二维投影预测三维火焰演化 | DOI 已核验，注意 OERF 源码 DOI 曾误写 |
| [9] | Huang et al. 2024 Computational Flow Visualization | 1. 研究意义 | 光学方法和计算方法结合可揭示复杂流动隐藏属性 | DOI 已核验 |
| [10] | He et al. 2025 NeRIF | 2.3 NeRIF；3. 研究目标；4. 技术路线 | 神经隐式折射率场是本课题主线方法来源 | DOI 已核验 |
| [11] | Zheng et al. 2025 PIV-BOST | 2.4 PIV-BOST；升级路线 | 同步 PIV-BOST 可用三维折射率场补偿速度测量折射误差 | DOI 已核验 |
| [11b] | Zheng et al. 2026 stereo-velocity PIV-BOST | 2.4 PIV-BOST 展望；风险边界 | PIV-BOST 已推进到 turbulent combustion 中的 stereo-velocity measurement，但本科阶段应先做 planar toy / image-layer / 误差传播 | DOI 已核验；若篇幅有限可不放 |
| [12] | He et al. 2026 4D BOST | 2.5 4D BOST；升级路线 | 4D BOST 用张量/时序表示处理高速时空重构，是挑战项来源 | DOI 已核验 |
| [13] | Molnar et al. 2026 open-source BOS dataset | 数据不可得时保底路线 | 开源 BOS/TBOS 数据可作为无组内数据时的 pipeline 预演 | DOI 已核验 |
| [14] | Molnar et al. 2025 NILAT | 方法邻居 | 神经隐式场也可用于吸收层析，说明方法具有跨模态反问题价值 | DOI 已核验 |
| [15] | Li et al. 2024 NeDF 或 Lu et al. 2026 NRIP | 相关工作/方法邻居 | sparse-view BOST 的神经场表示有外部相邻路线；NeDF 已有 Physics of Fluids 正式 DOI 和 AIP Figshare 补充材料，NRIP 已有 Combustion and Flame 正式 DOI | 若篇幅有限可不放 |

---

## 3. 可直接放进开题报告的引用句

这些句子可以改写后放入 `opening_report_draft.md`，方括号编号按上表。

1. 背景纹影技术利用背景图案在折射率梯度场中的表观位移来表征密度或折射率变化，是非接触流场诊断的重要方法之一 [1]。
2. 多视角 BOS 可进一步形成 BOST，用于三维火焰折射率场和相关密度/温度结构的重构 [2-4]。
3. 蔡伟伟团队长期围绕反应流光学层析诊断开展研究，包括吸收层析、体发射层析、BOST 以及计算流动可视化等方向 [3,4,6-9]。
4. 传统体素或统一 BOST 方法能够实现三维折射率重构，但在少视角、高分辨率和含噪场景下仍受到病态性、离散化和正则化选择影响 [5]。
5. NeRIF 将折射率场表示为连续神经隐式场，并通过可微前向模型与多视角背景位移建立重投影一致性，是本课题的直接方法基础 [10]。
6. 在 PIV-BOST 中，同步重构的折射率场可用于补偿 PIV 速度测量中的光线偏折误差，为本课题的实验升级提供方向；其后续工作已扩展到湍流燃烧中的 stereo-velocity measurement，但本科阶段仍应先控制在 planar PIV 或误差传播子问题 [11,11b]。
7. 对高速瞬态流场，4D BOST 将三维重构扩展到时空域，并引入张量/时序表示以降低逐帧重构的计算和一致性压力 [12]。
8. 若课题组真实数据暂时不可用，开源 BOS tomography dataset 可用于预演数据接口、标定、mask、deflection 和重构评估流程 [13]。

---

## 4. GB/T 风格草稿

说明：以下是中文开题报告常用格式的草稿版。最终提交前应按学院模板统一大小写、作者数量、页码或 article number。

```text
[1] Raffel M. Background-oriented schlieren (BOS) techniques[J]. Experiments in Fluids, 2015, 56(3). DOI: 10.1007/s00348-015-1927-5.

[2] Grauer S J, Unterberger A, Rittler A, et al. Instantaneous 3D flame imaging by background-oriented schlieren tomography[J]. Combustion and Flame, 2018, 196: 284-299. DOI: 10.1016/j.combustflame.2018.06.022.

[3] Liu H, Shui C, Cai W. Time-resolved three-dimensional imaging of flame refractive index via endoscopic background-oriented Schlieren tomography using one single camera[J]. Aerospace Science and Technology, 2020, 97: 105621. DOI: 10.1016/j.ast.2019.105621.

[4] Liu H, Huang J, Li L, et al. Volumetric imaging of flame refractive index, density, and temperature using background-oriented Schlieren tomography[J]. Science China Technological Sciences, 2021, 64(1): 98-110. DOI: 10.1007/s11431-020-1663-5.

[5] Grauer S J, Steinberg A M. Fast and robust volumetric refractive index measurement by unified background-oriented schlieren tomography[J]. Experiments in Fluids, 2020, 61(3). DOI: 10.1007/s00348-020-2912-1.

[6] Cai W, Kaminski C F. Tomographic absorption spectroscopy for the study of gas dynamics and reactive flows[J]. Progress in Energy and Combustion Science, 2017, 59: 1-31. DOI: 10.1016/j.pecs.2016.11.002.

[7] Grauer S J, Mohri K, Yu T, et al. Volumetric emission tomography for combustion processes[J]. Progress in Energy and Combustion Science, 2023, 94: 101024. DOI: 10.1016/j.pecs.2022.101024.

[8] Huang J, Liu H, Cai W. Online in situ prediction of 3-D flame evolution from its history 2-D projections via deep learning[J]. Journal of Fluid Mechanics, 2019, 875. DOI: 10.1017/jfm.2019.545.

[9] Huang J, Liu H, Zhu S, et al. Computational flow visualization to reveal hidden properties of complex flow with optical and computational methods[J]. Cell Reports Physical Science, 2024, 5(11): 102282. DOI: 10.1016/j.xcrp.2024.102282.

[10] He Y, Zheng Y, Xu S, et al. Neural refractive index field: Unlocking the potential of background-oriented Schlieren tomography in volumetric flow visualization[J]. Physics of Fluids, 2025, 37(1): 017143. DOI: 10.1063/5.0250899.

[11] Zheng Y, He Y, Chen J, et al. Instantaneous refractive index compensation on the velocity measurement using simultaneous PIV-BOST[J]. Experiments in Fluids, 2025, 66(9). DOI: 10.1007/s00348-025-04093-y.

[11b] Zheng Y, He Y, Feng L, et al. Instantaneous refractive index compensation on stereo-velocity measurement in turbulent combustion[J]. Proceedings of the Combustion Institute, 2026, 42: 106175. DOI: 10.1016/j.proci.2026.106175.

[12] He Y, Zheng Y, Xu S, et al. Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction[J]. ACM Transactions on Graphics, 2026, 45(5): 1-19. DOI: 10.1145/3809488.

[13] Molnar J P, Singh A K, Clifford C J, et al. Open-source BOS tomography dataset of high-speed flow over a flight body[J]. Experiments in Fluids, 2026, 67(4). DOI: 10.1007/s00348-026-04189-z.

[14] Molnar J P, Xia J, Zhang R, et al. Unsupervised neural-implicit laser absorption tomography for quantitative imaging of unsteady flames[J]. Combustion and Flame, 2025, 279: 114298. DOI: 10.1016/j.combustflame.2025.114298.

[15] Li J, Meng X, Xiong Y, et al. Neural deflection field for sparse-view tomographic background oriented Schlieren[J]. Physics of Fluids, 2024, 36(12): 127143. DOI: 10.1063/5.0241191.
```

可选替换 [15]：Lu X, Hu W, Liao Z, et al. Neural refractive index primitives for flame field reconstruction using background-oriented Schlieren[J]. Combustion and Flame, 2026, 290: 115082. DOI: 10.1016/j.combustflame.2026.115082.

---

## 5. 不建议放进开题主线参考文献的条目

| 条目 | 原因 | 处理方式 |
| --- | --- | --- |
| 微型计算光谱仪 Science/Nature 系列 | 是蔡老师重要方向，但偏硬件/光谱仪，和何远哲 BOST 主线资源不直接重合 | 只放课题组全貌，不进入主线参考文献 |
| 等离子体/极端材料 Nature/Nature Synthesis 系列 | 强方向但偏宁刘/等离子体化学，不服务 NeRIF/BOST 主线 | 只作 OERF 背景 |
| 金属颗粒燃烧/全息追踪 | 可能与何远哲某些多物理测量相关，但不是首选 BOST 主线 | 除非师兄给颗粒/全息数据，否则不放 |
| LES/flash boiling simulation | 偏 CFD 和数据同化，基础门槛高 | 写未来展望即可 |

---

## 6. 最后一轮提交前检查

- 所有 DOI 用出版社或 Crossref 再核一次。
- OERF 源码中有冲突的 DOI，不直接放进最终参考文献。
- 如果学院要求中文题名或中文期刊格式，按学院模板重排。
- 如果参考文献编号顺序按正文首次出现排序，开题报告正文也要同步改编号。
- arXiv 条目若已有期刊版，优先引用期刊版。
