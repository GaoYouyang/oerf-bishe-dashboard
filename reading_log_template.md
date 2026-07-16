# BOST / NeRIF 阅读追踪表

用途：避免“读了很多但没有转化”。每读一篇，只记录它对毕业设计能贡献什么：公式、数据、baseline、图、代码思路、问题。

## 每周节奏

- 周一：选 2 篇核心论文，写清楚为什么读。
- 周二到周四：每天完成一篇的公式/方法/图表摘取。
- 周五：把本周读到的内容转成一个小实验或一页图。
- 周末：更新问题清单，准备问何远哲。

## 阅读总表

| 状态 | 优先级 | 论文 | 为什么读 | 必须摘出的东西 | 转成我的工作 | 问题 |
| --- | --- | --- | --- | --- | --- | --- |
| 待读 | P0 | Neural refractive index field, Physics of Fluids 2025 | 毕设主线方法 | BOST forward model、loss、坐标编码、评价指标、实验设置 | 简化 NeRIF 复现 | 真实数据和代码是否可给 |
| 待读 | P0 | PIV-BOST, Experiments in Fluids 2025 | 第二阶段升级 | PIV 误差来源、折射补偿流程、同步数据格式 | 2D PIV 偏折补偿 toy | 有无同步 PIV-BOST 样例 |
| 待读 | P1 | Stereo-velocity refractive index compensation, POCI 2026 | PIV-BOST 冲刺/展望 | stereo-velocity、三维速度补偿、数据格式 | M3A 后续接口预留 | 本科阶段是否只做 2D PIV |
| 待读 | P0 | Tensor decomposition 4D BOST, ACM TOG 2026 | 挑战路线 | 4D 表示、张量分解、时间一致性指标、速度 | 低秩时序 phantom | 哪一块适合本科拆出来 |
| 待读 | P0 | Fast and robust volumetric RI measurement by UBOST, 2020 | 传统/统一 baseline | BOS 几何、正则项、速度和鲁棒性 | 体素/正则 baseline | baseline 应以哪篇为准 |
| 待读 | P1 | BOS techniques, Experiments in Fluids 2015 | 基础综述 | BOS 原理、位移估计、误差来源 | 写绪论和变量表 | 哪些误差在 OERF 系统最主要 |
| 待读 | P1 | Instantaneous 3D flame measurements by BOST, 2018 | BOST 前史 | flame RI/density/temperature 链路 | 解释 NeRIF 为什么改进 voxel | OERF 早期 BOST 与 NeRIF 数据差异 |
| 待读 | P1 | Volumetric emission tomography for combustion processes, PECS 2023 | 三维燃烧成像综述 | tomography 分类、评价指标、实验约束 | 写课题组图谱 | 与 BOST 的共性/差异 |
| 待读 | P1 | Online prediction of 3D flame evolution, JFM 2019 | 蔡组 AI for flame 前史 | projection history、网络输入输出、实时性 | 写“AI 不是泛 AI”背景 | 是否可借鉴时序预测指标 |
| 待读 | P1 | Computational flow visualization, Cell Reports Physical Science 2024 | 开题定位 | optical + computational methods reveal hidden flow properties | 写研究意义 | 我的 hidden property 是什么 |
| 待读 | P2 | NILAT / NeDF / Neural Refractive Index Primitives | 外部竞品 | 神经隐式层析表示、sparse-view、可微投影 | 做方法邻居对比 | 哪个最适合作对照 |

## 单篇精读卡模板

论文：

正式引用：

可读入口：

我读这篇是为了：

1. 这篇解决的物理/测量问题是什么？
2. 输入数据是什么？单位、维度、视角数、时间分辨率是什么？
3. 输出物理量是什么？折射率、密度、温度、速度还是辐射强度？
4. forward model 写成什么形式？
5. inverse problem 或 loss function 写成什么形式？
6. baseline 是谁？为什么这个 baseline 公平？
7. 评价指标有哪些？哪些适合我复现？
8. 图 1 到图 4 分别证明什么？
9. 哪些参数最敏感？
10. 作者没有解决什么？
11. 我能在一周内复现哪一个最小模块？
12. 我要问何远哲什么？

## 读完一篇后的输出

- 一张公式图：把观测、未知量、forward model 和 loss 放在一页。
- 一张实验表：列出视角数、噪声、分辨率、训练步数、指标。
- 一段中文总结：这篇给我提供公式 / 数据 / baseline / 图 / 背景中的哪一种。
- 一个最小动作：写一个函数、画一个图、复现一个指标、整理一个数据接口。
- 如果不知道读完后该做什么，先查 `paper_to_demo_map.md`，把论文映射到 M0-M3B 或 A/B/C 定题路线。

## 不值得继续深读的信号

- 只讲燃烧机理，和光学重构/数据处理没有关系。
- 只讲硬件制备，拿不到响应矩阵或原始数据。
- 只讲大模型/AI 概念，没有可复现的输入输出。
- 需要完整实验台搭建才能产生任何结果。
