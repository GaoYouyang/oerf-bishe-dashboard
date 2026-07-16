# OERF 全方向本科毕设研究问题池

生成日期：2026-07-06

用途：把 OERF 公开方向、蔡伟伟主页和代表性论文拆成本科毕业设计可以选择或备选的研究问题。这个文件不是让 Gao Youyang 同时做所有方向，而是为了防止和导师/师兄沟通时只知道 NeRIF 一个词。

结论先行：**主线仍应是 BOST / NeRIF / PIV-BOST / 4D BOST。** 其他方向可以作为背景、备选或第二阶段拓展，除非何远哲或蔡老师明确要求转向。

## 1. 总体方向树

| 大方向 | 官方关键词 | 与何远哲主线关系 | 本科毕设建议 |
| --- | --- | --- | --- |
| 极端反应流光学诊断 | PLIF、吸收光谱层析、BOS、全息、温度/组分/流动 | 最高 | 主线锁定 BOST/NeRIF；PLIF/TAS/全息作背景 |
| 计算成像与微型光谱仪 | 光场、线性/非线性层析、全息、微型光谱仪、AI reconstruction | 高 | BOST 属于计算成像；光谱仪只作备选 |
| 超快激光与等离子体能量转换 | 飞秒光谱、等离子体、氨合成、极端材料合成 | 低到中 | 方向强但偏离 He；只在导师指定时转 |
| 反应流仿真与数据同化 | LES、机器学习燃烧建模、数据同化、氨/氢燃烧 | 中 | 可作为 BOST 输出的上层应用，不建议从 LES 起步 |
| Agent for Science | 科研自动化、参数扫描、报告生成 | 中 | 可做 BOST pipeline 的辅助工具，不单独替代主线 |

## 2. 可研究问题 backlog

| 编号 | 研究问题 | 数据/代码需求 | 可毕业性 | 贴 He 程度 | 风险 | 建议 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | 少视角 BOST 的 NeRIF-style 神经隐式重构 | 合成 BOST、传统 baseline、PyTorch | 高 | 最高 | 中 | 首选主线 |
| B2 | BOST 重投影一致性和物理误差诊断 | 重构体、视角几何、位移场 | 高 | 最高 | 低 | 作为所有方案的共同模块 |
| B3 | BOST 位移提取误差对三维重构的传播 | 背景图/扰动图或合成位移噪声 | 高 | 高 | 中 | 可作为鲁棒性章节 |
| B4 | NeRIF 与 NeDF / NRIP 表示方式对比 | 合成 sparse-view 数据、开源论文设置 | 中 | 高 | 中 | 做小规模竞品对比，不承诺完全复现 |
| B5 | 面向真实九视角 BOST 的数据接口和可视化 | 真实或匿名数据、manifest | 高 | 最高 | 数据依赖中 | 若师兄给数据，优先做 |
| P1 | PIV-BOST 速度误差传播 toy | 合成速度场、折射位移场 | 高 | 高 | 低 | 已有 M3A，可升级 |
| P2 | PIV 粒子图像层折射补偿 | OpenPIV/粒子图像生成器 | 中 | 高 | 中高 | 真实数据前的进阶 |
| P3 | stereo-PIV-BOST 几何误差分析 | 立体标定、同步数据 | 低到中 | 高 | 高 | 只作展望或师兄指定子模块 |
| T1 | 4D BOST 低秩时序先验 | 合成 3D+T phantom、rank scan | 中高 | 高 | 中 | 已有 M3B，可作为挑战章节 |
| T2 | 4D BOST 时序一致性评价指标 | 重构体序列、轨迹指标 | 高 | 高 | 中 | 很适合本科做工具化贡献 |
| T3 | 4D BOST 加速/显存 profiling | 代码、分辨率/帧数扫描 | 中 | 中高 | 中 | 需要组内代码或简化实现 |
| L1 | PLIF / flame front 图像分割 | 火焰图像、U-Net/传统图像处理 | 中 | 中 | 数据依赖中 | 若 He 线转火焰前沿再做 |
| A1 | 吸收光谱层析的反问题入门复现 | 吸收线参数、投影数据 | 中 | 中 | 理论中 | 可作为 tomography 背景，不抢主线 |
| V1 | Volumetric emission tomography 综述与小 demo | 发射投影、SIRT/ART | 中 | 中 | 中 | 可做方法旁支，不建议主线 |
| H1 | 数字全息颗粒 3D 定位与轨迹连接 | 全息图像、检测/跟踪算法 | 中 | 低到中 | 数据依赖高 | 若师兄给颗粒数据再转 |
| M1 | 金属颗粒燃烧多物理测量的数据可视化 | 温度/速度/粒径/辐射同步数据 | 中 | 低到中 | 数据依赖高 | 作为备选，别和 NeRIF 混题 |
| S1 | 微型计算光谱仪谱恢复算法 | 响应矩阵、光谱库、噪声模型 | 中 | 低 | 中 | 算法可做，但偏离 He |
| S2 | 快照光谱相机重构和噪声鲁棒性 | 标定矩阵、快照图像 | 中 | 低 | 中高 | 导师指定时再转 |
| C1 | BOST 重构场到 CFD/data assimilation 的接口 | 重构网格、误差估计、简化模型 | 中 | 中 | 中高 | 后期拓展，不从 CFD 起步 |
| C2 | 简化数据同化 toy：用光学观测校正低维流场 | 合成观测、Kalman/变分同化 | 中 | 中 | 中 | 可作为“数据融合实验流体力学”接口 |
| G1 | BOST/NeRIF 参数扫描和自动报告工作台 | 现有 demo、实验 registry | 高 | 中高 | 低 | 很适合作为工程化加分项 |
| G2 | 论文图自动生成与失败案例索引 | demo 输出、Markdown/HTML | 高 | 中 | 低 | 服务主线，不单独成题 |

## 3. 最值得继续拓展的 6 个方向

### 3.1 BOST 数据接口与 manifest

为什么值得做：博士生方向常常缺的不是单个算法，而是稳定的数据整理、单位、坐标、mask 和实验记录。你已经有 `data_templates/`，下一步可以让它更像真实组内工具。

可交付：

- `bost_sample_manifest.json` 完整字段。
- 数据质量检查脚本。
- 视角图、mask、位移场、重投影误差自动图。
- “可公开/仅内部/不可外传”字段。

### 3.2 NeRIF vs 传统 baseline 的失败边界

为什么值得做：如果只展示 NeRIF 更好，容易显得像宣传；如果能指出哪些条件下传统法更强，反而更像严肃研究。

可交付：

- 视角数、噪声、field capacity、Fourier encoding、采样策略扫描。
- 成功/失败案例对照图。
- “何时值得用 neural field”的决策表。

### 3.3 PIV-BOST 的误差传播

为什么值得做：贴何远哲最新方向，也贴实验真实痛点。即使没有真实 PIV 图像，也能先把折射位移误差到速度误差的链条讲清。

可交付：

- 合成粒子图像或速度场。
- 补偿前后 error map。
- deflection noise sensitivity。
- 2D planar 到 stereo-PIV 的边界说明。

### 3.4 4D BOST 的时序一致性评价

为什么值得做：完整 4D 太重，但评价指标和可视化工具很适合本科生贡献。

可交付：

- framewise vs low-rank 的 rank trade-off。
- temporal smoothness。
- centroid trajectory error。
- 时间切片 GIF 或时序剖面图。

### 3.5 Open BOS/TBOS 数据预演

为什么值得做：如果组内数据暂时不给，公开数据能保住项目进度。

可交付：

- open data manifest。
- loader。
- 子采样视角实验。
- synthetic-to-open-data 差异分析。

### 3.6 Agent for Science 辅助 BOST pipeline

为什么值得做：蔡老师主页公开列出 Agent for Science，但本科阶段不能把它做成空壳。最稳方式是让 agent 服务 BOST 实验矩阵。

可交付：

- 参数配置生成。
- 实验运行记录。
- 自动汇总最佳/失败案例。
- 一键生成每周报告。

## 4. 不建议优先投入的方向

| 方向 | 暂不优先原因 | 保留方式 |
| --- | --- | --- |
| 完整高保真 LES | 流体/燃烧/数值实现要求过高 | 只做 BOST 输出到 CFD 接口 |
| 微型光谱仪硬件 | 偏器件和标定，远离 He 主线 | 做课题组全图谱背景 |
| 等离子体氨合成 | 方向强但知识栈差异大 | 只在导师明确安排时转 |
| 金属颗粒燃烧机理 | 实验和燃烧机理依赖强 | 若给图像数据，只做测量/可视化工具 |
| 完整 stereo-PIV-BOST | 标定和三维速度链条过重 | 先做 planar PIV-BOST toy |

## 5. 选题决策规则

1. **没有真实数据时**：选 B1 + B2 + B3，保底完成 synthetic / open data 研究。
2. **给了 BOST 数据时**：选 B5，所有努力转向真实数据接口和重投影验证。
3. **给了 PIV-BOST 数据时**：选 P1/P2，把 PIV 补偿作为第二阶段主线。
4. **给了 4D 数据或任务时**：选 T1/T2，不承诺完整复现，只做低秩/评价/可视化子模块。
5. **导师强调数据融合实验流体力学时**：把 BOST 输出的数据结构、不确定度和可复现实验记录讲清楚。
6. **导师强调 Agent for Science 时**：把 agent 写成 BOST pipeline 的自动化层，而不是另起炉灶。

## 6. 开题报告中怎么使用这份 backlog

- 第 1 章用它说明 OERF 全方向，不要让评委觉得你只看了一篇 NeRIF。
- 第 2 章只展开 BOST/NeRIF/PIV-BOST/4D BOST 的核心文献。
- 第 3 章把 B1/B2/B3 写成保底方法。
- 第 4 章把 P1/T1/T2 写成进阶实验或展望。
- 结论中说明：旁支方向不是忽略，而是因本科可控性和师兄同步性而收束。

## 7. 来源

- OERF Lab 研究方向源码：`src/data/research.ts`
- OERF Lab 成员源码：`src/data/members.ts`
- OERF Lab 代表性论文源码：`src/data/publications.ts`
- 蔡伟伟交大主页：https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html
- OERF Lab 官网：https://laserdiagnostics.github.io/oerf-lab-website/
- He et al., NeRIF, Physics of Fluids 2025.
- Zheng / He et al., PIV-BOST, Experiments in Fluids 2025.
- He et al., 4D BOST, ACM TOG 2026.
