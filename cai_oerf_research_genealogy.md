# 蔡伟伟 / OERF 研究谱系与毕设切入图

用途：把蔡伟伟老师和 OERF Lab 的公开方向翻译成“本科毕设能怎么贴、哪些要读、哪些只做背景”。这份文件里的“事实依据”和“毕设推断”分开写，避免把公开方向误读成具体任务安排。

最后核验：2026-07-09。

---

## 0. 总判断

OERF 的公开图谱可以压成三层：

1. **物理场景层**：航发、高超飞行器、反应流、燃烧、等离子体、低碳燃料。
2. **光学测量层**：PLIF、BOS/BOST、吸收光谱层析、发射层析、光场/全息成像、快照光谱。
3. **计算重构层**：线性/非线性层析、神经隐式场、张量分解、深度学习预测、数据同化和 Agent for Science。

Gao Youyang 当前最应该卡在第 2-3 层之间：**用光学观测和计算重构得到三维/四维流动物理场**。何远哲的 NeRIF、PIV-BOST、4D BOST 正是这条缝里的高价值方向。

## 0.1 官方项目对齐信号

2026-07-09 重新核验蔡老师上海交通大学中文主页，近期科研项目里有三条对毕设定位很关键：

- **2025-2029：基于数据同化的先进航空发动机非接触式测量技术研究，主持。**
- **2026-2030：国家自然科学基金创新研究群体 A 类项目，数据融合实验流体力学，骨干。**
- **2020-2023：基于单相机内窥背景纹影层析技术的火焰折射率场、密度场和温度场同步测量方法研究，主持，优秀结题。**

这意味着你的题目不应只表述为“复现一个 AI 重构算法”，而应表述为：

> 面向 OERF 非接触式流场测量和数据融合实验流体力学需求，建立 BOST/NeRIF 的可复现重构、误差分析和真实数据接口工具。

这个表述同时贴住：

- 何远哲主线：NeRIF、PIV-BOST、4D BOST。
- 蔡老师项目：非接触测量、数据融合、航空发动机/极端反应流诊断。
- 本科可交付：合成/开放数据 baseline、视角/噪声/几何误差扫描、自动报告、真实数据 manifest。

不建议一上来把题目写成“数据同化”或“航空发动机测量系统”。更稳的写法是：先做 BOST/NeRIF 重构工具，终章再说明重构场、误差图和不确定度图未来可以作为数据同化/CFD 校正的观测输入。

---

## 1. BOST / NeRIF / 折射率场主线

### 代表论文

- He et al., Neural refractive index field, Physics of Fluids, 2025.
- He et al., Tensor decomposition-based four-dimensional BOST, ACM TOG, 2026.
- Zheng et al., simultaneous PIV-BOST, Experiments in Fluids, 2025.
- Liu, Shui and Cai, endoscopic BOST using one single camera, Aerospace Science and Technology, 2020.
- Liu, Huang, Li and Cai, volumetric imaging of flame refractive index, density and temperature using BOST, Science China Technological Sciences, 2020.
- Grauer et al., instantaneous 3D flame imaging by BOST, Combustion and Flame, 2018.
- Grauer and Steinberg, UBOST, Experiments in Fluids, 2020.
- Raffel, BOS techniques, Experiments in Fluids, 2015.

### 对 OERF 的位置

这是 OERF “极端反应流诊断”和“计算成像”的交叉核心：火焰/高速流改变密度和折射率，折射率梯度造成背景位移，多视角 BOST 再反演三维折射率、密度或温度场。

### 对你的毕设意义

这是第一主线。它能同时满足：

- 贴何远哲：直接对应 NeRIF / PIV-BOST / 4D BOST。
- 贴蔡老师：属于光学诊断、计算成像、机器学习流动显示。
- 可毕业：可以从合成数据 + baseline + 简化神经场开始。
- 可升级：真实 BOST 数据、PIV 补偿、4D 时序都是自然延伸。

### 你要抽取的东西

- BOST 的 forward model：从折射率梯度到背景位移。
- 传统重构 baseline：体素、正则化、UBOST、SIRT/Landweber 风格。
- NeRIF 表示：坐标网络、采样、loss、重投影一致性。
- 指标：relative L2、SSIM/CC、reprojection error、temporal smoothness、runtime。
- 工程接口：相机几何、mask、displacement field、volume domain、unit conversion。

---

## 2. 吸收光谱层析 / 发射层析谱系

### 代表论文

- Cai and Kaminski, Tomographic absorption spectroscopy for gas dynamics and reactive flows, Progress in Energy and Combustion Science, 2017.
- Grauer, Mohri, Yu, Liu and Cai, Volumetric emission tomography for combustion processes, Progress in Energy and Combustion Science, 2023.
- Huang, Liu and Cai, online prediction of 3D flame evolution from history 2D projections, Journal of Fluid Mechanics, 2019.
- Huang, Liu, Wang and Cai, limited-projection volumetric tomography for time-resolved turbulent combustion diagnostics via deep learning, Aerospace Science and Technology, 2020.
- Cai, Huang, Deng and Wang, volumetric reconstruction via transfer learning and semi-supervised learning with limited labels, Aerospace Science and Technology, 2021.
- Molnar et al., neural-implicit laser absorption tomography, Combustion and Flame, 2025.

### 对 OERF 的位置

这条线关注 temperature/species/emission 等物理量的层析成像，是蔡老师早期和持续的重要方向之一。它和 BOST 的共同数学核心是 tomography/inverse problem，但观测物理量不同。

### 对你的毕设意义

这是重要背景，不建议作为主线。你可以用它来：

- 补反问题和层析重构基础。
- 在绪论中说明 OERF 不是只做 BOST，而是长期做 optical tomography。
- 作为 NeRIF 思想迁移的展望，例如 neural-implicit absorption tomography。
- 说明蔡组在 NeRIF 之前已经长期关注 limited projection、limited labels、transfer learning 和时序三维重构。

### 何时转向

只有当何远哲或蔡老师明确给你吸收层析/发射层析数据，或者明确希望你从 BOST 转到另一个 tomography 模态时，才把它变成主线。

---

## 3. 计算成像与微型光谱仪谱系

### 代表方向

- Miniaturized spectrometers and reconstructive spectrometers.
- Snapshot spectral cameras.
- Light-field imaging and holographic imaging.
- AI-driven multidimensional light-field reconstruction.

### 对 OERF 的位置

这是 OERF 的高影响力方向之一，公开网站列出 Science、Nature Photonics、Science Advances、eLight、Nature Electronics 等光谱仪/计算光谱成果。

### 对你的毕设意义

这条线很强，但和何远哲 BOST 主线不是同一条日常同步线。对你更适合作为：

- 了解 OERF 计算成像全景的背景。
- 如果导师明确转向硬件/光谱重构，则作为备选方向。
- 学习“硬件编码 + 计算反演”的思想，迁移到 BOST 的可逆前向模型。

### 不建议现在做的原因

- 需要光谱响应矩阵、器件标定、硬件误差模型。
- 和你已经建立的 BOST/NeRIF 预研积累不直接重合。
- 若没有组里明确数据，容易变成泛泛的压缩感知或谱恢复练习。

---

## 4. 超快激光光谱 / 等离子体 / 极端材料谱系

### 代表方向

- Femtosecond laser spectroscopy.
- Plasma-assisted energy conversion.
- Atmospheric-pressure plasma for extreme-temperature synthesis.
- Ammonia synthesis and combustion kinetics.

### 对 OERF 的位置

这是极端环境能量转换和反应动力学方向，偏实验装置、激光光谱、等离子体和材料过程。

### 对你的毕设意义

目前只做背景，不建议主动转主线。除非导师明确安排，否则它和何远哲 BOST/NeRIF 的算法闭环距离较远。

### 可迁移能力

- 光谱诊断基础。
- 高温/高压/非平衡环境下的测量不确定性。
- 物理约束和数据同化思想。

---

## 5. 反应流仿真与数据同化谱系

### 代表方向

- LES and high-fidelity CFD.
- Machine-learning augmented combustion simulation.
- Data assimilation for turbulent reacting flows.
- Low-carbon ammonia/hydrogen combustion.

### 对 OERF 的位置

这是从“测量物理场”走向“预测和控制反应流”的方向。BOST/NeRIF 得到的密度、折射率、温度或速度场，未来可以作为 data assimilation 的观测输入。

### 对你的毕设意义

它适合作为终章展望，不适合作为第一年主线。原因：

- 需要系统流体力学、湍流、燃烧和数值模拟基础。
- 本科阶段同时做 optical tomography 和 LES/assimilation 风险过高。
- 但你可以把 BOST 输出整理成未来可用于数据同化的结构化场。

### 可做的小接口

- 输出 uncertainty map。
- 统一网格、单位、坐标系和 mask。
- 把重构场保存为后续 CFD/assimilation 可读的数据格式。

---

## 6. Agent for Science 谱系

### 公开方向

蔡老师中文主页列出 Agent for Science，即智能体全自动科研。

### 对你的毕设意义

不要把它当成“做一个聊天机器人”。对你更有价值的定义是：

> 用自动化脚本和报告工具帮助 BOST/NeRIF 做参数扫描、实验记录、图表生成、失败案例索引和结果复现。

这可以作为工程副线，服务主线 A/B/C：

- 自动跑 view/noise/capacity/rank 扫描。
- 自动生成 HTML/PDF 报告。
- 自动检查真实数据 manifest。
- 自动整理每次实验的图、指标、配置和结论。

---

## 7. 选题优先级表

| OERF 谱系 | 对你的优先级 | 适合作用 | 原因 |
| --- | --- | --- | --- |
| BOST / NeRIF / 折射率场 | 最高 | 主线 | 直接贴何远哲，已有 demo，可毕业可升级 |
| PIV-BOST / aero-optics | 高 | 第二阶段升级 | 贴真实实验需求，但数据依赖更高 |
| 4D BOST / tensor decomposition | 高但要收缩 | 挑战章节 | 很亮眼，但完整复现过重 |
| 吸收/发射层析 | 中 | 理论背景/展望 | 数学相近，物理量不同 |
| 微型光谱仪/计算光谱 | 中低 | 备选方向 | 高影响力但偏离何远哲主线 |
| 全息颗粒/金属燃烧 | 中低 | 备选应用 | 需要图像跟踪和燃烧实验资源 |
| 反应流仿真/数据同化 | 中 | 论文展望/接口 | 可连接重构场与模拟，但基础门槛高 |
| Agent for Science | 中高 | 工程副线 | 可服务参数扫描和报告自动化 |
| 等离子体/超快光谱 | 低 | 了解背景 | 物理和实验系统距离当前主线较远 |

---

## 8. 你应该怎么讲自己的定位

不建议说：

> 我想做 AI for fluid，或者我想完整复现 4D BOST。

建议说：

> 我想把毕业设计收束在 OERF 的 BOST/NeRIF 主线：先建立合成数据、传统 baseline、神经隐式折射率场和鲁棒性分析闭环；如果组里有真实 BOST/PIV-BOST/4D 数据，我可以把第二阶段做成数据迁移、PIV 折射补偿误差传播或 4D 低秩时序子问题。

这个表述足够贴组里，又不会把本科毕设推到不可控范围。

---

## 9. 来源入口

- OERF Lab: https://laserdiagnostics.github.io/oerf-lab-website/
- OERF GitHub Pages 源码: https://github.com/laserdiagnostics/oerf-lab-website
- 蔡伟伟中文主页: https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html
- 蔡伟伟英文主页: https://me.sjtu.edu.cn/en/FullTimeTeacher/caiweiwei.html
- 本地 BibTeX 引用库: `references.bib`
- 本地来源审计: `source_audit.md`
