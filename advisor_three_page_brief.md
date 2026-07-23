# 给导师/师兄看的三页沟通 Brief

用途：这份材料不是完整调研报告，而是和何远哲、蔡老师沟通时可以快速打开的压缩版。建议控制在 8-10 分钟内讲完。

最后核验：2026-07-06。事实依据优先来自 OERF 公开网站、上海交通大学机械与动力工程学院蔡伟伟主页、出版社/DOI 页面和本工作台已整理的引用库。

---

## 第 1 页：我想做的问题和它为什么贴 OERF

### 一句话题目

面向少视角 Background-Oriented Schlieren Tomography 的神经隐式折射率场重构与鲁棒性分析。

### 事实依据

- OERF 公开研究方向包含极端反应流光学诊断、计算成像、微型计算光谱仪、超快光谱/等离子体、反应流仿真与数据同化。
- 蔡伟伟老师交大主页列出的方向包括 Agent for Science、航发/高超飞行器内外流诊断、实验流体力学、燃烧诊断、计算成像、微型计算光谱仪和机器学习流动显示。
- 何远哲师兄近年的 NeRIF、PIV-BOST、4D BOST 论文正好处在“光学诊断 + 计算重构 + 三维/四维流场可视化”的交叉处。

### 我的判断

最适合本科毕设的切入点不是泛 AI，也不是从零搭实验台，而是：

> 把 BOST 图像位移到三维/四维折射率场的反演过程做成可复现、可解释、可对比的算法与误差分析工具。

这样做有三个好处：

- 对 OERF 有用：能服务真实 BOST/NeRIF/PIV-BOST/4D 数据的处理、参数扫描和可视化。
- 对本科毕设可控：即使暂时拿不到内部真实数据，也能用合成数据和开源 BOS 数据完成闭环。
- 对后续深造有积累：能沉淀 inverse problem、optical diagnostics、neural field、scientific ML 的共同能力。

---

## 第 2 页：我已经做了什么，下一步想请师兄定什么

### 已有预研闭环

| 模块 | 内容 | 当前价值 |
| --- | --- | --- |
| M0 | 2D BOST / coordinate-field toy | 证明图像位移、传统 baseline、坐标场反演和指标输出已经跑通 |
| M1 | 3D-stack sparse-view BOST toy | 把问题推进到三维栈，显示 5 视角下坐标正则化有帮助 |
| M2 | view-noise-capacity robustness scan | 发现少视角下 coordinate regularizer 更稳，但视角充分时传统 baseline 更强 |
| M3A | PIV-BOST velocity compensation toy | 用向量场层面模拟折射位移如何污染 PIV 速度，以及 BOST-style compensation 的价值 |
| M3B | 4D BOST low-rank temporal toy | 用时序三维 phantom 测试低秩时序先验能否减少逐帧抖动 |

### 现在最需要师兄帮助判断的 3 件事

1. 毕设主线是否应定为 NeRIF/BOST 鲁棒性分析，而 PIV-BOST 和 4D BOST 只作为第二阶段升级？
2. 组里是否能给一小份可公开/可内部使用的样例数据：九视角 BOST、PIV-BOST 或 4D BOST 的简化数据包？
3. 真实数据的主要困难是什么：相机几何、view orientation、mask/ROI、背景位移提取、噪声、视角数、spatial resolution、重建速度，还是结果可视化和自动报告？

### 我希望拿到的最低数据/接口

- 原始背景图、扰动图或已经提取好的 displacement field。
- 相机/视角几何，至少包括 view angle、pixel size、object domain、mask。
- 论文或组内 baseline 的参考切片、误差指标或重投影误差。
- 如果已有：bad-view 记录、有效视场裁剪、resolution phantom/MTF 或其他分辨率评价口径。
- 如果是 PIV-BOST：粒子图像、时间间隔、PIV 参数、未补偿/补偿速度场。
- 如果是 4D BOST：时间序列帧数、每帧视角数、是否已有低秩/张量分解 baseline。

---

## 第 3 页：建议定题方案和一年路线

### 推荐定题

**主标题：**

面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

**更工程化标题：**

面向 OERF BOST/NeRIF 数据的可复现重构、误差分析与真实数据迁移工具

### 分阶段目标

| 阶段 | 时间 | 目标 | 交付 |
| --- | --- | --- | --- |
| 预研到开题 | 0-3 个月 | 补流体/光学/反问题基础，读 NeRIF/PIV-BOST/4D BOST，完善 M0-M3B | 开题报告、PPT、demo 图、数据需求清单 |
| 毕设前半程 | 1-3 个月 | 完成合成 3D BOST 数据、传统 baseline、简化 NeRIF/PyTorch 版 | 可运行代码、参数扫描、误差图谱 |
| 毕设中段 | 4-6 个月 | 接入真实或开源 BOST 数据，做数据接口和指标统一 | 数据 manifest、重构结果、baseline 对比 |
| 毕设后半程 | 7-9 个月 | 根据师兄需求选择 PIV-BOST 误差传播或 4D 低秩时序子问题 | 一个升级章节或工具模块 |
| 收尾 | 10-12 个月 | 完成论文、答辩图表、可复现实验包 | 论文、答辩 PPT、代码归档 |

### 风险控制

- 不承诺完整复现 4D BOST 论文，只做低秩时序先验和指标子问题。
- 不承诺完整真实 PIV cross-correlation pipeline，先做速度场层面误差传播，再根据数据升级。
- 不把光谱仪、金属颗粒、CFD、Agent for Science 混成主线，只作为 OERF 背景或备选迁移方向。
- 每周都留下一个可检查产物：图、脚本、表格、问题清单或阅读卡。

### 会后 7 天动作

1. 根据师兄建议把 A/B/C 路线收束成一个主线题目，并决定是否把 E 副线作为误差分析增强项。
2. 拿到或确认无法拿到样例数据；若暂时没有，则改用合成数据和开源 BOS 数据继续。
3. 把 M0-M3B 中最相关的一个升级成 PyTorch/可微版本。
4. 写一页正式 proposal，列出目标、数据、baseline、指标、风险和下周交付。

---

## 可附链接

- OERF Lab: https://laserdiagnostics.github.io/oerf-lab-website/
- 蔡伟伟中文主页: https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html
- NeRIF DOI: https://doi.org/10.1063/5.0250899
- PIV-BOST DOI: https://doi.org/10.1007/s00348-025-04093-y
- 4D BOST DOI: https://doi.org/10.1145/3809488
- 本地总工作台: `index.html`
