# 物理本科到 OERF / BOST-NeRIF 的基础补课路线

生成日期：2026-07-07

目标：不是把流体力学、燃烧、光学、反问题、PyTorch 全部系统学完，而是在三个月内补到能读懂 NeRIF / PIV-BOST / 4D BOST，并能做一个可运行 demo。

## 总原则

- 先补“能解释变量”的知识，再补“能推公式”的知识。
- 每学一个概念，都要落到 BOST 链条：温度 -> 密度 -> 折射率 -> 光线偏折 -> 图像位移 -> 三维重构。
- 不要从完整 LES、详细燃烧化学、实验光路设计开局。

## 第一层：流体力学最低可用基础

资源：

- MIT OCW 2.06 Fluid Dynamics: `https://ocw.mit.edu/courses/2-06-fluid-dynamics-spring-2013/`

必须掌握：

- 连续介质假设。
- 欧拉描述和拉格朗日描述。
- 控制体、质量守恒、动量守恒。
- Navier-Stokes 方程每一项的物理意义。
- 不可压/可压、层流/湍流、边界层。
- Reynolds number、Mach number、Prandtl number 的含义。

和毕设的连接：

- BOST 测到的不是“速度”，而是折射率/密度变化造成的光线偏折。
- PIV 测速度，但速度误差受折射率场影响。
- 你不需要一开始会求解 N-S，只需要能解释为什么密度/温度/速度是流场物理量。

最低验收：

- 能用 300 字解释为什么火焰会改变折射率。
- 能写出 `rho`, `T`, `n`, `u` 分别是什么物理量。
- 能解释 Reynolds number 对湍流火焰实验的意义。

## 第二层：燃烧与反应流入门

资源：

- Princeton CEFRC Combustion Summer School notes: `https://cefrc.princeton.edu/combustion-summer-school/lecture-notes`

必须掌握：

- 当量比、预混火焰、扩散火焰。
- Bunsen flame 的基本结构。
- 火焰中温度、密度、折射率的空间分布。
- 为什么反应流测量需要非接触光学诊断。
- OH/CH/PLIF、Rayleigh、LII、BOS/PIV 大概测什么。

和毕设的连接：

- NeRIF 的实验对象是 Bunsen flame 类反应流。
- PIV-BOST 是在 reacting field 里同时测速度和折射率。
- 你只需要能写清楚“为什么这是极端反应流诊断问题”，不需要一开始做详细化学机理。

最低验收：

- 能解释 premixed Bunsen flame 和 non-piloted flame 大概是什么。
- 能说明温度升高为什么通常导致密度下降。
- 能把 Gladstone-Dale 关系和密度场联系起来。

## 第三层：几何光学与 BOS/BOST

核心概念：

- 光线通过折射率梯度区域会偏折。
- 背景点阵在相机中发生位移。
- BOS 用图像相关/光流估计位移。
- BOST 用多视角位移反演三维折射率/密度场。

建议阅读：

- Raffel 2015 Background-oriented schlieren techniques。
- Grauer 2018 Instantaneous 3D flame imaging by BOST。
- Liu/Shui/Cai 2020 single-camera endoscopic BOST。
- NeRIF paper 的 BOST mathematical formulation。

最低验收：

- 能画出相机、背景板、火焰、光线偏折示意图。
- 能说明 image displacement 是观测量，refractive index field 是未知量。
- 能解释为什么多视角比单视角能重构三维场。

## 第四层：反问题、层析与正则化

资源：

- Kak & Slaney, Principles of Computerized Tomographic Imaging: `https://engineering.purdue.edu/~malcolm/pct/`

必须掌握：

- forward problem 和 inverse problem。
- 投影、Radon transform 的直觉。
- 病态问题、噪声放大、正则化。
- Tikhonov / Landweber 的基本思想。
- 为什么少视角层析需要先验。

和毕设的连接：

- BOST 是典型反问题：从有限投影/位移反推三维场。
- NeRIF 的价值不是“神经网络万能”，而是连续表示 + 物理前向模型 + 正则化式 loss。

最低验收：

- 能用一页图说明 `field -> projection -> measurement` 和 `measurement -> reconstruction`。
- 能说出 baseline 为什么必须存在。
- 能解释 re-projection error 为什么比单纯网络 loss 更有物理意义。

## 第五层：PIV 与图像位移估计

资源：

- OpenPIV docs: `https://openpiv.readthedocs.io/`
- PIVlab 文档和示例。
- PIVlab 2021 论文：传统互相关增强算法和误差表现。
- piv-image-generator / SynthPix：生成 synthetic particle image pair，避免一开始依赖组内原始 PIV 数据。

必须掌握：

- 粒子图像对、时间间隔、互相关窗口。
- 像素位移到速度的换算。
- window size、overlap、outlier removal。
- 折射率导致粒子像素位置偏差。
- raw image dewarping、correlation/displacement correction、velocity-vector correction 三层接口的区别。

和毕设的连接：

- 如果做 PIV-BOST，先从 2D synthetic particle image pair 开始。
- 不要一开始承诺 stereo-PIV。

最低验收：

- 能用 OpenPIV/PIVlab 处理一对合成粒子图像。
- 能人为给图像加一个位移场，再看速度误差。
- 能写清楚 PIV-BOST 补偿的输入和输出：原始图像、位移场还是速度场。

## 第六层：PyTorch 与神经隐式场

资源：

- PyTorch Learn the Basics: `https://docs.pytorch.org/tutorials/beginner/basics/intro.html`

必须掌握：

- Tensor、autograd、optimizer、loss。
- MLP、activation、Fourier feature。
- 坐标网络：输入 `(x,y,z)`，输出 `n`。
- mini-batch ray sampling。
- 可视化训练曲线和切片。

和毕设的连接：

- NeRIF 不是 CNN 图像识别，而是 coordinate-based neural field。
- 最小实现可以先只输出折射率，再逐步加入梯度 head。

最低验收：

- 能训练一个 MLP 拟合 `sin(x)+cos(y)+gaussian(z)`。
- 能对输出关于输入求梯度。
- 能把三维场切片画出来。

## 三个月补课安排

| 周 | 主线 | 输出 |
| --- | --- | --- |
| 1 | 流体力学变量和 BOST 链条 | 一页变量图：`T-rho-n-displacement` |
| 2 | BOS/BOST 原理 | 相机-背景-火焰示意图和位移场 toy |
| 3 | 反问题/层析 | 一个 2D Radon 或投影重建 notebook |
| 4 | PyTorch 坐标网络 | MLP 拟合三维 phantom |
| 5 | NeRIF 数学形式 | 写出 forward model 和 loss 草图 |
| 6 | 合成 BOST 数据 | 3/5/7/9 视角位移生成 |
| 7 | baseline | 体素/正则化粗重建 |
| 8 | NeRIF 简化版 | 坐标 MLP 重构 + 切片图 |
| 9 | 鲁棒性 | 视角数、噪声、编码扫描 |
| 10 | PIV-BOST toy 或 4D toy | 根据师兄反馈选一条 |
| 11 | 真实/开源数据接口 | 数据目录、读取、可视化 |
| 12 | 开题 memo | 6-8 页报告 + demo 图 |

## 不要过度学习的内容

- 完整湍流理论。
- 详细化学反应机理。
- 高保真 LES。
- 光学硬件制备。
- 大模型/agent 框架。

这些都可以了解，但不是你前三个月的主线。

## 推荐的每周学习产物

- 一张概念图。
- 一个最小代码实验。
- 一张结果图。
- 一个要问何远哲的问题。

如果一周没有产出图或代码，说明学习没有转化成毕设推进。
