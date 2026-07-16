# 本科毕业设计一页提案

## 拟题

面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

## 背景

背景纹影层析（Background-Oriented Schlieren Tomography, BOST）能够通过多视角背景图像位移反演三维折射率场，并进一步联系到密度、温度等反应流关键物理量。OERF Lab 公开方向中包含极端反应流光学诊断、计算成像、机器学习流动显示和数据融合实验流体力学；何远哲师兄近年的 NeRIF、PIV-BOST 和 4D BOST 工作正处在这些方向的交叉点。

传统体素重构方法在少视角、含噪和高分辨率条件下容易受到病态反演、离散化误差和计算量限制。神经隐式场方法用坐标网络连续表示折射率场，有潜力提升空间分辨率、降低内存成本，并通过重投影验证保持物理一致性。

## 核心问题

在少视角和含噪背景位移条件下，神经隐式折射率场重构相对传统体素/正则化 baseline 的优势边界在哪里？

## 当前预研基础

- 已完成 M0：2D BOST / coordinate-field toy demo，跑通合成折射率场、BOS-like deflection、baseline、coordinate-field inverse 和视角数曲线。
- 已完成 M1：3D-stack sparse-view BOST toy demo，显示 5 视角下三维坐标正则化可降低栈重建误差，但更多干净视角下传统 baseline 更强。
- 已完成 M2：noise-view-capacity robustness scan，系统显示 3/5 视角下 coordinate regularizer 更稳，7/9 视角下传统 stack baseline 更优，并给出表示容量过低会过度平滑的证据。
- 已完成 M3A：PIV-BOST 速度补偿 toy，显示 BOST-style 折射位移补偿可降低合成速度场 RMSE；最新调研已把下一步升级明确为 synthetic particle image pair -> PIVlab/OpenPIV 互相关 -> raw image / displacement / velocity correction 三层接口。
- 已完成 M3B：4D BOST low-rank temporal toy，显示低秩 rank 3 能把逐帧 4D toy 的 mean relative L2 从约 0.366 降到 0.347，并明显降低时序抖动。

## 研究内容

1. 建立合成 BOST 数据生成流程：三维折射率 phantom、多视角位移、噪声扰动。
2. 实现传统 baseline：体素网格、Tikhonov/Landweber/SIRT 简化版本。
3. 实现简化 NeRIF：坐标 MLP、Fourier/hash encoding、位移 loss、梯度一致性 loss。
4. 系统分析视角数、噪声、编码方式、采样策略、场表示方式对重构质量的影响。
5. 若获得课题组数据，接入真实九视角 BOST 或 PIV-BOST 数据，验证迁移性。

## 预期成果

- 一套可复现 Python/PyTorch 代码。
- 合成数据、开源 BOS 数据和 OERF 样例数据的统一接口。
- 重构切片、误差热图、重投影误差、视角数/噪声鲁棒性曲线。
- 开题报告和毕业论文中可直接使用的图表。
- 若数据条件允许，进一步分析 BOST 重构误差到 PIV 速度补偿误差的传播。

## 数据需求

最低需求：

- 合成 phantom 自建。
- Open-source BOS tomography dataset 作为公开预演。

希望从课题组获得：

- 九视角 BOST 原始背景图和扰动图。
- 相机/内窥视角标定参数。
- mask、参考重构结果、论文中可对照的切片或指标。
- 若做 PIV-BOST：同步 PIV 原始粒子图像、时间间隔、PIVlab/OpenPIV/DaVis 参数、BOST 位移/折射率场和导出的速度场。

## 风险与降级

- 如果拿不到真实数据：使用合成数据和开源 BOS 数据完成完整 pipeline。
- 如果 NeRIF 完整复现过重：先做 2D/2.5D 简化模型和粗 baseline。
- 如果 PIV-BOST 几何复杂：先做合成 2D PIV image-pair 补偿 toy，或降级为 velocity-field error propagation。
- 如果 4D BOST 过重：仅作为低秩时序先验 toy 后续拓展。

## 一句话定位

这个毕设不是泛泛做 AI，而是为 OERF 的 BOST/NeRIF 方向建立可复现的神经重构、误差分析和真实数据迁移工具。
