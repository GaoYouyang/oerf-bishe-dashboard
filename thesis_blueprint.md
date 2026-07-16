# 毕业论文蓝图：从 M0 demo 到 BOST / NeRIF 本科论文

用途：把前期调研、论文阅读和 demo 逐步收束成一篇能答辩的本科毕业论文。它不是最终目录，而是你和何远哲确认题目后可以直接修改的骨架。

## 推荐论文题目

首选：

面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

工程化版本：

面向 BOST / NeRIF 的可复现重构、误差分析与真实数据迁移工具

若接 PIV-BOST：

基于背景纹影层析折射率场的 PIV 速度测量误差补偿方法初探

若接 4D BOST：

四维背景纹影层析中的低秩时序先验及其合成数据验证

## 论文主线一句话

本文围绕反应流非接触光学诊断中的背景纹影层析重构问题，建立从合成折射率场、多视角偏折观测、传统基线反演到神经隐式场重构的可复现实验流程，并系统分析视角数、噪声、坐标编码和采样策略对重构质量的影响。

## 推荐目录

### 摘要

要写清楚：

- BOST 通过图像位移反演折射率/密度场。
- 传统体素方法在少视角和噪声下有病态性。
- 本文建立可复现 demo 与鲁棒性分析流程。
- 若有真实数据，接入 OERF 九视角 BOST 或 PIV-BOST。

关键词：

背景纹影层析；折射率场；神经隐式表示；反问题；流动显示

### 第 1 章 绪论

目标：

把题目放进 OERF 的真实需求里，而不是讲泛 AI。

建议小节：

1.1 极端反应流非接触测量需求  
1.2 背景纹影与背景纹影层析技术  
1.3 计算成像与神经隐式表示在流场重构中的应用  
1.4 本文研究内容与技术路线

必须引用：

- OERF / 蔡老师公开方向。
- BOS techniques 2015。
- BOST 前史。
- NeRIF / PIV-BOST / 4D BOST。
- Computational Flow Visualization 2024。

建议图：

- 图 1-1 OERF 方向和本文位置。
- 图 1-2 `T -> rho -> n -> ray deflection -> image displacement -> reconstruction` 链条图。

### 第 2 章 BOST 物理模型与反问题基础

目标：

把物理本科优势写出来：你不是只调网络，而是理解观测和未知量的物理关系。

建议小节：

2.1 折射率、密度与温度的关系  
2.2 小角度近似下的光线偏折模型  
2.3 多视角背景位移与层析反演  
2.4 传统体素反演与正则化  
2.5 评价指标：重构误差与重投影误差

建议公式：

- Gladstone-Dale 关系：`n - 1 = K rho`。
- 位移近似与折射率梯度积分。
- 离散 forward model：`y = A x + epsilon`。
- 正则化反演：`min ||Ax-y||^2 + lambda R(x)`。

建议图：

- 图 2-1 BOS/BOST 几何示意。
- 图 2-2 forward problem 与 inverse problem。
- 图 2-3 少视角病态性示意。

### 第 3 章 神经隐式折射率场方法

目标：

从 NeRIF 中抽象出本科可实现的简化方法。

建议小节：

3.1 坐标神经场表示  
3.2 Fourier / hash encoding 的作用  
3.3 位移重投影损失与梯度一致性  
3.4 本文简化实现和完整 NeRIF 的差异  
3.5 实验流程与代码结构

建议公式：

- `f_theta(x,y,z) -> n` 或 `f_theta(x,y,z) -> (n, grad n)`。
- `L = L_disp + alpha L_grad + beta L_reg`。
- 重投影误差定义。

建议图：

- 图 3-1 coordinate MLP 框架图。
- 图 3-2 M0/M1 实验流程图。
- 图 3-3 代码模块关系图。

注意：

如果还没有 PyTorch 版 NeRIF，就把 M0 的 coordinate-field toy 写成“最小坐标场思想验证”，不要声称完整复现 NeRIF。

### 第 4 章 合成数据与最小 demo

目标：

把 M0 demo 和后续 M1/M2 结果变成论文实验章。

建议小节：

4.1 合成折射率 phantom 构造  
4.2 多视角 BOS-like 偏折观测生成  
4.3 传统 baseline 与 coordinate-field toy  
4.4 视角数敏感性实验  
4.5 噪声与编码敏感性实验

当前已有材料：

- `demo_m0/results/m0_summary.png`
- `demo_m0/results/view_count_curve.png`
- `demo_m0/results/metrics.csv`
- `demo_m1/results/m1_volume_summary.png`
- `demo_m1/results/m1_view_count_curve.png`
- `demo_m1/results/metrics.csv`
- `demo_m2/results/m2_improvement_heatmap.png`
- `demo_m2/results/m2_noise_view_scan.png`
- `demo_m2/results/m2_capacity_scan.png`

建议图：

- 图 4-1 M0 ground truth / baseline / coordinate-field / error。
- 图 4-2 view-count curve。
- 图 4-3 M1 3D-stack sparse-view volume slices。
- 图 4-4 M1 3D-stack view-count sensitivity。
- 图 4-5 M2 noise-view robustness scan。
- 图 4-6 M2 improvement heatmap。
- 图 4-7 coordinate regularizer capacity scan。

结果叙事：

- M0 证明最小闭环已经跑通。
- 9 视角下 coordinate-field toy 略优于 baseline。
- 低视角时 coordinate-field 可能更不稳定，说明神经表示不是天然更好，需要视角数、编码、正则和采样策略共同约束。
- M1 进一步说明：三维坐标正则化在 3-5 视角 sparse-view 情况下能压低栈重建误差，但在干净的 7/9/13 视角条件下，传统栈重建会追上或超过它。
- M2 把这个结论系统化：3/5 视角全部噪声设置中 coordinate regularizer 胜出，7/9 视角全部设置中 baseline 胜出；容量扫描显示表示容量过低会过度平滑，容量提高后才有收益。
- 这正是后续鲁棒性研究的价值。

### 第 5 章 真实数据迁移或升级子问题

根据师兄给的数据选择一个版本。

#### A 版本：真实 BOST 数据迁移

写法：

- 数据格式、相机/内窥几何、mask、位移场。
- 从 synthetic pipeline 到真实数据 loader。
- 重投影验证和可视化。

图：

- 真实位移图。
- 真实重构切片。
- leave-one-view-out 重投影误差。

#### B 版本：PIV-BOST 补偿

写法：

- 折射率梯度导致 PIV 粒子像素偏移。
- 2D PIV toy 或真实同步数据。
- 补偿前后速度误差比较。
- 当前已有 `demo_m3a/` 向量场补偿 toy：observed PIV RMSE 约 0.0101，BOST-style compensation 后约 0.0067，95 分位误差从约 0.0240 降到约 0.0072。
- 诚实边界：当前不是粒子图像互相关，只是速度场误差传播；真实升级需要确认在原始粒子图像层、速度矢量层还是误差量化层做补偿。

图：

- 粒子图像位移示意。
- 速度矢量图。
- 补偿前后 error map。
- `demo_m3a/results/m3a_compensation_summary.png`。
- `demo_m3a/results/m3a_error_profile.png`。

#### C 版本：4D BOST low-rank toy

写法：

- 逐帧重构的抖动和计算成本。
- 低秩时序先验。
- rank、帧数、噪声对误差和时间一致性的影响。
- 当前已有 `demo_m3b/` 低秩时序 toy：逐帧 baseline mean relative L2 约 0.366，low-rank rank 3 后约 0.347；temporal smoothness 从约 0.279 降到约 0.177。
- 诚实边界：低秩先验能减少逐帧抖动，但不能自动修正系统性几何/forward model 偏差。

图：

- moving phantom。
- 逐帧 vs low-rank。
- temporal smoothness curve。
- `demo_m3b/results/m3b_4d_summary.png`。
- `demo_m3b/results/m3b_rank_scan.png`。
- `demo_m3b/results/m3b_temporal_trace.png`。

### 第 6 章 总结与展望

要写清楚：

- 本文完成了什么闭环。
- 哪些结果是可毕业的核心成果。
- 哪些是后续接入 OERF 真实系统的方向。
- 不夸大：如果只是 toy，就明确 toy 的边界。

展望：

- 完整 NeRIF 复现。
- 真实九视角 BOST 数据。
- PIV-BOST 速度补偿。
- 4D BOST 时序先验。
- 数据融合实验流体力学。

## 图表清单

| 编号 | 图表 | 来源 | 当前状态 |
| --- | --- | --- | --- |
| 图 1-1 | OERF 研究方向与本文位置 | `figures/oerf_position_map.png` | 已生成 |
| 图 1-2 | BOST 物理链条 | `figures/bost_physical_chain.png` | 已生成 |
| 图 2-1 | BOS/BOST 几何示意 | `figures/bost_physical_chain.png` 可先代替 | 后续可进一步画实验几何 |
| 图 2-2 | forward/inverse problem | `figures/nerif_pipeline.png` 可拆分 | 已有初版 |
| 图 3-1 | NeRIF coordinate-field 框架 | `figures/nerif_pipeline.png` | 已生成，避免直接搬论文图 |
| 图 3-2 | 代码模块图 | starter spec / `figures/nerif_pipeline.png` | 已有初版 |
| 图 4-1 | M0 demo 总图 | `demo_m0/results/m0_summary.png` | 已生成 |
| 图 4-2 | 视角数曲线 | `demo_m0/results/view_count_curve.png` | 已生成 |
| 图 4-3 | M1 3D-stack 体切片与误差 | `demo_m1/results/m1_volume_summary.png` | 已生成 |
| 图 4-4 | M1 3D-stack 视角数曲线 | `demo_m1/results/m1_view_count_curve.png` | 已生成 |
| 图 4-5 | M2 noise-view 鲁棒性曲线 | `demo_m2/results/m2_noise_view_scan.png` | 已生成 |
| 图 4-6 | M2 改进热图 | `demo_m2/results/m2_improvement_heatmap.png` | 已生成 |
| 图 4-7 | M2 coordinate regularizer 容量扫描 | `demo_m2/results/m2_capacity_scan.png` | 已生成 |
| 图 5-1 | M3A PIV-BOST 补偿 toy 总图 | `demo_m3a/results/m3a_compensation_summary.png` | 已生成 |
| 图 5-2 | M3A 中心线误差剖面 | `demo_m3a/results/m3a_error_profile.png` | 已生成 |
| 图 5-3 | M3B 4D low-rank toy 总图 | `demo_m3b/results/m3b_4d_summary.png` | 已生成 |
| 图 5-4 | M3B rank trade-off | `demo_m3b/results/m3b_rank_scan.png` | 已生成 |
| 图 5-5 | M3B temporal motion trace | `demo_m3b/results/m3b_temporal_trace.png` | 已生成 |
| 图 5-6 | 真实 BOST / PIV / 4D 升级图 | 根据师兄数据 | 待定 |
| 图 5-7 | 数据接口与请求清单 | `figures/data_interface_checklist.png` | 已生成 |
| 图 5-8 | 选题分支决策树 | `figures/topic_decision_tree.png` | 已生成 |
| 图 6-1 | 三个月推进路线 | `figures/three_month_roadmap.png` | 已生成 |

## 结果表清单

| 表格 | 内容 | 当前状态 |
| --- | --- | --- |
| 表 1 | 核心文献与本文关系 | 可由 references.bib 和网页整理 |
| 表 2 | 实验参数：视角数、噪声、表示容量 | M2 已有 |
| 表 3 | baseline vs coordinate-field / coordinate-regularized 指标 | M0/M1 已有 |
| 表 4 | PIV-BOST 补偿前后误差 | M3A 已有 |
| 表 5 | 4D low-rank rank scan | M3B 已有 |
| 表 6 | 真实数据格式与字段 | 等师兄反馈 |
| 表 7 | 风险与替代路线 | 网页已有风险表 |

## 写作节奏

第 1-2 周：

- 先写第 1 章和第 2 章草稿，不等实验全部完成。
- 每个公式后面写一句“本文中这个量对应什么代码变量”。

第 3-6 周：

- 写第 3 章方法。
- 把 M0/M1/M2 demo 的图表逐步放进第 4 章。

第 7-10 周：

- 根据师兄数据，写第 5 章升级路线。
- 如果没有真实数据，第 5 章改成开源数据迁移和 PIV/4D toy。

第 11-12 周：

- 写摘要、总结、致谢、参考文献。
- 全文统一变量、图编号和引用。

## 常见写作坑

- 不要把“神经网络效果更好”当成默认结论。要写“在哪些条件下更稳，在哪些条件下失败”。
- 不要只报告 L2，要报告重投影误差和物理解释。
- 不要直接搬 NeRIF / OERF 论文图，重画成自己的流程图。
- 不要把旁支论文堆进参考文献。主线 8-12 篇最重要。
- 不要等最后一个月才写论文。第 1 章和第 2 章现在就能写。

## 论文中的诚实边界

如果只完成 M0/M1：

本文完成了二维/简化三维合成数据上的 BOST-NeRIF 最小闭环和鲁棒性分析，为后续真实数据迁移提供代码框架。

如果完成真实 BOST：

本文进一步将合成数据流程迁移到 OERF 真实 BOST 样例，验证了数据接口和重投影一致性。

如果完成 PIV-BOST：

本文初步分析了 BOST 折射率场对 PIV 速度误差补偿的作用，并在合成或真实同步数据上给出补偿前后对比。

如果完成 4D toy：

本文初步探索了低秩时序先验在 4D BOST 合成数据中的作用，展示其对逐帧重构抖动的抑制效果，并指出 rank 选择、系统性几何偏差和真实时序数据接口仍是后续工作重点。
