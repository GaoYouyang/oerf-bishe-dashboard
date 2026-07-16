# 开题 PPT 逐页大纲：BOST / NeRIF 毕设

用途：准备和何远哲、蔡老师或学院开题汇报。建议 10 页，控制在 8-12 分钟。

## 推荐题目

面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

备选更工程化题目：

面向 BOST/NeRIF 的可复现重构、误差分析与真实数据迁移工具

## Slide 1: 题目与站位

标题：

面向少视角 BOST 的神经隐式折射率场重构与鲁棒性分析

要讲：

- 本科毕业设计跟随 OERF Lab / Prof. Cai Weiwei / He Yuanzhe。
- 主线是 BOST / NeRIF / PIV-BOST / 4D BOST，不是泛 AI。
- 目标是建立可复现的重构和误差分析流程。

图：

- OERF 方向图或 BOST 链条图。

## Slide 2: OERF 课题组需求

标题：

OERF 的共同关键词：光学诊断、计算成像、数据融合实验流体力学

要讲：

- OERF 公开方向包含极端反应流诊断、计算成像、机器学习流动显示、数据同化。
- 蔡老师主页列出航发/高超内外流诊断、计算成像、微型光谱仪、Agent for Science。
- 本题落在“光学观测 + 计算重构”中间层。

图：

- 四象限：Optical diagnostics / Computational imaging / ML reconstruction / Data assimilation。

## Slide 3: BOST 的物理问题

标题：

从温度/密度场到图像位移

要讲：

- 火焰或高速流改变密度和折射率。
- 折射率梯度使背景图案发生位移。
- 多视角背景位移可以反演三维折射率/密度场。

图：

```text
T field -> rho field -> n field -> ray deflection -> image displacement -> reconstruction
```

## Slide 4: 传统 BOST 的瓶颈

标题：

体素反演的三类问题

要讲：

- 少视角导致病态反演。
- 体素离散带来空间分辨率和伪影问题。
- 大规模矩阵导致内存和计算成本上升。

图：

- voxel grid vs continuous field。
- 一张简单误差示意：视角数少、噪声大、伪影多。

## Slide 5: NeRIF 的核心思想

标题：

用坐标神经场表示连续折射率场

要讲：

- 输入 `(x,y,z)`，输出 `n` 或 `grad n`。
- 沿光线采样并积分，预测背景位移。
- 用重投影误差和梯度一致性训练。
- 重点是物理前向模型，不是普通图像识别网络。

图：

```text
coordinates -> MLP -> refractive index / gradient -> ray integration -> displacement loss
```

## Slide 6: 本科切入点

标题：

先做可毕业闭环，再做真实数据升级

要讲：

- 第一阶段：合成 phantom + 多视角位移 + baseline + NeRIF 简化版。
- 第二阶段：视角数、噪声、编码、采样策略的鲁棒性分析。
- 第三阶段：接真实 BOST/PIV-BOST 数据或做 4D toy。
- 当前已经完成 M0/M1/M2/M3A/M3B 五个轻量闭环：2D toy、3D-stack sparse-view toy、noise-view-capacity robustness scan、PIV-BOST velocity compensation toy、4D low-rank temporal toy。

表：

| 阶段 | 数据 | 方法 | 输出 |
| --- | --- | --- | --- |
| 保底 | 合成 phantom | baseline + MLP | 切片、误差、重投影 |
| 加强 | 开源/真实 BOST | 参数扫描 | 鲁棒性图谱 |
| 冲刺 | PIV-BOST / 4D | 补偿或时序先验 | 工具或子问题 |

## Slide 7: 实验设计

标题：

参数扫描让“复现”变成“研究”

要讲：

- 视角数：3 / 5 / 7 / 9 / 12。
- 噪声：无噪声、低噪声、中噪声。
- 表示：体素 baseline、Fourier MLP、可选 hash encoding。
- 指标：L2、SSIM、CC、PSNR、re-projection error、时间、显存。
- 当前 M2 初步结果：3/5 视角下 coordinate regularizer 更稳，7/9 视角下传统 stack baseline 更强；容量过低会过度平滑。

图：

- `demo_m2/results/m2_improvement_heatmap.png`。
- `demo_m2/results/m2_capacity_scan.png`。

## Slide 8: 可选升级路线

标题：

根据数据条件选择 PIV-BOST 或 4D BOST

要讲：

- 如果有同步 PIV-BOST 数据：做折射率补偿误差传播。
- 当前 M3A toy 已经显示：BOST-style compensation 可把合成速度场 RMSE 从约 0.0101 降到约 0.0067，但局部残余误差仍存在。
- 如果有时序 BOST 数据：做低秩时序先验 toy。
- 当前 M3B toy 已经显示：低秩 rank 3 可把逐帧 4D toy mean relative L2 从约 0.366 降到 0.347，并把 temporal smoothness 从约 0.279 降到 0.177。
- 如果暂无数据：用开源 BOS/TBOS 数据预演 pipeline。

决策树：

```text
有真实 BOST -> 真实数据迁移
有 PIV-BOST -> 2D PIV compensation
有时序数据 -> 4D low-rank toy
都没有 -> synthetic + open-source BOS benchmark
```

可放图：

- `demo_m3a/results/m3a_compensation_summary.png`。
- `demo_m3b/results/m3b_4d_summary.png` 或 `demo_m3b/results/m3b_rank_scan.png`。

## Slide 9: 时间计划与成果

标题：

三个月开题准备 + 一年毕业设计路线

要讲：

- 1-4 周：基础、BOST forward、phantom、PyTorch 坐标场。
- 5-8 周：baseline、NeRIF 简化版、指标和图表。
- 9-12 周：鲁棒性扫描、数据请求、开题 memo。
- 一年：真实数据/补偿/4D 子问题 + 论文写作。

成果：

- 代码仓库。
- 可复现实验脚本。
- 参数扫描报告。
- 毕业论文图表。
- 真实数据接口或开源数据迁移。
- 当前已有可展示图：M0 总图、M1 体切片、M2 改进热图和容量扫描。
- 若讲 PIV-BOST 升级路线，可加 M3A 补偿总图；若讲 4D BOST 升级路线，可加 M3B rank trade-off 图。

## Slide 10: 需要课题组支持

标题：

我需要确认的数据、代码和边界

要讲：

- 是否有可给本科生的九视角 BOST 样例。
- 原始图、位移场、标定、参考重构是否能给一小份。
- PIV-BOST 是 2D 还是 stereo-PIV。
- 哪些数据/图可以写进本科论文。
- 师兄更希望我先做算法复现、真实数据接口、参数扫描还是自动报告工具。

最后一句：

我希望先用合成数据建立 NeRIF/BOST 最小闭环，再根据组内数据条件向真实 BOST、PIV-BOST 或 4D BOST 子问题升级。

## 答辩可能被问到的问题

1. 你的创新点是什么？
   - 不是提出全新大模型，而是围绕少视角、噪声、编码和采样策略，系统量化 NeRIF 类方法的适用边界，并建立可迁移接口。

2. 如果拿不到真实数据怎么办？
   - 使用合成 phantom 和开源 BOS/TBOS 数据完成完整 pipeline，真实数据作为升级项。

3. 为什么不直接做完整 4D BOST？
   - 完整 4D BOST 对数据、算力和实现要求过高，本科阶段更适合拆低秩时序先验 toy 或参数扫描子问题。

4. 为什么你一个物理本科能做？
   - 题目的核心是几何光学、反问题、数值优化和神经场，物理背景有优势；流体和燃烧部分按最小可用路线补。

5. 成果如何验证？
   - 合成数据有 ground truth；真实数据用 leave-one-view-out 重投影、SSIM/CC/PSNR 和物理一致性验证。
