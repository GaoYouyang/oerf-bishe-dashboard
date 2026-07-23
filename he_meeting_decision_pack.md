# 何远哲会前决策包

用途：把当前工作台、M0-M3B demo 和数据请求清单压缩成一次 15-30 分钟沟通能用的材料。目标不是把所有东西讲完，而是让师兄帮你定一个本科毕设主线。

## 一句话开场

师兄，我现在想把毕业设计收束在 BOST / NeRIF 主线：先做可复现的合成数据、baseline、坐标场/隐式场重构和鲁棒性分析；如果组里更需要 PIV-BOST 或 4D BOST 子模块，我已经各做了一个最小 toy，想请您帮我判断哪个方向最贴组里需求、数据最可能拿到、最适合本科毕业设计。

## 15 分钟展示顺序

### 1. 先讲主线判断，1 分钟

- 不做泛 AI。
- 不从完整 4D BOST 或真实湍流燃烧全流程开局。
- 主线是：光学观测位移 `->` 折射率/密度场 `->` 误差和鲁棒性分析。

### 2. 展示 M0-M2：NeRIF/BOST 保底主线，5 分钟

展示文件：

- `demo_m0/results/m0_summary.png`
- `demo_m1/results/m1_volume_summary.png`
- `demo_m2/results/m2_improvement_heatmap.png`
- `demo_m2/results/m2_capacity_scan.png`

要讲的结论：

- M0 证明我已经跑通 2D BOST / coordinate-field 闭环。
- M1 证明 3D-stack sparse-view 下，坐标正则化有可能帮助少视角重构。
- M2 证明这个优势有边界：3/5 视角下 coordinate regularizer 更稳，7/9 视角下传统 stack baseline 更强；容量太低会过度平滑。
- 因此主问题可以定为：隐式/坐标先验在少视角、含噪和真实几何受限 BOST 中的适用边界。下一步不只是换网络，而是把 view orientation、mask/ROI、spatial resolution 和 calibration/forward-model 误差纳入 M3C 质量控制。

### 3. 展示 M3A：PIV-BOST 备选升级，3 分钟

展示文件：

- `demo_m3a/results/m3a_compensation_summary.png`
- `demo_m3a/results/m3a_error_profile.png`

要讲的结论：

- 当前只是速度场误差传播 toy，不是完整 PIV 图像互相关。
- observed PIV RMSE 约 0.0101，BOST-style compensation 后约 0.0067。
- 真实问题是：组里更需要校正原始粒子图像、校正速度矢量场，还是量化 BOST 误差到 PIV 误差的传播？

### 4. 展示 M3B：4D BOST 备选升级，3 分钟

展示文件：

- `demo_m3b/results/m3b_4d_summary.png`
- `demo_m3b/results/m3b_rank_scan.png`
- `demo_m3b/results/m3b_temporal_trace.png`
- `demo_m3b/results/m3b_six_axis_overview.png`
- `demo_m3b/results/m3b_bias_dynamics_diagnostic.png`
- `demo_m3b/results/m3b_rank3_operating_map.png`
- `demo_m3b/results/m3b_rank_selection_stability.png`

要讲的结论：

- 当前只是 4D low-rank temporal toy，不是 ACM TOG 完整复现。
- 8 个配对种子的六轴实验中，rank 3 把全局相对 L2 改善 3.90%、时间一阶误差改善 44.68%、held-out deflection 改善 9.23%。
- 反例同样明确：无噪声场 L2 恶化 0.56%、mass trace 恶化 0.94%，sync lag 仍是最大系统误差。
- 交叉实验覆盖 80 个环境和 3,200 组配对比较：rank 3 是 regret 最低的固定默认值，但只在 20/80 环境逐格最优，3 views 全部偏向 rank 2，19/80 个 rank-3 环境 field 负收益。
- 跨形态无真值 selector 又覆盖 864 个观测格、6,048 个候选：固定 rank 3 mean/p95 regret 为 1.561%/5.635%；无 held-out 多特征 selector 为 0.252%/1.267%，带 held-out 为 0.210%/1.054%；直接 held-out minimum 反而失败。
- 真实问题现在是：下一轮最该做 leave-one-geometry-out、UQ/拒答、bias/sync/exposure blur，还是直接对真实数据做 failure-signature 报告？

### 5. 让师兄做选择，3 分钟

直接问：

1. 如果我只能选一个主线，您建议我选 T16 support-consistent 三维逆算子、NeRIF/BOST 鲁棒性、PIV-BOST 补偿，还是 4D BOST 可观测容量选择？
2. 组里目前最缺的是复现代码、真实数据接口、参数扫描、可视化图表，还是自动报告工具？
3. 您能否给我一小份不涉密样例数据，让我先做数据读取、可视化、mask/geometry 检查和重投影验证？

## 三种定题路径

### A 路线：师兄给 BOST / NeRIF 数据

建议题目：

面向真实 BOST 数据的神经隐式折射率场重构与鲁棒性分析

第一周交付：

- 整理 manifest。
- 读入九视角原始图或位移场。
- 画每个视角的 flow-off / flow-on / displacement / mask。
- 复现一个传统 baseline 或重投影验证。

需要数据：

- 九视角图像或位移场。
- 标定/视角几何。
- mask。
- 参考重构切片或指标。
- 如果有：resolution phantom / MTF、坏视角记录、有效视场裁剪、view orientation 说明。
- 可公开边界。

风险：

- 如果几何不给，先做图像/位移可视化和数据接口。
- 如果真实数据暂时不能公开，论文图用 synthetic / open-source，真实数据只做内部验证。

### B 路线：师兄更需要 PIV-BOST

建议题目：

基于 BOST 折射率场的 PIV 速度测量误差补偿与传播分析

第一周交付：

- 把 M3A 从向量场 toy 升级到粒子图像 toy。
- 明确补偿层级：图像层、向量层、误差传播层。
- 画补偿前后速度误差图。

需要数据：

- PIV 图像对或速度场。
- 同步 BOST 位移/折射率场。
- 时间间隔、像素到物理单位、laser sheet 位置。
- 师兄已有补偿公式或脚本接口。

风险：

- 如果没有 PIV 原始图，先做速度场误差传播。
- 如果 stereo-PIV 太复杂，本科阶段先做 2D PIV toy。

### C 路线：师兄更需要 4D BOST

建议题目：

面向少视角四维背景纹影层析的可观测时序容量选择与可靠性评价

第一周交付：

- 以已完成的六轴/8 种子结果作为会前证据，不再重复单因素扫描。
- 以已完成的 rank×noise×views×dynamics 交叉结果作为 operating-domain 证据，不再重复同一交互。
- 以已完成的 864 格 nested-LOFO selector 和特征消融作为容量控制证据，不再重复 pooled synthetic rank 选择。
- 根据师兄选择实现 leave-one-geometry-out、UQ/拒答、bias/sync/exposure-blur 或真实数据 failure-signature。
- 对齐 4D BOST 论文中的真实容量参数、held-out view、主频或显存/速度指标之一。

需要数据：

- 时序 BOST 样例或 synthetic 设置。
- 帧数、帧率、视角同步方式。
- 师兄希望扫的 rank / 分辨率 / 采样参数。
- 是否能看组内 4D BOST 的数据格式或伪代码。
- 公开的 Hyz617/TDBOST 是否可作为代码结构参考；若可参考，哪些模块、数据和参数不能写入公开本科材料。

风险：

- 不承诺完整 4D BOST 复现。
- 只做低秩时序先验、指标和可视化子模块。

## 会议必须得到的 8 个答案

1. 主线选 A/B/C 哪条，E 是否作为 A 的增强副线？
2. 一年内的最终论文题目是否允许包含 PIV-BOST 或 4D BOST？
3. 第一份真实数据最可能是哪类？
4. 数据能否写入本科论文和答辩 PPT？
5. baseline 要对齐哪篇论文或哪份组内代码？
6. 评价指标以重投影误差、L2/CC/SSIM、速度误差还是时序一致性为主？
7. 第一周要交付什么图：九视角总览、mask/ROI、重投影误差、resolution phantom，还是 view-quality report？
8. 下次沟通时间和验收标准是什么？

## 会后 7 天动作

### 如果定 A 路线

- Day 1: 填 `data_templates/bost_sample_manifest.json`。
- Day 2: 画九视角原图、mask、位移场总览。
- Day 3: 统一单位和坐标范围。
- Day 4: 做一个最小重投影验证。
- Day 5: 和 M0-M2 synthetic pipeline 对齐接口。
- Day 6: 写一页数据说明。
- Day 7: 发给师兄检查。

### 如果定 B 路线

- Day 1: 确认补偿层级。
- Day 2: 把 M3A 升级成粒子图像 toy 或真实速度场 loader。
- Day 3: 画补偿前后速度误差。
- Day 4: 扫折射位移估计噪声。
- Day 5: 写误差传播公式。
- Day 6: 做一页图。
- Day 7: 发给师兄检查。

### 如果定 C 路线

- Day 1: 用一页 brief 让师兄在 rank selector、bias/sampling、真实数据三项中定优先级。
- Day 2: 若选 selector，先实现 held-out reprojection + singular-energy elbow 两个无真值规则。
- Day 3: 用现有 80 环境结果生成 selector regret、failure case 与物理 trace 对照。
- Day 4: 把最小真实数据接入同一 metrics schema。
- Day 5: 对齐一个论文指标或图示。
- Day 6: 更新“为什么不完整复现”和 synthetic-to-real 边界。
- Day 7: 发给师兄检查。

### 如果接受 E 副线

- Day 1: 在 M1/M3B phantom 中加入多视角 time offset。
- Day 2: 增加 missing view、坏视角和随机遮挡设置。
- Day 3: 做直接重构、简单插值和轻量投影补全三个 baseline。
- Day 4: 画 time offset / missing view / noise 三维误差热图。
- Day 5: 写同步误差容忍范围和数据质量判据。
- Day 6: 整理一页“这组实验数据能不能可信重构”的自动报告样例。
- Day 7: 发给师兄判断是否贴组内真实痛点。

## 不要在会上说的话

- “我都可以，看师兄安排。”
- “我想做 AI。”
- “我还没想好，先学学。”
- “我想完整复现 4D BOST。”

更好的说法：

- “我现在有 A/B/C 三条可启动路线，也准备了 E 作为 NeRIF/BOST 主线的误差分析增强副线，希望师兄帮我按组内需求和数据可得性选一条主线。”
- “我可以先用 synthetic 保底，但希望真实数据接口从第一周就按组内格式设计。”
- “我不承诺完整 4D BOST 复现；我已经完成跨形态无真值容量选择的 synthetic 底盘，下一步只做组内真实容量、geometry transfer、UQ/拒答或小样例验证。”
