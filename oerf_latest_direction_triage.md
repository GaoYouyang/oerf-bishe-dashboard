# OERF 最新论文方向分流表

生成日期：2026-07-09

用途：蔡伟伟老师主页的“最新论文”很强，但不是每条都适合直接变成你的本科毕设。这个文件把最新公开方向按“和何远哲主线距离”分流，帮助你和师兄沟通时保持主线稳定。

## 1. 总判断

你现在最应该守住的主轴仍然是：

**BOST / NeRIF / PIV-BOST / 4D BOST：光学诊断问题 + 计算重构 + 真实实验数据接口。**

其他方向的作用分三类：

1. **可作为主线升级**：stereo PIV-BOST、4D BOST。
2. **可作为方法邻居**：U-Net flame-front detection、holographic 3D tracking、4D LII。
3. **只作 OERF 背景**：铁颗粒燃烧、氨共燃、碳烟诊断、金属颗粒综述。

## 2. 最新方向分流

| 主页最新条目 | 当前核验状态 | 和你主线的关系 | 本科建议 | 需要问师兄 |
| --- | --- | --- | --- | --- |
| Tensor Decomposition-Based 4D BOST | Crossref 核到 ACM TOG 45(5), 1-19，DOI `10.1145/3809488` | 何远哲主线第三层 | 六轴与交叉实验已完成；继续做无真值 rank selector、低秩时序/数据接口子问题，不完整复现 | 4D BOST 是否更需要 selector、bias/sampling 阈值、temporal metric 或真实数据清洗任务？ |
| Stereo-velocity PIV-BOST | Crossref 核到 Proceedings of the Combustion Institute 42, article 106175，DOI `10.1016/j.proci.2026.106175` | PIV-BOST 后续升级 | 不开局做 stereo；先做 2D PIV-BOST 或 image-layer toy | 组内 PIV-BOST 数据是 2D PIV 还是 stereo-PIV？ |
| U-Net-Based Method for Instantaneous Flame Front Detection | 目前只在蔡伟伟 SJTU 主页显示 2026 accepted，未检到 DOI/卷页；已缓存外部开放方法邻居 `10.1007/s00348-024-03814-z` | AI segmentation 方法邻居 | 可作“自动标注/前处理工具”备选，不替代 BOST；正式写 OERF 条目时等待 DOI | 是否有火焰前沿标注数据？这个任务是否服务 BOST/CTC 后处理？ |
| 4D multi-kHz time-resolved LII | Crossref / Optica 核到 Applied Optics early posting，DOI `10.1364/AO.603424`；正式题名为 `4D multi-kHz time-resolved laser-induced incandescence for comprehensive soot characterization in unsteady flames`，不是早期线索里的 `turbulent flames` | 另一个 4D 光学诊断范式 | 只借“4D measurement / time-resolved diagnostics”语言，不转碳烟主线 | 4D BOST 和 4D LII 在数据组织、指标、可视化上是否有共用工具？ |
| Tomographic single-shot time-resolved LII | Crossref 核到 POCI 40, article 105262，DOI `10.1016/j.proci.2024.105262` | 4D LII 前史，获奖工作 | 背景材料，不建议转主线 | 开题是否需要用它说明 OERF 在 4D 光学诊断上的积累？ |
| Iron particle heat exchange / ammonia co-firing | 已入库；何远哲参与 heat-exchange 2025；ammonia co-firing DOI `10.1016/j.combustflame.2026.114780` | 真实反应流实验背景 | 只用于说明 OERF 数据来自真实燃烧/颗粒实验 | 如果师兄给的是颗粒燃烧数据，是否仍可做图像/重构工具而非燃烧机理？ |
| Robust 3D tracking via holographic spatio-temporal similarity | Crossref 核到 Combustion and Flame 287, article 114899，DOI `10.1016/j.combustflame.2026.114899` | 计算成像 + 3D tracking 邻居 | 可作远期图像处理备选；和 BOST 都需要三维场/轨迹可视化 | 是否有数字全息/颗粒追踪的数据接口任务？ |
| Computational flow visualization | Cell Reports Physical Science 2024；论文库已有开放 PDF | 最像“总方法论”背景 | 必读背景之一，用于解释 optical + computational methods 的总路线 | 开题 related work 是否应把它放在 OERF 总纲位置？ |

## 3. 你不该怎么写

不要写：

> 我可以做 OERF 的所有方向，包括 BOST、PIV、4D LII、全息、铁颗粒、U-Net、数据同化。

更好的写法是：

> 我的主线聚焦 BOST/NeRIF/PIV-BOST/4D BOST。OERF 的 4D LII、全息颗粒追踪、U-Net 火焰前沿和金属颗粒燃烧说明课题组有丰富的高维光学诊断数据，但它们在本课题中只作为方法邻居或应用背景；除非师兄明确转向，否则不改变主线。

## 4. 可转成选题的备选包

| 备选包 | 题目雏形 | 适合条件 | 风险 |
| --- | --- | --- | --- |
| Flame front segmentation tool | 基于 U-Net 的火焰前沿瞬态检测与标注一致性分析 | 师兄给火焰图像和标注；需要快速做图像分割工具 | 可能偏离何远哲 BOST 主线 |
| Holographic tracking helper | 基于时空相似性的燃烧颗粒三维轨迹质量评估工具 | 组里有全息颗粒图像/轨迹结果需要后处理 | 需要新数据格式和光学背景 |
| 4D diagnostics common report | 面向 4D BOST / 4D LII 的时序重构质量报告 | 师兄希望你做工具而非算法复现 | 贡献像工程工具，论文创新性要包装好 |
| Particle-combustion image analytics | 金属颗粒燃烧图像的多物理场测量结果整理与误差报告 | 师兄转给你颗粒燃烧数据 | 会变成燃烧实验数据分析，不再是 He 主线 |

## 5. 会前问题

1. 这些最新论文里，您觉得本科生最可能帮上的是真实 BOST 数据接口、PIV-BOST 补偿，还是火焰/颗粒图像前处理？
2. U-Net flame front detection 是否已有公开 DOI 或内部数据？它和 BOST/NeRIF 后处理有关吗？
3. 4D LII 和 4D BOST 是否共享时间序列可视化、质量指标或报告工具？
4. 如果我主线保持 NeRIF/BOST，是否需要在开题中保留铁颗粒/全息/碳烟方向作为 OERF 应用背景？

## 6. 来源入口

- 蔡伟伟 SJTU 中文主页: https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html
- 4D BOST DOI: https://doi.org/10.1145/3809488
- Stereo PIV-BOST DOI: https://doi.org/10.1016/j.proci.2026.106175
- 4D LII Applied Optics DOI: https://doi.org/10.1364/AO.603424
- Single-shot LII POCI DOI: https://doi.org/10.1016/j.proci.2024.105262
- Iron ammonia co-firing DOI: https://doi.org/10.1016/j.combustflame.2026.114780
- Holographic tracking DOI: https://doi.org/10.1016/j.combustflame.2026.114899
