# 给何远哲师兄的快速审核包

生成日期：2026-07-10

用途：这份文件不是完整调研报告，而是给何远哲师兄快速判断“这个本科毕设方向是否贴组里需求”的入口。建议先看本页，再看首页和 3 个 demo 结果图。

公开网页：https://gaoyouyang.github.io/oerf-bishe-dashboard/

GitHub 仓库：https://github.com/GaoYouyang/oerf-bishe-dashboard

2026-07-10 补充入口：

- `he_yuanzhe_sync_dashboard.html`：何远哲同步主线总页，包含论文矩阵、路线排序、三个月计划、VPN 私有库边界和向师兄提问清单。
- 论文库当前为 604 篇入口、192 篇已验证开放 PDF；已覆盖 DeepONet、FNO、PINO、NIO、RecFNO 等算子学习入口，每篇有渲染详情页，每个本地公开 PDF 有术语/页码导读。
- `operator_learning_bost_bridge.html`：T16 算子学习 × BOST 专项页，给出 physics lift + residual 3D FNO、DeepONet 对照、OOD split、12 周路线和停止条件。
- `tdbost_reproducibility_audit.html`：4D BOST 的 ACM HTML/eReader 正式全文已精读，方法、实验、显存、DC 消融和 paper-vs-code 差异均已结构化展示；`m3b_six_axis_dashboard.html` 给出六轴筛查，`m3b_interaction_dashboard.html` 进一步给出 80 个环境、8 种子、3,200 组配对比较和固定-rank regret。raw PDF 尚未本地缓存。
- PIV-BOST、stereo PIV-BOST 等受限全文只进本机 `private_library/`；公开页只放合法开放文件、出版社入口和自己的分析。

## 1. 我希望师兄帮忙判断的 8 件事

1. **主线是否选对**：本科毕设是否应以 NeRIF/BOST 鲁棒性为主线，而不是一开始直接做 PIV-BOST 或完整 4D BOST？
2. **数据是否可给**：是否能给一份不涉密的最小 BOST / PIV-BOST / 4D BOST 样例数据，哪怕只有位移场、几何和参考切片？
3. **baseline 该对齐谁**：组里更认可 voxel/SIRT/Landweber/UBOST，还是 NeRIF 论文代码中的 baseline？
4. **评价指标该怎么写**：重投影误差、L2/SSIM/CC、速度误差、时序一致性，哪几项最符合组里口径？
5. **真实数据质量控制是否有用**：view orientation、mask/ROI、spatial resolution、bad view、calibration/forward-model error 是否是组里真实痛点？
6. **位移/光流质量控制是否值得做**：组里是否需要 OpenPIV/Farneback/DIS/RAFT-PIV 对照、bad-region mask、confidence、runtime 和 view/frame health score？
7. **第一月交付什么最有用**：数据 loader、可视化、参数扫描、真实数据接口、几何/质量报告、位移质量控制、PIV 补偿 toy、还是 4D rank scan？
8. **算子学习如何定义**：希望做 projection-to-volume inverse operator，还是 3D/4D evolution operator？组里是否已有 paired dataset、推荐模型或固定切分？

## 2. 我当前的主线判断

首选题目：

> 面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

选择理由：

- 直接贴何远哲的 NeRIF / BOST 方向。
- 可在合成数据上独立启动，不完全依赖真实数据。
- 真实数据一旦可给，可以自然升级为真实 BOST 数据迁移。
- PIV-BOST 和 4D BOST 都能作为第二阶段分支，而不必一开始承诺完整复现。
- 本科生可交付内容清晰：baseline、参数扫描、误差图、重投影验证、数据接口，以及 view/mask/resolution 质量报告。

## 3. 已经跑通的最小 demo

| Demo | 作用 | 当前结论 | 师兄需要帮忙判断 |
| --- | --- | --- | --- |
| M0 2D BOST / coordinate-field toy | 证明最小重构闭环能跑 | 9 视角下 coordinate-field 相对 baseline 有小幅改善 | 这个 toy 的 forward model 是否足够作为第一周入口？ |
| M1 3D-stack sparse-view toy | 把问题推到三维栈 | 5 视角下坐标正则化有帮助，视角变多时传统 baseline 更强 | 真实 BOST 是否也更像少视角/含噪场景？ |
| M2 robustness scan | 扫视角数、噪声、容量 | 3/5 视角更适合隐式/坐标先验，7/9 视角传统法可能更稳 | 应该重点扫哪些变量？ |
| M3A PIV-BOST compensation toy | 速度误差传播最小例子 | 补偿可降低 RMSE，但局部误差仍受折射估计噪声影响 | PIV-BOST 应该做图像层还是速度场层？ |
| M3B 4D low-rank toy | 4D BOST 子问题 | 六轴与交叉实验显示：rank 3 是 regret 最低的固定默认值，但仅在 20/80 环境逐格最优；19/80 环境 field 负收益，field 与 mass 可分叉 | 下一轮应做无真值 rank selector、bias/sampling 阈值，还是直接接真实数据？ |

## 4. 三条可定题路线

### T16. 物理约束神经算子与 BOST 三维重建

适合条件：师兄明确支持算子模型设计，允许批量生成 synthetic pair 或能提供小规模 paired data。

第一月交付：2D operator sanity check、24^3/32^3 SIRT lift、3D U-Net/FNO 对照、phantom-family OOD split、M3B-informed view/noise condition cells 和 held-out/积分量审计。静态模型未跑稳前，不进入 temporal operator。

最希望师兄确认：inverse operator 还是 evolution operator；输入/输出物理量；组内 baseline；可用几何与 paired data；是否接受 residual FNO 作为主模型；真实 BOST 必须保真的积分量/事件量；3/9 views 与噪声极端整格留作 OOD 是否合理。

### A. NeRIF / BOST 鲁棒性主线

适合条件：师兄认可本科阶段先做可复现算法和系统实验，真实数据可有可无。

第一月交付：

- 合成 BOST 数据生成器。
- 传统 baseline。
- NeRIF-style coordinate representation。
- view/noise/capacity/sampling 扫描。
- view orientation、mask/ROI、resolution phantom 和 forward-model error 小实验。
- 重构切片、重投影误差和指标表。

最希望师兄给：

- 一份真实 BOST 位移场或九视角原图。
- 相机几何、mask、单位说明。
- 坏视角/有效视场裁剪、resolution/MTF 或参考评价口径。
- 组内认可的 baseline 或参考结果。

### B. PIV-BOST 折射补偿升级

适合条件：组里当前更需要 PIV-BOST 数据处理或误差传播工具。

第一月交付：

- 从 M3A 升级到粒子图像或真实速度场 loader。
- 补偿前后速度误差图。
- 折射位移噪声敏感性分析。
- 明确“图像层/互相关层/速度场层”补偿边界。

最希望师兄给：

- 同步 PIV-BOST 最小样例。
- PIV 时间间隔、像素尺寸、laser sheet 位置。
- 已有补偿公式或脚本接口。

### C. 4D BOST 低秩时序子问题

适合条件：师兄希望本科生先做 4D 论文的轻量子模块，而不是完整复现。

第一月交付：

- 已完成 rank/noise/frames/views/bias/dynamics 六轴 OFAT 与 8 个配对随机种子。
- 已完成 rank×noise×views×dynamics 交叉验证：80 个环境、3,200 组配对比较、Student-t 95% CI 和固定-rank regret。
- 下一轮从无真值 rank selector、bias magnitude、sampling rate/exposure blur 或真实数据 failure signature 中选一项。
- 用真实或组内最小样例核验 synthetic failure signature 是否成立。
- 对齐 4D BOST 论文中的 held-out view、时序频率或显存/速度指标之一。

最希望师兄给：

- 4D 数据格式说明或伪代码。
- 组里最关心的 rank、分辨率、帧数、噪声范围。
- 哪些图可公开、哪些只内部使用。

### F. BOS/PIV-BOST 位移估计与质量控制

适合条件：组里最需要的是把真实图像、位移场、坏视角、mask 和置信度先处理稳定，而不是马上换一个重构网络。

第一月交付：

- synthetic background / particle image pair。
- OpenPIV、Farneback、DIS/TV-L1 或 RAFT-lite 的位移估计对照。
- bad-region mask、confidence、runtime 和 photometric residual。
- 把位移误差传到 M3A 速度补偿或 M3C 重构质量报告。

最希望师兄给：

- 一小段 reference/deformed background image 或 PIV raw image pair。
- 组内当前位移提取方法或软件流程。
- 已有 bad-view/bad-frame 筛选规则。
- 希望报告的字段：位移场、mask、confidence、quality score、runtime、还是 HTML report。

## 5. 我建议师兄直接回复的格式

可以只回复下面 6 行：

```text
1. 主线建议：A / B / C / E / F / 其他
2. 第一月最希望我交付：
3. 最小可给数据：
4. baseline 建议：
5. 评价指标建议：
6. 真实数据质量控制最痛的点：
7. 是否需要位移/光流质量控制：
8. 不能公开的边界：
```

## 6. 我会主动避免的风险

- 不会把公开网页写成“组内已经确定的任务”，所有判断都标为个人预研和待确认。
- 不会把 4D BOST 完整复现作为本科保底承诺。
- 没有真实数据时，不会假装结果代表真实火焰实验，只会写 synthetic/open data 边界。
- 如果真实数据涉及保密，公开网页和开题材料只放合成/公开数据图。
- 如果师兄建议改方向，会保留当前 NeRIF/BOST pipeline 作为基础，不把前期工作浪费掉。

## 7. 师兄如果只看 5 个文件

1. `he_review_feedback_pack.md`：本页，快速审核入口。
2. `top_topic_proposal_cards.md`：A/B/C/E/F 选题卡。
3. `he_meeting_decision_pack.md`：15 分钟会前汇报结构和会后动作。
4. `displacement_qc_route_f.md`：如果师兄觉得先做图像前处理/位移质量控制，这页是详细执行方案。
5. `m3b_interaction_brief.md`：如果师兄倾向 4D BOST，这页集中给出交叉证据、解释边界和下一轮选择。

## 8. 来源边界

本工作台基于公开来源整理，包括蔡伟伟交大主页、OERF 官网和 GitHub 源码、出版社/DOI 页面、arXiv 开放版本。所有涉及“组里是否需要”“真实数据是否可给”“代码是否能看”的内容都是待师兄确认的问题，不是事实断言。
