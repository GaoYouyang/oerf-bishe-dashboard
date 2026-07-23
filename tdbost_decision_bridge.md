# 4D BOST / TDBOST 决策桥梁

生成日期：2026-07-10

渲染版入口：`tdbost_reproducibility_audit.html`；六轴实验入口：`m3b_six_axis_dashboard.html`。前者负责正文/代码/单次 M3B 对账，后者展示 8 种子六轴统计、失败签名和原始结果。

用途：把何远哲 4D BOST 主线从“很高级的一篇 TOG”拆成本科毕设可判断、可降级、可向师兄确认的研究路线。这个文件和 `tdbost_module_mapping.md` 配套：前者回答“要不要做、怎么收缩”，后者回答“公开仓库结构怎样安全借鉴”。

## 1. 当前核验状态

| 来源 | 已核验信息 | 对你有什么用 |
| --- | --- | --- |
| Crossref / DOI `10.1145/3809488` | 论文为 ACM Transactions on Graphics 45(5), 1-19；online date 2026-06-29；print date 2026-10-31 | 正式引用入口，避免只写 accepted 或只写网页标题 |
| 蔡伟伟 SJTU 中文主页 | “Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography...” 列在流体光学诊断及颗粒燃烧方向最新论文中 | 说明这不是外部随便找的邻居，而是 OERF 当前公开重点之一 |
| ACM 正式全文 / eReader | ACM 标注 Open access / CC BY 4.0；2026-07-10 已核 HTML 全文和 19 页 eReader 可读。正文确认 MM `XY-ZT` / `XZ-YT` / `YZ-XT`、3x128 Swish decoder、L=3 encoding、偏折与路径位移双积分、TV+L1/boundary loss、6x200 DC、coarse-to-fine 和 mixed precision | 可直接精读方法、实验、消融与 Appendix；用 `tdbost_reproducibility_audit.html` 看结构化摘要，不再停留在索引摘要 |
| 全文文件边界 | eReader 显示约 29.4 MB，但无会话 raw PDF / signed CDN 请求仍返回 403；ResearchGate 不作缓存来源 | 网页提供 ACM HTML/eReader/官方视频，明确“在线可读、未本地缓存”，不要伪造本地 PDF 状态 |
| Hyz617/TDBOST README / API | 2026-07-10 核到 main tree `3393ca7`，59 blobs / 2,287,454 bytes，最新提交 2026-04-04，无 release / LICENSE；README 声明用于 4D BOS tomography 并列出 configs、dataloader、TDmodel、render、run.py | 适合学习工程组织和参数接口；README 的 open-source 文案不能替代许可 |
| 代码健康检查 | `run.py` 硬编码 `cuda:3`；投影扩展是 Linux x86-64 `.so`；trainer import 即读 `/home/PUBLIC_USER_REDACTED/...`；requirements 有重复 pyDOE、三套 OpenCV 和 `skimage==0.0`；fuel test 路径与 train 全重叠 | 公开快照不能写成已验证的一键运行；只能结构级学习并做 clean-room baseline |
| 正文-vs-code 对账 | 正文出现 components 40/40/40 与默认 `R=30,F=20` 两种描述；Appendix A 为 3x128 decoder，公开 config 为 3 层、width 200；正文说 test view 不参与重建，fuel manifest 却与 train 路径重叠 | 把这些变成找何远哲确认的版本问题，不自行解释，也不拿公开 config 冒充正式实验配置 |
| TensoRF / Tensor4D / K-Planes / HexPlane / TiNeuVox / D-TensoRF | 已缓存 6 篇 arXiv PDF，正式 DOI 覆盖 ECCV/CVPR/SIGGRAPH Asia；这些论文不做 BOST baseline，但解释了 tensor factorization、plane-factor representation、time-aware voxels 和 dynamic-scene 4D 表示 | 用作 4D BOST 的方法语言支撑：为什么要扫 rank、memory、runtime、temporal consistency，以及为什么不能把 radiance-field 结果直接等同折射率场重构 |
| DMD / dynamic tomography / low-rank sparse reconstruction | 新增 Schmid DMD、Tu DMD theory、sparsity-promoting DMD、dynamic X-ray motion model、dimension-reduction Kalman filter、low-rank+sparse dynamic MRI 和 Robust PCA；其中 5 篇 arXiv PDF 已缓存，Schmid/Otazo 只放 DOI/出版社入口 | 用作 4D BOST 评估语言支撑：temporal mode、coherent structure、rank selection、random-noise vs system-bias 区分、low-rank + sparse decomposition 和 dynamic inverse-problem baseline |
| 本地 M3B demo | 2026-07-10 确定性重跑：rank 3 mean relative L2 `0.347137`，相对逐帧 baseline `0.365743` 改善约 5.1%；temporal smoothness `0.278849 -> 0.176693`，但 centroid trace 仍有共同系统偏差 | 已有可展示的本科安全版本；下一步应扫 noise/view/frame/bias，不必一上来复现 TOG |
| M3B 六轴多种子实验 | 8 paired seeds、448 method rows；rank 3 默认 global relative L2 改善 `3.90%`、paper-style squared L2 `7.65%`、temporal-gradient `44.68%`、held-out deflection `9.23%`，但 mass trace 恶化 `0.94%`，noise=0 时场 L2 恶化 `0.56%` | 已把“低秩去抖”升级成有适用域和失败边界的证据；可直接拿 `m3b_six_axis_dashboard.html` 找师兄决定 interaction scan 或真实数据迁移 |
| M3B 跨形态无真值 selector | 4 morphology×3 dynamics×4 noise×3 views×6 seeds；864 个观测格、6,048 个候选、full-rank 拒绝平滑与 nested-LOFO。固定 rank 3 mean/p95 regret 为 `1.561%/5.635%`；多特征 no-held-out 为 `0.252%/1.267%`，with-held-out 为 `0.210%/1.054%` | 已把 oracle rank 问题转成测试时不看 field truth 的容量选择；直接 held-out minimum 失败，说明 innovation 应放在 support/spectrum/temporal 证据融合、UQ 与拒答，而不是单一 residual 或 ridge |

## 2. 总判断

4D BOST 很贴何远哲，也最亮眼，但它不适合作为本科毕设的第一主线。

更稳的策略是：

1. 主线仍定在 NeRIF / BOST 鲁棒性与真实数据接口。
2. 如果师兄明确希望贴 4D BOST，则只接一个子问题：低秩时序先验、rank/帧数/噪声扫描、temporal consistency 指标、数据质量报告或可视化工具。
3. 不承诺完整复现 ACM TOG，不承诺复刻 TDBOST 训练和 ray tracing 加速模块。

这样写并不显得保守，反而更像课题组真正需要的本科贡献：先把数据、指标和可复现实验做稳，再看能否接真实 4D 数据。

## 3. 技术拆解

| 层级 | TOG / TDBOST 可能涉及 | 本科可做版本 | 不该承诺 |
| --- | --- | --- | --- |
| 数据对象 | `X-Y-Z-T` 折射率/密度时空体、多个视角、多个时间帧 | `volume[x,y,z,t]` synthetic phantom；manifest 记录 `case/view/frame/mask/calibration/deflection` | 真实高速实验全量数据处理 |
| 观测模型 | 多视角 BOST deflection / ray tracing / background displacement | 简化 parallel-ray 或 pinhole projection，外加可控噪声和坏视角 | 完整可微 ray tracing 和真实内窥几何 |
| 表示方法 | MM 的 `XY-ZT`、`XZ-YT`、`YZ-XT` plane pairs，经 Hadamard product、F-dimensional projection 与 3x128 Swish decoder 输出折射率差 | SVD/CP/Tucker 风格低秩 toy；比较 rank、memory、runtime、误差和平滑性 | 复现论文中的全部 tensor factorization 细节 |
| 畸变校正 | forward 同时积分 deflection angle 与 ray-path displacement；6x200 DC MLP 在 `Nc=6000` 后启用并交替优化 | 做 forward-model error / displacement bias toy，观察低秩先验无法修正系统误差 | 真实畸变校正网络 |
| 指标 | 正文用 squared-norm-ratio L2、SSIM、test-view PSNR、frequency、speed/memory；100G0 显示 DC 明显增益 | M3B 非平方 relative L2、temporal smoothness、centroid trajectory、rank-memory-time、held-out reprojection 与频谱 | 把 M3B 0.3471 与正文 0.0018 直接比较 |
| 产物 | 高速高保真 4D 流场重构算法 | 数据接口、rank scan、可视化、质量报告、可复现 toy | 完整高保真实验系统 |

## 4. 最适合本科的 5 个子题

| 排名 | 子题 | 可行性 | 新意 | 风险 | 数据需求 | 给师兄的问法 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4D BOST 低秩时序先验的 rank / noise / frame-count 扫描 | 高 | 中 | 低 | synthetic 3D+T phantom 即可 | “如果我只扫 rank、噪声和帧数，能否对齐 4D BOST 论文里的一个子问题？” |
| 2 | 4D BOST temporal consistency 指标与自动报告 | 高 | 中 | 低 | synthetic 或少量真实 reconstruction slice | “组里判断 4D 重构好坏时，除了逐帧误差还看哪些时序指标？” |
| 3 | BOST/4D-BOST 数据 manifest 与质量检查工具 | 高 | 中 | 中 | 需要师兄给字段约定或一小份样例 | “我能不能先做一个 view/frame/mask/calibration/deflection 的数据检查器？” |
| 4 | Forward-model error 对低秩时序先验的影响 | 中 | 中高 | 中 | synthetic projection + bias/noise | “4D BOST 真实误差更来自噪声、坏视角、同步误差还是 ray model？” |
| 5 | TDBOST-like 模块结构的自写最小 PyTorch baseline | 中 | 高 | 高 | GPU、更多时间、师兄确认 | “是否有必要写一个小 PyTorch tensor baseline，还是 M3B toy 和报告就够？” |

## 5. 和现有 M3B demo 的升级路线

现有 M3B 已经证明：低秩先验能减少逐帧抖动，但 rank 太低会过度平滑，且系统性几何偏差不会自动消失。

第一轮计划中的三类图已经由 8 种子六轴实验完成：

1. 六轴 OFAT 总览：rank/noise/frame/view/bias/dynamics 的配对改善与 95% CI。
2. rank 多种子稳定性：field、temporal gradient、held-out deflection 三指标。
3. bias/dynamics failure signature：系统偏差绝对误差与动力学能量比。

rank×noise×views×dynamics 交叉实验已经完成，后续跨形态 selector 也已把 oracle rank 转成测试时不看 field truth 的选择规则：864 个观测格中，无 held-out 多特征模型把 mean regret 压到 0.252%，带 held-out 为 0.210%，并明确否定“直接最小化 held-out residual”。下一轮不再重复 pooled synthetic selector，而应做 leave-one-geometry-out、bias/sync/exposure-blur、UQ/拒答，或直接把同一 health report 接到一小段真实数据。

## 5.1 4D 表示法邻居怎么用

新增的 TensoRF、Tensor4D、K-Planes、HexPlane、TiNeuVox 和 D-TensoRF 应该只出现在 related work / 方法动机里，不作为 BOST 数值 baseline。它们的价值是给出一套可复用的问题语言：

- **低秩表示**：TensoRF / D-TensoRF 说明高维场可以拆成紧凑 tensor factors；对应 M3B 的 rank scan。
- **平面分解**：K-Planes / HexPlane / Tensor4D 说明 4D spacetime 可以用多个 2D feature planes 表示；对应 4D BOST 正文中的 `XY-ZT`、`XZ-YT`、`YZ-XT` MM plane-pair 表述。
- **显式体素与时间特征**：TiNeuVox 说明逐帧体素会遇到显存和训练速度问题；对应 M3B 的 memory/runtime table。
- **边界**：这些论文重建的是 radiance / appearance / rendering field，不是折射率场；论文中只能写“表示法邻居”，不能写成物理等价 baseline。

## 5.2 时序指标邻居怎么用

新增的 DMD、dynamic tomography、Robust PCA 和 low-rank+sparse dynamic MRI 不用于替代 4D BOST，也不用于声称已经有 BOST 数据 baseline。它们的价值更像“评价工具箱”：

- **DMD / sparsity-promoting DMD**：从 4D 重构结果里抽时间模态、主频、低维动力学和 coherent structures；适合做 reconstruction 之后的 temporal diagnostic，而不是直接替代重构器。
- **Dynamic X-ray tomography**：motion model 和 dimension-reduction Kalman filter 说明动态层析可以把时间演化当成状态估计问题；适合启发 M3B 的 frame-count、rank 和 temporal prior 扫描。
- **Robust PCA / low-rank+sparse**：把背景/慢变结构和突发扰动拆开；适合问 4D BOST 中哪些结构应由低秩张量表示，哪些可能是火焰/激波/噪声的 sparse transient。
- **本科可做落点**：在 M3B synthetic phantom 上加一个 `temporal_metrics.py`，输出 DMD mode energy、dominant frequency、low-rank residual、sparse residual ratio、temporal smoothness 和 centroid trajectory error。

最重要的边界：这些文献来自流体 DMD、医学/工业动态层析和矩阵分解，不是 BOST 本身。开题里应写成“4D BOST 评估和低秩时序先验的数学背景”，不能写成“同类 BOST baseline”。

## 6. 会前一句话版本

> 4D BOST 我不打算一上来完整复现 TOG。我已经把它拆成低秩时序先验与无真值容量选择：完成六轴、交叉工作域和四类 synthetic morphology 的 nested-LOFO selector。下一步想请您判断，是把这套 support/spectrum/temporal 选择规则接到一小段真实 4D BOST 数据，还是先做 geometry/bias/UQ。

## 7. 来源入口

- ACM DOI: https://doi.org/10.1145/3809488
- ACM landing page: https://dl.acm.org/doi/10.1145/3809488
- ACM HTML full text: https://dl.acm.org/doi/full/10.1145/3809488
- ACM eReader: https://dl.acm.org/doi/epdf/10.1145/3809488
- ACM official supplement video: https://dl.acm.org/doi/suppl/10.1145/3809488/suppl_file/tog-25-0039-File003.mp4
- 蔡伟伟 SJTU 中文主页: https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html
- ResearchGate 条目只作为发现线索，不作为缓存或正式引用依据。
- TDBOST GitHub: https://github.com/Hyz617/TDBOST
- Crossref DOI API: https://api.crossref.org/works/10.1145/3809488
- TensoRF: DOI `10.1007/978-3-031-19824-3_20`, arXiv `2203.09517`
- Tensor4D: DOI `10.1109/CVPR52729.2023.01596`, arXiv `2211.11610`
- K-Planes: DOI `10.1109/CVPR52729.2023.01201`, arXiv `2301.10241`
- HexPlane: DOI `10.1109/CVPR52729.2023.00021`, arXiv `2301.09632`
- TiNeuVox: DOI `10.1145/3550469.3555383`, arXiv `2205.15285`
- D-TensoRF: arXiv `2212.02375`
- Dynamic mode decomposition: DOI `10.1017/S0022112010001217`
- On dynamic mode decomposition: DOI `10.3934/jcd.2014.1.391`, arXiv `1312.0041`
- Sparsity-promoting dynamic mode decomposition: DOI `10.1063/1.4863670`, arXiv `1309.4165`
- Dynamic X-ray tomography with motion models: DOI `10.1088/1361-6420/aa99cf`, arXiv `1705.06079`
- Dimension-reduction Kalman filter for dynamic X-ray tomography: DOI `10.1109/TCI.2019.2896527`, arXiv `1805.00871`
- Low-rank plus sparse dynamic MRI: DOI `10.1002/mrm.25240`
- Robust PCA: DOI `10.1145/1970392.1970395`, arXiv `0912.3599`
