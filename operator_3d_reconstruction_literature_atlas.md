# 算子学习 × 三维重建文献图谱

更新：2026-07-10
用途：不是“读得越多越好”的书单，而是给 T16 / SC-DBNO 每个设计选择找到前人依据、必须对照的 baseline 和不能越界的主张。

## A. OERF / 何远哲物理主线：先定义问题

| 文献 | 直接价值 | 阅读时提取 |
| --- | --- | --- |
| [He et al., NeRIF, Physics of Fluids 2025](https://doi.org/10.1063/5.0250899) | BOST 连续折射率场、differentiable forward、边界与重投影的核心起点。 | 输入观测、ray model、坐标网络、loss、视角数、实验几何。 |
| [He et al., 4D BOST, ACM TOG 2026](https://doi.org/10.1145/3809488) | OERF 最新 4D 时空重建；张量分解、位移校正网络、显存与实验约束。 | 9-view 合成、500-frame 实验、tensor planes、DC network、失败/配置差异。 |
| [Zheng et al., simultaneous PIV-BOST, EIF 2025](https://doi.org/10.1007/s00348-025-04093-y) | 说明折射率重建不是“好看三维图”，可进入 PIV 速度误差补偿。 | 同步几何、误差传播、补偿前后指标、输出物理量。 |
| [Bo et al., GRU-BOST, Optics Express 2023](https://doi.org/10.1364/OE.505992) | OERF 邻近的时序 BOST 网络 baseline。 | GRU 输入输出、时序窗口、对照和速度。 |
| [Huang et al., limited-projection volumetric tomography, AST 2020](https://doi.org/10.1016/j.ast.2020.106123) | 蔡伟伟组“有限投影 + 深度学习 + 三维火焰”的直接方法前史。 | 投影限制、训练数据生成、跨工况与实验验证。 |
| [Huang et al., 3D flame evolution, JFM 2019](https://doi.org/10.1017/jfm.2019.545) | 组内 2D history -> 3D evolution 的时序学习前史。 | 预测任务与逆算子任务的差异、在线速度与时序指标。 |

**边界：** NeRIF 是逐实例 neural field，SC-DBNO 是跨样本 neural operator 候选；4D BOST 的 tensor rank 也不等于 FNO Fourier modes。

## B. 神经算子根架构：决定怎样表示函数到函数映射

| 文献 | 角色 | 对 T16 的启发 |
| --- | --- | --- |
| [DeepONet, Nature Machine Intelligence 2021](https://doi.org/10.1038/s42256-021-00302-5) | 必读根架构 | branch/trunk 适合“观测函数 + 坐标查询”，可作为第二模型族。 |
| [FNO, ICLR 2021](https://arxiv.org/abs/2010.08895) | 当前主干 | 规则 3D 网格的全局 spectral mixing；必须测试高频前缘失败。 |
| [Neural Operator, JMLR 2023](https://www.jmlr.org/papers/v24/21-1524.html) | 理论与统一语言 | 解释 discretization-invariance 的条件，避免把普通 CNN 误称 neural operator。 |
| [U-NO, TMLR 2023](https://openreview.net/forum?id=j3oQF9coJd) | 多尺度/显存对照 | U-shaped operator，适合 3D 局部+全局结构；可做第二阶段架构对照。 |
| [F-FNO, ICLR 2023](https://openreview.net/forum?id=tmIiMPl4IPa) | 高效 spectral 对照 | 可分离 spectral layer 与改进残差；适合显存受限时测试。 |
| [CNO, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f3c1951b34f7f55ffaecada7fde6bd5a-Abstract-Conference.html) | matched CNN/operator 对照 | 连续函数空间一致的卷积算子；比随意 U-Net 更公平。 |
| [MG-TFNO, TMLR 2024](https://openreview.net/forum?id=AWiDlO63bH) | 高分辨率升级 | multi-grid domain decomposition + tensorized spectral weights；不是本科首个模型。 |
| [Multiwavelet Neural Operator, NeurIPS 2021](https://openreview.net/forum?id=LZDiWaC9CGL) | 高频/多尺度候选 | 若 thin-front OOD 失败，可测试 wavelet，而非只加 Fourier modes。 |
| [Riesz Neural Operator, ICLR 2026](https://openreview.net/forum?id=Vjw7q1quNt) | 局部导数候选 | 把全局谱与局部方向导数结合，对薄前缘和局部非平稳性有针对性。 |

**本科优先级：** FNO + matched U-Net/CNO 足够形成第一轮；U-NO/MG-TFNO/RNO 只有在明确失败模式后再加入。

## C. 逆算子与几何：观测域和三维域并不在同一网格

| 文献 | 角色 | 对 T16 的启发 |
| --- | --- | --- |
| [Neural Inverse Operators, ICML 2023](https://openreview.net/forum?id=S4fEjmWg4X) | 必读 | 正式区分 operator-to-function inverse map；为 DeepONet/FNO 组合提供框架。 |
| [Ultrasound Tomography Inversion, MIDL 2023](https://openreview.net/forum?id=tSokLyjvW5) | 直接邻域 | TOF 观测到 sound-speed field 的一次前向逆算子，与 BOST 类比最清楚。 |
| [Learned Primal-Dual, IEEE TMI 2018](https://doi.org/10.1109/TMI.2018.2799231) | forward-aware baseline | 把 forward/adjoint 放进 unrolled network；提醒我们不能只在体素域后处理。 |
| [MoDL, IEEE TMI 2019](https://doi.org/10.1109/TMI.2018.2865356) | data-consistency baseline | 把学习先验与显式数值 data-consistency block 交替；是“物理求解可观测部分，网络补先验”的强架构前例。 |
| [Golub--Pereyra separable least squares, SIAM JNA 1973](https://doi.org/10.1137/0710036) | 数值基础 | 可分离模型中的线性系数可用伪逆/最小二乘先消元；T16 support-fit 是更简单的一维情形，不能把这步本身当原创算法。 |
| [Geo-FNO, JMLR 2023](https://www.jmlr.org/papers/v24/23-0064.html) | 变化几何候选 | 用 learned deformation 处理一般几何。 |
| [GINO, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) | 变化几何候选 | graph operator 把 irregular observations 映到规则 latent FNO grid。 |
| [Geometric Operator Learning with OT, JMLR 2025](https://jmlr.org/papers/v26/25-1380.html) | 高级几何 | instance-dependent geometry embedding；适合相机布局跨样本变化时参考。 |
| [Geometry-Aware Attenuation Learning, IEEE TMI 2024](https://doi.org/10.1109/TMI.2024.3473970) | 极关键邻域 | 2D encoder -> geometry backprojection -> 3D decoder；是 T16 输入层最直接的结构参考。 |

**关键选择：** 如果组内几何固定，先做显式 backprojection + 规则 3D operator；若几何变化，再证明 GINO/Geo-FNO 的额外复杂度有必要。

## D. 少视角三维重建：怎样利用跨视角信息

| 文献 | 角色 | 对 T16 的启发 |
| --- | --- | --- |
| [NAF, MICCAI 2022](https://arxiv.org/abs/2209.14540) | neural-field 邻域 | 无外部训练数据的坐标场 + differentiable projection；与 NeRIF 作方法类比。 |
| [SNAF, 2022 preprint](https://arxiv.org/abs/2211.17048) | view augmentation | 少视角时训练视角增强；启发 T16 view dropout/curriculum。 |
| [C2RV, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_C2RV_Cross-Regional_and_Cross-View_Learning_for_Sparse-View_CBCT_Reconstruction_CVPR_2024_paper.html) | 极关键邻域 | 明确指出不同 view 不应等权，并用 scale-view cross-attention 自适应聚合。 |
| [GSA-INF, ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Huang_Generalizable_Structure-Aware_INF_Biplanar-View_CT_Reconstruction_via_Disentangled_Implicit_Neural_ACCV_2024_paper.html) | 1/2-view prior | triplane volume prior + INR decoder；启发“population prior + per-instance field”。 |
| [Dynamic CT INR, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Reed_Dynamic_CT_Reconstruction_From_Limited_Views_With_Implicit_Neural_Representations_ICCV_2021_paper.html) | 4D 邻域 | template + motion field + self-supervised sinogram consistency；对应未来 4D BOST。 |
| [FACT, 2023 preprint](https://arxiv.org/abs/2312.01689) | warm-start/meta-learning | 元训练和 hash regularization 加速每个新体的 neural-field 优化。 |

**不要照搬 CT：** X-ray attenuation、BOS deflection 和折射率梯度的 forward physics 不同；可以借结构，不能借结论。

## E. 自监督与无真值可靠性：真实 BOST 的核心难点

| 文献 | 角色 | 对 T16 的启发 |
| --- | --- | --- |
| [Noise2Inverse, IEEE TCI 2020](https://doi.org/10.1109/TCI.2020.3019647) | 必读 | 把测量拆分成统计独立重构，在无 clean target 下训练；启发 view split。 |
| [Deep Null Space Learning, Inverse Problems 2019](https://doi.org/10.1088/1361-6420/aaf14a) | v2 数学核心 | 用 `Id-A^+A` 把 neural correction 投影到 forward null space，保留数据一致性；同时提醒精确伪逆在 3D BOST 中可能过贵。 |
| [Siamese Cooperative Learning, IEEE TPAMI 2024](https://doi.org/10.1109/TPAMI.2024.3359087) | v2 自监督核心 | 在不完整测量下让 range-space 和 null-space 双网络协作，用 data/mutual consistency 训练；比通用 MoE 更贴合 SC-DBNO 分解。 |
| [Equivariant Splitting, ICLR 2026](https://openreview.net/forum?id=upMIVpe467) | 最新必读 | incomplete observation + splitting + equivariance；包含 sparse-view CT。 |
| [3D turbulent flow without ground truth, DCE 2026](https://doi.org/10.1017/dce.2026.10038) | 实验流场邻域 | 用未见传感平面和 physics loss 选模型，训练不需完整 3D 真值。 |
| [Physics-informed shadowgraph network, 2025](https://www.sciencedirect.com/science/article/pii/S0894177725001566) | 光学诊断邻域 | 端到端自监督密度重建；要审计其 forward 假设和真实验证。 |

对 T16 最直接的操作不是“无监督”口号，而是：先用 support camera 求可观测的闭式融合；再让 query camera 只教受限修正或 support-nullspace 残差；测试时没有 query 就回退到 support-fit/UQ。

## F. 可靠性、专家路由与不确定性：把失败识别做成贡献

| 文献 | 角色 | 对 T16 的启发 |
| --- | --- | --- |
| [Spatial Mixture-of-Experts, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4c5e2bcbf21bdf40d75fddad0bd43dc9-Abstract-Conference.html) | 路由机制 | spatial gating、自监督 routing loss、expert damping；提醒防止专家塌缩。 |
| [Neural Experts, NeurIPS 2024](https://openreview.net/forum?id=wWguwYhpAY) | neural-field MoE | 局部专家 + gate、conditioning/pretraining；适合未来局部前缘专家。 |
| [MoE-POT, NeurIPS 2025](https://openreview.net/forum?id=PNgG4H3q9D) | operator MoE | router 按 PDE/dataset 激活专家；规模很大，只借路由思想。 |
| [Probabilistic Neural Operator, TMLR 2025](https://openreview.net/forum?id=gangoPXSRw) | 函数分布 | 输出函数空间分布；适合最终 uncertainty，但实现成本高。 |
| [LUNO, ICML 2025](https://openreview.net/forum?id=4Z04wVQ9FY) | 后验近似 | 把已训练 neural operator 线性化为 function-valued GP。 |
| [Approximate Bayesian Neural Operators, TMLR 2025](https://openreview.net/forum?id=6WvIkYsMA8) | 失败检测 | 官方结果强调 structured uncertainty 能识别 operator 失败样本。 |

本科最小实现先用 branch disagreement + deep ensemble；只有它能预测失败时，才值得升级 Bayesian/functional UQ。

## G. 稀疏流场重建与展示：证明不是只会做 CT 类比

| 文献 | 角色 | 对 T16 的启发 |
| --- | --- | --- |
| [RecFNO, IJTS 2024](https://doi.org/10.1016/j.ijthermalsci.2023.108619) | 必做 flow baseline | mask/Voronoi 等 sparse embedding + resolution-invariant field reconstruction。 |
| [Energy Transformer flow reconstruction, JCP 2025](https://doi.org/10.1016/j.jcp.2025.114148) | 强邻域 | 覆盖 2D vortex、真实 Schlieren、3D jet particle tracking；适合比较 sparse measurement operator。 |
| [Tensor-based flow reconstruction, JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/tensorbased-flow-reconstruction-from-optimally-located-sensor-measurements/E6F6DD6727238BC50C42919936E4D841) | 可解释 baseline | 保留多维 tensor structure，适合低秩/最优传感对照。 |
| [3D turbulent flow without ground truth, DCE 2026](https://doi.org/10.1017/dce.2026.10038) | 实验部署邻域 | 稀疏/noisy planar measurements、unseen plane validation、无完整真值。 |

## H. 可变传感器、任意几何与 operator-INR 桥

| 文献 | 组件依据 | T16 应提取什么 |
| --- | --- | --- |
| [VIDON, arXiv 2022](https://arxiv.org/abs/2205.11404) | random sensor count/location + permutation invariance | 相机数量与位置跨样本变化时的 set encoder；固定几何不需要先上。 |
| [GNOT, ICML 2023](https://proceedings.mlr.press/v202/hao23c.html) | irregular mesh、multiple inputs、geometric gating | 把 displacement、geometry、confidence 当异构函数输入；模型很大，只作后续对照。 |
| [CORAL, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html) | coordinate neural fields + operator learning | 任意空间采样/几何和 operator-to-NeRIF 连续 decoder 的直接桥梁。 |
| [Beyond Regular Grids, ICML 2024](https://proceedings.mlr.press/v235/lingsch24a.html) | direct spectral transforms on non-equispaced points | 作为 GINO/插值 latent grid 之外的任意点 spectral 对照。 |
| [Latent Neural Operator, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/39f6d5c2e310a5a629dcfc4d517aa0d1-Abstract-Conference.html) | Physics-Cross-Attention + arbitrary query | 大规模 ray/voxel token 压到 latent，再在任意坐标解码。 |
| [O-INR, ICML 2024](https://proceedings.mlr.press/v235/pal24a.html) | INR 的 operator-theoretic 重写 | 明确 population operator prior 与 per-instance coordinate field 的关系。 |

**边界：** 这些工作处理 PDE mesh、point cloud 或 general geometry。它们只提供输入表示和架构组件，不证明 BOST 反演精度。

## I. 局部高频、修正算子与可校准拒答

| 文献 | 组件依据 | T16 应提取什么 |
| --- | --- | --- |
| [Localized Integral and Differential Kernels, ICML 2024](https://arxiv.org/abs/2402.16845) | 全局 FNO 易过平滑；加入局部 integral/differential layers | thin-front 独立失败后做 global-only vs local/global matched ablation；论文百分比不可迁移。 |
| [DuFal, TMLR 2026](https://openreview.net/forum?id=2wAZjAtK16) | 极少视角 CBCT 的 global/local frequency 双路 | 薄前缘高频支路与 cross-attention fusion；X-ray forward 与 BOST 不同。 |
| [Neural Correction Operator, JCP 2026](https://arxiv.org/abs/2507.18875) | limited numerical reconstruction + learned correction | 对比 direct inverse operator 与 physics-lift correction；最接近 T16 组件级先例。 |
| [Calibrated UQ for Operator Learning, TMLR 2024](https://openreview.net/forum?id=cGpegxy12T) | finite-sample functional conformal calibration | 在独立 calibration set 上校准体场 coverage；不能在 test domain 现调阈值。 |
| [Conformalized-DeepONet, Physica D 2025](https://arxiv.org/abs/2402.15406) | split conformal operator intervals | 若最终需要 per-coordinate band，可作为 DeepONet/FNO 后处理对照。 |
| [SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) | reject option 与 risk-coverage | 把“拒答”单列为系统指标；fallback 的误差必须重新计算。 |
| [ClawNO, ICML 2024](https://proceedings.mlr.press/v235/liu24p.html) | hard conservation-law encoding | 只在后续输出速度/演化场且守恒变量明确时使用；折射率场不能随意设 divergence-free。 |

## J. 当前实验把文献问题压缩到哪里

1. Independent dual experts 已证明比共享双输出更有互补，但仍欠等参数单模型/ensemble 对照。
2. Exact support-null audit 证明有 38.626% 平均 oracle headroom，但 learned null corrector 总体为 -0.126%。
3. 一台 reserved query camera 的闭式幅度标定把均值转为 +0.746%，并在 15/15 seed-domain 单元为正；这是当前最该升尺度的候选。
4. M3B geometry audit 表明 uncertainty score 与 fallback 必须分别验证；不能把“高不确定度”直接等同“换 full rank/NeRIF 就会更好”。
5. 因此下一轮文献阅读优先围绕 query-view value of information、approximate null projection、correction operator 与 conformal selective prediction，而不是继续横向收集更多 FNO 变体。

## 读文献时统一记录的九个字段

1. 输入到底是 raw image、位移、投影、传感点还是粗重构？
2. 输出是 voxel field、coordinate field、future field 还是 parameter？
3. forward/adjoint 是否进入网络或 loss？
4. 是 per-instance optimization 还是 cross-sample operator？
5. 几何固定还是变化，网络怎样编码 geometry？
6. 训练是否使用完整 3D truth，测试时有什么可观测验收？
7. sparse-view、noise、family、resolution、geometry OOD 怎样切？
8. 与同参数 CNN、传统反演、neural field 是否公平比较？
9. 公开代码/数据/精确版本/硬件/运行时间是否足够复现？

## 推荐阅读顺序

1. NeRIF -> 4D BOST -> limited-projection OERF tomography；
2. FNO -> Neural Operator JMLR -> NIO；
3. Learned Primal-Dual -> MoDL -> Geometry-Aware Attenuation -> GINO；
4. Golub--Pereyra -> Deep Null Space Learning -> Siamese Cooperative -> Noise2Inverse -> Equivariant Splitting；
5. Spatial MoE -> LUNO/Approximate Bayesian NO；
6. 最后再读 U-NO/MG-TFNO/RNO 等架构升级，避免先陷入模型名称。
