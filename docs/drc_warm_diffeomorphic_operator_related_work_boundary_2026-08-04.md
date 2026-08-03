# DRC-Warm 与微分同胚算子学习：相关工作和可主张边界

更新：2026-08-04

## 先给结论

当前方法不能把“使用微分同胚、把不同坐标域映射到公共参考域”写成创新点。该一般机制在 Geo-FNO、DIMON、DNO、CT-FNO、DAFNO 和后续 diffeomorphic neural operator 工作中已有明确先例；“神经算子为三维光学求解器提供 warm start”也已被 2026 年 7 月的 EUV 电磁仿真工作直接覆盖。这不等于它们已经覆盖 BOST-specific、observation-only、exact refinement 与成本审计的完整组合。

当前尚可检验的窄贡献是：

> 面向多视角三维 BOST 逆问题，使用只读取部署可见观测与已知几何的坐标条件 warm initializer，在物理域中继续执行未修改的精确 `A/A^T` 与 CGLS/PCGLS 精化；只有逐单元 field、完整梯度、内部梯度和 observation 同精度时，才判断是否减少完整算子调用和真实资源成本。

这仍是待验证的研究假设，不是已经成立的创新结论。

## 当前方法到底是什么

当前 DRC-Warm 原型的作用链是：

1. 已知相机/射线几何和预注册坐标变化定义从物理坐标到参考坐标的映射。
2. 将 observation-derived cheap BP、支撑和几何描述搬到公共参考坐标。
3. 小型三维 CNN 在参考坐标预测 warm correction。
4. 先复合坐标，再对原始张量做一次 gather，减少重复重采样误差。
5. 把候选场送回物理坐标，使用精确物理算子和未修改 CGLS K1 精化。

因此，学习模块不是独立 PDE 求解器，也没有替换 forward model。它当前还是固定 `32 x 16 x 16` 网格上的三维 CNN；在没有跨分辨率一致性或离散无关性证据前，论文应称它为 `coordinate-conditioned 3D CNN warm initializer`，不能仅凭任务形式直接称为具有数学意义的通用 neural operator。

## 本次检索到的主要一级来源

检索截止日为 2026-08-04。范围包括公开可核对的论文/正式 proceedings 页面，关键词覆盖 diffeomorphic/common-reference-domain neural operator、geometry-aware operator、inverse operator、neural warm start、learned iterative reconstruction 与 BOST reconstruction。下面是直接邻居而非穷尽性排序，不能据此声称 global uniqueness。

| 工作 | 已覆盖的核心思想 | 与 DRC-Warm 的关键差别 |
|---|---|---|
| [Geo-FNO: Fourier Neural Operator with Learned Deformations for PDEs on General Geometries](https://arxiv.org/abs/2207.05209) | 学习平滑变形，把一般几何搬到规则潜在网格后使用 FNO | 目标是 PDE solution surrogate；当前方法是 BOST inverse warm start，之后仍调用精确成像算子和 Krylov 精化 |
| [DIMON: Learning Solution Operators of PDEs on a Diffeomorphic Family of Domains](https://arxiv.org/abs/2402.07250) / [Nature Computational Science 正式版本](https://www.nature.com/articles/s43588-024-00732-2) | 把不同物理域和输入函数运输到参考域，学习参考域中的 latent solution operator，再映回物理域 | 这是结构高度相似的直接邻居；当前差别只能来自逆问题、observation-only 输入、exact-physics refinement、调用账和失败门，不能来自“公共参考域”本身 |
| [Diffeomorphism Neural Operator](https://www.nature.com/articles/s42005-024-01911-3) | 将不同 PDE 域映射到 generic domain，在公共域学习跨形状/参数 operator | 同样否定“diffeomorphic common domain”首创；其目标是跨域 PDE 求解，当前目标是多视角 BOST 重建起点 |
| [Coordinate Transform Fourier Neural Operators](https://openreview.net/pdf?id=pMD7A77k3i) | 用坐标变换显式处理域形状和物理对称性，再应用 FNO | 已覆盖“坐标变换帮助对称与几何泛化”；当前方法必须靠 BOST-specific inverse refinement 和严格成本门区分 |
| [Domain Agnostic Fourier Neural Operators](https://proceedings.neurips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html) | 用平滑特征函数把几何信息编码进 Fourier operator，处理不规则甚至拓扑变化的域 | 它不是同一微分同胚实现，但会是几何泛化审稿中的强 neural-operator baseline |
| [CORAL: Operator Learning with Neural Fields](https://openreview.net/pdf?id=4jEjq5nhg1) | 用 coordinate-based neural fields 处理自由几何、自由采样和不同分辨率 | 提醒我们当前固定网格 CNN 尚未证明离散无关性；CORAL 属于未来外部比较邻居 |
| [Diffeomorphic Neural Operator Learning](https://arxiv.org/abs/2508.06690) | 学习到微分同胚群的 lift，通过群作用和复合表示流体演化 | 其重点是演化半群、守恒和 learned flow map；当前使用已知坐标变化处理 inverse warm start，不能借用其结构保持结论 |

## BOST 内部最直接的邻居

| 工作 | 已覆盖的核心思想 | 与当前问题的边界 |
|---|---|---|
| [Background-oriented Schlieren tomography using gated recurrent unit](https://doi.org/10.1364/OE.505992) | 利用多方向投影间的空间相关，以 GRU 直接重建；在数值甲烷燃烧与蜡烛火焰上报告精度和约 1.04 s/frame | 直接否定“首次用序列网络加速 BOST”一类表述；当前必须比较未修改 exact refinement、跨轨迹/坐标门和完整成本，而不能只比较单次网络推理 |
| [Neural refractive index field](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the) | 用隐式神经场替代 voxel representation，针对空间分辨率、离散误差、噪声和计算成本做 BOST 重建 | 这是何远哲师兄方向中的核心内部邻居；当前 warm initializer 不应声称取代 NeRIF，而应回答能否减少其或其他认可后端的初始化/精化成本 |
| [Tensor Decomposition-Based Four-dimensional BOST](https://doi.org/10.1145/3809488) | 将 `X-Y-Z-T` 组织为张量分解并结合轻量网络和 ray-distortion correction，面向高时空保真与内存/时间成本 | 当前 causal previous/current BP 只是很弱的时间信息；若未来主张时序贡献，必须与 4D tensor/temporal 方法区分并在统一数据合同下比较 |
| [Reconstruction refinement of hybrid BOST](https://doi.org/10.1063/5.0190778) | 连接 coarse 与 refined reconstruction，并比较 paraxial/non-paraxial ray tracing、初值和位移算法 | 直接支持“初值质量会影响 BOST refinement”这一物理问题，但也意味着 hybrid refinement 本身不是新贡献 |
| [Evolutionary BOST with self-adaptive parameter heuristics](https://doi.org/10.1364/OE.450036) | 使用 nonlinear ray tracing 和多 GPU evolutionary optimization 重建反应流 | 提供了非线性 forward 与高成本优化参照；当前 straight-ray proxy 正结果不能替代 curved/nonlinear BOST 门 |

## 比几何算子更危险的 warm-start 与层析近邻

“神经算子给传统迭代器提供初值”也不能作为当前工作的独立创新点。以下工作直接覆盖了这一上层范式：

| 工作 | 已覆盖的核心思想 | 对当前实验合同的直接要求 |
|---|---|---|
| [NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers](https://arxiv.org/abs/2511.02481) | 用 neural operator 生成初值，再交给未替换的 CG/GMRES 等 Krylov 方法；报告迭代数和端到端时间下降 | 当前必须把贡献限定到多视角 BOST inverse、坐标/相机条件、精确 `A/A^T` 账和严格同精度，不得把 hybrid warm start 本身写成创新 |
| [Physics-Informed Neural Operator for Warm-Starting Background-Decomposed and Preconditioned PSFD](https://arxiv.org/abs/2607.25330) | 将二维横向分支与一维轴向分支组成因子化 FNO，以物理方程自监督训练，并结合 spectral damping 为三维 EUV 电磁 PSFD 求解器提供初值 | 这是目前检索到的最直接“光学 + 三维 neural-operator warm start”近邻，阻断“首个光学 neural-operator warm start”表述；当前差异只能落在稀疏视角 BOST inverse、坐标/相机条件、逐单元同精度、精确 `A/A^T` 和失败边界。它于 2026-07-28 提交，当前是一级预印本而非已完成同行评审的结论 |
| [Neural operator-based super-fidelity: A warm-start approach for accelerating steady-state simulations](https://arxiv.org/abs/2312.11842) | 将低保真解映射为高保真初值，再由传统稳态流动求解器收敛；比较不同迭代器并报告至少两倍加速 | 当前 q8/cheap BP 到 warm field 的结构与之高度相似；必须说明我们解决的是成像逆问题而非 PDE forward，并公平计入低保真初值和模型推理成本 |
| [Spectrally Safe Neural Operator Warm-Starts for Large-Scale Newton Solvers](https://arxiv.org/abs/2606.21828) | 证明很低的平均 relative-L2 仍可能产生局部物理缺陷并破坏后端 Krylov 所需的谱性质，再以物理正则修复 | 强化当前逐单元 field/full-gradient/interior-gradient/observation 与 harm 门的必要性；平均误差、漂亮切片或迭代数下降绝不能单独证明 warm start 安全 |
| [Neural Operator Learning for Ultrasound Tomography Inversion](https://arxiv.org/abs/2304.03297) | 用 T-FNO 从二维发射器-接收器 TOF 场直接预测二维声速场，并与参数相近的 U-Net 比较 | 层析 inverse + FNO 也不是空白；未来 BOST 实验至少要有参数/输入公平的 FNO 与 U-Net，并强调三维多视角折射率/密度梯度和 exact refinement 的差别 |
| [Neural Inverse Operators for Solving PDE Inverse Problems](https://arxiv.org/abs/2301.11167) | 面向 inverse map 组合 DeepONet/FNO 表示，直接学习由多组观测到未知系数/场的映射 | 说明 inverse operator 也已有专门架构；若输入/输出空间可公平对齐，NIO 比 generic FNO 更适合作 inverse-specific baseline |
| [Invertible Fourier Neural Operators for Tackling Both Forward and Inverse Problems](https://proceedings.mlr.press/v258/long25a.html) | 以可逆 Fourier blocks 和变分表示联合处理 forward/inverse 问题 | iFNO 是必须讨论的相关工作；只有观测编码、输出网格和训练预算能公平对齐时才升级为实验基线 |
| [Learned Primal-Dual Reconstruction](https://arxiv.org/abs/1707.06474) | 将 learned primal/dual updates 与 forward/adjoint operator 结合，形成数据一致性的 learned iterative CT reconstruction | 这是 learned iterative inverse reconstruction 的经典近邻；当前选择“只学习初值、后端不修改”必须作为可审计设计差异，而不是忽略该类方法 |
| [ΨDONet: Learning Iterative Schemes for Optimization Problems in Imaging](https://arxiv.org/abs/2006.01620) | 将 ISTA 型成像迭代展开并学习模型误差/更新 | 如果不复现此类 unrolled control，需要说明其 online operator budget、训练数据或实现不可比，并至少在 related work 中讨论 |

上述一级来源把潜在论文标题进一步收窄为：不是“neural operator warm start”，也不是“diffeomorphic neural operator”，而是**在三维 BOST 代理逆问题中，对 coordinate-conditioned learned initializer 是否能在精确物理精化和严格成本合同下保留同精度的可证伪研究**。

## 已准备但尚未授权的最小 FNO 对照

为避免三个 CNN parent control 全部未否决后再临时设计强弱不公平的基线，当前已单独实现一个 conditional reference-chart 3-D FNO：

- 与 v111 候选读取完全相同的 deployment-visible q8、causal cheap BP、support、7 个 local-ray geometry channels、13 个 map channels 和 camera tokens；
- 复用同一个已知参考坐标，不额外学习 Geo-FNO deformation；
- 三层谱卷积，宽度 6，低模态 `(5, 3, 3)`，42,166 个实参数，对比 DRC-CNN 的 42,237 个参数，只差 71 个参数（0.168%，预注册容差 0.2%）；
- 保持同一个 `0.5` correction cap、物理 lift、未修改 K1 和每帧 `2A+2A^T` 逻辑账；
- 当前状态严格为 `NOT_AUTHORIZED_FOR_TRAINING`：默认实例的全部参数冻结，调用 `.train(True)` 会 fail closed，且尚不存在 authorized-training builder。只在三个 CNN parent families 均未否决候选后才可另行冻结授权入口。

这只是预先排除 baseline 设计偏差的工程准备，不是 neural-operator 结果，也不改变 `algorithm_breakthrough=false`。

## 三个 CNN 父控制真正回答什么

### Temporal CNN no-pose/no-map

它删除显式局部射线、map displacement/Jacobian 和 camera token，但 q8 与 cheap BP 仍可能隐式携带坐标效应。因此它只能回答：显式坐标通道是否给候选带来超出较小 causal 3D CNN 的收益；不能证明候选相对所有 coordinate-blind 方法都更好。

### Pose CNN no-map

它保留 local ray geometry 和 camera token，只把显式 map 通道置零。它回答已知 diffeomorphic map 是否提供额外作用，而不是回答完整几何条件是否必要。

### Reference CNN no-camera-token

它仍能看见 local ray geometry 和 map channels，只删除 camera token。它回答显式全局相机 token 是否必要；不能称为 geometry-free control。

三个父控制都使用同一数值种子、同一五条 leave-one-trajectory-out 轨迹、同一 `2A+2A^T` 逻辑成本和同一逐轨迹严格门。任一父控制通过全部轨迹门并否定同种子候选，当前 learned-advantage 主张就必须关闭。

## 论文中可以和不可以写的句子

### 当前可以写

- We study a BOST-specific, coordinate-conditioned warm initializer followed by unchanged exact-physics Krylov refinement.
- Known geometry and deployment-visible observations are used to construct the initializer; held-out truth is reserved for scoring.
- Diffeomorphic transport is an established device; our question is whether it is useful inside a cost-audited BOST inverse-reconstruction pipeline.

### 当前不能写

- first diffeomorphic neural operator for varying geometries
- first optical neural-operator warm start
- globally unique coordinate-equivariant reconstruction method
- discretization-invariant neural operator
- state of the art, external generalization, real BOST acceleration, or paper success

## 后续必须补齐的比较

即使三个 CNN 父控制都不能否定候选，也只完成内部 learned-advantage 门。投稿前仍需要：

1. 传统下限必须包括 Zero/cheap-BP cold start 的 CGLS/PCGLS、适用的 dual-ridge/正则化重建，并保留完整 `A/A^T` 和成本账。
2. 先在现有五轨迹 LOTO 合同中比较同输入、参数量相近的 physical/reference-chart 3D CNN 与已经预定义的 reference-chart FNO；只有内部机制成立，才进入独立公开反应流工况。
3. 若主张几何贡献，独立公开工况加入 Geo-FNO/DAFNO-style control；至少再有同预算 3D U-Net/DeepONet。若结果显示明显的轴向/横向频谱不对称，再增加 EUV-PINO 风格的 `2-D lateral + 1-D axial` 因子化 FNO 消融，但不能在当前父控制完成前借它无边界扩模。NIO/iFNO 仅在观测编码、输出和预算可公平对齐时作为实验基线，否则作为必须讨论的相关工作。
4. 至少讨论一个 learned iterative/data-consistency 邻居（Learned Primal-Dual 或 ΨDONet）；若不复现，要给出 online operator budget、数据或实现不可比的明确理由。
5. 比较分成两条不可混写的轨道：warm-start 方法必须使用相同 observation、几何、训练轨迹、未修改 CGLS/PCGLS 后端和完整成本账；NeRIF、NIO/iFNO、Learned Primal-Dual、ΨDONet 等 direct/unrolled 方法按各自完整端到端管线比较，不强制拼接同一后端，但必须报告实际 operator calls、wall、RSS 和训练/预处理边界。
6. 两条轨道都先过相同的逐单元 matched-accuracy/harm 门，再比较 exact `A/A^T`、fresh-process wall 和 whole-pipeline peak RSS；不得把 direct 单次推理与 warm-start 的不完整阶段直接对比。
7. 最终使用组内真实位移图、相机标定、重复测量噪声和认可重建基线重新定义实验同精度。
8. 跨分辨率证据是声称 `discretization-invariant` 或 `resolution-generalizing` 的必要条件；若没有，继续使用 `learned warm initializer` 这一更稳妥名称。

## 最稳妥的创新表述

在外部公开工况和真实 BOST 门通过前，最稳妥的表述是：

> A falsifiable study of coordinate-conditioned learned warm starts for multi-view BOST reconstruction under exact-physics refinement, equal-cost parent controls, cellwise matched-accuracy gates, and explicit failure boundaries.

这里的贡献候选是问题设置、BOST-specific 组合和严格证据链，不是微分同胚、FNO、公共参考域或 CNN 组件本身。
