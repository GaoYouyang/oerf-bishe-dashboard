# 少视角 / 折射射线 / BOST 三维与四维重建：算子学习新颖性审计

更新时间：2026-07-16
定位：给 Gao Youyang 与何远哲师兄讨论用的研究设计底稿。它提出可证伪候选，不承诺任何候选优于 DeepONet、FNO/FFNO、NeRIF 或 TDBOST。

## 0. 先把任务写对

目标不是“设计一个更大的神经网络”，而是学习一个带采集条件的逆算子：

`G_theta : (y, camera/ray geometry, mask, confidence/noise, time) -> n(x,y,z,t)`

其中 BOST 观测来自折射率梯度沿光路的积分；在弱折射、针孔和小位移近似下常写成线性算子 `y=A_g n+epsilon`。真实系统还会叠加光流/相位提取误差、有限孔径、弯曲光路、标定漂移、遮挡、相机异方差和时间不同步。新算法只有明确解决其中一项，并在该项出现时相对强基线占优，才有论文意义。

## 1. 真实物理困境

### P1：少视角和有限角造成非唯一性

NeDF 明确把受限光学窗口、狭小实验空间和少角度视作实际 TBOS 的核心困难；网络用位置编码与分层采样表示 density-gradient field，并用 LES 与非线性 ray tracing 生成数据。它已经占据“普通 neural field + sparse-view BOST”位置。[NeDF, arXiv 2409.19971](https://arxiv.org/abs/2409.19971)

因此，新的少视角算法不能只报告更高 PSNR；必须说明缺失角对应的不可见成分如何被约束、哪些细节只是先验补全，并给出数据一致性或不确定度证据。

### P2：折射和成像模型会失配

NeRIF 用连续隐式场同时输出折射率及梯度，并沿标定光路积分；它主要解决体素离散、分辨率、噪声和计算开销。[NeRIF, arXiv 2409.14722](https://arxiv.org/abs/2409.14722)

但已有工作分别覆盖了两个更真实的成像失配：cone-ray 模型处理有限孔径/景深，[AIAA Journal / arXiv 2402.15954](https://arxiv.org/abs/2402.15954)；单视角折射率神经场显式求弯曲光路，[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)。所以“给 NeRIF 加 ray tracing”本身不新。

### P3：观测噪声不是同方差白噪声

真实 HBOS 火焰实验报告了图像边缘和角落因照明减弱而产生更高相位噪声；同时有限光学通道要求少相机同步测量。2026 年的 model-based neural reconstruction 已在 14 个投影、真实旋流稳定火焰上展示了模型驱动 U-Net/FBP 路线，并把少投影与热声振荡作为明确应用。[Advanced Imaging, DOI 10.3788/AI.2025.10028](https://doi.org/10.3788/AI.2025.10028)

这给出一个很具体的缺口：相机、像素、背景频率和亮度决定不同的误差协方差，而大多数 operator baseline 把所有观测等权处理。

### P4：四维低秩会抹去瞬态创新

何远哲等人的 TDBOST 把 `X-Y-Z-T` 组织为张量，并使用 `(XY-ZT)/(XZ-YT)/(YZ-XT)` 分解结合轻量网络，已经占据“普通 4D 张量分解 + 网络”的位置。[ACM TOG, DOI 10.1145/3809488](https://doi.org/10.1145/3809488)

动态 CT 中也已有正交体/hash-grid 与时空 Transformer 的 STNF4D。[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33177) 因而新 4D 方法必须针对 BOST 的异步投影、相机 dropout、拓扑突变或火焰前缘稀疏创新，而不能只是换一种 factorization。

### P5：reprojection 好不等于三维场正确

有限角逆问题存在大近似零空间。不同三维场可以产生几乎相同观测，尤其是缺失方向上的结构。Deep null-space learning 说明可把学习修正限制到零空间并保留数据一致性；这也意味着“held-out reprojection 很好”仍不能单独证明体场真实。[Inverse Problems 2019 / arXiv 1806.06137](https://arxiv.org/abs/1806.06137)

## 2. 七类模型放在同一坐标系里比较

| 家族 | 最适合承担的角色 | 对 BOST 的天然短板 | 本项目中的强基线身份 |
|---|---|---|---|
| DeepONet | 观测函数/传感器值到任意查询坐标的函数值 | branch 对可变相机集合、mask 和高维观测压缩敏感；低秩 trunk 可能抹平薄前缘 | 必须有；固定几何 projection-to-volume control |
| FNO / FFNO | 规则网格上的全局、多尺度 field-to-field 映射 | 原生不理解不规则 ray set；FFT 周期/固定网格偏置与相机变化不匹配 | 必须有；同参数、同训练预算 control |
| geometry-informed operator | 点云/几何到规则 latent grid，再做全局 operator | 现有 GINO 主要学 forward PDE surrogate，不自动解决逆问题非唯一性 | 可变 geometry 的强对手，不是直接可抄的答案 |
| latent / neural field | 每个实例连续表示 `n(x)`，空间分辨率灵活 | 常需逐实例优化；先验可能产生不可观测细节；NeRIF/NeDF/NR primitives 已很拥挤 | NeRIF、NeDF、2026 primitives 都必须比较 |
| unrolled optimization | 每层显式使用 `F` 与 `F^T/J^T`，数据一致性清楚 | 训练/推理要多次 forward-adjoint；若只换 proximal，容易撞 learned primal-dual | 最适合做首个可执行候选 |
| low-rank tensor / time operator | 压缩 4D 时空场、利用时间冗余 | 固定低秩容易抹掉突发前缘、拓扑变化和非平稳事件 | TDBOST、framewise NeRIF、STNF4D、FFNO 必须同台 |
| uncertainty-aware operator | 标记多解、OOD 和不可见区域 | 方差不等于校准；MC dropout/ensemble 可能只反映模型差异 | 必须报告 coverage、risk-coverage 和 query-view calibration |

基础论文入口：[DeepONet](https://doi.org/10.1038/s42256-021-00302-5)、[FNO](https://openreview.net/forum?id=c8P9NQVtmnO)、[F-FNO](https://iclr.cc/virtual/2023/poster/10680)、[GINO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)、[Neural Inverse Operators](https://arxiv.org/abs/2301.11167)、[Learned Primal-Dual](https://arxiv.org/abs/1707.06474)、[Approximate Bayesian Neural Operators](https://arxiv.org/abs/2208.01565)。

## 3. 六个可实现、但必须先做撞车审计的候选

### A. CG-PDNO：协方差与几何条件的数据一致展开算子

**优先级：P0；最适合本科毕设首发。**

- 输入：多相机 displacement/phase `y`、active mask、每 ray 几何、support、像素/相机 confidence 或协方差、可选 ridge/FNO 初值。
- 输出：三维折射率 `n`；不先输出 learned stop。
- 结构：`C^{-1/2}` 预白化；共享 ray-set encoder；6–10 层 primal/gradient update；每层显式调用 `F_g` 与 `F_g^T/J_g^T`；小型共享 proximal/operator block；投影后强制 support/non-negativity。
- 物理归纳偏置：测量协方差、相机集合置换不变性、数据一致性、正确伴随、硬 support。
- 只预期在：照明不均、相机噪声不同、视角 dropout、布局变化时优于等权 FNO/DeepONet。固定干净布局上可能没有优势。
- 强基线：TV/Tikhonov/SART、fixed Landweber、PBB+discrepancy、SPG、ridge-FNO、DeepONet、matched U-Net/FNO、Learned Primal-Dual、GINO-style geometry encoder。
- 关键消融：去预白化；打乱 covariance；去 geometry；错误 `F^T`；unshared/shared layers；同调用预算；oracle covariance 与可部署估计分开。
- 撞车风险：**中高。** Learned Primal-Dual、geometry-informed operator 和 data-consistent network 都已有；可能的新意只能是 BOST 特有的 variable ray geometry + calibrated heteroscedastic covariance + 严格调用/尾部审计组合。
- 一票否决：在 fresh leave-one-geometry-out 上不能同时改善 mean、p10、harm 和 calibration；或优势完全被 deterministic prewhitening + SPG 解释。

### B. VN-UQNO：可见/近零空间分解的多假设算子

**优先级：P1；更像论文第二阶段。**

- 输入：`y, geometry, mask` 与可选 latent `z`。
- 输出：data-consistent mean、`K` 个近零空间样本、voxel/front uncertainty，而不是伪装成唯一真值。
- 结构：稳定正则化求 visible component；网络只提出 correction；用 `I-A^dagger A` 或 data-proximal 投影限制到近零空间；query camera 对 posterior 做校准。
- 物理归纳偏置：观测可见信息不被网络改写；不可见方向用分布表示；每个样本强制 reprojection tolerance。
- 只预期在：极少视角、有限角、薄前缘方向不可见时优于单点估计。它的优势应是风险识别与覆盖率，不一定是平均 L2。
- 强基线：deterministic FNO/DeepONet、deep ensemble、Bayesian NO、MC dropout、deep null-space network、data-proximal null-space network、conditional diffusion posterior。
- 关键消融：无 null projection；单点均值；不同 `K`；oracle/query-free calibration；可见/不可见频谱分区；错误 geometry。
- 撞车风险：**高。** null-space network 与 Bayesian inverse imaging 已成熟。BOST 贡献必须落在 ray-geometry-dependent approximate null space、front-event coverage 和 held-out optical view 校准。
- 一票否决：uncertainty 与真实误差无排序关系，或样本虽多但投影不一致。

### C. CR-MDO：曲线/cone-ray 模型失配修正算子

**优先级：P2；需真实标定后再开。**

- 输入：raw/reference BOS 图或 displacement、孔径/f-number、camera/background geometry、当前折射率场。
- 输出：折射率场与一个受限的 forward-model correction/ray-path correction。
- 结构：thin-ray inverse warm start；可微 eikonal/ray tracer 或 cone sampler；小型低秩 model-discrepancy operator；交替更新场与光路。
- 物理归纳偏置：弱折射极限 correction 必须趋于零；弯曲光路满足 ray equation；有限孔径是 ray bundle 而非单线。
- 只预期在：强折射梯度、大 flame、宽孔径、景深模糊时占优。小位移、f/22 条件下不应显著赢。
- 强基线：NeRIF、NeDF、cone-ray NIRT、single-view curved-ray neural field、UBOST、full nonlinear ray-tracing optimization。
- 关键消融：straight/cone/curved；learned correction 与纯物理 tracer；孔径 OOD；gradient strength sweep；弱折射零修正测试。
- 撞车风险：**很高。** 物理组件几乎都已有。只有“跨实例 amortized multi-view BOST model-error correction”且在真实高折射数据成立时才可能有新意。
- 一票否决：收益只来自更多 ray-tracing 计算或训练 generator 与测试共用同一 tracer。

### D. TRAIL-4D：transport-aware rank + sparse innovation 的异步 4D 算子

**优先级：P2；直接贴何远哲 4D 线，但数据门槛高。**

- 输入：带真实 timestamp 的多视角序列、camera mask/geometry、可选速度或粗 optical-flow cue。
- 输出：连续或离散 `n(x,y,z,t)`、adaptive rank、稀疏 innovation/event field。
- 结构：低秩时空 base；时间 operator/SSM 演化系数；sparse innovation 分支捕捉新生/熄灭/拓扑变化；每个实际曝光时刻执行 BOST forward consistency。
- 物理归纳偏置：因果时间、可选 advection/continuity residual、rank sparsity、事件稀疏性、异步测量算子。
- 只预期在：高帧率冗余、偶发瞬态、帧缺失或相机不同步时优于 framewise 方法。完全随机快速变化时可能失败。
- 强基线：framewise NeRIF、何远哲 TDBOST、STNF4D、Tensor4D、3D/4D FNO/FFNO、ConvLSTM/GRU、固定 rank 与 no-innovation 版本。
- 关键消融：固定/自适应 rank；无 innovation；无 timestamp；同步/异步；时间反转；不同事件率；相同显存和 latency。
- 撞车风险：**高。** 普通 tensor factorization、动态 neural field 和时空 operator 都已有。潜在新意是 BOST 异步曝光 + sparse innovation + optical forward consistency 的组合。
- 一票否决：只在平滑 synthetic 序列上赢，或 thin-front/event recall 下降。

### E. RI-DualBOST：原始图像与 deflection 双域算子

**优先级：P3；数据和工程量大。**

- 输入：reference/distorted background image、pattern spectrum、相机标定。
- 输出：带 confidence 的 displacement/phase 与三维折射率场。
- 结构：共享图像 encoder；deflection posterior branch；reconstruction operator；可微 image warp/ray renderer 闭环；UBOST-style raw-image consistency。
- 物理归纳偏置：背景图案频谱、brightness-dependent uncertainty、image formation、观测域与场域双重一致性。
- 只预期在：光流失败、图案变化、局部遮挡、低照度和大位移时占优。高质量已知 displacement 下不应胜太多。
- 强基线：DeepFlow/WOFA + NeRIF，UBOST，现代 optical flow + classical tomography，raw-image neural implicit reconstruction。
- 关键消融：只用 displacement；只用 raw image；confidence；pattern OOD；brightness sweep；same reconstruction backbone。
- 撞车风险：**很高。** UBOST 已统一 image distortion 与 reconstruction；深度 optical flow 很成熟。必须证明 uncertainty propagation 或跨 pattern 泛化是独立贡献。
- 一票否决：比较时上游光流和下游重建不匹配，或只在单一背景图案上有效。

### F. BA-ViewNO：风险约束的相机布局与逆算子协同设计

**优先级：P3；作为实验设计支线。**

- 输入：已有视角、候选 optical ports/ray sets、安装成本、当前 posterior/uncertainty。
- 输出：下一相机或 `K` 相机子集、对应重建及最坏场风险。
- 结构：permutation-invariant ray-set encoder；近似 Fisher/singular-value features；uncertainty reduction utility；离散选择用 greedy/Gumbel 或离线 policy。
- 物理归纳偏置：可安装区域、遮挡、同步成本、角间隔和 forward sensitivity。
- 只预期在：布局可调整、光学窗口受限、不同实验装置之间迁移时占优。固定 9 路内窥镜装置上没有意义。
- 强基线：uniform、maximum angular gap、random、QR/Fisher greedy、condition-number minimization、独立训练 operator 后再选布局。
- 关键消融：无 uncertainty；无成本；只看角度；只看奇异值；leave-one-rig-out；最坏场而非均值目标。
- 撞车风险：**中高。** active tomography 和 learned sensor placement 已有。BOST 特有安装约束与 operator-tail-risk 联合才可能构成贡献。
- 一票否决：收益来自增加有效相机数，或真实装置根本不能改变布局。

## 4. 推荐排序

1. **先做 A / CG-PDNO。** 当前已有 T16 forward/adjoint、FNO/DeepONet、PBB 和噪声停止审计，可最低成本建立公平基线。首先只实现 deterministic prewhitening；它若已解释全部收益，立即停止 learned proximal。
2. **把 B 作为 A 的可靠性扩展。** 不追求“唯一更准”，而是证明少视角不可见结构的 uncertainty 是否可校准。
3. **拿到连续高速组内数据后再做 D。** 没有真实 timestamp、相机同步信息和 TDBOST baseline 时，不启动 4D 新模型。
4. **C/E/F 只在师兄确认物理痛点和数据接口后启动。** 它们潜在价值高，但最容易因数据不匹配而变成架构展示。

## 5. A 候选的最小可发表实验矩阵

### 数据轴

- field family：smooth / flame sheet / thin front / thermoacoustic oscillation。
- view：固定数目、camera dropout、有限角、leave-one-layout-out。
- noise：white、camera-wise heteroscedastic、detector-correlated、brightness-dependent。
- model：thin ray、cone ray、轻度 curved-ray mismatch。
- domain：训练 generator、独立 LES/ray tracer、至少一个真实 BOST case。

### 方法轴

- classical：Tikhonov/TV/SART、Landweber、PBB+冻结 discrepancy、full-call SPG。
- direct learned：matched U-Net、FNO/FFNO、DeepONet、geometry-conditioned FNO。
- model-based learned：Learned Primal-Dual 或等价 unrolled baseline。
- candidate：prewhiten-only、geometry-only、unrolled-only、完整 A。

### 评价轴

- field：relative L2、front Chamfer/normal error、gradient error、峰值/积分量。
- observation：active reprojection、held-out camera、per-camera normalized residual。
- reliability：p10 gain、`>1%` harm、worst-layout、coverage、risk-coverage。
- compute：参数、FLOPs、峰值内存、wall time、`F/F^T/J^T` 调用数。
- statistics：source field 为独立单元；paired delta；field-cluster bootstrap；预注册 selection commit；fresh blind。

## 6. 论文创新门槛

以下任何一条缺失，都不应写“优于现有算法”：

1. 与 NeRIF、NeDF、2026 neural primitives、TDBOST 和 model-based neural BOST 的角色区别被明确写出。
2. candidate 与 deterministic physics upgrade 在同一 forward、同一初始化、同一调用预算比较。
3. 新方法只在预先声明的困难域占优，干净 IID 不强求全面获胜。
4. 结果同时覆盖 mean、tail、OOD、held-out optical evidence 和计算成本。
5. 所有 oracle（clean observation、truth-best iteration、真实 noise）与 deployable input 严格分栏。
6. 至少一次使用独立生成器或真实数据；不能在同一个 synthetic world 内训练、调参、测试和解释。

## 7. 现在要问何远哲的六个问题

1. 组内 projection 是 optical-flow displacement、相位、还是 raw background image？是否有每像素 confidence/亮度？
2. 九路内窥镜的 geometry 在不同实验间是否变化？是否存在 camera dropout、遮挡或标定漂移？
3. NeRIF/TDBOST 是否提供可调用的 `F(x)` 与 `J^T r`，还是只有训练脚本？
4. 4D 数据是否有每相机真实 timestamp，还是默认同步？是否存在缺帧？
5. 能否封存一套从未查看的真实 case/layout，并保留一台 camera 只做 audit？
6. 师兄最在意三维 field、held-out reprojection、flame-front、PIV correction，还是速度/显存？

## 8. 当前最诚实的论文题目草案

> **Covariance- and Geometry-Conditioned Data-Consistent Neural Operators for Sparse-View Background-Oriented Schlieren Tomography**

这个题目目前只是研究假设。只有 A 在独立 geometry、异方差噪声和真实 BOST 上通过上述门槛后，才保留；否则论文应转为一篇严格的 negative/mechanism study，解释为什么普通 neural operator 在 BOST 的几何与噪声变化下失效。

## 9. 2026-07-15 独立证据更新：哪些假设已经被否掉

这一轮不再停留在架构草图，已经实现并冻结了四级小规模证伪链。所有数字仍只属于 `8 x 8 x 5` 的 prescribed linear weak-deflection audit，不能外推到真实 OERF。

| 版本 | 相对独立锁集强 PBB | p10 | `>1%` harm | 判决 |
|---|---:|---:|---:|---|
| v1 fixed-PG Base-Correction | -17.903% | -31.424% | 100% | 强基线一换，方法失效；淘汰 fixed-PG base |
| v2 PBB Base-Correction | -0.102% | -2.819% | 19.44% | 均值接近零，但尾部不可接受 |
| v3 PBB + saturation guard | +0.281% | -0.831% | 7.5% | 风险下降但仍未过 `2% / 0 / 5%` 门槛 |
| v4 5-head selective ensemble | 0% | 0% | 0% | select 无可行阈值，按协议覆盖率 0、全回退 PBB |

v4 的五个 correction head 共享同一份四步 PBB 物理状态，因此推理物理预算仍为 `4 F + 4 F^T`，而不是五倍；额外成本是 5 个 5,989 参数的小卷积头。下降尺度由 tiny-grid 的 `exact_small_matrix` 谱范数给出，不再把有限次 power iteration 当严格上界。锁集还使用了选择集未出现的 `helical_plume` 与 `stratified_ignition` 两个场族。

但更严格的实现没有带来优势：v4 raw ensemble 在选择集相对 PBB 平均 -0.546%，伤害率 19.44%。网络权重分歧不能可靠地区分“修正有帮助”和“修正伤害”。因此下面这条说法已经被当前证据否掉：

> 多个相似 correction head 的方差可以直接作为 BOST 重建风险指标。

### 与已有工作的碰撞边界

[A Simple Guard for Learned Optimizers](https://proceedings.mlr.press/v162/premont-schwarz22a.html) 已经研究 learned optimizer 与 generic optimizer 的条件切换、OOD 失败和收敛。因而“learned correction + deterministic fallback + gate”不能作为本项目的独立新颖性。BOST 论文若要成立，必须给出这个成像问题独有、且推理时可观测的接受证据，例如 held-out camera、flow-off covariance、标定扰动稳定性或真实 ray-model discrepancy；不能把真值误差偷偷作为部署 gate。

### 下一版只允许从真实证据出发

1. **held-out optical view gate**：用未参与重建的一台相机比较 candidate 与 PBB，而不是用 ensemble 方差猜风险；必须单独核算额外 forward 成本。
2. **operator-mismatch gate**：数据由 perturbed/nonlinear ray model 生成，重建只拿 nominal operator，检查门控是否能识别标定和光路失配。
3. **风险头仅作次要候选**：若用 synthetic truth 训练 risk head，必须 leave-one-family-out、fresh lock，并在真实数据上改用 held-out view 标签；否则不可部署。
4. **停止扩宽网络**：在公开真实 BOS 或 OERF 最小样例到位前，不再通过增加 head、width 或 epoch 寻找微小均值收益。

开放现实锚点是 2025 年的 [Open-source BOS tomography dataset of high-speed flow over a flight body](https://arxiv.org/abs/2508.17120)：论文提供 70 views，并报告 nine-view limited reconstruction 与 validation deflection。站内可简称 OpenBOST，但它不是论文正式提出的算法名；应把它用作真实 loader、held-out optical evidence 和 cost audit，而不是当作带三维绝对真值的 benchmark。

## 10. 2026-07-16 首次开锁：有限孔径盲校准为什么失败

v5a 不再训练 correction network，而是先问一个更基础的问题：在数据由 cone-ray 有限孔径算子生成、重建只知道 pinhole nominal operator 时，能否从重建残差中盲选孔径半径，再决定是否接受校正？选择阶段冻结后，代码、配置和预注册协议先以 commit `90ff961` 推送，随后只打开一次独立锁集；审核相机既不参与重建，也不参与噪声尺度估计。

| 首次开锁指标 | v5a 结果 | 冻结门槛 | 判决 |
|---|---:|---:|---|
| mismatch penalty | 3.144% | `>= 2%` | 通过，失配不是过弱 |
| mean gain | +1.753% | `>= 2%` | 失败 |
| p10 gain | -17.784% | `>= 0%` | 失败 |
| 全锁集 `>1%` harm | 25.0% | `<= 5%` | 失败 |
| 预注册 audit 指标 | -0.439% | `<= 1%` | 通过 |

因此五项只过两项，v5a 的正式结论是 **FAILED**。独立 validator 复核 16,103 个原子检查，证明哈希链、锁前 commit、调用预算、掩码与汇总内部一致；它只证明实验记录可信，不把失败算法变成成功算法。

### 10.1 失败机理不是“网络太小”，而是参数不可辨识

- 36 个锁样本中，34 个孔径估计落在候选区间两端 `0` 或 `0.16`；只有 4 个选到最邻近真实孔径的候选。
- 18 个 `helical_plume` 全部被选成 `0`，而 18 个 `stratified_ignition` 中 16 个被选成 `0.16`。selector 主要在识别场形态，不是在测孔径。
- 已接受样本中，confidence 与真实 reconstruction gain 的 Spearman 相关仅 `-0.0026`，几乎没有排序能力。
- `helical_plume` 覆盖率高达 94.44%，但平均 gain 为 `-2.222%`、全族 harm 为 44.44%；这正是“错误但自信”的失败模式。

这构成当前最有价值的负证据：若每个场独立联合估计 `x` 与孔径参数 `theta`，场的高频/形态变化可以吸收光学失配，残差最小不等于 `theta` 可识别。下一版首先要消除 `x-theta` confounding，而不是扩宽网络。

### 10.2 风险指标必须按接受条件重算

v5a 原协议的 harm 分母包含了回退样本，因而 25.0% 是全锁集风险；在真正被 selector 接受的 24 个样本中，9 个受损，conditional harm 达 37.5%，accepted-only p10 为 `-24.779%`。审核相机的“逐样本百分比再平均”为 `-0.439%`，但“平均 RMS 的百分比变化”为 `+0.308%`，且 37.5% 的接受样本 audit residual 变差。

这不篡改已冻结的 v5a 判决，但暴露了以后必须同时报告的四组量：

1. 全样本风险与 `risk x coverage`；
2. 接受条件下的 mean、p10、harm；
3. audit 的 mean-of-ratios 与 ratio-of-means；
4. 接受样本中 audit 变差率和最大恶化。

### 10.3 v5b：RigCal-BOST 只是一项待证伪设计

下一候选不再对每个场自由盲估孔径，而把光学参数当作同一采集装置/批次共享的低维量，并以元数据先验约束：f-number、焦距、物面/背景距离、focus setting 与参考 PSF。对每个候选 `theta`，先求各场 nuisance reconstruction，再比较外层保留光线；这相当于 profile objective，而不是让单场形态直接决定光学参数。

可辨识性门槛使用局部 Jacobian：`J_x=A(theta)`、`J_theta=(dA/dtheta)x`，通过 Schur complement/profile Fisher 检查 `theta` 的敏感方向是否仍能被 `x` 的变化解释。只有 profile curvature、leave-one-scene-out 稳定性、元数据一致性和外层 held-out residual 同时合格，才允许使用 calibrated cone operator；否则回退 nominal cone，再回退 pinhole FISTA。

这不是凭组件命名制造新颖性。blind calibration、variable projection/profile likelihood、Fisher identifiability、operator correction 和 selective prediction 都已有成熟文献。潜在贡献必须收窄为：

> 在有限孔径 BOST 中，给出场形态与装置参数混淆的可复现证据，并验证装置级共享、可辨识性门控的 cone-ray 校准能否在独立装置、独立场族和真实 f-stop sweep 上降低尾部风险。

若没有真实 f-stop/PSF 元数据与至少两个 acquisition blocks，这条主张不能成立。仅在同一 synthetic generator 上赢 FNO、FFNO 或 DeepONet，也不能支持真实 BOST 论文。

### 10.4 下一轮冻结前的最低样本门槛

假如接受样本中观察到零次 `>1%` harm，单侧 95% Clopper-Pearson 上界要低于 5%，至少需要 59 个独立接受样本。若覆盖率约 20%，确认锁至少需要约 295 个独立 field/block 单元；同一 field 的多噪声副本不能冒充独立样本。因而 v5b 先做机制筛查，只有通过 identifiability 和开发集尾部条件后，才值得承担确认性锁集成本。

当前两个诚实题目草案分别是：

- 负结果/机制论文：**Identifiability before Calibration: A Failure-Driven Study of Finite-Aperture Operator Mismatch in Background-Oriented Schlieren Tomography**。
- 仅在 v5b 与真实 f-stop sweep 成功后保留：**Rig-Shared Identifiability-Gated Cone-Ray Calibration for Robust Background-Oriented Schlieren Tomography**。

两者现在都只是研究路线，不是可投稿结论。
