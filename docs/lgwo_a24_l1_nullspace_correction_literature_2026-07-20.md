# LGWO-A24-L1：Null-space、Correction Operator 与 Learned Krylov 的一级来源边界

**日期：** 2026-07-20
**状态：** `LITERATURE_AUDIT / PRE-TRAINING / NO_ALGORITHM_BREAKTHROUGH`
**用途：** 为 OERF 三维 BOST 可学习方向 pilot 划定一级来源边界，防止把已有的 null-space prior、粗解加校正、神经预条件或 neural operator 逆问题方法重新命名为 LGWO-A24 的“首次创新”。

> **最高优先级结论：目前没有算法突破。** 本文只完成一级来源核对、重叠识别、差异化假设和强基线设计。LGWO-A24 尚未训练，尚未打开 route/fresh，也没有真实 OERF BOST 结果、DeepONet/FNO 横向结果或泛化结果。下面所有“可能”“可主张”都必须等严格实验通过后才能升级为结果。

## 1. 先给师兄看的判断

LGWO-A24 最危险的误判是把题目写成“首次用神经网络学习三维 BOST 的 null space”。这条表述已经被近期一级来源分解为至少四类已有思想：

1. **直接学习 null-space 先验或其低维系数：** NPN 已经明确提出从 sensing matrix 的 null-space 选取低维投影，并用网络预测相应系数。
2. **先用物理/迭代方法得到粗解，再学习 correction：** Neural Correction Operator 已经把 inverse map 写成 reconstruction operator 加 correction operator 的组合，并在 EIT 上比较了直接 neural operator。
3. **让神经网络参与 Krylov/CG 的预条件或内层求解：** Neural-preconditioned Poisson Solver、Learning Preconditioners for CG PDE Solvers、FCG-NO 都已证明这一范式存在。
4. **为 inverse PDE 设计 DeepONet/FNO 的组合结构：** Neural Inverse Operators 与 iFNO 已经把 operator-to-function inverse、正反向共享或可逆结构作为明确研究对象。

因此，LGWO-A24 不能靠“用了 neural operator”“用了 null-space”“用了 Krylov”三个词获得新颖性。比较有希望、但仍未验证的差异化是：

> 在三维 BOST/TBOS 的有限视角、几何变化和观测误差背景下，是否可以在严格固定的 `24F/24A^T` 部署预算内，让一个小型 geometry-conditioned model 只学习第一修正方向，并由 exact data-consistency shell、support gating、B-ray 留出重投影和 fail-closed 安全门约束它；该方向是否在未见 geometry cluster 上比同预算的固定方向、直接 inverse operator 和神经预条件更稳。

这是一条**待检验的研究假设**，不是已有结果，也不是已证明的理论。

## 2. 术语拆开：四种“学习修正”不是一回事

| 名称 | 数学位置 | 网络学什么 | 是否天然保持数据一致性 | 与 LGWO-A24 的关系 |
|---|---|---|---|---|
| Null-space prior | `ker(A)` 或其低维子空间 | 从观测预测不可观测/未采样分量 | 只有在修正严格落入 `ker(A)` 时才保持；近似正交不等于 exact | NPN 是最接近的外部重叠；LGWO 必须做同预算、BOST 几何和安全尾部比较 |
| Correction operator | 粗解 `x_0=R(y)` 到真值/残差的映射 | 粗解的场域 correction | 取决于 correction 后是否重新经过 forward consistency | NCO 已占据“粗解 + 网络校正”一般范式；LGWO 需突出预算壳层和 optical operator |
| Neural preconditioner | 解线性系统的预条件 `M_theta` | 让 Krylov/CG 更快收敛的变换或近似逆 | 由外层迭代器保证，网络本身通常不是最终解 | Lan、Li、FCG-NO 已占据该范式；不能把一个方向 proposal 直接叫 preconditioner |
| Learned Krylov direction | `span{r, A^T r, ...}` 或受约束的更新方向 | 更新方向、系数或小子空间 | 只有外层 solver 重新检查 residual 才安全 | 这是 LGWO 的候选命名，但必须明确它不是 exact Krylov basis、不是 projector、不是“学到了真实 null space” |

## 3. 一级来源逐篇核对

### 3.1 NPN：最直接的 null-space 重叠

**文献**
Roman Jacome, Romario Gualdron-Hurtado, Leon Suarez, Henry Arguello, “NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems.”
**来源状态：** arXiv preprint，`arXiv:2510.01608`；本审计按预印本处理，不把其理论和实验写成已经由同行评审确认的定论。
**一级链接：** [arXiv abstract](https://arxiv.org/abs/2510.01608) · [arXiv HTML full text](https://arxiv.org/html/2510.01608) · [作者代码链接（论文内）](https://github.com/yromariogh/NPN)
**DOI：** 本审计未将其写成有 DOI 的期刊论文；引用时使用 arXiv 标识，并注明版本日期。

**它已经解决/主张了什么**

- 针对线性、欠采样、含噪 inverse problem，明确把 sensing matrix 的 null-space 作为先验结构，而不是只在 image domain 施加一般正则。
- 选择一个低维 null-space projection，训练网络从 measurements 预测该投影的系数；论文还讨论固定正交初始化与联合优化 projection matrix。
- 在 compressed sensing、deblurring、super-resolution、CT、MRI 等多个 imaging inverse problems 中，将 NPN 放入 plug-and-play、unrolling、DIP 和 diffusion 框架。
- 论文给出了特定假设下的 PnP convergence/reconstruction analysis，并引入与估计误差相关的 convergence improvement zone。其假设、任务和算法对象不能自动迁移到 BOST。
- 论文自己也指出选择 null-space projection 的规模重要，且 CT/MRI 中测量空间与选定 null-space 的非线性相关可能比 deblurring/SR 更难。这个限制对 BOST 的有限视角、几何失配和真实噪声尤其重要。

**LGWO-A24 不得再宣称**

- “首次学习 null-space”“首次让网络预测 null-space component”“首次做 sensing-matrix-specific null-space prior”。
- “只要输出经过 support mask 就是 null-space correction”。support 内输出与 `A d=0` 没有逻辑等价关系。
- “有限步、正则化或 damped Krylov update 是 exact null-space projector”。除非对当前非线性/线性化 operator 给出并验证 exact 证明，否则只能叫 `k-step near-null filter`、`learned direction proposal` 或 `data-consistency-constrained correction`。
- 把 NPN 在 CT/MRI/压缩感知的结果写成 BOST 或反应流结果；NPN 是影像逆问题邻居，不是 OERF 实验验证。

**仍可能成立的差异化命题（候选，不是结果）**

1. **部署预算差异：** NPN 的主要对象是 null-space prior/regularizer；LGWO-A24 可以检验一个在固定 `24F/24A^T` 壳层中只调用一次的 first-direction proposal 是否能带来净 field/H1 改善，同时守住 measured、clean、heldout-B 和尾部安全门。
2. **BOST 观测结构差异：** 输入不是普通 image 或 sinogram，而是多 camera BOS displacement、`A^T y`、显式 ray geometry 和 support。相机联合置换、ray layout、detector coordinates 与 physical spacing 是可审计的 BOST-specific contract。
3. **跨 rig/geometry shift：** 可以把训练簇与未见 geometry cluster 分离，考察“同一个 small model 是否能够在 rig geometry shift 下保留收益”。只有 route/fresh 真通过后，才可以把它写成有限 synthetic evidence。
4. **失败封闭：** model 不访问 truth、B data、SVD 或 evaluator-only projector；发生 non-finite、breakdown、clipping 或 residual violation 时 fallback，而不是静默继续。这是可复现和安全结构的贡献候选，不是自动的算法优越性。

**必须加入的强基线**

- 一个同预算 `NPN-style low-dimensional null-space coefficient` baseline：固定或 fit-only 学 projection，网络只预测系数；不得使用 route/fresh truth 选 projection。
- `exact-null oracle` 只能作为 evaluator-only upper-bound/representation diagnostic，不能作为 deployable baseline，也不能进入 loss、early stop 或调参。
- `projection orthogonality defect`、`||A d||`、field error、H1 error、B-ray reprojection 和尾部 harm 必须同时报告。

### 3.2 Neural Correction Operator：粗解加校正已是通用范式

**文献**
Amit Bhat, Ke Chen, Chunmei Wang, “Neural Correction Operator: A Reliable and Fast Approach for Electrical Impedance Tomography.”
**正式期刊：** *Journal of Computational Physics* 562, 115039 (2026), DOI `10.1016/j.jcp.2026.115039`.
**一级链接：** [ScienceDirect final article](https://www.sciencedirect.com/science/article/pii/S002199912600392X) · [DOI](https://doi.org/10.1016/j.jcp.2026.115039) · [arXiv:2507.18875](https://arxiv.org/abs/2507.18875) · [arXiv HTML](https://arxiv.org/html/2507.18875) · [作者代码](https://github.com/amitbhat31/neural-correction-operator)

**它已经解决/主张了什么**

- 把 EIT 的 inverse map 写成 `D_N -> R_K(D_N) = rough reconstruction -> C(R_K(D_N)) = corrected medium`。
- `R_K` 使用固定初始化和有限次 L-BFGS 获得低保真粗解，`C` 使用 CNN/neural operator 或 conditional diffusion 进行 correction。
- 与长迭代 L-BFGS、直接学习 inverse map 的 neural operator 以及 conditional diffusion 等作比较，并报告噪声鲁棒性、重建质量和速度收益。
- 其关键经验是：在严重 ill-posed inverse problem 中，网络先面对粗解残差可能比直接从 measurement 预测最终场更容易；但论文对象是 EIT，粗解、forward operator、数据分布和指标都与 BOST 不同。

**LGWO-A24 不得再宣称**

- “首次提出 rough reconstruction plus neural correction”。
- “任何 field residual correction 都叫 Neural Correction Operator”。若没有明确的粗解 operator、correction operator、训练目标和同等 baseline，不能使用该术语暗示方法等价。
- “网络 correction 一定比 direct DeepONet/FNO 更可靠”。NCO 的比较是 EIT-specific；LGWO 必须在 BOST operator、同预算、同数据合同下重新测试。
- 不得把 NCO 的 EIT noise robustness 或 speedup 数字迁移为 BOST 预测。

**仍可能成立的差异化命题（候选，不是结果）**

1. LGWO 的 correction 不是任意最终场 correction，而是嵌入已经封存的 `24F/24A^T` solver shell 的第一方向；部署时只改变首个有限预算方向，后续 CGLS/data-consistency path 保留。
2. BOST 具备一个 NCO 原文没有的实际压力：multi-view ray geometry、support、稀疏/不均匀投影和 held-out B rays。LGWO 可以把“粗解质量 + correction 后 field/H1 + measured/B residual”作为一个多门证据链，而不是只看 image metric。
3. 可以比较 three-way：`direct inverse operator`、`rough+correction`、`budget-preserving direction proposal`。只有第三者在同等 operator calls 下胜出，才有理由把预算约束写进论文题目。

**必须加入的强基线**

- zero-start CGLS-24 与 anchored `delta=0` parity control。
- `NCO-style`：用相同 `A`、相同 measurement 产生固定步数粗解，再由小网络直接预测 field correction；外层必须重新通过同一 BOST evaluator，不能只比较 field output。
- direct field predictor：DeepONet、FNO/FFNO 或小型 3D CNN；输入和输出 contract 要公开，训练/推理成本单独记账。
- rough-only、correction-only、rough+correction 三个消融，避免把粗解本身的改善误记为 neural correction 的改善。

### 3.3 Lan et al.：神经预条件器不是普通 direction head

**文献**
Kai Weixian Lan, Elias Gueidon, Ayano Kaneda, Julian Panetta, Joseph Teran, “A Neural-Preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions.”
**正式会议：** ICML 2024, PMLR 235:25976-25994.
**一级链接：** [PMLR official record](https://proceedings.mlr.press/v235/lan24a.html) · [official PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/lan24a/lan24a.pdf) · [ICML poster](https://icml.cc/virtual/2024/poster/33924) · [作者代码/预训练模型入口（论文页）](https://github.com/kai-lan/MLPCG/tree/icml2024)
**DOI：** PMLR official record 未列 DOI；引用 PMLR record。

**它已经解决/主张了什么**

- 针对混合 Dirichlet/Neumann boundary condition 的 Poisson 线性系统，训练轻量 spatially varying convolution 网络近似离散 Laplacian inverse，作为迭代 solver 的 neural preconditioner。
- 目标不是一次性输出最终 field，而是与外层 iterative solver 配合；论文报告对 domain shape、boundary condition、grid size 的一定泛化，并与 algebraic multigrid 等方法比较。
- 其贡献依赖 Poisson/Laplacian 结构、外层 solver 和 preconditioner 的配合，不能直接等价为 BOST inverse reconstruction。

**LGWO-A24 不得再宣称**

- “首次将 neural network 放入迭代 solver 作为预条件器”。
- “任何 learned first direction 都是 preconditioner”。严格称为 preconditioner 需要说明它如何作用于残差/线性系统，及外层 Krylov/FCG 的合同。
- 把 Poisson solver 的 grid/domain generalization 当作 BOST camera/rig generalization。

**仍可能成立的差异化命题（候选，不是结果）**

- LGWO 不试图学习一个通用 `A^{-1}`，而是把网络限制为低参数、一次调用、有限预算 first-direction proposal；这是更窄、更容易做安全审计的对象。
- 可以研究“BOST geometry-conditioned direction”与“neural preconditioner”在相同 `F/A^T` 调用、相同壁钟成本和相同 field/B residual 门下的差异。若没有严格成本账本，不能说谁更高效。

**必须加入的强基线**

- Lan-style preconditioner 适配到简化线性 BOST operator 的 toy benchmark，至少比较 residual decay、A/A^T calls、wall time 和 final field error。
- classical Jacobi/diagonal、Tikhonov、CG/CGLS/LSQR、以及不带网络的 fixed preconditioner。
- 若不能公平实现完整 Lan architecture，应将其列为 literature comparator/设计启发，不伪装成已经复现的强基线。

### 3.4 Li et al.：学习矩阵分解/预条件的可比范式

**文献**
Yichen Li, Peter Yichen Chen, Tao Du, Wojciech Matusik, “Learning Preconditioners for Conjugate Gradient PDE Solvers.”
**正式会议：** ICML 2023, PMLR 202:19425-19439.
**一级链接：** [PMLR official record](https://proceedings.mlr.press/v202/li23e.html) · [official PDF](https://proceedings.mlr.press/v202/li23e/li23e.pdf)
**DOI：** PMLR official record 未列 DOI；引用 PMLR record。

**它已经解决/主张了什么**

- 用 GNN 学习 approximate matrix factorization，并把它作为 PCG 的 preconditioner。
- 设计与 PDE data distribution 相关的 preconditioner loss；实验覆盖多种 2D/3D second-order linear PDE。
- 关键评价是 PCG 收敛、预条件效果和跨 PDE/网格泛化，不是直接 field image 的视觉质量。

**LGWO-A24 不得再宣称**

- “把 field reconstruction loss 加到网络上就等于学会了 preconditioner”。
- “在一个 12^3 toy BOST 上获得 field-L2 改善，就等于在 3D PDE/PDE solver 层面证明了泛化”。
- 不得省略 operator call、迭代数、预条件 setup 和 inference cost。

**仍可能成立的差异化命题（候选，不是结果）**

- LGWO 的研究对象更接近 measurement-to-field inverse 的 first update；Li et al. 的对象是可作用于 PCG 的 approximate factorization。若 LGWO 不能在 residual spectrum、iterations 或 cost 上改善，就应避免使用 preconditioner 语言。
- 可以从 Li et al. 借鉴“以 solver metric 训练”而不是只以 field loss 训练：例如将 `||A d||`、field error surrogate、下一步 residual reduction 和 B-ray stability 组成 multi-objective，但每一项都需固定权重和验证。

### 3.5 FCG-NO：神经 operator 作为 flexible CG 预条件器

**文献**
Alexander Rudikov, Vladimir Fanaskov, Ekaterina Muravleva, Yuri M. Laevsky, Ivan Oseledets, “Neural operators meet conjugate gradients: The FCG-NO method for efficient PDE solving.”
**正式会议：** ICML 2024, PMLR 235:42766-42782.
**一级链接：** [PMLR official record](https://proceedings.mlr.press/v235/rudikov24a.html) · [official PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/rudikov24a/rudikov24a.pdf)
**DOI：** PMLR official record 未列 DOI；引用 PMLR record。

**它已经解决/主张了什么**

- 将 discretization-invariant neural operator 用作 flexible conjugate gradient 的 nonlinear preconditioner。
- 利用 FCG 理论允许非线性 preconditioner，并研究不同分辨率复用、loss 与 training data collection。
- 这条线已经把“operator learning + Krylov/CG + resolution transfer”组合成正式研究对象。

**LGWO-A24 不得再宣称**

- “首次把 operator learning 和 Krylov 结合”。
- “模型的第一方向来自 Krylov，因此就是 FCG-NO”。若网络不作为外层 FCG 的 nonlinear preconditioner，就应使用更窄的术语。
- 不得用单一 final field-L2 取代 solver convergence、iterations、call count 和 resolution transfer 评估。

**仍可能成立的差异化命题（候选，不是结果）**

- LGWO 可以把问题从“学习整套 inverse operator/预条件器”收窄为“在固定 24-step BOST shell 中学习一个受约束 direction”，以验证小模型是否在 rig shift 下有更好的 cost-tail tradeoff。
- 若未来实验扩展到 FCG，必须另立协议，不可把当前 `24F/24A^T` shell 直接宣传为 flexible CG 理论保证。

**必须加入的强基线**

- 同一 BOST toy operator 上的 FCG-NO-style preconditioner 或至少 documented proxy。
- report：初始/逐步 residual、field/H1、A/A^T calls、per-case tail、wall time、memory、resolution/geometry shift。
- 如果实现成本过高，应在论文中诚实写成“未复现的相关工作”，并说明为什么不纳入数值表。

### 3.6 iFNO：反向 operator learning 的直接竞争者

**文献**
Da Long, Zhitong Xu, Qiwei Yuan, Yin Yang, Shandian Zhe, “Invertible Fourier Neural Operators for Tackling Both Forward and Inverse Problems.”
**正式会议：** AISTATS 2025, PMLR 258:3043-3051.
**一级链接：** [PMLR official record](https://proceedings.mlr.press/v258/long25a.html) · [official PDF](https://raw.githubusercontent.com/mlresearch/v258/main/assets/long25a/long25a.pdf) · [official code](https://github.com/BayesianAIGroup/iFNO)
**DOI：** PMLR official record 未列 DOI；引用 PMLR record。

**它已经解决/主张了什么**

- 针对 FNO 主要用于 forward、inverse problem 需要反向映射的问题，提出在 latent channel space 中共享参数并交换信息的 invertible Fourier blocks。
- 与 VAE 结合以处理 ill-posedness、data shortage 和 noise，并在七个 forward/inverse benchmark tasks 上评估。
- 这是直接 inverse neural operator 的高相关一级基线；即使任务不是 BOST，也不能只比较一个未经调参的普通 FNO 就宣称击败现有 operator。

**LGWO-A24 不得再宣称**

- “首次把 FNO 改成可用于 inverse reconstruction”。
- “小型方向 head 超过 FNO 就证明 neural operator 失效”。必须控制参数量、训练数据、forward calls、分辨率和 inverse task contract。
- 不得把 iFNO 的 VAE uncertainty/ill-posedness 机制当作 LGWO 已经具备的不确定性估计。

**仍可能成立的差异化命题（候选，不是结果）**

- LGWO 的候选优势不是全面取代 iFNO，而是：在低数据、固定 optical operator、严格 data-consistency 和有限调用预算下，方向修正是否比直接从 measurement 生成全 field 更稳。
- 如果最终任务尺寸不适合 iFNO，应报告工程限制，并用小型 direct FNO/DeepONet/3D CNN 做 matched-budget baseline；不得只选择不利的实现。

### 3.7 NIO：DeepONet/FNO 组合已用于 PDE inverse map

**文献**
Roberto Molinaro, Yunan Yang, Björn Engquist, Siddhartha Mishra, “Neural Inverse Operators for Solving PDE Inverse Problems.”
**正式会议：** ICML 2023, PMLR 202:25105-25139.
**一级链接：** [PMLR official record](https://proceedings.mlr.press/v202/molinaro23a.html) · [official PDF](https://proceedings.mlr.press/v202/molinaro23a/molinaro23a.pdf)
**DOI：** PMLR official record 未列 DOI；引用 PMLR record。

**它已经解决/主张了什么**

- 指出很多 PDE inverse problems 本质上是 operator-to-function，而常规 neural operator 主要是 function-to-function，因此提出由 DeepONet 与 FNO 组合的 Neural Inverse Operator。
- 以 inverse PDE mapping 为对象，报告与 direct/PDE-constrained optimization 等方法的性能和速度比较。

**LGWO-A24 不得再宣称**

- “DeepONet/FNO 不能处理 inverse BOST”。它们可以作为合理 baseline；真正的问题是如何把 BOST operator、geometry 和 data consistency 纳入比较。
- “把输入加上 geometry 就是新 inverse operator”。geometry-conditioning 本身不够构成新颖性。

**仍可能成立的差异化命题（候选，不是结果）**

- LGWO 可以提出一种与 direct NIO 不同的 protocol-level object：只学习低维 first direction 并由显式 optical shell 纠错；但必须以 NIO/DeepONet/FNO matched-budget 的 field、reprojection、tail 和 cost 结果支持。

### 3.8 Learned preconditioners 与 Krylov 的更近邻

#### Learning incomplete factorization preconditioners for GMRES

Paul Hausner, Aleix Nieto Juscafresa, Jens Sjolund, “Learning incomplete factorization preconditioners for GMRES.” NLDL 2025, PMLR 265:85-99.
**一级链接：** [PMLR official record](https://proceedings.mlr.press/v265/hausner25a.html) · [official PDF](https://raw.githubusercontent.com/mlresearch/v265/main/assets/hausner25a/hausner25a.pdf) · [official code](https://github.com/paulhausner/neural-incomplete-factorization)
**DOI：** PMLR official record 未列 DOI。

该文学习可逆的 incomplete factorization，并把它用于 GMRES preconditioning；其输出激活、稀疏结构和 spectral/iteration evaluation 是 LGWO 做安全网络设计时的邻居。它并不证明 LGWO 的 BOST direction，也不应被写成 BOST baseline 的已复现结果。

#### On Krylov Methods for Large Scale CBCT Reconstruction

Malena Sabate Landman et al., “On Krylov Methods for Large Scale CBCT Reconstruction.”
**来源状态：** arXiv preprint，`arXiv:2211.14212`；论文提供 TIGRE/open-source framework 方向和 CGLS、LSQR、LSMR、Tikhonov、TV 等 3D CT Krylov comparison。
**一级链接：** [arXiv abstract/full text](https://arxiv.org/abs/2211.14212) · [TIGRE toolbox](https://github.com/CERN/TIGRE)

它的意义不是 neural novelty，而是提醒 LGWO 的 3D inverse baseline 必须包含成熟的 Krylov family，并按 operator calls、regularization、3D reconstruction quality 和 runtime 比较。LGWO 不得把一个自定义 24-step CGLS 与弱化的 classical baseline 比较后称为普适胜出。

#### Learned Cone-Beam CT Reconstruction Using Neural Ordinary Differential Equations

Mareike Thies et al., “Learned Cone-Beam CT Reconstruction Using Neural Ordinary Differential Equations.”
**来源状态：** arXiv preprint，`arXiv:2201.07562`；按预印本处理。
**一级链接：** [arXiv](https://arxiv.org/abs/2201.07562)

它说明 learned iterative reconstruction 可以把物理 consistency 与 memory-efficient neural update 结合到 3D cone-beam inverse problem。它不是 BOST 结果，但在“如何比较 3D learned iterative reconstruction 与 classical solver、如何控制 GPU memory”上是重要邻居。

## 4. LGWO-A24 的新颖性边界矩阵

| 表述 | 状态 | 原因/证据 | 需要什么才能重新讨论 |
|---|---|---|---|
| “首次学习 null-space” | **禁止** | NPN 已明确学习 sensing-matrix null-space 的低维投影/系数 | 不能通过改名规避；最多主张 BOST-specific budgeted direction |
| “首次粗解加神经校正” | **禁止** | NCO 已正式发表并在 EIT 中比较 direct neural operator | 只能研究 BOST-specific shell、call budget 和 tail safety |
| “首次 neural operator + Krylov” | **禁止** | FCG-NO、Lan、Li 已覆盖 neural preconditioner + CG/FCG | 必须提出严格不同的 solver object，且给出 matched baseline |
| “首次 inverse DeepONet/FNO” | **禁止** | NIO、iFNO 已是直接 inverse operator 先例 | 只能把 direct operator 作为基线，不可否定其存在 |
| “几何条件化” | **仅候选机制** | geometry inputs 是合理设计，但不是单独创新 | 需证明 geometry shift 下比 no-geometry、fixed prior 和 direct baseline 更稳定 |
| “24F/24AT 固定预算” | **协议差异候选** | 当前 L1 已封存严格 operator ledger | 需报告与所有基线相同的 deployed calls、wall time 和 memory |
| “BOST multi-view displacement 到 3D field 的 first-direction proposal” | **最有希望的任务差异** | 文献邻居多为 CT/EIT/PDE，而非 OERF BOST contract | 需独立 BOST/renderer、held-out rig、真实数据或明确限制 |
| “fail-closed fallback + B-ray tail gate” | **方法学贡献候选** | 相关工作通常不共享当前这套 evidence contract | 需定义安全规则、失败率、拒答质量和成本，且不能只报告均值 |
| “比 DeepONet/FNO/FFNO 更好” | **尚未可说** | 当前尚未训练/比较，且不同任务/预算不可直接比较 | 预注册 matched data/operator/cost/metric，并报告全部 seeds、tail 和失败案例 |
| “算法突破” | **当前否定** | 目前只有 implementation gate 和训练基础设施 | 只有独立 route/fresh/真实数据的可复现实验才可讨论突破性，通常仍需同行评议 |

## 5. 可保留的候选算法命题

以下不是结果，只是按“最小实现 -> 逐层增加”的候选路线。每一项都必须用相同数据切分、相同 operator budget 和同一结果模板测试。

### H1：Budget-Preserving Learned First Direction

**形式：**

```text
x_24 = CGLS_24(A, y)
q_theta = proposal_theta(raw_BOS, A^T y, geometry, support)
x_hat = data_consistency_shell_24(x_24, q_theta)
```

网络只输出一个有限 support 内 direction proposal；它不能访问 truth、B data、SVD 或 evaluator。`delta=0` 必须逐元素回到 CGLS-24。该命题的核心不是“null-space”，而是“在固定部署预算内，学习方向能否降低 field/H1 而不损害 measurement/B-ray 一致性”。

**最低强证据：** full、fixed-direction、g-only、no-raw-y、no-geometry；A/B residual；field/H1；逐 cluster tail；A/A^T 账本；fallback/clipping；独立 process replay。

### H2：NPN-style Low-dimensional Null-space Coefficient Baseline

先构造只用于 fit 的低维 projection/representation，再预测系数；要报告 orthogonality defect 和 projection rank。它的主要作用不是挑战 NPN，而是判断 LGWO 的优势是否只是“一个低维 prior 就够了”。

**判别：** 如果 NPN-style baseline 与 full LGWO 持平，LGWO 的 geometry-conditioned architecture 可能没有必要；如果 full 只在同一 synthetic family 上更好，不能称泛化。

### H3：NCO-style Rough-plus-Correction Baseline

固定 `R_K` 产生粗解，网络预测 field correction，最后强制通过同一 evaluator。它能回答“LGWO 的 first-direction constraint 是否真的比直接 field correction 更有价值”。

**风险：** 若 NCO-style baseline 使用更多 forward calls 或直接访问更高维 field，必须拆分 equal-budget 与 expanded-budget 两张表；不能把不同预算的结果混在主结论。

### H4：Neural Preconditioner / FCG-NO Proxy

在小 toy operator 上实现 documented proxy，学习对 residual 的 preconditioner 或 flexible-CG update；将 residual decay、iterations、field error 与 runtime 放在同一表。它是解释“direction proposal 与 preconditioning 的本质差异”的诊断，不应为了凑 baseline 而写成未复现论文。

### H5：Geometry-shift safety model

把 geometry cluster、camera layout、ray density、support shape 和 noise level 分层；训练只用固定 clusters，route 只解封未见 clusters。主要 outcome 不是 mean gain，而是 cluster worst-case、harm rate、B clean residual 和拒答率。

**只有当 H1-H5 全部有预注册数据和结果时，才可以讨论一篇高质量论文的核心叙事。** 在此之前，最合适的论文题目仍应保持保守，例如“Budget-preserving learned direction proposals for geometry-conditioned BOST inverse reconstruction: a reproducibility-first pilot”，而不是“universal neural operator”或“null-space theorem”。

## 6. 必须冻结的强基线矩阵

### 6.1 Classical inverse / Krylov

1. zero-start CGLS-24，主同预算基线。
2. LSQR 或 LSMR，统一 stopping/iteration budget；用于检查 CGLS choice 是否给了 LGWO 不公平优势。
3. Tikhonov/Levenberg-Marquardt 或 ridge-filtered baseline；正则系数只能由 fit/development 选择。
4. SIRT/TV/PDHG/ADMM 中至少一种成熟正则化 baseline；若不实现，应记录为未纳入原因。
5. fixed direction：同样的 support、loss、clipping 和 deployable budget，验证网络的条件化是否有增益。

### 6.2 Learned inverse / correction

1. 3D CNN 或 U-Net direct field predictor：最小工程基线。
2. DeepONet：branch 输入 measurement/geometry，trunk 输入 voxel coordinates；报告参数量和训练样本数。
3. FNO/FFNO：若输入离散 ray geometry 不适配标准规则网格，必须说明 lifting/projection 和插值，不得拿不匹配实现当弱基线。
4. iFNO/NIO-style inverse operator：至少选择一个可运行的 official-code-inspired baseline，明确它是 adapted implementation 还是 exact reproduction。
5. NCO-style rough-plus-correction：粗解、corrector、combined 三者都要有。

### 6.3 Solver-aware learned baseline

1. Lan-style neural Poisson preconditioner proxy。
2. Li-style learned factorization/graph preconditioner proxy。
3. FCG-NO-style nonlinear preconditioner proxy。
4. 若资源不足，保留完整文献中的方法作为 related work，不假装已复现；主实验至少要有 `CGLS/LSQR + direct neural + NCO-style + LGWO`。

### 6.4 所有方法共用的指标

| 层级 | 必须报告 |
|---|---|
| field | relative-L2、H1/semi-norm、per-case min/median/mean/max |
| measurement | A measured/clean residual ratio、heldout-B clean reprojection、`||A d||`、Schur/orthogonality defect |
| robustness | geometry cluster、family、noise、support、camera/ray shift、harm rate、fallback、non-finite、clipping |
| solver | A/A^T calls、iteration count、residual trajectory、query count、exact/approx operator mode |
| engineering | wall time、CPU time、peak RSS/VRAM、checkpoint size、device fallback、fit cost 与 deployment cost 分开 |
| statistics | fixed seeds、cluster-level aggregation、bootstrap/CI、全部失败案例，不删负结果 |

只报告 pooled mean 不足以支持 BOST 算法论文；最坏 cluster、逐 rig 尾部和失败率必须在主表或补充表中出现。

## 7. 推荐阅读顺序与每篇提取问题

### 第一轮：先读协议最相关的四篇

1. **NPN**：画出 `A`、`S`、projection、network coefficient 和 PnP 外层的关系；写出它与 LGWO 的差异表；标注哪些 theorem assumption 不能用于 BOST。
2. **NCO**：画出 `R_K` 与 `C` 的计算图；列出粗解、corrector、direct neural、long-iteration baseline 的 calls 和指标。
3. **FCG-NO**：理解 nonlinear preconditioner 为什么需要 FCG；区分“预条件器”“更新方向”“最终 field predictor”。
4. **Lan**：提取 boundary condition、domain/grid shift、preconditioner loss、outer solver 和 benchmark cost；不要只看最终误差。

### 第二轮：建立 direct inverse/operator baseline 视野

5. **NIO**：理解 operator-to-function inverse 与 DeepONet/FNO composition，做 BOST branch/trunk 输入草图。
6. **iFNO**：提取 invertible block、VAE、forward/inverse joint training 和七个 benchmark 的训练/测试切分。
7. **Li et al.**：提取 learned factorization、PCG loss 和 2D/3D PDE generalization。
8. **Hausner et al.**：提取 invertible factorization activation、sparsity 和 GMRES metric。

### 第三轮：补 3D inverse/Krylov 评价习惯

9. **On Krylov Methods for Large Scale CBCT Reconstruction**：把 CGLS/LSQR/LSMR/Tikhonov/TV 指标整理成自己的 baseline checklist。
10. **Learned Cone-Beam CT Reconstruction Using Neural ODEs**：学习 3D memory budget、learned iterative consistency 和 sparse-view evaluation 的写法。

每篇读完必须产出四行笔记：

```text
已解决的物理/数值困难：
网络实际输入和输出：
外层 solver / data-consistency / evaluation：
对 LGWO-A24 新颖性边界的影响：
```

## 8. 给师兄的真实审阅问题

在打开正式 fit 之前，建议向何远哲师兄确认：

1. 你们实际的 BOST inverse operator 是 displacement-to-refractive-index、displacement-to-density，还是先重建 `grad n` 再积分？不同 target 会改变 direct operator 和 loss。
2. 现有实验数据是否有独立的 flow-off repeats、相机内外参、pixel-to-physical scale、view/camera pose、mask 和 synchronisation log？没有的话，field-L2 不能作为真实数据主指标。
3. 真实实验中最痛的瓶颈是少视角、光学标定 drift、PIV refractive correction、时间分辨率、噪声，还是 3D network memory？算法创新必须对准一个真实瓶颈。
4. 师兄更希望“提升 field fidelity”还是“减少 A/A^T calls / reconstruction time”？这两种论文的 baseline 与 loss 不同。
5. 是否已经有 lab 内部的 CGLS/SIRT/NeRIF/TDBOST implementation？若有，必须冻结 commit、版本、输入预处理和原始评估脚本，避免自己重写一个不等价 baseline。
6. 实验室是否允许在本机使用合成数据、公开数据和派生图；哪些数据只能在内部环境读取，哪些结果能写进论文/上传私有仓库？

## 9. 当前可写与不可写的摘要句

### 现在可以写

> 我们完成了一个训练前文献边界审计。近期一级来源已经分别覆盖 null-space projection、粗解加 correction、neural preconditioning、FCG-NO 和 inverse DeepONet/FNO。因此 LGWO-A24 不把这些通用范式本身作为创新，而把研究问题收窄为：在 BOST-specific geometry 和固定 `24F/24A^T` 部署预算下，一个小型 first-direction proposal 是否能在严格 data-consistency 与逐 cluster 尾部门下获得可复现改善。

### 现在不能写

- “我们提出了首个 null-space neural operator。”
- “我们证明网络学习到了真实 optical null space。”
- “我们已经优于 DeepONet、FNO、FFNO、NCO 或 NPN。”
- “我们已经证明跨 rig/跨 renderer/真实 OERF 泛化。”
- “我们取得突破或已有论文级性能。”

## 10. 证据状态与下一步门

当前证据只到：

- **已完成：** 一级来源检索、文献重叠审计、LGWO 新颖性边界、强基线矩阵、训练前阅读顺序。
- **已完成但仅属工程证据：** LGWO-A24 L1 implementation gate、CPU float64、完整 model seed 固定、24F/24A^T shell、五个 trainable arms、正式 loss、fit-only manifest、无 pickle checkpoint/resume contract。
- **尚未完成：** fit data materialization、任何训练 loss 曲线、early-stop eligible checkpoint、route-development、fresh-OOD、独立 renderer、真实 BOST、DeepONet/FNO/FFNO/NPN/NCO 对比。
- **突破监测：** `NO_ALGORITHM_BREAKTHROUGH`。

下一步只有在 fit manifest 和最终训练合同冻结后，才打开 fit-only data cache。fit 结果若不能通过 non-finite、residual、clipping、checkpoint 和 cost gates，route 保持关闭；若 fit 通过，也只能进入预声明的 synthetic route-development，不得把 fit gain 写成泛化或真实物理成功。

## 参考链接总表

- [NPN: arXiv 2510.01608](https://arxiv.org/abs/2510.01608)
- [Neural Correction Operator: JCP DOI 10.1016/j.jcp.2026.115039](https://doi.org/10.1016/j.jcp.2026.115039)
- [Neural Correction Operator: arXiv 2507.18875](https://arxiv.org/abs/2507.18875)
- [Lan et al., ICML 2024, PMLR 235](https://proceedings.mlr.press/v235/lan24a.html)
- [Li et al., ICML 2023, PMLR 202](https://proceedings.mlr.press/v202/li23e.html)
- [Rudikov et al., FCG-NO, ICML 2024](https://proceedings.mlr.press/v235/rudikov24a.html)
- [Long et al., iFNO, AISTATS 2025](https://proceedings.mlr.press/v258/long25a.html)
- [Molinaro et al., Neural Inverse Operators, ICML 2023](https://proceedings.mlr.press/v202/molinaro23a.html)
- [Hausner et al., learned incomplete factorization, NLDL 2025](https://proceedings.mlr.press/v265/hausner25a.html)
- [Sabate Landman et al., Krylov methods for large-scale CBCT](https://arxiv.org/abs/2211.14212)
- [Thies et al., learned cone-beam CT reconstruction with neural ODEs](https://arxiv.org/abs/2201.07562)

**最终声明：** 本文是一级来源与创新边界审计，不是论文结果。所有“差异化命题”都必须在真实可复现实验后重新审查；若数据或 route 失败，保留失败并收窄结论，不通过换措辞制造新颖性。
