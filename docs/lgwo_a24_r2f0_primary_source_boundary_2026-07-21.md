# LGWO-A24 R2-F0 一级来源与新颖性边界：有界 span-expressivity audit

- 审计日期：2026-07-21
- 文献检索截止：2026-07-21
- 直接目的：为 R2-F0 的运行前协议提供来源边界，不产生算法结果
- 审计对象：冻结 H1-20 span 后加入三类 truth-free 方向，并与 matched H1-21/22/23 比较
- 来源优先级：出版社正式页面、会议/PMLR 正式页面、作者 arXiv 原文；课题组主页仅用于 accepted 论文元数据
- 当前证据：R2-E0 的 6 个 post-open synthetic mechanism cases；`0 fresh / 0 real BOST`

> **强制声明：本文是来源与协议审计，不是实验结果、系统综述或专利检索。**
>
> 文中复述的任何速度、精度、泛化或重建效果都是论文作者对其自身实验的报告，不能转写为
> R2-F0 的证据。R2-F0 尚未运行；本文也不授权真实 BOST、fresh、算法优越性、泛化、论文成功或突破主张。

## 0. 审计结论

### 0.1 宽泛组合已经不新

以下表述均被一级来源直接覆盖，不能作为 R2-F0 的新颖性主张：

1. 把 residual backprojection 加入 Krylov/投影空间；
2. 用 TV、Huber、IRLS 或 lagged diffusivity 产生 edge-preserving 方向；
3. 用先验或非线性/迭代变化的预条件器改变 Krylov span；
4. 把多个预条件方向合并进同一搜索空间；
5. 缓存、扩充、压缩或回收 Krylov 子空间后在低维空间重优化；
6. 学习搜索方向、预条件器、逆算子作用或不变子空间；
7. 用神经场、密度/折射率梯度、非线性 ray tracing 或张量分解做三维/四维 BOST。

所以，`raw + Huber + edge-gated backprojection + Krylov` 只是一个合理的**机制实验组合**，不是组件级发明。

### 0.2 仍可能成立的窄命题

本次有界检索没有找到与下述完整协议同构的 BOST 工作，但“未找到”不是新颖性证明：

> 在固定 BOST 物理算子、固定 H1-20 成本和冻结方向定义下，先用 truth-free 的数据残差、Huber
> 先验和 ray/edge-conditioned 非线性预条件方向做**表示余量审计**；候选扩展 span 与继续运行
> H1 得到的 matched span 接受完全对称的 truth-only Pareto oracle、data envelope、调用账本和
> 逐 rig 尾部门。只有先证明 BOST 可观测方向提供 matched-H1 无法解释的联合 field/gradient
> headroom，才允许学习方向生成器，并在 curved/straight 模型失配和未见 rig 上测试拒答。

这个命题当前只能标为：

```text
BROAD_COMPONENT_NOVELTY = REJECTED
BOUNDED_BOST_PROTOCOL_DIFFERENTIATION = PLAUSIBLE_BUT_UNPROVEN
R2F0_RESULT = NOT_RUN
REAL_BOST_EVIDENCE = 0
BREAKTHROUGH = false
```

### 0.3 R2-F0 的正确角色

R2-F0 不是“提出算法并证明更好”，而是一个便宜、可失败的 representation audit：

- 若新增方向在 matched H1-21/22/23 的对称 oracle 下仍无联合余量，停止该方向族；
- 若只有 unrestricted truth oracle 好、压回 data envelope 后消失，判定不可部署；
- 若新增方向与下一条 H1 方向数值共线，判定没有扩展 span；
- 若有稳定 matched headroom，也只授权运行前冻结 fresh 协议，不授权训练或论文成功。

## 1. 被审计对象的精确定义

### 1.1 公共 H1-20 状态

在固定加权线性化 BOST proxy 中记：

```math
A_w x \simeq y_w,
```

`P_20=[p_1,...,p_20]` 是冻结 H1-20 已支付得到的场方向，`x_20=P_20 c_20`，

```math
r_{20}=y_w-A_wx_{20}, \qquad g_d=A_w^T r_{20}.
```

定义公共 span：

```math
\mathcal S_{20}=\operatorname{span}(P_{20}).
```

R2-F0 必须复用与 R2-E0 完全相同的 H1 参数、whitening、support、denominator floor、geometry digest、
方向缓存和 `20F/20A^T` 公共起点。若这些对象变化，就不是从 R2-E0 继续的 span audit。

### 1.2 三类方向只能读部署可见量

三类 raw 方向都在同一个冻结快照 `x_20,r_20` 上生成。不得先用 truth oracle 更新 `x`，再从更新后的
`x` 生成下一方向。

#### D1：raw residual backprojection

```math
d_r=A_w^T r_{20}.
```

这是平方数据项的负梯度方向，不是新算法。若 H1 recurrence 的下一步本来就使用同一梯度，`d_r` 可能与
H1-21 新方向相同或近共线；必须让数值 rank audit 决定，不能仅凭名字说它“跳出 H1 span”。

#### D2：Huber prior direction

对冻结空间差分算子 `D`、冻结阈值 `delta`，定义 isotropic Huber 导数 `psi_delta`，候选为：

```math
d_h=-D^T\psi_\delta(Dx_{20}).
```

`delta`、边界条件、voxel spacing、support 和 Huber 归一化必须在访问 truth 前绑定。该方向可以扩展
`S_20`，但 TV/Huber 先验梯度及其迭代更新已有长期文献基础；扩展成功不等于方法新颖。

#### D3：edge-gated / nonlinear-preconditioned backprojection

严格协议必须在运行前从下面两种中**选择一种**，不能看 oracle 后择优：

```math
\text{cheap gate:}\quad d_e=G_\delta(x_{20},m,\Sigma,\mathcal G)g_d,
```

或

```math
\text{lagged-diffusivity priorconditioner:}\quad
d_e=M_\delta(x_{20})^{-1}g_d,
\qquad M_\delta=\tau I+D^TW_\delta(x_{20})D.
```

其中 `m` 只能是 view/support mask，`Sigma` 只能是预先估计的 measurement covariance，`G` 只能是相机/
ray coverage 摘要。`G_delta`、`W_delta`、`tau`、内部求解容差和最大内部迭代均不得读 truth。cheap gate
和 priorconditioner 的数学、成本与稳定性不同；若两者都测，必须登记为两个独立方向族而不是合并名称。

### 1.3 先证明“新方向”，再比较重建

`P_20` 不应被假定已正交。应在 CPU float64 中以注册的 QR/SVD 规则做增量正交化：

```math
q_r=\operatorname{orth}(d_r;\mathcal S_{20}),
```

```math
q_h=\operatorname{orth}(d_h;\mathcal S_{20}\oplus\operatorname{span}\{q_r\}),
```

```math
q_e=\operatorname{orth}(d_e;\mathcal S_{20}\oplus\operatorname{span}\{q_r,q_h\}).
```

rank tolerance 必须由 dtype、基条件数和预写规则决定，不能由 field gain 决定。每个方向至少报告：

1. 正交化前后范数比；
2. 与 `S_20` 及 matched H1 新方向的 principal angle；
3. `||A_w q||` 及新增 projected matrix 的奇异值；
4. 两遍正交后的最大正交缺陷；
5. 非有限、低秩或近共线时的 fail-closed 状态。

field-space 独立但 `A_wq` 近零的方向不能简单删除：它可能揭示低可观测余量。但其系数无法由 active data
稳定识别，因此只能保留为 oracle 诊断；在出现 truth-free 选择规则前不能成为部署候选。

## 2. 必须对称的 span 比较

### 2.1 候选与 matched H1

按用户指定顺序形成嵌套候选：

```math
\mathcal F_1=\mathcal S_{20}\oplus\operatorname{span}\{q_r\},
```

```math
\mathcal F_2=\mathcal F_1\oplus\operatorname{span}\{q_h\},
```

```math
\mathcal F_3=\mathcal F_2\oplus\operatorname{span}\{q_e\}.
```

matched classical spans 为：

```math
\mathcal K_k=\operatorname{span}\{p_1,\ldots,p_{20+k}\},\qquad k=1,2,3.
```

因此必做比较是：

| 层级 | R2-F0 span | dimension-matched classical span |
|---|---|---|
| 1 | `F1 = H1-20 + raw` | `K1 = H1-21` |
| 2 | `F2 = H1-20 + raw + Huber` | `K2 = H1-22` |
| 3 | `F3 = H1-20 + raw + Huber + edge` | `K3 = H1-23` |

还需报告三个 singleton ablation：`S20+raw`、`S20+Huber`、`S20+edge`。嵌套顺序的增量不能替代 singleton，
否则第二、第三方向的价值会被前序正交化混淆。

### 2.2 最关键的公平性：oracle 对 oracle

R2-F0 的问题是 **span expressivity**，所以主比较必须是：

```text
truth-only oracle(Fk)  vs  truth-only oracle(Kk)
```

而不是：

```text
truth-only oracle(Fk)  vs  ordinary H1-(20+k) iterate
```

后者把“低维系数重优化”的优势错误归给新增方向。ordinary H1-21/22/23 仍应报告，但角色只是实际 solver
trajectory；每个 `K_k` 必须接受与 `F_k` 完全相同的 coefficient optimizer、residual envelope、field/
gradient evaluator、容差和失败规则。

### 2.3 dimension-matched 不自动等于 call-matched

公共 H1-20 成本固定为 `20F/20A^T`。之后必须分别记录：

- 每个 `q` 的 `A_wq` 投影通常支付一次 `F`；
- `g_d=A_w^Tr_20` 若已经是冻结 solver state，不能重复计费，也不能隐瞒其来源；
- Huber prior gradient 的 `D/D^T` 不是物理 `A/A^T`，但必须计 wall time 与内存；
- nonlinear priorconditioner 的内部迭代、multigrid 或 factorization 必须单列 setup/apply 成本；
- H1-21/22/23 通常分别再支付 `1/2/3 F` 与 `1/2/3 A^T`。

因此 `F_k` 对 `K_k` 首先是**维数匹配**。只有实际账本相等时才能写“调用匹配”；否则应同时报告：

1. dimension-matched 质量；
2. 不超过相同 `F/A^T` cap 的质量；
3. 端到端 wall time、峰值内存和 setup/apply 成本。

禁止为了表面相等而故意重复无用的 `A^T`，也禁止把 expensive inner solve 当作免费先验。

## 3. truth-only oracle 的允许用途

### 3.1 指标与约束

truth `x_*` 只进入 evaluator。对任意 span `S` 中的 `x(c)`，报告：

```math
E_f(c)=\frac{\|x(c)-x_*\|_2}{\|x_*\|_2},
```

```math
E_g(c)=\frac{\|D(x(c)-x_*)\|_2}{\|Dx_*\|_2},
```

以及：

```math
\rho_A(c)=\frac{\|y_w-A_wx(c)\|_2}{\|y_w-A_wx_{20}\|_2}.
```

若继承 R2-E0，residual-safe envelope 保持 `rho_A <= 1.02`；任何变更必须在运行前另行说明。gradient
denominator floor、front-band metric、held-out-B metric 也必须复用冻结定义。

主结果应是 `rho_A<=1.02` 下完整的 `(E_f,E_g)` Pareto frontier。若需要一个排序标量，可预写：

```math
J_\infty(c)=\max\left(\frac{E_f(c)}{E_f(x_{20})},
                       \frac{E_g(c)}{E_g(x_{20})}\right),
```

但不能在看到结果后改变 field/gradient 权重。R2-E0 已使用的 `5% field / 5% gradient` 联合门可作为
连续性门；R2-F0 还必须相对 `oracle(K_k)` 同时报告 matched gain，而不只相对较弱的 H1-20。

### 3.2 oracle 不可部署

oracle 只能回答：**这个冻结 span 是否存在一个值得继续寻找 truth-free 选择规则的点？** 它不能：

- 给当前 6 个 case 生成训练标签后再在同 6 case 上证明成功；
- 选择 Huber `delta`、edge gate、方向顺序、rank threshold 或 residual envelope；
- 证明 active data 能识别 oracle coefficient；
- 证明真实 BOST、未见 rig 或曲光线模型下仍有 headroom；
- 证明 learned selector 会接近 oracle。

unrestricted oracle 和 residual-safe oracle 必须并列。若前者好而后者失败，结论是“truth 方向与数据模型冲突”，
不是“还差一个更强网络”。

### 3.3 有界 GO/NO-GO 解释

| 观察 | 允许判决 | 禁止外推 |
|---|---|---|
| 新方向 rank collapse 到 `S20` 或 matched H1 | 该方向未扩展当前离散 span | 所有 residual/edge 方法无效 |
| `F_k` oracle 不优于 `K_k` oracle | 特殊方向不如继续支付 k 个 H1 方向 | 所有 flexible Krylov 无效 |
| unrestricted 好、safe oracle 失败 | 表示余量与 active model 不相容 | 算法只需学更聪明权重 |
| safe oracle 在 6 个已开 case 上过门 | 可以撰写 fresh 预注册草案 | 算法、泛化或真实 BOST 成功 |
| 只有 edge singleton 好 | 进入固定 edge 方向的 fresh 机制验证 | 直接训练 FNO/DeepONet |

## 4. 一级来源：flexible Krylov 与 nonlinear/multiple preconditioning

| 一级来源 | 原文实际覆盖 | 对 R2-F0 的具体影响 |
|---|---|---|
| Notay, **Flexible Conjugate Gradients**, SISC 2000, [SIAM/DOI](https://doi.org/10.1137/S1064827599362314) | 分析迭代间变化的预条件器，并用显式搜索方向正交化维持 flexible CG 性质 | 若 edge/nonlinear preconditioner 随当前场变化，不能仍按 fixed-PCG 短递推解释；R2-F0 的 one-shot span audit 可以先不实现 FCG，但后续迭代算法必须用合法 flexible shell |
| Chung & Gazzola, **Flexible Krylov Methods for $\ell_p$ Regularization**, SISC 2019, [SIAM/DOI](https://doi.org/10.1137/18M1194456) | 用迭代重加权二范数和 flexible Golub-Kahan 把变化正则矩阵纳入增长子空间，并处理 tomography | “Huber/edge 权重改变 span”不是新颖性；matched H1 与 projected regularization 是必比项 |
| Arridge, Betcke & Harhanen, **A Priorconditioned LSQR Algorithm for Linear Ill-Posed Problems with Edge-Preserving Regularization**, Inverse Problems 2014, [DOI](https://doi.org/10.1088/0266-5611/30/7/075009), [arXiv 原文](https://arxiv.org/abs/1308.6634) | 用 lagged diffusivity 处理 TV/Perona-Malik，并以 factorization-free priorconditioned LSQR 把先验嵌入 solution space | D3 的 nonlinear priorconditioned backprojection 有直接近邻；必须比较其固定经典版本，不能只和 raw backprojection 比 |
| Calvetti et al., **Bayes Meets Krylov: Statistically Inspired Preconditioners for CGLS**, SIAM Review 2018, [SIAM/DOI](https://doi.org/10.1137/15M1055061) | 明确研究 right preconditioner 如何改变欠定 CGLS iterates 所在的 Krylov subspace | “用先验进入有效 null-space/改变 span”是已知机制；R2-F0 只能研究 BOST 特定方向是否在固定预算下更合适 |
| Bridson & Greif, **A Multipreconditioned Conjugate Gradient Algorithm**, SIMAX 2006, [SIAM/DOI](https://doi.org/10.1137/040620047) | 用多个 preconditioner 产生多搜索方向并在扩展空间中组合 | raw/Huber/edge 三方向联合不是一般意义的新概念；必须做 singleton、rank、正交化和成本消融 |
| Spillane, **An Adaptive MultiPreconditioned Conjugate Gradient Algorithm**, SISC 2016, [SIAM/DOI](https://doi.org/10.1137/15M1028534) | 在普通 PCG 足够与 multipreconditioned step 之间自适应选择 | “只在需要时启用 edge 方向”也已有算法先例；未来 learned gate 必须胜过简单、truth-free adaptive rule |

**边界判决：**R2-F0 可以把这些思想作为强经典骨架，但不能把 `flexible`、`priorconditioned`、
`multipreconditioned` 或 `edge-aware Krylov` 本身写进贡献标题。

## 5. 一级来源：hybrid projection、augmentation 与 recycling

| 一级来源 | 原文实际覆盖 | 对 R2-F0 的具体影响 |
|---|---|---|
| Chung & Gazzola, **Computational Methods for Large-Scale Inverse Problems: A Survey on Hybrid Projection Methods**, SIAM Review 2024, [SIAM/DOI](https://doi.org/10.1137/21M1441420), [arXiv 原文](https://arxiv.org/abs/2105.07221) | 系统连接 Krylov 投影、variational regularization、参数选择、augmentation、recycling、lp 与 nonlinear 扩展 | “先生成低维 span，再在其中加先验重优化”是成熟框架；oracle 与 deployable parameter choice 必须分账 |
| Jiang, Chung & de Sturler, **Hybrid Projection Methods with Recycling for Inverse Problems**, SISC 2021, [SIAM/DOI](https://doi.org/10.1137/20M1349515), [arXiv 原文](https://arxiv.org/abs/2007.00207) | 用 Golub-Kahan、压缩和 recycling 复用 solution basis，并允许标准 projected parameter selection | 缓存 H1-20 后扩充方向不是新颖性；R2-F0 必须保存完整 projection identities、basis rank 和与 standard hybrid 的关系 |
| Gazzola, Scott & Spence, **Flexible Krylov Methods for Edge Enhancement in Imaging**, Journal of Imaging 2021, [期刊全文/DOI](https://doi.org/10.3390/jimaging7100216) | 在一个 approximation space 中处理 IRLS 和多类 edge-enhancing regularizer，使用 flexible Golub-Kahan | `Huber prior + flexible projected solve` 已直接覆盖；R2-F0 的价值只能来自 BOST 物理合同和严格 matched audit |
| Lindbloom et al., **Priorconditioned Sparsity-Promoting Projection Methods for Deterministic and Bayesian Linear Inverse Problems**, 2025 preprint, [arXiv 原文](https://arxiv.org/abs/2505.01827) | 把 priorconditioning、generalized Krylov、IRLS、restart 和 recycling 结合 | 2025 年相邻工作进一步排除“prior + generalized/recycled Krylov”的宽主张；该文是预印本，只作碰撞边界，不作权威性能基线 |

**协议后果：**每个 `F_k` 和 `K_k` 都必须接受相同 reduced optimizer。若只对候选做 coefficient oracle，
得到的是 optimizer advantage，不是 span advantage。

## 6. 一级来源：learned preconditioner、direction 与 operator/subspace learning

| 一级来源 | 原文实际覆盖 | 对 R2-F0 的具体影响 |
|---|---|---|
| Kaneda et al., **A Deep Conjugate Direction Method for Iteratively Solving Linear Systems**, ICML 2023, [PMLR 正式页](https://proceedings.mlr.press/v202/kaneda23a.html) | 网络近似 inverse action，并在每次迭代改进搜索方向 | “学习 Krylov/CG 方向”不能主张新；若 R2-F0 后续学习，每步方向学习是强对照 |
| Li et al., **Learning Preconditioners for Conjugate Gradient PDE Solvers**, ICML 2023, [PMLR 正式页](https://proceedings.mlr.press/v202/li23e.html) | 用 GNN 学 approximate matrix factorization，并把 PDE solution distribution 纳入 loss | “利用样本分布学预条件器”不是新；BOST 方向必须报告矩阵/几何输入、setup cost 和跨 rig 失效 |
| Rudikov et al., **Neural Operators Meet Conjugate Gradients: The FCG-NO Method for Efficient PDE Solving**, ICML 2024, [PMLR 正式页](https://proceedings.mlr.press/v235/rudikov24a.html) | 用 discretization-invariant neural operator 作 flexible-CG nonlinear preconditioner | “neural operator + flexible CG”已被直接覆盖；未实现 FCG 时不能把 R2-F0 方向叫 FCG-NO-style method |
| Luo et al., **Neural Krylov Iteration for Accelerating Linear System Solving**, NeurIPS 2024, [NeurIPS 正式页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e88870ec82f2469b0ddf32c817920c68-Abstract-Conference.html) | neural operator 预测 invariant subspace，经 QR 与 projection loss 后加速线性求解 | “学习一个补充子空间并正交化”已知；R2-F0 的 rank/principal-angle audit 是必要机制证据，不是新算法 |
| Kopaničáková & Karniadakis, **DeepONet Based Preconditioning Strategies for Solving Parametric Linear Systems of Equations**, SISC 2025, [SIAM/DOI](https://doi.org/10.1137/24M162861X) | DeepONet 既可近似 inverse action，也可输出 trunk basis 形成低维校正空间 | “DeepONet 生成方向/低维 basis 后接 Krylov”已知；未来不能只比 direct FNO/DeepONet，必须比其 preconditioner/subspace 角色 |
| Dimola, Coclite & Zunino, **Neural Preconditioning via Krylov Subspace Geometry**, 2025 preprint, [arXiv 原文](https://arxiv.org/abs/2507.15452) | 用 residual 与 Krylov span 的 principal angles 构造 solver-aligned loss，并反传 flexible GMRES | 几何条件化、principal-angle loss、可微 flexible solver 均已有相邻预印本；R2-F0 的 angle 诊断不能单独算学习创新 |
| Levrero-Florencio et al., **NSPOD**, 2026 preprint, [arXiv 原文](https://arxiv.org/abs/2605.07828) | 用 DeepONet 学 POD 子空间作为 Krylov preconditioner，并讨论复杂几何 | “operator-learned subspace for geometry variation”仍在快速推进；若进入学习阶段，检索必须刷新到投稿日 |

这些论文的作者性能声称只属于其 PDE/线性系统实验。它们不证明 learned preconditioner 对 BOST 有效，也不证明
R2-F0 会失败；它们只关闭宽泛新颖性措辞，并规定未来需要的强对照。

## 7. 一级来源：edge-preserving inverse problems

| 一级来源 | 原文实际覆盖 | 对 R2-F0 的具体影响 |
|---|---|---|
| Rudin, Osher & Fatemi, **Nonlinear Total Variation Based Noise Removal Algorithms**, Physica D 1992, [出版社/DOI](https://doi.org/10.1016/0167-2789%2892%2990242-F) | 以 TV 约束抑噪并保留 sharp edges | “保护边缘的非线性先验”是基础方法，不是 R2-F0 新意 |
| Chan & Mulet, **Convergence of the Lagged Diffusivity Fixed Point Method**, SINUM 1999, [SIAM/DOI](https://doi.org/10.1137/S0036142997327075) | 分析 TV lagged-diffusivity fixed point 的收敛 | `M_delta(x20)^{-1}g_d` 有清楚经典来源；inner solve、SPD 条件与冻结权重必须显式 |
| Ascher, Haber & Huang, **On Effective Methods for Implicit Piecewise Smooth Surface Recovery**, SISC 2006, [SIAM/DOI](https://doi.org/10.1137/040617261) | 对可能非线性的 inverse map 使用 modified TV/Huber，并讨论参数和 lagged diffusivity | Huber prior direction、adaptive Huber 参数和 nonlinear inverse setting 都不新；R2-F0 的 `delta` 不得由 truth oracle 选 |
| Arridge, Betcke & Harhanen 2014, [DOI](https://doi.org/10.1088/0266-5611/30/7/075009) | 在三维 inverse problem 中把 edge prior、lagged diffusivity、matrix-free LSQR 与 priorconditioning 合并 | 与 D3 最接近的经典方法之一；若 D3 有 headroom，下一阶段首先复现这个固定强基线，而不是立刻训练网络 |
| Grauer et al., **Instantaneous 3D Flame Imaging by Background-Oriented Schlieren Tomography**, Combustion and Flame 2018, [出版社/DOI](https://doi.org/10.1016/j.combustflame.2018.06.022) | BOST 火焰重建直接比较 Tikhonov 与 TV，指出 flame front 的 abrupt RI variation 与 TV prior 更相容 | edge prior 已进入真实 flame BOST；R2-F0 只能问“该先验方向在当前固定成本/span 中是否有额外价值” |

### 7.1 edge 结果的最低解释标准

即使 `edge` 方向 oracle 获胜，也不能只报 gradient relative-L2。至少要同时检查：

1. field relative-L2；
2. gradient/H1 与冻结 front-band error；
3. active 与 held-out reprojection；
4. 高频噪声、ringing、overshoot 和 support 外能量；
5. per-case harm count，而不是只报 mean；
6. 与 raw、Huber、matched H1、fixed TV/Huber priorconditioned LSQR 的消融。

原因是 edge-preserving prior 可以锐化正确界面，也可以锐化噪声或错误几何产生的伪界面。oracle field/gradient
headroom 只是表示能力，不是物理真实性。

## 8. 一级来源：BOST 特定物理与 OERF 宿主边界

| 一级来源 | 原文实际覆盖 | 对 R2-F0 的具体影响 |
|---|---|---|
| Grauer et al. 2018, [Combustion and Flame/DOI](https://doi.org/10.1016/j.combustflame.2018.06.022) | BOS deflection 来自沿光路的 RI gradient；BOST 是 ill-posed inverse problem，火焰前沿需要相容先验 | joint field/gradient 指标有物理动机，但“field+gradient”本身不是新颖性；模型误差和 prior veracity 必须并列 |
| Cai et al., **Direct Background-Oriented Schlieren Tomography Using Radial Basis Functions**, Optics Express 2022, [Optica 正式全文](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100), [DOI](https://doi.org/10.1364/OE.459872) | 用可微 RBF 把 RI gradient 积分并入 projection matrix，ART 后接 TV，并用额外相机 reprojection 验证 | direct field reconstruction、TV 更新和 held-out camera reprojection 都已有 BOST 先例；它们是基线/评价，不是贡献 |
| He et al., **Neural Refractive Index Field: Unlocking the Potential of Background-Oriented Schlieren Tomography in Volumetric Flow Visualization (NeRIF)**, Physics of Fluids 2025, [AIP/DOI](https://doi.org/10.1063/5.0250899), [arXiv 原文](https://arxiv.org/abs/2409.14722) | 隐式网络同时输出 RI 与 gradient，以 AD/ND 和双 displacement path 约束一致性；数值数据由 nonlinear ray tracing 生成，实验保留一视角 reprojection | 神经 RI 场、field-gradient consistency、随机 ray sampling 和留出视角都不能作为新意；R2-F0 应定位为 solver span 诊断而非另一种 neural field |
| Li et al., **NeDF: Neural Deflection Fields for Sparse-View Tomographic Background Oriented Schlieren**, Physics of Fluids 2024, [AIP/DOI](https://doi.org/10.1063/5.0241191), [arXiv 原文](https://arxiv.org/abs/2409.19971) | 面向 sparse-view TBOS，用轻量网络表示 density-gradient field，并用 high-fidelity nonlinear ray tracing 合成观测 | sparse view、gradient representation 与 nonlinear ray tracing 已有直接近邻；“考虑曲光线”单独不构成差异 |
| He et al., **Tensor Decomposition-Based Four-dimensional BOST**, ACM TOG 2026 accepted, [ACM DOI](https://doi.org/10.1145/3809488), [蔡伟伟老师 SJTU 主页元数据](https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html) | 官方课题组页面确认题名、TOG 2026 accepted；本审计未获得可逐段核验的官方正文，故不进一步转述实现细节 | 仅凭题名已足以关闭“4D BOST + tensor decomposition”宽主张；任何更细比较须阅读正式正文后更新，不能靠二手摘要补写 |
| Lu et al., **Neural Refractive Index Primitives**, Combustion and Flame 2026, [出版社/DOI](https://doi.org/10.1016/j.combustflame.2026.115082), [arXiv 原文](https://arxiv.org/abs/2605.11454) | 用 compact neural RI primitive、编码、gradient loss、3D mask 与 ray sampling 做 flame BOST | compact implicit field、gradient loss 和 mask-aware sampling 也已被覆盖；R2-F0 若学习，应学习 bounded solver direction 而非重做 RI representation |

### 8.1 什么才可能是 BOST-specific

下面四项是**候选差异命题**，不是已证新颖性：

1. **ray-observability-aware direction：**edge gate 由逐视角 covariance、ray coverage、mask 和标定不确定度
   构造，而不是只从 `|Dx|` 构造；必须有 geometry shuffle/no-covariance/no-edge 消融。
2. **cross-model safety：**方向在冻结 straight-ray operator 中生成，却同时接受独立 curved-ray 或高保真
   renderer 的 residual envelope；这需要真实 callable/JVP/VJP，当前 PSU inverse-crime proxy不能证明。
3. **matched-span physical Pareto：**不仅问 residual 收敛快不快，而是问同物理调用/同维数下，BOST 方向能否
   在 field、RI gradient、held-out view 和 front-band 上支配继续 H1 的 Pareto frontier。
4. **fail-closed cross-rig deployment：**只用部署可见量预测方向或拒答，在未见 rig/session 尾部失败时精确退回
   H1；评价规范本身不是算法新颖性，但若与独特方向结构共同产生可复核收益，可能构成完整贡献的一部分。

这四项都必须由进一步检索和实验支持。尤其是 `curved/straight`、真实 covariance 和 calibration uncertainty
尚无组内接口，不能写成已经实现。

## 9. 组合碰撞矩阵

| 候选表述 | 判定 | 最强碰撞 | R2-F0 应如何改写 |
|---|---|---|---|
| 首次把 raw residual 加入 H1 Krylov | **不新** | least-squares gradient、hybrid/augmented Krylov | 只写“raw direction rank/control” |
| 首次用 Huber 方向保边 | **不新** | Ascher-Haber-Huang、edge-enhancing flexible Krylov | 只写“fixed Huber singleton audit” |
| 首次用 nonlinear edge preconditioner 扩展 span | **不新** | lagged diffusivity、priorconditioned LSQR、flexible Krylov | 只写“BOST-conditioned fixed priorconditioner” |
| 首次联合多个方向 | **不新** | multipreconditioned CG | 做 rank、singleton、顺序和成本消融 |
| 首次缓存 H1 span 再重优化 | **不新** | hybrid projection、recycling | 作为工程与公平比较机制 |
| 首次学习 Krylov 方向/子空间 | **不新** | DCDM、FCG-NO、NeurKItt、DeepONet preconditioning | 仅在 fixed directions 有 headroom 后研究 BOST-specific controller |
| 首次神经 BOST field/gradient 重建 | **不新** | NeRIF、NeDF、Neural RI Primitives | R2-F0 不直接预测最终 RI field |
| BOST 可见量驱动、matched-span、cross-model、可拒答方向协议 | **可能有差异，未证实** | 本次有界检索未找到完全同构工作 | 保持为待证研究问题；投稿前重做系统检索 |

## 10. R2-F0 运行前必须冻结的协议清单

### 10.1 数据与公共状态

- 6 个 case id、partition 和“post-open mechanism-only”标签；
- `P20`、`A_wP20`、`x20`、`r20`、geometry 与所有输入 hash；
- H1 lambda、whitening、support、voxel spacing、boundary condition、denominator floor；
- source commit、dtype/device 与 CPU-double transfer 边界；
- 禁止把这些 6 case 叫 fresh、unseen rig 或真实 flame data。

### 10.2 方向定义

- `d_r` 的 residual 是 data-only 还是 augmented H1 residual；二者不能混称；
- Huber potential、`delta`、vector/scalar isotropy、normalization；
- D3 到底是 cheap diagonal gate 还是 lagged-diffusivity inverse；
- covariance、mask、geometry feature 的来源与允许字段；
- inner solve tolerance、iteration cap、setup reuse 与 failure fallback；
- 方向顺序、singleton 列表、两遍正交、rank/angle 容差；
- 所有方向只从 `x20,r20` 生成，truth access audit 必须为 0。

### 10.3 对手与 oracle

- H1-21/22/23 完整 direction/projection cache；
- `F1/F2/F3` 与 `K1/K2/K3` 使用同一个 reduced optimizer；
- unrestricted 与 `rho_A<=1.02` safe oracle；
- 完整 Pareto 与预写 `J_infinity`，不得结果后换权重；
- 同时报告相对 H1-20 与相对 matched `K_k` 的 field/gradient gain；
- ordinary H1 iterate 与 oracle(K) 分列；
- H1、raw、Huber、edge singleton 和 full nested 的全矩形结果。

### 10.4 成本与失败门

- 每 case 的 common、direction build、projection、reduced solve 分阶段 `F/A^T`；
- `D/D^T`、inner iterations、network=0、wall time、peak memory、setup/apply；
- 非有限、rank collapse、正交缺陷、projected conditioning、adjoint defect；
- active/held-out residual、field、gradient、front-band、support leakage；
- mean、逐 case、p10/worst、harm count；
- 任何失败必须保留原始 invalid artifact，不覆盖正式目录。

## 11. 运行后的合法决策树

```text
Are q_raw/q_huber/q_edge numerically independent of S20?
  no  -> SPAN_DUPLICATE_NO_GO; stop that direction
  yes -> Compare oracle(Fk) with oracle(Kk), symmetrically

Does Fk have joint field/gradient headroom over matched Kk?
  no  -> SPECIAL_DIRECTION_NO_MATCHED_HEADROOM; no learning
  yes -> Apply residual-safe envelope and held-out/front gates

Does safe headroom survive on the six opened cases?
  no  -> DATA_INCOMPATIBLE_ORACLE_NO_GO; no learning
  yes -> Write a separate fresh preregistration; do not train yet

Does a frozen truth-free rule predict/accept the useful direction on fresh cases?
  no  -> REPRESENTATION_ONLY_NO_DEPLOYABLE_SIGNAL
  yes -> Only then compare fixed rule, ridge/MLP, DCDM, FCG-NO-style,
         DeepONet-preconditioner and direct NeRIF/NeDF host baselines
```

## 12. 给何远哲师兄的五个 R2-F0 专属问题

1. 组内真实 solver 在 H1-20 时能否导出 `x20`、data residual、regularized gradient 和最后三条搜索方向？
2. `A/A^T` 是 straight-ray 线性算子，还是只有 nonlinear renderer 的 JVP/VJP？两者的 residual 分别在哪层？
3. 当前最真实的 edge failure 是 flame front 被抹平、位移噪声被锐化、有限视角伪影，还是标定/曲光线偏差？
4. 是否有 per-view confidence/covariance、flow-off repeats、calibration uncertainty 与 held-out camera？
5. 实验室认可的 matched baseline 是继续 H1/CGLS、TV/Huber priorconditioned LSQR、NeRIF、NeDF，还是
   TDBOST 中的某个 solver block；其 `F/A^T/JVP/VJP` 与 wall-time 预算是什么？

未得到这些回答前，R2-F0 只能在公开 PSU 几何与解析 proxy 上做离散机制诊断，不能替实验室定义真实痛点。

## 13. 最终 claim boundary

### 当前可以写

> 一级来源表明，raw residual、Huber/TV、nonlinear priorconditioning、multiple/flexible Krylov、hybrid
> recycling 与 learned preconditioner/subspace 均已有直接先例。R2-F0 因而被限定为一个 BOST-oriented、
> bounded span-expressivity audit：在冻结 H1-20 后，用三类部署可见方向扩展 span，并以完全对称的
> truth-only oracle 和 matched H1-21/22/23 判断是否存在联合 field/gradient headroom。

### 当前不能写

- “提出了一种新型 edge-aware Krylov 算法”；
- “首次将算子学习用于 BOST/Krylov”；
- “truth oracle 证明方法可部署”；
- “在 6 个 synthetic cases 上有 oracle headroom，因此可泛化”；
- “优于 DeepONet、FNO、NeRIF、NeDF、TDBOST 或 FCG-NO”；
- “解决了真实火焰曲光线、标定误差或跨 rig 问题”。

## 突破监测

**没有突破。** 本文新增的是一条更严格的来源边界和比较合同：三个方向族都有明确先例；真正需要证伪的
是它们在 BOST 固定物理预算下是否比继续 H1 更有 span expressivity。任何 R2-F0 正结果都必须先通过
`oracle(F_k) vs oracle(K_k)`、residual-safe、逐 case 尾部和真实接口门，才有资格进入 fresh；当前
`new_algorithm=false / experiment_run=false / real_bost=false / generalization=false / paper_success=false`。
