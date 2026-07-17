# v3 Grouped Majorizer 到真实 BOST 前向模型的 signed-primitive 接口地图

**日期：** 2026-07-17
**用途：** 把 `certified_grouped_majorizer` 的合成 smoke test 转化为一个可向何远哲师兄确认的真实 BOST 接口合同。本文不宣称已经迁移、没有报告真实数据结果，也不把“可能的物理分解”写成实验室事实。

## 0. 先给结论

v3 的保证只需要一个很具体的条件：在**同一个线性化点、同一个坐标系、同一个归一化约定**下，求解器实际使用的线性算子可以写成

\[
A = \sum_{\ell=1}^{L} C_\ell,
\qquad
M_{\mathcal P}=\sum_{G\in\mathcal P}\left|\sum_{\ell\in G}C_\ell\right| \succeq |A|.
\]

这里的绝对值是**离散矩阵/离散线性算子条目的逐元素绝对值**，不是向量范数。任意完整、不重叠的 partition \(\mathcal P\) 因三角不等式给出 majorizer；因此“安全性”不是学习器猜出来的，而是由构造保证。当前仓库的 v3 只在小型合成矩阵上验证这一命题，且其 selector 在 fresh geometry 上会伤害 4/8 个 case，故没有算法成功或论文优越性授权。

真实 BOST 最可行的首步不是把 NeRIF 的整个非线性反演强行塞进 v3，而是选择一个固定场 \(n_0\) 的**高斯-牛顿/一阶 Born 线性化**，导出其 Jacobian \(J(n_0)\)，再把 **同一个 Jacobian** 拆为带符号 primitive。若实验室无法稳定输出该接口，则 v3 应停止，不以“理论上可以”替代可审计实现。

## 1. 可核实的前向事实

下列项目是公开一手来源明确写出的事实。

1. [NeRIF 原文（He et al., arXiv HTML）](https://arxiv.org/html/2409.14722v2) 将 BOST 写成由预计算层析投影矩阵、微分矩阵和世界到相机坐标投影组成的大型病态线性系统；其叙述明确指出测得的是由折射率非均匀性造成的背景位移/偏转，并把其与折射率梯度的路径积分联系起来（论文第 II.1 节）。
2. 同一原文明确给出其 neural forward 的数据流：从相机像素向背景板反向追迹，在射线路径上采样坐标，网络输出折射率和梯度，按相机标定计算传感器平面的位移；实际实现采用随机射线采样，并在数值模拟中使用四阶 Runge--Kutta ray tracing。见 [NeRIF 第 II.2 节与附录 B](https://arxiv.org/html/2409.14722v2#S2.SS2)。
3. [蔡伟伟 SJTU 主页](https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html) 将 “Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction”列为 2026 accepted，也列出单相机内窥 BOST 的折射率、密度、温度同步测量项目；这确认本接口地图贴近实验室公开的 BOST/4D 线路，但**不等于**取得了该工作的私有实现接口。
4. 公开的 [Direct BOST using radial basis functions 原文（Optics Express）](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100) 明确使用相机内外参得到像素射线、计算射线穿过重构域的贡献，并把投影矩阵存为稀疏矩阵；该文也把 BOST 描述为病态逆问题。它是“camera/ray primitive 可以物理落地”的直接先例，但并未提出 v3 的 grouped-majorizer。
5. [Neural Refractive Index Primitives（Lu et al., arXiv 2026）](https://arxiv.org/html/2605.11454) 已公开使用“相机射线 marching、沿射线累积折射率梯度、离散 quadrature、mask 引导采样”，并在数值前向中模拟有限孔径的 multi-ray integration、背景 warping 与 optical flow 位移估计。它是本项目最直接的术语与实现撞题来源之一。
6. v3 所需的“majorizer 使一阶原始-对偶更新满足稳定条件”是数值优化中的既有思路，不是本项目首创；参见 [Chambolle--Pock 原始一阶原始-对偶论文](https://doi.org/10.1007/s10851-010-0251-1) 和 [Pock--Chambolle diagonal preconditioning 论文](https://doi.org/10.1109/ICCV.2011.6126441)。本文只把其与 BOST signed-primitives 的接口问题对接。

## 2. 从公开 forward 到 primitive 的映射

### 2.1 可直接由公开资料支持的层级

| forward 层级 | 对应的可见对象 | 适合作为 signed primitive 吗？ | 证据与边界 |
|---|---|---:|---|
| view / camera | 视角的标定、该视角的位移观测、世界到像平面投影 | **候选，需接口确认** | NeRIF 明确使用多视角、相机标定和相机坐标下位移；其模拟为 9 个视角，实验以光纤束形成 9 个输入端。[NeRIF](https://arxiv.org/html/2409.14722v2#S3) 未公开 `J_view` API。 |
| pixel ray | 单一像素反向追迹到背景板的采样点、积分权重 | **最强候选** | NeRIF 明确逐像素反向追迹与沿 ray 采样；RBF BOST 明确由像素和标定得到射线路径并累积穿过域的权重。[NeRIF](https://arxiv.org/html/2409.14722v2#S2.SS2)；[RBF BOST](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100)。 |
| spatial gradient component | \(\partial_x n,\partial_y n,\partial_z n\)，以及到相机平面的分量投影 | **候选，必须看真实离散化** | NeRIF 明确同时预测折射率及三方向梯度，体素式表述含 projection 与 differential matrices。能否把每个分量拆成独立且同一 Jacobian 的 \(C_\ell\)，公开资料没有给出。 |
| quadrature sample / path segment | 一条 ray 上的 sample point、段长或 quadrature weight | **候选，最细粒度** | NeRIF 的随机区间采样和后续工作中的离散 quadrature 是明确事实。[NeRIF](https://arxiv.org/html/2409.14722v2#S2.SS2)；[Neural Refractive Index Primitives](https://arxiv.org/html/2605.11454#S2.SS2.SSS2)。是否有足够的符号消去而值得分组，是待测量问题。 |
| aperture sub-ray | 有限孔径下同一 sensor sample 的多条子射线 | **条件候选** | 2026 neural-primitives 论文的**数值图像形成**明确含 finite-aperture multi-ray integration；NeRIF 原文公开实现没有把此列为实际实验反演的 primitive。因此只有实验室 forward 真的建模 aperture 时才能用。 |
| time / tensor factor | 4D 的帧、时间段、tensor component | **仅 4D TDBOST 接口确认后候选** | 蔡主页确认 TDBOST 接受发表，但公开页没有给出可逐项线性化的 tensor-factor forward API。不能从题名推断其 primitive 设计。 |

### 2.2 关键区分：事实、可行推断、禁止假设

**公开事实。** 对固定直线 ray 的一阶 BOST 表达，位移由折射率梯度沿 ray 累积并经相机几何投影得到；离散模型可由 ray-weight/projection 与 differential operator 组成。[NeRIF](https://arxiv.org/html/2409.14722v2#S2.SS1)；[RBF BOST](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100)。

**工程推断（尚未确认）。** 若实验室可输出 `Jv` 与 `J^Tq`，并且 `J` 的每个贡献可以按 camera、ray bundle、quadrature segment 或 gradient component 相加，那么这些贡献可定义为 \(C_\ell\)。对非线性 ray bending，必须在同一 \(n_0\) 处计算 Jacobian；不允许把不同迭代步、不同随机采样、不同 mask 或不同归一化下的项相加。

**禁止假设。** 不应假设：

- `camera contribution` 天然带符号且存在有用抵消；不同 camera 的观测通常是不同 residual 行块，可能根本不在同一输出行上，按 camera 分组只会增加实现复杂度而没有 tighter bound。
- NeRIF 的随机采样项可逐项复用为确定性 preconditioner；随机性若不冻结，会使 \(A\) 在每次迭代改变，破坏本轮安全审计的对象一致性。
- optical-flow 的输出、ray bending、有限孔径和背景 warping 仍是全局线性映射；它们一般是非线性的，只能在线性化后讨论 \(J\)。
- 仅有 `A@v` 和 `A.T@q` 就足以构造逐元素 \(|A|\) majorizer；若没有 signed block 的可审计表示或等价的正包络算子，v3 证明无法直接落地。

## 3. 建议给实验室的 signed-primitive 接口合同

### 3.1 最小接口（线性/线性化 BOST）

每一次 outer iteration 固定一个 `LinearizationContext`，它必须携带：

```text
context_id                 # hash: calibration + mask + field n0 + sampling seed + normalization
domain_grid                # world coordinate, voxel/basis order, units
measurement_layout         # (view, pixel, component) row order and valid-mask
field_parameter_layout     # column order: n / basis coeff / neural tangent parameter
forward_regime             # straight_ray | nonlinear_ray_jacobian
primitive_catalog          # primitive_id -> semantic type + output row support + input support
```

对每个 `primitive_id` 需要以下之一：

```text
apply_C(id, v)             # C_l v
apply_Ct(id, q)            # C_l^T q
materialize_C(id)          # 仅小问题/审计允许；给出稀疏 signed entries
positive_envelope(id)      # 可验证地支配 |C_l| 的算子；必须说明证明或审计方法
```

并需要：

```text
apply_A(v) == sum_l apply_C(l, v)
apply_At(q) == sum_l apply_Ct(l, q)
audit_seed + deterministic_sampling_manifest
cost_counter               # ray eval / gradient eval / CUDA kernel / memory, 不能只报 epoch
```

**必须满足的审计不变量：**

1. 对小规模 materialized case，逐条验证 `A == sum(C_l)`、`A.T == sum(C_l.T)`。
2. 每个 partition 的 \(M_{\mathcal P}\) 与 `A` 来自同一个 `context_id`。
3. all-in-one \(|A|\) 只能作 oracle lower-bound comparator；部署 selector 不得读取它。
4. selector 输入只能是提前定义的 geometry/calibration/mask 统计量，不能读取 fresh truth、held-out reprojection error 或由完整 \(A\) 计算出的 exact tightness。
5. 若 primitive 的行 support 不重叠，记录“无可抵消空间”，不要把这种分组包装为创新。

### 3.2 对 neural field 的正确落点

对于 NeRIF 类表示，直接对网络参数 \(\theta\) 做 v3 会得到超大且状态相关的 \(J_\theta\)。本科阶段的优先顺序应为：

1. 固定一个重构场/phantom 与 camera calibration；
2. 先对 voxel/RBF coefficient 或 frozen neural field 的小维 latent perturbation 建立 `J`；
3. 证明/审计 grouped majorizer 对该 `J` 安全；
4. 只有在第 3 步通过后，再尝试用 autograd 的 JVP/VJP 接到 neural parameters。

这避免把“神经网络优化不收敛”与“preconditioner 不安全”混在一起。NeRIF 本身强调折射率、梯度、随机采样和 AD/ND 一致性约束；这些都应保持为原 forward 的一部分，而不是被 v3 静默替换。[NeRIF](https://arxiv.org/html/2409.14722v2#S2.SS2)。

## 4. 可执行 partition catalog（先建小型审计版）

以下不是已验证的最优分组，而是按物理语义列出的候选目录。每一个都要在真实接口上先做 `A=sum C_l` 与 Schur audit，不能仅凭名称上线。

| catalog | primitive 定义 | 可能的抵消机制 | 构造成本与适用条件 | v3 初始地位 |
|---|---|---|---|---|
| P0 `singleton-factor` | 单个 deterministic ray-segment × gradient-component contribution | 无；即 \(\sum_\ell |C_\ell|\) | 最低逻辑风险；需保存/重放 segment 权重和符号。作为 deployable safe baseline。 | 必做基线 |
| P1 `ray-local` | 同一 pixel ray、同一 gradient component 的连续 quadrature segments | 折射率梯度沿同一路径局部正负变化可能抵消；**仅推断，需测** | 先累积一个 ray 的 signed block 再取绝对值；成本约为 P0 同样 primitive eval 加 group accumulation。要求 quadrature/mask 固定。 | 第一优先 |
| P2 `ray-component` | 同一 pixel ray 的 \(x,y,z\) gradient 投影贡献 | 相机平面投影后不同梯度分量可能相消；**仅推断** | 需导出 world-to-image projection 的带符号系数并保证三分量同一 residual row。若只是分开的二维测量行，不可用。 | 第二优先 |
| P3 `aperture-bundle` | 同一 sensor pixel 的有限孔径 sub-rays | sub-ray direction/weight 对同一像素的聚合可能抵消；**条件推断** | 每个 pixel 先在 aperture 内求 signed group，成本与 sub-ray 数成正比；只用于 forward 已真实建模 finite aperture 的场景。公开先例见 [Neural Refractive Index Primitives](https://arxiv.org/html/2605.11454#S3.SS1)。 | 有条件 |
| P4 `view-tile` | 同一 camera 的邻近 pixels/rays tile，或同一 view 的 calibration-consistent block | 邻域 ray 对同一 basis column 的符号结构可能相消；**高风险推断** | 需要 materialize/稀疏块或 block-envelope；内存风险最高。先只做 8×8 或 16×16 tile audit，不能在全分辨率直接试。 | 探索项 |

**不建议作为第一版 partition：** 按 camera/view 把所有 ray 合为一组。不同 camera 通常对应不同 measurement rows，若行 support 不重叠，\(|\sum C_l|=\sum|C_l|\) 或无法产生实际收紧；这不是有价值的 cancellation-aware 设计。

### 成本计量规则

不能用“每 epoch 时间”代替成本。每个 candidate 至少报告：

- `ray_samples`, `quadrature_evaluations`, `gradient_evaluations`, `primitive_accumulations`；
- peak GPU/CPU memory、`Jv/J^Tq` 调用数、wall-time（wall-time 仅描述性，除非同机同批量重复计量）；
- group materialization 的稀疏 nonzeros 或 matrix-free buffer bytes；
- 与 P0 完全相同的 forward/adjoint 和迭代预算。

若某 partition 需要完整 materialize \(A\) 才能构造，而 P0 不需要，则它不是同预算 deployable 比较；应降级为 oracle/diagnostic。

## 5. 实验室必须确认的问题

建议把下面清单原样给何远哲师兄。得到答复前，所有 P1--P4 只是接口假设。

1. 当前目标是哪个 forward：straight-ray BOST、nonlinear ray tracing、TDBOST 的 distortion-corrected ray tracing，还是某个 neural field 的 autograd Jacobian？每轮迭代能否冻结一个明确 `context_id`？
2. 观测向量的 row 是怎样排列的：`(view, pixel, u/v displacement)`、三维 deflection，还是 optical-flow 后的其他量？不同 view 是否共享 residual row？
3. 是否已有可调用/可导出的 `A@v`、`A.T@q`、稀疏 projection/differential matrices，或 PyTorch `jvp/vjp`？哪个对象才是求解器实际使用的线性算子？
4. 能否在小 ROI、低分辨率、少视角上导出 signed ray-segment weights，验证 `A=sum C_l`？若只能导出绝对权重，v3 无法开始。
5. ray 是否为 straight ray？若会随折射率迭代弯曲，Jacobian 的线性化点、ray re-tracing 频率和随机 seed 如何冻结？
6. 真实 forward 是否包含 finite aperture、multi-ray averaging、background warp、optical flow、camera distortion、mask？哪些位于 Jacobian 内，哪些是在观测预处理外？
7. 梯度是 direct predicted、AD、ND，还是组合？\(x/y/z\) 分量经过相机投影后能否追溯到同一 signed residual row？
8. 4D TDBOST 是否有可访问的 tensor factor 或 time-window primitives？它们是 forward 的可加项，还是只是 field representation 参数？
9. 哪些 calibration/mask/geometry features 在部署时已经可知，因而可合法喂给 selector？明确禁止使用哪些 held-out 视角误差、真值或完整 operator 统计量？
10. 是否可以给一个可公开/可合成的最小 calibration + mask + displacement bundle？没有这一最小包时，项目应停在接口与合成验证层，不报告真实 BOST 改进。

## 6. 创新碰撞检查：不能说“首创”

本轮只检查了题目允许的一手来源，结论是**存在明显邻近工作，不能作首创性宣称**：

- NeRIF 已在 BOST 中使用连续 neural refractive-index field、射线积分、随机采样、AD/ND 一致性和真实火焰验证。[NeRIF](https://arxiv.org/html/2409.14722v2)
- 2026 的 [Neural Refractive Index Primitives](https://arxiv.org/html/2605.11454) 已明确把 “neural refractive-index primitive”用于 BOST，并公开 ray sampling、mask、gradient formulation 与 aperture-aware 数值图像形成；因此本项目不能以“把 primitive 引入 BOST”作为创新点。
- 蔡组公开列出 TDBOST 的张量分解与神经网络路线；在没有全文接口和独立对比前，不能把“按时间/张量分解”包装成新概念。[SJTU 主页](https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html)
- Chambolle--Pock/Pock--Chambolle 的预条件与 majorization 也是成熟基础，不能以“安全 preconditioner”本身声称新。

**可能值得继续验证、但尚非创新主张的差异点：** 对真实 BOST 的**同一线性化 Jacobian**建立可审计的 signed ray-/component-level decomposition；在不访问 truth、held-out error 或 exact \(|A|\) 的前提下，用部署可知的 calibration/geometry 选择 partition；并证明每个 deployment case 的 majorization/Schur 安全，再以 held-out view 的 reprojection 与真实成本报告收益。这个组合是否新，需要在拿到具体 forward 后再做更完整的文献检索；当前不能称“首次”。

## 7. 本科阶段最小迁移实验

### M0：两周内的接口可行性审计

**目标：** 不是训练大模型，而是在一个 \(32^3\) 或更小 ROI 上验证 `A=sum C_l`。

1. 采用公开 NeRIF 的 straight-ray 数值设定或实验室允许公开的 calibration，构造一个可微 3D Gaussian / 双涡结构折射率 phantom；只输出 displacement，不把温度当真值主张。[NeRIF 的数值设置与多视角 ray tracing](https://arxiv.org/html/2409.14722v2#S3)。
2. 固定 view、pixel mask、quadrature points、seed 和 field \(n_0\)，导出小型 signed `J` 或以 `C_l` materialization 交叉验证。
3. 做 P0、P1、P2 三种 partition：逐元素检查 \(|J|\le M_{\mathcal P}\)，并记录每个 group 的 tightening ratio 与构造成本。
4. 先不训练 selector。只有 P1/P2 对至少一个非平凡 case 比 P0 更紧、且没有任何 safety violation 时，才进入 M1。

### M1：四到六周的可比较反演实验

**固定条件：** 相同初值、相同 outer linearization schedule、相同 `Jv/J^Tq` 预算、相同 regularizer、相同 deterministic seed。

**比较对象：** P0 factor baseline、一个训练选出的固定 partition、一个仅读 geometry/calibration feature 的低容量 selector、all-in-one exact 诊断 oracle（不部署）。

**结果指标：**

- safety：每个 case 的 \(|J|\le M\) 与 solver stability audit，零容忍；
- inverse quality：合成真值 relative-L2、真实实验则预先保留视角的 displacement reprojection loss/SSIM/CC；NeRIF 也使用 held-out projection 做实验验证，这一做法与组内公开路线一致。[NeRIF](https://arxiv.org/html/2409.14722v2#S4.SS2)；
- efficiency：上节成本计数，另报重复 wall-time；
- robustness：至少改变 view subset、noise、mask size 与 flame/phantom morphology，不能只换随机种子。

### M2：何时才值得接 neural field / TDBOST

只有当 M1 的 selector 在**预注册的 fresh geometry cases 全部不劣于**固定安全 partition，且有独立 held-out view 与成本结果，才尝试接 frozen NeRIF tangent/JVP-VJP；4D TDBOST 则要再确认 time/tensor primitive 是否属于同一 forward Jacobian。否则应把成果写成“BOST majorizer interface + negative selection evidence”，而不是算法论文。

## 8. 停止门（比继续堆模型更重要）

| stop gate | 触发条件 | 结论与下一步 |
|---|---|---|
| S0 interface stop | 不能导出 signed `C_l`，或无法在同一 context 验证 `A=sum C_l` | 停止 v3 真实迁移；保留为合成方法学，不训练 selector。 |
| S1 nonlinearity stop | ray/path、mask 或随机采样在同一内迭代中改变，导致没有固定 \(J\) | 先定义 outer linearization 与 deterministic manifest；未修复前不能引用 v3 保证。 |
| S2 no-cancellation stop | P1/P2/P3 在所有审计 case 与 P0 等价或更松 | 该 physical partition 没有价值，删除目录项，不扩大网络。 |
| S3 safety stop | 任一 fresh case 的 elementwise majorization/Schur audit 失败 | 立即停用该 partition/selector；不得用平均值掩盖。 |
| S4 deployment stop | selector 访问 exact \(|A|\)、fresh truth 或 held-out residual 才能选分组 | 这是 oracle leakage，降级为离线诊断。 |
| S5 value stop | 安全但全 fresh geometry 上不能稳定不劣于固定 partition，或构造成本抵消收益 | 报告严格负结果；转向固定 partition 或其他已知物理瓶颈。 |

## 9. 交给师兄的一句版本

> 我想把 BOST 在一个固定线性化点的 Jacobian 拆成可审计的 signed ray/gradient contributions；先验证 \(J=\sum C_l\)，再按 ray-local 或 ray-component 分组构造一定安全的 majorizer。请问现有 forward 能否导出这一层的 signed block、固定 sampling manifest 和 `Jv/J^Tq`？若不能，我不会把 v3 包装成真实 BOST 算法。

## 10. 证据范围与下一步

本文的事实链接均为实验室公开页、论文正式页面或 arXiv 作者版；没有使用二手综述来替代 forward 证据。它不能替代对实际代码、标定文件和数据协议的检查。下一步必须先由实验室对第 5 节问题给出接口答案，再决定实施 P0/P1/P2 中的哪一个；在此之前不需要租服务器，也不应开始大规模训练。
