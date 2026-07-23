# LGWO-A24 L1 新颖性主张边界：独立一级来源审计

- 审计日期：2026-07-20
- 审计对象：LGWO-A24 L1 的窄主张，而非整个 OERF 或所有 BOST 算法
- 文献边界：只采用论文作者页面、出版社页面、会议正式页面、arXiv 和论文官方代码仓库
- 本地协议依据：`demo_t16_operator/configs/lgwo_a24_l1_sealed_observable_pilot_v1.json` 及其绑定实现
- 审计性质：新颖性风险审计，不是专利检索，也不是系统综述

> **强制当前状态：`0 scientific cases / 0 optimizer steps / no breakthrough`。**
>
> 截至本审计时点，没有科学样本结果、没有优化器训练步、没有合格路线结果、没有真实 BOST 证据，也没有算法突破。下文所有“可能差异”都只是待检验的研究假设，不能写成已经实现的新颖性或性能结论。

## 1. 审计结论

| 审计问题 | 结论 | 当前可否写入论文主张 |
|---|---|---|
| “用神经网络修正逆问题重建”是否新颖 | 否。MoDL、Learned Primal-Dual、神经校正算子和 learned warm-start 已覆盖这一宽泛思路 | 不可 |
| “学习 CG/Krylov 搜索方向或预条件器”是否新颖 | 否。DCDM、FCG-NO、学习型 PCG 预条件器和 DeepONet-Krylov 混合方法已直接覆盖 | 不可 |
| “学习零空间信息”是否新颖 | 否。Deep Null Space Learning 和 NPN 已给出显式零空间结构及理论分析 | 不可 |
| “用 DeepONet/FNO/iFNO 学习正向或逆向算子”是否新颖 | 否。该方向已经成熟，并已扩展到逆问题和一般几何 | 不可 |
| “用神经表示或时序网络做 BOST 三维/四维重建”是否新颖 | 否。GRU-BOST、NeRIF、NeDF、TDBOST 和 Neural Refractive Index Primitives 已覆盖 | 不可 |
| “保留一组测量用于一致性损失或验证”是否新颖 | 否。Noise2Inverse、SSDU、SPLIT 已使用测量划分；BOST 中也已有留出相机验证 | 不可 |
| LGWO-A24 的完整窄组合是否已有完全相同论文 | 本次有界一级来源检索未找到完全相同组合 | 仍不可据此声称新颖 |
| 当前最合理的定位 | **可能有差异、尚未证实的 BOST 专用固定预算求解器增强协议** | 只能写成研究问题 |

总判定：

```text
BROAD_NOVELTY: REJECTED
EXACT_COMBINATION_MATCH_IN_THIS_BOUNDED_AUDIT: NOT FOUND
NARROW_DIFFERENTIATION: PLAUSIBLE BUT UNPROVEN
SCIENTIFIC_CLAIM_STATUS: NO-GO
CURRENT_EVIDENCE: 0 scientific cases / 0 optimizer steps / no breakthrough
```

“未找到完全相同论文”不是新颖性证明。它只说明可以继续做一次严格、可反驳的实验。最终是否构成论文贡献，取决于公平基线、机制证据、未见几何尾部表现、真实数据和再次检索，而不是组件清单。

## 2. 被审计的窄方法究竟是什么

按当前冻结协议，对观测 `y` 和 A 组几何，先计算一次伴随锚点：

```math
g = A^T y.
```

2729 参数模型只读取可部署输入 `y`、`g`、A 组几何特征和 support，输出原始修正 `r_theta`。实现逐样本施加相对范数约束：

```math
delta_theta = clip(r_theta; ||delta_theta||_2 <= 0.05 ||g||_2),
d_0 = (g + delta_theta) \odot support.
```

此后：

1. 只把 `d_0` 用作第一个搜索方向。
2. 沿 `d_0` 做精确一维最小二乘步。
3. 后续方向重新由残差伴随产生，并做两遍测量空间重正交化。
4. 部署每例严格保持 `24 A` 次和 `24 A^T` 次调用；提案网络自身不得调用 `A` 或 `A^T`。
5. `delta=0` 必须逐项恢复锚定 shell 基线。
6. 超出 fit-only 六标量校准包络或出现非有限输出时，动作是精确退回 `delta=0`。

模型禁止读取 truth、clean observation、family、partition、case id、seed、geometry digest、baseline error、exact-null basis 和任何 heldout-B 对象。训练损失和评价可以读取 truth。B 组是相同解析场下独立几何种子与射线束；fit B 是辅助损失，route B 是封闭评分，不是独立物理，也不是真实 BOST。

## 3. 必须立即纠正的术语

### 3.1 不能称为 learned null-space projection

Deep Null Space Learning 使用形如 `Id + (Id - A^+A)N` 的显式零空间结构，使网络修正位于 `ker(A)` 并保持数据一致性。NPN 进一步学习感知矩阵零空间的非线性低维投影。LGWO-A24 当前既没有 `A^+`、显式投影，也禁止在训练损失中使用 exact-null basis。

当前只有软惩罚：

```math
P_A = ||A delta||_2^2 / ||A g||_2^2,
```

且权重为 `0.02`。它至多鼓励修正进入 A 的低敏感方向，不能证明 `delta in ker(A)`，也不能证明严格保持 A 组数据一致性。允许表述是“near-null/low-observability tendency 的待检验诊断”，不允许表述是“learned null-space projection”。

### 3.2 不能称为 self-supervised BOST

模型在部署时看不到 B，但当前 B clean target 由同一个 synthetic truth 和解析渲染器生成，训练又同时使用 truth-based field loss。因此当前最准确的说法是：

> independent-ray cross-operator consistency regularizer in a supervised synthetic pilot

Noise2Inverse、SSDU 和 SPLIT 才分别在明确噪声假设或测量划分下建立 self-supervised 训练逻辑。除非未来真实 BOST 阶段只用实测 A/B 位移并去掉 truth-based 训练项，否则不能借用“自监督”主张。

### 3.3 fail-closed 不等于有概率保证的安全

当前包络是 24 个 fit cases 上六个标准化标量的逐轴 min/max，再外扩 25%。它没有有限样本 coverage 定理，不是 conformal prediction，也没有检测所有高维分布偏移的能力。它只能被称为：

> empirical axis-aligned calibration envelope with deterministic baseline fallback

另外，route 中只要发生一次 fallback 就不能通过 signal gate。因此“系统会退回基线”是部署行为，“路线成功”则要求没有触发退回，两者不能混为安全性证据。

### 3.4 unseen-rig tail gates 是评价设计，不是算法新颖性

逐 rig 尾部、最坏 family、case-seed harm rate 和 bootstrap lower bound 能减少平均数掩盖失败的风险，但这些门本身不构成新算法。当前配置也明确将 `unseen_rig_generalization` 设为 false。

## 4. 一级来源碰撞图

### 4.1 零空间学习

| 一级来源 | 已经建立的内容 | 对 LGWO-A24 的边界影响 |
|---|---|---|
| Schwab, Antholzer, Haltmeier, [Deep Null Space Learning](https://arxiv.org/abs/1806.06137), [DOI](https://doi.org/10.1088/1361-6420/aaf14a) | 显式 null-space network、数据一致性、正则化收敛与速率 | “学习零空间修正”不能作为新颖性；LGWO 还达不到其严格结构 |
| Jacome et al., [NPN](https://arxiv.org/abs/2510.01608) | 对感知矩阵零空间学习非线性低维投影，并讨论收敛和精度 | “非线性零空间先验”也已被覆盖 |

### 4.2 学习型方向、Krylov 与预条件

| 一级来源 | 已经建立的内容 | 对 LGWO-A24 的边界影响 |
|---|---|---|
| Kaneda et al., [Deep Conjugate Direction Method](https://proceedings.mlr.press/v202/kaneda23a.html), [arXiv](https://arxiv.org/abs/2205.10763) | 神经网络在每次迭代改善 CG 搜索方向，近似逆算子作用 | 直接排除“首次学习 CG 方向”的主张；这是最强近邻之一 |
| Rudikov et al., [FCG-NO](https://proceedings.mlr.press/v235/rudikov24a.html) | 把离散无关神经算子作为 flexible CG 的非线性预条件器，跨分辨率复用 | 排除“首次把神经算子放入 CG/Krylov”的主张 |
| Li et al., [Learning Preconditioners for CG PDE Solvers](https://proceedings.mlr.press/v202/li23e.html), [arXiv](https://arxiv.org/abs/2305.16432) | 用 GNN 学近似矩阵分解作为 PCG 预条件器，并引入数据分布先验损失 | 排除宽泛学习型预条件器主张 |
| Lan et al., [Neural-Preconditioned Poisson Solver](https://proceedings.mlr.press/v235/lan24a.html) | 对变化域形状、边界条件和网格使用快速神经预条件器 | 几何变化下的神经预条件也不是空白 |
| Kopaničáková and Karniadakis, [DeepONet Based Preconditioning](https://epubs.siam.org/doi/full/10.1137/24M162861X), [DOI](https://doi.org/10.1137/24M162861X) | DeepONet 直接近似逆作用，或用 trunk basis 建低维校正子空间，再与 Krylov 结合 | 排除“DeepONet 加速 Krylov”主张 |
| Trivedi et al., [Data-driven acceleration of photonic simulations](https://arxiv.org/abs/1902.00090) | 用学习/PCA 的解子空间增强 GMRES | 排除宽泛学习子空间增强 Krylov 主张 |

### 4.3 神经校正、展开和 warm-start

| 一级来源 | 已经建立的内容 | 对 LGWO-A24 的边界影响 |
|---|---|---|
| Aggarwal et al., [MoDL](https://arxiv.org/abs/1712.02862), [DOI](https://doi.org/10.1109/TMI.2018.2865356) | CNN 先验与 CG 数据一致性模块交替，固定展开步数 | 固定预算经典求解器和神经模块组合已知 |
| Adler and Öktem, [Learned Primal-Dual](https://arxiv.org/abs/1707.06474), [DOI](https://doi.org/10.1109/TMI.2018.2799231) | 在固定投影/伴随预算中学习原始和对偶更新 | 排除“固定算子预算的 learned iterative reconstruction”宽泛主张 |
| Bhat et al., [Neural Correction Operator](https://arxiv.org/abs/2507.18875), [DOI](https://doi.org/10.1016/j.jcp.2026.115039) | 先做有限步经典 EIT 重建，再由神经算子校正 | 排除“有限迭代后神经校正”宽泛主张 |
| Sambharya et al., [Learning to Warm-Start Fixed-Point Algorithms](https://www.jmlr.org/papers/v25/23-1174.html) | 网络给 warm start，随后运行预定步数，并给出 PAC-Bayes 泛化界 | 排除“学习初始状态再固定迭代”的宽泛主张 |

### 4.4 DeepONet、FNO、iFNO 与几何

| 一级来源 | 已经建立的内容 | 对 LGWO-A24 的边界影响 |
|---|---|---|
| Lu et al., [DeepONet](https://www.nature.com/articles/s42256-021-00302-5), [DOI](https://doi.org/10.1038/s42256-021-00302-5), [code](https://github.com/lululxvi/deeponet) | branch/trunk 网络学习函数到函数的非线性算子 | “算子学习”本身不是新颖性 |
| Li et al., [Fourier Neural Operator](https://arxiv.org/abs/2010.08895), [official code](https://github.com/neuraloperator/neuraloperator) | 用 Fourier 层学习参数化 PDE 的解算子并跨离散迁移 | FNO 是必须比较的直接算子基线，不是可被一句“我们的模型更物理”略过的背景 |
| Molinaro et al., [Neural Inverse Operators](https://proceedings.mlr.press/v202/molinaro23a.html), [arXiv](https://arxiv.org/abs/2301.11167) | 组合 DeepONet/FNO 结构学习 PDE 逆算子 | 排除宽泛逆算子新颖性 |
| Long et al., [iFNO](https://proceedings.mlr.press/v258/long25a.html), [code](https://github.com/BayesianAIGroup/iFNO) | 可逆 Fourier blocks 与生成潜变量同时处理正逆问题 | iFNO 需要作为后续 L2 的逆算子基线 |
| Li et al., [Geo-FNO](https://jmlr.org/papers/v24/23-0064.html) | 用学习形变处理一般几何 | 排除“给 FNO 加几何条件就是新颖”的主张 |

### 4.5 留出测量、一致性与不确定性

| 一级来源 | 已经建立的内容 | 对 LGWO-A24 的边界影响 |
|---|---|---|
| Hendriksen et al., [Noise2Inverse](https://arxiv.org/abs/2001.11801), [DOI](https://doi.org/10.1109/TCI.2020.3019647) | 从测量分组构造统计独立重建，自监督训练 tomography denoiser | 独立测量信息用于训练已知 |
| Yaman et al., [SSDU](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28378), [DOI](https://doi.org/10.1002/mrm.28378) | 把实测 k-space 分成数据一致性集和训练损失集 | “heldout measurement loss”概念已知 |
| Haltmeier et al., [SPLIT](https://arxiv.org/abs/2604.15651) | 非线性 tomography 中跨分区一致性和 measurement-domain fidelity | 与 B 组一致性非常接近，必须在后续检索中持续跟踪 |
| Moya et al., [Conformalized-DeepONet](https://www.sciencedirect.com/science/article/abs/pii/S0167278924003683), [DOI](https://doi.org/10.1016/j.physd.2024.134418) | split conformal prediction 给出 distribution-free coverage interval | 说明当前 min/max 包络不能被称为有覆盖保证的 UQ |

### 4.6 BOST、NeRIF 与 TDBOST

| 一级来源 | 已经建立的内容 | 对 LGWO-A24 的边界影响 |
|---|---|---|
| Grauer et al., [Instantaneous 3D flame BOST](https://www.sciencedirect.com/science/article/pii/S0010218018302694), [DOI](https://doi.org/10.1016/j.combustflame.2018.06.022) | 23 相机实验重建瞬时三维火焰折射率场 | BOST 三维火焰重建本身已成熟 |
| Cai et al., [Direct BOST using RBF](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100), [DOI](https://doi.org/10.1364/OE.459872) | RBF 直接重建；12 台相机参与重建，另 3 台相机用于验证 | BOST 留出相机验证已有明确先例 |
| Bo et al., [BOST using GRU](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-23-39182), [DOI](https://doi.org/10.1364/OE.505992) | 利用相邻投影空间相关性直接做快速三维重建 | 排除“神经网络用于 BOST 重建”主张 |
| He et al., [NeRIF](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the), [DOI](https://doi.org/10.1063/5.0250899), [arXiv](https://arxiv.org/abs/2409.14722) | 用隐式神经场替代体素场，验证模拟与湍流 Bunsen flame | LGWO 必须解释自己为何不是另一个表示网络，而是 amortized solver-direction policy |
| Li et al., [NeDF](https://arxiv.org/abs/2409.19971), [DOI](https://doi.org/10.1063/5.0241191) | sparse-view BOST 的神经偏折场、非线性 ray tracing、无需预训练的逐例优化 | sparse-view 和非线性射线失配已被直接处理 |
| He et al., [TDBOST](https://doi.org/10.1145/3809488) | X-Y-Z-T 张量分解、轻量网络、畸变修正和可微 ray tracing 的四维逐序列优化 | 时空压缩表示与 4D BOST 已知；LGWO 的可能区别在跨例摊销和固定 CGLS 预算 |
| Lu et al., [Neural Refractive Index Primitives](https://arxiv.org/abs/2605.11454) | 紧凑隐式表示、hash encoding、离散梯度损失及火焰验证 | 进一步收窄“神经折射率表示”的主张空间 |

## 5. Claim matrix

| 子主张 | 最接近的已有工作 | 判定 | 当前未被证明的部分 | 必须补的实验或证据 |
|---|---|---|---|---|
| 神经网络增强经典逆问题求解器 | MoDL、Learned Primal-Dual、NCO | **already known** | 无 | 不可作为贡献标题 |
| 网络学习 CG 搜索方向 | DCDM | **already known** | 无 | DCDM-style 公平基线 |
| 神经算子作为 Krylov 非线性预条件器 | FCG-NO、DeepONet preconditioning | **already known** | 无 | FCG-NO/DeepONet-preconditioner 公平基线 |
| 学习初值并运行固定步数 | learned warm-start | **already known** | 无 | learned warm-start 基线 |
| 学习显式零空间修正 | Deep Null Space Learning、NPN | **already known**；LGWO 目前不满足 | LGWO 的 `A delta` 小是否真的对应稳定 near-null 机制 | 小规模 SVD 诊断、null/range 能量分解、exact-null oracle；仅作 evaluator |
| 几何条件化的神经算子 | Geo-FNO、神经预条件 Poisson solver | **already known** | BOST 几何编码是否产生额外有效信息 | no-geometry、geometry shuffle、跨 rig 反事实消融 |
| DeepONet/FNO/iFNO 做逆映射 | NIO、iFNO | **already known** | LGWO 是否在同等端到端成本和数据量下更稳 | L2 中统一数据、统一 split、统一算力和算子调用核算 |
| AI 做 BOST 三维/四维重建 | GRU-BOST、NeRIF、NeDF、TDBOST | **already known** | LGWO 是否能跨例摊销且保留固定物理求解器 | 与逐例优化和直接网络分别比较质量、时间、内存、预训练成本 |
| 保留 B 组做一致性信号 | Noise2Inverse、SSDU、SPLIT | **already known concept** | 独立 B ray bundle 用于首方向安全约束的具体收益 | no-B loss、same-ray B、new-camera B、真实 B 位移消融 |
| BOST 中留出相机验证 | Direct BOST using RBF | **already known** | B 进入训练而非只验证是否有稳定收益 | 严格区分 training B、selection B、sealed scoring B |
| 只修正第一方向，之后回到固定 CGLS shell | 未在本次有界检索中发现完全相同实现；DCDM 是每步方向改进 | **possibly differentiating** | 首方向为何足够、何时比每步学习更稳 | first-only、every-k、every-step、warm-start、fixed-direction 等预算匹配消融 |
| `||delta|| <= 0.05||g||` 的有界首方向 | 本次近邻论文未发现同一 BOST/CGLS 组合 | **possibly differentiating, not yet evidenced** | 0.05 是否有机制意义，还是偶然超参数 | 未来另行预注册半径曲线；报告方向角、A-delta 比、field gain 和失效区间 |
| 提案零算子调用并保持 24F/24AT | 固定展开预算已有先例，但这一精确合同未见同构论文 | **possibly differentiating as cost contract** | 同等端到端成本下是否仍占优 | 同时报告 A/A^T、B evaluator calls、训练反传等价成本、墙钟和内存 |
| fit-only 包络外精确退回 `delta=0` | UQ/OOD 文献广泛；当前只是经验包络 | **engineering safeguard, not novelty** | 能否可靠识别危险 OOD；覆盖率未知 | calibration coverage、false-safe、false-fallback、风险-覆盖曲线，与 conformal gate 比较 |
| unseen-rig 尾部门 | 属于稳健评价规范 | **evaluation strength, not algorithm novelty** | 当前没有 unseen-rig 数据 | 相机姿态、数量、分辨率、噪声、support、ray-model mismatch 的预注册 OOD 网格 |
| 五项窄组件同时出现 | 本次有界一级来源检索未找到完全相同论文 | **not yet evidenced** | 组合是否产生不可由简单基线解释的协同增益 | 完整基线、消融、机制、未见 rig、真实 BOST 和独立复现 |

## 6. 哪一部分最可能成为真正贡献

当前最有希望的不是“发明新的神经算子”，而是下面这个较窄的问题：

> 在 BOST 的严重欠定几何下，一个只使用部署可观测量、显式编码相机几何、只修改 CGLS 首方向、范数受限且不增加 A/A^T 调用的轻量提案，能否在未见 rig 的尾部同时改善三维场误差并保持 A/B 测量一致性；一旦输入越出 fit 校准域，能否精确退回原求解器？

这句话目前是**问题**，不是结果。若最终成立，差异点来自五者的耦合，而不是任一单个组件：

1. BOST 专用的 observable/geometry conditioning。
2. 只改首方向而不是直接输出三维场、每步替换 Krylov 更新或做完整预条件。
3. 相对范数界、`delta=0` 精确同构基线和零隐藏算子调用。
4. 独立 B 射线一致性与场误差共同约束。
5. 未见 rig 尾部门和确定性 fallback 合同。

最强的文献碰撞是 DCDM 和 FCG-NO。论文必须用实验证明 first-only/bounded/fixed-budget 结构不是把二者简化后换到 BOST 上。最强的 BOST 碰撞是 NeRIF、NeDF 和 TDBOST。论文必须说明 LGWO 学的是跨例摊销的**求解方向策略**，不是逐例拟合的隐式场或时空张量表示。

## 7. 一个可证明但不够单独构成新颖性的结构事实

若 `g` 与 `delta` 已按同一 support 限制，且

```math
||delta||_2 <= eta ||g||_2,  0 <= eta < 1,
```

则 `d_0=g+delta` 满足

```math
g^T d_0 >= (1-eta)||g||_2^2 > 0,
cos(g,d_0) >= (1-eta)/(1+eta).
```

对当前 `eta=0.05`，这个保守角度下界为 `0.95/1.05 = 0.90476`。在精确线性 A 和精确一维最小二乘下，第一步不会增加 A 组平方残差；后续每一步若分母有效，也沿所选方向做精确最小化。

但这个事实只保证：

- 修正方向不会完全背离初始伴随梯度；
- exact arithmetic 下 A 组训练模型残差可非增。

它**不保证** field relative-L2 改善、B 组一致性、噪声稳定性、真实光线模型失配下安全、未见 rig 泛化，也不证明 `delta` 位于零空间。该引理很简单，不能独立支撑“突破性理论”。它可以作为方法设计的可审计结构，再由实验决定有没有科学意义。

## 8. 支撑窄主张所需的最小实验包

### 8.1 同预算基线

必须至少包含：

| 基线 | 目的 | 公平约束 |
|---|---|---|
| zero-start CGLS24 | 主经典基线 | 24F/24AT |
| anchored shell with `delta=0` | 验证实现同构 | 输出、残差、调用逐项一致 |
| `1.05g` scaling control | 排除单纯缩放方向 | 精确线搜索下应与 `g` 等价 |
| fixed learned direction | 排除仅记住固定体素偏置 | 同 support、同 bound、同训练步 |
| linear geometry-conditioned correction | 排除小网络收益只是线性回归 | 同输入、同 bound |
| learned warm-start + CGLS | 对照 first-direction 与 initial-field 学习 | 总调用和总参数明确 |
| DCDM-style direction proposal | 对照每步学习方向 | A/A^T 和网络调用完整核算 |
| FCG-NO/neural preconditioner | 对照非线性预条件 | 同精度目标或同预算双报告 |
| DeepONet/FNO/iFNO direct inverse | 对照直接算子学习 | 同训练集、同 OOD split、同端到端成本 |
| NeRIF/NeDF/TDBOST 类逐例优化 | 对照 BOST 主线方法 | 分开报告预训练和逐例优化成本 |

不能只比较 DeepONet、FNO 的默认弱配置，也不能把 LGWO 的 24 次求解成本与直接网络的一次 forward 简单并列。至少同时给出“固定算子预算”和“固定墙钟/算力预算”两张表。

### 8.2 决定新颖性是否成立的消融

1. full model。
2. no geometry。
3. geometry shuffle 或错误 rig metadata。
4. g-only。
5. no raw observation。
6. no heldout-B loss。
7. fixed direction。
8. unbounded correction。
9. first-only 对 every-k 和 every-step。
10. empirical envelope 对 conformal risk gate。

当前 L1 已冻结 `eta=0.05` 且不允许 sweep。任何半径曲线都应放入**另行预注册的 L2**，不能回头用 route 结果挑选半径。建议未来候选为 `0, 0.01, 0.025, 0.05, 0.10`，并同时报告 field gain、方向夹角、`||A delta||/||A g||`、B 残差和 fallback。

### 8.3 机制证据

仅有平均误差提升不够。需要回答“为什么首方向修正有效”：

1. 小规模 evaluator-only SVD 将 `delta` 分解为 row-space 与 exact-null 成分。
2. 报告 `||P_null delta||/||delta||`、`||A delta||/||A g||` 与 field gain 的逐例关系。
3. 加入 exact-null oracle、range-only oracle 和随机等范数方向。
4. 检查网络是否只学到与 `g` 共线的无效缩放。
5. 按 A 的奇异向量/近零奇异值区间观察能量注入。
6. 检查收益是否集中在单一 phantom family、单一 rig 或少数 seed。

exact-null oracle 只能是诊断上界，不能伪装成可部署输入。

### 8.4 未见 rig 和真实物理

未见 rig 至少要分轴改变：

- 相机方位覆盖、仰角和抖动；
- 相机数量和射线密度；
- detector resolution 与背景采样；
- 噪声类型、噪声强度和相关噪声；
- support 尺寸、物理尺度和体素分辨率；
- 直线射线模型对折射弯曲模型；
- 标定偏差、位移提取误差和遮挡；
- 解析 phantom 对 LES/CFD，再到真实火焰。

每个 rig 都要报告 median、90/95 percentile、worst case、harm rate 和 fallback rate。只有真实 BOST 中使用独立相机或独立采集验证，并保持收益，才有资格讨论现实泛化。

### 8.5 指标和统计门

核心表应同时包含：

- field relative-L2，唯一主终点；
- H1/梯度误差；
- A measured residual、A clean residual、B clean/measurement residual；
- 每 rig 尾部、每 family、每 model seed；
- A/A^T 调用、B evaluator 调用、反向传播等价成本；
- inference wall time、training time、peak memory 和参数量；
- fallback、breakdown、nonfinite 和大于 1% field harm rate。

不能选最好 seed。bootstrap 单位应是 geometry cluster，而不是把同一几何下三个 family 当三个独立样本。当前 24-case synthetic route 即使通过，也只能授权 L2，不能直接写成泛化或算法成功。

## 9. 论文中允许与禁止的措辞

### 当前允许

> We investigate an observable-only, geometry-conditioned, norm-bounded first-direction augmentation of a fixed-budget CGLS reconstruction shell for synthetic BOST.

> The protocol is designed to test whether an independent-ray consistency signal and a deterministic baseline fallback can improve field reconstruction without increasing deployment calls to the forward or adjoint operator.

### 当前禁止

- “the first learned CGLS direction method”
- “a novel neural Krylov solver”
- “learned null-space projection”
- “provably safe reconstruction”
- “self-supervised BOST”
- “generalizes to unseen rigs”
- “outperforms DeepONet/FNO/iFNO”
- “real-time BOST”
- “breakthrough”

### 只有完整证据后才可考虑的窄表述

> A BOST-specific, observable-only and norm-bounded first-direction policy improved a fixed 24F/24AT reconstruction shell across preregistered unseen rigs without degrading heldout-ray consistency, while reverting exactly to the baseline outside its calibrated operating envelope.

即使这句话未来被结果支持，也不应使用 “first” 或 “provably safe”，除非投稿前重新完成跨数据库检索、理论条件精确定义和真实实验验证。

## 10. 发表门槛

| 阶段 | 最多能说什么 | 仍不能说什么 |
|---|---|---|
| 当前 0 case / 0 step | 协议已形成、存在一个待证窄假设 | 算法有效、新颖、泛化、突破 |
| L1 synthetic route 通过 | 在预注册小型 synthetic route 上存在可复核 signal，值得开 L2 | 超越 FNO、未见 rig 泛化、真实 BOST 成功 |
| L2 大规模公平 benchmark 通过 | 在多几何、多噪声、多模型失配下优于强基线 | 真实实验有效、普适安全 |
| 真实 BOST 独立相机验证通过 | 对指定装置与物理条件有实验价值 | 跨实验室普适泛化 |
| 外部复现或跨装置验证 | 才能讨论较强泛化与可转移性 | 无条件安全或所有 BOST 场景最优 |

对高质量论文而言，最低可接受包应是：强基线公平比较、机制消融、未见 rig 尾部、真实 BOST 独立视角验证、成本核算、失败案例以及无选择性 seed 报告。单一 synthetic mean gain 即使显著，也不够。

## 11. 链接和标识符核验记录

核验日期为 2026-07-20。所有 arXiv `abs` 链接均现场返回可读条目；所有 DOI 均由 `doi.org` 解析到对应出版社。SIAM、AIP、ACM、Wiley 的部分自动化请求在出版社端返回 403，但 DOI 解析目标、出版社元数据或可读主页面已交叉确认，这不代表链接失效。

| 标识符 | 核验结果 |
|---|---|
| `10.1088/1361-6420/aaf14a` | DOI -> IOP；arXiv 1806.06137 可读 |
| `10.1109/TMI.2018.2865356` | DOI -> IEEE document 8434321；arXiv 1712.02862 可读 |
| `10.1109/TMI.2018.2799231` | DOI -> IEEE document 8271999；arXiv 1707.06474 可读 |
| `10.1016/j.jcp.2026.115039` | DOI -> Elsevier PII S002199912600392X；arXiv 2507.18875 可读 |
| `10.1137/24M162861X` | DOI -> SIAM JSC 正式页面；题名、作者和摘要已核 |
| `10.1038/s42256-021-00302-5` | DOI -> Nature Machine Intelligence；arXiv 1910.03193 可读 |
| `10.1109/TCI.2020.3019647` | DOI -> IEEE document 9178467；arXiv 2001.11801 可读 |
| `10.1002/mrm.28378` | DOI -> Wiley 正式全文页；方法中的 disjoint measurement split 已核 |
| `10.1016/j.physd.2024.134418` | DOI -> Elsevier PII S0167278924003683；coverage 主张已核 |
| `10.1016/j.combustflame.2018.06.022` | DOI -> Elsevier PII S0010218018302694 |
| `10.1364/OE.459872` | DOI -> Optica 30(11), 19100；12 reconstruction + 3 validation cameras 已核 |
| `10.1364/OE.505992` | DOI -> Optica 31(23), 39182；GRU-BOST 元数据和摘要已核 |
| `10.1063/5.0250899` | DOI -> AIP Physics of Fluids 37, 017143；arXiv 2409.14722 可读 |
| `10.1063/5.0241191` | DOI -> AIP Physics of Fluids 36, 121701；arXiv 2409.19971 可读 |
| `10.1145/3809488` | DOI -> ACM TOG；题名、作者、4D tensor/NN/ray-tracing 摘要元数据已核 |
| `arXiv:2510.01608` | NPN v2 可读，条目标注 NeurIPS 2025 accepted |
| `arXiv:2604.15651` | SPLIT 条目可读 |
| `arXiv:2605.11454` | Neural Refractive Index Primitives 条目可读 |

## 12. 最终边界

LGWO-A24 目前**不能**以“学习零空间”“神经 Krylov”“算子学习 BOST”中的任何一个宽泛标签宣称新颖，因为这些方向均已有直接一级来源。当前唯一值得继续检验的是一个更窄的组合：

> observable geometry-conditioned bounded first-direction correction inside a fixed-budget CGLS shell, evaluated with independent B consistency, deterministic fail-closed fallback, and unseen-rig tail gates.

本次有界检索没有找到把这五项以同一 BOST 部署合同组合起来的论文，但这仍然只是“possibly differentiating”。在 **0 scientific cases / 0 optimizer steps / no breakthrough** 的状态下，新颖性结论必须保持 `NO-GO`。下一步不是润色 claim，而是运行预注册实验、补强近邻基线、证明 first-only 机制、测试 OOD 尾部，并取得真实 BOST 独立视角证据。
