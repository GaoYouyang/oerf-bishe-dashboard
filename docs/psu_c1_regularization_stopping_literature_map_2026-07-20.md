# PSU-C1 之后：正则化与停止策略的角色化文献地图

> 状态：研究路线说明，不是算法结果。更新时间：2026-07-20。
>
> 适用范围：公开 PSU 九视角几何、解析反应场和后续经何远哲师兄确认的 OERF BOST callable。
> 当前证据：PSU-C1 的 bounded first-direction truth oracle 在 IID / family-OOD 仅有
> `+1.1288% / +1.2465%` field gain，未过预注册 5% 门；inverse-Sobolev5 虽有
> `+51.5234% / +45.0628%` analytic field gain，却把 active measured residual 推到
> `1.6752x / 4.5303x`。因此下一个问题不是“换更大网络”，而是“怎样在可观测信息下选择正则化和停止，并拒绝会破坏数据一致性的动作”。

---

## 0. 先说结论：现在最值得做什么

当前最合理的本科毕设候选不是直接训练 DeepONet、FNO 或新的三维生成网络，而是：

**在固定的 BOST 物理算子与 Krylov 重建壳中，学习一个低维、受约束、可回退的正则化/停止控制器。**

控制器只允许读取真实重建时可得到的量：

- 几何摘要：相机数、角覆盖、ray/path 长度分布、近似 condition/Ritz 值；
- 噪声摘要：flow-off noise floor、位移置信度、逐相机协方差；
- 迭代轨迹：`||r_k||`、`||A^T r_k||`、相邻迭代差、残差曲率、少量 Lanczos 标量；
- 独立一致性：不参与该步更新的 held-out view / B-operator residual；
- 可选时序摘要：连续帧创新量，但只有真实 4D 数据合同到位后才启用。

控制器只允许输出：

1. 有界的 `lambda_k` 或 `alpha_k`；
2. 预先固定的 2--4 个正则基之间的非负凸组合；
3. `continue / stop / reject`；
4. 触发 deterministic fallback 的拒答标志。

它**不直接生成三维场**，不读取真值，不绕开 `A/A^T`，也不允许在新 rig 上无边界外推。

这个设计的研究价值来自三件事的交集，而不是某一个网络名字：

- BOST 的稀疏视角、薄界面和几何变化；
- 大规模不适定逆问题中的半收敛与 hybrid projection；
- 带安全包络、成本匹配和独立视角验证的学习型求解器控制。

---

## 1. 初学者先理解的物理与数学冲突

### 1.1 为什么残差变小不等于三维场更真实

线性化 BOST 可写成：

```text
y = A x + epsilon
```

`x` 是折射率/密度相关三维场，`y` 是多相机位移观测，`A` 是由几何和光线路径决定的投影算子。
在稀疏视角下，很多不同的 `x` 会产生相近的 `A x`。因此只压低训练相机的
`||A x-y||`，可能是在吸收噪声、标定偏差或几何空缺，而不是恢复正确体场。

PSU-C1 的 inverse-Sobolev 对照正好把这个冲突显出来：它大幅改善 analytic field-L2 和 held-out B proxy，
但损害 active measured residual。当前只能说“正则化方向值得进一步分解”，不能说这个对照已经胜出。

### 1.2 什么是半收敛

对不适定问题，CGLS/LSQR 等方法常先恢复由较大奇异值承载的稳定结构，继续迭代后才逐渐放大噪声方向。
于是 field error 可能先降后升，而 measurement residual 仍继续下降。最佳停止步本身就是一种正则化。

毕业设计中必须画出每个 case 的四条轨迹，而不是只报最终一步：

```text
iteration k
  -> field relative-L2 (仅 synthetic / phantom 可用)
  -> active measured residual
  -> held-out B residual
  -> gradient/H1 或界面误差
```

### 1.3 为什么“学习停止”比“学习整个逆算子”更可控

直接学习 `y -> x` 要承担几何变化、分辨率变化、噪声变化和训练分布外结构的全部风险。学习一个低维控制量，
则可以把正确性主体留给 `A/A^T` 与经典迭代器，并用硬范围、独立残差和回退保护失败。这不保证一定更好，
但更适合数据有限、本机算力有限、且需要向师兄解释每一步物理含义的本科项目。

---

## 2. 文献角色图：先读什么、从中拿什么、不能声称什么

### A 组：必须先读的逆问题地基

| 优先级 | 一级来源 | 在本项目中的角色 | 重点提取 | 不能据此声称 |
|---|---|---|---|---|
| A1 | Chung & Gazzola, **Computational Methods for Large-Scale Inverse Problems: A Survey on Hybrid Projection Methods**, SIAM Review 2024, [DOI](https://doi.org/10.1137/21M1441420) | 总地图 | Krylov 子空间、投影问题、显式/隐式正则化、参数选择和半收敛怎样接起来 | hybrid Krylov 是我们的新算法 |
| A2 | Hansen, **Discrete Inverse Problems: Insight and Algorithms**, SIAM 2010, [book DOI](https://doi.org/10.1137/1.9780898718836) | 数学教材 | SVD/filter factors、Picard condition、L-curve、discrepancy principle、迭代正则化 | 仅靠 residual 下降就能证明重建更真 |
| A3 | Gazzola, Hansen & Nagy, **IR Tools**, Numerical Algorithms 2019, [DOI](https://doi.org/10.1007/s11075-018-0570-7), [official project](https://people.compute.dtu.dk/pcha/IRtools/) | 可复现经典基线 | 同一接口下的 CGLS/LSQR/hybrid/停止规则与测试问题组织方式 | 2D MATLAB demo 等同三维 BOST |
| A4 | Chung & Gazzola, **Flexible Krylov Methods for lp Regularization**, SISC 2019, [DOI](https://doi.org/10.1137/18M1194456) | 可变正则机制 | flexible Golub-Kahan、迭代重加权、低维投影内更新正则结构 | 学习可变权重本身具有新颖性 |
| A5 | Kilmer & O'Leary, **Choosing Regularization Parameters in Iterative Methods for Ill-Posed Problems**, SIMAX 2001, [DOI](https://doi.org/10.1137/S0895479899345960) | 经典参数选择基线 | projected problem 内选参数，怎样避免每次在全空间做昂贵搜索 | learned lambda 可不与经典规则比较 |
| A6 | Hansen & Jorgensen, **AIR Tools II**, Numerical Algorithms 2018, [DOI](https://doi.org/10.1007/s11075-017-0430-x), [official project](https://people.compute.dtu.dk/pcha/AIRtoolsII/) | 稀疏角层析基线 | ART/SIRT、relaxation、停止启发式和 tomography test problems | CT 的角度/噪声合同能直接替代 BOS/BOST |

**本科阅读目标：**读完 A1、A2 后，能够用自己的话解释为什么“迭代次数”与“正则化参数”都是模型选择；
读 A3/A6 时只复现一两个小问题，不必先把全部 MATLAB 工具迁移到 PyTorch。

### B 组：学习什么，而不是让网络接管什么

| 优先级 | 一级来源 | 在本项目中的角色 | 重点提取 | 必须避免的误用 |
|---|---|---|---|---|
| B1 | Afkham, Chung & Chung, **Learning Regularization Parameters of Inverse Problems via Deep Neural Networks**, [arXiv](https://arxiv.org/abs/2104.06594) | 最直接的先行工作 | 从 observation 到 instance-specific regularization parameter 的监督映射；训练标签怎样得到 | 不能把“学习 lambda”当成本项目新意；必须增加 BOST 物理、轨迹与 fail-closed 约束 |
| B2 | Sambharya et al., **Learning to Warm-Start Fixed-Point Optimization Algorithms**, JMLR 2024, [record/PDF](https://www.jmlr.org/papers/v25/23-1174.html) | 理论与评估边界 | 学习部分与确定性 fixed-point steps 的组合、残差/真值两类 loss、跨迭代步的测试 | 不能把 warm-start 的一般框架写成 BOST 特有发明 |
| B3 | Kaneda et al., **A Deep Conjugate Direction Method**, ICML 2023, [PMLR](https://proceedings.mlr.press/v202/kaneda23a.html) | learned direction 强基线与已关闭路线 | 网络改善搜索方向、卷积逆近似、跨问题规模实验 | PSU-C1 已表明当前 first-direction headroom 不足，不能照搬并预称有效 |
| B4 | Rudikov et al., **FCG-NO**, ICML 2024, [PMLR](https://proceedings.mlr.press/v235/rudikov24a.html) | learned preconditioner 强基线 | nonlinear preconditioner 与 flexible CG 怎样合法组合、跨分辨率评估 | 若没有真正实现 FCG，就不能称自己的模块为 FCG neural preconditioner |
| B5 | Aggarwal, Mani & Jacob, **MoDL**, IEEE TMI 2019, [DOI](https://doi.org/10.1109/TMI.2018.2865356), [open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC6760673/) | unrolled data-consistency 基线 | 共享权重、CG data-consistency block、模型与学习先验的交替 | MRI 结果不能作为 BOST 成绩；大 unroll 不是当前首选 |
| B6 | Adler & Oktem, **Learned Primal-Dual Reconstruction**, IEEE TMI 2018, [arXiv](https://arxiv.org/abs/1707.06474), [code](https://github.com/adler-j/learned_primal_dual) | 端到端 unrolling 上界对照 | forward/backprojection 嵌入网络、固定调用数、与 TV/FBP 对照 | 只有算力和真实数据合同到位后才适合作为大模型 baseline |
| B7 | Schwab, Antholzer & Haltmeier, **Deep Null Space Learning**, Inverse Problems 2019, [DOI](https://doi.org/10.1088/1361-6420/aaf14a), [arXiv](https://arxiv.org/abs/1806.06137) | data-consistency 结构边界 | 把学习校正限制到 null-space/一致结构，理解 regularization property | 不能再声称“首次用 null-space 网络保护数据一致性” |

**本项目从 B 组得到的不是网络清单，而是三条纪律：**

1. 学习对象越小，越容易审计其跨 rig 行为；
2. 必须把 deterministic solver 留在环中；
3. 学习结果要与同预算经典参数选择、fixed policy、oracle 和直接网络分别比较。

### C 组：OERF/BOST 宿主与新颖性边界

| 优先级 | 一级来源 | 在本项目中的角色 | 重点提取 | 不能外推 |
|---|---|---|---|---|
| C1 | He et al., **Neural Refractive Index Field**, Physics of Fluids 2025, [DOI](https://doi.org/10.1063/5.0250899), [open HTML](https://arxiv.org/html/2409.14722v2) | 何远哲师兄最近主线 | 折射率场/梯度表示、位移一致性、采样与实验评价；逐项确认真实 callable | 公开论文无法替代组内代码、标定和误差层级合同 |
| C2 | He et al., **Tensor Decomposition-Based Four-dimensional BOST**, ACM TOG 2026, [DOI](https://doi.org/10.1145/3809488) | 4D 宿主与低秩边界 | 时空张量分解、速度/保真度权衡、真实时序数据接口 | 没有连续真实 run 时不能启动“4D 更好”的主张 |
| C3 | Molnar et al., **Open-source BOS tomography dataset of high-speed flow over a flight body**, [PSU Data Commons](https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/) | 当前公开 callable 演练场 | loader、几何、位移/视角组织、held-out view 和成本测量 | post-open geometry rehearsal 不是未见 rig 泛化，也不是 OERF 火焰数据 |
| C4 | Lu et al., **Neural Refractive Index Primitives for Flame Field Reconstruction Using BOS**, 2026, [arXiv](https://arxiv.org/abs/2605.11454) | 最新碰撞边界 | single RI primitive、hash encoding、automatic/discrete gradient、mask 与层级采样 | 这些组件不能再单独作为我们的创新点 |

**必须向师兄核对的关键差异：**真实 OERF pipeline 的主误差究竟来自 sparse-view null-space、straight/curved-ray bias、
相机标定、位移估计、有限孔径，还是 4D 算力。如果答案不是半收敛/正则化，本文路线必须降级为辅线。

---

## 3. 候选算法：先从可证伪版本开始

暂用工作名 **OARS-BOST**：Observable Adaptive Regularization and Stopping for BOST。
这个名字只用于整理实验，不是论文题目，也不构成新颖性声明。

### 3.1 固定求解器壳

以 CGLS/LSQR 或经确认的组内 solver 为主干：

```text
input: y, geometry metadata, noise metadata
x_0 = deterministic initialization

for k = 0 ... K_max - 1:
    compute the ordinary Krylov state using A and A^T
    z_k = observable_summary(state_k, metadata)
    u_k = controller(z_k)
    u_k = project_to_registered_safe_set(u_k)
    x_{k+1} = deterministic_hybrid_step(state_k, u_k)

    if envelope_violation:
        return deterministic_fallback
    if u_k.stop:
        return x_{k+1}

return registered_K_max_solution
```

### 3.2 允许的控制变量

最小版本只输出两个标量：

```text
lambda_k in [lambda_min, lambda_max]
p_stop,k in [0, 1]
```

升级版本才允许输出固定谱/正则基的凸权重：

```text
w_k = softmax(q_k), sum_j w_kj = 1, w_kj >= 0
R_k = sum_j w_kj R_j
```

候选 `R_j` 必须在预注册前固定，例如 identity、first derivative、inverse-Sobolev-like 平滑和弱 edge-preserving 近似；
不能根据 test 结果临时增加。

### 3.3 物理安全包络

每一步都检查：

- active measured residual 不超过 matched deterministic baseline 的冻结包络；
- held-out B residual 不连续恶化；
- `lambda_k`、谱权重和停止步在注册范围内；
- NaN、负物理量、异常高频能量或视角不完整时直接 fallback；
- 新 rig 超出 calibration envelope 时拒答，而不是强行输出。

注意：这些结构只能降低某些失败模式，不能自动证明安全或泛化。

---

## 4. 五级模型阶梯：任何一级失败就停止升级

| 级别 | 模型 | 目的 | Mac 预计成本 | 升级门 |
|---|---|---|---|---|
| S0 | 固定 `lambda` 网格 + discrepancy/GCV/held-out-B stopping | 建立经典可重复基线，观察半收敛是否真实存在 | 小；CPU/MPS 均可 | 至少一个 split 上 oracle stopping 对 fixed-K 有稳定、非泄漏 headroom |
| S1 | ridge / logistic / monotone GAM | 验证可观测特征是否含有 instance-specific 信号 | 很小 | IID 与 family-OOD 均优于最强 fixed policy，且逐 rig 尾部不恶化 |
| S2 | 2 层小 MLP，输出 bounded `lambda` + stop | 检查非线性关系是否值得 | 小 | 在参数匹配、成本匹配下显著优于 S1，而非只拟合 train |
| S3 | trajectory policy：GRU/TCN 或小型 set/sequence model | 利用多步残差和 Ritz/Lanczos 轨迹 | 中 | 对 geometry/noise/joint shift 均有一致增益，并通过独立 B 与 reject 校准 |
| S4 | 4D recurrent/low-rank policy | 在连续真实 run 中共享时序信息 | 大且数据依赖 | 何师兄提供真实时序、timestamp、曝光、dropout、强 TDBOST baseline 后才开放 |

**严禁的跳步：**S0 没有显示 stopping/regularization headroom 时，不能直接租 GPU 训练 S3/S4。

---

## 5. 可复现实验矩阵

### 5.1 数据分区

最低限度需要六类：

| split | 变化 | 回答的问题 |
|---|---|---|
| IID | 与训练同族、不同随机实例 | 模型是否至少学到可重复信号 |
| family-OOD | 未见反应场形态/薄界面族 | 是否只记住 phantom 模板 |
| noise shift | 未见噪声强度/协方差 | stopping 是否对 noise floor 敏感 |
| geometry shift | 未见相机子集/角度扰动 | 几何摘要是否真的起作用 |
| joint shift | family + noise + geometry | 最接近实际迁移压力 |
| exact/replay | 冻结可复算小样本 | 检查实现、数值和审计一致性 |

真实数据到位后，再增加按 **rig/session/run** 分组的独立测试；禁止随机拆帧制造泄漏。

### 5.2 必须比较的基线

1. CGLS/LSQR 固定 `K`；
2. fixed `lambda` + fixed stopping；
3. discrepancy principle / GCV / hybrid projected selection；
4. 使用同一特征的 ridge/logistic；
5. oracle `lambda/stop`，只用于估计上限，不参与部署；
6. S2/S3 learned policy；
7. 数据和算力到位后，MoDL/LPD 或组内最强 learned reconstruction；
8. 若声称“learned Krylov/preconditioner”，必须实现 DCDM/FCG-NO 风格对照，而不是只引用。

### 5.3 指标门

主指标必须同时报告：

- field relative-L2（仅有真值时）；
- gradient/H1 或界面带误差；
- active measured residual；
- held-out B / held-out view residual；
- 每个 rig 的 median、p10/p90、worst 和 harm count；
- `A/A^T` 调用数、fit setup 成本、推理 wall time、峰值内存；
- reject coverage、selective risk 和 fallback 后的最终成绩。

不能用平均 field gain 掩盖某一 rig 的灾难性恶化，也不能只用 PSNR/SSIM 代替物理一致性。

### 5.4 最低发表门槛

只有同时满足下列条件，才可以把候选升级为“方法证据”：

1. 预注册后冻结数据分区、超参范围、基线、指标和失败判据；
2. 至少两个独立公开/真实合同，其中一个是未见 rig 或 session；
3. 在同 `A/A^T` 或同端到端成本下击败最强经典控制，而不是弱 baseline；
4. field/H1 与 A/B consistency 不出现未解释冲突；
5. 逐 rig 尾部、harm count 和置信区间通过；
6. 完成 feature、lambda、stop、B-validation、fallback 的逐项消融；
7. 独立脚本能从 raw rows 复算表格和图；
8. 何远哲师兄确认问题、接口和比较对象确实对应组内痛点。

当前一个条件也不能靠 PSU-C1 的 descriptive signal 自动视为满足。

---

## 6. 先在 Mac 上做的最小实验

### Phase R0：不训练，画清半收敛（1--2 天）

对冻结 PSU-C1 case 运行 `k=1...K_max` 的 deterministic trajectory，保存每一步的 field、A measured、A clean、B、H1、
Ritz 值和成本。先回答：是否真的存在随 case/geometry/noise 改变的最佳停止步？

**GO 条件：**oracle stopping 相对最佳全局 fixed-K 在 IID 和 family-OOD 都有稳定、非零且尾部可接受的 headroom。

### Phase R1：解析与经典选择（2--4 天）

实现 fixed grid、discrepancy、GCV/UPRE（若噪声合同足够）和 hybrid projected Tikhonov。严格记录额外
`A/A^T`、SVD/投影与超参搜索成本。

**GO 条件：**至少一个真实可用的 observable rule 接近 oracle，且不过度牺牲 active residual 或 B。

### Phase R2：线性可观测策略（2--3 天）

用 ridge 预测 `log lambda*`，用 logistic/ordinal model 预测 stop。训练标签只能来自 train/validation oracle，
test 不参与阈值选择。

**GO 条件：**两主分区都优于最强 fixed/经典 rule，并对 geometry/noise shift 不增害。

### Phase R3：小网络（3--7 天）

只有 R2 通过才用 2 层 MLP；保持参数量、训练时长和所有随机种子公开。网络必须经过输出投影和 fallback。

**GO 条件：**相对 R2 有明确增量，而不是仅相对 fixed-K。

### Phase R4：真实接口迁移（数据到位后）

冻结 synthetic 阶段，不在看到真实 test 结果后回改主假设。用师兄提供的 calibration/flow-off 数据重估
observable normalization 和 reject envelope，再做按 rig/session 的独立验证。

---

## 7. 现在要问何远哲师兄的九个问题

1. 目前真实重建主 callable 是 matrix-free `A/A^T`、NeRIF autograd，还是别的接口？
2. straight-ray、curved-ray、有限孔径和位移估计分别在哪一层形成 residual？
3. 是否能得到合法的 JVP/VJP，还是只有完整 forward 与反传？
4. 组内最强三维基线是哪一个版本，固定预算和真实 wall time是多少？
5. 真实失败最常见的是 field 模糊、薄界面丢失、reprojection、标定偏差，还是运行时间？
6. 能否提供 flow-off noise、相机协方差、标定不确定度和 per-view confidence？
7. 数据能否按 rig/session/run 划分，还是只有同一装置的连续帧？
8. 论文评价中最被重视的是 density/RI truth、held-out view、PIV compensation，还是 4D temporal fidelity？
9. 师兄是否认同先做“regularization/stopping controller”，并把 NeRIF/TDBOST 当宿主，而不是再做 direct inverse？

在这些答案到位前，下一步只做公开 PSU 的 R0/R1，不启动 4D 网络或真实论文结论。

---

## 8. 建议阅读顺序与学习产出

### 第一周：看懂为什么需要停止

1. A2：SVD、Picard、filter factor；
2. A1：hybrid projection 前半；
3. A3：跑一个 CGLS/LSQR semi-convergence demo；
4. C1：标出 NeRIF 的 forward、loss、observable 与 truth 分别在哪里。

产出：一张自己的 `field error vs residual vs iteration` 图，以及 500 字解释。

### 第二周：建立强经典基线

1. A5：projected parameter selection；
2. A4：flexible Krylov 与可变 regularizer；
3. A6：tomography stopping/relaxation；
4. 在 PSU-C1 上完成 R0/R1 预注册。

产出：固定 K、discrepancy、GCV/hybrid、oracle stopping 的同预算表。

### 第三周：只学最小控制量

1. B1：学习 lambda；
2. B2：learned component + deterministic iterations；
3. B7：data consistency / null-space 结构；
4. 实现 S1，先不写 MLP。

产出：逐 split、逐 rig、harm count 与 reject curve。

### 第四周：决定是否值得上神经网络

1. B3/B4：learned direction/preconditioner 的强比较；
2. B5/B6：unrolling 的数据与算力成本；
3. C2/C4：4D 与最新 BOS neural primitive 的新颖性边界；
4. 根据 R2 判决 GO/NO-GO。

产出：一页给师兄的结果卡，只能写已验证结论和明确失败。

---

## 9. 当前可写进开题报告的谨慎表述

> 本课题拟研究稀疏视角 BOST 三维重建中的实例自适应正则化与迭代停止。方法保留显式光学投影算子和
> Krylov/混合投影求解器，仅从几何、噪声与残差轨迹等部署时可观测量预测受限正则参数、停止或拒答，
> 并通过 active/held-out 视角一致性、逐装置尾部风险和端到端成本进行验证。现有 PSU-C1 解析演练只表明
> bounded first-direction 学习在当前公开 benchmark 上缺乏足够上限，并暴露了 field error 与 measurement
> consistency 的正则化张力；尚未证明所提控制策略、真实 BOST 或跨装置泛化有效。

这段话可以用于与师兄讨论方向；在 R0/R1 和真实接口回答到位前，不应写“提出了一种优于 DeepONet/FNO 的新算法”。

---

## 10. 与现有证据包的连接

- [PSU-C1 完整 NO-GO 判决](lgwo_a24_psu_c1_simple_controls_no_go_2026-07-20.md)
- [14 天保姆式学习路线](lgwo_a24_14_day_caretaker_route_2026-07-20.md)
- [论文工作稿](lgwo_a24_registered_manuscript_working_draft_2026-07-20.md)
- [给何远哲师兄的一页接口问题](n5_d5_advisor_first_contact_2026-07-19.md)
- [讲人话学习日志](operator_3d_learning_log.md)

最后的判决原则只有一句：**先证明部署时可观测量真的能预测“何时、用多强的正则”，再决定是否需要神经网络；
如果经典 hybrid/stopping 已经解释全部收益，就把 NO-GO 当成有价值的毕业设计结果，而不是硬造模型。**
