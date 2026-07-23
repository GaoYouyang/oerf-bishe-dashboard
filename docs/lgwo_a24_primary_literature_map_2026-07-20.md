# LGWO-A24 一级来源文献地图：哪些必须复现，哪些限制新颖性

**检索截止：** 2026-07-20

**使用规则：** 只把出版社、正式会议、作者 arXiv 全文和作者代码当一级来源。以下“作用”是对本项目的推断，不等于原论文作者的结论。

**新颖性结论：** 尚未发现与“有限角 3D BOST + 可变射线几何 + 极少 `A/A^T` 调用 + CGLS 初始方向 + measurement-consistency fail-closed fallback”完整组合相同的先例；但每个组成模块都有强先例。不能写“首次用 neural operator warm-start CG”。

## 1. 第一圈：先读懂师兄与 BOST 对手

### 1.1 NeRIF：OERF 主基线

Yuanzhe He et al., **Neural refractive index field: Unlocking the Potential of Background-oriented Schlieren Tomography in Volumetric Flow Visualization**, Physics of Fluids, 2025.

- [DOI / AIP](https://doi.org/10.1063/5.0250899)
- [arXiv HTML](https://arxiv.org/html/2409.14722v2)
- 要提取：ray model、折射率/梯度表示、loss、sampling、synthetic/experimental protocol、运行时间。
- 对 LGWO 的作用：必须实现或取得组内代码的逐场景 neural implicit 主基线。
- 不能声称：NeRIF 不是预训练 measurement-to-field operator；它没有自动证明未见 rig 泛化或低在线 `A/A^T` 成本。

### 1.2 TDBOST：4D 师兄主线

Yuanzhe He et al., **Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction**, ACM TOG, 2026.

- [DOI](https://doi.org/10.1145/3809488)
- 当前安全提取：题名、作者、4D BOST 与 tensor decomposition 定位。
- 对 LGWO 的作用：把贡献放在快速初始方向、变 rig、安全门或固定 rig 多帧摊销，不重新包装“4D 低秩”。
- 不能声称：在拿到正式全文前，不引用具体张量结构、速度、显存、误差或泛化数字。

### 1.3 Neural Refractive Index Primitives：当前代码型强对手

Xinyi Lu et al., **Neural Refractive Index Primitives for Flame Field Reconstruction Using Background-Oriented Schlieren**, 2026.

- [DOI](https://doi.org/10.1016/j.combustflame.2026.115082)
- [arXiv](https://arxiv.org/abs/2605.11454)
- [作者代码](https://github.com/Weihu22/Neural-Implicit-Reconstruction-for-BOS)
- 要提取：hash/Fourier encoding、mixed gradient、3D mask、hierarchical ray sampling、真实火焰 held-out reprojection。
- 对 LGWO 的作用：逐场景 neural implicit BOST 的强精度/速度基线。
- 不能声称：hash encoding、连续 refractive-index primitive 或 reprojection self-supervision 是本项目创新。

### 1.4 NeDF：稀疏视角与非线性射线

Jiawei Li et al., **Neural deflection field for sparse-view tomographic background oriented Schlieren**, Physics of Fluids, 2024.

- [DOI](https://doi.org/10.1063/5.0241191)
- [arXiv](https://arxiv.org/abs/2409.19971)
- 要提取：density-gradient field、positional encoding、hierarchical sampling、nonlinear ray tracing、view count。
- 对 LGWO 的作用：有限视角和 curved-ray 条件下必须对照。
- 不能声称：坐标网络、稀疏视角 BOS 或非线性 ray tracing 本身新颖。

### 1.5 GRU-BOST：预训练直接重建先例

Lin Bo et al., **Background-oriented Schlieren tomography using gated recurrent unit**, Optics Express, 2023.

- [DOI](https://doi.org/10.1364/OE.505992)
- 要提取：view ordering、ResNet/GRU 输入、训练场族、推理速度、相机数变化。
- 对 LGWO 的作用：证明“预训练网络直接从 BOST 投影到 3D”已有先例。
- 不能声称：顺序编码、快速 inference 或免逐场优化本身新颖。

### 1.6 经典真实 BOST

Samuel J. Grauer et al., **Instantaneous 3D flame imaging by background-oriented schlieren tomography**, Combustion and Flame, 2018.

- [DOI](https://doi.org/10.1016/j.combustflame.2018.06.022)
- 要提取：23-camera setup、Tikhonov/TV、标定、真实 flame evaluation。
- 对 LGWO 的作用：强经典物理基线与实验报告范式。
- 不能声称：只胜过 zero-regularized CG 就胜过传统 BOST。

Samuel J. Grauer, Adam M. Steinberg, **Fast and robust volumetric refractive index measurement by unified background-oriented schlieren tomography**, Experiments in Fluids, 2020.

- [DOI](https://doi.org/10.1007/s00348-020-2912-1)
- 要提取：图像畸变与重建联合优化、端到端成本。
- 对 LGWO 的作用：提醒位移前端成本和模型误差也要计入。

## 2. 第二圈：operator 与 warm-start 必须对比

### 2.1 通用 neural operator

Lu Lu et al., **Learning nonlinear operators via DeepONet**, Nature Machine Intelligence, 2021.

- [论文](https://www.nature.com/articles/s42256-021-00302-5)
- [代码](https://github.com/lululxvi/deeponet)
- 基线角色：branch 编 observation，trunk 查 3D coordinate。
- 边界：通用逼近不等于有限角逆稳定、data consistency 或任意 rig 泛化。

Zongyi Li et al., **Fourier Neural Operator for Parametric PDEs**, ICLR 2021.

- [论文](https://openreview.net/forum?id=c8P9NQVtmnO)
- [维护代码](https://github.com/neuraloperator/neuraloperator)
- 基线角色：固定规则体素网格的低成本 backbone。
- 边界：不原生处理不规则 ray set、inverse operator 或相机变化。

Alasdair Tran et al., **Factorized Fourier Neural Operators**, ICLR 2023.

- [论文](https://openreview.net/forum?id=tmIiMPl4IPa)
- [代码](https://github.com/alasdairtran/fourierflow)
- 基线角色：更轻的频谱 backbone；不能把 factorized spectrum 当本项目创新。

Roberto Molinaro et al., **Neural Inverse Operators for Solving PDE Inverse Problems**, ICML 2023.

- [PMLR 全文](https://proceedings.mlr.press/v202/molinaro23a.html)
- 基线角色：多探测响应算子到未知函数，最接近 observation-set-to-field 的理论先例。
- 边界：没有 BOST 特有的一致性、有限角 null-space 或调用预算保证。

Zongyi Li et al., **Geo-FNO** / **GINO**, 2023.

- [Geo-FNO](https://www.jmlr.org/papers/v24/23-0064.html)
- [GINO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)
- 基线角色：不规则域、SDF 与 geometry-conditioned latent grid。
- 边界：物体域几何变化不等于 measurement ray-set 变化。

### 2.2 learned solver 与 data consistency

Hemant K. Aggarwal et al., **MoDL: Model-Based Deep Learning Architecture for Inverse Problems**, IEEE TMI, 2019.

- [DOI](https://doi.org/10.1109/TMI.2018.2865356)
- [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6760673/)
- [代码](https://github.com/hkaggarwal/modl)
- 基线角色：CNN prior 与 CG data-consistency 交替。
- 边界：“网络 + CG”与 model-based unrolling 已有先例；penalized consistency 不等于精确 `Ax=y`。

Jonas Adler, Ozan Oktem, **Learned Primal-Dual Reconstruction**, IEEE TMI, 2018.

- [DOI](https://doi.org/10.1109/TMI.2018.2799231)
- [arXiv](https://arxiv.org/abs/1707.06474)
- [代码](https://github.com/adler-j/learned_primal_dual)
- 基线角色：每层显式 `A/A^T` 的 learned iterative solver。
- 边界：显式 operator calls 不自动保证 convergence 或 unseen-operator generalization。

Aniket Pramanik et al., **Ada-MoDL**, Magnetic Resonance in Medicine, 2023；Aviad Aberdam et al., **Ada-LISTA**, TPAMI, 2022.

- [Ada-MoDL](https://doi.org/10.1002/mrm.29750)
- [Ada-LISTA](https://doi.org/10.1109/TPAMI.2021.3125041)
- 基线角色：采集条件或变化字典调节 learned solver。
- 边界：把 rig、角度或 operator 作为条件输入本身不是创新。

### 2.3 warm-start 最近先例

Rajiv Sambharya et al., **Learning to Warm-Start Fixed-Point Optimization Algorithms**, JMLR, 2024.

- [全文](https://www.jmlr.org/papers/v25/23-1174.html)
- [代码](https://github.com/stellatogrp/l2ws)
- 要提取：solution/residual loss、固定点后处理、PAC-Bayes 审计。
- 边界：理论不能直接搬到病态 BOST/CGLS。

Mohammad Sadegh Eshaghi et al., **NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers**, 2026.

- [DOI](https://doi.org/10.1016/j.cma.2026.118989)
- [arXiv](https://arxiv.org/abs/2511.02481)
- [代码](https://github.com/eshaghi-ms/NOWS)
- 要提取：FNO/DeepONet initial guess、CG/GMRES、online time 与 offline amortization。
- 对 LGWO 的作用：最直接的 neural-operator warm-start 对手。
- 边界：NOWS 是正向 PDE 多查询，不是有限角 BOST inverse，也没有 unseen-rig/null-space fail-closed 结论。

Mo Zhou et al., **A neural network warm-start approach for the inverse acoustic obstacle scattering problem**, JCP, 2023.

- [DOI](https://doi.org/10.1016/j.jcp.2023.112341)
- [arXiv](https://arxiv.org/abs/2212.08736)
- 基线角色：inverse measurement -> CNN initial parameters -> Gauss-Newton refinement。
- 边界：inverse warm-start 也不是空白。

## 3. 第三圈：null-space、有限角与安全性

Johannes Schwab et al., **Deep Null Space Learning for Inverse Problems**, Inverse Problems, 2019.

- [DOI](https://doi.org/10.1088/1361-6420/aaf14a)
- [arXiv](https://arxiv.org/abs/1806.06137)
- 要提取：`I + P_ker N_theta`、正则化组合与收敛条件。
- 边界：exact pseudo-inverse/null projector 在大型变 rig BOST 中不便宜。

Simon Goppel et al., **Data-proximal null-space networks for inverse problems**, 2023.

- [arXiv](https://arxiv.org/abs/2309.06573)
- 要提取：null correction 外的 bounded range-space correction 与 limited-view CT。
- 对 LGWO 的作用：最接近“允许有限可观测偏差但封顶”的理论结构。

Tatiana A. Bubba et al., **Learning the Invisible**, Inverse Problems, 2019.

- [DOI](https://doi.org/10.1088/1361-6420/ab10ca)
- [arXiv](https://arxiv.org/abs/1811.04602)
- 要提取：limited-angle CT 的 visible/invisible wavefront 与 shearlet split。
- 边界：CT microlocal visibility 不能直接套到 refractive-gradient ray operator。

Sebastian Lunz et al., **On Learned Operator Correction in Inverse Problems**, SIAM JIS, 2021.

- [DOI](https://doi.org/10.1137/20M1338460)
- [arXiv](https://arxiv.org/abs/2005.07069)
- 要提取：forward-only correction 的问题与 forward-adjoint correction。
- 对 LGWO 的作用：若学习 curved-ray/calibration mismatch，必须同时审计伴随。

Jaemin Oh et al., **Spectrally Safe Neural Operator Warm-Starts for Large-Scale Newton Solvers**, 2026.

- [arXiv](https://arxiv.org/abs/2606.21828)
- 要提取：平均 relative-L2 很小仍会违反局部物理条件，以及 label-free energy repair。
- 对 LGWO 的作用：支持“field-L2 不足，必须 residual/tail/fallback”。
- 边界：Newton/hyperelasticity 的谱结论不直接证明 CGLS-BOST 安全。

Romario Gualdron-Hurtado et al., **GSNR: Graph Smooth Null-Space Representation for Inverse Problems**, CVPR 2026.

- [CVF 全文](https://openaccess.thecvf.com/content/CVPR2026/papers/Gualdron-Hurtado_GSNR_Graph_Smooth_Null-Space_Representation_for_Inverse_Problems_CVPR_2026_paper.pdf)
- 要提取：graph-smooth low-dimensional null modes、predictability 与计算成本。
- 对 LGWO 的作用：为 fixed-rig amortized near-null basis 提供最近先例。
- 边界：graph null basis 和低维 null representation 已有正式先例；变 rig 的 EVD/projection 成本仍未解决。

## 4. 四周阅读与复现顺序

### 第 1 周：BOST 问题本身

1. Grauer 2018：写出经典 BOST forward、TV/Tikhonov、真实实验误差源；
2. NeRIF：逐式整理 ray integration 与 gradient loss；
3. NeDF 与 Neural Primitives：比较 continuous representation、encoding、sampling、curved ray；
4. TDBOST：拿到全文后单独做 4D 张量分解卡片。

输出：一张“相同测量、不同 representation/optimizer/data-consistency”的对比表。

### 第 2 周：强神经基线

1. DeepONet/FNO/F-FNO：只实现小型固定网格基线；
2. MoDL/LPD：统计每个 inference 的 `A/A^T`；
3. NIO/Geo-FNO/GINO：只提取 observation-set 与 geometry encoding。

输出：统一接口 `predict(y, geometry, operator_budget) -> field, ledger`。

### 第 3 周：warm-start 与公平成本

1. L2WS：复制 solution loss 与 fixed-point residual loss；
2. NOWS：复制 online/offline amortization 表；
3. inverse scattering warm-start：比较“直接场”与“低维参数初值”。

输出：cold CGLS、simple damping、interpolated previous frame、FNO/DeepONet warm、LGWO-A24 的同成本表。

### 第 4 周：不可观测性与安全门

1. Deep Null Space 与 data-proximal null networks；
2. Learning the Invisible；
3. learned operator correction；
4. spectrally safe warm starts 与 GSNR。

输出：full/null-erased/row-erased 因果消融和 fail-closed gate 预注册。

## 5. 最稳妥的论文定位

当前不要写：

> 我们提出首个 neural operator warm-start CGLS。

可检验的定位是：

> 我们研究一种面向有限角三维 BOST 的、可变射线几何条件化且可安全回退的低成本初始方向算子，并在严格端到端算子调用预算下检验其未见 rig 加速与重建能力。

发表门必须同时比较 cold CGLS、简单阻尼/前帧插值、FNO/DeepONet/NIO、MoDL/LPD、NOWS-style warm start，以及 NeRIF/NeDF/Neural Primitives 的逐场景方案；报告 field、held-out-ray reprojection、逐 rig P90/P95、harm、fallback、端到端时间和显存。
