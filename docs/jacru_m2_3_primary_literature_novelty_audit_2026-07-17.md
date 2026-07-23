# JACRU M2.3 一手文献与创新重叠审计

- 审计日期：2026-07-17
- 审计对象：geometry-conditioned、matrix-free、measurement-space projection，用于 BOST/三维折射率重建中的学习残差校正
- 检索边界：出版社页面、会议官方页面、作者主页、PubMed/PMC、arXiv、DOI 元数据；未下载、未保存或提交任何受限 PDF
- 结论强度：本文只报告本轮有界检索能够支持的判断；“未检出 BOST 中完全相同的系统”不等于“全球首次”或“不存在先例”

## 0. 执行结论

### 0.1 直接回答

**问题 1：geometry-conditioned matrix-free measurement-space projection 是否已有直接同类？**

分三层回答：

1. **数学与数值线性代数层：有，而且是直接同类。** 目标映射

   \[
   x_{\mathrm{out}}
   =u-A_G^\top(A_GA_G^\top+\lambda I)^{-1}(A_Gu-y_*)
   \]

   在 \(\lambda=0\) 且采用 Moore--Penrose 逆时，就是把任意候选 \(u\) 正交投影到仿射集合 \(\{x:A_Gx=y_*\}\)。Kammerer--Nashed（1972）已经给出 CG 对奇异算子方程收敛到“伪逆解 + 初值的零空间分量”的结论；Paige--Saunders 的 CRAIG/CGNE 路线正是在 measurement space 解 \(A A^\top z=b\)，再令 \(x=A^\top z\)，且只要求实现 \(Av\) 与 \(A^\top w\)。因此，“measurement-space”“matrix-free”“保留 null-space correction”“CG/PCG”均不能单独作为新概念。

2. **学习重建层：有高度直接的同类。** Deep Null Space Learning 已把学习校正限制在零空间以保持数据一致性；MoDL 已把 CNN prior 与 CG data-consistency block 交替；Bacca（2025，预印本）更直接写出 \(A^\dagger y+(I-A^\dagger A)f(y)\) 的网络后投影。学习预条件器并嵌入 PCG、用 neural operator 作为 CG/FCG 预条件器，也分别已有 MRI 与 PDE 一手先例。

3. **BOST 完整系统层：本轮未检出完全相同的已发表系统。** 在本轮核验的一手来源中，尚未发现一篇同时覆盖：有限孔径 BOST、跨样本可变相机几何、共享学习残差 prior、以 \(A_G/A_G^\top\) 实现的 measurement-space Krylov 投影、由几何显式条件化的预条件器、独立 forward renderer 审计，以及 OOD fail-closed gate。这个“未检出”只能支持一个**窄而可检验的系统/实证增量候选**，不能支持“首个投影”“首个物理一致网络”或“首个 geometry-conditioned inverse method”。

**问题 2：可辩护增量在哪里？**

最稳妥的增量不是投影公式，而是：

> 在有限孔径、可变相机几何的 BOST 上，系统化实现并审计一个 matrix-free 仿射数据一致性校正；在相同 \(A/A^\top\) 调用、显存和墙钟预算下，检验几何条件化且满足 Krylov 合法性要求的预条件器，是否能跨几何提高有限步收敛，并用独立 renderer、no-harm/OOD gate 和真实或留出观测证明其边界。

这仍然首先是**面向 BOST 的算法实例化、计算权衡与可靠性证据**。只有实验显示跨几何、有限预算、独立审计下的稳定增益后，才可进一步主张一个新的经验性结果。若只是把已有 MoDL/投影公式从 image-space 改写到 measurement-space，或把相机参数喂给网络，则不足以构成方法创新。

**问题 3：哪些说法会被审稿人判为旧概念或不准确？**

- “首次使用 CG/PCG 强制数据一致性”：旧概念；MoDL、CGNE/CRAIG 和大量迭代重建已覆盖。
- “首次把网络输出投影回测量可行集”：旧概念；null-space networks 与 projection-based correction 直接覆盖。
- “首次 matrix-free 求解 \(AA^\top\)”：旧概念；这是经典 CGNE/CRAIG 表述。
- “首次学习预条件器”或“首次 neural operator + CG”：旧概念；Koolstra--Remis 与 FCG-NO 直接否定。
- “首次处理变化几何/变化传感器”：过宽；VIDON、GINO 及模型驱动成像中的 scan-specific operator 已覆盖相邻概念。
- “首次用神经算子求逆问题”：旧概念；Neural Inverse Operators 已直接覆盖 PDE inverse problems。
- “首次神经网络三维折射率/BOST 重建”：旧概念；NeDF、NeRIF、Neural Refractive Index Primitives 已覆盖。
- “\(\lambda>0\) 或固定 \(k\) 步仍是 exact projection/exact data consistency”：数学上不准确。
- “保持 \(A x_{\rm ref}\) 等于符合真实测量 \(y\)”：除非已证明 \(A x_{\rm ref}=y\)，否则不成立。
- “独立 renderer 通过即证明物理正确”：不成立；它最多证明跨实现/跨近似 forward consistency，不能替代真实观测与模型误差分析。

## 1. 被审计的 M2.3 映射

设基线重建为 \(x_{\rm ref}\)，学习模块输出校正 \(\delta_\theta\)，候选为

\[
u=x_{\rm ref}+\delta_\theta,
\qquad y_*=A_Gx_{\rm ref}.
\]

M2.3 的阻尼 measurement-space 校正可写为

\[
(A_GA_G^\top+\lambda I)z=A_G\delta_\theta,
\qquad
x_{\rm out}=x_{\rm ref}+\delta_\theta-A_G^\top z.
\]

当 \(\lambda=0\)、右端相容且线性系统求到收敛时，

\[
x_{\rm out}
=x_{\rm ref}+(I-A_G^\dagger A_G)\delta_\theta,
\]

即仅保留学习校正的零空间分量。若 \(A_GA_G^\top\) 奇异，应使用伪逆语义，或明确求解的是相容系统的最小范数解。

### 1.1 与 MoDL 型 data-consistency 的代数同一性

考虑标准 proximal/data-consistency 子问题

\[
\underset{x}{\arg\min}\;
\frac12\lVert A_Gx-y_*\rVert_2^2
+\frac{\lambda}{2}\lVert x-u\rVert_2^2.
\]

其解为

\[
x=(A_G^\top A_G+\lambda I)^{-1}(A_G^\top y_*+\lambda u)
=u-A_G^\top(A_GA_G^\top+\lambda I)^{-1}(A_Gu-y_*).
\]

令 \(u=x_{\rm ref}+\delta_\theta\)、\(y_*=A_Gx_{\rm ref}\)，就得到 M2.3 公式。故 image-space normal equation 与 measurement-space 公式是同一个 proximal map 的两种实现。M2.3 可以比较两者在 \(m\ll n\)、稀疏有限孔径 operator、显存与有限步 Krylov 误差上的工程差异，但不能把代数重写本身当作创新。

### 1.2 术语与正确性边界

- \(\lambda=0\) 且求解收敛：可称“仿射正交投影”或“保留基线预测的 exact null-space correction”。
- \(\lambda>0\)：应称“阻尼 data-consistency/proximal correction”，不是 exact projection。
- 固定 \(k\) 步 PCG/CGNE：应称“inexact/finite-budget Krylov correction”，除非报告最终残差并达到事先定义的容差。
- 直接在 \(A_GA_G^\top\) 上做 CG 是 CGNE/CRAIG 家族语义。若使用“LSQR”一词，必须明确 LSQR 所作用的矩形算子和目标问题；不要把所有只调用 \(A/A^\top\) 的 Krylov 法统称为 LSQR。
- 普通 PCG 要求线性、对称正定的预条件器（或在相应内积下满足条件）。若预条件器是随残差变化的非线性网络，应使用 flexible CG/相应理论，或给出保证 SPD 的参数化与失效回退。

## 2. 一手文献逐篇核验

核验标签：

- **已核验-A**：出版社/会议官方元数据与摘要、DOI 或作者官方页面相互印证。
- **已核验-B**：arXiv 题录和正文/摘要已核验，但本轮未确认同行评审版本；只能作为预印本引用。
- **待核验**：本轮没有足够一手证据确认关键题录或关键结论。此类来源不承担创新排他性结论。

### A. 仿射投影、CGNE/CRAIG 与 inverse-problem preconditioning

#### 1. *On the Convergence of the Conjugate Gradient Method for Singular Linear Operator Equations*

- 作者/年份：William J. Kammerer, M. Zuhair Nashed，1972。
- 稳定链接：[SIAM 出版社与 DOI](https://doi.org/10.1137/0709016)。
- 核验状态：**已核验-A**；出版社元数据和摘要已核验，受限全文未下载。
- 相关性：摘要明确给出 CG 对奇异线性算子方程收敛到 \(T^\dagger y+(I-P)x_0\)，其中 \(I-P\) 投影到 \(N(T)\)。这是“伪逆可辨识分量 + 保留初值零空间分量”的经典理论来源。
- 能引用：exact null-space-preserving affine limit 并非新结构；奇异/欠定系统中的初值零空间分量保留已有经典 CG 理论。
- 不能推出：不能据此声称固定步数、阻尼、学习预条件或 BOST 几何下必然稳定，也不能替代对离散 \(A_G\) 的秩与容差检查。

#### 2. *LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares*

- 作者/年份：Christopher C. Paige, Michael A. Saunders，1982。
- 稳定链接：[DOI](https://doi.org/10.1145/355984.355989)、[作者官方 LSQR 页面](https://web.stanford.edu/group/SOL/software/lsqr/)、[作者官方 CRAIG 页面](https://web.stanford.edu/group/SOL/software/craig/)。
- 核验状态：**已核验-A**；ACM DOI 题录及 Stanford SOL 官方算法页面已核验。
- 相关性：LSQR 只需 routines for \(Av\) and \(A^\top u\)，支持欠定最小范数与阻尼问题；CRAIG 页面更直接说明其等价于在 \(AA^\top y=b\) 上做 CG，再令 \(x=A^\top y\)。
- 能引用：matrix-free \(A/A^\top\) 接口、measurement-space \(AA^\top\) Krylov 求解和最小范数恢复均是成熟数值线性代数。
- 不能推出：LSQR/CRAIG 本身不提供 geometry-conditioned learning、BOST forward fidelity 或 OOD 保证；也不能把 CRAIG 与任意自定义固定步网络层无条件等同。

#### 3. *Conjugate-Gradient Preconditioning Methods for Shift-Variant PET Image Reconstruction*

- 作者/年份：Jeffrey A. Fessler, Scott D. Booth，1999。
- 稳定链接：[DOI](https://doi.org/10.1109/83.760336)、[作者官方题录](https://web.eecs.umich.edu/~fessler/papers/bibbase.php)。
- 核验状态：**已核验-A**；IEEE 题录由作者官方页面与 DOI 印证。
- 相关性：针对 tomography 中由非均匀噪声和正则化引起的 shift-variant Hessian，构造比对角/循环预条件更贴合系统结构的 CG 预条件器。
- 能引用：为扫描/系统结构设计 tomography preconditioner 是旧概念；只做 ray-coverage/Jacobi/几何启发式缩放通常属于经典预条件范畴。
- 不能推出：该文没有学习预条件器、measurement-space BOST 投影或跨相机几何泛化，因此不直接否定一个经过严格定义和验证的 BOST 学习型预条件器。

#### 4. *Preconditioning CGNE Iteration for Inverse Problems*

- 作者/年份：Herbert Egger，2007。
- 稳定链接：[Wiley 出版社与 DOI](https://doi.org/10.1002/nla.522)。
- 核验状态：**已核验-A**；出版社题录与摘要已核验，受限全文未下载。
- 相关性：直接研究 inverse problems 中 CGNE 的预条件、discrepancy regularization、收敛与迭代复杂度。
- 能引用：在 inverse problems 的 normal/measurement-space Krylov 求解中加入 preconditioning 和 stopping rule 是已有理论路线。
- 不能推出：不能据此断言任何神经预条件器保持正则化性质；M2.3 仍需验证预条件器合法性、噪声行为和 stopping criterion。

### B. Null-space networks、网络后投影与 data consistency

#### 5. *Deep Null Space Learning for Inverse Problems: Convergence Analysis and Rates*

- 作者/年份：Johannes Schwab, Stephan Antholzer, Markus Haltmeier，2019。
- 稳定链接：[IOP DOI](https://doi.org/10.1088/1361-6420/aaf14a)、[arXiv 作者稿](https://arxiv.org/abs/1806.06137)。
- 核验状态：**已核验-A**；DOI/期刊题录与 arXiv 作者稿已核验，IOP 页面本轮受 robots 限制。
- 相关性：两阶段“经典重建 + 学习校正”，并通过 null-space network 结构保持数据一致性；与 \((I-A^\dagger A)\delta\) 直接同类。
- 能引用：把学习增量限制在 forward operator 的零空间、从而不改变可辨识数据分量，是已有理论化方法。
- 不能推出：原文不覆盖有限孔径 BOST、跨几何共享模型、measurement-space PCG 实现或独立 renderer gate。

#### 6. *Deep Decomposition Learning for Inverse Imaging Problems*

- 作者/年份：Dongdong Chen, Mike E. Davies，2020。
- 稳定链接：[Springer DOI](https://doi.org/10.1007/978-3-030-58604-1_31)、[arXiv 作者稿](https://arxiv.org/abs/1911.11028)。
- 核验状态：**已核验-A**；ECCV/Springer 题录与 arXiv 作者稿已核验。
- 相关性：显式学习 inverse problem 的 range-space/null-space decomposition，并讨论两部分的不同信息角色。
- 能引用：range/null decomposition 作为 learned inverse imaging 的架构原则不是新概念。
- 不能推出：不能据此说明其分解等价于 M2.3 的有限步 measurement-space PCG，也不提供 BOST 几何泛化证据。

#### 7. *Projection-Based Correction for Enhancing Deep Inverse Networks*

- 作者/年份：Jorge Bacca，2025。
- 稳定链接：[arXiv](https://arxiv.org/abs/2505.15777)。
- 核验状态：**已核验-B**；arXiv 题录与方法公式已核验，本轮未确认同行评审版本或 DOI。
- 相关性：在任意 learned inverse 输出后求 \(\min_x\lVert x-f(y)\rVert^2\) s.t. \(Ax=y\)，并给出 \(A^\dagger y+(I-A^\dagger A)f(y)\)。这是 M2.3 exact affine projection 的最直接近期碰撞。
- 能引用：网络后仿射投影及其 range/null-space 闭式分解至少已有公开预印本；M2.3 不应声称首次提出该映射。
- 不能推出：预印本状态不等于同行评审定论；其摘要不能证明大规模 BOST 上的 matrix-free 可行性、速度或鲁棒性。

#### 8. *Data-Proximal Null-Space Networks for Inverse Problems*

- 作者/年份：Simon Göppel, Jürgen Frikel, Markus Haltmeier，2023。
- 稳定链接：[arXiv](https://arxiv.org/abs/2309.06573)。
- 核验状态：**已核验-B**；arXiv 版本已核验，本轮未确认期刊/会议 DOI。
- 相关性：把 null-space learning 扩展为 data-proximal 结构，在有限观测/模型条件下讨论一致性与收敛，并以 limited-view CT 为例。
- 能引用：从“硬零空间”放宽到 data-proximal/近似一致性已有明确研究，因此 \(\lambda>0\) 的软一致性也不是天然新点。
- 不能推出：不能把其 CT 结果外推到 BOST，也不能据预印本排除 M2.3 在几何自适应有限步求解上的增量。

#### 9. *Model-Based Deep Learning Architecture for Inverse Problems* (MoDL)

- 作者/年份：Hemant K. Aggarwal, Merry P. Mani, Mathews Jacob，2019。
- 稳定链接：[IEEE DOI](https://doi.org/10.1109/TMI.2018.2865356)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/30106719/)、[作者代码](https://github.com/hkaggarwal/modl)。
- 核验状态：**已核验-A**；IEEE/PubMed 题录、摘要和作者代码仓库已核验。
- 相关性：交替使用 CNN denoiser 与 CG numerical data-consistency block，支持一般 forward structure，并共享迭代权重。其 image-space 子问题经 Woodbury 恒等式就是第 1.1 节的 measurement-space 形式。
- 能引用：learned prior + CG data consistency 是直接旧概念；measurement-space 改写本身不产生方法新颖性。
- 不能推出：MoDL 没有给出有限孔径 BOST、可变 camera-set encoding、独立 renderer 审计或 measurement-space 相对 image-space 的规模收益。

#### 10. *Learned Primal-Dual Reconstruction*

- 作者/年份：Jonas Adler, Ozan Öktem，2018。
- 稳定链接：[IEEE DOI](https://doi.org/10.1109/TMI.2018.2799231)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/29870362/)、[arXiv 作者稿](https://arxiv.org/abs/1707.06474)。
- 核验状态：**已核验-A**。
- 相关性：把 forward operator 与 adjoint/backprojection 嵌入 unrolled primal-dual learned reconstruction，可处理 nonlinear forward operator。
- 能引用：反复调用物理 forward/adjoint 的 learned iterative reconstruction 是成熟路线。
- 不能推出：其 learned primal-dual update 不等同于仿射正交投影，也没有证明 exact consistency 或 BOST 跨几何泛化。

#### 11. *CNN-Based Projected Gradient Descent for Consistent CT Image Reconstruction*

- 作者/年份：Harshit Gupta, Kyong Hwan Jin, Ha Q. Nguyen, Michael T. McCann, Michael Unser，2018。
- 稳定链接：[IEEE DOI](https://doi.org/10.1109/TMI.2018.2832656)、[作者机构页面](https://bigwww.epfl.ch/publications/gupta1802.html)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/29870372/)。
- 核验状态：**已核验-A**。
- 相关性：将 CNN 与 projected/recursive gradient reconstruction 结合，用 measurement consistency 约束 CT 重建。
- 能引用：在 tomography 中把网络 prior 与显式 measurement-consistency iteration 结合不是新概念。
- 不能推出：其“projection”语义和实现不必然是 M2.3 的 \(AA^\top\) 仿射投影，也未处理 BOST forward mismatch。

#### 12. *Deep-Learning Projector for Optical Diffraction Tomography*

- 作者/年份：Fangshu Yang, Thanh-An Pham, Harshit Gupta, Michael Unser, Jianwen Ma，2020。
- 稳定链接：[Optica DOI](https://doi.org/10.1364/OE.381413)、[作者机构页面](https://bigwww.epfl.ch/publications/yang2001.html)。
- 核验状态：**已核验-A**；Optics Express/作者机构题录已核验。
- 相关性：在三维 refractive-index tomography 中使用 learned projector 与 measurement-consistency gradient，并比较不同 forward models；是与 BOST 邻近的光学折射率重建先例。
- 能引用：神经 prior/learned projector + 光学三维折射率重建 + measurement consistency 的宽泛组合已经出现。
- 不能推出：optical diffraction tomography 的波动方程、采样与 BOST deflection operator 不同；不能据此判定 M2.3 的有限孔径 BOST 实证增量已被直接发表。

#### 13. *Learned Operator Correction*

- 作者/年份：Sebastian Lunz, Andreas Hauptmann, Tanja Tarvainen, Carola-Bibiane Schönlieb, Simon Arridge，2021。
- 稳定链接：[SIAM 出版社与 DOI](https://doi.org/10.1137/20M1338460)。
- 核验状态：**已核验-A**；SIAM 开放页面、题录与摘要已核验。
- 相关性：在 data/solution spaces 学习近似 forward/adjoint 的校正，以降低模型误差，并在 limited-view photoacoustic tomography 验证。
- 能引用：用学习模块补偿 forward/adjoint mismatch 也是已有路线；M2.3 必须区分“投影求解器加速”与“operator correction”。
- 不能推出：学习校正过的 operator 不自动满足真实物理，也不等同于对固定 \(A_G\) 做 null-space projection。

### C. 学习预条件器、neural operators 与变化几何

#### 14. *Learning a Preconditioner to Accelerate Compressed Sensing Reconstructions in MRI*

- 作者/年份：Kirsten Koolstra, Rob Remis；online 2021，卷期 2022。
- 稳定链接：[Wiley DOI](https://doi.org/10.1002/mrm.29073)、[PMC 开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9299023/)。
- 核验状态：**已核验-A**。
- 相关性：训练 CNN 表示预条件器对向量的作用，并嵌入 PCG；训练输入包含采样 mask、coil sensitivity maps 和 regularization maps，同一模型测试不同采样、线圈与解剖。
- 能引用：学习一个依赖测量配置的 matrix-free preconditioner，并将其用于 PCG 加速 inverse reconstruction，已经有非常直接的一手先例。
- 不能推出：MRI 配置不是 BOST camera geometry；该文也明确不能保证所有 learned preconditioner 都使 CG 收敛。M2.3 若要有增量，必须同时比较经典预条件器、证明/约束 Krylov 合法性并报告真实墙钟收益。

#### 15. *Neural Operators Meet Conjugate Gradients: The FCG-NO Method for Efficient PDE Solving*

- 作者/年份：Alexander Rudikov, Vladimir Fanaskov, Ekaterina Muravleva, Yuri M. Laevsky, Ivan Oseledets，2024。
- 稳定链接：[ICML/PMLR 官方页面](https://proceedings.mlr.press/v235/rudikov24a.html)。
- 核验状态：**已核验-A**。
- 相关性：用 neural operator 作为 nonlinear preconditioner，嵌入 flexible conjugate gradient，并强调跨网格/分辨率复用。
- 能引用：“neural operator as a Krylov preconditioner”已经发表；非线性预条件器应进入 FCG 而不是未经说明的普通 PCG。
- 不能推出：该文面向 PDE forward solve，不是 BOST inverse projection，也没有 variable-camera measurement-space 结果。

#### 16. *Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators*

- 作者/年份：Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, George Em Karniadakis，2021。
- 稳定链接：[Nature 出版社与 DOI](https://doi.org/10.1038/s42256-021-00302-5)。
- 核验状态：**已核验-A**。
- 相关性：以 branch/trunk 网络学习 function-to-function operator，是后续 operator learning 的基础架构之一。
- 能引用：用神经网络跨实例学习算子映射不是新概念。
- 不能推出：DeepONet 原文不提供 inverse-problem data consistency、可变相机集合、BOST forward model 或 Krylov preconditioning。

#### 17. *Fourier Neural Operator for Parametric Partial Differential Equations*

- 作者/年份：Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar，2021。
- 稳定链接：[ICLR/OpenReview 官方页面](https://openreview.net/forum?id=c8P9NQVtmnO)、[arXiv](https://arxiv.org/abs/2010.08895)。
- 核验状态：**已核验-A**。
- 相关性：用 Fourier layers 学习参数化 PDE solution operators，并展示跨分辨率性质。
- 能引用：FNO、resolution transfer 和 function-space operator learning 均是成熟概念。
- 不能推出：FNO 的 forward PDE benchmark 不证明 inverse BOST、变化相机几何或 measurement-space consistency。

#### 18. *Variable-Input Deep Operator Networks*

- 作者/年份：Michael Prasthofer, Tim De Ryck, Siddhartha Mishra，2022。
- 稳定链接：[arXiv](https://arxiv.org/abs/2205.11404)。
- 核验状态：**已核验-B**；arXiv 题录、摘要与理论主张已核验，本轮未确认同行评审 DOI。
- 相关性：允许每个样本的传感器数量与位置变化，并对传感器排列置换不变。
- 能引用：可变数量/位置的 sensor set 与 permutation-invariant operator learning 已有明确先例；“支持任意 camera count/order”不能宽泛声称首次。
- 不能推出：VIDON 没有 BOST projector、有限孔径光学、measurement-space PCG 或几何条件化预条件器。

#### 19. *Neural Inverse Operators for Solving PDE Inverse Problems*

- 作者/年份：Roberto Molinaro, Yunan Yang, Björn Engquist, Siddhartha Mishra，2023。
- 稳定链接：[ICML/PMLR 官方页面](https://proceedings.mlr.press/v202/molinaro23a.html)。
- 核验状态：**已核验-A**。
- 相关性：组合 DeepONet/FNO 学习从观测算子到未知函数的 inverse operator，并在 PDE inverse problems 上验证。
- 能引用：神经算子直接学习 inverse-problem 映射已经是已发表概念。
- 不能推出：NIO 不保证对每个观测严格数据一致，也没有 BOST、null-space projection 或 PCG 预条件机制。

#### 20. *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*

- 作者/年份：Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, Anima Anandkumar，2023。
- 稳定链接：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)。
- 核验状态：**已核验-A**。
- 相关性：以 SDF/point cloud 表示变化几何，结合 graph 与 Fourier neural operators 学习不同三维几何上的 PDE solution operator。
- 能引用：geometry-informed/geometry-conditioned neural operator 的宽泛表述已被直接使用；几何编码本身不是 M2.3 的创新。
- 不能推出：GINO 的“geometry”是 PDE 域/物体形状，不是 BOST camera acquisition geometry；其 forward surrogate 也不是 measurement-space preconditioner。

### D. BOST 与三维折射率重建

#### 21. *Instantaneous 3D Flame Imaging by Background-Oriented Schlieren Tomography*

- 作者/年份：Samuel J. Grauer, Andreas Unterberger, Andreas Rittler, Kyle J. Daun, Andreas M. Kempf, Khadijeh Mohri，2018。
- 稳定链接：[Elsevier 出版社与 DOI](https://doi.org/10.1016/j.combustflame.2018.06.022)。
- 核验状态：**已核验-A**；出版社题录与摘要已核验，受限全文未下载。
- 相关性：从多视角 BOS deflections 以 Tikhonov/TV tomography 重建三维 refractive-index field，并在 23-camera flame experiment 验证。
- 能引用：多相机 BOST 的三维折射率线性/正则化反演是成熟任务与基线。
- 不能推出：该文没有 learned residual prior、跨几何模型或 network-output affine projection。

#### 22. *Forward and Inverse Modeling of Depth-of-Field Effects in Background-Oriented Schlieren*

- 作者/年份：Joseph P. Molnar, Elijah J. LaLonde, Christopher S. Combs, Olivier Léon, David Donjat, Samuel J. Grauer，2024。
- 稳定链接：[AIAA 出版社与 DOI](https://doi.org/10.2514/1.J064095)。
- 核验状态：**已核验-A**；出版社题录与摘要已核验，受限全文未下载。
- 相关性：直接处理 BOST 的 finite-aperture/depth-of-field forward 与 inverse modeling，并用 cone-ray model 和神经场/正则约束重建。
- 能引用：有限孔径 BOST forward physics 与相应 inverse modeling 已被直接研究，不能作为 M2.3 的单独首创点。
- 不能推出：该文未提供跨 acquisition geometry 的共享 residual prior、measurement-space PCG 投影或几何学习预条件器。

#### 23. *Neural Deflection Field for Sparse-View Tomographic Background Oriented Schlieren*

- 作者/年份：Jiawei Li, Xuhui Meng, Yuan Xiong, Tianqi Jia, Chong Pan, Jinjun Wang，2024。
- 稳定链接：[AIP 出版社与 DOI](https://doi.org/10.1063/5.0241191)、[arXiv 作者稿](https://arxiv.org/abs/2409.19971)。
- 核验状态：**已核验-A**；Physics of Fluids 出版社题录、DOI 与 arXiv 作者稿已核验。
- 相关性：以 lightweight neural field 和 ray tracing 处理 sparse-view BOST 折射率重建。
- 能引用：神经隐式表示用于 sparse-view BOST 已有直接先例。
- 不能推出：该方法以每场优化为主，不等同于跨实例 operator、null-space projection 或 matrix-free PCG data-consistency layer。

#### 24. *Neural Refractive Index Field: Unlocking the Potential of Background-Oriented Schlieren Tomography in Volumetric Flow Visualization*

- 作者/年份：Yuanzhe He, Yutao Zheng, Shijie Xu, Chang Liu, Di Peng, Yingzheng Liu, Weiwei Cai，2025。
- 稳定链接：[AIP 出版社与 DOI](https://doi.org/10.1063/5.0250899)、[arXiv 作者稿](https://arxiv.org/abs/2409.14722)。
- 核验状态：**已核验-A**；出版社题录与 arXiv 作者稿已核验。
- 相关性：用 neural implicit refractive-index field 从 BOS 观测重建三维场，并在仿真和真实 Bunsen flame 上验证。
- 能引用：BOST 的 neural refractive-index representation 与真实火焰验证已存在。
- 不能推出：NeRIF 不等同于 amortized variable-geometry reconstruction，也没有 M2.3 的 affine projector、Krylov cost audit 或 fail-closed gate。

#### 25. *Neural Refractive Index Primitives for Flame Field Reconstruction Using Background-Oriented Schlieren*

- 作者/年份：Xinyi Lu, Wei Hu, Zizhou Liao, Zheng Wang, Yue Zhang, Jingxuan Li，2026。
- 稳定链接：[Elsevier 出版社与 DOI](https://doi.org/10.1016/j.combustflame.2026.115082)、[arXiv 作者稿](https://arxiv.org/abs/2605.11454)。
- 核验状态：**已核验-A**；出版社 2026 题录、摘要、DOI 与 arXiv 作者稿已核验。
- 相关性：以紧凑 MLP、encoding、gradient loss、3D mask 和 ray sampling 改进 BOST neural implicit reconstruction，覆盖有限孔径仿真与真实火焰。
- 能引用：到 2026 年，BOST neural implicit reconstruction 的表示、mask、ray sampling 与 finite-aperture 评估已经相当具体。
- 不能推出：该文没有证明 post-reconstruction measurement-space projection、跨几何共享预条件器或独立 forward audit 已被覆盖。

## 3. 重叠判决矩阵

| M2.3 拟主张 | 判决 | 最接近一手先例 | 审稿口径 |
|---|---|---|---|
| 把候选解投影到 \(Ax=y_*\) | **直接旧概念** | Kammerer--Nashed；Deep Null Space；Bacca | 不得声称首次 |
| 保留学习校正的 null-space 分量 | **直接旧概念** | Schwab et al.; Chen--Davies | 只能作为采用的结构原则 |
| 在 \(AA^\top\) 上 matrix-free CG | **直接旧概念** | Paige--Saunders CRAIG/CGNE | 应使用标准术语 |
| learned prior + CG data consistency | **直接旧概念** | MoDL | measurement-space 改写不是新算法 |
| tomography 中 physics iteration + CNN | **旧概念** | Gupta et al.; Yang et al. | BOST 特定证据仍可新增 |
| inverse problem 的 preconditioned CGNE | **旧概念** | Egger；Fessler--Booth | 新意须在特定结构与证据 |
| 学习预条件器并嵌入 PCG | **直接旧概念** | Koolstra--Remis | 必须比较 classical/learned baselines |
| neural operator 作为 CG 预条件器 | **直接旧概念** | FCG-NO | nonlinear preconditioner 应用 FCG |
| neural operator 学 inverse map | **旧概念** | NIO；DeepONet/FNO | 不能称首个 inverse neural operator |
| 可变传感器数量/位置 | **旧概念** | VIDON | camera-set encoding 可作实现，不是首创 |
| geometry-informed neural operator | **旧概念** | GINO | 必须限定 acquisition geometry 与任务 |
| neural BOST/三维折射率重建 | **直接旧概念** | NeDF；NeRIF；NRIP | 不能称首个 neural BOST |
| finite-aperture BOST modeling | **直接旧概念** | Molnar et al. | 可作为必须纳入的 physics baseline |
| 上述全部与独立审计/OOD gate 的 BOST 系统组合 | **本轮未检出完全同类** | 各分量分别已有 | 只可称窄系统/实证缺口，禁止全球首创 |

## 4. 可辩护的贡献边界

### 4.1 现在就可写的保守表述

中文：

> 本工作不提出新的仿射投影或 Krylov 算法，而是在有限孔径 BOST 的可变相机几何下，对一种 matrix-free measurement-space data-consistency realization 进行受控实例化与审计。核心问题是：在相同 forward/adjoint 调用、墙钟和显存预算下，显式依赖 acquisition geometry 的合法预条件器能否改善有限步收敛，并在独立 forward 审计和 OOD/no-harm gate 下保持收益。

英文：

> We do not claim a new affine projection or Krylov solver. We instantiate and audit a matrix-free measurement-space realization of data consistency for finite-aperture BOST with varying camera geometries. Under matched forward/adjoint-call, wall-clock, and memory budgets, we test whether an acquisition-geometry-conditioned, Krylov-compatible preconditioner improves finite-iteration convergence while passing independent-forward and OOD/no-harm gates.

### 4.2 只有结果通过后才可写的增量

以下每条都必须由跨几何留出实验支撑：

- “在固定 \(k\) 和相同 \(A/A^\top\) 调用预算下，geometry-conditioned preconditioner 比无预条件、Jacobi/ray-density、fixed low-rank 与非几何 learned preconditioner 更快降低 measurement-space residual。”
- “相对于等价的 image-space MoDL/DC solve，measurement-space realization 在本 BOST 的 \(m/n\) 比和稀疏 operator 下获得可复现的显存或墙钟收益，同时达到相同容差。”
- “同一组权重跨 camera count、view angle、missing view 与 finite-aperture 参数泛化，并优于 pooled CNN 或固定几何模型。”
- “独立 renderer 与真实/留出观测上的残差同步改善，而不是只对训练 forward \(A_G\) 过拟合。”
- “fail-closed gate 在 OOD/forward mismatch 时可靠回退到 \(x_{\rm ref}\)，因此不劣于基线的覆盖率与代价可量化。”

### 4.3 不足以构成创新的实现变化

- 仅把 \((A^\top A+\lambda I)^{-1}\) 用 Woodbury 改写为 \((AA^\top+\lambda I)^{-1}\)。
- 仅用 sparse operators 避免显式组装矩阵。
- 仅把 camera parameters concatenate 到 CNN/MLP 输入。
- 仅使用 Jacobi、ray count、sensitivity normalization 或 diagonal coverage map。
- 仅把固定 \(k\) 步 CG 放进计算图，但没有与收敛容差、operator-call、墙钟基线比较。
- 只报告 train-forward residual，不报告 independent renderer、held-out geometry 与真实观测。
- 只与无物理约束 CNN 比较，而不与 MoDL/DC、null-space projection、CGNE/CRAIG 及经典预条件器比较。

## 5. 审稿人最可能追问的五个问题

1. **与 MoDL 的数学区别是什么？** 若回答只剩“我们在 measurement space 解”，必须提供 \(m\ll n\) 下的复杂度、内存、数值精度和墙钟证据；否则只是等价实现。
2. **与 Deep Null Space Learning/Bacca 投影的区别是什么？** 应回答 BOST 的有限孔径 operator、跨几何共享与 matrix-free 可扩展性，而不是重复 range/null-space 公式。
3. **“geometry-conditioned”究竟条件化了什么？** 必须区分：scan-specific \(A_G\)（模型驱动方法本来就有）、显式 geometry encoder、以及真正依 \(G\) 生成/调制的预条件器。只要 \(A_G\) 变化就称 geometry-conditioned，会被认为偷换概念。
4. **预条件器是否允许普通 PCG？** 需要证明/构造 linear SPD，或改用 FCG 并报告 residual monitoring、divergence detection 与回退。
5. **投影一致的是谁？** 若 \(y_*=A_Gx_{\rm ref}\)，方法保持的是 baseline-equivalent prediction，而不必是 raw-data consistency。应同时报告 \(\lVert A_Gx_{\rm ref}-y\rVert\)、\(\lVert A_Gx_{\rm out}-y\rVert\) 和 independent-forward residual。

## 6. 最低限度对照与证据门槛

### 6.1 必需对照

1. `x_ref`（no correction）。
2. `x_ref + delta_theta`（无投影 learned residual）。
3. pooled CNN/residual network + **同一个** projector（隔离 geometry encoder 的作用）。
4. exact small-grid SVD/pseudoinverse oracle（只作上界，不作可部署基线）。
5. unpreconditioned CGNE/CRAIG。
6. Jacobi/ray-density 或 sensitivity diagonal preconditioner。
7. fixed low-rank/circulant/结构型 preconditioner，按 operator 性质选择。
8. non-geometry learned preconditioner 与 geometry-conditioned learned preconditioner。
9. image-space MoDL-style CG data consistency，与 measurement-space 实现达到相同容差。
10. CGLS/Huber/TV 等传统 BOST reconstruction，匹配 forward/adjoint 调用或墙钟预算。

### 6.2 必报指标

- \(A/A^\top\) 调用次数、预条件器调用次数、墙钟、峰值显存；不能只报 iteration count。
- inner residual 对迭代的曲线及最终容差；报告 breakdown/divergence 率。
- raw measurement residual、baseline-equivalent residual、independent-renderer residual 三者分开。
- 3D field error、gradient/deflection error、结构边界误差；真实数据无 GT 时不得伪装成全量精度结论。
- 按 camera count、angle gap、missing view、aperture、噪声和 forward mismatch 分层的 OOD 结果。
- no-harm gate 的接受率、误接受率、回退率和回退后总体指标。
- 相同 weights 跨 geometry 的证据；每个 geometry 重新训练不能支持 geometry-generalization 主张。

### 6.3 必需数值正确性检查

- adjoint test：\(\langle A_Gx,v\rangle\approx\langle x,A_G^\top v\rangle\)。
- 对小系统显式核对 \(AA^\top\) solve、SVD 投影与 matrix-free 结果。
- \(\lambda=0\) 时验证 \(A_G(x_{\rm out}-x_{\rm ref})\approx0\)；\(\lambda>0\) 时不要使用 exact 字样。
- 检查 rank deficiency、相容右端和 preconditioner SPD/FCG 条件。
- 把训练 renderer、projection operator 与 independent renderer 的代码路径、参数和标定来源分开。

## 7. 建议的论文定位

**不建议的标题方向：**

- “A Novel Geometry-Conditioned Projection Network for Inverse Problems”
- “First Exact Data-Consistent Neural Reconstruction for BOST”
- “A New Matrix-Free PCG Layer”

这些标题把旧的投影、Krylov 或宽泛 geometry conditioning 误写成方法首创。

**较可防守的标题方向：**

- “Finite-Budget Measurement-Space Data Consistency for Variable-Geometry BOST: A Controlled Evaluation”
- “Auditing Geometry-Conditioned Krylov Preconditioning in Finite-Aperture BOST Reconstruction”
- “Cross-Geometry BOST Residual Reconstruction with Matrix-Free Consistency and Fail-Closed Validation”

最终最可信的故事是：**已有算法成分的 BOST 特定组合是否在严格成本匹配、跨几何与独立 forward 审计下真正有效**。若结果为阴性，仍可形成有价值的负结果：exact-null headroom 小、learned prior 与 pooled CNN 接近、或 geometry-conditioned preconditioner 的墙钟收益不足，都能清楚限定该方向，而不应通过放大新颖性来掩盖。

## 8. 待核验项与检索限制

- 本轮没有检出与 M2.3 全部组件完全一致的 BOST 论文；该结论是有界检索的 negative evidence，不是系统综述意义上的不存在证明。
- Bacca（2025）、Data-Proximal Null-Space Networks（2023）、VIDON（2022）在本报告中按 arXiv 预印本处理；投稿前应再次检查是否已有正式版本、题名变更或 DOI。
- 对受限出版社页面仅核验公开元数据/摘要/DOI，没有下载受限 PDF；报告中没有依赖无法公开复核的全文细节。
- 投稿前应以最终方法关键词追加一次向前/向后引文追踪，重点查：`affine projection deep inverse network`、`range-nullspace correction`、`measurement-space data consistency`、`learned preconditioner tomography`、`variable geometry BOST reconstruction`、`camera-conditioned Krylov`。
- 若 M2.3 后续改变为 nonlinear forward、learned forward correction、随机/非线性预条件器或原始测量 \(y\) 投影，应重新审计；本报告的代数同一性以当前线性化 \(A_G/A_G^\top\) 设计为边界。
