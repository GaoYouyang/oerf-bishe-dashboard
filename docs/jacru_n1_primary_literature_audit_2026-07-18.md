# JACRU N1：噪声/失配感知数据一致性的一手文献审计

- 审计日期：2026-07-18
- 对应证据起点：JACRU M2.8 `M2_8_INTERPOLATION_CALIBRATION_ENVELOPE_NO_GO`
- 检索范围：Morozov discrepancy principle、白化残差与协方差逆问题、迭代早停正则化、Huber/Student-t 稳健数据项、BOS/BOST 测量噪声与模型失配
- 来源限制：只使用原论文的出版社页面、DOI、MathNet、arXiv/作者页、作者所在机构库、PMC/OSTI 等一级或官方存档；不使用博客或聚合摘要承担技术结论
- 版权边界：本轮没有下载、保存或提交任何受限 PDF；本文只保存题录、官方链接和基于摘要/开放正文的审计结论
- 结论边界：本文是 N1 实验设计依据，不是当前算法成功、真实 BOST 泛化或方法优越性的证据

## 0. 执行摘要

### 0.1 M2.8 真正留下的问题

M2.7 已在打开的 12³ synthetic T0 上证明：使用不可部署的 exact camera-block 预条件器时，measurement-space PCG 在总调用预算内可以足够快地接近 `Ax=y`。M2.8 又证明：固定全局插值没有通过点；即使 evaluator 读取真值，为每个样本选择最优连续插值，K=10 仍只有 `97.22%` 的 development rows 存在同时满足逐样本重投影门与场误差要求的 alpha。固定受害样本 `single_interface / base_seed 2113` 在两个模型家族、六个模型种子上持续受害。

因此，当前最小可证伪假设不是“PCG 还不够快”，而是：

> 当观测含有随机位移误差、camera-correlated noise、标定偏差或 forward-model mismatch 时，把神经重建强行投到 `Ax=y` 会把观测中不可由真实三维场解释的分量写回欠定解；应把停止点和数据项改成噪声/失配感知形式，并对尾部样本实行 fail-closed 判决。

本轮文献支持这是假设合理的研究方向，但**不证明**它就是 M2.8 受害样本的真实因果机制。因果判断仍需 N1 的分解压力实验与真实 flow-off 数据。

### 0.2 五条可以直接落地的结论

1. **Morozov 原则需要可信的误差上界。** 经典形式是在残差首次降到与已知数据误差同量级时停止，而不是把残差压到零。噪声尺度未知、协方差估错或存在确定性模型偏差时，经典保证不能直接搬用。
2. **CG/PCGLS 的迭代编号本身就是正则化参数。** 早停可以抑制半收敛，但“训练集上最优 K”或“真值 oracle K”都不是可部署准则；统计噪声下朴素残差规则甚至可能次优或失效。
3. **白化必须对应真实协方差。** 若 `C` 是观测误差协方差，正确二次数据项是 `r^T C^{-1} r`。只做逐像素标准差缩放无法处理相机内相关、共享背景、光流窗口重叠和低频标定漂移。
4. **Huber 与 Student-t 只针对特定污染机制。** Huber 限制大残差影响但需尺度/阈值；Student-t 可更强地下调离群点，但数据项非凸、自由度与尺度敏感。二者都不能自动修复错误的 forward operator，也可能把真实火焰前沿的大残差当作离群点。
5. **BOS/BOST 误差是分层的。** 图像位移估计、梯度诱导模糊、积分离散、有限孔径/景深、相机标定和 refractive-ray 模型失配具有不同空间结构。N1 必须按 camera/block 和误差来源拆分，而不能只加一个全局 Gaussian `sigma`。

### 0.3 对当前 N1 的严格授权

文献与 M2.8 联合起来，只授权以下工作：

- 在**打开的 synthetic development** 上比较固定、非学习的 discrepancy-stopped CGLS/PCGLS、Tikhonov、Huber 和 Student-t ceiling；
- 构造 i.i.d.、异方差、camera-correlated、稀疏离群、标定偏差和 forward mismatch 六类可分解 stress；
- 使用 generator-known covariance 作为 evaluator oracle，同时测试可估计 covariance 与错误 covariance；
- 报告 field relative-L2、H1、逐 rig/形态尾部、重投影、白化校准、`A/A^T` 调用、setup 和端到端成本；
- 只有固定经典方法先出现联合可行区，才允许学习 stopping/scale/operator。

当前**不授权**：打开 fresh/final、声称 N1 已成功、把 exact covariance/exact camera block 当作部署方法、声称优于 DeepONet/FNO/NeRIF、或把 synthetic mean gain 写成真实 BOST 结论。

## 1. 数学合同：N1 到底要停止在哪里

设实际观测为

\[
y=A_*x^\dagger+d_{\rm model}+b_{\rm cam}+\varepsilon,
\qquad \varepsilon\sim(0,C_\varepsilon),
\]

而重建代码使用的算子是 `A`。其中：

- `A_*`：真实光学测量过程；
- `A`：薄光线、有限孔径近似或离散 renderer；
- `d_model=(A_*-A)x^dagger`：结构化模型失配；
- `b_cam`：相机标定、背景配准或低频系统偏差；
- `epsilon`：随机图像/位移误差。

若 `C_epsilon=L L^T`，白化残差定义为

\[
r_w(x)=L^{-1}(Ax-y),
\qquad \Phi_C(x)=\lVert r_w(x)\rVert_2^2
=(Ax-y)^T C_\varepsilon^{-1}(Ax-y).
\]

经典 Morozov 语义是：已知数据误差界 `delta` 时，选择正则化参数或最早迭代 `k_delta`，使残差首次进入 `tau*delta` 邻域，其中 `tau>1` 留出安全裕度。N1 不应把这改写成“残差越小越好”。

### 1.1 为什么 `||r_w||^2 approximately m` 不能无条件使用

只有在以下理想条件同时成立时，真值处

\[
C_\varepsilon^{-1/2}(A x^\dagger-y)\sim\mathcal N(0,I_m),
\]

从而其平方范数服从 `chi-square_m`、期望为测量维数 `m`：

1. `A=A_*`，没有模型失配；
2. 偏差已移除；
3. `C_epsilon` 已知且可逆；
4. 噪声确为零均值 Gaussian；
5. 这里评估的是真值，而不是已经用同一观测拟合出的估计量。

对由同一 `y` 拟合的线性预测 `y_hat_k=H_k y`，残差是 `(I-H_k)y`。即使模型正确，其方差和有效自由度也被 `H_k` 改变；若有正则化偏差，平方白化残差还是非中心二次型。故 N1 的第一版应同时报告：

- 原始 `chi-square_m` 参考线，仅标记为 ideal reference；
- 由独立 flow-off repeats 或 held-out synthetic noise 得到的经验分位数；
- 每相机/每 block 的 normalized innovation squared；
- 有效秩、condition number、白化后自相关与跨相机相关；
- 错误 covariance 下的校准漂移。

不得用一个接近 `m` 的 aggregate 数字直接证明模型正确或重建准确。

### 1.2 模型误差不是“再加一点 Gaussian noise”

若 `d_model` 或 `b_cam` 非零，即便 `epsilon` 被完美白化，

\[
r_w=C_\varepsilon^{-1/2}(A x-y)
\]

仍含结构化均值。把它继续压到 Gaussian noise ball，可能迫使 `x` 吸收错误算子的残差。N1 必须把随机 covariance 与模型失配 stress 分开；可把后者作为 bounded bias、camera low-rank bias 或 alternative renderer discrepancy 分支，但不能悄悄并入 `C_epsilon` 后宣称“统计校准完成”。

## 2. 一手文献逐篇审计

核验标签：

- **A：出版社/官方数据库已核验。** DOI、题录和摘要或开放正文由出版社、MathNet、PMC/OSTI 等官方来源确认。
- **B：作者稿/机构库已核验。** 同时给出 DOI 或正式题录；正文来自作者/机构公开存档。
- **P：同行评审前预印本。** 只可作为预印本引用，不能当作期刊定论。

### A. Morozov discrepancy principle 与误差尺度

#### A1. Morozov (1966), *Regularization of Incorrectly Posed Problems and the Choice of Regularization Parameter*

- 官方链接：[MathNet 题录](https://www.mathnet.ru/eng/zvmmf7494)；[DOI](https://doi.org/10.1016/0041-5553%2866%2990046-2)
- 题录：V. A. Morozov, *USSR Computational Mathematics and Mathematical Physics*, 6(1), 242–251 (1966)。
- 核验：**A**。
- 真正支持：正则化参数不应任意取到极限，而应与观测误差水平联系；这是 discrepancy-based parameter choice 的原始根系之一。
- 不能支持：不能推出当前 BOST 噪声是 Gaussian、协方差已知、`delta` 可由训练 residual 估计，也不能说明 M2.8 的受害样本一定源于过拟合噪声。
- N1 作用：把 `delta/tau` 明确写入预注册；`delta` 必须来自独立 noise contract，不能由待评分场的真值或最终误差反推。

#### A2. Morozov (1968), *The Error Principle in the Solution of Operational Equations by the Regularization Method*

- 官方链接：[MathNet 题录](https://www.mathnet.ru/eng/zvmmf7206)；[DOI](https://doi.org/10.1016/0041-5553%2868%2990034-7)
- 题录：V. A. Morozov, *USSR Computational Mathematics and Mathematical Physics*, 8(2), 63–87 (1968)。
- 核验：**A**。
- 真正支持：使用误差/残差原则为 operator equation 的正则化选择参数；残差达到与已知数据误差相称的水平即可停止。
- 不能支持：经典确定性误差界不等于随机白噪声的 `chi-square` 校准；也不覆盖估计 covariance、非线性神经表示、有限孔径 BOST 或 data-dependent PCG preconditioner。
- N1 作用：第一条基线必须是“最早进入阈值”的 stopping rule；禁止从整条迭代路径读取真值后选择最佳 K，再把它称为 Morozov。

#### A3. Mead and Hammerquist (2013), *Chi-Square Tests for the Choice of the Regularization Parameter in Nonlinear Inverse Problems*

- 官方链接：[SIAM DOI](https://doi.org/10.1137/12088447X)；[作者机构题录](https://scholarworks.boisestate.edu/math_facpubs/121/)
- 题录：*SIAM Journal on Matrix Analysis and Applications*, 34(3), 1213–1230 (2013)。
- 核验：**A/B**；出版社题录与作者机构版本已核验。
- 真正支持：在带已知 Gaussian weighting 的离散非线性逆问题中，可以对 regularized residual 建立明确的 `chi-square` 检验；自由度依赖被检验的具体二次泛函，而不是永远等于观测数。
- 不能支持：不能把任意 whitened data residual 机械设为 `chi-square_m`；也不能在 covariance、先验或模型误差未知时保证检验有效。
- N1 作用：将“raw data residual 的 ideal `m` 参考”和“包含正则项/拟合自由度的校准统计量”分开实现；所有自由度公式要由当前 estimator 推导或用独立 Monte Carlo 校准。

### B. CG/迭代早停作为正则化

#### B1. Nemirovskii (1986), *Regularizing Properties of the Conjugate Gradient Method for Ill-Posed Problems*

- 官方链接：[MathNet 题录](https://www.mathnet.ru/eng/zvmmf4030)；[DOI](https://doi.org/10.1016/0041-5553%2886%2990002-9)
- 题录：A. S. Nemirovskii, *USSR Computational Mathematics and Mathematical Physics*, 26(2), 7–16 (1986)。
- 核验：**A**。
- 真正支持：CG 对病态问题具有迭代正则化性质；停止索引参与控制噪声放大，求到线性系统残差最小并不等于重建误差最小。
- 不能支持：不能据此断言任意固定步 PCG、学习预条件器或 measurement-space `AA^T` 迭代都保留相同理论；预条件、内积与 stopping rule 的条件必须重新核对。
- N1 作用：保存完整 `k=0...Kmax` 路径，画 field/H1 与 residual 的半收敛曲线；把 K 当正则化参数而不是只当计算预算。

#### B2. Blanchard and Mathé (2012), *Discrepancy Principle for Statistical Inverse Problems with Application to Conjugate Gradient Iteration*

- 官方链接：[IOP DOI](https://doi.org/10.1088/0266-5611/28/11/115011)；[作者论文页](https://www.imo.universite-paris-saclay.fr/~gilles.blanchard/publi/index.html)
- 题录：*Inverse Problems*, 28(11), 115011 (2012)。
- 核验：**A/B**；DOI 与作者官方题录已核验。
- 真正支持：统计白噪声下朴素 discrepancy principle 可能失效或得到次优速率；论文引入修正规则，并分析包括 CG 在内的正则化方法。
- 不能支持：不能把论文的 trace-class/统计模型结论直接外推到有限孔径 BOST、估计 covariance、camera bias 或非 Gaussian outliers。
- N1 作用：至少加入一个 smoothed/blockwise residual stopping control，不能只测全局 `||r_w||`；记录何种噪声下 plain rule 与 modified rule 分叉。

#### B3. Blanchard, Hoffmann and Reiß (2018), *Optimal Adaptation for Early Stopping in Statistical Inverse Problems*

- 官方链接：[SIAM 出版社](https://epubs.siam.org/doi/10.1137/17M1154096)；[arXiv 作者稿](https://arxiv.org/abs/1606.07702)
- 题录：*SIAM/ASA Journal on Uncertainty Quantification*, 6(3), 1043–1075 (2018)。
- 核验：**A/B**。
- 真正支持：残差型早停可以在一定条件下相对 oracle stopping 取得适应性误差界；prediction error 与 reconstruction error 需要不同的偏差–方差转移分析。
- 不能支持：低重投影误差不自动推出低 field error；其理论也不证明某个 BOST stopping feature 能识别 `single_interface` 尾部样本。
- N1 作用：把 measurement/prediction gate 与 field/H1 gate 分列，禁止只凭重投影改善选择方法；oracle-K 只作 ceiling，不进入部署算法。

#### B4. Hucker and Reiß (2025), *Early Stopping for Conjugate Gradients in Statistical Inverse Problems*

- 官方链接：[Springer 开放正文](https://link.springer.com/article/10.1007/s00211-025-01469-4)；[DOI](https://doi.org/10.1007/s00211-025-01469-4)
- 题录：*Numerische Mathematik*, 157, 1739–1791 (2025)。
- 核验：**A**，开放获取。
- 真正支持：标准 CGNE 的早停本身构成正则化；论文给出 data-driven stopping 与 prediction/reconstruction oracle-type 分析，并明确指出噪声水平估计误差不能成为主导项。
- 不能支持：结果建立在明确的统计 inverse model 上；不能为估错 covariance、deterministic camera bias、operator mismatch 或 learned nonlinear stopping 提供自动保证。
- N1 作用：加入 `delta_hat/delta_true` 敏感性曲线，例如 0.5、0.75、1、1.25、1.5 倍；如果小幅噪声尺度误差就改变尾部符号，N1 必须判 fail-closed。

### C. 白化残差、协方差与模型误差

#### C1. Stuart (2010), *Inverse Problems: A Bayesian Perspective*

- 官方链接：[Cambridge 出版社](https://www.cambridge.org/core/journals/acta-numerica/article/inverse-problems-a-bayesian-perspective/587A3A0D480A1A7C2B1B284BCEDF7E23)；[DOI](https://doi.org/10.1017/S0962492910000061)
- 题录：*Acta Numerica*, 19, 451–559 (2010)。
- 核验：**A**。
- 真正支持：观测噪声模型是 inverse problem 定义的一部分；Gaussian likelihood 的负对数数据项自然包含 noise covariance 的逆，且不确定性建模与 regularization 不可分离。
- 不能支持：Bayesian formulation 本身不证明选定 covariance 正确，也不把 posterior/likelihood consistency 等同于真实 BOST field accuracy。
- N1 作用：在配置文件中显式记录 `noise_mean`、`covariance_model`、估计数据 split、正则化和 likelihood；不得把 whitening 当作无来源的数值 trick。

#### C2. Moorkamp and Avdeeva (2020), *Using Non-Diagonal Data Covariances in Geophysical Inversion*

- 官方链接：[Oxford Academic 开放正文](https://academic.oup.com/gji/article/222/2/1023/5836719)；[DOI](https://doi.org/10.1093/gji/ggaa235)
- 题录：*Geophysical Journal International*, 222(2), 1023–1033 (2020)。
- 核验：**A**。
- 真正支持：广义最小二乘数据项为 `(s(m)-d)^H C_D^{-1}(s(m)-d)`；非对角 covariance 能表达观测分量相关，并可分块实现。忽略变量变换诱导的相关性可能产生偏差。
- 不能支持：地球物理阻抗的 covariance 结构不等于 BOS detector/camera 结构；论文不告诉我们 OERF 应使用哪种 block、rank 或 shrinkage。
- N1 作用：比较 `I`、diagonal、camera-block、low-rank-plus-diagonal 四类 `C_hat`；报告每类 setup、solve、校准和 tail，而不是只报 mean field gain。

#### C3. Stayman et al. (2014), *Generalized Least-Squares CT Reconstruction with Detector Blur and Correlated Noise Models*

- 官方链接：[PMC 开放作者稿](https://pmc.ncbi.nlm.nih.gov/articles/PMC4201055/)；[SPIE DOI](https://doi.org/10.1117/12.2043067)
- 题录：*Proceedings of SPIE*, 9033, 903335 (2014)。
- 核验：**A/B**。
- 真正支持：tomographic detector blur 会产生相关测量；同时建模 blur 与 correlated noise 可改变 noise–resolution trade-off，独立噪声假设并非无害简化。
- 不能支持：X-ray scintillator blur 与 BOS 背景/光流误差的物理机制不同；其 simulation improvement 不能证明 camera-block whitening 会改善 N1。
- N1 作用：把 detector/optical blur 作为 covariance 与 forward blur 两种互斥解释分别测试，避免 double counting；若只改 covariance 无法解释受害样本，转入 model-mismatch 分支。

**自由度复用提醒：** A3 的 Mead and Hammerquist (2013) 还表明，不同 residual functional 对应不同 `chi-square` 自由度；尤其在未知量数不少于观测数时，regularized residual 与 raw residual 不能混用。N1 因此要为每个 stopping statistic 写独立单元测试，使用 Monte Carlo noise-only fixture 检查经验均值、方差与预设分位数；不通过则禁用该统计量。

### D. Huber/Student-t 稳健数据项

#### D1. Huber (1964), *Robust Estimation of a Location Parameter*

- 官方链接：[Project Euclid DOI](https://doi.org/10.1214/aoms/1177703732)
- 题录：P. J. Huber, *The Annals of Mathematical Statistics*, 35(1), 73–101 (1964)。
- 核验：**A**；DOI/期刊题录已核验。
- 真正支持：在 contaminated distribution 下，通过限制大残差的 influence 可以在 Gaussian efficiency 与离群鲁棒性间折中；这是 Huber M-estimation 的理论根源。
- 不能支持：不能证明固定 `delta=1.345 sigma` 适合 BOS，也不能把所有大 residual 都解释为坏点；空间连续的火焰前沿、shock 或 forward mismatch 可能产生真实且成片的大残差。
- N1 作用：Huber scale 必须由独立 noise data 或训练 split 冻结；至少扫描 standardized threshold，并报告按形态/相机的 downweighted fraction 与前沿区域覆盖率。

#### D2. Farquharson and Oldenburg (1998), *Non-linear Inversion Using General Measures of Data Misfit and Model Structure*

- 官方链接：[Wiley 出版社](https://onlinelibrary.wiley.com/doi/10.1046/j.1365-246x.1998.00555.x)；[DOI](https://doi.org/10.1046/j.1365-246x.1998.00555.x)
- 题录：*Geophysical Journal International*, 134(1), 213–227 (1998)。
- 核验：**A**。
- 真正支持：非 `L2` 的 robust misfit 可通过 IRLS 嵌入线性化逆问题；当数据含 outliers 时，稳健数据项可能减少伪结构。
- 不能支持：IRLS 的稳健性不等于对 operator bias 稳健；多层迭代会增加成本，也可能收敛到依赖初始化/尺度的结果。
- N1 作用：提供 Huber-IRLS ceiling，但必须把每轮 `A/A^T` 调用与重加权 setup 纳入同一预算；禁止与 K=9 PCGLS 做不等成本比较。

#### D3. Lange, Little and Taylor (1989), *Robust Statistical Modeling Using the t Distribution*

- 官方链接：[Taylor & Francis 出版社](https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478852)；[DOI](https://doi.org/10.1080/01621459.1989.10478852)
- 题录：*Journal of the American Statistical Association*, 84(408), 881–896 (1989)。
- 核验：**A**。
- 真正支持：Student-t 误差模型为长尾数据提供 Gaussian 的稳健扩展；自由度控制尾重，且可用于线性/非线性回归及协方差估计。
- 不能支持：不能把 t likelihood 当作无参数的万能 robust loss；自由度、尺度和多变量相关结构都需估计或冻结。
- N1 作用：把 `nu` 与 scale 作为预注册有限网格；同时运行 Gaussian/Huber control，并用 held-out standardized residual 检验 tail model，而不是只比较 field mean。

#### D4. Kazantsev et al. (2017), *A Novel Tomographic Reconstruction Method Based on the Robust Student's t Function for Suppressing Data Outliers*

- 官方链接：[IEEE DOI](https://doi.org/10.1109/TCI.2017.2694607)；[CWI 机构题录与开放作者稿](https://ir.cwi.nl/pub/27368)
- 题录：*IEEE Transactions on Computational Imaging*, 3(4), 682–693 (2017)。
- 核验：**A/B**。
- 真正支持：在 tomography 中，defective pixels、miscalibrated sensors、scattering、missing angles 等可造成 gross outliers；Student-t misfit 与 TV regularization 可被直接用于稳健重建，并可估计尺度参数。
- 不能支持：该文的 neutron/CT 数据与 BOST 光流误差不同；Student-t 的成功案例不能证明它在 smooth correlated noise、camera bias 或有限孔径 mismatch 下优于 Huber/PCGLS。
- N1 作用：Student-t 只在 sparse/camera-local outlier stress 中作为主候选；在纯 Gaussian、低频 correlated 和 model-mismatch stress 中作为负对照，检查是否牺牲 clean fidelity。

### E. BOS/BOST 测量噪声、相关性与 forward mismatch

#### E1. Goldhahn and Seume (2007), *The Background Oriented Schlieren Technique: Sensitivity, Accuracy, Resolution and Application to a Three-Dimensional Density Field*

- 官方链接：[Leibniz University Hannover 机构题录](https://research.uni-hannover.de/en/publications/the-background-oriented-schlieren-technique%2876583ff0-c064-4309-9c33-d437fbccc413%29.html)；[DOI](https://doi.org/10.1007/s00348-007-0331-1)
- 题录：*Experiments in Fluids*, 43, 241–249 (2007)。
- 核验：**A/B**。
- 真正支持：BOS sensitivity 依赖焦距、物体/背景相对位置和最小可检位移；resolution 取决于 cross-correlation interrogation window，且不是常数。
- 不能支持：不能据此给 OERF 当前端镜系统指定一个数值 covariance，也不能证明三维 tomography 的全部误差都来自位移检测。
- N1 作用：synthetic noise 应随 camera geometry、位移幅值与 interrogation scale 变化；禁止所有 rays 共用同一 `sigma` 作为唯一主实验。

#### E2. Atcheson, Heidrich and Ihrke (2009), *An Evaluation of Optical Flow Algorithms for Background Oriented Schlieren Imaging*

- 官方链接：[UBC 作者页面与预印本](https://www.cs.ubc.ca/labs/imager/tr/2008/Atcheson_BOS_EiF/)；[DOI](https://doi.org/10.1007/s00348-008-0572-7)
- 题录：*Experiments in Fluids*, 46, 467–476 (2009)。
- 核验：**A/B**。
- 真正支持：BOS 位移质量依赖 optical-flow 算法与背景纹理；multiscale background 和算法选择会显著改变误差。
- 不能支持：不能将某一种 optical flow 的误差分布普遍视作 Gaussian 或 independent；也不证明图像域误差经过 BOST operator 后仍保持同一 covariance。
- N1 作用：若可获得 raw reference/flow-on image，covariance 合同应在位移估计后重新估计；不同 flow estimator 作为 domain shift，而不是只改变重建网络 seed。

#### E3. Rajendran et al. (2020), *Uncertainty Quantification in Density Estimation from Background-Oriented Schlieren Measurements*

- 官方链接：[OSTI 官方题录](https://www.osti.gov/pages/biblio/1598804)；[arXiv 作者稿](https://arxiv.org/abs/1909.06643)；[DOI](https://doi.org/10.1088/1361-6501/ab60c8)
- 题录：*Measurement Science and Technology*, 31(5) (2020)。
- 核验：**A/B**。
- 真正支持：可从 cross-correlation displacement uncertainty 出发，经稀疏 Poisson operator 传播到空间分辨的 density uncertainty；sharp gradients 与边界/积分路径会使不确定度空间变化。
- 不能支持：二维 density integration uncertainty 不等于多相机三维 BOST covariance；其传播方法不能直接校准 NeRIF 或 N1 的 field posterior。
- N1 作用：将 per-pixel displacement uncertainty 传播到 measurement weights，至少比较 uniform 与 spatially varying diagonal whitening；报告界面附近和远离边界区域的校准。

#### E4. Rajendran, Bane and Vlachos (2020), *Uncertainty Amplification Due to Density/Refractive-Index Gradients in Background-Oriented Schlieren Experiments*

- 官方链接：[Springer DOI](https://doi.org/10.1007/s00348-020-02978-8)；[arXiv 作者稿](https://arxiv.org/abs/1910.09379)
- 题录：*Experiments in Fluids*, 61 (2020)。
- 核验：**A/B**。
- 真正支持：refractive-index gradient 的非线性会模糊 dot image，放大图像噪声对 centroid/位移估计的影响；放大量与图像尺寸、强度和光学参数有关，并在实验中呈空间非均匀。
- 不能支持：CRLB 给的是特定估计模型下的下界，不是当前 optical-flow error 的完整 covariance；也不能证明 M2.8 的 single-interface harm 就由该机制引起。
- N1 作用：新增 gradient-amplified heteroscedastic stress，使噪声尺度与真值梯度/渲染 blur 相关；同时设置一个只依赖可观测图像质量的 deployable scale estimator，避免部署时读取 truth gradient。

#### E5. Anand et al. (2023), *Quantifying Numerical Uncertainty in Background-Oriented Schlieren*

- 官方链接：[Springer/PMC 开放正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10543532/)；[DOI](https://doi.org/10.1007/s00348-023-03734-4)
- 题录：*Experiments in Fluids* (2023)。
- 核验：**A**。
- 真正支持：BOS measurement chain 除随机位移误差外还有 density-gradient integration 的 numerical/systematic uncertainty；sharpness、noise level 和 flow wavelength 会影响 bias 与不确定度预测。
- 不能支持：Richardson extrapolation 针对积分离散，不会自动覆盖 tomography angular undersampling、neural representation bias 或 finite-aperture mismatch。
- N1 作用：把 discretization/grid refinement stress 与 stochastic noise 分开；同一 noise realization 在至少两种 voxel/implicit resolution 上复现，检查 stopping rule 是否错误吸收离散偏差。

#### E6. Grauer and Steinberg (2020), *Fast and Robust Volumetric Refractive Index Measurement by Unified Background-Oriented Schlieren Tomography*

- 官方链接：[Penn State 机构题录](https://pure.psu.edu/en/publications/fast-and-robust-volumetric-refractive-index-measurement-by-unifie/)；[DOI](https://doi.org/10.1007/s00348-020-2912-1)
- 题录：*Experiments in Fluids*, 61, 80 (2020)。
- 核验：**A/B**。
- 真正支持：传统 BOST 先解 optical flow 再重建，前一逆问题及其用户参数是潜在误差源；UBOST 直接拟合图像畸变，可减少该模型误差并改变计算成本。
- 不能支持：UBOST 的数值实验不证明所有 staged BOST 都应废弃，也不说明 covariance weighting 能替代更正确的 image-domain forward model。
- N1 作用：设置 `deflection-domain A` 与 `image/unified-domain A*` 的 renderer mismatch stress；若 robust/whitened residual 只能在同算子噪声中有效、遇到跨算子就失败，应明确转向 forward-model correction 而不是继续调 loss。

#### E7. Molnar et al. (2024), *Forward and Inverse Modeling of Depth-of-Field Effects in Background-Oriented Schlieren*

- 官方链接：[AIAA Journal](https://doi.org/10.2514/1.J064095)；[arXiv 作者稿](https://arxiv.org/abs/2402.15954)
- 题录：*AIAA Journal*, online 2024；DOI `10.2514/1.J064095`。
- 核验：**A/B**。
- 真正支持：thin-ray pinhole model 会遗漏 finite-aperture/depth-of-field blur；cone-ray forward model 在模拟和实验 BOS 上可显著改变重建，说明 operator mismatch 能形成结构化而非白噪声残差。
- 不能支持：不能把该文对特定 buoyancy/hypersonic cases 的改进数值外推为 OERF 系统的误差大小；也不能把 exact cone-ray renderer 当作免费 oracle。
- N1 作用：把 thin-ray reconstruction / cone-ray generation 设为预注册 mismatch stress，分别测试 covariance、Huber、Student-t 是否 fail-closed；记录 cone-ray setup 和每次 forward 成本。

#### E8. Grauer et al. (2018), *Instantaneous 3D Flame Imaging by Background-Oriented Schlieren Tomography*

- 官方链接：[Elsevier 出版社](https://www.sciencedirect.com/science/article/pii/S0010218018302694)；[DOI](https://doi.org/10.1016/j.combustflame.2018.06.022)
- 题录：*Combustion and Flame*, 196, 284–299 (2018)。
- 核验：**A**。
- 真正支持：多相机 BOS deflection 可用于三维火焰折射率重建；Tikhonov 与 TV priors 对突变 flame-front 的适配不同，且 synthetic phantom 与真实 23-camera demonstration应分开解释。
- 不能支持：真实火焰可视化不等于有三维真值；重投影或可视化合理也不能单独证明绝对 field accuracy。
- N1 作用：经典 Tikhonov/TV 或 Huber-TV 作为结构先验 control；界面 H1/位置误差必须单列，避免 L2 mean 掩盖 flame-front 损伤。

### F. 与何远哲/OERF 当前主线直接相邻的来源

#### F1. He et al. (2025), *Neural Refractive Index Field: Unlocking the Potential of Background-Oriented Schlieren Tomography in Volumetric Flow Visualization*

- 官方链接：[AIP/Physics of Fluids](https://doi.org/10.1063/5.0250899)；[arXiv 作者稿](https://arxiv.org/abs/2409.14722)
- 题录：Y. He et al., *Physics of Fluids*, 37, 017143 (2025)。
- 核验：**A/B**。
- 真正支持：NeRIF 用隐式神经场替代 voxel representation，目标是缓解空间分辨、离散误差、noise immunity 和计算成本问题，并给出数值与 Bunsen-flame 实验展示。
- 不能支持：摘要中的“improved noise immunity/accuracy”不等于已给出 flow-off covariance、独立三维真值或针对 M2.8 尾部的 no-harm guarantee；也不能证明 N1 应学习一个更大网络。
- N1 作用：N1 应作为 NeRIF/pooled-CNN proposal 后的可观测 data-consistency layer 或固定 classical control，而不是替换其表示主线；评价必须保留 NeRIF 相关的空间分辨与界面指标。

#### F2. Zheng et al. (2025), *Instantaneous Refractive Index Compensation on the Velocity Measurement Using Simultaneous PIV-BOST*

- 官方链接：[Springer 出版社](https://link.springer.com/article/10.1007/s00348-025-04093-y)；[DOI](https://doi.org/10.1007/s00348-025-04093-y)
- 题录：Y. Zheng, Y. He et al., *Experiments in Fluids*, 66, 164 (2025)。
- 核验：**A**；出版社摘要、Appendix re-projection validation 和 data-availability statement已核验。
- 真正支持：OERF 已把九视角 BOST 与 planar PIV 同步用于 turbulent Bunsen flame，并用三维折射率估计补偿瞬时光偏折；论文报告该小火焰下瞬时 velocity error 约 `±2%`。附录使用一视角重投影进行 convergence/consistency 检查。
- 不能支持：一视角重投影一致性和 PIV correction 不等于 N1 covariance 已知或三维折射率真值已验证；出版社页面还明确写明当前论文没有生成/分析可公开数据集，故不能假定我们已拥有训练数据。
- N1 作用：最优真实数据合同是同 geometry 的 flow-off temporal repeats、每相机 raw/reference/flow-on 图、位移场、mask、calibration residual，以及永久留出的 audit view；这些应向师兄明确索取，而不是从论文图反推 covariance。

## 3. 文献对 N1 的直接实验映射

### N1-A：noise/covariance contract gate

**问题：** 不看 reconstruction truth，能否从独立数据估出足以校准残差的 `C_hat`？

**数据分割：**

1. `cov_fit`：只含 flow-off repeats 或独立 noise-only synthetic realizations；
2. `cov_select`：用于选择 diagonal/block/rank/shrinkage，不能参与最终校准；
3. `cov_audit`：只检查白化后均值、方差、自相关、跨相机相关和分位数覆盖；
4. reconstruction development fields：不允许反向更新 covariance；
5. fresh/final：继续冻结关闭。

**至少比较：**

| covariance | 含义 | 必须报告的失败模式 |
|---|---|---|
| `I` | 未白化 control | 异方差/相关噪声下阈值漂移 |
| diagonal | per-pixel variance | 跨像素、跨光流窗口、相机内相关残留 |
| camera-block | 每相机完整或结构化 block | 样本不足、逆矩阵不稳、setup 成本 |
| low-rank + diagonal | 背景/标定共享模式 | rank 选择泄漏、漏掉局部噪声 |
| generator-known exact | evaluator oracle | 明确标记不可部署，不参与方法主张 |

**Gate：** `cov_audit` 中每相机白化均值接近 0、方差/分位数覆盖可接受、跨相机相关不显著恶化；condition number、eigenvalue floor 和有效秩全部公开。若 exact covariance 有效而估计 covariance 无法迁移，只能写“oracle headroom”，不能写 N1 方法成功。

### N1-B：六类最小压力族

| Stress | 生成方式 | 要识别的物理/算法问题 | 不得混为一谈 |
|---|---|---|---|
| G0 i.i.d. Gaussian | 独立零均值同方差 | ideal Morozov/chi-square sanity | 不代表真实 BOS |
| G1 heteroscedastic | variance 随 camera/pixel/可观测图像质量变 | sensitivity 与 gradient amplification | 不等于 correlation |
| G2 camera-correlated | camera block + spatial low-frequency modes | 背景共享、窗口重叠、读出/配准相关 | 不等于 deterministic bias |
| G3 sparse gross outlier | dead pixels、局部错配、camera burst | Huber/Student-t 的目标场景 | 不等于 flame-front signal |
| G4 calibration bias | 固定/缓慢 camera offset、pose/background drift | 非零均值系统误差 | 不能用零均值 covariance 洗掉 |
| G5 operator mismatch | cone-ray generation、thin-ray reconstruction；或 image-domain vs deflection-domain | finite aperture/optical-flow model mismatch | 不能称随机噪声 |

每个 stress 必须覆盖至少 smooth plume、single interface、multi-interface/filament 等形态；受害的 `single_interface / 2113` 保留为 opened diagnostic，不可冒充 fresh。

### N1-C：固定经典方法 Pareto ceiling

在相同 `A/A^T` 调用和总成本账本下比较：

1. matched CGLS/PCGLS fixed-K；
2. plain Morozov stopping；
3. whitened global discrepancy stopping；
4. camera-block/smoothed discrepancy stopping；
5. Tikhonov 固定有限网格；
6. covariance-weighted Tikhonov；
7. Huber data fidelity + 固定 proposal-center penalty；
8. Student-t data fidelity，只在 G3 作为主候选、其余 stress 作 control；
9. truth-oracle K/scale/alpha，仅作为 evaluator ceiling；
10. exact covariance/exact camera-block，仅作为 oracle ceiling。

在固定方法没有联合可行点前，不训练 stopping network、scale network、covariance network 或新的 neural operator。

### N1-D：指标与硬门

**重建质量：**

- field relative-L2；
- H1/gradient relative error；
- interface location/thickness 或 topology-aware error；
- 每 morphology、rig、camera-count、noise family 的 median/p10/worst；
- `>1% harm` 比例和 worst paired degradation。

**观测校准：**

- raw reprojection 与 matched baseline ratio；
- whitened residual total、每 camera、每 block；
- empirical coverage：目标分位数与 `cov_audit` 分位数的差；
- whitened residual autocorrelation、cross-camera correlation；
- robust downweight map 与 flame-front overlap；
- model-mismatch residual 在 alternative renderer/audit view 上的迁移。

**成本：**

- `A`、`A^T`、alternative renderer 调用；
- covariance fit/setup、factorization、每帧 solve；
- Huber/Student-t outer/inner iterations；
- peak memory、wall time、geometry reuse 后的 amortized cost。

**最小硬门建议：** 继续沿用 M2.8 的 no-harm 与逐 rig tail 逻辑；任何 aggregate mean 通过但 `single_interface` 或新 morphology 出现稳定伤害时，判 NO-GO。具体百分比在运行前写入 config，不在看结果后修改。

### N1-E：噪声尺度错误敏感性

对每种可部署 stopping rule 固定测试

```text
delta_hat / delta_true in {0.50, 0.75, 1.00, 1.25, 1.50}
```

并至少加入 covariance misspecification：

```text
exact -> diagonal -> wrong camera block -> identity
```

若只有 `delta_true` 或 exact covariance 一点通过，则判 oracle-only。若 under-estimated noise 导致更深迭代并复现 M2.8 harm，而 over-estimated noise 只回退到 proposal，可形成机制证据；但仍需独立真实 noise contract 才能迁移到 OERF。

### N1-F：fail-closed 输出合同

候选只有在以下可观测量同时位于 `cov_select` 冻结区间内时才接管：

- camera-wise whitened residual concentration；
- correction norm 与 proposal norm 比；
- held-out ray/camera reprojection（若数据合同允许）；
- robust downweight fraction 与空间集中度；
- covariance likelihood/whiteness audit；
- alternative renderer discrepancy（只用于 audit，不得读取真值）。

否则返回 matched classical baseline。coverage=0 虽可实现零伤害，但不构成算法成功；必须同时报告 coverage、conditional gain 和 unconditional gain。

## 4. 论文中可以直接使用的谨慎表述

### 4.1 Introduction/Methods 可用

> Classical discrepancy principles select a regularization level or stopping index by matching the residual to a known data-error scale rather than driving the residual to zero (Morozov, 1966, 1968). For iterative solvers such as conjugate gradients, early stopping itself acts as regularization, while statistically valid residual stopping depends on the noise model and on sufficiently accurate noise-level estimation (Nemirovskii, 1986; Blanchard and Mathé, 2012; Hucker and Reiß, 2025).

> BOS measurement uncertainty is spatially and optically structured: displacement estimation depends on background and image-processing choices, refractive-index gradients can amplify localization uncertainty, numerical integration adds systematic uncertainty, and thin-ray models can miss finite-aperture depth-of-field effects (Atcheson et al., 2009; Rajendran et al., 2020; Anand et al., 2023; Molnar et al., 2024).

> We therefore evaluate covariance-weighted and robust data fidelities as falsifiable controls for a post-reconstruction data-consistency step. These controls are not assumed to correct forward-model error, and all oracle covariance or truth-selected settings are reported separately from deployable candidates.

### 4.2 当前不可用

- “M2.8 证明 BOST 噪声导致过拟合”：错误；M2.8 只给出与该机制相容的 synthetic tail conflict。
- “Morozov 保证 N1 在 BOST 上收敛到真值”：错误；真实误差界、模型正确性和统计条件尚未建立。
- “白化残差约等于测量维数，所以 covariance 正确”：错误；拟合自由度、bias、estimated covariance 和 mismatch 均会改变分布。
- “Huber/Student-t 自动消除模型失配”：错误；稳健 loss 只能降低某些 residual 的影响，不能修正错误 ray geometry。
- “NeRIF 噪声鲁棒，所以 N1 已有真实依据”：错误；NeRIF 的实验展示不是 N1 covariance/stopping 的验证。
- “PIV-BOST 的一视角重投影验证就是独立三维 ground truth”：错误；它是 projection consistency evidence。
- “exact covariance/exact camera block 的增益可作为部署性能”：错误；它们是 oracle ceiling，必须单列 setup 与不可部署边界。

## 5. 给师兄的最小数据问题清单

1. 同一相机、同一 geometry 是否有至少一段 flow-off temporal repeats？能否保留原始图像而不只保留位移场？
2. reference background 是单帧、平均帧还是动态更新？相机间是否共享光源/背景板/触发，可能产生相关误差？
3. 当前 optical-flow/cross-correlation 的窗口、overlap、subpixel estimator 和 uncertainty output 是什么？
4. 是否有 calibration residual、dark/flat frames、camera pose covariance、背景配准误差或稳定自由流区域？
5. 是否可永久留出一台 camera 或一组 ray/pixels，不参与 reconstruction/stopping 参数选择？
6. 当前 forward 是 thin-ray、bent-ray 还是 finite-aperture/cone-ray？`A` 与 `A^T/J^T` 是否可调用，dot test 是否已有？
7. NeRIF/PIV-BOST 当前 validation view 是否参与过模型/超参数选择？有没有完全独立的 cross-modality evidence？
8. 同一 geometry 会复用多少帧？这决定 camera-block covariance/factorization setup 是否能摊销。
9. 师兄认为实验主误差排序是什么：位移噪声、背景漂移、标定、有限孔径、ray bending、composition/Gladstone–Dale，还是时间同步？
10. 哪些原始数据可在组内私有使用、哪些可制作匿名小 fixture、哪些完全不得离开实验室？

## 6. 优先阅读顺序

### 第一轮：先懂 N1 的三个核心概念

1. Morozov 1968：只抓住“残差到误差尺度即停”。
2. Nemirovskii 1986 + Hucker/Reiß 2025：理解 CG 半收敛、早停和噪声尺度误差。
3. Stuart 2010：理解 covariance 为什么进入数据项，而不是后处理权重。
4. Huber 1964 + Kazantsev et al. 2017：区分 robust statistics 与 tomography implementation。

### 第二轮：把统计假设对到 BOS 物理链

5. Atcheson et al. 2009：位移估计不是无误差传感器。
6. Rajendran et al. 2020 两篇：空间异方差与梯度放大。
7. Anand et al. 2023：随机误差之外还有数值/systematic uncertainty。
8. Grauer/Steinberg 2020 + Molnar et al. 2024：某些 residual 来自错误 forward，不应只靠 robust loss 吞掉。

### 第三轮：贴回 OERF/何远哲

9. NeRIF 2025：确认 proposal 表示、实验场景和现有验证边界。
10. PIV-BOST 2025：确认九视角端镜、Bunsen flame、重投影 validation 与可获取数据合同。

## 7. 审计后的研究判决

### 7.1 最值得先做的候选

**首选：camera/block covariance + discrepancy-stopped classical solver。**

原因不是它“最新”，而是它直接测试 M2.8 的核心解释：过拟合含结构噪声的 `y`。它参数少、容易与 CGLS/Tikhonov 做同预算比较，也能在没有真实 field truth 时先做 flow-off calibration audit。

### 7.2 第二候选

**Huber on whitened residual + proposal-center penalty。**

它适合测试局部错配/重尾，但必须先白化再 robust，且用 downweight map 检查是否误伤界面。若只在 sparse outlier stress 有效，这是正确但较窄的结论；不要强行包装成通用 BOST 方法。

### 7.3 只作 ceiling 的候选

**Student-t data fidelity。**

它对 gross outlier 有合理先例，但非凸、参数更多，本科项目风险高。第一轮只作为有限参数网格和 oracle ceiling；若 Huber 已达到同等 Pareto frontier，不继续扩大 Student-t 模型。

### 7.4 必须 fail-closed 的分支

**operator mismatch。**

若 thin-ray reconstruction 对 cone-ray/image-domain generation 的 residual 持续结构化，且 covariance/Huber/Student-t 都只通过牺牲真实界面来减小 loss，应停止 N1 loss escalation，转向 forward-model correction、alternative renderer audit 或 N2 held-out camera gate。

### 7.5 当前总状态

```text
JACRU_N1_PRIMARY_LITERATURE_AUDIT_COMPLETE
N1_EXPERIMENT_AUTHORIZATION = SYNTHETIC_DEVELOPMENT_ONLY
METHOD_SUCCESS = FALSE
REAL_BOST_VALIDATION = FALSE
FRESH_FINAL_OPEN = FALSE
```

本审计把 N1 从“给 residual 加一个阈值/robust loss”收紧为一个可证伪问题：**在独立估计的误差合同下，噪声/失配感知停止是否能同时改善观测校准和三维场尾部，且不依赖真值、exact covariance 或不可摊销 setup？** 在这个问题通过前，不应训练更复杂的 stopping operator，也不应撰写算法成功结论。
