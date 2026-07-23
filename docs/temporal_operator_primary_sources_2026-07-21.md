# OERF 4D-BOST、动态逆问题与 nuisance profiling 一级文献路线

> 用途：为“算子学习 + 三维/四维重构 + 几何不确定度”提供可核对的主干文献。只列正式出版页、作者公开稿或官方项目，不收录受限 PDF 副本。
>
> 阅读原则：每篇都分成“它支持什么”和“它不能替我们证明什么”，避免把邻域方法当作 OERF 真实证据。

## A. 与何远哲师兄最直接的三篇

### A1. NeRIF：神经隐式折射率场

**He et al. (2025), Neural refractive index field: Unlocking the potential of background-oriented schlieren tomography in volumetric flow visualization. Physics of Fluids 37, 017143.**

- [AIP 正式页](https://doi.org/10.1063/5.0250899)
- [arXiv 作者公开稿](https://arxiv.org/abs/2409.14722)
- 提取：voxel 表示的空间分辨率、离散误差、噪声和计算成本问题；NeRIF 的坐标网络表示、损失、数值与 Bunsen flame 实验。
- 对本项目：把网络视为一个受限 field tangent；必须计算它是否真的缩小 nuisance tangent，并检查独立场误差。
- 不能证明：joint camera/field identifiability、data-only `q_cal`、多帧可观测性或跨 rig 泛化。

### A2. TDBOST：四维张量分解重构

**He et al. (2026), Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction. ACM Transactions on Graphics.**

- [ACM DOI](https://doi.org/10.1145/3809488)
- 提取：时空张量表示、4D 重构工作流、rank/time-window、速度和保真度评价。
- 对本项目：这是必须复现或接口化的直接基线；重点询问 birth/extinction、拓扑断裂、缺帧和同步误差下的失败模式。
- 不能证明：低秩张量对所有反应流成立，或低秩本身解除 field/geometry gauge。

### A3. PIV-BOST：下游物理终点

**Zheng et al. (2025), Instantaneous refractive index compensation on the velocity measurement using simultaneous PIV-BOST. Experiments in Fluids 66, 164.**

- [Springer 正式页](https://doi.org/10.1007/s00348-025-04093-y)
- 提取：九视角 BOST、PIV 同步、折射误差补偿，以及下游 velocity error 的评价合同。
- 对本项目：如果 field-L2 不对应真实用途，PIV velocity compensation 可成为更有物理意义的 endpoint。
- 不能证明：盲相机标定、神经算子或自由场条件下的几何可辨识。

## B. OERF 相邻的物理约束路线

### B1. Tomo-BOS + PINN 推断速度/压力

**Cai et al. (2021), Flow over an espresso cup: Inferring 3-D velocity and pressure fields from tomographic background oriented Schlieren via physics-informed neural networks. JFM 915, A102.**

- [Cambridge DOI](https://doi.org/10.1017/jfm.2021.135)
- [arXiv](https://arxiv.org/abs/2103.02807)
- 提取：把已测三维温度场与 Navier-Stokes/传热方程结合，推断速度和压力，并用独立 PIV 平面验证。
- 不能证明：PDE residual 会自动修复错误相机几何；错误物理先验仍可能制造 profile curvature。

## C. 动态 tomography：低秩与 transport 到底提供了什么

### C1. 联合重构与低秩分解

**Arridge, Fernsel & Hauptmann (2022), Joint reconstruction and low-rank decomposition for dynamic inverse problems. Inverse Problems and Imaging 16, 483-523.**

- [期刊全文页](https://doi.org/10.3934/ipi.2021059)
- 提取：动态 CT 与空间/时间低秩因子的联合估计、非负矩阵分解和分离正则。
- 不能证明：BOST 场一定低秩，或未知双因子低秩带来几何信息。我们的切空间反例正好说明后者一般不成立。

### C2. 物理运动模型的动态 X-ray tomography

**Burger et al. (2017), A Variational Reconstruction Method for Undersampled Dynamic X-ray Tomography based on Physical Motion Models. Inverse Problems 33, 124008.**

- [IOP DOI](https://doi.org/10.1088/1361-6420/aa99cf)
- [arXiv](https://arxiv.org/abs/1705.06079)
- 提取：稀疏时序投影下联合估计图像与 optical-flow-like motion。
- 不能证明：反应流满足纯输运。火焰新生、熄灭、膨胀、扩散和热释放必须作为 innovation stress。

### C3. Low-rank + sparse

**Otazo, Candes & Sodickson (2015), Low-rank and Sparse Matrix Decomposition for Accelerated Dynamic MRI with Separation of Background and Dynamic Components. Magnetic Resonance in Medicine 73, 1125-1136.**

- [DOI](https://doi.org/10.1002/mrm.25240)
- [作者稿 / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4207853/)
- 提取：低秩背景与稀疏动态成分的分离、incoherence 假设。
- 不能证明：火焰/射流可干净分成低秩背景与稀疏事件；transport 会让 support 移动，反应会改变拓扑。

### C4. PSM + RED denoising

**Iskender, Klasky & Bresler (2023), RED-PSM: Regularization by Denoising of Partially Separable Models for Dynamic Imaging. ICCV 2023.**

- [CVF 官方论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Iskender_RED-PSM_Regularization_by_Denoising_of_Partially_Separable_Models_for_Dynamic_ICCV2023_paper.html)
- 提取：partially separable low-rank model、RED、ADMM 与 stationary-point convergence。
- 不能证明：stationary point 等于正确恢复，也没有覆盖 BOST 相机 nuisance。

### C5. 可复现动态 tomography 工具

**Papoutsellis et al. (2021), Core Imaging Library - Part II: Multichannel Reconstruction for Dynamic and Spectral Tomography. Philosophical Transactions A.**

- [DOI](https://doi.org/10.1098/rsta.2020.0193)
- [arXiv](https://arxiv.org/abs/2102.06126)
- [CIL 官方项目](https://www.ccpi.ac.uk/cil/)
- 用途：学习规范的动态/spectral tomography baseline、数据结构与重建算子；不是 BOST `q_cal` 证明。

## D. nuisance profiling 与自标定

### D1. Variable projection 原点

**Golub & Pereyra (1973), The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate. SIAM Journal on Numerical Analysis 10, 413-432.**

- [SIAM DOI](https://doi.org/10.1137/0710036)
- 提取：消去线性 nuisance、对剩余非线性参数优化，以及 rank regularity 条件。
- 对本项目：`S=C^T(I-P_B)C` 的基本思想来源；工具本身不保证 `S` 非零。

### D2. 逆问题中的 nuisance 参数

**Aravkin & van Leeuwen (2012), Estimating Nuisance Parameters in Inverse Problems. Inverse Problems 28, 115016.**

- [IOP DOI](https://doi.org/10.1088/0266-5611/28/11/115016)
- [arXiv](https://arxiv.org/abs/1206.6532)
- 提取：ML/MAP 下 profiling、variance、robust model、automatic calibration 与 experimental design。
- 不能混淆：MAP/regularized curvature 是 prior-conditioned，不是 data-only Fisher information。

### D3. Differentiable uncalibrated imaging

**Gupta et al. (2023), Differentiable Uncalibrated Imaging.**

- [arXiv](https://arxiv.org/abs/2211.10525)
- 提取：通过 differentiable forward 联合优化 measurement coordinates 与场，包含 2D/3D CT 实验。
- 不能证明：宽 BOST voxel 场与 pose 可辨；需要正面对比我们的自由场 `0/5` 反例。

### D4. 有严格假设的 self-calibration

**Ling & Strohmer (2015), Self-Calibration and Biconvex Compressive Sensing. Inverse Problems 31, 115002.**

- [IOP DOI](https://doi.org/10.1088/0266-5611/31/11/115002)
- [arXiv](https://arxiv.org/abs/1501.06864)
- 提取：未知对角 gain 与 sparse signal 的特定模型下，SparseLift 的精确/稳健恢复条件。
- 不能迁移：BOST pose、ray geometry、detector shift 与 curved-ray mismatch 不是未知对角 gain，折射率场也不自动稀疏。

## E. 建议阅读顺序

1. 先读 D1、D2：学会为什么要 profile nuisance，以及 MAP 曲率为什么不是纯数据信息。
2. 再读 A1、A2：把 NeRIF/TDBOST 的网络或张量表示翻译成 field tangent，而不是只看网络结构图。
3. 读 C1、C2、C3：比较未知低秩、transport 与 innovation 的假设强度。
4. 读 A3、B1：决定论文真正的物理 endpoint 是 field、PIV velocity、front 还是 held-out reprojection。
5. 最后读 D3、D4：寻找联合标定的算法形式，同时逐条检查其可辨识假设是否在 BOST 中成立。

## F. 从文献到本机实验的映射

| 文献概念 | 本机对照 | 当前判决 |
|---|---|---|
| 未知低秩因子 | `unknown_low_rank_both_factors` | `0/5`，结构 NO-GO |
| 固定 spatial prior | pilot/teacher PCA | 曲率强但模型偏差大 |
| physical transport | `transport_exact` | `5/5`，但 reference SNR 下 practical NO-GO |
| L+S / innovation | shared birth / free innovation | shared birth 削弱信息；free innovation 回到 `0/5` |
| variable projection | local q estimator | 数学链路正确，reference SNR 不足 |
| uncertainty calibration | NIS + plug-in radius | reference 下全部拒答，消除旧 false accept |

最值得继续的研究句子不是“设计一个更大的 FNO”，而是：**学习一个能够压缩真实 transport/innovation tangent 的算子提议器，并由真实 BOST forward、held-out view/time 与校准不确定度严格验证；当 nuisance model 不可信时必须拒绝几何更新。**
