# 面向候选 A/B 的相消感知算子学习文献证据地图

> 版本：2026-07-17  
> 目标：只为当前两条候选路线服务，不再扩散到 OERF 的全部研究方向。  
> 核验规则：每条文献均在线核对出版社、正式会议论文集、arXiv 原文页或作者官方代码仓库；索引站只用于交叉核对元数据，不作为结论来源。本文件不保存、分发或链接任何非授权 PDF。

## 0. 先把候选 A/B 说清楚

### 候选 A：面向证书的几何条件化绝对质量学习

给定真实的、保留符号的 BOST 离散前向算子 $A\in\mathbb{R}^{m\times n}$，定义逐元素绝对值行列质量

\[
r_i^\star=\sum_j |A_{ij}|,\qquad s_j^\star=\sum_i |A_{ij}|.
\]

网络只预测几何到质量的映射

\[
g\longmapsto (\hat r,\hat s),
\]

经正值化后得到候选质量 $(\hat r,\hat s)$。只有在声明所对应的信息合同内，才能进一步称其为上界或证书；随后用获准的 $(\bar r,\bar s)$ 构造

\[
\sigma_i=\frac{\eta}{\bar r_i+\varepsilon},\qquad
\tau_j=\frac{\eta}{\bar s_j+\varepsilon},\qquad 0<\eta<1.
\]

它学习的是**求解器度量/预条件信息**，不是直接学习三维密度场。迭代中仍调用真实的有符号 $A$ 与 $A^\top$。候选 A 的论文命题应是：在新几何、孔径和采样配置上，减少构造安全对角度量的时间或内存，同时保持可审计的稳定边界；重建误差改善只能作为实验结果，不能由安全边界直接推出。

### 候选 A 的 oracle、校准与证书分层

| 层级 | 测试 rig 时允许读取什么 | 能证明/能声称什么 | 不能声称什么 |
|---|---|---|---|
| Offline teacher / exact oracle | 完整精确 $r^\star,s^\star$ sweep | 生成训练标签、测量预测误差、给出 exact-metric 参考轨迹 | 不能声称省掉 exact sweep，也不是 oracle-free 部署 |
| Exact-mass max clipping（oracle 诊断） | 对该测试 rig 计算 $r^\star,s^\star$，再取 $\bar r=\max(\hat r,r^\star)$、$\bar s=\max(\hat s,s^\star)$ | 可检查“在精确质量之上增加学习型阻尼”时的轨迹，且逐元素不低估 | 不能称 learned substitute、成本节省或可部署证书；它已经付过完整 exact sweep |
| Oracle-free predictor | 只读部署可得几何、校准量和冻结参数，不读测试 rig 的 exact mass/field truth | 可评价真实预测误差、成本与 OOD 退化 | 单凭平均误差或零次偶然 violation，不能称安全证书 |
| Safety-calibration envelope | 独立 safety-calibration rigs 上冻结的缩放、分位数或包络；测试 rig 不再调参 | 在预先写明的切分和统计假设下，声称经验覆盖率或概率型风险控制 | 不是逐 rig 的确定性证书；几何 OOD 时覆盖假设可能失效 |
| 真正可部署的 certificate | 部署时可计算、且成本单独记账的确定性上界/验证量 | 只在证明前提、数值精度和输入合同均通过时，声称该次运行满足安全不等式 | 若证书本身遍历全部 $|A|$，可称安全但仍不能称节省 exact sweep |

**刚完成的独立审计边界**：当前 smoke 中逐 held-out rig 的 exact-mass maximum clipping 属于第二行。它在数学上恢复了不低估，但方法实质是“exact metric + 预测过大部分带来的额外阻尼”。因此 Schur 检查全通过只能说明 oracle-clipped 路径没有破坏该安全条件，不能打开“预测器替代 exact sweep”或“可部署”的门。

### 候选 A 的强制低复杂度基线

| 基线 | 定义/信息合同 | 它排除的伪创新 |
|---|---|---|
| Scalar PDHG / scalar step | 不学习行列结构，只用冻结标量步长 | 排除“任何对角化都显得先进” |
| Exact mass | 直接使用 $r^\star,s^\star$，明确标为 oracle | 给出 tight exact-metric 参考，不是部署成本基线 |
| Simple scalar damping | 至少包含 $c\,r^\star,c\,s^\star$ 的 oracle 对照；只有 $c\ge1$ 才是相对 exact 的保守阻尼，$0<c<1$ 会放大步长并可能失去证书；另设仅由 calibration 冻结 $c$ 的 oracle-free 版本 | 检查所谓 learned gain 是否只是全局改变步长 |
| Exact–factor interpolation | $r_\lambda=(1-\lambda)r^\star+\lambda r^{\rm factor}$，$s_\lambda=(1-\lambda)s^\star+\lambda s^{\rm factor}$，固定 $\lambda\in[0,1]$；仅当 factor 逐坐标不小于 exact mass 时该线段保持保守 | 检查 gain 是否只是从 tight exact 向保守 factor 连续增加阻尼；只要 $\lambda<1$ 且读取 exact，就仍是 oracle 诊断 |
| Factor majorizer | 只用已有解析/三角不等式 majorizer | 检查一个更便宜、无需学习的安全替代是否已经足够 |
| Unclipped predictor | 测试时完全不读 exact mass | 暴露真实 underprediction、Schur violation 与 OOD 风险 |
| Frozen calibration envelope | predictor 乘以只在 calibration rigs 冻结的单侧因子/包络 | 检查复杂网络是否胜过简单校准，但不得把经验包络叫确定性证书 |
| Exact-max-clipped predictor | $\max(\hat m,m^\star)$，只列在 oracle 分栏 | 量化“oracle + 额外阻尼”，禁止混入部署主结果 |

上述方法必须同时报告 exact teacher/sweep 次数、部署阶段 exact-audit 次数、metric setup 时间、$A/A^\top$ 调用、Schur violation、field error、residual 和停止规则。若部署主方法的 test-rig exact-audit 次数不为零，就不能把 setup 节省写进摘要。

### 候选 B：真实残差把关的低秩/历史校正

在候选 A 已可靠的基础上，使用当前梯度/残差

\[
g_k=P_S A^\top(Ax_k-b)
\]

及有限历史 $H_k$，生成包含对角项、低秩全局项和历史项的候选方向 $q_k$。任何学习方向必须用真实算子重新检查，例如比较

\[
\|A(x_k+\alpha q_k)-b\|_2
\]

并保留拒绝、缩步或回退到基线方向的机制。候选 B 的命题不是“网络替代物理算子”，而是：在相同支持集、真实 $A/A^\top$ 调用预算和停止规则下，学习器是否能提供比对角预条件、Anderson/GMRES 类历史基线更好的安全方向。

### 必须长期保留的四个边界

1. **两种绝对值不是一回事。** 候选 A 的“绝对值”是逐元素 $|A_{ij}|$；对称不定线性系统文献中的矩阵绝对值通常是谱定义 $|A|=V|\Lambda|V^\ast$。二者不能互换。
2. **残差下降不等于场误差下降。** 病态逆问题可出现 semi-convergence：数据残差继续下降，而三维场误差已经回升。
3. **神经算子能力不自动迁移到 BOST。** PDE 正问题上的分辨率泛化、几何泛化或低误差，均不能替代 BOST 的相机/光线分布外测试、伴随一致性和真实残差审计。
4. **exact oracle 不等于可部署证书。** 在测试 rig 上先完成 exact-mass sweep 再 clipping，至多证明 oracle 辅助路径安全；它既没有省 sweep，也无法证明 oracle-free predictor 安全。校准包络与确定性证书也必须分栏。

## 1. 证据地图总览

| 证据簇 | 最强直接证据 | 对 A 的作用 | 对 B 的作用 | 当前判断 |
|---|---|---|---|---|
| 对角预条件与 Schur 证书 | Pock–Chambolle 2011 | 理论核心 | 安全回退基线 | A 可先做 |
| 谱绝对值/Schur 预条件 | Vecharynski–Knyazev 2013；Schur 型界 | 概念对照，防止术语误用 | 结构保持启发 | 不可直接移植定理 |
| 学习预条件器 | Li et al. 2023 | 输出约束与 solver-in-loop 训练参考 | 数据条件损失参考 | 高相关邻域证据 |
| 学习优化/历史方向 | Learned Primal-Dual；FCG-NO；Anderson | 反例与对照 | 架构和强制基线 | B 必须与经典历史法比较 |
| 神经算子 | DeepONet、FNO、GINO、NIO | 几何编码备选，不是安全证明 | 低秩/全局模块备选 | 只能作组件或基线 |
| BOS/BOST 物理 | Raffel；Grauer；统一 BOST | 定义输入、几何与真实算子 | 定义残差和可观测量 | 两候选共同物理地基 |
| 有限孔径/景深 | Molnar et al. 2024 | 关键 OOD 轴与教师算子成本 | 前向模型失配压力测试 | 最值得纳入首轮实验 |
| OERF 近邻 | NeRIF；TDBOST | 明确不得重复已有贡献 | 4D、隐式场和张量基线 | 必须围绕而非改名复刻 |
| 不精确/学习算子校正 | Lunz et al.；Learned ReSeSOp；inexact Krylov | 审计误差解释 | 理论和安全机制核心 | B 的必读簇 |
| 早停与 semi-convergence | Hansen；Blanchard et al. | 评价协议 | 接受/停止协议 | 不能只报最终残差 |

## 2. 对角、绝对值与 Schur 型预条件

### 2.1 Pock–Chambolle：一阶原始–对偶算法的对角预条件

**文献**：T. Pock and A. Chambolle, *Diagonal Preconditioning for First Order Primal-Dual Algorithms in Convex Optimization*, ICCV 2011.  
**一级来源**：[IEEE DOI 10.1109/ICCV.2011.6126441](https://doi.org/10.1109/ICCV.2011.6126441)；公式复核可见作者的 [Acta Numerica 综述，DOI 10.1017/S096249291600009X](https://doi.org/10.1017/S096249291600009X)。  
**arXiv**：本次未核验到作者正式 arXiv 版本。

- **真正证明了什么**：对凸原始–对偶一阶方法，可以用易计算的对角步长矩阵替代单一标量步长；在满足 $\|\Sigma^{1/2}KT^{1/2}\|\le 1$ 等条件时保留收敛保证。其 $\alpha=1$ 的构造与算子逐元素绝对值的行、列和直接相关。
- **不能替我们证明什么**：没有学习几何到质量的映射；没有 BOST、有限孔径或分布外实验；没有证明更快的残差下降必然带来更小的三维场误差；也没有允许用未经审计的预测替代精确质量。
- **应提取的公式/实验**：对角 $T,\Sigma$ 的一般 $\alpha$ 公式；范数条件；其证明中如何由逐元素绝对值构造可验证界；标量步长与对角步长的迭代数/时间对照。
- **对应 A/B 的角色**：**A 的理论母体**。A 的创新必须落在“几何条件化预测 + 安全审计 + 成本交叉点”，不能声称发明了绝对行列和预条件。对 B，它是必须保留的无学习安全回退。

### 2.2 加权 Schur 型证书：候选 A 最适合本科阶段完整证明的一步

**来源**：Pock–Chambolle 的对角构造及 [Chambolle–Pock Acta Numerica 综述](https://doi.org/10.1017/S096249291600009X) 中的范数条件；这里使用的是有限维 Schur test 的直接推论。  
**DOI**：10.1017/S096249291600009X。  
**arXiv**：不需要单独依赖预印本。

- **真正证明了什么**：先把证明限制在 $r_i^\star>0,s_j^\star>0$ 的 active row/column support；零质量行没有数据耦合，零质量列必须固定或从更新支持剔除，不能直接代入逆平方根。令该正质量支持上的 $D_r=\operatorname{diag}(r^\star)$、$D_s=\operatorname{diag}(s^\star)$。对 $B=D_r^{-1/2}AD_s^{-1/2}$，用权重 $u_i=\sqrt{r_i^\star}$、$v_j=\sqrt{s_j^\star}$ 可得到 $\|B\|_2\le 1$。因此精确质量下，乘上 $\eta<1$ 可构造保守安全度量。
- **不能替我们证明什么**：它只给范数上界，不说明界是否紧，不给收敛速度最优性，更不说明场误差。若网络低估任一质量，上述证书可能失效；平均相对误差很小也不能代替单侧上界。用测试 rig 的 exact mass 做 `max` clipping 虽可恢复不等式，却不能证明预测器本身安全或省掉 exact sweep。
- **应提取的公式/实验**：完整写出两侧加权不等式和 positive-support/zero-mass 处理；证明预测上界 $\bar r\ge r^\star,\bar s\ge s^\star$ 时证书仍成立；分别设计 oracle-clipped、oracle-free calibrated envelope 和 deployable certificate 三条审计；至少记录“最大低估率、证书/包络 violation、回退率、保守因子、exact sweep 次数、总构造时间”。
- **对应 A/B 的角色**：**A 的可证明核心与审计接口**；但证书名称必须绑定信息合同和成本。对 B，它定义任何学习方向失败后的安全尺度。

### 2.3 谱绝对值预条件：重要，但绝不能与候选 A 混写

**文献**：E. Vecharynski and A. V. Knyazev, *Absolute Value Preconditioning for Symmetric Indefinite Linear Systems*, SIAM Journal on Scientific Computing, 2013.  
**一级来源**：[SIAM 官方页](https://epubs.siam.org/doi/abs/10.1137/120886686)；[DOI 10.1137/120886686](https://doi.org/10.1137/120886686)。  
**arXiv**：[1104.4530](https://arxiv.org/abs/1104.4530)。

- **真正证明了什么**：对方阵、对称、不定的 $A=V\Lambda V^\ast$，谱绝对值 $|A|=V|\Lambda|V^\ast$ 是正定对象；理想的 $|A|^{-1}$ 与预条件 MINRES 有极强谱性质，文章还研究了可实现近似及移位 Laplacian 示例。
- **不能替我们证明什么**：BOST 的 $A$ 通常是矩形投影/敏感度算子；候选 A 使用的是逐元素绝对值。该文的两步收敛等结论不能移植到 $\sum_j|A_{ij}|$ 或 BOST 重建，更不能作为神经预测安全性的证明。
- **应提取的公式/实验**：谱定义；理想预条件后的特征值结构；近似谱等价条件；精确绝对值不可承受时的近似策略和代价。
- **对应 A/B 的角色**：对 A 是**术语防错与结构保持启发**；对 B 可启发“学习校正必须保持求解器所需结构”，但不是直接基线。

## 3. 学习预条件器、学习优化器与历史方向

### 3.1 Learning Preconditioners for Conjugate Gradient PDE Solvers

**文献**：Li et al., *Learning Preconditioners for Conjugate Gradient PDE Solvers*, ICML 2023.  
**一级来源**：[PMLR 正式论文页](https://proceedings.mlr.press/v202/li23e.html)。  
**arXiv**：[2305.16432](https://arxiv.org/abs/2305.16432)。  
**DOI**：PMLR 官方页未列 DOI。

- **真正证明了什么**：在由二阶 PDE 离散得到的稀疏 SPD 系统上，图网络可以预测带受约束非零对角的下三角因子 $L_\theta$，以 $P_\theta=L_\theta L_\theta^\top$ 构造正定预条件器，并在其实验分布上减少 CG 求解成本。单写 $LL^\top$ 只自动保证半正定；正定还依赖 $L$ 满秩/对角约束。文章也展示了从全方向 Frobenius 拟合转向数据条件损失的价值。
- **不能替我们证明什么**：它没有处理矩形、有符号的 BOST 算子，没有 Schur 单侧安全证书，没有三维逆场精度或有限孔径 OOD 结论；$LL^\top$ 的 SPD 保证也不等于候选 A 的质量上界保证。
- **应提取的公式/实验**：$P_\theta=L_\theta L_\theta^\top$ 的结构约束；全矩阵损失与 $\|P_\theta x_i-A_i x_i\|^2$ 数据条件损失；训练时间是否被重复求解摊薄；迭代数、墙钟时间和分布外网格测试。
- **对应 A/B 的角色**：A 的**最邻近 learned-preconditioner 对照**：借鉴“保证由参数化给出”，但输出改为正值质量和可审计上界。对 B，它说明 solver-in-the-loop 损失应覆盖真实迭代方向，而不是只拟合随机向量。

### 3.2 Neural Operators Meet Conjugate Gradients：FCG-NO

**文献**：A. Rudikov, V. Fanaskov, E. Muravleva, A. Laevsky and I. Oseledets, *Neural Operators Meet Conjugate Gradients: The FCG-NO Method for Efficient PDE Solving*, ICML 2024, PMLR 235:42766--42782.  
**一级来源**：[PMLR 正式论文页](https://proceedings.mlr.press/v235/rudikov24a.html)；[OpenReview 页面](https://openreview.net/forum?id=J0ty1o7nCj)。  
**arXiv**：[2402.05598](https://arxiv.org/abs/2402.05598)。  
**DOI**：本次未核验到正式 DOI。

- **真正证明了什么**：神经算子可作为随迭代状态变化的非线性预条件器嵌入 flexible CG，并在文中的 PDE 正问题上与传统/学习基线比较；这比一次性端到端输出更接近“学习器提出方向、经典求解器控制迭代”的范式。
- **不能替我们证明什么**：FCG 的理论背景主要依赖 SPD 线性系统；它不证明矩形 BOST 逆问题、semi-convergence 下的场误差或有限孔径泛化，也不自动保证神经方向不会破坏正则化路径。
- **应提取的公式/实验**：神经预条件调用位于 FCG 的哪一步；训练损失；每轮神经推理与矩阵调用计数；分辨率迁移；失败/不收敛样本。
- **对应 A/B 的角色**：对 A 是远邻。对 B 是**重要架构参考和必须讨论的相关工作**，但 B 应加入真实 $Aq$ 接受测试、逆问题停止规则及 Anderson/无学习历史对照。

### 3.3 Learned Primal-Dual Reconstruction

**文献**：J. Adler and O. Öktem, *Learned Primal-Dual Reconstruction*, IEEE Transactions on Medical Imaging, 2018.  
**一级来源**：[DOI 10.1109/TMI.2018.2799231](https://doi.org/10.1109/TMI.2018.2799231)。  
**arXiv**：[1707.06474](https://arxiv.org/abs/1707.06474)。

- **真正证明了什么**：将已知前向算子及其伴随嵌入有限步原始–对偶展开，学习更新模块，在 CT 数据上获得了文中报告的重建效果；它证明了“物理算子 + 学习更新”是可训练的工程路线。
- **不能替我们证明什么**：有限层展开不自带无限迭代收敛、OOD 稳定性、BOST 几何泛化或证书；医学 CT 的训练/测试分布也不能替代反应流实验。
- **应提取的公式/实验**：每层何处调用前向/伴随；记忆通道；监督目标；与解析/迭代基线的预算是否匹配；层数变化与数据偏移测试。
- **对应 A/B 的角色**：对 B 是“可学习更新器”的强基线思路，同时也是警告：若直接端到端学习整个更新，论文必须额外提供真实残差把关和失败回退。对 A 不应采用如此宽的输出空间。

### 3.4 Anderson acceleration：候选 B 不能绕开的经典历史基线

**文献**：H. F. Walker and P. Ni, *Anderson Acceleration for Fixed-Point Iterations*, SIAM Journal on Numerical Analysis, 2011.  
**一级来源**：[SIAM 官方页](https://epubs.siam.org/doi/abs/10.1137/10078356X)；[DOI 10.1137/10078356X](https://doi.org/10.1137/10078356X)。  
**arXiv**：本次未核验到作者正式 arXiv 版本。

- **真正证明了什么**：Anderson acceleration 用有限残差历史求解小型最小二乘混合问题；在线性问题和特定条件下，它与 GMRES 存在明确联系。有限历史本身不是神经网络的新发明。
- **不能替我们证明什么**：该文不处理病态逆问题的最优早停，不给学习低秩项，也不保证历史加速改善三维真值误差。
- **应提取的公式/实验**：历史矩阵、最小二乘系数、memory depth、阻尼、重启和拒绝规则；在线性情形与 GMRES 的关系；按真实算子调用数而非只按迭代数比较。
- **对应 A/B 的角色**：**B 的强制基线**。若 B 仅因用了历史而变快，不能算学习创新；必须证明学习的几何/相消结构在相同历史长度和调用预算下有增益。

## 4. 神经算子：可用的组件，不是现成答案

### 4.1 DeepONet

**文献**：L. Lu et al., *Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators*, Nature Machine Intelligence, 2021.  
**一级来源**：[Nature 官方页与 DOI 10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5)。  
**arXiv**：[1910.03193](https://arxiv.org/abs/1910.03193)。

- **真正证明了什么**：给出基于 branch/trunk 分解的算子近似架构，并以算子通用逼近理论和多个数值任务说明其表达能力。
- **不能替我们证明什么**：通用逼近是存在性/表达能力结果，不是有限数据训练成功、OOD 几何泛化、逆问题稳定或 Schur 安全性的保证。
- **应提取的公式/实验**：$G(u)(y)\approx\sum_k b_k(u(x_1),\ldots,u(x_m))t_k(y)+b_0$；传感器位置假设；branch/trunk 宽度；不同函数族上的泛化测试。
- **对应 A/B 的角色**：A 可把几何描述放 branch、行/列查询坐标放 trunk，形成可变尺寸质量预测器；B 可用它生成低秩基，但任何输出仍须真实残差检查。

### 4.2 Fourier Neural Operator（FNO）

**文献**：Z. Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.  
**一级来源**：[ICLR 官方论文页](https://iclr.cc/virtual/2021/poster/3281)。  
**arXiv**：[2010.08895](https://arxiv.org/abs/2010.08895)。  
**DOI**：ICLR 官方页未列 DOI。

- **真正证明了什么**：在规则网格上用频域截断参数化全局核，学习多个参数 PDE 的解算子，并展示文中设置下的跨分辨率推理。
- **不能替我们证明什么**：规则网格频域结构不自动适配相机–射线二部图、非均匀体素支持或改变的孔径；PDE 正问题误差不能替代 BOST 逆问题的稳定性和早停评价。
- **应提取的公式/实验**：$v_{l+1}=\sigma(Wv_l+\mathcal F^{-1}(R_l\cdot\mathcal Fv_l))$；保留模态数；零填充/边界处理；分辨率和训练分布外测试。
- **对应 A/B 的角色**：A 仅在行列质量可映射到规则参数网格时使用；B 可把 FNO 作为全局低频校正组件，但必须与更便宜的低秩 SVD/卷积基线比较。

### 4.3 Geometry-Informed Neural Operator（GINO）

**文献**：Z. Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023.  
**一级来源**：[NeurIPS 正式论文集](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)。  
**arXiv**：[2309.00583](https://arxiv.org/abs/2309.00583)。  
**DOI**：NeurIPS 官方页未列 DOI。

- **真正证明了什么**：用图算子在不规则几何与规则潜在网格之间转换，再在潜在网格使用 FNO；在其三维 CFD 几何数据上展示了新几何/边界组合的实验泛化。
- **不能替我们证明什么**：汽车表面/CFD 几何泛化不等于新相机姿态、射线锥、遮挡和体素支持上的 BOST 泛化；SDF 也未必是描述光学系统的充分变量。
- **应提取的公式/实验**：geometry-to-grid、latent FNO、grid-to-geometry 三段映射；SDF/点云输入；邻域半径；新几何切分方式；参数量和显存。
- **对应 A/B 的角色**：对 A 是**几何编码候选**，但首版应先用可解释的小模型证明任务可学；对 B 是处理不规则支持的高级组件，只有在简单低秩/DeepONet 失败后才值得引入。

### 4.4 Neural Inverse Operators（NIO）

**文献**：R. Molinaro et al., *Neural Inverse Operators for Solving PDE Inverse Problems*, ICML 2023.  
**一级来源**：[PMLR 正式论文页](https://proceedings.mlr.press/v202/molinaro23a.html)。  
**arXiv**：[2301.11167](https://arxiv.org/abs/2301.11167)。  
**DOI**：PMLR 官方页未列 DOI。

- **真正证明了什么**：组合 DeepONet 与 FNO 类模块，从一组观测/算子响应学习未知函数，在论文中的 PDE 逆问题上提供了端到端 neural-operator 方案。
- **不能替我们证明什么**：没有 BOST 的光线物理、有限孔径、相机 OOD 或安全预条件证书；直接预测未知场与候选 A 的“只预测质量”是不同研究问题。
- **应提取的公式/实验**：多观测如何编码为函数；DeepONet 与 FNO 的组合顺序；训练网格与测试网格；噪声、观测数量变化和基线。
- **对应 A/B 的角色**：作为**直接逆映射上界/对照**，不作为 A 的教师。对 B，只有在同一数据、同一 rig split、相同训练真值使用规则下才能公平比较。

## 5. BOS/BOST 的物理、有限孔径与 OERF 近邻

### 5.1 BOS 基础综述

**文献**：M. Raffel, *Background-oriented schlieren (BOS) techniques*, Experiments in Fluids, 2015.  
**一级来源**：[Springer 官方页](https://link.springer.com/article/10.1007/s00348-015-1927-5)；[DOI 10.1007/s00348-015-1927-5](https://doi.org/10.1007/s00348-015-1927-5)。  
**arXiv**：本次未核验到正式 arXiv 版本。

- **真正证明了什么**：系统整理 BOS 的参考/受扰图像、折射率梯度引起的视位移、灵敏度与空间分辨率等实验基础，是建立 BOST forward chain 的可靠入口。
- **不能替我们证明什么**：综述不提供候选 A/B 的学习算法，也不保证任意位移估计器或三维反演器的准确性。
- **应提取的公式/实验**：$\rho\rightarrow n$（Gladstone–Dale）→光线偏折→背景位移的链条；放大率、背景距离、像素尺度、噪声和空间分辨率的关系；参考图误差来源。
- **对应 A/B 的角色**：两候选的**物理先修**。A 的几何输入必须能解释该链条中的敏感度变化；B 的真实残差必须回到实际可观测位移，而不是网络潜变量。

### 5.2 瞬时三维火焰 BOST

**文献**：S. Grauer et al., *Instantaneous 3D Flame Imaging by Background-Oriented Schlieren Tomography*, Combustion and Flame, 2018.  
**一级来源**：[Elsevier 官方页](https://www.sciencedirect.com/science/article/pii/S0010218018302694)；[DOI 10.1016/j.combustflame.2018.06.022](https://doi.org/10.1016/j.combustflame.2018.06.022)。  
**arXiv**：本次未核验到正式 arXiv 版本。

- **真正证明了什么**：展示了多相机 BOST 对湍流本生火焰瞬时三维场的重建流程，并用数值 phantom 与实验数据讨论 Tikhonov/TV 等先验；论文使用 23 个视角，是 OERF 三维光学诊断路线的重要物理先例。
- **不能替我们证明什么**：它不证明少视角、有限孔径或学习预条件器的泛化；phantom 上的真值误差与实验上的可观测一致性必须分开。
- **应提取的公式/实验**：相机/背景布局；位移到线积分/体场的离散算子；正则项；phantom 生成；合成真值与真实实验各自报告的指标。
- **对应 A/B 的角色**：为 A 提供真实 $A$ 的来源与几何变量，为 B 提供不得绕开的经典重建基线和物理量级。

### 5.3 Unified BOST：把位移估计与层析重建统一

**文献**：S. Grauer and A. M. Steinberg, *Fast and Robust Volumetric Refractive-Index Measurement by Unified Background-Oriented Schlieren Tomography*, Experiments in Fluids, 2020.  
**一级来源**：[DOI 10.1007/s00348-020-2912-1](https://doi.org/10.1007/s00348-020-2912-1)。  
**arXiv**：本次未核验到正式 arXiv 版本。

- **真正证明了什么**：将传统“先估二维位移、再做三维层析”的两个阶段统一成一个模型，在文中数值 phantom 上减少方程数量并讨论速度、鲁棒性和模型误差。
- **不能替我们证明什么**：统一模型并不等于学习预条件器；文中的加速/鲁棒性不能直接外推到新孔径或新 rig，也不能证明候选 B 的学习校正有益。
- **应提取的公式/实验**：传统两阶段与统一前向模型的矩阵结构；方程数量和内存；位移估计误差如何进入重建；统一/分步方法的相同数据对照。
- **对应 A/B 的角色**：A 应同时测试分步 $A$ 与统一 $A$ 的质量分布是否可迁移；B 可把统一模型的失配作为压力测试，但不能把其模型改进算作学习器贡献。

### 5.4 有限孔径与景深：从 thin ray 到 cone ray

**文献**：J. P. Molnar et al., *Forward and Inverse Modeling of Depth-of-Field Effects in Background-Oriented Schlieren*, AIAA Journal, 2024.  
**一级来源**：[AIAA 官方页与 DOI 10.2514/1.J064095](https://doi.org/10.2514/1.J064095)。  
**arXiv**：[2402.15954](https://arxiv.org/abs/2402.15954)；[arXiv HTML](https://arxiv.org/html/2402.15954)。

- **真正证明了什么**：显式积分孔径上的光线锥，研究有限孔径/景深对折射场成像与反演的影响；其中浮力湍流是数值案例，高超声速球体同时包含数值与实验数据，并比较 thin-ray 与 cone-ray 建模。
- **不能替我们证明什么**：它不证明任意 geometry-conditioned 学习器可跨孔径泛化，也不提供候选 A 的单侧质量证书；神经隐式重建效果不能替代相同算子调用预算下的求解器比较。
- **应提取的公式/实验**：thin-ray 模型与孔径平均 cone-ray 模型；$D=f/N$ 的孔径关系；f-number 扫描；用 thin/cone 模型分别生成和反演数据的四格失配实验；每种模型的构造时间与显存。
- **对应 A/B 的角色**：对 A 是**最关键的几何 OOD 轴和高成本教师算子来源**；对 B 是前向模型失配测试床。第一篇可执行论文应至少包含“训练孔径、插值孔径、外推孔径”三档。

### 5.5 NeRIF：OERF 最接近的神经隐式 BOST 工作

**文献**：Y. He et al., *Neural Refractive Index Field: Unlocking the Potential of Background-Oriented Schlieren Tomography in Volumetric Flow Visualization*, Physics of Fluids, 2025.  
**一级来源**：[AIP 官方页](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)；[DOI 10.1063/5.0250899](https://doi.org/10.1063/5.0250899)。  
**arXiv**：[2409.14722](https://arxiv.org/abs/2409.14722)；[arXiv HTML v2](https://arxiv.org/html/2409.14722v2)。

- **真正证明了什么**：用隐式网络表示折射率及其梯度，以光线/位移一致性训练；论文报告了数值 phantom 的场指标，并在九视角光纤束、1 kHz 实验上展示三维结果和留出投影检验。论文中还描述了随机光线采样、自动微分/有限差分一致性和多头网络等实现细节。
- **不能替我们证明什么**：实验数据没有体场真值，因此投影一致性不能等同于实验场精度；它不证明候选 A 的度量学习、单侧证书、跨相机 OOD，也不证明候选 B 的历史校正。
- **应提取的公式/实验**：网络输出 $n,\nabla n$ 的方式；位移损失与梯度一致性损失；射线采样数；九视角切分；数值/实验指标的严格分栏；每个待重建场的优化/重建总时间。NeRIF 是 per-instance optimization，不应写成训练一次后的 amortized “单帧推理时间”。
- **对应 A/B 的角色**：**两候选最重要的 OERF 邻居**。A 的差异是只摊薄安全度量构造而不替代三维场；B 可围绕 NeRIF/离散求解器的残差轨迹提出安全校正，但不得把“神经隐式 BOST”当作新贡献。

### 5.6 NeDF：稀疏视角神经密度梯度场的近邻

**文献**：J. Li et al., *Neural Deflection Field for Sparse-View Tomographic Background Oriented Schlieren*, Physics of Fluids, 2024.  
**一级来源**：[DOI 10.1063/5.0241191](https://doi.org/10.1063/5.0241191)。  
**arXiv**：[2409.19971](https://arxiv.org/abs/2409.19971)。

- **真正证明了什么**：针对稀疏视角/有限角度 BOST，用神经场表达密度梯度或偏折信息，并在论文设定上展示重建。
- **不能替我们证明什么**：稀疏视角有效不等于孔径、相机位置和噪声共同 OOD；也没有给候选 A 的预条件证书或候选 B 的残差接受定理。
- **应提取的公式/实验**：神经场究竟表示密度、梯度还是偏折；视角数量扫描；损失中的物理约束；是否使用真值/验证视角选择超参数。
- **对应 A/B 的角色**：候选 A 的少视角 OOD 测试参考；候选 B 的直接神经场基线。优先级低于 NeRIF，但可防止把“少视角神经 BOST”误报为创新。

### 5.7 TDBOST：四维张量分解路线

**文献**：Y. He et al., *Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction*, ACM Transactions on Graphics, 2026.  
**一级来源**：[ACM 官方页与 DOI 10.1145/3809488](https://dl.acm.org/doi/10.1145/3809488)；[作者官方代码仓库](https://github.com/Hyz617/TDBOST)。  
**arXiv**：截至本次核验未找到可确认的正式 arXiv 页面。

- **真正证明了什么**：将 $X-Y-Z-T$ 四维场分解为多组二维分量，并结合轻量网络、畸变校正、双线性光线追踪/反传和混合精度；论文官方摘要报告了无预训练逐帧重建效率，代码仓库提供实现入口。
- **不能替我们证明什么**：公开摘要和代码不能证明候选 A/B；“快速四维”不等于跨 rig 泛化、预条件安全或比所有神经算子更优。若原始喷雾数据另行托管，未获得数据前不能承诺完整复现。
- **应提取的公式/实验**：三组二维因子的精确定义；时空秩；光线追踪和损失；每帧时间的硬件/分辨率；消融；原始数据、校准和代码实际可获得边界。
- **对应 A/B 的角色**：对 A，可检验同一 rig 的质量度量能否跨时间帧摊薄，但这只是待证假设。对 B，它是**四维主线最近邻**：学习历史校正必须与 TDBOST 的显式时空低秩分解区分，并在同一数据协议上比较。

## 6. 不精确算子与学习校正：候选 B 的安全文献簇

### 6.1 On Learned Operator Correction in Inverse Problems

**文献**：S. Lunz et al., *On Learned Operator Correction in Inverse Problems*, SIAM Journal on Imaging Sciences, 2021.  
**一级来源**：[SIAM 官方页](https://epubs.siam.org/doi/10.1137/20M1338460)；[DOI 10.1137/20M1338460](https://doi.org/10.1137/20M1338460)。  
**arXiv**：[2005.07069](https://arxiv.org/abs/2005.07069)。

- **真正证明了什么**：当只有近似前向算子时，仅校正 forward map 可能受近似伴随的值域限制；文章研究 forward/adjoint correction、递归训练，并在明确假设下给出收敛到邻域的分析和有限视角光声案例。
- **不能替我们证明什么**：理论假设与邻域结论不能自动套到 BOST；它不允许分别学习两个互不相容的 forward/adjoint 后仍声称物理一致，也不保证场误差优于精确算子基线。
- **应提取的公式/实验**：近似算子、forward correction 和 adjoint correction 的组合；梯度一致性；递归收集迭代轨迹；收敛假设；校正前后的数据/梯度误差。
- **对应 A/B 的角色**：对 B 是**关键理论警戒线**。最稳妥设计仍让真实 $A,A^\top$ 掌握残差和接受判据；若学习近似算子，必须成对审计 forward/adjoint。对 A，它说明精确质量教师与预测器误差应分别记账。

### 6.2 Learned ReSeSOp：面向模型不精确性的学习迭代

**文献**：M. S. Feinler and B. N. Hahn, *Learned RESESOP for Solving Inverse Problems with Inexact Forward Operator*, Inverse Problems, 2025.  
**一级来源**：[IOP 官方 DOI 10.1088/1361-6420/adef73](https://doi.org/10.1088/1361-6420/adef73)。  
**arXiv**：[2410.23061](https://arxiv.org/abs/2410.23061)。

- **真正证明了什么**：把 sequential subspace optimization 的结构与学习模块结合，显式面向未知/变化的模型不精确性，并在论文的动态逆问题上评价。
- **不能替我们证明什么**：没有证明 BOST cone-ray/thin-ray 误差满足同样模型，也没有替候选 B 给出数据集、误差界或严格的调用预算。
- **应提取的公式/实验**：搜索子空间、残差条带/误差预算、学习器输入输出、动态数据设置、已知与未知模型误差的切分。
- **对应 A/B 的角色**：B 的**最相关结构化学习基线之一**；可借鉴“学习误差预算/子空间而非直接改写物理”，并用有限孔径失配构造真实压力测试。

### 6.3 Inexact Krylov：每一步允许多大算子误差

**文献**：V. Simoncini and D. B. Szyld, *Theory of Inexact Krylov Subspace Methods and Applications to Scientific Computing*, SIAM Journal on Scientific Computing, 2003.  
**一级来源**：[SIAM 官方页](https://epubs.siam.org/doi/abs/10.1137/S1064827502406415)；[DOI 10.1137/S1064827502406415](https://doi.org/10.1137/S1064827502406415)。  
**arXiv**：本次未核验到正式 arXiv 版本。

- **真正证明了什么**：为 Krylov 迭代中的不精确矩阵–向量乘给出可计算的误差控制思想，说明允许误差可随残差阶段调整而仍维持目标收敛行为。
- **不能替我们证明什么**：线性 Krylov 残差理论不等于学习非线性校正的稳定性，也不处理逆问题真值误差和 semi-convergence。
- **应提取的公式/实验**：matvec 误差 $e_k$ 与当前残差/目标容差的关系；松弛策略；真实残差重算频率；理论残差与递推残差漂移。
- **对应 A/B 的角色**：B 的**逐步误差预算和重算协议**来源。可把学习近似 $Aq_k$ 视作 inexact product，但最终接受前仍应周期性或逐步使用真实 $Aq_k$。

## 7. 早停与 semi-convergence：论文评价不能缺的一层

### 7.1 Hansen：迭代正则化与 semi-convergence

**文献**：P. C. Hansen, *Rank-Deficient and Discrete Ill-Posed Problems: Numerical Aspects of Linear Inversion*, SIAM, 1998，特别是第 6 章。  
**一级来源**：[SIAM 第 6 章 DOI 10.1137/1.9780898719697.ch6](https://doi.org/10.1137/1.9780898719697.ch6)。  
**arXiv**：不适用。

- **真正证明了什么**：在离散病态线性逆问题中，迭代次数本身可充当正则化参数；Krylov/迭代解先逼近有用的正则化解，随后逐渐吸收小奇异值方向上的噪声，这就是 semi-convergence。
- **不能替我们证明什么**：教材不提供 BOST 的自动最优停止轮数，也不保证最小数据残差处有最小场误差；真实实验无真值时尤其不能直接选 oracle stop。
- **应提取的公式/实验**：SVD/filter factors；Picard 条件；CGLS/LSQR 误差随迭代的轨迹；hybrid regularization；同时画残差、场误差、梯度误差与频谱能量。
- **对应 A/B 的角色**：A/B **共同评价地基**。所有方法都要报告 best-oracle（只作诊断）、固定预算和 observation-only stop 三种结果；训练与调参不能偷看测试真值的最佳轮数。

### 7.2 Optimal Adaptation for Early Stopping in Statistical Inverse Problems

**文献**：G. Blanchard, M. Hoffmann and M. Reiß, *Optimal Adaptation for Early Stopping in Statistical Inverse Problems*, SIAM/ASA Journal on Uncertainty Quantification, 2018.  
**一级来源**：[DOI 10.1137/17M1154096](https://doi.org/10.1137/17M1154096)。  
**arXiv**：本次以正式期刊页为核验依据。

- **真正证明了什么**：在论文规定的统计逆问题模型与假设下，基于残差/差异原则的停止先控制 prediction/weak error，并通过 weak-to-strong transfer 建立强 $L^2$ 误差的 oracle-adaptation bounds，同时分析适应范围和限制。
- **不能替我们证明什么**：该 weak-to-strong 转移依赖论文中的噪声、谱衰减与正则性假设；BOST 噪声可能相关、异方差且含 forward/model error，不能未经检验就把该强误差界移植到三维场。
- **应提取的公式/实验**：停止统计量、阈值与噪声尺度；weak-to-strong transfer 的前提；最小迭代下界；不同噪声和奇异值衰减下的适应/失败区间。
- **对应 A/B 的角色**：为 A/B 提供**不看真值的停止规则候选**。论文应预注册停止参数，并把 oracle best 仅作为“还能改进多少”的诊断上界。

## 8. 哪些结论现在可以写，哪些还不可以

### 可以有文献支撑地写

- 逐元素绝对值行列和可构造原始–对偶对角度量，并可通过 Schur 型论证给出范数证书。
- 学习预条件器、学习迭代更新和 neural operator 已在 PDE/医学逆问题中显示可行，但其保证依赖问题结构与实验分布。
- 有限孔径会改变 BOS/BOST 前向成像，thin-ray/cone-ray 失配是现实模型误差，而不是人为构造的纯机器学习难题。
- NeRIF 已经把神经隐式场用于 OERF 相关三维 BOST；TDBOST 已把张量分解和轻量网络用于四维重建。
- 病态逆问题存在 semi-convergence；评价不能只看最后一轮残差。

### 目前绝对不能写

- “学习 $r,s$ 后仍自动拥有 Pock–Chambolle 收敛保证。”只有通过单侧上界审计或精确回退后才可能这样说。
- “逐测试 rig exact-max clipping 证明预测器可部署并节省 exact sweep。”该路径已经读取完整 exact mass，只是 oracle 加额外阻尼。
- “calibration rigs 上零 violation，所以得到确定性证书。”这最多支持冻结包络在相应统计假设下的经验/概率结论。
- “数据残差更小，所以三维场更准确。”必须有合成真值、独立模态/投影或其他外部测量。
- “首次将神经网络用于 BOST/三维重建。”NeRIF、NeDF、TDBOST 等已经排除该表述。
- “比 FNO/DeepONet/GINO 更好。”除非输入、训练数据、参数量、算子调用、停止规则和 OOD 切分全部对齐。
- “有限孔径上泛化。”除非外推孔径、相机布局和会话级留出均通过；同一 rig 内随机切帧不算几何泛化。

## 9. 本科生按先修顺序的 12 篇/章核心阅读路线

> 原则：先懂测量和病态性，再懂安全度量，随后学习 neural operator，最后进入 OERF 近邻。每篇都必须留下一个可检查产物，而不是只做摘要。

| 顺序 | 核心阅读 | 读前先修 | 必须产出的学习成果 |
|---:|---|---|---|
| 1 | Raffel et al. 2015, BOS review | 几何光学、折射率 | 一页 forward chain；列出 8 个实验误差源及其影响方向 |
| 2 | Grauer et al. 2018, 3D flame BOST | 线积分、最小二乘 | 重画 23 视角重建流程；区分 phantom 真值指标与实验可观测指标 |
| 3 | Pock–Chambolle 2011 | 凸优化、矩阵范数 | 从行列绝对和完整推导 $\|\Sigma^{1/2}AT^{1/2}\|\le1$ 的有限维版本 |
| 4 | Hansen 1998, Chapter 6 | SVD、噪声传播 | 用一个病态矩阵复现 residual 下降而 field error 回升的 semi-convergence 图 |
| 5 | DeepONet 2021 | MLP、函数空间概念 | 手写 branch/trunk 前向式；说明如何输出可变行/列查询的质量 |
| 6 | FNO 2021 | FFT、卷积 | 实现一层 Fourier layer；记录规则网格假设何时破坏 BOST 几何 |
| 7 | GINO 2023 | 图网络、点云/SDF | 画 geometry→grid→FNO→geometry 数据流；与相机–射线二部图逐项对照 |
| 8 | Li et al. 2023, learned preconditioner | CG、SPD、Cholesky | 复现 $LL^\top$ 结构保证；把损失改写成候选 A 的单侧低估惩罚草案 |
| 9 | Walker–Ni 2011, Anderson acceleration | 最小二乘、固定点迭代 | 在同一 toy inverse problem 上实现 memory 0/2/5；按 matvec 次数比较 |
| 10 | Lunz et al. 2021, learned operator correction | 伴随、梯度法 | 解释为什么只校正 forward 不够；写出候选 B 的真实 $A/A^\top$ 审计伪代码 |
| 11 | He et al. 2025, NeRIF | BOS、隐式神经场、自动微分 | 按“有真值/无真值”重建其证据表；复写损失和射线采样协议 |
| 12 | He et al. 2026, TDBOST | 低秩分解、时空场 | 画三组二维因子的 4D 表达；列出候选 B 与 TDBOST 必须不同的三项贡献 |

### 并行补读，不占 12 篇名额

- Molnar et al. 2024：在开始任何跨孔径实验前完成。
- Simoncini–Szyld 2003：候选 B 开始使用近似 $Aq$ 前完成。
- NIO 2023：准备端到端 inverse-operator 对照时完成。
- Learned ReSeSOp 2025：准备模型失配论文命题时完成。
- FCG-NO：准备“神经预条件器”相关工作和算力预算时完成。

## 10. 五个明确、可证伪的创新空白

> “空白”表示在本次一级来源核验范围内，尚未找到同时满足下列组合的工作；它不是全球首次声明。正式投稿前仍须做数据库级系统检索、引文向前/向后追踪和导师确认。

### 空白 1：带可部署证书的 geometry-conditioned BOST 绝对质量算子（候选 A，首选）

- **缺失组合**：Pock–Chambolle 型逐元素绝对质量 + BOST 相机/孔径几何编码 + oracle-free 预测 + 信息/成本合同清楚的单侧安全机制 + 新 rig 外推。
- **可检验假设**：测试 rig 不读取 $r^\star,s^\star$ 时，预测器配合冻结 calibration envelope 或真正可部署的确定性 bound，能降低度量构造总成本；若只用 envelope，主张降为概率/经验安全，不能写成逐 rig certificate。
- **最小实验**：thin/cone 两类算子；训练、独立 safety-calibration、fresh geometry-OOD 三段切分；线性/MLP/DeepONet；factor、scalar PDHG、exact、simple scalar damping、exact–factor interpolation、unclipped predictor、frozen envelope 和 exact-max-clipped oracle 八组。
- **论文门槛**：部署主结果的 test-rig exact sweep/audit 次数为零；teacher、oracle 诊断、calibration 和 deployment 四栏分开；给出时间–内存交叉点，并按固定停止和 observation-only stop 报告场误差。若只有 exact-max clipping 安全，结论必须停在 oracle diagnostic；若 exact sweep 本来就便宜，则诚实判为不值得学习。

### 空白 2：部分精确查询下的单侧质量包络与主动审计（候选 A，高风险升级）

- **缺失组合**：不是预测均值，而是用少量精确行/列查询建立上包络，并主动选择最可能低估的行列；关键空白是如何约束**未查询**行列，而不是只让已查询位置通过。
- **可检验假设**：只计算一部分 $r_i^\star,s_j^\star$，结合可证明的几何/Lipschitz majorizer 可形成确定性 bound；若只依赖独立 calibration 分位数，则目标改为明确覆盖率的 safety envelope。
- **最小实验**：随机审计、按不确定度审计、几何覆盖审计、全量精确、simple scalar damping 和 exact–factor interpolation；横轴用“精确查询比例/实际 nonzero 访问量”，纵轴用 violation、覆盖率、保守度、总时间和迭代数。
- **论文门槛**：确定性与概率结论分开；概率保证说明 exchangeability/OOD 失效边界；若为构造特征、包络或最终 clipping 仍遍历全部非零元，就没有真实加速。exact-factor interpolation 只作 oracle tightness 曲线，不能混入部署结果。

### 空白 3：真实残差接受的相消感知低秩–历史校正（候选 B，A 通过后再做）

- **缺失组合**：针对 BOST 有符号行贡献的相消结构，学习低秩/历史候选方向，但每步由真实 $Aq_k$ 接受、缩步或拒绝；并与 Anderson、GMRES/LSQR 和 FCG-NO 风格方法按调用预算对齐。
- **可检验假设**：相消统计和几何上下文能预测“经典有限历史未捕获但真实残差会接受”的方向，从而降低达到同一 observation-only stop 所需的 $A/A^\top$ 调用。
- **最小实验**：无历史、Anderson、随机低秩、只历史网络、相消感知网络；相同支持、相同历史长度、相同真实 matvec 预算；记录接受率和拒绝后的退化行为。
- **论文门槛**：优势必须在多个噪声、孔径和留出 rig 上存在；若仅 oracle best 场误差更好、现实停止规则下不更好，则不能宣称实用胜利。

### 空白 4：有限孔径模型失配下的 forward–adjoint 一致学习校正（候选 B，可与 A 联合）

- **缺失组合**：以 thin-ray 为便宜近似、cone-ray 为高保真参考，学习受约束的校正或误差预算，同时显式审计 forward/adjoint 一致性和真实 cone-ray 残差。
- **可检验假设**：比直接端到端重建更窄的 operator correction 能以较少 cone-ray 调用恢复大部分有限孔径误差，并对未见 f-number 更稳健。
- **最小实验**：thin→thin、cone→cone、cone 数据/thin 反演、学习校正四组；增加 forward-only 与 forward+adjoint 校正；报告梯度方向夹角和真实 cone residual。
- **论文门槛**：训练和测试都不能使用不可获得的真值尺度；必须包含完全不校正、提高物理模型精度和 Learned ReSeSOp 类结构化基线。

### 空白 5：不看体场真值的 BOST semi-convergence 停止器（A/B 共用）

- **缺失组合**：把差异原则、残差谱/留出投影、迭代稳定性与 BOST 的相机相关噪声结合，预测接近场误差最优点的停止，而不在测试时看体场真值。
- **可检验假设**：在合成数据上校准、在留出几何和实验留出投影上冻结的停止器，能比固定轮数和单纯 residual threshold 更接近 oracle best，同时不偏袒学习方法。
- **最小实验**：固定轮数、Morozov/残差阈值、L-curve 类规则、留出投影、学习停止器；所有规则在测试前冻结；画 stop regret $E_{\text{stop}}-E_{\text{oracle}}$。
- **论文门槛**：必须分别报告弱/投影误差和强/体场误差；如果只在同一 morphology 随机切帧有效，不可宣称跨实验泛化。

## 11. 给当前项目的决策结论

1. **先做候选 A，不要同时开大 B。** A 有最清晰的数学结构、标签可由算子精确生成、无需实验体场真值，并能形成“成功/失败都可解释”的本科论文；但当前 exact-max-clipped smoke 只算 oracle 负审计，尚未通过 oracle-free 替代门。
2. **把有限孔径设为第一现实困难。** 它有正式 BOST 文献支撑，可生成 thin/cone 成对算子，也能检验几何模型是否真的学到光学变化。
3. **把候选 B 限制为 residual-verified direction proposer。** 保留真实 signed $A/A^\top$、Anderson 基线、调用预算和回退，避免变成无法解释的端到端网络。
4. **论文主指标不是“某一张图更漂亮”。** A 首先看分层后的 certificate/envelope violation、exact sweep 次数、保守度和构造/摊薄成本，并强制比较 scalar damping 与 exact–factor interpolation；B 首先看真实调用数、接受率、现实停止下的误差和 OOD 退化。
5. **拿到实验室数据前可完成方法学底座，但不能写实验结论。** 公开/合成数据用于证明代码、协议和可证伪假设；拿到师兄的校准、真实位移和数据划分后，再做最终迁移。

## 12. 应向何远哲师兄一次性确认的问题

1. 实验室当前实际使用的是分步 BOST、unified BOST、NeRIF、TDBOST，还是并行保留多条 pipeline？候选 A 要服务哪一个真实 $A$？
2. `A` 是否显式存储、按光线即时生成，或由自动微分隐式给出？计算逐元素 `|A|` 行列质量的真实耗时、内存和复用周期是多少？
3. 哪些几何会变化：相机姿态、镜头、f-number、背景距离、视角数、体素支持、折射率范围、实验会话？哪些只在标定时变化？
4. 能否提供至少两套独立标定/rig 或 thin/cone 模拟器，使跨几何测试不只是同一配置随机切帧？
5. 现有求解器及其停止规则是什么？是否观察到 residual 继续下降而场质量变差？有没有独立投影、温度/密度或 CFD 场可作外部验证？
6. TDBOST 的训练/优化是逐场、跨场预训练还是混合？论文中的 12.6 s/frame 包含哪些预处理和硬件条件？
7. 对投稿而言，师兄更看重“可证明安全的成本摊薄”（A）还是“更快/更准的四维方向校正”（B）？可接受的失败率和额外代码复杂度是多少？
8. 师兄是否同意把 exact teacher/oracle、safety-calibration 和真正部署测试完全拆成三套 rig？部署阶段允许读取哪些解析 majorizer，哪些量一旦读取就等同于重新做 exact sweep？

---

### 一句话研究定位

**不让网络替代 BOST 物理，而让网络预测可审计的求解器结构；先用候选 A 建立证书和真实成本，再让候选 B 在真实残差约束下学习相消感知的低秩/历史方向。**
