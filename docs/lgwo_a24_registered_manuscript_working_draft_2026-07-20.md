# 固定算子预算下的几何条件有界首方向策略：面向三维 BOST 重建的预注册论文工作稿

**英文工作题目：** *Observable Geometry-Conditioned Bounded First-Direction Learning in a Fixed-Budget CGLS Shell for Three-Dimensional BOST Reconstruction*

**内部代号：** LGWO-A24-L1
**文档类型：** Registered-report 风格工作稿；可持续填入真实结果，但不得回写或覆盖预注册规则
**版本日期：** 2026-07-20
**作者：** `[待确认]`
**指导教师与合作成员：** `[待确认后填写]`
**目标装置/数据：** OERF 三维背景纹影（BOST）；真实接口与数据合同尚未取得
**当前科学状态：** `NO-GO / NOT RUN`

> **不可删除的状态声明**
>
> 截至本稿冻结时点：**0 scientific cases / 0 optimizer steps / no breakthrough**。
> 当前没有可学习模型结果、没有 route-development 结果、没有 fresh/OOD 结果、没有真实 BOST 结果，
> 也没有相对 DCDM、FCG-NO、DeepONet、FNO、iFNO、NeRIF、NeDF 或 TDBOST 的性能证据。
> 已完成的算法壳、调用账本、梯度与可重复性检查属于工程证据，不是科学样本。
> 先前 6 个 opened synthetic oracle cases 只说明 truth-informed exact-null direction 在特定 toy operator 中存在
> 表示余量；它们永久排除在本研究的训练、选模和结论之外。

---

## 摘要（注册阶段，不含结果）

### 背景

三维背景纹影通过多视角背景位移反演折射率或密度相关标量场。有限视角、噪声、几何标定误差、直线与弯曲
光线模型失配会使逆问题严重欠定。直接神经逆映射、隐式神经场、时空低秩表示和神经 Krylov/预条件方法均已有
直接先例，因此“神经网络用于 BOST”或“学习共轭方向”本身不构成新颖性。

### 目的

本研究拟证伪或支持一个较窄的问题：在部署时只使用原始位移、一次伴随场、相机/射线几何与 support 的条件下，
一个轻量、范数受限、只修改 CGLS 第一搜索方向的策略，能否在保持每例 `24A/24A^T` 的固定预算时，改善未见
几何簇上的三维场误差，并且不破坏观测 A 与独立射线束 B 的一致性。

### 方法

主候选 `LD-FiLM-8` 含 2,729 个参数，输出相对伴随锚点范数不超过 5% 的首方向修正。修正为零或输入越出
fit-only 经验校准包络时，算法精确回退到锚定的 CGLS-24 壳层。研究按完整 geometry cluster、rig、session 与
时间块隔离数据；主终点为 field relative-L2；同时报告逐 rig 尾部、A/B 重投影一致性、`A/A^T` 调用、端到端
墙钟、训练成本与峰值内存。所有直接网络、神经预条件、逐例优化和经典方法均按预注册的双预算协议比较。

### 当前结果

`NOT RUN`。当前合格科学样本数为 0，优化器更新步数为 0，因此不能给出效应量、置信区间、图表、优胜方法或
任何突破性结论。

### 预注册结论边界

只有在 L1、独立 L2、未见 rig/renderer 与真实 OERF 独立视角验证依次通过后，才可讨论一种 BOST 专用的
固定预算首方向策略。单一 synthetic mean gain、最好 seed、打开后的超参数选择或工程测试均不构成算法成功。

**关键词：** 背景纹影；三维重建；算子学习；CGLS；有限视角；几何条件化；固定预算；预注册

---

## 1. 一句话研究主张与可证伪边界

本稿研究的不是“一个新的通用神经算子”，而是以下**待证问题**：

> 在三维 BOST 的欠定几何中，observable-only、geometry-conditioned、norm-bounded 的首方向修正，是否能在
> 不增加部署 `A/A^T` 调用的前提下，比同预算 CGLS-24 和强学习基线更好地恢复场，并在独立 B 射线束、未见
> rig 与真实装置上守住尾部风险；当输入超出校准域时，是否能逐元素回退到原求解器？

这句话必须能够被结果否定。出现以下任一情况，核心主张即失败：

1. field relative-L2 没有达到预注册效应门，或置信区间下界不大于 0；
2. 平均改善由单一 seed、单一 family、单一 rig 或少数样本驱动；
3. A 或 B 一致性劣化超过门限；
4. 同预算下不优于 fixed direction、线性修正、DCDM-style 或 FCG-NO-style 强基线；
5. 优势在新 geometry、renderer shift 或真实 BOST 中消失；
6. 墙钟、内存或训练摊销成本使所谓“固定调用优势”失去现实意义；
7. 结果依赖 truth、case id、split、geometry digest、exact-null basis 或其他不可部署输入；
8. 需要在看过 route/fresh 结果后更换 seed、半径、网络、损失或停止规则。

---

## 2. 给物理本科生的最小概念表

| 术语 | 本文中的含义 | 容易误解的地方 |
|---|---|---|
| forward operator `A` | 给定三维场与已标定几何，预测 detector 上的背景位移 | 它不是普通矩阵乘法的代名词；真实系统可能含弯曲光线、插值、遮挡和终止分支 |
| adjoint `A^T` | 把测量残差反投影回三维空间的线性算子 | adjoint 不等于 inverse；`A^T y` 通常只是有条纹/模糊的反投影锚点 |
| 欠定/病态 | 很多不同三维场都能产生接近的有限视角测量 | 数据残差很小不代表真实场误差小 |
| null/near-null | 对当前 A 完全或近似不可见的场方向 | 当前方法没有显式 null 投影，不能称为 learned null-space projection |
| rig | 一套相机数量、姿态、标定、探测器与光线路径配置 | 相邻帧共享 rig；随机拆帧会造成信息泄漏 |
| heldout-B | 与 A 不同的独立射线束/相机几何，用于训练辅助或封闭评分 | 当前 synthetic B 与 A 共享同一解析 truth 和 renderer 家族，不是独立物理实验 |
| fixed budget | 比较方法支付相同的 forward/adjoint 调用，另报墙钟和内存 | 相同 `A/A^T` 不代表相同总成本；神经推理、重正交与反传都要计时 |
| amortized policy | 网络跨多个样本学习后，为新样本快速提出一个求解方向 | 它不同于 NeRIF/NeDF/TDBOST 的逐例或逐序列参数优化 |
| fail-closed fallback | 合同或经验包络不满足时，修正置零并运行原基线 | 这是工程回退，不是概率安全保证，也不是新颖性贡献 |

---

## 3. 物理问题

### 3.1 BOST 观测链

设待重建标量场为 `x(r)`。它可以代表折射率扰动，也可以在温度、组分与波长条件明确后通过物性关系转换为
密度相关量。本文不在数据合同缺失时把 `x` 强行解释为温度或密度；物理量、单位、波长和转换常数必须由真实
实验记录确认。

在小偏折、直线射线近似下，第 `c` 台相机的二维位移可抽象写为

```math
y_c = \mathcal{S}_c\!\left[\int_{\ell_c} \nabla_\perp x(\mathbf r)\,\mathrm d\ell\right] + \varepsilon_c,
```

其中 `\ell_c` 是由标定几何决定的光线，`\nabla_\perp` 是垂直于光线的场梯度，`\mathcal S_c` 汇总成像比例、
背景距离、插值和 detector 采样，`\varepsilon_c` 包含背景位移提取误差与噪声。离散后写为

```math
y_A = A_\gamma x_\star + \varepsilon_A,
```

`\gamma` 表示完整 rig/标定合同。若折射足以弯曲光线，则观测更诚实地写成非线性算子

```math
y_A = \mathcal F_\gamma(x_\star) + \varepsilon_A.
```

当前 LGWO-A24 壳层只对线性 `A_\gamma` 建立。真实实验是否使用 straight、curved 或局部线性化
`A=J_{\mathcal F}(x_0)`，必须在何远哲师兄确认 callable、JVP/VJP 和 residual 层级后冻结。任何 wrapper 末端的
两张 detector map 相减，都不能冒充 ray/sample 层原生 residual。

### 3.2 为什么场误差与数据误差会分离

有限相机只观测沿光线积分后的横向梯度。若 `h` 满足 `\|A_\gamma h\|` 很小，则 `x` 与 `x+h` 在 detector 上
近乎不可区分，即使二者三维结构差异明显。因此经典 CGLS 容易先恢复可观测分量，而把弱可观测分量留给先验、
正则化或学习策略。LGWO 的窄假设是：部署可观测的 `y_A`、`A^T y_A` 和几何也许含有跨样本统计线索，足以对
首方向作一个小而有用的修正；这不是已知事实，正是需要被证伪的对象。

### 3.3 现实物理失效轴

真实验证至少要分别暴露以下误差来源，不能把它们混成一个“噪声等级”：

1. 相机方位角、仰角、数量与遮挡造成的有限视角；
2. detector 分辨率、背景纹理、位移算法与亚像素误差；
3. 外参、内参与背景距离的标定偏差；
4. straight-ray 与 curved-ray 模型失配；
5. support 错位、边界截断与视场外结构；
6. 折射率到密度/温度转换中的物性与组分不确定性；
7. 火焰/反应流的界面、薄层、高梯度和时序变化；
8. 插值、ray termination、hard mask、自适应采样造成的非光滑导数。

---

## 4. 数学问题与主终点

给定 A 组观测 `y_A`、几何 `\gamma_A`、support 投影 `P_S` 和固定预算 `K=24`，定义强基线

```math
\hat x_b = \operatorname{CGLS}_{24}(A_{\gamma_A},y_A,P_S).
```

主候选产生

```math
\hat x_\theta
= \operatorname{LGWO}_{24}(A_{\gamma_A},y_A,\gamma_A,P_S;\theta).
```

在 synthetic/CFD truth 可用时，唯一主终点为 support 内 field relative-L2：

```math
E_{\mathrm{field}}(\hat x,x_\star)
= \frac{\|P_S(\hat x-x_\star)\|_2}
       {\|P_Sx_\star\|_2+\epsilon}.
```

相对基线增益定义为

```math
G_{\mathrm{field}}
= 1-\frac{E_{\mathrm{field}}(\hat x_\theta,x_\star)}
          {E_{\mathrm{field}}(\hat x_b,x_\star)+\epsilon}.
```

`G_field>0` 表示候选误差较低。真实实验若没有体积 truth，field relative-L2 不得伪造；此时必须使用独立相机、
可测 phantom、同步诊断或可信 CFD/标定体作为外部参照，并把证据等级单独标注。

---

## 5. LGWO-A24 方法

### 5.1 部署输入防火墙

网络只允许读取：

```text
observation_uv       y_A
pooled_adjoint_field g = P_S A^T y_A
geometry_features    gamma_A 中预声明且 truth-free 的特征
support              P_S
```

部署 API 禁止读取：truth、clean observation、family、partition、case/phantom seed、geometry digest、baseline
误差、评价指标、exact-null basis、B geometry、B observation、B residual 及其任何派生量。proposal 前后 operator
ledger 必须完全一致，proposal 内部为 `0A/0A^T`。

### 5.2 有界首方向

先计算伴随锚点

```math
g=P_SA^Ty_A.
```

令 2,729 参数的 `LD-FiLM-8` 输出 `d_{raw}=M_\theta(y_A,g,\gamma_A,P_S)`。实际修正为

```math
d=P_Sd_{raw},\qquad
\delta_\theta
=d\min\!\left(1,
\frac{\eta\|g\|_2}{\max(\|d\|_2,\epsilon)}\right),
\qquad \eta=0.05,
```

并定义首方向

```math
p_0=g+\delta_\theta,\qquad u_0=Ap_0.
```

`eta=0.05` 在 L1 固定，不得用 route 结果 sweep。未来半径曲线只能在另行预注册的 L2 中测试。

### 5.3 精确首步与 23 步延续

从 `x_0=0`、`r_0=y_A` 出发：

```math
\alpha_0=\frac{\langle r_0,u_0\rangle}{\langle u_0,u_0\rangle},\qquad
x_1=P_S(\alpha_0p_0),\qquad
r_1=r_0-\alpha_0u_0.
```

对 `j=1,\ldots,23`：

```math
s_j=P_SA^Tr_j,\qquad v_j=As_j.
```

对所有历史 `(p_i,u_i=Ap_i)` 做两遍 measurement-space modified Gram-Schmidt。每一遍都执行

```math
\beta_{ji}=\frac{\langle v_j,u_i\rangle}{\langle u_i,u_i\rangle},\qquad
s_j\leftarrow s_j-\beta_{ji}p_i,\qquad
v_j\leftarrow v_j-\beta_{ji}u_i.
```

然后令 `p_j=s_j`、`u_j=v_j`，并做精确一维最小二乘：

```math
\alpha_j=\frac{\langle r_j,u_j\rangle}{\langle u_j,u_j\rangle},\qquad
x_{j+1}=P_S(x_j+\alpha_jp_j),\qquad
r_{j+1}=r_j-\alpha_ju_j.
```

非有限数、分母失效或合同错误必须记录 breakdown；不得静默减少步数、换精度或改求解器。

### 5.4 固定调用预算

| 阶段 | `A` | `A^T` | 神经网络调用 |
|---|---:|---:|---:|
| pooled anchor `g=P_SA^Ty` | 0 | 1 | 0 |
| 有界 proposal | 0 | 0 | 1 |
| `A(g+delta)` | 1 | 0 | 0 |
| 23 个延续步 | 23 | 23 | 0 |
| **总计** | **24** | **24** | **1** |

两遍重正交、几何特征、网络推理、B evaluator、训练反传、墙钟与内存必须另报。`24A/24A^T` 只是物理算子
调用合同，不等于“总成本相同”。

### 5.5 训练目标

对 synthetic fit case，令 `x_theta` 为 24 步输出，`x_b` 为 CGLS-24，`x_star` 为 truth。定义

```math
E_2=\frac{\|x_\theta-x_\star\|_2}{\|x_\star\|_2+\epsilon},\qquad
E_{H^1}=\frac{\|\nabla(x_\theta-x_\star)\|_2}{\|\nabla x_\star\|_2+\epsilon}.
```

并令

```math
q_f=E_2/(E_{2,b}+\epsilon),
\quad q_{Am}=\|r_{Am,\theta}\|/(\|r_{Am,b}\|+\epsilon),
```

```math
q_{Ac}=\|r_{Ac,\theta}\|/(\|r_{Ac,b}\|+\epsilon),
\quad q_{Bc}=\|r_{Bc,\theta}\|/(\|r_{Bc,b}\|+\epsilon),
```

```math
P_A=\frac{\|A\delta_\theta\|_2^2}{\|Ag\|_2^2+\epsilon},
\qquad h(q,t)=\max(0,\log(q/t))^2.
```

冻结的 L1 case loss 为

```math
\mathcal L_{case}
=E_2+0.25E_{H^1}+0.02P_A
+h(q_{Am},1.02)+h(q_{Ac},1.02)
+2h(q_{Bc},1.02)+h(q_f,1.00),
```

总损失为 fit cases 平均。B clean projection 是显式 evaluator-only `B.forward(x_theta)`，不得藏入部署账本。
`P_A` 是低敏感方向软惩罚，不是 exact-null 投影证明。

### 5.6 可证明但有限的结构性质

若 `\|\delta_\theta\|_2\le\eta\|g\|_2` 且 `0\le\eta<1`，则

```math
g^T(g+\delta_\theta)\ge(1-\eta)\|g\|_2^2>0,
```

```math
\cos(g,g+\delta_\theta)\ge\frac{1-\eta}{1+\eta}.
```

当 `eta=0.05` 时，该保守余弦下界为 `0.90476`。在精确线性 A 与有效分母下，每步精确线搜索使当前 A
测量残差非增；当 `delta=0` 时，精确算术下恢复锚定 CGLS-24 轨迹。

这些性质**不保证** field relative-L2 改善、B 一致性、噪声稳定性、真实曲光线安全、未见 rig 泛化或网络可学性，
也不能称为突破性理论。

---

## 6. 与一级来源的碰撞边界

| 已有方向 | 一级来源 | 已覆盖的核心思想 | LGWO 不能声称什么 | 仍待证的窄差异 |
|---|---|---|---|---|
| DCDM | [PMLR 正式页](https://proceedings.mlr.press/v202/kaneda23a.html)、[arXiv](https://arxiv.org/abs/2205.10763) | 明确在**每一次迭代**使用深度网络对搜索方向作数据驱动改进，并以此加速共轭梯度类求解 | “首次学习 CG/共轭方向” | first-only、5% bound、BOST geometry、`delta=0` 同构与固定 `24/24` 是否带来更稳的尾部/成本权衡 |
| FCG-NO | [PMLR 正式页](https://proceedings.mlr.press/v235/rudikov24a.html) | 明确把**离散无关 neural operator** 学作 flexible conjugate gradient 的非线性预条件器，并讨论跨分辨率复用 | “首次把神经算子接入 Krylov”或“首次使用非线性神经预条件器” | 一个零算子调用的首方向 policy 是否能用更低网络/内存成本达到可比收益 |
| DeepONet | [Nature Machine Intelligence](https://www.nature.com/articles/s42256-021-00302-5)、[官方代码](https://github.com/lululxvi/deeponet) | branch/trunk 学习函数到函数映射 | “算子学习本身新颖” | 不直接输出场、而只输出受限首方向是否更容易守住物理一致性与未见 rig 尾部 |
| FNO | [arXiv](https://arxiv.org/abs/2010.08895)、[官方代码](https://github.com/neuraloperator/neuraloperator) | Fourier 层学习 PDE 解算子并跨离散迁移 | “首次跨网格学习算子” | 小模型 geometry-conditioned policy 在相同数据与端到端成本下是否比直接逆映射更稳 |
| iFNO | [PMLR 正式页](https://proceedings.mlr.press/v258/long25a.html)、[官方代码](https://github.com/BayesianAIGroup/iFNO) | 可逆 Fourier blocks 同时处理正逆问题 | “首次学习可逆/逆向算子” | 固定经典壳层是否能用更少训练数据换取更强一致性；必须实测，不可凭结构推断 |
| NeRIF | [AIP 正式页](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)、[arXiv HTML](https://arxiv.org/html/2409.14722v2) | 针对 voxel 离散误差、权重矩阵/内存随体素数膨胀、空间分辨率瓶颈、噪声鲁棒性和计算成本；网络逐坐标输出折射率 `n` 与梯度，沿每条 ray 随机采样 60--200 点，并用自动/数值微分一致性约束逐例优化隐式场 | “首次用神经场做 BOST”，或把 NeRIF 误写成跨例预训练的 fixed-budget solver policy | LGWO 是跨例摊销的 solver-direction policy，而非逐例隐式表示；质量、噪声、空间分辨率、每例优化时间和总成本必须公平比较 |
| NeDF | [arXiv HTML](https://arxiv.org/html/2409.19971)、[AIP DOI](https://doi.org/10.1063/5.0241191) | 直接针对严苛实验环境中光学窗口数量与布置空间受限所造成的 sparse views、limited viewing angles 和严重病态；用无预训练的隐式神经偏折场，并以高保真非线性 ray tracing 合成 LES 火焰 BOS 图像进行评价 | “首次处理稀疏视角、有限光学窗口或高保真非线性 ray tracing” | 在线性/局部线性壳层的固定预算策略在何种失配范围内仍有效，超过范围是否应失败而非硬迁移 |
| TDBOST | [ACM DOI](https://doi.org/10.1145/3809488) | X-Y-Z-T 张量分解、轻量网络、畸变修正和可微 ray tracing | “首次四维 BOST/时空压缩” | 固定 rig 多帧时能否摊销一个方向策略；在真实时序合同到位前不启动该主张 |

碰撞审计的结论是：宽泛新颖性已被否定。本稿只允许把“observable-only + geometry-conditioned + bounded
first-direction + fixed `24/24` + independent-B/tail gate”写成**可能有差异但尚未证实的组合**。没有检索到完全
相同组合不等于新颖性证明。

### 6.1 NeRIF 与 NeDF 对本课题的具体物理提醒

NeRIF 的出发点不是抽象的“用 MLP 替代体素”而已。其一级来源明确指出，经典 voxel formulation 会带来空间
离散误差、分辨率上限、噪声敏感、巨型投影/权重矩阵与计算成本问题。NeRIF 将空间坐标输入多头网络，同时输出
折射率与其梯度；沿反向光线使用随迭代增长的 60--200 个随机采样点近似连续积分，并把折射率输出的自动微分
梯度、网络直接输出的梯度、数值微分位移和实测位移放入一致性约束。它是在**每个待重建对象上优化隐式表示**，
不是从多个训练对象学得、随后每例只调用一次的 fixed-budget first-direction policy。因此未来比较不能只看最终
relative-L2，还要并列空间分辨率、噪声、逐例优化迭代、ray samples、墙钟、显存/内存和是否需要预训练。

NeDF 的物理问题更直接对应燃烧室和风洞：可用光学窗口少、布置空间受限，迫使 TBOS 使用 sparse views 与有限
视角，逆问题随之严重病态。其一级来源使用无预训练的隐式 neural deflection field、位置编码与分层采样，并用
LES 湍流预混火焰和高保真非线性 ray tracing 合成 BOS 图像进行量化评价。这意味着 LGWO 若只在 straight-ray
toy operator 上通过，仍不能声称解决 NeDF 所面对的现实 sparse-view/curved-ray 困境；必须在 E4 中冻结并报告
模型失配边界，必要时 fail closed，而不是把线性壳层强行迁移。

TDBOST 在本稿中只保留既有一级来源审计边界：X-Y-Z-T 张量分解、轻量网络、畸变修正与可微 ray tracing 的
四维逐序列优化。真实连续序列与组内数据合同到位前，不补充未经核验的结构、数字或性能主张。

---

## 7. 预注册研究问题与假设

### 7.1 主研究问题 RQ1

在冻结的 24 步壳层和 route-development split 上，full LGWO 是否同时通过场误差、A/B 一致性、尾部、失败率、
调用账本和统计门？

**联合零假设 `H0-primary`：** 至少一个主门失败。
**研究假设 `H1-primary`：** 所有主门同时通过。任何单一指标通过都不能拒绝联合零假设。

预注册主门：

1. 8 个 route geometry clusters 上 seed-averaged mean `G_field >= 5%`；
2. 三个固定 model seeds 的 mean `G_field` 均 `>=2%`；
3. 50,000 次 geometry-cluster bootstrap 的双侧 percentile 95% 区间下界 `>0%`；
4. 至少 `7/8` clusters 的 seed/family-averaged `G_field>0`；
5. 每个 family 的 seed-averaged mean `G_field>=0`；
6. mean H1 gain `>=3%`；
7. mean `q_Am`、`q_Ac`、`q_Bc` 均 `<=1.05`，每 cluster 三者均 `<=1.10`；
8. `G_field<-1%` 的 case-seed harm rate `<=5%`，worst case-seed gain `>=-5%`；
9. fallback、breakdown、non-finite 均为 0；
10. maximum residual increase `<=1e-12`，measurement-direction orthogonality defect `<=1e-10`；
11. candidate 与 baseline 均精确支付 `24A/24A^T`；
12. full mean `G_field` 比 train-fitted fixed direction 至少高 `1.0` percentage point。

### 7.2 次研究问题 RQ2：几何条件是否真正有用

**`H0-geometry`：** full 与 no-geometry/geometry-shuffle 的差异不稳定，或 full 不优于 fixed direction。
**`H1-geometry`：** full 在同一 sealed route 上稳定优于 fixed direction 至少 1 percentage point，且 no-geometry、
geometry-only mismatch 与跨 rig 结果共同支持几何条件的可解释作用。

这里不把“错配后输出变化更大”定义为成功。必须同时观察场误差和 A/B 一致性，避免网络只对 metadata 敏感。

### 7.3 次研究问题 RQ3：first-only 是否有必要

**`H0-first`：** first-only 不优于 matched warm-start、every-k、every-step/DCDM-style 或 FCG-NO-style 方法。
**`H1-first`：** 在同 `A/A^T` 与端到端成本报告下，first-only 在 field-tail-consistency 三者中形成不可由简单
模型解释的 Pareto 优势。

RQ3 属于 L2，L1 通过只授权开展比较，不自动支持该假设。

### 7.4 机制问题 RQ4：修正是否进入低可观测方向

**`H0-mechanism`：** 修正只是 `g` 的缩放、固定场先验或随机方向；near-null 指标与收益无稳定关系。
**`H1-mechanism`：** evaluator-only SVD 中 `P_null delta`、`A delta`、方向角和 field gain 呈预注册的一致模式，且
null-erased/row-erased 消融支持该解释。

即使 `H1-mechanism` 得到支持，也只能称为离散 toy/benchmark operator 上的 low-observability tendency，不能称为
真实光学 null-space 学习。

### 7.5 外部问题 RQ5：能否迁移到真实 BOST

**`H0-real`：** synthetic 优势在独立 renderer、真实 callable、未见 rig/session 或独立相机验证中消失。
**`H1-real`：** 冻结模型/规则在预注册真实条件下保持 field proxy 或独立视角收益、守住一致性与尾部，并在成本
上有现实意义。

RQ5 在真实接口、标定、数据权限和主指标未由师兄确认前保持 `CLOSED`。

---

## 8. 数据、split 与泄漏防火墙

### 8.1 已冻结 L1 设计

下表是计划合同，不是已经生成的科学数据：

| partition | A geometry | B geometry | clusters x families | cases | 允许的决定 | 当前状态 |
|---|---|---|---:|---:|---|---|
| fit | `train` | 独立 `train` seeds | `8 x 3` | 24 | 梯度更新、normalization、经验 envelope | `NOT MATERIALIZED` |
| early-stop | disjoint `train` | 独立 `train` seeds | `2 x 3` | 6 | 只选 eligible epoch | `NOT MATERIALIZED` |
| route-development | non-opened `development` | `ood` | `8 x 3` | 24 | 所有方法冻结后一次 pass/fail | `SEALED / 0 cases` |
| fresh OOD | `ood` 或实验室数据 | closed | `[另行预注册]` | 0 | L1 禁止打开 | `CLOSED` |

三个解析 families 固定为 `smooth_no_interface`、`single_interface`、`two_interface`。family 可用于分层汇总，不能
进入模型。每个 cluster 内三种 family 共享 A geometry，但 cluster 才是统计独立单位。

### 8.2 永久排除数据

先前 JACRU/O1 打开的 geometry seeds、6 个 truth-informed oracle cases、任何调参预览以及由它们产生的派生统计，
都不得进入 fit、early-stop、route 或 fresh。它们只能在引言中作为“为什么值得做一次严格实验”的动机，并明确
标注 opened、toy、truth-informed 和 non-scientific-for-L1。

### 8.3 L2、公开数据与真实 OERF split

L2 必须在打开前另存不可变 manifest，并至少按以下层级隔离：

| 层级 | 必须整体留出 | 禁止做法 | 待填合同 |
|---|---|---|---|
| synthetic/CFD | geometry cluster、renderer、phantom/flow realization | 随机拆 voxel、ray 或重叠 crop | 数据集、版本、许可、hash：`[待填]` |
| 静态真实 BOST | rig、标定版本、实验 session、工况 | 将同一 session 相邻帧随机分 train/test | 装置、相机、独立验证视角：`[待师兄确认]` |
| 高速时序 | 完整时间 block 与 run | 按单帧随机切分 | frame rate、block 长度、漂移：`[待确认]` |
| 跨装置 | 整套 rig/lab | 用测试装置做 normalization 或 envelope | 外部装置与复现者：`[待定]` |

真实 raw data、标定和权重默认私有本地，不上传公开仓库。论文只报告获授权的匿名汇总、必要示意和可公开代码。

### 8.4 B 组的证据等级

1. `synthetic B-clean`：同一 truth、同一 renderer 家族、独立 ray bundle；只证明跨射线一致性。
2. `synthetic B-renderer-shift`：同一 truth、不同数值 renderer；可测试部分模型失配。
3. `real B-camera`：真实独立相机/采集；不参与训练时证据更强。
4. `cross-lab B`：另一装置或实验室；才接近外部可转移性。

报告中不得把第 1 级写成第 3 或第 4 级。

---

## 9. 预算匹配基线

### 9.1 两套公平比较必须同时存在

**协议 A：固定物理算子预算。** 候选与迭代基线每例最多 `24A/24A^T`；额外 evaluator 调用单列。
**协议 B：固定端到端资源。** 在同一硬件、dtype、batch、数据与计时边界下比较 wall time、peak memory、训练时间
和摊销成本。任何方法只能在协议 A 或 B 中满足公平，不得用一个协议的优势掩盖另一个协议的劣势。

### 9.2 必须实现或诚实标注不可比的基线

| 基线 | 角色 | 固定预算要求 | 最低调参公平性 | 当前状态 |
|---|---|---|---|---|
| zero-start CGLS-24 | 主经典基线 | `24A/24A^T` | 无学习参数；同 support/dtype | `ENGINEERING SHELL ONLY` |
| anchored shell, `delta=0` | 同构检查 | `24/24` | 与 CGLS 逐项 parity | `ENGINEERING ONLY` |
| `1.05g` scaling control | 排除精确线搜索下的纯缩放伪收益 | `24/24` | 不训练 | `PLANNED` |
| random equal-norm direction | 排除任意扰动收益 | `24/24` | 同 eta、固定 seeds | `PLANNED` |
| train-fitted fixed direction | 排除平均场先验 | `24/24` | 同 loss、epochs、bound | `PLANNED` |
| linear geometry-conditioned correction | 排除小网络只是线性回归 | `24/24` | 同输入、bound、split | `L2 PLANNED` |
| learned warm-start + CGLS | 对照学习初场 | 总 `A/A^T` 不超预算，另报 proposal 成本 | 同参数档/训练数据 | `L2 PLANNED` |
| DCDM-style every-step direction | 最强方向学习近邻 | `24/24` 或同墙钟双表 | 官方思想、合理宽度、相同 seeds/split | `L2 PLANNED` |
| FCG-NO-style preconditioner | 最强神经 Krylov 近邻 | 同迭代/调用或同精度成本 | 同网格与跨分辨率设置 | `L2 PLANNED` |
| DeepONet direct inverse | 直接算子基线 | 推理 A 调用可为 0；评价 A/B 单列 | 同训练样本、合理 branch/trunk 容量 | `L2 PLANNED` |
| FNO direct inverse | 频域直接算子基线 | 同上 | 官方实现、合理 modes/width 搜索 | `L2 PLANNED` |
| iFNO inverse | 可逆算子基线 | 同上 | 潜变量/采样成本完整报告 | `L2 PLANNED` |
| NeRIF | BOST 隐式场逐例优化 | 分开报告每例优化与预处理 | 按论文/官方实现可复现配置 | `REAL/CFD PLANNED` |
| NeDF | sparse-view/非线性 ray 近邻 | ray/sample 与 wall 完整报告 | 相同相机/射线/停止准则 | `REAL/CFD PLANNED` |
| TDBOST | 4D 时序近邻 | 仅在真实连续序列合同到位后比较 | 同时间窗、分辨率与显存 | `CLOSED` |

### 9.3 调参预算

1. 所有学习方法共享 fit/early-stop/route/fresh split；route 不参与超参数选择。
2. 每个基线的搜索空间、试验次数、总 GPU/CPU 小时和失败运行都登记。
3. 不只运行 DeepONet/FNO 的默认弱配置；至少提供“小模型成本匹配”和“作者推荐性能配置”两个档，资源不足则明确
   标记 `NOT EXECUTED`，不能把缺失当失败。
4. LGWO 不得拥有更多训练样本、更多 seed、更多 route 反馈或隐藏的 operator cache。
5. 直接逆网络的低推理调用是其真实优势；LGWO 只能用一致性、尾部或数据效率证明自身价值，不能改写计费方式。
6. 逐例方法分开报告一次性预训练、每例优化与达到停止准则的时间。

---

## 10. 指标与记录合同

### 10.1 重建质量

| 指标 | 定义/统计 | 角色 |
|---|---|---|
| field relative-L2 | 第 4 节定义，support 内计算 | **唯一主终点** |
| H1 relative error | `||grad(xhat-x*)||/(||grad x*||+eps)` | 梯度/界面结构次终点 |
| slice/line-profile error | 预注册位置或物理特征对齐后计算 | 可解释性，不替代主终点 |
| structure metrics | 仅在注册定义与物理意义后加入 | 探索性，不能事后挑最好看指标 |

### 10.2 A/B 一致性

以同 case 的 CGLS-24 为分母：

```math
q_{Am}=\frac{\|A\hat x-y_{A,meas}\|_2}
             {\|A\hat x_b-y_{A,meas}\|_2+\epsilon},
```

```math
q_{Ac}=\frac{\|A\hat x-y_{A,clean}\|_2}
             {\|A\hat x_b-y_{A,clean}\|_2+\epsilon},
\qquad
q_{Bc}=\frac{\|B\hat x-y_{B,clean}\|_2}
             {\|B\hat x_b-y_{B,clean}\|_2+\epsilon}.
```

真实数据没有 clean target 时报告 `q_Am` 与独立实测 `q_Bm`，不得用经过 candidate 调整的 denoised target 替代
clean。A residual 下降但 B residual 上升通常意味着对观测几何过拟合；二者都必须与场指标或外部参照联合解释。

### 10.3 逐 rig 尾部与失败

每个 rig/geometry cluster 必须报告：`n`、mean、median、P90、P95、worst、`G_field<-1%` harm rate、fallback rate、
breakdown、non-finite 与 missing runs。总体平均不能替代逐 rig 表。

### 10.4 成本

| 成本项 | 记录方式 |
|---|---|
| `A/A^T` | solver API ledger，按 case-equivalent 与 batched invocation 双报 |
| B evaluator | 单列 `B-F/B-AT`，不得并入部署 `24/24` |
| ray/sample evaluations | 真实 curved/straight callable 能提供时记录 |
| network calls/parameters | 每例次数、参数量、checkpoint 大小 |
| training compute | optimizer steps、forward/backward 等价成本、总 CPU/GPU 小时 |
| inference wall | 同硬件、同 dtype、固定 warm-up/重复协议；报告 median 与 P90 |
| end-to-end wall | 含数据搬运、几何预处理、重正交、fallback 与 solver |
| peak memory | CPU RSS、GPU/MPS peak 分开记录 |
| amortized cost | `training cost/N + per-case inference`；预注册 `N={1,10,100,1000}` |

不同设备结果不能直接排成单一速度名次。若某方法只能在 GPU 运行而另一方法在 CPU，需分别给同设备子表与现实部署
表，并说明缺失原因。

---

## 11. 实验矩阵

| 编号 | 阶段 | 数据/物理 | 方法 | 主要判定 | 开放条件 | 当前状态 |
|---|---|---|---|---|---|---|
| E0 | 工程门 | 永久排除 seeds、小矩阵/toy | `delta=0`、training/deploy twin | parity、gradient、ledger 与证据链；**不把 resume/crash-safe 视为完成** | 无 | `PARTIAL: 2 EVIDENCE P0 CLOSED / 1 CROSS-EPOCH TRANSACTION P0 OPEN; 0 SCIENTIFIC CASES` |
| E1 | L1 fit/early | 24 fit + 6 early synthetic cases | full 三 seeds | 是否存在 eligible checkpoint | advisor-interface first；师兄确认路线值得真实 fit 后，才最小实现跨 epoch transaction P0 | `BLOCKED; 0 SCIENTIFIC OPTIMIZER STEPS` |
| E2 | L1 sealed route | 8 clusters x 3 families，A-development/B-ood | full + L1 controls | RQ1 联合门 | 所有 arms 冻结并哈希 | `SEALED` |
| E3 | L2 matched benchmark | 更多 geometry/noise/support/resolution | DCDM、FCG-NO、DeepONet、FNO、iFNO 等 | RQ2/RQ3、公平 Pareto | E2 全门通过；另行注册 | `CLOSED` |
| E4 | renderer/model shift | straight、curved、标定偏差、位移误差 | E3 冻结模型 | 失配边界和 fail-closed 行为 | 真实或公开第二 renderer 合同 | `CLOSED` |
| E5 | 静态真实 OERF | 独立 rig/session/相机 | 组内强基线 + 冻结候选 | RQ5、现实成本 | 师兄确认 callable、数据与发布权限 | `WAITING FOR CONTRACT` |
| E6 | 4D/高速序列 | 完整时间块 | TDBOST + 时序扩展 | 是否有跨帧摊销价值 | 连续数据与时间 split 到位 | `NOT AUTHORIZED` |

### 11.1 L2 因子网格（打开前必须定值）

| 因子 | 预注册水平 | 为什么重要 | 当前值 |
|---|---|---|---|
| camera count/coverage | `[待填]` | 有限视角强度 | `CLOSED` |
| azimuth/elevation/jitter | `[待填]` | unseen rig | `CLOSED` |
| detector/ray density | `[待填]` | 分辨率与离散迁移 | `CLOSED` |
| noise type/SNR | `[待填]` | 位移测量误差 | `CLOSED` |
| calibration perturbation | `[待填]` | 现实外参/内参偏差 | `CLOSED` |
| support mismatch | `[待填]` | 视场与边界风险 | `CLOSED` |
| field family | analytic、CFD/LES、real `[待确认]` | 从 toy 到反应流 | `PARTIAL DESIGN ONLY` |
| ray model | straight、curved、native residual `[待确认]` | forward mismatch | `WAITING FOR CALLABLE` |

任何水平不得在看过 fresh 结果后改成更容易通过的范围。

---

## 12. 消融计划

### 12.1 L1 强制消融

| 消融 | 回答的问题 | 失败解释 |
|---|---|---|
| `delta=0` | learned shell 是否精确恢复基线 | 不一致则实现失败，禁止训练 |
| `1.05g` | 收益是否只是方向缩放 | 若等价控制也“改善”，说明指标/实现有问题 |
| fixed direction | 网络是否只记住平均体素先验 | full 不多 1 pp，则 geometry-conditioned 主张失败 |
| g-only | raw displacement/geometry token 是否需要 | full 无优势则输入设计缺乏证据 |
| no raw `y` | raw detector pattern 是否贡献 | 仅作为机制解释，不替代主门 |
| no geometry | 几何编码是否贡献 | full 无稳定优势则不能宣称 geometry-conditioned 有效 |
| joint camera permutation | 模型是否对 camera 顺序正确不变 | 相对差大于 `1e-6` 为实现门失败 |
| geometry-only mismatch | 模型是否错误依赖 metadata | 只报告场/一致性变化，不设置“越敏感越好” |

### 12.2 L2 机制与结构消融

1. no heldout-B loss；
2. no `A delta` penalty；
3. unbounded correction；
4. linear geometry-conditioned correction；
5. first-only、every-k、every-step/DCDM-style；
6. warm-start 与 FCG-NO-style preconditioner；
7. empirical envelope 与另行注册的 conformal/risk gate；
8. evaluator-only exact-null、row-only、random equal-norm oracle；
9. full correction、null-erased correction、row-erased correction；
10. 另行预注册的 `eta in {0,0.01,0.025,0.05,0.10}` 曲线。

第 8/9 项的 SVD 或 truth 只能用于 evaluator，不得进入部署 API、loss、epoch 选择或 route gate。L1 的 `eta=0.05`
不可因 L2 半径曲线倒推修改。

---

## 13. Planned figures：每张图必须回答什么

所有结果图当前状态均为 `PLANNED / NO DATA`。禁止用随机数、手绘趋势、opened oracle 或工程测试填充科学图。

| 图号 | 计划内容 | 判定问题 | 必须显示 | 不能隐藏 |
|---|---|---|---|---|
| Fig. 1 | BOST A/B 几何与 LGWO-24 流程图 | 方法究竟读取什么、调用几次 A/AT、哪里可回退 | input firewall、24/24 ledger、B evaluator 分离 | truth/B 输入不得画进 proposal |
| Fig. 2 | split 与信息防火墙 | rig/session/cluster 是否真正隔离 | opened-excluded、fit、early、route、fresh | 不得只画 case 随机切分 |
| Fig. 3 | 每 cluster 的 `G_field` forest/paired plot | 平均收益是否跨 rig 稳定 | 三 seeds、三 families、95% cluster bootstrap CI | 不得只画最好 seed 或总体均值 |
| Fig. 4 | field error 对 A/B residual Pareto | 场改善是否以数据一致性为代价 | CGLS、full、fixed、DCDM/FCG/direct baselines | 不得删掉失败或 fallback cases |
| Fig. 5 | 逐 rig 尾部与 harm 分布 | 是否存在灾难性尾部 | median、P90/P95、worst、harm/fallback | violin/boxplot 不能省略原始 cluster 点 |
| Fig. 6 | `delta` 机制诊断 | 修正是缩放、固定先验还是低可观测成分 | angle、`||A delta||/||Ag||`、null fraction、gain | SVD 只标 evaluator-only |
| Fig. 7 | 消融效应图 | 哪个组件产生不可替代贡献 | full 与所有预注册消融的 paired differences | 不得按结果选择消融 |
| Fig. 8 | 质量-成本 Pareto | 固定调用优势在 wall/memory 下是否仍成立 | calls、wall、memory、parameters、training amortization | 不得只报 network forward 时间 |
| Fig. 9 | renderer/rig mismatch 热图 | 优势在哪些现实失配轴失效 | straight-curved、calibration、noise、support 网格 | 失败格不能空白或裁掉 |
| Fig. 10 | 真实 BOST slices/line profiles/独立相机 residual | 三维结构改善是否有现实外部证据 | 同色标、同切片、独立视角、误差/不确定性 | 无 truth 时不得画“ground-truth error” |

图注必须写样本单位、split、模型 seeds、error bar 计算、fallback 处理、硬件/dtype 和是否使用 clean truth。

---

## 14. Planned tables：每张表必须回答什么

| 表号 | 计划内容 | 判定问题 | 当前状态 |
|---|---|---|---|
| Table 1 | 数据/rig/renderer 合同 | 读者能否复建每个 split，是否存在泄漏 | `PLANNED` |
| Table 2 | 方法、参数与预算 | 比较是否同 `A/A^T`、同数据、同调参机会 | `PLANNED` |
| Table 3 | L1 主终点与联合门 | RQ1 是否严格通过 | `NOT RUN` |
| Table 4 | 每 rig/family/seed 尾部 | 均值是否掩盖局部失败 | `NOT RUN` |
| Table 5 | A/B 一致性与 failure counts | 改善是否以重投影或 fallback 为代价 | `NOT RUN` |
| Table 6 | 消融与机制 | geometry、first-only、B loss、bound 是否必要 | `CLOSED` |
| Table 7 | wall/memory/training amortization | 方法是否有现实计算价值 | `NOT RUN` |
| Table 8 | 真实 BOST 外部验证 | 是否超出 synthetic 封闭体系 | `WAITING FOR DATA CONTRACT` |

---

## 15. 统计规则

### 15.1 独立单位

1. L1 的独立统计单位是 geometry cluster，不是 72 个 case-seed observations。
2. 先在 cluster 内平均三个 families 与三个固定 model seeds，再做跨 cluster 统计。
3. 真实时序数据以 rig/session/time block 为单位；相邻帧不是独立重复。
4. 多相机射线、voxel 和 crop 都不是独立样本。

### 15.2 固定 seed 与汇总

模型 seeds 固定为 `[1630052265, 102254037, 2925587]`。全部报告，不选 best seed。若任一 primary seed 没有 eligible
checkpoint，L1 primary interface 失败，不能删掉该 seed 或追加“更稳定”的 seed。

### 15.3 区间与符号门

1. 主终点在 8 个 cluster 统计量上做 50,000 次有放回 cluster bootstrap；报告双侧 percentile 95% CI。
2. bootstrap 随机种子、实现版本与原始 cluster 表必须冻结。
3. `7/8` 正 cluster 在独立对称符号零假设下的精确单侧概率为 `9/256=0.03515625`；这里只作为预注册符号门，
   不把 8 个小样本写成强泛化证明。
4. 同时报 effect size、CI、逐 cluster 原始点和最坏值；不能只报 p 值。

### 15.4 多重比较

主确认性比较只有 `full LGWO vs CGLS-24`，且使用第 7.1 节联合门。其他基线、消融与分组属于机制或探索性
分析；若对多个比较给出确认性 p 值，必须预先列出 family 并使用 Holm 校正，同时保留未校正值与效应量。不得在
结果后把最有利的次终点升级为主终点。

### 15.5 缺失、失败与异常值

1. 不删除 finite 的极端样本；它们进入 worst/harm/tail。
2. non-finite、breakdown、超时与显存不足各自计数，不能统一写成 missing。
3. primary 运行缺失默认使联合门失败；只有可证明的外部中断可按同 checkpoint 精确恢复，并保留日志。
4. 不因结果差而重跑 seed。硬件故障重跑必须保留原失败记录、原因、时间和新 run id。
5. 不使用事后 winsorization；若未来需要 robust summary，必须另行注册并与原始结果并报。

### 15.6 结果可见性与冻结顺序

```text
protocol/config/hash
  -> fit
  -> early-stop checkpoint freeze
  -> all baselines and ablations freeze
  -> one-time route materialization
  -> route report/hash
  -> evaluator-only mechanism analysis
  -> new preregistration for L2/fresh
```

任何偏离都写入 deviation log；不得覆盖旧配置或旧结果。

---

## 16. 失败报告协议

### 16.1 实现门失败

若 parity、gradient、permutation、ledger、finite、resume 或输入防火墙失败：

> `IMPLEMENTATION_GATE_FAILED`。没有科学训练或评价；本次失败不能用于支持或否定算法假设。

修复必须发生在科学 partition materialization 之前，并记录源 commit、原因和新 hash。

### 16.2 Fit/early-stop 失败

若 30 epochs 内任一 primary seed 无 eligible checkpoint，或出现 fallback/non-finite/breakdown：

> 预注册的 LGWO-A24-L1 学习接口未形成可安全打开 route 的候选；optimizer 运行结果为负面工程/学习证据，
> route 保持密封。

禁止换 seed、放宽 residual 门、扩大模型或用消融替代 full。

### 16.3 Route 失败

若联合门任一项失败：

> 预注册的三种子、双射线束 synthetic pilot 未通过 route-development 门。opened oracle headroom 没有转化为
> 当前允许输入下的稳定模型证据。

报告全部指标、失败 cluster、tail、A/B trade-off 和成本。不得打开 fresh，也不得把局部正均值写成成功。

### 16.4 L2/真实失败

L1 通过但 L2 或真实数据失败时，应明确区分：

- synthetic overfit；
- geometry/renderer shift 失败；
- callable/导数不可靠；
- 独立相机不一致；
- 真实成本不占优；
- 证据不足而非“趋势上成功”。

负结果仍可形成有价值的边界论文，但不能保持原成功标题和摘要措辞。

### 16.5 偏离注册

每次偏离填写：

| 字段 | 内容 |
|---|---|
| deviation id | `[待生成]` |
| 发现时间 | `[UTC/local 待填]` |
| 是否已看目标结果 | `YES/NO` |
| 原规则/hash | `[待填]` |
| 修改内容与原因 | `[待填]` |
| 影响哪些 claim | `[待填]` |
| 处理 | `confirmatory 保留 / exploratory 降级 / phase 重注册` |

---

## 17. 审稿人攻击清单

投稿前，作者必须能逐项给出证据或在 limitations 中承认失败。

### 17.1 新颖性

- [ ] 是否把 DCDM/FCG-NO 已有的学习方向/预条件思想改名包装？
- [ ] 是否只凭“未找到完全相同组合”宣称 first/novel？
- [ ] 是否错误使用 learned null-space、self-supervised、provably safe 或 generalization？
- [ ] first-only、bound、geometry、B consistency 与 fixed budget 是否各有必要性证据？

### 17.2 物理可信度

- [ ] A 是 straight、curved 还是线性化 Jacobian？units、axis order、spacing、wavelength 是否完整？
- [ ] synthetic A/B 是否共享 renderer，是否被误称独立物理？
- [ ] hard mask、termination、occupancy pruning、自适应采样是否破坏 JVP/VJP？
- [ ] 无 volumetric truth 的真实实验用了什么外部证据？
- [ ] support、标定和位移提取是否把 truth 信息泄漏给模型？

### 17.3 数据泄漏

- [ ] split 是否按完整 geometry cluster、rig、session、时间块？
- [ ] normalization、calibration envelope、epoch 和超参数是否只用允许 partition？
- [ ] model API 是否能间接读取 case id、seed、digest、family 或 evaluator？
- [ ] opened O1/JACRU seeds 是否永久排除？
- [ ] route 是否只打开一次，失败后是否另行注册而非原地重训？

### 17.4 基线公平

- [ ] DCDM、FCG-NO、DeepONet、FNO、iFNO 是否来自合理实现与调参预算？
- [ ] 是否同时有固定 operator-call 与固定 wall/memory 两套比较？
- [ ] direct inverse 的零 solver-call 优势是否诚实保留？
- [ ] NeRIF/NeDF/TDBOST 的逐例/逐序列优化成本是否完整？
- [ ] 训练成本、失败 trial 和 amortization 是否报告？

### 17.5 统计与可视化

- [ ] 是否把 case-seed 当独立样本扩大 n？
- [ ] 是否完整显示三个 seeds、每个 rig、worst/harm/fallback？
- [ ] error bar、bootstrap 单位与 missing 处理是否写在图注？
- [ ] 是否只展示好看切片或改变色标？
- [ ] 是否因多重比较挑出一个显著次指标？

### 17.6 可复现性

- [ ] config、manifest、source、environment、checkpoint 和结果 hash 是否绑定？
- [ ] `A/A^T` ledger 是否能由独立 validator 重算？
- [ ] checkpoint/resume 是否不会改变 optimizer trajectory？
- [ ] 私有数据不能公开时，是否提供最小匿名 callable/schema 和公开替代 benchmark？
- [ ] 所有 claim 是否能追溯到具体表格、图或失败日志？

---

## 18. 结果填充区：当前一律不得提前填数

### 18.1 当前证据登记

| 项目 | 当前值 | 科学解释 |
|---|---:|---|
| 合格 scientific cases | **0** | 无可用于算法效果的样本 |
| optimizer steps | **0** | 模型尚未训练 |
| route-development cases materialized | **0** | route 密封 |
| fresh/OOD cases | **0** | 未授权打开 |
| 真实 OERF cases | **0** | 接口/数据合同未取得 |
| matched-baseline results | **0** | 不存在优胜方法 |
| breakthrough | **NO** | 无突破 |

### 18.1A Post-open addendum：PSU-C1 解析简单对照筛查

这一附录是 `non-L1 / non-neural / public-geometry analytic rehearsal / post-open`，不填入 18.2 的 L1 主结果表，也不改变
`scientific cases = 0`、`optimizer steps = 0` 和 `route sealed`。审计修订 1.1 只补 held-out-B 全覆盖、oracle evaluator
成本、fit setup 成本、固定可视化样本、公开几何身份和失败留痕；没有改变方法、split、随机种子、5% 半径或门槛。

| PSU-C1 项目 | IID | family-OOD | 冻结门 | 判决 |
|---|---:|---:|---:|---|
| truth-angular oracle mean field gain | +1.1288% | +1.2465% | 每个主分区 >=5% | **FAIL** |
| linear observable mean field gain | +1.0302% | +1.0542% | 每个主分区 >=2% | **FAIL** |
| fixed direction mean field gain | +0.6987% | +0.6149% | descriptive | 小信号 |
| inverse-Sobolev5 mean field gain | +51.5234% | +45.0628% | 强对照，不事后改门 | 机制线索 |
| inverse-Sobolev5 A-measured ratio | 1.6752 | 4.5303 | 不得只报 field | 明显冲突 |
| inverse-Sobolev5 held-out-B ratio | 0.3619 | 0.5073 | proxy only | 不等于真实体真值 |

独立 validator 复算 1,296 行指标和 54 个聚合单元后返回 `VALID`，最终状态为
`NO_GO_FIRST_DIRECTION_HEADROOM_ABSENT_POSTOPEN`。它只关闭“5% 有界 first-direction 是下一优先训练对象”这一窄假设，
不关闭 learned regularization、learned stopping、multi-step trajectory、curved-ray correction、NeRIF 或 4D TDBOST。
完整解释见 [PSU-C1 NO-GO 报告](lgwo_a24_psu_c1_simple_controls_no_go_2026-07-20.md)。

### 18.1B Post-open addendum：R0 半收敛轨迹与 field/gradient 冲突

R0 同样不属于 L1 主结果。它在公开 PSU 几何、解析反应形态和合成噪声上保存 168 条 `k=1...24` 的零修正、
fully reorthogonalized CGLS 轨迹，共 4,032 行。每例算法成本严格为 `24F/24A^T`；主协议和独立 validator 均返回
结构有效，但最终状态为 `VALID_NO_GO_STOPPING_HEADROOM_OR_DIVERSITY_ABSENT_POSTOPEN`。

| R0 项目 | test-IID | family-OOD | 解释 |
|---|---:|---:|---|
| validation-global `k` | 24 | 24 | validation mean field curve 末步最优 |
| truth-field oracle mean `k` | 24.00 | 24.00 | 48/48 cases 全选末步 |
| truth-field oracle unique `k` / IQR | 1 / 0 | 1 / 0 | 无 instance-specific stopping label |
| truth-field oracle field gain vs `k=24` | 0.0000% | 0.0000% | 5% headroom 门失败 |
| discrepancy mean `k` | 23.75 | 23.88 | 近似退化为末步 |
| discrepancy field gain vs `k=24` | -0.0097% | -0.0049% | 未产生可部署收益 |
| field improvement `k1→k24` | +6.405% | +7.065% | field 一路改善 |
| gradient change `k1→k24` | **恶化 30.961%** | **恶化 23.803%** | gradient 一路恶化 |
| mean field/gradient Pareto checkpoints | 24/24 | 24/24 | 选旧 checkpoint 不能消除冲突 |

独立 post-open 机制脚本重新读取封存 CSV，复算 split/checkpoint mean、standard deviation、逐 case 单调性、oracle
和 discrepancy 摘要；独立 validator 完成 1,098 项检查并返回 `VALID`。IID 和 family-OOD 的 24/24 条轨迹均为
field nonincreasing、gradient nondecreasing。front top-10% F1 同时上升，说明 front localization 与全局 gradient
relative-L2 不能互换，后续必须并列报告。

该附录关闭的是**在当前 24 步路径上训练 instance-specific stopping policy**；R1 observable stopping screen 未获授权，
没有训练模型。它只开放一个更窄的下一问题：显式 H1/Sobolev、Tikhonov、TV/Huber、hybrid projection 或
edge-superiorization 能否在相同调用预算下产生支配旧 field/gradient Pareto 路径的新 checkpoint。只有新路径出现
truth headroom、observable 线性规则保留信号且强经典基线无法解释时，才允许测试有界正则化控制器。

完整结果见 [R0 早停 NO-GO 与正则化冲突](lgwo_a24_r0_semiconvergence_no_go_2026-07-20.md)。

### 18.2 L1 主结果表模板

只有在结果 bundle、config、source commit 与独立验证 hash 完整后填写。

| method | seeds | clusters | field rel-L2 | `G_field` | 95% cluster CI | H1 gain | qAm | qAc | qBc | harm | worst | fallback | A/AT | wall | memory | gate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| CGLS-24 | `NOT RUN` | 0 | `NOT RUN` | reference | `N/A` | `NOT RUN` | 1.0 ref | 1.0 ref | 1.0 ref | `NOT RUN` | `NOT RUN` | `NOT RUN` | `24/24 planned` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| anchored `delta=0` | `NOT RUN` | 0 | `NOT RUN` | `NOT RUN` | `N/A` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `24/24 planned` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| fixed direction | `NOT RUN` | 0 | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `24/24 planned` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| full LGWO | `NOT RUN` | 0 | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `24/24 planned` | `NOT RUN` | `NOT RUN` | `NOT RUN` |

### 18.3 逐 rig/cluster 尾部模板

| rig/cluster | n families | n seeds | mean gain | median | P90 loss | P95 loss | worst gain | harm >1% | fallback | A/B violation | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `[id]` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `[fill after seal]` |

### 18.4 Matched-baseline 模板

| method | data budget | tuning trials | params | train compute | A/AT inference | B eval | end-to-end wall | peak memory | field rel-L2 | rig P95 | consistency gate | status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| DCDM-style | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `NOT RUN` |
| FCG-NO-style | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `NOT RUN` |
| DeepONet | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `NOT RUN` |
| FNO | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `NOT RUN` |
| iFNO | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `CLOSED` | `NOT RUN` |
| LGWO | `CLOSED` | `CLOSED` | 2,729 planned | `0 now` | `24/24 planned` | `CLOSED` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NO CLAIM` |

### 18.5 真实 OERF 模板

| dataset/session | rig | train/selection/test isolation | physical quantity/unit | baseline | independent evidence | field/proxy result | A/B result | wall/memory | failure | authorization |
|---|---|---|---|---|---|---|---|---|---|---|
| `[待师兄确认]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` | `[待填]` |

### 18.6 主张账本模板

| claim id | 拟写主张 | 所需证据 | 实际证据位置 | 状态 | 允许措辞 |
|---|---|---|---|---|---|
| C0 | 5% bounded first-direction 值得进入神经 fit | truth-oracle headroom + simple controls | PSU-C1 valid post-open rehearsal | `NO-GO` | 只能说该窄假设在当前 benchmark 上未获授权；不得外推全部算子学习 |
| C0-R0 | 当前 24 步 CGLS 轨迹值得训练 instance-specific stopping | truth-field oracle headroom + checkpoint diversity + observable proxy | R0 valid post-open trajectory | `NO-GO` | 只能说当前轨迹没有可学停止标签；下一步应改变 regularization path |
| C1 | L1 出现稳定 synthetic signal | E2 全部联合门 | `NOT RUN` | `NO-GO` | 只能说“待检验” |
| C2 | geometry conditioning 有必要 | 消融 + unseen rig | `NOT RUN` | `NO-GO` | 不得说有效 |
| C3 | first-only 优于强近邻 | L2 matched baselines | `NOT RUN` | `NO-GO` | 不得说优于 DCDM/FCG-NO |
| C4 | 真实 BOST 有价值 | 独立相机/装置验证 | `NOT RUN` | `NO-GO` | 不得说真实有效 |
| C5 | 可泛化/可转移 | fresh + cross-rig/lab | `NOT RUN` | `NO-GO` | 不得说泛化 |
| C6 | 算法突破 | 多层证据、再检索、外部审计 | `NONE` | `REJECTED NOW` | **no breakthrough** |

---

## 19. 讨论与局限性写作槽

结果产生后，讨论必须按以下顺序写，而不是从最有利数字出发：

1. 主联合门是否通过；若未通过，首先说明失败项。
2. field 与 A/B consistency 是否同向；若冲突，解释物理含义。
3. 收益是否跨 rig/family/seed，尾部是否可接受。
4. fixed direction、linear、DCDM/FCG-NO/direct inverse 能否解释结果。
5. `delta` 机制证据支持哪种解释，哪些仍是相关而非因果。
6. straight/curved、标定、support 和噪声失配边界。
7. wall/memory/training amortization 下是否仍有现实价值。
8. synthetic B 与真实独立物理之间的证据差距。
9. 对 OERF 装置的适用范围与明确禁用条件。

PSU-C1 已经给出一个必须保留的冲突案例：inverse-Sobolev 在六个描述性 split 上得到约 42%--53% 的 analytic
field gain，并明显改善 held-out-B residual；但 active measured residual ratio 在多数 split 大于 1，family-OOD 达到
4.5303。最合理的当前推断是 regularization/semiconvergence 比 bounded first direction 更控制误差，但不能据此宣布
Sobolev 胜出。下一候选应学习受限的 regularization strength、stop/continue 或少量谱基权重，并保留 deterministic
fallback；该候选仍需新的冻结 split、强 fixed/discrepancy/TV/FCG-NO 对照和真实接口确认。

R0 随后进一步否定了“在现有 CGLS 轨迹上学停止点”这一分支：两个主分区的 truth-field oracle 全部选择 `k=24`，
而 noise/joint-OOD 中更有多样性的 discrepancy 选择反而同时损害 field 与 held-out-B。更值得解释的结果是 field
relative-L2 从 `k=1` 到 `k=24` 改善约 6%--7%，gradient relative-L2 却恶化约 24%--31%，同时 front F1
仍能改善。后续讨论不得把三者压成单一“重建质量”；下一候选首先要改变 regularization path，并以 field/gradient/front
及 A/B consistency 的 Pareto 支配为门，而不是只选择旧路径中的一个 checkpoint。

即使所有预注册门通过，仍需保留以下局限：

- 经验轴对齐 calibration envelope 没有 distribution-free coverage；
- 5% 范数界是冻结设计选择，不是普适最优常数；
- A residual 非增不推出 field error 非增；
- 离散 operator 的 near-null 结构不等于真实光学 null space；
- L1 B 组共享 renderer，不能证明跨物理泛化；
- 2,729 参数小模型的优势可能依赖当前 `12^3` toy 尺度；
- 真实 curved-ray callable、JVP/VJP 与数据权限尚未取得。

---

## 20. 发表门槛

| 证据阶段 | 必须达到 | 最多允许的结论 | 仍禁止的结论 |
|---|---|---|---|
| P0 工程门 | parity、ledger、gradient、防火墙；当前 **2 个证据 P0 closed、1 个跨 epoch transaction P0 open**，不能写成 resume/crash-safe 完成 | 只能说已有部分可审计证据；师兄确认真实 callable/物理痛点并认可路线值得 fit 后，才最小补跨 epoch 原子恢复 | 算法有效、训练成功、训练管线已完整或 crash-safe |
| P1 L1 sealed route | 第 7.1 节全部联合门 | 小型 supervised synthetic pilot 值得进入 L2 | 优于 DCDM/FNO、unseen 泛化、真实成功 |
| P2 L2 matched benchmark | 强近邻、公平预算、更多 rig/noise/renderer、机制消融 | 在指定 benchmark 范围形成稳定 Pareto 证据 | 真实装置有效、普适安全 |
| P3 真实 OERF | 独立 rig/session/相机或外部参照、现实成本、失败案例 | 对指定装置与工况有实验价值 | 跨实验室泛化 |
| P4 外部复现 | 独立装置/团队复现或公开 benchmark 重现 | 可讨论较强可转移性 | 无条件安全、所有 BOST 最优 |

**面向高质量完整论文的最低门槛：P0+P1+P2+P3 全部满足。** 只有 P1 的 synthetic mean gain，即使数值显著，
也不足以支撑高质量算法论文。若 P2/P3 失败，应转为边界/负结果论文或缩小主张，而不是降低门槛。

未来只有在重新完成投稿时文献检索、全部强基线、公平成本、未见 rig、真实独立视角和外部审计后，才可考虑以下
窄措辞：

> A BOST-specific, observable-only and norm-bounded first-direction policy improved a fixed-budget CGLS shell across
> preregistered rigs without degrading heldout-ray consistency, while reverting exactly to the baseline outside its
> calibrated operating envelope.

即便如此也不自动允许使用 “first”“provably safe”“general-purpose”“breakthrough”。

---

## 21. 收到何远哲师兄回复前后的真实接口门

当前下一科学门不是继续扩大 toy 安全基础设施，而是确认以下真实合同：

1. 组内 NeRIF/BOST forward 的仓库、版本与入口函数；
2. 输入是三维场还是 decoder 参数；
3. curved/straight 路径及 residual 在 ray/sample/integrand 还是 detector 层形成；
4. 是否支持运行时任意 `x/v/q`，以及 JVP/VJP/autograd；
5. hard mask、termination、occupancy pruning、自适应采样等离散分支；
6. 最小匿名 `x0`、4--16 rays 或最小合法 batch 与简化 geometry；
7. Linux/container、GPU、PyTorch/CUDA 与编译扩展；
8. 真实主痛点、主指标和必须比较的组内基线；
9. 源码、权重、标定、样本和日志的保密/发布边界。

接口分支：

| 可取得接口 | 首个真实实验 | 允许进入的阶段 |
|---|---|---|
| forward + JVP/VJP + native residual | 导数稳定、相消误差、局部 adjoint/Schur 检查 | E4 前置 |
| curved/straight 但无 native residual | 诚实双路径，不伪造第三路 residual | E4 前置 |
| 只有 forward | finite-difference/complex-step 可行性与 branch map | 暂不训练神经算子 |
| 主痛点为有限视角 | 先冻结 SIRT/TV/PCGLS/NeRIF 基线 | E3/E5 |
| 主痛点为时间/调用 | 构建 calls/rays/wall/memory/失败率 Pareto | E3/E5 |
| 有连续高速序列 | 按 rig/session/time block 留出后才讨论 4D | E6 |

源码不能离开实验室时，可在组内服务器提供远程 callable 或由本研究编写 wrapper 后交师兄审核；不得把私有数据、
VPN 下载内容、凭据或受限论文上传公开仓库。

---

## 22. 一级来源阅读与引用清单

本节只列现有审计中已核验的作者/出版社/会议/arXiv/官方代码入口，不新增未核验 DOI。

### 22.1 学习方向、Krylov 与神经算子

1. Kaneda et al., **Deep Conjugate Direction Method**：[PMLR](https://proceedings.mlr.press/v202/kaneda23a.html)，[arXiv](https://arxiv.org/abs/2205.10763)。
2. Rudikov et al., **FCG-NO**：[PMLR](https://proceedings.mlr.press/v235/rudikov24a.html)。
3. Lu et al., **DeepONet**：[Nature Machine Intelligence](https://www.nature.com/articles/s42256-021-00302-5)，[DOI](https://doi.org/10.1038/s42256-021-00302-5)，[官方代码](https://github.com/lululxvi/deeponet)。
4. Li et al., **Fourier Neural Operator**：[arXiv](https://arxiv.org/abs/2010.08895)，[官方代码](https://github.com/neuraloperator/neuraloperator)。
5. Molinaro et al., **Neural Inverse Operators**：[PMLR](https://proceedings.mlr.press/v202/molinaro23a.html)，[arXiv](https://arxiv.org/abs/2301.11167)。
6. Long et al., **iFNO**：[PMLR](https://proceedings.mlr.press/v258/long25a.html)，[官方代码](https://github.com/BayesianAIGroup/iFNO)。
7. Kopaničáková and Karniadakis, **DeepONet Based Preconditioning**：[SIAM 正式页](https://epubs.siam.org/doi/full/10.1137/24M162861X)，[DOI](https://doi.org/10.1137/24M162861X)。

### 22.2 BOST 与神经三维/四维重建

8. Grauer et al., **Instantaneous 3D flame BOST**：[Elsevier 正式页](https://www.sciencedirect.com/science/article/pii/S0010218018302694)，[DOI](https://doi.org/10.1016/j.combustflame.2018.06.022)。
9. Cai et al., **Direct BOST using RBF**：[Optica 正式全文](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100)，[DOI](https://doi.org/10.1364/OE.459872)。
10. Bo et al., **BOST using GRU**：[Optica 正式全文](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-23-39182)，[DOI](https://doi.org/10.1364/OE.505992)。
11. He et al., **NeRIF**：[AIP 正式页](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)，[DOI](https://doi.org/10.1063/5.0250899)，[arXiv](https://arxiv.org/abs/2409.14722)。
12. Li et al., **NeDF**：[arXiv](https://arxiv.org/abs/2409.19971)，[DOI](https://doi.org/10.1063/5.0241191)。
13. He et al., **TDBOST**：[ACM DOI](https://doi.org/10.1145/3809488)。

### 22.3 一致性与主张边界

14. Schwab, Antholzer, Haltmeier, **Deep Null Space Learning**：[arXiv](https://arxiv.org/abs/1806.06137)，[DOI](https://doi.org/10.1088/1361-6420/aaf14a)。
15. Hendriksen et al., **Noise2Inverse**：[arXiv](https://arxiv.org/abs/2001.11801)，[DOI](https://doi.org/10.1109/TCI.2020.3019647)。
16. Yaman et al., **SSDU**：[Wiley 正式全文](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28378)，[DOI](https://doi.org/10.1002/mrm.28378)。
17. Haltmeier et al., **SPLIT**：[arXiv](https://arxiv.org/abs/2604.15651)。

---

## 23. 可复现性与材料声明模板

### 23.1 每次结果必须绑定

```text
protocol version + canonical hash
source commit
environment lock/container digest
data/manifest hash
split and geometry schema hash
model state hash
optimizer state and exact step count
normalization/calibration hash
operator and evaluator ledgers
raw per-case/per-seed metrics
independent validation report
figure/table generation script hash
```

### 23.2 数据与代码可用性（待填）

> 公开代码：`[待填 commit/release]`。公开 synthetic manifest 与生成器：`[待填]`。OERF 原始数据、标定、权重与
> callable 的共享范围由实验室授权决定；若不能公开，将提供不含私密数据的 schema、最小公开替代 benchmark、
> 运行环境和聚合结果审计材料。受限论文 PDF、VPN 内容与凭据不进入仓库。

### 23.3 作者贡献（待填）

| 角色 | 人员 |
|---|---|
| Conceptualization | `[待填]` |
| Physics/data contract | `[待填]` |
| Methodology | `[待填]` |
| Software | `[待填]` |
| Validation/audit | `[待填]` |
| Experiments | `[待填]` |
| Writing/review | `[待填]` |
| Supervision | `[待填]` |

---

## 24. 工作稿更新规则

1. 本文件的预注册规则不得在看过目标结果后无痕修改。
2. 新阶段以带日期/版本的 addendum 追加；原始状态与 hash 永久保留。
3. 结果只能填入第 18 节对应表，并附机器可读证据路径；没有证据时保持 `NOT RUN/NOT AVAILABLE`。
4. 摘要中的“当前结果”只有在主结果 bundle 独立验证后更新。
5. 突破标签只有在 P2/P3、强近邻、公平成本、真实证据和再次新颖性审计均通过后，才可提交给独立审稿人判断；
   作者不能凭内部 synthetic 均值自行宣布突破。
6. 任何失败都保留，并优先更新失败报告、limitations 与 claim ledger，而不是先修改标题。

---

## 25. 注册时自检结论

| 自检项 | 结论 |
|---|---|
| 是否虚构摘要结果 | 否；摘要明确 `NOT RUN` |
| 是否虚构图表或性能 | 否；L1 主结果仍为 `PLANNED/NOT RUN/CLOSED`，PSU-C1 与 R0 仅在清楚标注的 post-open addendum 登记封存证据 |
| 是否保持 0 scientific cases | 是 |
| 是否保持 0 optimizer steps | 是 |
| 是否声称 breakthrough | 否；明确 `no breakthrough` |
| 是否把工程验证当科学结果 | 否；E0 与科学阶段分离 |
| 是否列明可证伪主问题 | 是；RQ1 联合门及失败条件已冻结为工作稿骨架 |
| 是否覆盖物理、数学与方法方程 | 是 |
| 是否覆盖指定近邻碰撞 | 是；DCDM、FCG-NO、DeepONet、FNO、iFNO、NeRIF、NeDF、TDBOST |
| 是否覆盖 split、基线、指标、实验、消融、统计与失败 | 是 |
| 是否给出每张 planned figure/table 的判定问题 | 是 |
| 是否给出结果填充表与发表门槛 | 是 |
| 是否只使用现有一级来源链接 | 是；未新增未核验 DOI |

**注册时最终结论：** 当前只有一份可反驳、可持续填充的研究协议与论文骨架。没有模型训练，没有科学结果，
没有真实 BOST 证据，没有优于现有算法的结论，也没有突破。下一有效门是 **advisor-interface first**：先发送
`docs/n5_d5_advisor_first_contact_2026-07-19.md`，与何远哲师兄确认真实 callable、JVP/VJP、几何标定、物理主痛点、
基线、数据和宿主合同。PSU-C1 已关闭 bounded first-direction，R0 又关闭当前路径上的 learned stopping；收到回复前，
只允许在公开 PSU 上比较固定 H1/Sobolev、Tikhonov、TV/Huber、hybrid projection 与 edge-superiorization 路径，
不得启动神经 fit。只有新路径出现可观察 headroom，且师兄确认它贴合真实痛点，才最小实现仍开放的跨 epoch transaction P0。
当前为 **2 个证据 P0 closed、1 个跨 epoch transaction P0 open**；在 advisor 确认与该 P0 闭合之前，任何
scientific optimizer step 均保持 `BLOCKED`，科学结果只能从 `0 cases / 0 steps` 之后按本稿顺序产生。
