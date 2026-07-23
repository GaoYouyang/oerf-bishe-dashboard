# 少查询结构化残差算子校准的一个保守方法草稿

**状态：工作草稿；非预注册；非投稿稿；不构成论文就绪或性能优越性主张。**

**日期：2026-07-16**

## 摘要

本文提出一个待检验的少查询线性算子校准构造：先以具有已知 ray-local 结构的 gate 算子拟合可解释部分，再以同一批 `K` 个高保真 forward 观测的未解释残差构造秩至多为 `K` 的校正项。我们把该构造暂称为 Structured Residual Calibration Operator（SRCO）；用噪声地板缩放其幅度的版本暂称为 DF-gated SRCO。本文当前的全部实证证据都只是**post-open 的内部合成 toy**：V6C/V6D 的公式、噪声、失配低秩结构和 rigs 均来自同一 toy generator，且方法是在看到此前 misspecification toy 后提出。因此，这些数字既不是预注册检验，也不是外部 renderer、真实 BOS、PSU 或 OERF 数据上的结果；没有测试 inverse 重建价值，不能支持对 DeepONet、FNO 或任何学习算子的优越性声明。

本文可严格给出的仅是有限维线性代数性质，包括秩界、无正则时的最小 Frobenius 范数插值、未查询正交补上的零作用以及解析伴随的严格共轭。V6C 给出必要的负例：always-on SRCO 在 in-class 噪声下把噪声视为残差，明显劣于 gate-only。V6D 表明，在同一 toy 中使用 generator-known 对角噪声 covariance 时，DF gate 可在该负例中接近关闭更新、并在手工设定的模型外低秩残差上保留改善；它仍只是一条需要在独立开发环境冻结后再预注册的假设。

## 1. 问题与范围

令 (A_\star:\mathbb R^P\to\mathbb R^M) 为目标 rig 的未知高保真线性化 forward operator，(A_0) 为可调用的 nominal thin-ray operator。部署时只允许获得预冻结 probes

\[
X=[x_1,\ldots,x_K]\in\mathbb R^{P\times K},\qquad
Y=A_\star X\in\mathbb R^{M\times K}.
\]

一列 (A_\star x_k) 计为一次 vector-query。目标不是从 (K\ll P) 的观测恢复完整未知算子，而是在明确的 query 合同下，构造可调用的 (widehat A) 及其解析伴随 (widehat A^T)，并在**独立、封存**的 operator 和 inverse 指标上与同预算方法比较。少查询适配不等于 zero-shot；它更不自动推出真实物理场、跨 aperture 泛化或 inverse 改善。

V6B 已验证该合同的 toy 管线可以拒绝第 (K+1) 次 forward query、记录 `K forward + 0 truth-adjoint`，并在同一 toy 中保留 in-class 正控制与模型外负控制；其状态是 `PASS_PROTOCOL_CONFORMANCE_ONLY`，fresh V6B 仍为 `UNCONSTRUCTED`。本文不把 V6B 的协议一致性写成科学结论。

## 2. 与相邻方法的碰撞边界

本构造的公式新颖性关闭，以下谱系必须被明确承认。

| 谱系 | 已覆盖的核心思想 | 本文不能声称的内容 |
|---|---|---|
| Broyden / least-change / multisecant | 由有限 input-output 对得到满足 secant 条件的最小改动 Jacobian 或线性更新 | 不能称首次少量观测低秩更新，也不能把原点输入输出对严格称为 quasi-Newton secant |
| Learned Operator Correction（LOC） | 用数据校正近似 forward；forward-only 改善不保证 inverse 或梯度正确 | 不能以 dot-product 代替真实残差梯度和重建检验 |
| 多保真 operator learning | 低保真物理/数值算子加少量高保真残差学习 | 不能称首次将物理算子与数据残差、或多保真与残差 operator 结合 |
| Bayesian optimal experimental design | 在有限预算下选择传感器或观测，并可显式处理相关观测误差 | 不能称首次用信息准则选择有限观测；真实 probe 设计必须把 flow-off covariance 纳入目标 |
| active neural-operator learning | 主动选择高保真轨迹、状态或时间步以节省训练数据 | 不能把“主动获取算子数据”当作新意；它与目标 rig 上选择校准输入向量相关但不等价 |
| DeepONet、FNO 及其他神经算子 | 数据驱动函数到函数映射与跨实例泛化的广阔比较对象 | 本文没有与其公平训练、数据量、查询量、速度或外部任务比较，绝不声称优于它们 |

更准确的定位是：SRCO 是一个**可审计的、部署时不更新主网络的有限秩残差校准器**。它只有在后续证明保持 BOST 几何/掩码/零空间、给出 probe 覆盖或误差界、并在独立数据上满足预注册的 operator、梯度和 inverse 门槛后，才可能形成贡献。`X,Y` 是从原点定义的 input-output 对；若要使用严格的 Broyden/multisecant 语言，必须改为局部差分 (\Delta X,\Delta Y) 或 Jacobian 语义。

这里的 probe 设计也要采用窄定义。OED 文献主要选择传感器、观测位置或数据采集策略；active neural-operator 文献主要选择训练轨迹、参数样本或时间步。本文若继续，研究对象应限定为：在目标 BOST rig 上、严格 `K` 次高保真 forward 预算内，选择可实际施加或数值调用的输入场 (x_k)，以覆盖**有限孔径/布局造成的结构化 operator residual**。这只是待检验的应用层差异，不自动构成方法新颖性；若实验装置不能定义这些输入，主动 probe 叙事应停止。

## 3. 方法

### 3.1 结构化基算子

令 (A_g) 为由 `K` 个观测拟合的 ray-local 27-channel gate 算子。抽象地，可写为

\[
A_gx=A_0x+\sum_{q=1}^{Q}g_q\,D_q S_qx,\qquad Q=27,
\]

其中 (S_q) 为已知局部移位/核通道，(D_q) 为由 nominal ray 条件和冻结系数形成的对角权重。gate 由 ridge 问题拟合：

\[
g=\arg\min_g\|Gg-r\|_2^2+\lambda_g\|g-\mathbf1\|_2^2,
\]

其中 (G,r) 仅由 (A_0)、冻结结构和同一 cache (X,Y) 构造。实现配置给出相对 ridge (\rho_g)，实际惩罚为 (\lambda_g=\rho_g\operatorname{tr}(G^TG)/Q)；手稿中的 (\lambda_g) 均指这个换算后的绝对惩罚。其伴随为

\[
A_g^Tz=A_0^Tz+\sum_{q=1}^{Q}S_q^TD_q^T(g_qz).
\]

这一定义是已实现 toy 的有限维版本；真实系统是否能保留 mask、尺度、相机分块、边界条件和 nullspace，尚未证明。

### 3.2 SRCO 残差更新与解析伴随

令 gate 未解释的校准残差为

\[
E=Y-A_gX.
\]

采用 ridge 形式（(lambda\ge0)）

\[
C_\lambda=E(X^TX+\lambda I_K)^{-1}X^T,
\qquad A_{\mathrm{SRCO}}=A_g+C_\lambda.
\]

配置中的 `secant_relative_ridge` 记为 (\rho)，实现使用 (\lambda=\rho\operatorname{tr}(X^TX)/K)；命题与公式中的 (\lambda) 均指换算后的绝对惩罚。当 (\lambda=0) 时，上式应理解为 (C_0=EX^\dagger)。实现可保留两个因子，避免 materialize (M\times P) 矩阵：

\[
C_\lambda x=E\big[(X^TX+\lambda I)^{-1}(X^Tx)\big],
\]

\[
C_\lambda^Tz=X(X^TX+\lambda I)^{-1}(E^Tz),
\qquad
A_{\mathrm{SRCO}}^Tz=A_g^Tz+C_\lambda^Tz.
\]

### 3.3 DF gate

设 (H_g=G(G^TG+\lambda_gI)^{-1}G^T)，其有效自由度为

\[
df_g=\operatorname{tr}(H_g).
\]

令 gate ridge 的残差映射为 (R=I-H_g)。若噪声协方差为 (\Sigma)，gate 拟合后应扣除的期望残差噪声能量是

\[
E_{\rm noise,res}=\operatorname{tr}(R\Sigma R^T).
\]

对固定设计和同方差线性 smoother，即 (\Sigma=\sigma^2I)，残差噪声自由度为

\[
\nu_{\rm res}=\operatorname{tr}[(I-H_g)^T(I-H_g)]
=MK-2\operatorname{tr}(H_g)+\operatorname{tr}(H_g^TH_g).
\]

于是 (E_{\rm noise,res}=\sigma^2\nu_{\rm res})。V6D toy 实际按 probe 列使用不同方差，现已保留每个 design row 的对角方差并直接计算 (\operatorname{tr}(R\Sigma_{\rm diag}R^T))，不再用全局同方差平均。定义

\[
\alpha=\left[1-\frac{E_{\rm noise,res}}{\|E\|_F^2}\right]_+,
\]

并定义

\[
A_{\mathrm{DF\text{-}SRCO}}=A_g+\alpha C_\lambda.
\]

对角 covariance trace 对 V6D 的固定线性 toy smoother 是精确的，但由残差能量估计“信号比例”仍是启发式；相关噪声、数据依赖 probe、非线性 gate 拟合和 covariance 估计误差都可使它失效。真实 BOS 应 whitening，或由独立 flow-off repeats 估计 (\operatorname{tr}(R\Sigma R^T))。V6D 使用的是 generator-known 对角方差，不是独立 flow-off repeats，故不能被解释为真实实验中的可用噪声估计。

## 4. 可严格证明的有限维命题

以下命题以精确算术表达代数事实；浮点实现仍须独立做 matrix-free/materialized 与 dot-product 检查。

**命题 1（秩与查询局限）。** 对任意 (lambda\ge0)，
\(\operatorname{rank}(C_\lambda)\le\operatorname{rank}(E)\le K\)。若 (x\in\ker(X^T))，则 (C_\lambda x=0)。

*证明。* (C_\lambda) 是 (M\times K)、(K\times K)、(K\times P) 三个矩阵的乘积，故秩不超过 (K)。对 (X^Tx=0)，代入定义即得 (C_\lambda x=0)。因此未查询子空间没有自动改进保证。\(\square\)

**命题 2（无正则的插值与最小范数）。** 若 (X) 满列秩且 (lambda=0)，则 (C_0X=E)；并且 (C_0=EX^\dagger) 是所有满足 (CX=E) 的矩阵中 Frobenius 范数最小者。

*证明。* 满列秩时 (X^\dagger X=I_K)，所以 (C_0X=EX^\dagger X=E)。任意可行 (C) 可按输入空间正交分解为 (C=EX^\dagger+N(I-XX^\dagger))；两项 Frobenius 正交，故最小值在 (N=0) 取得。\(\square\)

**命题 3（ridge 不严格插值）。** 若 (lambda>0)，则
\[
C_\lambda X=E(X^TX+\lambda I)^{-1}X^TX,
\]
一般不等于 (E)，其偏差为
\[
E-C_\lambda X=\lambda E(X^TX+\lambda I)^{-1}.
\]

*证明。* 右乘 (X)，再用 (I-(B+\lambda I)^{-1}B=\lambda(B+\lambda I)^{-1})（取 (B=X^TX)）即可。\(\square\)

**命题 4（解析伴随）。** 若 (A_g^T) 是 (A_g) 的欧氏伴随，则上节给出的 (A_{\rm SRCO}^T) 是 (A_{\rm SRCO}) 的欧氏伴随；DF 版本同理，只需将 (E) 或校正因子乘以实数 (alpha)。

*证明。* 令 (B=(X^TX+\lambda I)^{-1})。该矩阵对称，故
\[
\langle EBX^Tx,z\rangle=\langle x,XBE^Tz\rangle.
\]
再与 (A_g) 的伴随恒等式相加即可。\(\square\)

**命题 5（固定同方差 ridge 的残差噪声能量）。** 令观测噪声 (\varepsilon) 满足 (\mathbb E\varepsilon=0)、(\operatorname{Cov}(\varepsilon)=\sigma^2I_n)，ridge 的 hat matrix 为 (H)，残差映射为 (R=I-H)。则

\[
\mathbb E\|R\varepsilon\|_2^2
=\sigma^2\operatorname{tr}(R^TR)
=\sigma^2[n-2\operatorname{tr}(H)+\operatorname{tr}(H^TH)].
\]

*证明。* 由 (\mathbb E\|R\varepsilon\|_2^2=\operatorname{tr}[R\operatorname{Cov}(\varepsilon)R^T]) 得第一式；展开 ((I-H)^T(I-H)) 并利用 trace 的循环不变性得第二式。\(\square\)

当 (H) 是未正则的正交投影时，(H^T=H=H^2)，上式才简化为 (\sigma^2[n-\operatorname{tr}(H)])。若噪声协方差为一般 (\Sigma)，正确量是 (\operatorname{tr}(R\Sigma R^T))；这正是 OERF flow-off repeats 后续必须估计的对象。

这些命题不证明残差是物理的、不证明 (X) 覆盖未知误差、不证明 (A_{\rm SRCO}^T\approx A_\star^T)，也不证明任何 inverse 成功。

## 5. 精确查询与伴随账本

| 项目 | 合同 | V6B/V6C/V6D toy 实现状态 |
|---|---|---|
| 高保真 forward | 对每列 (x_k) 只调用一次 `measure(x_k)`；总计 `K` | `K=4/8/16/32`，每 rig 的计数等于 `K` |
| 高保真 adjoint | 不可调用；总计 0 | 报告为 0 |
| cache 复用 | gate、SRCO、DF-SRCO 共用同一 (X,Y)，禁止额外测量 | 是；残差由已缓存观测减去 gate 预测得到 |
| 评分 | prediction hash 应先于 hidden labels 解封 | toy 写入 prediction hash；真正 fresh 仍需独立 evaluator 进程 |
| 候选伴随 | 仅调用 nominal/结构算子的伴随和低秩因子转置 | 已有 float64 dot-product 检查；V6C 最大 defect (4.62\times10^{-13})，异方差 covariance 修正后的 V6D 为 (2.88\times10^{-13}) |

“候选内部 dot-product 通过”只表示实现的 forward/adjoint 互为转置，不表示它接近未知 truth adjoint。后续 G3 必须比较真实残差梯度 (A^T(Ax-y))，不能继续使用候选输出作为 truth 的替代品。

## 6. 已有 post-open toy 诊断

两个表均为 `K=32` 的 hidden-action relative L2 中位数，数字越低越好。`in-class` 是 truth 被 27-gate 家族生成再加噪声；`out-of-class` 是在其上手工加入 12% 相对 Frobenius 低秩残差再加噪声。它们不是独立 test split，更不是数据结果。

### 6.1 V6C：always-on SRCO 的必要负例

配置为 64 输入维、96 输出维、每 stratum 12 rigs、64 hidden probes、24 gradient points、0.2% calibration noise；状态为 `POST_OPEN_HYBRID_NOT_DOMINANT`。

| stratum | nominal | 同预算 calibration | 27-gate | always-on SRCO |
|---|---:|---:|---:|---:|
| in-class + noise | 0.100878 | 0.073041 | **0.000163** | 0.001970 |
| out-of-class + noise | 0.157698 | 0.109722 | 0.119993 | **0.083835** |

负面解释必须保留：在 in-class 条件，always-on SRCO 的中位误差约为 gate-only 的 12 倍，说明它可把校准噪声插值为更新。模型外 toy 下的改善只是这个手工构造上的后验线索，不能被汇总成“统一胜者”。

### 6.2 V6D：DF gate 的条件性正、负信号

配置为相同维度、每 stratum 16 rigs、独立固定 seed、64 hidden probes、24 gradient points、同样的 0.2% generator noise 与 12% 手工失配。状态为 `SECOND_STAGE_POST_OPEN_DIAGNOSIS`，未授权预注册。

| stratum | nominal | 同预算 calibration | 27-gate | always-on SRCO | DF-gated SRCO |
|---|---:|---:|---:|---:|---:|
| in-class + noise | 0.099685 | 0.070129 | **0.00017815** | 0.00199675 | 0.00017827 |
| out-of-class + noise | 0.158364 | 0.112605 | 0.121422 | 0.08767861 | **0.08767794** |

V6D 的正面 toy 信号是：在手工模型外残差上，DF-SRCO 的中位数低于 gate（相对改善约 28.78%），且 16/16 rigs 相对 gate 改善，逐 rig 最小相对改善为 25.0%。其负面/限制同样关键：DF-SRCO 并未在 in-class 上优于 gate；它只是近似关闭更新（in-class `alpha` 中位数 0、p90 为 0.01149），并且该现象依赖 generator-known 对角 covariance。out-of-class 的 `alpha` 中位数为 0.99971（p10 0.99969），这与人为植入的信号结构相符，却不能外推至真实噪声、真实 misspecification 或未知 probe 分布。

V6D 还报告 in-class 相对 gate 的 p90/max 退化为 0.78%/1.73%，最大绝对变化 (3.41\times10^{-6})。这些是内部 toy 的局部风险描述，不是普遍无害性保证。

## 7. 后续独立预注册与强基线

所有 V6C/V6D seed、rig、hidden probes、公式选择和阈值均永久归档为 post-open hypothesis data；不得回收为 fresh 证据。下一步应先在全新 opened-development renderer/数据接口上一次性冻结：(lambda,lambda_g)、DF 公式、噪声估计、probe family、K 曲线、主基线 (B^*)、统计代码、config hash 和 manifest。随后才可构造首开数据。

**Probe 设计不能只比“聪明方法 vs 随机”。** 至少冻结同预算的 random orthonormal、反应场/POD、thin-front/shock dictionary、nominal leverage/D-optimal、最大 predicted residual 五族；对相关噪声使用 whitening 或以 (\Sigma) 加权的信息准则。每种 probe family 的构造成本、是否访问训练分布、是否需要候选 truth、数值 renderer 与实验可施加性必须单独记账。主动学习论文中的样本选择只能提供 acquisition 设计启发，不能充当目标-rig vector-query 的直接先例或性能证据。

**数据层级。**

| 层级 | 允许回答的问题 | 不能回答的问题 |
|---|---|---|
| 独立 external synthetic renderer（例如 `photon`） | 独立代码路径上的 aperture/layout 迁移和 query 机制 | 真实光学或 OERF 有效 |
| PSU fixed rig | MAT schema、单位、mask、reprojection 和固定装置一致性 | 跨 aperture 泛化或无 truth 的绝对场 L2 |
| OERF | 独立 session、phantom、held-out camera、重复性和物理积分量 | 没有 3D truth 时的绝对 field 精度或完整 operator identification |

**同预算强基线。** 每种可部署基线必须复用完全相同的 (X,Y) cache、K 个 forward、0 个 truth-adjoint，并在 development 等预算调优后冻结。

1. `A0` nominal operator；
2. gate-only；
3. origin-based ridge / pseudoinverse calibration；
4. truncated-SVD 与递归 Broyden/multisecant；
5. always-on SRCO 与 DF-gated SRCO；
6. 每 ray 27-coefficient ridge 加邻接 ray graph Laplacian；
7. geometry polynomial/B-spline-to-kernel ridge；
8. 相同 K 的 kernel/GP discrepancy；
9. 相同 K、相同步数和同等可训练参数约束的 LoRA/last-layer adaptation；
10. full high-fidelity / best-rank-K oracle，仅作不可部署上界，绝不参加公平胜负。

DeepONet/FNO 可作为扩展的学习算子比较，但必须明确训练数据、参数量、目标 rig 访问、query、wall-clock、硬件和可用的 truth 信息；没有该公平实验，就不作任何相对结论。

## 8. 统计门槛与顺序门禁

建议采用 V6B 的 fresh 设计：A（未见 aperture/f-number/focus）、B（未见 layout/angles）、C（joint OOD）各 16 rigs，共 48 rigs；每 rig 仅有 K 个可见 calibration probes、32 个封存 operator probes 和 12 个封存 inverse fields。rig 是唯一统计单位，views、rows、重复噪声和 fields 不能伪装为独立样本。主统计量先在 rig 内聚合：

\[
d_r=\log\frac{e_{\mathrm{cand},r}+\sigma_r}{e_{B^\ast,r}+\sigma_r},
\]

再对 A/B/C 等权；以 rig-cluster bootstrap 给单侧 95% CI。

| 门禁 | 预注册通过条件 | 失败后的处理 |
|---|---|---|
| G0 physics gap | nominal mismatch RMS 至少为独立 noise floor 的 2 倍 | `PHYSICS_GAP_NOT_RESOLVED`，不写成算法成败 |
| G1 数值有效性 | 20 对 dot：float64 defect <= (10^{-10})，float32 <= (10^{-5})；有限差分梯度通过 | `INVALID`，禁止进入科学评分 |
| G2 operator | 相对 (B^*) 改善 >=10%，单侧 95% rig-cluster CI 排除 0，48 中至少 36 rigs 胜出，且每 stratum 至少 8/16；p90 paired degradation <=5%，max <=20%，各 stratum 聚合非负 | 不解封 inverse；不能称 operator 成功 |
| G3 梯度 | 在零场、扰动 truth、nominal/CGLS 中间点：cosine 中位数 >=0.99、5% 分位 >=0.95、无负 cosine、相对梯度误差中位数 <=0.25 | 只可报告 forward/operator，不能主张优化或 inverse 一致性 |
| G4 inverse | 相对 `A0` field error 改善 >=3% 且 CI 排除 0；同时胜 (B^*)，或在 2% 非劣界内且有预注册 runtime 优势 | 保留 `OPERATOR_ONLY` |
| G5 runtime | 同硬件/精度/批量报告 acquisition、fit、materialization、F/F^T、内存、整次重建、100 次重建摊销和 break-even | 关闭效率主张，不影响其他已通过门禁 |

强 operator GO 可额外定义为改善 >=25% 且 A/B/C 各至少 12/16 胜出。fresh 首开后不得修改 K、probe family、核半径、宽度、正则网格、seed、阈值或 (B^*)；任何 bug 修复都必须升级协议版本并使用全新 fresh manifest。

## 9. 威胁、停止规则与可解释边界

1. **后验选择。** V6D 因看过 V6C 的失败而设计，不能用于确认 DF 公式；必须重新预注册。
2. **toy 共源性。** truth 和候选共享 nominal/generator 家族；in-class 由候选家族生成，out-of-class 残差低秩且手工设定，都会高估可识别结构。
3. **噪声模型。** 实际 BOS 噪声可能相关、异方差、漂移且与亮度/相机相关。没有 independent flow-off repeats、逐相机/逐像素 covariance 与 whitening 验证时，停止 DF-SRCO 的真实部署主张。
4. **可操作 query。** 若真实装置不能主动施加已知 (x_k)，则必须说明 K 个 query 对应何种可操作测量；数值 renderer query 与实验标定场必须分账。无法定义时停止“少查询主动校准”叙事。
5. **结构破坏。** 低秩全局补丁可能破坏 BOST mask、ray locality、尺度或 nullspace。若结构契约、matrix-free 一致性或 G1 失败，停止该候选而非调参掩盖。
6. **未查询方向。** 命题 1 表明正交补不受补丁影响；若 probe 选择在 external synthetic 或 OERF held-out view 上不能覆盖残差，停止关于稀少 query 泛化的主张。
7. **inverse 风险。** 内部共轭和 hidden-action L2 都不足以验证 inverse。G2/G3/G4 任一未过，不得把 operator 数字外推为重建价值。
8. **外部可迁移性。** PSU fixed rig 不能替代 aperture/layout fresh strata；OERF 无 calibration phantom 时最多报告 measurement-domain adaptation，不称真实 operator identification。

当前唯一合规结论是：DF-gated SRCO 是一个值得带到独立 renderer/PSU/OERF 数据合同中**重新冻结并预注册**的候选；不是已验证的方法结果，不具备投稿或发布就绪性，也没有任何对 DeepONet/FNO 的比较性结论。

## 10. 可复核来源

- 假设与边界：[srco_postopen_hypothesis_2026-07-16.md](srco_postopen_hypothesis_2026-07-16.md)
- V6B 协议：[v6b_limited_query_preregistration_2026-07-16.md](v6b_limited_query_preregistration_2026-07-16.md)
- V6B/V6C/V6D configs：`demo_t16_operator/configs/v6b_protocol_conformance.json`、`v6c_hybrid_gate_secant_postopen.json`、`v6d_df_gated_srco_postopen.json`
- V6B/V6C/V6D runners 与低秩实现：`demo_t16_operator/run_v6b_protocol_conformance.py`、`run_v6c_hybrid_gate_secant_postopen.py`、`run_v6d_df_gated_srco_postopen.py`、`limited_query_calibration.py`
- V6C/V6D reports：`demo_t16_operator/results/v6c_hybrid_gate_secant_postopen/report.json`、`demo_t16_operator/results/v6d_df_gated_srco_postopen/report.json`
- 相关噪声下 OED：[Attia & Constantinescu, SIAM J. Sci. Comput. 2022](https://epubs.siam.org/doi/10.1137/21M1418666)
- 主动 neural-operator 数据获取：[Subedi & Tewari 2024/2025](https://arxiv.org/abs/2410.19725)、[Li et al., AISTATS 2024](https://proceedings.mlr.press/v238/li24k.html)、[Pickering et al. 2022](https://arxiv.org/abs/2204.02488)、[Kim et al., ICML 2025](https://proceedings.mlr.press/v267/kim25m.html)
