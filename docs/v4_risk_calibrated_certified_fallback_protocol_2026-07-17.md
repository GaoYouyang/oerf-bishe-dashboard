# v4 风险校准认证回退协议

> **状态：预注册开发协议，不是成功结果。**  
> **研究对象：固定线性化、可审计 signed-primitive 的 BOST / 层析逆问题。**  
> **当前边界：v3 在 130 次合成安全审计中 0 违例，但 selector 只赢 4/8 个 fresh rigs，最坏 field-L2 伤害为 +0.414402；所有真实 BOST、泛化和论文优越性授权仍为 false。**

## 1. 为什么必须做 v4

v2 已经说明，直接让网络预测 row / column mass 或安全上界会在 geometry-OOD 上失控：calibrated envelope 与 raw learned estimator 都在 4/4 个 fresh rigs 上违反 Schur 安全条件。v3 把安全性改成数学构造：网络只能从预先证明安全的 partitions 中选择，所以选择错了也不破坏 majorization。

v3 仍不能作为论文算法，原因不是平均值不好，而是尾部风险没有被控制：

- geometry-conditioned selector 的 mean field-L2 为 0.437171；
- train-selected fixed `paired_cross` 的 mean field-L2 为 0.489638；
- 平均改善 10.7155%，但仅 4/8 个 fresh rigs 胜出；
- 最坏 fresh harm 为 +0.414402；
- 6 个原称 `safety_calibration` 的 rigs 实际参与了 finalist 选择，只能视为 model-selection 数据；
- 当前成本是 analytic proxy，不是 streaming pass、peak memory 或端到端 wall time。

因此 v4 不再问“网络平均能否选得更好”，而问：

> **在不读取 fresh truth、target、exact mass 或 solver trajectory 的前提下，能否只在风险有独立上界时让候选接管，否则回退到固定安全基线？**

## 2. 暂定算法名与可主张贡献

暂定名：**Risk-Calibrated Certified Fallback（RCCF）**，中文为“风险校准的认证分组与安全回退”。

若未来所有门都通过，可考虑的论文贡献只有三层：

1. 从真实 BOST 固定线性化 Jacobian 导出并审计 signed primitive decomposition；
2. 对每个候选 partition 给出确定性 grouped-majorizer 安全证书；
3. 用独立 risk-calibration split 控制选择性接管的尾部伤害，并在证据不足时回退。

以下主张当前明确禁止：

- 首次把 neural primitive 用于 BOST；
- 已经优于 NeRIF、TDBOST、DeepONet、FNO 或 FFNO；
- synthetic rigs 证明了真实反应流场泛化；
- 平均改善可以抵消最坏样本伤害；
- all-in-one exact comparator 已被证明不可部署。

## 3. 确定性安全层

在一次冻结的线性化上下文中，要求

\[
A = \sum_{\ell=1}^{L} C_\ell.
\]

对预先登记的 partition \(\mathcal P\)，定义

\[
M_{\mathcal P}
=
\sum_{G\in\mathcal P}
\left|\sum_{\ell\in G}C_\ell\right|.
\]

由三角不等式逐元素得到

\[
M_{\mathcal P} \geq |A|.
\]

令

\[
r_i=\sum_j(M_{\mathcal P})_{ij},\qquad
c_j=\sum_i(M_{\mathcal P})_{ij},
\]

并构造

\[
\sigma_i=\frac{\eta}{r_i},\qquad
\tau_j=\frac{\eta}{c_j}.
\]

每个候选和 fallback 都必须独立审计：

\[
\sigma_i\sum_j|A_{ij}|\leq\eta,
\qquad
\tau_j\sum_i|A_{ij}|\leq\eta,
\]

以及

\[
\left\|D_\sigma^{1/2}AD_\tau^{1/2}\right\|_2^2\leq\eta^2.
\]

统计风险模型不能替代这些证书。任一 fresh case 出现 pointwise、row、column 或 spectral violation，相关 partition 立即淘汰；fallback 失败则整个 v4 停止。

## 4. 候选与 fallback

首轮只使用已经过 v3 构造审计的有限集合：

```text
candidate partitions:
  singleton_factor
  paired_local
  paired_cross
  triad_bridge

fallback:
  paired_cross

offline exact comparator only:
  all_in_one_exact
```

`all_in_one_exact` 不得进入 selector、risk model、threshold selection 或 deployment。它只用于画 accuracy-cost 上限，且必须实测其 streaming pass、peak memory、构造时间和 solver 总时间后，才能讨论 deployability。

## 5. 四路完整 rig 切分

同一 3D 场、同一 trajectory、同一 camera session 或其 ray / pixel 子集只能属于一个 split。

### 5.1 Train

允许：训练候选 partition policy 与 risk predictor；离线使用 truth 形成 harm 标签。  
禁止：决定最终接管阈值；查看 fresh 结果后改特征。

### 5.2 Model selection

只允许冻结：

```text
candidate catalogue
fallback partition
observable feature schema
normalization version
risk model class and complexity
finite threshold grid
harm tolerances
risk confidence level
coverage floor
```

这一 split 结束后，模型结构与阈值网格必须锁定。v3 的 `safety_calibration` 以后统一改称 `model_selection`，避免把选模数据误写成独立安全验证。

### 5.3 Risk calibration

只能在冻结策略族上计算风险上界、coverage 下界和最终 acceptance threshold。不得重新比较特征、partition 或模型结构。

### 5.4 Fresh test

只运行一次并报告：takeover coverage、fallback rate、conditional harm、worst harm、field / residual / front、真实成本和每个几何层的结果。任何 fresh 信息不得回流。

## 6. Deployment 可见信息

首轮 risk model 只接收 `DeploymentGeometry` 中冻结并签名的可观测特征，例如：

```text
view count and camera geometry
grid shape / spacing / domain extent
ray length summary
ray segment count summary
angle anisotropy
aperture / f-number / depth range
valid-mask and dropout fractions
calibration residual
reference-repeat noise summary
sampling / quadrature contract id
candidate partition id
analytic construction metadata
```

禁止作为 fresh selector 输入：

```text
truth or morphology label
target values
signed matrix or signed primitives
exact abs(A) / exact row-column mass
exact spectral norm
fresh reconstruction error
future solver trajectory
held-out reprojection error computed after selection
```

训练与 risk calibration 可以离线使用 truth 形成标签，但必须记录用途；validator 要通过反事实篡改证明 fresh selection 对 truth、target 和 primitives 不敏感。

## 7. 风险事件与接管规则

固定 fallback 损失为 \(\ell_b\)，候选损失为 \(\ell_g\)。对每个主终点定义

\[
H_m(x)=\ell_{g,m}(x)-\ell_{b,m}(x).
\]

预注册容差 \(\delta_m\)，并定义联合危险事件

\[
B(x)=\mathbf 1\left[\bigvee_m H_m(x)>\delta_m\right].
\]

首轮 tiny-vector micro-smoke 只能合法定义：

```text
field relative L2 harm
normalized residual harm
```

它没有空间网格和可信的前沿几何，因此禁止临时发明一个“front 指标”。进入 ASTRA/TIGRE 空间层析、flight-body BOS 或 OERF 数据后，必须把 front-location / front-F1 加入同一个 OR 危险事件；缺少 front 时不能开放空间重建论文主张。

风险模型输出 \(s(x)\)，阈值为 \(t\)：

\[
\pi_t(x)=
\begin{cases}
g(x),&s(x)\leq t\text{ 且 support gate 通过},\\
b,&\text{否则}.
\end{cases}
\]

任何 schema、context、sampling manifest 或 support mismatch 都必须 fallback；`A != sum(C_l)`、candidate/fallback 证书失败则必须 abort，不能用 fallback 掩盖实现错误。

## 8. 有限样本风险校准

对一个冻结阈值 \(t\)，设 calibration 中有 \(m_t\) 个接管样本，其中 \(k_t\) 个发生危险事件。计算选择条件风险的单侧 Clopper-Pearson 上界

\[
R_U(t)=\operatorname{Beta}^{-1}
\left(1-\alpha_t;k_t+1,m_t-k_t\right).
\]

若 \(m_t=0\) 或 \(k_t=m_t\)，规定 \(R_U(t)=1\)。同时对总 calibration 样本 \(n\) 的接管数 \(m_t\) 计算 coverage 下界

\[
C_L(t)=\operatorname{Beta}^{-1}
\left(\beta_t;m_t,n-m_t+1\right).
\]

若 \(m_t=0\)，规定 \(C_L(t)=0\)。风险与 coverage 使用两个 family budget；联合同时覆盖概率至少为 \(1-\alpha-\beta\)。正式 95% 联合协议取 `alpha=0.025`、`beta=0.025`，而不是让两条界各自都花掉 5%。

阈值只能从 model-selection 阶段冻结的有限网格 \(T\) 中选择。首版使用 Bonferroni 修正：

\[
\alpha_t=\alpha/|T|,\qquad
\beta_t=\beta/|T|.
\]

选择满足以下条件的最大 coverage 阈值：

\[
R_U(t)\leq R_{\max},
\qquad
C_L(t)\geq C_{\min}.
\]

没有阈值通过时，全局回退到 `paired_cross`。这不是失败处理漏洞，而是算法定义的一部分。

### 8.1 样本量现实约束

在 95% 单侧置信度、零次失败的理想情况下，要让未修正 CP 上界不超过 5%，也至少需要约 59 个**接管** calibration 样本。当前三阈值网格若把 2.5% 风险 family budget 作 Bonferroni 修正，则至少需要约 94 个零失败接管样本；总 calibration 数还要更大。因此几十个 toy rigs 不可能授权论文级 5% 风险主张。

开发 micro-smoke 只验证接口、泄漏防护和 fallback，不开放 paper claim。正式风险实验应先做 power / sample-size analysis，再冻结 cluster 数量。

## 9. 首轮 micro-smoke

目标运行时间：Mac CPU 单次低于 10 秒，测试低于 30 秒。

```text
train:              >= 8 complete rigs
model_selection:    >= 6 complete rigs
risk_calibration:   >= 12 complete rigs
fresh_geometry_ood: >= 8 complete rigs
dtype: torch.float64
device: cpu
candidate partitions: 4
fallback: paired_cross
risk family alpha: 0.025
coverage family beta: 0.025
threshold grid: reject-all + two frozen leaf scores
```

micro-smoke 无论数值多好都只能得到：

```text
SYNTHETIC_INTERFACE_GATE_PASS / NO-GO
research_claim_authorized = false
real_bost_claim_authorized = false
generalization_claimed = false
paper_superiority_claimed = false
```

## 10. Validator 必须重放的证据

公开 report 不能自证。独立 validator 必须从 config 重新生成 rigs 并重放：

- 四路 split 和稳定 seed；
- observable feature schema、hash、normalization 与 train/model-selection 支持包络；
- 候选策略与 risk model；
- 冻结三阈值网格、Bonferroni 后的 CP 上界、coverage 下界与 95% 联合 family budget；
- 从冻结 config 独立计算完整 calibration-policy SHA-256，并与校准对象内重算的指纹一致；该指纹必须同时绑定 rule、阈值网格、多重比较、alpha 分配、risk/coverage 门、joint harm endpoints 与容差；
- fresh selected partition 与 fallback reason；
- 求解器实际使用的 `A` 是否等于 `sum(C_l)`；
- 每个 candidate / fallback 的 grouped-majorizer 和 Schur audit；
- 相同 A / A-transpose 调用预算下的 trajectory；
- aggregate、claim boundary、provenance 和 checksum。

发布模式必须启用 clean-source validation：记录的 commit 必须真实存在，五个源码/config blob 的 SHA-256 必须与该 commit 中对应文件一致，config snapshot 的规范化语义也必须一致。仅检查“40 位 SHA 字符串”不算来源绑定。

validator 分成两层：全量 deterministic replay 复用同一 tiny solver；CP Beta 分位数、coverage、fresh selection-conditional denominator 和 worst takeover 则由 SciPy 路径独立重算。它不是第二套独立求解器，因此只能叫“重放 + 独立统计交叉检查”，不能夸大为完全独立数学实现。

即使攻击者同步修改 CSV、JSON 和 checksum，以下篡改也必须失败：

1. risk score；
2. calibration threshold；
3. fallback flag / reason；
4. split role；
5. 把 `all_in_one_exact` 注入 selector；
6. feature schema / hash；
7. `uses_truth`、`uses_target` 或 `uses_primitives`；
8. aggregate harm / coverage；
9. source commit / source hash；
10. 把 analytic proxy 冒充 measured wall time。
11. 放宽 risk/coverage 门或 harm 容差，并同步重算校准对象自己的内部 hash。

反事实测试还要证明：只篡改 fresh truth、target 或 primitive 时，冻结 selection 决策不变；离线 safety / performance audit 应随真实输入变化或拒绝不一致 bundle。

## 11. 公开数据迁移顺序

### P0：可控层析接口

[ASTRA Toolbox](https://astra-toolbox.com/docs/) 与 [TIGRE](https://github.com/CERN/TIGRE) 用于生成 2D / 3D 稀疏视角、噪声、几何扰动和经典 CGLS / SART / TV 对照。第一目标是验证 signed decomposition 与成本记账，不是训练大网络。

### P1：真实多视角 BOS

[Open-Source BOS Tomography Dataset of High-Speed Flow Over a Flight Body](https://arxiv.org/html/2508.17120) 提供 70 个视角，并在论文的数据说明中列出 calibration、flow-off/on、mask、deflection、3D reconstruction 与 NIRT code；数据入口为 [DOI 10.26208/1VE2-5C19](https://doi.org/10.26208/1VE2-5C19)。下载前仍需在数据仓库页面逐项确认许可证和体量。

这个数据集是当前最重要的真实外部门：

- 九视角 limited-view 重建；
- 未参与重建的 held-out deflection；
- 70-view / limited-view 对照；
- calibration / mask / reference-repeat 敏感性；
- shock-front localization 与 reprojection fidelity；
- NIRT / TV / classical iterative baseline。

### P2：神经算子基线

[PDEBench](https://github.com/pdebench/PDEBench) 的官方 [DaRUS 数据页](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2FDARUS-2986&version=8.0) 包含 1D/2D/3D compressible Navier-Stokes、diffusion-reaction、Darcy 等 HDF5 数据，采用 CC BY 4.0。它适合先冻结 DeepONet / FNO / U-Net 等统一训练和跨参数评估代码，但不等价于 BOST inverse。

[NeuralOperator](https://github.com/neuraloperator/neuraloperator) 可作为 FNO / TFNO / UNO / GINO 等实现入口。所有模型必须共享 split、输入几何、训练样本、参数量级、优化预算和评估脚本。

### P3：时间与测量噪声

[Helium-jet BOS dataset](https://zenodo.org/records/6136052) 含两组 raw BOS 与两组 traditional schlieren 序列，每组 3000 帧、6000 fps，总文件约 3.8 GB。它适合 deflection、噪声、时序与 reference robustness，不是 3D tomography 真值集；下载前确认页面许可证。

## 12. 公平 baseline 矩阵

### 经典逆问题

```text
FBP / FDK where geometrically applicable
ART / SART
CGLS / LSQR
Tikhonov
TV / Huber
fixed factor majorizer
fixed grouped partitions
all-in-one exact comparator as measured accuracy-cost upper bound
```

### 连续 / 神经重建

```text
NeRIF / NIRT-style neural implicit field
TDBOST-compatible low-rank / tensor route when code and data contract allow
3D U-Net finite-dimensional control
DeepONet
FNO / TFNO / FFNO
UNO / GINO
PINO only when the PDE residual and boundary conditions are genuinely available
```

不能把 forward surrogate 与 inverse reconstructor 混成一个排行榜。每个 baseline 必须先说明输入、输出、geometry conditioning、训练监督和 inference budget。

## 13. 论文级指标与图

至少报告：

```text
field relative and absolute L2
gradient error
shock/front location and width
held-out view reprojection residual
physics residual where valid
p90 / p95 degradation
worst-case harm
positive-rig fraction
takeover coverage and fallback rate
risk upper bound and observed fresh harm rate
forward / transpose calls
streaming passes
peak memory
construction and end-to-end wall time
parameter count and training budget for neural baselines
```

正式论文图至少包括：

1. 数据与四路 split 图；
2. signed primitive 到 grouped majorizer 的物理结构图；
3. 风险阈值-coverage-harm 曲线；
4. 每个 fresh cluster 的 paired result；
5. accuracy-cost Pareto；
6. shock / front 三维可视化与 held-out reprojection；
7. 失败样本和 fallback 原因；
8. calibration / geometry / noise OOD 分层结果。

## 14. 停止门

以下任一条件成立，立即停止扩网络或租 GPU：

- 真实 forward 无法验证 `A = sum(C_l)` 与 transpose 对应关系；
- candidate 或 fallback 出现任何 fresh safety violation；
- selector 必须读取 exact mass、truth、target 或 future residual 才能工作；
- 风险校准后 coverage 太低，或 conditional risk upper 仍超门；
- 独立 fresh 中 worst harm 超过预注册容差；
- grouped 方法没有端到端成本优势；
- exact comparator 实测并不比 grouped 方法昂贵；
- 真实数据只有一个物理场而把 camera subsets 当成独立重复；
- 无法说明相对 NeRIF、2026 neural refractive-index primitives 和传统 diagonal preconditioning 的新增贡献。

## 15. 何时需要 GPU

当前 Mac 足以完成算法、validator、32^3/64^3 层析 smoke 和小型 DeepONet/FNO。只有以下三项同时满足才租服务器：

1. v4 在 micro / public-small 上通过泄漏、安全和风险逻辑；
2. flight-body 或组内数据接口已下载并完成 checksum / schema audit；
3. 正式比较矩阵、seed、训练预算和停止规则已冻结。

届时最低建议 16 GB VRAM、32 GB RAM、200 GB NVMe；更稳妥为 24 GB VRAM、64 GB RAM、500 GB NVMe。128^3 以上 3D、多 seed FNO 与 4D 模型不应长期压在本机统一内存上。

## 16. 给师兄的最小阻塞问题

1. 当前 forward 在形成 factor majorizer 前，能否暴露满足 `A=sum(C_l)` 的 signed primitive contributions？
2. 自然分组是 view、gradient component、ray segment、aperture sample、camera tile，还是别的物理块？
3. 能否提供或审计 `Jv`、`J^Tq`、ray、mask、grid、unit、相机内外参和 sampling manifest？
4. 组内主要误差更接近 finite-aperture / calibration mismatch，还是 4D 异步、缺帧和时序变化？
5. 是否保存未经平均的 flow-off repeats、bad-pixel / confidence、timestamp、曝光和 sync / dropout？
6. 认可的主终点是 field、front、held-out displacement、PIV compensation 还是 velocity？
7. 能否永久封存一台 camera 和一个 session，不参与训练、选模或风险阈值设置？

如果第 1–3 项是否定答案，RCCF 不能直接迁移到组内真实 forward；届时应转向不依赖 signed primitive 的 held-out reprojection / finite-aperture mismatch 或 4D uncertainty 路线。
