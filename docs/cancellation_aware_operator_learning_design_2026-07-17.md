# 面向抵消感知的算子学习与三维 BOST：分阶段研究设计

日期：2026-07-17
状态：**研究设计 + synthetic negative evidence；没有真实 BOST、泛化或算法胜出主张**

## 1. 决策摘要

**当前最适合本科毕设继续证伪的主线是候选 A-v4：在逐元素可证明安全的 signed-primitive partitions 中做 geometry-conditioned 选择，并用独立风险门决定接管或回退最佳固定 partition。** 直接回归 row/column mass 的 v1/v2 已关闭；当前价值在于它形成了一条从 D0 根因、v2 失败到 v3 安全构造的可审计证据链：

1. 固定 signed-`A`、相同样本和相同 `K=128` 下，`exact-abs-view` 相对 formal factor-view 的平均 normalized residual gain 为 **64.1206%**，16/16 配对材料增益；
2. 只将 factor 的 view-max 行聚合换成 factor 行和，增益仅 **2.011%**，故主导缺口是 `M >= |A|` 的 factor majorizer 在符号抵消处的松弛，而不是 view 聚合；
3. high-quantile factor slack 的中位数为 **0.9728**，说明该松弛在尾部不是微小实现误差；
4. 但 exact-abs-row 的平均 field relative-L2 仅从 `K=64` 的 **0.9114** 到 `K=128` 的 **0.9136**，没有随残差继续改善。因此 D0 不支持“更小 residual 即更准三维场”，也不支持直接训练黑盒重建器；
5. full-support graph-PCGLS 与 reduced-support PDHG 的变量支持、先验和历史方向不同，故其明显更低的 field error 只能作为 **nonbinding headroom**，不能作为 D0 的同问题排名或训练 teacher。
6. v2 在完全 oracle-free 的 4 个 fresh geometry rigs 上仍出现 4/4 不安全，calibrated envelope 共 39 次 Schur violation；连续 mass 回归因此严格 NO-GO。
7. v3 的 130 次 partition 审计全部零违反，selector fresh mean 比最佳固定分组好 10.7155%，但只赢 4/8、最坏 harm 0.414402；安全构造成立，可靠选择仍失败。

候选 A-v4 的贡献上限应限定为：**在固定线性化算子 `J=ΣC_l` 的可审计 signed primitive 接口上，所有候选 partition 由构造保证 `M_P>=|J|`；学习器只用部署可见几何选择 partition，并由完全独立的 risk-calibration split 冻结接管/回退规则。** 只有在 fresh rigs 上 selection-conditional harm、coverage、field/front 与真实构造成本同时过门，才可升级到 larger synthetic 或组内数据。当前不能写“exact 一定不可部署”，也不能写“首次 primitive BOST”“优于 NeRIF/FNO/graph-PCGLS”或真实反应场有效。

候选 B 是更有潜在重建价值、但风险明显更高的第二阶段：在同一 data-coupled reduced support 内，引入受物理残差验证的低秩全局校正与有限 Krylov 历史。候选 C 只在获得真实 4D 连续序列、时间戳和不确定度记录后启动；不能把静态 D0 结果外推成 4D 成功。

## 2. 已冻结的 D0 证据与不可跨越的边界

### 2.1 证据对象

在真实 PSU detector geometry、opened analytic reaction phantoms、opened synthetic correlated graph heat、两个 replicate 的八类形态上，D0 固定：同一 signed forward/adjoint `A/A^T`、同一初值、同一几何、同一 checkpoint，且 absolute operator **只用于 diagonal metric**。统计单位是“两 replicate × 八种配对 morphology”，不是 16 个独立同分布样本。

令

\[
b \approx Ax,\qquad s_j=\sum_i |A_{ij}|,\qquad r_i=\sum_j|A_{ij}|.
\]

formal factor-view 用逐元素上界 `M >= |A|` 构造步长；exact-abs-view / exact-abs-row 将对应的 row/column mass 改为 `r,s`，但求解递推始终仍是 signed `A`。当 `T=diag(\tau)`、`\Sigma=diag(\sigma)` 时，D0 的安全对象为

\[
B=\Sigma^{1/2}AT^{1/2},\qquad
\sigma_i r_i\le \eta,\quad \tau_j s_j\le\eta
\Longrightarrow \|B\|_2^2\le \eta^2.
\]

这是 Schur 型充分安全证书；power iteration 仅为 nonbinding stress estimate，不能替代上界。[Pock 与 Chambolle 的对角预条件原始论文](https://doi.org/10.1109/ICCV.2011.6126441)是行/列绝对和 metric 的方法来源。

### 2.2 D0 数字应如何使用

| 比较，均为 `K=128` | 相对 formal factor-view 的 mean normalized-residual gain | material paired gains | 设计含义 |
|---|---:|---:|---|
| factor-row hybrid | 2.0110% | 0/16 | 单改 view-max 行聚合不是主因 |
| exact-abs-view | 64.1206% | 16/16 | factor majorizer 的抵消松弛是 material descriptive signal |
| exact-abs-row | 64.1831% | 16/16 | row-wise 聚合没有额外改变主判断 |

`exact-abs-row` 的 mean field relative-L2 在 `K=64` 为 `0.9114229`，`K=128` 为 `0.9135935`。因此后续候选的主指标必须至少同时报告 data residual、field/gradient/front（有合成 truth 时）、held-out reprojection、worst unit 与 harm fraction。仅凭 D0 不得报告下列结论：

- exact-|A| 已改善真实场或视觉质量；
- 422 个 data-uncoupled / `M`-zero 坐标给出了 `dim ker(A)`；
- 已证明 Krylov 的因果机制，或已找到可迁移的神经网络目标；
- 已优于 full-support graph-PCGLS、NeRIF、TDBOST、DeepONet 或 FNO；
- 已对真实 OERF/何远哲实验、不同几何、真实噪声或 4D 序列泛化。

有限孔径和深度场效应本身已有 cone-ray BOS forward/inverse 建模，不能把“考虑 aperture”写成创新；真正需验证的是少量算子/校准信息下的 BOST 特定、可证书化策略。[Molnar 等的原始论文](https://arxiv.org/abs/2402.15954)

## 3. 所有候选共享的实验合同

### 3.1 物理与支持合同

1. **signed physics 不变。** 所有迭代数据项由同一个冻结的 `A` 与精确离散伴随 `A^T` 计算；`|A|`、预测 mass 或网络输出不可充当物理 forward。
2. **support 不偷换。** 主试验只在 D0 定义的 data-coupled reduced support `S` 上更新，记投影为 `P_S`。full-support graph/Sobolev 方法可作为另列 headroom，永远标作 nonbinding，不能向 A/B 提供 teacher、输入、标签或后处理。
3. **算子、场和几何分组切分。** 同一 `rig_id` 的所有 rays、probes、fields、replicates 和 calibration frame 必须留在同一 split；至少设置 geometry-OOD、aperture/depth-OOD、noise-OOD 与 joint-OOD。禁止随机拆同一体场的 rays。
4. **不使用不可部署 oracle。** D0 的 `scale_by_view` 由 clean-truth projection RMS 得到，只可用于合成机制重放；真实部署的 scale 必须来自 flow-off / reference repeats 或预先冻结的 calibration。
5. **先冻结，后开分数。** 开发集只选架构、rank、特征、阈值和停止规则；最终集只执行一次。若某分支在 opened development 上失败，不再扫宽度、`eta`、rank 或 `K` 寻找过线结果。

### 3.2 共同基线与公平账本

每个候选至少比较：formal factor-view、factor-row hybrid、exact-abs-view / row oracle（只作上限或 teacher）、scalar PDHG、同 reduced support 的 CGLS/PCGLS、Tikhonov-CGLS、固定 inverse-Sobolev 或 TV/Huber（若目标确实加入该先验）。每行记录：`F` / `A^T` 调用数、exact-absolute streaming pass、参数量、训练/推理 wall-clock、峰值内存、硬件、随机种子、stop rule 与 input contract。

DeepONet 与 FNO 只在下述条件同时满足时进入横向比较：它们看到与本方法完全相同的几何/观测/支持输入，使用同一 train/dev/test rig split、同一 truth 可得性、同级参数与 GPU-hour 搜索预算，并报告训练数据量、物理 calls、推理时间及 OOD/harm。否则它们只能是文献背景，不能作为弱对手。其方法原始入口分别为 [DeepONet](https://arxiv.org/abs/1910.03193) 与 [FNO](https://arxiv.org/abs/2010.08895)。

### 3.3 分层指标和通用失败门

| 层级 | 必报指标 | 立即失败/回退 |
|---|---|---|
| 数值有效性 | dot-product defect、finite difference、Schur audit、NaN/Inf、符号与 unit manifest | `A/A^T` 不成对、Schur 证书失效、用 full-support 信息混入主方法 |
| operator/metric | relative mass error、tail error、certificate violations、exact-sweep cost | 在最终集出现任何未标记的 unsafe deployment；准确但不省成本也不改善轨迹 |
| inverse | residual、field/gradient/front、held-out view、p95/worst/harm | 仅平均 residual 变好而 field/reprojection 尾部恶化；超过 frozen call budget |
| transfer | flow-off noise、calibration drift、session/camera OOD | 没有独立 session/camera 或 oracle scale，结论降为 synthetic only |

没有实验三维 truth 时，真实数据主结论只能是 held-out optical/reprojection consistency、repeatability、calibration sensitivity 与运行成本，不能把作者重建或训练重投影当作 density truth。NeRIF 已给出 neural field BOST 的数值和实验邻近协议，不能声称首次 neural-field BOST。[He 等](https://arxiv.org/abs/2409.14722)

## 4. 候选 A：oracle-free 几何条件化 mass 估计 + 分层安全审计

### 4.1 问题、输入与输出

目标不是学习 `x`，而是给每个 geometry `g` 与可观测局部算子描述 `q_i,q_j`，估计 exact absolute row/column mass：

\[
\hat r_i=f_{\theta,r}(q_i,g)>0,\qquad
\hat s_j=f_{\theta,s}(q_j,g)>0.
\]

`q_i` 可包含 camera id、ray origin/direction、detector coordinate、aperture basis/radius、path sample rule、whitening weight 与局部 coverage summary；`q_j` 可包含 voxel coordinate、support mask、到各相机的相对几何、ray-hit histogram 与 frozen discretization metadata。输出是 `\hat r\in\mathbb R_+^m`、`\hat s\in\mathbb R_+^{|S|}`、预测区间/置信分数 `u_r,u_s`，以及不确定时的 reject flag；不是三维场、不是 signed operator。

无实验室数据时，标签由既有可执行 synthetic geometry/ray generator 以 streaming exact-absolute sweep 产生：

\[
r_i^{\star}=\sum_j|A_{ij}|,\qquad s_j^{\star}=\sum_i|A_{ij}|,
\qquad \ell^{\star}_{ij}=M_{ij}/(|A_{ij}|+\epsilon).
\]

训练可用 log-relative Huber loss、上尾加权和单调/非负约束：

\[
\mathcal L_A=
\operatorname{Huber}(\log(\hat r+\epsilon)-\log(r^\star+\epsilon))
+\operatorname{Huber}(\log(\hat s+\epsilon)-\log(s^\star+\epsilon))
+\lambda_q\mathcal L_{\rm upper\ tail}.
\]

训练/开发阶段可看到 `r^\star,s^\star`，但最终 inverse 评分不能用 field truth 调参。每个 geometry 的 exact sweep 是一次可审计的 teacher construction，不是对每个 reconstruction 都免费可用的 oracle。

### 4.2 exact oracle、经验 envelope 与部署等级必须分账

纯预测的 `\hat r,\hat s` 不具有全局安全保证。为了不把 oracle 包装成学习器，必须把下列三种对象分开报告：

**离线 teacher / exact oracle：** 训练 rig 可做 streaming exact sweep 产生 `r^\star,s^\star`；held-out rig 的 exact mass 只用于 oracle 上限、事后审计和论文诊断。它的构造时间、内存与 streaming pass 必须完整计入，不能混入“学习推理成本”。

**oracle-clipped 诊断（安全但不加速）：** 若对每个测试 rig 先计算 exact mass，再构造

\[
\bar r_i=\max\{\hat r_i,r_i^\star\},\qquad
\bar s_j=\max\{\hat s_j,s_j^\star\},
\]
\[
\sigma_i=\eta/(\bar r_i+\epsilon),\qquad
\tau_j=\eta/(\bar s_j+\epsilon),\quad 0<\eta<1.
\]

于是逐元素有 `\sigma_i r_i^\star\le\eta`、`\tau_j s_j^\star\le\eta`，可恢复 D0 相同的 Schur 上界。但该方法必然至少与 exact metric 一样保守，且没有省掉 exact sweep；它只能验证接口和“额外阻尼会怎样”，不能进入部署成本排名。若 estimator 只在部分 row/column 使用 exact audit，未审计位置必须用可证明的 deterministic envelope，而不是把 calibration quantile 假装成定理。

**oracle-free 部署候选（尚未授权）：** 推理接口只能接收 geometry/operator descriptors，类型层面不得携带 exact mass、truth 或 target。在独立 safety-calibration split 上冻结保守 envelope `c_r,c_s`，使用 `\bar r=c_r(\hat r+u_r)`、`\bar s=c_s(\hat s+u_s)`；它只能称为经验性 coverage rule。fresh geometry 上只做事后 exact 审计，不可用审计值修补预测后再计分。只要最终集发生一次 under-envelope，主判决立即为不授权；回退 exact 只能算 fallback 成本，不得称为 Schur-certified deployment。

### 4.3 比较、消融、门禁与主张边界

**基线：** formal factor-view、factor-row、exact-abs-view/row oracle、简单几何 ridge、nearest-geometry mass、仅 global mean mass、无 learning 的 deterministic conservative envelope、训练集选择的 `c × exact` 标量阻尼和 exact-factor 标量插值。后两者虽使用 exact 只能作 oracle 控制组，却能检验小网络是否只是学到一个全局阻尼常数。DeepONet/FNO 不适合这个低维 metric 任务，除非另建完全同输入/输出的 operator-regression 基线；不应为了“有神经网络”而加入。

**消融：** 去掉 geometry；只用 camera id；只预测 row 或 column；不用 uncertainty；oracle-free raw prediction 对比 calibration envelope 与 oracle-clipped 诊断；以 `M` 而非 `|A|` 为标签；随机 field split 对比 rig split；只报 mean 对比 upper-tail / geometry-OOD。所有消融保持 signed `A`、`F/A^T` 预算和 `S` 不变。

**冻结门：**

1. 数值门：Schur 检查必须从 signed `A` 自行重算 exact row/column mass，并同时把行、列和谱范数超界计入 violation；调用方传入的 teacher 值不能作为证据。oracle-free fresh rig 任一 under-envelope 即不授权。
2. metric 门：在 fresh geometry-OOD 上预先报告 relative mass error、p95、worst row/column、coverage、interval width、rejection 与 fallback；若不能稳定胜过 factor 和简单标量/插值控制组，只能报告 predictor 失败。
3. inverse 门：主指标冻结为 field relative-L2；同时报告 residual、gradient/front、held-out reprojection、p95/worst/harm。只有 oracle-free 路线在预注册容许差内不劣于 exact oracle，才允许称“近似 metric 有替代潜力”。
4. efficiency 门：teacher、training、safety calibration、prediction、fallback/audit、solver 分项记时并报告 `A/A^T` 调用；若逐测试 rig exact audit 使总成本无优势，就不能主张加速。

**计算计划：** Mac 首轮只跑 `16^3`、40--60 个 geometry rigs、3 个固定 seeds、small MLP/gradient-boosted regression 与 streaming teacher cache；不进行 3D CNN 或大网格网格搜索。32^3 只在 A 的 geometry-OOD、certificate 与同预算轨迹均通过后做确认。服务器触发条件是冻结后的 `32^3` multi-rig/multi-seed 或 `64^3` exact sweep 超出 Mac 内存/一夜预算；申请前先提交包含 teacher sweep、训练、inverse 和 audit 四项的 profiling 表，禁止把预估时长写成实测。

**可写主张上限：** 若 fresh geometry-OOD 尚未过门，只能写“完成了 oracle-free metric surrogate 的接口、负例和安全审计协议”。只有 coverage、field/reprojection、尾部和成本均过门后，才可写“在冻结 synthetic BOST geometry distribution 上以经验性 envelope 近似 exact-|A| diagonal metric，并通过独立事后审计”；仍不能把经验 coverage 称为数学 Schur 证书，也不能写“学习了物理 forward”“恢复了 nullspace”“提高真实场精度”或“优于所有神经算子”。

### 4.4 初版接口 smoke 的独立审计：负结果，不是算法胜出

首个 9-rig 小型 smoke（7 train、2 held-out）证明代码可确定重放、signed-`A/A^T` 迭代账本一致、现有 6 次 Schur 检查为零违反；但它使用 held-out exact mass 做逐元素 clipping，因此没有部署价值。独立补算 `K=32` 后，两 held-out rig 的平均 field relative-L2 为：factor `0.84284`、exact oracle `0.36928`、oracle-clipped learned `0.40398`。learned 虽比刻意宽松的 factor 好 `52.07%`，却比 exact oracle 差 `9.40%`；两个 rig 中一个改善、一个明显恶化，不存在 aggregate superiority。原生成器的 geometry 还只沿 rig index 的一维曲线变化，多 seed 只改 jitter/noise，不能称 multi-seed geometry-OOD。

因此初版结论固定为 `metric_substitution_authorized=false`。下一 smoke 必须采用 train / safety-calibration / fresh geometry-OOD 三分、独立采样几何、部署输入类型隔离，并比较 factor、exact oracle、简单阻尼/插值、oracle-free learned 与 calibration-envelope；若 fresh OOD 不能稳定击败 factor 和简单基线，就停止扩大网络。

### 4.5 v2 oracle-free smoke：稳定几何种子复跑后，准确度与安全门同时 NO-GO

v2 使用 8 train、3 safety-calibration 与 4 fresh geometry-OOD complete rigs；推理 dataclass 不携带 signed `A`、exact mass、truth 或 target。`K=32` 的 fresh 平均 field relative-L2 为：

| 方法 | mean field relative-L2 | unsafe rigs | 角色 |
|---|---:|---:|---|
| exact oracle | 0.703056 | 0/4 | 非部署上限 |
| calibrated envelope | 151737.302297 | 4/4 | oracle-free 候选，OOD 灾难性失稳 |
| train-selected `0.5 x factor` | 0.862560 | 4/4 | 简单控制组，均值较低但不安全 |
| factor majorizer | 0.988963 | 0/4 | 可部署安全基线 |
| raw oracle-free learned | `2.180e26` | 4/4 | OOD 发散 |

exact-factor interpolation 在训练集选择 `alpha=1.0`，数值完全等于 exact oracle，故不是独立证据点。calibrated envelope 虽在 `ood-00/02` 的 field 指标同时胜过 factor 与 scalar baseline，但在 `ood-01` 发散到 `606945.292717`，在 `ood-03` 也明显退化；4 个 fresh rigs 共出现 39 次 Schur violation。raw learned 有 68 次 violation，simple baseline 有 28 次；只有 factor 与 exact oracle 为零。四个 rig 上 envelope 的 violation 数依次为 `11/18/1/9`，因此不能让 `2/4` 个局部胜例掩盖尾部失败。

因此两项授权保持 `false`。首轮独立审计指出：learned 特征读取 factor mass，端到端账本必须计 factor 构造；rig seed 不能依赖配置顺序；fresh exact 禁止访问要由运行时 guard 而非报告常量证明；布尔配置测试不能使用 `bool("0")`；源码必须能从记录的 commit 恢复。这些工程问题现已修复，rig seed 改为稳定的 `SHA256(base_seed, rig_id, split_role)` 派生，候选设置阶段 exact 调用由 guard 实测为零，15 个聚焦测试通过。seed 修复改变了 fresh geometry，所以旧数值全部作废；当前包仍是未提交源码快照，必须完成新一轮独立重算、提交后 clean-snapshot 重跑与 checksum 复核，才能作为可引用的 development negative evidence。

### 4.6 v3 假设：学习安全 partition，而不是直接猜上界数值

v2 暴露了一个结构矛盾：预测 mass 若比 factor 小才可能加快，但只要发生低估就可能失去 Schur 证书。一个更可证伪的替代是利用 forward 的 signed primitive decomposition。若

\[
A=\sum_{\ell=1}^{L}C_\ell,
\]

对任意 primitive partition `P` 定义

\[
M_P=\sum_{G\in P}\left|\sum_{\ell\in G}C_\ell\right|.
\]

三角不等式逐元素保证 `M_P >= |A|`。singleton partition 等于原 factor；all-in-one 等于 exact oracle；中间 partition 提供构造成本与 tightness 的离散前沿。学习器只从 deployment geometry 在预先验证的 partitions 中选择，不预测 mass，因此选择错误也不应破坏确定性上界。

该方向的论文门不是“某个分组比 singleton 好”，因为固定分组也能做到；必须证明 geometry-conditioned selector 在 fresh rigs 上逐例不劣于训练选择的最佳固定 partition，且没有总是退化为 all-in-one oracle，构造成本代理/实测与 field/front 同时有价值。真实迁移还要求组内 forward 暴露形成 factor majorizer 之前的 signed primitive contributions；若只能得到已合并 `A` 或最终 factor mass，该算法没有可执行接口。

### 4.7 v3 synthetic smoke：构造安全成立，选择性能仍未过门

最小 CPU 原型采用 12 个 train rigs、6 个 model-selection rigs 与 8 个 fresh geometry-OOD rigs，比较 singleton factor、三个中间安全 partition、训练选择的最佳固定 partition、depth-1 geometry selector 与 all-in-one exact comparator。这里原名 `safety_calibration` 的 6 个 rigs 实际参与了 finalist 选择，所以只能叫 model-selection split，不能当作独立验证证据。

所有 26 个 rigs × 5 个 partitions 共 130 次独立审计均满足 `M_P >= |A|`，row/column Schur 乘积上限为 `0.7`，最大归一化谱平方 `0.460370 < 0.49`。这支持“在该 synthetic primitive generator 上，partition 构造保持了确定性安全”，不支持真实 BOST 或任意非线性 forward 的安全性。

fresh 结果为：

| 方法 | mean field relative-L2 | 相对 train-selected fixed | 角色 |
|---|---:|---:|---|
| train-selected fixed `paired_cross` | 0.489638 | 基线 | 训练集选择的强固定分组 |
| geometry-conditioned selector | 0.437171 | +10.7155% | 平均有信号，但只赢 4/8 |
| all-in-one exact comparator | 0.316393 | 比 selector 低 27.6271% | 当前 toy 中的 exact 上限 |

selector 只选择 `paired_local` 或 `triad_bridge`，不能访问 all-in-one exact；但它在 4 个 fresh rigs 上回退，最坏 field-L2 绝对恶化 `0.414402`。因此 `research_claim_authorized=false`、`synthetic_algorithmic_smoke_gate_passed=false`。平均 +10.72% 只能用于生成下一假设，不能作为算法胜出或论文结果。

独立审计还收紧了三条工程边界：

1. 当前 toy 为所有方法提供完整 primitives 与 signed matrix，所以“all-in-one exact 不可部署”只是待验证的工程假设；真实迁移必须冻结流式访问、峰值内存、primitive materialization 与 wall-time 合同。
2. `cost_proxy_units` 的解析算术自洽，但不是实测运行时间，也尚未包含真实特征构造、I/O 与峰值内存；不能写效率优势。
3. 当前源码仍在 dirty worktree，旧 validator 对同步改写后的性能/成本 CSV 防护不足。正式引用前必须补全 trajectory/cost 独立重算，提交源码后 clean-snapshot 重跑。

下一候选不是更深 selector，而是**带性能风险回退的分层安全选择器**：所有候选 partition 继续保持数学安全；第一层只在可观测特征对“selector 相对最佳固定 partition 的收益下界”足够可信时接管，否则回退 `paired_cross`。新协议至少要分成 train / model-selection / risk-calibration / fresh 四层，并冻结 selection-conditional harm、coverage、worst-case field/front 与构造成本门。若 risk gate 不能把当前 4 个 fresh harm 全部挡回，同时保留足够覆盖率，就关闭 selector 支线。

真实接口与创新碰撞的逐项核对见 [v3 real-BOST interface map](v3_real_bost_interface_map_2026-07-17.md)。其中所有 camera/view、gradient component、aperture/quadrature 与 ray-segment 分组都只是待组内 forward 验证的候选，不能把非线性 ray tracing、随机采样或整个 NeRIF 优化过程直接写成固定线性分解。

## 5. 候选 B：抵消感知的低秩全局残差校正 + 有限 Krylov 历史

### 5.1 动机与受限公式

D0 表明静态 `|A|` metric 可大幅改善 residual，但 `K=64` 后 field error 不继续改善；而 full-support graph-PCGLS 的 headroom 又不能直接比较。这提示要检验的不是“再学一个静态步长”，而是：**同一 reduced support、同一 signed physics、同一调用预算下，有限历史的全局方向是否能补足静态对角度量缺失的耦合信息。**

设

\[
g_k=P_S A^T(Ax_k-b),\qquad
H_k=[g_k,g_{k-1},\ldots,g_{k-h+1},p_{k-1},\ldots,p_{k-h}],
\]

其中缺失历史补零并显式输入 mask。几何条件化模块输出 restricted low-rank basis `U_\phi(g)\in\mathbb R^{|S|\times r}` 与系数，候选方向写为

\[
q_k=-D_{\psi}(g)g_k
+P_S U_\phi(g) C_\theta(z_k)U_\phi(g)^T g_k
+\sum_{\ell=0}^{h-1}\alpha_{\theta,\ell}(z_k)g_{k-\ell},
\]

其中 `z_k` 仅由 `H_k` 的 norms/inner products、residual statistics、geometry、uncertainty 和 iteration index 构成。更新幅度不由 field truth 决定，而由真实数据残差的小线搜索或预冻结 trust-region 给出：

\[
a_k=\arg\min_{a\in[0,a_{\max}]}\|A(x_k+a q_k)-b\|_2^2+\lambda_a a^2,
\qquad x_{k+1}=x_k+a_kq_k.
\]

每一个非零 `q_k` 需用真实 `Aq_k` 验证。`U` 的列、所有 history 和所有 update 均投影到 `S`；不得将 full-support graph-PCGLS 轨迹、其 Sobolev prior 或其 full-field 结果作为 target，从而避免不公平 full-support leakage。

### 5.2 可得训练目标

没有实验室数据时，用未见解析/CFD-like synthetic phantoms 与冻结几何生成器构造：

- teacher direction：**同 reduced support、同 exact `F/A^T` budget** 的 CGLS/PCGLS 下一步或短窗 residual-minimizing direction；
- data-consistency target：`K` 步 residual area-under-curve、最终 residual、线搜索接受率；
- 有合成 truth 时的辅助 target：field/gradient/front loss，但不得在最终 test 上选择 `h,r,lambda`；
- safety target：预测方向相对基础 CGLS 的 residual increase、tail harm 与 geometry-OOD reject label。

训练可以是 imitation + residual-risk loss：

\[
\mathcal L_B=\mathcal L_{\rm residual\ curve}
+\lambda_d\|q_k-q_k^{\rm teacher}\|_{D^{-1}}^2
+\lambda_h\max(0,\Delta\ell_{\rm heldout}-\delta)^2.
\]

它不宣称 teacher 是全局最优，只是把“可由可比较的 reduced-support Krylov 历史得到的方向”定义为监督对象。

### 5.3 基线、消融和失败门

**基线：** reduced-support CGLS/PCGLS、Jacobi/exact-abs metric PDHG、fixed inverse-Sobolev、Tikhonov-CGLS、TV/Huber（同目标时）、固定 PCA/HOSVD low-rank correction、Anderson/multisecant/limited-memory Broyden（同 residual history）。full-support graph-PCGLS 仅单列 headroom。若获得足够规模的等输入训练数据，才补 DeepONet/FNO；否则不作比较性陈述。有关 inexact-operator/动态不确定度，应至少正面对比或解释与 [Learned ReSeSOp](https://arxiv.org/abs/2410.23061) 的差异；若学习 forward/adjoint correction，则必须遵守 [Learned Operator Correction](https://arxiv.org/abs/2005.07069) 所强调的双空间一致性，不能分别训练两个不相容黑盒。

**消融：** `h=0`（无历史）；`r=0`（无低秩全局项）；固定 `U` 对比 geometry-conditioned `U`；无 line search；无 reject/fallback；只看 residual 的 teacher；移除 cancellation features；低秩项改为 full-support（该行只能作为明确禁止的 leakage red-team，不进入主排名）。

**冻结门：**

1. 公平门：每个方法严格对齐 `F/A^T` calls、stop depth、support、正则化和初始化；不满足即不排名。
2. 数值门：`Aq_k` 与 update 逐步留痕；出现 residual acceptance 以外的 truth-selected step、NaN 或 support escape 即失败。
3. 效果门：最终几何/噪声/joint-OOD 必须同时报告 cluster mean、positive-rig fraction、p95/worst harm 与 held-out reprojection。仅 IID 平均正增益不解封真实测试。
4. 解释门：若 `h=0` 或 fixed low-rank 已等效，不能把结果称为“Krylov-like history”；若只等价于增大迭代数，必须按真实 wall-clock 与 calls 诚实报告。

**计算计划：** Mac 仅允许 `h<=4`、`r<=16` 的 `16^3` 机制试验，先缓存可重放 `A/A^T` trace，并用 linear/low-rank head 验证；不先训练大 3D U-Net。32^3 的多 rig × 多 seed、或需要每步额外 full forward 的 hyperparameter sweep 是服务器任务，且只能在 A 的 exact-metric audit 或独立 strong classical baseline 已完成后申请。B 的最大风险不是参数量，而是每个候选方向额外 `Aq` 使物理 calls 翻倍，故 compute 表以 calls/trajectory 为一级成本。

**可写主张上限：**“在冻结的 reduced-support synthetic BOST 协议中，受 signed residual 检查的有限历史方向是否改善同预算收敛和风险。”没有独立真实 session 与对齐 classical budget，不得写“学得 Krylov”“超过 graph-PCGLS”或“解决 nullspace”。

## 6. 候选 C：面向未来 4D 数据的事件/不确定度感知停止与正则化

### 6.1 启动前提与模型

候选 C 的问题不是从静态 D0 推出一个 4D 网络，而是在真实时间 metadata 到位后，检验“何时应继续迭代、何时应增强时间先验、何时应拒绝输出”。将 4D BOST 状态分解为 transport 与 sparse innovation：

\[
x_t=\mathcal W_{\varphi}(x_{t-1};u_t)+s_t,
\]

其中 `u_t` 是低维 motion/transport descriptor，`s_t` 表示新生、熄灭、薄前缘断裂等事件。数据项仍是每时刻的 signed operator：

\[
y_t=A_t x_t+\epsilon_t,
\qquad
\min_{x_{1:T}}\sum_t\|W_t(A_tx_t-y_t)\|_2^2
+\lambda_t\|x_t-\mathcal W_\varphi(x_{t-1};u_t)\|_1
+\gamma_t\mathcal R_{\rm spatial}(x_t).
\]

事件/不确定度网络只输出 `\lambda_t,\gamma_t`、迭代停止概率 `\pi_t(k)`、预测 interval `v_t` 与 reject flag：

\[
(\lambda_t,\gamma_t,\pi_t,v_t)=c_\theta(\rho_t,\Delta t_t,e_t,\mathsf{mask}_t,
\mathsf{exposure}_t,\mathsf{sync}_t,\mathsf{flowoff}_t),
\]

其中 `\rho_t` 是当前 residual / innovation statistics，`e_t` 为仅从已观测 history 得到的 event features。网络不得直接输出未验证三维场；停止必须受 residual floor、max calls 和 uncertainty threshold 三重约束。

### 6.2 无实验室数据的预训练与真实数据的最小验证

无实验数据时，只可用合成 4D 生成器预训练：解析反应场的平移/形变、受控 birth/death、不同 cadence、dropout、曝光积分、同步抖动、相关噪声与 calibration drift。标签为 synthetic event mask、state error、coverage 与 oracle-free stopping proxy（例如独立 noise-floor residual crossing）。这只能验证接口与负例，不可声称真实 4D 性能。

真实启动必须有冻结的按 **session** 切分：train session、development session、一次性 final session；同一连续 run 不能切帧混入不同 split。最低基线是逐帧 CGLS/PBB/NeRIF、transport-only、innovation-only、固定 Tucker/CP、TDBOST、time-only smoothing；只有同输入、同时间 mask、同训练序列、同算力下，3D+time FNO 才是公平扩展基线。TDBOST 已占据 4D BOST 张量/轻量网络路线，不能声称首次 4D BOST；其原始论文入口为 [ACM DOI](https://doi.org/10.1145/3809488)。

### 6.3 消融、门禁与成本

**消融：** 无 event head；无 uncertainty；固定 stopping；只 transport；只 innovation；无 timestamp/exposure；随机帧 split（仅作为明确错误的 leakage red-team）；不确定度不校准；不做 flow-off noise normalization。

**冻结门：**

1. metadata 门：缺 timestamp、exposure、sync/dropout 或 flow-off repeats 时，不训练 C 的控制器，只保留逐帧 classical baseline。
2. event 门：主报 birth/death、thin-front、temporal derivative、缺帧恢复、p95 与最坏 session；只报平均 frame-L2 即失败。
3. uncertainty 门：报告 coverage、interval width、selective risk 与 rejection rate；假置信或在 event 窗口系统性欠覆盖即回退固定规则。
4. 真实性门：无独立真实场 truth 时只报告 held-out views/temporal consistency/repeatability，绝不将 temporal smoothing 当作真实结构恢复。

**计算计划：** Mac 只做 20--40 条短合成序列的接口、时序 manifest 与 small-controller smoke；不在本机对大体素 4D FNO/transformer 作系统性搜索。服务器需等真实数据合同、session split、基线和事件指标全部冻结后，才用于 `16^3--32^3` 多序列训练。显存/时长以“每帧 signed operator calls × 序列长度 × seed”预先列账，而不是以网络参数量代替。

**可写主张上限：**若通过真实 final session，也只能声称“在指定 BOST 采集合同下，对事件窗口和不确定度作出受约束的停止/正则选择”；不等于得到 4D density truth 或普适动态 BOST 解法。

## 7. 建议的执行顺序与论文叙事

1. **D0 复核封存。** 在新文稿中固定报告 `64.1206%`、`16/16`、`2.011%`、`0.9728`、`0.9114 -> 0.9136` 和 graph nonbinding，连同 report/config/source hashes；不再把它当可继续调参的 leaderboard。
2. **封存 A-v2，推进 A-v4 风险回退。** 保留 v2 的 oracle-free 接口与负结果，不再增加 mass estimator 容量。先确认真实 `J=sum(C_l)` 接口，再把 train / model-selection / risk-calibration / fresh 四层冻结；风险门只决定 selector 接管还是回退最佳固定 partition。
3. **A-v4 通过逐 rig geometry-OOD、真实成本和接口门后才做 B。** B 先与 reduced-support CGLS、limited-memory classical directions 比；证明有限历史有独立贡献后，再讨论 learned global correction。
4. **C 等数据，不等想象。** 没有连续序列和 metadata，C 只能是清晰的未来接口，不启动所谓 4D superiority 实验。

本科论文当前更准确的标题方向是“**三维 BOST 线性化算子的抵消感知安全分组预条件与选择性回退**”。是否保留“有限孔径”取决于组内 forward 是否真实暴露 aperture sub-rays，不能先写进题目。可发表式价值必须来自：真实 Jacobian 的 signed primitive 合同、每个 partition 的构造证书、selection-conditional 风险门、强固定分组基线、geometry-OOD 与真实成本。若接口不存在或选择器不能保护尾部，毕业设计仍可诚实交付一条 D0→v2→v3 的 negative-evidence chain；不应为了“算法创新”转为无证据的大模型。

## 8. 必须向何远哲索取并书面确认的迁移合同

### 8.1 静态 A/B 所需最小包

1. 每台相机的内外参、像素坐标约定、焦距/物距、f-number/aperture、曝光、reference/tare、mask、单位、符号和世界坐标；版本化 `calibration_manifest`。
2. raw/未平均的 flow-off 与 reference repeats、坏点/置信度、相机/rotation/session id，用于独立 noise floor 和 `W`，不使用 clean truth oracle scale。
3. 可调用或可审计的 nominal `A_0,A_0^T`；若有高保真 ray/cone/aperture operator，则给出 `A_\star,A_\star^T` 或有限 calibration probe 的 forward/adjoint I/O。必须附 dot-product test、grid/support、boundary/gauge 与 hash。
4. 在一个小 ROI / 低分辨率固定线性化点导出的 signed primitive 接口：`Jv/J^Tq`、`C_l v/C_l^T q` 或可审计稀疏块，并能验证 `J=sum(C_l)`；同时给出 deterministic sampling manifest、streaming/materialization、峰值内存与成本计数。
5. 至少一个 calibration phantom、已知几何靶或可独立测量的温度/密度积分量；没有它时，真实主指标退回 held-out optical consistency。
6. 一个永久不参与训练、选模、阈值调整的 camera 或 session，外加明确允许的 data use、派生物、论文图和不重分发边界。

### 8.2 4D C 所需增量包

1. 50--200 帧以上连续 run 的真实 timestamp、frame rate、曝光积分、camera sync error、丢帧/坏帧日志；不是仅导出的帧序号。
2. 事件定义或可审核 proxy：点火/熄灭、thin front、shock/flare topology change；若无人工标签，至少给出可复算的由独立观测产生的窗口规则。
3. 训练/dev/final 的 session-level 切分建议及不得混用的相机、日期、工况；明确是否存在跨日标定漂移。
4. TDBOST/NeRIF 当前输入输出、rank/encoding、loss、训练和推理账本，以及哪些数据/代码允许只读审计、运行或修改。

### 8.3 需要当面确认的六个问题

1. 组内当前真正痛点是 finite-aperture / calibration forward mismatch，还是连续 4D 的新生、熄灭、异步与缺帧？
2. `A` 中负号主要来自何处，`A_0` 与高保真模型在何种物理层不同，是否能给出 paired probe？
3. 真实目标最看重 field、front 位置/宽度、held-out displacement、PIV/压力积分还是速度？每项的可接受误差/重复性 floor 是多少？
4. 哪个 camera/session 可以永久封存为 audit，且不被用于 calibration、training、model selection？
5. 是否有两档 aperture/focus、phantom 或 flow-off repeats，使 A/B 的 operator/uncertainty 门可独立验证？
6. 数据、标定、模型与图像的使用许可是什么；毕业论文、代码、聚合表与模型权重分别能否公开？

没有上述合同，A/B 只能保留为 synthetic protocol；C 不启动。拿到合同也不自动解封算法，仍须先冻结 split、baselines、calls、风险指标和停止规则。

## 9. 一级参考入口（仅列已核验的原始论文/官方入口）

1. Pock, T. and Chambolle, A. *Diagonal Preconditioning for First Order Primal-Dual Algorithms in Convex Optimization.* [ICCV 2011 DOI](https://doi.org/10.1109/ICCV.2011.6126441).
2. Molnar, J. P. et al. *Forward and inverse modeling of depth-of-field effects in background-oriented schlieren.* [arXiv:2402.15954](https://arxiv.org/abs/2402.15954).
3. He, Y. et al. *Neural refractive index field: Unlocking the Potential of Background-oriented Schlieren Tomography in Volumetric Flow Visualization.* [arXiv:2409.14722](https://arxiv.org/abs/2409.14722).
4. Lunz, S. et al. *On Learned Operator Correction in Inverse Problems.* [arXiv:2005.07069](https://arxiv.org/abs/2005.07069).
5. Feinler, M. S. and Hahn, B. N. *Learned RESESOP for solving inverse problems with inexact forward operator.* [arXiv:2410.23061](https://arxiv.org/abs/2410.23061).
6. Lu, L. et al. *DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators.* [arXiv:1910.03193](https://arxiv.org/abs/1910.03193).
7. Li, Z. et al. *Fourier Neural Operator for Parametric Partial Differential Equations.* [arXiv:2010.08895](https://arxiv.org/abs/2010.08895).
8. *TDBOST.* [ACM DOI](https://doi.org/10.1145/3809488).

## 10. 可复核的本地证据入口

- D0 guide：`docs/d0_exact_absolute_learning_guide_2026-07-17.md`
- D0 原始结果：`demo_t16_operator/results/psu_b0_exact_absolute_root_cause/report.json`
- D0 校验：`demo_t16_operator/results/psu_b0_exact_absolute_root_cause/validation_report.json`
- Gate B NO-GO：`docs/psu_b0_factor_pdhg_gate_b_no_go_2026-07-17.md`
- 4D / operator transfer contract：`docs/route_b_dco_trail_research_contract_2026-07-16.md`
