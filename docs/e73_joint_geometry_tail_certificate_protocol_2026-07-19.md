# E73-B：联合几何与相机尾部证书冻结前协议

> **状态：`DRAFT_UNFROZEN_NO_NEW_FLOW_SCORES_OPENED`。**
> **工作名：Joint Geometry-Tail Conformal Envelope（JGTCE）。**
> **这不是性能结果，也没有证明算法原创性。** 当前只完成了可单元测试的证书核心；没有训练预测器、没有打开新 flow 分数、没有 field truth、没有真实独立重复，也没有获得相对 DeepONet、FNO、FFNO、NeRIF 或 Pyramid BOST 的优势。

## 1. 它要解决的不是“再猜一个 k”

E72 在同一个真实 PSU 流场上暴露了一个很具体的问题：`16³/k3` 的 outer-50 等权相机 macro 略有改善，但 group p95、camera 5 relative-L2 和 camera 5 p95 同时变差；`32³/k2` 的 macro 与相机尾部也都变差。因此，**一个平均分不能替一台受伤的相机签字**。

这也解释了为什么当前不能直接堆一个 FNO：如果网络只优化平均 field loss 或 pooled reprojection，它仍可能把少数几何、相机或强梯度区域的伤害藏在均值里。E73-B 先建立一个更窄的统计接口：

1. 候选动作、指标、特征函数、预测器与尺度先冻结；
2. 对每个独立 flow/session，同时校准所有候选动作和所有相机尾部指标的上界；
3. 只有某个候选的**全部**上界都通过，才允许接管；
4. 样本不足、特征合同不一致、预测器哈希不一致或几何越界时，返回完全相同的经典 fallback。

它是后续 E73-C 神经 correction 的安全选择层，不等于三维重建网络本身。

## 2. 物理量与动作

固定经典回退为 `cgls_k4`。首轮候选集合在看 calibration 标签前冻结为

\[
\mathcal A=\{k1,k2,k3,k6,k8,k12\}.
\]

`k4` 不进入候选轴，防止“候选”和“回退”混成同一个对象。将任一误差指标记为 \(L_m\)，定义**加性伤害**

\[
H_{a,m}=L_m(x_a)-L_m(x_{k4}).
\]

正数表示候选比 fallback 差，负数表示候选更好。机器合同将 `ratio_to_fallback_metric_forbidden=true`：E73 禁止再用 `candidate/fallback` 比值作伤害，因为 fallback 接近零时比值会被任意小的分母放大或扭曲。基础误差本身必须非负、有限；signal RMS 的物理归一化和零信号 fail-closed floor 必须在指标定义中预先固定。

无 field truth 时，指标至少同时包含：

- 全 ray projection relative-L2；
- equal-camera macro relative-L2；
- worst-camera relative-L2；
- group p95 error / signal RMS；
- 每个冻结 camera slot 的 relative-L2 与 p95。

有可靠 CFD/phantom truth 时，另加 field relative-L2、梯度误差和预先定义的 front 指标。不能用 image-space 指标冒充三维场精度。

## 3. 联合上侧包络

在只含 model-fit units 的数据上训练固定预测器 \(\widehat H_{a,m}(X)\)，并固定正尺度 \(s_{a,m}>0\)。对第 \(i\) 个 calibration 独立单位，定义一个标量 nonconformity：

\[
R_i=\max_{a\in\mathcal A,\,m\in\mathcal M}
\frac{H_{i,a,m}-\widehat H_{a,m}(X_i)}{s_{a,m}}.
\]

设 calibration 数为 \(n\)，误覆盖率为 \(\alpha\)，取

\[
r=\left\lceil(n+1)(1-\alpha)\right\rceil.
\]

若 \(r>n\)，代码不把秩强行截到 \(n\)，而令 \(q=+\infty\)，所有候选回退。否则 \(q\) 是 \(R_1,\ldots,R_n\) 的第 \(r\) 个顺序统计量，并输出

\[
U_{a,m}(X)=\widehat H_{a,m}(X)+q s_{a,m}.
\]

### 3.1 可以安全写进协议的定理

若下列条件同时成立：

1. calibration units 与未来 test unit 可交换；
2. 预测器、尺度、动作集合、指标集合和特征函数不使用 calibration 标签拟合或选择；
3. 一个独立单位内部的所有 camera/view/ray 都留在同一行；
4. test 预测不读取 test harm、field truth 或动作执行后的 held-out 分数；

则标准 split-conformal 秩论证给出

\[
\Pr\left\{H_{n+1,a,m}\le U_{a,m}(X_{n+1}),
\ \forall a,m\right\}\ge 1-\alpha.
\]

因为事件一次覆盖冻结集合中的全部 \(a,m\)，随后只根据 \(X\) 和这些上界选动作，不需要再为“选了哪个动作”另做一次事后多重比较。若选出的 \(a^*\) 满足每个 \(U_{a^*,m}\le\delta_m\)，则在同一个联合事件上，全部真实伤害也不超过冻结容差。

若协议还要求某个主指标至少改善 \(\varepsilon>0\)，正确门是

\[
U_{a,m^*}< -\varepsilon,
\]

而不是 `U < tolerance - epsilon`。后者在 tolerance 为正时可能把仍有正伤害的候选误叫成“严格改善”；这一反例已经进入单元测试。

这个结论是**边际联合覆盖**。它不是“每个被接管样本都有 90% 安全”、不是 conditional-on-acceptance 保证，也不覆盖任意 OOD、时间漂移或依赖流序列。相关 split-conformal 入门与有限样本边界见 [Angelopoulos & Bates](https://arxiv.org/abs/2107.07511)。

把 support 拒答也写进策略后，最稳妥的概率表述是

\[
\Pr\{\text{support pass 且所选候选违反联合上界}\}\le\alpha.
\]

如果 support 外的输出在数值和预处理上都与 fallback 完全一致，则可把整套策略理解为“候选违例概率受边际控制、其余情况使用 fallback”。但这仍不等价于 \(\Pr(\text{违例}\mid\text{support pass})\le\alpha\)。

## 4. 几何 support gate 能做什么，不能做什么

当前代码只在 model-fit 特征上拟合低维 axis-aligned box。test 特征超出 box 就返回 fallback。这是透明的拒答规则，可以阻止明显的外推，却**不会创造 OOD 统计保证**：

- support 内部也不保证条件覆盖；
- 高维 min/max box 可能几乎没有辨别力；
- axis-aligned box 会接受训练中从未联合出现过的特征角点；
- rotation、ray coverage、aperture 等特征若变换顺序或单位，哈希必须变化；
- support gate、特征名称顺序、特征合同 SHA-256 与预测器产物 SHA-256 必须全部在运行时重验。

首轮只允许少量有物理意义的特征。若以后换成 Mahalanobis、one-class 或 learned support score，它本身也必须在 model-fit split 上冻结，不能看 calibration harm 后调阈值。

当前核心会核对调用方提交的 feature/predictor SHA-256，但“字符串相等”只是接口防呆，不是预测来源证明。新增 development runner 已把静态模型包与 deployment 分开哈希，自己打开固定 JSON ridge artifact 并在进程内生成 `predicted_harm`，不再接受外部同时提交“预测值 + 自称正确的哈希”。但它尚未从 raw BOST/source manifest 计算 feature；所有 caller-precomputed feature 即使内部 diagnostic 看似通过，也只能返回 `k4`，状态固定为 `DEVELOPMENT_ONLY_CALLER_FEATURES_UNTRUSTED_NO_CANDIDATE_AUTHORIZATION`。

不要把它与 Phase-0 data-foundation runner 混为一谈：后者已在 fixture 内内生计算 29 维 feature、密封 CGLS trajectory 与真实 fallback bytes。独立的 fixture-only scorer 现已验证 7 次 forward、24 指标与 `candidate-k4` harm，但尚未与 private finalizer 集成，更没有选择权。证书核心证明“冻结预测器到动作合同”的 fail-closed 接线，foundation 证明“观测到同轨迹 feature/fallback”的数据来源，scorer 只证明指标和符号接口；当前没有任何一套拥有真实 candidate authorization。

同样，动作名、指标名及其顺序必须跟 prediction table 一起绑定。尺度既要有来源 artifact SHA-256，也要有 canonical float64 数值 SHA-256 和 fit-unit ID hash，且 scale-fit units 只能来自 model-fit split。核心数组使用以不可变 `bytes` 为底层的 NumPy 只读视图，不能通过重新打开 `WRITEABLE` 标志改值；prediction 和 scale 的数值哈希还会在计算上界前再次复验。test/deployment unit 必须提交新 ID 与相同 unit kind；如果它和 fit/calibration 重叠，立即拒绝。以上检查能防止接线、构造后篡改和复现错误，但“ID 不重叠”本身仍不是统计独立的证明，独立性要由实验 manifest 与采集过程支持。

## 5. 为什么相机不能冒充样本量

同一瞬时流场经过三台相机观察，三行误差共享同一个三维场、同一实验时刻和大量共同系统误差。把三台 camera 当成三个独立 calibration units，会伪造样本量并破坏交换性论证。

合法单位优先级是：

1. 独立 flow instance；
2. 独立实验 session/run；
3. 预先定义且间隔足够的 condition-time block；
4. 仅用于开发的独立 analytic phantom instance。

camera、view、ray、pixel、aperture sample、checkpoint 都只能成为同一单位内部被联合保护的维度。相机越多，指标轴越宽，不代表 \(n\) 越大。

## 6. 样本量的硬边界

仅让经验分位数有限所需的最小独立 calibration 行数是：

| \(\alpha\) | 最小 \(n\) | 这意味着什么 |
|---:|---:|---|
| 0.10 | 9 | 只够得到有限最大阶附近分位数，绝不是稳定论文证据 |
| 0.05 | 19 | 仍极端依赖最坏 calibration unit |
| 0.025 | 39 | 适合开发一侧置信预算，仍需功效分析 |
| 0.01 | 99 | 数据要求很高，不能拿 camera/ray 凑数 |

这里的“有限”只表示不必自动回退，不表示上界足够窄或 acceptance rate 足够高。正式论文必须同时报告包络宽度、接管率、回退率与 empirical joint coverage；不能只报告 nominal \(1-\alpha\)。

这个秩保证只针对**一个新的可交换独立单位**。连续上线到许多未来单位时，“至少一次违例”的概率会累积；当前实现没有 anytime、sequence-wise 或 familywise deployment guarantee，也不能自适应地边看结果边更新 predictor。

## 7. 数据分工与防泄漏

### 7.1 Model fit

允许拟合：harm predictor、标准化、正尺度、低维 support、候选内部超参数。所有搜索都在这里结束。

### 7.2 Calibration

只计算冻结 predictor 的 \(R_i\) 与最终 \(q\)。禁止再改动作、指标、容差、特征、尺度、网络结构或 support threshold。

### 7.3 Development holdout

检查代码、成本、接管率和失败模式。看过以后可以开发下一版本，但这一 split 不能重新命名成“fresh final”。

### 7.4 Real confirmation

必须是独立 flow/session/condition blocks，并在打开分数前封存 protocol commit、config hash、predictor hash 和 feature hash。只有这一层能支持真实实验主张；analytic morphology proxy 不能升级为 CFD 或实验场。

## 8. Mac 上的两阶段实验

### Phase 0：16 个 phantom 的接口 pilot

复用私有 PSU compact B0 几何与 `psu_b0_reaction_phantoms.py` 的确定性解析形态，扩成八个 family、每类两个 seed。每个 phantom：

1. 生成 16³ truth proxy；
2. 用同一 BOST forward 合成九个 view；
3. 运行一条 CGLS 至 `k12`，保存冻结 checkpoint；
4. 相对 `k4` 计算 action × camera-tail harm；
5. 验证 public summary 不含观测、逐射线值、体素或 unit ID。

这一阶段只验证 schema、调用预算、拒答和实际运行时间。按 E72 的 Mac 成本估计约 2–3 小时，但这是待实测的工程估计，不是承诺。16 行不能支持 95% 有限包络，更不能形成论文性能结论。

### Phase 1：至少 64 个 analytic instances

建议先固定为 `24 model-fit + 20 calibration + 20 sealed audit`，每个 unit 必须运行全部冻结动作。使用 family-aware split，加入 iid noise proxy、camera-correlated bias proxy 与几何扰动。它可以筛掉明显无效的 predictor，比较 ridge、浅树和 constant predictor；即便通过，也只说明“在真实 PSU 几何上的解析形态压力测试可行”，不说明真实反应流场泛化。

## 9. 必须比较的强基线

1. fixed `CGLS-k4`；
2. 最小训练 residual 与 sparse L-curve，仅作 train-only 诊断；
3. 有独立噪声/模型误差合同后的 discrepancy stopping；
4. H1/Tikhonov；
5. TV/Huber primal-dual；
6. 忠实的 Pyramid BOST，而不是简单 `resize(x16)`；
7. outer/field oracle，只作不可部署上界。

所有可比方法报告 A、Aᵀ 调用、wall time、峰值内存、pooled、macro、worst、p95、每相机尾部、接管率和回退率。学习方法不能只对比未收敛的 DeepONet/FNO。

## 10. 与相关工作的真实关系

- [NeRIF，Physics of Fluids 2025](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the) 用连续神经场进行 BOST 体重建，是何远哲师兄最直接的表示学习主线；它没有给出本协议的 action × camera-tail 选择证书。
- [Pyramid BOST，Experiments in Fluids 2026](https://link.springer.com/article/10.1007/s00348-025-04153-3) 同步处理场、背景与投影矩阵的多尺度变化，是必须复现的强经典基线；普通上采样不等价。
- [Hucker & Reiß，Numerische Mathematik 2025](https://link.springer.com/article/10.1007/s00211-025-01469-4) 在明确统计逆问题假设下分析 CG prediction/reconstruction risk 和 data-driven stopping；E73 的经验相机尾部包络不能继承其 oracle inequality。
- [Conformal Risk Control，ICLR 2024](https://research.google/pubs/conformal-risk-control/) 控制单调 loss 的期望风险；[Learn then Test](https://arxiv.org/abs/2110.01052) 用多重假设检验校准有限候选。当前核心是更简单的 split-conformal 联合上侧预测包络，不能把这两篇的 risk-control 定理直接写成自己的保证。
- [GINO，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) 说明几何条件化神经算子已有成熟先例，因此“加入几何特征”本身不构成创新点。
- [Learning Preconditioners for CG，ICML 2023](https://proceedings.mlr.press/v202/li23e.html) 与 [FCG-NO，ICML 2024](https://proceedings.mlr.press/v235/rudikov24a.html) 说明学习可以进入 Krylov 求解器与预条件层；它们不是 BOST 相机尾部安全证书，也不能替代 H1/TV/Pyramid BOST 对照。

JGTCE 的潜在贡献只能是：**把固定 BOST 重建动作、几何部署特征、每相机尾部伤害和精确 fallback 放进一个可审计联合选择合同，并在多独立真实 flow 上证明它确实带来非平凡接管率。** 这一原创性仍需系统检索和真实实验才能成立。

## 11. Go / No-Go

以下任一项失败，E73-B 当前版本就是 NO-GO：

- predictor、feature、scale、action 或 metric 在 calibration 后变化；
- model-fit 与 calibration unit ID 重叠；
- 把 camera/ray 当独立行；
- 秩超出 \(n\) 却仍返回有限值；
- 任一被接管动作的某个联合上界超过容差；
- OOD 时没有返回 fallback；
- rejected 输出与 fallback 数组、调用预算或预处理不一致；
- 只在均值赢，worst/p95/某台 camera 受伤；
- 只有 analytic proxy，没有独立真实确认；
- 通过放宽 no-harm 容差换取接管率；
- 只赢弱基线或不同算力预算。

## 12. 现在需要问何远哲师兄的六件事

1. 能提供多少真正独立的 flow state、run、session 或 condition-time block？
2. 是否有一部分 CFD/仿真真值，哪怕只用于 field-level ranking？
3. camera pose、rotation、aperture、valid mask、ray coverage 的 manifest 是否齐全？
4. 是否有 flow-off / repeated-background 数据估计噪声和模型误差？
5. E72 中 view 5 是否对应已知的 calibration、遮挡、aperture、饱和或覆盖异常？
6. 哪些聚合数字、切片和误差图可以私下发给师兄审核，但不公开原始数据？

如果独立真实单位少于 20，优先把毕业设计目标定为“方法原型 + 严格 negative/feasibility study”，不要承诺 95% 分布无关真实安全证书。若能获得 40–100 个独立单位，才值得把 joint certificate 作为论文主轴。

## 13. 初学者最短学习路线

1. 先读 split conformal 的 rank 例子，手算 \(n=8,9,18,19\) 时为何会或不会自动回退；
2. 用两动作、两指标的 NumPy toy 验证 `max(action, metric)`；
3. 复现 E72：macro 为负伤害但 camera-tail 为正伤害时必须回退；
4. 学 CGLS 半收敛与 prediction risk / reconstruction risk 的区别；
5. 再实现 ridge harm predictor，不先上 FNO；
6. 最后才把同一证书套到受限神经 correction 或 geometry-conditioned operator 上。

当前可运行核心：`demo_t16_operator/joint_geometry_tail_certificate.py`。它实现的是证书和拒答，不是已训练模型。独立代码审计提出的两个 P0（错误 strict-improvement 参照点、直接构造 envelope 绕过尺度门）和第二轮两个 P1（NumPy `WRITEABLE` 可重开、scale artifact 未绑定实际数值）均已修复并进入攻击回归测试；最终窄范围复审为 `3 CLOSED / 0 PARTIAL / 0 OPEN`。当前构造器会重验正尺度、rank/quantile 一致性、fit/calibration/scale IDs 与哈希，结构化 prediction 会绑定 action/metric 顺序。新增 deployment runner 已关闭 caller-forged prediction、任意路径覆盖与重复声明单元消费；第二轮 runner 复审发现的“任意动作可冒充 fallback”P0 也已通过硬编码 `cgls_k4`、重哈希攻击测试和单根 descriptor 事务闭合。最新 data-foundation runner 已在微型 fixture 内闭合 geometry-only `y=A(x_proxy)`、同轨迹 sufficient statistics、精确调用序列和逐字节 `k4` finalizer；私有 unit manifest 另行绑定 run/unit、冻结 phantom 行、config 与 cache source，bundle 外原子 unit claim 与 final anchor 拒绝换名重复、合法重标和保范数 checkpoint + checksum 联动篡改。claim 后断电只允许在根锁内隔离可证明未发布且逐字节一致的 residue；已有 finalization 但缺 anchor 时禁止自动恢复。fixture-only scorer 已额外验证 7 次 forward、9-view、24-metric 与 harm 符号，但尚未与 private finalizer 集成。通用 store 只能得到 fixture 状态，生产 digest 只能来自权限收紧且严格解析的私有 attestation。这里明确依赖可信本地 Python 进程且没有密码学签名；真实 private unit、private metric/harm integration、16-unit schema pilot、物理 flow 别名防护和签名归档仍未闭合，因此不能授权 candidate。详细边界见 [核心独立代码复审](e73_joint_geometry_tail_certificate_independent_audit_2026-07-19.md)、[runner/Phase-0 前置审计](e73_formal_runner_and_phase0_preflight_2026-07-19.md)、[Phase-0 数据基础层](e73_phase0_data_foundation_2026-07-19.md) 与 [fixture metric scorer](e73_phase0_fixture_metric_scorer_2026-07-19.md)。
