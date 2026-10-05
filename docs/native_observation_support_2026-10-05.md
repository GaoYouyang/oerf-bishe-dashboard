# 原生观测支持与完整视野

## 中文

本次更新只发布私有三维标量代理的定性接口检查，不公开原始场、相机几何、数值数组、图表或组内源码。物理语义与单位仍未确认，不能称真实像素 BOS 位移或真实 BOST 迁移。

内部独立第二实现分别生成几何、重建单元积分与伴随，并验证相机乱序。旧粗探测器存在零列节点；结果前冻结的几何配置使用已知坐标包围盒确定完整视野，并按原生网格维度密采样，消除了零列。保持旧视野、使用相同密度的对照也解除支持下界否决，因此不能将效果解释为独特算法、学习模型或完整视野公式的独占优势。

对零列集合 U，零初值且不混合未观测空间节点的解法，其 gauge-centered 场误差至少为 `||rho_U-mean(rho_U)|| / ||rho-mean(rho)||`。本次只检查这个必要下界。消除零列不代表算子满列秩，也不代表四项物理精度或参考稳定性通过。

更密的探测器改变每次 A/A^T 的成本，不能与旧问题的调用次数直接相比宣称提速。旧 PoolFire 参考与 GCV 判决保持冻结，不因这个不同观测家族的检查而改写。下一科学门仍是另行冻结的经典参考解及其数值和物理精度核对；warm initializer、较少调用、fresh wall/RSS、外部门与真实 BOST 均未由本次检查证明。

这不是公开外部基准或第三方复现。页面不能提供私有实验的完整可复算证据；详细输入和结果只在本地保留，供获授权的组内审查。

### 后续物理歧义与先验排序

未正则化原生参考未通过冻结预算内的收敛证书和独立一致性要求，保留不可判定，不能称已收敛最小范数解或精确零空间下界。随后只对同一个已封存固定场做新鲜物理重放，独立实现确认三维场差别很大时二维观测仍可接近。这是后开封有限场对的可识别性诊断，不是所有算法误差的普适下界，也没有假设实验噪声大小。

真值可见的固定场对检查进一步确认：各向同性 TV 偏好正确场，平方梯度惩罚反而偏好错误场。这个排序没有拟合参数或新逆解，不能证明 TV 可行解唯一或 TV 重建通过。TV 在 BOS 火焰层析中已有[研究先例](https://www.sciencedirect.com/science/article/pii/S0010218018302694)，不属于本项目的首次方法贡献。

随后单独冻结的经典干净观测约束 TV 求解完成。成熟求解库与单独递推、几何和梯度实现的有限步输出通过内部独立一致性核对，首帧三维精度改善；但新鲜可行性与固定点收敛证书没有通过，权威参考判决仍不可判定。不能把绝对精度门通过替代参考认证，也不能把昂贵经典求解当成便宜 initializer。没有开启完整时序扩展，不授权学习、提速、外部泛化或真实 BOST 结论；不事后加深迭代、放宽证书或挑选更容易的场。

### 后开封固定初值与精确细化

原样 CGLS 接在已封存的 TV 有限步初值后，独立复算确认二维拟合改善而已有三维精度未受伤害；同样短的零初值对照未达到联合精度门。这只支持结构性初值与未修改细化器的兼容性，不是盲测、合格参考或等总成本算法优势。

新鲜伴随见证只核对细化增量，并未证明完整的非线性 TV 初值能由 exact lift 产生。昂贵 TV 初值的历史成本全部保留，收敛认证也没有因观测拟合改善而改写。下一实质门是合格参考、完整初始化的表示可行性和便宜 observation/geometry-only 初始化的公平比较；没有授权训练、完整时序、未打开数据或速度结论。

### 条件性伴随表示限制

内部独立见证进一步给出必要场误差界：在该冻结原生离散算子和结果前固定的有限对偶范数条件下，pure-adjoint 场族不能达到原场精度门。已知伴随表示对照没有误报。局部字典的独立门失败仍保留为不可判定，不因同一见证的物理重放通过而改写。

这只关闭该有界表示族，不证明精确零空间、无界对偶不可达性、连续光学或整个 C 路线不可能。一般 PCGLS 可能改变原伴随范围，没有被排除；普通 CGLS 的终端累计对偶范数也未在这个界中测量。不能转移到 PoolFire、另一网格、完整时序或真实 BOST。更大的预测器无法改变一个固定表示族的容量，但本次并未训练预测器。私有条件值、数值结果与图表仍不发布。

### 主线 PoolFire 同输入正规作用

主线另行结果前冻结了全体既有哨兵的同输入重放：两套物理实现使用完全相同的状态、观测和已选参数，独立核对前向、伴随、正规残差、标量和分层尾部。完整检查通过；两种 K128 状态均未满足继承的完整空间驻点门。它将同输入算子作用的一致性与有限 Krylov 参考问题区分开，不能覆盖旧的独立超门判决，也没有新逆解、CFD 真值读取或未打开数据访问。

[Hybrid projection methods 综述](https://arxiv.org/html/2105.07221v2) 区分早停的迭代正则化与完整空间优化。这里的含义是：非驻点不是所有有限预算重建的精度否决；未来比较必须预先区分合格优化参考与有限应用精度目标，不能事后将旧失败参考换名为成功。当前没有 learned algorithm、同成本优势、资源节省、外部门或真实 BOST 成果。

### 主线固定目标的完整 lift

随后另行冻结的主线表示审计通过：两套实现分别重建未修改的方向及三角伴随关系，使用相同固定有限目标生成离线系数，再做新鲜物理伴随和前向重放。全部哨兵目标都满足冻结复现与有限范数包络门。这里只证明固定目标的表示可行；不是三维真值可恢复性、完整序列、相机乱序或新相机泛化证明。

系数构造读取目标且重复昂贵方向计算，所有离线与重放调用保留，不能将一次伴随应用说成廉价在线算法。没有训练、同精度省调用或资源成果。旧参考失败不变；未来预测必须另冻目标政策、整轨迹隔离与公平经典对照，不能把这次表示证书当作学习授权，也不能把不同原生代理的条件性限制转移到 PoolFire。

## English

This update publishes only a qualitative interface check on a private 3D scalar proxy. Raw fields, camera geometry, numerical arrays, figures and group source code are not released. Physical semantics and units remain unconfirmed, so this is neither calibrated pixel BOS displacement nor real-BOST transfer.

An internal independent second implementation separately generates geometry, rebuilds cell integration and its adjoint, and checks camera reordering. The old coarse detector contains zero-column nodes. A preregistered geometry configuration derives a complete field of view from the known coordinate box and samples densely at native-grid dimensions, removing these columns. The equally dense old-view control also clears the support-bound veto. The effect cannot be attributed to a unique algorithm, learned model or exclusive advantage of the complete-view formula.

For a zero-column set U, zero-start methods that do not mix unobserved spatial nodes have a gauge-centered field-error lower bound of `||rho_U-mean(rho_U)|| / ||rho-mean(rho)||`. Only this necessary bound is checked. Removing zero columns does not establish full column rank, four-metric physical accuracy or reference stability.

Denser detectors change the cost of each A/A^T call, preventing a direct call-count speed comparison with the old problem. Frozen PoolFire reference and GCV decisions are not revised by this different observation family. The next scientific gate remains a separately frozen classical reference with numerical and physical-accuracy checks. This interface check establishes no warm-initializer benefit, reduced calls, fresh wall/RSS advantage, external gate or real BOST.

This is not an open external benchmark or third-party reproduction. The page does not provide complete reproducible evidence for the private experiment; detailed inputs and results remain local for authorized group review.

### Later ambiguity and prior ranking

The unregularized native reference fails its frozen-budget convergence certificate and independent consistency gates, remaining inconclusive rather than a certified minimum-norm solution or exact nullspace floor. A later fresh physical replay of an identical sealed fixed field independently confirms that large 3D differences can coexist with nearly identical 2D observations. This is a post-open finite-pair identifiability diagnosis, not a universal error bound for every algorithm, and no experimental noise level is assumed.

A truth-visible fixed-pair audit also confirms that isotropic TV favors the true field while a squared-gradient penalty favors the wrong field. This ranking fits no parameters and computes no new inverse solution; it proves neither uniqueness nor successful TV reconstruction. TV already has [BOS flame-tomography precedents](https://www.sciencedirect.com/science/article/pii/S0010218018302694) and is not a first-method contribution here.

The separately frozen classical TV solve under clean observation constraints is now complete. Finite iterates from an established solver library and a separate recurrence, geometry and gradient implementation pass internal independent consistency checks and improve first-frame 3D accuracy. Fresh feasibility and fixed-point convergence certificates nevertheless fail, leaving the authoritative reference verdict inconclusive. Absolute accuracy is not reference certification, and an expensive classical solve is not a cheap initializer. No full-sequence escalation, learning, speed, external generalization or real-BOST claim is authorized; iteration depth, certificates and target fields are not revised after results.

### Post-open fixed initial state and exact refinement

Unchanged CGLS from the sealed finite TV initial states independently improves observation fit without harming their existing 3D accuracy. The same short zero-start control fails the joint accuracy gate. This supports compatibility of a structured initial state with unchanged refinement, not blind confirmation, a qualified reference or equal-total-cost algorithm advantage.

A fresh adjoint witness covers only the refinement increment: it does not establish that the complete nonlinear TV initial state can be produced by exact lift. Historical costs of the expensive TV state are retained, and better observation fit does not revise convergence certification. The next substantive requirements remain a qualified reference, full-initializer representability and a fair comparison of genuinely cheap observation/geometry-only initialization. Training, full-sequence escalation, unopened data and speed claims remain unauthorized.

### Conditional adjoint representation limit

An internal independent witness supplies a necessary field-error bound: for the frozen native discretization and a preregistered finite dual norm, the pure-adjoint field family cannot meet the unchanged field-accuracy gate. A known adjoint-representable control gives no false positive. The failed local dictionary remains inconclusive; agreement when replaying an identical witness does not revise that failed dictionary audit.

This closes only that bounded representation family. It establishes neither an exact nullspace, unbounded-dual impossibility, continuous optical certification nor failure of the C route. General PCGLS may change the original adjoint range and is not excluded; the accumulated terminal dual norm of ordinary CGLS is also not measured by this bound. No transfer to PoolFire, another grid, a full sequence or real BOST follows. A larger predictor cannot alter a fixed representation's capacity, but no predictor was trained. Private condition values, numerical results and figures remain unpublished.

### Main PoolFire identical input normal action

A separate preregistered replay of every existing sentinel uses identical states, observations and selected parameters in both physical implementations. It independently checks forward, adjoint and normal actions, scalar measures and stratum tails. All checks pass; neither K128 arm meets the inherited full-space stationarity gate. This separates identical-input operator agreement from finite-Krylov reference questions, without replacing the original failed independent comparison. No new inverse solve, CFD truth reading or unopened input access was performed.

The [hybrid projection survey](https://arxiv.org/html/2105.07221v2) distinguishes early-stopped iterative regularization from full-space optimization. The implication here is that nonstationarity is not an accuracy veto for every finite-budget reconstruction. Future comparisons must preregister whether they target a qualified optimization reference or finite application accuracy; an old failed reference cannot be relabeled as success. There is still no learned algorithm, equal-cost advantage, resource reduction, external gate or real BOST result.

### Main-route full lift of fixed targets

A subsequently preregistered main-route representation audit passes. Two implementations separately rebuild unchanged directions and the triangular adjoint relation, generate offline coefficients for identical fixed finite targets, then freshly replay physical adjoint and forward actions. Every sentinel target meets the frozen reproduction and finite-norm-envelope gates. This proves fixed-target representability, not CFD-truth recoverability, full-sequence coverage, permutation testing or new-camera generalization.

Coefficient construction reads the targets and repeats expensive direction generation. All offline and replay calls remain accounted for; a single adjoint application is not a cheap online algorithm. There is no training, matched-accuracy call saving or resource result. Old reference failures remain unchanged. Prediction needs a separate target policy, complete-trajectory isolation and fair classical controls; this certificate alone authorizes no learning, and a different native proxy's conditional obstruction cannot be transferred to PoolFire.
