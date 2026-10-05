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

### 主线小模型学习哨兵判决

随后另行结果前冻结的小模型实验，按完整轨迹隔离训练与评分，仅读取部署可见观测、报告几何和精确正规作用来预测对偶初始化，再接原样 CGLS。两套实现分别构建特征、训练、提升、物理重放与评分，内部独立检查一致。

固定小模型守住基本绝对精度门，但未达到固定有限步算法对照的四指标精度等价门。同输入线性和便宜经典对照也未通过该精度等价门，因此不是以已通过的便宜对照否决模型。结果关闭这个固定模型，不扩完整序列、修改训练配方或用更大网络救援。此次是已开封训练数据的学习哨兵否决，不是未开封泛化、噪声鲁棒性或真实 BOST 验证。

先前表示可行性通过仍有效，旧优化参考失败仍保留。能表达固定目标不等于能廉价预测；基本重建精度不等于同精度加速。本次没有证明整个方向不可能，也没有有效调用、时间或内存节省。训练参数、私有数值、源码和图表不公开。

### 同算子张量初值与更简单的有限步基线

另行冻结的静态三维张量表示对照已经完成：每个样本只用当前观测优化，随后接原样 CGLS。部分空间误差改善，但四指标精度等价门未过，优化与细化的总调用还高于普通 CGLS。普通 CGLS 在更低预算下守住了全部已测哨兵的有限目标匹配与基本绝对精度门，提供了更简单的应用基线；不是驻点认证或完整序列结论。

两套优化实现使用共同冻结的离散矩阵输入，独立重建几何与物理算子对最终场重放。求解全过程并非独立生成算子系数。完整状态先于真值评分封存；坐标读取接口错误只接续缺失评分，没有重训、重跑细化或修改门。原分离算子压力测试失败与旧优化参考失败均保留。

关闭这个固定简化方案，不调模型规模或优化深度救援。它不是完整 TDBOST 复现，不否定师兄原算法，也不证明 C 路线不可能。没有跨轨迹学习、完整序列、外门、真实实验或资源成果；私有输入、参数、数值与图表不发布。

### 有限积分盒导数补偿初值

一个结果前固定的经典初值对照已完成：只用当前观测、已知几何与便宜 BP 状态估计有限积分盒和离散标量投影导数的差异，做固定软滤波与精确伴随提升，再接原样 CGLS。独立复算确认基础精度通过，但四指标同精度失败；在每个已检验哨兵上，便宜 BP 初值和普通 CGLS 对照的四项误差都更低。补偿优于单独滤波的部分样本不构成加速，关闭这个固定机制，不修改滤波、边界或预算救援。

两套实现使用共同冻结的离散矩阵作为求解输入，并由独立重建的物理算子重放最终场，全部状态先于真值评分封存。合成核验中的光滑边界恒等式不适用于截断插值分支；单独解析检查解释了差异，原失败记录保留，候选和科学门不变。额外导数作用与准备成本不免费。这不是学习模型、精确无旋约束、完整序列、外部泛化、真实 BOST 或资源优势，也不证明整个 C 路线不可能。

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

### Main-route small-model learning sentinel

A subsequently preregistered small-model experiment isolates complete trajectories for fitting and scoring. Only deployment-visible observations, reported geometry and an exact normal action predict a dual initializer before unchanged CGLS. Two implementations separately reconstruct features, train, lift, physically replay and score, with matching internal independent checks.

The fixed small model meets basic absolute accuracy but fails four-metric matched accuracy against the fixed finite-iterate algorithm comparator. Same-input linear and cheaper classical controls also fail that matched gate, so this is not rejection through an already passing cheap control. This fixed model closes without full-sequence escalation, training-recipe revision or larger-network rescue. It is a learning-sentinel veto on opened training data, not unopened generalization, noise robustness or real BOST.

Earlier representability remains valid, and old optimization-reference failures remain unchanged. Representing a fixed target does not establish cheap prediction; basic reconstruction accuracy is not matched-accuracy acceleration. This establishes neither impossibility of the entire direction nor valid call, time or memory saving. Trained parameters, private numerical results, source and figures are not released.

### Same operator tensor initializer and simpler finite baseline

A separately preregistered static 3D tensor-field control is complete. Each instance is optimized only against its current observation before unchanged CGLS. Some spatial errors improve, but four-metric matched accuracy fails and total optimization plus refinement calls exceed plain CGLS. Plain CGLS matches the finite target and basic absolute gates on every tested sentinel at lower budget, providing a simpler application baseline, not a stationary reference or full-sequence result.

Two optimizer implementations use a shared frozen discrete matrix input. Independently reconstructed geometry and physical operators replay every final field; operator coefficients are not independently generated throughout each solve. Complete states are sealed before truth scoring. A coordinate-access API failure resumes only missing scoring, without fitting, refinement reruns or gate changes. The original split-physics stress-test failure and old optimization-reference failures remain unchanged.

This fixed adaptation closes without model-size or optimizer-depth rescue. It is not full TDBOST reproduction, does not disprove the original algorithm and is not an impossibility proof for the C route. There is no cross-trajectory learning, full sequence, external gate, real experiment or resource result. Private inputs, parameters, numerical values and figures are not released.

### Finite box derivative compensation initializer

A preregistered classical initializer control is complete. It uses only the current observation, known geometry and a cheap BP state to estimate the finite-box and discrete scalar-projection derivative defect, then applies a fixed soft filter, exact adjoint lift and unchanged CGLS. Independent recomputation confirms basic accuracy but fails four-metric matched accuracy. On every tested sentinel, cheaper BP initialization and plain CGLS controls have lower errors in all four metrics. Improvement over the filter-only arm on some samples is not acceleration. This fixed mechanism closes without filter, boundary or budget rescue.

Two implementations use a shared frozen matrix for solves, independently reconstructed physics replays every final field and all states are sealed before truth scoring. A synthetic smooth-boundary identity does not hold for the clipped interpolation branch; separate analytic checks explain the difference while preserving the failed record and leaving the candidate and scientific gates unchanged. Additional derivative actions and setup are not free. This is not a learned model, exact curl-free constraint, full sequence, external generalization, real BOST or resource advantage, and does not establish impossibility of the C route.

### 完整相机块初值

完整相机双分量几何伪逆、精确伴随提升和原样CGLS的经典对照已独立封存。相对于BP与逐行归一化初值，细化后每个已检验训练哨兵的四项误差均更低；但仍未达到高预算有限基线的四指标同精度门。因此局部改善不是加速成功，固定机制关闭，不调截断、分块、阻尼或步数救援。

两套实现分别构造几何因子，以共同冻结的矩阵求解，独立重建物理算子重放全部终点，完整状态先于真值评分封存。合成病态矩阵的有限步细化失败保留，结果前控制诊断解释浮点敏感性，不取消实际样本的原数值和物理门。几何分解、缓存和块作用并非免费；没有学习、完整序列、外部泛化、噪声标定鲁棒性、真实BOST或时间内存优势。私有数值、源码、因子和参数不公开。

### Full camera block initializer

The independently sealed classical control combines the full within-camera two-component geometry pseudoinverse, exact adjoint lift and unchanged CGLS. After refinement, all four errors are lower than BP and row-energy warm starts on every tested training sentinel. Nevertheless, four-metric matched accuracy against the higher-budget finite baseline fails. Local improvement is not acceleration; the fixed mechanism closes without cutoff, block, damping or depth rescue.

Two implementations construct geometry factors independently, solve with a shared frozen matrix and replay every endpoint through independently reconstructed physics. All states seal before truth scoring. The ill-conditioned synthetic finite-step refinement failures remain failed; pre-science controlled diagnosis explains floating-point sensitivity without exempting actual-data numerical or physical gates. Factorization, cache storage and block actions are not free. No learned, full-sequence, external, noise/calibration-robustness, real BOST or resource result follows. Private numerical results, source, factors and parameters are not released.

This is an existing classical block projection idea, not a new invention. See [Sorensen and Hansen, block reconstruction methods](https://backend.orbit.dtu.dk/ws/files/127276212/BlockAIRv3.pdf). That paper does not establish BOST acceleration in this experiment.

### 10月6日 信息审计与输入舍入

受控信息审计仍未通过完整教师数值一致性门，结论保留为不确定。单独冻结的交叉输入对照使用全部已检验训练哨兵：两套原样教师代码在相同精确输入上输出一致；九相机受控样本的极小输入舍入差异却放大到原一致性门之外。交叉状态先封存，独立汇总核验原因，不改原容差、扰动或目标。

该诊断不能证明局部特征不足、三维重建不可能或旧模型失败的原因；旧基本重建和同精度失败结论保留。没有新真值、测试或训练，新增投影只是共同矩阵作用而非新物理验证。全部诊断计算是离线成本，不是加速，不能据此改造已关闭模型或租GPU。私有数值、源码、系数和参数不公开。

### October 6 Information audit and input rounding

The controlled information audit remains inconclusive because complete teacher numerical closure fails. A separately frozen crossed-input comparison covers every tested training sentinel. Both unchanged teacher implementations agree on the same exact input, whereas tiny input-rounding differences in the controlled nine-camera samples amplify beyond the original consistency gate. Crossed states seal before independently recomputed diagnostic aggregation; no original tolerance, perturbation or target is changed.

This does not prove insufficient local features, impossible 3D reconstruction or the cause of the old model failure. Old basic-accuracy and matched-accuracy decisions remain unchanged. There is no new truth, test or training; new projection checks use the shared matrix rather than a newly rebuilt physical operator. All diagnostic work is offline cost, not acceleration, and cannot authorize rescue of a closed model or GPU rental. Private numerical results, source, coefficients and parameters remain unpublished.

Finite-precision CGLS/LSQR behavior is established numerical analysis, not a new invention. [Bjorck, Elfving and Strakos](https://epubs.siam.org/doi/10.1137/S089547989631202X) study attainable accuracy for particular implementations; their paper alone does not establish the cause in this proxy.

### 10月6日 观测前置选择学习实验

另行冻结的小模型先对当前观测做共享非线性选择，再执行精确跨相机传播与伴随提升，接原样CGLS。使用经典算法的三维输出场监督，不直接拟合任意对偶坐标。整轨迹留出的训练和独立第二实现已封存；基础精度通过，四指标同精度失败。同价后置学习对照也未通过。关闭这个固定结构与配方，不通过加宽、改目标损失或加深细化救援。

封存结果的事后描述发现：初值更接近目标，部分便宜初值的误差也改善，但近预算的普通CGLS仍更好。不是纯粹没学到，而是改善不足以形成最终同精度成本优势。后置对照参数量不同，不能把这个比较说成严格隔离的顺序因果证明。两套训练使用共同冻结矩阵，并由独立重建的物理算子重放终点；不是所有算子系数都独立生成。训练、教师、几何缓存、诊断与部署成本分开披露，没有完整时序、封存测试、噪声标定鲁棒性、时间内存优势、外部泛化或真实BOST结论。私有原始数据、协议、数值数组、源码、权重与图表不公开。

### October 6 Learning observation selection before physics propagation

A separately frozen small model applies shared nonlinear selection to the current observation before exact cross-camera propagation and adjoint lift, then unchanged CGLS. Supervision uses a classical algorithm's physical field rather than arbitrary dual coordinates. Complete-trajectory held-out training and an independent second implementation are sealed: basic accuracy passes, four-metric matched accuracy fails, and the equal-call-budget post-selection learned control also fails. This fixed architecture and recipe close without width, target/loss or refinement-depth rescue.

Post-open description finds a closer initializer and improvements over some cheap warm starts, but near-budget plain CGLS remains better. Some learning occurs, yet it does not yield final matched-accuracy cost benefit. The control has a different parameter count, so this is not a parameter-matched causal proof of ordering. Both fits use a shared frozen matrix and independently reconstructed final physical replay; operator coefficients are not independently generated throughout each solve. Training, teacher, geometry-cache, diagnostics and deployment costs remain separate. No full sequence, sealed test, noise/calibration robustness, resource speedup, external or real-BOST result follows. Private data, protocols, arrays, source, weights and figures remain unpublished.

Physics-embedded learning and variable-input equivariance have established precedents in [learned primal-dual reconstruction](https://arxiv.org/abs/1707.06474) and [VIDON](https://arxiv.org/abs/2205.11404). Neither supplies evidence of BOST acceleration in this trial.

### 10月6日 固定频谱经典对照

在同一已开封公开训练哨兵上，新增一个只依赖网格和已知几何的固定半阶预条件PCGLS对照。
它不训练参数，不改变物理算子、最小二乘目标、原精度门或迭代预算。
正式快速变换和独立直接余弦实现分别重建滤波、求解和物理重放，独立门全部通过。
基础精度通过，严格四指标同精度失败；关闭这一固定配方，不调指数、尺度、边界或深度救援。

事后描述确认，相同A/AT调用预算下，它改善不少样本相对普通CGLS或Jacobi的四项误差，
但存在退化样本，不是逐样本稳定支配。观测一致性在所有样本未过门，也不是唯一瓶颈。
这一局部信号要求后续学习模型面对更强的经典竞争，不能只靠低预算误差或场梯度汇总选择模型。
预条件作用、缓存和旧学习训练不是免费；数值验证耗时不是公平部署时延。
没有学习加速、完整时序、封存外门、资源或真实BOST结论。私有公式、合同、数组、源码与图表不发布。

### October 6 Fixed spectral classical control

A fixed half-order auxiliary preconditioner adds a stronger classical PCGLS control on the same opened public train sentinels.
It reads only the grid and known geometry, with zero new fitted parameters and no change to the physical operator,
least-squares objective, accuracy gates or iteration budget. Fast transforms and separate direct-cosine matrices
independently reconstruct the filter, solver and physical replay; all independent checks pass.
Basic accuracy passes but strict four-metric matched accuracy fails. The exact fixed recipe closes without exponent,
scale, boundary or depth rescue. This is neither a return to the closed older dataset route nor a regularization retune.

Post-open descriptions agree across both implementations: at the same A/AT budget, many samples improve over
plain CGLS or Jacobi, but some regress, so there is no uniform dominance. Observation consistency fails in every
sample without being the only bottleneck. Future learning must face this stronger classical competition;
pooled field/gradient improvement is not a replacement for four per-cell gates.
Preconditioner applications, caches and prior learning are not free, and validator elapsed time is not deployment latency.
No learned acceleration, full-sequence, sealed external, resource or real-BOST conclusion follows.
Private formulas, protocols, arrays, source and figures remain unpublished.

### 10月6日 观测拟合与三维场精度分离

同一已开封公开训练哨兵上的固定AMG经典对照现已独立封存。
不训练模型，不改变物理数据项；严格四指标同精度失败，基础分层只有部分通过。
相比相同算子调用预算的普通CGLS，全部观测拟合更好，却有多数三维场更差。
该信号说明残差下降不能代替三维物理精度；不证明某一种零空间或层级构造是唯一根因。
关闭这一个固定配方，不修改层级、平滑或深度救援，也不关闭整个C路线。

两套实现分别构造层级和细化，以共同冻结矩阵求解，独立重建物理算子重放全部终点。
原独立时限中断保留，只复用已完成前缀并按原科学合同接续剩余状态。
后续模块导入和元数据写入顺序错误同样保留；不放宽门，不重置单次凭证，不重复完整计算。
缓存正规矩阵作用、几何层级、中断和重复准备工作、旧训练均不是免费。
独立运行耗时不能充当公平部署时延。只有三帧哨兵，没有完整时序、噪声标定扰动、
封存测试、外部、真实BOST或学习加速结论；高水平论文目标仍未达到。
已有可表示性与基本经典重建证据保留。私有数值数组、公式、协议、源码、因子和图表不发布。

### October 6 Observation fit versus 3D field accuracy

The fixed AMG classical control is independently sealed on the same opened public train sentinels.
It trains no model and changes no physical data term. Strict four-metric matched accuracy fails;
only some basic strata pass. Every observation fit improves over plain CGLS at the same operator-call budget,
yet most 3D field errors worsen. A lower residual is not sufficient for physical accuracy.
This does not identify a unique nullspace or hierarchy failure cause. The exact fixed recipe closes
without hierarchy, smoothing or depth rescue, not the whole C route.

Both implementations separately build hierarchy and refinement, solve with a shared frozen matrix
and replay every endpoint through independently rebuilt physics. The original independent time-limit
failure remains preserved; only completed prefixes are reused and remaining states continue under
the unchanged science contract. Import and metadata-output-order failures also remain preserved.
No accuracy gate is relaxed, single-use receipt reset or complete computation repeated.
Cached normal actions, geometry hierarchies, interrupted and duplicate setup work and prior training
are not free. Validator elapsed time is not deployment latency. This is three-frame sentinel evidence only,
not learned acceleration, full-sequence, noise/calibration robustness, sealed-test, external, real-BOST,
resource speedup or paper success. Earlier representability and basic reconstruction evidence remain valid.
Private arrays, formulas, protocols, source, factors and figures remain unpublished.

### 10月6日 固定方向本身缺少场精度容量

后续解析容量审计已独立封存，13项检查全真。在同一99个已开封训练哨兵上，
即使允许真值可见的最优标量选择，沿这条固定经典状态修正方向也无法达到原场精度门。
因此不是仅仅在观测与场的最优系数之间折中不够好；该方向的场精度容量本身不足。
观测单项控制也未达到四指标同精度。不训练这一固定方向的标量选择器。
这只关闭当前方向，不证明全部学习初始化、三维BOST或论文可能性为假。
它是事后容量诊断，不是部署算法、完整时序、真实实验或算力突破。
下一阶段要改变有物理依据的表示，而不是继续给已排除的方向调系数。

### October 6 The fixed direction lacks field-accuracy capacity

The analytic capacity audit is independently sealed with all 13 checks passing. On the same 99
opened train sentinels, even truth-aware optimal scalar selection along this fixed classical-state
contrast cannot meet the original field gate. Failure is not merely a compromise between observation
and field optimal coefficients: the direction itself lacks field-accuracy capacity. The observation-only
control also misses four-metric matched accuracy. No scalar selector for this fixed direction is trained.
This closes only the current direction, not all learned initialization, 3D BOST or paper potential.
It is post-open attribution, not a deployable algorithm, full sequence, real experiment or resource gain.
The next research question must concern a physically justified representation, not coefficient tuning
along an already excluded direction. Private arrays, protocols, source and figures remain unpublished.

### 10月6日 完整观测学习有局部收益但不够

新的非局部初值从多相机观测和报告几何预测，再精确伴随提升、原样CGLS细化。
整条留出轨迹不参加任何拟合，评价仍只包含每条三帧，共99个已开封训练哨兵。
14项独立核验通过，基础精度33/33分层通过；严格四指标同精度0/99，关闭唯一
冻结的记忆核配方，不改模型或细化预算挽救。有限经典比较不是收敛认证。

两套程序分别汇总封存误差：近似同算子调用预算下，97/99个样本的四项误差均
不高于普通CGLS，场误差中位数约低3.2%；两个样本的场或内部梯度稍有伤害。
对更强PCGLS，四项同时不高于对照65/99、同时不低于对照10/99。
线性和最近邻记忆对照也未达到严格同精度；不能宣称普遍非线性优势。

这保留有限的学习信号，不证明信息已经足够、不改变原失败门，也不否定全方向。
大型记忆库、教师标签、训练和几何准备不免费；没有fresh wall/RSS部署优势。
仅已有5/7/9相机子集和输入乱序，不是任意新位姿、完整时序、封存测试、外部或
真实BOST验证，更不是论文成熟度。私有数组、源码、协议、记忆库与权重不发布。

### October 6 Whole-observation learning helps locally but is insufficient

A new nonlocal initializer reads multi-camera observations and reported geometry before
exact adjoint lift and unchanged CGLS refinement. The entire held trajectory is excluded
from fitting; evaluation still covers only three frames per trajectory, 99 opened train
sentinels. All 14 independent checks and 33/33 basic strata pass, but strict four-metric
matched accuracy is 0/99. Close this fixed memory-kernel recipe without model or refinement
rescue. The finite classical comparator is not a stationarity certificate.

Two separate post-closure reductions agree: at a near-equal operator-call budget, all four
errors are no higher than plain CGLS in 97/99 cells, with about 3.2% lower median field error;
two cells have slight field or interior-gradient harm. Against stronger PCGLS, all-four
nonworse and nonbetter counts are 65/99 and 10/99. Linear and nearest memory controls also
miss strict matched accuracy, so universal nonlinear superiority is not established.

This retains a limited learning signal, not proof of information sufficiency, a revised
success gate or global refutation. Large memory banks, teacher labels, fitting and geometry
preparation are not free. No fresh wall/RSS deployment advantage is measured. Evidence
covers existing 5/7/9-camera subsets and input reordering, not arbitrary new poses, full
sequences, sealed tests, external generalization, real BOST or paper maturity. Private
arrays, source, protocols, banks and weights remain unpublished.

### 10月6日 训练库投影归因不是全方向证伪

后续事后归因已独立封存，11项检查通过。所有训练库方向先于留出经典目标封存；
读取该有限目标做最佳初始场投影时，相对目标误差中位数约38.3%、p90约45.2%。
它改善了此前初值误差，却仍有明显表示缺口。接相同短程细化后，严格同精度
0/99；只用观测求系数的对照也0/99，两者基础精度33/33。

因此不能单以“系数没学准”解释当前失败。但初始场最优不等于细化后四指标全局
最优，未证明所有系数选择、训练库或学习方向不可能。原固定核模型保持关闭，
不调模型、扩大训练库或加深细化来改判。目标可见诊断不是部署预测，没有新学习、
完整序列、少调用、实测速率、外部或真实BOST成果。私有数组、源码、协议和权重
不发布；基础交付与论文创新仍须分开。

### October 6 Bank projection attribution is not global refutation

A subsequent post-open audit is independently sealed with all 11 checks passing. Fit-only
bank directions seal before held classical targets are opened. Their best initial field
projection has about 38.3% median and 45.2% p90 relative error to the finite target. It improves
the earlier initializer but leaves a substantial representation gap. After unchanged short
refinement, strict matched accuracy remains 0/99; observation-only coefficient inversion
also gives 0/99. Both pass 33/33 basic strata.

Coefficient prediction error alone does not explain this result. However, optimal initial
field projection is not globally optimal over all refined four-metric paths; no impossibility
claim covers every coefficient choice, the entire bank or learning direction. The original
kernel recipe remains closed without model, bank-size or refinement-depth rescue. This
target-visible diagnostic is not deployment prediction, new learning, full-sequence evidence,
fewer calls, measured speed, external generalization or real BOST. Private arrays, source,
protocols and weights remain unpublished. Basic delivery and paper innovation stay separate.
