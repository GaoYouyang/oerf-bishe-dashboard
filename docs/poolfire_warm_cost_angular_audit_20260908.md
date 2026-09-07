# 暖启动实际成本与缺失视角误差 / Warm Actual Cost and Omitted-View Error

2026-09-08

## 最新对照：几何逆模式没有省下求解 / Latest Control: Geometry Inverse Modes Do Not Save Work

新对照只用几何生成16个逆算子模式、只用二维观测确定系数，随后执行原CGLS。独立复算后0/5点优于对照：该初始化需208至239对正向/伴随调用，零初始化需125至167对；未滤波随机模式也未带来优势。固定配方已关闭，不增加维数或更换种子。已有505样本的求解度量收益仍保留，但这不是暖启动、速度或论文成功。

A new control generates 16 inverse-operator modes from geometry alone, chooses coefficients from 2D observations, then runs unchanged CGLS. Independent recomputation finds 0/5 wins: it needs 208-239 forward/adjoint pairs, versus 125-167 for zero initialization; raw random modes also provide no advantage. The fixed recipe is closed without rank or seed changes. The separate 505-sample solver-metric benefit remains, but this is not warm-start, speed or paper success.

| 旧中点 / Midpoint | 几何逆模式 / Inverse modes | 原始随机模式 / Raw modes | 零初始 / Zero | 旧神经暖初始 / Old neural |
|---|---:|---:|---:|---:|
| 1 | 217 | 156 | 140 | 140 |
| 2 | 208 | 143 | 135-136 | 135 |
| 3 | 210 | 138-139 | 129-130 | 130 |
| 4 | 239 | 194 | 167 | 167 |
| 5 | 218 | 137 | 125 | 125 |


每个数字同时表示A和A转置调用数，已包含暖初始化的精确lift及初始残差计算。首次与持续四项1%达标的次数相同；区间仅表示两种数值实现的差异，不是统计置信区间。这些停止点在事后通过真值评分得到，不是可部署的停止规则。

Each number is both the forward and adjoint count, including the warm exact lift and initial residual evaluation. First and sustained four-metric 1% crossings agree; ranges are differences between two numerical implementations, not statistical confidence intervals. These stopping points are evaluated retrospectively with truth, not deployable stopping rules.

结果前固定16个随机世界坐标探针，经过一次几何正规算子逆作用，形成主候选空间；未滤波探针构成对照。基空间不读场、误差、时间或轨迹标签，系数只读二维观测。它不同于此前从场误差训练的全局子空间，但随机子空间方法本身已有文献。两个新增方案都在五点输给零初始；逆滤波方案也输给原始探针对照。

Sixteen random world-coordinate probes were fixed before results. One inverse-normal geometry action generates the primary span; unfiltered probes form the control. Neither field, error, time nor trajectory labels enter the span, and coefficients use only2D observations. This differs from the earlier field-error-trained global space, but randomized subspace methods are established literature. Both new arms lose to zero at all five points; inverse filtering also loses to raw probes.

全部5140个新状态先封存，再独立进行四指标评分和原生正向重放；评分最大相对差2.32e-15，重放差7.08e-16，初始场独立差9.81e-12。实际新增递推为5140A+5120A转置，初始lift为20A转置；几何构造与核验160A+96A转置、192个完整因子向量三角求解；初始化核验20A+20A转置；评分5140A+5140原生A+10一致性A。完整几何分解的继承成本并不免费；系数与dual读取还需16方向稠密运算。直接输出同一缓存场在数学上等价且可少一次A转置，因此没有dual编码自身的优势。

All 5140 new states were sealed before independent four-metric scoring and native forward replay. Maximum relative discrepancies are 2.32e-15 for scoring, 7.08e-16 for replay and 9.81e-12 for independent initials. New recurrences consume 5140forward/5120adjoint actions; initialization 20adjoints; geometry construction/audits 160forward/96adjoints and 192 full-factor vector triangular solves; initial audits 20forward/20adjoints; scoring 5140forward/5140native-forward/10consistency actions. Inherited full geometry factorization is nonfree, and readout requires dense 16-direction work. Direct output of the same cached field is mathematically equivalent and avoids one adjoint, so there is no intrinsic dual-encoding advantage.

这只关闭固定16模式、一次逆作用、一次观测投影的方案，不证明所有几何表示、合成误差训练或暖启动都不可能。没有训练新模型、扩跑完整轨迹或打开新数据，也没有真实BOST、端到端提速或论文成功。已有505样本的学习求解度量结果单独保留。当前数据足够继续有边界的虚拟研究，不需要为了此失败租GPU或追加数据。

Only the fixed 16-mode/one-inverse/one-observation-projection recipe is closed, not all geometry representations, synthetic-error training or warm starts. No new model, full-trajectory expansion or new data was used, and no real-BOST, end-to-end speed or paper success is established. The separate 505-sample learned solver-metric result remains. Existing data suffice for bounded virtual research; this failure does not require GPU rental or additional data.

方法背景 / Method context: [Randomized subspace iteration](https://arxiv.org/abs/1408.2208); [canonical-angle analysis](https://doi.org/10.1137/18M1179432). 文献背景不是本实验性能或首创证明 / Literature is neither performance evidence for this experiment nor an originality claim.

此前独立结论与反事实图保留在下方 / Earlier independent conclusions and counterfactual figure remain below.


## 最新诊断：相同误差大小，不同细化难度 / Latest: Equal Error Size, Different Refinement Difficulty

新诊断把误差大小与方向分开：在五个已打开中点，把初始场误差调到相同范数后，旧暖启动的误差方向仍需多38至50步才满足原四项1%精度。独立复算说明瓶颈不只是初始误差大小。这是需要完整参考解的离线反事实诊断，不是可部署暖启动、实际调用节省或速度突破。

A new diagnosis separates error size from direction: at five opened midpoints, with initial field-error norms matched, the old warm error direction needs38-50 more refinements to meet the original four1% accuracy gates. Independent recomputation shows that initial error size is not the only issue. These offline counterfactuals require a full reference; they are not deployable warm starts, achieved call savings or a speed breakthrough.

![Offline counterfactual refinement comparison](../assets/figures/poolfire_warm_error_shape_20260908.png)

| 旧中点 / Midpoint | 零方向、暖范数 / Cold direction, warm norm | 原暖启动 / Original warm | 原零初始 / Original zero | 暖方向、零范数 / Warm direction, zero norm |
|---|---:|---:|---:|---:|
| 1 | 101 | 139 | 140 | 178 |
| 2 | 91 | 134 | 135-136 | 176-177 |
| 3 | 81 | 129 | 129-130 | 173 |
| 4 | 116-117 | 166 | 167 | 212 |
| 5 | 84 | 124 | 125 | 165 |


表格是细化步数，不是在线A/A转置成本。用已有观测求得的合格完整参考场t和冻结暖初始w，令a=||t-w||/||t||。两个额外离线初始场为(1-a)t和t-(t-w)/a，分别保持零初始、暖初始的误差方向，并把误差范数调至另一组。它们事先已经需要完整解，不能拿来宣称省下求解成本。原两个对照沿用封存轨迹，额外两组真实执行未改动的CGLS细化，不用缩放旧曲线冒充重算。

The table lists refinement steps, not online forward/adjoint cost. From the qualified full observation-derived reference t and frozen warm initial w, set a=||t-w||/||t||. Two additional offline initials, (1-a)t and t-(t-w)/a, retain zero-start and warm error directions while matching the other error norm. They already require the full solution and cannot claim to save its cost. The two original controls reuse sealed curves; both new arms actually run unchanged CGLS refinement rather than rescaling old curves as simulated computation.

暖初始误差的L2范数是零初始的39.09%至46.01%，不是能量比例。把范数固定在暖初始层级，保留零初始误差方向的对照比原暖初始少38至50步；固定在零初始层级，暖误差方向多38至45步。两层级、五点、首次与持续达标均一致。只看field的预注册次要结果也呈现同向差异，因此不是只有梯度验收条件造成的现象；但它不替换四指标主判据。

The warm initial L2 error norm is39.09%-46.01% of zero's norm, not an energy fraction. At the warm norm, retaining zero's error direction needs38-50 fewer refinements than the original warm state; at the zero norm, warm direction needs38-45 more. Both norm levels, all five points, and first/sustained hits agree. Preregistered secondary field-only crossings show the same directional effect, so it is not solely the gradient acceptance criterion; that secondary result does not replace the four-metric primary.

独立执行20条额外数值轨迹，共5140状态；全体状态先封存再读取CFD真值评分。独立范数差1.42e-16，稀疏/网格评分差8.68e-15，原生重放差7.11e-16。新增递推5140A+5120A转置，度量2570F+2560F转置+5120T；构造验证20A、10原生A、10A转置；评分5140A、5140原生A和10一致性A，均为离线诊断成本。完整参考、模型训练与缓存不免费。

Twenty additional numerical paths produce5140 states, all sealed before CFD-truth scoring. Independent norm discrepancy is1.42e-16, sparse/grid scoring8.68e-15, and native replay7.11e-16. New recurrences consume 5140A+5120adjoints and metric2570F+2560F-transpose+5120T. Construction validation uses20A,10nativeA,10adjoints; scoring 5140A,5140nativeA and10consistencyA. All are offline diagnostic costs; full reference, training and caches are nonfree.

该受控干预支持：这套初始化的剩余误差形状抵消了大部分幅度改善收益。它只适用于这五个已打开中点和冻结求解器，不是完整轨迹或外部验证；没有识别专属低频/近零空间模式，也没有证明某种新损失或模型一定有效。后续优先审查如何改变慢收敛误差结构，而不盲目降低初始L2、加大模型或重训旧配方。没有新算法突破、实际提速、真实BOST或论文成功。

This controlled intervention supports that this initializer's remaining error shape offsets much of its amplitude-reduction benefit. It is conditional on five opened midpoints and the frozen solver, not complete-trajectory or external evidence. It does not identify exclusive low-frequency/near-null modes or prove a new loss/model will work. Next prioritize how to change slow error structure, not blindly reduce initial L2, enlarge models or refit old recipes. No new algorithm breakthrough, achieved speedup, real-BOST or paper success.

方法背景而非本实验结论来源 / Method context, not evidence of this experiment's outcome: [CGLS, Fong thesis Algorithm1.7](https://web.stanford.edu/group/SOL/dissertations/david-fong-thesis-online.pdf); [Axelsson on initial error and convergence phases](https://doi.org/10.1016/S0378-4754(02)00097-6).

下方保留此前封存结果 / Previously sealed results remain below.


## 新增：一次残差复用 / Addition: One-Shot Residual Reuse

残差复用检验也未带来调用收益：固定16步后，把旧小模型应用于当前残差，只校正一次再继续原CGLS，五个已打开中点均未胜过便宜对照。独立复算已封存；不扩跑、不调深度或放大旧模型。它只关闭这套不重训的复用配方，不否定全部残差学习。既有固定九相机学习度量收益保留，但暖启动、资源与论文成功仍未成立。

Residual reuse also gives no call savings: after a fixed16-step prefix, the old small model corrects the current residual once before unchanged CGLS resumes. It fails against cheaper controls at all five opened midpoints, with independent recomputation sealed. No expansion, depth tuning or model enlargement. This closes only the no-refit reuse recipe, not all residual learning. Existing fixed-nine-camera learned-metric evidence remains; warm, resource and paper success remain unproved.

| 中点 / Midpoint | 残差模型 / Residual model | BP校正 / BP correction | T校正 / T correction | 零初始 / Zero | 旧暖启动 / Old warm | K4 ridge |
|---|---|---|---|---|---|---|
| 1 | 143, 142 | 143, 142 | 142, 141 | 140, 140 | 140, 140 | 140, 140 |
| 2 | 139, 138 | 139, 138 | 138, 137 | 135-136, 135-136 | 135, 135 | 136, 136 |
| 3 | 133, 132 | 133, 132 | 132, 131 | 129-130, 129-130 | 130, 130 | 131, 131 |
| 4 | 171, 170 | 170, 169 | 169, 168 | 167, 167 | 167, 167 | 167-168, 167-168 |
| 5 | 129, 128 | 129, 128 | 127-128, 126-127 | 125, 125 | 125, 125 | 126, 126 |


每格为(A调用数, A转置调用数)，不是求和或迭代数。模型不重训、不改参数；固定16步零初始迭代后，用原小模型读取残差和已知几何，经精确伴随lift与观测线搜索只校正一次，再继续未修改CGLS。BP与T校正使用相同前缀和预算；另复用零初始、旧暖启动和K4 ridge封存对照。三个新方案上限为256A+255A转置，旧对照比较窗口截至256A；前缀、lift、线搜索投影和初始重放全部计费，不把线下共享前缀当免费部署计算。

Each cell lists (A calls, adjoint calls), not their sum or an iteration count. Without refitting or changing parameters, the old small model reads residual and known geometry after a fixed16-step zero-start prefix, supplies one exact-adjoint correction with an observation line search, then unchanged CGLS resumes. BP and T corrections share the same prefix and budget. Zero, old-warm and K4-ridge controls reuse sealed curves. New arms end at256A+255adjoints; reused controls are compared through256A. Prefix, lift, line-search projection and initial replay are charged; offline prefix sharing is not free deployment computation.

首次四项1%达标与持续达标结论一致。五个模型校正点全部被便宜对照以更少或相等的两种调用击败；四点与BP校正同价，第四点还多一次A和A转置。该结果不能证明所有残差学习无效，也不是HINTS复现。五点仍是已开封轨迹的中点，而非五条完整序列或独立新工况。两种递推的一步差异保留为区间；这些是真值可见的事后理想停止时刻，不是部署停止规则。

First four-metric 1% hits and sustained hits agree. Every model-correction point is beaten by cheaper controls in both action counts; four tie BP correction, while midpoint4 costs one extra forward and adjoint. This does not disprove all residual learning or reproduce HINTS. These are midpoints of opened trajectories, not five complete sequences or independent new conditions. One-step differences between recurrences remain intervals. Crossings are truth-visible retrospective ideal stopping times, not deployable stops.

全部7710个状态先封存再读真值，逐状态稀疏与网格导数独立评分最大相对差2.87e-15，原生物理重放差7.21e-16，相机逆序差2.46e-16。新增递推7360A+7330A转置；度量应用3670F+3650F转置+7310T另计；离线评分7710A、原生重放7710A、一致性10A另计。预测、训练和几何缓存均不免费。没有fresh wall/RSS、可变相机预测、外部、真实BOST或论文成功。

All7710 states sealed before truth scoring. Independent sparse/grid scoring differs by at most2.87e-15 relatively, native replay by7.21e-16 and reversed-camera inference by2.46e-16. New recurrences consume7360A+7330adjoints; metric work3670F+3650F-transpose+7310T is separate. Offline scoring7710A, native replay7710A and consistency10A are separate. Inference, training and geometry caches are nonfree. No fresh wall/RSS, variable-camera prediction, external, real-BOST or paper success.

后续不调这套复用配方的深度、预算或模型大小。既有学习度量证据保留，但不能把预条件收益称为暖启动成功。下方为此前已封存的结果，不因本次补充而重算或改写。

Do not tune this reuse recipe's depth, budget or model size. Existing learned-metric evidence remains, but preconditioning gains are not warm-start success. The prior sealed results below are retained, not rerun or rewritten by this addition.


旧非线性暖启动的实际达标成本已核清：五个已打开中点中，可靠节省调用为0/5，4个比较未过，1个因一调用区间重叠仍不确定。5920个恢复状态均经独立物理评分，旧预算和计算轨迹未变；不是只有停止证明太保守。另一个误差归因显示，减少相机后三类方法剩余误差都更容易被缺失视角看见，但未确认单一共享误差方向。两项都不是算法突破；固定九相机的既有学习度量收益保留，暖启动收益仍未成立。

The old nonlinear warm start now has actual-accuracy costs: 0/5 robust savings at five opened midpoints, four failed comparisons and one inconclusive one-call overlap. All 5920 recovered states received independent physical scoring, with the original budgets and numerical trajectories unchanged; certificate conservatism is not the only issue. A separate attribution finds remaining errors more visible from omitted cameras in all three methods, without confirming one shared error direction. Neither audit is an algorithm breakthrough. The existing fixed-nine-camera learned-metric benefit remains, while warm benefit is unproved.

![Retrospective cost and squared relative-error ratio](../assets/figures/poolfire_warm_cost_angular_audit_20260908.png)

## 实际成本 / Actual Cost

| 旧中点 / Old midpoint | 非线性 / Nonlinear | 零初始 / Zero | 标量 / Scalar | K4 dual-ridge |
|---|---:|---:|---:|---:|
| 1 | 140 | 140 | 141 | 140 |
| 2 | 135 | 135-136 | 136 | 136 |
| 3 | 130 | 129-130 | 130 | 131 |
| 4 | 167 | 167 | 167 | 167-168 |
| 5 | 125 | 125 | 126 | 126 |


每个数字表示A调用数与A转置调用数各为多少，不是二者之和。全部方法使用同一学习度量；三个暖启动均计初始精确lift和投影，零初始不虚计A(0)。首次达标与持续达标至第256步的结果一致。两个独立递推实现的一步差异保留为区间，不能选择有利的一条。第二中点与零初始区间重叠，所以是“不确定”，不能写成五个都失败或一个胜出。

Each number is the count of A calls and separately the equal count of adjoint calls, not their sum. All arms use the same learned metric. Warm starts pay for their initial exact lift and projection; zero does not pay a fictitious A(0). First hits and sustained hits through step256 agree. One-step differences between independent recurrences remain intervals, not a favorable path choice. Midpoint2 overlaps the zero-start interval: it is inconclusive, neither a fifth failure nor a win.

五个查询都是已打开轨迹的固定中点，各自模型训练时排除了整条查询轨迹；这不是五条完整序列或外部验证。仅恢复原来20条数值轨迹，共5920个状态，原始前三步、端点、残差逐位一致，预算、停止条件和调用记录均未改。所有状态封存后才读取CFD真值；原始初始场未单独保存，仅由同一封存dual和同一精确lift重建。独立评分最大相对差3.47e-15，原生物理投影差7.08e-16。结果为FAIL_NONLINEAR_RETROSPECTIVE_ACTUAL_COST，旧停止证明成本失败结论另行保留。

All five queries are fixed midpoints of opened trajectories; each model excluded its query's complete trajectory during training. This is not full-sequence or external validation. Only the original20 numerical paths were recovered, totaling5920 states, with bitwise-identical first three states, endpoints and residuals, and unchanged budgets, stops and ledgers. States sealed before CFD-truth scoring. The old initial field was not separately stored; it is rebuilt by the same sealed dual and exact lift. Independent scoring differs by at most3.47e-15 relatively and native projection by7.08e-16. The result is FAIL_NONLINEAR_RETROSPECTIVE_ACTUAL_COST; the old certificate-cost rejection remains separate.

这些调用数是依赖真值的事后理想停止时刻，不是可部署停止规则。度量应用、预测、训练、几何缓存和停止常数成本均非免费。完整缓存直接解仍未被击败，不能声称端到端速度、内存或算力突破。本轮不授权重训、扩跑或放大旧模型。

These are truth-visible retrospective ideal crossing costs, not a deployable stopping rule. Metric applications, inference, training, geometry caches and certificate constants are nonfree. Full cached direct remains unbeaten; there is no end-to-end speed, memory or compute breakthrough. No refit, expansion or enlargement of this old model is authorized.

## 缺失视角 / Omitted Views

另一个独立诊断只读取减少相机实验的60个封存端点和由现有观测求得的直接参考场，不读取CFD真值。对误差e=参考场-端点，以及参考场t，比较缺失相机与保留相机的每相机平均投影能量，并用t的同类能量比归一化。所得E是“缺失视角相对观测误差 / 保留视角相对观测误差”的平方，不是场误差倍数或振幅倍数。保留视角残差接近零会放大比值。

A separate audit reads only60 sealed camera-removal endpoints and direct reference fields derived from available observations, without CFD truth. For error e=reference minus endpoint and reference t, it compares per-camera mean projection energies in omitted versus retained views and normalizes by the analogous ratio for t. E is the SQUARED ratio of omitted-view to retained-view relative observation errors, not a field-error multiplier or amplitude ratio. Near-zero retained-view residuals amplify this ratio.

普通CGLS、Jacobi和学习度量均为20/20端点E>1，中位数分别约90059、89988、35598。这些端点包含两种数值实现，不是60个独立工况。三类方法都留下偏向缺失视角可见的误差，但学习方法相对两类经典误差的有符号余弦仅0.165至0.612，40个比较均未达到预注册0.9门，未确认单一共享误差方向。不同端点停止成本不同，不能用E较小宣称同价优势；也不排除多个共享几何模式。

Ordinary CGLS, Jacobi and the learned metric each have20/20 endpoints with E>1, with medians about90059,89988 and35598. Endpoints include two numerical implementations, not60 independent conditions. All three leave errors preferentially visible in omitted views. However, signed cosine between learned and classical field errors ranges from0.165 to0.612; none of40 comparisons reaches the preregistered0.9 gate, so one shared error direction is not confirmed. Different endpoint stopping costs prevent interpreting smaller E as matched-cost superiority; multiple shared geometry modes remain possible.

两个forward实现、独立标量归约与原生重放一致，最大角度统计差3.00e-15；零估计解析对照E=1。正式、独立和原生审计各80次A、零次A转置，全部属于离线诊断，不是新算法成本。

Two forward implementations, independent scalar reductions and native replay agree; maximum angular-summary discrepancy is3.00e-15. The analytic zero-estimate control gives E=1. Formal, independent and native audits each use80 forward actions and zero adjoints, all offline diagnostic work, not a new algorithm cost.

## 结论边界 / Limits

固定九相机完整505帧的既有学习度量收益不被改写，但它不等于暖启动有效。本轮关闭旧非线性初始化的实际成本补充门；减少相机的既有失败也保留。不改预算、门或模型大小挽救。后续优先审查物理上不同的观测可见初始方向，并排除只是冷启动Krylov一步的改名。没有新的外部泛化、真实BOST、资源优势或论文成功结论。

The existing full505-frame fixed-nine-camera learned-metric evidence remains, but does not establish warm-start value. This audit closes the missing actual-cost gate for the old nonlinear initializer, while the camera-removal failure also remains. No budget, threshold or model-size rescue. Next, review physically different observation-visible initial directions and exclude renamed cold Krylov steps. No new external-generalization, real-BOST, resource or paper-success claim.

[脱敏汇总 / Redacted summary](poolfire_warm_cost_angular_audit_20260908.json) · [减少相机小门 / Camera-removal pilot](poolfire_camera_subset_metric_20260908.md)
