# 暖启动实际成本与缺失视角误差 / Warm Actual Cost and Omitted-View Error

2026-09-08

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

First four-metric1% hits and sustained hits agree. Every model-correction point is beaten by cheaper controls in both action counts; four tie BP correction, while midpoint4 costs one extra forward and adjoint. This does not disprove all residual learning or reproduce HINTS. These are midpoints of opened trajectories, not five complete sequences or independent new conditions. One-step differences between recurrences remain intervals. Crossings are truth-visible retrospective ideal stopping times, not deployable stops.

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
