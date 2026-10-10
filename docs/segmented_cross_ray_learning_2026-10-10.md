# 完整留轨迹学习：有小幅收益，尚无强参考加速

2026-10-10。这次不是训练内拟合或真值可见容量诊断。一个3,841参数的共享有符号跨相机射线图网络完成11个完整轨迹留出外折：每条101帧，5/7/9相机，共3,333个held查询、33个完整轨迹×相机层。

## 实际做了什么

最新信息定位直接重放33组已封存的other-fold训练均值：中心化d=CFD均值-有限teacher均值，R(v)=||Av||²/||v||²。33/33的R(d)/R(CFD均值)<1，中位**1.7981e-4**；三维差异范数/原均值范数中位**17.897%**，对应投影差异/原投影范数中位**0.2401%**。差异有明显场幅度却相对弱可观测。这不是单帧重建误差、所有teacher误差根因、精确核空间或真实噪声阈值，不证明不可识别或伴随接口不可能。

两路径各自均值、物理CSR与NumPy/Torch算术，43/43比较通过；数组/统计最大差1.49e-15/4.44e-16。每路径新增99A+0AT，无新fit、网络更新、solver步数或held真值读取。继承setup/teacher/均值拟合与验证非免费，8.52秒/790MiB是混合诊断，不是部署速度。原均值和网络的强参考FAIL不变；下一学习机制应先解释原始场弱信息的合法稳定作用，不继续扩旧网络，也不以有限teacher或投影拟合好代替三维学习。

10月11日新增一个不同的信息源对照：每个完整留轨迹外折只用其他十条全部帧，分别拟合原始CFD和有限CGLS128的三维场均值。相同观测范数归一化、当前观测精确K1校正与原样CGLS32，共3333查询/33完整层；无held标签、统计或选择。两种先验均基本质量33/33，强参考匹配0/3333、0/33。

原始CFD先验相对冷CGLS35的场/全梯度/内部梯度/观测误差改善中位数14.3834%/13.6394%/12.2175%/18.3361%；四项同时不差3105/3333，伤害228。相对有限teacher先验，场改善9.1331%，四项不差2742/3333，并非处处更好。对Jacobi/dual-ridge四项不差2411/3333和2736/3333。这些是封存后的描述性配对，不替代预注册失败。简单先验的信息来源值得继续辨清，但不是同容量神经网络比较。

两路径独立构造先验、求解与CFD物理评分，114/114比较通过；先验/场/物理图像相对差8.23e-16/6.96e-15/2.48e-14，指标/汇总绝对差1.09e-14/2.93e-14。**显式训练场先验加exact A^T校正不是pure A^T-dual初值，不改变既有严格range-only合同，也没有分離原生精确核空间。** 三个已知相机集合缓存不是可变基数神经算子，22先验拟合/实现，每个先验每折35547系数。两路径共享canonical输入、编排和数值库；不是独立数据或神经重训。

每查询35A+34AT，与BP-Warm33等账；冷/Jacobi/dual35多一个AT，不能误称逐项更便宜。每审计路径含双臂预测/物理评分/lift共243309A+233310AT，继承35547几何setup等价投影与2旧factory A、先验拟合、CFD和teacher构造均非免费。230.85秒/1.23GiB混合审计不是fresh部署测量。关闭字面均值/归一化/K1/Warm32配方，不调权重、深度或门救援；优先研究原始场信息如何通过合法物理接口产生稳定纠错，而非扩旧网络。主目标仍未完成。[核空间学习原论文](https://arxiv.org/abs/1806.06137)在自己的形式中使用数据一致修正，并不证明本对照的原生核空间、BOS或速度结论。

最新总体机制复核固定原11个外折权重，在全部3,333查询中仅把网络增量替换成零信号处几何门控的一阶JVP，保留原数据依赖K1、归一化、基项及原样CGLS33。它不是整个映射关于原始观测的线性化；门还依赖以前训练的权重，不是无学习几何逆。

事先冻结的90%改善保留门覆盖33完整层的四指标均值/p90/最坏值，共396比较；176项和2/33完整层通过。四指标逐层均值改善保留率中位数为86.46%/88.14%/89.12%/91.90%。大部分平均小收益可保留，但高阶观测作用对一致尾部仍有贡献；不能把2/33写成全部收益主要来自非线性，更不证明非线性不可替代。相对冷35/原非线性网络四项均不差3,296/3,333和192/3,333；强参考仍0/3,333匹配，绝对质量33/33。

两实现独立重建图/JVP/精化/物理评分，165比较一致；共享原权重与canonical输入，不是独立重训。场/图像相对差1.85e-14/6.63e-14、指标/汇总绝对差1.13e-14/5.93e-13。单查询仍35A+35AT，每审计路径含评分/lift共123321A+119988AT，继承几何与旧训练另计。199.26秒、2.78GiB混合审计与减少图作用次数均不是部署加速。这项归因到此为止，不调展开点、Taylor阶数或重训救援；下一投入仍是不同有效纠错信息与强控制，而非扩大已关闭模型。

先对同一离散梯度的三线性插值表示做分段射线积分，另行独立核验新的观测与有限CGLS128参考。随后结果前固定唯一图网络：由已知射线之间的有符号物理耦合传递消息，输入只含当前观测、精确K1残差和reported geometry；按相机等权池化，参数共享，不把相机序号、时间或轨迹标签作为特征。预测dual修正，fresh exact A^T lift后接未修改CGLS33。在线字面账35A+35A^T，几何/图缓存和网络工作另计。

每个外折只用其他十条轨迹的有限CGLS128场作teacher，330次固定CPU更新，每个训练轨迹×相机约11帧，不是全数据epoch；held轨迹teacher不进入输入、标准化、参数选择、停止或回退。全部模型先封存，再预测全部held查询，最后才读取CFD真值评分。不是独立数据集或未打开的外部门。

## 两个不能混淆的结果

**预注册主门失败：0/3,333查询、0/33完整层匹配强参考的全部四指标。** 匹配是每项误差不超过同查询有限CGLS128误差的1.01倍加1e-10；不是“三维误差只有1%”。基本绝对p90门33/33通过，不能替代强参考匹配。CGLS128本身只是合格有限参考，不是收敛逆解。

封存之后的描述性配对读回显示：相对同35A+35A^T的冷CGLS35，**3,333/3,333查询的场、全梯度、内部梯度和观测误差均不差**，使用绝对比较容差1e-10。四项配对相对改善中位数为**0.3459% / 0.3928% / 0.2604% / 3.5481%**。这支持小而一致的留轨迹学习信号，但不是新预注册成功门，也不改变主门FAIL。

更强同动作控制没有被稳定排除：相对Jacobi-PCGLS35四项不差为684/3,333；相对dual-ridge-CG35为0/3,333。网络比更便宜的BP初值+CGLS33及自身未训练结构在全部查询均不差，仍不能据此只选弱控制包装加速。

## 复算与成本边界

独立路径重新构造图、特征、NumPy网络、exact lift、求解器精化和物理评分，23/23检查通过。最大逐项指标差1.90e-14；事后配对中位数最大差2.69e-15。两条路径共享封存权重、canonical CSR、观测与有限teacher；不是独立重训或独立实验。

共3,630次训练更新，训练账14,520A+10,890A^T，另有梯度审计264A+198A^T。每条独立实现路径的全部方法共享预测账693,264A+689,931A^T，物理评分29,997A；不是单模型部署账。teacher、图构建与缓存亦不免费；1,403.18秒是完成运行的混合离线审计，不含更早启动失效的几何开销，更不是部署计时。没有fresh部署wall/RSS、native12相机、扰动门、未开封外部泛化、曲线光线或真实实验BOST结论。观测仍是线性弱偏折直线射线代理，不是标定像素位移。

## 受控噪声下的小信号是否保留

另行结果前冻结一次有限鲁棒性检查，不改变旧模型：原11个完整留轨迹外折权重在99个0/50/100帧锚点、5/7/9相机上，各接受两份相对L2半径1%的观测扰动，共198查询、66三帧层。无重训、参数选择或回退。固定半径球面扰动不是实验噪声或已知方差的独立高斯模型。

相对同动作冷CGLS35，198/198查询四项均不差且至少一项严格更好，场/全梯度/内部梯度/含噪观测误差的配对相对改善中位数为**0.3332%/0.3745%/0.2655%/3.0475%**，比较容差1e-10。相对同动作Jacobi/dual-ridge四项不差仅**39/198和0/198**；强参考匹配仍**0/198**。学习与含噪有限参考均守住66/66采样绝对门，不能替代严格匹配。

独立图、网络、原样求解、exact lift、物理投影及CFD评分42/42检查通过；最大场/投影相对差1.94e-14/6.99e-14，指标差1.59e-14。两路径共享旧权重、canonical CSR与新扰动输入，不是独立重训。每路径实际68508A+66726AT包括九个方法、新含噪参考和评分，单模型仍35A+35AT；几何、图及旧训练非免费。99.25秒、约2.542GiB为混合审计，不是部署计时。

这保留了一个有限正信号，不改变原干净FAIL、授权旧模型扩展或证明完整噪声序列、任意扰动、同精度加速及真实BOST。[统计CG研究](https://arxiv.org/abs/2406.15001)已有预测/重建与噪声相关分析；本次不移植其统计风险界或停止规则。

## 剩余误差方向：为什么小收益还不够

随后对已封存的全部3,333个最终场和物理投影，做了结果前冻结的零训练归因。令E为有限CGLS128场减冷CGLS35场，D为学习最终场减冷CGLS35场，比较R(v)=||Av||²/||v-mean(v)||²。两条路径分别读取自身已封存场/投影，独立重建梯度、内积与统计；数组/汇总最大差3.30e-12/1.88e-12。新增0A+0A^T，不重训、不重新求解，继承数据与投影成本仍非免费。

学习修正的R(D)/R(E)中位数为**34.70**；5/7/9相机分别**19.62/33.94/43.15**，3,333/3,333查询和33/33完整层中位数均大于1。场方向对齐cosine中位数**0.3483**，修正幅度约为剩余场差的**5.00%**，实际有限teacher场差范数减少中位数**1.73%**。即便teacher可见地沿这个固定最终方向自由缩放，可解释场差能量的中位数也只有**12.13%**；这不是任意初值加精化的容量上界。

这支持“修正主要作用于相对更强可观测方向，剩余有限参考差更弱可观测”的限定解释，**不是完整特征谱分析、CFD真实误差判决或全部失败根因证明**。控制R比值为Jacobi1.39、dual-ridge57.15、BP-warm55.86；较大比值本身不决定算法优劣。有限teacher更接近也不自动等于CFD更准确。

下一投入先检查新机制是否能提供强控制尚未有效提供的剩余方向，而不是直接扩大网络。[Graph Neural Preconditioners](https://arxiv.org/html/2406.00809v3)讨论弱谱方向的训练覆盖，但它的非线性预条件器使用FGMRES；不能直接塞进本项目原样CGLS/PCGLS，也不是BOS有效性的证据。这项诊断不重开原配方或授权新训练。

## 总体决定

补充一项限定数值归因：99个原生规模采样查询、33个三帧层中，沿每档相机一个固定范围方向，对封存学习初值作正/负1e-12相对微扰，CGLS33的四映射相对终点变化最多2.66e-13；一个固定行序打乱最多1.90e-13。独立递推与物理响应18/18检查通过，最大响应差1.66e-13。两条路径共享原初值/终点与canonical观测，不是独立重训。不读CFD或teacher、不测新准确率、不证明任意扰动、噪声或完整序列稳定。不能把不利制造算例直接归因到原生求解器；此前未过资格的几何构造不因此重开。

每路径新增13,860A+13,071A^T，继承准备和原训练成本非免费；55.28秒约633MiB为混合诊断开销，不是部署资源证据。停止这轮数值细节深挖，回到有效剩余方向与强控制比较，不把资格审计当成算法成果。

保留小幅完整留轨迹信号和可靠物理/经典基准；关闭当前字面图网络配方，不加宽、加轮次、换loss、阈值或精化深度救援。下一投入必须解释它能修复什么强经典控制未便宜提供的剩余误差，再另行结果前冻结。不是整个C路线关闭，也不是算法突破或论文就绪。

图消息算子已有[Graph Kernel Network](https://arxiv.org/abs/2003.03485)先例。此处不是该论文的BOS复现，不作first或SOTA声明。

## English

The latest physical information-location audit replays33 immutable other-fold training-mean pairs. For centered d=CFDmean-finiteTeacherMean andR(v)=||Av||²/||v||², all33 ratiosR(d)/R(CFDmean)<1, median1.7981e-4. Median relative field/image differences are17.897%/0.2401%. This is relatively weak training-mean information, not single-frame errors, all teacher-error causes, exact kernels or calibrated noise bounds. It does not prove nonidentifiability or impossibility of an adjoint initializer.

Own means/physicalCSR/NumPy versusTorch arithmetic pass43 checks with array/statistic differences1.49e-15/4.44e-16. Each path99A+0AT, no newfit/network/solve/held-truth input. Setup, earlier teacher/mean fitting andvalidation remain nonfree. The8.52-second/790-MiB mixed audit is not deployment timing. Preserve the raw-field clue but keep old mean/learnerFAIL unchanged; require a genuinely new lawful stable information mechanism before fitting, not expansion of the old network.

AnOct11 comparator fits raw-CFD and finiteCGLS128 field means from exactly other trajectories under11 complete folds, with identical normalization, observation correction and unchangedCGLS32. Both pass33 absolute strata but match0/3333 strong references. Raw-CFD has median field/observation gains14.38%/18.34% versus cold35 and field9.13% versus finite-teacher mean, but harms228 versus cold and is not uniformly better. These are post-seal descriptive effects, not new success gates. All114 independent checks pass, including own priors/recurrence/physics/CFD scoring and fold exclusion.

This explicit trained field prior plus exact adjoint correction is NOT a pure-range initializer or native exact-kernel identification, and does not change the strict initializer contract. Three known-cardinality caches,22 fits per implementation and35547 coefficients per prior/fold are not a shared neural operator. Per-query35A+34AT equalsBP-Warm33; cold/Jacobi/dual35 cost one extraAT. Each two-arm audit243309A+233310AT, inherited setup35547 forward equivalents and2 old factoryA, fitting/data/teacher/cache costs remain nonfree. The230.85-second/1.23-GiB mixed audit is not deployment timing. Close this literal mean recipe; prioritize a lawful stable raw-field information path rather than expansion of the old network. No neural, resource, external, real-BOST or paper success.

The latest global mechanism review keeps all11 models unchanged and replaces only the neural increment by its exact zero-signal geometry-gated JVP on all3333 queries. Data-dependentK1 normalization/base/residual and unchangedCGLS33 remain; gates depend on learned weights, so this is neither whole-map linearization nor an untrained geometry inverse. The prefrozen90%-effect-retention gate passes176/396 mean/p90/worst comparisons and2/33 complete strata. Median per-stratum mean gain retention is86.46%/88.14%/89.12%/91.90%. Much of the modest average signal survives; higher-order observation dependence affects uniform tails. This does not prove nonlinear necessity or that all gains are nonlinear. Joint nonharm againstcold35/nonlinear is3296/3333 and192/3333, while strong-reference matches remain0/3333 and absolute strata33/33.

Independent graph/JVP/refinement/physical scoring agrees on165 comparisons under shared weights/canonical inputs, not independent retraining. Field/image relative differences1.85e-14/6.63e-14 and metric/summary absolute differences1.13e-14/5.93e-13. Per-query35A+35AT is unchanged; each audit path uses123321A+119988AT including scoring/lift, plus nonfree inherited setup/fitting. The199.26-second,2.78-GiB mixed audit and smaller graph-action count are not deployment speed evidence. End this attribution without Taylor-order, expansion-point or training rescue; prioritize different effective repair and strong controls before model expansion.

A separately frozen robustness check keeps all11 terminal held-trajectory models unchanged. Each of99 opened anchors receives two1%-relative-L2 spherical observation perturbations:198 queries and66 three-frame strata. No fitting or selection. The learned endpoint retains joint nonworse performance against equal-action coldCGLS35 on198/198 queries, with median paired gains0.3332%/0.3745%/0.2655%/3.0475% for field/full-gradient/interior/noisy-observation errors. AgainstJacobi/dual-ridge the counts are39/198 and0/198; strong finite-reference matches remain0/198. Both learner and noisy reference retain66 absolute strata. Independent reconstruction passes42 checks, with field/image relative differences1.94e-14/6.99e-14 and metric difference1.59e-14. Weights, canonicalCSR and perturbed inputs are shared, not independently refitted. Each path's68508A+66726AT covers nine methods/reference/scoring; the learner remains35 pairs plus nonfree graph/setup/oldtraining. The99.25-second,2.542-GiB mixed audit is not deployment timing. Fixed-radius perturbations are not calibrated experimental noise, full noisy sequences or statistical guarantees; originalFAIL stays closed and no training expansion, resource or real-BOST claim follows.

A shared3,841-parameter signed cross-camera ray graph completed11 whole-trajectory LOTO folds,101 frames and5/7/9 cameras:3,333 held queries and33 complete strata. It reads current observations, exact-K1 residuals and reported geometry only. Camera IDs route grouping but are not learned features. Known signed ray couplings provide nonlinear messages; a fresh exact adjoint lift is followed by unchangedCGLS33, costing35A+35AT per standalone query plus nonfree geometry/graph/network work.

Each fold uses only the other ten trajectories' finite CGLS128 fields as teachers, with 330 fixed updates and about 11 frames per training trajectory-camera pair, not full-data epochs. Held teachers do not affect features, normalization, selection, stopping or fallback. All models seal before held predictions, which seal before CFD scoring. This is opened-family evidence, not an untouched external test.

**The preregistered primary fails:0/3,333 four-metric reference matches and0/33 complete matched strata.** Each match requires every error to be at most1.01 times the same-query finiteCGLS128 error plus1e-10. Basic absolute-p90 quality passes33/33 but does not substitute for the strong matched gate. The new segmented reference is a qualified finite comparator, not a converged exact inverse.

**A separate post-seal descriptive readback finds all3,333 queries jointly no worse than equal-action coldCGLS35**, with absolute comparison tolerance1e-10. Median paired relative improvements are0.3459% field,0.3928% full gradient,0.2604% interior gradient and3.5481% observation. This is a modest consistent whole-trajectory learning signal, not a new success gate. Against equal-action Jacobi-PCGLS35 the joint nonworse count is684/3,333; against dual-ridge-CG35 it is0/3,333. Beating the cheaper BP warm control or the untrained structure cannot establish superiority over strong controls.

Independent graph/features/NumPy prediction/lift/refinement/physical scoring passes 23/23 checks; maximum metric difference is 1.90e-14 and paired-median readback difference 2.69e-15. The paths share sealed weights, canonical CSR, observations and finite teachers; this is not independent retraining. The 3,630 training updates cost 14,520A+10,890AT, with another 264A+198AT for gradient audits. Each implementation's shared all-method inference uses 693,264A+689,931AT, plus 29,997A for physical scoring; these are not one model's standalone deployment counts. Teachers, graphs and caches are nonfree. The completed-run 1,403.18-second mixed audit excludes earlier aborted startup setup and is not fresh deployment wall/RSS. Native 12 cameras, perturbations, unopened transfer, curved rays and paired real BOS are untested.

A separately frozen post-open attribution reads each path's own sealed endpoints and physical replay images for all3,333 queries. E is the finiteCGLS128-minus-cold35 gap and D the learned-minus-cold35 endpoint correction. R(v)=||Av||²/||center(v)||². Median R(D)/R(E) is34.70, with5/7/9-camera medians19.62/33.94/43.15; all queries and33 complete-stratum medians exceed1. Median field cosine is0.3483, step/gap norm is5.00%, and realized finite-teacher gap-norm reduction is1.73%. Teacher-visible scaling along this fixed final direction explains a median12.13% of field-gap energy, not arbitrary warm-seed capacity. Independent derivative/arithmetic/statistic reconstruction differs by at most3.30e-12 per array and1.88e-12 per summary, with0 additionalA/AT. Original state/replay costs remain nonfree.

This supports a limited relative-observability hypothesis, not a full spectral decomposition, CFD true-error result or proof of every failure cause. Control median ratios are1.39 forJacobi,57.15 fordual-ridge and55.86 forBP-warm; a larger ratio alone is not a quality ranking. Closeness to a finite teacher is not automatically closeness toCFD. GNP motivates weak-spectrum coverage but uses nonlinearFGMRES, not unchangedCGLS/PCGLS; it is not BOS evidence. No new training or reopening follows.

Close this literal recipe without more width, epochs, changed loss, graph thresholds or refinement depth. Retain the small held-trajectory signal and qualified physics/classical comparison. Prioritize genuinely different residual-repair action that strong cheap controls do not explain. No algorithmic breakthrough, resource speedup or paper-ready claim follows. Graph message operators have existing GKN precedent; nofirst/SOTA claim is made.

A separate99-anchor/33-sampled-stratum native diagnostic finds at most2.66e-13 four-map endpoint response to a fixed plus/minus1e-12 relative range perturbation of sealed learned seeds. One row-order shuffle gives1.90e-13. Independent recurrence/physical responses pass18 checks, with maximum disagreement1.66e-13. Common seed/endpoint bytes and canonical observations are disclosed; this is not retraining, new accuracy or universal/full-sequence/noise stability. NoCFD/teacher scoring occurred. Manufacturing eligibility failures cannot be generalized into native solver failure; failed geometry recipes remain unqualified. Each path costs13,860A+13,071AT plus inherited preparation. The55.28-second,633-MiB mixed diagnostic is not deployment timing. End this numerical digression and return to effective residual repair and strong controls.

[去隐私结构化摘要 / Privacy-safe structured summary](segmented_cross_ray_learning_2026-10-10_public_summary.json)
