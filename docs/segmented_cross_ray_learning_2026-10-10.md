# 完整留轨迹学习：有小幅收益，尚无强参考加速

2026-10-10。这次不是训练内拟合或真值可见容量诊断。一个3,841参数的共享有符号跨相机射线图网络完成11个完整轨迹留出外折：每条101帧，5/7/9相机，共3,333个held查询、33个完整轨迹×相机层。

## 实际做了什么

先对同一离散梯度的三线性插值表示做分段射线积分，另行独立核验新的观测与有限CGLS128参考。随后结果前固定唯一图网络：由已知射线之间的有符号物理耦合传递消息，输入只含当前观测、精确K1残差和reported geometry；按相机等权池化，参数共享，不把相机序号、时间或轨迹标签作为特征。预测dual修正，fresh exact A^T lift后接未修改CGLS33。在线字面账35A+35A^T，几何/图缓存和网络工作另计。

每个外折只用其他十条轨迹的有限CGLS128场作teacher，330次固定CPU更新，每个训练轨迹×相机约11帧，不是全数据epoch；held轨迹teacher不进入输入、标准化、参数选择、停止或回退。全部模型先封存，再预测全部held查询，最后才读取CFD真值评分。不是独立数据集或未打开的外部门。

## 两个不能混淆的结果

**预注册主门失败：0/3,333查询、0/33完整层匹配强参考的全部四指标。** 匹配是每项误差不超过同查询有限CGLS128误差的1.01倍加1e-10；不是“三维误差只有1%”。基本绝对p90门33/33通过，不能替代强参考匹配。CGLS128本身只是合格有限参考，不是收敛逆解。

封存之后的描述性配对读回显示：相对同35A+35A^T的冷CGLS35，**3,333/3,333查询的场、全梯度、内部梯度和观测误差均不差**，使用绝对比较容差1e-10。四项配对相对改善中位数为**0.3459% / 0.3928% / 0.2604% / 3.5481%**。这支持小而一致的留轨迹学习信号，但不是新预注册成功门，也不改变主门FAIL。

更强同动作控制没有被稳定排除：相对Jacobi-PCGLS35四项不差为684/3,333；相对dual-ridge-CG35为0/3,333。网络比更便宜的BP初值+CGLS33及自身未训练结构在全部查询均不差，仍不能据此只选弱控制包装加速。

## 复算与成本边界

独立路径重新构造图、特征、NumPy网络、exact lift、求解器精化和物理评分，23/23检查通过。最大逐项指标差1.90e-14；事后配对中位数最大差2.69e-15。两条路径共享封存权重、canonical CSR、观测与有限teacher；不是独立重训或独立实验。

共3,630次训练更新，训练账14,520A+10,890A^T，另有梯度审计264A+198A^T。每条独立实现路径的全部方法共享预测账693,264A+689,931A^T，物理评分29,997A；不是单模型部署账。teacher、图构建与缓存亦不免费；1,403.18秒是完成运行的混合离线审计，不含更早启动失效的几何开销，更不是部署计时。没有fresh部署wall/RSS、native12相机、扰动门、未开封外部泛化、曲线光线或真实实验BOST结论。观测仍是线性弱偏折直线射线代理，不是标定像素位移。

## 总体决定

保留小幅完整留轨迹信号和可靠物理/经典基准；关闭当前字面图网络配方，不加宽、加轮次、换loss、阈值或精化深度救援。下一投入必须解释它能修复什么强经典控制未便宜提供的剩余误差，再另行结果前冻结。不是整个C路线关闭，也不是算法突破或论文就绪。

图消息算子已有[Graph Kernel Network](https://arxiv.org/abs/2003.03485)先例。此处不是该论文的BOS复现，不作first或SOTA声明。

## English

A shared3,841-parameter signed cross-camera ray graph completed11 whole-trajectory LOTO folds,101 frames and5/7/9 cameras:3,333 held queries and33 complete strata. It reads current observations, exact-K1 residuals and reported geometry only. Camera IDs route grouping but are not learned features. Known signed ray couplings provide nonlinear messages; a fresh exact adjoint lift is followed by unchangedCGLS33, costing35A+35AT per standalone query plus nonfree geometry/graph/network work.

Each fold uses only the other ten trajectories' finite CGLS128 fields as teachers, with 330 fixed updates and about 11 frames per training trajectory-camera pair, not full-data epochs. Held teachers do not affect features, normalization, selection, stopping or fallback. All models seal before held predictions, which seal before CFD scoring. This is opened-family evidence, not an untouched external test.

**The preregistered primary fails:0/3,333 four-metric reference matches and0/33 complete matched strata.** Each match requires every error to be at most1.01 times the same-query finiteCGLS128 error plus1e-10. Basic absolute-p90 quality passes33/33 but does not substitute for the strong matched gate. The new segmented reference is a qualified finite comparator, not a converged exact inverse.

**A separate post-seal descriptive readback finds all3,333 queries jointly no worse than equal-action coldCGLS35**, with absolute comparison tolerance1e-10. Median paired relative improvements are0.3459% field,0.3928% full gradient,0.2604% interior gradient and3.5481% observation. This is a modest consistent whole-trajectory learning signal, not a new success gate. Against equal-action Jacobi-PCGLS35 the joint nonworse count is684/3,333; against dual-ridge-CG35 it is0/3,333. Beating the cheaper BP warm control or the untrained structure cannot establish superiority over strong controls.

Independent graph/features/NumPy prediction/lift/refinement/physical scoring passes 23/23 checks; maximum metric difference is 1.90e-14 and paired-median readback difference 2.69e-15. The paths share sealed weights, canonical CSR, observations and finite teachers; this is not independent retraining. The 3,630 training updates cost 14,520A+10,890AT, with another 264A+198AT for gradient audits. Each implementation's shared all-method inference uses 693,264A+689,931AT, plus 29,997A for physical scoring; these are not one model's standalone deployment counts. Teachers, graphs and caches are nonfree. The completed-run 1,403.18-second mixed audit excludes earlier aborted startup setup and is not fresh deployment wall/RSS. Native 12 cameras, perturbations, unopened transfer, curved rays and paired real BOS are untested.

Close this literal recipe without more width, epochs, changed loss, graph thresholds or refinement depth. Retain the small held-trajectory signal and qualified physics/classical comparison. Prioritize genuinely different residual-repair action that strong cheap controls do not explain. No algorithmic breakthrough, resource speedup or paper-ready claim follows. Graph message operators have existing GKN precedent; nofirst/SOTA claim is made.

[去隐私结构化摘要 / Privacy-safe structured summary](segmented_cross_ray_learning_2026-10-10_public_summary.json)
