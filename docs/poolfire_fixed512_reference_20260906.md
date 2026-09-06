# 完整序列的固定预算经典基线 / Fixed-budget classical sequence reference

## 中文

2026-09-06。普通CGLS主基线与几何Jacobi PCGLS同价对照，固定512步，正式和独立实现均通过505/505帧、5/5条完整轨迹。每条101帧，每帧密度、全梯度、内部梯度和观测相对误差均不超过1%，不是只让平均值通过。最坏值在两种实现中取较大者：

| Metric / 指标 | Worst primary error / 主基线最坏误差 |
|---|---:|
| Density / 密度 | 0.659559% |
| Full gradient / 全梯度 | 0.560067% |
| Interior gradient / 内部梯度 | 0.978795% |
| Observation / 观测 | 0.008466% |

这把此前的五点可恢复性检查扩展成完整已打开序列上的经典精度标尺。它支持后续研究在有用的最终精度下减少计算，而不是只模仿误差较大的K4。内部梯度仍接近1%门，不能推断噪声鲁棒。

范围必须说明：固定512来自已经看过的五点试点；复用5个试点，另外500帧首次在这个预算下评价，但五条轨迹都是已打开的训练资料。只有一个干净九相机几何，数据生成和反演使用同一离散forward。不是独立测试、相机增删泛化、学习成功、真实BOST或论文突破。此前中间迭代曲线的数值不确定判决保持不变；本次没有重新认证它们，也没有证明512是最小调用数。

独立检查区分了同一场的物理重放与第二条求解路径。跨路径场/投影/指标最大差为8.98e-5/1.12e-5/9.99e-5，均在原有数值界内；同一场的投影/指标差仅7.87e-16/2.02e-17。每帧1%判断完全一致。严格同价非劣计数主基线为86/88、Jacobi为301/301；前者的数值排序不完全一致，因此不声称主基线更优。

每次逻辑求解为512A+512AT，不是加速结果。两臂、两实现的2000次新增离线求解共1,024,000A+1,024,000AT。复用几何的8192次基向量forward及29700解析行构建、原试点全部40960A+40960AT和本次额外4040次forward重放均单独披露。没有训练参数，也没有端到端时间或内存优势结论。

## English

Both the ordinary CGLS primary and equal-cost geometry-Jacobi PCGLS control at fixed512 steps pass505/505 frames and5/5 complete101-frame sequences in both implementations. EVERY frame meets1% relative density, full-gradient, interior-gradient and observation error, not merely the mean. The table takes the worse value across both implementations.

This extends the earlier five-point recoverability check to a useful-accuracy classical reference across the opened sequences. It motivates reducing computation at equivalent final accuracy, rather than only imitating the weak K4 comparator. Interior-gradient error remains close to the gate; noise robustness does not follow.

The budget is pilot-informed: five design points are reused and500 other frames are newly evaluated at this budget, but all trajectories are already-opened training material. Only one clean nine-camera geometry and the same discrete forward are used for data generation and inversion. This is not an independent test, variable-camera generalization, learned success, real BOST or a paper breakthrough. The old intermediate-curve numerical verdict remains inconclusive and is NOT requalified;512 is not proven minimal.

Same-field physical replay and the second solver trajectory are checked separately. Cross-path field/image/metric differences8.98e-5/1.12e-5/9.99e-5 satisfy the original bounds; same-field image/metric differences are7.87e-16/2.02e-17. All1% cell decisions agree. Strict equal-cost nonharm counts are86/88 for the primary and301/301 for Jacobi, so the primary ranking is not numerically identical and no primary-superiority claim is made.

Each logical solve costs512A+512AT. The2000 new offline solves cost1,024,000A+1,024,000AT, separately from reused geometry setup (8192 canonical forwards and29700 analytical rows), the entire inherited pilot (40960A+40960AT), and4040 new physical forwards. No parameters are trained and no end-to-end time or memory speedup is established.

![Worst errors across five opened sequences / 五条已打开轨迹的最坏误差](../assets/figures/poolfire_fixed512_reference_20260906.png)

[Redacted aggregate / 去隐私汇总](poolfire_fixed512_reference_20260906.json)

## 后续方向诊断 / Follow-on direction diagnosis

密度误差指预处理后的固定规范网格场，不是原分辨率CFD或实验绝对密度精度。本次没有训练模型或追加求解迭代，只读取已封存的K4、K512、观测和已验证的几何算子。

方向诊断：K4到合格参考的修正，在全部505帧上平均法向算子灵敏度更低。沿实际观测残差梯度，任何单个标量步长的场误差下界都至少29.57%，不能一步达到1%目标；这不排除后续多步CGLS或多方向暖启动，尚无学习加速结论。

K4到参考的场差异逐轨迹中位数约40.6%至48.6%。修正量的平均灵敏度更低，但灵敏度比在0.159至0.627之间，不能宣称全部落在近零特征值或某个空间低频带。参考误差法向方向上的最优标量只能消除18%至32%的场差异能量。

第二项检查使用真实部署可见方向 `g4=AT(y-Ax4)`，没有把它与参考误差的法向方向混为一谈。令 `R=x512`，`m=min_alpha ||R-x4-alpha*g4||/||R||`，由已验证的参考误差不超过1%，三角不等式给出每个真实场误差的下界 `max(0,0.99*m-0.01)`。每帧下界都超过29.57%；表格是下界，不是测得误差的范围。

| Trajectory / 轨迹 | Median late/K4 Rayleigh / 灵敏度比中位数 | Minimum scalar-step error lower bound / 标量一步误差下界最小值 |
|---|---:|---:|
| p=14kw_size=05 | 0.2347 | 38.1634% |
| p=22kw_size=03 | 0.4242 | 31.5886% |
| p=33kw_size=01 | 0.4792 | 29.5737% |
| p=45kw_size=05 | 0.2722 | 32.5921% |
| p=58kw_size=03 | 0.3621 | 33.3510% |

两实现逐帧排除结论一致，下界最大绝对差为2.31e-6，步长最大相对差为1.99e-6。合计新增3030A+2020AT，无CFD真值数组解析、训练或新求解终态。该排除只适用于单次标量修正直接完成任务，不排除其后继续CGLS、多方向修正或其他暖启动；旧学习失败和本页经典基线通过的结论都不变。

Density errors concern preprocessed gauge-fixed grid fields, not original-resolution CFD or calibrated experimental density. No training or additional solver iterations occur: the diagnostic reads sealed K4/K512 fields, observations and verified geometry operators.

Direction diagnosis: the K4-to-qualified-reference correction has lower mean normal-operator sensitivity on all 505 frames. Along the actual observation-residual gradient, every scalar step has a field-error lower bound of at least 29.57%, excluding one-step attainment of the 1% gate. This does not exclude further CGLS or multidirection warm starts and establishes no learned speedup.

Trajectory-median K4-to-reference field gaps are 40.6%--48.6%. The lower mean sensitivity ratio ranges from 0.159 to 0.627; it is NOT an exclusive near-null or spatial low-frequency band certificate. An oracle scalar along the reference-error normal direction removes 18%--32% of late field-error energy.

The second check instead uses the actual visible direction `g4=AT(y-Ax4)`. For qualified reference `R=x512`, define `m=min_alpha ||R-x4-alpha*g4||/||R||`. The verified 1% reference guarantee and reverse triangle imply the truth-field error lower bound `max(0,0.99*m-0.01)` for EVERY scalar. The table reports lower bounds, not measured-error ranges. Both implementations exclude all 505 frames, with maximum absolute bound difference 2.31e-6 and relative alpha difference 1.99e-6. Total new work is 3030A+2020AT, with no CFD truth array parsing, model fitting or new solver endpoint. Further CGLS, multidirection corrections and other warm initializers are not excluded.

文献提供的是方法提醒，不是本任务的成功证据：[HINTS](https://arxiv.org/abs/2208.13273)强调神经与经典迭代的误差成分互补；[DL-HIM训练与更新可靠性研究](https://arxiv.org/html/2602.06842v1)讨论训练初始右端项和部署残差之间的分布差异。这里没有复现其PDE速度结果，也没有证明本任务出现其假固定点现象。

The literature supplies methodological context, not our performance evidence: [HINTS](https://arxiv.org/abs/2208.13273) motivates complementary error correction; the [DL-HIM training/update study](https://arxiv.org/html/2602.06842v1) discusses mismatched initial training inputs and deployment residuals. Their PDE speed results are not reproduced here, and their false-fixed-point phenomenon has not been established for our one-shot initializer.

## 局部近似逆对照 / Local approximate-inverse control

经典对照进展：局部稀疏近似逆配合固定256步，在五个已打开哨兵的四项误差上均优于同预算Jacobi；全部1%门通过3/5，对照为0/5。两个失败点的内部梯度误差约1.111%与1.186%。当前配置封存，不加深或调参；不是完整序列、学习加速或真实BOST成果。

Classical control: a local sparse approximate inverse at fixed 256 steps improves all four errors over same-budget Jacobi on five opened sentinels. The 1% gate passes 3/5 versus 0/5; two interior-gradient errors remain about 1.111% and 1.186%. This configuration is closed without depth or parameter tuning. No complete-sequence, learned-speedup or real-BOST result is established.

| Opened midpoint / 已打开中点 | FSAI interior error / 内部梯度误差 | Jacobi interior error / 内部梯度误差 |
|---|---:|---:|
| p=14kw_size=05 | 1.110643% | 2.887197% |
| p=22kw_size=03 | 1.185324% | 2.306541% |
| p=33kw_size=01 | 0.832263% | 2.113687% |
| p=45kw_size=05 | 0.851966% | 2.937497% |
| p=58kw_size=03 | 0.793125% | 2.108898% |

这只是同一干净九相机设置的五个历史中点，不是五条完整轨迹。FSAI的场、全梯度与观测误差五点均通过，只有上表前两个内部梯度误差失门；不能事后放宽1%标准。两实现逐项判决一致，场/观测/指标最大差约2.04e-4/3.22e-5/1.88e-4，守住预先固定的数值界。相同场的独立物理回放差约7.03e-16。

These are five historical midpoints in one clean nine-camera acquisition, not five complete trajectories. FSAI field, full-gradient and observation errors pass all five points; only the first two interior-gradient errors miss the unchanged 1% threshold. Both implementations agree on every gate. Maximum field/image/metric differences are about 2.04e-4/3.22e-5/1.88e-4 within the preset numerical bounds; independent replay of the same field differs by about 7.03e-16.

每个求解仍需256A+256AT，另有256次L和257次L转置稀疏乘法。每套几何因子有103548个CSR存储项，约1.22MiB，需8192个局部求解；不能把几何预处理和因子运算当免费。20个离线终态合计5120A+5120AT，评分/探针另33A。没有端到端速度或新学习参数结果。

Each solve still needs 256 A+256 AT plus 256 L and 257 transposed-L sparse actions in the formal path. Each geometry factor stores 103548 CSR entries, about 1.22 MiB, and requires 8192 local solves; setup and factor actions are not free. Twenty offline endpoints total 5120 A+5120 AT, plus 33 scoring/probe A calls. There is no end-to-end speed or new learned-parameter result.

最初启动在观测/评分前发现常数零空间假设错误，已作为工程失效保留。现有forward先施加外边界零支撑再求梯度，因此不能任意减去求解均值；修正版本不做该后处理，不改因子配置、步数或门。几何边界应由实际算子决定，不能直接套用连续无限域直觉。

The first attempt stopped before observations/scoring because its constant-nullspace premise was invalid. This forward applies outer-zero support before differentiation, so arbitrary mean subtraction is not legitimate. The corrected run omits that postprocessing without changing factor settings, depth or gates. Boundary semantics must follow the actual operator, not an unbounded-domain intuition.

[FSAI原始论文 / Original FSAI paper](https://epubs.siam.org/doi/10.1137/0614004)是该经典方法的来源，不是本任务的突破证据。This is established numerical methodology, not component originality or proof of learned BOST acceleration.

## 三维状态小算子 / Small Field-State Operator

新一轮学习验证：81参数三维状态小算子完成10个完整轨迹外折拟合，505个预测先封存。七种同预算方法在五个预定中点均为0/5通过；候选内部梯度误差2.12%至3.02%，高于1%门，且普通CGLS、BP与旧dual-ridge的四项误差都更低。每次部署258A+258AT，另计几何准备与映射；已关闭该配置并跳过余下500帧重建。不是完整序列通过、学习加速或真实BOST成果。

New learning check: an 81-parameter field-state operator completed 10 whole-trajectory outer-fold fits, with all 505 predictions sealed first. All seven same-budget methods passed 0/5 prescribed midpoints. Candidate interior-gradient errors were 2.12% to 3.02%, above the 1% gate; ordinary CGLS, BP and frozen dual-ridge had lower errors on all four metrics. Each deployment costs 258 A+258 AT, plus geometry setup and maps. This configuration is closed and the remaining 500 refinements were skipped. No complete-sequence, learned-speedup or real-BOST success is established.

| 已开中点 / Opened midpoint | 正式内部梯度误差 / Formal interior error | 独立复算 / Independent |
|---|---:|---:|
| p=14kw_size=05 | 2.941796% | 2.948222% |
| p=22kw_size=03 | 2.285119% | 2.284554% |
| p=33kw_size=01 | 2.127788% | 2.121805% |
| p=45kw_size=05 | 3.015663% | 3.017217% |
| p=58kw_size=03 | 2.144515% | 2.148611% |

本次真正训练了一个新的共享参数模型：观测先经物理伴随与固定局部几何映射汇合到三维，再由逐点小网络预测修正，经精确正向与伴随构造暖启动，最后使用未修改的CGLS。训练目标是已通过精度门的K512参考，不再是较弱K4。每折仅用其余四条轨迹的404帧，留出的101帧不进入训练目标或归一化；两种表示各五折，每个模型81参数。只有干净九相机的历史数据证据，不是新工况泛化。

A new shared-parameter model was actually trained: an adjoint and fixed local geometry map fuse observations into a 3D state, a pointwise small network predicts a correction, exact forward/adjoint actions form the initializer, and unchanged CGLS refines it. Teachers are the qualified K512 reference rather than weak K4. Each fold trains on404 frames from four other trajectories; its101 withheld frames do not enter teacher fitting or normalization. Two representations use five folds each, with81 parameters per model. Evidence is restricted to opened clean nine-camera data, not a new-condition generalization result.

在两套数值实现中，候选四项误差均优于自身无学习几何滤波对照，但均劣于同预算普通CGLS、BP和旧dual-ridge。对同参数量Jacobi小网络的细微优势未在两套实现中完全一致，不能称稳定优势。候选场与全梯度误差也失门，只有观测误差通过。因此小幅降低训练损失不是节省精确调用的证据，合格参考仍未被更便宜的学习算法替代。

In both numerical paths the candidate improves all four errors over its own no-learning geometry filter, but loses on all four to same-budget ordinary CGLS, BP and frozen dual-ridge. Small differences from the same-size Jacobi network do not yield identical dominance in both implementations and are not a reliable advantage. Candidate field and full-gradient errors also fail; only observation passes. Lower training loss is therefore not evidence of exact-call savings, and a cheaper learned algorithm has not replaced the qualified reference.

五个必要中点的逐指标判决一致，独立状态/观测/指标最大差为2.43e-4/3.78e-5/2.25e-4，守住预定数值界；同一场的原生物理回放差7.12e-16。这里不是五条完整轨迹：根据结果前停止规则，其余500帧未做最终重建。全部505个预测已经封存，不能把预测数量写成505个准确重建。

Every metric decision agrees at the five necessary midpoints. Independent state/image/metric discrepancies of2.43e-4/3.78e-5/2.25e-4 satisfy the preset numerical bounds; same-field native replay differs by7.12e-16. These are not five complete trajectories: the pre-result stopping rule skipped the remaining500 final reconstructions. All505 predictions were sealed, but that count must not be reported as505 accurate reconstructions.

新离线训练、预测、核验与评分总账329650A+249850AT；实际70个终态求解另18030A+17990AT，初始化已在预测阶段执行并单列。每种方法的单次部署账均为258A+258AT；几何因子、映射以及已有参考的构建成本不免费。没有fresh wall/RSS优势、GPU需求或论文成功结论。不扩大或调参挽救这个固定配置，也不据此否定所有三维算子学习。

New offline training, prediction, audits and scoring total329650 A+249850 AT; the70 actual endpoint solves add18030 A+17990 AT, with initializer work executed and recorded in the prediction stage. Each method's per-deployment ledger is258 A+258 AT; geometry factors, maps and inherited reference construction are not free. No fresh wall/RSS advantage, GPU need or paper-success claim is established. This fixed configuration will not be enlarged or retuned, and its failure does not rule out all3D operator learning.

## 固定几何经典直接解 / Fixed-Geometry Classical Direct Solve

补齐关键经典对照：固定九相机几何的缓存直接分解，经独立QR复算，在505帧、5条完整轨迹上四项1%精度门全部通过。每种方法三次新进程测量，直接分解/CGLS512/Jacobi-PCGLS512的505帧总耗时中位数为2.80/166.85/168.86秒，进程峰值内存中位数为1.21/0.72/0.73GiB。包含重新分解、求解和写盘；不包含原始BOS矩阵构建与观测生成。该结果是经典方法对照，不是学习算法加速或真实BOST成果。

Classical comparator completed: cached direct factorization for one fixed nine-camera geometry passed all four 1% gates on505 frames and five complete trajectories, independently checked by rectangular QR. Across three fresh processes per method, median505-frame wall times for direct factorization/CGLS512/Jacobi-PCGLS512 are2.80/166.85/168.86seconds; median process peak memory is1.21/0.72/0.73GiB. Measurements include refactorization, solves and output writing, but exclude original BOS matrix construction and observation generation. This is a classical comparison, not learned acceleration or a real-BOST result.

| 经典方法 / Classical method | 505帧总秒数中位数 / Median seconds | 进程峰值GiB中位数 / Median peak GiB | 准备秒数中位数 / Median setup seconds |
|---|---:|---:|---:|
| factor | 2.798 | 1.208 | 0.9211 |
| cgls | 166.852 | 0.725 | 0.0000 |
| pcgls | 168.856 | 0.725 | 0.0076 |

这次补齐的是一个简单但关键的对照：同一相机几何被505帧复用时，可以先分解有效场空间的正规矩阵，再对每帧做一次伴随和两次三角求解。独立实现不形成正规方程，而是对相机行块乱序后的矩形矩阵做带列主元QR。未加新的正则、截断或训练参数；两者的场最大差2.242e-12、观测最大差3.680e-14，全部轨迹均通过。直接解的四项误差也都小于已合格的两种K512参考。

The missing comparator is simple but important:505 frames share the same camera geometry, so the active field normal matrix can be factored once and each frame solved using one adjoint and two triangular solves. The independent implementation forms no normal equations: it uses column-pivoted rectangular QR after camera-block row permutation. No new regularization, truncation or trained parameters are used. Maximum field and observation discrepancies are2.242e-12 and3.680e-14; every complete trajectory passes. Direct-solve errors are also lower on all four metrics than both qualified K512 references.

计时在独立精度门之后另行冻结。三种方法按固定轮换次序各启动三次，重新加载矩阵与观测并重建各自准备；全部九次的505个场和物理残差都通过独立回放与精度核验。直接分解使用8个BLAS线程，迭代法使用8个帧线程且每帧BLAS单线程。内存是整个求解进程峰值，包含因子、临时数组、输入和输出；监督进程另列，二者峰值之和只是并发内存上界，不是同步采样的精确峰值。操作系统文件缓存未清空。

Resource measurements were frozen separately after independent accuracy qualification. Each method ran three times in a fixed rotated order, reloading matrices and observations and rebuilding its own preparation. All505 fields and physical residuals from all nine runs passed independent replay and accuracy checks. Direct factorization uses eight BLAS threads; iterative methods use eight frame threads with single-thread BLAS. Peak memory covers the entire solve process, including factors, temporaries, inputs and outputs. Supervisor memory is listed separately; the sum of separate peaks is only a concurrent-memory upper bound, not an exactly synchronized peak. Operating-system file caches were not purged.

保留的直接因子约263.8MiB；其几何准备成本和三角求解不能当作免费操作。返回场及物理残差时直接法为1A+1AT加两次三角求解，两种K512迭代法为513A+512AT，其中一次A用于重新计算物理残差。这里没有证明512倍加速、最小迭代步数、任意相机变化、观测噪声稳定性或完整BOS端到端加速。现有几何矩阵与无噪声观测已就绪是比较的前提。

The retained direct factor occupies about263.8MiB; geometry preparation and triangular solves are not free. Returning a field and physical residual costs1A+1AT plus two triangular solves for the direct method, versus513A+512AT for either K512 method, including one fresh residual projection. This does not establish a512-fold speedup, minimum iteration depth, arbitrary camera changes, noise stability or complete BOS end-to-end acceleration. Existing geometry matrices and clean observations are prerequisites for this comparison.

科学判断：固定小规模、干净线性代理上的精度与调用数，单独不足以支持学习优势。后续学习必须说明为何不能直接复用这个经典解，并在同样计入准备与内存的条件下胜过它；不能只拿较慢的迭代法当对照。当前没有学习算法突破或真实实验结论。

Scientific consequence: accuracy and exact-call counts alone on this small, fixed, clean linear proxy do not support a learned advantage. Future learning must explain when this classical solution cannot simply be reused and beat it with preparation and memory included, not compare only against slower iterative solvers. No learned algorithm breakthrough or real-experiment conclusion follows.

## 合成噪声边界 / Synthetic Noise Boundary

适用边界已确认：同一固定几何加入1%合成观测噪声，三个固定种子共1515个样本，直接解为0/1515、完整轨迹0/5。场/全梯度/内部梯度误差p90为5.59%/6.19%/10.10%，都超过1%门；观测残差却全部通过。Zero、BP和两种512步迭代对照在15个固定中点也均未通过。干净数据的快速结果仍有效，但不能当作噪声下的准确重建；这不是实验噪声或学习成果。

Applicability boundary confirmed: adding1% synthetic observation noise to the same fixed geometry gives1515 samples across three fixed seeds. The direct inverse passes0/1515 samples and0/5 complete trajectories. Field/full-gradient/interior-gradient p90 errors are5.59%/6.19%/10.10%, above the1% gates, although every observation-residual gate passes. Zero, BP and both512-step controls also fail their15 fixed midpoints. Clean-data speed remains valid, but does not establish accurate noisy reconstruction. These are not measured experimental noise or learned results.

| 轨迹 / Trajectory | 场p90 / Field | 全梯度p90 / Full gradient | 内部梯度p90 / Interior gradient | 噪声观测p90 / Noisy residual |
|---|---:|---:|---:|---:|
| p=14kw_size=05 | 5.617% | 5.762% | 10.022% | 0.898% |
| p=22kw_size=03 | 4.657% | 5.769% | 8.669% | 0.898% |
| p=33kw_size=01 | 4.820% | 6.423% | 8.988% | 0.898% |
| p=45kw_size=05 | 6.389% | 6.259% | 10.898% | 0.898% |
| p=58kw_size=03 | 5.072% | 5.919% | 9.738% | 0.898% |

每行汇总同一轨迹101帧、三个预先固定噪声种子，共303个样本。噪声是作用于全部观测分量、按干净观测L2范数归一到1%的高斯方向，不是归一化后仍相互独立的高斯噪声，也不是实验测量。输入构造不读取密度真值，求解器只读有噪观测与已知几何；全部直接解及中点对照先封存，再读取真值评分。两套实现分别构造输入、执行Cholesky/QR求解与物理回放，逐指标离散判决完全一致。

Each row covers101 frames and three pre-fixed noise seeds, or303 samples. Noise is a Gaussian direction over every observation component, normalized to1% of the clean observation's L2 norm. Its components are not independent Gaussian samples after normalization, and it is not experimentally measured noise. Input construction does not read density truth; solvers receive only noisy observations and known geometry. All direct fields and midpoint controls are sealed before truth scoring. The two paths independently construct inputs, solve by Cholesky/QR and replay the physics, with identical discrete metric decisions.

独立场/观测/指标最大差2.32e-12/3.81e-14/6.70e-13，正规方程驻点残差1.11e-16。观测残差p90约0.898%，但内部梯度误差p90约10.10%，最坏12.48%。干净到有噪的场变化也通过独立线性回放。因此这里暴露的是该未正则化估计器对所测噪声的放大，不是求解尚未收敛。观测拟合好，不等于三维场恢复准确。

Maximum independent field/image/metric discrepancies are2.32e-12/3.81e-14/6.70e-13, with normal-stationarity residual1.11e-16. Observation-residual p90 is about0.898%, but interior-gradient p90 is10.10% and worst error12.48%. Clean-to-noisy field changes also pass independent linear replay. This exposes amplification of the tested noise by the unregularized estimator, not an unconverged solve. A good observation fit does not imply accurate3D recovery.

四个经典对照各自的0/15只针对五个固定中点乘三个种子，不是它们完整轨迹的结论，也不是最优迭代深度的结论。没有事后换参考、改噪声、放宽门或加大模型。上一节干净数据的505/505和计时结果保持原结论，仅将其适用范围限定清楚：当前直接逆不能作为1%噪声下满足同一精度门的教师。该负结果不证明所有去噪先验或估计器不可能；它要求后续方法先解决噪声下的估计稳定性，再讨论暖启动加速。

Each classical control's0/15 refers only to five fixed midpoints times three seeds, not its complete trajectories or an optimal iteration count. There is no post-hoc reference switch, noise change, relaxed gate or larger model. The previous clean505/505 and timing result remains intact; its boundary is now explicit: this direct inverse is not a teacher meeting the same accuracy gate at1% noise. This negative result does not prove all denoising priors or estimators impossible. It requires future work to address estimation stability under noise before claiming warm-start acceleration.

## 四参数非线性去噪 / Four-Parameter Nonlinear Denoising

四参数去噪已完成独立复算：五折完整轨迹留出先封存1515个预测，再检验15个固定中点，最终0/15通过。相比同样15个样本的直接逆，场/全梯度/内部梯度p90从6.14%/6.22%/10.82%降到5.67%/5.80%/10.13%，但仍远超1%门，观测残差也升至1.14%。学习阈值优于不训练的通用阈值，但不足以解决噪声恢复；已跳过剩余1500次重建验证，关闭这个固定配置。不是完整重建、加速或真实BOST成功。

Four-parameter denoising independently checked: five complete-trajectory outer folds seal1515 predictions before15 fixed midpoint tests; the final result is0/15. Against the direct inverse on those same15 samples, field/full-gradient/interior-gradient p90 decreases from6.14%/6.22%/10.82% to5.67%/5.80%/10.13%, still far above the1% gates; observation-residual p90 rises to1.14%. Learned thresholds outperform untrained universal thresholds, but do not solve noisy recovery. The remaining1500 reconstruction checks are skipped and this fixed configuration is closed. This is not complete reconstruction, acceleration or real BOST success.

| 方法 / Method | 场p90 / Field | 全梯度p90 / Full gradient | 内部梯度p90 / Interior gradient | 噪声观测p90 / Noisy residual |
|---|---:|---:|---:|---:|
| 直接逆 / Direct inverse | 6.144% | 6.221% | 10.820% | 0.898% |
| 学习阈值 + K1 / Learned thresholds + K1 | 5.673% | 5.799% | 10.125% | 1.136% |
| 通用阈值 + K1 / Universal thresholds + K1 | 14.209% | 13.391% | 23.594% | 6.706% |
| 学习阈值，不迭代 / Learned thresholds, no refinement | 5.727% | 5.860% | 10.256% | 1.246% |

表内严格比较同样的五条轨迹中点乘三个噪声种子，合计15个样本，不能与上一节1515个样本的分位数混用。三项场指标在全部15个配对样本上都比直接逆改善，但观测拟合在全部15个样本上变差；未达到四指标同时通过。学习模型好于通用阈值，只是这个小型对照的相对改善，不是稳定算法优势。

Every row compares the same five trajectory midpoints times three noise seeds, totaling15 samples. These quantiles must not be mixed with the previous1515-sample summary. All three field metrics improve over the direct inverse in every paired sample, while observation fit worsens in every sample. The four metrics do not pass together. Beating universal thresholds is a relative improvement within this small control, not an established algorithmic advantage.

只学习四个共享的尺度阈值；逐系数的噪声放大量仅由已知几何计算。训练、目标归一化都只用外折中的其他完整轨迹，部署输入只含有噪观测与几何。两套独立实现分别重新训练五折，并核对预测、精确lift、原有一步CGLS与物理观测。最终场/观测/指标最大差为1.65e-12/3.11e-12/2.65e-13，全部离散判决一致。1515是先封存的外折预测数，不是1515次已经通过或完成的重建验证。

Only four shared scale thresholds are learned; coefficientwise noise amplification comes solely from known geometry. Training and target normalization use only the other complete trajectories in each outer fold. Deployment receives noisy observations and geometry only. Two implementations independently refit all five folds and check predictions, exact lift, one unchanged CGLS step and physical observations. Maximum final field/image/metric discrepancies are1.65e-12/3.11e-12/2.65e-13, with identical discrete decisions. The1515 count denotes sealed outer predictions, not1515 completed or successful reconstruction checks.

输入先经过直接逆，再去噪、精确lift与K1，完整逻辑在线账为3A+3A^T和四次三角求解；几何因子与噪声放大表构建另计，不能把缓存当作免费。通用阈值、有/无K1、Zero、BP、CGLS3、Jacobi3和历史dual-ridge+K2均作明确对照，各0/15。不训练的直接逆对照已从上一轮封存证据复用，没有重复重跑。没有做新的wall/RSS速度比较。

The input first passes through the direct inverse, followed by denoising, exact lift and K1. The complete logical online ledger is3A+3A^T and four triangular solves. Geometry factorization and the noise-amplification table cost extra; cached work is not free. Explicit controls include universal thresholds, with/without K1, Zero, BP, CGLS3, Jacobi3 and historical dual-ridge+K2, each0/15. The untrained direct control is reused from the previous sealed evidence without rerunning it. No new wall/RSS speed comparison was performed.

本轮只关闭这个固定四阈值、小波表示和训练损失的配置，不证明其他非线性先验不可能。数据仍是已打开的公开训练轨迹与固定九相机合成噪声，不是未打开的外部门、可变相机泛化或实验位移图。小波收缩是经典方法，本项目不作首创声明，参见[Donoho与Johnstone的原始研究](https://statistics.stanford.edu/technical-reports/ideal-spatial-adaptation-wavelet-shrinkage)。

This closes the fixed four-threshold wavelet representation and training-loss configuration, not every nonlinear prior. The data remain previously opened public training trajectories with fixed nine-camera synthetic noise, not an untouched external condition, variable-camera generalization or experimental displacement images. Wavelet shrinkage is classical; no first-use claim is made. See [Donoho and Johnstone's primary report](https://statistics.stanford.edu/technical-reports/ideal-spatial-adaptation-wavelet-shrinkage).

## 局部相关性与误差来源 / Local Correlation and Error Sources

局部统计先验仍未过关：用其他完整轨迹学习邻近密度相关性，配合几何噪声协方差与一步CGLS，15个固定中点仍为0/15。场/全梯度/内部梯度误差p90为5.10%/5.41%/9.08%，较单体素控制改善，但远超1%门。独立误差分解显示，残留输入误差较大，先验本身也会改动真实结构；这不支持简单加强去噪或增加迭代。已关闭这个固定配置，跳过剩余1500次重建验证；不是完整重建、加速或真实BOST成功。

Local statistical prior still fails: neighboring-density correlations learned from other complete trajectories, combined with geometry-noise covariance and one CGLS step, give0/15 on fixed midpoints. Field/full-gradient/interior-gradient p90 errors are5.10%/5.41%/9.08%, better than the pointwise control but far above1%. Independent error decomposition finds substantial remaining input error and distortion of true structure by the prior itself. This does not support simply stronger denoising or more iterations. This fixed configuration is closed and1500 further reconstruction checks are skipped. No complete reconstruction, acceleration or real BOST success.

| 同一15个中点 / Same 15 midpoint-seed cases | 场p90 / Field | 全梯度p90 / Full gradient | 内部梯度p90 / Interior gradient | 观测p90 / Observation |
|---|---:|---:|---:|---:|
| 局部联合先验+K1 / Local joint prior+K1 | 5.097% | 5.410% | 9.079% | 1.134% |
| 单体素先验+K1 / Pointwise prior+K1 | 6.129% | 6.210% | 10.788% | 0.949% |
| 局部先验，无迭代 / Local prior, no refinement | 5.170% | 5.505% | 9.304% | 1.310% |
| 单体素先验，无迭代 / Pointwise prior, no refinement | 6.131% | 6.210% | 10.789% | 0.966% |

先验只用外折中的其他完整轨迹估计共享局部均值与协方差；单体素控制只使用中心均值与方差。两个先验各自封存1515个外折预测，然后进行15个必要中点检查。独立实现分别重建统计量、噪声相关性、增益、预测、精确lift、原有K1与物理观测，13项检查全真。场/观测/指标最大差1.85e-12/5.49e-12/3.93e-13。表内四臂均0/15，不是它们已完成1515次重建。旧的直接逆、小波、Zero、BP、CGLS、Jacobi和dual-ridge证据仅复用，没有换参数重跑。

The prior estimates shared local means and covariance only from other complete trajectories in each outer fold; the pointwise control uses central mean and variance only. Both priors seal1515 outer predictions before15 necessary midpoint checks. Independent implementations rebuild statistics, noise correlation, gains, predictions, exact lift, unchanged K1 and physical observations. All13 checks pass; maximum field/image/metric differences are1.85e-12/5.49e-12/3.93e-13. Each table arm passes0/15, not1515 completed reconstruction checks. Earlier direct, wavelet, Zero, BP, CGLS, Jacobi and dual-ridge evidence is reused without retuned reruns.

局部联合先验对全部15个配对样本的场与两种梯度指标都比单体素控制好，但观测拟合全部变差。完整逻辑在线账仍为3A+3A^T和四次三角求解，几何因子、逆矩阵、局部噪声表及外折增益构建另计。估计405个统计量不等于神经网络训练；没有新的实测速度或内存优势。

The local joint prior improves all three field metrics over the pointwise control in all15 paired samples, while observation fit worsens in all15. The complete logical online ledger remains3A+3A^T and four triangular solves; geometry factorization, inverse, local-noise tables and fold gain construction cost extra. Estimating405 statistics is not neural-network training. No new measured speed or memory advantage is established.

### 结果后归因 / Post-Open Attribution

封存后才读取已评分真值，分解“最终误差 = 先验偏差 + 保留的输入误差 + K1校正”，不重训、不选参数、不新增A/A^T调用。状态闭合相对差7.78e-15，平方指标闭合差6.08e-18，独立能量项最大差4.61e-14。这是用于理解失败的真值可见诊断，不是部署可用的去噪器。

Only after sealing, already-scored truth is used to decompose final error into prior bias, propagated inverse-state error and K1 correction. There is no refitting, parameter selection or extra A/A^T call. Relative state closure is7.78e-15, squared-metric closure6.08e-18 and maximum independent energy-term difference4.61e-14. This truth-aware diagnostic explains failure; it is not a deployable denoiser.

| 分量中位相对范数 / Median component norm | 先验偏差 / Prior bias | 保留的输入误差 / Propagated inverse error | K1校正 / K1 correction |
|---|---:|---:|---:|
| 场 / Field | 1.878% | 3.694% | 0.417% |
| 全梯度 / Full gradient | 2.564% | 4.517% | 0.621% |
| 内部梯度 / Interior gradient | 4.617% | 6.943% | 1.030% |

这些范数不是能相加的百分比份额，分量之间有带符号交叉项。三项指标中，全部15个样本的保留输入误差能量都大于先验偏差能量，但先验偏差本身也不小。K1在全部15个样本上改善了三项场指标，仍不足以通过门。下一步不能把问题简单归咎于算得不够久，也不能假设加强去噪一定有效。

These norms are not additive percentage shares: signed cross terms are present. For all three metrics, propagated input-error energy exceeds prior-bias energy in every one of15 cases, but the bias itself is substantial. K1 improves all three field metrics in all15 cases and still does not pass. The gap cannot simply be attributed to insufficient computation, nor does it imply stronger denoising must work.

本轮关闭这个固定局部高斯先验配置，不调整块大小、协方差秩、数值载荷或深度来补救。数据仍为已打开的公开训练轨迹、固定九相机与合成噪声。高斯先验只是工作模型，不证明实际信号或归一化后的噪声具有高斯独立分布，也不提供经校准的后验区间。贝叶斯小块模型已有[经典原始研究](https://www.ipol.im/pub/art/2013/16/)；本试验不是NL-Bayes完整复现、组件首创、外部门或真实BOST结果。

This fixed local-Gaussian configuration is closed, without patch-size, covariance-rank, numerical-load or depth rescue. Data remain opened public training trajectories with fixed nine-camera synthetic noise. The Gaussian prior is a working model, not proof that actual signals or normalized noise are independent Gaussian samples, nor a calibrated posterior interval. Bayesian patch models have [established primary literature](https://www.ipol.im/pub/art/2013/16/). This is not a full NL-Bayes reproduction, component novelty, untouched external result or real BOST result.

## 真实场与噪声观测门 / Truth Oracle and the Noise Gate

观测门审计发现一个小但真实的问题：加入1%合成噪声后，旧观测门会拒绝308/1515个真实三维场输入。噪声按干净观测归一化，评分却除以带噪观测；最大超门仅0.000141个百分点，解释不了已有5%至10%的场与梯度误差。同一15个中点的旧候选，即使按已知噪声预算诊断，联合通过仍均为0/15。独立复算确认，这是评价口径问题，不是算法成功；旧失败不翻案，后续新试验才可另冻噪声一致的指标。

Observation-gate audit finds a small but real issue: with1% synthetic noise, the old gate rejects308/1515 exact true-field inputs. Noise is normalized by clean observations, but scoring divides by noisy observations. Maximum excess is only0.000141 percentage points and cannot explain existing5% to10% field/gradient errors. On the same15 midpoints, all inherited candidates still pass0/15 jointly even under a known-noise-budget diagnostic. Independent recomputation confirms a metric-semantics issue, not algorithm success. Old failures stand; only new preregistered experiments may adopt noise-consistent metrics.

令干净投影为b，噪声为e，观测y=b+e，且||e||=delta||b||，delta=0.01。
真实场的旧观测分数为delta/sqrt(1+delta^2+2delta*c)，其中c是b与e夹角余弦。
它超过delta当且仅当c<-delta/2。这是范数恒等式，不是新算法或新定理。

Let clean projection be b, noise e, y=b+e and ||e||=delta||b|| with delta=0.01.
The truth oracle's old score is delta/sqrt(1+delta^2+2delta*c), where c is the angle cosine between b and e.
It exceeds delta exactly when c<-delta/2. This is a norm identity, not a new algorithm or theorem.

独立矩阵各重放505个真实场，合计离线1010A、0A^T；1515个噪声样本逐项复算，三个seed分别拒绝95/112/101个。五条轨迹分别拒绝65/73/53/64/53个，每条303样本。两实现离散判决完全一致，公式差1.74e-17，重放相对差7.86e-16。球面均匀噪声方向下理论拒绝概率约19.4435%，本次实际20.3300%；前者不是本次实测比例。归一化后的噪声方向不是独立高斯分量，也不是实验测得噪声。

Each independent matrix replays505 true fields, totaling1010 offline A calls and0 A^T. All1515 noisy cases are recomputed; three seeds reject95/112/101 cases. The five trajectories reject65/73/53/64/53 out of303 each. Discrete decisions agree exactly; formula discrepancy is1.74e-17 and relative replay discrepancy7.86e-16. Uniform sphere directions imply about19.4435% theoretical rejection probability; observed rejection is20.3300%, not the theoretical value. Normalized noise directions do not have independent Gaussian components and are not measured experimental noise.

| 同一15个中点 / Same15 midpoints | 干净投影误差p90 / Clean projection | 带噪拟合误差p90 / Noisy fit | 场与梯度联合通过 / Field+gradient passes |
|---|---:|---:|---:|
| 直接逆 / Direct inverse | 0.4502% | 0.8982% | 0/15 |
| 小波+K1 / Haar+K1 | 0.8075% | 1.1356% | 0/15 |
| 局部先验+K1 / Local prior+K1 | 0.7971% | 1.1338% | 0/15 |
| 单体素先验+K1 / Pointwise prior+K1 | 0.5415% | 0.9487% | 0/15 |
| 局部先验，无迭代 / Local prior, no refinement | 1.0350% | 1.3105% | 0/15 |

四个带迭代/直接逆候选在这15个样本中，干净投影误差都低于1%，但场与梯度均未过关。投影接近并不意味着三维恢复准确。平方带噪残差由干净投影误差能量、噪声能量和带符号交叉项组成，独立闭合差5.43e-20。不能把去噪后残差增大直接等同于三维恢复变差，也不能把投影变准当作三维成功。

For the four refined/direct candidates, clean projection error is below1% on all15 samples, yet field/gradient accuracy fails throughout. Projection agreement does not imply accurate3D recovery. Squared noisy residual decomposes into clean-projection error energy, noise energy and a signed cross term, with independent closure discrepancy5.43e-20. A larger post-denoising residual is not by itself worse3D recovery, and a better projection is not3D success.

新增的已知噪声预算残差||prediction-y||/||e||仅作诊断，单位门加1e-12相对舍入余量；它不替换旧门，也不让旧失败候选通过。未来若采用，需要结果前另行冻结，真实实验还需要独立噪声估计，不能偷读真值生成噪声预算。[经典差异原则背景](https://www.imm.dtu.dk/~pcha/Regutools/)不构成本项目首创。

The known-budget residual ||prediction-y||/||e|| is diagnostic only, with a unit bound and1e-12 relative roundoff allowance. It neither replaces frozen gates nor rescues failed candidates. Future use requires a new preregistration; real experiments need independent noise estimation, not a truth-derived budget. [Classical discrepancy-principle context](https://www.imm.dtu.dk/~pcha/Regutools/) is not project novelty.

执行备注：第一次任务在数组封存后因JSON布尔序列化失败退出；保留原始失败、只重建报告，未重放、未改数组或门。独立数值检查14项与额外终态核验通过。工程修复不是科学结果；本轮没有训练、速度优势、外部泛化或真实BOST结论。

Execution note: the first task failed during JSON boolean serialization after array sealing. The failure was retained and only the report rebuilt, without replay or altered arrays/gates. All14 numerical checks and a separate terminal audit pass. Engineering repair is not science; no training, speed advantage, external generalization or real BOST result is claimed.

## 实际场是否互相混淆 / Are Actual Source Fields Ambiguous?

实际CFD配对审计：505个已打开的处理后三维场共127,260对，在1%有界观测噪声下没有相互重叠，最接近的一对也需约8.93%噪声才相交。只看观测幅值则有11,468对发生歧义，完整投影结构包含额外信息。但505/505个场的最近邻都来自自身轨迹，有限样本可区分不等于未见轨迹可重建。独立复算已确认；这是对失败原因的约束，不是学习、重建或真实BOST成功。

Actual-CFD pair audit: among127,260 pairs of505 opened, processed3D fields, no observation balls overlap at1% bounded noise. Even the closest pair needs about8.93% noise to intersect. Amplitude-only observations admit11,468 ambiguous pairs, so full projection structure carries additional information. However, nearest neighbors of505/505 fields belong to their own trajectories. Finite-sample distinguishability is not unseen-trajectory reconstruction. Independently verified; this constrains failure explanations, not learned, reconstruction or real BOST success.

本轮仅使用已有五条公开训练轨迹的505个规范与边界处理后CFD目标，不合成任意节点扰动、不改变幅值、不打开新数据。对观测b_i和b_j，两个相对有界噪声球相交的最小半径为eta=||b_i-b_j||/(||b_i||+||b_j||)。若它不超过1%，而某个场或梯度分离比B_D=||D*x_i-D*x_j||/(||D*x_i||+||D*x_j||)超过1%，同一个观测下就不能同时为两个真值提供1%相对精度。这是三角不等式，不是新定理。

Only505 existing gauge- and boundary-processed CFD targets from five public training trajectories are used, without arbitrary nodal perturbations, amplitude changes or new data. For observations b_i,b_j, the minimum relative bounded-noise radius for ball intersection is eta=||b_i-b_j||/(||b_i||+||b_j||). If eta is at most1% while a field/gradient separation ratio B_D=||D*x_i-D*x_j||/(||D*x_i||+||D*x_j||) exceeds1%, a single estimate at the common observation cannot be within1% of both truths. This follows from the triangle inequality, not a new theorem.

全部127,260对的eta都大于1%，最小为8.92899335%。最近一对的场/全梯度/内部梯度分离比为8.4470%/10.9610%/18.7920%，但构造共同观测需要两端各8.9290%噪声，因此不能拿它充当1%的反例。只用观测范数这一便宜信息时，11,468对会混淆；这里没有运行任何新重建器。

Every one of127,260 pairs has eta above1%; the minimum is8.92899335%. The closest pair has field/full-gradient/interior-gradient separation ratios8.4470%/10.9610%/18.7920%, but its common observation needs8.9290% noise at either endpoint. It is therefore not a1%counterexample. Using only the cheap observation norm admits11,468 ambiguous pairs; no new reconstructor is run here.

| 已打开轨迹 / Opened trajectory | 最近邻来自同轨迹 / Same-trajectory nearest | 最近邻噪声交叠半径p90 / Nearest overlap radius p90 |
|---|---:|---:|
| 1 | 101/101 | 12.0986% |
| 2 | 101/101 | 16.8384% |
| 3 | 101/101 | 15.5848% |
| 4 | 101/101 | 19.2312% |
| 5 | 101/101 | 20.5547% |

正式Gram距离与独立直接差分距离、独立导数的最大比值差4.23e-14，全部阈值与最近邻判决一致。独立物理重放最近一对及其加权组合，总计离线6A、0A^T；共同观测相对差7.43e-16以内。7项数值检查与单独终态核验通过，无模型训练和速度主张。

Formal Gram distances and independent direct differences/derivatives agree within4.23e-14 in ratios; all threshold and nearest-neighbor decisions match. Independent physical replay of the closest endpoints and their weighted combination totals6offline A calls and0 A^T, with common-observation relative discrepancy below7.43e-16. All7 numerical checks and a separate terminal audit pass, without model training or speed claims.

这只排除了“已存这505个实际场在1%有界噪声下彼此不可区分”的解释。有限字典辨认甚至可以靠记忆完成，不证明连续CFD流形、帧间状态、未见轨迹、物理模型偏差或真实实验可辨识。它也不是之前球面随机噪声的平均风险结论。最近邻全部属于自身轨迹，更要求保持整轨迹外折，不能拿随机帧划分冒充外推。旧先验和旧预测器失败保持不变；不恢复已关闭的幅值、最近邻库或更大模型路线。

This only excludes mutual1% bounded-noise ambiguity among these505 stored actual fields. Finite-dictionary identification can be memorization; it proves nothing about a continuous CFD manifold, between-frame states, unseen trajectories, physical-model mismatch or real experiments. It is not an average-risk result for earlier random sphere noise. All nearest neighbors being within their own trajectories reinforces complete-trajectory outer folds; random-frame splits cannot stand in for extrapolation. Old prior/predictor failures remain unchanged; closed amplitude, nearest-library and larger-model paths are not revived.

物理先验用于约束病态逆问题是[经典背景](https://www.imm.dtu.dk/~pcha/Regularization/regu.html)。本次增量是实际源类别的可审查证据，不是组件首创、算法突破、泛化或真实BOST结果。

Physical solution constraints for ill-posed inverses are [established background](https://www.imm.dtu.dk/~pcha/Regularization/regu.html). This increment is inspectable evidence about the actual source class, not component novelty, algorithm breakthrough, generalization or real BOST.
