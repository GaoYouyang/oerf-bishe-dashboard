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
