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
