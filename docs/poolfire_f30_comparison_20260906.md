# PoolFire：局部改善已复现，稳定同精度尚未通过

2026-09-06。五条公开训练轨迹、505帧、完整轨迹留一验证：随机特征神经算子通过409帧、2/5条完整轨迹；线性化对照通过412帧、1/5条轨迹。两者都没有通过完整目标。

## 比什么，如何判定

不是等不到数据：用已有五条公开PoolFire训练轨迹各101帧，在同一套九相机F30几何上做完整轨迹留一。每折其他404帧用于拟合，留出101帧不参与输入标准化、超参、回退或停止选择；全部预测先封存再读评分真值。五条轨迹很少，时间帧相关，不能说成505个独立实验或未打开外部验证。

三维目标使用固定32×16×16网格、零外层支撑与内部去均值；观测是干净的straight-ray梯度积分代理，不是实测像素位移。当前扩展探测器为50×33，共29,700个标量观测，旧32×16版本只有9,216个；不能拿新旧每次A调用当成相同工作。新增采样改善几何覆盖，但不等于算法提升。

每帧必须同时满足field、full-gradient、interior-gradient、observation四个误差均不高于CGLS K2，且不超过CGLS K4的1.01倍；整条轨迹要求全部101帧通过。K4是相对比较器，不是绝对正确的三维真值证书。下表统一复核封存的两套评分，未重跑、未改判据。

| 方法 / Method | 同时通过四指标 / Cells | 完整轨迹 / Trajectories | 逻辑在线调用 / Calls |
|---|---:|---:|---:|
| Zero / Zero | 0/505 | 0/5 | 0A + 0AT |
| 观测线搜索 BP / CGLS K1 / Line-search BP / CGLS K1 | 0/505 | 0/5 | 1A + 1AT |
| CGLS K2 / CGLS K2 | 0/505 | 0/5 | 2A + 2AT |
| Jacobi-PCGLS K2 / Jacobi-PCGLS K2 | 0/505 | 0/5 | 2A + 2AT |
| BP 起点再走 K1 / BP restart + K1 | 0/505 | 0/5 | 2A + 2AT |
| CGLS K4 相对参考 / CGLS K4 relative reference | 505/505 | 5/5 | 4A + 4AT |
| 直接场 ridge + K1 / Direct-field ridge + K1 | 0/505 | 0/5 | 2A + 1AT |
| Dual ridge + K1 / Dual ridge + K1 | 0/505 | 0/5 | 2A + 2AT |
| 共享标量 + K1 / Shared scalar + K1 | 0/505 | 0/5 | 2A + 2AT |
| 四系数局部滤波 + K1 / Four-coefficient local filter + K1 | 0/505 | 0/5 | 2A + 2AT |
| 固定逆频率滤波 + K1 / Fixed inverse-frequency filter + K1 | 0/505 | 0/5 | 2A + 2AT |
| 随机特征神经算子 + K1 / Random-feature neural operator + K1 | 409/505 | 2/5 | 2A + 2AT |
| 线性化对照 + K1 / Linearized control + K1 | 412/505 | 1/5 | 2A + 2AT |
| 固定系数去全局消息消融 / Fixed-coefficient context-off ablation | 412/505 | 1/5 | 2A + 2AT |

BP指观测线搜索后的反投影，在此线性问题中与零起点CGLS K1等价。单元通过数不是总体成功概率；即使K4在表中通过，它也是定义比较线的参考，不是一次新算法发现。

## 最小神经比较与便宜解释

随机特征比较器以每条射线的带符号双分量观测和几何编码为输入，包含跨相机全局均值上下文。隐层352个随机量固定，仅学习17个共享读出系数。训练目标是K4的场、全梯度、内部梯度和投影的等权归一化物理误差，不是把留出CFD真值喂给预测器。它是读出训练的随机特征神经算子，不能概括所有完整训练的神经算子。

| 轨迹 / Trajectory | 直接场ridge | 神经比较器 | 线性化对照 | 固定系数去消息消融 |
|---|---:|---:|---:|---:|
| p14-s05 | 0/101 | 101/101 | 101/101 | 101/101 |
| p22-s03 | 0/101 | 91/101 | 98/101 | 98/101 |
| p33-s01 | 0/101 | 101/101 | 100/101 | 100/101 |
| p45-s05 | 0/101 | 22/101 | 23/101 | 23/101 |
| p58-s03 | 0/101 | 94/101 | 90/101 | 90/101 |

神经比较器在501/505帧的四指标全部优于直接场ridge，线性化对照为502/505，但少数尾部仍失败；该ridge还少一次AT，不能从公平对照中删掉。两者均优于便宜的K2/标量基线，却未稳定守住K4误差。这是有局部学习改善、无完整稳定优势，不是“所有学习都没用”，也不是算法突破。

## 解释出了什么

线性化模型的全局观测消息范数贡献极小：初始场最大0.0245%，投影最大0.0273%。保持系数不变关闭它，505帧的整体通过/失败一个不变；41帧四指标全改善、12帧全变差，其余混合。最大指标恶化比1.00012031，严格“删掉无损”失败。这是事后消融，不是重新训练或成功的简化候选；它也保留了几何上下文，且移除了包含同一相机的全局观测均值，不能叫纯跨相机删项。

这说明当前线性化实现的有效观测耦合很弱，不证明所有全局池化失败、不证明抵消是唯一原因，也不证明改成局部网络会成功。相关的[极线特征传递](https://arxiv.org/abs/2005.04551)针对点对应；BOS则沿光线积分梯度，不能直接照搬。

## 追加：物理消息有信息，但普通方向已经解释很多

五条已打开轨迹各取5个预先固定的等间隔帧，共25帧。把精确物理消息分为跨相机项、同相机项，以及两者之和的普通normal项 `AA^T y`。在不训练、不运行K1的情况下，检验单个方向可消除多少当前17方向无法拟合的K4教师损失。这里是四块归一化平方教师损失的表示下界，不是三维真值误差或最终重建改善。

| 轨迹 / Trajectory | 跨相机 / Cross-camera | 普通normal / Whole normal | 同相机 / Same-camera |
|---|---:|---:|---:|
| p14-s05 | 43.29% | 40.36% | 10.30% |
| p22-s03 | 37.85% | 38.62% | 13.08% |
| p33-s01 | 31.25% | 32.22% | 20.07% |
| p45-s05 | 51.46% | 50.93% | 34.28% |
| p58-s03 | 29.16% | 30.62% | 21.26% |

表中是逐轨迹中位可消除比例。跨相机项在五条轨迹都超过预设25%门，但只在10/25帧不弱于更便宜的普通normal项，未通过要求全部25帧守住的对照门。结论是物理方向确实补充信息，但不能把它包装成单独跨相机的稳定额外价值；也不证明所有局部耦合无效。

独立矩阵/梯度重建与直接QR残差计算复核了正式Schur计算，比例最大绝对差约1.01e-12，离散判定一致。两条实现共用冻结几何合同，不是独立实验标定。精确normal消息本身需要一次AT和A，隔离跨相机项还需要同相机计算；这些都不是免费特征。本次仅是25帧已开封诊断，没有新预测器、同精度调用减少或真实BOST结果。[Learned Primal-Dual](https://arxiv.org/abs/1707.06474)早已将物理前/反投影用于学习重建，这一思想本身不构成创新。

## 独立复算、成本和下一步

各比较器使用正式与独立物理/预测/评分实现；训练方向用独立矩阵和算子重建，读出用不同求解法，最终场和观测做直接重放。最新消融的场/指标最大相对差约1.07e-13/7.61e-14，汇总最大绝对差5.70e-14，判定一致。数值一致不等于科学成功。

表中只列预测壳的逻辑在线A/AT。训练教师、方向投影、几何矩阵构建、特征计算、缓存和内存都不是免费；评估重放也不计作部署节省。尚无同精度的稳定调用减少、fresh-process wall或whole-pipeline RSS优势。

精确normal对照已完成：后续机制必须说明如何用更便宜的观测/几何映射取得有效信息，并排除普通迭代方向的解释。当前固定表示和系数重训路线保持关闭；不改门、不加大模型追通过，现有数据仍可用于有依据的研究。本次仅用公开训练数据的固定干净几何，没有验证5/7/12相机精度、未知位姿、观测噪声、外部条件或真实BOST。私有数据不发布，也不重复索取已经找不到的实验配对。全部成功标志继续false。

# PoolFire: local gains reproduce, stable matched accuracy remains unmet

Across five public training trajectories and 505 frames with complete-trajectory LOTO, the random-feature neural operator passes 409 cells and 2/5 complete trajectories; the linearized control passes 412 cells and 1/5 trajectories. Neither passes the complete objective.

## Data, criterion and comparison

Existing data are usable: five original public PoolFire training trajectories provide 101 frames each on one nine-camera F30 geometry. Complete-trajectory LOTO fits on 404 frames and excludes all 101 query frames from input normalization, hyperparameters, fallback and stopping. Predictions seal before truth scoring. Frames are time-correlated, and five trajectories are a small sample, not 505 independent experiments or untouched external validation.

The fixed32×16×16 target has a zero outer support and demeaned interior. Clean observations are straight-ray gradient integrals, not measured pixel displacements. The expanded50×33 detector supplies29,700 scalar observations versus 9,216 on the old32×16 detector; one A call has different work across these grids. Better sampling coverage is not an algorithm gain.

Every cell must satisfy all four field/full-gradient/interior-gradient/observation errors<=CGLS K2 and<=1.01*CGLS K4. All101 frames must pass per trajectory. K4 is a relative comparator, not an absolute-truth certificate. The first table reaggregates independently sealed scores without a rerun or criterion change. Line-search BP equals zero-start CGLS K1 here. Cell counts are not population success probabilities; K4 remains the defining reference, not a new discovery.

## Learned comparator and cheaper explanation

The random-feature comparator reads signed two-component ray observations and geometry with a global multiview mean context. Its352 random hidden quantities are fixed; only 17 shared readout coefficients are fitted. Training distills equal normalized physical K4 field/full-gradient/interior-gradient/projection losses. Query CFD truth is never a prediction input. This readout-trained random-feature neural operator does not represent all fully trained neural operators.

The second table compares identical trajectory rosters. The neural comparator dominates direct-field ridge in all four metrics on 501/505 cells; the linearized control does so on 502/505. The remaining tails matter, and ridge uses one fewer AT. Both improve on K2/scalar controls but do not consistently preserve K4 errors. This is local learned improvement without complete stable advantage, not evidence that all learning fails or that an algorithm breakthrough exists.

## What the diagnosis establishes

The linearized global observation message contributes at most0.0245% of initializer field norm and 0.0273% of projected-field norm. Removing it with coefficients fixed changes none of 505 joint decisions. The ablation improves all four metrics on 41 cells, worsens all four on 12, and has mixed effects otherwise. Maximum error ratio is 1.00012031, so strict harmless removal fails. This is post-open ablation, not a refitted or successful simplified predictor. Geometry context remains; the removed global observation mean includes the same camera, so this is not a pure other-camera-only intervention.

The actual observation coupling is weak in this fixed instance. This does not show all pooling fails, cancellation is the sole cause, or a local network will succeed. [Epipolar feature transfer](https://arxiv.org/abs/2005.04551) concerns point correspondence; BOS instead integrates gradients along rays, so direct transplantation is not justified.

## Added: physical messages carry information, but ordinary normal information explains much of it

Five fixed equally spaced frames from each of the five opened trajectories give 25 post-open probes. Exact signed messages are split into cross-camera and same-camera terms; their sum is the ordinary normal message `AA^T y`. Without training or K1 refinement, the third table measures the median fraction of the existing 17-direction initializer's K4 teacher-loss floor removed by one cue. This is a four-block normalized squared teacher-loss diagnostic, not CFD truth error or final reconstruction improvement.

Cross-camera medians exceed the preset 25% gate in all five trajectories, but cross is at least as useful as the cheaper ordinary-normal cue on only 10/25 frames. The required all-25 control comparison therefore fails. Physical directions supply complementary information, but isolating this cross-camera moment has no established stable extra value. This does not prove all local coupling useless.

Independent matrices, hand-coded gradients and direct QR residual projections verify the formal Schur computation; fraction differences are at most about 1.01e-12 and decisions agree. Both share the frozen geometry contract, not independent experimental calibration. A normal message requires an AT and an A; isolating cross-camera information adds same-camera work. These are not free features. No new predictor, matched-accuracy call saving or real-BOST result follows from these 25 teacher-visible probes. [Learned Primal-Dual](https://arxiv.org/abs/1707.06474) already integrates physical forward/backprojection into learned reconstruction; that idea itself is not a novelty claim.

## Independence, accounting and limits

Formal and independent physics/prediction/scoring implementations reconstruct training directions and readouts and directly replay final fields/projections. For the latest ablation, maximum field/metric relative differences are 1.07e-13/7.61e-14; summary absolute difference is 5.70e-14, with matching decisions. Numerical agreement is not scientific success.

The table lists logical online A/AT calls only. Offline teachers, direction projections, geometry construction, features, caches and memory are not free; evaluation replay is not deployment saving. Stable same-accuracy call reduction, fresh-process wall and whole-pipeline RSS benefit remain unproven.

The exact-normal comparison is complete. Any next mechanism must justify a genuinely cheaper observation/geometry map and exclude ordinary iterative-direction explanations. The fixed representations and readout-retraining route remain closed; no threshold or larger-model rescue. Existing data remain usable for justified research. This fixed clean public-train geometry does not validate 5/7/12-camera accuracy, unseen pose, noise, external conditions or real BOST. Private data are not published and already-lost experimental pairs are not repeatedly requested. All success flags remain false.
