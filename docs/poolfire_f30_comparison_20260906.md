# PoolFire：局部改善已复现，稳定同精度尚未通过

2026-09-06。同一505帧完整轨迹留一：369参数全训练比较器通过420帧、1/5条完整轨迹；旧随机特征为409帧、2/5，线性对照为412帧、1/5。稳定同精度与便宜对照门仍未通过。

## 全训练小模型：帧数增加，完整目标仍失败

原比较器只训练17个读出值。这次保持8/8宽度与同一初始化，训练全部369参数，另存每折一个仅由训练集决定的输出尺度。五折各20轮，共10,100步；只用训练轨迹的K4教师，不用留出真值、时间或轨迹标签调参，不挑最佳轮次。预测先封存再评分。优化方式不同，这不是隐层训练的单因素因果实验，固定预算也不保证收敛。

逐轨迹通过101/89/90/49/91帧，合计420/505、1/5完整轨迹。p45由旧随机特征22帧提高到49帧，但p33由101降到90；总数增加不等于稳定性提高。更便宜直接场ridge的四指标无伤害仅504/505帧，标量对照为505/505，固定审裁仍失败。field/full-gradient/interior-gradient/observation超过1.01倍K4的帧数分别58/12/21/57，计数有重叠。

第二实现共享封存的最终权重，独立重建推理、精确lift、未修改K1与505帧物理评分；不是独立重训。100个局部训练梯度及优化器转移另行核对，不代表第二条完整优化轨迹。预测/场/指标最大相对差约6.71e-15/3.87e-15/1.48e-15，汇总7.78e-14，离散判定一致。

训练本身80,800A+80,800AT；两实现教师图像各505A，独立梯度探针800A+800AT，两实现最终重建各1,010A+1,010AT、评分各505A，另有29,700行解析几何构建。候选逻辑在线仍2A+2AT。约72.53分钟、峰值3.916GiB只作本次运行遥测，不是fresh资源比较或加速结论。

关闭此固定预算实例，不加轮数、换学习率/种子/宽度追通过；不概括所有神经网络失败。当前只有局部改善，仍无完整稳定学习优势、速度或论文突破。

## 损失归因：不是只差把最后一层训好

固定隐层的独立审计：精确优化读出仅能再消除1.9%-4.3%的训练目标损失；505帧留出初始化教师损失均以现有方向外的部分为主，逐轨迹中位占比81.6%-88.2%。学到的隐层改善464/505帧的表示下界，但不等于最终K1重建成功，也不证明整个网络已经收敛；没有替换模型或追加训练。

| 轨迹 / Trajectory | 训练目标可消除 / TRAIN removable | 留出表示下界占比中位 / QUERY floor fraction median | 下界降低帧 / Lower floor |
|---|---:|---:|---:|
| p14-s05 | 4.28% | 81.63% | 101/101 |
| p22-s03 | 2.40% | 85.16% | 101/101 |
| p33-s01 | 1.90% | 87.90% | 60/101 |
| p45-s05 | 2.63% | 88.21% | 101/101 |
| p58-s03 | 2.62% | 86.91% | 101/101 |

这里的下界是给定这17个学到的物理方向、允许逐样本读取K4教师的正交投影剩余，不是所有网络的能力上限或最终CFD误差下界。固定训练预算仍关闭。即使精确优化TRAIN读出，p45留出教师损失均值也从0.141960变为0.146902，说明训练目标下降不保证迁移改善。没有用诊断系数替换已封存预测。

两套实现独立重建2,525个折-样本物理状态、505个查询投影及5个TRAIN最优解；QR/直接SVD和Cholesky/eigh的分解、目标与汇总差均不超过3.65e-14。每套离线42,925A+42,925AT及505次教师图像A，另有29,700行解析几何。约38.93分钟、峰值3.676GiB只作审计遥测；没有新K1、CFD真值评分或部署节省。

因此下一步不能只依赖读出微调，应先说明新的低成本物理信息如何进入预测器。这不是已经找到新机制。线性与非线性变量分离是既有数值分析思想：[Golub与Pereyra，1973](https://epubs.siam.org/doi/10.1137/0710036)，不是本项目首创。

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
| 全训练小型神经比较器 + K1 / Fully trained small neural comparator + K1 | 420/505 | 1/5 | 2A + 2AT |
| 固定系数去全局消息消融 / Fixed-coefficient context-off ablation | 412/505 | 1/5 | 2A + 2AT |

BP指观测线搜索后的反投影，在此线性问题中与零起点CGLS K1等价。单元通过数不是总体成功概率；即使K4在表中通过，它也是定义比较线的参考，不是一次新算法发现。

## 最小神经比较与便宜解释

随机特征比较器以每条射线的带符号双分量观测和几何编码为输入，包含跨相机全局均值上下文。隐层352个随机量固定，仅学习17个共享读出系数。训练目标是K4的场、全梯度、内部梯度和投影的等权归一化物理误差，不是把留出CFD真值喂给预测器。它是读出训练的随机特征神经算子，不能概括所有完整训练的神经算子。

| 轨迹 / Trajectory | 直接场ridge | 神经比较器 | 线性化对照 | 固定系数去消息消融 | 全训练比较器 |
|---|---:|---:|---:|---:|---:|
| p14-s05 | 0/101 | 101/101 | 101/101 | 101/101 | 101/101 |
| p22-s03 | 0/101 | 91/101 | 98/101 | 98/101 | 89/101 |
| p33-s01 | 0/101 | 101/101 | 100/101 | 100/101 | 90/101 |
| p45-s05 | 0/101 | 22/101 | 23/101 | 23/101 | 49/101 |
| p58-s03 | 0/101 | 94/101 | 90/101 | 90/101 | 91/101 |

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

## 追加：不能把精确normal缓存当作便宜特征

在当前一套扩展F30九相机几何上，只检查 `B=AA^T` 的结构，不读观测、CFD真值或教师场，不训练、不重建。按结果前固定的浮点舍入阈值去掉数值零后，A含2,960,876个非零项，B仍有64,746,812个，CSR需777,080,548字节，约741.08 MiB。内存没有超过预设1 GiB上限；失败的是算术项数门：一次B乘法为直接A/AT分解的10.933726倍，即使保守数值下界也有10.931226倍。

正式逐基向量forward/稀疏乘积与独立解析行/稠密点积重建了全部29,700行；第二实现实际倒序相机。逐行支持计数完全一致，完整Gram相对差约1.68e-16，保留矩阵作用与直接A/AT的探针差不超过6.70e-15。两实现仍共用同一几何合同，不是独立实验标定。只存小型计数证据，没有保存大矩阵。

结论仅是当前几何的精确normal CSR未通过便宜控制筛选。**10.93倍是算术项数，不是实测慢了10.93倍**。离线构造和B乘法不能从成本账中消失；旧三轴简化几何的稀疏捷径不能直接套用。未测试新的近似、结构化分解或学习器，505帧重建结论不变，不能宣称调用减少、资源优势或算法突破。

# PoolFire: local gains reproduce, stable matched accuracy remains unmet

On the same 505-frame complete-trajectory LOTO, the fully trained 369-parameter comparator passes 420 cells and 1/5 trajectories; the old random features pass 409 and 2/5, and the linear control 412 and 1/5. Stable matched accuracy and the cheaper-control gate remain unmet.

## Fully trained small model: more passing cells, still no complete success

The old comparator fitted only17 readout values. Keeping width8/8 and the same initialization, this run trains all369 parameters plus one stored train-only output scale per fold. Five folds each use20 epochs, totaling10,100 steps. Only training-trajectory K4 teachers enter fitting; query truth, time and trajectory IDs do not guide tuning, stopping or iterate selection. Predictions seal before scoring. Optimization differs from the old readout solve, so this is not a single-factor causal ablation and finite-budget convergence is not guaranteed.

Trajectory counts are101/89/90/49/91, totaling420/505 and1/5 complete trajectories. p45 rises from the old random-feature22 to49, but p33 falls from101 to90. A larger total does not establish better stability. All-four nonharm against cheaper direct-field ridge holds on504/505 cells; the scalar control on505/505. The fixed adjudication fails. Counts exceeding1.01 timesK4 are58/12/21/57 for field/full-gradient/interior-gradient/observation, with overlaps.

The second implementation shares sealed final weights but independently reconstructs inference, exact lift, unchangedK1 and all505 physical scores; it is not independent retraining. Separate checks of100 local gradients and optimizer transitions are not a second full optimization trajectory. Prediction/field/metric maximum relative differences are about6.71e-15/3.87e-15/1.48e-15, with summary7.78e-14 and identical discrete decisions.

Training alone costs80,800A+80,800AT. Each implementation prepares505 teacher-imageA calls; independent gradient probes add800A+800AT; each final reconstruction uses1,010A+1,010AT and scoring505A, plus29,700 analytical geometry rows. Logical online cost remains2A+2AT. About72.53 minutes and3.916GiB peakRSS are run telemetry, not a fresh resource comparison or speedup.

This fixed-budget instance closes without extra epochs, learning-rate/seed/width rescue. It does not establish that all neural networks fail. Local improvements remain, without complete stable learned advantage, speedup or a paper breakthrough.

## Loss attribution: not merely an unfinished last layer

With hidden features fixed, an independent audit finds only 1.9%-4.3% of the training objective removable by exact readout optimization. All 505 query initializer teacher losses are dominated by the part outside the current span, with trajectory medians of 81.6%-88.2%. Learned features lower the span floor on 464/505 cells, not a final-K1 success or proof of full-network convergence. No model replacement or extra training occurs.

The bilingual loss-attribution table reports a teacher-visible orthogonal-projection residual in the particular 17 learned physical directions, not the capacity limit of all networks or a final CFD-error bound. The fixed training budget remains closed. Even exact TRAIN-head optimization changes p45 query mean teacher loss from0.141960 to0.146902: lower training loss does not ensure better transfer. Diagnostic coefficients never replace sealed predictions.

Two implementations rebuild2,525 fold-sample physical states,505 query projections and5 TRAIN optima. QR/directSVD and Cholesky/eigh decomposition, objective and summary differences are at most3.65e-14. Each path costs42,925A+42,925AT plus505 teacher-imageA calls, with29,700 analytical geometry rows. About38.93 minutes and3.676GiB peakRSS are audit telemetry, not newK1, CFD truth scoring or deployment savings.

Readout adjustment alone is not the supported next explanation; a new predictor must first justify genuinely different cheap physical information transfer. No such successful mechanism is established here. Linear/nonlinear variable separation is classical numerical analysis: [Golub and Pereyra,1973](https://epubs.siam.org/doi/10.1137/0710036), not a project novelty.

## Data, criterion and comparison

Existing data are usable: five original public PoolFire training trajectories provide 101 frames each on one nine-camera F30 geometry. Complete-trajectory LOTO fits on 404 frames and excludes all 101 query frames from input normalization, hyperparameters, fallback and stopping. Predictions seal before truth scoring. Frames are time-correlated, and five trajectories are a small sample, not 505 independent experiments or untouched external validation.

The fixed32×16×16 target has a zero outer support and demeaned interior. Clean observations are straight-ray gradient integrals, not measured pixel displacements. The expanded50×33 detector supplies29,700 scalar observations versus 9,216 on the old32×16 detector; one A call has different work across these grids. Better sampling coverage is not an algorithm gain.

Every cell must satisfy all four field/full-gradient/interior-gradient/observation errors<=CGLS K2 and<=1.01*CGLS K4. All101 frames must pass per trajectory. K4 is a relative comparator, not an absolute-truth certificate. The method-comparison table reaggregates independently sealed scores without a rerun or criterion change. Line-search BP equals zero-start CGLS K1 here. Cell counts are not population success probabilities; K4 remains the defining reference, not a new discovery.

## Learned comparator and cheaper explanation

The random-feature comparator reads signed two-component ray observations and geometry with a global multiview mean context. Its352 random hidden quantities are fixed; only 17 shared readout coefficients are fitted. Training distills equal normalized physical K4 field/full-gradient/interior-gradient/projection losses. Query CFD truth is never a prediction input. This readout-trained random-feature neural operator does not represent all fully trained neural operators.

The per-trajectory comparison table uses identical rosters. The random-feature comparator dominates direct-field ridge in all four metrics on 501/505 cells; the linearized control does so on 502/505. The remaining tails matter, and ridge uses one fewer AT. Both improve on K2/scalar controls but do not consistently preserve K4 errors. This is local learned improvement without complete stable advantage, not evidence that all learning fails or that an algorithm breakthrough exists.

## What the diagnosis establishes

The linearized global observation message contributes at most0.0245% of initializer field norm and 0.0273% of projected-field norm. Removing it with coefficients fixed changes none of 505 joint decisions. The ablation improves all four metrics on 41 cells, worsens all four on 12, and has mixed effects otherwise. Maximum error ratio is 1.00012031, so strict harmless removal fails. This is post-open ablation, not a refitted or successful simplified predictor. Geometry context remains; the removed global observation mean includes the same camera, so this is not a pure other-camera-only intervention.

The actual observation coupling is weak in this fixed instance. This does not show all pooling fails, cancellation is the sole cause, or a local network will succeed. [Epipolar feature transfer](https://arxiv.org/abs/2005.04551) concerns point correspondence; BOS instead integrates gradients along rays, so direct transplantation is not justified.

## Added: physical messages carry information, but ordinary normal information explains much of it

Five fixed equally spaced frames from each of the five opened trajectories give 25 post-open probes. Exact signed messages are split into cross-camera and same-camera terms; their sum is the ordinary normal message `AA^T y`. Without training or K1 refinement, the normal-message table measures the median fraction of the existing 17-direction initializer's K4 teacher-loss floor removed by one cue. This is a four-block normalized squared teacher-loss diagnostic, not CFD truth error or final reconstruction improvement.

Cross-camera medians exceed the preset 25% gate in all five trajectories, but cross is at least as useful as the cheaper ordinary-normal cue on only 10/25 frames. The required all-25 control comparison therefore fails. Physical directions supply complementary information, but isolating this cross-camera moment has no established stable extra value. This does not prove all local coupling useless.

Independent matrices, hand-coded gradients and direct QR residual projections verify the formal Schur computation; fraction differences are at most about 1.01e-12 and decisions agree. Both share the frozen geometry contract, not independent experimental calibration. A normal message requires an AT and an A; isolating cross-camera information adds same-camera work. These are not free features. No new predictor, matched-accuracy call saving or real-BOST result follows from these 25 teacher-visible probes. [Learned Primal-Dual](https://arxiv.org/abs/1707.06474) already integrates physical forward/backprojection into learned reconstruction; that idea itself is not a novelty claim.

## Independence, accounting and limits

Formal and independent physics/prediction/scoring implementations reconstruct training directions and readouts and directly replay final fields/projections. For the latest ablation, maximum field/metric relative differences are 1.07e-13/7.61e-14; summary absolute difference is 5.70e-14, with matching decisions. Numerical agreement is not scientific success.

The table lists logical online A/AT calls only. Offline teachers, direction projections, geometry construction, features, caches and memory are not free; evaluation replay is not deployment saving. Stable same-accuracy call reduction, fresh-process wall and whole-pipeline RSS benefit remain unproven.

The exact-normal comparison is complete. Any next mechanism must justify a genuinely cheaper observation/geometry map and exclude ordinary iterative-direction explanations. The fixed representations and readout-retraining route remain closed; no threshold or larger-model rescue. Existing data remain usable for justified research. This fixed clean public-train geometry does not validate 5/7/12-camera accuracy, unseen pose, noise, external conditions or real BOST. Private data are not published and already-lost experimental pairs are not repeatedly requested. All success flags remain false.

## Added: an exact-normal cache is not a cheap feature here

This checks only `B=AA^T` on the current expanded F30 nine-camera geometry: no observations, CFD truth, teacher fields, training or reconstruction. The preregistered roundoff-only cutoff leaves 2,960,876 nonzeros in A and 64,746,812 in B. B would need 777,080,548 CSR bytes, about 741.08 MiB, within the preset 1 GiB memory ceiling. Arithmetic eligibility fails: one B product has 10.933726 times the scalar multiply-add terms of factorized A/AT; even the conservative numerical lower bound is 10.931226.

Formal canonical-column forward/sparse products and independent analytical-row/dense products reconstruct all 29,700 rows. The second implementation actually reverses camera order. Per-row support counts agree exactly; complete-Gram relative difference is about 1.68e-16 and retained-action versus factorized probes differs by at most 6.70e-15. Both share the geometry contract, not independent experimental calibration. Only small count evidence remains; no large matrix is persisted.

Only this exact-normal CSR cheap-control screen fails. **10.93 is an arithmetic-term ratio, not a measured 10.93-fold slowdown.** Offline construction and B multiplication cannot vanish from the ledger; old three-axis sparsity does not transfer automatically. No new approximation, structured factorization or learner was tested. The 505-frame reconstruction conclusions remain unchanged, with no call-saving, resource or algorithm-breakthrough claim.
