# Hybrid-GCV 参考资格审查 / Hybrid-GCV Reference Qualification

日期 / Date: 2026-10-05

## 中文结论

**独立实现一致，但固定参考资格不通过。** 重正交的 Hybrid LSQR（全观测空间 GCV 选参）在 11 条已开封 PoolFire 训练轨迹、每条 101 帧和 5/7/9 相机上覆盖 3,333 个单元、23,331 条检查点评分。36 个分层是 33 个轨迹×相机层，加 3 个按相机合并层；它们不是 36 组独立样本，也不是新的未见数据测试。

- 128 步的 field/full-gradient/observation 绝对 p90 门分别为 0.50/0.75/0.20，**36/36 分层通过**。
- 冻结的参考稳定性要求：64 到 128 步，field/full-gradient/interior-gradient/observation 四项 p90 的绝对变化在每层均不超过 0.001。**0/36 分层通过**，四项指标各自都在全部 36 层超门。
- 两套独立算子/求解实现的全部保存状态及物理指标已核对。恢复审查 13/13 项通过；逐检查点状态最大相对差为 3.06e-11，物理指标最大绝对差为 1.84e-11，分层汇总最大差为 1.39e-12。离散判决完全一致。

| 64 到 128 步的变化 | 36 层中最大 p90 绝对变化 | 冻结上限 |
|---|---:|---:|
| 场 | 0.049488 | 0.001 |
| 全梯度 | 0.079524 | 0.001 |
| 内部梯度 | 0.083842 | 0.001 |
| 观测 | 0.005240 | 0.001 |

### 做了什么，为什么

此前未重正交的实现未通过数值资格检查。另行冻结的本次实现使用完整基的双遍重正交，以及与之对应的稠密投影问题；它通过数值资格后才进行本次场评分。Hybrid LSQR/GCV 是经典对照，不是新提出的学习算法。这里要回答的是：它能否提供一个满足冻结精度与稳定性要求的参考，供后续 warm start 公平比较。

独立运行已经完成全部 3,333 个单元，但最终汇总遗漏了外层记录的轨迹/相机标签，没有生成原完成记录。审查在新的目录中仅恢复内存中的标签视图，再核对全部原始输出和数据身份；没有修改原文件、数值、阈值或原完成记录，也没有重跑求解器。源码缺陷已在纯元数据记录上复现；没有保留下来的历史异常日志，因此不声称看到过当时的异常堆栈。

### 后续参数诊断：显式正则化几乎没有收缩解

对同一批**已开封**结果作只读诊断：两套独立实现保存的 23,331 条正则化参数与有效自由度已逐条核对，λ 的最大差为 9.11e-18，有效自由度的最大差为 2.84e-14。λ 全部非零，但 `k − effective_dof` 的中位数仅 1.59e-5，最大值仅 3.58e-5。因此，在相同 Krylov 子空间内，显式 Tikhonov 对解的相对收缩上界约为 3.58e-5；迭代截断本身仍可能提供正则化。

冻结代码用原始有效观测维数 `m = 8446/11954/15446`（对应 5/7/9 相机）计算 `||r||²/(m − effective_dof)²`，也用这个式子选 λ。它是投影估计器的**全观测空间 GCV**，不同于仅在 `(k+1)` 维投影问题上选参的普通 GCV；不能仅凭不同就断言前者数学上错误。由于 `m` 远大于 `k`，该式的分母随 λ 变化很小，与这次近乎无显式收缩的观测一致。训练观测由 `y=Aρ` 无噪声生成。此诊断指出可检验的机制线索，**未证明它是稳定性失败的唯一原因**；原 0/36 稳定性判决不变，也未事后改用另一条 GCV 规则宣称成功。只读诊断新增 0A+0A^T。[Hybrid 方法综述](https://arxiv.org/html/2105.07221v2)、[IR Tools 的投影 GCV 实现](https://github.com/jnagy1/IRtools/blob/master/Extra/TikGCV.m)。

**补齐汇总是工程修复；经独立一致性核验后拒绝参考资格，才是本轮科学判断。** 本次恢复新增 0A+0A^T。原算法在 128 步的逻辑账为每样本 128A+128A^T，另有几何准备、重正交、投影 SVD/GCV 和基向量存储成本，不能把这些成本当成免费或以逻辑调用账替代实测速度。

### 独立机制对照：换成普通投影 GCV 仍未稳定

另行结果前冻结了一次 **99 单元哨兵对照**：同一 11 条已开封训练轨迹的首、中、末三帧，5/7/9 相机；不是完整轨迹验证。唯一变化是普通投影 GCV 用 `(k+1−effective_dof)²` 作为分母，与旧规则的 `(m−effective_dof)²` 比较。两规则共用相同方向、迭代、参数网格、物理算子和预算，每个样本只有一次 128A+128A^T 子空间构建；真值不参与选参。

两套实现各产生 396 条候选评分，**14/14 独立检查通过**。旧 control 精确复现封存状态、选参和指标，三个最大差均为 0；新对照的状态、投影矩阵、二维观测和指标最大差分别为 6.99e-12、3.46e-10、3.31e-15、4.01e-12。

| 同一 99 单元对照 | K128 绝对门 | K64→K128 四指标稳定性 | 完整实验扩展 |
|---|---:|---:|---|
| 普通投影 GCV | 36/36 分层 | 0/36 分层 | 不授权 |
| 旧全观测空间 GCV | 36/36 分层 | 0/36 分层 | 不授权 |

普通投影规则确实增强了显式过滤：K128 的 `k−effective_dof` 中位数在 5/7/9 相机下为 0.178852/0.289410/0.462285，旧 control 为约 2.95e-5/2.02e-5/2.15e-5。但普通投影规则在两个检查点的全部 99 个样本上，场、内部梯度和观测误差都更大；全梯度只在 K64 的 13/99、K128 的 17/99 样本改善。更强过滤没有解决固定稳定性门。

这关闭“**只更换 GCV 选参分母就足以解决当前无噪声参考问题**”的假设，不扩展为完整 3,333 单元实验，不改网格、参数范围、权重、步数或门来救援。它不否定所有 GCV 方法或 C 路线，也未确定唯一根因。离线每套实现另有 99A 观测生成和 396A 物理评分回放；重正交、投影求解和存储仍有实际成本，没有实测速度或学习算法突破。[本次脱敏对照摘要](poolfire_projected_gcv_sentinel_2026-10-05_public_summary.json)。

### 对路线的影响

这关闭本次固定 Hybrid-GCV 参考尝试，不追加深度、不改阈值或正则化参数来救援。参考未合格时，较少步数的检查点只能作描述性诊断，不能据此声明 matched-accuracy 调用节省。该结论不证明全部 Hybrid 方法或 C 路线不可能，也不允许把不同历史算子合同下的数字直接合并。

下一项比较必须先有独立可解释的质量目标或合格参考、部署时可见的停止规则，以及同价或更便宜的经典控制。不能将“都通过宽泛绝对门”偷换为“四项误差匹配”。现有三维场仍能继续受控虚拟研究，找不到的旧二维实验配对不是虚拟研究的前置条件；像素级/真实 BOST 结论仍需要实际光学尺度和实验依据。

未打开 validation/test/独立外部数据；未训练新的预测器；没有 wall/RSS、外部泛化、真实 BOST、算法突破或论文成功结论。

## English Result

**Independent implementations agree, but this fixed reference fails qualification.** Reorthogonalized Hybrid LSQR with full-observation-space GCV selection covers 3,333 cells and 23,331 checkpoint scores from 11 already-opened PoolFire training trajectories, 101 frames each, and 5/7/9 cameras. The 36 strata comprise 33 trajectory-camera strata and three camera-pooled strata; they are not 36 independent datasets or a fresh generalization test.

At 128 steps, all **36/36 strata** pass absolute p90 limits of 0.50 for field, 0.75 for full gradient and 0.20 for observation. However, **0/36** pass the frozen reference-stability requirement: the absolute p90 change from 64 to 128 steps must not exceed 0.001 for any of field, full gradient, interior gradient or observation. Each metric exceeds that limit in every stratum. Their maximum changes are respectively **0.049488, 0.079524, 0.083842 and 0.005240**.

All stored states and physical metrics from the separate operator/solver implementations were compared. All 13 recovery checks pass. Maximum per-checkpoint state relative, physical-metric absolute and summary absolute differences are **3.06e-11, 1.84e-11 and 1.39e-12**. Every discrete decision agrees. Numerical consistency does not imply a stable reconstruction reference.

### Recovery and Cost

The earlier implementation without reorthogonalization failed numerical eligibility. This separately frozen attempt uses two full reorthogonalization passes and the corresponding dense projected problem, passing numerical eligibility before field scoring. Hybrid LSQR/GCV is a classical control, not a new learned method.

The independent run computed all 3,333 cells, but its final summarizer omitted trajectory/camera labels present in enclosing records. Recovery reconstructed only an in-memory metadata view and verified all original outputs and input identities in a separate audit. Original files, numerical values, gates and original receipts remain untouched; no solver was rerun. A metadata-only record reproduces the source defect. No historical exception log survives, so no original traceback is claimed.

### Later Parameter Diagnosis: Almost No Explicit Shrinkage

A read-only, **post-open** audit compared all 23,331 saved parameter and effective-degree-of-freedom rows from both independent implementations. Their maximum differences are 9.11e-18 for λ and 2.84e-14 for effective degrees of freedom. Every λ is positive, yet median and maximum `k − effective_dof` are only 1.59e-5 and 3.58e-5. Within the same Krylov subspace, the relative solution change due to explicit Tikhonov filtering is thus bounded by about 3.58e-5. Truncating the Krylov iteration may still regularize the solution.

The frozen code selects λ using `||r||²/(m − effective_dof)²`, where the original active observation dimensions `m` are 8446/11954/15446 for 5/7/9 cameras. This is **full-observation-space GCV for the projected estimator**, not ordinary GCV on the `(k+1)`-dimensional projected problem. The distinction alone does not make the full-space criterion mathematically invalid. Here `m` greatly exceeds `k`, so the denominator varies little with λ, consistent with the observed weak explicit filtering. The training observations are noiseless synthetic `y=Aρ`. This is a testable mechanism clue, **not proof of a unique cause** of failed stability. The original 0/36 stability verdict remains, with no post-hoc switch to another GCV rule or new success claim. Readback adds 0A+0A^T. [Hybrid projection survey](https://arxiv.org/html/2105.07221v2), [IR Tools projected-GCV implementation](https://github.com/jnagy1/IRtools/blob/master/Extra/TikGCV.m).

Summary recovery is engineering; the independently checked reference rejection is the scientific decision. Recovery adds **0A+0A^T**. The original 128-step algorithm logically requires **128A+128A^T per sample**, excluding nonfree geometry setup, reorthogonalization, projected SVD/GCV and basis storage. No fresh wall-time or RSS measurement was performed.

### Independent Mechanism Control: Ordinary Projected GCV Still Fails Stability

A separately frozen **99-cell sentinel** uses the first, middle and last opened frames of the same eleven train trajectories under 5/7/9 cameras. It is not complete-trajectory validation. The unique change is selecting with ordinary projected GCV, denominator `(k+1−effective_dof)²`, versus the old full-observation denominator `(m−effective_dof)²`. Directions, iteration, parameter grid, physical operator and budget are unchanged. Both rules share one 128A+128A^T subspace per cell; truth does not select parameters.

Each implementation produces 396 candidate-score rows. **14/14 independent checks pass**. The old control exactly reproduces sealed parent states, selections and metrics, all with maximum difference zero. New state, projected-matrix, observation and metric differences are bounded by 6.99e-12, 3.46e-10, 3.31e-15 and 4.01e-12.

Both rules pass **36/36 K128 absolute strata**, but **0/36** pass K64-to-K128 four-metric stability. Ordinary projected GCV produces genuinely stronger filtering: median K128 `k−effective_dof` is 0.178852/0.289410/0.462285 for 5/7/9 cameras, versus roughly 2.95e-5/2.02e-5/2.15e-5 for the old control. Nevertheless, field, interior-gradient and observation errors worsen in all 99 cells at both checkpoints. Full-gradient error improves in only 13/99 at K64 and 17/99 at K128. Stronger filtering does not qualify the fixed reference.

Close the hypothesis that **switching the GCV selection denominator alone solves this noiseless reference problem**. Do not escalate to 3,333 cells or change the grid, parameter range, weights, depth or gates to rescue it. This is not a general rejection of GCV or the C route, or proof of a unique cause. Per implementation, offline observation generation adds 99A and physical scoring adds 396A; reorthogonalization, projected solving and storage are nonfree. No measured speed or learned-algorithm breakthrough follows. [Redacted sentinel-control summary](poolfire_projected_gcv_sentinel_2026-10-05_public_summary.json).

### Consequence and Limits

Close this fixed reference attempt without increasing depth or changing thresholds or regularization to rescue it. With an unqualified reference, lower-step checkpoints are descriptive only, not evidence of matched-accuracy call savings. This does not rule out all Hybrid methods or the C route, and results from different historical operator contracts must not be pooled as if directly comparable.

Any next comparison needs an independently justified quality target or qualified reference, a deployment-visible stopping rule and equal-or-cheaper classical controls. Passing broad absolute gates is not four-metric accuracy matching. Existing 3D fields still support controlled virtual research; lost experimental 2D pairs are not a prerequisite for that work. Pixel-calibrated or real-BOST claims still require optical scale and experimental evidence.

No validation/test/independent external data were opened and no new predictor was trained. Algorithmic breakthrough, paper success, external generalization, resource speedup and real BOST all remain **false**.

[Redacted numerical summary](poolfire_hybrid_gcv_reference_2026-10-05_public_summary.json)
