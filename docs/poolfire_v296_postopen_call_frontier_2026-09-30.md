# v296 · Post-open matched-quality checkpoint frontier

**Evidence class:** post-open diagnostic on an already-open synthetic CFD roster. Not a prospective outer test.

## Question

At the sampled CGLS budgets, can the learned warm start match the four-metric error vector of Zero or normalized BP with fewer exact forward and adjoint calls?

## Frozen data and accounting

The diagnostic uses 25 already-open rows: five fixed offsets on each of five PoolFire trajectories. Observations came from the continuous straight-ray proxy; refinement used the frozen discrete operator, intentionally retaining that operator-mismatch stress condition. For every compared point, total counts of `A` and `A^T` were equal to one of 2, 3, 5, 9, or 17 per operator. A match requires all four truth-scored metrics (field, full gradient, interior gradient, observation residual) to be no worse. Only already-recorded checkpoints are searched.

## Result

At target budget 3, the warm iterate at budget 2 matched the Zero four-metric vector on 24/25 rows and the BP vector on 23/25, saving one `A` and one `A^T` application on those rows. At target budget 5, warm at budget 3 matched Zero on 4/25 rows and BP on 7/25, saving two applications of each operator. At target budgets 9 and 17, warm did not match either control at any lower sampled budget. By contrast, neither Zero nor BP matched the warm four-metric vector at a lower sampled budget on any row. The 17-call controls matched warm at the same budget on 4/25 rows.

Trajectory heterogeneity matters: at target budget 5, warm matched Zero on 4/5 rows for only one trajectory and on 0/5 for the other four; against BP the counts ranged from 0/5 to 4/5. This is an early-budget signal, not a stable complete-trajectory advantage.

## Independent numerical check

The deterministic checkpoint comparison was applied separately to the formal and independent metric arrays. Their maximum absolute difference was `2.44e-15`; the discrete match counts agreed. A script independently reproduces the aggregation from the two sealed arrays.

## Limits

This is truth-scored, post-open analysis on 25 selected rows, under a continuous-observation/discrete-solver mismatch. It does not establish an observation-only stopping rule, accuracy at checkpoints between the sampled budgets, full-trajectory tail control, fresh wall time/RSS, unseen-trajectory generalization, or real BOST. The early matched-quality call reduction is a hypothesis-generating result only. No algorithm breakthrough or paper-success claim is made.

### Plain-language summary

The learned start sometimes reached the same four-error quality one or two steps sooner at very early budgets. That advantage mostly disappeared at later budgets and varied by trajectory. We still do not know how a deployed solver could tell, without seeing the true 3D field, when that quality has been reached.

### 中文摘要

在已打开的 25 个样本和离散检查点上，学习初值在早期预算有事后同精度信号：目标预算 3 时，warm 用预算 2 达到 Zero/BP 四项误差向量的样本数为 24/25、23/25；目标预算 5 时降为 4/25、7/25；预算 9 和 17 时没有更早匹配。它不能证明存在可部署的真值无关停机规则，也不是整轨迹泛化、资源加速或算法突破。
