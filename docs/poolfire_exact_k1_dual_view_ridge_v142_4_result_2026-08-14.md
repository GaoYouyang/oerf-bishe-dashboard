# v142.4：共享线性 K1-dual 预测器未通过完整轨迹迁移门

日期：2026-08-14  
证据等级：已开封 PoolFire straight-ray 代理上的结果前冻结、完整轨迹留一开发实验  
正式判决：`FAIL_SHARED_LINEAR_RIDGE_REPRESENTATION_V142_4`

## 先说结论

固定 `pair_depth_projection_only` teacher 已经证明成对深度方向有表示容量，但当前部署可见特征无法把 teacher 系数跨完整轨迹做共享线性预测。

在五条 PoolFire 轨迹、`5/7/9/12` 相机和 `3700` 个单元上，正式特征视图与独立重建特征视图都只有 `1/3700` 个单元通过冻结四指标门，完整轨迹为 `0/5`。两套实现的逐单元指标比最大差只有 `2.59e-11`，远低于结果前固定的 `1e-4`，所以这不是几何重建数值漂移造成的假失败。

按照结果前规则，共享线性 ridge 路线现在关闭。本分支不允许改目标、重扫 lambda、事后选择另一个特征视图，或用更大的 CNN/FNO 挽救。

## 实验问题

候选只使用部署可见信息：

- exact CGLS K1 detector dual 与 residual；
- 报告相机几何；
- target、peer 与 active-camera pooled 特征；
- 成对深度方向的 observation-space 特征和随相机置换共同变化的 token。

五个完整 trajectory leave-one-out folds 每折使用四条轨迹拟合共享线性 ridge，在整条未参与拟合的轨迹上预测。正式视图使用封存特征，第二实现从几何和 K1 状态重新生成方向与特征；两者共享同一组已经封存的线性权重和归一化，独立视图不得重拟合。

候选通过 exact `A^T` lift、可观测线搜索和一次未修改 restarted CGLS，完整在线账为 `3A+3A^T`；参考 Zero-CGLS K4 为 `4A+4A^T`。

## 数值结果

| 评估臂 | 通过单元 | 通过完整轨迹 | 最坏逐单元四指标比 |
|---|---:|---:|---:|
| 正式特征视图共享线性 ridge | 1 / 3700 | 0 / 5 | 1.93336 |
| 独立特征视图共享线性 ridge | 1 / 3700 | 0 / 5 | 1.93336 |
| joint-LS warm-restart K1 control | 0 / 3700 | 0 / 5 | 1.74351 |

冻结门要求每个单元四指标比不超过 `1.05`，并要求每条轨迹各指标 `p90-higher <= 1.02`、`worst <= 1.05`。共享线性 ridge 在五条轨迹上全部失败：

- field 的逐轨迹 p90 比约为 `1.225` 至 `1.299`；
- full-gradient 的逐轨迹 p90 比约为 `1.054` 至 `1.102`；
- interior-gradient 的逐轨迹 p90 比约为 `1.126` 至 `1.178`；
- observation 的逐轨迹 p90 比约为 `1.579` 至 `1.861`。

observation 是最大缺口，但 field 与两个 gradient 也没有同时守门，因此不能通过“只强调某一指标改善”来保留候选。

## 独立复算与失败修复

第一次完整独立运行完成 `3700` 条预测和 `3700` 次物理重放后，仅在写最终 JSON 时遇到 NumPy 布尔标量不可序列化。该次尝试被永久保留为 `INCONCLUSIVE`，没有读取或复用它的 partial 数组。

机械后继只把 NumPy 标量转换为等值 Python 标量后再写 JSON；模型、数据、折分、阈值、物理重放、对照和决策规则均未改变。随后完整重跑，`19/19` 完整性检查通过：

- 所有独立预测先封存，再读取已开封评分真值；
- exact K1 dual、residual 与正式预测重算最大差均为 `0`；
- 两视图 detector-dual 相对 L2 最大差为 `4.41e-10`；
- 两视图逐单元指标比最大差为 `2.59e-11`；
- 两个视图和 joint-LS 都实际重放 `3A+3A^T`；
- 输入树、正式预测树和旧失败证据在验证前后不变。

共享的冻结 physics kernels 仍被两套实现共同使用，因此没有声称端到端物理实现完全独立。

## 对照与准确解释

Zero-CGLS K3、scaled BP K2 和 geometry-PCGLS K3 都没有通过完整 matched-accuracy 门。`scaled_bp_k2` 自己仍失败，但在冻结支配判据下依然优于当前线性预测器；这进一步说明当前 learned ridge 没有建立可保留的优势。

这个结果关闭的是“当前固定 teacher + 当前部署可见特征 + 共享线性 ridge”的映射，不是关闭整个 C 路线，也不是证明所有非线性或所有物理表示都不可能。下一步若继续，必须另行结果前冻结一个物理上不同、可证伪的表示或低成本安全门，不能在本分支事后调参。

## 证据边界

- `fixed_teacher_mechanism_capacity_proven=true`；
- `deployable_linear_predictor_proven=false`；
- `matched_accuracy_call_reduction_proven=false`；
- `resource_gate_authorized=false`；
- `algorithm_breakthrough=false`；
- `paper_success=false`；
- `external_generalization=false`；
- `resource_speedup=false`；
- `curved_ray_validated=false`；
- `real_bost=false`。

## English summary

The preregistered shared linear exact-K1 direction ridge fails complete-trajectory transfer. Across five opened PoolFire trajectories and 3,700 cells, both the sealed formal feature view and an independently rebuilt geometry-feature view pass only `1/3,700` cells and `0/5` complete trajectories. Their maximum per-cell metric-ratio difference is only `2.59e-11`, far below the frozen `1e-4` stability limit, so numerical feature-view drift does not explain the failure. The worst per-cell four-metric ratio is `1.93336`, with observation p90 ratios between roughly `1.579` and `1.861`; field and gradient tails also miss their gates.

The first full replay became inconclusive only while serializing a NumPy boolean into the final JSON report. Its partial arrays were not reused. A serialization-only successor repeated all 3,700 independent predictions and all 3,700 physical replays unchanged, and all `19/19` integrity checks passed. The preregistered decision therefore closes the current shared linear ridge branch: no target switch, lambda retuning, feature-view selection, or larger-model rescue is allowed. Fixed-teacher mechanism capacity remains established, but deployable prediction, matched-accuracy call reduction, resource speedup, external generalization, curved-ray validity, real BOST, and paper success remain unproven.
