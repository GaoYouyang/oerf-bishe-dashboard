# v229：嵌套双 PRESS 校准在已开封数据上通过严格留一门

## 结论

v228 发现原始 block-PRESS 与几何白化 PRESS 的固定 OR 在已开封 Case 2/5 上有互补信号，但那是看过两个父结果失败位置后的事后诊断。v229 在读取连续分数前固定一个嵌套校准规则：

- 每个 Case 5 目标 rig 的阈值和组合倍率只由其他 `12` 个 rig 生成；
- Case 2 的校准只读 Case 5，Case 2 的分数、真值和 rig 身份都不进入校准；
- 两种分数分别除以 fold-local 父阈值，取两者较小的归一化分数，再用内层留一 rig 的固定 `10%` 下尾顺序统计决定接受线。

正式与完全独立的第二实现一致得到：

`POST_OPEN_FOLD_LOCAL_DUAL_PRESS_CALIBRATION_HEADROOM_V229`

它在 Case 5 接受 `136/546` 个安全单元，危险误接为 `0`，最差 rig 为 `5/42=11.90%`；Case 2 接受 `318/715` 个安全单元，危险误接仍为 `0`，最差 rig 为 `19/55=34.55%`。两条工况均保持 `13/13` 完整 rig 精度通过，且每个 rig 的平均逻辑 `A/A^T` 账均低于 Zero-PCGLS K16。

## 为什么这比 v228 更强

v228 直接对两个已封存的离散判决做 OR。v229 不使用目标 rig 的分数来调自己的阈值，也不使用留出真值、失败身份、物理指标或 Case 2 统计来调校准。它因此排除了“必须直接读目标 rig 才能补回效用”的解释。

但这还不是外部验证：嵌套公式是在 v228 的事后互补线索出现后选定，Case 2/5 也已经开封。所以它是一个更严格的 **post-open development mechanism**，只授权另行冻结一个未开封工况外门。

## 结果

| 条件与规则 | 接受单元 | 最低 rig 接受率 | 危险误接 | 完整 rig 精度 | 判决 |
|---|---:|---:|---:|---:|---|
| Case 5，原始 v226 | `126/546` | `4/42=9.52%` | `0` | `13/13` | 效用失败 |
| Case 5，白化 v227 | `123/546` | `4/42=9.52%` | `0` | `13/13` | 效用失败 |
| Case 5，v228 事后 OR | `140/546` | `5/42=11.90%` | `0` | `13/13` | 事后 control 通过 |
| Case 5，v229 嵌套校准 | `136/546` | `5/42=11.90%` | `0` | `13/13` | 开封开发门通过 |
| Case 2，v229 嵌套校准 | `318/715` | `19/55=34.55%` | `0` | `13/13` | 开封开发门通过 |

v229 没有追求比 v228 OR 接受更多；它丢弃了 Case 5 的 `4` 个与 Case 2 的 `6` 个接受，换来了目标 rig 分数不参与自身校准的固定程序。Case 5 各 rig 接受数为 `7, 8, 16, 12, 5, 6, 22, 10, 5, 24, 9, 6, 6`；Case 2 为 `24, 23, 28, 26, 19, 22, 26, 24, 21, 30, 24, 24, 27`。

最大 matched ratio 在 Case 5/2 为 `1.007896 / 1.027761`。Direct K11 路径逻辑账为 `12A+11A^T`，回退 Zero-PCGLS K16 为 `16A+16A^T`。这仍只是已开封开发数据上的逻辑账，不能写成已建立的 exact-call、wall 或 RSS 收益。

## 独立复算

正式实现读取两套封存的正式连续分数。独立实现只读两套分别封存的独立分数，用 Python 稳定排序与显式内外层 rig 循环重建全部阈值、倍率、决策、物理策略和调用账，不导入正式 v229 帮助函数。

必需的 `17/17` 项检查全部通过，离散决策完全一致；父分数、校准数组和汇总的最大绝对差分别为 `1.11e-15 / 8.88e-16 / 2.57e-11`。独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_NESTED_DUAL_PRESS_CALIBRATION_V229`

底层冻结 physics kernels 仍由父证据共享，因此 `end_to_end_physics_independence_proven=false`。

## 证据边界

- v229 是在已开封 Case 2/5 上进行的 post-open development 实验，不是 fresh/external gate；
- 它证明的是两种可观测分数可以通过目标 rig 无泄漏的固定校准达到开封开发门，不证明它在未见工况泛化；
- 不调 `10%` 比例、顺序统计、归一化、组合公式、阈值 floor、rank 或 solver depth；
- 不授权训练大模型、租 GPU、运行 fresh wall/RSS 或宣称论文成功；
- 下一步只能另行冻结一个未开封工况外门，失败就关闭这条组合校准路线。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v229: Nested dual-PRESS calibration passes the strict leave-rig-out gate on opened data

## Conclusion

v228 found that fixed OR between raw block-PRESS and geometry-studentized PRESS carries complementary signal on opened Cases 2 and 5, but that was retrospective. Before reading the continuous score arrays, v229 fixes one nested calibration rule. Each target Case 5 rig is calibrated only from the other 12 rigs; Case 2 is calibrated only from all Case 5 rigs and never enters calibration. Each score is normalized by its fold-local parent threshold, their pointwise minimum forms an envelope, and a fixed inner leave-rig-out 10-percent lower-tail statistic sets the acceptance multiplier.

The formal and fully separate implementations agree on `POST_OPEN_FOLD_LOCAL_DUAL_PRESS_CALIBRATION_HEADROOM_V229`.

The rule accepts `136/546` safe Case 5 cells with zero unsafe accepts and a worst rig of `5/42=11.90%`. It accepts `318/715` safe Case 2 cells with zero unsafe accepts and a worst rig of `19/55=34.55%`. Both conditions retain `13/13` complete-rig accuracy, and every rig's mean logical call ledger remains below Zero-PCGLS K16.

This is stronger than v228 in one narrow sense: the target rig's scores do not enter its own calibration, and no Case 2 score, truth, failure identity, or physical metric tunes the rule. It is still not external validation. The formula was chosen after the v228 retrospective lead, and both conditions were already opened. The result therefore remains post-open development headroom and only authorizes a separately frozen unopened-condition gate.

## Results and independent recomputation

The single-parent controls remain below the frozen Case 5 utility gate at `126/546` and `123/546`. Retrospective fixed OR accepts `140/546`, while nested calibration accepts `136/546`. The latter gives up four opened Case 5 accepts and six Case 2 accepts to enforce target-rig score isolation. Maximum matched ratios are `1.007896` and `1.027761` for Cases 5 and 2.

The independent implementation reads separately sealed parent score arrays and rebuilds all order statistics, thresholds, multipliers, decisions, physical-policy replays, and call ledgers with explicit Python loops, without importing the formal v229 helper. All `17/17` required checks pass and discrete decisions match exactly. Maximum parent-score, calibration-array, and summary differences are `1.11e-15 / 8.88e-16 / 2.57e-11`. Shared frozen physics kernels remain, so end-to-end physics independence is not proven.

## Evidence boundary

v229 is post-open development evidence on opened Cases 2 and 5, not a fresh or external gate. It establishes no deployment algorithm, stable exact-call benefit, wall/RSS speedup, external generalization, curved-ray validation, or real-BOST result. No calibration fraction, statistic, formula, floor, rank, or solver-depth retuning is allowed. No large model, GPU rental, or resource run is authorized. The next valid step is one separately frozen unopened-condition gate; failure closes this combination-calibration route.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
