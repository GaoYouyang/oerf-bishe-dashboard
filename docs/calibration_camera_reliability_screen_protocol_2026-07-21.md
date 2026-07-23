# 三维 BOST 相机残差一致性权重筛选协议

> 状态：`POSTOPEN_DEVELOPMENT_PROTOCOL`
>
> 输入：冻结的 `calibration_mismatch_v2` camera-fold ledger
>
> 研究问题：相机间残差一致性是否能提供有界、真值盲的可靠性权重，从而提高标定修正的 heuristic fallback 检出功效？

## 1. 证据时间线

这个协议是在查看 `calibration_mismatch_v2` 的结果后写成的。探索阶段已经看见：

- uniform camera-block LCB 在零失配时会全部回退；
- 1 档与 2 档存在收益，但 0.5 档功效不足；
- camera block 2 在多个 synthetic rig 上经常与其余相机给出相反的残差改善方向；
- target-local range、slope、curvature 等权重值得比较。

因此，本轮只能叫 **post-open development screen**。leave-one-rig-out 只能减少直接目标泄漏，不能把已经打开的数据重新变成 fresh test，也不能证明跨实验装置泛化。

## 2. 冻结输入

只读使用以下 v2 产物：

- `learning_labs/results/calibration_mismatch_v2/report.json`
- `learning_labs/results/calibration_mismatch_v2/case_metrics.csv`
- `learning_labs/results/calibration_mismatch_v2/candidate_curves.csv`
- `learning_labs/results/calibration_mismatch_v2/camera_fold_scores.csv`
- `learning_labs/results/calibration_mismatch_v2/checksums.sha256`

deployment 阶段只验证并打开 `camera_fold_scores.csv`，从 CSV 自身推导 rig、family、档位和候选，不解析或携带完整上游报告。所有决策冻结后，evaluator 阶段才核对整包校验和并打开 `report.json`、`candidate_curves.csv` 与 field 指标；任何不一致都 fail closed。不得修改或覆盖 v2。

## 3. 统计单位与分组

- 外层泛化单元：`base_seed` 对应的 synthetic rig，共 6 个。
- 每个 rig 内有 3 个解析 morphology proxy。它们共享几何，只用于降低场形态偶然性，**不是 3 个独立 rig/session**。
- 每个 rig 有 6 个 camera block。六个 LOCO fold 的训练集高度重叠，camera block 既不是独立实验重复，也不能按 IID 解释普通标准误。
- 失配档位：`0 / 0.5 / 1 / 2`。
- 候选修正：`-0.5 / 0 / 0.5 / 1 / 1.5 / 2 / 2.5`。

## 4. 主候选：LORO consensus-correlation weight

对每个目标 rig `r*`：

1. 只用其余 5 个 rig 拟合权重。
2. 对训练 rig 的每个失配档和非零候选，计算每个 camera block 相对 `correction=0` 的 LOCO score 改善。
3. camera `c` 的参考信号是同一条记录中其余 5 台相机改善的中位数。
4. 可靠性为 camera 改善与参考信号的 Pearson correlation，负相关截断为 0。
5. 可靠性除以六台相机可靠性的中位数，随后固定截断到 `[0.5, 2]`，最后归一化为和为 1 的非负权重；归一化后的 `max/min` 不超过 4。

这个模型只有 6 个闭式标量，没有神经网络、梯度训练或 GPU。它是未来 bounded weight network 必须击败的便宜基线。

## 5. 目标 rig 上允许读取的信息

目标 rig 选择阶段只读取：

- noisy held-out-camera LOCO score；
- candidate correction 标签；
- camera block 标识；
- 从其余 5 个 rig 冻结得到的权重。

选择器不得读取：

- 目标体真值；
- clean observation；
- target field relative-L2；
- exact geometry teacher；
- truth-selected candidate；
- 目标 rig 的 evaluator gain。

评分器在决策完成后才附加 field relative-L2。

## 6. 加权 heuristic fallback 规则

这里不能叫正式 fail-closed 安全门。原因有两个：六个 LOCO fold 相互相关；同一批 camera score 又同时参与候选准入和候选排序，存在选择后推断。下式只用于与 v2 保持一致的描述性 replay，不产生 t 分布 coverage、显著性或单 session 安全证明。

对目标 rig、某一失配档和候选 `j`，先在三个 morphology proxy 内平均，得到 6 个 camera-block 改善 `d_c`。设归一化权重为 `p_c`：

```text
mu_w  = sum_c p_c d_c
n_eff = 1 / sum_c p_c^2
var_w = sum_c p_c (d_c - mu_w)^2 / (1 - sum_c p_c^2)
LCB_w = mu_w - 2.015 * sqrt(var_w / n_eff)
```

只有 heuristic `LCB_w > 0` 的非零候选进入候选集合；从中选择加权 LOCO score 最低者，若集合为空则回退 `correction=0`。

`2.015` 沿用 v2 的 one-sided t5-style heuristic。camera folds 相关、权重数据依赖且存在多候选选择，因此不得称为置信证书、统计显著性或安全保证。

## 7. 相同在线物理预算比较对象

所有策略使用相同的 3024 条唯一 camera-fold score、候选集合，且新增 forward、adjoint 和重建调用都为 0；但它们的离线聚合成本并不相同。uniform 在线选择读取 3024 个 score value；主 LORO 在六折中额外读取 15120 个训练 score value，再读取 3024 个目标 score value，总计 18144。因而只能称为相同在线物理调用/重建预算，不能称为端到端同计算预算。当前未测 wall time 与 peak memory，报告必须把这两项写成未测。

1. `uniform_lcb`：复算 v2 的 multiframe camera-block LCB；
2. `target_local_range_lcb`：按相机 score range 加权；
3. `target_local_slope_lcb`：按 `score(+0.5)-score(-0.5)` 绝对值加权；
4. `target_local_curvature_lcb`：按 nominal 附近二阶差分绝对值加权；
5. `target_local_inverse_nominal_lcb`：按 nominal LOCO score 的倒数加权；
6. `robust_median_uniform_gate`：中位数用于候选排序，uniform LCB 用于准入；
7. `loro_sign_agreement_lcb`：其余 5 个 rig 的相机符号一致率权重；
8. `loro_consensus_correlation_lcb`：本轮主候选。

target-local 权重会同时读取目标 rig 的候选 score surface，必须单列，不能与 LORO 候选混称跨 rig 学习。

## 8. 主指标与冻结门

每个失配档报告：

- 18 个 field case 的 mean / median / worst relative-L2 gain；
- field improvement 与 non-worse fraction；
- 6 个 rig 的 mean / worst gain；
- fallback fraction；
- selected correction；
- 有效相机数 `n_eff`；
- 相对 `uniform_lcb` 的 paired gain difference。

沿用 v2 机制门：

- 所有非零档 mean field gain 至少 `5%`；
- 所有非零档 field improvement fraction 至少 `75%`；
- 所有非零档 worst field gain 不低于 `-5%`；
- 零失配 mean field gain 不低于 `-5%`。

主候选若未全部满足，状态必须是 `NO-GO`。不得为了 0.5 档启动率调低 LCB multiplier 或事后移动门槛。

## 9. 必须输出的负面对照

- 若简单 target-local 权重与 uniform 决策等价，要明确写出 gate dominates weighting。
- 若某权重提高平均收益但伤害单个 field，要同时报告 worst/non-worse。
- 若 LORO 权重主要因为固定 camera identity 起效，要明确说明它依赖相同相机角色对齐，不能自动迁移到任意 rig。
- 若主候选只在 1/2 档提高功效而 0.5 档失败，不能把它写成通用标定恢复。

## 10. 允许与禁止的结论

允许：

- 在已打开的 synthetic v2 ledger 上，某种有界相机一致性权重改变了 heuristic gate 的功效；
- 该权重可作为未来神经权重模型必须击败的便宜基线；
- 结果生成一个需要 fresh rig 或实验室数据验证的假设。

禁止：

- 新算法已经成功；
- 已证明真实 BOST 标定更准；
- 已证明跨 rig、跨几何或跨流态泛化；
- 已优于 NeRIF、TDBOST、DeepONet 或 FNO；
- leave-one-rig-out 等于独立外部验证；
- weighted LCB 有正式置信覆盖。

## 11. “可靠性”与“可观测性”必须分开

- 留出相机残差只能支持 prediction consistency / measurement reliability。
- 真正的局部标定可观测性需要残差对几何参数的 Jacobian、尺度归一化 `J^T J` 谱、近零特征方向和参数耦合。
- 一维扰动曲线只说明已知方向的敏感性，不能发现焦距、位姿和平移的耦合零空间。
- 高几何敏感性既可能便于校准，也意味着误差传播更严重；不能未经验证就赋予更高重建权重。

因此本轮产物统一称为 **camera residual-consistency reliability weight**，不称 observability weight。未来若拿到 geometry JVP/VJP，再单独构造 `q_rel`、`q_cal` 与视角边际信息 `q_field`，而不是把三者混成一个漂亮但含义不清的分数。

## 12. 下一道外部门

只有拿到何远哲师兄确认的真实 callable、几何标定、straight/curved residual 层级、JVP/VJP 或 `A/A^T`、主要失败模式、强基线和数据 split 后，才能判断相机权重是否对应实验室真实痛点。正式验证还需要独立 select/sentinel camera-frame 角色、连续且不与候选网格对齐的未知误差、至少 30 个 sealed audit rig/session；否则本轮停在 synthetic post-open 机制证据。
