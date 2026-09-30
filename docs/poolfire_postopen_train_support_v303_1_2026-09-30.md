# V303.1 Post-Open Train-Support Sensitivity

**Evidence status:** independently recomputed negative development evidence. The additional trajectory had already been opened; this is neither a confirmatory result nor fresh generalization.

## Question

Would adding one more official training trajectory repair the cross-trajectory support gap under the unchanged deployment-visible feature and leave-one-trajectory-out support contract?

## Result

The expanded roster contains 11 already-open development trajectories, 8,140 samples, and 67,155 active-camera rows. Overall nearest-neighbour support is **86.89%**. Support on the added trajectory is only **16.13%**, below the frozen 90% gate. The frozen camera-stratum and factor/severity-stratum gates also fail.

The formal and independent implementations agree on discrete support decisions and numeric summaries within the preregistered tolerance. All 25 integrity and recomputation checks pass; the failed checks are the two scientific coverage gates, not integrity failures.

Candidate state construction used 740 forward and 740 adjoint applications offline. The support audit itself added **0 forward and 0 adjoint calls**. This is not an online-call reduction or speed result.

## Decision and limits

**FAIL_V303_1_POSTOPEN_TRAIN_ROSTER_SUPPORT_SENSITIVITY.** One additional already-open training trajectory did not repair cross-trajectory support. This closes only this roster-extension attempt, not the entire C route. It does not authorize predictor fitting and does not establish matched reconstruction accuracy, physical replay, external generalization, resource savings, or real BOST performance. The opened trajectory must not be described as held out.

Algorithmic breakthrough, paper success, external generalization, resource speedup, curved-ray validation, and real-BOST success remain unestablished.

## 中文摘要

**证据状态：**独立复算的负向开发证据。新增轨迹此前已经开封，因此不是确认性结果，也不是新的泛化结果。

**问题：**在不改变部署可见特征、留一完整轨迹支持合同和 90% 门槛的情况下，再加入一条官方训练轨迹，能否弥补跨轨迹支持缺口？

**结果：**扩展后共有 11 条已开封开发轨迹、8,140 个样本和 67,155 条有效相机观测行。整体近邻支持率为 **86.89%**，新增轨迹仅为 **16.13%**，低于冻结的 90% 门槛；相机分层与 factor/severity 分层门也未通过。正式与独立实现的离散判决一致，数值汇总在预注册容差内。25 项完整性与复算检查全部通过；未通过的是两项科学覆盖门，不是数据完整性或独立复算失败。

候选状态离线构造耗费 740 次 forward 和 740 次 adjoint；支持评分新增 **0 次 forward 和 0 次 adjoint**。这不是在线算子调用减少或速度结果。

**判决：FAIL_V303_1_POSTOPEN_TRAIN_ROSTER_SUPPORT_SENSITIVITY。**多加一条已开封训练轨迹没有修复跨轨迹支持缺口；这只关闭本次工况扩展尝试，不关闭整个 C 路线。它不授权预测器训练，也不证明同精度重建、物理 replay、外部泛化、资源收益或真实 BOST 性能。新增轨迹此前已开封，不得称为留出确认。

算法突破、论文成功、外部泛化、资源加速、弯曲光线验证和真实 BOST 成功均未建立。
