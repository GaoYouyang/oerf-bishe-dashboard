# v284 Common-Stop Independent Audit (2026-09-29)

## 中文

### 问题与范围

在同一套冻结的 AMG-PCGLS 迭代与共同停止证书下，已经训练并封存的学习初值能否在满足相同最终精度时，稳定地同时减少精确前向算子 `A` 和伴随算子 `A^T` 调用？本审计只覆盖五条已经开封的 PoolFire 轨迹、505 帧，以及既有封存预测；它不是新训练、盲测、未开封外部门或真实 BOST 验证。

### 独立复算与结果

独立验证对物理投影、残差、场/全梯度/内部梯度/观测四项指标、停止证书和精确调用账的 8 项检查全部通过。505/505 帧均满足冻结的四项精度门。

但严格同时少用 `A` 与 `A^T` 的帧数取决于对照：相对 Zero-AMG 为 490/505，相对归一化 BP 为 485/505，相对 dual-ridge 为 464/505。逐轨迹地要求候选在全部帧上严格胜过全部三个对照时，只有 1/5 条轨迹通过。最弱折的逐对严格胜数为 88/101、85/101、73/101；其余四折也没有任何额外整轨迹通过。

因此，冻结判决为 **FAIL_LEARNED_INITIALIZER_NO_ROBUST_CALL_ADVANTAGE**：精度兼容得到支持，但当前学习初值没有证明稳定、整轨迹的双算子调用优势。这不推翻 v284 的其他分项结果，也不是对所有学习初值的普遍否定。

### 证据边界与后续

该审计使用已开封数据；学习训练、推理及几何准备资源没有纳入这项调用判决，也没有本审计专属的 fresh wall-time 或 RSS 测量。它不证明端到端加速、外部泛化、真实实验 BOST 或算法突破。当前固定学习方案不应通过追加训练、改阈值或挑选累计总量来挽救。后续应保留该负结果和成本边界；只有新的物理信息或预先冻结的不同机制才构成继续依据。

## English

### Question and scope

Under the same frozen AMG-PCGLS recurrence and common stopping certificate, does the already-trained and sealed learned initializer robustly reduce both exact forward (`A`) and adjoint (`A^T`) calls at matched final accuracy? This audit covers only five already-opened PoolFire trajectories, 505 frames, and the existing sealed predictions. It is not new training, a blind test, an unopened external gate, or experimental BOST validation.

### Independent replay and result

All eight independent checks passed for physical projections, residuals, field/full-gradient/interior-gradient/observation metrics, stopping certificate, and exact-call ledger. All 505/505 frames satisfy the frozen four-metric accuracy gate.

Strict paired savings in both `A` and `A^T` depend on the comparator: 490/505 frames versus Zero-AMG, 485/505 versus normalized BP, and 464/505 versus dual ridge. Requiring strict wins over all three controls on every frame of a complete trajectory, only 1/5 trajectories passes. The weakest fold has 88/101, 85/101, and 73/101 strict wins against the three controls; none of the other four folds adds a complete-trajectory pass.

The frozen verdict is **FAIL_LEARNED_INITIALIZER_NO_ROBUST_CALL_ADVANTAGE**: accuracy compatibility is supported, but this learned initializer has not established a stable, complete-trajectory reduction in both operator calls. This does not overturn other component results from v284 and is not a universal rejection of learned initializers.

### Evidence boundary and next step

This audit uses opened data. Learned fitting, inference, and geometry-preparation resources are not included in this call-count verdict, and this audit has no dedicated fresh wall-time or RSS measurement. It establishes neither end-to-end acceleration, external generalization, real experimental BOST, nor an algorithmic breakthrough. The fixed learned recipe should not be rescued by more fitting, threshold changes, or selecting aggregate totals. Preserve this negative result and its cost boundary; any continuation requires new physical information or a genuinely different mechanism frozen in advance.
