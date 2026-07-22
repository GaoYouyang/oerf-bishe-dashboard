# GCT-KMix 零初值支配上界 NO-GO 协议

## 为什么可以绕过 tail-safe solver 的退化

O1 的逐 ray tail-safe 可行集记为 `F_tail`，只要求单纯形的乐观可行集记为
`F_simplex`。显然：

`F_tail subset F_simplex`。

对每个 case，同一个 field relative-L2 目标满足：

`min_{w in F_simplex} error(w) <= min_{w in F_tail} error(w)`。

因此 unconstrained simplex truth oracle 相对同一 `GCT-KSelect` 的 field gain，
是 tail-safe oracle 能达到的**乐观上界**。若这个超集上界的 split mean 都低于
结果前冻结的 5% field 门，任何 tail-safe 权重都不可能通过该门。由于总 gate
是合取，field 门失败已经足以停止 learner；不需要把 `AlmostSolved` 的 tail-safe
权重当成最优，也不需要 H1 先失败。

## 证书必须检查

1. attempt 包 checksums 完整且逐文件匹配；
2. development/OOD 各 9 个 case，unconstrained 与 KSelect case 一一对应；
3. 所有 unconstrained oracle 为 `Solved`；
4. gap、primal residual、dual residual 都在原冻结 `1e-8` 门内；
5. solver primal objective 与 `0.5 * field_relative_l2^2` 一致到 `1e-12`；
6. 每 case unconstrained error 不大于作为其可行点的 KSelect error；
7. 分别计算 per-case gain 后取 split mean，不用 aggregate-error 比率偷换；
8. development 与 public exploratory OOD 的乐观 mean gain 上界都严格低于 5%。

任何一项失败，证书 fail closed。证书不修改原 attempt 的 solver-failure 状态。

## 允许的结论

若证书通过，只能把它标成
`ZERO_START_FIELD_HEADROOM_UPPER_BOUND_NO_GO_LEARNER_NOT_AUTHORIZED`，并写：

> 在当前 16³ JACRU、零初值 CGLS checkpoint library 和预注册 field 门下，即使
> 移除全部 ray 安全约束，最优 truth-conditioned 凸组合的平均 headroom 仍不足；
> 所以 tail-safe zero-start GCT-KMix 不可能过门，不应训练权重网络。

不能写算法失败于真实 BOST、算子学习无效、优于/劣于 FNO/DeepONet/NeRIF、
跨 rig 泛化结论或论文突破。下一步必须改变信息来源，而不是继续调零初值权重：
优先测试固定 learned warm start 是否保留 M2.2 所示 null-space 成分，并保持相同
逐 ray/camera 安全门；没有 learned/temporal/第二诊断信息时，row-space mixing
不能恢复不可观成分。
