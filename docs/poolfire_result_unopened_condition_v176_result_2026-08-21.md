# v176：一次结果未开工况否定当前最小共享选择器迁移

更新：2026-08-21

## 结论

v175 在已开封开发代理上通过了完整标定与完整三维场外折，但这只允许做一次真正的结果未开工况检验。v176 冻结 v175 的最终小模型、相机选择规则、`H1-K0` 重建、阈值和调用账，在一个此前没有读取结果的 PoolFire stopping-validation 工况上评估 `13` 套报告标定与 `4` 个冻结帧，共 `52` 个单元。没有重新拟合、调参或更换目标。

独立复算后的结果是：

- 最小共享选择器严格安全 `0/52`；
- 完整标定通过 `0/13`，帧分层通过 `0/4`；
- field / gradient / observation 全局 p90 为 `0.880095 / 0.994590 / 0.416498`；
- 相对同一相机子集的 Zero-CGLS K4，`52/52` 个单元出现联合伤害，其中 `50/52` 为严重伤害；
- fit-static、v169 low-mode D-opt、ray-axis maximin 三个冻结对照也都没有完整通过。

正式科学判决是：

`FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176`

这否定了当前最小共享选择器的跨工况迁移主张。它不是一次可以靠继续调 ridge、扩大网络或换目标来包装的“差一点通过”。

## 为什么问题不只是选错相机

更重要的机制诊断是：主策略同相机子集的 Zero-CGLS K4 参考也严格安全 `0/52`，而四条冻结策略各自的 K4 参考严格安全数全部为 `0`。主策略 K4 的 field / gradient / observation p90 为 `0.875520 / 0.786218 / 0.286067`，同样越过冻结绝对门。

因此，当前负结果不能只归因于“选择器把五台相机排错了”。这个工况暴露的是更广的五相机 reference / representation mismatch：在讨论学习选择器是否有优势之前，冻结的五相机重建参考本身就没有达到 inherited matched-accuracy 要求。

这不证明所有五相机方案数学上不可能，也不关闭整个 C 路线；它只关闭当前最小共享选择器和当前参考壳在跨工况上的成功主张。

## 独立复算

完全独立第二实现重新构造选择、物理场、同子集 K4 参考、二维残差、逐单元门、完整标定与帧尾部以及 exact-call 账。`35/35` 项检查全部通过：

- 逐单元指标最大绝对差 `2.07e-11`；
- 候选场 / K4 参考场最大相对差 `9.49e-11 / 3.94e-12`；
- 候选残差 / K4 参考残差最大相对差 `2.53e-11 / 8.89e-12`；
- 预测最大绝对差 `9.45e-12`；
- adjoint identity 最大相对误差 `2.13e-16`；
- exact-call 差为 `0`；
- 所有离散选择、通过/失败和最终判决完全一致。

## 执行异常与恢复边界

正式链保留了全部失败证据。前两次启动在读取科学输入前分别因模块导入和路径转义失败；第三次在固定选择和预测已经封存、科学输入打开后，因残差审计数组长度假设错误而中止。当时还没有生成逐单元指标、调用账、正式报告或 READY。

恢复只把变长残差保存为带长度回执的零填充审计缓冲，不改变数据、选择、预测、阈值、物理算子或判决。恢复后的正式预测变化为 `0`，独立复算预测差为 `9.45e-12`。因此这是一条透明披露的存储修复，不是看到结果后的算法修改。

## 路线动作与证据边界

当前最小共享选择器迁移路线关闭：不继续调参，不用更大 CNN/FNO/UNO/DeepONet 挽救，不租 GPU，不开启资源门，也不打开仍封存的测试集。若继续现有工况，只能另行结果前冻结一个 post-open 机制诊断，先回答五相机参考与表示为何不充分，不能把诊断重新包装成外部成功证据。

v176 只是一条受控 straight-ray 代理上的单工况结果未开负证据。它不是广泛外部泛化、真实 wall/RSS 加速、curved ray、真实 BOST、论文成功或算法突破：

`algorithm_breakthrough=false`、`paper_success=false`、`broad_external_generalization=false`、`resource_speedup=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v176: one result-unopened condition rejects the current minimal shared-selector transfer

Updated: 2026-08-21

v175 passed complete-calibration and complete-field outer isolation on the opened development proxy. v176 freezes the final v175 small model, camera-selection rule, H1-K0 reconstruction, thresholds, and call ledger, then evaluates one previously result-unopened PoolFire stopping-validation condition. The evaluation covers thirteen reported calibrations and four frozen frames, for `52` cells. No refitting, retuning, or target replacement is allowed.

After independent recomputation, the minimal shared selector is strict-safe on `0/52` cells, clears `0/13` complete calibrations and `0/4` frame strata, and reaches field / gradient / observation global p90 values of `0.880095 / 0.994590 / 0.416498`. All `52/52` cells are jointly harmed relative to their own same-subset Zero-CGLS K4 reference, with `50/52` severe. Fit-static, v169 low-mode D-opt, and ray-axis maximin also fail to pass completely.

Decision: `FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176`.

The failure is deeper than selector ranking alone. The primary same-subset K4 reference is also strict-safe on `0/52`, and the K4 references of all four frozen policies have strict-safe counts of zero. The primary K4 field / gradient / observation p90 values are `0.875520 / 0.786218 / 0.286067`, so the frozen absolute gate already fails before learned-selector advantage can be assessed. The condition therefore exposes a broader five-camera reference or representation mismatch. This does not prove that every five-camera approach is mathematically impossible and does not close the full C route.

A fully independent second implementation rebuilds selections, physical fields, same-subset K4 references, 2D residuals, cell gates, calibration and frame tails, and exact-call accounting. All `35/35` checks pass. Maximum per-cell metric difference is `2.07e-11`; candidate/reference field relative differences are `9.49e-11 / 3.94e-12`; candidate/reference residual relative differences are `2.53e-11 / 8.89e-12`; exact-call difference is zero; every discrete decision agrees.

The execution history is disclosed rather than hidden. Two launches failed before scientific input was opened. A third run opened the frozen condition after sealing selections and predictions, then failed because a residual-audit buffer assumed a fixed vector length. No cell metrics, call ledger, formal report, or READY existed at that point. The recovery changed only residual storage to a zero-padded buffer with explicit length receipts. Frozen selections were recovered exactly, formal predictions changed by zero, and the independent prediction difference is `9.45e-12`. This is a storage-only repair, not a post-result algorithm change.

The current minimal shared-selector transfer is closed. No parameter retuning, larger-model rescue, GPU rental, resource gate, or untouched-test opening is authorized. Any further work on this opened condition must be a separately frozen post-open mechanism diagnostic and cannot be repackaged as external success.

This is one result-unopened controlled straight-ray condition, not broad external generalization, a resource speedup, curved-ray validation, real BOST, paper success, or an algorithm breakthrough: `algorithm_breakthrough=false`, `paper_success=false`, `broad_external_generalization=false`, `resource_speedup=false`, `curved_ray_validated=false`, `real_bost=false`.
