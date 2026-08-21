# v177：穷举排除“只是选错相机”，瓶颈落在低深度场参考

更新：2026-08-21

## 结论

v176 已经证明冻结选择器在一次结果未开的 PoolFire 工况上失败，但当时仍有两个解释：它可能只是挑错了五台相机，也可能是当前五相机 K4 参考本身不够。v177 在同一个已经开封的工况上做 post-open 容量诊断，不再训练或调整选择器。

对 `13` 套标定、`4` 个冻结帧和每套标定全部 `126` 个五相机子集，v177 分别运行未修改 Zero-CGLS K4 与 K8；另以九相机 Zero-CGLS K4 作传感器数量对照。每档五相机深度共 `6,552` 个候选。

独立复算后的结果是：

- 五相机 K4：严格安全候选 `0/6552`，逐单元容量 `0/52`，标定共享容量 `0/13`；
- 五相机 K8：严格安全候选仍为 `0/6552`，逐单元容量 `0/52`，标定共享容量 `0/13`；
- 九相机 K4：严格安全单元 `0/52`，完整标定 `0/13`，帧分层 `0/4`；
- 因此没有任何一个标定-帧单元能靠“换一个五相机子集”通过完整 field / gradient / observation 联合门。

正式科学判决是：

`FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177`

## K8 提供了最关键的归因

这个负结果并不是“算得更深也完全没变化”。按每个单元分别看三个指标的真值可见上界：

- 五相机 K4 的 field / gradient / observation 可通过数为 `0/52 · 45/52 · 1/52`；
- 五相机 K8 变为 `0/52 · 52/52 · 52/52`；
- 九相机 K4 为 `0/52 · 52/52 · 0/52`。

也就是说，K8 已经把五相机的梯度和观测分别救到全部单元，但 field 仍然一个都过不了。五相机 K8 的 cellwise witness p90 为 `0.768040 / 0.778511 / 0.134546`；其中 field 门是 `0.50`。即使在全部候选里分别寻找最小 field，最小值仍为 `0.578590`，没有跨过 field 门。

所以现在可以有根据地排除“v176 只是选择器挑错相机”。更精确的定位是：当前低深度 Zero-CGLS 参考壳的三维场表示不充分。增加 Krylov 深度能改善观测与梯度，却没有解决 field reference adequacy；增加到九台相机的 K4 对照也没有得到严格安全单元。

这不是对所有 CGLS 深度或所有场表示的数学不可能性证明，也不关闭整个 C 路线。

## 独立复算

完全独立第二实现重新构造了全部五相机子集、K4/K8 状态、九相机 K4 对照、二维观测、逐候选和逐单元指标、标定共享判决、调用账以及相机乱序审计。`25/25` 项检查全部通过：

- 五相机逐指标最大差 `1.29e-12`；
- 九相机逐指标最大差 `6.72e-13`；
- 容量汇总最大差 `5.00e-15`；
- adjoint identity 最大相对误差 `2.13e-16`；
- 相机乱序最大相对差 `2.86e-16`；
- 所有离散通过、失败和最终判决完全一致。

## 路线动作与证据边界

当前低深度 Zero-CGLS reference shell 关闭，不再围绕它训练另一个 selector 或 predictor，也不事后调 K、放宽阈值、扩大网络、租 GPU、跑资源门或打开封存 test。若继续现有虚拟代理，下一机制必须结果前冻结，并在物理上改变 field reference 或 representation；另一条有效输入是工况匹配的真实二维双分量 BOS 位移。

v177 是已开封工况上的 post-open 机制诊断，不是部署算法、外部泛化、真实资源收益、curved ray、真实 BOST、论文成功或算法突破：

`algorithm_breakthrough=false`、`paper_success=false`、`broad_external_generalization=false`、`resource_speedup=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v177: exhaustive capacity rules out “the selector merely chose the wrong cameras”

Updated: 2026-08-21

v176 shows that the frozen selector fails on one previously result-unopened PoolFire condition, but leaves two explanations: the selector may have chosen the wrong five cameras, or the five-camera K4 reference itself may be inadequate. v177 is a post-open capacity diagnostic on that same opened condition; it trains or retunes no selector.

Across thirteen calibrations, four frozen frames, and all 126 five-of-nine subsets per calibration, v177 runs unchanged Zero-CGLS K4 and K8. A nine-camera Zero-CGLS K4 control tests whether sensor count alone repairs the reference. Each five-camera depth therefore contains `6,552` candidates.

After independent recomputation, five-camera K4 has `0/6552` jointly strict-safe candidates, `0/52` cellwise capacity, and `0/13` calibration-shared capacity. Five-camera K8 remains `0/6552`, `0/52`, and `0/13`. Nine-camera K4 is strict-safe on `0/52`, with `0/13` complete calibrations and `0/4` frame strata. No calibration-frame cell can pass the complete field / gradient / observation gate simply by choosing another five-camera subset.

Decision: `FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177`.

K8 gives the decisive attribution. The per-metric truth-aware upper bounds for five-camera K4 are `0/52 · 45/52 · 1/52` for field / gradient / observation. At K8 they become `0/52 · 52/52 · 52/52`; the nine-camera K4 control is `0/52 · 52/52 · 0/52`. K8 therefore makes gradient and observation individually feasible on every cell, while field remains infeasible on every cell. The K8 cellwise-witness p90 values are `0.768040 / 0.778511 / 0.134546`, with a frozen field gate of `0.50`. Even the minimum field error over every candidate is `0.578590`.

The evidence rules out “v176 merely chose the wrong cameras.” The active bottleneck is the adequacy of the low-depth Zero-CGLS field reference or representation. Increasing Krylov depth improves observation and gradient but does not repair the field; adding all nine cameras at K4 also produces no strict-safe cell. This is not a mathematical impossibility result for every CGLS depth or every field representation, and it does not close the full C route.

A fully independent second implementation rebuilds all five-camera subsets, K4/K8 states, the nine-camera K4 control, 2D observations, candidate and cell metrics, calibration-shared decisions, call ledgers, and camera-order audits. All `25/25` checks pass. Maximum five-camera and nine-camera metric differences are `1.29e-12` and `6.72e-13`; the capacity-summary difference is `5.00e-15`; adjoint and camera-permutation relative errors are `2.13e-16` and `2.86e-16`; every discrete decision agrees.

The current low-depth Zero-CGLS reference shell is closed as the basis for another selector or predictor. No post-result K tuning, gate relaxation, larger-model rescue, GPU rental, resource gate, or untouched-test opening is authorized. Any continuation on the virtual proxy must separately preregister a physically different field reference or representation; condition-matched experimental two-component BOS displacement is the other valid new input.

v177 is a post-open mechanism diagnostic, not a deployable algorithm, external generalization, a resource speedup, curved-ray validation, real BOST, paper success, or an algorithm breakthrough: `algorithm_breakthrough=false`, `paper_success=false`, `broad_external_generalization=false`, `resource_speedup=false`, `curved_ray_validated=false`, `real_bost=false`.
