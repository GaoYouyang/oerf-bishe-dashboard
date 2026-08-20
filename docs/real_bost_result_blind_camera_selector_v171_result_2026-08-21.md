# v171：结果不可见的几何选择器找回了五相机容量

更新：2026-08-21

## 先说结论

v170 已证明每套标定的九选五有限家族里存在合格子集，但它依靠已开封三维真值寻找见证。v171 回答下一步更严格的问题：**完全不看留出标定的重建结果，只读报告几何，能否选到同样安全的五相机子集？**

结果是肯定的。一个最多 `357` 参数的线性 ridge 排序器，在 13 个 leave-one-complete-calibration-out 外折中实现：

- 留出标定严格本地安全 `13/13`；
- 四个时间分层 `4/4` 全部通过；
- fit-static 对照只有 `2/13` 本地安全，v169 固定几何对照为 `0/13`；
- 没有便宜对照通过完整门。

正式与独立第二实现共同给出：

`PASS_RESULT_BLIND_GEOMETRY_SELECTOR_HEADROOM_V171`

这次确实改变了判断：v170 不再只是一条“存在但部署时找不到”的真值容量证据。当前小型几何选择机制在留出报告标定上能够结果不可见地找回该容量，而且不能被两个冻结便宜对照解释。

## 选择器究竟看了什么

每个五相机候选只由报告几何生成特征。程序在固定世界 DCT 坐标中构造 26 个非恒定低频模态的 H1-whitened Gram，取 trace 归一化后的上三角和五个谱标量，共 `356` 个特征。外折 fit 侧使用已经开封的候选风险拟合固定 `lambda=0.01` 的 ridge；留出侧预测包中没有任何候选真值行。

预测进程没有读取父指标，且相机顺序反转得到完全相同的特征与选择。这里证明的是受控 API 与封存数据流中的结果不可见，不宣称整个操作系统级文件访问都被物理隔离。

## 四个时间分层

| 归一化时间 | field p90 / worst | gradient p90 / worst | observation p90 / worst | 判决 |
| ---: | :--- | :--- | :--- | :--- |
| 0.00 | 0.330732 / 0.361056 | 0.612250 / 0.702318 | 0.130085 / 0.145829 | PASS |
| 0.25 | 0.337735 / 0.394694 | 0.623236 / 0.649852 | 0.119602 / 0.125096 | PASS |
| 0.75 | 0.323098 / 0.357847 | 0.630384 / 0.692196 | 0.117321 / 0.120210 | PASS |
| 1.00 | 0.322180 / 0.357996 | 0.617378 / 0.636703 | 0.119457 / 0.124125 | PASS |

最关键的 `t=0.75` gradient p90 从冻结 H1 的 `0.758639` 降到 `0.630384`，不再是擦线通过；gradient worst 为 `0.692196`。十三个留出标定的最坏已选本地风险为 `0.936425`，全部低于 1。

## 便宜对照没有解释结果

fit-static 对照从 fit 标定中选一个固定子集，再直接用于留出标定。它只在 `2/13` 个留出标定本地安全，四个时间的 gradient p90 为 `0.851393 / 0.852112 / 0.868619 / 0.875558`，因此 `0/4` 分层通过。

v169 的固定几何启发式在 `0/13` 个留出标定本地安全，四个 gradient p90 为 `0.895479 / 0.883457 / 0.895914 / 0.860270`，同样 `0/4`。因此这条正结果不是简单固定名单或旧低频可观测性指标的复述。

## 独立复算

独立程序重新构建全部几何特征、13 个外折拟合、结果不可见预测、三个策略选择、物理指标汇总和成本账。`21/21` 项检查全部通过。

特征、预测和策略报告的最大绝对差分别为 `2.33e-11 / 1.96e-11 / 8.04e-12`；fold 风险差和留出标签突变后的预测变化均为 `0`；全部策略选择逐项一致。

## 成本和严格边界

几何特征 cache 需要 `13×26=338` 个低模态 forward-equivalent 投影。cache 建成后，selector 拟合为 `0A+0A^T`，一个已选 H1 重建的逻辑在线账是 `1A+1A^T`。这只是调用账，没有 fresh-process wall 或 RSS 测量，不能写成真实速度或资源突破。

更重要的是，13 套标定的完整候选结果已经在 v170 开封。v171 的外折确实阻止留出标定结果进入拟合和预测，但它仍是 post-open development diagnostic，不是 fresh field/time、独立公开工况或真实实验泛化。它也只完成了相机子集选择，没有完成 observation/geometry-only warm initializer、exact lift 与 unchanged CGLS 的完整链。

所以当前准确结论是：**结果不可见的几何子集选择存在开发集机制 headroom。** 下一门应固定整场/时间隔离，再把通过的子集策略接回完整 warm-start 物理链。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v171: a result-blind geometry selector recovers five-camera capacity

Updated: 2026-08-21

v170 established that adequate five-of-nine subsets exist for every calibration, but it used opened 3D truth to find the witnesses. v171 asks the stricter follow-up: can a selector that never sees the held-out calibration outcomes and reads only reported geometry recover the same capacity?

The answer is positive. A linear ridge ranker with at most `357` parameters achieves `13/13` strict local-safe held-out calibrations under leave-one-complete-calibration-out evaluation, and all `4/4` global time strata pass. The fit-static control reaches only `2/13`, while the frozen v169 geometry heuristic reaches `0/13`; neither cheap control passes the complete gate.

Formal and independent implementations return:

`PASS_RESULT_BLIND_GEOMETRY_SELECTOR_HEADROOM_V171`

This changes the judgment. The v170 capacity is no longer merely an existence result with no result-blind way to find it. A small geometry-only mechanism recovers that capacity on held-out reported calibrations, and the result is not explained by either frozen cheap control.

Each candidate subset is represented only from reported geometry. The selector forms an H1-whitened Gram matrix for 26 nonconstant low-frequency modes in fixed world-DCT coordinates, then uses the trace-normalized upper triangle plus five spectral scalars, for `356` features. The ridge multiplier is frozen at `0.01`. Held-out packages contain no candidate target rows, the prediction process does not read parent metrics, and reversing camera order leaves the features and selections unchanged.

Field p90 values across the four times are `0.330732 / 0.337735 / 0.323098 / 0.322180`; gradient p90 values are `0.612250 / 0.623236 / 0.630384 / 0.617378`; observation p90 values are `0.130085 / 0.119602 / 0.117321 / 0.119457`. At `t=0.75`, gradient p90 improves from the frozen H1 value of `0.758639` to `0.630384`, with worst `0.692196`. The worst selected local risk over the thirteen held-out calibrations is `0.936425`, below the unit gate.

The fit-static control has gradient p90 values of `0.851393 / 0.852112 / 0.868619 / 0.875558`; the v169 control has `0.895479 / 0.883457 / 0.895914 / 0.860270`. Both fail all four global strata.

An independent second implementation rebuilds all geometry features, thirteen outer fits, result-blind predictions, three policy selections, physical metric summaries, and the cost ledger. All `21/21` checks pass. Maximum feature, prediction, and policy-report differences are `2.33e-11`, `1.96e-11`, and `8.04e-12`. Fold-risk difference and prediction change under held-out-label mutation are both zero, and every discrete selection agrees.

The geometry feature cache requires `338` forward-equivalent low-mode projections. After caching, selector fitting costs `0A+0A^T`, and one selected H1 reconstruction has a logical online ledger of `1A+1A^T`. No fresh-process wall or RSS experiment has been run, so this is not speed or resource evidence.

All thirteen calibration outcome families were already opened by v170. The fold barrier prevents held-out outcomes from entering v171 fitting and prediction, but this remains a post-open development diagnostic rather than fresh field/time or external generalization. It also selects cameras only; it does not yet establish the full observation/geometry-only warm initializer, exact lift, and unchanged-CGLS pipeline.

The precise conclusion is development-set mechanism headroom for result-blind geometry subset selection. The next gate must add whole-field/time separation and then integrate any passing subset policy into the complete physical warm-start chain.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
