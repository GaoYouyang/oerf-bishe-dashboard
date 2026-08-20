# v172：几何选择器通过整场与时间三重隔离

更新：2026-08-21

## 先说结论

v171 已经证明，只读报告相机几何的小型选择器可以在留出整套标定时找到安全的五相机子集。但它的 fit 标签仍汇总了全部九个已开封三维场和四个时间点，因此还不能排除选择器只是适配了这组场与时间。

v172 把隔离提高到三条轴同时成立。每个预测都留出：

- 一整套报告标定；
- 一个完整三维场模型；
- 一个完整时间点。

拟合侧只剩 `12` 套标定、`8` 个场和 `3` 个时间。总计 `13×9×4=468` 个外折预测。留出的三类结果都不能进入风险目标、特征标准化、ridge、静态对照、tie-break 或回退。

独立复算后的正式判决是：

`PASS_WHOLE_FIELD_TIME_ISOLATED_GEOMETRY_SELECTOR_HEADROOM_V172`

主策略实现：

- 严格安全单元 `468/468`；
- 完整标定安全 `13/13`；
- 完整三维场安全 `9/9`；
- 时间分层 `4/4`；
- 最坏严格风险 `0.975390 < 1`。

fit-static 与 v169 固定几何对照分别只有 `323/468` 与 `192/468` 单元安全，完整三维场和时间层都是 `0` 个通过。因此这个结果不能由两个便宜选择规则解释。

## 三重外折究竟隔离了什么

对一个留出三元组 `(标定, 场, 时间)`，fit 目标先删除该整套标定的全部候选结果，再删除该场在其余标定上的全部结果，还删除该时间在其余场和标定上的全部结果。之后才在剩余 `12×8×3` 的结果上，为每个五相机候选形成冻结的最坏归一化风险。

预测器仍是 v171 的最多 `357` 参数 Gram-ridge，只读 `356` 维报告几何特征。它不读观测、密度、模型编号、时间编号或留出指标。一次批量拟合只是在 CPU 上同时求解同一几何设计矩阵的多个固定 ridge 目标，没有超参数搜索。

## 四个时间分层

| 归一化时间 | field p90 / worst | gradient p90 / worst | observation p90 / worst | 判决 |
| ---: | :--- | :--- | :--- | :--- |
| 0.00 | 0.330732 / 0.410876 | 0.613132 / 0.702318 | 0.130085 / 0.145829 | PASS |
| 0.25 | 0.341935 / 0.394694 | 0.623236 / 0.644840 | 0.119602 / 0.125096 | PASS |
| 0.75 | 0.323257 / 0.385484 | 0.632018 / 0.731543 | 0.117337 / 0.123690 | PASS |
| 1.00 | 0.322180 / 0.357996 | 0.621204 / 0.636703 | 0.119413 / 0.124125 | PASS |

最紧的 `t=0.75` gradient p90 为 `0.632018`，低于冻结门 `0.750000`，worst 为 `0.731543`。四个时间都不是擦线失败。

## 便宜对照

fit-static 在每个外折中只从 fit 侧选择一个固定子集。它的四个 gradient p90 为 `0.820956 / 0.852112 / 0.868619 / 0.875558`，只安全 `323/468` 个单元，完整三维场 `0/9`、时间层 `0/4`。

v169 固定几何启发式的四个 gradient p90 为 `0.895479 / 0.883457 / 0.895914 / 0.860270`，只安全 `192/468` 个单元，完整标定 `0/13`、完整三维场 `0/9`、时间层 `0/4`。

## 独立复算

独立程序不导入 v172 正式预测器。它使用 v170 独立候选指标、v171 独立几何特征和增广最小二乘，而正式实现使用批量正规方程。

`22/22` 项检查全部通过：

- fold 包目标差 `0`；
- 三类留出标签同时突变后的 fit 目标变化 `0`；
- 特征最大差 `2.33e-11`；
- 预测最大差 `2.58e-11`；
- 策略报告最大差 `8.04e-12`；
- 三种策略的全部离散选择逐项一致。

## 成本和边界

v172 继承的几何特征 cache 为 `338` 个低模态 forward-equivalent 投影。cache 建成后，selector 拟合是 `0A+0A^T`，已选 H1 重建的逻辑在线账是 `1A+1A^T`。这些仍只是调用账，没有 fresh-process wall 或 whole-pipeline RSS，不能写成实际加速。

更重要的是，所有有限候选结果在 v170 已经开封。v172 的三重隔离防止这些留出结果进入本次拟合和预测，但它仍是 post-open 受控代理证据，不是 fresh 外部泛化。它只解决了相机子集选择，没有完成 observation-only warm initializer、exact lift 与 unchanged CGLS，也没有使用工况匹配的实验二维双分量位移。

所以本轮是明确的科学增量：**小型几何选择器的正结果不能用九个已开封场或四个时间点的简单适配来解释。** 下一门现在可以接完整 warm-start 物理链，并与 Zero / BP / CGLS / PCGLS / dual-ridge 做同成本、同精度比较。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v172: the geometry selector passes whole-field and time isolation

Updated: 2026-08-21

v171 showed that a small reported-geometry-only selector can find a safe five-camera subset when a complete calibration is held out. Its fit labels, however, still summarized all nine opened reconstructed fields and all four times. v172 therefore holds out three axes at once: one complete calibration, one complete field model, and one complete time.

Each fold fits on only twelve calibrations, eight field models, and three times, producing `13×9×4=468` held-out predictions. No outcome from any held-out axis may enter target construction, feature normalization, ridge fitting, the static control, tie breaking, or fallback.

Independent recomputation returns:

`PASS_WHOLE_FIELD_TIME_ISOLATED_GEOMETRY_SELECTOR_HEADROOM_V172`

The primary selector is strict-safe on `468/468` held-out cells, `13/13` complete calibrations, `9/9` complete field models, and `4/4` time strata. Its worst strict risk is `0.975390`, below the unit gate. Fit-static reaches `323/468` cells and the frozen v169 heuristic reaches `192/468`; neither passes a complete field or time stratum.

Across `t=0.00 / 0.25 / 0.75 / 1.00`, primary field p90 values are `0.330732 / 0.341935 / 0.323257 / 0.322180`, gradient p90 values are `0.613132 / 0.623236 / 0.632018 / 0.621204`, and observation p90 values are `0.130085 / 0.119602 / 0.117337 / 0.119413`. At the tightest `t=0.75` stratum, gradient worst is `0.731543`, below the frozen worst gate.

An independent program uses the independent v170 candidate metrics, independent v171 geometry features, and augmented least squares instead of the formal batched normal equations. All `22/22` checks pass. Fold-package and held-out-axis mutation differences are both zero; maximum feature, prediction, and policy-report differences are `2.33e-11`, `2.58e-11`, and `8.04e-12`, with every discrete selection agreeing.

The inherited geometry cache represents `338` forward-equivalent low-mode projections. After caching, selector fitting costs `0A+0A^T`, while one selected H1 reconstruction has a logical online ledger of `1A+1A^T`. No fresh wall or RSS experiment has been run.

All finite candidate outcomes were opened by v170. Triple-axis isolation prevents held-out outcomes from entering v172 fitting and prediction, but the result remains a post-open controlled-proxy diagnostic rather than fresh external generalization. It selects cameras only and does not yet establish the complete observation-only warm initializer, exact lift, unchanged CGLS, real resource saving, or condition-matched experimental BOST.

The precise increment is that simple adaptation to the nine opened field models or four opened times no longer explains the positive geometry selector. The next gate may now integrate the passing policy with the complete physical warm-start chain and compare it fairly against Zero, BP, CGLS, PCGLS, and dual-ridge.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
