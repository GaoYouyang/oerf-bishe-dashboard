# v266：半射线失败不是单纯的未选中射线溢出

## 为什么做

v265.1 已独立确认：固定偶相位半射线修正在已打开 BLASTNet Case 19 完整序列上只有 `200/429` 个单元通过 K16-matched 门，`229` 个失败全部来自 observation。v266 不修改候选，也不再搜索相位、比例、深度、权重或阻尼；它只把已封存的观测残差精确拆成“被半射线机制选中的 `1152` 条射线”和其余 `1152` 条射线，检验一个更窄的问题：失败是否只是修正污染了未选中的另一半射线。

该诊断不读取真值或密度，不构造新场，不训练模型，也不增加精确算子调用，新增成本为 `0A+0A^T`。

## 独立结果

完全独立第二实现通过 `18/18` 项检查，并准确复现 v265.1 的 `200` 个 matched 单元、`229` 个失败单元和 `0/13` 完整轨迹。相对封存父状态，候选在全部 `429/429` 个单元上都没有增大选中射线残差；同时不存在“选中部分下降、未选中部分反而上升”的单元。

在 `229` 个 matched 失败中，`119` 个仅在未选中射线部分相对 K16 留有缺口，另外 `110` 个在选中与未选中两部分都留有缺口；selected-only 和数值抵消均为 `0`。逐 rig 的 p90 违规由 `1` 个 complement-only 和 `12` 个双侧缺口组成；`13/13` 个 worst 违规全部是双侧缺口。

观测 matched 比仍为 p50 / p90-higher / worst = `1.06094 / 1.23503 / 1.85517`。正式能量的独立重算最大相对差为 `6.01e-15`，能量守恒差不超过 `4.60e-15`，选中射线轨迹差不超过 `4.34e-15`，比值重建差为 `0`。

## 判决边界

封存判决是 `MIXED_SELECTED_AND_UNSELECTED_RAY_DEFICIT_V266`。数据否定了“v265.1 只是把误差推到未选中射线”的单一解释：最严重的尾部在选中与未选中两部分都缺少 K16 水平的恢复。固定偶相位半射线路线继续关闭，不能靠换相位、比例、深度、权重或更大模型挽救。

这次分型是 post-open 机制归因，不是新候选、重建成功、速度或外部泛化证据。由于 `119` 与 `110` 呈混合分布，它也不能在全局上否定所有“单侧目标但显式观测补集”的未来机制。后续只接受物理上真正不同、能够同时约束完整观测的 solver-native 机制，或工况匹配的真实二维 BOST 数据。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v266: half-ray failure is not pure unselected-ray spillover

## Why this diagnostic

v265.1 independently established that the fixed even-quincunx half-ray correction clears the K16-matched gate on only `200/429` cells of the complete opened BLASTNet Case 19 sequence, with all `229` failures coming from observation. v266 changes no candidate and searches no phase, fraction, depth, weighting, or damping. It only partitions the sealed observation residual into the `1152` rays selected by the half-ray mechanism and the other `1152` rays, asking a narrower question: are the failures caused solely by contaminating the unselected half?

The diagnostic reads neither truth nor density, constructs no new field, trains no model, and adds no exact operator call. Its incremental ledger is `0A+0AT`.

## Independent result

A fully independent second implementation passes `18/18` checks and exactly reproduces the v265.1 counts of `200` matched cells, `229` failed cells, and `0/13` complete trajectories. Relative to the sealed parent state, the candidate does not increase selected-ray residual in any of the `429/429` cells. There is also no cell in which selected residual falls while complement residual rises relative to the parent.

Among the `229` matched failures, `119` retain a K16-relative deficit only on the unselected complement, while `110` retain deficits on both selected and unselected partitions. The selected-only and numerical-cancellation classes are both `0`. Across rig-level p90 violations, `1` is complement-only and `12` are two-sided; all `13/13` worst violations are two-sided.

The observation matched p50 / p90-higher / worst ratios remain `1.06094 / 1.23503 / 1.85517`. The maximum relative difference in independently recomputed formal energies is `6.01e-15`, energy-conservation difference is at most `4.60e-15`, selected-trace difference is at most `4.34e-15`, and ratio reconstruction differs by `0`.

## Decision boundary

The sealed decision is `MIXED_SELECTED_AND_UNSELECTED_RAY_DEFICIT_V266`. The data reject the single explanation that v265.1 merely pushes error into unselected rays: the most severe tails lack K16-level recovery on both selected and unselected partitions. The fixed even-quincunx half-ray route remains closed and cannot be rescued by phase, fraction, depth, weighting, or larger-model retuning.

This is a post-open mechanism attribution, not a new candidate, reconstruction success, speed result, or external-generalization result. Because the `119` versus `110` split is mixed, it also does not globally close every possible one-sided objective that explicitly observes its complement. Any next step must use a genuinely different solver-native mechanism that constrains the complete observation, or condition-matched real two-component BOST data.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
