# v267：同步双颜色局部步发生强跨块干扰

## 为什么做

v266 已经说明，固定半射线候选的失败不只是未选中射线上的溢出：最严重的尾部在两半观测上都缺少 K16 水平的恢复。v267 因此检验一个物理上更完整但仍可证伪的确定性机制：把两分量探测器射线按偶、奇棋盘颜色分成两块，每块都从同一个未修改父残差出发，独立计算精确局部 normal、预条件方向与线搜索，最后同步相加两块修正。

该候选只读取部署时可见的观测、报告几何与求解器状态，不读取 CFD 真值、模型标签或轨迹标签；真值只在预测封存后用于评分。没有训练参数，也没有相位、顺序、比例、阻尼、深度或超参数搜索。候选逻辑账为 `16A+15A^T`，K16 reference 为 `16A+16A^T`。

## 独立结果

formal 与完全独立第二实现覆盖已打开 BLASTNet Case 19 的 `13` 套相机、每套 `33` 帧，共 `429` 个单元。独立验证通过 `24/24` 项检查；新场最大相对差为 `1.33e-9`，残差观测归一化差为 `2.08e-10`，指标最大绝对差为 `1.47e-10`。局部伴随误差不超过 `7.90e-14`，物理重放误差不超过 `1.94e-14`，相机乱序差为 `0`。

候选通过绝对门 `429/429` 个单元与 `13/13` 套相机，但相对 K16 的 matched-accuracy 只有 `2/429` 个单元、`0/13` 套完整相机。其余 `427` 个失败全部只来自 observation，field、full-gradient 与 interior-gradient 的 matched 失败均为 `0`。同工作量 full-row control 达到 `219/429`，封存的单半射线 control 为 `200/429`，v258 父状态为 `4/429`，K15 为 `0/429`；这些方法同样没有一条通过完整轨迹门。

## 为什么失败

两个颜色块各自单独评估时，都在 `429/429` 个单元上改善自己的局部目标；但是把两步同步相加后，组合结果在 `429/429` 个单元上都比每块自己的对角预测更差。组合状态相对各自局部预测的残差比中位数约为 `1.417` 与 `1.413`。全观测残差相对 v258 父状态在 `419/429` 个单元上变差，只在 `10/429` 个单元上改善。

这说明两个局部最优步不是可独立叠加的：每一块忽略了另一块经完整 forward 产生的响应，跨块耦合抵消了局部收益。它不是浮点误差，也不是某个单元偶然越线。

## 判决边界

封存判决是 `FAIL_CASE19_TWO_COLOR_ADDITIVE_SCHWARZ_V267`。当前精确同步双颜色路线关闭，不再改奇偶划分、更新顺序、阻尼、深度或线搜索，也不以更大的 CNN / FNO / UNO、GPU 或更多算力挽救。

这是已打开 Case 19 上的 post-open 负机制证据。它不是未打开外部门、真实 BOST、wall/RSS、curved-ray 或资源收益结果。低层物理与数据读取仍共享，所以不能声称端到端物理独立。后续只有在结果前能明确冻结一个物理上真正不同、显式处理跨块耦合的完整观测机制，或拿到工况配对的真实二维双分量 BOST 数据时，才值得继续开新科学门。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`effective_exact_call_reduction=false`、`resource_speedup=false`、`real_bost=false`。

# v267: synchronous two-color local steps interfere strongly

## Why this test

v266 showed that the fixed half-ray candidate does not fail merely through spillover onto unselected rays: the most severe tails lack K16-level recovery on both observation halves. v267 therefore tests a more complete but still falsifiable deterministic mechanism. Detector rays from both components are split into even and odd checkerboard colors. From the same unchanged parent residual, each block independently computes its exact local normal, preconditioned direction, and line search; the two corrections are then added synchronously.

The candidate reads only deployment-visible observations, reported geometry, and solver state. It reads no CFD truth, model label, or trajectory label; truth is used only for scoring after predictions are sealed. There are no trainable parameters and no search over phase, order, fraction, damping, depth, or hyperparameters. The candidate ledger is `16A+15AT`, versus `16A+16AT` for the K16 reference.

## Independent result

The formal and fully independent implementations cover all `429` cells from `13` rigs by `33` frames in the already-opened BLASTNet Case 19. Independent validation passes `24/24` checks. The maximum relative difference in the new field is `1.33e-9`, the maximum observation-normalized residual difference is `2.08e-10`, and the maximum absolute metric difference is `1.47e-10`. Partial-adjoint error is at most `7.90e-14`, physical-replay error at most `1.94e-14`, and camera-permutation difference is `0`.

The candidate clears the absolute gate on `429/429` cells and `13/13` rigs, but K16-matched accuracy reaches only `2/429` cells and `0/13` complete rigs. All remaining `427` failures are observation-only; matched field, full-gradient, and interior-gradient failures are all `0`. The equal-work full-row control reaches `219/429`, the sealed single-half control `200/429`, the v258 parent `4/429`, and K15 `0/429`; none clears the complete-rig gate.

## Why it fails

Evaluated alone, each color block improves its own local objective on all `429/429` cells. After the two steps are added synchronously, however, the combined result is worse than each block's own diagonal prediction on all `429/429` cells. The median combined-to-local residual ratios are about `1.417` and `1.413`. Full-observation residual worsens relative to the v258 parent on `419/429` cells and improves on only `10/429`.

The two locally optimal steps are therefore not independently additive: each block ignores the complete-forward response produced by the other, and cross-block coupling cancels the local gain. This is neither a floating-point artifact nor an isolated threshold crossing.

## Decision boundary

The sealed decision is `FAIL_CASE19_TWO_COLOR_ADDITIVE_SCHWARZ_V267`. The exact synchronous two-color route is closed. There will be no rescue by changing parity, update order, damping, depth, or line search, and no larger CNN / FNO / UNO, GPU, or extra-compute rescue.

This is post-open negative mechanism evidence on the already-opened Case 19. It is not an unopened external gate, real-BOST result, wall/RSS result, curved-ray validation, or resource gain. Low-level physics and data loading remain shared, so end-to-end physics independence is not established. A new scientific gate is justified only if one physically distinct complete-observation mechanism that explicitly handles cross-block coupling can be frozen before results, or if condition-matched real two-component BOST data arrive.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `effective_exact_call_reduction=false`, `resource_speedup=false`, `real_bost=false`.
