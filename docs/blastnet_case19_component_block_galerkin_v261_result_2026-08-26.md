# v261：双分量 2×2 Galerkin 修正没有恢复 Case 19 的 matched-accuracy

## 这次只检验一个物理上不同的机制

v260 已经否定了按相机残差能量加权。v261 不再调权重，而是利用 BOS 位移天然包含两个有符号探测器分量这一结构：从 geometry-Jacobi PCGLS K13 的部署可见残差出发，始终保留两个完整分量；每个分量分别做一次精确伴随和冻结预条件，得到两个场方向及其测量投影，再解一个固定的 `2×2` Galerkin 系统混合这两个方向。

规则不读取真值、时间、rig 标签或相机 ID，不挑选分量，也没有阈值、回退或学习系数。主候选为 `15A+15A^T`，与未修改 K15 同价；v258 对照为 `15A+14A^T`，raw K14 为 `14A+14A^T`，K16 reference 为 `16A+16A^T`。试验仍只覆盖已经开封的 Case 19 十三套 rig 的首帧。

## 独立结果

完全独立的第二实现重新构造 K13、两个有符号分量、各自的伴随与预条件方向、`2×2` 相关系统、相机换序、物理残差 replay、四个对照和全部门。`41/41` 项检查全真。两套实现的最大场相对差为 `2.26e-9`，归一化残差差为 `5.60e-10`，指标绝对差为 `1.10e-10`，分量系数相对差为 `6.13e-10`，物理 replay 绝对差为 `2.61e-16`。

主候选只通过 `3/13` 个绝对门，K16-matched 为 `0/13`。绝对指标的 p90-higher（field / full-gradient / interior-gradient / observation）为：

- `0.36604 / 0.65275 / 0.82805 / 0.07081`

相对 K16 的 matched 比值 p90-higher 为：

- `1.17163 / 1.14928 / 1.11832 / 1.33611`

worst 比值为：

- `1.17856 / 1.15739 / 1.11966 / 1.34725`

四项 matched 指标都超过冻结的 `1.05` 门。公平对照也排除了“只是少迭代一步”的解释：同价未修改 K15 通过 `9/13` 个绝对门，便宜一个伴随调用的 v258 通过 `13/13`；两者都优于主候选。raw K14 为 `7/13`。因此，按两个探测器分量构造的小块 Galerkin 方向不仅没有补上 K16 差距，还损伤了已有的绝对精度。

## 判决与边界

封存判决为 `FAIL_CASE19_COMPONENT_BLOCK_GALERKIN_FRAME_ZERO_V261`。当前“两个完整有符号分量 + 两个预条件方向 + 固定 2×2 Galerkin 混合”机制族关闭；不扩大分组或块数，不训练系数预测器，不扩展到完整序列，不租 GPU，也不运行 wall/RSS 资源门。

这是一条有效的开封后机制负证据，不是算法通过、外部门、速度结果、曲折光线验证或真实 BOST 结果。它不证明整条 C 路线不可能。后续需要另行结果前冻结一个物理上真正不同的部署可见机制，或获得真正配对的二维双分量 BOST 数据。`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v261: the two-component 2×2 Galerkin correction does not recover Case 19 matched accuracy

## The single physically distinct mechanism tested

v260 rejected residual-energy camera weighting. v261 does not retune those weights. Instead, it uses the fact that BOS displacement has two signed detector components. Starting from the deployment-visible residual after geometry-Jacobi PCGLS K13, it always retains both complete components. Each component receives its own exact adjoint and frozen preconditioner, producing two field directions and measurement projections, which are mixed by one fixed `2×2` Galerkin system.

The rule reads no truth, time, rig label, or camera ID, selects no component, and uses no threshold, fallback, or learned coefficient. The primary costs `15A+15AT`, equal to unchanged K15. The v258 control costs `15A+14AT`, raw K14 costs `14A+14AT`, and the K16 reference costs `16A+16AT`. The experiment remains limited to frame zero of the thirteen already-opened Case 19 rigs.

## Independent result

A fully independent implementation rebuilds K13, both signed components, their adjoint and preconditioned directions, the `2×2` correlation system, camera permutation, physical residual replay, four controls, and every gate. All `41/41` checks pass. Maximum cross-implementation differences are `2.26e-9` relative for the field, `5.60e-10` for normalized residuals, `1.10e-10` absolute for metrics, `6.13e-10` relative for component coefficients, and `2.61e-16` absolute for physical replay.

The primary clears only `3/13` absolute cells and reaches `0/13` under K16-matched accuracy. Its p90-higher absolute metrics (field / full gradient / interior gradient / observation) are:

- `0.36604 / 0.65275 / 0.82805 / 0.07081`

Its K16-matched p90-higher ratios are:

- `1.17163 / 1.14928 / 1.11832 / 1.33611`

The worst ratios are:

- `1.17856 / 1.15739 / 1.11966 / 1.34725`

All four matched metrics exceed the frozen `1.05` gate. Fair controls also rule out a simple one-iteration explanation: equal-cost unchanged K15 clears `9/13` absolute cells, while v258, which uses one fewer adjoint call, clears `13/13`; both outperform the primary. Raw K14 reaches `7/13`. The detector-component block therefore fails to close the K16 gap and damages absolute accuracy already available from the controls.

## Verdict and boundary

The sealed decision is `FAIL_CASE19_COMPONENT_BLOCK_GALERKIN_FRAME_ZERO_V261`. The exact family consisting of two complete signed components, two preconditioned directions, and one fixed `2×2` Galerkin mix is closed. There is no larger grouping or block, coefficient predictor, full-sequence expansion, GPU rental, or wall/RSS resource gate.

This is valid post-open mechanism evidence, not an algorithmic pass, external gate, speed result, curved-ray validation, or real BOST evidence. It does not prove the entire C route impossible. Further work requires a genuinely different deployment-visible physical mechanism under a separate result-before-run contract or genuinely paired two-component BOST observations. `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
