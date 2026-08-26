# v213：实际源场的谱方向把 Case 5 两类九相机几何严格分开

## 结论

v210 用固定 64 个低频场探针检查几何本身的 Gram 谱下界，在 `169` 个“虚拟环形九相机 vs 师兄标定九相机”配对中有 `167/169` 个方向正确，但两族仍有两个重叠。v211 的局部无符号覆盖和 v212 的有符号射线相消也没有解释这个缺口。

v213 检验一个物理上不同的问题：**几何的弱谱方向，是否恰好承载了 Case 5 实际三维密度场的主要低频能量？**

结果是肯定的。把已经开封的 `42` 帧密度场投影到同一固定 low-64 子空间后，结果前冻结的“最坏帧源加权调和可观测性”实现 `169/169` 个跨族比较全部严格符合预期。正式科学判决为：

`PASS_ACTUAL_SOURCE_ALIGNMENT_STRICTLY_SEPARATES_CASE5_REFERENCE_V213`

## 固定主指标

对每套几何，用同一个 exact low-64 response 得到迹归一 Gram：

`G = 64 (M^T M / n) / trace(M^T M / n)`

把第 `t` 帧密度场在固定 low-64 正交基中的系数记为 `c_t`。若 `G = V diag(lambda) V^T`，则每帧主量为：

`h_t = ||c_t||^2 / sum_i ((v_i^T c_t)^2 / lambda_i)`

每套几何的唯一主指标取 `42` 帧中的最小值。数值越大，表示实际源场的低频能量越少落入该几何的弱谱方向。成功门在结果前固定为：13 套虚拟九相机的每一个主指标，都必须严格高于 13 套师兄标定九相机的每一个主指标。

## 结果

| 证据 | 师兄标定九相机 | 虚拟环形九相机 | 判读 |
|---|---:|---:|---|
| 源加权调和可观测性，最小值 | `0.51347` | `0.89028` | 虚拟族仍严格更高 |
| 源加权调和可观测性，中位数 | `0.52811` | `0.97869` | 差异明显 |
| 源加权调和可观测性，最大值 | `0.60360` | `1.03812` | 两族没有重叠 |
| 严格跨族比较 | `169` 对 | `169/169` 虚拟更高 | 主门通过 |
| 源盲 v210 谱下界 | `169` 对 | `167/169` 虚拟更高 | control 仍有重叠 |

虚拟九相机最小值减去师兄标定族最大值为 `0.28668`。固定 low-64 子空间捕获每帧三维场能量的 `77.69%` 到 `79.38%`，中位数为 `78.47%`。

## 独立复算

独立程序没有导入 v213 正式数值核心。它重新读取 `42` 个原始密度文件和三张网格，用不同分块与插值表达式重建密度场；用 SVD 而不是 QR 构造同一 low-64 子空间；再独立重建 `39` 套几何、全部 Gram、源投影、逐帧指标、最坏帧和 `169` 个比较。

`19/19` 项检查全部通过：

- low-64 投影最大差：`6.39e-15`
- 逐帧指标最大差：`3.60e-14`
- 逐几何指标与汇总最大差：`3.40e-14`
- Gram 特征值最大差：`1.15e-14`
- 相机换序最大差：`2.62e-14`

第一次独立程序在完成全部数值重建后，因为一个 `float32` 审计量无法写入 JSON 而失效，没有形成可用判决。v213.1 只把该值转换为内置 `float`，并从原始输入完整重跑正式与独立链；数据、主指标、阈值和几何均未改变。

## 这项正结果意味着什么

它把 v210 的“几何谱有方向性”推进为更具体的机制判断：**reference 是否充分，不只由最弱特征值决定，还取决于实际源场是否把能量加载到几何的弱谱方向。** 这也解释了为什么源盲谱下界仍有两个重叠，而源加权指标可以完全分开两族。

## 它仍不意味着什么

v213 使用了已经开封的真实三维密度场系数，因此只是 post-open、truth-aware 机制归因。它没有给出部署时可计算的预测器或 warm start，也没有证明 exact-call 减少、wall/RSS 加速、外部泛化、curved ray 或真实 BOST。

`algorithm_breakthrough=false`

## 下一门

下一步只检验这条机制能否被部署可见的信息复现：用二维观测和已知几何构造一个无训练、无真值输入的最小谱代理，仍要求 `169/169` 严格分离并由第二实现独立复算。代理若失败，v213 就保留为真值可见归因，不用 CNN、FNO、UNO、DeepONet 或 GPU 挽救。

---

# v213: Actual-Source Spectral Direction Strictly Separates the Two Case 5 Nine-Camera Families

## Conclusion

v210 used 64 fixed low-frequency field probes to evaluate the source-blind Gram spectral floor. It moved in the expected direction for `167/169` virtual-ring-nine versus supplied-nine comparisons, but two overlaps remained. Neither v211's unsigned local coverage nor v212's signed-line cancellation explained the gap.

v213 asks a physically different question: **do weak spectral directions of each geometry carry the actual low-frequency energy of the Case 5 density trajectory?**

Yes. After projecting all `42` opened density frames into the same fixed low-64 span, the preregistered worst-frame source-weighted harmonic observability moves strictly as expected in all `169/169` comparisons. The sealed decision is:

`PASS_ACTUAL_SOURCE_ALIGNMENT_STRICTLY_SEPARATES_CASE5_REFERENCE_V213`

## Fixed Primary

For each geometry, the exact low-64 response defines the trace-normalized Gram

`G = 64 (M^T M / n) / trace(M^T M / n)`.

Let `c_t` be the coefficients of density frame `t` in the fixed orthonormal low-64 basis. For `G = V diag(lambda) V^T`, the frame metric is

`h_t = ||c_t||^2 / sum_i ((v_i^T c_t)^2 / lambda_i)`.

The unique geometry primary is the minimum across all `42` frames. Larger values mean that less actual source energy lies in weak spectral directions. Before results, success was fixed to require every virtual-nine value to exceed every supplied-nine value.

## Results

| Evidence | Supplied nine | Virtual ring nine | Reading |
|---|---:|---:|---|
| Source-weighted harmonic minimum | `0.51347` | `0.89028` | virtual family remains higher |
| Source-weighted harmonic median | `0.52811` | `0.97869` | clear shift |
| Source-weighted harmonic maximum | `0.60360` | `1.03812` | no family overlap |
| Strict cross-family comparisons | `169` pairs | `169/169` virtual higher | primary passes |
| Source-blind v210 spectral floor | `169` pairs | `167/169` virtual higher | control still overlaps |

The strict gap between the virtual-nine minimum and supplied-nine maximum is `0.28668`. The fixed low-64 span captures `77.69%` to `79.38%` of each 3D field's energy, with a `78.47%` median.

## Independent Recalculation

The independent program does not import the formal v213 numerical core. It rereads all `42` raw density files and three grids, reconstructs the fields with different chunking and interpolation expressions, builds the same low-64 span with SVD rather than QR, and independently rebuilds all `39` geometries, Grams, source projections, frame metrics, worst frames, and `169` comparisons.

All `19/19` checks pass:

- maximum low-64 projection difference: `6.39e-15`
- maximum frame-metric difference: `3.60e-14`
- maximum geometry-metric and summary difference: `3.40e-14`
- maximum Gram-eigenvalue difference: `1.15e-14`
- maximum camera-permutation difference: `2.62e-14`

The first independent program completed all numerical reconstruction but failed to serialize one `float32` audit value, so it produced no usable decision. v213.1 changes only that value to a built-in `float` and reruns the complete formal and independent chain from raw inputs; the data, primary, thresholds, and geometry are unchanged.

## What This Positive Result Means

It sharpens v210's directional-geometry evidence into a source-specific mechanism: **reference adequacy depends not only on the weakest eigenvalue, but also on whether the actual source loads weak spectral directions of the geometry.** This explains why the source-blind floor retains two overlaps while the source-weighted metric fully separates the families.

## What It Still Does Not Mean

v213 reads the opened 3D density coefficients and is therefore a post-open, truth-aware mechanism attribution. It provides no deployable predictor or warm start and establishes no exact-call reduction, wall/RSS speedup, external generalization, curved-ray result, or real BOST result.

`algorithm_breakthrough=false`

## Next Gate

The next gate asks only whether deployment-visible 2D observations and known geometry can reproduce this mechanism with one untrained, truth-free minimal spectral proxy. It retains the `169/169` strict-separation gate and independent second implementation. If the proxy fails, v213 remains truth-aware attribution only; no CNN, FNO, UNO, DeepONet, or GPU rescue is authorized.
