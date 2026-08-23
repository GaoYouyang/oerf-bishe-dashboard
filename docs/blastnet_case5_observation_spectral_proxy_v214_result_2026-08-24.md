# v214：二维观测与已知几何足以复现 Case 5 的谱对齐判别

## 结论

v213 证明，Case 5 实际三维源场在几何弱谱方向上的能量加载，可以把两类九相机几何严格分开；但它直接读取了已开封的三维密度真值。

v214 把这个真值输入去掉。对每套已知几何，代理只接收固定 low-64 响应 `M` 和当前二维观测 `y`，用无正则的 Moore-Penrose 解估计 64 个系数。它不读真值系数、帧或几何类别、父结果、重建、残差或任何拟合阈值。

结果是：观测代理在 `169/169` 个“虚拟环形九相机 vs 师兄标定九相机”配对中全部严格更高。封存科学判决为：

`PASS_OBSERVATION_ONLY_SPECTRAL_ALIGNMENT_PROXY_STRICTLY_SEPARATES_CASE5_REFERENCE_V214`

## 固定代理

对报告几何的 low-64 响应 `M` 做 economy SVD：

`M = U diag(s) V^T`

结果前唯一冻结的估计为：

`c_hat = V diag(1/s) U^T y`

所有 64 个数值奇异方向都必须有效；不允许 ridge、秩搜索、缩放、裁剪、回退或候选切换。再把 `c_hat` 放入 v213 已冻结的调和可观测性公式，每套几何取 42 帧最小值。

## 严格结果

| 证据 | 师兄标定九相机 | 虚拟环形九相机 | 判读 |
|---|---:|---:|---|
| 观测代理调和可观测性，最小值 | `0.19783` | `0.98917` | 虚拟族更高 |
| 中位数 | `0.32483` | `1.06574` | 明显分离 |
| 最大值 | `0.59186` | `1.11703` | 两族无重叠 |
| 严格跨族比较 | `169` 对 | `169/169` 虚拟更高 | 主门通过 |
| 源盲 v210 control | `169` 对 | `167/169` 虚拟更高 | 仍有两个重叠 |

虚拟九相机最小值减去师兄标定族最大值为 `0.39730`。这是严格间隔，不是事后设置的分类阈值。

## 独立复算

独立程序从 42 个原始密度文件与网格重建三维场，用 SVD 而不是正式侧 QR 重建同一物理子空间，独立重建 39 套几何的物理算子和完整二维观测，并用 `M^T M` 特征分解而不是正式侧 SVD 求代理。

`19/19` 项检查全部通过：

- 物理代理场最大相对差：`1.12e-13`
- 逐帧指标最大差：`1.71e-13`
- 逐几何指标最大差：`8.49e-14`
- 奇异值最大相对差：`6.10e-14`
- 相机换序最大差：`1.87e-13`

## 这项正结果意味着什么

它排除了一个关键悲观解释：为区分 Case 5 两类 reference 几何，并不必须在部署时读取三维 CFD 真值。已知几何与当前二维观测已包含足以复现谱对齐判别的信息。

## 它仍不意味着什么

v214 仍是已开封 Case 5 上的 post-open 机制诊断。三维真值用于生成无噪声合成二维观测，且尚未把代理场作为 warm start 进入未修改 CGLS 做物理重放。因此它不是部署算法、matched-accuracy、exact-call 减少、wall/RSS 加速、外部泛化或真实 BOST 结果。

`algorithm_breakthrough=false`

## 成本边界

- 几何 low-64 响应缓存：`2496` 次 forward-equivalent 探针
- 本次合成观测生成：`1638` 次 `A`-equivalent
- 在观测和几何缓存已存在后，代理阶段的逻辑账：`0A+0A^T`

最后一行不能与前两行分开来声称端到端加速。

## 下一门

结果前单独冻结一个最小 warm-start 可行性实验：把 v214 的 observation-only 代理场作为初值，进入未修改 CGLS，与 Zero/BP/CGLS/PCGLS 和便宜控制在实际 `A/A^T` 账上公平比较。只有字段、梯度和观测精度门全部通过，才能谈调用减少。

---

# v214: 2D Observations and Known Geometry Reproduce the Case 5 Spectral-Alignment Decision

## Conclusion

v213 showed that loading of actual Case 5 source energy onto weak geometry-spectrum directions strictly separates the two nine-camera families, but it directly read opened 3D density truth.

v214 removes that truth input. For each known geometry, the proxy receives only the fixed low-64 response `M` and current 2D observation `y`, then estimates 64 coefficients with the unregularized Moore-Penrose solution. It receives no truth coefficients, frame or family label, parent outcome, reconstruction, residual, or fitted threshold.

The observation proxy is strictly higher for the virtual ring-nine family in all `169/169` virtual-nine versus supplied-nine comparisons. The sealed decision is:

`PASS_OBSERVATION_ONLY_SPECTRAL_ALIGNMENT_PROXY_STRICTLY_SEPARATES_CASE5_REFERENCE_V214`

## Fixed Proxy

For the reported-geometry low-64 response, compute the economy SVD

`M = U diag(s) V^T`

and the only preregistered estimate

`c_hat = V diag(1/s) U^T y`.

All 64 numerical singular directions must be retained. No ridge, rank search, scaling, clipping, fallback, or candidate switch is allowed. The estimate is inserted into the frozen v213 harmonic-observability formula, and each geometry takes its minimum across 42 frames.

## Strict Results

| Evidence | Supplied nine | Virtual ring nine | Reading |
|---|---:|---:|---|
| Observation-proxy harmonic minimum | `0.19783` | `0.98917` | virtual family higher |
| Median | `0.32483` | `1.06574` | clear separation |
| Maximum | `0.59186` | `1.11703` | no family overlap |
| Strict cross-family comparisons | `169` pairs | `169/169` virtual higher | primary passes |
| Source-blind v210 control | `169` pairs | `167/169` virtual higher | two overlaps remain |

The virtual-nine minimum minus the supplied-family maximum is `0.39730`. This is a strict family gap, not a post-hoc classification threshold.

## Independent Recalculation

The independent program rebuilds the 3D fields from all 42 raw density files and grids, uses SVD rather than the formal QR to reconstruct the same physical low-mode span, independently reconstructs all 39 physical operators and full 2D observations, and solves the proxy through an eigendecomposition of `M^T M` rather than the formal SVD.

All `19/19` checks pass:

- maximum physical proxy-field relative difference: `1.12e-13`
- maximum frame-metric difference: `1.71e-13`
- maximum geometry-metric difference: `8.49e-14`
- maximum singular-value relative difference: `6.10e-14`
- maximum camera-permutation difference: `1.87e-13`

## What This Positive Result Means

It rejects one important pessimistic explanation: deployment-time access to 3D CFD truth is not necessary to distinguish the two Case 5 reference-geometry families. Known geometry and the current 2D observation already contain enough information to reproduce the spectral-alignment decision.

## What It Still Does Not Mean

v214 remains a post-open mechanism diagnostic on opened Case 5. The 3D truth generates noiseless synthetic observations, and the proxy field has not yet been replayed as a warm start through unchanged CGLS. It therefore establishes no deployable algorithm, matched accuracy, exact-call reduction, wall/RSS speedup, external generalization, or real-BOST result.

`algorithm_breakthrough=false`

## Cost Boundary

- geometry low-64 response cache: `2496` forward-equivalent probes
- synthetic-observation generation in this diagnostic: `1638` `A`-equivalents
- logical proxy stage after observation and geometry cache exist: `0A+0A^T`

The final line cannot be separated from the first two to claim end-to-end speedup.

## Next Gate

Freeze a separate minimal warm-start feasibility experiment before results: use the v214 observation-only proxy field as an initializer for unchanged CGLS and compare it fairly against Zero/BP/CGLS/PCGLS and cheap controls using actual `A/A^T` receipts. Exact-call reduction can be discussed only after field, gradient, and observation accuracy gates all pass.
