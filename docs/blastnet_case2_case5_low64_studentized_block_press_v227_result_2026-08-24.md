# v227：几何白化提高安全接受，但逐 rig 效用门仍失败

## 结论

v226 的原始相机分块 PRESS 已经做到 Case 2 零危险误接，但 Case 5 最差留一 rig 只接受 `4/42`。v227 不改阈值或效用门，而是检验一个物理和统计上不同的问题：被遮相机在当前几何下越难预测，其误差是否应该按几何预测协方差进行白化后再聚合。

正式程序与完全独立第二实现得到一致判决：

`FAIL_LOW64_STUDENTIZED_BLOCK_PRESS_CERTIFICATE_V227`

几何白化确实改变了筛选结果。Case 2 的安全接受由 v226 的 `297` 提高到 `323`，危险误接仍为 `0`；混合策略在 Case 2 和 Case 5 仍都达到 `13/13` 完整 rig 精度。但是 Case 5 最差 rig 仍只有 `4/42=9.52%`，低于冻结 `10%` 门要求的 `5/42`。失败位置从 v226 的 rig 11 移到 v227 的 rig 4，说明白化有作用，却没有解决跨 rig 校准稳定性。

这必须按原门记录为失败，不能因为只差一个帧而包装成成功。

## 做了什么

对被遮相机 `j`，用其余八台相机的固定 Low-64 响应 `M_-j` 拟合系数，再在第 `j` 台相机上得到预测误差 `e_j`。v227 只从 reported geometry 生成预测协方差：

`V_j = I + M_j (M_-j^T M_-j)^-1 M_j^T`

主分数为：

`S = sqrt(sum_j e_j^T V_j^-1 e_j) / max(E_full, 1e-12 ||y||)`

分数越低越安全。它只读取部署可见的二维观测和 reported geometry；CFD 真值、危险标签、轨迹名称和未开封工况不进入分数、阈值或回退决策。阈值继续只由已知安全 Case 5 生成：Case 5 做完整留一 rig，Case 2 在读取真值门前使用全部 Case 5 形成的单一阈值。等号接受，没有协方差 floor、ridge、clip、尺度拟合、阈值搜索或异常点删除。

v226 原始未白化 PRESS 作为冻结父 control 全量重建，分数、阈值和离散决策差均为 `0`。

## 结果

| 策略与条件 | 接受单元 | 各 rig 最低接受率 | 接受的不安全单元 | 完整 rig 精度通过 | 判决 |
|---|---:|---:|---:|---:|---|
| Studentized block-PRESS，Case 5 LORO | `123 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | 效用门失败 |
| Studentized block-PRESS，Case 2 | `323 / 715` | `34.55%` | `0` | `13 / 13` | 安全与精度通过 |
| v226 原始 block-PRESS，Case 5 LORO | `126 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | 效用门失败 |
| v226 原始 block-PRESS，Case 2 | `297 / 715` | `32.73%` | `0` | `13 / 13` | 安全与精度通过 |

白化后的 Case 2 安全接受率为 `62.36%`，逐 rig 接受率为 `34.55%` 至 `54.55%`。Case 5 总计接受 `123/546`，逐 rig 接受率为 `9.52%` 至 `47.62%`。最差 rig 的第五个目标分数仍高于留一阈值约 `1.20e-4`，所以它不是浮点等号问题。

混合策略的 Case 2/5 最大 matched ratio 分别为 `1.027761` 和 `1.007896`。被接受的 Direct K11 逻辑账为 `12A+11A^T`，回退 Zero-PCGLS K16 为 `16A+16A^T`，每个 rig 的平均 `A` 与 `A^T` 都低于 `16`。但证书整体失败，所以这些只是封存调用账，不授权 fresh wall/RSS，也不能声称部署资源收益。

## 独立复算

正式实现使用 response SVD 与显式协方差 Cholesky 白化；独立实现改用正规矩阵特征分解和 Woodbury 二次型。独立程序重建全部 `1261` 个单元、九个遮相机系统、几何协方差、白化分数、Case 5 顺序统计阈值、接受决策、逐 rig 物理门和调用账。

所有 `19/19` 项必需检查通过。正式与独立特征、阈值、汇总和相机换序最大差分别为 `1.11e-15`、`2.22e-16`、`2.57e-11` 和 `1.55e-15`，离散决策完全一致。协方差最小/最大特征值约为 `1.00/8.24`，白化重建最大差为 `3.77e-15`。

独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_STUDENTIZED_BLOCK_PRESS_V227`

共享冻结的底层 physics kernels 仍存在，因此 `end_to_end_physics_independence_proven=false`。

## 证据边界

- v227 是已开封 Case 2/5 上的 post-open 跨工况证书诊断，不是 fresh 外门；
- 它关闭的是当前 geometry-studentized block-PRESS 单分数证书，不证明全部多视角证书或整个 C 路线不可能；
- 不允许看到结果后修改协方差公式、数值 floor、阈值、`10%` 接受比例、Low-64 秩或 PCGLS 深度；
- 公式在代数上支持可变相机数，但本次只测试九相机，不能声称 `5/7/12` 相机泛化；
- 不训练 CNN/FNO/UNO/DeepONet，不租 GPU，也不运行 wall/RSS 或打开新工况；
- 没有部署算法、稳定 exact-call 收益、外部泛化、曲线光路或真实 BOST 结果。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v227: Geometry Whitening Raises Safe Acceptance but Still Fails the Per-Rig Utility Gate

## Conclusion

The raw camera-block PRESS certificate in v226 reaches zero unsafe Case 2 accepts, but the worst held-out Case 5 rig accepts only `4/42` frames. v227 does not change the threshold or utility gate. It tests a physically and statistically different question: when a held-out camera is harder to predict under the reported geometry, should its error be whitened by geometry-derived predictive covariance before aggregation?

The formal program and fully separate second implementation agree on:

`FAIL_LOW64_STUDENTIZED_BLOCK_PRESS_CERTIFICATE_V227`

Geometry whitening does change the selection. Safe Case 2 accepts rise from `297` in v226 to `323`, while unsafe accepts remain `0`; the mixed policy still reaches `13/13` complete rigs in both Cases 2 and 5. However, the worst Case 5 rig remains at `4/42=9.52%`, below the `5/42` required by the frozen `10%` gate. The failure moves from rig 11 in v226 to rig 4 in v227. Whitening therefore has an effect but does not solve cross-rig calibration stability.

This remains a failure under the preregistered gate. A one-frame shortfall cannot be repackaged as success.

## What was done

For held-out camera `j`, the other eight fixed Low-64 responses `M_-j` fit the coefficients, yielding prediction error `e_j` on camera `j`. v227 uses only reported geometry to form predictive covariance:

`V_j = I + M_j (M_-j^T M_-j)^-1 M_j^T`

The primary score is:

`S = sqrt(sum_j e_j^T V_j^-1 e_j) / max(E_full, 1e-12 ||y||)`

Lower is safer. The score reads only deployment-visible 2D observations and reported geometry. CFD truth, unsafe labels, trajectory names, and unopened conditions do not enter score, threshold, or fallback decisions. Thresholds still come only from known-safe Case 5: complete leave-one-rig-out evaluation for Case 5 and a single all-Case-5 threshold for Case 2, sealed before Case 2 truth gates are read. Equality accepts. There is no covariance floor, ridge, clipping, scale fit, threshold search, or outlier deletion.

The raw unwhitened v226 PRESS is rebuilt as a frozen parent control with zero score, threshold, and discrete-decision differences.

## Results

| Policy and condition | Accepted cells | Minimum rig acceptance | Accepted unsafe cells | Complete rigs passing accuracy | Decision |
|---|---:|---:|---:|---:|---|
| Studentized block-PRESS, Case 5 LORO | `123 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | utility gate fails |
| Studentized block-PRESS, Case 2 | `323 / 715` | `34.55%` | `0` | `13 / 13` | safety and accuracy pass |
| Raw v226 block-PRESS, Case 5 LORO | `126 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | utility gate fails |
| Raw v226 block-PRESS, Case 2 | `297 / 715` | `32.73%` | `0` | `13 / 13` | safety and accuracy pass |

The whitened Case 2 safe-accept fraction is `62.36%`, with per-rig acceptance from `34.55%` to `54.55%`. Case 5 accepts `123/546` overall, with per-rig acceptance from `9.52%` to `47.62%`. The fifth target score in the worst rig remains about `1.20e-4` above its leave-one-rig-out threshold, so this is not a floating-point equality issue.

Maximum matched ratios for the mixed policy are `1.027761` in Case 2 and `1.007896` in Case 5. The accepted Direct K11 path has a logical ledger of `12A+11A^T`; fallback Zero-PCGLS K16 uses `16A+16A^T`, and every rig has mean `A` and `A^T` below `16`. Because the certificate fails overall, these remain sealed call ledgers and do not authorize fresh wall/RSS or a deployment resource claim.

## Independent recomputation

The formal implementation uses response SVD and explicit predictive-covariance Cholesky whitening. The independent implementation uses normal-matrix eigendecomposition and a Woodbury quadratic form. It rebuilds all `1261` cells, nine held-out-camera systems, geometry covariances, whitened scores, Case 5 order-statistic thresholds, accept decisions, rig physics gates, and call ledgers.

All `19/19` required checks pass. Maximum formal-independent feature, threshold, summary, and camera-permutation differences are `1.11e-15`, `2.22e-16`, `2.57e-11`, and `1.55e-15`, with identical discrete decisions. Predictive-covariance minimum/maximum eigenvalues are approximately `1.00/8.24`, and the maximum whitening reconstruction difference is `3.77e-15`.

The independent status is:

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_STUDENTIZED_BLOCK_PRESS_V227`

Frozen low-level physics kernels remain shared, so `end_to_end_physics_independence_proven=false`.

## Evidence boundary

- v227 is a post-open cross-condition certificate diagnostic on opened Cases 2 and 5, not a fresh external gate;
- it closes this geometry-studentized block-PRESS single-score certificate, not every multiview certificate or the C route;
- the covariance formula, numerical floor, threshold, `10%` acceptance fraction, Low-64 rank, and PCGLS depths may not be changed after results;
- the formula is algebraically variable-cardinality, but only nine-camera behavior is tested here; `5/7/12`-camera generalization is not established;
- no CNN/FNO/UNO/DeepONet training, GPU rental, wall/RSS run, or new-condition opening is authorized;
- there is no deployment algorithm, stable exact-call gain, external generalization, curved-ray, or real-BOST result.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
