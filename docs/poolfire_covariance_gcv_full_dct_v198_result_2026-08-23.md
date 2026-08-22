# v198：候选全过，但更简单的同价控制也全过

## 结论先说

v198 的经验协方差 GCV 候选在已经打开的 p22 完整开发轨迹上通过了全部 **2626/2626** 个严格单元和 **26/26** 个完整标定组，逻辑在线账为 **2A+1A^T**。但预注册的 identity-GCV 控制在完全相同的调用账下也达到 **2626/2626、26/26**。

因此正式判决是 `PASS_CHEAPER_CONTROL_EXPLAINS_COVARIANCE_GCV_V198`：结果支持“完整 DCT 表示加简单正则可以修复 K1 尾部”，不支持“经验协方差是必要成分”。`algorithm_breakthrough=false`。

## 做了什么

候选保留每个相机的完整 DCT 观测坐标，不再做固定选列或 sketch。它从既有 fit 场的仿射奇异值构造经验坐标协方差，并仅用当前观测和已知几何，通过 GCV 在冻结的十档网格上选择正则强度；随后执行一次未修改的物理 CGLS K1。

同时运行了同价 identity-GCV、未正则 full-DCT K1、v197 full-DCT K2 reference、Zero、BP/CGLS、PCGLS 与历史 dual-ridge 等控制。预测坐标、GCV 分数、控制输出和调用 receipt 都在读取 p22 真值前封存。

## 关键数字

| 方法 | 严格单元 | 完整组 | 逻辑在线账 | 判定 |
|---|---:|---:|---:|---|
| 经验协方差 GCV + K1 | 2626/2626 | 26/26 | 2A+1A^T | 通过 |
| identity-GCV + K1 | 2626/2626 | 26/26 | 2A+1A^T | 通过，阻断协方差专属解释 |
| 未正则 full-DCT K1 | 2623/2626 | 25/26 | 2A+1A^T | 五相机仍有 3 个单元失败 |
| v197 full-DCT K2 reference | 2626/2626 | 26/26 | 3A+2A^T | 通过 |

五相机臂中，经验协方差候选的 field / gradient / observation p90 为 **0.356362 / 0.587547 / 0.127540**；identity-GCV 为 **0.343619 / 0.569120 / 0.126901**。九相机臂两者也都通过。

identity-GCV 在全部 2626 个单元都选择 `tau=2^-8`。所以这次开发结果甚至没有证明“逐观测自适应”是必要的；它更直接地指出简单 identity-prior 正则值得在下一份结果前合同中被单独检验。

## 为什么可信

独立第二实现重新组装射线、向量/标量算子、势积分、完整 DCT、`gesvd` GCV、控制、物理 CGLS、逐单元门和完整组尾部，完成后才读取正式数组比较。独立状态为 `PASS_INDEPENDENT_RECOMPUTATION_COVARIANCE_GCV_FULL_DCT_V198`。

指标最大绝对差约 **1.01e-11**，坐标块相对差约 **6.63e-11**，汇总最大差约 **7.48e-12**；相机换序后的坐标与 GCV 相对差约 **1.82e-14 / 2.13e-14**。正式树和输入树在验证前后均未改变。独立实现仍共享冻结的数值内核和同一原始输入，因此 `end_to_end_physics_independence_proven=false`。

## 成功、失败与下一步

**成功：** 找到一个只读部署可见观测、在完整 p22 轨迹上达到绝对精度门、并比 K2 reference 少一次 A 和一次 A^T 的简单正则化方向。

**失败：** 经验协方差没有获得相对于同价 identity-GCV 的专属优势，因此该协方差路线关闭。当前也没有 wall/RSS、p14、外部数据或真实 BOST 证据。

下一步不能继续调整经验协方差。应把 p22 的开发发现蒸馏为一个固定 identity-prior 正则候选，在读取 p14 前冻结一次性验证合同，并保留 K1 父方法、v197 K2 reference、便宜控制、完整调用账和独立第二实现。

# v198: the candidate passes, but a simpler equal-cost control also passes

## Bottom line

The empirical-covariance GCV candidate passes all **2626/2626** strict cells and all **26/26** complete calibration groups on the already-opened complete p22 development trajectory, with a logical online ledger of **2A+1AT**. The preregistered identity-GCV control reaches the same **2626/2626 and 26/26** at the same call cost.

The sealed decision is therefore `PASS_CHEAPER_CONTROL_EXPLAINS_COVARIANCE_GCV_V198`: the result supports full-DCT coordinates plus simple regularization as a repair for the K1 tail, but it does not support empirical covariance as a necessary ingredient. `algorithm_breakthrough=false`.

## What was run

The candidate retains every per-camera DCT observation coordinate instead of selecting columns or sketching. It constructs an empirical coordinate covariance from previously sealed fit-field singular values and uses only the current observation and reported geometry to select regularization strength by GCV over a frozen ten-value grid. One unchanged physical CGLS K1 step follows.

Equal-cost identity-GCV, unregularized full-DCT K1, the v197 full-DCT K2 reference, Zero, BP/CGLS, PCGLS, and the historical dual-ridge control were also run. Prediction coordinates, GCV scores, control outputs, and call receipts were sealed before p22 truth was read.

## Key numbers

| Method | Strict cells | Complete groups | Logical online ledger | Verdict |
|---|---:|---:|---:|---|
| Empirical-covariance GCV + K1 | 2626/2626 | 26/26 | 2A+1AT | Pass |
| Identity-GCV + K1 | 2626/2626 | 26/26 | 2A+1AT | Pass; blocks covariance-specific attribution |
| Unregularized full-DCT K1 | 2623/2626 | 25/26 | 2A+1AT | Three five-camera cells remain unsafe |
| v197 full-DCT K2 reference | 2626/2626 | 26/26 | 3A+2AT | Pass |

Under five cameras, field / gradient / observation p90 values are **0.356362 / 0.587547 / 0.127540** for the empirical-covariance candidate and **0.343619 / 0.569120 / 0.126901** for identity-GCV. Both also pass under all nine cameras.

Identity-GCV selects `tau=2^-8` for all 2626 cells. This development result therefore does not establish that per-observation adaptation is necessary. More directly, it identifies simple identity-prior regularization as the mechanism worth testing under the next pre-result contract.

## Independent recomputation

The independent implementation rebuilds rays, vector/scalar operators, potential integration, full DCT features, `gesvd` GCV, controls, physical CGLS, per-cell gates, and complete-group tails before reading formal arrays. Its status is `PASS_INDEPENDENT_RECOMPUTATION_COVARIANCE_GCV_FULL_DCT_V198`.

The maximum absolute metric difference is about **1.01e-11**, the coordinate-block relative difference is about **6.63e-11**, and the maximum summary difference is about **7.48e-12**. Camera-reordering coordinate and GCV relative differences are about **1.82e-14 / 2.13e-14**. Formal and input trees remain unchanged. Frozen numerical kernels and raw inputs are shared, so `end_to_end_physics_independence_proven=false`.

## What succeeded, what failed, and what follows

**Succeeded:** a deployment-visible regularized full-DCT direction reaches the complete p22 absolute-accuracy gate while using one fewer A and one fewer A^T than the K2 reference.

**Failed:** empirical covariance has no specific advantage over equal-cost identity-GCV, so the covariance route closes. There is still no wall/RSS, p14, external-data, or real-BOST result.

The next step is not to tune covariance. It is to distill the p22 development finding into one fixed identity-prior regularized candidate and freeze a one-shot validation contract before reading p14, retaining the K1 parent, v197 K2 reference, cheap controls, complete call accounting, and an independent second implementation.
