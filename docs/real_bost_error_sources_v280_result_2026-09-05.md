# v280：误差来源已独立分清，尚不是算法改善

2026-09-05。1,404 个已开封虚拟单元完成独立误差分解。clean 场误差中逆放大项的有符号贡献中位数为 82.27%–84.10%；pose 中几何项为 81.79%–88.41%。这是归因，不是重建改善。

## 数据与判决

同一 9 个三维场、13 套九相机几何、4 个时间与 clean/pose/combined 条件，共 1,404 单元、5,616 个单元-误差空间记录。DCT1024 去掉常量后保留 1,023 个系数。GELSD 正式求解与独立显式 SVD、独立物理算子和差分实现完成 15/15 有效性检查；场/观测状态最大归一化差 1.44e-10，逐行与汇总最大差均约 8.17e-10，低于冻结 1e-8。输入与结果树保持不变。

权威状态为 `PASS_INDEPENDENT_LINEAR_SOURCE_BUDGET_V280`，只表示归因有效。v279 的跨实现差异仍使其保持不确定，不能借此补成通过。

## 场误差的有符号贡献中位数

| 条件 / Condition | t | 遗漏 / Omitted | 逆放大 / Alias | 几何 / Geometry | 噪声 / Noise |
|---|---:|---:|---:|---:|---:|
| clean | 0 | 0.158978 | 0.841022 | -0.000000 | 0.000000 |
| clean | 0.25 | 0.177295 | 0.822705 | -0.000000 | 0.000000 |
| clean | 0.75 | 0.173021 | 0.826979 | -0.000000 | 0.000000 |
| clean | 1 | 0.167801 | 0.832199 | -0.000000 | 0.000000 |
| pose | 0 | 0.030698 | 0.151797 | 0.817924 | 0.000000 |
| pose | 0.25 | 0.031758 | 0.129429 | 0.829717 | 0.000000 |
| pose | 0.75 | 0.028380 | 0.134223 | 0.848435 | 0.000000 |
| pose | 1 | 0.028573 | 0.077289 | 0.884050 | 0.000000 |
| combined | 0 | 0.031412 | 0.163989 | 0.806449 | 0.000099 |
| combined | 0.25 | 0.037722 | 0.184331 | 0.781292 | 0.000943 |
| combined | 0.75 | 0.036982 | 0.171907 | 0.789629 | 0.000349 |
| combined | 1 | 0.028156 | 0.147550 | 0.824236 | 0.000476 |

每个数是对 117 个模型-相机组单元分别计算后取中位数；仅逐单元四项相加为 1，表格各列中位数不保证相加为 1。负值保留，表示抵消。这里没有定义或事后选择“主因通过门”。完整梯度、内部梯度和真实几何观测空间的全部分层见脱敏 JSON。

## 为什么影响下一步

令 B 为正交降维基，c0=B^T x，h=x-Bc0，M=A_reported B。满列秩时，最小二乘场误差精确分为：

`B M+ y - x = -h + B M+ A_true h + B M+ (A_true-A_reported)Bc0 + B M+ noise`。

clean 时，逆放大项在场误差中的贡献较大，而它在观测残差方向上的投影约为零。这与最小二乘残差正交性一致：更贴近观测不保证更贴近真实三维场。pose 条件的几何失配贡献更大。combined 的噪声场范数相对总误差范数的中位数为 0.01233–0.01397，但这仅针对固定虚拟噪声，不能说真实实验噪声不重要。pose 与 combined 没有配对同一几何扰动，不能用两组差值估计单独加噪的因果影响。

下一步先审计观测侧模型误差处理，而不是继续增加迭代深度或网络。近邻文献已有模型误差均值/协方差补偿；它应作为经典对照，不是本项目首创。相关公开摘要与算法说明来自 [Kolehmainen 等，2009](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-26-10-2257)。受限全文未读，光扩散层析结果不能当作 BOS 证据。

所有统计只用已开封数据作诊断。未来估计误差统计必须按完整模型留出，不能把留出模型的真值或时间帧泄漏到均值、协方差、归一化、回退和停止中。

## 成本与边界

本轮全部是离线诊断：468 次观测生成 forward，53,196 次基构造 forward-equivalent。没有在线候选成本、fresh wall/RSS、学习、外部泛化或真实 BOST 成功。当前观测是归一化密度梯度射线积分，不是配对实测像素位移。

`algorithm_breakthrough=false; paper_success=false; resource_speedup=false; external_generalization=false; real_bost=false; predictor_training_authorized=false`。

# v280: independently resolved error sources, not an algorithm improvement

Independent error decomposition covers 1,404 opened virtual cells. Median signed contributions to field error are 82.27%–84.10% for inverse aliasing in clean data and 81.79%–88.41% for geometry mismatch under pose error. This is attribution, not improved reconstruction.

## Data and validity

The same nine fields, thirteen nine-camera geometries, four times and three conditions yield 1,404 cells and 5,616 cell-space records. The unchanged DCT1024 basis excludes its constant mode, leaving 1,023 coefficients. Formal GELSD and independently rebuilt physics, explicit SVD and derivative stencils pass all 15 validity checks. Maximum normalized field/observation state difference is 1.44e-10; row and summary differences are about 8.17e-10, below the frozen 1e-8 gate. Input and output trees are unchanged.

`PASS_INDEPENDENT_LINEAR_SOURCE_BUDGET_V280` validates attribution only. v279 remains inconclusive and is not retrospectively repaired.

## Reading the table and identity

The bilingual table above reports median signed field-error contributions, each over 117 model-rig cells. Only individual-cell contributions sum to one; column medians need not. Negative values represent cancellation. No dominant-source threshold was selected after results. All full-gradient, interior-gradient and true-geometry observation summaries are available in the redacted JSON.

For orthonormal B, c0=B^T x, h=x-Bc0 and full-column-rank M=A_reported B, the displayed identity splits the least-squares error into omitted structure, inverse-amplified aliasing, geometry mismatch and noise. In clean data, aliasing contributes substantially to field error but has approximately zero signed contribution along the observation residual, consistent with least-squares orthogonality. Better data fitting need not mean a more accurate volume. Geometry mismatch contributes more under pose error. Combined-condition noise norm-ratio medians are 0.01233–0.01397 under this fixed synthetic distribution, not a claim that experimental noise is negligible. Pose and combined use unpaired geometry draws, so their difference is not a causal noise-only contrast.

## Consequence and limitations

Audit observation-side model discrepancy before increasing solver depth or network size. Prior work already compensates model error using estimated means and covariances: this is a required classical comparator, not our invention. The public abstract and algorithm description of [Kolehmainen et al. 2009](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-26-10-2257) were read, not the restricted full article. Diffuse optical tomography results do not establish BOS performance.

Any subsequent statistical estimator requires complete-model holdouts and training-only discrepancy statistics; held-out truth and frames cannot inform means, covariances, normalization, fallback or stopping. This post-open diagnostic establishes no predictive generalization.

All work is offline: 468 observation-generation forwards and 53,196 basis-setup forward equivalents. There is no online saving, fresh wall/RSS, learned or external result. Observations remain normalized density-gradient ray integrals rather than paired measured pixel displacement. All six claim flags above remain false.
