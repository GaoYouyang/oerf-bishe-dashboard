# v281：跨模型误差补偿未通过精度门

2026-09-05。九个完整模型轮流留出，在一套预选九相机几何上得到 108 个虚拟单元。协方差补偿、不补偿和只减均值均为 0/12 分层通过。部分场误差下降，但梯度尾部仍失败。

## 做了什么

沿用已有九个三维场、四个时间和 clean/pose/combined 条件，只选结果前按身份固定的一套九相机几何。每折排除一个模型的全部四个时间，其他八个模型的 32 个场估计粗表示遗漏导致的观测误差均值与协方差。主方法使用固定解析收缩和加权最小二乘；便宜对照是不补偿和只减均值。没有用留出真值选超参或回退。所有预测先封存，随后才读取留出真值评分。

这是经典 approximation-error 对照，而不是新的神经算子。解析协方差规则来自 [Chen 等的 OAS 公式](https://arxiv.org/pdf/0907.4698)，没有参数搜索；其各向同性部分是模型误差正则，不是测得的实验噪声。训练时间相关、未证明高斯，因此不继承该公式的理想统计最优保证。

## 独立判决

正式 SVD/GELSD 与独立 QR/eigh/SVD 分别重建训练统计、物理算子、预测、直接重放和四指标。16/16 有效性检查通过；最大场/观测状态差 2.05e-13，逐单元指标差 5.02e-13，分层差 2.83e-13。输入和完整输出树不变。

科学判决是 `FAIL_LOMO_DISCREPANCY_SENTINEL_V281`。三种方法全部 0/12 分层通过。下面是主方法的 p90-higher；每层只有九个模型，因此这里 p90 就是最坏样本，不是稳定的总体尾部估计。

| 条件 / Condition | t | Field | Full gradient | Interior gradient | Observation |
|---|---:|---:|---:|---:|---:|
| clean | 0.0 | 0.298756 | 0.859309 | 0.941865 | 0.151293 |
| clean | 0.25 | 0.380346 | 1.087914 | 1.148985 | 0.148331 |
| clean | 0.75 | 0.307779 | 0.865252 | 1.121841 | 0.146490 |
| clean | 1.0 | 0.391794 | 1.109642 | 1.208121 | 0.150506 |
| pose | 0.0 | 1.300038 | 2.351332 | 2.524353 | 0.319146 |
| pose | 0.25 | 1.225650 | 2.134968 | 3.092691 | 0.302645 |
| pose | 0.75 | 1.165914 | 2.290964 | 2.420241 | 0.303508 |
| pose | 1.0 | 1.058048 | 1.983797 | 1.943134 | 0.313287 |
| combined | 0.0 | 1.057850 | 1.892621 | 2.069233 | 0.330708 |
| combined | 0.25 | 1.126238 | 2.364285 | 2.627121 | 0.328118 |
| combined | 0.75 | 1.332588 | 2.799860 | 3.062046 | 0.312542 |
| combined | 1.0 | 1.021103 | 2.016635 | 2.241761 | 0.334918 |

冻结 p90 门依次是 0.50 / 0.75 / 0.75 / 0.20，worst 门为 0.75 / 1.00 / 1.00 / 0.35，任一失败都不能通过。

clean 四层的场误差均低于 0.5，但梯度仍为 0.8593–1.1096，全部高于 0.75。相对不补偿，主方法在 46/108 个单元的场误差改善超过 2%，却也在 26/108 个单元恶化超过 2%，最坏场误差比为 1.4412。相对只减均值，改善/恶化为 51/11 个单元。不能拿局部改善替代稳定精度。

## 为什么有意义，以及不能说什么

v280 识别误差来源，并不保证误差统计能跨模型迁移。现在一次真实的训练折到留出模型测试排除了这套固定补偿作为合格参考的充分性；不是算力不足，也不是复算代码意见不一致。它不否定所有模型误差方法或整条 C 路线。训练只模拟 reported geometry 下的细节遗漏，并未估计未知的真实位姿误差。

每套实现的离线账：36 次生成观测 A、4,092 次几何基 forward-equivalent、864 次训练误差 A、648 次直接审计重放 A，27 次折-条件估计器调用。没有在线提速或 fresh wall/RSS 证据。只计算一套相机，不能写成十三套或可变相机数验证；这些归一化梯度积分也不是实测像素位移。

停止调协方差、迭代深度或网络；下一步先审视有限背景像素观测与合格参考的物理接口。只能在明确合成光学假设和独立审计下继续，不能把模拟当成实测。 不调收缩、秩或门挽救本轮。`algorithm_breakthrough=false; paper_success=false; resource_speedup=false; external_generalization=false; real_bost=false`。

# v281: cross-model discrepancy correction fails the accuracy gate

Leaving out each of nine complete models gives 108 virtual cells on one preselected nine-camera rig. Covariance correction, no correction and mean-only correction all pass 0/12 strata. Some field errors decrease, but gradient tails still fail.

## Experiment and independence

The existing nine 3D sources, four times and clean/pose/combined conditions are tested on one identity-preselected nine-camera rig. Each outer fold excludes all four times of one model; 32 samples from the other eight estimate the mean and covariance of reported-operator fine-minus-coarse discrepancy. The fixed analytic-shrinkage weighted-LS primary is compared with uncorrected and mean-only LS. Held-out truth selects no hyperparameter or fallback; all predictions seal before scoring.

This is a classical approximation-error comparator, not a new neural operator. The analytic rule is the [OAS formula of Chen et al.](https://arxiv.org/pdf/0907.4698), without parameter search. Its isotropic term regularizes model discrepancy, not measured experimental noise. Correlated time samples with unestablished Gaussianity do not inherit the formula's ideal statistical optimality.

Formal SVD/GELSD and independently rebuilt physics plus QR/eigh/SVD agree on train statistics, predictions, direct replay and four metrics. All 16 checks pass. Maximum field/observation-state, cell-metric and summary differences are 2.05e-13, 5.02e-13 and 2.83e-13. Inputs and complete output trees are unchanged. Scientific decision: `FAIL_LOMO_DISCREPANCY_SENTINEL_V281`.

## Results and interpretation

The table above lists the primary's higher-p90 metrics. With nine models in each stratum, higher p90 equals its worst sample, not a reliable population-tail estimate. Frozen p90 limits are 0.50/0.75/0.75/0.20 and worst limits 0.75/1.00/1.00/0.35. All three methods pass 0/12 strata. Clean field p90 stays below 0.5, but all four gradient values, 0.8593–1.1096, exceed 0.75.

Against uncorrected LS, primary field error improves by over 2% in 46/108 cells but worsens by over 2% in 26/108, with maximum ratio 1.4412. Against mean-only LS, improvement/worsening counts are 51/11. Local gains do not establish stable accuracy or noninferiority.

v280 attribution did not guarantee transferable error statistics. This complete-model holdout test closes the fixed correction as a sufficient reference, not all approximation-error methods or the C route. Training models reported-geometry truncation, not unknown actual pose error. The issue is neither insufficient compute nor disagreeing implementations.

Per-implementation offline accounting: 36 observation-generation A, 4,092 basis forward equivalents, 864 training-discrepancy A, 648 direct audit-replay A and 27 fold-condition estimator calls. No online or fresh wall/RSS saving was measured. One rig is not thirteen-rig or variable-cardinality validation; normalized gradient integrals are not measured pixel displacements.

Stop covariance, iteration-depth and network variations. Next examine the physical interface between finite-background pixel observations and an adequate reference, under explicit synthetic optical assumptions and independent audits, never labeling simulations as measurements. No shrinkage, rank or gate tuning will rescue this run. The five claim flags shown above remain false.
