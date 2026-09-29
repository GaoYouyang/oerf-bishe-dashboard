# v301: Camera-Wise Residual Heterogeneity / 相机间残差异质性

**Date / 日期:** 2026-09-30  
**Evidence grade / 证据等级:** independent post-open descriptive recomputation / 独立复算的已开封描述性诊断

## 中文

### 问题与设置

V300 表明单一或分量级全局比例不能消除连续射线观测与冻结离散投影之间的差异。V301 只检查剩余差异是否明显集中于个别视角。分析限定在相同五条已开封 PoolFire 轨迹、25 个样本、9 个视角；沿用既有有效射线掩码，逐视角统计残差比，不拟合相机增益、不训练、不运行求解器，也未打开新数据。

### 结果

- 汇总的相机残差比变异系数约 `0.05528`（5.5%）。
- 各视角 p50 的最大/中位比约 `1.134`，p90 的最大/中位比约 `1.196`，worst 的最大/中位比约 `1.255`。
- 各相机的典型残差 p50 约落在 `0.217–0.256`；存在温和视角差异，但没有一个视角单独支配整体差异。
- formal 与独立实现的汇总最大绝对差为 `1.17×10⁻¹⁴`；上游封存输入保持不变。

### 解释与边界

这削弱了“单一异常相机是主要来源”的解释，但不能识别物理根因；相机、分量、射线有效区域、空间离散和密度到光学量的映射仍可能共同作用。该分析是已开封样本上的描述性诊断，不支持拟合逐相机增益，也不是新的 forward、重建结果、部署规则或算法。它没有建立 matched-accuracy 调用节省、wall/RSS、外部泛化或真实 BOST 证据；既有严格成本判决不变。

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.

## English

### Question and setup

V300 found that a pooled or component-wise global gain does not remove the discrepancy between continuous-ray observations and the frozen discrete projection. V301 asks only whether the remaining residual is strongly concentrated in individual views. It uses the same 25 samples from five already-open PoolFire trajectories and nine views, with the existing valid-ray masks. It summarizes residual ratios per view; it fits no camera gains, trains no model, runs no solver, and opens no new data.

### Results

- The pooled coefficient of variation across camera residual ratios is about `0.05528` (5.5%).
- The maximum-to-median ratio across views is about `1.134` for p50, `1.196` for p90, and `1.255` for worst-case residual ratios.
- Typical per-view residual p50 values lie around `0.217–0.256`: view differences are modest, with no single view dominating the discrepancy.
- The formal and independent summary implementations differ by at most `1.17×10⁻¹⁴`; upstream sealed inputs remain unchanged.

### Interpretation and limits

This weakens the explanation that one anomalous camera is the main source, but it does not identify the physical cause. Camera geometry, displacement components, valid-ray coverage, spatial discretization, and the density-to-optics mapping may all contribute. This is descriptive attribution on opened samples; it does not support fitting per-camera gains and is not a new forward model, reconstruction result, deployment rule, or algorithm. No matched-accuracy call savings, wall/RSS benefit, external generalization, or real BOST result is established; the existing strict-cost decision is unchanged.

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.
