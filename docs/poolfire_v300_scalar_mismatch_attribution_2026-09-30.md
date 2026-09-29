# v300: Scalar-Gain Attribution of the Forward Mismatch / 前向不匹配的比例归因

**Date / 日期:** 2026-09-30

**Evidence grade / 证据等级:** independent post-open descriptive recomputation / 独立复算的已开封描述性诊断

## 中文

### 问题与设置

V298–V299 已显示：在五条已开封 PoolFire 轨迹的 25 行上，连续射线生成的观测与冻结离散算子投影有明显差异；步长从 2 cm 减半到 1 cm 后，连续积分自身已收敛，差异没有随之消失。V300 只检验这个差异能否由一个闭式最小二乘比例解释。对每行的真值场计算冻结离散投影 `g=A x_truth`，在相同有效射线上拟合 `y≈αg`；另按两个位移分量分别汇总拟合。没有训练模型、改算子、重跑求解器或打开新数据。

### 结果

- 两个分量合并的最佳比例为 `1.194445`；归一化残差由 `0.255085` 降至 `0.199042`，约减少 `21.97%`，但仍明显非零。
- 分量 0 的最佳比例为 `1.060664`，残差由 `0.192022` 降至 `0.183607`，减少 `4.38%`。
- 分量 1 的最佳比例为 `1.262022`，残差由 `0.274101` 降至 `0.182942`，减少 `33.26%`。
- SciPy/NumPy formal 与独立 `math.fsum` 复算最大绝对差 `2.31×10⁻¹⁴`；相关封存输入未变。

### 解释与边界

单一整体增益不能解释主要差异；即使允许两个分量各自使用事后最佳比例，仍有约 `0.183` 的归一化残差。分量增益差异提示尺度/分量定义值得核对，剩余残差则与空间插值、离散梯度、射线几何或 forward model 差异相容。由于比例使用已开封真值作事后拟合，这不是物理校准值、因果证明、可部署修正或新算法。它不改变 V284 的完整轨迹严格成本失败，也没有证明 matched-accuracy 算子节省、端到端速度、内存收益、外部泛化或真实 BOST。

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.

## English

### Question and setup

V298–V299 found a visible difference between continuous-ray observations and the frozen discrete projection on 25 rows from five already-open PoolFire trajectories. Halving the ray step from 2 cm to 1 cm had already shown that the continuous integral itself was converged, without removing the discrepancy. V300 asks whether one closed-form least-squares gain explains it. For each row, the frozen discrete projection is `g=A x_truth`; `y≈αg` is fit over the same valid rays, then summarized separately for the two displacement components. No model is trained, operator or solver is changed, or new data opened.

### Results

- The pooled two-component gain is `1.194445`; normalized residual falls from `0.255085` to `0.199042`, a `21.97%` reduction, but remains substantial.
- Component 0 gain is `1.060664`; residual falls from `0.192022` to `0.183607` (`4.38%` reduction).
- Component 1 gain is `1.262022`; residual falls from `0.274101` to `0.182942` (`33.26%` reduction).
- The SciPy/NumPy formal and independent `math.fsum` reductions differ by at most `2.31e-14`; sealed inputs are unchanged.

### Interpretation and limits

A single global gain does not explain the discrepancy. Even separate post-hoc gains for the two components leave normalized residuals near `0.183`. The unequal gains motivate checking component scaling/definitions; the remaining residual is compatible with spatial interpolation, discrete gradients, ray geometry, or other forward-model differences. Because these gains are fitted retrospectively using opened truth, they are not physical calibration constants, causal evidence, deployable corrections, or a new algorithm. V284's complete-trajectory strict-cost failure is unchanged. No matched-accuracy call savings, end-to-end speed, memory benefit, external generalization, or real BOST result is established.

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.
