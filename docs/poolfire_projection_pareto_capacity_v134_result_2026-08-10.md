# v134: Projection-prioritized Pareto tuning narrows the gap but does not close it

> 中文标题：**v134：优先投影误差能继续缩小缺口，但固定全局频谱表示仍不够**  
> Updated / 更新日期：2026-08-10

## One-sentence verdict / 一句话结论

**English.** On the same `3700` opened PoolFire cells, the same per-camera four-band DCT span, and the same `2A+2A^T` candidate shell, a preregistered finite Pareto roster raises all-four-metric passes from v133's `2353/3700` to `2591/3700`. Yet complete trajectories remain `0/5`; all `1109` failures are observation-only. Simple objective reweighting is therefore not sufficient, and no predictor or GPU training is authorized.

**中文。** 在同一 `3700` 个已开封 PoolFire 单元、同一逐相机四频带 DCT span 和同一 `2A+2A^T` 候选壳中，结果前冻结的有限 Pareto 候选把四指标严格通过从 v133 的 `2353/3700` 提高到 `2591/3700`。但完整轨迹仍为 `0/5`，其余 `1109` 个失败全部只在 observation。简单调整目标权重不足以解决问题，因此不授权预测器或 GPU 训练。

## 1. Question / 研究问题

v133 had already shown that the global spectral representation can preserve field, full-gradient, and interior-gradient quality in every cell, while observation tails remain. One ambiguity remained: was the representation too small, or did the equal-weight field/projected surrogate simply choose the wrong point inside an adequate span?

v133 已经证明当前全局频谱表示能让每个单元的 field、完整梯度和内部梯度过门，只剩 observation 尾部。唯一尚未排除的歧义是：表示本身不够，还是等权 field/projected surrogate 在一个本来够用的 span 里选错了点？

v134 isolates that question without changing data, physics, representation, or exact-call budget. It is a truth-aware, post-open **capacity diagnostic**, not a deployable method.

v134 在不改变数据、物理、表示和精确调用账的前提下单独检验这个问题。它是真值知晓、已开封数据上的**容量诊断**，不是可部署方法。

## 2. Frozen finite Pareto roster / 冻结的有限 Pareto 候选

The roster contains projection weights `1, 4, 16, 64, 256, 1024` plus a projection-only endpoint. Every candidate is solved in the same v133 DCT span. Selection is lexicographic:

1. If any candidate passes all four per-cell gates, choose the one with the smallest worst normalized metric and then the smallest observation ratio.
2. Otherwise, among candidates preserving field, full-gradient, and interior-gradient, choose the smallest observation ratio.
3. Only if no physically feasible candidate exists, choose the smallest physical-field minimax value.

候选包括投影权重 `1、4、16、64、256、1024` 以及 projection-only 端点，全部在同一个 v133 DCT span 中求解。选择顺序是：先找四指标全过门的点；若没有，则在三类场与梯度仍过门的点中选择 observation 最小者；只有连物理场可行点都不存在时才回退到物理指标 minimax。

The truth-aware selector is used only to ask whether the finite roster contains a feasible point. The online candidate shell remains `2A+2A^T`; Zero-CGLS K4 remains `4A+4A^T`.

真值只用于询问有限候选中是否存在可行点。候选在线壳仍是 `2A+2A^T`，Zero-CGLS K4 仍是 `4A+4A^T`。

## 3. Result / 结果

| Diagnostic / 诊断 | v133 | v134 |
|---|---:|---:|
| Cells passing all four metrics | `2353/3700` | `2591/3700` |
| Complete trajectories | `0/5` | `0/5` |
| Observation-only failures | `1347` | `1109` |
| Any field or gradient failure | `0` | `0` |

v134 adds `238` passing cells, or `6.43` percentage points of the full sample set. This is a real improvement, but not the required result: no complete trajectory passes the frozen p90/worst gate.

v134 多通过 `238` 个单元，占全部样本的 `6.43` 个百分点。这是真实改善，但不是所需结果：五条轨迹没有一条通过冻结的 p90/worst 门。

The projection-only endpoint is selected in `2718/3700` cells and by itself passes `2564/3700`. The entire finite Pareto roster adds only `27` passes beyond that endpoint. This is strong evidence that merely increasing the observation weight has nearly saturated within the fixed global DCT4x2 representation.

projection-only 端点在 `2718/3700` 个单元中被选中，它单独就通过 `2564/3700`；全部有限 Pareto 候选只在此基础上再增加 `27` 个通过。这说明在固定全局 DCT4x2 表示内，继续提高 observation 权重已经接近饱和。

## 4. Where the failures live / 失败集中在哪里

| Cameras | v133 strict cells | v134 strict cells | Total |
|---:|---:|---:|---:|
| 5 | 272 | 310 | 925 |
| 7 | 553 | 625 | 925 |
| 9 | 744 | 833 | 925 |
| 12 | 784 | 823 | 925 |

Sparse camera sets remain the clearest geometric difficulty. Five cameras pass only `310/925`, while nine and twelve cameras pass `833/925` and `823/925`. More cameras help, but they do not remove the strict tail.

稀疏相机仍是最明确的几何困难：5 相机只有 `310/925`，9 和 12 相机分别为 `833/925` 与 `823/925`。增加相机有帮助，但不能消除严格尾部。

| Trajectory | Strict cells | Observation p90 / worst |
|---|---:|---:|
| `14 kW, size 05` | `705/740` | `1.0380 / 1.0961` |
| `22 kW, size 03` | `578/740` | `1.0646 / 1.1016` |
| `33 kW, size 01` | `577/740` | `1.0710 / 1.1227` |
| `45 kW, size 05` | `275/740` | `1.1363 / 1.1758` |
| `58 kW, size 03` | `456/740` | `1.2029 / 1.3593` |

The p45 and p58 morphologies dominate the remaining tail. By contrast, clean/combined/intrinsics/noise/pose/rotation/translation groups have similar observation p90 values, roughly `1.09-1.10`, and medium versus stress pass rates are nearly identical. The evidence therefore points more strongly to localized morphology and camera coverage than to a single global perturbation severity.

p45 与 p58 形态主导剩余尾部。相反，clean、combined、intrinsics、noise、pose、rotation、translation 各组的 observation p90 都在约 `1.09-1.10`，medium 与 stress 的通过率也几乎相同。因此证据更指向局部形态与相机覆盖，而不是某一种全局扰动强度。

## 5. Independent recomputation / 独立复算

A second program independently rebuilds the finite roster, weighted systems, physical replay, selector, metrics, and summaries. It also confirms that weight `1` reproduces v133 within numerical tolerance.

第二个程序独立重建有限候选、加权方程、物理重放、选择器、指标和汇总，并确认权重 `1` 在数值容差内复现 v133。

| Check / 检查 | Maximum difference / 最大差 |
|---|---:|
| Candidate coefficients | `1.90e-12` |
| Candidate metrics | `2.23e-15` |
| Selected coefficients | `1.26e-12` |
| Selected metrics | `2.23e-15` |
| Diagnostics | `2.57e-10` |
| Complete summary | `1.34e-15` |
| Exact-array failures | `0` |

The independent status is `PASS_INDEPENDENT_RECOMPUTATION_PROJECTION_PARETO_V134`. Formal inputs remain unchanged, and validation/test truth remains unread. Both implementations still share the frozen physics kernels, so end-to-end physics independence is not proven.

## 6. Scientific decision / 科学决策

The exact conclusion is not that the continuous DCT span is mathematically impossible. It is that the preregistered finite objective roster does not contain a candidate passing all `3700/3700` cells, and projection-only already captures almost all of the available gain. Simple objective-weight mismatch is therefore not a sufficient explanation.

准确结论不是“连续 DCT span 在数学上不可能”，而是“结果前冻结的有限目标候选中没有一个能让 `3700/3700` 全部通过，且 projection-only 已经吸收几乎全部收益”。因此，简单的目标权重失配不足以解释剩余问题。

The next experiment will not train a larger network. It will first test a small deterministic **local space-frequency representation** that strictly contains v133, preserves variable camera count and permutation invariance, and can express morphology-dependent detector corrections. Only `3700/3700` capacity plus `5/5` complete trajectories can authorize a minimal observation/geometry-only predictor.

下一步不训练更大的网络，而是先检验一个严格包含 v133 的小型确定性**局部空间-频率表示**：保持相机数量可变与排列不变性，同时能够表达随局部形态变化的 detector correction。只有容量达到 `3700/3700` 且完整轨迹 `5/5`，才允许训练最小 observation/geometry-only 预测器。

## Evidence boundary / 证据边界

- `finite_objective_roster_passed=false`
- `continuous_span_impossibility_proven=false`
- `objective_weight_mismatch_sufficient=false`
- `local_space_frequency_hypothesis_proven=false`
- `minimal_predictor_authorized=false`
- `algorithm_breakthrough=false`
- `resource_speedup=false`
- `external_generalization=false`
- `curved_ray_validated=false`
- `real_bost=false`
- `paper_success=false`

This is a post-open truth-aware capacity diagnostic on synthetic PoolFire data. It is not a deployable algorithm, a speed result, an external-generalization result, or a laboratory BOST result.

这是已开封 PoolFire 合成数据上的真值知晓容量诊断，不是可部署算法、速度结果、外部泛化结果或组内真实 BOST 结果。
