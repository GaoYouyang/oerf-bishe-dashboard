# v133: Per-camera spectral representation narrows the gap but still fails strict capacity

> 中文标题：**v133：逐相机频谱表示显著缩小缺口，但严格容量仍失败**  
> Updated / 更新日期：2026-08-10

## One-sentence verdict / 一句话结论

**English.** A truth-aware four-band, two-component DCT mixture raises strict cellwise passes from v132's `61/3700` to `2353/3700`, while field, full-gradient, and interior-gradient ratios pass in all `3700/3700` cells. Yet `1347` cells fail observation only and no complete trajectory passes the frozen tail gate. The representation is promising but not sufficient under the preregistered equal-weight surrogate; no coefficient predictor is authorized.

**中文。** 真值知晓的四频带、双分量 DCT 混合把严格逐单元通过数从 v132 的 `61/3700` 提高到 `2353/3700`；field、完整梯度和内部梯度均为 `3700/3700` 通过。但仍有 `1347` 个单元只在 observation 上失败，五条完整轨迹没有一条通过冻结尾部门。因此表示显著接近目标，却仍不足以授权系数预测器。

## 1. Why this experiment matters / 为什么做这一步

v132 already showed that one scalar per camera cannot repair the learned correction dual. v133 therefore changes detector-space shape rather than merely camera-level amplitude. For each active camera and each of the two displacement components, a fixed orthonormal `16x16` DCT-II is divided into four exact bands: low-low, low-high, high-low, and high-high. This produces eight directions per camera, or `40/56/72/96` coefficients for `5/7/9/12` cameras.

v132 已经排除了“每台相机只差一个总增益”。v133 不再只调相机幅值，而是允许 correction dual 的 detector-space 频率形状发生变化：每台相机的两个位移分量分别做固定正交 `16x16` DCT-II，再精确划分为 LL、LH、HL、HH 四个频带。每台相机因此有八个方向，`5/7/9/12` 台相机分别形成 `40/56/72/96` 维表示。

This family strictly contains v132 scalar mixing, because equal coefficients over the eight directions reconstruct one per-camera scalar. DCT components themselves are established tools and are not claimed as novel. The question here is narrower: does this variable-camera, proposal-adaptive spectral span have enough capacity in the frozen `2A+2A^T` warm-start shell?

该表示严格包含 v132 的标量混合，因为八个频带方向取同一系数就能恢复逐相机标量。DCT 本身是成熟工具，不主张组件级首创；这里检验的是它放入变机位 proposal-adaptive、`2A+2A^T` warm-start 壳后是否具有足够容量。

## 2. Frozen comparison / 冻结比较

The truth-aware oracle minimizes an equal-weight normalized mismatch in field-lift and projected-lift space. Truth is used only for this already-opened offline capacity test. A deployment-visible spectral least-squares control fits the K1 residual in detector space and uses no truth. Both candidates then use the same observable line search and unchanged CGLS K1 refinement.

真值知晓 oracle 以等权方式最小化 field-lift 和 projected-lift 的归一化误差；真值只用于已开封数据上的离线容量检查。便宜控制则完全不看真值，只在 detector space 里用频带最小二乘拟合 K1 residual。两者随后都进入同一个可观测线搜索与未修改 CGLS K1。

| Arm / 方法 | Exact candidate budget | Strict cells | Complete trajectories |
|---|---:|---:|---:|
| v132 per-camera scalar oracle | `2A+2A^T` | `61/3700` | `0/5` |
| v133 spectral truth-aware oracle | `2A+2A^T` | `2353/3700` | `0/5` |
| v133 observation-only spectral LS | `2A+2A^T` | `0/3700` | `0/5` |
| Zero-CGLS K4 reference | `4A+4A^T` | reference | reference |

## 3. What improved, and what still fails / 改善了什么，哪里仍失败

The most important diagnostic is metric-specific:

| Cellwise ratio gate `<=1.05` | Passed cells |
|---|---:|
| Field | `3700/3700` |
| Full gradient | `3700/3700` |
| Interior gradient | `3700/3700` |
| Observation | `2353/3700` |

All `1347` failed cells are **observation-only failures**. There is no field or gradient failure. This localizes the remaining problem far more sharply than v132: the spectral span can reconstruct a K4-compatible volumetric correction under the three physical-field metrics, but the preregistered equal-weight surrogate does not keep every projected observation inside the strict gate.

全部 `1347` 个失败都是 **只失 observation**，没有任何 field 或 gradient 失败。这比 v132 更精确地定位了问题：当前频谱 span 已能在三类体场指标上形成 K4-compatible correction，但预注册的等权 surrogate 没有让所有投影视图都守住严格门。

| Trajectory | Strict cells | Field p90 / worst | Full-gradient p90 / worst | Interior-gradient p90 / worst | Observation p90 / worst |
|---|---:|---:|---:|---:|---:|
| `14 kW, size 05` | `687/740` | 1.0107 / 1.0184 | 1.0064 / 1.0256 | 1.0036 / 1.0144 | 1.0452 / 1.1019 |
| `22 kW, size 03` | `564/740` | 1.0177 / 1.0293 | 1.0292 / 1.0437 | 1.0300 / 1.0434 | 1.0689 / 1.1077 |
| `33 kW, size 01` | `562/740` | 1.0089 / 1.0132 | 0.9973 / 1.0180 | 1.0020 / 1.0054 | 1.0725 / 1.1261 |
| `45 kW, size 05` | `188/740` | 1.0172 / 1.0254 | 1.0059 / 1.0135 | 1.0071 / 1.0140 | 1.1484 / 1.1936 |
| `58 kW, size 03` | `352/740` | 1.0164 / 1.0215 | 1.0193 / 1.0341 | 1.0278 / 1.0492 | 1.2033 / 1.3868 |

More cameras help but do not eliminate the tail. Strict pass counts are `272/925`, `553/925`, `744/925`, and `784/925` for 5, 7, 9, and 12 cameras. The 12-camera observation median falls below the K4 reference (`0.9991`), yet its p90 and worst remain `1.0653` and `1.1071`. A favorable average is therefore not sufficient.

相机更多时通过率明显提高，但尾部仍没有消失。`5/7/9/12` 台相机分别通过 `272/925`、`553/925`、`744/925`、`784/925`。12 相机 observation 中位数已经优于 K4（`0.9991`），但 p90 与 worst 仍为 `1.0653/1.1071`；漂亮均值不能替代严格尾部门。

## 4. Independent recomputation / 独立复算

A second implementation independently rebuilds the DCT bands, per-camera lifts, spectral systems, cheap control, K1 shell, metrics, summaries, and gates. It does not import the formal spectral-basis or oracle-solver helpers.

第二个程序独立重建 DCT 频带、逐相机 lift、频谱方程、便宜控制、K1 物理壳、指标、汇总和判决，不导入正式频谱基或 oracle 求解 helper。

| Independent check / 独立检查 | Maximum difference / 最大差 |
|---|---:|
| Oracle coefficients | `1.27e-12` |
| Diagnostics | `1.20e-10` |
| Oracle metrics | `9.99e-16` |
| Spectral-LS coefficients / metrics | `0 / 0` |
| Complete summaries | `6.66e-16` |

The independent status is `PASS_INDEPENDENT_RECOMPUTATION_DETECTOR_SPECTRAL_CAPACITY_V133`. Candidate call receipts have zero failures, sealed inputs remain unchanged, and validation/test truth remains unread. Both paths still share the frozen physics kernels, so end-to-end physics independence is not proven.

## 5. Decision / 决策

The exact scientific conclusion is **not** that the DCT span is mathematically incapable. It is that no passing candidate was found under the preregistered equal-weight field/projected surrogate. Because every physical-field metric already passes and only observation fails, immediately enlarging or training a network would confound representation capacity with objective design.

准确结论不是“DCT span 数学上不可能”，而是“在预注册的等权 field/projected surrogate 下没有找到严格通过候选”。由于三类体场指标已全部通过、只剩 observation 失败，立即扩大或训练网络会把表示容量与目标函数设计混在一起。

The next experiment is therefore a result-prior frozen, projection-prioritized Pareto feasibility diagnostic in the same span. It will ask whether reweighting or lexicographically prioritizing observation can preserve all field/gradient gates while closing the observation tail. Only if all `3700/3700` cells pass would a minimal observation/geometry-only coefficient predictor become eligible.

所以下一门是在同一 span 内冻结 projection-prioritized Pareto 可行性诊断：检查提高投影误差优先级后，能否在不破坏 field/gradient 门的前提下修复 observation 尾部。只有达到 `3700/3700`，最小 observation/geometry-only 系数预测器才可能获准。

## Evidence boundary / 证据边界

- `strict_representation_capacity_passed=false`
- `objective_weight_mismatch_ruled_out=false`
- `coefficient_predictor_authorized=false`
- `algorithm_breakthrough=false`
- `resource_speedup=false`
- `external_generalization=false`
- `curved_ray_validated=false`
- `real_bost=false`
- `paper_success=false`

This is an independently recomputed post-open capacity diagnostic on synthetic PoolFire data. It is not a deployable algorithm, not an external-generalization result, and not a laboratory BOST result.

这是已开封 PoolFire 合成数据上的独立复算容量诊断，不是可部署算法、外部泛化结果或组内真实 BOST 结果。
