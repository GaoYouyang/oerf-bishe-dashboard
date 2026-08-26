# v224：逐相机删除稳定度仍无法安全分离跨工况伤害

## 结论

v223 已经证明，把 Low-64 风险压成调和可观测性或全局拟合残差这两个一维分数，无法对全部单元建立 fail-closed 回退。v224 不改场、不训练模型，改问一个物理上不同的问题：**如果删掉九个相机中的任意一个，Low-64 对完整九视角观测的预测会不会明显漂移；这种“最坏删相机稳定度”能否事先识别 direct Low-64 PCGLS K11 的不安全单元？**

正式程序和独立第二实现得到相同判决：

`FAIL_LOW64_CAMERA_JACKKNIFE_RISK_OVERLAP_V224`

在 `1261` 个已开封单元中，冻结标签仍为 `1064` 个安全、`197` 个不安全。九个删相机子问题在所有几何中都保持 `64` 阶满秩，因此失败不是因为删掉某台相机后代数系统直接坍塌。

## 做了什么

对每个当前二维观测和 reported geometry，先用全部九个相机拟合 Low-64 系数；随后依次删掉一台相机，用剩余八台重新拟合。唯一主指标取九次重拟合中，对完整九相机预测造成的最大相对变化，方向在结果前固定为“越低越安全”。

便宜 control 不做九次重拟合，只取全量拟合在九台相机上的最大分块残差，同样固定为“越低越安全”。完整观测特征数组先封存，之后才读取真值评分形成安全标签；没有训练、阈值搜索、分位数豁免或异常点剔除。

## 结果

| 可观测风险分数 | 安全区间 | 不安全区间 | 严格分离 margin | 判决 |
|---|---:|---:|---:|---|
| 最坏删相机漂移 `J`，低更安全 | `0.020138 - 0.091682` | `0.037873 - 0.178650` | `-0.053809` | 重叠 |
| 最大逐相机拟合残差 `C`，低更安全 | `0.478493 - 0.736101` | `0.565201 - 0.850216` | `-0.170900` | 重叠 |

主指标确实捕捉到一部分相机冗余和跨视角一致性差异，但安全与不安全区间仍显著重叠。任何一维阈值都会接受至少一个已知不安全单元，或者拒绝至少一个已知安全单元。结果前冻结的门要求对全部 `1261` 个单元严格分开，所以没有阈值、没有回退策略，也没有 exact-call 节省可供评分。

## 独立复算

独立程序使用不同的 Low-64 正交化和正规方程特征分解，自行重建全部九相机与删相机解、两个分数、安全标签、区间和分离门。正式与独立特征最大差为 `5.62e-15`，分离统计最大差为 `1.46e-15`，相机换序最大差为 `7.09e-15`，离散策略差为 `0`。独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_CAMERA_JACKKNIFE_RISK_V224`

共享冻结的底层 physics kernels 仍存在，所以 `end_to_end_physics_independence_proven=false`。这不改变本次标量重叠的判决，但禁止声称完全独立的端到端物理实现。

## 证据边界

- v224 是已开封 Case 2/5 上的 post-open 机制容量诊断，不是部署算法或 fresh 外部门；
- 它只关闭“用单一最坏删相机漂移或单一逐相机残差做 fail-closed 回退”这条路线，不证明所有多视角自一致机制不可能；
- 不允许看到结果后反转方向、调阈值、换相机分组、改归一化或用大模型/GPU 挽救；
- 没有候选物理重放、exact-call 减少、wall/RSS、外部工况、曲线光路或真实 BOST 结果。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v224: Leave-One-Camera-Out Stability Still Cannot Safely Separate Cross-Condition Harm

## Conclusion

v223 shows that compressing Low-64 risk into either harmonic observability or global fit residual cannot establish a fail-closed fallback across every cell. v224 changes no field and trains no model. It asks a physically distinct question: **if any one of the nine cameras is removed, how much does the Low-64 prediction of the full nine-camera observation move, and can this worst-camera stability identify unsafe direct Low-64 PCGLS K11 cells in advance?**

The formal and independent implementations reach the same decision:

`FAIL_LOW64_CAMERA_JACKKNIFE_RISK_OVERLAP_V224`

Among `1,261` opened cells, the frozen labels remain `1,064` safe and `197` unsafe. Every leave-one-camera-out system retains numerical rank `64` for every geometry, so the failure is not an algebraic collapse after deleting a sensor.

## What was done

For each current 2D observation and reported geometry, all nine cameras first fit the Low-64 coefficients. Each camera is then removed in turn and the remaining eight cameras refit the coefficients. The unique primary is the largest relative change that any of the nine refits produces in the full nine-camera prediction, with a preregistered lower-is-safer orientation.

The cheap control performs no jackknife refits. It takes the largest camera-block residual under the single full fit, also with a preregistered lower-is-safer orientation. The complete observable-feature array is sealed before truth scores are opened to construct safety labels. There is no training, threshold search, quantile exception, or outlier removal.

## Results

| Observable risk score | Safe range | Unsafe range | Strict margin | Decision |
|---|---:|---:|---:|---|
| Worst leave-one-camera-out drift `J`, lower is safer | `0.020138 - 0.091682` | `0.037873 - 0.178650` | `-0.053809` | overlap |
| Maximum per-camera fit residual `C`, lower is safer | `0.478493 - 0.736101` | `0.565201 - 0.850216` | `-0.170900` | overlap |

The primary captures some camera-redundancy and cross-view consistency variation, but the safe and unsafe ranges still overlap substantially. Any one-dimensional threshold would accept at least one known unsafe cell or reject at least one known safe cell. Because the frozen gate requires strict separation across all `1,261` cells, no threshold or fallback policy is established and there is no exact-call saving to score.

## Independent recomputation

The independent implementation uses a different Low-64 orthogonalization and normal-matrix eigensystems to rebuild all full and leave-one-camera-out solutions, both scores, safety labels, ranges, and separation gates. Maximum formal-independent feature difference is `5.62e-15`, maximum separation-statistic difference is `1.46e-15`, maximum camera-permutation difference is `7.09e-15`, and the discrete policy difference is `0`. The independent status is:

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_CAMERA_JACKKNIFE_RISK_V224`

Frozen low-level physics kernels remain shared, so `end_to_end_physics_independence_proven=false`. This does not alter the scalar-overlap verdict, but it prevents a claim of fully independent end-to-end physics.

## Evidence boundary

- v224 is a post-open mechanism-capacity diagnostic on opened Cases 2 and 5, not a deployment algorithm or fresh external gate;
- it closes only the claim that a single worst-camera drift or per-camera residual supports fail-closed fallback; it does not prove that every multiview self-consistency mechanism is impossible;
- orientation, threshold, camera grouping, and normalization may not be changed after results, and a larger model or GPU may not rescue this route;
- there is no candidate physical replay, exact-call reduction, wall/RSS, external-condition, curved-ray, or real-BOST result.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
