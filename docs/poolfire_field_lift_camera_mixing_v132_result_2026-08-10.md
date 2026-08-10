# v132: Truth-aware per-camera amplitude mixing still cannot recover K4 accuracy

> 中文标题：**v132：真值知晓的逐相机幅值混合仍无法恢复 K4 同精度**  
> Updated / 更新日期：2026-08-10

## One-sentence verdict / 一句话结论

**English.** Even after using the already opened K4 teacher to choose one optimal signed scalar per active camera, the frozen v131.1 spatial proposal passes none of the five complete-trajectory matched-accuracy gates. Independent recomputation confirms the result. Camera-relative gain is not the main missing mechanism; the detector-space spatial or spectral shape must change.

**中文。** 即使允许使用已开封 K4 teacher，为每台有效相机选择一个最优带符号标量，v131.1 的冻结空间 proposal 仍在五条完整轨迹上全部失败。独立复算确认该结果。因此缺口不是简单的相机间增益，而是 detector-space 的空间或频率形状。

## 1. Why this diagnostic was necessary / 为什么需要做这个诊断

v131.1 learned a correction-dual direction with median cosine `0.9915`, yet its final field and observation errors remained above CGLS K4. Two explanations were still possible:

1. the model predicted the right pixelwise shape but used the wrong relative amplitude across cameras;
2. the proposal shape itself was wrong in directions amplified by `A^T`, `A`, and the field metrics.

v131.1 的 correction dual 与 teacher 的中位 cosine 已达 `0.9915`，但最终 field 和 observation 仍没有追平 K4。当时仍有两种可能：模型已学对像素形状，只是不同相机之间的幅值错了；或者 proposal 在会被 `A^T/A` 放大的形状方向上本身就不对。

v132 isolates these explanations without training another model.

## 2. Frozen truth-aware camera-mixing oracle / 冻结的真值知晓相机混合 oracle

For active camera `c`, split the frozen v131.1 proposal into `zhat_c` and evaluate

```text
h_c = A_c^T zhat_c
p_c = A h_c
```

The oracle uses the already opened K4 correction teacher to solve one small `5/7/9/12`-dimensional ridge system:

```text
min_s  ||sum_c s_c h_c - h_teacher||^2 / ||h_teacher||^2
     + ||sum_c s_c p_c - p_teacher||^2 / ||p_teacher||^2
```

The signed coefficients are normalized to unit RMS because the downstream observation-only line search absorbs one common scale. The candidate still uses the same logical `2A+2A^T` K1-plus-correction shell, while K4 uses `4A+4A^T`.

这个 oracle 使用 teacher 信息，所以它不是可部署方法；它只问一个容量问题：如果 proposal 的每台相机像素形状都保持不变，只允许调整一个标量，最佳情况能否守住 K4 同精度。

A deployment-visible residual-RMS calibration is also run as a cheap deterministic control. It uses no truth and no extra exact calls.

## 3. Result / 结果

Each entry below is `candidate error / K4 error`. The frozen gate requires `p90-higher <= 1.02` and `worst <= 1.05` for field, full gradient, interior gradient, and reported-observation error on every trajectory.

| Trajectory / 轨迹 | Oracle field p90 / worst | Oracle gradient p90 / worst | Oracle interior-gradient p90 / worst | Oracle observation p90 / worst |
|---|---:|---:|---:|---:|
| `14 kW, size 05` | 1.0891 / 1.1290 | 1.0227 / 1.0530 | 1.0269 / 1.0465 | 1.2164 / 1.2782 |
| `22 kW, size 03` | 1.1142 / 1.1612 | 1.0459 / 1.0810 | 1.0575 / 1.0976 | 1.3082 / 1.4300 |
| `33 kW, size 01` | 1.1336 / 1.2273 | 1.0178 / 1.0323 | 1.0291 / 1.0675 | 1.4135 / 1.5158 |
| `45 kW, size 05` | 1.0968 / 1.1144 | 1.0365 / 1.0535 | 1.0281 / 1.0544 | 1.3532 / 1.4459 |
| `58 kW, size 03` | 1.0946 / 1.1195 | 1.0356 / 1.0565 | 1.0528 / 1.0881 | 1.4081 / 1.5951 |

The oracle modestly improves the parent v131.1 observation p90 on every trajectory, but remains far from the `1.02` gate. Field p90 also remains between `1.0891` and `1.1336`. The cheap RMS control fails as well and is generally slightly worse than the original proposal.

oracle 对每条轨迹都有小幅改善，但 observation p90 仍在 `1.2164–1.4135`，距 `1.02` 还很远；field p90 也仍在 `1.0891–1.1336`。便宜的 residual-RMS 相机校准同样失败，并且总体略差于原始 proposal。

Before the downstream line search, even the truth-aware camera mixture has:

- field-lift relative L2: p50 `0.3052`, p90 `0.4716`, worst `0.7280`;
- projected-lift relative L2: p50 `0.4417`, p90 `0.6774`, worst `1.1947`.

The small ridge systems are not numerically pathological: their regularized condition-number p90 is only `5.83`. The failure is therefore not explained by an unstable 5–12 dimensional solve.

## 4. Independent recomputation / 独立复算

The formal path uses grouped adjoints and a direct SPD solve. A separate validator instead constructs one standard-adjoint batch per camera and solves the same frozen system through an eigendecomposition. It then independently rebuilds K1, the correction line search, all four metrics, complete-trajectory summaries, and the final gate.

| Independent check / 独立检查 | Maximum difference / 最大差 |
|---|---:|
| Oracle camera coefficients | `6.66e-15` |
| Oracle diagnostics | `1.33e-14` |
| Oracle metrics | `4.44e-16` |
| RMS-control coefficients / metrics | `0 / 0` |
| Complete-trajectory summaries | `4.44e-16` |
| Rebuilt K1 residual | `2.54e-13` |
| Reported pose tokens | `2.22e-16` |

The independent status is `PASS_INDEPENDENT_RECOMPUTATION_FIELD_LIFT_CAMERA_MIXING_V132`, and the sealed formal tree is unchanged. Both paths still share the frozen physical operator kernels, so end-to-end physics independence is not proven.

## 5. What this closes and what remains / 关闭什么，保留什么

v132 closes the hypothesis that v131.1 mainly needs one scalar calibration per camera. Because even a truth-aware best-case mixture fails, training a camera-coefficient predictor would spend compute on a representation that lacks capacity. No such predictor, larger CNN, FNO, UNO, DeepONet, GPU run, resource gate, or external gate is authorized.

v132 关闭了“v131.1 主要差在每台相机一个标量系数”这个假设。由于真值知晓的最佳情况也不足，再训练相机系数预测器只会在没有容量的表示上浪费算力。

The surviving route must change pixelwise detector-space shape. The next falsifiable gate is therefore:

1. build a small normal-operator-sensitive spatial or spectral correction family from deployment-visible K1 residual and reported geometry;
2. before training, use opened fit truth only to test whether that family has `3700/3700` K4-equivalent capacity;
3. compare a cheap deterministic control first;
4. stop immediately if capacity is absent;
5. only after capacity exists may one minimal observation/geometry-only sentinel be trained.

下一个机制必须能改变 detector-space 的像素级空间/频率形状。训练前先做真值知晓容量门；如果新表示不能在 `3700/3700` 单元上守住 K4，就立即停止，不用大模型挽救。

## Evidence boundary / 证据边界

- `per_camera_scalar_mixing_capacity=false`
- `camera_coefficient_predictor_authorized=false`
- `algorithm_breakthrough=false`
- `resource_speedup=false`
- `external_generalization=false`
- `curved_ray_validated=false`
- `real_bost=false`
- `paper_success=false`

This is an independently recomputed, truth-aware capacity failure on already opened synthetic PoolFire fit data. It is not a deployable algorithm, not an external-generalization result, and not a laboratory BOST result.

这是已开封 PoolFire 合成 fit 数据上的真值知晓容量负结果，已独立复算。它不是可部署算法、外部泛化结果或组内真实 BOST 结果。
