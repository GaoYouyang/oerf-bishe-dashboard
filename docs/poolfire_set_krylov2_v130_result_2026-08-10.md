# v129-v130：从失败的重启 K1 到双状态 Krylov 机制容量突破

## 一句话结论 / One-sentence verdict

**中文。** v129 证明“预测 K3 解、再重启一次 CGLS”在结构上不够；v130 加回 K3 到 K4 所需的共轭方向后，在相同 `2A+2A^T` 部署预算内，对 `3700/3700` 个已开封 PoolFire 合成 BOS 单元以约 `1e-15` 的数值误差复现了 Zero-CGLS K4。这个结果是**机制容量突破**，不是已训练算法突破。

**English.** v129 shows that predicting the K3 solution and then restarting one CGLS step is structurally insufficient. After v130 restores the conjugate direction needed to move from K3 to K4, the same `2A+2A^T` deployment budget reproduces Zero-CGLS K4 on `3,700/3,700` opened PoolFire synthetic-BOS cells to about `1e-15`. This is a **mechanism-capacity breakthrough**, not a validated learned-algorithm breakthrough.

## 1. 实际问题 / The actual problem

我们的目标不是让网络直接猜三维密度场，而是让网络从一个可乱序、可增删的多相机集合中预测低成本 Krylov 状态。每个样本包含：

- `5/7/9/12` 台相机；
- 每台相机 `16x16x2` 的合成 BOS 观测；
- 每台相机 18 维 reported pose token；
- 观测噪声、旋转、平移、焦距、主点与联合扰动；
- 相机 mask，保证集合可变长且置换等变。

The model does not directly guess the 3D density field. It predicts a low-cost Krylov state from a reorderable and variable-cardinality camera set. Each sample contains `5/7/9/12` cameras, a `16x16x2` synthetic-BOS observation per camera, an 18D reported pose token, factorized observation/calibration errors, and a camera mask.

## 2. v129 为什么失败 / Why v129 failed

v129 使用精确 K3 solution dual `z3`，先计算 `h=A^T z3`，再用观测求一个缩放系数，最后从这个初值重启 CGLS K1。即使教师完全正确，五条轨迹仍全部未通过冻结门：

| 指标 / Metric | p50 | p90-higher | worst |
|---|---:|---:|---:|
| Field / 场 | 1.05509 | 1.08840 | 1.14114 |
| Full gradient / 全梯度 | 1.01515 | 1.03241 | 1.05879 |
| Interior gradient / 内部梯度 | 1.01338 | 1.03180 | 1.06291 |
| Observation / 观测 | 1.08799 | 1.11253 | 1.14475 |

这里的数值是 candidate error / K4 error，理想值应接近 1。独立程序重新生成观测、位姿、mask、K3 dual、缩放系数与全部指标，最大差均为 `0`。

The ratios above are candidate error divided by K4 error, so values near one are required. An independent program regenerates the observations, poses, masks, K3 duals, scale coefficients, and all metrics with maximum difference `0`.

**物理解释 / Physical interpretation:** restart 丢掉了 CGLS 累积的共轭方向。这个失败属于参数化本身，不属于训练不充分，因此 v129 被关闭，禁止用更大网络挽救。

## 3. v130 改了什么 / What v130 changes

v130 预测两组 detector dual：

1. `z3`：满足 `A^T z3 = x3`，表示 K3 解；
2. `e3`：满足 `A^T e3 = p3`，表示 K3 后用于第 4 步的共轭方向。

部署时固定执行：

```text
h = A^T z3
p = A^T e3
u = A h
v = A p
(c0, c1) = argmin ||y - c0 u - c1 v||2,  c0,c1 in [-2,3]
x0 = c0 h + c1 p
```

因此在线精确调用账恒为 `2A+2A^T`；参考 K4 为 `4A+4A^T`。

The online exact-call ledger is fixed at `2A+2A^T`, versus `4A+4A^T` for the K4 reference.

## 4. 正式与独立复算 / Formal and independent recomputation

| 检查 / Check | 正式最大差 / Formal maximum | 独立最大差 / Independent maximum |
|---|---:|---:|
| Relative field difference to K4 | `7.32e-16` | `2.50e-15` |
| Relative residual difference to K4 | `1.87e-15` | `7.44e-15` |
| Absolute metric difference to K4 | `3.33e-16` | `6.66e-16` |
| K3 solution dual difference | `0` | `0` |
| K3 direction dual difference | n/a | `0` |

正式结果为 `PASS_V130_STAGE_A_DUAL_STATE_KRYLOV2_CAPACITY`；独立结果为 `PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_SET_KRYLOV2_STAGE_A_V130`。独立程序重新构造 CGLS recurrence、两组 dual、二维盒约束求解和全部指标；它仍共享冻结的低层物理 kernel，因此没有声称端到端物理实现完全独立。

The formal status is `PASS_V130_STAGE_A_DUAL_STATE_KRYLOV2_CAPACITY`; the independent status is `PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_SET_KRYLOV2_STAGE_A_V130`. The independent implementation rebuilds the CGLS recurrence, both dual states, the box-constrained 2D solve, and all metrics. It still shares frozen low-level physics kernels, so end-to-end physics independence is not claimed.

## 5. 后续学习验证结果 / Subsequent learning result

首个可学习模型固定为 `OddPoseSetKrylov2Warm`，只有 `11,504` 个参数。它使用共享相机编码器、18 维位姿 MLP、masked mean/max DeepSets 聚合与双状态 decoder；训练采用五条完整轨迹留一、固定 30 epoch、固定 seed、无 early stopping、无 epoch 选择。

The first learnable model is fixed at `11,504` parameters. It uses a shared camera encoder, an 18D pose MLP, masked mean/max DeepSets aggregation, and a dual-state decoder. Training uses five complete-trajectory leave-one-out folds, exactly 30 epochs, one fixed primary seed, no early stopping, and no epoch selection.

所有五个 checkpoint 在任何 held-out 重建指标被读取前统一封存。随后才用冻结的 `2A+2A^T` 壳逐单元评分，并公平比较 Zero-K2、geometry-PCGLS K2、fit-only ridge、no-pose、wrong-pose 与相机共置换。

All five checkpoints were sealed before any held-out reconstruction metric was read. Every held-out cell was then scored through the frozen `2A+2A^T` shell against K4 and equal-cost controls.

正式 v130.1 结果已经完成并由第二个程序独立复算：五条完整留出轨迹为 `0/5` 通过。全局 candidate/K4 的 observation error ratio 为 `p50=1.2668`、`p90-higher=1.4499`、`worst=1.7946`。模型优于同成本 K2 控制，但没有达到冻结的 K4 同精度目标，因此当前双状态学习表示按合同关闭，不追加大模型或多 seed 挽救。

The formal v130.1 evaluation is complete and independently recomputed: `0/5` complete held-out trajectories pass. The global candidate/K4 observation-error ratio is `p50=1.2668`, `p90-higher=1.4499`, and `worst=1.7946`. The model beats equal-cost K2 controls but does not meet the frozen K4 matched-accuracy target, so the current dual-state learned representation is closed without a larger-model or extra-seed rescue.

完整负结果见 [v130.1 整轨迹留一证据](../document_reader.html?doc=docs%2Fpoolfire_set_krylov2_loto_v130_1_result_2026-08-10.md)。下一机制从精确 K1 residual 出发，只预测一组 detector-space correction dual；在 exact-teacher 容量与便宜确定性控制通过前，不授权新训练。

See the [complete v130.1 LOTO evidence](../document_reader.html?doc=docs%2Fpoolfire_set_krylov2_loto_v130_1_result_2026-08-10.md). The next mechanism starts from the exact K1 residual and predicts one detector-space correction dual; no new training is authorized before exact-teacher capacity and cheap deterministic controls pass.

## 6. 证据边界 / Evidence boundary

- `mechanism_capacity_breakthrough=true`
- `learned_initializer_validated=false`
- `algorithm_breakthrough=false`
- `resource_speedup=false`
- `external_generalization=false`
- `curved_ray_validated=false`
- `real_BOST=false`
- `paper_success=false`

这个结果说明“少一半精确算子调用的表示壳层在代数上可行”，尚未说明网络能在完整留出轨迹上预测到足够准确的两组 dual。真实实验数据到位后，还需要真实位移图、相机标定/ID/mask、坐标约定、重复测量噪声与组内认可基线。

The result establishes algebraic feasibility of a representation shell using half the exact operator calls. It does not yet establish that a network can predict both dual states accurately on complete held-out trajectories. Real-data transfer will still require displacement maps, camera calibration/IDs/masks, coordinate conventions, repeated-measurement noise, and the in-lab accepted baseline.
