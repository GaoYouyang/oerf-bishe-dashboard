# v131: K1-residual single-correction-dual mechanism capacity

> 中文标题：**v131：K1 residual 单 correction-dual 机制容量与便宜控制**  
> Updated / 更新日期：2026-08-10

## One-sentence verdict / 一句话结论

**English.** On 3,700 noisy, calibration-perturbed synthetic-BOS cells from five opened PoolFire trajectories, one exact detector-space correction dual reproduces the CGLS K4 field from an exact K1 state with a `2A+2A^T` online ledger. Five equal-or-cheaper deterministic controls all fail the frozen trajectory-level gate. An independent program reproduces the result, so a minimal learned predictor is now scientifically justified, but no learned-model or generalization result exists yet.

**中文。** 在五条已开封 PoolFire 轨迹的 3,700 个带观测噪声和标定扰动的合成 BOS 单元上，从精确 K1 状态出发，一组精确 detector-space correction dual 可以用 `2A+2A^T` 在线调用账数值复现 CGLS K4。五个同价或更便宜的确定性控制全部未通过冻结的逐轨迹门。独立程序已复算确认，因此现在有理由训练一个最小预测器；但目前还没有学习模型或泛化成功。

## 1. Mechanism / 机制

Let `x1` be the exact zero-start CGLS K1 field and `r1 = y - A x1` its deployment-visible residual. The offline teacher is the single correction dual

```text
delta_z = z4 - z1
h       = A^T delta_z = x4 - x1
p       = A h
alpha   = <r1, p> / <p, p>
x       = x1 + alpha h
```

`alpha` uses only the current observation residual. With the exact teacher it is one to numerical precision, so the corrected field equals CGLS K4. Constructing the teacher requires the offline K4 reference and is not a deployable shortcut; deployment must predict `delta_z` from `r1`, the reported camera poses, and the active-camera mask.

令 `x1` 为零初值 CGLS K1 的精确场，`r1 = y - A x1` 为部署时可见残差。离线教师只保留一组 correction dual。经一次精确 `A^T` 提升、一次 `A` 投影和仅看观测的标量线搜索后，候选回到 K4。教师本身需要离线 K4 参考，因此不是可部署捷径；真正部署时必须只根据 `r1`、报告相机位姿和有效相机 mask 预测它。

The exact online ledgers are:

| Arm / 方法 | Exact `A` | Exact `A^T` |
|---|---:|---:|
| CGLS K4 reference | 4 | 4 |
| Exact or predicted single correction dual | 2 | 2 |
| Zero-CGLS K2 | 2 | 2 |
| Zero-CGLS K1 | 1 | 1 |

## 2. Data and frozen gate / 数据与冻结门

- five opened PoolFire fit trajectories;
- 3,700 cells in total;
- `5 / 7 / 9 / 12` active cameras, treated as a reorderable set;
- 37 clean, observation-noise, rotation, translation, focal-length, principal-point, and combined conditions;
- a reported 18D pose/calibration token per camera;
- the same straight-ray forward/adjoint and the same K4 reference for every arm.

- 五条已开封 PoolFire fit 轨迹；
- 共 3,700 个单元；
- `5 / 7 / 9 / 12` 台有效相机，按可乱序集合处理；
- 37 种 clean、观测噪声、旋转、平移、焦距、主点和联合扰动条件；
- 每台相机一组 18 维报告位姿/标定编码；
- 所有方法共享同一 straight-ray forward/adjoint 和 K4 参考。

For every trajectory and each of field, full-gradient, interior-gradient, and reported-observation error, the frozen gate requires `candidate/K4 p90-higher <= 1.02` and `worst <= 1.05`.

冻结门逐轨迹检查 field、full-gradient、interior-gradient 和 reported-observation 四项指标，并要求每项 `candidate/K4 p90-higher <= 1.02` 且 `worst <= 1.05`。

## 3. Exact capacity / 精确容量

The exact teacher passes every metric on all five trajectories. Across all 3,700 cells:

| Exact comparison / 精确比较 | Maximum difference / 最大差 |
|---|---:|
| Relative field difference to K4 | `6.12e-16` |
| Relative observation-residual difference to K4 | `1.34e-15` |
| Absolute metric difference to K4 | `3.33e-16` |
| `|alpha - 1|` | `6.66e-16` |

This establishes exact mechanism capacity for the smaller target. It does not show that a model can predict the target on a held-out trajectory.

这证明了更小目标的精确机制容量，但不证明模型能够在完整留出轨迹上预测它。

## 4. Cheap controls / 便宜控制

Every value below is the global `control error / K4 error`. Passing requires trajectory-level, rather than merely global, tails to remain inside the frozen gate. None of the five controls passes all five trajectories.

下表是全局 `control error / K4 error`。正式判决使用更严格的逐轨迹尾部；五个控制没有一个通过全部五条轨迹。

| Control / 控制 | Field p90 / worst | Observation p90 / worst | Full trajectory gate |
|---|---:|---:|---:|
| Zero-CGLS K1 | `1.4646 / 1.6413` | `2.1541 / 2.4022` | `0 / 5` |
| Zero-CGLS K2 | `1.2862 / 1.4492` | `1.6226 / 1.7995` | `0 / 5` |
| Raw K1 residual dual | `1.3835 / 1.5545` | `1.8018 / 1.9380` | `0 / 5` |
| View-balanced residual dual | `1.3837 / 1.5557` | `1.8022 / 1.9242` | `0 / 5` |
| Constant-preserving `3x3` box residual | `1.2608 / 1.4176` | `1.8263 / 2.0873` | `0 / 5` |

The best simple field control is the `3x3` box residual, but its observation tail is still far outside the gate. The headroom therefore cannot be explained by one extra CGLS step, the raw residual, per-camera RMS balancing, or a fixed local detector filter.

最好的简单 field 控制是 `3x3` box residual，但它的 observation 尾部仍远远越线。因此当前余量不能由“多跑一步”、原始残差、逐相机 RMS 均衡或固定局部 detector 滤波解释。

## 5. Independent recomputation / 独立复算

A second program does not import the formal runner or the v131 mechanism core. It reconstructs CGLS, the single-dual correction, controls, metrics, tails, and gates. It reproduces all 3,700 cells with:

- zero difference in K1 residual, correction dual, metrics, alphas, parity, and reported-pose tokens;
- the same `6.12e-16` field and `1.34e-15` residual differences to K4;
- no change to the formal result tree.

第二个程序不导入正式 runner 或 v131 机制 core，而是重新实现 CGLS、单 dual 修正、控制、指标、尾部与门。3,700 个单元的 K1 residual、correction dual、metrics、alpha、parity 和报告位姿编码差均为 `0`，并复现同样的 K4 数值误差。两个程序仍共享冻结的底层 physics kernels，这一边界保留披露。

Independent status: `PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_K1_RESIDUAL_CAPACITY_V131`.

## 6. Decision / 决策

The v131 result authorizes exactly one small next experiment: a permutation-invariant, variable-cardinality camera-set model that predicts one correction dual from the K1 residual, reported camera geometry, and mask. The primary five-fold leave-one-complete-trajectory-out model must be scored before no-pose variants, extra seeds, larger CNN/FNO/UNO models, resource benchmarks, or external data are considered.

v131 只授权一个很小的后续实验：用可交换顺序、可变相机数量的集合模型，从 K1 residual、报告相机几何和 mask 预测一组 correction dual。主模型必须先完成五折整轨迹留一评分；在它通过前，不追加 no-pose、多 seed、大 CNN/FNO/UNO、资源测试或外部数据门。

When real OERF data arrive, transfer requires synchronized displacement maps, camera IDs and valid masks, full calibration and coordinate conventions, repeated measurements for observation noise, calibration uncertainty or repeated calibration, and the lab-approved reconstruction baseline.

组内真实数据到位后，需要同步位移图、相机编号与有效 mask、完整标定和坐标约定、用于估计观测噪声的重复测量、标定不确定度或重复标定，以及课题组认可的重建基线。

## Evidence boundary / 证据边界

- `mechanism_capacity_headroom=true`
- `cheap_control_explanation_found=false`
- `learned_initializer_validated=false`
- `algorithm_breakthrough=false`
- `resource_speedup=false`
- `external_generalization=false`
- `curved_ray_validated=false`
- `real_bost=false`
- `paper_success=false`

This is a mechanism-level advance and a justified training gate, not an algorithmic breakthrough or a paper-ready result.

这是机制层面的实质进展和有依据的训练授权，不是算法突破，也不是已经可以投稿的结果。
