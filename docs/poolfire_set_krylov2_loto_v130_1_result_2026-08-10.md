# v130.1: SetKrylov2 complete-trajectory LOTO negative result

> 中文标题：**v130.1：SetKrylov2 整轨迹留一负结果**  
> Updated / 更新日期：2026-08-10

## One-sentence verdict / 一句话结论

**English.** The exact dual-state v130 shell can reproduce CGLS K4 with `2A+2A^T`, but the frozen 11,504-parameter observation-and-reported-pose camera-set model does not predict the two dual states accurately enough on any of five held-out PoolFire trajectories. The formal result is a failure, independently recomputed to numerical precision.

**中文。** v130 的精确双状态壳可以用 `2A+2A^T` 复现 CGLS K4，但冻结的 11,504 参数观测/报告位姿相机集合模型，在五条完整留出 PoolFire 轨迹上都没能把两组 dual 预测到足够准确。正式结果为失败，独立程序已在数值精度内复算确认。

## 1. What was tested / 实际测试了什么

The model receives a reorderable, variable-cardinality camera set:

- `5/7/9/12` active cameras;
- a `16x16x2` synthetic-BOS observation per camera;
- an 18D reported-pose token and a camera mask;
- observation noise and camera rotation, translation, focal-length, principal-point, and combined perturbations.

模型输入是可乱序、可变相机数量的集合：

- `5/7/9/12` 台有效相机；
- 每台相机 `16x16x2` 的合成 BOS 观测；
- 每台相机 18 维报告位姿编码和 mask；
- 观测噪声，以及旋转、平移、焦距、主点和联合扰动。

Five complete-trajectory leave-one-out folds were trained for exactly 30 epochs. All five checkpoints were sealed before any held-out prediction or reconstruction metric was generated. The 3,700 primary predictions and 3,700 wrong-pose predictions were then sealed before opened-fit truth or classical controls were read.

五个完整轨迹留一折都固定训练 30 轮。所有 checkpoint 在任何留出预测和重建指标产生前统一封存；随后 3,700 个主预测和 3,700 个错位姿预测也先封存，再读取已开封 fit 真值与经典控制。

The candidate always uses the same exact online ledger:

```text
learned solution dual + learned direction dual
-> two exact A^T lifts
-> two exact A projections
-> observation-only bounded 2D solve
= 2A + 2A^T
```

The reference is zero-start CGLS K4 with `4A+4A^T`.

## 2. Frozen matched-accuracy result / 冻结同精度结果

Each value below is `candidate error / K4 error`; values near one are required. The frozen gate requires every trajectory and every metric to satisfy `p90-higher <= 1.02` and `worst <= 1.05`.

下表均为 `candidate error / K4 error`，理想值应接近 1。冻结门要求每条轨迹、每项指标同时满足 `p90-higher <= 1.02` 和 `worst <= 1.05`。

| Held-out trajectory / 留出轨迹 | Field p90 / worst | Gradient p90 / worst | Interior-gradient p90 / worst | Observation p90 / worst |
|---|---:|---:|---:|---:|
| `14 kW, size 05` | 1.0654 / 1.1036 | 1.0154 / 1.0444 | 1.0419 / 1.0698 | 1.3174 / 1.4267 |
| `22 kW, size 03` | 1.0686 / 1.1128 | 1.0421 / 1.0664 | 1.0779 / 1.1052 | 1.4168 / 1.5046 |
| `33 kW, size 01` | 1.0669 / 1.0986 | 1.0351 / 1.0491 | 1.0794 / 1.1150 | 1.4544 / 1.5434 |
| `45 kW, size 05` | 1.0768 / 1.0933 | 1.0313 / 1.0453 | 1.0564 / 1.0709 | 1.5123 / 1.7946 |
| `58 kW, size 03` | 1.0737 / 1.0951 | 1.0139 / 1.0353 | 1.0543 / 1.0635 | 1.4440 / 1.5678 |

All five trajectories fail the full gate. Across all 3,700 cells, candidate/K4 `p50 / p90-higher / worst` is:

- field: `1.0501 / 1.0726 / 1.1128`;
- full gradient: `1.0085 / 1.0308 / 1.0664`;
- interior gradient: `1.0359 / 1.0658 / 1.1150`;
- observation: `1.2668 / 1.4499 / 1.7946`.

五条轨迹全部未通过完整门。模型仍明显优于同成本的 Zero-CGLS K2 和 geometry-PCGLS K2，而且没有被这两个控制全局支配；但“优于 K2”不等于“达到 K4 同精度”，因此科学判决仍是失败。

The model remains substantially better than the equal-cost Zero-CGLS K2 and geometry-PCGLS K2 controls and is not globally dominated by either. Nevertheless, outperforming K2 is not equivalent to matching K4, so the scientific verdict remains a failure.

## 3. Independent recomputation / 独立复算

A second program does not import the formal scorer, formal two-direction shell, or fit runner. It reloads all five checkpoints, recomputes primary and wrong-pose predictions, reconstructs the bounded two-direction solve, replays the exact physical metrics and call ledger, and verifies that the formal result tree is unchanged.

第二个程序不导入正式评分器、正式双方向壳或训练 runner。它重新加载五个 checkpoint，重算主预测和错位姿预测，重建盒约束双方向求解，重放物理指标与精确调用账，并确认正式结果树未改变。

| Independent check / 独立检查 | Maximum difference / 最大差 |
|---|---:|
| Primary predictions / 主预测 | `0` |
| Wrong-pose predictions / 错位姿预测 | `0` |
| Candidate metrics / 候选指标 | `6.66e-16` |
| Aggregate summaries / 聚合摘要 | `6.66e-16` |
| Coefficients / 系数 | `2.00e-15` |
| K4 and Zero-K2 controls / K4 与 Zero-K2 控制 | `0` |
| Call-ledger mismatches / 调用账不匹配 | `0` |

The independent status is `PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_SET_KRYLOV2_LOTO_SCORE_V130_1`, confirming the formal scientific status `FAIL_V130_1_PRIMARY_HELDOUT_ACCURACY`.

## 4. What failed physically / 物理上卡在哪里

A post-open root-cause audit compares the predictions with the already-opened exact dual teachers:

| Diagnostic / 诊断 | p50 | p90-higher | worst |
|---|---:|---:|---:|
| K3 solution-dual relative L2 | 0.2226 | 0.3576 | 0.5763 |
| K3-to-K4 direction-dual relative L2 | 0.3870 | 0.5210 | 1.0500 |
| Correct-vs-wrong-pose prediction delta, relative L2 | 0.0731 | 0.1188 | 0.2019 |

The direction-dual error has a correlation of about `0.65` with the final observation-error ratio. The failure is also camera-count dependent: the observation-ratio median is about `1.42` with five cameras and `1.16` with twelve cameras.

方向 dual 的误差与最终观测误差比相关系数约为 `0.65`。失败也明显依赖相机数：5 相机时 observation ratio 中位数约 `1.42`，12 相机时约 `1.16`。

This supports a narrow conclusion: the current network can extract useful signal, but predicting two complete K3 dual states across a held-out flow trajectory and changing camera sets is too difficult for this representation. It does not prove that larger networks, arbitrary camera-set transfer, or real BOST must fail.

这只支持一个窄结论：当前网络确实提取到有效信号，但要跨完整留出流场轨迹和变化相机集合，同时预测两组完整 K3 dual，对这个表示来说过难。它不证明更大网络、任意相机集合迁移或真实 BOST 必然失败。

## 5. Decision and next mechanism / 判决与下一机制

The preregistered decision is enforced: no no-pose sibling, fit-only ridge, replication seeds, larger CNN/FNO/UNO/U-Net, resource benchmark, or external gate will be used to rescue this representation.

结果前合同已经执行：不再追加 no-pose、fit-only ridge、多 seed、大 CNN/FNO/UNO/U-Net、资源测试或外部门来挽救这个表示。

The next falsifiable mechanism starts from an exact CGLS K1 state and asks a smaller set model to predict only one detector-space correction dual from the deployment-visible K1 residual and reported poses. One exact `A^T` lift, one exact `A` projection, and an observation-only line search retain a total budget no larger than `2A+2A^T`. Exact-teacher capacity and cheap deterministic controls must pass before any new training.

下一条可证伪路线从精确 CGLS K1 状态出发，只让更小的集合模型根据部署可见的 K1 residual 和报告位姿预测一组 detector-space correction dual。再做一次精确 `A^T` 提升、一次精确 `A` 投影和仅看观测的线搜索，总预算仍不超过 `2A+2A^T`。只有精确教师容量和便宜确定性控制先通过，才允许新训练。

## Evidence boundary / 证据边界

- `v130_mechanism_capacity_breakthrough=true`
- `v130_1_learned_initializer_validated=false`
- `algorithm_breakthrough=false`
- `resource_speedup=false`
- `external_generalization=false`
- `curved_ray_validated=false`
- `real_bost=false`
- `paper_success=false`

This is an independently verified negative result on opened, noise-perturbed, known-reported-geometry, straight-ray synthetic PoolFire data. It is not a result on laboratory BOST measurements.

这是在已开封、带噪声和报告几何误差的 straight-ray 合成 PoolFire 数据上的独立验证负结果，不是组内真实 BOST 实验结果。
