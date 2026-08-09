# v131.1: Minimal K1-residual set model fails complete-trajectory transfer

> 中文标题：**v131.1：最小 K1-residual 相机集合模型整轨迹留一负结果**  
> Updated / 更新日期：2026-08-10

## One-sentence verdict / 一句话结论

**English.** The exact v131 correction-dual mechanism can match CGLS K4 with `2A+2A^T`, but the frozen 11,484-parameter observation-and-reported-pose set model reaches the strict matched-accuracy gate on `0/5` held-out PoolFire trajectories. Independent recomputation reproduces every prediction, metric, summary, and call receipt exactly.

**中文。** v131 的精确 correction-dual 机制可以用 `2A+2A^T` 追平 CGLS K4，但冻结的 11,484 参数观测/报告位姿相机集合模型，在五条完整留出 PoolFire 轨迹上只有 `0/5` 通过严格同精度门。独立程序对预测、指标、汇总和调用账的复算差全部为 `0`。

## 1. What was actually tested / 实际测试了什么

The experiment keeps the v131 physics shell fixed and changes only the source of one detector-space correction dual:

```text
exact CGLS K1 residual + reported camera geometry
-> permutation-invariant variable-camera set model
-> one predicted detector-space correction dual
-> one exact A^T lift + one exact A projection
-> observation-only scalar line search
-> final 3D field
```

实验固定 v131 的物理壳，只改变一组 detector-space correction dual 的来源：模型读取精确 CGLS K1 residual、报告相机几何与有效相机 mask，预测一组 correction dual；随后执行一次精确 `A^T` 提升、一次精确 `A` 投影和只看观测的标量线搜索。

The complete-trajectory leave-one-out roster contains:

- five opened PoolFire fit trajectories and `3,700` held-out cells;
- `5/7/9/12` active cameras, with camera order allowed to change;
- clean observations plus observation, rotation, translation, focal-length, principal-point, and combined perturbations;
- one frozen seed, 30 fixed epochs per fold, no early stopping, and no held-out checkpoint selection;
- a candidate ledger of `2A+2A^T`, compared with zero-start CGLS K4 at `4A+4A^T`.

五条完整轨迹各留出一次，共评分 `3,700` 个单元。相机数可为 `5/7/9/12`，顺序可交换；数据同时覆盖 clean、观测噪声、旋转、平移、焦距、主点和联合扰动。每折固定 30 epoch、单一主 seed，不 early stop，也不利用留出结果选择 checkpoint。

## 2. Frozen matched-accuracy result / 冻结同精度结果

Every table entry is `candidate error / K4 error`. The frozen gate requires both `p90-higher <= 1.02` and `worst <= 1.05` for field, full gradient, interior gradient, and reported-observation error on every trajectory.

下表均为 `candidate error / K4 error`。冻结门要求每条轨迹上的 field、完整梯度、内部梯度和报告观测误差同时满足 `p90-higher <= 1.02` 与 `worst <= 1.05`。

| Held-out trajectory / 留出轨迹 | Field p90 / worst | Gradient p90 / worst | Interior-gradient p90 / worst | Observation p90 / worst |
|---|---:|---:|---:|---:|
| `14 kW, size 05` | 1.0965 / 1.1346 | 1.0221 / 1.0504 | 1.0254 / 1.0431 | 1.2377 / 1.3229 |
| `22 kW, size 03` | 1.1305 / 1.1871 | 1.0438 / 1.0784 | 1.0537 / 1.0983 | 1.3337 / 1.4605 |
| `33 kW, size 01` | 1.1479 / 1.2521 | 1.0090 / 1.0269 | 1.0256 / 1.0694 | 1.4764 / 1.6225 |
| `45 kW, size 05` | 1.1045 / 1.1358 | 1.0414 / 1.0545 | 1.0265 / 1.0565 | 1.4135 / 1.5409 |
| `58 kW, size 03` | 1.1023 / 1.1344 | 1.0324 / 1.0586 | 1.0501 / 1.0905 | 1.4154 / 1.6236 |

The formal scientific status is `FAIL_V131_1_PRIMARY_HELDOUT_ACCURACY`: all five complete trajectories fail. The model is better than Zero-CGLS K2 in observation p90 on every trajectory and no frozen cheap control globally dominates it, so it did learn useful signal. That is still not enough: the stated claim is K4-equivalent accuracy at half the exact-call budget, not merely improvement over K2.

正式科学状态是 `FAIL_V131_1_PRIMARY_HELDOUT_ACCURACY`：五条完整轨迹全部失败。候选在每条轨迹的 observation p90 上都优于 Zero-CGLS K2，且没有冻结便宜控制能全局支配它，说明模型并非什么都没学到。但论文目标是以一半精确调用达到 K4 同精度，而不是只比 K2 好，因此不能判成功。

## 3. Independent recomputation / 独立复算

A separate validator reloads the frozen model outputs, reconstructs the physical shell, recomputes all `3,700` cells and the control comparisons, and checks the real call sites rather than trusting a declared budget.

第二个程序重载封存输出，重建物理壳，重新计算全部 `3,700` 个单元与控制比较，并在真实 forward/adjoint 调用点计账，而不是只相信声明的成本。

| Independent check / 独立检查 | Result / 结果 |
|---|---:|
| Prediction maximum difference / 预测最大差 | `0` |
| Metric maximum difference / 指标最大差 | `0` |
| Summary and control-summary maximum difference / 汇总与控制汇总最大差 | `0` |
| Alpha, parity, K1 residual, and pose-token maximum difference | `0` |
| Actual exact calls / 实际精确调用 | `7,400 A + 7,400 A^T` |
| Per-cell ledger / 单元调用账 | `2A+2A^T` |
| Validation/test truth opened / validation/test 真值打开 | `false / false` |

The independent status is `PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_K1_RESIDUAL_SCORE_V131_1`. The shared frozen physics kernels mean end-to-end physics independence is not proven; the result is nevertheless independently recomputed at the prediction, shell, metric, aggregation, and call-ledger levels.

独立状态为 `PASS_INDEPENDENT_RECOMPUTATION_POOLFIRE_K1_RESIDUAL_SCORE_V131_1`。正式与独立路径仍共享冻结 physics kernels，所以没有证明端到端物理实现完全独立；但预测、物理壳、指标、聚合和调用账已经分别复算。

## 4. What the failure teaches us / 失败告诉了我们什么

The post-open diagnostic shows a subtle but important mismatch:

- predicted-versus-teacher dual relative L2: p50 `0.2451`, p90 `0.4484`, worst `0.7190`;
- dual cosine similarity: p10 `0.9846`, median `0.9915`;
- line-search-scaled physical correction relative error: p50 `0.3107`, p90 `0.4384`, worst `0.6077`;
- observation p90 by camera count: `1.5000`, `1.3748`, `1.3070`, and `1.2397` for 5, 7, 9, and 12 cameras.

后验诊断揭示了一个关键错配：模型预测的 dual 与教师方向夹角其实很小，中位 cosine 达到 `0.9915`，但有效物理修正仍有约 `31%` 的中位相对误差；而观测误差在 5 相机时明显比 12 相机严重。clean 条件的 observation p90 也约为 `1.4109`，和多种噪声/位姿扰动条件相近，因此主因不是某一种扰动太难。

This is consistent with an ill-conditioned inverse problem: ordinary active-camera relative dual-L2 rewards broad directional agreement, but does not sufficiently penalize small errors in directions that are strongly amplified by `A^T`, `A`, and the final field metrics. The scalar line search can correct overall gain, not the remaining spatial/detector-space shape error.

这与病态逆问题的性质一致：普通 active-camera dual-L2 能让总体方向接近，却没有充分惩罚那些会被 `A^T`、`A` 和最终场指标强烈放大的敏感误差方向。标量线搜索可以修正整体幅值，却不能修复剩余的 detector-space / 三维空间形状误差。

The diagnostic is descriptive and post-open; it identifies the next mechanism question but is not itself a new validated algorithm.

这部分是打开正式结果后的描述性归因，用来收缩下一机制问题，不是新的已验证算法。

## 5. Decision / 路线判决

The current target and loss are closed. We will not rescue them with extra seeds, a larger CNN, FNO, UNO, DeepONet, GPU rental, resource benchmarking, or an external gate. Those steps would spend more compute without fixing the demonstrated objective mismatch.

当前 prediction target 与普通 dual-L2 训练目标正式关闭。不追加多 seed、大 CNN、FNO、UNO、DeepONet、GPU 租赁、资源测试或外部门来挽救，因为这些操作不会先解决已经暴露的目标错配。

The next valid mechanism question is smaller and physics-sensitive: before any new training, test whether a target or low-dimensional representation weighted by the exact normal operator / field lift can preserve the K4-equivalent correction under variable camera counts. It must be frozen before results, use only opened fit data, compare a cheap deterministic control first, and stop immediately if capacity is absent.

下一条有效问题是：在任何新训练之前，先检查由精确 normal operator / field lift 加权的目标或低维表示，能否在变相机数量下保住 K4 等价修正。必须先冻结合同，只用已开封 fit 数据，先比较便宜确定性控制；若容量不存在就立即停止。

## Evidence boundary / 证据边界

- `v131_mechanism_capacity_headroom=true`
- `v131_1_primary_trajectory_gate_passed=0/5`
- `v131_1_learned_initializer_validated=false`
- `current_dual_l2_representation_closed=true`
- `algorithm_breakthrough=false`
- `resource_speedup=false`
- `external_generalization=false`
- `curved_ray_validated=false`
- `real_bost=false`
- `paper_success=false`

This is an independently recomputed negative result on opened, noisy, variable-camera, known-reported-geometry, straight-ray synthetic PoolFire data. It is not a laboratory BOST result and does not prove that all learned warm starts are impossible.

这是在已开封、带噪、变相机数量、已知报告几何的 straight-ray 合成 PoolFire 数据上的独立复算负结果。它不是组内真实 BOST 结果，也不证明所有 learned warm start 都不可能。
