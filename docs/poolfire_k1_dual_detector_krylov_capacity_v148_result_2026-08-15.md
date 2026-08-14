# v148：分组探测器 Krylov 容量首次通过完整哨兵与轨迹门

## 先说结论

v147 表明样本近邻的 action 跨度即使放宽到 32 维仍不能覆盖完整轨迹。v148 因此不再寻找邻居，也没有扩大网络，而是直接利用当前样本自己的部署可见 K1 对偶状态、带符号残差和报告几何，构造一个物理上不同的探测器 Krylov 方向族。

结果出现了明确的正向机制证据：给所有探测器分组共用四个真值 oracle 系数时，只通过 `18/20` 哨兵和 `3/5` 轨迹尾部；改为每个相机/分量组各自使用四阶 Krylov 系数后，通过 `20/20` 哨兵与 `5/5` 轨迹尾部，误差中位数 / p90 / worst 为 `0.08437 / 0.15074 / 0.21897`。独立第二实现复现了该判决。

科学判决是：

`HEADROOM_BLOCK_DETECTOR_KRYLOV4_V148`

这把剩余问题从“是否存在足够的当前样本物理方向”推进到“能否只靠 observation / geometry 预测每个探测器组的少量系数”。它是真值 oracle 容量 headroom，不是已经可部署的算法；现在仍不需要租 GPU。

## 冻结问题

- 已开封 PoolFire 池中的 `20` 个结果前冻结哨兵，覆盖五条轨迹与 `5 / 7 / 9 / 12` 相机条件；
- 每个样本只用 exact-K1 对偶、带符号 K1 residual 与报告几何重建固定的 `96` 个探测器方向；
- 每个方向只作用于自己的目标相机/分量，并按行范数归一化；
- 从可见种子依次生成 `[s, Hs, H^2s, H^3s]` 四阶 Krylov 列；
- `visible_seed_control` 不拟合系数；`global_krylov4` 全局共用四个真值系数；`block_krylov4` 每个相机/分量组使用四个真值系数；
- 门保持不变：每哨兵尺度不变 relative-L2 `<=0.45`、cosine `>=0.90`，每轨迹 p90-higher `<=0.35`；
- 训练参数 `0`，新增精确调用 `+0A/+0A^T`，未做物理场重放。

真值只用于解出容量上限的 oracle 系数，不能作为未来部署输入。

## 实际数字

| 方法 | 哨兵通过 | 轨迹尾部通过 | 误差中位数 | 误差 p90 | 最坏误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| visible seed control | 18/20 | 2/5 | 0.31292 | 0.50457 | 0.54845 |
| global Krylov-4 oracle | 18/20 | 3/5 | 0.27411 | 0.45006 | 0.54507 |
| block Krylov-4 oracle | 20/20 | 5/5 | 0.08437 | 0.15074 | 0.21897 |

分组方法的五条轨迹 p90-higher 分别为 `0.14560 / 0.11303 / 0.05864 / 0.21897 / 0.11814`，全部低于 `0.35`。其 cosine 中位数 / p10-lower / worst 为 `0.99643 / 0.98857 / 0.97573`，也全部越过冻结门。

全局四系数方法在 p45 与 p58 失败，轨迹 p90-higher 分别为 `0.54507` 与 `0.35192`。因此正结果不是“任意四个系数都够”，而是支持一个更具体的解释：不同探测器相机/分量组需要不同的低阶谱响应。

## 独立复算

正式程序用 thin SVD 求容量投影；第二实现改用列 Gram 矩阵的对称特征分解，并在物理空间做独立的约化 QR 清理。第二实现重新构造状态、方向、三种预测、误差、cosine、轨迹尾部与科学判决。

- `14/14` 完整性检查通过；
- 离散投影秩完全一致；
- 状态浮点数组最大差 `1.92e-15`；
- block / global / seed 预测最大差分别为 `1.46e-12 / 5.33e-15 / 1.11e-16`；
- error 与 cosine 最大差分别为 `2.56e-14 / 1.89e-15`；
- 正式结果树在验证前后不变。

两个实现仍共享冻结的物理核，因此 `end_to_end_physics_independence_proven=false`。

## 对主线的影响

v148 首次在当前 20 哨兵容量门上给出 `20/20` 与 `5/5` 的完整正结果。它否定了“当前样本自身没有足够物理方向”这一更悲观解释，并把下一步压缩为一个小问题：能否由 observation / geometry 预测分组的四阶系数，而不读取 CFD 真值。

下一步只冻结一个小型、共享参数、相机置换等变的分组系数预测器，并做完整 trajectory-level leave-one-trajectory-out。它仍可在 CPU 上完成。只有严格外折预测通过，且 CPU 训练时间或模型规模真正成为瓶颈，才重新评估 GPU。

当前边界：

- `groupwise_spectral_capacity_headroom=true`；
- `oracle_is_deployable=false`；
- `observation_only_predictor_passed=false`；
- `neural_training_authorized=false`；
- `gpu_rental_authorized=false`；
- `algorithm_breakthrough=false`；
- `resource_speedup=false`；
- `external_generalization=false`；
- `curved_ray_validated=false`；
- `real_bost=false`；
- `paper_success=false`。

## English checkpoint

v148 replaces v147's sample-neighbor span with a physically distinct, sample-specific detector Krylov family. Exact-K1 dual state, signed K1 residual, and reported geometry build 96 frozen detector directions. A visible seed generates four Krylov columns, `[s, Hs, H^2s, H^3s]`.

The visible-seed control passes `18/20` sentinels and `2/5` trajectory tails. A truth-aware oracle with one global set of four coefficients reaches `18/20` and `3/5`. Allowing four truth-aware coefficients per detector camera/component group reaches `20/20` and `5/5`, with median / p90 / worst error `0.08437 / 0.15074 / 0.21897`. All five trajectory p90-higher values remain below the frozen `0.35` gate.

An independent implementation replaces the formal thin SVD with a symmetric column-Gram eigensolve and physical-space reduced QR cleanup. All fourteen checks pass; projection ranks match exactly, the maximum block-prediction difference is `1.46e-12`, and metric differences are at most `2.56e-14`. The two implementations still share frozen physics kernels, so end-to-end physics independence is not proven.

This is genuine mechanism-capacity headroom, not a deployable warm-start algorithm. Oracle coefficients use opened CFD truth, no physical replay or matched-accuracy reconstruction was run, and no call, wall-time, memory, external-generalization, curved-ray, or real-BOST claim follows. The next valid step is a small permutation-equivariant observation/geometry-only group-coefficient predictor with complete-trajectory leave-one-out evaluation. That experiment remains CPU-sized, so GPU rental is still not warranted.
