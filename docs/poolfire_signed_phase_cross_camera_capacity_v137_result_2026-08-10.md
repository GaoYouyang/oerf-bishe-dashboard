# v137 有符号相位与跨相机几何容量诊断

> 公开日期：2026-08-10
>
> 正式状态：`FAIL_V137_SIGNED_PHASE_CROSS_CAMERA_CAPACITY`
>
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_SIGNED_PHASE_CROSS_CAMERA_V137_1`
> 结论边界：`algorithm_breakthrough=false`，不授权预测器、GPU 训练、资源门、外门或真实 BOST 声明。

## 一句话结论

v137 保留 K1 residual 的正负相位，并用已知相机坐标系把其他相机的残差向量耦合到当前相机。在同一五条 PoolFire 轨迹、3700 个单元和 `2A+2A^T` 候选壳中，严格通过从 v136 的 **3215/3700** 提高到 **3351/3700**：自相位救回 **92** 个，跨相机耦合再独立救回 **44** 个。机制确实增加了容量，但仍有 **349** 个 observation-only 失败、完整轨迹 **0/5**；其中 **343/349** 位于 5 相机，故 v137 仍判失败并关闭。

## 为什么做这一步

v136 只看 residual 平方能量的质心和宽度，丢掉了正负号，也没有真正利用相机之间的方向关系。v137 检验两个更物理、仍可部署的假设：

1. residual 的正负相位是否携带 K4 correction 所需的局部方向信息；
2. 把不同相机的二维 residual 向量通过已知 `right/up` 轴转换到世界坐标，再投影到目标相机，是否能补足跨视角信息。

真值只用于已开封开发集上的容量判定，不进入表示生成、便宜 control、validation 或 test。

## 保持不变的实验合同

- 五条已开封 PoolFire CFD 三维密度轨迹；
- `5/7/9/12` 个活跃相机，支持相机增删和换序；
- clean、观测噪声、旋转、平移、内参、全位姿和 combined 扰动；
- `32x16x16` 三维逆问题和冻结 straight-ray forward；
- 候选 `2A+2A^T`，Zero-CGLS K4 参考 `4A+4A^T`；
- field、full-gradient、interior-gradient、observation 四类指标；
- 逐单元比值不超过 `1.05`，逐轨迹 p90 不超过 `1.02`、worst 不超过 `1.05`；
- 两条 validation 和两条 test 均未打开。

## v137 表示

对每个活跃相机和位移分量，v137 先计算

`phase = tanh(residual / RMS(residual))`。

这个变换有界、保留正负号、对正比例缩放不敏感。随后：

1. 保留 v136 每相机 32 个方向；
2. 加入 8 个由本相机 signed phase 调制的 DCT 方向；
3. 把 peer residual 按已知相机 `right/up` 轴提升为世界向量并对活跃相机集合求平均；
4. 将平均世界向量投影回目标相机，再加入 8 个 peer-phase DCT 方向。

完整表示每相机 48 个方向，对 5/7/9/12 相机分别为 240/336/432/576 维。它严格包含 v136，且对相机换序等变、支持可变相机数量。

重要限制是：peer residual 仍在**相同归一化 detector 像素**上聚合，没有沿真实射线做深度对应或极线重采样。v137 正是在检验这一级几何耦合是否已经足够。

## 主要结果

| 证据 | v135 | v136 | v137 |
|---|---:|---:|---:|
| 四指标严格通过 | 3162/3700 | 3215/3700 | **3351/3700** |
| 相对上一阶段新增 | - | +53 | **+136** |
| 剩余失败 | 538 | 485 | **349** |
| 完整轨迹 | 0/5 | 0/5 | **0/5** |

三类场与梯度均为 **3700/3700**，349 个失败仍全部只超过 observation 门。失败单元的 observation/K4 比值为 p50 **1.07140**、p90-higher **1.11668**、worst **1.15988**。

### 哪部分真的有用

| 增量来源 | 新救回 |
|---|---:|
| 本相机 signed phase | **92** |
| 跨相机坐标耦合的额外贡献 | **44** |
| 合计 | **136** |

因此不能说相位或跨相机信息“没用”。它们确实把 136 个原失败单元推过严格门；但这个增量没有达到预注册的 3700/3700 容量标准。

部署可见的 signed-phase joint least-squares 便宜 control 为 **0/3700**，说明这些方向必须经过 truth-aware 选择才能出现当前容量，尚不能直接成为部署算法。

### 少视角成为决定性瓶颈

| 相机数 | v136 通过 | v137 通过 | 新救回 | 剩余失败 |
|---:|---:|---:|---:|---:|
| 5 | 516/925 | **582/925** | 66 | **343** |
| 7 | 875/925 | **919/925** | 44 | 6 |
| 9 | 920/925 | **925/925** | 5 | 0 |
| 12 | 904/925 | **925/925** | 21 | 0 |

v137 已让 9 相机和 12 相机全部通过，7 相机只剩 6 个失败；但 5 相机仍只有 **62.9%** 通过，并贡献 **98.3%** 的剩余失败。当前核心不再是 residual 相位本身，而是少视角下的几何对应或有效秩。

### 高功率轨迹仍主导失败

| 轨迹 | v136 通过 | v137 通过 | 新救回 | 剩余 | observation p90 / worst |
|---|---:|---:|---:|---:|---:|
| p14-s05 | 737 | 737 | 0 | 3 | 1.03707 / 1.06611 |
| p22-s03 | 717 | 726 | 9 | 14 | 1.04641 / 1.07478 |
| p33-s01 | 690 | 726 | 36 | 14 | 1.04642 / 1.06634 |
| p45-s05 | 507 | 568 | 61 | **172** | 1.06526 / 1.08791 |
| p58-s03 | 564 | 594 | 30 | **146** | 1.09250 / 1.15988 |

p45 与 p58 合计 **318/349** 个失败。clean、噪声、位姿和内参各层都有增益与剩余失败，表明主要问题不是某一种人工扰动，而是稀疏视角面对更复杂三维形态时的跨视图约束不足。

## 独立复算

第二实现没有调用正式 v137 表示函数，重新构造本相机相位、世界坐标 peer 向量、目标相机投影、基函数、候选、物理重放和全部门：

- self phase 最大差：`0`；
- peer phase 最大差：`0`；
- peer residual RMS 最大差：`0`；
- 便宜 control 指标最大差：`3.33e-16`；
- 系数最大差：`1.76e-11`；
- 局部候选指标最大差：`4.55e-15`；
- 最终选择指标最大差：`3.89e-15`；
- summary 最大差：`4.88e-15`；
- 离散数组不一致：`0`；
- 正式结果树与 v136 父证据在验证前后均未变化。

所以 **3351/3700、+136、92+44 的增量分解、349 个 observation-only 失败和 0/5 完整轨迹** 是可信的开发集负结果。

## 决策

1. 关闭 v137 的“同一 detector 像素 + 世界向量平均”跨相机表示，不再调相位尺度、权重或增加 DCT 频带。
2. 不训练 predictor、CNN、FNO、UNO 或 DeepONet，也不租 GPU；容量门尚未通过。
3. 下一门改成物理上不同的**射线重叠/极线输运**：沿目标相机射线设置结果前冻结的深度锚点，通过已知内外参将 3D 锚点投到其他相机，重采样 peer signed residual 后再聚合。这样跨相机信息在几何对应位置交流，而不是在相同像素编号上交流。
4. 新表示仍须相机换序等变、支持 5/7/9/12 相机、保持 `2A+2A^T`，并先过便宜 control 和 truth-aware 3700/3700、5/5 容量门。
5. 只有容量全过，才允许最小 observation/geometry-only predictor；独立公开外门、wall/RSS 和真实 BOST 仍排在其后。

---

# English: v137 Signed-Phase and Cross-Camera Geometry Capacity Diagnostic

## Bottom line

v137 preserves the sign and phase of the deployment-visible K1 residual and couples peer residual vectors through the reported camera coordinate frames. On the same five opened PoolFire trajectories, 3,700 cells, and `2A+2A^T` candidate shell, strict passes rise from **3215/3700** to **3351/3700**. Self phase rescues **92** cells and peer-camera coupling adds **44** more. The mechanism carries real capacity, but **349** observation-only failures remain and complete trajectories are still **0/5**. Because **343/349** failures occur with five cameras, v137 fails its preregistered gate and is closed.

## What changed

Each residual map is converted to a bounded odd phase map with `tanh(residual / RMS)`. Eight self-phase-modulated detector-frequency directions are added per camera. Peer two-component residuals are lifted through each camera's reported `right/up` axes into world vectors, averaged over the active camera set, projected into the target camera frame, and used for another eight directions. Together with the 32-direction v136 parent, the full basis has 48 directions per active camera and strictly contains v136.

The construction is camera-permutation equivariant and supports 5/7/9/12 active cameras. Its deliberate limitation is that peer messages are matched at the same normalized detector coordinate; it does not yet use ray-depth correspondence or epipolar reprojection.

## Evidence

- Strict joint passes: **3351/3700**, a gain of **136** over v136.
- Attribution: **92** self-phase rescues plus **44** peer-geometry incremental rescues.
- Remaining failures: **349**, all observation-only.
- By camera count: **582/925**, **919/925**, **925/925**, and **925/925** for 5/7/9/12 cameras.
- Five-camera cells contribute **343/349** remaining failures.
- p45-s05 and p58-s03 contribute **318/349** failures.
- The deployment-visible signed-phase joint-LS control passes **0/3700**.
- All five complete trajectories still fail.
- Independent recomputation matches selected metrics to `3.89e-15`, with zero discrete-array failures.

## Scientific interpretation

Residual sign and cross-camera pose coupling are useful, but they are not sufficient. The fact that 9- and 12-camera cells all pass while 98.3% of unresolved cells occur with five cameras points to sparse-view geometric correspondence or effective rank, rather than phase alone. The current same-pixel peer aggregation is therefore closed.

The next falsifiable carrier is ray-overlap or epipolar transport. Frozen depth anchors along each target ray will be projected through known intrinsics and extrinsics into peer cameras before signed residuals are sampled and aggregated. The new carrier must remain variable-cardinality, camera-permutation equivariant, and inside the same exact-call shell. Deterministic controls and truth-aware capacity come first; no neural predictor or GPU training is authorized before all 3,700 cells and all five trajectories pass.

This is an independently recomputed post-open proxy negative result. It is not an algorithmic breakthrough, external-generalization result, resource-speedup result, curved-ray validation, paper success, or real-BOST validation.
