# v138 真实射线重叠与极线相位容量诊断

> 公开日期：2026-08-10
>
> 正式状态：`FAIL_V138_RAY_OVERLAP_EPIPOLAR_CAPACITY`
>
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_RAY_OVERLAP_EPIPOLAR_V138_3`
> 边界：`algorithm_breakthrough=false`，不授权预测器、GPU、资源门、外门或真实 BOST 声明。

## 一句话结论

v138 不再把各相机的同编号像素当成同一三维位置，而是重建报告相机射线，沿目标射线放置三个固定深度锚点，投到 peer 相机的极线对应位置采样 signed residual。在同一 3700 个单元和 `2A+2A^T` 壳中，严格通过从 **3351/3700** 提高到 **3397/3700**，新救回 **46** 个；但仍剩 **303** 个 observation-only 失败、完整轨迹 **0/5**，故按预注册规则关闭该表示。

## 为什么做这一步

v137 证明 residual 正负相位和跨相机坐标耦合有用，但它在各相机的相同归一化 detector 像素上交换信息，并不代表真实三维对应。v138 专门检验：用已知内外参恢复真实射线重叠与极线采样，是否足以修复 v137 的 349 个剩余失败。

保持不变的合同包括：五条已开封 PoolFire 三维密度轨迹、5/7/9/12 个活跃相机、37 种 clean/噪声/位姿/内参/组合条件、`32x16x16` 逆问题、Zero-CGLS K4 参考、四类误差与八道门。validation 和 test 真值仍未打开。

## 表示与成本

1. 从每个相机的 18 维报告几何重建 finite-source 或 orthographic 射线。
2. 射线被截断在重建立方体内，用三个固定 Gauss-Legendre 深度锚点。
3. 锚点通过 peer 相机内外参重投影，在对应极线位置双线性采样 signed K1 residual。
4. peer residual 被提升到世界坐标，对有效 peer/深度对称平均，再投回目标像素局部射线基。
5. 它调制四个 detector 频带，在 v137 父表示上每相机增加 8 个方向，总计每相机 56 个。

表示对相机换序等变，支持可变相机数，严格包含 v137。候选在线仍是 `2A+2A^T`，K4 参考为 `4A+4A^T`。真值只用于已开封开发集的容量上界，不是可部署系数预测器。

## 主要结果

| 证据 | v136 | v137 | v138 |
|---|---:|---:|---:|
| 四指标严格通过 | 3215/3700 | 3351/3700 | **3397/3700** |
| 相对上一阶段新增 | +53 | +136 | **+46** |
| 剩余失败 | 485 | 349 | **303** |
| 完整轨迹 | 0/5 | 0/5 | **0/5** |

field、full-gradient 和 interior-gradient 仍全部为 **3700/3700**。303 个失败只越过 observation 门，其 observation/K4 比值为 p50 **1.06822**、p90-higher **1.09100**、worst **1.11956**。部署可见的 ray-overlap joint-LS 便宜对照为 **0/3700**。

### 少视角仍是决定性瓶颈

| 相机数 | v137 通过 | v138 通过 | 新救回 | 剩余 |
|---:|---:|---:|---:|---:|
| 5 | 582/925 | **627/925** | 45 | **298** |
| 7 | 919/925 | **920/925** | 1 | 5 |
| 9 | 925/925 | **925/925** | 0 | 0 |
| 12 | 925/925 | **925/925** | 0 | 0 |

v138 的 46 个救回中有 45 个位于 5 相机，说明真实射线对应确实有用；但 **298/303** 剩余失败仍是 5 相机，表示距离完整容量还很远。

### 轨迹尾部

| 轨迹 | v137 通过 | v138 通过 | 新救回 | 剩余 | observation p90 / worst |
|---|---:|---:|---:|---:|---:|
| p14-s05 | 737 | 738 | 1 | 2 | 1.03707 / 1.05937 |
| p22-s03 | 726 | 727 | 1 | 13 | 1.04641 / 1.07409 |
| p33-s01 | 726 | 726 | 0 | 14 | 1.04642 / 1.05896 |
| p45-s05 | 568 | 610 | **42** | **130** | 1.06056 / 1.07944 |
| p58-s03 | 594 | 596 | 2 | **144** | 1.07648 / 1.11956 |

p45 对射线几何最敏感，贡献 42/46 个救回；但 p45/p58 仍合计占 274/303 个剩余失败。这更像复杂三维形态下的深度混叠/少视角约束不足，不是单一噪声或某类标定误差。

## 独立复算与审计修复

第二实现独立重建射线、极线采样、世界向量聚合、基、候选、物理重放与全部门：

- ray phase 最大差：`1.84e-14`；
- 有效 peer 比例与权重边界差：`0`；
- 系数最大差：`5.97e-11`；
- 局部候选指标最大差：`7.77e-15`；
- 最终选择指标最大差：`6.99e-15`；
- 离散数组不一致：`0`；
- 正式结果树和父证据未被改动。

初始独立复算曾两次 fail-closed：第一次是病态基下 direct solve 与 eigensolve 的求解器诊断容差过严，第二次是修复函数误把 6 列诊断写成 5 列。两次都在发出独立结果之前中止并保留证据。最终 v138.3 不改算法重跑；三次正式运行的 15 个数值数组全部逐字节一致，再由独立实现通过修订后的数值审计。

## 科学解释与决策

真实射线对应不是无效的，它救回了 46 个单元，尤其改善 p45。但 v138 在生成最终 detector 方向前，已将 peer 身份和三个深度锚点平均。对少视角问题，这会把“哪个深度、哪组相机相互支持或冲突”压缩掉。

因此：

1. 关闭 v138 的 GL3 射线平均表示，不再调锚点数、相位尺度或 Pareto 权重。
2. 不训练 predictor/CNN/FNO/UNO/DeepONet，不租 GPU，不启动 wall/RSS 或外门。
3. 下一可证伪载体是**深度分辨的三维射线一致性体**：将每相机 signed residual 沿真实射线回投到冻结粗体素网格，分别保留跨相机一致均值与冲突/方差，再投回 detector 方向。
4. 新载体仍必须换序等变、支持 5/7/9/12 相机、保持 `2A+2A^T`，并先过便宜 control 与 truth-aware 3700/3700、5/5 容量门。

---

# English: v138 Geometry-Faithful Ray-Overlap and Epipolar-Phase Capacity Diagnostic

## Bottom line

v138 reconstructs each reported camera ray, places three fixed depth anchors on the target ray, reprojects those anchors into peer cameras, and samples signed residuals at geometrically corresponding epipolar locations. On the same 3,700 opened PoolFire cells and the same `2A+2A^T` candidate shell, strict passes rise from **3351/3700** to **3397/3700**, rescuing **46** cells. Yet **303** observation-only failures remain and complete trajectories stay at **0/5**. The representation therefore fails its preregistered capacity gate and closes.

All field and gradient metrics pass 3,700/3,700. Of the unresolved cells, **298/303** occur with five cameras and **274/303** belong to p45 or p58. The deployment-visible ray-overlap joint-LS control passes **0/3700**. Exact ray correspondence is incrementally useful, especially for p45, but the gain is far too small to authorize predictor training.

The likely missing structure is depth-resolved multiview consistency. v138 averages peer identity and all three depth anchors before constructing its final detector directions, erasing which depths and camera subsets agree or conflict. The next falsifiable carrier therefore backprojects signed residuals into a frozen coarse 3D volume, retains camera-set consensus and disagreement separately at each voxel, and projects those depth-resolved fields back into detector directions. It must remain variable-cardinality, camera-permutation equivariant, and inside the same exact-call shell.

An independent implementation rebuilt the geometry, epipolar sampling, basis, candidates, physical replay, and every gate. The maximum selected-metric difference is `6.99e-15`, with zero discrete-array failures. Two earlier validation attempts stopped fail-closed at audit-layer checks and emitted no validated scientific result; after versioned repairs, three unchanged formal runs produced all 15 numeric arrays byte-for-byte identically and v138.3 passed independent recomputation.

This is a trustworthy post-open proxy negative result. It is not an algorithmic breakthrough, external-generalization result, resource-speedup result, curved-ray validation, paper success, or real-BOST validation.
