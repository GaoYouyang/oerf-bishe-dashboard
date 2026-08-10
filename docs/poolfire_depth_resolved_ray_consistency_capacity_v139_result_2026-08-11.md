# v139 深度分辨三维射线一致性容量诊断

> 公开日期：2026-08-11
>
> 正式状态：`FAIL_V139_DEPTH_RESOLVED_RAY_CONSISTENCY_CAPACITY`
>
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_DEPTH_RESOLVED_RAY_CONSISTENCY_V139_3`
>
> 边界：`algorithm_breakthrough=false`，不授权预测器、GPU、wall/RSS、外门或真实 BOST 声明。

## 一句话结论

v139 将每路 signed K1 residual 沿报告射线回投到 `16x8x8` 粗体素网格，逐体素保留 leave-one-camera-out 一致均值、agreement、normalized disagreement 与 visibility，再沿目标射线读出固定 P1/P2 深度矩。在同一 3700 个已开封 PoolFire 单元和 `2A+2A^T` 壳中，严格通过从 **3397/3700** 提高到 **3549/3700**，新救回 **152** 个；但仍剩 **151** 个 observation-only 失败、完整轨迹 **0/5**，所以固定 P1/P2 深度矩表示按预注册门关闭。

## 为什么做这一步

v138 已证明真实射线对应有用，但它把三个深度锚点和 peer camera 身份过早平均。v139 检验一个更强的物理载体：把 signed residual 放回三维空间，在体素层面区分跨相机一致与冲突，再投影回 detector 方向，是否足以解决少视角深度歧义。

保持不变的合同包括：五条已开封 PoolFire 三维密度轨迹、5/7/9/12 个活跃相机、37 种 clean/噪声/位姿/内参/组合条件、`32x16x16` 逆问题、Zero-CGLS K4 参考、四类误差与八道门。validation 和 test 真值仍未打开。

## 表示与成本

1. 用报告的每相机 18 维几何重建 finite-source 或 orthographic rays。
2. 沿每条射线取 12 个固定中点样本，将 signed K1 residual 向量三线性 splat 到 `16x8x8` 体素网格。
3. 每个目标相机都只聚合其他活跃相机，形成 leave-one-camera-out consensus、agreement、normalized disagreement 和 visibility。
4. 沿目标射线读回这些体场，并用固定 P1 奇深度矩与 P2 偶深度矩形成新 detector-space 方向。
5. 每相机在 v138 父表示上增加 16 个方向，总计 72 个；5/7/9/12 相机对应 360/504/648/864 个方向。

表示严格包含 v138、对相机换序等变，并支持可变相机数。候选在线仍为 `2A+2A^T`，K4 参考为 `4A+4A^T`。真值只用于已开封开发集的容量上界，不是可部署系数预测器。

## 主要结果

| 证据 | v137 | v138 | v139 |
|---|---:|---:|---:|
| 四指标严格通过 | 3351/3700 | 3397/3700 | **3549/3700** |
| 相对上一阶段新增 | +136 | +46 | **+152** |
| 剩余失败 | 349 | 303 | **151** |
| 完整轨迹 | 0/5 | 0/5 | **0/5** |

field、full-gradient 和 interior-gradient 仍全部为 **3700/3700**。151 个失败只越过 observation 门，其 observation/K4 比值为 p50 **1.05891**、p90-higher **1.07272**、worst **1.08672**。部署可见的 depth-resolved joint-LS 便宜对照为 **0/3700**。

### 5 相机成为唯一逐单元瓶颈

| 相机数 | v138 通过 | v139 通过 | 新救回 | 剩余 |
|---:|---:|---:|---:|---:|
| 5 | 627/925 | **774/925** | **147** | **151** |
| 7 | 920/925 | **925/925** | 5 | 0 |
| 9 | 925/925 | **925/925** | 0 | 0 |
| 12 | 925/925 | **925/925** | 0 | 0 |

v139 消除了 7/9/12 相机的全部逐单元失败，且 152 个救回中 147 个来自 5 相机，说明深度分辨的一致/冲突体确实抓住了主要缺失结构。但余下 151 个失败也全部来自 5 相机，容量门仍未完成。

### 轨迹尾部

| 轨迹 | v138 通过 | v139 通过 | 新救回 | 剩余 | observation p90 / worst |
|---|---:|---:|---:|---:|---:|
| p14-s05 | 738 | 740 | 2 | 0 | 1.03707 / 1.04996 |
| p22-s03 | 727 | 734 | 7 | 6 | 1.04604 / 1.06268 |
| p33-s01 | 726 | 740 | 14 | 0 | 1.04562 / 1.04993 |
| p45-s05 | 610 | 667 | 57 | **73** | 1.04997 / 1.06961 |
| p58-s03 | 596 | 668 | 72 | **72** | 1.04996 / 1.08672 |

p14 与 p33 已实现 740/740 逐单元通过，但完整轨迹门还要求 observation p90 不超过 1.02，因此五条轨迹仍全部失败。p45/p58 合计占 145/151 个剩余失败，说明问题不只是单帧离群，而是高功率复杂形态与低角度冗余共同导致的深度不可辨识。

## 独立复算与审计边界

第二实现没有导入 v139 正式 core 或 runner，独立重建 ray clipping、三线性 splat、leave-one-camera-out 统计、P1/P2 readout、候选求解、物理重放、3700 个门、失败分层与调用账。最终核对结果包括：

- selected metric 最大差：`1.54e-11`；
- local candidate metric 最大差：`1.58e-11`；
- trajectory summary 最大差：`1.04e-11`；
- 非唯一系数最大差：`2.66e-8`，但离散选择完全一致；
- solver stationarity 最大值：`1.49e-14`；
- physical solver diagnostic 最大差：`9.75e-9`；
- 精确数组不一致：`0`；
- 正式结果树和父证据在验证前后未改变。

一次早期 formal 因手抄 camera-mask 摘要与封存输入不一致而作废，随后完整重跑。先前独立复算已经匹配全部科学指标、离散选择、门、计数和失败分层，但旧统一容差把物理输出、表示诊断与病态嵌套基中的非唯一系数混在一起，因而 fail-closed。v139.3 在重新完整复算前冻结分型数值合同：科学指标仍用 `1e-9`，轨迹摘要用 `1e-10`，所有改变科学判决的离散量要求严格相等。最终通过的是独立复算可信度，不是算法性能门。

## 科学解释与决策

v139 是目前这条表示链中最大的单步增量之一：它将失败数从 303 减到 151，并消除了所有 7/9/12 相机逐单元失败。这证明“把跨视角 signed residual 放回三维空间、区分一致与冲突”有物理价值。

但它仍把目标射线上的深度结构压缩成固定 P1/P2 两个低阶矩，也没有保留 peer baseline 或三角测量角的身份。在 5 相机下，同一条目标射线可能存在多峰或互相竞争的深度支持；两个低阶矩无法表达这种多假设结构。

因此：

1. 关闭 v139 的固定 P1/P2 深度矩表示，不再调体素网格、矩尺度或 projection 权重。
2. 不训练 predictor/CNN/FNO/UNO/DeepONet，不租 GPU，不启动 wall/RSS 或外门。
3. 下一可证伪载体改为**target-ray 条件的多假设深度代价体**：保留固定 depth bins，并按 peer residual 的符号一致性、冲突和三角测量角/基线分层，而不是压成两个低阶矩。
4. 新载体仍必须只读部署可见 residual 与报告几何、换序等变、支持 5/7/9/12 相机、保持 `2A+2A^T`，并先过便宜 control 与 truth-aware `3700/3700`、`5/5` 容量门。

---

# English: v139 Depth-Resolved 3D Ray-Consistency Capacity Diagnostic

## Bottom line

v139 splats each camera's signed K1 residual vector along its reported rays into a `16x8x8` coarse volume, retains leave-one-camera-out consensus, agreement, normalized disagreement, and visibility per voxel, and reads those fields back along target rays through fixed P1/P2 depth moments. On the same 3,700 opened PoolFire cells and the same `2A+2A^T` candidate shell, strict passes rise from **3397/3700** to **3549/3700**, rescuing **152** cells. Yet **151** observation-only failures remain and complete trajectories stay at **0/5**, so the fixed P1/P2 depth-moment representation fails its preregistered capacity gate and closes.

All field and gradient metrics pass 3,700/3,700. Every seven-, nine-, and twelve-camera cell now passes; 147 of the 152 rescues occur with five cameras. However, all 151 unresolved cells also use five cameras, and p45/p58 account for 145 of them. The deployment-visible depth-resolved joint-LS control passes 0/3,700.

An independent implementation rebuilt the ray geometry, trilinear splat, leave-one-camera-out fields, depth readout, candidates, physical replay, gates, strata, and exact-call receipts. Its maximum selected-metric difference is `1.54e-11`, every science-changing discrete array matches exactly, and all bound evidence trees remain unchanged. A typed audit contract was frozen before this fresh full recomputation so that strict physical metrics were not conflated with non-unique coefficients in a high-condition nested basis.

The result provides strong mechanism evidence but not algorithmic success. Depth-resolved multiview consensus is materially useful, yet compressing target-ray structure into two low-order moments still loses multimodal depth support and peer-baseline identity under five-camera conditions. The next falsifiable carrier is a target-ray-conditioned multi-hypothesis depth cost volume that retains fixed depth bins, signed peer agreement or conflict, and triangulation-angle or baseline strata. It must pass all 3,700 cells and all five trajectories before any predictor or GPU training is authorized.

This is an independently validated post-open proxy negative result. It is not an algorithmic breakthrough, external-generalization result, resource-speedup result, curved-ray validation, paper success, or real-BOST validation.
