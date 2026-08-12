# v142 精确 K1-dual 成对深度暖启动哨兵诊断

> 公开日期：2026-08-13  
> 正式状态：`PENDING_INDEPENDENT_VALIDATION_PASS_K1_DUAL_SENTINEL_V142`  
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_K1_DUAL_SENTINEL_V142`  
> 科学判决：`PASS_K1_DUAL_PAIR_DEPTH_WARM_SENTINEL_V142`  
> 边界：4 个结果前固定的 post-open 困难单元；`algorithm_breakthrough=false`。

## 一句话结论

旧 v141 堆叠预测链在产生预测或评分前因跨 outer-fold 上游泄漏而作废。v142 删除学习上游，直接用精确 CGLS-K1 detector dual、K1 residual 与报告相机几何生成成对深度方向。在 5/7/9/12 相机各一个结果前固定的困难单元上，`3A+3A^T` 的 warm-restart K1 为 **4/4**，最大四指标误差比为 **0.99645**；三个同成本对照均至少有一项越过 `1.05` 门。第二实现从 K1/K4 状态、方向、稳定求解、物理场、观测残差到调用账全部重算，得到同一判决。

这是值得运行全 3700 单元的**机制正信号**，不是算法突破。尤其要保留一个反证边界：更便宜的 initializer-only `2A+2A^T` 在这四个哨兵单元上也为 4/4。因此四单元结果不能证明额外一次 K1 精化已经必要或占优。

## 为什么换路线

v140.4 已证明成对 target-peer、六个 depth bins 与两个角度分支具有全量固定目标容量。原计划随后生成教师并训练堆叠预测器，但静态与运行前审计发现：上游构造跨越了 outer-fold 边界。该链在任何 prediction、score 或科学输出出现前 fail-closed 停止，旧结果目录没有被当作负结果或正结果解释。

v142 因此回答一个更小也更物理的问题：不借助任何 learned upstream，只使用部署时可计算的精确 K1 dual 与 residual，成对深度表示是否仍有能力构造有效暖启动？

## 实际机制与成本

1. 从同一多相机 observation 和报告几何运行精确 CGLS K1，得到 detector-space dual `z1`、K1 场与 residual。
2. 用 `z1`、signed K1 residual 和 target-peer/depth-bin 几何生成成对深度方向；没有 CNN、FNO、教师预测器或跨折上游。
3. 离线容量目标固定为 exact CGLS detector-space correction `z4-z1` 的 projection-only ridge，ridge 为 `1e-6`。
4. CFD 真值只用于 field、full-gradient、interior-gradient、observation 四类指标的否决门，不参与在线载体生成。
5. 主候选从混合 dual 做精确 `A^T` lift，随后运行未修改 warm-restart CGLS K1；完整理论调用账为 `3A+3A^T`，对照 Zero-CGLS K4 为 `4A+4A^T`。

这仍是容量诊断：当前系数由 exact K4 correction 离线求得，部署时还没有 observation/geometry-only predictor。

## 四个哨兵单元

| active cameras | initializer-only 最大比值 | warm K1 最大比值 | Zero-K3 | scaled-BP + K2 | geometry-PCGLS-K3 |
|---:|---:|---:|---:|---:|---:|
| 5 | 1.04729 | **0.99645** | 1.27281 | 1.41091 | 1.15437 |
| 7 | 1.01552 | **0.99057** | 1.21993 | 1.33976 | 1.16159 |
| 9 | 1.00785 | **0.98656** | 1.21351 | 1.36586 | 1.15904 |
| 12 | 1.00281 | **0.97306** | 1.23769 | 1.35567 | 1.18491 |

比值是该单元四个物理误差比的最大值，冻结门为 `<=1.05`。warm K1 与 initializer-only 均为 4/4；三个 `3A+3A^T` 同成本对照都不是 4/4。

因此当前严格结论是：**K1-dual 成对深度表示在四个固定困难单元上有机制 headroom，同成本经典解释没有复现；但更便宜 initializer-only 尚未被排除。**

## 独立复算

独立程序没有复用正式求解与混合 helper，重新构造全部四个单元。formal 与 independent 的最大差为：

- mixed dual 相对差：`1.25e-8`；
- initializer field 相对差：`8.97e-10`；
- warm field 相对差：`8.87e-10`；
- 四指标绝对差：`1.59e-11`；
- 指标比绝对差：`1.84e-11`；
- K1 dual lift 绝对差：`2.78e-16`。

所有物理状态、指标、判决与调用账均通过冻结门。共享的底层 straight-ray physics kernels 仍未做到端到端代码独立，因此不能写成真实 BOST 或 curved-ray 验证。

## 成功、失败与下一门

成功的是路线筛选：去掉泄漏上游后，一个更小、更物理的 K1-dual 载体在四个固定困难单元上成立，足以授权全量 3700 单元试验。

尚未成功的是算法主张：

- 只有 4 个 post-open 单元，不是 3700/3700 或 5/5 完整轨迹；
- exact K4 correction 仍参与离线容量系数，不是部署预测；
- initializer-only 同样 4/4，额外 K1 refinement 的必要性未证；
- 没有 fresh wall/RSS、独立公开外门、curved ray 或真实 BOST。

当前全量门保持同一 3700 单元、5 条轨迹、5/7/9/12 相机和 clean/noise/pose/intrinsics/combined 分层，分别报告逐单元与逐轨迹尾部，并由第二实现重建全部物理场与调用账。只有全量独立通过，才允许冻结最小 observation/geometry-only predictor；在此之前不租 GPU。

---

# English: v142 Exact K1-Dual Pair-Depth Warm-Start Sentinel

The prior v141 stacked-predictor path was invalidated before any prediction or score because its upstream construction crossed outer-fold boundaries. v142 removes that learned upstream. The exact CGLS-K1 detector dual, signed K1 residual, and reported camera geometry generate a pair-depth basis; an offline projection-only ridge target fits the exact detector-space correction `z4-z1` with ridge `1e-6`.

On one preregistered difficult post-open cell at each of 5, 7, 9, and 12 active cameras, the `3A+3A^T` warm-restart K1 primary passes **4/4**, with a maximum four-metric ratio of **0.99645** to Zero-CGLS K4. Zero-K3, scaled-BP plus K2, and geometry-PCGLS-K3 each fail at least one same-cost gate. A second implementation independently rebuilds all states, directions, solves, physical fields, residuals, metrics, decisions, and call ledgers; the maximum metric-ratio difference is `1.84e-11`.

One important lower-cost counterfactual remains: the initializer-only `2A+2A^T` arm also passes all four sentinel cells, with a worst ratio of `1.04729`. The four-cell diagnostic therefore establishes mechanism headroom and authorizes a full-roster audit, but it does not establish that the extra K1 refinement is necessary or superior.

The ongoing next gate applies the unchanged no-learned-upstream mechanism to all 3,700 opened PoolFire cells, five complete trajectories, 5/7/9/12 cameras, and clean/noise/pose/intrinsics/combined strata, with independent reconstruction of every physical state, tail statistic, and call receipt. Until that gate passes, there is no deployable predictor, algorithmic breakthrough, resource speedup, external generalization, curved-ray validation, real-BOST result, or paper success.
