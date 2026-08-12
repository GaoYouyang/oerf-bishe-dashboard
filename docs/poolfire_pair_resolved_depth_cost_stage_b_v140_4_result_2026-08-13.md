# v140.4 成对深度代价全量容量与固定目标判决

> 公开日期：2026-08-13
>
> 正式状态：`PASS_V140_4_STAGE_B_HIERARCHICAL_STABLE_CAPACITY`
>
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_PAIR_RESOLVED_DEPTH_COST_STAGE_B_V140_4`
> 边界：固定目标容量已过；全量教师与部署预测器未完成，`algorithm_breakthrough=false`。

## 一句话结论

v140.4 在五条已开封 PoolFire 轨迹、`3700` 个物理单元、`5/7/9/12` 相机和七类 clean / noise / pose / intrinsic 条件上完成了全量容量门。固定 `2199` 个 active-tail 单元为 **2199/2199**，与父证据合并后为 **3700/3700**，五条完整轨迹为 **5/5**；独立程序重建新增方向、稳定求解、物理图像、四指标尾部与调用回执后得到同一判决。

更关键的是，结果出现前预注册的唯一固定目标 `pair_depth_projection_only` 自身也是 **3700/3700、5/5**。该目标使用同一观测与报告几何下的精确 CGLS detector-dual 差 `z4-z1`，不使用 CFD 场真值选目标；真值只负责否决 field / gradient / observation 失真。因此逐单元 truth-aware 目标选择器可以被移除。

这是当前主线一个明确的机制里程碑，但仍不是可部署学习算法。Stage B 保存的系数只覆盖由真值定义的 `2199` 个 active-tail 单元，部署时不能知道这张成员表。下一门必须先为全部 `3700` 个单元重建并独立验证同一个固定教师目标，再允许最小预测器做整轨迹留一外折。

## 为什么这一步重要

v140 Stage A 只在 v139 的 `151` 个固定硬失败上证明了新表示有容量。它回答“值得不值得继续”，却不能回答“全量是否安全”。v140.4 因此保持样本、方向、候选、四指标、八门和 `2A+2A^T` 在线壳不变，只把病态的 `1e-10` ridge 改成结果前由四个相机数 sentinel 验证过的 `1e-6` 稳定求解，然后运行固定的完整 Stage B。

同时，固定目标在正式全量结果出现前单独预注册。这样即使七个候选都表现不错，也不能在看见 v140.4 结果后挑一个更容易预测的目标。

## 正式与独立结果

| 门 | 正式结果 | 独立结果 |
|---|---:|---:|
| Stage A 固定硬集 | 151/151 | 已冻结通过 |
| Stage B active tail | **2199/2199** | **2199/2199** |
| 合并全量 | **3700/3700** | **3700/3700** |
| 完整轨迹 | **5/5** | **5/5** |
| 廉价 joint-LS Stage B | **0/2199** | **0/2199** |
| 剩余 5/7/9/12 相机失败 | 0/0/0/0 | 0/0/0/0 |

active tail 由 `878 / 592 / 423 / 306` 个 5 / 7 / 9 / 12 相机单元组成；clean、noise、rotation、translation、intrinsics、pose-all 与 combined 各层剩余失败均为 `0`。

独立程序不导入正式方向、求解或 canonicalization helper。它从已经开封的原始 `rho`、观测、报告几何与 K1 residual 重建 Stage B。两种求解器在完整物理 field / projection 图像上的归一化最大差分别为 `4.49e-9` 与 `4.46e-9`，低于结果前冻结的 `1e-6` 物理容差；固定候选逐轨迹摘要最大差为 `4.23e-12`。影响科学判决的精确数组失败为 `0`，K1、selected、cheap 与离线 batch 调用回执失败均为 `0`。

共享冻结 physics kernels 仍存在，因此 `end_to_end_physics_independence_proven=false`。

## 预注册固定目标

固定目标 `pair_depth_projection_only` 使用：

1. 部署可见 observation 与报告 geometry；
2. 同一观测下 zero-start CGLS 的 detector-dual 差 `z4-z1`；
3. v140 冻结的成对 target-peer、六 depth bins、两个角度分支方向；
4. projection-only loss 与 `1e-6` mean-Gram-diagonal ridge；
5. CFD truth 只用于四指标最终否决门，不进入目标定义。

它在 active tail 为 `2199/2199`，合并全量为 `3700/3700`，五条轨迹均通过 p90-higher `<=1.02` 与 worst `<=1.05`。五条轨迹的四指标 worst 最大值都没有越过 `1.05`；其中全局最高值是 p22 observation 的 `1.03020`。

这意味着 v140 的容量不依赖“每个单元看真值挑一个目标”。未来预测器只需学习一个固定、物理含义清楚的 detector-space 缺失修正目标。

## 当前为何仍不训练模型

不能把 `2199` 个 Stage-B 系数直接拿来训练。active-tail 身份由 v139 的真值指标定义；若训练或预测时使用它，就等于把不可见真值泄漏成一个分流开关。其余 `1501` 个单元继承父候选，也不是同一个固定教师目标。

因此已经冻结下一道更严格的屏障：

- 对全部 `3700` 个单元，用同一 observation / geometry / K1 residual 生成方向；
- 对全部单元求同一个 `pair_depth_projection_only` 教师系数；
- 不允许 active-tail 成员标签、轨迹 ID、扰动标签、未来帧或 CFD truth 进入生成、训练或预测；
- 正式与第二实现都必须重建并封存完整教师；
- 屏障通过后，才比较零参数 residual joint-LS、共享线性 direction-token ridge 和不超过 `1.6 万` 参数的最小 Deep Sets 方向预测器；
- 使用五条完整轨迹留一，每折 `2960` 个 fit、`740` 个 held-out，归一化、正则与停止全部只在 fit 轨迹内；
- 任一轨迹、尾部、harm 或同价控制支配门失败，就关闭该路线，不用 CNN/FNO/UNO/DeepONet 放大挽救。

## 成本与论文边界

容量候选的逻辑在线壳仍是 `2A+2A^T`，Zero-CGLS K4 为 `4A+4A^T`。但本轮 truth-free teacher 生成与离线容量求解使用了额外算子工作，不能拿来宣称部署加速。

目前只允许写：**在限定的、已开封 PoolFire straight-ray 代理中，一个结果前预注册且不依赖 CFD truth 的固定 detector-space 目标，在全量 3700 单元与五条轨迹上具有独立复算的表示容量。**

目前不能写：学习算法成功、GPU 训练成功、wall/RSS 加速、外部泛化、curved-ray 验证、真实 BOST、SOTA 或论文成功。`algorithm_breakthrough=false`。

---

# English: v140.4 Full-Roster Pair-Resolved Depth-Cost Capacity and Fixed-Target Decision

## Bottom line

v140.4 completes the full capacity gate on five opened PoolFire trajectories, `3,700` physical cells, `5/7/9/12` active cameras, and seven clean/noise/pose/intrinsic profiles. The fixed active tail passes **2,199/2,199**; after merging the sealed parent evidence, the result is **3,700/3,700** and **5/5** complete trajectories. A separate implementation rebuilds the new directions, stable solve, physical images, four metric families, trajectory tails, and call receipts and reaches the same decision.

The sole target preregistered before the full result, `pair_depth_projection_only`, also passes **3,700/3,700 and 5/5**. It fits the frozen direction roster to the exact CGLS detector-dual correction `z4-z1` derived from the same observation and reported geometry. CFD truth only rejects field, gradient, or observation harm; it does not define or select the target. A per-cell truth-aware objective selector is therefore unnecessary for capacity.

The independent normalized differences in full physical field and projection images are `4.49e-9` and `4.46e-9`, below the frozen `1e-6` tolerance. The maximum fixed-candidate trajectory-summary difference is `4.23e-12`; all science-changing exact arrays and call receipts pass.

This is a genuine mechanism milestone, not a deployable learned algorithm. The stored Stage-B coefficients cover only the truth-defined 2,199-cell active tail. Deployment cannot observe that membership bit, and the other 1,501 cells inherited a different parent candidate. The next gate therefore rebuilds and independently verifies the same fixed teacher target for all 3,700 cells before any model fitting.

After that barrier, five complete-trajectory leave-one-out folds may compare a zero-parameter rule, a shared linear direction-token ridge, and one at-most-16k-parameter permutation-equivariant Deep Sets predictor. Inputs are restricted to observations, K1 residuals, the observation-only correction dual, reported camera tokens, and active masks. Any trajectory, tail, harm, or same-cost dominance failure closes the route without a larger-model rescue.

The logical candidate shell remains `2A+2A^T` versus `4A+4A^T` for Zero-CGLS K4, but offline teacher generation and capacity search are not deployment costs. Resource speedup, external generalization, curved-ray performance, real BOST, and paper success remain unproven; `algorithm_breakthrough=false`.
