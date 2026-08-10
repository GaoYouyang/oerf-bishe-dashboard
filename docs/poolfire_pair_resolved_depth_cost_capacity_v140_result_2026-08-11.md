# v140 成对深度代价表示 Stage A 容量诊断

> 公开日期：2026-08-11
>
> 正式状态：`PASS_V140_STAGE_A_HARD_FAILURE_CAPACITY`
>
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_PAIR_RESOLVED_DEPTH_COST_V140`
>
> 边界：只证明 v139 的 151 个固定硬失败存在 truth-aware 表示容量；Stage B 尚未运行，`algorithm_breakthrough=false`。

## 一句话结论

v140 保留完整 v139 三维射线一致性父表示，并把目标相机、peer 相机、六个固定双样本深度 bin 和两个角度分支分别编码。在 v139 已开封开发集的 **151 个固定硬失败**上，truth-aware Stage A 为 **151/151** 找到通过四指标与八道门的候选，廉价 deployment-visible joint-LS 对照为 **0/151**；第二实现从射线、重投影、方向、求解到物理重放全部复算后得到同一判决。它证明新的成对深度载体有机制容量，但**尚不证明全量 3700/3700、完整轨迹 5/5、可部署预测器或算法成功**。

## 为什么做这一步

v139 将 signed K1 residual 回投到三维体后，已把严格逐单元通过提高到 3549/3700，但余下 151 个失败全部来自 5 相机并只越过 observation 门。固定 P1/P2 深度矩会把不同 peer camera、不同深度假设和不同三角测量角过早压缩。v140 因此检验一个更具体的问题：如果不再提前平均这些身份，当前 `2A+2A^T` warm-start 壳是否仍有足够自由度达到 K4 的冻结精度门。

## 表示与成本

1. 完整保留 v139 的每相机 72 个方向，保证新表示严格包含父空间。
2. 对每个有序 target-peer 相机对，沿目标报告射线取 12 个固定样本并划分为 6 个双样本 depth bins。
3. 把样本重投影到 peer detector，双线性读取 signed K1 residual；分别保留常数角分支与 centered-sine-squared 角分支。
4. 新增方向数为 `24*C*(C-1)`，总方向数为 `72*C + 24*C*(C-1)`；5 相机时为 840 个方向。
5. 表示生成只读部署可见 K1 residual 与报告几何，对相机换序等变并支持可变相机数；但本次容量系数由已开封真值辅助求取，所以不是部署算法。

候选在线精确调用账仍是 `2A+2A^T`，Zero-CGLS K4 参考为 `4A+4A^T`。离线容量搜索成本不等于部署成本。

## Stage A 结果

| 项目 | 结果 |
|---|---:|
| v139 固定硬失败 | 151 |
| v140 Stage A 评估 | 151 |
| 严格通过 | **151/151** |
| 剩余 | **0** |
| 廉价 joint-LS 对照 | **0/151** |
| Stage B active tail | **2199，待运行** |
| 全量 3700/3700 已证明 | **否** |
| 完整轨迹 5/5 已证明 | **否** |

151 个通过候选中，132 个由 projection-only 目标选中；其余由固定权重 1、4、16、64 分别选中 4、6、7、2 个。权重 256、1024、v139 父端点和便宜对照均未被选中。151 个候选全部在“所有物理门可行后最小化 minimax”分支内找到，不依赖事后放宽门槛。

在这 151 个硬单元上，候选相对 K4 的 field / full-gradient / interior-gradient / observation worst 比值分别为 **1.00729 / 1.00962 / 1.00888 / 1.02930**，均位于冻结的逐单元门内。便宜对照的对应 worst 为 **1.18914 / 1.08073 / 1.05588 / 1.90159**，因此 0/151。

## 独立复算

第二实现不导入 v140 正式 core 或 runner，独立重建 target rays、peer reprojection、双线性采样、signed phase、840 个方向、eigensolve、候选、物理重放、八门、分层和调用回执。最终结果：

- selected metric 最大差：`8.69e-12`；
- cheap-control metric 最大差：`3.60e-12`；
- pair diagnostic 最大差：`8.33e-16`；
- selected quantile 最大差：`4.29e-12`；
- 非唯一系数最大差：`2.63e-7`；
- condition number 相对差：`1.75e-8`；
- 影响科学判决的精确数组不一致：`0`；
- 调用回执不一致：`0`；
- 正式结果与父证据在验证前后未改变。

共享冻结 physics kernels 仍被正式和独立路径共同使用，所以 `end_to_end_physics_independence_proven=false`。

## 数值审计披露

独立验证不是一次无波折地通过。第一次正式验证前检查发现合同字段层级解析错误，因此在读数据前停止。后续一次完整复算已经匹配所有科学指标、选择、门、计数与回执，但旧的统一绝对容差把最高约 `2.00e9` 条件数的诊断值和科学输出混在一起，导致 fail-closed。

我们没有改候选、门槛、样本、选择器或科学容差。诊断原因明确后，先公开冻结分型数值审计：科学/表示数组仍用 `2e-8` 绝对容差，所有改变判决的离散数组仍须完全相等；只有 condition number 改用 `1e-7` 相对容差，非唯一系数使用 `2e-6` 绝对容差。随后从头完整复算，最终通过。这个 post-open repair 是审计修复，不是性能调参，必须随结果一起披露。

## 科学解释与下一门

这项结果是真实的机制进步：v139 的最后 151 个硬失败并非因为 `2A+2A^T` 壳天然没有容量，而是因为父表示过早压缩了 target-peer 与深度结构。成对深度代价表示在固定硬集上补齐了这一容量。

但 Stage A 是刻意缩小的 hard-set screen，不能和 v139 父结果直接拼成“3700/3700 已通过”。下一步必须结果前单独冻结 Stage B，在固定 **2199 个 v139 active-tail 单元**上运行同一表示，再与封存父结果合并并独立复算完整 3700 个单元和 5 条轨迹。只有达到 **3700/3700 与 5/5**，才允许冻结一个最小 permutation-equivariant observation/geometry-only 系数预测器。

因此当前仍然：不训练 CNN/FNO/UNO/DeepONet，不租 GPU，不启动 wall/RSS、外部门或真实 BOST 声明；`algorithm_breakthrough=false`。

---

# English: v140 Pair-Resolved Depth-Cost Stage-A Capacity Diagnostic

## Bottom line

v140 retains the complete v139 3D ray-consistency parent basis and separately encodes ordered target-peer identities, six fixed two-sample depth bins, and two angle branches. On the fixed set of **151 v139 hard failures**, truth-aware Stage A finds **151/151** candidates that pass all frozen cellwise metrics and gates, while the cheap deployment-visible joint-LS control passes **0/151**. A second implementation independently rebuilds the rays, reprojection, features, basis, solves, physical replay, gates, and receipts and reaches the same decision.

This is mechanism-capacity evidence, not a deployable algorithm. The representation itself uses only deployment-visible K1 residuals and reported geometry and remains permutation equivariant for variable camera counts, but the Stage-A coefficients are truth aware. The candidate shell remains `2A+2A^T`, compared with `4A+4A^T` for Zero-CGLS K4.

The independent maximum selected-metric difference is `8.69e-12`, all science-changing discrete arrays match exactly, and there are no call-receipt mismatches. A post-open typed numerical audit is disclosed: a prior full recomputation matched every scientific decision but rejected high-condition-number solver diagnostics under an inherited absolute tolerance. Scientific thresholds, samples, candidates, selectors, and exact-array requirements were unchanged before a fresh full recomputation passed.

Stage A does **not** prove 3,700/3,700 cells or 5/5 complete trajectories. It only authorizes a separately frozen Stage B over the fixed 2,199-cell v139 active tail. A minimal observation/geometry-only coefficient predictor remains unauthorized until Stage B and its independent recomputation establish both full-cell and complete-trajectory gates.

Accordingly, this is not an algorithmic breakthrough, paper success, external-generalization result, resource-speedup result, curved-ray validation, or real-BOST validation.
