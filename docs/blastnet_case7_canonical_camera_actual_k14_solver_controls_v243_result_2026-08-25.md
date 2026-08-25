# v243：相机 ID 规范化后，实际未修改 K14 warm solver 在已开封 Case 7 全部过门

## 为什么做

v241 已经证明，更新前 FIFO16 cache 加当前 K1-K14 方向的空间具备逐指标必要容量，但它允许四个指标使用四组真值可见系数，因此还不是一个实际解。下一步必须直接检验实际未修改的 K14 solver，并和同价或更便宜的经典 controls 比较。

第一次 v242 与勘误 v242.1 没有给出可解释的科学判决。相机顺序反转后，所有离散通过/失败判决都一致，但浮点累加顺序造成的场、残差或指标差超过结果前冻结的数值门。v242 的指标漂移为 **1.24e-8**；v242.1 的场与恢复残差漂移为 **1.90e-8 / 1.19e-6**。这两次记录继续保持 `INCONCLUSIVE`，没有通过放宽容差把它们改写为成功。

## 实际做了什么

v243 只解决实现歧义：每次先按稳定 ASCII 相机 ID 排序，并把各相机连续观测块按同一顺序重排。这个适配器不做数值运算，不改变 dtype，也不增加 `A/A^T`。原始顺序和反转顺序由此生成逐字相同的规范相机 ID 与观测。

随后在同一 13 条已开封 Case 7 rig、每条 42 帧、共 546 个单元上运行：

- 主候选：因果更新前 FIFO16 warm state + 实际未修改 geometry-Jacobi PCGLS K14；
- Zero geometry-Jacobi PCGLS K14；
- Zero unpreconditioned CGLS K14；
- normalized-backprojection geometry-Jacobi PCGLS K13；
- normalized-backprojection unpreconditioned CGLS K13；
- Zero geometry-Jacobi PCGLS K16 作为充分 reference。

主候选的 warm state、预测与 cache 更新只读取当前及历史部署可见观测与已知几何。所有预测先写入封存 barrier，随后才读取 CFD 真值和 K16 reference 评分；真值没有进入 warm state、solver、相机规范化或停止规则。

## 结果

实际 K14 warm 主候选同时达到 **546/546** 个绝对安全单元、**546/546** 个 K16 同精度单元，以及 **13/13** 条绝对安全和同精度完整 rig。

| 指标 | p90-higher | worst | 冻结绝对门 |
| --- | ---: | ---: | ---: |
| Field | 0.278920 | 0.309085 | 0.500000 |
| Full gradient | 0.468104 | 0.593012 | 0.750000 |
| Interior gradient | 0.586947 | 0.731493 | 0.750000 |
| Observation | 0.037880 | 0.054089 | 0.200000 |

四个同价或更便宜 controls 均未通过：

| Control | 绝对安全单元 | K16 同精度单元 | 绝对安全完整 rig | 同精度完整 rig |
| --- | ---: | ---: | ---: | ---: |
| Zero geometry-J PCGLS K14 | 507/546 | 13/546 | 7/13 | 0/13 |
| Zero CGLS K14 | 282/546 | 13/546 | 0/13 | 0/13 |
| BP geometry-J PCGLS K13 | 510/546 | 13/546 | 7/13 | 0/13 |
| BP CGLS K13 | 280/546 | 13/546 | 0/13 | 0/13 |

这说明当前正结果不能由“直接少跑两步”或“便宜 BP 起点”解释；在这组已开封数据上，因果 warm state 对实际未修改 K14 有特异作用。

## 调用账

每条完整 42 帧 rig 的逻辑精确调用账为：

- K14 warm 主候选：`631A+590A^T`；
- K16 reference：`672A+672A^T`；
- 每个同价或更便宜 control：`590A+590A^T`。

主候选相对 K16 的总精确调用名义少 **9.1518%**，而且这次确实满足了同精度门。不过当前尚未做 fresh-process wall time、whole-pipeline RSS 或独立外部工况，因此只能称“已开封 Case 7 上的有效调用账 headroom”，不能称资源加速或部署收益。

## 独立复算

正式有效性门 **24/24** 全真，独立第二实现 **39/39** 项全真。正式与独立的场、残差、指标、汇总、物理 replay 和 cache 最大差全部为 **0**。原始与反转相机顺序在规范化后的场、残差、指标、调用账、replay 和 cache 也全部逐值相同。

K16 锚点和 K1 父证据的最大相对差均为 **8.79e-10**，v241 必要下界的正向违例为 **0**，伴随检查最大相对差为 **1.91e-15**。因此正式判决为 `POST_OPEN_CASE7_CANONICAL_CAMERA_ACTUAL_K14_WARM_SPECIFIC_HEADROOM_V243`。

## 科学边界与下一门

这是对已开封 Case 7 的 post-open 机制正结果，不是学习算法、前瞻外部泛化、资源速度、curved ray 或真实 BOST 证据。v242/v242.1 的 inconclusive 记录继续保留；v243 通过消除相机顺序歧义得到可复算结果，而不是放宽容差。

下一步必须在打开结果前冻结一条此前未开的独立公开反应流序列，原样复用同一个相机规范化适配器、K14 主候选、四个 controls、全部精度门与禁止调参规则。只有该外部门通过，才允许测 fresh wall 和 RSS。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v243: after camera-ID canonicalization, the actual unchanged K14 warm solver passes all opened Case 7 gates

## Why this audit was run

v241 establishes metric-specific necessary capacity in the pre-update FIFO16 cache plus current K1-K14 span, but it permits four truth-aware coefficient vectors and therefore does not test one actual solution. The next gate must directly run the unchanged K14 solver and compare it with equal-or-cheaper classical controls.

Neither the original v242 run nor the v242.1 erratum produces an interpretable scientific verdict. Reversing camera order leaves every discrete pass/fail decision unchanged, but finite summation-order drift exceeds the preregistered numerical gates. v242 has metric drift **1.24e-8**; v242.1 has field and restored-residual drift **1.90e-8 / 1.19e-6**. Both records remain `INCONCLUSIVE`; their tolerances were not relaxed after seeing the result.

## What was executed

v243 resolves only the implementation ambiguity. Cameras are first sorted by stable ASCII camera ID, and their contiguous observation blocks are reindexed in the same order. The adapter performs no arithmetic, changes no dtype, and adds no `A/A^T` call. Original and reversed source orders therefore produce byte-identical canonical camera IDs and observations.

The same thirteen opened Case 7 rigs, forty-two frames per rig, and 546 cells then run:

- the causal pre-update FIFO16 warm state followed by actual unchanged geometry-Jacobi PCGLS K14;
- zero-start geometry-Jacobi PCGLS K14;
- zero-start unpreconditioned CGLS K14;
- normalized-backprojection geometry-Jacobi PCGLS K13;
- normalized-backprojection unpreconditioned CGLS K13; and
- zero-start geometry-Jacobi PCGLS K16 as the adequate reference.

The primary warm state, prediction, and cache update read only current or historical deployment-visible observations and known geometry. Every prediction is sealed behind a barrier before CFD truth and the K16 reference are read for scoring. Truth does not enter the warm state, solver, camera adapter, or stopping rule.

## Result

The actual K14 warm primary reaches **546/546** absolute-safe cells, **546/546** K16-matched cells, and **13/13** complete rigs on both absolute and matched accuracy.

| Metric | p90-higher | worst | Frozen absolute limit |
| --- | ---: | ---: | ---: |
| Field | 0.278920 | 0.309085 | 0.500000 |
| Full gradient | 0.468104 | 0.593012 | 0.750000 |
| Interior gradient | 0.586947 | 0.731493 | 0.750000 |
| Observation | 0.037880 | 0.054089 | 0.200000 |

All four equal-or-cheaper controls fail:

| Control | Absolute-safe cells | K16-matched cells | Absolute-safe rigs | Matched rigs |
| --- | ---: | ---: | ---: | ---: |
| Zero geometry-J PCGLS K14 | 507/546 | 13/546 | 7/13 | 0/13 |
| Zero CGLS K14 | 282/546 | 13/546 | 0/13 | 0/13 |
| BP geometry-J PCGLS K13 | 510/546 | 13/546 | 7/13 | 0/13 |
| BP CGLS K13 | 280/546 | 13/546 | 0/13 | 0/13 |

The result is therefore not explained by simply stopping two steps earlier or by using a cheap backprojection start. On this opened dataset, the causal warm state has a specific effect on the actual unchanged K14 solver.

## Call ledger

The logical exact-call ledger per complete 42-frame rig is:

- K14 warm primary: `631A+590A^T`;
- K16 reference: `672A+672A^T`; and
- each equal-or-cheaper control: `590A+590A^T`.

The primary nominally uses **9.1518%** fewer total exact calls than K16 and now satisfies matched accuracy on this opened condition. No fresh-process wall-time, whole-pipeline RSS, or independent external-condition gate has been run, so this is effective-call headroom on opened Case 7, not a resource-speed or deployment result.

## Independent recomputation

All **24/24** formal validity checks and **39/39** independent checks pass. Formal and independent fields, residuals, metrics, summaries, physical replay, and cache states have maximum difference **0**. Canonicalized original and reversed camera orders also give exact agreement in fields, residuals, metrics, call ledgers, replay, and cache states.

The K16 anchor and K1 parent comparisons have maximum relative difference **8.79e-10**, the v241 necessary lower bound has zero positive violation, and the adjoint check has maximum relative error **1.91e-15**. The exact decision is `POST_OPEN_CASE7_CANONICAL_CAMERA_ACTUAL_K14_WARM_SPECIFIC_HEADROOM_V243`.

## Scientific boundary and next gate

This is a post-open mechanism-positive result on Case 7. It is not a learned algorithm, prospective external generalization, resource speedup, curved-ray validation, or real-BOST evidence. The v242/v242.1 inconclusive records remain preserved; v243 obtains a reproducible result by eliminating camera-order ambiguity, not by relaxing a tolerance.

Before opening results, the next experiment must freeze one previously unopened independent public reacting-flow sequence and reuse the same canonical adapter, K14 primary, four controls, accuracy gates, and no-retuning rule. Fresh wall and RSS measurements are authorized only if that external gate passes.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
