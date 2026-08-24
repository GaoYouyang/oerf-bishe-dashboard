# v228：两种 PRESS 判据互补，但只构成事后机制线索

## 结论

v226 原始 block-PRESS 与 v227 几何白化 block-PRESS 各自在 Case 5 的一个留一 rig 上只接受 `4/42`，低于冻结门要求的 `5/42`。失败位置不同：原始分数卡在 rig 11，白化分数卡在 rig 4。v228 不重新拟合分数或阈值，只回顾性检验一个固定逻辑：

`accept_union = accept_raw OR accept_studentized`

正式实现和完全独立的第二实现得到一致判决：

`POST_OPEN_COMPLEMENTARY_DUAL_PRESS_SIGNAL_V228`

固定 OR 在 Case 5 接受 `140/546` 个安全单元，危险误接为 `0`，最差 rig 达到 `5/42=11.90%`；在 Case 2 接受 `324/715` 个安全单元，危险误接仍为 `0`。两套工况都保持 `13/13` 完整 rig 精度通过，逻辑平均调用账也都低于 Zero-PCGLS K16。

这是真实的机制增量：原始和白化分数携带互补信息，单分数的标量化或跨 rig 校准是当前瓶颈的一部分。但 v226/v227 的失败 rig 在 v228 前已经可见，所以这个 OR 结果是 **post-open retrospective diagnostic**，不是结果前注册的算法成功。

## 做了什么

v228 直接读取两套已经独立封存的离散决策：

- v226 原始相机分块 PRESS；
- v227 reported-geometry studentized block-PRESS。

它只形成四个固定集合：两者都接受、仅原始接受、仅白化接受、两者都拒绝。OR 决策在读取真值指标前封存；随后才用冻结的 Case 2/5 指标重放安全门、完整 rig 物理精度门和逻辑调用账。没有新特征、分数、协方差、阈值、拟合、异常点删除、模型参数或未开封工况。

## 结果

| 条件与判据 | 接受单元 | 最低 rig 接受率 | 接受的不安全单元 | 完整 rig 精度 | 判决 |
|---|---:|---:|---:|---:|---|
| Case 5，原始 v226 | `126/546` | `4/42=9.52%` | `0` | `13/13` | 效用失败 |
| Case 5，白化 v227 | `123/546` | `4/42=9.52%` | `0` | `13/13` | 效用失败 |
| Case 5，固定 OR | `140/546` | `5/42=11.90%` | `0` | `13/13` | 回顾门通过 |
| Case 2，固定 OR | `324/715` | `19/55=34.55%` | `0` | `13/13` | 回顾门通过 |

互补性不是由一个边缘浮点值造成的。在 Case 5，`109` 个单元被两者共同接受，`17` 个仅被原始分数接受，`14` 个仅被白化分数接受；rig 4 的接受数为 `5 / 4 / 5`，rig 11 为 `4 / 6 / 6`，顺序均为原始 / 白化 / OR。Case 2 中还有 `1` 个仅原始接受和 `27` 个仅白化接受，全部为安全单元。

OR 策略的 Case 2/5 最大 matched ratio 分别为 `1.027761` 和 `1.007896`。Direct K11 路径逻辑账为 `12A+11A^T`，回退 Zero-PCGLS K16 为 `16A+16A^T`；所有 rig 的平均逻辑 `A` 与 `A^T` 都低于 `16`。但这仍不是一个已冻结的部署算法，因此不能把该账写成已经建立的 exact-call、wall 或 RSS 收益。

## 独立复算

正式实现从封存的正式父决策形成 OR；独立实现只读取两套分别封存的独立父决策，用单独的 NumPy 逻辑、逐条件逐 rig 循环、物理策略重放和闭式成本账重建全部结果。两边都覆盖 `1261` 个单元，并在读取真值指标前封存互斥且完备的决策分区。

全部 `17/17` 项必需检查通过，离散决策完全一致，正式汇总最大绝对差为 `2.57e-11`。正式树、两套父证据树和源码闭包在验证前后均未变化。独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_RETROSPECTIVE_DUAL_PRESS_UNION_V228`

底层冻结 physics kernels 仍由父证据共享，因此 `end_to_end_physics_independence_proven=false`。

## 证据边界

- v228 只是在已开封 Case 2/5 上进行的事后机制诊断，不是 fresh 或 external gate；
- 它支持“原始与几何白化 PRESS 含互补可观测信号”，不证明固定 OR 能在未开封工况泛化；
- 不允许据此继续搜索 AND、加权组合、其他布尔公式、阈值或协方差参数；
- 不打开 Case 4/6，不授权训练、GPU、fresh wall/RSS 或论文成功主张；
- 没有部署算法、稳定 exact-call 收益、外部泛化、曲线光路或真实 BOST 结果。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v228: Two PRESS Criteria Are Complementary, but Only as a Retrospective Mechanism Lead

## Conclusion

The raw block-PRESS score in v226 and the geometry-whitened block-PRESS score in v227 each accept only `4/42` cells in one held-out Case 5 rig, below the frozen `5/42` minimum. They fail in different places: raw fails rig 11 and studentized fails rig 4. v228 fits no score or threshold. It retrospectively tests one fixed rule:

`accept_union = accept_raw OR accept_studentized`

The formal implementation and a fully separate second implementation agree on:

`POST_OPEN_COMPLEMENTARY_DUAL_PRESS_SIGNAL_V228`

The fixed OR accepts `140/546` safe Case 5 cells with `0` unsafe accepts, and the worst rig reaches `5/42=11.90%`. It accepts `324/715` safe Case 2 cells, again with `0` unsafe accepts. Both conditions retain `13/13` complete-rig accuracy, and every rig's logical mean call ledger remains below Zero-PCGLS K16.

This is a substantive mechanism increment: raw and whitened scores carry complementary information, so single-score scalarization or cross-rig calibration is part of the current bottleneck. The failing v226/v227 rigs were already visible before v228, however, so the OR result is a **post-open retrospective diagnostic**, not a preregistered algorithmic success.

## What was done

v228 directly reads two independently sealed decision arrays: raw camera-block PRESS from v226 and reported-geometry studentized block-PRESS from v227. It forms exactly four fixed partitions: both accept, raw only, studentized only, and neither. The OR decisions are sealed before truth metrics are loaded; frozen Case 2/5 metrics are read only afterward to replay safety, complete-rig physical accuracy, and the logical call ledger. No feature, score, covariance, threshold, fit, outlier deletion, model parameter, or unopened condition is added.

## Results

| Condition and rule | Accepted cells | Minimum rig acceptance | Unsafe accepts | Complete-rig accuracy | Decision |
|---|---:|---:|---:|---:|---|
| Case 5, raw v226 | `126/546` | `4/42=9.52%` | `0` | `13/13` | utility fails |
| Case 5, studentized v227 | `123/546` | `4/42=9.52%` | `0` | `13/13` | utility fails |
| Case 5, fixed OR | `140/546` | `5/42=11.90%` | `0` | `13/13` | retrospective gate passes |
| Case 2, fixed OR | `324/715` | `19/55=34.55%` | `0` | `13/13` | retrospective gate passes |

Complementarity is not a single floating-point edge case. In Case 5, `109` cells are accepted by both scores, `17` by raw only, and `14` by studentized only. Rig 4 accepts `5 / 4 / 5` cells and rig 11 accepts `4 / 6 / 6`, ordered as raw / studentized / OR. Case 2 contributes another `1` raw-only and `27` studentized-only accepts, all safe.

Maximum matched ratios for the OR policy are `1.027761` in Case 2 and `1.007896` in Case 5. Direct K11 has a logical ledger of `12A+11A^T`, while fallback Zero-PCGLS K16 uses `16A+16A^T`; every rig has mean logical `A` and `A^T` below `16`. The rule is not a frozen deployment algorithm, so these ledgers do not establish exact-call, wall-time, or RSS benefit.

## Independent recomputation

The formal implementation forms the OR from sealed formal parent decisions. The independent implementation reads only the separately sealed independent v226/v227 decisions and rebuilds all outputs with separate NumPy logic, explicit condition-and-rig loops, physical-policy replay, and a closed-form cost ledger. Both cover all `1261` cells and seal disjoint, exhaustive decision partitions before loading truth metrics.

All `17/17` required checks pass, discrete decisions match exactly, and the maximum absolute formal-summary difference is `2.57e-11`. The formal tree, both parent evidence trees, and source closure remain unchanged across validation. The independent status is:

`PASS_INDEPENDENT_RECOMPUTATION_RETROSPECTIVE_DUAL_PRESS_UNION_V228`

Frozen low-level physics kernels remain shared through the parent evidence, so `end_to_end_physics_independence_proven=false`.

## Evidence boundary

- v228 is a retrospective mechanism diagnostic on opened Cases 2 and 5, not a fresh or external gate;
- it supports complementary observable information in raw and geometry-whitened PRESS, but does not prove that fixed OR generalizes to unopened conditions;
- AND, weighted combinations, other Boolean formulas, thresholds, and covariance parameters may not be searched after this result;
- Cases 4/6 remain unopened, and no training, GPU rental, fresh wall/RSS, or paper-success claim is authorized;
- no deployment algorithm, stable exact-call gain, external generalization, curved-ray validation, or real-BOST result is established.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
