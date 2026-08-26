# v248：固定三维体素块预条件器未通过首帧门，独立判决保持不确定

## 先说结论

v248 在已经开封的 Case 19 上检验了一个与坐标线不同的 solver-native 机制：把 `32×16×16` 三维场按固定 `4×2×2` 体素块划分，对每块构造报告几何下的精确正规算子子块，再用于 Zero-PCGLS K16。formal 完成 13 套 rig 的共同首帧并通过 **21/21** 项有效性检查；完全独立的第二实现完成复算，但只通过 **26/28** 项。权威科学判决是：

`INCONCLUSIVE_INVALID_CASE19_GEOMETRY_VOXEL_BLOCK_JACOBI_FRAME_ZERO_V248`

独立侧与 formal 的观测数组最大差为 **1.33e-15**。虽然它极小，但结果前合同要求这一项逐位完全相同，因此不能把 formal 或独立侧的通过数包装成有效性能结果。

## 检验了什么

固定块形状按 `z/y/x = 4/2/2` 排列，在三个归一化世界轴上具有相同边长。512 个不重叠块覆盖全部 5880 个内部单元；边界保持固定。每个精确局部块只增加 `1e-6 × diag(diag(B))` 的结果前固定稳定项，不搜索分块、平移、重叠、loading、深度或阈值。

三个冻结 arm 都使用 **16A+16A^T**：

- 主候选：Zero + 固定体素块 Jacobi PCGLS K16；
- 便宜对照：Zero + 几何对角 Jacobi PCGLS K16；
- 基础对照：Zero + 未预条件 CGLS K16。

本门只看 13 套 rig 的共同首帧。只有主候选 13/13 通过且两个对照未完整通过，才会授权另行冻结完整 429 单元序列；v248 没有达到这个条件。

## 独立闭环为什么不成立

两套实现分别采用批量块装配与显式逐块矩阵乘法/线性求解。大多数比较均在冻结界内：

| 比较项 | 最大差 | 冻结界 | 判定 |
| --- | ---: | ---: | --- |
| 精确体素块，相对差 | 2.95e-16 | 1e-11 | 通过 |
| 体素块逆，相对差 | 1.74e-15 | 1e-8 | 通过 |
| 候选场，相对差 | 9.20e-10 | 1e-8 | 通过 |
| 残差，按观测范数归一 | 5.60e-10 | 1e-8 | 通过 |
| 逐单元指标，绝对差 | 1.10e-10 | 1e-9 | 通过 |
| 汇总，绝对差 | 4.62e-11 | 1e-9 | 通过 |
| 观测数组，绝对差 | **1.33e-15** | **必须为 0** | **失败** |

第二个失败检查 `formal_independent_decision_exact` 是这个缺口的派生结果：formal 的 pending 判决不能覆盖独立侧的 fail-closed 状态。

## 为什么不放宽规则重跑

观测逐位相同的规则在结果前已经冻结。看到 `1.33e-15` 后改成容差比较，会让同一批数据同时参与定规则和证明规则。三次前置执行问题都在任何 arm 评分前停止并原样保留；最终评分链没有利用这些失败选择候选或阈值。

更关键的是，两套实现的诊断结论一致：主候选只有 **0/13** 个首帧单元通过，未预条件对照也是 **0/13**，而更便宜的对角 Jacobi 对照是 **12/13**。主候选的 field / gradient / interior-gradient / observation p90 为 **0.57068 / 1.00702 / 1.57716 / 0.04105**；它不是一个接近通过、值得为数值容差重跑的机制。

这些通过数只用于停止投入，不能作为独立验证通过后的性能证据。

## 路线动作与证据边界

固定 `4×2×2` 三维体素块 Jacobi 机制退出继续投入，完整 429 单元序列不运行，也不增加块大小、重叠、shift、loading 或深度来补救。关闭的是这个固定实现，不是整个 C 路线，也不是数学不可能证明。

Case 19 已经开封，所以 v248 只是 post-open 机制诊断。三个 arm 的成本均为 16A+16A^T，没有减少精确调用；没有 fresh wall/RSS、外部泛化、真实 BOST、curved-ray、预测器训练或 GPU 结论。

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`

---

# v248: fixed 3D voxel blocks fail the frame-zero gate and remain inconclusive

## Bottom line

v248 tests a solver-native mechanism that is physically distinct from coordinate lines on already-opened Case 19. The `32×16×16` field is partitioned into fixed `4×2×2` voxel blocks; each block receives an exact reported-geometry normal-operator submatrix for Zero-PCGLS K16. Formal completes the common first frame of all 13 rigs and passes **21/21** validity checks. A fully independent second implementation completes the recomputation but passes only **26/28** checks. The authoritative decision is:

`INCONCLUSIVE_INVALID_CASE19_GEOMETRY_VOXEL_BLOCK_JACOBI_FRAME_ZERO_V248`

The independent and formal observation arrays differ by at most **1.33e-15**. This is tiny, but the preregistered contract requires bitwise equality for this item, so neither set of pass counts is admissible as a validated performance result.

## What was tested

The fixed block shape is `z/y/x = 4/2/2`, giving equal side lengths along the three normalized world axes. There are 512 non-overlapping, origin-fixed blocks covering all 5,880 interior cells while the boundary remains fixed. Each exact local block receives only the preregistered stabilization `1e-6 × diag(diag(B))`; there is no search over partition, shift, overlap, loading, depth, or thresholds.

All three frozen arms use **16A+16A^T**:

- primary: Zero + fixed voxel-block Jacobi PCGLS K16;
- cheap control: Zero + geometry-diagonal Jacobi PCGLS K16;
- base control: Zero + unpreconditioned CGLS K16.

This gate scores only the common first frame of the 13 rigs. A full 429-cell sequence would be separately frozen only if the primary passed 13/13 and neither control passed completely. v248 does not meet that condition.

## Why independent closure fails

Formal uses batched block assembly, while the independent implementation uses explicit per-block matrix multiplication and linear solves. Most comparisons stay inside their frozen limits:

| Comparison | Maximum difference | Frozen limit | Result |
| --- | ---: | ---: | --- |
| Exact voxel blocks, relative | 2.95e-16 | 1e-11 | Pass |
| Block inverses, relative | 1.74e-15 | 1e-8 | Pass |
| Candidate fields, relative | 9.20e-10 | 1e-8 | Pass |
| Residuals, observation-normalized | 5.60e-10 | 1e-8 | Pass |
| Cell metrics, absolute | 1.10e-10 | 1e-9 | Pass |
| Summaries, absolute | 4.62e-11 | 1e-9 | Pass |
| Observation arrays, absolute | **1.33e-15** | **must equal 0** | **Fail** |

The second failed check, `formal_independent_decision_exact`, is derivative: a formal pending decision cannot override the independent fail-closed state.

## Why there is no tolerance relaxation or rerun

Bitwise observation equality was frozen before the result was seen. Replacing it with a tolerance after observing `1.33e-15` would use the same data to define and prove the rule. Three preceding execution issues stopped before any arm was scored and remain preserved; none was used to select a candidate or threshold.

More importantly, both implementations agree diagnostically that the primary passes **0/13** frame-zero cells, the unpreconditioned control also passes **0/13**, and the cheaper diagonal-Jacobi control passes **12/13**. Primary p90 field / gradient / interior-gradient / observation errors are **0.57068 / 1.00702 / 1.57716 / 0.04105**. This is not a near-pass worth rerunning for a numerical tolerance.

These counts guide the stop decision only; they are not independently validated performance evidence.

## Route action and evidence boundary

The fixed `4×2×2` 3D voxel-block Jacobi mechanism is retired. The full 429-cell sequence does not run, and block size, overlap, shift, loading, or depth are not expanded to rescue it. This closes the fixed implementation, not the C route, and it is not a mathematical-impossibility result.

Case 19 is already open, so v248 is a post-open mechanism diagnostic. All arms cost 16A+16A^T, with no exact-call reduction. There is no fresh wall/RSS, external-generalization, real-BOST, curved-ray, predictor-training, or GPU result.

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`
