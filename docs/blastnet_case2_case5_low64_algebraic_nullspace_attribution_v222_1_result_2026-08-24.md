# v222.1：正交去除 null(A) 后 Case 5 保留、Case 2 伤害仍在

## 结论

v221 把 Low-64 起点经过 `A^T A` 谱重加权、单缩放和 PCGLS K10 后，在 Case 5 与 Case 2 都得到 0 个 matched 单元。它证明该构造失败，却没有区分失败来自“去除 `null(A)`”还是来自 `A^T A` 对可观测谱的重加权。

v222 试图直接计算正交行空间投影 `P_row x=A^T(AA^T)^{-1}Ax`，再运行未修改 PCGLS K11。正式运行完成 `1261/1261` 个单元，但冻结的 direct-vs-projected K11 residual-equivalence 最大差为 `1.43918e-9`，高于结果前固定的 `1e-9`，首个独立 validator 也没有通过。因此 v222 必须保持：

`INCONCLUSIVE_INVALID_ORTHOGONAL_ROWSPACE_ATTRIBUTION_V222`

v222.1 没有放宽容差、重跑或把 v222 改写成成功。它只做一个明确标注为 **post-open retrospective** 的代数归因：利用精确算术下 PCGLS 对初始 `null(A)` 分量不更新的恒等式，构造

`x_algebraic = x_direct_final - (x_direct_initializer - P_row x_direct_initializer)`。

正式程序和独立第二实现都支持同一判决：

`POST_OPEN_ROWSPACE_PRESERVES_CASE5_BUT_CASE2_HARM_REMAINS_V222_1`

## 结果

| 工况 | 绝对严格安全单元 | K16 matched 单元 | 完整几何 | 最大 matched ratio |
|---|---:|---:|---:|---:|
| Case 5 | `546/546` | `546/546` | `13/13` | `1.02190` |
| Case 2 | `715/715` | `518/715` | `0/13` | `1.91394` |

这些数字与 direct Low-64 K11 逐项相同。也就是说：

1. 去掉 Low-64 起点的 `null(A)` 分量后，Case 5 的 `546/546、13/13` 完整保留；
2. 同样的去除没有救回 Case 2，它仍是 `518/715、0/13`；
3. 因此 `null(A)` 分量既不是 Case 5 正效应所必需，也不能解释 Case 2 的跨工况伤害；
4. v221 的失败应归因于其 `A^T A` 谱重加权，而不是“观测行空间本身容量不足”。

## 独立复算

独立程序自行重建正交投影、代数场、二维观测、四项指标和完整几何汇总。独立 observation、投影起点、代数场、逐单元指标和汇总最大差分别为 `7.24e-15`、`2.51e-13`、`1.95e-13`、`1.92e-14` 和 `4.35e-14`，`16/16` 项检查全部通过。

共享冻结的底层 physics kernels 仍存在，所以 `end_to_end_physics_independence_proven=false`。这不影响本次代数归因通过，但禁止把它写成完全独立物理实现。

## 证据边界

- v222 本身仍然不可判定，不能被 v222.1 重新包装成结果前通过；
- v222.1 是已开封数据上的事后机制归因，不是部署方法、fresh validation 或算法性能门；
- 没有训练参数，没有 exact-call、wall/RSS、外部工况、曲线光路或真实 BOST 结果；
- 后续若继续，必须直接针对可观测行空间中的谱作用提出结果前冻结的新机制，不能重调 v221/v222 或用大模型、GPU 挽救。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v222.1: Orthogonal null(A) Removal Preserves Case 5 While Case 2 Harm Remains

## Conclusion

v221 applies `A^T A` spectral reweighting, one scale, and PCGLS K10 to the Low-64 start and obtains zero matched cells in both Case 5 and Case 2. It falsifies that construction but does not separate null-space removal from spectral reweighting inside the observable row space.

v222 attempts the exact orthogonal projection `P_row x=A^T(AA^T)^{-1}Ax` followed by unchanged PCGLS K11. All `1261/1261` cells complete, but the frozen direct-versus-projected K11 residual-equivalence difference is `1.43918e-9`, above the preregistered `1e-9` tolerance, and its first independent validator also fails. v222 therefore remains:

`INCONCLUSIVE_INVALID_ORTHOGONAL_ROWSPACE_ATTRIBUTION_V222`

v222.1 neither relaxes that tolerance nor reruns or relabels v222. It performs an explicitly **post-open retrospective** algebraic attribution using the exact-arithmetic PCGLS identity that the initializer component in `null(A)` is not updated:

`x_algebraic = x_direct_final - (x_direct_initializer - P_row x_direct_initializer)`.

The formal and independent implementations support the same decision:

`POST_OPEN_ROWSPACE_PRESERVES_CASE5_BUT_CASE2_HARM_REMAINS_V222_1`

## Results

| Condition | Absolute strict-safe cells | K16-matched cells | Complete rigs | Maximum matched ratio |
|---|---:|---:|---:|---:|
| Case 5 | `546/546` | `546/546` | `13/13` | `1.02190` |
| Case 2 | `715/715` | `518/715` | `0/13` | `1.91394` |

These outcomes are cellwise identical to direct Low-64 K11. Removing the Low-64 component in `null(A)` fully preserves the Case 5 result and does not repair the Case 2 transfer harm. That component is therefore neither required for the Case 5 benefit nor an explanation for the Case 2 failure. The v221 failure points to its `A^T A` spectral reweighting rather than insufficient row-space capacity.

## Independent recomputation

The independent implementation rebuilds the orthogonal projection, algebraic field, 2D observation, all four metrics, and complete-rig summaries. Maximum differences in independent observation, projected initializer, algebraic field, cell metric, and summary are `7.24e-15`, `2.51e-13`, `1.95e-13`, `1.92e-14`, and `4.35e-14`. All `16/16` checks pass.

Frozen low-level physics kernels remain shared, so `end_to_end_physics_independence_proven=false`. This does not invalidate the algebraic attribution, but it prevents a claim of fully independent end-to-end physics.

## Evidence boundary

- v222 remains inconclusive and is not relabelled as a preregistered pass;
- v222.1 is a post-open mechanism attribution, not a deployment method, fresh validation, or algorithm-performance gate;
- there are no trained parameters and no exact-call, wall/RSS, external-condition, curved-ray, or real-BOST result;
- any next mechanism must directly address spectral action within the observable row space and be frozen before results, rather than retuning v221/v222 or using a larger model or GPU.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
