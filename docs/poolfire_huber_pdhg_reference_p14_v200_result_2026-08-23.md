# v200：Huber-TV 改善了五相机参考，但仍未达到完整充分性门

## 结论先说

v200 在历史上已经暴露的 p14 开发轨迹上，结果前固定了一条物理上不同于旧 identity-prior 与全局二次谱正则的参考机制：从 full-DCT K2 场出发，用尺度等变、边缘保持的 Huber-TV primal-dual refinement 做固定 128 步计算，不调权重、不调 Huber 阈值、不早停，也不读取真值选参数。

它把五相机严格安全单元从 K2 的 **1213/1313** 提高到 **1289/1313**，完整标定组从 **0/13** 提高到 **5/13**。但是仍有 **24 个单元**和 **8 个完整组**没有通过冻结门，因此正式科学判决是 `FAIL_HUBER_PDHG_REFERENCE_ADEQUACY_V200`。这条固定 Huber 参考按协议关闭；五相机 reference 仍不充分，v199 的相对调用数比较继续不可解释。`algorithm_breakthrough=false`。

## 为什么做

v199 的固定 identity-prior K1 在五相机下达到 1268/1313 和 3/13，明显优于未正则 K1；但用于比较的 full-DCT K2 reference 自身只有 1213/1313 和 0/13。参考尺子没有过门时，不能因为候选账面上少算了 `A/A^T` 就声称等精度加速。

v200 因此不再调 `tau` 或 Krylov 深度，而是先问一个更基础的问题：一个结果前固定、局部边缘保持的经典变分参考，能否让五相机参考本身在所有 1313 个单元和 13 组上都合格？

## 实际运行

- 起点：封存的 full-DCT K2 场。
- 唯一 primary：尺度等变 Huber-TV PDHG，固定 128 步。
- 固定参数：regularization weight `0.001`，Huber delta `0.08`。
- 边界与离散：物理轴上的 forward-Neumann 梯度及其精确转置。
- 搜索：0 次；没有 clipping、fallback 或 early stopping。
- 逻辑在线账：K2 起点 `3A+2A^T`，PDHG 再用 `128A+128A^T`，合计 `131A+130A^T`。

这是一条昂贵的**参考充分性诊断**，不是部署候选，也不是资源优化。

## 关键数字

| 方法 | 五相机严格安全单元 | 完整标定组 | field / gradient / observation p90 | 逻辑在线账 |
|---|---:|---:|---:|---:|
| Huber-TV reference | 1289/1313 | 5/13 | 0.418272 / 0.660069 / 0.020585 | 131A+130A^T |
| full-DCT K2 parent | 1213/1313 | 0/13 | 0.449851 / 0.737940 / 0.116022 | 3A+2A^T |
| v199 fixed identity K1 | 1268/1313 | 3/13 | 0.421892 / 0.697287 / 0.148866 | 2A+1A^T |

Huber-TV 对场、梯度和观测都带来明显改善，特别是 observation p90 从约 0.1160 降到 0.0206。但完整门要求每一个单元都安全、每个标定组的 p90 与 worst 都合格；1289/1313 和 5/13 仍不能改写成“差一点就成功”。

## 独立复算

独立程序没有导入正式 Huber-PDHG 数值核心，而是分别实现 CGLS、有限差分及转置、尺度归一化、步长构造、Huber 对偶近端、汇总、门和判决。它从封存输入重建 13 套标定下的全部场和指标，再与正式输出事后比较。

独立状态为 `PASS_INDEPENDENT_RECOMPUTATION_HUBER_PDHG_REFERENCE_P14_V200`。reference field block 相对差约 **1.78e-16**，指标最大绝对差约 **2.22e-16**，离散通过掩码完全一致；一次性 release 已消费，正式树和输入树保持不变。物理 operator builder 与封存输入仍共享，所以 `end_to_end_physics_independence_proven=false`。

## 成功、失败与边界

**成功：** Huber-TV 把参考从 1213 提高到 1289 个安全单元，并首次让 5/13 个完整组通过，说明局部边缘保持先验确实比继续增加低深度 CGLS 更接近五相机所需的参考结构。

**失败：** 它没有达到 1313/1313 和 13/13。按结果前合同，这条固定 Huber 目标、参数和 128 步求解器关闭，不得看到结果后微调权重、阈值、迭代数、归一化或起点包装成成功。

**边界：** p14 是历史已暴露开发轨迹；本轮不是 fresh validation、blind test、部署算法、exact-call 减少、wall/RSS、外部泛化、curved ray 或真实 BOST 证据。没有充分 reference 时，v199 fixed K1 既不能被宣布成功，也不能完成等精度调用数判决。

# v200: Huber-TV improves the five-camera reference but still misses the complete adequacy gate

## Bottom line

On the historically exposed p14 development trajectory, v200 preregisters a reference mechanism physically distinct from the old identity prior and global quadratic spectral filters. Starting from the full-DCT K2 field, it runs a fixed 128-step, scale-equivariant, edge-preserving Huber-TV primal-dual refinement without parameter search, early stopping, or truth-dependent selection.

It raises five-camera strict-safe cells from **1213/1313** for K2 to **1289/1313**, and complete calibration groups from **0/13** to **5/13**. Yet **24 cells** and **eight complete groups** still miss the frozen gate. The scientific decision is therefore `FAIL_HUBER_PDHG_REFERENCE_ADEQUACY_V200`. This fixed Huber reference is closed by protocol, five-camera reference adequacy remains unestablished, and v199's relative-call comparison remains uninterpretable. `algorithm_breakthrough=false`.

## Why this was run

v199 fixed identity-prior K1 reaches 1268/1313 and 3/13 under five cameras, clearly improving on unregularized K1. Its full-DCT K2 comparison reference, however, reaches only 1213/1313 and 0/13. An inadequate reference cannot support an equivalent-accuracy call-reduction claim.

v200 therefore does not tune `tau` or Krylov depth. It asks whether one fixed local edge-preserving classical variational reference can first make the five-camera ruler adequate in every one of 1313 cells and all 13 calibration groups.

## What was run

- Starting point: the sealed full-DCT K2 field.
- Sole primary: scale-equivariant Huber-TV PDHG for exactly 128 steps.
- Fixed constants: regularization weight `0.001`, Huber delta `0.08`.
- Discretization: forward-Neumann differences on physical axes and their exact transpose.
- Search: zero; no clipping, fallback, or early stopping.
- Logical online ledger: `3A+2AT` for the K2 start plus `128A+128AT` for PDHG, totaling `131A+130AT`.

This is an expensive **reference-adequacy diagnostic**, not a deployable candidate or a resource optimization.

## Key numbers

| Method | Five-camera strict-safe cells | Complete groups | Field / gradient / observation p90 | Logical online ledger |
|---|---:|---:|---:|---:|
| Huber-TV reference | 1289/1313 | 5/13 | 0.418272 / 0.660069 / 0.020585 | 131A+130AT |
| Full-DCT K2 parent | 1213/1313 | 0/13 | 0.449851 / 0.737940 / 0.116022 | 3A+2AT |
| v199 fixed identity K1 | 1268/1313 | 3/13 | 0.421892 / 0.697287 / 0.148866 | 2A+1AT |

Huber-TV materially improves field, gradient, and observation errors, especially lowering observation p90 from about 0.1160 to 0.0206. The complete gate still requires every cell to be safe and every calibration group to satisfy its p90 and worst-case limits. The observed 1289/1313 and 5/13 cannot be relabeled as success.

## Independent recomputation

The independent program does not import the formal Huber-PDHG numerical core. It separately implements CGLS, finite differences and their transpose, scale normalization, step construction, the Huber dual proximal map, summaries, gates, and decisions. It rebuilds every field and metric for all 13 calibrations before comparing with formal outputs post hoc.

Its status is `PASS_INDEPENDENT_RECOMPUTATION_HUBER_PDHG_REFERENCE_P14_V200`. The reference-field block relative difference is about **1.78e-16**, the maximum metric difference is about **2.22e-16**, and discrete pass masks match exactly. The single-use release is consumed, and formal and input trees remain unchanged. The immutable physical operator builder and raw inputs are shared, so `end_to_end_physics_independence_proven=false`.

## What succeeded, what failed, and the boundary

**Succeeded:** Huber-TV raises the reference from 1213 to 1289 strict-safe cells and produces the first 5/13 complete groups. Local edge preservation is therefore materially closer to the five-camera reference requirement than simply extending low-depth CGLS.

**Failed:** it does not reach 1313/1313 and 13/13. Under the preregistered contract, this exact Huber objective, constants, and 128-step solver are closed. Its weight, threshold, iteration count, normalization, and starting point must not be tuned after seeing the result.

**Boundary:** p14 is a historically exposed development trajectory. This is not fresh validation, a blind test, a deployable algorithm, exact-call reduction, wall/RSS evidence, external generalization, curved-ray validation, or real BOST. Without an adequate reference, v199 fixed K1 can neither be declared successful nor complete an equivalent-accuracy call-count adjudication.
