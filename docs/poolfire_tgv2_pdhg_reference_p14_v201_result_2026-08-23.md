# v201：TGV2 把观测拟合得更好，却没有救回一个失败单元

## 结论先说

v201 在历史上已经暴露的 p14 开发轨迹上，结果前固定了一条二阶 TGV2 参考：从 full-DCT K2 场出发，以固定一阶权重 `0.001`、二阶权重 `0.002` 运行 256 步 PDHG，不搜索参数、不早停，也不读取真值选参数。

TGV2 在 **1313/1313** 个单元上都把 observation error 压得比 v200 Huber-TV 更低，但严格安全单元仍是 **1289/1313**，完整标定组仍是 **5/13**。它的 **24 个失败单元与 Huber 完全相同**，一个也没有救回；24 个失败单元全都越过 gradient 门，其中 4 个同时越过 field 门，没有 observation 门失败。

因此正式科学判决是 `FAIL_TGV2_PDHG_REFERENCE_ADEQUACY_V201`。固定 TGV2 路线关闭；五相机 reference 仍不充分，v199 的相对调用数比较继续不可解释。`algorithm_breakthrough=false`。

## 为什么做

v200 已经证明局部保边 Huber-TV 能把 K2 的 1213/1313、0/13 明显提高到 1289/1313、5/13，但剩下的 24 个失败集中在场的过渡帧，而且全部涉及三维梯度。v201 因而不再调 Huber 权重，而是检验一个物理上不同的问题：允许一阶辅助场并惩罚其对称梯度的二阶 TGV2，能否在保留斜坡结构的同时消除这 24 个失败？

## 实际运行

- 起点：封存的 full-DCT K2 场。
- 唯一 primary：二阶 TGV2 PDHG，固定 256 步。
- 固定权重：一阶 `0.001`，二阶 `0.002`。
- 离散：物理轴 forward-Neumann 梯度与精确转置；辅助向量场使用六通道对称梯度。
- 搜索：0 次；没有 clipping、fallback、early stopping 或事后选择。
- 逻辑在线账：K2 起点与 TGV2 迭代合计 `259A+258A^T`。

这是一条昂贵的**参考充分性诊断**，不是部署候选或资源优化。

## 关键数字

| 方法 | 严格安全单元 | 完整标定组 | field / gradient / observation p90 | 逻辑在线账 |
|---|---:|---:|---:|---:|
| TGV2 reference | 1289/1313 | 5/13 | 0.418041 / 0.660039 / 0.014690 | 259A+258A^T |
| v200 Huber-TV parent | 1289/1313 | 5/13 | 0.418272 / 0.660069 / 0.020585 | 131A+130A^T |
| full-DCT K2 parent | 1213/1313 | 0/13 | 0.449851 / 0.737940 / 0.116022 | 3A+2A^T |

TGV2 的 observation p90 从 Huber 的 0.020585 进一步降到 0.014690，而且 1313 个单元逐个都更低；但 field 与 gradient 尾部几乎没有改变，gradient worst 反而从 0.904970 略升到 0.905846。失败集合的交集是 24/24，异或为 0。

这给出了比“仍未过门”更具体的机理结论：继续压低五相机观测残差，并不足以恢复正确的三维梯度。当前剩余瓶颈更接近稀疏视角下的三维不可辨性，而不是优化器没有把观测拟合好。

## 独立复算

独立程序没有导入正式 TGV2 数值核心，而是分别实现 CGLS、有限差分及转置、六通道对称梯度及转置、步长上界、两个对偶投影、PDHG、物理观测、汇总与门。

独立状态为 `PASS_INDEPENDENT_RECOMPUTATION_TGV2_PDHG_REFERENCE_P14_V201`。reference field 相对差约 **2.40e-16**，辅助场哨兵相对差约 **1.40e-16**，指标最大绝对差约 **2.22e-16**；相机换序后的 field 与辅助场相对差分别约 **2.62e-16** 与 **6.21e-17**。离散通过掩码完全一致，一次性 release 已消费，正式树和输入树保持不变。物理 operator builder 与封存输入仍共享，所以 `end_to_end_physics_independence_proven=false`。

## 成功、失败与边界

**成功：** TGV2 在所有 1313 个单元上都进一步降低了 observation error，证明二阶斜坡保持机制确实改变了优化解，而不是没有工作。

**失败：** 它没有救回任何一个 Huber 失败单元，也没有增加一个完整组。按结果前合同，这条固定 TGV2 目标、权重与 256 步求解器关闭，不得看到结果后调整权重、迭代数、步长或边界包装成成功。

**边界：** p14 是历史已暴露开发轨迹。本轮不是 fresh validation、blind test、部署算法、exact-call 减少、wall/RSS、外部泛化、curved ray 或真实 BOST 证据。reference 不充分时，v199 fixed K1 既不能被宣布成功，也不能完成等精度调用数判决。

# v201: TGV2 fits observations better but rescues no failed cell

## Bottom line

On the historically exposed p14 development trajectory, v201 preregisters a second-order TGV2 reference. Starting from full-DCT K2, it runs 256 fixed PDHG iterations with first-order weight `0.001` and second-order weight `0.002`, with no search, early stopping, or truth-dependent parameter choice.

TGV2 lowers observation error relative to v200 Huber-TV in **all 1313 cells**, yet strict-safe cells remain **1289/1313** and complete calibration groups remain **5/13**. Its **24 failed cells are exactly the same as Huber's**, with zero rescues. Every failed cell violates the gradient gate, four also violate field, and none violates observation.

The scientific decision is therefore `FAIL_TGV2_PDHG_REFERENCE_ADEQUACY_V201`. The fixed TGV2 route is closed, five-camera reference adequacy remains unestablished, and v199's relative-call comparison remains uninterpretable. `algorithm_breakthrough=false`.

## Why this was run

v200 showed that local edge-preserving Huber-TV can materially improve K2 from 1213/1313 and 0/13 to 1289/1313 and 5/13. The remaining 24 failures cluster around transition frames and all involve 3D gradients. Rather than retuning Huber, v201 asks a physically different question: can second-order TGV2, which introduces a first-order auxiliary field and penalizes its symmetric gradient, preserve ramps while removing those 24 failures?

## What was run

- Starting point: the sealed full-DCT K2 field.
- Sole primary: second-order TGV2 PDHG for exactly 256 steps.
- Fixed weights: first order `0.001`, second order `0.002`.
- Discretization: forward-Neumann physical gradients and exact transposes; a six-channel symmetric gradient for the auxiliary vector field.
- Search: zero; no clipping, fallback, early stopping, or post-result selection.
- Logical online ledger: `259A+258AT`, including the K2 start and TGV2 iterations.

This is an expensive **reference-adequacy diagnostic**, not a deployment candidate or a resource optimization.

## Key numbers

| Method | Strict-safe cells | Complete groups | Field / gradient / observation p90 | Logical online ledger |
|---|---:|---:|---:|---:|
| TGV2 reference | 1289/1313 | 5/13 | 0.418041 / 0.660039 / 0.014690 | 259A+258AT |
| v200 Huber-TV parent | 1289/1313 | 5/13 | 0.418272 / 0.660069 / 0.020585 | 131A+130AT |
| Full-DCT K2 parent | 1213/1313 | 0/13 | 0.449851 / 0.737940 / 0.116022 | 3A+2AT |

TGV2 lowers observation p90 from Huber's 0.020585 to 0.014690, and every one of 1313 cells improves. Field and gradient tails, however, barely move; gradient worst slightly increases from 0.904970 to 0.905846. The failure-set intersection is 24/24 and the symmetric difference is zero.

This is more informative than merely saying the gate still fails. Further reducing the five-camera observation residual is insufficient to recover the correct 3D gradient. The remaining bottleneck is closer to sparse-view 3D ambiguity than to inadequate observation fitting.

## Independent recomputation

The independent program does not import the formal TGV2 numerical core. It separately implements CGLS, finite differences and transposes, the six-channel symmetric gradient and transpose, the step bound, both dual projections, PDHG, physical observations, summaries, and gates.

Its status is `PASS_INDEPENDENT_RECOMPUTATION_TGV2_PDHG_REFERENCE_P14_V201`. The reference-field relative difference is about **2.40e-16**, the auxiliary sentinel difference is about **1.40e-16**, and the maximum metric difference is about **2.22e-16**. Camera-reordering field and auxiliary differences are about **2.62e-16** and **6.21e-17**, respectively. Discrete pass masks match exactly, the single-use release is consumed, and formal and input trees remain unchanged. The immutable physical operator builder and raw inputs are shared, so `end_to_end_physics_independence_proven=false`.

## What succeeded, what failed, and the boundary

**Succeeded:** TGV2 further lowers observation error in every cell, showing that the second-order ramp-preserving mechanism materially changes the optimized solution.

**Failed:** it rescues none of the Huber failures and adds no complete group. Under the preregistered contract, this exact TGV2 objective, weights, and 256-step solver are closed. Its weights, iterations, steps, and boundaries must not be tuned after seeing the result.

**Boundary:** p14 is a historically exposed development trajectory. This is not fresh validation, a blind test, a deployable algorithm, exact-call reduction, wall/RSS evidence, external generalization, curved-ray validation, or real BOST. Without an adequate reference, v199 fixed K1 can neither be declared successful nor complete an equivalent-accuracy call-count adjudication.
