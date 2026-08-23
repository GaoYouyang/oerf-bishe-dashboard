# v205：势函数正规方程缓存移除了在线稠密响应矩阵

## 这次真正回答了什么

v204 已证明：在历史上已经暴露的 p14 开发轨迹上，九相机 full-DCT K1 可以通过全部 `1313/1313` 个单元和 `13/13` 个完整组，但它需要保留并扫描稠密几何响应。v205 只检验一个更窄的问题：**能否在不改变物理结果的前提下，换成紧凑、只读部署可见观测与报告几何的缓存路径？**

答案是肯定的。正式程序与完全独立的第二实现都得到判决：

`PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205`

`PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_COMPACT_CACHE_V205`

## 做了什么，以及为什么这样做

零均值双分量探测器位移可以积分成势函数。v205 先在观测平面完成这一步，再用一次精确 `A^T` 把势函数提升回三维场；随后把结果投影到固定的 1009 维非直流 DCT 基，并用几何预计算得到的正规方程 Cholesky 因子求解。这样在线阶段不再保留或扫描稠密响应矩阵。

这不是低秩近似、随机 sketch 或重新训练的预测器；它利用的是势函数表示与 full-DCT 正规方程之间的精确线性关系。保留状态从五相机的 `2,900,875`、九相机的 `5,221,575` 个标量，降到固定的 `509,545` 个标量，分别缩小约 `5.69x` 与 `10.25x`。

## 独立复算结果

- 五/九相机共 `2626` 个单元，覆盖 `13` 套标定和每套 `101` 帧。
- 相对正式实现的最大坐标差为 `1.43e-12`；相对封存父结果的最大坐标差为 `9.84e-13`。
- 相对封存父结果的最大三维场差为 `9.84e-13`。
- 直接右端项重建最大差为 `2.04e-14`，正规方程因子重建最大差为 `2.04e-13`。
- 相机乱序后的最大坐标差为 `4.24e-13`，Gram 矩阵差为 `4.28e-16`。

这些数值说明紧凑路径在当前合同下复现了稠密路径，而不是仅仅得到相似的通过/失败标签。

## 准确率继承了什么

v205 不改变 v204 的准确率判决：九相机仍通过 `1313/1313` 个单元和 `13/13` 个完整组。五相机仍只有 `1268/1313` 个单元和 `3/13` 个完整组通过。因此，本轮没有证明五/九相机都稳定成功，也没有建立更广的可变相机数泛化。

## 为什么现在还不能说更快

紧凑 initializer 本身需要 `0A+1A^T`；接上未修改 K1 后，总逻辑账为 `2A+2A^T`。稠密 full-DCT K1 的账是 `2A+1A^T`，所以紧凑路径反而多一次精确伴随；它只比 K2 的 `3A+2A^T` 少一次 `A`。

此外，正式 setup 仍会瞬时构造响应矩阵。本轮没有 fresh-process wall、worker/process-tree RSS 或整流水线峰值内存实验。**缓存标量减少不等于已证明速度或峰值内存下降。**

## 结论与下一门

成功的是：v204 的在线稠密响应缓存可以被一个数值等价、相机乱序稳定的紧凑缓存替代。

没有成功的是：五相机准确率没有被救回，精确调用没有优于稠密 K1，真实 wall/RSS、外部工况、曲线光路和真实 BOST 都没有通过。

下一门只做一次结果前冻结的 fresh-process 资源审计：先移除 setup 的瞬时稠密响应，保留相机换序等变、原生支持可变相机数的接口但不声称所有基数已过门；把 setup 放入 worker，保留便宜 CPU 对照，并同时比较紧凑 K1、稠密 K1 与 K2 的端到端 wall、worker/process-tree RSS 和整流水线峰值。若资源门失败，就不能只凭缓存计数宣称加速。

`algorithm_breakthrough=false`

---

# v205: a potential-normal cache removes the online dense response matrix

## What this run actually answers

v204 established that all-nine full-DCT K1 clears `1,313/1,313` cells and `13/13` complete groups on the historically exposed p14 development trajectory, but it retains and scans a dense geometry response. v205 asks one narrower question: **can that cache path be made compact without changing the physical result, while reading deployment-visible observations and reported geometry only?**

Yes. The formal program and a fully independent second implementation both seal:

`PASS_POTENTIAL_NORMAL_COMPACT_CACHE_V205`

`PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_COMPACT_CACHE_V205`

## What was done and why

Zero-mean two-component detector displacement can be integrated into a potential. v205 performs this operation in the observation plane, lifts the potential into the 3D field with one exact `A^T`, projects it into the fixed 1,009-dimensional non-DC DCT basis, and solves with a geometry-precomputed normal-equation Cholesky factor. The online path therefore no longer retains or scans the dense response matrix.

This is neither a low-rank approximation, a random sketch, nor a newly trained predictor. It uses the exact linear relation between the potential representation and the full-DCT normal equations. The retained state falls from `2,900,875` scalars for five cameras and `5,221,575` for nine cameras to a fixed `509,545`, reductions of about `5.69x` and `10.25x`.

## Independent recomputation

- `2,626` five/all-nine-camera cells across `13` calibrations and `101` frames each.
- Maximum coordinate difference to the formal implementation: `1.43e-12`; to the sealed parent: `9.84e-13`.
- Maximum 3D-field difference to the sealed parent: `9.84e-13`.
- Maximum direct right-hand-side reconstruction difference: `2.04e-14`; normal-factor reconstruction difference: `2.04e-13`.
- Maximum coordinate difference under camera permutation: `4.24e-13`; Gram-matrix difference: `4.28e-16`.

These values show that the compact path reproduces the dense path under the frozen contract, rather than merely matching its pass/fail labels.

## What accuracy is inherited

v205 does not change the v204 accuracy verdict. All-nine still passes `1,313/1,313` cells and `13/13` complete groups. Five-camera reconstruction still passes only `1,268/1,313` cells and `3/13` complete groups. This run therefore does not establish stable success for both camera counts or broader variable-cardinality generalization.

## Why this is not yet a speed result

The compact initializer itself uses `0A+1A^T`; with unchanged K1, the total logical ledger is `2A+2A^T`. Dense full-DCT K1 uses `2A+1A^T`, so the compact path adds one exact adjoint. It is only one `A` below the K2 ledger of `3A+2A^T`.

Formal setup also constructs the response matrix transiently. No fresh-process wall, worker/process-tree RSS, or whole-pipeline peak-memory gate was run. **Fewer cached scalars do not by themselves prove lower wall time or peak RSS.**

## Conclusion and next gate

What succeeded: the online dense response cache from v204 can be replaced by a numerically equivalent, camera-permutation-stable compact cache.

What did not: five-camera accuracy was not rescued, exact calls did not beat dense K1, and real wall/RSS, external conditions, curved rays, and real BOST remain untested.

The next gate is one preregistered fresh-process resource audit: first remove the transient dense setup response and preserve the camera-permutation-equivariant, variable-cardinality interface without claiming that every cardinality passes; place setup inside each worker, retain cheap CPU controls, and compare compact K1, dense K1, and K2 on end-to-end wall time, worker/process-tree RSS, and whole-pipeline peak memory. If the resource gate fails, cache counts alone cannot support a speedup claim.

`algorithm_breakthrough=false`
