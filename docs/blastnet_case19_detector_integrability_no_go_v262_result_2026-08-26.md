# v262：探测器可积性投影不是当前离散 forward 的精确约束

## 为什么先做零调用审计

v261 已关闭“双分量 + 两个精确方向 + 2×2 Galerkin”机制。v262 没有继续扩大分块，而是先审计一个物理上不同的想法：BOS 的二维双分量位移若来自同一个探测器平面标量势，应落在一个无旋、可积的梯度子空间里。一个自然候选是先把 K14 残差投影到这个子空间，再构造修正方向。

但在线性、无噪声的当前 forward 中存在一个必须先回答的二分问题。记观测为 `y`、K14 预测为 `Ax14`、残差为 `r14 = y - Ax14`，固定线性投影为 `P`：

- 如果 `P` 同时保持 `y` 和 `Ax14`，那么线性关系强制 `P r14 = r14`，投影没有新方向；
- 如果 `P` 明显改变 `y` 或 `Ax14`，它就不是当前离散 forward 的精确物理约束，不能据此删除残差分量。

因此 v262 只做可识别性审计，不生成候选场、不读三维真值或场指标，也不新增 `A/A^T` 调用。

## 冻结实现

每个相机的 `16×16×2` 数据采用固定探测器梯度矩阵 `G`：分量 0 是 `u` 方向导数，分量 1 是 `v` 方向导数；内部用中心差分，两端用固定二阶单边差分。正式实现通过 `G^T G` 的对称特征分解构造 `range(G)` 的正交投影，独立实现从头构造 `G` 并使用完整薄 SVD。固定秩为 `255`，相对谱门为 `1e-12`，forward 不变性门为 `1e-8`。

## 独立结果

独立第二实现通过 `24/24` 项检查。投影数组最大相对差为 `2.70e-10`，缺陷数组最大绝对差为 `7.04e-10`，汇总最大差为 `1.04e-10`；线性闭合最坏为 `3.90e-16`，正交能量闭合最坏为 `9.23e-16`，相机换序差为 `0`。

按每个相机观测范数归一化，投影缺陷的 p50 / p90-higher / worst 为：

- 观测 `y`：`0.10506 / 0.12752 / 0.13863`
- K14 预测 `Ax14`：`0.11460 / 0.13518 / 0.14428`
- K14 残差 `r14`：`0.04484 / 0.04956 / 0.05433`

三类数据各自都只有 `0/117` 个相机块落在 `1e-8` 不变性门内。以残差自身范数归一化时，其缺陷 p90 为 `0.74660`，被投影移除的能量比例 p90 为 `0.55742`。这不是舍弃可忽略的浮点噪声，而是在大幅改变当前离散 forward 实际产生的残差。

## 判决与边界

封存判决为 `FAIL_CASE19_DETECTOR_INTEGRABILITY_PROJECTOR_NOT_FORWARD_INVARIANT_V262`。固定的“探测器标量势梯度子空间”不是当前离散 straight-ray forward 的精确不变量，因此不构造投影残差候选，也不运行完整序列、训练、GPU 或 wall/RSS。

这条负结果只关闭这一套固定梯度、边界和正交投影，不证明所有局部、非线性、噪声感知或真实 BOST 机制都不可能。它也不是重建、matched-accuracy、减调用、速度、外部泛化、曲折光线或真实 BOST 成果。`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v262: detector integrability is not an exact invariant of the current discrete forward

## Why a zero-call audit comes first

v261 closed the two-component, two-direction `2x2` Galerkin mechanism. Rather than enlarge the block, v262 audits a physically different idea. If a two-component BOS displacement is the gradient of one detector-plane scalar potential, it should lie in a curl-free integrable subspace. A natural proposal is to project the K14 residual into that subspace before constructing a correction direction.

The current linear noise-free forward creates a necessary dichotomy. Let `y` be the observation, `Ax14` the K14 prediction, `r14 = y - Ax14` the residual, and `P` a fixed linear projector:

- if `P` preserves both `y` and `Ax14`, linearity forces `P r14 = r14`, so the projector creates no new direction;
- if `P` materially changes either `y` or `Ax14`, it is not an exact invariant of the current discrete forward and cannot justify deleting residual content.

v262 therefore performs only an identifiability audit. It creates no candidate field, reads no 3D truth or field metric, and adds no `A/AT` call.

## Frozen implementation

Each `16x16x2` camera block uses one fixed detector-gradient matrix `G`. Component 0 is the detector-u derivative and component 1 is the detector-v derivative, with centered interior differences and fixed second-order one-sided boundaries. The formal implementation obtains the orthogonal `range(G)` projector through a symmetric eigensolve of `G^T G`; the independent implementation rebuilds `G` and uses a full thin SVD. The frozen rank is `255`, the relative spectral floor is `1e-12`, and the forward-invariance tolerance is `1e-8`.

## Independent result

The second implementation passes all `24/24` checks. Maximum differences are `2.70e-10` relative for projected arrays, `7.04e-10` absolute for defect arrays, and `1.04e-10` for summaries. Worst linearity closure is `3.90e-16`, worst orthogonal-energy closure is `9.23e-16`, and the camera-permutation difference is zero.

Normalized by each camera observation norm, p50 / p90-higher / worst projection defects are:

- observation `y`: `0.10506 / 0.12752 / 0.13863`
- K14 prediction `Ax14`: `0.11460 / 0.13518 / 0.14428`
- K14 residual `r14`: `0.04484 / 0.04956 / 0.05433`

All three sources have `0/117` camera blocks inside the `1e-8` invariance gate. Relative to the residual's own norm, its p90 defect is `0.74660`, and the p90 removed-energy fraction is `0.55742`. The projector is therefore changing substantial forward-produced residual content rather than floating-point noise.

## Verdict and boundary

The sealed decision is `FAIL_CASE19_DETECTOR_INTEGRABILITY_PROJECTOR_NOT_FORWARD_INVARIANT_V262`. The fixed detector scalar-potential gradient subspace is not an exact invariant of the current discrete straight-ray forward. No projected-residual candidate, full sequence, training, GPU, or wall/RSS run follows.

This negative result closes only this fixed gradient, boundary, and orthogonal projector. It does not rule out every local, nonlinear, noise-aware, or real-BOST mechanism. It is not reconstruction, matched accuracy, effective call reduction, speed, external generalization, curved-ray, or real-BOST evidence. `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
