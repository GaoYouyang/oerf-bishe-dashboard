# v212：固定有符号射线相消标量不能解释 Case 5 reference 差异

## 讲人话结论

v210 发现实际低模 forward 的全局谱下限具有很强方向性，v211 又排除了固定的局部无符号覆盖下尾。v212 因此检验两者之间一个物理上不同的桥梁：局部覆盖即使不弱，低频场沿射线正负交替时，线积分是否会互相抵消，从而让五相机 reference 变差？

结果仍是否定的。结果前固定的有符号线积分相干比，在 `169` 个跨族比较中，虚拟九相机只严格高于师兄标定族 `7/169` 次；严格成功门要求 `169/169`。师兄标定族与虚拟九相机的主指标中位数分别为 `0.64597` 和 `0.62922`。正式科学判决是 `FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212`。

这个结论只关闭当前固定标量：64 个低频正弦模态、64 点射线中点积分、u/v 双分量和相机等权聚合。它不等于“所有有符号相位结构都无关”，也不否定 v210 实际 forward Gram 的方向性；它说明不能把 v210 的谱差异简单归结为这一种逐射线正负相消比。

## 做了什么，为什么这样做

本轮只读取 reported geometry，不读取密度场、二维观测、重建、残差或旧科学指标数组。对每一套几何：

1. 把每条 detector ray 裁剪到固定三维重建盒；
2. 在有效线段内取 `64` 个固定开区间中点；
3. 使用频率 `(1..4)^3` 的 `64` 个固定可分离正弦场；
4. 解析计算每个场的世界坐标梯度，并投影到每条射线的图像平面 `u/v` 方向；
5. 同时计算有符号积分的绝对值和逐点绝对值积分包络；
6. 每台相机先在射线、模态和双分量内平均，再让 active camera 等权；
7. 唯一主指标取“有符号相干平方能量 / 无符号包络平方能量”的平方根。

指标范围为 `[0,1]`，越高表示相消越少。唯一成功门在结果前固定为：13 套虚拟九相机中的每一个值都必须严格高于 13 套师兄标定九相机中的每一个值。模态、相位、积分点、坐标映射、相机权重和判决规则均未在看到结果后调整。

审计共覆盖 39 套几何、`131359` 条 active rays 和 `8406976` 个中点样本。它是几何机制诊断，离线 forward-equivalent 探针为 0，逻辑部署账为 `0A+0A^T`，训练参数为 0；这些数字不能写成部署速度或算法收益。

## 严格数值结果

| 几何族 | 主指标 min / median / max | 64 模态中最弱相干比的 min / median / max | 模态相干比中位数的 min / median / max |
| --- | ---: | ---: | ---: |
| 师兄九相机标定 | `0.63212 / 0.64597 / 0.66702` | `0.16959 / 0.18287 / 0.36560` | `0.62610 / 0.63822 / 0.66942` |
| 虚拟环形九相机 | `0.61902 / 0.62922 / 0.64015` | `0.35386 / 0.42068 / 0.45062` | `0.62767 / 0.63138 / 0.64648` |
| 虚拟环形十二相机，仅诊断 | `0.62742 / 0.62848 / 0.63169` | `0.42049 / 0.42699 / 0.44733` | `0.62249 / 0.62367 / 0.62514` |

虚拟九相机确实在“单个最弱模态”上更好，但预注册主指标聚合全部模态能量后，只赢 `7/169` 个跨族配对，且整体中位数更低。诊断指标不能替代唯一主指标；十二相机也不能事后升级为相机数量收益。

## 独立复算

完全独立的第二实现不导入正式 runner，自行重建相机与虚拟 rig、裁剪射线，并用显式逐模态世界梯度数组重新计算 39 套几何、64 个模态比和全部跨族判决。状态为 `PASS_INDEPENDENT_RECOMPUTATION_SIGNED_LINE_CANCELLATION_ATTRIBUTION_V212`，15 项有效性检查全部通过。

正式与独立逐几何指标最大差为 `4.44e-16`，逐模态相干比最大差为 `3.33e-16`，汇总最大差为 `2.22e-16`，相机顺序反转差为 0。两套实现的离散判决完全一致。

## 是否成功、是否突破

- **诊断执行成功：** 正式与独立第二实现一致，足以否定预注册解释。
- **科学主假设失败：** 预期方向只有 `7/169`，远未达到 `169/169`。
- **关闭范围有限：** 只关闭当前 64 模态、固定相位和固定聚合的相消标量。
- **不是算法结果：** 没有 predictor、warm start、物理 replay 或 exact-call 减少。
- **不是资源或外部结果：** 没有 wall/RSS、新外门或配对真实 BOST。
- **没有突破：** `algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

## 路线动作

关闭当前固定有符号射线相消标量，不事后修改模态族、相位、积分、相机权重或阈值。v210-v212 共同说明：实际低模 forward 谱有方向性，但局部无符号覆盖与当前相消比都不足以解释 reference 差异。下一步只接受工况匹配的真实二维 BOS 双分量位移与映射，或一个与三条已审计机制都物理上不同、结果前冻结的假设。预测器、资源门、神经训练和 GPU 继续不授权。

---

# v212: The Fixed Signed-Line Cancellation Scalar Does Not Explain the Case 5 Reference Difference

## Plain-language conclusion

v210 found a strongly directional global spectral floor in the actual low-mode forward response, while v211 ruled out the fixed unsigned local-coverage lower tail. v212 therefore tests a physically different bridge between them: even if local coverage is not weak, could positive and negative low-frequency phase cancel along rays and make the five-camera reference inadequate?

The answer is still no. Under the preregistered signed-line coherence ratio, virtual nine cameras exceed the supplied family in only `7/169` cross-family comparisons; strict success requires `169/169`. The supplied and virtual-nine medians are `0.64597` and `0.62922`. The sealed decision is `FAIL_SIGNED_LINE_CANCELLATION_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V212`.

This closes only the current scalar: 64 fixed low-frequency sine modes, 64-point midpoint ray integration, two u/v components, and equal-camera aggregation. It does not establish that every signed phase structure is irrelevant, nor does it negate v210's directional actual-forward Gram. It shows that the v210 spectral difference cannot be reduced to this particular per-ray positive-negative cancellation ratio.

## What was done and why

The audit reads reported geometry only. It reads no density, 2D observation, reconstruction, residual, or previous scientific metric array. For each geometry it:

1. clips every detector ray to the fixed 3D reconstruction box;
2. takes `64` fixed open midpoint samples on each active segment;
3. uses the `64` fixed separable sine fields with frequencies `(1..4)^3`;
4. analytically evaluates each field's world-coordinate gradient and projects it onto ray-specific image-plane `u/v` directions;
5. computes both the absolute signed integral and the integral of the pointwise absolute envelope;
6. averages within each camera before giving every active camera equal weight; and
7. defines the unique primary as the square root of coherent signed energy divided by unsigned-envelope energy.

The ratio lies in `[0,1]`; larger means less cancellation. Strict success requires all 13 virtual-nine values to exceed all 13 supplied-nine values. Modes, phase, quadrature, coordinate mapping, camera weighting, and the decision rule were fixed before reading results.

The audit covers 39 geometries, `131359` active rays, and `8406976` midpoint samples. It is a geometry mechanism diagnostic with zero offline forward-equivalent probes, a logical `0A+0A^T` deployment ledger, and zero trainable parameters. These counts are not deployment speed or algorithmic gains.

## Strict numerical result

| Geometry family | Primary min / median / max | Weakest of 64 mode ratios min / median / max | Median mode ratio min / median / max |
| --- | ---: | ---: | ---: |
| Supplied nine-camera | `0.63212 / 0.64597 / 0.66702` | `0.16959 / 0.18287 / 0.36560` | `0.62610 / 0.63822 / 0.66942` |
| Virtual-ring nine-camera | `0.61902 / 0.62922 / 0.64015` | `0.35386 / 0.42068 / 0.45062` | `0.62767 / 0.63138 / 0.64648` |
| Virtual-ring twelve-camera, diagnostic only | `0.62742 / 0.62848 / 0.63169` | `0.42049 / 0.42699 / 0.44733` | `0.62249 / 0.62367 / 0.62514` |

Virtual nine cameras are better on the single weakest mode, but after the preregistered primary aggregates energy over all modes they win only `7/169` cross-family pairs and have a lower median. Diagnostics cannot replace the unique primary, and twelve cameras cannot be upgraded after the result into a camera-count benefit claim.

## Independent recomputation

A fully independent second implementation does not import the formal runner. It independently rebuilds supplied and virtual geometry, clips all rays, and uses explicit per-mode world-gradient arrays to recompute all 39 geometries, 64 mode ratios, and every cross-family decision. Its status is `PASS_INDEPENDENT_RECOMPUTATION_SIGNED_LINE_CANCELLATION_ATTRIBUTION_V212`, with all 15 validity checks passing.

Maximum formal-independent geometry-metric difference is `4.44e-16`, maximum mode-ratio difference is `3.33e-16`, maximum summary difference is `2.22e-16`, and camera-order reversal difference is zero. Both implementations produce the same discrete decision.

## Success and breakthrough boundary

- **The diagnostic executes successfully:** formal and independent implementations agree, so the preregistered explanation is falsified.
- **The scientific primary fails:** only `7/169` comparisons move in the expected direction, far below `169/169`.
- **The closure is narrow:** only the current 64-mode, fixed-phase, fixed-aggregation cancellation scalar closes.
- **This is not an algorithm result:** no predictor, warm start, physical replay, or exact-call reduction is produced.
- **This is not resource or external evidence:** no wall/RSS, new external gate, or paired real-BOST data are used.
- **No breakthrough is claimed:** `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.

## Route action

Close the current fixed signed-line cancellation scalar. Do not change its mode family, phase, integration, camera weighting, or threshold after seeing the result. Across v210-v212, the actual low-mode forward spectrum remains directional, but neither unsigned local coverage nor this cancellation ratio explains reference adequacy. Continue only with condition-matched real two-component BOS displacement and its mapping, or with a preregistered hypothesis physically distinct from all three audited mechanisms. Predictor, resource, neural-training, and GPU stages remain unauthorized.
