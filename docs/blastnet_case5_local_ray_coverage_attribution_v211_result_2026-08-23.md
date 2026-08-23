# v211：局部射线覆盖下尾方向相反，关闭这一 reference 充分性解释

## 讲人话结论

v210 发现，固定低模全局谱下限在 `169` 个跨族比较中有 `167` 个偏向通过的虚拟九相机，但两族仍有重叠。v211 因而检验一个物理上不同的问题：是不是师兄标定族在局部空间里留下了更弱的射线覆盖区域，导致同一个三维场和同一个 K16 reference 失败？

结果不是。结果前固定的局部下尾主指标在 `169` 个比较中，虚拟九相机严格高于师兄标定族的次数是 `0/169`；反方向是 `169/169`。师兄标定族主指标中位数为 `0.12501`，虚拟九相机为 `0.07912`。正式判决是 `FAIL_LOCAL_RAY_COVERAGE_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V211`。

这个负结果必须收窄解释。虚拟九相机的逐体素局部下限中位数反而更高：`0.20784` 对 `0.14353`。因此差异集中在空间分布的下 10% 尾部，并不代表虚拟几何在每个位置都更弱。v211 关闭的是这一种固定、归一化、局部下 10% 标量，不是“所有局部几何都无关”。结合 v210，当前更可信的解释仍是全局低模耦合和条件结构。

## 做了什么，为什么这样做

本轮只读取 reported geometry，不读取密度场、二维观测、重建、残差或父实验科学数组。对每条有效射线：

1. 把射线裁剪到固定三维重建盒；
2. 在盒内取 `64` 个固定中点样本；
3. 用三线性权重把样本沉积到 `32x16x16` 网格；
4. 令每个 active camera 的总权重相等；
5. 在每个体素累积横向灵敏度张量 `I - d d^T`；
6. 排除一层边界，保留 `30x14x14 = 5880` 个内部张量；
7. 用该几何内部张量的平均 trace 做归一化。

唯一主指标是每个体素最小特征值的 `10th-percentile-higher`。唯一成功门要求：13 套虚拟九相机中的每一个值，都严格高于 13 套师兄标定九相机中的每一个值，即 `169/169`。分位数、采样数、坐标映射、边界、归一化、张量定义和 tie-break 都在读结果前固定。

审计覆盖 13 套师兄九相机、13 套虚拟环形九相机，以及只作诊断的 13 套虚拟环形十二相机，共 39 行几何、`131359` 条 active rays 和 `8406976` 个射线中点样本。所有计算是几何代数，不调用 forward；离线 forward-equivalent 探针为 0，部署账为 `0A+0A^T`，训练参数为 0。

## 严格数值结果

| 几何族 | 局部下 10% 主指标 min / median / max | 逐体素局部下限中位数 min / median / max | 局部 trace 下 10% min / median / max |
| --- | ---: | ---: | ---: |
| 师兄九相机标定 | `0.10381 / 0.12501 / 0.16574` | `0.12249 / 0.14353 / 0.18797` | `0.88368 / 0.89335 / 0.89568` |
| 虚拟环形九相机 | `0.07348 / 0.07912 / 0.08570` | `0.17681 / 0.20784 / 0.21196` | `0.54164 / 0.54847 / 0.56830` |
| 虚拟环形十二相机，仅诊断 | `0.09488 / 0.09624 / 0.09900` | `0.22413 / 0.22496 / 0.22732` | `0.54790 / 0.55594 / 0.57210` |

主指标上，师兄标定族的最小值 `0.10381` 仍高于虚拟九相机的最大值 `0.08570`，所以不是“稍微没过门”，而是完整反向分离。十二相机只作诊断，不能事后替换唯一主判决，也不能据此宣称相机数量收益。

## 独立复算

完全独立的第二实现自行重建标定规范化、虚拟几何、射线裁剪、三线性沉积、局部张量、特征值、全部 39 行指标和 `169` 个判决，不导入正式实现。状态为 `PASS_INDEPENDENT_RECOMPUTATION_LOCAL_RAY_COVERAGE_ATTRIBUTION_V211`，15 项检查全部通过。

正式与独立指标最大差为 `5.33e-15`，局部特征值最大差为 `1.44e-15`，汇总最大差为 `4.44e-15`，相机顺序反转差为 0。正式和独立判决完全一致。

## 是否成功、是否突破

- **机制诊断执行成功：** 正式与独立第二实现一致，结果足以否定预注册假设。
- **主假设失败：** 预期方向为 `0/169`，反方向为 `169/169`。
- **关闭范围有限：** 只关闭固定的归一化局部下 10% 射线覆盖标量；不扩大为所有局部几何都无关。
- **不是算法结果：** 没有 predictor、warm start、物理 replay 或 exact-call 减少。
- **不是资源或外部结果：** 没有 wall/RSS、未打开新外门，也没有真实 BOST 配对数据。
- **没有突破：** `algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

## 路线动作

关闭当前局部射线覆盖下尾标量，不事后改分位数、归一化、采样数、坐标映射或张量公式。v210 的全局低模耦合/条件结构仍是更可信的归因方向，但它还不是部署算法。只有工况匹配的真实二维 BOS 双分量位移与映射到位，或先冻结另一条物理上真正不同、可独立证伪的机制，才继续算法门。预测器、资源门、神经训练和 GPU 继续不授权。

---

# v211: The Local Ray-Coverage Lower Tail Moves in the Opposite Direction, Closing This Reference-Adequacy Explanation

## Plain-language conclusion

v210 found that a fixed global low-mode spectral floor favors the passing virtual-nine family in `167` of `169` cross-family comparisons, although the family ranges still overlap. v211 therefore tests a physically different question: does the supplied family leave locally weak ray-coverage regions that explain why the same 3D fields and K16 reference fail?

It does not. Under the preregistered local lower-tail primary, virtual nine cameras are strictly higher than the supplied family in `0/169` comparisons; the opposite direction holds in `169/169`. The primary median is `0.12501` for the supplied family and `0.07912` for virtual nine cameras. The sealed decision is `FAIL_LOCAL_RAY_COVERAGE_DOES_NOT_EXPLAIN_CASE5_REFERENCE_V211`.

The negative result has a narrow boundary. The median voxelwise local floor is actually higher for virtual nine cameras: `0.20784` versus `0.14353`. The difference is therefore concentrated in the lower 10% spatial tail, not in every location. v211 closes this fixed, normalized, lower-10-percent scalar; it does not establish that all local geometry is irrelevant. Together with v210, global low-mode coupling and conditioning remain the more credible explanation.

## What was done and why

The audit reads reported geometry only. It reads no density field, 2D observation, reconstruction, residual, or parent scientific metric. For every valid ray it:

1. clips the ray to the fixed reconstruction box;
2. takes `64` fixed midpoint samples inside the box;
3. deposits each sample onto the `32x16x16` grid with trilinear weights;
4. gives every active camera equal total weight;
5. accumulates transverse-sensitivity tensor `I - d d^T` at every voxel;
6. excludes one boundary layer, retaining `30x14x14 = 5880` interior tensors; and
7. normalizes by the geometry's mean interior trace.

The unique primary is the `10th-percentile-higher` of the voxelwise minimum eigenvalue. Success requires every one of 13 virtual-nine values to exceed every one of 13 supplied-nine values, or `169/169`. Quantile, sample count, coordinate mapping, boundary, normalization, tensor definition, and tie-break were fixed before reading results.

The audit covers 13 supplied nine-camera rigs, 13 virtual-ring nine-camera rigs, and 13 diagnostic-only virtual-ring twelve-camera rigs: 39 geometry rows, `131359` active rays, and `8406976` midpoint samples. All operations are geometry algebra. There are zero offline forward-equivalent probes, a `0A+0A^T` deployment ledger, and zero trainable parameters.

## Strict numerical result

| Geometry family | Local lower-10% primary min / median / max | Voxelwise local-floor median min / median / max | Local-trace lower-10% min / median / max |
| --- | ---: | ---: | ---: |
| Supplied nine-camera | `0.10381 / 0.12501 / 0.16574` | `0.12249 / 0.14353 / 0.18797` | `0.88368 / 0.89335 / 0.89568` |
| Virtual-ring nine-camera | `0.07348 / 0.07912 / 0.08570` | `0.17681 / 0.20784 / 0.21196` | `0.54164 / 0.54847 / 0.56830` |
| Virtual-ring twelve-camera, diagnostic only | `0.09488 / 0.09624 / 0.09900` | `0.22413 / 0.22496 / 0.22732` | `0.54790 / 0.55594 / 0.57210` |

For the primary, the supplied minimum `0.10381` remains above the virtual-nine maximum `0.08570`. The result is therefore complete separation in the opposite direction, not a near miss. Twelve cameras remain diagnostic only and cannot replace the unique primary after the result or support a camera-count benefit claim.

## Independent recomputation

A fully independent second implementation rebuilds calibration canonicalization, virtual geometry, ray clipping, trilinear deposition, local tensors, eigenvalues, all 39 geometry rows, and all `169` decisions without importing the formal implementation. Its status is `PASS_INDEPENDENT_RECOMPUTATION_LOCAL_RAY_COVERAGE_ATTRIBUTION_V211`, with all 15 checks passing.

Maximum formal-independent metric difference is `5.33e-15`, maximum local-eigenvalue difference is `1.44e-15`, maximum summary difference is `4.44e-15`, and camera-order reversal difference is zero. Formal and independent decisions match exactly.

## Success and breakthrough boundary

- **The mechanism diagnostic succeeds operationally:** formal and independent implementations agree, so the preregistered hypothesis is falsified.
- **The primary hypothesis fails:** the expected direction is `0/169`; the opposite direction is `169/169`.
- **The closure is narrow:** only the fixed normalized lower-10-percent local ray-coverage scalar is closed; the result does not generalize to all local geometry.
- **This is not an algorithm result:** no predictor, warm start, physical replay, or exact-call reduction is produced.
- **This is not resource or external evidence:** no wall/RSS or new external gate runs, and no paired real-BOST data are used.
- **No breakthrough is claimed:** `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.

## Route action

Close the current local ray-coverage lower-tail scalar. Do not change its quantile, normalization, sample count, coordinate map, or tensor definition after seeing the result. v210's global low-mode coupling and conditioning remain the more credible attribution direction, but they are not yet a deployable algorithm. Continue only when condition-matched real two-component BOS displacement and its mapping arrive, or after preregistering another physically distinct and independently falsifiable mechanism. Predictor, resource, neural-training, and GPU stages remain unauthorized.
