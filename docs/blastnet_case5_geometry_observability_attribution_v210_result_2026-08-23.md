# v210：几何可观测性强烈偏向虚拟环形族，但固定谱下限没有严格分离

## 讲人话结论

v209 已经证明：同一批 Case 5 三维场和同一个 Zero-CGLS K16，在师兄提供的九相机标定族下失败，在虚拟环形九相机下却完整通过。因此 v210 不再重建场，也不读取密度、二维观测、残差或父实验指标，而是只看相机几何，问一个更窄的问题：能不能用一个事先固定的几何可观测性数字，把失败与通过的九相机几何严格分开？

结果给出强烈但不充分的证据。虚拟九相机与师兄九相机共有 `13 x 13 = 169` 个跨族配对，其中 `167/169` 个配对的虚拟几何谱下限更高，方向性优势为 `98.8166%`。师兄标定族的主指标中位数为 `0.02146`，虚拟九相机为 `0.30756`，高约 `14.3` 倍；中位归一条件数则从 `152.13` 降到 `11.65`。

但是严格门要求 `169/169`。虚拟九相机的最小值 `0.10932` 仍低于师兄标定族的最大值 `0.13159`，所以存在两个反向配对。正式判决是 `PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210`：几何/条件性是主要因素，但这个固定 64 维低模谱下限不能单独充当 reference 充分性的分类器。

## 做了什么，为什么这样做

唯一主指标使用 `32x16x16` 网格上的固定 64 维低频 Dirichlet 正弦基。对每套相机几何，把 64 个基方向经过冻结的 exact straight-ray forward，得到响应矩阵 `M`，构造

`G = M^T M / observation_scalar_count`

并按迹归一化为

`G_norm = 64 G / trace(G)`。

主指标是 `G_norm` 的最小非负特征值。它衡量该相机布局对固定低频三维模式中最弱方向的观测强度。所有 64 个方向、归一化、非负特征值规则和严格 `169/169` 门都在读取结果前固定。

审计覆盖三组各 13 套几何：师兄提供的九相机、虚拟环形九相机，以及只作诊断的虚拟环形十二相机。共 39 行几何、`2496` 次离线 forward-equivalent 探针。这个账是离线几何诊断，不是部署算法成本；本轮不读任何科学场或观测数组，部署账为 `0A+0A^T`，训练参数为 0。

## 严格数值结果

| 几何族 | 主指标 min / median / max | 归一条件数 min / median / max | 有效秩 min / median / max | 横向灵敏度下限 min / median / max |
| --- | ---: | ---: | ---: | ---: |
| 师兄九相机标定 | `0.01186 / 0.02146 / 0.13159` | `22.92 / 152.13 / 282.54` | `0.7570 / 0.7883 / 0.8786` | `0.3675 / 0.4348 / 0.5786` |
| 虚拟环形九相机 | `0.10932 / 0.30756 / 0.34008` | `10.02 / 11.65 / 30.75` | `0.8437 / 0.8725 / 0.8856` | `0.5466 / 0.6779 / 0.6931` |
| 虚拟环形十二相机，仅诊断 | `0.39399 / 0.40561 / 0.42763` | `8.01 / 8.56 / 8.81` | `0.8988 / 0.9001 / 0.9032` | `0.7507 / 0.7560 / 0.7592` |

虚拟九相机在绝大多数跨族配对中同时表现出更高谱下限、更低条件数和更高有效秩，支持 v209 的几何/覆盖归因。十二相机诊断进一步朝有利方向移动，但它不参与唯一主判决，也不能事后被当作相机数量收益。

## 独立复算与失效记录

完全独立的第二实现重建固定基、forward 响应、Gram 矩阵、全部谱指标和 `169` 个跨族判决。状态为 `PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_OBSERVABILITY_ATTRIBUTION_V210`。正式与独立主指标最大差为 `2.25e-11`，Gram 特征值最大差为 `1.24e-14`，横向张量特征值最大差为 `4.00e-15`，相机顺序反转差为 0；全部独立门通过。

执行过程中保留了两次不能当作科学结果的失效：第一次在写任何结果前发现 active-ray 行数映射错误，直接 fail-closed；第二次离散判决一致，但浮点归约顺序令相机反转差超过数值门，因此封存为 inconclusive。最终只把矩阵行按物理键做确定性规范化，未改主指标、数据、阈值或判决规则，独立实现使用不同的稳定排序重算后通过。这些是工程与数值完整性记录，不是算法增量。

## 是否成功、是否突破

- **归因诊断部分成功：** `167/169` 的跨族优势和大幅条件数差说明几何/条件性是主要因素。
- **严格分类失败：** 两族区间仍重叠，唯一主门要求的 `169/169` 没有达到。
- **关闭一条解释：** 不再把固定 64 维低模谱下限当作充分性分类器，也不事后调阈值、换基或挑别的指标包装成功。
- **不是部署算法：** 不读取部署观测，不生成 warm start，没有物理 replay，也没有 exact-call 减少。
- **不是外部或真实结果：** Case 5 已经开封，二维投影仍为 synthetic straight-ray；没有 wall/RSS 或真实 BOST。
- **没有突破：** `algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

## 路线动作

v210 把“几何完全无关”排除掉，却也证明一个漂亮的低模谱指标还不足以决定 reference 是否合格。下一步不继续修饰这一个指标。只有拿到工况匹配的真实二维 BOS 双分量位移与映射，或先冻结一个物理上真正不同、可证伪的几何机制，才继续算法门。预测器、资源门、神经训练和 GPU 仍不授权。

---

# v210: Geometry Observability Strongly Favors the Virtual-Ring Family, but the Fixed Spectral Floor Does Not Strictly Separate It

## Plain-language conclusion

v209 established that the same Case 5 fields and zero-start CGLS K16 fail under the supplied nine-camera family but pass completely under virtual-ring nine-camera geometry. v210 therefore does not reconstruct fields or read density, 2D observations, residuals, or parent metrics. It asks a narrower geometry-only question: can one preregistered observability number strictly separate the failing and passing nine-camera families?

The answer is strong but insufficient evidence. Across `13 x 13 = 169` cross-family pairs, the virtual-nine spectral floor is higher in `167/169`, or `98.8166%`. The primary median rises from `0.02146` for the supplied family to `0.30756` for virtual nine cameras, about `14.3` times higher. Median normalized condition number falls from `152.13` to `11.65`.

The strict gate, however, requires `169/169`. The minimum virtual-nine value, `0.10932`, remains below the maximum supplied-family value, `0.13159`, reversing two pairwise comparisons. The sealed decision is `PARTIAL_OVERLAPPING_GEOMETRY_ONLY_OBSERVABILITY_EVIDENCE_V210`: geometry and conditioning are major factors, but this fixed 64-mode spectral floor is not a sufficient reference-adequacy classifier by itself.

## What was done and why

The unique primary uses a fixed 64-dimensional low-frequency Dirichlet sine span on the `32x16x16` grid. For each camera geometry, the 64 basis directions pass through the frozen exact straight-ray forward to form response matrix `M`. The audit constructs

`G = M^T M / observation_scalar_count`

and trace-normalizes it as

`G_norm = 64 G / trace(G)`.

The primary is the smallest nonnegative eigenvalue of `G_norm`, measuring the weakest observed direction in this fixed low-frequency 3D span. All 64 directions, normalization, eigenvalue convention, and strict `169/169` gate were fixed before reading results.

The audit covers three families of 13 rigs each: supplied nine-camera, virtual-ring nine-camera, and diagnostic-only virtual-ring twelve-camera geometries. That is 39 geometry rows and `2,496` offline forward-equivalent probes. This is offline geometry-diagnostic cost, not deployment cost. No scientific field or observation array is read, the deployment ledger is `0A+0AT`, and there are zero trainable parameters.

## Strict numerical result

| Geometry family | Primary min / median / max | Normalized condition min / median / max | Effective rank min / median / max | Transverse floor min / median / max |
| --- | ---: | ---: | ---: | ---: |
| Supplied nine-camera | `0.01186 / 0.02146 / 0.13159` | `22.92 / 152.13 / 282.54` | `0.7570 / 0.7883 / 0.8786` | `0.3675 / 0.4348 / 0.5786` |
| Virtual-ring nine-camera | `0.10932 / 0.30756 / 0.34008` | `10.02 / 11.65 / 30.75` | `0.8437 / 0.8725 / 0.8856` | `0.5466 / 0.6779 / 0.6931` |
| Virtual-ring twelve-camera, diagnostic only | `0.39399 / 0.40561 / 0.42763` | `8.01 / 8.56 / 8.81` | `0.8988 / 0.9001 / 0.9032` | `0.7507 / 0.7560 / 0.7592` |

Virtual nine cameras have higher spectral floors, lower condition numbers, and higher effective rank in nearly every cross-family comparison, supporting v209's geometry and coverage attribution. The twelve-camera diagnostic moves further in the favorable direction, but it is not part of the unique primary and cannot be used post hoc as a camera-count claim.

## Independent recomputation and failed attempts

A fully independent implementation rebuilds the fixed span, forward responses, Gram matrices, every spectral metric, and all `169` cross-family decisions. Its status is `PASS_INDEPENDENT_RECOMPUTATION_GEOMETRY_OBSERVABILITY_ATTRIBUTION_V210`. Maximum formal-independent primary difference is `2.25e-11`, maximum Gram-eigenvalue difference is `1.24e-14`, maximum transverse-tensor eigenvalue difference is `4.00e-15`, and the camera-order reversal difference is zero. Every independent gate passes.

Two attempts are preserved as non-scientific failures. The first found an active-ray row-mapping error before writing any result and failed closed. The second reproduced all discrete decisions, but floating-point reduction order exceeded the frozen camera-reversal tolerance and was sealed as inconclusive. The final execution only canonicalizes matrix rows by physical keys; it does not change the primary, data, threshold, or decision. A differently implemented stable sort independently reproduces the result. These records establish engineering and numerical integrity, not an algorithmic increment.

## Success and breakthrough boundary

- **The attribution diagnostic partly succeeds:** `167/169` cross-family superiority and the condition-number gap identify geometry and conditioning as major factors.
- **Strict classification fails:** the ranges overlap, so the unique `169/169` gate is not met.
- **One explanation is closed:** the fixed 64-mode spectral floor is not a sufficient classifier, and no threshold, basis, or alternate diagnostic will be selected after seeing the result.
- **This is not a deployable algorithm:** no deployment observation, warm start, physical replay, or exact-call reduction is produced.
- **This is not external or real-data evidence:** Case 5 is already open, and projections remain synthetic straight rays; no wall/RSS or real-BOST stage runs.
- **No breakthrough is claimed:** `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.

## Route action

v210 rules out the idea that geometry is irrelevant, while showing that one attractive low-mode spectral metric does not determine reference adequacy. Do not keep refining this metric. Continue only when condition-matched real two-component BOS displacement and its mapping arrive, or after preregistering a physically distinct, falsifiable geometry mechanism. Predictor, resource, neural-training, and GPU stages remain unauthorized.
