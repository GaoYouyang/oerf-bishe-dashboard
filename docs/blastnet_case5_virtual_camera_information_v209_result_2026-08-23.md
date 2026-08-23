# v209：虚拟环形几何救回 Case 5 reference，不能归功于增加相机数量

## 讲人话结论

v208 发现，师兄提供的 13 套九相机标定下，Zero-CGLS K16 仍只有 `0/546` 个严格单元和 `0/13` 个完整组通过。v209 改问一个更具体的物理问题：失败究竟是 Case 5 三维场本身不可重建，还是这组相机几何没有提供足够的三维信息？

在同一批 42 个 Case 5 三维场、同一 `32x16x16` 网格和同一 Zero-CGLS K16 下，v209 使用此前独立验证过的虚拟环形相机几何。九相机控制已经达到 `546/546` 个严格单元和 `13/13` 个完整组；十二相机 primary 也达到 `546/546` 和 `13/13`。

因为九相机控制本身已经全过，不能把结果写成“增加三台相机救回了重建”。精确判决是 `PASS_SYNTHETIC_RING_GEOMETRY_NOT_CARDINALITY_RESCUES_CASE5_REFERENCE_V209`：改善来自虚拟环形几何/覆盖，而不是已证明的相机数量收益。它也说明 v208 的失败不是 Case 5 数据、网格或 K16 本身必然不可重建。

## 做了什么，为什么这样做

v209 保持三维场、网格、straight-ray forward、绝对精度门和 K16 迭代深度不变，只替换相机几何。每个虚拟 rig 都有嵌套的九相机控制和十二相机 primary：九相机严格取冻结十二相机顺序的前九台，因此两臂共享的相机参数完全相同。两臂各覆盖 13 个 rig × 42 个场，即各 `546` 个单元。

这个设计可以拆开两个解释：

- 如果十二相机通过而九相机失败，才有资格说额外三台相机提供了必要信息。
- 如果九、十二相机都通过，只能说明这一虚拟几何族比师兄提供的九相机标定族更有利。

两臂的 K16 逻辑诊断账都为 `16A+16A^T`。这是 reference 充分性诊断，不是部署候选成本，也没有运行 wall/RSS。

## 严格数值结果

| K16 reference 几何 | 严格单元 | 完整组 | Field p90 范围 | Gradient p90 范围 | Observation p90 范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 师兄提供的九相机标定族（v208 父证据） | `0/546` | `0/13` | `0.7252-0.7608` | `0.5627-0.6795` | `0.0473-0.0567` |
| 虚拟环形九相机控制 | `546/546` | `13/13` | `0.3199-0.3515` | `0.6152-0.6581` | `0.0625-0.0689` |
| 虚拟环形十二相机 primary | `546/546` | `13/13` | `0.2764-0.3074` | `0.5516-0.5836` | `0.0573-0.0622` |
| 冻结 p90 门 | - | - | `<=0.50` | `<=0.75` | `<=0.20` |

虚拟九相机的 field / gradient / observation worst 范围分别为 `0.3237-0.3633 / 0.6189-0.6670 / 0.0636-0.0708`；虚拟十二相机分别为 `0.2778-0.3130 / 0.5534-0.5880 / 0.0582-0.0635`，也全部守住冻结 worst 门。

## 独立复算与残差闭环

独立第二实现重新构建虚拟 rigs、二维观测、K16 轨迹、三维场和全部指标。场最大相对差约为 `1.12e-9`，观测最大相对差约为 `6.82e-16`，逐单元指标与汇总最大差约为 `6.75e-12 / 6.41e-13`；所有离散通过判决一致。

原始独立比较把残差差除以一个接近零的残差范数，得到最高约 `8.25e-8`，超过 `1e-8` 数值门，因此按规则先记为 inconclusive。v209.2 没有重跑或改科学数组，而是对封存场和观测做结果前固定的残差方程闭环：统一以独立观测范数归一化，同时分别验证 `r = b - Ax`。

对 `2 × 13 × 42 = 1092` 个单元，独立残差与正式残差最大差为 `4.34e-9`；正式和独立各自的递归残差闭环最大误差为 `7.07e-16` 和 `2.04e-16`，正式与独立观测最大差为 `6.82e-16`。全部 11 项闭环检查通过，恢复正式科学判决。

## 是否成功、是否突破

- **机制诊断成功：** 虚拟环形九/十二相机 reference 均完整通过。
- **几何归因成立：** Case 5 在更有利的 straight-ray 相机几何下可由 K16 重建，v208 失败不是数据/网格/K16 单独决定的。
- **相机数量收益未成立：** 九相机已经全过，额外三台相机不能被记作必要贡献。
- **不是外部泛化：** Case 5 已经在 v207 打开，v209 是 post-open synthetic diagnostic。
- **没有资源或算法结论：** 未运行 wall/RSS，未减少 exact calls，没有训练模型；`algorithm_breakthrough=false`、`global_resource_speedup=false`。
- **没有真实 BOST：** 仍是公开 CFD 三维场与 straight-ray 虚拟投影；`real_bost=false`。

## 路线动作

下一门不再继续增加相机数量，也不直接训练预测器。应在结果前冻结一组只由 reported geometry 计算的可观测性、覆盖和条件数指标，对比“师兄提供的失败标定族”和“虚拟环形通过族”，定位究竟是哪种几何性质决定 K16 reference 充分性。只有这个归因门完成，才判断是否值得为真实相机布局设计几何选择或采集方案。

---

# v209: Virtual-Ring Geometry Rescues the Case 5 Reference, Not Additional Camera Count

## Plain-language conclusion

v208 found that zero-start CGLS K16 still achieved only `0/546` strict-safe cells and `0/13` complete groups under the 13 supplied nine-camera calibrations. v209 asks a more specific physical question: is Case 5 intrinsically unreconstructable, or does that camera family provide insufficient 3D information?

Using the same 42 Case 5 fields, the same `32x16x16` grid, and the same zero-start CGLS K16, v209 substitutes a previously independently validated virtual-ring geometry. The nine-camera control reaches `546/546` strict-safe cells and `13/13` complete groups. The twelve-camera primary also reaches `546/546` and `13/13`.

Because the nine-camera control already passes in full, the result cannot be reported as a rescue by three additional cameras. The exact verdict is `PASS_SYNTHETIC_RING_GEOMETRY_NOT_CARDINALITY_RESCUES_CASE5_REFERENCE_V209`: the supported cause is virtual-ring geometry or coverage, not established camera-count benefit. The result also shows that the v208 failure is not intrinsic to the Case 5 data, grid, or K16 alone.

## What was done and why

v209 preserves the 3D fields, grid, straight-ray forward model, absolute accuracy gates, and K16 depth while changing only camera geometry. Each virtual rig contains a nested nine-camera control and twelve-camera primary. The control uses exactly the first nine cameras in the frozen twelve-camera order, so all shared camera parameters are identical. Each arm covers 13 rigs times 42 fields, or `546` cells.

The design separates two explanations:

- Twelve cameras passing while nine fail would support necessary information from the additional three cameras.
- Nine and twelve both passing supports only a more favorable virtual geometry family relative to the supplied nine-camera family.

Both arms retain a `16A+16A^T` K16 diagnostic ledger. This is reference-adequacy cost, not deployment-candidate cost, and no wall/RSS stage runs.

## Strict numerical result

| K16 reference geometry | Strict cells | Complete groups | Field p90 range | Gradient p90 range | Observation p90 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplied nine-camera family (v208 parent) | `0/546` | `0/13` | `0.7252-0.7608` | `0.5627-0.6795` | `0.0473-0.0567` |
| Virtual-ring nine-camera control | `546/546` | `13/13` | `0.3199-0.3515` | `0.6152-0.6581` | `0.0625-0.0689` |
| Virtual-ring twelve-camera primary | `546/546` | `13/13` | `0.2764-0.3074` | `0.5516-0.5836` | `0.0573-0.0622` |
| Frozen p90 limit | - | - | `<=0.50` | `<=0.75` | `<=0.20` |

Virtual-nine field / gradient / observation worst ranges are `0.3237-0.3633 / 0.6189-0.6670 / 0.0636-0.0708`. Virtual-twelve ranges are `0.2778-0.3130 / 0.5534-0.5880 / 0.0582-0.0635`. Every group also clears the frozen worst limits.

## Independent recomputation and residual closure

The independent implementation rebuilds the virtual rigs, 2D observations, K16 trajectories, 3D fields, and all metrics. Maximum field and observation relative differences are about `1.12e-9` and `6.82e-16`; maximum cell-metric and summary differences are about `6.75e-12` and `6.41e-13`. Every discrete pass decision agrees.

The original independent comparison divided residual differences by a near-zero residual norm, producing a maximum near `8.25e-8` above the `1e-8` numerical gate, so it correctly failed closed as inconclusive. v209.2 neither reruns nor changes any scientific array. It performs a preregistered equation-closure adjudication on sealed fields and observations, normalizing consistently by the independent observation norm and checking `r = b - Ax` on both sides.

Across `2 × 13 × 42 = 1092` cells, the maximum independent-formal residual difference is `4.34e-9`. Formal and independent recursive residual closure errors are at most `7.07e-16` and `2.04e-16`, and the maximum formal-independent observation difference is `6.82e-16`. All 11 closure checks pass, restoring the formal scientific verdict.

## Success and breakthrough boundary

- **The mechanism diagnostic succeeds:** virtual-ring nine- and twelve-camera references pass completely.
- **Geometry attribution is supported:** Case 5 is reconstructable by K16 under a more favorable straight-ray geometry; the v208 failure is not determined by the data, grid, or K16 alone.
- **Camera-count benefit is not established:** nine cameras already pass, so the extra three cameras are not a necessary credited contribution.
- **This is not external generalization:** Case 5 was opened in v207, making v209 a post-open synthetic diagnostic.
- **No resource or algorithm claim:** no wall/RSS stage, exact-call reduction, or model training occurs; `algorithm_breakthrough=false` and `global_resource_speedup=false`.
- **No real BOST result:** this remains public CFD with synthetic straight-ray projections; `real_bost=false`.

## Route action

Do not keep increasing camera count or train a predictor yet. Preregister geometry-only observability, coverage, and conditioning measures that compare the failing supplied calibration family with the passing virtual-ring family. Identify which geometric property controls K16 reference adequacy before deciding whether a geometry-selection or acquisition-design mechanism is warranted for real camera layouts.
