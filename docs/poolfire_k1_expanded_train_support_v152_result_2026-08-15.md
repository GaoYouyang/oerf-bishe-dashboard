# v152：新增同功率训练轨迹仍未关闭 5 相机跨尺寸支持缺口

## 这次真正问了什么

v151 说明五条已开封 PoolFire 训练轨迹之间的可观测状态覆盖很不均衡。v152 没有继续堆模型，而是先加入一条此前未进入审计、但已经属于公开 train split 的同功率不同尺寸轨迹，然后只问一个更基础的问题：扩大训练工况后，p33 的两个尺寸条件能否在 `5/7/9/12` 相机下互相支持？

本轮保持以下约束不变：

- 只读公开 train split，validation/test 和独立外部门继续封存；
- 新增轨迹生成 `740` 个样本，来自 5 帧、4 档相机数量和 37 个冻结扰动条件；
- 与原五轨迹合并后共有 `4,440` 个样本和 `36,630` 个 active camera rows；
- primary 是每台相机 45 维 deployment-visible observation/K1-residual/K1-dual/geometry 状态；
- 标准化、支持阈值和最近邻都只由 fold 内训练轨迹决定；
- 不读取 Krylov 系数目标，不拟合 predictor，不做物理 replay，也不打开 validation/test。

新增轨迹的离线状态构造使用 `740A + 740A^T`。这只是建立支持审计输入，不是部署算法成本，也不能写成 exact-call 节省。

## 独立确认的结果

加入同功率不同尺寸轨迹后，原 p33-s01 的四档相机支持率变化为：

| active cameras | 扩展前 | 扩展后 | 冻结 90% 门 |
|---:|---:|---:|---:|
| 5 | 76.86% | 83.68% | 未通过 |
| 7 | 87.18% | 91.81% | 通过 |
| 9 | 93.09% | 97.24% | 通过 |
| 12 | 94.50% | 97.79% | 通过 |

新增的 p33-s03 作为完整留出轨迹时，`5/7/9/12` 相机支持率分别为 `95.68% / 98.92% / 98.32% / 98.87%`，四档都通过。新增轨迹还确实救回了原 p33-s01 的 `265` 个 active camera rows，占该轨迹全部 active rows 的 `4.34%`。

但 primary 要求 p33 两个尺寸条件的四档相机数全部至少达到 `90%`。p33-s01 的 5 相机支持率仍只有 `83.68%`，所以正式科学判决是：

`FAIL_P33_SAME_POWER_MUTUAL_SUPPORT_V152`

便宜的样本内中心/RMS 归一化 control 也没有修复该缺口：p33-s01 的 5 相机支持率为 `84.22%`。因此失败不能简单归因于幅值或中心尺度没有归一化。

## 独立复算

第二实现独立重建新增轨迹的一步 CGLS 状态、每相机特征、fold-only normalization 和 SciPy 最近邻距离，然后才读取正式数组做事后比较。`17/17` 项检查全部通过：

- 新增物理状态最大差：`1.78e-14`；
- 相机特征最大差：`1.78e-14`；
- 距离最大差：`2.00e-15`；
- normalization 与汇总数字最大差：`0`；
- nearest indices、support flags 和科学判决逐项一致；
- 正式结果树与输入在验证前后不变。

正式与独立实现仍共享冻结的 physics kernels，所以 `end_to_end_physics_independence_proven=false`。但本次支持判决不是由网络、断网、随机训练或数值分叉造成的。

## 这次失败改变了什么

同功率新增轨迹不是完全无效：它让 7/9/12 相机的原 p33 条件全部过门，并且新增轨迹自身四档相机数都受原训练集支持。真正剩下的是**少相机下的跨尺寸覆盖缺口**。这比“再加一条同功率数据就会解决”更具体，也说明当前不应直接训练 predictor 或租 GPU。

下一门改成一个结果前冻结、仍然不读目标的坐标规范化诊断：先比较便宜的多视角仿射 control，再测试一个由 observation 与 reported geometry 生成的最小单调输运 warp。它只问坐标/尺度变化能否把 5 相机跨尺寸支持率推过 90%。如果仍失败，就停止当前跨轨迹预测路线，等待真正更广的训练工况或组内实验数据，而不是用更大网络硬补覆盖缺口。

当前边界：

- `predictor_training_authorized=false`；
- `physical_replay=false`；
- `gpu_rental_authorized=false`；
- `algorithm_breakthrough=false`；
- `resource_speedup=false`；
- `external_generalization=false`；
- `curved_ray_validated=false`；
- `real_bost=false`；
- `paper_success=false`。

公开证据：

- `docs/poolfire_k1_expanded_train_support_v152_public_summary.json`
- `assets/figures/poolfire_k1_expanded_train_support_v152.png`

## English checkpoint

v152 adds one previously unused public PoolFire training trajectory at the same power and a different size. It does not fit a predictor or read any Krylov-coordinate target. The audit asks only whether the two p33 size conditions mutually support each other under `5/7/9/12` active cameras using deployment-visible observation, exact-K1 residual/dual state, and reported geometry.

The added trajectory contributes `740` samples. The combined train-only audit contains `4,440` samples and `36,630` active camera rows. All normalization and support thresholds are fold-train-only; validation and test remain sealed.

For the original p33-s01 trajectory, support changes from `76.86% / 87.18% / 93.09% / 94.50%` to `83.68% / 91.81% / 97.24% / 97.79%` for `5/7/9/12` cameras. The newly added p33-s03 trajectory reaches `95.68% / 98.92% / 98.32% / 98.87%` when held out. The added trajectory rescues `265` p33-s01 camera rows, or `4.34%`, but the five-camera stratum remains below the frozen `90%` gate. A cheap within-sample center/RMS normalization control also fails at `84.22%`.

An independent implementation rebuilds the one-step CGLS state, camera features, fold normalization, and SciPy nearest-neighbor support. All `17` checks pass; maximum state, feature, and distance differences are `1.78e-14`, `1.78e-14`, and `2.00e-15`, while all discrete decisions match.

The scientific decision is `FAIL_P33_SAME_POWER_MUTUAL_SUPPORT_V152`. Same-power coverage helps but is insufficient under sparse views across size conditions. This is not a reconstruction, predictor, exact-call saving, resource, external-generalization, curved-ray, real-BOST, or paper-success result. Predictor fitting and GPU rental remain unauthorized. The next target-free gate tests observation-derived multiview coordinate canonicalization; if the five-camera gap persists, this predictor route stops until genuinely broader training conditions become available.
