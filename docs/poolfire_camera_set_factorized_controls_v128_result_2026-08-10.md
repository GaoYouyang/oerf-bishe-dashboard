# v128：PoolFire 变相机因子化误差与经典控制审计

> English title: **v128: Factorized camera perturbations and classical-control audit on PoolFire**  
> 更新日期 / Updated: 2026-08-10

## 一句话结论 / One-sentence result

v128 在五条已开封的 PoolFire CFD 轨迹上完成了 37 种 clean、观测噪声、相机旋转、平移、内参、综合位姿与联合扰动条件的因子化实验。结果说明：扰动会稳定抬高误差尾部，但当前更大的限制仍是相机数量和迭代深度；即使加入 geometry-PCGLS，`2A+2A^T` 预算仍不能达到 Zero-CGLS K4 的场精度。因此，下一步开发最小的 permutation-invariant SetDual-Warm 初始化器有明确的剩余空间，但这还不是算法突破。

v128 completes a factorized experiment over 37 clean, observation-noise, camera-rotation, translation, intrinsics, aggregate-pose, and combined conditions on five opened PoolFire CFD trajectories. Perturbations consistently increase error tails, but camera count and iteration depth remain the larger limitations. Even geometry-PCGLS cannot match the field accuracy of Zero-CGLS K4 under a `2A+2A^T` budget. This leaves a concrete development target for a minimal permutation-invariant SetDual-Warm initializer, but it is not yet an algorithmic breakthrough.

## 为什么做这一步 / Why this experiment was necessary

师兄提出了三个必须落实的数据条件：相机之间应相互独立、相机顺序可以交换、相机可以增加或删除；同时还要把观测噪声与位姿/标定误差加入 CFD 代理数据。v126 验证了这些数据机制，v127 给出了 clean / medium / stress 的整体难度图。v128 进一步把误差来源拆开，避免把“噪声造成的退化”“相机几何造成的退化”和“迭代预算不足”混为一谈。

The senior collaborator required three concrete data properties: cameras must be independent set elements, camera order must be exchangeable, and cameras must be addable or removable. Observation noise and pose/calibration errors also need to be injected into the CFD proxy. v126 validates those mechanics and v127 produces an aggregate clean/medium/stress difficulty map. v128 separates the error sources so that noise degradation, geometry degradation, and insufficient iteration depth are not conflated.

## 实际执行 / What was executed

| 项目 / Item | 数值 / Value |
| --- | ---: |
| PoolFire fit 轨迹 / trajectories | 5 |
| 每条轨迹抽帧 / frames per trajectory | 5 |
| 相机数 / camera counts | 5, 7, 9, 12 |
| 因子化条件 / factorized conditions | 37 |
| 独立 rig / rigs | 740 |
| 物理单元 / physical cells | 3,700 |
| 经典控制行 / classical-control rows | 33,300 |

每个条件都比较 Zero、scaled exact BP、Zero-CGLS K1/K2/K4、Jacobi-equalized BP 与 geometry-PCGLS K1/K2/K4。所有候选使用同一个三维 `32x16x16` 粗网格和同一九视角 straight-ray 密度梯度代理。validation 与 test truth 均未打开。

Each condition compares Zero, scaled exact BP, Zero-CGLS K1/K2/K4, Jacobi-equalized BP, and geometry-PCGLS K1/K2/K4 on the same three-dimensional `32x16x16` coarse grid and the same nine-view straight-ray density-gradient proxy. Validation and test truth remain unopened.

## 主要数值 / Main numbers

### 1. 相机数量仍是主要瓶颈 / Camera count remains a primary bottleneck

clean 条件下，Zero-CGLS K4 的场相对 L2 中位数从 5 相机的 `0.748560` 降至 12 相机的 `0.611195`，相对降低 `18.35%`。

Under clean conditions, the median field relative L2 of Zero-CGLS K4 decreases from `0.748560` with five cameras to `0.611195` with twelve cameras, a relative reduction of `18.35%`.

### 2. 联合扰动主要伤害尾部 / Combined perturbations mainly damage the tails

相对同相机数 clean 条件，combined stress 的 p90 harm 为：场 `+1.84%`、梯度 `+3.74%`、观测 `+6.81%`。这证明噪声与标定误差不是“没有影响”，但它们在当前代理问题上的量级仍小于相机数与迭代深度的影响。

Relative to clean conditions at the same camera count, combined stress produces p90 harm of `+1.84%` in field error, `+3.74%` in gradient error, and `+6.81%` in observation error. Noise and calibration errors therefore matter, but their current proxy-scale effect is smaller than camera-count and iteration-depth effects.

### 3. 两次调用预算仍有稳定缺口 / A stable two-call gap remains

相对 Zero-CGLS K4，Zero-CGLS K2 的场误差比值为 p50 `1.1821`、p90 `1.2862`；geometry-PCGLS K2 为 p50 `1.1768`。在 3,700 个单元中，两者都没有任何单元达到 K4 的场精度。这是继续研究低成本 warm initializer 的直接依据。

Relative to Zero-CGLS K4, the field-error ratio of Zero-CGLS K2 is `1.1821` at p50 and `1.2862` at p90; geometry-PCGLS K2 reaches `1.1768` at p50. Neither matches K4 field accuracy in any of the 3,700 cells. This is the direct reason to continue studying a low-cost warm initializer.

### 4. 更低观测残差不等于更好物理场 / Lower observation residual is not sufficient

geometry-PCGLS K4 相对 CGLS K4 的观测误差 p50 降低 `4.12%`，但场误差 p50 反而增加 `1.04%`，梯度也没有稳定改善。后续模型不能只优化观测残差，必须继续同时通过 field、gradient 与 observation 门。

Geometry-PCGLS K4 lowers median observation error by `4.12%` relative to CGLS K4, yet median field error increases by `1.04%`, with no stable gradient improvement. Future models therefore cannot optimize observation residual alone; they must continue to pass field, gradient, and observation gates jointly.

## 独立复算 / Independent recomputation

第二个验证程序不导入 v128 正式 core 或 runner，重新生成 3,700 个单元与 33,300 行控制，并核对逐数组哈希、逐单元指标、rig、Jacobi 对角项、聚合与 paired effects。数组、指标、rig、Jacobi、聚合和 paired-effect 最大差均为 `0`，正式结果树在验证前后保持不变。

A second validator does not import the v128 formal core or runner. It regenerates all 3,700 cells and 33,300 control rows and checks per-array hashes, cell metrics, rigs, Jacobi diagonals, aggregates, and paired effects. Maximum array, metric, rig, Jacobi, aggregate, and paired-effect differences are all `0`, and the formal result tree remains unchanged before and after validation.

独立程序仍与正式程序共享冻结的底层 physics kernels，因此 `end_to_end_physics_independence_proven=false`。

The independent program still shares frozen low-level physics kernels with the formal program, so `end_to_end_physics_independence_proven=false`.

## 科学判决 / Scientific decision

- `minimal_set_initializer_development_authorized=true`
- `algorithm_breakthrough=false`
- `paper_success=false`
- `external_generalization=false`
- `real_bost=false`
- `validation_truth_opened=false`
- `test_truth_opened=false`

这里授权的是一个最小、可证伪的开发实验，不是授权大模型，也不是 GPU 租赁信号。

This authorizes one minimal, falsifiable development experiment. It does not authorize a large model or GPU rental.

## 下一门 / Next gate

下一步是 SetDual-Warm：把每台相机的 `16x16x2` 观测、18 维位姿/标定编码与 mask 作为无序集合输入；共享每相机编码器，经 mean/max 集合聚合后输出 detector-dual proposal，再通过精确 `A^T` lift、可观测 alpha line search 和未修改 CGLS K1 完成 `2A+2A^T` 候选。它必须采用 trajectory-level leave-one-trajectory-out，比较 Zero-CGLS K2、geometry-PCGLS K2、fit-only dual ridge、no-pose 与 wrong-pose/permutation controls。

The next gate is SetDual-Warm: each camera contributes a `16x16x2` observation, an 18-dimensional pose/calibration code, and a mask as an unordered set element. A shared per-camera encoder and mean/max set aggregation produce a detector-dual proposal, followed by an exact `A^T` lift, observable alpha line search, and one unmodified CGLS refinement for a `2A+2A^T` candidate. Evaluation must use trajectory-level leave-one-trajectory-out and compare Zero-CGLS K2, geometry-PCGLS K2, fit-only dual ridge, no-pose, and wrong-pose/permutation controls.

## 公开材料 / Public artifacts

- [双语脱敏摘要 / Bilingual redacted summary](../docs/poolfire_camera_set_factorized_controls_v128_public_summary.json)
- [结果图 / Result figure](../asset_viewer.html?asset=assets%2Ffigures%2Fpoolfire_camera_set_factorized_controls_v128.png)
- [v127 经典难度图 / v127 classical difficulty map](../document_reader.html?doc=docs%2Fpoolfire_camera_set_classical_screen_v127_result_2026-08-10.md)
- [v126 camera-set 数据底座 / v126 camera-set data foundation](../document_reader.html?doc=docs%2Fcamera_set_virtual_bos_dataset_v126_public_result_2026-08-10.md)
