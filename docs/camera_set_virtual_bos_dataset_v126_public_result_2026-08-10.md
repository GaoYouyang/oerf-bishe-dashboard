# v126：变机位虚拟 BOS 数据底座

## 一句话结论

已经把何远哲师兄提出的四项要求落成一条可运行、可独立复算的数据链：**相机可乱序、可增删、数量可变，并同时加入观测噪声与位姿/标定误差**。正式批次覆盖 64 个三维 Gaussian-field 样本，独立程序逐数组复算的最大差为 0。

这是一项数据与物理接口工程增量，不是算法性能结果。当前仍是 `algorithm_breakthrough=false`、`paper_success=false`、`real_bost=false`。

## 实际做了什么

- 三维场：`32 × 16 × 16`，每个样本由三个随机 Gaussian component 组成。
- 相机母库：12 个候选方位，正式样本分别选择 `5 / 7 / 9 / 12` 个相机。
- 观测：每个相机输出 `16 × 16 × 2` 的有限光源会聚 straight-ray 密度梯度代理。
- 噪声档：clean、40 dB、30 dB、20 dB。
- 真实装置变化：逐相机方位、俯仰、滚转、相机中心、目标点、焦距和主点微扰。
- 报告标定误差：在真实装置上另加独立逐相机误差，严格区分“真实相机”与“模型收到的标定”。
- 集合操作：随机选择相机子集，并用稳定 camera ID 绑定全部随机量；相机列表倒序不能改变物理场、单相机观测、噪声或位姿误差，只能改变相机轴顺序。

## 正式结果

| 检查 | 结果 |
| --- | ---: |
| 正式样本 | 64 |
| 相机数量 | 5 / 7 / 9 / 12 |
| 扰动档 | 4 |
| 最大 forward/adjoint 相对误差 | `2.758 × 10^-15` |
| 相机乱序恢复最大绝对误差 | `0` |
| clean 观测噪声 | `0` |
| clean 标定位姿误差 | `0` |
| 噪声强度相对目标最坏偏差 | `2.128%` |
| 独立复算数组最大差 | `0` |
| 独立复算汇总指标最大差 | `0` |

独立验证程序不导入 v126 数据生成器；它重新实现随机键、相机子集、Gaussian 场、真实装置变化、报告标定误差与观测噪声，只共享冻结的底层 forward/adjoint 和 pose-token 物理核。

## 为什么这一步必要

之前的九视角实验虽然向网络输入了相机 pose token，但正式训练仍固定使用九台相机，没有验证相机交换、删减、增加和标定漂移。那只能称为“已知固定离散位姿条件输入”，不能称为对机位变化的泛化。

v126 先把输入数据结构改成真正的 camera set：每个相机是独立元素，camera ID 绑定该视角的观测与标定，模型后续必须对相机顺序不敏感，并允许集合大小变化。这样未来拿到实验数据时，可把同一相机的位移图、内外参和噪声描述直接装入同一接口。

## 尚未证明什么

- 尚未把这套扰动合同接到 PoolFire CFD trajectory。
- 尚未比较 clean / noisy / pose-error / camera-dropout 下的 Zero、BP、CGLS、PCGLS 与历史 warm initializer。
- 尚未训练 permutation-invariant、variable-cardinality 的神经算子。
- 尚未证明跨机位、跨相机数量或跨噪声泛化。
- 尚未测量 matched-accuracy、exact `A/A^T`、wall time 或 peak RSS。
- 尚未接入真实 BOST 位移图、相机标定与重复测量噪声。

## 下一道有效门

把完全相同的 camera-set 采样、观测噪声和标定误差合同接到已打开的 PoolFire fit trajectories。先在不训练网络的情况下比较经典重建基线，确认噪声与变机位确实构成可重复、可量化的性能缺口；只有存在稳定 headroom，才训练最小的 permutation-invariant camera-set warm initializer。

---

# v126: Variable-Camera Virtual BOS Data Foundation

## One-sentence conclusion

The four requirements proposed by the senior collaborator are now implemented as a runnable and independently reproducible data pipeline: **cameras can be permuted, added or removed, the cardinality can vary, and both observation noise and pose/calibration error are injected**. The formal batch contains 64 three-dimensional Gaussian-field samples, with zero maximum array difference under independent recomputation.

This is a data and physics-interface increment, not an algorithm-performance result. The current boundaries remain `algorithm_breakthrough=false`, `paper_success=false`, and `real_bost=false`.

## What was executed

- Volume: `32 × 16 × 16`, with three random Gaussian components per sample.
- Camera bank: 12 candidate azimuths; formal samples use `5 / 7 / 9 / 12` cameras.
- Observation: a `16 × 16 × 2` finite-source convergent straight-ray density-gradient proxy per camera.
- Noise levels: clean, 40 dB, 30 dB, and 20 dB.
- True-rig variation: per-camera azimuth, elevation, roll, camera centre, target point, focal length, and principal-point perturbations.
- Reported-calibration error: a second independent per-camera perturbation on top of the true rig, keeping physical and reported cameras distinct.
- Set operations: camera subsets are sampled and all randomness is keyed by stable camera IDs. Reversing the camera list may only reorder the camera axis; it cannot change the field, per-camera observation, noise, or pose error.

## Formal result

| Check | Result |
| --- | ---: |
| Formal samples | 64 |
| Camera counts | 5 / 7 / 9 / 12 |
| Perturbation profiles | 4 |
| Maximum forward/adjoint relative error | `2.758 × 10^-15` |
| Maximum camera-order restoration error | `0` |
| Clean observation noise | `0` |
| Clean calibration-pose error | `0` |
| Worst relative deviation from target noise level | `2.128%` |
| Independent maximum array difference | `0` |
| Independent maximum summary-metric difference | `0` |

The independent validator does not import the v126 generator. It reimplements random keys, camera subsets, Gaussian fields, true-rig variation, reported-calibration error, and observation noise, sharing only the frozen low-level forward/adjoint and pose-token physics kernels.

## Why this matters

Earlier nine-view experiments supplied camera pose tokens but still trained and evaluated with exactly nine cameras. They therefore supported known fixed discrete poses, not generalization to camera changes.

v126 changes the data representation to an actual camera set. Every camera is an independent element whose observation and calibration are bound by camera ID. A future model must be permutation-invariant and accept variable cardinality. The same interface can later ingest each experimental camera's displacement image, intrinsics, extrinsics, and noise description.

## What remains unproven

- The contract has not yet been attached to PoolFire CFD trajectories.
- Zero, BP, CGLS, PCGLS, and historical warm initializers have not yet been compared under clean, noisy, pose-error, and camera-dropout conditions.
- No permutation-invariant variable-cardinality neural operator has been trained.
- Cross-pose, cross-cardinality, or cross-noise generalization has not been demonstrated.
- Matched accuracy, exact `A/A^T`, wall time, and peak RSS have not been measured.
- Real BOST displacement images, calibration, and repeated-measurement noise are not yet connected.

## Next valid gate

Apply the identical camera-set sampling, observation-noise, and calibration-error contract to the opened PoolFire fit trajectories. First compare classical reconstruction controls without neural training and establish a repeatable performance gap. A minimal permutation-invariant camera-set warm initializer is authorized only if stable headroom remains.
