# v278.1：旧三维场与九相机标定已用于扰动验证，参考解在位姿误差下失效

## 为什么做

现有数据并没有被搁置。v278.1 直接使用此前提供的九个可执行三维场与十三套九相机标定，按已确认的约定从三维密度场生成二维 BOS 代理观测。因为没有与这些场逐帧配对的真实二维位移，本轮严格称为**受控虚拟 BOS**，不称真实 BOST。

结果前固定四个归一化时间、九相机硬件、clean / 观测噪声 / 位姿误差 / 内参误差 / 组合误差五种条件，并比较一个 H1 warm + 未修改 CGLS K1 候选、一个更便宜的 H1 direct control、同调用 Zero-CGLS K2 与 Zero-CGLS K16 reference。总覆盖为 `9 × 13 × 4 × 5 = 2,340` 个实验单元、`9,360` 条评分行。

## 独立复算

正式实现与完全独立第二实现的 `21/21` 项检查全部通过。逐行、分层汇总与最终审裁的最大数值差分别为 `1.25e-11 / 1.70e-12 / 4.99e-12`；像素锁定、扰动重建、相机乱序等变、算子与调用账均闭合。因此本轮不是程序没跑完，也不是实现不一致。

## 结果

K16 reference 在 clean、观测噪声和内参误差下通过全部 `12/12` 个条件-时间分层，但在四个位姿误差分层与四个组合误差分层全部失败，总计仅 `12/20`。其最坏归一化门负担在位姿条件约为 `1.72–1.76`，组合条件约为 `1.82–1.84`；通过线为 `1.0`。

主候选和更便宜 control 对 reference 的 `20/20` matched 仅能作为诊断。冻结的 reference-first 顺序要求：裁判答案自身任一分层不合格，就不能继续宣布候选成功、失败或被便宜 control 支配。因此权威判决是：

`INCONCLUSIVE_REFERENCE_INADEQUATE_FIXED9_ROBUSTNESS_V278_1`

## 讲人话

旧数据确实能继续推进，而且已经把流程推进到“标定误差是否会破坏重建”的层面。现在暴露的不是数据完全不可用，而是轻微相机姿态误差会让当前 K16 裁判本身明显偏离真场。没有合格裁判时，候选看起来多好都不能算正式成果。

下一步应先建立或独立论证一个对位姿扰动可靠的 reference，再重新判候选；不能针对已经打开的八个失败分层回头调候选。当前仍为 `algorithm_breakthrough=false`、`real_bost=false`、`resource_speedup=false`，不授权神经训练或租 GPU。

---

# v278.1: Existing 3D fields and nine-camera calibrations reach a perturbation gate, but pose error invalidates the reference

## Why this was run

The existing data were not set aside. v278.1 directly uses the previously supplied nine executable 3D fields and thirteen nine-camera calibration sets, generating 2D BOS proxy observations from the 3D density channel under the agreed convention. Because no measured 2D displacement is paired frame by frame with these fields, this remains a **controlled virtual-BOS** experiment rather than real BOST.

Four normalized times, the fixed nine-camera hardware, and five conditions were frozen before scoring: clean, observation noise, pose error, intrinsic error, and their combination. The four arms are an H1 warm start followed by unchanged CGLS K1, a cheaper direct-H1 control, an equal-call Zero-CGLS K2 control, and a Zero-CGLS K16 reference. Coverage is `2,340` cells and `9,360` scored rows.

## Independent recomputation

All `21/21` independent checks pass. Maximum row, stratum-summary, and adjudication differences are `1.25e-11 / 1.70e-12 / 4.99e-12`; pixel locking, perturbation reconstruction, camera-permutation equivariance, operators, and call ledgers all close. This is therefore neither an unfinished run nor an implementation disagreement.

## Result and boundary

The K16 reference passes all `12/12` clean, observation-noise, and intrinsic-error strata, but fails all four pose-error and all four combined-error strata, for `12/20` overall. Its worst normalized gate burden is approximately `1.72–1.76` under pose error and `1.82–1.84` under combined error, against a passing line of `1.0`.

The primary and cheaper control each match the reference diagnostically in `20/20` strata, but reference-first adjudication forbids promoting those diagnostics to candidate success, failure, or control dominance. The authoritative decision is `INCONCLUSIVE_REFERENCE_INADEQUATE_FIXED9_ROBUSTNESS_V278_1`.

This confirms that the old data support a meaningful calibration-robustness experiment, while exposing the current reference as unreliable under pose perturbations. A perturbation-robust reference must be established before judging the warm-start candidate. This is not real BOST, a resource result, neural evidence, or an algorithmic breakthrough.
