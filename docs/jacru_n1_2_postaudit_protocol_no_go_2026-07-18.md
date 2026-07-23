# JACRU N1.2 post-audit protocol: NO-GO

- 日期：2026-07-18
- 状态：`N1_2_SESSION_CONFORMAL_DUAL_REFERENCE_NO_GO`
- 证据级别：`E0_OPENED_SYNTHETIC_PROTOCOL_DEVELOPMENT_NO_FRESH`
- 结论边界：这是修正协议后的 pilot，不是完整 confirmation、真实 BOST 或算法优越性证据。

## 1. N1.2 修正了什么

N1.1 把 synthetic flow-off 均值和 covariance 用进重建，但把普通 95% 分位数、同一场条件化
噪声尺度、sensor noise 与 forward mismatch 混在了一起。N1.2 将协议拆成：

1. session-level flow-off packet，而不是把同一 session 的帧当独立实验；
2. 64 个 calibration score 时采用第 62 个次序统计量；
3. global/per-camera upper gate 与 anti-over-regularization lower gate；
4. strongest classical 与 raw learned center 双参考；
5. candidate-specific calibration sanity；
6. sensor covariance 与 voxel-versus-continuous forward mismatch 分账；
7. 运行开始时记录 CLI、Git commit 与传递依赖哈希。

## 2. post-audit pilot 规模

| 项目 | 数量 |
|---|---:|
| session | 3 |
| evaluation case | 5 |
| model seed | 1 |
| candidate | 8 |
| candidate-method decision | 16 |
| metric row | 80 |
| calibration row | 24 |
| model-mismatch row | 5 |
| dense ceiling pass | 0 |
| evaluator-only oracle ceiling pass | 0 |

全部 checksum 通过。完整机器摘要位于
[`summary.json`](../demo_t16_operator/results/jacru_n1_2_session_conformal_dual_reference_scratch_postaudit_pilot1/summary.json)。

## 3. 关键负证据

五个 pilot case 中，voxel forward 与独立 continuous-clean renderer 的相对差为
`15.73%–27.79%`，均值 `22.41%`。这些行明确标记为 `is_sensor_noise=false` 且
`available_to_selector=false`。因此：

- flow-off covariance 不能解释这部分误差；
- 把 mismatch 塞进 detector covariance 会产生错误的置信尺度；
- evaluator 即使读取 exact target mismatch 构造 ceiling，也没有一个 oracle 通过全部门；
- mismatch-aware lower gate 在部分 oracle 行变为不具信息性并 fail closed，而不是自动放宽接管。

## 4. 允许与不允许的表述

**允许：** N1.2 的 session/conformal/双参考/失配分账协议可运行，且在该 opened pilot 中拒绝了
所有候选。

**不允许：** N1.2 已获得 95% 真实覆盖率、已经保护薄界面、已经优于 NeRIF/DeepONet/FNO，或
可以打开 fresh/OOD。

## 5. 对 N1.3 的约束

N1.3 只能测试固定 classical robust data fidelity，不得直接训练 neural selector。必须先完成
`mean correction × whitening × quadratic/Huber × spatial regularization` 的全因子消融，并把
coordinatewise Huber 在 full whitening basis 下的依赖单独标注。

## 6. 重放入口

- runner：[`run_jacru_n1_2_session_conformal_dual_reference.py`](../site_tools/run_jacru_n1_2_session_conformal_dual_reference.py)
- config：[`jacru_n1_2_session_conformal_dual_reference_development_v1.json`](../demo_t16_operator/configs/jacru_n1_2_session_conformal_dual_reference_development_v1.json)
- result checksums：[`checksums.sha256`](../demo_t16_operator/results/jacru_n1_2_session_conformal_dual_reference_scratch_postaudit_pilot1/checksums.sha256)
