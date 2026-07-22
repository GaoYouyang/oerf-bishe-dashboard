# rig/session 层级校准教学实验：冻结协议

> 冻结日期：2026-07-22  
> 证据等级：`EDUCATIONAL_SYNTHETIC_CLUSTER_CALIBRATION_MECHANISM_ONLY`  
> 目标：解释 cluster-size bias、可观测异方差和 distribution shift；不模拟 BOST forward，不激活 fresh BOST audit。

## 1. 预先冻结的问题

同一 rig/session 内的 frame 共享装置和难度。如果 easy rig 采到更多 frame，把所有 frame 当作等权 calibration observation，是否会让 hard rig 的覆盖被平均值掩盖？一个只用部署可见特征的低容量尺度模型，什么时候能改善这一问题，什么时候必须失败？

本实验只验证这两个统计机制，不回答三维场、曲光线、NeRIF、TDBOST 或真实实验泛化。

## 2. 独立单位与数据切分

- `fit`、`calibration`、`evaluation` 各自产生互不重叠的 synthetic rig；固定为 `120 / 120 / 400` 个 rig。
- 每个 rig 内有 `30--240` 个 frame；难 rig 的 frame 更少，故 frame 数本身构成 cluster-size bias。
- 每帧 score 为 `scale_g * chi2(df=5)`。
- 目标 coverage 为 95%；所有随机数由固定 namespace、scenario、split 和 rig id 派生。
- `fit` 只拟合尺度模型，`calibration` 只选最终阈值，`evaluation` 只在全部规则冻结后打开。

## 3. 三个预先冻结的情景

| 情景 | fit / calibration 的真尺度 | evaluation 的真尺度 | 作用 |
|---|---|---|---|
| `observable_scale` | `exp(0.9 z + 0.18 u)` | 相同 | 正例：`z` 能解释主要异方差 |
| `hidden_scale` | `exp(0.9 u)` | 相同 | 负对照：`z` 与难度无关 |
| `sign_flip_shift` | `exp(0.9 z + 0.18 u)` | `exp(-0.9 z + 0.18 u)` | 反例：support 相同但条件关系反转 |

其中 `z,u ~ N(0,1)`；`z` 是部署可见特征，`u` 永远隐藏。frame 数由真 `log(scale)` 单调递减映射到 `[30,240]`，不额外暴露给尺度模型。

## 4. 冻结方法

1. `frame_pooled`：所有 calibration frame 等权，用 finite-sample order statistic 取阈值。
2. `equal_rig`：每个 rig 的经验 CDF 权重相同，再取 95% 加权分位数。它是层级描述性基线，不宣称一般 finite-sample 保证。
3. `worst_rig`：每个 calibration rig 单独取 finite-sample 95% 阈值，再取最大值；预期保守。
4. `log_ridge_normalized`：在 fit rigs 上回归 `log(median(score))-log(median(chi2_5))` 对 `z`，截距不惩罚、斜率 ridge=`0.1`；在 calibration rigs 上对 `score / predicted_scale(z)` 做 equal-rig 分位数。
5. `oracle_normalized`：用不可部署的真 scale 正规化；只作参考上界。

禁止：神经网络、在 evaluation 上重拟合、按结果改特征、改情景系数或挑 seed。

## 5. 冻结评价

对 400 个新 rig 分别计算覆盖率，再报告：

- `rig_mean_coverage`：400 个 rig 等权平均；
- `observation_weighted_coverage`：所有 frame 等权平均，用于暴露两种口径差异；
- `rig_p10_coverage`：逐 rig coverage 的第 10 百分位；
- `hard_quartile_coverage`：真 scale 最大四分之一 rig 的等权平均；
- `fraction_rigs_ge_90`：逐 rig coverage 至少 90% 的比例；
- `radius_proxy_median/p90`：`sqrt(threshold * predicted_scale)`，只比较相同 score 生成机制下的效率。

同时记录 fit 斜率、fit `R^2`、阈值、rig/frame 数和 SHA-256。

## 6. 预注册教学判据

这不是算法晋级门，而是检查 toy 是否真的展示了预期机制：

1. `observable_scale`：log-ridge 的 hard-quartile coverage 至少比 frame-pooled 高 `0.05`，且不低于 `0.90`；其 median radius 不高于 equal-rig。
2. `hidden_scale`：log-ridge 与 equal-rig 的 hard-quartile coverage 差的绝对值不超过 `0.03`，说明没有凭空创造信息。
3. `sign_flip_shift`：log-ridge 的 hard-quartile coverage 低于 `0.90`，必须显式暴露关系反转失败。
4. oracle 在三种情景的 rig-mean coverage 均落在 `[0.93,0.97]`；否则先检查实现或 Monte Carlo 误差。

若任一条不满足，保留原始结果并写明 `TOY_MECHANISM_CHECK_FAILED`；不得换 seed 后重跑。

## 7. 允许和禁止的结论

允许：

- 展示按 frame pooling 与按 rig 评价的差异；
- 展示可观测尺度模型的适用条件和 OOD 失败；
- 用于理解为什么下一次真实数据必须按 acquisition/session split。

禁止：

- “提出了新 conformal 算法”；
- “实现了 BOST 校准/三维重建”；
- “证明跨 rig 泛化”；
- “达到论文结果”或“出现突破”。

## 8. 与真实主线的唯一接口

toy 完成后只能产生一张给师兄看的问题表：真实独立单位是什么、哪些 geometry/noise 摘要在部署时可见、是否有 flow-off repeat，以及 support 外 rig 应如何拒答。只有这些合同得到确认，才允许把同样的 split 逻辑接到真实 callable；[fresh BOST 审计休眠合同](temporal_qcal_fresh_audit_dormant_contract_2026-07-22.md)仍保持冻结。
