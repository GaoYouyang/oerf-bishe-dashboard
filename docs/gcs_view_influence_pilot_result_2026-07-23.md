# Exact leave-one-view 28 路径 pilot：工程门通过，机制仍未知

## 0. 判决

正式状态：`VIEW_INFLUENCE_PILOT_INVARIANTS_PASS_NOT_A_MECHANISM_RESULT`。

本轮只验证逐视角 exact retraining 能否在当前 Mac/MPS、冻结 seed 和相机块合同下稳定执行。4 条 full replay、24 条 leave-one-train-view retraining 和 4 条 influence summary 全部生成；full replay 与冻结源的最大 observable absolute difference 为 `0`。

该结果授权实现并运行完整 912 路径 post-open mechanism panel。它不授权 mechanism signal、新 calibration、fresh audit、Stage B、安全 gate、算子学习模型或论文主张。

**突破监测：没有突破。** 这是工程门通过，不是算法或物理结果。

## 1. 为什么第一次启动被挡住

第一次运行在训练前发现 correction grid 的 `half_width=1.0` 与已有边界窗合同冲突。处理方式不是放宽底层函数，而是：

1. 把 correction 评估网格改为 `half_width=0.95`；
2. 在 protocol validator 中新增严格 `(0,1)` 检查；
3. 重新提交干净 commit 后从头运行。

第一次没有进入任何训练，也没有把失败产物当结果。这个事件说明 pilot 确实能拦截协议实现错误。

## 2. 第二次执行账本

| 项目 | 结果 |
|---|---:|
| source commit | `a7b51ae4184a17d81a556385f0d8914fa960ef65` |
| Apple MPS wall time | 241.527 s |
| full fit | 4/4 |
| exact leave-one-view fit | 24/24 |
| influence row | 4/4 |
| finite numeric outputs | 全部 |
| observable full corrections | 4/4 |
| source replay maximum difference | 0 |
| tolerance | `1e-6` |

pilot 只使用两个已经打开的 phantom seed 3301/3401、两档噪声、nominal six-view 和 `base101_residual307`。没有读取新 phantom，也没有打开 oblique/shock Stage B。

## 3. 不依赖真值的结构观察

24 次去视角重训中，18 次 residual admitted、6 次精确回退：

| 被删除的训练视角 | admitted / 4 |
|---:|---:|
| 0° | 3/4 |
| 30° | 4/4 |
| 60° | 4/4 |
| 90° | 2/4 |
| 120° | 2/4 |
| 150° | 3/4 |

四个 path 的 LOO admission fraction 分别为 `4/6、5/6、5/6、4/6`。这只能说明删视角会改变 development admission；样本是特意选择的 engineering pilot，不能声称 90°/120° 在物理上更重要。

因为零 correction 会把 cosine minimum 推到 0，完整协议现在显式预注册 `loo_admitted_fraction`，不再把回退信息隐含在 norm/cosine 中。这个修改发生在任何 truth-label join 和完整面板之前。

## 4. 为什么还不能看“好坏分离”

这四组 pilot 对应的冻结真值结果已经在上一轮打开，但 pilot 的角色合同禁止把它们连接回来做 feature selection。否则会出现典型的双重使用：先看四点的答案，再决定保留哪些逐视角统计，最后在同四点上宣布有效。

完整机制面板才会在所有 observable feature CSV 写完之后连接 72 个既有 condition label；即便如此，它仍是 post-open engineering，不是 final audit。

## 5. 完整面板的增量门

完整面板必须同时运行三套固定 feature set：

1. `source_observable_control`：上一轮的总场能量、holdout 和 roughness；
2. `view_influence_only`：14 个 exact LOO permutation-invariant features；
3. `source_plus_view_influence`：二者拼接。

三者使用相同的 ridge lambda、phantom leave-one-group-out fold 和 label。组合模型除 grouped AUC 至少 0.75、每 family 至少 0.65、cluster bootstrap 下界至少 0.50 外，还必须比 source-only grouped AUC 高至少 0.05。

如果没有这个增量，说明新计算只是重复旧能量信号；exact LOO 支线应关闭。

## 6. 产物与校验

- `learning_labs/results/gcs_view_influence_pilot_v0_run1/summary.json`
- `learning_labs/results/gcs_view_influence_pilot_v0_run1/full_replay_rows.csv`
- `learning_labs/results/gcs_view_influence_pilot_v0_run1/leave_one_view_rows.csv`
- `learning_labs/results/gcs_view_influence_pilot_v0_run1/influence_rows.csv`
- `learning_labs/results/gcs_view_influence_pilot_v0_run1/checksums.sha256`

## 7. 禁止主张

不能声称 view influence 已有 signal、某个相机角更重要、exact LOO 优于 energy gate、JVP/VJP 近似有效、已提出新算子、真实 BOST/PIV-BOST 验证、跨 rig 泛化、论文成功或突破。
