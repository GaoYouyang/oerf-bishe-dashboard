# v3d FNO Validation-Plateau 审计简报

更新时间：2026-07-11

## 一句话判决

**历史判决：K=6 ridge-FNO 到 96 epochs 仍未达到预注册 validation plateau，因此当时禁止开始新结构。后续 240-epoch optimizer audit 已锁定 FNO 固定 epoch 基线；跨架构算力仍待补，最新状态见文末。**

## 为什么要先做这一步

v3c 的 frozen per-view adapter 相对 24-epoch base FNO 有 `+1.425%`，但相对同 checkpoint 继续训练 12 epochs 的 FNO 为 `-4.965%`。这说明架构比较被 baseline 训练时长混淆。v3d 因此先锁定：

- 相同 K=6 inputs、ridge anchor、loss 和数据；
- 24-epoch base schedule 与 v3c 相同；
- 此后从上一个 validation-selected checkpoint 开始新的 12-epoch block；每个 block 都重新初始化 AdamW 与 cosine scheduler，不携带 optimizer state；
- 每个 block 若 validation 没有至少 `1e-5` 绝对改善，则保留上一个 checkpoint；
- plateau 判决只读取 validation，不读取 dev2 test 或 `Q_audit`。
- validation 对每个三维场等权；最后 4-sample batch 按样本数加权，不与 12-sample batch 等权。

## 预注册 plateau 规则

连续末端两个 12-epoch blocks 必须同时满足：

1. 三种子平均 validation 相对改善不超过 `0.5%`；
2. 任一种子的改善不超过 `1.0%`。

若后续 block 重新出现明显改善，之前的临时平台不算 plateau。

## 结果

| 累计 epochs | mean validation L2 | 相对上个 block 改善 | 最大 seed 改善 | plateau block |
|---:|---:|---:|---:|---|
| 24 | 0.16646 | 0.00% | 0.00% | 否 |
| 36 | 0.14862 | 10.68% | 11.81% | 否 |
| 48 | 0.13703 | 7.76% | 9.81% | 否 |
| 60 | 0.12964 | 5.39% | 6.16% | 否 |
| 72 | 0.12444 | 4.01% | 4.09% | 否 |
| 84 | 0.12085 | 2.88% | 3.14% | 否 |
| 96 | 0.11746 | 2.81% | 2.96% | 否 |

24→96 epochs 的 mean validation L2 降低 `29.44%`。最后一个 block 仍明显高于 `0.5% / 1.0%` plateau 阈值，且 21 个 seed-checkpoints 中没有一个需要回退到上一 checkpoint。

这里的 `96 epochs` 是六个独立重启 optimizer/scheduler 的 12-epoch continuation blocks 与最初 24 epochs 的累计标签，不是一次 optimizer state 连续不断的 96-epoch run。这个设计复现了 v3c 的追加训练日程，但 plateau 结论只适用于这套已锁定的 block schedule。

## dev2 后置诊断

这些复用 diagnostics 在本轮 validation-only plateau 判决之后计算，不能用于选择停止点；它们在早期项目阶段已经查看过，不是新的 untouched audit：

- domain-equal field L2：`0.35133 → 0.31253`；相对 epoch 24 改善 `+16.534%`，field-cluster 95% CI `[+15.607%, +17.446%]`。
- clean `Q_audit` L2：`0.41972 → 0.34210`，改善约 `18.49%`。
- 三个 model seeds 在 epoch 96 相对 epoch 24 均为正：`+18.26% / +15.49% / +15.72%`。
- IID、noise、family、joint 四个 dev2 域的均值均未反向。

这些仍只是 `128` 个已查看 linear synthetic dev2 fields，不能声称真实 BOST superiority。

## 对毕业设计路线的影响

1. **暂停新结构。** 当前不能把 24/36-epoch FNO 作为 F-Adapter、LoRA 或自有模型的最终对手。
2. **延长 baseline audit。** 下一轮保持阈值不变，将 validation-only continuation 延伸到更高上限；不能因为想尽快 plateau 而放宽规则。
3. **geometry manifest 可并行。** 与何远哲先确认真实 geometry 是否变化；若完全固定，acquisition-conditioned 路线直接 No-Go。
4. **blind final 继续关闭。** plateau、F-Adapter control、geometry 消融和独立 forward 均完成后才有资格申请。

## 本科生应该从图里学会什么

- epoch 相同不等于训练公平；需要看 validation trajectory。
- 参数更少不等于模型更好，也不等于训练更快。
- “显著超过旧 baseline”可能只是旧 baseline 没训够。
- test 曲线可以解释结果，但不能反过来选择 epoch。
- plateau 是当前 optimizer/schedule 下的操作性判据，不是证明找到了全局最优解。

## 给何远哲的两个判断题

1. 是否认可下一轮继续延长同一 FNO continuation protocol，在 plateau 前暂缓所有新 adapter？
2. 真实组内 geometry 是否跨 case 变化，足以支持 `v3d_geometry_data_manifest.md` 中的 Go 条件？

## 可复跑入口

- 配置：`demo_t16_operator/configs/v3d_fno_saturation_audit.json`
- 训练与分析：`demo_t16_operator/run_v3d_fno_saturation_audit.py`
- 结果校验：`python -m demo_t16_operator.validate_v3d_fno_saturation_results`
- 机器报告：`demo_t16_operator/results/v3d_fno_saturation_audit/v3d_fno_saturation_report.json`

## 后续更新

240-epoch 三优化协议审计已接续本结果。它锁定 carry-continuation-Adam/restart-cosine 为 FNO 固定 epoch 冠军，同时保留已 plateau 的 long-cosine control；v3e 已补 FLOPs/内存/time-to-target，v3f 已补当前 DeepONet/FNO development frontier。完整边界见 `v3d_fno_optimizer_protocol_brief.md` 与 `v3f_deeponet_fno_frontier_brief.md`。
