# T16 v3c K=6 dev2 负结果与 v3d 转向简报

更新日期：2026-07-11

状态：`V3C_K6_DEV2_ADAPTER_GATE_FAIL`

## 一句话判断

当前 **frozen FNO + per-view 3D zero-init adapter** 应停止作为主候选。它相对未追加训练的 base FNO 有稳定小改善，但在同一基础 checkpoint、同一追加 12 epochs 下，明显输给直接 continued FNO，并且更慢。

## 协议是什么

- 任务：K=6 ridge-initialized 3D inverse operator。
- 数据：328 个全新 dev2 fields；160 train、40 validation、4×32 test fields。
- 模型种子：3。
- 基础阶段：每个种子先训练 ridge-FNO 24 epochs。
- 追加阶段：从同一 validation-selected checkpoint 分叉。
- 对照 A：全量 continued FNO，追加 12 epochs。
- 候选 B：冻结 FNO，只训练 4,988 参数 zero-init set adapter，追加 12 epochs。
- A/B 使用相同 batch order、loss、learning rate、optimizer 和 validation checkpoint rule。
- 这是追加 epochs 匹配，不是 FLOPs 匹配。
- `Q_audit` 不进入输入、训练、early stop 或方法选择。

## 核心结果

| 比较 | domain-equal mean field superiority | 95% field-cluster CI | p10 | harm >1% | clean Q_audit superiority |
| --- | ---: | ---: | ---: | ---: | ---: |
| adapter vs continued FNO | -4.965% | [-5.569%, -4.397%] | -12.143% | 74.219% | -5.246% |
| adapter vs base FNO | +1.425% | [1.281%, 1.570%] | +0.351% | 0% | +1.111% |
| continued FNO vs base FNO | +5.846% | [5.342%, 6.353%] | +1.497% | 0% | +5.268% |

adapter 相对 continued FNO 的三个种子均值为 `-5.389% / -3.669% / -5.898%`，没有种子翻转。

## 逐域判断

adapter 相对 continued FNO：

| dev2 域 | mean field superiority |
| --- | ---: |
| IID | -10.127% |
| noise OOD | -7.629% |
| thin-front family OOD | -1.063% |
| joint OOD | -1.041% |

四个域都为负。因此不能把失败解释为“只是某一个 OOD 域拖后腿”。

## 计算成本

| 追加方法 | 平均训练时间 | 平均推理时间/样本 | 可训练参数 |
| --- | ---: | ---: | ---: |
| continued FNO | 2.391 s | 0.223 ms | 44,203 |
| zero-init adapter | 17.164 s | 1.440 ms | 4,988 |

adapter 训练时间约为 `7.18×`，推理时间约为 `6.46×`。原因是它为每个相机分别运行 3D encoder；参数少不等于计算便宜。

## 这个负结果证明了什么

1. 24 epochs 的 FNO 尚未饱和。continued FNO 在三个种子和四个域全部继续改善。
2. “同 epochs”不足以支撑架构 superiority。后续必须报收敛曲线、wall time、推理时间和饱和检查。
3. ray-set 信息不是完全无效。adapter 相对 base FNO 四域全正、p10 为正且无 >1% harm，但这个收益不足以抵消冻结底座的机会成本。
4. 当前瓶颈更像是优化自由度和计算路径，而不是简单缺少一个更大 adapter。

## v3d 建议（最近邻查重后更新）

不能把 low-rank spectral modulation 本身当创新。NeurIPS 2025 F-Adapter 已覆盖 Fourier operator 的 frequency-adaptive PEFT，R2-FFNO/MG-TFNO 已覆盖低秩或张量化谱核。未来只保留 **BOST acquisition-geometry-conditioned F-Adapter** 假设，不再对每个相机运行完整 3D encoder；当前仍被 baseline plateau 与 geometry manifest 双闸门阻断。完整查重见 `v3d_prior_art_and_novelty_gate.md`。

### 必须同时运行的对照

1. saturated continued FNO：先用 validation 曲线锁定足够训练长度。
2. LoRA/static low-rank spectral update：不使用 geometry，参数与自有方法匹配。
3. vanilla adapter 与 F-Adapter-style frequency allocation：排除 adapter 形式或频率容量分配本身的收益。
4. acquisition-conditioned F-Adapter：只用 angle/ray coverage/mask/noise/calibration 的相机级统计生成频带系数。
5. last-block fine-tuning：解冻最后谱块，可训练参数与 adapter 同量级。
6. geometry-shuffled、constant-geometry、same-rank static controls：验证模型真的使用正确采集几何。

只有正确 geometry 版本在同参数、同 wall-time 和同收敛规则下超过 F-Adapter/LoRA/last-block，且 shuffle/constant/static 都显著下降，才能把收益归因于 BOST acquisition geometry。

## 推荐停止项

- 停止当前 frozen per-view 3D adapter 主线。
- 不用当前 dev2 负结果开启 blind final。
- 不通过增加 adapter width 来直接追平 continued FNO。
- 不再用单一固定 epochs 结果宣称新结构胜出。

## 请何远哲师兄判断

1. 组内任务更重视单次推理速度，还是最终三维场精度？
2. 真实 BOST geometry 是否在样本间变化？如果完全固定，geometry-conditioned operator 的必要性会明显降低。
3. 是否同意把 v3d 收窄为“F-Adapter vs acquisition-geometry-conditioned F-Adapter + geometry controls”，而不再扩展一个独立大网络？
4. 能否提供最小真实 geometry/forward 接口，用来判断现在的 canonical-angle toy 是否值得继续？

## 结论边界

本结果只针对 K=6、`8×16×16` linear synthetic dev2、当前 optimizer/loss 和三个模型种子。它足以停止当前 adapter 开发路径，但不能否定所有 geometry-conditioned operator、NeRIF warm-start 或真实 BOST 中的学习型初值。

## 后续状态更新：96 epochs 仍未 plateau

后续 `v3d_fno_validation_plateau_brief.md` 已把同一 K=6 FNO 从 24 延长到 96 epochs。末个 12-epoch block 的 mean/max-seed validation 改善仍为 `2.67%/2.83%`，未满足 `0.5%/1.0%` plateau 阈值；dev2 field 相对 epoch 24 的 `+16.54%` 只作后置诊断。

因此本 brief 中的 F-Adapter/geometry 对照现在是 **plateau 与 geometry manifest 双闸门之后的未来实验**，不是立即执行任务。

## 后续状态更新：240-epoch FNO 固定 epoch 基线

`v3d_fno_optimizer_protocol_brief.md` 已比较 continuation blocks 间 restart/carry Adam 与 block/long cosine。carry-continuation-Adam/restart-cosine 是 epoch-240 validation 冠军但仍未 plateau；long-cosine 已 plateau 却弱 1.70%。geometry 为 Go 后可写功能 pilot；跨架构 compute accounting、blind final 与“击败饱和 FNO”表述仍关闭。
