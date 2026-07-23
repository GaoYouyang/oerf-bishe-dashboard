# T16 v3b-v3g 自有算法开发审核 brief

更新日期：2026-07-12

## 一句话结论

早期 ray-set augmentation 已在更强 continued FNO 下失败并停止。当前已经锁定 240-epoch FNO、跨架构成本、DeepONet/FNO matched frontier，并完成 `8 架构 × 3 学习率 × 3 种子` 的有界 DeepONet 容量审计：短程 rank-64 冠军到 240 epochs 为 `0.176094`，仍略逊 rank-48 reference `0.175725`，FNO 为 `0.094139`。因此冻结 rank-48 DeepONet control，不再靠加 rank 追结果；唯一保留的自研假设是 **FNO 空间主干 + BOST acquisition-geometry conditioner**，且必须通过 wider-FNO、static、mask-only 与 shuffled-geometry controls。

## 用户真正的研究目标

最终目标不是复现或微调 DeepONet/FNO，而是：

1. 提出一个 BOST 专属的自有模型；
2. 在相同相机、输入信息、训练预算和评价协议下击败 U-Net、FNO、DeepONet；
3. 在确认性阶段再加入 CGLS/TV/RBF、GRU-BOST、NeRIF/NeDF/NRIP；
4. 把 operator-only 优势升级为 `operator -> NeRIF physics refinement` 的端到端时间/质量优势；
5. 用全新 blind fields、独立 forward 和真实 repeated acquisitions 支撑论文结论。

## v3b 模型做了什么

共同输入为：

- validation-locked ridge 粗场；
- 每台重建相机独立 ray backprojection；
- 每台相机 mask 与显式 `sin(theta) / cos(theta)`；
- support、view fraction 和 `(z,y,x)` 坐标。

所有 U-Net、FNO、DeepONet 和自有模型得到相同信息。自有候选增加：

1. shared per-view 3D encoder；
2. ridge-conditioned voxel-wise masked attention；
3. set aggregate、spread 和 entropy；
4. 完整共同输入 + set statistics 的 residual FNO trunk；
5. support-limited field correction。

attention 子模块在相机通道和角度一起置换时保持不变；完整主干仍使用 canonical camera slots，因此当前只能称“显式几何 + missing-camera aware”，不能声称任意输入顺序下整个网络严格 permutation invariant。

## 公平性合同

- K=4/6/8 每个预算独立训练。
- 60 degree `Q_audit` 不进入输入、训练、early stop 或 baseline 选择。
- 每个模型同一 dataset、batch、24 epochs、optimizer、loss 和模型种子集合。
- 参数量：U-Net `94,193`、FNO `44,203`、DeepONet `49,313`、自有候选 `45,973`。
- FNO/DeepONet/U-Net 也得到 per-view ray、mask 和 angle，不允许自有模型多拿 geometry 信息。
- 神经对手只按 validation mean 锁定；三个预算均锁为 FNO。
- 96 个独立 test fields，3 个模型种子先按 field 折叠，再做四域等权、20,000 次分层 bootstrap。
- 当前 `Q_audit` 是 synthetic clean held-out reprojection，只能作 retrospective oracle-style audit，不能冒充真实 noisy camera 验收。

## 三种子主结果

| K | 锁定神经对手 | 自有方法 mean field superiority | 95% field-cluster CI | p10 | harm >1% | clean Q_audit superiority | 开发门槛 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | FNO | +1.77% | [1.14%, 2.43%] | -0.74% | 7.29% | +0.13% | 未过：CI 下界不足 2% |
| 6 | FNO | +5.36% | [4.49%, 6.25%] | -0.72% | 5.21% | +1.61% | 通过当前 development gate |
| 8 | FNO | +1.09% | [0.57%, 1.64%] | -1.02% | 12.50% | -0.76% | 未过：尾部伤害率过高 |

这组结果说明 set statistics 有可重复的正信息，但当前实现还不能稳定控制每个 field 的修正方向。

## 训练种子不能隐藏

相对 FNO 的逐种子平均 field superiority：

| 模型种子 | K=4 | K=6 | K=8 |
| --- | ---: | ---: | ---: |
| 20260731 | +3.45% | +3.65% | +1.43% |
| 20260732 | +3.04% | +7.17% | -0.96% |
| 20260733 | -1.31% | +5.11% | +2.60% |

K=6 三个种子均为正，是当前最可靠的机制信号。K=4/K=8 各有一个种子为负，证明 seed-collapsed 小正均值不能写成“稳定胜出”。确认性实验要新增 seed SD、worst seed、两层 field/seed bootstrap 和逐种子 gate。

## 分布外边界

- K=6 的主要收益来自 IID 与 noise OOD；thin-front family OOD 和 joint OOD 对 FNO 仍有轻微退化。
- K=8 也主要改善 IID/noise，family/joint 的 field mean 略差。
- 所以当前模型更像“在常见形态上利用视角集合统计减少误差”，还没有解决 near-null thin-front/high-gradient 结构。
- 不能用总体 +5.36% 掩盖逐域翻转。

## v3c 协议设计与执行记录

当前 test fields 已用于 v3a/v3b 诊断，不能继续承担确认性评价。下一版应：

1. 已生成新的 `dev2` 协议：328 fields，与 v3b sample seeds 零重叠；数据文件本体不进公开仓库。
2. 已实现 **zero-initialized set adapter**：49,191 总参数，其中 4,988 个可训练；初始输出与冻结 FNO 最大差为 `0.0`，三次 optimizer steps 后底座参数漂移仍为 `0.0`。
3. 让修正幅度由 K、attention entropy、view spread 和 observable residual 控制；高视角、低不确定度时自动收缩到 0。
4. 加 CVaR/worst-domain 或 paired harm penalty，目标从提高均值改为降低 K=8 尾部伤害。
5. 为 thin-front 增加局部梯度/高频 adapter，但只在 dev2 选择，不看 blind final。
6. 加入同输入/同算力 U-Net、FNO、DeepONet，以及 validation-tuned CGLS/TV/RBF。
7. 轮换多个 audit cameras，clean/noisy 双版本，并加入 independent cone-ray/nonlinear forward。

blind final 目前只完成预注册，尚未生成种子或封存数据。在师兄确认 geometry、CGLS/TV/RBF 调参范围和失败规则前，不允许进入确认性开启。

### v3c K=6 dev2 已返回负结果

- frozen zero-init adapter 相对同 checkpoint continued FNO 为 `-4.965%`，95% CI `[-5.569%, -4.397%]`，p10 `-12.143%`，>1% harm `74.219%`。
- adapter 相对未追加训练的 base FNO 仍有 `+1.425%`，但 continued FNO 相对 base 为 `+5.846%`。
- adapter 训练约慢 `7.18×`，推理约慢 `6.46×`。
- 判决：停止当前 frozen per-view 3D adapter；不开 blind final。

### v3d 最近邻查重后的更正

- NeurIPS 2025 **F-Adapter** 已直接研究 Fourier operator 的 frequency-adaptive PEFT，并指出普通 LoRA 在该场景可能明显弱于 adapter。
- ACML 2025 **R2-FFNO** 与 TMLR 2024 **MG-TFNO** 已覆盖 reduced-rank / tensorized spectral kernels；NeurIPS 2023 **GINO** 已使用 geometry-informed operator 概念。
- 因此“低秩谱调制”本身不能作为创新。v3d 只保留 **BOST acquisition-geometry-conditioned F-Adapter** 假设：相机角度、ray coverage、mask、noise、calibration 只生成轻量频带系数，不再运行 per-view 3D encoder。
- 强制对手改为 saturated/full FNO、last-block、LoRA、vanilla adapter、F-Adapter；机制消融为 correct/shuffled/constant/static geometry。
- 完整查重与门槛：`v3d_prior_art_and_novelty_gate.md`。

### 历史 v3d 96-epoch plateau 闸门

- 在只读取 validation 的预注册规则下，同一 K=6 FNO 已从 24 延长到 96 epochs。
- 红队复核后按每个三维场等权，mean validation L2 从 `0.16646` 降到 `0.11746`；末个 12-epoch block 的 mean/max-seed 改善仍为 `2.81% / 2.96%`，高于 plateau 阈值 `0.5% / 1.0%`。
- dev2 field 后置诊断相对 epoch 24 为 `+16.54% [15.61%,17.45%]`，三种子和四域全正；它不参与停止点选择。
- 当时判决：暂停新分支并先审计 optimizer protocol；后续 240-epoch 双闸门结果见下一节。
- 详情：`v3d_fno_validation_plateau_brief.md`、`v3d_geometry_data_manifest.md`。

### v3d 240-epoch FNO 固定 epoch 基线已锁

- 三策略共享相同 epoch-24 checkpoint 与 block batch seeds；validation 冻结后才读取 dev2。
- `carry continuation-Adam + restart cosine` 是样本等权 validation 冠军：epoch 240 mean L2 `0.094139`，但末 block 仍改善 `0.905% / 0.961%`，未 plateau。continuation optimizer 在 base 后新建。
- `carry continuation-Adam + long cosine` 从 epoch 192 plateau，但 validation `0.095726`，比冠军弱 `1.66%`。plateau 不等于最强 baseline。
- FNO 固定 epoch 与 v3e 五架构成本 schema 已通过：geometry 为 Go 后可开始功能 pilot；跨架构比较前必须补所有候选的 matched 60/120/180/240 error–compute curves。
- 详情：`v3d_fno_optimizer_protocol_brief.md`。

### v3f-v3g DeepONet 公平补强与预算停止点

- v3f 的 rank-48 pooled DeepONet 已有四学习率、三优化协议、三种子 24→240 curve；最终 validation `0.175725`，FNO 为 `0.094139`。当前四个固定 DeepONet checkpoints 均被已观测 FNO error-time 点支配。
- v3g 进一步预声明 8 个可训练 rank/pooling 变体、3 个学习率和 3 个种子，共 72 次固定 24-epoch screen；两个超过 reference `1.5×` 参数的高分辨率变体在训练前排除。
- screen 选中 `rank 64 / pool 4×4×4 / lr 0.002`，但其 240-epoch 最优 validation `0.176094` 比 rank-48 reference 差 `0.210%`。
- 冻结后复用 dev2 相对 rank-48 reference 的 mean field superiority 为 `-0.160%`，field-cluster 95% CI `[-0.339%, +0.006%]`，三个模型种子中两个为负，development gate 失败。
- v3g 按规则只延长短程冠军，不能外推为“所有 DeepONet 变体都更差”。毕设执行档冻结 rank 48；只有师兄明确要求时才一次性预注册 top-3 long-horizon 投稿补充，不允许看 dev2 后继续扩表。
- 详情：`v3f_deeponet_fno_frontier_brief.md`、`v3g_deeponet_capacity_audit_brief.md`。

## 什么时候才允许说“击败现有模型”

至少同时满足：

- 全新 blind fields；
- 对 U-Net、FNO、DeepONet 以及 validation-selected 最强 PEFT/F-Adapter control 做 paired superiority，不只选一个弱对手；
- paired 95% CI 下界大于 0；
- p10 不低于 0，`>1%` harm 不超过 5%；
- 每个 OOD 域都不反向；
- 三个预算和三个种子分别报告；
- 参数、FLOPs/MACs、显存、训练/推理时间并列；
- operator-only 和所有 baseline + same NeRIF refinement 都比较；
- 真实数据至少给出 held-out cameras、独立重复采集和一条外部物理 trace。

## 请何远哲优先审核

1. 是否认可把 `carry continuation-Adam + restart cosine · 240 epochs` 冻结为 FNO epoch 冠军，并保留 long-cosine plateau control？
2. 实际相机布局是否固定 canonical slots，还是必须支持任意角度/数量/顺序？
3. 组内最强直接学习 baseline 是 GRU-BOST、其他 CNN，还是尚未公开模型？
4. CGLS/TV/RBF/UBOST 中哪几个必须作为组内传统强基线？
5. 能否提供 NeRIF 的统一 forward/loss/stop rule，让所有 operator 使用同一 refinement？
6. 无真实 3D GT 时，最认可多个 held-out cameras、PLIF/front、PIV compensation 还是积分量？
7. 真实数据最常见的 view count、geometry drift、bad view 和 noise 是什么？
8. 是否认可 v3e 的 FLOPs-v1/MPS 时间/内存口径，并要求候选共同跑 60/120/180/240 matched learning curves？
9. 是否认可冻结 v3f rank-48 DeepONet 作为毕设 control；投稿前是否还需要一次性 top-3 long-horizon 补充？

## 当前禁止写入摘要的句子

- “提出的新算法全面优于 FNO/DeepONet。”
- “模型在分布外薄前缘上稳定泛化。”
- “Q_audit 证明真实三维场准确。”
- “已经加速 NeRIF。”
- “首次把 attention/FNO 用于 BOST。”

当前可写的是：**在 K=6 linear synthetic development protocol 上，frozen per-view adapter 比同 checkpoint continued FNO 差 -4.96%；样本等权三策略审计锁定了 240-epoch FNO 固定 epoch 冠军；五架构成本账本证明少量可训练参数不等于低训练成本。v3f-v3g 又完成当前 DeepONet 的学习率、优化协议、rank 与 pooling 有界审计：短程 rank-64 冠军长程未胜 rank-48 reference，两者仍明显落后 FNO。复用 dev2 不是新的全项目盲集；真实 geometry、GC-SRO controls、自研候选 matched curve 与确认性 superiority 尚未完成。**
