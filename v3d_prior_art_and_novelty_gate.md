# v3d 最近邻文献与创新门槛

更新时间：2026-07-11

## 一句话判决

不能把“给 FNO 加 LoRA、低秩谱核或 frequency adapter”本身写成创新。最近邻工作已经覆盖这些组件。当前仅保留一个更窄的待验证假设：

> **用 BOST 的采集几何与视角可靠性，条件化一个轻量 frequency adapter；证明它在相同信息、参数、训练和墙钟预算下，稳定优于不看采集几何的 F-Adapter、LoRA、last-block fine-tuning 与 continued/full FNO。**

这是工作假设，不是原创性结论。最终是否可写成方法贡献，仍取决于系统检索、师兄判断、独立数据和 blind evidence。

## 当前双闸门状态

| 闸门 | 当前证据 | 判决 |
|---|---|---|
| fixed-epoch FNO baseline | 三策略已到 240 epochs；carry-continuation-Adam/restart-cosine 是 validation 冠军，long-cosine 是 plateaued control | **FNO 内部闸门通过** |
| cross-architecture compute | 尚未补 FLOPs、峰值内存与 time-to-target | **未通过；只可写功能 pilot** |
| confirmatory plateau | validation 冠军末 block 仍改善 0.958%/1.041%，未满足 0.5%/1.0% 阈值 | **未通过；不得声称击败饱和 FNO** |
| acquisition geometry | 尚未获得组内逐 case geometry/data manifest | **待何远哲确认；固定 geometry 可直接 No-Go** |

优化协议与固定 epoch 合同见 `v3d_fno_optimizer_protocol_brief.md`，组内接口见 `v3d_geometry_data_manifest.md`。geometry 为 Go 后可实现 development pilot；跨架构算力、确认性 plateau 与论文门槛通过前，它仍不是 superiority 结论。

## 最近邻工作：它们已经做了什么

| 角色 | 论文 | 已覆盖内容 | 对本项目的约束 |
|---|---|---|---|
| 最近邻强基线 | [F-Adapter, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a12e362d89d4e0b40760f839f91550ee-Abstract-Conference.html) | 系统比较 Fourier operator 的 PEFT；指出 stacked LoRA 在 Fourier layers 上可能出现误差下界，并按频率复杂度给不同频段分配 adapter 容量。 | “频率自适应 adapter”不再新；必须复现或实现同等强度的 F-Adapter control。 |
| 低秩谱核 | [R2-FFNO, ACML 2025 / PMLR](https://proceedings.mlr.press/v304/chou26a.html) | 对 FFNO spectral kernels 做 reduced-rank factorization，并讨论 rank 饱和和高频 augmentation。 | “低秩 spectral kernel”不再新；rank 必须按 validation 选择并报告曲线。 |
| 张量化谱算子 | [MG-TFNO, TMLR 2024](https://openreview.net/forum?id=oFqHIkw8sd) | 用全局 tensor factorization 压缩 Fourier-domain parameters，并结合 multi-grid domain decomposition。 | 必须区分“参数压缩”与“采集几何条件化”；不能只凭参数更少作贡献。 |
| 几何神经算子 | [GINO, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) | 用 point cloud/SDF、graph operator 与 latent-grid FNO 处理可变物理域几何。 | “geometry-informed operator”已存在；本项目必须明确 geometry 指相机/光线采集算子，而不是物理域形状。 |
| 一般谱参数更新 | [FourierFT, ICML 2024 / PMLR](https://proceedings.mlr.press/v235/gao24o.html) | 在一般模型微调中只学习权重更新的少量 Fourier coefficients。 | 仅学习频域更新系数不构成 BOST 创新，但可作参数效率参考。 |

## v3d 的可检验模型假设

令每个样本的采集描述为：

```text
G = {(theta_i, ray_coverage_i, mask_fraction_i, noise_level_i, calibration_i)} for i=1...K
```

先用轻量、排列不变的 set encoder 得到 `z_G`。对第 `l` 个 FNO block 和频带 `b`，只生成少量调制系数：

```text
Delta W_l,b(G) = sum_r alpha_l,b,r(z_G) * A_l,b,r * B_l,b,r
W'_l,b = W_l,b + Delta W_l,b(G)
```

关键限制：

1. 不再用 per-view 3D convolution；几何分支只处理相机级标量或低维统计，避免重演 v3c 的 7.18x/6.46x 成本失败。
2. `alpha` 的末层零初始化，使模型初始严格等于 saturated FNO。
3. `A/B` 的 rank、插入层和总参数与 plain low-rank control 匹配。
4. 若相机布局固定、`z_G` 没有跨样本变化，这条方法没有可识别收益，应直接停止。

## 必须同时出现的六个对手

| 对手 | 回答的问题 |
|---|---|
| validation-saturated FNO | 继续训练底座是否已经足够？ |
| full/last-block fine-tuning | 普通可训练自由度是否解释收益？ |
| static LoRA / low-rank spectral update | 低秩本身是否解释收益？ |
| vanilla bottleneck adapter | 非低秩 adapter 是否更适合 operator？ |
| F-Adapter-style frequency allocation | 频率自适应本身是否解释收益？ |
| geometry-shuffled v3d | 模型是否真的使用了正确采集几何，而非额外噪声或参数？ |

## 三个决定机制归因的消融

1. **geometry shuffle**：保持每个样本的观测不变，随机置换 `z_G`。若性能不降，geometry claim 失败。
2. **constant geometry**：用训练集均值几何替换 `z_G`。若与条件化版本等价，方法退化为普通 F-Adapter。
3. **same-rank static control**：保留相同 `A/B`、rank、层数和参数，只把 `alpha(G)` 改为全局可训练常数。

## 开发门槛

v3d 只有同时满足以下条件，才允许冻结方法并申请 blind final：

- 相对 validation-selected 最强 control 的 field superiority 95% field-cluster CI 下界大于 0。
- 三个 model seeds 的均值全部为正；不能靠 seed 平均掩盖翻转。
- `p10 >= 0`，且 field degradation 超过 1% 的 harm rate 不高于 5%。
- clean `Q_audit` 非劣，并在 thin-front、noise、family、joint OOD 中没有系统性反转。
- geometry shuffle 和 constant-geometry 两个消融都显著劣于正确 geometry。
- 参数、训练墙钟、单样本推理延迟和显存全部报告；目标是推理不超过 saturated FNO 的 1.5x。

这些是 development gates，不是论文发表保证。

## 本科可执行的十天阶段闸门

六个对手是论文级完整矩阵，不要求十天内同时跑完。前十天只回答“真实 geometry 是否值得继续”和“最小 F-Adapter control 能否复现”；任一中间闸门失败就停，不用为了凑模型把本机一直占满。

| 天 | 只做一件事 | 可验收产物 |
|---:|---|---|
| 1 | 与何远哲核对 geometry 是否跨样本变化 | geometry/data manifest；固定且无扰动则停止 v3d |
| 2 | 延长 FNO 训练 | error-vs-epoch/time 曲线 |
| 3 | 锁 validation saturation | checkpoint hash 与 stop rule |
| 4 | 跑 matched last-block control | 参数、墙钟、收敛表 |
| 5 | 实现最小 F-Adapter-style control | 频带容量表、shape 与 zero-init tests |
| 6 | 定义相机级 `G` | angle/coverage/mask/noise/calibration feature manifest |
| 7 | 实现 acquisition-conditioned coefficients | exact fallback、permutation 与 parameter-match tests |
| 8 | 单种子 K=6 smoke | 只判故障、成本和明显负向，不选最终结论 |
| 9 | correct vs shuffled/constant geometry | 第一张机制对照图；无差异则停止 |
| 10 | 写继续或停止 brief | 师兄可审核的一页判决与下一阶段预算 |

只有第 10 天决定继续后，才补 static LoRA、vanilla adapter、same-rank static control、三种子 dev2 和独立 forward；这些属于下一阶段，不伪装成十天内必做完。

## 给何远哲的三个问题

1. 组内相机布局是否跨工况变化？若布局固定，能否用遮挡、缺失相机、标定漂移或不同 K 构成真实的 acquisition-geometry distribution？
2. 是否认可 F-Adapter-style control 是 v3d 的最近邻强基线，而不是只比较 DeepONet/FNO？
3. 组内最可信的独立审计信号是 held-out camera、PIV correction、PLIF/front、积分量，还是重复实验？

## 当前禁止表述

- 不说“首次把低秩/Frequency Adapter 用于神经算子”。
- 不说“geometry-informed neural operator 是原创概念”。
- 不因 v3c 比 base FNO 好 +1.42% 就声称 ray-set adapter 有效；同追加训练下它比 continued FNO 差 -4.96%。
- 不在 dev2 反复试 rank 后把同一 dev2 当确认集。
- 不开放 blind final，不提前接 NeRIF，不把 synthetic 小场结果写成真实 BOST superiority。
