# 三维 BOST 算子学习收敛路线：几何条件化、近零空间感知、可拒答暖启动

> 当前级别：研究设计与待验证候选，不是算法成功。
>
> **突破监测：尚无算法突破。** 本文没有产生新的 field score，没有打开 sealed fresh，也没有证明任何方法优于 DeepONet、FNO、F-FNO/FFNO、NeRIF 或 TDBOST。

## 1. 一句话收敛

最值得做的主线不是“训练一个网络直接把九幅 BOS 投影变成三维场”，而是：

> **让算子模型根据相机/射线几何识别当前 inverse 中“哪些成分看得见、哪些近似看不见”，只对经典解做受控暖启动或修正；精确 forward/data consistency 和逐相机尾部证书不通过时必须回退。**

用符号写，对装置/几何 `g`：

`y_g = A_g x + epsilon_g + b_g`

`A_g` 取决于相机内外参、背景板、VOI、有限孔径与光线模型。近零空间分量不能由 `y_g` 单独证明；它们只能来自先验、时间邻帧、第二种诊断或明确的不确定性/拒答。

## 2. 这个问题从师兄论文中怎样长出来

| 一级来源 | 它真正证明了什么 | 它没有证明什么 | 对本课题的启示 |
|---|---|---|---|
| [NeRIF, Physics of Fluids / arXiv 全文](https://arxiv.org/html/2409.14722v2) | 用坐标网络连续表示折射率及梯度；数值验证使用 9 视角、RK4 造数与高斯噪声；实验用 8 视角重建、1 视角留出重投影 | 实验没有三维 truth；没有证明跨 rig 的 amortized operator 泛化 | 网络可以是场表示，但换几何与有限视角的可辨识性必须另行处理 |
| [TDBOST, ACM DOI](https://doi.org/10.1145/3809488) 与 [作者代码](https://github.com/Hyz617/TDBOST) | 用张量分解压缩/表示四维折射率场，把时空共享引入 BOST；作者仓库给出数据生成、loader 和 reconstruction 入口 | 低秩时空先验不等于对未见几何/不可观模态有保证 | 我们不应声称“首个 4D BOST”；创新点应放在 geometry/null-space/safety 上 |
| [simultaneous PIV-BOST, Experiments in Fluids](https://link.springer.com/article/10.1007/s00348-025-04093-y) | 一机九路内窥镜同步重建 3D 折射率并补偿 PIV 的热折射误差；摘要报告小型无引导火焰中瞬时速度误差约 `±2%` | 没有证明对所有火焰、平面和几何泛化 | 最终现实价值可以是“三维重建不只好看，还能改善速度诊断” |

基线边界也要说清：[DeepONet](https://www.nature.com/articles/s42256-021-00302-5)、[FNO](https://openreview.net/forum?id=c8P9NQVtmnO) 与 [F-FNO/FFNO](https://arxiv.org/abs/2111.13802) 是必要强基线，但它们原始论文没有自动给出 BOST 的可变相机、未知相关弯曲光线、逐相机尾部与拒答合同。“原始合同没有覆盖”不等于“这些架构永远做不到”。

## 3. 八个可证伪的 research question

| RQ | 真实物理/测量困境 | 最小实验 | 一票否决 |
|---|---|---|---|
| RQ1 跨机位几何条件化 | `A_g` 随相机位姿、背景距离和视角弧段改变 | 5/7/9 视角、完整/有限弧随机 rig；比 geometry-blind、pose concat、ray-set encoder | 均值提高但 worst-rig/逐相机尾部变差，或参数匹配后优势消失 |
| RQ2 近零空间与自信幻觉 | 不同 3D 场可在噪声内产生几乎相同投影 | 用线性化 `A` 的小奇异向量造 indistinguishable phantom pair | 重投影好但 field 错，且模型仍输出狭置信区间 |
| RQ3 非线性光线路径 | 光线本身依赖未知 `n(x)` | 用 RK4 高保真 renderer 生成不同梯度强度；比直线、低秩 correction、精确迭代 | 只降低直线代理损失，精确光线残差反而增大 |
| RQ4 稀疏/不同步 4D | 掉帧、曝光与多路时差会把空间误差写成时间演化 | 20/40/60% 缺帧加相机时偏；报前沿速度、相位和频谱 | 画面更平滑但传播相位/频谱错 |
| RQ5 结构化噪声/坏光线 | 背景纹理不足、饱和、自发光、光纤缺陷和 optical-flow outlier 不是 iid Gaussian | 连续遮挡、整视角缺失、异方差和 outlier 阶梯 | 只在高斯噪声有效，真实 mask 失效或产生高置信伪影 |
| RQ6 重建到 PIV 补偿 | 折射位移除以脉冲间隔后成为速度 bias | 已知 `n,u` 的合成 PIV-BOST，然后一个真实工况 | 速度 RMSE 不降或引入系统 bias |
| RQ7 Gladstone-Dale 不确定 | 从 `n` 到密度依赖波长和组分相关系数 | 造产生近似 `n` 的不同 `(K,rho)` 对，比固定 K、K-conditioned、第二模态 | 对不可区分样本仍输出唯一且高置信密度 |
| RQ8 安全暖启动 | 学习先验可缩短求解，最终结果仍由精确 forward 审核 | 同一精确 solver 比零初值、NeRIF/TDBOST 初值、算子暖启 | 迭代数少但端到端不加速，或最终残差/field 更差 |

## 4. 三个算法候选

### 4.1 优先判伪：GCT-KMix

**Geometry-Conditioned Tail-certified Krylov Mixer** 不直接生成三维场。它先跑同一条 `K` 步 CGLS，得到 `x_0...x_K`，然后只在已通过逐相机残差门的 checkpoint 上输出单纯形权重：

`x_mix = sum_k pi_k(g, early residual, coverage) x_k`, `pi_k >= 0`, `sum pi_k = 1`.

如果每个观测 camera 的残差函数对 `x` 是凸的，那么同一安全集内 checkpoint 的凸组合仍在该残差上界内。模型只学习“在哪些已经安全的停止点之间怎样混合”；E73 联合尾部上界不足时直接返回 `k4`。

- 预计参数：`10k-25k`，两层 MLP 足够做首轮。
- 额外物理调用：0；与同 `K` CGLS 一样。
- 最有可能优势：不同 rig 的 semiconvergence 可从早期轨迹与几何中预测。
- 必败情形：真值改善必须离开 Krylov span，或所有 checkpoint 都伤害某个 camera。
- 创新边界：CG 插值或 early stopping 本身不是新概念；可能的新意只能是 **BOST 几何、camera-tail 风险合同与拒答整体**。

### 4.2 数据条件到位后：AP-LOC

**Adjoint-Paired Low-rank Operator Correction** 用严格成对的低秩 correction 处理廉价直线 operator `A_0` 与高保真光学 forward 之间的系统偏差：

`A_theta = A_0 + U diag(s_theta(g)) V^T`

`A_theta^T = A_0^T + V diag(s_theta(g)) U^T`

同一组因子同时定义 forward 和 adjoint，不允许两个黑盒各学各的。`U,V` 列正交，`|s_j|` 有显式谱范数上界。

- 预计 rank：`4-8`；小型 `32^3` 机制实验可在 Mac 上做。
- 最有可能优势：有限孔径、标定漂移或直线/弯曲光线差异在多个场之间共享低秩结构。
- 必败情形：误差强依赖未知场、本质高秩，或 optical-flow bias 没有稳定结构。
- 前置数据：同一 rig/field 的 low/high-fidelity forward pair 与可校验 adjoint/JVP/VJP。
- 创新边界：[learned operator correction](https://epubs.siam.org/doi/abs/10.1137/20M1338460) 已是已有方向；只有 BOST 严格伴随配对、几何泛化和 null-space 漂移审计可能形成新贡献。

### 4.3 高风险、高上限：ADP-NS

**Abstaining Data-Proximal Null-Space prior** 让 FNO/CNN/DeepONet 类小模型只提议 correction `z_theta`，再用 `A/A^T` 构造近零空间投影，并按可观泄漏上限缩放。

- 优势情形：严重欠采样，但时间邻帧、其他诊断或反应场 morphology 对 null-space 成分有可学预测力。
- 根本必败：真实 null-space 成分与所有辅助信息统计独立；这时任何网络都只是猜。
- 成本：基础重建外再增加若干 `A/A^T`，正式实验必须报端到端调用和时间。
- 创新边界：[Deep Null Space Learning](https://arxiv.org/abs/1806.06137) 和 [data-proximal null-space networks](https://arxiv.org/abs/2309.06573) 已覆盖相邻思想；不可声称首次 null-space learning。

## 5. 为什么第一个做 GCT-KMix

| 标准 | GCT-KMix | AP-LOC | ADP-NS |
|---|---:|---:|---:|
| 当前数据可启动性 | 高 | 低，需 paired renderer | 中 |
| Mac 可判伪 | 高 | 中 | 中 |
| 额外 `A/A^T` | 0 | 每次 operator 增低秩计算 | 显著增加 |
| 可证安全结构 | 观测残差凸组合 + 拒答 | 严格配对 adjoint + 谱界 | 泄漏上界 + 拒答 |
| 论文方法上限 | 中 | 高 | 高但风险大 |
| 最快的否证门 | Krylov convex-hull oracle 没有 headroom | correction 不低秩/不跨场 | 辅助信息对 null-space 无互信息 |

所以当前顺序应是：

1. 先算 GCT-KMix 的 **truth-oracle convex-hull ceiling**；如果最佳凸组合都不能在保护逐相机尾部时胜最强同预算经典法，直接 NO-GO，不训网络。
2. ceiling 有足够 headroom 才训 `10k-25k` 小模型，并用独立 risk-calibration 冻结接管/回退。
3. 同时向何远哲询问 paired low/high-fidelity forward；有数据才启动 AP-LOC。
4. ADP-NS 只在时间邻帧或第二诊断真正提供 null-space 信息时才做。

## 6. 公平比较合同

所有方法使用同一 `A,W,S`、同一 train/development/risk-calibration/sealed-fresh rig split，并报告：

- field relative-L2 与 gradient/H1；
- 逐 camera/view 的 L2、p95 和 worst-rig；
- `>1%` harm 比例、拒答率和 fallback 后指标；
- held-out reprojection，但不把它当三维 truth；
- `A/A^T` 调用、端到端 wall time、峰值内存、训练数据量与搜参预算；
- raw learned output 与套上同一 safety wrapper 的结果，防止只给自己加回退。

| 基线 | 不可偷换的条件 |
|---|---|
| CGLS/PCGLS/H1/TV/Huber | 同初值、同物理调用；同时报固定步、development 选步和 discrepancy stopping |
| DeepONet | 同数据、几何输入、参数档和训练步；明确 branch sensor 与 trunk query |
| FNO/F-FNO | 同 grid、lift、coverage/pose channels、modes、参数与 HPO 预算 |
| NeRIF | 它是逐实例优化，不做虚假参数匹配；比同 rays/views 的 time-to-quality、显存、field 和留出视角 |
| TDBOST | 同序列、时间 mask、几何、训练/优化时间与事件条件指标 |

exact oracle 只能作 ceiling，不得进入可部署排行榜。同一 rig 下的多个 field/frame 不得冒充独立 rig。

## 7. Mac 上的两周最小实验

| 时间 | 任务 | 停止门 |
|---|---|---|
| D1-2 | 冻结 `A/A^T` dot test、单位、support、camera groups、噪声白化和调用计数 | adjoint error 超阈值就停 |
| D3-4 | `16^3` 跑 CGLS/H1/TV/Huber 路径，算 GCT-KMix oracle convex-hull ceiling | 相对最强同预算经典法没有有意义 headroom，或任一 camera harm 超门，不训网络 |
| D5-6 | 训 `10k-25k` MLP，3 seeds；输入只含部署可见几何/早期轨迹 | 去 geometry 后表现不降，说明没有 geometry-learning 证据 |
| D7-8 | 独立 risk calibration；冻结接管/回退阈值 | 样本量不足就只报经验风险，不写保证 |
| D9-10 | geometry、view-count、noise、thin-front/full-family OOD | 均值赢但 worst-rig/逐相机输，NO-GO |
| D11-12 | 消融和 DeepONet/FNO/F-FNO 同预算比较；raw 与 gate-wrapped 两版 | 优势只来自自己有 fallback，失败 |
| D13 | 冻结 code/config/split/hash/图表模板 | sealed fresh 在此前不得打开 |
| D14 | 一次性 fresh 评估 | 只有 rig-cluster CI、全 camera/tail no-harm 与端到端成本同时过门才升级 |

本机适合 `16^3` 完整判伪与小型 `32^3` mechanism screen。不应显式存储百万/千万 ray 的稠密矩阵，必须 matrix-free。系统级搜参、大序列 4D FNO/transformer 等到真实数据合同与小规模 headroom 都通过再租 GPU。

## 8. 最先问何远哲的十个问题

1. 能否提供每个 view 的内参 `K`、外参 `R,t`、镜头畸变、背景平面、VOI 和坐标单位？
2. 9 个 view 的精确顺序和 camera/fiber 对应关系是什么？
3. 实验室的 forward 是直线、有限孔径还是弯曲光线？有无可调的 `A/A^T` 或 JVP/VJP？
4. 能否为同一 field/rig 生成 low-fidelity 与 high-fidelity 成对 forward？
5. 真实数据中哪些因素会变：camera pose、背景距离、火焰工况、视角数、光纤路径？
6. 是否保留 raw reference/distorted image、optical-flow displacement、confidence、saturation/坏光纤 mask？
7. 多路图像是否有逐 view 时间戳、同步误差、曝光与掉帧记录？
8. 真实实验的统计独立单位应按 run、day、session 还是 rig 分？
9. 是否有同步 PIV 图像对、脉冲间隔、激光平面位姿/厚度和低折射参考工况？
10. 师兄认为当前最真的误差主导项是有限视角、直线光线偏差、optical-flow 噪声、标定漂移还是时间不同步？

## 9. 没拿到 OERF 数据前的公开数据梯子

当前没有一个公开数据同时满足“真实反应流 + 同步少视角 + 3D/4D BOST + 折射率 truth + 完整 `A/A^T`”。因此必须分层，不能用 PDEBench 或 CT 上的好结果代替 OERF 证据。

| 层 | 数据/公开入口 | 能验证什么 | 不能写什么 |
|---|---|---|---|
| G0 算子正确性 | [HTC 2022](https://zenodo.org/records/6984868) 的小型真实 sinogram | CPU forward/adjoint、有限角、CGLS/TV 基线和 dot test | X 射线衰减不是 BOST 梯度偏折 |
| G1 动态逆问题 | [STEMPO](https://zenodo.org/records/8239013) 的 2D+t/3D+t 小文件 | 时间缺投影、low-rank/L+S、loader 与事件指标 | 不是折射率光学 |
| G2 核心 synthetic BOST | 解析 Gaussian/thin-front/shock-like + 独立高保真 ray integration；可配 [photon](https://github.com/lalitkrajendran/photon) | 跨 forward、跨几何、null-space pair、GCT-KMix/AP-LOC 机制 | 同一 generator 内的 mean gain 不是真实 BOST 泛化 |
| G3 真实三维 BOS | [Penn State Open BOS](https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/) | 真实 flow-off/on、位移、mask、标定、多视角 cone-ray 与留出投影 | Mach 4.8 飞行体、时间平均和旋转采集不是同步火焰 4D |
| G4 真实反应流前端 | [TU Graz HBOS sample](https://repository.tugraz.at/records/cvcf2-28b98) 与 [9-angle reactive tomography](https://repository.tugraz.at/records/nzz9b-rn487) | 火焰图像、配准、位移/相位稳定性、九角度截面层析 | 没有完整同步 3D ray bundle/field truth 时不报 field-L2 |
| G5 软件基线 | [PDEBench](https://github.com/pdebench/PDEBench)、[NeuralOperator](https://github.com/neuraloperator/neuraloperator)、[DeepXDE](https://github.com/lululxvi/deepxde)、[FourierFlow](https://github.com/alasdairtran/fourierflow) | 检查 FNO/DeepONet/FFNO 实现、优化器、分辨率转移和公平 HPO | 无相机、无 BOST forward，无论多高的成绩都不构成 BOST 成功 |

Mac 首批不必一次下 50 GB：先取 HTC 约 1.5 MB、STEMPO 小型 2D+t/3D+t 与几何约 7 MB，再取 TU Graz HBOS sample 中约 274 MB 的样例压缩包。Penn 51.7 GB 全量数据只在 loader、指标、几何合同和存储目录冻结后再下载。

OERF adapter 最低需要下列字段：`run_id/rig_id/timestamp/split_group`，flow-off/on 图像，位移与 confidence/mask，逐 view `K/distortion/R/t`，背景平面/波长/像素尺寸，以及线性 `forward/adjoint` 或非线性 `forward/Jv/J^Tv`。数据 adapter 可替换，split、种子、基线、指标和门槛不能随 OERF 结果调整。

## 10. 什么时候才标“突破”

只有下列条件在未见 rig/session 上同时满足，网页才允许出现“突破性进展”：

1. 相对最强同预算经典法与同输入 DeepONet/FNO/F-FNO，field relative-L2 的 rig-cluster 置信区间下界越过预注册门；
2. 每个 camera/view-tail 的 no-harm 门同时通过，不靠 pooled mean 掩盖；
3. raw 和 safety-wrapped 结果都完整报告，非平凡接管率不是靠几乎全 fallback 获得；
4. `A/A^T`、增值 forward、失败重试、训练/HPO 和端到端 wall time 均计入成本；
5. geometry、noise、view-count、thin-front 与真实数据至少一类外部移位完成；
6. 关键模块消融、参数匹配、多 seed、完整失败案例和可重现代码同时到位。

任何只在当前 synthetic generator 上的 mean gain、oracle gain、单个 seed 或重投影改善，都只能标成“机制信号”，不是算法、真实或泛化成功。
