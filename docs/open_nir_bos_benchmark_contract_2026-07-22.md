# NIR-BOS 三维 benchmark 与算法候选合同

状态：`REGISTERED_DESIGN_ONLY`

适用范围：公开 Phantom 1 开发、后续 OERF/何远哲数据接线

禁止解释：本文件不是实验结果，也不授权算法、真实或泛化成功

## 1. 研究问题

在相同相机、相同观测、相同 ray budget、相同参数量级和相同端到端成本下，能否用一个**噪声与饱和感知、支持域外自动回退的表示/修正算子**，同时改善三维折射率场、梯度/前沿、held-out displacement 和部署尾部，而不靠更多 rays、更多迭代或 test-time 调参？

如果只改善单个 phantom 的 PSNR，研究问题没有被回答。

## 2. 数据证据阶梯

| 阶段 | 数据 | 合法结论 | 禁止结论 |
|---|---|---|---|
| D0 | 作者 Phantom 1，12/2/2 文件；val/test 复用 train 位姿且彼此像素重复 | loader、损失、梯度、成本与可视化能否工作 | unseen-view、跨 field/OOD/真实泛化 |
| D1 | Phantom 1 的冻结 noise/blur corruption | 同一 field 的鲁棒性曲线 | 新物理场泛化 |
| D2 | 至少 5 个独立 synthetic 3D fields，leave-one-field-out | 跨形态 synthetic field | 跨 geometry/真实火焰 |
| D3 | 未见 camera layout、view count、aperture 或 calibration | 跨 geometry synthetic robustness | 实验 apparatus 泛化 |
| D4 | 独立实验 flame/session，固定审计 split | 该装置下的 held-out measurement/物理 endpoint | 三维真值，除非真有独立 GT |
| D5 | OERF 独立 acquisition 与重复实验 | 受数据合同约束的实验结论 | 超出 apparatus 和组成模型的普适结论 |

所有 split 以 field/sequence/rig/session 为单位。frame、pixel、ray 或同序列 patch 不能跨 train/test 后被称为独立样本。

## 3. 五个不可缺的 baseline

| ID | baseline | 作用 |
|---|---|---|
| B0 | CGLS 或 TV-CGLS | 防止网络只打败弱迭代器 |
| B1 | Fourier + discrete gradient | 当前 Mac 最先可做的稳定神经隐式底座 |
| B2 | Fourier + hybrid gradient | 判断 AD monitor 是否真有信息 |
| B3 | smoothstep hash + discrete gradient | 对照薄前沿分辨率与噪声过拟合 |
| B4 | smoothstep hash + hybrid gradient | 2026 论文的直接新颖性边界 |

再加三个防作弊对照：零扰动场、训练集均值场，以及 `A^T W y + ridge`。若进入跨实例 operator 赛道，再加入同预算 3D U-Net；否则 DeepONet/FNO 的胜负可能只是输入表示不公平。

若样本量足够做跨实例映射，再加 DeepONet、FNO/iFNO。单实例 neural implicit fitting 与跨实例 neural operator 不是同一任务，不能只比较最终误差而忽略训练样本、test-time optimization 和推理成本。

## 4. 三个候选算法

### H1：NASH-Mix

暂名：`Noise-Aware Saturation-Hardened Fourier-Hash Mixture`。

```text
measurement + geometry summaries
              |
         bounded gate g
          /           \
Fourier base       hash residual
          \           /
        RI field proposal
              |
discrete-gradient driver + AD consistency monitor
              |
 exact held-out reprojection / fail-closed fallback
```

核心限制：hash residual 的能量和频带受限；gate 在 support 外固定回到 Fourier；训练主损失不允许通过边界 clipping 获得假低误差。

最小成功门：在 D2 的 leave-one-field-out 上，field、gradient/front、held-out displacement 和 saturation 四项中至少三项优于 B1--B4，且最差 field 不退化超过预注册容忍度；同预算 wall-clock 与 ray samples 不能偷跑。

关闭条件：收益被参数量匹配、更多 rays 或单个 corruption seed 消除。

### H2：SAT-4D

暂名：`Saturation-Aware Temporal Residual Operator`。

每帧先由冻结空间 base 重建，时序算子只输出有界 residual 与置信门。front birth/extinction、视角缺失或 residual support 变化时，门应停止传播历史并退回逐帧 base。

最小成功门：独立 sequence 上同时改善 field/gradient temporal error、held-out view、flicker 和极端帧 p90；必须对比逐帧 B1、简单 temporal smoothing、low-rank tensor、GRU/TDBOST-style baseline。

关闭条件：只让画面更平滑却抹掉真实前沿，或收益只存在于相邻帧泄漏 split。

### H3：SGEC

暂名：`Support-Gated Exact Correction`。

网络只输出 warm start、编码权重或低秩 residual proposal；真实 forward/JVP/VJP 再做固定 1--2 次 correction。support gate 不过时回退到 B0/B1。

最小成功门：相同 exact-call 数下超过 simple damping、fixed interpolation、classical warm start 和直接网络；逐 rig/session 尾部与成本同时过门。

关闭条件：没有真实 callable/JVP/VJP，或 correction 成本未与 baseline 对齐。

## 5. 冻结指标

### 有三维真值时

- 折射率扰动 `delta n = n - n0` 的 field relative-L2，避免完整 `n` 接近 1 稀释误差；
- gradient relative-L2 或前沿加权 gradient error；
- front location：阈值固定后的 Chamfer/Hausdorff 或距离误差；
- saturation fraction：落在值域上下界容忍带内的 voxel 比例；
- field PSNR/SSIM，只作补充。

### 所有数据都要有

- held-out displacement relative-L2 与逐 camera p90；
- measurement residual 与一个独立审计 forward 的一致性；
- wall-clock、peak memory、参数量、训练 steps、ray samples；
- forward、JVP、VJP 或 `A/A^T` 调用数；
- 每 field/rig/session 的中位、p90、worst，不只报 pooled mean。

首版建议冻结三个工程门，之后只允许由师兄在看结果前调整：相对最强 tuned baseline 的主 field error 至少改善 5%，paired hierarchical bootstrap 95% CI 下界大于 0；新相机 reprojection 不差于基线的 1.05 倍；定义单场误差超过基线 1.05 倍为 harm，harm fraction 不超过 10%。这三个数是预注册工程阈值，不是自然定律。

### 实验温度解释

折射率到密度/温度需要组成、压力、Gladstone--Dale 或状态方程假设。没有独立组成/温度证据时，温度只能标成模型依赖派生量；到边界的饱和必须单独报告，不能裁剪后只报漂亮图。

## 6. 预注册 corruption 与 OOD

开发前冻结：

1. 位移噪声：至少 4 档，加独立 seed；
2. defocus/blur：至少 3 档，kernel 不与训练完全同族；
3. view dropout：固定和随机两种；
4. limited angle：冻结角域；
5. calibration perturbation：位姿、焦距或尺度中只选实验真实存在的项；
6. value-range stress：检验 saturation，不允许用 test GT 重调 bound。

D0/D1 的 corruption 只能筛反例。主张 robustness 至少需要 D2；主张 geometry OOD 至少需要 D3。

## 7. 最小消融矩阵

| 消融 | 要回答的问题 |
|---|---|
| Fourier only / hash only / mixture | 收益来自组合还是单专家 |
| fixed 50:50 / learned gate | gate 是否超过简单插值 |
| discrete / AD / hybrid | 导数项是否真贡献而非成本增加 |
| no saturation penalty | 边界保护是否必要 |
| no support fallback | OOD 尾部是否被门保护 |
| matched params / matched rays / matched wall | 优势是否来自额外预算 |
| exact correction 0/1/2 steps | correction 的边际收益与成本 |

## 8. 统计与判决

- 至少 5 个训练 seed；正式 seed 在协议中冻结。
- field/rig/session 是统计单位，不以百万 voxel 假装百万独立样本。
- 报 paired difference、bootstrap confidence interval 和逐实例散点。
- 主方法必须在预注册主指标上胜出；不能看完结果再把最佳子组改成主结果。
- mean gain 与 tail safety 分开。若 mean 改善但 worst/p90 明显退化，状态为 `NO-GO`。
- 任何 NaN、路径跳样本、test-time GT 调参、split 泄漏或预算不匹配都 fail closed。

## 9. 计算顺序

### Mac 阶段

1. loader/path contract；
2. Fourier encoder/MLP MPS 单元测试；
3. `32^3` 或更小 clean-room analytic BOS smoke；
4. B0/B1 的 50--200 step overfit test；
5. 固定指标、日志、checkpoint manifest 与图表；
6. H1 极小版本，只做反例筛选。

### NVIDIA 阶段

1. 作者 baseline 50--100 step smoke；
2. B1--B4 单 seed 小网格；
3. 参数/ray/wall 匹配；
4. D0/D1 多 seed；
5. 只有 D2 数据到位才做正式 H1；
6. 只有时序/OERF 合同到位才打开 H2/H3。

## 10. 给师兄审核的一句话

> 我准备先以公开 NIR-BOS primitive 为强基线，研究噪声与温度饱和下 Fourier/hash 表示的互补和 fail-closed 修正；但当前公开仓库只有一个 Phantom 1 且 CUDA/Windows 耦合明显。请师兄先确认组内真实 forward/JVP/VJP、独立 field/session 数量、主物理痛点和可用 NVIDIA 环境，我再冻结 H1、H2 或 H3，不会先训练再补物理故事。

**突破监测：没有突破。** 本合同只把可发表问题、强对手、反例和成本门写清楚。
