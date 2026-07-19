# PSU rotation-40 网格分辨率迁移：预注册 NO-GO 与算法转向

> 机器判决：`SUPPORT_RESOLUTION_GAIN_DID_NOT_CLEAR_NUMERICAL_TRANSFER_GATE_NO_GO`
>
> 证据单位：同一组三台物理相机的一个未参与重建 rotation run，不是新相机，也不是三个独立重复。
>
> 预注册性质：`32³` 的 rotation-40 分数在冻结前已知，`16³` 分数未打开；这是 **16³ 正式评分前冻结的单侧 development 比较**，不是完全盲的 confirmatory experiment。
>
> 最重要的结论：冻结的 `32³ + CGLS4` reconstruction package 在九个 support views 上拟合更好，却在 rotation-40 的真实位移观测上对三台相机全部更差。继续单纯增大离散分辨率没有得到授权。

## 1. 一句话讲人话

可以把 support views 想成练习题，把 rotation-40 想成同一套仪器换了一个没有参与重建的观察角度。`32³` 有更多自由度，在练习题上把误差从 `0.787711` 降到 `0.627132`；但换到 rotation-40 后，它的误差从 `16³` 的 `0.843263` 上升到 `0.959591`。

因此，当前看到的是一个 **support-fit / held-out-reprojection reversal**：冻结的 `32³+CGLS4` package 能更好解释已见投影，却没有把这种拟合迁移到这一个新旋转运行。它提示“额外自由度或不同求解轨迹可能吸收 support-specific 成分”，但这只是候选机制。相同四步 CGLS 在 16³/32³ 上并不对应相同谱滤波或收敛阶段；仅凭一个 rotation block 不能把 reversal 单独归因于网格、过拟合、噪声、标定误差、有限视角 null space、forward mismatch 或真实高频结构中的任一种。

## 2. 为什么这次结果可信到什么程度

正式评分前先做了两个 Git commit：

1. protocol commit `ba77a17f7c7bb5ba42a9a55a9de776e267d5f9d0` 固定配置、runner、测试、forward、active-ray store、metric 与说明；此时正式结果目录不存在。
2. 第二个 commit 单独保存 attestation，记录 protocol SHA、全部受监控文件哈希、两个场及其生成报告哈希、support split、rotation-40 payload/geometry 哈希与 `ROTATION_RUN_NOT_CAMERA` 身份。

runner 在评分前重新验证：

- 16³/32³ 场 SHA、shape、dtype 与私有生成报告；
- 固定四步 CGLS、零场初始化、九个全 active support views、QMC-16、outer-zero gauge；
- camera 2/3/4 的 payload shard、geometry manifest 与所有 `.npy` 的 SHA、shape、dtype；
- payload manifest 到 geometry manifest 的交叉哈希；
- attestation 已由 Git 跟踪、无未提交改动，protocol commit 是当前 HEAD 的祖先；
- 正式输出目录不存在，且公开报告不含逐射线 measurement、prediction、geometry、volume 或私有路径。

全部 `3,847,050` 条 rotation-40 active rays 被使用；每个候选恰好一次完整 forward，所有 ray 命中 B0，输出有限，outer boundary 为零。公开 checksum 逐文件通过。

独立审计还指出一条未完全机械闭合的 provenance 边界：候选场与九个匿名 support view 的报告/哈希已绑定，配置也冻结 camera 2/3/4 × rotation 0/50/90；但 runner 没有再绑定记录匿名 view ID 到 camera/rotation 映射的 held-out protocol/source-script 哈希。因此 same-camera rotation holdout 的语义与公开数据/既有协议一致，却不能写成端到端机械证明已经完备。

## 3. 冻结数字

### 3.1 主判决与 pooled 标签勘误

| 指标 | 16³ | 32³ | `16³ - 32³` 改善量 |
|---|---:|---:|---:|
| 九 support views pooled rel-L2 | 0.7877107 | 0.6271325 | +0.1605782 |
| rotation-40 pooled rel-L2 | **0.8432631** | 0.9595912 | **-0.1163281** |
| rotation-40 equal-camera macro | **0.8251725** | 0.9309102 | **-0.1057377** |

预注册筛查要求 pooled 改善至少 `+0.01` 且三台相机均不退化。实际 pooled 是 `-0.116328`，方向相反，因此直接 NO-GO。这里的 `0.01` 只是预声明数值线，不是物理或工程显著性阈值。

**Post-open metric-label erratum：**冻结配置/summary 把 pooled weighting 写成 `RAY_COUNT_WEIGHTED_OVER_ALL_SELECTED_ROWS`，这个标签不够准确。实际实现是先拼接全部 ray，再计算

```text
sqrt( sum_c ||prediction_c - measurement_c||² / sum_c ||measurement_c||² )
```

它不是三个 camera relative-L2 的 ray-count-weighted arithmetic mean；更准确的名称是 **concatenated-all-ray vector relative-L2**，其平方可理解为按各相机 measured energy 加权。公式与数值没有变化，且 equal-camera macro 与三台 camera-wise delta 方向一致，所以勘误不改变 NO-GO 判决。

### 3.2 每台相机

| 物理相机 | rays | 16³ rel-L2 | 32³ rel-L2 | `16³ - 32³` | 判定 |
|---|---:|---:|---:|---:|---|
| camera 2 | 959,098 | **0.762675** | 0.824194 | -0.061519 | 32³ 退化 |
| camera 3 | 1,643,912 | **0.872902** | 0.982907 | -0.110005 | 32³ 退化 |
| camera 4 | 1,244,040 | **0.839941** | 0.985630 | -0.145689 | 32³ 退化 |

三个 camera-wise delta 全部同向，所以 reversal 不是单纯由 camera 3 的较多 rays 在 pooled 指标中加权造成。但三台相机共享一个 rotation-40 物理运行，不能把它们当三次独立复现实验。

### 3.3 幅值与成本诊断

rotation-40 measured vector RMS 为 `0.302716 px`。16³ prediction RMS 是 `0.143223 px`，32³ 只有 `0.082605 px`；两者都明显低估观测幅值，32³ 更严重。没有做 post-hoc amplitude scaling，因为从 rotation-40 拟合尺度再评价同一 rotation 会造成泄漏。

由 rel-L2 与三组 RMS 事后反推的 pooled prediction/measurement cosine 约为 `0.542`（16³）与 `0.282`（32³）。这是一项未预注册的探索性诊断，只提示 32³ 不仅幅值更小，整体方向/空间对齐也更弱；它不参与判决，必须由空间残差图、ROI 与独立 rotation 重复验证。

16³/32³ 的一次完整 forward 分别约 `6.63 s / 6.66 s`，调用后 peak RSS 约 `1.250 / 1.284 GB`。这只是当前 matrix-free interpolation 实现的一次 forward；32³ 的名义网格值是 8 倍、自由内点约 9.84 倍，因此“同调用数”不等于同参数量或公平算法成本。

## 4. 允许说与禁止说

### 允许说

> 在 PSU 已打开的 rotation-40 development、同一物理相机集合上，固定四步 CGLS 的 32³ support-fit 优势没有迁移；相对 16³，pooled 和三台相机的 image-space relative-L2 均退化。

### 不能说

- 不能说 16³ 恢复了更准确的真实三维密度场：没有实验 volumetric truth。
- 不能说 32³ 或细网格发生了统计意义上的过拟合：只有一个独立 rotation block，且网格与固定四步 CGLS 求解轨迹混杂。
- 不能说 16³ 算法优于 32³ 算法：两者是同一 CGLS 流程的不同离散网格，参数量与成本不公平。
- 不能说找到了可投稿的新算法：这次是基线诊断和路线否证。
- 不能打开 final rotations 来反复挑正则、网络、尺度或停止步数。

## 5. 它暴露的真实研究问题

现阶段最值得研究的不是“用 FNO 代替 CGLS”，而是：

> 如何让高分辨率 correction 只保留对未见旋转仍稳定的成分，并在证据不足时退回更稳的 coarse reconstruction？

这比直接比较 DeepONet/FNO/FFNO 更贴近三维 BOST 的物理困难：有限视角 inverse 的高频自由度可能处于弱可观测或 null-space-like 子空间；calibration、位移估计、mask 与 ray model 的小失配会在细网格上被放大。神经网络若只优化 support residual，很可能学会复现同一种失配。

## 6. 下一算法候选：先做最小可证伪版本

暂定研究名：**Rotation-Transfer-Gated Multiresolution Correction，RTG-MRC**。这只是工作名，必须完成一级文献原创性检索后才能称“新方法”。

### 6.1 结构

1. 先显式比较 prolongated coarse field `U x_16` 与原生 `x_32`，隔离表示网格变化和求解轨迹变化；再决定 correction 的输入。
2. fine branch 只预测 correction `delta_theta(y, geometry)`，输出 `x_f = U x_c + delta`，而不是从零直接生成三维场。
3. 对 correction 加 coarse consistency：`R delta ≈ 0`，避免改写 coarse 可观测主体。
4. 对 support 数据加 exact data-consistency 或固定预算 Krylov projection，网络不能绕过 `A`。
5. 用 rotation-group cross-validation 检查 correction：在 0°/50°/90° 中按整组旋转留一组，而不是随机拆 ray。
6. 若 correction 在任一留出 rotation 伤害 relative-L2、幅值或 tail，则 gate 退回 `U x_c`。

### 6.2 最小对照

在训练网络前，先完成以下经典基线；否则无法知道收益来自学习还是正则化：

- 32³ CGLS 的固定 1/2/3/4 步轨迹，但停止规则只能由 support 内 leave-one-rotation-out 或 flow-off noise floor 决定；
- 32³ Tikhonov/H1、TV 与 coarse-to-fine continuation；
- 16³ 与 32³ 的统一幅值、tail、逐 rotation 指标和成本；
- coarse prolongation `U x_16` 与原生 `x_32`，隔离“表示分辨率”与“求解轨迹”的影响。

### 6.3 第一阶段成功门

在不看 final rotations 的前提下，至少要求：

- 三个 support rotation-group folds 全部不伤害 coarse baseline；
- rotation-40 development pooled、equal-camera macro 与 worst camera 全部不伤害；
- 没有从 rotation-40 拟合 amplitude scale；
- 固定调用预算下报告 `A/A^T` 次数、wall time、RSS 与参数量；
- synthetic/CFD truth 上同时报告 field relative-L2、前沿/激波位置和守恒或 PDE residual；
- 通过后才预注册一次 final rotation audit。

## 7. 现在应向何远哲师兄确认的五件事

1. 组内真实目标究竟是折射率/密度场，还是只要求 held-out BOS image consistency？是否有 CFD 或其他三维参考？
2. 是否有同一 camera/rotation 条件下的 flow-off 或重复采集，用于估 measurement repeatability、camera covariance 与实际显著性下限？
3. 何师兄当前 NeRIF/TDBOST pipeline 的 train/validation 单位是 ray、camera、rotation、time block 还是整个 rig/session？
4. 组内最常见失败是幅值偏小、shock/front 变钝、局部伪影、跨相机不一致，还是曲光线/标定 mismatch？能否提供一个匿名失败 case？
5. 可否提供最小 callable：`forward(x, geometry)`、JVP/VJP 或至少固定输入输出样例，以及每次 ray/sample 的真实成本账本？

## 8. 复现与资产

- [预注册说明](psu_rotation40_resolution_transfer_prereg_2026-07-19.md)
- [独立科学与复现审计](psu_rotation40_resolution_transfer_independent_audit_2026-07-19.md)
- [独立 clone 完整 replay](psu_rotation40_resolution_transfer_replay_2026-07-19.md)
- [运行环境指纹](psu_rotation40_resolution_transfer_environment_2026-07-19.json)
- [公开包独立 validator](../site_tools/validate_psu_rotation40_resolution_transfer_public.py)
- [机器 summary](../demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/summary.json)
- [聚合 CSV](../demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/comparison_rows.csv)
- [诊断图 PNG](../demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/diagnostic.png)
- [诊断图 PDF](../demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/diagnostic.pdf)
- [公开边界 README](../demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/README.md)
- [checksums](../demo_t16_operator/results/psu_rotation40_resolution_transfer_public_v1/checksums.sha256)

本结果仍是公开数据上的一块真实观测证据，不是 OERF 组内实验结果。它最大的价值是把下一步从“继续加分辨率/加网络”收窄为“设计能在旋转留出上拒绝不稳定高频 correction 的多分辨率、数据一致算子”。
