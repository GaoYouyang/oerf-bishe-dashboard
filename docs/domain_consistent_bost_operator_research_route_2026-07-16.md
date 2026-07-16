# 域一致有限孔径 BOST 与神经算子研究路线

> 定位：何远哲 NeRIF / TDBOST 直接邻近主线
> 当前阶段：真实 PSU 几何门禁和有限孔径支持审计
> 论文状态：`METHOD HYPOTHESIS OPEN / SUPERIORITY LOCKED`

## 1. 真正可能有新意的问题

有限孔径 BOS、cone-ray forward、景深误差、NeRIF/NeDF/NRIP 和算子校正都已经有论文。因此不能把新意写成：

- 首次在 BOS 中考虑有限孔径；
- 首次使用 cone ray；
- 首次用 FNO/DeepONet 做三维重建；
- 首次用网络修正物理算子；
- 仅靠低秩、傅里叶 adapter 或张量分解即构成算法创新。

当前最可辩护的缺口是：

> **面向稀疏视角、有限孔径 BOST 的显式计算支持合同，并将域一致性、孔径越界、mask 选择和少量查询算子校正在 held-out camera 上闭环验证。**

它有两层：

1. **Domain-Consistent BOST（DC-BOST）**：A0/A1/B0/B1/B2/B3 定义、审计、mask 政策和 held-out 门禁；
2. **Domain-Conditioned Operator Correction（DCO）**：只在 DC-BOST 证明误差大于噪声/标定不确定度后，学习前向/伴随的少量校正，不直接猜三维场。

## 2. 当前真实证据已经说了什么

### A0 作者混合域

- 49,766,400 条真实中心线已全量普查；
- 0/3/6 视角的非零双锥段稳定落在 reconstruction box 外；
- pooled cone 路径域外比例 9.8976%；
- 判决：`NO-GO`。

### A1 作者兼容裁剪

- 将非零 cone 段裁到前向 box，保留双锥和 miss→box；
- 改变 1,879,113 条射线，移除作者混合域总路径 2.4282%；
- 判决：`AUTHOR_COMPATIBILITY_ABLATION_ONLY`。

### B0/B1 声明的计算域

- B0 = 前向归一化 ray ∩ box；
- B1 = B0 ∩ 归一化单叶 cone，miss 不回退；
- 端点 box、端点 cone 径向、中点 cone、长度子集和有限性不变量在九视角全部通过；
- B1 只保留 B0 总路径 15.1880%；
- B1 排除 view 0 的 1,350 条 active 中心线，它们正好是作者 cone miss 后 box fallback 集；
- 判决：声明的计算域被机械执行，物理域未验证。

### B2 有限孔径离散支持

QMC-8 / QMC-16 / QMC-32 都保留原固定样本分母：

- B0 active 支持保留率都是 100%；
- B1 active 支持保留率为 99.99465% / 99.99198% / 99.96442%；
- 三档之间的最大差只有 0.03022 个百分点，说明 active 测量的总孔径权重损失很小；
- 但 B1 active any-OOD ray 从 2,660、7,689 增至 99,617；
- 三组确定性低差异点并非嵌套设计，因此这些变化不是置信区间，也不能当成单调收敛证明；
- 判决：样本级 `1_D(x)` 可以继续作最小假设基线，但不能直接使用 `drop_any_out` 作默认 B3。

### B3 整射线选择政策

五种政策被预先写死后再统一读取 QMC-8/16/32 直方图：

| active B1 政策 | QMC-8 排除 | QMC-16 排除 | QMC-32 排除 | 当前解释 |
|---|---:|---:|---:|---|
| `indicator_keep` | 0 | 0 | 0 | 保留中心线命中；逐样本越界由 B2 置零 |
| `drop_empty` | 0 | 0 | 0 | active B1 没有整束孔径为空 |
| support floor 87.5% | 1,290 | 1,223 | 1,773 | 相对稳定的政策消融，不是已选阈值 |
| support floor 93.75% | 2,660 | 2,719 | 4,405 | 更严格消融，不是已选阈值 |
| `drop_any_out` | 2,660 | 7,689 | 99,617 | 被采样设计显著放大，不宜默认使用 |

这组结果没有替我们选择 B3。当前最小假设参考仍是 **B0 + fixed-denominator indicator**；`drop_empty` 只作数值清理。任何阈值、B1 支持域或整射线删除政策都必须由 held-out camera 重投影和 flow-off 不确定度决定。

## 3. B0–B3 与文献的精确关系

| 合同 | 最近一级文献 | 已知 | 我们仍可做的 |
|---|---|---|---|
| B0 | NeRIF；Photon renderer | 在 reconstruction volume 内截断路径已是惯例 | 把 miss/tangent/behind/boundary 写成可测合同并对真实数据审计 |
| B1 | PSU conical hull；localized gradient-index | support prior 可降低计算或限定场 | 单叶/no-fallback 没有被证明是物理定律，只能做消融 |
| B2 | Molnar finite-aperture BOS | 完整孔径平均和固定 Monte Carlo 分母有物理依据 | `1_D(x)` 相当于域外场置零，必须声明并做 held-out 对照 |
| B3 | 物体遮挡/model mask | 遮挡光线与模型区域可屏蔽 | 因部分孔径越计算域就丢整条 ray 没有文献保证，必须作政策消融 |

## 4. 可执行的自有算法候选

### 候选 M0：DC-NeRIF（低风险）

NeRIF/NRIP 的坐标场不变，替换 forward contract：

```text
ray domain = B0 or declared B1 ablation
aperture sample weight = 1_D(x_m)
normalization = original fixed M
mask policy = keep / empty-drop / predeclared support floor
loss = active camera + held-out camera audit + same regularization budget
```

新意不是 MLP，而是域合同、边界支持、mask 政策和尾部失败完整闭环。

### 候选 M1：DCO-ReSeSOp（中风险）

保留 B0/B2 的显式 `F/F*`，每次迭代学习：

- nominal residual；
- domain-support summary；
- camera/ray geometry metadata；
- 异方差 covariance 白化后的搜索方向；
- 可拒答的 operator-inexactness correction。

必须以同样 `F/F*` calls、wall time、正则和停止规则对比 learned ReSeSOp、PBB/CGLS 和 learned primal-dual。

### 候选 M2：RayKernel-DCO（高风险、有已有小信号）

在一个新 rig 上最多只允许 K 次 high-fidelity forward，用它们拟合逐 ray 局部 3D kernel/gates，严格共享 transpose 以保持伴随。

必须击败：

- 同预算最小范数/multisecant/Broyden；
- full-matrix ridge 和非神经局部多项式；
- Learned ReSeSOp；
- solver-plus-correction operator；
- direct FNO/F-FNO、DeepONet/NIO 和 GINO-style geometry encoder。

### 候选 M3：Domain-Set Operator（最高风险）

输入不是规则图像堆叠，而是可变长 ray set：

```text
{camera id, origin, direction, aperture basis/radius,
 domain support histogram, measured displacement, confidence}
    -> latent 3D field / NeRIF initialization
```

可用 permutation-invariant set encoder + geometry backprojection + FNO/INR decoder。可能的价值是跨 camera layout，不是在固定九视角上换一个更大网络。

## 5. 必须击败的方法矩阵

| 对手 | 为什么必须有 | 公平账本 |
|---|---|---|
| PBB / CGLS / SPG / TV | 强传统 inverse | calls、正则、stopping、wall time |
| NeRIF | 何远哲直接主线 | 同 ray sample、iterations、encoding |
| NeDF / NRIP | 静态 BOST 最近邻居 | 同 synthetic truth、noise、camera split |
| FNO / F-FNO | 直接算子对手 | 同参数、GPU hours、support views |
| DeepONet / NIO | function/operator inverse 对手 | 同 variable-view 输入合同 |
| Learned ReSeSOp | operator mismatch 最近概念碰撞 | 同 `F/F*` calls、同迭代次数 |
| differentiable calibration | 显式测试时几何修正 | 同 calibration parameters 和运算预算 |
| solver + neural correction | 防止用弱 warm-start 刷分 | 同 deterministic initialization 与总物理 calls |

## 6. 三层数据证据设计

### 层 1：BLASTNet 反应流 CFD truth

优先两组：

1. premixed H2-air slot flame：5 snapshots，651×401×201，约 11 GB；
2. vitiated H2-air flame case 3：25 snapshots，1152×128×128，约 28.5 GB。

用途：自建有限孔径多相机 BOST forward，保留完整时间、物理 case 和 camera layout 作 OOD，不随机拆同一个体场的 rays。

许可：CC BY-NC-SA 4.0；发布派生数据需保留归属和相同方式共享边界。

### 层 2：PSU Mach 4.8 真实 BOS

用 9-view 做 reconstruction/support，剩余公开 deflections 做 held-out image-space 验证。它没有独立 3D density truth，因此只能主报 held-out deflection/image consistency、重复性和标定不确定度。

许可风险：公开访问不等于有明确标准许可；不重分发原始/派生大文件，只发布代码、聚合评估与来源指针。

### 层 3：OERF/He 真实小样本

不等全数据库，只要：

- 一套 ray/calibration/mask/grid/unit；
- 一条 held-out camera 或 calibration phantom；
- flow-off repeats 用于 covariance；
- 如做 4D，则 50–200 帧 timestamp/缺帧/异步信息。

## 7. 稳妥可发布的门禁

### 前向门

- B0/B1/B2/B3 所有合同测试和 dot/adjoint 测试通过；
- QMC sample-count 敏感性完整报告；不把非嵌套设计误写成统计置信度；
- fixed denominator 与 survivor renormalization 做显式对照；
- 几何参数敏感性不得改写方法定义。

### held-out 门

- 真实/independent-renderer held-out camera 的 RMSE/MAE 改善；
- 分 freestream、shock/front、expansion、boundary 区域报告；
- 不仅报均值，还报 p90/p95 退化与 harm rate；
- 改善须大于 optical-flow/calibration 不确定度。

### inverse/operator 门

- field relative L2 / RMSE / SSIM / CC；
- front 位置、宽度与梯度误差；
- held-out ray/image residual；
- wall time、峰值内存、parameters、ray samples、`F/F*` calls；
- 跨场、跨时间、跨 camera layout 测试。

## 8. 算力规划

本机 M5 / 32 GB 足够：

- 全量几何和 B2/B3 支持审计；
- BLASTNet crop/downsample 预处理；
- 16³/32³ MPS 网络与小基线；
- 代码、测试、报告和图。

本机实测 QMC-8 / QMC-16 / QMC-32 分别约 88 / 131 / 226 秒，最大常驻内存均低于 0.8 GiB 且无 swap。GPU 租用只在 held-out 门给出正信号后触发。NeRIF 论文报告 RTX 4090 约 10 GB、稍超 20 min；geometry-aware 3D CBCT 对手使用 A100 80 GB 约 20 h。这些只是上下界参考，实际租卡必须依据我们自己 32³ profile 外推。

## 9. 一级文献与公开资源

### BOST 与有限孔径

- Molnar et al., *Forward and Inverse Modeling of Depth-of-Field Effects in Background-Oriented Schlieren*, AIAA Journal 2024: [DOI](https://doi.org/10.2514/1.J064095) · [arXiv](https://arxiv.org/abs/2402.15954)
- Molnar et al., *Open-Source BOS Tomography Dataset of High-Speed Flow Over a Flight Body*, Experiments in Fluids 2026: [DOI](https://doi.org/10.1007/s00348-026-04189-z) · [arXiv](https://arxiv.org/abs/2508.17120) · [dataset](https://doi.org/10.26208/1VE2-5C19)
- Rajendran et al., synthetic PIV/BOS renderer: [DOI](https://doi.org/10.1088/1361-6501/ab1ca8) · [Photon](https://github.com/lalitkrajendran/photon)
- He et al., NeRIF: [DOI](https://doi.org/10.1063/5.0250899) · [arXiv](https://arxiv.org/abs/2409.14722)
- NRIP 2026: [DOI](https://doi.org/10.1016/j.combustflame.2026.115082) · [arXiv](https://arxiv.org/abs/2605.11454)
- NeDF: [DOI](https://doi.org/10.1063/5.0241191) · [arXiv](https://arxiv.org/abs/2409.19971)
- TDBOST: [DOI](https://doi.org/10.1145/3809488) · [author repository](https://github.com/Hyz617/TDBOST)

### 算子和校正对手

- Neural Inverse Operators: [PMLR](https://proceedings.mlr.press/v202/molinaro23a.html) · [code](https://github.com/mroberto166/nio)
- GINO: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) · [neuraloperator](https://github.com/NeuralOperator/neuraloperator)
- Geometry-Aware Attenuation Learning: [DOI](https://doi.org/10.1109/TMI.2024.3473970) · [code/data](https://github.com/ShanghaiTech-IMPACT/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction)
- Learned ReSeSOp: [DOI](https://doi.org/10.1088/1361-6420/adef73) · [arXiv](https://arxiv.org/abs/2410.23061)
- Neural Correction Operator: [DOI](https://doi.org/10.1016/j.jcp.2026.115039) · [code](https://github.com/amitbhat31/neural-correction-operator)
- Differentiable geometry calibration: [2023 DOI](https://doi.org/10.1088/1361-6560/acf90e) · [code](https://github.com/mareikethies/geometry_gradients_CT) · [2026 ray-based preprint](https://arxiv.org/abs/2606.21405)
- F-FNO: [ICLR](https://iclr.cc/virtual/2023/poster/10680) · [code](https://github.com/alasdairtran/fourierflow)

### 反应流数据

- BLASTNet premixed H2-air slot flame: [dataset page](https://blastnet.github.io/premixed_slot_flame_h2air)
- BLASTNet vitiated H2-air flame: [dataset page](https://blastnet.github.io/vitiated_h2air_flame)
- BLASTNet license/index: [datasets](https://blastnet.github.io/datasets)
- Momentum128: [dataset page](https://blastnet.github.io/blastnet_momentum)
- ScalarFlow（只作时序/相机协议检查，不是 BOST 物理证据）: [project](https://ge.in.tum.de/publications/2019-scalarflow-eckert/) · [data](https://mediatum.ub.tum.de/1521788)

## 10. 当前最需师兄回答的四句话

1. 组内的 cone axis/vertex/angle 是物理支持域还是计算裁剪？axis 的符号是否有意义？
2. cone miss 后 box fallback 是有意的混合域，还是为避免零长度？
3. 有限孔径样本超出 reconstruction domain 时，组内希望置零、丢 ray、扩大域，还是用其他场模型？
4. 能否给一条 held-out camera/phantom 和 flow-off repeats，让域合同的好坏不依赖 3D truth 也能判决？

这四个答案会决定毕设是 DC-NeRIF 稳妥线，还是继续做 RayKernel-DCO 的高风险算子创新。

## 11. 当前公开复核入口

- [B2 QMC-8/16/32 严格公开摘要](psu_aperture_sensitivity_public_summary.json)
- [B3 五政策严格公开摘要](psu_b3_policy_public_summary.json)
- [B2/B3 四联图 PNG](../demo_t16_operator/results/psu_b3_policy_audit/psu_b3_policy_sensitivity_figure.png)
- [B2/B3 四联图 PDF](../demo_t16_operator/results/psu_b3_policy_audit/psu_b3_policy_sensitivity_figure.pdf)

公开文件不含私有路径、逐射线索引、原始样本点、运行时 provenance 或数据散列。它们证明离散支持与政策敏感性，不证明 continuous-aperture containment、重建改善或算法胜出。

## 12. B1 参数敏感性把研究问题进一步收窄

12 组预声明真实九视角审计已经排除一个危险假设：B1 的 axis/angle/vertex 不能被当成无关紧要的 renderer 细节。

- axis 反号后 active hit 和 support IoU 都为 0；
- 15/20 度只保留 48.7807% / 84.8418% active hits；
- 30/35 度几乎保留全部 hit，但 support IoU 仍只有 73.1381% / 57.4330%；
- 六个 5 mm vertex stress 的 active support IoU 为 89.3126% 到 93.0639%。

这使候选 M0/M1 的输入和消融更明确：任何 geometry-conditioned 方法都必须看到 axis、vertex、angle、support length 和 boundary distance；任何“通用算子”主张都必须跨未见 rotation/camera，而不能只在同一九视角上换网络。

同时它也提高了创新门槛。不能把“把 cone 参数放进 MLP”写成创新；真正可发表的是：

1. 参数语义由独立物理信息或 development run 冻结；
2. forward/adjoint 共用同一 domain contract；
3. final audit 不参与参数选择；
4. 改善超过 flow-off 与 calibration uncertainty；
5. 六个 rotation blocks 全部同向，不靠像素伪重复。

## 13. 70 视角协议让算法路线变成可判决实验

完整 70 视角现在不是随机 train/test split，而是五层用途：

| 层 | 视角数 | 可以做什么 |
|---|---:|---|
| support | 9 | reconstruction、训练、support residual |
| development | 7 | 一次性模型选择、停止和 practical margin |
| primary audit | 18 | 六个同相机未见旋转块的唯一主检验 |
| camera audit | 12 | 相机布局外推 |
| joint audit | 24 | 相机和旋转联合外推压力测试 |

主算法首轮只允许比较一个冻结候选和 B0 reference。primary gate 是六个未见 rotation block 的 active vector relative L2 全部降低；随后还要过 repeatability、p95、ambient 和 calibration 四门。评分器会拒绝重复视角、哈希漂移、路径逃逸、mask 重叠和不完整 18 视角网格。

这条协议给 M0-M3 排了现实优先级：

1. 先做 B0-reference DC-NeRIF/PBB 小规模重建接口；
2. 若 development residual 没超过重复性地板，停止学习 operator correction；
3. 若存在可重复结构失配，再比较 RayKernel-DCO、Learned ReSeSOp 和非神经少查询基线；
4. 只有 64³ 以上多模型多种子阶段才租 CUDA。

## 14. 当前最稳妥的毕设和论文双层目标

### 毕设保底成果

- 真实 PSU A0-B3 与 B1 参数敏感性完整复核；
- 70 视角防泄漏协议和可运行评分器；
- B0/PBB 或低分辨率 NeRIF 的 support reconstruction；
- development rotation 40 的重投影和 repeatability 估计；
- 无论正负都完整报告六块 final audit。

这套成果即使算法没有胜出，也有明确物理发现、软件产物、评测方法和负结果价值。

### 论文升级条件

- 一个 BOST-specific 候选相对强 B0/PBB/NeRIF 邻近基线，在六个 primary blocks 全胜；
- RMSE 降幅超过 flow-off floor，p95 和 ambient 不恶化；
- calibration perturbation 后仍成立；
- camera/joint audit 说明泛化边界；
- synthetic/CFD truth 上补 field-L2、front location 和 inverse stability；
- 参数、calls、wall time、峰值内存和失败 case 全部公开记账。

未满足这些条件时，论文应定位为 forward-domain diagnosis / benchmark protocol，而不是新算子优越性。

新增入口：

- [B1 参数敏感性与 70 视角协议](psu_b1_parameter_sensitivity_and_heldout_protocol_2026-07-16.md)
- [B1 参数敏感性图](../demo_t16_operator/results/psu_b1_parameter_sensitivity/psu_b1_parameter_sensitivity_figure.png)
- [70 视角协议摘要](psu_heldout_camera_protocol_public_summary.json)
- [最终评分器](../site_tools/score_psu_heldout_reprojection.py)

## 15. B0 重建接口先冻结，再进入真实 inverse

现在已有独立于作者 TensorFlow 入口的低分辨率线性参考：

\[
\hat d_r = \frac{L_r C_r}{M}\sum_m P_r I_{p_{r,m}} D x.
\]

forward 与 adjoint 共用有限差分、三线性插值、投影和固定分母原语；边界 gauge 显式处理 BOS 的加性常数零空间。tiny materialization、dot-product、固定分母和调用记账均已有测试。

在真实九视角 support geometry 的 2,304 条确定性 active 子集上：

- 16³/32³ CPU float64 dot defect 为 `4.97e-16 / 1.78e-16`；
- MPS float32 为 `7.28e-8 / 9.89e-8`；
- 没有读取位移值做训练或选射线；
- rotation 40 与 final audit 均未打开。

合成 12³ Landweber 把 measurement relative L2 降到 `0.005028`，field-L2 仍是 `0.4504`。这使论文叙事必须区分：

1. 离散接口闭合；
2. support projection fit；
3. held-out image-space consistency；
4. synthetic/CFD truth field accuracy；
5. experimental 3D truth，不可从 PSU 数据直接获得。

接口冻结时写下的优先级是：

1. 全 support streaming B0 `F/F*`；
2. 16³ Landweber/PBB/CGLS 同 calls 基线；
3. gauge、Tikhonov/TV 和停止规则；
4. 冻结 baseline hash 后才打开 rotation 40；
5. development 若没有超过 repeatability floor 的结构残差，停止 DCO；
6. 有结构残差才比较 DC-NeRIF、DCO-ReSeSOp 与 RayKernel-DCO。

新增入口：

- [B0 重建接口门禁](psu_b0_reconstruction_interface_gate_2026-07-16.md)
- [B0 接口公开摘要](psu_b0_reconstruction_interface_public_summary.json)
- [B0 接口四联图](../demo_t16_operator/results/psu_b0_interface_audit/psu_b0_interface_audit_figure.png)

## 16. 全 support inverse 把主线推进到 32³ reference

九个 support views 的 `10,628,822` 条 active rays 已经全部进入流式 B0 operator。QMC-16 下，每个逻辑 `A` 或 `Aᵀ` 约处理 1.70 亿有限孔径 samples，329 个 chunks 合计只记一次调用。

全规模 precision gate 排除了 float32 作为权威 classical baseline：

- float32 observation-dual defect `8.49e-4`；
- float32 random-dual defect `2.04e-5`，仍略高于 `2e-5`；
- float64 random-dual defect `3.46e-15`。

在固定 4 步、相同 `4F+5Aᵀ` 下：

| 网格 | direct support relative L2 | pair time | max RSS |
|---|---:|---:|---:|
| 16³ | `0.7877107` | `53.43 s` | `5.344 GiB` |
| 32³ | `0.6271325` | `50.55 s` | `5.349 GiB` |

32³ 的绝对下降 `0.1605782` 通过预先冻结的 `0.02` practical signal；九个 support views 全部改善。因此 32³ 成为正式低分辨率 reference。

这改变了算法优先级：

1. 先冻结 32³ support checkpoint/config/hash；
2. rotation 40 只检验迁移，不再从 support curve 选迭代；
3. classical 邻近基线增加 Tikhonov-CGLS、TV/primal-dual；
4. learned candidate 优先做 Krylov preconditioner，而不是直接生成最终体；
5. 全规模 float32 负结果生成一个独立候选：adjoint-safe compressed streaming；
6. 只有 development residual 超过 repeatability floor，才训练 ray-conditioned correction。

目前 Mac 无需服务器。32³ 完整 pair 约 51 秒、RSS 约 5.35 GiB；真正需要 CUDA 的是后续数千次全遍历、多模型多种子，而不是当前 reference。

新增入口：

- [全 support CGLS 与分辨率门禁](psu_b0_full_support_cgls_and_resolution_gate_2026-07-16.md)
- [16³ 摘要](psu_b0_streaming_baseline_public_summary.json)
- [32³ 摘要](psu_b0_streaming_32cubed_public_summary.json)
- [同 calls 分辨率摘要](psu_b0_streaming_resolution_public_summary.json)
- [precision gate](psu_b0_streaming_precision_public_summary.json)
- [16³/32³ 分辨率图](../demo_t16_operator/results/psu_b0_streaming_resolution/psu_b0_streaming_resolution_figure.png)

## 17. 快速 exact physics layer 已经可用

32³/QMC16 的固定 stencil 已写入 5.017 GB 私有 cache。缓存前后 forward/adjoint relative difference 都是 `0.0`；same-session 完整 pair 从 `37.92 s` 降到 `17.04 s`，加速 `2.225×`。

完整 4-step CGLS 回放得到：

- support residual absolute difference `0.0`；
- volume relative difference `1.17e-16`；
- optimization `218.03 s → 74.95 s`；
- speedup `2.909×`。

这使后续研究顺序更明确：

1. cached 32³ B0 是 exact physics layer；
2. 先补 Tikhonov-CGLS、TV/primal-dual、Jacobi/PBB 邻近强基线；
3. learned module 优先作为 Krylov preconditioner，不直接替代重建；
4. 所有候选按相同完整 `A/Aᵀ` calls、wall time 和内存记账；
5. rotation 40 只做一次 development 迁移判断；
6. 没有超过 flow-off floor 的结构残差就不启动 correction。

缓存只提高研究吞吐，不提高物理证据等级。它不能把 support fit 变成 held-out generalization 或实验三维真值。

新增入口：

- [紧凑缓存与快速参考门禁](psu_b0_compact_cache_and_fast_reference_gate_2026-07-16.md)
- [缓存 benchmark](psu_b0_compact_cache_public_summary.json)
- [缓存 CGLS 对照](psu_b0_cached_reference_public_summary.json)

## 18. Learned Krylov 首轮筛选：强 Sobolev 基线改变了创新门槛

在真实 PSU support 几何上完成了首轮固定四步 positive-spectral preconditioner 筛选。观测由 QMC-32 B0 有限孔径算子生成，重建使用 QMC-8 名义算子；解析 reaction-morphology fields 提供 field/gradient/front 指标。它们不是 CFD 或实验三维真值。

验证集只用于从 `p=0,...,6` 中选择 inverse-Sobolev 方向，最优为 `p=5`。随后 2,227 参数 learned multiplier 从该方向精确零初始化，网络只能作用于精确伴随梯度，且每步使用解析线搜索。候选与 Sobolev 都是 `4F+4Aᵀ`，调用预算相同。

三种子得到：

| split | learned vs Sobolev mean field gain | tail/risk |
|---|---:|---|
| IID | `+4.26%` 到 `+4.62%` | p10 全部 `>+2%`，无 `>1% harm` |
| noise OOD | `+4.28%` 到 `+4.46%` | 无 `>1% harm` |
| 4–5-view OOD | `+1.41%` 到 `+1.77%` | 少量尾部退化 |
| family OOD | `+0.67%` 到 `+0.88%` | p10 为负 |
| joint OOD | `-0.43%` 到 `-0.20%` | p10 约 `-4.5%`；harm `33.3%` |

因此当前无门控候选为 `NO-GO`，通过种子 `0/3`。这次结果改变后续方法路线：

1. 任何 FNO、DeepONet、learned Krylov 或 neural preconditioner 都必须先击败 validation-selected inverse-Sobolev，而不能只比较四步 CGLS；
2. 数据项单调下降不能替代 paired field、tail harm 和 held-out residual；
3. learned direction 必须有显式 baseline fallback，不能在分布外继续满强度外推；
4. 当前 opened splits 降级为 development，fresh gate 必须换形态、相关噪声、视角数和种子；
5. PSU 没有实验三维真值，真实阶段仍以 held-out reprojection、repeatability 和 calibration perturbation 为主门。

下一候选是：

\[
P_{\theta,\tau}
=P_{\mathrm{Sobolev}}
+\tau(z)(P_\theta-P_{\mathrm{Sobolev}}),
\quad 0\le\tau\le1.
\]

`τ=0` 必须逐值等于 Sobolev；训练加入 view dropout、thin-front stress、相关噪声和相对 baseline residual-risk penalty。若 fresh joint OOD 只能靠全回退达到零增益，则它只能作为 selective safety mechanism，不能声称通用重建优越性。

新增入口：

- [首轮谱预条件器 NO-GO](psu_b0_spectral_preconditioner_no_go_2026-07-16.md)
- [严格公开摘要](psu_b0_spectral_preconditioner_pilot_public_summary.json)
- [论文级四联图](../demo_t16_operator/results/psu_b0_spectral_preconditioner_pilot/psu_b0_spectral_preconditioner_pilot_figure.png)
- [冻结配置](../demo_t16_operator/configs/psu_b0_spectral_preconditioner_pilot_v1.json)

## 19. Post-open 支持域诊断：exact fallback 不是泛化

用原三种子 checkpoint 在已打开 split 上增加固定 active-view envelope：

\[
P_{\mathrm{env}}=
\begin{cases}
P_\theta,&6\le N_{\mathrm{view}}\le9,\\
P_{\mathrm{Sobolev}},&\text{otherwise}.
\end{cases}
\]

它不重训、不扫阈值，也不增加 `F/Aᵀ` calls。支持域内逐值选择原 learned direction，支持域外逐值选择 Sobolev。

结果：

| split | coverage | 结果 |
|---|---:|---|
| IID / noise / exact | 100% | 保留原 learned 正信号 |
| family OOD | 100% | p10 与 20.8%–25% harm 原样保留 |
| view OOD | 0% | 回到 Sobolev，harm 归零 |
| joint OOD | 0% | 三种子 33.3% harm 归零，但 gain 也约为 0 |

这证明 active-view count 是一个有效拒答触发量，却不能识别训练视角范围内的未见形态。下一轮方法的最低结构因此变成：

```text
hard support envelope
  × learned residual-risk trust
  × bounded positive spectral correction
  + exact Sobolev fallback
  + analytic line search through exact A/A^T
```

fresh protocol 必须同时约束：

1. correction coverage 不能靠全回退变为 0；
2. accepted samples 的 paired p10 不为负；
3. total `>1% harm ≤5%`；
4. fresh family/joint mean gain 不退化；
5. IID 至少保留 2%；
6. 同 calls、wall time、MPS/CUDA memory 完整记账。

当前诊断只生成下一轮假设，不改变首轮 `0/3 NO-GO`。

新增入口：

- [视角支持域回退诊断](psu_b0_support_envelope_postopen_diagnosis_2026-07-16.md)
- [严格公开摘要](psu_b0_support_envelope_postopen_public_summary.json)
- [机制四联图](../demo_t16_operator/results/psu_b0_support_envelope_postopen/psu_b0_support_envelope_postopen_figure.png)

## 20. OCRRG fresh：从“学习器总是出手”推进到选择性物理回退

> **Post-open contract update：**后续 504-row 特征复算发现 development /
> calibration 在 support 投影后提取方向特征，deployment 却在投影前提取。
> 两种顺序改变 7/504 条 accept/fallback 决策，最大 prediction shift 为
> 0.826 个百分点。因此下述经验 fresh 指标仍保留，但 split-conformal
> 解释撤回；v2 必须先统一 canonical feature function。

冻结候选将 hard support envelope、16 维可观测 residual-risk 特征、
per-seed ridge gain predictor 和 one-sided split-conformal overprediction
quantile 合并。风险门在第一步做一次固定决策，不先跑完 candidate 与
Sobolev 两条路径；部署仍保持 `4F+4Aᵀ`。

预注册配置与 checkpoint 哈希先在提交 `cd5d4a0` 中冻结并推送，随后一次性
打开七组 fresh split。三种模型种子全部通过本轮 synthetic-only 候选门。
在五个 6 至 9 视角 support split 上，三种子平均 coverage 为 26.4% 至
47.2%，平均 field gain 为 +1.04% 至 +1.66%；在两个 3 至 5 视角 split 上
coverage 精确为 0，并在 `1.2e-7` 数值尺度内回退 Sobolev。

但判决不能写成“安全”或“优于现有模型”。独立验证仍发现 4 条 accepted
`>1% harm`，集中于一个 6-view plume 和一个强相关噪声的 6-view
oblique shock。conformal 校准来自当前解析生成器，不能为任意 OOD、PSU
实验值或 OERF 装置提供覆盖保证。

因此下一阶段的方法问题已经进一步收窄：

```text
exact support envelope
  × group-conditional observable risk lower bound
  × bounded positive learned spectral correction
  + exact validation-selected Sobolev fallback
```

优先实验：

1. 新 seeds 独立重复，不复用当前 fresh 做阈值选择；
2. leave-one-morphology-family-out 风险训练；
3. 6-view / 7-view / 8-view 分组风险与 worst-case calibration；
4. 使用真实 flow-off repeats 替换合成相关噪声；
5. 同 calls 比较 CGLS、Tikhonov、TV、Sobolev、FCG-NO 和代表性
   DeepONet/FNO；
6. 真实噪声与 calibration perturbation 过门后，才打开 rotation 40。

新增入口：

- [OCRRG fresh 判决](psu_b0_residual_risk_fresh_result_2026-07-16.md)
- [OCRRG post-open 特征契约诊断](psu_b0_residual_risk_postopen_diagnosis_2026-07-16.md)
- [OCRRG 公开摘要](psu_b0_residual_risk_fresh_public_summary.json)
- [post-open 公开诊断摘要](psu_b0_residual_risk_postopen_diagnosis_public_summary.json)
- [OCRRG 论文图](../demo_t16_operator/results/psu_b0_residual_risk_fresh/psu_b0_residual_risk_fresh_figure.png)
- [post-open 四联图](../demo_t16_operator/results/psu_b0_residual_risk_postopen_diagnosis/psu_b0_residual_risk_postopen_diagnosis_figure.png)

## 21. 强 PCGLS 改写主线：从单步方向转向固定 SPD 预条件器

补齐 L2/H1 Tikhonov、普通 CGLS、各向异性 Sobolev、分阶段 Sobolev 和
Sobolev-PCGLS 后，当前 learned spectral steepest direction 被同预算
PCGLS-4 淘汰。

PCGLS-4 只在 `risk_validation` 选择 `strength=4, epsilon=0.05`，随后在未
参与选择的 `risk_calibration` 相对三个 raw learned seed 的逐场均值降低
field error 4.94%；validation 降低 5.00%。固定四步最后不计算未使用的
`AT r4`，所以真实预算是 `4F+4AT`，与学习方法一致。

这使旧 OCRRG / Multi-Veto 路线失去主算法对象：即使门控完全修好，也只是在
保护一个已经弱于 PCGLS 的候选。它们继续保留为风险诊断资产，不再投入新
fresh repeat。

下一主线冻结为 `BOST-GC-SPD-PCGLS`：

1. 置换不变编码 active camera set、几何、噪声和初始白化 residual；
2. 只输出七个低维频谱 basis 系数；
3. 生成有界正 multiplier，并做几何均值归一化；
4. multiplier 在四步 PCGLS 内保持固定；
5. 零初始化逐值等于 validation-selected PCGLS-4；
6. validation 与 calibration 都过门后，才生成新的 independent repeat。

若预条件器随 residual 或 stage 变化，必须切换到 Flexible CG 并加入显式
方向正交化；不能继续套用标准 PCG。

新增入口：

- [PCGLS 强基线 no-go](psu_b0_pcgls_no_go_2026-07-16.md)
- [BOST-GC-SPD-PCGLS 研究合同](bost_gc_spd_pcgls_research_contract_2026-07-16.md)
- [强频谱开发摘要](psu_b0_strong_spectral_frontier_development_public_summary.json)
- [PCGLS 四联图](../demo_t16_operator/results/psu_b0_pcgls_no_go/psu_b0_pcgls_no_go_figure.png)

## 22. BOST-GC-SPD-PCGLS V1 证伪：下一步先测 conditional headroom

首个 2,527 参数低维条件化预条件器已经按上述合同实现。零初始化、固定 SPD、
正 multiplier、几何均值归一化和 `4F+4AT` 调用账本全部通过；训练只使用
`risk_train`，checkpoint 只由 `risk_validation` 选择，`risk_calibration`
只读取一次，opened fresh 完全未加载。

科学结果为 `0/3 NO-GO`。三种子 validation mean field gain 为 +0.016% 至
+0.054%，calibration 为 -0.165% 至 +0.056%，所有 bootstrap 下界均小于
0。与此同时，measurement residual 平均改善 +0.596% 至 +1.709%，再次给出
projection fit 与 3D truth 解耦的直接证据。

因此研究时序改为：

```text
finite PCGLS family
  -> truth-oracle / family-oracle conditional headroom
  -> observable view/noise strata transfer
  -> only then decide selector, TV, stopping, or Flexible CG
```

在 headroom 审计前，不增加网络宽度，不生成新 fresh，不削弱 PCGLS-4
baseline。完整入口：

- [V1 开发 NO-GO](psu_b0_conditioned_pcgls_development_no_go_2026-07-16.md)
- [严格公开摘要](psu_b0_conditioned_pcgls_development_public_summary.json)
- [冻结配置](../demo_t16_operator/configs/psu_b0_conditioned_pcgls_development_v1.json)
- [四联图](../demo_t16_operator/results/psu_b0_conditioned_pcgls_development/psu_b0_conditioned_pcgls_development_figure.png)

## 23. Conditional headroom 审计：频谱家族可用，简单条件不可用

V1 no-go 后没有扩网络，而是冻结 105 个固定 SPD-PCGLS 候选，保持统一
`4F+4AT`，逐层测试：

```text
one global expert
  -> view-count expert
  -> view-count + noise expert
  -> nondeployable morphology expert
  -> per-sample truth oracle
```

validation / calibration 的 paired field gain 分别为：

| 路由 | Validation | Calibration | 可部署 |
|---|---:|---:|---|
| 全局 | +0.352% | -0.216% | 是 |
| view count | +0.759% | -0.260% | 是 |
| view + noise | -0.106% | -5.647% | 是 |
| synthetic family label | +2.689% | +2.378% | 否 |
| sample truth oracle | +6.522% | +7.218% | 否 |

这形成一个关键机制证据：有限 SPD 频谱族有足够上限，但相机数量与噪声强度
不能跨 split 识别正确专家。下一学习对象必须是由测量诱导的形态/风险表征，
而不是更大的 geometry-only MLP。

新增入口：

- [conditional headroom 报告](psu_b0_pcgls_conditional_headroom_2026-07-16.md)
- [公开摘要](psu_b0_pcgls_conditional_headroom_public_summary.json)
- [四联图](../demo_t16_operator/results/psu_b0_pcgls_conditional_headroom/psu_b0_pcgls_conditional_headroom_figure.png)

## 24. OGSE-PCGLS V2：可观测收益路由接近门槛，但不能解锁 fresh

当前算法主线已经从 geometry-conditioned coefficient MLP 改为
`Observable Gain-regressed Spectral Experts`：

```text
y
  -> shared first normal field g0 = A^T W y
  -> 44 observable spectral/spatial features
  -> per-expert gain regressor
  -> confidence + baseline-top exact fallback
  -> fixed positive log-space expert mixture
  -> four-stage PCGLS
```

专家只在 `risk_train` 上贪心选择；selector 只在 train OOF 屏选；validation
与 calibration 都是 post-open development diagnostic；opened fresh 未加载。

修正“基线 top-1 仍被混合”的实现错误后，严格路线为：

| 指标 | Validation | Calibration |
|---|---:|---:|
| mean field gain | +2.423% | +1.651% |
| bootstrap 95% lower | +1.237% | +0.700% |
| p10 | 0 | 约 0 |
| `>1% harm` | 0% | 0% |

因此六项开发门通过 5 项，只因 calibration mean 未达到 +2% 而判
`NO-GO`。诊断路线虽有 +3.560% / +2.554%，却出现 -12.669% /
-7.369% 最坏退化，不能用于论文主张。

下一候选冻结为 `Risk-Quantile OGSE`，优先级如下：

1. 用 quantile ridge / conformal residual 预测逐专家收益下分位，而非只预测
   均值；
2. 新增 grouped whitened-residual spectrum、角向 backprojection
   imbalance、first-step contraction 与 search-direction angle；
3. 从全专家 softmax 混合改为 baseline 到单专家的一维有界插值；
4. 同时跑 TV-superiorized PCGLS、learned stopping 和 UNO-CG/NeuralIF
   可实现规模对照；
5. development 两层全过后才生成独立 repeat；当前 fresh 继续封存。

新增入口：

- [OGSE V2 NO-GO](psu_b0_ogse_pcgls_development_no_go_2026-07-16.md)
- [严格公开摘要](psu_b0_ogse_pcgls_development_public_summary.json)
- [算法图](../demo_t16_operator/results/psu_b0_ogse_pcgls_development/psu_b0_ogse_pcgls_development_figure.png)

## 25. RQ-OGSE 与多目标风险审计：下一表示必须保留 per-view 冲突

RQ0/RQ1 已按上一节实现：

```text
44 pooled g0 features
  -> mean / lower quantile / field-harm heads
  -> exact baseline or one bounded expert path
  -> fixed four-stage PCGLS
```

有限动作缓存把 648 个路由配置的最新开发复跑缩短到约 11.0 秒。mean-only
严格路线在 validation/calibration 的 field gain 为 `+3.321% / +2.907%`，
主 field/risk 八项门通过；但 calibration 有一个 `-1.897%` field harm，
front-F1 均值为 `-0.261%`，最坏 shock front 为 `-30.876%`。

quantile/harm 联合路线把 field harm 清零，但 field gain 降到
`+1.979% / +1.777%`。再加入 front lower-quantile 与 front-harm 头后，
多目标路线为：

| Split | Field gain | Front mean | Field harm |
|---|---:|---:|---:|
| validation | +1.192% | +0.375% | 0% |
| calibration | +1.382% | -0.060% | 0% |

因此当前总体判决为 `HOLD`。它不是简单的风险阈值问题，而是表示缺失：
所有相机的 adjoint contribution 被先求和，camera disagreement 与局部相关
噪声被抹掉。

下一候选改为 `View-Decomposed Risk Router`：

1. 暴露每视角 `g0,v = A_v^T W_v y_v`；
2. 提取每视角白化频谱、高频比例、norm share 与两两 cosine；
3. 用 permutation-invariant set encoder 聚合可变相机集合；
4. 分开预测 field utility 与 front-preservation risk；
5. 动作仍限制为 exact baseline 或一个 bounded SPD expert；
6. leave-one-family-out 与 leave-one-noise-profile-out 同时过门后，才冻结
   新 independent repeat。

必须先确认 per-view contribution 是否能在当前 adjoint 内部免费暴露。若需
额外 `AT`，调用预算必须重新匹配。

另外，first-step contraction 不能用于第一步之前的固定 preconditioner。
若使用第一步动态，只能：

- baseline probe one step + restart/FCG；
- 或增加调用并显式记账。

新增入口：

- [RQ-OGSE HOLD 判决](psu_b0_rq_ogse_pcgls_development_hold_2026-07-17.md)
- [RQ 论文工作草稿](rq_ogse_manuscript_working_draft_2026-07-17.md)
- [RQ/front 四联图](../demo_t16_operator/results/psu_b0_rq_ogse_pcgls_development/psu_b0_rq_ogse_pcgls_development_figure.png)
