# 何远哲主线论文深挖：从论文到毕设执行

生成日期：2026-07-10

这份文件只服务一个问题：何远哲近年的 NeRIF、PIV-BOST、4D BOST 三条线，如何拆成本科毕设可以执行的实验、章节和沟通问题。

## 总链条

```text
BOS 图像位移
  -> BOST 多视角层析
  -> NeRIF 连续折射率场重构
  -> PIV-BOST 折射补偿 / 4D BOST 时序重构
  -> 数据融合实验流体力学
```

你的最佳站位不是“泛 AI”，而是这条链中的中段：把光学观测变成连续三维/四维物理场，并给出可复现的误差分析。

## 2026-07-10 补记：把论文深挖接到当前论文库

当前公开论文库已扩展到 561 篇入口，其中 192 篇开放 PDF 已通过解析/来源校验，并生成 192 个单篇术语/页码导读。4D BOST 已核 ACM OA / CC BY 4.0 HTML 与 19 页 eReader 正式全文，并完成方法、实验、消融、显存、代码边界和 M3B 对照精读。根据师兄最新提出的“三维重建 + 算子学习”，工作重心已转为 T16 physics-lift residual FNO；NeRIF/BOST 继续提供 forward physics、per-instance baseline 和 refinement，PIV-BOST/完整 4D BOST 暂列升级线。

新增网页入口：

- `he_yuanzhe_sync_dashboard.html`：把何远哲主线、论文矩阵、三个月计划、VPN 私有库边界和师兄问题收成一个页面。
- `he_bost_citation_spine.html`：把 BOS 基础、UBOST、Direct BOST-RBF、NeRIF、NeDF/NRIP、PIV-BOST 和 4D BOST 压成可向师兄解释的引用脊柱。
- `he_bost_14day_reading_plan.html`：把引用脊柱转成 14 天精读执行包，每天固定输出公式、图表、demo/动作和要问师兄的问题。
- `tdbost_reproducibility_audit.html`：4D BOST 正式全文的可视化精读页，含 MM/DC/forward model、实验数字、显存、paper-vs-code 对账、M3B 图和交互 rank 实验。
- `fulltext_access_queue.html`：跟踪 PIV-BOST、stereo PIV-BOST 等受限全文，以及 4D BOST“在线已读、raw PDF 未缓存”的状态；订阅 PDF 本体不上传 GitHub。
- `xmu_vpn_private_library_protocol.html`：记录 WebVPN/私有库边界。命令行直连 Springer PDF 会返回 `/fingerprint` 验证页，因此无人值守时不强行下载，白天由用户过验证后放入 `private_library/inbox/` 再自动匹配。

## Paper 1: NeRIF, Physics of Fluids 2025

正式信息：

- Title: Neural refractive index field: Unlocking the potential of background-oriented schlieren tomography in volumetric flow visualization
- Authors: Yuanzhe He, Yutao Zheng, Shijie Xu, Chang Liu, Di Peng, Yingzheng Liu, Weiwei Cai
- DOI: `10.1063/5.0250899`
- 读论文入口：`https://arxiv.org/html/2409.14722v2`

### 它解决什么

传统 BOST 往往把三维场离散成体素，再建一个很大的投影/微分矩阵。问题是：

- 体素越细，矩阵和内存急剧膨胀。
- 体素内部常值假设带来离散化误差。
- 少视角和噪声条件下容易有伪影。
- 空间分辨率被体素网格限制。

NeRIF 的核心是用一个坐标神经网络表示连续折射率场：输入空间坐标，输出该点折射率和梯度，再通过光线积分预测背景图像位移。

### 方法骨架

- 输入：三维坐标点。
- 输出：折射率 `n` 和折射率梯度。
- 观测：多视角 BOS/BOST 图像位移。
- 前向模型：沿相机光线采样，积分折射率梯度，得到背景板上的位移。
- 损失：让网络输出的折射率、折射率梯度和观测位移互相一致。
- 稳定策略：随机光线采样、自动微分与数值微分结合、重投影验证。

论文中给出的具体实现线索：

- 网络为 7 层全连接结构，每层 256 个神经元，第 4 层有 skip connection。
- 输出分成多个 head，用于折射率和梯度。
- 采用 sinusoidal frequency encoding。
- 采样点数随迭代从约 60 增到 200。
- 数值验证用了 9 个视角、200 x 200 像素投影、50 mm 级重建体积，并加入高斯噪声。
- 真实实验用了单相机九输入端光纤束系统，高速相机 1 kHz，DeepFlow 估计位移。
- 指标包括 L2 error、SSIM、correlation coefficient、PSNR 和重投影一致性。

### 你能复现什么

第一周可做：

- 生成一个三维 Gaussian blob 或火焰薄层 phantom。
- 用简化直线光线模型生成 3/5/7/9 视角位移。
- 做一个体素 baseline，不要求完整矩阵，只要能重建粗场。
- 用小 MLP 表示 `n(x,y,z)`，最初可以不预测梯度 head。
- 输出中心切片、重投影误差和 L2。

一个月内可做：

- 加入梯度一致性 loss。
- 加入视角数和噪声扫描。
- 加入 Fourier encoding / hash encoding 对照。
- 和 Tikhonov/Landweber/简单体素法比较。

毕设可写章节：

1. BOST 物理模型和 Gladstone-Dale 链路。
2. 体素层析反演的病态性与计算负担。
3. NeRIF 简化实现。
4. 合成数据与鲁棒性实验。
5. 真实数据接口或开源数据迁移。

要问何远哲：

- 组内 NeRIF 的真实数据格式是什么：原始图、位移场、标定、重构结果分别怎么存？
- 本科阶段是否允许看组内代码？如果不能，baseline 应该对齐哪篇论文？
- 真实数据是否能提供 1-3 帧作为接口验证？
- 论文中哪些图能作为本科论文对照，哪些不适合复刻？

## Paper 2: PIV-BOST, Experiments in Fluids 2025

正式信息：

- Title: Instantaneous refractive index compensation on the velocity measurement using simultaneous PIV-BOST
- Authors include Yutao Zheng, Yuanzhe He, Shijie Xu, Yingzheng Liu, Weiwei Cai
- DOI: `10.1007/s00348-025-04093-y`
- Springer 页面：`https://link.springer.com/article/10.1007/s00348-025-04093-y`

### 它解决什么

PIV 测速度时，粒子图像通过热羽流/火焰折射率梯度传播到相机。折射率不均匀会造成粒子像素位置偏移，进而造成速度误差。PIV-BOST 的想法是同时测：

- BOST：三维折射率场。
- PIV：中心平面二维速度场。

再利用 BOST 的折射率梯度估算 PIV 成像平面上的像素偏移，把速度误差补偿掉。

### 论文公开页面给出的实验线索

- 使用 one-to-nine endoscope system，为一个相机提供九个 BOST 视角。
- 对象是 turbulent non-piloted Bunsen flame。
- 三维折射率场由神经网络重构。
- PIV 是中心平面的低速二维速度测量。
- BOST 与 PIV 通过两套数字延迟/脉冲发生器同步，并设置固定相位延迟。
- 像素偏移由 PIV 相机和成像平面之间的热致折射率梯度估算。
- 小型无引燃火焰中，瞬时速度误差量级约为正负 2%。
- Springer 页面显示当前没有公开数据集可直接下载。

### 你能复现什么

保底 toy：

- 生成一张 2D 粒子图像。
- 构造一个折射率梯度场。
- 根据梯度场给粒子图像施加像素偏移。
- 用 OpenPIV 或 PIVlab 算速度。
- 对比补偿前后速度误差。

升级版：

- 用 NeRIF/BOST 重构出的三维 `n(x,y,z)` 切出 PIV 光路相关区域。
- 沿 PIV 相机光线积分梯度，估计像素偏移。
- 对真实 PIV 图像做几何校正。

毕设可写章节：

1. PIV 测速原理和粒子图像互相关。
2. 折射率梯度导致的像素偏移。
3. BOST 折射率场到 PIV 误差补偿的链路。
4. 合成 PIV 粒子图像验证。
5. 真实同步数据接口。

要问何远哲：

- 组内 PIV-BOST 数据是 2D PIV 还是 stereo-PIV？
- 是否能给同步时间戳、PIV 图像对、BOST 位移场、相机标定？
- 补偿公式是否已有 MATLAB/Python 版本？
- 本科阶段是否只做 planar PIV 就足够？

## Paper 2B: stereo-velocity PIV-BOST, Proceedings of the Combustion Institute 2026

正式信息：

- Title: Instantaneous refractive index compensation on stereo-velocity measurement in turbulent combustion
- Authors: Yutao Zheng, Yuanzhe He, Lianhao Feng, Shijie Xu, Ning Liu, Miaosheng He, Hong Liu, Yingzheng Liu, Weiwei Cai
- DOI: `10.1016/j.proci.2026.106175`
- ScienceDirect/DOI 入口：`https://doi.org/10.1016/j.proci.2026.106175`

### 它和 2025 PIV-BOST 的关系

这篇不是新的独立主线，而是 PIV-BOST 的自然升级：2025 Experiments in Fluids 论文先建立 simultaneous PIV-BOST 对速度测量折射误差的补偿框架；2026 Proceedings of the Combustion Institute 论文把问题推进到 turbulent combustion 中的 stereo-velocity measurement。对本科毕设最重要的信息不是“我要复现 stereo-PIV”，而是确认何远哲方向确实在向更真实、更复杂的速度测量误差补偿推进。

### 对本科题目的意义

- 如果师兄希望你贴 PIV-BOST，最稳的本科版本仍然是 planar/image-layer toy + 误差传播。
- stereo-velocity 论文适合放在相关工作或展望，证明这个问题有后续价值。
- 不要把题目一开始写成“完整 stereo-PIV-BOST 补偿”，除非组里明确给数据、标定、相机模型和评价标准。

要问何远哲：

- 这篇 stereo-velocity 工作里，本科生最可能参与的是图像校正、标定接口、误差指标，还是可视化报告？
- 如果我只做 2D PIV-BOST toy，是否需要在接口上预留 stereo-PIV 的相机参数和双视角速度字段？
- 真实 stereo-PIV-BOST 数据是否有可匿名的小样例，还是只适合论文阅读和展望？

## Paper 3: Tensor Decomposition-Based 4D BOST, ACM TOG 2026

正式信息：

- Title: Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction
- Authors: Yuanzhe He, Yutao Zheng, Shijie Xu, Ning Liu, Yingzheng Liu, Weiwei Cai
- DOI: `10.1145/3809488`
- ACM 页面：`https://dl.acm.org/doi/10.1145/3809488`
- ACM HTML 全文：`https://dl.acm.org/doi/full/10.1145/3809488`
- ACM eReader：`https://dl.acm.org/doi/epdf/10.1145/3809488`
- 本地结构化精读：`tdbost_reproducibility_audit.html`

访问边界：ACM 标注 Open access / CC BY 4.0；2026-07-10 已核验 HTML 全文与 19 页 eReader 可读，eReader 显示约 29.4 MB。无会话 raw PDF/CDN 请求仍返回 403，所以当前有“在线全文”，但没有“本地缓存 PDF”。

### 它解决什么

NeRIF 是三维重构。如果每个时间帧都独立做一次三维重构，内存、时间和时序一致性都会变成问题。4D BOST 把 `X-Y-Z-T` 当成一个联合时空对象，用紧凑张量表示共享跨帧结构；同时又显式处理强折射下“光线不再沿理想直线路径传播”的误差。它的核心并不是简单地给 NeRIF 加时间坐标，而是把表示、forward model、畸变校正和混合精度一起设计。

### 正文方法骨架

- **MM matrix-matrix 表示：**使用 `(XY-ZT)`, `(XZ-YT)`, `(YZ-XT)` 三组 plane pairs；对应特征做 Hadamard product，再线性投影到 F 维，最后由 MLP 解码折射率差。
- **主解码器：**Appendix A 给出 3 个隐藏层、每层 128、Swish 激活、标量输出；frequency encoding 层级 `L=3`。
- **可微 forward model：**在 ray 上做 randomized stratified sampling，以中心差分计算折射率梯度，同时积分 deflection angle 和 ray-path displacement；路径位移的双重积分用 cumulative sum 高效实现。
- **观测构造：**正文用 DeepFlow 从 background image pairs 提取水平/垂直 displacement，说明图像位移质量是 4D 重构的输入上限之一。
- **损失：**displacement MSE + regularization + boundary condition；regularization 包括 plane factors 的 TV 与 L1。Appendix B 给出 `lambda_reg=1e-4`、`lambda_bc=1.0`。
- **Distortion Correction：**单独的 DC MLP 输入 ray origin、direction、time、depth，结构为 6 层 x 200、3 输出；`Nc=6000` 后启用，主场与 DC 模块交替优化两轮。
- **训练策略：**数值实验从 60^3 coarse grid 起步，在 3k/6k iteration 上采样到 200^3，约 10k 后趋于稳定；前向 FP32、反向 FP16。论文报告混合精度减少约 40%-50% peak GPU memory。

### 正文实验与数字

- **Synthetic：**Fuel Injection 和 Spray Combustion；9 个 200 x 200 投影视角，0-170 度等角分布，20 micrometer pixel、50 mm focal length、50 mm cubic VOI，并加入 Gaussian noise；平台为 64-core Xeon、512 GB RAM、RTX 4090。
- **强折射消融：**在 `100G0`，NeRIF 的 L2/SSIM 为 `0.0190/0.9632`，本文 no-DC 为 `0.0107/0.9866`，full model 为 `0.0064/0.9948`。这说明强折射收益不能只归因于 tensor，DC/forward correction 是关键贡献。
- **速度：**synthetic 序列总计 200+ frames，正文报告 42 min、平均 12.6 s/frame。
- **真实火焰：**Photron AX100 mini + 9-input/1-output fiber bundle，1 kHz、100 microsecond、1024 x 1024；Bunsen premixed flame 共 500 frames，空间 tensor grid 120^3、time grid 200。test-view 平均 PSNR 约 27 dB；时序方法约 30 dB，而对照约 25 dB。
- **训练时间：**500-frame 实验 20k iterations 用时 80 min，约 9.6 s/frame；正文称 10k iterations 已有满意结果，约 4 s/frame。
- **动力学：**Fourier 分析提取到 20 Hz 与 40 Hz 主频，并指出主要扰动低于 50 Hz。由此可见 4D 评价不能只有逐帧 L2，还应看频谱、轨迹和瞬态保真。
- **显存：**Adjoint 200^3 为 13.90 GB，NeRIF batch 2048 为 22.35 GB，CP mixed 为 23.97 GB，本文 MM FP32 为 23.89 GB，MM mixed 为 13.08 GB。显存优势来自 MM 表示与 mixed precision 的组合，不是“张量分解天然省显存”。

指标提醒：正文 Eq. 24 的 L2 error 是平方范数比 `||error||^2 / ||GT||^2`；本地 M3B 的 `mean_rel_l2` 是非平方范数比 `||error|| / ||GT||`。二者只能比趋势，不能直接比较数值。

### 正文与公开仓库需要对账的地方

- 正文 4.4 写三组 tensor components 为 40/40/40；超参数节又把 `R=30, F=20` 写成默认质量/成本折中；公开 config 使用 30/30/30 与 F=20。
- Appendix A 写 decoder 为 3 x 128；公开 config 写 3 层、width 200。
- 正文强调 test view 未参与重建；公开 fuel manifest 的 18 个 test paths 全部与 train 重叠，而 spray 的 162/18 paths 才互不重叠。
- 公开仓库根目录无 LICENSE，并且存在 `cuda:3`、Linux x86-64 projection `.so`、`/home/PUBLIC_USER_REDACTED/...` import-time 路径和冲突 requirements。它是重要结构线索，不是已验证的一键复现包。

### 你能复现什么

本科不建议完整复现 TOG 论文。合理切法是：

- 构造 4D synthetic phantom：一个随时间移动/摆动的三维 Gaussian flame。
- 做逐帧 3D 重构 baseline。
- 做低秩时间表示：例如 `n(x,y,z,t) = sum_k a_k(t) * phi_k(x,y,z)`。
- 比较重构误差、时间一致性、训练时间和显存。
- 增加 held-out reprojection、frequency/event retention 和 geometry/sync/optical-flow bias，区分随机去噪与系统误差。
- 只写“4D BOST 子问题”，不写“完整 4D BOST 复现”。

毕设可写章节：

1. 为什么高速流场需要时序重构。
2. 逐帧重构的问题：噪声抖动、计算慢、缺少时间先验。
3. 低秩时序先验 toy。
4. 逐帧 NeRIF 与低秩 toy 的对比。
5. 真实高速 BOST 数据接口和未来工作。

要问何远哲：

- 4D BOST 现在最缺本科生能帮的部分是什么：参数扫描、可视化、低秩 toy、真实数据清洗、还是论文图复现？
- 是否有不涉密的 candle flame 或 synthetic 时序数据？
- 张量 rank、帧数、分辨率、视角数哪个最需要扫？
- 正文的 40/40/40 vs R=30/F=20、decoder 128 vs config 200、fuel test overlap 应以哪个实验版本解释？
- 如果只做 toy，师兄是否认可它作为本科毕业设计的挑战项？

## 三篇论文之间的选题收束

最稳：

- 主线：NeRIF/BOST 鲁棒性。
- 副线：自动报告和数据接口。
- 冲刺：真实 BOST 样例。

最贴实验：

- 主线：NeRIF 重构。
- 第二阶段：PIV-BOST 2D compensation toy。
- 冲刺：真实同步 PIV-BOST 数据。

最有新意：

- 主线：NeRIF 简化版。
- 第二阶段：4D low-rank temporal prior toy。
- 冲刺：和 4D BOST 论文做概念对照。

## 论文到开题报告的章节映射

| 开题章节 | 主要来源 | 写法 |
| --- | --- | --- |
| 研究背景 | OERF 官网、蔡老师主页、Computational Flow Visualization | 光学诊断 + 计算重构是 OERF 主轴 |
| 技术基础 | BOS techniques, BOST 前史 | 图像位移、折射率梯度、层析重构 |
| 现有问题 | NeRIF introduction | 体素离散、噪声、少视角、内存 |
| 方法方案 | NeRIF method | 坐标神经场、重投影 loss、鲁棒性实验 |
| 拓展方向 | PIV-BOST / 4D BOST | 折射补偿或时序低秩子问题 |
| 预期成果 | 本工作台 starter spec | 代码、数据接口、误差图谱、论文图 |

## 第一轮精读任务

1. NeRIF：只读 Introduction、Mathematical formulation、Algorithm、Numerical validation。
2. PIV-BOST：先读 Springer 摘要、图 1-3、实验系统、补偿公式和结果曲线；如果全文不可读，先向师兄要 PDF 或让他讲公式。
3. 4D BOST：先读 abstract、method overview、tensor decomposition schematic、实验指标；本科阶段只吸收思想。

## 源

- NeRIF arXiv HTML: `https://arxiv.org/html/2409.14722v2`
- NeRIF AIP page: `https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the`
- PIV-BOST Springer page: `https://link.springer.com/article/10.1007/s00348-025-04093-y`
- 4D BOST ACM page: `https://dl.acm.org/doi/10.1145/3809488`
