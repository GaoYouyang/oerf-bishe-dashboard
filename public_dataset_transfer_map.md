# 算子学习 × 三维重建：公开数据与迁移地图

更新时间：2026-07-15
目标：在尚未拿到 OERF 数据时，建立可训练、可否证、以后可平移的算法管线。这里不把二维燃烧预测、真实 BOS 重投影和三维真值误差混成同一种证据。

## 1. 为什么必须用三层数据

完整任务是：

`(raw BOS images / displacement, camera rays, mask, noise) -> refractive-index or density volume`

目前没有一个公开数据集同时给出真实背景图、完整相机几何、逐像素噪声和实验三维真值。因此实验必须分层：

| 证据层 | 解决什么 | 不能证明什么 |
| --- | --- | --- |
| 可控三维 phantom | 正确伴随、优化机制、场误差、严格消融 | 真实图像与模型失配 |
| 公开反应流 CFD/LES | 火焰前缘、激波、温度/密度形态的跨家族压力测试 | 真实相机链与实验真值 |
| OpenBOST / 未来 OERF | 标定、位移、遮挡、坏点、真实噪声、留相机重投影 | 没有独立真值时的绝对三维准确度 |

只有三层结论一致，才进入论文 superiority claim。

## 2. 当前最有用的数据源

### A. OpenBOST：现实成像链主锚点

- 70 个真实 BOS 视角，含标定、flow-off/flow-on、mask、deflection、三维重建和 NIRT 代码。
- 官方说明明确：九视角可做重建并用其余视角验证；数据还包含 cross-correlation、wavelet optical flow 和 Horn-Schunck optical flow 对照。
- 公开说明记录 2000 帧、10 Hz 的平均观测，噪声包含风洞运动、热噪声和 shot noise；它不是逐帧湍流真值。
- 适合验证：loader、可变视角、held-out camera、标定扰动、端到端成本、真实残差分布。
- 不适合直接声称：实验三维场 L2 更低，因为没有独立体真值。

入口：[数据目录](https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/) · [论文/预印本](https://arxiv.org/abs/2508.17120)

完整数据约 48.1 GiB。当前策略是先抽说明、代码和一个小视角包，不整库盲下。官方说明未给出常见开源许可证，因此本仓库只保存索引和自写适配器，不公开转存原数据或原代码。

### B. RealPDEBench Combustion：反应流与 sim-to-real 主锚点

- 30 条真实 OH* 化学发光序列和 30 条匹配 LES 序列，2001 帧、128×128。
- LES 分支有压力、放热率、CH4/CO/CO2/H2O/NH2/NH3/OH、温度和三速度分量等 15 通道。
- 公开 Arrow 版本是二维场，不是三维体；它适合训练时间算子、反应流形态先验和 sim-to-real，不可当作 3D BOST 真值。
- 许可证为 CC BY-NC 4.0。先取 metadata/real 小分支，再决定是否下载约 2 GB 的单个 numerical shard。

入口：[官方数据卡](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench) · [论文](https://arxiv.org/abs/2601.01829)

### C. Michigan 2D combustor：低成本时间算子热身

- 2D 单喷注器火箭燃烧室，包含压力、速度、温度和组分，时间步长 `1e-7 s`。
- 全量 58.4 GB，但 grid 和 sample code 可单独取，适合先验证 field loader、窗口切分、DeepONet/FNO/FFNO 时间预测。
- 它只负责学习工程和反应流时序，不进入三维重建主结论。

入口：[Deep Blue Data](https://deepblue.lib.umich.edu/data/concern/data_sets/gx41mj04d)（CC BY-NC 4.0）

### D. SDRBench S3D：大体场压力测试候选

- 11 个 500³ 双精度三维燃烧场，约 44 GB。
- 优点是体场真实来自 S3D；缺点是当前公开包巨大、样本数量有限，且不是相机观测。
- 只有在使用独立的 nonlinear/cone-ray 生成器形成观测后，才能测试 inverse crime 是否缓解；单快照不能训练并证明泛化。

入口：[SDRBench](https://sdrbench.github.io/)

### E. FiReSMOKE：独立生成器候选

- GPLv3 的 OpenFOAM + OpenSMOKE++ 有限速率反应流求解套件，源包约 25.6 MB。
- 它不是现成训练集，但可用于生成与当前 analytic phantom 完全不同的场，最适合做第二生成器。
- 首轮只跑一个小网格测试例；若本机 OpenFOAM 环境成本过高，再迁移到服务器。

入口：[Mendeley Data](https://data.mendeley.com/datasets/s3cgvw4rsg)

## 3. 下载优先级

1. **立即可做：** 保留 OpenBOST 255 KB 文件索引和 3.75 MB readme；用 HTTP range 抽取小代码说明，不下载 12 个大包。
2. **小于 10 MB：** 下载 Michigan README、grid 与 sample package，完成 2D reacting-flow loader。
3. **存储确认后：** 下载 RealPDEBench 一个 real trajectory；确认 Arrow 解码与训练速度后，才取一个 numerical trajectory。
4. **师兄审核算法后：** 选择 OpenBOST 一组 9/13-view 所需分卷，而不是整库。
5. **需要独立三维生成器时：** 在 S3D 单快照和 FiReSMOKE 小算例之间按网络与磁盘成本二选一。

## 4. 统一数据接口

代码入口：`demo_t16_operator/measurement_contract.py`

每个数据源最终只需提供：

- `observation [B,D,V,N]`
- `view_mask [B,V]`
- `noise_std`，可逐相机或逐像素
- `view_angles_degrees` 与 `geometry_id`
- `support [B,1,D,H,W]`
- 一个严格通过内积检验的 `forward(x)` / `adjoint(r)`
- 可选 `truth`；没有真值时禁止计算伪 field L2

真实 OERF loader 不应改模型，只应替换 forward adapter、标定与单位转换。

## 5. 第一篇论文可接受的证据组合

主张限定为“低调用次数、几何/噪声变化下的 BOST 重建”时，最低组合是：

1. 可控 phantom：fresh fields + fresh geometries + 独立 nonlinear forward。
2. 公开 CFD：至少一个训练未见的反应流场家族。
3. OpenBOST：真实 held-out camera、残差白化、坏视角与端到端 wall time。
4. OERF：至少一个九视角真实工况；若没有三维真值，只能报告 data consistency、重复性和物理边界。
5. 强基线：CGLS/TV、预白化 SPG/PBB + Morozov、FNO、FFNO、DeepONet、Learned Primal-Dual/同规模 unrolled control。

当前缺口不是“没有更多模型名字”，而是公开三维反应流样本、独立 optical generator 和组内真实 noise/calibration。它们已经被明确放入数据请求合同。
