# LGWO-A24 公开数据桥：从 toy 到反应流 BOST，不能跨级冒充

**状态：** 数据路线已核验，尚未构成真实 BOST 结果。

**核心结论：** 截至 2026-07-20，本次对 Zenodo、TU Graz Repository、Penn State Data Commons、论文出版社页面和作者公开仓库的检索，没有找到同时具备“真实多视角 3D/4D BOST、明确开放许可、完整标定与可重算 `A/A^T`、独立 3D 折射率/密度真值”的公开数据集。这不是穷尽性不存在证明；在出现合格资源前必须用分层组合基准，不能把通用 CT 或 synthetic projection 写成 OERF 实验验证。

## 1. 四层基准

| 层级 | 目标 | 首选资源 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| P0 算子正确性 | 伴随、梯度、预算、baseline recovery | 显式矩阵 CT / 本地小矩阵 | 代码和线性代数正确 | BOST 或反应流有效 |
| P1 动态三维逆问题 | 稀疏视角、3D+time、时空正则 | [STEMPO v1.2](https://doi.org/10.5281/zenodo.8239013) | 动态 tomography 可复现 | 折射光学或燃烧 |
| P2 反应流物理体场 | 真实物理形态、温度/密度/组分、跨流态 | [MILD CH4/H2 DNS](https://doi.org/10.5281/zenodo.15575682) | physics-derived synthetic BOST | 实验相机链路 |
| P3 真实 BOS 前端 | 背景、火焰、光流/位移、光照与噪声 | [TU Graz HBOS](https://doi.org/10.3217/cvcf2-28b98) | 真实 BOS 图像前端稳定性 | 3D field truth |
| P4 OERF 迁移 | 同 rig 多视角、多帧、标定、论文主任务 | 何远哲师兄数据合同 | 真实组内任务 | 未见 rig 泛化需另验 |

## 2. 立即可用的三个公开资源

### 2.1 STEMPO：先把动态三维算子做对

STEMPO 是物理动态 X-ray phantom。官方 v1.2 提供静态、连续 360 投影以及 `8x45`、`8x180` 的 2D/3D 序列，并附几何表和示例算法，许可证为 CC BY 4.0。

Mac 首选：

```text
stempo_seq8x45_3d_b32.mat
size: 6.8 MB
md5: ec03dad2a0a823d410cb6897750588ce
```

用途：

- 测试 3D+time 数据读取与 geometry split；
- 用 ASTRA/HelTomo 或自写 matrix-free pair 验证 `A/A^T`；
- 比较 CGLS-24、TV/Huber、时间正则与 LGWO 壳层；
- 记录 end-to-end time、memory 和 call ledger。

边界：X-ray attenuation tomography 不是 BOS 偏折成像。它只能验证逆问题工程与动态结构。

### 2.2 MILD DNS：把 toy morphology 换成反应流物理体场

官方数据为 CC BY 4.0，含 7 个时间点，网格 `133x83x66`，提供 35 个组分及温度、密度、速度、HRR 等变量。单文件约 116.5--287.5 MB。

第一阶段只下载一个 HDF5，完成：

1. 读取 `rho, T, Y_k` 与坐标；
2. 明确波长与 Gladstone-Dale/混合物折射率模型；
3. 重采样到 `64^3`，保存坐标与单位；
4. 用同一 BOST operator pair 生成 train/clean/noisy displacement；
5. 按时间点或完整物理工况划分，禁止随机 voxel/crop 泄漏。

公开 DNS 上的结果必须写成 `physics-derived synthetic BOST`。它比解析 plume 更真实，但仍不是实验 BOST。

### 2.3 TU Graz HBOS：让真实图像前端进入验证

TU Graz 数据记录为 CC BY 4.0，包含真实 background、flame/time-signal TIFF 和 BOS ray-tracing 示例。它适合测试：

- background/deflected 图像预处理；
- optical flow 或 correlation 位移估计；
- illumination、amplitude modulation、饱和、坏点与 confidence；
- flow-off repeat 的 noise-floor 与 fallback envelope。

它没有与当前任务等价的独立 3D 折射率真值，不能拿 reprojection 变好直接写成 3D field 更准。

## 3. 独立的算子学习支线

[RealPDEBench](https://realpdebench.github.io/) 包含真实实验与 CFD，公开基线覆盖 FNO、DeepONet、CNO 等；其 combustion 场景是 NH3/CH4/air 火焰的 OH* 高速成像与 LES。

它适合回答“operator learning 能否从 simulation 迁移到 real observation”，但当前公开场景不是三维 BOST 逆重建。应作为独立 sim-to-real 支线，不能与 LGWO 的三维 field accuracy 混成一个主结论。

## 4. 统一数据与算子合同

```text
field[t,z,y,x,c]                 SI units, right-handed coordinates
measurement[t,view,v,u,2]       displacement px or deflection rad

A_theta = W_theta C_theta P_theta G_perp
A_theta^T = G_perp^T P_theta^T C_theta^T W_theta^T
```

每个资源固定：DOI/version、许可证、下载日期、原始文件哈希、文件清单、单位、坐标、重采样方法、split manifest 和 operator semantic hash。

### 伴随与非线性门

- 线性薄射线模型：float64 伴随内积相对误差 `<=1e-5`；
- float32：`<=1e-4`；
- 弯曲光线模型：验证 Jacobian-vector 与 Jacobian-transpose-vector，而不是把固定线性 `A^T` 套上去；
- FBP/FDK 不是自动等于离散 adjoint，必须单独验证。

### 划分

- 训练/验证/测试按完整工况、时间块、rig 与视角划分；
- 相邻帧不得随机散入不同 split；
- 重叠 crop 不得跨 split；
- 测试归一化不能重新估计；
- OOD/fresh 不得参与路线选择。

### 指标

- field relative-L2；
- H1/gradient error；
- measured、independent-clean 与 held-out-ray reprojection；
- 逐 rig/工况 P95、harm、worst；
- front/shock location 或 F1；
- `A/A^T` calls、端到端时间、峰值内存、参数量。

## 5. 向何远哲师兄索取的最小合同

1. 完整视角/时间索引的原始 REF/DEF 图像、位深、格式、哈希与容量；
2. 每台相机内外参、畸变、背景平面位姿、标定板定义和 reprojection error；
3. 触发、时间戳、曝光、帧率、掉帧与跨相机最大时间差；
4. 气体组成、压力、温度、流量、波长及相机/流场/背景距离；
5. 位移或偏折产品的单位、像素到角度换算、算法、窗口、正则与版本；
6. 每帧有效区、遮挡、背景缺失、饱和、坏点、confidence 和 uncertainty；
7. 薄射线或弯曲射线、平行/锥束、积分插值规则与边界条件；
8. matrix-free `A/A^T`、ray weights 或重建它们所需的完整几何，以及组内认可的 adjoint tolerance；
9. CFD、标定 phantom、独立视角或其他验证依据及其配准、单位和不确定度；没有也应明确写“无”；
10. 固定 train/validation/test rig、工况和时间块，发表/共享权限，现有 baseline 脚本与硬件预算；
11. 同一 rig 通常连续多少帧，以判断离线 basis 的摊销 break-even；
12. 是否有 flow-off repeats，可否用 25% 时间块永久估计 noise floor。

## 6. 执行顺序

```text
P0: LGWO solver shell + paired A/A^T unit tests
 -> P1: STEMPO b32 dynamic 3D adapter
 -> P2: one MILD HDF5 -> refractive-index field -> synthetic BOST
 -> P3: TU Graz real-image displacement/noise front end
 -> P4: OERF geometry/data adapter, keeping every gate frozen
```

每一级只补前一级缺少的证据。任何一级失败，都保留失败原因，不靠换 split、换指标或打开下一层数据补救。
