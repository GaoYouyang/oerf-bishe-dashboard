# N1.5 条件化前向模型近似误差：算法与验证协议

- 日期：2026-07-18
- 状态：`DESIGN_ONLY_NOT_YET_IMPLEMENTED`
- 研究问题：有限孔径、景深、光线弯曲、相机标定和体素离散造成的 forward mismatch，能否在不
  读取测试真值的前提下，用低参数统计模型或小型算子学习模块补偿？
- 边界：下述内容是研究假设与预注册草案，不是算法成功、真实数据结果或论文结论。

## 1. 为什么这个问题贴合何远哲/NeRIF 主线

何远哲等人的 NeRIF 明确把 voxel discretization、空间分辨率、噪声免疫和计算成本列为 BOST
难点，并用连续 neural field 与随机光线积分缓解体素表示问题。它并不意味着 finite aperture、
标定漂移或 flow-on optical-flow error 已被完整统计建模。

N1.2 pilot 中 voxel-versus-continuous mismatch 已达到 `15.73%–27.79%`；N1.3/N1.4 又显示只换
covariance、Huber 和 edge prior 无法保护同一薄界面反例。因此下一步应直接建模 `G_H-G_L`。

## 2. 高低保真观测模型

定义：

```text
y = G_H(x; theta + delta_theta, q) + e
G_L(x; theta) = low-fidelity thin-ray / straight-ray / voxel forward
epsilon(x,z) = G_H(x; theta + delta_theta, q) - G_L(x; theta)
```

其中 `x` 是折射率或密度场，`q` 包含 f-number、焦平面、焦距、背景距离与曝光，`delta_theta`
表示内外参和畸变不确定性。高保真 `G_H` 至少包含 aperture sampling；若梯度足够强，再用 eikonal
ray tracing 处理弯曲路径。

Molnar 等人的 cone-ray BOS 工作直接证明有限孔径 blur 会使 thin-ray 重建不稳，并在 f/22 到
f/4 的范围比较 cone-ray 模型。因此有限孔径不是“多加一点 IID noise”，而是结构化 forward
mismatch。

## 3. 三层候选，从最小模型开始

### A. AEM-Mean：无网络均值校正

在独立 physics-development phantoms 上生成 paired `G_L(x_j),G_H(x_j)`：

```text
mu_epsilon(z_bin) = mean_j epsilon_j
y_corrected = y - mu_epsilon(z)
```

先比较 `G_L`、`G_L+global mean`、按 camera/f-number/bin 条件化均值。若简单均值已没有稳定增益，
不应训练网络。

### B. AEM-LR：低秩均值与协方差

```text
epsilon | z ~ N(mu_epsilon(z), U(z) diag(sigma(z)^2) U(z)^T + D(z))
```

先用 PCA/随机 SVD 得到固定 `U`，仅拟合小维度系数；比较 global、per-camera、geometry-
conditioned 三档。最终 inverse 使用 `Sigma_sensor + Sigma_epsilon`，两者必须分账。

### C. AEM-Op：小型条件算子网络

只有 A/B 在 locked development 上存在 headroom，才让小型网络读取部署可见的 `z`，输出
`mu` 的低秩系数与非负 variance。网络不输出三维场，不接收 test truth、family label 或 held-out
camera residual。forward/adjoint 必须来自同一 correction parameterization，并通过伴随恒等式。

## 4. 最小因子实验

| 组别 | forward | mismatch mean | mismatch covariance | 用途 |
|---|---|---|---|---|
| L0 | `G_L` | 无 | sensor only | 低保真基线 |
| L1 | `G_L` | global | 无 | 均值是否有 headroom |
| L2 | `G_L` | camera/f-number conditioned | 无 | 条件信息价值 |
| L3 | `G_L` | conditioned | low-rank + diagonal | 完整 AEM |
| H-oracle | `G_H` | 不需要 | sensor only | synthetic ceiling，不可部署基线 |

每组保持同一 reconstructor、同一迭代预算和同一正则选择协议。N1.3 的 fixed quadratic/Huber 与
N1.4 的 zero-start `lambda=0.2` 都要作为强 classical controls，而不是只和旧 CGLS 比。

## 5. 数据分区与泄漏防火墙

1. `physics-fit`：拟合 mismatch mean/subspace；phantom family、CFD trajectory、geometry seed 不得
   与 confirmation 重叠。
2. `selection-development`：只选 rank、binning、ridge 与 fixed solver；按完整 session/rig 分组。
3. `confirmation`：冻结代码、哈希、阈值和候选后一次打开；不得再调模型。
4. `real audit camera`：整台相机和完整时间段只做最终 held-out projection/image consistency。
5. `G_H` 在测试时若参与迭代，就必须称 high-fidelity inversion，不能再宣称便宜 AEM。

## 6. 必报指标与门槛

### 合成 confirmation

- field relative-L2、H1、mean bias；
- front location、thickness、双界面分离误差；
- held-out high-fidelity reprojection；
- NLL、95% interval coverage 与 calibration slope；
- 每 session、每 family、每 f-number 的均值和 worst；
- `A/A^T` calls、high-fidelity pair generation cost、setup amortization、wall time 和 peak memory。

### 真实无真值实验

- 永久 held-out camera 的 displacement 或 raw-image residual；
- 不同 f-number/焦平面下重建的一致性；
- phantom/PIV/pressure/thermocouple/CFD 中至少一种独立证据；
- failure rate 与 fail-closed coverage，不能只报接管样本均值。

### development 最低门

- field mean gain `>=5%`；
- H1 mean gain `>=3%`；
- worst field gain `>=-1%`；
- `>1%` harm 为 0，且给 session-clustered 置信上界；
- held-out high-fidelity residual mean `<=1.0x`、worst `<=1.05x` 强基线；
- 所有 family/session/f-number mean 非负；
- 若任何门失败，不打开 confirmation/OOD，不训练 AEM-Op。

## 7. 向何远哲师兄确认的十个问题

1. 当前 BOST/NeRIF forward 是 straight-ray、paraxial 还是 curved-ray？
2. 是否显式采样镜头孔径；每个 pixel/ray 有多少 aperture samples？
3. 能否获得每台相机的 f-number、焦距、对焦距离、相机到流场/背景距离？
4. 是否保留内外参 bootstrap、reprojection residual、distortion 与 bad-pixel mask？
5. 原始 distorted/reference image、optical-flow displacement 和 confidence 是否都在？
6. 能否固定一台相机和一个完整 session 永久不参与 reconstruction/selection？
7. 是否有同一场在不同光圈或焦平面下的 paired acquisition？
8. 是否有 phantom、CFD、PIV、pressure 或 thermocouple 可作独立物理证据？
9. 何师兄最想修复的是 finite aperture、ray bending、camera calibration 还是 optical-flow error？
10. 实验室允许公开的代码、几何 metadata、匿名化 projection 与结果图边界是什么？

## 8. 一级来源阅读顺序

1. [He et al., NeRIF, Physics of Fluids 2025](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)，[作者稿](https://arxiv.org/html/2409.14722v2)：先读 BOST 方程、连续表示、随机积分和 held-out camera 验证。
2. [Molnar et al., depth-of-field/cone-ray BOS, author manuscript](https://arxiv.org/abs/2402.15954)，[DOI](https://doi.org/10.2514/1.J064095)：理解 finite aperture 为什么属于 forward physics。
3. [Kolehmainen et al., Bayesian approximation error, JOSA A 2009](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-26-10-2257)：学习如何用 accurate/coarse model pairs 建模均值与协方差。
4. [Grauer and Steinberg, unified BOST, Experiments in Fluids 2020](https://link.springer.com/article/10.1007/s00348-020-2912-1)：理解将 optical-flow 与 tomography 解耦会怎样传播误差。
5. [Cai et al., direct RBF BOST, Optics Express 2022](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100)：学习 12-camera reconstruction + 3-camera validation 的无真值协议。

## 9. 当前最诚实的论文命题

> 在三维 BOST 中，将有限孔径和几何不确定性产生的 high/low-fidelity discrepancy 与 detector
> covariance 分离，并以条件低秩 approximation-error likelihood 约束 NeRIF/经典重建，能否在
> 未见 rig、未见形态和 held-out camera 上同时改善薄界面误差、投影一致性和不确定度校准？

这个命题有真实物理意义，也允许失败。只有 confirmation 数据支持时，才升级为方法论文结论。
