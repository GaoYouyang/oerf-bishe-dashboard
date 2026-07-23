# VD0-B 真实探测器图 + 相机姿态探针：压力信号存在，安全路由 NO-GO

日期：2026-07-17

## 一句话结论

这轮恢复了 PSU 数据的真实 detector pixel 坐标，在每台相机的 256 条不规则抽样射线上建立 8 邻域图，并从白化二维位移提取局部 Jacobian、front 集中度、方向各向异性、散度/旋度平衡和世界坐标方向一致性。它确实把训练集的留一形态、留一噪声平均场增益从负数推到了正数，但转移到 validation/calibration 后伤害率和 front 指标仍然失败，因此按预先写下的规则停止 DeepSets，不生成独立重复，不宣称算法优越。

正式判决：

```text
VD0B_DETECTOR_POSE_NOT_TRANSFER_SUPPORTED_STOP_SET_ENCODER
```

## 1. 为什么必须重做 detector 几何

PSU 当前 B0 接口每台相机从一百多万条 active rays 中按分位点固定选出 256 条射线。之前的 synthetic noise 代码为了制造行列相关噪声，把这 256 条射线按抽样顺序临时排成 `16 x 16`；但抽样序号相邻不代表原始 detector pixel 相邻。因此：

- 不能在这个伪网格上直接使用 Sobel、FFT ridge 或普通卷积；
- 否则得到的“front”可能只是抽样顺序的产物；
- 必须先由源图像线性索引恢复真实行列坐标，再在不规则点集上定义局部导数。

作者预处理代码把相机图像以 MATLAB `epsu(:)` / `epsv(:)` 展平，并用 `find(mask)` 形成索引。已经审核过的 mask adapter 保存的是 `MATLAB find index - 1`。对图像高 `H`，本轮使用：

```text
row = index mod H
column = floor(index / H)
```

两个轴使用同一个 pixel scale 居中归一化，避免把 2160 x 2560 传感器错误拉伸为正方形。单元测试覆盖了列主序恢复和边界索引。

数据与论文入口：

- PSU 70-view 数据论文：[Molnar et al., Open-source BOS tomography dataset of high-speed flow over a flight body](https://arxiv.org/abs/2508.17120)
- 官方公开数据目录：[Penn State Data Commons download directory](https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/)
- 正式论文 DOI：[Experiments in Fluids 67, 44 (2026)](https://doi.org/10.1007/s00348-026-04189-z)

## 2. 这轮到底测了什么

### 2.1 数据和算子

- 9 台真实 PSU support cameras；
- 每台 256 条固定 active rays，共 2304 条；
- 真实 detector image size：2160 x 2560；
- 32 x 32 x 32 reconstruction grid；
- truth forward 使用 32 个有限孔径样本，nominal inverse 使用 8 个；
- 形态：`plume`、`wavy_front`、`thin_front`、`double_front`、`annular_kernel`、`oblique_shock`；
- risk train / validation / calibration：48 / 24 / 30；
- 同一 PCGLS-4 baseline、同一四专家有限动作库、同一 `4F + 4A^T`；
- real PSU deflection values 没有进入本轮路由或指标；
- fresh audit 没有打开。

### 2.2 不规则 detector graph

每条射线在真实归一化 pixel 坐标中找 8 个最近邻，并用加权局部最小二乘估计二维位移 Jacobian：

```text
J_i = argmin_J sum_j w_ij ||(d_j - d_i) - J (x_j - x_i)||^2 + lambda ||J||^2
```

图几何审计：

| 项目 | 数值 |
|---|---:|
| views | 9 |
| rays / view | 256 |
| neighbors | 8 |
| median nearest-neighbor distance | 0.01454 |
| median 8th-neighbor distance | 0.04804 |

这些距离使用同一 detector pixel scale 归一化，只用于邻域与局部导数，不声称是物理长度。

### 2.3 30 个显式特征

每台相机先得到 6 个 observable quantities，再对 active views 做 mean/std/min/max：

1. whitened displacement vector RMS；
2. neighbor contrast / signal ratio；
3. local Jacobian RMS；
4. top-10% front energy share；
5. 2D structure-tensor anisotropy；
6. divergence-curl energy balance。

再加入 6 个跨相机量：

1. active view fraction；
2. 世界坐标 front orientation dyadic 的三个归一化特征值；
3. orientation entropy；
4. effective front-view fraction。

二维主方向通过作者数据中的 `Ruvecs/Rvvecs` 投到世界坐标。作者预处理把它们定义为 camera `pu/pv` 方向在 background plane normal 上去除法向分量后的 projection vectors。本轮对每台相机取均值并做数值正交化；这是一个可审计的投影 frame，不等同于宣称完整相机外参已被重新标定。

特征严格禁止：

- 三维 truth；
- morphology label；
- 重建结果；
- 第一步之后的 residual；
- PSU 实测 `epsu/epsv`。

## 3. 公平结果

### 3.1 转移场指标

| strict route | validation field gain | calibration field gain | validation harm >1% | calibration harm >1% |
|---|---:|---:|---:|---:|
| pooled initial normal | +3.321% | +2.907% | 0.0% | 3.33% |
| detector graph only | +1.984% | +1.879% | 12.5% | 3.33% |
| pooled + detector graph | +2.805% | +2.901% | 12.5% | 10.0% |

`pooled + detector` 没有在两个 split 都击败 pooled；validation p10 为 -1.113%，calibration p10 为 -0.155%，伤害率超过 5% 门槛。

### 3.2 front 安全指标

| strict route | validation front mean | calibration front mean | validation front p10 | calibration front p10 |
|---|---:|---:|---:|---:|
| pooled initial normal | +1.177% | -0.261% | -3.219% | -3.515% |
| detector graph only | -0.241% | -1.694% | -4.622% | -6.809% |
| pooled + detector graph | -0.077% | -0.778% | -7.873% | -7.242% |

新增特征没有保护 front，反而使尾部更差。对反应流、shock 和 flame front 任务，这一条足以阻止“平均场误差看起来还可以”被升级成方法成功。

### 3.3 为什么说它不是完全没信息

训练集压力审计的 strict route：

| feature set | leave-one-family mean field gain | leave-one-noise mean field gain |
|---|---:|---:|
| pooled initial normal | -0.999% | -2.217% |
| detector graph only | +1.051% | +1.118% |
| pooled + detector graph | +0.907% | +0.947% |

这支持一个很窄的机制判断：真实 detector 邻域和二维 front 描述比“只看三维 pooled adjoint”更能区分开发集中的某些风险变化。

但它不能被写成成功，因为：

1. 这是 post-open synthetic development；
2. validation/calibration 的 field harm 和 front safety 没通过；
3. legacy `camera_correlated` noise 是在伪 `16 x 16` 排列上生成的，不是 measured covariance；
4. 74 维拼接特征面对 48 个训练样本，虽然有 OOF ridge 和强正则，仍可能学习开发生成器的偶然结构；
5. 没有真实 3D field truth，也没有独立实验 session。

## 4. 与何远哲方向的关系

NeRIF 把 BOST 写成沿已标定光路积分 refractive-index gradient 的 inverse problem，并在实验中使用 8 个视角重建、1 个视角做 reprojection 验证。它明确指出位移来自 distorted/reference images 的 optical flow，并通过 camera projection matrix 将世界梯度映射到相机坐标。[NeRIF 开放稿](https://arxiv.org/abs/2409.14722)

本轮没有替代 NeRIF，也没有训练新的 INR。它试图回答 NeRIF/TDBOST 前面的一个更小问题：

> 在开始昂贵的三维优化前，能否仅从逐相机二维位移、噪声尺度和姿态判断哪种低预算逆算子更安全？

当前答案是：在这套 opened synthetic development 上，显式 detector graph 能看到压力变化，但不能安全路由 front，因此不能继续用 DeepSets 扩容。

相关背景：

- BOS 位移是沿光路积分的密度/折射率梯度观测，成像清晰度、焦深和相机几何会限制空间细节：[Raffel, Background-oriented schlieren techniques](https://doi.org/10.1007/s00348-015-1927-5)
- 高空间分辨率位移对 shock reconstruction 很关键，低分辨率 correlation vectors 会直接破坏压力场恢复：[Hayasaka et al., OF-BOS for an underwater shock](https://doi.org/10.1007/s00348-016-2271-0)
- 有限孔径提高光量但缩小景深并模糊 deflection，需要显式 forward model，而不是交给路由器猜：[Molnar and Grauer, Reconstructing Hypersonic Flow Over a Bluff Body](https://doi.org/10.2514/6.2024-2493)
- 稀疏视角本身使 TBOS 严重病态，高频结构需要受几何约束的表示和采样策略：[NeDF](https://arxiv.org/abs/2409.19971)

## 5. 为什么现在不训练 DeepSets/FNO

DeepSets 能表达更复杂的 camera interaction，但它不能修复三个事实：

1. 标签仍来自同一 opened finite-action bank；
2. synthetic correlated noise 还不是 detector-geometry-consistent covariance；
3. 显式特征已经在 transfer front 上方向错误。

在这个节点继续加网络，最可能发生的是：

- OOF 平均数继续上升；
- validation/calibration harm 被模型容量掩盖；
- 论文最后只能解释“为什么开发集很好看、真实数据不工作”。

因此 VD1 保持封存。

## 6. 下一步只做数据真实性，不做容量搜索

### P0：真实 measurement-distribution audit

使用 PSU 公开 `epsu/epsv`，只计算 30 个 detector graph descriptors，不训练路由器，不用三维 truth。比较真实观测与 synthetic train/validation/calibration 的 robust distance、每特征区间覆盖、相机间方向结构和 front concentration。

停止条件：

- 若大部分真实样本落在 synthetic 95% 区间之外，当前 synthetic route study 只能作为接口练习；
- 先改噪声、mask、sampling 和 forward mismatch，不能调网络。

### P1：flow-off covariance / reference repeats

请师兄提供同一相机、同一设置下的 flow-off/reference repeats。需要：

- 每台相机至少 20-50 帧；
- raw/reference 或 displacement 均可；
- 保留 timestamp、exposure、f-number、ROI、mask；
- 不要先平均掉 row/column、固定图样和时间相关噪声。

用这些数据估计 graph covariance、per-view bias、cross-component correlation 和 temporal drift，替换伪 `16 x 16` noise generator。

### P2：一条 held-out camera 作为无真值终点

只有真实 covariance 接口闭合后，才重新冻结一个低容量 route。主终点优先：

1. held-out camera reprojection；
2. front/ridge localization；
3. risk-coverage 与拒答；
4. 同 `F/F^T` 调用数和 wall-clock。

仍不能只报 pooled field mean。

## 7. 现在请师兄确认

1. 组内能否给 flow-off/reference repeats，而不必先给完整 3D 数据？
2. NeRIF/TDBOST 当前输入保存的是 dense optical-flow field，还是已经筛选后的 ray rows？
3. 是否保留 detector pixel coordinates、ROI/mask 和每台相机 `P/R/t` 或等价 projection vectors？
4. 组内真实失败主要是 front 被抹平、伪影、相关噪声、相机不同步，还是几何/有限孔径失配？
5. 能否固定一台相机不参与重建，只做 held-out reprojection？
6. 如果只能先给 1-3 帧，能否同时给对应 reference frame、mask、位移和 baseline reconstruction？

## 8. 可复现入口

- 配置：`demo_t16_operator/configs/psu_b0_detector_pose_probe_v1.json`
- 特征：`demo_t16_operator/psu_b0_detector_graph_features.py`
- 评估 runner：`site_tools/run_psu_b0_view_decomposed_probe.py`
- 公开 summary：`docs/psu_b0_detector_pose_probe_public_summary.json`
- 绘图：`site_tools/plot_psu_b0_detector_pose_probe.py`
- 图：`demo_t16_operator/results/psu_b0_detector_pose_probe/psu_b0_detector_pose_probe_figure.png`

本机结果：

```text
wall time: 3.48 s
peak RSS: 429 MB
feature operator calls: 0F + 0A^T
transfer reconstruction: 4F + 4A^T per sample
decision: NO-GO
```
