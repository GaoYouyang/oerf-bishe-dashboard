# BOST / NeRIF 最小 Demo 实验协议

目标：用最小可运行实验证明你不是只会读论文，而是已经能把 BOST/NeRIF 拆成数据、模型、指标和图表。

## 总验收标准

一个合格 demo 必须能自动输出：

- 输入 phantom 图。
- 多视角 simulated displacement。
- baseline 重构结果。
- neural field 重构结果。
- 中心切片误差图。
- 重投影误差图。
- 指标表：L2、CC、SSIM/PSNR、训练时间。

## Demo M0: 2D 折射率场玩具模型

目的：

先在二维里理解“折射率场 -> 位移 -> 重构”。

输入：

- 2D Gaussian blob 或两个 Gaussian blob。
- 3 / 5 / 7 个投影视角。

前向：

- 近似位移 = 沿视线方向积分折射率梯度。
- 可以先直接生成位移，不做真实背景图像。

反演：

- baseline：粗网格最小二乘 + Tikhonov。
- neural：MLP 输入 `(x,y)` 输出 `n`。

指标：

- L2 error。
- CC。
- 重投影 error。

通过标准：

- 能看到视角数增加时重构更稳定。
- 能输出一张 2 x 3 对比图：ground truth / baseline / neural / error。

当前状态：

- 已完成，路径：`demo_m0/`。
- 主结果图：`demo_m0/results/m0_summary.png`。
- 视角数曲线：`demo_m0/results/view_count_curve.png`。
- 当前 9 视角结果：baseline relative L2 约 0.113，coordinate-field inverse relative L2 约 0.108。
- 诚实边界：这是 2D toy，不是完整 NeRIF；低视角下 coordinate-field 可能不稳，正好引出 M1/M2 的鲁棒性问题。

## Demo M1: 3D synthetic BOST

目的：

接近 NeRIF 论文的最小设置，但降低分辨率。

当前已实现版本：

- 路径：`demo_m1/`。
- 使用一个 `44 x 44 x 22` 三维折射率 phantom。
- 先做 2.5D / 3D-stack sparse-view BOST：每个 z 切片生成 BOS-like deflection，再逐层做简化 filtered backprojection。
- 再用一个 compact 3D random Fourier coordinate representation 对 stack reconstruction 做体正则化。
- 这不是完整三维 ray geometry，也不是 NeRIF 论文复现；它是一个“少视角下隐式体先验是否有用”的压力测试。

输入：

- 3D phantom：Gaussian blob、火焰薄层或多个随机 blob。
- 重构体积：例如 `64 x 64 x 64` 或更低。
- 视角：3 / 5 / 7 / 9。

前向：

- 平行光线或简化相机光线。
- 沿 ray 采样梯度并积分。
- 生成每个视角的 2D displacement map。

反演：

- baseline：低分辨率体素法。
- neural：MLP 输入 `(x,y,z)` 输出 `n`。
- 可选：Fourier feature。

指标：

- 体积 L2。
- 中心切片 SSIM/CC。
- leave-one-view-out re-projection error。
- 训练时间和显存。

通过标准：

- 9 视角结果明显优于 3 视角。
- neural field 在同等粗网格 baseline 下至少有一项指标更好，或给出失败原因。

当前结果：

- 主压力测试使用 5 视角。
- baseline relative L2 约 0.257，3D coordinate-regularized stack relative L2 约 0.234。
- 视角数扫描显示：3-5 视角下坐标正则化降低误差；7/9/13 视角下，干净的传统栈重建更强。
- 可用于开题的结论：隐式/神经场不是天然更好，它的研究价值应当放在少视角、噪声、真实几何、mask、跨时间/跨物理量约束中讨论。

当前输出：

- `demo_m1/results/m1_volume_summary.png`
- `demo_m1/results/m1_metrics_card.png`
- `demo_m1/results/m1_view_count_curve.png`
- `demo_m1/results/metrics.csv`
- `demo_m1/results/view_count_metrics.csv`

下一步升级：

- 把 post-FBP coordinate regularizer 换成 PyTorch NeRIF-style forward-model loss。
- 加入真实 9 视角几何、mask、位移噪声和 leave-one-view-out reprojection。
- 若师兄给 4D 或 PIV-BOST 数据，再把 M1 作为真实数据前的 sanity check。

## Demo M2: NeRIF 鲁棒性扫描

目的：

把“复现”升级成“可写论文的系统分析”。

当前已实现版本：

- 路径：`demo_m2/`。
- 复用 M1 的三维 synthetic BOST proxy。
- 扫描变量：视角数 `3 / 5 / 7 / 9`，deflection noise `0 / 0.03 / 0.06 / 0.10`。
- 对比方法：传统 stack baseline 与 3D coordinate-regularized stack。
- 额外扫描：在 5 视角、noise 0.06 条件下测试 random Fourier feature capacity。

变量：

- 视角数：3 / 5 / 7 / 9 / 12。
- 噪声：0 / 0.05 / 0.10 / 0.15。
- 表示：plain MLP / Fourier MLP / 可选 hash encoding。
- 采样数：32 / 64 / 128 / 200。

输出：

- 每个变量一张曲线。
- 每组实验自动保存 config 和结果。
- 一张总表：最佳设置、失败设置、耗时。

通过标准：

- 能回答“NeRIF 在什么条件下比 baseline 稳，什么条件下会失败”。

当前结果：

- 3/5 视角的 8 个噪声设置中，coordinate regularizer 全部降低相对 L2。
- 7/9 视角的 8 个噪声设置中，传统 stack baseline 全部更好。
- 容量扫描显示：feature count 40/80 时过度平滑或不稳定，120 以后开始超过 5 视角 noisy baseline。
- 可用于论文的结论：隐式体先验的优势边界与视角数、噪声和表示容量耦合；毕业设计的价值不是证明 AI 永远更好，而是给出何时该用、何时不该用的实验图谱。

当前输出：

- `demo_m2/results/m2_noise_view_scan.png`
- `demo_m2/results/m2_improvement_heatmap.png`
- `demo_m2/results/m2_capacity_scan.png`
- `demo_m2/results/m2_metrics_grid.csv`
- `demo_m2/results/m2_capacity_metrics.csv`

## Demo M3A: PIV-BOST 2D 补偿 toy

适用条件：

师兄给 PIV-BOST 数据，或你想把第二阶段贴实验。

当前已实现版本：

- 路径：`demo_m3a/`。
- 当前不是完整粒子图像互相关，而是速度向量场层面的误差传播 toy。
- 物理逻辑：观测 PIV 位移 = 真实速度位移 + 粒子运动前后折射图像位移的变化量。
- 用 BOST-style 折射位移估计扣除偏差项，得到补偿后速度场。

输入：

- 合成粒子图像对。
- 已知速度场。
- 已知折射率梯度导致的像素偏移。

流程：

1. 生成无折射扰动粒子图像对。
2. 对第二帧施加速度位移。
3. 对两帧施加折射率像素偏移。
4. 用 OpenPIV / PIVlab 估计速度。
5. 用已知偏移做补偿。
6. 比较补偿前后速度误差。

通过标准：

- 能画速度矢量图。
- 能显示补偿后误差下降，或说明误差不下降的原因。

当前结果：

- observed PIV velocity RMSE 约 0.0101。
- BOST-style compensated velocity RMSE 约 0.0067。
- 95 分位误差从约 0.0240 降到约 0.0072。
- 局部最大误差仍可能残留，原因是折射位移估计加入了噪声，且补偿端点使用近似估计。

当前输出：

- `demo_m3a/results/m3a_compensation_summary.png`
- `demo_m3a/results/m3a_error_profile.png`
- `demo_m3a/results/metrics.csv`

下一步升级：

- 加入真实或合成粒子图像。
- 用 OpenPIV / PIVlab 风格窗口互相关替代向量场 toy。
- 比较三种策略：先校正粒子图像、先估计速度再校正、只做误差传播量化。

## Demo M3B: 4D BOST low-rank toy

适用条件：

师兄希望你靠近 4D BOST，或暂时没有 PIV 数据。

当前已实现版本：

- 路径：`demo_m3b/`。
- 构造 `X-Y-Z-T` 合成折射率体序列：三维 Gaussian / flame sheet 随时间移动和呼吸。
- 每一帧用 sparse-view BOS-like deflection 和逐帧 stack baseline 重构。
- 把逐帧重构展平成 `time x voxel` 矩阵，用 SVD 做低秩时序表示。
- 这不是 ACM TOG 4D BOST 的完整复现，而是低秩时序先验的本科可控子问题。

输入：

- 一个随时间移动或变形的 3D Gaussian flame。
- 时间帧：20 / 50 / 100。

方法：

- baseline：逐帧重构。
- low-rank toy：`n(x,y,z,t) = sum_k a_k(t) * phi_k(x,y,z)`。
- 可选：坐标网络输入 `(x,y,z,t)`。

指标：

- 每帧 L2。
- 帧间平滑度。
- 训练时间。
- rank 对误差的影响。

通过标准：

- 能说明低秩时间先验什么时候减少抖动，什么时候过度平滑。

当前结果：

- framewise baseline mean relative L2 约 0.366。
- low-rank rank 3 mean relative L2 约 0.347。
- temporal smoothness 从约 0.279 降到约 0.177。
- centroid trajectory RMSE 从约 0.0401 降到约 0.0381。
- rank 1 过度平滑且 L2 变差；rank 3 在当前设置下最优；更高 rank 逐步保留更多逐帧噪声。
- 诚实边界：低秩先验能压制时序抖动，但不能自动修正系统性几何偏差或错误 forward model。

当前输出：

- `demo_m3b/results/m3b_4d_summary.png`
- `demo_m3b/results/m3b_rank_scan.png`
- `demo_m3b/results/m3b_temporal_trace.png`
- `demo_m3b/results/metrics.csv`
- `demo_m3b/results/rank_metrics.csv`

## 推荐代码模块

```text
src/
  phantom.py              # 2D/3D/4D ground truth
  ray_geometry.py         # views, rays, sampling
  forward_bos.py          # n field -> displacement
  voxel_baseline.py       # baseline inverse
  nerif_model.py          # coordinate MLP
  train_nerif.py          # training loop
  piv_compensation.py     # optional PIV-BOST toy
  lowrank_4d.py           # optional 4D toy
  metrics.py              # L2, CC, SSIM, PSNR, reprojection
  visualize.py            # slices, curves, vector fields
configs/
  m0_2d.yaml
  m1_3d.yaml
  m2_robustness.yaml
  m3a_piv.yaml
  m3b_4d.yaml
```

## 图表命名规范

```text
results/
  2026-07-xx_m0_2d/
    config.yaml
    metrics.csv
    gt_slice.png
    baseline_slice.png
    nerif_slice.png
    error_map.png
    reprojection_error.png
  2026-07-xx_m2_noise_scan/
    metrics_all.csv
    noise_curve.png
    view_count_curve.png
```

## 一周最小交付

第 1 天：画 BOST 链条图，写 `phantom.py`。

第 2 天：写 2D forward displacement。

第 3 天：写 baseline 或直接做 neural fitting。

第 4 天：加入指标和可视化。

第 5 天：加入视角数扫描。

第 6 天：整理 README 和一页图。

第 7 天：拿一页图去问何远哲。

## 向师兄展示时只展示三张图

1. BOST 链条图。
2. Ground truth / reconstruction / error 对比图。
3. 视角数或噪声鲁棒性曲线。

展示时不要说“我还在学很多东西”。更好的说法是：

我已经把合成数据、forward model、最小重构和指标跑成闭环了。下一步想请师兄帮我判断，应该优先接真实 BOST 数据、PIV-BOST 补偿，还是 4D BOST 的低秩时序子问题。
