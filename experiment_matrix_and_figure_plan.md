# 实验矩阵与论文图表计划

用途：把 M0-M3B、未来 NeRIF/PyTorch、真实/开源数据迁移转成正式论文可用的实验矩阵。每个实验都必须回答一个明确问题，生成一张可解释图，支持论文某一节，而不是“跑了很多参数但不知道说明什么”。

最后更新：2026-07-06。

---

## 1. 总实验叙事

毕业论文的实验不应写成“我跑了几个 demo”。建议叙事顺序是：

1. **闭环存在性**：M0 证明 `折射率场 -> 位移 -> 重构 -> 指标` 可以跑通。
2. **三维扩展**：M1 证明少视角三维栈重构中，坐标/隐式先验有一定帮助，但不是总赢。
3. **优势边界**：M2 系统扫描 view/noise/capacity，回答 NeRIF-style prior 什么时候有用。
4. **实验升级**：M3A 或真实 PIV-BOST 说明折射率重构如何服务速度误差补偿。
5. **时序升级**：M3B 或 4D 子问题说明低秩/时序先验如何减少逐帧抖动。
6. **数据迁移**：open BOS 或 OERF 样例数据说明 pipeline 不只服务 synthetic toy。

核心观点：

> 本课题的创新点不是声称 neural field 永远优于传统方法，而是建立一个可复现的 BOST/NeRIF 实验框架，并量化隐式先验在少视角、噪声和真实数据接口条件下的适用边界。

---

## 2. 实验矩阵总表

| 实验编号 | 问题 | 数据 | 方法对比 | 变量 | 核心指标 | 论文位置 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | BOST 最小闭环能否跑通？ | 2D synthetic phantom | baseline vs coordinate-field | views | rel L2, CC, SSIM proxy | 第 4.1-4.3 节 | 已完成 M0 |
| E1 | 三维少视角下坐标正则化是否有帮助？ | 3D stack synthetic | stack baseline vs 3D coord-regularized stack | views | rel L2, CC | 第 4.3-4.4 节 | 已完成 M1 |
| E2 | 隐式先验的优势边界在哪里？ | 3D synthetic | baseline vs coord regularizer | views x noise | improvement heatmap, winner | 第 4.5 节 | 已完成 M2 |
| E3 | 表示容量是否越大越好？ | 5-view noisy synthetic | feature count scan | num_freq | rel L2, improvement | 第 4.5 节 | 已完成 M2 |
| E4 | PyTorch NeRIF-style forward loss 能否替代 post-FBP regularizer？ | 3D synthetic | coord regularizer vs MLP forward-loss | loss / sampling | rel L2, reprojection | 第 3-4 章 | 待做 |
| E5 | 开源 BOS 数据能否跑通数据接口？ | open BOS flight body | open data loader + baseline | 70->9/5 views | reprojection, loader completeness | 第 5 章 A/B 路线 | 待做 |
| E6 | BOST-style 折射补偿能否降低 PIV 速度误差？ | synthetic velocity/refraction | observed vs compensated | deflection noise | velocity RMSE, p95 | 第 5 章 B 路线 | 已完成 M3A |
| E7 | 低秩时序先验能否减少 4D 逐帧抖动？ | 4D synthetic | framewise vs low-rank | rank | mean L2, temporal smoothness | 第 5 章 C 路线 | 已完成 M3B |
| E8 | 真实 OERF 数据能否接入同一接口？ | OERF sample | baseline / NeRIF-style / report | view geometry, mask | reprojection, visualization | 第 5 章 A 路线 | 等师兄数据 |

---

## 3. 已有结果如何写

### E0 / M0：2D BOST toy

现有数据：

- 9 视角 baseline relative L2: `0.1133`。
- 9 视角 coordinate-field relative L2: `0.1079`。
- 3/5/7 视角下 coordinate-field 当前不稳定，9/13 视角才略优。

推荐图：

- `demo_m0/results/m0_summary.png`
- `demo_m0/results/view_count_curve.png`

推荐图注：

> 图 4-1 二维 BOST toy 的最小闭环结果。合成折射率场经多视角 BOS-like deflection 观测后，分别使用传统 baseline 和坐标场反演重构。该实验用于验证数据生成、反演、误差计算和图表输出流程。

推荐结论：

> M0 不能证明 neural field 在所有 sparse-view 条件下更优，只能证明闭环跑通；低视角下的失败正好引出三维和鲁棒性扫描。

### E1 / M1：3D stack sparse-view

现有数据：

- 5 视角 baseline relative L2: `0.2571`。
- 5 视角 coordinate-regularized relative L2: `0.2344`。
- 3/5 视角坐标正则化有帮助，7/9/13 视角 baseline 更强。

推荐图：

- `demo_m1/results/m1_volume_summary.png`
- `demo_m1/results/m1_view_count_curve.png`

推荐结论：

> 三维坐标先验在少视角伪影明显时有帮助，但当视角数增加且噪声较低时，传统 stack baseline 可获得更低误差。因此本课题应关注“适用区间”而不是单点胜负。

### E2/E3 / M2：鲁棒性扫描

现有数据：

- 3/5 视角所有 noise 设置中，coordinate regularizer 胜出。
- 7/9 视角所有 noise 设置中，baseline 胜出。
- feature count 40/80 过度平滑，120 以后开始超过 5-view noisy baseline。

推荐图：

- `demo_m2/results/m2_noise_view_scan.png`
- `demo_m2/results/m2_improvement_heatmap.png`
- `demo_m2/results/m2_capacity_scan.png`

推荐结论：

> 隐式/坐标先验的收益与视角数、噪声和表示容量耦合。少视角条件下它可缓解部分伪影；视角数充分时，传统 baseline 可能更稳定。容量不足会过度平滑，容量提高后才出现收益。

### E6 / M3A：PIV-BOST compensation toy

现有数据：

- observed PIV RMSE: `0.0101`。
- compensated RMSE: `0.0067`。
- p95 error: `0.0240 -> 0.0072`。

推荐图：

- `demo_m3a/results/m3a_compensation_summary.png`
- `demo_m3a/results/m3a_error_profile.png`

推荐结论：

> BOST-style deflection compensation 能降低速度场层面的平均误差和高分位误差，但局部最大误差仍受折射位移估计噪声和端点近似影响。真实 PIV-BOST 升级需明确补偿发生在图像层、互相关层还是速度矢量层。

### E7 / M3B：4D low-rank toy

现有数据：

- framewise baseline mean L2: `0.3657`。
- low-rank rank 3 mean L2: `0.3471`。
- temporal smoothness: `0.2788 -> 0.1767`。
- rank 1 最平滑但误差高，rank 3 在误差和平滑之间较均衡。

推荐图：

- `demo_m3b/results/m3b_4d_summary.png`
- `demo_m3b/results/m3b_rank_scan.png`
- `demo_m3b/results/m3b_temporal_trace.png`

推荐结论：

> 低秩时序先验能减少逐帧抖动，但 rank 过低会过度平滑，rank 过高又接近逐帧噪声。它不能自动修正系统性几何或 forward model 偏差。

---

## 4. 待做实验优先级

### P0：必须做

| 实验 | 原因 | 最小交付 |
| --- | --- | --- |
| E4 PyTorch NeRIF-style forward loss | 当前 M1/M2 仍是 post-FBP coordinate regularizer，需要更贴 NeRIF | 一张 MLP forward-loss vs M1 regularizer 对比图 |
| E5 open BOS data loader | 无组内数据时的保底公开验证 | 一个 open BOS manifest + view subset 可视化 |
| E8 OERF data manifest dry run | 即使没数据，也能证明你知道要哪些字段 | 填一份假数据/空路径 manifest 并通过 `--allow-missing` 校验 |

### P1：根据师兄方向选择

| 实验 | 适用条件 | 最小交付 |
| --- | --- | --- |
| PIV 粒子图像互相关版 M3A | 师兄更希望贴 PIV-BOST | synthetic particle pair + OpenPIV/PIVlab 参数表 |
| 4D rank/noise/frame 扫描 | 师兄更希望贴 4D BOST | rank x noise heatmap |
| NeDF/NRIP concept comparison | 师兄关注外部竞品 | 表示方式对比表，不一定复现 |

### P2：有时间再做

| 实验 | 原因 |
| --- | --- |
| 完整真实相机几何 | 需要组内标定和师兄确认 |
| stereo-PIV compensation | 本科风险偏高 |
| 70-view open BOS 完整重构 | 数据和代码量可能较大，可先做 9-view subset |

---

## 5. 论文图表编号建议

| 编号 | 图/表 | 文件 | 支撑问题 |
| --- | --- | --- | --- |
| 图 1-1 | OERF 方向与本文位置 | `figures/oerf_position_map.png` | 为什么这个题贴课题组 |
| 图 1-2 | BOST 物理链条 | `figures/bost_physical_chain.png` | 流体变量如何进入 BOS 观测 |
| 图 3-1 | NeRIF-style pipeline | `figures/nerif_pipeline.png` | 方法框架 |
| 图 4-1 | M0 2D summary | `demo_m0/results/m0_summary.png` | 最小闭环 |
| 图 4-2 | M0 view-count curve | `demo_m0/results/view_count_curve.png` | 视角数影响 |
| 图 4-3 | M1 volume summary | `demo_m1/results/m1_volume_summary.png` | 三维 sparse-view 重构 |
| 图 4-4 | M1 view-count curve | `demo_m1/results/m1_view_count_curve.png` | 3D 视角数边界 |
| 图 4-5 | M2 noise-view scan | `demo_m2/results/m2_noise_view_scan.png` | 视角/噪声鲁棒性 |
| 图 4-6 | M2 improvement heatmap | `demo_m2/results/m2_improvement_heatmap.png` | 何时 coord prior 胜出 |
| 图 4-7 | M2 capacity scan | `demo_m2/results/m2_capacity_scan.png` | 表示容量边界 |
| 图 5-1 | M3A PIV compensation | `demo_m3a/results/m3a_compensation_summary.png` | PIV-BOST 升级可行性 |
| 图 5-2 | M3B rank scan | `demo_m3b/results/m3b_rank_scan.png` | 4D 低秩时序边界 |
| 表 4-1 | M0/M1/M2 指标表 | CSV 汇总 | baseline vs coord-field |
| 表 5-1 | A/B/C 路线对照 | `topic_decision_matrix.md` | 定题取舍 |

---

## 6. 实验失败时怎么写

失败结果不要删。可以按以下方式解释：

| 失败现象 | 不好的写法 | 好的写法 |
| --- | --- | --- |
| neural field 不如 baseline | 模型没调好 | 在视角充分、噪声较低时，传统 baseline 的投影信息足以约束重构，坐标先验可能过度平滑 |
| 少视角下误差很大 | 结果不好 | 少视角 BOST 本身是病态反问题，需要正则、先验或真实几何约束 |
| 低秩 rank 1 太平滑 | low-rank 失败 | rank 过低会牺牲空间细节换取时间平滑，存在 rank trade-off |
| PIV 补偿局部误差不降 | 补偿无效 | 局部误差受折射位移估计噪声、端点近似和速度梯度影响 |
| open BOS 与 OERF 不一致 | 数据不相关 | open BOS 用于 pipeline 预演，不替代火焰 BOST 物理结论 |

---

## 7. 实验登记字段

每次跑实验都要登记：

- `experiment_id`
- `date`
- `dataset`
- `method`
- `view_count`
- `noise_level`
- `main_metric`
- `best_figure`
- `claim_supported`
- `failure_or_boundary`
- `next_action`

模板见：`experiment_templates/experiment_registry_template.csv`。
