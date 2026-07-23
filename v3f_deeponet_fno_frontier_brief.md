# T16 v3f：DeepONet / FNO 同场误差—算力前沿与下一版自研模型决策

## 一句话判决

**当前 49,313 参数固定网格 DeepONet 更轻、更快，但在同一 K=6 synthetic development protocol 下被 44,203 参数 FNO 的质量—时间前沿完整支配。** 下一版自研模型不应把 pooled DeepONet 当主重建器；更合理的路线是保留 FNO 的三维空间混合，把 DeepONet 式 branch/set encoder 降级为采集几何条件编码器，并用严格机制消融证明它确实利用了 BOST acquisition geometry。

这不是“FNO 普遍优于 DeepONet”的结论，也不是论文 superiority。它只冻结了当前小尺寸开发基线，并给下一轮结构设计提供了可证伪方向。

## 实验合同

- 数据：同一 K=6、8×16×16、42-channel ridge/ray/mask/angle/support/coordinate 输入。
- 三个模型种子：`20260901 / 20260902 / 20260903`。
- 基础训练：24 epochs；DeepONet 先在验证集筛选 `5e-4 / 1e-3 / 2e-3 / 4e-3`。
- 延续训练：每 12 epochs 一个 block，到 240 epochs。
- 三种协议：restart Adam + restart cosine；carry continuation-Adam + restart cosine；carry continuation-Adam + long cosine。
- 公平项：相同数据、loss、seed、batch size、block seed 与 batch-order contract。
- 不相同项：参数、FLOPs、每步 wall time 和架构归纳偏置；因此这里只叫 matched attempted epochs，不叫 equal-compute training。
- 时间边界：两条 trajectory 均在同一台 Apple Silicon MPS 上测得，但来自不同日期的训练运行，并非 interleaved fresh-worker benchmark；`4.31×` 差距可作开发诊断，投稿前仍需按 v3e fresh-worker 方式复测端到端 time-to-quality。
- validation 聚合：每个三维场等权；12/12/12/4 的最后小 batch 按样本数加权，不能把四个样本当成十二个样本。
- 选择：学习率、策略、checkpoint 和跨架构赢家全部只看 validation；本轮选择冻结后才计算 dev2 与 clean Q_audit。
- dev2 边界：这里复用的是 v3c/v3d 已查看过的 synthetic development diagnostic，并非整个研究流程从未触碰的新独立审计集；它只能检验本轮冻结比较，不能承担下一轮反复调方向后的确认性结论。
- 统计单位：128 个三维场；三个模型种子先折叠，不能冒充 384 个独立物理样本。

## 学习率与优化协议

DeepONet 全局学习率冠军为 `0.002`：三个种子的 epoch-24 样本等权平均 prefix-best validation L2 为 `0.18615`。四个学习率 × 三个种子的筛选总耗时约 `26.04 s`，其中约 `19.59 s` 属于被淘汰配置；该成本单独报告，没有偷偷塞进或抹去 selected-run curve。

DeepONet 的三条延续策略均达到操作性 plateau：

| DeepONet 策略 | epoch-240 validation L2 | plateau onset | mean train s / seed |
|---|---:|---:|---:|
| restart Adam + restart cosine | 0.17770 | 60 | 20.63 |
| carry Adam + restart cosine | **0.17573** | 72 | 20.51 |
| carry Adam + long cosine | 0.17612 | 96 | 20.52 |

验证集冠军仍是 `carry continuation-Adam + restart cosine`，但它只比 long-cosine 的复用 dev2 field error 好约 `0.08%`，且 field-cluster CI 跨过零，说明 optimizer 微调无法修复当前结构的主要表达瓶颈。

## FNO 与 DeepONet 同场结果

| checkpoint | DeepONet validation L2 / 秒 | FNO validation L2 / 秒 |
|---:|---:|---:|
| 60 | 0.18239 / 5.23 | 0.12617 / 11.60 |
| 120 | 0.17979 / 10.34 | 0.10729 / 23.41 |
| 180 | 0.17749 / 15.40 | 0.09925 / 35.29 |
| 240 | 0.17573 / 20.51 | **0.09414 / 46.46** |

关键不是 FNO 多跑了几秒，而是：

1. FNO 在 epoch 24、约 `4.76 s/seed` 时，样本等权 validation L2 已为 `0.16646`，比 DeepONet epoch 240 更低；达到“超过 DeepONet 最终质量”的时间约快 `4.31×`。
2. DeepONet 的 60/120/180/240 四个固定 checkpoint，在已观测 FNO checkpoints 中全部存在同时更快且误差更低的点，即 `4/4` 被观察到的 error–time Pareto 前沿支配。
3. FNO epoch-240 validation error 相对 DeepONet 低 `46.43%`。
4. 本轮冻结后在复用 dev2 上，FNO 的 field error 平均优势为 `25.50%`，field-cluster bootstrap 95% CI `[24.17%, 26.84%]`，p10 为 `7.24%`，超过 1% 的伤害率为 `0%`。CI 在折叠模型种子后重采样 128 个场，不包含 seed-level 不确定性。
5. 四个域的平均优势均为正：family OOD `9.53%`、IID `47.28%`、joint OOD `9.93%`、noise OOD `35.26%`；三个种子分别为 `25.68% / 24.69% / 26.13%`。

这里的 development gate 通过，含义只是“按 validation 冻结的 FNO 确实稳定超过当前 DeepONet control”。它不开放 blind final，也不提供真实 BOST 或论文级确认性证据。

## DeepONet 的真实优势仍需保留

相对 FNO，当前 DeepONet：

- partial forward FLOPs-v1 为 `0.928×`；
- MPS batch-1 inference p50 为 `0.516×`；
- MPS batch-12 full training-step p50 为 `0.352×`；
- 参数为 49,313，对比 FNO 的 44,203。

所以不能写成“DeepONet 没用”。正确结论是：**它的 branch-trunk 计算路径很高效，但当前 pooled branch + rank-48 global trunk basis 没有把这种效率转化为 K=6 三维重建质量。** 它更适合成为条件编码器、低维控制器或连续查询 refinement，而不是直接替换三维空间混合主干。

## 对下一版自研算法的机制约束

暂定工作名：**GC-SRO（Geometry-Conditioned Spectral Residual Operator）**。这是工程假设，不是原创性结论。

### 保留什么

1. 保留 ridge-calibrated residual formulation，避免网络重新学习稳定的线性逆解。
2. 保留 FNO 主干，负责体素空间中的全局三维混合。
3. 保留 physics loss 与 clean Q_audit，不以单一 field L2 换取投影不一致。

### 新增什么

1. 用共享权重的 per-view branch 编码每个相机的 ray residual、mask、angle 与 calibration descriptor。
2. 用 permutation-invariant set aggregation 得到 acquisition embedding；DeepONet 的 branch 思想只负责这个低维条件。
3. 让 embedding 调制少量 Fourier modes、spectral groups 或 residual gates，而不是再建立完整逐视角 3D decoder。
4. 先使用 zero-init modulation，确保初始输出逐元素等于冻结或可训练的强 FNO。
5. 若真实 geometry 不随 case 变化，立即停止“geometry-conditioned”主张，转向 mask/noise-conditioned robustness 或 learned proximal/data-consistency 方向。

### 必做消融

| 版本 | 目的 | 通过条件 |
|---|---|---|
| FNO-240 | 强基线 | 完整复用 v3d validation champion |
| parameter-matched FNO | 排除只是多参数 | 参数与 GC-SRO 相差不超过 5% |
| static adapter | 排除固定额外容量 | 与 GC-SRO 同结构但条件常量化 |
| geometry shuffle | 检验是否真正使用相机几何 | 打乱 geometry 后优势显著下降 |
| mask-only control | 区分相机缺失与真实几何 | 明确 geometry descriptor 的独立增益 |
| zero/no-condition fallback | 工程安全 | 初始输出与 FNO 最大差异接近数值零 |

GC-SRO 只有在 validation-only selection 后，同时满足 field CI、p10、harm、每域、每 seed、Q_audit 和成本门槛，才进入下一层真实/组内数据。否则保留为负结果，不通过加宽网络追平。

## v3g 已完成 DeepONet 的最低公平补强

论文中不能把当前单一 rank/pool 配置写成“DeepONet 已充分调优”。v3g 已把原计划升级为有上限、三种子完整 sensitivity audit：

- rank：`32 / 48 / 64 / 96`；
- pool：`2×4×4 / 4×4×4 / 8×2×2 / 2×8×4 / 2×4×8` 等合法分配；
- 参数上限：不超过 rank-48 reference 的 `1.5×`，两个越界方案训练前排除；
- `8 架构 × 3 学习率 × 3 种子 = 72` 次固定 24-epoch validation-only screen；
- 只对短程冠军补三优化协议、三种子 240-epoch curve，总搜索成本单列，不根据 dev2 扩表。

结果是 rank-64 短程冠军到 240 epochs 为 `0.176094`，略逊 rank-48 reference `0.175725`，FNO 为 `0.094139`；冻结后复用 dev2 也未通过替换门槛。因此冻结 rank 48 为高效但弱质量的 control。该协议只延长短程冠军，不能外推成“所有 DeepONet 都弱”；完整限制与下一步见 `v3g_deeponet_capacity_audit_brief.md`。

## Gao Youyang 下一步学习顺序

1. 复画 `v3f_architecture_frontier.csv`：横轴分别换成 epoch、秒和 FLOPs-v1，理解三种“公平”的区别。
2. 手写一个最小 DeepONet：branch 输出系数，trunk 输出坐标基，逐点内积成场；解释 rank-48 为何是全局低秩瓶颈。
3. 手写 FNO spectral layer，标出 retained modes、local path 和 residual lift。
4. 实现一个只输出 12–32 维 acquisition embedding 的 set branch，不生成三维场。
5. 先做 static / shuffled / mask-only 三个 toy modulation；在机制未通过前不训练大模型。
6. 向师兄口述本页结论：为什么 DeepONet 单步更快，却在 error–time 前沿上仍被 FNO 支配。

## 请师兄优先审核

1. 是否认可“FNO 做空间主干，DeepONet/set branch 只做 acquisition conditioning”的结构转向？
2. 组内真实 geometry/calibration descriptor 是否逐 case 变化；能否提供矩阵、ray bundle 或相机外参？
3. 毕设阶段优先做 GC-SRO 机制验证，还是先补 DeepONet rank/pool sensitivity？
4. 真实数据的最小可用形式是什么：projection + reference volume、multi-view displacement，还是只有无真值场？
5. 论文更看重 fixed-time quality、最终 quality、Q_audit/data consistency，还是跨 geometry 泛化？

## 证据入口

- 配置：`demo_t16_operator/configs/v3f_deeponet_frontier.json`
- 训练与分析：`demo_t16_operator/run_v3f_deeponet_frontier.py`
- 独立 validator：`demo_t16_operator/validate_v3f_deeponet_frontier.py`
- 学习率筛选：`demo_t16_operator/results/v3f_deeponet_frontier/v3f_deeponet_lr_tuning.csv`
- matched frontier：`demo_t16_operator/results/v3f_deeponet_frontier/v3f_architecture_frontier.csv`
- time-to-target：`demo_t16_operator/results/v3f_deeponet_frontier/v3f_time_to_target.csv`
- dev2 配对统计：`demo_t16_operator/results/v3f_deeponet_frontier/v3f_cross_architecture_pairwise.csv`
- 域与种子：`v3f_cross_architecture_domains.csv`、`v3f_cross_architecture_seeds.csv`
- 机器报告：`v3f_deeponet_frontier_report.json`

## 不能声称

- 不能声称 FNO 普遍优于所有 DeepONet；这里只测试了一个明确配置和一个小型 LR screen。
- 不能声称 GC-SRO 是新算法；最近邻查重和真实 geometry 机制证据尚未完成。
- 不能把相同 epochs 写成 equal compute。
- 不能用 dev2 的通过反向选择架构或超参数。
- 不能声称 synthetic development 结果可迁移到 OERF 真实 BOST。
- blind final、真实 geometry、确认性 superiority 和 NeRIF 端到端比较继续关闭。
