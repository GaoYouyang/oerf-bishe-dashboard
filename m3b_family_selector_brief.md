# M3B 跨形态无真值 rank 选择：给何远哲的一页审核 brief

日期：2026-07-10
用途：请师兄判断“少视角 4D BOST 的可观测时序容量选择”是否值得作为本科毕设子问题，以及下一轮应接真实数据、标定误差还是算子学习。
边界：这是 clean-room straight-ray synthetic benchmark，不是 TDBOST 复现，也不证明真实 OERF 泛化。

## 一句话题目

> 面向少视角 4D BOST 的跨流场形态、无真值时序 rank 选择：融合奇异谱、support-view 数据一致性与时序物理统计。

## 为什么继续做这一题

旧 M3B 已发现：固定 rank 3 在单一 phantom 的 rank×noise×views×dynamics 扫描中平均稳健，但不是逐格最优。旧结论有两个缺口：

1. 只有一个 phantom family，无法判断 rank 规律是否依赖流场形态；
2. oracle rank 使用 ground-truth field L2，真实实验不能部署。

## 新实验设计

- 4 个 synthetic morphology family：`blobs_sheet`、`vortex_ring`、`expanding_shell`、`jet_filaments`。
- 3 类时序：smooth、chirp、transient。
- 4 个噪声层：0、0.07、0.14、0.28。
- 3 个 support-view 数：3、5、7。
- 6 个配对观测噪声 seed。
- 7 个容量候选：rank 1/2/3/5/8/12/18，其中 rank 18 等价于不做低秩压缩。
- 共 144 个不含 seed 的环境、864 个观测格、6,048 个候选重建。
- 外层严格 leave-one-phantom-family-out；测试 family 不进入训练。每个外层 fold 内再用其余 family 选择 ridge 正则强度。

测试时 selector 只能看到：rank/奇异谱、support 重投影残差、时序梯度/曲率、mass-trace 曲率、负值比例、观测高频 proxy、view count；带 held-out 版本额外看到一台未参与重建相机的重投影残差。`family`、真值 field error、oracle rank、真实 noise label 都禁止进入特征。

## 核心结果

| 方法 | Mean oracle regret | P95 | 1% 内覆盖 | 相对固定 rank 3 的平均 field 改善 |
|---|---:|---:|---:|---:|
| 固定 rank 3 | 1.561% | 5.635% | 53.8% | 0 |
| 训练 family 上选一个固定 rank | 0.865% | 5.878% | 77.1% | 0.658% |
| 95% singular-energy threshold | 1.484% | 10.593% | 66.6% | 0.033% |
| 单独最小化 support residual | 0.770% | 3.669% | 78.4% | 0.748% |
| 单独最小化 held-out residual | 2.423% | 9.467% | 47.3% | -0.880% |
| LOFO ridge，不用 held-out view | **0.252%** | 1.267% | 92.4% | 1.259% |
| LOFO ridge，加入 held-out view | **0.210%** | **1.054%** | **94.4%** | **1.301%** |

与固定 rank 3 相比，带 held-out 的 selector 在 78.9% 观测格取得更低 field L2。六个 seed 的 clustered 95% CI 为 mean regret ±0.0136 percentage point；它只覆盖观测噪声变化，不覆盖真实工况不确定度。

## 三个最重要的研究判断

### 1. morphology 是 rank 选择的一阶变量

864 个 oracle 选择中：rank 2/3/5/8/12/18 分别出现 77/116/230/245/124/72 次。`blobs_sheet` 主要需要 rank 2-5；`vortex_ring` 主要需要 rank 8-18。固定 rank 3 在 vortex-ring 上 mean regret 为 2.63%，因此旧单 phantom 结论不能外推。

### 2. 不能把 held-out residual 直接当 field quality

单独最小化 held-out residual 的 mean regret 为 2.42%，最坏超过 10%；其全局 Pearson 相关只有 0.037，逐格 pairwise rank-order accuracy 为 65.0%。support residual 的逐格 Pearson 为 0.901、pairwise accuracy 为 79.6%，但单独最小化它仍倾向 full rank，说明“数据一致性”还需要容量和时序先验共同约束。

### 3. 多特征互补是真正有效部分

只用 capacity+spectrum、support 或 temporal feature 的 mean regret 分别为 0.789%、0.614%、0.668%；组合后降到 0.252%。额外 held-out view 只再降低 0.042 percentage point。当前 synthetic 证据因此不支持“为了选 rank 必须专门留一台相机”，但支持把 held-out 作为风险校验而非单独目标。

## 可以发展成论文的模型假设

暂定名：**Observable Capacity Selector for 4D BOST (OCS-BOST)**，仅为工作名，不主张命名或方法原创性。

1. 共享 4D reconstruction backbone 产生多容量候选或可截断 tensor factors。
2. 从 support data consistency、factor spectrum、temporal derivatives、integral traces 形成可观测特征。
3. selector 输出容量分布和不确定度，不只输出一个 rank。
4. 当不确定度高或所有候选违反数据/物理门槛时，回退 full rank 或触发 per-instance NeRIF refinement。
5. 训练采用 leave-one-flow-family / leave-one-geometry-out；测试不使用 field truth。

真正的创新不能写成“用 ridge 选 rank”。可投稿升级点应是：**针对 BOST 观测算子，把 support 一致性、时序物理量与可截断 4D 表示联合成可部署的容量控制和拒答机制**。

## 当前最强反例

- `blobs_sheet + transient + noise 0.07 + 5 views`：no-held-out selector 常选 rank 2，oracle 为 rank 5，最坏 regret 2.58%。
- `jet_filaments + chirp + noise 0.28 + 7 views`：with-held-out selector 选 rank 12，oracle 为 rank 5，最坏 regret 2.84%。
- `vortex_ring + chirp + noise 0 + 7 views`：固定 rank 3 的 regret 7.67%，oracle 为 full rank 18。

下一版模型应专门解释这些 failure cells，而不是只继续降低 pooled mean。

## 请师兄判断的 8 个问题

1. 组内 4D BOST/TDBOST 是否存在可调 rank、plane factor 数、feature width 或其他容量超参数？
2. 实际工作中这些容量是人工固定、按 case 调，还是有 validation/test view？
3. 能否提供一小段 18-30 帧、3/5/7/9 views 的脱敏 deflection 或 reconstruction tensor？
4. 是否能保留一台 camera 只做 validation；如果不能，support residual 是否可从训练视角计算？
5. 真实结果必须保真的量是 field/test-view、峰值、频率、mass/integrated density，还是火焰/激波位置？
6. `blobs / ring-front / expanding front / jet-filament` 这四种 synthetic morphology 是否接近组内 failure mode；应替换成哪些形态？
7. 下一轮优先做 leave-one-geometry-out、bias/sync/exposure blur，还是接真实数据？
8. 这条线适合作为毕业设计主线，还是更适合作为 T16 neural operator 的 temporal capacity-control 子模块？

## 可复现入口

- 渲染页：`m3b_family_selector_dashboard.html`
- 配置：`demo_m3b/configs/family_selector.json`
- 正式脚本：`demo_m3b/run_m3b_family_selector.py`
- 结果验证器：`demo_m3b/validate_family_selector_results.py`
- 原始候选表：`demo_m3b/results/family_selector/family_selector_raw.csv`
- selector 决策表：`demo_m3b/results/family_selector/family_selector_selected.csv`
- 特征消融：`demo_m3b/results/family_selector/family_selector_ablation_summary.csv`
- 完整模型/外层 fold 审计：`demo_m3b/results/family_selector/family_selector_report.json`
- 核心表 SHA-256：`demo_m3b/results/family_selector/family_selector_checksums.sha256`

## 解释边界

- 四种 morphology 都是人为 stress test，不是四个真实流动工况。
- LOFO 证明的是 synthetic morphology transfer，不是 synthetic-to-real generalization。
- 六个 seed 只改变观测噪声；不能把 864 格写成 864 个独立实验样本。
- 全局 affine field alignment 和 oracle rank 只用于离线评价。
- 目前没有 curved ray、真实 calibration、同步误差、曝光模糊、mask、相机 PSF 或真实流动守恒约束。
