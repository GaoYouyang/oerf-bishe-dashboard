# QC-SNCO 红队审计与论文恢复条件

更新：2026-07-11  
对象：T16 算子学习 × 少视角 BOST 三维重构  
当前状态：**本科保底线继续；QC-SNCO 论文升级线暂停扩模型**

> **2026-07-11 v2d 复核结果：**致命对照已实现为 controlled inference audit。固定 Q_fit=80°、Q_audit=60°；19,008 条方法-体场-种子结果在按体场折叠种子后做 20,000 次五来源域等权 bootstrap。当前 checkpoint 路径在 K=4/6/8 的同重建预算门槛为 0/3：fixed learned correction 相对 `S∪Q direct` 分别为 `-20.17% / -13.07% / -15.11%`。因此停止扩 QC-SNCO，并转向 direct inverse operator / operator warm-start NeRIF。由于训练 mask 未匹配、88 fields 已参与开发且 forward 仍为 linear synthetic stack，状态严格记为 `PILOT_ONLY_CURRENT_CHECKPOINT_PATH_FAILS`。详见 [v2d 判决页](operator_fair_budget_dashboard.html) 与 [师兄简报](fair_camera_budget_review_brief.md)。

## 一句话判决

现有实验证明：

1. support-fit 是强而可解释的物理基线；
2. support nullspace 中存在很大的 truth-oracle headroom；
3. 直接学满幅 null correction 会失败；
4. adaptive query 可以限制一个很弱修正方向的破坏。

它还没有证明：

> 网络已学会 support cameras 缺失的三维信息，或者“多留一个 query view 做标定”比“把同一个 view 直接加入重建”更值得。

因此 <code>+0.746%</code> 只称为 **adaptive-query synthetic pilot**，不称为 closure passed、新算法已成立或固定一台验证相机有效。

## 现有数字的正确语义

| 证据 | 数字 | 能说什么 | 不能说什么 |
| --- | ---: | --- | --- |
| independent support-fit | `79.17%` 单元达到/优于最佳端点 | 两专家的连线上存在可观测互补 | 双专家已胜过等参数单宽模型/ensemble |
| exact truth-null oracle | `+38.626%` | support-fit 的误差有大量不被 support 相机观测的结构成分 | 存在可部署的学习方向 |
| hard-null learned corrector | `-0.126%` | 硬约束可以守住 support，但学习方向失败 | 只要零空间一致就会改善 field |
| field-truth 最优 alpha 作用于 learned direction | `+1.212%` | 当前方向本身的上界很低 | query alpha 是主要瓶颈 |
| adaptive one-query line search | `+0.746%` | 候选 query 可对弱方向做轻微幅度约束 | 一台预先固定的相机可部署地改善 BOST |
| support leakage | `<2.3e-9` | 同一个 synthetic 线性矩阵下的代数闭合正确 | 真实 ray geometry、标定与 forward mismatch 下仍一致 |

learned direction 的 field-oracle 收益与 exact truth-null headroom 的均值比约为 `1.212 / 38.626 ≈ 3.1%`。在方向质量本身没有显著上升前，继续改 alpha router 不是当前主矛盾。

## 六个致命问题

### F1. 总相机预算不公平

当前 `S support + 1 query` 没有与以下方法比较：

- 同样总数的 `S∪Q` SIRT/TV/Tikhonov；
- 同样总数的 `S∪Q` support-fit；
- 同样总数的 `S∪Q` NeRIF/NIRT；
- 同样总数的 `S∪Q` direct FNO/U-Net。

若不做这组对照，改善只说明“多一个测量有用”，无法证明相机分工或 query calibration 有附加价值。

### F2. one-query 不是预先固定一台相机

`demo_t16_operator/run_query_calibrated_nullspace.py` 会按每个样本的修正投影能量，从多个未用角中选一个视角。对瞬态多相机实验，这意味着：

- 要么事先安装所有候选相机；
- 要么无法在瞬态场出现后再移动相机。

后续必须分开报告：预先固定视角、随机视角、最大角度间隔、可微/Fisher 设计与当前 adaptive energy policy。

### F3. query 拟合和 held-out 验收未完全分离

one-query 的 query 观测同时影响 alpha 拟合和当前 held-out 评估；all-query 更是用全部 withheld views 拟合后再在同一集合验收。正式设计应拆成：

```text
S_support  -> initial reconstruction / support-fit
Q_fit      -> choose direction or fit alpha
Q_audit    -> final held-out reprojection only
```

`Q_audit` 不得参与选角、方向选择、alpha 拟合或超参数选择。

### F4. inverse crime 与真实 forward 缺失

当前数据生成、support consistency、null projection、query selection 和验收使用同一类线性矩阵。`demo_t16_operator/bost_physics.py` 的现有 toy 实际是逐 `z` 独立的二维方向梯度投影/反投影堆栈，不包含：

- 真实 3D cone rays 和跨层耦合；
- finite aperture / depth of field；
- ray bending / nonlinear forward；
- calibration mismatch；
- displacement extraction 的空间相关偏差；
- mask、background texture、震动和光学畸变。

下一轮至少需要一个与训练生成器独立的高分辨率/cone-ray forward，并优先接入开放真实 BOS tomography 数据。

### F5. 统计单元和反复查看 test 问题

264 个 sample-seed cells 的底层是 88 个独立测试体场 × 3 个模型初始化。不能把 264 当作 264 个独立物理样本，也不能把 `3 seeds × 5 domains = 15` 当作 15 个独立研究。

统计建议：

1. 先在同一体场内跨 model seeds 汇总；
2. 以体场或真实独立采集为 cluster 做 paired bootstrap/mixed-effects；
3. 预注册一个 primary estimand；
4. 同时报告绝对差、相对差、median、p10、CVaR 和 harm rate；
5. 当前反复用于 v1-v2c 开发的 test set 降级为 development set，新建真正锁定的 final test。

### F6. 新颖性不能来自现有组件名的组合

下列组件都有强相邻方法：

- hard-null correction：Deep Null Space Learning；
- range/null 双网络：Siamese Cooperative Learning；
- measurement split/self-supervision：SSDU、Noise2Inverse；
- physics reconstruction + learned correction：MoDL、Neural Correction Operator；
- scalar coefficient elimination：variable projection / least squares。

因此可能保留的论文贡献只能来自：

> **BOST 特有的、成本约束下的相机分工、真实 ray geometry 和可重复实验结论。**

## 必须补的强基线

### 相机预算基线

1. `S only` SIRT/TV/Tikhonov/support-fit/NeRIF；
2. `S∪Q` 直接重建，总相机数与新方法一致；
3. `S + Q_fit correction + Q_audit evaluation`；
4. 同一个 query 重复曝光以提高 SNR 的成本对照。

### 不学习的 query-null 对照

至少实现一个线性/局部线性化更新：

```text
delta = P_N A_Q^T (A_Q P_N A_Q^T + lambda I)^-1 r_Q
```

它直接用 query residual 在 support nullspace 中求解，可检验 learned direction 是否比普通数值更新有价值。

### 方向对照

- random null direction；
- PCA / training-error null direction；
- `P_N(x_res-x_abs)` branch-difference direction；
- learned direction；
- truth direction / exact null oracle（只用于上界）。

### 容量对照

- shared dual；
- independent dual；
- 等总参数 single-wide FNO；
- 两个同类专家的 independent ensemble；
- matched U-Net/CNO；
- short reconstruction + Neural Correction Operator-style corrector。

## 最小消融矩阵

| 轴 | 最小水平 | 主图/主表 |
| --- | --- | --- |
| 相机预算 | S-only / S+Q correction / S∪Q direct | field-front-reprojection 和成本 Pareto |
| query 策略 | fixed / random / max-gap / adaptive energy | 收益分布与固定视角稳定性 |
| 方向 | random / branch / PCA / learned / truth | field-oracle 上界与 query 可辨识性 |
| forward | matched A / independent high-res A / cone-ray / calibration mismatch | leakage、field harm 与 Q_audit residual |
| projector | none / exact / truncated-SVD / tangent/Krylov | consistency-计算成本 |
| alpha | 0 / 1 / binary / ridge LS / clipped-unclipped | p10、CVaR、harm rate |
| supervision | field-only / query-only / field+query | 区分方向学习和幅度标定 |
| randomness | field seed / noise seed / model seed | 方差分解 |

## 论文升级的恢复条件

| 项目 | 恢复论文线 | 停止/降级线 | 当前 |
| --- | --- | --- | --- |
| learned direction 上界 | field-oracle gain `>= max(5%, 2×真实重复性 CV)` | 低于门槛 | `1.212%`，暂停扩模型 |
| 同预算收益 | 胜最强 `S∪Q` 直接基线，cluster CI 下界 > 0 | 任一关键域失败 | 未测 |
| 固定 query | 预先固定一台保留 >=80% adaptive 收益 | 只有样本级 adaptive 有效 | 未测 |
| 真实迁移 | >=3 工况 × 5 独立采集，Q_audit 独立，收益 > 2×重复性噪声 | query 与真实改善不对齐 | 未测 |
| reliability | error ranking Spearman >=0.5 且 harm AUROC >=0.8 | <0.3 或 <0.7 | 未测 |
| NeRIF warm-start | wall time 降 >=20% 或失败率降 >=30%，最终误差非劣 <=1% | 不满足 | 未测 |

## 现在可以稳定完成的本科成果

1. 线性 synthetic BOST 的完整 forward/inverse-operator benchmark；
2. physics lift、U-Net、Residual/Absolute FNO、support-fit 公平对照；
3. family/view/noise/geometry OOD 和负结果地图；
4. support-null identifiability 与 learned-null failure 机制分析；
5. 同相机预算的 query VOI 实验；
6. 开放真实 BOS 数据接口；
7. 可复跑代码、checksums、逐样本统计和三维展示。

这些足以形成一个强本科毕设。若 QC-SNCO 的公平对照失败，可把它作为“约束正确但方向学习失败”的负结果章，不会破坏毕设完整性。

## 当前推荐的执行主线

### 阶段 A：同相机预算的物理粗重建 + 受约束修正

这是当前应执行的主实验，也是 QC-SNCO、operator warm-start NeRIF 和 geometry-aware operator 的共同基础。

### 阶段 B：operator warm-start NeRIF

如果能获得 NeRIF forward/loss 和真实样例，比较 random/SIRT/operator 初值的总 wall time、最终 residual 和失败率。它直接对齐何远哲主线，比继续增加 query router 更值得优先评估。

### 阶段 C：恢复或终止 QC-SNCO

只有阶段 A 的同预算、独立 audit 和真实 forward 通过，且 learned direction 上界显著提高后，才恢复 QC-SNCO 作为论文模型。

## 请何远哲优先判断

1. “算子学习”是 projection-to-volume inverse operator，还是 3D/4D evolution operator？
2. 组内实验的总相机预算和实际可用相机数是多少？
3. 能否把视角分为 support、Q_fit 和独立 Q_audit？
4. 能否提供最小 displacement + ray/camera geometry + mask + calibration + baseline reconstruction 数据包？
5. 是否有 NeRIF/NIRT/TDBOST 的 forward projector 可作独立强基线？
6. 真实无三维真值时，组内最认可的验收量是 held-out reprojection、front location、density integral、PIV correction 还是重复性？

## 直接相关的方法先例

- Schwab, Antholzer, Haltmeier, *Deep Null Space Learning for Inverse Problems*, Inverse Problems 2019, [arXiv](https://arxiv.org/abs/1806.06137).
- Molinaro et al., *Neural Inverse Operators for Solving PDE Inverse Problems*, ICML 2023, [PMLR](https://proceedings.mlr.press/v202/molinaro23a.html).
- Lunz et al., *On Learned Operator Correction in Inverse Problems*, SIAM Journal on Imaging Sciences 2021, [DOI](https://epubs.siam.org/doi/10.1137/20M1338460).
- Bhat, Chen, Wang, *Neural Correction Operator*, Journal of Computational Physics 2026, [arXiv](https://arxiv.org/abs/2507.18875).
- Aggarwal, Mani, Jacob, *MoDL*, IEEE Transactions on Medical Imaging 2019, [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC6760673/).
- He et al., *Neural Refractive Index Field*, Physics of Fluids 2025, [arXiv](https://arxiv.org/abs/2409.14722).
- Molnar et al., *Open-source BOS tomography dataset of high-speed flow*, Experiments in Fluids 2026, [publisher](https://link.springer.com/article/10.1007/s00348-026-04189-z), [data DOI](https://doi.org/10.26208/1VE2-5C19).

这些 CT/EIT/MRI/PDE 论文只提供方法先例和对照设计，不能将其性能数字或结论直接移植为 BOST 证据。
