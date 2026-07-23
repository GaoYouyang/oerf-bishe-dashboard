# 面向少视角 BOST 的视角可靠性路由双分支神经算子

状态：共享 v1、独立专家 v2a、matched null v2b、adaptive-query v2c 与 controlled-budget v2d 已完成；当前 QC-SNCO checkpoint 路径 0/3 通过并停止扩模型

工作名：Query-Calibrated Support-Null Correction Operator，简称 QC-SNCO（暂用，不作原创性宣称）
应用目标：从多视角 BOS 位移/投影快速重建三维折射率、密度或温度相关场

> 当前 <code>+0.746%</code> 来自逐样本从多个候选角中选 query 的 synthetic pilot，不是固定一台相机的部署结论。下一轮先做同总相机预算、Q_fit/Q_audit 隔离、non-learning query-null update 和独立 forward。完整判决见 `qc_snco_red_team_audit.md`。

> v2d 已完成上述 controlled inference 对照：fixed learned correction 相对 S∪Q direct 在 K=4/6/8 为 `-20.17%/-13.07%/-15.11%`。因此本提案作为机制性负结果与历史推导保留；当前主线改为 direct inverse operator + operator warm-start NeRIF。训练 mask 未匹配，故不作最终算法否定。详见 `fair_camera_budget_review_brief.md`。

## 0. 2026-07-10 原型检查点

- 首版共享双输出 FNO：43,485 参数，3 个优化种子，64 个基础训练样本扩展为 128 个固定 support/query 变体。
- 等权双分支在 IID field 上为 0.2577，优于 Residual 0.2629 与 Absolute 0.2612，说明融合有可用互补。
- query-trained MLP 在五域中的 Residual 权重始终约 0.50，field 选择准确率 40.5%，没有学会可部署路由。
- 仅用 support views 求解的闭式融合把五域 field oracle regret 从等权的 0.197% 降到 0.014%，且 75.8% 的样本-种子单元达到或优于两个端点专家。
- 当前只能说明“物理一致性锚点值得继续”，不能说明 SC-DBNO 已经成立；线性合成 forward operator、共享专家与小尺寸数据都是强边界。

## 0.1 v2a-v2c 新检查点

### v2a：独立专家与 support-fit

- 两个独立 FNO 专家共 86,823 参数，平均 branch disagreement 从共享 v1 的约 0.119 提高到约 0.190。
- support-fit 在 264 个样本-种子单元中，79.17% 达到或优于较好端点；平均相对端点 oracle regret 为 `-1.364%`。
- 容量约翻倍；等参数单宽 FNO 与 independent ensemble 仍是强制对照。

### exact null oracle：存在性上界

- 独立 support-fit 剩余误差中平均 59.459% 能量位于 declared support nullspace。
- truth-oracle null correction 的平均 field improvement 为 38.626%，最弱域仍为 23.580%，264/264 单元都改善。
- support projection 最大变化 5.51e-16；这只证明结构性 headroom，不是可部署方法。

### v2b：matched learned corrector

- free/null 两个 13,105 参数 corrector 在每个种子使用相同初始化。
- free correction 总体 field improvement 为 -32.06%，且一个种子明显失稳。
- hard-null correction 把最大 support leakage 压到 2.59e-8，但总体 field/held-out improvement 为 -0.126%/-0.412%。
- 结论：硬零空间约束是稳定器和一致性保证，不足以让网络学到正确方向。

### v2c：query-calibrated amplitude

网络输出 support-null direction `d_N`；当前实现按样本从多个未参与 support reconstruction 的候选 query 角中选视角，并闭式求：

```text
alpha_Q = clip(<A_Q d_N, y_Q - A_Q x_S> / ||A_Q d_N||^2, 0, 1)
x_hat = x_S + alpha_Q d_N
```

- one-query line search：总体 field improvement `+0.746%`，优于 base 比例 52.65%；
- all-query line search：`+0.956%`，优于 base 比例 65.53%；
- 3 seeds × 5 domains 的 15 个汇总均值为正，但底层只有 88 个独立测试体场，model seeds 不当作额外独立样本；
- 最大 support leakage 2.26e-9；
- field-truth oracle alpha 只有 `+1.212%`，说明当前 learned direction 本身仍较弱。

v2c 只保留为待证假设，不继续加大 router/corrector。下一轮核心是：固定总相机数，比较 `S only / S∪Q direct / S+Q correction`；拆分 Q_fit 与 Q_audit；比较 fixed/random/adaptive query；再接独立 cone-ray/高分辨率 forward 和真实数据。

## 1. 研究问题

给定相机几何 `G={G_v}` 和部分视角观测 `y_S={y_v | v in S}`，学习跨样本逆算子：

```text
F_theta: (y_S, G_S, quality_S) -> n_hat(x, y, z)
```

它不同于 NeRIF 的逐样本坐标场优化：算子在多个训练样本上离线学习，测试样本一次前向即可得到体场；随后可选择用 NeRIF/forward loss 做少量 test-time refinement。

当前 T16 合成实验显示：简单 gate 与从零学习的 query router 都未处理好工况翻转；support-view 闭式融合、hard null consistency 和 query-camera line search 则形成了连续证据。因此研究问题进一步收缩为：**能否在真实 BOST 几何下，用最少额外 query information 稳定校准少视角零空间修正，并证明它相对等容量模型、传统重建和 NeRIF 的价值。**

## 2. 核心假设

**H1：** BOST 的粗物理反演不是恒定可信的先验，其可靠性随视角覆盖、角度缺口、位移质量、遮挡和几何误差变化。

**H2：** 与其让一个标量缩放同一 residual backbone，不如让两个输出参数化形成真正的专家，并先用 support forward consistency 求解可解释锚点：

```text
x0      = P_G(y_S)                 # adjoint / FBP / K-step SIRT physics lift
z_res   = O_res(x0, E_G(y_S), q)    # trust-and-correct operator
z_abs   = O_abs(x0, E_G(y_S), q)    # rebuild operator
x_res   = x0 + H_res(z_res)
x_abs   = H_abs(z_abs)
w_S     = argmin_[0,1] ||A_S(w*x_res + (1-w)*x_abs) - y_S||^2
delta_w = R_psi(r_S, d_branch, identifiability)
w       = clip(w_S + delta_w, 0, 1)
x_hat   = w * x_res + (1-w) * x_abs + delta_x_null
```

`r_S` 只能由部署时可见信息构成：view fraction、最大角度缺口、ray coverage、mask fraction、位移置信度、support reprojection residual；`d_branch` 是两分支差异或预测方差。`identifiability = ||A_S(x_res-x_abs)||^2` 表示两专家在当前观测中是否真能被区分。禁止把 field truth 或 test split 标签送进 router。

**H3：** 仅靠有真值样本的 field loss 会让 router 学到数据集捷径。训练时随机把可用相机拆成 support `S` 与 query `Q`，但 query 不应再只生成近 0.5 的软专家标签；它更适合监督受限幅 `delta_w`、专家排序或 support-nullspace 体场残差。只有当 query 信号与 field 失败在多工况中稳定对齐，才能把它写成自监督贡献。

## 3. 推荐架构

### 3.1 几何感知输入层

本科可执行版先用固定网格和已知投影矩阵：每视角 2D 特征经伴随算子反投影到 3D，再按 ray coverage 加权融合。不要只把所有 view 图像按通道堆叠，否则改变相机数就改变网络接口。

如果组内相机几何跨实验变化，再升级为 GINO/Geo-FNO 风格的 irregular-to-grid embedding；这属于第二阶段，不是开题第一周的必需品。

### 3.2 Operator experts

首选 3D FNO/TFNO；matched 3D U-Net、CNO/U-NO 是必须保留的架构对照。v1 两个输出几乎共享所有参数，分支的中位相对误差差异只有约 2%，路由目标可识别性不足。v2 应先训练两个独立或低共享专家，再以匹配参数的单模型/双宽模型控制容量；只有确认专家互补后才压缩共享。

薄火焰前缘存在高频与局部非平稳性，若 family OOD 仍失败，可测试局部卷积分支、Riesz derivative branch 或 multiwavelet，而不是无依据地堆大模型。

### 3.3 双分支专家

- Residual expert：适合 physics lift 已保留主体形态的工况，学习小修正。
- Absolute expert：允许在 lift 严重缺视角或失配时重建整体结构。
- v2 首先使用独立或低共享专家验证真实分工；再比较共享率，并始终保留参数量匹配对照，避免把容量收益误归因于路由。

### 3.4 Physics-anchored router

对线性 support forward operator，两专家连线上的最佳观测一致性权重有闭式解：

```text
d_S = A_S(x_res - x_abs)
w_S = clip(<d_S, y_S - A_S(x_abs)> / (||d_S||^2 + epsilon), 0, 1)
```

非线性 ray bending 时可改为一维线搜索或 Gauss-Newton 步。`||d_S||^2` 同时是路由可识别度；当分母过小，观测无法区分两专家，模型应回退到等权/先验权重或输出 abstention，不应把噪声拟合成精确路由。

Router 不读取平铺后的 3D 常数通道，而读取一个低维 reliability vector：

```text
[view_fraction,
 max_angular_gap,
 ray_coverage_mean/std/min,
 mask_fraction,
 displacement_confidence_mean/p10,
 support_reprojection_mean/p95,
 branch_disagreement_mean/p95]
```

输出可以是：

1. 一个受限幅 `delta_w`，以 `w_S` 为锚点，是当前首选；
2. 一个低分辨率空间权重场 `w(x)`，能处理局部遮挡，但更容易塌缩；
3. 两专家概率 + abstention score，用于“不可信时转 NeRIF”。

本科主线从 1 开始，论文升级再做 2/3。

## 4. 训练目标

```text
L = lambda_f * L_field
  + lambda_g * L_gradient
  + lambda_s * L_support_projection
  + lambda_q * L_query_projection
  + lambda_b * L_boundary
  + lambda_c * L_conservation_or_integral
  + lambda_r * L_router
```

- `L_field`：仅在有合成/数值真值时用；真实数据可以缺失。
- `L_query_projection`：由未送入模型的相机计算，但必须先报它与 field error 的对齐，不默认它就是有效伪标签。
- `L_router`：首选对 `delta_w` 的限幅回归、pairwise ranking 或 identifiability-weighted query loss；避免再次使用集中在 0.5 附近的软专家标签。
- `L_null`（可选）：学习 support operator 不可见但 query/field 可验收的体场残差，并约束 `A_S(delta_x_null)` 不破坏已观测一致性。
- `L_gradient`：对薄前缘比单纯 voxel L2 更敏感。
- `L_boundary`：折射率扰动远场归零或满足组内边界约束。

训练课程：先 7/5 views 稳定学习，再逐步加入 3 views、遮挡、噪声与几何扰动；每个 batch 随机 support/query split。三视角不能只作为从未见过的外推域，否则 router 无法学习该工况。

## 5. 最小可发表实验矩阵

### Baselines

1. physics lift / SIRT；
2. NeRIF per-instance；
3. matched 3D U-Net；
4. Residual FNO；
5. Absolute FNO；
6. uniform average dual head；
7. support-fit closed-form mixture（已完成的新强基线）；
8. fixed metadata gate 与从零学习的 query router（已完成的负对照）；
9. independent dual + support-fit（已完成）；
10. matched free/null correction（已完成）；
11. one-query binary / line search / all-query line search（已完成 adaptive synthetic pilot）；
12. 同总相机数的 `S∪Q` SIRT/TV/Tikhonov/support-fit/NeRIF/direct operator（待做）；
13. non-learning query-null update（待做）；
14. fixed/random/max-gap/adaptive query + independent Q_audit（待做）；
15. QC-SNCO + short NeRIF refinement（前置门槛过关后恢复）。

### 必做消融

| 消融 | 回答的问题 |
| --- | --- |
| 去掉 physics lift | 收益是否真的来自物理初值？ |
| 单 head vs 双 head | 路由是否只是增加参数？ |
| 共享 vs 独立专家 | 路由塌缩是否来自专家过于相似？ |
| uniform vs support-fit vs learned correction | 学习部分是否真正超过可解释物理基线？ |
| metadata-only vs query-trained router | held-out camera 是否提供可迁移监督？ |
| 无 identifiability weighting | 路由器是否在不可区分样本上拟合噪声？ |
| 无 nullspace constraint | 学习残差是否破坏 support-view 一致性？ |
| 无 view dropout | 少视角泛化是否来自训练覆盖？ |
| 无 reprojection loss | forward consistency 是否必要？ |
| FNO vs matched U-Net/CNO | 是否真是 operator architecture 收益？ |
| 固定 geometry vs geometry perturbation | 对标定误差是否稳健？ |
| random vs operator warm-start NeRIF | 是否真正节省优化时间并降低失败率？ |
| S-only vs S∪Q direct vs S+Q correction | query 相机分工是否超过“直接加视角”？ |
| Q_fit 与 Q_audit 分离 | 是否把标定视角又用作了最终验收？ |
| random/PCA/branch/non-learning/learned direction | 网络是否真的学到比数值方法更好的修正方向？ |

### 数据切分

- sample identity 严格隔离；
- phantom/flow family 整族留出；
- view-count 与 angular-layout 整格留出；
- noise 与 geometry perturbation 整格留出；
- resolution transfer 必须重新生成高分辨率 forward data，不能插值同一数组冒充；
- 若有真实数据，按实验工况/采集批次划分，不能随机拆帧泄漏时序相关性。

### 指标

- 有真值：field relative L2、gradient L2、PSNR/SSIM、front location/thickness、mass/centroid；
- 无真值：observed 与 held-out reprojection、边界误差、跨相机一致性；
- 可靠性：risk-coverage curve、error-vs-uncertainty Spearman、95% interval coverage；
- 工程：参数量、显存、单体推理、physics lift 时间、NeRIF refinement 步数与总耗时；
- 失败图：逐工况 heatmap、best/worst/median、field 与 held-out 排名分叉。

## 6. 论文贡献梯度

### 本科毕业设计可完成

- 公开/合成 BOST paired dataset pipeline；
- physics lift + Residual/Absolute/dual-head FNO；
- support/query 相机拆分；
- 五类 OOD 与严格消融；
- 可交互三维切片、等值面、重投影与误差热图展示。

### 有希望形成高质量论文

现有小尺度 adaptive pilot 仍不足以投稿；在计数“其中三项”前，同相机预算、Q_fit/Q_audit 隔离和独立 forward 三项都必须通过：

1. 组内真实 BOST/PIV-BOST/4D BOST 数据；
2. query-calibrated null correction 在真实/近真实未知工况稳定超过 support-fit 强基线；
3. 与 NeRIF、传统 BOST、matched CNN 和至少一种 geometry-aware 方法完整对照；
4. 对相机标定误差、遮挡、位移噪声和流态变化的系统性结论；
5. operator warm-start 让 NeRIF 总耗时/失败率显著降低，同时保持最终物理一致性；
6. 可公开的代码、合成数据协议和复现实验。

单靠“把 FNO 用到 BOST”或“加一个 attention/gate”不够。当前 corrector 使用 field truth 训练，不得称为自监督。若后续全部门槛通过，更可信的贡献表述是：**发现并量化 physics-lift reliability flip，在同相机预算下比较可观测物理解与受约束修正，并在 BOST 真实 ray geometry、相机 VOI 与 NeRIF refinement 中证明价值。**

## 7. 风险与停止条件

- 若独立双分支在 matched parameter budget 下不优于最佳单分支/等容量 ensemble，停止增加 router 复杂度。
- 若学习修正不能跨种子、跨工况稳定超过 support-fit，把闭式融合保留为毕设算法主体，不宣称 neural routing 创新。
- 若 query reprojection 与 field error 相关性低，不能把它当部署可靠性指标；改做不确定性或 abstention。
- 若 geometry error 主导，优先做标定/位移质量工具，不继续堆网络。
- 若真实样本少于可跨工况切分的最低量，论文主张收缩为 method feasibility + synthetic closure + real case study。
- 若 NeRIF warm-start 不减少总 wall time 或失败率，不把它写成加速贡献。

## 8. 三个月执行

### 第 1--2 周（已完成合成版）

冻结数据合同、真实接口与 baseline；实现 view dropout 和 support/query forward projection；把现有 168-sample 实验升级为可变 view layout。

### 第 3--4 周（v1 与 v2a 已完成）

已实现共享/独立专家、uniform average、support-fit、oracle audit 与三种子稳定性检查。当前还缺等容量单宽模型与 independent ensemble 对照。

### 第 5--6 周（v2b/v2c 小尺度已完成）

已保留 metadata/query router、free corrector 和 full learned null 负对照；已实现 exact null projector 与 query-camera alpha。下一步在独立分辨率和真实 ray geometry 扫 query angle/noise/sync，报告 value-of-information、support leakage、field/held-out 与 risk-coverage。

### 第 7--8 周

接入更真实的 ray geometry、遮挡/标定扰动和 thin-front family；若有组内数据，先只跑 inference/interface audit。

### 第 9--10 周

做 NeRIF warm-start、运行时间和失败率；固定所有图表脚本与随机种子。

### 第 11--12 周

补 5 seeds / bootstrap、完整消融、展示网页、师兄复核与开题/论文初稿。

## 9. 师兄审核时只问这八个问题

1. 任务到底是 displacement-to-volume，还是 volume-history-to-future？
2. 可否训练时留相机做 query view？
3. 真实几何是否固定，是否有 ray table / projection matrix？
4. 哪个物理量是最终输出，是否需要 Gladstone-Dale/温度换算？
5. 哪个传统重构是组内认可 baseline？
6. NeRIF/TDBOST 代码能否只开放接口或一小段脱敏样例？
7. 高质量论文最看重速度、少视角、真实稳健性还是 4D？
8. 哪些结果可公开、哪些仅组内展示？

## 10. v2 最直接的四篇新文献

1. Golub and Pereyra, SIAM JNA 1973, DOI `10.1137/0710036`：可分离最小二乘/变量投影的数值基础。它提醒我们，support-fit 消元本身不是原创点。
2. Schwab, Antholzer and Haltmeier, Inverse Problems 2019, DOI `10.1088/1361-6420/aaf14a`：用 `Id-A^+A` 把神经修正投到 forward null space，保持 data consistency，是 `delta_x_null` 的数学直接前例。
3. Quan et al., IEEE TPAMI 2024, DOI `10.1109/TPAMI.2024.3359087`：不完整测量下 range/null-space 双网络协作与自监督一致性，比通用 MoE 更贴合新分解。
4. Aggarwal, Mani and Jacob, IEEE TMI 2019, DOI `10.1109/TMI.2018.2865356`：MoDL 把学习先验与数值 data-consistency block 交替，支持“不让自由网络隐式猜物理”的结构选择。

这四篇只支持组件合理性，不证明 SC-DBNO 原创或对 BOST 有效。真正的方法贡献必须来自 BOST 特有 forward geometry、可识别度、跨工况结论和真实验验证。
