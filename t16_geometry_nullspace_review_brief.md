# T16 算子学习 × 三维重建：几何、零空间与 query 假设审计

更新：2026-07-11
对象：何远哲师兄方向讨论  
状态：v2d 同重建预算 controlled inference audit 已完成；当前 QC-SNCO checkpoint 路径 0/3 通过并停止扩模型

> **2026-07-11 重要修正：**当前 query 视角是按每个样本从多个候选角中选出，不是一台事先固定的相机；同时缺少同总相机数的直接重建对照、独立 `Q_audit` 和独立 forward。因此 <code>+0.746%</code> 不再表述为 closure passed。详见 [QC-SNCO 红队审计](qc_snco_red_team_audit.md)。

> **v2d 后续判决：**现已固定 Q_fit=80°、Q_audit=60° 并加入同重建预算 direct/numerical controls。fixed learned correction 相对 direct 在 K=4/6/8 为 `-20.17%/-13.07%/-15.11%`，0/3 通过。训练 mask 未匹配，因此只否定当前 checkpoint 路径；最新会前材料改用 [v2d 师兄简报](fair_camera_budget_review_brief.md)。

## 一句话判断

当前最值得继续的不是“再换一个更大的 FNO”，也不是直接扩大 QC-SNCO，而是：

> 在同样的总相机预算下，比较“物理粗重建 + 受约束修正”、“直接 S∪Q 重建”和“不学习 query-null 数值更新”。只有前者在固定 query、独立 `Q_audit` 和独立/真实 forward 中仍显著占优，才恢复 **QC-SNCO** 作为论文模型。

这个名称只是工作代号，不是原创性结论。正式论文前必须做系统检索、真实 BOST 对照和组内确认。

## 哪些证据已成立，哪些主张已暂停

### 1. 独立专家让互补性变得真实

- 共享双输出 v1：43,485 参数，分支平均差异约 0.119；query router 接近 0.5，不能可靠路由。
- 独立双专家 v2a：86,823 参数，分支平均差异约 0.190，提升约 59%。
- 独立专家的 support-fit 在五域 264 个样本-种子单元中，平均相对最佳端点 regret 为 `-1.364%`；`79.17%` 单元达到或优于最佳端点。
- 这说明“先生成互补候选，再用可见物理量闭式融合”比自由 MLP router 更可信。

边界：独立专家约翻倍模型容量，尚未完成等参数单宽模型/独立 ensemble 对照，不能把全部收益归因于结构。

### 2. support nullspace 里有很大结构性空间

在独立 support-fit 基线上，把真实场误差精确投影到 support operator 的零空间：

- 264/264 个样本-种子单元都可改善；
- 平均 field improvement `38.626%`；
- 最弱 domain 的平均改善仍为 `23.580%`；
- 剩余误差能量平均 `59.459%` 位于 support nullspace；
- support projection 最大变化 `5.51e-16`。

这是“存在可改空间”的上界证明，不是可部署算法，因为修正方向使用了 field truth。

### 3. 直接学零空间修正并没有兑现上界

匹配 13,105 参数、相同初始化的 free/null FNO corrector：

- free corrector 平均 field improvement `-32.06%`，一个随机种子明显失稳；
- hard-null corrector 最大 support leakage 仅 `2.59e-8`，训练稳定很多；
- 但 hard-null corrector 总体 field improvement 为 `-0.126%`，held-out improvement 为 `-0.412%`；
- 只捕获了近乎为零的 oracle headroom；
- noise/family OOD 有小幅收益，view/joint OOD 反而变差。

结论：硬零空间投影是有价值的稳定器，但“约束正确”不等于“方向学对”。这项负结果必须保留。

### 4. adaptive query 在当前 closure 中给出微弱正信号

令网络给出零空间方向 `d_N`，再按每个样本从多个未参与 support reconstruction 的候选角中选一个视角并求：

```text
alpha_Q = clip( <A_Q d_N, y_Q - A_Q x_S> / ||A_Q d_N||^2, 0, 1 )
x_hat = x_S + alpha_Q d_N
```

因为 `A_S d_N ≈ 0`，任何 `alpha_Q` 都不破坏 support consistency。

三随机种子、五个测试域、88 个独立测试体场 × 3 个模型种子，共 264 个样本-种子结果单元：

| 方法 | 总体 field improvement | 优于 base 比例 | p10 improvement | 额外观测 |
| --- | ---: | ---: | ---: | --- |
| 全量 learned null，`alpha=1` | -0.126% | 60.98% | -4.955% | 0 |
| 单 query 二元接受/拒绝 | +0.677% | 44.32% | -0.023% | 1 view |
| **单 query 闭式 line search** | **+0.746%** | **52.65%** | **-0.338%** | **1 view** |
| 全部 query 闭式 line search | +0.956% | 65.53% | 0.000% | all withheld views |
| field-truth oracle line search | +1.212% | 81.82% | 0.000% | 不可部署 |

单 query line search 在 `3 seeds × 5 domains` 的 15 个 seed-domain 均值上全部为正：

- IID `+0.865%`；
- view OOD `+0.813%`；
- noise OOD `+1.079%`；
- joint OOD `+0.668%`；
- family OOD `+0.369%`。

全部 query 版本同样在 15/15 个 seed-domain 汇总单元为正。其 support leakage 最大仅 `2.26e-9`。这 15 个聚合值不能当作 15 份独立物理样本；统计推断必须回到 88 个独立体场并按体场聚类模型种子。

这是一项需要重新设计的 pilot，不是论文级结论。底层统计单元是 88 个独立测试体场，3 个 model seeds 不是额外独立物理样本；当前 query 拟合与 held-out 验收也尚未完全分离。先补同预算直接重建、固定 query、Q_fit/Q_audit 隔离与独立 forward，不继续堆更大模型。

## M3B 几何迁移给这条线的约束

M3B 新增六类 synthetic angular layout、四形态、三动力学、三噪声、两视角、四随机种子：

- 1,728 observation cells；
- 12,096 rank candidates；
- nested leave-one-geometry-out，无 outer geometry 泄漏；
- 无 held-out-camera selector mean/p95 regret `0.273% / 1.196%`；
- `93.81%` 单元在 oracle 1% 内；
- geometry 并非主要风险，morphology/dynamics/noise 更主导；
- naive combined UQ 几乎失效，Spearman `0.015`、高风险 AUC `0.532`；
- prediction std 的高风险 AUC 约 `0.697`，可做筛查，但 full-rank fallback 反而使 system risk 变坏。

对 T16 的含义：不能把“不确定度高”直接等同于“切换到另一个模型”。拒答、fallback 和额外 query camera 必须分别验证。

## 暂定模型：QC-SNCO

```text
multi-view BOS displacement + geometry
  -> geometry-aware lift / short iterative reconstruction
  -> independent residual and absolute operators
  -> analytic support-fit mixture x_S
  -> learned correction direction d
  -> exact/approximate support-null projection d_N
  -> one reserved query camera chooses alpha_Q
  -> x_hat = x_S + alpha_Q d_N
  -> optional NeRIF refinement using x_hat as initialization
```

### 可能的论文贡献

1. **支撑一致性结构**：修正被限制在 declared support nullspace，而不是靠 soft loss 猜数据一致性。
2. **验证相机闭式校准**：额外相机不直接参与三维重建，只标定不可观测修正幅度；代价和收益可以明确画 Pareto 曲线。
3. **算子到 NeRIF 的连接**：operator 提供跨样本快速初值，NeRIF 负责 per-instance 连续精化。
4. **可证伪可靠性**：按 family/view/noise/geometry/resolution 分域，报告失败率、risk-coverage 和 query-camera value of information。

这些目前都只是候选贡献。尤其是第 2 点必须先做相似方法检索，不能因为组合方式在当前书目中没出现就宣称原创。

## 投稿级实验缺口

### 必须补齐

1. 组内真实 BOST forward/ray geometry 与标定扰动。
2. 独立 `24^3/32^3` 数据，而不是把 `8×16×16` 插值放大。
3. 86,823 参数双专家与等参数单模型、ensemble、U-Net/CNO/GINO/DeepONet 公平对照。
4. query camera 数量、角度与噪声的 value-of-information 曲线。
5. nonlinear ray / approximate null projector 下的一致性误差。
6. 至少五随机种子，独立 phantom families 和真实样例。
7. operator warm-start NeRIF：总 wall time、最终 forward residual、失败率，而不只比较初始 RMSE。
8. 真实数据无 field truth 时，用 held-out camera、重复实验、边界/积分量和专家判断验收。

### 停止或降级条件

- 若真实 query camera 不能稳定预测 field/physics improvement，保留 support-fit，把 null correction 降为负结果章节。
- 若等参数单模型追平独立专家，论文贡献不能写“dual expert”，应转向 query-calibrated physics correction。
- 若 approximate projector 在真实几何下开销过高，改用局部线性化、Krylov null projection 或 soft constrained correction。
- 若 operator warm-start 不降低 NeRIF 总耗时/失败率，不把它列为主贡献。

## 文献模块怎样服务模型

| 组件 | 必读来源 | 只提取什么 | 不可照搬的边界 |
| --- | --- | --- | --- |
| 可变传感器 | VIDON；GNOT | permutation invariance、随机 sensor、geometry tokens | PDE sensor 不是 BOS camera/ray |
| 任意几何 | CORAL；GINO；Beyond Regular Grids | irregular input/query、latent grid、direct spectral transform | 只有真实几何变化证明需要时再上 |
| 全局+局部 | Localized kernels；DuFal | FNO 过平滑、局部高频支路、薄前缘 | 其公开收益不能迁移到 BOST |
| 物理粗解+修正 | Neural Correction Operator；MoDL | short solver + learned correction、显式 data consistency | EIT/MRI forward 与 BOST 不同 |
| range/null 分解 | Deep Null Space Learning；Siamese Cooperative | null projector、range/null training、identifiability | 线性精确伪逆假设可能失效 |
| 可靠性 | Calibrated UQ；Conformalized-DeepONet；SelectiveNet | functional coverage、risk-coverage、reject option | 校准必须在独立 calibration set 上重做 |
| operator-INR 桥 | O-INR；CORAL；NeRIF | function-space mapping、coordinate decoder、warm-start | NeRIF 是 per-instance，不是天然 operator |
| 物理守恒 | ClawNO | 只有目标变量和方程支持时编码 conservation | 折射率场本身不能随意声明 divergence-free |

## 请师兄优先回答的八个问题

1. 组内真实部署能否留一台相机只做 query validation，还是每台相机都必须进入重建？
2. 何师兄建议的“算子学习”目标更接近位移到三维折射率，还是三维/四维场演化？
3. 可提供的最小数据包是否包含 displacement、camera/ray geometry、mask、标定和一个可信 baseline reconstruction？
4. 组内相机布局是否跨实验变化，变化到什么程度？
5. NeRIF 当前最大痛点是时间、初始化失败、少视角、几何误差还是薄前缘？
6. 真实任务最重要的验收量是 field slice、held-out reprojection、密度积分、前缘位置还是后续 PIV 补偿？
7. 是否认可“query camera value of information”作为实验设计贡献？
8. 更希望本科阶段先完成高质量 synthetic closure，还是必须包含真实样例才能作为主线？

## 当前建议

把毕业设计主标题先写成：

> **面向少视角背景纹影层析的物理粗重建与受约束神经修正算子**

开题阶段只承诺：强 baseline、同相机预算、支撑一致性、跨工况审计和真实接口。论文级 QC-SNCO 要等固定 query、独立 Q_audit、独立/真实 forward、等预算强基线和 learned-direction 上界通过后再恢复。
