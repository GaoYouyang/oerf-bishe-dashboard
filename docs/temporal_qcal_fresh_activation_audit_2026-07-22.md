# 多帧 BOST `q_cal` fresh 审计激活检查：先拿到真实接口，再开新盒子

> 日期：2026-07-22  
> 当前状态：`HOLD_INTERFACE_AND_CALIBRATION_RULE_UNRESOLVED`  
> 突破监测：**否**  
> 决策：不运行已经冻结的 12-rig dormant audit，不在 v4 的三个 opened rig 上继续调阈值。

## 1. 为什么 v4 结束后仍不能直接跑 fresh

[dormant contract](temporal_qcal_fresh_audit_dormant_contract_2026-07-22.md) 规定两个激活条件：

1. exact-score Godambe 在有限半径下提供有信息量的结果；
2. 何远哲师兄确认真实目标参数、callable/JVP/VJP、独立 acquisition、covariance 单位、geometry manifest 和主物理终点。

当前逐项审计如下。

| 激活项 | 当前证据 | 判定 |
|---|---|---|
| exact Godambe 数值有限 | 10,500/10,500 行有效；半轴中位/p90 为 `0.2562/0.2782 q_ref` | PASS |
| 点估计有 development signal | iterative full-profile 的 q 中位误差比 plug-in 低 16.46%，field/sequence 低 2.87%/3.09% | PASS，但仅 opened synthetic |
| 95% 校准门 | frozen global-max 覆盖 97.90%，CP95 为 97.48%--98.27%，显著过度覆盖 | FAIL |
| “informative”是否足以激活 | 原合同没有把 informative 写成可判定布尔式 | **UNRESOLVED，不能事后解释** |
| 真实 `q` 的物理含义 | 未收到师兄确认 | MISSING |
| callable 与 JVP/VJP | 未收到师兄确认 | MISSING |
| 独立统计单位与 covariance | 未收到师兄确认 | MISSING |
| geometry、主基线和物理终点 | 未收到师兄确认 | MISSING |

因此 condition 2 为 0 项闭合，condition 1 也存在预注册语义缺口。最严格的处理不是“差不多满足”，而是维持 HOLD。

## 2. 三篇何远哲/OERF 主线论文实际告诉了我们什么

### 2.1 NeRIF：体素瓶颈是真问题，但实验场没有体真值

一级来源：[NeRIF arXiv HTML](https://arxiv.org/html/2409.14722v2)、[Physics of Fluids DOI](https://doi.org/10.1063/5.0250899)。

论文明确把 voxel BOST 的问题归纳为分辨率、离散化误差、噪声鲁棒性、巨大矩阵内存和计算成本。NeRIF 用坐标网络同时输出折射率及其梯度，沿标定 ray 随机采样，并用 AD/ND 一致性与双位移损失约束重建。

与当前课题最相关的边界是：

- synthetic 数据用 4 阶 Runge--Kutta ray tracing 生成 9 个视角；
- 实验系统为一拖九内窥成像，DeepFlow 提取位移；
- 真实火焰没有三维 ground truth，8 个投影参与重建，第 9 个只做 re-projection 验证；
- 结论把 nonlinear ray tracing / non-paraxial assumptions 描述为可继续集成的能力。

**可以推出的研究问题：** straight/approximate reconstruction 与 curved high-fidelity evaluator 之间的 mismatch 值得测，held-out view 必须保留。  
**不能推出的结论：** 不能从论文措辞断言组内当前代码一定使用 straight ray，也不能把第 9 视角一致性等同于三维 field truth。

### 2.2 PIV-BOST：三维场误差会进入真实速度测量

一级来源：[Experiments in Fluids 官方页面](https://link.springer.com/article/10.1007/s00348-025-04093-y)。

该工作同步使用 9 视角 BOST 与中央平面 PIV，以固定相位延迟同步两套系统；三维折射率梯度用于估计 PIV 光路上的像素偏移。官方摘要报告，小型无引燃火焰中的瞬时速度误差量级约为 `±2%`，并进行逆向补偿。

这给毕业设计一个比“reprojection 更低”更现实的终点：

```text
三维折射率场 / 梯度误差
    -> PIV 像面位置偏差
    -> 速度误差及补偿后的残差
```

**可以推出的研究问题：** field-L2、gradient/front 与下游 velocity correction 应分开报告；时间同步、相机坐标映射和折射路径都可能形成系统误差。  
**不能推出的结论：** `±2%` 是该小火焰条件的估计，不是所有 OERF flow 的统一阈值；出版社页面还写明本研究没有公开数据集。

### 2.3 TDBOST：4D 共享表示是主线，但不能只看总体平均

一级入口：[ACM DOI](https://doi.org/10.1145/3809488)、[蔡伟伟教授当前主页](https://me.sjtu.edu.cn/teacher_directory1/caiweiwei.html)、[本地官方全文结构化精读](../tdbost_reproducibility_audit.html)。

蔡教授主页当前将该工作列为 2026 年 ACM Transactions on Graphics accepted。此前从 ACM 官方全文固定的本地精读记录显示：方法把场写成 `X-Y-Z-T`，使用 `XY-ZT/XZ-YT/YZ-XT` plane pairs、轻量解码和独立 distortion-correction 模块；强折射消融中，DC 与 tensor 表示的贡献不能混写。

**可以推出的研究问题：** 学习 tangent、warm start 或低秩时空基应分别做消融；强折射、频率保真、显存和 test-view 都要进入评价。  
**不能推出的结论：** 题名中的 high-speed/high-fidelity 不自动说明真实 `q_cal`、coverage 或跨 rig 泛化已经解决；公开仓库无 LICENSE，不能直接搬代码。

## 3. 现在最可信的四个物理困境

| 物理困境 | 观测层表现 | 当前可做的最小实验 | 真正需要师兄提供什么 |
|---|---|---|---|
| 曲光线 / 近轴模型失配 | held-out ray residual 有结构，强折射下 field/gradient 偏差 | 同一 field 的 straight/curved paired forward 与 signed-q sensitivity | 两层 callable、单位、坐标、DC 所在层级 |
| 相机标定与同步漂移 | 多视角 residual、PIV/BOST 相位和像面映射不一致 | flow-off repeat + 标定扰动 JVP/VJP + time-offset sweep | 标定文件、timestamp、repeat、允许扰动的参数 |
| 高维场吸收低维几何 | plug-in coverage 72.93%，full-profile 改善但仍需校准 | profile score、signed axes、anchor weak direction | `A/A^T` 或 forward/JVP/VJP、场正则与 q 定义 |
| 无体真值下的验收 | 第 9 视角可一致，三维场仍可能错 | held-out view + repeatability + field integral/front + PIV endpoint | 组内认可基线、hard case、最终物理指标 |

这四项是由论文与现有 synthetic 证据共同提出的**待验证假设**，不是已经观测到的实验室失败。

## 4. dormant contract 还缺哪几条可执行定义

### 缺口 A：校准阈值没有唯一算法

合同写了每 rig 1,000 个独立 null calibration replicate，却没有明确：

- confidence-region coverage 使用 pooled、per-rig、global worst-cell 还是两层 order statistic；
- null test size 与非零 q coverage 是否共用同一阈值；
- 新 rig 没有历史 cell 身份时怎样选择尺度。

v4 已经证明这个选择会翻转结论：global max 过度覆盖，post-hoc pooled 接近 95%。因此必须在 fresh 结果打开前把 threshold functional 写成代码和哈希。

### 缺口 B：统计单位不是 frame

NeRIF/TDBOST 的多帧共享相机、背景、标定和流动历史；奇偶帧 cross-fit 在 v4 中也已失败。严格 split 必须以独立 shot/session/rig 为单位。若真实数据只有一段高速序列，只能做 sensitivity/consistency，不得声称 distribution-free acquisition coverage。

### 缺口 C：coverage target 尚未分层

至少要区分：

1. 对 5 维 `q` 的联合椭球 coverage；
2. 每 rig 或每 acquisition 的 coverage floor；
3. 体场逐 voxel/point-fraction coverage；
4. 整个体场 simultaneous coverage；
5. 下游 velocity-error 或 no-harm 事件。

它们不是同一件事。Conformalized-DeepONet 或 operator-UQ 的 point-fraction coverage 不能替代整场置信集，也不能替代 PIV 补偿安全门。

### 缺口 D：运行预算需要先算

v4 的 10,500 条稠密参考运行用时 634.70 s。若同实现线性外推，dormant primary 的 `12 rigs x 13 cells x 1,000` 约需 2.62 小时，尚未包含 secondary stress、额外 covariance、真实 ray tracing 或多个网络。Mac 可以承担同规模 dense toy，但真实 NeRIF/TDBOST forward 的时间与显存必须由 callable 实测，不能套用这个估算。

## 5. 四个候选算法怎样排优先级

| 候选 | 学什么 | 经典底座 | 最小成功门 | 主要风险 | 当前优先级 |
|---|---|---|---|---|---|
| full-profile + acquisition-pooled calibration | 不学习；固定 pooled/order-statistic 规则 | iterative profile + GN/exact score | fresh rig 总体 95%、逐 rig `>=90%`、半轴有功效 | 只有 marginal coverage，难适应异质 rig | **A0，必须先做** |
| physics-normalized score calibration | 从 geometry/noise/信息谱预测 score scale | log-ridge/isotonic + disjoint conformal factor | 比 A0 更窄且不伤 fresh rig 尾部 | 已有 heteroscedastic/group conformal 先例，网络本身不新 | **A1，先低容量** |
| learned tangent/warm start + exact correction | 低秩 transport/nuisance tangent、q 初值或预条件 | 1--2 次真实 profile correction + fallback | 同物理调用预算下优于 fixed warm start，保持 field/velocity 门 | 需要真实 JVP/VJP，容易只学 generator | **B，最贴算子学习** |
| physical-target orthogonal score | 用独立 target/repeat 修 ridge 中心偏差 | joint estimating equation + cluster Godambe | signed axes、weak direction、真实 endpoint 同时改善 | 需要 flow-off/known target/独立 acquisition | **C，论文潜力高但数据依赖强** |

直接从投影或 residual 输出 `q` 并跳过 exact correction 的黑盒网络暂列 **REJECT**。v4 的 full-profile 第一步中位误差为 0.505 `q_ref`，已经说明“一次神经 proposal + 一步修正”不能未经验证当成迭代终点。

## 6. physics-normalized calibration 的可执行形状

对每个独立 acquisition/rig 构造只含部署可见量的摘要 `z_g`：

```text
z_g = [noise covariance summary,
       smallest generalized information eigenvalue,
       bread condition number,
       view angular coverage,
       held-out residual autocorrelation,
       straight-vs-curved discrepancy if available]
```

在 fit rigs 上用低容量模型估计正尺度 `s_hat(z_g)`，例如 log-ridge 或 monotone GAM；在完全独立 calibration rigs 上对

```text
R_gi = score_gi / s_hat(z_g)
```

计算最终 finite-sample inflation；test rig 只使用 `s_hat(z_test)` 与冻结 inflation。网络只有在 log-ridge、isotonic、pooled 和 per-group shrinkage 都被公平打败后才进入。

这不是现成创新。split conformal 只给 exchangeable unit 的 marginal guarantee，exact conditional distribution-free coverage 在一般条件下不可能；group clustering、heteroscedastic conformal 和 learned partitions 都已有工作。真正可能形成 OERF 贡献的是：

- BOST 可解释的 condition features；
- curved/straight 与 acquisition covariance 的角色隔离；
- exact profile correction；
- field/gradient/PIV velocity 多终点；
- 跨 rig 尾部与 fail-closed fallback。

## 7. 给师兄最短的激活请求

> 我们在 opened synthetic 多帧 BOST 上发现，iterative full-profile 能把 q 中位误差降低 16.46%，但 worst-cell 校准会过度覆盖；奇偶帧 cross-fit 也失败。为了不继续刷 synthetic，我想先确认真实联合参数是什么、straight/curved forward 与 JVP/VJP 在哪一层、什么算独立 acquisition，以及最终应看 field/gradient 还是 PIV velocity correction。能否给一个 flow-off repeat、一个 easy/hard sequence、geometry/timestamp、当前基线和 metric command？受限材料只留本地，不上传。

完整 8 问版本见 [师兄最小确认单](temporal_qcal_profile_inference_advisor_questions_2026-07-22.md)。

## 8. 一级来源与新颖性边界

- Lei et al., [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116)：split conformal 的有限样本 marginal coverage 与 locally varying width。
- Barber et al., [The limits of distribution-free conditional predictive inference](https://arxiv.org/abs/1903.04684)：无假设 exact conditional coverage 的不可能性。
- Barber et al., [Conformal Prediction Beyond Exchangeability](https://doi.org/10.1214/23-AOS2276)：交换性破坏与加权/随机化边界。
- Dunn, Wasserman, and Ramdas, [Distribution-Free Prediction Sets for Two-Layer Hierarchical Models](https://arxiv.org/abs/1809.07441)：组内相关、组间异质时不能把所有 observation 当 iid。
- Romano et al., [Conformalized Quantile Regression](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)：异方差区间形状与 disjoint calibration。
- Ma et al., [Calibrated UQ for Operator Learning](https://openreview.net/forum?id=cGpegxy12T) 与 Moya et al., [Conformalized-DeepONet](https://doi.org/10.1016/j.physd.2024.134418)：operator 输出 coverage 的邻近基线，不自动提供 BOST 物理正确性。

**最高允许表述仍是：完成了 fresh 审计的激活缺口与算法路线审计。真实接口、fresh run、新算法、真实重建、泛化、论文成功和突破均未发生。**
