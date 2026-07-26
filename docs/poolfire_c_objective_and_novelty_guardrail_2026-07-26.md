# PoolFire C 路线：优化目标、预期优势与原创性护栏

> 日期：2026-07-26  
> 当前结论：方向明确，存在可防御的组合创新空间，但尚不能承诺“全球独一无二”。  
> Cross14 定位：低容量、可审计的 sentinel/control，不是最终论文算法。

## 1. 唯一优化目标

我们的任务不是让神经网络单独输出一张更好看的三维图，而是：

> 在相同最终三维场精度下，用只依赖部署可见观测的 warm initializer，减少完整多视角
> `A/A^T` 调用，并在真实端到端计时中降低 wall time 和峰值内存。

这是一个有约束、分层排序的目标。

### 第一层：精度与伤害约束

候选经过同一个冻结 CGLS/PCGLS refinement 后，必须逐 trajectory 同时满足：

- field relative-L2 与目标基线等价；
- gradient relative-L2 与目标基线等价；
- observation/reprojection error 与目标基线等价；
- p50、p90、worst 都报告；
- harm rate 不得被平均值掩盖；
- 不使用 heldout truth 决定采用候选还是 fallback。

没有“同精度”，后面的速度没有意义。

### 第二层：物理算子成本

在第一层通过后，才比较：

- 完整多视角 `A` 调用；
- 完整多视角 `A^T` 调用；
- setup、BP、网络、line search、refinement 与 fallback 的完整总账。

调用数减少是机制证据，不自动等于速度提升。

### 第三层：真实部署成本

最后比较：

- fresh-process 端到端 wall time；
- whole-pipeline peak RSS；
- 模型文件、临时缓存和输入表示的存储；
- Mac CPU 与未来 GPU 两种执行路径。

最终判决按“先精度、再调用、再 wall、再内存”的字典序进行，不能用一项优势抵消前一项
失败。

## 2. 预期成果为什么有价值

如果最终方法过门，它的优势不是“网络名字更新”，而是以下五点同时成立。

1. **快速但不替代物理。** 网络只提供起点，最终答案仍由未修改的物理算子和 Krylov
   求解器得到。
2. **只看部署可见量。** 输入限于 observation、raw BP、geometry sensitivity、
   residual/view-balance/spectral summaries，不在线读取真实三维场。
3. **可纠正。** warm field 被限制在物理求解器能够继续修正的可观测子空间，避免网络
   注入永远留在解中的零空间幻觉。
4. **成本公平。** 接受分支、拒绝分支、经典 controls 和网络开销全部进入同一账本。
5. **能迁移到组内流程。** 公开 CFD proxy 先证明方法与失败边界，随后只替换真实
   observation、geometry 和单位合同，不重写评价规则。

## 3. 什么已经被别人做过，不能冒充创新

以下宽泛主张都已被强在先工作覆盖：

| 宽泛想法 | 代表性一级来源 | 为什么不能当我们的独创点 |
|---|---|---|
| 用神经网络重建 BOST 三维折射率场 | [NeRIF](https://arxiv.org/abs/2409.14722) | 已用隐式神经场直接表示折射率及梯度，并做数值与火焰实验 |
| 用神经表示处理 sparse-view TBOS | [NeDF](https://arxiv.org/abs/2409.19971) | 已用神经 deflection field、位置编码和分层采样处理少视角病态性 |
| 用更紧凑的神经 primitive 重建火焰 | [Neural Refractive Index Primitives](https://arxiv.org/abs/2605.11454) | 已把 hash encoding、梯度一致性与三维 mask 用于 BOST 最终重建 |
| 学一个初值再接固定迭代器 | [Learning to Warm-Start Fixed-Point Optimization Algorithms](https://www.jmlr.org/papers/v25/23-1174.html) | 已系统研究 warm-start 网络与固定迭代的端到端训练及泛化界 |
| 神经算子给 Krylov 求解器 warm start | [NOWS](https://arxiv.org/abs/2511.02481) | 已明确用 neural operator 初始化 CG/GMRES 并比较迭代和时间 |
| 神经算子与 CG 结合 | [FCG-NO](https://openreview.net/forum?id=J0ty1o7nCj) | 已把神经算子作为 flexible CG 的非线性预条件器并给出理论分析 |
| 让 neural warm start 对后续求解器更安全 | [Spectrally Safe Neural Operator Warm-Starts](https://arxiv.org/abs/2606.21828) | 已通过离散能量微调恢复 Newton 线性化的正定谱；“solver-safe warm start”本身也不是新主张 |
| `D^-1 A^T y` 或对角预条件 | 经典 Jacobi/灵敏度归一化思想 | geometry equalization 本身是经典组成部分，不是新算法 |
| 严格 split、日志、校验和与 fail-closed | 可复现研究规范 | 它们提高可信度，但不是科学方法贡献 |

因此，“BOST + FNO/DeepONet + CGLS”或“Cross14 比 BP 好”都不足以形成高质量论文的
核心创新。

最新近邻也迫使我们进一步收窄表述：GEOK-Warm 不能泛称“第一个物理安全 warm
start”。它要验证的是更具体的逆问题机制，即自由三维初值中的 `Null(A)` 分量为什么
不能被后续 CGLS 修正，以及把候选限制在 BOST 的 `Range(A^T)` / observable Krylov
子空间后，能否在同精度和完整成本门下取得优势。这与 Newton 迭代中保持 Jacobian
正定谱是不同的安全对象。

## 4. 暂定最终方法族：GEOK-Warm

当前最值得验证的论文级假设暂命名为：

> **GEOK-Warm: Geometry-Equalized Observable-Krylov Warm Start**

名称只是工作标签，真正贡献必须由结构与结果证明。

令 `P` 表示冻结 gauge projector：

```text
q0 = P A^T y
e0 = P D_tau^-1 q0
q1 = P A^T A q0
```

其中 `e0` 只作为几何灵敏度与局部可观测性的条件输入。一个轻量 observation-
conditioned 网络读取部署可见的 `y, q0, e0, D` 与 view-balance/spectral summaries，
只预测少量全局系数：

```text
(c0, c1) = G_theta(observable summaries)
h = c0 q0 + c1 q1
```

所以：

```text
h in K2(A^T A, A^T y) subset Range(A^T)
```

这与任意 3D CNN/FNO 直接吐出完整场不同：它不会主动加入后续 CGLS 无法消除的
`Null(A)` 分量。

再用已经需要计算的 `A h` 做解析尺度校准：

```text
alpha* = argmin over alpha in [0, alpha_max] ||y - alpha A h||_2^2
x0 = alpha* h
```

最后只运行冻结的一至两步 CGLS/PCGLS。若 deployment-visible gate 不可信，则回退到
预注册的经典同成本分支。

### 最小理论贡献

若采用该方法，论文至少应给出并验证：

1. **可观测子空间命题：** `h` 位于 `Range(A^T)`，不会主动注入不可纠正零空间。
2. **尺度校准命题：** 因为 `alpha=0` 可选，校准后的初始 measurement residual
   不劣于零初值；这不等于 field no-harm。
3. **Krylov 保持命题：** 后续 refinement 仍解同一个冻结物理问题。
4. **完整成本命题：** BP、Krylov basis、网络、尺度校准、refinement 和 fallback 的
   `A/A^T` 全部入账。

论文中不得把这些命题合并成未经证明的“全局安全保证”。`Range(A^T)` 只约束线性
代理下的可纠正性；它不自动保证 field no-harm、非线性 curved-ray 稳定性、Jacobian
正定或跨 geometry 泛化。

Cross14 仍应先完成 outer LOTO。它的作用是判断“局部自由三维残差有没有 headroom”，
并作为 GEOK-Warm 的自由输出反例和低容量消融，不作为最终原创算法。

## 5. 专利在先技术带来的边界

本轮 Google Patents 权利要求初筛没有发现单一专利完整覆盖 GEOK-Warm 的全部组合，
但以下近邻风险必须保留：

| 在先技术 | 与我们的重叠 | 需要保持的区别 |
|---|---|---|
| [CN114067047B](https://patents.google.com/patent/CN114067047B/zh) | BOST 几何修正后再次重建 | 不把“修正 BOST 几何再重建”写成创新 |
| [US20180197317A1](https://patents.google.com/patent/US20180197317A1/en) | 神经网络产生初始重建并减少迭代 | learned warm start 本身不是新颖点 |
| [CN110060314B](https://patents.google.com/patent/CN110060314A/en) | AI 处理当前迭代图像后继续迭代 | 网络只调用一次，不读历史迭代、不反复跳步 |
| [US10970887B2](https://patents.google.com/patent/US10970887B2/en) | 传统初始图像再由深度学习给最终图像 | 网络不输出最终结果，最终结果由固定物理 solver 产生 |
| [US11790598B2](https://patents.google.com/patent/US20220189100A1/en) | 2D projection、反投影与 3D 网络重建 | 避免两网络夹心最终重建架构 |
| [US20240331127A1](https://patents.google.com/patent/US20240331127A1/en) | 神经算子用于层析/折射率反演 | 不替代真实 forward；仅学习受限初值 |
| [US11087508B2](https://patents.google.com/patent/US11087508B2/en) | 对角预条件 CG 层析重建 | `D^-1` 只是条件输入；主张落在 BOST 特定受限 warm start 与效果 |

这只是技术初筛，不是法律上的自由实施意见。正式申请专利或商业化前仍需专利代理人
检查中国、美国、PCT/EPO 家族和法律状态。

## 6. 论文级独特性必须过的六道门

### N1 文献与专利 claim chart

逐步骤对照最危险论文和专利，不只比较标题。每次冻结论文题名和 novelty claim 前重跑
检索。

### N2 数学结构差异

最终方法必须有“可观测 Krylov 约束、解析尺度校准或同等级别”的明确结构差异，不能
只换成 FNO、UNO、DeepONet 或更大的 3D U-Net。

### N3 强经典与强学习基线

至少同场比较 Zero-K3/K4、raw/equalized BP、normalized BP、PCGLS、dual ridge、
Cross14、同参数量 MLP/3D CNN，以及最相关 neural warm-start control。

### N4 trajectory-level 泛化

随机帧切分无效。必须完整 trajectory outer LOTO，再做锁模后的 fresh proxy holdout；
两条 untouched test 只能在全部规则冻结后一次打开。

### N5 真实效果

逐 trajectory 达到 matched field/gradient/observation accuracy，并同时证明：

- 完整 `A/A^T` 减少；
- p90/worst/harm 不恶化；
- fresh-process wall time 下降；
- whole-pipeline peak RSS 不增加到抵消收益。

### N6 组内真实 BOST 迁移

公开 CFD proxy 只能支撑方法开发。最终需要至少一个组内真实 BOST 样例，绑定相机、
单位、reference/gauge、现用 baseline 和停止规则。真实样例不过门时，只能写 proxy
方法预研，不能写通用 BOST 成功。

## 7. 现在能否保证“独一无二”

诚实答案是：**不能保证绝对全球唯一，但可以把作品做成可防御、难以与现有工作混淆的
独立成果。**

截至当前有三层判断：

1. **宽泛想法不唯一：** BOST 神经重建、learned warm start、neural operator +
   Krylov、geometry equalization 都有先例。
2. **完整组合暂未发现同构先例：** 在已核对论文与权利要求中，尚未发现同时采用
   BOST geometry-equalized observable、可观测 Krylov 受限初值、未修改 solver、
   matched-accuracy `A/A^T` 成本目标和 deployment-visible fallback 的方法。
3. **独特性仍需结果兑现：** 如果 GEOK-Warm 不能稳定击败强 controls，它只是一个
   有新意的失败假设，不是高质量算法成果。

我们能真正“确保”的不是没人想过类似词语，而是：

- 不抄现有方法的核心结构；
- 每项声称都有逐条在先技术对照；
- 模型、数据、成本和失败边界可独立复核；
- 论文只主张真实通过的差异；
- 最终代码、图表和实验谱系构成一套属于 Gao Youyang 的可复现作品。

## 8. 需要向师兄确认的三件事

1. 组内是否已有未发表的 neural warm start、BP normalization、Krylov-space initializer
   或相关专利方案？
2. 真实流程中最终重建的是 `rho`、`n`、`n-1` 还是 `Delta n`，现用 solver、停止规则
   与主要耗时在哪里？
3. 公开 proxy 门通过后，哪一个最小真实 BOST case 可用于迁移，能够提供哪些相机参数、
   displacement、reference/gauge 与 baseline 输出？

这三问同时保护科学方向、组内知识产权和后续真实迁移。
