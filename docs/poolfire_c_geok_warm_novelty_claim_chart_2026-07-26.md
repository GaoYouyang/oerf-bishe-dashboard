# GEOK-Warm 原创性 Claim Chart 与作品指纹

> 日期：2026-07-26
> 角色：冻结 v9.4 实验合同之后新增的文献审计附录
> 不修改：`poolfire_c_objective_and_novelty_guardrail_2026-07-26.md` 的冻结字节与 SHA
> 当前状态：`algorithm_breakthrough=false`

## 0. 一句话确认优化目标

本项目只优化一件事：

> **在逐 trajectory 的 field、gradient、observation 最终精度等价且 harm 不增加的
> 前提下，用只读部署可见观测的三维 warm initializer，稳定减少完整 `A/A^T` 调用、
> fresh-process wall time 和 whole-pipeline peak RSS。**

判决顺序固定为：

```text
最终精度与尾部伤害
    -> 完整 A/A^T 调用
    -> fresh-process wall time
    -> whole-pipeline peak RSS
```

后面的优势不能补偿前面的失败。只减少迭代但终点更差，不是成功；只减少 `A/A^T`
但 wall 不降，只能写成 operator-call reduction；平均值变好但某条轨迹明显受伤，也
不能写成稳定优势。

## 1. 作品身份

暂定论文方法族为：

> **GEOK-Warm: Geometry-Equalized Observable-Krylov Warm Start**

它不是“FNO 做 BOST”，也不是“神经网络加速 CGLS”。它由七项共同构成：

1. **BOST 特定逆问题：** 输入是多视角偏折观测，输出是带 reference/gauge 语义的
   三维折射率或密度差场。
2. **部署可见输入：** 在线只读 `y`、`A/A^T`、几何灵敏度和由它们得到的摘要，不读
   heldout 三维真值。
3. **一次性初值：** 网络只在 refinement 前运行一次，不在每次 Krylov 迭代中反复
   充当预条件器。
4. **可观测 Krylov 约束：** 学习模块只决定小型 `Range(A^T)` / Krylov 字典中的系数，
   不自由生成最终三维场。
5. **同一物理收尾：** 最终结果来自完全相同的冻结 CGLS/PCGLS。
6. **无真值回退：** 接受、拒绝和 fallback 只能使用部署可见 residual、view balance
   或几何量，并对全部分支记成本。
7. **完整证据：** trajectory-level split、p90/worst/harm、`A/A^T`、wall、RSS 和
   至少一个真实 BOST 迁移样例共同裁决。

其中任何单项都不是全球新思想。项目的可防御独特性只能来自这套结构与真实结果同时
成立。

## 2. 暂定数学结构

令 `P` 是冻结 gauge projector，`D=diag(A^T A)` 是冻结几何的灵敏度：

```text
q0 = P A^T y
e0 = P D_tau^-1 q0
q1 = P A^T A q0
```

轻量网络只读部署可见摘要并输出少量有界系数：

```text
(c0, c1) = G_theta(y, q0, e0, D, view summaries)
h = c0 q0 + c1 q1
```

因此在线性代理中：

```text
h in K2(A^T A, A^T y) subset Range(A^T)
```

再用需要显式入账的 `A h` 做一维解析尺度校准：

```text
alpha* = argmin_{alpha in [0, alpha_max]} ||y - alpha A h||_2^2
x0 = alpha* h
```

最后运行固定预算的 CGLS/PCGLS。`Range(A^T)` 约束只说明线性代理下不主动加入
`Null(A)` 分量，不自动保证 field no-harm、curved-ray 稳定或跨 geometry 泛化。

## 3. 最危险的公开近邻

这些工作会直接决定论文表述和强基线，不是装饰性引用。

| 公开一级来源 | 已经覆盖的部分 | 对本项目的约束 |
|---|---|---|
| [Pyramid-BOST](https://link.springer.com/article/10.1007/s00348-025-04153-3) | 多分辨率 coarse-to-fine、前一层结果初始化后一层，并在合成和真实 BOS 上验证 | 不能主张首个 BOST 初值或逐级精修；必须加入 pyramid-style control |
| [UBOST](https://link.springer.com/article/10.1007/s00348-020-2912-1) | 统一偏折估计和重建算子，减少方程与 solver 成本 | 不能主张首个 physics-aware BOST acceleration |
| [Direct BOST with RBF](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100) | 用 RBF 低维表示直接恢复折射率场 | RBF/reduced-basis 必须成为经典控制 |
| [Hybrid BOST refinement](https://doi.org/10.1063/5.0190778) | 粗重建、warping、投影修正与精细重建 | 不能主张首个 hybrid 或 coarse-to-refined BOST |
| [GRU-BOST](https://doi.org/10.1364/OE.505992) | 多视角观测到三维场的快速神经映射 | 不能主张首个 neural、direct 或 fast BOST |
| [Cone-ray BOS](https://arxiv.org/abs/2402.15954) | 把有限孔径、景深和光学几何嵌入神经重建 | 不能泛称首次 physics-aware；真实迁移必须考虑成像几何误差 |
| [NeRIF](https://arxiv.org/abs/2409.14722) | OERF 前作，神经隐式折射率场与实验火焰 | 差异必须是 amortized initializer、同一 CGLS 收尾与成本前沿 |
| [NeDF](https://arxiv.org/abs/2409.19971) | sparse-view BOST、神经 deflection field 和非线性 ray tracing | 不能主张首个 sparse-view neural BOST |
| [Neural Refractive Index Primitives](https://arxiv.org/abs/2605.11454) | 紧凑神经表示、hash encoding、mask 和快速逐实例优化 | 不能主张首个 compact neural BOST |
| [4D TDBOST](https://dl.acm.org/doi/10.1145/3809488) | OERF 时空低秩、轻量网络、速度与内存 | 不能复用泛泛的低秩、速度或内存故事 |
| [Inverse acoustic neural warm start](https://www.sciencedirect.com/science/article/pii/S0021999123004369) | 测量值到神经初值，再由传统 Gauss-Newton 精修 | 不能主张首个 inverse-problem neural warm start |
| [Learning to Warm-Start](https://www.jmlr.org/papers/v25/23-1174.html) | 学习初值与固定迭代器的训练和泛化框架 | learned warm start 本身不是贡献 |
| [NOWS](https://arxiv.org/abs/2511.02481) | neural operator 初始化 CG/GMRES，保持原离散与 solver | 不能主张首个 neural-operator Krylov warm start |
| [FCG-NO](https://openreview.net/forum?id=J0ty1o7nCj) | neural operator 作为 flexible-CG 非线性预条件器 | 必须区分一次性 initializer 与每步 preconditioner |
| [HINTS](https://www.nature.com/articles/s42256-024-00910-x) | 神经算子补充经典迭代器的谱盲区 | “网络补低频或慢模态”本身不新 |
| [NeurKItt](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e88870ec82f2469b0ddf32c817920c68-Abstract-Conference.html) | 学习不变子空间以减少 Krylov 迭代与 wall | 必须与 classical deflation 和 learned-subspace control 比较 |
| [Neural Krylov geometry preconditioning](https://arxiv.org/abs/2507.15452) | 主角度损失、可微 FGMRES 与 Krylov 几何 | “学习 Krylov 几何”不是新主张 |
| [Deep Null Space Learning](https://arxiv.org/abs/1806.06137) | 神经网络、数据一致性与 null-space 分量 | 不能主张首次利用 range/null-space 分解 |
| [Bayes Meets Krylov](https://epubs.siam.org/doi/10.1137/15M1055061) | 用先验与右预条件器改变 CGLS 子空间 | 必须区别受限初值与经典先验预条件 |
| [Spectrally Safe Warm Starts](https://arxiv.org/abs/2606.21828) | warm start 的求解器安全修正与完整大规模成本 | 不能泛称首个 solver-safe warm start |

## 4. 最强反例：解析二维 Krylov 投影

`q0,q1` 已经构成二维 Krylov basis。只使用部署可见的 `y`，就能在
`span(q0,q1)` 内做 exact projected least-squares / Galerkin，解析求出 measurement
residual 最优系数。

所以网络预测 `(c0,c1)` 不能因为“用了学习”获得原创性。正式预注册必须同时比较：

```text
exact 1D line search
exact 2D projected least-squares / Galerkin
zero-start call-matched CGLS
fit-only fixed coefficients
observation-conditioned coefficients
```

只有 observation-conditioned 方法在最终 field/gradient 尾部稳定优于解析投影，
同时没有隐藏额外 `A/A^T`、wall 或 RSS，才能说明它学到了场先验，而不是重新发明
CGLS。

## 5. 必做 controls

### 经典 controls

- Zero-CGLS K3/K4；
- raw BP、geometry-equalized BP、normalized BP；
- exact-diagonal PCGLS；
- dual ridge；
- RBF / reduced basis；
- PCA/DCT；
- pyramid-style coarse initialization；
- classical deflation；
- exact 1D 与 exact 2D Krylov projection。

### 学习 controls

- Cross14 capacity sentinel；
- 同参数量 MLP；
- 小型 3D CNN；
- direct 3D U-Net/FNO/UNO/DeepONet；
- learned-subspace / HINTS-style control；
- 最接近的一次性 neural warm-start control。

所有方法必须共享 trajectory split、forward、inverse、reference/gauge、停止规则、种子、
终点容差和成本账。

## 6. 预期优势怎样才算成立

### A 级：机制成立

- 候选留在规定的可观测 Krylov 子空间；
- 解析尺度使初始 measurement residual 不劣于零初值；
- 后续求解器、算子和终点定义完全不变；
- 每个额外 basis、BP、line search 与 fallback 都记入成本。

这一级只能叫结构正确，不能叫性能成功。

### B 级：公开 proxy 成立

- 完整 trajectory outer LOTO；
- field/gradient/observation p50、p90、worst 与 harm 全部过门；
- 比最强经典 control 减少完整 `A/A^T`；
- fresh-process wall 中位数和尾部不慢；
- whole-pipeline peak RSS 不被模型与缓存抵消；
- fresh holdout 仍成立。

这一级可以支撑公开 CFD proxy 上的方法结果，不能自动写成真实 BOST。

### C 级：真实 BOST 迁移成立

- 绑定真实相机、单位、reference/gauge、现用 baseline 与停止规则；
- 至少一个组内真实 case 复现 matched-accuracy 成本优势；
- 报告光流误差、标定误差、有限孔径或 curved-ray 模型差异；
- 方法失败时能使用部署可见量回退。

这一级才可能形成完整 BOST 论文主张。

## 7. 可允许与禁止的论文表述

当前可写：

> We investigate a BOST-specific, observation-conditioned one-shot initializer
> followed by an unchanged CGLS/PCGLS refinement.

完整检索后可有限定地写：

> Among the public sources reviewed as of 26 July 2026, we did not identify a
> prior method jointly satisfying the complete structure and evaluation
> contract listed in this work.

当前不能写：

- first neural warm start；
- first neural-operator Krylov solver；
- first neural BOST；
- first physics-aware BOST；
- first hybrid neural-classical BOST；
- guaranteed convergence；
- real-time；
- generalizable；
- SOTA；
- real BOST success；
- algorithm breakthrough。

## 8. 能确保什么，不能确保什么

不能诚实保证“全世界绝对没人做过”。公开检索不覆盖所有专利、博士论文、中文数据库、
刚接收未索引稿件和课题组未发表方案。

可以确保：

1. 每个算法零件标明来源，不靠换名字冒充新方法；
2. 用最危险近邻和最强便宜 control 主动反证；
3. 结果前冻结输入、终点、成本、回退和 claim；
4. 保存代码、协议、commit、图表和失败结果的完整谱系；
5. 论文只主张真实通过的部分；
6. 最终形成一套可追溯到 Gao Youyang 独立研究劳动的作品。

截至本轮 20 项核心公开近邻审计，尚未发现同时覆盖七项作品指纹的同构方法。这是
**限定检索范围内的未发现**，不是全球唯一证明。

## 9. 立即向师兄确认

1. 组内是否已有未发表的 neural warm start、BP normalization、Krylov-space
   initializer、pyramid initialization 或相关专利？
2. 真实流程最终重建的是 `rho`、`n`、`n-1` 还是 `Delta n`，现用 solver、停止规则
   和主要耗时分别是什么？
3. 公开 proxy 过门后，哪一个最小真实 BOST case 可用于迁移，能够提供哪些相机参数、
   displacement、reference/gauge 和 baseline 输出？

这三问同时保护科学方向、组内知识产权和真实迁移。
