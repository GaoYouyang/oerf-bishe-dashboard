# C 路线锁定：面向 BOST 三维重建的神经算子 Warm Start

> 状态：`DIRECTION_LOCKED_BY_SENIOR / ALGORITHM_NOT_YET_VALIDATED`
>
> 2026-07-23，何远哲师兄选择 C：**用算子学习生成 warm start，在最终精度相同的前提下降低三维重建成本。**
>
> 这是选题收敛，不是算法突破。本文冻结问题、比较协议和下一步接口；在真实或独立高保真数据完成前，不宣称优于现有方法。

## 1. 一句话研究问题

给定多视角 BOST 位移观测 \(y\)、相机/光线几何 \(g\) 和物理 forward \(F_g(x)\)，在参考状态 \(x_{\rm ref}\) 处记局部 Jacobian 为 \(J_g\)。这里的重建变量 \(x\) 必须是相对已知背景的折射率或密度扰动，而不是带任意常数偏置的绝对场。学习一个轻量初始化器

\[
x_0 = G_\theta\!\left(x_{\rm ref},\,J_g^\top[y-F_g(x_{\rm ref})],\,g\right),
\]

再由固定的物理迭代求解器细化：

\[
x_{k+1} = \mathcal{S}(x_k; F_g, J_g, J_g^\top, y).
\]

第一阶段若使用冻结直线光线/线性投影，可令 \(x_{\rm ref}=0\)、\(F_g(x)=A_gx\)，上式退化为常见的 \(A_g^\top y\)。若师兄的实际重建使用随场弯曲的光线，则必须保留非线性 outer iteration，并将 warm start 接到同一个 Gauss-Newton/一阶物理求解流程；不能把固定 \(A^\top\) 的结果冒充曲光线算法。

研究目标不是让 \(G_\theta\) 单独给出最漂亮的三维场，而是回答：

> 在相同终点精度、相同数据、相同硬件和相同停止规则下，学习型初值能否稳定减少线性阶段的 `A/A^T`，或非线性阶段的 forward/JVP/VJP 调用与端到端时间，同时不增加坏尾部和跨工况失败？

暂定题目：

- 中文：**面向 BOST 三维反应流场重建的物理校正神经算子 Warm Start 方法**
- English: **Physics-Corrected Neural-Operator Warm Starts for Cost-Efficient 3D BOST Reconstruction**

## 2. 为什么这条路线贴合课题组

1. NeRIF 已证明连续神经折射率场可以服务 BOST 三维体场表达；本课题不重复“再做一个神经场”，而是研究如何更快到达同一物理解。
2. TDBOST/4D BOST 的核心诉求之一是高时空分辨率下的重建效率；若单帧三维 warm start 成立，可自然扩展为相邻帧、低秩时空因子或 tensor state 的初始化。
3. PIV-BOST 表明三维折射率场不是只用于展示，它会进入折射误差补偿等下游任务。因此不能只追求网络指标，必须保留物理细化和重投影验证。
4. 师兄明确判断“速度提升最显著、别人更容易理解”。这使论文主张可以被压缩成清楚的成本-精度前沿，而不是难解释的内部证书或局部代理指标。

## 3. 数据与物理链

### 3.1 第一阶段数据：PoolFire CFD

公开 REALM PoolFire 目前可作为**三维反应流高保真场源**，不是现成 BOST 数据：

- 网格：`80 x 80 x 200`
- 变量：`CH4, CO2, H2O, O2, T, rho, Ux, Uy, Uz`
- 轨迹：15
- 公开仓库体量约 98 GB

本机已下载元数据，但发现需要先审计的差异：

- REALM 网站和仓库旧配置写 21 个时间步；
- 已下载的 `data.npz` 中 `times.shape == (101,)`；
- 已下载的统计 YAML 数值可疑，不能直接用于归一化。

因此，打开首个真实轨迹前只允许声称“元数据已获取”。必须先检查轨迹数组的 key、shape、dtype、时间轴、单位和缺失值，再冻结 loader。

### 3.2 从 CFD 到 BOST 训练对

预定链条：

1. 读取 PoolFire 的密度场 \(\rho(x,y,z)\)，并从实验可得的环境/flow-off/boundary 条件冻结 \(\rho_{\rm ref}\)；
2. 在师兄确认波长、混合物假设和单位后，用 Gladstone-Dale 关系构造 \(x=\Delta n=n-n_{\rm ref}\)，或等价的 \(\Delta\rho=\rho-\rho_{\rm ref}\)；
3. 用与组内一致或明确近似的 BOST forward 生成多视角位移 \(y=F_g(x)+\epsilon\)；
4. 在线性阶段计算 \(b=A_g^\top y\)；非线性阶段计算参考状态的伴随残差 \(b=J_g^\top[y-F_g(x_{\rm ref})]\)；
5. 训练 \(G_\theta(b,g)\to x_0\)；
6. 用固定 CGLS/PCGLS/PDHG 或组内求解器从 \(x_0\) 继续；
7. 在未见 trajectory、火源功率/尺寸和几何上做 matched-accuracy 计时。

必须按完整 trajectory 或物理工况切分，禁止随机帧切分。相邻帧高度相关，随机帧会把同一火焰演化泄漏到训练与测试。

### 3.3 重建变量与 gauge

BOS 主要响应折射率梯度，空间常数偏置通常不可由观测辨认。因此所有方法必须共享同一个 gauge/reference 合同：

- 首选重建 \(x=\Delta n=n-n_{\rm ref}\)；若用密度参数化，则重建 \(\Delta\rho=\rho-\rho_{\rm ref}\)。
- \(n_{\rm ref}/\rho_{\rm ref}\) 只能来自实验可得的环境值、flow-off 图像、已知边界或训练/validation 冻结规则，不能从 test truth 的体均值反推。
- 若没有物理 reference，则对所有方法施加相同的 Dirichlet/零均值 gauge，并同时报告 gauge-invariant 的梯度/H1 指标；不能只给神经网络绝对背景标签。
- baseline、学习器和 scorer 使用相同背景扣除、边界约束与后处理。网络不得读取功率/尺寸标签来猜不可观常数，除非这些量在部署时同样提供给所有基线。

## 4. 最小算法：先把成本账算对

旧的 `JACRU-M2.9 fixed warm + CGLS` 合成实验给出一个重要负结果：学习型 proposal 的场误差可能更低，但旧特征准备需要约 `13F + 13A^T`，已经吃掉了迭代预算，无法形成真正端到端加速。

因此第一版冻结为 Lean Geometry-Conditioned Warm Operator：

```text
input preparation:  b = A^T y                         0F + 1A^T
learned proposal:   x0 = G_theta(b, geometry)         network only
fresh projection:   r0 = y - A x0                    1F + 0A^T
refinement:         CGLS-23                           23F + 23A^T
candidate physical-call subtotal:                     24F + 24A^T
control physical-call subtotal:                       24F + 24A^T
```

这只是物理调用小计的第一张机制图，不含网络推理、几何准备、数据搬运和同步。端到端账本必须另报这些成本。真正的论文结果应比较“达到同一精度需要多少步”，不能预先固定 candidate 一定用 23 步。

## 5. 三个算法备选

### C0：BP-GNO Warm Start

- 输入：一次反投影 \(A^\top y\) + 几何通道。
- 模型：小型 3D U-Net/FNO/CNO 中选择一个主模型，其他作为容量匹配对照。
- 作用：建立最简、最容易复现的 warm-start 基线。
- 风险：直接 field-L2 训练可能给出看似接近真值、但测量残差很大的初值。
- 优先级：最高，必须先做。

### C1：Observable Krylov Warm Start

- 不让网络自由输出整个体场；先构造便宜、可解释的 Krylov 字典，例如
  \[
  q_0=A^\top y,\quad q_1=A^\top A q_0,\quad q_2=A^\top A q_1,
  \]
  网络只预测少量场系数、频带权重或空间门控，组成 \(x_0\)。
- 优点：初值被限制在与观测相关的子空间内，较容易控制重投影残差和坏尾部。
- 创新可能：**geometry-conditioned、observable-subspace-constrained neural warm start for BOST**。
- 风险：每增加一个字典向量都要增加 `A/A^T` 调用，必须证明省下的迭代超过特征成本。
- 优先级：第二；C0 跑通后做。

### C2：Frontier-Trained Correctable Warm Start

- 训练目标不是单独最小化 \(\|x_0-x^\ast\|\)，而是同时优化：
  - 初值场误差；
  - 初值重投影残差；
  - 经过固定 1/2/4 次物理细化后的误差；
  - 在冻结部署预算内的可纠正性。
- 直觉：最好的 warm start 不一定是单步最准的预测，而是**最容易被后续物理求解器纠正**的预测。
- 可实现形式：
  \[
  L = \lambda_0 L_{\text{field}}(x_0)
    + \lambda_1 L_{\text{meas}}(F_g(x_0),y)
    + \sum_{k\in\{1,2,4\}}\lambda_k L_{\text{field}}(x_k).
  \]
- 推理时间和物理调用数默认是**外层模型选择约束**，不是随手加入的可微 loss。只有定义并验证了明确的 architecture-cost proxy，才允许把成本代理写进训练目标。
- 创新可能：把 `time-to-target` 或固定短程 refinement trajectory 直接写进 BOST warm-start 学习。
- 风险：反传穿过迭代器会增大训练显存和工程复杂度；必须先用截断/停止梯度版本做小场证伪。
- 优先级：第三；只有 C0/C1 显示真实 headroom 才进入。

不建议第一阶段直接做大型 FFNO、扩散模型或 4D Transformer。它们会让“到底是模型容量、时间先验还是 warm start 机制带来收益”无法归因。

## 6. 公平比较合同

### 6.1 必须存在的基线

1. `Zero-CGLS`：零初值或组内默认初值 + 固定求解器。
2. `BP-CGLS`：\(A^\top y\) 或组内便宜解析初值 + 同一求解器。
3. `Classical-PCGLS`：最强、部署可得的经典预条件器。
4. `Direct operator`：FNO/DeepONet/3D U-Net 直接输出，不做物理细化。
5. `Operator + refinement`：本课题候选。
6. `Oracle warm`：只用于估计 headroom，绝不作为可部署结果。

所有方法共用 forward、adjoint、数据划分、归一化、容差、硬件与计时器。模型训练成本和单次部署成本分开报告。

### 6.2 “同精度”如何定义

在 validation 上冻结场/观测等价区间、部署停止规则与最大预算，test 只开一次。建议同时给两种误差：

- 场门：\(\|x_k-x^\ast\|_2/\|x^\ast\|_2 \le \tau_x\)，其中 \(x\) 已按 3.3 节冻结 reference/gauge
- 观测门：\(\|F_g(x_k)-y\|_2/\|y\|_2 \le \tau_y\)

必须拆成两张账，不能混写：

1. **离线 oracle headroom**：在有真值的合成 test 上，事后读取完整轨迹，报告每个 case 第一次达到冻结 \((\tau_x,\tau_y)\) 的调用与时间。它只回答“理论上最多能省多少”，不能冒充部署停止器或真实在线耗时。
2. **部署可用主结果**：实际停止只能使用 validation 冻结的迭代数、measurement discrepancy/重投影、保留相机或组内认可的部署可见量。test truth 只在停止后用于检查终点场误差是否落入预注册等价区间，不能决定何时停。

若实验数据没有完整场真值，部署主门使用组内认可的重投影、保留相机、下游 PIV 补偿或高保真 reference。离线 oracle 表继续单独标注，不替代部署主表。

两张账都对每个 test case 记录：

- 线性阶段的 `N_A`、`N_AT`；非线性阶段另记 forward、JVP、VJP、ray/sample 总量；
- 网络推理、数据搬运、forward/adjoint、求解器和总 wall time；
- 峰值内存；
- 终点 field relative-L2、gradient/H1、measurement residual；
- p50、p90、worst case、harm rate 和未达标率。

论文主表必须以**部署可用停止规则**给出 paired per-case calls/time 与终点等价性；离线 `time-to-target` 作为 headroom 辅表。不能只报平均推理速度，也不能用 test truth 控制真实运行。

## 7. 论文成功门与停止条件

### 最低可继续门

- 在独立 PoolFire trajectory 上，至少三种子；
- 部署可用停止规则下，终点误差通过等价门，且 median 与 p90 的物理调用都减少；
- 端到端 wall time 有稳定正收益，而不是只省迭代、网络更慢；
- 终点场误差、重投影和坏尾部不恶化；
- 对未见功率/尺寸或未见几何至少有一个 OOD 门。

### 高质量论文候选门

- 在 PoolFire 高保真合成 BOST 与至少一个独立公开/组内 BOST 样例上成立；
- 相对最强经典初始化和至少两类学习初始化仍有收益；
- C0/C1/C2 消融能说明收益来自哪一个结构；
- 报告失败域，并给出 fail-closed fallback：不可信时退回经典初值；
- 师兄确认物理 forward、评价指标和实际速度瓶颈与组内代码一致。

### 立即停止或降级

- 学习初值推理成本大于省下的物理迭代；
- 只在随机帧 IID 成立，按 trajectory/OOD 后消失；
- field 指标改善但 measurement residual 或下游物理量恶化；
- 部署停止或 fallback gate 使用了 oracle truth；离线 headroom 曲线必须另表标注；
- 相同精度无法达到，只能拿更差终点换速度；
- PoolFire forward 与组内 BOST forward 差异过大且无法校准。

## 8. 接下来两周

### 第 1 周：冻结可运行数据合同

1. 下载一个 PoolFire train trajectory，不先下载全部 98 GB。
2. 低内存检查 NPZ 成员、shape、dtype、单位、时间轴和 NaN/Inf。
3. 让师兄确认 `rho_ref -> Δn` 的波长/常数、reference/gauge、BOST observable 与当前基线求解器。
4. 用降采样 `32^3` 或 `40 x 40 x 64` 扰动场生成第一个多视角 toy。
5. 做 adjoint dot test，确认 \(A^\top\) 与 \(A\) 配对。

### 第 2 周：只跑 C0 最小闭环

1. `Zero-CGLS`、`BP-CGLS`、`PCGLS` 三个强基线。
2. 小型 3D U-Net 或 FNO：`A^T y + geometry -> x0`。
3. 主表使用 validation 冻结的部署停止规则并检查终点等价；另画 oracle time-to-target headroom。
4. 先用 3 个 trajectory 做 train/val/test smoke，确认流程后再扩数据。
5. 若 C0 没有 headroom，先查初值 measurement residual 与 feature cost，不扩模型。

## 9. 现在向师兄确认的四件事

```text
那我锁定 C：算子学习 warm start，在最终精度相同的前提下降低三维 BOST 重建成本。
具体准备使用 PoolFire CFD 密度场生成多视角 BOST 位移，神经算子输出三维初值，再接固定的物理迭代。

你告诉我：
1. 组里当前要加速的重建基线具体是哪一个，入口脚本/函数是什么；
2. forward 和 adjoint/自动微分代码入口在哪里；
3. “同精度”主要按三维场误差、重投影误差还是下游指标判断；
4. PoolFire 是直接用 REALM 的公开数据，还是组里另有版本。
```

还需要确认一个关键语义：

> REALM 原任务是“当前 CFD 场预测未来 CFD 场”；本课题拟做的是“CFD 密度真值生成 BOST 观测，再从观测重建三维场”。两者不是同一个任务。请师兄确认支持后者。

## 10. 第一批必读文献

1. He et al., **Neural refractive index field**
   [arXiv](https://arxiv.org/abs/2409.14722) · [Physics of Fluids](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)
   提取：组内 BOST forward、折射率场参数化、损失、实验评价。不能据此声称 warm start 已有效。

2. He et al., **Tensor Decomposition-Based Four-dimensional BOST**
   [ACM DOI](https://doi.org/10.1145/3809488)
   提取：组内当前效率瓶颈、4D 参数化和实际成本口径。需要正式全文核对后再引用具体数值。

3. Eshaghi et al., **NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers**
   [arXiv](https://arxiv.org/abs/2511.02481)
   提取：operator initialization + classical Krylov solver 的直接邻近工作。它让“warm start”本身不再足够新颖；BOST 物理、可观测子空间、geometry conditioning 与 matched-accuracy 证据必须构成本课题差异。

4. Zhou et al., **Neural operator-based super-fidelity**
   [arXiv](https://arxiv.org/abs/2312.11842)
   提取：学习初值而非替代求解器、端到端成本和跨求解器评价。不能把 steady forward PDE 的结论直接迁移到 BOST 逆问题。

5. Molinaro et al., **Neural Inverse Operators for Solving PDE Inverse Problems**
   [PMLR](https://proceedings.mlr.press/v202/molinaro23a.html)
   提取：inverse operator 的输入/输出结构和 FNO/DeepONet 组合。作为直接预测基线，不代表物理细化后的 time-to-target。

6. Mao et al., **REALM: Benchmarking neural surrogates on realistic spatiotemporal multiphysics flows**
   [arXiv](https://arxiv.org/abs/2512.18595) · [PoolFire 数据](https://huggingface.co/datasets/realm-bench/realm-bench-PoolFire/tree/main)
   提取：PoolFire 的物理工况、trajectory split 和已有 surrogate 失败。它提供高保真 CFD 场，不提供 BOST 光学链。

7. Sambharya et al., **Learning to Warm-Start Fixed-Point Optimization Algorithms**
   [JMLR](https://jmlr.org/papers/v25/23-1174.html)
   提取：fixed-point residual/solution-distance 两类训练目标、固定下游迭代与未见样本的理论条件。必须逐项检查这些条件能否迁移到 BOST 逆问题。

8. Zhou et al., **A Neural Network Warm-Start Approach for the Inverse Acoustic Obstacle Scattering Problem**
   [arXiv](https://arxiv.org/abs/2212.08736)
   提取：学习初值接传统逆问题求解器、有限孔径和噪声测试，以及训练样本复杂度限制。它比 forward PDE warm start 更接近本课题，但物理算子与三维场参数化仍不同。

## 11. 当前证据边界

- 已确认：师兄选择 C；研究问题和比较原则可冻结。
- 已有：合成 BOST forward/adjoint、CGLS/PCGLS 强基线、旧 warm-start 负结果和调用记账框架。
- 尚未完成：PoolFire 首轨迹下载与数组审计、\(\rho\to n\) 常数、真实 BOST forward 对齐、C0 训练、matched-accuracy 结果。
- 因而当前允许表述：**方向已锁定，最小实验已设计。**
- 当前禁止表述：**算法已创新、速度已提升、优于 FNO/DeepONet、可泛化、可投稿。**
