# LGWO-A24 / N5-D5：14 天保姆式学习与复现实验路线

> 对象：尚未系统学过流体力学、层析重建与 Krylov 方法的物理本科生
> 每日投入：建议 2--3 小时，连续 14 天
> 硬件结论：**这 14 天全部不需要 GPU，也不应租卡**
> 最终边界：第 14 天最多达到“能解释、能复跑、能接真实接口的实验准备状态”，**不等于训练完成、算法成功、真实 BOST 成功、泛化成立或论文成功**。

## 0. 先把终点说清楚

这 14 天不是让你背完所有公式，也不是让你立刻训练 DeepONet、FNO 或 LGWO-A24。真正目标只有六个：

1. 能用自己的话解释三维 BOST 从折射率场到背景位移的物理链条。
2. 能手算并用代码验证一次伴随恒等式，知道“伴随不是逆”。
3. 能解释 CGLS 为什么只需 `A` 与 `A^T`，以及 Krylov 子空间从哪里来。
4. 能读懂仓库中的 PSU 风格 matrix-free BOST 算子、固定调用预算和基础测试。
5. 能准确说出 LGWO-A24 只改 CGLS 首方向的窄假设、允许输入、成本合同和科学红线。
6. 收到何远哲师兄回复后，能按“有无 JVP/VJP、是否有 native residual、真实主痛点”选择下一项小实验，而不是先选一个网络盲训。

当前必须牢记的事实：

```text
scientific cases = 0
optimizer steps = 0
real BOST authorization = false
algorithm breakthrough = none
```

仓库已有大量工程测试，只说明代码合同可检查。它们不能代替真实数据、真实前向模型、科学对照或论文结果。

## 1. 每天固定节奏与记录位置

建议每天按同一节奏推进：

| 模块 | 时间 | 做法 |
| --- | ---: | --- |
| 直觉 | 20--40 分钟 | 不看代码，先画图、说人话、写量纲 |
| 代码与实验 | 60--80 分钟 | 只跑当天列出的文件和测试 |
| 手算或口述 | 20--30 分钟 | 不借助 AI，先自己答，再对照 |
| 记录与复盘 | 20--30 分钟 | 保存命令、输出、错误和自己的解释 |

第一次开始时，在终端执行：

```bash
export OERF_REPO="/path/to/oerf-bishe-dashboard"
export OERF_LOG="$HOME/Desktop/OERF_14天学习记录"
mkdir -p "$OERF_LOG"
cd "$OERF_REPO"
```

每天重新打开终端时，重新执行上面四行。所有个人笔记放在桌面记录目录，不要把实验室私有代码、几何、权重或 raw trace 写进公开仓库。

### 卡住时的统一处理顺序

1. 先保存完整报错，不要只截最后一行。
2. 检查自己是否在仓库根目录：`pwd`。
3. 检查解释器：`.venv/bin/python --version`。
4. 把任务缩到当天列出的单个测试。
5. 仍失败时，记录“输入、预期、实际、已尝试”，当天不跨级到训练。

## 2. 14 天总览

| 天 | 主线 | 当天不做什么 | GPU |
| ---: | --- | --- | --- |
| 1 | BOST 物理链条与三维未知量 | 不看网络结构 | 不需要 |
| 2 | 线性算子、内积与伴随 | 不把 `A^T` 当 `A^{-1}` | 不需要 |
| 3 | PSU matrix-free 三维 BOST 算子 | 不追求大网格 | 不需要 |
| 4 | CGLS 与 Krylov 子空间 | 不调参、不训练 | 不需要 |
| 5 | 欠定性、gauge、support 与正则化 | 不宣称唯一真解 | 不需要 |
| 6 | 算子学习路线图与近邻工作 | 不写“首次”或“突破” | 不需要 |
| 7 | LGWO-A24 的窄方法与结构界 | 不运行 scientific fit | 不需要 |
| 8 | 可部署输入、A/B 几何与泄漏防线 | 不让 proposal 看 truth/B | 不需要 |
| 9 | 五个消融臂与公平基线 | 不只和弱 FNO 比 | 不需要 |
| 10 | 指标、尾部风险与成本账本 | 不只报平均误差/网络推理时间 | 不需要 |
| 11 | N5-D5 的 53 次接口证书 | 不把 synthetic PASS 写成物理正确 | 不需要 |
| 12 | 给师兄的最小真实接口沟通 | 不索要整套私有数据 | 不需要 |
| 13 | 按师兄回复分支接线 | 不在接口语义不清时开训 | 不需要 |
| 14 | 复现包、口试与下一阶段 GO/NO-GO | 不写论文结论 | 不需要 |

---

## Day 1：先看懂 BOST 在测什么

**建议时间：** 2 小时 20 分钟。**GPU：不需要。**

### 学习目标

建立一条不含神经网络的物理链：反应流中的温度/组分变化，导致密度变化；密度通过 Gladstone--Dale 关系影响折射率；折射率横向梯度使背景光线偏折；相机记录背景图案位移；多视角位移再用于三维反演。

### 20--40 分钟直觉

把折射率场想成空间中缓慢变化的“光学地形”。光线沿传播方向穿过它，真正推动横向偏折的是垂直于光线的折射率梯度，而不是折射率绝对值本身。直线射线、小角度近似下，可把单条 ray 的二维位移抽象为

```math
y_r \approx C_r \int_{\text{ray }r} P_r\nabla n(\mathbf{x})\,\mathrm{d}s.
```

其中 `n(x)` 是三维折射率场，`P_r` 把三维梯度投影到相机的 `u/v` 两个方向，`C_r` 汇总光学几何与系统常数。离散后就是 `y = A x + noise/model mismatch`。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '1,120p' docs/n5_d5_minimum_real_interface_bridge_2026-07-19.md
sed -n '1,90p' demo_t16_operator/psu_b0_reconstruction_interface.py
printf '# Day 1：BOST 物理链\n\n' > "$OERF_LOG/day01.md"
```

在 `day01.md` 画出或用箭头写出：

```text
temperature/species -> density -> refractive index -> ray deflection
-> background displacement -> multi-view inverse problem -> 3D field
```

再写清每个箭头需要的假设，例如 Gladstone--Dale 系数、几何标定、小角度、ray path、背景位移提取。

### 手算或口述检查

不看资料回答：

1. 均匀增加整个空间的折射率，为什么可能几乎不产生 BOS 位移？
2. 一台相机的一条 ray 最终给几个观测分量？
3. 从 detector 位移直接恢复每个 voxel，为什么会欠定？

### 产物

`$OERF_LOG/day01.md`，至少含一张物理链图、三个主要近似和三个误差源。

### 退出标准

能在 90 秒内向同学解释“BOST 不是拍到三维密度，而是从多视角背景位移反演折射率/密度场”，且不提任何网络名。

### 卡住后的降级路线

先只保留三句话：测到的是背景位移；位移来自折射率横向梯度的路径积分；多视角和先验共同约束三维场。当天不必推导完整几何光学。

---

## Day 2：伴随不是逆，但它把 detector 残差搬回三维场

**建议时间：** 2 小时 30 分钟。**GPU：不需要。**

### 学习目标

理解线性算子 `A`、伴随 `A^T`、Jacobian、JVP 与 VJP 的角色；会用内积恒等式检查一对离散 forward/adjoint。

### 20--40 分钟直觉

- `A x`：给定三维场，预测 detector 位移。
- `A^T q`：把 detector 上的权重或残差 `q` 反投影到三维场空间。
- `A^T` 通常不是逆；有限视角下，很多不同的 `x` 可以产生相似 `A x`。
- 非线性 forward `F(x)` 在当前点附近用 Jacobian `J(x)` 描述。`Jv` 是场扰动对 detector 的影响；`J^Tq` 是 detector 残差对场/参数的反向作用。

离散伴随最基本的检查是：

```math
\langle Ax,q\rangle = \langle x,A^Tq\rangle.
```

这只能证明两段代码互为转置，不能单独证明物理 forward 正确。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '230,430p' demo_t16_operator/psu_b0_reconstruction_interface.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py::test_finite_difference_gradient_has_exact_declared_adjoint \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py::test_matrix_free_b0_operator_passes_dot_product_identity
printf '# Day 2：伴随检查\n\n' > "$OERF_LOG/day02.md"
```

把两个测试分别在检查哪一层写入 `day02.md`：第一项是有限差分梯度 `G/G^T`；第二项是完整 BOST 离散算子 `A/A^T`。

### 手算或口述检查

令

```math
A=\begin{bmatrix}1&2\\0&1\end{bmatrix},\quad
x=\begin{bmatrix}1\\-1\end{bmatrix},\quad
q=\begin{bmatrix}2\\3\end{bmatrix}.
```

手算 `Ax`、`A^Tq`、`q^T(Ax)` 与 `x^T(A^Tq)`。两边都应为 `-5`。然后口述为什么这仍不证明 `A` 是正确的 BOST 物理模型。

### 产物

`$OERF_LOG/day02.md`，含完整手算、两个测试名、你对“dot test 的能力边界”的一句话。

### 退出标准

能无提示说出：forward 把 field 送到 measurement，adjoint 把 measurement residual 拉回 field；伴随恒等式通过不等于 forward 物理正确。

### 卡住后的降级路线

先把 `A` 当普通矩阵，暂时不管无限维函数空间。只要求会算矩阵转置和四个内积；第二天再回看代码中的 `finite_difference_gradient_adjoint`。

---

## Day 3：拆开 PSU 风格 matrix-free 三维 BOST 算子

**建议时间：** 2 小时 40 分钟。**GPU：不需要。**

### 学习目标

能沿 shape 追踪仓库中的直线射线算子：voxel 场、有限差分梯度、三线性插值、相机平面投影、沿 ray 求和和系统尺度；理解 matrix-free 的含义。

### 20--40 分钟直觉

仓库中的 `PSUB0VoxelGradientOperator` 没有存一张巨大的稠密矩阵。它把 `A` 分解成可检查的原语：

```text
x[z,y,x]
  -> Gx[3,z,y,x]
  -> trilinear sample[3,ray,sample]
  -> project to u/v[ray,sample,2]
  -> sum and scale[ray,2]
```

伴随则严格反序执行：尺度与投影转置、scatter-add 三线性插值转置、有限差分转置。matrix-free 节省存储，也让 CGLS 只通过 callable 使用 `A/A^T`。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '430,860p' demo_t16_operator/psu_b0_reconstruction_interface.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py::test_matrix_free_forward_matches_tiny_materialized_matrix \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py::test_fixed_denominator_keeps_invalid_sample_slots \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py::test_grouped_adjoint_matches_individual_groups_and_pooled_sum
printf '# Day 3：PSU 算子 shape 账本\n\n' > "$OERF_LOG/day03.md"
```

在笔记中给 `_forward` 与 `_adjoint` 的每个中间张量写 shape，并解释为什么 invalid sample 权重为零，但分母仍保留固定 sample slot。

### 手算或口述检查

1. 三线性插值一个采样点最多连接几个 voxel？
2. 为什么 `scatter_add_` 是插值 gather 的自然伴随？
3. 若显式矩阵有 `M` 个观测和 `N` 个 voxel，需要存 `M x N`；matrix-free 版本主要存什么？

### 产物

`$OERF_LOG/day03.md`，含 forward/adjoint 双向流程图、shape 账本和固定分母解释。

### 退出标准

能指着 `PSUB0VoxelGradientOperator._forward` 逐段说出物理/数值意义，并知道它只是 PSU 风格公开基线接口，不等于实验室 NeRIF/TDBOST 的真实 curved-ray renderer。

### 卡住后的降级路线

先忽略 batch 和 camera grouping，只追踪一个体场、一条 ray、三个 sample、两个 detector 分量。只要单例流程通了，再恢复 batch 维度。

---

## Day 4：CGLS 与 Krylov 不是神秘网络，而是逐步搜索

**建议时间：** 2 小时 40 分钟。**GPU：不需要。**

### 学习目标

理解 CGLS 用于最小化 `||Ax-y||_2^2`；知道每一步只调用 forward/adjoint，不需要显式 `A^TA`；能解释 Krylov 子空间。

### 20--40 分钟直觉

从 `x0=0` 出发，第一条最自然的下降信息是

```math
g_0=A^Ty.
```

CGLS 后续方向由残差和 `A^T` 递推产生。第 `k` 步解位于

```math
\mathcal K_k(A^TA,A^Ty)
=\operatorname{span}\{A^Ty,(A^TA)A^Ty,\ldots,(A^TA)^{k-1}A^Ty\}.
```

“共轭”不是方向彼此普通正交，而是在问题相关的度量中避免重复搜索。有限精度下还要关心重正交化、分母 breakdown 和调用账本。

### 动手任务

```bash
cd "$OERF_REPO"
rg -n "def cgls|class CGLS|forward_calls|adjoint_calls" demo_t16_operator/interface_baselines.py
sed -n '1,180p' demo_t16_operator/interface_baselines.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_interface_baselines.py::test_cgls_uses_exact_fixed_call_budget_and_preserves_support \
  demo_t16_operator/test_interface_baselines.py::test_cgls_converges_for_simple_identity_operator \
  demo_t16_operator/test_psu_b0_classical_baselines.py::test_identity_preconditioned_cgls_matches_cgls
printf '# Day 4：CGLS 手算与调用账本\n\n' > "$OERF_LOG/day04.md"
```

### 手算或口述检查

取 `A=diag(1,2)`、`y=(1,1)^T`、`x0=0`：

1. `s0=A^Ty=(1,2)^T`，`p0=s0`。
2. `q0=Ap0=(1,4)^T`。
3. `alpha0=(s0^Ts0)/(q0^Tq0)=5/17`。
4. `x1=(5/17,10/17)^T`。
5. 验证新 residual `r1=y-Ax1=(12/17,-3/17)^T` 与 `q0` 正交。

口述每一步至少需要哪一次 `A` 和哪一次 `A^T`。

### 产物

`$OERF_LOG/day04.md`，含上述手算、一段不超过 12 行的 CGLS 伪代码和一次调用预算说明。

### 退出标准

能回答“为什么神经网络不必直接输出整个三维场，而可以只建议第一条搜索方向”，同时承认这只是 LGWO-A24 的待检验设计动机。

### 卡住后的降级路线

只做两维对角矩阵的一步更新，不追完整 `beta` 递推。当天最低目标是理解 `A^T residual` 给出 field-space 搜索信息。

---

## Day 5：看见欠定性、gauge、support 和正则化

**建议时间：** 2 小时 30 分钟。**GPU：不需要。**

### 学习目标

理解 BOS 对常数偏移不敏感的 gauge/nullspace、有限视角造成的低可观测方向，以及 support/边界条件与正则化为什么不可随意混用。

### 20--40 分钟直觉

若观测主要依赖 `n` 的梯度，则 `n+c` 和 `n` 可能产生相同位移。除此之外，有限相机角度还会留下更多弱可观测方向。求解器必须明确：

- 如何固定加性自由度，例如 outer boundary 置零；
- 哪些 voxel 属于物理 support；
- 是用 Tikhonov、Sobolev、TV/edge 还是表示网络提供先验；
- 观测模型错误与正则化偏差如何区分。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '360,455p' demo_t16_operator/psu_b0_reconstruction_interface.py
sed -n '1,250p' demo_t16_operator/psu_b0_classical_baselines.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py::test_dirichlet_gauge_removes_boundary_and_respects_support \
  demo_t16_operator/test_psu_b0_classical_baselines.py::test_quadratic_tikhonov_is_monotone_and_call_matched \
  demo_t16_operator/test_psu_b0_classical_baselines.py::test_zero_lambda_matches_unpreconditioned_exact_line_search
printf '# Day 5：不可辨识性与先验\n\n' > "$OERF_LOG/day05.md"
```

在笔记中分开写三列：物理不可辨识、几何欠采样、数值/噪声不稳定。每列至少给两个例子，不要都叫“数据少”。

### 手算或口述检查

1. 对常数体场 `x=c`，离散梯度 `Gx` 为何接近零？边界离散会带来什么例外？
2. `A delta` 很小能否推出 `delta` 是 exact null-space 向量？为什么不能？
3. support mask 是部署可观测的几何先验，还是可偷偷携带 truth 的标签？怎样防止后者？

### 产物

`$OERF_LOG/day05.md`，含三类困难表和一段“为什么 field relative-L2 不能由 measurement residual 单独替代”的解释。

### 退出标准

能明确区分 nullspace、near-null/low-observability、噪声与 forward model mismatch，不再把它们统称为“网络拟合不好”。

### 卡住后的降级路线

用导数算子的一维例子：`D[x1,x2,x3]=[x2-x1,x3-x2]`。先找出所有满足 `Dx=0` 的向量，再讨论多相机 BOST。

---

## Day 6：把“算子学习”拆成四种完全不同的工作

**建议时间：** 2 小时 30 分钟。**GPU：不需要。**

### 学习目标

能区分直接逆映射、逐例隐式场、神经预条件器/迭代器和 LGWO-A24 首方向策略；知道 DeepONet/FNO/DCDM/FCG-NO/NeRIF/TDBOST 是近邻而不是空白背景。

### 20--40 分钟直觉

| 路线 | 学什么 | 典型近邻 | 主要风险 |
| --- | --- | --- | --- |
| 直接逆算子 | `y, geometry -> x` | DeepONet、FNO、iFNO | 训练分布外几何与数据一致性 |
| 逐例神经表示 | 用网络参数表示当前 `x` | NeRIF、NeDF、TDBOST | 每例优化成本与局部极值 |
| 学习预条件/每步方向 | 改善整个 Krylov 过程 | DCDM、FCG-NO | 成本核算、稳定性、强基线 |
| 有界首方向策略 | 只改 `d0`，随后回到固定 shell | 当前 LGWO-A24 假设 | 首方向是否足够、是否只是缩放 |

这些路线都可以使用神经网络，但论文问题、部署成本和失败方式完全不同。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '1,170p' docs/lgwo_a24_l1_novelty_claim_boundary_2026-07-20.md
sed -n '170,330p' docs/lgwo_a24_l1_novelty_claim_boundary_2026-07-20.md
rg -n "DeepONet|FNO|DCDM|FCG-NO|NeRIF|NeDF|TDBOST" docs/lgwo_a24_l1_novelty_claim_boundary_2026-07-20.md
printf '# Day 6：四类算子学习路线\n\n' > "$OERF_LOG/day06.md"
```

优先读一级来源的摘要/方法图：

- [DeepONet](https://doi.org/10.1038/s42256-021-00302-5)
- [FNO](https://arxiv.org/abs/2010.08895)
- [DCDM](https://proceedings.mlr.press/v202/kaneda23a.html)
- [FCG-NO](https://proceedings.mlr.press/v235/rudikov24a.html)
- [NeRIF](https://doi.org/10.1063/5.0250899)
- [TDBOST](https://doi.org/10.1145/3809488)

每篇只回答四问：输入是什么、输出是什么、物理算子何时被调用、训练与单例推理各花多少成本。不要第一遍就读所有网络层。

### 手算或口述检查

给出一个新方法时，连续追问：它直接输出 `x`，还是输出 warm start、方向、预条件器或 residual？部署是否调用 `A/A^T`？几何改变时哪些输入变？

### 产物

`$OERF_LOG/day06.md`，含四路线对比表和六篇论文各四行摘要。

### 退出标准

能解释为什么“把 FNO 接到 BOST”本身不构成创新，以及 LGWO-A24 不能声称“首次神经 Krylov”或“learned null-space projection”。

### 卡住后的降级路线

只比较两个极端：直接网络 `y -> x` 与传统 CGLS。然后把 warm start、preconditioner、first-direction 依次放在两者之间。

### Day 2--6 综合检查点：先跑一个 CPU 小实验

现在可以把伴随、gauge、CGLS 半收敛、operator shift 与最小几何条件化放进同一个确定性实验：

```bash
cd "$OERF_REPO"
.venv/bin/python -m pytest -q learning_labs/test_operator_foundations_lab.py
.venv/bin/python -m learning_labs.operator_foundations_lab \
  --output-dir "$OERF_LOG/operator_foundations_my_run"
```

先读 [算子基础小实验中文导读](operator_foundations_lab_guide_2026-07-21.md)，再看四联图。你需要亲口解释：为什么伴随检查不证明物理正确、为什么常数平移不可见、为什么 CGLS residual 继续下降时 field error 会升高、为什么带几何的最小线性映射只能算条件化基线、为什么掌握精确算子的 Tikhonov 是 privileged teacher。

这个实验的证据等级是 `EDUCATIONAL_SYNTHETIC_LINEAR_PROXY_ONLY`。它不是 DeepONet/FNO 训练，不是三维 BOST 重建，也不授权算法、外推或论文成功主张。

### Day 6.5 可选桥接：亲手看见 `A_true != A_est`

如果 Day 3 的 ray geometry 和 Day 6 的 geometry-conditioned operator 已经能说清，再跑第二个 CPU 小实验：

```bash
cd "$OERF_REPO"
.venv/bin/python -m pytest -q learning_labs/test_calibration_mismatch_lab.py
.venv/bin/python learning_labs/calibration_mismatch_lab.py \
  --output-dir "$OERF_LOG/calibration_mismatch_my_run"
```

先读 [三维 BOST 标定失配小实验导读](calibration_mismatch_lab_guide_2026-07-21.md)。完成后应能解释：

1. 为什么生成观测的 `A_true` 与重建收到的 `A_est` 必须分别记账；
2. 为什么 held-out-camera residual 最小仍可能在零失配时误修正；
3. half-step damping、single-frame LCB 与 multiframe camera-block LCB 分别牺牲了什么；
4. 为什么跨帧共享标定只有在场景对该参数有可观测信息时才增加功效；
5. 为什么残差曲线只能先定义 reliability weight；真正的 observability 还需要 geometry JVP/VJP，而两者都比无界回归完整相机位姿更容易审计。

默认结果是 `NO-GO`，证据等级为 `SYNTHETIC_3D_BOST_POSE_MISMATCH_MECHANISM_ONLY`。不要通过移动阈值、删除失败 case 或只汇报 2 档结果把它改写成成功。

### Day 6.75 可选桥接：把 reliability、observability 和 field information 分开

上一实验已经跑通并能解释后，再重放冻结的逐相机 ledger：

```bash
cd "$OERF_REPO"
.venv/bin/python -m pytest -q learning_labs/test_calibration_camera_reliability_screen.py
.venv/bin/python learning_labs/calibration_camera_reliability_screen.py \
  --output-dir "$OERF_LOG/calibration_camera_reliability_my_run"
```

先读 [相机可靠性权重回放结果](calibration_camera_reliability_screen_result_2026-07-21.md)，然后完成三张手写卡片：

1. `q_rel`：什么 deployment-visible residual 能说明相机证据可靠，为什么需要独立 sentinel 帧；
2. `q_cal`：geometry JVP/VJP、scaled `J^T J` 的小特征值和参数耦合分别代表什么；
3. `q_field`：一个视角对标定很敏感，为什么仍可能不能约束三维场 near-null mode。

你还要从 `rig_metrics.csv` 找到 2 档 seed 503：主候选虽然仍有正 field gain，但相对 uniform 少约 `0.99` 个百分点。能够主动指出这个尾部，才算真正读懂图，而不是只记住 `11.20%` 的均值。

默认判决是 `POSTOPEN_CAMERA_RELIABILITY_WEIGHT_REPLAY_NO_GO`。它只复用 3,024 条已打开 camera score，新增 forward、重建、训练和 fresh rig 均为 0。完成这一节以后，下一动作不是改权重追 PASS，而是带着 `q_rel/q_cal/q_field` 三分法向何远哲师兄确认真实 JVP/VJP、select/sentinel/target 数据角色和 split。

### Day 6.9 可选桥接：把未知三维场从几何信息里消去

只有在你已经能分清 `q_rel` 与 `q_cal` 后，才跑这个 CPU 剖面实验：

```bash
cd "$OERF_REPO"
.venv/bin/python -m pytest -q learning_labs/test_calibration_qcal_profile_lab.py
.venv/bin/python -m learning_labs.calibration_qcal_profile_lab \
  --output-dir "$OERF_LOG/calibration_qcal_profile_my_run"
```

先读 [`q_cal` 剖面结果导读](calibration_qcal_profile_result_2026-07-21.md)，再看四联图和 `alpha_metrics.csv`。你需要不看答案讲清五件事：

1. 为什么 raw `C^T C` 大于零仍不能证明 joint reconstruction 中几何可辨识；
2. 为什么 `A in R^(300 x 1000)` 满行秩时，自由 voxel nuisance 会使 `S0=0`；
3. 为什么 `S_lambda>0` 必须叫 prior-conditioned curvature，不能叫 data-only Fisher information；
4. 为什么参考 alpha 的排序信号可以过门，总判决仍然是 NO-GO；
5. 为什么六相机 pilot 辅助的下一次三相机布置，不等于 subset-only 当帧重建。

然后用一张纸画出下一研究问题：将每帧 1000 个自由 voxel 换成 4D 共享低秩基后，nuisance tangent 的维数如何变化，`S0` 的最小特征值是否可能从零抬起。这张图比直接搭 DeepONet/FNO 更重要，因为它先证明你知道网络要解决什么结构问题。

当前专项测试为 `13 passed`。结果只是 `POSTOPEN_SYNTHETIC_QCAL_PROFILE_DIAGNOSTIC_ONLY`；真实 callable、实验参数尺度、噪声协方差、subset-only 重建和 sealed rig 均未到位。

### Day 6.95 可选桥接：低秩、输运与 practical identifiability

Day 6.9 的图画完以后，再运行真正的六帧 nuisance-tangent 实验：

```bash
cd "$OERF_REPO"
.venv/bin/python -m pytest -q learning_labs/test_temporal_qcal_tangent_lab.py
.venv/bin/python -m learning_labs.temporal_qcal_tangent_lab \
  --output-dir "$OERF_LOG/temporal_qcal_tangent_my_run"
```

先读 [v2 严格结果](temporal_qcal_tangent_result_2026-07-21.md)，再按下面顺序读图：

1. A 图：为什么未知低秩双因子和固定时间基都仍是 `0/5`；
2. known target 为什么是 `5/5`，它在检查什么实现问题；
3. 精确输运为什么能抬秩，但最弱广义 retention 仍只有约 `1e-4`；
4. B 图：为什么 PCA 曲率更大反而更危险；
5. C 图：rank 不随噪声倍率改变，CRLB/经验误差为什么随 sigma 近似线性；
6. D 图：为什么 NIS 通过仍不足以授权参数更新，还需要置信半径；
7. shared birth 为什么同时改善场模型、削弱几何信息。

完成三项动手题：

- 从 `structural_controls.csv` 找出五类零控制与 known target，口述它们的 nuisance 自由度；
- 从 `q_trials.csv` 对比 noise multiplier `0`、`1/128` 和 `1` 的 plugin/teacher error、teacher CRLB 与授权数；
- 用一句话解释：若只靠独立重复把标准误差降低 128 倍，为什么量级上要约 16384 倍信息，而 target/sentinel 或更好的视角可能更值得。

退出标准不是记住数字，而是能说清：**低秩是场表示，transport 是动力学约束，profile rank 是结构问题，CRLB/NIS/coverage 是统计问题；四者不能互相冒名。** 当前 v2 是 post-open corrective NO-GO，不是新算法。下一实验是 q-amplitude + 500-noise bootstrap 和迭代 variable projection；真实训练仍等待师兄的 callable、JVP/VJP、timestamp、covariance、anchor 与基线合同。

---

## Day 7：读懂 LGWO-A24 只改首方向的窄假设

**建议时间：** 2 小时 40 分钟。**GPU：不需要。绝对不要启动 fit。**

### 学习目标

能写出当前方法的输入、方向约束、固定 CGLS shell、fallback 和调用预算；知道哪些是结构事实，哪些仍是科学假设。

### 20--40 分钟直觉

当前部署思路是：

```math
g=A^Ty,
\qquad
\|\delta_\theta\|_2\le 0.05\|g\|_2,
\qquad
d_0=(g+\delta_\theta)\odot support.
```

网络只读部署可观测量，提出一个小修正 `delta`。求解器沿 `d0` 做精确一维最小二乘，后续方向重新由 residual adjoint 产生。proposal 不得暗中调用 `A/A^T`；越出 fit-only calibration envelope 时，必须精确回退到 `delta=0`。

范数界可推出

```math
g^Td_0\ge(1-\eta)\|g\|_2^2,
\qquad
\cos(g,d_0)\ge\frac{1-\eta}{1+\eta}.
```

当 `eta=0.05`，保守下界约为 `0.90476`。这只保证方向不完全背离 `g`，不保证 field error 改善、真实物理安全或泛化。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '1,230p' demo_t16_operator/lgwo_a24_anchored_cdls.py
sed -n '1,220p' demo_t16_operator/lgwo_a24_learnable_direction.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_lgwo_a24_anchored_cdls.py::test_zero_correction_recovers_zero_start_cgls_and_exact_call_budget \
  demo_t16_operator/test_lgwo_a24_anchored_cdls.py::test_bounded_proposal_is_clipped_and_residual_is_monotone \
  demo_t16_operator/test_lgwo_a24_anchored_cdls.py::test_calibration_fallback_skips_proposal_and_recovers_baseline
printf '# Day 7：LGWO-A24 方法卡\n\n' > "$OERF_LOG/day07.md"
```

### 手算或口述检查

用 Cauchy--Schwarz 从 `||delta|| <= eta||g||` 推出 `g^T(g+delta) >= (1-eta)||g||^2`。然后说出这个不等式没有保证的至少四件事。

### 产物

`$OERF_LOG/day07.md`，含 10 行以内方法伪代码、允许输入/禁止输入表、结构保证/未保证表。

### 退出标准

能准确说出：“LGWO-A24 当前是 BOST 专用、observable-only、geometry-conditioned、norm-bounded 的 first-direction 待检验策略，不是已成功的新算法。”

### 卡住后的降级路线

把网络暂时替换成人工向量 `delta`。先看清 `g -> g+delta -> line search -> fixed CGLS`，第二遍再看模型如何生成 `delta`。

---

## Day 8：守住可部署输入，阻断 truth 与 B 泄漏

**建议时间：** 2 小时 40 分钟。**GPU：不需要。**

### 学习目标

理解 A 几何用于部署输入、B 几何用于训练辅助评价的分层；能识别数据泄漏、inverse crime 和错误 split。

### 20--40 分钟直觉

LGWO proposal 当前只允许读取：

```text
A noisy observation y
A-derived pooled adjoint g=A^T y
A geometry features
support
```

它不能读取 truth、clean observation、phantom family、partition、case/seed、B observation、baseline error、exact-null basis。训练 loss 可以读 truth 和 B clean target，但这些对象不能进入 proposal。A/B 共享同一个解析场，却使用不同 geometry/ray bundle；这仍只是 synthetic independent-ray regularizer，不是独立真实物理，也不能叫 self-supervised BOST。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '1,260p' demo_t16_operator/lgwo_a24_l1_fixture.py
sed -n '180,350p' demo_t16_operator/test_lgwo_a24_l1_fixture.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_lgwo_a24_l1_fixture.py::test_ab_replay_keeps_phantom_truth_but_changes_geometry \
  demo_t16_operator/test_lgwo_a24_l1_fixture.py::test_b_continuous_observation_preserves_inverse_crime_barrier \
  demo_t16_operator/test_lgwo_a24_l1_fixture.py::test_a_proposal_contract_excludes_all_evaluation_only_quantities \
  demo_t16_operator/test_lgwo_a24_l1_fixture.py::test_proposal_builder_rejects_complete_case_object
printf '# Day 8：输入权限与泄漏审计\n\n' > "$OERF_LOG/day08.md"
```

### 手算或口述检查

对以下变量逐个分类为 proposal 可读、loss/evaluator 可读或任何阶段禁用：`y_A`、`A^Ty_A`、A pose、truth、family name、B clean target、case seed、exact-null basis、support、baseline field error。

### 产物

`$OERF_LOG/day08.md`，含三栏权限表和两个 inverse-crime 例子。

### 退出标准

看到一个“效果非常好”的模型时，第一反应会检查输入权限、训练/选择/封闭评分 split 和几何种子，而不是先看平均指标。

### 卡住后的降级路线

记住一条部署测试：真实使用时拿不到的量，默认不能进 proposal。若训练阶段必须用它，只能进入 loss/evaluator，并在代码类型和 manifest 上隔离。

---

## Day 9：五个消融臂不是装饰，它们负责拆穿伪创新

**建议时间：** 2 小时 30 分钟。**GPU：不需要。**

### 学习目标

理解 `full`、`fixed_direction`、`g_only`、`no_raw_observation`、`no_geometry` 各自排除什么简单解释；补上论文必须面对的外部强基线。

### 20--40 分钟直觉

如果 full 有改善，至少还需问：

- 是否一个全局固定体素方向就能做到？
- 是否只靠 `g=A^Ty`，不需要 raw observation？
- 是否 raw observation 没贡献？
- 是否 geometry metadata 没贡献？
- 是否只把 `g` 乘了一个标量，而精确线搜索本来就会抵消这种缩放？

内部消融只能解释组件；它们不能替代 DCDM-style、FCG-NO/neural preconditioner、learned warm-start、DeepONet/FNO/iFNO 和 NeRIF/NeDF/TDBOST 类基线。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '1,260p' demo_t16_operator/lgwo_a24_l1_arms.py
sed -n '1,220p' demo_t16_operator/test_lgwo_a24_l1_arms.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_lgwo_a24_l1_arms.py::test_exact_registry_and_parameter_counts \
  demo_t16_operator/test_lgwo_a24_l1_arms.py::test_input_masking_causality_for_ablation_arms \
  demo_t16_operator/test_lgwo_a24_l1_arms.py::test_fixed_direction_is_one_exact_global_interior_parameter \
  demo_t16_operator/test_lgwo_a24_l1_arms.py::test_targeted_channel_pruning_explains_exact_ablation_counts
printf '# Day 9：消融与强基线矩阵\n\n' > "$OERF_LOG/day09.md"
```

### 手算或口述检查

解释为什么 `d0=1.05g` 在精确线搜索下通常与 `d0=g` 给出相同的一维搜索直线。再说明 fixed-direction 与 geometry shuffle 分别检查什么。

### 产物

`$OERF_LOG/day09.md`，含“假设、对应消融、失败时结论、成功后仍需证据”四列表。

### 退出标准

能为 full model 的每个输入通道指出至少一个消融，并能说出三类不能由内部消融替代的外部强基线。

### 卡住后的降级路线

只比较 `full`、`g_only`、`fixed_direction` 三项。先回答“是否真的需要按 case 变化”和“是否真的需要 observation/geometry”，再扩到五臂。

---

## Day 10：先冻结评价与成本，再讨论谁更好

**建议时间：** 2 小时 50 分钟。**GPU：不需要。绝对不要运行 `run_lgwo_a24_l1_fit.py` 的 scientific fit。**

### 学习目标

理解 field relative-L2 主终点、H1/梯度误差、A/B residual、逐 rig 尾部、harm/fallback 以及 `A/A^T` 调用和训练反传成本必须同时报告。

### 20--40 分钟直觉

measurement residual 小不等于 field 真值近，因为欠定方向可能几乎不改变 detector。平均 gain 也可能掩盖某个 rig 的严重退化。因此未来表格至少要含：

- field relative-L2 作为唯一主终点；
- H1/gradient error；
- A measured/clean residual 与 B consistency；
- 每 rig median、p90/p95、worst、harm rate、fallback rate；
- `A/A^T`、B evaluator、backward-equivalent、wall time、memory、参数量；
- 非有限、breakdown 和失败案例。

同样是“24 次”，batched API 调用和 case-equivalent 工作量也必须分开。

### PSU-C1 判读练习：为什么 NO-GO 反而节省了训练

先打开 [PSU-C1 结果解释](lgwo_a24_psu_c1_simple_controls_no_go_2026-07-20.md)，再用
[正则化与停止策略文献地图](psu_c1_regularization_stopping_literature_map_2026-07-20.md) 的 A1--A3 补齐半收敛与
hybrid projection，最后看
`demo_t16_operator/results/lgwo_a24_psu_simple_controls_visualization_v1/psu_c1_scale_separated.png`。不查答案，口述：

1. truth-only oracle 在两个主分区只有约 1.13%/1.25%，没过 5% 门，究竟关闭了什么？
2. linear observable 接近 oracle，但没过 2% 和 fixed-margin 门，为什么仍不能扩成神经网络？
3. inverse-Sobolev 的 field 更好、held-out B 更好，但 active measured residual 更差，谁“赢”了？
4. 为什么结果只关闭 bounded first-direction，不关闭 learned regularization、stopping 或全部算子学习？

合格回答必须同时出现 `post-open`、`analytic truth`、`proxy`、`same 24F/24A^T` 和 `no breakthrough`。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '90,250p' docs/lgwo_a24_l1_fit_runner_contract_2026-07-20.md
sed -n '250,430p' docs/lgwo_a24_l1_fit_runner_contract_2026-07-20.md
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_lgwo_a24_l1_losses.py::test_manual_per_case_terms_and_exact_hinge_formula \
  demo_t16_operator/test_lgwo_a24_l1_losses.py::test_h1_uses_physical_spacing_and_only_two_active_support_endpoints \
  demo_t16_operator/test_lgwo_a24_l1_operator_ledger.py::test_complete_epoch_seals_batched_and_case_equivalent_costs \
  demo_t16_operator/test_lgwo_a24_l1_operator_ledger.py::test_missing_call_hard_fails_seal
printf '# Day 10：指标与成本注册表\n\n' > "$OERF_LOG/day10.md"
```

### 手算或口述检查

一个 epoch 有 8 个 geometry clusters，每个 batch 同时处理 3 个 family。根据当前合同，核对：

```text
batched API:     192 A-F + 192 A-A^T + 8 B-F
case-equivalent: 576 A-F + 576 A-A^T + 24 B-F
planned optimizer steps: 8 (example only; not executed)
```

解释为什么只报网络 forward 时间会对经典/学习方法不公平。

### 产物

`$OERF_LOG/day10.md`，含主终点、次终点、成本、统计单位、失败门五栏的预注册草表。

### 退出标准

面对任何“比 FNO 好”的说法，能先问：同数据 split 吗、同算子预算吗、同墙钟/训练成本吗、报未见 rig 尾部吗、是否选最好 seed？

### 卡住后的降级路线

只保留四项：field relative-L2、worst-rig harm、`A/A^T` 调用、wall time。先学会同时看质量、风险和成本。

---

## Day 11：N5-D5 的 53 次请求到底在防什么

**建议时间：** 2 小时 50 分钟。**GPU：不需要。**

### 学习目标

理解 curved、straight、direct residual 三路径，JVP/VJP、all-h 中心差分、branch state 与调用账本；能复跑 synthetic protocol 并守住其证据边界。

### 20--40 分钟直觉

对三条路径定义：

```math
F_c(x),\quad F_s(x),\quad R(x).
```

若 direct residual 真正在同一离散语义下形成，应检查：

```math
R(x)=F_c(x)-F_s(x),
```

```math
J_Rv=J_cv-J_sv,
\qquad
J_R^Tq=J_c^Tq-J_s^Tq.
```

另外用真实 `F(x+hv)` 与 `F(x-hv)` 检查 JVP，并用 `q^TJv=v^TJ^Tq` 检查伴随。协议固定消费 `h=1e-5,1e-4,1e-3`，不能事后只挑最好看的步长。

53 次请求由 `2 describe + 42 forward + 6 JVP + 3 VJP` 构成。它能检查接口语义、导数一致性和成本账本，不能证明真实 BOST 物理正确。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '1,240p' docs/n5_d5_minimum_real_interface_bridge_2026-07-19.md
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_n5_d5_synthetic_reference_adapter.py \
  site_tools/test_run_n5_d5_minimum_interface_bridge.py \
  site_tools/test_validate_n5_d5_minimum_interface_bridge.py
.venv/bin/python site_tools/run_n5_d5_minimum_interface_bridge.py \
  --config demo_t16_operator/configs/n5_d5_minimum_interface_bridge_preregistered_v1.json \
  --output /tmp/n5_d5_caretaker_replay
.venv/bin/python site_tools/validate_n5_d5_minimum_interface_bridge.py \
  /tmp/n5_d5_caretaker_replay
printf '# Day 11：N5-D5 53 请求图\n\n' > "$OERF_LOG/day11.md"
```

### 手算或口述检查

1. 为什么 JVP/VJP 互为伴随仍可能来自同一个错误 Jacobian？
2. 为什么 direct residual 在 wrapper 最后做 `curved - straight` 不能冒充 native residual primitive？
3. diagnostic state 改变为什么不一定代表 forward 控制流 branch 改变？
4. 为什么独立 validator 不应导入 runner 与 adapter？

### 产物

`$OERF_LOG/day11.md`，含 53 请求树、三路径三条恒等式、五类假阳性及其防线。

### 退出标准

能准确读出 `SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION`，并立即补充“无真实物理授权、无三维重建结果、无算法突破”。

### 卡住后的降级路线

先只画一条 path 的 `base forward + 2 tangents x 3 h x +/- + JVP + VJP`。理解后再复制到三条 path，并加两个 describe。

---

## Day 12：只向师兄索取最小决定信息

**建议时间：** 2 小时。**GPU：不需要。**

### 学习目标

把“我想做算子学习”变成师兄可以回答的接口、物理痛点、基线、指标和隐私问题；发出一次短而准确的沟通，不要求师兄先整理整套数据。

### 20--40 分钟直觉

网络结构不是第一决策。真实 forward 的路径语义、导数能力、分支、标定、主要痛点和评价基线，才决定论文应走 residual-native、branch-aware、有限视角 inverse 还是求解加速。

### 动手任务

```bash
cd "$OERF_REPO"
sed -n '1,260p' docs/n5_d5_advisor_first_contact_2026-07-19.md
cp docs/n5_d5_advisor_first_contact_2026-07-19.md \
  "$OERF_LOG/day12_给师兄的问题草稿.md"
```

把草稿压缩成一次微信消息，先问最关键的 9 项：

1. 真实 forward 的仓库、版本和入口函数是什么？
2. 第一版接 field，还是 decoder parameters 到 field 再到 detector？
3. curved/straight 是否都存在？是否共享 sample/interpolation/aperture？
4. residual 是 ray/sample/integrand 层 native 形成，还是两张 detector map 最后相减？
5. 能否对运行时任意 `x/v/q` 调用 forward、JVP、VJP？
6. 是否有 hard mask、termination、occupancy pruning 或 adaptive sampling？
7. 最小匿名合法例子、真实 dtype/units/axis order 和运行环境是什么？
8. 主痛点与主指标是什么？必须比较哪些组内基线？
9. 哪些源码、几何、标定、权重、trace 和图绝不能公开？

### 手算或口述检查

做一次 3 分钟模拟汇报：前 60 秒讲 BOST/CGLS；中间 60 秒讲 LGWO-A24 窄假设；最后 60 秒只提接口请求和停止边界。不能说“已经优于”“已经泛化”或“突破”。

### 产物

`$OERF_LOG/day12_给师兄的问题草稿.md`，以及实际发送内容和发送时间。若尚未发送，明确标记 `NOT_SENT`，不要写成已沟通。

### 退出标准

师兄即使只回复“有/没有 forward、JVP、VJP、native residual”和“当前主痛点”，你也能进入 Day 13 的一个明确分支。

### 卡住后的降级路线

只发送四问：forward 入口、JVP/VJP 有无、native residual 有无、最希望解决的主痛点。其余问题放到第二次沟通。

---

## Day 13：根据真实接口能力分流，不假装所有路都已打开

**建议时间：** 2--3 小时。**GPU：不需要；即使师兄给了 GPU 环境，也先不租卡、不训练。**

### 学习目标

把师兄回复转换为唯一的下一项小实验。若还没有回复，也有一条不越界的等待路线。

### 20--40 分钟直觉

先按能力分支，再按物理问题选算法：

| 分支 | 已有能力 | 第一项实验 | 暂时禁止 |
| --- | --- | --- | --- |
| A | forward + JVP + VJP + native residual | 4--16 rays 三路径 FD/adjoint/structure 小证书 | 直接训练 FNO/LGWO |
| B | forward + JVP/VJP，但无 native residual | honest curved/straight 双路径，各自做 FD/adjoint；把 residual-native 记为缺失 | wrapper 末端相减冒充第三路 |
| C | forward + VJP，无 JVP | 用中心 FD 给少量 `v` 建参考，再比 autodiff VJP 的 bilinear identity | 声称完整 Jacobian 正确 |
| D | 只有 forward | 先做 determinism、precision、`h` 梯和 branch map，估算 FD 成本 | 神经算子训练、梯度法结论 |
| E | 暂无 callable 或尚未回复 | 完成公开 PSU rehearsal 与问题清单 | 替实验室臆造接口/数据 |

### 动手任务

所有分支先执行：

```bash
cd "$OERF_REPO"
mkdir -p "$HOME/Desktop/OERF_真实接口_私有"
cp data_templates/n5_d5_lab_interface.placeholder.json \
  "$HOME/Desktop/OERF_真实接口_私有/interface_contract.json"
printf '# Day 13：真实接口分支判决\n\n' > "$OERF_LOG/day13.md"
```

然后只做自己分支的任务：

#### 分支 A：有完整导数与 native residual

在 `day13.md` 填入三条 callable 的真实入口、输入输出 shape、units、dtype、branch 来源和允许的最小 ray 数。先让师兄审核 wrapper 设计；raw trace 默认只写入私有目录。复跑 synthetic N5-D5 作为 transport 参照，但**不要在未经审核的真实程序上自行扩大调用**。

#### 分支 B：有导数但没有 native residual

把 descriptor 限定为 curved/straight 两条真实路径，明确 `direct_residual = unavailable`。分别预注册两条 path 的 base repeat、all-h FD、JVP/VJP dot test 和成本账本；把 `F_c-F_s` 仅作为 evaluator 派生量，不能伪装成第三个 callable。

#### 分支 C：只有 VJP

为两个固定 seeded tangents 预注册 `h=1e-5,1e-4,1e-3` 的中心差分；用 FD 近似 `Jv`，再检查 `q^T FD(v)` 与 `v^T VJP(q)`。若 branch 改变或信号落在数值地板，判为 unresolved，不挑 best `h`。

#### 分支 D：只有 forward

只登记同输入重复输出、float32/float64 能力、三个 `h` 的差分响应、branch/termination 记录和每次 forward 成本。先回答“导数是否可检验、成本是否承受”，不设计需反向传播的模型。

#### 分支 E：尚无接口

执行公开 rehearsal：

```bash
cd "$OERF_REPO"
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py \
  demo_t16_operator/test_interface_baselines.py
```

把仍不确定的问题按“物理语义、几何/标定、导数、成本、隐私”五类整理，不新增 synthetic 科学结论。

### 手算或口述检查

口述自己的分支判决，并回答：当前能检查的是 transport、数值导数、离散伴随、物理正确、重建质量中的哪几层？哪几层仍是 `unknown`？

### 产物

`$OERF_LOG/day13.md` 和桌面私有 `interface_contract.json`。笔记必须出现一个明确标签：`BRANCH_A/B/C/D/E`，不能写“差不多都有”。

### 退出标准

选定唯一分支，并把下一项实验压缩到能在最小 rays/field 上运行、能失败、能保存全部步长和成本的规模。

### 卡住后的降级路线

任何信息不确定都降级到能力更弱的分支。例如不知道 JVP 是否真的对运行时任意 `v` callable，就按“无 JVP”处理；不知道 residual 是否 native，就按分支 B 处理。

---

## Day 14：做一个诚实的复现包，而不是提前写胜利结论

**建议时间：** 3 小时。**GPU：不需要。**

### 学习目标

复跑核心公开合同，整理 14 天证据、错误和接口分支；完成一次 10 分钟口试；作出下一阶段 GO/NO-GO，而不是宣告论文完成。

### 20--40 分钟直觉

可复现准备包应回答：我运行了什么、在哪个 commit、哪些测试通过、哪些层级未验证、下一项真实实验需要什么。它不应把“测试代码工作”写成“算法工作”，也不应把 synthetic protocol 写成真实 BOST。

### 动手任务

```bash
cd "$OERF_REPO"
git rev-parse HEAD > "$OERF_LOG/day14_commit.txt"
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_psu_b0_reconstruction_interface.py \
  demo_t16_operator/test_interface_baselines.py \
  demo_t16_operator/test_lgwo_a24_anchored_cdls.py \
  demo_t16_operator/test_lgwo_a24_simple_controls.py \
  site_tools/test_run_lgwo_a24_psu_simple_controls_rehearsal.py \
  site_tools/test_validate_lgwo_a24_psu_simple_controls.py \
  demo_t16_operator/test_lgwo_a24_l1_fixture.py \
  demo_t16_operator/test_lgwo_a24_l1_arms.py \
  demo_t16_operator/test_lgwo_a24_l1_losses.py \
  demo_t16_operator/test_lgwo_a24_l1_operator_ledger.py \
  demo_t16_operator/test_n5_d5_synthetic_reference_adapter.py \
  site_tools/test_run_n5_d5_minimum_interface_bridge.py \
  site_tools/test_validate_n5_d5_minimum_interface_bridge.py \
  | tee "$OERF_LOG/day14_core_tests.txt"
printf '# Day 14：复现准备总结\n\n' > "$OERF_LOG/day14.md"
```

在 `day14.md` 填写：

1. 当前 commit；
2. 实际运行命令和测试结果；
3. 五个自己真正理解的概念；
4. 三个仍不理解的问题；
5. 师兄回复与 Day 13 分支；
6. 下一项最小实验的输入、输出、失败门、成本和隐私位置；
7. 明确边界：`no paper result / no real BOST claim / no generalization claim`。

### 手算或口述检查

完成 10 分钟闭卷口试：

1. 2 分钟：BOST 物理链和直线射线近似。
2. 2 分钟：`A/A^T`、JVP/VJP 和伴随恒等式。
3. 2 分钟：CGLS/Krylov 与固定调用预算。
4. 2 分钟：LGWO-A24 的输入、范数界、fallback 和未证明事项。
5. 2 分钟：N5-D5 53 请求与自己的真实接口分支。

### 产物

`day14_commit.txt`、`day14_core_tests.txt`、`day14.md`，以及 Day 1--13 的连续笔记。复现包还应列出 PSU-C1 的
`summary.json`、独立 `validation.json`、固定 `test_family_ood-000` 切片、分尺度论文图和 attempt 0 无效记录。失败输出同样是产物，不能删除后只保留成功截图。

### 退出标准

同时满足以下条件才算完成 14 天：

- 核心测试可复跑，或失败原因被完整记录；
- 10 分钟口试不混淆 adjoint/inverse、synthetic/real、engineering/science；
- 已确定 Day 13 分支，或诚实标为 E；
- 下一项实验有明确输入、指标、成本、隐私和停止条件；
- 没有把 14 天结果写成论文成功。

### 卡住后的降级路线

若整套测试太慢或某个历史依赖失败，按 Day 2、3、4、7、11 的单项测试依次复跑，并在总结中区分“环境失败、代码合同失败、科学实验未运行”。不要为了全绿改测试阈值。

---

## 3. 收到师兄回复后的第二层决策

能力分支只决定“能做什么检查”，主痛点决定“应该研究什么”。两者要同时满足。

| 师兄确认的真实主痛点 | 建议第一研究问题 | 最小对照 | 还不能做的主张 |
| --- | --- | --- | --- |
| curved/straight 相消或导数噪声 | residual-native paired primitive 是否改善 all-h FD、adjoint 与重建稳定性 | separate-path subtraction、不同 precision | 不能先说更精确或更物理 |
| hard mask/termination 导致导数跳变 | branch map、局部稳定半径、边界项或可审计 reparameterization | 原始 autodiff、FD、固定分支局部模型 | 不能把任意平滑叫正确导数 |
| 有限视角重建误差 | 固定 CGLS/PCGLS/TV/NeRIF 基线上测试 bounded first-direction | zero-start、warm-start、DCDM-style、FCG-NO、direct inverse | 不能只在一个 rig 报平均 gain |
| 同精度下调用太多/太慢 | 同质量 Pareto：field error 对 `A/A^T`、ray/sample、wall、memory | CGLS/PCGLS、组内当前 solver | 不能只报网络推理时间 |
| 连续高速 4D 数据 | 先冻结 session/rig split，再测试时空低秩或算子策略 | frame-wise 3D、TDBOST/组内基线 | 无真实时序合同前不启动 |

### 有 JVP/VJP 时

优先完成导数与路径语义的小证书，再讨论优化器。只有 callable 能接受运行时任意 `x/v/q`、branch 可记录、units/axis order/dtype 固定，导数检查才有可解释性。

### 没有 JVP/VJP 时

先把问题降级为 forward-only 数值实验：重复性、precision floor、中心差分 `h` 梯、branch map 和调用成本。若每个 FD 方向需要两次昂贵 forward，应先算清预算；没有可信梯度合同，不开需要反向传播的训练。

### 只有 VJP 时

这是常见 autodiff 能力。用少量预注册 `v` 的中心 FD 近似 JVP，再做 bilinear identity。它只能检查这些方向，不能证明完整 Jacobian。

### 没有 native residual 时

保留 honest curved/straight 双路径。可以在 evaluator 中计算两者差，但不能把 wrapper 最后一行减法写成 residual-native 创新。只有真实 sample/integrand 层共同计算的 primitive 才有资格研究相消与共享采样机制。

## 4. 什么时候绝对不租卡、不开训

出现以下任一情况，答案都是“不租 GPU、不启动模型训练”：

1. 还没有师兄确认真实主痛点和必须比较的组内基线。
2. 真实 callable、shape、units、axis order、geometry/calibration 版本不清楚。
3. 只有预计算数组，没有能接受运行时任意输入的 callable。
4. forward 重复性、branch、precision floor 或 `A/A^T`/JVP/VJP 合同未过。
5. 数据 split 仍按 frame 随机切分，可能把同一 rig/session 泄漏到训练和测试。
6. proposal 可能读取 truth、B、family、seed、case ID 或 baseline error。
7. 评价只定义了平均 measurement loss，没有 field 主终点、逐 rig 尾部和 harm gate。
8. DeepONet/FNO/DCDM/FCG-NO/NeRIF 等公平基线及成本口径尚未冻结。
9. 当前 LGWO-A24 scientific case 仍为 0、optimizer step 仍为 0，fit 授权与 epoch 事务 P0 未关闭。
10. 想租卡的理由只是“机器闲着也是闲着”或“先跑起来再想问题”。

只有在数据合同、模型输入、baseline、split、指标、预算、停止条件和保存格式全部冻结后，才评估是否需要 GPU。小网格、单元测试、N5-D5 最小接口和多数数值梯度检查优先用 CPU；租卡不是研究进度的替代品。

## 5. 术语小抄

| 术语 | 初学者解释 | 最容易犯的错 |
| --- | --- | --- |
| BOS/BOST | 从背景图案位移推断折射引起的密度/折射率信息；BOST 强调多视角层析 | 把位移图当成三维场本身 |
| refractive index field | 空间折射率 `n(x)` | 忽略它与密度/温度关系的适用条件 |
| straight ray | 固定直线路径近似 | 折射强时仍当作真实 ray |
| curved ray | 路径受折射率场影响 | 不报告 termination/branch |
| forward operator `A/F` | field 到 detector observation | 把实现通过测试当物理正确 |
| inverse problem | observation 反推 field | 默认存在唯一稳定逆 |
| matrix-free | 不显式存大矩阵，通过 callable 算 `Ax/A^Tq` | 以为完全没有内存/计算成本 |
| adjoint `A^T` | 满足内积恒等式的离散转置作用 | 当成 `A^{-1}` |
| Jacobian `J` | 非线性 forward 在当前点的局部线性化 | 把一个点的 J 当全局模型 |
| JVP | `Jv`，输入扰动对输出的方向导数 | 只测一个低信号方向 |
| VJP | `J^Tq`，输出权重反传到输入/参数 | dot test 通过就宣称物理正确 |
| finite difference | 用 `F(x+hv)-F(x-hv)` 检查方向导数 | 只挑最好 `h`，忽略分支变化 |
| CGLS | 对最小二乘问题使用 CG 思路的 matrix-free 方法 | 不核算每步 `A/A^T` |
| Krylov subspace | 由 `A^Ty` 和反复 `A^TA` 作用生成的搜索空间 | 把“神经 Krylov”当无人研究 |
| preconditioner | 改善问题尺度/谱，使迭代更快 | 只比迭代数，不比每步成本 |
| support | 允许非零的物理/几何区域 | 用 truth 生成 support 造成泄漏 |
| gauge | 为不可辨识自由度选定参考，例如边界置零 | 把 gauge 选择说成物理测量 |
| nullspace | 满足 `A delta=0` 的方向 | 用 `A delta` 小冒充 exact null |
| near-null | 对观测影响很小但不严格为零 | 声称严格数据一致性 |
| regularization | 用平滑、稀疏、边缘或先验稳定反演 | 不报告偏差和超参选择 |
| operator learning | 学函数到函数/离散无关映射的一类方法 | 把任何 CNN 都叫新算子 |
| warm start | 给经典求解器一个更好的初值 | 与 first-direction 混为一谈 |
| first-direction policy | 只建议第一条搜索方向 | 声称等于完整预条件器 |
| heldout B | 同一 synthetic truth 下独立 ray geometry 的辅助/评价对象 | 叫成独立真实物理或自监督 |
| inverse crime | 生成数据与反演使用完全相同离散模型，结果过于乐观 | 用 toy 高分外推真实实验 |
| rig | 一套相机/光学几何配置 | 随机按 frame 切分导致 rig 泄漏 |
| tail metric | p90/p95/worst/harm 等尾部风险 | 只看 mean gain |
| fallback | 条件不满足时精确退回基线 | 把经验 envelope 称为概率安全 |
| call ledger | 记录 `A/A^T`、B evaluator 和等价反传成本 | 隐藏网络训练和内部重算 |
| native residual | residual 在共同 sample/integrand primitive 中原生形成 | wrapper 最后两张图相减冒充 native |

## 6. 给师兄的问题清单：按优先级发，不要一次轰炸

### 第一优先级：决定能否接线

1. 真实 forward 入口、版本、环境和最小合法 batch/rays 是什么？
2. 输入是 voxel field、隐式场参数，还是 decoder parameters？
3. curved、straight、direct residual 哪些是真实独立 callable？
4. 是否支持运行时任意 `x/v/q` 的 JVP/VJP？
5. 最希望优先解决哪个真实痛点，主指标和组内当前 baseline 是什么？

### 第二优先级：决定导数实验是否可信

6. sampling、interpolation、aperture weights、boundary、termination 是否共享？
7. hard mask、occupancy pruning、adaptive sampling、ray termination 如何记录？
8. dtype/mixed precision、units、axis order、wavelength 和 gradient-check noise floor 是什么？
9. 能否记录 `A/A^T`、ray/sample evaluations、wall time、memory、失败与重试？

### 第三优先级：决定论文评价与公开边界

10. 可提供哪些 rig/session/phantom/真实火焰 split？是否有独立相机验证？
11. 必须比较哪些组内代码、公开方法和指标？
12. 哪些源码、权重、几何、标定、raw trace 和图绝不能出组？哪些匿名汇总可写毕设？

## 7. 科学红线

1. **Synthetic 不等于 real。** PSU toy、JACRU fixture 和 N5-D5 synthetic protocol 只能证明对应离散/接口合同。
2. **测试通过不等于算法有效。** 单元测试检查实现义务，不产生 field gain 或泛化证据。
3. **伴随通过不等于物理正确。** 同一个错误 Jacobian 也可能自洽。
4. **不挑 best seed、best `h` 或 best rig。** 预注册的 seed、全部 `h` 和失败轨迹都要保留。
5. **不泄漏。** truth、clean target、B、family、partition、case/seed、exact-null evaluator 不得进入 proposal。
6. **不隐藏成本。** 同时报 `A/A^T`、B evaluator、反传等价成本、wall、memory 和训练成本。
7. **不拿弱基线垫高自己。** 经典 CGLS/PCGLS/TV、warm-start、DCDM-style、FCG-NO、direct inverse 和组内方法要公平配置。
8. **不把 `A delta` 小叫 exact nullspace。** 只能写 low-observability tendency，除非有严格投影/证明。
9. **不把经验 fallback 叫 provably safe。** 当前 min/max envelope 没有覆盖率定理。
10. **不把 B consistency 叫 self-supervised real BOST。** 当前 B 仍由同一 synthetic truth/renderer 产生。
11. **不把 wrapper subtraction 叫 native residual。** 路径层级必须由真实实现与师兄确认。
12. **不公开私有材料。** 实验室源码、VPN 获取内容、受限论文、几何、标定、权重和 raw trace 默认留在私有位置。
13. **不把工程基础设施当论文贡献。** 日志、哈希、授权和 fail-closed 很重要，但不能替代科学机制与结果。
14. **允许失败。** 若真实接口显示 LGWO-A24 没有 signal，应诚实停止或换到已证实的真实物理困境；不能通过换 seed、改 split 或删失败样本制造成功。

## 8. 第 14 天之后真正的下一步

14 天完成后，只能在以下条件同时满足时进入“小规模 scientific pilot 设计”：

```text
真实主痛点已由师兄确认
+ 最小 callable 与隐私边界明确
+ forward/derivative 能力分支明确
+ baseline、split、主终点、尾部门和成本口径冻结
+ 小实验能 fail closed
+ raw 数据与公开摘要分离
```

下一阶段仍应从最小规模开始：4--16 rays 的接口证书、`12^3` 或同等级小网格、1--2 个固定方向、一个明确失败门。只有该层通过，才讨论完整训练数据、GPU 预算与论文实验矩阵。

**最终提醒：** 这份路线的成功标准不是“14 天后有一篇论文”，而是你能独立判断下一项实验是否真实、是否公平、是否值得花算力。高质量论文来自真实问题、清楚合同、强对照、失败可见和可复现实验的积累，不能由一轮 toy 训练替代。
