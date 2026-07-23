# JACRU M2 本科生学习日志：今天真正学到了什么

日期：2026-07-17  
对象：物理本科生  
范围：M0、M0.1、M1、M2-T0、M2.1、M2.2，以及尚未产生实验判决的 M2.3  
当前总状态：**M0/M1/M2.1 均为 NO-GO；M2.2 只找到 oracle headroom；没有 real BOST 成功。**

## 0. 先用一句话说清楚

今天不是把一个新网络“做成功了”，而是把问题逐层缩小了：

1. M0 证明“让一个复杂模型从近乎空白状态包办整个反演”不可靠；
2. M1 留下“经典求解器打底，学习模型只修残差”的微弱路线信号；
3. M2-T0 证明网络能学到合成三维形态，但会严重违背相机观测；
4. M2.1 证明普通有限步 Landweber 来不及把这种冲突清掉；
5. M2.2 用不可部署的 dense SVD oracle 证明：在当前小型离散算子里，“保留场收益”和“保持内部重投影一致”在数学上可以同时做到；
6. 所以 M2.3 才有一个值得实现的问题：能否不用 dense SVD，只靠有限次 `A` 和 `A^T` 调用近似完成同样的 row-space removal。

这是一条从失败中得到的研究路线，不是一条已经通过验证的算法结论。

## 1. 证据标签不能混用

本文沿用总日志的 L0-L3 口径：

| 等级 | 本文含义 | 今天有没有 |
|---|---|---|
| L0 | 真实实验或论文级证据 | **没有** |
| L1 | 首开前冻结、一次性 held-out synthetic 门禁 | 今天的 M2.2 **不是** |
| L2 | 已看过数据后的 synthetic 诊断，只能产生下一步假设 | M0.1、M1、M2.1、M2.2 的主要结论属于这里 |
| L3 | 代码、哈希、调用次数、数值恒等式、测试等实现证书 | 有，但它只证明“算得可审计”，不证明方法有效 |

机器摘要里还使用了 `E1_*` 这套内部标签。例如 M2.2 写的是
`E1_OPENED_T0_DENSE_NULLSPACE_HEADROOM_ORACLE_NO_FRESH`。这里的 `E1` 不能被误读成上表的 L1；
真正关键的限定词是 `OPENED`、`ORACLE` 和 `NO_FRESH`。

### 四个容易混在一起的词

- **synthetic：**观测和三维真值由程序生成。当前观测端使用解析 phantom/梯度 renderer，逆解端使用独立的体素有限差分与三线性插值算子。两条代码链分开能减少最窄的 inverse crime，但它仍不是相机拍到的实验数据。
- **opened：**我们已经看过这个 split 的指标，并据此修改了方法或提出了下一步。opened 数据适合诊断，不再适合充当一次性确认考卷。
- **oracle：**为了回答“理论上有没有空间”，使用真值、标签或现实中不可承受的计算工具得到上界。M2.2 的投影本身不读 field truth，但它依赖 dense matrix 和完整 SVD，而且收益只能靠 synthetic truth 评分，所以仍是 oracle headroom。
- **real BOST：**真实光源、背景、流场、相机、标定、噪声、有限孔径和图像处理共同形成的测量。真实 BOST 还可能没有独立三维 field truth，因此必须依赖 held-out camera、独立 renderer、flow-off 噪声和外部传感器等证据，不能照搬 synthetic 的 field-L2 结论。

因此，今天的正信号应完整写成：

> **在 opened synthetic 12^3 toy 上，dense exact-nullspace oracle 找到 headroom。**

不能缩写成“JACRU 成功”“BOST 重建成功”或“真实数据可用”。

## 2. 今天实际做了什么

### 2.1 先把 M0 的漂亮界面分数撤回

初始化审计发现，固定 `x` 方向的初始平面在读取任何观测之前，就对两个单界面样本得到
`F1@1dx = 1.000`；最终 JACRU 反而降到 `0.973958`。平滑场的初始假阳性率还是 `100%`。

这说明 M0 曾经出现的 `79.68%` ASSD 改善和 F1 提升被 data-free geometry alignment 混淆，
不能记在算法头上。今天做对的一件事不是保住好看的数，而是把它作废。

### 2.2 用 M0.1 和 M1 区分“实现错误”与“路线错误”

M0.1 修复尺度、随机化平面并关闭初始 gate 后，field relative-L2 从 `1.987836` 降到
`0.769049`。这证明修复确实有用，但它仍比 Huber-PDHG 的 `0.480119` 差 `60.18%`。

M1 改成 `CGLS-18 + JACRU-6`，field relative-L2 降到 `0.495027`：只比 CGLS-24 的
`0.498911` 好 `0.78%`，仍比 Huber-PDHG 差 `3.11%`，H1 还差 `15.74%`。因此 M1 没赢，
但它支持继续问“学习模块是否只该做小残差修正”。

### 2.3 把 M2 变成真正的跨样本学习问题

M2-T0 不再为每个测试样本重新拟合一套参数。每个样本先做 CGLS，再把逐相机残差的伴随 lift、
pose、mask、support 和底座场交给共享网络。四个小模型使用同一批 `32` 个 train、`12` 个
development、`18` 个 exploratory-OOD case，各跑 `3` 个模型 seed：

- JACRU-M2；
- pooled 3D CNN；
- fixed-grid DeepONet；
- pooled FNO。

JACRU-M2 的 field-L2 相对 CGLS 在 development / exploratory-OOD 改善
`46.16% / 32.38%`，但逐 case 重投影误差放大 `28.56x / 35.10x`。更小的 pooled CNN 场指标还
略好。这轮的结论是“监督形态先验很强，也很会幻觉”，不是 JACRU 结构优越。

### 2.4 作废 M2.1 的不公平首版，再做同预算比较

M2.1 第一版让 learned 路径多跑了 11 步，却仍与 CGLS-13 比较。这相当于一边多做 11 道演算，
一边宣称自己更聪明，所以首版只保留为流程错误记录。

修订版把最终比较统一到 `24F/24A`。JACRU 加 11 步 measured pullback 后，field 仍比同预算最强
经典场基线改善 `45.34% / 35.68%`；但重投影相对 CGLS-24 仍差
`43.12x / 41.95x`。零个候选点通过完整 gate。

### 2.5 用 M2.2 问“目标是否存在”，暂时不问“算法是否可用”

M2.2 在 12^3 toy 上把每个几何的 active operator 组装成 `150 x 1000` dense matrix，并用
float64 SVD 做精确 row/null decomposition。12 个几何的 numerical rank 都是 `150`，因此
numerical nullity lower bound 为 `850`。

JACRU 的 exact-null 解相对 CGLS-24 得到：

| 指标 | Development | Exploratory OOD |
|---|---:|---:|
| field gain | `45.28%` | `37.54%` |
| H1 gain | `43.75%` | `40.19%` |
| reprojection / CGLS-24 | 约 `1.000000000000012` | 约 `1.000000000000012` |
| null correction norm / total | `0.913` | `0.903` |

pooled CNN 也得到 `44.24% / 37.38%` 的 field gain，仍与 JACRU 很接近。于是唯一新增授权是
`continue_matrix_free_projection_research=true`；部署、方法胜出、real BOST、fresh/final 的授权全是
`false`。

## 3. M0、M1、M2.1 为什么失败

### 3.1 M0：任务太多，初始化还偷看似地对上了答案形状

M0 想同时从很少的观测中优化三维场、界面位置、跳跃强度和 camera bias。这个问题本来就欠定，
参数之间还会互相补偿：场错一点可以让 bias 顶上，界面错一点可以让 amplitude 顶上。

更严重的是，固定初始平面恰好对齐了生成器里的单界面方向。模型还没读数据就已经拿到高 F1，
所以界面分数不能说明模型从观测中学到了界面。最终 field、H1 和重投影也都明显输给经典基线。

**本科生版结论：**不是“训练轮数不够”，而是考题设计、初始化和模型职责一起出了问题。

### 3.2 M0.1：修好了明显 bug，但方法本身仍不够好

M0.1 把 field 误差从约 `1.99` 降到 `0.77`，说明尺度和初始化确实是大问题。可是经典 Huber-PDHG
已经做到约 `0.48`。修 bug 后仍相差 60%，就不能继续把失败解释成“小实现问题”。

**本科生版结论：**调通代码只让方法获得公平考试资格，不会自动让方法及格。

### 3.3 M1：职责分配变合理了，但 0.78% 不是方法胜出

CGLS 先解决大部分可观测成分后，JACRU 只修最后一小段，结果几乎追平 CGLS。这说明“物理底座
+ 小残差”的方向比“网络包办全部”合理。

但 M1 没有赢最强经典场基线，H1 也更差，而且只有四个 opened development case。`0.78%` 只够
支持下一轮机制实验，不够支持性能主张。

**本科生版结论：**M1 是路线提示，不是成功结果。

### 3.4 M2-T0：网络答得像标准答案，却没有回答这一次观测

网络看过许多 synthetic truth 后，学会了 smooth field、single interface 等“常见长相”。在欠定
问题中，它可以把结果推向这些常见形状，所以 field-L2 很漂亮。

但把结果重新送入 forward operator 后，它预测的相机位移与输入观测严重不符。这就是
truth-space 指标改善、measurement-space 指标崩坏同时发生的原因。

**本科生版结论：**三维图像看着像，不代表它能解释相机拍到的东西。

### 3.5 M2.1：普通 Landweber 方向没错，但在固定预算内太慢

M2.1 的 near-null filter 为

```text
delta_(j+1) = delta_j - tau A^T A delta_j
```

它会较快压低大奇异值方向，却很慢地处理小奇异值方向。11 步后，JACRU 的
`||A delta_k|| / ||y-Ax0||` 仍为 `2.282 / 3.189`，离预定的 `<=0.10` 很远。

因此 M2.1 不是“一点点没过门”，而是有限步谱过滤还没有接近精确零空间投影。继续把 11 改成
21、51、101，也会增加调用预算；经典 CGLS/Huber 必须同步增加预算，不能靠多算取得伪优势。

**本科生版结论：**方向是下降观测误差，但工具和预算不匹配。

## 4. M2.2 为什么是积极信号，但不是成功

### 4.1 积极在哪里

M2.1 失败后有两种可能：

1. 网络里的有用场信息和观测冲突完全纠缠，任何数据一致性投影都会把收益一起删掉；
2. 有用信息主要在当前算子的 numerical null space，普通 Landweber 只是来不及把 row-space 冲突删干净。

M2.2 支持第二种解释。精确删除 row-space 分量后，内部重投影回到 CGLS-24，同时保留了大部分
field/H1 收益。这说明 M2.3 有一个明确的线性代数目标，不必再盲目扩大网络。

### 4.2 为什么不能叫成功

1. **dense SVD 不可部署。**真实三维 BOST 的 rays 和 voxels 可能达到百万级，不能显式组装 `A` 再完整分解。
2. **只约束 approximate `A`。**`ker(A_voxel)` 不等于连续有限孔径光学 forward 的真零空间。
3. **数据已经 opened。**M2.2 由 M2-T0 和 M2.1 的结果启发，只能作为 post-open mechanism diagnosis。
4. **仍依赖 synthetic truth 评分。**没有真值就无法知道保留下来的零空间分量是正确结构还是训练集模板。
5. **没有 fresh。**没有首开前冻结的新 synthetic test，更没有真实实验确认。
6. **JACRU 没有独占信号。**pooled CNN 几乎得到相同结果，JACRU 的集合编码优势尚未建立。
7. **没有 runtime 胜出。**SVD setup 故意不进入方法预算，因此这轮不能报告加速或效率优势。

850 维 numerical null space 既提供了放置有用先验的空间，也提供了藏入幻觉的空间。M2.2 只证明
“存在一个数学上兼顾两项内部指标的解”，没有证明“我们能廉价找到它”，更没有证明“它是真实流场”。

## 5. M2.3 每个公式的直观含义

### 5.1 先认清符号

| 符号 | 直观含义 |
|---|---|
| `y` | 相机测到的位移数据，measurement space 中的向量 |
| `x_ref` | 同预算经典方法给出的可靠底座，例如 CGLS |
| `x_net` | 网络给出的候选三维场 |
| `delta = x_net - x_ref` | 网络想在经典底座上增加的修正 |
| `A` | 把三维场投到相机位移的 forward operator |
| `A^T` | 把 measurement-space 残差回传到三维体素的 adjoint，不等于真正的逆 |
| `P_row` | 取出修正中会改变当前算子观测的部分 |
| `lambda` | 稳定线性系统的小正则，不是随意调漂亮结果的旋钮 |
| `z` | measurement space 中用于解释并消除可见冲突的辅助变量 |

### 5.2 exact oracle 公式：我们想近似什么

```text
P_row delta = A^T (A A^T)^dagger A delta
```

分四步看：

1. `A delta`：问“网络修正会让相机预测改变多少”；
2. `(A A^T)^dagger`：在 measurement space 中处理不同测量方向的相关与尺度；
3. `A^T`：把这份可见变化送回体素空间；
4. 得到 `P_row delta`：修正中能被当前观测看见、也可能破坏观测一致性的部分。

`dagger` 是 Moore-Penrose pseudoinverse。M2.2 用 SVD 精确算它；M2.3 不能这样做，只把它当目标。

```text
delta_null = delta - P_row delta
```

直观上是“从网络建议里减掉相机能看见的冲突，只留下当前离散算子看不见的部分”。理想情况下
`A delta_null = 0`。

```text
x_oracle = x_ref + delta_null
```

底座负责解释观测，网络只在不改变当前观测的方向上补充先验。这里的 `oracle` 表示精确投影上界，
不是部署模型。

### 5.3 M2.3 第一步：先把冲突送到 measurement space

```text
b = A delta
```

`b` 不是新的相机数据，而是“如果接受网络修正，预测观测会被改动多少”。若 `b` 接近零，说明
修正已经近似处于 null space；若很大，说明网络建议和观测明显冲突。

### 5.4 M2.3 第二步：解一个 measurement-space 线性系统

```text
(A A^T + lambda I) z = b
```

可以把它理解为：寻找一份 measurement-space 权重 `z`，使它经过 `A^T` 回到体素空间后，正好
解释网络修正中可见的那一部分。

- `A A^T` 描述不同测量方向之间怎样通过三维场耦合；
- `lambda I` 防止近奇异方向把噪声和数值误差放大；
- `lambda` 太大时会删不干净 row-space，太小时系统可能难解或放大噪声；
- M2.3 必须在 opened development 上冻结选择规则，不能看 fresh 后再挑 `lambda`。

### 5.5 为什么它可以 matrix-free

```text
(A A^T) v = A(A^T v)
```

给一个 measurement vector `v`，先用 `A^T` 回到三维空间，再用 `A` 投回相机空间，就完成一次
`A A^T` 乘法。整个过程只需要调用算子，不需要把 `A` 写成巨大矩阵。

这正是 CG/PCG 可以工作的原因：它只需要“给我一个向量，我能算出矩阵乘向量”，不要求看到
矩阵全部元素。

### 5.6 M2.3 最后一步：删掉可见冲突

```text
x_hat = x_ref + delta - A^T z
```

`x_ref + delta` 是网络候选，`A^T z` 是估计出的 row-space 冲突。减掉后，希望同时满足：

```text
A x_hat approximately equals A x_ref
```

以及

```text
x_hat keeps useful field information from x_net
```

第一句可以在没有 field truth 时检查；第二句在 synthetic 中可以用 truth 检查，在 real BOST 中通常
不能直接检查，所以还需要 held-out camera、独立 renderer 或外部传感器。

### 5.7 PCG 和预条件器在做什么

把 `G = A A^T + lambda I`，PCG 实际上反复缩小残差：

```text
r_k = b - G z_k
```

`r_k` 是“当前 `z_k` 还有多少 measurement conflict 没解释”。预条件器 `M^-1` 把不同尺度的方向
先拉到比较接近的难度：

```text
q_k = M^-1 r_k
```

- unpreconditioned CG：`M = I`，最便宜的基线；
- Jacobi：用 `G` 的对角近似尺度校正；
- fixed low-rank：用少量主方向近似难解部分；
- geometry-conditioned preconditioner：根据相机几何预测预条件结构。只有前三种在固定预算下明显不够，才有理由学习这一项。

### 5.8 调用预算为什么必须看实际 counter

按最直接的写法：

1. `b = A delta` 用一次 forward；
2. 每个 PCG iteration 的 `A(A^T v)` 用一次 adjoint 和一次 forward；
3. 循环结束后显式计算 `A^T z`，再用一次 adjoint。

这样 `k` 步是 `(k+1)F/(k+1)A`，也是 M2.2 文档中便于审计的保守记法。

截至本日志完成时，工作区已有的 M2.3 接口草稿采用另一种等价递推：每一步算 `A^T p_k` 时，同时累计
`A^T z_k`，于是不用在循环结束后额外回投。该实现的核心路径把 `k` 步记为 `(k+1)F/kA`。
这目前只是 L3 实现信号，没有 M2.3 结果摘要或性能判决；预条件器对角、低秩基或其他 setup
若调用了物理算子，还必须在 runner 中另行入账。

所以公平比较不能只背某一个公式，必须读取实际 counter，并给 CGLS、Huber、base-only CG
相同的总调用预算。

## 6. 现在需要补的知识

### 6.1 线性代数：先补这些，不必先学完整本教材

1. 向量空间、子空间、线性无关、基与维数；
2. rank-nullity：`rank(A) + nullity(A) = n`；
3. row space、null space 及其正交关系；
4. 正交投影，为什么投两次仍等于投一次；
5. SVD：`A = U Sigma V^T`，奇异值大小与可辨识性；
6. pseudoinverse，为什么秩亏矩阵没有普通逆仍可求最小范数解；
7. 对称正定/半正定矩阵，为什么 CG 要求这类结构；
8. condition number，为什么小奇异值让 Landweber 和 CG 变慢；
9. regularization，`lambda` 如何在稳定与偏差之间取舍；
10. preconditioning，它改变求解难度，不应改变目标答案的定义。

最低掌握标准：能用一个 `2 x 3` 矩阵手算 null space，能解释 `A^T` 为什么不是 `A^-1`，能说出
M2.2 的 `150 x 1000, rank=150` 为什么至少留下 850 维 numerical null space。

### 6.2 逆问题：要分清“拟合观测”和“接近真值”

1. forward problem：给定三维场，预测相机观测；
2. inverse problem：从有限、带噪观测反推三维场；
3. ill-posed / underdetermined：不同三维场可能产生近似相同观测；
4. data consistency：`Ax` 是否接近 `y`；
5. field accuracy：`x` 是否接近 truth；真实实验里 truth 常常不存在；
6. regularization/prior：在多个可行解中偏好平滑、稀疏或符合形态先验的解；
7. inverse crime：生成数据与求逆完全使用同一离散模型，会高估方法；
8. independent renderer：减少最窄 inverse crime，但仍不等于真实光学；
9. opened、sealed、fresh、exploratory OOD 的实验角色；
10. matched budget：方法比较必须统一 `F/A` 调用、停止规则和数据访问。

最低掌握标准：看到“field gain 45%，reprojection 43 倍更差”时，能立刻判断为 NO-GO，而不是只
汇报前半句。

### 6.3 PyTorch：为读懂 M2.3 只需先掌握这些

1. tensor 的 shape、dtype、device，以及 batch/view/channel/3D spatial 维；
2. `nn.Module.forward`、参数注册、`state_dict`；
3. 最后一层 zero initialization 与“训练前精确回退到底座”；
4. autograd、`.backward()`、`.grad`，以及 `torch.no_grad()`/`.detach()` 的边界；
5. `float32` 训练与 `float64` 数值审计的区别；
6. 用 Python callable 表示 matrix-free `A` 和 `A^T`；
7. dot-product adjoint test：`<Ax,y>` 与 `<x,A^T y>` 应接近；
8. 固定 seed、保存 config/hash、逐 case metric rows；
9. 测试 zero-init、support mask、相机置换不变性和调用计数；
10. 不让 truth、family label、interface mask 偷渡进 deployment API。

最低掌握标准：能读懂 `test_zero_initialization_is_exactly_the_cgls_base` 和
`test_reconstruction_decomposition_and_orthogonality_hold` 在保护什么科学合同。

## 7. 七天可执行练习

每天建议 `60-90` 分钟。先写答案，再运行代码；不要只看输出。

### Day 1：手算 row space 和 null space

取

```text
A = [[1, 0, 1],
     [0, 1, 1]]
```

任务：

1. 手算 `rank(A)` 和 `ker(A)` 的一组基；
2. 找一个 `delta`，把它拆成 row-space 与 null-space 两部分；
3. 验证 `A delta_null = 0`；
4. 用一句话解释“null-space 修正不改变观测”。

完成标准：不查答案也能解释 rank-nullity。

### Day 2：用 SVD 重做一个迷你 M2.2

任务：

1. 阅读 `demo_t16_operator/test_jacru_m2_exact_nullspace_oracle.py` 的第一个测试；
2. 用 `torch.linalg.svd` 对 Day 1 的矩阵做分解；
3. 数值检查 `delta = delta_row + delta_null`、两者正交、`A delta_null` 接近零；
4. 把 SVD tolerance 改大，观察 numerical rank 为什么会变化。

可运行：

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  demo_t16_operator/test_jacru_m2_exact_nullspace_oracle.py -q
```

完成标准：能解释“numerical null space”为什么依赖容差。

### Day 3：理解 forward、adjoint 和重投影

任务：

1. 画出 `x -> A -> y` 与 `r -> A^T -> voxel lift` 两条箭头；
2. 手写 `<Ax,r> = <x,A^T r>`，逐项说明左右两边的空间；
3. 阅读 `demo_t16_operator/jacru_m2_data_consistency.py`；
4. 运行其合同测试，找出步长上界和调用计数检查。

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  demo_t16_operator/test_jacru_m2_data_consistency.py -q
```

完成标准：不再把 adjoint 叫作 inverse。

### Day 4：比较 Landweber 与 exact projection

任务：

1. 在 Day 1 小矩阵上从同一个 `delta` 出发，跑 `1/3/5/20` 步 Landweber；
2. 每一步记录 `||A delta_k||`；
3. 与 SVD exact-null 的一次投影比较；
4. 改变奇异值比例，观察病态程度如何拖慢有限步过滤。

完成标准：能用奇异值而不是“训练不够”解释 M2.1。

### Day 5：读懂 learned residual 的 PyTorch 合同

任务：

1. 阅读 `demo_t16_operator/test_jacru_m2_learned_residual.py`；
2. 找到 zero-init、support-limited、bounded residual、view permutation 和 no-truth API 五类测试；
3. 画出一个 batch 的关键 tensor shape；
4. 回答：为什么最后一层全零时仍必须检查学习梯度不为零？

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  demo_t16_operator/test_jacru_m2_learned_residual.py -q
```

完成标准：能区分“初始输出为零”和“网络永远学不动”。

### Day 6：自己写一个 20 行左右的 matrix-free CG

任务：

1. 只把 `G(v)` 写成函数，不显式保存 `G`；
2. 解一个小型 SPD 系统 `Gz=b`；
3. 每步记录 `||r_k||`；
4. 加入 Jacobi preconditioner，对比达到同一残差所需步数；
5. 单独计数 `A` 与 `A^T` 调用，分别验证“显式末次回投”和“递推累计回投”的 `k` 步预算；
6. 把预条件器 setup 的调用另列一行，不能藏在免费预处理里。

完成标准：能解释 M2.3 为什么可以 matrix-free，以及预条件器到底省了什么。

### Day 7：做一次不夸大的结果汇报

任务：用一页纸填写下表，不允许只选最好看的列：

| 方法 | F/A | field gain | H1 gain | reproj ratio | harm | worst | 数据状态 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| CGLS |  |  |  |  |  |  | opened/fresh | baseline |
| JACRU + projector |  |  |  |  |  |  | opened/fresh |  |
| CNN + projector |  |  |  |  |  |  | opened/fresh |  |

然后口头回答三问：

1. 哪个结论只在 synthetic 成立？
2. 哪个步骤使用了 oracle？
3. 如果换成 real BOST 且没有三维 truth，还能检查什么、不能检查什么？

完成标准：能在 90 秒内准确说出“信号、失败门、下一步”，不说“整体成功”。

## 8. M2.3 出现不同结果时怎样决策

### 情况 A：实现合同先失败

表现：adjoint dot-product test 不过、`lambda < 0`、CG 残差 NaN、support 外非零、调用数对不上，
或 exact toy 对照无法复现。

决策：**立即停在工程调试。**这时没有科学 NO-GO，更没有算法结论；先修实现和测试。

### 情况 B：普通 CG/Jacobi 在固定预算内无法压低重投影

表现：M2.3 reprojection 仍高于 matched CGLS 的 `1.10x / 1.15x`。

决策：先确认 exact SVD headroom、`lambda` 规则和调用账本，再测试固定 low-rank preconditioner。
只有这些非学习方法确实够不到 oracle，才允许研究 geometry-conditioned preconditioner。仍不打开 fresh。

### 情况 C：重投影过门，但保留不到 50% oracle field gain

表现：物理一致性恢复了，但有用 field 信息也被删掉。

决策：画 `k/lambda` 的 field-reprojection Pareto 图，并与 base-only CG 对照。若冻结范围内始终保留
不到 50%，就停止当前 learned residual + projector 路线，不靠加网络容量救场。

### 情况 D：保留了 50%，但没有赢最强同预算经典场基线

硬门是相对最强 matched classical field 至少改善 `5% / 2%`（development / exploratory OOD）。

决策：判为**机制可运行但没有方法收益**。可以留下实现资产，不进入 superiority 主张。

### 情况 E：JACRU 与 pooled CNN 仍接近

决策：删除“JACRU 集合编码器更优”的主张，优先保留更简单、参数更少的 CNN 作为主学习基线。
只有 camera-count、pose、mask OOD 上出现预先定义且多 seed 稳定的差异，才重新讨论 JACRU 结构价值。

### 情况 F：approximate `A` 通过，独立 renderer 失败

表现：内部 measured reprojection 很好，独立解析 renderer 的 clean reprojection 仍恶化。

决策：判为 **operator mismatch NO-GO**。先改 forward/geometry/finite-aperture 合同，不能继续让网络
补偿错误物理。

### 情况 G：opened development 的所有门都通过

决策：只写“opened synthetic mechanism signal”。随后冻结代码、配置、seed、`lambda/k`、预条件器、
阈值、哈希和唯一候选，再建立未看的 synthetic split。不能把 opened pass 改名为 fresh pass。

### 情况 H：一次性 fresh synthetic 也通过

决策：最多写“在预注册内部 synthetic fixture 上通过一次性门禁”。下一步才是 real BOST held-out
image consistency、camera/session transfer、flow-off noise 和外部传感器。仍不能声称真实三维场准确。

### 情况 I：real BOST held-out reprojection 通过，但没有独立三维 truth

决策：可以声称“对未参与重建的真实图像/相机具有更好的观测一致性”，前提是协议冻结且基线公平；
不能声称 field-L2、shock 位置或折射率三维真值恢复正确。需要 PIV、压力、schlieren、CFD 或其他
独立证据才能继续加强物理解释。

### 情况 J：real BOST 失败或 synthetic-to-real 特征明显越界

决策：保留 synthetic 算法与负结果，回到噪声、标定、finite aperture、geometry 和 renderer mismatch；
不在真实 opened 数据上反复调阈值直到过门。

## 9. M2.3 的最低实验清单

在任何“下一轮成功”表述之前，至少需要同时看到：

- [ ] matrix-free `A A^T` 与 dense toy 结果一致；
- [ ] adjoint、support、dtype、determinism 和调用计数测试通过，preconditioner setup 也入账；
- [ ] unpreconditioned CG、Jacobi、fixed low-rank 都有同预算结果；
- [ ] base-only measurement-space CG 对照存在；
- [ ] JACRU 与 pooled CNN 接受完全相同的 projector；
- [ ] `k`、`lambda`、停止规则和 preconditioner 选择只用 opened development 冻结；
- [ ] exact SVD 只作 oracle 上界，不进入 runtime 排名；
- [ ] field、H1、reprojection、harm、worst case 和三模型 seed 同时报出；
- [ ] independent renderer clean reprojection 单独过门；
- [ ] opened 通过后才建立并一次性打开新的 fresh synthetic；
- [ ] real BOST 结果与 synthetic 结果分表，不用同一个“成功”标签。

## 10. 今天可以写进周报的话

> 在已打开的 12^3 synthetic T0 上，M0/M1 和有限步 M2.1 均未通过方法门。M2-T0 的监督残差虽
> 明显改善 field/H1，却严重破坏重投影；同预算 M2.1 仍比 CGLS-24 差约 42 倍。M2.2 的 dense
> SVD exact-null oracle 证明，在当前 approximate inverse operator 中，数据一致性与大部分场收益
> 可以共存，因此授权继续研究有限调用的 matrix-free measurement-space projector。该结果属于
> opened synthetic oracle headroom，不是部署算法、fresh confirmation 或 real BOST 成功。

## 11. 直接证据入口

- [M0-M1 负证据判决](jacru_m0_m1_negative_evidence_2026-07-17.md)
- [M2-T0 监督残差 NO-GO](jacru_m2_t0_supervised_residual_no_go_2026-07-17.md)
- [M2.1 匹配预算 NO-GO](jacru_m2_1_matched_data_consistency_no_go_2026-07-17.md)
- [M2.2 exact-nullspace headroom](jacru_m2_2_exact_nullspace_headroom_2026-07-17.md)
- [M2 严格预注册草案](jacru_m2_preregistered_gate_2026-07-17.md)
- [M2.2 机器摘要](../demo_t16_operator/results/jacru_m2_2_exact_nullspace_oracle_postopen_public/summary.json)
- [M2.2 oracle 合同测试](../demo_t16_operator/test_jacru_m2_exact_nullspace_oracle.py)
- [M2.3 matrix-free 接口草稿（仅 L3，无实验判决）](../demo_t16_operator/jacru_m2_matrix_free_projection.py)

最后提醒自己：**负结果不是“白做”，但只有把失败门、数据状态和下一步停止规则写全，负结果才会
真正变成研究资产。**

## 12. 实际跑完 M2.3–M2.8 后，路线怎样改变

前面的学习任务是在结果出来前写的。现在六轮 opened synthetic 试验已经完成，结论比原计划更具体：

1. M2.3 的 reference target 错在终点，不是 CG 代码写错。
2. M2.4 的 affine target 数学上可达，但 identity CG 太慢。
3. M2.5 exact Jacobi 失败，因此不做 Hutchinson diagonal。
4. M2.6 exact camera-block 很快，但伤害一个 single-interface case。
5. M2.7 在 K=9、23F 已过平均 residual 门，证明瓶颈不再是速度。
6. M2.8 连 truth-aware alpha ceiling 都失败，证明简单插值/校准不能救。

最值得记住的一句话是：**更快满足含噪的 `Ax=y`，不等于更准确地恢复真实三维场。**

下一周的学习重点从“再学一种预条件器”改为：

- 为什么 inverse problem 要用 discrepancy principle，而不是把 noisy residual 压到零；
- covariance whitening、Huber data fidelity 和 Tikhonov/proximal center 各解决什么；
- 如何永久留出 audit camera，让风险回退只读真实可观测证据；
- 为什么 truth-oracle ceiling 只能用于证伪，不能成为算法输入。

完整数字、停止规则和向师兄确认的问题见
[M2.3–M2.8 打开数据判决](jacru_m2_3_to_m2_8_opened_evidence_2026-07-17.md)。
