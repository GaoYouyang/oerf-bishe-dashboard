# D0：从 signed A 到 exact-|A| 诊断

> 面向初学者的学习笔记（2026-07-17）。本文解释当前 PSU-B0 / BOST 诊断协议中的数学对象、正式结果和证据边界；结果已经过独立校验器 61 项检查。

## 0. 先记住一句话

在 BOST 层析里，真正描述“体场改变会怎样改变观测”的前向算子是**带符号的** `A`。本次诊断把 `A` 的绝对值用于设计对角步长和预条件，但求解递推仍然使用 signed `A` 及其伴随。也就是说：

```text
signed A：负责物理前向、残差和梯度
|A| 或 M：只负责构造安全的对角度量
```

把这两件事混在一起，会误以为“换了一个更好的物理模型”，这是不准确的。

## 1. signed forward operator A 是什么

把待重建的体场离散成向量 `x`，把所有相机/视角的观测排成向量 `b`。在一个固定几何、固定采样规则和固定线性化点下，BOST 的离散前向关系可以写成

\[
    b \approx A x, \qquad A\in\mathbb{R}^{m\times n}.
\]

`A[i,j]` 表示第 `j` 个体素坐标对第 `i` 个观测行的线性影响。它可以为正、为负，也可以为零。负号不是“错误”：它可能来自梯度方向、投影方向、有限孔径采样中的导数符号，或不同物理贡献的抵消。

本仓库的 A-only 诊断固定在真实 PSU 探测器几何上，网格为 `16^3`，9 个视角，每视角 256 条射线、有限孔径采样数为 8；展开后的测量行数是 `4608`，支持内坐标数是 `2744`。这些是协议的运行形状，不是对一般 BOST 问题的定理。

求解器的基本残差是

\[
    r(x)=Ax-b,
\]

并通过 `A^T r` 更新体场。因此，在当前诊断中，“signed A recurrence”指的是：每次数据前向和数据伴随都回到同一个 signed `A`。

## 2. `|A|` 与 factor majorizer `M` 不一样

### 2.1 elementwise `|A|`

逐元素绝对值定义为

\[
    |A|_{ij}=|A_{ij}|.
\]

它保留了 `A` 的形状和每个耦合位置，只是去掉符号。它可以回答：如果不允许正负抵消，一个体素坐标在某一行上有多少“绝对影响”？

注意：`|A|` 不是用于求解残差的算子。一般来说，`|A|x` 不等于 `|Ax|`，因为后者仍然包含正负抵消。

### 2.2 factor majorizer `M`

`M` 是现有 factor pipeline 对 `|A|` 的一个逐元素上界：

\[
    M_{ij}\ge |A_{ij}| \quad\text{对所有 }i,j.
\]

它来自可分解的、便于流式计算的绝对测量因子，而不是重新定义一个 signed 物理前向模型。由于 `M` 可能比 `|A|` 大，特别是在多个采样因子被拆开、再通过三角不等式组合时，`M` 会留下“安全但可能松”的余量。这个余量会影响由行和、列和构造的对角步长。

当前实现不把完整稠密矩阵存入生产路径，而是按基础向量分批流式生成 signed 列和 absolute/factor 列。v2 amendment 在每一批 signed/absolute forward 后、finite 检查和 CPU 归约前显式执行 `torch.mps.synchronize()`；但重复 MPS basis replay 仍不稳定，因此 v3 把 exact `A/M`、Schur 证书和 power estimate 移到 CPU float64，solver 轨迹仍用 MPS float32。两次修订都发生在性能数据打开前，不改 `A`、`M`、阈值、样本、检查点或 solver math。

## 3. M-active 数量不是 nullspace 维数

若某个坐标 `j` 的 `M` 列和大于零，就把它称为 M-active；若列和为零，就从这个**计算支持**中删除。v2 的预期形状是：2744 个支持内体素中，2322 个 M-active，422 个 M-zero。

这只能说明：在 majorizer 的绝对耦合图中，这 422 个坐标没有被当前 `M` 连接到数据行。它不能说明 signed `A` 的零空间维数。

线性代数中的 nullspace 是

\[
    \ker(A)=\{v:Av=0\},
\]

其维数是 `n-rank(A)`。要知道它，至少要研究 signed `A` 的秩、奇异值或等价的可辨识性问题；只数零列、零行或 `M`-active 坐标都不够。一个坐标可能在 `M` 中是零，但仍与 signed A 的零空间、边界约束或其他坐标耦合有关；反过来，一个 M-active 坐标也可能参与某个非零向量的全局抵消。

因此协议明确写出：`factor_active_coordinates_are_nullspace_dimension = false`，并把 signed operator nullspace dimension 标为 unknown。

## 4. 四种对角度量到底比较什么

令 `\eta=0.7`。每种度量都会产生 primal 对角量 `\tau_j` 和 data-dual 对角量 `\sigma_i`，然后把它们放入同一个 signed-A PDHG 递推。下表中的“列”对应体素坐标，“行”对应观测。

### 4.1 formal factor-view

这是正式 factor 方案：

\[
  \tau_j=\frac{\eta}{\sum_i M_{ij}},
  \qquad
  \sigma_i=\frac{\eta}{\max_{i\in v}\sum_j M_{ij}},
\]

其中 `v` 表示同一个视角的行块，所以一个视角内共享同一个 dual 步长。它同时包含两个选择：用了可能偏松的 `M`，并且把行信息压缩成了每视角最大行和。

### 4.2 factor-row hybrid

它保留 `M` 的列和，因此 primal 步长不变；但把 per-view 最大值换成逐行的 `M` 行和：

\[
  \tau_j=\frac{\eta}{\sum_i M_{ij}},
  \qquad
  \sigma_i=\frac{\eta}{\sum_j M_{ij}}.
\]

所以它主要隔离“视角内最大值聚合”是否造成损失，而不是隔离 factorization 本身。

### 4.3 exact-abs-view

这里把 `M` 的列和换成 exact `|A|` 的列和，同时保留按视角的 exact 行最大值：

\[
  \tau_j=\frac{\eta}{\sum_i |A_{ij}|},
  \qquad
  \sigma_i=\frac{\eta}{\max_{i\in v}\sum_j |A_{ij}|}.
\]

相对 formal factor-view，它主要测试 factor majorizer 的松弛/抵消是否重要；视角级行聚合仍然存在。

### 4.4 exact-abs-row

这是最紧的静态对角诊断：

\[
  \tau_j=\frac{\eta}{\sum_i |A_{ij}|},
  \qquad
  \sigma_i=\frac{\eta}{\sum_j |A_{ij}|}.
\]

它同时去掉 `M` 的松弛和 per-view 最大值。若它有改善，而前两个“单独替换”没有达到协议规定的 material threshold，只能描述为“联合静态对角因素可能重要”；不能直接说已经证明了某个因果机制。

exact-|A| 度量要求所有冻结的 M-active 坐标在 exact `|A|` 中也非零。若出现 M-active/A-zero 坐标，exact metric 按协议无定义，运行应停止而不是偷偷改变支持。

## 5. Schur 安全证书与 power-iteration 估计

把归一化算子写成

\[
    B=\Sigma^{1/2} A T^{1/2},
\]

其中 `T=diag(\tau)`、`\Sigma=diag(\sigma)`。PDHG 的稳定性需要控制 `\|B\|_2^2`。

### 5.1 Schur row/column safety certificate

先验证两个前提：

1. 逐元素 `M >= |A|`；
2. 对 exact row/column sums，有
   `sigma_i * sum_j |A_ij| <= eta`，以及
   `tau_j * sum_i |A_ij| <= eta`。

Schur 检验给出一个定理支持的上界：

\[
    \|\Sigma^{1/2}AT^{1/2}\|_2^2\le \eta^2=0.49.
\]

它是安全证书，不是对真实谱范数的精确测量；只要前提通过，即使没有把整个稠密 `A` 存下来，也可以使用这个 bound。

### 5.2 power iteration

代码还用固定随机种子和 24 次迭代估计同一个量 `\|B\|_2^2`。这是数值 stress estimate：它通常会接近某个主特征值，但有限迭代、初始向量和 MPS 浮点误差都可能影响它。它**不是上界**，所以报告中明确写 `power_value_is_upper_bound = false`。

最简单的记忆法是：Schur 证书回答“理论上不会超过哪里”，power iteration 回答“从这个起点迭代看起来有多大”。前者用于安全门，后者用于诊断和 sanity check；二者不能互换。

## 6. 为什么 full-support graph-PCGLS comparator 不具有约束力

graph-PCGLS 使用 full-support Sobolev 表示和自己的图方向，而 factor/exact-|A| PDHG 使用的是已经冻结的 reduced support：只在 M-active 数据耦合坐标上更新。两者的变量空间、先验/正则化方向和支持合同不相同。

因此 graph-PCGLS 可以作为“还有多少 headroom”的参考，但不能作为本次同算子机制判定的 binding comparator。若把一个更强、但问题定义也不同的 full-support 方法放进主排名，就无法回答“只改 `M`/`|A|` 行列聚合后，signed A 的同一递推发生了什么”。当前协议因此明确：graph comparison binding 为 false，graph 不进入 primary same-operator decision。

## 7. same-operator normalized residual 能证明什么

在相同 signed `A`、相同 target、相同初始化、相同 geometry、相同样本和相同 checkpoint `K` 下，可以记录类似

\[
    \frac{\|Ax_K-b\|_2}{\|b\|_2}
\]

的 normalized residual，并比较四种对角度量的轨迹。这种对照能支持：

- 某个对角步长设计在当前冻结问题上，是否让数据残差下降得更快或更稳定；
- factor-view 到 factor-row 的差异是否提示 per-view max 聚合的影响；
- formal factor-view 到 exact-abs-view 的差异是否提示 `M` 相对 `|A|` 的松弛影响；
- exact-abs-row 是否显示两种静态替换的联合效果。

它不能单独证明：

- 重建场的真实误差一定更小，或视觉质量一定更好；
- 已经找到了 `ker(A)` 或证明了某个 nullspace 维数；
- 这是因果的 Krylov 机制证明，而不只是固定协议下的描述性结果；
- 对新几何、真实流场、不同噪声或未见形态具有泛化性；
- 某个候选算法已经优于 graph-PCGLS 或可以重新打开 Gate B。

v1/v2 amendment 都发生在性能数据打开之前：父运行在第一个样本的 exact-|A| basis streaming 审计阶段失败，未产生可用的 trajectory、tightness、power 或排名行。v3 使用 CPU float64 完成 exact 审计，并保持 MPS float32 solver 轨迹；正式运行与独立校验已经完成。

## 8. 正式结果：残差加速不等于三维场已经解决

冻结端点 K=128 相对 formal factor-view 的 normalized residual 平均改善为：

- factor-row hybrid：`+2.011%`，达到至少 10% 改善的配对场 `0/16`；
- exact-abs-view：`+64.121%`，达到至少 10% 改善的配对场 `16/16`；
- exact-abs-row：`+64.183%`，达到至少 10% 改善的配对场 `16/16`。

高分位 factor slack 的中位数为 `0.972832`。它来自 p05 tail tightness，不是平均 factor 误差，更不能读成“97% 更新无效”。exact-abs-view 与 exact-abs-row 几乎相同，而 factor-row 只有约 2% 改善，因此当前最窄、最可靠的解释是：在这个冻结的 same-signed-A 合成问题中，主要静态对角差异来自 factor majorizer 与 exact `|A|` 的松弛，而不是 per-view max 行聚合。它不证明唯一 cancellation 因果机制。

但是 field relative-L2 没有得到 64% 的同幅改善：formal factor-view 在 K=128 为 `0.959944`，exact-abs-row 为 `0.913594`，按均值之比只改善 4.83%。而且 exact-abs-row 在 K=64 达到更好的 `0.911423`，到 K=128 反而略微恶化，尽管 normalized residual 仍继续下降到 `0.221037`；gradient error 也继续恶化。这是典型的 semi-convergence 信号，说明噪声、不可辨识性、先验和停止规则已经成为下一层瓶颈。

因此结果状态是 `FACTOR_MAJORIZER_CANCELLATION_MATERIAL_DESCRIPTIVE`，不是新算法胜出。Gate B 仍然关闭，graph-PCGLS 仍只是 full-support nonbinding headroom，真实/泛化结论仍不存在。

还要记住两个统计边界：16 行来自两个 replicate 集群内重复的八种 morphology，不是 16 个 IID 实验；synthetic view scaling 又来自 clean-truth projection RMS，所以完整 pipeline 不是 truth-blind。D0 不能据此报告显著性或真实部署性能。

## 9. 对未来 geometry-conditioned neural/operator learning 的启发

这个诊断把“学习什么”拆得更清楚了。未来算法可以让网络读取几何和可观测的算子特征，例如视角方向、射线覆盖、有限孔径采样、局部 row/column mass、detector graph 特征，再输出：

1. 一个 geometry-conditioned 的对角度量或预条件器；
2. 一个 support/null correction 的候选，但必须显式标注它不是已知 nullspace；
3. 一个在固定 signed-A 物理层上工作的 operator correction。

更稳妥的路线是：先用 exact row/column sums 和 Schur bound 做 teacher/constraint，再让网络预测可验证的近似；训练和评估都保留 signed forward/adjoint，禁止网络把 `|A|` 当成物理 forward。数据划分应按 geometry 留出，而不只是随机切 field；还应检查 held-out geometry、真实/合成噪声迁移和最坏样本，而不只看平均 normalized residual。

正式结果把模型设计进一步约束为三步：先学习可被 Schur-safe clipping 的 cancellation-aware metric；再补充 local diagonal 无法表达的 global/history correction；最后用不读 truth 的 residual/front proxy 做 uncertainty-aware stopping。三步必须分别消融，不能把一个大网络的最终输出当作机制证明。

## 10. 紧凑阅读清单

- [ ] 能写出 `b≈Ax`，并解释 `A[i,j]` 的正负号。
- [ ] 能区分 `|A|`、majorizer `M` 和 signed `A` 的职责。
- [ ] 能解释为什么 `M-active=2322` 不是 `dim ker(A)`。
- [ ] 能从“列和控制 tau、行和控制 sigma”复述四种 metric。
- [ ] 能说明 Schur bound 是上界，而 power iteration 不是上界。
- [ ] 能说明 full-support graph-PCGLS 为什么只是 nonbinding headroom。
- [ ] 能列出 same-operator normalized residual 的至少两个可证明结论和两个不能证明的结论。
- [ ] 能指出 v2 修的是 MPS 同步边界，v3 修的是精度/设备审计边界，而不是算法或阈值。
- [ ] 能解释为什么 normalized residual 改善 64% 仍不能叫作重建误差改善 64%。
- [ ] 能从 K=64 到 K=128 的 field 恶化说明早停/正则化为何是下一问题。

## 11. 可以问 He Yuanzhe 的问题

1. 在当前 BOST 离散化里，`A` 的负号主要来自哪一层：空间梯度、投影方向、有限孔径采样，还是它们的组合？
2. 对 `M>=|A|`，我们最希望解释的是 factorization 的松弛，还是 per-view max 行聚合的保守性？
3. 对本科毕设，是否应把“support-active”统一叫作 data-coupled coordinate，避免学生误读成 nullspace 维数？
4. 若 exact-abs-row 只改善优化残差但不改善场误差，下一步应优先研究先验、噪声模型，还是可辨识性？
5. geometry-conditioned 模型的第一版应预测 diagonal metric、support correction，还是直接预测 operator kernel？为什么？
6. 未来的 held-out geometry 应如何设计，才能区分“记住视角布局”和真正学习了几何条件下的算子规律？
7. 对当前 K=64 后 residual 继续下降而 field 开始恶化的现象，组内真实数据是否有可用的 stopping proxy 或 flow-off noise estimate？
8. H2 rotation-40 forward mismatch bundle 能否提供相机几何 provenance、作者 forward、mask、单位和 manifest？

## 参考路径（仓库内）

- `demo_t16_operator/configs/psu_b0_exact_absolute_root_cause_v3_cpu64_audit_amendment.json`
- `docs/psu_b0_exact_absolute_cpu64_audit_amendment_2026-07-17.md`
- `docs/psu_b0_exact_absolute_root_cause_result_2026-07-17.md`
- `demo_t16_operator/psu_b0_exact_absolute_diagnostic.py`
- `site_tools/run_psu_b0_exact_absolute_diagnostic.py`
- `site_tools/validate_psu_b0_exact_absolute_diagnostic.py`
- `demo_t16_operator/test_psu_b0_exact_absolute_diagnostic.py`
