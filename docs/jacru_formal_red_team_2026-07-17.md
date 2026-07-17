# JACRU 数学与新颖性正式红队审查

**审查日期：** 2026-07-17
**审查对象：** `Jump-Aware Cone-Ray Unrolling (JACRU)` 当前研究假设、
[`phase_interface_bost.py`](../demo_t16_operator/phase_interface_bost.py) 原型，以及仓库内
OERF / NeRIF / TDBOST 证据链。
**证据范围：** 仓库代码与文档；作者论文页、期刊页、arXiv 和作者代码。本文不是穷尽式
systematic review，不把“本轮没有搜到”写成“文献中不存在”。

## 0. 红队结论

> **当前判决：新颖性主张 RED；数学机制验证 AMBER。**

1. 当前 JACRU 草案中的多数单件已经被覆盖：连续背景与 level set 的交替层析、phase-field
   周长松弛、显式界面神经算子、FNO / shift-DeepONet 的不连续解非线性重构、带精确物理算子的
   展开反演、NeRIF 的神经折射率场、finite-aperture cone-ray BOST、TDBOST 的四维张量表示，
   以及重建后 level-set / Rankine-Hugoniot 压力映射。
2. 当前 `phase_interface_bost.py` **不是 JACRU 实现**。它是
   `background + gate * amplitude * tanh(phi / epsilon)` 的固定步数 Adam 拟合；传入的
   `adjoint` 只做一次初值，代码里没有逐层 forward/VJP、有限孔径相机模型、可变相机集合编码、
   两侧场或分块 learned update。
3. JACRU 只有在一个很窄的联合定义下才**可能**新：面向有限孔径 BOST 的、离散目标一致的
   forward/VJP，在非冗余显式界面状态上进行 smooth/interface/camera-nuisance 分块展开，并处理
   无序、数量可变的相机观测。即便如此，目前也只能称“待证研究假设”，不能称“首次”。
4. 最高优先级不是训练网络，而是选定唯一的离散路径，证明/数值验证 forward、Jacobian 与
   transpose 的一致性。体素梯度和显式 `delta(phi)` 同时加入会双计跳跃；曲光线中冻结路径会
   漏掉路径导数；近切向射线会使锐界面形状导数病态。
5. 在数学门、同预算 generic Learned Primal-Dual 对照、无界面负对照和相机几何 OOD 之前，
   不应写 JACRU 论文的 superiority、general operator 或 physical shock recovery 结论。

因此，本报告建议把题目暂时写成：

> **A falsifiable study of discrete-objective-consistent, interface-aware unrolling for
> finite-aperture BOST**

而不是“一个已经优于 DeepONet/FNO/NeRIF 的新算法”。

---

## 1. 审查口径与证据等级

本文使用四种标签：

- **[P] 一手来源事实：** 论文、期刊页或作者代码明确支持。
- **[C] 代码事实：** 当前仓库可直接检查。
- **[D] 数学推导：** 由写明的假设推得；不是外部论文实验结论。
- **[H] 待证假设：** 可作为实验问题，尚不能作为论文结论。

审查分开回答三个问题：

1. 某个**组件**以前是否出现过；
2. JACRU 的**联合问题定义**是否仍可能构成贡献；
3. 当前代码和证据是否已经实现、验证该贡献。

“组件已有”不自动否定合理的系统贡献；“组合看起来少见”也不自动构成算法创新。

---

## 2. 当前原型到底实现了什么

### 2.1 可由代码确认的模型

当前场表示是 [C]：

\[
q_h = b_h + \sum_{m=1}^{M}
\sigma(\ell_m)a_m\tanh\!\left(\frac{\phi_{m,h}}{\varepsilon}\right),
\qquad M\in\{0,1,2\}.
\]

它不是草案中的

\[
n=(1-H_\varepsilon(\phi))n^-+H_\varepsilon(\phi)n^+,
\]

也没有独立的空间跳跃场、两侧光滑场或相机偏差场。代码事实与后果如下。

| 当前实现 | 红队解释 |
|---|---|
| `amplitudes` 初值为 0 | 第一步数据项对 `phi` 和 gate 的梯度为 0；gate 稀疏项却立即把 gate 往关闭方向推，存在 dead-gate 风险。 |
| gate 初始 logit 为 `-1.5` | 初始概率约 `0.182`，低于部署阈值 `0.5`；soft 优化与 hard 部署可能不连续。 |
| 分量为 `g a tanh(phi/epsilon)` | 两侧极限差是 `2ga`，所以变量 `a` 不是物理跳跃 `[q]`。 |
| `g*a` 共同缩放 | gate 与 amplitude 不可辨识；同时存在界面排列对称和 `(phi,a) -> (-phi,-a)` 对称。 |
| `phase_smoothness` 是二阶差分平方 | 这是曲率型平滑，不含双阱/双障碍势，不能据此声称 phase-field 能量或 Gamma 收敛。较准确名称是 diffuse-interface / regularized level-set prototype。 |
| `forward` 是任意可微 callable | 代码不保证它是 BOST、cone-ray、finite-aperture 或物理正确的 forward。 |
| `adjoint` 只调用一次 | 它只生成 background 初值；没有逐迭代 `J^T r`，因而不是 learned/unrolled primal-dual。 |
| hard gate 仅在返回体场时使用 | hard-gated 结果没有再次 forward 重投影；部署输出可能没有数据一致性证据。 |
| `observation_weight` 先乘残差再平方 | 实际权重是输入值的平方。若输入本意是 precision，应传平方根或重命名。 |
| 对 `background * support` 求差分 | support 边界本身可能被当作背景梯度并受罚，需与物理域边界处理分开。 |

### 2.2 当前可以、不能说什么

当前代码可以称为：**固定 forward-call 预算下的可证伪 diffuse-interface 参数化基线**。

当前代码不能称为：JACRU、phase-field tomography、finite-aperture BOST solver、exact-adjoint
unrolling、operator learner，或已验证的 shock reconstruction method。

---

## 3. 既有方法已经覆盖了哪些组件

### 3.1 Level-set tomography 覆盖

[Kadu、van Leeuwen 与 Batenburg](https://arxiv.org/abs/1704.00568) 已明确研究“连续变化背景 +
由 level set 表示的已知灰度异常”，将背景和 level-set 系数写成双层优化并交替求解 [P]。
因此以下内容不新：

- smooth background 与 level-set anomaly 的联合表示；
- 背景更新与界面更新分开/交替；
- 在层析 forward 与 transpose 下对 level-set 参数做梯度或 Gauss-Newton 更新；
- limited-data phantom 上以 TV / 离散层析方法为对照。

JACRU 若只把这种交替迭代改写成若干层网络，不能仅凭“split update”成立原创性。

### 3.2 Phase-field 覆盖

[Dunbar 与 Elliott](https://arxiv.org/abs/1811.02865) 已把二值 traveltime tomography 写成
data misfit 加 phase-field 周长松弛，使用 double-obstacle potential，并给出离散、下降过程和
Gamma-convergence 论证 [P]。因此以下内容不新：

- diffuse interface 近似 sharp binary region；
- phase-field / perimeter regularization；
- `epsilon` 与网格分辨率的联合参数研究；
- 以 adjoint 导数优化界面状态。

当前原型没有 double-well/double-obstacle potential，也没有相应收敛论证；把二阶平滑项称作
phase-field 会同时产生新颖性和数学准确性问题。

### 3.3 DeepONet、FNO 与显式界面算子覆盖

- [DeepONet](https://arxiv.org/abs/1910.03193) 的基本 branch/trunk 结构已经从固定传感器值到
  查询坐标学习非线性算子 [P]。
- [FNO](https://arxiv.org/abs/2010.08895) 已以 Fourier-space integral kernel 学习函数空间之间
  的映射 [P]。
- [Lanthaler 等](https://arxiv.org/abs/2210.01074) 已指出，带线性 reconstruction 的标准
  DeepONet/PCA-Net 在其研究的不连续 PDE 算子上存在效率下界，而 FNO 和 shift-DeepONet 的
  非线性 reconstruction 可绕开所研究的下界 [P]。该结论不自动外推到 BOST 逆问题。
- [IONet](https://arxiv.org/abs/2308.14537) 已按界面划分子域，使用多 branch/trunk 网络并在
  loss 中处理界面条件 [P]。
- 2026 年预印本 [`phi`-DeepONet](https://arxiv.org/abs/2604.08076) 已把界面 one-hot/非线性
  latent embedding 注入 trunk，并用多 branch 处理不连续输入 [P, preprint]。
- [IANO 官方 AAAI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/39887) 已报告界面信息
  编码和 geometry-aware positional encoding 用于多相流神经算子 [P]。

所以“把 `phi`、界面位置或拓扑送进 DeepONet/FNO”“给 smooth field 一个 FNO proximal”以及
“interface-aware operator”都不能单独作为 JACRU 的新颖性。

### 3.4 Unrolled inverse methods 覆盖

- [Learned Primal-Dual](https://arxiv.org/abs/1707.06474) 已将可能非线性的 forward operator
  放进固定层数 primal-dual 展开，并以 CNN 替代 proximal [P]。
- [MoDL](https://arxiv.org/abs/1712.02862) 已把显式 forward/data-consistency 数值块与共享权重
  learned prior 交替 [P]。
- [Variational Network](https://arxiv.org/abs/1704.00447) 已将迭代变分重建展开为可训练网络 [P]。

因此以下内容不新：固定 `K` 层、每层调用物理 forward/backprojection、learned proximal、
跨层共享权重或显式 data-consistency。JACRU 必须证明**为什么针对 BOST 跳跃几何的状态与更新
分裂产生了独立收益**，而不是只换变量名的 Learned Primal-Dual。

### 3.5 OERF / BOST 邻近工作覆盖

| 一手来源 | 已覆盖内容 | 不能替 JACRU 证明的内容 |
|---|---|---|
| [NeRIF](https://doi.org/10.1063/5.0250899)；[开放全文](https://arxiv.org/html/2409.14722v2) | 连续三维折射率及梯度神经场、沿 ray 随机采样、AD/ND 一致性、两种位移监督；真实火焰以留一投影重投影验证 [P] | 显式 sharp-interface 状态、跨样本 inverse operator、JACRU split unrolling |
| [Finite-aperture BOST](https://arxiv.org/html/2402.15954) | cone-ray / depth-of-field forward 与 neural-implicit inverse；TV 用于含 shock 的 piecewise-smooth 场 [P] | 跳跃分布项的 exact shape adjoint、可变相机 set-unrolling |
| [TDBOST DOI](https://doi.org/10.1145/3809488)；[作者代码](https://github.com/Hyz617/TDBOST) | 面向 4D BOST 的张量分解表示、轻量网络与公开 reconstruction 工程 [P] | JACRU 的 3D 界面分块更新或其优越性 |
| [Optics Letters 2025](https://doi.org/10.1364/OL.571452) | time-resolved multi-angle BOS 后进行 3D 折射率重建、level-set、Gladstone-Dale 与 Rankine-Hugoniot 压力映射，并与传感器对照 [P] | 从任意折射率图自动识别物理激波；JACRU 端到端恢复压力 |

结论：JACRU 不能声称“首个 neural BOST”“首个 finite-aperture inverse”“首个 4D neural
BOST”“首个 level-set shock pressure method”。

---

## 4. JACRU 仅在何种精确定义下可能新

### 4.1 先消除状态冗余

只能选以下一种参数化，不能把三者全部作为自由变量：

**方案 A：两个侧场，跳跃由 trace 导出**

\[
n_\theta=(1-H_\varepsilon(\phi))n^-+H_\varepsilon(\phi)n^+,
\qquad [n]_\Gamma=(n^+-n^-)\vert_{\Gamma}.
\]

**方案 B：一个连续背景加一个跳跃场**

\[
n_\theta=b+H_\varepsilon(\phi)j,
\qquad [n]_\Gamma=j\vert_{\Gamma}.
\]

若同时自由学习 `n-`、`n+` 和第三个 `[n]`，同一体场可以由多组参数表示，数据项无法区分；
“jump branch”会变成不可辨识的重复自由度 [D]。本科第一版推荐方案 B，参数最少、跳跃含义直接。

### 4.2 可检验的 JACRU-v0 定义

令第 `c` 个相机的已知几何为 `G_c`，观测位移、mask/reliability 为 `(y_c,W_c)`，相机 nuisance
参数为 `beta_c`。定义**唯一部署版**离散 forward：

\[
F_{c,h}(\theta,\beta_c)
=\mathcal M_{G_c,h}^{\mathrm{cone}}
\bigl(n_h(\theta),\beta_c\bigr),
\]

以及目标：

\[
\mathcal J_h(\theta,\beta)
=\frac12\sum_c
\left\|W_c\left(F_{c,h}(\theta,\beta_c)-y_c\right)\right\|_2^2
+R_h(\theta,\beta).
\]

只有同时满足以下条件，JACRU 才有一个**可能新但尚未证实**的精确定义 [H]：

1. **问题特定：** 输入是无序、数量可变的 `{(G_c,y_c,W_c)}`，输出是 BOST 折射率/界面，
   不是规则网格 PDE surrogate。
2. **同一 forward：** 训练、每层 data-consistency、验证重投影和部署都调用同一版明确的
   finite-aperture cone-ray forward；近似版必须单独命名。
3. **离散目标一致：** 每一层使用当前 iterate 处 `F_h` 的真实 JVP/VJP；“exact”只指对这份
   离散计算图精确，不指连续光学方程绝对精确。
4. **非冗余状态：** 使用 `(b,j,phi,beta)` 或 `(n-,n+,phi,beta)`，跳跃只定义一次。
5. **有意义的分块：** `K` 层分别更新 smooth field、interface/jump 与 camera nuisance；消融必须
   证明收益不是来自更多参数、更多 forward calls 或普通 learned proximal。
6. **几何置换不变：** set encoder 对相机顺序不敏感，并实际测试 view count / pose OOD；仅在
   固定相机堆叠通道上训练不能称 variable-camera operator。
7. **物理约束有范围：** shock jump 条件只有在已知上游状态、EOS/组分和界面类型时进入循环；
   接触面、火焰面和爆轰不能共用同一 Rankine-Hugoniot penalty。
8. **证据闭环：** 体场、界面、held-out reprojection、几何 OOD、运行成本和不可辨识性都报告。

最保守、可审稿的候选表述是：

> We formulate and evaluate a discrete-objective-consistent split unrolling method for
> finite-aperture BOST with an explicit interface state and variable-camera measurements.

即使这句话也必须等实现、查新和同预算实验后才能写成贡献，当前只能写成研究问题。

### 4.3 更强但风险更高的真正算法方向

比“FNO proximal”更可能形成数学贡献的是：

> 对 cone ray 与移动界面的交点做显式 surface quadrature，推导含交点移动、法向变化、跳跃
> trace 和 aperture 权重的离散 shape derivative/VJP，再把它放入固定层数分块展开。

这条路线与普通体素 autograd 有实质差别，但近切向、拓扑变化和折射路径变化都很难。若它在
解析平面界面上不能比“先组装体场再差分”的简单路径更准或更稳，应立即放弃复杂化。

---

## 5. Forward、adjoint 与跳跃分布项的数学风险

### 5.1 连续恒等式成立不等于离散算法成立

在 `H_epsilon` 可微且 `delta_epsilon=H_epsilon'` 时，

\[
n=n^-+H_\varepsilon(\phi)(n^+-n^-)
\]

形式上给出 [D]：

\[
\nabla n=(1-H_\varepsilon)\nabla n^-
+H_\varepsilon\nabla n^+
+[n]\,\delta_\varepsilon(\phi)\nabla\phi.
\]

但离散差分一般不满足精确乘法法则：

\[
D_h\{(1-H_h)n_h^-+H_hn_h^+\}
\ne
(1-H_h)D_hn_h^-+H_hD_hn_h^+
+[n]_h\delta_h(\phi_h)D_h\phi_h.
\]

因此“先离散再自动微分”和“先写连续分布公式再分别离散”是两种不同算法；混用时 forward 与
adjoint 即使各自看似合理，也不再对应同一目标。

### 5.2 只能选择一条离散路径

**路径 V：volume-first，推荐作为第一版。**

\[
n_h(\theta)\longrightarrow D_hn_h
\longrightarrow F_h=P_{G,h}^{\mathrm{cone}}D_hn_h.
\]

- 跳跃由 `D_h n_h` 的窄带差分隐式表示；
- VJP 必须由这条完整计算图或其严格 transpose 生成；
- **禁止**再额外加入显式 `delta_epsilon(phi) grad(phi)`，否则双计跳跃。

**路径 S：surface-first，只适合后续高风险分支。**

\[
F_h=F_h^{\mathrm{side}}(n^-,n^+,\phi)
+Q_{G,h}^{\Gamma}(\phi,[n]).
\]

- `Q^Gamma` 对 ray-interface 交点做独立 surface quadrature；
- Jacobian 必须包括交点位置、法向、跳跃 trace 与 quadrature weight 随 `phi` 的变化；
- **禁止**同时对含 sharp jump 的 assembled volume 再求差分。

两条路径必须在同一解析 phantom 上分别做收敛测试，不能把 V 路径的 forward 与 S 路径的
“手写 adjoint”拼接。

### 5.3 锐界面沿射线积分的病态项

对直线 ray `x(s)=o+s r`，若交点 `s_i` 都是横截的，即
`grad(phi)(x_i) dot r != 0`，则 distribution identity 给出 [D]：

\[
\int [n]\delta(\phi(x(s)))\nabla\phi(x(s))\,ds
=\sum_i [n](x_i)
\frac{\nabla\phi(x_i)}{|\nabla\phi(x_i)\cdot r|}.
\]

实际 BOS 还要乘投影到传感器方向的算子、光学常数，并对 aperture sub-rays 加权。该式暴露
三个风险：

1. **grazing/tangent ray：** 分母接近 0，shape sensitivity 可爆炸；交点出现/消失时不可光滑。
2. **多交点/拓扑变化：** 交点排序与数量变化，普通自动微分的 root finder 未必给出可靠导数。
3. **模型适用性：** 真正折射率不连续时，光线方向满足 eikonal/Snell 条件。把 straight-ray
   weak-deflection 积分直接作用于分布项，需要额外说明弱折射/线性化范围。

尤其是 ray equation 中类似 `(1/n) grad(n)` 的写法，在 `n` 跳跃时涉及不连续系数与分布的
乘积；除非从界面条件推导，或明确以常数 `1/n0` 线性化，否则连续表达本身可能不唯一 [D]。

### 5.4 nonlinear forward 的“exact adjoint”定义

若光线路径也由折射率决定，`F(theta)` 的导数含两部分：

\[
\frac{dF}{d\theta}
=\left.\frac{\partial F}{\partial n}\right|_{x(s)}\frac{dn}{d\theta}
+\frac{\partial F}{\partial x(s)}\frac{dx(s)}{d\theta}.
\]

冻结 ray path 会漏掉第二项。它可以是一个明确的近似，但不能称对 deployed curved-ray forward
的 exact VJP。类似地，2026 年 [level-set PINN 预印本](https://arxiv.org/abs/2607.03772)
通过重新定义界面反传导数来扩大支持区 [P, preprint]；这种 surrogate gradient 可能改善训练，
却不是原 forward 的数学导数。JACRU 必须二选一并如实命名：

- **objective-consistent VJP：** 精确求部署离散 `F_h` 的 transpose-Jacobian；或
- **surrogate interface gradient：** 明说优化的是替代更新，并单独验证偏差与稳定性。

“用了 PyTorch autograd”本身不能证明 forward 离散正确，也不能证明连续伴随正确。

### 5.5 finite aperture 的一致性清单

forward 与 VJP 必须共享：

- 同一组 sub-ray 样本、aperture quadrature 权重与归一化；
- 同一 camera transform、坐标单位、传感器轴投影与 row ordering；
- 同一体素/神经场 interpolation kernel 及边界行为；
- 同一 mask、reliability、pixel aggregation 与有效光线路径；
- 若随机采样，gradient test 时固定随机数或使用可重放样本。

任一项不一致都可能让网络学会补偿算子错误，synthetic 指标仍然好看，却无法迁移到 OERF。

### 5.6 必须先通过的数值数学测试

以下阈值是本项目的 stop gate，不是通用定理；测试用 `float64`，并对步长做 log sweep。

1. **线性 dot-product test**

   \[
   e_A=\frac{|\langle Au,v\rangle-\langle u,A^Tv\rangle|}
   {\max(1,|\langle Au,v\rangle|,|\langle u,A^Tv\rangle|)}<10^{-6}.
   \]

2. **非线性 directional derivative**

   \[
   \frac{F(\theta+tp)-F(\theta)}{t}\to J(\theta)p.
   \]

   在截断误差与舍入误差之间应出现预期收敛区；最佳相对误差须 `<1e-5`。

3. **Jacobian-adjoint identity**

   \[
   \frac{|\langle Jp,r\rangle-\langle p,J^Tr\rangle|}
   {\max(1,|\langle Jp,r\rangle|,|\langle p,J^Tr\rangle|)}<10^{-6}.
   \]

4. **分块测试：** 分别冻结其他变量，对 `b/n-`、`j/n+`、`phi`、camera bias 做 2–3。
5. **无双计解析测试：** 单一平面界面、常数两侧场、已知横截 ray；V 与 S 路径分别对高分辨率
   reference 收敛，组合路径不得产生约两倍 jump response。
6. **离散收敛：** 扫 `h`、`epsilon/h`、ray step 和 aperture sample count；界面响应、重投影和
   gradient error 不应随分辨率无规律漂移。
7. **hard-deployment test：** soft 与 hard gate 均重新 forward；hard 输出若显著破坏 held-out
   reprojection，则禁止展示为最终重建。

任一分块 VJP 失败时，停止网络训练；不能让 learned block 去“吸收”伴随错误。

---

## 6. 最小消融、基线与失败门

### 6.1 四阶段最小实验，不先铺大模型矩阵

| 阶段 | 必须比较 | 唯一要回答的问题 | 通过后才允许 |
|---|---|---|---|
| A. 数学门 | volume-first、surface-first、错误双计版 | 哪条 forward/VJP 离散在平面/球面界面上收敛且稳定？ | 保留一条离散路径 |
| B. 表示门 | smooth TV/Huber；Kadu-style 非学习 level set；当前 diffuse-interface Adam | 显式界面是否在同一 forward、同预算下改善 field 与 interface，而不伤 smooth control？ | 训练 learned update |
| C. 架构门 | generic Learned Primal-Dual；单一 learned proximal；JACRU split update | 收益来自 jump-aware split，还是普通 unrolling/更多参数？ | 写“JACRU mechanism” |
| D. 几何门 | fixed-camera；set encoder；去掉 camera nuisance；thin-ray vs cone-ray | 相机集合、finite aperture 与 bias 分支是否各自有可复现作用？ | 讨论 OERF 迁移 |

最小论文主表建议只保留六个方法：

1. cone-ray CGLS-TV/Huber；
2. 非学习 level-set/diffuse-interface optimization；
3. generic Learned Primal-Dual；
4. JACRU single-proximal（去 split）；
5. JACRU split 但去 camera-set/nuisance；
6. full JACRU candidate。

FNO、DeepONet、NeRIF/TDBOST 不是都要在第一天完整复现。先将 FNO 作为 matched proximal、
NeRIF 作为 BOST neural-field 对手；TDBOST 只有进入 4D 后才是同任务主基线。不同任务上的论文数字
不能横向抄进 superiority 表。

### 6.2 公平预算必须冻结

所有 learned arms 使用相同：

- train/validation/blind parent-field split，禁止相邻切片跨 split；
- camera/view/noise/pose distribution 与新鲜随机种子；
- forward/VJP 调用次数 `K`、优化 wall-clock 和 peak memory 报告；
- 参数量或至少提供 parameter-matched control；
- 同一输入可见信息、mask、校准和 stopping rule；
- 至少 5 seeds，并报告逐 field 配对差值与 bootstrap 95% CI；
- synthetic forward 与 audit forward 分开，避免 inverse crime。

### 6.3 预注册失败门

以下是项目决策阈值，不是文献公认标准。

| 失败门 | 触发后的诚实结论 |
|---|---|
| 任一核心 dot/JVP/VJP test 未达上述阈值 | forward/adjoint 不合格；停止所有 superiority 训练 |
| 显式 surface path 不随 `h, epsilon, ray-step` 收敛，或比 volume path 双计 jump | 删除 surface formulation |
| generic Learned Primal-Dual 与 full JACRU 的配对 95% CI 包含 0，或材料性改善 `<5%` | 不声称 JACRU 架构贡献 |
| 匹配参数量/forward calls 后优势消失 | 只能说容量或计算预算收益 |
| 0-interface smooth control 的 field error 恶化 `>1%`，或稳定产生假界面 | jump-aware representation 失败 |
| interface ASSD/HD95 改善 `<10%` 且 F1@1 voxel 改善 `<0.03` | 不声称界面恢复更好 |
| held-out reprojection error 超过最强基线 `1.05x` | 判为 prior-driven hallucination，不选漂亮体图 |
| geometry/view/noise OOD 保留的域内增益 `<50%` | 禁止称 general/operator generalization |
| 多初始化重投影近似相同、界面显著不同 | 报告不可辨识区间；禁止挑最好看的初始化 |
| exact surface path 明显更慢且不优于 volume-first | 拒绝复杂路径，保留简单方法 |
| 真实实验没有 volumetric truth | 只报告 held-out image-space、重复采集和外部传感器证据 |

“达到某个单次均值”不是通过；需要 blind split、配对置信区间和负对照同时满足。

---

## 7. 不能声称的内容

在获得对应证据前，论文、网页和汇报中禁止以下表述：

1. **“首个 level-set / phase-field tomography。”** 已有直接先例。
2. **“首个 interface-aware DeepONet/FNO。”** IONet、`phi`-DeepONet、IANO 等已覆盖邻近结构。
3. **“首个 physics-unrolled inverse network。”** Learned Primal-Dual、MoDL、Variational Network
   已覆盖核心范式。
4. **“首个 neural / finite-aperture / 4D BOST。”** NeRIF、finite-aperture NIRT、TDBOST 已分别覆盖。
5. **“autograd 给出了 exact adjoint。”** 只能证明对实际计算图求导；不能证明物理或离散实现正确。
6. **“显式 `delta(phi)` 必然更物理。”** 未处理 Snell/eikonal、切向 ray 和离散双计前不能这样说。
7. **“重建出尖锐表面就是激波。”** 接触面、火焰面、折射率组分界面都可能尖锐。
8. **“由折射率唯一恢复压力/温度/完整反应流。”** 还需要组分依赖的 Gladstone-Dale、EOS、
   上游状态及适用的守恒关系；不同界面物理不同。
9. **“优于 DeepONet/FNO/NeRIF/TDBOST。”** 任务、数据、forward、预算和指标未匹配时不能比较。
10. **“真实实验 field-L2 更低。”** 没有独立 volumetric truth 时不存在该指标的真实标签。
11. **“phase-field 收敛已保证。”** 当前代码不是已分析的 phase-field energy。
12. **“对 curved ray/finite aperture 精确。”** 若 forward 是 straight ray、冻结 ray path 或不匹配
    aperture transpose，只能写相应近似。
13. **“可泛化到任意相机数与几何。”** 单 rig/fixed stack 实验最多支持该分布内表现。
14. **“多初始化 spread 就是校准不确定度。”** 它至多暴露优化/不可辨识敏感性，未做 coverage
    calibration 前不是概率置信区间。
15. **“JACRU 是 4D 方法。”** 当前定义是静态 3D；加入时间维、动态先验和 TDBOST 对照后才可写 4D。

---

## 8. 可验证主张与一手来源边界

| 可验证主张 | 一手来源 | 来源实际支持 | 来源不支持的外推 |
|---|---|---|---|
| 连续背景与 level-set anomaly 可联合、交替做 tomography | [Kadu et al.](https://arxiv.org/abs/1704.00568) | parametric level set、双层/交替优化、limited-data phantom [P] | finite-aperture BOST、learned split 的优越性 |
| Phase-field 可作二值 tomography 的 perimeter relaxation | [Dunbar & Elliott](https://arxiv.org/abs/1811.02865) | double-obstacle、离散下降、Gamma-convergence [P] | 当前二阶 `phi` 平滑代码拥有同样性质 |
| DeepONet 使用固定传感器 branch 与查询坐标 trunk | [DeepONet](https://arxiv.org/abs/1910.03193) | 连续算子学习基本结构 [P] | 可变相机 BOST 或不连续逆问题已解决 |
| FNO 是 Fourier-space kernel neural operator | [FNO](https://arxiv.org/abs/2010.08895) | 所研究 PDE 上的算子学习 [P] | 任意 BOST inverse 或 sharp interface 保证 |
| 线性 reconstruction 对所研究不连续 PDE 算子可能低效 | [Lanthaler et al.](https://arxiv.org/abs/2210.01074) | 对特定问题的下界；FNO/shift-DeepONet 结果 [P] | JACRU 必然胜 DeepONet，或 BOST 共享同一下界 |
| 显式界面可进入 neural operator | [IONet](https://arxiv.org/abs/2308.14537)、[`phi`-DeepONet](https://arxiv.org/abs/2604.08076)、[IANO](https://ojs.aaai.org/index.php/AAAI/article/view/39887) | 子域网络、界面 embedding/geometry encoding等 [P] | 这些方法已完成 BOST tomography |
| 非线性 forward 可放进 learned primal-dual unrolling | [Learned Primal-Dual](https://arxiv.org/abs/1707.06474) | 固定层数、forward/backprojection 与 learned proximal [P] | JACRU 的界面分块天然新或更好 |
| 显式 data-consistency 与 learned prior 可迭代组合 | [MoDL](https://arxiv.org/abs/1712.02862) | model-based learned inverse 与共享权重 [P] | cone-ray jump VJP 已被解决 |
| NeRIF 已表示折射率与梯度并处理二者一致性 | [NeRIF 开放全文](https://arxiv.org/html/2409.14722v2) | ray sampling、AD/ND loss、BOST 实验留投影验证 [P] | 跨样本 operator 或 sharp-interface unrolling |
| Finite aperture 已进入 BOS forward 与 neural implicit inverse | [Molnar et al.](https://arxiv.org/html/2402.15954) | thin-ray/cone-ray 对照、NIRT、TV/shock 场 [P] | 分布 jump 的 exact surface adjoint |
| TDBOST 已面向 4D BOST 做张量分解重建 | [ACM DOI](https://doi.org/10.1145/3809488)、[作者代码](https://github.com/Hyz617/TDBOST) | 4D 表示与公开工程结构 [P] | JACRU 静态界面方法的有效性 |
| BOS tomography 后已有 level-set 与 RH 压力映射 | [Optics Letters](https://doi.org/10.1364/OL.571452) | 特定 shock 实验、速度/密度与压力映射、传感器对照 [P] | 仅凭任意 `n` 场即可唯一恢复压力 |
| 可人为扩大 level-set 的反传支持区 | [Level-set PINN 预印本](https://arxiv.org/abs/2607.03772) | gravimetry domain inverse 中的 interface-aware surrogate backprop [P, preprint] | surrogate 等于 JACRU forward 的 exact VJP |

可在论文中安全写的负面陈述是：

> Existing works cover level-set tomography, phase-field regularization, interface-aware
> operator learning, physics-based unrolling, neural BOST fields, and finite-aperture BOST
> separately. We therefore test a narrower hypothesis: whether a BOST-specific,
> discrete-consistent split update over field, interface, and camera nuisance variables adds
> value under matched budgets.

这是一条**可检验的研究动机**，不是“已证明文献空白”。

---

## 9. 最小 Go / No-Go 决策

### 允许继续的顺序

1. 冻结方案 A 或 B；推荐 `b + H(phi)j`，删除冗余 jump 自由度。
2. 先实现 volume-first cone-ray forward 与 exact discrete VJP。
3. 通过平面界面、球面界面、0-interface 的 dot/JVP/VJP 与收敛测试。
4. 跑非学习 level-set/diffuse-interface 与 generic Learned Primal-Dual。
5. 只有 generic unrolling 被同预算 split update 稳定击败，才引入 set encoder/FNO proximal。
6. 只有 synthetic blind 与 geometry OOD 通过，才接 OERF 的 held-out reprojection/外部传感器。

### 立即 No-Go 的情况

- 无法从组内 forward 获得可测试的 VJP/JVP，且无法独立实现一致版本；
- 真实任务的界面类型、观测量、相机几何和校准误差都未定义；
- JACRU 的优势只存在于生成训练数据的同一个 forward；
- generic Learned Primal-Dual 在公平预算下等效或更好；
- 多初始化表明界面不可辨识，但没有外部观测约束；
- 研究叙事仍依赖“把 level set、FNO 与 unrolling 拼在一起就是创新”。

## 最终红队意见

JACRU 目前不是一个已成立的算法结果，而是一个有价值但风险很高的**机制假设**。最有可能形成
贡献的部分不是网络名字，而是：

1. BOST 跳跃场的非冗余状态；
2. finite-aperture cone-ray forward 与界面导数的离散一致性；
3. generic unrolling 无法替代的 field/interface/camera 分块更新；
4. 对相机数量、几何、forward mismatch 和不可辨识性的诚实证据。

在这些门通过前，结论应保持 **RED for novelty claim / AMBER for falsification work**。若最终
generic baseline 获胜，得到“显式 jump-aware split 没有额外价值”的严格负结果，也比用不一致
forward 造出一次局部胜利更可靠。
