# 从中心差分到 BOST 神经场混叠：零基础学习路线

适用对象：刚开始学流体光学、三维反演和神经隐式场的物理本科生。

目标：不是背网络名字，而是能解释“为什么同一个场在离散 renderer 上很好，在连续 renderer 上却很差”，并能独立复现一个反例。

## 1. 先把 BOST 观测说成人话

空间中温度、组分和密度变化会引起折射率 `n(x,y,z)` 变化。光线穿过非均匀折射率场时发生偏折；在弱偏折近似下，一个像素的两个方向位移可以理解为：沿这条光线，把折射率梯度在相机平面两个方向上的分量加起来。

简化表达是：

\[
y_r \approx C_r\int_{\text{ray }r}P_r\nabla n(x)\,ds + \epsilon_r.
\]

- `n(x)`：要恢复的三维折射率场；
- `nabla n`：空间梯度，决定光线往哪边弯；
- `P_r`：把三维梯度投影到相机的两个传感方向；
- 积分：一条 ray 上许多位置的贡献叠加；
- `epsilon`：光流、背景、标定、有限孔径等误差。

因此，BOST 不是普通 CT 的“积分场值”，而是“积分场梯度”。导数怎样计算会直接进入 forward model。

## 2. 自动导数和中心差分到底差在哪里

设一维 Fourier 分量为：

\[
n_f(x)=\sin(\pi f x).
\]

自动导数给出连续导数：

\[
n_f'(x)=\pi f\cos(\pi f x).
\]

中心差分步长为 `h` 时：

\[
D_h n_f(x)=\frac{n_f(x+h)-n_f(x-h)}{2h}
=\frac{\sin(\pi f h)}{h}\cos(\pi f x).
\]

二者振幅之比是：

\[
\frac{D_h n_f}{n_f'}=\operatorname{sinc}(\pi f h)
=\frac{\sin(\pi f h)}{\pi f h}.
\]

这个比值就是中心差分对不同频率的传递函数。它不是始终接近 1。

本地实验使用 `h=2/15`，各频率的比例是：

| `f` | `sinc(pi f h)` | 直观含义 |
|---:|---:|---|
| 1 | `0.9710` | 几乎保留 |
| 2 | `0.8871` | 轻微衰减 |
| 4 | `0.5936` | 已明显衰减 |
| 5 | `0.4135` | 只剩约四成 |
| 6 | `0.2339` | 很弱 |
| 8 | `-0.0620` | 幅度很小且符号翻转 |
| 16 | `0.0607` | 几乎看不见 |

这解释了关键反例：若网络包含 `f=8,16`，它可以在 `FD(h)` 训练链中放入很强的高频结构，却只对离散投影产生很小贡献；换成 continuous AD，这些结构会重新以大导数出现。

## 3. 为什么“AD 精确”也不等于“结果正确”

AD 精确计算的是当前神经函数的导数。若函数为了拟合噪声长出了错误高频，AD 会非常准确地把错误高频求导出来。问题不在微积分实现，而在表示带宽、采样尺度、噪声和有限视角反演共同决定的可辨识性。

反过来，FD 可以抑制高频噪声，却也可能让网络利用差分传递函数的盲区。二者都不是单独的真理：

- AD：连续语义清楚，但会放大高频表示中的噪声和伪影；
- FD：与特定网格/步长一致，但可能产生分辨率依赖和 aliasing；
- hybrid：能折中，但固定比例并不自动解决零空间和跨分辨率问题。

## 4. 本地证据应该怎样读

### 4.1 预检

低频 `[1,2,4]` 与高频 `[1,2,4,8,16]` 都用 `FD(h)` 训练。高频模型的 central held-out projection 略好，三维场却在 5/8 单元明显更差；central-test 与场损害的符号一致率只有 0.25。

这一步只能提示共享离散 renderer 可能掩盖问题，不能定因。

### 4.2 连续审计

在 14 个新角度上，分别用 `FD(h)`、`FD(h/2)`、`FD(h/4)` 与 AD 重渲染同一网络。7 个未用于事后探针的单元全部出现：高频 AD 比低频 AD 差，且高频 AD 比自身 `FD(h)` 差。

这一步支持“连续/离散梯度混叠机制在该 synthetic proxy 中存在”，仍不支持真实火焰或新算法。

### 4.3 MGRS NO-GO

MGRS 让高频残差同时通过四个 renderer 的 development 门。它改善了 dense-angle continuous projection，但没有稳定降低场误差。原因是所有 renderer 仍观察有限角投影，无法完全约束投影零空间。

## 5. 文献按角色阅读

### A. 直接组内主线

**He et al., NeRIF**

[开放 HTML](https://arxiv.org/html/2409.14722v2) · [Physics of Fluids](https://doi.org/10.1063/5.0250899)

重点提取：网络如何同时输出折射率和梯度；AD/数值微分一致性如何进入损失；随机 ray sampling、九视角 numerical validation 与实验 fiber-bundle 系统各承担什么证据。不要只看最终图。

### B. 最直接的新颖性碰撞

**Lu et al., Neural Refractive Index Primitives (2026 preprint)**

[开放 HTML](https://arxiv.org/html/2605.11454) · [作者代码](https://github.com/Weihu22/Neural-Implicit-Reconstruction-for-BOS)

重点提取：Fourier/hash encoding；automatic/discrete/hybrid gradient；为何 Fourier + AD 出现梯度噪声；为何离散梯度和 hash 更稳定；jitter、mask 与 hierarchical sampling。它已经排除了“换 hash”“加 hybrid”“加 jitter”作为单独创新点。

### C. 算子学习里的 aliasing

**Bartolucci et al., Representation Equivalent Neural Operators**

[arXiv](https://arxiv.org/abs/2305.19913)

重点提取：连续 operator 与离散 representation 的关系；operator aliasing 怎样表现为跨网格/分辨率不一致。它提供概念框架，但没有替我们证明 BOST 梯度投影中的具体机制。

### D. 成像里的尺度感知抗混叠

**Barron et al., mip-NeRF**

[arXiv](https://arxiv.org/abs/2103.13415)

重点提取：为什么一个像素不是一条无限细 ray；怎样把 sampling footprint 放进 positional encoding。后续若做 BOST footprint-aware Fourier field，应借鉴它并明确差别：BOST 积分的是折射率梯度，而且还涉及 detector pixel、aperture、ray interval 与光流位移。

### E. 曲光线与更强 forward

**Zhao et al., Single View Refractive Index Tomography with Neural Fields**

[CVPR 2024 正式开放页](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)

重点提取：可微 curved-ray tracing 如何进入 neural field optimization；不要把单视角光源先验直接等同于 BOST 多相机设置。

## 6. 七天最短学习路径

### Day 1：手推导数传递函数

手算上面的 `sinc(pi f h)`，画 `f in [0,20]` 曲线，标出零点、符号翻转和 Nyquist 附近。

验收：不用看笔记，能解释为何 `f=8` 在 `h=2/15` 的中心差分下会近乎消失且变号。

### Day 2：跑单个 Fourier 分量

用 PyTorch 构造 `sin(pi f x)`，比较 autograd 和 central difference；依次改 `h` 与 `f`。

验收：数值比值与解析 sinc 在 `1e-4` 内一致。

### Day 3：读 BOST forward

读 `learning_labs/gcs_fourier_bost_lab.py` 的 `coordinate_gradient()` 与 `render_neural_bundle()`，画出 `field -> gradient -> camera projection -> ray sum -> displacement`。

验收：能指出哪一步改变 `h`，哪一步改变 camera angle，哪一步改变 quadrature。

### Day 4：复现连续审计

读配置、运行 runner、核对 checksum。不要先看结论，自己从 `paired_units.csv` 判断 7/7 是否成立。

### Day 5：理解零空间

对一个小矩阵 `A` 求 SVD，构造 `delta x in ker(A)`，验证 `A(x+delta x)=Ax`。再解释为什么增加多个差分 renderer 不一定等于增加新相机。

### Day 6：读 NeRIF 与 2026 NIR Primitives

各写一页：输入、输出、forward、gradient、loss、sampling、split、真实实验、最弱证据点。特别比较“两个头的一致性”和“单一 n primitive + 不同 gradient operator”。

### Day 7：写算法卡，不写大网络

算法卡必须包含：物理失败、可部署输入、基线、失败回退、主指标、独立 split、计算预算和禁止主张。先写 H1-minimal residual，再判断学习模块是否真的需要。

## 7. 论文级最低门槛

一个可信结果至少要同时满足：

1. 多个独立三维场，不把 rays、相机或优化 seed 当场样本；
2. 训练、development、test camera/session 明确分离；
3. continuous/curved/finite-aperture forward 至少有一项不与训练 renderer 共用离散链；
4. field relative-L2、gradient/H1、逐 camera 尾部和 displacement 同时报告；
5. 与 NeRIF、2026 NIR-Primitives 的 automatic/discrete/hybrid、低频模型及经典 H1/TV 比较；
6. 同样的 forward 调用数、wall time、显存和参数量至少报告一套公平预算；
7. 真实 OERF 数据至少有 held-out view/session，且温度推断的组分假设单独说明；
8. 失败单元、阈值和 post-hoc 分析全部保留。

在这些门之前，最准确的称呼是“synthetic mechanism study”或“candidate audit”，不是高质量论文完成。
