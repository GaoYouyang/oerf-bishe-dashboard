# PDHG 一手文献与可创新边界

日期：2026-07-17

## 1. 先说结论

当前 one-pair PDHG 是必须做强的确定性基线，不是论文创新本身。以下表述均不
成立：

- 首次将 PDHG 用于层析；
- 首次做 3D TV 或 Huber-TV；
- 首次把物理 forward/adjoint 放入 primal-dual；
- 首次将测量算子与 TV 差分算子分开；
- 首次做 block/diagonal primal-dual step；
- 首次 learned primal-dual。

较可信的新问题是：

> 能否为含 centered gradient、有限孔径 ray integration 与 detector covariance
> whitening 的带符号 matrix-free BOST 算子，构造可证明安全、按相机或探测器
> 结构分块的 primal-dual majorizer，并在同物理调用预算下改善反应前沿和坏尾？

这条研究假设已经完成一次窄口径证伪。scalar-step PDHG v2 在两个 opened
replicate 上暴露重复近零场停滞；随后 signed factor Gate A mechanics 通过，但
deterministic Gate B 的 voxel-factor 相对 scalar 只改善 1.321%，与 graph-PCGLS
的 mean error gap 仍为 133.439%，八门过五门并严格 NO-GO。它不授权 learned
proximal，也不外推为一般 PDHG 或真实 OERF 无效。

## 2. 当前离散分账为何合理

冻结算子写成：

```text
A = W P G_c
K = [A ; D_+]
```

- `G_c`：BOST forward 内部的 voxel-centered gradient；
- `P`：ray sampling、camera-plane projection 与路径积分；
- `W`：detector covariance whitening；
- `D_+`：只定义 TV/Huber 先验的 forward-Neumann gradient。

`G_c` 与 `D_+` 不必相同，但 `A/A^T` 和 `D_+/D_+^T` 必须分别通过精确离散
伴随测试。中心差分会消去 Nyquist checkerboard mode，而 forward difference
会惩罚它；因此把 `D_+` 用作先验不是重复物理梯度，而是在测量零空间附近加入
明确的结构选择。

论文中应使用以下窄表述：

> We distinguish the centered finite-difference operator used exclusively in
> the calibrated BOST forward model from the forward-Neumann operator used to
> define the isotropic TV/Huber prior. Both operator pairs are implemented as
> exact discrete adjoints.

## 3. Huber 共轭近端必须怎样缩放

对

```text
g(z) = lambda * h_delta(||z||_2)
```

及 unit-slope Huber：

```text
h_delta(r) = r^2/(2 delta), r <= delta
h_delta(r) = r - delta/2,   r > delta
```

有：

```text
g*(q) = indicator(||q||_2 <= lambda)
        + delta/(2 lambda) * ||q||_2^2

prox_{sigma g*}(v)
  = projection_ball_lambda(v / (1 + sigma delta/lambda))
```

不能只把 `lambda` 吸收到 `D_+` 而保持原 `delta`；Huber 情形下那会改写目标
函数。当前实现让 `lambda` 保留为 dual-ball radius，`delta` 单独冻结。

## 4. Block step 的安全条件

两个 dual block 的精确条件为：

```text
tau * lambda_max(
  sigma_A * A^T A + sigma_D * D_+^T D_+
) < 1
```

一个更保守、可审计的充分条件是：

```text
tau * (
  sigma_A * ||A||^2
  + 4 sigma_D * (hx^-2 + hy^-2 + hz^-2)
) < 1
```

普通 power iteration 只是近似，不是 certified upper bound。当前 E2 smoke 使用
固定迭代、固定安全膨胀与更保守 `eta` 压力检查；报告必须把 norm setup 的
物理调用和单次 reconstruction 分开。

## 5. 下一算法候选：Covariance-aware signed majorizer

### 5.1 为什么普通对角预条件不能直接照搬

Pock–Chambolle 对角步长依赖 `|K|` 的真正绝对行列和。BOST 的 `A` 含带符号
centered difference，不能用 `A 1` 或 `A^T 1` 代替 `|A|1`、`|A|^T1`：正负
抵消可能把结果错误地压到零，并产生不安全的大步长。

### 5.2 可研究的结构

若将 `A = B G_c`，可以寻找非负 majorizer：

```text
M = |B| |G_c| >= |B G_c| = |A|
```

再利用：

- detector covariance 的 view/block spectrum；
- ray-local interpolation 权重与路径长度；
- camera pose / detector graph block；
- support 与 forward-Neumann regularizer 的解析行列和；

构造 per-view data dual steps 与 voxel primal steps。需要证明或数值认证
`||Sigma^(1/2) K T^(1/2)|| < 1`，并把 absolute-majorizer setup 成本列入账本。

### 5.3 论文创新门槛

这个方向只有同时满足以下条件才可能形成方法贡献：

1. 相比 scalar block-PDHG，在相同 `F/A^T` calls 下改善 mean 与坏尾；
2. absolute majorizer 有推导、tiny dense oracle 与 matrix-free 复算；
3. 不是依赖 truth 或 morphology 标签的 per-field 调参；
4. covariance 输入来自独立 flow-off，而不是 synthetic oracle；
5. held-out camera/session 和 geometry/noise OOD 不发散；
6. 与普通 diagonal PDHG、PCGLS、TV/Huber、learned primal-dual 同预算比较。

## 6. Learned proximal 仍未解锁

只有 deterministic PDHG 暴露出重复、可观测的剩余缺口，才允许学习 bounded
step、threshold map 或小型 residual correction。若网络不是某个凸函数的真实
proximal 或可证明的平均非扩张映射，应称为 learned denoiser/unrolled update，
不能继承 Chambolle–Pock 收敛定理。

## 7. 一手来源阅读顺序

1. Chambolle & Pock, *A First-Order Primal-Dual Algorithm for Convex
   Problems with Applications to Imaging*, JMIV 2011：算法、
   forward-Neumann gradient、Huber conjugate。
   [作者开放稿](https://hal.science/hal-00490826/document) ·
   [DOI](https://doi.org/10.1007/s10851-010-0251-1)
2. Sidky, Jørgensen & Pan, *Convex optimization problem prototyping for image
   reconstruction in computed tomography with the Chambolle-Pock algorithm*：
   CT 离散算子、伴随与 problem prototyping。
   [开放稿](https://arxiv.org/abs/1111.5632)
3. Pock & Chambolle, *Diagonal Preconditioning for First Order Primal-Dual
   Algorithms in Convex Optimization*, ICCV 2011：绝对行列和与对角 metric。
   [DOI](https://doi.org/10.1109/ICCV.2011.6126441)
4. Ma et al., *Understanding the Convergence of the Preconditioned PDHG
   Method: A View of Indefinite Proximal ADMM*：固定预条件 PDHG 的收敛条件与
   紧性；不能把固定 metric 结论自动外推到逐步学习 metric。
   [开放稿](https://arxiv.org/abs/2301.02984) ·
   [DOI](https://doi.org/10.1007/s10915-023-02105-9)
5. Loris & Verhoeven, *On a generalization of the iterative soft-thresholding
   algorithm for the case of non-separable penalty*：层析中的 Huber-TV。
   [开放稿](https://arxiv.org/abs/1203.4451)
6. González et al., *Three-dimensional optical diffraction tomography by
   Lp minimization*：光学梯度层析、TV、边界与常数零空间。
   [开放稿](https://arxiv.org/abs/1209.0654) ·
   [DOI](https://doi.org/10.3934/ipi.2014.8.421)
7. Grauer et al., 3D BOS tomography with TV regularization：证明 BOST-TV
   本身不是空白。
   [DOI](https://doi.org/10.1016/j.combustflame.2018.06.022)
8. Adler & Oktem, *Learned Primal-Dual Reconstruction*, IEEE TMI：每层显式
   forward/adjoint 的 learned baseline。
   [开放稿](https://arxiv.org/abs/1707.06474) ·
   [DOI](https://doi.org/10.1109/TMI.2018.2799231)

## 8. 当前证据边界

- 当前 synthetic runner 的 detector `scale_by_view` 由解析 truth 的 clean
  projection RMS 构造。这适合固定 relative-noise mechanism diagnostic，但真实
  部署时不可获得，因此不能把当前 covariance/majorizer 结果称为完全
  truth-independent；
- 下一份 majorizer 协议若要支持实验结论，`scale_by_view` 必须来自独立
  flow-off repeats 或 calibration acquisition，并与 reconstruction truth 隔离；
- 上述内容是方法与 prior-art 审计，不是实验胜出；
- scalar PDHG v2 已在 E1 116/116 tests、12/12 stress 和 784 条 paired rows 下
  得到严格 NO-GO；该结果只覆盖 zero-init、scalar-step 与冻结 K/alpha 网格；
- signed factor 已完成逐元素 BOST majorizer、13/13 Gate A attestation、256 条
  Gate B 方法行与 4,048 项独立复核；Gate B 为 E2 mechanism NO-GO；
- 没有 OERF flow-off、held-out camera/session 或实验三维真值；
- fresh 与 learned proximal 继续封存。
