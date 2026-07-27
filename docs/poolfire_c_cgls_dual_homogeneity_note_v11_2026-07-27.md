# 为什么 v11 必须“奇对称但非线性”

## 结论

PoolFire C 主线要预测的 K3 dual certificate 不是一个固定线性滤波器。它有两个同时
成立的结构：

1. **严格一阶齐次和奇对称：**`G(a y)=a G(y)`；
2. **通常不可加：**`G(y1+y2) != G(y1)+G(y2)`。

因此，任意普通全线性 KRR 都只能近似一个方向相关的 Krylov 映射；而 v11 使用
per-frame RMS normalization 加 odd symmetrization，恰好保留第一个结构，同时允许
第二个结构存在。

这不是“新定理”或论文突破，而是模型设计必须满足的数值线性代数依据。

## dual certificate 递推

令 `A` 是冻结的三视角 straight-ray proxy，`y` 是 observation。zero-start CGLS
可以同时维护 field-space 方向和 detector-space dual 方向：

```text
r0 = y
g0 = A^T r0
p0 = g0
d0 = r0
x0 = 0
z0 = 0

qk       = A pk
alpha_k  = ||gk||^2 / ||qk||^2
x{k+1}  = xk + alpha_k pk
z{k+1}  = zk + alpha_k dk
r{k+1}  = rk - alpha_k qk
g{k+1}  = A^T r{k+1}
beta_k   = ||g{k+1}||^2 / ||gk||^2
p{k+1}  = g{k+1} + beta_k pk
d{k+1}  = r{k+1} + beta_k dk
```

由构造始终有：

```text
xk = A^T zk
```

所以网络只需预测 detector-space `z3`，再用一次精确 `A^T` 就能把初值严格限制在
`Range(A^T)`。

## 为什么严格齐次

把 observation 换成 `a y`。初始 `r0、g0、p0、d0` 全部乘以 `a`；`alpha_k`
和 `beta_k` 都是两个二次型的比值，因此分子分母同时乘以 `a^2`，数值不变。递推中
所有向量继续乘以 `a`，所以：

```text
z3(a y) = a z3(y)
```

令 `a=-1`，立即得到 `z3(-y)=-z3(y)`。这就是 v11 不使用普通带偏置 CNN 输出，
而对同一网络做：

```text
0.5 * (f(y / rms) - f(-y / rms)) * rms
```

的原因。

## 为什么通常不是线性的

仅看第一步已经足够。令 `B=A A^T`，则：

```text
alpha_0(y) = (y^T B y) / (y^T B^2 y)
z1(y)      = alpha_0(y) y
```

如果 `B` 在数据子空间上不是单位阵的标量倍数，`alpha_0` 就随 `y` 的谱方向变化。
取 `B` 的两个不同特征值对应方向 `u、v`：

```text
alpha_0(u) != alpha_0(v)
alpha_0(u+v)
```

又是两个方向能量加权后的新比值。于是一般有：

```text
z1(u+v) != z1(u) + z1(v)
```

后续 `alpha_k、beta_k` 继续引入方向相关的有理非线性。K3 certificate 因此是一个
odd、degree-1 homogeneous、但非 additive 的 observation operator。

## 冻结 PoolFire 算子上的数值核验

我从五条 fit trajectory 的 505 帧中，用固定随机种子抽取 40 对 observation，
独立重算 K3 dual certificate。没有读取 fresh、stopping validation 或 test。

| 检验 | 结果 |
|---|---:|
| `||G(2.75y)-2.75G(y)|| / ||G(2.75y)||` 最坏 | `3.21e-16` |
| `||G(-y)+G(y)|| / ||G(y)||` 最坏 | `0` |
| 可加性相对偏差中位 | `4.15%` |
| 可加性相对偏差 p90 | `6.77%` |
| 可加性相对偏差最坏 | `9.33%` |

这里的可加性偏差定义为：

```text
||G(y1+y2)-G(y1)-G(y2)|| / ||G(y1+y2)||
```

数值结果与推导一致：齐次和 odd 性达到浮点精度，但不可加性不是舍入噪声，而是
`4%–9%` 量级的真实结构。

## 对论文算法的含义

v10.9 的 full-linear KRR 允许所有 detector 坐标耦合，却仍为 `0/5`。它失败不只
可能来自样本少和过拟合，也来自模型族与 Krylov 映射结构不一致。

v11 的设计因此不是“换成 CNN 试试看”，而是：

- **odd symmetrization** 精确满足符号对称；
- **RMS normalize / denormalize** 精确满足一阶尺度齐次；
- **dilated local convolution** 表示 detector 上局部、多尺度谱交互；
- **global channel context** 表示 view 间方向相关的 Krylov 系数；
- **exact `A^T` lift** 保证 field 初值在 `Range(A^T)`；
- **K1 后物理 loss** 避免只降低 certificate L2 却伤害 observation。

是否成功仍必须由五条完整 trajectory LOTO、独立 checkpoint replay、fresh release
和真实 BOST 迁移逐层判定。当前理论依据成立不等于算法突破。
