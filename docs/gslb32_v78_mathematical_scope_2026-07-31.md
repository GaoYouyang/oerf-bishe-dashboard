# GSLB32 v78 数学作用域与不可声称边界

> 日期：2026-07-31  
> 角色：结果无关的线性代数说明；不依赖 v78 正式数值。  
> 当前状态：`algorithm_breakthrough=false`。

## 1. 对象与记号

令 `A: R^n -> R^m` 是冻结九视角 straight-ray 代理算子，`S in R^(n x 127)` 是
8x4x4 三线性样条经过 support 和 zero-gauge 处理后的有效父空间。定义：

```text
C = S^T A^T A S
G = (L_I S)^T(L_I S) + epsilon (M S)^T(M S)
```

其中 `L_I` 是冻结 interior-gradient，`M` 是 support 加权场算子，`epsilon>0` 按冻结
规则给定。在 127 维有效父空间上 `G` 为正定。广义特征向量按

```text
C v_i = lambda_i G v_i,
lambda_1 <= ... <= lambda_127
```

排列，并令 `u_i=S v_i`。GSLB32 使用 `U32=[u_1,...,u_32]`。

## 2. 命题一：模式是低观测能量模式，不是精确零空间

由于模式按 `G` 正交归一，任意单个模式满足：

```text
||A u_i||_2^2 = lambda_i,
||L_I u_i||_2^2 + epsilon ||M u_i||_2^2 = 1.
```

因此较小的 `lambda_i` 只表示：相同正则化梯度能量下，该模式在当前冻结几何中的
observation energy 较低。

它不推出：

- `A u_i = 0`；
- `u_i in Null(A)`；
- `u_i` 对真实 BOST、另一套几何或 curved ray 仍低可观测；
- 在 `U32` 中增加校正不会伤害 field 或 gradient。

所以允许名称是 geometry-only low-observability spline modes，禁止名称是 exact nullspace、
learned nullspace 或 guaranteed data-consistent correction。

## 3. 命题二：精确 lift 只属于 anchor，不属于完整初值

在线初值为：

```text
h = A^T z
x0 = h + U32 a
```

其中 `z` 来自只读 observation 的便宜 dual proposal。由定义可知
`h in Range(A^T)`。但一般不存在证据表明 `U32 a in Range(A^T)`，所以也不能推出
`x0 in Range(A^T)`。

准确表述必须是：

> exact-adjoint-lifted anchor plus a geometry-only low-observability spline correction

而不能把完整 GSLB32 初值写成 exact `A^T` lift、Range-safe initializer 或 observable-
Krylov initializer。

## 4. 命题三：CGLS 不会消除初值中的真零空间分量

冻结的一步 CGLS 为：

```text
r0 = y - A x0
s0 = A^T r0
t0 = A s0
alpha = <s0,s0> / <t0,t0>
x1 = x0 + alpha s0
```

因为 `s0 in Range(A^T)=Null(A)^perp`，设 `P_N` 为到 `Null(A)` 的正交投影，则：

```text
P_N x1 = P_N x0 = P_N(U32 a).
```

这个等式对任意后续标准 CGLS 更新同样成立。也就是说，如果样条校正含有错误的真零空间
分量，未修改 CGLS 无法把它修掉。GSLB32 的 field/full-gradient/interior-gradient harm
门因此不是装饰性指标，而是该方法能否成立的必要防线。

## 5. 命题四：一步尺度只保证 observation residual 的线搜索最优

只要冻结 breakdown 条件 `||s0||>0`、`||A s0||>0` 成立，上述

```text
alpha = ||s0||_2^2 / ||A s0||_2^2
```

恰好最小化 `||r0-alpha A s0||_2^2`。由于 `alpha=0` 是可行比较点，一步后的
observation residual 不大于 `x0` 的 residual。

这仍不保证：

- field error 降低；
- gradient error 降低；
- 相对 Zero-K2 或 Zero-K4 等价；
- 噪声、几何误差或 curved ray 下稳定；
- wall time 或 peak RSS 下降。

## 6. 成本恒等式

对每个部署 cell：

```text
1 A^T : h = A^T z
1 A   : r0 = y - A x0
1 A^T : s0 = A^T r0
1 A   : t0 = A s0
```

所以在线精确账是 `2A+2A^T`。v78 truth-aware coefficient search 本身不新增精确调用，
是因为 `A U32`、`A^T A U32` 和 `A A^T A U32` 已在 geometry setup 中预计算；这部分
冷启动成本仍须单列为每套几何 `160A+32A^T`，当前没有证明可在真实流程中充分摊销。

## 7. 对后续算法设计的直接要求

若 v78 75/75 通过，下一阶段也只能训练 observation-only 的 32 系数预测器，并同时：

1. 与 Zero/BP/CGLS/PCGLS/dual-ridge、解析投影和 reduced-basis controls 公平比较；
2. 使用部署可见 gate 决定接受或回退，绝不能在线读取 truth；
3. 逐 trajectory 报 field、full-gradient、interior-gradient、observation 的 p50/p90/worst；
4. 把 setup、接受、拒绝和 fallback 全部计入 `A/A^T`、wall 与 RSS；
5. 在外部门和真实 BOST 迁移前保持 `algorithm_breakthrough=false`。

若 v78 失败，只能说明 32 维空间在冻结 21-start 搜索下未找到全 cell witness，不能写成
数学不可行，也不能用更大神经网络绕过表示上界诊断。

