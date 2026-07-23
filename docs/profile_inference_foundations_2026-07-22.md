# 从 variable projection 到物理目标正交分数：给初学者的学习路线

这份导读服务于当前的窄问题：在三维/四维 BOST 中，低维几何或运动参数
`q` 与高维折射率场 `x` 同时未知时，怎样避免把场的不确定性误当成几何信息。
它不是统计学术语表，也不是把几个公式贴到现有网络外面。

## 先把问题说成人话

局部模型写成：

```text
y = B(q) x + noise
```

- `y`：多相机、多帧的背景位移或投影观测；
- `B(q)`：由相机几何、射线模型和输运参数决定的 forward operator；
- `x`：要重建的三维折射率/密度场；
- `q`：少量但关键的几何、运动或光路参数。

麻烦在于：`q` 改一点造成的观测变化，常常也能通过改 `x` 来解释。若先在
`q=0` 重建一次 `x_hat`，随后把它当真值估 `q`，置信区间通常会过窄。v3
已经用 500 次独立噪声复本实证了这种 plug-in 欠覆盖。

## 第一层：为什么要 variable projection

### 必读 1：Golub 与 Pereyra，1973

原始论文：[*The Differentiation of Pseudo-Inverses and Nonlinear Least
Squares Problems Whose Variables Separate*](https://doi.org/10.1137/0710036)

先掌握三件事：

1. 对每个 `q`，先精确或近似求出最优场 `x*(q)`；
2. 把高维 `x` 消去，只在低维 `q` 上优化 profiled residual；
3. 求导时必须包含 `dx*(q)/dq`，不能把场冻结。

这篇论文的投影公式针对普通最小二乘。当前 BOST 使用 ridge/Tikhonov，
residual-forming matrix 通常不再是正交幂等投影，不能只把符号替换一下。

### 必读 2：O'Leary 与 Rust，2013

官方入口：[NIST publication page](https://www.nist.gov/publications/variable-projection-nonlinear-least-squares-problems)

重点不是背算法，而是拆完整 Jacobian 的两部分：直接几何响应与“最优场随
`q` 改变”的间接响应。第二项在 residual 小时可能较弱，但不能未经检验就删。
读完后应能解释：v3 的 one-step plug-in 与 iterative full-profile 差在哪里。

### 必读 3：正则化 VarPro

- Espanol 与 Pasha，2023：[*Variable Projection Methods for Separable
  Nonlinear Inverse Problems with General-Form Tikhonov Regularization*](https://doi.org/10.1088/1361-6420/acdd1b)
- Espanol 与 Jeronimo，2025：[*Local Convergence Analysis of a Variable
  Projection Method for Regularized Separable Nonlinear Inverse Problems*](https://doi.org/10.1137/24M1639087)

这里学习怎样把 Tikhonov 写成增广最小二乘，以及大规模问题中怎样用
LSQR/LSMR 近似内层解与 Jacobian。它们主要保证优化与局部收敛，不自动保证
`q` 的 95% 区间真的覆盖 95%。

## 第二层：Wald、profile contrast 与 sandwich 分别修什么

### 1. Wald

Wald 区域把点估计附近的目标函数看成椭球。它便宜，但在弱辨识、边界解、
强偏差或非线性区域最脆弱。把 covariance 乘大就能提高 coverage，所以必须
同时报告最大半轴；“覆盖了但区间比整个可行域还宽”没有科研价值。

### 2. Profile contrast

Murphy 与 van der Vaart 的 [*On Profile Likelihood*](https://doi.org/10.1080/01621459.2000.10474219)
说明，在正规 likelihood 条件下，profile likelihood 可以有二次展开和渐近
卡方性质。

当前目标含 `lambda ||x||^2`。若它只是数值正则项，profile objective 不是数据
likelihood；若把它解释成 Gaussian prior，它是 MAP，仍不等于对 `x` 积分后的
marginal likelihood。因此当前只能叫 **penalized profile contrast**，卡方阈值
只能作诊断，不能冒充严格 profile-likelihood CI。

### 3. Godambe / sandwich

当 estimating score 的曲率与随机波动不一致时，用：

```text
V = A^-1 M A^-T
```

其中 `A` 是 score sensitivity，`M` 是 score variability。它能修“标准误算错”，
不能把带偏的中心 `q_lambda` 推回真实物理 `q_true`。这正是当前最重要的区分：

```text
区间欠覆盖 = 中心偏差 + 方差估计错误 + 弱辨识/边界/非线性
```

只修最后一层 covariance，不能宣称算法已解决联合反演。

## 第三层：为什么 cross-fitting 不是随便切帧

Chernozhukov 等人的 [Double/Debiased Machine Learning](https://doi.org/10.1111/ectj.12097)
依赖两个条件：score 对 nuisance 误差一阶不敏感，以及评价样本不参与 nuisance
训练。真正的 cross-fitting 是独立样本上的角色交换。

在一段 BOST 序列中，偶数帧和奇数帧仍共享：

- 同一个初始三维场与输运历史；
- 同一套相机几何和背景图；
- 相同的标定漂移和模型误差。

所以切帧只能减少部分噪声复用，不能凭空创造独立实验。像素随机切 fold 更糟，
因为相邻 ray 由同一物理场耦合。可发表的 cross-fit 单位更可能是独立 shot、
独立 acquisition、独立 time block 或独立 rig，这需要师兄给出真实数据合同。

## 当前可复现推导

令 ridge-profile state 满足：

```text
H = B^T B + lambda I
x = H^-1 B^T y
r = y - Bx
s_k = dx/dq_k = H^-1 (B_k^T r - B^T B_k x)
d_j = B_j x
c_k = d_k + B s_k
```

profile score、精确 bread 与 observation derivative 为：

```text
u_j   = d_j^T r
A_jk  = d_j^T c_k - r^T B_j s_k
D_j,: = c_j^T
M     = sigma^2 D D^T
```

因此 v4 同时报：

```text
Gauss-Newton sandwich:  G^-1 M G^-1
exact-score Godambe:    A^-1 M A^-T
```

其中 `G=C_aug^T C_aug`。源码对 `A` 关于 `q` 的导数、`D` 关于 `y` 的导数分别
做中心差分，不靠“看起来像对的公式”。

## 7 天学习安排

### Day 1：线性最小二乘与 ridge

手推 normal equation、SVD ridge filter factor、hat matrix 与 residual-forming
matrix。用一个 `m=20,n=8` NumPy 例子验证。

### Day 2：separable nonlinear least squares

读 Golub--Pereyra 的问题定义和 projector derivative，不追全部证明。手写一个
“未知幅度 + 未知频率”的指数/正弦拟合。

### Day 3：完整 VarPro Jacobian

读 O'Leary--Rust，逐项解释 direct 与 indirect term。运行本库的随机系统差分
测试，故意删掉 `dx/dq` 看误差怎样增大。

### Day 4：反问题正则化

读 Espanol--Pasha 的增广形式。回答：penalty row 为什么不是带独立高斯噪声
的观测？为什么不能把它计入 residual degrees of freedom？

### Day 5：coverage 与偏差

模拟 500 次噪声，区分点误差、coverage、区间半径、type-I error 和 power。
构造一个有偏但方差极小的估计器，观察它为何严重欠覆盖。

### Day 6：orthogonal score 与 cross-fitting

读 DML 的定义与 sample split。列出真实 OERF 数据中可能构成独立 cluster 的
单位，并写下无法从当前 synthetic 假设得到的 covariance 信息。

### Day 7：复跑 v4 并向师兄汇报

只回答四个问题：

1. 欠覆盖主要来自中心偏差还是 covariance？
2. 区间变宽到什么程度才能恢复 coverage？
3. full-profile 第一步与继续迭代分别贡献多少？
4. 师兄能否提供独立 shot/session、真实 JVP/VJP 和标定重复？

## 真正可能形成创新的下一步

最值得研究的不是再套一层 sandwich，而是 **physical-target orthogonal score**：
显式补偿 ridge 把真实场 `x_0` 推向 pseudo-field `x_lambda` 后产生的中心偏差。
一个候选结构是：

```text
psi_q^perp = psi_q - A_qx A_xx^-1 psi_x + bias_transport_correction
```

这里前三项是联合 estimating equation 的正交化，最后一项必须来自独立 anchor、
物理输运约束或重复实验，而不能读取 synthetic truth。只有它在新 rig、signed
axes、anchor weak direction、真实 covariance 与组内 callable 上通过，才值得
进入 DeepONet/FNO 学习预条件器、warm start 或低维 correction operator 的阶段。
