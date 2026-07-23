# JACRU-M2.3：matrix-free measurement-space projection 形式化红队

**审查日期：** 2026-07-17  
**角色：** M2.3 形式化红队  
**状态：** `M2_3_FORMAL_RED_TEAM_CURRENT_POSTOPEN_NO_GO_NO_FRESH_AUTHORIZATION`  
**审查范围：** M2.2 exact dense oracle、M2.1 data-consistency 路径、对应配置、测试、runner 与证据包  
**本文件性质：** 设计约束与可证伪门；不是主算法实现，也不构成 GO

## 0. 红队判决

> **M2.2 只证明当前 12^3 离散 toy 中存在 numerical-`ker(A)` headroom。它没有证明有限步
> CG/PCG 能在公平端到端预算内到达该 headroom。M2.3 在通过本文全部硬门前，不得称
> null-space method、projector success、JACRU superiority 或可部署算法。**

当前最重要的五个风险是：

1. **端到端预算不闭合。** M2.2 取 `x_ref=CGLS-24`，现有 learned prediction 又依赖
   `CGLS-12 + 1F/1Adj`。即使共享 CGLS 前缀，投影前也至少是 `25F/25Adj`，已经超过主协议
   `24F/24Adj`；不能只给 projector 记 `(k+1)` 对调用。
2. **`lambda>0` 有确定的 row-space 偏差。** 即使线性系统被精确解出，输出一般仍满足
   `A delta_hat=lambda z != 0`，所以不是 null-space correction。
3. **有限步 CG/PCG 不是严格 projector。** 它留下由 residual polynomial 决定的可观测分量，
   CG 多项式还依赖当前右端项；有限步映射通常既不固定线性，也不幂等。
4. **当前 M2.2 没有覆盖奇异 `AA^T`。** 12 个几何均为 `A in R^(150x1000)`、rank `150`，
   所以 `A` 有至少 850 维 null space，但 `AA^T` 在这些 case 上是满秩 SPD。奇异 measurement
   modes、CG breakdown 与兼容性仍未被实验证据触及。
5. **内部 `A`-一致不等于光学不可见。** support、mask、离散 FD/trilinear inverse、hidden rays、
   独立 renderer 与真实有限孔径光学各有不同 kernel。内部 measured reprojection 过门不能替代
   independent-forward 审计。
6. **当前并发生成的 M2.3 packet 已给出 NO-GO。** `3,240` 个 candidate rows 中两种 backbone
   都没有 development-eligible 配置；这与下文 anchor feasibility 必要条件一致。

因此本轮只授权：在已经打开的数据上实现并诊断一个严格记账的 matrix-free 候选。它不授权打开
fresh/final，也不授权把 M2.2 的 `EXACT_NULLSPACE_HEADROOM_FOUND_ORACLE_ONLY` 改写成算法成功。

## 1. 已审证据与不能外推的部分

### 1.1 M2.1

`jacru_m2_data_consistency.py` 对有限步 Landweber 的命名是正确的：
`base_nullspace_filter` 每步各用一次 forward/adjoint，有限步只是 near-null spectral filter。
M2.1 在匹配 `24F/24Adj` 后仍发现，11 步 JACRU near-null 路径的
`||A delta_k|| / ||y-Ax0||` 为 development `2.282`、exploratory-OOD `3.189`，远高于下一候选
应达到的 `0.10`。

### 1.2 M2.2

`jacru_m2_exact_nullspace_oracle.py` 在 CPU float64 上组装 dense active-support matrix，以
`rank_tolerance=max(rank_atol, rank_rtol*sigma_max)` 和严格 `sigma_i>rank_tolerance` 定义数值秩。
冻结配置使用 `rank_rtol=1e-10`。它计算的是冻结离散矩阵的 truncated-SVD row/null split，
而不是连续光学真零空间。

M2.2 的 12 个 development/exploratory-OOD 几何均为：

| 项 | 已有证据 |
|---|---:|
| active matrix shape | `150 x 1000` |
| numerical rank | `150` |
| numerical nullity lower bound | `850` |
| `sigma_max` | `2.891--3.031` |
| smallest retained `sigma` | `0.303--0.331` |
| maximum internal projection residual | `2.62e-15` |

JACRU 与 pooled CNN 都通过 oracle headroom gate，二者结果仍非常接近。合法结论是“通用 learned
residual 加 affine consistency 有可研究 headroom”，不是“JACRU 集合结构已胜出”。

### 1.3 runner 账本的边界

M2.2 runner 明确把 dense assembly/SVD 标成
`DENSE_TOY_ORACLE_SETUP_NOT_RECONSTRUCTION_BUDGET`，并禁止 runtime/deployment claim。这在 oracle
证据中是诚实的，但不能被 M2.3 继承为免费 setup。

runner 把 1000 个 basis fields 按 batch 256 组装，账面是 5 次 Python-level forward invocation。
这不能成为 M2.3 的调用口径：若一个 batch 含 `q` 个独立 probe/reconstruction vectors，应计
`q` 个 sample-equivalent F，而不是把批处理并行度伪装成一次物理信息调用。

M2.2 结果行把 oracle 输出记成 24 次优化调用，是为了与 `x_ref=CGLS-24` 评分，不是完整的
M2.3 成本证明；learned prediction 的生成、dense setup 和 SVD 均不在该字段中。

### 1.4 当前 M2.3 post-open candidate 的只读检查点

审查进行中，工作区新增了 `jacru_m2_matrix_free_projection.py`、对应 config/test/runner 和结果包。
本红队没有修改这些文件，只做了只读复核。14 个现有 unit tests 通过，结果包 checksum 通过；
尚未发现独立的 M2.3 evidence validator。

实现有两点符合本文推导：

- 每步缓存 `A^T p_i` 并累计 `A^T z_k`，所以 core 账本确为 `(K+1)F + KAdj`；
- docstring/config 明确 finite CG 不是 exact optical null-space projector，且 candidate API 不接收
  truth/metric 参数。

当前结果状态为 `M2_3_POSTOPEN_MATRIX_FREE_PROJECTION_NO_GO`，包含 `3,240` 个 candidate rows 与
`540` 个 matched-baseline rows。development 上最有代表性的冲突如下：

| Backbone | 点 | visible mean / max | reprojection ratio vs matched CGLS | field gain | harm rate |
|---|---|---:|---:|---:|---:|
| JACRU | best reprojection, damped `k=4` | `0.1717 / 0.3978` | `14.7946x` | `45.88%` | `0%` |
| JACRU | damped `k=8` | `0.0738 / 0.1419` | `20.2768x` | `42.86%` | `2.78%` |
| JACRU | damped `k=12` | `0.0324 / 0.0745` | `32.5520x` | `41.35%` | `8.33%` |
| pooled CNN | best reprojection, damped `k=4` | `0.1785 / 0.4162` | `14.8033x` | `44.63%` | `0%` |
| pooled CNN | damped `k=8` | `0.0778 / 0.1515` | `20.5801x` | `41.64%` | `8.33%` |
| pooled CNN | damped `k=12` | `0.0336 / 0.0823` | `32.8889x` | `40.16%` | `8.33%` |

也就是说，finite CG 可以逐步逼近 base-anchored null target 并保留 field headroom，但没有任何点
同时接近 `1.10x` measured-reprojection 门。当前 packet 的 NO-GO 是正确判决。

该 candidate 仍触发以下形式化阻断项：

1. damped variants 的 `lambda` 使用 `_dense_norm_squared_bound`。它为每个 geometry 组装
   `150x1728` dense matrix 并做 SVD，结果包共报 12 geometry、84 次 batched Python forward，
   实际包含 20,736 个 basis-vector F-equivalents；该 setup 被排除在 paired budget 外。若 damped
   variant 被选中，端到端方法不能称纯 matrix-free，也不能作未计 setup 的效率主张。
2. runner 报 recursive system residual 与一次 `A delta_k`，但没有独立重算并验证
   `A delta_k=r_k+lambda z_k` closure，也没有 true regularized-system residual 字段。
3. `visible_fraction`、reprojection 与 correction norm 使用 `max(denominator,1e-30)`，没有
   `ratio_defined`/scaled-absolute 分支。
4. selection 允许 development visible mean `<=0.10` 但 worst `<=0.25`，OOD 只要求 mean
   `<=0.15`；这弱于本文逐 case `0.10` mechanism 门。
5. packet 没有 `eta_reference`、hidden-clean gate、row-gap gate、closure gate、code/prediction hash
   账本与 failure rows；现有 tests 也未覆盖奇异 `AA^T`、零算子、恶意 adjoint、scale mutation。
6. M2.3 把 anchor 改为 prepared CGLS-12，correction 改成 `x_net-x_base12`。因此它的 retrospective
   exact oracle 是一个新 target，不能直接写成“逼近 M2.2 的 CGLS-24-anchored oracle”。

## 2. 数学对象必须先固定

令

```text
A       : R^n -> R^m              frozen linear measurement map
A^T     : declared Euclidean adjoint of exactly that A
x_ref   : truth-free reference reconstruction
x_net   : truth-free learned proposal
delta   = x_net - x_ref
b       = A delta
B       = A A^T
```

support、active-camera mask、ray weights 与 whitening 若参与重建，必须被组合进同一个 `A`，并由
同一内积下的 `A^T` 配对。若代码实际使用加权内积，应写 `A*` 并重做推导，不能仍把普通 transpose
符号当作正确 adjoint。

以下任一情况成立时，本文 projector 推导不适用：

- forward 是非线性的、stateful 的、随机的，或调用间改变 geometry/mask；
- 所谓 adjoint 不是该次 forward 的 transpose/VJP；
- support 只在 output 端裁剪，却没有同时进入 forward 与 adjoint；
- `A` 是在 `x_ref` 处的 Jacobian，但报告把一阶 tangent null space 写成 nonlinear-forward
  null space。

若使用 Jacobian `J(x_ref)`，最多只能称 tangent-space correction，并必须另测
`F(x_ref+delta_hat)-F(x_ref)`；不能由 `J delta_hat` 小推出 nonlinear measurement 不变。

## 3. exact target、`lambda` 偏差与命名边界

### 3.1 exact finite-matrix target

设 rank-`r` SVD 为

```text
A = U_r Sigma_r V_r^T,
delta = V_r alpha + V_0 beta.
```

冻结有限矩阵的 Euclidean exact projector 是

```text
P_row(A) = A^T (A A^T)^dagger A = V_r V_r^T
P_ker(A) = I - P_row(A)
delta_0  = P_ker(A) delta = V_0 beta
x_0      = x_ref + delta_0.
```

它满足

```text
A delta_0 = 0
A x_0 = A x_ref
P_ker^T = P_ker
P_ker^2 = P_ker.
```

M2.2 实际 target 是由 `rank_rtol=1e-10` 定义的 `P_ker,tau(A)`。在病态矩阵上，algebraic
`ker(A)`、float64 numerical kernel 与该 truncated-SVD kernel 不是同一个对象。M2.3 必须记录
自己要逼近哪一个，不能在三者间移动术语。当前 12 个 M2.2 geometry 的 retained `sigma_min`
远高于 rank tolerance，所以这个区别尚未被现有结果压力测试。

### 3.2 正则化解的确定偏差

对 `lambda>0`，令

```text
(B + lambda I) z_lambda = b
delta_lambda = delta - A^T z_lambda.
```

则

```text
delta_lambda
  = V_0 beta
  + V_r diag(lambda / (sigma_i^2 + lambda)) alpha,

A delta_lambda = lambda z_lambda.
```

所以 `lambda>0` 会保留每个 row-space mode 的
`lambda/(sigma_i^2+lambda)`。它改善条件数，但不是无偏 projector。若 `sigma_min+` 是最小非零
奇异值，则

```text
||delta_lambda - P_ker delta||
  <= lambda/(sigma_min+^2 + lambda) * ||P_row delta||,

||A delta_lambda|| / ||A delta||
  <= lambda/(sigma_min+^2 + lambda)       when A delta != 0.
```

小 system residual 不能消除这项偏差。正则化候选应命名为
`Tikhonov-filtered measurement-space correction`，不得命名 `null-space projector`。

### 3.3 `lambda` 的尺度

`lambda` 与 `sigma^2` 同量纲。若 `A` 被单位变换为 `cA`，保持同一滤波器必须同时令
`lambda' = c^2 lambda`。因此跨 geometry 冻结绝对 `lambda` 通常不公平；至少应冻结

```text
alpha = lambda / L_hat,    L_hat approximately ||A||_2^2.
```

任何 per-geometry norm estimate 所调用的 F/Adj 都属于 method setup。不能使用 M2.2 dense SVD 的
`sigma_max/sigma_min` 为 opened OOD case 单独选 `lambda`，再把选择说成 matrix-free。

`lambda<0` 会破坏 PSD/SPD 保证，直接为 preflight NO-GO。

### 3.4 affine anchor feasibility 是必要条件

任何 exact/tolerance-tight null correction 都满足

```text
A x_hat = A x_ref.
```

因此，在总预算 `B` 下与 matched comparator `x_cmp,B` 比较时，以下是**与网络、CG、PCG 和
preconditioner 都无关的必要条件**：

```text
R_anchor(B)
  = ||A x_ref-y|| / ||A x_cmp,B-y||
  <= declared reprojection gate.
```

若 `R_anchor(B)>gamma`，则不存在任何 `delta_hat in ker(A)` 能让该 anchor 通过 `gamma` 门。有限步
或 `lambda>0` 可以通过改变 measurements 偶然降低 residual，但那一刻它已经不是 exact null
correction。这个 anchor-only precheck 应在训练网络或扫描 preconditioner 前执行。

当前 M2.3 用 CGLS-12 作 anchor、候选总 F 预算为 `14+k`。development 上 CGLS-12 anchor 相对
matched CGLS 的逐 case ratio 均值为：

| k | matched budget | mean `R_anchor` | min--max |
|---:|---:|---:|---:|
| 0 | 14 | `1.599` | `1.505--1.723` |
| 4 | 18 | `4.358` | `3.733--5.421` |
| 8 | 22 | `10.881` | `9.438--14.757` |
| 12 | 26 | `25.232` | `18.695--32.517` |

所以该 anchor 在所有扫描预算上都不可能让 exact null correction 通过 `1.10x` 门。当前 NO-GO
首先是 reference/budget contract 的结构矛盾，其次才是 CG 收敛速度问题。

## 4. 有限步 CG/PCG 到底保证什么

令 `B_lambda=B+lambda I`，从冻结的 `z_0=0` 开始做 `k` 个成功 Krylov updates。定义

```text
r_k       = b - B_lambda z_k                 true system residual
delta_k   = delta - A^T z_k
d_k       = A delta_k                        actual visibility defect.
```

精确算术下有最重要的 closure identity：

```text
d_k = r_k + lambda z_k.                      (I)
```

这条等式必须由独立 evaluator 重算。它直接给出两个红队结论：

1. `lambda=0` 时 system residual 就是 visibility defect；
2. `lambda>0` 时只盯 `||r_k||` 会漏掉 `lambda z_k`，即使 `r_k=0` 也通常不 null。

### 4.1 有限步谱滤波

从零开始的 CG 可写成

```text
z_k = q_(k-1)(B_lambda) b.
```

奇异方向 `sigma_i` 上，剩余 row component 的 multiplier 是

```text
1 - sigma_i^2 q_(k-1)(sigma_i^2 + lambda).
```

只有收敛到相应逆或伪逆时，它才到达上一节 target。小奇异值方向通常最慢；早停本身就是一种
依赖 spectrum 与右端项的 regularization。

CG 的系数由当前 `b=A delta` 决定，因此固定 `k` 的 CG procedure 一般不是一个对所有 `delta`
固定的线性算子。未收敛时也通常不幂等。严格措辞应是“k-step Krylov near-null filter”或
“approximate projection procedure”，不是“the projector”。

### 4.2 奇异 `AA^T`

当 `lambda=0` 且 forward/adjoint 精确配对时：

```text
range(A A^T) = range(A),
b=A delta is in range(A A^T).
```

所以即使 `B` 奇异，系统仍一致。解 `z` 可以不唯一，但任意两解之差属于 `ker(A^T)`，故
`A^T z` 唯一。零初始化 CG 在精确算术下留在有效 range 中，并可在至多 `rank(B)` 个成功步内
到达解。

这不是工程免责条款。以下情况必须 fail closed：

- `b` 因 forward/adjoint mismatch、mask mismatch 或数值污染不再与 `B` compatible；
- 未达到停止门前出现 `p^T B_lambda p <= curvature_tol`；
- residual 含无法由 `B` 消去的 measurement-null component；
- solver 用 `lambda>0` 掩盖 breakdown，却继续声称 exact null-space；
- 重启、换精度或 fallback 没有进入调用账本。

### 4.3 PCG 的附加合同

标准 PCG 需要固定、线性、对称正定的 `M`，并实际应用 `M^-1 r`。若 `M` 随 iteration/`b`
变化、含非线性网络或不是 SPD，应改称 flexible/nonlinear Krylov，并重新冻结算法与基线。

PCG 每个成功步仍只需要一次 `B_lambda p`；纯代数 preconditioner application 可记 0F/0Adj，
但构造 `M` 的 probing、norm estimate、low-rank sketch、Jacobi diagonal estimate 或 learned
geometry encoder 的 F/Adj 必须记入 method setup。随机或 batched `q` probes 是 `q` 个
sample-equivalent operator applications。

奇异 `B` 下只允许可证明 SPD 的 `M`；semidefinite、indefinite 或 data-dependent hidden shift
不满足标准 PCG 前提。

### 4.4 提前停止

主比较应冻结固定 `k`。若另报 adaptive stopping，必须同时冻结：

- `k_max`、绝对/相对阈值、residual replacement 周期、breakdown 与 fallback policy；
- 只可使用 `b`、recursive/true residual、公开 geometry 与已冻结 preconditioner；
- 禁止使用 truth、field/H1、family、hidden rays、renderer B、dense SVD gap 或 split aggregate；
- 所有用于决定继续、重启、fallback 的 true-residual recomputation 都计入 reconstruction F/Adj；
- scorer 在 prediction hash 后做的独立 recomputation 只记 evaluation calls，且不得反馈算法；
- adaptive 结果必须另列 actual-call Pareto 表，不能只和固定 24 步弱基线比较。

recursive CG residual 在有限精度下可能虚假变小。只报 recursive residual 而没有 final true
visibility defect，判 `M2_3_PROJECTOR_EVIDENCE_INVALID`。

## 5. k 步 CG/PCG 的 F/Adj 预算

为避免矩阵符号 `A` 与 adjoint-call 混淆，本文用 `F` 表示一次 full active-camera `A(v)`，用
`Adj` 表示一次 full-stack `A^T(w)`。

### 5.1 projector core

从 `z_0=0` 开始，且没有 warm start、restart、residual replacement 或 operator-based
preconditioner setup：

| 操作 | F | Adj |
|---|---:|---:|
| `b=A delta` | 1 | 0 |
| 每个 `B_lambda p=A(A^T p)+lambda p` | 1 | 1 |
| k 个成功 CG/PCG updates | k | k |
| 单独计算 final `A^T z_k` | 0 | 1 |

因此 M2.2 文档中的 literal implementation 是：

```text
projector_core_literal(k) = (k+1)F + (k+1)Adj.
```

但这不是不可降低的下界。每步已经算出 `t_i=A^T p_i`，可以随
`z_(i+1)=z_i+alpha_i p_i` 同步累计
`g_(i+1)=g_i+alpha_i t_i=A^T z_(i+1)`。冻结并测试这种 cache 后，合法最小 core 是：

```text
projector_core_cached(k) = (k+1)F + k Adj.
```

两种实现都可以，但必须预先冻结并按实际调用报告，不能在候选与基线之间选择更有利的口径。

额外成本如下：

| 条件 | 额外 reconstruction 成本 |
|---|---:|
| nonzero `z_0` 且无可信 residual cache | `+1F +1Adj` |
| 每次 true `B_lambda z` recomputation | `+1F +1Adj` |
| 每次 operator-based preconditioner probe | 按 probe 数逐个计 F/Adj |
| final evaluator-only `A delta_k` | `+1 evaluation F`，不得反馈 |
| fallback/line search/restart | 实际发生多少记多少 |

`k=0` raw-prior control 应直接返回 `x_net` 并记 projector `0F/0Adj`。若实现为了判断 `b=0`
仍执行 `A delta`，则必须记实际 `1F`；不能一边执行诊断一边把它写成零调用。

### 5.2 端到端账本

主表比较必须使用从 observation 到 final field 的调用 DAG：

```text
total = reference/base construction
      + learned feature construction
      + method-specific setup
      + projector core
      + stopping/restart/fallback calls
      - only demonstrably shared cached calls.
```

当前 source T0 的 learned path 是 `CGLS-12 + 1F/1Adj feature pair = 13F/13Adj`。M2.2 的
`x_ref` 是另一个 `CGLS-24`。于是：

| 计算方式 | projector 前 | 加 cached k-step core 后 |
|---|---:|---:|
| 完全分开计算 `x_net` 与 `x_ref` | `37F/37Adj` | `(k+38)F / (k+37)Adj` |
| 最大限度共享 CGLS-12 prefix，再继续到 CGLS-24 | 至少 `25F/25Adj` | `(k+26)F / (k+25)Adj` |
| 上一行但 final adjoint 不缓存 | 至少 `25F/25Adj` | `(k+26)F / (k+26)Adj` |

因此，**原样使用 M2.2 的 CGLS-24 reference 不可能进入 24F/24Adj 主协议。** 合法选择只有：

1. 缩短并重新冻结 `x_ref/base`，使完整 learned+projector DAG 不超过 24F/24Adj；
2. 明确做 expanded-budget diagnostic，并让 CGLS、Huber-PDHG、base-only Krylov 与 learned baselines
   获得相同二维调用 cap；
3. 停止该候选。

若 `x_ref` 同时是 learned model 的 `t`-step base，feature pair 为 `1F/1Adj`，使用 cached core，
则最低总成本是

```text
F_total   = t + 1 + (k+1) = t+k+2
Adj_total = t + 1 + k     = t+k+1.
```

要进入 24F/24Adj，必须有 `t<=22-k`。若 final adjoint 单独执行，两边都是 `t+k+2`。这项预算
必须在训练新 residual 之前冻结，因为改变 `x_ref` 也改变了 `delta` 的学习目标。

当前 M2.3 runner 选择 `t=12`，因此实际 candidate cap 为 `(14+k)F/(13+k)Adj`，并给经典方法
`(14+k)F/(14+k)Adj`。`k<=8` 尚在 24-pair cap 内，`k=12` 是 `26F/25Adj` expanded-budget
diagnostic。该改动闭合了 learned path 的成本，却不再逼近 M2.2 的 CGLS-24 anchor，并触发第
3.4 节的不可能性条件。

### 5.3 调用单位

一次 F 是一个 reconstruction sample 上全部 active views 的 forward。一次 Adj 是对应 full-stack
adjoint；masked per-view 调用同时报告 raw invocations 和 view-equivalents。batch size、GPU
parallelism 或一次 Python function invocation 不能改变信息调用数。

候选若有 `(B_F,B_Adj)` 非对称预算，基线获得同一个二维上限，并可使用该上限内最强的预注册
配置。不能强迫经典基线浪费可用调用，也不能把 candidate-only setup 归入共享 cache。

## 6. 公平基线

任何 M2.3 主表至少包含：

| 类别 | 必须比较的方法 | 公平目的 |
|---|---|---|
| strong classical | CGLS 与 Huber-PDHG，在同一 `(B_F,B_Adj)` cap 下的最强冻结版本 | 防止额外物理调用制造收益 |
| base-only continuation | 从同一 `x_ref` 用剩余预算继续 CGLS/LSQR/CGNE | 检查 projector 成本是否只等于多迭代 |
| base-only measurement CG | 解 `(AA^T+lambda I)z=y-Ax_ref`，输出 `x_ref+A^Tz` | 隔离 learned `delta` 的价值 |
| raw learned | 同一 `x_net`，`k=0` | 显示投影带来的 gain/consistency 交换 |
| learned + unpreconditioned CG | 同一 `k/lambda` | 隔离 preconditioner 贡献 |
| learned + PCG variants | Jacobi、fixed low-rank、geometry-conditioned `M` | 每个 setup 单列成本 |
| architecture control | JACRU、pooled CNN 及预注册 generic learned baselines 接受同一 projector | 防止把通用 residual headroom 归给 JACRU |
| residual controls | zero、shuffled、sign-flipped、norm-matched random `delta` 经同一 projector | 检查 morphology/data dependence |
| exact dense SVD | 只在 12^3 opened toy 上作 headroom ceiling | 禁止进入 runtime/主排名 |

所有方法必须共享 observation、公开 geometry、support/mask、precision audit、split、scorer、硬件与
prediction-first 流程。`lambda/k/tolerance/M` 的 HPO trial 数和 development 访问次数也必须匹配。

若 JACRU 在 camera-count/pose/mask OOD 上不能显著超过接受同一 projector 的 pooled CNN，算法
可以作为 generic learned-prior mechanism 继续，但必须删除 JACRU/set-encoder superiority 主张。

## 7. 泄漏边界

### 7.1 candidate process 可见

- measured `y`、公开 geometry、公开 support/mask；
- 受 counter 保护的 F/Adj API；
- `x_ref`、`x_net`、`delta` 与由它们产生的 `b`；
- 冻结的 `k/lambda/tolerance/z0/fallback`；
- 只由 train/development 学得并冻结的 preconditioner state；
- 当前 case 的 solver residual、curvature 与非标签数值故障。

### 7.2 candidate process 不可见

- evaluation truth、family/interface label、generator seed/latent；
- clean observations、hidden rays、renderer B 输出；
- dense `A`、SVD vectors、rank、`sigma_min/max` 或 M2.2 exact row/null components；
- field/H1/front/reprojection score、case ranking 或 split aggregate；
- private OOD/fresh manifest、其他 case 的 solver state 或 warm-start state；
- scorer 产生的 final true residual，若该 residual 已被分类为 evaluation-only。

### 7.3 特别禁止的泄漏路径

1. 用 M2.2 exact row basis 训练或初始化 M2.3 preconditioner，再称 matrix-free。
2. 按每个 evaluation geometry 的 dense spectrum 选择 `lambda/k`。
3. 先看 OOD/fresh field 或 hidden reprojection，再调 early-stop/fallback。
4. 从上一 evaluation case 继承 `z`、Krylov basis 或 adaptive statistics。
5. 把 candidate-only norm/preconditioner cache 标为 shared geometry setup。
6. 用 scorer 已算出的 `A x_net`、clean forward 或 hidden forward 免费构造 `b`。
7. 只输出成功收敛 rows，把 breakdown/fallback rows 当 missing data 删除。

当前 M2.2 的 development 与 exploratory-OOD 都已打开，且 exact spectrum、逐行结果与 aggregate
已可见。任何据此选出的 M2.3 配置在这些数据上只能称 opened development diagnostic。若未来要
confirmatory 结论，必须冻结新 candidate hash、使用全新 OOD/fresh manifests，并遵守已有
one-open transaction；不能把当前 exploratory-OOD 重新密封。

## 8. 可观测失败模式

| 失败模式 | 可观测信号 | 红队判决 |
|---|---|---|
| `lambda` bias 被漏报 | `||r_k||` 小但 `||r_k+lambda z_k||` 大 | projector NO-GO；禁止 null-space 措辞 |
| finite-k 未收敛 | `eta_visible`、oracle row gap 或 repeated-application drift 大 | mechanism NO-GO |
| recursive residual 欺骗 | recursive `r` 小，独立 true residual 大 | evidence invalid |
| singular compatible system breakdown | 未过门前 `p^TBp` 近零、NaN 或 stagnation | fail closed，不得删 row |
| incompatible `b` | true residual 有不可消 component | preflight invalid，查 adjoint/mask |
| 非 SPD preconditioner | symmetry/Rayleigh test 失败或 negative curvature | PCG 名称与结果无效 |
| hidden setup cost | F/Adj counter 与 ledger/hash 不闭合 | budget NO-GO |
| batched probes 少记 | batch width `q>1` 仍只记一次 F/Adj | budget fraud，packet invalid |
| support drift | inactive voxel 非零，或 `A`/`A^T` 用不同 mask | preflight invalid |
| nonlinear/stateful forward | zero/linearity/determinism test 失败 | 不能使用本 projector 推导 |
| adjoint mismatch | dot-product defect 超门 | CG/PCG 数学无效 |
| mixed-precision loss | closure、orthogonality、true residual 随 dtype 反转 | numerical NO-GO |
| near-zero denominator | ratio 因 clamp 看似通过 | ratio 标记 undefined，改用 scaled absolute gate |
| accidental data-fit cancellation | measured reprojection 改善但 `A(x-x_ref)` 大 | 不是 null correction |
| weak-mode hiding | measurement residual小而 field oracle gap 大 | 报 `1/sigma_min+` 放大；mechanism 不成立 |
| all-fallback success | 大量返回 `x_ref`，reprojection 自动好看 | coverage/harm gate NO-GO |
| approximate-kernel overclaim | inverse `A` 通过，renderer B/hidden rays 失败 | independent-forward NO-GO |
| architecture overclaim | JACRU 与 pooled CNN 等效 | 删除 JACRU superiority |

对 `lambda=0`，若 `e_k=delta_k-P_ker delta` 位于 row space，则

```text
e_k = A^dagger r_k,
||e_k|| <= ||r_k|| / sigma_min+.
```

这说明在病态 geometry 上，极小 measurement residual 仍可对应很大的 field-space projector gap。
因此 `measured reprojection`、`eta_visible` 和 toy 上的 exact-oracle field gap必须分开报告。

## 9. 可机器验证的不变量

每个 case 至少持久化：

```text
case_id, geometry_sha256, operator_sha256, support_sha256,
method, model_seed, k_planned, k_completed, lambda, z0_policy,
preconditioner_id, preconditioner_sha256,
setup_forward_equivalents, setup_adjoint_equivalents,
reconstruction_forward_equivalents, reconstruction_adjoint_equivalents,
evaluation_forward_calls,
norm_delta, norm_b, norm_z, norm_recursive_r, norm_true_r,
norm_visibility_defect, closure_defect,
eta_visible, eta_reference, eta_scaled,
curvature_minimum, breakdown_code, fallback_code,
prediction_sha256
```

分母低于预冻结 floor 时，ratio 字段必须为 `null` 并设置 `ratio_defined=false`；不能仅用 `1e-30`
clamp 后把不稳定比值送入 pass/fail。

以下 invariant 可直接写成 pytest/validator：

1. **Adjoint identity**

   ```text
   |<Av,w>-<v,A^T w>| / max(||Av||||w||, ||v||||A^T w||, eps)
   <= 1e-10 float64; <= 1e-5 float32.
   ```

2. **Zero, linearity, determinism**

   ```text
   A0=0;
   A(alpha u+beta v)=alpha Au+beta Av;
   repeated identical calls agree within frozen dtype tolerance.
   ```

3. **Support identity**

   ```text
   delta = support*delta;
   delta_k = support*delta_k;
   inactive output is exactly zero;
   forward/adjoint both contain the same support operator.
   ```

4. **Call identity** for zero-start, fixed `k`, no hidden recomputation

   ```text
   cached:  F_core=k+1, Adj_core=k
   literal: F_core=k+1, Adj_core=k+1
   ```

   Counter event sequence必须与 `F(delta)`、k 次 `Adj(p_i)`/`F(A^T p_i)`、可选 final
   `Adj(z_k)` 一一对应。

5. **Output identity**

   ```text
   delta_k == delta - A^T z_k
   x_k == x_ref + delta_k
   ```

6. **Residual closure**

   ```text
   d_k=A delta_k
   r_k=b-(AA^T+lambda I)z_k
   ||d_k-r_k-lambda z_k|| / scale <= closure_tol.
   ```

   final float64 toy audit 取 `closure_tol<=1e-10`；experiment packet 最宽不得超过 `1e-8`。

7. **Exact lambda identity**，对 dense tiny matrix 解

   ```text
   A delta_lambda == lambda z_lambda.
   ```

8. **Kernel preservation**，对已知 `delta in ker(A)`

   ```text
   b=0, z=0, delta_k=delta
   ```

   允许 short-circuit，但调用必须按实际报告。

9. **Row removal**，对 full-column-rank `A`

   ```text
   lambda=0 and converged => delta_k=0.
   ```

10. **Singular solution invariance**

    ```text
    q in ker(A^T) => A^T(z+q)=A^Tz.
    ```

11. **Scaling invariance**

    ```text
    output(A,lambda) == output(cA,c^2 lambda).
    ```

12. **Dense-to-matrix-free agreement**，只在 opened tiny/toy evaluator 中

    ```text
    lambda=0, converged => delta_k approximately P_ker,tau(A) delta
    null-component drift approximately 0
    row-removal gap approximately 0.
    ```

13. **Idempotence boundary**

    ```text
    exact P_ker(P_ker delta)=P_ker delta.
    ```

    `lambda>0` 或 finite-k procedure 若不幂等，测试应确认“不得标 exact projector”，而不是把
    非幂等失败隐藏。

14. **No-label sentinel**

    candidate process 对 truth/family/clean/hidden/SVD keys 的任何访问都抛异常并终止。

15. **Failure completeness**

    planned Cartesian product 中每个 case/method/seed 恰有一行；breakdown 与 fallback 也是结果，
    不得缺失。

## 10. 最小测试矩阵

所有 expected-value 测试先用 float64，`atol=rtol=1e-12`。再做 float32 宽容差镜像，但正式
mechanism audit 保留 float64 true residual。

### T1：已知 row/null split 与有限步残留

```text
A1 = [[1,0,0],
      [0,2,0]]
delta = [2,-3,4]^T
b = [2,-6]^T
B = diag(1,4)
```

`lambda=0` 的 exact 结果：

```text
z_dagger = [2,-3/2]^T
delta_exact = [0,0,4]^T.
```

零初始化普通 CG 只做 1 步时：

```text
z_1 = [20/37,-60/37]^T
delta_1 = [54/37,9/37,4]^T
A1 delta_1 = [54/37,18/37]^T != 0.
```

第 2 步才应到 exact。该矩阵同时验证 k-step 不能提前叫 null-space。

取 `lambda=1` 并精确解时：

```text
z_lambda = [1,-6/5]^T
delta_lambda = [1,-3/5,4]^T
A1 delta_lambda = [1,-6/5]^T = lambda z_lambda.
```

### T2：奇异 `AA^T`、一致 RHS 与非唯一 z

```text
A2 = [[1,0],
      [1,0]]
delta = [3,4]^T
b = [3,3]^T
B = [[1,1],
     [1,1]].
```

`B` rank 1。零初始化 CG 应在 1 步得到 minimum-norm
`z=[3/2,3/2]^T`，输出 `[0,4]^T`。对任意
`q=t[1,-1]^T in ker(A2^T)`，`A2^T(z+q)=A2^Tz`。该测试禁止把“B 奇异”等同于“无解”。

### T3：零算子与零 RHS

```text
A3 = zeros(2,3)
delta = [1,-2,5]^T.
```

对任意 `lambda>=0` 都应返回 `z=0, delta_k=delta`，不能除零、产生 NaN 或伪造一次成功 CG
update。ratio 应按 denominator policy 标记 undefined。

### T4：弱可观测方向与 `lambda` bias

```text
A4 = diag(1, 1e-6, 0)
delta = [1,1,1]^T
lambda = 1e-8.
```

精确 regularized 输出为

```text
delta_lambda = [lambda/(1+lambda),
                lambda/(1e-12+lambda),
                1]
             approximately [1e-8, 0.99990001, 1].
```

第二个方向虽然不是 algebraic null，却几乎完整保留。该测试验证“conditioning 变好”不能被翻译成
“弱方向已投到 null space”。

### T5：没有 null space

```text
A5 = I2
delta = [1,-2]^T.
```

`lambda=0`、收敛后输出必须为零；`lambda>0` 的精确输出必须为
`lambda/(1+lambda)*delta`。若此 case 仍报告非零“null correction”，名称或实现错误。

### T6：恶意 adjoint

沿用 `A=[1,0]`，但故意提供 `A_bad^T(w)=[-w,0]^T`。dot-product test 必须先失败；若仍进入
CG，会出现负 curvature。该测试确保 runner 不是只在理想 matrix helper 上正确。

对 T1--T5 再做 mutation：`A'=7A, lambda'=49lambda`，输出应与原问题一致；只缩放 `A` 而不缩放
`lambda` 时结果必须不同，从而证明实现没有偷偷把绝对 `lambda` 当无量纲量。

## 11. 严格 go/no-go 门

所有门按顺序执行，任一失败立即停止后续 claim。通过较晚的 accuracy 指标不能覆盖较早的数学、
泄漏或预算失败。

### G0：形式与 API preflight

必须全部满足：

| Check | 门槛 |
|---|---|
| matrix-free boundary | candidate 进程只能访问 F/Adj API；不能 import dense oracle/SVD/truth |
| linearity/zero/determinism | 第 9 节 invariant 全通过 |
| adjoint | float64 defect `<=1e-10`，float32 `<=1e-5` |
| support/mask | forward、adjoint、delta、output 使用同一冻结 support |
| PCG validity | `M` 固定、线性、SPD；否则换算法名并重新冻结 |
| tiny matrices | T1--T6 与 scaling mutations 全通过 |
| counter | 第 25 次 F 或 Adj 在 24-cap protocol 中立即拒绝 |
| sentinel | truth/family/hidden/clean/SVD 访问 fail closed |

失败状态：`M2_3_PREFLIGHT_INVALID`。不得生成方法排名。

### G1：预算与运行完整性

必须全部满足：

- 完整 observation-to-field DAG 在冻结 `(B_F,B_Adj)` 内；
- 当前 M2.2 `CGLS-24 reference + existing prior` 不得冒充 24-cap candidate；
- 对任何 null-space claim，anchor-only `R_anchor` 必须先达到同预算 reprojection 门；
- setup、probe、warm start、true-residual stop、restart 与 fallback 均入账；
- sample/view-equivalent 记账，不按 batch invocation 少记；
- 所有强基线在同一二维 cap 下可运行；
- planned rows 100% 存在，NaN/breakdown/fallback 不删除；
- config、operator、preconditioner、prediction 与 ledger 均有 hash。

anchor 不可行时输出 `M2_3_ANCHOR_INFEASIBLE_NO_GO`；其他失败输出
`M2_3_BUDGET_OR_COMPLETENESS_NO_GO`。

### G2：projection mechanism gate

令

```text
eta_visible   = ||A delta_k|| / ||A delta||,
eta_reference = ||A delta_k|| / ||y-A x_ref||,
eta_scaled    = ||A delta_k|| / (L_hat^0.5 ||delta||).
```

分母低于冻结 floor 时不用相对 ratio，改判 `eta_scaled`。必须全部满足：

| Check | 门槛 |
|---|---|
| true closure | 每 case `closure_defect<=1e-8`；tiny float64 `<=1e-10` |
| visible removal | 所有 ratio-defined case 的 `eta_visible<=0.10` |
| affine preservation | 所有 ratio-defined case 的 `eta_reference<=0.10` |
| near-zero cases | `eta_scaled<=0.10`，且无 denominator-clamp 假通过 |
| dense opened-toy row gap | `||delta_k-P_ker,tau delta||/||P_row,tau delta||<=0.10` |
| dense opened-toy null drift | `||P_ker,tau delta_k-P_ker,tau delta||/||delta||<=1e-8` |
| robustness | 无未处理 breakdown、negative curvature、NaN 或 residual-replacement 漏账 |

`0.10` 是进入 empirical comparison 的 near-null feasibility 门，不是 exact-null 命名阈值。
`lambda>0` 即使通过本门，也只能称 filtered/near-null。

失败状态：`M2_3_MATRIX_FREE_PROJECTION_NO_GO`。

### G3：公平 development gate

在同预算最强 classic、base-only 与 learned baseline 中取逐 case/聚合最强者。候选必须同时满足：

| 维度 | Development | 已打开 exploratory-OOD diagnostic |
|---|---:|---:|
| field gain vs strongest matched baseline | `>=5%` 且 paired CI lower `>0` | `>=2%` 且不反转 |
| H1 gain | `>=3%` 且 CI lower `>0` | `>0` |
| exact-oracle gain retention | field 与 H1 均 `>=50%` | field 与 H1 均 `>=50%` |
| measured reprojection vs matched CGLS | `<=1.10x` | `<=1.15x` |
| hidden clean reprojection vs strongest baseline | `<=1.05x` | `<=1.05x` |
| tail | harm-1% `<=5%`，worst gain `>=-5%`，severe harm `0` | 同左 |
| seeds | 当前 3/3 seed mean field gain `>0` | 当前 3/3 |
| residual dependence | full 比 zero/shuffled/random 至少好 `3%`；controls 保留 full gain 不得 `>25%` | 不反转 |

exact SVD 不参与 runtime 或“最强 baseline”排名，只提供 gain ceiling。任何一项失败输出
`M2_3_DEVELOPMENT_NO_GO`；不能用“接近 oracle”覆盖同预算基线失败。

### G4：generalization 与 independent-forward gate

必须全部满足：

- camera-count、pose、mask、noise/bias 与 morphology strata 没有系统性 field/H1 反转；
- renderer B/hidden-ray clean reprojection `<=1.05x` strongest matched baseline；
- prediction 在 evaluator 解锁 truth/hidden rays 前已完整生成并 hash；
- approximate `A` 的 `eta_visible` 通过，但 renderer B 失败时明确判 mismatch NO-GO；
- JACRU 若不胜 pooled CNN，删除 set-geometry superiority，而不是删除 CNN row。

当前 OOD 已打开，所以即使全部通过，状态也只能是
`M2_3_OPENED_MATRIX_FREE_MECHANISM_SIGNAL`。G4 失败输出
`M2_3_INDEPENDENT_FORWARD_OR_OOD_NO_GO`。

### G5：fresh 仍保持关闭

只有 G0--G4 在 opened development 上通过、candidate/config/ledger/validator 已冻结、全新 OOD/fresh
manifest 已建立，并重新满足原 M2 preregistration 的 prediction-first one-open transaction，才可另行
申请 fresh。本文自身永远不把 `open_fresh_or_final` 置为 true。

## 12. 何时绝对不能称 null-space

以下任一项为真，论文、网页、图例和 JSON status 中都禁止出现无修饰的 `null-space projector`：

1. `lambda>0`；
2. `k` 有限且没有逐 case true-defect certificate；
3. 只报 system residual，未报 `r_k+lambda z_k`；
4. forward/adjoint、support、mask 或 inner product 未通过配对测试；
5. 使用 nonlinear forward 的单点 Jacobian；
6. finite-k procedure 未验证幂等且 CG polynomial 依赖 `delta`；
7. numerical rank threshold 未声明；
8. 只对 approximate inverse `A` 不可见，independent renderer/hidden rays 可见；
9. projector 依赖 dense SVD、truth、family 或 evaluation spectrum；
10. 调用预算、breakdown rows 或 fallback coverage 不完整。

推荐术语梯度：

| 条件 | 允许术语 |
|---|---|
| 任意有限 k | `matrix-free measurement-space correction` |
| finite k，G2 通过 | `k-step Krylov near-null filter` |
| `lambda>0` | `Tikhonov-filtered correction` |
| `lambda=0` 且每 case 达冻结 defect tolerance | `tolerance-certified numerical A-null correction` |
| finite dense matrix 上伪逆/SVD、残差与幂等达机器精度 | `exact projector for the frozen discrete A` |

最后一行也不能缩写为 true optical null space。

## 13. M2.3 冻结前最小产物

主算法尚未实现前，下一冻结包至少需要：

```text
configs/jacru_m2_3_*.json
budget_contract.json
operator_api_manifest.json
preconditioner_manifest.json
solver_event_ledger.csv
metric_rows.csv
aggregate_rows.csv
failure_rows.csv
selection_commit.json
report.json
checksums.sha256
```

validator 必须自行重算：调用数、closure、所有 ratio/floor、dense opened-toy gap、matched baseline
选择、gain retention、harm/worst/seed gates、failure completeness 与 authorization booleans。只校验文件
hash 而不重算公式不够。

## 14. 最终建议

M2.2 的 headroom 值得继续，但 M2.3 的第一实现决策不应是“选哪种 PCG”，而应先冻结一个真正
可闭合的端到端预算：要么将 `x_ref` 缩短到能容纳 feature pair 与 k-step projector，要么承认这是
expanded-budget study，并把同等预算交给强基线。

当前 post-open run 已证明，单纯把 anchor 缩到 CGLS-12 并不可行：越接近 exact null target，
measurement 越回到弱 anchor，而 matched CGLS 随总预算继续改善。下一候选在任何训练前必须先过
`R_anchor` necessary-condition test；过不了就应停止该 affine-anchor 设计，而不是继续扫描更大的
`k`、更小的 `lambda` 或更复杂的 preconditioner。

其后，`lambda=0` 的 compatible singular-CG 路径与 `lambda>0` 的稳定 filtered path 应作为两个
不同 hypothesis，不得混成一条成功曲线。只有前者在 true visibility、toy oracle gap、公平 field/H1、
tail、independent renderer 和完整账本上同时过门，才有资格写“matrix-free numerical A-null
correction”。即便如此，现有 opened 数据仍只产生 mechanism signal，不产生 fresh/final 结论。
