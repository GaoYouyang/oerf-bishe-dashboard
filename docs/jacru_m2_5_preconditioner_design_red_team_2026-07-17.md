# JACRU-M2.5：预条件器数学、预算与创新边界红队

- **审查日期：** 2026-07-17
- **角色：** M2.5 预条件器数学/预算红队
- **状态：** `M2_5_DESIGN_RED_TEAM_DIAGONAL_AND_CAMERA_BLOCK_ORACLES_NO_GO_TARGET_PARETO_NEXT_NO_FRESH`
- **证据范围：** M2.3/M2.4 results、M2.5 exact-Jacobi oracle、M2.6 camera-block oracle、M2.3 formal red team、matrix-free solver、M2.3 primary-literature novelty audit
- **文件性质：** 设计约束、现有 preconditioner-oracle 判决、候选顺序与 stop rules；不是新实现或 fresh 授权

## 0. 红队判决

> **M2.5 只能作为 M2.4 `Ax=y` 仿射观测校正的有限预算加速研究，不能用来挽救 M2.3 的
> `Ax=Ax_ref` 弱 anchor。固定 SPD 预条件器改变 Krylov 收敛速度，不改变收敛极限。**

直接结论如下。

1. **预条件器研究有明确靶点。** M2.4 的 exact dense affine oracle 能把 measured residual 压到
   约 `6.0e-16`，说明仿射目标本身在当前 toy 上可达；identity CG 在 `k=8` 时只把 development
   mean visible fraction 压到 `0.0988`，但 measured reprojection 仍为同预算 CGLS 的
   `20.89x`。当前失败主要是有限预算谱收敛，不是仿射可行集不存在。
2. **但更快到达 `Ax=y` 不等于更好重建。** M2.4 exact affine oracle 的 development clean
   reprojection mean 仍为 `0.0206`，而 measured residual 接近机器零；噪声、camera bias 与 forward
   mismatch 可被更快拟合。任何预条件器必须同时过 field/H1、clean/hidden renderer 和 no-harm 门。
3. **exact Jacobi 值得做的那一次已经完成，并正式 NO-GO。** checksum-valid packet 状态为
   `M2_5_EXACT_JACOBI_PRECONDITIONER_ORACLE_NO_GO`；JACRU 与 pooled CNN 均无 development
   selection。它只能保留为 diagonal-family negative control，不得继续调 `q`、floor 或 damping 来救。
4. **exact Jacobi 后的正确下一步 camera-block oracle 也已完成。** 它把 `k=8` JACRU reprojection
   从 identity 的 `20.89x` 降到 `1.263x`，证明 camera-local coupling 是有效机制；但 24-call 内仍未过
   `1.10x`，且 harm/worst 失败。`k=12` 虽过 reprojection，却是 expanded budget，仍未过 no-harm。
5. **现在不能直接进入 learned SPD 或 Nyström。** 下一步先做 target/no-harm Pareto feasibility：若更快
   接近 exact `Ax=y` 必然造成 tail harm，任何更强预条件器都只会更快到达同一个失败极限。
   Nyström/low-rank 只可在 Pareto ceiling 证明仍有联合可行点后，用来补 camera blocks 间的全局耦合。
6. **非线性 learned preconditioner 必须进入 FCG。** 把随 residual/iteration 变化的网络塞入普通
   PCG 后仍报告 PCG 收敛，判数学无效。FCG 是最后一层候选，不是绕过 SPD/fixedness 门的捷径。
7. **当前不授权 fresh/final。** 所有现有 M2 数据均已打开；M2.5 最多先产生 opened-mechanism
   evidence。只有固定结构、完整预算和独立 validator 在新 manifest 上预注册后，才可另行申请 fresh。

## 1. 继承证据与问题定位

### 1.1 M2.3 不能由预条件器修复

M2.3 解

```text
b = A(x_net-x_ref),
(AA^T + lambda I)z = b,
x_out = x_ref + (x_net-x_ref) - A^T z.
```

当 `lambda=0` 且求到收敛时，必有 `Ax_out=Ax_ref`。M2.3 formal red team 已证明 CGLS-12 anchor
相对 matched CGLS 的 `R_anchor` 随总预算从 `1.599x` 上升到 `25.232x`。无论 `M` 是 identity、
Jacobi、low-rank 还是 learned SPD，都不能改变 exact limit。若有限步偶然改善 `Ax-y`，那是没有到达
原目标，不是预条件器解决了 anchor infeasibility。

因此 M2.5 禁止继续使用 `reference_reprojection` 作为主 target。

### 1.2 M2.4 提供了真正的加速问题

M2.4 改为

```text
u = x_net,
b = Au-y,
H_lambda = AA^T + lambda I,
H_lambda z = b,
x_k = u-A^T z_k.
```

对 `lambda=0`、full-row-rank `A` 和收敛 solve，`x_*` 是 `u` 到 `{x:Ax=y}` 的 Euclidean
orthogonal projection。M2.4 identity-CG 的 development 结果为：

| `k` | JACRU visible mean / max | reprojection vs matched CGLS | field gain | harm rate |
|---:|---:|---:|---:|---:|
| 1 | `0.4476 / 0.5603` | `19.99x` | `46.73%` | `0%` |
| 2 | `0.3090 / 0.5980` | `16.40x` | `47.34%` | `0%` |
| 4 | `0.2054 / 0.5143` | `15.96x` | `45.26%` | `0%` |
| 8 | `0.0988 / 0.1878` | `20.89x` | `41.36%` | `8.33%` |
| 12 | `0.0422 / 0.0962` | `19.66x` | `39.54%` | `8.33%` |
| 32 | `2.87e-4 / 8.79e-4` | `33.38x` | `35.22%` | `8.33%` |

pooled CNN 呈同一趋势。没有 candidate development-eligible，packet 正确判为
`M2_4_POSTOPEN_AFFINE_CG_NO_GO`。这说明：

- exact affine target 有 field headroom；
- identity CG 会收敛，但在 learned preparation 已花 `13F/13Adj` 后太慢；
- matched CGLS 随总预算继续降低 residual，不能拿固定弱基线作对照；
- 更大的 `k` 同时增加 harm，不能把 residual speedup 当作最终成功。

### 1.3 本轮只读谱诊断

为判断 exact Jacobi 是否值得独立开发，本红队使用冻结 M2.4 operators 在内存中重建 12 个
`A in R^(150x1000)` active-support matrices，并只计算 evaluator-side spectra；未写结果包、未改变
任何 candidate 输入。当前 measurement shape 是 `[75,2]`，对应 3 个 camera blocks，每块
`25 rays x 2 components = 50` 个 measurement coordinates。

| 谱量 | minimum | mean | maximum |
|---|---:|---:|---:|
| `kappa(AA^T)` | `81.59` | `87.34` | `91.53` |
| `max(diag)/min(diag)` | `2.92` | `3.51` | `4.16` |
| exact-Jacobi `kappa(D^-1/2 AA^T D^-1/2)` | `89.45` | `93.14` | `102.40` |
| identity/Jacobi condition-number factor | `0.865` | `0.939` | `1.015` |
| exact camera-block condition number | `13.18` | `14.33` | `15.07` |
| identity/block condition-number factor | `5.70` | `6.10` | `6.83` |

解释边界：

- exact Jacobi 在这些 12 个 geometry 上平均使谱条件数恶化，不支持 diagonal scaling 是主解法；
- camera-block oracle 显示病态性主要含明显的 camera-local correlation，值得进入下一 headroom gate；
- 条件数不是特定 RHS 的有限步 CG 排名定理；随后出现的 exact-Jacobi packet 已完成该 empirical control；
- 这些数字是本设计审查的只读诊断，不是 checksum-sealed M2.5 evidence。若进入论文或主表，必须由
  独立脚本、CSV/JSON、hash 和 validator 重新生成。

### 1.4 Exact-Jacobi empirical control 已封口

审查过程中工作区并发生成了冻结配置与 checksum-valid、自判决结果包
`jacru_m2_5_exact_jacobi_preconditioner_oracle_postopen_public`。本红队未修改该 packet。其正式状态为：

```text
status = M2_5_EXACT_JACOBI_PRECONDITIONER_ORACLE_NO_GO
jacru_m2 selection = null
pooled_cnn selection = null
continue_deployable_preconditioner_estimation = false
open_fresh_or_final = false
```

JACRU 的同 `k` development 对照为：

| `k` | identity reprojection ratio | exact-Jacobi ratio | identity visible mean | exact-Jacobi visible mean |
|---:|---:|---:|---:|---:|
| 2 | `16.40x` | `15.82x` | `0.3090` | `0.3000` |
| 4 | `15.96x` | `15.19x` | `0.2054` | `0.1942` |
| 8 | `20.89x` | `17.66x` | `0.0988` | `0.0867` |
| 12 | `19.66x` | `15.19x` | `0.0422` | `0.0332` |

这是真实但不足的 RHS-specific 改善：离 `1.10x` matched-CGLS 门仍有数量级差距，`k=8/12`
的 harm rate 仍为 `8.33%`。`k=12` 还是 `26F/25Adj` expanded-budget point，不属于 24-call 主线。
exact diagonal 由 dense active matrix 构造，每 case 明报 `1001F-equiv/0Adj` setup，并排除在 matched
budget 与效率主张外。因此该 packet 既没有 deployable headroom，也没有 runtime evidence；它正确地
关闭了 Hutchinson `diag(AA^T)` 路线，同时保留继续研究其他 matrix-free preconditioner 结构的空间。

证据限定：本轮核验了 packet 内 `checksums.sha256`，但当前独立 semantic validator
`validate_jacru_m2_3_m2_4_evidence.py` 不覆盖 M2.5。故上述 NO-GO 可作为保守决策使用，不能称
“M2.5 已独立验证”。下一 oracle packet 前必须新增 validator，重算 selection、setup equivalents、
closure、matched baseline、authorization 和 planned-row completeness，而不只校验 hashes。

### 1.5 Camera-block oracle：机制成立，联合门失败

随后出现的 checksum-valid、自判决 M2.6 packet 正式状态为
`M2_6_CAMERA_BLOCK_PRECONDITIONER_ORACLE_NO_GO`。两种 backbone 都在 development 选择 exact
camera-block、`k=12`，但最终 checks 均因 development harm rate 与 worst case 失败。

| Backbone | `k` | reprojection ratio | visible mean / max | field gain | harm | worst gain |
|---|---:|---:|---:|---:|---:|---:|
| JACRU | 8 | `1.263x` | `0.00624 / 0.01398` | `40.14%` | `8.33%` | `-8.75%` |
| JACRU | 12 | `0.270x` | `5.58e-4 / 1.06e-3` | `39.01%` | `8.33%` | `-9.31%` |
| pooled CNN | 8 | `1.343x` | `0.00672 / 0.01345` | `39.45%` | `8.33%` | `-11.78%` |
| pooled CNN | 12 | `0.288x` | `6.01e-4 / 1.08e-3` | `38.33%` | `8.33%` | `-12.31%` |

每个表中点均无 breakdown，closure maximum 约 `1e-15`，所以这不是 solver correctness 失败。
真正阻断项是：

1. `k=8` 位于 `22F/21Adj` 主 cap 内，但 reprojection 仍高于 `1.10x`，harm/worst 也失败；
2. `k=12` 为 `26F/25Adj` expanded-budget point，不可作为 24-call 主结果；
3. `k=12` 即使逼近 exact affine target，tail harm 仍不消失，说明更快收敛不能自动修复 target risk；
4. exact block setup仍是 dense `1001F-equiv/0Adj` oracle，并被排除在 matched budget 外；
5. packet authorization 明确令 `continue_deployable_preconditioner_estimation=false`、fresh=false。

因此 camera-block 给出的是 **spectral/mechanism headroom found, reconstruction gate NO-GO**。与 M2.5
一样，本轮只核验 packet checksums；当前没有覆盖 M2.6 的独立 semantic validator。

## 2. 数学合同与统一预算

### 2.1 预条件器对象

记

```text
H = AA^T + lambda I,
M approximately H,
P = M^-1,
z_pre = P r.
```

普通 PCG 的每个 geometry/case 必须满足：

```text
M is fixed during one solve;
M is linear, symmetric, and positive definite;
P is applied deterministically under the declared precision;
the forward, adjoint, mask, support, lambda, and inner product do not drift.
```

`M` 可以由 geometry 生成，也可以在 solve 前依赖公开 operator descriptors；但生成后必须固定。
为了让“geometry-conditioned”可归因且可复用，主候选不得使用 `y`、`b` 或当前 residual 来生成 `M`。
若预条件 action 随 residual 或 iteration 改变，必须改用明确冻结的 FCG/nonlinear Krylov 算法。

`lambda>0` 使 `H` SPD，但输出满足

```text
Ax_k-y = r_k + lambda z_k,
r_k = b-(AA^T+lambda I)z_k.
```

所以 system residual 小不等于 data residual 小；`lambda>0` 仍只能称 damped/filtered correction。
M2.5 第一主假设应使用 `lambda=0`，避免把 dense norm setup 和 damping bias 混进预条件器判断。

### 2.2 F/Adj-equivalent 口径

沿用 M2.3 formal red team：

- `1F`：一个 reconstruction sample 上全部 active views 的 `A(v)`；
- `1Adj`：对应 full measurement stack 的 `A^T(w)`；
- batch 中有 `q` 个独立 probe vectors，计 `q` 个 equivalents，不计一次 Python invocation；
- preconditioner setup、norm estimate、sketch、restart 和 residual replacement 均不能免费；
- evaluator 在 prediction hash 后做的 dense spectrum/true residual 可以单列 evaluation calls，但不得反馈算法。

M2.4 learned preparation 已是 `13F/13Adj`。缓存 `A^T p_i` 的 `k` 步 PCG/FCG core 为

```text
core(k)  = (k+1)F + kAdj,
total(k) = (14+k+s_F)F + (13+k+s_Adj)Adj,
```

其中 `(s_F,s_Adj)` 是当前 geometry 的冷启动 setup。若主 cap 是 `24F/24Adj`：

```text
k+s_F   <= 10,
k+s_Adj <= 11.
```

这条不等式是 M2.5 的硬约束。`q=8` 的 online sketch 只剩最多 `k=2`；不能把 sketch 从账本删掉后
再报告 8 步 PCG。

同一 geometry 上有 `N_reuse` 个独立 observations 时，可另报 amortized setup：

```text
amortized setup = (s_F/N_reuse, s_Adj/N_reuse).
```

但主表必须同时报告 cold-start latency、实际整数调用和 cache provenance。只有协议事先保证 geometry
cache 可跨 observations 复用时，amortized 表才有部署意义；当前 toy 中跨 model seed/family 重用同一
operator 不能自动等价于真实部署中的免费 cache。

## 3. 候选逐项比较

下表的“每步”只列 preconditioner-local algebra；所有 PCG/FCG candidate 另有共同的
`1F+1Adj` normal-operator product。

| 候选 | 冷启动 setup F/Adj-equiv | preconditioner-local 每步 | SPD/固定性 | 跨 geometry | 主要泄漏/失真风险 | 红队顺序 |
|---|---:|---:|---|---|---|---:|
| identity | `0 / 0` | `O(m)` copy | 固定 SPD | 完全可用 | 无结构泄漏；只是慢 | 0，永久基线 |
| exact Jacobi oracle | 最省 black-box 为 `0 / mAdj`；当前 dense path 为 `(n_a+1)F / 0` | `O(m)` divide | positive diagonal；solve 内固定 | 每 geometry 重算；冷启动不可扩展 | dense/operator-spectrum tuning；batch 少记 | 1，仅一次 control |
| exact camera-block Jacobi oracle | `0 / mAdj`；或 `(n_a+1)F / 0`；仅有 `H` API 时 `mF/mAdj` | `O(sum m_c^2)` triangular solves | 每块 SPD 或冻结 jitter；固定 | camera partition 天然支持 variable count | 用 opened spectra 选 block/jitter；跨 camera 顺序错误 | 2，下一 headroom |
| Hutchinson diagonal | `qF / qAdj` | `O(m)` divide | 估计值须正值化并固定 probes | matrix-free；每 geometry 付 `q` | negative/noisy entries、seed/HPO、setup 挤占 `k` | 仅 exact diagonal 通过后 |
| Nyström/low-rank | one-pass `qF/qAdj`；`s` 次 power iteration 为 `(s+1)q` pairs | `O(mq+q^2)` Woodbury | 必须有 positive base shift/diagonal；固定 sketch | 可变 `m` 可用，storage `O(mq)` | rank/seed 看 OOD spectrum；近奇异 Woodbury | block 后补全局 mode |
| geometry-conditioned learned SPD | 无 online probes 时 `0/0`；否则实际 `q/q` | diagonal `O(m)`；block/low-rank `O(sum m_c^2+mq)` | 显式 SPD 参数化；每 solve factor hash 固定 | 需 permutation-equivariant、variable-cardinality | train/eval geometry 泄漏、teacher SVD 泄漏、只报 online cost | oracle 结构通过后 |
| FCG nonlinear preconditioner | 通常 `0/0` operator setup；若网络内部调用 operator 则逐次计 | 每步 `1` network apply，另计 FLOPs/time/memory | 不满足普通 PCG fixed-linearity；需 FCG safeguards | 理论上可跨 geometry，实证风险最高 | residual trajectory/label 泄漏、普通 PCG 冒名、fallback 隐藏 | 最后 |

当前 toy 为 `m=150,n_a=1000,m_c=50`。exact Jacobi 的当前 runner 路径使用 dense active matrix，故
应计 `1001F-equiv`，不能写成 5 个 batch calls。用 basis measurements 调 `A^T e_i` 可将 exact
diagonal 降到 `150Adj-equiv`：

```text
D_ii = ||A^T e_i||_2^2 + lambda.
```

同样的 150 个 pulled-back rows 已足以形成完整 `AA^T`，所以 exact camera-block 的这个构造路线
只是结构 oracle，不是效率方案。若已经付出 `mAdj` 并形成全部 row Gram，就必须承认 full dense solve
也可用；不得拿 exact block 的每步便宜掩盖它的 setup。

### 3.1 Identity

用途：

- 每个 `k`、每个 RHS 和每个 budget 的不可删除基线；
- 隔离 solver、target 与 preconditioner 的收益；
- 为 setup-adjusted 方法提供“把 probes 换成更多 CG steps”的公平 comparator。

任何带 `q` setup probes 的方法，至少同时比较：

```text
candidate: q setup pairs + k PCG steps,
identity:  0 setup pairs + (q+k) CG steps,
```

并遵守两维 F/Adj cap。只与 identity 的相同 `k` 比较，会系统性偏袒有 setup 的 candidate。

### 3.2 Exact Jacobi oracle

定义：

```text
M_J = diag(AA^T) + lambda I,
P_J r = r / diag(M_J).
```

优点是数学干净、fixed SPD 容易验证、每步几乎免费；它回答的是“measurement coordinate scaling
能否解释当前慢收敛”。缺点是 exact diagonal 并不是所有 diagonal preconditioner 的数学上界，且
在当前 operators 上 row norms 只变化约 `3--4x`，主要病态性来自相关性而不是尺度。

**判决：**

- reviewer-proof control 已完成，覆盖 `lambda=0`、两种 backbone 与固定
  `k={0,1,2,4,8,12,20,32}`；正式状态为 exact-Jacobi oracle NO-GO。
- 不再发展：spectral preflight 与 RHS-specific empirical gate 均失败；禁止随后扫描 damping、floor、
  per-case `k` 或 diagonal exponent 来寻找事后正结果。
- 立即停止 Hutchinson `diag(AA^T)` estimator，因为它只会以噪声和额外 setup 逼近同一 exact
  diagonal。若未来提出不同目标的 learned diagonal，必须作为新假设重新给出 oracle 与预注册。

### 3.3 Camera-block Jacobi

按 camera index 将 measurement coordinates 分块：

```text
M_B = blockdiag(A_1 A_1^T, ..., A_C A_C^T) + tau I,
P_B r = block_solve(M_B, r).
```

`tau` 是 preconditioner floor，不等于 system damping `lambda`。它可以保证 `M_B` SPD，但必须按
operator scale 归一化并在 development 前冻结。每块 Cholesky 只做一次，随后每步 triangular solve。

只读谱诊断给出约 `6.10x` 条件数改善，M2.6 empirical packet 随后确认 camera-local
ray/component correlations 确实解释了大量 identity-CG 慢收敛。它仍只是经典 block Jacobi，不构成
方法创新；而且它只证明 solver mechanism，未通过 24-call/no-harm 联合门。

只有 T1 target Pareto 通过后，部署路线才可依次比较：

1. 由已知 ray weights/geometry 解析构造的 block surrogate；
2. 小 `q` operator-probed block sketch；
3. geometry-conditioned learned block Cholesky；
4. block plus low-rank cross-camera correction。

### 3.4 Hutchinson diagonal

对 Rademacher probes `omega_j`：

```text
d_hat = (1/q) sum_j omega_j .* (AA^T omega_j),
M_H = diag(positive_floor(d_hat)) + lambda I.
```

每个 probe 必须调用一次 `A^T omega_j` 和一次 `A(A^T omega_j)`，故 setup 为 `qF/qAdj`。原始
Hutchinson diagonal 可出现负值；只能使用预注册的 positive transform，例如 scale-aware clipping 或
softplus，并报告 clipping rate。按 evaluation case 选择“最幸运的 random seed”属于泄漏。

本项目中的 stop rule 很简单：

- exact Jacobi J1 已失败，Hutchinson diagonal 自动 `NO_GO`；
- 不允许借 random estimator noise 重新搜索一个偶然更好的 diagonal；
- 若未来在不同 target 下重开该假设，toy evaluator 仍必须报告 `d_hat` 对 exact diagonal 的 relative
  error、sign/clipping rate 和 preconditioned spectrum，但 exact quantities不得进入 candidate process。

### 3.5 Nyström / low-rank

对固定随机 sketch `Omega in R^(m x q)`：

```text
Y = (AA^T) Omega,
C = Omega^T Y,
AA^T approximately Y C^dagger Y^T.
```

one-pass setup 是 `qF/qAdj`。推荐 SPD 形式不是裸 rank-`q` matrix，而是

```text
M_LR = D_positive + U S U^T,
```

其中 `D_positive` 可以是 frozen scalar、analytic block floor 或已通过的 block preconditioner；通过
Woodbury 应用 `M_LR^-1`。所有 small Gram inverses 都要有 eigenvalue/pivot audit。

在当前结构下，优先测试 **block plus low-rank**，用 low-rank 捕获跨 camera coupling。纯 Nyström
若没有 positive complement，在 `q<m` 时是 singular，不能作为普通 PCG 的 SPD preconditioner。

单次 24-call protocol 中 `q+k<=10`，所以 Nyström 很可能在冷启动上输给 identity。只有以下一种
情况允许继续：同一真实 geometry 有预注册的多 observation reuse，且 cold/amortized 两张表同时通过。

### 3.6 Geometry-conditioned learned SPD preconditioner

允许的参数化按风险由低到高为：

```text
positive diagonal:
    M_G = diag(softplus(d_theta(G)) + epsilon_G)

camera blocks:
    M_G,c = L_theta(G_c) L_theta(G_c)^T + epsilon_G I

block plus low rank:
    M_G = blockdiag(M_G,c) + U_theta(G) S_theta(G) U_theta(G)^T,
    S_theta(G) positive semidefinite.
```

主候选必须满足：

- 同一 weights 跨 camera count、pose、mask、detector layout；
- camera permutation 导致严格对应的 matrix permutation，而不是输出变化；
- `M_G` 只由公开 geometry/operator descriptors 生成，solve 内 factor hash 不变；
- training geometry 可用 exact block/spectrum 作 teacher，但 development/OOD/fresh teacher 严禁进入训练；
- online `0F/0Adj` 不等于训练免费，必须另报 offline teacher F/Adj、GPU hours、参数量和数据量；
- 与 non-geometry learned SPD、analytic block、exact block oracle、Nyström 和 identity 同表比较。

“把 camera parameters concatenate 到 MLP”不足以成为创新。潜在贡献只能来自一个明确的、
Krylov-compatible、camera-set equivariant SPD 结构及其跨 geometry 证据。

### 3.7 FCG nonlinear preconditioner

形式为

```text
z_i = P_theta(G, r_i, history_i),
```

它通常不是固定线性 map，不能通过 ordinary PCG fixedness test。若授权，必须：

- 明确实现并命名 FCG 变体，冻结 recurrence、orthogonalization、memory、restart 和 residual replacement；
- 每步仍只允许一个共同的 `H p_i`，网络内部额外 F/Adj 逐次计账；
- 记录 `rho_i=r_i^T z_i`、true residual、curvature、fallback 与 factor/network hash；
- 对 `rho_i<=rho_floor`、NaN、residual explosion fail closed 到预注册的 identity/fixed-SPD 路径；
- 报告 network calls、FLOPs、wall-clock、peak memory，不能用 `0F/0Adj` 宣称零成本；
- 与同一网络生成一次 fixed SPD factor 的 PCG 版本比较，隔离“非线性”是否真的必要。

只有 fixed block/low-rank 和 learned SPD 均留下明确 finite-budget gap，且 train-only residual-trajectory
diagnostic 显示 RHS-dependent action 有可复用 headroom，才允许进入 FCG。否则停止。

## 4. 泄漏边界

### 4.1 Candidate 可以看

- measured `y`、公开 geometry、support/mask 和受 counter 保护的 `F/Adj`；
- 对 FCG，当前 case 的 residual/curvature/history；
- train-only 学到并在 development 前冻结的 weights/state；
- 固定 random sketch seed、`q`、block partition、rank、floor 和 fallback rule；
- 当前 solve 的非标签数值故障。

### 4.2 Candidate 不可以看

- truth、family/interface label、clean observations、hidden rays、renderer B；
- evaluation dense `A`、SVD/eigenvectors、rank、condition number 或 exact block/diagonal；
- field/H1/clean score、case ranking、split aggregate；
- OOD/fresh trajectory 用于训练或选择 `q/rank/floor/k`；
- 上一个 evaluation case 的 sketch、Krylov basis、warm start 或 adaptive statistics，除非它们是同一
  预注册 geometry cache 且不含 observation-dependent state；
- scorer 已计算的 true residual，若已分类为 evaluation-only。

exact Jacobi、exact block 和 exact spectral low-rank 在 opened toy 上都只能是 oracle/headroom。它们
不含 truth 不代表可以免费进入部署算法；operator access、setup cost 与 opened-spectrum tuning 是另一类
信息边界。

## 5. 顺序门与 stop rules

所有门按顺序执行。通过后门不能覆盖前门的数学、预算或泄漏失败。

### G0：冻结 M2.4 target 与 preflight

必须全部满足：

- target 为 `affine_observation`，`lambda=0` 主线与 damped diagnostic 分开；
- forward/adjoint、linearity、determinism、support 和 dtype tests 通过；
- 当前 full-row-rank toy 与专门构造的 singular-compatible toy 都通过；
- learned preparation 固定为 `13F/13Adj`，core 固定为 cached `(k+1)F/kAdj`；
- exact affine oracle 仅 evaluator-side；prediction-first hash 生效；
- identity 的 `k={0,1,2,4,8,10}` 重新作为不可删基线；
- 所有 preconditioner setup 都有独立 counter namespace。

失败：`M2_5_TARGET_OR_PREFLIGHT_INVALID`，停止全部预条件器实验。

### J1：Exact Jacobi 一次性 control

spectral continuation 门：

```text
median kappa(H) / kappa(M_J^-1/2 H M_J^-1/2) >= 1.25,
and no geometry worsens by more than 10%.
```

当前 12-geometry 谱诊断已失败该门，checksum-valid RHS-specific run 也没有任何 development
selection。若将来复现，该 control 仍必须在某个预冻结 `k<=10` 同时满足：

- true residual/visible defect 相对 identity 同 `k` 的 median 至少改善 `2x`；
- 90th-percentile defect 不恶化；
- measured reprojection vs matched CGLS mean `<=1.10x`；
- field gain `>=5%`、H1 gain `>=3%`、harm rate `<=5%`、worst gain `>=-5%`；
- 无 breakdown，且 setup 明确标 oracle、不得进入 runtime claim。

现有 packet 已触发 `M2_5_EXACT_JACOBI_PRECONDITIONER_ORACLE_NO_GO`。其直接后果是：

```text
stop Hutchinson(diag(AA^T));
do not HPO diagonal floors/exponents on opened OOD;
advance only to exact camera-block oracle.
```

### B1：Exact camera-block headroom

先过 spectrum gate：所有 geometry 条件数改善 `>=2x`。然后在相同 RHS、相同 `k` 上过 J1 的
finite-budget、field/H1、tail 和 reprojection 门。另加：

- camera permutation equivariance 逐 bit/tolerance 通过；
- 每块 Cholesky 无非正 pivot；若使用 jitter，jitter rule 跨 geometry 冻结；
- full dense solve、exact block 与 identity 的 setup/operator calls 分开报告；
- exact block 的 gain 不能只来自 per-case `k` 或 jitter selection。

M2.6 已通过 spectrum、closure、breakdown 与大部分 finite-k mechanism checks，但 B1 的联合
reconstruction gate 失败：`k=8` 仍未过 reprojection，`k=12` 超主预算且 development harm/worst
失败。正式状态是 `M2_6_CAMERA_BLOCK_PRECONDITIONER_ORACLE_NO_GO`。

因此当前不得进入 deployable block/learned SPD。先执行下面的 T1；只有 T1 证明 24-call 内存在联合
可行点，才允许一次 block-plus-low-rank oracle或 deployable block estimator。

### T1：Target/no-harm Pareto feasibility

这不是新 preconditioner HPO，而是判断“更快到达 `Ax=y`”是否值得继续的 evaluator-side ceiling：

1. 在 opened development 上补齐 exact block 的 `k={9,10}`，因为现有 packet 从 8 跳到 12；不得
   改 network、target、block、floor 或 baseline。
2. 对每个预冻结 `k<=10` 重算 measured、clean/hidden、field/H1、harm/worst 和总调用预算。
3. 另报 truth-visible per-case best-`k` oracle，仅作为存在性上界；它不得成为 candidate stopping rule。
4. 检查是否存在一个不看 truth 的固定 `k` 或只看 solver residual 的冻结 threshold，在两种 backbone
   上同时过所有门。

T1 GO 必须同时满足：

```text
F_total<=24 and Adj_total<=24;
measured reprojection ratio<=1.10;
field gain>=5% and H1 gain>=3%;
harm rate<=5% and worst gain>=-5%;
independent/clean reprojection<=1.05x strongest matched baseline;
same fixed rule works for both backbones without OOD selection.
```

若连 per-case best-`k` ceiling 都无联合可行点，输出 `M2_5_AFFINE_TARGET_PARETO_NO_GO`，停止
Nyström、learned SPD 和 FCG。下一研究问题应改为 noise-aware damping、selective fallback 或新的
data-consistency target；这属于新 hypothesis，不能继续叫“预条件器加速”。

若 per-case ceiling 存在但固定 truth-free rule 不存在，只允许研究可观测 risk/stopping gate，不允许
用 geometry/family label 训练隐藏 router。只有固定 truth-free rule 已可行而 `k<=10` 仍差少量 residual
时，才允许一次 block-plus-low-rank `r in {4,8,16}` oracle。

### D1：Deployable diagonal/block/sketch

**当前状态：blocked by T1。** M2.6 的 `continue_deployable_preconditioner_estimation=false` 必须被
尊重；不能因 camera-block 谱改善明显就提前训练 learned SPD。

任何 Hutchinson、Nyström 或 probed block 必须同时比较：

1. 相同 `k` 的 identity，测纯 preconditioner action；
2. 相同总 `q+k` operator budget 的 identity，测净收益；
3. cold-start 和合法 reuse 下的 amortized 两张表；
4. F/Adj calls、wall-clock、peak memory 与 setup latency。

继续门：

- 总 `24F/24Adj` cap 内达到 B1 的 reconstruction gates；
- 相对 total-budget identity，time-to-frozen-tolerance 至少降低 `20%`；
- 3 个固定 sketch seeds 不反转，breakdown/fallback 为 0；
- Hutchinson clipping rate `<=5%`，toy exact-diagonal median relative error `<=20%`；
- low-rank/block Woodbury dense-tiny relative solve error `<=1e-8` float64；
- setup cache miss、hash mismatch 或 geometry change 时 fail closed，不静默复用。

失败：对应结构 `NO_GO`，不得通过删掉 setup 后改写为 speedup。

### L1：Geometry-conditioned learned SPD

继续门除 D1 外还包括：

- PCG fixedness/linearity/symmetry/SPD invariants 100% 通过；
- 同一 weights 覆盖全部 held-out camera count/pose/mask；
- development 相对 non-geometry learned SPD 和最佳 classical deployable preconditioner均有 `>=5%`
  time-to-tolerance 或 matched-time residual gain，paired CI lower `>0`；
- opened OOD 不反转，3/3 model seeds mean gain `>0`；
- independent renderer `<=1.05x` strongest matched baseline；
- online speedup在计入 geometry encoder/factorization后仍存在；
- exact block oracle gap 被明确报告，不把 teacher ceiling 写成 candidate 成功。

失败：`M2_5_LEARNED_SPD_NO_GO`。若失败原因是结构/跨 geometry，而不是确有 RHS-dependent residual
headroom，不得自动升级 FCG。

### F1：FCG nonlinear gate

FCG 只在 L1 后单独预注册。必须满足：

- ordinary PCG 名称完全删除；FCG recurrence 的 tiny-matrix reference test 通过；
- 每一步 `rho_i=r_i^T P_i(r_i)>rho_floor`，true residual finite；
- fallback rate `<=5%`、severe harm `0`，fallback rows不删除；
- 相对最佳 fixed-SPD PCG 的净 wall-clock/time-to-tolerance gain `>=10%`；
- nonlinear network 相对“同网络一次性生成 fixed SPD”确有 paired gain；
- train-only trajectory 训练，development 选择一次，OOD/fresh 不更新 weights/state。

失败：`M2_5_FCG_NONLINEAR_NO_GO`。不得用“neural operator 更灵活”替代实证门。

### G2：Fresh 仍保持关闭

只有 G0、对应 headroom 门、D1/L1 和 independent-forward gate 在 opened data 上全部通过，且
candidate/config/code/weights/ledger/validator hashes 冻结，才可另写新的 preregistration。本文始终要求：

```text
open_fresh_or_final = false
claim_deployable_algorithm = false
claim_real_bost_generalization = false
```

## 6. 机器可验不变量

### 6.1 每 case 最小持久化字段

```text
case_id, split, geometry_sha256, operator_sha256, support_sha256,
target_mode, lambda, rhs_sha256,
preconditioner_kind, preconditioner_role, preconditioner_sha256,
preconditioner_weights_sha256, preconditioner_random_seed,
block_partition_sha256, rank_q, floor_rule, cache_key, cache_hit,
setup_forward_equivalents, setup_adjoint_equivalents,
setup_wall_seconds, setup_peak_memory_bytes, setup_reuse_count,
reconstruction_forward_equivalents, reconstruction_adjoint_equivalents,
preconditioner_apply_calls, network_apply_calls,
k_planned, k_completed, dtype,
norm_b, norm_z, norm_recursive_r, norm_true_r,
norm_measurement_defect, closure_defect,
minimum_preconditioned_rho, minimum_curvature,
breakdown_code, fallback_code, prediction_sha256
```

`preconditioner_role` 必须是 `identity`、`oracle_headroom` 或 `deployable_candidate` 之一。oracle setup
不能通过改字段进入主效率排名。

### 6.2 通用 solver invariants

1. **Adjoint/linearity/determinism/support**：继承 M2.3 thresholds，float64 adjoint defect `<=1e-10`，
   float32 `<=1e-5`。
2. **Core call identity**：zero start、cached pullbacks、无 replacement 时，`F_core=k+1`、
   `Adj_core=k`；setup counters 不得混入或丢失。
3. **Batch equivalence**：probe tensor width为 `q` 时，ledger 增加 `q` equivalents。
4. **Affine output identity**：`x_k == u-A^T z_k`，inactive support exactly zero。
5. **Residual closure**：

   ```text
   d_k = Ax_k-y,
   r_k = b-(AA^T+lambda I)z_k,
   ||d_k-r_k-lambda z_k||/scale <= 1e-10 tiny float64,
                                      <= 1e-8 experiment packet.
   ```

6. **Converged-solution invariance**：同一 `H,b,lambda` 下 identity/Jacobi/block/low-rank 在 dense tiny
   solve 达 tolerance 后输出相同 `z` 与 `x`；预条件器不得改变目标。
7. **Scaling**：`A->cA, y->cy, lambda->c^2 lambda, M->c^2M` 后 output 不变至 tolerance。
8. **Failure completeness**：planned Cartesian product 每项恰有一行；breakdown/fallback/NaN 不得缺失。
9. **No-label sentinel**：candidate 访问 truth/family/clean/hidden/dense-SVD keys 立即终止。
10. **Prediction-first**：scorer 解锁 truth/hidden renderer 前，field、preconditioner、ledger hashes 已落盘。

### 6.3 Ordinary PCG preconditioner invariants

对实际应用的 `P=M^-1`，在每个 geometry 上用固定 probes `u,v,a,b` 验证：

1. **Fixedness**：每一步 `preconditioner_sha256` 相同；相同 residual 重复调用 bitwise/tolerance 一致。
2. **Linearity**：

   ```text
   ||P(au+bv)-aP(u)-bP(v)|| / scale <= 1e-10 float64,
                                           <= 1e-5 float32.
   ```

3. **Symmetry**：

   ```text
   |u^T P(v)-P(u)^T v| / scale <= 1e-10 float64,
                                      <= 1e-5 float32.
   ```

4. **Positive definiteness**：所有非零 test vectors 满足 `v^T P(v)>rho_floor*||v||^2`；同时记录
   minimum Rayleigh quotient。
5. **No hidden operator calls**：一次 `P(r)` 前后 F/Adj counters 不变，除非 manifest 明确把该方法
   定义为 operator-using preconditioner并逐次记账。
6. **Apply count**：initial residual 与每个 next residual 的 apply 次数按实现冻结。若优化掉 final apply，
   必须所有方法一致并由 event sequence 证明，不能只改报表数字。

### 6.4 结构专属 invariants

**Exact/Hutchinson diagonal**

```text
all d_i finite;
all d_i >= frozen_scale_aware_floor;
exact toy: d_i == ||A^T e_i||^2 + lambda;
apply(r) == r/d;
```

另报 raw-negative count、clipped count 与 exact-to-estimated error。

**Camera blocks**

```text
partition is disjoint and covers all measurement coordinates exactly once;
M_c == M_c^T;
Cholesky(M_c) succeeds under frozen jitter rule;
camera permutation Pi gives M(Pi G) == Pi M(G) Pi^T;
block solve agrees with dense solve <=1e-10 float64.
```

**Low-rank/Woodbury**

```text
D diagonal/block base is SPD;
S is PSD under frozen parameterization;
small Gram matrix has positive pivots/eigenvalues;
Woodbury apply agrees with explicit dense inverse on tiny systems;
Omega, q, power iterations, seed, and orthogonalization hashes are fixed.
```

**Learned SPD**

```text
factor generated once per solve;
factor hash remains fixed for all iterations;
camera permutation and mask deletion are equivariant;
weights hash belongs to train-only freeze manifest;
no development/OOD/fresh teacher target appears in training manifest.
```

**FCG nonlinear**

FCG 不应通过 ordinary-PCG linearity/fixedness test；validator 必须反而确认方法被标为 nonlinear，并验证：

```text
rho_i = r_i^T z_i > rho_floor or declared fallback fires;
algorithm-specific beta/orthogonalization recurrence matches a tiny reference;
true residual replacement follows frozen schedule;
network/operator/fallback calls equal event ledger;
no hidden state crosses case boundaries.
```

## 7. 最小实验矩阵

第一轮 opened-data M2.5 只允许下列矩阵，避免同时展开所有候选：

| 阶段 | 方法 | `k`/rank | 目的 |
|---|---|---|---|
| P0 | identity | `k={0,1,2,4,8,10}` | 重建基线与 total-budget comparator |
| P1 | exact Jacobi oracle | 已完成，正式 NO-GO | diagonal-family negative control |
| P2 | exact camera-block oracle | 已完成，机制 signal / 联合 NO-GO | camera-local correlation 已验证，harm 未解 |
| T1 | target/no-harm Pareto | 补 `k={9,10}`，加 per-case ceiling | 判断纯加速是否仍有联合可行点 |
| P3 | block-plus-exact-spectral-rank oracle | `r={4,8,16}`，仅 T1 GO 后 | 判断剩余 cross-camera low-rank headroom |
| P4 | deployable block/learned-SPD | 当前 blocked | 仅 T1/P3 授权后检验净预算收益 |
| P5 | FCG | 当前 blocked | 只有 fixed SPD 与 target 均可行时另立协议 |

Hutchinson 不在默认矩阵中，因为 exact Jacobi 的 spectral 与 empirical gates 均已失败。Nyström、
learned SPD 与 FCG 在 T1 前全部 blocked：它们不能用更复杂的 solver 隐藏 `Ax=y` target 的 tail harm。

每阶段必须同时报告：

- recursive residual、true residual、`Ax-y` closure；
- matched `F/Adj`、wall-clock、peak memory、setup cold/amortized；
- measured、clean 与 independent-renderer reprojection；
- field L2、H1、harm/worst/severe harm、3/3 seed direction；
- condition number只作 opened-toy evaluator metric，不作 candidate stopping input；
- JACRU 与 pooled CNN 使用同一 preconditioner，防止把通用 solver收益归给 set encoder。

## 8. 论文可辩护创新边界

M2.3 primary-literature audit 已确认以下均是旧概念：

- `AA^T` measurement-space CG/CGNE/CRAIG；
- network output 的 affine/data-consistency projection；
- learned prior 加 CG data-consistency；
- inverse-problem learned preconditioner；
- neural operator 作为 FCG preconditioner；
- variable-sensor/operator learning 与 geometry-informed neural operator；
- neural BOST、finite-aperture BOST 和三维折射率重建。

因此以下都**不能**写成创新：

- 使用 identity、Jacobi、camera-block Jacobi、Hutchinson 或 Nyström 本身；
- 把 `AA^T` 从 dense 改成 matrix-free；
- 把 camera parameters 输入一个 MLP；
- 用 Cholesky/softplus 让矩阵 SPD；
- 把 nonlinear network 放进 FCG；
- 在 opened 12^3 toy 上降低 iteration count。

只有结果全部通过后，可能辩护的窄贡献是：

> 在有限孔径、可变相机几何 BOST 中，构造并审计一种 camera-set equivariant、显式 SPD、
> Krylov-compatible 的 geometry-conditioned block/low-rank preconditioner；在完整冷启动与复用预算下，
> 它相对 identity、exact Jacobi、camera-block、Nyström、non-geometry learned SPD 和同预算传统重建，
> 跨 geometry 缩短 time-to-tolerance，同时通过 independent-forward、OOD 和 no-harm gates。

即便通过，这首先仍是 **BOST-specific algorithm instantiation and controlled empirical result**，不是新的
投影公式、Krylov 理论或“首个 learned preconditioner”。若 exact camera-block 已能解释全部收益，论文
应把结论写成经典结构发现与负结果，不应为追求 novelty 强行加入网络。

若所有 deployable preconditioners 在计入 setup 后都不胜 identity/CGLS，可辩护的负结果是：

> 当前 finite-aperture toy 的 affine-projection headroom 存在，但 one-shot variable-geometry setting 中
> exact diagonal 无有效 headroom；camera-local blocks 显著加速 measured consistency，却只在 expanded
> budget 达到 reprojection 门，并且没有消除 field tail harm。因而当前阻断项已从线性求解速度转为
> noisy affine target、预算与 no-harm 的联合不可行性。

这比把 oracle setup 删除后宣称加速更可信。

## 9. 最终执行建议

推荐的唯一顺序为：

```text
identity freeze
  -> exact Jacobi one-shot control: COMPLETE / NO-GO
  -> exact camera-block oracle: COMPLETE / MECHANISM SIGNAL / OVERALL NO-GO
  -> target/no-harm Pareto ceiling at k<=10: NEXT
  -> if and only if target Pareto passes: block + low-rank oracle
  -> deployable or learned SPD only after oracle and total-budget gates
  -> FCG only under a new, separately justified protocol
  -> fresh remains closed
```

对“exact Jacobi 是否值得做 headroom”的最终回答是：**值得做的那一次已经做完，结果明确不够。**
它以最低概念复杂度关闭了 diagonal scaling 假设并补齐了论文 negative control；`1001F-equiv` oracle
setup、空 selection 和 `15x--18x` 量级的 reprojection ratio 一起禁止 deployable diagonal estimation。
立即停止 Hutchinson diagonal。其后应做的 exact camera-block oracle 也已经完成：约 `6.1x` 的谱改善
转化成了强 residual acceleration，但 `k=8` 仍差 reprojection 门，`k=12` 超预算且 harm/worst 失败。
所以下一步不是更复杂的 preconditioner，而是 T1 target/no-harm Pareto ceiling。T1 不通过就停止整个
预条件器加速分支；T1 通过后，才有资格测试 block-plus-low-rank 与 geometry-conditioned learned SPD。
