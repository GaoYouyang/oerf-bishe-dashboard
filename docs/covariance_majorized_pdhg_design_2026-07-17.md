# Covariance-aware signed BOST factor-majorized diagonal/block PDHG 设计说明

日期：2026-07-17
状态：`FORMAL_GATE_A_ATTESTED / GATE_B_E2_MECHANISM_NO_GO / NO_FRESH_REAL_OR_WIN_CLAIM`

2026-07-17 实现进度说明：tiny CPU/float64 signed-factor oracle、显式活动坐标
`E/E^T`、production `P/P^T`、`|G_c|/|G_c|^T`、组合后取绝对值的
`|H R Q|/|H R Q|^T`，以及 `|D_+|/|D_+|^T` 已落地。
production matrix-free ones-pass、camera/site shared step、exact-zero mask、删除
常数账本、底层物理调用账本与固定 6 步 Huber recurrence 也已与独立 site-major
dense oracle 逐步对齐。当前实现只接受一个冻结 measurement-scale 实例和彼此独立
的 view-local whitening blocks；声明跨 camera covariance 时会 fail closed。这仍不等于
正式 Gate A 已在 clean source commit `53f1bccb287744a6ab97d7d6c2a86d556515e34a`
上生成：13/13 E1 PASS，20 个 selector 展开为 34 个 case、零跳过，独立 validator
完成 333 项 core checks，并通过第二次 `--no-write` 复核。范围仍严格限于单一冻结
scale、view-local synthetic mechanics。随后 V4 Gate B 已在 source commit
`204bbe8` 上完成 256 条方法行，并由独立 validator 重算 4,048 项：voxel-factor
相对 scalar mean gain 仅 1.321%，与 graph-PCGLS 的 mean error gap 为 133.439%，
八门过五门，正式 NO-GO。fresh、真实数据和方法胜出均未运行。

> 本文记录候选、证明义务、实现门禁和已完成的证伪协议，不报告任何胜出结果。
> 全文严格分为“可证明事实 / 实现假设 / 待证伪假设”；条件性定理不等于当前实现已通过。

| 可证明事实 | 实现假设 | 待证伪假设 |
|---|---|---|
| 在所列非负因子条件下，因子乘积给出 `|A_b|` 的逐元素上界，并可推出 `eta^2` 步长安全界 | 当前 streaming BOST 可按同一因子边界重构 absolute-factor kernel，且 camera/site 共享步长可无歧义实现 | 更紧的 covariance-aware factor majorizer 能在同物理调用预算下改善收敛、结构指标和坏尾 |
| Pock-Chambolle 对角预条件及绝对行列和思想属于 prior art | `|W_b|` 可在 covariance block 尺度显式物化，`P/P^T` 可流式执行而不组装 `A` | 独立 flow-off/calibration 得到的尺度仍能复现 synthetic oracle-scale 机制信号 |
| 当前 synthetic runner 的 `scale_by_view` 由 clean truth 的 forward projection 构造 | 新实现会对因子、尺度来源、cache fingerprint 和调用账本 fail closed | 候选相对 scalar PDHG、强 graph-PCGLS 和冻结消融满足同预算门；Gate B 已证伪此假设 |

## 一、可证明事实

### 1. 问题、因子和成立条件

在 support/gauge 已消元后的活动变量 `x in R^n` 上，目标仍是现有确定性问题：

```text
min_x  0.5 * sum_b ||A_b x - y_b||_2^2
       + lambda * sum_v rho_delta(||(D x)_v||_2)
       + I_C(x).
```

对每个 covariance block `b`，冻结分解为：

```text
A_b = W_b P G_c E
D   = D_+ E
```

各因子的审计含义如下。

- `E` 是活动变量到完整 voxel grid 的 0/1 selection/embedding，故 `E >= 0`；support 外和 gauge-fixed voxel 不进入活动变量。
- `G_c` 是物理 forward 使用的带符号 voxel-centered gradient；当前离散在内部使用 centered difference，在盒边界使用已声明的一侧差分。
- `P` 只包含逐分量的 ray sampling、非负三线性插值和非负求积/路径累加权重，必须满足 `P >= 0`。
- 令 `Q_b` 为每条 ray 的 camera-plane `u/v` 投影，`R_b` 为带符号 `ray_scale` 的对角作用，`H_b` 为 covariance whitening，`s_d` 为冻结 measurement normalization；定义
  `W_b = s_d H_b R_b Q_b`。因此 camera projection 和完整的带符号 `ray_scale` 必须并入 `W_b`，不能留在 `P` 中。
- `D_+` 是先验使用的 forward-Neumann 3D gradient；它与物理 `G_c` 分账，并有独立精确伴随。
- 若 whitening 存在跨 camera 非零项，`b` 必须取整个 covariance-connected block；不得把一个耦合矩阵伪拆成独立 camera block。当前 production 接口尚未实现该一般情形，因此只能接受每个 view 独立的 block；任何 cross-view coupling 声明都会在构造阶段被拒绝。

`P >= 0` 只在上述分账下成立。若使用带负权的高阶插值、带符号求积，或把 camera projection / signed ray scale 留在 `P`，下面的 majorization 证明立即失效。

### 2. Signed factor majorizer

绝对值均指**逐元素绝对值**，不是谱意义的 matrix absolute。对每个 block 定义：

```text
M_b = |W_b| P |G_c| E
N   = |D_+| E
```

由于 `P,E >= 0`，矩阵乘积三角不等式给出：

```text
|A_b|
  = |W_b P G_c E|
 <= |W_b| |P| |G_c| |E|
  = |W_b| P |G_c| E
  = M_b.
```

同理 `|D| = |D_+E| <= N`。因此用户要求的核心条件可写为：

```text
A_b = W_b P G_c E,
D   = D_+ E,
M   = |W_b| P |G_c| E >= |A_b|,
```

其中最后一个不等式是有条件的确定性事实，不是经验抽样结论。用 `A_b 1` 或 `A_b^T 1` 代替绝对行列和不安全，因为 `G_c`、camera projection、ray scale 和 whitening 的正负抵消会虚假地产生小值或零值。

### 3. `omega_A / omega_D` 行列和与 block inflation

令 `1_k` 表示相应维度的全一向量。原始 majorizer 行列和为：

```text
omega_A,row,b = M_b 1_n
omega_A,col   = sum_b M_b^T 1_{m_b}

omega_D,row   = N 1_n
omega_D,col   = N^T 1_{m_D}

omega_col     = omega_A,col + omega_D,col.
```

为保持 block prox 的原定义，dual 行步长不逐分量自由变化，而使用以下安全膨胀。

```text
camera block maximum:
rho_A,b = max_i omega_A,row,b[i]          # 仅在非零耦合行上取 max

isotropic TV site shared maximum:
rho_D,v = max_{c in {x,y,z}} omega_D,row[v,c]
```

- 同一 covariance/camera block 的所有活动 data dual 行共享 `sigma_A,b`。
- 同一 TV site 的 `x/y/z` 三个 dual 分量共享 `sigma_D,v`；只要该 site 任一分量耦合，三个分量必须共用一个标量步长。
- 这种 site 共享是必要的：否则当前 isotropic TV 球投影和 Huber 径向 shrink 不再是所写 diagonal metric 下的 prox，必须另解 weighted-ball prox。

### 4. `T / Sigma` 公式

固定 `0 < eta < 1`。在所有活动 primal 列上定义：

```text
tau_j = eta / omega_col[j]
T     = diag(tau_j).
```

在活动 data 行和活动 TV site 上定义：

```text
sigma_A,b = eta / rho_A,b
Sigma_A,b = sigma_A,b * I_b

sigma_D,v = eta / rho_D,v
Sigma_D,v = sigma_D,v * I_3

Sigma = blockdiag(Sigma_A,1, ..., Sigma_A,B,
                  Sigma_D,1, ..., Sigma_D,V).
```

零耦合不能用任意 epsilon 掩盖：

- data 行若 `omega_A,row,b[i] = 0`，从活动 saddle system 中删除；其 data term 是常数，若记录 dual gap，可固定解析 dual `p_i=-y_i`。
- 整个 camera block 若 `rho_A,b=0`，整块删除并单列 constant-data ledger。
- TV site 若 `rho_D,v=0`，固定 `q_v=0`；若仅某一分量行是零但 site 仍活动，三个分量继续共享同一 `sigma_D,v`。
- primal 列若 `omega_col[j]=0`，必须从 `E` 中移除并固定 gauge/default 值；不得令 `tau_j=inf`。活动空间上的 `T,Sigma` 必须正定。

### 5. `eta^2` 安全证明

把所有 `M_b` 与 `N` 纵向堆叠为非负矩阵 `H`，则 `|K| <= H`，其中 `K=[A_1;...;A_B;D]`。令每个活动行的实际分母 `r_i` 为所在 camera 的 `rho_A,b` 或所在 TV site 的 `rho_D,v`，于是：

```text
r_i >= sum_j H_ij,
sigma_i = eta / r_i,
omega_col[j] = sum_i H_ij,
tau_j = eta / omega_col[j].
```

对任意 `z`，由加权 Cauchy-Schwarz：

```text
||Sigma^(1/2) K T^(1/2) z||_2^2
 = sum_i sigma_i |sum_j K_ij sqrt(tau_j) z_j|^2
<= sum_i sigma_i r_i sum_j H_ij tau_j z_j^2
 = eta sum_j omega_col[j] tau_j z_j^2
 = eta^2 ||z||_2^2.
```

因此：

```text
||Sigma^(1/2) K T^(1/2)||_2^2 <= eta^2 < 1.
```

这个证明不依赖 power iteration；它依赖 `M_b/N` 的逐元素 dominance、完整 block 划分和零耦合显式消元。任一前提未通过时，不得引用该结论。

### 6. 对应 PDHG 更新保持同一目标

在 camera/site 共享步长下：

```text
p_b^(k+1) = (p_b^k + sigma_A,b * (A_b x_bar^k - y_b))
            / (1 + sigma_A,b)

v_v = q_v^k + sigma_D,v * (D x_bar^k)_v

TV:    q_v^(k+1) = v_v / max(1, ||v_v||_2 / lambda)

Huber: u_v = v_v / (1 + sigma_D,v * delta / lambda)
       q_v^(k+1) = u_v / max(1, ||u_v||_2 / lambda)

x^(k+1) = P_C[x^k - T(A^T p^(k+1) + D^T q^(k+1))]
x_bar^(k+1) = x^(k+1) + theta * (x^(k+1) - x^k).
```

因为 `P_C` 在这里是坐标清零型 support/gauge projection，它在正 diagonal `T` metric 下仍是同一个 mask。`lambda=0` 继续走独立 data-only 分支，不执行含 `1/lambda` 的表达式。

### 7. Prior art 与候选贡献边界

Pock-Chambolle 的 diagonal preconditioning、绝对行列和步长、Chambolle-Pock/PDHG、TV/Huber、block step 和层析中的 primal-dual 都不是创新。仓库现有审计已明确这一边界：[PDHG 一手文献与可创新边界](./pdhg_primary_sources_and_innovation_boundary_2026-07-17.md)。相关一手入口沿用该文档记录，不另造 DOI：

- Chambolle & Pock 基础算法：[HAL 开放稿](https://hal.science/hal-00490826/document)。
- Sidky、Jorgensen & Pan 的 CT problem prototyping：[arXiv 开放稿](https://arxiv.org/abs/1111.5632)。
- Pock & Chambolle 对角预条件：[已审计 DOI 入口](https://doi.org/10.1109/ICCV.2011.6126441)；当前 primary-sources 文档未记录可确认的开放稿，因此本文不虚构开放 PDF。
- Loris & Verhoeven 的 non-separable/Huber-TV 路线：[arXiv 开放稿](https://arxiv.org/abs/1203.4451)。
- 光学梯度层析与 TV：[arXiv 开放稿](https://arxiv.org/abs/1209.0654)。

真正的**候选**贡献边界只能是以下组合，且仍需 prior-art 扩展检索与实验共同证伪：

1. 为含 signed centered-gradient、camera projection、signed ray scale 和 detector-covariance whitening 的 matrix-free BOST，给出可复核的 factor majorizer；
2. 证明并实现 camera covariance block 与 isotropic-TV site 兼容的安全 metric；
3. 用 tiny dense oracle、matrix-free 复算和完整 setup ledger 证明实现没有靠抵消或漏账；
4. 在独立 calibration scale、held-out camera/session 和同物理调用预算下同时改善 mean 与坏尾。

仅“使用 Pock-Chambolle diagonal preconditioning”“把 covariance 放进 PDHG”或“在 BOST 上更快”都不足以构成贡献。

### 8. 当前 `scale_by_view` 的证据上限

当前 synthetic runner 的 `_build_replicate_context` 在 [run_psu_b0_pdhg_screen.py](../site_tools/run_psu_b0_pdhg_screen.py) 中先执行：

```text
truth -> clean = operator(truth)
      -> view_rms = RMS(clean by view)
      -> scale_by_view = relative_noise * max(view_rms, floor) * view_factors.
```

随后 [detector_covariance_whitening.py](../demo_t16_operator/detector_covariance_whitening.py) 用该 `scale_by_view` 除 whitening 输出。故 whitening、`A_b`、majorizer 和步长都间接依赖 clean truth。即使某个 stress 函数签名不接收 truth，也不能消除上游依赖。

因此当前 runner 的最高合法表述只能是：

```text
ORACLE_SCALE_MECHANISM_DIAGNOSTIC
```

它不能支持“可部署尺度估计”“真实 covariance-aware 算法”或 superiority。真实研究必须在 flow-on reconstruction 之前，用样本/会话独立的 flow-off repeats 或 calibration acquisition 冻结均值、covariance 和 `scale_by_view`；其 session、hash、拟合参数及 held-out whiteness 必须入账。

## 二、实现假设

### 9. Matrix-free majorizer setup

实现不组装巨型 `A_b`。对每个 block，用 absolute-factor kernel 计算：

```text
# row sums
u0 = E 1
u1 = |G_c| u0
u2 = P u1
omega_A,row,b = |W_b| u2

# column sums
v0 = |W_b|^T 1
v1 = P^T v0
v2 = |G_c|^T v1
omega_A,col,b = E^T v2

omega_D,row = |D_+| E 1
omega_D,col = E^T |D_+|^T 1.
```

实现假设如下，未通过 E1 前均不视为事实。

- trilinear stencil 的 stored weights 全部 finite、非负，且 absolute forward/transpose 使用完全相同的 index、mask、sample count 和 chunk order。
- `|W_b|` 是组合后 `W_b=s_d H_b R_b Q_b` 的逐元素绝对值；不能用 `abs(H_b @ z)` 冒充 `|H_b|z`。
- `|W_b|` 可在 camera/covariance block 尺度物化；若只能使用 `W_hat_b >= |W_b|`，必须单独证明 dominance，并把更松的 bound 写入方法名。
- 一般算法若 batch 中 calibration scale 不同，metric 必须按样本独立计算；不得广播一个 truth-derived 或其他样本的 metric。当前 production setup 更窄：只接受一个冻结 scale 实例和 `batch_index=0`，多实例输入直接拒绝，尚不主张 batched metric 已实现。
- cache fingerprint 至少包含 geometry、projection、ray scale、whitener、scale provenance、support/gauge、grid shape/spacing、stencil、measurement normalization、dtype 和代码 commit。任一变化均使 cache stale 并 fail closed。

### 10. Setup 成本与 physical-call ledger

majorizer 不是免费预处理。正式 ledger 必须分栏：

| 成本层 | 必记字段 | 是否计为 signed BOST `F/A^T` |
|---|---|---|
| flow-off/calibration acquisition | frames、session、bytes、wall time、hash | 否，但单列 acquisition |
| covariance fit / `W_b` materialization | block 数、fit time、peak memory、scale provenance | 否，但进入 cold start |
| absolute-factor setup | `P_abs_forward_passes`、`P_abs_transpose_passes`、`W_abs_matvecs`、bytes、wall time | 专用 kernel 时不冒充 `F/A^T`，但不得隐藏 |
| signed norm/SVD audit | 实际 signed forward/adjoint 次数 | 若调用生产 `A/A^T`，必须计入 setup physical calls |
| 每次 PDHG iteration | `1F + 1A^T + 1D + 1D^T` | `F/A^T` 各 1 |
| checkpoint scorer | 每 checkpoint 的额外 `1F + 1D` | `F` 单列为 metric call |
| prefix reuse | logical calls、physical calls、saved calls | 必须同时报告，不把复用写成算法便宜 |

至少保存这些机器可读字段：

```text
majorizer_setup_wall_seconds
majorizer_setup_peak_memory_bytes
absolute_factor_forward_passes
absolute_factor_transpose_passes
signed_setup_forward_calls
signed_setup_adjoint_calls
solver_forward_calls
solver_adjoint_calls
solver_gradient_calls
solver_gradient_adjoint_calls
checkpoint_metric_forward_calls
logical_calls_if_independent
physical_calls_with_prefix_reuse
calibration_provenance
cache_fingerprint
```

若 absolute-factor 实现内部直接调用 signed production operator，相关调用必须按实际 `F/A^T` 计数，不能因目的叫“setup”而归零。cold-start、单次 reconstruction 和多次摊销三张表都必须保留。

当前 formal mechanics 实现还做两项窄审计：每次 wrapper 调用前后都读取 measurement
factor 与 regularization factor 自身的计数器，物理增量不是恰好一次就失败；setup 同时
保存 wrapper 与 physical 两份 ledger。对 exact-zero data 行，运行时保存活动/删除索引、
被删除 target 值和 `0.5 * ||y_deleted||^2`，目标函数只在活动 residual 上计算后显式加回
该常数。二者都只证明本 fixture 没有漏账，不替代正式 fingerprint attestation。

为防 setup 后修改 factor 使 zero ledger 失效，当前 formal mechanics 路径只接受 sealed
exact measurement/regularization 实现，并保存全部 setup-critical tensors 的 identity、
pointer、shape、dtype、device、PyTorch version 与内容 SHA-256；solver/scorer 每次入口
都重新核对。内容 hash 专门覆盖 `tensor.data[...]` 这类不增加 `_version` 的 storage 写入。
该检查会同步并复制 tensor 到 CPU，因此它只用于小型 mechanics fixture，**不得用于
Gate B wall-time**。性能路径必须改成运行前一次 attestation、不可变执行副本与独立的
post-run hash，再重新冻结计时协议。

### 11. Tiny dense oracle：固定 6 步

1. 构造两个 camera block 的 tiny float64 问题，显式给出 0/1 `E`、signed `G_c`、nonnegative `P`、signed `W_b`、`D_+`，并刻意包含一个零 data 行和一个零 TV site。
2. 显式组装 `A_b=W_bPG_cE`、`D=D_+E`、`M_b`、`N`；对照 production factor forward/transpose 与 dense matmul。
3. 检查 `min(M_b-|A_b|)` 和 `min(N-|D|)` 在缩放容差内非负；把 projection 或 signed ray scale 错留在 `P` 的 negative fixture 必须被 `P>=0` 门拒绝。
4. 分别用 dense 与 matrix-free ones passes 计算全部 `omega`、camera maximum、site maximum、`T`、`Sigma`，逐值一致；覆盖局部和整块零耦合。
5. 显式 SVD 计算 `||Sigma^(1/2)KT^(1/2)||_2^2`，要求不超过 `eta^2` 加浮点容差；另有遗漏 `abs(W_b)` 或故意放大步长的 fixture，必须确定性越门或被 dominance 门拒绝。
6. 从同一零状态运行**恰好 6 次** dense PDHG 与 matrix-free PDHG，对照每步 `x/x_bar/p/q`、prox、objective 和 ledger；不得只比较终点。

### 12. E1 自动测试门

正式性能运行前，以下项目全部 `PASS` 才达到 E1；任何 `FAIL/MISSING` 都只能判实现无效。

| ID | 冻结检查 | 建议 float64 门限 |
|---|---|---:|
| E1-01 | `E>=0`、`P>=0`、view-local block partition、cross-view metadata 拒绝、projection/ray scale 位于 `W_b` | exact / fail closed |
| E1-02 | `A/A^T`、`P/P^T`、`G_c/G_c^T`、`D/D^T` dot-product | relative error `<=1e-8` |
| E1-03 | tiny dense `M_b>=|A_b|`、`N>=|D|` | violation `<=1e-12 * scale` |
| E1-04 | dense 与 matrix-free 行列和、camera/site maximum | relative error `<=1e-10` |
| E1-05 | exact SVD 安全界 | norm squared `<=eta^2+1e-12` |
| E1-06 | camera shared data prox、site-shared TV/Huber prox、`lambda=0` | max error `<=1e-10` |
| E1-07 | 6-iteration dense/matrix-free recurrence | relative state error `<=1e-10` |
| E1-08 | 零 data 行、零 camera block、零 TV site、零 primal 列 | 无 `inf/NaN`，按合同消元 |
| E1-09 | stale geometry/whitener/scale/support fingerprint negative fixtures | solver 调用前拒绝 |
| E1-10 | setup、solver、scorer、prefix-reuse ledger negative controls | exact counts |
| E1-11 | setup API 不接收 truth/morphology；scale provenance 审计 | real 路径仅允许 independent calibration |
| E1-12 | CPU float64 与 MPS float32 固定 fixture | field relative difference `<=5e-4` |
| E1-13 | 非零 coupling 无论多小都保留；非零 `zero_tolerance` 与多 scale instance 输入 | fail closed |

E1 只证明数值合同和账本可信，不证明重建效果。

## 三、待证伪假设

### 13. 唯一研究假设与解锁条件

待证伪假设是：在独立 calibration scale 下，signed-factor majorizer 比现有 scalar block-PDHG 更贴合 BOST/covariance 的异质耦合，从而在相同 signed `F/A^T` calls 下改善优化效率，并且不以 front、p10 或 worst tail 为代价。

在以下条件满足前不授权性能主张：

1. 现有 scalar PDHG screen 已有可审计诊断，且指出重复的 conditioning 缺口；
2. 本文 E1 全通过；
3. `scale_by_view` 已从独立 flow-off/calibration 冻结，或运行被明确标为 `ORACLE_SCALE_MECHANISM_DIAGNOSTIC`；
4. 候选、消融、预算、top-level independent unit 和门限在开封前写入 hash；
5. fresh/held-out camera/session 未因当前 opened synthetic 结果而污染。

### 14. 冻结消融

所有消融使用完全相同的 `A/W/D/C`、`lambda/delta/theta/eta`、初始化、checkpoint、dtype 和硬件；只改变安全 metric。不得在看到结果后删除“太慢”或“不好看”的项。

| ID | Metric | 回答的问题 |
|---|---|---|
| B0 `scalar-block` | 现有 safety-inflated norm scalar/block steps | 强确定性主基线 |
| B1 `factor-global` | 将 full candidate 的 `T/Sigma` 各自收缩到最小活动对角值 | 仅有 factor certificate 是否足够 |
| B2 `primal-diag-only` | `T=T_full`，`Sigma` 收缩为全局安全标量 | voxel column heterogeneity 是否贡献 |
| B3 `dual-block-only` | `T` 收缩为全局安全标量，保留 camera/site `Sigma_full` | camera/TV row heterogeneity 是否贡献 |
| B4 `coarse-W-envelope` | solver 仍用同一 `A`，majorizer 改用已证明支配 `|W_b|` 的 coarse block envelope | covariance 细结构是否真的有用 |
| C0 `full-factor-majorized` | 本文完整 `T` 与 camera/site `Sigma` | 候选组合机制 |
| O0 `exact-|K|-PC` | 仅 tiny dense 上用显式 `|K|` 的经典 Pock-Chambolle 对角 metric | 衡量 factor majorizer 松弛，不进入大规模胜负表 |

B1-B3 通过把 full metric 收缩为更小正 diagonal metric 保持安全；B4 的 envelope 必须先通过同一 dominance/SVD 门。遗漏 absolute、错误拆 camera、truth scale 等只作为 negative controls，不得伪装成性能消融。

### 15. 同预算性能门

下一协议必须以一个已冻结 finalist objective 运行上述消融。每个 `K` 的 solver 逻辑预算均为：

```text
K forward + K adjoint = 2K signed BOST calls,
K D + K D^T,
无 history 额外调用。
```

primary comparison 是 `C0` 对相同 `(penalty,lambda,K)` 的 `B0`；同时必须对现有 `graph_budget_frontier(K)`。prefix reuse 只降低实际 screen 执行量，不改变每个候选的逻辑预算。候选只有同时满足以下冻结门，才可标为“进入下一验证”，仍不得称为胜出：

```text
mean field gain vs B0                         >= +1.0%
field p10                                     >= -1.0%
field harm rate for gain < -1%                <= 1/16
worst field gain                              >= -3.0%
mean gradient gain                            >= 0.0%
pooled mean front gain                        >= 0.0
front-critical mean front gain                >= 0.0
each top-level replicate/session mean gain    > 0.0%
same logical signed F/A^T calls                exact
median wall-time ratio vs graph frontier      <= 3.0
```

并且：

- C0 还必须按现有门独立击败或至少通过 `graph_budget_frontier(K)`，仅击败较弱 scalar 配置不够；
- 报告 warm solver、cold start（含 covariance/majorizer setup）和预注册重建数下的 amortized time；
- mean、p10、harm、worst、每个 morphology、每个 camera/session 和失败运行全部保留；
- 当前 truth-scaled synthetic runner 即使过门，也只能得到 `ORACLE_SCALE_MECHANISM_SIGNAL_ONLY`；
- 真实研究还需 independent flow-off scale、held-out whiteness、held-out camera/session 和无三维真值时的 reprojection/独立诊断门。

### 16. NO-GO 解释合同

| 观测 | 唯一允许的窄解释 | 禁止解释 |
|---|---|---|
| dominance、adjoint、SVD、prox 或 ledger 失败 | `MAJORIZER_E1_INVALID`，实现/证明链未成立 | “该算法性能差” |
| exact `|K|` tiny oracle 快而 factor candidate 慢 | `FACTOR_MAJORZER_TOO_LOOSE_NO_GO` | “PDHG 无效” |
| oracle-scale 有信号，独立 calibration scale 无信号 | `ORACLE_SCALE_DEPENDENCE_NO_GO` | “真实 covariance-aware 已验证” |
| C0 与 B0 同调用无改善 | `SAME_BUDGET_MECHANISM_NO_GO` | “换 fresh 或加网络还能算成功” |
| mean 正但 p10/harm/worst 失败 | `TAIL_SAFETY_NO_GO` | “总体平均更好” |
| field 好但 gradient/front 失败 | `STRUCTURE_NO_GO`，可能只是平滑/振幅拟合 | “前沿恢复更好” |
| calls 相同但 wall/cold-start 失败 | `RUNTIME_NO_GO` 或仅有数值信号 | “同调用即同成本” |
| B4 与 C0 无差，或 B2/B3 单项已解释全部收益 | 声称的 covariance/组合机制未被识别 | “完整方法每个组件都必要” |
| flow-off whitening 在 held-out session 不白 | `REAL_CALIBRATION_NO_GO` | “用更强 TV 可补偿错误噪声模型” |
| held-out camera/session 发散或坏尾扩大 | `TRANSFER_NO_GO` | “synthetic mean 足以证明泛化” |
| 全部门通过但只有 opened/oracle synthetic | `MECHANISM_SIGNAL_ONLY` | “优于现有方法”“validated”“可投稿结论” |

### 17. 最终声明边界

截至最新状态，signed factor majorizer 的 tiny dense oracle、production 因子组件、
matrix-free ones-pass、exact-zero/physical-call ledger 和 6 步 TV/Huber recurrence
已在 clean commit 的单一冻结 scale、view-local covariance fixture 上通过正式 Gate A。
证明记录、JSON 与 checksum 见 `docs/psu_b0_gate_a_attestation_2026-07-17.md`。
同预算 Gate B 随后已完成并严格 NO-GO：factor vs scalar +1.321%，factor vs
view-block +1.242%，graph gap 133.439%，5 门通过、3 门失败，最终 NO-GO。独立 flow-off/calibration
scale 仍不存在。因此本文不声称候选优于 scalar PDHG、PCGLS、TV/Huber、learned
primal-dual 或任何现有方法；当前 factor-PDHG 路线已关闭。
