# LGWO-A24 R2-F0 协议红队：扩展 span 表示诊断冻结前审查

> 日期：2026-07-21<br>
> 审查对象：拟议的 R2-F0 truth-free 方向生成与 truth-only expanded-span oracle<br>
> 证据范围：6 个已经多次打开的 synthetic development cases；公开 PSU 几何；inverse-crime 型 QMC8 forward<br>
> 当前判决：`HOLD_R2F0_PROTOCOL_NOT_READY_TO_FREEZE`<br>
> 本文性质：协议红队，不是实验结果；没有运行 R2-F0，也没有授权 fresh、learner、real、generalization、algorithm 或 paper claim

## 0. 执行摘要

R2-E0 已经给出一个有效的窄结论：冻结 H1-20 的 20 维缓存 span 内，连 residual-safe
truth joint oracle 都只有 `+0.0040% field / +0.1647% gradient`，因此下一步转向“生成真正跳出旧
span 的方向”是合理的。拟议 R2-F0 的三个方向也有清楚的物理/算法含义：

- `R`：raw weighted-residual backprojection；
- `H`：negative Huber-prior gradient；
- `E`：edge-gated backprojection；
- 以及 `RH / RE / HE / RHE` 二、三方向 union。

但当前文字协议仍有 **5 个 P0 冻结阻断项**：

1. expanded-span truth oracle 若只对比 H1-21/22/23 的普通 solver endpoint，会把“oracle 重配系数”误归因为“新方向更好”；必须同时建立 matched H1-k **span oracle**。
2. `H`、二方向 union、三方向 union 的 `F/A^T` 不对称，不能压成一个“逻辑调用数”或“call pair”；必须逐方法报告 `(F,A^T)` 向量、standalone 成本和 sweep 共享成本。
3. “投影出旧 span”尚未定义是 field、data 还是 H1 augmented metric；若只看 field outside fraction，truth oracle 可沿近测量零空间产生虚假 headroom。
4. 方向公式虽然应当不看 truth，但 family 选择、joint objective、ray endpoint 和结果排序都会看 truth；必须把 direction packet 与 oracle packet 结构隔离，且禁止从这 6 个已见 case 选出“可部署赢家”。
5. source closure 与 independent validator 尚未在运行前冻结。R2-D1 已经出现漏绑 helper，R2-E0 的独立 validator 又是运行后实现；这两种风险不能带入 R2-F0。

在这些 P0 清零之前，不应写 config 或运行正式结果。清零后，R2-F0 仍只能产生两类状态：

- `POSTOPEN_R2F0_NO_USEFUL_EXPANDED_SPAN_HEADROOM_NO_GO`；或
- `POSTOPEN_R2F0_REPRESENTATION_SIGNAL_ONLY_NO_AUTHORIZATION`。

第二种状态也只是“值得另写一个全新、预冻结的协议”，不等于 fresh 已通过，更不授权 learner、真实 BOST、泛化、算法优越性或论文成功。

## 1. 本次红队实际读取的证据

### 1.1 R2-E0

- `demo_t16_operator/configs/lgwo_a24_r2e0_cached_krylov_diagnosis_v3.json`
- `site_tools/run_lgwo_a24_r2e0_cached_krylov_diagnosis.py`
- `demo_t16_operator/cached_krylov_reoptimization.py`
- `demo_t16_operator/results/lgwo_a24_r2e0_cached_krylov_diagnosis_v3/report.json`
- `demo_t16_operator/results/lgwo_a24_r2e0_cached_krylov_diagnosis_v3/independent_validation.json`
- `docs/lgwo_a24_r2e0_cached_krylov_subspace_no_go_2026-07-21.md`

需要继承的教训：

1. v1 的 MPS-to-CPU 合并转换污染 reduced least-squares RHS；v2 又遗漏了 Huber reoptimizer 的 MPS float64 路径。所有 reduced SVD、rank、oracle 与 ray 运算必须统一走“先以源 dtype 复制到 CPU，再升 float64；回程先在 CPU 降 dtype，再进入 MPS”。
2. v3 的 import closure、108 个几何输入摘要、前后 source/geometry revalidation 是正确方向，应继续使用。
3. v3 的 independent validator 有 `873` 项复算，但它在正式运行后实现，只能证明保存证据的事后一致性，不能冒充 pre-run protocol enforcement。
4. R2-E0 的 `relative_singular_value_floor=1e-12` 作用于来自 float32/MPS 的矩阵。R2-F0 若要据此判断“新增秩”，不能继续把低于输入精度的奇异值当有效维度。

### 1.2 R2-D1

- `demo_t16_operator/configs/lgwo_a24_r2d1_budget_allocation_diagnosis_v1.json`
- `site_tools/run_lgwo_a24_r2d1_budget_allocation_diagnosis.py`
- `demo_t16_operator/results/lgwo_a24_r2d1_budget_allocation_diagnosis_v1/report.json`
- `demo_t16_operator/results/lgwo_a24_r2d1_budget_allocation_diagnosis_v1/independent_validation.json`
- `docs/lgwo_a24_r2d1_edge_budget_allocation_no_go_2026-07-21.md`

需要继承的教训：

1. R2-D0 因 `denominator_floor` 漂移、缺 matched `rho=0` 控制、伴随缺陷只记录不拒绝、缺 front/heldout 门而被正确判 invalid。R2-F0 的任何一项 baseline、projection metric 或 gate 漂移都应同样 fail closed。
2. R2-D1 表明 edge 项相对纯数据控制确有局部机制信号，但替换 H1 迭代仍输给 H1-20；所以 R2-F0 必须同时对比“没有该结构的 matched control”和“完整 H1 continuation”。
3. R2-D1 手工 `BOUND_SOURCES` 漏掉提供 `_gain/_relative_l2/_synchronize` 的 helper。R2-F0 不能再依赖手写的不完整源码列表。
4. R2-D1 的 `20F/20A^T` 只证明物理 API 次数相同，不证明 wall time、局部梯度计算、内存或共享缓存成本相同；R2-F0 的预算更不对称，必须拆开记录。

## 2. 必须先冻结的数学对象

令加权数据算子为 `A_w`，support-internal difference 为 `D`，冻结 H1 正则权重为
`lambda_H1=1e-3`。H1 使用的 augmented operator 是

```text
G x = [ A_w x ; sqrt(lambda_H1) D x ].
```

独立 H1-20 cache 保存：

```text
P20 = [p_1, ..., p_20]              field directions
Z20 = G P20                         augmented projected directions
x20 = P20 c20
r20_data = y_w - A_w x20
r20_aug  = [y_w; 0] - G x20.
```

三个原始方向只能读取部署时可见量：

```text
d_R = A_w^T r20_data
d_H = - D^T psi_delta(D x20)
d_E = M_edge(x20) elementwise-multiply d_R.
```

还应增加一个不参与“新方法”排名、只作归因与实现校验的 quadratic-prior control：

```text
d_Q = - D^T D x20.
```

`Q` 与 `H` 具有相同的物理调用结构；而 `d_R + lambda_H1 d_Q` 正是 H1 augmented normal 的未投影形式。

其中 `M_edge` 的确切公式目前尚未给出，这是冻结阻断项。至少要预写：差分定义、是否平滑、scale
估计、阈值/分位数、gate floor/ceiling、指数、support 边界、零 scale 回退和所有常数。不得在看到
R2-F0 truth 指标后再改。

### 2.1 “投影出旧 span”不能只写一句话

H1 的递推是在 augmented measurement metric 中正交化，R2-F0 的主 projector 应与它一致：

```text
z_j       = G d_j
a_j       = argmin_a ||z_j - Z20 a||_2
q_j       = d_j - P20 a_j
z_j_perp  = G q_j = z_j - Z20 a_j.
```

求解必须使用 CPU float64 SVD/QR，而不是在 MPS float32 上用连续 dot product 猜 rank。每个方向同时报告：

```text
eta_aug   = ||G q_j||_2 / max(||G d_j||_2, eps)
eta_field = ||q_j||_2   / max(||d_j||_2, eps)
eta_data  = ||A_w q_j||_2 / max(||A_w d_j||_2, eps)
sense_j   = ||A_w q_j||_2 / max(||q_j||_2, eps).
```

只报告 `eta_field` 不够：一个方向可以在 field space 明显跳出 `P20`，却几乎落在 `A_w` 的零空间。
truth oracle 能沿这种方向显著靠近 synthetic truth，同时几乎不改变 residual；这说明“truth 知道该怎么填
nullspace”，不能说明部署算法知道。

### 2.2 union 必须按子空间而不是按方向顺序定义

`RH / RE / HE / RHE` 中，各方向先投影旧 span 后仍可能高度相关。不能按 R、H、E 的任意顺序做一次
Gram-Schmidt，然后把 nominal direction count 当 rank。建议：

1. 将所有 `G q_j` 归一化后组成小矩阵；
2. CPU float64 SVD 或 rank-revealing QR 一次确定 union projector；
3. 以固定 tie-break 选择符号/列顺序，只用于保存，不影响 span；
4. 对 R/H/E 的所有排列做单元测试，principal angles、effective rank 和 oracle 指标必须不变；
5. 相关方向合并为一个有效维度，不得按两个方向收费后又声称“新增两维”。

## 3. P0 冻结阻断项

| ID | 风险 | 为什么会使结论无效 | 冻结前修复 |
|---|---|---|---|
| P0-1 | candidate oracle 对 solver endpoint | candidate 使用 truth 重新优化全部 21--23 个系数，而普通 H1-k 只使用迭代系数；任何收益可能来自 oracle coefficient fitting，而不是 R/H/E | 对每个 `k=21,22,23` 同时构造 H1-k native endpoint 和 H1-k span truth joint oracle；candidate 必须对比同 objective、同 ray、同 trust gate 的 matched H1-k span oracle |
| P0-2 | `(F,A^T)` 被压成单个“逻辑调用数” | H 单方向是 `21F/20A^T`，pair/triple 是 `22/21`、`23/21`；它们并非 H1-21/22/23 的 equal-call 方法 | 冻结逐阶段 call vector，区分 standalone logical cost、sweep-amortized cost、evaluator cost和 data-generation cost；只能称 matched H1 为 same-F/nominal-dimension strong control，除非另有实测 cost matching |
| P0-3 | old-span metric、rank 和 near-null 未定义 | field-outside 但 data-near-null 的方向可被 truth oracle 放大，residual-safe 仍可能全放行；相关 union 也可产生伪新增秩 | 主投影用 `G=[A_w;sqrt(lambda)D]` metric；同时报告 field/data/augmented novelty、sensitivity、singular spectrum、effective rank、pairwise cosine 和 principal angles；加 field trust 与 rank floor |
| P0-4 | truth 隔离与 post-open family 选择 | 方向公式不看 truth，不代表整条 pipeline 不看 truth；joint coefficients、family winner、ray endpoint、field/gradient gate 都看 truth | direction builder 只收 whitelist packet；先保存并 hash direction packet，再打开 oracle packet；7 个 family 全量报告且不产生 deployable winner；任何 future family 选择都标成 post-open development decision |
| P0-5 | source/validator 未 pre-bind | R2-D1 已实证手工源码闭包会漏 helper；R2-E0 validator 是事后实现 | runner、core、config、validator、prior evidence、108 geometry inputs 与动态/非 Python 依赖全部在 HEAD；validator 在正式运行前实现、提交、测试并由 config 绑定；运行后不得补 validator 再称 contemporaneous proof |

### 3.1 P0-1：正确的 comparator ladder

每个 direction family `S`、`k=|S|` 至少需要以下 6 层，且全部使用相同 field/gradient 定义：

| 层 | 方法 | truth 使用 | 作用 |
|---|---|---|---|
| C0 | H1-20 native | evaluator only | 公共起点 |
| C1 | old P20 joint oracle + safe ray | coefficients + evaluator | 复现 R2-E0 式 old-span 上界，隔离“仅重配旧系数” |
| C2 | H1-(20+k) native | evaluator only | 同 forward count/名义维数的 solver continuation |
| C3 | H1-(20+k) span joint oracle + safe ray | coefficients + evaluator | **表示能力的 matched oracle control** |
| C4 | `span(P20,Q_S)` unconstrained joint oracle | coefficients + evaluator | 只显示理论方向，不可部署 |
| C5 | `span(P20,Q_S)` residual+field-trust safe joint oracle | coefficients + evaluator | R2-F0 唯一可用于 representation gate 的候选 |

若只比较 C5 与 C2，结论最多是“truth 重新配系数胜过 H1 普通迭代”，不能归因于新方向。R2-F0 真正要问的是
C5 是否稳定胜过 C3，并且 C5-C1 的增量确实来自 `Q_S`。

另加一个不进入候选池的 **H1-next identity control**：

```text
d_Q = A_w^T r20_data - lambda_H1 D^T D x20.
```

按相同 augmented projector 投影 `d_R + lambda_H1 d_Q` 后，它应与独立 H1-21 的第 21 个方向一致。这个 positive control
能同时抓出 residual sign、double weighting、support、lambda、projector 和 reorthogonalization 漂移。若该控制不过门，
整个 R2-F0 必须判 invalid，不能把 R/H/E 的结果解释为科学 NO-GO。

### 3.2 P0-2：冻结调用矩阵

以下是按拟议公式得到的 **standalone logical solve cost**。H1-20 cache 的 `20F/20A^T` 必须计入每个方法；
reduced oracle coefficient solve 应为 `0F/0A^T`。表中尚不含统一 evaluator `+1F/0A^T`、data generation、
warm-up 与独立 reference run。

| 方法 | 新方向 | 新方向共享 | 新增 `(F,A^T)` | 总 solve `(F,A^T)` | matched H1 | 公平性限定 |
|---|---:|---|---:|---:|---|---|
| H1-20 | 0 | -- | `(0,0)` | `(20,20)` | H1-20 | baseline |
| R | 1 | -- | `(1,1)` | `(21,21)` | H1-21 `(21,21)` | 真正 call-vector matched |
| H | 1 | -- | `(1,0)` | `(21,20)` | H1-21 `(21,21)` | H1 多 1 次 `A^T`，只能称 strong control |
| Q control | 1 | quadratic-H1 prior attribution only | `(1,0)` | `(21,20)` | H1-21 `(21,21)` | 与 H 同 call vector；不得列作新方向胜利 |
| E | 1 | 复用一次 `d_R` backprojection | `(1,1)` | `(21,21)` | H1-21 `(21,21)` | call-vector matched；local gate 成本另报 |
| H1-next identity | 1 | 先组合 `d_R+lambda_H1 d_Q`，再投影一次 | `(1,1)` | `(21,21)` | H1-21 `(21,21)` | 必须等价；不是候选 |
| RH | 2 | R 的 backprojection 只算一次 | `(2,1)` | `(22,21)` | H1-22 `(22,22)` | H1 多 1 次 `A^T` |
| RE | 2 | R/E 共享 backprojection | `(2,1)` | `(22,21)` | H1-22 `(22,22)` | H1 多 1 次 `A^T` |
| HE | 2 | E 使用一次 backprojection | `(2,1)` | `(22,21)` | H1-22 `(22,22)` | H1 多 1 次 `A^T` |
| RHE | 3 | R/E 共享 backprojection | `(3,1)` | `(23,21)` | H1-23 `(23,23)` | H1 多 2 次 `A^T` |

必须同时保存两本账：

1. **standalone ledger**：如果部署时只运行该 family，它实际依赖多少 `F/A^T`；
2. **sweep ledger**：研究代码为 7 个 family 共用 H1 cache、`d_R` 和方向投影后，整个 sweep 实际调用多少。

跨 arm 共享不能把某个 arm 的 standalone 成本写成 0。H1-21/22/23 若从一次 H1-23 run 的 prefixes 提取，也要同时报告
“sweep 实际 23/23”和“每个 control 的 logical prefix k/k”。任何“同预算”文字都必须明确是 same-F、same-vector、
measured-time 还是 amortized-sweep。

### 3.3 P0-4：结构性 truth firewall

仅在 report 里写 `uses_truth=false` 不够。建议形成两个不可混用的数据类/文件：

```text
DirectionPacket (truth forbidden)
  x20, cached residual, P20, Z20, support, spacing,
  observed y_w, view mask, whitening/noise factors,
  bound geometry summaries, frozen constants, operator handle + call ledger

OraclePacket (opened only after DirectionPacket hash is frozen)
  synthetic truth, truth gradient, clean projection for evaluator,
  frozen direction hashes, common joint-objective constants
```

Direction builder 禁止接收：`truth`、`clean`、field/gradient/front metrics、heldout-clean residual、family、field seed、noise
seed、带有 `plume/thin/shock/...` 语义的 case ID、oracle callback 或整个 runner locals。可以记录 opaque case hash，但不得据此分支。

特别注意：

- 使用 `r20_data` 是允许的，因为它来自 observation；
- 使用 fresh recomputed residual 构方向会额外消耗 `1F`，必须计入 solve ledger；若想保持上表预算，只能使用 cached residual，fresh `1F` 只能事后核验，核验失败则整 case invalid；
- residual-safe ray 的 endpoint 已由 truth oracle 决定，因此即使 ray length 只读 observed residual，它仍是 truth-conditioned oracle，不能改名为 deployable candidate；
- 7 个 family 可逐一通过/失败，但不得按这 6 个 truth cases 选一个“truth-free winner”。方向生成 truth-free 与 family selection truth-free 是两件事。

## 4. P1 高风险项

| ID | 风险 | 必须增加的控制 |
|---|---|---|
| P1-1 | residual-safe 但 field-unbounded | 同时输出 unconstrained、residual-only、residual+field-trust 三层 oracle；只有第三层进入 representation gate |
| P1-2 | cached residual 漂移或 weight double application | 明确 `r_w=y_w-A_wx`，调用 weighted operator 的 adjoint；每个终点用独立 raw `1F` 重算 residual，saved-vs-recomputed 不过门即 invalid |
| P1-3 | MPS float32 把数值噪声判成新增秩 | projector/SVD/oracle/ray 全部 CPU float64；rank floor 与 sensitivity sweep 高于 float32 输入噪声；禁止 MPS float64 请求 |
| P1-4 | H1-20 起点或 matched prefixes 漂移 | 独立 frozen reference H1-20、candidate cache H1-20、H1-23 prefix@20 三者互验；H1-next identity control 必须通过 |
| P1-5 | edge gate 仍有隐含自由度 | gate 公式、delta、scale、threshold、clip、support 与 fallback 全写入 config；不允许参数网格，不允许运行后修订同一 output ID |
| P1-6 | union 顺序依赖/相关方向双计 | CPU rank-revealing factorization；排列不变测试；pairwise cosine、principal angles、有效秩和 rejected direction 原样输出 |
| P1-7 | matched H1 物理调用多但 local 成本少 | 除 `(F,A^T)` 外，分开记录 cache、direction construction、projection、CPU transfer/SVD、oracle、evaluator 的 wall time 与 peak RSS；随机化计时顺序只作性能描述，不改变科学结果 |
| P1-8 | 6 cases 上 7 family 的 winner's curse | 全矩形输出、逐 case 输出、无删除失败方向；不报 p-value/置信成功；任何 best family 只标 post-open development selection |
| P1-9 | only-mean 门掩盖逐 case 伤害 | residual gate 必须逐 case；field/gradient 同时报告 mean、median、worst、harm count；9-view 无 heldout camera 的 case 不得伪造 heldout 值 |
| P1-10 | 保存证据不足以独立复算 projector/oracle | 输出精简 audit tensor packet 或等价的可重算矩阵，而不只存 aggregate/PNG/hash；validator 必须能独立复算 rank、outside fraction、oracle、ray 和 metrics |

## 5. 建议的数值与结构门槛

这些是冻结建议，不是已通过结果。若要修改，应在第一次正式 R2-F0 run 前提交新的红队回应，不能看完 truth 后静默调整。

### 5.1 基线、算子与 residual 合同

| 检查 | 建议门槛 | 失败状态 |
|---|---:|---|
| candidate cache vs independent frozen H1-20 field relative difference | `<= 2e-6` | whole-run invalid |
| candidate cache vs independent frozen H1-20 augmented residual difference | `<= 2e-6` | whole-run invalid |
| H1-23 prefix@20 vs candidate H1-20 field/residual | 各 `<= 2e-6` | whole-run invalid |
| H1-next identity field-direction cosine | `>= 0.99999` | whole-run invalid |
| H1-next normalized augmented-projection relative difference | `<= 2e-5` | whole-run invalid |
| cached vs fresh weighted residual relative difference | `<= 3e-5` | case invalid；任一 case invalid 则不作科学判决 |
| weighted operator adjoint-identity relative defect | `<= 5e-5` | whole-run invalid |
| old normalized augmented Gram max defect | `<= 5e-5` | whole-run invalid |
| reduced/oracle physical calls | 精确 `0F/0A^T` | whole-run invalid |
| all tensors、coefficients、metrics、histories | 全 finite | whole-run invalid |

### 5.2 outside-span、相关性与 rank

| 检查 | 建议门槛 | 解释 |
|---|---:|---|
| raw direction `||d_j||` 与 `||Gd_j||` | 大于 frozen absolute floor；floor 随 dtype/shape 写入 config | 防止零方向除法 |
| `eta_aug` 非平凡下限 | 至少 5/6 cases `>= 1e-3`，case median `>= 1e-2` | 小于此量级只算 near-old-span |
| `eta_field` 非平凡下限 | 至少 5/6 cases `>= 1e-3` | 防止 field projection 数值残差 |
| normalized new directions pairwise augmented cosine | `abs(cos) <= 0.995` 才可算两个有效方向 | 更相关时 union effective rank 不得加 2 |
| SVD relative rank floor | 主分析 `1e-6` | 输入来自 float32；不再把 `1e-12` 当科学秩 |
| smallest accepted singular value / largest | `>= 1e-4` | 低于此值只作 ill-conditioned diagnostic |
| rank sensitivity | floor=`1e-5,1e-6,1e-7` 时 effective rank 与最终状态不变 | 否则判 numerical ambiguity |
| union permutation invariance | principal-angle max 与 oracle metrics 在预写容差内一致 | 防止方向顺序造结果 |

`1e-3/1e-2/1e-4` 是“机制 triage 不被 float32 尾数支配”的保守建议，不是一般数学定理。所有原始 singular values、
fractions 和 failed cases 仍必须输出，不能只存 pass/fail。

### 5.3 residual-safe 与 field-trust oracle

对每个 span 先求固定 joint objective 的 truth oracle endpoint `c*`，再从 H1-20 系数 `c0` 沿
`c(t)=c0+t(c*-c0)` 限制：

```text
||r_data(t)||_2 / ||r_data(0)||_2 <= 1.02,   0 <= t <= 1
||x(t)-x20||_2 / ||x20||_2 <= 0.25.
```

建议直接在 CPU float64 解 residual 二次不等式的最大可行根，并用独立 bisection 交叉验证。不能只依赖缓存投影；最终
必须再用统一 evaluator `1F` 验证真实 weighted residual。门槛是：

- 每个 case 的 fresh-evaluated residual ratio `<= 1.02 + 3e-5`；
- 每个 case 的 field shift `<= 0.25 + 1e-6`；
- 若 `x20` 范数接近零，使用 config 中预写的绝对 radius，不能按 case 临时决定；
- residual-only oracle 可保留作“near-null amplification”诊断，但不得进入 representation PASS。

joint objective 必须在 config 中写成完整公式。建议沿用并显式写出 truth-conditioned normalization：

```text
J_truth(x) = ||x-x_truth||_2^2 / max(||x20-x_truth||_2^2, eps)
           + ||D x-D x_truth||_2^2 / max(||D x20-D x_truth||_2^2, eps).
```

它逐 case 使用 truth 归一化，所以只能是 evaluator oracle。所有 C1/C3/C4/C5 必须使用同一公式、同一 SVD floor、
同一 ray 和同一 trust gate；不得给 candidate 与 H1 control 使用不同 objective。

### 5.4 representation signal 建议门

任一 family 只有同时满足下列条件，状态才可写成
`POSTOPEN_R2F0_REPRESENTATION_SIGNAL_ONLY_NO_AUTHORIZATION`：

1. 所有 P0、数值、cache、call-ledger 与 effective-rank 门通过；
2. C5 相对 matched C3 的 mean field gain `>= 5%` 且 mean gradient gain `>= 5%`；
3. C5 相对 C1 的新增 mean field/gradient gain 各 `>= 3` 个百分点，证明收益不是旧 span coefficient re-fit；
4. field 与 gradient 的 median gain 各 `>= 3%`；
5. field 与 gradient 的 worst-case gain各 `>= -1%`，各自 harm count `<= 1/6`；
6. 对有 heldout camera 的 5 个 cases，mean heldout-clean ratio vs matched C3 `<= 1.02`，worst `<= 1.05`；
7. 每 case fresh weighted residual 与 field trust 全通过；
8. 对 union，必须比其最佳 singleton/pair 的 field 与 gradient gain都多至少 `1` 个百分点，并确实增加 effective rank，才可称“union synergy”；否则只能称 directions pooled；
9. edge-gated family 若不胜过同调用数 raw-R control，不得把增益归因于 edge gate；
10. Huber family 若不胜过 negative quadratic-H1-gradient control，不得把增益归因于 Huber prior。

这些门只用于在已打开 6 cases 上决定“expanded span 是否还有足够大的表示上界值得继续”。即使全部通过，也不能：

- 选择网络结构或训练 learner；
- 声称未见 seed/rig 泛化；
- 声称真实 BOST 或反应流物理有效；
- 声称击败 DeepONet/FNO/FFNO/NeRIF/TDBOST；
- 声称新算法或论文成功。

## 6. 选择偏差与结果矩形

当前 3 singleton + 3 pair + 1 triple 共 7 个 candidate spans。每个 span 至少有 C4/C5 两个 truth oracle，另有
C0/C1 和对应 C2/C3 controls。必须预先定义完整 method rectangle，运行中不得删掉 rank-degenerate、负增益、ray fraction=0
或 MPS 边界 cases。

建议报告顺序固定为：

```text
H1-20 native
oldspan oracle / oldspan safe oracle
H1-21 native / H1-21 oracle / H1-21 safe oracle
R, H, E each: unconstrained / residual-only / residual+field-safe
H1-22 native / H1-22 oracle / H1-22 safe oracle
RH, RE, HE each: unconstrained / residual-only / residual+field-safe
H1-23 native / H1-23 oracle / H1-23 safe oracle
RHE: unconstrained / residual-only / residual+field-safe
identity and rank controls
```

不得先按 truth 选 `best family` 再只画它。主表应同时给出 6 个逐 case 值、mean、median、worst、harm count、safe fraction、
effective rank、outside fractions、call vector 和 comparator。图可以突出一个用于讲解的 family，但标题必须写
`post-open illustrative row`，不能写 selected algorithm。

## 7. 调用账本的最小 schema

每个 case/method 至少保存：

```text
data_generation_forward_calls
warmup_forward_calls / warmup_adjoint_calls
reference_h1_forward_calls / reference_h1_adjoint_calls
candidate_cache_forward_calls / candidate_cache_adjoint_calls
direction_raw_backprojection_forward_calls / adjoint_calls
direction_huber_forward_calls / adjoint_calls
direction_edge_forward_calls / adjoint_calls
reduced_oracle_forward_calls / adjoint_calls
evaluator_forward_calls / adjoint_calls
standalone_logical_forward_calls / adjoint_calls
sweep_actual_forward_calls / adjoint_calls
```

同时记录：

- `d_R` 在 R/E/union 中的 tensor hash 必须相同，证明真正共享的是同一个 backprojection；
- 每个 `F/A^T` 前后 wrapper ledger delta，不接受事后按公式填常数；
- local `D/D^T`、edge gate、CPU transfer、SVD、oracle 与 plotting 不计入物理 call，但要分阶段计时；
- evaluator call 对所有方法一致，且不反馈给 direction/oracle construction；
- data generation 的 `A(truth)` 和 MPS warmup 不得混入 solve budget，也不得被省略不报。

若 wrapper 只统计 base physical operator，报告中要明确 H1 local regularization rows 不增加 physical calls，但会增加 local compute。

## 8. MPS 与数值红队

### 8.1 必须复用的安全传输模式

```text
MPS float32 tensor
  -> CPU source-dtype copy
  -> CPU float64 promotion
  -> finite check
  -> SVD/QR/oracle/ray
  -> CPU target-dtype cast
  -> target-device transfer
  -> finite check.
```

禁止任何直接请求 MPS float64 的 `.to(device="mps", dtype=torch.float64)` 或等价路径。以下定向回归必须在 formal run
前通过：

1. singleton projector 的完整 MPS round trip；
2. 3-direction union SVD 与 permutation invariance；
3. matched H1-k span oracle；
4. residual quadratic root 与 bisection cross-check；
5. field reconstruction 和 independent fresh residual evaluator；
6. rank-degenerate、zero-direction、near-null、non-finite 与 denominator-floor fail-closed paths。

### 8.2 不允许用极小正 gain 掩盖数值尺度

- 所有 gain 同时报告绝对 error difference；
- 小于 cache/evaluator reproducibility envelope 的 gain 标 `NUMERICALLY_UNRESOLVED`，不得计 pass；
- direction combination 在 CPU float64 计算一份 reference，再与 MPS float32 field 比较；
- union 列顺序变化若让结论改变，状态必须是 `NUMERICAL_AMBIGUITY_NO_DECISION`；
- 不能因一个 SVD floor 给出好结果就只报该 floor。

## 9. source binding 与原子运行

### 9.1 运行前 source/data closure

R2-F0 至少要绑定：

1. frozen config 的 HEAD blob；
2. runner 的完整本地 Python import closure；
3. runtime 实际加载的每个 local module 的 `__file__` 与 SHA-256；
4. dynamic import、relative import、package `__init__.py` 和非 Python 资源；
5. R2-E0 v3 config/report/validation、R2-D1 config/report/validation，以及相关 invalid-attempt/notice；
6. 90 个 geometry arrays + 18 个 manifests 的逐文件 hash 和 canonical bundle digest；
7. direction formula constants、metric definitions、method rectangle、decision table 与 validator hash；
8. Python、PyTorch、NumPy、OS、device 与关键环境变量 provenance。

仅做 AST import closure 仍不足以覆盖动态导入和运行时打开的资源；应把“静态 closure”和“runtime loaded/consumed closure”都存下。
任何 source/config/geometry drift 都应在创建 canonical result 前失败。

### 9.2 原子输出

建议使用：

```text
output_final must not exist
run into output_tmp_<nonce>
preflight -> execute -> postflight source/geometry recheck -> self-validation
only then atomic rename tmp -> final
```

失败时只写到另一个版本化 `invalid_attempt.json`，并保存异常阶段、source head、config hash、已观察行数和
`rerun_same_config_authorized=false`。不得像普通脚本那样先写 CSV/PNG，再在最后发现 source drift 后留下看似完整的结果目录。

## 10. independent validator 风险与要求

R2-F0 的 validator 必须在 formal run 前实现并提交，但仍要明确：预先实现只证明约束先存在，不自动证明它真的独立。
最低要求：

1. 不 import R2-F0 runner、projector、oracle、aggregation 或 selection core；
2. 从保存的 audit tensor packet 独立复算 P20/Q、augmented/data projections、outside fractions、singular values、effective rank、principal angles、joint coefficients、safe-ray root、field/gradient metrics和所有 gates；
3. 独立复算 7-family 完整矩形、matched H1 mapping 与 call vectors；
4. 校验 source/config/validator HEAD blobs、108 geometry inputs、prior evidence digests 和 output manifests；
5. 检查 DirectionPacket hash 在 oracle 开启前后不变，truth 字段不在 builder 输入或 saved direction provenance 中；
6. 检查 claim boundary 全 false，且 PASS status 仍包含 `POSTOPEN_INTERNAL_CONSISTENCY_ONLY` 限定；
7. tamper tests 至少覆盖：改一行 metric、删一个 case、交换 method label、伪造一个 call、改 rank floor、改 direction hash、把 truth flag 改 false、删一个 geometry manifest、改 source blob、破坏 PNG/NPZ；
8. validator 输出 check count，但页面/论文不得把“检查数很多”写成科学成功。

若只保存 aggregate CSV、report 和 PNG，validator 无法独立证明 projector、rank 或 oracle 系数正确；它最多能复算汇总算术。
因此正式 result packet 应包含足以重算的小矩阵/张量，例如：

```text
P20/Z20 hashes + normalized Gram
x20/r20
each raw d_j and projected q_j
A_w q_j and G q_j
joint design/target normalization scalars
oracle coefficients and ray coefficients
H1-k matched direction/projection packets
truth/evaluator tensor hashes or bound deterministic regeneration contract.
```

这些 synthetic audit tensors 可以压缩保存；若不保存原 tensor，必须给出另一个真正独立的 deterministic recomputation route，
并把它的额外 operator calls 标为 validation calls，而不是 formal solve calls。

## 11. 冻结前 MUST 清单

以下项目必须全部勾选，才可以把 config 状态写成 `FROZEN_BEFORE_FIRST_R2F0_RUN`：

### A. 边界与 provenance

- [ ] 固定新 experiment ID、唯一 output path，且 final output 不存在。
- [ ] 明写 6 cases 已打开、post-open、same-QMC8 inverse crime、无 fresh/real/learner authorization。
- [ ] 绑定 R2-E0、R2-D1 和 invalid-attempt/notice 的路径、状态与 SHA-256。
- [ ] 绑定 108 个几何输入和 bundle digest。
- [ ] claim boundary 中 algorithm/new algorithm/learner/real/fresh/generalization/comparator/paper/breakthrough 全为 false。

### B. 方向与 projector

- [ ] 精确冻结 R/H/E 公式和 edge gate 的全部常数/fallback。
- [ ] DirectionPacket whitelist 与 OraclePacket 隔离已由 API 和测试强制。
- [ ] 不向 builder 传 truth、clean、semantic case ID、family、seed 或 evaluator callback。
- [ ] old-span 主 metric 明定为 H1 augmented `G` metric，并同时报告 field/data novelty。
- [ ] CPU float64 rank-revealing projector、normalization、sign 和 union order contract 已冻结。
- [ ] H1-next quadratic-prior identity control 已加入，且失败会使 whole run invalid。
- [ ] R/H/E/RH/RE/HE/RHE 全部预列，无运行后增删 family 或参数 grid。

### C. comparator 与预算

- [ ] H1-20 reference、candidate H1-20 cache、H1-23 prefix@20 三者独立并有等价门。
- [ ] H1-21/22/23 native endpoints 已冻结。
- [ ] H1-21/22/23 matched span joint oracles 使用与 candidate 完全相同的 truth objective、rank、ray 与 trust gate。
- [ ] old P20 span oracle 由同一实现重新计算，不能直接拿不同代码路径的旧数字硬比较。
- [ ] 每个 family 的 standalone `(F,A^T)` 与整个 sweep actual `(F,A^T)` 都已预写。
- [ ] evaluator/data-generation/warmup/reference/validation calls 与 solve calls 分栏。
- [ ] 所有 physical call ledger 由 wrapper delta 实测，不能按 method label 填常数。

### D. oracle 与 decision gate

- [ ] joint truth objective 公式、normalization 和 epsilon 已冻结。
- [ ] unconstrained、residual-only、residual+field-safe 三层 oracle 分开命名。
- [ ] residual `<=1.02` 是逐 case fresh-evaluated gate，不是 mean-only gate。
- [ ] field relative shift `<=0.25` 进入唯一 representation PASS gate。
- [ ] C5 必须对比 C3 和 C1；5%/3pp/median/worst/harm/heldout 门已写入 config。
- [ ] family 不产生 deployable winner；decision status 只有 post-open NO-GO 或 representation-signal-only。
- [ ] union synergy 和 edge/Huber attribution controls 已冻结。

### E. 数值、源码与验证

- [ ] 所有 MPS-to-CPU-to-MPS 路径使用统一安全 helper，并有完整 MPS regression tests。
- [ ] rank floor 主值与 sensitivity values 已冻结，结论变化即 numerical ambiguity。
- [ ] finite、adjoint identity、residual consistency、orthogonality、prefix identity 均为 fail-closed。
- [ ] config、runner、core、validator 与完整静态/runtime closure 已提交在同一 HEAD。
- [ ] 108 geometry inputs 在 run 前后重验；source/config 在 local imports 前捕获并在最终 rename 前重验。
- [ ] validator 在 run 前提交，不 import runner/core，能从 audit packet 独立重算而不只是重汇总。
- [ ] validator tamper tests、runner protocol tests、MPS tests、output-absence test 全通过。
- [ ] 临时目录写入与原子 rename 已实现；失败只留下显式 invalid attempt。
- [ ] 独立红队复核 config/runner/validator 后给出 `NO_P0_P1_FREEZE_BLOCKER`，再允许一次 formal run。

## 12. 建议的决策表

| 条件 | 状态 | 能说什么 |
|---|---|---|
| source/config/geometry/cache/call/truth-firewall/identity 任一失败 | `INVALID_R2F0_PROTOCOL_OR_EXECUTION_CONTRACT` | 仅软件/协议失败；无科学结论；禁止同 output/config 静默重跑 |
| 数值结果依赖 rank floor、union 顺序或 MPS 尾数 | `R2F0_NUMERICAL_AMBIGUITY_NO_DECISION` | 当前精度不足；不是算法 NO-GO |
| 结构有效，但所有 C5 不过预写 representation 门 | `POSTOPEN_R2F0_NO_USEFUL_EXPANDED_SPAN_HEADROOM_NO_GO` | 这 3 个方向族在 6 个已见 synthetic cases 上没有足够上界；不外推 |
| 至少一个 C5 过全部门 | `POSTOPEN_R2F0_REPRESENTATION_SIGNAL_ONLY_NO_AUTHORIZATION` | 某个 truth-free direction span 有 truth-oracle 表示上界，值得另立协议；仍无算法/learner/fresh/real claim |

不存在 `R2F0_ALGORITHM_SUCCESS`、`FRESH_AUTHORIZED`、`LEARNER_AUTHORIZED` 或 `PAPER_SUCCESS` 分支。

## 13. 红队最终建议

R2-F0 可以继续，但应把目标写窄：

> 在完全保留独立 H1-20、严格记录 `(F,A^T)`、方向不接触 truth 的前提下，R/H/E 是否产生数值可分辨、
> 非退化且不靠近测量零空间的新增 span；在相同 truth joint oracle、相同 residual/field trust 和 matched
> H1-k span oracle 下，它们是否仍具有至少 5% 的联合 field/gradient 表示 headroom？

这比“新方向能不能击败 H1”更准确，因为本轮没有部署系数规则，只有 truth oracle。若上述问题回答 NO，便得到一个有价值的
方向族 NO-GO；若回答 YES，也只得到下一轮研究假设，而不是算法成功。

当前建议维持 `HOLD`。先清零 P0，尤其是 **matched H1-k span oracle、vector call ledger、augmented-rank/nullspace gate、
truth firewall、pre-run independent validator**，再冻结 config。否则任何正结果都可能只是 comparator 不对称、调用记账压缩、
truth 选择或数值近秩退化造成的表面提升。

## 突破监测

**没有突破，也没有 R2-F0 结果。** 本文新增的是冻结前的证伪合同：它指出怎样才能让未来的 expanded-span 正/负结果具备
可归因性。只要上述 P0 尚未清零，R2-F0 就不应运行正式结果，更不能进入网页的“算法进展”或论文结果段。
