# Scalar-step PDHG NO-GO 与下一轮 block-diagonal gate

日期：2026-07-17
状态：`HISTORICAL SCALAR NO-GO / FOLLOW-UP GATE A PASSED / FACTOR GATE B NO-GO`

后续更新：本文提出的 deterministic block/factor 路线已完成。Gate A mechanics
正式通过；V4 Gate B 在 source `204bbe8` 上形成 256 条方法行并由独立 validator
重算 4,048 项，但 voxel-factor 相对 scalar mean gain 仅 1.321%，八门过五门，
因此按本文第 7 节停止规则关闭，不进入 TV、warm start、nullspace 或网络。完整
结果见 [Factor Gate B NO-GO](psu_b0_factor_pdhg_gate_b_no_go_2026-07-17.md)。

## 1. 先给严格结论

v2 正式筛选得到：

```text
POSTOPEN_PDHG_SCALE_NO_GO
```

它只否掉当前冻结的 scalar-step、zero-initialized、`K={4,8,16,32}`、
`alpha={1/256,1/64,1/16,1/4}` TV/Huber 网格。它没有否掉：

- PDHG 一般有效性；
- TV/Huber 在 BOST 上的一般有效性；
- diagonal/block-preconditioned PDHG；
- warm-started primal-dual；
- fresh camera/session 或真实 OERF 数据上的方法。

排名第一的 `pdhg_huber_a1of256_k4` 只是 32 个失败候选中的排序第一，不是通过
候选。它相对同预算 graph-PCGLS：

| 指标 | 观测 | 冻结门 | 结果 |
|---|---:|---:|---|
| mean field gain | -68.432% | >= +1% | FAIL |
| field gain p10 | -120.638% | >= -1% | FAIL |
| field harm rate | 16/16 | <= 1/16 | FAIL |
| worst field gain | -140.923% | >= -3% | FAIL |
| mean gradient gain | -31.464% | >= 0 | FAIL |
| mean front gain | -0.2201 | >= 0 | FAIL |
| replicate 0 mean | -59.190% | > 0 | FAIL |
| replicate 8 mean | -77.675% | > 0 | FAIL |
| median paired wall-time ratio | 1.207 | <= 3 | PASS |

完整公开聚合结果位于
`demo_t16_operator/results/psu_b0_pdhg_scale_smoke_v2_public/`。原始 784-row
bundle、JUnit、环境和私有 geometry 路径只保存在本地私有审计库。

## 2. 为什么主因是 conditioning，而不是“alpha 不够好”

### 2.1 data-only 控制几乎停在零场

| K | data-only PDHG field-L2 | graph-PCGLS field-L2 |
|---:|---:|---:|
| 4 | 0.999644 | 0.628707 |
| 8 | 0.999121 | 0.549110 |
| 16 | 0.998029 | 0.463761 |
| 32 | 0.995881 | 0.421089 |

零初始化的 relative field error 约为 1。PDHG 到 32 次仍为 0.9959，说明首先
失败的是 data-fitting 推进速度；不能先把责任归给 TV/Huber 形状。

### 2.2 正则化对 data-only 的影响小且全部为负

32 个正式候选相对各自同 K data-only 控制的最好 mean gain 为约
`-0.000003%`，最差为 `-0.100092%`。alpha 跨 64 倍也没有形成正向信号。

这支持“当前 K 内 regularization-inactive”的诊断，但不区分两种可能：

1. scalar primal step 太小，体场尚未进入先验可发挥作用的区域；
2. `alpha / n_D` 的当前尺度在该 normalization 下不合适。

因为 data-only 已经几乎不动，第一种是更优先的可证伪假设。

### 2.3 data 与 spatial block 的尺度严重失衡

正式 norm audit：

| replicate | data norm squared estimate | gradient norm squared | 比值 | root-scale 比值 |
|---:|---:|---:|---:|---:|
| 0 | 2.775668 | 78,759.94 | 28,375 | 168.4 |
| 8 | 2.106601 | 78,619.85 | 37,320 | 193.2 |

`D` 使用米制网格间距，故其范数自然远大于 normalized measurement block。统一
scalar metric 会让 data block 也继承 spatial block 的最坏尺度。上表中的 power
estimate 不是 certified upper bound，因此这里只把它用作 conditioning 诊断，
不写成收敛定理。

## 3. 下一候选只回答一个问题

下一算法暂称：

```text
CA-SFM-BPDHG
Covariance-Aware Signed Factor-Majorized Block-Diagonal PDHG
```

它不是当前可宣称的创新。第一轮只回答：

> 在不改变 objective、数据、候选 budget 和物理调用账本的情况下，安全的
> block/diagonal metric 能否让 data-only PDHG 离开零场附近，并缩小与
> graph-PCGLS 的收敛差距？

signed factor majorizer 的条件、证明和 prior-art 边界已经在
[covariance_majorized_pdhg_design_2026-07-17.md](covariance_majorized_pdhg_design_2026-07-17.md)
与
[pdhg_primary_sources_and_innovation_boundary_2026-07-17.md](pdhg_primary_sources_and_innovation_boundary_2026-07-17.md)
中冻结。这里不重复把 Pock-Chambolle diagonal preconditioning 写成新贡献。

## 4. Block metric 与更新式

在活动 support/gauge 空间写：

```text
K z = [A_1 z; ...; A_B z; D z]
```

其中 `A_b` 是完整 covariance-connected measurement block，`D` 是
forward-Neumann 3D prior gradient。构造非负逐元素 majorizer：

```text
M_b = |W_b| P |G_c| E >= |A_b|
N   = |D_+| E            >= |D|
```

定义：

```text
omega_col = sum_b M_b^T 1 + N^T 1
rho_A,b   = max_i (M_b 1)_i
rho_D,v   = max_c (N 1)_(v,c)

tau_j     = eta / omega_col[j]
sigma_A,b = eta / rho_A,b
sigma_D,v = eta / rho_D,v
```

同一 covariance block 的 data dual rows 共享 `sigma_A,b`；同一 isotropic-TV site
的三个方向共享 `sigma_D,v`，否则原球投影不再是对应 metric 下的 prox。

更新保持原 objective：

```text
p_b+ = (p_b + sigma_A,b * (A_b z_bar - b_b)) / (1 + sigma_A,b)

v_v  = q_v + sigma_D,v * (D z_bar)_v
TV:      q_v+ = v_v / max(1, ||v_v||_2 / lambda)
Huber:   u_v  = v_v / (1 + sigma_D,v * delta / lambda)
         q_v+ = u_v / max(1, ||u_v||_2 / lambda)

z+     = P_C[z - T(sum_b A_b^T p_b+ + D^T q+)]
z_bar+ = z+ + (z+ - z)
```

若 factor dominance 与零耦合消元成立，则可审计
`||Sigma^(1/2) K T^(1/2)||^2 <= eta^2 < 1`。这个结论只在全部前提逐项通过时
允许引用。

## 5. 分阶段实验，禁止一口气堆组合

### Gate A：majorizer 与实现门

先不计算任何 truth performance：

当前只完成了独立于正式 runner 的 **Gate A0 CPU 结构原型**：signed factor 的
block-norm majorizer、正步长校验、一步 PDHG plumbing 与 10 项 CPU 测试。默认
拒绝 matrix-free power estimate；只有显式提供声明的 factor norm bound 才能直接
生成更新参数。该原型还没有接入 BOST 的逐元素
`|W| P |G_c| E` / `|D_+| E` majorizer，也没有通过下面的正式 Gate A，因此不能
写成“块预条件算法已实现”或“收敛已证明”。

1. tiny dense system 逐元素验证 `M_b >= |A_b|`、`N >= |D|`；
2. matrix-free row/column sums 与 dense oracle 一致；
3. `A/A^T`、`D/D^T` 继续通过精确伴随；
4. camera block、TV site、零行、零列、support/gauge 全部 fail-closed；
5. 随机探针验证 diagonal metric norm 小于 1；
6. setup calls、cache bytes、wall time 与 solver calls 分账；
7. 每 iteration 仍为 `1F + 1A + 1D + 1D^T`。

任一失败：停止，不打开 performance。

### Gate B：data-only conditioning smoke

只比较：

- scalar data-only PDHG；
- block-diagonal data-only PDHG；
- 同 `K={4,8,16,32}` 的 graph-PCGLS。

这是已见 E2 上的 mechanism gate。为防止继续在零场附近浪费搜索，block method
至少需要：

1. K=32 mean field-L2 相对 scalar data-only 降低 >= 25%；
2. 两个 replicate 的 mean 都改善；
3. 16 个场中至少 12 个改善，worst degradation 不超过 3%；
4. K=4/8/16/32 的 mean error 单调不增；
5. wall time 不超过同 K graph-PCGLS 的 3 倍。

这个 25% 是看到 v2 NO-GO 后设定的 development mechanism threshold，不是 fresh
confirmatory threshold，必须在报告中注明。

未过 Gate B：停止 block-PDHG，不加入 TV、warm start、nullspace 或网络。

### Gate C：TV/Huber activation

只有 Gate B 通过才允许加入正则化。第一轮只用两个预先冻结尺度，不做大网格：

```text
alpha in {alpha_low, alpha_high}
penalty in {TV, Huber}
K in {8, 16, 32}
```

alpha 的定义必须先用 objective dimensional analysis 或不看 truth 的 discrepancy
尺度冻结，不能从 16 个 field-L2 中挑。相对同 K block data-only，至少要求：

- mean field gain >= +0.5%；
- p10 >= -1%；
- harm <= 1/16；
- mean gradient 与 mean front 均不为负；
- 每个 replicate mean 都为正。

未过 Gate C：保留 block data-only 结果，关闭 TV/Huber 分支。

### Gate D：graph-PCGLS warm start

只有 block method 已明显离开零场、但低预算仍落后 graph-PCGLS 时才测试 warm
start。总物理预算必须写成：

```text
B_total = S_graph-PCGLS + K_PDHG
```

建议只用 `(S,K)={(2,6),(4,12),(4,28)}`，分别对齐总预算 8、16、32 的
graph-PCGLS。warm start 不免费，不能把其 calls 隐藏在 setup。

### Gate E：geometry-only nullspace

只有 Gate C/D 暴露出可重复的结构坏尾才加入近零空间 quadratic block：

```text
0.5 * beta_N * ||Z^T z||^2
```

`Z` 只能由 geometry/operator 与固定随机种子构造，不能读取 truth、morphology 或
validation 标签。必须单列 `||A Z||`、basis setup calls 和额外内存。若固定
`Z` 不改善坏尾，nullspace 分支停止。

## 6. 算子学习何时解锁

learned component 不能替代 Gate A-C。只有 classical block metric 已通过且留下
以下可观测残差之一，才允许学习：

1. bounded correction to `log tau/log sigma`；
2. graph-PCGLS warm-start stage selector；
3. geometry-only nullspace coefficient shrinkage；
4. averaged/nonexpansive learned denoiser，明确不冒充 convex prox。

网络输入只能使用部署可见的 geometry、flow-off covariance、residual trajectory、
camera/block spectrum；不得使用 clean truth-derived scale 或 morphology label。

最小 learned baseline 必须同时比较：

- formula-only diagonal PDHG；
- scalar PDHG；
- graph-PCGLS 同总预算 frontier；
- warm-start diagonal PDHG；
- learned variant 的参数/调用/时间与回退策略。

只有 fresh seed、held-out camera/session 与 flow-off-independent scale 全部通过，才
允许讨论论文级 operator learning。当前状态仍为 `neural_training_authorized=false`。

## 7. 本机与服务器门

当前 16^3 / 9-view E2 screen 全流程约 34.3 秒、峰值约 540 MB；Apple M5 / 32 GB
足够 majorizer tiny oracle、matrix-free setup 和第一轮 Gate B。暂时不租服务器。

只有出现以下任一条件再迁移：

- 64^3 以上 full detector/ray batch 无法在 32 GB 内稳定运行；
- 需要 3 seeds 以上 learned unrolling 或 4D sequence；
- 单个冻结 experiment 预计超过 12 小时；
- 需要 CUDA-only sparse/ray kernel。

## 8. 需要问何远哲的最小问题

1. 能否提供每相机同条件、未经平均的 flow-off/reference repeats，以便独立估计
   covariance 和 `scale_by_view`？
2. 组内正式 forward 是否可拆成 camera projection、ray interpolation/integration、
   physical gradient 与 whitening 因子？
3. 真实评价更重视 refractive-index/density field、front location、积分量，还是
   PIV compensation？
4. 是否有固定 geometry 下的时间序列或多工况重复，供后续 operator learning 做
   held-out session，而不是随机拆同一场？
5. 毕设阶段是否允许先交付严格 classical preconditioner，再把 learned metric 作为
   条件性扩展？

这五个回答将决定下一步是继续 CA-SFM-BPDHG、转 warm-start/nullspace，还是对齐
TDBOST 的时空低秩路线。
