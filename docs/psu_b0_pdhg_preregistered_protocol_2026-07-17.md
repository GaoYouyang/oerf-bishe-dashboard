# Covariance-weighted PDHG opened-data 尺度 smoke 预注册协议

日期：2026-07-17
状态：`PREREGISTERED_NOT_RUN`
允许证据上限：`POSTOPEN_TWO_REPLICATE_SCALE_SMOKE`

> 本文件只规定运行前的方法、候选、预算、指标和停止规则，不包含任何 PDHG
> 运行结果。任何尚未生成的数值不得回填为“预期结果”，也不得在看到结果后
> 修改本协议并仍称为预注册。

## 0. 这次实验要回答什么

前两轮审计已经固定了三个事实：

1. covariance-conditioned graph-PCGLS 相对同强度 component-PCGLS 只有约
   1.4% 的 pooled mean 增量，而且存在明显的 morphology-dependent
   semiconvergence；
2. 仅使用 pooled trajectory scalars 的早停规则不能保护 p10、harm 和 worst
   tail；
3. TV/Huber SupPCG 在同 stage 下有极小的结构信号，但扰动后的 residual
   rebuild 多用一次 forward，按总 `forward + adjoint` 调用比较时严格失败。

因此，本实验不再问“TV 能否稍微平滑体场”，而只问一个更窄、可证伪的问题：

> 在同一 full-graph covariance data metric 下，直接求解
> `data fidelity + 3D TV/Huber` 的 one-pair PDHG，能否在不超过 graph-PCGLS
> 调用预算的条件下，同时改善 field error、gradient error 和 front recovery，
> 且不恶化稀有形态尾部？

这只是两个已经打开 replicate 上的**尺度 smoke**。即使所有门都通过，也不
证明泛化、实验有效或算法优越，只能说明该数值尺度值得另写协议继续验证。

依据文档：

- `docs/psu_b0_covariance_conditioning_stopping_no_go_2026-07-17.md`
- `docs/psu_b0_edge_superiorization_budget_no_go_2026-07-17.md`
- PDHG 方法来源：[Chambolle and Pock (2011)](https://doi.org/10.1007/s10851-010-0251-1)

## 1. 数据边界与禁止事项

### 1.1 本轮唯一允许打开的数据

- replicate indices：`[0, 8]`；
- 每个 replicate 固定 8 个解析反应场代理：
  `plume`、`wavy_front`、`thin_front`、`double_front`、
  `annular_kernel`、`oblique_shock`、`vortex_pair`、`multi_plume`；
- 总计 16 个配对场；
- 使用现有真实 PSU detector/ray geometry；
- 使用现有合成 flow-on graph-correlated noise；full-graph covariance fit 按冻结
  seed/config **确定性重拟合**，并在任何 stress 或性能评分前与预运行 anchor
  逐 buffer 精确核对；
- 合成 `scale_by_view` 继承前轮 benchmark，由解析 truth 的 clean projection
  RMS 构造；这只能支持 oracle-scale mechanism diagnostic，不能当作实验部署时
  可获得的 noise calibration。真实迁移必须改用独立 flow-off/calibration；
- 所有场、噪声、mask、camera geometry 和 truth 必须按前两轮 smoke 的冻结生成
  路径确定性重建，不得为 PDHG 重新生成“更合适”的样本。旧 conditioning 产物
  未保存可反序列化 whitener，因此本轮**不声称复用了旧内存状态**。

replicate 0 与 8 早已在 SupPCG 调试中打开，因此不能再把它们称为 validation
或 blind test。16 个场也不能当成 16 个独立实验重复；报告必须分别列出
replicate 0、replicate 8 和 pooled 描述统计，不做 field-level bootstrap、
p-value 或置信区间。

### 1.2 继续封存

本协议**不授权**：

- 打开其余 opened replicates；
- 打开任何 fresh seed、fresh replicate、held-out camera 或 held-out session；
- 使用真实 temporal flow-off、OERF 私有实验场或实验三维真值；
- 训练 learned proximal、learned step size、controller、FNO、FFNO、DeepONet、
  NeRIF 或其他神经网络；
- 根据这 16 个场的 truth 做 per-field 参数选择、动态早停或 morphology selector；
- 在正式 screen 后追加一个“只差一点”的 lambda、Huber delta、迭代深度或
  step-size 网格。

若协议通过，只能授权起草下一份、更广 opened-data 的新预注册；fresh 与 neural
仍然封存。

### 1.3 输入依赖与 fail-closed SHA-256

冻结配置中的 `source_sha256` 是“源路径字段名 -> SHA-256”映射，必须包含下列
**完整且唯一**的六项外部输入；路径本身由同名顶层字段给出。摘要按文件原始
bytes 计算，不对 JSON 或 CSV 做重排、换行归一化或语义 canonicalize：

| 路径字段 / `source_sha256` key | 路径 | SHA-256 |
|---|---|---|
| `source_multiseed_config` | `demo_t16_operator/configs/psu_b0_dg_wpcgls_multiseed_v1.json` | `0c906c512fa0d7191fe83ddeb569257a8c9481c89209b9e46ffdd7614cec1452` |
| `source_smoke_config` | `demo_t16_operator/configs/psu_b0_dg_wpcgls_smoke_v1.json` | `1cd60ba7aee1f01bfec5eae4cf798a0141704355e76e830f51d43937d35d48a1` |
| `source_conditioning_report` | `demo_t16_operator/results/psu_b0_covariance_conditioned_pcgls_diagnosis/report.json` | `11de5cbeb7b20c8e93018e1981cebcc608b4c9c1d9899a7e72ba5de9898c5961` |
| `source_runtime_anchor_manifest` | `demo_t16_operator/configs/psu_b0_pdhg_runtime_anchors_v1.json` | `1e69b5208bdc8ab600d9c374205dd50d310b96cbe46ecb57e839ef13c7b73903` |
| `source_superiorization_rows` | `demo_t16_operator/results/psu_b0_edge_superiorization_scale_smoke/metric_rows.csv` | `77bea31d19e11ca1011eb431d1faaa07c535eef35df9983f7184d95c1757f843` |
| `source_superiorization_tail_rows` | `demo_t16_operator/results/psu_b0_edge_superiorization_tail_smoke/metric_rows.csv` | `4154e6a8d1cd9ef783bde01ddc0a4b310e36615ecb30ffb08321b01465c264a8` |

配置还必须包含独立对象 `geometry_audit_manifest`：

| 字段 | 冻结值 |
|---|---|
| `filename` | `all_view_geometry_audit.json` |
| `path` | `private_library/external_datasets/psu_bost_flight_body/all_view_geometry_audit/all_view_geometry_audit.json` |
| `sha256` | `3d5b19cd0a52e9706660d63102ace1a37d8eb0728e4af5cdbc7312c75fb23261` |

该 manifest 锁定此前已经完成的 source-stream verification audit：其冻结快照记录
`resume_audit.sha256_verified=true`，9 个 view 的 `bundle_status` 均为
`VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED`。本协议每次运行只复算
这个小型 manifest 的 SHA-256 并检查上述语义，不逐次重新读取或重哈希约 9.8GB
view shards。若 shards 需要替换或重建，必须先在独立审计中重新做 source-stream
verification 并生成新 manifest，再修订协议；不得只改配置摘要。这个依赖只证明
source streams 的机械校验已完成，不改变 manifest 内保留的 geometry audit
scientific verdict，也不把其 `NO_GO` 改写成通过。

`source_runtime_anchor_manifest` 是在正式候选运行前单独生成的小型 sidecar。生成
阶段允许按冻结 replicate 0/8 构造解析 truth 的 clean projection，以复现本来就
truth-derived 的合成 `scale_by_view`，也允许按固定 seed 重拟合 covariance；但
禁止运行任一 PDHG/PCGLS candidate、禁止计算 field/gradient/front 指标。sidecar
逐项锁定：实际加载的六组 geometry 数组、9 个 view 的 leaf-manifest 与 selected
index provenance、zero-boundary support、grid contract，以及两个 replicate 的
`scale_by_view`、component/full-graph whitener 四个 operational buffers、gate rows
和 seeds。正式 runner 必须重算这些 anchor 并 exact-match；这证明“当前确定性重拟合
与预运行锚点相同”，不倒推证明它等于历史 conditioning 运行的内存状态。

runner 必须在解析这些输入、构造 replicate 或创建正式输出前逐项复算。路径缺失、
`source_sha256` 的六个字段缺键/多键、摘要不是 64 位小写十六进制，或任一摘要不匹配时，
一律 fail closed 为 `PDHG_PREFLIGHT_INVALID`：不得打开 screen、不得生成排名，也
不得以 warning 继续。`geometry_audit_manifest` 的 filename/path/hash 或上述冻结
语义任一不匹配时使用同一判决。协议文件自身则按配置中 `protocol.sha256` 使用
相同规则独立校验。

## 2. 冻结问题定义

记原始 BOST 线性算子为 `A`，由冻结 seed/config 确定性重拟合并通过 runtime
anchor 核对的 full-graph whitening 为 `W_g`：

```text
A_g = W_g A
b_g = W_g y
```

`W_g` 不得在正式 screen 中改 seed、改 fit rule 或按结果重拟合。由于旧审计未
序列化 fit/whitener，本协议采用“预运行确定性重拟合生成 anchor，正式运行再次
重拟合并逐 buffer exact-match”的边界；不得把它写成 historical state exact reuse。

为减小不同样本振幅对正则化尺度的影响，对第 `i` 个场定义：

```text
m_i = active whitened measurements 的数量
s_i = max(||b_g,i||_2 / sqrt(m_i), 1e-12)
b_0,i = b_g,i / (s_i sqrt(m_i))
A_0,i = A_g,i / sqrt(m_i)
x_i = s_i z_i
```

PDHG 在无量纲变量 `z_i` 上求解：

```text
min_z  0.5 ||A_0,i z - b_0,i||_2^2
       + (alpha / n_D) sum_v rho_delta(||(D z)_v||_2)
       + I_C(z)
```

其中：

- `D` 是使用真实 `spacing_xyz` 的 forward-Neumann 3D finite-difference
  gradient：每轴末端差分固定为零，`D^T` 必须是其数值精确转置；它与物理
  `A` 内部使用的 voxel-centered gradient 分账，避免 centered stencil 对
  checkerboard mode 近乎失明；
- 令冻结的二值 support 为 `S`，盒外索引视为 `false`，并定义
  `S_D = {v : S(v) or S(v+e_x) or S(v+e_y) or S(v+e_z)}`；`n_D = |S_D|`，即
  forward-Neumann 梯度向量可能因 site 自身是 support 内源点，或其
  `+x/+y/+z` 邻点位于 support 内而非零的 site 数。一个 site 即使有多个可能
  非零分量也只计一次；`n_D` **不是** `count_nonzero(S)`，也不是 active support
  voxel 数；
- `C` 是现有 operator support 与一层 Dirichlet gauge 的交集；
- `P_C` 只清零 support 外和 gauge boundary，不做非负截断，因为当前解析场和
  gauge-fixed refractive-index perturbation 可以带符号；
- TV 使用 `rho_0(r) = r`；
- Huber 使用
  `rho_delta(r) = r^2/(2 delta)`（`r <= delta`）和
  `r - delta/2`（`r > delta`）；
- Huber 的 `delta` 固定为 `0.5`，继承 SupPCG smoke，不参与搜索；
- 输出后恢复 `x_i = s_i z_i`，所有 field/gradient/front 指标均在 `x_i` 上算。

## 3. 冻结 PDHG 更新

初始化：

```text
z^0 = 0
z_bar^0 = 0
p^0 = 0
q^0 = 0
theta = 1
```

令 `lambda = alpha / n_D`。第 `t` 次更新严格为：

```text
v_p       = p^t + sigma_A A_0 z_bar^t
p^(t+1)   = (v_p - sigma_A b_0) / (1 + sigma_A)

v_q       = q^t + sigma_D D z_bar^t

TV:       q^(t+1)_v = v_q,v / max(1, ||v_q,v||_2 / lambda)

Huber:    u_v = v_q,v / (1 + sigma_D delta / lambda)
          q^(t+1)_v = u_v / max(1, ||u_v||_2 / lambda)

z^(t+1)   = P_C[z^t - tau(A_0^T p^(t+1) + D^T q^(t+1))]
z_bar^(t+1) = z^(t+1) + theta(z^(t+1) - z^t)
```

内核 history 只记录本轮已经计算的同一个 extrapolated 点 `z_bar^t`：复用
`A_0 z_bar^t - b_0` 得到 data objective，复用 `D z_bar^t` 得到 TV/Huber edge
penalty，并把二者相加得到 extrapolated total objective。history 不得为了日志
再调用一次 `A_0` 或 `D`，也不得把这些 `z_bar^t` 数值冒充精确的 `z^(t+1)` /
checkpoint objective。

不得增加 truth-aware restart、backtracking、early stopping、learned correction
或候选专属初始化。

`alpha = 0` 只用于第 5.2 节的 data-only 数值控制：此时固定 `q^t = 0`，不执行
任何含 `1/lambda` 的表达式。为让 instrumented kernel 使用同一调用合同，该控制
仍恰好执行一次 `D z_bar^t` 与一次 `D^T 0`，但二者的代数贡献必须严格为零；
第 6 节的独立 recurrence oracle 不依赖这两个零项。它不属于正式候选，也不能
作为看到结果后的 fallback。

### 3.1 固定步长

步长安全系数固定为 `eta = 0.90`，不参与搜索：

```text
sigma_A = eta / sqrt(L_A2_safe)
sigma_D = eta / sqrt(L_D2_bound)
tau     = eta / (sqrt(L_A2_safe) + sqrt(L_D2_bound))
```

其中：

- `L_D2_bound = 4(dx^-2 + dy^-2 + dz^-2)`，对应本协议冻结的
  forward-Neumann stencil；
- `L_A2_power` 使用固定 seed、16 次 matrix-free power iteration，同时估计
  batch 中每个 whitened `A_0` block，取后 8 次 Rayleigh quotient 的最大值；
- `L_A2_safe = 1.50 * max_i L_A2_power_i`；graph whitening 与
  `1/sqrt(m)` measurement scale 必须都包含在 probe 内；
- 普通 power iteration 不是数学上认证的上界，因此报告只能称其为
  `safety-inflated power estimate`，不能写成 proven bound；
- 另以 `eta=0.50` 做不看 truth 的数值压力检查；具体轨迹、窗口、归一化和失败
  阈值严格使用第 6.1 节，不再保留“明显失稳”等事后解释空间；
- power probe 的 16F+16A、耗时与内存单独记为 per-rig shared setup，不得藏入
  或重复摊入某个方法的 solver time；shared-setup summary 必须单列，且不得把
  当前已执行 preflight 的进程伪称为 cold-start。

按上述定义，目标条件为：

```text
tau (sigma_A L_A2_safe + sigma_D L_D2_bound) = eta^2 < 1
```

该等式是基于安全膨胀估计的实现合同，不是未经条件的收敛证明。如果实际代码
采用不同缩放或 step rule，必须先修订协议并重新冻结，不能把它作为本协议的
实现细节直接运行。

## 4. 调用预算与公平比较

账本中的 `F/A` 分别指一次 `A_0` forward 与一次 `A_0^T` adjoint。一次 PDHG
kernel iteration 必须且只能包含：

- 1 次 `A_0` forward；
- 1 次 `A_0^T` adjoint；
- 1 次 `D` 与 1 次 `D^T`；
- pointwise proximal 与 `P_C`，同样进入 wall-time。

也就是逐迭代严格为 `1F + 1A + 1D + 1D^T`。内核中的 history 必须复用这一次
`F` 和 `D` 的结果；objective helper、checkpoint 保存、finite 检查或日志都不得
在 kernel 内产生第二次调用。`D/D^T` 不计入昂贵的 BOST operator calls，但必须
各自计数并进入 wall-time。

因此 PDHG-`K` 的逻辑预算固定为：

```text
forward calls = K
adjoint calls = K
total BOST operator calls = 2K
```

精确 checkpoint objective 只能由内核外的显式 scorer 在已保存的 `z^K`
（runner 中可命名为 `x_K`）上另算：每个被评分 checkpoint 额外执行 `1F + 1D`，
不执行 `A` 或 `D^T`。其中额外 `F` 必须记入独立的
`checkpoint_metric_forward_calls`，scorer 的 `D` 记入
`checkpoint_metric_gradient_calls`；它们不得并入或伪装成 solver calls。报告的
exact checkpoint data、edge 与 total objective 只能来自这个 scorer，而不能从
`z_bar` history 抄值。scorer 调用只做离线记录，不参与停止、回退或候选选择。

checkpoint 可以复用同一条 trajectory 以节省物理执行，但每个候选的**逻辑**
调用仍按独立运行的 `2K` 报告。报告必须同时列出 logical calls、实际 physical
solver calls、checkpoint scorer calls、prefix reuse 节省量和 shared setup；
不得用 physical prefix reuse 把 PDHG 描述成比独立部署更便宜，也不得把 scorer
的额外 forward 藏在评估脚本外。

## 5. 冻结候选与强基线

### 5.1 32 个正式 PDHG 候选

唯一允许的正式网格为：

```text
penalty = {tv, huber}                                  # 2
alpha   = {1/256, 1/64, 1/16, 1/4}                   # 4
K       = {4, 8, 16, 32}                             # 4
theta   = 1, eta = 0.90, huber_delta = 0.5           # fixed
```

正式候选总数严格为 `2 x 4 x 4 = 32`。配置文件中同时保存 alpha 的 exact fraction
与 decimal (`0.00390625, 0.015625, 0.0625, 0.25`)，避免字符串或浮点解析造成
重复候选。

候选 ID 固定为：

```text
pdhg_{tv|huber}_a{1of256|1of64|1of16|1of4}_k{4|8|16|32}
```

16 个场每个候选一条记录，共 `32 x 16 = 512` 条正式候选记录。

### 5.2 17 个控制与强参考

以下项目不进入 32 个正式候选计数，但必须在同一数据、device、dtype 和 metric
实现下重算或逐值核对：

1. `alpha = 0` 的 unregularized PDHG：`K = 4/8/16/32`，共 4 个数值控制；
2. full-graph covariance、Sobolev strength 3 的 graph-PCGLS：
   `K = 2/3/4/5/6/8/12/16/24/32`，共 10 个强基线 checkpoint；
3. component covariance、Sobolev strength 3 的 `component_s3_k4`，共 1 个
   经典 anchor；
4. 已冻结 SupPCG 候选
   `sup_huber_k3_n1_g0p32_a0p7` 与
   `sup_huber_k6_n1_g2p56_a0p7`，共 2 个历史机制参考，不允许重新选 SupPCG
   参数。

控制/参考共 17 个；加上 32 个正式候选，总计 49 个方法、`49 x 16 = 784`
条逐场记录。若 runner 生成的 candidate count、method count 或 row count 不等于
上述数字，运行必须失败而不是静默继续。

### 5.3 主基线不是弱同阶段基线

对 PDHG-`K`，先在所有 `graph-PCGLS-k` 且 `k <= K` 的 checkpoint 中选择
**pooled mean field error 最低**的一个固定 stage，记作
`graph_budget_frontier(K)`；若 mean 完全相同，选择调用更少者。

- 该选择对 16 个场使用同一个 graph stage，不允许 per-field oracle；
- 该 pooled stage 仍是在同一组 E2 场上选择的 data-adaptive strong baseline，
  因而只能用于本次保守机制筛选，不能称为独立 development split 上冻结的部署基线；
- 主排名与停止门均相对 `graph_budget_frontier(K)`；
- 另行报告相对 graph-PCGLS 同 `K`、`component_s3_k4`、unregularized PDHG
  同 `K` 的结果；
- `graph_budget_frontier(K)` 的调用不超过 PDHG-`K`，因此候选不能靠把
  graph-PCGLS 推入明显过迭代区间制造优势。

## 6. 预运行数值门

以下检查全部通过后才允许执行 32 候选 screen：

1. `A_g/A_g^T` 与 forward-Neumann `D/D^T` 的 float64 adjoint relative error 均不超过
   `1e-8`；
2. TV dual projection 与 Huber dual proximal 必须对照**独立 prox oracle**：
   oracle 使用单独的标量/NumPy 小向量闭式实现以及 tiny convex/Moreau 检查，
   不得 import、包装或调用 production prox helper；float64 最大绝对误差不超过
   `1e-10`；
3. `P_C` 对 support 外与 gauge boundary 严格为零，且不会截断内部负值；
4. `alpha=0` 必须从全零状态逐步对照一份独立写出的 checkpoint recurrence：
   `p` 使用冻结 data-dual prox，`z` 只含 `A_0^T p` 与 `P_C`，`z_bar` 使用冻结
   `theta`，且 `q` 在每一步严格为零；在 `K={4,8,16,32}` 每个 checkpoint 的
   relative volume error 均不超过 `1e-8`，不得只比较终点或仅检查“最终收敛”；
5. 每个 `K` 的 instrumented solver ledger 精确等于
   `K F + K A + K D + K D^T`，history 的额外 `F/D` 均为零；另以故意双调用的
   wrapper 做 negative control，注入任一第二次 `F`、`A`、`D` 或 `D^T` 都必须
   被拒绝，不能只在报告中留下 warning；
6. 精确展开 32 个 candidate ID，并让**全部 32 候选分别**在重置状态的 3-step
   fixture 上运行，不得只用 8 条去重 trajectory 代替；每个候选的 primal、
   extrapolated state、两个 dual 和 history 全部 finite，TV/Huber edge-dual 的
   maximum site norm 不超过 `lambda + 1e-7`；
7. stale norm estimate 必须 fail closed：至少覆盖不同 operator/whitener、support
   fingerprint、batch、measurement scale、grid shape 或 spacing 的负向 fixture，
   并验证在任何 solver `F/A/D/D^T` 调用发生前拒绝；
8. explicit checkpoint scorer 对每个 checkpoint 恰好增加 `1F + 1D`、不增加
   `A/D^T`，且其 exact objective 与 tiny direct scorer 一致；把 scorer 偷放进
   kernel 的 double-call fixture 必须失败；
9. CPU float64 fixture 与 MPS float32 fixture 的 field relative difference
   不超过 `5e-4`；
10. 第 1.3 节六项 `source_sha256`、`geometry_audit_manifest` 与
    `protocol.sha256` 在任何数据开封前完整匹配；geometry 检查只哈希 manifest，
    不逐次重哈希 9.8GB shards；
11. 使用同一 `.venv/bin/python` 在数据开封前执行冻结的三个 pytest 文件；JUnit
    必须恰含 114 个 frozen nodes，node-set SHA-256 必须等于配置值，且 failure、
    error、skip 均为零；pytest 前后必须是同一 clean HEAD；
12. `source_runtime_anchor_manifest` 的 geometry/whitener anchors 按第 1.3 节逐项
    exact-match，且在 observation-only stress 前完成；
13. 第 6.1 节冻结的 12-run observation-only `eta` stress gate 全部通过；
14. 配置、代码 commit、Python/PyTorch 版本、device、dtype、seed、输入 manifest
    和 SHA-256 在正式运行前写入报告头。

任何一项失败，状态只能是 `PDHG_PREFLIGHT_INVALID`。此时允许修复实现并以新
commit 重跑**同一冻结网格**，但不得把失败解释为 PDHG 性能 NO-GO。

### 6.1 `eta=0.90` vs `eta=0.50` observation-only stress gate

该门只接收一个 frozen/slots context：operator、冻结 norm estimate、observation
`b_0`、unit sigma/mask、ray count 与 normalization 常数；其类型中不存在 truth、
reaction label、front label、output amplitude 或 performance callback。这里 `b_0`
仍条件于第 1.1 节已经披露的 truth-derived synthetic `scale_by_view`，所以该门只能
称为“oracle-scale synthetic benchmark 内部 observation-only”，不能称为实验部署
blind gate。每个 replicate `0`、`8` 都独立运行以下三条 `K=32`
trajectory，且每条分别运行 `eta=0.90` 与 `eta=0.50`，总计每 replicate 6 条：

```text
pdhg_data_only_k32
pdhg_tv_a1of4_k32
pdhg_huber_a1of4_k32
```

两种 `eta` 均复用该 replicate 同一冻结 norm estimate、初始化、observation 与
其他 solver 参数。每条运行首先都必须满足：全部 primal/extrapolated/dual/history
值 finite；support/gauge 严格保持；`alpha=0` 的 `q` 严格为零；TV/Huber 的每 site
dual norm 不超过 `lambda + 1e-7`。任一失败立即判
`PDHG_PREFLIGHT_INVALID`。

为避免与第 3 节从 `t=0` 开始的更新下标混淆，令 runner 的记录标签
`j=1,...,32` 对应从 state `j-1` 到 state `j` 的一次更新。对每个场只从内核
状态定义：

```text
d_j = ||A_0 z_bar^(j-1) - b_0||_2^2 / max(||b_0||_2^2, 1e-12)

||(z,p,q)||_M^2 = ||z||_2^2/tau
                  + ||p||_2^2/sigma_A
                  + ||q||_2^2/sigma_D

r_j = ||(z^j-z^(j-1), p^j-p^(j-1), q^j-q^(j-1))||_M
      / max(||(z^(j-1),p^(j-1),q^(j-1))||_M,
            ||(z^j,p^j,q^j)||_M,
            1e-12)
```

这里 `d_j` 复用同一点的 extrapolated history，`r_j` 是本门唯一允许的 normalized
fixed-point residual；二者均由本轮已经产生的量计算，不增加 `F/A/D/D^T`。窗口
与 reducer 冻结为：

```text
head iterations = 1..8
tail iterations = 25..32
head_fp   = median(r_j, j in head)
tail_fp   = median(r_j, j in tail)
tail_data = median(d_j, j in tail)
```

先逐场判定，再按“任一场失败即该 trajectory/replicate 失败”汇总，不允许用 8 个
形态的 pooled mean 掩盖单场爆炸。以下任一 **absolute explosion** 成立即失败：

```text
max(d_j, j=1..32) > 100
tail_fp > 100 * max(head_fp, 1e-12)
```

若两种 `eta` 各自未触发 absolute explosion，再对同 replicate、同 trajectory、
同场配对；以下两个条件**同时**成立时判 `eta=0.90` comparative instability：

```text
tail_fp(eta=0.90)   > 10 * max(tail_fp(eta=0.50), 1e-12)
tail_data(eta=0.90) >  2 * max(tail_data(eta=0.50), 1e-12)
```

stress gate 报告必须保存 12 条运行的调用账本，以及逐 replicate / trajectory / 场
的 `head_fp`、`tail_fp`、`tail_data`、两个 absolute 布尔值和 comparative 布尔值。
该门只决定数值实现是否可进入正式 screen，不参与候选排名，也不得在看到 truth
结果后修改窗口、reducer、阈值或“任一场失败”的汇总规则。

## 7. 冻结指标

### 7.1 单场原始指标

每个方法、replicate、reaction family 必须保存：

- `field_relative_l2`；
- `gradient_relative_l2`；
- `front_top10_f1`；
- whitened data objective；
- TV/Huber penalty 与 total objective；
- primal update norm、dual feasibility violation；
- forward、adjoint、total logical calls；
- solver wall-time；
- 是否 finite、是否满足 support/gauge。

在计算 frontier、gain 或 ranking 前，512 条正式 PDHG 行必须逐条现场重算
`row_valid`：`execution_status=ok`、finite flag、support/gauge、field/gradient/front、
data/regularization/total objective、primal update、dual violation、elapsed 与 output
scale 都必须存在且有限；`total=data+regularization`，dual violation 不超过 `1e-7`，
logical calls 必须等于 candidate `K/K/2K`，checkpoint scorer forward 必须为 1。
任一行失败即停止排名，不允许把 NaN、失败状态或不完整 objective 变成默认值。
历史 SupPCG 行因来源 CSV 没有这些 objective 字段，只参与已冻结的机制参考与计数，
不伪装成重算 PDHG 行。

### 7.2 相对主基线的 gain

对 field error：

```text
field_gain_percent = 100 * (E_baseline - E_candidate) / max(E_baseline, 1e-12)
```

对 gradient error 使用同一公式。对 front 使用绝对 F1 差：

```text
front_gain = F1_candidate - F1_baseline
```

主表必须至少包含：

- field gain 的 mean；
- field gain 的 p10，固定使用
  `numpy.quantile(values, 0.10, method="linear")`；
- `field_gain_percent < -1.0` 的 harm count 与 harm rate；
- field gain 的 worst，即 16 个值的 minimum；
- gradient gain 的 mean、p10 和 worst；
- front gain 的 pooled mean、p10 和 worst；
- front-critical 子集
  `{wavy_front, thin_front, double_front, annular_kernel, oblique_shock}`
  的 mean front gain；
- 8 个 morphology 各自的两-replicate mean field/gradient/front gain；
- replicate 0 与 replicate 8 各自的 mean 和 worst；
- logical calls、physical calls、shared setup 与 wall-time ratio。

只有两个 replicate，因此不报告显著性星号、p-value、置信区间或“统计显著”。

### 7.3 wall-time 协议

- 正式设备固定为 Apple MPS、float32；不得给不同方法使用不同 device/dtype；
- 每次计时前后调用 `torch.mps.synchronize()`；
- PDHG 与 graph-PCGLS 各做 1 次明确记录、从所有 ratio 中排除的 timing-family
  warm-up；由于强制数值 preflight 已先使用 MPS，这不是 process cold-start，报告
  必须明确写 `process_cold_start_claimed=false`；
- 32 个正式候选各自在每个 replicate 上做 1 次 independent-`K` timing-only
  运行，并与自己的 graph frontier baseline 紧邻成对；replicate 0 固定为
  candidate-then-baseline，replicate 8 固定为 baseline-then-candidate，同时两边的
  candidate 遍历顺序相反。该路径不得接收 truth 或计算 field/gradient/front 指标；
- 每个候选的初筛 wall-time ratio 固定为两个 replicate 的 **paired ratio median**，
  不是全局 candidate median 除以全局 baseline median，也不能用共享 max-`K`
  trajectory 的总时间冒充较小 `K` 的独立耗时；
- 唯一入选候选及其 graph frontier baseline 随后在相同 batch 上每个 replicate
  各独立运行 5 次；两 replicate 合计必须恰为 5 次 AB、5 次 BA，runtime gate
  使用 10 个 paired ratios 的 median。candidate/baseline 各自 median、IQR 以及
  ratio-of-global-medians 只作描述；
- trajectory prefix reuse 的 screen 总耗时与独立部署耗时必须分开；
- wall-time 不替代 operator-call 公平性，两者都要报告。

## 8. 冻结排名与停止规则

### 8.1 唯一排名规则

32 个正式候选按以下顺序排序，全部使用未四舍五入原值：

1. 更高的 mean field gain vs `graph_budget_frontier(K)`；
2. 更高的 field p10；
3. 更低的 `>1%` harm rate；
4. 更高的 worst field gain；
5. 更高的 mean gradient gain；
6. 更高的 pooled mean front gain；
7. 更低的 measured wall-time ratio；
8. 更小的 alpha；
9. candidate ID 字典序。

该规则只能选出一个“opened scale-smoke winner”，不能把它称为 final model。

### 8.2 一次性通过门

排名第一的候选必须同时满足：

```text
mean field gain vs graph budget frontier     >= +1.0%
field p10                                     >= -1.0%
field harm rate for gain < -1%                <= 1/16 (6.25%)
worst field gain                              >= -3.0%
mean gradient gain                            >= 0.0%
pooled mean front gain                        >= 0.0
front-critical mean front gain                >= 0.0
median wall-time ratio vs graph frontier      <= 3.0
both replicate-specific mean field gains      > 0.0%
```

任何一项不通过，均不得用其他候选替换 winner 来“过门”，因为这会形成事后
constrained selection。必须按失败类型记录以下唯一状态之一：

- `POSTOPEN_PDHG_SCALE_NO_GO`：mean/p10/harm/worst 或 replicate consistency
  未通过；
- `POSTOPEN_PDHG_STRUCTURE_NO_GO`：field 门通过但 gradient/front 任一门失败；
- `POSTOPEN_PDHG_RUNTIME_NO_GO`：accuracy 与 structure 门通过但 wall-time 失败；
- `POSTOPEN_PDHG_SCALE_SIGNAL_ONLY`：全部通过。

`POSTOPEN_PDHG_SCALE_SIGNAL_ONLY` 仍不是成功论文结果，只允许：保存完整报告、
写明 post-open 偏倚，并另写一个覆盖更多 opened replicates 的新预注册。它不
自动授权 fresh 或 neural。

### 8.3 禁止追加搜索

正式结果写出后：

- 不追加 alpha 中点、极端值、Huber delta、theta、eta 或新的 K；
- 不按 morphology 选择 TV/Huber；
- 不更换主基线；
- 不删除失败场、慢运行或 NaN；
- 不因“趋势很好”打开 fresh；
- 不训练 learned proximal 来挽救失败的 deterministic baseline。

若 32 候选全失败，本协议对应的 scalar-step TV/Huber PDHG 分支关闭。后续若要
测试 diagonal preconditioning、line-search PDHG、TGV、wavelet 或 learned
proximal，必须把它们视为新算法，重新写问题、预算和停止门。

## 9. 失败模式应当怎样解释

失败状态只允许支持与观测相匹配的窄解释：

| 观测 | 允许解释 | 禁止解释 |
|---|---|---|
| preflight 不通过 | 实现、prox、adjoint、bound 或 MPS 数值链尚不可信 | “PDHG 无效” |
| 同 K 改善但输给 budget frontier | 正则化有局部机制信号，但迭代/预条件效率不足 | “已击败 PCGLS” |
| mean 正而 p10/harm/worst 失败 | 收益依赖形态或 replicate，尾部风险未解决 | “总体更好所以可泛化” |
| field 好而 gradient/front 差 | 主要是平滑或振幅拟合，不是物理边界恢复 | “三维重建质量全面提升” |
| TV 好、Huber 差 | 本尺度下 edge penalty 形状敏感 | “Huber 一般无效” |
| Huber 好、TV 差 | 本尺度下平滑转折可能缓解 staircasing/优化困难 | “Huber 已被证明最优” |
| 小 alpha 与 alpha=0 无差 | 当前尺度的 proximal 作用太弱或数值精度不足 | “无正则化已经最好” |
| 大 alpha field/gradient/front 同时差 | 本尺度过强，造成 bias 或结构破坏 | “所有 TV 类方法都失败” |
| 两个 replicate 符号相反 | noise/covariance realization 敏感 | “均值仍正即可继续” |
| accuracy 通过但 wall-time 失败 | 存在数值信号，但当前实现不具预算效率 | “调用相同所以速度相同” |
| 所有候选均失败 | 此冻结 normalization、scalar steps 与网格不具备继续价值 | “PDHG/TV/Huber 理论上无效” |

特别地，若 `L_A_bar` 极端保守导致更新几乎不动，报告应将其标成 scalar-step
conditioning failure，而不能暗中改用 power estimate。改进 bound 或使用
preconditioned PDHG 属于下一份协议。

## 10. 证据等级

本项目采用以下等级，后续摘要和网页必须原样标注：

| 等级 | 所需证据 | 本协议是否可达到 |
|---|---|---|
| E0 | 方法与门槛在运行前冻结 | 是，当前文件即 E0 |
| E1 | 数值内核、adjoint、prox、调用账本通过独立测试 | 运行前可达到 |
| E2 | 两个已见 replicate 的 post-open scale smoke | 本协议最高等级 |
| E3 | 预注册的 broader opened-replicate 验证 | 否，需新协议 |
| E4 | 完全封存的 fresh replicate/camera/session | 否，继续封存 |
| E5 | OERF 真实 flow-off 与实验数据上的独立验证 | 否，当前无数据 |

即使状态为 `POSTOPEN_PDHG_SCALE_SIGNAL_ONLY`，证据仍只能写成 E2。不得使用
“validated”“generalizes”“state of the art”“可投稿结论”等措辞。

## 11. 运行产物最低要求

正式 runner 在执行前必须读取冻结 JSON 配置，并至少生成：

```text
report.json
metric_rows.csv                 # 784 rows
candidate_summaries.csv         # 32 ranked candidates
method_summaries.csv            # 49 methods
operator_call_ledger.json
timing_audit.json
environment.json
e1_attestation.json
e1_pytest.xml
README.md
checksums.sha256
```

若第 6 节预检失败，runner 必须在同一原子输出目录写出
`preflight_invalid.json`、`README.md` 与 `checksums.sha256`，保存失败判据、完整
stress 调用账本、config/protocol/commit/environment 摘要，并明确
`truth performance rows generated = false`；随后以非零状态退出。不得生成空的
784-row 文件，也不得因异常而丢失失败证据。

`report.json` 必须包含：协议文件 SHA-256、代码 commit、输入 manifest、preflight
结果、49-method count、784-row count、winner、全部门的逐项布尔值、唯一判决
状态，以及 fresh/neural 仍封存的显式字段。图表只能从 CSV/JSON 生成，不允许
手工录入数值。`operator_call_ledger.json` 必须把 solver、history、checkpoint
scorer、norm setup 与 prefix reuse 分栏；`input manifest` 必须逐项回显第 1.3 节
的配置键、路径、expected SHA-256、observed SHA-256 与 match 布尔值，并单列
geometry audit manifest 的 source-stream verification 语义检查；不得制造逐次 shard
rehash 已执行的假记录。

正式进程只允许 `--root` 指向当前导入代码的同一 repository root，不能借另一个
clean clone 绕过检查。运行开始时保存 HEAD 与 config/protocol/六项 source/
geometry manifest 的字节摘要；成功或 preflight-invalid bundle 原子发布前必须
再次核对同一 clean HEAD 和同一文件摘要。任何中途变化都不得发布结果目录。
成功路径也必须先在同一父目录的唯一 staging 目录写完全部 CSV/JSON/JUnit/
README/checksum；只有所有写入成功后才允许一次 `replace` 发布正式目录，异常时
必须删除 staging，不能留下可被误读的半成品正式目录。

## 12. 运行前冻结清单

- [ ] 本协议已提交，SHA-256 已写入配置；
- [ ] `source_sha256` 恰含第 1.3 节六个源路径字段，且路径、摘要逐项 fail-closed
      匹配；
- [ ] `geometry_audit_manifest` 文件名、路径、摘要及 source-stream verification
      语义匹配，且没有逐次重哈希 9.8GB shards；
- [ ] 配置严格生成 32 candidates、17 controls/references、49 methods；
- [ ] replicate 只有 `[0, 8]`；
- [ ] 8 个 morphology 名称与现有 phantom schema 完全一致；
- [ ] runtime geometry anchor 与实际六组 loaded arrays、leaf provenance、support、
      grid contract 一致；
- [ ] replicate 0/8 的 deterministic-refit whitener 与预运行 anchor 逐 buffer 一致，
      且没有宣称 historical in-memory reuse；
- [ ] PDHG 更新、normalization、step bound 与本文件逐项一致；
- [ ] `n_D` 按 potentially-nonzero forward-Neumann site mask 计算，不是 active support voxel；
- [ ] kernel 每步恰为 `1F + 1A + 1D + 1D^T`，history 无额外调用；
- [ ] exact checkpoint scorer 的额外 `F/D` 已与 solver ledger 分开；
- [ ] E1 的 114-node JUnit attestation 及第 6.1 节 stress gate 全部通过；
- [ ] runner 对 method/row count 与 512 条正式 PDHG row validity 采用 fail-closed；
- [ ] timing warm-up 已排除，replicate 0/8 使用相反 AB/BA 顺序，winner 10 pairs
      恰为 5 AB + 5 BA；
- [ ] graph budget frontier 在候选排名前一次性计算并冻结；
- [ ] fresh、其余 opened replicates 与所有 neural 代码路径均不可达；
- [ ] 输出目录为空，报告中没有预填结果。

只有全部打勾后，才允许开始这一次 opened-data scale smoke。
