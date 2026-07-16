# V6B-1.0 少查询 RayKernel-DCO 预注册协议

状态：**DRAFT / UNCONSTRUCTED / NO FRESH LABELS OPENED**

日期：2026-07-16

用途：在任何新 aperture/layout 数据首开前冻结接口、基线、门槛、统计单位和停止规则。

## 1. 研究问题

在新 BOST rig 上只允许 `K` 次高保真 forward calibration measurements 时，能否用 ray-conditioned local 3D kernel 校正 nominal operator，并在隐藏 operator probes 和 inverse fields 上稳定优于同查询预算的非神经基线？

V6A 只提供结构假设，不提供可继承权重：它训练时读取完整高保真 operator rows 与 truth geometry，结果也是 `NO_GO_STOP_CAPACITY_ESCALATION`。V6B 必须从 query-limited train cache 重新训练；新 rig 经 `K` 次校准后称 **few-query adaptation**，不能称 zero-shot。

## 2. 冻结候选

### 2.1 基础结构

- 单模型 `33 -> 64 -> 64 -> 27`，radius-1 ray-local voxel kernel；
- 只输入可测/可标定的 nominal aperture、focus、pose、pixel/ray coordinates；
- 禁止 truth radius、renderer fidelity、generator seed、隐藏 field family；
- 同一核必须构造 forward 与 transpose，禁止单独训练 adjoint；
- 单 seed、固定训练步数；fresh 不 early-stop、不挑 ensemble、不改核半径或宽度。

### 2.2 少查询校准器

设冻结超网络为每条 ray 输出 `C in R^(M x 27)`。对预冻结 probe `x_k`，高保真 oracle 只返回：

\[
y_k=A_{star}x_k,\qquad r_k=y_k-A_0x_k.
\]

不微调 8,091 个网络参数，只拟合 27 个 channel gates：

\[
g=\arg\min_g \sum_k\|\sum_q g_q C_{:q}\odot A_0S_qx_k-r_k\|_W^2
+\lambda_g\|g-\mathbf 1\|_2^2.
\]

最终 callable 为：

\[
A_{corr}x=A_0x+\sum_q g_q C_{:q}\odot A_0S_qx,
\]

\[
A_{corr}^Tz=A_0^Tz+\sum_q S_q^TA_0^T((g_qC_{:q})\odot z).
\]

这比“在 fresh rig 上继续训练整个网络”更容易审计信息预算，也允许严格共轭。

## 3. Query 合同与泄漏防火墙

- primary synthetic `K=32`；`K=4/8/16` 只在 opened development 做剂量曲线；
- 一次 `A_star x` 返回全视角向量记 1 vector-query，同时报告 `K x views` view-equivalents、rays、path/aperture samples；
- 候选校准预算固定为 `K forward + 0 high-fidelity adjoint`；
- oracle 只暴露 `measure(x)`；第 `K+1` 次调用立即报错；不得暴露 `.operator`、rows、truth geometry 或 truth adjoint；
- V6B runner 禁止 import `_build_operator_pair`、`truth_parameter_vector`、`build_ray_basis_and_targets`；
- 完整 `A_star` 只存在于独立 evaluator 进程；预测和模型先哈希，再解封评分 probes/fields；
- normalization、mask、noise covariance 只能来自 train、calibration 或 flow-off repeats。

## 4. Split 与统计单位

- train：30 个 opened rigs，只通过 K-query API 重训；
- development：12 个 opened rigs，选择 K、正则、固定步数和唯一主基线 `B*`；
- fresh-A：16 个未见 aperture/f-number/focus rigs；
- fresh-B：16 个未见 layout/angles rigs；
- fresh-C：16 个 aperture x layout joint-OOD rigs；
- 每个 fresh rig：K 个可见 calibration probes、32 个封存 operator probes、12 个封存 inverse fields；
- rig 是统计单位；views、rows、probe repeats、noise seeds 和 fields 不得伪装成独立样本。

## 5. 同查询预算强基线

| ID | 方法 | 高保真 fresh 预算 |
|---|---|---:|
| B0 | nominal `A0` | 0 |
| B1 | minimum-norm secant `Delta A = R(X^T X + lambda I)^-1 X^T` | K forward |
| B2 | 每 ray 27-coefficient ridge + 邻接 ray graph Laplacian | K forward |
| B3 | geometry polynomial/B-spline-to-kernel ridge | K forward |
| Ref | V6A full-matrix ridge / 完整 `A_star` | oracle reference，不参与公平胜负 |

B1-B3 使用完全相同的 `X, R, g` cache。主基线 `B*` 只能在 opened development 一次性选择；fresh 禁止按 rig 取最好基线组成 oracle envelope。

## 6. 顺序门禁

### G0 - Physics gap

若 nominal mismatch RMS 小于独立 repeat/noise floor 的 2 倍，判 `PHYSICS_GAP_NOT_RESOLVED`，不把它写成算法失败或成功。

### G1 - Numerical validity

- 20 对随机向量 dot-product test：float64 defect `<=1e-10`，float32 `<=1e-5`；
- finite-difference gradient check 通过；
- 任一失败，判 `INVALID`，不进入 operator scoring。

### G2 - Operator 解锁 inverse

- 对 `B*` 的 noise-whitened hidden-probe error 改善 `>=10%`；
- 单侧 95% rig-cluster bootstrap CI 排除 0；
- 48 rigs 至少 36 个胜出，每个 A/B/C stratum 至少 8/16；
- 90th percentile paired degradation `<=5%`；最大 paired degradation `<=20%`；每个 stratum 聚合不得为负。

`max degradation <=5%` 只保留为额外 `ZERO_OBSERVED_HARM` 标签，不再作为唯一安全门。强 operator GO 为改善 `>=25%` 且 A/B/C 各至少 12/16 胜出。

主统计量先在 rig 内聚合：

\[
d_r=\log\frac{e_{cand,r}+\sigma_r}{e_{B^*,r}+\sigma_r},
\]

再对 A/B/C 等权，避免 rows 数量制造虚假精度。

### G3 - Gradient fidelity

废除旧实现的 `Ahat^T(Astar x)` 对 `Astar^T(Astar x)` 指标。新指标必须使用真实残差梯度：

\[
\nabla_x \tfrac12\|Ax-y\|^2=A^T(Ax-y).
\]

在零场、扰动 truth、nominal/CGLS 中间迭代点分别比较：cosine 中位数 `>=0.99`、5% 分位 `>=0.95`、无负 cosine、相对梯度误差中位数 `<=0.25`。

### G4 - Inverse value

- PBB/CGLS 相对 `A0` field error 改善 `>=3%` 且 cluster CI 排除 0；
- 同时胜过 B*，或在 B* 的 2% 非劣界内并有预注册 runtime 优势；
- 正则参数允许各方法在 development 等预算调优后冻结；
- 分别报告等 forward/adjoint calls 和等 wall-clock 两套结果。

真实实验没有 3D truth 时，不报告 field-L2；改用 held-out view、已知 phantom、重复性、front/shock location 和物理积分量。

### G5 - Runtime

报告 calibration acquisition、拟合、kernel materialization、单次 `F/F^T`、峰值内存、整次重建和 break-even reconstruction count。`matvec <=0.5 x A_star` 只在目标分辨率、同硬件/精度/批量下作为效率 GO；另冻结 100 次重建的摊销总时间。

## 7. 外部证据的称呼

| 等级 | 可以写 | 不可以写 |
|---|---|---|
| internal synthetic fresh | K-query 机制在新 generator 参数上有效 | 外部 renderer 或实验有效 |
| `photon` external synthetic | 独立代码路径上的光圈/布局迁移 | 真实光学或 OERF 有效 |
| PSU fixed-rig real | 真实 flight-body BOS 的 reconstruction/reprojection consistency | 跨光圈 operator 泛化；绝对 field-L2 |
| OERF real | 对独立 session/phantom/held-out camera 的 apparatus-specific 收益 | 无 3D truth 时的绝对场精度 |

## 8. 停止规则

- fresh 首开后禁止改 K、probe family、核半径、宽度、正则网格、seed、阈值或 B*；
- bug 修复必须升级协议版本，并换全新 fresh manifest；
- G2 失败则不解封 inverse；G4 失败保留 `OPERATOR_ONLY`；runtime 失败只关闭效率主张；
- PSU 只能通过 E0-E2 固定 rig 门，不能替代 fresh-A/B/C；
- OERF 没有已知 calibration phantom 时，V6B 最多称 measurement-domain adaptation，不能称真实 operator identification。

## 9. 必须新增的测试

1. matrix-free 与 materialized forward 等价；
2. 同核 transpose 的严格 dot-product；
3. 第 K+1 次 high-fidelity query 拒绝；
4. truth-key sentinel 不可访问；
5. 完整-row API 调用即失败；
6. 27-gate synthetic recovery；
7. checkpoint/center/scale/offset loader 固定；
8. prediction hash 时间早于评分标签解封；
9. 按 rig 的 bootstrap/sign test，不按 row 统计；
10. 报告同时列出 vector-query、view-equivalent、rays 和 wall-clock。

只有上述实现、测试、外部数据接口和 config hash 全部存在，V6B 才从 `UNCONSTRUCTED` 升为 `READY_TO_PREREGISTER`。

## 10. 2026-07-16 实现检查点：协议内核通过，fresh 仍未构造

已实现并可复核：

- `BudgetedForwardOracle.measure(x)`、第 `K+1` 次拒绝和 `K forward + 0 truth-adjoint` 记账；
- 27-channel gate design/fit、matrix-free forward、同核 adjoint 与小规模 materialization 对照；
- 20 对 float64 dot-product、真实残差梯度有限差分、27-gate recovery 与 pre-score SHA-256；
- 64 维欠定 toy 上的 in-class 正控制和 misspecified 负控制；
- 同预算 minimum-norm secant 基线，且模型外 K=32 中候选被 secant 反超，失败可见。

产物：[`limited_query_calibration.py`](../demo_t16_operator/limited_query_calibration.py)、[`run_v6b_protocol_conformance.py`](../demo_t16_operator/run_v6b_protocol_conformance.py)、[`report.json`](../demo_t16_operator/results/v6b_protocol_conformance/report.json) 与 [`metrics.csv`](../demo_t16_operator/results/v6b_protocol_conformance/metrics.csv)。报告只判 `PASS_PROTOCOL_CONFORMANCE_ONLY`，`scientific_claims_unlocked` 为空。

尚缺：V6A checkpoint/normalization loader、独立 evaluator 进程、B2/B3、rig-cluster statistics、完整 view/ray/runtime 账本、PSU loader conformance、`photon` 外部 renderer、fresh-A/B/C manifest 与 OERF 数据合同。因此本协议状态继续是 **`UNCONSTRUCTED`**，不能升级为 `READY_TO_PREREGISTER`。
