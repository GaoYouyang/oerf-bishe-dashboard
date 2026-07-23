# JACRU-N1：噪声感知三维重建协议红队

- **审查日期：** 2026-07-18
- **角色：** N1 实验协议红队
- **状态：** `N1_PROTOCOL_READY_FOR_FREEZE_OPENED_SYNTHETIC_ONLY`
- **审查对象：** M2.7/M2.8 冻结配置、summary、逐行指标、runner、独立 validator 与其 mutation tests
- **文件性质：** N1 预注册协议草案与停止规则；不是算法实现、结果包、fresh 授权或方法成功声明

## 0. 红队总判决

> **N1 可以继续，但必须先更换“选择合同”，不能只在 M2.8 后面再加一个 noise-aware
> solver。当前证据已经说明：更快地拟合 noisy observation 会稳定伤害同一个界面样本；而
> M2.7/M2.8 的开发集选择仍读取 field truth，exact camera-block 的 1001F-equivalent setup
> 也没有进入主预算。**

本轮只读核验得到以下结论。

1. M2.7 与 M2.8 的冻结 packet 均通过现有独立 validator；`16` 项 mutation/contract tests
   通过。正式判决仍分别是
   `M2_7_TARGET_NO_HARM_PARETO_ORACLE_NO_GO` 与
   `M2_8_INTERPOLATION_CALIBRATION_ENVELOPE_NO_GO`。
2. M2.7 的 exact camera-block 在 `K=9`、`23F/22Adj` 的候选账本下把 development mean
   measured-reprojection ratio 压到 `0.8515x`（JACRU）与 `0.9136x`（pooled CNN），但两者
   都有 `8.33%` harm rate；最坏 field gain 分别为 `-8.89%` 和 `-11.89%`。
3. 失败不是随机 seed 抖动。`single_interface / base_seed=2113 / case_id=423593699e2b6baeef3f`
   在两种 backbone、三个 model seed、`K=9/10` 下全部受害。`K=10` 时 measured residual
   已约为 `5.3e-4` 至 `6.7e-4`，但对 clean observation 的 residual 仍约 `1.48e-2`。
   这是明确的 noisy-target overfit 证据，而不是 solver 未收敛。
4. M2.8 即使允许 evaluator 读取 truth，在每个样本上选择最优 alpha，`K=10` 的
   reprojection-feasible rate 仍只有 `97.22%`；对上述 `2113/single_interface`，truth-oracle
   alpha 被迫落在约 `[0.989, 0.992]` 附近，field gain 仍为负。固定插值路线应保持关闭。
5. 当前 synthetic noise 合同不能直接当部署协方差。iid noise 和 camera bias 都按未知
   `clean signal RMS` 缩放；真实候选看不到该量。camera bias 又是每 camera、每分量的常数偏置，
   在当前 `A in R^(150x1000)` 且 `rank(A)=150` 的 toy 中与 field 完全不可辨识。
6. 因此 N1 的唯一合理主问题是：**独立校准得到的测量协方差、truth-free discrepancy
   stopping 与 fail-closed fallback，能否在严格同预算下同时改善 field mean 和 tail？**
7. N1 即使通过，也最多是 `OPENED_SYNTHETIC_GATE_PASS`。没有真实 flow-off repeats、稳定
   camera-bias 合同和独立 audit camera 前，不得写成 real BOST generalization、实验成功或论文主结果。

## 1. 审查证据与复核状态

### 1.1 冻结输入

- [M2.7 config](../demo_t16_operator/configs/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_v1.json)
- [M2.7 summary](../demo_t16_operator/results/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public/summary.json)
- [M2.7 metric rows](../demo_t16_operator/results/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public/metric_rows.csv)
- [M2.8 config](../demo_t16_operator/configs/jacru_m2_8_interpolation_calibration_ceiling_postopen_v1.json)
- [M2.8 summary](../demo_t16_operator/results/jacru_m2_8_interpolation_calibration_ceiling_postopen_public/summary.json)
- [M2.8 fixed rows](../demo_t16_operator/results/jacru_m2_8_interpolation_calibration_ceiling_postopen_public/fixed_interpolation_rows.csv)
- [M2.8 truth-oracle rows](../demo_t16_operator/results/jacru_m2_8_interpolation_calibration_ceiling_postopen_public/truth_oracle_ceiling_rows.csv)
- [M2.3-M2.7 shared runner](../site_tools/run_jacru_m2_3_matrix_free_projection.py)
- [M2.8 runner](../site_tools/run_jacru_m2_8_interpolation_calibration_ceiling.py)
- [M2.7/M2.8 independent validator](../site_tools/validate_jacru_m2_7_m2_8_evidence.py)
- [validator mutation tests](../site_tools/test_validate_jacru_m2_7_m2_8_evidence.py)
- [T0 source config](../demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json)
- [synthetic fixture](../demo_t16_operator/jacru_synthetic_fixture.py)

复核命令：

```bash
PYTHONPATH=. .venv/bin/python site_tools/validate_jacru_m2_7_m2_8_evidence.py --packet all
PYTHONPATH=. .venv/bin/python -m pytest -q site_tools/test_validate_jacru_m2_7_m2_8_evidence.py
```

本轮结果为两个 packet 均独立验证通过，测试为 `16 passed`。这证明 frozen public packet 的
schema、hash、行数、派生指标、预算字段和 NO-GO authorization 与 validator 合同一致；它不证明
public CSV 可从原始 observation/prediction arrays 独立重放，因为这些 arrays 不在 public packet 中。

### 1.2 现有 validator 做对了什么

现有 validator 不 import runner、model 或 optical operator，而从 public rows 重新计算：

- M2.7 的 `1980` 个 metric rows、`990` 个 matched baseline rows 与 `570` 个 reference rows；
- M2.7 的 field/H1 gain、reprojection ratio、harm、aggregate、K=9 development selection；
- M2.8 的 `3240` 个 fixed-alpha rows、`360` 个 truth-oracle rows 与 `180` 个 baseline rows；
- alpha=`0/0.5/1` 三点恢复二次函数、feasible interval 和 truth-oracle minimizer；
- truth/deployability 标志、1001F-equivalent exact-block setup 与 authorization 不可升级。

### 1.3 N1 validator 必须补上的盲点

N1 不能只复制现有 validator。它还必须检查：

1. `case_id -> split/family/base_seed/geometry_digest/nuisance_seed` 的 manifest 映射，而不是只检查
   case_id 是否出现在 30-case catalog 中；
2. checkpoint SHA-256、covariance-calibration packet SHA-256、candidate-selection commit 与
   prediction hash；
3. 删除所有 truth/evaluator columns 后，selection 函数仍给出完全相同的候选；
4. model seed 不能被当作独立物理样本，family 共享同一个 base-seed geometry 时必须 cluster；
5. standalone selected-candidate replay 的真实 F/Adj、preconditioner/whitening applications、wall time、
   peak memory 与 cold-start setup；
6. covariance SPD、Woodbury/Schur 数值安全、whitening round-trip 与 fail-closed 路径；
7. private raw arrays 上重算 prediction metrics；public validator 只验证经过 hash 的衍生表。

## 2. M2.7/M2.8 对 N1 的直接警告

### 2.1 “候选不读 truth”不等于“选择不读 truth”

M2.7 的 candidate eligibility 主要看 visible fraction、mean reprojection、closure、breakdown 与调用预算，
但最终排序字段是：

```text
maximum_development_mean_field_gain_to_best_matched_classical
```

runner 也按 development `field_gain_mean` 降序选 K。M2.8 的 fixed alpha 在单次推理时确实不读 truth，
但 development eligibility 同时使用 field gain、H1 gain、harm rate、worst field gain，排序仍以
field gain 为首项。因此：

- M2.7/M2.8 的 “truth-free” 只适用于候选公式或单样本 inference；
- 它不适用于超参数、K、alpha、checkpoint 或方法族选择；
- N1 必须把 **candidate execution** 与 **candidate selection** 两层都做成 truth-free。

另外，M2.7/M2.8 runner 会重新训练网络，training summary 的 `best_epoch` 由 development field-L2
决定。N1 若要声称选择 truth-free，必须先冻结现有 checkpoint；若确需重训，只能在 training split
内部做 nested CV，N1 selection/lock truth 不得参与 checkpoint 选择。

### 2.2 best-of-two field oracle 只能做 evaluator control

当前 field gain 的分母是每个 case 上 CGLS 与 Huber 两者中 field error 更小者；H1 又分别取两者中
H1 error 更小者。这是保守的 evaluator-side “best classical oracle”，但不是一个现实中可选的单一基线。
N1 必须预先冻结一个 primary comparator，并分别报告对 CGLS、Tikhonov/Huber 的 paired gain；
per-case best-of-classical 只能保留为更难的 evaluation ceiling，不能成为部署基线。

### 2.3 均值会掩盖唯一但稳定的失败簇

M2.7 的 `case_model_count=36` 来自 `12 cases x 3 model seeds`。三个 model seeds 不是 36 个独立
物理实验；同一 base seed 下的两个 family 又共享 geometry。正确统计单位应是：

```text
model-seed rows -> 先在同一 case 内聚合
case/family rows -> 再按 base_seed + geometry_digest 聚类
最终独立单位 -> geometry/base-seed block
```

`2113/single_interface` 在所有六个模型实例中都失败，说明 tail 是 case-level 结构问题，而不是
一个偶然训练 seed。N1 禁止以 pixel、ray、model seed 或 nuisance repeat 作为独立 field 样本来缩窄 CI。

### 2.4 当前调用账本不是端到端成本证据

M2.7 对每个 case 一次运行到 `maximum K=10` 并保存全部 snapshots。行内的 F/Adj 被还原为各 K 的
理论预算，但 `preconditioner_applications` 在所有 K 行中都记录完整 sweep 的 `11` 次。M2.8 同样先跑
到 K=10，再评估 K=9/10 和所有 alpha；rows 又把 `neural_inference_seconds` 写为 `0.0`。

这不影响已声明的 oracle/no-go 结论，但意味着：

- K sweep 只适合离线筛选，不能提供 selected K 的 standalone latency；
- 1001F-equivalent exact camera-block setup 被明确排除在 matched budget 外；
- dense norm setup、model inference、whitening、factorization 与 data transfer 尚未形成可比较的端到端账本；
- N1 通过前必须单独重放一次冻结候选和每个主基线。

## 3. 当前噪声模型与 camera-bias 可辨识性

### 3.1 当前生成合同

每个 case 的 observation 可写为

```text
y = A_cont(x_true) + B beta + epsilon,
epsilon ~ N(0, sigma_n^2 s_clean^2 I),
beta    ~ N(0, sigma_b^2 s_clean^2 I_6),
sigma_n = 0.01,
sigma_b = 0.02.
```

这里共有 `3 cameras x 25 rays x 2 components = 150` 个标量 measurement coordinates；
`B in R^(150x6)` 把每个 camera/component 的常数偏置复制到该 camera 的 25 条 rays。若把 beta
边缘化，当前 generator 的理论 covariance 为

```text
C_oracle = s_clean^2 (0.01^2 I + 0.02^2 B B^T).
```

因为 `B^T B = 25 I_6`，camera-bias 子空间与其正交补上的 covariance eigenvalue 之比为

```text
(0.01^2 + 25 * 0.02^2) / 0.01^2 = 101.
```

所以当前误差绝不是 iid white noise。更重要的是，`s_clean` 来自 candidate 看不到的 clean
observation；直接把 config 中的 `0.01/0.02` 和 clean RMS 交给 N1 就是 oracle covariance。

### 3.2 camera bias 在当前 toy 中不可从单帧联合识别

M2.7 frozen audit 对 active matrix 给出：

```text
A in R^(150x1000), rank(A)=150.
```

因此 `range(A)=R^150`。对任意 `B beta`，都存在某个 `delta x` 使得

```text
A delta x = B beta.
```

于是 `(x,beta)` 与 `(x+delta x,0)` 产生同一 observation。没有独立 background/flow-off frame、
已知 ambient-zero rays、时间稳定先验或额外 field prior 时，单帧里联合拟合 beta 不是“难”，而是
结构上不可辨识。N1 禁止：

- 从当前 y 同时自由拟合 x 与每-camera offset，然后把 residual 下降解释为 bias correction；
- 把已知 generator beta 提供给 candidate；
- 在候选内按 clean residual 或 true beta 选择去偏强度。

可部署做法只有三类：

1. 用同曝光、同相机状态的 flow-off/background repeats 估计 bias mean 与 drift covariance；
2. 使用协议保证为零偏转的 ambient detector region；
3. 把 beta 作为有独立校准 prior 的 random effect 边缘化，而不是从单帧无约束识别。

### 3.3 sensor noise 不是全部 discrepancy

synthetic fixture 用 continuous analytic gradient 生成 observation，却用 voxel FD/trilinear A 做反演。
本轮只读计算发现 development case 上

```text
||y-clean|| / ||clean||              = 0.0148 to 0.0269
||A x_true - clean|| / ||clean||     = 0.1608 to 0.4387
```

对反复失败的 `2113/single_interface`，后一项约为 `0.4387`。这意味着标准 Morozov 假设
`y_clean=A x_true` 在当前 evaluator field 上不成立。仅用 flow-off sensor covariance 设 threshold，
无法覆盖 forward-model discrepancy。N1 必须把以下三项分开：

```text
y = A_voxel x + delta_model + B beta + epsilon.
```

- `epsilon`：detector/PIV/BOS displacement noise，可由 flow-off repeats 估计；
- `B beta`：camera-static 或慢漂移 bias，需独立 reference；
- `delta_model`：continuous renderer、finite aperture、voxelization、calibration error 等 forward mismatch，
  不能从 flow-off repeats 得到。

训练 synthetic phantoms 上由 truth/clean renderer 估计的 `delta_model` 只能形成
`TRAIN-CALIBRATED SYNTHETIC MECHANISM`；要转到真实 BOST，必须有 calibration target、multi-fidelity
renderer 或 held-out camera evidence。

## 4. N1 的冻结研究问题与假设

### 4.1 唯一主问题

```text
在不读取 reconstruction truth、clean observation、generator nuisance、fresh exact mass
或 per-case best baseline 的条件下，独立校准 covariance + first-crossing discrepancy stop
+ fail-closed fallback，能否在 <=24 paired-call slots 内，较冻结 classical comparator
同时改善 field relative-L2、H1 和 case/rig tail？
```

### 4.2 主假设与否证假设

- **H1，校准假设：** 独立 calibration packet 估计的 structured covariance 在 held-out calibration
  repeats 上达到预注册 coverage，且数值 SPD/Schur gate 通过。
- **H2，选择假设：** 只用 observation、A/A^T、geometry、calibration statistics 和 cost 即可冻结
  一个 candidate；删除 evaluator truth 后 selection 不变。
- **H3，重建假设：** 冻结 candidate 在 opened lock 上对 primary comparator 的 cluster-level mean field
  gain 至少 `5%`、H1 gain 至少 `3%`，同时满足 no-harm 和 per-rig tail。
- **H4，预测假设：** candidate 在未用于 reconstruction/selection 的 audit rays/camera 上改善
  bias-corrected observation prediction；support residual 下降不能单独替代该门。
- **H5，成本假设：** candidate 不超过 frozen F/Adj cap，且 cold end-to-end cost 处于预注册上限内。

任一 H1/H2/H5 失败为 `INVALID_NO_RANKING`；H1/H2/H5 有效而 H3 或 H4 失败为科学
`N1_NO_GO`。

## 5. 数据、防火墙与 covariance 来源

### 5.1 不把已打开的六个 development seeds 包装成 blind lock

现有 development 与 OOD 结果已经反复打开。把其中三个 seeds 重新命名为 “lock” 不能产生新证据。
它们只允许用于 N1 协议设计、failure reproduction 和 regression tests。

N1 应冻结一个新的 deterministic synthetic manifest，但证据标签仍为
`E1_OPENED_SYNTHETIC_N1_NO_FRESH`。最小两阶段建议：

| 阶段 | 独立 geometry/base-seed blocks | 用途 | 是否可看 field truth |
|---|---:|---|---|
| `n1_selection` | 至少 24 | truth-free covariance/threshold/candidate selection | selector 永不读取；冻结后仅作诊断 |
| `n1_opened_lock` | 至少 24 | 冻结后一次打开，检验 mean、family 与 tail | 只在 prediction/config hash 后 evaluator 打开 |
| `n1_covariance_stress` | 至少 16 | unseen bias/noise amplitude、drift、geometry stress | 同 lock |

这只够 mechanism screen。若要以 one-sided 95% confidence 支持 `harm rate <=5%`，零 harm 时也至少
需要 `59` 个真正独立 blocks；nuisance repeats、model seeds 和共享 geometry 的 families 不能充数。
因此 claim-grade opened synthetic lock 应使用至少 `60` 个独立 geometry/base-seed blocks，或明确
把 tail 结论降级为描述性 evidence。

seed 列表、family 列表、geometry jitter、nuisance seeds、audit-ray mask 和 manifest SHA-256 必须在
任何 N1 reconstruction 执行前冻结。split assignment 应由固定 SHA-256 规则产生，不得根据 M2.7
已知失败样本人工平衡。

### 5.2 calibration packet 与 reconstruction packet 必须独立

每个 rig/geometry family 至少提供：

1. `flow_off_fit`：估计 bias mean、camera/component covariance 与 iid/diagonal floor；
2. `flow_off_holdout`：只检验 whitening coverage，不回调参数；
3. `reconstruction_support`：candidate 唯一可见的 flow-on observations；
4. `selection_audit`：selection fields 的 held-out rays/camera，只用于 truth-free selection；
5. `lock_audit`：candidate freeze 后才打开，不得影响 selection 或 stopping。

建议 structured covariance，而不是在少量 repeats 上拟合 150x150 full matrix：

```text
C_hat = D_hat + B Sigma_beta_hat B^T,
D_hat = positive diagonal or per-camera diagonal,
Sigma_beta_hat = 6x6 shrinkage covariance from independent repeats.
```

每个主 rig 建议至少 `64` 个 fit repeats 与 `64` 个 holdout repeats；若实测不足，缩减参数维数，
不要用 reconstruction residual 补样本量。synthetic primary generator 也应使用 detector-unit absolute
noise scale或独立 reference scale，禁止继续按该 case 的 clean RMS 给 candidate 构造 covariance。

### 5.3 primary 与 oracle covariance 分层

| covariance | 来源 | candidate 可用 | 证据上限 |
|---|---|---:|---|
| `C_iid_cal` | independent flow-off repeats 的单一 floor | 是 | deployable-form control |
| `C_camera_cal` | `D + B Sigma_beta B^T`，独立 repeats | 是 | N1 primary |
| `C_total_train` | train-only synthetic forward mismatch + sensor calibration | 仅 synthetic | mechanism，不能外推 real BOST |
| `C_generator_exact` | clean RMS、true beta/noise parameters | 否 | oracle ceiling only |
| `C_per_case_residual_fit` | 当前 y 或 reconstruction residual | 否 | leakage，`INVALID_NO_RANKING` |

## 6. truth-free candidate selection

### 6.1 selector 允许与禁止的输入

允许：

```text
y_support, A, A^T, support mask, geometry metadata,
independent calibration covariance, frozen checkpoint output,
iteration residuals, step norms, numerical flags, F/Adj/time/memory ledger,
selection-only held-out measurement residuals.
```

禁止：

```text
x_true, family label, interface count, clean observation,
additive_noise_uv, camera_bias_uv, generator sigma or clean RMS,
field/H1 error, per-case best classical identity,
M2.8 truth alpha, exact projection field error,
n1_opened_lock or lock-audit rows.
```

最终 selector 应在一个专门的 unit test 中接收“已物理删除 forbidden columns”的 rows。仅靠
`truth_used_by_candidate=False` 标志不够。

### 6.2 discrepancy threshold

对 frozen covariance 定义

```text
r_k = y - A x_k,
d_k = ||C_hat^(-1/2) r_k||_2^2 / m_active.
```

若 bias 已由独立 calibration 边缘化而没有从当前 y 拟合，`m_active=150`；不得擅自减去 6 个
degrees of freedom。若在当前 observation 上拟合 beta，则因第 3.2 节不可辨识，主协议直接无效。

threshold 不从 field error、matched-CGLS residual ratio 或 generator sigma 选择。采用 split-conformal
calibration score：

```text
q_hi = order_statistic(
  { ||C_hat^(-1/2) z_j||^2 / m_active : z_j in flow_off_holdout },
  ceil((n_cal + 1) * (1-alpha))
),
alpha = 0.05.
```

若使用 train-calibrated model-discrepancy residual，则需另一套完全独立 holdout scores。最终 threshold
由 sensor 与 model-discrepancy 两者中协议预先指定的组合给出，不能看 lock truth 后放宽 tau。

停止规则固定为：

```text
k_stop = first k in {0,...,Kmax} such that
         pooled d_k <= q_hi
         and every camera score d_k,c <= q_hi,c
         and numerical/cost/Schur gates are valid.
```

不能用 mean pooled score 掩盖一个 camera 的偏置尾部。若初始 snapshot 已满足 threshold，就返回该
snapshot；若到 Kmax 仍未满足、score 非有限、camera gate 失败或 covariance OOD，则 fail closed。

### 6.3 fail-closed 不能额外跑第二条完整路径

N1 primary fallback 固定为已经在 feature preparation 中得到的 `prepared CGLS base-12`，或从同一条
trajectory 已缓存的预注册 snapshot。禁止 candidate 失败后再免费运行一条 24-step baseline。

建议冻结以下状态机：

```text
CALIBRATION_INVALID -> no reconstruction, INVALID_NO_RANKING
FIRST_CROSSING_VALID -> return x_k_stop
NO_CROSSING          -> return prepared_CGLS_base_12
NUMERICAL_BREAKDOWN  -> return prepared_CGLS_base_12
COVARIANCE_OOD       -> return prepared_CGLS_base_12
```

fallback rate 必须按 rig/family 报告。高 fallback rate 可以保安全，但会否定新方法的实用价值；
不得只在 candidate 被采用的子集上报告 gain。

### 6.4 候选集合与冻结顺序

第一轮只允许确定性、可解释、有限集合：

| ID | 数据项/停止 | preconditioner | 作用 |
|---|---|---|---|
| `N1-C0` | unweighted least squares，fixed K | identity | matched CGLS control |
| `N1-C1` | `C_iid_cal` whitening + first crossing | identity | 仅测试噪声尺度 |
| `N1-C2` | `C_camera_cal` whitening + per-camera first crossing | identity | N1 primary |
| `N1-C3` | C2 + frozen Tikhonov damping | identity | simple damping control |
| `N1-C4` | C2 + frozen Huber threshold | identity | outlier robust control |
| `N1-C5` | C2 + fail-closed snapshot state machine | identity | primary safety candidate |

Tikhonov lambda 与 Huber delta 只能由 calibration/selection held-out measurements 选择。候选数、网格、
排序和 tie-break 在运行前冻结；若一个 candidate family NO-GO，不得在同一 opened lock 上继续扫更密网格。

以下只作为 evaluator controls：

- M2.7 exact camera-block oracle，含 `1001F-equivalent` setup；
- M2.8 fixed interpolation 与 truth-oracle alpha envelope；
- `C_generator_exact` 与 true camera bias；
- per-case best-of-CGLS/Huber field oracle；
- Factor Gate B 的 exact/factor controls。Factor Gate B 位于另一 operator/data contract，不能跨数据表
  直接排名；若移植到 N1，必须在同一 A、同一 y、同一预算重新运行，并保留其既有 NO-GO 边界。

learned tau/controller/proximal 暂不进入 N1 primary。只有 C2/C5 先证明 deterministic covariance 与
fail-closed 机制过门后，才可冻结一个小型 learned extension；否则神经 controller 只会增加 truth
leakage 与 HPO freedom。

## 7. 严格同预算与端到端成本

### 7.1 paired-call slot

沿用现有口径但补齐向量账本：

```text
1F   = 一次完整 active-view A(x)
1Adj = 一次对应 A^T(r)
paired_call_slots = max(total_F, total_Adj)
```

现有 learned feature preparation 为 `13F/13Adj`，K-step correction 的 candidate ledger 为

```text
total_F   = 14 + K,
total_Adj = 13 + K,
Kmax      = 10,
hard cap  = 24F / 24Adj / 24 paired slots.
```

主 baseline 可使用同样 `24` paired slots。必须同时报告实际 `(F,Adj)`，不能只写 max；candidate
少一次 Adj 是事实，baseline 不得因为 Python helper 的实现方式得到额外免费 setup。

### 7.2 setup 规则

- 任何依赖当前 geometry 的 `A` probes、norm estimate、sketch、mass、Schur/factor construction 都计入
  online/cold-start F/Adj；
- exact camera-block 的 `1001F-equiv` 继续列为 oracle setup，绝不能藏进 cache；
- calibration-only `C_hat` fitting 不计 reconstruction F/Adj，但单列 acquisition count、CPU time、
  storage、reuse contract 与 `N_reuse=1/100` amortized cost；
- covariance 若在每个 flow-on sample 上重新估计，就是 online setup，必须计时且不得读 current residual；
- audit-camera forward 只计 evaluator calls，不能反馈 candidate 输出。

### 7.3 standalone replay 与成本门

selection 可共享 K=0...10 trajectory，但 freeze 后必须分别 standalone replay：

1. frozen candidate；
2. frozen primary CGLS comparator；
3. frozen Tikhonov/Huber comparator；
4. prepared CGLS base fallback。

每个 replay 报：cold/warm median、p95 wall time、peak RSS/accelerator memory、network inference、
covariance setup/apply、preconditioner apply、F、Adj、evaluation-only calls。N1 默认硬门为：

```text
F <= 24, Adj <= 24,
cold median end-to-end <= 1.5x primary matched CGLS,
cold p95 end-to-end <= 2.0x,
peak memory <= 2.0x,
breakdown/nonfinite rate = 0.
```

若课题组希望换阈值，必须在第一次 N1 run 前改 config 并 hash；结果打开后不能调整。

## 8. SPD、Schur 与数值安全门

对 primary low-rank camera covariance，写 `Sigma_beta=L L^T`、`U=B L`：

```text
C = D + U U^T,
D_ii >= d_min > 0,
Sigma_beta = L L^T >= 0,
S = I + U^T D^(-1) U.
```

这个写法允许 Sigma_beta 为低秩，不需要对它求逆。实际实现应使用 square-root parameterization、
positive diagonal floor 与 Woodbury form。
CPU-float64 decision audit 定义

```text
schur_violation = max(0, -lambda_min(S)) / max(lambda_max(S), 1e-30).
```

硬门：

```text
all D_ii finite and > 0,
all covariance/Schur Cholesky factorizations succeed,
schur_violation <= 1e-10,
||W^T W r - C^(-1)r|| / ||C^(-1)r|| <= 1e-8,
whitening permutation test passes,
camera storage order does not change scores.
```

若后续加入 factor/majorizer preconditioner，还须单独验证 claimed Schur complement 或
`M-H` 的 PSD 安全；exact mass 参与该验证只能是 evaluator audit，不能反馈 candidate 参数。
任何 Schur/majorizer violation 都是 `INVALID_NO_RANKING`，不能用 field mean gain 覆盖。

## 9. 统计单位、指标与门禁

### 9.1 聚合顺序

对每个 candidate-comparator pair：

1. 在同一 case 内先聚合 model seeds，或预先冻结一个单模型/ensemble；
2. nuisance repeats 在同一 field/geometry 内汇总 median、p95 与 worst；
3. families 在同一 base-seed geometry 内保留分层，并形成 geometry-block mean；
4. bootstrap/sign test 只在独立 geometry/base-seed blocks 上进行；
5. 同时发布所有 case、family、rig rows，禁止 pooled-only 图。

### 9.2 primary 与 secondary endpoints

| 层 | 指标 | 角色 |
|---|---|---|
| field | relative-L2 gain vs frozen primary comparator | primary synthetic endpoint |
| structure | H1-seminorm gain | co-primary |
| tail | harm rate for gain `<-1%`、worst gain、p95 loss | hard safety |
| rig | 每 geometry/rig block mean gain | hard safety |
| data | calibrated whitened discrepancy coverage、per-camera max | selection + calibration |
| audit | held-out ray/camera vector relative-L2 与 p95 | truth-free predictive evidence |
| numerics | closure、Schur violation、breakdown | validity |
| cost | F、Adj、setup、wall time、memory | hard deployment-form gate |

`clean_reprojection_relative_l2`、true beta/noise、exact projection error 和 field truth 只能在 candidate
freeze 后进入 evaluator tables；它们不得进入 selector features。

### 9.3 N1 opened-lock 硬门

一个 candidate 只有同时满足下列条件，才可标为
`N1_OPENED_SYNTHETIC_GATE_PASS`：

1. primary cluster-level mean field gain `>=5%`，H1 gain `>=3%`；
2. cluster bootstrap 95% lower bound of mean field gain `>0`；
3. 所有 model-seed means 为正；每个 family mean 与每个 rig/geometry-block mean 均 `>0`；
4. `gain<-1%` 的 case-level harm rate `<=5%`，worst field gain `>=-5%`；
5. selection 与 lock 的 calibrated discrepancy coverage 通过预注册 binomial/conformal check；
6. lock-audit measurement primary endpoint 改善，且任何 rig 的 audit p95 不恶化超过 calibration
   repeatability floor；
7. fallback 也计入全体指标，不能只报告 accepted subset；
8. F/Adj、end-to-end、memory、Schur、closure 与 nonfinite gates 全部通过；
9. candidate 分别不劣于 frozen CGLS、Tikhonov/Huber controls；best-of-classical oracle 单独报告；
10. independent validator 从 frozen rows 重算同一判决。

若 lock 少于 59 个独立 blocks，即使观察到 0 harms，也不得写“95% confidence 下 harm rate <=5%”；
只能写 observed harm count 与 exact interval。

## 10. 判决等级

### 10.1 `INVALID_NO_RANKING`

以下任一项出现，整次 run 无排名意义：

- selection、threshold、checkpoint 或 fallback 读取 field truth、clean y、family、true nuisance；
- covariance 使用 current reconstruction residual 或 clean-signal RMS；
- camera bias 从单帧无独立 prior 联合拟合；
- lock/audit 数据在 candidate freeze 前被访问；
- exact camera-block/mass/norm setup 未计成本却进入 deployable candidate；
- F/Adj cap、standalone replay、Schur/SPD、hash/manifest 或 case partition 失败；
- 把 model seeds、pixels、rays、nuisance repeats 当成独立 field blocks；
- 在打开 lock 后改变 tau、lambda、Huber delta、候选网格或主 baseline。

### 10.2 `N1_NO_GO`

协议有效但发生以下任一项：

- 没有 truth-free eligible candidate；
- discrepancy coverage 或 per-camera coverage 失败；
- field/H1 mean、bootstrap lower bound、family/rig sign、harm/worst 任一失败；
- held-out audit measurement 或 tail gate 失败；
- fallback rate 过高导致总体 gain/tail 不过门；
- cold end-to-end、memory 或调用预算失败。

NO-GO 是有价值结果，应冻结 packet，停止在同一 opened lock 上继续调参。

### 10.3 `MECHANISM_SIGNAL_ONLY`

以下结果最多只能写 mechanism signal：

- exact camera-block、exact covariance、true beta、clean RMS 或 truth-alpha 才改善；
- mean field gain 为正，但 harm、worst、rig 或 family gate 失败；
- support measured residual 改善，但 clean/audit residual 或 field tail 不改善；
- 只有 selection set 通过，opened lock 未打开或失败；
- synthetic train-estimated model discrepancy 有效，但没有真实 calibration contract；
- covariance calibration 有效，但 reconstruction 不优于 matched classical；
- candidate 超预算，只在 expanded budget 下通过；
- audit-camera improvement 存在，但没有 independent 3D truth；这只说明 image-space consistency。

### 10.4 `N1_OPENED_SYNTHETIC_GATE_PASS`

只有第 9.3 节全部通过才使用此标签。它仍然不授权：

```text
fresh/final confirmation,
real BOST field accuracy,
experimental superiority,
CFD validation,
4D generalization,
paper-ready state-of-the-art claim.
```

## 11. 停止条件与下一阶段授权

### 11.1 运行前停止

如果没有独立 covariance-calibration packet、checkpoint hash、truth-deleted selector test、
standalone budget ledger 或 lock manifest，N1 不启动。

### 11.2 运行中停止

- covariance/SPD/Schur/whitening 失败立即 fail closed，不生成性能排名；
- 任一 forbidden input 被 selector 访问立即作废；
- candidate 到 Kmax 无 crossing 时返回 frozen fallback，不扩大 K；
- numerical breakdown 或 nonfinite 立即返回 fallback并计为失败事件。

### 11.3 打开 lock 后停止

- 首次 valid NO-GO 后不在同一 lock 上追加 tau/alpha/lambda 网格；
- 若只差一个 tail case，也不能删除该 case、按 model seed 稀释或改 harm threshold；
- 若 exact oracle 通过而 deployable covariance 失败，只授权改善 calibration source，不授权 learned gate；
- H2/4D、learned proximal 和更大 backbone 在 N1 deterministic gate 前保持关闭。

### 11.4 何时允许 N2

N1 只有在 deterministic C2/C5 证明 covariance calibration、truth-free selection、tail 与成本可以同时
成立后，才授权 N2。N2 可研究 bounded learned tau/proximal，但必须：

- 从 deterministic path 零初始化或严格 fallback；
- 只读 observable features；
- 保留同一个 lock firewall 与预算；
- 与 deterministic N1 candidate 直接比较，而不是只对弱 baseline；
- 不把 discrepancy principle、whitening 或 learned SPD 本身包装成 novelty。

## 12. 需要向何远哲确认的真实数据合同

在 synthetic N1 完成前即可发送以下问题，但没有答案不妨碍继续做 E1 机制研究：

1. 是否有同相机、同曝光、同 background 处理链的 flow-off repeated displacement fields？每个 rig/day
   有多少 repeats，单位是 pixel displacement 还是折射角？
2. camera bias 更像固定 offset、随时间慢漂移、空间低频场，还是由 background registration 产生？
3. 是否存在保证零偏转的 ambient detector region，可否作为 per-frame bias check？
4. 三维重建实际使用几台 camera；能否永久留出一台 camera 或若干 rotation runs 只做 audit？
5. 同一 geometry 会复用多少 flow-on frames？这决定 covariance/setup 能否摊销。
6. 实际 forward model 是否包含 finite aperture、depth of field、camera calibration uncertainty 与 ray bending？
7. 师兄最关心的是 field-L2、density-gradient/front、held-out deflection，还是时序稳定性？
8. 真实数据是否有可公开/可本地留存的 calibration manifest，而不是只给处理后的单帧 observation？

若缺少 flow-off repeats，N1 仍可做 synthetic mechanism；但任何 real-data discrepancy threshold、
practical-significance 或 camera-bias correction 都保持锁定。

## 13. N1 freeze 前清单

- [ ] 新 N1 config 标注 `E1_OPENED_SYNTHETIC_N1_NO_FRESH`。
- [ ] seed/family/geometry/nuisance/audit manifests 在首次 run 前 hash。
- [ ] checkpoint 与 training provenance hash，N1 不重新用 dev truth 选 epoch。
- [ ] covariance fit/holdout packets 独立，candidate 不读取 clean RMS/true nuisance。
- [ ] selector 在物理删除 truth columns 后仍运行且选择不变。
- [ ] primary comparator 是一个冻结可部署方法，不是 per-case field oracle。
- [ ] candidate grid、tau、lambda、Huber delta、tie-break 与 fallback 状态机冻结。
- [ ] F/Adj/setup/application/end-to-end/memory schema 冻结。
- [ ] SPD、Schur、permutation、whitening round-trip tests 先通过。
- [ ] selection audit 与 lock audit 严格分离。
- [ ] model-seed/family/base-seed 聚合顺序冻结。
- [ ] mean、CI、harm、worst、per-rig、audit、cost 均为 binding gates。
- [ ] validator 不 import runner，并有 truth leak、budget、cluster、Schur、hash mutation tests。
- [ ] 首次 valid lock 后无二次调参；NO-GO 也冻结并发布。

## 14. 最短可执行路线

1. 先只实现 `C_iid_cal` 与 `C_camera_cal` 的 calibration/holdout validator，不接 reconstruction。
2. 用现有 opened M2.7 rows 做 regression：证明 naïve measured-residual minimization会复现
   `2113/single_interface` tail harm。
3. 冻结 existing backbone/checkpoint，构造 N1 selection/lock manifests 与 truth-deleted selector。
4. 运行 C0/C1/C2/C5；不先上 learned controller，也不引入 exact mass。
5. 选择后 standalone replay，核对 24F/24Adj、Schur、latency 与 memory。
6. 一次打开 opened lock；按独立 geometry block 报 field/H1/rig/tail 与 audit-camera evidence。
7. 若 NO-GO，定位是 calibration、model discrepancy、tail 还是成本，不在同一 lock 上继续扫参数。

这条路线把 N1 的创新空间放在真实问题上：**有独立 calibration 依据的 covariance、不可辨识 bias 的
正确边缘化、model-discrepancy 与 sensor-noise 的分离、以及可审计的 fail-closed stopping。** 它不会
因为一个 synthetic mean gain 就提前把方法写成成功，也为以后接入实验室三维 BOST 数据保留了明确接口。
