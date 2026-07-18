# N2-PVGR-N2 独立证据审计

> 审计日期：2026-07-18
>
> 审计对象：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1` 及其直接生成代码
>
> 审计方式：只读静态核对、JSON/CSV 独立重算、manifest 哈希核验和小规模单元测试；未重跑正式 9 格实验
>
> 总结论：`PASS WITH WARNINGS — DEVELOPMENT MECHANISM EVIDENCE ONLY`

## 1. 结论先行

当前证据足以支持下面这个窄结论：

> 在当前三个 development rig、两个已打开的合成 phantom family、三个应力倍率组成的 9 个开发格中，解析 OCBH 与同一离散中央差分/RK4 程序的 forward-mode bend-homotopy JVP 在 float64 误差内一致；OCBH 通过预置在当前配置文件中的开发门，且修正后的 Picard-1/2 是更强的描述性前向基线。

当前证据**不支持**把 9/9 写成自有算法成功、论文成功、真实 BOST 成功、三维重建成功、跨分布泛化成功或相对 DeepONet/FNO/FFNO 的成功。`result.json` 中四个授权字段均为 `false`，这是正确且必须保留的边界。

审计没有发现会推翻现有 9 个开发格机器结果的数字错误。主要问题是证据包完整性和确认性不足，而不是当前表内数字算错：manifest 没有覆盖全部生成物和测试，配置与结果均未进入 Git 历史，无法仅凭仓库证明门槛先于结果冻结；Picard 虽然更强，但没有被纳入同一机器门。

## 2. 审计范围

逐项读取并核对：

- `demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/result.json`
- `metrics.csv`、`teacher_metrics.csv`、`reference_sentinel.csv`、`timing.csv`
- `manifest.json`、`config_snapshot.json`、`summary.md` 和结果图
- `demo_t16_operator/configs/n2_pvgr_n2_operator_consistent_bridge_v1.json`
- `demo_t16_operator/run_n2_pvgr_n2_operator_consistent_bridge.py`
- `demo_t16_operator/discrete_rk4_jvp_predictor.py`
- `demo_t16_operator/operator_consistent_homotopy_predictor.py`
- `demo_t16_operator/picard_curved_ray_baseline.py`
- 四个对应测试文件
- `docs/n2_pvgr_n2_operator_consistent_bridge_2026-07-18.md`

没有打开 reserved phantom，没有访问真实数据，没有运行 96 格实验，也没有重生成正式结果目录。

## 3. PASS / WARN / FAIL 总表

| ID | 等级 | 审计项 | 结论 |
|---|---|---|---|
| E01 | PASS | manifest 已列 13 个文件 | 13/13 的当前 SHA-256 与 byte size 均匹配 |
| E02 | PASS | JSON 与 CSV | 36 个方法行、9 个教师行、9 个 sentinel 行、15 个计时行逐字段一致 |
| E03 | PASS | 配置快照 | `config_snapshot.json` 与源配置 JSON 语义相同；仅数组排版不同 |
| E04 | PASS | OCBH 主门 | 8 项主门在 9/9 个 OCBH 开发格通过 |
| E05 | PASS | 解析/教师等价门 | 9/9 通过；最坏 output relative-L2 为 `2.159156e-14` |
| E06 | PASS | H256/H512 sentinel | 9/9 通过；最坏 output 为 `4.364822e-5`，matched residual 为 `0.004257435` |
| E07 | PASS | OCBH 墙钟门 | 3/3 rig 通过；最坏 p90/H128 p10 为 `0.150789733` |
| E08 | PASS | OCBH logical query 门 | 9/9 通过；独立重算为 `115136/286720 = 0.4015625` |
| E09 | PASS | Picard off-by-one 修复 | 最终路径确有额外七点输出重算；P1/P2 成本分别为 2/3 个七点 batch |
| E10 | PASS | exact-high 输入隔离 | 三个 predictor 均不接受 H128/H256/H512 结果作为输入；primary OCBH 报告 exact-high 调用为 0 |
| E11 | PASS | 声称边界 | runner、result、summary 和报告都明确关闭 reserved、真实数据、重建、神经算子和论文授权 |
| W01 | WARN | manifest 覆盖不完整 | `config_snapshot.json`、`summary.md`、说明报告和测试文件未被 manifest 哈希覆盖 |
| W02 | WARN | 缺少可证明的预注册时序 | 新配置、runner 和结果当前不在 Git 历史中；manifest 只能证明当前自洽，不能证明阈值先于结果冻结 |
| W03 | WARN | Picard 不是机器门对象 | primary、timing、query pass count 都只计算 OCBH；Picard 数字是描述性强基线，不是通过同门的候选 |
| W04 | WARN | logical query 不是总成本 | OCBH 的 4 次坐标 reverse sweep、autodiff 算术、内存、host sync 和 peak RSS 不在 point-query 数中 |
| W05 | WARN | 计时仍是同进程实现证据 | candidate 与 high 共进程；未测 peak RSS、field JVP/VJP 和 host scalar synchronization |
| W06 | WARN | 教师 `5.4-6.0 s` 未进入正式计时表 | 报告中的教师耗时没有对应 `timing.csv` 行，只能视为未归档的开发观察 |
| W07 | WARN | “逐元素相同”措辞过强 | 证据是 float64 tolerance 下 relative-L2 约 `1e-14`，不是 bitwise `torch.equal` |
| W08 | WARN | “Picard 更准”需要限定指标 | matched residual 在 9/9 格优于 OCBH；global no-harm 仅 6/9，Q95 no-harm 仅 4/9 的数值更低 |
| W09 | WARN | sentinel 只能排除部分步长误差 | H256/H512 共用 renderer、场网格与中央差分步长，不能发现共同实现错误或空间离散误差 |
| W10 | WARN | 配置存在未执行字段 | `teacher_discrete_jvp_step_count` 当前等于 128，但 runner 实际直接使用 `execution_step_count`；两个 hard-conclusion 字符串也由 runner 硬编码 |
| W11 | WARN | 9 格不是 9 个独立统计重复 | 两个 field seed 被多个 rig/stress 重用；当前没有 field-level CI、held-out family 或独立实验终点 |
| C01 | FAIL/CLOSED | “OCBH 是最佳前向算法” | Picard-1/2 在当前 matched residual 和墙钟上更强，不能提出该声称 |
| C02 | FAIL/CLOSED | “优于 DeepONet/FNO/FFNO” | 未在同一数据、三维重建任务和预算下运行这些模型 |
| C03 | FAIL/CLOSED | “真实反应流或实验泛化” | 没有真实 OERF 数据、实验几何、噪声合同或独立物理终点 |
| C04 | FAIL/CLOSED | “论文级或创新授权” | `paper_claim_authorization=false`；历史新颖性与完整基线尚未闭环 |
| C05 | FAIL/CLOSED | “reserved audit 已通过” | 两个 reserved family 明确未打开 |

这里的 `FAIL/CLOSED` 是对越界声称的判定，不是说当前开发实验失败。

## 4. 机器结果与 CSV 对账

### 4.1 行数与字段

独立解析结果为：

| 文件 | 行数（不含表头） | 对应 JSON | 对账 |
|---|---:|---|---|
| `metrics.csv` | 36 | 4 methods x 9 cells | PASS |
| `teacher_metrics.csv` | 9 | `teacher_rows` | PASS |
| `reference_sentinel.csv` | 9 | `reference_sentinel_rows` | PASS |
| `timing.csv` | 15 | 5 methods x 3 rigs | PASS |

CSV 中空 gate 字段只出现在非 OCBH 方法，因为 runner 只给 `operator_consistent_homotopy` 构建 primary gates。这些空值不是 Picard/N1 的 gate failure；它们表示“没有接受该门的判定”。因此不能用 CSV 的 `all_primary_gates_pass=False` 宣称 Picard 失败，也不能把 Picard 算入 9/9。

### 4.2 四方法最坏格

| 方法 | matched residual rel-L2 最大 | H256 global no-harm 最大 | Q95 no-harm 最大 | risk Spearman 最小 | OCBH 主门通过数 |
|---|---:|---:|---:|---:|---:|
| N1 continuous affine | `0.06854056` | `1.77360412` | `1.68767517` | `0.92591575` | N/A |
| OCBH | `0.01337110` | `1.06440736` | `1.05611208` | `0.99867216` | 9/9 |
| Picard-1 | `0.00170918` | `1.00101472` | `1.00094397` | `0.99981685` | N/A |
| Picard-2 | `0.000498265` | `1.00098609` | `1.00090838` | `0.99990842` | N/A |

报告中 wrinkled-wide 的两个旧失败也可逐项复核：

| stress | N1 global no-harm | OCBH global no-harm | OCBH 门 `<=1.1` |
|---:|---:|---:|---|
| 3x | `1.14282834` | `1.00681734` | PASS |
| 10x | `1.77360412` | `1.06440736` | PASS |

这些数字支持“当前开发格的机制修正消除了原两个 no-harm failure”。它们不支持跨场泛化。

## 5. 门槛逐项重算

### 5.1 OCBH primary gates

| 门 | 9 格最坏值 | 阈值 | 判定 |
|---|---:|---:|---|
| base output relative-L2 最大 | `1.668316e-16` | `<=2e-10` | PASS |
| matched residual relative-L2 最大 | `0.01337110` | `<=0.02` | PASS |
| corrected residual variance ratio 最大 | `0.000670352` | `<=0.01` | PASS |
| per-ray risk Spearman 最小 | `0.99867216` | `>=0.99` | PASS |
| valid-ray fraction 最小 | `1.0` | `>=1.0` | PASS |
| candidate-to-H256 relative-L2 最大 | `0.000951873` | `<=0.002` | PASS |
| global reference no-harm 最大 | `1.06440736` | `<=1.1` | PASS |
| Q95 reference no-harm 最大 | `1.05611208` | `<=1.1` | PASS |

重算的 primary pass count 为 `9/9`，与 `result.json` 一致。这里的 primary candidate 只有 OCBH。

### 5.2 解析 OCBH 对 forward-mode 教师

| 门 | 9 格最坏值 | 阈值 | 判定 |
|---|---:|---:|---|
| output tangent relative-L2 | `2.159156e-14` | `<=2e-10` | PASS |
| position tangent relative-L2 | `4.950399e-15` | `<=2e-10` | PASS |
| direction tangent relative-L2 | `5.379379e-15` | `<=2e-10` | PASS |
| teacher valid-ray fraction | `1.0` | `>=1.0` | PASS |

这证明两个当前实现对同一个离散 homotopy 导数一致，不证明该 homotopy 是实验世界的完整模型。

### 5.3 H256/H512 sentinel

| 门 | 9 格最坏值 | 阈值 | 判定 |
|---|---:|---:|---|
| H256-to-H512 output relative-L2 | `4.364822e-5` | `<=1e-4` | PASS |
| 256-to-512 matched residual relative-L2 | `0.004257435` | `<=0.01` | PASS |

因此可以把 H256 称为“当前 9 格、当前 renderer 下通过步数 sentinel 的 synthetic evaluator”。不能称其为真值。H256/H512 使用同一代码、同一场网格和同一中央差分步长，common-mode error 仍不可见。

### 5.4 墙钟与 query 门

OCBH 三个 rig 的最坏 `candidate p90 / H128 p10` 为 `0.150789733 <= 0.25`，3/3 通过。该比值只对应本机 CPU、float64、当前 Python/PyTorch 批处理实现。

query 算术如下，`r=64`、`s=128`：

| 方法 | 公式 | logical scalar point queries | 相对 H128 | dispatch / sweep 元数据 |
|---|---|---:|---:|---|
| OCBH | `7*r*(2s+1)` | `115136` | `0.4015625` | 7 batched interpolation dispatches + 4 coordinate reverse sweeps |
| forward JVP teacher | `35*r*s` | `286720` | `1.0` | 4480 interpolation dispatches + 1 forward JVP |
| Picard-1 | `(1+1)*7*r*s` | `114688` | `0.4` | 1 path sweep + 1 final-output batch |
| Picard-2 | `(2+1)*7*r*s` | `172032` | `0.6` | 2 path sweeps + 1 final-output batch |
| H128 | `35*r*s` | `286720` | `1.0` | 4480 interpolation dispatches + 1 exact-high evaluation |

OCBH 的 `0.4015625 <= 0.45` 在 9/9 行通过。注意 `point query` 只是在一个坐标上评价一次标量网格原语的逻辑计数；不同方法的 autodiff、张量尺寸、内存流量和 Python dispatch 完全不同，所以不能把该表直接翻译成硬件无关复杂度或加速比。

## 6. 三个 predictor 的代码审计

### 6.1 `discrete_rk4_jvp_predictor.py` — PASS with scope warning

- `bend_strength` 只乘在 trajectory curvature 上，见约第 204-225 行。
- 输出仍由完整 central-difference midpoint integrand 计算，见约第 286-313 行。
- `torch.func.jvp` 在 `epsilon=0` 计算完整离散程序的一阶切线，见约第 358-387 行。
- 返回前检查非有限数、预测路径 stencil margin、位置/方向线性化半径和归一化切线正交性，失败 ray 被赋予无限 risk，见约第 396-452 行。
- 公共入口不接收任何 high output；字段和相机输入被 detach。
- 逻辑 query `35*r*s` 与 `_ray_rhs` 每步 4 个七点 stage 加一个七点 midpoint output 相符。

小测试还验证了 `epsilon=1` 与已有同步数 high route 一致，以及 JVP 与中心有限差分一致。它仍只是当前 high renderer 的自动微分教师，不是独立物理 oracle。

### 6.2 `operator_consistent_homotopy_predictor.py` — PASS with deployment warning

- 标量分母的坐标导数来自相同插值原语的 automatic gradient；中央差分分子的 Jacobian 直接对六查询算子求导，见约第 163-206 行。
- `position_jacobian` 和 raw-direction `direction_jacobian` 的矩阵顺序与报告公式一致，见约第 247-269 行。
- 轨迹切线传播把 `A/B` 置零，只使用 forcing，见约第 351-368 行；`A_delta/B_delta` 只进入 output integrand 路径导数，见约第 379-401 行。
- endpoints 与 midpoints 共 `2s+1` 个系数点，query 计数为 `7*r*(2s+1)`，见约第 460-493 行。
- 输出全部 detach，`reverse_mode_field_vjp_evaluations=0`。因此它当前不是可训练三维重建的 differentiable renderer，这一点与报告边界一致。

解析实现与 forward JVP 的 `1e-14` 量级差异可判为 float64 数值等价；应避免写成 bitwise 或数学上“逐元素完全相同”。

### 6.3 `picard_curved_ray_baseline.py` — off-by-one 修复 PASS

代码执行顺序是：

1. 第 328-435 行完成指定次数的冻结路径 curvature sweep，并更新 `positions/directions`。
2. 第 437-485 行在**最终更新后的路径**重新取 midpoint、做七点场查询并计算 `output_curvature`。
3. 第 487-502 行对最终 curvature 积分并投影为 detector deflection。
4. 第 515-531 行把额外输出 batch 计入 `(sweeps+1)*7*r*s`。

测试 `test_one_and_two_sweeps_preserve_the_declared_frozen_history_contract` 验证 `two.curvature_history[1] == one.output_curvature`；query monkeypatch 测试实际观察到 Picard-2 的三次、每次 `7*r*s` 调用。因此旧版“P1 实际仍返回 straight output”的 off-by-one 在当前文件中已修复。

Picard 的 fail-closed 是 whole-batch exception；只要函数返回，`valid_mask` 全为 true。当前没有收敛证明、caustic 处理或真实 detector displacement 标定。

## 7. runner 与 machine decision 审计

runner 的关键逻辑正确：

- 验证 CPU/float64、128/256/512 递增步数和 reserved family 不相交。
- primary OCBH 不接受 high output；H128/H256/H512 只在 runner 内用于事后 evaluator。
- machine decision 只有 primary、teacher、sentinel、OCBH timing 和 OCBH query 五组计数全部满足时才变为 `MECHANISM_BRIDGE_SIGNAL_ONLY_96_CELL_RECONSTRUCTION_AND_REAL_DATA_GATES_CLOSED`。
- 无论这些开发门是否通过，`development_bridge_authorization`、`reserved_audit_authorization`、`real_data_authorization`、`paper_claim_authorization` 都固定为 `false`。

需要修复但不影响本次数字的配置契约缝隙：

- `teacher_discrete_jvp_step_count` 写入配置却未被 runner 读取或校验；教师实际使用 `execution_step_count`。
- `hard_conclusion_if_all_bridge_screens_pass` 和 `hard_conclusion_otherwise` 的当前值与 runner 一致，但 runner 使用硬编码字符串，未来配置漂移不会自动报错。
- Picard 的 gates、timing gates 和 query gates 没有进入 machine decision；目前只能作为描述性 baseline。

## 8. 报告逐项核对

`docs/n2_pvgr_n2_operator_consistent_bridge_2026-07-18.md` 中以下核心数字均与机器结果一致：

- OCBH/teacher 最坏 output tangent `2.16e-14`；
- N1 matched residual 最坏 `6.85%`，OCBH `1.34%`；
- wrinkled 3x/10x global no-harm 从 `1.143/1.774` 到 `1.007/1.064`；
- OCBH/Picard-1/Picard-2 的 worst matched、no-harm、Q95、risk 表；
- OCBH timing `0.147-0.151` 和 query ratio `0.4015625`；
- H256/H512 sentinel 的 `4.365e-5` 与 `0.004257`；
- 9/9 primary、9/9 teacher、9/9 sentinel、3/3 OCBH timing 和 9/9 OCBH query；
- 所有越界声称均明确关闭。

需保留三个措辞警告：

1. “两个实现逐元素相同”应理解为 float64 tolerance 内一致，不是 bitwise equality。
2. “Picard 更准”对 matched residual 是逐格 9/9 成立；对 global no-harm 只有 6/9 数值更低，对 Q95 no-harm 只有 4/9 数值更低。更稳妥的表述是“Picard 在 matched residual 和九格 worst-case 汇总上更强”。
3. 教师 `5.4-6.0 s` 没有写入正式 `timing.csv`；H128 的正式记录约为 `0.460-0.469 s`。前者在补入机器计时证据前不应作为正式性能数据引用。

## 9. manifest 审计

manifest 当前列出的 13 项全部通过 SHA-256 和 byte-size 核验，覆盖：配置、源配置、旧 N1 结果、runner、三个 predictor、四个机器表/结果和图。

但它没有覆盖：

- `config_snapshot.json`
- `summary.md`
- `docs/n2_pvgr_n2_operator_consistent_bridge_2026-07-18.md`
- 四个测试文件

此外这些 N2 文件当前没有 Git commit 历史，因此当前 manifest 只能回答“这些文件现在彼此匹配”，不能回答“门槛是在看结果之前冻结的”。在进入 96 格或 reserved audit 前，应先把 config、runner、测试和 claim boundary 单独提交并记录 commit SHA，再在独立输出目录运行。

## 10. 本次实际执行的小测试

执行命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  demo_t16_operator/test_discrete_rk4_jvp_predictor.py \
  demo_t16_operator/test_operator_consistent_homotopy_predictor.py \
  demo_t16_operator/test_picard_curved_ray_baseline.py \
  demo_t16_operator/test_run_n2_pvgr_n2_operator_consistent_bridge.py
```

结果：

```text
60 passed, 18 warnings in 11.77s
```

18 个 warning 均来自 PyTorch `torch.jit.script` deprecation，不是数值失败。本次禁用了 bytecode 和 pytest cache，未写入仓库测试缓存。

## 11. 审计后处置说明（主任务整合者补记）

本节不是独立审计员的原始判定，也不改变上面的 `PASS WITH WARNINGS` 结论。收到审计后，主任务
整合者已将 `config_snapshot.json`、`summary.md`、说明报告和四个核心测试加入 manifest 生成逻辑，
当前清单由审计时的 13 项扩展为 20 项。该修改修补了 W01 所指的覆盖缺口，但扩展后的 20 项尚未
接受第二轮独立审计；W02 的预注册时序问题和 W03-W11 的证据边界仍然原样成立。

未执行正式 runner，也未覆盖或重生成原结果目录。

## 11. 可复现审计命令

### 11.1 查看授权与计数

```bash
jq '{machine_decision,
     development_bridge_authorization,
     reserved_audit_authorization,
     real_data_authorization,
     paper_claim_authorization,
     primary:[.primary_screen_pass_count,.primary_screen_required_count],
     teacher:[.teacher_screen_pass_count,.teacher_screen_required_count],
     sentinel:[.reference_sentinel_pass_count,.reference_sentinel_required_count],
     timing:[.timing_screen_pass_count,.timing_screen_required_count],
     queries:[.point_query_screen_pass_count,.point_query_screen_required_count]}' \
  demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/result.json
```

### 11.2 重算四方法最坏值

```bash
jq -r '
  .method_rows | group_by(.method_id)[] |
  [.[0].method_id,
   length,
   (map(.metrics.matched_residual_prediction_relative_l2)|max),
   (map(.metrics.candidate_reference_error_to_high_execution_reference_error_ratio)|max),
   (map(.metrics.candidate_q95_reference_error_to_high_execution_q95_ratio)|max),
   (map(.metrics.per_ray_risk_spearman)|min)] | @tsv
' demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/result.json
```

### 11.3 核验 manifest

```bash
jq -r '.files[] | [.sha256,.bytes,.path] | @tsv' \
  demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/manifest.json |
while IFS=$'\t' read -r expected_hash expected_bytes file_path; do
  actual_hash="$(shasum -a 256 "$file_path" | awk '{print $1}')"
  actual_bytes="$(stat -f '%z' "$file_path")"
  test "$actual_hash" = "$expected_hash" && test "$actual_bytes" = "$expected_bytes" || exit 1
done
```

### 11.4 检查 Picard 最终路径重算与成本

```bash
nl -ba demo_t16_operator/picard_curved_ray_baseline.py | sed -n '328,531p'
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  demo_t16_operator/test_picard_curved_ray_baseline.py \
  -k 'frozen_history_contract or query_accounting'
```

### 11.5 正式重放入口（本审计未执行）

为了不覆盖原证据，未来应写到新目录：

```bash
.venv/bin/python demo_t16_operator/run_n2_pvgr_n2_operator_consistent_bridge.py \
  --config demo_t16_operator/configs/n2_pvgr_n2_operator_consistent_bridge_v1.json \
  --output /tmp/n2_pvgr_n2_operator_consistent_bridge_audit_replay
```

重放后必须重新核验哈希、JSON/CSV 行级一致性和机器环境；不能只比较 summary 文本。

## 12. 最终审计判定

**PASS：**当前机器证据在自身声明的 development-only 范围内自洽；核心数字、门槛、query 算术、Picard off-by-one 修复和 H256/H512 sentinel 均可由现有文件复核。解析 OCBH 与 forward JVP 的一致性是可信的当前实现证据。

**WARN：**确认性、独立性和完整复现链尚不足。尤其是阈值冻结时序不可由 Git 证明、manifest 覆盖不完整、Picard 未进入同门、计时同进程且未测内存/host sync、H256/H512 存在 common-mode blind spot。

**FAIL/CLOSED：**任何把 9/9 扩写为算法、论文、真实、三维重建或泛化成功的说法都不通过本审计。下一阶段必须先冻结 96 格 field-level 设计和独立运行证据，再做 differentiable reconstruction、等预算神经算子比较、cone-ray 与真实 OERF 合同。
