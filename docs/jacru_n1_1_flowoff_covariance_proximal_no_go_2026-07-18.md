# N1.1 flow-off covariance proximal：平均信号很强，安全门仍严格 NO-GO

更新：2026-07-18

证据等级：opened synthetic mechanism audit

正式状态：`N1_1_FLOWOFF_COVARIANCE_PROXIMAL_NO_GO`

## 先说人话

N1.0 已经证明：只看一个残差标量来决定“什么时候停”，无法同时保护三维场尾部和干净观测一致性。N1.1 因此没有再造一个更大的网络，而是先问一个更基础的问题：

> 如果用独立 flow-off 重复帧估计相机偏置均值和 detector covariance，再把 JACRU 或 pooled CNN 的输出拉回一个与噪声相容的加权 Tikhonov 解，能否同时超过 matched classical baseline、保护最坏样本并守住干净目标？

答案是：**平均场误差改善很大，但 14 个候选-模型组合没有一个通过全部门。** 同期配对 flow-off 路线仍稳定伤害一个 development 界面样本；不配对的 structured covariance 能保护 field tail，却破坏 clean-target consistency。连精确 nuisance oracle 也没有闭合这个矛盾。

这不是“算法差一点成功”，也不是“flow-off 没用”。它说明当前 toy 里的主要障碍已经从协方差估计误差推进到 **target-conditioned calibration、模型失配、不可辨识 bias 方向和逐样本安全选择**。

## 1. 实验到底做了什么

### 1.1 三份互斥 flow-off 数据

每个 evaluation case 生成三份稳定哈希、随机流互斥的 repeats：

1. `fit=64`：只估计均值和 covariance；
2. `threshold_cal=64`：只定 discrepancy threshold；
3. `audit=64`：只检查未见覆盖率。

`paired_static` 假设 flow-off 与 flow-on 属于同一 session，共享目标的 camera/component offset；`unpaired_distribution` 只学习偏置分布，不知道目标帧的具体偏置。

候选 payload 不暴露 field truth、clean observation 或目标 nuisance；但生成器仍按每个 case 的 clean-signal RMS 决定噪声尺度。这个“目标条件化”只允许做 synthetic mechanism audit，不能叫部署校准。

### 1.2 低参数 covariance

估计器不是秩最多 63 的 `150×150` 经验 covariance，而是：

```text
C = diagonal iid variance + camera/component shared random effect
```

每个 camera/component group 估一个 iid 方差与 shared 方差，再做 shrinkage 和正定 floor。正式 audit 中：

| 模式 | audit coverage mean | covariance 相对误差 mean / max | 最大条件数 |
|---|---:|---:|---:|
| paired static | 93.490% | 3.540% / 5.036% | 1.450 |
| unpaired distribution | 93.542% | 13.795% / 20.114% | 141.419 |

注意：配置写的是经验 `q=0.95`，但 64 个样本直接取普通 higher quantile 时，有限样本新点覆盖的次序统计量期望约为 `61/65=93.85%`，不是严格 95%。正式结果与这个数一致。N1.2 必须改用 split-conformal 第 `ceil((64+1)×0.95)=62` 个次序统计量，并报告二项置信区间。

### 1.3 anchored covariance-weighted Tikhonov ceiling

以冻结网络输出 `x0` 为中心，求：

```text
min_x  0.5 (Ax-y)^T C^-1 (Ax-y) + 0.5 λ ||x-x0||²

xλ = x0 - A^T (A A^T + λ C)^-1 (Ax0-y)
```

选择满足 calibration discrepancy 的最大 `λ`，即满足噪声门时尽量少改 `x0`。这是 data-dependent center 的确定性 proximal/Tikhonov 更新，**不能称为严格 Bayesian MAP**。

本轮直接组装 dense `A A^T`，只作上界：12 个几何、每个 `150×1000` 且满行秩 150；共 `12,012` 个 unbatched forward-equivalents、204 次 batch forward 和 12 次 SVD。它不在 matched reconstruction budget 内，也不支持速度或部署主张。

## 2. 正式结果

正式运行使用 3 个模型种子、80 epochs、30 个 evaluation cases，得到：

- 60 条 flow-off calibration rows；
- 1,260 条 candidate metric rows；
- 84 条 model-seed aggregate rows；
- 14 个候选-模型最终判决；
- 0 个全部过门，0 个 oracle 过门；
- 运行时间 66.18 s，设备 MPS；这不包含现实采集成本，也不是部署速度。

### 2.1 最接近的 paired structured sensor

| 模型 / split | 对 best matched 的 field gain | H1 gain | continuous-clean-target ratio / base | harm | worst field gain |
|---|---:|---:|---:|---:|---:|
| JACRU / development | +42.133% | +36.742% | 0.619× | 8.33% | -6.054% |
| JACRU / OOD | +34.420% | +31.096% | 0.668× | 0% | +11.975% |
| pooled CNN / development | +41.009% | +35.785% | 0.616× | 8.33% | -8.891% |
| pooled CNN / OOD | +34.292% | +30.379% | 0.667× | 0% | +8.949% |

它过了平均 field、H1、clean target、OOD、target crossing 和数值闭合门，只因 development harm 与 worst 两门失败。失败并非随机散落：`base_seed=2113 / single_interface / case=423593699e2b6baeef3f` 在 JACRU 和 pooled CNN 的三个模型种子上全部受害。

### 2.2 为什么不能说“只差一个异常点”

同一个 case 上，paired structured candidate 的 field-L2 为：

- JACRU：`0.7427–0.7579`；
- pooled CNN：`0.7622–0.7781`；
- matched Huber-PDHG：`0.7146`；
- matched CGLS：`0.9314`。

它明显胜过 CGLS 的场误差并把 clean-target residual 压到约 `0.0146`，但仍系统性输给 Huber 的场误差。这个样本正好暴露了“最强 data consistency”和“最强 robust field recovery”不是同一个经典对照。不能为了均值好看而删掉它。

### 2.3 exact nuisance 也没有救回

`paired_exact_mean_iid_sensor_oracle` 直接使用 synthetic generator 的精确持久偏置均值和精确 IID covariance，仍然得到：

- JACRU development harm `8.33%`、worst `-6.169%`；
- pooled CNN development harm `8.33%`、worst `-9.002%`。

因此，本 toy 中增加 flow-off 帧数、把均值估得更准，不能单独解决受害界面样本。

### 2.4 unpaired structured 的另一种失败

unpaired structured 能把 JACRU development harm 降到 0、worst 提到 `+0.937%`，但 continuous-clean-target ratio 变成 `1.207× / 1.401×`（development / OOD）；pooled CNN 为 `1.152× / 1.432×`。它把 camera-shared 方向当随机效应降权，保护了场尾部，却没有充分校正观测一致性。

truth-residual oracle 对 raw center 更安全，但 clean-target ratio 高达约 `8.86×–9.05×`，而且直接读取 truth，只能用于解释冲突。

## 3. 新增的 post-open raw-center 安全审计

正式协议只比较了 matched classical references。打开结果后，我们补问：

> correction 相比它自己的 raw learned proposal，是否真的更安全？

这个问题没有预注册，因此单独标为 `POSTOPEN_RAW_CENTER_SAFETY_GAP_CONFIRMED`，不能修改正式 N1.1 判决，也不能授权算法。

| 候选 / 模型 / split | mean gain vs raw | >1% harm | worst gain vs raw |
|---|---:|---:|---:|
| paired structured / JACRU / development | +0.716% | 27.78% | -22.662% |
| paired structured / JACRU / OOD | +8.757% | 7.41% | -6.120% |
| paired structured / pooled / development | -2.394% | 38.89% | -23.229% |
| paired structured / pooled / OOD | +8.012% | 9.26% | -2.860% |
| unpaired structured / JACRU / development | +2.141% | 22.22% | -16.774% |
| unpaired structured / pooled / development | -0.910% | 38.89% | -23.385% |

六项 raw-center diagnostic safety checks 下，可观测、非 oracle 候选通过数仍为 0。下一协议必须把“相对 raw center 不伤害”与“相对 matched classical 有收益”同时冻结，而不能只选其中一个参照。

## 4. 独立红队发现的协议缺口

这些缺口不会把本轮 NO-GO 变成成功，但限制了它能证明什么：

1. `--seed-limit` / `--epochs` 没有写进 summary；scratch 若误用默认输出目录可覆盖正式产物。N1.2 必须记录 run mode、完整 CLI、有效 seeds/epochs，并用临时目录原子发布。
2. flow-off 噪声尺度来自每个测试目标的 clean RMS，paired calibration 还共享目标的真实 camera bias；这是对候选有利的 target-conditioned synthetic ceiling。
3. 冻结哈希没有覆盖实际读取的 `matched_baseline_rows.csv`、T0 runner、fixture、模型与 scoring 的全部传递依赖；下一轮要冻结 checkpoint 与完整依赖清单。
4. 普通 empirical 95th quantile 不是 64 样本下的 finite-sample 95% conformal coverage。
5. exact covariance oracle 的最终 `calibration_valid` 仍错误复用了 estimated-covariance coverage；本次两套 coverage 都未改变总 NO-GO，但实现必须拆开。
6. clean-target 指标是同一个 voxel operator `A` 对 continuous clean target 的 residual，不是独立 renderer 或 held-out camera。
7. 当前只用 global discrepancy；没有 per-camera maximum upper gate，也没有防止过度正则的 lower gate。
8. candidate cache 同时持有 selection、audit 与 oracle 对象；虽然当前 deployable 分支没有读取 lock/truth，下一版仍应使用能力受限的 payload，结构上阻止误读。
9. `paired_isotropic_sensor` 实际是 isotropic proximal + structured selector，名称需要拆清。
10. T0 checkpoint 曾用 development truth 选取，development 只能继续作为 opened debugging evidence。
11. 独立 validator 能重算所有已发布 scalar aggregates、门与 authorization，但公开包没有逐样本 prepared-base denominator，因此不能从 `reference_rows.csv` 重新推导 `clean_reprojection_ratio_to_base` 本身；N1.2 必须发布该分母或它的不可变摘要。

## 5. 这轮真正学到了什么

### 可以说

- 在当前 opened synthetic toy 中，same-session flow-off mean correction 有很强的平均机制信号。
- covariance estimation 本身不是主要失败源，因为 exact nuisance oracle 仍失败。
- camera-shared covariance 能改变 field-tail 与 clean-target 的 Pareto 位置，但不能自动识别真实 bias 与可解释场。
- 同一个 `single_interface` case 跨两种网络、六个种子稳定受害，是下一步必须解释的反例。
- dense anchored GLS-Tikhonov 的代数、support、尺度不变性与物理 residual closure 已被测试。

### 不能说

- 不能说提出了优于 DeepONet、FNO、FFNO、NeRIF、TDBOST 或组内方法的新算法。
- 不能说已经得到严格 95% coverage。
- 不能说 continuous-clean-target residual 是 independent renderer 验证。
- 不能说 synthetic flow-off 等于真实 OERF detector covariance。
- 不能说 dense `AA^T` 路线可部署、快速或在相同成本下胜出。
- 不能打开 fresh/final，也不能写真实 BOST 泛化结论。

## 6. N1.2 的冻结路线

下一轮不应再扩大网络，而应依次完成：

1. **session-level calibration**：一个 calibration packet 服务同一 rig/session 的多个场；噪声尺度只来自 flow-off/background intensity 等部署可见量。
2. **finite-sample conformal threshold**：fit / threshold-cal / audit 三分，使用第 62 个次序统计量和二项置信区间。
3. **candidate-specific coverage**：每个 mean/covariance/threshold 组合使用自己的 audit coverage；oracle 绝不借 estimated gate。
4. **多尺度 discrepancy**：同时冻结 global upper、per-camera max upper 与 lower gate。
5. **双参考安全门**：同时对 strongest matched classical 与 raw learned center 报 mean、cluster tail、harm、worst。
6. **model-mismatch floor**：单列 `A_voxel x_true_grid - y_clean_continuous`，不得混进 sensor covariance。
7. **正式 controls**：no correction、简单 damping/interpolation、IID anchored Tikhonov、structured GLS、whitened CGLS/LSQR、Huber/Student-t。
8. **matrix-free ceiling**：用固定 K-step Lanczos / multi-shift solve 评估 λ，不按候选重复组装 `AA^T`。
9. **不可变运行清单**：冻结 checkpoint、CLI、环境、Git commit、输入 CSV、fixture、scoring 与所有传递依赖哈希。
10. **新数据后再学 operator**：只有经典 covariance-proximal route 同时过门，才学习 bounded `λ`、robust weight 或 proximal step；网络输出必须受 fail-closed 层约束。

## 7. 现在给何远哲师兄的四个具体问题

1. 同一固定 geometry / exposure / background 下，是否能提供按时间顺序保存的 64–200 帧 flow-off displacement 或 reference repeats？
2. flow-off 与 flow-on 是否同一 session、会不会重拍背景；camera bias 在多长时间尺度上近似稳定？
3. 能否永久留出一台 camera、一个 ray block 或第二 session，只做 audit，不进入 calibration、reconstruction 与 threshold selection？
4. 对 `base_seed=2113 / single_interface` 这类薄界面反例，实验上更可能对应 camera bias、有限孔径、标定漂移、离散化误差还是 robust data term 不足？组里是否有 phantom、CFD/PIV/pressure 或 high-fidelity renderer 可作独立证据？

## 8. 复现入口

- [冻结配置](../demo_t16_operator/configs/jacru_n1_1_flowoff_covariance_proximal_postopen_v1.json)
- [正式 runner](../site_tools/run_jacru_n1_1_flowoff_covariance_proximal.py)
- [独立 validator](../site_tools/validate_jacru_n1_1_flowoff_covariance_proximal.py)
- [正式结果机器摘要](../demo_t16_operator/results/jacru_n1_1_flowoff_covariance_proximal_postopen_public/summary.json)
- [正式 evidence checksums](../demo_t16_operator/results/jacru_n1_1_flowoff_covariance_proximal_postopen_public/checksums.sha256)
- [raw-center post-open audit runner](../site_tools/audit_jacru_n1_1_raw_center_postopen.py)
- [raw-center 结果机器摘要](../demo_t16_operator/results/jacru_n1_1_raw_center_postopen_audit_public/summary.json)
- [raw-center audit checksums](../demo_t16_operator/results/jacru_n1_1_raw_center_postopen_audit_public/checksums.sha256)

正式 evidence 与 raw-center audit 的 checksums 分开保存；任何文件变化都应 fail closed。
