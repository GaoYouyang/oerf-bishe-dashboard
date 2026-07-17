# JACRU M2 严格预注册门：跨样本残差算子的单次 Fresh 证伪协议

**协议日期：** 2026-07-17  
**当前状态：** `DRAFT_PROTOCOL_ONLY / NOT_FROZEN / M2_IMPLEMENTATION_NOT_AUDITED_OR_FROZEN`  
**数据状态：** `TRAIN_AND_DEVELOPMENT_MAY_BE_CONSTRUCTED / OOD_AND_FRESH_NOT_CONSTRUCTED`  
**Fresh 授权：** `false`  
**允许的最高证据等级：** 通过全部门后仍仅为
`INTERNAL_SYNTHETIC_FRESH_SIGNAL_WITH_INDEPENDENT_RENDERER`。

> 本文件定义如何让 M2 失败，不记录 M2 已经通过。只有协议、代码、数据 manifest、
> checkpoint、scorer、环境与全部 SHA-256 在同一 Git 提交中冻结，且独立 validator 判定
> `READY_FOR_OOD_SINGLE_OPEN` 后，预注册才生效。当前没有这些产物，因此不得打开 OOD 或
> fresh，不得把本文状态写成 `FROZEN`。仓库中存在 M2 草稿代码或 config 也不改变这一状态；
> 未经过本协议 preflight、预算审计和 hash freeze 的实现一律视为 opened development work。

> **2026-07-17 T0 状态说明：**仓库中的 `jacru_m2_learned_residual_t0_v1.json` 已运行一个
> 小规模、已经打开的 exploratory geometry+morphology `ood` screen。这个目录名中的 `ood`
> 只表示相对该 T0 train split 的分布变化；它不是本协议尚未构造、尚未封存的 confirmatory
> OOD，更不授权冻结或打开 fresh。T0 的 reprojection NO-GO 只能用于修改 M2.1 与完善本草案。

---

## 0. 继承的负证据与本轮唯一授权

本协议继承而不改写以下证据：

| 轮次 | 候选 | mean field relative-L2 | mean H1 | measured reprojection | 已验证判决 |
|---|---|---:|---:|---:|---|
| M0 | JACRU + bias | 1.987836 | 3.174674 | 1.913683 | `NO_GO` |
| M0.1 | 尺度修复、随机平面、关闭初始 gate | 0.769049 | 1.138376 | 0.408105 | `NO_GO` |
| M1 | CGLS-18 + 6 对逐样本残差更新 | 0.495027 | 1.011757 | 0.010213 | `NO_GO` |
| CGLS-24 | 经典基线 | 0.498911 | 1.022353 | **0.003739** | baseline |
| Huber-PDHG-24 | 经典基线 | **0.480119** | **0.874186** | 0.114924 | current champion |

关键审计事实：

1. M0 的固定 `x` 平面在读取观测前就在单界面 case 上得到
   `F1@1dx = 1.000`；最终均值为 `0.973958`，没有数据驱动的界面增量。
2. 同一初始化在平滑场上的初始假阳性率为 `100%`。因此 M0 的界面收益已经作废。
3. M0.1 的尺度修复使 field-L2 改善 `61.31%`，但仍比 Huber-PDHG 差
   `60.18%`，说明修复了实现问题，没有建立方法优势。
4. M1 比 CGLS 仅改善 `0.78%`，仍比 Huber-PDHG 差 `3.11%`，H1 差
   `15.74%`；其七个 gate 中只有 reprojection 通过。
5. 现有 validator 只授权 `continue_learned_residual_operator=true`，同时冻结
   `claim_jacru_superiority=false`、`claim_interface_gain=false` 和
   `open_fresh_or_final_split=false`。

证据入口：

- [M0-M1 负证据总判决](jacru_m0_m1_negative_evidence_2026-07-17.md)
- [M0 summary](../demo_t16_operator/results/jacru_m0_synthetic_gate_public/summary.json)
- [M0.1 summary](../demo_t16_operator/results/jacru_m0_1_diagnostic_public/summary.json)
- [M1 summary](../demo_t16_operator/results/jacru_m1_cgls_residual_diagnostic_public/summary.json)
- [data-free initialization audit](../demo_t16_operator/results/jacru_m0_initialization_audit_public/summary.json)
- [chained validator output](../demo_t16_operator/results/jacru_m0_m1_evidence_validation.json)

本轮唯一授权问题是：

> 在固定 24F/24A 重建预算、相同训练预算、相同输入可见信息和独立观测 renderer 下，
> 一个跨样本训练、跨相机集合的有界残差算子，能否稳定胜过 CGLS、Huber-PDHG、
> 参数匹配 3D CNN、FNO 与 DeepONet，同时守住界面、重投影和尾部风险？

---

## 1. 可证伪假设与判决逻辑

M2 的总假设是五个假设的合取，不允许用一个指标补偿另一个失败。

| ID | 待证假设 | 直接证伪条件 |
|---|---|---|
| `H_FIELD` | M2 学到经典底座的可迁移结构化残差 | 对最强冻结基线的 field-L2 材料性改善不足，或 CI 不排除 0 |
| `H_STRUCTURE` | 收益同时保留梯度和真实界面 | H1、front/interface 任一硬门失败，或平滑场出现稳定假界面 |
| `H_SAFETY` | correction bound 与 fallback 控制伤害 | harm rate、p10、worst-case、coverage 或 reprojection 任一失败 |
| `H_GEOMETRY` | set/geometry 输入提供跨相机价值 | camera-count/pose OOD 反转，或 full 不优于 `no_geometry` |
| `H_DATA_DEPENDENCE` | 输出由观测残差而非 data-free prior 驱动 | 零/打乱 residual 后仍保留主要收益，或观测前已有高界面分数 |

只有 `H_FIELD && H_STRUCTURE && H_SAFETY && H_GEOMETRY && H_DATA_DEPENDENCE`
在 development、单开 OOD 和单开 fresh 依次成立，才可输出
`M2_INTERNAL_SYNTHETIC_FRESH_GO`。任何一步失败都停止后续开封。

---

## 2. 冻结候选 M2

### 2.1 输入、状态与输出

每个 case 的模型可见输入严格限于：

```text
y_v                  measured displacement for active camera v
geometry_v           calibrated pose, detector coordinates, aperture metadata
active_mask_v        active camera and valid-pixel masks
F_v, A_v             frozen inverse-side forward and exact discrete adjoint API
train-only statistics normalization constants frozen before OOD
```

模型 API 禁止出现 `truth`、`family`、`interface_mask`、生成器 seed、renderer 隐变量、
clean observation、评分用 holdout mask 或任何由 truth 计算的尺度。validator 必须通过
truth-key sentinel 和 import denylist 主动验证，而不是只检查函数签名。

主路径固定为：

```text
x0 = CGLS_18(y, geometry)
r_v = y_v - F_v(x0)
l_v = A_v(r_v)
delta_raw = R_theta({l_v, pose_v, active_mask_v}, x0)
delta = support * pointwise_clip(delta_raw) * global_norm_clip(delta_raw)
x_candidate = DC_5(x0 + delta)
x_hat = observable_fallback(x0, x_candidate, y, geometry)
```

其中：

- `R_theta` 对 camera 顺序必须置换不变；随机置换测试的最大相对输出差
  `<=1e-6`（float64）或 `<=1e-5`（float32）。
- correction 的全局约束为
  `||delta||2 <= rho * max(||x0||2, train_scale_floor)`；`rho` 只能从
  `{0.02, 0.05, 0.10}` 在 development 选择一次。
- pointwise bound 为 `|delta_i| <= 0.25 * train_dynamic_range`；该尺度只能由 train
  truth 统计，不能由待评分 truth 计算。
- support mask 必须由公开几何/物理域给出，不能由 truth interface 得到。
- `DC_5` 是所有 learned arms 共享的固定五步 data-consistency correction；步长、
  stopping 与 projection 规则在 development 后统一冻结，不能为 M2 单独调优。
- fallback 只读取残差范数、held-in discrepancy、correction norm、gate 与几何有效性；
  不得读取 field error、interface metric、clean renderer 或模型 seed 排名。

### 2.2 零初始化与可回退合同

M2 最后一层权重与 bias 必须精确为零，初始 hard gate 必须关闭。训练前对每个 preflight
case：

```text
max_abs(delta) == 0
x_hat == x0
gate_probability <= 0.001
active_interface_count == 0
```

float32 仅允许 `max_abs(x_hat - x0) <= 1e-7 * max(1, max_abs(x0))`。不满足时判
`M2_PREFLIGHT_DATA_FREE_INITIALIZATION_LEAK`，停止训练。

为保持准确的调用账本，完整候选路径仍执行到 24F/24A；fallback 最终可以返回 `x0`，但不能
把已执行的 residual lift、candidate 或 DC 调用从账本删除。另报告一个真实 early-fallback
延迟列，但它不进入主 accuracy 比较。

### 2.3 训练后的 data-free 负对照

每个冻结 checkpoint 必须在打分前运行以下负对照：

| 控制 | 输入修改 | 必须满足 |
|---|---|---|
| `zero_observation` | `y=0, x0=0, r=0`，保留真实 geometry | gate activation `<=1%`，无界面输出，correction norm ratio `<=1e-3` |
| `zero_residual` | 保留 `x0, geometry`，令所有 `r_v/l_v=0` | 不得保留 full M2 field gain 的 `>25%` |
| `shuffled_residual` | 在同 stratum、同 camera count 内打乱 residual owner | 相对 full 的 field-L2 至少恶化 `3%`，paired CI 排除 0 |
| `geometry_only` | `x0=0, r=0`，仅保留 pose/mask | 单/多界面 F1 不得高于随机平面 null 的 95% 分位；平滑假阳性 `<=1%` |
| `permuted_cameras` | 相机及其数据共同随机置换 20 次 | 输出符合置换不变容差 |

最终界面表必须同时给出 `data-free initialization`、`CGLS-18 base`、`final raw`、
`final fallback` 四个状态。只报最终绝对 F1 视为协议违规。

---

## 3. Split、密封和统计单位

### 3.1 固定规模

统计单位是 `parent_field_id`。同一 parent 的所有分辨率、相机排列、噪声重复、切片、时间邻帧
和增强版本必须留在同一 split；view、ray、pixel、noise replicate 和训练 seed 都不能伪装成
独立样本。

| Split | Parent fields | 三类场 | 每 parent 观测重复 | truth 可见性 | 允许用途 | 打开规则 |
|---|---:|---|---:|---|---|---|
| `train` | 240 | 每类 80 | 2 | 仅 trainer loss 可见 | 拟合参数与 train-only normalization | opened |
| `development` | 90 | 每类 30 | 2 | scorer 可见，模型不可见 | HPO、checkpoint、阈值、唯一候选冻结 | opened，可反复读取但全部决定入账 |
| `ood_validation` | 180 | 6 strata x 每类 10 | 1 | 独立 evaluator 可见 | 一次性证伪跨域性并决定是否解锁 fresh | 最多打开一次；调参后立即烧毁 |
| `fresh` | 240 | 8 strata x 每类 10 | 1 | 独立 evaluator 可见 | 唯一 confirmatory synthetic 判决 | 最多打开一次；无第二次机会 |

三类场固定为：

| `family` | Train/development 范围 | morphology OOD 范围 | 主要负对照 |
|---|---|---|---|
| `smooth_no_interface` | 低频 RBF、Gaussian plume、各向异性平滑场 | sheared plume、窄谱波包、边界邻近平滑结构 | 任何预测界面均为假阳性 |
| `single_interface` | SO(3) 随机方向平面、球面、椭球面；offset 与 jump sign 随机 | corrugated、强曲率、oblique/nested single front | 固定轴初始化与形状模板泄漏 |
| `multi_interface` | 2 个随机方向、分离或嵌套界面 | 3 个界面、近接、相交或不同 jump sign | missed/duplicate/false surfaces |

界面法向由均匀 SO(3) 采样产生。任何 split 的 `|n_x|`、`|n_y|`、`|n_z|` 分布不得显示
固定轴偏置；KS/energy test 的异常只作泄漏报警，不用来选择更好看的 manifest。

### 3.2 ID 与 OOD 条件

| Stratum | Camera count | Pose / detector shift | Noise | Bias | Observation renderer |
|---|---|---|---|---|---|
| `iid` | `{3,4}` | azimuth `<=1.5 deg`，elevation `<=1.0 deg`，shift `<=0.025` domain | iid Gaussian `0.5%-1.5%` | camera offset/gain `0%-2%` | `analytic_renderer_A` |
| `camera_count_ood` | `{2,5,6}` | ID range | ID range | ID range | A |
| `pose_ood` | `{3,4}` | azimuth `4-8 deg`，elevation `2-4 deg`，shift `0.05-0.10` | ID range | ID range | A |
| `noise_ood` | `{3,4}` | ID range | `2%-4%`，含 detector-correlated noise | ID range | A |
| `bias_ood` | `{3,4}` | ID range | ID range | offset/gain `4%-8%`，含 detector affine bias | A |
| `renderer_mismatch_ood` | `{3,4}` | ID range | ID range | ID range | 独立实现 `audit_renderer_B` |
| `morphology_ood` | `{3,4}` | ID range | ID range | ID range | A；使用上表未见形态 |
| `joint_shift` | `{2,5}` | pose OOD | `4%` correlated | `6%` offset/gain | B；同时使用 morphology OOD |

`ood_validation` 只含六个单轴 OOD strata，不含 `iid` 和 `joint_shift`；每个 stratum
30 个 parent。`fresh` 含上表八个 strata，每个 stratum 30 个 parent。

这些范围是协议值。若 fixture 无法构造某个条件，应在任何 OOD/fresh manifest 生成前把状态
保持为 `M2_PROTOCOL_NOT_READY`，不能静默缩窄考卷。

### 3.3 防止 split 泄漏

- parent latent、truth array、几何 seed、noise seed、bias seed 全部 split-disjoint。
- train/development 的任何 renderer cache、norm cache、normalization 或 interface extractor
  不得由 OOD/fresh 统计量更新。
- OOD/fresh 的 private manifest 由 evaluator 持有；model runner 只收到 observation、公开几何、
  mask 与受预算保护的 F/A API。
- 预开封只提交 public manifest hash；private seed 与 truth hash 在最终判决后才可发布。
- 必须做 exact hash、parent lineage 和近重复审计；发现跨 split 复制时，整个下游 split 作废，
  不能只删掉撞车样本后继续。

---

## 4. 防止 inverse crime

### 4.1 三条代码链

| 链 | 作用 | 必须独立的部分 | 允许共享 |
|---|---|---|---|
| `generator/renderer A` | train、development、IID 与多数单轴 OOD 观测 | 连续解析场梯度、ray integration、aperture aggregation | 坐标单位与公开 geometry contract |
| `inverse F/A` | 所有方法的重建 API | 体素有限差分、三线性插值、离散 transpose | 同一 measured ray geometry |
| `audit renderer B` | mismatch OOD 与 joint fresh | 独立模块/实现者、不同 quadrature 和 interpolation 路径，至少 4 倍积分采样或自适应 reference | 物理量定义与 detector contract |

禁止用 inverse-side `F` 生成任何 train/development/OOD/fresh 观测。禁止 renderer A/B import
inverse operator 的 matrix、row builder、interpolation kernel 或 adjoint。允许共享的模块必须写入
`shared_code_allowlist`；allowlist 外任何共同 import 都使 preflight fail closed。

### 4.2 必须通过的数值检查

1. inverse F/A 的 20 对 float64 dot-product defect `<=1e-10`，float32 `<=1e-5`。
2. directional derivative 最佳相对误差 `<1e-5`，并显示合理的 step-size 收敛区。
3. 分相机 masked adjoint 的 view-equivalent 求和必须等于 full-stack adjoint，relative error
   `<=1e-10`（float64）。
4. renderer A 对 analytic reference 做分辨率/step convergence；renderer B 对更高精度 reference
   独立收敛。
5. A 与 inverse F 在非零 case 上必须存在可测 mismatch；若二者逐 row/逐值相同或共享离散链，
   判 `M2_PREFLIGHT_INVERSE_CRIME`。
6. scorer 的 truth arrays 与 clean holdout observations 在模型进程中不可导入、不可 mmap、不可由
   文件路径猜测访问。

---

## 5. 强基线、调用预算与训练预算

### 5.1 必须存在的主表方法

| ID | 方法 | 24F/24A 合同 | 输入信息 | 主要公平性要求 |
|---|---|---|---|---|
| `cgls_24` | CGLS | 24 次迭代 | y、geometry、F/A | 固定初始化与停止；不按 case early select |
| `huber_pdhg_24` | Huber-PDHG | 24 对 primal/dual 物理调用 | 同上 | lambda/delta/step 仅在 development 固定网格选择 |
| `cnn3d_residual` | 参数匹配 3D CNN | CGLS-18 + 1 residual/lift pair + DC-5 | 与 M2 相同的 padded per-view lifts、pose、mask、x0 | 训练预算、bound、fallback 与 M2 相同 |
| `fno_residual` | 参数匹配 FNO | 同上 | 同上 | modes/width 进入冻结 HPO 网格 |
| `deeponet_residual` | 参数匹配 DeepONet | 同上 | padded branch set + mask；trunk 为 voxel xyz | 不得读取固定传感器之外的 truth 坐标信息 |
| `m2_set_residual` | full M2 | 同上 | per-view shared encoder + permutation-invariant aggregation | 本协议候选 |

可以增加 generic Learned Primal-Dual 作为次要强基线，但不能用它替代上述五个基线。NeRIF、
TDBOST 或论文中的异任务数字不得拼入本主表。

所有 learned arms 必须看到同一信息集合。CNN/FNO 可用固定 `Vmax=6` 的 padded channels 与
active mask；DeepONet 使用同一 padded set；M2 的区别只能是集合聚合和 residual update 结构，
不能是多看了 pose、clean data 或额外 adjoint。

### 5.2 F/A 记账定义

- 一次 full active-camera forward 记 `1 F`；一次 full-stack adjoint 记 `1 A`。
- 分相机 masked adjoint 同时报告 raw invocation count 与
  `adjoint_view_equivalents = sum(active_view_fraction)`；处理全部 active views 合计记 `1 A`。
- 每个 learned arm 固定为 `18 + 1 + 5 = 24 F` 与 `24 A`。
- 几何相关 norm estimation、preconditioner、warm start、fallback probe、line search、restarts 和
  method-specific preprocessing 只要调用 F/A，就必须记入本方法预算。
- 真正与方法无关的 geometry cache 可以共享，但需单列 `shared_setup_*` 和摊销时延；fresh 新
  geometry 的 cache 建造不能伪装成 train-time 免费成本。
- scorer 为每个最终输出调用的 clean/heldout forward 记入
  `evaluation_forward_calls`，不得反馈给方法，也不计入 reconstruction 24F。
- 超过任一预算立即 fail；少用调用可以保留，但不能把未执行调用写成已执行，且必须报告实际值。

### 5.3 参数与训练预算

所有 learned arms 使用以下相同预算：

| 项 | 冻结值 |
|---|---|
| trainable parameter target | `1,000,000`，允许 `+/-10%` |
| HPO trials | 每个 arm 最多 12 个；每 trial 1 个固定 HPO seed |
| HPO steps | 每 trial 10,000 optimizer steps |
| final model seeds | 5 个固定 seed：`1101, 2203, 3307, 4409, 5519` |
| final steps | 每 seed 30,000 optimizer steps；禁止 early stop |
| batch / presentations | batch 4；每 seed 120,000 train sample presentations |
| optimizer family | AdamW；允许的 LR/weight-decay 组合必须预先列入 12-trial 网格 |
| checkpoint candidates | 仅 `10k, 20k, 30k`；选择规则见下文 |
| precision | train float32；所有最终指标与 F/A 数值审计 float64 |
| data order | 由 model seed 决定的固定 sampler；各 arm 使用同一 case 顺序 |
| accelerator cap | HPO 每 trial 最多 4 device-hours；final 每 seed 最多 12 device-hours |

若某 arm 达不到参数容差，必须同时报告一个 natural-size arm 与一个 matched-size arm；M2 必须
面对两者中更强者。若硬件无法在 cap 内完成固定 steps，使用最后一个预注册 checkpoint 并记
`TRAIN_BUDGET_EXHAUSTED`，不能临时给该方法延长时间。

训练 cache 只构造一次并由所有 learned arms 共享。必须记录 cache 生成的 train F/A、字节数、
SHA-256 与 wall time；禁止把 candidate-only 离线求解藏在输入 preprocessing 中。

### 5.4 Development 选择规则

先应用安全可行性门：reprojection、harm、smooth false positive、budget、NaN 任一失败的 config
直接淘汰。剩余 config 按以下固定标量最小化：

```text
selection_score =
    0.50 * mean_family_rank(field_relative_l2)
  + 0.30 * mean_family_rank(h1_seminorm_relative_error)
  + 0.20 * mean_interface_rank(front_loss)
```

rank 在同一 development arm/config 集内计算；平滑场的 `front_loss` 使用 false-positive penalty。
差异小于 `0.001` 时依次选择参数更少、p95 latency 更低、config ID 字典序更小者。不得人工查看
切片后改 checkpoint。每个方法只能冻结一个 config；fresh 禁止按 stratum、family 或 case 选择
不同 seed/config，禁止 best-seed 与 oracle ensemble。

---

## 6. 必做消融与负对照

| ID | 改动 | 是否重训 | 要证伪的解释 |
|---|---|---:|---|
| `m2_full` | 完整候选 | 是，5 seeds | 主假设 |
| `m2_no_zero_init` | 最后一层用标准随机初始化，其他完全相同 | 是，5 seeds | 零初始化是否只是在事后叙事 |
| `m2_no_fallback` | 同一 full checkpoint，强制总是输出 candidate | 否 | safety 是否来自 fallback |
| `m2_no_geometry` | pose、camera ID/count descriptor 置零，仅保留 mask 与 lifts | 是，5 seeds | 跨几何收益是否来自 geometry/set 信息 |
| `m2_zero_residual` | 冻结 full checkpoint，residual/lift 置零 | 否 | 是否依赖观测残差 |
| `m2_shuffled_residual` | 冻结 full checkpoint，stratum 内打乱 residual owner | 否 | 是否存在 data-free shape prior |
| `m2_no_dc` | 冻结 full checkpoint，跳过 DC-5；预算仍如实报告 | 否 | 收益来自 learned correction 还是固定 DC |

`no_geometry` 通过输入置零而不是删层，保持参数量与主候选一致。`no_fallback`、
`zero_residual`、`shuffled_residual` 和 `no_dc` 使用同一 checkpoint，禁止另行调参。

零初始化消融的主要对象是训练安全而非制造一个必须失败的对手。必须额外报告 5 个 seed 在
`10k` 首个 checkpoint 的 divergence/NaN、harm rate、correction norm 和 gate activation。
只有 full 相对 `no_zero_init` 的首 checkpoint harm 或 divergence 至少下降 `30%`，且最终指标
不劣，才允许声称“零初始化提高训练安全”；否则只能写“零初始化保证训练前严格回到底座”，
不能把它叙述成已验证的性能来源。

所有消融在 OOD/fresh 打分前与主方法一起生成预测并提交 hash。不能先看 full fresh 结果再决定
是否运行某个消融。

---

## 7. 指标、统计与错误处理

### 7.1 冻结指标

所有场指标只在同一物理 support 内计算，voxel spacing 必须进入梯度和表面距离。

| 维度 | 主指标 | 辅助指标 | 反作弊要求 |
|---|---|---|---|
| field | `field_relative_l2 = ||xhat-x||2 / max(||x||2, eps_train)` | RMSE、dynamic-range NRMSE、mean bias | `eps_train` 仅由 train 冻结 |
| H1 | relative H1 seminorm error | gradient closure RMS | 同一差分 stencil、同一边界处理 |
| front/interface | F1@1dx、ASSD | F1@2dx、HD95、missed/false count | 所有方法使用同一 train-frozen field-to-interface extractor |
| smooth control | false-positive rate、active interface count | predicted interface area | empty truth 不得被记为 perfect F1 而忽略假阳性 |
| reprojection | hidden-ray clean relative-L2 | measured-ray relative-L2、full clean relative-L2 | 每相机 20% hash-selected rays 不进入 reconstruction API |
| safety | harm rate、p10 paired gain、worst paired gain | fallback coverage、correction norm | 以 parent field 聚合，不按 ray 统计 |
| cost | total p50/p95 latency、peak memory | train time、setup、F/A、FLOPs、serialized bytes | 同硬件、precision、batch 与同步方式 |

主 front 指标从每个方法的最终 field 通过同一个冻结 extractor 得到。M2 自己的显式 interface head
只能作为 secondary 诊断，不能与基线的 field-derived surface 做不对称比较。多界面使用 Hungarian
一对一匹配；missed 与 false surface 分别以 `F1=0` 和 domain diagonal distance 惩罚，不允许删除
难匹配表面。

hidden-ray mask 在 private manifest 中生成，每台相机固定留出 20% detector pixels；这些 rays、
clean values 和 mask 在 prediction commit 前对模型不可见。

### 7.2 Paired gain 与 harm

对 error metric `e`，相对冻结比较基线 `b` 定义：

```text
gain(i, seed) = (e_b(i) - e_m2(i, seed)) / max(e_b(i), eps_train)
harm_1pct     = gain < -0.01
severe_harm   = gain < -0.05
```

主分析先对 5 个 model seeds 在每个 parent 内求均值，再以 parent 为 cluster 做 10,000 次
stratified paired bootstrap；bootstrap seed 固定为 `20260717`。同时逐 seed 报告，不把 model seed
当作扩大样本量的独立 field。

每个指标分别与五个强基线配对。材料性门面对该 split 上聚合最强的完整 baseline arm，并对五个
pairwise superiority tests 使用 Holm family-wise `alpha=0.05`。禁止按 case 组成 oracle baseline
envelope 作为主比较；可以把它作为更保守的 secondary 表完整披露。

### 7.3 失败、NaN 与缺失 case

- 方法级 NaN、OOM、超预算、未输出或 checksum drift 记为该 case 最差值和 harm，不能静默丢行。
- 数据级损坏只有在解封 truth 前、且对所有方法同样成立时才可由 validator 排除；fresh 不补抽
  替代 case。
- 任一 split 的数据级无效 parent 超过 2%，整个 split 判 `INVALID_DATA_MANIFEST`。
- metric rows 数量、方法集合、seed 集合、case 集合必须与 manifest 笛卡尔积精确一致。

---

## 8. 顺序硬门

### G0: Preflight，失败即不训练

以下全部为真才可训练：

| Check | 门槛 |
|---|---|
| M0-M1 evidence chain | 现有 validator 仍为 `VALIDATED_NEGATIVE_EVIDENCE_PACKET` |
| split lineage | parent、seed、truth、observation、geometry 无跨 split 重叠 |
| inverse F/A | dot/JVP/VJP 达到第 4 节阈值 |
| inverse-crime audit | renderer A/B 与 inverse F/A 的独立性通过 |
| truth sentinel | 模型进程无法访问 truth/family/interface/generator keys |
| zero initialization | `x_hat == x0`，gate closed，无观测前界面 |
| camera permutation | 达到置换不变容差 |
| budget counter | 第 25 次 F 或 A 立即拒绝；masked view-equivalent 正确 |
| baseline availability | 六个主表 arms 均可运行；不能因候选先完成而删除强基线 |
| scorer lock | metric code、threshold、bootstrap、missing-row policy 均已有 SHA-256 |

任一失败输出 `M2_PREFLIGHT_INVALID`，不得生成方法排名。

### G1: Development 门，失败则 OOD 保持密封

`m2_full` 必须同时满足：

| 维度 | 最低门槛 |
|---|---|
| field | mean gain `>=5%`，Holm-adjusted paired 95% CI lower bound `>0` |
| H1 | mean gain `>=3%`，adjusted CI lower bound `>0` |
| single/multi front | F1@1dx absolute gain `>=0.03` 且 ASSD reduction `>=10%`；HD95 ratio `<=1.05` |
| smooth | field harm `<=1%`；interface false-positive rate `<=1%` 且不高于最强基线 `+1 pp` |
| reprojection | measured ratio vs CGLS-24 `<=1.10`；hidden clean ratio vs best baseline `<=1.05` |
| tail | `harm_1pct <=5%`，p10 gain `>=0`，worst gain `>=-5%`，severe harm count `0` |
| seeds | 至少 4/5 seeds 的 mean field gain `>0`；任何 seed 不得违反 worst/harm 门 |
| fallback | candidate coverage overall `>=60%`，每 family `>=50%`；不可全回退过门 |
| residual dependence | full 优于 zero/shuffled residual `>=3%` 且 CI 排除 0；控制不得保留 full gain 的 `>25%` |
| ledger | 参数、训练、F/A、memory、p50/p95 latency 无缺失 |

若只赢 CGLS、不赢 Huber-PDHG 或任一 learned baseline，仍为
`M2_DEVELOPMENT_NO_GO`。所有 development 失败都允许继续做 opened diagnosis，但原 OOD manifest
不得打开；任何改动必须产生新的 candidate/config hash。

### G2: 单次 OOD 门，失败则 Fresh 保持密封

OOD 只打开一次，并要求：

| 维度 | 最低门槛 |
|---|---|
| pooled OOD field | mean gain `>=2.5%`，即至少保留 development 5% 门的一半；CI lower `>0` |
| per-stratum field | 六个 OOD stratum 的 mean gain 均 `>0`，不得系统性反转 |
| H1/front | pooled H1 gain `>0`；F1/ASSD 不劣于最强基线；smooth FP 继续达门 |
| camera/pose mechanism | full 相对 `no_geometry` 的 pooled field gain `>=3%`，CI lower `>0` |
| mismatch | `renderer_mismatch_ood` mean field gain `>0`，reprojection 与 tail 门不失败 |
| safety | 与 G1 相同的 harm、worst、reprojection、seed 和 coverage 门 |
| fallback effect | 相对 `no_fallback`，harm rate 至少下降 `30%`，且 candidate coverage 达门 |

任一失败输出 `M2_OOD_NO_GO`。若开发者看过 OOD aggregate/rows 后修改模型、数据、阈值、
normalization、fallback 或 checkpoint，该 OOD split 永久降级为 opened development；必须另建全新
OOD manifest 才能测试新版本，原 fresh 仍不得打开。

### G3: 单次 Fresh 总门

只有 G0-G2 全通过且 `selection_commit.json` 已冻结，fresh evaluator 才接受一次事务。
Fresh 必须同时满足：

| 维度 | 最低门槛 |
|---|---|
| pooled field | 对聚合最强 baseline 的 mean gain `>=5%`；Holm-adjusted 95% CI lower `>0` |
| pooled H1 | mean gain `>=3%`；adjusted CI lower `>0` |
| three families | 每类 mean field gain `>0`；single/multi front 达 G1 材料性门；smooth 达无害门 |
| each OOD stratum | mean field gain `>0`；pooled OOD gain 至少为 fresh IID gain 的 `50%` |
| joint shift | field、H1 不反转；harm/reprojection 门通过 |
| reprojection | measured ratio `<=1.10`，hidden clean ratio `<=1.05` |
| tail | `harm_1pct <=5%`，p10 `>=0`，worst `>=-5%`，severe harm count `0` |
| seeds | 4/5 mean field gain `>0`，无 seed 违反 safety 门；禁止只报最好 seed |
| fallback | overall coverage `>=60%`，每 stratum `>=50%`，并达到 no-fallback harm reduction |
| data dependence | zero/shuffled residual 控制达到 G1 门；geometry-only 不产生 data-free surface |
| geometry mechanism | count+pose OOD 上 full vs no-geometry gain `>=3%`，CI lower `>0` |

Runtime 不得藏在 accuracy 判决中：

- 若 accuracy 全通过，但 total reconstruction p50 超过最强非劣 learned baseline `1.25x`，或 p95
  超过 `1.50x`，输出 `M2_FRESH_ACCURACY_SIGNAL_RUNTIME_NO_GO`，禁止效率主张。
- 只有 accuracy、safety、OOD、mechanism 与 runtime 全通过，才输出
  `M2_INTERNAL_SYNTHETIC_FRESH_GO`。

没有“6/8 strata 通过也算趋势成功”的规则。任一硬门失败即为相应 NO-GO，并完整报告通过项与
失败项。

---

## 9. 停止条件与修订政策

| 触发事件 | 必须动作 | 禁止动作 |
|---|---|---|
| 数学/F/A/preflight 失败 | `M2_PREFLIGHT_INVALID`，停止训练或排名 | 让网络吸收 adjoint bug |
| Development 材料性门失败 | `M2_DEVELOPMENT_NO_GO`，保留 opened diagnosis | 因“趋势好”打开 OOD/fresh |
| OOD 任一 stratum 反转 | `M2_OOD_NO_GO`，fresh 继续密封 | 删除失败 stratum 或改阈值后沿用同一 OOD |
| Fresh 任一硬门失败 | 写最终 NO-GO，停止该 protocol/candidate | 重跑 seed、挑 checkpoint、再开 fresh |
| Fresh 后发现算法/data bug | 本 fresh manifest 烧毁，结果降级为 invalid diagnostic | 修 bug 后把同一 fresh 再称 confirmatory |
| Fresh 后发现纯 scorer bug | 只能对同一 prediction hash 重算；公开 amendment | 重新运行模型或改变预测 |
| 超预算/OOM/NaN | 按最差 case 计入并触发相应门 | 删除失败行或用 baseline 值填充 |
| 只通过 field，H1/front 失败 | `M2_STRUCTURE_NO_GO` | 声称 interface-aware 优势 |
| 只通过 IID，OOD 失败 | `M2_OOD_NO_GO` | 声称 operator/generalization |
| 只通过 accuracy，runtime 失败 | `M2_FRESH_ACCURACY_SIGNAL_RUNTIME_NO_GO` | 声称更快、更省算力 |

任何算法、loss、输入、数据范围、budget、seed、baseline、metric、阈值、fallback、checkpoint 或
selection rule 的变化都必须升级为 `M2.x`，重新走 development，并使用全新 OOD/fresh manifests。
只允许在开封前做 protocol amendment；amendment 必须列出旧/新 hash 和理由。

凡是在看到 OOD 或 fresh 的 aggregate、逐行指标、图、truth slice 或排名后做出的上述变化，均为
**后验调参**。后验调参可以形成新的 opened-data diagnosis，但不能继续使用原预注册标签、原
confirmatory fresh 预算或原主张谱系的“首次验证”措辞。

---

## 10. 防止 Fresh 反复打开的原子事务

### 10.1 开封前冻结包

同一 Git commit 必须包含或锚定：

1. 本协议与 machine config 的 SHA-256；
2. train/development public manifests 与 OOD/fresh encrypted private manifest hashes；
3. 六个主方法、七个消融、F/A、renderer A/B、scorer 和 validator 的 code hashes；
4. 所有 frozen checkpoints、normalization、interface extractor 与 fallback threshold hashes；
5. container/lockfile、hardware ID、precision、thread count 和 environment hash；
6. `selection_commit.json`，声明唯一 config、5 seeds、baseline set 与全部门槛；
7. 空的 append-only `fresh_open_registry.jsonl`，其前一行 hash 写入 selection commit。

### 10.2 Prediction-first、label-second

Fresh 事务固定为：

```text
1. evaluator verifies protocol/config/code/checkpoint/manifest hashes
2. runner receives y, public geometry, masks and budgeted F/A only
3. every method, seed and ablation writes prediction bundle
4. evaluator verifies complete Cartesian product and budget ledger
5. prediction bundle is made read-only and SHA-256 committed
6. only now scorer unlocks truth and hidden rays
7. scorer emits metric_rows, aggregate report and their hashes atomically
8. registry appends open_count=1 and terminal status
```

`fresh_open_count_max=1`。模型不得在第 6 步后再次执行；图、切片与网页只能由已冻结 metric rows
和 prediction bundle 生成。

更严格地，冻结包必须声明唯一
`claim_lineage_id = jacru-m2-residual-operator-2026-07-17`，且该主张谱系的
`confirmatory_fresh_budget=1`。这个预算跨 `M2`、`M2.1`、重命名分支、目录复制和新 manifest
累计，而不是每个文件各有一次。第一次产生任何 fresh label-derived 输出后，谱系预算永久耗尽；
后续新数据只能称 post-fresh follow-up 或独立 replication，不能再称本假设的首个 confirmatory
fresh。全局 registry 必须按 `claim_lineage_id` 拒绝第二次事务。

### 10.3 崩溃与重试

- 在任何 label-derived byte 输出前崩溃：只允许用完全相同的 prediction bundle hash 恢复 scorer；
  不允许重跑 stochastic model。
- 在任何 metric、排名、truth slice、日志摘要输出后崩溃：视为 fresh 已打开；只可从同一 prediction
  hash 原子重建同一结果。
- prediction 不完整、hash 不符或 runner 需要重跑：该 fresh manifest 烧毁。若尚未产生任何
  label-derived byte，可在同一 claim-lineage 以新协议重新准备；若已经产生，则 confirmatory
  谱系预算已经耗尽，后续即使用新 candidate 与新 manifest 也只能是 post-fresh follow-up。
- 不存在“换 5 个 seed 再试一次”“只重跑最差 stratum”或“把 fresh 改名 final”的通道。

独立 evaluator 最好由不参与 M2 调参的人或隔离进程持有。若无人独立托管，最低要求是只读私有
manifest、网络关闭的 scorer 进程、一次性 token 与 append-only registry；否则证据等级自动降为
`SELF_SCORED_OPENED_DIAGNOSTIC`。

---

## 11. 建议的机器可读产物

本轮只写协议，不创建下列文件。开始实现前建议冻结这些路径和 schema：

| Artifact | 最少内容 |
|---|---|
| `demo_t16_operator/configs/jacru_m2_preregistered_v1.json` | 协议状态、split、方法、预算、seed、门槛、禁止主张 |
| `.../manifests/{train,development}_public.csv` | public case/parent/geometry/observation digests |
| `.../manifests/{ood,fresh}_private.json.enc` | private seeds、truth、hidden-ray mask；仅 evaluator 可读 |
| `.../split_audit.json` | lineage、hash overlap、near-duplicate、orientation audit |
| `.../initialization_audit.json` | pretrain identity 与五个 data-free controls |
| `.../selection_commit.json` | 唯一 candidate/checkpoint/baseline/scorer/config hashes |
| `.../prediction_hashes_before_scoring.json` | 每 method/seed/case 的 prediction hash 与提交时间 |
| `.../metric_rows.csv` | 每 parent/method/seed 的全部 accuracy、safety、cost 字段 |
| `.../budget_ledger.csv` | train/setup/reconstruction/evaluation F/A、时间、内存和参数账本 |
| `.../fresh_open_registry.jsonl` | append-only 一开事务与链式 previous-record hash |
| `.../report.json` | gate-by-gate verdict、失败原因、claims unlocked/forbidden |

### 11.1 Preregistration JSON 核心字段

```json
{
  "schema_version": "jacru-m2-preregistered-gate-1.0",
  "status": "DRAFT_PROTOCOL_ONLY",
  "protocol_date": "2026-07-17",
  "claim_lineage_id": "jacru-m2-residual-operator-2026-07-17",
  "confirmatory_fresh_budget": 1,
  "prior_fresh_label_open_count": 0,
  "parent_protocol_sha256": "<required>",
  "source_negative_evidence_sha256": {
    "m0_summary": "<required>",
    "m0_1_summary": "<required>",
    "m1_summary": "<required>",
    "initialization_audit": "<required>",
    "evidence_validator": "<required>"
  },
  "splits": {
    "train": {"parent_count": 240, "constructed": false, "required_sealed": false},
    "development": {"parent_count": 90, "constructed": false, "required_sealed": false},
    "ood_validation": {
      "parent_count": 180,
      "constructed": false,
      "required_sealed_before_use": true,
      "open_count_max": 1
    },
    "fresh": {
      "parent_count": 240,
      "constructed": false,
      "required_sealed_before_use": true,
      "open_count_max": 1
    }
  },
  "families": ["smooth_no_interface", "single_interface", "multi_interface"],
  "fresh_strata": [
    "iid", "camera_count_ood", "pose_ood", "noise_ood", "bias_ood",
    "renderer_mismatch_ood", "morphology_ood", "joint_shift"
  ],
  "methods_required": [
    "cgls_24", "huber_pdhg_24", "cnn3d_residual", "fno_residual",
    "deeponet_residual", "m2_set_residual"
  ],
  "model_seeds": [1101, 2203, 3307, 4409, 5519],
  "budget": {
    "reconstruction_forward_calls_max": 24,
    "reconstruction_adjoint_calls_max": 24,
    "learned_base_cgls_pairs": 18,
    "residual_lift_pairs": 1,
    "data_consistency_pairs": 5,
    "trainable_parameters_target": 1000000,
    "trainable_parameters_tolerance_fraction": 0.10,
    "hpo_trials_max": 12,
    "hpo_steps": 10000,
    "final_train_steps": 30000,
    "batch_size": 4
  },
  "gates": {
    "field_gain_min_fraction": 0.05,
    "h1_gain_min_fraction": 0.03,
    "front_f1_gain_min_absolute": 0.03,
    "front_assd_gain_min_fraction": 0.10,
    "hidden_reprojection_ratio_max": 1.05,
    "measured_reprojection_ratio_max": 1.10,
    "harm_rate_max": 0.05,
    "p10_gain_min": 0.0,
    "worst_gain_min": -0.05,
    "candidate_coverage_min": 0.60,
    "per_stratum_coverage_min": 0.50,
    "ood_gain_retention_min_fraction": 0.50,
    "geometry_ablation_gain_min_fraction": 0.03,
    "fallback_harm_reduction_min_fraction": 0.30
  },
  "statistics": {
    "unit": "parent_field_id",
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260717,
    "familywise_alpha": 0.05,
    "multiplicity": "holm"
  },
  "fresh_policy": {
    "prediction_hash_before_labels": true,
    "open_count_max": 1,
    "claim_lineage_open_count_max": 1,
    "rerun_model_after_scoring": false,
    "same_manifest_after_algorithm_fix": false,
    "posthoc_tuning_can_reuse_confirmatory_label": false
  },
  "claim_boundary": {
    "experimental_reconstruction": false,
    "cfd_validation": false,
    "physical_shock_identification": false,
    "oerf_transfer": false,
    "four_dimensional_method": false,
    "general_neural_operator_superiority": false
  }
}
```

### 11.2 Manifest、metric 与账本字段

`case_manifest.csv` 至少包含：

```text
case_id,parent_field_id,split,family,stratum,data_seed,geometry_seed,noise_seed,
bias_seed,renderer_id,renderer_code_sha256,camera_count,pose_bin,noise_level,
bias_level,truth_sha256,observation_sha256,geometry_sha256,hidden_mask_sha256,
parent_lineage_sha256,sealed
```

private 字段不得复制到 model runner 的 public manifest。

`metric_rows.csv` 至少包含：

```text
protocol_sha256,run_id,case_id,parent_field_id,split,family,stratum,method,
model_seed,config_sha256,checkpoint_sha256,prediction_sha256,field_relative_l2,
h1_seminorm_relative_error,front_f1_at_1dx,front_f1_at_2dx,front_assd,
front_hd95,false_positive_count,missed_truth_count,active_interface_count,
measured_reprojection_relative_l2,hidden_clean_reprojection_relative_l2,
field_gain_vs_champion,harm_1pct,severe_harm,fallback_used,candidate_coverage,
correction_norm_ratio,optimization_forward_calls,
optimization_adjoint_view_equivalents,evaluation_forward_calls,wall_seconds,
peak_memory_bytes,status
```

`budget_ledger.csv` 至少包含：

```text
protocol_sha256,phase,method,model_seed,case_id,hardware_id,precision,batch_size,
trainable_parameters,total_parameters,optimizer_steps,sample_presentations,
shared_setup_forward_calls,shared_setup_adjoint_calls,method_setup_forward_calls,
method_setup_adjoint_calls,reconstruction_forward_calls,
reconstruction_adjoint_raw_calls,reconstruction_adjoint_view_equivalents,
evaluation_forward_calls,train_wall_seconds,setup_wall_seconds,
preprocess_wall_seconds,network_wall_seconds,dc_wall_seconds,total_wall_seconds,
latency_p50_seconds,latency_p95_seconds,peak_memory_bytes,serialized_bytes
```

`fresh_open_registry.jsonl` 每行必须包含：

```text
run_id,claim_lineage_id,claim_lineage_prior_open_count,previous_record_sha256,
protocol_sha256,git_commit,config_sha256,
private_manifest_sha256,selection_commit_sha256,prediction_bundle_sha256,
prediction_committed_at,labels_opened_at,open_count,scorer_sha256,
metric_rows_sha256,report_sha256,terminal_status
```

validator 必须拒绝未知列被悄悄用于选择，同时允许报告 schema 版本显式升级。

---

## 12. 判决状态机

允许的 terminal/intermediate 状态只有：

```text
M2_DRAFT_PROTOCOL_ONLY
M2_PREFLIGHT_INVALID
M2_DEVELOPMENT_NO_GO
M2_READY_FOR_OOD_SINGLE_OPEN
M2_OOD_NO_GO
M2_READY_FOR_FRESH_SINGLE_OPEN
M2_FRESH_NO_GO_ACCURACY
M2_FRESH_NO_GO_STRUCTURE
M2_FRESH_NO_GO_SAFETY
M2_FRESH_NO_GO_OOD
M2_FRESH_ACCURACY_SIGNAL_RUNTIME_NO_GO
M2_INTERNAL_SYNTHETIC_FRESH_GO
M2_INVALID_FRESH_REOPEN
M2_INVALID_HASH_OR_MANIFEST_DRIFT
```

`report.json` 必须逐 gate 给出 `passed`、observed value、threshold、CI、comparison method、
row count 和 failure reason。`overall_passed` 只能由 validator 按合取逻辑生成，runner 不得自行
写入成功状态。

---

## 13. 禁止主张

在任何结果前，全部禁止。即使 `M2_INTERNAL_SYNTHETIC_FRESH_GO`，仍禁止：

1. “M2/JACRU 已在真实 OERF、实验 BOST、CFD 或飞行器数据上验证。”
2. “M2 恢复了真实 shock、压力、温度、组分或 Rankine-Hugoniot 状态。”
3. “M2 普遍优于 DeepONet、FNO、3D CNN、NeRIF 或 TDBOST。”允许的比较仅限本协议同任务实现。
4. “M2 是首个 interface-aware、level-set、phase-field、unrolled 或 finite-aperture 方法。”
5. “低 measured reprojection 等于三维场正确。”
6. “autograd 或 dot-product test 证明了物理 exact adjoint。”它只验证冻结离散实现。
7. “通过固定 synthetic renderer 就证明跨 forward 泛化。”mismatch OOD 失败时尤其禁止。
8. “相机数可任意泛化。”只允许写本协议 `{2,3,4,5,6}` 与冻结 pose 范围。
9. “fallback 提供校准不确定度或安全保证。”它只是本考卷上的可观测回退策略。
10. “多 seed spread 是概率置信区间。”bootstrap 单位是 parent field，不是 seed。
11. “M2 是 4D 方法。”本协议是静态 3D。
12. “一次 fresh 失败后修复并再跑仍是同一预注册确认。”第二次只能是新协议的新考卷。

若只通过 development，可写“opened development mechanism signal”；若通过 OOD 但未开 fresh，
可写“single-open synthetic OOD validation signal”；只有 fresh 全门通过，才能写：

> Under the preregistered internal synthetic fixture, independent observation/inverse
> discretizations, matched 24F/24A reconstruction budgets, matched training budgets, and
> the declared camera/noise/bias/mismatch ranges, M2 passed the one-open fresh gate.

该句也必须同时给出失败门、范围、成本和证据边界，不能缩写成“JACRU 已成功”。

---

## 14. 冻结前清单

- [ ] 本协议没有未填的关键阈值、方法、seed 或样本数量。
- [ ] M0/M0.1/M1 summary、metric rows、初始化审计与 validator hash 已锚定。
- [ ] train/development/OOD/fresh parent lineage 与三类场平衡已验证。
- [ ] renderer A、inverse F/A、renderer B 的代码独立性与 convergence 已验证。
- [ ] 六个主方法和七个消融都能产生完整 prediction bundle。
- [ ] 24F/24A counter 对 norm、warm start、lift、fallback 与 preprocessing 无漏记。
- [ ] 五个 model seeds、HPO 网格、checkpoint 规则和参数容差已冻结。
- [ ] data-free、zero/shuffled residual、camera permutation 与 truth sentinel tests 全通过。
- [ ] field、H1、front、reprojection、harm、worst-case 与 cost scorer 已冻结并哈希。
- [ ] OOD/fresh private manifests 对 model runner 不可达。
- [ ] selection commit、prediction-first scorer 与 append-only open registry 已实现。
- [ ] validator 已证明 fresh 第 2 次开封和第 25 次 F/A 调用都会 fail closed。
- [ ] 当前 Git commit、环境、hardware ID、precision 与全部 checkpoint hash 已写入冻结包。

清单未全部满足时，状态保持 `M2_DRAFT_PROTOCOL_ONLY`。最重要的失败结果也必须能够完整落盘：
若 generic CNN/FNO/DeepONet、CGLS 或 Huber-PDHG 在公平预算下等效或更强，M2 就被证伪；这比
在已打开 fresh 上继续调到赢为止更有研究价值。
