# LGWO-A24-L1：三种子、双射线束可学习方向密封试验

**状态：** `PASS_IMPLEMENTATION_GATES_FIT_RUNNER_IMPLEMENTATION_AUTHORIZED_ROUTE_SEALED`

**机器可读协议：** `demo_t16_operator/configs/lgwo_a24_l1_sealed_observable_pilot_v1.json`

**训练前修订 1.1：** 首次只在永久排除工程种子上运行 implementation gate 时发现，原协议只固定了 correction head 的随机种子，没有明确固定其余卷积层的 constructor RNG。此时 fit、early-stop、route-development、fresh-OOD 均为 0 个已生成 case，因此在任何科学 partition 打开前加入完整模型初始化合同：在 `torch.random.fork_rng(devices=[])` 内按当前 model seed 调用 `torch.manual_seed`，CPU 构造完整 `LDFieldFiLM8`，再转 `float64`，且不得推进进程全局 RNG。JSON 保存被取代的 `d6c7f771...` 哈希与修订原因；这属于可重复性修复，不改变 claim 或结果门。

**用途：** 只回答一个可证伪问题：在保持 LGWO-A24 部署路径严格为 `24F/24A^T`、`delta=0` 精确恢复强基线的前提下，一个 2,729 参数模型能否仅从可部署输入学习第一方向修正，并在未参与训练的几何簇、三种反应场形态和独立射线束上稳定保留收益。

**当前不能声称：** 模型尚未训练；route-development 仍密封；没有 fresh、真实 BOST、独立物理 renderer、DeepONet/FNO 对比、泛化、突破或新颖性结论。本文的 `heldout-B` 只是同一解析 renderer 家族下的另一组射线，不是第二套物理实验。

## 1. 已观察信号的证据边界

`LGWO-A24-O1` 在已经打开的 6 个 JACRU development synthetic cases、3 个 geometry clusters 上，用 CPU float64 dense SVD 和 truth 构造了 evaluator-only 第一方向。其 `exact_null_truth` arm 在相对半径 `0.05` 和 `0.10` 下都显示 field/H1 headroom，同时 measured/clean residual 基本不变：

| radius | mean field gain | mean H1 gain | worst field gain | mean measured ratio | mean clean ratio |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 6.98% | 6.57% | 3.74% | 1.0000 | 1.0000 |
| 0.10 | 13.72% | 13.00% | 7.26% | 1.0000 | 1.0000 |

这只能说明在该 opened toy operator 的数值 kernel 中存在一个 truth-informed、有限半径的表示方向。它不说明允许输入可预测该方向，也不说明 SGD 能学到它，更不说明独立 renderer、真实光学系统或未见 geometry 中存在同样收益。

除 O1 的 6 个 cases 外，既有 JACRU 学习/诊断流程已经打开更多几何种子。L1 按最保守口径永久隔离 28 个 geometry seeds：train `{1101,...,1319}` 共 16 个、development `{2101,2113,2129,2141,2153,2161}` 共 6 个、OOD `{3109,3121,3137,3163,3181,3191}` 共 6 个；机器可读 JSON 存完整列表。

- 不进入 fit、early-stop 或 route-development；
- 不用于 normalization、calibration envelope、epoch、loss weight、模型或 `eta` 选择；
- 不重新运行模型并写成验证结果；
- 只作为本设计的表示可行性动机。

本 pilot 固定 `eta=0.05`，依据是此前候选协议已经为 Mac pilot 预先指定该值，而不是用 O1 的 opened 结果在 `0.05/0.10` 中择优。本 pilot 不做 `eta` sweep。

## 2. 决策问题与零假设

主问题：

```text
给定 A 射线束上的 raw displacement y、一次 pooled adjoint g=A^T y、
显式 A geometry 和 support，一个 2,729 参数方向模型能否在 8 个新的
route-development geometry clusters、3 种 morphology、3 个固定模型种子上，
相对同预算 CGLS-24 提高 field/H1，同时守住 A measured、A clean、
heldout-B clean reprojection、harm、fallback、统计和完整成本门？
```

零假设是允许输入不能稳定预测有用修正，或者可预测修正在 24 步壳层后没有剩余收益。任何失败都保留为有效 pilot 结论；不得通过换 seed、换 split、改 `eta`、加大网络或事后放宽门补救。

## 3. 部署输入合同

### 3.1 对外 API

模型只能由现有 proposal 合同调用：

```text
proposal(
    observation_uv,          # [1, ray, 2]
    pooled_adjoint_field,    # [1, 1, z, y, x]
    geometry_features,       # declared truth-free geometry
    support,                 # [z, y, x]
) -> raw_correction          # [1, 1, z, y, x]
```

禁止输入 truth、clean observation、family、split、case/base/phantom seed、geometry digest、评价指标、CGLS baseline error、exact-null projector/basis 或任何由这些量派生的标量。`geometry digest` 只可在外部审计日志中标识 case，不可进入模型或 calibration rule。

proposal 前后的 operator ledger 必须完全相同。模型内部不得调用 `A`、`A^T`、dense matrix、SVD、迭代 solver 或评价 renderer。

这里的前后计数只能发现正常的 counter drift，不能把任意 Python closure 变成安全沙箱。正式 L1 runner 只准实例化源码中精确的 `LDFieldFiLM8` 类型并从纯 `state_dict` 加载参数，不接受用户传入 callable、forward hook、operator/truth 属性或 monkey patch；配置明确写入 `counter_check_is_capability_security=false`。因此我们把它称为“审计过的能力隔离”，不声称密码学或进程级防泄漏。

### 3.2 Pilot 张量化

固定 toy 合同为 `B=1` 部署、`V=3` cameras、detector `Hd=Wd=6`、`R=108` rays、volume `D=H=W=12`。训练时一个 batch 只含同一 geometry cluster 的三个 family cases，即 `B=3`；这样 operator 可共享，family 仍不进入模型。

| 名称 | 网络内形状 | 通道定义 |
|---|---|---|
| `y_grid` | `[B,V,2,6,6]` | camera-major reshape 后的 `u/v` displacement |
| `ray_geometry` | `[B,V,7,6,6]` | normalized detector `u,v`；`sin/cos(azimuth)`；`sin/cos(elevation)`；normalized line length |
| detector input | `[B*V,9,6,6]` | `concat(y_grid, ray_geometry)` |
| `g` | `[B,1,12,12,12]` | `P_S A^T y` |
| `S` | `[B,1,12,12,12]` | 与 CGLS-24 完全相同的 support |
| `xyz` | `[1,3,12,12,12]` | voxel-center coordinates in `[-1,1]` |
| volume input | `[B,5,12,12,12]` | `concat(g,S,x,y,z)` |
| raw output | `[B,1,12,12,12]` | 未限幅的方向修正 |

camera-major ray ordering、detector axes、角度单位、坐标手性和 line-length 定义都必须进入 input schema hash。ordering 或 schema 不匹配时不得猜测 reshape。

`y` 的两个通道和 `g` 各自使用只由 fit partition 估计的全局 mean/scale；scale 下限为 `1e-6`。geometry 和 `xyz` 使用合同给定的物理范围归一化，不从 early-stop 或 route-development 重新估计。所有 normalization 数值与 schema hash 随 checkpoint 冻结。

JACRU operator 的审计基准是 CPU float64，因此 L1 的训练、early-stop 和正式 route scoring 全部固定为 CPU float64。当前阶段不在 MPS/float32 与 CPU/float64 之间静默切换；未来若实现独立的 float32 operator，必须另做数值等价协议，不能沿用本次结果。

## 4. 最小模型

模型名暂定为 `LD-FiLM-8`。它只产生第一方向修正，不预测最终 field。

### 4.1 Detector branch

1. 对每台 camera 共享一个 `Conv2d(9,8,k=3,pad=1)` 和 GELU；
2. 接两个 width-8 residual depthwise-separable 2D blocks：`DWConv3x3 -> PWConv1x1 -> GroupNorm(4) -> GELU`；
3. 对每台 camera 做 spatial mean 和 max，得到 `[B,V,16]`；
4. 对 cameras 做 masked mean 和 max，得到 permutation-invariant token `[B,32]`。

联合置换 camera 的 `y` 与 geometry 时 token 必须不变；只置换其中一项是故意的错配测试，不是数据增强。

### 4.2 Volume branch

1. `Conv3d(5,8,k=1)` 将 `[g,S,x,y,z]` 映射到 width 8；
2. 两个 width-8 residual depthwise-separable 3D blocks：`DWConv3x3x3 -> PWConv1x1x1 -> GroupNorm(4) -> GELU`；
3. 每个 block 由独立 `Linear(32,16)` 产生 8 个 scale 和 8 个 bias，执行 FiLM；
4. `Conv3d(8,1,k=1)` 产生 raw correction；该 head 的 weight 和 bias 都零初始化；
5. 输出先乘 support，再由外部 LGWO 壳层限幅。

没有 U-Net skip pyramid、attention、FNO、DeepONet、坐标 MLP 或 learned operator call。第一轮的目标是检验输入信息和训练接口，而不是 backbone 容量。

零 head 专用于先证明 `delta=0` 时与 CGLS-24 精确一致；若直接从严格零范数分支反传，梯度也会为零。实现门通过后、第一步 optimizer 之前，runner 按当前固定 model seed 对 head weight 做一次且仅一次 `Normal(0,1e-4)` 初始化，bias 保持零。该初始化不重试、不调尺度，并与 checkpoint 一起记录；其余层沿 PyTorch 固定 seed 初始化。route 推理使用训练后的权重，baseline recovery 则使用独立的 zero-head control。

### 4.3 精确参数量

| 模块 | 计算 | 参数量 |
|---|---:|---:|
| detector stem | `9*8*3*3 + 8` | 656 |
| 2D DS blocks x2 | `2*((8*3*3+8)+(8*8+8)+2*8)` | 336 |
| volume stem | `5*8 + 8` | 48 |
| FiLM x2 | `2*(32*16 + 16)` | 1,056 |
| 3D DS blocks x2 | `2*((8*3*3*3+8)+(8*8+8)+2*8)` | 624 |
| zero head | `8*1 + 1` | 9 |
| **总计** |  | **2,729** |

mean/max pooling、GELU、support masking、坐标与 normalization buffers 无可训练参数。实现必须用 `sum(p.numel() for p in model.parameters()) == 2729` 锁死；参数漂移直接失败。

### 4.4 限幅与精确回退

令网络输出为 `d_raw`，实际修正为：

```text
d = P_S d_raw
limit = 0.05 ||g||_2
delta_theta = d * min(1, limit / max(||d||_2, eps))
z = g + delta_theta
```

因此 `||delta_theta||_2 <= 0.05||g||_2`。当 zero head 保持初值、模型被 calibration 拒绝或明确启用 baseline control 时，`delta_theta` 必须逐元素为零，随后路径恢复 anchored/reorthogonalized CGLS-24。

## 5. Calibration 与 fail-closed fallback

fallback 只决定“用模型还是精确回到 `delta=0`”，不能选择另一个模型、另一个 `eta` 或额外 solver 步数。

### 5.1 冻结 envelope

结构 envelope 在生成任何结果前写入 manifest：

- volume `12^3`、3 cameras、detector `6x6`、2 displacement components；
- support schema/hash 与训练合同一致；
- camera-major ordering、右手坐标和单位一致；
- `abs(elevation) <= 8 degrees`，detector coordinates、line length 和 samples-per-ray 位于声明的 pilot geometry 范围；
- 所有 tensor finite，且 active camera/ray fraction 为 1。

数值 envelope 只由 24 个 fit cases 建立。特征为 `log RMS(y_u)`、`log RMS(y_v)`、`log RMS_S(g)`、三个 `maxabs/RMS`、三台 camera 的 energy fraction。每维采用 fit min/max，并向两侧扩展 `0.25*(max-min)`；若 range 小于 `1e-6`，使用 `1e-6`。这些 bounds 在 early-stop 前冻结。

### 5.2 动作与审计

- 合同、shape、ordering、dtype 或 finite 检查失败：hard error，不产生重建；
- 输入合法但超出 calibration envelope：设置 `delta_theta=0`，记录 reason code，仍运行 `24F/24A^T` 基线壳层；
- 网络输出 non-finite：设置 `delta_theta=0`，同时将该模型运行标为 breakdown；
- clipping 不是 fallback，但必须记录 raw/applied norm 和 clipping flag；
- fallback rule 不得读取 truth、clean observation、baseline score 或 route label。

route-development 上任何 fallback 都使本最小 learnable pilot 的 signal gate 失败。这样不能靠大量回退把平均安全指标伪装成模型有效。

## 6. 数据与决策隔离

在生成 tensor 前先提交只含 ID、partition、fixture schema、family balance 和 geometry hash 预期字段的 manifest。base seeds 由以下规则确定，禁止手选：

```text
seed(partition,i) =
  int.from_bytes(SHA256("LGWO-A24-L1|partition|i").digest()[:8], "big") mod 2^31
```

若与 opened 隔离名单、其他 partition 或既有 case ID 冲突，则在字符串后追加确定性的 `|retry=k`。实际 seeds 和 manifest hash 必须在生成 observation/truth 前冻结。

| partition | A geometry | heldout-B geometry | clusters x families | cases | 可做的决定 |
|---|---|---|---:|---:|---|
| fit | `train` | `train`，独立 B seeds | `8 x 3` | 24 | 梯度更新、normalization、numeric envelope；B 只是 auxiliary loss |
| early-stop | `train`，disjoint A seeds | `train`，独立 B seeds | `2 x 3` | 6 | 只选 epoch/是否存在 eligible checkpoint |
| route-development | `development`，非 opened seeds | `ood` | `8 x 3` | 24 | 全部方法冻结后一次性 pass/fail |
| fresh OOD | `ood` 或实验室数据 | closed | closed | 0 | L1 禁止打开 |

三个 families 固定为 `smooth_no_interface`、`single_interface`、`two_interface`。每个 geometry cluster 各一个 case，family 只用于生成、分层汇总和错误分析，不进入模型。同 cluster 的三个 cases 共享 A geometry/operator，可组成 batch；任何 case、frame、crop 或 truth 派生量不得跨 partition。

这里的“密封”是预提交程序隔离，不是密码学盲测：seed 和生成器都在私有仓库中可读。约束力来自先提交配置、runner 只允许按阶段 materialize、文件哈希与不按 route 反馈重训；因此报告必须写 `programmatically sealed`，不能写 `blinded`。

每个 A case 的解析 phantom spec 和 voxel-grid truth 原样用于 B，但 B 使用独立 SHA256 派生的 geometry seed，再生成 geometry jitter、ray bundle、continuous clean observation 和 voxel operator。fit/early 使用 `train(A) -> train(B)`，避免把 route-A 的 development pose family提前放进训练；route 才使用 `development(A) -> ood(B)`。B 的任何内容都属于 evaluator-only：模型看不到 B geometry、B observation 或 B residual；训练器只能用 fit-B 计算 auxiliary loss，checkpoint selector 只能用 early-B，route-B 直到统一评分时才生成。它减少“只对 A 射线拟合”的可能，但仍没有更换解析 renderer 或物理模型。

执行顺序固定：

1. 冻结 manifest、模型、loss、optimizer、`eta`、early-stop、calibration 和 route gates；
2. 只打开 fit，依次训练三个固定 model seeds；任一 seed 没有 eligible checkpoint 就停止，route 保持关闭；
3. 三个 full-model seeds 都 eligible 后，训练并冻结所有预声明 baseline/ablation；
4. 保存全部 checkpoint、normalization、envelope 和代码 commit 的 SHA256；
5. 所有方法都冻结后一次性生成并打开 route-development，先计算全部方法，再统一出表；
6. route 报告封存后，才在预声明的 fit 子集运行 exact-null alignment 诊断；
7. 不打开 OOD/fresh，不根据 route 结果产生第二轮训练。

## 7. 训练与 early stop

### 7.1 训练壳层前置门

现有部署函数带 `torch.no_grad()`，因此训练需要一个数学相同、可微的 training twin。它在允许训练前必须通过：

- CPU float64、`delta=0` 相对 CGLS-24：field difference `<=1e-10`，measurement difference `<=1e-11`；
- 固定随机 bounded `delta` 时，training twin 与部署壳层 field/residual 相对差都 `<=1e-10`；
- checkpoint 24 显式 ledger 恰为 `24F/24A^T`；
- 两遍 measurement-space modified Gram-Schmidt，最大 residual increase `<=1e-12`；
- 对模型参数的 directional finite-difference gradient relative error `<=5e-3`；
- zero-initialized head 的首个 forward 必须逐元素返回零修正。

任一失败先修训练壳层，不允许开始拟合。

### 7.2 损失

对每个 fit case，令 `x_theta` 为 A 上 24 步输出，`x_b` 为缓存的同预算 CGLS-24，`x_star` 为 synthetic truth。`r_Am` 是 A measured residual，`r_Ac` 是相对 A continuous-clean observation 的 residual，`r_Bc` 是把同一个重建场投影到 heldout-B 后相对 B continuous-clean observation 的 residual。定义：

```text
E2      = ||x_theta-x_star||_2 / (||x_star||_2+eps)
EH1     = ||grad(x_theta-x_star)||_2 / (||grad(x_star)||_2+eps)
q_field = E2 / (E2_baseline+eps)
q_Am    = ||r_Am(theta)||_2 / (||r_Am(baseline)||_2+eps)
q_Ac    = ||r_Ac(theta)||_2 / (||r_Ac(baseline)||_2+eps)
q_Bc    = ||r_Bc(theta)||_2 / (||r_Bc(baseline)||_2+eps)
P_A     = ||A delta_theta||_2^2 / (||A g||_2^2+eps)
h(q,t)  = max(0, log(q/t))^2

L_case = E2 + 0.25 EH1
       + 0.02 P_A
       + 1.0 h(q_Am,1.02)
       + 1.0 h(q_Ac,1.02)
       + 2.0 h(q_Bc,1.02)
       + 1.0 h(q_field,1.00)
L = mean_fit_cases(L_case)
```

`A g` 从 `delta=0` baseline 的首个 projected direction 缓存；当前 candidate 的首个 projection已经给出 `A(g+delta_theta)`，两者相减得到 `A delta_theta`，因此 `P_A` 不增加 solver call。`A x_theta=y_A-r_Am(theta)` 可复用，所以 A clean residual 不新增 forward call；`r_Bc` 每次评分必须显式支付一个 evaluator-only `B.forward(x_theta)`，单独记账，不能藏进 `24F/24A^T`。

truth 和 clean observation 只存在于 training/evaluator 侧，绝不传入 proposal。exact-null direction、SVD basis 和 alignment 指标不进入损失。

### 7.3 优化器与 epoch 选择

```text
optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
batch: 3 cases sharing one geometry cluster, one per family
model seeds: [1630052265, 102254037, 2925587], all reported, no best-seed selection
gradient clip: global norm 1.0
epochs: min 8, max 30
eligible epoch range: 8..30
adaptive termination: false; all 30 epochs are run
device/dtype: CPU float64; no silent fallback
```

每个 epoch 后只在 6 个 early-stop cases 上评分。eligible epoch 必须同时满足：

- mean A measured、A clean、heldout-B clean ratio 各自 `<=1.05`；
- 每个 case 的三种 residual ratio 都 `<=1.10`；
- breakdown、non-finite 和 fallback 数量均为 0。

在第 8--30 epoch 的 eligible checkpoints 中最小化 `mean(E2+0.25*EH1)`；相同到 `1e-8` 时取更早 epoch。固定运行 30 epochs，不使用自适应 patience。若任一固定 model seed 没有 eligible checkpoint，则 L1 primary interface 失败，route-development 保持关闭；禁止只保留表现最好的 seed。

## 8. 显式 `A/A^T` 成本

模型 proposal 本身为 `0F/0A^T`。部署一例的账本始终是：

| 阶段 | F | A^T |
|---|---:|---:|
| pooled anchor `g=A^T y` | 0 | 1 |
| first projection `A(g+delta)` | 1 | 0 |
| 23 continuation steps | 23 | 23 |
| **每 case 总计** | **24** | **24** |

以同 geometry 的三个 cases batch 为一次 operator API invocation，最大 30 epochs 的单个 model-seed 路线为：

| 工作 | batched A-F | batched A-AT | case-equivalent A-F/A-AT | batched evaluator B-F |
|---|---:|---:|---:|---:|
| 24 fit cases x 30 epochs | 5,760 | 5,760 | 17,280 / 17,280 | 240 |
| 6 early-stop cases x 30 | 1,440 | 1,440 | 4,320 / 4,320 | 60 |
| frozen model 的 24-case route | 192 | 192 | 576 / 576 | 8 |
| **单 model seed 上界** | **7,392** | **7,392** | **22,176 / 22,176** | **308** |
| **三个 primary seeds 上界** | **22,176** | **22,176** | **66,528 / 66,528** | **924** |

L1 的 5 个 trainable methods（full、fixed direction、g-only、no-raw-y、no-geometry）都跑 3 seeds。按固定 30 epochs、只对 fit 反传、route 最终评分计算，冻结的全实验最坏账本为：

- A solver：`335,088F / 335,088A^T` case-equivalent，或各 `111,696` 次 three-family batched API；
- B evaluator：`13,962F` case-equivalent，或 `4,654` 次 batched API；
- fit autograd 反向传播另含约 `259,200F / 259,200A^T` 的线性映射等价重算，绝不并入 deployable `24/24`；
- route 封存后若运行 exact-null alignment，另付 `2,002F + 2` 次 `216x1000` SVD。

其中共享 CGLS-24 cache 覆盖 54 cases；`delta=0` 与 `1.05g` scaling control 只在 route 额外重跑。route 部分只有 eligibility 门通过后才生成。以上是上界，不是运行耗时；实际 summary 必须逐方法、逐 seed 分账。

autograd 对 operator primitive 的 backward 计算不伪装成新的显式 solver call，但必须通过 wall time、peak memory 和 device fallback 另行报告。若实现不能可靠区分 API ledger 与反向传播成本，至少报告上述 case-equivalent 数和端到端计时。

## 9. Exact-null alignment：仅训练集、仅事后诊断

该诊断不能成为 loss、early-stop、模型/epoch/`eta` 选择条件或 route gate。为排除无意反馈，它只在 route-development 报告及其 hash 已封存后运行。

诊断子集预先固定为 fit partition 的前 2 个 geometry clusters，共 6 cases。每个 geometry 只构造一次 CPU float64 dense projector：当前 `12^3` outer-boundary support 有 1,000 个 active voxels、216 个 measurements，预计每个 projector setup 为 1,001 个 evaluator-only `F`、`0A^T`，随后做一次 `216x1000` thin SVD。两个 clusters 合计 2,002 个 evaluator-only `F` 和两次 SVD；这些调用不得并入 deployable `24/24` 账本。

对每个诊断 case：

```text
n_star       = P_null(A) x_star
d            = applied bounded delta_theta
d_null       = P_null(A) d
signed_cos   = <d,n_star> / (||d|| ||n_star|| + eps)
null_fraction= ||d_null||^2 / (||d||^2 + eps)
target_capture = ||d_null|| / (||n_star|| + eps)
```

同时报告 `||A d_null||/(||A|| ||d||+eps)`、oracle rank/tolerance、raw/applied correction norm 和 clipping。若任一 norm 小于 `1e-12`，对应 alignment 记为 undefined，不填零。

只允许逐 case 表和无阈值描述，例如“在这 6 个 fit cases 上方向与 exact-null truth component 同向/不同向”。不计算通过门，不因 alignment 低而重训，也不能把 alignment 高写成 unseen predictability、泛化或物理 null-space 学习。离散 toy `ker(A)` 也不等于真实光学系统的不可观测空间。

## 10. 强基线与消融

所有方法使用同一 manifest、24 步壳层、`eta=0.05`、early-stop 规则和 route 一次打开。full model 是唯一 primary candidate；不能在 route 上挑表现最好的消融替代它。

| 方法 | 参数 | 目的 |
|---|---:|---|
| zero-start CGLS-24 | 0 | 同预算主基线 |
| anchored shell, `delta=0` | 0 | 证明 learned shell 精确恢复主基线 |
| train-fitted fixed direction | 1,000 | 检验平均 field prior 是否已足够；只存 active support 参数 |
| `g-only` volume model | 681 | 去掉 raw `y` 和 geometry/FiLM |
| no raw `y` | 2,585 | 只用 geometry token 与 `[g,S,xyz]` |
| no geometry | 2,225 | detector 只看 raw `y` |
| **full `LD-FiLM-8`** | **2,729** | 预声明 primary candidate |

fixed direction 从零初始化并用同一最终-field loss训练，但 proposal 不读取 case 输入。它与 full model 同样限幅并支付同样部署预算。full model 若不能在 route mean field gain 上超过 fixed direction 至少 1.0 percentage point，则没有证据说明条件化网络优于平均训练先验。

另做三个不训练的一致性测试：

1. 同时置换 cameras 的 `y` 与 geometry，输出相对差 `<=1e-6`；
2. 只置换 geometry，记录输出变化和 route 指标，但不设“变化越大越好”的门。
3. 用 `1.05g` 替换首方向的纯缩放控制，验证 exact line search 对方向缩放的不变性，防止把数值缩放误写成学习收益。

若三个 full-model seeds 中任一个在 early-stop 阶段无 eligible checkpoint，后续消融不运行，route 不打开。若三个 seeds 都 eligible，则所有消融先训练并冻结，再统一打开 route，避免按 route 反馈增删实验。

## 11. 失败门

### 11.1 训练前硬门

- 输入合同或 forbidden-input audit 失败；
- 参数量不等于 2,729，zero head 不为零；
- proposal 前后 ledger 改变；
- `delta=0` baseline recovery、training/deployment parity 或 gradient check 失败；
- joint camera permutation invariance 失败；
- 任一显式路径不是 checkpoint-24 的 `24F/24A^T`。

动作：不训练，状态写成 `IMPLEMENTATION_GATE_FAILED`。

### 11.2 Fit/early-stop 门

- loss、gradient、field、residual 或 correction 出现 non-finite；
- residual monotonicity 失败；
- fit/early-stop calibration fallback 非零；
- 超过 25% 的 fit cases 在选中 epoch 触发 correction clipping；
- 30 epochs 内没有 eligible checkpoint。

动作：冻结失败日志，route 保持关闭；不得更换 seed 或超参重试。

### 11.3 Route-development 信号门

在 CPU float64、8 个 geometry clusters x 3 families = 24 个 route cases、3 个固定 model seeds 上，full model 必须全部满足。独立统计单位是 geometry cluster：先在 cluster 内平均 3 个 families 和 3 个 model seeds，再做跨 cluster 汇总；不把 72 个 case-seed observations 假装成 72 个独立样本。

- seed-averaged mean field gain `>=5%`，每个 model seed 的 mean field gain 都 `>=2%`；
- 以 50,000 次 cluster bootstrap 得到的 field gain 双侧 percentile 95% 区间下界 `>0%`；该区间仍只标为小样本 synthetic pilot 描述；
- 至少 7/8 clusters 的 seed/family-averaged field gain `>0%`，每个 family 的 seed-averaged mean field gain `>=0%`；7/8 在独立对称符号零假设下的精确单侧概率为 `9/256 = 0.03515625`；
- seed-averaged mean H1 gain `>=3%`；
- mean A measured、A clean、heldout-B clean residual ratio 均 `<=1.05`；
- 每个 cluster 的三种 residual ratio 均 `<=1.10`；
- `>1%` field harm 的 case-seed rate `<=5%`，worst case-seed field gain `>=-5%`；
- fallback、breakdown、non-finite 均为 0；
- maximum residual increase `<=1e-12`，measurement-direction orthogonality defect `<=1e-10`；
- candidate 与 baseline 都恰为 `24F/24A^T`；
- full mean field gain 比 train-fitted fixed direction 至少高 1.0 percentage point。

任一项失败，结论只能是“本预声明 learnable-direction pilot 未通过”。不能打开 OOD/fresh，也不能把某个消融的事后优势改写成 primary success。

即使全部通过，也只允许结论：“三模型种子、24-case route-development synthetic signal 值得进入另行预注册的 L2 基准。”它仍不是算法成功，更不是 DeepONet/FNO 优势、fresh 泛化、真实 BOST 或突破证据。L2 才能加入 matched-budget DeepONet/FNO/PINO、closed fresh geometry、更多 renderer 或真实实验室数据，并必须另立文档重新冻结。

## 12. Mac 预计运行成本

以下是 Apple Silicon、16 GB unified memory 的保守工程估算。最终冻结 implementation gate 在 source commit `5230a5f` 上，用永久排除的 seeds `1101/1117` 完成三-family、`12^3`、K=24 CPU-float64 forward/B-evaluator/backward，总 step wall time 约 `0.083 s`、peak RSS 约 `271 MB`；这些时间只用于工程容量判断，不是性能 benchmark。正式训练仍必须逐 arm/seed/epoch 记录 wall time、CPU time、peak RSS、device fallback 和输出体积。

| 项目 | 预计时间 | 预计峰值资源 |
|---|---:|---:|
| 合同/parity/gradient tests | 1--3 min | <2 GB |
| 30-case pre-route CGLS-24 cache | 2--6 min | <2 GB |
| 单个 model seed，固定 30 epochs | 1--5 min | 2--4 GB |
| 三个 primary seeds | 3--15 min | 2--4 GB，串行运行 |
| 一次 24-case CPU float64 route scoring | 3--10 min | <2 GB |
| 2-cluster train-only exact-null diagnostic | 2--10 min | 2--4 GB |
| 5 trainable methods x 3 seeds | 10--60 min | 2--4 GB，串行运行 |

模型 checkpoint 本身小于 0.1 MB；包含 manifests、CSV、JSON、日志和必要诊断图的 pilot 输出目标小于 100 MB。不得持久化每个 epoch 的完整 autograd graph 或 24 步全部 field history。若单个 seed 超过 60 min、peak memory 超过 8 GB 或必须改变数学路径才能运行，则状态记为 `MAC_COST_GATE_FAILED`，不通过减少 CGLS 步数或降低 scoring 精度偷渡。

## 13. 允许的报告措辞

运行前只允许：

> Opened synthetic exact-null truth-oracle 显示了表示 headroom；最小可学习模型尚未运行，这不是算法成功证据。

若 route 失败：

> 预声明的三种子、双射线束 learnable-direction pilot 未通过 route-development 门；opened oracle signal 没有转化为当前允许输入下的稳定模型证据。

若 route 全部门通过：

> 预声明的三模型种子、8 geometry clusters、24 cases、heldout-B synthetic pilot 通过了进入 L2 matched-baseline benchmark 的门；尚无 DeepONet/FNO 优势、fresh、真实 BOST、泛化、突破或新颖性结论。

禁止使用“学会了真实 null space”“解决了 LGWO/BOST”“优于现有方法”“可泛化到未见 rig”“取得突破”等措辞。

## 14. Implementation gate 冻结结果

最终机器状态为 `PASS_IMPLEMENTATION_GATES_FIT_RUNNER_IMPLEMENTATION_AUTHORIZED_ROUTE_SEALED`。它只授权继续实现 fit-only runner，不授权开始训练或打开任何科学 partition。报告与独立验证位于：

- `demo_t16_operator/results/lgwo_a24_l1_implementation_gates_v1/summary.json`
- `demo_t16_operator/results/lgwo_a24_l1_implementation_gates_v1/independent_validation/validation.json`
- `demo_t16_operator/results/lgwo_a24_l1_implementation_gates_v1/checksums.sha256`

最终可核验事实：

- frozen config canonical SHA256 为 `c2aa25a5...04b3b`，validator 共执行 137 项协议断言；
- 完整模型与 head 由 seed `1630052265` 重建后的 state SHA256 固定为 `8d1629b1...17028`；两个独立进程除 timing/RSS 外全部字段一致；
- A 侧完整路径为 `24F/24A^T`，B evaluator 为 `1F/0A^T`；
- 34/34 参数张量梯度有限且非零，CPU float64、device fallback 为 0；
- repo 专用 L1 route/fresh 扫描命中 0，科学 partition materialized case 数仍为 0；
- 独立 validator 不导入 runner，重建模型 state、复核 loss 算术、来源 commit、调用账本与 claim boundary，共通过 57 项断言。

首次正式调用因协议元数据 `schema` 被误传给 fixture constructor 而 fail-closed；修复并补全端到端测试后，第二次运行又发现完整 backbone 未由 model seed 控制。修订 1.1 在任何 fit/early/route/fresh case 生成前固定全模型 RNG，第三次运行与独立进程 replay 才稳定。保留这两次失败是证据的一部分：当前通过的是可重复工程门，不是模型效果。

**突破监测：无算法突破。** 新增的是可信训练基础设施和一条明确的下一门：先冻结 final loss、H1 stencil、normalization、五个 trainable arms、checkpoint/resume 与成本账本，再实现 fit-only runner；在此之前不得生成 fit data 或宣称开始训练。
