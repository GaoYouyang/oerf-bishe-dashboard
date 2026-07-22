# LGWO-A24-L1 fit-only runner 合同

**状态：** `FROZEN_PREFIT_AUDIT_REMEDIATION_GATES_PENDING_NO_SCIENTIFIC_FIT_AUTHORIZATION`

**机器协议：** `lgwo-a24-l1-sealed-observable-pilot-config-1.3`，当前 canonical SHA256 为
`7de695ebc419ad2e7a7edba0c10d45b9ff31027a0252c41c2481faf56f4eb085`。

**当前边界（2026-07-20 复核）：** scientific fit/early-stop/route-development/fresh-OOD
物化数均为 `0`，optimizer step 为 `0`。没有训练结果、算法成功、未见 rig 泛化、真实 BOST
结论或突破。本文只冻结第一次 scientific fit 物化和第一次 optimizer step 之前的合同。

**重要证据失效规则：** 协议、critical source 和测试已经不同于旧 implementation-gate commit，
所以旧 implementation evidence 不能继续背书当前训练栈。必须先形成干净 source commit，再在该
commit 上重新生成 validator report、聚焦测试记录、implementation evidence、checksums 和独立审计；
在此之前 Stage 1 与 Stage 2 都不得签发。

## 1. 不可逆执行顺序

1. 对 protocol 1.3、loss、arm registry、fit manifest/cache、normalization、operator ledger、
   checkpoint、授权根和 fit materializer 跑完聚焦回归；
2. 修清全部失败门，形成并推送一个干净 source commit；
3. 在该 commit 上重新生成 protocol report、implementation evidence、聚焦测试清单/结果、raw pytest
   JUnit XML 与 checksums；
4. 取得独立第三次审计 `GO_EXACT_24_FIT_ONLY_MATERIALIZATION_NO_TRAINING`，并保持所有 claim flags 为 false；
5. 在仓库外生成 **Stage 1 materialization authorization root**；
6. 只生成并核对一个 24-entry fit manifest，然后只 materialize 24 个 fit cases；
7. 从这 24 个 fit-A observable cache 冻结 normalization 与 calibration envelope；
8. 核对 24 x 3 = 72 个 cache manifests、216 个数组、双重哈希和 optimizer step = 0；
9. 在仓库外生成 **Stage 2 training authorization root**；
10. Stage 2 通过后才可构造 optimizer，并为五个 arms x 三个 model seeds 固定运行 30 epochs；
11. 只有全部 15 条轨迹都有 eligible checkpoint，才允许后续协议规定的 early/route 解封；
12. route 打开后禁止回到 fit 重训、改 loss、换 seed、换 arm 或删除失败轨迹。

任何一步 hard fail 都保留日志并停在原层级。不得通过换 seed、重试超参、降低精度、减少
24 步或只展示最好模型恢复流程。Stage 1 只允许 fit 物化，绝不隐含训练许可；Stage 2 只在
fit 数据与统计量已经封存、且 optimizer 仍为 0 步时签发。

## 2. Fit manifest 与权限分层

fit 固定为 8 个 geometry clusters，顺序严格按协议 `base_seeds` 的索引 `0..7`；每个 cluster 内 family 顺序严格为：

```text
smooth_no_interface
single_interface
two_interface
```

每个 cluster 的 A/B seed 和 split 只能由 `build_lgwo_a24_l1_protocol_pair_contract(config, partition="fit", cluster_index=i)` 产生。禁止调用者自填 B split 或 B seed。

持久化层分开：

- `observable_manifest`：A observation、A geometry schema/hash、support schema/hash、A-derived pooled adjoint hash；不含 truth、clean target、family、split、case ID、seed、B 或 baseline error；
- `training_label_manifest`：synthetic truth 与 A continuous-clean target，仅 loss/evaluator 可读；
- `heldout_b_manifest`：B operator reconstruction contract 与 B continuous-clean target，仅辅助 loss/evaluator 可读；
- `audit_manifest`：partition、cluster、family、seed、digest 和跨层绑定，只供 runner 外层审计，不能传入 proposal。

operator、Python callable、autograd graph 和任意 pickle 对象都不得持久化。数组只能使用显式 dtype/shape 的 NPZ/NPY，加载必须 `allow_pickle=False`；operator 由 frozen geometry/config 在进程内重建。

### 2.1 Canonical support 是物理合同，不是可调 mask

所有 fit-A 与 heldout-B operator 必须在任何 proposal/cache/loss 调用前绑定同一个
`12 x 12 x 12` zero-outer-boundary support：最外一层为 0，中央 `10 x 10 x 10` 为 1，
因此 active voxel **恰好为 1000**。禁止使用全 1 的 1728-voxel mask，也禁止用另一个“同样有
1000 个点”的平移/重排 mask 冒充它。

```text
canonical support SHA256:
5e36e442695dbc7ef6f32b6ca9096aa7edaf2ce04efd2f9d9d463bda62146436

fixed-direction semantics SHA256:
fdf49827439522a2090028631c4bf3cd10ba846f7ef9fd9ea57922af9ecbacb4
```

参数到体素的映射固定为全局 flattened z-y-x 顺序中的中央 interior，不能按某个 case 的 support
相对位置重新编号。24 个 fit cases 的 A/B support digest 必须完全相同；任一 operator 在 support
绑定前被调用、A/B support 不相同或 support digest 漂移，都 hard fail。

## 3. Fit-only normalization

统计只读 24 个 fit-A observable payload，按 manifest 固定顺序在 CPU float64 中做两遍确定性归约。early、route、B、truth、clean target、family 与评价指标都不参与。

### 3.1 Observation

对 A noisy displacement 的 `u/v` 两通道分别计算 population mean 与 population standard deviation：

```text
mu_c    = mean_all_fit_cases_and_rays(y_c)
sigma_c = sqrt(mean((y_c-mu_c)^2))
scale_c = max(sigma_c, 1e-6)
```

### 3.2 Pooled adjoint

对 `g=P_S A^T y` 只在 `support>0.5` 的 active voxels 计算一个全局 scalar mean/scale：

```text
mu_g    = mean_all_fit_cases_and_active_voxels(g)
sigma_g = sqrt(mean((g-mu_g)^2))
scale_g = max(sigma_g, 1e-6)
```

输出 manifest 同时保存十进制值、`float.hex()`、样本数、固定归约顺序、schema SHA256 与输入 tensor hashes。任何非有限值、空 active support 或重复生成 hash 不一致都 hard fail。

## 4. Observable-only calibration envelope

每个 fit case 在 normalization 后只计算六个可部署标量：

```text
u_rms, u_max_abs, v_rms, v_max_abs, g_active_rms, g_active_max_abs
```

每个 feature 的 fit envelope 为：

```text
lo = min_fit(feature)
hi = max_fit(feature)
pad = 0.25 * max(hi-lo, abs(lo), abs(hi), 1e-6)
accepted = lo-pad <= feature <= hi+pad
```

geometry 另外执行 exact schema、ray order、camera count、detector shape、finite pose 与 line-length contract；不使用 geometry digest 或 partition 名称作 acceptance feature。任何 feature 越界时 correction 必须精确设为零并记录 fallback；L1 route 要求 fallback 为 0 才可能通过。

## 5. 正式 loss

所有 norm 按 case 计算，`epsilon=1e-12`。`E2` 是 support 内 field relative-L2。`EH1` 是 gradient-only relative error：在 x/y/z 三轴使用物理 spacing 的 forward difference；只有一条 edge 的两个端点都满足 `support>0.5` 才计入。无 one-sided boundary extrapolation，不把 support 外的零填充值制造成界面梯度。

```text
E2      = ||S(x_theta-x_star)||_2 / max(||S x_star||_2, epsilon)
EH1     = ||D_S(x_theta-x_star)||_2 / max(||D_S x_star||_2, epsilon)
q_field = E2 / max(E2_baseline, epsilon)
q_Am    = ||r_Am(theta)||_2 / max(||r_Am(baseline)||_2, epsilon)
q_Ac    = ||r_Ac(theta)||_2 / max(||r_Ac(baseline)||_2, epsilon)
q_Bc    = ||r_Bc(theta)||_2 / max(||r_Bc(baseline)||_2, epsilon)
P_A     = ||A delta_theta||_2^2 / max(||A g||_2^2, epsilon)
h(q,t)  = relu(log(max(q,epsilon)/t))^2

L_case = E2 + 0.25 EH1 + 0.02 P_A
       + h(q_Am,1.02) + h(q_Ac,1.02)
       + 2 h(q_Bc,1.02) + h(q_field,1.00)
L = mean over the three-family cluster batch
```

物理 spacing 对 volume shape `[D,H,W]` 按 z/y/x 分别为：

```text
dz = (z_max-z_min)/(D-1)
dy = (y_max-y_min)/(H-1)
dx = (x_max-x_min)/(W-1)
```

`A delta` 必须从已支付的首方向投影与缓存 `Ag` 相减得到；A measured/clean projection必须复用 solver residual/已有 `A x`；B clean 每个 batch/epoch 显式支付一次 batched evaluator forward。不得为 loss 隐藏额外 A/A^T 调用。

## 6. 五个训练臂

固定 registry：

1. `full`（2729 参数）：raw observation + pooled adjoint + geometry + support；primary；
2. `fixed_direction`（1000 参数）：只学习全局 `12^3` 网格中央 `10^3` canonical interior 的固定体方向，不读取 case 输入；
3. `g_only`（681 参数）：只读 pooled adjoint + support + xyz；
4. `no_raw_observation`（2585 参数）：与 full 共享声明架构，但 raw observation 通道结构性删除；
5. `no_geometry`（2225 参数）：与 full 共享声明架构，但 geometry value 通道结构性删除。

每个 arm 都必须有 exact class/registry、constructor seed、state schema/hash、参数量、forbidden
attribute/hook audit 和输入屏蔽因果测试。不得用任意 closure 冒充消融。参数量、canonical support
digest 与 fixed-direction 坐标语义已经写入协议 1.3；任一项漂移都不得 materialize fit。

## 7. 优化与固定顺序

```text
device/dtype: CPU float64
optimizer: AdamW(lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=1e-4, amsgrad=false, maximize=false,
                 foreach=null, capturable=false, differentiable=false,
                 fused=null, decoupled_weight_decay=true)
gradient clipping: global norm 1.0
epochs: exactly 30
eligible epochs: 8..30
shuffle: false
batch: one geometry cluster, exactly three families in fixed order
model seeds: [1630052265, 102254037, 2925587]
adaptive termination: false
hyperparameter retries: 0
seed replacement: 0
```

epoch 外层为 `1..30`，cluster 内层为 `0..7`。每个 arm/seed 都完整运行 240 optimizer steps；即使某一步已经不 eligible，也不能提前终止或跳过后续 epoch。

## 8. Checkpoint 与 resume

checkpoint schema 固定为 `lgwo-a24-l1-fit-checkpoint-1.1`，epoch metrics schema 固定为
`lgwo-a24-l1-fit-epoch-metrics-1.1`。checkpoint 分成 canonical JSON manifest 与 tensor-only
safetensors（不可用时 NPZ，加载 `allow_pickle=False`）；禁止 `torch.save`/pickle。

每条 arm/seed 轨迹固定含 epoch `0..30` 共 31 个节点。epoch 0 是 optimizer state 尚未产生、
operator ledger hash 为 64 个零的零成本初始化；epoch 1..30 每个节点必须绑定：

- protocol hash、source commit、fit manifest hash；
- **Stage 2 training authorization root SHA256**；
- exact registered arm、model seed、参数/缓冲区 schema、精确参数量与 dtype；
- 完整 AdamW parameter ID/order、slot tensors 与超参；
- normalization hash、calibration hash、model/optimizer/metric-history hashes；
- operator-ledger schema、API mode、ledger-file SHA256、API 与 case-equivalent 双重成本；
- route/early 仍封存、nonfinite/fallback/resource ledger。

仓库外还必须为每个 epoch 保存一个
`lgwo-a24-l1-checkpoint-node-anchor-1.0` 节点锚。锚同时绑定 checkpoint manifest、tensor、
model、optimizer、metrics 与 parent-chain hashes；不能仅靠 checkpoint 所在目录自证完整。

只允许从同 arm、同 seed、同 config/source/fit manifest/normalization/calibration/Stage-2 root、
同 optimizer state 的下一连续 epoch resume。临时文件先 `fsync` 再原子落盘；残留 `.tmp` 只能
隔离并 hard fail，不能自动当作完成节点。hash 不一致、锚缺失、ledger 文件与 metrics 不一致、
跨 arm/seed、epoch 缺口、改超参、route 已读、best-seed 选择或删除失败 epoch都 hard fail。

## 9. 成本账本

operator-ledger schema 为 `lgwo-a24-l1-operator-call-ledger-1.0`，API mode 为
`batched_three_case_cluster`。一次 batched API 调用同时处理同一 geometry cluster 的三个 family，
所以必须同时报告真实 API 调用数与 case-equivalent 工作量；只能报其中一个会 hard fail。

每个 epoch、每个 arm/seed 的固定成本为：

```text
batched API:       192 A-F + 192 A-A^T + 8 B-F
case-equivalent:   576 A-F + 576 A-A^T + 24 B-F
training backward: 552 A-F-equivalent + 576 A-A^T-equivalent + 24 B-A^T-equivalent
optimizer steps:   8
explicit ledger events: 392
```

每个 arm/seed 的 30-epoch fit 账本固定为：

```text
A solver batched API: 30 * 8 * 24 = 5,760 F and 5,760 A^T
A solver case-equivalent: 30 * 24 * 24 = 17,280 F and 17,280 A^T
B evaluator batched API: 30 * 8 = 240 F
B evaluator case-equivalent: 30 * 24 = 720 F
optimizer steps: 30 * 8 = 240
```

每条 arm/seed 的 30 个非零 epoch 还必须有 30 个不同的非零 operator-ledger hash，累计
`11,760` 个 explicit events。五 arms x 三 seeds：batched API 合计
`86,400 A-F / 86,400 A-A^T + 3,600 B-F`；case-equivalent 合计
`259,200 A-F / 259,200 A-A^T + 10,800 B-F`；training backward 另列
`248,400 A-F-equivalent / 259,200 A-A^T-equivalent + 10,800 B-A^T-equivalent`，另加一次共享 baseline cache。

wrapper 的 API event、底层 operator counter 与 checkpoint metrics 必须三方对账。autograd 线性映射
重算单列，不伪装成显式 solver API call，也不能从总成本中消失。每条轨迹保存每 epoch 的 wall/CPU、
peak RSS、fallback、nonfinite、clipping rate、A/B calls 与 artifact bytes。

## 10. 解封前硬门

### 10.1 Stage 1：只允许 scientific fit 物化

Stage-1 external root schema 为
`lgwo-a24-l1-materialization-authorization-root-1.0`。它必须绑定同一个干净 source commit、
critical-path file hashes、protocol report、重新生成的 pre-fit evidence/checksums、exact 24-entry
fit manifest、授权输出根和独立第三次 `GO` 审计；claim flags 必须全部为 false，route/fresh 扫描
必须为 0。它唯一允许的是生成这 24 个 fit cache triplets，optimizer training 明确为 false。

pre-fit evidence schema 1.1 不能只接受一份人工汇总的“tests passed”。它必须同时绑定
`focused_test_manifest.json`、`focused_test_results.json` 和 pytest 原始 JUnit XML 的文件字节
SHA256。packer 会直接解析 raw XML，逐条拒绝 `failure`、`error` 或 `skipped`，以
`classname::name` 形成唯一 testcase identity，核对 testcase 总数、排序后 identity-list SHA256，
并确认预声明的每个 negative-gate testcase 确实出现在原始 JUnit 中。Stage-1 root 还要再次绑定这些
raw-input hashes；JSON summary 不能替代原始测试输入。

Stage 1 通过后，materializer 仍要独立复核授权包、fit entry 顺序、canonical support、A/B provenance、
输出根为空且不可覆盖，并保证物化结束时仍是 0 optimizer steps、0 early、0 route、0 fresh。

### 10.2 Stage 2：fit 数据封存后才允许训练

Stage-2 external root schema 为 `lgwo-a24-l1-training-authorization-root-1.0`。它必须链回 Stage 1，
并逐文件复核：

- 24 个 proposal、24 个 training-label、24 个 heldout-B evaluator，共 72 个 cache manifests；
- 上述三类 cache 共 216 个 NPY arrays，manifest 内 hash 与文件 payload hash 均一致；
- exact fit manifest、24-entry order、normalization manifest 与 calibration-envelope manifest；
- operator-ledger schema `lgwo-a24-l1-operator-call-ledger-1.0`；
- seal 时 optimizer step 严格等于 0，early/route/fresh 仍未物化。

只有 Stage-2 root SHA256 被 checkpoint contract 1.1 绑定后，runner 才能构造 optimizer 并执行第
一个 step。Stage 1 与 Stage 2 都必须位于受审计的外部根中，runner、cache 目录或 checkpoint 目录
不能给自己签授权。

### 10.3 全部先决条件

在 fit runner 首次科学物化或训练前，必须全部通过：

- 24-entry fit manifest 双次生成 hash 相同，early/route/fresh 路径不存在；
- proposal payload 的类型/字段/来源 audit 不含 truth、clean、family、split、case/seed、B；
- A/B operator 都在首次调用前绑定 exact canonical 1000-voxel support；
- five-arm registry、参数量、input-mask 因果和 full-state seed replay 通过；
- loss 对手算、batch、support edge、空 residual/projection 拒绝、零分母和 directional gradient 通过；
- zero-head baseline parity、audited wrapper 与 API/case-equivalent 双账本通过；
- normalization/envelope 双次生成 bitwise/hash 一致；
- checkpoint 31 节点 chain、exact registered arm/state、Stage-2 root、external node anchors、ledger-file
  hashes、round-trip、tamper、crash residue 与 resume-continuous 等价测试通过；
- 两阶段授权根的正向与 tamper/unknown/symlink/no-clobber 负向测试全部通过；
- source/config/test commit 已推送，当前 commit 的 validator、implementation evidence、聚焦测试结果、
  independent GO audit 与 checksums 已封存。

## 11. 当前验证快照与阻断项

以下数字只对应 2026-07-20 本次实际运行，不是模型性能：

- 独立 protocol validator：`180` 项断言通过，canonical config SHA256 为
  `7de695ebc419ad2e7a7edba0c10d45b9ff31027a0252c41c2481faf56f4eb085`；报告确认
  `scientific_partition_case_count = 0`、L1 route/fresh 文件系统命中数为 0、所有 claim flags 为 false；
- 18-file 聚焦 pytest：`361 passed`；当前所列 runner、materializer、授权根、raw-JUnit evidence、implementation
  gate、checkpoint、
  operator-ledger、normalization、cache、loss、arm 与 protocol 测试均无失败；
- 本次交互式回归用于复核当前工作树，未把 console PASS 冒充正式 evidence。Stage 1 所需的下一次
  clean-commit 运行必须用 pytest 直接输出 raw JUnit XML，并让 evidence packer 绑定 XML 原始字节
  SHA256、`classname::name` testcase identity-list SHA256 与 `361` 条实际 testcase 记录；
- 当前总体判定仍是 **NO-AUTH / fail closed**，原因已从“测试红灯”收敛为“尚未形成干净 commit，
  尚未重建 raw-input-bound implementation evidence、独立 GO 与 Stage-1 root”，并且 epoch 级参数更新、
  ledger、checkpoint、external anchor 仍不是单一可恢复事务。测试全绿本身不签发授权；
- scientific case 仍为 `0`，optimizer step 仍为 `0`，没有模型结果或突破。

独立崩溃审计进一步确认：当前实现能在 ledger/checkpoint/anchor 前缀不一致时 fail-closed，但不能在
`optimizer.step()` 后的任意中断点可靠恢复。第一次科学 step 前必须完成 staging transaction、独立 anchor、
原子 `TIP` 与故障注入；完整边界见
[epoch 事务与恢复独立审计](lgwo_a24_l1_epoch_transaction_audit_2026-07-20.md)。在何远哲师兄确认真实
callable、数据合同和主问题之前，不继续扩建该基础设施。

第三轮只读复审发现的两个元数据 P0 已在当前工作树关闭：checkpoint 验证每个 AdamW slot 的
`step == epoch * 8`；独立 chain validator 从明确 ledger root 读取 30 份原始文件，复算 11,760 条事件、
双成本与 240 个逻辑 steps。当前 `361 passed` 仍不能授权训练，因为 epoch 跨事务 P0、干净提交、真实
接口确认与外部独立 GO 仍未完成；完整分级见
[第三轮独立只读复审](lgwo_a24_l1_third_readonly_audit_2026-07-20.md)。

形成干净 commit 后必须重新运行同一套件并用当时的 testcase 数覆盖本节；不能假设仍是 `361`。
旧 commit `5230a5f` 的 PASS 和 protocol 1.2 的 `123 passed` 只能作为历史记录，不能授权 protocol 1.3。

## 12. Claim boundary

**突破监测：无算法突破。** 本合同只把下一次运行变得可证伪、可复算、不可事后换规则。
只有 route 的预声明 field/H1、A/B residual、cluster tail、harm 与 fixed-direction margin 全部门通过，
才允许另立 L2；即使如此也还没有 DeepONet/FNO 优势、fresh 泛化或真实 OERF 结论。
