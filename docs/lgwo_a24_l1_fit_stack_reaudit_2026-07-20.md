# LGWO-A24-L1 protocol 1.2 与 fit stack 第二轮独立只读审计

**审计日期：** 2026-07-20
**审计范围：** protocol 1.2、`arms`、`losses`、`fit_manifest`、`fit_cache`、`checkpoint`、checkpoint chain validator、protocol validator、fit runner contract。
**执行方式：** 独立静态阅读、只读测试、反例推演和工作树 provenance 检查。
**写入边界：** 本轮只写本文件；没有编辑实现、配置、测试或网页，没有 materialize 任何 scientific data，也没有访问 `early/route/fresh/OOD` 科学分区。
**重要声明：** 本轮没有算法突破，没有训练结果，没有真实 BOST 结果，也没有泛化或优于 DeepONet/FNO/iFNO 的证据。

## 1. 总结结论

结论为：

```text
NO-GO_FOR_SCIENTIFIC_FIT_MATERIALIZATION
NO_ALGORITHM_BREAKTHROUGH
```

当前实现已经比第一轮更接近可审计的训练基础设施：loss、五臂、缓存文件格式、checkpoint 单节点和 31 节点链 validator 都有明确的失败关闭逻辑；聚焦工程回归为 `123 passed`，protocol validator 当前报告 `165` 项断言通过，L1 route/fresh 文件扫描为 0 个命中。

但是，这些结果仍不能授权第一次科学 fit。第二轮发现的核心原因不是普通单元测试失败，而是以下四个系统性缺口：

1. **typed cache 仍不能证明数值内容是 observable-only。** proposal 的 `geometry_features` 和 `support` 仍可由调用者直接注入，当前只做 shape、finite 和部分结构检查，不能排除把 truth 或 case information 编码进数值通道。
2. **checkpoint 的 model schema 与 optimizer state 仍可内部自洽但不是注册模型的真实状态。** 参数总数和自报 schema hash 通过，不等于 exact arm architecture 已被验证。
3. **metric-history 与 provenance 仍是内部自洽链，不是外部不可改写证据。** 可以在不改变外部 contract 的情况下重写整条链并重新计算 parent/history/hash；source commit 也只做格式检查，未验证为当前 Git 对象或当前 protocol 1.2。
4. **成本账本锁定了 batched API 计数，却没有在 checkpoint/chain 中同时锁定 case-equivalent 计数、batch contract 和实际 operator call mode。** 论文表格若只读取当前字段，容易把 `192 F/epoch` 与 `576 case-equivalent F/epoch` 混成同一种成本。

因此当前应继续停留在 **pre-fit implementation contract**，不得生成 fit case、启动训练、生成 early-stop 结果，或把工程门证据写成算法性能证据。

## 2. 可复核基线

### 2.1 已执行或观察到的证据

| 项目 | 当前证据 | 审计解释 |
| --- | --- | --- |
| 聚焦工程回归 | `123 passed` | 说明现有测试覆盖的工程路径成立，不等于真实 runner 已经存在或科学数据无泄漏 |
| protocol validator | `165` 项断言通过，canonical config SHA 为 `74ade98f298d6f03b54a76fae9de7423d48d6a3172c1e90bd71e5d00894cdeae` | 说明当前 JSON 与 validator 的锁定值一致；不能单独证明 source commit、runner 和缓存内容已外部封存 |
| dedicated L1 route/fresh scan | 0 个匹配路径 | 仅覆盖 validator 定义的 `demo_t16_operator/results` 范围，不能替代未来 runner 的写入审计 |
| scientific partition | 未 materialize | 这是正确的安全状态，本轮没有改变它 |
| 当前 Git 状态 | protocol 1.2、核心 fit stack 文件和若干测试仍有未提交改动 | source commit 还不能作为第一次 fit 的外部锁定根 |

### 2.2 代码和合同的边界

`docs/lgwo_a24_l1_fit_runner_contract_2026-07-20.md` 是一份较完整的执行合同，但当前仓库中没有一个真正把 manifest、三类 cache、A/B operator call ledger、loss、optimizer、checkpoint 和 resume 串起来的 LGWO-A24-L1 scientific fit runner。现有 `site_tools/test_lgwo_a24_l1_fit_stack_integration.py` 是小尺寸 engineering tensor 的连接测试，文件开头明确说明不导入 case builder、不生成 scientific partition。

这意味着目前可以审计“各个接口如何失败”，不能审计“真实训练时是否真的通过这些接口”。这一区分决定了本轮是 NO-GO。

## 3. 第一轮 P1-01..07 逐项复核

状态定义：

- **CLOSED_UNIT：** 组件级合同和负向测试足以支持该局部结论，但还不代表真实 runner 系统级关闭。
- **PARTIAL：** 主要修复存在，但仍有可构造反例或缺外部证据。
- **OPEN:** 当前不能安全授权科学 fit。

| 原问题 | 当前状态 | 第二轮判定 | 关键证据 |
| --- | --- | --- | --- |
| P1-01 typed cache / descriptor 泄漏 | 分离根、NPY schema、hash、symlink、pickle 和 immutable artifact 已实现 | **OPEN** | 见第 4 节 |
| P1-02 manifest 覆盖、closed-path 绕过和残留物 | root component 检查和目标覆盖拒绝已实现 | **PARTIAL** | 未拒绝已有 `.tmp` 或非目标文件，见第 5 节 |
| P1-03 manifest 顶层自洽篡改 | 完整 expected manifest equality 已加入 | **PARTIAL** | expected manifest 使用 manifest 自报的 source/config，仍无外部期望根 |
| P1-04 空 residual / projection | loss 对每个 residual group、A delta/Ag 做 shape、batch、nonempty 检查 | **CLOSED_UNIT / RUNNER_OPEN** | 见第 6 节 |
| P1-05 checkpoint metric history、cost ledger、sealed 状态 | typed epoch schema、31 节点、parent/history link、batched cumulative cost 已实现 | **PARTIAL** | 历史链没有外部 attestation，且没有 case-equivalent 账本，见第 7 节 |
| P1-06 checkpoint 未绑定真实 arm state | arm registry、参数量、state schema digest 字段已加入 | **OPEN** | digest 可由任意同参数量 mapping 自行计算，未与 registry 构造出的 exact state schema 比较，见第 8 节 |
| P1-07 缺外部锁定根 | chain validator 接收 external expected contract | **OPEN** | external contract 本身没有由 protocol validator、Git object 和独立 checksum evidence 绑定；工作树也未提交，见第 9 节 |

第一轮报告中写成“已关闭”的 P1-02、P1-05、P1-06，本轮均不接受为系统级完全关闭；这不是否定已经完成的代码，而是把局部实现证据与科学授权门分开。

## 4. P1-01 复核：typed cache 仍有数值内容泄漏路径

### 4.1 proposal 数值数组不是由几何合同完全推导

`demo_t16_operator/lgwo_a24_l1_fit_cache.py:566-598` 的 `build_proposal_cache_arrays` 做了以下事情：

- `observation_uv` 来自 typed `JACRUInferencePayload`；
- `pooled_adjoint_field` 由 payload 内 operator 的 adjoint 计算；
- 但 `geometry_features` 直接接受调用者传入的 `Any`，只在 `:590-598` 进入通用 normalization；
- `support` 也直接接受调用者传入的 tensor，仅检查 CPU、finite、shape 和后续 binary/active 条件，没有要求它与 `a_inference.operator.support` 数值相等。

`_normalize_arrays` / `_validate_cross_array_contract`（`lgwo_a24_l1_fit_cache.py:290-340`）能够证明数组具有预期 dtype、shape、7 个 geometry channels、ray count 和 binary support，却不能证明 geometry channel 的含义来自 geometry，也不能证明其中没有编码 truth volume、family、case identity、baseline error 或 B-side信息。

因此以下反例在当前接口层面是可表达的：调用者把一个和 geometry shape 相同、finite、7-channel 的标签编码数组作为 `geometry_features` 传入；或者把另一个 support mask 传入，只要 shape 和 active voxel 数满足约束，cache writer 仍可接受。文件名和数组 hash 只能保证“写入后内容没被改”，不能保证“写入前内容没有科学泄漏”。

### 4.2 generic proposal writer 仍然可以绕过 typed builder

`write_proposal_cache`（`lgwo_a24_l1_fit_cache.py:601-623`）直接接受四个任意数组，并在 docstring 中写“scientific fit must use the typed builder”。这是一条约定，不是运行时权限边界。只要未来 runner 调用了 generic writer，`JACRUInferencePayload` 的 typed provenance 就不会参与构造。

### 4.3 关闭门

P1-01 只有在下面所有条件同时满足后才能标记 CLOSED：

1. proposal builder 内部从 frozen geometry 重建并校验 geometry features，而不是接受任意数值特征；support 必须与 typed payload/operator support 做 exact digest 或 exact equality 检查；
2. scientific runner 的 proposal 写入 API 只暴露 typed builder，generic engineering writer 从科学执行路径移除或带不可绕过的 engineering-only token；
3. 独立 validator 打开 proposal NPY 后执行 semantic allow-list，而不只是文件名、shape 和 hash 检查；
4. 增加“把 truth 编码进 geometry_features 数值”“替换 support 但保持 shape/active count”“调用 generic writer”三类负向测试。

在此之前，proposal cache leakage gate 为 **NO-GO**。

## 5. P1-02 复核：manifest materializer 的残留目录策略仍不完整

已确认的修复：

- `_reject_dedicated_closed_paths` 现在检查 root 的每个 path component，并递归检查已有子项名称；
- `materialize_fit_manifest` 要求 40-hex source commit 和 64-hex config SHA；
- config SHA 必须等于当前 protocol mapping 的 canonical hash；
- 目标 `fit_manifest.json` 已存在时 hard fail；
- 写入使用临时文件、fsync 和 atomic replace。

仍存在的缺口：

- `materialize_fit_manifest`（`lgwo_a24_l1_fit_manifest.py:437-467`）没有在 `root.mkdir(..., exist_ok=True)` 前后拒绝已有的 `.tmp` 文件；
- 没有要求 output root 在 materialization 前为空，已有非 pickle、非 closed-token 文件可以与新 manifest 共存；
- `_atomic_write_json` 对临时文件做 fsync，但没有把“root 中无残留临时文件、无未知文件”作为 materializer 的 hard gate；
- 当前测试覆盖了目标覆盖和专用 closed path，但没有证明“已有任意非目标文件 + stale temp”必然失败。

这不会直接篡改已经存在的 manifest，但会使输出目录不是一个排他的、可审计的 artifact root，后续打包或 hash 扫描可能把混入文件误认为同一实验产物。P1-02 判为 **PARTIAL**，fit 仍不能打开。

## 6. P1-04 复核：loss 局部关闭，真实支付仍未被证明

`compute_lgwo_a24_l1_loss`（`demo_t16_operator/lgwo_a24_l1_losses.py:229-262`）现在确实拒绝：

- residual pair 的空 tensor；
- A delta / A g 的空 tensor；
- batch 维不匹配；
- residual/projection ray shape 不一致；
- support 没有 active voxel；
- 非 finite、错误 dtype/device 和错误 volume shape。

因此 P1-04 在纯 loss API 层面可标记 **CLOSED_UNIT**。

但 loss API 接收的是调用者已经计算好的 residual 和 projection。它不能证明这些 tensor 是由合同规定的 A `24/24` 和 B `1/0` 调用产生，也不能证明 `a_delta_forward` 是“已支付的首方向投影减缓存 Ag”。合同文字有这项要求，当前没有真实 fit runner 和 call-counting operator wrapper 把它变成可验证不变量。因此 P1-04 的系统级状态仍为 **RUNNER_OPEN**。

## 7. P1-05 与成本口径：metric history 有链，成本却只有 batched 视角

### 7.1 已关闭的局部内容

`validate_fit_epoch_metrics`（`demo_t16_operator/lgwo_a24_l1_checkpoint.py:448-531`）和 chain validator（`site_tools/validate_lgwo_a24_l1_checkpoint_chain.py:206-256`）已经固定了每个非零 epoch：

```text
192 A forward
192 A adjoint
8 B forward
0 B adjoint
8 optimizer steps
8 cluster batches
24 cases
0 fallback
0 nonfinite
```

31 节点 chain validator 还检查 parent manifest hash、parent metric-history hash、累计 batched cost 和 route/early seal。这些是有价值的工程进展。

### 7.2 metric-history 的残余漏洞

checkpoint manifest 没有自身的独立 `manifest_sha256` 字段；`load_checkpoint`（`lgwo_a24_l1_checkpoint.py:666-762`）主要验证内部字段和 tensor/hash，chain validator 再计算当前文件的 digest。若攻击者同时重写 epoch 0..30 的 metrics、metrics hash、history hash、parent links 和 state manifest，且 external contract 不变，当前 chain validator 仍可能把它看成一条新的自洽链。它没有读取一份独立的、事先封存的每节点 manifest/state checksum evidence。

因此 metric-history hash 是完整性链，不是不可重写的 provenance attestation。P1-05 判为 **PARTIAL**。

### 7.3 batched 与 case-equivalent 成本混淆

合同 `docs/lgwo_a24_l1_fit_runner_contract_2026-07-20.md:160-171` 同时写出了两种口径：

- batched API：每个 epoch `192 F + 192 A^T + 8 B-F`；
- case-equivalent：每个 epoch `576 F + 576 A^T + 24 B-F`。

但 checkpoint metric schema 和 chain validator 只存、只重算第一组 batched counts。当前字段没有：

- `operator_api_mode`；
- `a_case_equivalent_forward_calls` / `a_case_equivalent_adjoint_calls`；
- `b_case_equivalent_forward_calls`；
- 每次 batch 的 case count、内部 vectorization width 和是否真的是一次 batched operator call；
- explicit solver-call ledger 与 autograd linear map recomputation 的分离记录。

所以未来报告 `192` 时，读者无法从 checkpoint 判断它是 8 次每次 24-case 的 batch，还是 192 次单 case API 被事后分组记录。当前实现不能支持严谨的跨方法成本比较。此项新增为 **P1-RA-01 cost accounting ambiguity**，至少在论文比较前必须关闭；在 fit 开始前也应关闭，避免生成无法重解释的轨迹。

### 7.4 成本关闭门

需要一个只读 call-counting wrapper 和 typed ledger，同时记录：

```text
api_mode = batched | case_equivalent
batch_count
case_count
explicit_A_forward
explicit_A_adjoint
explicit_B_forward
autograd_linear_recompute
```

每个 epoch 和累计 ledger 都必须同时满足固定换算关系，且调用 wrapper 的实际事件数必须和 ledger 相等。没有这个证据，不能把 batched 成本写成模型总成本，也不能和 DeepONet/FNO 的样本级成本直接横向比较。

## 8. P1-06 复核：checkpoint schema 仍可用任意 mapping 自洽伪造

`model_state_schema_sha256`（`demo_t16_operator/lgwo_a24_l1_checkpoint.py:130-171`）会验证：

- arm 名字在 registry 中；
- parameter count 等于 registry 的 count；
- parameter names 是 state mapping 的子集；
- 参数总 numel 等于声明 count；
- 每个 state entry 的 name、shape、dtype、parameter/buffer kind 参与 hash。

这比只写 `parameter_count` 强很多，但它仍然是“对调用者提供的 mapping 做 schema hash”。`save_checkpoint`（`lgwo_a24_l1_checkpoint.py:560-569`）只比较这个 mapping 生成的 hash 与 contract 自报 hash；`load_checkpoint`（`:747-757`）重复同样的 self-consistency 检查。它没有：

- 构造 `build_seeded_lgwo_a24_l1_arm(arm, model_seed)` 并计算 registered model 的 exact state schema digest；
- 比较 state key、shape、buffer、dtype 与该 registered model 的 exact schema；
- 把 `model_seed` 约束到 protocol 1.2 的三个 model seeds；
- 验证 optimizer parameter ids 覆盖了全部模型参数。

因此，一个拥有正确参数总数但不同 tensor 分解、不同 state names 或缺失大部分 optimizer slots 的 mapping，可能生成一个新的 schema digest，然后被写入 contract 并通过 `save_checkpoint` 的自洽检查。P1-06 判为 **OPEN**。

另外，`_flatten_optimizer_state`（`lgwo_a24_l1_checkpoint.py:183-227`）只检查 parameter ids 不重复、slot 可序列化和 group hyperparameters；没有把 optimizer parameter ids 与 model parameter names/ordered parameter inventory 建立一一对应关系。这个问题会直接影响 resume 的真实性。

## 9. P1-07 复核：external expected contract 仍不是 protocol/code attestation

chain validator 的确把 `expected_contract` 作为函数参数，并在 `site_tools/validate_lgwo_a24_l1_checkpoint_chain.py:286-318` 对每个节点比较完整 contract。这关闭了“每个节点各自声明不同 contract”的一类错误。

但它仍没有验证：

1. `expected_contract.config_sha256` 等于 protocol validator 当前 canonical SHA；
2. `expected_contract.fit_manifest_sha256` 等于外部真实 `fit_manifest.json` 的 hash；
3. `expected_contract.source_commit` 是当前仓库中实际存在且包含全部 fit stack 文件的 Git object；
4. `normalization_sha256` 和 `envelope_sha256` 对应真实、外部封存的 artifact；
5. chain 的每个 manifest/state checksum 与外部 `checksums.sha256` 或 Merkle root 一致。

`CheckpointContract.from_mapping`（`lgwo_a24_l1_checkpoint.py:310-374`）只做 40-hex / 64-hex 格式和内部字段检查。`validate_checkpoint_chain` 的 external contract 可以是调用者自行构造的任意合法 contract。当前 protocol 1.2、validator、arms/losses/checkpoint 等核心实现也仍在工作树未提交状态，不能把当前 source commit 作为已封存实验根。

P1-07 判为 **OPEN**。必须先建立外部 contract JSON、真实 Git object 检查、fit manifest 独立 hash、normalization/envelope artifact hash 和独立 checksum/Merkle evidence，才能授权科学 fit。

## 10. 第一轮 P2-01..03 逐项复核

| 原问题 | 第二轮结论 | 说明 |
| --- | --- | --- |
| P2-01 optimizer 漂移 | **PARTIAL** | AdamW hyperparameters 做了 exact equality，one-group 也有检查；但 optimizer parameter-id 覆盖、slot dtype、与 exact model parameter inventory 的关系仍未锁定 |
| P2-02 fixed-direction 坐标漂移 | **OPEN** | `FixedDirectionProposal.forward` 在 `arms.py:377-396` 每次根据当前 support 的 active indices 生成 mapping；只保证同一 batch 内 index map 一致，没有跨 cluster/rig 的冻结 canonical index map digest |
| P2-03 arm 容量不等 | **OPEN_AS_INTERPRETATION_RISK** | 2729/2585/2225/681/1000 参数仍是不等容量输入/能力消融；没有参数匹配对照，不能把 full 与其余 arm 的差异写成纯算法优势 |

P2-02 的具体风险是：同样的 `direction[j]` 在不同 geometry cluster 的 support 布局下可能散射到不同 voxel。当前实现得到的是“support-relative fixed direction”，不是全局坐标固定方向。若这正是想要的 baseline，论文必须改名并预注册其含义；若要称 fixed-direction，需要冻结跨 cluster 的 canonical index map。

## 11. fit runner contract 审计

当前 contract 对执行顺序、loss 公式、24 fit entries、normalization、calibration envelope、五个 arms、30 epochs、三个 seeds、checkpoint chain 和 route seal 写得较完整，尤其是 `docs/lgwo_a24_l1_fit_runner_contract_2026-07-20.md:1-184` 的规则可作为后续实现输入。

但是“合同写了”不等于“runner 已实现”。当前没有可执行对象验证以下关键不变量：

- 每一个 fit entry 三类 cache 是否按同一外部 provenance 读取；
- normalization 是否真的只读取 24 个 fit-A observable；
- geometry features/support 是否由 geometry 重建，而不是来自 caller 数值注入；
- A `24/24` 和 B `1/0` 是真实 wrapper 事件，而不是 metrics 手填；
- B evaluator 是否一次 batch 调用而不是隐式逐 case 循环；
- `a_delta_forward` 是否复用已支付的 `A g` 和首方向投影；
- loss reduction 是否严格为 cluster 内三 family mean，再进入 optimizer；
- 每个 arm/seed 是否完整 31-node checkpoint，不选 best seed/arm；
- checkpoint 的 optimizer slots 是否覆盖 exact model parameters；
- batched 与 case-equivalent cost 是否同时写入；
- 任何 failure 是否停在 fit 层而不 materialize early/route/fresh。

因此 fit runner contract 当前只能评为 **SPECIFICATION_READY / IMPLEMENTATION_NOT_PRESENT / SCIENTIFIC_NO-GO**。

## 12. 配置与代码漂移审计

### 已发现的正面部分

- protocol validator 的 canonical hash 已更新到当前 1.2 JSON，且当前 JSON 自洽；
- 1.2 amendment history 记录了 1.1 superseded hash，并记录 amendment 前 scientific partition cases 为 0；
- protocol 中写入了五臂参数数、loss contract、cache schema、checkpoint schema 和工程门字段。

### 仍未完成的封存

- `git status` 显示 protocol 1.2、validator 改动、`arms.py`、`losses.py`、`fit_manifest.py`、`fit_cache.py`、`checkpoint.py` 及其测试仍有未提交改动；
- `origin/main` 仍停在早于这批完整 fit stack 的提交，不能把当前未提交内容描述成远端已冻结实现；
- protocol validator 只知道固定 canonical config SHA，不会从 Git 验证 source commit 是否包含实现和测试；
- fit manifest 的 direct validator 没有接收外部 expected source/config hash，且 `validate_fit_manifest` 在 `fit_manifest.py:402-408` 用 manifest 自报的 hash 重新构建 expected manifest；
- checkpoint chain validator 也没有直接读取 protocol JSON、fit manifest 和 Git object 做交叉验证。

所以当前不存在可供第一次 scientific fit 使用的单一、可复核、外部封存的 `protocol + source + manifest + cache + runner` provenance bundle。

## 13. GO / NO-GO 门

### 当前结论

```text
NO-GO_FOR_SCIENTIFIC_FIT_MATERIALIZATION
NO-GO_FOR_EARLY_STOP
NO-GO_FOR_ROUTE_DEVELOPMENT
NO-GO_FOR_FRESH_OOD
NO_ALGORITHM_BREAKTHROUGH
```

### 必须完成的最小解锁集合

1. 修复 proposal semantic leakage：geometry features/support 必须从 typed frozen geometry 派生或 exact-checked；scientific path 禁止 generic writer；新增三类数值注入负向测试。
2. materializer 在写入前拒绝 stale `.tmp`、未知文件和非空 artifact root，并让独立 validator 复读完整目录。
3. 给 fit manifest 和 checkpoint chain 增加外部 expected source/config/fit hash；source commit 必须经 Git object 和关键文件列表验证。
4. checkpoint 由 exact registered arm constructor 计算 expected state schema；保存/加载都要比较 exact key/shape/dtype/buffer schema，并验证 optimizer slots 覆盖所有参数。
5. 增加 batched/case-equivalent 双账本和实际 call-counting wrapper；每个 epoch 同时记录 API mode、batch count、case count、显式 A/B calls 和 autograd recomputation。
6. 固定跨 cluster/rig 的 fixed-direction index map，或正式改名为 support-relative direction 并重新声明其比较含义。
7. 实现真实 fit runner 后，只运行工程 gate 和 fit-only；runner 必须不调用 early/route/fresh materializer。完成后重新做独立第二审，才可打开 24 个 fit cases。

## 14. 最终研究解释

本轮没有发现可以写成论文创新点的性能突破。当前真正可靠的进展是：已经把 LGWO-A24-L1 的若干危险接口显式化，并且发现了下一层更重要的审计边界。下一步的科学价值不在于尽快跑出一个数字，而在于先证明数字来自：

```text
同一 protocol
同一 exact registered arm
同一 observable-only proposal cache
同一真实 A/B call budget
同一 batched/case-equivalent accounting
同一不可重写 checkpoint provenance
```

只有这些条件成立后，fit 结果才有资格进入后续的 field relative-L2、H1、residual、cluster tail 和 fixed-direction margin 门；即使这些门全部通过，也只能先称为 frozen synthetic fit evidence，不能自动称为真实 OERF 泛化、算法突破或高水平论文结论。
