# N5-D5-L2-C：外部见证、事件链与成本胶囊

> 状态：`ENVELOPE / VERIFIER-CONTROLLED REGISTRY / EVENT-CHAIN VERIFIER IMPLEMENTED`
> 当前证据：21 项定向测试通过；真实外部见证 0 次；真实 adapter 0 次。
> 科学边界：不授权真实 BOST、三维重建、模型训练、算法优越性、泛化或投稿结论。

## 先说结论

L2-B 已经能说明“一个受控入口只打算发送两次 `describe`”，但当前 Mac 仍缺五项生产能力。L2-C 新解决的是下一个问题：

> 将来换到真正受保护的 Linux/实验室宿主后，怎样让 runner 以外的两个角色对宿主能力声明、运行顺序和实际协议成本签名，并让另一个 verifier 独立检查？

本轮实现了这个**验证端合同**，没有实现私钥签名服务，也没有把当前 Mac 伪装成生产宿主。公开 trust-anchor registry 当前为空，所以真实 CLI 会在读取 bundle、policy、subject 或 evidence 前返回 `NO_PRODUCTION_TRUST_ANCHOR_ENROLLED`。

验证器即使完整通过，也只返回：

```text
L2C_SIGNED_STATEMENTS_STRUCTURALLY_VERIFIED_REPLAY_AND_EVIDENCE_REVIEW_REQUIRED
```

它仍明确返回：

```text
host_capability_truth_independently_proven = false
production_backend_authorized = false
formal_replay_authorized = false
model_training_authorized = false
```

## 为什么要做这一层

当前 L2-B 的五个生产 blocker 是：

1. `POST_LAUNCH_EXEC_REPLACEMENT_NOT_DENIED`
2. `PRIVATE_INPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED`
3. `DURABLE_NONCE_LEDGER_ROOT_NOT_PROTECTED`
4. `OUTPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED`
5. `BACKEND_CAPABILITY_ATTESTATION_NOT_EXTERNALLY_VERIFIED`

前四项是宿主隔离能力，第五项是“谁来证明前四项”。如果 runner 自己写下 `all_good=true`，那只是自报。L2-C 因此要求两个不同的、由固定信任政策认可的 Ed25519 密钥角色：

| 角色 | 负责签什么 | 不能证明什么 |
|---|---|---|
| `host_capability_witness` | 四项宿主能力声明和各自证据摘要 | 不能仅靠签名证明探针真实、完备或未被攻破 |
| `run_cost_observer` | 运行事件链和 describe-only 外部成本 | 不能由两次 describe 推出三维重建效率 |

同一密钥不能同时冒充两个角色；同一个角色的重复签名也不能凑够阈值。两类 key 还使用不同的 domain 和 payload scope：host role 不替 cost 声明背书，cost role 也不形式上签署 capability truth。

但是两个不同 key 只证明 key separation。当前 verifier 固定返回：

```text
role_operational_independence_proven = false
```

因为两把私钥仍可能由同一操作者、同一 signer service 或同一可信域控制。

## 初学者先弄懂四个概念

### 1. 摘要：这批字节有没有变

SHA-256 把任意长度字节映射成固定长度摘要。文件改一个字节，摘要通常就完全不同。

它解决的是完整性定位，不回答“谁生成的”“内容是否真实”。

### 2. 签名：哪个受信任密钥认可了哪些字节

Ed25519 私钥签名，公钥验证。L2-C 不包含任何私钥或签名入口，只包含 verifier。

签名能支持的最强表述是：

> 该受信任密钥对这组规范化字节作过签名，字节签名后未被修改。

签名不能支持：

- 物理模型正确；
- adapter 没有隐藏计算；
- 见证进程没有被攻破；
- 三维重建效果好；
- 自有模型胜过 FNO、FFNO 或 DeepONet。

### 3. 事件哈希链：记录有没有被删改或重排

每个事件都保存上一个事件摘要。L2-C 固定 14 个事件：

```text
RUN_CREATED
HOST_CAPABILITY_CAPTURE_STARTED
HOST_CAPABILITY_CAPTURE_FINISHED
AUTHORIZATION_BOUND
AUTHORIZATION_CONSUMED
PROCESS_STARTED
DESCRIBE_SENT #1
DESCRIBE_RECEIVED #1
DESCRIBE_SENT #2
DESCRIBE_RECEIVED #2
PROCESS_EXITED
OUTPUT_BOUND
COST_OBSERVATION_FINALIZED
RUN_SEALED
```

删除、复制、交换事件会破坏序号、前置摘要或最终封口摘要。

但哈希链仍不能证明观察者看到了现实中的每件事。若观察者漏记了隐藏进程，链可以“完整地记录一份不完整观察”。所以生产宿主隔离和独立观察仍要人工审查。

### 4. 成本胶囊：究竟测了哪一段成本

本轮 cost capsule 的 phase 被硬编码为：

```text
DESCRIBE_ONLY_PROTOCOL_TELEMETRY
```

它只允许：

```text
describe = 2
forward = 0
jvp = 0
vjp = 0
```

ray segments、integration samples、kernel evaluations 必须为 `null`，因为 describe-only 根本没有资格声称观察到了物理算子工作量。

## 当前成本与未来论文成本不是一回事

| 层级 | 当前 L2-C 是否覆盖 | 未来论文必须补什么 |
|---|---:|---|
| 两次 describe 外部请求 | 是 | 无 |
| wall / CPU / 进程数 | 是，但仅 describe-only | 同硬件、同预热、同 cache、失败重试全计入 |
| 内存 | 允许明确写 `NOT_RELIABLY_OBSERVED` | CPU RSS 与设备显存分别独立观察 |
| `A / A^T` 调用 | 否 | 每 case、rig、session、attempt 的完整调用账本 |
| ray/sample 工作量 | 否 | active rays、ray visits、sample evaluations 与采样规则摘要 |
| 神经模型成本 | 否 | 参数量、neural forward、MAC/FLOP、精度、batch、输入输出 shape |
| 端到端重建成本 | 否 | preprocess、operator、solver、network、postprocess 和总 wall time |
| 失败率与重试 | 只允许本次 prior failure 为 0 | intended/success/failed cases、OOM/timeout/数值失败和总成本 |
| 跨 rig/session 泛化 | 否 | rig/session 级 split、逐 rig tail、field relative-L2 和失败率 |

因此，`36` 次 dual-v2 primary 调用和 `106` 次完整协议预算也不是模型速度结果。它们是接口审计预算。

## L2-C 的主体绑定

签名不是只覆盖一个 `PASS` 字段，而是同时覆盖：

```text
authorization_sha256
replay_plan_sha256
foundation_report_sha256
adapter_entrypoint_sha256
runner_source_sha256
trust_policy_sha256
challenge_commitment_sha256
output_manifest_sha256
```

其中 trust policy 摘要不能由调用者作为参数提供，也不能相信 bundle 里自带的公钥。第一次红队实际复现了“攻击者换 policy、重算 digest、用自己的两把 key 重签”仍能过旧接口的问题。现在 production API 只读取固定路径的 registry；registry 必须预先登记 policy id、epoch、digest、predicate、角色和独立审核摘要。第二次红队又指出固定路径仍可被同 UID 替换，因此当前空 registry 的完整文件 SHA-256 也被编译进 verifier。路径内容变化会先返回 `VERIFIER_CONTROLLED_REGISTRY_BYTES_CHANGED`。

当前公开 registry 明确写：

```text
production_trust_anchor_enrolled = false
active_policy_pins = []
```

因此它不是一个带测试私钥的假生产根。未来登记真实 policy 时必须同时修改 registry、编译摘要常量并经过发布审查；测试 happy path 只能调用下划线前缀的 development registry helper，返回状态也固定带 `DEVELOPMENT_REGISTRY`。这仍不是 TPM 或只读系统安装证明，所以 production authorization 继续为 false。

## 与通用 attestation 规范的关系

这一设计借鉴而不是冒充现成标准：

- [in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/blob/main/spec/README.md) 将 attestation 分为 predicate、statement、envelope 和 bundle；L2-C 同样把“被声明内容”和“签名封装”分开。
- [SLSA v1.2 artifact verification](https://slsa.dev/spec/v1.2/verifying-artifacts) 要求验证信任根、签名、subject、predicate type 和预期参数。L2-C 因此不接受 bundle 自己决定信任根，也拒绝未知字段。
- [Sigstore blob verification](https://docs.sigstore.dev/cosign/verifying/verify/) 可在未来替代本地固定密钥，提供证书和透明日志 bundle；当前没有引入联网签名或透明日志依赖。
- [Linux Landlock 官方文档](https://docs.kernel.org/userspace-api/landlock.html) 说明限制一旦施加只能继续收紧，并可继承到子进程；它是未来 Linux backend 的候选部件，不等于单独解决所有进程、账本、输出和远程证明问题。

当前自定义 predicate URI 只是项目内部合同：

```text
https://gaoyouyang.github.io/oerf-bishe-dashboard/attestations/
n5-d5-l2c-external-witness/v1
```

不得写成“已经达到 SLSA 某级”或“已经兼容 Sigstore 生产证明”。

## 已实现文件

```text
data_templates/n5_d5_l2c_trust_policy.schema.json
data_templates/n5_d5_l2c_trust_policy.placeholder.json
data_templates/n5_d5_l2c_trust_anchor_registry.schema.json
data_templates/n5_d5_l2c_trust_anchor_registry.json
data_templates/n5_d5_l2c_external_witness_bundle.schema.json
data_templates/n5_d5_l2c_external_witness_bundle.placeholder.json
site_tools/n5_d5_l2c_external_witness.py
site_tools/test_n5_d5_l2c_external_witness.py
site_tools/requirements-n5-d5.txt
```

验证器特意没有：

- `Ed25519PrivateKey`；
- signer；
- fixture/bypass/unsafe override 参数；
- 自动修改 L2-B `production_backend_capabilities()` 的路径；
- 任何将科学 claim 设为 true 的路径。

## 本地复现

安装依赖：

```bash
.venv/bin/python -m pip install -r site_tools/requirements-n5-d5.txt
```

运行定向测试：

```bash
.venv/bin/python -m pytest \
  site_tools/test_n5_d5_l2c_external_witness.py -q
```

当前结果：

```text
21 passed
```

与冻结主线合并后：

```text
L1/L2-A/L2-B/dual-v2 81 + L2-C 21 = 102 passed
focused page contracts = 69 passed
fast matrix = 226 passed
```

medium 并行层不是全绿：`2211 passed, 3 failed, 55 warnings`。三项失败仍属于冻结 N2/D4c artifact 的已知 hash/目录状态漂移；L2-B 的 macOS containment 测试因并发 `killpg` 会出现 `EPERM`，现已从四进程池移到串行队列，18 项串行通过。3 项 MPS 也另行串行通过。这里不把调度修复写成算法或安全能力成功。

真实验证命令不接受 policy pin、registry path 或 subject digest 字符串。它必须给出七个 subject 文件和私有 evidence 文件；verifier 自己 snapshot 并重算摘要。当前公开 registry 没有生产 anchor，因此即使攻击者带着自洽 policy 和签名调用 CLI，也会在读取这些私有输入前 fail closed。

## 红队已经覆盖什么

当前负向测试包括：

- 未签名篡改；
- 有权密钥重签但事件哈希没有重算；
- 事件重排；
- output manifest 错绑；
- 重复 signer key；
- 不受信任 signer；
- 调用者替换 policy 与两把 key，但固定 registry 不承认；
- subject 或 evidence 文件缺失、内容摘要不符；
- adapter subject 错绑；
- 过期和超长有效期；
- capability evidence 重复；
- describe-only 中偷偷出现 forward；
- 不可靠内存观测却填写伪精确 RSS；
- 软化关键 limitation；
- 重复 JSON key；
- NaN；
- 全零公钥；
- schema-valid placeholder 被运行时拒绝。

## 还没有解决什么

1. 没有真实独立 witness signer 服务。
2. 没有 TPM/TEE/远程证明。
3. 没有 Linux Landlock/seccomp/cgroup/只读 mount 的生产 backend。
4. 没有私有 evidence artifact 的人工审查结果。
5. 没有把签名后的 capability assertion 接入一次性生产授权。
6. 没有受保护的 replay ledger；同一结构化 bundle 可重复做离线验证，所以报告固定写 `one_time_acceptance_proven=false`。
7. 两个 key 的操作独立性没有被 TPM、独立 signer service 或不同信任域证明。
8. 没有 primary replay 的 `A/A^T`、ray/sample、失败重试和完整模型成本胶囊。
9. 没有真实 BOST 数据、重建或模型比较。

## 下一次请师兄确认的最小问题

不需要师兄先交完整代码，可以先确认：

1. 实验室真实 callable 最终运行在 macOS、普通 Linux 服务器，还是容器/调度集群？
2. 是否允许用一个独立父进程观察 request IPC、进程树和 wall/CPU/RSS？
3. 能否提供两个只含公钥的独立角色：宿主能力见证与成本观察；私钥不进入本仓库？
4. 真实 backend 能否接受任意运行时 `x/v/q`，并输出动态 ray/sample counter？
5. 每个 case 能否提供匿名 `rig_id_hash`、`session_id_hash`、geometry/calibration/sampling digest？
6. 对最终论文，组内更关心同精度下少 `A/A^T`、少 ray/sample、低 wall time，还是更低逐 rig tail？

## 对毕业设计真正有用的落点

L2-C 本身不是算法创新。它的价值是让后面真正比较 DeepONet、FNO、FFNO 和自有模型时，不能通过漏算 `A/A^T`、忽略失败重试、混用 cache 或泄漏 test rig 得到虚假优势。

真正的算法创新仍应落在：

- 同等完整物理成本下更低 field relative-L2；
- 相同误差下更少 `A/A^T` 或 ray/sample；
- 对新 rig/session 更小的 tail risk；
- 在折射、遮挡、有限视角或 calibration shift 下仍保持 fail-closed 或可校准的不确定性。

只有真实接口和 N2 inverse 数据合同到位后，才进入这些性能门。
