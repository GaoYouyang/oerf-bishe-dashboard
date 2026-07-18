# N5-D5-L2-B 与 dual-path L1-v2：协议能演练，当前 Mac 生产路径主动拒绝

**日期：** 2026-07-19

**当前机器结论：** `L2B_DEVELOPMENT_TEST_DOUBLE_TESTED`、`PRODUCTION_BACKEND_BLOCKED_FIVE_CAPABILITIES`、`DUAL_L1_V2_STATIC_READY_IN_FIXTURE`、`REAL_ADAPTER_NOT_RECEIVED`、`PRIVATE_DESCRIBE_NOT_AUTHORIZED`、`FORMAL_REPLAY_LOCKED`、`MODEL_TRAINING_LOCKED`

## 1. 一分钟先读懂

这一轮完成了两块以前只有文字、没有机器实现的工作：

1. **L2-B describe-only runner**：两条外部 `describe`、一次性 token、限流、私有输出和恶意 fixture 已写成代码；但当前 macOS backend 不满足五项生产能力：禁止启动后 `exec` 替换、保护私有输入根、保护持久 nonce 账本根、保护输出根、外部验证 backend capability。生产入口因此在读取私有授权前 fail closed。测试只能 monkeypatch 内部 capability provider 进入明确标记的 development test double；公开调用参数里没有 bypass 开关，也不会产生 production pass。
2. **独立 dual-path L1-v2**：当实验室没有原生 `direct_residual` callable 时，只承认 curved 与 straight 两条路径，primary 固定为 36 requests；不允许把两张 detector map 最后相减后改名为第三条原生 residual。

聚合定向测试当前是：

```text
旧三路径 L1 / L2-A       39
L2-B test double / gate  18
dual-path L1-v2          24
--------------------------------
合计                     81 passed
```

这 81 项都是**协议、静态检查、恶意 fixture、host fail-closed 和预算推导测试**。仓库中没有何远哲师兄的真实 adapter，本轮没有执行实验室 renderer，没有训练网络，也没有得到 BOST 三维重建结果。

## 2. 为什么 describe 必须单独授权

以前容易把“一次接口检查”想成一条命令。严格拆开后，至少有五个不同权限：

| 权限 | 会做什么 | 当前状态 |
|---|---|---|
| L1 static | 读 Schema、源码 AST、hash 和 `.npy`，不 import | triple 与 dual 代码均已实现 |
| L2-A foundation | 绑定私有输入、物理合同、预算和输出政策，0 adapter calls | 已实现 |
| L2-B describe-only | 隔离进程中发送恰好两条外部 describe | test-double runner 已实现；当前生产 backend 因五项 capability 缺失而主动阻塞 |
| primary | 36 或 53 次局部 forward/JVP/VJP 核对 | 未授权 |
| validator | 独立基础重跑与 32/48 次秘密 challenge | 未授权 |

如果 describe 通过后自动连跑 primary，那么“两次只读设备铭牌”就不是一个真实权限边界。因此授权 Schema 把以下内容写死：

```text
operations = [describe]
external_request_count = 2
auto_chain_primary = false
internal_operation_absence_claimed = false
all scientific claim authorizations = false
```

最后一行很重要。父进程可以证明自己只发了两条外部 describe；它不能只凭 JSONL transport 证明 adapter 内部没有偷偷做 forward/JVP/VJP，因此合同明确不作这个主张。

## 3. 一次性 token 现在能防到什么范围

授权文件包含一个 256-bit `one_time_nonce`。runner 不在授权文件旁边写 `.consumed`，而是在已经打开并持有的私有账本 inode 中按 nonce 原子创建唯一记录：

```text
private_library/n5_d5_control/.l2b_nonce_ledger/<nonce>.consumed
```

在同一个已打开账本 inode 中，把授权 JSON 复制到另一路径或换输出目录仍会因 `O_EXCL` 被拒。marker 使用 `O_EXCL + O_NOFOLLOW` 相对持有的 directory FD 创建并 `fsync`，所以本次写入不会因路径在随后被替换而改投别处；请求和 attestation 只记录 nonce 的 SHA-256。

但这**没有证明全局唯一性**。同一 UID 或外部进程仍可能在打开前替换整个账本根，或让另一份进程看到另一个 inode。报告因此固定写 `global_nonce_uniqueness_proven=false`、`nonce_uniqueness_scope=CURRENT_OPEN_LEDGER_INODE_ONLY`。生产 backend 必须另行证明持久账本根受保护；当前机器为 false，并在读取授权前阻断。

当前仍没有数字签名或可信签发者身份。`human_approved=true` 表示本机的一次性意图，不等于密码学身份认证；签名 capsule 是下一层工作。

## 4. 文件为什么改成 FD snapshot

只做“先 hash 路径、稍后再打开路径”存在 TOCTOU：两次打开之间文件可能被替换。现在 authorization、plan、foundation 都通过同一个 `O_NOFOLLOW` 文件描述符完成：

```text
open FD
→ fstat(dev, inode, size, mtime)
→ bounded read
→ hash the same bytes
→ second fstat
→ compare path inode
```

L2-A 不再按 plan 路径二次读取，而是直接消费 runner 已持有的 snapshot bytes 与 stat identity；重算结束后仍重新 snapshot plan，若 inode、hash 或 bytes 改变，则在 nonce 消费前拒绝。私有 bundle staging 同样从固定 FD 读取并边复制边 hash。

输出目录由父进程相对已打开的父目录原子创建，并持有 output directory FD。三个 payload 用 `O_EXCL + O_NOFOLLOW` 相对该 FD 写入，manifest 从内存中的实际 bytes 计算；闭世界文件集合、类型、link count、size 与 hash 都在同一个 output FD 上重读验证，最后才比较 inode 并关闭 FD，不再关闭后按路径重开。

这能消除 runner 自己的“关闭 FD 后再按路径重开”竞态，却不能证明验证结束后同一 UID 的外部进程不能改写结果。因此 attestation 固定写 `output_root_external_mutation_denied=false`；生产 backend 还必须提供受保护输出根。

第四轮审计又指出，authorization 与 bundle 的末级 `O_NOFOLLOW` 不能封闭中间私有目录被同 UID 进程替换的窗口；而 Python 模块里的 capability provider 本身也不是安全边界。生产门因此再增加 `private_input_root_external_mutation_denied` 与 `backend_capability_attestation_externally_verified`，当前都为 false。完整门是：进程身份、私有输入根、持久账本根、输出根、外部 backend attestation 五项同时成立。

## 5. 沙箱究竟观察到了什么

当前 Mac 上 `/usr/bin/sandbox-exec` 存在，但 Apple 已将该设施标记为 deprecated。实际正反 fixture 观察到：

| 探针 | 当前机器观察 |
|---|---|
| 写 stage、HOME、TMP、repo | 被拒 |
| TCP connect 到 localhost | `EPERM` |
| `subprocess.run` / `os.fork` | 被拒 |
| 宿主自定义环境变量 | 子进程不可见 |
| 人为设为 inheritable 的 FD 99 | 子进程不可见 |
| 读取 sibling `/private/tmp` secret | 被拒 |
| stdout / stderr 洪泛 | 达上限后终止并 fail closed |
| 一条响应后挂起 | wall timeout 后终止并 fail closed |

但必须同时保留八个限制：

1. profile 为启动 Python 需要 `process-exec`；同一进程可以 exec 到另一个会伪造两条合法响应的程序，精确 JSON 对比不能证明进程身份未替换。因此这不是一个可接受的剩余风险：**默认生产入口在消费授权前直接拒绝**。
2. system read 使用较宽的 allow，再排除 `/Users`、`/Volumes`、`/Network`、其他 temp 等前缀；**整个读取面不是 closed-world**。
3. process-group 消失已检查，但不能从这一点证明任何可能脱离进程组的 descendant 在全系统绝对不存在。
4. macOS 的 `RLIMIT_RSS` 不能当作硬内存 kill 保证，所以 attestation 只记录 cumulative child resource observation，不声称硬内存界。
5. 当前账本原子性只在本次持有的 ledger inode 内成立，没有证明跨同 UID 路径替换后的全局 nonce 唯一性。
6. 输出在持有 FD 上完成闭世界验证，但没有证明验证后不受同 UID 外部改写。
7. 私有输入路径的边界检查与末级 `O_NOFOLLOW` 没有证明中间目录不被同 UID 进程替换。
8. development test double 通过进程内 monkeypatch 提供 capability，只是测试机制，不是 production attestation。

因此当前机器只有 development test-double 状态：

```text
L2B_DEVELOPMENT_TEST_DOUBLE_PASS_NO_SCIENCE_AUTHORIZATION
production blockers = POST_LAUNCH_EXEC_REPLACEMENT_NOT_DENIED
                    + PRIVATE_INPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED
                    + DURABLE_NONCE_LEDGER_ROOT_NOT_PROTECTED
                    + OUTPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED
                    + BACKEND_CAPABILITY_ATTESTATION_NOT_EXTERNALLY_VERIFIED
```

它不是 `SAFE_SANDBOX_PROVEN`，也不是 `REAL_BOST_INTERFACE_PASS`。默认 `run_authorized_describe` 不会为这个 backend 消费 nonce、创建结果目录或执行 adapter。

## 6. descriptor 为什么不能是任意 JSON

第一次红队发现：若 `expected_descriptor` 允许任意键，授权文件可以预先写入：

```json
{"physical_correctness_authorized": true}
```

adapter 原样回显后，外层 claims 虽然仍为 false，读者却可能被内部字段误导。现在 descriptor Schema：

- 禁止额外顶层字段；
- implementation hash 必须等于 inventory 中的 entrypoint bytes；
- capability 只能是诚实的双路径或原生 direct 三路径；
- `internal_operation_absence_proven=false`；
- `real_bost_authorized=false`；
- `physical_correctness_authorized=false`。

两次 response 还必须逐字等于授权前冻结的 canonical descriptor，并报告累计 ledger：第一次 describe count 1，第二次 count 2，forward/JVP/VJP 全为 0。

## 7. dual-path L1-v2 为什么必须新建

冻结三路径 v1 强制：

```text
curved + straight + native direct_residual
```

直接把旧 Schema 的 `minItems/maxItems=3` 改成 2，会让已经发布的 synthetic 三路径证据失去固定含义。因此新建独立版本：

```text
schema_version  = n5-d5-minimum-bost-interface-dual-2.0
protocol_variant = dual_path_no_native_direct_v2
paths            = exactly curved + straight
native_direct_residual_supported = false
```

静态 auditor 除了复用 v1 的 identity、adapter、field、observation、probe、state、tolerance、privacy 和 claims 合同，还额外检查：

- path role、path ID、callable ID、semantic digest 各自唯一；
- `precomputed_probe_arrays_are_sufficient=false`；
- callable 接受任意运行时 `x/v/q` 的合同被冻结；
- AST 中没有 `direct_residual` callable marker；
- AST 中没有 `curved endpoint - straight endpoint` 的直接或别名包装器模式；
- 两条路径成本必须精确为 36。

静态 AST heuristic 能抓住已测试的直接/别名相减模式，但不能证明任意运行时程序行为。报告因此写的是 `FORBIDDEN_AND_NOT_DETECTED_BY_STATIC_HEURISTIC`，不是“数学上证明不存在”。

## 8. 36 和 106 的每一项从哪里来

每条路径基础审计：

| 操作 | 每路径数量 |
|---|---:|
| base forward + repeat | 2 |
| 两个 JVP | 2 |
| 一个 VJP | 1 |
| 2 tangents × 3 h × two-sided FD forward | 12 |
| 合计 | 17 |

双路径 primary：

```text
2 describe + 2 paths × 17 = 36
```

分操作账本：

```text
describe = 2
forward  = 2 × (2 + 12) = 28
JVP      = 2 × 2 = 4
VJP      = 2 × 1 = 2
total    = 36
```

完整分阶段预算：

```text
describe-only         2
dual primary         36
dual validator base  36
private challenge    32
-----------------------
total               106
```

L2-A 现在能接受独立 dual-v2 L1 report，并重算出 `path_count=2`、`primary=36`、`total=106`。同时保留：

```text
formal_36_call_replay_authorized = false
formal_53_call_replay_authorized = false
formal_replay_authorized = false
```

## 9. 81 项测试真正覆盖什么

新增 L2-B 测试覆盖：

- 合法两次 describe；
- stdout/stderr flood；
- one-response hang；
- third response；
- forward label；
- duplicate key、NaN、深嵌套；
- 环境、FD、写文件、TCP、fork/subprocess；
- 默认生产入口因五项 backend capability 缺失在读取授权前拒绝；公开 API 没有 fixture/bypass/override 参数；测试只通过 monkeypatch capability provider 使用明确 test double；
- plan tamper 与运行中替换；
- 同一个已打开 ledger inode 内的 nonce 重放与复制授权重放；
- descriptor claim injection；
- 私有输出闭世界和 hash；
- L1 claims 缺键也会拒绝，不再把空字典误作“全部关闭”。

新增 dual-v2 测试覆盖：

- 合法双路径；
- 旧 v1 仍拒绝双路径；
- 第三路径、重复角色、native-direct=true；
- 从 path/tangent/cotangent/h schedule 独立推导 36，并拒绝成本漂移与 lookup-table-only；
- path/callable/digest alias；
- 直接、赋值别名、元组解包、字典下标、`np.subtract`、`operator.sub` 与 `from operator import sub as minus` endpoint subtraction；
- 包括普通函数、赋值别名、lambda、导入别名和 `setattr` 形式 `native_direct_residual*` 在内的 direct-residual marker；
- source/base tamper；
- malformed shared contract；
- dual L1 进入 L2-A 后准确得到 106。

尚未覆盖真实实验室 dependency closure、GPU runtime、动态库、真实 geometry/calibration、真实 ray sample counter、跨系统 sandbox、密码学签名、独立成本 observer 和真实 adapter 的 source review。这些不能从 fixture tests 外推。

## 10. 现在可以发给师兄的审核问题

> 师兄好，我已经把无原生 direct 的接口独立写成 dual-path v2：只承认 curved/straight，两路径 primary 为 36，完整 describe+primary+validator+private challenge 为 106，不会把末端相减包装成 residual-native。L2-B 也拆成了一次性授权的两次 describe，不会自动连跑 forward/JVP/VJP。能否请你先确认：实验室 renderer 是否确实没有 native direct；能否提供一个标准库可启动的轻量 describe entrypoint，以及接受任意运行时 x/v/q 的真实 backend callable；field/decoder shape、axis、units、geometry/calibration hash、sampling/interpolation/boundary/termination、branch/diagnostic、dtype、noise floor 和动态 ray/sample counter 分别是什么？私有源码和 trace 只留本机。当前所有 formal replay、训练和论文 claim 都仍锁定。

## 11. 下一步，不提前跨门

1. 等何远哲师兄确认 native direct 能力和 descriptor 字段。
2. 真实文件进入 `private_library/` 后，先跑 dual/triple L1；人工读 source 和 physical contract。
3. 先实现并红队验证一个同时禁止 post-launch exec replacement、保护私有输入根、持久 nonce 账本根与输出根，并由外部 attestation 验证 capability 的 backend；当前 Mac backend 不签发真实 describe 授权。
4. 新 backend 通过后再单独生成一次性 describe authorization；运行前再次确认，不自动执行。
5. 两次 descriptor 一致也只进入“是否授权 primary”的人工评审。
6. primary 与 validator 各用新 token；增加签名 capsule、事件链和独立 observed cost。
7. 只有真实局部门暴露可重复 failure mode，才据此选择 residual-native、branch-safe、decoder-chain 或 operator-learning 创新。
8. 多视角 N2 数据合同、rig/session split、field relative-L2、逐 rig tail、Schur violation 和端到端成本未到位前，不训练 DeepONet/FNO/FFNO，也不写算法胜出。

## 12. 复现命令

```bash
cd oerf-bishe-dashboard

.venv/bin/python -m pytest -q \
  site_tools/test_n5_d5_private_lab_readiness.py \
  site_tools/test_n5_d5_private_replay_foundation.py \
  site_tools/test_n5_d5_l2b_describe_runner.py \
  site_tools/test_n5_d5_private_lab_readiness_dual_v2.py
```

当前预期：`81 passed`。

不要用 placeholder 执行真实 describe。当前 macOS backend 即使具备完整私有 L1/L2-A 报告也会默认拒绝；公开 API 没有 development bypass。真实运行还必须先有经独立验证、能禁止 post-launch exec replacement、保护私有输入根/持久账本根/输出根并提供外部 capability attestation 的 backend，再具备人工批准的非零 nonce、严格 descriptor、存在且为空的私有输出父目录，并再次确认本轮只授权两次 describe。
