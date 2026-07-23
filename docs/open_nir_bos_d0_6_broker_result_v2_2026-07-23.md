# D0.6 split broker v2：独立审计、阻断修复与新证据链

> 日期：2026-07-23  
> 状态：`D0_6_BROKER_INTERNAL_PROBE_PASS_RUNNER_CLOSED`  
> 优化器步数：0  
> 完整 runner dry-run：未开始  
> 突破监测：**否**

## 先说人话

审计前版本的数据拆分是确定的，但“门已经严格锁上”的证据不够。独立审计发现了两个 blocker：

1. 内部 worker 可以被绕过正式 launcher 直接调用，并把外部传入的布尔值写成 PASS；
2. `close_fds=True` 不会关闭标准输入，因此禁止文件可能通过 fd 0 被带进 worker。

这两个问题不说明审计前运行已经偷读 GT，但它们足以推翻当时过强的 OS 隔离声明。所以旧 attestation 被保留为审计前历史，不再作为当前授权证据。

v2 已修复这些问题，用新的私有目录重做真实 stage 和两次 seal。这仍然只是数据平面，不是模型或论文结果。

## 修复了什么

### 1. worker 必须在自己的进程内做负向探针

现在 worker 不再接收“已阻断”布尔值。它在自身进程内必须同时证明：

- 获准文件可读；
- 冻结 GT 负向探针返回 `EPERM/EACCES`；
- 网络连接返回 `EPERM/EACCES`；
- 实际 Seatbelt profile 的 SHA 与 launcher 绑定一致。

直接在非沙箱中调用 worker 时，禁止文件可读，负测会立即失败，不会产生封存目录。

### 2. 关闭 stdin 传递通道并盘点 FD

正式 launcher 把 stdin 固定为 `/dev/null`，并在 worker 打开数据文件前检查：

- fd 0 确实是 `/dev/null`；
- fd 3 及以上没有仍然打开的继承描述符；
- stdout/stderr 不是继承的普通文件。

定向反例把禁止文件作为 stdin 传入 Seatbelt worker，新实现在读数据之前就会拒绝。

### 3. 原子不覆盖与根目录封闭

- 目录发布改为 macOS `renamex_np(RENAME_EXCL)`，无论目标是空目录还是非空目录都不得替换；
- capability verifier 现在核对完整文件和目录节点集；
- private shard verifier 拒绝额外目录、symlink、绝对路径、`..` 和非冻结 shard 文件名；
- 外部输入的中间路径组件若是 symlink，也会 fail-closed。

## 真实 v3 seal 证据

| 项目 | 结果 |
|---|---:|
| 外部 release commit | `a385cce83d88df24ed05dccfd6fde20e124f5604` |
| 安全修复 commit | `0705adbe5102de13401dc797ad5a16c288951950` |
| broker 定向测试 | 13 / 13 |
| stage worker release-root 内容白名单 | 14 |
| worker 内部 GT/read 负测 | BLOCKED |
| worker 内部 network 负测 | BLOCKED / BLOCKED |
| stdin | `/dev/null` |
| 多余继承 FD | 0 |
| fit / dev / audit | 4,800 / 672 / 672 |
| 身份交集 | 0 |
| 身份并集 | 6,144 |
| 两次 seal | 三个 shard 均 byte-identical |
| optimizer / checkpoint | 0 / 0 |

| split | rays | canonical SHA-256 |
|---|---:|---|
| fit | 4,800 | `bced02526a9a7890be9ec03aee49f3a8f8965ccaa3e9ee3f5e17203e4364c895` |
| dev | 672 | `e8ecfd3a62c76cfe0a36f6e598263f1d0e9ab5920d13019415119a9c47f3520d` |
| audit | 672 | `506583d65c7118fff459d8b12c6239f017684658c952088e293db1aeb2d8524b` |

网页只发布数量、缩略 hash、方法和声明边界；不发布 shard、observation、ray origin/direction、GT 或 checkpoint。

## 威胁模型的准确边界

v2 能证明的是：由官方 launcher 启动的 Seatbelt worker，在指定 release root 内的常规路径内容读取被限制到冻结白名单，禁止 GT 读取和网络连接在 worker 内部真实被 OS 拒绝，且未继承额外原始 FD。

v2 不能证明：

- 父进程整个文件系统只能读 14 个文件；
- 原生 macOS 已提供 Linux mount namespace 等价隔离；
- 同 UID 管理员或人类研究者无法看到公开 Phantom；
- 数据是 fresh blind；
- 任何模型已成功。

## 为什么仍然不开始正式训练

broker 完成不等于实验 runner 完成。下一道门仍然需要：

1. 三个真实独立模型与共同 boundary envelope；
2. 唯一入口 `AuditedD05Projector`；
3. 每个成功 arm/seed 精确 260 个事件的预生成账本；
4. S1/S2 前 80 步真实 lockstep 与第 81 步 optimizer reset receipt；
5. 9 个 checkpoint 或 failure tombstone 的 terminal bundle；
6. 不读真实 audit/GT、不产生科学结论的 synthetic full dry-run。

## 最终证据边界

**已得到：** 审计后的 worker 内部负测、stdin/FD 封闭、原子不覆盖、严格节点集与路径验证、真实两次确定性 seal。

**仍未得到：** runner、synthetic full dry-run、任何 optimizer 训练、三维重建指标、参数化优势、算子学习、真实 OERF 数据、跨 field/geometry 泛化、论文成功或突破。

## 证据入口

- 公开 attestation v2：`docs/open_nir_bos_d0_6_broker_attestation_v2_2026-07-23.json`
- execution overlay：`learning_labs/open_nir_bos_d0_6_execution_semantics_overlay.json`
- broker 源码：`learning_labs/open_nir_bos_d0_6_split_broker.py`
- broker 测试：`learning_labs/test_open_nir_bos_d0_6_split_broker.py`

