# N5-D5-L2-A 私有回放基础：从 53 次局部门到完整授权预算

**日期：** 2026-07-19

**公开源码提交：** `96419d7`

**当前状态：** `PRIVATE_REPLAY_FOUNDATION_IMPLEMENTED`、`ADAPTER_NOT_EXECUTED`、`FORMAL_REPLAY_LOCKED`、`MODEL_TRAINING_LOCKED`

> **后续进度（同日）：** 本文记录冻结的 L2-A 基础。其后已经独立实现 dual-path L1-v2 与 L2-B describe-only development test double，并完成聚合 `81 passed` 的静态/fixture/host-gate 测试。四轮红队确认当前 macOS backend 不满足五项生产能力：禁止 post-launch exec replacement、保护私有输入根、保护持久 nonce 账本根、保护输出根、外部验证 backend capability。因此默认生产入口会在读取授权前 fail closed；真实 adapter 仍未收到，也没有运行私有 describe。新状态、红队修复和限制见 [L2-B 与 dual-path v2 说明](n5_d5_l2b_dual_v2_mechanism_2026-07-19.md)。本文第 12 节的“尚未实现”清单保留为当时快照，不再代表最新代码状态。

## 1. 先说人话结论

以前网页把“53 次请求”写得太像真实接口验证的总成本。严格拆开后，53 只是一轮三路径基础回放自身：

- 在正式回放前，还需要一次只读的 2-request describe-only 阶段；
- primary 需要 53 requests；
- 独立 validator 不能只读 primary 的答案，还要重新跑 53 requests；
- 为降低适配器背诵公开固定向量的风险，validator 还要消费 48 个私有 challenge requests。

所以有原生 `direct_residual` 的完整计划是：

```text
2 + 53 + 53 + 48 = 156 requests
```

如果实验室只有 curved 与 straight，没有原生 direct，诚实方案不是在 wrapper 末端做一次相减，而是新建双路径协议：

```text
2 + 36 + 36 + 32 = 106 requests
```

本轮完成的是**预算、来源、物理说明、私有探针和输出边界的静态基础**。没有执行实验室 adapter，没有生成真实 L2 报告，也没有开始三维重建或模型训练。

## 2. 为什么 53 不是总成本

### 2.1 三路径 primary 的 53

公开 D5 协议使用 curved、straight、direct-residual 三条路径；每条路径使用 2 个 tangent、1 个 cotangent 和 3 个有限差分步长。

每条路径的基础请求为：

| 操作 | 数量 | 用途 |
|---|---:|---|
| base forward / repeatability | 2 | 确认基准输出与重复性 |
| JVP | 2 | 两个运行时 tangent |
| VJP | 1 | 一个运行时 cotangent |
| two-sided FD forward | `2 × 2 × 3 = 12` | 两个 tangent、三个 h、正负两侧 |

三条路径加两次 describe：

```text
primary = 2 + 3 × (2 + 2 + 1 + 12) = 53
```

这与冻结 synthetic 结果中的 `2 describe + 42 forward + 6 JVP + 3 VJP` 一致。

### 2.2 validator 为什么还要再跑一轮基础协议

validator 若只读取 primary 保存的 output、metric 和 decision，只能检查文件是否自洽；它不能发现 runner 与 adapter 串通、错误重放或结果文件被有技巧地一起改写。

因此独立 validator 至少要重新启动 adapter、重新发送基础协议，并自己重算 FD、adjoint 与 structure。三路径又是 53；双路径又是 36。这是新的执行授权，不允许 primary 成功后自动触发。

### 2.3 私有 challenge 的真实调用数

L2 计划冻结：每条路径额外使用 2 个私有 tangent、2 个私有 cotangent，并从 3 个预注册数量级区间各抽一个秘密 h。

每条路径需要：

```text
2 JVP + 2 VJP + 2 tangents × 3 h × 2-sided forward = 16 requests
```

所以：

```text
triple private challenge = 3 × 16 = 48
dual private challenge   = 2 × 16 = 32
```

少算任一正负 FD forward，都会把审计成本写得比实际更低。L2-A 现在从 config 的 path、tangent、cotangent、h 数量自动推导预算，并把它与计划中的声明逐字段比较；不一致即 fail closed。

## 3. 为什么必须有双路径分支

真实实验室后端可能只有：

```text
F_curved(x), F_straight(x)
```

而没有在同一 ray sample、interpolation query 或 aperture sample 层直接形成的：

```text
R_native(x)
```

此时在 wrapper 最后写：

```python
return F_curved(x) - F_straight(x)
```

只能得到 detector-map 末端差，不证明 residual-native primitive 存在，也无法检验“先形成小残差再累计”是否真的降低相消。

因此 L2-A 增加明确能力字段：

- `direct_residual_supported`
- `direct_residual_semantics`
- `arbitrary_runtime_directions_supported`
- `precomputed_probe_arrays_are_sufficient=false`
- `backend_dtype_preserved_and_reported`
- `dynamic_ray_sample_ledger_required`
- `fixed_zero_ray_sample_ledger_forbidden`

当前冻结 L1 Schema 只接受三路径。若师兄明确没有 native direct，L2-A 会输出 `BUILD_DUAL_PATH_L1_V2`，而不是鼓励绕过 Schema。下一步需要新建独立的 dual-path L1-v2，并保留已有 synthetic 三路径结果不变。

## 4. 师兄需要提供 callable，不是三份答案数组

“给两个 Jv、一个 Jᵀq”容易被误解成让师兄提前计算三份数组。这样 validator 无法在运行后换一个未知方向，固定向量查表也可能蒙混过关。

正确责任是：

| 后端能力 | 运行时输入 | 返回 |
|---|---|---|
| forward | 任意合同内 `x` 与 path | detector output、真实 branch、diagnostic、成本 |
| JVP | 任意合同内 `x, v` 与 path | 同一离散 forward 的 `J(x)v` |
| VJP | 任意合同内 `x, q` 与 path | 同一离散 forward 的 `J(x)^T q` |

公开合成协议中的 2 个 tangent 与 1 个 cotangent只是基础探针数；私有 validator 还会在 attestation 后生成新的 `v/q/h`。静态 support flag 不能证明真实 callable 接受任意方向，因此 L2-A 报告会保留 `ARBITRARY_RUNTIME_DIRECTIONS_OBSERVED` warning，直到 L2-B/validator 真正观察到运行时行为。

## 5. 私有 bundle 为什么要闭世界

未来一个私有输入 bundle 至少必须精确列出五种角色：

1. `config`
2. `adapter_source`
3. `base_input`
4. `environment_lock`
5. `physical_contract`

还可以显式加入标定、模型权重、native library、network profile 与 cost instrumentation。实际目录文件集合必须与 inventory 完全相等；每个文件绑定 SHA-256 与 byte count，并满足：

- 位于 `private_library/`；
- 被 Git ignore 且未被 Git 跟踪；
- 不是 symlink 或 hardlink；
- 不含凭据、绝对本机路径或 placeholder；
- plan、L1 report 与 immutable input bundle 分离。

这能阻止“计划只列 adapter.py，运行时却偷偷读取未声明权重或缓存”的最简单版本。它仍不是完整 OS sandbox：动态加载、子进程、系统库和 GPU driver 的真实依赖闭包要在 L2-B 继续实现和观察。

## 6. 物理合同至少要绑定什么

[`n5_d5_private_physical_contract.placeholder.json`](../data_templates/n5_d5_private_physical_contract.placeholder.json) 要求：

- field 或 decoder parameterization；
- tensor shape、spacing、axis order、field/observation units；
- 坐标手性；
- geometry 与 calibration SHA-256；
- 光学 wavelength policy；
- sampling、interpolation、boundary、termination rule；
- backend dtype、wire dtype 与转换政策；
- dynamic sampling 与实际 ray/sample ledger；
- native direct capability；
- decoder checkpoint SHA-256（若适用）；
- 有单位且大于零的 noise floor。

工具还会交叉检查物理合同与 L1 config 的 input dimension、parameterization、axis、units 与 wire dtype。人工 `review_digest_sha256` 必须等于这份物理合同文件的实际 hash，不能随便填一个看似合法的摘要。

但要注意：字段完整、hash 一致，只说明“某份说明被固定了”，不证明说明与真实实验装置一致。geometry/calibration、噪声和容差仍需何远哲师兄审核，真实 BOST 数据仍需独立验证。

## 7. 私有未知探针怎样生成

L2-A 的 `PrivateProbeBank` 使用系统 CSPRNG 提供至少 32 bytes entropy，然后：

1. 把 input/output dimension、tangent/cotangent 数量与 h intervals 写入派生上下文；
2. 对上下文做 SHA-256；
3. 从 seed 派生独立 tangent、cotangent 与 finite-difference-h 随机流；
4. 用 QR 得到单位正交方向；
5. 在每个预注册 h interval 内按 log-uniform 抽一个正值；
6. adapter 启动前只保存 commitment、context hash 与数量，不保存 seed、向量或 h；
7. adapter 退出后才把 seed、方向 hash 和 h 写入私有 `challenge_reveal.json`。

该机制降低“根据公开 seed/向量背答案”的风险。它不能证明整个高维 Jacobian：有限随机探针只能提供概率性检测，弱故障和未探测子空间仍可能存在。

## 8. 结果目录为什么不能写公开摘要

L2-A 将未来私有结果 payload 固定为八个文件：

```text
requests.jsonl
responses.jsonl
metrics.json
finite_difference_rows.csv
result.json
private_diagnostic.png
process_attestation.json
challenge_reveal.json
```

再加一个由 payload hash 生成的 `manifest.json`。校验器会枚举实际目录，拒绝：

- 未列出的额外文件；
- 缺失或被篡改的 payload；
- 嵌套目录；
- symlink、hardlink 或非普通文件；
- `public_summary.json`、`summary.md`、`interface_bridge.png`、`validation_report.json`；
- 覆盖已有 run 目录。

因此实验室 raw trace 不会因为 runner 的默认行为被顺手写进公开摘要。当前机制尚未增加数字签名、事件链和不可变 capsule，这些属于后续 L2-B/L2-C，而不是本轮已经完成的事实。

## 9. 从零学习时按什么顺序读

### 第 1 层：先懂三个映射

- `F(x)`：折射率场或 decoder 参数怎样变成 detector displacement；
- `J(x)v`：场沿方向 `v` 小改动时，detector 怎样变化；
- `J(x)^Tq`：detector 残差 `q` 怎样回传到 field/decoder 参数。

### 第 2 层：再懂三种核对不是一回事

- actual FD：JVP 是否对应实际 forward；
- adjoint identity：JVP/VJP 是否互为转置作用；
- path structure：direct residual 是否等于 curved-straight 的同层语义。

### 第 3 层：分清“静态说明”与“运行观察”

- hash、Schema、physical contract 是静态声明与绑定；
- isolated subprocess、fresh runtime directions、wall/CPU/RSS、真实 ray samples 是运行观察；
- 只有后者实际发生，才能关闭对应 warning。

### 第 4 层：最后才进入 inverse 与 operator learning

D5 只查一个匿名局部接口。三维重建还需要多视角几何、标定、观测、noise/mismatch、rig/session split、field-level metric 和端到端成本；算子模型还要有经典 PCGLS/unrolling、DeepONet/FNO/FFNO 等公平基线。不能从一个小接口 pass 直接跳到“训练自己的模型”。

## 10. 拿到师兄接口后的严格顺序

| 阶段 | 请求数 | 允许得出的结论 |
|---|---:|---|
| L1 static | 0 | 私有文件的静态入口未发现 blocker |
| L2-A foundation | 0 | 输入、物理说明、预算与输出政策已冻结 |
| L2-B describe-only | 2 | 两次描述在隔离环境中一致 |
| triple primary / dual primary | 53 / 36 | 一个匿名工作点的有限局部义务未失败或暴露具体 failure |
| triple validator / dual validator | 53+48 / 36+32 | 独立重跑与私有挑战未发现已覆盖的错误 |
| N2 inverse | 另行预注册 | 多视角重建在 field、tail 与成本门上如何表现 |
| model training/audit | 另行预注册 | 冻结基线、隔离 split 与 untouched audit 上的性能 |

每个阶段使用独立授权 token；前一阶段成功不会自动启动下一阶段。失败结果也要封存，不能不断换 seed 直到抽到一个通过。

## 11. 现在可以发给何远哲师兄的话

> 师兄好，我把最小真实接入进一步拆成了静态 L1、零调用的 L2-A、单独两次 describe、primary 和独立 validator。需要的是能接受任意运行时 `x/v/q` 的 forward/JVP/VJP callable，不是预先算好的两个 Jv 和一个 Jᵀq。能否先提供一个匿名 4--16 rays 小例，并确认 field/decoder 参数化、shape/spacing/axis/units、curved/straight/direct 能力、geometry/calibration、precision、noise floor、sampling/interpolation/boundary/termination、真实 branch 与动态 ray/sample 成本？如果没有原生 direct 请明确，我会建立 36-request 双路径协议，不会在 wrapper 末端相减冒充 residual-native。私有文件和 raw trace 都只留本机；每个执行阶段另行确认，不会自动开跑或把局部接口通过写成三维重建、模型胜出或论文结论。

## 12. 当前验证与尚未完成

已验证：

- L1+L2 targeted suite：`37 passed`；
- Python compile、Ruff 与 `git diff --check` 通过；
- replay-plan JSON Schema 自身有效；
- placeholder replay plan 能通过 Schema；
- 预算漂移、能力冲突、物理单位/噪声/摘要篡改、h 区间倒置、秘密/绝对路径、额外文件、symlink、hardlink、结果篡改均 fail closed。

未完成：

- 真实 adapter 与匿名 base input 尚未收到；
- dual-path L1-v2 尚未实现；
- L2-B isolated describe runner 尚未实现；
- OS 级无网络执行、非继承环境、独立成本 observer 尚未运行观察；
- manifest 签名、事件链、失败 capsule 与同 run-id 防重试尚未实现；
- 没有真实 BOST、三维重建、算法基线比较、泛化或论文结果。

这份 L2-A 的价值是把未来实验的错误授权、少算成本、私有泄漏和固定向量查表风险提前变成机器 blocker。它是研究证据基础设施，不是研究结果本身。
