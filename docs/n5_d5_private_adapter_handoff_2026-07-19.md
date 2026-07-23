# N5-D5-L1 私有真实适配器接线与静态门禁

**日期：** 2026-07-19

**当前状态：** `REAL_ADAPTER_NOT_RECEIVED`、`STATIC_PREFLIGHT_IMPLEMENTED`、`L2_A_FOUNDATION_IMPLEMENTED`、`FORMAL_REPLAY_LOCKED`、`MODEL_TRAINING_LOCKED`

**本文用途：** 把实验室匿名最小 callable 安全接入本机证据链。它不是 BOST 结果、三维重建结果或算法性能结果。

## 1. 先讲结论

目前没有收到何远哲师兄的真实 renderer、匿名场向量或实验室配置，因此没有运行真实 `describe`，更没有运行 36/53-request primary 或独立 validator。已经完成的是一个**不执行私有源码**的静态预检器，以及零调用的 L2-A 回放基础。

当前可以诚实地说：

- 公开的 synthetic D5 协议已经冻结并通过独立回放；
- 私有真实接入前的 L1 静态门禁与零调用 L2-A 回放基础已有代码；L1+L2 targeted suite 共 37 项测试；
- 当前没有绿色实验室报告；
- 七类科学授权仍全部为 `false`；
- 36/53-call primary、独立 validator、decoder-chain、三维重建和模型训练仍锁定。

## 2. 为什么不能直接把真实 adapter 填进现有 D5 runner

这里存在一个真实的 public/private provenance 冲突。

1. [`n5_d5_lab_interface.placeholder.json`](../data_templates/n5_d5_lab_interface.placeholder.json) 要求真实 adapter、配置和输入放在 `private_library/`。
2. `.gitignore` 会忽略整个 `private_library/`，目的是阻止实验室源码、数据、几何、标定和 raw trace 进入 GitHub Pages。
3. 冻结的 public D5 runner 要求 schema、config 和 adapter source 都由公开 Git commit 跟踪且工作树干净。
4. 冻结 validator 还会使用 `git show <commit>:<file>` 从同一个公开 commit 重取源码。

所以真实私有 adapter 不可能同时满足“不能进入公开 Git”与“必须存在于公开 commit”。删除 `.gitignore` 不是修复：Git ignore 不是权限系统，`git add -f` 仍能误加入文件，而且私有源码一旦进入 public history 就已经泄露。

正确架构应拆成三层：

| 层 | 保存什么 | 可以公开吗 |
|---|---|---|
| Public protocol provenance | Schema、调用预算、判决规则、公开审计工具的 commit/hash | 可以 |
| Private implementation provenance | adapter、依赖闭包、权重、标定、匿名输入的本地或私有 Git commit/hash | 不可以 |
| Private result provenance | requests、responses、状态、成本、指标、manifest、验证报告 | 默认不可以 |

本轮只实现第一层到第二层之间的**静态入口检查**。独立的 private formal replay 仍要另建，不能偷偷改写已经冻结的 synthetic 证据。

## 3. 师兄最少需要提供或实现的六个函数

公开教学骨架是 [`n5_d5_private_adapter_skeleton.py`](../data_templates/n5_d5_private_adapter_skeleton.py)。把它复制到 `private_library/` 后，真实后端需要实现：

| 函数 | 师兄需要接入的真实责任 | 学生验收重点 |
|---|---|---|
| `describe_renderer()` | 输入/输出维度、单位、轴序、dtype、ray 顺序、三路径身份、状态和成本语义 | 描述重复调用是否完全一致 |
| `forward_renderer()` | `curved`、`straight`、`direct_residual` 的真实 forward | output、真实 branch、diagnostic、调用成本必须分开返回 |
| `jvp_renderer()` | 接受任意运行时 `x, v`，返回同一离散 forward 的真实 JVP | 是 field 方向还是 decoder 参数方向，不能含查表答案 |
| `vjp_renderer()` | 接受任意运行时 `x, q`，返回同一离散 forward 的真实 VJP | 返回 field 梯度还是 decoder 参数梯度，必须明确 |
| `canonical_field_vector()` | 与配置完全一致的扁平场向量或 decoder 参数 | shape、dtype、轴序、单位、hash 一致 |
| `source_review_notes()` | 路径语义、branch 来源、单位、成本和源码审阅记录 | 谁确认了什么，哪些仍未确认 |

`forward_renderer()` 还有两个不能省略的责任：

- **actual branch state**：真正改变 forward 控制流的 active ray、sample count、hard mask、occupancy pruning、termination 等；
- **diagnostic state**：domain margin、support 距离、输出分箱等只用于观察的量，不能冒充 branch。

如果实验室没有原生 `direct_residual`，必须明确写“不提供”。不能在 wrapper 中临时用两张最终 detector map 相减，再把它伪装成 residual-native primitive。

这里要特别更正旧说法：师兄需要交付的是接受审计器任意运行时 `v/q` 的 callable，不是预先计算好的“两个 Jv、一个 Jᵀq”数组。L1 只做 AST/文件静态检查，尚不能证明六个函数在运行时都存在且接受 fresh directions；这必须由后续隔离 describe/validator 观察。

## 4. 私有目录怎样放

建议只在本机建立：

```text
private_library/
└── n5_d5_anonymous_case_v1/
    ├── adapter.py
    ├── base.npy
    ├── config.json
    └── readiness-report-001.json
```

要求：

- 所有文件都必须在 `private_library/` 内；
- 不使用符号链接或硬链接；
- 不被 Git 跟踪，并且确实命中 `.gitignore`；
- 配置只写仓库相对路径，不写 `/Users/...` 或 `file://...`；
- adapter 中不写 VPN、token、password、API key 或绝对本机路径；
- 原始实验条件、rig/camera ID、标定、未发表图像和结果都不进入 Pages。

## 5. L1 静态预检怎样运行

先从占位合同复制私有配置，逐项填写并重新计算 adapter/base 的 SHA-256。然后运行：

```bash
.venv/bin/python site_tools/n5_d5_private_lab_readiness.py \
  --config private_library/n5_d5_anonymous_case_v1/config.json \
  --output private_library/n5_d5_anonymous_case_v1/readiness-report-001.json
```

报告默认拒绝覆盖旧文件，下一次审计请换新文件名。工具不会 import 或执行 adapter，也不会调用 renderer。

### 绿色状态到底表示什么

`STATIC_PRIVATE_INTAKE_READY_FORMAL_REPLAY_LOCKED` 只表示：当前私有文件在静态层面满足位置、忽略规则、哈希、Schema、数组和基础源码检查，可以进入人工 source review 与一个私有 `describe` 探针的准备工作。

它**不表示**：

- 真实 BOST 接口已经验证；
- forward 符合真实折射光学；
- JVP/VJP 正确；
- 36/53-request primary 或独立 validator 已获准；
- 三维重建、DeepONet、FNO、FFNO 或自有模型可以开训；
- 算法优于基线、具备泛化或可以投稿。

### 红色状态怎样处理

`PRIVATE_INTAKE_READINESS_BLOCKED` 表示至少一个静态 blocker 未通过。先读 `blocker_codes`，修完后生成新报告；不得使用 `--allow-uncommitted-public-sources` 绕过真实运行。

## 6. L1 现在具体检查什么

- public protocol 源码存在、由 Git 跟踪、与 HEAD 一致，并绑定公开 commit/hash；
- private config、adapter 和 base input 位于私有根目录，忽略、未跟踪、非 symlink、非 hardlink；
- 配置通过冻结 JSON Schema，明确匿名、真实材料存在、公开 trace/summary 禁止；
- 七类 claim authorization 全为 false；
- 命令使用 `{python}`，只标出一个私有入口脚本，不含 shell 元字符；
- adapter 可被 AST 解析，无教学 placeholder、明显网络 import、凭据字面量和绝对路径字面量；
- 入口脚本 hash 与配置一致；
- `.npy` 禁用 pickle 后可加载，size、dtype、finite 和 SHA-256 与配置一致；
- 三条 path semantic digest 不再是占位值；
- 报告不记录私有路径，只记录逻辑标签、字节数和 hash；报告只能写回 `private_library/`。

12 项测试覆盖公开目录、强制跟踪、哈希篡改、错误 size/dtype、NaN、placeholder、网络 import、秘密字面量、绝对路径、脏 public source、symlink、hardlink 和报告越界/覆盖。

## 7. 为什么绿灯后仍不能运行 36/53-call primary

L1 会故意留下六个 warning。L2-A 已经把其中的 provenance 输入、物理合同、依赖清单、闭世界结果政策、禁 public summary 与 CSPRNG 私有探针写成静态机制，但这仍不是运行时观察：

1. `FORMAL_REPLAY_PRIVATE_ATTESTATION_AVAILABLE`：建立公开协议 commit 与私有实现 provenance 的双重绑定；
2. `PHYSICAL_TOLERANCES_REVIEWED`：由实验室根据精度、噪声地板和离散误差审核 FD/adjoint 阈值；
3. `PRIVATE_DEPENDENCY_CLOSURE_REVIEWED`：递归 content-address Python 模块、动态库、模型权重、配置、标定和缓存；
4. `FORMAL_REPLAY_CLOSED_WORLD_MANIFEST_AVAILABLE`：结果目录的实际文件集合必须等于 manifest，拒绝未列出的额外文件；
5. `LAB_PUBLIC_SUMMARY_HARD_GUARD_AVAILABLE`：`public_summary_permitted=false` 必须变成硬写入禁令，而不是只靠目录位置；
6. `UNPREDICTABLE_PRIVATE_PROBES_AVAILABLE`：validator 在运行后私下生成未知 probe，降低针对公开固定向量查表的风险。

L2-A 还修正了调用预算。原生 direct 三路径完整授权是 `2 + 53 + 53 + 48 = 156`；没有原生 direct 的双路径是 `2 + 36 + 36 + 32 = 106`。详细推导见 [N5-D5-L2-A 私有回放基础](n5_d5_l2_private_replay_foundation_2026-07-19.md)。

L2-B 仍要补：真正不继承宿主环境的 isolated describe runner、OS 级无网络执行、独立成本 instrumentation、branch 来源审阅、签名/事件链，以及人工确认 residual 究竟在 sample/integrand 层形成还是末端 map 相减。

## 8. 拿到私有包后的最短路线

1. L1 静态预检，任何 blocker 立即停止；
2. 人工逐函数 source review，确认三路径、单位、轴序、precision、sampling/interpolation/termination；
3. L2-A 核对双 provenance、closed-world inputs/outputs、physical contract、能力分支与完整预算；
4. 实现 L2-B 隔离 runner 后，单独授权两次私有 `describe`，不触碰 forward；
5. 若有原生 direct，冻结 53-request 三路径 primary；若没有，先建立 36-request dual-path L1-v2；
6. 分别授权 primary 与 validator；三路径完整上限 156 requests，双路径完整上限 106 requests；
7. 根据真实失败分流：`FAIL_BRANCH` 研究稳定半径/边界，`FAIL_STRUCTURE` 研究 residual-native primitive，`FAIL_FD/ADJOINT` 研究可审计 JVP/VJP；
8. 真实接口义务未失败后，才接 decoder-chain 和小型 6+2 view inverse；
9. inverse 有 field-level、逐 rig tail 和成本证据后，才比较经典 PCGLS/unrolling 与 DeepONet/FNO/FFNO；
10. 训练集、验证集、风险校准集和最终 audit 必须按 rig/session 隔离。

## 9. 可直接发给何远哲师兄

> 师兄好，我已经把最小真实接入拆成零调用的 L1/L2-A、单独两次 describe、primary 和独立 validator。需要的是能接受任意运行时 `x/v/q` 的 forward/JVP/VJP callable，不是预先算好的两个 Jv 和一个 Jᵀq。能否先给我一个匿名最小包，不需要整套实验数据：4--16 条 rays、一个 field 或 decoder 向量、curved/straight/direct-residual 能力（没有原生 direct 请明确），以及真实 shape/spacing/units/axis、geometry/calibration、dtype、noise floor、sampling/interpolation/boundary/termination、branch-state 与动态 ray/sample 成本说明？adapter、输入和 raw trace 全部只留在本机 `private_library/`。每个执行阶段会另行确认，不会自动开跑，也不会把局部接口通过写成重建或算法成功。

## 10. 当前证据边界

这份接线说明与静态门禁没有使用、复制或发布任何受限论文、实验室源码、实验室数据、VPN 信息或个人凭据。当前外部依赖仍是师兄提供匿名最小接口并确认物理语义；等待期间可以继续完善 L2 私有 provenance/replay，但不能用合成数据替代真实接口宣布科学成功。
