# PoolFire C v10.2：数值 proposal 制品与一次性授权代码门

> 日期：2026-07-27
> 状态：same-process 数据制品与 callback-free 物理准备代码门已实现
> 训练、outer prediction、score、fresh、test：全部关闭
> 突破状态：`algorithm_breakthrough=false`

## 1. 这一轮解决什么

v10.1 已经有严格 CGLS/PCGLS，但旧机制仍把 Python 模型回调直接交给物理代码。
即使回调参数里只有 observation，闭包仍可能捕获 truth、原始 operator 或其他文件。
因此 v10.2 不传函数，只传一个固定格式的数值制品：

```text
observation y
-> 独立推理端产生 z_theta(y)
-> canonical JSON header + 2072 个 little-endian float64
-> broker 一次性验证
-> strict operator 内部 A^T z、A(A^T z) 与 alpha
-> 一次性 initializer authorization
-> 未修改 strict CGLS checkpoints=(1,)
```

冻结合同是：

`learning_labs/protocols/poolfire_c_dual_proposal_artifact_contract_v10_2.json`

合同 SHA-256：

```text
2f46a398805bc10d42692643068c4300dd074fe0f02daebf7a893b3df51da4a8
```

## 2. 制品格式

制品不用 pickle、NPY、NPZ 或任意对象反序列化。字节顺序固定为：

```text
ASCII magic
-> 4-byte big-endian header length
-> canonical ASCII JSON header
-> 2072 x little-endian float64 payload
-> no trailing bytes
```

header 精确绑定：

- request ID 与 256-bit nonce；
- observation 摘要；
- proposal payload 摘要、shape 与 dtype；
- 冻结 model schema、model checkpoint 与 model manifest 摘要；
- preprocessing、geometry 与 worker source 摘要；
- `truth_accessed=false`、`role_accessed=false`、`score_accessed=false`；
- `python_callable_serialized=false`；
- `algorithm_breakthrough=false`。

解析器拒绝额外成员、缺失成员、非 canonical JSON、错误长度、尾随字节、错误 dtype /
shape、`NaN/Inf` 和 payload 摘要漂移。proposal 还要满足冻结结构上限：

```text
||z - y||_2 <= 0.5 ||y||_2
```

零 observation 必须对应逐位精确的零 proposal。

## 3. 三个一次性门

### 3.1 Request

broker 打开 request 后保存 observation 和全部预期绑定。第一次验证尝试会原子删除
request；即使 artifact 无效，也不能拿同一 request 自适应重试。

### 3.2 Verified proposal

artifact 通过后，broker 只签发一个 verified token。第一次物理消费尝试会原子删除它；
observation 替换、错误 dtype/shape、重放和并发双消费都会失败。

### 3.3 Initializer authorization

strict operator 不再暴露接收任意 `InitializerPreparation` 的 issuer。唯一签发路径要求
exact broker 和 exact verified token，并在同一函数内完成：

```text
consume verified z
-> A^T z                         1 A^T
-> A(A^T z) and cache            1 A
-> alpha = clip(<y,Ah>/<Ah,Ah>, 0, 2)
-> x0 = alpha A^T z
-> issue one strict authorization
```

授权只允许送入 exact strict CGLS `checkpoints=(1,)`。PCGLS 和其他 depth 在消费前
拒绝，因此不会留下孤立 cache；正确 K1 仍可随后消费。成功的 K1 refinement 再支付
`1A+1A^T`，strict-wrapper 完整代码账是：

```text
preparation  1A + 1A^T
refinement   1A + 1A^T
total        2A + 2A^T
```

这叫“代码账跑通”，不叫“完整推理成本下降已经证明”。

## 4. 独立红队发现并修掉的漏洞

第一轮独立审计发现一个 P1：早期版本虽然拒绝 plain forged initializer，但普通调用者
仍可直接调用内部 issuer，把手工 `1A+1A^T` 和虚假 artifact SHA 包装成授权。

修复后：

- 旧 `_issue_verified_initializer` 已删除；
- broker 消费、物理 range lift、line search、cache 和授权签发收敛为一个入口；
- 调用者不能再交入自制 `InitializerPreparation`；
- observation mismatch、额外 operator call 和数值失败都会销毁授权并清理 cache；
- verified authorization 在消费前锁死为 CGLS K1，PCGLS 旁路也被关闭。

最终窄复审在约定的同进程代码门边界内得到：

```text
P0=0
P1=0
P2=0
```

同进程 Python 反射从来不被当作安全隔离。

## 5. 已覆盖的反例

定向攻击包括：

- request、verified token 和 initializer authorization 的顺序重放；
- 三层令牌的并发双消费；
- observation、model、manifest、preprocessing、geometry、worker 替换；
- header 删除/加字段、flag 漂移、payload 单字节篡改；
- 错误 shape/dtype/length、尾随字节、非有限值和越界 proposal；
- 无效第一次尝试后再提交原 artifact；
- 普通 initializer 复制 verified 状态字符串；
- 直接 issuer 绕过；
- authorization 替换 observation 或 operator；
- authorization 后额外调用 operator；
- PCGLS 与非 K1 CGLS 误用；
- 成功路径 cache 与完整 `2A+2A^T` 生命周期账。

当前本轮可复现结果：

```text
v10.2 artifact focused: 54 passed
v10.2 artifact + strict focused: 80 passed
v10.2 model / controls / contract / legacy joint: 141 passed
tracked PoolFire C suite: 775 passed
```

这些是工程测试，不是 PoolFire 重建性能结果。

## 6. 仍未证明什么

header 中的 `physical_operator_calls_inside_worker=0` 目前只是被绑定的 worker 声明，
因此回执字段明确叫
`declared_physical_operator_calls_inside_worker`。当前继续保持：

```text
worker_authenticity_proven=false
filesystem_noninterference_proven=false
capability_isolated_worker_proven=false
formal_callback_free_physics_worker_implemented=false
formal_accepted_branch_call_reduction_proven=false
model_training_authorized=false
outer_performance_opened=false
algorithm_breakthrough=false
```

`InitializerPreparation` 中的 `truth_blind_attested=true` 只是旧数据类型要求的声明，
不能进入正式论文证据。worker 是否真的拿不到 truth / `A/A^T`，wall/RSS 是否真由
父进程测量，必须等待 sibling `execve` 能力隔离后再证明。

## 7. 下一有效门

下一轮仍不训练。顺序固定为：

1. 把 inference worker 与 physics worker 拆成没有共享模型 callback 的 sibling
   `execve` 进程；
2. 父进程测量完整 inference wall、process-tree peak RSS 与 artifact I/O；
3. 为每帧生成不可重放的完整调用 receipt；
4. 实现 pre-`A^T` accept / fallback 状态机；
5. 把相同能力隔离和成本账扩展到全部便宜 controls；
6. 前置门全部通过后，才重新讨论训练与 102-output outer barrier。

即使未来正式 accepted branch 仍是 `2A+2A^T`，是否有论文结果仍取决于逐 trajectory
field / gradient / observation 非劣、harm、fresh-process wall、whole-pipeline RSS、
fresh holdout 和组内真实 BOST 迁移。
