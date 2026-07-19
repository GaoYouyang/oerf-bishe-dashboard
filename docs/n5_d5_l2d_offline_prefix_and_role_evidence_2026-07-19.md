# N5-D5-L2-D0：离线账本前缀、静态下界 checkpoint 与角色证据

> 状态：OFFLINE PREFIX VERIFIER IMPLEMENTED / LIVE ATOMIC CONSUME NOT IMPLEMENTED
> 当前证据：31 项定向测试通过；真实外部 ledger、monitor、gossip、adapter 调用均为 0。
> 判决边界：只能证明“从 L2-B authorization 按 schema 解析出的语义 nonce”在调用者提供的签名前缀中只出现一次；不能证明前缀对全局完整、一次性接受、日志无分叉、角色真实独立或任何 BOST 科学结论。

## 1. 为什么名字写成 D0，而不是声称 L2-D 已经完成

L2-C 已经能检查两个角色分别签了什么、14 步事件链有没有被改、subject/evidence 文件有没有换。
它仍明确返回：

```text
replay_ledger_checked=false
one_time_acceptance_proven=false
role_operational_independence_proven=false
```

本轮补的离线 verifier 把第一项推进到“调用者声明从 genesis 开始的账本前缀已重算”，但外部完整性未证明，也没有在线共享状态。相同 capsule 可以被
离线验证两次；这不会产生第二条账本记录，也不会神奇地让它变成一次性。为了避免名字比证据跑得快，
本页把当前实现称为 **L2-D0 offline prefix gate**。真正 L2-D 还要有在线原子消费与跨观察者 gossip。

## 2. 当前实现到底检查什么

工具：[`site_tools/n5_d5_l2d_replay_role_verifier.py`](../site_tools/n5_d5_l2d_replay_role_verifier.py)

1. 公开 API 不接受 registry、公钥、endpoint 或 signer 参数。
2. verifier 先读取固定 registry；当前公开 registry 没有生产 anchor，因此在读取私有 bundle 前返回
   `NO_L2D_PRODUCTION_TRUST_ANCHOR_ENROLLED`。
3. 只有被 registry 固定摘要的 policy 才能进入后续检查；policy 同时固定日志 ID、log epoch、namespace、三种角色和
   一个静态 floor checkpoint。这个 floor 只是下界，不是完整 anti-rollback 证明。
4. 从 index 0 开始读取调用者提供的前缀，每条叶子按 RFC 9162 的域分离规则计算
   `SHA256(0x00 || canonical_body)`，内部节点计算 `SHA256(0x01 || left || right)`。
5. 重新计算旧 checkpoint 与新 checkpoint 的 Merkle root，旧 checkpoint 必须等于 policy 的静态 floor。
6. L2-B authorization 必须通过原 schema；verifier 直接读出 `one_time_nonce`，计算域分离摘要，不再把任意 commitment 文件的 SHA-256 当作 nonce。
7. 调用者前缀中的所有 acceptance ID、authorization ID、authorization 摘要和“被账本声明的 nonce 摘要值”都要全前缀唯一；不只扫目标记录。只有目标记录可由当前提供的 L2-B authorization 证明摘要确实来自语义 nonce；历史记录没有附 authorization/issuer proof，因此不得称为“全部语义 nonce 已验证”。
8. sequencer、monitor A、monitor B 必须使用三把不同 Ed25519 key；三个签名共同绑定 registry/policy 摘要、log epoch、challenge、subject、checkpoint、限制和 scientific claims，monitor 还绑定前缀与角色证据。
9. bundle 有效期必须完全落在 policy 有效期内，实际 `checkpoint_time - acceptance_time` 必须不大于 checkpoint 自己签署的 MMD，而该 MMD 又不得超过 policy 上限。
10. 三份私有角色证据文件按实际 bytes 重算摘要；policy 中的 operator-domain label 与 service identity
   也必须各不相同。
11. L2-C bundle 与 report 只按实际 bytes 绑定；不解释 report 自报 `status`，不把它当作已认证 attestation，也不重跑 L2-C verifier。

## 3. 通过后仍然固定为 false 的字段

```text
online_atomic_consume_observed=false
latest_checkpoint_queried_online=false
cross_observer_gossip_performed=false
global_log_view_consistency_proven=false
log_fork_freedom_proven=false
ledger_service_atomic_first_write_proven=false
role_operational_independence_proven=false
prefix_completeness_external_to_bundle_proven=false
anti_rollback_protection_proven=false
l2c_report_authenticity_proven=false
same_uid_trust_root_replacement_excluded=false
verifier_binary_integrity_proven=false
ledger_state_machine_transitions_verified=false
one_time_acceptance_proven=false
production_backend_authorized=false
formal_replay_authorized=false
model_training_authorized=false
```

原因不是保守措辞，而是 31 项测试中已明确构造的反例：两个从同一静态 floor 分出的签名前缀可以各自通过；三把 key
可能由同一个人或同一 IAM 管理；同 UID 还可以一起替换 Python verifier、registry 和编译摘要常量；离线 JSON 也无法证明服务端执行过线性化 `compare-and-swap`。

## 4. 31 项测试覆盖什么

| 攻击或误解 | 当前结果 |
| --- | --- |
| 修改 acceptance 但不重签 | 拒绝 |
| 重建 Merkle root 并由三把合法测试 key 重签，但在另一条目录记录重用目标语义 nonce 摘要 | 拒绝 |
| 重复发生在两条非目标历史记录的自报 nonce 摘要之间 | 拒绝；但不声称已验证两条历史 raw nonce |
| 只改 authorization 的 JSON 序列化 | 文件摘要变，语义 nonce 摘要不变 |
| previous checkpoint 不等于固定 floor | 拒绝 |
| 同一 floor 延伸两个不同签名分支 | 两者可各自通过，故 `fork_freedom=false` |
| 当前 root 与调用者前缀不一致 | 拒绝 |
| prefix index 不从 0 连续增长 | 拒绝 |
| L2-C report 指向另一份 bundle | 拒绝 |
| L2-C report 打开 reconstruction claim | 拒绝 |
| 伪造 L2-C `status=ALL_GATES_PASSED` | 只绑定 bytes，仍返回 `authenticity_proven=false` |
| role evidence 文件被替换 | 拒绝 |
| 一份 evidence 同时冒充两个角色 | 拒绝 |
| policy 复用 operator domain、service identity 或公钥 | 拒绝 |
| 缺一个 monitor signature | 拒绝 |
| checkpoint 超出 freshness 窗口 | 拒绝 |
| bundle 有效期越出 policy 窗口 | 拒绝 |
| 实际 acceptance-to-checkpoint 时差超过 MMD | 拒绝 |
| checkpoint 自报 MMD=1s、policy 上限=120s、实际延迟=10s | 拒绝；不能只按 policy 放行 |
| monitor 签名跨 policy context 重放 | 拒绝 |
| enrollment review 摘要全零 | 拒绝 |
| 弱化“不能证明全局唯一”的 limitation | 拒绝 |
| 调用者替换 production registry bytes | `L2D_VERIFIER_CONTROLLED_REGISTRY_BYTES_CHANGED` |
| 空生产 registry + 不存在的私有输入路径 | 先返回 no-anchor，不读取私有输入 |
| 同一有效 bundle 离线验证两次 | 两次都保持 `one_time_acceptance_proven=false` |

测试命令：

```bash
.venv/bin/python -m pytest -q site_tools/test_n5_d5_l2d_replay_role_verifier.py
```

当前结果：`31 passed`。

## 5. 一级来源怎样约束设计

| 一级来源 | 本项目只抽取什么 | 不能外推什么 |
| --- | --- | --- |
| [in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/blob/v1.2.0/spec/README.md) | subject/predicate/envelope/bundle 分层和摘要绑定 | predicate 内容真实、nonce 只用一次 |
| [SLSA v1.2 Verifying Artifacts](https://slsa.dev/spec/v1.2/verifying-artifacts) | verifier 固定 trust root、验证预期 identity 与参数 | 平台或内部人员必然诚实 |
| [Sigstore Bundle Format](https://docs.sigstore.dev/about/bundle/) | 把签名、日志条目、checkpoint 和时间证据装进可携带 bundle | 离线 bundle 是最新全局视图 |
| [RFC 9162 §2.1.3–2.1.4](https://www.rfc-editor.org/rfc/rfc9162.html#section-2.1.3) | inclusion/consistency 与 Merkle 域分离 | inclusion proof 是非成员证明 |
| [RFC 9162 §11.3](https://www.rfc-editor.org/rfc/rfc9162.html#section-11.3) | split-view 需要客户端交换视图/gossip | 单个 signed tree head 排除分叉 |
| [transparency-dev Witness](https://github.com/transparency-dev/witness/blob/main/README.md) | witness 持久保存旧 checkpoint，只对 consistency 后的新 checkpoint 联签 | 多把 key 自动等于不同组织控制 |
| [Sigstore Rekor Monitor](https://github.com/sigstore/rekor-monitor/blob/main/README.md) | tamper-evident 日志仍需持续监控 | 通用 monitor 自动理解本项目 nonce 语义 |
| [TUF Specification](https://theupdateframework.github.io/specification/v1.0.35/) | role/threshold、版本单调、expiry、rollback/freeze 检查 | 无持久旧版本的离线客户端能识别 freeze |
| [RFC 8555 §6.5](https://www.rfc-editor.org/rfc/rfc8555.html#section-6.5) | nonce 必须由服务器签发并在使用后失效 | 客户端写一个 `used=true` 就构成一次性 |
| [RFC 9449 §11.1](https://www.rfc-editor.org/rfc/rfc9449.html#section-11.1) | 多副本严格 single-use 需要共享耐久状态 | 分散副本无需协调即可防重放 |

## 6. 真正 L2-D 还缺的状态机

```text
ISSUED --FIRST_CONSUME--> CONSUMED --FIRST_ACCEPT--> ACCEPTED
                                  \--ABORT--------> ABORTED
```

正式服务需要两阶段而不是“最后补一条签名”：

1. **启动前**：在线 authority 在共享耐久账本中原子执行 `ISSUED → CONSUMED`，唯一键至少绑定
   trust domain 与 nonce；返回的 receipt 绑定 authorization、challenge、plan、foundation、adapter 和 runner。
2. **执行后**：由 verifier 提供新鲜 challenge，执行 `CONSUMED → ACCEPTED`；最终 receipt 再绑定完整
   L2-C bundle、final event 与 output manifest。
3. **独立 witness**：持久保存旧 checkpoint，验证状态转换和 nonce 唯一性，再对新 checkpoint 联签。
4. **gossip/monitor**：至少两个独立观察者交换 checkpoint，并扫描同 namespace 内的重复 nonce。
5. **失败关闭**：服务离线、响应过期、旧 challenge、`ALREADY_ACCEPTED`、分支不连续或 epoch 无迁移证明，
   都不能降级成本地 marker。

当前 D0 没有实现这五项，因此不会清除 L2-B 的生产 blocker。

## 7. 为什么这不该成为第一次和师兄沟通的主角

师兄最先需要审核的是物理接口：真实 forward 在哪里、residual 在 ray/sample 层还是 detector map 末端形成、
是否有 hard branch、能否做任意方向 JVP/VJP，以及组内真正想解决的失败。密码学账本是拿到真实接口后的
生产执行保障，不应要求师兄第一次就提供公钥、HSM 或监控服务。

可直接发送的简版问题见 [N5-D5 师兄首次沟通单](n5_d5_advisor_first_contact_2026-07-19.md)。

## 8. 当前证据等级

- `MECHANISM-TESTED`：Schema、固定 registry、调用者前缀重算、目标 L2-B 语义 nonce、全前缀自报 nonce 摘要值去重、三角色 trust-context 签名、角色证据 bytes 与负向测试。
- `NOT OBSERVED`：在线原子 consume、共享耐久状态、日志 gossip、真实 operator separation、实验室 adapter。
- `NO SCIENTIFIC CLAIM`：没有真实 BOST、三维重建、DeepONet/FNO/FFNO 比较、泛化或论文结果。
