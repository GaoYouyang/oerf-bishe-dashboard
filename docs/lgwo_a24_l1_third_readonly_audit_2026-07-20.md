# LGWO-A24-L1 第三轮独立只读复审

日期：2026-07-20
判定：**NO-GO / 0 scientific cases / 0 optimizer steps**

## 复审发现

| 等级 | 发现 | 为什么会影响证据 | 当前动作 |
|---|---|---|---|
| P0 → CLOSED | AdamW slot 只要求正整数，没有绑定 `step = epoch * 8` | 一次真实 step 可能被 checkpoint 元数据描述成 30 epoch / 240 steps | 写入/加载均已精确绑定，负向测试通过 |
| P0 → CLOSED | 独立 chain validator 只读 checkpoint 自报 ledger hash，不读真实 ledger | 11,760 个事件与 240 steps 可被纯元数据伪造 | 已读取 30 份真实 ledger 并逐事件复算 |
| P0 | epoch 的 step、ledger、checkpoint、anchor 跨事务 | 崩溃后会 fail-closed，但可能无法唯一恢复 | 真实接口确认后再实现最小事务协议 |
| P1 | Stage 2 没有冻结物化树的精确允许路径 | 外部 root 中增加 `route/fresh/early/ood` 风格文件仍可能进入 checksums 后通过 | 后续冻结 exact member inventory |
| P1 | “第三独立审计”身份仍由 JSON 自报 | 当前只能证明文件一致，不能证明审计者身份独立 | 正式运行时使用外部签名/人工双人确认 |
| P1 | external anchor 仍位于同一 mutable output root | checkpoint 与 anchor 可被同一位置共同替换 | 正式运行时移出 run root |
| P1 | Python/Torch/NumPy runtime 未进入授权 critical inventory | 专用 `.venv` 与旧 `deeponet_env` 的回归结果不同 | 绑定 runtime lock 与解释器指纹 |
| P2 | Stage 2 多次读取之间仍有本地 TOCTOU 窗口 | 普通误改会拒绝，但主动并发替换不是单快照验证 | 在真实训练前再做单快照读取 |

## 已确认有效的部分

- 六份 raw prefit 文件均进入 checksums；
- implementation validation 同时绑定 report 原始文件 SHA256 和规范 JSON SHA256；
- negative-gate testcase 使用完整 `classname::name`，不是只信汇总数字；
- Stage 2 consumer 会重验 summary、cache triplets、normalization 与 calibration envelope；
- autograd 控制图实测反向 case-equivalent 为 `552 A-F / 576 A-A^T / 24 B-A^T`。

最后一组数字只描述训练 epoch 内的反向算子调用，不包含 shared baseline、materialization、checkpoint I/O，
不能称为端到端总成本。

## 当前决策

两个可被元数据伪造、范围明确的 P0 已关闭：optimizer step 精确绑定，真实 ledger 全量消费；18 文件聚焦
回归为 `361 passed`。epoch 事务、外部 anchor、runtime lock 与 TOCTOU 不继续无边界扩建；先向何远哲
师兄确认真实 callable、residual 层级、JVP/VJP、标定、数据和组内主痛点。真实接口不成立时，停止
LGWO scientific fit 路线。

即使两个 P0 修复、测试全绿，也仍是工程证据，不是模型效果、三维重建、真实 BOST、泛化或论文突破。
