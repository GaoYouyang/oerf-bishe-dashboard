# N2-PVGR N4 v1 正式执行中止记录

日期：2026-07-18

## 冻结版本

- 协议提交：`1761ff4eca74a904350ecd90ddfe2e83fffc47a9`
- attestation 提交：`1b11ca0`
- 正式结果目录：`demo_t16_operator/results/n2_pvgr_n4_evaluator_convergence_v1`
- 工作目录：`demo_t16_operator/results/n2_pvgr_n4_evaluator_convergence_v1_work`

## 已发生的执行

正式 runner 在 attestation 之后首次启动。它完成了以下 6 个原子 level checkpoint：

- `smooth-s1729-orientation_58-wide__stress_1` 的 H256、H512、H1024；
- `smooth-s1729-orientation_58-narrow__stress_1` 的 H256、H512、H1024。

控制台只公开了第一格的聚合状态：

`smooth-s1729-orientation_58-wide__stress_1 H1024=PASS final=PASS`

随后第二格在进入 H2048 升级判断时退出。由堆栈只能确定该格触发了至少一个 H1024 门，不能据此宣称其 H2048 会通过或失败。

## 实现缺陷

冻结 v1 runner 的循环先调用 `_cell_decision(cell, levels, ...)` 判断 `requires_h2048_escalation`，而 `_cell_decision` 在 H1024 任一门失败且 `levels` 尚无 H2048 时立即抛出：

`ValueError: H2048 escalation payload is missing`

因此 v1 runner 对“无需升级”的格可运行，但对任何真正需要升级的格都无法执行预注册的 H2048 分支。这是控制流缺陷，不是收敛门本身的机器判定。

## 完整性证据

- 日志：`logs/n2_pvgr_n4_evaluator_convergence_2026-07-18.log`
- 日志 SHA-256：`4c723c9528061b86b62fb38e153de5ab121797edfbdce2ab62582111e454a13f`
- v1 正式结果目录在退出后不存在；没有生成 `result.json`、图、CSV 或 machine decision。
- 6 个 level checkpoint 保留在本地 ignored work directory，不纳入 Git，也不复用到 N4.1。

检查点 SHA-256：

| cell / level | SHA-256 |
|---|---|
| narrow H1024 | `cb76d71b0438494388d21e989cd57dddb385010d00f1deb978ccf701826d7e0f` |
| narrow H256 | `5fec751299b459eb3b5345111a1d9122502c57c63dc16addd6a3a1e3d83e3ca7` |
| narrow H512 | `01e415d2b9c2c6eafa5fa48e3f0573820090f1e6848fb841219c420057558879` |
| wide H1024 | `e24400c2b2336f76809684a3e07546a2b6367eaa0d93b0a05532d57f4eb1a515` |
| wide H256 | `18c262ee2a1700b0e041577cb38d7267fd1bfe8e0b863c7de9b84194873b9dde` |
| wide H512 | `4c43c13171be80575ff60292220c388f763c0ba5277a1ac9302a40d5b5f0d7e8` |

## N4.1 允许的唯一修改

N4.1 必须继承 v1 的全部 16 对样本、数值路线、阈值、topology gate、停止规则和 claim boundary。唯一允许的执行修改是：先用 H256/H512/H1024 计算完整 H1024 gate bundle，再决定是否加载 H2048；加载后仍调用冻结 v1 的最终 `_cell_decision`。

N4.1 使用新的配置哈希、新的 formal work/output 目录和新的 attestation，不读取或复用上述 6 个 v1 checkpoint。已知的第一格 PASS 与第二格触发升级必须在 N4.1 协议中披露，因此 N4.1 仍是 selected mechanism audit，不是独立验证。
