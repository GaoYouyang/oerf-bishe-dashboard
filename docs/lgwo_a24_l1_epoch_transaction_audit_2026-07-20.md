# LGWO-A24-L1 epoch 事务与恢复独立审计

日期：2026-07-20
状态：**P0 OPEN / scientific optimizer step 禁止**

## 一句话结论

当前 runner 是 fail-closed 的，不会把残缺运行冒充成功；但一次 epoch 的参数更新、operator ledger、
checkpoint 和 external anchor 不是一个可恢复事务。第一次科学 `optimizer.step()` 之前必须关闭这个窗口，
否则断电或进程终止可能让“已经执行了多少步”无法被磁盘证据唯一证明。

这不是算法失败，也不是算法突破。当前仍为 `0 scientific cases`、`0 optimizer steps`、`0 real BOST
results`。

## 已确认的崩溃窗口

runner 目前先执行 `optimizer.step()`，再依次发布 ledger、checkpoint 和 anchor。checkpoint 目录已使用
staging、校验、`fsync` 与原子 rename，但 ledger 和 anchor 仍是独立提交：

| 崩溃位置 | 可观察后果 | 当前判定 |
|---|---|---|
| step 后、ledger 前 | 内存中发生过更新，磁盘无法证明该步 | 必须废弃该 run |
| ledger 后、checkpoint 前 | ledger 比 checkpoint 多一个 epoch | resume fail-closed |
| checkpoint 后、anchor 前 | 新节点缺 anchor，旧节点又与 ledger 前缀不一致 | resume fail-closed |
| ledger/anchor 写到一半 | 最终路径可能已占用，不能安全自动覆盖 | 人工 NO-GO |

因此，已有测试证明的是“发现不一致后拒绝继续”，不是“任何崩溃点都能无损恢复”。

## 最小修复门

如果何远哲师兄确认真实 callable、数据和主问题值得启动 LGWO fit，再实现以下最小协议：

1. epoch 前持久化 `PREPARING` 事务，绑定 run、epoch、parent、arm 与 seed；
2. ledger、checkpoint、内部 anchor 一起写入 staging 目录，完整校验并 `fsync` 后原子 rename；
3. 向独立、不可覆盖的 anchor root 幂等发布，确认后才原子推进 `TIP`；
4. 重启时只允许从已确认 `TIP` 恢复；hash 冲突直接人工 NO-GO；
5. 在 ledger、rename、anchor、TIP 四个边界加入故障注入；
6. epoch 中途崩溃时，当前“零重试”合同下废弃整个 run 并换新 run ID。若要续跑，需另行设计逐
   cluster WAL/checkpoint，并披露 aborted 或 uncertain executions。

## 当前允许与禁止

**允许：** 继续做公开 PSU / synthetic 的教学复现、callable wrapper、Schema、JVP/VJP 与有限差分核查、
成本探针、负向测试和师兄沟通。

**禁止：** scientific partition materialization、正式 optimizer step、private challenge、N2 inverse，以及
“真实 BOST、三维重建、泛化、优于 DeepONet/FNO/NeRIF、论文结果”主张。

## 下一有效动作

先发送 [N5-D5 第一次真实接口沟通单](n5_d5_advisor_first_contact_2026-07-19.md)，确认真实 forward
callable、curved/straight residual 的形成层级、JVP/VJP、几何标定、数据 split、强基线和组内首要痛点。
在接口不成立时，不应继续扩建训练事务基础设施；在接口成立后，再按本页的最小修复门关闭 P0。
