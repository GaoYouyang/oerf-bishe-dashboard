# PoolFire C v17：单持久线程修复了 RSS，但还没有修复完整部署成本

## 结论

v17 是一次真实但未完全成功的机制修复。

我没有改模型、checkpoint、物理算子、`A^T` 提升、解析 `alpha` 或 CGLS。唯一变化是把 v16.1 的“四个网络工作线程 + 每批一次临时协调线程”替换成一个持久 proposal 线程。候选仍是 `2A+2A^T`，Zero-CGLS K4 仍是 `4A+4A^T`。

18 组 fresh-process 配对实验经过独立重算后，正式判决为：

`FAIL_POST_OPEN_PERSISTENT_SERIAL_RESOURCE_GATE_V17`

`algorithm_breakthrough=false`

## 为什么做这一步

v16.1 已经证明，候选在一条额外公开 development trajectory 上保持 `101/101` 精度兼容并减少 50% 完整算子对，但资源门失败：

- CPU 时间中位数比 Zero-K4 高 23.13%；
- peak-RSS p90 高 5.81%；
- 第一次冷进程有 350.30% 配对 wall harm。

代码审计发现，旧 runtime 同时维护四份网络 scratch、四个持久工作线程，并在约 13 个 proposal batch 上反复创建和回收协调线程。因此 v17 只修这个已经定位的机制，不借机换模型或调阈值。

## 我实际实现了什么

新的 native runtime：

1. 只创建一个持久 proposal 线程；
2. `begin()` 只提交下一批观测并唤醒该线程；
3. 主线程继续处理当前批的 `A^T`、`A`、解析步长和 CGLS K1；
4. `finish()` 等待持久线程完成，不再逐批 `pthread_create/join`；
5. 网络逐样本算术顺序保持不变；
6. 四份 feature scratch 降为一份。

结果前固定了同一批大小 8、18×2 冷进程、第一次运行必须计入、wall/CPU/RSS 门和失败动作。结果出来后没有更换线程数或删除异常轮次。

## 数值是否保持

保持。

| 检查 | v17 |
|---|---:|
| 相对 v16 正式候选 field rel-L2 p90 | `9.24e-8` |
| 相对 v16 正式候选 field rel-L2 worst | `1.04e-7` |
| Zero-K4 相对 v16 正式参考 | 精确一致 |
| 候选调用账 | `202A + 202A^T` |
| 参考调用账 | `404A + 404A^T` |

因此这次比较测到的是 runtime 变化，不是偷偷换了算法。

## 资源结果

| 指标 | v17 candidate | Zero-K4 | 判决 |
|---|---:|---:|---|
| fresh-process wall 中位数 | `0.2280 s` | `0.2638 s` | 候选快 11.06%，通过 |
| 候选更快的配对比例 | `16/18` | - | 88.89%，通过 |
| peak-RSS p90 ratio | `0.99488` | `1.0` | 通过 |
| child CPU 中位 ratio | `1.11277` | `1.0` | 多 11.28%，失败 |
| 第一次配对 wall harm | `236.82%` | - | 失败 |

这说明单持久线程修复了两件事：

- v16.1 的 RSS 失败从 `1.05805` 降到 `0.99488`；
- 典型 wall 从快 22.02%但不稳定，变成仍快 11.06%，且 16/18 配对更快。

但它没有解决：

- 网络推理带来的总 CPU 开销；
- 第一次真正冷加载模型、动态库和代码页的长尾。

所以不能写“端到端稳定加速已经成立”。

## 这是不是突破

不是。

这是一个有用的工程机制修复和负结果：它证明旧内存失败主要可以通过运行时结构修复，也把剩余问题从“线程、scratch、调度混在一起”收缩为“网络计算 CPU + 冷加载长尾”。但实验只在已经消费过的一条公开 development trajectory 上完成，不是独立泛化、真实 BOST 或物理同精度证据。

## 对下一项算法实验的直接影响

继续调线程的边际价值已经很低。下一项实验应减少网络实际执行次数，而不是继续微调 allocator：

- 只在观测变化显著的关键帧计算 learned dual proposal；
- 非关键帧使用因果 hold/transport，离线 4D 模式可另测双端插值；
- 每帧仍经过精确 `A^T` 和未修改 CGLS K1；
- 必须同时比较逐帧 CNN、previous-dual hold、运动/线性传输和 Krylov recycling；
- 仍以 field/gradient/observation 尾部、harm、CPU wall、冷启动和 RSS 判决。

一级来源红队显示，关键帧推理、时间复用和 learned warm start 各自都不是新原语。可能的论文贡献只能来自 BOST 专用的完整组合、可拒绝观测门和严格成本/非劣验证，不能宣称“首次提出关键帧”。
