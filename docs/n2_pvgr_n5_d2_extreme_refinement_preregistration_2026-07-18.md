# N2-PVGR N5-D2 H4096/H8192 极限加密预注册

日期：2026-07-18

## D1 之后还剩什么问题

D1 已在预注册门下排除“浮点累加顺序解释 N4.1 两个失败格”的假说：四种非 raw 累加与 raw 的差，
在两个失败格上只占 H1024-H2048 refinement 差的 `1.27e-9` 与 `5.19e-10`。因此 D2 不再尝试补偿
求和技巧，而只问：

> 对完全相同的四个 post-N4/post-D1 单元，真正把中点步数提高到 H4096 与 H8192 后，残差尾部是否继续收缩？

## 固定选择与观测量

- 精确复用 D1 的四格：两个 N4.1 最终失败格及其 wide controls；
- 从 D1 已冻结的 H2048 `paired_neumaier` 数组出发；
- 新算 H4096、H8192；每层仍为 256 条共同 Sobol rays、CPU float64；
- primary observable 是 `paired_neumaier`，同时保存 raw 与其余累加方法，只用于监测高 H 时的浮点差；
- 不重跑 D1，也不重新选择 field、rig、stress 或阈值。

## 结果前冻结的门

每一格都必须同时满足：

| 门 | 阈值 | 理由 |
|---|---:|---|
| H4096-H8192 residual relative-L2 | `6.25e-4` | 原 N4.1 `1.25e-3` 门的一半，保留一档加密余量 |
| `(H4096-H8192)/(H2048-H4096)` | `0.5` | 沿用 N4 相邻差收缩门，对应至少一阶观测收缩 |
| H8192 raw-paired 差 / final refinement 差 | `1%` | 防止高 H 下累加误差重新主导 |
| finite ray fraction | `1.0` | 不允许丢 ray |
| domain / stencil margin | `>=0` | 不允许越域 |
| direction norm error | `<=1e-12` | 沿用 N4 几何数值门 |

只有四格全部过门，机器才输出 `D2_SELECTED_TAIL_RESOLVED_AT_H8192`；任一格失败都输出
`D2_SELECTED_TAIL_NOT_RESOLVED_AT_H8192`，禁止平均掩盖。

## Richardson 与尾差的边界

D2 会报告：

1. 观测阶 `p = log2(d2048-4096 / d4096-8192)`；
2. 固定二阶 Richardson correction 指标 `d4096-8192 / 3`；
3. 按当前观测收缩比外推的 geometric-tail 指标。

这些是诊断指标，不是严格误差上界。一次观测到的收缩不能证明未来仍以同一比例收缩。

## 允许与禁止

D2 即使 resolved，也只允许另开一个预注册的 32 格 adaptive-reference reconciliation，把 N4.1 已通过的
30 格与 D2 的 2 个失败格汇总。D2 本身不授权 fresh reference、field JVP/VJP、三维重建、神经算子训练、
真实数据、泛化或论文主张。
