# BLASTNet 上的冻结 w16d2 dual warm-start：有迁移信号，但没有通过同精度门

## 一句话结论

PoolFire 上训练完成的 `w16d2` detector CNN 在 BLASTNet H2-air 开封代理上
**不是完全失效**：加入第二步 CGLS 后，它在相同 `3A+3A^T` 成本下比
Zero-K3 略微改善 field 与 observation 中位误差；但是 gradient 变差，而且仍
无法追平 `4A+4A^T` 的 Zero-K4。因此当前判决是
`FAIL_POSTOPEN_W16D2_DUAL_K2_TRANSFER_HEADROOM_V55`，
`algorithm_breakthrough=false`。

## 为什么做这两个控制

v40-v53 已经说明：依赖 Direct-field 与六个手工方向的 BLASTNet 修正路线没有
通过完整门。继续扩大同一方向搜索的价值很低。仓库中仍缺一个更直接的问题：

> PoolFire 上已经通过五折轨迹检验的 observation-space dual CNN，本身能否
> 零适配迁移到 BLASTNet？

因此先冻结模型权重、输入排布和精确 `A^T` lift，不做 BLASTNet 训练：

1. v54：`w16d2 proposal -> A^T lift -> observable alpha -> CGLS K1`，
   总成本 `2A+2A^T`；
2. v55：只延续同一 CGLS recurrence 一步，总成本 `3A+3A^T`；
3. 分别与同成本 Zero-K2 / Zero-K3 以及高一档 Zero-K4 比较；
4. 四个目标时刻都必须同时通过 field、gradient、straight observation 三项。

BLASTNet truth 在 v40 时已经打开，所以这里只能称为 post-open 机理控制，不能
称为新的外部泛化测试。

## v54：两对算子调用不够

| 方法 | 完整算子对/帧 | field 中位 | gradient 中位 | observation 中位 |
|---|---:|---:|---:|---:|
| w16d2 Dual-K1 | 2 | 0.963449 | 0.976921 | 0.800665 |
| Zero-K2 | 2 | 0.948967 | 1.020615 | 0.754943 |
| Zero-K4 | 4 | 0.887832 | 1.128520 | 0.554804 |

相对同成本 Zero-K2，w16d2 的 gradient 改善约 `4.28%`，但 field 恶化约
`1.53%`，observation 恶化约 `6.06%`。这说明模型保留了一点抑制高频误差的
正则化信号，却没有形成更好的整体重建。

## v55：三对算子调用出现小信号，但仍未过门

| 方法 | 完整算子对/帧 | field 中位 | gradient 中位 | observation 中位 |
|---|---:|---:|---:|---:|
| w16d2 Dual-K2 | 3 | 0.908949 | 1.111039 | 0.645687 |
| Zero-K3 | 3 | 0.912750 | 1.093312 | 0.653189 |
| Zero-K4 | 4 | 0.887832 | 1.128520 | 0.554804 |

相对同成本 Zero-K3：

- field 改善约 `0.42%`；
- observation 改善约 `1.15%`；
- gradient 恶化约 `1.62%`。

相对 Zero-K4：

- field 仍差约 `2.38%`；
- observation 仍差约 `16.38%`；
- gradient 反而好约 `1.55%`。

四个目标时刻都没有同时进入 Zero-K4 的 `1.01` 三指标 envelope，所以不能把
它写成“同精度少一次迭代”。它更准确的含义是：**旧 CNN 学到了一点可迁移
的谱正则化，但没有学到能跨数据族保持数据一致性的 dual proposal。**

## 独立复算

v54 与 v55 都由不导入正式 runner 的另一套实现重新完成：

- checkpoint 逐张量装载；
- detector 网络推理；
- `A/A^T` 调用账；
- dual lift 与 CGLS recurrence；
- field、gradient、observation 三项指标；
- 每个目标时刻与三条臂的聚合。

两轮独立复算的最大指标差均为 `0`。因此负判决不是浮点容差或报告脚本造成的。

## 现在应当改变什么

不应当直接把同一 detector CNN 加宽，原因是：

1. `w16d2` 已经有足够容量在 PoolFire 上通过；
2. BLASTNet 上的问题表现为数据一致性与梯度正则化之间的系统权衡；
3. 多加一步 CGLS 只能把 trade-off 向 observation 移动，仍不能同时闭合三项；
4. 更大的 FNO 如果仍直接在原 detector 坐标中预测 dual proposal，很可能只是
   更昂贵地重复同一失配。

下一条最小、可证伪的表示是：

> 用冻结几何 `A` 生成 measurement-range basis，再让一个约数百参数的
> observation-only gate 只预测该几何子空间中的 dual 修正。

先比较 geometry-only 线性投影控制，再训练极小 gate。只有它在未参与拟合的
完整轨迹上稳定保留 headroom，才值得扩展到 FNO/UNO/DeepONet。

## 声明边界

- 这是 BLASTNet 已开封数据上的 post-open 机理控制；
- 没有 BLASTNet 零适配成功；
- 没有 matched-accuracy 调用减少；
- 没有 wall time 或 whole-pipeline RSS 结论；
- 不是实验室真实 BOST；
- 不是论文成功，也不是顶刊结果；
- `algorithm_breakthrough=false`。
