# v11.3 一次性 fresh proxy：精度与调用通过，wall / RSS 失败

## 一句话结论

冻结的 77,020 参数 detector CNN 在项目预先选定、从未参与训练的
`p=45kw_size=03` PoolFire proxy 轨迹上，101 帧全部进入 Zero-CGLS K4 的
field / gradient / observation 兼容包络，材料性 harm 为 0，并把每帧完整算子账从
`4A+4A^T` 降到 `2A+2A^T`。但是 17 次全新进程实测中，Candidate 的中位 wall
慢 6.35%，峰值 RSS p90 高 17.16%，所以正式判决是：

```text
PASS_FRESH_PROXY_ACCURACY_CALLS_RESOURCE_GATE_FAILED
algorithm_breakthrough=false
```

这不是“算法成功”，也不是“路线完全失败”。它把问题准确分成了两部分：

1. 学到的 dual proposal 在一条锁模后轨迹上保持了 K4 兼容精度，并真实减少物理算子调用；
2. 当前网络推理开销大于这个小型 straight-ray proxy 上省下的求解开销。

## 为什么这次结果可信

正式打开 fresh payload 前已经完成：

- 用五条 fit trajectory 的 505 帧训练唯一 full-fit checkpoint；
- 训练进程只接收 observation-only 镜像，不包含 `gauge_truth.npy` 或 truth 派生指标；
- 独立实现重做 505 帧 checkpoint 推理，最大输出差约 `5.6e-16`；
- 冻结模型、checkpoint、源代码、几何、两条 arm、17 次执行顺序、指标与失败动作；
- 在任何下载前写入一次性 open receipt，禁止替换 fresh 轨迹。

fresh 运行顺序是：

```text
官方文件校验与提取
-> 只生成 101 帧 observations
-> Candidate / Zero-K4 各 17 个全新进程
-> truth 打开前独立重做两条算法与 34 份逐帧调用账
-> 单次 score token
-> 构造 coarse proxy truth 并评分
-> 第二套指标实现独立复算
```

truth 打开前的数值重放对 Candidate 与 Zero-K4 都得到最大绝对差 `0.0`；最终独立评分
的全部数值叶子最大差也为 `0.0`。

## 101 帧 fresh 结果

### Candidate

| 指标 | p50 | p90 | worst |
|---|---:|---:|---:|
| field relative-L2 | 0.49014 | 0.52045 | 0.53600 |
| gradient relative-L2 | 0.90536 | 0.96876 | 1.00250 |
| observation relative-L2 | 0.24764 | 0.26913 | 0.29023 |

### Zero-CGLS K4

| 指标 | p50 | p90 | worst |
|---|---:|---:|---:|
| field relative-L2 | 0.48632 | 0.51713 | 0.53334 |
| gradient relative-L2 | 0.90170 | 0.97090 | 1.00219 |
| observation relative-L2 | 0.24701 | 0.26831 | 0.29010 |

Candidate 并不是逐项误差更小；它通过的是预先冻结的单侧兼容包络：

```text
joint matched = 101 / 101
joint harm = 0 / 101
field harm = 0 / 101
gradient harm = 0 / 101
observation harm = 0 / 101
severe harm = 0
```

因此正确说法是“以一半完整算子调用达到冻结兼容精度”，不能说“误差全面优于 K4”。

## 成本账

| 项目 | Candidate | Zero-K4 | 判决 |
|---|---:|---:|---|
| 每帧 A | 2 | 4 | 下降 50% |
| 每帧 A^T | 2 | 4 | 下降 50% |
| 17 次全新进程 wall 中位数 | 1.1232 s | 1.0562 s | 慢 6.35% |
| whole-worker peak RSS p90 | 343.82 MB | 293.47 MB | 高 17.16% |

wall 的冻结门是至少快 10%，RSS 门是不超过 reference 的 1.05 倍。两项都没有通过。

进一步的同进程分解表明：

```text
101 帧 CNN proposal 中位时间约 0.116 s
Candidate 两对 A/A^T 求解约 0.064 s
Zero-K4 四对 A/A^T 求解约 0.135 s
```

CNN 额外约 0.116 s，而两对算子只省约 0.071 s，所以在当前小 CPU proxy 上净亏。
这也给出了清晰的 break-even 条件：模型推理必须显著压到约 0.07 s 以下，或真实
BOST 的一次 A/A^T 成本必须明显高于当前 straight-ray proxy，调用减半才会转化为
wall 优势。

## 成功了什么，没成功什么

已成功：

- fit-only 五折 `5/5` 后，又在一条锁模后的 project-managed fresh proxy 上
  `101/101` compatibility pass；
- 每帧 `2A+2A^T` 的调用账由 34 个进程、逐帧 receipt 和独立 replay 共同确认；
- 没有通过牺牲 field、gradient 或 observation 尾部换取调用减少。

未成功：

- 没有 wall speedup；
- 没有 RSS no-harm；
- 只有一条 fresh proxy，不是广泛泛化；
- 它是公开 CFD density-gradient straight-ray proxy，不是真实 BOST；
- 没有达到论文完成或算法突破门。

## 下一条执行路线

不再修改 v11.3 checkpoint，也不在 `p45-s03` truth 上调权重或阈值。后续只测试预先
冻结的 compact capacity ladder：先用五条 fit trajectory 做 `w16d2` 完整 LOTO，
只有 `5/5` 才训练 full-fit 并测 observation-only runtime；失败再运行一次
`w24d3`。目标是保留同一物理链与兼容精度，同时减少 proposal 推理和激活内存。

两条 untouched test 继续封存；新模型若最终冻结，必须在 untouched trajectory 和
组内真实 BOST 上重新过门。
