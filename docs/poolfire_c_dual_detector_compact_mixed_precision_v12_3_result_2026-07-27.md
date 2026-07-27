# v12.3 五轨迹 FP32 诊断：精度与 wall 全过，但 RSS 只有 3/5，正式停止

## 一句话结论

v12.2 在一条已经烧掉的 `p45-s03` development proxy 上得到 `12.64%` 常驻
wall 优势，但峰值 RSS 高 `15.71%`。v12.3 没有重训模型，也没有改变 exact
`A^T`、observable alpha 或 strict CGLS K1；唯一变化是把 10,548 参数 proposal
网络的输入、权重、激活和输出改为 float32，网络输出立即转回 float64。

五条已经开放的 fit trajectory 分别运行 5 个 session；每个 session 预热一条完整
101 帧轨迹，再为 Candidate 和 Zero-K4 各计时 17 条完整轨迹。每个臂每条轨迹有
85 个 measured pass，五条合计 425 个：

```text
五条 compatibility: 100%
五条 harm:          0
五条 severe:        0
五条 wall reduction: 12.62% - 14.93%  (全部通过 >=10%)

RSS ratio:
p14 = 1.0042  PASS
p22 = 1.0433  PASS
p33 = 1.0696  FAIL
p45 = 1.0262  PASS
p58 = 1.0668  FAIL
```

冻结合同要求五条全部通过，不允许用平均收益覆盖某条失败。因此正式判决是：

```text
FAIL_FIT_ONLY_FP32_PROPOSAL_ALL_GATES
formal row pass count = 3 / 5
postopen p45 replay authorized = false
algorithm_breakthrough=false
```

## 为什么做这个实验

v12.2 已经证明调用减半不是假账：常驻 Candidate 的 proposal 约 49 ms，
`2A+2A^T` solver 约 63 ms，而 Zero-K4 的 `4A+4A^T` solver 约 129 ms。
问题只剩下网络进程的额外内存。

模型只有 10,548 个参数，float64 权重本身不到 0.1 MB，因此把模型缩得更小未必有用。
FP32 是一个最小、可证伪的诊断：

```text
如果 RSS 主要来自权重和激活，FP32 应降低峰值；
如果 RSS 主要来自运行时、缓存或测量协议，FP32 不会稳定过门。
```

结果更接近第二种情况。Candidate 峰值基本稳定在约 411-416 MB，但 Zero-K4 进程在
不同轨迹的峰值约 389-409 MB，因此 `p33` 和 `p58` 仍超过 1.05。

## 精度有没有被 FP32 破坏

没有观察到材料性破坏。与原 float64 proposal 比较：

```text
proposal worst relative-L2 <= 2.91e-7
final field worst relative-L2 <= 1.75e-7
冻结 parity 门              = 5.00e-5
```

五条轨迹的最终场相对 Zero-K4 都达到 101/101 joint match、0 harm、0 severe。
这说明本次失败不是数值精度失败，而是内存合同失败。

独立 validator 不导入正式 runner 或 worker，重新实现了 Dual-K1、Zero-K4、
field/gradient/observation compatibility、资源统计、逐帧调用 receipt 和最终判决：

```text
candidate field replay max difference = 0
reference field replay max difference = 0
scientific numeric max difference      = 0
```

网络前向实现仍共享冻结模型定义，因此报告明确写
`network_forward_implementation_shared=true`，不冒充完全独立网络实现。

## 逐轨迹结果

| Fit trajectory | Candidate ms | Zero-K4 ms | wall reduction | RSS ratio | 正式行判决 |
|---|---:|---:|---:|---:|---|
| p14-s05 | 110.26 | 128.23 | 14.02% | 1.0042 | PASS |
| p22-s03 | 110.61 | 127.16 | 13.02% | 1.0433 | PASS |
| p33-s01 | 110.35 | 126.28 | 12.62% | 1.0696 | FAIL RSS |
| p45-s05 | 110.23 | 128.13 | 13.97% | 1.0262 | PASS |
| p58-s03 | 108.57 | 127.62 | 14.93% | 1.0668 | FAIL RSS |

每条轨迹五个配对 session 都是 Candidate 更快；最弱的 session 仍快约 10.34%。
所以 wall 正结果稳定，不能被解释成少数幸运计时拉高均值。

## 红队为什么把证据再降一级

只读红队发现 0 个 P0、3 个 P1。

### 1. RSS reference 被 Torch 导入污染

Zero-K4 本来不需要神经网络，但当前同一 worker 在选择 arm 前已经导入 Torch 和候选
模块。这会抬高 Zero-K4 的 RSS，令 ratio 变小，系统性偏袒 Candidate。

这不会把本次全局失败翻成成功，反而使失败更强：

```text
当前已经在偏袒 Candidate 的口径下失败 2/5；
移除 reference 的 Torch 开销只会让 RSS ratio 更难通过。
```

因此三条形式上的 RSS PASS 不再当作干净部署证据，且不运行 p45 后续 replay。

### 2. parity 合同只冻结 p90，没有冻结 worst

协议理论上允许少数帧出现更大 FP32/FP64 差异。实际数据中 worst 只有
`1.75e-7` 到 `2.91e-7`，远低于 `5e-5`，所以这个漏洞没有改变本次结果；后续合同
仍必须同时冻结 p90 与 worst。

### 3. FP32 是受已烧掉 p45 资源结果启发的开发诊断

因此 v12.3 只能称 post-open development，不能称独立 confirmation。协议原来写的
“成功后再做 p45 confirmation”不再执行，p45 也没有在本轮被打开。

## 成功、失败与突破性进展

成功：

- 五条 fit trajectory 都保持 100% compatibility、0 harm、0 severe；
- FP32 与原 float64 最坏场差约 `1.75e-7`；
- 五条常驻 wall 都快 `12.62%-14.93%`；
- 完整调用仍从 `4A+4A^T` 降到 `2A+2A^T`；
- 独立求解器、指标、资源账和 decision 复算差为 0。

失败：

- RSS 只有形式上的 3/5，未达到五条全过；
- RSS reference 还被 Torch 导入抬高，三条形式 PASS 不能升级为干净证据；
- p45 replay、validation、untouched test 和真实 BOST 均未授权；
- 这仍不是物理实验定义的“同精度”。

突破性判断：

```text
fit_only_precision_preserved=true
fit_only_persistent_wall_headroom=true
clean_whole_pipeline_RSS_advantage=false
postopen_p45_replay_authorized=false
algorithm_breakthrough=false
```

这是一个有用但明确的负结果：继续只改浮点精度不能闭合整体资源门。下一次有效实验
必须让 Zero-K4 真正走无 Torch 的纯物理基线，并让候选 proposal 使用不会带来相同
运行时峰值的轻量后端；否则再做同一模型的 batch/精度微调只是优化被污染的口径。
