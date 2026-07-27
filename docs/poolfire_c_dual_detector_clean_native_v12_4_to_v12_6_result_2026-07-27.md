# v12.4-v12.6 干净部署门：调用与速度成立，峰值内存连续三次失败

## 一句话结论

冻结的 10,548 参数 Compact Dual-K1 初值器，在已经开放的
`p=14kw_size=05` fit trajectory 上，始终保持：

```text
101 / 101 joint match
harm = 0
severe harm = 0
Candidate = 2A + 2A^T
Zero-K4  = 4A + 4A^T
```

把带 Torch 的旧测量口径拆掉后，三种干净执行后端都通过了 `>=10%` wall
门，但都没有通过 `Candidate / Zero-K4 peak RSS <= 1.05`：

| 版本 | Proposal 后端 | Candidate / Zero-K4 | wall 降低 | RSS ratio | 判决 |
|---|---|---:|---:|---:|---|
| v12.4 | MLX GPU，整条 101 帧 | 74.00 / 131.12 ms | 43.56% | 1.3367 | FAIL RSS |
| v12.5 | 原生 C，batch 16 | 109.96 / 130.07 ms | 15.46% | 1.0745 | FAIL RSS |
| v12.6 | 原生 C，batch 8 | 110.94 / 130.94 ms | 15.27% | 1.1395 | FAIL RSS |

因此最终状态是：

```text
FAIL_CLEAN_RUNTIME_DEPLOYMENT_RESOURCE_GATE
remaining_fit_expansion_authorized=false
fresh_or_validation_or_test_opened=false
algorithm_breakthrough=false
paper_success=false
```

这不是“模型没效果”。模型在这条开发轨迹上保住了冻结兼容性，完整算子调用减半，
并且三种干净实现都测到 wall 优势。失败的是更严格的整体部署主张：它没有同时把峰值
内存压到 Zero-K4 的 1.05 倍以内。

## 我实际做了什么

### v12.4：先去掉被 Torch 污染的 reference

v12.3 的 Zero-K4 worker 在选择 arm 之前已经导入 Torch，抬高了 reference RSS，
因此 RSS 比例偏袒 Candidate。v12.4 把两臂拆成独立新进程：

- Candidate 只加载 MLX；
- Zero-K4 只加载 NumPy 物理求解器；
- 每个 arm 运行 5 个 session；
- 每个 session 先预热 1 条 101 帧轨迹，再计时 17 条；
- 总计每臂 85 个完整轨迹 pass；
- 同时记录完整逐帧 `A/A^T` receipt 和 whole-worker `ru_maxrss`。

MLX 把 proposal median 压到 `11.82 ms`，最终总时间只有 `74.00 ms`，比
Zero-K4 快 `43.56%`。但是 Candidate RSS p90 为 `214.38 MB`，Zero-K4 为
`160.38 MB`，比例 `1.3367`。速度大幅成功，整体内存明确失败。

MLX 与冻结 Torch 前向的 proposal worst relative-L2 为 `4.87e-5`，最终场 worst
为 `2.46e-5`，都在预注册 `5e-5` parity 门内。失败原因不是精度。

### v12.5：把神经运行时替换成最小原生 C

为了判断 MLX 框架本身是不是主要内存来源，我实现了只依赖系统 `pthread` 和 C
标准库的 float32 卷积前向：

- 权重、网络结构、输入归一化和奇对称构造不变；
- 网络输出立即转回 float64；
- exact `A^T`、observable alpha、strict CGLS K1 全部不变；
- Candidate 与 reference 都不加载 Torch 或 MLX；
- C 源经过 Clang 静态分析；
- 原生前向与冻结 Torch 的 proposal worst relative-L2 为 `3.19e-7`；
- 最终场 worst relative-L2 为 `2.04e-7`。

结果把 RSS ratio 从 `1.3367` 压到 `1.0745`，同时保留 `15.46%` wall 优势。
这说明移除神经框架确实解决了大部分额外内存，但仍差最后 `3,762,586 bytes`
才达到 1.05 门。独立审计复算了 85 次计时、5 个 session、10 个 worker、
逐帧调用账和隔离状态，确认超限不是单个异常值：5 个 session 只有 1 个通过 RSS。

### v12.6：最后一次 batch 8 反证

在不读取 fit truth 或精度结果的 synthetic 101 帧筛查中，batch 8 的 proposal
median 为 `53.55 ms`，仍可能保留约 10% 以上总时间优势，同时进程 RSS 比 batch 16
更低。于是我只授权一次透明标注为 post-open development 的最终适配：

```text
唯一变化：native proposal batch 16 -> 8
不变：二进制、权重、结构、精度边界、A^T、alpha、K1、K4 reference、全部门槛
```

正式结果没有解决问题。wall 仍快 `15.27%`，但 RSS ratio 反而为 `1.1395`。
Candidate RSS p90 `166.38 MB`，Zero-K4 `146.01 MB`。因此“减小 batch 就能闭合
整体 RSS”被数据否定，后端 batch/精度微调到此停止。

只读独立审计重新汇总 85 次 pass，得到相同的 `15.2747%` wall 降低和
`1.13947` RSS ratio；5 个 session 分别快
`14.61% / 12.75% / 14.85% / 15.04% / 18.86%`，85 次配对没有一次慢超过
5%。审计还逐项核对了 10 个 worker 的 batch 8、无 Torch/MLX、101 帧 receipt、
`2A+2A^T` 对 `4A+4A^T` 调用账、shape/dtype/finite 和未打开
fresh/validation/test/truth。正式失败判决与原报告一致。

不能把 ratio 变差简单解释成 batch 8 本身必然占用更多内存。`ru_maxrss` 记录的是
整个常驻进程历史高水位，会同时受到 allocator、线程生命周期、临时数组和基线进程
波动影响。可以下的结论只有：在冻结的真实测量合同下，batch 8 没有提供可复现的
RSS 改善。

### 失败后的单 pass 机制诊断

为了区分“Candidate 本身占内存”与“17 次循环累计 allocator 高水位”，正式失败后
又做了一个不参与判决的诊断：每臂启动 5 个全新进程，每个只预热 1 次、测量 1 次。

```text
Candidate RSS p90 = 143,147,008 bytes
Zero-K4 RSS p90   = 127,090,688 bytes
ratio             = 1.12634
```

它仍明显超过 1.05。因此长循环可能影响绝对高水位，但不是失败的唯一原因；仅把 17
次循环改写成更短或复用 allocator，不能被当作足够的解决方案。下一种机制如果还要守住
内存门，必须实质减少 Candidate 特有的活跃缓冲区或执行状态，并重新接受独立验证。
这条单 pass 结果是 post-hoc mechanism diagnostic，不是正式算法门，也没有授权新数据。

## 为什么不继续试 batch 4、线程数和更多编译参数

因为这会把论文问题退化成在同一条已经看过结果的开发轨迹上寻找幸运配置：

1. v12.5 已经根据 synthetic batch 4/8/16/32 筛查选过一次；
2. v12.6 又根据 v12.5 的正式 RSS 缺口做了一次 post-open 适配；
3. v12.6 仍失败，再继续改线程栈、batch 或编译器就不再是独立证据；
4. 即使偶然过门，也不能说明跨工况稳定，更不能说明真实 BOST 泛化。

因此现在停止的是“围绕同一后端继续调参”，不是停止整个研究。下一条科学上有价值的
路线必须改变可检验的算法机制或论文主张，并重新冻结独立的跨轨迹验证，而不是继续
挤这条 p14 的几 MiB。

## 成功、失败与突破性判断

成功：

- 干净 reference 不再导入 Torch 或 MLX；
- 三个版本均为 101/101 joint match、0 harm、0 severe；
- 原生 C 与冻结 Torch 的数值差约 `1e-7`；
- 完整调用从每帧 `4A+4A^T` 降为 `2A+2A^T`；
- 三个干净后端的 wall 都通过 10% 门；
- v12.5 的 15.46% wall 优势在 85 次 pass 和 5 个 session 中没有一次慢超过 5%。

失败：

- v12.4、v12.5、v12.6 的 whole-worker RSS ratio 分别为
  `1.3367 / 1.0745 / 1.1395`，全部超过 1.05；
- v12.4-v12.6 都只在同一条 post-open fit trajectory 上运行；
- 其余 fit、fresh、validation、untouched test 和真实 BOST 均未放行；
- 公开 CFD proxy 的 compatibility 不能代替实验重复测量定义的物理“同精度”。

突破性判断：

```text
clean_operator_call_reduction=true
single_fit_clean_wall_headroom=true
clean_whole_worker_rss_advantage=false
cross_trajectory_confirmation=false
real_bost_evidence=false
algorithm_breakthrough=false
```

最重要的真实结论是：当前方法的算子调用与 wall-time 机制有可信开发证据，但“同精度、
更快、内存不增”这个完整论文命题尚未成立。后续工作必须正面解决这一缺口，不能把前两项
成功合并成第三项也成功。
