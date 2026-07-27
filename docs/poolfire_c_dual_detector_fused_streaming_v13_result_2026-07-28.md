# PoolFire C 路线 v13：融合流式运行时正式合成预检

日期：2026-07-28
状态：`FAIL_SYNTHETIC_PREFIT_FUSED_STREAMING_V13`

## 一句话结论

v13 把冻结 compact-w16d2 Dual-K1 的完整调用数从 Zero-K4 的
`4A+4A^T` 保持为 `2A+2A^T`，在 101 帧确定性合成预检上把中位耗时降到
`0.06504 s`，在预热后的 proposal + solver + 输出写入区间比流式 Zero-K4 快
`42.68%`；但 whole-worker peak RSS 比值为
`1.05118`，略高于事前写死的 `1.05000`。因此正式裁决仍是失败，不授权打开
fit、fresh、validation 或 test。

这不是算法突破，也不是实际 BOST 或跨工况结果。

## 为什么要做 v13

v12.5 原生 C 已证明 frozen proposal 的数值、调用账和 p14 开发轨迹 wall
headroom 都成立，但单次 fresh-process 诊断仍是：

```text
Candidate peak RSS p90 = 143,147,008 bytes
Reference peak RSS p90 = 127,090,688 bytes
RSS ratio               = 1.12634
```

要达到 `1.05`，Candidate 至少需要减少 `9,701,786 bytes`。继续换 batch 或
编译参数没有价值，所以 v13 改的是内存生命周期本身：

1. float64→float32、RMS 归一化、正负奇对称构造和反归一化全部进入 C；
2. 四个工作线程常驻，每个线程只保留三份单样本特征 scratch；
3. 网络按样本并行，不再每层反复创建线程或保存三份 full-batch 特征；
4. 下一块 8 帧 proposal 与当前块的精确 `A^T + alpha + K1` 重叠执行；
5. Candidate 和 Zero-K4 都逐帧写同一 NPY 输出合同，不把 101 帧字段全部留在 RAM。

模型权重、网络算术、float32 proposal、float64 物理算子、alpha 区间、K1/K4、
指标和门槛均未改变。

## 事前合同

正式运行前写死：

- 101×2072 float64 高斯合成观测，seed `130013`；
- 32×32×64 坐标构造同一个 16×16×32 三视角 straight-ray inverse；
- 每个 arm 5 个全新进程，每进程 1 次预热、1 次测量；
- Candidate 必须严格为 `202A + 202A^T`；
- Zero-K4 必须严格为 `404A + 404A^T`；
- proposal 和最终场相对 L2 的 p90/worst 均不超过 `5e-6`；
- v13 相对 v12.5 至少减少 `9,701,786 bytes`；
- v13 / 流式 Zero-K4 RSS p90 不超过 `1.05`；
- v13 中位 wall 至少快 10%，且每个配对 session 最多慢 5%；
- 任一门失败即整体失败，没有均值覆盖和事后放宽。

临时随机权重只用于先跑通 15-worker 队列和测量程序，不计科学证据。正式裁决使用
冻结训练 checkpoint，但没有读取任何 fit observation、field truth 或评分指标。

## 正式数值结果

### 数值与调用账

```text
proposal relative-L2 p90 / worst = 1.381e-7 / 1.427e-7
field relative-L2 p90 / worst    = 7.668e-8 / 8.429e-8

v13 Dual-K1   = 202A + 202A^T
Zero-K4       = 404A + 404A^T
```

五个 session 的输出均逐字节确定；15 个 worker 都未加载 Torch 或 MLX。

### 稳态 measured wall

```text
v12.5 full-trajectory Candidate median = 0.127755 s
v13 fused-streaming Candidate median   = 0.065039 s
streaming Zero-K4 median               = 0.113459 s

v13 vs streaming Zero-K4 reduction     = 42.676%  PASS
```

五个配对 session 的 v13 都更快，最弱一组也快 `39.36%`，所以时间成功不是由
单个异常 run 拉出来的。这个口径不含进程启动、checkpoint 加载和 native context
创建，因此不能扩写成完整冷启动部署加速。

### Whole-worker peak RSS

```text
v12.5 Candidate p90        = 82,984,960 bytes
v13 Candidate p90          = 60,571,648 bytes
streaming Zero-K4 p90      = 57,622,528 bytes

v13 vs v12.5 reduction     = 22,413,312 bytes  PASS
v13 / streaming reference = 1.05117998         FAIL
frozen 1.05 cap            = 60,503,654.4 bytes
excess above cap           = 67,993.6 bytes
```

v13 确实消掉了旧实现约 27.0% 的峰值内存，但预注册规则要求相对公平流式参考也必须
不超过 5%。最终只多约 68 kB，仍然是失败。

### 独立审计指出的统计边界

独立只读审计没有发现数据、算术、调用账或足以推翻结论的代码问题，确认按冻结
validator 必须判 `FAIL`。同时它指出：

- validator 的 p90 使用 NumPy `method="higher"`；
- 只有 5 个 session 时，这个 p90 实际等于样本最大值；
- 协议正文没有显式写出该插值方法，这是一个应公开的 P1；
- 若事后改用默认 linear p90，RSS ratio 会是 `1.048859` 并通过。

正式运行前绑定的 validator 已经固定使用 `higher`，所以不能在看到结果后换算法
改判。正确结论是“保守 higher-p90 严格门失败，RSS 接近阈值且统计稳健性有限”，
而不是“v13 已被证明在内存上必然更差”。

## 为什么不把它写成成功

`1.05118` 和 `1.05` 很接近，但门槛的作用就是阻止看完结果后说“差不多也算过”。
如果现在把阈值改成 1.052，论文里的资源结论就不可审计。

允许写：

- 融合流式 C 实现保留了冻结模型与最终字段；
- 完整 `A/A^T` 调用减半；
- 在固定合成 inverse 上 wall 明显更快；
- 相对旧 native runtime 的 RSS 降低 22.4 MB；
- RSS 相对公平参考仍以很小余量失败。

禁止写：

- 通过完整部署资源门；
- 跨轨迹、跨工况或真实 BOST 泛化；
- 物理同精度已经建立；
- 新算法优于现有方法；
- 完整冷启动端到端部署已经加速；
- 该路线没有任何内存潜力或已被数学上证伪；
- `algorithm_breakthrough=true`；
- 已具备论文成功结论。

## 下一策略

v13 已回答后端问题：网络推理和算子调用可以流水重叠，原生运行时也能把额外内存压到
接近参考；本轮的 5-session RSS 统计不足以宣判这条路线没有潜力，但继续在已见结果
上围绕 68 kB 调线程栈、allocator 或 batch 也不会改变论文科学判断。

因此这条后端路线在此冻结。下一轮回到模型层，只接受能改变跨 trajectory
compatibility、harm 或真实迁移判断的工作；在新的科学机制通过独立门以前，不打开
fresh、validation 或 test。

## 突破监测

```text
runtime_mechanism_headroom = true
formal_steady_state_wall_gate_passed = true
formal_peak_RSS_gate_passed = false
RSS_statistical_robustness_limited = true
fit_execution_authorized = false
algorithm_breakthrough = false
paper_success = false
```
