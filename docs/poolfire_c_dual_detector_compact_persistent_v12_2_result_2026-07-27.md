# v12.2 常驻进程诊断：调用减半终于转化为 12.64% 内核 wall 优势，但 RSS 仍失败

## 一句话结论

上一轮 v12.1 在 17×2 个全新进程里只快 `0.28%`，因此无法判断是算法计算没有优势，
还是 Python/Torch 启动和重复加载把优势淹没。v12.2 没有改模型、batch、alpha、
CGLS K1 或阈值，只把部署口径改成“模型、几何、观测各加载一次，然后反复重建”。

五个独立常驻 session、每个 session 预热 1 条完整轨迹后计时 17 条完整轨迹，
Candidate 与 Zero-K4 各得到 85 个 101 帧 pass：

```text
Compact Candidate = 112.46 ms / 101 frames
Zero-CGLS K4      = 128.74 ms / 101 frames
wall reduction    = 12.64%  (门槛 >= 10%，PASS)

Candidate RSS p90 = 398.13 MB
Zero-K4 RSS p90   = 344.06 MB
RSS ratio         = 1.1571  (门槛 <= 1.05，FAIL)
```

独立 validator 重算全部 10 个 worker、85×2 个计时记录、调用 receipt、兼容性、会话
尾部和最终判决，最大兼容性差为 `0.0`。正式判决：

```text
PASS_POSTOPEN_PERSISTENT_WALL_HEADROOM_RSS_FAILED
algorithm_breakthrough=false
```

## 为什么这次实验能回答问题

v12.1 的 cold-process wall 包含：

```text
Python 进程启动
+ NumPy / Torch 导入
+ observation / geometry 加载
+ checkpoint 加载
+ proposal
+ A/A^T 与 CGLS
+ 字段序列化
```

这些一次性成本在当前便宜的 straight-ray proxy 上比物理计算还大，因此 Candidate
和 Zero-K4 都在约 1.08 秒，调用减半几乎不可见。

v12.2 每个 worker 只加载一次模型、几何和 observation；计时区间只包含每次真正需要
重复执行的 proposal 与未修改 solver。它不是完整冷启动延迟，但更接近服务进程、
实验工作站连续处理帧或批量重建时的稳态计算。

## 时间从哪里来

常驻 Candidate 的中位数分解为：

| 组成 | 101 帧中位时间 |
|---|---:|
| compact CNN proposal | 49.06 ms |
| exact `A^T` lift + alpha + strict CGLS K1 | 63.11 ms |
| Candidate 合计 | 112.46 ms |
| Zero-CGLS K4 solver | 128.74 ms |

这说明真正的 break-even 关系是：

```text
模型推理成本 49.06 ms
<
从四调用降到两调用节省的 solver 成本约 65.63 ms
```

所以净剩约 `16.28 ms`，对应 `12.64%` 的稳态 wall 优势。五个配对 session 的
Candidate 都更快，最弱一组仍快 `9.63%`，不是少数快样本拉高均值。

## 精度与调用账没有变化

`p45-s03` 早已被 v11.3 烧掉，本轮明确只作 post-open development；模型没有再训练，
也没有根据真值选择 batch 或阈值。全部 worker 完成后才打开已有 pair truth 评分：

```text
joint matched = 101 / 101
joint harm = 0
severe harm = 0
Candidate = 2A + 2A^T / frame
Zero-K4 = 4A + 4A^T / frame
```

因此 wall 差来自相同输出机制下的实际计算路径，不是放松精度门或少算隐藏步骤。

## 为什么仍不能称突破

第一，峰值内存仍高 `15.71%`，比冻结的 no-harm 门差得很清楚。10,548 个 float64
参数本身不到 0.1 MB，约 54 MB 的进程差主要来自 Torch 执行、批量激活和缓存，而
不是权重文件大小。继续删参数未必能解决。

第二，`12.64%` 只属于“已烧掉 p45 + 当前 Mac CPU + straight-ray proxy + 常驻内核”
这一口径。cold process 仍没有加速，RSS 仍失败，两条 untouched test 没有打开，
真实 BOST 的 curved ray、相机标定、噪声和实际 `A/A^T` 成本也没有进入。

第三，RSS 数值来自每个 worker 的 `getrusage(RUSAGE_SELF)`，validator 独立重算了
统计，但没有第二套父侧 OS 采样。因此 RSS 失败方向可信，证据口径仍需如实保留。

## 成功、失败与突破性进展

成功：

- 85 次/臂常驻 pass 的 wall 中位数快 `12.64%`，超过预注册 10% 门；
- 五个 session 全部更快，没有会话级 wall harm；
- 兼容精度、0 harm 和调用减半保持不变；
- 独立 validator 重算 decision，最大兼容性差为 0。

失败：

- whole-worker peak RSS 高 `15.71%`；
- cold-process wall 仍没有优势；
- 仍没有 untouched trajectory 或真实 BOST 证据。

突破性判断：

```text
persistent_kernel_wall_headroom=true
whole_worker_RSS_benefit=false
cold_process_wall_speedup=false
algorithm_breakthrough=false
```

这是一项**重要阶段性机制发现**：此前“调用减半却不加速”的矛盾已经被定位为部署
边界，而不是调用账虚假。它把下一步问题压缩成两个真实工程物理量：组内实际
`A/A^T` 每次多贵，以及真实重建是常驻连续处理还是一次性冷启动。只有接入这些真实
成本并继续保持精度、wall 和 RSS，才可能升级为论文主结果。
