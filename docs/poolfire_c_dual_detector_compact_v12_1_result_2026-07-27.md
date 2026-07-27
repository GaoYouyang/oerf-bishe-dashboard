# v12.1 紧凑 DualRange-K1：参数缩小 7.30 倍，精度与调用保住，冷进程资源仍失败

## 一句话结论

我把 v11.3 的 77,020 参数 detector CNN 压缩为 10,548 参数的 `w16d2`，没有改动
后面的精确 `A^T`、可观测 alpha 或 CGLS K1。它在五条 fit trajectory 的完整 LOTO
仍为 `5/5`，又在已经烧掉的 `p45-s03` post-open development proxy 上保持
`101/101` joint match、`0` harm，并继续把每帧调用从 `4A+4A^T` 降为
`2A+2A^T`。

但是，17×2 个全新进程的资源判决仍然失败：

```text
Candidate wall median = 1.0814 s
Zero-K4 wall median   = 1.0845 s
wall reduction        = 0.28%   (门槛 >= 10%)

Candidate RSS p90     = 353.71 MB
Zero-K4 RSS p90       = 295.53 MB
RSS ratio             = 1.1969  (门槛 <= 1.05)
```

正式判决是：

```text
PASS_POSTOPEN_DEVELOPMENT_ACCURACY_CALLS_RESOURCE_GATE_FAILED
algorithm_breakthrough=false
```

## 为什么做这个紧凑模型

v11.3 已经证明原网络在一条锁模后的 fresh proxy 上能保持兼容精度并将完整算子调用
减半，但它自己太重：CNN 推理成本吃掉了省下的两对 `A/A^T`，wall 慢 6.35%，RSS
高 17.16%。

因此容量选择没有做架构海选，而是提前冻结两级阶梯：

1. 先跑 `w16d2`，10,548 参数；
2. 只有它不能在五条 LOTO 达到 `5/5`，才运行 `w24d3`。

`w16d2` 已经 `5/5`，所以第二个 33,336 参数模型没有运行。这个停止动作避免了看结果
后挑网络，也避免了无意义耗算。

## 五条完整 LOTO

| 留出轨迹 | joint match | joint harm | severe | teacher-q p90 |
|---|---:|---:|---:|---:|
| P14-S05 | 100% | 0% | 0 | 0.09846 |
| P22-S03 | 100% | 0% | 0 | 0.09347 |
| P33-S01 | 100% | 0% | 0 | 0.11177 |
| P45-S05 | 100% | 0% | 0 | 0.13402 |
| P58-S03 | 100% | 0% | 0 | 0.11093 |

独立 validator 重新解析 5 个 checkpoint、生成 505 帧 proposal，并用另一套 NumPy
路径重算 `A^T -> alpha -> K1`、三类指标与调用账。最大科学数值差为
`4.44e-16`，总调用为 `1010A + 1010A^T`。

相对原网络：

```text
parameter count: 77,020 -> 10,548  (缩小 7.30 倍)
full-fit wall:   290.86 s -> 112.22 s  (训练约快 2.59 倍)
```

训练速度不是部署速度，但它说明参数压缩真实降低了训练成本。

## p45-s03 的 post-open 结果

`p45-s03` 已经被 v11.3 的一次性 fresh 评分烧掉，所以这里不能再次称 fresh，也不能
用来证明泛化。它只回答一个开发问题：紧凑模型是否保住原有机制，以及资源瓶颈是否
随参数量下降而消失。

### 兼容精度

| 指标 | Compact Candidate p90 | Zero-K4 p90 |
|---|---:|---:|
| field relative-L2 | 0.51897 | 0.51713 |
| gradient relative-L2 | 0.96560 | 0.97090 |
| observation relative-L2 | 0.27174 | 0.26831 |

```text
joint matched = 101 / 101
joint harm = 0 / 101
field / gradient / observation harm = 0
severe harm = 0
```

正确说法仍是“以一半完整调用进入冻结兼容包络”，不是“每个误差都优于 K4”。

### 冷进程成本

| 项目 | Compact Candidate | Zero-K4 | 判决 |
|---|---:|---:|---|
| 每帧 A | 2 | 4 | 下降 50% |
| 每帧 A^T | 2 | 4 | 下降 50% |
| outer execve wall median | 1.0814 s | 1.0845 s | 只快 0.28%，失败 |
| whole-worker RSS p90 | 353.71 MB | 295.53 MB | 高 19.69%，失败 |

17 个 Candidate 与 17 个 Zero-K4 使用相同入口、相同 observation、相同线程数、相同
字段序列化，并交替先后顺序。第二个 validator 独立检查了 34 条进程记录、每个进程
的 101 条连续调用 receipt、wall/RSS 统计、兼容性和最终 decision；兼容性数值差为
`0.0`。

RSS 的原始采样仍来自 worker 自身的 `getrusage(RUSAGE_SELF)`，不是父侧第二套 OS
采样。不过 wall 已独立由父进程测量且没有达到 10% 门，因此即使不采信 RSS，这次
资源门仍然失败。

## 真正学到的东西

参数量已经不是主要矛盾。进一步分解显示，Candidate 与 Zero-K4 的 worker 内部
中位时间几乎相同，约为 `0.468 s`；外层 Python 启动、Torch 初始化、几何加载和
序列化形成了大块共同成本。网络变小后，省下的算子时间只是被这些共同成本淹没，而
不是算子调用减少消失。

探索性 batch 诊断还显示：减小 batch 可以把 Candidate 峰值内存压近 reference，
但会增加 proposal 时间；增大 batch 会略快，却扩大激活内存。单纯继续删参数或盲目
放大网络都不是下一步答案。

## 成功与失败边界

已成功：

- 第一个预注册紧凑容量在五条完整 LOTO 达到 `5/5`；
- 参数缩小 7.30 倍，full-fit 训练约快 2.59 倍；
- 在 post-open p45 仍为 `101/101` compatibility、0 harm；
- 每帧完整算子调用稳定减少 50%；
- 两级独立复算分别得到 `4.44e-16`、`1.39e-17` 和 `0.0` 的数值差。

未成功：

- cold-process wall 没有达到 10% 加速；
- whole-worker RSS 没有 no-harm；
- p45 已是 development，不是第二条 fresh；
- 两条 untouched test 没有打开；
- 尚未迁移到组内真实 BOST、真实相机标定和实验噪声；
- 不能写成算法突破、广泛泛化、SOTA 或论文完成。

## 当前工程判断

这条路线没有被否掉，但研究问题已经从“网络能不能学到 K4-compatible dual proposal”
转为“调用减半在什么部署与物理算子成本下能转化为真实 wall/RSS 优势”。在当前很便宜
的 straight-ray CPU proxy 和冷进程口径下，容量压缩不足以过门。下一次 untouched
开封之前，必须先冻结一种更贴近实际重建部署的常驻模型成本口径，或接入组内真实
`A/A^T` 的单次成本；不能靠重复使用 p45 调到好看。
