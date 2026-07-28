# v27 外部曲折代理资源实测：调用减少成立，但完整资源门失败

## 一句话判决

固定的 Observable Reduced Warm K1 在 v26.3 已经通过三条 external-to-fit
PoolFire 轨迹、两个离散曲折强度的重建兼容门，但 v27 的正式 fresh-process
资源实测没有通过：

```text
FAIL_EXTERNAL_CURVED_RECONSTRUCTION_RESOURCE_V27
algorithm_breakthrough=false
paper_success=false
```

这不是程序崩溃、统计没有收敛或验证器不一致。独立验证器从 3888 份 worker/parent
回执重新生成全部调用账与资源统计，最大差为 `0`：

```text
PASS_INDEPENDENT_RECOMPUTATION_EXTERNAL_CURVED_RESOURCE_V27
```

真实结论是：当前实现确实少做了精确算子调用，也比完整父方法稳定快约 10%，但它
没有相对最重要的 Zero-CGLS K4 对照达到 10% wall 加速，而且 direct-child peak
RSS 在六个单元里全部越过 no-harm 门。因此不能把“调用减少”写成“稳定部署加速”。

## 为什么做这轮

v26.3 已经回答精度问题：不重训、不改阈值时，固定方法能够承受
`新功率/尺寸组合 + field-dependent curved forward`。但论文还必须回答一个更硬的
问题：

> 少算的 `A/A^T` 是否真的换成了 wall-time 与内存收益？

如果只报告调用数，不测真实进程时间和内存，便宜的网络推理、模型加载、数据搬运与
内存分配可能吃掉全部理论收益。

## 冻结方法与账本

三种方法共享同一 observation、几何和求解语义：

```text
Observable Reduced Warm K1   202A + 152A^T = 354
Full Parent Warm K1          202A + 202A^T = 404
Zero-CGLS K4                 404A + 404A^T = 808
```

主方法相对父方法少 `50 A^T`，相对 Zero-K4 少 `454` 次完整调用。v27 没有重新
训练、重新拟合或根据运行结果修改阈值。

## 实验规模

正式实验使用三条独立轨迹：

```text
p33-s05
p45-s01
p58-s05
```

每条轨迹分别测试 `beta=0.001` 和 `beta=0.002`。两个 beta 共用同一条底层轨迹，
所以科学独立单位是 `3`，不是 `6`。

每个 trajectory-beta 单元执行：

- 102 个正式配对 triad；
- 三个 arm 的六种运行顺序各 17 次；
- 6 个 warmup triad，只记录、不进入判决；
- 长度 5 的 circular moving-block bootstrap，50000 次抽样。

总计：

```text
正式 fresh process   1836
warmup               108
总 fresh process     1944
worker + parent 回执 3888
```

父进程从 `execve` 前计时到子进程退出，并用 `wait4` 记录子进程 CPU 与
direct-child `ru_maxrss`。曲折 observation 的离线生成不在计时范围内；本轮也没有
测整棵进程树或完整实验链路内存。

## 正式结果

### 相对 Full Parent Warm K1

六个单元的主方法 wall 中位降幅为：

```text
10.33% - 11.16%
```

子进程 CPU 中位 ratio 为：

```text
0.8859 - 0.8951
```

这是可信的正向机制信号：删去 50 次精确 `A^T` 确实减少了当前 worker 的时间。
主方法在每个单元的 96.08%-100% 配对运行中更快，wall p90 ratio 也保持在
`0.9132-0.9451`。

但这一比较仍未整体通过：

- 2/6 单元的 wall 单侧 95% 保守降幅低于冻结的 10%；
- 6/6 单元的 RSS p90 ratio 为 `1.0686-1.1103`，超过 `1.05`；
- 6/6 单元的 RSS worst ratio 为 `1.1511-1.2038`，超过 `1.10`。

### 相对 Zero-CGLS K4

这是更关键、也更不利的结果。六个单元的 wall 中位降幅只有：

```text
2.19% - 2.89%
```

其单侧 95% 保守降幅只有 `1.40%-2.45%`，远低于冻结的 10% 目标。wall p90 ratio
为 `1.0037-1.0345`，即慢尾部没有形成优势；三个单元的更快比例低于 80%，三个
单元的单次最坏 wall harm 超过 25%。

资源内存同样失败：

```text
RSS p90 ratio   1.1099 - 1.1269
RSS worst ratio 1.1776 - 1.2330
```

CPU 中位 ratio 为 `0.9686-0.9773`，说明主方法只节省约 2.3%-3.1% CPU。大量
精确调用减少没有按比例转成 CPU/wall 收益，表明当前廉价线性 proxy 中，proposal
加载与推理、几何构建、Python 进程启动和公共固定开销占据了主要成本。

## 为什么失败

不是精度失败。v26.3 已经冻结并验证了相同方法在这些 observation 上的
field/gradient/observation 兼容性。

也不是“统计波动把一个明显正结果判没了”。相对 Zero-K4 的点估计本身就只有约
2%-3%，距离 10% 很远；六个单元的 RSS 点估计也全部越界。

根因可以分成两层：

1. **当前 proxy 的一次 `A/A^T` 太便宜。** 理论调用减少很大，但省下的算子时间
   没有压过模型/package 与 fresh-process 固定开销。
2. **可学习 proposal 占用额外常驻与临时内存。** 主方法的 direct-child RSS
   在所有单元都高于两个对照，严格内存门没有模糊空间。

因此 v27 否定的是：

> 当前固定实现，在这台 Mac、这个 16x16x32 廉价线性 inverse、每条序列一次
> fresh process 的口径下，已经获得稳定 wall/RSS 部署优势。

它没有否定：

- warm initializer 的重建兼容性；
- 精确 `A/A^T` 调用减少；
- 相对完整父方法的约 10% worker wall 收益；
- 在真实昂贵 nonlinear forward/JVP/VJP 或常驻服务中出现更大收益的可能性。

但后三项都必须另做实验，不能从本轮外推。

## 对论文路线的直接影响

当前证据链已经把问题定位得很窄：

```text
精度兼容：通过
曲折 forward 压力：通过
完整调用减少：通过
廉价 proxy fresh-process wall/RSS：失败
真实 BOST：未测试
```

所以继续微调同一个 fresh worker、重复增加运行次数或换一个更大的网络都不能改变
论文判断。下一项有价值的实验必须改变物理成本层级：

1. 优先接入组内真实 nonlinear forward/JVP/VJP，或一个计算成本同等级的公开
   curved inverse，直接测每少一次物理调用究竟省多少时间；
2. 同时报告 one-shot fresh 与模型常驻两种部署口径，不能只挑有利口径；
3. 保留 Zero-K4、Full Parent、最便宜 classical control 和同一精度门；
4. 若在昂贵物理算子下仍不能稳定超过 Zero-K4，关闭当前方法的“速度贡献”主张，
   只保留精度/调用数的负边界结果。

## 证据边界

本轮可以说：

- 三条 external-to-fit PoolFire 轨迹、两个离散曲折压力档；
- 1944 个 fresh reconstruction-worker 进程；
- 配对 wall、child CPU、direct-child RSS 与独立全量复算；
- 当前实现相对父方法有稳定的时间机制信号，但完整资源目标失败。

本轮不能说：

- 真实 BOST 加速；
- 真实相机、噪声或标定下同精度；
- 跨数据集泛化；
- whole-pipeline 或 process-tree 内存优势；
- 算法突破、SOTA 或论文已经成功。

```text
algorithm_breakthrough=false
real_BOST=false
cross_dataset_generalization_proven=false
whole_experimental_pipeline_resource_advantage_proven=false
paper_success=false
```

公开专页：
`operator-learning/external-curved-resource-v27.html`

机器可读脱敏结果：
`docs/poolfire_c_external_curved_resource_v27_public_summary.json`

结果图：
`assets/poolfire_c_external_curved_resource_v27.png`
