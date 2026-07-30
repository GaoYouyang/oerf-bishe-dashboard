# v72：固定已知几何工件首次同时通过在线 wall 与 RSS 开发门

日期：2026-07-31  
独立验证：`PASS_INDEPENDENT_RECOMPUTATION_CALIBRATED_FACTOR_ONLINE_V72`  
科学判决：`PASS_CALIBRATED_FACTOR_ONLINE_HEADROOM_PROBE_V72`

## 一句话结论

v70.1 已证明 q8-K1 在 `1515/1515` 个 PoolFire proxy 单元守住冻结精度，
并把 exact 调用从 `4A+4A^T` 减到 `2A+2A^T`，但每个 fresh worker
重新构造 factor 造成 RSS 失败。v72 没有再改算法，而是把冻结已知几何的
`5,899,392` 字节 q8 factor 编译为只读、哈希校验的 geometry-bound artifact。

在一条已经开封的 101 帧 observation stream 上，六个结果前冻结的 fresh-process
配对区组得到：

```text
loaded q8-K1 / Zero-K4

outer wall ratio p50 / p90 / worst
  0.566428 / 0.583830 / 0.583830        PASS

worker-self RSS ratio p50 / p90 / worst
  1.002156 / 1.029576 / 1.029576        PASS
```

这是当前路线第一次在“固定几何、在线只加载 factor”的开发口径下同时通过 wall
和 worker-self RSS 门。它是重要的阶段性正结果，但还不是算法突破：这里只覆盖
一条已打开轨迹、一个已知合成几何，完整 loaded-artifact Stage B 尚未运行。

```text
algorithm_breakthrough=false
paper_success=false
real_bost=false
```

## 1. 为什么做这一步

v70.1 的在线数学候选已经成立：

```text
observation
-> q8 detector-space cheap solve
-> exact A^T lift
-> unchanged exact CGLS K1

candidate: 2A + 2A^T
control:   4A + 4A^T
```

当时的内存失败主要来自每个 fresh worker 反复建立 exact stencil、tile 工作区和
randomized SVD 临时数组，而不是最终只占约 5.63 MiB 的 factor。真实装置一旦
几何固定，这些 factor 没有必要随每一段 observation 重建。因此 v72 问的是：

> 不隐藏一次性构造成本，把 geometry-bound factor 无损存盘并在在线进程中校验、
> 映射、完整触页后，时间优势和内存门能否同时成立？

这里没有标定图像、相机参数估计、重投影误差或漂移测试，所以它不能称为“完成
相机标定”。准确名称是“冻结已知合成几何的预编译重建工件”。

## 2. 两张成本账

### 离线已知几何编译

单个 fresh process 完成 exact operator 构造、tile4/rank8/p0 factor 构造、
54 个无损 NPY 数组写入、manifest/READY 哈希、重新打开与完整校验：

| 项目 | 结果 |
|---|---:|
| 完整 fresh process | 2.2163 s |
| factor 构造、序列化与重开 | 1.6982 s |
| worker-self peak RSS | 446.20 MiB |
| factor payload | 5,899,392 bytes |
| artifact 实际磁盘 | 5,918,355 bytes |

构建过程没有读取 observation、truth 或 raw rho。18 个 block 写入和重开后，
逐数组最大绝对差、随机 forward 相对差和 adjoint 相对差全部为 `0`。

### 在线重建

两条臂都在计时 fresh worker 内加载同一 observation 并重新构造相同 exact
operator。候选额外导入、哈希验证、映射并完整触达 q8 artifact；对照不导入
factor 模块，也不访问 artifact 路径。因此公共 exact geometry 成本没有偷偷
转嫁给 Zero-K4。

| 统计量 | loaded q8-K1 | Zero-K4 |
|---|---:|---:|
| outer wall p50 | 8.6864 s | 15.3683 s |
| outer wall p90-higher | 8.8951 s | 15.3746 s |
| worker-self RSS p50 | 474.50 MiB | 478.11 MiB |
| worker-self RSS p90-higher | 492.25 MiB | 478.91 MiB |

注意：RSS 应按配对比值判门，不能用两组分位数相除代替。六个配对区组的
RSS ratio 全部位于 `0.97905-1.02958`，所以 p50 与 p90 都低于冻结的 `1.05`。

## 3. 摊销没有被隐藏

按完整离线 fresh process `2.2163 s` 计算，而不是只取最有利的 factor kernel：

```text
median online saving per 101-frame stream
  = 15.3683 - 8.6864
  = 6.6819 s

break-even
  = ceil(2.2163 / 6.6819)
  = 1 stream = 101 frames
```

这只说明当前 Mac、当前已开封 stream 的离线构造在一条 101 帧序列内即可摊销。
它不代表其他机器、其他几何、短序列或真实 BOST 也有相同临界点。

## 4. 独立复算

独立 validator 不导入正式 probe/controller，完成：

1. 重新哈希 manifest、READY 与全部 54 个数组文件；
2. 核对 18 个 view-component block 顺序和 5,899,392 字节 payload；
3. 核对 12 个 fresh worker 的 arm、调用账、数据读取边界和输出摘要；
4. 从原子 wall/RSS 回执重新组成六个相邻配对区组；
5. 独立实现 `higher` 分位数，重算 wall、RSS、PASS/FAIL 与 break-even；
6. 正式摘要和独立复算的最大数值差为 `0`。

两套程序仍共享冻结的 v66/v70 数值核心，所以这里证明的是资源统计和工件绑定
的独立复算，不是端到端物理实现完全独立。

## 5. 现在能说什么

### 已经成立

- geometry-bound artifact 对构建时 q8 factor 做到数组、forward、adjoint 零差异；
- 候选在线账保持 `2A+2A^T`，Zero-K4 为 `4A+4A^T`；
- 单条 101 帧开发 stream 的六个 fresh 配对区组同时通过 wall 与 worker-self RSS；
- 完整离线构造成本、峰值内存和磁盘体积单列，未藏入在线账；
- 原子回执、分位数、工件哈希和摊销由独立程序重算。

### 尚未成立

- 只跑了一条已开封轨迹和一个已知合成几何；
- loaded artifact 尚未重新通过三档几何、五条轨迹、全部 1,515 个精度单元；
- 尚无完整多轨迹 worker-tree/pipeline RSS 正式门；
- 这不是相机标定、神经算子、curved ray、外部数据族或真实 BOST；
- 没有算法突破、论文成功、SOTA 或广泛泛化。

## 6. 下一道唯一有效门

现在不再微调 tile，也不提前训练大网络。下一步只做：

1. 对冻结的三档已知九视角几何分别生成 geometry-bound q8 artifact；
2. 用加载路径重新跑五条 PoolFire fit trajectory、三档几何、全部 1,515 cells；
3. 保持 v70 的 field / gradient / interior-gradient / observation 门和调用账不变；
4. 只有 `1515/1515` 再次通过，才允许用 loaded artifact 运行正式多轨迹
   fresh resource Stage C；
5. 正式 Stage C 通过后，才进入独立公开反应流族和组内真实 BOST 迁移。

## 公开附件

- 机器摘要：`docs/nine_view_calibrated_factor_online_v72_public_summary.json`
- 结果图：`assets/nine_view_calibrated_factor_online_v72.png`
