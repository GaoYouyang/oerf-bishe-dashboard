# 本机算力与服务器门禁

> 状态：`NO_USER_ACTION_REQUIRED`
> 结论：当前几何、有限孔径、mask 与小模型工作继续在本机跑；暂不租 GPU。

## 1. 已核验的本机

- MacBook Pro，Apple M5；
- 10 核 CPU、10 核 GPU；
- 32 GB 统一内存；
- 当前可用磁盘约 443 GiB。

## 2. 真实任务的实测

| 任务 | 真实工作量 | 墙钟时间 | 进程最大常驻内存 | swap | 判决 |
|---|---:|---:|---:|---:|---|
| A0/A1 shard 与几何审计 | 49,766,400 条射线 | 已完成 | 低于整机限额 | 0 | 本机合适 |
| B0/B1 解析中心线普查 | 9 视角，49,766,400 条 | 35.9 s | 约 0.82 GiB | 0 | 本机合适 |
| B2 QMC-8 有限孔径支持审计 | 约 3.96 亿个合格样本候选 | 88.3 s | 约 0.72 GiB | 0 | 本机合适 |
| B2 QMC-16 有限孔径支持审计 | 约 7.92 亿个合格样本候选 | 130.9 s | 低于 0.8 GiB | 0 | 本机合适 |
| B2 QMC-32 有限孔径支持审计 | 约 15.85 亿个合格样本候选 | 225.5 s | 约 0.80 GiB | 0 | 本机合适 |
| B0 合成 12³ interface inverse | 147 rays × 16 samples；60 次 Landweber | 约 0.03 s 核心迭代 | 约 0.21 GiB 进程峰值 | 0 | 接口 smoke 通过 |
| B0 真实几何 16³ profile | 2,304 rays × 16 samples | MPS F/Aᵀ 约 0.94/1.58 ms | MPS current 约 3.54 MiB | 0 | 子集接口通过 |
| B0 真实几何 32³ profile | 2,304 rays × 16 samples | MPS F/Aᵀ 约 0.90/1.39 ms | MPS current 约 7.78 MiB | 0 | 子集接口通过 |
| B0 全 support 16³ float64 | 10,628,822 rays × QMC16 / call | F+Aᵀ 53.43 s；4-step CGLS 238.22 s | 约 5.34 GiB | 0 | 本机通过 |
| B0 全 support 32³ float64 | 同 rays/QMC/calls | F+Aᵀ 50.55 s；4-step CGLS 218.03 s | 约 5.35 GiB | 0 | 本机通过 |
| 32³ compact stencil cache build | 170,061,152 个固定 aperture samples | 14.94 s；5.017 GB | 低于 32 GB | 0 | 一次建库，本机合适 |
| 32³ direct vs cached pair | 同 rays/QMC/dtype；8 threads；各 2 次 | 37.92 s → 17.04 s median | benchmark process 约 9.59 GiB | 0 | 2.225× same-session speedup |

2,304-ray 两行只用于接口 smoke；全 support 与 cache 行均为全部 support active rays 的实测。所有时间都是本地诊断，不是算法速度优势声称。

## 3. 网络是不是当前短板

不是。

- 2026-07-17 再测 `networkQuality` 下行约 `70.1 Mbps`；
- base RTT 约 `254 ms`，交互延迟偏高，但不影响已经下载到本地的 PSU 反演；
- GitHub、arXiv、Springer 首页总耗时约 `1.64 / 1.07 / 3.51 s`；
- 当前 `A/Aᵀ`、CGLS、缓存与绘图都不访问互联网；
- chunk profile 显示约 `82%` 的单次时间花在重复生成 aperture geometry 与 trilinear stencil，而不是下载。

吞吐会随代理和时间波动，Springer 的首字节明显慢于 GitHub/arXiv，但当前不需要
用户调整网络或 VPN。网络只会影响后续下载新数据、论文或远端服务器传输，
不会加速本机 support inverse。

## 3.1 本轮时序优化的实测收益

covariance-conditioned PCGLS 网格原先对 stage 2/3/4/5 分别从头求解。
现在固定 covariance 与 Sobolev strength 后只运行一条最大 5-stage trajectory，
再读取前缀 checkpoint：

- 120 candidates × 16 replicates；
- 逻辑 `A/Aᵀ` calls 保持 `6,784`；
- 物理 calls 从 `6,784` 降到 `2,464`，减少 `63.68%`；
- 墙钟 `41.15 s`，进程峰值约 `573 MB`；
- checkpoint 与独立求解逐值测试一致。

因此当前最有效的“增加并行”不是让多个 MPS 训练互相抢统一内存，而是先消除
重复轨迹，再把文献核验、统计审计、网页/日志/制图作为独立轻任务并行推进。

## 4. 为什么不同时跑多个重型反演

本机只有一套 10 核 CPU、统一内存和同一块 SSD。多个完整 inverse 同时运行会争用：

1. CPU 向量化与 PyTorch 线程池；
2. 统一内存带宽；
3. 5 GB cache 的顺序读取；
4. scatter-add 的缓存层级。

实测 4→8 threads 只有约 6% 小幅提升，说明单纯增加线程不是主解。正确并行方式是：

- 重型 `A/Aᵀ` 与 CGLS 串行；
- 文档、网页、轻量测试和图表并行；
- 固定几何先建一次 cache，再让所有算法复用；
- 未来服务器上把“不同 seed / 不同模型”分配到独立 GPU，而不是让多个任务抢同一张卡。

## 5. 本机继续承担的工作

1. 冻结 32³ B0 checkpoint/config/prediction hash；
2. gauge、Tikhonov/TV 与固定停止规则消融；
3. rotation 40 唯一 development run 的 held-out reprojection；
4. stencil cache 与 adjoint-safe mixed-precision 机制实验；
5. 图表、报告、公开汇总、网页和论文工作稿；
6. 只有 development 正信号出现后，才下载 BLASTNet 小裁剪并启动反应场 truth 基线。

## 6. 什么时候才触发租服务器

必须同时满足：

1. B0/B1/B2/B3 的物理审核和 held-out 重投影至少给出一条可证伪正信号；
2. 接口已在 16³/32³ 过 forward/adjoint smoke，候选网络还需 inverse smoke；
3. 全 support streaming 固定 batch 已实测单步时间、峰值内存和吞吐；
4. 按 64³/128³、视角数、孔径样本数和 seeds 完成显存/时长外推；
5. 强基线、停止规则、评估指标与 split 已冻结，不会上服务器后再随意挑模型。

触发后再根据 profile 决定是单张 24/48 GB CUDA GPU，还是 80 GB 级别或多卡。在没有 profile 前直接租卡，只会把“问题定义错”变成“更快地训错问题”。

## 7. 用户协助规则

目前不需要用户操作，因此不在桌面创建执行文档。

只有以下情况才会在 Mac 桌面直接创建一份 `.txt`：

- 需要用户租用 GPU；
- 需要用户提供组内数据/权限；
- 需要人工通过登录、人机验证或许可协议；
- 需要师兄确认一个会改变训练定义的物理语义。

文档将包含：需要做什么、为什么、预计费用/时长、完整命令、输出路径和失败回退。
