# v68.3：九视角 PoolFire Fresh 资源门独立验证结果

日期：2026-07-31
独立验证：`PASS_INDEPENDENT_VALIDATION_POOLFIRE_FRESH_RESOURCE_V68_3`
科学判决：`FAIL_POOLFIRE_FRESH_RESOURCE_STAGE_C_V68_3`

## 一句话结论

固定 q8-K1 在 v67.1 已经守住全部 1,515 个精度单元，并把理论精确算子账从
Zero-K4 的 `4A+4A^T` 降到 `2A+2A^T`。v68.3 现在补上了之前缺失的
fresh-process 实测：

```text
330 timed workers / 165 paired blocks
global outer-wall ratio p50 / p90 / worst
  0.655236 / 0.661113 / 0.671715          PASS

global worker-self RSS ratio p50 / p90 / worst
  1.355162 / 1.384474 / 1.418835          FAIL
global sampled worker-tree RSS ratio
  1.348131 / 1.376802 / 1.410150          FAIL
global sampled pipeline RSS ratio
  1.204059 / 1.311116 / 1.342926          FAIL
```

所以当前实现的中位端到端时间约减少 `34.5%`，但三种冻结内存口径的 p90
分别增加约 `38.4%`、`37.7%` 和 `31.1%`。Stage C 要求时间和内存同时通过，
因此整体结论必须是 **FAIL**，不能只挑速度写成“加速成功”。

```text
public_proxy_fresh_wall_gate_pass=true
public_proxy_sampled_rss_gate_pass=false
stage_c_pass=false
algorithm_breakthrough=false
paper_success=false
```

## 1. 实际比较了什么

比较双方没有在资源实验中改变：

```text
candidate
  observation
  -> fixed q8 geometry-compressed detector proposal
  -> exact known-geometry A^T lift
  -> unchanged exact CGLS K1
  exact budget: 2A + 2A^T

control
  observation
  -> zero-start exact CGLS K4
  exact budget: 4A + 4A^T
```

v67.1 已经在五条公开 PoolFire 轨迹、每条 101 帧、三档已知九视角
straight-ray 几何上证明 q8-K1 守住冻结的 field、完整 gradient、内部 gradient
和 observation compatibility。v68.3 没有重新挑帧或改精度门，只测资源：

- 5 条已经开封的 PoolFire fit trajectory；
- 3 档固定九视角几何；
- 每个 trajectory-by-geometry 层 11 个随机相邻完整区组；
- 30 个 reference worker；
- 330 个 timing worker；
- 165 个 q8/Zero 配对区组；
- 所有 worker 串行且时间区间不重叠；
- 两条 test trajectory 继续封存。

每个 fresh worker 都把 factor setup、cheap factor actions、精确
`A/A^T`、Python 启动和结果序列化放入同一个外层计时。资源门没有把 setup
藏到父进程，也没有把理论调用减少冒充 wall-time 下降。

## 2. 速度结果：稳定通过

| 统计量 | q8-K1 / Zero-K4 | 冻结上限 | 判决 |
|---|---:|---:|---|
| global outer wall p50 | 0.655236 | 0.90 | PASS |
| global outer wall p90-higher | 0.661113 | 1.05 | PASS |
| global outer wall worst | 0.671715 | 披露 | PASS |

绝对时间的全局中位数是：

```text
q8-K1      10.2671 s
Zero-K4    15.6663 s
```

五条轨迹的 outer-wall p50 比值全部在 `0.6546-0.6570`，没有靠某一条轨迹
撑起平均数：

| trajectory | wall p50 | wall p90-higher |
|---|---:|---:|
| p14-s05 | 0.654571 | 0.662173 |
| p22-s03 | 0.654662 | 0.659412 |
| p33-s01 | 0.657025 | 0.659662 |
| p45-s05 | 0.655652 | 0.661808 |
| p58-s03 | 0.654704 | 0.661361 |

这说明 v67.1 的 50% 精确调用减少在当前 Mac CPU、固定
`32×16×16`、已知九视角 straight-ray proxy 上确实转化成了稳定 wall-time
收益。它是资源证据中的真实正部分。

## 3. 内存结果：三种口径全部失败

| 资源口径 | ratio p50 | ratio p90-higher | 冻结上限 | 判决 |
|---|---:|---:|---:|---|
| worker-self high-water RSS | 1.355162 | 1.384474 | 1.05 | FAIL |
| sampled worker-tree peak RSS | 1.348131 | 1.376802 | 1.05 | FAIL |
| sampled controller+worker pipeline RSS | 1.204059 | 1.311116 | 1.05 | FAIL |

q8-K1 worker-self RSS 的全局 p90 是约 `642.5 MiB`，Zero-K4 是约
`474.9 MiB`；增加约 `167.6 MiB`。但正式回执同时告诉我们：

```text
persistent q8 factor state              5,899,392 bytes  = 5.63 MiB
largest materialized rearranged block  33,554,432 bytes = 32.00 MiB
observed worker-self p90 increase      about 167.6 MiB
factor setup wall range                1.529-1.611 s
```

因此不能把 167.6 MiB 全归因于 5.63 MiB 的常驻 factors。高水位测量覆盖了
第二份 ray bundle、`32 MiB` block 构造、随机 SVD 临时数组、保留 factors 和
后续重建；当前证据把瓶颈定位在 **factor 构造及其瞬时工作区**，而不是单独
定位在 persistent state。这个区分直接改变下一算法设计：优先流式构造、工作区
复用与避免重复 ray bundle，而不是只压缩最后留下的 5.63 MiB。

五条轨迹的 worker-self RSS p90 比值都在 `1.3760-1.3971`。因此内存失败不是
个别异常 worker，也不是某条工况造成的。pipeline RSS 是采样下界而不是精确
峰值，但它自己也已经明显越过 1.05 门，所以无需靠更精细采样才能判失败。

## 4. 独立验证做了什么

正式 controller 只发布 pending 结果。独立 validator 随后：

1. 从原始 pair truth 重建 1,515 个 observation；
2. 独立重放 30 个 reference worker，共 3,030 帧计算；
3. 30/30 canonical digest 和实际 `A/A^T` 调用账完全一致；
4. 检查全部 360 个 worker 时间区间串行且不重叠；
5. 独立重算 165 个配对区组，最大数值差为 0；
6. 独立重算 903 个聚合数字，最大数值差为 0；
7. 确认 source、pair、observation 与正式结果在验证前后不变；
8. 确认 test truth 未读取。

验证中唯一需要修复的是一个时钟量化边界：某条 receipt 的外层计时比两个
`mach_absolute_time` 时间戳之差小约 `6.0e-11 s`，而本机时钟分辨率约为
`4.17e-8 s`。修复只允许不超过时钟分辨率的这个比较误差；没有修改正式
receipt、聚合数字或科学阈值。修复后 adjustment count 为 1，完整独立验证通过。
随后另一个只读审计又确认修复后的原子结果已经在命令行尾部打印错误发生前完成
发布，因此本页使用的验证结果仍然有效。

独立程序仍共享冻结的 v66 数值核心，故：

```text
independent_resource_recomputation=true
end_to_end_numerical_independence_proven=false
```

## 5. 为什么这是有价值的失败

v68.3 排除了两个互相矛盾但都很常见的模糊说法：

1. “少一半 `A/A^T` 肯定更快”现在不再是理论猜测；fresh wall 真实通过。
2. “快了就是整体更好”也被否定；factor 构造路径把峰值内存推高了约
   31%-38%。

这把下一步从“继续调 rank、训练大网络”收窄为一个具体工程科学问题：

> 能否用 tiled/streamed 构造、工作区复用、ray-bundle 共享或低精度存储加
> 高精度累积，消除 factor setup 的约 168 MiB 高水位差，同时保住 q8-K1 的
> 精度、50% 精确调用减少和约 34.5% 的 wall-time 收益？

已有的 tiled 小网格原型只证明 forward/adjoint 代数可做，并显示 synthetic
内存可能下降；它的 factor 构造和外层时间反而更慢。它还不是科学结果，不能
直接合并后宣称修复。下一版必须先重新通过完整 v67.1 Stage B，再重新跑同一
fresh wall/RSS 门。

## 6. 成功与未成功

### 已成功

- q8-K1 的 50% 精确算子对减少转化成约 34.5% 的 fresh median wall 收益；
- 五条轨迹的时间收益稳定，没有 trajectory harm；
- 30 个 reference、330 个 timing、165 个 block 由独立程序完整复算；
- 内存瓶颈被定位到可操作的 factor 构造/工作区，而不是继续盲目扩大模型。

### 未成功

- Stage C 整体失败，因为三种 RSS p90 均超过 1.05；
- q8 是固定 geometry-compressed factor，不是 neural operator；
- 没有独立公开反应流族、曲折光线、相机标定或真实 BOST；
- 没有证明 whole-pipeline exact peak memory，只测得 worker high-water 与
  sampled lower bounds；
- 没有算法突破、全局原创性、广泛泛化或论文成功。

因此本轮应称为：

> **经独立复算的 fresh wall 正结果与 factor-setup high-water 负结果。**

不能称为：

> 内存友好加速算法、神经算子突破、真实 BOST 加速或论文完成。

## 公开附件

- 机器摘要：`docs/nine_view_poolfire_fresh_resource_v68_public_summary.json`
- 结果图：`assets/nine_view_poolfire_fresh_resource_v68.png`
