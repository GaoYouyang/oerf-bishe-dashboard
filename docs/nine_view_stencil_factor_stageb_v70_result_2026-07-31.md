# v70：低内存 exact-stencil tiled p0 通过全轨迹精度门

日期：2026-07-31  
独立验证：`PASS_INDEPENDENT_RECOMPUTATION_TILED_EXACT_P0_STAGE_B_V70`  
科学判决：`PASS_TILED_EXACT_P0_FULL_TRAJECTORY_COMPATIBILITY_V70`

## 一句话结论

v68.3 已证明旧 q8-K1 比 Zero-K4 的 fresh wall 快约 34.5%，但 factor setup
令三种 RSS p90 增加 31%-38%，所以资源门整体失败。v70 没有训练大网络，而是
直接修正这个瓶颈：

1. 复用 exact operator 已经保存的 trilinear stencil，不再构造第二份 ray bundle；
2. 每次只流式构造 4 行 detector-v tile；
3. 把 randomized SVD 的 power iteration 从 2 固定为 0，减少重复 tile passes；
4. 保留 rank 8、同一随机种子、4 步廉价 detector CR、精确 `A^T` lift 和未修改
   CGLS K1。

新候选在五条已开封 PoolFire fit trajectory、三档已知九视角几何、全部 505 帧
形成的 1,515 个单元上：

```text
candidate cells                 1515
matched-accuracy pass           1515
failed cells                       0
exact deployment budget        2A + 2A^T
independent maximum metric diff 5.68e-14
```

因此 **低内存表示已经守住完整 Stage B 精度门**。这是真实的阶段性正进展，
但 fresh wall/RSS 仍未重新运行，所以：

```text
stage_b_pass=true
fresh_resource_stage_authorized=true
fresh_resource_result=false
algorithm_breakthrough=false
paper_success=false
```

## 为什么这样改

旧实现先构造 exact operator 的 trilinear ray stencil，随后 factor builder 又独立
生成一套 ray bundle、path sums 和 `1024×4096` rearranged blocks。同一个
view 的 path sum 还会为 u/v 两个分量重复计算。v68.3 的正式回执显示：

```text
persistent factor state                 5.63 MiB
largest explicit rearranged block      32.00 MiB
worker-self p90 increase              about 167.6 MiB
```

所以主要矛盾不是最终 factors 太大，而是 setup 瞬时数组与重复几何构造。v70
直接复用 exact stencil，并让 tile 进入 range finder，不再先形成完整 block。

理论上，旧路径的三个主导 float64 数组是：

```text
path sum       32 MiB
row buffer     32 MiB
rearranged     32 MiB
declared sum   96 MiB
```

v70 每次只处理四行 detector-v tile，声明的三个主导数组合计约 12 MiB。
这是 **数组生命周期上界的减少**，不是 whole-process RSS 结论。

## 结果前选择与开发 Pareto

在读取 Stage B 结果前，五个构造方法按三个随机完整区组、每个 worker 16 个
synthetic observation 做了 fresh-process 开发筛选。它不读 truth、raw rho，
也不等价于正式 Stage C。

相对旧 dense q8：

| 构造 | wall ratio 中位 | worker-tree RSS ratio 中位 | 解释 |
|---|---:|---:|---|
| exact-stencil reuse p2 | 0.9078 | 0.9541 | 更快，但内存只降约 4.6% |
| 旧 geometry tiled p2 | 2.1457 | 0.8341 | 内存下降，时间代价过大 |
| **exact-stencil tiled p0** | **1.0509** | **0.8009** | 内存约降 19.9%，wall 只慢约 5.1% |

选择 p0 不是因为它已经“通过资源门”，而是因为它是唯一同时具备明显 RSS
headroom 和可能保留旧 q8 对 Zero-K4 速度优势的候选。参数在 Stage B 结果前
固定，之后没有按精度结果改 tile、rank、seed 或 gate。

## 全轨迹精度结果

门槛与 v67.1 完全相同：

- 相对 Zero-K4 的 field、gradient、interior-gradient、observation harm
  每个单元都不超过 `1.01`；
- 相对同 exact 调用的 Zero-K2，四项误差比每个单元都不超过 `1.00`；
- 1,515 个单元必须全部通过。

新候选的全局分布：

| 指标比 | p50 | p90-higher | worst | 门 |
|---|---:|---:|---:|---:|
| field / Zero-K4 | 0.98464 | 0.98980 | 1.00252 | 1.01 |
| gradient / Zero-K4 | 0.99496 | 0.99724 | 0.99949 | 1.01 |
| interior-gradient / Zero-K4 | 0.97891 | 0.99020 | 1.00281 | 1.01 |
| observation / Zero-K4 | 0.89338 | 0.91081 | 0.93356 | 1.01 |
| field / Zero-K2 | 0.80087 | 0.84667 | 0.91067 | 1.00 |
| gradient / Zero-K2 | 0.96422 | 0.97676 | 0.98631 | 1.00 |
| interior-gradient / Zero-K2 | 0.85708 | 0.91327 | 0.95529 | 1.00 |
| observation / Zero-K2 | 0.57830 | 0.60647 | 0.64662 | 1.00 |

最接近门槛的是 interior-gradient / Zero-K4 的 worst `1.00281`，仍低于
`1.01`。这不是靠均值掩盖尾部：全部 15 个 trajectory-by-geometry 层、
每层 101 帧均通过。

## 独立验证

正式 runner 用新的 **exact-stencil tiled p0** 构造。独立 validator 不导入
正式 runner，也不复用 tiled builder，而是用旧的 **dense geometry p0**
重新生成 factors，再重算所有 observation、candidate field、residual 和指标。

```text
recomputed cells                    1515
pass decisions                      1515
maximum absolute metric difference 5.684e-14
maximum relative metric difference 5.976e-16
formal payload unchanged            true
pair inputs unchanged               true
```

两条路径仍共享冻结的 voxel-gradient 与 trilinear-stencil 物理核心，因此
`end_to_end_physics_independence_proven=false`。但它已经排除了“tiled 累加顺序
或实现错误碰巧让正式结果通过”的主要风险。

## 已成功与尚未成功

### 已成功

- 针对 v68.3 的真实 RSS 瓶颈设计了具体、可执行的低内存构造；
- 新表示在完整 1,515 单元上保持 `2A+2A^T` 和全部精度门；
- tiled 与 dense 两条构造路径的独立复算最大差仅 `5.68e-14`；
- 不需要用更大网络、更多 exact calls 或放宽门槛救结果。

### 尚未成功

- 三区组、16-probe Pareto 只是开发筛选，不是正式资源结果；
- 尚未用与 v68.3 相同的 30 reference、330 timing worker、165 paired blocks
  重跑 Zero-K4 对照；
- 没有 fresh wall/RSS、独立公开反应流族、curved ray、相机标定或真实 BOST；
- p0 factor 是固定 geometry-compressed representation，不是 neural operator；
- 不能宣称算法突破、SOTA、广泛泛化或论文完成。

## 下一道唯一科学门

冻结 v70 fresh resource protocol，保持五条 fit trajectory、三档几何、
Zero-K4 对照、随机相邻完整区组与 v68.3 的全部阈值不变。正式实验必须同时满足：

```text
outer wall ratio p50              <= 0.90
outer wall ratio p90-higher       <= 1.05
each trajectory wall p50          <= 1.05
worker-self RSS p90 ratio         <= 1.05
worker-tree RSS p90 ratio         <= 1.05
pipeline RSS p90 ratio            <= 1.05
```

只有独立验证后的 fresh 资源门同时通过，才能把当前结论升级为“在公开
noise-free known-geometry straight-ray PoolFire proxy 上实现 matched-accuracy
的时间与内存共同收益”。之后才值得进入独立数据族与组内真实 BOST。

## 公开附件

- 机器摘要：`docs/nine_view_stencil_factor_stageb_v70_public_summary.json`
- 结果图：`assets/nine_view_stencil_factor_stageb_v70.png`
