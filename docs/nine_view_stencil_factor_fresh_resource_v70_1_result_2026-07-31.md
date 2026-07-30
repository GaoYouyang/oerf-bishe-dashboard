# v70.1：低内存 tiled p0 速度通过，RSS 仍未通过

日期：2026-07-31  
独立验证：`PASS_INDEPENDENT_VALIDATION_TILED_EXACT_P0_FRESH_RESOURCE_V70_1`  
科学判决：`FAIL_TILED_EXACT_P0_FRESH_RESOURCE_STAGE_C_V70_1`

## 一句话结论

v70 已经在五条 PoolFire fit 轨迹、三档已知九视角几何、全部 1,515 个单元上
守住冻结精度门。v70.1 用 30 个 reference worker、330 个 timing worker 和
165 个随机相邻区组补测 fresh-process 资源：

```text
Tiled p0 q8-K1 / Zero-K4

outer wall p50 / p90 / worst
  0.670558 / 0.677727 / 0.701055        PASS

worker-self RSS p50 / p90 / worst
  1.086393 / 1.134775 / 1.171482        FAIL
sampled worker-tree RSS
  1.084989 / 1.132203 / 1.168128        FAIL
sampled pipeline RSS
  1.041881 / 1.064684 / 1.096941        FAIL
```

所以 tiled p0 把旧 v68.3 约 `34.5%` 的 wall 收益基本保住了，但没有把内存
高尾压回冻结的 `1.05` 门。Stage C 要求时间与三类内存同时通过，整体必须写
**FAIL**：

```text
public_proxy_fresh_wall_gate_pass=true
public_proxy_sampled_rss_gate_pass=false
stage_c_pass=false
algorithm_breakthrough=false
paper_success=false
```

## 1. 正式比较没有改变什么

候选与 v70 Stage B 完全相同：

```text
observation
-> exact-stencil tiled rank-8 p0 factor
-> 4 cheap detector CR steps
-> exact A^T lift
-> unchanged exact CGLS K1

exact budget: 2A + 2A^T
```

对照仍是 zero-start exact CGLS K4：

```text
exact budget: 4A + 4A^T
```

每个 fresh worker 都重新建立 exact operator 和 candidate factor，再处理同一
轨迹的 101 帧。Python 启动、factor setup、全部 cheap/exact actions 与退出均
进入 outer wall；candidate-specific setup 没有藏到父进程。

## 2. 速度：五条轨迹稳定通过

全局绝对时间与比值：

| 统计量 | tiled p0 q8-K1 | Zero-K4 | ratio |
|---|---:|---:|---:|
| outer wall p50 | 10.9114 s | 16.2845 s | 0.670558 |
| outer wall p90 | 11.0163 s | 16.3705 s | 0.677727 |
| outer wall worst | 11.4228 s | 17.3180 s | 0.701055 |

五条轨迹的 wall p50 ratio 为：

| trajectory | p50 ratio |
|---|---:|
| p14-s05 | 0.668524 |
| p22-s03 | 0.671470 |
| p33-s01 | 0.670024 |
| p45-s05 | 0.671090 |
| p58-s03 | 0.669547 |

因此 `2A+2A^T` 对 `4A+4A^T` 的理论调用减半，确实在当前 Mac、固定
`32x16x16`、noise-free known-geometry straight-ray proxy 上转化成稳定的
约 `33%` 端到端 wall 收益。

## 3. 内存：改善明显，但高尾仍失败

| 口径 | candidate p50 | candidate p90 | ratio p50 | ratio p90 | 门 |
|---|---:|---:|---:|---:|---:|
| worker-self RSS | 559.6 MB | 581.9 MB | 1.086393 | 1.134775 | 1.05 |
| sampled worker-tree RSS | 569.4 MB | 591.7 MB | 1.084989 | 1.132203 | 1.05 |
| sampled pipeline RSS | 1176.9 MB | 1203.6 MB | 1.041881 | 1.064684 | 1.05 |

相对 v68.3 的 worker-self p90 ratio `1.384474`，v70.1 已降到
`1.134775`；这证明 exact-stencil reuse 和 tiled p0 确实消除了大部分 setup
高水位。但三类 p90 仍分别超过门 `8.48`、`8.22` 和 `1.47` 个百分点。五条
轨迹的 worker-self p50 也全部高于 1.05，不是单个异常 worker。

## 4. 独立验证

独立 validator 没有导入正式 controller/worker，完成了：

1. 从原始 pair truth 重建 1,515 个 observation，最大差 0；
2. 独立重放 30 个 reference、共 3,030 帧；
3. 30/30 输出摘要与实际 `A/A^T` 账完全一致；
4. 核对 360 个 worker 串行、不重叠；
5. 独立重算 165 个配对区组与 903 个聚合数字，最大差均为 0；
6. 核对 v70 Stage B 父证据仍是 1,515/1,515 PASS；
7. 确认 source、pair、observation 与 formal payload 在验证前后不变。

RSS 原始逐采样轨迹没有持久化，validator 核对的是正式回执中的覆盖摘要与聚合，
因此 sampled tree/pipeline 仍是下界；但下界自己已经越过 1.05，足以判 FAIL。
正式与独立程序仍共享冻结的 v66 数值核心：

```text
independent_resource_recomputation=true
end_to_end_numerical_independence_proven=false
```

## 5. v71：tile2 接近，但不能临时放宽门

正式失败后先做了 fresh import attribution。基础 worker、只导入 candidate 模块
与只导入 `numpy.linalg` 的高水位只差约 3-4 MB，说明主要内存差不是 Python
模块本身，而是 factor 构造工作区与 allocator high-water。

随后只把 detector-v tile 从 4 改为 2：

```text
declared dominant workspace  12 MiB -> 6 MiB
persistent factor state      unchanged at 5,899,392 bytes
factor algebra max abs diff  <= 1.56e-13 across three geometries
```

在一条已打开的 101 帧 observation stratum 上，结果前冻结六个 fresh 配对区组：

| ratio | p50 | p90-higher | 门 |
|---|---:|---:|---:|
| outer wall | 0.661271 | 0.678067 | 0.90 / 1.05 |
| worker-self RSS | 1.040516 | 1.060203 | 1.05 / 1.05 |

12 份回执的两臂摘要均稳定，实际调用账正确；独立程序重新组合六个区组与
`method="higher"` 分位数，摘要最大差为 0。内存中位通过，但高尾超过门约
`1.02` 个百分点，所以准确状态是：

```text
PASS_INDEPENDENT_RECOMPUTATION_TILE2_RESOURCE_PROBE_V71
FAIL_TILE2_RESOURCE_HEADROOM_PROBE_V71
```

这只是 post-open development probe，不是正式 Stage B/Stage C。按结果前规则，
tile-size tuning 到此关闭：不运行另一批 330 worker，也不靠放宽 1.05 门宣布
成功。

## 6. 现在能说什么

### 已经成立

- v70 在 1,515/1,515 单元守住冻结的 field/gradient/observation 精度门；
- v70.1 把完整精确调用从 `4A+4A^T` 减为 `2A+2A^T`；
- fresh outer wall 在五条轨迹稳定下降约 33%；
- exact-stencil tiled p0 相比旧构造明显降低了内存伤害；
- 两轮资源数字均由独立程序从原子回执重算。

### 尚未成立

- v70.1 的三类 RSS p90 均未通过 1.05；
- v71 tile2 高尾仍失败，不能进入完整 Stage B；
- 当前 q8 是固定几何压缩表示，不是 neural operator；
- 没有独立公开反应流族、curved ray、相机标定或真实 BOST；
- 没有算法突破、广泛泛化或论文成功。

## 7. 下一条有价值的路线

继续缩 tile 已经缺少足够 headroom。真实 BOST 装置的相机几何在标定后固定，
因此下一门应把成本拆成两张不可混写的账：

1. **cold calibration cost**：一次性构造 factors 的 wall、peak RSS 与磁盘体积；
2. **online reconstruction cost**：只加载校准绑定 factors，处理 observation stream，
   与同样预先绑定几何的 Zero-K4 比较 wall/RSS；
3. 给出以帧数或序列数表示的摊销临界点，绝不把离线构造成本抹掉。

如果在线账仍不能同时通过 wall/RSS，就关闭当前 factor warm-start 资源主张；
如果通过，也只能称固定标定几何下的在线重建收益，仍须另做独立公开族与组内
真实 BOST。

## 公开附件

- 机器摘要：`docs/nine_view_stencil_factor_fresh_resource_v70_1_public_summary.json`
- 结果图：`assets/nine_view_stencil_factor_fresh_resource_v70_1.png`
