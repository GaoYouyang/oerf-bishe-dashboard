# v77 GSLB8 容量结果：只有 7/75 个单元找到完整见证

> 日期：2026-07-31  
> 科学判决：`NO_ALL_CELL_GSLB8_WITNESS_UNDER_FROZEN_SEARCH_V77`  
> 独立验证：`PASS_INDEPENDENT_RECOMPUTATION_GSLB8_V77`  
> 突破状态：`algorithm_breakthrough=false`

## 先说结论

v76 已经严格证明，两个全局系数构成的 `span{h,n}` 在 17/75 个 Case 3
单元里根本没有合格答案。v77 因此没有继续放大同类系数网络，而是第一次把
修正写成空间变化的三维场：

```text
x0 = h + U_g,8 a
x1 = one unchanged exact CGLS step from x0
```

`U_g,8` 是只由冻结几何构造的 8 个低可观测 spline 模态；truth-aware oracle
只负责在每个已经开封的 development 单元里寻找最有利的 8 个系数。最终：

```text
完整八门见证找到       7 / 75
冻结搜索下未找到      68 / 75
数值不确定             0 / 75
```

三档几何都远未达到“25/25”：

| 几何 | 找到见证 | 冻结搜索下未找到 | best max gate p50 / p90 / worst |
|---|---:|---:|---:|
| F12+ | 3/25 | 22/25 | 0.03333 / 0.07115 / 0.09414 |
| F15+ | 2/25 | 23/25 | 0.02875 / 0.04769 / 0.08964 |
| F30+ | 2/25 | 23/25 | 0.01295 / 0.02402 / 0.04624 |

按结果前规则，GSLB8 现在关闭，只授权早已预注册的 GSLB32 容量测试。神经
训练、Case 4/6、wall/RSS 和真实 BOST 均未获授权。

## 为什么先做 oracle，而不是直接训练网络

如果连看得见真值的 oracle 都无法在固定表示里找到合格场，那么只看 observation
的网络更不可能稳定完成同一任务。先测表示容量，可以用很小的成本排除“网络
再大一点也许会好”的无边界试错。

v77 的 parent space 是 `8×4×4` 三线性 spline 控制格，经边界置零和零均值
gauge 后得到 127 维空间。对每档冻结几何，用同一个 generalized eigenproblem
按照“单位内部梯度能量对应的 observation 能量”从低到高排序，再取前 8 个
模态：

```text
C_g v = lambda G v
U_g,8 = the first eight geometry-only low-observability modes
```

这些不是精确 nullspace，也没有“自动数据一致”的保证。它们只是几何决定的
低可观测方向，目的是让空间修正尽量少破坏 observation，同时补当前 anchor
缺失的局部结构。

## 精确调用账没有增加

候选仍保持和 v76 相同的在线精确预算：

```text
loaded-q8 cheap dual proposal z
h = A^T z
x0 = h + U_g,8 a
r0 = y - A x0
s0 = A^T r0
t0 = A s0
x1 = x0 + alpha s0
```

因此每个部署单元是：

| 方法 | exact A | exact A^T |
|---|---:|---:|
| GSLB8 exact-K1 | 2 | 2 |
| Zero-K2 同调用对照 | 2 | 2 |
| Zero-K4 精度对照 | 4 | 4 |

oracle 搜索只复用已经形成的数组，不增加精确 `A/A^T`。但 oracle 看了真值，
所以这只是表示容量诊断，不是可部署算法。

## 真正卡在哪里

GSLB8 对 field 和 observation 很有效，却没有补足同调用 Zero-K2 的梯度：

| 冻结门 | 通过数 |
|---|---:|
| field / Zero-K4 harm | 75/75 |
| field / Zero-K2 equal-call | 75/75 |
| full-gradient / Zero-K4 harm | 56/75 |
| full-gradient / Zero-K2 equal-call | 27/75 |
| interior-gradient / Zero-K4 harm | 74/75 |
| interior-gradient / Zero-K2 equal-call | 7/75 |
| observation / Zero-K4 harm | 75/75 |
| observation / Zero-K2 equal-call | 75/75 |

相对 Zero-K2 的整体 ratio 更直接：

| 指标 | p50 | p90-higher | worst |
|---|---:|---:|---:|
| field | 0.96778 | 0.98582 | 0.99682 |
| full-gradient | 1.00387 | 1.01105 | 1.02441 |
| interior-gradient | 1.01120 | 1.02409 | 1.04601 |
| observation | 0.62988 | 0.69804 | 0.74269 |

也就是说，8 模态 correction 明显降低了 observation residual，也让 field
优于同调用 K2；但它把可用容量主要花在“看起来更符合投影”上，未能稳定恢复
局部三维梯度。对 BOST 来说，这个失败很有物理意义：折射率梯度正是位移图
形成的核心，不能用 field 平均误差或 observation residual 的改善来掩盖。

## “68 个 negative”不等于数学不可行

这点必须和 v76 区分：

- v76 的 17 个失败有精确 dual certificate，证明二维凸交集为空；
- v77 的系数经过一轮 CGLS 后，门对系数是非凸的；
- v77 只说明结果前冻结的 12 个起点、条件触发重启、系数球和 SLSQP 搜索
  没有在 68 个单元里找到见证；
- 它不证明 GSLB8 的所有系数在数学上都不可能通过。

不过，实验合同在运行前已经规定“75/75 才允许训练”。在 7/75 的结果下继续
训练 GSLB8 predictor 没有合理依据，因此工程路线仍应关闭。

## 独立复算与数值修复

独立 validator 没有导入正式 runner、optimizer helper 或 spline/mode helper，
重新构造三档几何模态，并重跑每个单元的 12 个声明起点及所有条件触发重启。
最终：

```text
formal / independent 最大 metric 差       1.7764e-14
最大 gate 差                              1.2023e-11
稳定 projector spectral distance          2.0165e-13
formal payload unchanged                  true
raw payload unchanged                     true
```

本轮公开披露了三次 post-open 修复：

1. 只对满足严格条件的有限非成功 endpoint 增加一次同设置重启；
2. provenance gate 比较改用预先存在的 `1e-9` 复算容差；
3. 把存在消减误差的 `sqrt(1-sigma^2)` projector 距离，换成数学等价的
   对称正交残差谱范数，容差仍保持 `1e-9`。

每次修复后都从头重跑 formal 和独立 validator，没有复用旧结果行。数据、表示、
rank、系数半径、起点、八个门、controls 和调用账均未改变。最终 validator
通过，但这些修复使 v77 只能作为已开封 development 诊断，不能承担外部确认。

## 下一步为什么是 GSLB32

GSLB8 的结果支持两个判断：

1. 空间变化 correction 的方向是有信息的，因为 observation 和 field 显著改善；
2. 8 个低可观测模态不足以同时补齐三维梯度，容量可能过窄。

因此下一步不是训练 FNO、UNO 或 U-Net，而是运行同一 127 模态排序的预注册
32 模态前缀。这样只改变表示容量，不改数据、物理算子、精确调用账、八门或
refinement：

```text
GSLB8  ->  GSLB32
2A+2A^T unchanged
same 75 development cells
same eight matched-accuracy gates
```

如果 GSLB32 仍不能 75/75，通过再大的 predictor 也没有意义；如果它能通过，
才值得冻结一个只看 observation 的最小系数预测器。

## 证据边界

```text
gslb8_all_cell_headroom=false
gslb8_closed_under_frozen_search=true
gslb32_capacity_stage_authorized=true
deployment_rule_found=false
neural_training_authorized=false
resource_stage_authorized=false
case4_or_case6_opened=false
external_generalization=false
real_bost=false
algorithm_breakthrough=false
paper_success=false
```

脱敏摘要与图表：

- `docs/nine_view_geometry_spline_low_observability_v77_public_summary.json`
- `assets/nine_view_geometry_spline_low_observability_v77.png`
