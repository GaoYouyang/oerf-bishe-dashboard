# BLASTNet H2-air S2：冻结六维真值可行候选表仍为 0 / 2,312

更新：2026-07-29

## 这轮为什么值得做

v51 已经证明第六个 residual-adjoint 方向不是数值重复项，而且部署可见的
observation-only 搜索确实使用了它。但 v51 的最终点虽然把 observation 推进
`1.01` 门内，gradient / Direct-K4 仍为 `1.030041`。

当时还不能判断失败来自哪里：

1. 固定六方向空间里其实有合格点，只是 observation-only selector 没找到；
2. 固定六方向空间本身缺少同时满足 field、gradient、observation 的点。

v52 因而只在已打开的 BLASTNet `phi=0.5` S2 快照上做表示封顶诊断，不训练网络、
不测速，也不打开新数据。

## 候选表怎样在看 observation 结果前冻结

候选场固定为：

```text
x(w) = gauge(Direct-K3 + sum_i w_i d_i),  i = 1,...,6
w_i in [-2, 3]
```

`d1-d4` 是四步 straight-CGLS 增量，`d5-d6` 是已经独立验证的两个
curved residual-adjoint 方向。六个方向和全部父证据在生成候选表前冻结。

答案可见部分只用于构造 field/gradient 真值可行域：

```text
field <= min(1.01 * Direct-K4, Zero)
gradient <= min(1.01 * Direct-K4, Zero)
```

这个可行域中的请求由固定规则组成：

| 来源 | 请求数 | 规则 |
|---|---:|---|
| 全局 Sobol 射线 | 2,048 | 512 个固定方向，各取真值可行射线的 0.25/0.5/0.75/1.0 |
| v51 S6 邻域 | 256 | 64 个后续 Sobol 方向，正负两侧，各取 0.5/1.0 步长 |
| 历史锚点 | 8 | Direct-K3、v43/v44/v47/v50/v51 的固定点 |
| 合计 | **2,312** | 不去重、不早停 |

v47 与 v50 历史锚点重复，所以六维权重的唯一值为 `2,311`；协议保留请求重数，
runner 和 validator 都必须执行全部 `2,312` 次 curved forward。

本轮局部中心来自同一已打开快照上 v51 的 observation-selected S6，因此 v52
不是 observation-independent 实验。它禁止看 v52 结果后加点，但不能冒充外部
泛化或全局盲测。

## 正式运行与独立复算

完整 roster 在第一次 curved forward 前写入并密封。正式 runner 与 validator
分别完成：

```text
runner       2,312 F + 0 JVP + 0 VJP
validator    2,312 F + 0 JVP + 0 VJP
roster 最大数值差       0
score 最大数值差        0
source/input 前后复核   PASS
```

validator 不导入 v52 runner，独立重建六个方向、真值约束、Sobol roster、每个候选
场、三项指标、完整门和最终判决。两者仍共享冻结的 v42 真值约束类以及 v44 的
curved-forward、几何和部分数值内核，所以这不是外部团队独立物理实现。

在冻结提交前，独立红队还发现并修复了两个 P1：

1. 非有限数可能绕过最大数值差比较；
2. 长跑结束后没有重新哈希完整 source/input identity。

正式结果来自修复后的提交；任何 `NaN/Inf` 或输入前后变化都会 fail closed。

## 结果

正式判决：

```text
NO_WITNESS_IN_FROZEN_TRUTH_FEASIBLE_ROSTER_S2_V52
complete-gate passes    0 / 2,312
robust passes           0 / 2,312
```

候选通过各个单项检查的数量为：

| 检查 | 通过数 |
|---|---:|
| 三指标均不劣于 Zero | 1,867 |
| 不被 Direct-K3 Pareto 支配 | 1,432 |
| observation 不劣于 Direct-K3 | 778 |
| 三指标均在 Direct-K4 的 `1.01` 包络内 | **0** |

完整门余量最大的候选相对 Direct-K4 为：

| field | gradient | observation | 最小门余量 | 完整门 |
|---:|---:|---:|---:|---|
| 0.985041 | 1.010000 | 1.038412 | -0.028412 | FAIL |

这个点在 field 上有余量，gradient 几乎正好用完 `1.01` 额度，但 observation
仍比允许线高约 `2.84` 个百分点。全部 `2,312` 个真值可行请求中，最小
observation / Direct-K4 也只是 `1.038412`。

## 这改变了什么

v51 的原始 S6 点能把 observation 推进门内，却以 gradient=`1.030041` 为代价。
v52 把候选限制回 field/gradient 真值可行域后，最佳点的 gradient 回到 `1.01`，
observation 又退到 `1.038412`。这把冲突进一步定位为：

> 当前固定六方向 residual-only 表示在“梯度安全”和“曲线观测拟合”之间存在明显
> 张力，至少在这张预先冻结的 2,312 请求候选表中没有同时满足二者的点。

因此当前没有合格标签可供六系数 selector 学习。此时训练 FNO、DeepONet、MLP
或更大的系数网络，只会学习一个尚未证明存在合格输出的目标族。

## 成功了什么，失败了什么

成功的是研究决策：

1. 对 2,312 个预先冻结的请求各评分两次：runner 与 validator 合计执行
   4,624 次 curved forward，并得到逐请求一致的结果；
2. 证明这张有限候选表没有完整门 witness；不能据此排除更密的有限表或连续优化
   找到窄可行口袋；
3. 找到当前最紧的数值瓶颈是 observation，而不是 field；
4. 在连续约束 oracle 完成前，暂停没有合格标签支撑的六系数网络训练。

失败的是算法目标：

1. 没有三指标同精度 warm start；
2. 没有调用减少、wall 或内存优势；
3. 没有外部泛化、真实 BOST 或论文成功；
4. 没有证明连续六维无解，更没有证明全局数学不可能。

这是可信的有界负结果，不是突破性进展：

```text
algorithm_breakthrough=false
```

## 下一项只做最后一个连续域判别

有限候选表仍可能漏掉很窄的连续可行口袋。因此在正式放弃固定六方向前，只允许再做
一次结果前冻结的六维连续约束 oracle：

1. 保持同一六方向、权重盒、field/gradient 二次约束和完整门；
2. 使用 curved observation 的精确 VJP 得到六维目标梯度；
3. 固定多起点、停止规则、调用上限和独立请求轨迹重放；
4. 不允许根据结果临时增加起点或放宽 `1.01`；
5. 若仍没有 witness，停止六系数 selector，转向显式改变表示的新方向；
6. 若存在 witness，才研究部署可见的 gradient-aware surrogate 与回退门。

新的表示方向应直接处理梯度安全，而不是继续机械追加同类 residual-adjoint：
例如受控 Sobolev/TV 方向、局部多尺度场增量，或在粗梯度与 curved residual 的联合
目标下生成的新 Krylov 方向。它们仍必须先有 oracle headroom，再授权神经训练。

```text
finite_roster_negative=true
continuous_six_space_nonexistence_proven=false
six_coefficient_selector_training_authorized=false
matched_accuracy=false
speedup=false
external_generalization=false
real_bost=false
paper_success=false
algorithm_breakthrough=false
```
