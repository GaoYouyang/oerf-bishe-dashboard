# v41：四个 Krylov 增量逐帧调权，仍没有找到通过候选

## 先说结论

这轮真实执行了，不是方案草图。

我们把 v40.2 的 straight-ray CGLS 修正拆成四个逐步增量，不再让它们都固定乘
1，而是对每个已经开封的 BLASTNet `phi=0.5` 快照，单独用 observation residual
选择四个权重。每帧做 3 个确定性起点的有界 L-BFGS-B，权重范围固定为
`[-2, 3]`。

最终仍是：

```text
snapshot_pass_count = 0 / 4
frozen_v40_gate_pass = false
algorithm_breakthrough = false
```

四方向确实比原来的一维幅度更灵活，但没有找到同时守住 field、gradient 和
observation 的候选。

## 为什么做这一步

v40.2 已经排除了两件事：

1. PoolFire 上冻结的 SARC-K3-M4 不能零适配迁移到另一类燃烧场；
2. 失败不能只靠一个全局 correction 增益修复。

还剩一个关键歧义：是原修正方向完全不对，还是把四步 CGLS correction 分开调权
就足够？

v41 专门回答这个问题。四个 basis 都由当前 observation residual 和冻结的
straight-ray operator 生成；权重优化不读取 truth。truth 只在权重选完后用于
field/gradient 评分。

## 真实数值

相对 Direct-K4 的逐帧比值如下，冻结兼容线是 `1.01`：

| 快照 | field / K4 | gradient / K4 | observation / K4 | 判决 |
|---|---:|---:|---:|---|
| S1 | 0.976484 | 1.016197 | 1.019140 | FAIL |
| S2 | 0.977307 | 1.031789 | 1.019967 | FAIL |
| S3 | 0.974200 | 1.027001 | 1.004856 | FAIL |
| S4 | 0.977260 | 1.014603 | 1.056837 | FAIL |

四帧都进一步降低了 field error，幅度约为 Direct-K4 的
`97.42%-97.73%`；但 gradient 全部高于兼容线，observation 有三帧高于兼容线。
S2 的 gradient 甚至比 Zero 高约 `0.664%`。

跨四帧中位数：

| 方法 | field | gradient | observation |
|---|---:|---:|---:|
| 四方向候选 | 0.955901 | 0.989316 | 0.310794 |
| Direct-K4 | 0.979420 | 0.971054 | 0.301608 |

这不是“全面更差”，而是一种清楚的冲突：四方向候选换来了更低 field error，
却牺牲了 gradient 与 observation。

## 独立复算

独立验证器没有导入正式诊断器，重新做了：

1. 检查 v40.2 确实是已封存的外部失败结果；
2. 重建四步 CGLS increments 与 `4A + 4A^T` straight ledger；
3. 重新执行全部 12 个 L-BFGS-B 起点；
4. 重新生成候选场；
5. 重算 Direct-K3、Direct-K4、Zero、候选的全部指标和完整冻结门。

结果：

```text
PASS_INDEPENDENT_RECOMPUTATION_KRYLOV_INCREMENT_SPAN_V41
maximum metric difference       = 3.58e-15
maximum optimizer weight diff   = 0
maximum candidate field diff    = 0
```

四帧最终被选中的最佳起点都报告收敛；但 S3 有一个没有被选中的起点未收敛。
因此正式状态保守保留为
`POST_OPEN_FOUR_INCREMENT_SPAN_SEARCH_INCONCLUSIVE`。更精确的事实表述是：

```text
NO_PASSING_CANDIDATE_FOUND_UNDER_OBSERVATION_OBJECTIVE
```

它不等于“四方向空间在数学上绝不可能通过”。

## 成本边界

每帧构造四个 straight-ray basis 的确只需要：

```text
4A + 4A^T
```

但这不是完整 v41 成本。为了逐帧找 observation 最优权重，正式诊断总共执行了
`1425` 次 curved forward。它比真正候选算法昂贵得多，只能用于开封后的机理
诊断：

```text
online_deployable_as_run = false
same_cost_or_speed_claim_authorized = false
resource_gate_authorized = false
```

独立验证覆盖 v41 的 CGLS 编排、优化、候选场、指标和判门；正式端与验证端仍共享
冻结的 curved forward、插值和几何核，所以不能写成“端到端独立物理实现”。

## 创新性审计

进一步检索后，`Krylov basis + learnable mixing` 本身不能作为创新点：

- [Reconstruct Anything Model](https://arxiv.org/abs/2503.08915) 已使用
  Krylov Subspace Module 学习 `A^T A` 型 basis 的组合；
- [CASSI CG unrolling](https://arxiv.org/abs/2607.20138) 已把复杂光学 forward、
  CG 和条件化学习结合；
- [FCG-NO](https://proceedings.mlr.press/v235/rudikov24a.html) 已用神经算子做
  flexible CG 的非线性预条件；
- [Deep Conjugate Direction Method](https://proceedings.mlr.press/v202/kaneda23a.html)
  已学习更有效的共轭方向。

仍可能有价值的窄问题不是“四个可学习权重”，而是：

> 在 BOST 的 straight-to-curved 物理失配下，能否用每个样本自己的低保真 CGLS
> increments 构造 correction subspace，再由部署可见 observation 选系数，并在
> 固定高保真调用预算和 fail-closed 回退下通过 matched-accuracy、harm、wall 与
> RSS 的联合验收。

截至本轮，这个命题还没有成功。

## 下一步由结果决定

直接训练系数网络现在缺少依据。下一项有价值的实验是同一四方向空间内的
truth-aware constrained oracle feasibility：

- 把 field 与 gradient 的冻结门作为约束；
- 在约束内最小化 curved observation residual；
- 若连 oracle 都找不到通过点，关闭当前四方向空间；
- 只有 oracle 能通过，才证明“basis 够、selector 不够”，这时训练 observation-only
  coefficient predictor 才合理。

`phi=0.5` 已经开封，以上仍只能是机理开发。新的外部等价比与真实 BOST 必须继续
封存，不能拿来反复挑模型。
