# N2-PVGR N5-D1 paired residual 预注册

日期：2026-07-18

## 问题边界

N4.1 结束后，只有两个 `smooth-s1871 / orientation_58 / narrow` controls 的
H1024-H2048 matched-residual relative-L2 没过 `0.125%`。D1 只问一个更窄的问题：

> 把每个弧长中点上的 curved/straight integrand 先相减，再求和，能否解释 N4.1 的残差门失败？

中点求积是线性的，所以这种重排必须与冻结 N4 observable 在精确算术下等价。它不能换射线方程、
梯度模板、积分节点、detector basis 或查询预算。若结果不同，只能来自浮点累加顺序。

## 选择与非独立性

- 选 N4.1 两个最终失败格及其同 field、同 stress 的 wide controls，共四格；
- 只跑 H1024/H2048；
- 这是明确的 post-N4 mechanism audit，不是 fresh test，不产生泛化证据；
- 额外保留 constant、弱非对称 Gaussian 和径向 Gaussian 三个 toy contract。

## 五种累加顺序

1. `raw_separate_subtraction`：两个 midpoint 总和相减；
2. `paired_naive`：逐节点相减后调用普通 sum；
3. `paired_pairwise`：逐节点相减后做固定二叉树求和；
4. `paired_neumaier`：逐节点相减后做 Neumaier 补偿求和；
5. `separate_neumaier_subtraction`：两个积分分别补偿求和后再相减。

五种方法共享同一条 curved trace 和同一批 curved/straight midpoint field queries。paired route 的逻辑
查询仍为每 ray-step 42 个标量场点；用于核对冻结 route 的额外 14 个查询单列为 audit overhead。

## 预冻结门

| 门 | 阈值 |
|---|---:|
| constant field 最大绝对输出 | 0 |
| 弱场二阶 finite-difference relative-L2 | 3% |
| 径向场 90° equivariance relative-L2 | `2e-10` |
| 新内核与冻结 curved/straight route relative-L2 | `5e-12` |
| 重现 N4 H1024-H2048 指标绝对差 | `1e-12` |

解释门也在结果前冻结：对两个失败格，若所有四种非 raw 累加与 raw 的 L2 差都小于真实
H1024-H2048 raw refinement L2 差的 1%，判为“累加顺序太小，不能解释 N4 floor”；若两个失败格
都至少有一种达到 10%，判为“可能解释”；其余一律 `INCONCLUSIVE`。

## 允许与禁止

允许 D1 通过等价性后另开 D2 H4096/H8192 截断误差探针。禁止改变 N4.1 门、把 D1 当新 reference、
开放 field JVP/VJP、做三维重建、训练神经算子，或写成真实数据/泛化/论文成功。

## 预期机器判决

- `D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR`
- `D1_ACCUMULATION_ORDER_PLAUSIBLY_EXPLAINS_N4_FLOOR`
- `D1_ACCUMULATION_ORDER_INCONCLUSIVE`
- `D1_CONTRACT_OR_REPRODUCTION_FAILED_CLOSED`

无论哪一种，都不授权算法或论文主张。
