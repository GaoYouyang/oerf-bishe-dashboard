# BLASTNet H2-air S2：v49 无结论，v50 固定候选表得到可信负结果

更新：2026-07-29

## 这轮要回答什么

v47 只在五个方向的局部邻域精确评分了 48 个候选，不能排除较远处仍有合格
组合。这里用一笔封顶预算继续回答一个很窄的问题：

> 在已经打开的 BLASTNet H2-air `phi=0.5` S2 快照上，当前五个方向能否找到
> 同时满足 field、gradient、observation 完整门的候选？

这是开封后的机理诊断，不是外部泛化检验，也不是部署算法。

## v49 为什么不能用于下结论

v49 先运行 1,664 次确定性全局搜索，再从四个起点调用 Powell，每个起点最多
160 次。runner 记录了 2,304 个唯一候选，诊断通过数为 0。

但是独立 validator 在第 48 个局部请求处发现 runner 与重放的请求几何不一致。
根因是目标值中小于 `1e-10` 的浮点差异改变了 Powell 的分支。四次 Powell 也都
没有报告收敛。于是：

```text
v49_scientific_decision=INCONCLUSIVE
```

“runner 看见 0 个候选”不能当作算法负结果。

## v50 如何修正

v50 保留相同的 1,664 次全局搜索和四个起点，但把数据依赖的 Powell 替换为
目标函数无关的固定局部候选表。每个起点恰好 160 个请求，包括中心点、轴向
幂次位移、两两方向位移和耦合位移；runner 与 validator 分别实现并重建这张表。

独立 validator 随后：

1. 重放全局优化器；
2. 独立重建四张固定局部表；
3. 对 2,304 个唯一候选逐一做精确 curved forward；
4. 重新计算三项指标和完整门；
5. 验证输入在评分前后未变化。

正式状态为：

```text
PASS_INDEPENDENT_RECOMPUTATION_FEASIBLE_GLOBAL_SEARCH_V50
```

## 实际结果

```text
总请求                         2,308
唯一精确评分候选               2,304
完整门通过候选                 0
field / Direct-K4 最小值       0.972222
gradient / Direct-K4 最小值    0.996404
observation / Direct-K4 最小值 1.045775
```

完整门中，2,282 个候选不劣于 Zero，2,122 个候选的 observation 不劣于
Direct-K3，2,253 个候选不被 Direct-K3 Pareto 支配；但是没有候选同时处在
Direct-K4 三指标的 `1.01` 包络内。

最有信息量的候选为：

```text
field / Direct-K4       0.983386
gradient / Direct-K4    1.010000
observation / Direct-K4 1.045775
```

它已经把 gradient 用满到安全边界，但 observation 仍比阈值高约 3.58 个百分点。
因此当前瓶颈不是“再把同五个系数调细一点”，而是现有 span 无法在该候选表中
同时提供梯度安全和 curved-observation 改善。

## 成功了什么，没成功什么

成功的是判别链：v49 的不可重放问题被识别并拒绝使用；v50 的全部候选由第二套
实现逐一精确重算，得到可信的固定候选表负结果。

没有成功的是算法目标：没有得到通过完整门的 warm-start 候选，没有同精度、
调用减少、wall/RSS 加速、真实 BOST 或论文成功。

固定候选表不是连续五维域的覆盖证明，所以五维路线没有被关闭，也不能宣称
数学无解。它只说明继续扩大同类随机或局部搜索的预期信息价值已经很低。

## 下一项真实实验

不训练五系数网络，也不继续堆同类搜索。下一步在 Direct-K3 的 curved residual
上构造第二个固定线性化 residual-adjoint 增广方向：

1. 先沿现有第五方向消掉可解释的 observation residual；
2. 对剩余 residual 做一次精确 VJP；
3. 投到粗网格并从现有五方向 span 中正交化；
4. 只在旧 span 外能量、伴随恒等式和完整门都通过后继续。

这会真正改变可达空间，而不是继续精调原有五个旋钮。它与投影
Gauss-Newton / augmented Krylov 思路有关，但不能命名为完整 Hessian 或二阶求解器。

```text
fixed_candidate_roster_negative=true
five_space_engineering_route_closed=false
mathematical_nonexistence_proven=false
stage_2_authorized=false
algorithm_breakthrough=false
paper_success=false
```
