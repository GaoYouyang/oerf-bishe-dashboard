# v48 判决失效记录

状态：`INVALIDATED_BEFORE_PUBLICATION`

## 结论

v48 原始运行不能支持 `FAIL_FIXED_BUDGET_GLOBAL_COUNTEREXAMPLE_S2_V48`，
也不能据此关闭五系数工程路线。正确证据等级是：

`INCONCLUSIVE_FIXED_BUDGET_GLOBAL_COUNTEREXAMPLE_S2_V48`

`algorithm_breakthrough=false`

这不是算法成功，也不是数学不可行证明。未跟踪的 v48 公共摘要、图和
renderer 草稿不得发布。

## 为什么原判决无效

冻结协议要求 bounded-negative 同时满足：

1. 候选中没有完整门通过者；
2. 至少 3/4 个精确梯度优化成功终止；
3. 每一个被评分点都有限且满足真值约束；
4. 所有冻结预算成立；
5. 独立复核同意。

原 runner 只检查四个优化终点的约束，却把 SLSQP 的全部中间求值加入
候选库。两名只读审计者独立指出这一漏检。随后用冻结输入和
`TruthConstraintModel` 重新计算 86 个 refinement 求值点：

| 项目 | 复核结果 |
|---|---:|
| refinement 求值点 | 86 |
| 低于 `-5e-8` 容差的点 | 42 |
| 最差点 | `refinement_3_eval_3` |
| 最差 normalized margin | `-4.213936483323799e-4` |
| 原报告候选数 | 166 |
| 原报告完整门通过数 | 0 |
| 成功终止 refinement | 4/4 |

因此第 3 个必要条件已经真实失败。即使候选库中没有 witness，也只能判
`INCONCLUSIVE`，不能判 bounded-negative。

## 额外审计问题

- validator 从报告自带的优化 history 重建候选 roster，无法发现 runner
  和报告同时漏记的求值点。
- pair、v40、v43、v44 与五方向数组没有完整内容哈希绑定；runner 和
  validator 可能在同一份漂移输入上保持一致。
- `rint` 分桶不是严格的绝对容差去重。该问题本次未触发，但实现不满足
  冻结文字合同。
- 数值差函数没有先拒绝 NaN；共同物理内核也只能称重放一致，不能称完全
  独立实现。
- 四个局部优化终点接近只说明这四条轨迹接近，不能推出“唯一 basin”。

## 仍然有效的观察

在本次固定 S2、固定五方向、固定 `[-2, 3]^5` 搜索与已记录的预算内，
记录候选中没有找到完整门 witness。这是一条有限范围的机制诊断，不是
连续五维空间的全局结论。

## 再运行前必须修复

1. 对 screen、每个 objective evaluation 和候选库成员逐点重算并记录
   finite、box 与 truth-feasibility。
2. bounded-negative 判决显式要求所有 scored points 通过上述检查。
3. validator 独立重跑四个冻结 SLSQP 起点，不能以报告 history 作为
   “应有 roster”。
4. 对输入文件和五个方向数组做内容级绑定。
5. 用真正的绝对 `L_inf <= 1e-10` 规则去重。
6. 所有 metric、ratio、gradient 与 margin 比较先 fail-closed 拒绝
   NaN/Inf。

完成这些修复并产生新的、独立通过的结果前，v48 不进入网页、论文结果
表或路线淘汰证据。
