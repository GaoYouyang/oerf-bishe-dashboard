# v269.1：缓存 Krylov 历史联合重正交化保持不确定，固定机制关闭

## 为什么做

v258 的热修正会先对已有 K13 测量方向做正交化，但随后追加的 restarted K1 步并不保证继续与旧历史正交。v269 因此做一个必要的经典控制：保留 K13 的 13 对场/投影方向和 restarted K1 的 1 对方向，在最终残差上联合重解一个 14 维最小残差问题。所有方向都已缓存，所以候选仍是 `15A+14A^T`，没有新增精确 `A/A^T`，也没有训练或超参数搜索。

## 执行修正

v269.0 在未读取候选指标或汇总前因审计定义错误而停止：它把只属于热初始化器的“active 区域零均值”要求错误地套到了最终 PCGLS 场。v269.1 只修正这一个执行审计，把边界支撑继续作为有效性门，把 active 区域均值保留为诊断量；候选、方向、求解、成本、控制、数据、指标、阈值和判决顺序均未改变，v269.0 输出也未复用。

## 独立结果

formal 的 `21/21` 项检查全部通过。完全独立第二实现通过 `28/29` 项，唯一失败是要求两套独立重建的观测数组逐位完全相等；实际最大归一化差为 `5.20e-16`。候选场最大相对差为 `2.49e-12`，候选残差最大归一化差为 `1.94e-12`，指标最大绝对差为 `4.76e-11`，物理残差重放误差为 `2.21e-16`，这些数值闭环均在各自冻结界内。

因为逐位相等检查失败，权威判决必须保持 `INCONCLUSIVE_INVALID_CASE19_KRYLOV_HISTORY_JOINT_REORTHOGONALIZATION_FRAME_ZERO_V269_1`。已经生成的性能计数只能作 post-open 诊断：候选绝对门为 `13/13`，但 K16-matched 为 `0/13`，13 个失败全部只来自 observation；observation matched p50 / p90-higher / worst 为 `1.15178 / 1.22366 / 1.35496`。这些诊断不能升级为正式 PASS 或 FAIL，也不会触发第二次修门或重跑。

## 判决与边界

按结果前冻结的 fail-closed 规则，固定 14 列缓存历史联合重正交化机制关闭。它没有获得完整序列、训练、wall/RSS、外部泛化、曲线射线或真实 BOST 资格，也不能用更大模型或 GPU 挽救。关闭的是这一种固定 solver-history 联合求解，不是整条 C 路线；后续只接受物理上真正不同且结果前可证伪的机制，或工况配对的真实二维双分量 BOST 数据。

`algorithm_breakthrough=false`，`paper_success=false`，`external_generalization=false`，`resource_speedup=false`，`real_bost=false`。

# v269.1: cached Krylov-history joint reorthogonalization remains inconclusive

## Why this was tested

The v258 heat correction is orthogonalized against the existing K13 measurement directions, but the subsequent restarted K1 step is not guaranteed to remain orthogonal to that history. v269 therefore runs a necessary classical control: retain the thirteen K13 field/projection pairs and the restarted-K1 pair, then jointly solve a 14-dimensional minimum-residual problem against the final residual. Every direction is cached, so the candidate remains at `15A+14AT`, with no additional exact operator call, training, or hyperparameter search.

## Execution amendment

v269.0 stopped before any candidate metric or summary was read because of an audit-definition error. It incorrectly applied the heat initializer's active-domain zero-mean requirement to the final PCGLS field. v269.1 changes only that execution audit: boundary support remains a validity gate, while active-domain mean remains a disclosed diagnostic. Candidate, directions, solve, costs, controls, data, metrics, thresholds, and decision order are unchanged, and no v269.0 output is reused.

## Independent result

Formal passes all `21/21` checks. The fully independent second implementation passes `28/29`; the sole failure is a requirement that the independently rebuilt observation arrays be bitwise identical. Their maximum normalized difference is `5.20e-16`. Maximum relative candidate-field difference is `2.49e-12`, maximum normalized candidate-residual difference `1.94e-12`, maximum absolute metric difference `4.76e-11`, and physical residual-replay error `2.21e-16`, all within their respective frozen numerical bounds.

Because bitwise observation equality fails, the authoritative verdict must remain `INCONCLUSIVE_INVALID_CASE19_KRYLOV_HISTORY_JOINT_REORTHOGONALIZATION_FRAME_ZERO_V269_1`. The generated performance counts are diagnostic only: the candidate clears `13/13` absolute cells but `0/13` K16-matched cells, with all thirteen misses coming only from observation. Observation matched p50 / p90-higher / worst are `1.15178 / 1.22366 / 1.35496`. These diagnostics cannot be promoted to a formal PASS or FAIL and will not trigger a second gate amendment or rerun.

## Verdict and boundary

Under the preregistered fail-closed rule, the fixed fourteen-column cached-history joint solve is closed. It receives no full-sequence, training, wall/RSS, external-generalization, curved-ray, or real-BOST authorization and will not be rescued with a larger model or GPU. This closes one fixed solver-history mechanism, not the C route. Continue only with a physically distinct preregistered mechanism or condition-matched real two-component BOST data.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
