# v147：把局部邻域放宽到 32 维真值跨度，仍过不了完整轨迹门

## 先说结论

v146 证明固定八近邻预测不行，但还留下一个合理疑问：可观测特征也许已经找到了正确的局部样本，只是逆距离加权不会把这些样本组合成目标。v147 因此没有训练网络，而是在同一 `20` 个冻结哨兵上做更强的真值可见上限诊断：先只用部署可见的 155 维方向局部特征选择最近样本，再允许 CFD 真值为已经封存的 `8` 或 `32` 个 action 求最佳正交跨度投影。

扩大到 `32` 维确实带来明显改善。跨轨迹从 `0/20` 提高到 `14/20`，同轨迹上限从 `1/20` 提高到 `18/20`。但冻结合同要求 `20/20` 哨兵和 `5/5` 完整轨迹尾部门同时通过；实际只有 `1/5` 与 `2/5` 轨迹通过。因此独立复算后的科学判决仍是：

`FAIL_LOCAL_SPAN_CAPACITY_V147`

这说明 v146 的失败不只是逆距离权重太简单。当前样本级方向局部表示即使允许 `K=32` 的真值最优线性重组，也不能稳定覆盖完整轨迹。继续加大预测器或租 GPU 没有科学依据。

## 冻结问题

- 已开封 PoolFire 池：五条轨迹、`3700` 个可用单元；本轮只读结果前冻结的 `20` 个哨兵；
- 相机数：`5 / 7 / 9 / 12`，候选必须匹配查询的相机数；
- 邻域选择：只看部署可见 observation / geometry 派生的 `155D` 方向局部特征；
- 目标 action：`96D`，在候选顺序封存后才由 CFD 真值读取；
- 候选池：跨轨迹每查询 `740` 个，同轨迹上限每查询 `184` 个；
- 比较：nearest、IDW-8、IDW-32、truth-aware span-8、truth-aware span-32；
- 每哨兵门：尺度不变 relative-L2 `<=0.45` 且 cosine `>=0.90`；
- 每轨迹四哨兵 p90-higher `<=0.35`；
- 训练参数 `0`，新增精确调用 `+0A/+0A^T`，Stage B 与物理重放均未运行。

真值跨度只是 post-open capacity oracle，不是部署算法。它只能回答“当前局部候选的线性包络有没有足够容量”，不能在未来测量上直接使用。

## 实际数字

| 方法 | 哨兵通过 | 轨迹尾部通过 | 误差中位数 | 误差 p90 | 最坏误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cross span-8 oracle | 0/20 | 0/5 | 0.69022 | 0.88182 | 0.88928 |
| cross span-32 oracle | 14/20 | 1/5 | 0.34630 | 0.62727 | 0.67313 |
| within span-8 oracle* | 1/20 | 0/5 | 0.62304 | 0.79854 | 0.80553 |
| within span-32 oracle* | 18/20 | 2/5 | 0.30284 | 0.44817 | 0.44951 |

`*` 同轨迹方法读取同一条已开封轨迹的其他时刻，只是不可部署的机制上界。

`cross span-32` 的五条轨迹 p90 为 `0.43970 / 0.36087 / 0.32701 / 0.67313 / 0.48745`，只有 p33 通过。`within span-32` 为 `0.38282 / 0.25030 / 0.30476 / 0.44951 / 0.38286`，只有 p22 与 p33 通过。p45 是最明确的剩余瓶颈；即使同轨迹、32 个基向量、真值最优投影也仍有两个单元的 cosine 低于 `0.90`。

nearest 与所有 IDW control 都是 `0/20`，所以简单地把八近邻增加到 32 个并做加权平均不能解释改善；改善来自允许有符号线性重组，但其容量仍不足。

## 相对邻域冲突

对每个查询，v147 检查距离最近的 `5%` 候选。跨轨迹 `740/740` 个、同轨迹 `200/200` 个近邻都不满足同一 action 兼容门，`20/20` 查询都找不到一个兼容的单独近邻。

这只能称为**相对邻域冲突**：在当前标准化距离的最近区域里，特征相近并不意味着 action 相容。它不是精确 feature collision，也没有证明所有可观测表示在数学上都不可辨识。

## 独立复算

第二实现不导入正式 v147 投影或评分模块。它独立重建邻居、用 Gram 特征分解而不是正式 thin-SVD 求跨度投影，并重算误差、cosine、轨迹尾部、冲突与最终判决。

- 邻居索引与冲突标志完全一致；
- 距离最大差 `0`；
- 投影最大差 `9.01e-15`；
- 指标和门摘要最大差均为 `2.22e-16`；
- 最小跨度秩 `8`，最大条件数 `56.18`，最大 stationarity `2.31e-15`；
- `13/13` 完整性检查通过，正式结果树在验证前后不变。

因此负判决不是任务暂停、网络、GPU、求解器病态或独立实现漂移造成的。

## 对主线的影响

v147 关闭的是：**在当前 155D 样本级方向局部表示下，用不超过 32 个邻近 action 做线性重组来解释目标。** 它没有关闭整个 C 路线，也没有证明更有物理意义的可观测状态或目标表示失败。

下一步应改变信息本身，而不是扩大拟合器：加入能够区分全局流动工况、三维形态或残差场结构的部署可见物理状态，或重新定义更稳定的 correction target，再先做一个小型 CPU 容量门。只有新的表示先出现完整轨迹 headroom，才值得冻结最小预测器并考虑 GPU。

当前边界：

- `sample_level_direction_local_span_k_le_32_closed=true`；
- `exact_feature_collision_proven=false`；
- `global_unidentifiability_proven=false`；
- `neural_training_authorized=false`；
- `gpu_rental_authorized=false`；
- `algorithm_breakthrough=false`；
- `resource_speedup=false`；
- `external_generalization=false`；
- `curved_ray_validated=false`；
- `real_bost=false`；
- `paper_success=false`。

## English checkpoint

v147 tests whether v146 failed merely because inverse-distance weighting was too rigid. On the same 20 preregistered sentinels, deployment-visible 155D direction-local features first seal the nearest samples. Only then does a truth-aware oracle compute the best orthogonal projection of the 96D target action onto spans of the nearest 8 or 32 actions. This is a post-open capacity diagnostic, not a deployable predictor.

The wider span helps substantially but does not pass the frozen gate. Cross-trajectory span-32 passes `14/20` sentinels and `1/5` trajectory tails; the non-deployable same-trajectory span-32 upper bound passes `18/20` and `2/5`. The frozen requirement remains `20/20` and `5/5`. Nearest-neighbor and all inverse-distance controls pass `0/20`.

Every candidate in the nearest 5% neighborhood conflicts with the target gate: `740/740` cross-trajectory candidates and `200/200` same-trajectory candidates. This is only a relative-neighborhood conflict under the current metric, not proof of an exact feature collision or global mathematical impossibility.

An independent implementation uses a Gram eigendecomposition instead of the formal thin SVD and exactly reproduces neighbor indices, conflict flags, and the scientific decision. The maximum projection difference is `9.01e-15`, metric and gate-summary differences are at most `2.22e-16`, and all 13 integrity checks pass.

The current sample-level direction-local representation with spans up to 32 is therefore closed. The next useful move is to change the physically observable state or correction target and test that change with a small CPU capacity gate. Enlarging the neural model or renting a GPU is not authorized by this evidence.
