# v149.1：完整轨迹外折未通过，独立验证因 RFF 取样不连续而不确定

## 先说结论

v148 已证明，当前样本的分组 Krylov 表示在真值可见的容量上限下可以覆盖 `3700/3700` 个已开封 PoolFire 单元与 `5/5` 条轨迹。v149.1 随后做真正更难的一步：不再读取 CFD 真值，只从部署时可见的 observation、K1 residual、exact-K1 dual 与报告几何中提取分组特征，并用完整轨迹留一外折预测四个 canonical Krylov 系数。

正式程序中，oracle 容量仍为 `3700/3700`、`5/5`；但 fit-only mean 只有 `11/3700`，线性 ridge 为 `3089/3700`，RFF ridge 为 `2137/3700`。三个 observation-only 方法都是 `0/5` 完整轨迹通过，因此正式分支为：

`FAIL_OBSERVATION_ONLY_GROUP_KRYLOV_PREDICTOR_V149`

独立第二实现精确复现了 oracle、visible seed、mean 与线性 ridge 的结论，线性预测最大差仅 `3.67e-13`。但 RFF 的长度尺度使用“对标准化浮点特征取哈希后选择 512 行”的规则。两套实现的局部特征最大只差 `9.06e-15`，标准化后微小差异却改变了几乎整个哈希子集，使五折长度尺度最大相差 `0.46577`，RFF 预测最大相差 `0.01168`。这超过结果前冻结的独立容差，所以最终状态必须保持：

`INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_GROUP_KRYLOV_PREDICTOR_V149`

这不是算法成功，也不能被改写成“数学上证明预测不可能”。按结果前规则，当前分组坐标预测器族关闭；不做物理 replay，不用更大的 CNN/FNO/UNO 补救，也不租 GPU。

## 冻结问题

- 五条已开封 PoolFire 轨迹、`3700` 个单元、`5 / 7 / 9 / 12` 相机与 `20` 个轨迹-相机分层；
- 完整 trajectory-level leave-one-trajectory-out，held-out trajectory 真值不进入拟合、标准化、超参数或停止规则；
- 每个 active 相机/分量组使用 `61` 维局部 deployment-visible 特征，再拼接 active-set 的 mean/std/min/max，形成 `305` 维置换等变特征；
- 比较 visible seed、fit-only mean、线性 ridge、128-feature RFF ridge 与 truth-aware oracle；
- 每单元 relative-L2 `<=0.45`、cosine `>=0.90`，每轨迹和每轨迹-相机分层 p90-higher `<=0.35`；
- 本轮只是坐标预测门，新增精确调用为 `+0A/+0A^T`，没有物理重放。

## 正式与独立结果

| 方法 | 正式单元通过 | 独立单元通过 | 轨迹通过 | 分层通过 | p90-higher |
| --- | ---: | ---: | ---: | ---: | ---: |
| truth-aware block Krylov-4 oracle | 3700/3700 | 3700/3700 | 5/5 | 20/20 | 0.15531 |
| visible seed control | 2951/3700 | 2951/3700 | 0/5 | 2/20 | 0.49698 |
| fit-only mean control | 11/3700 | 11/3700 | 0/5 | 0/20 | 0.78909 |
| linear set ridge | 3089/3700 | 3089/3700 | 0/5 | 2/20 | 0.47852 |
| RFF set ridge | 2137/3700 | 2139/3700 | 0/5 | 0/20 | 0.56475 / 0.56516 |

容量上限完整通过，而最好的严格 observation-only 模型仍有 `611` 个单元失败，并且五条轨迹尾部全部越线。这说明“表示里存在答案”与“部署可见特征能稳定预测答案”是两件不同的事。

## 为什么独立验证不能写成 PASS

状态与线性路径具有很强的一致性：

- canonical basis 最大差 `1.75e-11`；
- group local feature 最大差 `9.06e-15`；
- oracle prediction 最大差 `4.02e-13`；
- linear prediction 最大差 `3.67e-13`；
- 正式结果树验证前后不变。

分叉只集中在 RFF。原规则把标准化后的浮点行四舍五入到 12 位，再按 SHA-256 排序选择 512 行估计长度尺度。该选择是离散的：极小的数值扰动会完全改变入选行。五折的 formal / independent 长度尺度最大相差 `0.46577`，进而使 RFF 模型参数、预测与 gate summary 超出冻结容差。

不能在看到结果后放宽容差，也不能让独立程序读取正式子集来制造一致。正确处理是保留 `INCONCLUSIVE`，并把这一取样规则作为不可复用的数值设计缺陷记录下来。

## 对主线的影响

v148 的容量正结果仍然成立；v149.1 没有推翻“分组谱方向存在”，而是显示当前 `305` 维 deployment-visible 特征加 mean/linear/RFF 三个小预测器没有形成可独立验证的完整轨迹映射。

当前动作：

- 关闭当前分组 canonical-coordinate predictor family；
- 不运行物理 replay、wall/RSS 或外部门；
- 不以更大 CNN/FNO/UNO/DeepONet 挽救；
- 不租 GPU；
- 下一门先审计 sealed deployment-visible 特征对 canonical target 的跨轨迹条件歧义，区分“特征信息不足”和“小模型容量不足”。

当前边界：

- `independently_validated_formal_negative=false`；
- `mathematical_impossibility=false`；
- `current_group_coordinate_predictor_family_closed=true`；
- `physical_replay_authorized=false`；
- `gpu_rental_authorized=false`；
- `algorithm_breakthrough=false`；
- `resource_speedup=false`；
- `external_generalization=false`；
- `curved_ray_validated=false`；
- `real_bost=false`；
- `paper_success=false`。

## English checkpoint

v149.1 moves beyond the v148 truth-aware capacity upper bound and asks whether a shared camera-permutation-equivariant predictor can recover four canonical groupwise Krylov coordinates from deployment-visible observation, K1 residual, exact-K1 dual state, and reported geometry under complete-trajectory leave-one-out.

The formal run retains full oracle capacity at `3700/3700` cells and `5/5` trajectory tails. However, fit-only mean, linear ridge, and RFF ridge reach only `11/3700`, `3089/3700`, and `2137/3700`; all three pass `0/5` complete trajectories. The formal branch therefore finds no passing observation-only predictor.

The independent implementation reproduces the oracle, visible seed, mean, and linear paths, with a maximum linear-prediction difference of `3.67e-13`. The RFF arm is not independently reproducible: its lengthscale chose 512 rows by hashes of rounded floating features. A maximum local-feature difference of only `9.06e-15` changes almost the entire subset, producing a maximum fold lengthscale difference of `0.46577` and an RFF prediction difference of `0.01168`. The frozen validation tolerance therefore fails and the correct final status is `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_GROUP_KRYLOV_PREDICTOR_V149`.

This is neither an algorithmic success nor a proof of impossibility. The current group-coordinate predictor family is closed by the preregistered policy, with no physical replay, larger neural rescue, GPU rental, resource claim, external-generalization claim, curved-ray claim, or real-BOST claim. The next diagnostic will test cross-trajectory conditional ambiguity in the sealed deployment-visible feature-to-coordinate map.
