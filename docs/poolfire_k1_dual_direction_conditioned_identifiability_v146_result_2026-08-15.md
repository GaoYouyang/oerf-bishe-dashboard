# v146：硬匹配相机数后，方向条件近邻仍不能辨识 Riesz-action 目标

## 先说结论

v145 留下了一个具体疑问：跨轨迹邻居混入不同相机数量，会不会掩盖逐方向局部特征与全局相机状态的真实信号？v146 在结果前固定了更严格的 CPU 诊断，要求每个查询只从相同相机数的候选中找邻居，并分别比较 global moments、observable-pose coupling、155 维方向局部特征，以及它们的组合。

结果仍然没有通过。三个跨完整轨迹的方向局部方法都只通过 `1/20` 个哨兵；最好的 `cross local-only` 误差中位数为 `0.61131`，p90 为 `0.86701`。即使允许从同一条已开封轨迹找邻居，最好的 `within local-only` 也只有 `9/20`，五条轨迹的 p90 全部高于 `0.35` 门。

按冻结合同，Stage A 没有任何跨轨迹方法全过，所以全 `3700` 单元 Stage B 没有启动。准确结论不是 `0/3700`，而是：**当前硬相机数匹配的方向条件八近邻家族在 20 个冻结哨兵上失败，因此被提前关闭。**

## 冻结问题

- 已开封 PoolFire 可用池：五条完整轨迹、`3700` 个单元；
- Stage A：每条轨迹四个固定哨兵，共 `20` 个；
- 相机数：`5 / 7 / 9 / 12`，查询与候选必须完全相同；
- 结构方向键：`96`；每个方向只由部署可见观测与报告几何构造；
- 逐方向局部块：`155D`；全局 moments：`90D`；observable-pose coupling：`486D`；
- 跨轨迹每次 `740` 个候选；同轨迹上界每次 `184` 个候选；固定 `k=8`；
- 所有特征块分别白化，再以等块权重组合距离；
- 每哨兵门：尺度不变 relative-L2 `<=0.45` 且 cosine `>=0.90`；
- 每轨迹四哨兵尾部门：p90-higher `<=0.35`；
- 训练参数 `0`，新增精确调用 `+0A/+0A^T`。

CFD 真值只在所有邻居已封存后用于读取目标与评分，不进入特征、白化、距离、邻居或方法选择。

## Stage A 实际数字

| 方法 | 哨兵通过 | 误差中位数 | 误差 p90 | 最坏误差 | cosine 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| same-count mean control | 0/20 | 0.90579 | 0.95844 | 0.96384 | 0.42372 |
| cross global moments | 0/20 | 0.88567 | 0.99406 | 0.99822 | 0.46429 |
| cross global pose-coupled | 0/20 | 0.92119 | 0.99333 | 0.99433 | 0.38910 |
| cross local-only | 1/20 | 0.61131 | 0.86701 | 0.88427 | 0.79123 |
| cross local + moments | 1/20 | 0.63200 | 0.86800 | 0.89085 | 0.77496 |
| cross local + pose-coupled | 1/20 | 0.62992 | 0.87511 | 0.88300 | 0.77573 |
| within local-only* | 9/20 | 0.44989 | 0.60564 | 0.64921 | 0.89309 |
| within local + moments* | 7/20 | 0.48794 | 0.66591 | 0.73158 | 0.87174 |
| within local + pose-coupled* | 4/20 | 0.54621 | 0.74411 | 0.83185 | 0.83703 |

`*` 同轨迹方法读取同一条已开封轨迹的其他时刻，只是机制上界，不能部署。

`cross local-only` 的五条轨迹 p90 为 `0.69808 / 0.64035 / 0.67567 / 0.88427 / 0.86683`；`within local-only` 为 `0.44907 / 0.57902 / 0.47349 / 0.59124 / 0.64921`。两者都是 `0/5` 轨迹通过。

## 独立复算

第二实现没有导入正式 v146 模块或 runner。它重新生成 `96` 个结构键、逐样本方向索引、局部和全局特征块、白化、硬相机数候选、邻居、逆距离预测、目标指标与判决，并在读取 action labels 前先封存邻居屏障。

- 整数方向与邻居数组完全一致；
- 浮点数组最大绝对差 `8.88e-16`；
- 门汇总最大差 `1.11e-16`；
- 非数值字段不一致数为 `0`；
- `9/9` 完整性检查通过；
- 科学判决完全一致，正式结果树验证前后不变。

因此这次失败不是 VPN、任务暂停、运行中断、数值漂移或 GPU 不足造成的。

## 为什么不扩到 3700，也不租 GPU

Stage B 的授权条件是至少一个跨轨迹方法先通过全部 `20/20` 哨兵和 `5/5` 轨迹尾部门。当前最好只有 `1/20`，与门的距离很大；同轨迹上界也只有 `9/20`。把同一近邻规则扩到 `3700` 个单元只会花更多 CPU 时间，不会改变预注册判决。

这次结果关闭的是：硬相机数匹配下，当前 local/global 特征与固定八近邻预测的组合。它没有证明所有非线性映射都不可能，但也没有提供值得训练大模型的正信号。下一步若继续，必须先用更小的 CPU oracle-span 或碰撞证书区分两种原因：是逆距离权重太弱，还是部署可见状态本身没有携带足够目标信息。

当前边界：

- `hard_count_direction_conditioned_neighbor_family_closed=true`；
- `stage_b_full_roster_run=false`；
- `neural_training_authorized=false`；
- `gpu_rental_authorized=false`；
- `physical_replay_authorized=false`；
- `algorithm_breakthrough=false`；
- `resource_speedup=false`；
- `external_generalization=false`；
- `curved_ray_validated=false`；
- `real_bost=false`；
- `paper_success=false`。

## English checkpoint

v146 tests the specific explanation left open by v145. Every query is hard-matched to candidates with the same active-camera count, and fixed eight-neighbor prediction compares global moments, observable-pose coupling, a 155-dimensional direction-local representation, and their block-balanced combinations. All inputs to neighbor selection are deployment-visible observations and reported geometry.

The result is negative. The three cross-trajectory direction-local methods each pass only `1/20` frozen sentinels. The best, cross local-only, has median scale-invariant error `0.61131` and p90 `0.86701`. Even the non-deployable same-trajectory local-only upper bound passes only `9/20`, and all five trajectory p90 gates fail. No cross method meets the preregistered Stage-A gate, so the 3,700-cell Stage B is not run; this result must not be reported as `0/3700`.

An independent implementation rebuilds structural keys, direction indices, local and global blocks, whitening, same-count candidate rosters, neighbors, predictions, target metrics, and the decision without importing the formal v146 implementation. Integer arrays match exactly, the maximum floating-point array difference is `8.88e-16`, the maximum gate-summary difference is `1.11e-16`, and all nine integrity checks pass.

The frozen hard-count direction-conditioned nearest-neighbor family is therefore closed. This does not prove that every nonlinear model is impossible, but it supplies no positive signal that would justify neural training or GPU rental. Any continuation must first use a smaller CPU oracle-span or collision diagnostic to distinguish an inadequate inverse-distance rule from genuinely missing observable state. Physical replay, resource speedup, external generalization, curved rays, real BOST, algorithmic breakthrough, and paper success remain unproven.
