# v145：全局 camera-set 状态仍不能辨识共享 Riesz-action 目标

## 先说结论

v144 表明当前逐方向 155 维局部邻域不足。v145 随后直接检验一个物理上不同的解释：也许每个方向缺少的是整组相机的全局观测状态，而不是更复杂的局部回归器。

正式实验给每个样本构造两种相机排列不变的全局签名：逐相机部署特征的均值与标准差，以及 detector observable 与报告位姿的中心化耦合。两者都只读取部署可见的观测和已知几何，并用固定八近邻预测 96 个方向的 Riesz-action 目标。

结果仍然失败。跨完整轨迹的 moments 与 coupled 方法都是 `0/20` 哨兵、`0/3700` 单元、`0/5` 轨迹；不能部署的同轨迹诊断也同样是 `0/20` 和 `0/3700`。这关闭了“共享全局邻居度量就足够”的假设，但不关闭方向条件化的局部加全局表示，也不关闭整个 C 路线。

## 冻结实验

- 已开封 PoolFire：五条完整轨迹、`3700` 个单元；
- 相机数：`5 / 7 / 9 / 12`；
- 结构方向键：`96`；固定哨兵：`20`；
- 每相机部署特征：`45D`；
- moments 签名：相机数 one-hot、均值、标准差，共 `94D`；
- observable-pose coupled 签名：moments 加中心化 `27 x 18` 交叉协方差，共 `580D`；
- 固定 `k=8` 逆距离邻居；训练参数 `0`；新增精确调用 `+0A/+0A^T`；
- 每哨兵门：尺度不变 relative-L2 `<=0.45` 且 cosine `>=0.90`；
- 每条完整轨迹门：p90-higher `<=0.35`。

跨轨迹折只从其余四条完整轨迹取候选。额外的同轨迹诊断排除自身，但它是 post-open 上限，不是部署模型。CFD 真值只用于计算目标和评分，不进入签名或邻居选择。

## 实际数字

| 方法 | 哨兵 | 完整单元 | 轨迹 | 误差中位数 | 误差 p90 | 最坏误差 | cosine 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| camera-count mean control | 0/20 | 0/3700 | 0/5 | 0.90790 | 0.95704 | 0.99300 | 0.41918 |
| cross moments | 0/20 | 0/3700 | 0/5 | 0.91938 | 0.97765 | 1.00000 | 0.39338 |
| cross observable-pose coupled | 0/20 | 0/3700 | 0/5 | 0.92444 | 0.98299 | 1.00000 | 0.38132 |
| same-trajectory moments* | 0/20 | 0/3700 | 0/5 | 0.86064 | 0.96271 | 1.00000 | 0.50922 |
| same-trajectory coupled* | 0/20 | 0/3700 | 0/5 | 0.85903 | 0.96354 | 1.00000 | 0.51193 |

`*` 同轨迹方法只能用于已开封机制诊断，不能部署。

## 独立复算

第二实现没有导入正式 v145 模块。它用逐样本循环重建两个全局签名，并以独立距离实现重建邻居、预测、目标指标和判决：

- 整数邻居与方向数组完全一致；
- 浮点数组最大绝对差 `1.31e-12`；
- 汇总量最大绝对差 `1.45e-14`；
- `9/9` 完整性检查通过；
- 科学判决一致，正式结果树在验证前后不变；
- 两套实现的新增精确调用账均为 `+0A/+0A^T`。

因此这不是运行中断、数值漂移或 GPU 不足造成的失败。

## 相机数量混合审计

结果后审计发现，跨轨迹邻居确实混合了不同相机数：moments 的同相机数边比例为 `55.8%`，coupled 只有 `30.7%`。但这不能单独解释失败，因为同轨迹邻居的同相机数边比例已经达到 `97.1% / 95.7%`，仍然是 `0/20` 哨兵和 `0/3700` 单元。

准确结论是：相机数量硬匹配值得加入下一门，但单独消除相机数混合并不足以证明目标可辨识。下一步必须把 v144 的逐方向局部状态与 v145 的全局 camera-set 状态结合，而不是继续调一个共享全局距离。

## 路线动作与边界

下一门只在 CPU 上比较三个结果前固定的方法：相机数硬匹配的 local-only、local + moments、local + observable-pose coupling。它们必须继续按完整轨迹留一、20 哨兵和五条轨迹尾部判决。只有这一门出现清晰正信号，才有理由冻结最小可训练模型并重新评估是否需要租 GPU。

当前边界：

- `shared_global_neighbor_metric_closed=true`；
- `direction_conditioned_local_global_metric_ruled_out=false`；
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

v145 tests whether the v144 failure can be repaired by conditioning every target direction on a permutation-invariant summary of the complete camera set. It compares a 94-dimensional camera-count/moment signature and a 580-dimensional observable-pose coupled signature, both built solely from deployment-visible observations and reported geometry. Fixed eight-neighbor prediction is evaluated over all 3,700 opened PoolFire cells, five complete trajectories, 5/7/9/12 cameras, 20 frozen sentinels, and 96 structural action keys.

The result is negative. Cross-trajectory moments and coupled methods both pass `0/20` sentinels, `0/3700` cells, and `0/5` trajectory-tail gates. The non-deployable same-trajectory diagnostics also pass `0/20` and `0/3700`. An independent implementation rebuilds signatures, neighbors, predictions, metrics, and decisions without importing the formal v145 module. Integer arrays match exactly, the largest floating-point array difference is `1.31e-12`, the largest summary difference is `1.45e-14`, and all nine integrity checks pass.

A post-result audit finds substantial camera-count mixing in cross-trajectory neighborhoods, but same-trajectory neighborhoods already contain more than 95% same-count edges and still fail every sentinel and cell. Camera-count mixing is therefore not a sufficient explanation. The next CPU-only gate hard-matches camera count and compares the existing 155-dimensional direction-local representation against local-plus-moments and local-plus-observable-pose coupling. No neural training, GPU rental, physical replay, speedup, external-generalization, curved-ray, real-BOST, breakthrough, or paper-success claim is authorized.
