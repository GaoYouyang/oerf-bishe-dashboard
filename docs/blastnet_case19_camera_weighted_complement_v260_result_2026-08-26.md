# v260：按相机残差能量加权没有修复 Case 19 的观测 matched-accuracy

## 这次只检验一个机制

v259 独立确认，v258 相对 K16 的剩余观测超额在结果前冻结的优先级下具有相机局部结构。v260 因此只检验一个部署可见的最小规则：从 K13 残差与当前观测计算每台相机的归一化残差能量，将九个权重归一到均值 1，再用它们构造完整 K13 测量投影，最后执行一次未修改的 geometry-Jacobi PCGLS。

这个规则不读取真值、时间、rig 标签或相机 ID，不使用 top-k、指数、裁剪、阈值或回退。主候选和未加权 v258 对照均为 `15A+14A^T`；raw K14 为 `14A+14A^T`，K16 reference 为 `16A+16A^T`。试验仍只覆盖已经开封的 Case 19 十三套 rig 的首帧。

## 独立结果

完全独立的第二实现重新构造 K13 迭代、逐相机能量、均值为 1 的权重、加权相关系统、物理 replay、四个对照和全部门。`52/52` 项检查全真。两套实现的最大场相对差为 `1.02e-9`，归一化残差差为 `2.41e-10`，指标绝对差为 `4.76e-11`；相机分数与权重最大相对差分别为 `4.73e-11` 与 `6.27e-11`。相机乱序后的状态、场和残差差均为 0。

主候选通过 `13/13` 个绝对门，但相对 K16 的 matched-accuracy 为 `0/13`。绝对指标的 p90-higher（field / full-gradient / interior-gradient / observation）为：

- `0.13875 / 0.26286 / 0.36108 / 0.06274`

相对 K16 的 matched 比值 p90-higher 为：

- `0.45169 / 0.46651 / 0.49534 / 1.22811`

worst 比值为：

- `0.47397 / 0.47493 / 0.50343 / 1.36863`

因此唯一系统性阻塞仍是观测项，它超过冻结的 `1.05` matched 门。更关键的是，同价未加权 v258 对照的观测 matched p90-higher / worst 为 `1.22636 / 1.36693`，都比 v260 略好；加权分别恶化约 `0.00176 / 0.00169`。线性热对照同样是 `0/13` matched，raw K14 只通过 `7/13` 个绝对门。

## 判决与边界

封存判决为 `FAIL_CASE19_CAMERA_WEIGHTED_COMPLEMENT_FRAME_ZERO_V260`。v259 找到的是症状位置，不等于简单的残差能量相机权重就是充分修复规则。当前这套分数、均值归一化、加权投影与一次 PCGLS 的机制族关闭；不再调指数、floor、裁剪、rank、heat、深度或 lambda，也不扩展到 429 帧，不训练网络、不租 GPU、不运行资源门。

这是一条有效的开封后机制负证据，不是算法通过、外部门、速度结果、曲折光线验证或真实 BOST 结果。它也不证明整条 C 路线不可能。后续只有在另行结果前冻结一个物理上不同的机制，或获得真正配对的二维双分量 BOST 数据后才能继续。`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v260: residual-energy camera weighting does not repair Case 19 matched observation accuracy

## The single mechanism tested

v259 independently established that the remaining v258-over-K16 observation excess is camera-local under the preregistered priority. v260 therefore tests one minimal deployment-visible rule: compute normalized residual energy per camera from the K13 residual and current observation, normalize the nine weights to mean one, use them to form the complete K13 measurement projection, and then execute one unchanged geometry-Jacobi PCGLS step.

The rule reads no truth, time, rig label, or camera ID and uses no top-k choice, exponent, clipping, threshold, or fallback. The primary and the unweighted v258 control both cost `15A+14AT`; raw K14 costs `14A+14AT`, and the K16 reference costs `16A+16AT`. The experiment remains limited to frame zero of the thirteen already-opened Case 19 rigs.

## Independent result

A fully independent implementation rebuilds the K13 recurrence, per-camera energy, mean-one weights, weighted correlation system, physical replay, four controls, and every gate. All `52/52` checks pass. Maximum cross-implementation differences are `1.02e-9` relative for the field, `2.41e-10` for normalized residuals, and `4.76e-11` absolute for metrics. Maximum relative differences for camera scores and weights are `4.73e-11` and `6.27e-11`. State, field, and residual differences after camera permutation are all zero.

The primary clears all `13/13` absolute cells but reaches `0/13` under K16-matched accuracy. Its p90-higher absolute metrics (field / full gradient / interior gradient / observation) are:

- `0.13875 / 0.26286 / 0.36108 / 0.06274`

Its K16-matched p90-higher ratios are:

- `0.45169 / 0.46651 / 0.49534 / 1.22811`

The worst ratios are:

- `0.47397 / 0.47493 / 0.50343 / 1.36863`

Observation is therefore the only systematic blocker and remains above the frozen `1.05` matched gate. More importantly, the equal-cost unweighted v258 control has observation-matched p90-higher / worst ratios of `1.22636 / 1.36693`, both slightly better than v260. Weighting worsens them by about `0.00176 / 0.00169`. The linear-heat control also reaches `0/13` matched cells, while raw K14 clears only `7/13` absolute cells.

## Verdict and boundary

The sealed decision is `FAIL_CASE19_CAMERA_WEIGHTED_COMPLEMENT_FRAME_ZERO_V260`. v259 localized the symptom; it did not establish that simple residual-energy camera weights were a sufficient repair rule. This exact score, mean normalization, weighted projection, and one-step PCGLS mechanism family is closed. No exponent, floor, clipping, rank, heat, depth, or lambda tuning is authorized; there is no 429-frame expansion, training, GPU rental, or resource gate.

This is valid post-open mechanism evidence, not an algorithmic pass, external gate, speed result, curved-ray validation, or real BOST result. It does not prove the entire C route impossible. Further work requires a physically different mechanism under a separate result-before-run contract or genuinely paired two-component BOST observations. `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
