# v251：首帧线性解与后续暖启动不能直接拼成 matched-accuracy

## 为什么做

v250 证明首帧固定平滑存在 headroom，但同价线性热扩散完整解释了 Charbonnier 的首帧结果。一个自然问题随之出现：能否在首帧使用已封存的线性热解，在第 1 至 32 帧继续使用 v244 已封存的因果暖 K14，从而得到一条完整序列？

这个组合在写 v251 合同前已经被只读检查过，因此 v251 **不是结果盲的前瞻试验**，而是 post-open 回顾性审计。它不重读真值场、观测或原始密度，不运行新的算子或求解器，只读取两套父结果各自的 formal 与 independent 指标和调用账。目的不是制造一次成功，而是防止“绝对误差全过”被误写成“与 K16 同精度”。

## 最坏包络与独立复算

组合规则只有一个：首帧取 v250 的同价线性热扩散结果，后 32 帧取 v244 的因果暖 K14；比较对象是同一批次的 K16 序列。由于 v244 的两套实现存在已封存的微小数值差，v251 对每个 rig、帧和指标都使用 formal/independent 两副本的逐分量最坏包络：候选取上界，reference 取下界。

formal 与不导入 formal 数值实现的逐标量 validator 均完成。独立验证通过 **17/17** 项，完整诊断最大差为 **0**；父证据树和 formal 树前后不变。v251 新增调用为 `0A+0A^T`。

## 结果

绝对误差门看起来很好：组合通过 **429/429** 个单元，完整绝对门为 **13/13** 套 rig。它的四项全局 p90 为 `0.256331 / 0.498740 / 0.624303 / 0.039316`，都低于冻结绝对门。

但 matched-accuracy 只有 **416/429**，完整 rig 为 **0/13**。十三个缺口全部位于首帧，且全部只发生在 observation 指标；对应的逐单元 candidate/K16 比值为 `3.017` 至 `3.265`，远高于冻结的 `1.05`。后续帧没有新增 matched 单元失败。

按调用账，组合每套 rig 为 `495A+462A^T`，K16 为 `528A+528A^T`，算术上相差 `9.375%`。但 matched-accuracy 没有成立，所以这只是**名义调用差**，不是有效 exact-call 减少，更不是速度或内存结果。

## 权威判决

独立状态为 `PASS_INDEPENDENT_RECOMPUTATION_CASE19_COLD_START_LINEAR_HYBRID_ATTRIBUTION_V251`，科学判决为 `FAIL_CASE19_COLD_START_LINEAR_HYBRID_MATCHED_ACCURACY_V251`。

这关闭的是“首帧线性热解 + 后续因果暖 K14”的直接拼接，也关闭用 K15 热滤波或继续扩展线性平滑来挽救这条路线。它不关闭整条 C 路线，也不证明不存在其他物理机制。没有有效 exact-call 减少、fresh wall/RSS、外部泛化、曲折光线或真实 BOST 结果；`algorithm_breakthrough=false`、`paper_success=false`，不训练大模型，不租 GPU。

# v251: frame-zero linear smoothing and later warm starts do not compose into matched accuracy

## Motivation

v250 establishes frame-zero smoothing headroom, while its equal-call linear heat control fully explains the Charbonnier result. This raises a natural composition question: can the sealed linear-heat solution be used at frame zero and the sealed v244 causal warm K14 solution be used on frames 1 through 32 to form a complete sequence?

The composition had already been inspected read-only before the v251 contract was written. v251 is therefore **not a result-blind prospective experiment**; it is a post-open retrospective audit. It rereads no truth field, observation, raw density, operator, or solver output. It consumes only the sealed formal and independent metric and call arrays. Its purpose is to prevent an all-absolute-gates pass from being mislabeled K16-matched accuracy.

## Worst-case envelope and independent recomputation

There is one composition rule: use the v250 equal-call linear-heat result at frame zero and the v244 causal warm K14 result on the remaining 32 frames, compared against the corresponding K16 sequence. Because v244 retains a sealed small numerical difference between its two implementations, v251 uses a componentwise adverse envelope at every rig, frame, and metric: the candidate takes the upper bound and the reference takes the lower bound.

Formal and a scalar-loop validator that imports no formal numerical implementation both complete. Independent validation passes **17/17** checks, with a maximum complete-diagnostic disagreement of **0**. Parent and formal evidence trees remain unchanged. v251 adds `0A+0A^T` calls.

## Results

The absolute gate looks strong: the composition passes **429/429** cells and **13/13** complete rigs. Its field, full-gradient, interior-gradient, and observation global p90 values are `0.256331 / 0.498740 / 0.624303 / 0.039316`, all within the frozen absolute limits.

Matched accuracy reaches only **416/429** cells and **0/13** complete rigs. All thirteen misses occur at frame zero and only in the observation metric. Their cellwise candidate/K16 ratios range from `3.017` to `3.265`, far above the frozen `1.05` limit. No later frame adds a matched-cell failure.

The nominal ledger is `495A+462A^T` per rig for the hybrid versus `528A+528A^T` for K16, an arithmetic difference of `9.375%`. Because matched accuracy fails, this is only a **nominal call difference**, not effective exact-call reduction and not a wall-time or memory result.

## Authoritative verdict

The independent status is `PASS_INDEPENDENT_RECOMPUTATION_CASE19_COLD_START_LINEAR_HYBRID_ATTRIBUTION_V251`; the scientific verdict is `FAIL_CASE19_COLD_START_LINEAR_HYBRID_MATCHED_ACCURACY_V251`.

This closes the direct “frame-zero linear heat plus later causal warm K14” composition and forbids rescue through K15 heat or further linear-smoothing expansion. It does not close the full C route or prove that every distinct physical mechanism is impossible. No effective exact-call reduction, fresh wall/RSS, external generalization, curved-ray validation, or real-BOST result is established. `algorithm_breakthrough=false` and `paper_success=false`; no larger model or GPU run is authorized.
