# v263.1：九个既有低成本候选连真值 oracle 选择也无法通过完整门

## 为什么先做这个审计

在训练“选择哪个候选”的模型之前，先问一个更严格也更便宜的问题：如果允许一个知道真值的裁判，在每套相机中从九个已经独立封存、成本不超过 `15A+15A^T` 的候选里挑最好，是否至少存在通过绝对精度门与保守 K16-matched 门的选择？如果这个 truth-aware oracle 都失败，只看部署可见 observation/geometry 的选择器就没有可学习的成功标签。

v263.1 只读取这些父结果已封存的真值派生指标，不读取原始密度，不生成新候选场，也不新增精确算子调用。为避免不同父实现的微小浮点差异制造假通过，候选取正式与独立复算中较差的指标，K16 reference 取多个封存副本中较好的指标。

## 独立结果

完全独立第二实现通过 `19/19` 项检查。正式与独立的数值数组和汇总最大差均为 `0`；重复 control 指标最大差为 `8.77e-11`，跨父 K16 reference 最大差为 `5.83e-10`。

九个候选中，有五个能在全部 `13/13` 套相机上通过绝对门，但每一个候选的 K16-matched 通过数都是 `0/13`。即使逐 rig 使用真值挑选最好候选，完整联合门仍只有 `0/13`；最佳联合负担的 p50 / p90-higher / worst 为 `1.06082 / 1.06693 / 1.06876`，而通过线是 `1.0`。

这意味着最好的旧候选距离门只差约 `6.08%-6.88%`，但它仍是严格失败，不能在看过结果后放宽门或改写成“基本通过”。

## 判决边界

正式判决是 `FAIL_CASE19_EXISTING_CHEAP_ARM_ORACLE_SELECTOR_CAPACITY_V263_1`。关闭的是：在这九个确定候选上训练 selector、gate 或更大模型来做逐 rig 选择。因为连真值 oracle 都是 `0/13`，模型容量不能修复候选池本身没有通过者的问题。

它不关闭整条 C 路线，也不排除物理上真正不同的新候选。该审计只覆盖已经打开的 Case 19 首帧，不是完整序列、外部泛化、真实 BOST、减调用、wall/RSS 或算法突破。两个实现都复用封存的父指标，因此没有证明端到端物理独立性。

下一步只接受两类新信息：结果前冻结且物理上不同的新候选机制，或真正配对的二维双分量 BOST 位移及其相机/帧/标定映射、噪声重复与认可基线。不为这九个旧候选训练网络，也不租 GPU。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v263.1: even a truth-aware oracle cannot rescue the nine existing cheap arms

## Why this audit comes first

Before training a model to choose among candidate arms, v263.1 asks a stricter and cheaper question. If an oracle that knows the truth may select the best of nine already independently sealed arms for each rig, all costing no more than `15A+15AT`, does any selection satisfy both the absolute-accuracy gate and the conservative K16-matched gate? If that truth-aware oracle fails, an observation/geometry-only selector has no successful label to learn within this arm pool.

v263.1 reads only sealed truth-derived parent metrics. It does not read raw density, construct a new candidate field, or add exact operator calls. To prevent small cross-parent floating-point differences from creating a false pass, each candidate uses the worse metric across its formal and independent replicas, while the K16 reference uses the better metric across sealed replicas.

## Independent result

The fully independent implementation passes `19/19` checks. Maximum formal-independent differences are `0` for both numerical arrays and summaries. Duplicate-control metrics differ by at most `8.77e-11`, and sealed K16 references across parents differ by at most `5.83e-10`.

Five of the nine arms clear the absolute gate on all `13/13` rigs, but every arm reaches only `0/13` K16-matched rigs. Even truth-aware per-rig selection therefore reaches `0/13` joint passes. The best joint-burden p50 / p90-higher / worst are `1.06082 / 1.06693 / 1.06876`, against a passing line of `1.0`.

The best existing arm is only about `6.08%-6.88%` beyond the gate, but it is still a strict failure. The threshold cannot be relaxed or repackaged after observing the result.

## Decision boundary

The sealed decision is `FAIL_CASE19_EXISTING_CHEAP_ARM_ORACLE_SELECTOR_CAPACITY_V263_1`. It closes a selector, gate, or larger-model rescue restricted to these exact nine arms: even a truth-aware oracle has no passing arm to select.

It does not close the entire C route or rule out a physically distinct new candidate. This audit covers only frame zero in already-opened Case 19. It is not a full-sequence, external-generalization, real-BOST, exact-call-reduction, wall/RSS, or algorithm result. Both implementations reuse sealed parent metric arrays, so end-to-end physics independence is not established.

The next admissible input is either a preregistered physically different mechanism or genuinely paired two-component BOST displacement with camera/frame/calibration mapping, repeated-noise information, and an accepted baseline. No network or GPU will be used to rescue these nine arms.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
