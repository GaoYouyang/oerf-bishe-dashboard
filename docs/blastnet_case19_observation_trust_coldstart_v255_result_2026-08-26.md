# v255：观测信任冷启动因独立数值合同未闭合而保持不确定

## 为什么做

v251 已把完整序列的缺口定位到首帧观测误差，v254 又说明无序全局 K1 子空间不能恢复有序暖启动结构。v255 因此检验一个更窄、物理上不同的问题：能否只根据首帧部署可见的归一化观测残差，在两个已经封存的物理端点之间确定一个信任混合，再沿用后续 32 帧的封存因果 K14 路径。

结果前固定了唯一残差信任规则、两个端点、后续路径、四指标绝对门、K16-matched 门、两个端点 control、调用账和独立数值容差。该规则不读取 CFD 真值、rig 标签或失败标签，训练参数为 0。

## 独立合同没有通过

独立第二实现完成了物理 replay，但只通过 `26/29` 项冻结检查。三项失败分别是：混合系数最大绝对差 `6.73e-9`，高于 `1e-12` 容差；残差最大相对差 `7.51e-9`，高于 `1e-10` 容差；逐单元指标最大绝对差 `1.02e-8`，高于 `1e-9` 容差。它们分别约为冻结界的 `6727.64× / 75.11× / 10.22×`。

场最大相对差 `5.09e-10` 通过 `1e-8` 界，观测与物理 replay 差也很小，但通过部分检查不能覆盖三项明确越界。正式状态因此是 `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_OBSERVATION_TRUST_COLDSTART_V255`，科学判决为 `INCONCLUSIVE_INVALID_CASE19_OBSERVATION_TRUST_COLDSTART_V255`。

## 离散计数只能作诊断

两套实现的离散门判决一致：primary 诊断计数为绝对 `429/429、13/13`，K16-matched `429/429、13/13`；端点零 control 为绝对 `428/429、12/13`、matched `429/429、13/13`；端点一 control 为绝对 `429/429、13/13`、matched `416/429、0/13`。K16 reference 自身只有绝对 `417/429、9/13`。

这些数字不能写成算法 headroom、通过或失败，因为独立数值合同已经失效。离散一致不能替代连续浮点闭环，reference 不充分也不能把无效结果转化成正结果。

## 成本与路线动作

primary 的名义序列账为 `498A+465A^T`，K16 reference 为 `528A+528A^T`，算术差为 `8.8068%`。由于科学结果不成立，这不是有效 exact-call 减少，也不授权 wall time、RSS、外部门、训练或 GPU。

当前观测信任混合关闭；不放宽数值界、不重跑、不重新调节信任规则、端点、平滑或深度。这不关闭整条 C 路线，也不证明该类问题数学上不可能。下一步只能来自新的配对真实 BOST 信息，或一个结果前唯一冻结、部署可见、可独立证伪且物理上真正不同的机制。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v255: observation-trust cold start remains inconclusive after independent numeric closure fails

## Motivation

v251 localizes the whole-sequence gap to frame-zero observation error, while v254 shows that an unordered global K1 subspace cannot recover the structure preserved by chronological warm starts. v255 therefore tests a narrower, physically distinct question: can the deployment-visible normalized frame-zero observation residual select a trust blend between two sealed physics endpoints, followed by the sealed causal K14 path for the remaining 32 frames?

The unique residual-trust rule, two endpoints, later-frame path, four absolute gates, K16-matched gates, two endpoint controls, call ledger, and independent numerical tolerances were fixed before results. The rule reads no CFD truth, rig label, or failure label and has zero trainable parameters.

## The independent contract does not pass

The independent implementation completes physical replay but passes only `26/29` frozen checks. The three failures are a maximum blend-coefficient absolute difference of `6.73e-9` against a `1e-12` tolerance, maximum residual relative difference of `7.51e-9` against `1e-10`, and maximum cell-metric absolute difference of `1.02e-8` against `1e-9`. These are approximately `6727.64x / 75.11x / 10.22x` their frozen limits.

Maximum field relative disagreement of `5.09e-10` passes its `1e-8` limit, and observation and physical-replay differences are also small. Passing a subset of checks cannot override three explicit failures. The formal status is therefore `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_OBSERVATION_TRUST_COLDSTART_V255`, with scientific decision `INCONCLUSIVE_INVALID_CASE19_OBSERVATION_TRUST_COLDSTART_V255`.

## Discrete counts are diagnostic only

The two implementations agree on discrete gates. Diagnostic primary counts are absolute `429/429 cells and 13/13 rigs` and K16-matched `429/429 and 13/13`. The zero-endpoint control reaches absolute `428/429 and 12/13` and matched `429/429 and 13/13`; the one-endpoint control reaches absolute `429/429 and 13/13` but matched `416/429 and 0/13`. The K16 reference itself reaches only absolute `417/429 and 9/13`.

None of these numbers is admissible as algorithmic headroom, a pass, or a fail because the independent numerical contract is invalid. Discrete agreement cannot replace continuous floating-point closure, and reference inadequacy cannot convert an invalid result into a positive one.

## Cost and route action

The nominal primary sequence ledger is `498A+465A^T`, versus `528A+528A^T` for K16, an arithmetic difference of `8.8068%`. Because the scientific result is invalid, this is not effective exact-call reduction and does not authorize wall time, RSS, an external gate, training, or GPU use.

The current observation-trust blend closes without relaxing numerical limits, rerunning, or retuning its trust rule, endpoints, smoothing, or depth. This does not close the C route or prove mathematical impossibility. Any next step requires new paired real-BOST information or one uniquely preregistered, deployment-visible, independently falsifiable, and physically distinct mechanism.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.
