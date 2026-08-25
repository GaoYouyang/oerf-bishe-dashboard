# v249：Haar-MAD 首帧诊断为 13/13，但独立系数门未闭合

## 为什么做

v248 的固定体素块机制退出后，v249 结果前冻结一个物理上不同、仍然不训练模型的多尺度校正。主候选先从零起点运行 reported-geometry 对角 Jacobi-PCGLS K13，再对 `32×16×16` 三维场做一层正交 Haar 变换；用 HHH 高频子带的 MAD 估计尺度，以 `sigma * sqrt(2 * log(7168))` 对七个细节子带做软阈值，保留低频近似，恢复边界与规范后执行一次精确投影和未修改的 restarted PCGLS K1。实际逻辑账为 `15A+14A^T`，训练参数为 0。

结果前同时冻结三个对照：同价的“只保留 Haar 低频”对照 `15A+14A^T`、更便宜的原始 K14 对照 `14A+14A^T`，以及 K16 reference `16A+16A^T`。只检查已开封 Case 19 的 13 套 rig 各自首帧；只有主候选 `13/13`、对照不完整且独立检查全部通过，才允许另冻完整 429 单元序列。

## 正式运行与独立复算

formal 完成 13 个首帧单元并通过 **22/22** 项有效性检查。完全独立第二实现用显式嵌套循环重建 Haar 配对变换、观测、预处理、Jacobi、PCGLS、候选场、物理观测、四项指标、门和调用账，最终通过 **33/35** 项检查。

唯一原发失败是 Haar 系数相对差 **1.8209834396e-11**，高于结果前冻结的 **1e-12**。`formal_independent_decision_exact` 随之失败。其余关键差异均在各自冻结界内：校正初始化场相对差 `1.72e-11 <= 1e-10`，最终场相对差 `1.02e-9 <= 1e-8`，归一化残差差 `2.41e-10 <= 1e-8`，观测归一化差 `8.38e-17 <= 1e-12`，指标绝对差 `4.76e-11 <= 1e-9`，汇总绝对差 `2.16e-11 <= 1e-9`。相机换序、保留系数个数、输入和封存树均一致。

## 只读归因

判决之后只做了不改变任何结果的数值归因：把 formal 的 K13 基场送入独立 Haar 实现，最大相对差为 `6.63e-16`；把 independent 的 K13 基场送入 formal Haar 实现，最大相对差为 `6.83e-16`。两套 K13 基场本身最大相对差为 `1.57e-11`。因此未闭合项来自独立重建的预处理、观测与 K13 数值传播，而不是 Haar 配对变换本身。

这项归因不能修复结果前冻结的系数门，也不能把 v249 改判为通过。没有放宽阈值、没有 v249.1 补跑，也没有换实现挑结果。

## 诊断数字与权威判决

两套实现的离散计数完全一致，但因为独立门未全过，只能作为诊断：Haar-MAD 主候选 **13/13**，同价低频近似对照 **0/13**，原始 K14 对照 **7/13**，K16 reference **12/13**。主候选四指标 p90 为 `0.361612 / 0.719495 / 0.641305 / 0.128624`，均在冻结的 `0.5 / 0.75 / 0.75 / 0.2` 首帧门内。

权威判决仍是 `INCONCLUSIVE_INVALID_CASE19_HAAR_MAD_WARM_FRAME_ZERO_V249`。这些诊断数字提示固定多尺度校正在首帧上可能有值得重视的物理信号，但合同不允许据此声明 headroom，更不能授权完整 429 单元序列。当前固定 v249 不继续投入；这不是数学不可能证明，也不关闭整条 C 路线。

没有有效 exact-call 减少、fresh wall/RSS、外部泛化、曲折光线或真实 BOST 结果。`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`；不训练大模型、不租 GPU。

# v249: Haar-MAD is diagnostically 13/13 at frame zero, but the independent coefficient gate does not close

## Motivation and frozen mechanism

After v248 retires the fixed voxel-block mechanism, v249 preregisters a physically distinct, non-learned multiscale correction. The primary runs zero-start reported-geometry diagonal-Jacobi PCGLS K13, applies one orthonormal 3D Haar level to the `32x16x16` field, estimates scale from the HHH-detail MAD, soft-thresholds all seven detail subbands at `sigma * sqrt(2 * log(7168))`, preserves the low-frequency approximation, reapplies the support and gauge, and then performs one exact projection plus one unchanged restarted PCGLS iteration. Its actual logical ledger is `15A+14A^T`, with zero trainable parameters.

Three controls are frozen before results: an equal-cost Haar approximation-only arm at `15A+14A^T`, a cheaper raw K14 arm at `14A+14A^T`, and the K16 reference at `16A+16A^T`. Only one frame-zero cell from each of 13 already-opened Case 19 rigs is scored. A separate full 429-cell protocol requires a 13/13 primary, incomplete controls, and every independent check to pass.

## Formal and independent recomputation

Formal completes all 13 cells and passes **22/22** validity checks. A fully independent second implementation rebuilds the Haar transform with explicit nested loops and independently reconstructs preprocessing, observations, Jacobi, PCGLS, candidate fields, physical observations, four metrics, gates, and call ledgers. It passes **33/35** checks.

The sole primary failure is Haar-coefficient relative disagreement of **1.8209834396e-11**, above the preregistered **1e-12** limit; `formal_independent_decision_exact` consequently fails. All other key comparisons remain within their frozen limits: corrected-initializer relative disagreement is `1.72e-11 <= 1e-10`, final-field disagreement is `1.02e-9 <= 1e-8`, normalized-residual disagreement is `2.41e-10 <= 1e-8`, normalized-observation disagreement is `8.38e-17 <= 1e-12`, metric disagreement is `4.76e-11 <= 1e-9`, and summary disagreement is `2.16e-11 <= 1e-9`. Camera permutation, retained-coefficient counts, inputs, and seals agree.

## Read-only attribution

Post-decision attribution does not alter any scientific output. Passing the formal K13 base field through the independent Haar implementation gives a maximum relative difference of `6.63e-16`; passing the independent K13 base field through the formal Haar implementation gives `6.83e-16`. The independently reconstructed K13 base fields themselves differ by `1.57e-11`. The missed coefficient gate therefore originates upstream in independently rebuilt preprocessing, observation, and K13 numerical propagation, rather than in the Haar pair transform itself.

This attribution cannot repair the preregistered gate. There is no tolerance relaxation, v249.1 rescue run, or implementation selection after seeing the result.

## Diagnostic counts and authoritative verdict

Discrete counts agree exactly across both implementations, but remain diagnostic because the independent gate is incomplete: Haar-MAD reaches **13/13**, the equal-cost approximation-only control reaches **0/13**, raw K14 reaches **7/13**, and the K16 reference reaches **12/13**. The primary p90 ratios are `0.361612 / 0.719495 / 0.641305 / 0.128624`, within the frozen `0.5 / 0.75 / 0.75 / 0.2` frame-zero limits.

The authoritative decision remains `INCONCLUSIVE_INVALID_CASE19_HAAR_MAD_WARM_FRAME_ZERO_V249`. These diagnostics suggest a potentially meaningful frame-zero multiscale signal, but the contract forbids claiming headroom or authorizing the full 429-cell sequence. The fixed v249 implementation receives no further investment. This is not a mathematical-impossibility claim and does not close the C route.

No effective exact-call reduction, fresh wall/RSS, external generalization, curved-ray validation, or real-BOST result is established. `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`; no larger model or GPU run is authorized.
