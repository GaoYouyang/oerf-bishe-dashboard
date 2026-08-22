# v199：固定正则有改善，但 p14 的比较参考本身不够格

## 结论先说

v199 把 v198 在 p22 上选出的固定 `identity-prior tau=2^-8` 原样带到历史上已经暴露的 p14 开发轨迹，并保留 full-DCT K1 父方法、full-DCT K2 reference 和便宜控制。固定正则 K1 在九相机达到 **1313/1313** 个严格单元与 **13/13** 个完整组；在五相机达到 **1268/1313** 与 **3/13**，明显优于未正则 K1 的 **1173/1313、0/13**。

但预注册的 full-DCT K2 reference 在五相机自身只有 **1213/1313、0/13**。按照结果前冻结的判决顺序，参考不充分必须优先，因此正式科学判决是 `INCONCLUSIVE_P14_REFERENCE_INADEQUATE_V199`。这不是候选成功，也不是候选被有效否定；`algorithm_breakthrough=false`。

## 做了什么

这次没有重新选择正则强度。唯一候选固定使用 p22 已经选出的 `tau=2^-8`，从当前多相机观测与报告几何构造 full-DCT 坐标，得到一个 identity-prior 初始化，再执行一次未修改的物理 CGLS K1。逻辑在线账为 **2A+1A^T**。

对照包括同价未正则 full-DCT K1、**3A+2A^T** 的 full-DCT K2 reference、固定 initializer-only、Jacobi PCGLS1、BP-CGLS1、Zero-K2 和历史 dual-ridge K1。预测 barrier、调用 receipt、物理 replay 和一次性独立验证均在读取评分真值前封存。

## 关键数字

| 方法 | 九相机严格单元 / 完整组 | 五相机严格单元 / 完整组 | 逻辑在线账 |
|---|---:|---:|---:|
| 固定 identity-prior + K1 | 1313/1313 · 13/13 | 1268/1313 · 3/13 | 2A+1A^T |
| 未正则 full-DCT K1 | 1313/1313 · 13/13 | 1173/1313 · 0/13 | 2A+1A^T |
| full-DCT K2 reference | 1313/1313 · 13/13 | 1213/1313 · 0/13 | 3A+2A^T |

五相机固定候选的 field / gradient / observation p90 为 **0.421892 / 0.697287 / 0.148866**，worst 为 **0.539642 / 0.925636 / 0.196865**。未正则 K1 的 p90 为 **0.456048 / 0.751378 / 0.156668**；K2 reference 为 **0.449851 / 0.737940 / 0.116022**。

因此固定正则确实改善了五相机尾部，而且严格安全单元甚至多于当前 K2 reference。但它仍没有守住完整五相机门，reference 也没有资格支撑相对调用数比较。不能把 `2A+1A^T` 对 `3A+2A^T` 的账面差写成 exact-call 减少或速度收益。

## 为什么可信

独立第二实现重建观测、full-DCT 坐标、固定正则正规方程、全部控制、物理 CGLS、逐单元门、13 组尾部和调用账，再与正式数组做事后比较。独立状态为 `PASS_INDEPENDENT_RECOMPUTATION_FIXED_IDENTITY_PRIOR_P14_V199`。

独立 direct-normal solve 最大相对差约 **4.42e-13**，观测重建最大相对差约 **1.96e-11**，正式与独立指标最大绝对差约 **1.23e-11**，汇总最大差约 **1.13e-11**。相机换序后的坐标相对差约 **1.82e-14**，近零 stationarity 绝对差约 **5.18e-15**。正式树、输入树和一次性 release 均保持封存。两条实现仍共享冻结的数值内核与同一输入，所以 `end_to_end_physics_independence_proven=false`。

## 成功、失败与下一步

**成功：** 固定正则在没有重新调参的情况下，把五相机严格安全单元增加了 95 个，并首次让 3/13 个五相机完整组通过。这说明 v198 的简单正则线索不是只在 p22 单点成立。

**失败：** 候选没有达到完整五相机门；更关键的是 K2 reference 自身也为 0/13 完整组，使这次比较无法回答“是否以更少调用达到同等精度”。因此结果只能记为 inconclusive。

下一步不再在 p14 上调整 `tau`、Krylov 深度或事后更换 reference。只有结果前冻结、物理上不同且先证明五相机 reference 充分性的机制才值得继续；否则等待工况匹配的真实二维 BOST 位移、标定映射、噪声重复与认可基线。p14 不是 fresh validation，test、wall/RSS、外部门和 GPU 仍不授权。

# v199: fixed regularization helps, but the p14 comparison reference is inadequate

## Bottom line

v199 carries the fixed `identity-prior tau=2^-8` selected on p22 into the historically exposed p14 development trajectory without retuning. It retains the full-DCT K1 parent, full-DCT K2 reference, and cheap controls. Fixed-regularized K1 reaches **1313/1313** strict cells and **13/13** complete groups under all nine cameras, and **1268/1313** with **3/13** groups under five cameras, clearly improving on unregularized K1 at **1173/1313 and 0/13**.

The preregistered full-DCT K2 reference itself reaches only **1213/1313 and 0/13** under five cameras. The frozen adjudication order gives reference inadequacy precedence, so the scientific decision is `INCONCLUSIVE_P14_REFERENCE_INADEQUATE_V199`. This is neither a candidate pass nor a valid candidate rejection. `algorithm_breakthrough=false`.

## What was run

Regularization was not selected again. The sole candidate uses the p22-fixed `tau=2^-8`, constructs full-DCT coordinates from current multiview observations and reported geometry, forms an identity-prior initializer, and runs one unchanged physical CGLS K1 step. Its logical online ledger is **2A+1AT**.

Controls include equal-cost unregularized full-DCT K1, the **3A+2AT** full-DCT K2 reference, fixed initializer-only, Jacobi PCGLS1, BP-CGLS1, Zero-K2, and historical dual-ridge K1. Prediction barriers, call receipts, physical replay, and one-shot independent validation were sealed before scoring truth was read.

## Key numbers

| Method | All-nine strict cells / groups | Five-camera strict cells / groups | Logical online ledger |
|---|---:|---:|---:|
| Fixed identity-prior + K1 | 1313/1313 · 13/13 | 1268/1313 · 3/13 | 2A+1AT |
| Unregularized full-DCT K1 | 1313/1313 · 13/13 | 1173/1313 · 0/13 | 2A+1AT |
| Full-DCT K2 reference | 1313/1313 · 13/13 | 1213/1313 · 0/13 | 3A+2AT |

For the five-camera fixed candidate, field / gradient / observation p90 values are **0.421892 / 0.697287 / 0.148866**, and worst values are **0.539642 / 0.925636 / 0.196865**. Unregularized K1 p90 values are **0.456048 / 0.751378 / 0.156668**; the K2 reference reaches **0.449851 / 0.737940 / 0.116022**.

Fixed regularization therefore improves the five-camera tail and even produces more strict-safe cells than the current K2 reference. It still misses the complete five-camera gate, and the reference is not adequate for a relative-call comparison. The nominal `2A+1AT` versus `3A+2AT` difference cannot be reported as exact-call reduction or speedup.

## Independent recomputation

The independent implementation rebuilds observations, full-DCT coordinates, the fixed-regularized normal solve, all controls, physical CGLS, every cell gate, 13-group tails, and call accounting before comparing against formal arrays. Its status is `PASS_INDEPENDENT_RECOMPUTATION_FIXED_IDENTITY_PRIOR_P14_V199`.

The maximum relative difference for the independent direct-normal solve is about **4.42e-13**, observation reconstruction differs by at most **1.96e-11**, the maximum formal-versus-independent metric difference is about **1.23e-11**, and the maximum summary difference is about **1.13e-11**. Camera-reordering coordinate difference is about **1.82e-14**, while the near-zero stationarity absolute difference is about **5.18e-15**. Formal and input trees remain sealed and the one-shot release is consumed. Frozen numerical kernels and raw inputs are shared, so `end_to_end_physics_independence_proven=false`.

## What succeeded, what failed, and what follows

**Succeeded:** without retuning, fixed regularization adds 95 five-camera strict-safe cells and produces the first 3/13 passing complete groups. The simple-regularization lead from v198 is therefore not confined to a single p22 result.

**Failed:** the candidate does not clear the complete five-camera gate, and more importantly the K2 reference itself reaches 0/13 complete five-camera groups. The run cannot answer whether equivalent accuracy is achieved with fewer calls and must remain inconclusive.

Do not tune `tau`, Krylov depth, or the reference post hoc on p14. Continue only with a preregistered, physically distinct mechanism that first establishes five-camera reference adequacy, or wait for condition-matched real 2D BOST displacement, calibration mapping, repeated-noise data, and an accepted baseline. p14 is not fresh validation; test, wall/RSS, external, and GPU gates remain closed.
