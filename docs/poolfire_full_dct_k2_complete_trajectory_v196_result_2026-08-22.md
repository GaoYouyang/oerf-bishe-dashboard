# v196：稠密 full-DCT K2 全部过门，但冻结 Zero-K4 参考不足

## v196.1 后续纠偏：reference 身份审计已完成

独立审计确认，v196 五相机 Zero-K4 与 v176 的 primary selected-subset Zero-K4 是同一个物理参考。两者在共同的 `13 x 4 x 3 = 156` 个指标上最大绝对差仅 `5.55e-17`；两份保留场逐位一致。关键问题不在数值复现，而在合同可判别性：v176 在 v196 冻结前已经独立封存该参考为 `0/52`，v175 也曾在候选过门时把自身 `0/468` 的 Zero-K4 称作 reference。

因此，v196 的“Zero-K4 任一严格单元失败就记 inconclusive”分支在冻结前已被旧证据预定触发。v196 的计算、`2626/2626` 的 full-DCT K2 绝对结果和原始 `INCONCLUSIVE_REFERENCE_ZERO_K4_INADEQUATE_V196` 判决均保持有效；纠偏的是解释边界：**v196 不能再描述为一次前瞻性的 comparative-headroom 检验**。full-DCT K2 只能保留为 post-open 绝对容量诊断，不能据此声称相对调用减少。没有事后替换 reference。

## 为什么做

v195.2 显示 full-DCT K1 在九相机已经完整通过，五相机只剩 `3/1313` 个单元和 `1/13` 个标定组未过。v196 结果前单独冻结一个不可与 v195 互换的问题：在稠密 full-DCT 起点上再做一步未修改 CGLS，能否跨过完整 p22 开发轨迹，并相对冻结的 Zero-CGLS K2/K3/K4 参考建立可比较的 headroom。

## 实际做了什么

- 保持同一条已开封 p22 开发轨迹：`101` 帧、13 套标定、五相机与九相机两臂，共 `2626` 个物理评分单元。
- 比较 full-DCT K1 父臂、full-DCT K2、Zero-CGLS K2/K3 与预注册 Zero-CGLS K4 reference。
- 逻辑在线调用分别为 `2A+1A^T`、`3A+2A^T`、`2A+2A^T`、`3A+3A^T` 和 `4A+4A^T`。几何 cache 的 `26 x 1009` 次 setup projection 单独披露，不算进在线调用收益。
- 没有训练参数、没有紧凑表示、没有调参或 fallback；p14 与两条 test 继续封存。

## 结果

| 方法 | 逻辑在线账 | 五相机严格单元 / 完整标定 | 九相机严格单元 / 完整标定 |
| --- | ---: | ---: | ---: |
| full-DCT K1 parent | `2A+1A^T` | `1310/1313 · 12/13` | `1313/1313 · 13/13` |
| **full-DCT K2** | `3A+2A^T` | **`1313/1313 · 13/13`** | **`1313/1313 · 13/13`** |
| Zero-CGLS K2 | `2A+2A^T` | `0/1313 · 0/13` | `0/1313 · 0/13` |
| Zero-CGLS K3 | `3A+3A^T` | `0/1313 · 0/13` | `0/1313 · 0/13` |
| **Zero-CGLS K4 reference** | `4A+4A^T` | **`0/1313 · 0/13`** | **`0/1313 · 0/13`** |

full-DCT K2 的五相机 field / gradient / observation p90 为 `0.363959 / 0.599450 / 0.098924`，九相机为 `0.249912 / 0.417821 / 0.088978`，两臂全部过冻结绝对精度门。这是一条很强的诊断线索：稠密起点再走一步 CGLS 在当前已打开轨迹上精度足够。

但冻结的 Zero-K4 reference 并没有成为可接受参考：两臂都是 `0/1313 · 0/13`。其五相机 field / gradient / observation p90 为 `0.872453 / 0.752918 / 0.281930`，九相机为 `0.813283 / 0.668945 / 0.313174`。因此它不能承担“候选相对参考等精度”的基础。

## 独立复算

完全独立的第二实现重建物理算子、各 CGLS 臂、候选场、观测、逐单元门、13 个标定组与调用账，`23/23` 项检查全真。正式/独立的指标、残差、哨兵和汇总差全部为 `0`；观测相对误差与 K1 父臂误差均为 `0`，相机换序相对误差约 `9.46e-17`。

独立程序生成 truth observation 时有 `2626` 次离线 forward，这是独立评分构造，与上表的逻辑在线 arm 账分开。两实现仍共享冻结 physics kernel 定义，所以 `end_to_end_physics_independence_proven=false`。

## 科学判决与边界

正式判决是 `INCONCLUSIVE_REFERENCE_ZERO_K4_INADEQUATE_V196`。结果前冻结的顺序规定：**先检查 reference 是否足够；reference 失效时，必须停在 inconclusive**。因此，不能因 full-DCT K2 绝对过门就宣称相对 headroom、exact-call 减少、速度或算法成功，也不能在看到结果后为 v196 更换 reference。

v196.1 已完成 reference 身份追溯，并证明旧 reference 在冻结前已知不充分。下一步必须先单独冻结并独立建立一个不可交换、在候选结果出现前就通过绝对充分性门的 reference，之后才能评价新的候选。p14、test、wall/RSS、预测器、神经训练和 GPU 仍不授权。这不是紧凑或学习初始化器，也不是外部泛化、真实 BOST 或论文成功。`algorithm_breakthrough=false`。

---

# v196: Dense full-DCT K2 passes every gate, but the frozen Zero-K4 reference is inadequate

## v196.1 follow-up correction: the reference-identity audit is complete

An independent audit confirms that the five-camera Zero-K4 arm in v196 is the same physical reference as the primary selected-subset Zero-K4 arm in v176. Across the shared `13 x 4 x 3 = 156` metrics, the maximum absolute difference is only `5.55e-17`, and two retained fields are bitwise identical. The problem is therefore not numerical reproduction but protocol informativeness: v176 had independently sealed this reference at `0/52` before v196 was frozen, while v175 had also called a `0/468` Zero-K4 arm a reference when its candidate passed.

The v196 branch stating that any strict-cell failure by Zero-K4 forces an inconclusive verdict was thus predetermined by evidence available before freeze. The v196 computation, the `2626/2626` absolute full-DCT K2 result, and the original `INCONCLUSIVE_REFERENCE_ZERO_K4_INADEQUATE_V196` decision remain valid. The correction is interpretive: **v196 is not a prospective comparative-headroom test**. Full-DCT K2 remains only a post-open absolute-capacity diagnostic and cannot support a relative exact-call claim. No reference is replaced post hoc.

## Why this test

v195.2 showed that full-DCT K1 already passed the complete all-nine arm and missed only `3/1313` cells and `1/13` calibration groups under five cameras. v196 separately preregisters a non-exchangeable question: does one additional unchanged CGLS step from the dense full-DCT starting point pass the complete opened p22 development trajectory and establish comparable headroom against frozen Zero-CGLS K2/K3/K4 references?

## What was executed

- The same opened p22 development trajectory is retained: `101` frames, 13 calibrations, five-camera and all-nine arms, and `2626` physical scoring cells.
- Full-DCT K1, full-DCT K2, Zero-CGLS K2/K3, and the preregistered Zero-CGLS K4 reference are compared.
- Their logical online ledgers are `2A+1A^T`, `3A+2A^T`, `2A+2A^T`, `3A+3A^T`, and `4A+4A^T`. The geometry cache uses `26 x 1009` setup projections and is disclosed separately from online calls.
- There are no trainable parameters, compact representation, tuning, or fallback. Fresh p14 validation and both tests remain sealed.

## Results

| Method | Logical online ledger | Five-camera strict cells / complete groups | All-nine strict cells / complete groups |
| --- | ---: | ---: | ---: |
| Full-DCT K1 parent | `2A+1A^T` | `1310/1313 · 12/13` | `1313/1313 · 13/13` |
| **Full-DCT K2** | `3A+2A^T` | **`1313/1313 · 13/13`** | **`1313/1313 · 13/13`** |
| Zero-CGLS K2 | `2A+2A^T` | `0/1313 · 0/13` | `0/1313 · 0/13` |
| Zero-CGLS K3 | `3A+3A^T` | `0/1313 · 0/13` | `0/1313 · 0/13` |
| **Zero-CGLS K4 reference** | `4A+4A^T` | **`0/1313 · 0/13`** | **`0/1313 · 0/13`** |

Full-DCT K2 five-camera field / gradient / observation p90 values are `0.363959 / 0.599450 / 0.098924`; all-nine values are `0.249912 / 0.417821 / 0.088978`. Every cell and calibration group passes the frozen absolute gate. This is a strong diagnostic clue that the dense start followed by one additional CGLS step is accurate on the opened trajectory.

The frozen Zero-K4 reference is not adequate: both arms are `0/1313 · 0/13`. Its five-camera field / gradient / observation p90 values are `0.872453 / 0.752918 / 0.281930`; all-nine values are `0.813283 / 0.668945 / 0.313174`. It therefore cannot support a matched-accuracy comparison.

## Independent recomputation

A fully separate implementation rebuilds the physics operator, every CGLS arm, candidate fields, observations, cell gates, all 13 calibration groups, and call ledgers. All `23/23` checks pass. Formal/independent metric, residual, sentinel, and summary differences are exactly `0`; observation relative error and K1-parent error are both `0`, and camera-reordering relative error is approximately `9.46e-17`.

The independent truth-observation construction uses `2626` offline forwards, explicitly separate from each logical online arm. Both implementations still share the frozen physics-kernel definition, so `end_to_end_physics_independence_proven=false`.

## Verdict and boundary

The formal decision is `INCONCLUSIVE_REFERENCE_ZERO_K4_INADEQUATE_V196`. The result-blind order requires reference adequacy to be checked first; **an inadequate reference must stop adjudication as inconclusive**. Full-DCT K2's absolute pass therefore cannot establish relative headroom, exact-call reduction, speedup, or algorithm success, and the reference cannot be replaced post hoc inside v196.

v196.1 completes the identity audit and shows that the old reference was already known to be inadequate before freeze. The next gate must first freeze and independently establish a non-exchangeable reference that satisfies the absolute adequacy gate before any candidate result is seen; only then may another candidate be evaluated. p14, tests, wall/RSS, predictor work, neural training, and GPU use remain unauthorized. This is not a compact or learned initializer, external generalization, real BOST, or paper success. `algorithm_breakthrough=false`.
