# v195.2：固定对角 signed-sketch 修正在完整 p22 开发轨迹上失败

## 为什么做

v194 在四帧、13 套标定的已开封哨兵上发现：完整 Hessian 单位步会严重过冲，而逐坐标对角缩放在 `104/104` 个单元通过。那只能算机制线索。v195.2 把这个逐坐标公式单独冻结，并检验它能否覆盖同一条已打开 p22 开发轨迹的全部 `101` 帧，而不是继续用四帧重复证明。

## 实际做了什么

- 固定 `101` 帧、13 套标定、五相机与九相机两臂，共 `2626` 个物理评分单元。
- primary 只做一次已冻结的对角 signed-sketch 修正，再运行一次未修改的物理 CGLS K1。
- 同时公平比较 fit-mean K1、原 signed-seed K1 与同价 full-DCT K1。
- 不搜索步长、阻尼、ridge、裁剪、阈值、sketch、fallback 或模型，不打开 p14 新鲜验证与两条 test。

## 结果

| 方法 | 五相机严格安全单元 | 五相机完整标定 | 九相机严格安全单元 | 九相机完整标定 |
| --- | ---: | ---: | ---: | ---: |
| 固定对角 primary + K1 | `987/1313` | `0/13` | `1234/1313` | `3/13` |
| fit-mean + K1 | - | `0/13` | - | `0/13` |
| signed seed + K1 | - | `0/13` | - | `0/13` |
| full-DCT + K1 | `1310/1313` | `12/13` | `1313/1313` | `13/13` |

primary 在两档相机都失败。五相机的 field / gradient / observation p90 为 `0.474417 / 0.813224 / 0.187029`，其中 gradient p90 越过 `0.75`；worst 为 `0.553433 / 1.001548 / 0.222202`。九相机 p90 为 `0.367619 / 0.651307 / 0.192660`，但 gradient 与 observation worst 分别为 `0.763263 / 0.244199`，仍不能形成完整标定通过。

full-DCT 同价对照显著更强，却也没有完成两档相机门：九相机为 `1313/1313 · 13/13`，五相机仍是 `1310/1313 · 12/13`。因此它只能帮助定位“固定 diagonal sketch 仍丢失了关键坐标信息”，不能事后替换 primary 或包装成成功方法。

## 独立复算

完全独立的第二实现重新构造观测、坐标、候选、物理 K1、逐单元指标、13 个标定组、相机换序检查和调用账，`27/27` 项检查全真。观测相对差为 `0`，坐标相对差最大 `2.38e-14`，指标绝对差最大 `6.77e-15`，汇总差最大 `4.44e-15`，相机射线换序误差为 `0`。

前两次执行在科学评分前因合同或有效性实现错误停止，partial 不进入判决。v195.2 是唯一完成正式评分与独立复算的版本。这些修复属于工程完整性，不是算法进展。

## 科学判决与边界

正式判决是 `FAIL_DIAGONAL_SIGNED_SKETCH_COMPLETE_TRAJECTORY_V195_2`。关闭的范围很窄：**只关闭固定的一步对角 signed-sketch 修正在已打开完整 p22 开发轨迹上的路线**。它不证明 C 路线不可能。

判决在 primary 失败处停止。不能调公式或换名重跑，不能把 full-DCT 对照事后升格，也不打开 p14、资源门、外部门、神经训练或 GPU。后续只有真正新的物理信息，或另行结果前冻结且表示上不同的机制，才值得继续。

这不是部署算法、exact-call 减少、wall/RSS 收益、外部泛化、curved ray、真实 BOST 或论文成功。`algorithm_breakthrough=false`。

---

# v195.2: Frozen diagonal signed-sketch correction fails on the complete p22 development trajectory

## Why this test

v194 found a useful clue on four opened frames and 13 calibrations: the full-Hessian unit step overshot badly, while coordinate-wise diagonal scaling passed `104/104` sentinel cells. That was not complete-trajectory evidence. v195.2 separately freezes the coordinate-wise formula and tests all `101` frames of the same already-opened p22 development trajectory instead of reusing the four sentinels as proof.

## What was executed

- The test covers `101` frames, 13 calibrations, five-camera and all-nine-camera arms, and `2626` physical scoring cells.
- The primary applies the frozen diagonal signed-sketch correction once, followed by one unchanged physical CGLS K1 step.
- Fit-mean K1, the original signed-seed K1, and equal-call full-DCT K1 are evaluated as controls.
- There is no search over step size, damping, ridge, clipping, thresholds, sketches, fallback, or learned models. Fresh p14 validation and both tests remain unopened.

## Results

| Method | Five-camera strict-safe cells | Five-camera complete groups | All-nine strict-safe cells | All-nine complete groups |
| --- | ---: | ---: | ---: | ---: |
| Frozen diagonal primary + K1 | `987/1313` | `0/13` | `1234/1313` | `3/13` |
| Fit-mean + K1 | - | `0/13` | - | `0/13` |
| Signed seed + K1 | - | `0/13` | - | `0/13` |
| Full-DCT + K1 | `1310/1313` | `12/13` | `1313/1313` | `13/13` |

The primary fails both sensor arms. Five-camera field / gradient / observation p90 values are `0.474417 / 0.813224 / 0.187029`; gradient p90 exceeds `0.75`, and worst values are `0.553433 / 1.001548 / 0.222202`. All-nine p90 values are `0.367619 / 0.651307 / 0.192660`, but gradient and observation worst values remain above their limits at `0.763263 / 0.244199`, preventing complete calibration-group success.

The equal-call full-DCT control is substantially stronger but still does not pass the complete two-sensor gate: all-nine reaches `1313/1313 · 13/13`, while five-camera reaches only `1310/1313 · 12/13`. This helps localize information lost by the fixed diagonal sketch, but it cannot replace the primary post hoc or be presented as a successful method.

## Independent recomputation

A fully separate implementation rebuilds observations, coordinates, candidates, physical K1, cell metrics, all 13 calibration groups, camera-reordering checks, and the call ledger. All `27/27` checks pass. Maximum observation relative, coordinate relative, metric absolute, and summary absolute differences are `0 / 2.38e-14 / 6.77e-15 / 4.44e-15`; camera-ray permutation error is `0`.

Two earlier executions stopped before scientific scoring because of contract or validity-implementation errors, and their partial artifacts are excluded. v195.2 is the only version completing formal scoring and independent recomputation. These repairs establish engineering integrity, not algorithmic progress.

## Verdict and boundary

The scientific decision is `FAIL_DIAGONAL_SIGNED_SKETCH_COMPLETE_TRAJECTORY_V195_2`. The scope is narrow: **close only the fixed one-step diagonal signed-sketch correction on the already-opened complete p22 development trajectory**. This does not prove the broader C route impossible.

Adjudication stops at primary failure. Do not tune or rename the formula, promote full-DCT after results, or open p14, resource gates, external gates, neural training, or GPU use. Continue only with genuinely new physical information or a separately preregistered, representationally distinct mechanism.

This is not a deployable algorithm, exact-call reduction, wall/RSS gain, external generalization, curved-ray validation, real BOST, or paper success. `algorithm_breakthrough=false`.
