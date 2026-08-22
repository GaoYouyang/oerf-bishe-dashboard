# v194：全法向一步修正失败，对角缩放对照提供新线索

## 为什么做

v193 的 signed CountSketch 在五/九相机下已经达到 `51/52 · 49/52`，说明保留大量弱符号贡献有用，但固定 sketch 本身仍有少数失败。v194 检验一个很窄的机制问题：不改 sketch、不训练模型，只用完整法向残差和 sketch Hessian 做一次固定单位步修正，能否补齐缺口？

## 实际做了什么

- 保持同一个 v193 seed、四个已开发帧、13 套标定、五相机/九相机两臂与未修改的物理 CGLS K1。
- primary 用完整 signed-sketch Hessian 的逆对完整法向残差做一次单位步修正。
- 结果前同时冻结一个便宜对照：只用 Hessian 对角元做逐坐标缩放。
- 没有线搜索、阻尼、ridge、裁剪、额外迭代、sketch 搜索或学习模型。

## 结果

| 一轮未修改 K1 后 | 五相机 | 九相机 |
| --- | ---: | ---: |
| v193 seed | `51/52` | `49/52` |
| full-Hessian primary | `0/52` | `0/52` |
| diagonal control | `52/52` | `52/52` |

full-Hessian primary 不是“擦线失败”，而是在 `104/104` 个单元全部失败。它的坐标修正范数 p50 为 `62.35`，完整法向残差比 p50 为 `17.04`，说明固定单位步产生了明显过冲。对角对照的对应数值只有 `0.795` 和 `2.775`，并在两臂全部过门。

这支持一个机制线索：**当前全 Hessian 单位步中的非对角耦合有害，逐坐标缩放明显更安全。**

## 独立复算

完全独立的第二实现使用不同的特征分解求解链，重建 seed、法向残差、两种修正、物理 K1、逐单元门和调用账，`17/17` 项检查全真。普通数组最大相对差为 `1.06e-10`，近零数组最大绝对差为 `6.51e-11`，相机换序的特征与响应误差都是 `0`。

## 科学判决与边界

正式判决是 `FAIL_SIGNED_SKETCH_FULL_NORMAL_REFINEMENT_V194`。预注册顺序要求 full-Hessian primary 首先通过，才能在 primary 和便宜对照中选择。因此，对角对照的 `104/104` 只能作为 post-open 诊断线索，不能事后替换为 v194 的“成功方法”。

关闭 full-Hessian 单位步 primary，不用步长、迭代数、阻尼、ridge、裁剪、sketch 或模型搜索挽救。对角结果只保留为未来单独预注册机制的设计线索，不用这批已开单元重复证明。

该结果不是完整轨迹、部署算法、exact-call 减少、wall/RSS 收益、外部泛化、curved ray、真实 BOST 或论文成功。`algorithm_breakthrough=false`。

---

# v194: Full-normal one-step refinement fails; the diagonal control provides a new clue

## Why this test

v193 signed CountSketch already reaches `51/52 · 49/52` under five/all-nine cameras. v194 asks one narrow mechanism question without changing the sketch or training a model: can one frozen unit correction, built from the full normal residual and preconditioned by the sketch Hessian, close the remaining gap?

## What was executed

- The same sealed v193 seed, four opened frames, 13 calibrations, five/all-nine sensor arms, and unchanged physical CGLS K1 are retained.
- The primary applies one unit full-normal correction through the complete signed-sketch Hessian.
- A preregistered cheap control uses only the Hessian diagonal for coordinate-wise scaling.
- There is no line search, damping, ridge, clipping, additional iteration, sketch search, normalization search, or learned model.

## Results

| After one unchanged K1 step | Five cameras | All nine |
| --- | ---: | ---: |
| v193 seed | `51/52` | `49/52` |
| Full-Hessian primary | `0/52` | `0/52` |
| Diagonal control | `52/52` | `52/52` |

The full-Hessian primary fails all `104/104` cells. Its median coordinate-correction norm is `62.35`, and the median full-normal-residual ratio is `17.04`, showing strong overshoot. The diagonal control has corresponding medians of only `0.795` and `2.775` and passes both arms completely. This supports the mechanism-level clue that off-diagonal coupling is harmful in this unit-step construction, while coordinate-wise scaling is substantially safer.

## Independent recomputation

A fully separate implementation uses a different eigensolver chain and rebuilds the seed, normal residual, both corrections, physical K1, cell gates, and call ledger. All `17/17` checks pass. Maximum ordinary-array relative and near-zero absolute differences are `1.06e-10 / 6.51e-11`; camera-permutation feature and response errors are both `0`.

## Verdict and boundary

The formal decision is `FAIL_SIGNED_SKETCH_FULL_NORMAL_REFINEMENT_V194`. The preregistered order requires the full-Hessian primary to pass before selecting between the primary and cheap control. The diagonal control's `104/104` result is therefore a post-open diagnostic clue, not a retrospectively selected successful v194 method.

Close the frozen full-Hessian unit-step primary without step-size, iteration, damping, ridge, clipping, sketch, or learned rescue. Retain the diagonal outcome only as a clue for a separately preregistered future mechanism on new evidence.

This is not complete-trajectory evidence, a deployable algorithm, exact-call reduction, wall/RSS speedup, external generalization, curved-ray validation, real BOST, or paper success. `algorithm_breakthrough=false`.
