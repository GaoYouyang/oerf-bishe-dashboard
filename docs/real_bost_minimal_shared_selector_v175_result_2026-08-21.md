# v175：最小共享 CPU 相机选择器通过完整场外折门

更新：2026-08-21

## 结论

v174 证明“怎样选相机”本身有开发集价值，但它复用了 v172 已封存的逐场景选择。v175 把这一步真正收缩成一个小模型：每个外折只拟合一个标量 Gram-ridge 风险函数，最多 `357` 个参数；推理时只读取报告相机几何，不读取三维场、时间、观测误差或真值。

外折同时留出一套完整标定和一个完整三维场，共 `13 × 9 = 117` 折。每折只输出一个五相机子集，并把同一个选择用于该留出场的四个时间点。独立复算后的结果为：

- 最小共享选择器：严格安全 `468/468`，完整标定 / 三维场 / 时间 `13/13 · 9/9 · 4/4`；
- fit-static：`328/468`，完整组 `1/13 · 0/9 · 0/4`；
- v169 low-mode D-opt：`192/468`，完整组 `0/13 · 0/9 · 0/4`；
- ray-axis maximin：`455/468`，完整组 `9/13 · 4/9 · 1/4`。

三个便宜对照都没有完整通过，科学判决是：

`PASS_MINIMAL_SHARED_SELECTOR_HEADROOM_V175`

这说明在已开封的受控 straight-ray 代理上，**一个很小、共享参数、时间不变的 CPU 相机选择器确实可以工作**。它不再依赖为每个时间切换不同输出，因此比 v174 更接近可执行的选择机制。

## 公平物理重放

| 策略 | 严格安全单元 | 完整标定 / 场 / 时间 | field / gradient / observation p90 | exact `A / A^T` | 判决 |
| :--- | ---: | :---: | :--- | :---: | :---: |
| Minimal shared Gram-ridge | 468 / 468 | 13 / 13 · 9 / 9 · 4 / 4 | 0.327494 / 0.620640 / 0.118422 | 1 / 1 | PASS |
| Fit-static | 328 / 468 | 1 / 13 · 0 / 9 · 0 / 4 | 0.339055 / 0.861694 / 0.131662 | 1 / 1 | FAIL |
| v169 low-mode D-opt | 192 / 468 | 0 / 13 · 0 / 9 · 0 / 4 | 0.324087 / 0.882023 / 0.134096 | 1 / 1 | FAIL |
| Ray-axis maximin | 455 / 468 | 9 / 13 · 4 / 9 · 1 / 4 | 0.274764 / 0.712584 / 0.128909 | 1 / 1 | FAIL |

所有策略都使用同一个固定 `H1-K0` 重建和 `1A+1A^T` 逻辑在线账，并各自与同一相机子集的 Zero-K4 比较。最小共享选择器的 matched-reference harm 和 severe harm 都是 `0`。

选择器训练不增加 exact forward/adjoint；报告的 `338` 次特征 setup 与 `13299` 个 forward-equivalent 几何缓存构建是离线准备成本，不能写成部署速度。fresh wall 和 whole-pipeline RSS 尚未测量。

## 隔离与独立复算

每个外折在构造标签前排除留出标定和留出完整三维场。对这两组留出标签做大幅数值突变后，训练目标和输出变化仍为 `0`。一个选择在四个留出时间上保持不变。

独立第二实现使用增广最小二乘代替正式正规方程，并以不同的稀疏算子、解析 DCT、H1 解法和 CGLS 递推重建全部物理结果。`31/31` 项检查全部通过：

- 预测风险最大差 `2.19e-11`；
- 逐单元物理指标最大差 `1.62e-11`；
- 候选场和残差最大相对差 `1.47e-11 / 2.80e-11`；
- selector fit 汇总最大差 `2.01e-12`；
- adjoint identity 最大相对误差 `1.78e-16`；
- exact-call 差为 `0`；
- 所有选择、通过/失败与最终判决完全一致。

## 下一门与证据边界

v175 允许冻结一个使用全部开发数据的最终小模型，并在**此前未打开的公开反应流工况**上做一次外部门。只有外部门继续守住 matched accuracy，才值得进行 fresh-process wall/RSS；在 CPU 小模型已经足够的当前阶段，没有租 GPU 或训练大型 FNO/UNO/U-Net 的理由。

这仍是 post-open 受控代理证据。它没有使用工况匹配的实验二维位移，也没有证明外部泛化、真实资源收益、curved ray 或真实 BOST。

因此这是重要里程碑，但还不是突破性结论：

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v175: a minimal shared CPU camera selector passes complete-field outer isolation

Updated: 2026-08-21

v174 established selector-only development headroom but replayed selections already sealed by v172. v175 compresses the selection mechanism into one scalar Gram-ridge risk model per outer fold, with at most `357` parameters. Inference reads reported camera geometry only; it reads no 3D field, time, outcome error, or truth.

Each of the `13 × 9 = 117` folds leaves out one complete calibration and one complete 3D field. The fold emits one five-camera subset, and that same selection is used for all four times of the held-out field.

After independent recomputation, the minimal shared selector is strict-safe on `468/468` cells and clears `13/13` calibrations, `9/9` complete 3D fields, and `4/4` time strata. Fit-static reaches `328/468` and `1/13 · 0/9 · 0/4`; v169 low-mode D-opt reaches `192/468` and `0/13 · 0/9 · 0/4`; ray-axis maximin reaches `455/468` and `9/13 · 4/9 · 1/4`. No cheap control passes completely.

Decision: `PASS_MINIMAL_SHARED_SELECTOR_HEADROOM_V175`.

Every policy uses the same fixed H1-K0 reconstruction, the same logical online ledger of `1A+1A^T`, and its own same-subset Zero-K4 reference. The primary field / gradient / observation p90 values are `0.327494 / 0.620640 / 0.118422`, with zero matched-reference harm and zero severe harm.

The held-out calibration and complete-field outcomes are excluded before target reduction. Mutating both held-out label groups changes the fit target by exactly zero, and each fold's selection is constant across all four held-out times.

The independent implementation uses augmented least squares instead of formal normal equations, together with different sparse-operator, analytic-DCT, H1, and CGLS paths. All `31/31` checks pass. Maximum predicted-risk and physical-metric differences are `2.19e-11` and `1.62e-11`; candidate-field and residual relative differences are `1.47e-11` and `2.80e-11`; exact-call difference is zero; all discrete decisions agree.

This is substantive evidence for a minimal shared CPU selector on the opened controlled proxy. It authorizes one previously unopened public reacting-flow external gate, followed by fresh wall/RSS only if that gate passes. It does not authorize GPU rental or a larger neural operator.

No condition-matched experimental 2D displacement, external generalization, fresh resource result, curved-ray validation, or real-BOST result is present. This is an important milestone, not an algorithmic breakthrough: `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `curved_ray_validated=false`, `real_bost=false`.
