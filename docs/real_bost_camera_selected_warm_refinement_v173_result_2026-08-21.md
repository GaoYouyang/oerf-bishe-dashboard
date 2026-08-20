# v173：完整 warm refinement 过门，但更便宜的 H1-K0 解释了结果

更新：2026-08-21

## 先说结论

v173 把 v172 通过三重隔离的五相机选择策略真正接回重建链：对每个留出的标定、完整三维场和时间单元，先在已选五相机观测上做固定 H1 初始化，再运行一次未修改 CGLS。

这条 `H1-K1` 主策略本身表现很好：

- `468/468` 个单元严格安全；
- 完整标定、完整场和时间分别通过 `13/13 · 9/9 · 4/4`；
- field / gradient / observation 的全局 p90 为 `0.326808 / 0.610169 / 0.075055`；
- 相对 Zero-K4 没有伤害；
- 逻辑在线账为 `2A+2A^T`。

但唯一通过的阻断对照更便宜：使用同一五相机子集与同一 H1 初始化、完全不做 CGLS 的 `H1-K0`，同样达到 `468/468`、`13/13 · 9/9 · 4/4`，field / gradient / observation p90 为 `0.327496 / 0.621204 / 0.118422`，逻辑在线账只有 `1A+1A^T`。

因此独立复算后的科学判决是：

`FAIL_CLASSICAL_CONTROL_EXPLAINS_CAMERA_SELECTED_WARM_V173`

精确含义不是“完整链算错了”，而是：**增加一次未修改 CGLS K1 没有建立稳定优势，因为去掉它的同子集经典 H1-K0 已经以一半 exact-call 账通过全部门。**

## 公平对照

| 方法 | 严格安全单元 | 完整标定 / 场 / 时间 | exact `A / A^T` | 完整判决 |
| :--- | ---: | :---: | :---: | :---: |
| Selected H1-K1 | 468 / 468 | 13 / 13 · 9 / 9 · 4 / 4 | 2 / 2 | PASS |
| Selected H1-K0 | 468 / 468 | 13 / 13 · 9 / 9 · 4 / 4 | 1 / 1 | PASS，阻断 K1 优势 |
| Zero-K2 | 0 / 468 | 0 / 13 · 0 / 9 · 0 / 4 | 2 / 2 | FAIL |
| Zero-K4 | 0 / 468 | 0 / 13 · 0 / 9 · 0 / 4 | 4 / 4 | 绝对门 FAIL；作为 matched reference |
| BP-K1 | 0 / 468 | 0 / 13 · 0 / 9 · 0 / 4 | 2 / 2 | FAIL |
| Jacobi PCGLS-K2 | 0 / 468 | 0 / 13 · 0 / 9 · 0 / 4 | 2 / 2 | FAIL |
| Fit-static H1-K1 | 334 / 468 | 1 / 13 · 0 / 9 · 0 / 4 | 2 / 2 | FAIL |
| v169 fixed H1-K1 | 222 / 468 | 0 / 13 · 0 / 9 · 0 / 4 | 2 / 2 | FAIL |

Zero-K4 仍是冻结的相对精度参考。它自己没有通过当前绝对场/梯度门，并不矛盾；主策略和 H1-K0 同时还要满足绝对门与相对 Zero-K4 的无害门。

## H1-K1 与 H1-K0 的差别

| 方法 | field p90 / worst | gradient p90 / worst | observation p90 / worst | exact calls |
| :--- | :--- | :--- | :--- | :--- |
| Selected H1-K1 | 0.326808 / 0.410351 | 0.610169 / 0.721757 | 0.075055 / 0.084898 | 2A + 2A^T |
| Selected H1-K0 | 0.327496 / 0.410876 | 0.621204 / 0.731543 | 0.118422 / 0.145829 | 1A + 1A^T |

K1 的确把三个误差都进一步降低，尤其是观测残差；但论文目标不是“误差再低一点就算赢”，而是在完整精度门已经满足时，用更少昂贵调用获得等价的最终结果。H1-K0 已经完成这一点，所以不能把 K1 写成必要贡献。

## 独立复算

独立第二实现重新生成全部 `468` 个单元、八个方法、物理场、观测残差、六项绝对门、相对 Zero-K4 的无害门和真实调用账。

`21/21` 项检查全部通过：

- 逐单元指标最大差 `1.64e-11`；
- 方法汇总最大差 `8.66e-12`；
- exact-call 最大差 `0`；
- 主策略 field 与 residual 相对差分别约 `1.47e-11` 与 `3.87e-11`；
- 全部离散通过/失败、阻断对照和科学判决完全一致。

## 下一步为什么变了

当前 H1+K1 warm-refinement 主张已经关闭，不再围绕 K1 调步数、调正则或换更大网络。仍值得单独回答的窄问题是：**在所有相机选择策略都使用同一个更便宜的 H1-K0 时，v172 的部署可见 selector 是否仍优于 fit-static、v169 和结果不可见的确定性几何对照？**

这会把“相机选得好”与“后端多做一步”彻底拆开。如果同成本 selector-only 门也失败，当前 learned contribution 就应关闭；如果通过，也只能称相机选择 headroom，仍需 fresh 外部工况、真实资源和配对实验 BOST。

本轮没有 wall/RSS 实测，没有工况匹配的实验二维位移，也没有 curved-ray 验证。几何 cache 和离线真值观测成本与逻辑在线账分开披露。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v173: the full warm refinement passes, but cheaper H1-K0 explains the result

Updated: 2026-08-21

v173 connects the v172 triple-isolated five-camera policy to the reconstruction chain. For each held-out calibration, complete 3D field, and time cell, it computes the fixed H1 initializer on the selected observations and then runs one unchanged CGLS step.

The `H1-K1` primary is safe on all `468/468` cells and clears `13/13` calibrations, `9/9` fields, and `4/4` times. Global field, gradient, and observation p90 values are `0.326808 / 0.610169 / 0.075055`, with no harm relative to Zero-K4. Its logical online ledger is `2A+2A^T`.

The blocking control is cheaper. With the same selected subset and the same H1 initializer but no CGLS refinement, `H1-K0` also clears `468/468` cells and `13/13 · 9/9 · 4/4` complete axes. Its field, gradient, and observation p90 values are `0.327496 / 0.621204 / 0.118422`, while its logical ledger is only `1A+1A^T`.

Independent recomputation therefore returns:

`FAIL_CLASSICAL_CONTROL_EXPLAINS_CAMERA_SELECTED_WARM_V173`

The exact conclusion is not that the integrated chain is incorrect. It is that the extra unchanged CGLS K1 step has no established stable advantage because the same-subset classical H1-K0 control already passes every gate with half the exact forward/adjoint calls.

Zero-K2, Zero-K4, BP-K1, and Jacobi PCGLS-K2 are strict-safe on `0/468` cells. Fit-static H1-K1 reaches `334/468`, while the frozen v169 H1-K1 control reaches `222/468`; neither passes a complete field or time axis. H1-K0 is the only blocking control that passes.

The independent implementation rebuilds all 468 cells, eight arms, physical fields, residuals, absolute gates, matched Zero-K4 checks, and exact-call receipts. All `21/21` checks pass. Maximum per-cell metric and arm-summary differences are `1.64e-11` and `8.66e-12`, the maximum call difference is zero, and all discrete decisions agree.

The H1+K1 warm-refinement claim now closes. The remaining narrow question is whether the deployment-visible selector itself adds value when every policy uses the same cheaper H1-K0 reconstruction. The next gate will compare the v172 selector with fit-static, v169, and result-free deterministic geometry controls at identical `1A+1A^T` cost. Even a pass would establish only camera-selection headroom. It will not add K1, a larger model, or GPU training.

No fresh wall/RSS measurement, condition-matched experimental 2D displacement, or curved-ray validation was performed. Geometry-cache and offline truth-observation work remain separate from the logical online ledger.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `curved_ray_validated=false`, `real_bost=false`.
