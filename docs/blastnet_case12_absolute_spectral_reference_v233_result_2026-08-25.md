# v233/v233.1：稳定的观测拟合仍不能作为合格的三维 reference

## 结论

v232.1 已经证明，继续加深 PCGLS 会放大浮点级差异，因此当前 deep-PCGLS reference 壳不能释放。v233 改用一个物理和数值上不同的绝对 reference：在 `32x16x16` 场上固定零均值 DCT1024 表示，用只依赖观测与已知几何的机器精度 ridge，一次求解 1023 个非恒定系数。DCT 维数、零均值约束、ridge 公式、13 个 rig、每 rig 46 帧、598 个单元和全部绝对精度门都在结果前固定。

数值问题解决了。正式实现使用薄 SVD，独立实现显式重建余弦基、逐列物理 forward，并用 Gram/Cholesky 求解。独立程序重算全部 598 个单元后，`17/17` 项检查全真；正式与独立的场相对差最大为 `1.24529e-13`，指标绝对差最大为 `6.21725e-14`，汇总差最大为 `3.26406e-14`，最大归一化驻点残差为 `1.24109e-16`。这说明 v233 的失败不是求解器漂移或相机换序造成的。

科学充分性没有通过。严格安全单元为 `0/598`，完整 rig 为 `0/13`。p90-higher 结果与冻结门如下：

| 指标 | v233 | 冻结上限 | 判定 |
| --- | ---: | ---: | --- |
| Field | `0.820180` | `0.500000` | 失败 |
| Full gradient | `1.231545` | `0.750000` | 失败 |
| Interior gradient | `0.779164` | `0.750000` | 失败 |
| Observation | `0.133957` | `0.200000` | 通过 |

最差值也显示同一个结构：observation 为 `0.202491 <= 0.350000`，但 field、full-gradient 和 interior-gradient 分别为 `0.914075 > 0.750000`、`1.367261 > 1.000000` 和 `1.055472 > 1.000000`。

**讲人话：** 这套 reference 能把九相机二维观测拟合得很好，却恢复出了错误的三维场和梯度。二维投影吻合并不自动意味着三维重建正确。正式科学判决是：

`FAIL_INADEQUATE_CASE12_ABSOLUTE_SPECTRAL_REFERENCE_V233`

因此固定 DCT1024 + machine-ridge 绝对 reference 关闭；不再调整 DCT rank、ridge、基、截断或精度门。这个结果没有裁决 dual-PRESS，也没有证明不存在别的合格 reference。候选策略、exact-call 减少、wall/RSS、外部泛化、curved ray 与真实 BOST 仍不可解释。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v233/v233.1: a stable observation fit is still not an adequate 3D reference

## Conclusion

v232.1 showed that deeper PCGLS amplifies roundoff-scale differences, so the current deep-PCGLS reference shell cannot be released. v233 uses a physically and numerically different absolute reference: a fixed zero-mean DCT1024 representation on the `32x16x16` field, with a machine-precision ridge derived only from observations and known geometry, solving the 1023 nonconstant coefficients in one step. The DCT dimension, gauge constraint, ridge formula, 13 rigs, 46 frames per rig, 598 cells, and all absolute-accuracy gates were frozen before results.

The numerical problem is resolved. The formal implementation uses a thin SVD; the independent implementation explicitly rebuilds the cosine basis, applies the physical forward map column by column, and solves through Gram/Cholesky. After independently recomputing all 598 cells, all `17/17` checks pass. Maximum formal-independent field-relative, metric-absolute, and summary differences are `1.24529e-13`, `6.21725e-14`, and `3.26406e-14`, while maximum normalized stationarity is `1.24109e-16`. The v233 failure is therefore not solver drift or camera permutation.

Scientific adequacy fails. The reference reaches `0/598` strict-safe cells and `0/13` complete rigs. Its p90-higher values versus the frozen limits are:

| Metric | v233 | Frozen limit | Decision |
| --- | ---: | ---: | --- |
| Field | `0.820180` | `0.500000` | Fail |
| Full gradient | `1.231545` | `0.750000` | Fail |
| Interior gradient | `0.779164` | `0.750000` | Fail |
| Observation | `0.133957` | `0.200000` | Pass |

The worst-case values show the same structure: observation is `0.202491 <= 0.350000`, while field, full-gradient, and interior-gradient are `0.914075 > 0.750000`, `1.367261 > 1.000000`, and `1.055472 > 1.000000`.

In plain language, this reference fits the nine-camera 2D observations well while recovering the wrong 3D field and gradients. Agreement in projection space does not establish correct volumetric reconstruction. The scientific decision is `FAIL_INADEQUATE_CASE12_ABSOLUTE_SPECTRAL_REFERENCE_V233`.

The fixed DCT1024 machine-ridge absolute reference is closed, with no DCT-rank, ridge, basis, cutoff, or gate retuning. This does not adjudicate dual-PRESS and does not prove that no adequate reference exists. Candidate policy, exact-call reduction, wall/RSS, external generalization, curved rays, and real BOST remain uninterpretable.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
