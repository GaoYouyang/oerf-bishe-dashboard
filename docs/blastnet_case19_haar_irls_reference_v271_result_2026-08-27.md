# v271：Haar-IRLS reference 两条数值路径各自过门，但跨实现不一致

## 讲人话结论

v271 不是在训练新模型，而是在修复一个更基础的问题：Case 19 的 K16 reference 自身不够稳定，所以先尝试用固定的一层三维 Haar 表示和 12 轮 smoothed-L1 IRLS 构造更充分的首帧 reference。正式实现使用 LSMR，独立第二实现重新构造 Haar 与算子，并用反向列序的 LSQR 求解；相机、数据、阈值和停止规则均未按结果调整。

两条路径单独看都很漂亮：formal 和 independent 各自在 `13/13` 套九相机上通过 field、full-gradient、interior-gradient 与 observation 绝对门。正式有效性门为 `18/18`。然而独立验证只通过 `24/31` 项，关键的跨实现一致性失败：场相对差 `6.0517e-3`、指标差 `1.8435e-3`、目标值相对差 `2.3628e-4`、汇总差 `1.8435e-3`，都高于结果前冻结的 `2e-5` 门。归一化残差差 `4.9282e-6` 和观测差 `8.38e-17` 虽在界内，不能覆盖场与指标的不一致。

因此 `13/13` 只能作为诊断，不能称 reference 通过。权威判决是 `INCONCLUSIVE_INVALID_CASE19_HAAR_IRLS_REFERENCE_V271`。这条结果排除了一个危险的假阳性：只运行其中任意一套代码，都会得到“全过”的表象；独立第二路径显示该固定 IRLS reference 对求解路径仍有材料性敏感。

固定 Haar-IRLS reference 到此关闭，不重跑、不放宽 `2e-5` 门，也不调整轮数、平滑参数、solver 或正则。该结果没有建立 warm initializer、matched-accuracy、有效减调用、完整序列、wall/RSS、外部泛化或真实 BOST。它定位的是数值参考的不稳定，不是否定整条 C 路线。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v271: Both Haar-IRLS paths clear the absolute gates, but cross-implementation agreement fails

## Plain-language conclusion

v271 does not train a new model. It addresses a more basic blocker: the Case 19 K16 reference is inadequate, so the audit tests a fixed one-level 3D Haar representation with twelve rounds of smoothed-L1 IRLS as a potentially stronger frame-zero reference. The formal path uses LSMR. A separate implementation rebuilds the Haar system and operator and solves it with reverse-column LSQR. Cameras, data, thresholds, and stopping rules remain fixed.

Each path looks excellent in isolation. Formal and independent results each clear the field, full-gradient, interior-gradient, and observation absolute gates on `13/13` nine-camera rigs, while formal validity reaches `18/18`. Independent validation nevertheless reaches only `24/31` checks. Cross-implementation field difference is `6.0517e-3`, metric difference `1.8435e-3`, objective relative difference `2.3628e-4`, and summary difference `1.8435e-3`, all above the preregistered `2e-5` limit. The normalized-residual difference of `4.9282e-6` and observation difference of `8.38e-17` remain within bounds but cannot override the field and metric failures.

The two `13/13` counts are therefore diagnostic only, not a valid reference pass. The authoritative verdict is `INCONCLUSIVE_INVALID_CASE19_HAAR_IRLS_REFERENCE_V271`. This prevents a consequential false positive: either implementation alone would report a complete absolute pass, while the independent numerical path reveals material solver-path sensitivity.

The fixed Haar-IRLS reference closes without rerun, relaxation of the `2e-5` gate, or tuning of rounds, smoothing, solver, or regularization. It establishes no warm initializer, matched accuracy, effective call reduction, full sequence, wall/RSS, external generalization, or real BOST. It localizes numerical reference instability and does not reject the broader C route.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
