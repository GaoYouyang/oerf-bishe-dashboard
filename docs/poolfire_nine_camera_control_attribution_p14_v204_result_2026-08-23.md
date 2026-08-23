# v203-v204：九相机救回全部五相机失败，但余量仍依赖稠密表示

## 结论先说

v203 和 v204 没有重新生成三维场，也没有调整算法参数。它们只审计此前已经独立封存的 p14 五/九相机结果，回答两个按顺序冻结的问题：

1. 五相机 TGV2 剩下的 24 个失败，增加到九相机后是否仍失败？
2. 九相机 K2 若能通过，是否已经有更便宜的封存对照也能在完整 1313 个单元上通过？

结果很清楚。九相机 full-DCT K2 把 **24/24** 个五相机失败全部救回；在全部 **1313** 个单元和 **13** 个标定组上，unregularized full-DCT K1、fixed-identity full-DCT K1 和 full-DCT K2 都达到 **1313/1313 · 13/13**。K1 的逻辑在线账为 `2A+1A^T`，比 K2 的 `3A+2A^T` 少一次 `A` 和一次 `A^T`。

但是，Zero、BP-CGLS1、Zero-CGLS K2 和 affine Jacobi-PCGLS1 都是 **0/1313**；initializer-only 为 **654/1313**，dual-ridge 为 **42/1313**。因此正式科学判决是 `PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204`：九相机确实提供了物理信息余量，也存在更低 exact-call 的 K1 余量，但当前余量依赖稠密 full-DCT 表示，尚未变成低成本部署算法。

`algorithm_breakthrough=false`。

## 为什么这样做

v201 已经排除“继续压低五相机观测残差就能恢复三维梯度”的解释：TGV2 在 1313/1313 个单元上改善观测，却没有救回 24 个失败中的任何一个。v202 随后尝试做五相机 row/null-space 归因，但冻结的 LSQR 在读取真值前就因不收敛而停止，所以只能记为 inconclusive，不能据此解释失败。

与其调旧求解器或扩大模型，v203 直接检验物理上不同的干预：增加四个视角。v204 再做控制归因，防止把“九相机更容易”误写成“学习式 warm start 有效”。

## v203：24 个失败的视角归因

在相同 24 个五相机失败单元上：

| 封存参考 | 严格通过 | field / gradient / observation p90 |
|---|---:|---:|
| 五相机 full-DCT K2 | 0/24 | 0.574347 / 0.972972 / 0.148936 |
| 九相机 full-DCT K2 | 24/24 | 0.365356 / 0.611084 / 0.120974 |

九相机不只是降低平均误差，而是在原封不动的严格门下救回全部 24 个单元。因此 v203 的结论是 `PASS_NINE_CAMERA_PHYSICAL_INFORMATION_HEADROOM_V203`：这些失败至少在当前封存参考中确实受视角信息量制约。

## v204：完整轨迹控制归因

v204 固定读取 v199 已封存的九个 arm、九相机臂、13 套标定和 101 帧，共 1313 个单元；没有新 `A/A^T` 调用。

| arm | 严格单元 | 完整组 | p90: field / gradient / observation | 逻辑在线账 |
|---|---:|---:|---:|---:|
| full-DCT K1 parent | 1313/1313 | 13/13 | 0.318154 / 0.517536 / 0.144493 | 2A+1A^T |
| fixed-identity full-DCT K1 | 1313/1313 | 13/13 | 0.315450 / 0.510925 / 0.143595 | 2A+1A^T |
| full-DCT K2 reference | 1313/1313 | 13/13 | 0.308633 / 0.489549 / 0.103101 | 3A+2A^T |
| initializer-only | 654/1313 | 0/13 | 0.330623 / 0.550027 / 0.225800 | 0A+0A^T |
| dual-ridge K1 | 42/1313 | 0/13 | 0.400175 / 0.496317 / 0.375281 | 2A+2A^T |
| Zero / BP-CGLS1 / Zero-CGLS K2 / affine Jacobi-PCGLS1 | 0/1313 | 0/13 | 各自至少一门整体失败 | 0 至 4 次 exact calls |

这说明固定 identity prior 不是九相机成功的必要解释，因为未正则 full-DCT K1 同样全过门；便宜经典迭代也不能解释成功。更精确的结论是：**九相机 + 稠密 full-DCT 表示 + K1** 有完整开发集余量。

## 独立复算

v203 和 v204 都通过独立验证。v204 的正式与独立九相机指标数组最大绝对差为 **0**，严格掩码和通过 arm 名单完全一致；正式树与父证据树保持不变。

这些验证保证的是封存数组的索引、门值、调用账和判决一致性。它们没有把共享的早期物理生成链变成端到端独立实验，也没有新增 fresh 数据。

## 成功、失败与边界

**成功：** 增加四个视角救回全部 24 个五相机失败；九相机 full-DCT K1 在 1313 个单元上与 K2 一样全过门，并在逻辑账上少 `1A+1A^T`。

**失败：** 当前 K1 依赖稠密 full-DCT 几何缓存和特征路径。没有 whole-pipeline wall、peak RSS 或 fresh-process 证据，所以不能把调用数差直接写成速度或内存收益。五/七/九/十二相机可变基数也尚未由同一紧凑机制统一通过。

**边界：** p14 是历史已暴露开发轨迹。本轮不是 fresh validation、blind test、外部泛化、curved ray 或真实 BOST；也没有训练神经网络。下一步只允许结果前冻结一个最小的稠密缓存移除门，先用 CPU 小模型与便宜对照，不租 GPU。

# v203-v204: nine cameras rescue every five-camera failure, but the headroom still depends on a dense representation

## Bottom line

v203 and v204 do not regenerate 3D fields or tune algorithmic parameters. They audit previously sealed and independently validated p14 five/all-nine-camera results in a fixed order:

1. Do the 24 failures left by the five-camera TGV2 reference persist with nine cameras?
2. If nine-camera K2 passes, does any cheaper sealed control already pass the complete set of 1,313 cells?

The answer is precise. All-nine full-DCT K2 rescues **24/24** five-camera failures. Across all **1,313** cells and **13** calibration groups, unregularized full-DCT K1, fixed-identity full-DCT K1, and full-DCT K2 each reach **1313/1313 and 13/13**. K1 has a logical online ledger of `2A+1AT`, one `A` and one `AT` below K2's `3A+2AT`.

Zero, BP-CGLS1, Zero-CGLS K2, and affine Jacobi-PCGLS1 each reach **0/1313**. Initializer-only reaches **654/1313**, and dual ridge reaches **42/1313**. The scientific decision is therefore `PASS_ALL_NINE_DENSE_REPRESENTATION_CALL_HEADROOM_V204`: nine cameras provide physical-information headroom and K1 provides lower exact-call headroom, but the headroom currently depends on a dense full-DCT representation rather than a low-cost deployable algorithm.

`algorithm_breakthrough=false`.

## Why this was run

v201 rules out the explanation that further reducing five-camera observation residuals recovers the 3D gradient: TGV2 improves observations in all 1,313 cells but rescues none of the 24 failures. v202 then attempts a five-camera row/null-space attribution, but its frozen LSQR contract stops before truth access because it does not converge. That result is inconclusive and cannot explain the failure.

Instead of tuning the old solver or enlarging a model, v203 tests a physically distinct intervention: four additional views. v204 then performs control attribution so that “nine cameras are easier” is not mislabeled as learned warm-start success.

## v203: view attribution on the 24 failures

| Sealed reference | Strict passes | Field / gradient / observation p90 |
|---|---:|---:|
| Five-camera full-DCT K2 | 0/24 | 0.574347 / 0.972972 / 0.148936 |
| All-nine full-DCT K2 | 24/24 | 0.365356 / 0.611084 / 0.120974 |

All nine views do more than improve an average: they rescue every cell under the unchanged strict gate. v203 therefore concludes `PASS_NINE_CAMERA_PHYSICAL_INFORMATION_HEADROOM_V203`. Under this sealed reference, these failures are constrained by physical view information.

## v204: complete-trajectory control attribution

v204 freezes all nine existing v199 arms, the all-nine sensor arm, 13 calibrations, and 101 frames. It evaluates 1,313 cells with no new `A/AT` calls.

| Arm | Strict cells | Complete groups | p90: field / gradient / observation | Logical online ledger |
|---|---:|---:|---:|---:|
| Full-DCT K1 parent | 1313/1313 | 13/13 | 0.318154 / 0.517536 / 0.144493 | 2A+1AT |
| Fixed-identity full-DCT K1 | 1313/1313 | 13/13 | 0.315450 / 0.510925 / 0.143595 | 2A+1AT |
| Full-DCT K2 reference | 1313/1313 | 13/13 | 0.308633 / 0.489549 / 0.103101 | 3A+2AT |
| Initializer-only | 654/1313 | 0/13 | 0.330623 / 0.550027 / 0.225800 | 0A+0AT |
| Dual-ridge K1 | 42/1313 | 0/13 | 0.400175 / 0.496317 / 0.375281 | 2A+2AT |
| Zero / BP-CGLS1 / Zero-CGLS K2 / affine Jacobi-PCGLS1 | 0/1313 | 0/13 | At least one gate fails globally | 0 to 4 exact calls |

The fixed identity prior is not necessary for the all-nine pass because unregularized full-DCT K1 also passes. Cheap classical iterations do not explain the result. The exact supported statement is: **all-nine views plus a dense full-DCT representation plus K1** have complete development-set headroom.

## Independent recomputation

Both v203 and v204 pass independent validation. In v204, the maximum formal-versus-independent metric-array difference is **0**. Strict masks and the passing-arm roster match exactly, and formal and parent evidence trees remain unchanged.

This validates sealed-array indexing, gates, call ledgers, and adjudication. It does not turn the shared upstream physical generation chain into an end-to-end independent experiment, nor does it add fresh data.

## What succeeded, what failed, and the boundary

**Succeeded:** four additional views rescue all 24 five-camera failures. All-nine full-DCT K1 clears all 1,313 cells like K2 with `1A+1AT` fewer logical exact calls.

**Failed:** K1 currently relies on a dense full-DCT geometry cache and feature path. There is no whole-pipeline wall, peak-RSS, or fresh-process evidence, so the call-count difference is not yet a speed or memory result. One compact mechanism has not yet passed variable 5/7/9/12-camera gates.

**Boundary:** p14 is a historically exposed development trajectory. This is not fresh validation, a blind test, external generalization, curved-ray validation, or real BOST, and no neural model is trained. The next gate is one preregistered minimal dense-cache-removal test with cheap CPU controls; GPU rental remains unauthorized.
