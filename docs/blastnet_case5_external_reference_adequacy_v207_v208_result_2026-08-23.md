# v207-v208：BLASTNet Case 5 外部门与 K16 参考充分性判决

## 讲人话结论

这次没有得到“算法在新工况上失败”或“算法泛化成功”的结论。v207 把此前未打开的 BLASTNet Case 5 作为一次性公开外部门，但候选、稠密 K1 和冻结的稠密 K2 reference 都是 `0/546` 个单元通过；标准答案本身不合格，所以这场比较不能判输赢。

v208 随后在已经打开的同一工况上做了一次更强但仍固定的参考诊断：从零开始分别运行 CGLS K4、K8 和唯一主判据 K16。K16 把观测误差明显压低，13 个标定组的 observation p90 已到 `0.0473-0.0567`，gradient p90 也在 `0.5627-0.6795`；但 field p90 仍为 `0.7252-0.7608`，全部高于冻结的 `0.50` 门。因此 K16 仍是 `0/546` 个严格单元、`0/13` 个完整标定组通过。

精确判决为 `INCONCLUSIVE_CASE5_REFERENCE_REMAINS_INADEQUATE_AT_ZERO_CGLS_K16_V208`。这关闭的是当前 Case 5 的 straight-ray、`32x16x16` 网格和零起点 CGLS K16 参考族，不是整条 C 路线，也不是数学不可能证明。

## 做了什么，为什么这样做

v206 在历史已暴露的 PoolFire p14 九相机上建立了真实的 fresh wall/RSS headroom，但还没有外部泛化证据。v207 因此冻结一个此前未打开的公开反应流工况，并保持相机数、三维网格、straight-ray 正反算子、候选实现和准确率门不变，想检验同一九相机结论能否迁移。

v207 的正式与独立程序都发现，冻结 K2 reference 自身无法达到绝对精度门。按结果前的 fail-closed 规则，这不能算候选失败，资源门也不得启动。v208 的唯一作用就是排查“只是 K2 迭代太少”这个解释：K4 和 K8 只作收敛诊断，K16 是唯一允许决定 reference 是否充分的 primary；三条轨迹分别保留 `4A+4A^T`、`8A+8A^T`、`16A+16A^T` 的真实逻辑账。

## 严格数值结果

| Reference | 严格单元 | 完整标定组 | Field p90 范围 | Gradient p90 范围 | Observation p90 范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-CGLS K4 | `0/546` | `0/13` | `0.8155-0.8422` | `0.5951-0.6746` | `0.2124-0.2456` |
| Zero-CGLS K8 | `0/546` | `0/13` | `0.7644-0.7967` | `0.5381-0.6764` | `0.1065-0.1258` |
| Zero-CGLS K16 primary | `0/546` | `0/13` | `0.7252-0.7608` | `0.5627-0.6795` | `0.0473-0.0567` |
| 冻结 p90 门 | - | - | `<=0.50` | `<=0.75` | `<=0.20` |

K16 的 field worst 范围为 `0.7265-0.7635`，其中部分标定组也越过 `0.75` worst 门。更关键的是，所有单元的 field 指标都高于 strict `0.50` 门，所以不是一两个尾部单元造成的失败。观测拟合继续改善，而三维场误差停在高位，更符合当前逆问题/先验不足，而不是“再跑几步就自然合格”。

## 独立复算

独立第二实现没有重读正式判决来选结果，而是重新构建三维真值预处理、13 套几何、546 份观测以及 K4/K8/K16 的零起点 CGLS 轨迹，再重算逐单元指标、完整标定组尾部和调用账。

所有独立检查通过。独立观测最大相对差为 `1.26e-15`，K16 场最大相对差为 `3.60e-14`，残差最大相对差为 `2.11e-12`，指标与汇总最大绝对差分别为 `3.33e-16` 和 `2.22e-16`。正式与独立判决一致。

## 是否成功、是否突破

- **执行成功：** formal 与独立复算均完成，科学判决可信。
- **参考门失败：** Zero-CGLS K16 仍未提供合格三维场 reference。
- **外部门不确定：** v207 不能被包装成 warm start 的外部失败或成功。
- **没有资源结论：** wall/RSS 阶段按协议没有启动。
- **没有算法突破：** `algorithm_breakthrough=false`。
- **没有真实 BOST：** 当前仍是公开 CFD 三维场生成的 straight-ray 代理，`real_bost=false`。

## 路线动作

停止在同一 Case 5 straight-ray/网格/零起点 CGLS 参考族上追加迭代、调门或训练更大模型，也不租 GPU。下一次重新建立外部门，必须依赖新的物理信息，或在看结果前冻结一个物理上不同、能够先证明绝对场精度充分的 reference。v206 的 p14 九相机 post-open 资源正结果仍保留，但不能外推到 Case 5。

---

# v207-v208: BLASTNet Case 5 External Gate and K16 Reference-Adequacy Verdict

## Plain-language conclusion

This run does not establish either external failure or external success of the algorithm. v207 opened BLASTNet Case 5 once as a preregistered public external condition, but the candidate, dense K1, and the frozen dense K2 reference all achieved `0/546` strict-safe cells. Because the reference itself was inadequate, the comparison could not adjudicate the method.

v208 then performed a stronger but fixed post-open reference diagnostic on the same condition. It ran zero-start CGLS K4, K8, and the unique primary K16. K16 substantially reduced observation error: observation p90 is `0.0473-0.0567` across the 13 calibration groups, and gradient p90 is `0.5627-0.6795`. Field p90, however, remains `0.7252-0.7608`, above the frozen `0.50` limit in every group. K16 therefore still reaches only `0/546` strict-safe cells and `0/13` complete calibration groups.

The sealed verdict is `INCONCLUSIVE_CASE5_REFERENCE_REMAINS_INADEQUATE_AT_ZERO_CGLS_K16_V208`. This closes the current Case 5 gate under the straight-ray, `32x16x16` grid, and zero-start CGLS K16 reference family. It does not close the full C route and is not a mathematical impossibility proof.

## What was done and why

v206 established fresh wall/RSS headroom for all nine cameras on historically exposed PoolFire p14, but it did not establish external generalization. v207 therefore froze a previously unopened public reacting-flow condition while retaining the camera count, 3D grid, straight-ray forward/adjoint, candidate implementation, and accuracy gates.

Both v207 implementations found that the frozen K2 reference itself missed the absolute-accuracy gates. Under the preregistered fail-closed rule, this was not a candidate failure and did not authorize resource measurement. v208 tests the specific explanation that K2 was simply too shallow. K4 and K8 are convergence diagnostics only; K16 is the sole primary for reference adequacy. Their logical ledgers remain `4A+4AT`, `8A+8AT`, and `16A+16AT` respectively.

## Strict numerical result

| Reference | Strict cells | Complete calibration groups | Field p90 range | Gradient p90 range | Observation p90 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-CGLS K4 | `0/546` | `0/13` | `0.8155-0.8422` | `0.5951-0.6746` | `0.2124-0.2456` |
| Zero-CGLS K8 | `0/546` | `0/13` | `0.7644-0.7967` | `0.5381-0.6764` | `0.1065-0.1258` |
| Zero-CGLS K16 primary | `0/546` | `0/13` | `0.7252-0.7608` | `0.5627-0.6795` | `0.0473-0.0567` |
| Frozen p90 limit | - | - | `<=0.50` | `<=0.75` | `<=0.20` |

K16 field worst values range from `0.7265` to `0.7635`, and some calibration groups also exceed the `0.75` worst limit. More importantly, every cell violates the strict `0.50` field limit. This is not a small tail failure. Observation fit improves while 3D-field error remains high, which points to inadequate inversion/reference structure rather than a few more iterations naturally solving the gate.

## Independent recomputation

The independent implementation rebuilds 3D truth preprocessing, all 13 geometries, 546 observations, and every zero-start K4/K8/K16 CGLS trajectory before recomputing cell metrics, complete-group tails, and call ledgers.

Every independent check passes. Maximum relative differences are `1.26e-15` for observations, `3.60e-14` for K16 fields, and `2.11e-12` for residuals. Maximum absolute metric and summary differences are `3.33e-16` and `2.22e-16`. Formal and independent decisions agree.

## Success and breakthrough boundary

- **Execution succeeded:** formal and independent recomputation completed.
- **The reference gate failed:** zero-start CGLS K16 still lacks adequate 3D-field accuracy.
- **The external gate is inconclusive:** v207 cannot be repackaged as either failure or success of the warm start.
- **No resource result:** wall/RSS was not run.
- **No algorithmic breakthrough:** `algorithm_breakthrough=false`.
- **No real BOST result:** this remains a straight-ray proxy generated from public CFD 3D fields, with `real_bost=false`.

## Route action

Stop adding iterations, retuning thresholds, or training larger models within the same Case 5 straight-ray, grid, and zero-start CGLS reference family. Do not rent a GPU. Reopen an external gate only with new physical information or a physically distinct reference whose absolute field accuracy is preregistered and established first. The v206 post-open all-nine p14 resource result remains valid but cannot be extrapolated to Case 5.
