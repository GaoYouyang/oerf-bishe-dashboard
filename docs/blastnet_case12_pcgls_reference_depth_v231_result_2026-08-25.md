# v231：K1-K64 全部算完，但数值不变量先失效，最小合格深度仍未裁决

## 结论

v230.1 已经确认 Case 12 的冻结 K16 reference 只通过 `594/598` 个严格单元与 `11/13` 个完整 rig，因此 dual-PRESS 策略不能继续裁决。v231 不修改策略，也不放宽精度门；它在同一已开封 Case 12、13 个 rig、每 rig 46 帧上，对未修改的零起点 geometry-Jacobi PCGLS 一次运行到 K64，并保存 K1-K64 的全部检查点。唯一问题是：是否存在一个对全部 `598` 个单元和 `13` 个 rig 都合格的最小全局深度。

正式程序与完全独立第二实现都完成了 `598 x 64` 个深度检查点，K16 父证据连续性逐值一致：场、残差、指标与调用账最大差均为 `0`，观测最大绝对差为 `8.88e-16`。独立实现的物理残差方程闭合到 `8.35e-14`。这些结果说明输入、K16 父结果和基本物理重放没有漂移。

但是，结果前冻结的数值不变量没有通过。合同要求相机块换序后，完整深层 PCGLS 轨迹的场、残差和指标差异都不超过 `1e-8`。实际最大差如下：

| 检查 | 正式实现 | 独立实现 | 冻结门 |
|---|---:|---:|---:|
| 相机换序场相对差 | `1.08496e-2` | `8.71070e-3` | `1e-8` |
| 相机换序残差相对差 | `3.44567e-1` | `3.51017e-1` | `1e-8` |
| 相机换序指标绝对差 | `8.40165e-3` | `7.00570e-3` | `1e-8` |

两套实现之间的完整轨迹也出现同量级差异：场相对差最大 `1.05405e-2`、指标绝对差最大 `8.37684e-3`、汇总差最大 `4.07330e-3`。相机块顺序理论上不应改变物理问题，因此不能在这个数值合同失效后继续读取 K1-K64 科学数组并挑选一个方便的深度。

正式科学判决为：

`INCONCLUSIVE_INVALID_CASE12_PCGLS_REFERENCE_DEPTH_V231`

这不是“已经证明 K64 以内没有合格 reference”，也不是 dual-PRESS 的算法失败。它只说明当前深层 PCGLS 实现对相机块浮点求和顺序敏感，导致预注册的 reference-depth qualification 无法释放科学结论。`selected_depth=null`，dual-PRESS 的接受、安全、exact-call、wall/RSS 和资源收益继续不裁决。

下一步只能另行结果前冻结一个结果盲的相机 ID 规范排序，在算子组装和 PCGLS 之前把相机块顺序固定；K1-K64、598 个单元、全部精度门、Jacobi 预条件、独立第二实现和 `1e-8` 数值门保持不变。该修复只是让 reference-depth 问题可裁决，不是算法成果。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v231: K1-K64 completes, but the numerical invariance contract fails before a reference depth can be adjudicated

## Conclusion

v230.1 established that the frozen Case 12 K16 reference passes only `594/598` strict cells and `11/13` complete rigs, so the dual-PRESS policy cannot be adjudicated. v231 neither changes the policy nor relaxes an accuracy gate. On the same opened Case 12 roster of 13 rigs and 46 frames per rig, it runs the unchanged zero-start geometry-Jacobi PCGLS once through K64 and saves every K1-K64 checkpoint. The sole question is whether one minimum global depth is adequate for all `598` cells and all `13` rigs.

Both the formal program and a fully independent second implementation complete all `598 x 64` checkpoints. K16 parent continuity is exact for fields, residuals, metrics, and call ledgers, while the maximum observation difference is `8.88e-16`. The independent physical residual-equation closure is `8.35e-14`. Inputs, the K16 parent evidence, and basic physical replay therefore remain continuous.

The preregistered numerical invariance contract nevertheless fails. Camera-block reordering was required to change full deep-PCGLS fields, residuals, and metrics by at most `1e-8`. Formal field, residual, and metric discrepancies reach `1.08496e-2`, `3.44567e-1`, and `8.40165e-3`; the independent implementation reaches `8.71070e-3`, `3.51017e-1`, and `7.00570e-3`. Across the two complete implementations, the maximum field-relative, metric-absolute, and summary differences are `1.05405e-2`, `8.37684e-3`, and `4.07330e-3`.

Camera-block order should not alter the physical problem. Once this numerical contract fails, the K1-K64 science arrays cannot be opened to select a convenient depth. The scientific decision is therefore `INCONCLUSIVE_INVALID_CASE12_PCGLS_REFERENCE_DEPTH_V231`, with `selected_depth=null`.

This does not show that no adequate reference exists through K64, and it is not an algorithm failure of dual-PRESS. It shows only that the current deep-PCGLS implementation is sensitive to floating-point camera-block summation order, so the preregistered reference-depth qualification cannot release a scientific verdict. Policy acceptance, safety, exact calls, wall/RSS, and resource benefit remain unadjudicated.

The only next step is a separately preregistered, result-blind canonical camera-ID ordering before operator assembly and PCGLS. K1-K64, all 598 cells, every accuracy gate, the Jacobi preconditioner, the independent second implementation, and the `1e-8` numerical threshold remain unchanged. This numerical representation repair would make the reference-depth question adjudicable; it would not itself be an algorithmic result.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
