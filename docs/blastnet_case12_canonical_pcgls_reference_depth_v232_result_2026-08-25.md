# v232/v232.1：相机顺序已规范，但浮点级 Jacobi 差异仍使 K17 reference 无法独立释放

## 结论

v231 已经完成同一 Case 12 的 `598 x 64` 个 PCGLS 检查点，但相机块换序会使深层轨迹发生约 `1e-2` 的场差，因此没有裁决 reference 深度。v232 只做一个结果前冻结的数值表示修复：按相机 ID 规范排序完整 `16x16x2` 观测块，再组装算子、Jacobi 和未修改 PCGLS；K1-K64、598 个单元、精度门与 `1e-8` 数值门全部不变。

规范排序是有效的。正式实现和 v232.1 独立实现各自把相机换序后的观测、Jacobi、场、残差与指标差都降到 `0`。两套实现也都独立得到一个未释放的 provisional 结果：K16 仍只有 `594/598` 个严格单元与 `11/13` 个完整 rig，K17 则达到 `598/598` 与 `13/13`。

但跨实现数值合同仍未通过。v232.1 在完全相同的封存观测上重算全部 598 个单元和 K1-K64 后，正式与独立的场相对差最大为 `1.17927e-2`，指标绝对差最大为 `8.61528e-3`，汇总差最大为 `4.99600e-3`。第一次越过冻结场差门发生在 K17、rig 0、frame 43：

`1.67429e-8 > 1e-8`

K16 的对应最大场差仍为 `3.27799e-9`，第一次指标越门在 K19。这意味着不能因为两边都给出 K17 和同样的离散通过数，就忽略连续物理场已经不满足结果前冻结的独立复算门。

封存后根因诊断进一步定位到 Jacobi 对角量的浮点归约。正式实现使用 `sum(rows * rows)`，独立实现使用数学等价的 `einsum`。13 个 rig 上两份 inverse diagonal 的最大相对差只有 `2.24977e-16`，但字节哈希全部不同。对首个失败单元，只要两套 PCGLS 使用完全相同的 Jacobi，K1、K16、K17、K18、K19、K26、K32 和 K64 都与正式场逐值一致；仅替换为独立归约的 Jacobi，就会把 K17 场差放大到 `1.67429e-8`。

最终科学判决是：

`INCONCLUSIVE_INVALID_CASE12_CANONICAL_PCGLS_REFERENCE_DEPTH_V232_1`

发布判决中的 `selected_depth=null`。K17 只是两套失效执行各自得到的 provisional 深度，不能作为合格 reference。这个结果关闭当前 canonical deep-PCGLS reference-depth 壳，不证明 K64 内没有合格 reference，也不裁决 dual-PRESS 的成功或失败。

下一步不能继续对同一壳做第三次修复、放宽 `1e-8`、替换深度或挑一个归约方式包装成功。若继续，必须另行结果前冻结一个数值稳定、保存全部 reference 状态并带独立解证书的绝对 reference；在它通过以前，不解释策略接受、安全、exact-call、wall/RSS、训练或 GPU。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v232/v232.1: canonical camera order works, but roundoff-scale Jacobi differences still prevent an independent K17 reference release

## Conclusion

v231 completed all `598 x 64` PCGLS checkpoints on the same opened Case 12 condition, but camera-block permutation changed deep trajectories by about `1e-2` in field space, so no reference depth was adjudicated. v232 makes one preregistered numerical-representation change: canonically sort complete `16x16x2` observation blocks by camera ID before assembling the operator, Jacobi map, and unchanged PCGLS. K1-K64, all 598 cells, every accuracy gate, and the `1e-8` numerical threshold remain unchanged.

Canonical ordering works. Within both the formal implementation and the v232.1 independent implementation, camera permutation changes canonical observations, Jacobi states, fields, residuals, and metrics by exactly `0`. Both implementations also derive the same unreleased provisional result: K16 still reaches only `594/598` strict cells and `11/13` complete rigs, whereas K17 reaches `598/598` and `13/13`.

The cross-implementation numerical contract still fails. After v232.1 recomputes all 598 cells and K1-K64 from the exact same sealed observations, maximum formal-independent field-relative, metric-absolute, and summary differences are `1.17927e-2`, `8.61528e-3`, and `4.99600e-3`. The first frozen field-tolerance violation occurs at K17, rig 0, frame 43: `1.67429e-8 > 1e-8`. K16 remains below that field threshold at `3.27799e-9`, and the first metric violation appears at K19.

A post-seal diagnostic localizes the remaining instability to floating-point reduction in the Jacobi diagonal. The formal path uses `sum(rows * rows)`, while the independent path uses the mathematically equivalent `einsum`. Across 13 rigs, the maximum relative difference between inverse diagonals is only `2.24977e-16`, yet every byte hash differs. For the first failing cell, giving both PCGLS implementations the exact same Jacobi produces value-identical formal fields at K1, K16, K17, K18, K19, K26, K32, and K64. Substituting only the independently reduced Jacobi amplifies K17 field error to `1.67429e-8`.

The final scientific decision is `INCONCLUSIVE_INVALID_CASE12_CANONICAL_PCGLS_REFERENCE_DEPTH_V232_1`, with released `selected_depth=null`. K17 is only a provisional depth derived separately by two invalid executions and cannot serve as an adequate reference. This closes the current canonical deep-PCGLS reference-depth shell. It neither proves that no adequate reference exists through K64 nor adjudicates dual-PRESS.

No third repair of the same shell, tolerance relaxation, depth substitution, or favorable reduction choice is allowed. Any continuation must separately preregister a numerically stable absolute reference that seals its full reference state and carries an independent solution certificate. Policy acceptance, safety, exact calls, wall/RSS, training, and GPU use remain uninterpretable until that gate passes.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
