# v274.2：数值修复仍未通过冻结复现门，固定多几何 reference 关闭

## 做了什么

v274.2 只修复 v274.1 已定位的数值复现问题，没有改变科学数据、13 个 leave-one-rig-out 目标、每目标 12 套训练几何、33 帧、零起点无预条件 CGLS K16、控制组、精度门或调用账。唯一变化是先把带标签的 rig block 规范到固定顺序，再用补偿求和累加各 block 的伴随贡献。结果前规则明确规定：原容差不放宽，若仍有任何复现门失败，就以 `INCONCLUSIVE` 关闭这条固定 reference，不再做第三次数值修补。

## 独立复算与判决

正式执行完成，15 项有效性检查通过 14 项；完全独立的第二实现完成，32 项检查通过 27 项。修复确实把带标签 rig 乱序差压到严格的 `0`，物理 replay 差也只有 `3.33e-16`。但相机乱序差仍为 `1.96e-9`，高于冻结的 `1e-10`；独立乱序差为 `1.52e-9`。正式与独立之间的场、目标投影和目标残差差分别为 `1.32e-9`、`5.14e-9` 和 `9.93e-8`，均高于冻结的 `1e-9` 一致性界。

因此权威判决为 `INCONCLUSIVE_CASE19_LORO_FUSION_REFERENCE_V274_2`。虽然两条实现各自都给出 `429/429` 个单元和 `13/13` 套完整 rig 的诊断命中，但这些数字发生在复现合同失效的批次中，不能解释为科学通过，也不能授权预测器、训练、资源门或新外门。

## 证据边界

固定 Case 19 leave-one-rig-out 零起点 CGLS K16 多几何 reference 现已关闭。不再放宽容差、不再修改求和、不调迭代深度、不加权重或正则，也不换另一个 BLASTNet 工况补考。这个结果关闭的是当前固定 reference，不关闭整条 C 路线，也不证明多几何融合在数学上不可能。后续只有工况配对的真实二维双分量 BOST 数据，或一个结果前冻结且物理上真正不同的新机制，才值得重新开启科学门。`algorithm_breakthrough=false`。

# v274.2: The numerical repair still misses frozen reproducibility limits, closing the fixed multi-geometry reference

## What was tested

v274.2 repairs only the numerical reproducibility issue isolated in v274.1. It does not change the scientific data, thirteen leave-one-rig-out targets, twelve training geometries per target, thirty-three frames, zero-start unpreconditioned CGLS K16 solver, controls, accuracy gates, or call ledger. The sole changes canonicalize labeled rig blocks before assembly and use compensated summation for block-adjoint contributions. The preregistered rule keeps every tolerance unchanged and closes this fixed reference as `INCONCLUSIVE` if any reproducibility gate still fails, with no third numerical repair.

## Independent recomputation and verdict

Formal execution completes with 14 of 15 validity checks, while the fully independent implementation completes with 27 of 32 checks. The repair makes labeled-rig permutation difference exactly zero, and physical replay differs by only `3.33e-16`. Camera permutation nevertheless remains at `1.96e-9`, above the frozen `1e-10` limit, while independent permutation reaches `1.52e-9`. Formal-independent field, target-projection, and target-residual differences are `1.32e-9`, `5.14e-9`, and `9.93e-8`, all above the frozen `1e-9` agreement limit.

The authoritative verdict is therefore `INCONCLUSIVE_CASE19_LORO_FUSION_REFERENCE_V274_2`. Both implementations individually report diagnostic counts of `429/429` cells and `13/13` complete rigs, but those counts come from an execution that fails the reproducibility contract. They are not a scientific pass and cannot authorize a predictor, training, resource testing, or another external gate.

## Evidence boundary

The fixed Case 19 leave-one-rig-out zero-start CGLS K16 multi-geometry reference is closed. Tolerances, summation, depth, weights, and regularization will not be changed, and another BLASTNet condition will not be used as a rescue attempt. This closes the fixed reference, not the wider C route, and does not prove multi-geometry fusion mathematically impossible. A new scientific gate now requires either condition-matched real two-component BOST data or a preregistered mechanism that is physically distinct from this reference. `algorithm_breakthrough=false`.
