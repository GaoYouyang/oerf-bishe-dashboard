# v182：一步可观测 Jacobi-PCGLS 改善残差，但仍未通过 observation 门

## 做了什么，为什么这样做

v181 已排除固定的几何条件 rank-16 逆因子，但它没有回答一个更直接的问题：能否让更新方向随**当前观测残差**变化，而不是继续用固定谱模态？

v182 因此冻结了一个没有可调超参数的一步机制。它在既定的 `1,009` 维训练场仿射坐标空间中，只读取当前二维观测、报告相机几何、fit-only 仿射中心/基和预先封存的几何 Jacobi 对角量：

`g = M^T r`，`z = D^-1 g`，`q = Mz`，`beta = (g^T z)/(q^T q)`，再令 `x0 = mean + beta U^T z`。

这里的 `beta` 是当前观测残差沿该方向的精确一维最小化，不搜索、不阻尼、不裁剪、不回退，也不读取目标三维真值。之后再运行一轮完全未修改的物理 CGLS K1。这样做是为了直接检验：一个便宜、相机换序等变、随观测变化的对角预条件坐标步，能否把 v181 的宽谱失配压进冻结精度门。

## 正式结果

答案是否定的，但比 v181 更有定位价值。

- 五相机 K0 的 field / gradient / observation p90 为 `0.479636 / 0.679311 / 0.381231`；K1 后变为 `0.457733 / 0.621564 / 0.244595`。
- 九相机 K0 为 `0.459510 / 0.637158 / 0.404757`；K1 后变为 `0.421828 / 0.547715 / 0.266826`。
- 冻结 p90 门是 field `<=0.50`、gradient `<=0.75`、observation `<=0.20`。两档相机的 field 和 gradient 都通过，但 observation 都仍越线。
- 五相机和九相机的 K0、K1 都是严格通过 `0/52`；完整标定为 `0/13`，完整帧为 `0/4`。

因此 v182 不能按“残差变小”包装成成功。一步 Jacobi-PCGLS 的确显著降低 observation p90，但降低幅度不足以达到预先冻结的 matched-accuracy 门。

正式科学判决是 `FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182`。

## 独立复算与一次工程失效

完全独立第二实现重建仿射坐标、Jacobi 对角、一步精确线搜索、物理场、观测、未修改 CGLS K1、全部指标、调用账与相机乱序审计，`47/47` 项检查全真。候选场最大相对差为 `5.07e-12`，逐单元指标最大绝对差为 `8.38e-13`，`beta` 最大绝对差为 `1.36e-12`，所有离散判决完全一致。

第一次独立验证已经完成科学评分，但在写最终 JSON 时遇到 NumPy 布尔值序列化错误，因此按合同保持 `INCONCLUSIVE_ENGINEERING_FAILURE_BEFORE_REPORT_SEAL`，原失败记录和一次性消费凭据均被保留。随后只修正 JSON 标量归一化，并由恢复 release 绑定原失败证据；候选、数据、数组、阈值和正式结果都没有变化。这是工程完整性证据，不是算法增量。

## 成本与证据边界

逻辑在线账为：K0 `1A+1A^T`，K1 `3A+2A^T`；直接 CGLS K4 是 `4A+4A^T`。但因为 matched accuracy 没有成立，不能声称减少 exact calls，也没有启动 wall/RSS 资源门。

v182 只关闭当前**一步、对角预条件、仿射坐标 Jacobi-PCGLS1**机制。不能调 `beta`、阻尼、对角 floor、裁剪或门槛来事后挽救，也不能用 CNN/FNO/UNO/DeepONet 或 GPU 扩大模型救援。它没有关闭完整 C 路线，更不是数学不可能性证明。

下一步只有两个诚实入口：冻结一个物理上真正不同、可证伪的机制，或者等待工况匹配的真实二维双分量 BOS 位移与完整对应关系。

`algorithm_breakthrough=false`、`paper_success=false`、`exact_call_reduction=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

# v182: one observable Jacobi-PCGLS step improves residuals but still misses the observation gate

## What was tested and why

v181 rejects a fixed geometry-conditioned rank-16 inverse factor, but it does not test whether the update can adapt to the **current observation residual**. v182 freezes a single hyperparameter-free coordinate step in the established `1,009`-dimensional affine field space.

The mechanism reads only the current 2D observation, reported camera geometry, fit-only affine center and basis, and a sealed geometry-specific Jacobi diagonal. It computes `g = M^T r`, `z = D^-1 g`, `q = Mz`, `beta = (g^T z)/(q^T q)`, and `x0 = mean + beta U^T z`, then applies one unchanged physical CGLS K1 step. The scalar `beta` is the exact observable line minimizer; there is no search, damping, clipping, fallback, target truth, or trainable parameter.

## Formal result

The answer is negative, although the diagnosis is sharper than v181.

- Five-camera K0 field / gradient / observation p90 values are `0.479636 / 0.679311 / 0.381231`; K1 reduces them to `0.457733 / 0.621564 / 0.244595`.
- All-nine K0 values are `0.459510 / 0.637158 / 0.404757`; K1 reduces them to `0.421828 / 0.547715 / 0.266826`.
- Frozen p90 gates are field `<=0.50`, gradient `<=0.75`, and observation `<=0.20`. Field and gradient pass under both sensor arms, but observation still fails.
- K0 and K1 are each strict-safe on `0/52` cells under both sensor arms, with `0/13` complete calibrations and `0/4` complete frames.

The observation residual materially improves, but improvement alone is not matched accuracy. Decision: `FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182`.

## Independent recomputation and execution boundary

A fully independent implementation rebuilds affine coordinates, the Jacobi diagonal, exact line minimization, fields, observations, unchanged CGLS K1, every metric, call ledger, and camera-permutation audit. All `47/47` checks pass. Maximum candidate-field relative difference is `5.07e-12`, maximum metric absolute difference is `8.38e-13`, maximum `beta` absolute difference is `1.36e-12`, and every discrete decision agrees.

The first independent attempt completed scientific scoring but failed while serializing a NumPy boolean into the final JSON. It therefore remains `INCONCLUSIVE_ENGINEERING_FAILURE_BEFORE_REPORT_SEAL`; the failure and single-use receipt are preserved. A recovery release changed only JSON scalar normalization. Candidate generation, data, arrays, gates, and the formal result remained unchanged. This is engineering assurance, not an algorithmic result.

## Cost and claim boundary

The logical ledgers are `1A+1A^T` for K0 and `3A+2A^T` for K1, versus `4A+4A^T` for direct CGLS K4. Because matched accuracy fails, v182 establishes no exact-call reduction and does not authorize wall/RSS testing.

This closes only the frozen one-step, diagonally preconditioned affine-coordinate Jacobi-PCGLS1 mechanism. Do not retune `beta`, damping, diagonal floors, clipping, or gates, and do not rescue it with CNN/FNO/UNO/DeepONet or GPU scale. It does not close the C route and is not an impossibility result.

The next honest entry point is either one preregistered physically distinct mechanism or new condition-matched paired two-component BOS evidence.

`algorithm_breakthrough=false`, `paper_success=false`, `exact_call_reduction=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
