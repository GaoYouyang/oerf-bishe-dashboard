# 正则化遗漏分量 / Regularized Missing-Part Target

## 中文

经典逆解遗漏的部分可以由实际相机算子定义，而不依赖五相机专用的QR补空间。原条件核算法和经典逆解保持固定；每折完整排除一条轨迹101帧，仅用其他轨迹1010个源。五、七、九相机各三个冻结时刻，共99个干净查询，非完整序列。

22项独立检查全部一致。强参考四指标匹配99/99；33/33采样绝对和匹配分层通过。场误差相对均值/无先验配对中位改善19.61%/27.60%。五、七、九相机相对无先验的场改善分别为38.27%/28.62%/16.64%。

但是相对均值和无先验便宜控制分别只有93/99四指标不伤害。两组各6个受伤查询并集7个，全在九相机，涉及场与梯度，个别还涉及观测。因此正式状态是**FAIL_REGULARIZED_COMPLEMENT_TARGET**，不改成通过、不挑五/七相机子集、不微调核/正则/深度，也不训练大模型挽救。

标签不是严格零空间；它允许受控的可见部分，已经分别复算源标签、模型、候选物理场、观测、指标和尾部。每查询5A+5A^T之外，完整经典因子、核求解、场库和新几何重建成本均非免费。这一小门只建立已开封固定几何下的统计信息价值与其局限，没有神经、资源提速、新几何、外部泛化或真实BOST成功。既有五相机完整先验正证据保持不变。

## English

Define the classical inverse's missing field information through the actual camera operator rather than a five-camera-specific QR complement. Keep the original conditional kernel and qualified classical inverse fixed. Each fold excludes all101 frames of its held trajectory and uses1010 other-trajectory sources. The99 clean queries cover three frozen anchors at native5/7/9, not complete sequences.

All22 independent checks agree. Strong-reference four-error matching is99/99; sampled absolute and matched strata both pass33/33. Paired median field error improves19.61%/27.60% over mean/zero prior. Improvements over zero prior at5/7/9 are38.27%/28.62%/16.64%.

Joint nonharm to each cheap mean/zero control is only93/99. Their two six-query harm sets have a seven-query union, all at nine cameras, involving field and gradient errors and occasionally observation error. The authoritative decision is**FAIL_REGULARIZED_COMPLEMENT_TARGET**. Do not promote the result, select a5/7 subset, retune the kernel/loading/depth, or rescue it with larger networks.

The target is not exact null. Independent implementations rebuild source labels, models, candidate fields, observations, metrics and tails. Besides5A+5A^T, full classical factors, kernel solves, source banks and new-geometry rebuilding remain nonfree. This is opened-family fixed-geometry statistical evidence and its limitation, not neural, resource-speedup, new-geometry, external or real-BOST success. Preserve the existing complete native5 prior evidence.

Regularized missing-part learning and data proximity have prior art: [RegNets](https://arxiv.org/abs/1812.00965), [Data-proximal null-space networks](https://arxiv.org/abs/2309.06573). This finite-data fixed-regularization statistical screen inherits no convergence theorem, optimality or first/SOTA claim.
