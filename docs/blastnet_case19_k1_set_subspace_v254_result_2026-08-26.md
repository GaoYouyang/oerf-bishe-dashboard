# v254：无序 K1 配对方向子空间未通过完整序列 matched-accuracy

## 为什么做

v253 已经关闭了“先选一个锚点再传播”的路线。v254 改问一个物理上不同的问题：不选锚、不读时间顺序，能否把同一套 rig 的 33 帧部署可见观测共同压缩成一个稳定的 solver-native 空间，再用于每帧暖启动？

结果前固定的方法是：每帧从零起点运行一次 geometry-Jacobi PCGLS K1，保留成对的三维场方向与其精确二维投影，并按投影范数归一化；对 33 个无序方向构造唯一 rank-16 子空间。每帧只用当前二维观测投影到这个子空间得到初始化，再运行未修改的 PCGLS K14。表示生成不读取真值、时间、frame index、rig 标签或失败标签，并对帧集合与相机换序等变。

## 独立复算通过，科学判决为负

独立第二实现不用 formal 的 SVD 路径，改用 Gram 特征分解重建子空间、初始化、物理 replay、四指标和全部门。`29/29` 项检查全部通过。formal 与 independent 的场最大相对差为 `7.79e-10`，初始化最大相对差为 `7.62e-13`，逐单元指标最大绝对差为 `3.88e-11`，汇总最大差为 `1.64e-11`。帧集合换序后的场最大相对差为 `3.02e-13`，成对 forward 最大差为 `3.25e-15`。

但主候选只达到 `383/429` 个绝对安全单元和 `6/13` 条完整 rig。相对稳健 K16 envelope 的 matched-accuracy 更明确地失败，为 `0/429` 单元、`0/13` 完整 rig。全局 field / gradient / interior-gradient / observation 的 p90-higher 分别为 `0.32567 / 0.61605 / 0.75098 / 0.06059`；看似接近的总体尾部不能替代逐单元、逐 rig matched 门。

同价的 self-K1 restart control 为 `371/429、4/13`，Zero-PCGLS K15 control 为 `383/429、4/13`，两者的 matched 完整 rig 同样是 `0/13`。已封存的按时间 FIFO16-K14 诊断在 matched 门为 `429/429、13/13`，但绝对门仍只有 `428/429、12/13`，且它不是无序帧集合等变方法。稳健 K16 reference 自己的绝对门也只有 `417/429、9/13`，所以这里不存在可以包装成成功的完整精度闭环。

## 成本与路线动作

主候选完整序列的逻辑账为 `495A+495A^T`，K16 reference 为 `528A+528A^T`，名义合计差为 `6.25%`。由于主候选没有通过绝对门或 matched-accuracy，这只是算术预算，不是有效减调用、wall time、RSS 或速度结果；资源门不授权。

正式判决是 `FAIL_CASE19_K1_SET_SUBSPACE_MATCHED_ACCURACY_V254`。当前“全 rig 共享、无序、归一化 K1 配对方向的 rank-16 子空间”关闭，不通过增加 rank、改变深度或换归一化挽救，也不训练更大模型或租 GPU。这不关闭整条 C 路线；后续只能来自新的配对真实 BOST 信息，或一个不再属于全局 K1 子空间池化、物理上真正不同且结果前唯一冻结的 geometry-local、solver-native 或 nonlinear 机制。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v254: the unordered paired-K1 subspace fails whole-sequence matched accuracy

## Motivation

v253 closes the strategy of selecting one anchor before propagation. v254 asks a physically different question: without an anchor or temporal ordering, can all 33 deployment-visible frames in a rig define a stable solver-native space that improves every frame's initialization?

The preregistered method runs one zero-start geometry-Jacobi PCGLS K1 step per frame, retains each paired 3D field direction and exact 2D projection, and normalizes both by the projection norm. One rank-16 subspace is built from the unordered set of 33 directions. Each frame uses only its current 2D observation to form a subspace initializer before unchanged PCGLS K14. Representation construction reads no truth, time, frame index, rig label, or failure label and is equivariant to frame-set and camera permutations.

## Independent recomputation passes; the scientific result is negative

The independent implementation replaces formal SVD with a Gram eigendecomposition and independently rebuilds the subspace, initializers, physical replay, four metrics, and every gate. All `29/29` checks pass. Maximum formal-independent field and initializer relative differences are `7.79e-10` and `7.62e-13`; maximum cell-metric absolute difference is `3.88e-11` and maximum summary difference is `1.64e-11`. Maximum frame-permutation field disagreement is `3.02e-13`, and maximum paired-forward disagreement is `3.25e-15`.

The primary nevertheless reaches only `383/429` absolute-safe cells and `6/13` complete rigs. Against the robust K16 envelope, matched accuracy fails more decisively at `0/429` cells and `0/13` complete rigs. Global field / gradient / interior-gradient / observation p90-higher values are `0.32567 / 0.61605 / 0.75098 / 0.06059`; close aggregate tails do not replace the cellwise and complete-rig matched gates.

The equal-cost self-K1 restart control reaches `371/429 and 4/13`, while zero-PCGLS K15 reaches `383/429 and 4/13`; both remain at `0/13` matched complete rigs. The sealed chronological FIFO16-K14 diagnostic reaches `429/429 and 13/13` on the matched gate but only `428/429 and 12/13` on absolute accuracy, and it is not an unordered frame-set-equivariant method. The robust K16 reference itself reaches only `417/429 and 9/13` on absolute accuracy. There is no complete accuracy closure that can be packaged as success.

## Cost and route action

The primary whole-sequence ledger is `495A+495A^T`, versus `528A+528A^T` for K16, a nominal combined difference of `6.25%`. Because the primary fails both absolute and matched accuracy, this remains an arithmetic budget rather than effective call reduction, wall time, RSS, or speed evidence. The resource gate is not authorized.

The formal decision is `FAIL_CASE19_K1_SET_SUBSPACE_MATCHED_ACCURACY_V254`. The current rig-global unordered rank-16 pool of normalized paired K1 directions closes, with no rank increase, depth change, alternate normalization, larger model, or GPU rescue. This does not close the C route. Any next step requires new paired real-BOST information or a uniquely preregistered, physically distinct geometry-local, solver-native, or nonlinear mechanism that is not another global K1-subspace pool.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.
