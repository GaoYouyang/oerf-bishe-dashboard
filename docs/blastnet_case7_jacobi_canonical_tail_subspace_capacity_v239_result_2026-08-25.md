# v239/v239.2：Jacobi 对称坐标仍不能让 Case 7 的 rank-64 尾差跨 rig 迁移

## 结论

v236 已经说明：在物理场坐标中，用其他 12 条 rig 学到的全局 rank-64 尾差空间不能覆盖第 13 条 rig。v239 只检验一个物理上不同的解释：不同 rig 使用不同的几何 Jacobi 预条件器，也许同一尾差在物理坐标中不共享，但在 PCGLS 的对称变量中可以共享。

每个 fold 仍完整留出一条 rig。其他 `12x42=504` 个 `8192` 维尾差先按各自冻结的几何 Jacobi 逆对角变换到对称坐标，构成 `504x8192` 训练矩阵；只保留未中心化 rank 64 子空间，再用留出 rig 的几何映射回物理场。留出真值只用于最优投影，因此它仍是不可部署的容量上界，不是 observation-only 预测器。

结果为 **`0/13` 完整 rig**。Jacobi 坐标的全局 p50/p90/worst 为 **`0.644473 / 0.734855 / 0.813573`**；同维数的物理场全局 rank 64 对照为 **`0.645458 / 0.731692 / 0.805609`**。中位数只改善 `0.000985`，但 p90 恶化 `0.003164`、worst 恶化 `0.007964`。全帧 p90 在 6 条 rig 改善、7 条恶化；更关键的后期帧 p90 在 **13/13** 条 rig 上全部恶化。

| 尾差空间 | 维数 | 完整 rig | 全局 p50 | 全局 p90 | 全局 worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 Low64 控制 | `64` | `0/13` | `0.953580` | `0.974420` | `0.990718` |
| v236 物理场全局 rank 64 | `64` | `0/13` | `0.645458` | `0.731692` | `0.805609` |
| v239 Jacobi 对称坐标 rank 64 | `64` | `0/13` | `0.644473` | `0.734855` | `0.813573` |

冻结门仍要求每条 rig 的全帧与后期帧同时满足 `p90 <= 0.316228`、`worst <= 0.5`。因此这不是门限附近的成功：Jacobi 全局 p90 仍约为门的 `2.32` 倍，而且每条 rig 都失败。

## 它改变了什么判断

这个结果排除了一个具体解释：**当前跨 rig 失配并不只是因为各 rig 的 Jacobi 几何尺度不同。** 把尾差放进 PCGLS 的对称几何坐标后，跨 rig rank-64 容量没有恢复，后期尾部反而一致变差。

因此，当前 symmetric geometry-Jacobi rank-64 表示关闭；不增加 rank、不换 Jacobi 缩放方向，也不使用 CNN、FNO 或 GPU 去挽救已经失败的 truth-aware 容量门。这个负结果不关闭整条 C 路线，也不证明数学不可能；它只说明下一候选必须引入真正不同的、非固定低秩坐标机制。

## 独立验证与两次前置失败

正式实现使用样本 Gram 特征分解并将子空间映回物理场；独立程序不导入正式实现，对全部 13 个 `504x8192` 训练矩阵直接做 economy SVD，再独立重建 Jacobi 映射和物理子空间。

前两次 validator 都在任何独立 SVD、残差或科学评分之前 fail-closed：第一次错误地要求数学等价的 Jacobi 归约具有逐字节相同 hash；第二次又对标量 floor 作逐位相等要求。两份失败记录原样保留。v239.2 只把这项验证改为结果前已经冻结的数值容差，正式输出、候选、fold、rank 和门均未重跑或修改。

最终独立复算 `18/18` 项检查全真。Jacobi 逆对角最大相对差为 `9.22e-17`，残差与汇总最大差为 `1.12e-14 / 1.02e-14`，相机乱序最大相对差为 `4.54e-16`；最小训练数值秩为 `504`。最终科学判决为 `FAIL_CASE7_JACOBI_CANONICAL_TAIL_SUBSPACE_CAPACITY_V239`。

它是已开封 Case 7 上的 truth-aware 表示容量负结果，不是部署算法、调用减少、wall/RSS、外部泛化、curved ray 或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v239/v239.2: Jacobi symmetric coordinates still do not make the Case 7 rank-64 tail transferable across rigs

## Conclusion

v236 shows that a global rank-64 tail space learned from twelve rigs in physical-field coordinates does not cover the thirteenth rig. v239 tests one physically different explanation: rigs use different geometry-Jacobi preconditioners, so perhaps the same tail is not shared in physical coordinates but becomes shared in the symmetric PCGLS variable.

Each fold still holds out one complete rig. The other `12x42=504` tails, each of dimension `8192`, are transformed by their frozen geometry-Jacobi inverse diagonals into symmetric coordinates, forming a `504x8192` training matrix. An uncentered rank-64 subspace is retained and mapped back to the physical field through the held-out rig geometry. Held-out truth supplies only the optimal projection, so this remains a nondeployable capacity upper bound rather than an observation-only predictor.

The result is **`0/13` complete rigs**. Jacobi-coordinate global p50/p90/worst is **`0.644473 / 0.734855 / 0.813573`**, versus **`0.645458 / 0.731692 / 0.805609`** for the equal-dimensional physical global rank-64 control. The median improves by only `0.000985`, while p90 worsens by `0.003164` and worst worsens by `0.007964`. All-frame p90 improves on six rigs and worsens on seven; more importantly, late-frame p90 worsens on **13/13** rigs.

| Tail space | Dimension | Complete rigs | Global p50 | Global p90 | Global worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Low64 control | `64` | `0/13` | `0.953580` | `0.974420` | `0.990718` |
| v236 physical global rank 64 | `64` | `0/13` | `0.645458` | `0.731692` | `0.805609` |
| v239 Jacobi-symmetric rank 64 | `64` | `0/13` | `0.644473` | `0.734855` | `0.813573` |

The frozen gate still requires both all-frame and late-frame `p90 <= 0.316228` and `worst <= 0.5` for every rig. This is therefore not a near-threshold success: the Jacobi global p90 remains about `2.32` times the limit, and every rig fails.

## What judgment changes

The result rejects one specific explanation: **the current cross-rig mismatch is not merely caused by different Jacobi geometry scales across rigs.** Moving tails into symmetric PCGLS geometry coordinates does not recover cross-rig rank-64 capacity, and the late tail becomes consistently worse.

The symmetric geometry-Jacobi rank-64 representation therefore closes without a rank increase, reversed scaling, CNN, FNO, or GPU rescue of a failed truth-aware capacity gate. This negative does not close the full C route or prove mathematical impossibility. It says that the next candidate must introduce a genuinely different mechanism rather than another fixed low-rank coordinate map.

## Independent validation and two pre-scoring failures

The formal implementation uses a sample-Gram eigendecomposition and maps the span back to the physical field. The independent program imports no formal implementation, directly computes an economy SVD for all thirteen `504x8192` training matrices, and independently rebuilds both the Jacobi map and the physical span.

The first two validator attempts fail closed before any independent SVD, residual, or scientific scoring. The first incorrectly requires mathematically equivalent Jacobi reductions to have byte-identical hashes; the second also requires bitwise equality for the scalar floor. Both failed records remain preserved. v239.2 only applies the already frozen numerical tolerance to this validation step. The formal output, candidate, folds, rank, and gates are neither rerun nor changed.

All final `18/18` independent checks pass. Maximum Jacobi inverse-diagonal relative difference is `9.22e-17`; maximum residual and summary differences are `1.12e-14 / 1.02e-14`; camera-permutation relative difference is `4.54e-16`; and the minimum training numerical rank is `504`. The final decision is `FAIL_CASE7_JACOBI_CANONICAL_TAIL_SUBSPACE_CAPACITY_V239`.

This is a truth-aware post-open representation-capacity negative on Case 7, not a deployment algorithm, exact-call reduction, wall/RSS result, external generalization, curved-ray validation, or real BOST result.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
