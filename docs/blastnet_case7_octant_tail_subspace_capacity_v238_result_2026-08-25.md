# v238：同为 64 维，固定八分块尾差空间比全局空间更差

## 结论

v236 已经否定固定全局 rank-64 尾差空间跨 rig 迁移，但保留了一个具体解释：也许尾差不是“全局低秩”，而是由空间局部结构组成。v238 在同一个已开封 BLASTNet Case 7、同一 13 条 rig、42 帧和 `K16-K11` 尾差上，只检验这一条解释。

三维场固定切成互不重叠的 `2x2x2` 八个 octant，每块用其他 12 条 rig 的 `504` 个局部尾差建立 rank-8 空间。八块总维数仍是 `8x8=64`，与 v236 全局 rank 64 完全相同；留出 rig 仍使用真值可见的最优投影系数，因此这是容量上界，不是部署预测器。

结果仍为 `0/13` 完整 rig。更关键的是，固定八分块不但没有救回迁移，反而在每条 rig 上都比全局 rank 64 更差：全局 p50/p90/worst 从 `0.645458 / 0.731692 / 0.805609` 恶化为 `0.667501 / 0.751069 / 0.833760`。

| 尾差空间 | 总维数 | 完整 rig | 全局 p50 | 全局 p90 | 全局 worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 Low64 控制 | `64` | `0/13` | `0.953580` | `0.974420` | `0.990718` |
| v236 全局 rank 64 | `64` | `0/13` | `0.645458` | `0.731692` | `0.805609` |
| v238 八分块 rank 8 x 8 | `64` | `0/13` | `0.667501` | `0.751069` | `0.833760` |

冻结门仍是每条 rig 的全帧与后期帧都满足 `p90 <= sqrt(0.1)=0.316228`、`worst <= 0.5`。v238 不是擦线失败：八分块全局 p90 是门的约 `2.37` 倍，worst 也高于门 `0.333760`。

## 它改变了什么判断

v236 只说明一个共享的全局低秩空间不能迁移；当时仍可能把失败归因于“全局基把互不相同的局部结构混在一起”。v238 做了同维数的直接对照：八个固定局部块在全部 13 条 rig 的全帧 p90 和后期 p90 上都比全局 rank 64 更差。

因此，当前缺失结构不能仅用“固定空间局部性”解释。固定不重叠 octant、每块 rank 8 的表示关闭；不增加每块 rank、不改成结果自适应分块，也不使用更大预测器把这次容量失败包装成成功。这不排除几何连续、非线性或 solver-native 的其他机制，更不是数学不可能证明。

## 独立验证

正式实现对每个留一 rig fold 的八块分别做样本 Gram 特征分解；独立程序不导入正式 runner，对全部 `8 x 13` 个 `504x1024` 训练块直接做 economy SVD。`12/12` 项检查全真，组合残差、分块残差和汇总最大差分别为 `3.33e-16 / 8.88e-16 / 2.22e-16`，最小训练数值秩为 `504`，分块基最大正交误差为 `3.55e-15`。

最终科学判决是 `FAIL_CASE7_OCTANT_TAIL_SUBSPACE_CAPACITY_V238`。它是 Case 7 开封后的 truth-aware 表示容量负结果，不是部署算法、调用减少、wall/RSS、外部泛化、curved ray 或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

---

# v238: at the same 64 dimensions, fixed octant tail spaces are worse than the global space

## Conclusion

v236 rejects cross-rig transfer of a fixed global rank-64 tail space, but leaves one concrete explanation open: perhaps the tail is not globally low-rank but composed of spatially local structure. v238 tests only that explanation on the same already opened BLASTNet Case 7, the same 13 rigs and 42 frames, and the same `K16-K11` tail.

The 3D field is partitioned into eight fixed, non-overlapping `2x2x2` octants. In every held-out-rig fold, each block receives a rank-8 basis from the `504` local tails in the other twelve rigs. The direct sum remains exactly `8x8=64` dimensions, matching the v236 global rank-64 control. Truth-aware optimal projection coefficients are still used on the held-out rig, so this is a capacity upper bound, not a deployment predictor.

The result remains `0/13` complete rigs. More importantly, fixed octants do not rescue transfer and are worse on every rig: global p50/p90/worst degrades from `0.645458 / 0.731692 / 0.805609` for global rank 64 to `0.667501 / 0.751069 / 0.833760` for octant rank 8 x 8.

| Tail space | Total dimension | Complete rigs | Global p50 | Global p90 | Global worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Low64 control | `64` | `0/13` | `0.953580` | `0.974420` | `0.990718` |
| v236 global rank 64 | `64` | `0/13` | `0.645458` | `0.731692` | `0.805609` |
| v238 octant rank 8 x 8 | `64` | `0/13` | `0.667501` | `0.751069` | `0.833760` |

The frozen gate remains all-frame and late-frame `p90 <= sqrt(0.1)=0.316228` and `worst <= 0.5` for every rig. This is not a borderline miss: the octant global p90 is about `2.37` times the limit, and worst exceeds its limit by `0.333760`.

## What judgment changes

v236 only shows that one shared global low-rank space does not transfer. The failure could still have been attributed to a global basis mixing distinct local structures. v238 supplies an equal-dimensional direct comparison: the eight fixed local blocks are worse than global rank 64 on both all-frame and late-frame p90 for all 13 rigs.

The missing structure therefore cannot be explained by fixed spatial locality alone. The non-overlapping octant, rank-8-per-block representation closes without increasing block rank, adapting partitions after results, or invoking a larger predictor to rescue a capacity failure. This does not exclude geometry-continuous, nonlinear, or solver-native mechanisms and is not a mathematical impossibility proof.

## Independent validation

The formal implementation applies a sample-Gram eigendecomposition to each of eight blocks in every held-out-rig fold. The independent program imports no formal runner and directly computes an economy SVD for all `8 x 13` `504x1024` training blocks. All `12/12` checks pass. Maximum combined-residual, block-residual, and summary differences are `3.33e-16 / 8.88e-16 / 2.22e-16`; minimum training numerical rank is `504`, and maximum block-basis orthogonality error is `3.55e-15`.

The final scientific decision is `FAIL_CASE7_OCTANT_TAIL_SUBSPACE_CAPACITY_V238`. This is a truth-aware post-open representation-capacity negative on Case 7, not a deployment algorithm, exact-call reduction, wall/RSS result, external generalization, curved-ray validation, or real BOST result.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
