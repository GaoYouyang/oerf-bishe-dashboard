# v114 经典迁移控制：五个经典解释均未通过同精度门

> **证据状态：** `PASS_INDEPENDENT_RECOMPUTATION_DUAL_RIDGE_SCORE_V114`  
> **科学边界：** 这是已开封 PoolFire 开发集上的经典控制排除证据，不是算法突破、外部泛化、资源优势或真实 BOST 结果。

## 中文摘要

v111 的坐标条件 warm initializer 已在五条 PoolFire 轨迹、三套已知九视角几何和三个随机种子上通过 Formal Stage A。v114 进一步追问：这个信号是否能被更简单、同价或更便宜的经典方法解释？

我们在完全相同的观测、几何、候选引用、Zero-K4 基线和八项 matched-accuracy 门下，依次测试五个控制：

1. 缩放的精确反投影，直接作为 K0 起点；
2. 缩放的精确反投影，再接未修改 K1；
3. 几何 Jacobi-PCGLS K1；
4. 几何 Jacobi-PCGLS K2；
5. 用 K4 dual teacher 拟合、在线保持 `2A + 2A^T` 的条件 dual ridge，再接未修改 K1。

实验覆盖 `5` 条轨迹、`3` 套几何、`6` 个坐标图和每图 `11` 帧。前四个控制各有 `990` 个单元，dual ridge 另有 `990` 个单元，共 `4950` 个经典控制单元；同时引用 `2970` 个已冻结候选单元。

## 结果

| 控制 | trajectory gate | joint pass | severe harm | 最坏轨迹 observation p90 / Zero-K4 |
|---|---:|---:|---:|---:|
| scaled exact BP K0 | 0 / 5 | 0 / 990 | 990 / 990 | 2.4182 |
| scaled exact BP K1 | 0 / 5 | 0 / 990 | 990 / 990 | 2.1898 |
| geometry PCGLS K1 | 0 / 5 | 0 / 990 | 990 / 990 | 2.5229 |
| geometry PCGLS K2 | 0 / 5 | 0 / 990 | 990 / 990 | 1.9787 |
| conditional dual ridge + K1 | 0 / 5 | 0 / 990 | 990 / 990 | 1.7088 |

冻结的 observation harm 门是 `1.01`。dual ridge 是这五个方法中最强的经典控制，但五条轨迹的 observation p90 比值仍为 `1.3493 / 1.3166 / 1.7088 / 1.5602 / 1.4721`，全部超门。三个候选随机种子在所有单元上都同时优于 dual ridge 的 field 与 interior-gradient。

dual ridge 在线仍使用与候选相同的 `2A + 2A^T` 壳。其离线 K4 teacher 实际产生 `3960A + 3960A^T`，在线预测与 K1 实际产生 `1980A + 1980A^T`；离线成本明确披露，不被隐藏进“免费训练”。`90` 个拟合上下文中，`63` 个选择 `lambda=0.01`，`27` 个选择 `lambda=1.0`。

## 独立验证

独立程序没有导入正式 dual-ridge 预测器、选择器或 Krylov 包装器。它重新生成 K4 teacher、核岭模型、lambda 选择、dual K1 recurrence、全部指标、尾部和判决：

- 预测场、dual、残差、模型与选择最大差均为 `0`；
- score 的逐单元、聚合、成本与判决最大差均为 `0`；
- 独立实测完整账为 `5940A + 5940A^T`；
- API 级 truth-mutation noninterference 已通过；process-level never-read 尚未证明。

## 当前结论

v114 排除了五个具体的经典解释：它们都不能在同一成本壳和同一精度门下重现 v111 的开发集信号。因此 CNN 父控制序列获得继续运行的依据。

但这不是“我们的算法已经成功”。三个 CNN 父控制仍未全部完成；FNO 训练仍未授权；fresh wall/RSS、独立公开反应流外门与组内真实 BOST 均未通过。因此状态保持：

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `real_bost=false`

---

# v114 classical-transfer controls: five classical explanations fail the matched-accuracy gate

## English summary

v114 asks whether the v111 coordinate-conditioned warm-start signal can be explained by simpler classical methods under the same observations, geometries, candidate references, Zero-K4 baseline, and eight matched-accuracy gates.

Five controls are tested: scaled exact backprojection at K0; scaled exact backprojection followed by unchanged K1; geometry-Jacobi PCGLS K1; geometry-Jacobi PCGLS K2; and a conditional dual-ridge proposal trained against K4 dual teachers and followed by unchanged K1. The online dual-ridge shell remains `2A + 2A^T`.

The study covers five trajectories, three geometries, six coordinate maps, and eleven frames per map. The four Stage-0 controls contribute `3960` cells and dual ridge contributes another `990`, for `4950` classical-control cells, while referencing `2970` frozen candidate cells.

All five controls pass `0/5` trajectory gates. Each obtains `0/990` joint passes and `990/990` severe-harm cells. Conditional dual ridge is the strongest of the five, but its worst trajectory observation-p90 ratio versus Zero-K4 is `1.7088`, well above the frozen `1.01` gate. For every seed, the candidate has lower field and interior-gradient error than dual ridge in every cell.

An independent implementation rebuilds the teachers, kernel-ridge models, lambda selection, dual-K1 recurrence, metrics, tails, cost ledger, and decisions. Every reported numeric difference is zero, and the independently measured total is `5940A + 5940A^T`. API-level truth-mutation noninterference is established; process-level never-read is not.

This rules out five specific classical explanations on the opened PoolFire development family. It does **not** establish algorithmic breakthrough, resource acceleration, external generalization, or real-BOST performance. The CNN parent sequence has resumed; FNO training remains unauthorized.

