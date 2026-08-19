# v162：全张量几何耦合确实改善尾部，但仍差 0.001035 过门

更新：2026-08-20

## 先说结论

v162 是现有数据上最后一个全局二次型几何机制审计。它不再把三个世界轴分开加权，而是从 active cameras 真正进入 forward 的单位射线构造完整横向灵敏度张量 `S = mean(I - d d^T)`，再固定使用 `W = S^-1`。完整有限差分梯度二次型保留所有非对角交叉项；倍数仍为 `0.03`，没有搜索矩阵函数、旋转、floor、归一化或 lambda。

正式运行与独立第二实现得到相同判决：`FAIL_FULL_TENSOR_GEOMETRY_H1_TEMPORAL_V162`。

- 12 个时间×相机分层通过 `11/12`；
- 唯一失败仍是 `t=0.75`、五相机；
- 该层 field / gradient / observation p90 为 `0.447236 / 0.751035 / 0.120629`；
- gradient p90 门是 `0.750000`，因此还差 `0.001035`；
- 但它确实优于各向同性 H1 的 `0.758639` 和 v161 对角几何 H1 的 `0.768197`。

这不是“完全没作用”。非对角耦合把最后的梯度尾部朝正确方向推进了，但没有越过结果前冻结的绝对门，因此不能包装成成功。

## 十二个冻结分层

下表列出全张量主策略的 p90，顺序为 field / gradient / observation。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.464 / 0.711 / 0.134，通过 | 0.376 / 0.582 / 0.141，通过 | 0.249 / 0.511 / 0.149，通过 |
| 0.25 | 0.478 / 0.709 / 0.119，通过 | 0.371 / 0.587 / 0.134，通过 | 0.251 / 0.514 / 0.146，通过 |
| 0.75 | 0.447 / **0.751** / 0.121，失败 | 0.361 / 0.611 / 0.131，通过 | 0.249 / 0.542 / 0.143，通过 |
| 1.00 | 0.458 / 0.718 / 0.121，通过 | 0.361 / 0.582 / 0.139，通过 | 0.240 / 0.507 / 0.148，通过 |

冻结 p90 门分别是 field `0.50`、gradient `0.75`、observation `0.20`。对应 worst 门、九相机 CGLS K16 no-harm 和调用账也全部逐项检查。

## 为什么这个负结果可信

正式程序覆盖 `39` 个 operator setups、`1,404` 个 cells 和 `5,616` 条四臂结果。耦合矩阵的非对角相对 Frobenius 范围为 `0.0374–0.4173`，说明主策略确实保留了实质性交叉轴信息，而不是把 v161 对角机制换名重跑。相机换序后，完整二次型最大相对差为 `6.20e-15`。

独立程序用标量循环重建 `S`，用显式 edge-order-2 差分重建余弦基导数，并用另一种广义特征值求解重算所有候选、观测和门。`21/21` 项检查通过；正式与独立逐 cell、汇总和算子数值最大差分别为 `1.64e-11 / 7.65e-12 / 7.05e-10`。

## 接下来怎么走

当前全局二次型几何各向异性家族到此关闭，包括看见结果后继续改矩阵公式、floor、归一化或 lambda。这个结论不关闭 C 路线，也不证明别的物理表示不可能。

下一步优先等待逐工况实验二维双分量位移，以及 camera/frame/calibration/checkpoint/t、单位、符号、crop/resize/mask、重复背景噪声和组内认可基线。若继续受控代理，必须先有一个物理上真正不同、不是全局二次型调参的新机制。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`、`gpu_rental_authorized=false`。

---

# v162: full-tensor geometry coupling improves the tail but still misses by 0.001035

Updated: 2026-08-20

v162 is the final audit of the current global quadratic geometry family on the available controlled data. Active forward rays define the full transverse-sensitivity tensor `S = mean(I - d d^T)`, and the unique primary uses `W = S^-1`. The finite-difference gradient quadratic retains every off-diagonal cross-axis term. The multiplier stays fixed at `0.03`; no matrix function, rotation, floor, normalization, or lambda is selected from v162 results.

The formal run and independent implementation agree on `FAIL_FULL_TENSOR_GEOMETRY_H1_TEMPORAL_V162`. The primary clears 11/12 frozen time-by-camera strata. The sole miss remains five cameras at `t=0.75`, where field / gradient / observation p90 are `0.447236 / 0.751035 / 0.120629`. The frozen gradient gate is `0.750000`, leaving a gap of `0.001035`.

The change is physically meaningful rather than null: the missed gradient tail improves from `0.758639` for isotropic H1 and `0.768197` for diagonal geometry H1 to `0.751035`. Nevertheless, a real improvement that remains above a preregistered absolute gate is still a failure.

The off-diagonal coupling norm ranges from `0.0374` to `0.4173`, and camera reordering changes the complete quadratic by at most `6.20e-15` relatively. A second implementation reconstructs tensors by scalar accumulation, differentiates analytic cosine factors with an explicit finite-difference stencil, uses a different generalized eigensolver, and recomputes all fields, observations, metrics, and decisions. All `21/21` checks pass; maximum per-cell, summary, and operator differences are `1.64e-11`, `7.65e-12`, and `7.05e-10`.

This result closes the current family of global quadratic geometry-anisotropy mechanisms and any post-hoc matrix or lambda tuning. It does not close the C route. Further work waits for condition-matched experimental two-component displacements with complete metadata, or for a genuinely different preregistered physical mechanism. This is not predictor training, a GPU case, a resource result, paired experimental BOST, or an algorithm breakthrough.
