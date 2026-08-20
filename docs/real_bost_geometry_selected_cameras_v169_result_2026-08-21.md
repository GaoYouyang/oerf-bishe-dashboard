# v169：纯几何相机选择反而放大五相机梯度尾部

更新：2026-08-21

## 先说结论

v169 检验了一个比继续改重建先验更便宜、也更直接的解释：此前五相机失败，会不会只是固定相机组合选得不好？

唯一主策略在读取结果前冻结。它只读取报告相机几何，在九相机中穷举五相机或七相机子集；用 63 个低频、去常数并经 H1 白化的 DCT 模态构造几何观测 Gram 矩阵，再按秩、对数伪行列式、最小正特征值和迹的固定次序选子集。选择过程不读取三维真值或二维观测。

正式运行与完全独立第二实现得到相同判决：`FAIL_GEOMETRY_SELECTED_CAMERAS_V169`。

主策略通过 `8/12` 个冻结时间×相机分层。七相机和九相机的八个分层全部通过，但四个五相机分层全部失败。四个五相机 gradient p90 为 `0.895479 / 0.883457 / 0.895914 / 0.860270`，均高于 `0.75` 门；冻结 H1 对照在原固定五相机组合上分别为 `0.708532 / 0.695875 / 0.758639 / 0.712033`。

选择器并没有退化成旧组合：13 套几何的五相机和七相机选择都与旧固定组合不同，五相机得到 10 个不同子集，七相机得到 7 个。也就是说，这次负结果确实检验了新的物理相机组合，而不是把旧输入换名重跑。

因此关闭这个精确定义的“纯几何低频可观测性选相机可以修复稀疏视角梯度尾部”假设。它不证明所有传感器设计都无效，也不关闭整个 C 路线；但瓶颈不能再简单归因于原固定五相机名单。

## 机制与成本

每套九相机几何先按真正进入 forward 的旋转、平移、内参和畸变参数做确定性规范化。对每个候选子集，63 个 DCT 模态通过该子集的直线射线算子投影，得到 H1 白化后的 measurement Gram 矩阵。选择目标依次最大化：

1. 数值秩；
2. 对数伪行列式；
3. 最小正特征值；
4. 迹。

特征值 cutoff 固定为 `1e-12`，完全并列时取字典序最先子集。五相机每套几何比较 `126` 个子集，七相机比较 `36` 个，九相机只有完整集合。

几何缓存构建披露为 `13,299` 个 forward-equivalent setup projections；它不是免费部署成本。缓存存在后，相机选择本身为 `+0A+0A^T`。冻结 H1 每个 cell 为 `1A+1A^T`，同一新子集上的 CGLS K16 control 为 `16A+16A^T`。

## 十二个冻结分层

下表为主策略的 gradient p90 / worst；绝对门分别为 `0.75 / 1.00`。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.895479 / 0.953989 FAIL | 0.604097 / 0.647006 PASS | 0.494193 / 0.529490 PASS |
| 0.25 | 0.883457 / 1.135656 FAIL | 0.604143 / 0.670835 PASS | 0.510066 / 0.524849 PASS |
| 0.75 | 0.895914 / 1.026562 FAIL | 0.605283 / 0.688576 PASS | 0.538779 / 0.573544 PASS |
| 1.00 | 0.860270 / 0.920675 FAIL | 0.606967 / 0.632028 PASS | 0.505160 / 0.529099 PASS |

五相机四层的 field p90 为 `0.333968 / 0.323619 / 0.331804 / 0.309466`，observation p90 为 `0.133857 / 0.134689 / 0.132072 / 0.135170`，都守住各自绝对门。失败仍集中在三维梯度尾部。

## 独立复算

第二实现独立重建 DCT 模态、射线算子、全部 `39` 个相机选择、冻结 H1、同子集 CGLS K16、`1,404` 个 cells、`2,808` 条评分记录和十二个分层。`27/27` 项独立检查全部通过。

选择分数、H1 系数、CGLS 系数、逐 cell 指标与汇总指标的最大差分别为 `6.19e-12`、`8.24e-11`、`1.83e-10`、`1.95e-11` 和 `6.65e-12`。算子最大差为 `1.48e-11`；reduced/direct forward 与 quadratic/direct residual 最大差为 `9.06e-14 / 9.30e-14`。离散子集、逐层判决和总判决完全一致。

## 证据边界

这关闭的是 63 个低频 H1 白化 DCT 模态、固定字典序目标和当前直线射线几何下的纯几何子集选择器。它不排除任务自适应、观测自适应、噪声鲁棒或硬件约束下的其他传感器设计。

二维位移仍由可执行三维密度场和相机几何仿真，不是逐工况配对的真实 BOST。没有 predictor、fresh wall/RSS、独立外部工况或真实 BOST 结果。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v169: geometry-only camera selection amplifies five-camera gradient tails

Updated: 2026-08-21

v169 tests a cheaper and more direct explanation than another reconstruction-prior variant: did the five-camera failure arise merely because the previous fixed roster was poorly chosen?

The sole primary was frozen before results. It reads only reported camera geometry, enumerates all five- or seven-camera subsets of each nine-camera setup, and evaluates 63 zero-mean low-frequency DCT modes whitened by the frozen H1 penalty. Subsets are selected by a fixed lexicographic objective over numerical rank, log pseudodeterminant, minimum positive eigenvalue, and trace. Neither 3D truth nor 2D observation enters selection.

The formal run and fully independent second implementation agree on `FAIL_GEOMETRY_SELECTED_CAMERAS_V169`.

The primary clears `8/12` frozen time-by-camera strata. All seven- and nine-camera strata pass; every five-camera stratum fails. Five-camera gradient p90 values are `0.895479 / 0.883457 / 0.895914 / 0.860270`, all above the `0.75` gate. The frozen H1 control on the previous fixed five-camera roster gives `0.708532 / 0.695875 / 0.758639 / 0.712033`.

The selector is nondegenerate. Its five- and seven-camera subsets differ from the previous fixed roster in all 13 geometries, yielding ten unique five-camera subsets and seven unique seven-camera subsets. This is therefore a genuine camera-layout test rather than a renamed replay of the previous inputs.

Each five-camera geometry compares 126 subsets and each seven-camera geometry compares 36. Geometry-cache construction is disclosed as 13,299 forward-equivalent setup projections and is not treated as free deployment work. After that cache exists, selection adds `0A+0A^T`; frozen H1 uses `1A+1A^T` per cell, while same-selected-subset CGLS K16 uses `16A+16A^T`.

The independent implementation rebuilds the DCT modes, ray operators, all 39 selections, frozen H1, same-subset CGLS K16, 1,404 cells, 2,808 scored rows, and all twelve strata. All `27/27` checks pass. Maximum selection-score, H1-coefficient, CGLS-coefficient, per-cell, and summary differences are `6.19e-12`, `8.24e-11`, `1.83e-10`, `1.95e-11`, and `6.65e-12`. Discrete subsets and every verdict agree exactly.

This closes only the exact geometry-only low-frequency H1-whitened subset selector. It does not rule out task-adaptive, observation-adaptive, noise-aware, or hardware-constrained sensor design, and it does not close the C route. The sparse-view miss can no longer be explained merely by the previous fixed five-camera roster under this criterion.

Observations remain simulated from executable reconstructed 3D density fields and camera geometry rather than condition-matched experimental BOST. No predictor, fresh wall/RSS, external-condition, real-BOST, paper-success, or algorithm-breakthrough claim is established.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
