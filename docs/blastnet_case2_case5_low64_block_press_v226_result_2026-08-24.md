# v226：相机分块 PRESS 实现零危险误接，但逐 rig 效用门差一个帧

## 结论

v225 证明固定角谱加跨 rig 一类互支持仍会接受危险单元。v226 因此换成物理上不同的预测一致性证书：逐个遮掉一台相机，用另外八台相机拟合固定 Low-64 响应，再检验被遮相机的二维观测是否能被预测。

正式程序与完全独立第二实现得到一致判决：

`FAIL_LOW64_BLOCK_PRESS_CERTIFICATE_V226`

这是一个明显收窄瓶颈、但必须按原门判负的结果。主 PRESS 证书在 Case 2 接受 `297/715` 个单元，接受的 `297` 个全部安全，`197` 个不安全单元全部被拒绝。Direct Low-64 PCGLS K11 / Zero-PCGLS K16 混合策略在 Case 2 和 Case 5 都达到 `13/13` 完整 rig 精度，且每个 rig 的平均 `A/A^T` 账都严格低于 K16 reference。

但是，结果前冻结的最低效用门要求每个 Case 5 留一 rig 至少接受 `10%`。最差 rig 只接受 `4/42=9.52%`，而通过至少需要 `5/42`。因此当前证书差的是一个预注册效用帧，不能事后改门或包装成成功。

## 做了什么

对每个九相机单元，固定 Low-64 响应矩阵记为 `M`，观测记为 `y`。正式证书先计算全相机拟合残差 `E_full=||Mc-y||`；然后依次遮掉第 `j` 台相机，用其余八台相机拟合 `c_-j`，并在被遮相机上计算预测误差。九个遮挡误差的平方和开根号得到 `E_press`，主分数为：

`P = E_press / max(E_full, 1e-12 ||y||)`

分数越低越安全。它只读取部署可见的二维观测和 reported geometry；没有读取 CFD 真值、危险标签、轨迹名称或未开封工况来生成特征或阈值。

阈值只由已知安全 Case 5 校准。每个训练 rig 先取升序分数的固定 `ceil(0.1n)-1` 顺序统计量，再取这些 rig 统计量的最大值；Case 5 做完整留一 rig，Case 2 使用全部 Case 5 生成的单一阈值。等号接受，没有阈值搜索、超参数搜索、异常点删除或危险标签训练。

便宜 control 只使用全拟合相对观测残差 `E_full/||y||`，并沿用相同的 Case 5 校准和全部门。

## 结果

| 策略与条件 | 接受单元 | 各 rig 最低接受率 | 接受的不安全单元 | 完整 rig 精度通过 | 判决 |
|---|---:|---:|---:|---:|---|
| Block-PRESS，Case 5 LORO | `126 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | 效用门差一个帧 |
| Block-PRESS，Case 2 | `297 / 715` | `32.73%` | `0` | `13 / 13` | 安全与精度通过 |
| 全拟合残差 control，Case 5 LORO | `89 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | 效用门失败 |
| 全拟合残差 control，Case 2 | `553 / 715` | `74.55%` | `61` | `0 / 13` | 安全与精度失败 |

主策略在 Case 2 的安全接受率为 `57.34%`，逐 rig 接受率为 `32.73%` 至 `47.27%`；在 Case 5 的总接受率为 `23.08%`，但逐 rig 范围是 `9.52%` 至 `54.76%`。混合策略的 Case 2/5 最大 matched ratio 分别为 `1.027761` 和 `1.007896`，均守住冻结完整精度门。

被接受的 Direct K11 逻辑账为 `12A+11A^T`，回退 Zero-PCGLS K16 为 `16A+16A^T`。主策略每个 rig 的平均 `A` 和 `A^T` 均严格低于 `16`；最保守的 Case 5 rig 为 `15.619A+15.524A^T`。但本次效用证书整体失败，所以这些仍只是封存的实际调用账，不授权 fresh wall/RSS，也不能声称部署资源收益。

## 独立复算

独立程序不用正式 SVD，改用正规矩阵特征分解；它以显式帧和相机循环重建全拟合、九个八相机拟合、PRESS 分数、Case 5 顺序统计阈值、接受决策、逐 rig 物理门与调用账。所有 `16/16` 项必需检查通过；正式与独立特征、阈值、汇总和相机换序最大差分别为 `1.11e-15`、`2.22e-16`、`3.30e-11` 和 `1.55e-15`，离散接受决策完全一致。

独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_BLOCK_PRESS_CERTIFICATE_V226`

共享冻结的底层 physics kernels 仍存在，因此 `end_to_end_physics_independence_proven=false`。

## 证据边界

- v226 是已开封 Case 2/5 上的 post-open 跨工况证书诊断，不是 fresh 外门；
- 它关闭的是当前九相机 exact block-PRESS 证书，不证明全部多视角证书或整个 C 路线不可能；
- 不允许看到结果后修改 PRESS 公式、数值 floor、阈值、`10%` 接受比例、Low-64 秩或 PCGLS 深度；
- 当前只定义九相机公式，没有建立 `5/7/12` 相机可变基数行为；
- 不训练 CNN/FNO/UNO/DeepONet，不租 GPU，也不运行 wall/RSS 或打开 Case 4/6；
- 没有部署算法、稳定 exact-call 收益、外部泛化、曲线光路或真实 BOST 结果。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v226: Camera-Block PRESS Reaches Zero Unsafe Accepts but Misses the Per-Rig Utility Gate by One Frame

## Conclusion

v225 shows that fixed angular-spectrum cross-rig one-class support still accepts unsafe cells. v226 therefore uses a physically different predictive-consistency certificate: hold out each camera, fit the fixed Low-64 response on the other eight cameras, and test whether the held-out 2D observation can be predicted.

The formal program and fully separate second implementation agree on:

`FAIL_LOW64_BLOCK_PRESS_CERTIFICATE_V226`

This result sharply narrows the bottleneck but must remain a failure under the preregistered gate. The primary PRESS certificate accepts `297/715` Case 2 cells. All `297` accepted cells are safe, and all `197` unsafe cells are rejected. The Direct Low-64 PCGLS K11 / Zero-PCGLS K16 mixed policy reaches `13/13` complete-rig accuracy in both Cases 2 and 5, with mean `A/A^T` ledgers strictly below the K16 reference in every rig.

However, the preregistered utility gate requires at least `10%` acceptance in every held-out Case 5 rig. The worst rig accepts only `4/42=9.52%`, while passing requires at least `5/42`. The current certificate therefore misses by one preregistered utility frame and cannot be rescued by changing the gate after results.

## What was done

For each nine-camera cell, let the fixed Low-64 response matrix be `M` and the observation be `y`. The certificate first computes the full-camera residual `E_full=||Mc-y||`. Camera `j` is then held out, the remaining eight cameras fit `c_-j`, and prediction error is measured on the held-out block. The root-sum-square of the nine held-out errors is `E_press`, and the primary score is:

`P = E_press / max(E_full, 1e-12 ||y||)`

Lower is safer. The score reads only deployment-visible 2D observations and reported geometry. CFD truth, unsafe labels, trajectory names, and unopened conditions do not enter feature or threshold generation.

Thresholds are calibrated only from known-safe Case 5. Each fit rig contributes the fixed ascending order statistic at index `ceil(0.1n)-1`, and the threshold is the maximum of those rig statistics. Case 5 uses complete leave-one-rig-out evaluation; Case 2 uses one threshold formed from all Case 5 rigs. Equality accepts. There is no threshold search, hyperparameter search, outlier deletion, or unsafe-label training.

The cheap control uses only the full-fit relative observation residual `E_full/||y||`, with the same Case 5 calibration and all the same gates.

## Results

| Policy and condition | Accepted cells | Minimum rig acceptance | Accepted unsafe cells | Complete rigs passing accuracy | Decision |
|---|---:|---:|---:|---:|---|
| Block-PRESS, Case 5 LORO | `126 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | utility gate misses by one frame |
| Block-PRESS, Case 2 | `297 / 715` | `32.73%` | `0` | `13 / 13` | safety and accuracy pass |
| Full-fit residual control, Case 5 LORO | `89 / 546` | `4/42 = 9.52%` | `0` | `13 / 13` | utility gate fails |
| Full-fit residual control, Case 2 | `553 / 715` | `74.55%` | `61` | `0 / 13` | safety and accuracy fail |

The primary safely accepts `57.34%` of Case 2 safe cells, with per-rig acceptance from `32.73%` to `47.27%`. It accepts `23.08%` of Case 5 overall, but per-rig acceptance ranges from `9.52%` to `54.76%`. Maximum matched ratios for the mixed policy are `1.027761` in Case 2 and `1.007896` in Case 5, satisfying the frozen complete-accuracy gates.

The accepted Direct K11 path has a logical ledger of `12A+11A^T`; fallback Zero-PCGLS K16 uses `16A+16A^T`. Every primary rig has mean `A` and `A^T` below `16`; the most conservative Case 5 rig uses `15.619A+15.524A^T`. Because the overall utility certificate fails, these remain sealed actual-call ledgers and do not authorize fresh wall/RSS or a deployment resource claim.

## Independent recomputation

The independent program replaces the formal SVD with normal-matrix eigendecomposition. Explicit frame and camera loops rebuild the full fit, all nine eight-camera fits, PRESS scores, Case 5 order-statistic thresholds, accept decisions, rig-level physics gates, and call ledgers. All `16/16` required checks pass. Maximum formal-independent feature, threshold, summary, and camera-permutation differences are `1.11e-15`, `2.22e-16`, `3.30e-11`, and `1.55e-15`, and every discrete accept decision matches.

The independent status is:

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_BLOCK_PRESS_CERTIFICATE_V226`

Frozen low-level physics kernels remain shared, so `end_to_end_physics_independence_proven=false`.

## Evidence boundary

- v226 is a post-open cross-condition certificate diagnostic on opened Cases 2 and 5, not a fresh external gate;
- it closes this exact nine-camera block-PRESS certificate, not every multiview certificate or the C route;
- the PRESS formula, numerical floor, threshold, `10%` acceptance fraction, Low-64 rank, and PCGLS depths may not be changed after results;
- only the nine-camera formula is defined here; `5/7/12` variable-cardinality behavior is not established;
- no CNN/FNO/UNO/DeepONet training, GPU rental, wall/RSS run, or Case 4/6 opening is authorized;
- there is no deployment algorithm, stable exact-call gain, external generalization, curved-ray, or real-BOST result.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
