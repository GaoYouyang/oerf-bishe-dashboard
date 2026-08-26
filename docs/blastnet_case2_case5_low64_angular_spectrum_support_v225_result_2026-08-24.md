# v225：完整九相机角谱互支持仍无法建立安全回退

## 结论

v224 已经排除“把九个相机压成单一最坏 Jackknife 分数或单一最大视角残差”这两个标量回退。v225 因此不再压成一个阈值，而是保留九台相机的完整模式：每台相机的删一重拟合漂移、分块残差和二者局部配对，再用 reported geometry 的固定角向基进行排列不变编码。

正式程序与完全独立第二实现得到一致判决：

`FAIL_LOW64_ANGULAR_SPECTRUM_MUTUAL_SUPPORT_V225`

主策略在已开封 Case 2 的 `715` 个单元中接受了 `523` 个，其中 `145` 个是冻结门定义下的不安全单元；在已知安全 Case 5 的 leave-one-rig-out 审计中，至少一个 rig 的接受率为 `0%`。因此它既没有 fail-closed 安全性，也没有每个 rig 都达到 `10%` 的最低效用。

## 做了什么

对每个九视角二维观测，Low-64 响应先计算九个删一相机漂移和九个相机分块残差。每一组九维量都作固定 L2 归一化，并加入逐相机乘积通道；随后用相机 reported direction 的方位角构造 `1, cos(m a), sin(m a)`、`m=1..4` 的固定九阶角向基，得到 `27` 维排列不变特征。当前九相机角向矩阵满秩，所以这个编码没有把九个相机再次压成一个标量。

策略只用已知安全 Case 5 拟合：每个训练行的允许半径等于它到另一 rig 最近邻的距离。Case 5 采用完整 leave-one-rig-out；Case 2 在其真值标签未读取时，由全部 Case 5 行一次性评分并封存决策。没有阈值搜索、超参数搜索、unsafe 标签训练、异常点剔除或事后改变调和阶数。

便宜 control 使用同一个互支持规则，但只保留 v224 的两个最大值：最坏删相机漂移和最大逐相机残差。

## 结果

| 策略与条件 | 接受单元 | 各 rig 最低接受率 | 接受的不安全单元 | 完整 rig 精度通过 | 判决 |
|---|---:|---:|---:|---:|---|
| 27 维角谱，Case 5 LORO | `252 / 546` | `0.00%` | `0` | `13 / 13` | 效用门失败 |
| 27 维角谱，Case 2 | `523 / 715` | `5.45%` | `145` | `0 / 13` | 安全与精度失败 |
| 两标量 control，Case 5 LORO | `339 / 546` | `14.29%` | `0` | `13 / 13` | 此条件通过 |
| 两标量 control，Case 2 | `186 / 715` | `18.18%` | `132` | `0 / 13` | 安全与精度失败 |

主角谱策略在 Case 2 接受的 `523` 个单元中，只有 `378` 个安全；最大 matched ratio 达到 `1.87529`。便宜 control 接受的 `186` 个单元中有 `132` 个不安全，最大 matched ratio 为 `1.91394`。两者都不能驱动 Direct Low-64 PCGLS K11 / Zero-PCGLS K16 的安全回退。

结果前的最低效用门要求每个 rig 至少接受 `10%`。主策略在 Case 5 的最低接受率为 `0%`，在 Case 2 为 `5.45%`；因此即使忽略不安全接受，它也没有稳定的逐 rig 调用收益。`12A+11A^T` 与 `16A+16A^T` 只是候选和回退的逻辑账；本次没有建立实际 exact-call 减少，更没有 fresh wall/RSS 结果。

## 独立复算

独立程序使用正规矩阵特征分解而不是正式 SVD，逐相机重建全部 Low-64 解，并用显式三重循环而不是正式向量化 Gram 距离重算互支持。`17 / 17` 项必需检查全部通过；正式与独立特征、距离、汇总和相机换序最大差分别为 `4.41e-14`、`8.42e-13`、`5.70e-11` 和 `4.35e-14`，所有离散接受决策完全一致。

独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_ANGULAR_SPECTRUM_SUPPORT_V225`

共享冻结的底层 physics kernels 仍存在，因此 `end_to_end_physics_independence_proven=false`。

## 证据边界

- v225 是已开封 Case 2/5 上的 post-open 跨工况策略诊断，不是 fresh 外门；
- 它关闭的是“固定角谱表示 + Case 5 跨 rig 一类互支持”这一条策略，不证明所有多视角机制不可能；
- 不允许看到结果后调整尺度、局部半径、调和阶数、特征通道或 `10%` 接受门；
- 不训练 CNN/FNO/UNO/DeepONet，不租 GPU，也不运行 wall/RSS 或打开 Case 4/6；
- 没有部署算法、exact-call 收益、外部泛化、曲线光路或真实 BOST 结果。

`algorithm_breakthrough=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

---

# v225: The Full Nine-Camera Angular-Spectrum Support Still Cannot Establish a Safe Fallback

## Conclusion

v224 rules out two scalar fallbacks that compress nine cameras into either the worst jackknife score or the maximum view residual. v225 therefore preserves the complete nine-camera pattern: each camera's leave-one-out refit drift, block residual, and local pairing, encoded through a fixed permutation-invariant angular basis derived from reported geometry.

The formal program and fully separate second implementation agree on:

`FAIL_LOW64_ANGULAR_SPECTRUM_MUTUAL_SUPPORT_V225`

The primary accepts `523` of `715` opened Case 2 cells, including `145` cells that are unsafe under the frozen gates. In leave-one-rig-out auditing on known-safe Case 5, at least one rig has `0%` acceptance. The policy therefore provides neither fail-closed safety nor the minimum `10%` utility in every rig.

## What was done

For every nine-view 2D observation, the Low-64 response produces nine leave-one-camera drifts and nine camera-block residuals. Each nine-vector receives fixed L2 normalization and a pointwise interaction channel. Reported camera directions then define the fixed angular basis `1, cos(m a), sin(m a)` for `m=1..4`, producing a `27`-dimensional permutation-invariant feature. The current nine-camera angular matrix is full rank, so the encoding does not collapse the cameras back to one scalar.

The one-class policy fits only known-safe Case 5. Each training row receives a radius equal to its nearest row from another rig. Case 5 uses complete leave-one-rig-out evaluation; Case 2 decisions are generated and sealed from all Case 5 rows before Case 2 truth labels are read. There is no threshold search, hyperparameter search, unsafe-label training, outlier removal, or post-result harmonic-order change.

The cheap control uses the same mutual-support policy but retains only the two v224 maxima: worst leave-one-camera drift and maximum per-camera residual.

## Results

| Policy and condition | Accepted cells | Minimum rig acceptance | Accepted unsafe cells | Complete rigs passing accuracy | Decision |
|---|---:|---:|---:|---:|---|
| 27-D angular spectrum, Case 5 LORO | `252 / 546` | `0.00%` | `0` | `13 / 13` | utility gate fails |
| 27-D angular spectrum, Case 2 | `523 / 715` | `5.45%` | `145` | `0 / 13` | safety and accuracy fail |
| Two-scalar control, Case 5 LORO | `339 / 546` | `14.29%` | `0` | `13 / 13` | passes this condition |
| Two-scalar control, Case 2 | `186 / 715` | `18.18%` | `132` | `0 / 13` | safety and accuracy fail |

Only `378` of the `523` Case 2 cells accepted by the primary are safe, and its maximum matched ratio reaches `1.87529`. The cheap control accepts `132` unsafe cells among `186`, with a maximum matched ratio of `1.91394`. Neither can safely control the Direct Low-64 PCGLS K11 / Zero-PCGLS K16 fallback.

The preregistered utility gate requires at least `10%` acceptance in every rig. The primary reaches only `0%` minimum acceptance on Case 5 and `5.45%` on Case 2. Even if unsafe acceptance were ignored, stable per-rig call utility is absent. The `12A+11A^T` versus `16A+16A^T` figures are only logical ledgers for the candidate and fallback; no actual exact-call reduction or fresh wall/RSS result is established.

## Independent recomputation

The independent program uses normal-matrix eigensystems instead of the formal SVD, rebuilds every per-camera Low-64 solution, and evaluates mutual support with explicit triple loops instead of formal vectorized Gram distances. All `17 / 17` required checks pass. Maximum formal-independent feature, distance, summary, and camera-permutation differences are `4.41e-14`, `8.42e-13`, `5.70e-11`, and `4.35e-14`, and every discrete acceptance decision matches.

The independent status is:

`PASS_INDEPENDENT_RECOMPUTATION_LOW64_ANGULAR_SPECTRUM_SUPPORT_V225`

Frozen low-level physics kernels remain shared, so `end_to_end_physics_independence_proven=false`.

## Evidence boundary

- v225 is a post-open cross-condition policy diagnostic on opened Cases 2 and 5, not a fresh external gate;
- it closes the specific fixed angular-spectrum plus Case 5 cross-rig one-class mutual-support policy, not every possible multiview mechanism;
- scale, local radius, harmonic order, feature channels, and the `10%` acceptance gate may not be changed after results;
- no CNN/FNO/UNO/DeepONet training, GPU rental, wall/RSS run, or Case 4/6 opening is authorized;
- there is no deployment algorithm, exact-call gain, external generalization, curved-ray, or real-BOST result.

`algorithm_breakthrough=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
