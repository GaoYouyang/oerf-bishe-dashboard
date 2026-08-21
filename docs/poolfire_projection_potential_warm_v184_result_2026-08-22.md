# v184：残差大多可积，但标量势逆提升不是可用的三维暖启动

## 做了什么

v183 表明按相机与探测器分量建模比一条全局残差方向更有效，但仍没有通过完整门。随后对封存残差做的只读诊断发现，大部分二维双分量残差可由每台相机的一张标量势图的梯度解释。v184 因此在看结果前冻结了一个物理上不同、无训练的机制：

1. 每台相机独立把当前双分量残差积分成零均值探测器平面势场；
2. 用同一报告几何下的标量 straight-ray 投影和精确转置把势场提升到冻结的 1,009 维仿射坐标；
3. 只在当前观测上求一个标量步长；
4. 比较直接暖场 K0 与再走一步未修改物理 CGLS K1。

整个预测只读取当前观测、有效相机 ID、报告几何和 fit-only 仿射基，没有真值、训练、候选搜索、ridge、阻尼、裁剪或回退。

## 独立复算后的结果

势场拟合本身数值稳定：所有图连通，至少保留 `99.895%` 的已定义导数分量，最差相机块仍解释 `88.569%` 的探测器残差能量；KKT、零均值和可观测线搜索 stationarity 都远低于冻结数值门。

但三维逆提升失败，而且不是边界失败：

| K1 arm | field p90 | gradient p90 | observation p90 | 严格通过 |
|---|---:|---:|---:|---:|
| 五相机 | 0.661613 | 0.911014 | 0.402227 | 0/52 |
| 九相机 | 0.636139 | 0.841591 | 0.446146 | 0/52 |

冻结 p90 门分别为 field `0.50`、gradient `0.75`、observation `0.20`。两档相机的三个 p90 都越门，13 套标定和 4 个时间层也都没有完整通过。相较 v183 K1 的五相机 `0.445694 / 0.612373 / 0.226659` 与九相机 `0.371621 / 0.508927 / 0.207224`，v184 在三个指标上都明显退化。

完全独立的第二实现使用不同的三点 Gauss 标量射线积分与自建稠密 KKT，先封存独立候选，再读取正式数组比对。`50/50` 检查全真；候选场、势场与指标最大差分别约为 `6.12e-12`、`1.48e-11`、`1.85e-12`，离散判决完全一致。

## 科学结论

正式判决是 `FAIL_PROJECTION_POTENTIAL_WARM_V184`。

这次结果把两个命题分开了：**“残差大多是可积的”得到支持，但“该标量势经当前 scalar-ray Jacobi 逆提升就是有用三维暖方向”被否定。** 二维可积性并不能解决三维逆问题的不可辨识性，也不能保证提升方向与 field / gradient 兼容。

因此关闭当前精确的“零均值 detector potential + scalar-ray Jacobi lift + 单次观测线搜索 + 未修改 K1”机制。不事后调整差分、gauge、ridge、阻尼、步长、相机子集或门槛，也不用 CNN/FNO/UNO/DeepONet 或 GPU 挽救。它没有关闭完整 C 路线，但在没有真实二维 BOS 位移或物理上真正不同的新机制前，不再扩建这条分支。

该试验仍只是已开封 PoolFire 四帧上的 post-open 机制诊断。没有建立 exact-call 减少、wall/RSS 加速、独立外部泛化、curved ray 或真实 BOST 结论；`algorithm_breakthrough=false`。

# v184: the residual is mostly integrable, but the scalar-potential inverse lift is not a useful 3D warm start

## What was tested

v183 showed that camera-component structure is more useful than one global residual direction, yet still failed the complete gate. A sealed read-only diagnosis then found that most of the two-component residual can be explained as the detector-plane gradient of one scalar potential per camera. Before seeing v184 results, we froze a physically distinct, training-free mechanism:

1. independently integrate each camera residual into a zero-mean detector-plane potential;
2. lift the potential through a scalar straight-ray projection and exact transpose into the frozen 1,009-dimensional affine coordinates;
3. solve one scalar step using only the current observation;
4. compare the direct warm field K0 and one unchanged physical CGLS step K1.

Prediction reads only the current observation, active camera IDs, reported geometry, and the fit-only affine basis. It uses no truth, training, candidate search, ridge, damping, clipping, or fallback.

## Independently recomputed result

Potential fitting itself is numerically stable: every mask is connected, at least `99.895%` of derivative components are defined, and the worst camera block still explains `88.569%` of detector residual energy. KKT, zero-mean, and observable line-search stationarity are all far below their frozen numerical limits.

The 3D inverse lift nevertheless fails by wide margins:

| K1 arm | field p90 | gradient p90 | observation p90 | strict-safe |
|---|---:|---:|---:|---:|
| Five cameras | 0.661613 | 0.911014 | 0.402227 | 0/52 |
| All nine | 0.636139 | 0.841591 | 0.446146 | 0/52 |

The frozen p90 gates are `0.50`, `0.75`, and `0.20`, respectively. All three p90 metrics fail under both sensor arms, and no calibration or time stratum passes completely. Relative to v183 K1, which reached `0.445694 / 0.612373 / 0.226659` with five cameras and `0.371621 / 0.508927 / 0.207224` with all nine, v184 is materially worse on every metric.

The fully independent second implementation uses three-point Gauss scalar-ray integration and a separately assembled dense KKT system. It seals independent candidates before opening formal arrays. All `50/50` checks pass; maximum candidate-field, potential, and metric differences are about `6.12e-12`, `1.48e-11`, and `1.85e-12`, with identical discrete decisions.

## Scientific conclusion

The formal verdict is `FAIL_PROJECTION_POTENTIAL_WARM_V184`.

This separates two claims: **the residual is predominantly integrable, but that scalar potential does not become a useful 3D warm direction under the frozen scalar-ray Jacobi lift.** Detector-plane integrability neither resolves 3D non-identifiability nor guarantees field- and gradient-compatible lifting.

We therefore close this exact zero-mean detector-potential plus scalar-ray Jacobi lift, one observable line search, and unchanged K1 mechanism. No post-hoc difference stencil, gauge, ridge, damping, step, camera-subset, threshold, larger-model, or GPU rescue is allowed. This does not close the full C route, but this branch will not be expanded without real paired 2D BOS displacement or a genuinely different physical mechanism.

This remains a post-open mechanism diagnostic on four already opened PoolFire frames. It establishes no exact-call reduction, wall/RSS speedup, independent external generalization, curved-ray validity, or real-BOST result; `algorithm_breakthrough=false`.
