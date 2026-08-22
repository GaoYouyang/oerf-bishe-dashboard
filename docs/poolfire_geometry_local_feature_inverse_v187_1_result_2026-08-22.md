# v187.1：去掉共享回归后仍失败，信息损失定位到当前汇聚特征

## 为什么做

v185 已证明，对每个相机的 detector potential 保留完整稠密响应时，五相机与九相机都有足够容量。v186.1 把它压缩为固定 DCT12 + 报告射线 Plucker 池化特征，再用一套跨几何共享线性映射预测，结果未通过。

v187.1 专门回答失败归因：究竟是“一套共享回归器太弱”，还是“汇聚后的特征本身已丢信息”？

## 怎么检验

保持 v186.1 的数据、13 套报告标定、4 个已开封时间层、五/九相机两臂、DCT12 + Plucker 特征、K0/K1、六个绝对误差门和调用账不变。唯一改动是：

1. 不再从 1,010 个 fit 场学一套共享权重；
2. 对每个固定相机集合与报告标定，直接构造 `1009x1144` 的响应特征矩阵；
3. 用结果前冻结的 Moore-Penrose 伪逆，门限固定为最大奇异值乘 `sqrt(float64 epsilon)`；
4. 不加 ridge、阻尼、回退、候选搜索或真值调参；
5. 直接评分 K0，再跑一次未修改物理 CGLS K1。

这是一条结果后已开封的表示容量诊断，不是部署算法。

## 独立复算后的结果

| primary arm | field p90 | gradient p90 | observation p90 | 严格通过 | 完整标定 | 完整时间层 |
|---|---:|---:|---:|---:|---:|---:|
| 五相机 K0 | 0.391476 | 0.690432 | 0.467224 | 0/52 | 0/13 | 0/4 |
| 九相机 K0 | 2.703148 | 5.156636 | 3.500012 | 0/52 | 0/13 | 0/4 |
| 五相机 K1 | 0.365208 | 0.620812 | 0.241597 | 2/52 | 0/13 | 0/4 |
| 九相机 K1 | 2.378947 | 4.577949 | 1.792774 | 0/52 | 0/13 | 0/4 |

五相机 K1 的 field 和 gradient 尾部仍在冻结门内，但 observation p90 为 `0.241597`，高于 `0.20`；且四个时间层都没有完整通过。这说明去掉共享回归器后，五相机的观测缺口仍然存在。

九相机结果更具决定性：field、gradient 和 observation 三项都大幅失败。26 个 setup-local 伪逆的保留秩只有 `715-1001`，条件数约为 `4.14e5-6.65e7`，暴露出当前汇聚特征空间的严重信息损失或病态性。

完全独立的第二实现没有导入正式 v187 求解器，用不同 LAPACK SVD driver 重建特征响应、伪逆、坐标、候选场、物理 K1、指标、分层尾部与调用账。`40/40` 项检查全真；候选场相对差与指标绝对差最大为 `2.83e-9 / 4.43e-9`，所有离散判决一致。相机换序差和 held-out truth mutation 影响均为 `0`。两个实现仍共用冻结物理 kernel，因此不声称端到端物理独立已证明。

## 科学结论

正式判决为 `FAIL_GEOMETRY_LOCAL_FEATURE_CAPACITY_V187_1`。

这一轮排除了“v186.1 只是被共享跨几何回归器限制”的解释。在同一 DCT12 + 报告射线 Plucker 汇聚特征上，即使每套几何都用自己的精确固定门伪逆，仍无法恢复 v185 的匹配精度。因此当前汇聚特征图本身关闭，包括共享线性和 setup-local 两种逆。

这不推翻 v185 的稠密 camera-resolved 势域容量，也不证明所有紧凑表示都不可能。下一个合格问题只是另行冻结的“camera-resolved 与 pooled 频谱容量”归因诊断，用来区分是跨相机池化丢失，还是 DCT12 频谱截断丢失。

当前不调 SVD 门、不事后加 ridge，不用 CNN/FNO/UNO/DeepONet 或 GPU 挽救该表示。虽然 K1 逻辑在线账为 `2A+1A^T`，但精度门失败，且该诊断仍需 26 个稠密响应矩阵，所以没有 exact-call 减少、wall/RSS、外部泛化或真实 BOST 结论。`algorithm_breakthrough=false`。

# v187.1: removing the shared fit still fails, locating the loss in the pooled feature map

## Why this test was needed

v185 shows that retaining the complete dense camera-resolved detector-potential responses provides sufficient capacity under both five and nine cameras. v186.1 compresses those responses into fixed DCT12 plus reported-ray Plucker pooled features and fits one shared cross-geometry linear map; it fails the complete gate.

v187.1 isolates the cause: was one shared regressor too restrictive, or did the pooled feature map itself discard essential information?

## Frozen diagnostic

All v186.1 data, 13 reported calibrations, four opened times, five/all-nine sensor arms, DCT12 plus Plucker features, K0/K1 replay, six absolute gates, and call accounting remain unchanged. The only change is to remove the fit-data shared map and apply one setup-local fixed-threshold Moore-Penrose inverse to each `1009x1144` response-feature matrix. The threshold is the largest singular value times `sqrt(float64 epsilon)`. There is no ridge, damping, fallback, candidate search, or truth-based tuning.

## Independently recomputed result

| primary arm | field p90 | gradient p90 | observation p90 | strict-safe | complete calibrations | complete times |
|---|---:|---:|---:|---:|---:|---:|
| Five-camera K0 | 0.391476 | 0.690432 | 0.467224 | 0/52 | 0/13 | 0/4 |
| All-nine K0 | 2.703148 | 5.156636 | 3.500012 | 0/52 | 0/13 | 0/4 |
| Five-camera K1 | 0.365208 | 0.620812 | 0.241597 | 2/52 | 0/13 | 0/4 |
| All-nine K1 | 2.378947 | 4.577949 | 1.792774 | 0/52 | 0/13 | 0/4 |

Five-camera K1 keeps field and gradient tails inside their frozen limits, but observation p90 remains above `0.20`, and no time stratum passes completely. The all-nine setup-local inverse fails field, gradient, and observation by wide margins. Retained ranks range from `715` to `1001`, while condition numbers range from roughly `4.14e5` to `6.65e7`, exposing severe information loss or ill-conditioning in the pooled feature space.

A fully independent second implementation uses a different LAPACK SVD driver and rebuilds response features, pseudoinverses, coordinates, candidate fields, physical K1, metrics, stratum tails, and call ledgers without importing the formal v187 solver. All `40/40` checks pass. Maximum candidate-field relative and metric absolute differences are `2.83e-9` and `4.43e-9`; all discrete decisions agree. Camera reordering and heldout-truth mutation each have zero effect. Both implementations still share frozen physics kernels, so end-to-end physics independence is not claimed.

## Scientific conclusion

The formal decision is `FAIL_GEOMETRY_LOCAL_FEATURE_CAPACITY_V187_1`.

The shared cross-geometry regression is not the sole explanation for v186.1. Even an exact fixed-threshold inverse for each geometry cannot recover v185-level matched accuracy from the current DCT12 plus reported-ray Plucker pooled features. Both shared and setup-local inverses on this feature map are therefore closed.

The dense camera-resolved v185 capacity result remains valid, and this is not a proof against every compact representation or the overall C route. The next eligible diagnostic must be frozen separately and compare camera-resolved versus pooled spectral capacity, distinguishing camera pooling loss from DCT12 truncation. No SVD-threshold retuning, post-result ridge, larger neural model, or GPU rescue is authorized. The failed accuracy gate and dense setup matrices establish no exact-call reduction, wall/RSS benefit, external generalization, or real-BOST result. `algorithm_breakthrough=false`.
