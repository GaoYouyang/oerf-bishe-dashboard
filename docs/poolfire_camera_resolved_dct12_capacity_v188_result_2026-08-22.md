# v188：逐相机保留 DCT12 后仍失败，池化不是唯一瓶颈

## 为什么做

v187.1 已经排除“只是共享跨几何回归器太弱”：即使每套标定单独求固定门伪逆，汇聚后的 DCT12 + Plucker 特征仍不能通过完整容量门。

v188 专门拆开最后一个歧义：失败主要来自跨相机池化，还是每台相机只保留 DCT12 低频本身就不够？

## 怎么检验

保持同一条已开封 PoolFire p22 轨迹、4 个时间层、13 套报告标定、五/九相机两臂、1009 维仿射场空间、K0/K1、六个绝对误差门与调用账不变。唯一变化是取消跨相机池化：

1. 每台 active camera 的零均值 detector potential 独立放回固定探测器网格；
2. 每台相机分别做正交二维 DCT，保留 `12x12` 方块并去掉 DC，得到 `143` 个系数；
3. 按 canonical camera ID 排序后拼接，五相机和九相机分别得到 `715` 与 `1287` 维特征；
4. 对每套标定分别使用固定门 Moore-Penrose 伪逆；正式与独立实现分别使用不同 LAPACK SVD driver；
5. 不加 ridge、阻尼、回退、候选搜索、真值调参或可训练参数。

这是已开封数据上的表示容量归因，不是部署算法。

## 独立复算后的结果

| arm | field p90 | gradient p90 | observation p90 | 严格通过 | 完整标定 | 完整时间层 |
|---|---:|---:|---:|---:|---:|---:|
| 五相机 K0 | 0.391476 | 0.690432 | 0.467224 | 0/52 | 0/13 | 0/4 |
| 九相机 K0 | 0.917717 | 1.575271 | 1.165587 | 0/52 | 0/13 | 0/4 |
| 五相机 K1 | 0.365208 | 0.620812 | 0.241597 | 2/52 | 0/13 | 0/4 |
| 九相机 K1 | 0.797161 | 1.353802 | 0.594341 | 0/52 | 0/13 | 0/4 |

五相机 K1 与 v187.1 的汇聚版本在数值上几乎完全相同：field、gradient 和 observation p90 的相对变化都低于 `3.2e-11`。场和梯度通过，但 observation p90 仍是 `0.241597`，高于冻结的 `0.20`。

九相机确实明显改善。相对 v187.1，field、gradient 和 observation p90 分别下降约 `66.5% / 70.4% / 66.8%`；条件数最大值也从约 `6.65e7` 降到 `4.33e4`。但改善后仍为 `0.797161 / 1.353802 / 0.594341`，三项都越过冻结门，严格通过仍是 `0/52`。

独立第二实现重新构造逐相机 DCT、矩形伪逆、三维候选、未修改物理 K1、指标、时间/标定尾部和调用账，`44/44` 项检查全真。候选场最大相对差为 `5.53e-11`，指标最大绝对差为 `1.37e-11`，相机换序对响应和观测特征的影响均为 `0`。

第一次独立验证在正式科学数组读取和独立评分前，因独立 scalar operator 的元数据字段名接错而停止，失败证据已保留。随后只冻结并修复该元数据适配，DCT、伪逆、门、K0/K1 和判决均未变化。这个修复属于工程完整性，不是科学增量。

## 科学结论

正式判决为 `FAIL_CAMERA_RESOLVED_DCT12_CAPACITY_V188`。

跨相机池化确实是九相机病态性的重要来源，但不是唯一瓶颈：去掉池化后，九相机大幅改善却仍全线失败，五相机则没有实质变化。因此当前 DCT12 截断在 pooled 与 camera-resolved 两种形式下都关闭，不事后调奇异值门、不加 ridge 或阻尼，也不用大网络和 GPU 挽救。

这不推翻 v185 的稠密 camera-resolved 势域容量，也不证明所有紧凑表示都不可能。下一条合格问题只能另行冻结稠密逐相机 detector-potential 参考，检验剩余损失究竟是否就是 DCT12 截断，还是更深的仿射逆限制。

虽然 K1 逻辑账为 `2A+1A^T`，精度门失败，且诊断依赖 setup-local 响应矩阵；因此没有 exact-call 减少、wall/RSS、外部泛化、曲线光路或真实 BOST 结论。`algorithm_breakthrough=false`。

# v188: camera-resolved DCT12 still fails, so pooling is not the sole bottleneck

## Why this test was needed

v187.1 rejects the explanation that only one shared cross-geometry regressor was too weak. Even a separate fixed-threshold inverse for every calibration cannot pass the complete capacity gate on pooled DCT12 plus Plucker features.

v188 separates the remaining ambiguity: is cross-camera pooling the dominant loss, or is retaining only DCT12 content per camera already insufficient?

## Frozen diagnostic

The opened PoolFire p22 trajectory, four times, 13 reported calibrations, five/all-nine sensor arms, 1009-dimensional affine field space, K0/K1 replay, six absolute gates, and call accounting remain unchanged. The only change is to remove cross-camera pooling.

Each active camera's zero-mean detector potential is placed on its frozen detector grid, transformed by an orthonormal two-dimensional DCT, cropped to the leading `12x12` square, and stripped of its DC coefficient. The resulting `143` coefficients per camera are concatenated in canonical camera-ID order, producing `715` and `1287` features for five and nine cameras. Every setup uses its own fixed-threshold Moore-Penrose inverse. Formal and independent implementations use different LAPACK SVD drivers. There is no ridge, damping, fallback, candidate search, truth-based tuning, or trainable parameter.

## Independently recomputed result

| arm | field p90 | gradient p90 | observation p90 | strict-safe | complete calibrations | complete times |
|---|---:|---:|---:|---:|---:|---:|
| Five-camera K0 | 0.391476 | 0.690432 | 0.467224 | 0/52 | 0/13 | 0/4 |
| All-nine K0 | 0.917717 | 1.575271 | 1.165587 | 0/52 | 0/13 | 0/4 |
| Five-camera K1 | 0.365208 | 0.620812 | 0.241597 | 2/52 | 0/13 | 0/4 |
| All-nine K1 | 0.797161 | 1.353802 | 0.594341 | 0/52 | 0/13 | 0/4 |

Five-camera K1 is numerically unchanged from the pooled v187.1 result: relative changes in all three p90 metrics are below `3.2e-11`. Field and gradient pass, but observation p90 remains `0.241597`, above the frozen `0.20` limit.

All-nine improves substantially. Relative to v187.1, field, gradient, and observation p90 decrease by about `66.5% / 70.4% / 66.8%`, and the maximum condition number falls from about `6.65e7` to `4.33e4`. Yet the resulting `0.797161 / 1.353802 / 0.594341` still fail all three frozen p90 limits, with `0/52` strict-safe cells.

The independent second implementation rebuilds per-camera DCT features, rectangular pseudoinverses, 3D candidates, unchanged physical K1, metrics, calibration/time tails, and call accounting. All `44/44` checks pass. Maximum candidate-field relative and metric absolute differences are `5.53e-11` and `1.37e-11`; camera reordering changes neither response nor observation features.

The first independent-validation attempt stopped before reading formal scientific arrays or constructing independent scores because the independent scalar operator's metadata field was addressed by the wrong name; the failure evidence was preserved. A narrowly frozen metadata adapter repair changed no DCT, inverse, gate, K0/K1 replay, or decision. This repair is engineering evidence, not a scientific gain.

## Scientific conclusion

The formal decision is `FAIL_CAMERA_RESOLVED_DCT12_CAPACITY_V188`.

Cross-camera pooling is a major source of all-nine ill-conditioning, but it is not the sole bottleneck. Removing pooling greatly improves all-nine results without passing any complete gate and leaves the five-camera result unchanged. DCT12 truncation is therefore closed in both pooled and camera-resolved forms under this diagnostic, without post-result singular-threshold tuning, ridge, damping, larger-network rescue, or GPU rental.

The dense camera-resolved v185 potential-domain capacity remains valid, and this is not a proof against every compact representation. A separately frozen dense per-camera detector-potential reference may next test whether the remaining loss is specifically DCT12 truncation or a deeper affine-inverse limitation.

The logical K1 ledger is `2A+1A^T`, but the accuracy gate fails and the diagnostic requires setup-local response matrices. No exact-call reduction, wall/RSS benefit, external generalization, curved-ray validation, or real-BOST result is established. `algorithm_breakthrough=false`.
