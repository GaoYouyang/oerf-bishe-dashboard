# v189：完整逐相机 DCT 恢复容量，v188 的根因是 DCT12 截断

## 为什么做

v188 已把跨相机池化拆掉，但每台相机只保留 `12x12` 低频方块中的 `143` 个非 DC 系数。五相机 K1 只有 `2/52`，九相机是 `0/52`。当时还剩一个关键歧义：失败究竟来自 DCT12 丢掉高频信息，还是即便保留完整逐相机势场，固定仿射逆本身也不够？

v189 只回答这个归因问题，不训练模型，也不声称部署性能。

## 怎么检验

保持同一条已开封 PoolFire p22 轨迹、4 个时间层、13 套报告标定、五/九相机两臂、1009 维仿射场空间、K0/K1、六个绝对误差门和调用账不变。唯一变化是把每台相机的 detector-potential 表示从截断 DCT12 扩展为完整 `24x24` 正交二维 DCT，并只去掉 DC：

1. 每台相机保留 `575` 个非 DC 系数；
2. 五相机与九相机分别得到 `2875` 与 `5175` 维特征；
3. 仍按 canonical camera ID 拼接，并为每套标定使用固定门 Moore-Penrose 逆；
4. 不加 ridge、阻尼、回退、候选搜索、真值调参或可训练参数；
5. 额外要求与 v185 的稠密逐相机势域解逐单元数值等价。

这是已开封数据上的容量归因。它依赖 setup-local 稠密响应矩阵，不是可部署紧凑算法。

## 独立复算后的结果

| arm | field p90 | gradient p90 | observation p90 | 严格通过 | 完整标定 | 完整时间层 |
|---|---:|---:|---:|---:|---:|---:|
| 五相机 K0 | 0.343477 | 0.562938 | 0.186872 | 50/52 | 11/13 | 3/4 |
| 九相机 K0 | 0.249360 | 0.445739 | 0.180337 | 52/52 | 13/13 | 4/4 |
| 五相机 K1 | 0.338439 | 0.549518 | 0.118081 | 52/52 | 13/13 | 4/4 |
| 九相机 K1 | 0.240014 | 0.409766 | 0.116577 | 52/52 | 13/13 | 4/4 |

关键变化发生在未修改的物理 K1 后：五相机从 v188 的 `2/52` 恢复到 `52/52`，九相机从 `0/52` 恢复到 `52/52`；13 套标定和 4 个时间层在两臂都完整通过。

这不是一个“换了更宽松门”的成功。误差门、数据、标定、K1 和求逆规则都没有变化；变化只是恢复了 v188 截掉的探测器频率。保留秩在五/九相机下都达到 `1009`，条件数范围约为 `48.93–187.41`。

v189 与 v185 稠密势域参考也数值一致：候选场逐单元最大相对差为 `1.62e-14`，坐标为 `2.24e-14`，奇异值为 `4.44e-15`，指标最大绝对差为 `3.22e-15`。因此这不是碰巧过门，而是同一稠密容量的另一种正交坐标表达。

完全独立第二实现重建完整 DCT、响应矩阵、伪逆、候选场、物理 K1、指标、分层与调用账，`50/50` 项检查全真。正式与独立实现之间，候选场最大相对差为 `7.47e-12`，指标最大绝对差为 `1.30e-12`；Parseval 和 DCT 往返误差都低于 `1e-15`。相机换序对特征、坐标和候选场的影响均为 `0`，固定观测下修改真值也不会改变预测。

正式启动前两次工程失败分别发生在模块导入和输入路径核验阶段，均未生成候选或指标；失败证据保留，之后只修复启动环境和路径，不改科学合同。这些是工程完整性，不是算法成果。

## 科学结论

正式判决为 `PASS_DCT12_TRUNCATION_ROOT_CAUSE_V189`。

在这条冻结、已开封、setup-local 仿射容量诊断下，恢复完整逐相机 detector-potential 频谱后，五/九相机 K1 都达到 `52/52`，并复现 v185 稠密解。因此 v188 的失败可归因于 DCT12 频谱截断，而不是更深的固定仿射逆容量不足。

这条正结果只回答“信息是否存在”。`575` 个系数/相机和稠密 setup-local 响应矩阵并不紧凑，也没有 observation/geometry-only 预测器、exact-call 减少、fresh wall/RSS、外部工况、曲线光路或真实 BOST 证据。下一条合格问题是另行结果前冻结一个保留关键高频、相机换序等变、可变相机数且只读部署可见输入的紧凑表示，再先做容量门。

`algorithm_breakthrough=false`，`paper_success=false`，GPU 与神经训练仍未授权。

# v189: full per-camera DCT restores capacity and attributes v188 to DCT12 truncation

## Why this test was needed

v188 removed cross-camera pooling but retained only `143` non-DC coefficients from each camera's leading `12x12` low-frequency DCT square. K1 reached only `2/52` under five cameras and `0/52` under all nine. One ambiguity remained: did DCT12 discard essential high-frequency information, or would the frozen affine inverse remain insufficient even with complete per-camera potentials?

v189 answers only this attribution question. It trains no model and establishes no deployment performance.

## Frozen diagnostic

The opened PoolFire p22 trajectory, four times, 13 reported calibrations, five/all-nine sensor arms, 1009-dimensional affine field space, K0/K1 replay, six absolute gates, and call accounting remain unchanged. The only change is to replace truncated DCT12 with the complete orthonormal `24x24` per-camera DCT, excluding only DC.

Each camera retains `575` coefficients, yielding `2875` and `5175` features under five and nine cameras. Features are concatenated in canonical camera-ID order, and each setup uses the same fixed-threshold Moore-Penrose inverse. There is no ridge, damping, fallback, candidate search, truth-based tuning, or trainable parameter. A separate gate also requires cellwise numerical equivalence to the dense v185 potential-domain result.

## Independently recomputed result

| arm | field p90 | gradient p90 | observation p90 | strict-safe | complete calibrations | complete times |
|---|---:|---:|---:|---:|---:|---:|
| Five-camera K0 | 0.343477 | 0.562938 | 0.186872 | 50/52 | 11/13 | 3/4 |
| All-nine K0 | 0.249360 | 0.445739 | 0.180337 | 52/52 | 13/13 | 4/4 |
| Five-camera K1 | 0.338439 | 0.549518 | 0.118081 | 52/52 | 13/13 | 4/4 |
| All-nine K1 | 0.240014 | 0.409766 | 0.116577 | 52/52 | 13/13 | 4/4 |

After one unchanged physical K1 step, five-camera rises from v188's `2/52` to `52/52`, while all-nine rises from `0/52` to `52/52`. All 13 calibrations and all four time strata pass in both arms.

The gates, data, calibration roster, K1 replay, and inverse rule are unchanged; only the omitted detector frequencies are restored. Retained rank is `1009` in both arms, with condition numbers ranging from about `48.93` to `187.41`.

v189 is also numerically equivalent to the dense v185 reference. Maximum cellwise relative differences are `1.62e-14` for candidate fields, `2.24e-14` for coordinates, and `4.44e-15` for singular values; the maximum metric absolute difference is `3.22e-15`. This is the same dense capacity in another orthonormal coordinate system, not an accidental threshold pass.

A fully independent second implementation rebuilds the full DCT, response matrices, pseudoinverses, candidate fields, physical K1 replay, metrics, strata, and call ledger. All `50/50` checks pass. Maximum formal-versus-independent candidate-field relative and metric absolute differences are `7.47e-12` and `1.30e-12`. Parseval and DCT round-trip errors are below `1e-15`. Camera reordering changes features, coordinates, and fields by exactly `0`, and mutating truth under fixed observations does not change predictions.

Two pre-execution engineering failures occurred during module import and input-path verification, before any candidate or metric was generated. Their evidence was preserved, and only startup environment and path handling were repaired. These repairs are engineering assurance, not scientific gains.

## Scientific conclusion

The decision is `PASS_DCT12_TRUNCATION_ROOT_CAUSE_V189`.

Under this frozen, opened-data, setup-local affine capacity diagnostic, restoring the full per-camera detector-potential spectrum gives `52/52` K1 cells in both sensor arms and reproduces the dense v185 solution. The v188 failure is therefore attributable to DCT12 spectral truncation rather than a deeper capacity limit of the frozen affine inverse.

This positive result establishes only that the information exists. `575` coefficients per camera and dense setup-local response matrices are not compact. No observation/geometry-only predictor, exact-call reduction, fresh wall/RSS benefit, external condition, curved-ray validation, or real-BOST evidence has been established. The next eligible question is a separately preregistered compact, high-frequency-preserving, camera-permutation-equivariant, variable-cardinality representation using deployment-visible inputs only, beginning with another capacity gate.

`algorithm_breakthrough=false`, `paper_success=false`, and neither GPU rental nor neural training is authorized.
