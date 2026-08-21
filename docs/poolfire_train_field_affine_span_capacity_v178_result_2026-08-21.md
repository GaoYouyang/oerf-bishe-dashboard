# v178：训练场仿射张成空间通过容量门，但近满秩限制仍然关键

更新：2026-08-21

## 结论

v177 已经把失败定位到低深度场参考，但还不能区分两种解释：现有 PoolFire 训练族是否根本没有能覆盖目标场的线性表示，还是这种表示存在、却尚不能从部署可见观测中选出来。v178 对这个问题做了一次真值可见、结果后开的表示容量诊断。

我们把已经开封的 `10` 条 PoolFire 训练轨迹、共 `1,010` 个三维场组成一个仿射张成空间；数值稳定秩为 `1,009`。随后在 v177 已开封的 `13` 套标定 × `4` 帧上，分别检查冻结的五相机传感器和九相机对照：先把每个目标场投影到训练场仿射空间，再选择是否接一轮完全未修改的 CGLS K1。静态训练场均值作为便宜对照。

独立复算后的结果是：

- 仿射投影 K0 在五相机下通过 `52/52` 单元、`13/13` 标定和 `4/4` 帧；
- 仿射投影 + 未修改 CGLS K1 同样通过 `52/52`、`13/13` 和 `4/4`；
- 九相机下的两个仿射臂也都通过 `52/52`；
- 静态训练场均值 K0 与 K1 在五相机下均为 `0/52`。

正式科学判决是：

`PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178`

这是一条真实的机制容量正结果：现有训练场的线性张成空间确实包含能满足冻结 field / gradient / observation 门的见证。它排除了“v177 是因为训练场族缺少线性场表示容量”这个解释。

## 五相机结果

仿射投影 K0 的 field / gradient / observation p90 为：

`0.142673 / 0.265479 / 0.190757`

对应 worst 为：

`0.142673 / 0.265479 / 0.196129`

接一轮未修改 CGLS K1 后，p90 进一步变为：

`0.129631 / 0.233712 / 0.099735`

对应 worst 为：

`0.130176 / 0.235891 / 0.104122`

该 K1 候选的逻辑在线账为 `2A+1A^T`。但这里的仿射坐标是利用目标三维真值离线求出的，因此这个调用账不能写成已经实现的部署节省。

静态均值对照清楚地区分了“空间容量”与“固定先验”。均值 K0 的 p90 是 `0.717625 / 0.962843 / 0.882708`，均值 K1 是 `0.649488 / 0.813888 / 0.464839`，两者都为 `0/52`。所以正结果不是因为所有目标场都靠近一个固定平均场，而是因为不同目标需要不同的仿射坐标。

## 最重要的限制：1009/1010 近满秩

稳定仿射秩是 `1009/1010`，几乎等于样本数。这意味着 v178 证明的是“训练样本张成空间足够大”，而不是“发现了一个紧凑、容易学习的低维流形”。

更具体地说，v178 没有证明部署时能只看二维观测和相机几何预测 `1,009` 个坐标，也没有拟合 selector、predictor 或神经网络。真值只用于离线构造容量见证；因此当前不能把它称为 warm initializer、算子学习、泛化或重建算法。

## 独立复算

完全独立第二实现使用不同的稳定 SVD 路径，重新构造训练场仿射空间、投影、五/九相机 forward 与 adjoint、K1 候选、逐单元指标、分层门、调用账和相机乱序检查。`26/26` 项检查全部通过：

- 保留奇异值最大相对差 `1.25e-15`；
- 投影场最大相对差 `4.21e-15`；
- 候选场最大相对差 `9.28e-13`；
- 指标最大绝对差 `4.48e-12`；
- 仿射坐标最大绝对差 `2.11e-13`；
- 相机乱序最大相对差 `1.15e-16`；
- 所有离散通过、失败和最终判决完全一致。

第一次独立验证曾因把物理源时间直接与归一化时间标签比较而 fail-closed；该 inconclusive 证据被保留。随后只修复验证器的时间归一化检查，没有改变协议、正式数组、候选、阈值、指标或判决规则。

## 路线动作与证据边界

下一门不是直接训练大模型，而是另行结果前冻结一个最小的 observation + geometry-only 仿射坐标可预测性诊断，并采用完整轨迹隔离与便宜确定性对照。只有这个严格可预测性门通过，才有理由讨论一个真正可部署的 warm initializer。

v178 不是紧凑表示、部署算法、exact-call 减少、资源加速、外部泛化、curved ray、真实 BOST、论文成功或算法突破：

`algorithm_breakthrough=false`、`paper_success=false`、`broad_external_generalization=false`、`resource_speedup=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v178: the training-field affine span passes the capacity gate, with a near-full-rank limitation

Updated: 2026-08-21

v177 localizes the failure to the low-depth field reference but leaves two explanations: the opened PoolFire training family may lack any linear field representation that covers the target fields, or such a representation may exist but remain unpredictable from deployment-visible observations. v178 performs a truth-aware, post-open representation-capacity diagnostic to separate these explanations.

The ten already opened PoolFire training trajectories contribute `1,010` three-dimensional fields. Their affine span has stable rank `1,009`. On the thirteen calibrations and four already opened v177 frames, each target field is projected into this affine span. The diagnostic then evaluates the frozen five-camera sensor and an all-nine sensor control, before and after one completely unchanged CGLS K1 step. The static fit mean is the cheap control.

After independent recomputation, affine projection K0 passes `52/52` cells, `13/13` calibrations, and `4/4` frame strata under five cameras. Affine projection followed by unchanged CGLS K1 also passes `52/52`, `13/13`, and `4/4`. Both affine arms pass `52/52` under all nine cameras. The five-camera static-mean K0 and K1 controls both remain `0/52`.

Decision: `PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178`.

This is a genuine positive mechanism-capacity increment. The linear span of the existing training fields contains witnesses that satisfy the frozen field, gradient, and observation gates. It rules out missing linear field-span capacity as the explanation for v177.

For five-camera affine projection K0, field / gradient / observation p90 values are `0.142673 / 0.265479 / 0.190757`, with worst values `0.142673 / 0.265479 / 0.196129`. After unchanged CGLS K1, the p90 values improve to `0.129631 / 0.233712 / 0.099735`, with worst values `0.130176 / 0.235891 / 0.104122`. The K1 arm has a logical online ledger of `2A+1A^T`, but its affine coordinates are computed offline from the target 3D truth, so this ledger is not an established deployment saving.

The static-mean controls distinguish span capacity from a fixed prior. Mean K0 has p90 `0.717625 / 0.962843 / 0.882708`, while mean K1 has `0.649488 / 0.813888 / 0.464839`; both are `0/52`. Different targets require different affine coordinates.

The decisive limitation is the stable rank: `1,009/1,010`, almost the full sample count. v178 therefore establishes that the sample span is sufficiently large, not that a compact or readily learnable low-dimensional manifold has been found. It does not show that a deployed system can infer 1,009 coordinates from 2D observations and reported geometry, and it fits no selector, predictor, or neural network.

A fully independent second implementation uses a different stable SVD path and rebuilds the affine span, projections, five- and nine-camera operators, K1 candidates, cell metrics, strata gates, call ledgers, and camera-order checks. All `26/26` checks pass. Maximum retained-singular-value, projected-field, candidate-field, metric, coefficient, and camera-permutation differences are `1.25e-15`, `4.21e-15`, `9.28e-13`, `4.48e-12`, `2.11e-13`, and `1.15e-16`; every discrete decision agrees.

The first independent validation failed closed because it compared physical source times directly with normalized time labels. That inconclusive attempt remains preserved. The repair changed only the validator's normalization check; it changed no protocol, formal array, arm, threshold, metric, or decision rule.

The next gate is a separately preregistered minimal observation-and-geometry-only predictability diagnostic for the affine coordinates, with complete-trajectory isolation and cheap deterministic controls. Neural training, GPU rental, resource tests, and untouched-test opening remain unauthorized.

v178 is not a compact representation, deployable algorithm, exact-call reduction, resource speedup, external generalization, curved-ray validation, real BOST, paper success, or an algorithm breakthrough: `algorithm_breakthrough=false`, `paper_success=false`, `broad_external_generalization=false`, `resource_speedup=false`, `curved_ray_validated=false`, `real_bost=false`.
