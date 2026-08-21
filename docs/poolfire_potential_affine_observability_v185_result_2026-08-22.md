# v185：同一势域坐标保住完整仿射信息，K1 在五相机与九相机均 52/52

## 做了什么

v184 已经证明二维 BOS 残差大多可积，但把势场经一条 scalar-ray Jacobi 方向提升回三维会丢掉关键的 field compatibility。v185 不再猜一条三维方向，而是检验一个更精确的问题：**如果观测与仿射基的每一列都经过完全相同的零均值 detector-potential 变换，势域是否仍保留求解三维仿射坐标所需的全部可观测信息？**

冻结机制为：

1. 对当前中心化多视角观测做零均值势场积分；
2. 对冻结的 1,009 个仿射基 forward-response 列逐列做同一势场积分；
3. 用固定相对奇异值门构造势域仿射伪逆，直接恢复 1,009 维坐标；
4. 比较直接暖场 K0，以及再运行一次完全未修改的物理 CGLS K1。

预测接口只读取当前观测、有效相机 ID、报告几何和 fit-only 仿射基；没有真值输入、训练、候选搜索、ridge、阻尼、裁剪或回退。

## 独立复算后的结果

势域映射在所有单元保留的秩都精确为 `1009/1009`。九相机 K0 已经完整通过；五相机 K0 为 `50/52`，只在两套标定的 observation p90 上轻微越过 `0.20` 门，分别是 `0.203064` 和 `0.206737`。

一轮未修改物理 CGLS K1 后，两档相机全部通过：

| arm | field p90 | gradient p90 | observation p90 | 严格通过 | 完整标定 |
|---|---:|---:|---:|---:|---:|
| 五相机 K1 | 0.338439 | 0.549518 | 0.118081 | 52/52 | 13/13 |
| 九相机 K1 | 0.240014 | 0.409766 | 0.116577 | 52/52 | 13/13 |

冻结 p90 门分别为 field `0.50`、gradient `0.75`、observation `0.20`；四个时间层也全部通过。作为便宜反证，一方向 potential-coordinate CGLS1 的 K0 与 K1 在五相机和九相机下仍全部是 `0/52`。因此结果不是“只要把残差积分成势场，再走一小步就行”，而是依赖观测与完整仿射响应在同一坐标系中的联合可观测结构。

完全独立第二实现重新构造势场、全部势域响应列、固定门伪逆、三维候选、物理 K1、指标、调用账与相机换序审计。`32/32` 检查全真；候选场、仿射坐标、势场和逐单元指标最大差约为 `8.40e-12 / 1.84e-11 / 1.48e-11 / 4.48e-12`，离散判决完全一致。相机换序后的 primary 场相对差为 `1.39e-14`，held-out truth mutation 对预测的最大影响为 `0`。

两次独立验证启动缺陷也原样保留：一次在读取科学数组前混淆物理时间与归一化标签，另一次在最终报告比较时把二元返回值当成标量。两次都 fail-closed，失败数组未复用；修复没有改变正式 runner，三次完整 formal 的 22 个科学数组与两个 barrier 逐字节一致。这些只属于工程完整性，不是算法成果。

## 科学结论

正式判决是 `PASS_POTENTIAL_AFFINE_K1_CAPACITY_V185`。

这次结果改变了对 v184 的归因：**失败的不是 detector-potential 压缩本身，而是把势场压成一条 scalar-ray Jacobi 三维方向的 lift。** 当观测与全部仿射响应列共享同一个势域坐标时，1,009 个可观测仿射自由度没有丢失，并足以让一轮未修改物理 K1 在已开封五相机和九相机单元上完整守门。

但 v185 仍不是部署算法。稠密势域逆每套 sensor setup 需要处理 `1,013` 个势变换右端，并继承 `26,260` 个 forward-equivalent 的几何缓存构造；逻辑在线 K1 虽为 `2A+1A^T`，当前还不能据此声称 exact-call 减少，更没有 wall/RSS、外部工况、curved ray 或真实 BOST 证据。

因此只授权下一门：结果前冻结一个**紧凑、共享参数、observation/geometry-only** 的势域逆近似，并与便宜 controls 公平比较。v185 本身只是 post-open information-capacity diagnostic；`algorithm_breakthrough=false`。

# v185: a shared potential-domain coordinate preserves the full affine state, reaching 52/52 after K1 under five and nine cameras

## What was tested

v184 showed that the 2D BOS residual is predominantly integrable, but lifting its potential into 3D through one scalar-ray Jacobi direction destroys field compatibility. v185 asks a more exact question: **if the observation and every affine-basis response column undergo the same zero-mean detector-potential transform, does the potential domain retain all information needed to solve the 3D affine coordinates?**

The frozen mechanism:

1. integrates the centered multiview observation into zero-mean detector potentials;
2. applies exactly the same integration to all 1,009 frozen affine-basis forward-response columns;
3. constructs a fixed-threshold potential-domain affine pseudoinverse and recovers all 1,009 coordinates directly;
4. compares the direct K0 warm field and one fully unchanged physical CGLS K1 step.

The prediction interface reads only the current observation, active camera IDs, reported geometry, and the fit-only affine basis. It uses no target truth, training, candidate search, ridge, damping, clipping, or fallback.

## Independently recomputed result

The potential-domain map retains rank `1009/1009` in every cell. All-nine K0 already passes completely. Five-camera K0 reaches `50/52`; only two calibration-level observation p90 values narrowly exceed the `0.20` gate, at `0.203064` and `0.206737`.

After one unchanged physical CGLS K1 step, both sensor arms pass completely:

| arm | field p90 | gradient p90 | observation p90 | strict-safe | complete calibrations |
|---|---:|---:|---:|---:|---:|
| Five-camera K1 | 0.338439 | 0.549518 | 0.118081 | 52/52 | 13/13 |
| All-nine K1 | 0.240014 | 0.409766 | 0.116577 | 52/52 | 13/13 |

The frozen p90 gates are `0.50`, `0.75`, and `0.20`, and all four time strata also pass. A cheap falsification control, one-direction potential-coordinate CGLS1, remains at `0/52` for K0 and K1 under both camera arms. The result is therefore not explained by merely integrating the residual and taking one small step; it depends on the joint observable structure of the observation and the full affine response in one common coordinate system.

A fully independent second implementation rebuilds detector potentials, every potential-domain response column, the fixed-threshold inverse, 3D candidates, physical K1 replay, metrics, call ledgers, and camera-permutation audits. All `32/32` checks pass. Maximum candidate-field, affine-coordinate, potential, and per-cell metric differences are about `8.40e-12`, `1.84e-11`, `1.48e-11`, and `4.48e-12`, with identical discrete decisions. The primary field changes by only `1.39e-14` under camera reordering, and held-out truth mutation changes predictions by exactly `0`.

Two independent-validation launch defects remain preserved: one confused physical source times with normalized labels before scientific arrays were read; the other treated a two-value report comparator as a scalar at final comparison. Both failed closed, no failed arrays were reused, and the formal runner did not change. Across three complete formal runs, all 22 scientific arrays and both barriers are byte-identical. This is engineering assurance, not an algorithmic result.

## Scientific conclusion

The formal verdict is `PASS_POTENTIAL_AFFINE_K1_CAPACITY_V185`.

This changes the attribution of v184's failure: **detector-potential compression itself is not the problem; the lossy scalar-ray Jacobi lift is.** When the observation and all affine response columns share one potential-domain coordinate system, none of the 1,009 observable affine degrees of freedom is lost, and one unchanged physical K1 step clears every opened five- and nine-camera gate.

v185 is still not a deployable algorithm. The dense potential inverse processes `1,013` transform right-hand sides per sensor setup and inherits a `26,260` forward-equivalent geometry-cache construction. Although the logical online K1 ledger is `2A+1A^T`, this does not yet establish exact-call reduction, wall/RSS benefit, external-condition transfer, curved-ray validity, or real BOST.

The result authorizes only the next separately preregistered gate: a **compact, shared-parameter, observation/geometry-only** approximation to the potential-domain inverse, compared fairly with cheap controls. v185 remains a post-open information-capacity diagnostic; `algorithm_breakthrough=false`.
