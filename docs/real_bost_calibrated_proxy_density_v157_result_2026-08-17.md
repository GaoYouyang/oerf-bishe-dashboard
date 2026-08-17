# v157：九相机受控代理过参考门，五/七相机仍未过

更新：2026-08-17

## 先说结论

师兄提供的新数据让研究第一次能够用**组内三维重建场和真实相机标定**构造受控 forward，而不再只依赖公开 PoolFire 的合成几何。数据中有 9 个可直接执行的三维场和 13 套九相机标定；本轮把二者做固定交叉组合，共形成 117 个“场 × 标定”设置。由于对应的实验二维位移投影尚未提供，这 117 项不是 117 次独立真实实验，只能称为**标定驱动的受控代理**。

v156 先验证了 forward / adjoint 数值闭合，但 8×8 的每相机观测过稀。v157 因此只检查一个更基础的问题：把每相机采样提高到 16×16、24×24，并在固定三维 DCT 子空间内做谱正则化后，经典 CGLS 能否形成可信参考。

独立第二实现确认：

- 24×24、9 相机的 DCT1024-CGLS K16 通过结果前冻结的 field / gradient / observation 门；
- 同样方法在 5、7 相机下仍分别卡在 field 与 gradient 尾部；
- DCT1024 的 truth-aware oracle 容量通过，说明当前失败不能归因于“这个平滑三维表示本身装不下目标”。

科学判决是 `FAIL_REFERENCE_ADEQUACY_V157`：九相机参考条件成立，但 5/7 相机缺视角条件仍不够稳健。它是一个有用的正负分界，不是算法突破，也不授权预测器训练或租 GPU。

## 关键数字

主候选是 24×24、DCT1024-CGLS K16。冻结门为 field p90 ≤ 0.50、gradient p90 ≤ 0.75、observation p90 ≤ 0.20；worst 也必须分别不超过 0.75、1.00、0.35。

| 活跃相机 | field p90 / worst | gradient p90 / worst | observation p90 / worst | 判决 |
| ---: | ---: | ---: | ---: | :--- |
| 5 | 0.637 / 0.675 | 0.904 / 0.963 | 0.143 / 0.156 | field、gradient 未过 |
| 7 | 0.578 / 0.619 | 0.793 / 0.950 | 0.159 / 0.162 | field、gradient 未过 |
| 9 | 0.482 / 0.526 | 0.720 / 0.753 | 0.166 / 0.172 | 全部门通过 |

从 8×8 增加到 24×24 后，三种相机数的 field 与 gradient p90 都改善，因此更密观测确实有价值。另一方面，5/7 相机即使 observation residual 已经较低，三维 field 与 gradient 仍未达门，说明不能只凭拟合观测就宣称重建可靠。

DCT1024 oracle 的 field p90 / worst 为 `0.143 / 0.143`，gradient p90 / worst 为 `0.493 / 0.493`，容量门通过。这把下一步收缩为一个更具体的问题：5/7 相机的缺视角条件需要更合适、但仍固定且可审计的经典正则化，而不是直接增加学习模型。

## 独立复算与成本边界

正式运行覆盖 9 个三维场、13 套标定、3 档观测密度和 3 种活跃相机数，共 1,053 个 cells、21,060 条候选记录。独立实现从原始三维场与相机矩阵重建射线、算子、DCT 方向、经典迭代和全部指标：

- 17 项独立检查全部通过；
- 每 cell 指标最大差 `4.91e-9`；
- 汇总指标最大差 `1.82e-11`；
- 独立伴随误差最大 `1.99e-13`；
- 常量响应误差最大 `4.16e-16`；
- 离散判决完全一致。

本轮是在建立 classical reference，不是在减少调用。主候选使用 K16，不能写成 warm-start 节省、速度或资源收益。真实二维实验投影仍未配对，因此也不能写成真实 BOST 重建或外部泛化。

## 路线动作

下一步只允许一个结果前冻结的经典正则化诊断：在 24×24 下，用固定、deployment-visible 的平滑正则审计 5/7 相机缺视角条件，并保留 9 相机作正对照。若仍失败，当前 variable-cardinality 预测路线停止，等待更广三维场或配对二维实验位移；不用 CNN / FNO / UNO / DeepONet 或 GPU 挽救。

当前边界：`algorithm_breakthrough=false`、`paper_success=false`、`real_bost=false`、`predictor_training_authorized=false`、`gpu_rental_authorized=false`。

---

# v157: the nine-camera controlled proxy clears the reference gate; five and seven cameras do not

Updated: 2026-08-17

The new group data makes it possible to build a controlled forward model from reconstructed three-dimensional fields and measured camera calibration. Nine executable 3D fields are crossed with thirteen nine-camera calibration sets, producing 117 fixed field-by-calibration setups. Because condition-matched experimental 2D displacement maps are not yet available, these are not 117 independent real experiments; the evidence remains a calibration-driven controlled proxy.

v156 first verified numerical forward/adjoint closure and showed that 8×8 samples per camera were too sparse. v157 then tests observation densities of 8×8, 16×16, and 24×24 together with fixed 3D DCT spectral regularization.

At 24×24, DCT1024-CGLS K16 passes the preregistered field, gradient, and observation gates with nine cameras. The same method still fails field and gradient tails with five and seven cameras. A truth-aware DCT1024 oracle passes its capacity gate, so the negative sparse-view result is not explained by insufficient capacity of the smooth 3D representation itself.

For five / seven / nine cameras, field p90 is `0.637 / 0.578 / 0.482`, gradient p90 is `0.904 / 0.793 / 0.720`, and observation p90 is `0.143 / 0.159 / 0.166`. The frozen thresholds are `0.50 / 0.75 / 0.20`. All counts improve from 8×8 to 24×24, but only nine cameras clear every tail gate.

The formal run contains 1,053 cells and 21,060 candidate rows. An independent second implementation rebuilds rays, operators, DCT directions, classical iterations, and all metrics from the field and calibration inputs. All 17 checks pass; the maximum per-cell and summary differences are `4.91e-9` and `1.82e-11`, with maximum adjoint and constant-response errors of `1.99e-13` and `4.16e-16`.

The scientific decision is `FAIL_REFERENCE_ADEQUACY_V157`: the nine-camera reference is adequate, while sparse-view five/seven-camera conditions remain conditioning- or regularization-limited. This is not a learned algorithm, exact-call saving, resource result, external generalization, or real-BOST reconstruction. The next and only authorized gate is one fixed classical smoothness-regularization diagnostic at 24×24; no predictor, larger neural model, GPU rental, or real-BOST claim is authorized.
