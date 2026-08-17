# v158：经典谱平滑改善缺视角重建，但五相机场尾部仍未过门

更新：2026-08-17

## 先说结论

v158 没有训练网络。它沿用 v157 的 9 个可执行三维场、13 套九相机标定、24×24 每相机观测和 DCT1024 表示，只检查一个更小的问题：能否用**部署时可见的观测残差**自动选择 H1 Tikhonov 平滑强度，在 5/7/9 相机下同时形成可信经典参考。

结果前固定的主策略是：从十二个正则倍率中选择仍满足 observation residual ≤ 0.18 的最大值。正式运行与独立第二实现完全同判决：

- 7、9 相机通过全部 field / gradient / observation 门；
- 5 相机的 gradient 和 observation 已通过，field p90 为 `0.629665`，仍高于冻结门 `0.50`；
- 因为合同要求 5/7/9 相机全部通过，科学判决为 `FAIL_SPECTRAL_SMOOTHNESS_REFERENCE_V158`。

这不是“方法完全没用”。相对 v157 的 K16 父对照，5 相机 field p90 从 `0.636964` 改善到 `0.629665`，gradient p90 从 `0.903568` 改善到 `0.692700`，但改善幅度不足以跨过绝对场误差门。

## 独立确认后的数字

| 活跃相机 | field p90 / worst | gradient p90 / worst | observation p90 / worst | 判决 |
| ---: | ---: | ---: | ---: | :--- |
| 5 | 0.630 / 0.652 | 0.693 / 0.705 | 0.175 / 0.180 | field p90 未过 |
| 7 | 0.452 / 0.532 | 0.611 / 0.661 | 0.174 / 0.180 | 全部通过 |
| 9 | 0.323 / 0.376 | 0.583 / 0.623 | 0.172 / 0.179 | 全部通过 |

冻结门为 field p90 ≤ 0.50、gradient p90 ≤ 0.75、observation p90 ≤ 0.20；worst 分别不超过 0.75、1.00、0.35。

正式运行覆盖 39 个 operator setups、351 个 cells 和 4,914 条 arm rows。独立实现不用正式 v158 求解器，重新构造解析 DCT 基、直接射线积分、约化算子、广义特征分解、可观测选参、物理场和全部指标。`18/18` 项检查通过；逐 cell / 汇总指标最大差为 `4.42e-10 / 3.54e-10`，约化 forward 与直接 forward 最大差为 `9.95e-14`。

## 为什么固定正则诊断不能改写结论

结果中，固定倍率 `0.03` 和 `0.1` 的诊断行恰好在 5/7/9 相机绝对门上全部通过。这是有价值的线索：当前瓶颈可能包含“怎样仅从观测选择正则强度”，而不只是表示容量。

但这些行在结果前被明确标成 diagnostic-only，不能替代主策略。现在把它们改成主方法会构成事后挑选。它们最多只能形成一个将来在**真正新数据**上预注册确认的假设，不能被写成 v158 成功。

## 成本与证据边界

主策略在几何缓存后记为 `1A+1A^T`，但构造 DCT1024 约化几何需要每套标定 1,023 次 basis projection，13 套共 13,299 次 forward-equivalent setup；这部分没有被隐藏成“免费”。

本轮仍没有逐工况配对的实验二维位移图，没有运行 learned predictor、fresh wall/RSS、独立公开外门或真实 BOST。当前 private variable-cardinality predictor 路线按合同关闭，等待更广三维场或配对二维实验位移；不以 CNN / FNO / UNO / DeepONet 或 GPU 挽救。

当前边界：`algorithm_breakthrough=false`、`paper_success=false`、`real_bost=false`、`predictor_training_authorized=false`、`gpu_rental_authorized=false`。

---

# v158: classical spectral smoothing helps sparse-view reconstruction, but the five-camera field tail still fails

Updated: 2026-08-17

v158 trains no network. It retains the nine executable 3D fields, thirteen nine-camera calibrations, 24×24 samples per camera, and DCT1024 representation from v157. The preregistered primary selects the largest H1 Tikhonov multiplier whose deployment-visible observation residual remains at or below 0.18.

The independent result is exact at the decision level. Seven and nine cameras pass every field, gradient, and observation gate. Five cameras pass the gradient and observation gates, but field p90 is `0.629665`, above the frozen `0.50` threshold. The all-camera decision is therefore `FAIL_SPECTRAL_SMOOTHNESS_REFERENCE_V158`.

The primary does improve the five-camera parent: field p90 moves from `0.636964` to `0.629665`, and gradient p90 from `0.903568` to `0.692700`. It still does not establish an adequate five-camera reference.

The formal run contains 39 operator setups, 351 cells, and 4,914 arm rows. A second implementation independently rebuilds the analytic DCT basis, direct ray quadrature, reduced operators, eigensystems, observable selector, physical fields, and all metrics. All `18/18` checks pass. Maximum per-cell and summary differences are `4.42e-10` and `3.54e-10`; the maximum reduced-versus-direct forward difference is `9.95e-14`.

Fixed multipliers `0.03` and `0.1` pass the absolute diagnostic gates at all camera counts, suggesting selection headroom. They were explicitly frozen as diagnostic-only and cannot replace the primary after results. They may motivate a confirmatory hypothesis on genuinely new data, not a post-hoc v158 success.

The primary is logically `1A+1A^T` after geometry caching, while the disclosed reduced-geometry setup costs 1,023 basis projections per calibration, 13,299 forward-equivalent projections in total. No paired experimental 2D displacements, learned predictor, fresh wall/RSS test, external gate, or real-BOST reconstruction is present. The current private variable-cardinality predictor route closes pending richer 3D fields or paired experimental 2D measurements; GPU rental remains unauthorized.
