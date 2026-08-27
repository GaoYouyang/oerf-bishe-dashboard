# v272：v271 两个解之间是单向支配，不足以判成共同欠收敛

## 做了什么

v272 不重跑 Haar-IRLS，也不生成新重建。它只读取 v271 已封存的两组系数、算子观测和各自最终平滑参数，在每套 rig 上检查两端点的连接线。13 条线的算子观测最大相对差为 `5.20e-16`，因此在冻结的 `1e-12` 数值口径下都属于观测零方向；新增调用为 `0A+0A^T`，没有读取密度真值、三维场或四项评分。

## 独立复算与判决

正式实现采用 NumPy 顺序归约；完全独立的第二实现反向逐项使用 `math.fsum`，自行执行固定 80 次二分。独立验证通过 `14/14` 项，连续量最大绝对差 `1.42e-14`、最大相对差 `2.32e-16`，离散判决完全一致。

结果具有明确的单向性：独立端点在 `13/13` 条线上都能向正式端点严格降低自身固定目标，最小相对下降为 `1.218e-4`；但正式端点在预注册的线段内部为 `0/13` 严格下降，线段最小点全部停在 `alpha=1`。因此不能声称“两套实现都欠收敛”，也不能把差异直接解释成结构性非唯一。权威判决是 `MIXED_OR_NEAR_FLAT_CASE19_HAAR_IRLS_NULL_LINE_V272`。

## 证据边界

v272 收紧了根因描述，但没有修复 reference。固定一层 Haar-IRLS 仍关闭：不重跑、不放宽门、不改轮数、平滑参数、solver 或正则，也不从这些系数训练模型。它没有建立 warm initializer、matched-accuracy、有效减调用、完整序列、wall/RSS、外部泛化或真实 BOST，`algorithm_breakthrough=false`。

# v272: The two v271 endpoints show one-way dominance, not common underoptimization

## What was tested

v272 neither reruns Haar-IRLS nor creates a new reconstruction. It reads only the two sealed v271 coefficient arrays, operator observations, and final smoothing values, then audits the segment joining the endpoints. The maximum relative operator-observation difference across all thirteen lines is `5.20e-16`, making every line numerically observation-null under the frozen `1e-12` criterion. The diagnostic adds `0A+0AT` and reads no density truth, reconstructed field, or accuracy metric.

## Independent recomputation and verdict

The formal implementation uses vectorized NumPy reductions. A separate implementation traverses coefficients in reverse with `math.fsum` and performs its own fixed eighty-step bisection. Independent validation passes `14/14` checks; the maximum absolute scalar difference is `1.42e-14`, the maximum relative difference `2.32e-16`, and every discrete decision matches exactly.

The outcome is one-sided. On `13/13` lines, the independent endpoint strictly lowers its own frozen objective toward the formal endpoint, with minimum relative descent `1.218e-4`. No formal endpoint has preregistered inward descent on the tested segment (`0/13`); every segment minimum lies at `alpha=1`. The evidence therefore does not support common underoptimization and cannot identify structural nonuniqueness. The authoritative verdict is `MIXED_OR_NEAR_FLAT_CASE19_HAAR_IRLS_NULL_LINE_V272`.

## Evidence boundary

v272 narrows the root-cause statement but does not repair the reference. The fixed one-level Haar-IRLS route remains closed without rerun, gate relaxation, or tuning of rounds, smoothing, solver, or regularization, and these coefficients cannot seed a predictor. It establishes no warm initializer, matched accuracy, effective call reduction, full sequence, wall/RSS, external generalization, or real BOST. `algorithm_breakthrough=false`.
