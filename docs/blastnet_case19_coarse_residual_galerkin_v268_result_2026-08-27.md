# v268：固定单步全局粗空间修正未守住 K16 观测匹配门

## 为什么做

v267 表明，把局部方向同步相加会产生明显跨块干扰。v268 因此检验一个物理上不同、仍然可证伪的确定性机制：从已封存的 v258 父状态出发，把完整观测残差的精确 normal 限制到固定 `16×8×8` 物理粗网格，再提升回 `32×16×16` 场，并只做一次全残差最小二乘步。预测只读取部署可见的观测、报告几何和求解器状态，不读真值，不训练，也不搜索粗网格、步数、阻尼或正则。

## 独立结果

在已经打开的 BLASTNet Case 19 首帧、13 套九相机几何上，候选通过绝对门 `13/13`，但相对 K16 的 matched-accuracy 只通过 `7/13`。六个失败全部只来自 observation；field、full-gradient 和 interior-gradient 没有 matched 失败。

候选相对 K16 的 observation 比值为：p50 `1.035899`、p90-higher `1.089650`、worst `1.209549`，高于冻结的 `1.05` 匹配线。候选逻辑在线账为 `16A+15A^T`，K16 reference 为 `16A+16A^T`。同一首帧小门上，已封存的 full-row control 与 single-half control 都是 `13/13`，所以该受限门本身也不能隔离出新机制优势。

完全独立第二实现通过 `25/25` 项检查。候选场最大相对差为 `3.65e-11`，指标最大绝对差为 `4.76e-11`，相机乱序差为 `0`，物理残差重放误差为 `1.65e-14`。低层几何、物理核和原始数据读取仍共享，因此不能称端到端物理独立。

## 判决与边界

封存判决是 `FAIL_CASE19_COARSE_RESIDUAL_GALERKIN_FRAME_ZERO_V268_4`。固定 `16×8×8`、单步、全局粗空间 residual correction 路线关闭；不通过改粗网格、加步数、换精确粗解、加正则或训练模型来挽救。

这是已打开 Case 19 上的 post-open 首帧负机制证据，不是完整序列结果，不是有效减调用、wall/RSS、外部泛化、曲线射线或真实 BOST 结果。`algorithm_breakthrough=false`，训练和 GPU 仍未授权。

# v268: a fixed one-step global coarse correction misses K16 observation matching

## Why this was tested

v267 showed that synchronously adding local directions creates strong cross-block interference. v268 therefore tests a physically distinct but still falsifiable deterministic mechanism. Starting from the sealed v258 parent, the exact normal of the full-observation residual is restricted to a fixed `16×8×8` physical coarse grid, lifted back to the `32×16×16` field, and used for one full-residual least-squares step. Prediction reads only deployment-visible observations, reported geometry, and solver state. It uses no truth, training, or search over coarse grids, step count, damping, or regularization.

## Independent result

On frame zero of the already-opened BLASTNet Case 19 across 13 nine-camera rigs, the candidate clears the absolute gate on `13/13` cells but reaches only `7/13` K16-matched cells. All six failures are observation-only; field, full-gradient, and interior-gradient have no matched failures.

The candidate-to-K16 observation ratios are p50 `1.035899`, p90-higher `1.089650`, and worst `1.209549`, crossing the frozen `1.05` matching line. The logical online ledger is `16A+15A^T`, versus `16A+16A^T` for K16. On this same restricted frame-zero gate, the sealed full-row and single-half controls both reach `13/13`, so the gate would not isolate a new mechanism advantage either.

The fully independent second implementation passes `25/25` checks. Maximum relative candidate-field difference is `3.65e-11`, maximum absolute metric difference `4.76e-11`, camera-permutation difference `0`, and physical residual-replay error `1.65e-14`. Low-level geometry, physics kernels, and raw-data loading remain shared, so end-to-end physics independence is not established.

## Verdict and boundary

The sealed verdict is `FAIL_CASE19_COARSE_RESIDUAL_GALERKIN_FRAME_ZERO_V268_4`. The fixed `16×8×8`, one-step, global coarse residual-correction route is closed. It will not be rescued by changing the coarse grid, adding steps, switching to an exact coarse solve, adding regularization, or training a model.

This is post-open negative frame-zero mechanism evidence on the opened Case 19. It is not a complete-sequence result and establishes no effective call reduction, wall/RSS result, external generalization, curved-ray validity, or real BOST result. `algorithm_breakthrough=false`; training and GPU use remain unauthorized.
