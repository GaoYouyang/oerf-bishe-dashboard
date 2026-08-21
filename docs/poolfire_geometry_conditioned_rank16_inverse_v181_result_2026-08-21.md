# v181：显式几何条件化之后，固定 rank-16 逆因子仍未通过

更新：2026-08-21

## 结论

v179 已证明，在当前已开封的 PoolFire 代理上，五相机观测与报告几何足以精确辨识全部 `1,009` 个仿射坐标。v180 随后否定了一个不随几何变化的共享 diagonal + rank-16 逆近似。v181 直接检验最自然的剩余解释：**是不是只要让同样大小的低秩因子显式依赖当前相机几何，就能恢复精确逆？**

唯一主候选为每套报告几何单独构造的 Jacobi whitening + 固定 `16` 个谱校正模态。因子生成只读 observation 与 reported geometry，不读留出三维真值，不训练神经网络；候选之后接一轮完全未修改的 CGLS K1。

正式运行与完全独立第二实现均通过，科学判决为：

`FAIL_GEOMETRY_CONDITIONED_RANK16_INVERSE_V181`

五相机与全九相机在 K0、K1 后都只有 `0/52` 个严格安全单元，完整标定与完整帧也都是零。显式加入几何依赖仍不足以救回固定 rank-16 结构。

## 主候选结果

五相机 K1 的 field / gradient / observation p90 为：

`0.510874 / 0.819616 / 0.568073`

对应 worst 为：

`0.538749 / 0.842309 / 0.602679`

全九相机 K1 的 p90 为：

`0.483807 / 0.693110 / 0.581855`

对应 worst 为：

`0.516694 / 0.730880 / 0.633869`

冻结绝对门要求 field / gradient / observation p90 分别不超过 `0.50 / 0.75 / 0.20`。五相机三项 p90 全部越线；九相机 field 与 gradient 守门，但 observation 仍远高于 `0.20`。因此两档相机都没有任何严格安全单元。

便宜的纯 Jacobi 几何控制、一次仿射坐标 CGLS 与静态训练均值也都失败。主候选的逻辑在线账虽然是 K1 的 `2A+2A^T`，但 matched-accuracy 没有通过，所以不能声称 exact-call 减少。

## 为什么这不是“差一点，多加几个模态即可”

几何白化后的谱范围为 `0.01460` 至 `5.98186`，说明几何条件化本身正常工作。问题在于误差不是集中在少数异常模态上：Jacobi 逆残差 p90 为 `1.024692`，增加固定 `16` 个谱校正后只有 `1.017375`，相对下降约 `0.71%`。

这支持一个更具体的负判断：当前失配是宽谱的，固定 rank-16 校正太窄。它不支持结果后继续调 rank、特征值 floor、ridge 或选模评分；这些都会把同一失败假设改造成事后搜索。

## 独立复算

完全独立第二实现重新构造几何法方程、Jacobi 白化、特征分解、16 个固定模态、坐标、候选场、物理观测、K1 与所有门。`48/48` 项检查全真，所有离散判决一致。

- factor action 最大相对差 `1.82e-11`；
- 坐标最大相对差 `1.35e-11`；
- 候选场最大相对差 `1.03e-11`；
- 指标最大绝对差 `5.92e-12`；
- arm 汇总最大绝对差 `5.21e-12`。

相机换序不改变法方程、因子或预测，几何依赖与 permutation invariance 的机械链成立；失败来自科学精度门，而不是实现漂移。

## 科学判断改变在哪里

v180 留下“共享因子没有显式看到几何”这一解释；v181 已直接否定这个具体解释。关闭的是**固定 Jacobi 白化、按几何构造、rank-16 谱校正**这一家族，不是整条 C 路线，也不是对所有非线性或 observation-adaptive 机制的不可能性证明。

后续不扩大 rank、不调谱规则、不训练 CNN / FNO / UNO / DeepONet，也不租 GPU 挽救。只有一个物理上真正不同、结果前冻结的 observation-adaptive 机制，或新的配对二维 BOS 物理信息，才值得继续。

v181 不是部署算法、exact-call 减少、wall/RSS 加速、外部泛化、curved ray、真实 BOST、论文成功或算法突破：

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`broad_external_generalization=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v181: explicit geometry conditioning still does not make a fixed rank-16 inverse pass

Updated: 2026-08-21

v179 establishes that five-camera observations and reported geometry identify all `1,009` affine coordinates on the opened PoolFire proxy. v180 rejects a geometry-independent shared diagonal-plus-rank-16 inverse approximation. v181 tests the most direct remaining explanation: can the same compact rank become sufficient when its factor is rebuilt explicitly for each reported camera geometry?

The sole primary uses a geometry-specific Jacobi whitening plus exactly `16` spectral corrections. Factor construction reads only the observation and reported geometry, never held-out 3D truth, and trains no neural network. One unchanged physical CGLS K1 step follows the initializer.

Formal execution and a fully independent second implementation pass, but the scientific decision is `FAIL_GEOMETRY_CONDITIONED_RANK16_INVERSE_V181`.

Both five-camera and all-nine arms remain `0/52` strict-safe after K0 and K1, with zero complete calibrations and frames.

For five-camera K1, field / gradient / observation p90 values are `0.510874 / 0.819616 / 0.568073`, with worst values `0.538749 / 0.842309 / 0.602679`. For all-nine K1, p90 values are `0.483807 / 0.693110 / 0.581855`, with worst values `0.516694 / 0.730880 / 0.633869`.

The frozen p90 limits are `0.50 / 0.75 / 0.20`. All three five-camera tails fail. All-nine field and gradient satisfy their limits, but observation remains far above `0.20`. Geometry-specific Jacobi, one-step affine-coordinate CGLS, and the static fit mean also fail.

The whitened spectrum spans `0.01460` to `5.98186`, so the geometry-conditioned factorization is numerically valid. The mismatch is broad rather than concentrated in a few modes: inverse-residual p90 changes only from `1.024692` under Jacobi to `1.017375` after the fixed 16 corrections, a relative reduction of about `0.71%`.

A fully independent implementation reconstructs the geometry normal system, whitening, eigensystem, fixed 16 modes, coordinates, candidate fields, physical observations, K1, and every gate. All `48/48` checks pass and all discrete decisions agree. Maximum factor-action, coordinate, and candidate-field relative differences are `1.82e-11`, `1.35e-11`, and `1.03e-11`; maximum metric and arm-summary absolute differences are `5.92e-12` and `5.21e-12`.

v181 closes only the fixed Jacobi-whitened, geometry-conditioned rank-16 spectral-correction family. It is not an impossibility result for nonlinear or observation-adaptive mechanisms and does not close the C route.

Do not tune rank, eigenvalue floors, ridge, or the mode score after seeing this result, and do not use a larger CNN, FNO, UNO, DeepONet, or GPU as rescue. A next attempt requires one preregistered physically different observation-adaptive mechanism or new paired 2D BOS evidence.

v181 is not a deployed algorithm, exact-call reduction, wall/RSS speedup, external generalization, curved-ray validation, real BOST, paper success, or an algorithmic breakthrough: `algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `broad_external_generalization=false`, `curved_ray_validated=false`, `real_bost=false`.
