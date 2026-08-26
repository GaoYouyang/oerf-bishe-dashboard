# v250：首帧平滑通过，但 Charbonnier 非线性优势未被隔离

## 为什么做

v249 因独立系数门不闭合而停止后，v250 结果前冻结一个物理上不同且不训练模型的局部非线性机制。主候选从零起点 reported-geometry 对角 Jacobi-PCGLS K13 出发，在 `32×16×16` 三维 active graph 上用 deployment-visible MAD 尺度执行 12 步显式 Charbonnier 扩散，再做一次精确投影和一次未修改的 restarted PCGLS K1。主候选逻辑账为 `15A+14A^T`，训练参数为 0。

最重要的结果前对照不是更便宜的 K14，而是**同价线性热扩散**：它使用完全相同的 K13、12 步扩散、精确投影与 restarted K1 shell，只把 Charbonnier conductance 固定为 1，逻辑账同样是 `15A+14A^T`。另保留原始 K14 `14A+14A^T` 与 K16 reference `16A+16A^T`。只检查已开封 Case 19 的 13 套 rig 各自首帧；只有主候选通过、同价对照未通过且独立复算闭合，才允许完整 429 单元序列。

## 正式运行与独立复算

formal 完成 13 个首帧单元并通过 **24/24** 项有效性检查。完全独立第二实现从原始已开封输入重建 active graph、MAD、两种扩散、投影、PCGLS、物理观测、四项指标、相机换序和调用账，通过 **38/38** 项检查。

正式与独立结果闭合：最终场最大相对差 `1.0219e-9`，指标最大绝对差 `4.7584e-11`，汇总最大绝对差 `2.1601e-11`；相机换序、离散判决和调用账全部一致。没有放宽容差，也没有在评分后补跑。

## 结果与控制归因

Charbonnier 主候选通过 **13/13** 个首帧单元和 **13/13** 套 rig；四项 p90 比值为 `0.222143 / 0.353339 / 0.396675 / 0.108549`，均低于冻结的 `0.5 / 0.75 / 0.75 / 0.2` 首帧门。

然而，同价线性热扩散对照也通过 **13/13**；其四项 p90 为 `0.221348 / 0.326790 / 0.398222 / 0.169375`，同样全部过门。原始 K14 只有 **7/13**，K16 reference 为 **12/13**。这说明“在 K13 后加入固定局部平滑再重启一次”存在首帧 headroom，但现有证据不能把收益归因于 Charbonnier 的非线性 conductance。

## 权威判决

权威判决为 `PASS_CASE19_FRAME_ZERO_BUT_CHARBONNIER_ADVANTAGE_NOT_ISOLATED_V250`。独立验证通过，首帧平滑信号成立；但同价确定性线性对照完整解释了 13/13，因此 **Charbonnier 特异优势没有被隔离**。按结果前规则，不运行完整 429 单元序列，关闭当前 Charbonnier 特异路线，也不把已经开封数据上的线性热对照重新包装成前瞻成功。

这不是整条 C 路线失败，也不是数学不可能证明。它没有建立有效的序列级 exact-call 减少、fresh wall/RSS、外部泛化、曲折光线或真实 BOST 结果。`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`；不训练大模型，不租 GPU。

# v250: frame-zero smoothing passes, but the Charbonnier-specific advantage is not isolated

## Motivation and frozen mechanism

After v249 stops at an incomplete independent coefficient gate, v250 preregisters a physically distinct, non-learned local nonlinear mechanism. The primary starts from zero-start reported-geometry diagonal-Jacobi PCGLS K13, applies 12 explicit Charbonnier-diffusion steps on the active `32x16x16` graph using a deployment-visible MAD scale, and then performs one exact projection plus one unchanged restarted PCGLS iteration. Its logical ledger is `15A+14A^T`, with zero trainable parameters.

The decisive preregistered comparator is an **equal-call linear heat-diffusion control**. It uses the identical K13, 12-step diffusion, exact projection, and restarted-K1 shell, changing only the Charbonnier conductance to a constant 1. Its ledger is also `15A+14A^T`. Raw K14 at `14A+14A^T` and the K16 reference at `16A+16A^T` are retained. Only the frame-zero cell from each of 13 already-opened Case 19 rigs is scored. The full 429-cell sequence requires the primary to pass, the equal-call control to fail, and independent recomputation to close.

## Formal and independent recomputation

Formal completes all 13 frame-zero cells and passes **24/24** validity checks. A fully independent second implementation rebuilds the active graph, MAD scale, both diffusion rules, projection, PCGLS, physical observations, four metrics, camera permutation, and call ledgers from the opened inputs. It passes **38/38** checks.

Formal and independent outputs close numerically: maximum final-field relative disagreement is `1.0219e-9`, maximum metric disagreement is `4.7584e-11`, and maximum summary disagreement is `2.1601e-11`. Camera permutation, discrete decisions, and call ledgers all agree. No tolerance is relaxed and no post-score rerun is used.

## Result and control attribution

The Charbonnier primary passes **13/13** frame-zero cells and **13/13** rigs. Its field, full-gradient, interior-gradient, and observation p90 ratios are `0.222143 / 0.353339 / 0.396675 / 0.108549`, all below the frozen `0.5 / 0.75 / 0.75 / 0.2` frame-zero limits.

The equal-call linear heat control also passes **13/13**, with p90 ratios of `0.221348 / 0.326790 / 0.398222 / 0.169375`, again below every limit. Raw K14 reaches only **7/13**, while the K16 reference reaches **12/13**. The evidence therefore supports frame-zero headroom for fixed local smoothing after K13 and one restart, but it cannot attribute that headroom to Charbonnier's nonlinear conductance.

## Authoritative verdict

The authoritative verdict is `PASS_CASE19_FRAME_ZERO_BUT_CHARBONNIER_ADVANTAGE_NOT_ISOLATED_V250`. Independent validation passes and the frame-zero smoothing signal is real, but the equal-call deterministic linear control completely explains the 13/13 result. Under the frozen rule, the full 429-cell sequence does not run, the Charbonnier-specific route closes, and the already-opened linear-heat diagnostic is not repackaged as a prospective success.

This does not close the C route or prove mathematical impossibility. It establishes no effective sequence-level exact-call reduction, fresh wall/RSS benefit, external generalization, curved-ray validation, or real-BOST result. `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`; no larger model or GPU run is authorized.
