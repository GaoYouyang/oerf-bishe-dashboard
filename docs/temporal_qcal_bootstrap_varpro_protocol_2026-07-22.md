# 多帧 BOST `q_cal` 幅度、coverage 与 variable projection 预注册

> 状态：`PREREGISTERED_POSTOPEN_SYNTHETIC_ESTIMATOR_AUDIT`
>
> 上游证据：`temporal_qcal_tangent_v2` 已证明精确输运使 profile rank 为 `5/5`，但在注册噪声下 `q` relative-L2 中位为 `9.41`，且 `0/9` 获得不确定度授权。
>
> 证据边界：本轮仍只使用同族 synthetic voxel forward；不使用 OERF 私有数据、VPN 材料或实验场真值，不声称真实标定、重建、算子学习、泛化或论文成功。

## 1. 本轮严格回答三个问题

1. v2 的局部线性估计在多大 `q` 扰动内还有可用的 95% 覆盖率？
2. 由同一 noisy observation 先拟合场、再构造 `C_est` 的 plug-in covariance 是否系统性欠覆盖？
3. 在完全相同的精确输运场模型上，迭代 affine variable projection 是否比 one-step plugin 有稳定、有尾部保护的改善？

任何结果都只是 estimator audit。它不能跳过真实 callable、JVP/VJP、covariance、timestamp、anchor 与强基线合同。

## 2. 固定 forward、rig 与扰动

- 沿用 v2 的 `8 x 8 x 8` voxel grid、`4 x 4` detector、6 帧、3 个 rig、三个 analytic phantom family 与 5 个 scaled pose modes。
- 场序列固定为 `x_t = W_t x_0`；`W_t` 是 v2 的零边界三线性平移。本轮不把该 proxy 写成真实反应流动力学。
- 每个 rig 固定 3 个 v2 同定义随机方向；只改变模长，不按结果重抽方向。
- 参考尺度 `q_ref=0.04`。幅度倍率固定为

```text
0, 1/8, 1/4, 1/2, 1, 2
```

即 `||q|| = 0, 0.005, 0.01, 0.02, 0.04, 0.08`。`0.08` 仍低于 v2 的 `0.1` 局部参数包络，但不预设它仍属于可信线性区间。
- 观测使用实际组合 pose 重建的非线性 voxel operator `A(q_true)` 生成；估计器只使用报告几何下的 `A(0)` 和预先有限差分得到的 `D_j`。

## 3. 独立噪声 coverage 网格

每个 `rig x q-direction x q-amplitude x noise-multiplier` 单独生成 500 个独立标准高斯噪声复本。one-step、teacher 和迭代法在同一 cell 内使用同一批 observation，但不在不同幅度或噪声 cell 之间复用标准噪声。这避免把配对噪声造成的平滑曲线写成独立稳定性。

噪声倍率固定为

```text
1/128, 1/64, 1/32, 1/16, 1
```

`sigma_base` 由 nominal clean response `B(0)x_0` 固定，不随 `q` 幅度改变，避免把信号幅值差异混入噪声定义。

随机 namespace 与 v2 分离：

```text
oerf-temporal-qcal-bootstrap-varpro-1.0
```

不使用 v2 的单条标准噪声方向作为 bootstrap 复本。

## 4. 两个 coverage 估计器

### 4.1 `teacher_linear`

- 使用 synthetic true `x_0` 构造 `C_teacher`，只作正控制/评估上界。
- 协方差 `sigma^2 (C_teacher^T P C_teacher)^-1` 在固定 teacher 线性高斯模型中可称 CRLB；当 `A(q_true)` 的非线性余项不可忽略时，它只是错定线性近似的理论参考。
- 不参与任何 deployable 授权。

### 4.2 `one_step_plugin`

- 对每个 noisy observation 先在 nominal transport nuisance `B(0)` 上做与 v2 相同的 ridge 拟合；
- 用该复本自身的 `x_hat` 构造 `C_est`，再在 `P_B` 正交补上求 `q_hat`；
- 协方差始终称 `plug-in covariance proxy`，不称 CRLB；
- 保留 v2 的 99% 双侧 NIS、95% 最大半径、`||q_hat||+r95<=0.1`、Wald 显著性与拒答合同。

`C_est` 由同一 observation 拟合，而且场估计含 tiny ridge，所以 `m-rank(B)-5` 的卡方 NIS 只保留为与 v2 配对的诊断量，不是严格有限样本分布证明。本轮的主校准证据是 500 次直接 empirical coverage，不用 in-sample residual 代替它。

## 5. coverage、偏差与尾部指标

每个网格和方法固定输出：

- 95% 五维置信椭球 empirical coverage；
- coverage 的双侧 95% Clopper-Pearson 区间；
- `||q_hat-q_true||/q_ref` 的 mean、median、p90、p95 和 maximum；
- 对于非零 `q`，另报 relative-L2；
- 经验 bias vector 及 `||bias||/q_ref`；
- NIS pass fraction、授权率、授权样本的误差尾部和 false-accept fraction；
- `C_est` 欠秩、非有限、协方差非 PSD 或自由度非正时直接 fail closed。

椭球覆盖使用

```text
(q_hat-q_true)^T Cov^-1 (q_hat-q_true) <= chi2_5(0.95).
```

不用五个边际 95% 区间冒充联合 95% 覆盖。

## 6. coverage 门与多重网格边界

对单个 500-replicate cell，coverage 通过必须同时满足：

1. empirical coverage `>=0.925`；
2. 95% Clopper-Pearson 区间包含 nominal `0.95`。

不对 270 个 method cells 挑选最好网格。主门固定在 v2 已指出的 `q=q_ref`、noise=`1/128`：

- `teacher_linear` 正控制需要 9/9 `rig x direction` cells 通过，否则先判非线性/实现问题；
- `one_step_plugin` 的注册-cell校准 screen 同样需要 9/9 coverage cells 通过，且 pooled median relative-L2 `<=0.25`、p90 `<=0.75`、false accept `<=1%`；
- 噪声、幅度更低的网格只用于定位 validity boundary，不能替代主门；
- 本轮网格、门和 `1/128` 都源自已打开的 v2，因此即使全过也只是 development confirmation，不是 fresh 或论文成功。

三个 v2 随机方向没有覆盖五个单轴正负方向和 rig-specific 最弱广义特征向量。因此本轮注册-cell screen 即使通过，也只能解锁下一个 direction-stress audit，不得直接标记 estimator candidate 或五维方向泛化。

## 7. 迭代 affine variable projection 参考

局部 forward 固定为

```text
A_aff(q) = A_0 + sum_j q_j D_j
B_aff(q) = stack_t A_aff(q) W_t.
```

观测仍由实际非线性 `A(q_true)` 生成。每次迭代：

1. 在固定 tiny ridge 下解 `x(q)=argmin ||y-B_aff(q)x||^2+lambda||x||^2`；
2. 对增广 residual `[y-Bx; -sqrt(lambda)x]` 构造 full separable-LS Jacobian，不忽略 `dx/dq`；对每个 `D_j`，使用共享 Hessian 解 `H s_j = D_j^T r - B^T D_j x`，再取 residual derivative `J_j=-D_jx-Bs_j`；
3. 解带 damping 的 5 维 Gauss-Newton step；
4. 每步 `||delta q||<=0.5 q_ref`，总参数 `||q||<=0.1`；
5. 候选点必须重新 profile 场解，并用 actual/predicted reduction ratio 接受 trust step；失败时只允许预冻结 Armijo backtracking，仍无下降则拒绝该步。

为避免看到结果后调参，数值控制现在冻结为：最多 8 次 Gauss--Newton 迭代，初始 trust radius `0.02`、最大 `0.04`、最小 `1e-6`；LM 对角线系数是 `1e-4 * mean(diag(C^T C))`；ratio 接受阈值 `0.1`，小于 `0.25` 将半径减半，大于 `0.75` 且触边时将半径翻倍；Armijo 常数 `1e-4`，最多 8 次 `1/2` 回溯；`||delta q||<=1e-6` 时停止。这些值不使用 `q_true` 选择。

full profile Jacobian 在每个 rig 上用中心差分 `h=1e-4` 独立验算，相对误差必须 `<=5e-5`；这只是实现符号/链式求导门，不是估计器性能门。

迭代 pilot 固定使用：

- `q/q_ref = 1/2, 1, 2`；
- noise `=1/128, 1/64`；
- 3 rigs x 3 directions x 每格 16 个配对噪声复本；
- one-step 和 iterative 必须看相同 observation、相同 `A0/D_j/W_t` 与相同 ridge。

one-step 基线固定为：在 `q=0` 先用 nominal ridge 场构造 plug-in Jacobian，只做一次 `q` 更新；若 raw update 超出 `||q||<=0.1`，按半径做确定性径向投影并记录 boundary hit；然后在该 `q` 上只重新 profile 一次场，得到共同目标、初场和六帧终点。iterative 从 `q=0` 开始，不使用 one-step 或 `q_true` 热启动。主配对比较使用两者都受同一 `0.1` 局部包约束的终点；raw one-step 仍单独报告，不隐藏裁剪。

每个迭代复本还必须报告共同初场 relative-L2 和六帧 sequence field relative-L2。`q` 更准但场误差或时间序列误差更差，不能记为三维/4D 改进。

当前实现允许稠密 `B` 与 Cholesky，但必须记录 normal-matrix assembly、factorization、line-search evaluation 和峰值规模。该成本不等于真实 matrix-free `A/A^T/JVP/VJP` 调用，不得声称端到端加速。

稠密 pilot 使用同一 `B_aff(q)` 和同一 ridge 目标评价 one-step 与 iterative；每个数值失败都留在分母中并记为失败，不得丢弃难例。另报 oracle nonlinear `A(q_true)` 与 affine `A_aff(q_true)` 的数据失配，防止把架构改善与 forward approximation error 混为一谈。

迭代研究信号必须同时满足：

- pooled median error 相对 one-step 至少下降 10%；
- p90 和 maximum 均不劣于 one-step；
- 至少 95% replicate objective 单调不增；
- trust-bound hit fraction `<=5%`；
- 不使用 truth 选择迭代次数、damping 或 line-search step。

本门过线也只授权下一轮 matrix-free 实现，不授权算法优越性。当前没有 held-out camera/frame，也没有同调用预算的 matrix-free baseline，因此 iterative gate 不可能在本轮升格为论文主结论。

## 8. 固定输出

- `report.json`：门、证据边界、计数与 claim authorization；
- `bootstrap_cells.csv`：每个 500-replicate cell 的 coverage、区间、偏差、尾部与拒答；
- `nonlinear_remainder.csv`：每个 rig/direction/amplitude 的 projected nonlinear remainder；
- `varpro_trials.csv`：预冻结 iterative pilot 的配对逐复本记录；
- `varpro_cells.csv`：one-step 与 iterative 配对聚合及峻尾门；
- `temporal_qcal_bootstrap_varpro.png`：coverage、非线性、尾部与迭代对照四联图；
- `checksums.sha256`：全部固定产物摘要。

不保存 270,000 级的全量 bootstrap 逐样本 CSV；固定 seed namespace、源码、配置、逐 cell 聚合与 checksum 必须足以完整重放，同时避免把大型中间表塞入私有 Git 历史。

## 9. 无论结果如何都禁止

- 把 `teacher_linear` 写成 deployable method；
- 把 plug-in proxy 写成严格 CRLB；
- 用低噪声或小幅度格替换失败的注册主格；
- 只报 mean improvement，隐藏 p90、maximum、coverage、false accept 或 abstention；
- 把稠密 synthetic factorization 计时写成真实 BOST 加速；
- 声称已优于 TDBOST、NeRIF、DeepONet、FNO 或真实实验基线。

下一道真实门仍是何远哲师兄确认 callable、JVP/VJP、几何容差、真实 covariance、timestamp、anchor/sentinel、TDBOST 失效案例与终点指标。
