# 多帧 BOST `q_cal` 的 500 噪声覆盖审计与 variable projection 结果

> 日期：2026-07-22
> 机器状态：`POSTOPEN_SYNTHETIC_ESTIMATOR_NO_GO`
> 证据等级：`PREREGISTERED_POSTOPEN_SYNTHETIC_ESTIMATOR_AUDIT`
> 突破监测：**否**
> 一句话：独立噪声复本证实了“场先拟合、几何后估计”的 plug-in 协方差会在部分 rig/方向严重欠覆盖；经典迭代 variable projection 只得到混合的弱改善，没有过预注册门。

## 1. 这次补上了 v2 缺的什么

v2 把同一条高斯噪声方向放大/缩小，用来定位 SNR 门槛；它不能回答95% 置信域是否真有 95% 覆盖。本轮固定：

- 3 个 synthetic rig、每个 rig 3 个 v2 方向；
- `||q||/q_ref = 0, 1/8, 1/4, 1/2, 1, 2`；
- 噪声倍率 `1/128, 1/64, 1/32, 1/16, 1`；
- 每个 `rig x direction x amplitude x noise` 独立生成 500 个高斯噪声复本；
- teacher 和 plug-in 同 cell 配对，不同 amplitude/noise cell 不复用标准噪声；
- 主门是 `q=q_ref, noise=1/128`，但这两个值已在 v2 中打开，所以不是 fresh。

共产生 540 行方法-cell、270,000 次估计器评估、54 行非线性余项和 864 个配对 variable-projection trial。完整运行用时 64.35 s，无数值失败。

## 2. 预注册主门：三道都没有全过

| 门 | 实际 | 预注册要求 | 判决 |
|---|---:|---:|---|
| teacher 95% coverage cell | 8/9 | 9/9 | FAIL |
| plug-in 95% coverage cell | 5/9 | 9/9 | FAIL |
| plug-in pooled relative-L2 median / p90 | 0.0743 / 0.1390 | <=0.25 / <=0.75 | 单项 PASS |
| plug-in false accept | 0% | <=1% | 单项 PASS |
| iterative q median reduction | 8.91% | >=10% | FAIL |
| iterative p90 / maximum | 0.2715 / 0.7768 | 均不劣于 0.2729 / 0.8091 | PASS |
| objective monotone | 100% | >=95% | PASS |
| trust-bound hit | 86.0% | <=5% | FAIL |

teacher 唯一失败 cell 的覆盖率是 0.928，95% Clopper--Pearson 区间上端是 0.9491，刚好不包含 0.95。这说明在最低噪声下，非线性余项即使很小，也能被 coverage 检出；不应事后把门改成 8/9。

## 3. 最有价值的负结果：plug-in 场解在吃掉几何信号

在 `q=2 q_ref, noise=1/128` 时：

- teacher 九个 cell 的平均 coverage 仍是 0.949，9/9 过单 cell 门；
- plug-in 平均 coverage 降到 0.480，只有 1/9 过门；
- 两个 rig-direction cell 的 plug-in coverage 为 0，另一个只有 0.016；
- 此时 projected nonlinear remainder 中位只有线性响应的 0.54%，teacher 仍基本校准。

因此，此网格的主要失败不能简单归因于 `A(q)` 太非线性。更直接的机制是：plug-in 用同一份 observation 在 nominal `B(0)` 下先估 `x_hat`，真实 `q` 产生的一部分信号被错当成场变化吸收；随后用这个 `x_hat` 构造 `C_est`，但 `sigma^2(C_est^T P C_est)^-1` 没有包含“场拟合误差 + 同数据相关性 + Jacobian 变化”。

一个后验描述性检查与此一致：在非零 amplitude 的低噪声 cell 中，plug-in coverage 与场误差的 Spearman 相关为 -0.638；teacher-minus-plugin coverage gap 与场误差的相关为 0.755。这两个数是看过结果后才算的，而且 amplitude 是共同混杂因素，**只能用来设计下一轮，不是论文结论或因果证明**。

## 4. variable projection 为什么还不能叫成功

稠密 iterative reference 每步都重新 profile `x(q)`，并使用含 `dx/dq` 的完整 separable-LS Jacobian。三个 rig 的中心差分相对误差为 `1.98e-6`--`5.35e-6`，实现门通过。

汇总看，它对 one-step 有小改善：

- q relative-L2 中位从 0.1094 降到 0.0996，下降 8.91%；
- field relative-L2 中位从 0.02364 降到 0.02332，只下降 1.33%；
- sequence field relative-L2 中位从 0.01518 降到 0.01489，只下降 1.90%；
- q 误差 60.4% 的配对复本优于 one-step，并非稳定逐例胜出。

异质性很强：`q=2 q_ref` 两档噪声下，中位 q 误差分别从 0.0516 降到 0.0341、从 0.0876 降到 0.0713；但 `q=0.5 q_ref` 的最低噪声中位数反而从 0.1385 升到 0.1441。这个 subgroup 是已打开结果，不能拿 `q=2` 替代全局门。

86% trial 触发 trust bound，主要因为从 `q=0` 起步且初始半径只是 0.02。这不等于数值发散：864/864 的 objective 都单调，最后都以 step tolerance 停止；但预注册门已经规定 trust-hit `<=5%`，所以必须判 FAIL。下一轮可以把“触发限幅”和“最终打到 `||q||=0.1` 局部边界”分开预注册，但不能回头改本轮判决。

## 5. 由负结果导出的三个算法候选

### A. Profile-orthogonal score + sandwich covariance（优先）

不再把 plug-in `C_est` 当作固定量。先由 variable projection 构造对 nuisance field 一阶正交的 profile score，再用包含场估计影响的 sandwich/Godambe 协方差。最小研究问题是：在不读 `x_true`、不使用 fresh exact mass 的前提下，能否把 `q=q_ref` 的 9/9 coverage 恢复，同时不把置信域扩到毫无功效？

### B. Frame/view cross-fitting（只作对照）

用一组帧/视角估场，在不重叠的帧/视角上构造 geometry score，交换后聚合。它可以减少“同份噪声同时影响 `x_hat` 和 score”，但不会自动消除场估计误差；如果 3 帧子集对 512 维初场不足以稳定估计，它应 fail closed，不应强行叫“去偏”。

### C. 可学的 tangent proposal（真实接口后）

DeepONet/FNO/NeRIF 组件不直接输出并授权 `q`，只预测 transport/innovation tangent、初值或 damping proposal。经典 profile score、真实 forward JVP/VJP、held-out camera/time 与置信域决定是否接受。它的创新性只能来自 BOST 几何-反应场耦合、可验证的失败路由和真实终点，不是把 FNO 名字放进 solver。

## 6. 下一轮怎样才不是重复刷 synthetic

1. 先把本轮当作 development data，只用于冻结 A/B 的公式、阈值和失败门。
2. 新 audit 至少覆盖五个 mode 的正/负单轴与每个 rig 的最弱广义方向；三个 v2 随机方向不够。
3. 另外冻结新 rig/新噪声 namespace，不在本轮 270,000 条开放样本上反复调阈值。
4. 同时保留 q coverage、field/sequence error、逐 rig 尾部、false accept、abstention、factorization/JVP/VJP 调用和 wall time。
5. 更重要的外部门仍是请何远哲师兄确认真实 callable、straight/curved residual 层级、JVP/VJP、几何标定、timestamp、anchor/sentinel、TDBOST 主要失败例和组内基线。

## 7. 初学者这周应读什么

1. 手算一个两参数可分离最小二乘，理解为什么每次改 `q` 都要重新解 `x(q)`。
2. 推导 `H dx/dq_j = D_j^T r - B^T D_j x`，再用中心差分检查符号。
3. 学联合置信椭球与五个边际区间的区别，手算 Clopper--Pearson 区间。
4. 理解 plug-in covariance、sandwich covariance、profile likelihood 和 cross-fitting 各在补什么假设。
5. 把四联图 D 的每个点当作配对实验；会解释为什么“中位提高 8.91%”不等于稳定优越。

入口：[Variable Projection 经典文献](https://epubs.siam.org/doi/10.1137/0710036)、[Nuisance parameters in inverse problems](https://arxiv.org/abs/1206.6532)、[动态 transport tomography](https://arxiv.org/abs/1705.06079)、[NeRIF](https://arxiv.org/html/2409.14722v2)、[TDBOST](https://doi.org/10.1145/3809488)。

## 8. 复现

```bash
cd /path/to/oerf-bishe-dashboard
.venv/bin/python -m learning_labs.temporal_qcal_bootstrap_varpro_lab \
  --output-dir learning_labs/results/temporal_qcal_bootstrap_varpro_v1
.venv/bin/python -m pytest -q \
  learning_labs/test_temporal_qcal_bootstrap_varpro_lab.py
cd learning_labs/results/temporal_qcal_bootstrap_varpro_v1
shasum -a 256 -c checksums.sha256
```

固定产物：`report.json`、`bootstrap_cells.csv`、`nonlinear_remainder.csv`、`varpro_trials.csv`、`varpro_cells.csv`、`temporal_qcal_bootstrap_varpro.png` 和 `checksums.sha256`。

## 9. 不能说的话

- 不能说 variable projection 已稳定优于 one-step；
- 不能用 `q=2` 的后验子组替代全局失败；
- 不能把 teacher 当成可部署方法；
- 不能把 plug-in proxy 叫严格 CRLB；
- 不能声称优于 TDBOST、NeRIF、DeepONet、FNO 或真实实验基线；
- 不能声称新算法、真实三维/4D 重建、跨 rig 泛化、论文成功或突破。

**当前最真实的研究进展，是把“为什么 plug-in 不可信”定位到可直接设计下一个估计器的机制，而不是得到了一个已成熟模型。**
