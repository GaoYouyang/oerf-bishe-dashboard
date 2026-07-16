# v3k-E 师兄审核简报：投影 BB 加速成立，但噪声停止门槛失败

更新时间：2026-07-15  
定位：T16“算子学习 × 少视角三维重构”强数值对照，不是新算法成果。

## 一句话结论

在同样的 64 次 forward 与 64 次 adjoint 调用下，validation-only 选出的 FNO-start alternating projected BB（PBB）把绝对场均 relative L2 从 fixed Landweber 的 `0.147083` 降到 `0.138640`；按独立字段先配对再平均的相对 gain 是 `1.401%`。但它在 validation 有 `37.5%` 的独立字段退化超过 1%，在 noise-OOD 相对 fixed Landweber 平均退化 `11.56%`。因此现在不能训练 learned scalar，更不能声称 PBB 是创新；下一门槛应是 noise-aware stopping。

## 本轮究竟比较了什么

- 起点：锁定 FNO 与 validation-tuned ridge，各自先做同一 nonnegative hard-support projection。
- 方法：BB1、BB2、BB1/BB2 deterministic alternation。
- strict 保护：`alpha * L(mask)` 限制在 `[0.05, 1.9]`。
- wide 保护：上界在 validation 中从 `{3, 5, 10, 100}` 选择；下界仍为 `0.05`。
- 迭代数：`{2, 4, 8, 16, 32, 64}`。
- 统计单元：先把每个 source field 的四个布局折叠，再计算 mean、CI、p10 和 harm rate。
- 成本：每次 PBB 迭代严格计 `1 A + 1 A^T`；最终 objective/field/reprojection 的评估 forward 不计入求解器预算并单独标记。
- 防泄漏：只用 validation 选择 strict/wide winner；先写 `selection_commit`，再构造其余四个 development audit 域。

## 关键数字

| 比较 | validation field L2 | A + A^T | 相对 fixed Landweber | 结论 |
|---|---:|---:|---:|---|
| FNO fixed Landweber | 0.147083 | 128 | 0 | 同预算强基线 |
| FNO quadratic step | 0.143855 | 193 | +1.244% | 成本更高，且 noise-OOD 反向 |
| FNO strict PBB, BB1 | 0.147083 | 128 | 近似退回 fixed-step 路径 | 98.44% 的非首步被上界裁剪 |
| FNO wide PBB, alternating, cap 10 | **0.138640** | **128** | **+1.401%** | 均值更好，但尾部与噪声域失败 |

wide PBB 相对 fixed Landweber：

| 域 | mean gain | p10 gain | harm > 1% |
|---|---:|---:|---:|
| validation | +1.401% | -14.043% | 37.5% |
| IID | +1.753% | -13.363% | 37.5% |
| noise OOD | **-11.561%** | -25.020% | 62.5% |
| family OOD | +14.832% | +12.199% | 0% |
| joint OOD | +12.285% | +8.602% | 0% |

## 机制解释

1. strict PBB 并没有真正展示 spectral adaptation。其候选步长几乎总被 `1.9/L(mask)` 上界截断，所以轨迹退化为 fixed Landweber。
2. wide alternating PBB 在相同调用数下明显加快了 data-fidelity 优化；这证明“固定步过于保守”是一个真实机制。
3. 但是含噪逆问题具有 semi-convergence：更充分地拟合 noisy observations 不等于更接近真实三维场。当前实验中 data objective 基本稳定下降，noise-OOD field error 却显著上升。
4. 所以仅加 monotone/nonmonotone line search 解决不了核心问题。SPG 是必须比较的优化基线，但下一项研究门槛是 regularizing stopping，而不是更激进的下降。

## 下一项建议：先做确定性 noise-aware stopping

### H1：whitened discrepancy PBB

沿同一条 PBB 轨迹，选择第一个满足下式的迭代：

`r_t^T Sigma_epsilon^{-1} r_t / m <= tau`

其中 `r_t = M(Ax_t-y)`，`Sigma_epsilon` 只能由部署时可得的背景图、位移置信度或 calibration data 估计；`tau` 只在 validation 上选择并冻结。必须与 fixed `T=64`、oracle best-T、L-curve、residual plateau 和 SPG 做同预算对照。

### H2：operator-conditioned risk stop

只有 H1 仍留下稳定 headroom，才训练一个很小的 stop/risk model。输入限定为部署可得摘要：normalized residual、residual slope、BB step、clip/fallback、camera-mask spectral descriptors、noise/confidence summaries 和 iteration index。输出是“继续/停止”或下一步失败风险，不直接生成三维场。

### H3：BOST-specific contribution

可投稿贡献不能是“把 BB 用于 BOST”。候选贡献应组合：

- variable-camera noise/confidence-aware discrepancy；
- 对 field tail、held-out camera、front/gradient 和 noise-OOD 的风险控制；
- 与 fixed Landweber、quadratic、PBB、SPG、oracle best-T、FNO/DeepONet warm start 的完整消融；
- 独立 forward/geometry 与真实或开放 BOS 数据确认；
- fresh lock 上一次性确认，不在开发域反复调规则。

## 请师兄只回答四个决策问题

1. 真实 BOST pipeline 能否给出每个位移像素/光线的 noise、confidence 或 covariance proxy？
2. 真实 forward `F(x)` 与 adjoint/VJP 是否已经可调用；hard support 在部署时是否已知？
3. 组内更关心固定时间内的误差、最终精度、最坏样本，还是 held-out camera consistency？
4. 能否封存新的 fields、layouts、noise levels 和真实 case，作为停止规则选定后的 fresh lock？

## 复核入口

- [可视化学习与研究决策页](./projected_bb_noise_gate_lab.html)
- [selection commit](./demo_t16_operator/results/v3k_e_projected_bb_gate/v3k_e_selection_commit.json)
- [字段级 pairwise summary](./demo_t16_operator/results/v3k_e_projected_bb_gate/v3k_e_pairwise_summary.csv)
- [步长与 objective audit](./demo_t16_operator/results/v3k_e_projected_bb_gate/v3k_e_bb_step_audit.csv)
- [算子调用 ledger](./demo_t16_operator/results/v3k_e_projected_bb_gate/v3k_e_operator_call_ledger.csv)
- [独立 validator](./demo_t16_operator/validate_v3k_e_projected_bb_results.py)

## 关键文献边界

- Barzilai & Borwein (1988), [Two-Point Step Size Gradient Methods](https://doi.org/10.1093/imanum/8.1.141)：BB1/BB2 的无约束两点谱步长来源。
- Birgin, Martínez & Raydan (2000), [Nonmonotone Spectral Projected Gradient Methods on Convex Sets](https://doi.org/10.1137/S1052623497330963)：SPG 的非单调线搜索与收敛分析；不能把结论直接转移给本轮无 line search 的 clipped PBB。
- Dai & Fletcher (2005), [Projected Barzilai-Borwein methods for large-scale box-constrained quadratic programming](https://doi.org/10.1007/s00211-004-0569-y)：投影 BB 可能循环及相应保护/线搜索问题。
- Hanke (2017), [A Taste of Inverse Problems, Chapter 13](https://doi.org/10.1137/1.9781611974942.ch13)：Landweber 含噪迭代的 discrepancy stopping 基础。
- Hansen (1998), [Iterative Regularization Methods](https://doi.org/10.1137/1.9780898719697.ch6)：迭代次数作为正则化参数与 semi-convergence。

## 当前边界

结果只来自当前 `8 x 16 x 16` 线性 synthetic development protocol；现有 test 域均已用于开发审计，不是最终 blind confirmation。私有 NPZ 与 checkpoint 未发布，本轮没有训练新权重。
