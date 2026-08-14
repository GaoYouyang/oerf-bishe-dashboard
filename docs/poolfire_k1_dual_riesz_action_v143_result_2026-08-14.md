# v143 Riesz-action 坐标哨兵：共享线性可预测性失败

> 日期：2026-08-14  
> 范围：五条已开封 PoolFire straight-ray 三维代理轨迹，5/7/9/12 个活动相机  
> 判决：`FAIL_SHARED_LINEAR_RIESZ_ACTION_PREDICTABILITY_V143`  
> 边界：`algorithm_breakthrough=false`，不授权 GPU、神经训练、资源门、外部门或真实 BOST 声明

## 先说人话

v142.4 已经证明：直接从部署可见特征线性预测一组非正交基系数，整轨迹迁移失败。v143 进一步排查一个更具体的解释：会不会只是“系数坐标不好”，而真正的物理作用其实容易预测？

为此，v143 不再预测原始系数，而是预测每个 target-camera / detector-component 分组中，成对深度基在 detector space 里的 Riesz / Gram action。这个坐标把样本相关的非正交求逆从学习目标中移到固定的几何局部恢复步骤。模型容量、五折整轨迹留一、部署可见输入和 20 个结果前哨兵都保持冻结。

结果是否定的。正则化 Riesz 逆本身数值稳定，20 个哨兵的 oracle 重建误差最大只有 `0.00308`，低于 `0.02` 门；但共享线性模型预测的 action 几乎与目标正交：余弦中位数 `0.0269`，最低 `0.0107`，而冻结门是每个哨兵至少 `0.90`。尺度不变相对误差中位数为 `0.99964`，20 个哨兵通过数是 `0/20`。

所以，当前失败不能再归因于“只差一个更自然的线性坐标”。这关闭的是共享线性 Riesz-action 假设，不是所有非线性模型，也不是整个 C 路线。

## 实际结果

| 检查 | 结果 | 冻结门 | 判决 |
|---|---:|---:|---|
| Oracle Riesz 逆最大尺度不变误差 | `0.00308` | `<= 0.02` | 通过 |
| 预测 action 误差中位数 | `0.99964` | 每单元 `<= 0.45` | 失败 |
| 预测 action 误差最坏值 | `0.99994` | 每单元 `<= 0.45` | 失败 |
| 预测 dual 余弦中位数 | `0.02688` | 每单元 `>= 0.90` | 失败 |
| 20 个哨兵完整通过 | `0/20` | `20/20` | 失败 |
| 五条轨迹 p90-higher | `0.99963–0.99994` | 每条 `<= 0.35` | 全失败 |
| 最大 solver stationarity | `1.12e-11` | `<= 1e-7` | 通过 |

三个目标空间诊断的中位误差分别为：

- v142.4 原始系数预测：`0.71224`；
- deployment-visible joint LS：`0.56963`；
- v143 Riesz-action 共享线性预测：`0.99964`。

这不是说 raw coefficient 或 joint LS 已经可用；它们都没有通过原先的物理门。这个对照只说明，Riesz action 重参数化没有把跨轨迹关系变成一个可学习的共享线性映射。

## 为什么第一次独立状态是 Inconclusive

独立程序完整重建了 3700 个 action label、五个外折模型、20 个哨兵预测、正则化 Riesz 逆和全部目标空间指标。预测 action、系数与 dual 的正式/独立差分别约为 `3.24e-11`、`1.00e-9` 和 `2.25e-13`。

但第一版 validator 把八列不同量纲的指标统一用 `1e-8` 绝对误差比较。条件数约为 `10^7`，正式与独立条件数的最大绝对差是 `0.00546`，相对差却只有 `8.35e-11`；于是唯一的 `metrics` 完整性检查被错误触发，原始独立状态按合同保留为 `INCONCLUSIVE`。

随后做的是结果后审计修复，不是新实验：封存数组、模型、预测、哨兵、科学阈值和判决规则全部不变；无量纲指标继续使用绝对误差，条件数诊断改用对称相对误差。除条件数外的指标最大绝对差为 `3.03e-11`，审计通过并恢复原本由独立数组直接支持的负判决。这个过程公开披露，因为审计修复不能伪装成结果前预注册。

## 对“要不要租算卡”的直接结论

现在不租。当前任务使用 NumPy CPU 做确定性线性代数和独立重算，GPU 不会改变表示是否可预测。更重要的是，v143 已按冻结门关闭当前共享线性 Riesz-action 假设，继续用更大网络或更多算力挽救会混淆“机制是否存在”和“模型是否够大”。

只有后续小型 CPU 诊断先证明：部署可见特征的局部邻域中，held-out action target 具有一致、可辨识的条件结构，才值得另冻一个最小非线性 sentinel；再由实测吞吐决定是否租 GPU。

## 证据边界

- 这是已开封 PoolFire straight-ray 代理上的 20 哨兵目标空间诊断；
- 没有运行完整 3700 单元物理 replay，因为哨兵门已经失败；
- 没有减少可用算法的 exact `A/A^T` 调用；
- 没有 wall time、RSS、独立公开外门、curved ray 或真实 BOST 结果；
- 不能写成数学不可能、全球唯一、SOTA 或论文成功。

## English checkpoint

v143 tests whether the v142.4 failure is merely a poor raw-coefficient coordinate. Instead of predicting coefficients in a sample-dependent non-orthogonal pair-depth basis, it predicts the geometry-local Riesz/Gram action for each target-camera and detector-component group, then recovers coefficients through a fixed regularized inverse. The oracle inverse is numerically suitable: its maximum scale-invariant error is `0.00308`, below the `0.02` gate. Cross-trajectory shared-linear prediction nevertheless fails all `20/20` preregistered sentinels. Median and worst prediction errors are `0.99964 / 0.99994`, median cosine is `0.02688`, and all five trajectory p90-higher errors are approximately one, far beyond their frozen gates.

The first independent report remained inconclusive because one absolute tolerance was incorrectly applied to both near-unit dimensionless metrics and condition numbers near `1e7`. The sealed independent arrays were preserved. A transparent post-result dimensional audit kept every scientific gate, model, prediction, and sentinel unchanged, comparing the condition diagnostic relatively; its maximum relative difference is `8.35e-11`, while the maximum non-condition metric difference is `3.03e-11`. The resulting scientific decision is `FAIL_SHARED_LINEAR_RIESZ_ACTION_PREDICTABILITY_V143`.

This closes the shared-linear Riesz-action hypothesis, not all nonlinear models or the full C route. No full physical replay, GPU training, resource gate, external generalization, curved-ray validation, or real-BOST claim is authorized. `algorithm_breakthrough=false`.
