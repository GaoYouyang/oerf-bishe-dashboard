# GCT-KMix O1 冻结证伪协议

> 状态：结果盲态下冻结的合成 truth-oracle 合同。
>
> **突破监测：尚无算法突破。** 本协议不是模型、真实 BOST 结果或泛化证据。

## 1. 只问一个问题

在同一条零初值 CGLS 轨迹的 `k={1,2,3,4,6,8,12}` 中，是否存在单纯形凸组合，既比逐样本最强的安全离散 checkpoint 明显更接近三维 truth，又不让任何一条观测 ray 的二维残差模超过 `k4`？

若答案是否定，停止训练 geometry-conditioned 权重网络。若答案肯定，也只得到“表示空间有合成 headroom”，不自动得到可部署算法。

## 2. 数学合同

冻结线性 B0 forward `A`、checkpoint `x_k`、残差 `r_k=y-Ax_k`。求

`min_w 0.5 ||sum_k w_k x_k - x*||_2^2 / ||x*||_2^2`

满足：

- `w_k >= 0`，`sum_k w_k = 1`；
- 每条 ray `i` 都满足 `||sum_k w_k r_{k,i}||_2 <= ||r_{4,i}||_2 + delta`；
- `delta = max(1e-10 * RMS(y), 1e-12)`，只吸收数值误差，不是可调安全裕量。

这是凸二次目标加二阶锥约束。使用 Clarabel `0.11.1` 同时给出 primal/dual objective、gap、feasibility residual；不把 SLSQP 的 `success=True` 当全局证书。独立再算 `A(sum w_k x_k)`，核对其与 `sum w_k A x_k` 的 closure。

逐 ray 门严格强于逐 camera 的 L2 和经验 p95 no-harm；但最终仍逐 camera 重新评分，不用理论推断替代结果表。

## 3. 数据与成本

- JACRU 16³，解析连续梯度造观测、离散有限差分/三线性插值做 inverse，避免同离散器 inverse crime；
- 三类 morphology：smooth、single interface、two interface；
- seeds `101/211/307`；development 与 OOD 各 9 个 case；
- 3 camera，每 camera `6x6` rays，24 samples/ray；1% iid noise 加 2% camera bias；
- 零初值 CGLS 到 k12 恰为 `12 A + 12 A^T`；混合本身零额外物理调用；closure 的一次 `A` 单列为 evaluator 成本。

JACRU 是反应流形态和稀疏 BOST 机制测试，不是 CFD truth、折射率定标或 OERF 实验数据。

## 4. 不可弱化的基线

1. `CGLS-k4`：逐 ray 安全锚点。
2. `CGLS-k12`：同物理调用预算终点。
3. `GCT-KSelect`：逐样本、使用 truth 的最强安全离散 checkpoint，是主基线。
4. unconstrained simplex oracle：量化逐 ray 安全门损失的 field headroom。
5. development 选出的单一安全固定 checkpoint，在 OOD 原样评估；不得把逐样本 truth 选步冒充部署规则。

若未来升级为“重建方法”主张，还必须补同预算 H1/Tikhonov、TV/Huber、DeepONet、FNO/F-FNO、NeRIF/TDBOST 适配比较。本轮不能跳过表示证伪直接做排行榜。

## 5. 预注册通过门

所有条件同时满足才写 `SYNTHETIC_ORACLE_HEADROOM`：

- development 与 OOD 相对 `GCT-KSelect` 的 mean field gain 均至少 5%；
- development 与 OOD 的 mean H1 gain 均至少 3%；
- worst-case field gain 不低于 -0.1%，`>1%` field harm 率为 0；
- 非平凡混合比例至少 50%；
- 全 case Clarabel 状态为 `Solved`，relative primal-dual gap、primal residual、dual residual 均不超过 `1e-8`；
- 逐 ray violation 不超过 `5e-9 * RMS(y)`；
- 独立重评分的每 camera L2 与 p95 相对 `k4` 增幅均不超过 `1e-8`；
- `|sum(w)-1| <= 1e-8` 且 `min(w) >= -1e-8`；
- 独立 forward closure relative-L2 不超过 `1e-10`；
- 每 case 调用严格为 `12 A + 12 A^T`。

任一 solver、closure、调用或逐 ray 条件失败，fail closed；不能把失败 case 删除后重算均值。

## 6. 与既有结果的差异和局限

- M2.8 只在 learned field 与一个 PCG endpoint 的线段上插值，且用整体 reprojection 门；O1 查完整 CGLS checkpoint 凸包并约束每条 ray。
- N1.7 是 rank-4 measurement-space geometry-Krylov correction basis；O1 不造新 correction basis，只重组普通 CGLS iterates。
- M2.2 已发现 exact null-space oracle headroom；零初值 O1 的所有 checkpoint 共享 row-space/Krylov 来源，**不能创造 null-space 信息**。即使 O1 通过，真正值得训练的下一步也应优先测试固定 learned warm start 是否保留共同 null-space 分量，而不是宣称零初值 mixing 已解决不可辨识性。
- truth 只进入 evaluator；oracle weights 不得进入 predictor 输入、部署排名或“算法胜出”叙述。

## 7. 结果前的判决规则

- 通过：只授权设计 observable weight predictor 与独立 calibration/fallback，不授权算法成功。
- 不通过：冻结 `GCT_KMIX_ZERO_START_CONVEX_HULL_NO_GO`，转向 fixed learned warm start 或有真实 paired low/high-fidelity 数据后测试 AP-LOC。
- 页面“突破性进展”仍保持否；只有未见 rig/session、真实/外部数据、强基线、逐 camera tail、调用与端到端成本同时过门才可升级。
