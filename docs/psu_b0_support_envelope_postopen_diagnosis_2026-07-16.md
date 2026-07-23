# PSU 视角支持域回退：机制有效，方法仍未通过

> 证据等级：L2 post-open mechanism diagnosis + L3 implementation audit
> 状态：`POSTOPEN_SUPPORT_ENVELOPE_DIAGNOSIS_COMPLETE_NOT_FRESH`
> 禁止外推：fresh 泛化、实验三维真值、算法优越性

## 1. 为什么做这个诊断

首轮 2,227 参数正定谱预条件器在 IID 相对强 inverse-Sobolev 提升约 4%，但 joint OOD 三种子全部退化。joint OOD 同时只有 4–5 个 active views，而训练只见过 6–9 个 views。

因此先测试一个最小机制，不重新训练、不扫阈值：

\[
P_{\mathrm{env}}=
\begin{cases}
P_\theta, & 6\le N_{\mathrm{view}}\le9,\\
P_{\mathrm{Sobolev}}, & \text{otherwise}.
\end{cases}
\]

这不是新的 fresh candidate。它使用已经打开的同一批 split 和原 checkpoint，只问：**视角数越界是不是 joint OOD 伤害的一个可观测触发条件？**

## 2. 实现合同

- 三个原始 checkpoint 不重训；
- train support 固定为 6–9 views；
- 支持域内逐样本选择原 learned direction；
- 支持域外逐样本选择 validation-selected Sobolev；
- 使用布尔 `torch.where`，避免 `fallback + τ(candidate-fallback)` 在 `τ=0/1` 时留下浮点舍入；
- learned raw 与 enveloped 都是 `4F+4Aᵀ`；
- 每步仍使用精确伴随和解析线搜索；
- rotation 40 与 final audit 均未打开。

端到端连续指标容差为 `1e-6`；top-10% front F1 是离散阈值指标，MPS 两次运行可能因一个边界体素产生约 `3.05e-4` 差异，因此冻结容差为 `5e-4`。方向选择本身另有逐值单元测试。

## 3. 结果

### 支持域覆盖

| split | active views | learned correction coverage |
|---|---:|---:|
| IID | 6–8 | 100% |
| family OOD | 6–8 | 100% |
| noise OOD | 6–8 | 100% |
| view OOD | 4–5 | 0% |
| joint OOD | 4–5 | 0% |
| exact-operator control | 6–8 | 100% |

### 关键变化

- view OOD：原 learned mean gain `+1.41%` 到 `+1.77%`，但有尾部伤害；回退后均值约 0、`>1% harm=0`；
- joint OOD：原 learned mean gain `-0.43%` 到 `-0.20%`、三种子 `harm=33.3%`；回退后均值约 0、`harm=0`；
- IID/noise/exact control：与原 learned 结果等价，约保留 4% 正信号；
- family OOD：因为仍在 6–9-view support 内，原风险完全保留，p10 为负，`harm=20.8%–25%`。

## 4. 应怎样解释

这轮证明的是：

1. active-view count 是 joint OOD 伤害的一个有效、可观测触发变量；
2. exact fallback 可以保证该支持域外不比 Sobolev 更坏；
3. joint OOD 的 0% harm 来自 0% learned coverage，不是模型在 OOD 上产生正增益；
4. 视角数不能识别 6–8-view 内的未见形态，所以 view-only gate 不足以成为下一代算法。

正式结论：

> **回退机制通过实现门，view-only 方法没有通过科学门。**

## 5. 下一代算法必须增加什么

下一候选仍用

\[
P_{\theta,\tau}
=P_{\mathrm{Sobolev}}
+\tau(z)(P_\theta-P_{\mathrm{Sobolev}}),
\qquad 0\le\tau\le1,
\]

但 `z` 至少要包含：

- active-view support margin；
- 各视角白化 residual 的均值、最大值和离散度；
- candidate 相对 Sobolev 的可观测 residual-risk proxy；
- 迭代阶段和 correction magnitude；
- camera dropout、相关噪声和 thin-front stress 下的训练内风险。

下一次必须在新 `oblique-shell / triple-front`、相关噪声、3-view 和新随机种子上冻结评估。最低门槛不是简单“joint harm=0”，而是：

1. fresh IID 保留至少 2% 均值增益；
2. fresh family/joint p10 不为负；
3. `>1% harm ≤5%`；
4. correction coverage 不能靠全回退降到 0；
5. 同 `F/Aᵀ` calls，并报告 wall time；
6. 真实阶段只用 held-out reprojection 与 repeatability 选模，不用不存在的实验 field truth。

## 6. 可复核入口

- [严格公开摘要](psu_b0_support_envelope_postopen_public_summary.json)
- [四联图 PNG](../demo_t16_operator/results/psu_b0_support_envelope_postopen/psu_b0_support_envelope_postopen_figure.png)
- [四联图 PDF](../demo_t16_operator/results/psu_b0_support_envelope_postopen/psu_b0_support_envelope_postopen_figure.pdf)
- [回退实现](../demo_t16_operator/psu_b0_spectral_preconditioner.py)
- [诊断 runner](../site_tools/run_psu_b0_support_envelope_diagnosis.py)

原 checkpoint、逐样本指标和本机路径只保存在忽略 Git 的私有报告中。
