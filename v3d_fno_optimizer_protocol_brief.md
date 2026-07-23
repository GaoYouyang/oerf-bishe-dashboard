# v3d FNO 优化协议与固定 Epoch 基线简报

更新时间：2026-07-12

## 一句话判决

**FNO 内部的固定 epoch 开发基线已经锁定为 `carry continuation-Adam + restart cosine` 的 240-epoch K=6 ridge-FNO；它的 validation 最好，但仍未 plateau。geometry 为 Go 后可以开始代码/功能 pilot；不同架构比较前仍须补 FLOPs、峰值内存、wall time 与 time-to-target，不能声称已经击败“饱和 FNO”。**

## 为什么不能只把训练继续拉长

上轮把 FNO 从 24 延长到 96 epochs 时，每个 12-epoch block 都重新初始化 AdamW 与 cosine scheduler。它证明旧 FNO 训练不足，却没有回答两个问题：

1. 改善来自重新获得较大学习率，还是来自 Adam moments？
2. 一条已经 plateau 的曲线，是否一定是 validation 最强曲线？

本轮让三条策略从每个种子完全相同的 24-epoch validation checkpoint 出发，并给每个 block 相同 batch seed：

| 策略 | Adam moments | cosine horizon | 它回答什么 |
|---|---|---|---|
| restart Adam + restart cosine | 每 12 epochs 重置 | 每 12 epochs 重置 | 复现历史式分段追加训练 |
| carry continuation-Adam + restart cosine | 只在 continuation blocks 间保留 | 每 12 epochs 重置 | 单独看 continuation moments 是否有益 |
| carry continuation-Adam + long cosine | 只在 continuation blocks 间保留 | 覆盖完整 continuation | 长程衰减是否先达到 plateau |

三条 continuation optimizer 都在共同 epoch-24 model checkpoint 之后新建，**没有恢复 base 训练阶段的 Adam moments**；`carry` 只表示随后 18 个 continuation blocks 之间是否保留 moments。训练状态从每个 block 的 raw endpoint 继续；用于公开比较的 checkpoint 是截至该 block 的 validation prefix-best。三条策略全部完成后，本轮才计算复用 dev2 field 与 `Q_audit`；该 synthetic dev2 在 v3c 及早期分析中已经查看过，不是整个项目从未触碰的独立审计集。

每个 strategy-seed-block 都记录绑定实际 train indices 的 batch-order contract hash；validation 对每个三维场等权，最后 4-sample batch 不再与 12-sample batch 等权。机器报告同时记录 experiment config、dataset config、训练脚本、共享 train/eval 依赖与本地 synthetic NPZ 的 SHA-256。NPZ 与 checkpoint 权重不公开，hash 只用于确认复跑对象未漂移。

## Validation-only 结果

| 策略 | epoch 240 mean validation L2 | 最后 block mean / max-seed 改善 | plateau | 平均本机训练时间/seed |
|---|---:|---:|---|---:|
| restart Adam + restart cosine | 0.099472 | 0.862% / 1.134% | 否 | 45.68 s |
| **carry continuation-Adam + restart cosine** | **0.094139** | **0.905% / 0.961%** | **否，validation 冠军** | 46.46 s |
| carry continuation-Adam + long cosine | 0.095726 | 0.014% / 0.027% | 是；onset 192 | 43.35 s |

validation 冠军相对 restart-both 低 `5.36%`，相对已经 plateau 的 long-cosine 低 `1.66%`，相对共同 epoch-24 base 低 `43.45%`。

**关键认识：plateau 只说明在某套学习率日程下不再明显改善，不等于它是最强 baseline。** long-cosine 已 plateau，却不是 validation 冠军；因此不能为了尽快“过闸门”选择一条较弱但更早停下的曲线。

## Dev2 后置诊断

这些复用 development diagnostics 在本轮 validation 冠军冻结后才计算，不参与本轮策略或 checkpoint 选择：

| 策略 | domain-equal field L2 | 相对 epoch-24 base field 改善 | clean Q_audit L2 |
|---|---:|---:|---:|
| restart Adam + restart cosine | 0.304178 | +20.91% | 0.321964 |
| **carry continuation-Adam + restart cosine** | **0.303196** | **+22.09%** | 0.322595 |
| carry continuation-Adam + long cosine | 0.303700 | +21.68% | 0.322350 |

配对 field-cluster 诊断显示：

- validation 冠军相对 restart-both 平均 `+2.08%`，95% CI `[+1.75%, +2.38%]`，但 p10 为 `-1.22%`、`16.4%` 体场退化超过 1%，最差 `family OOD` 域为 `-0.78%`。
- validation 冠军相对 long-cosine 平均 `+0.72%`，95% CI `[+0.65%, +0.79%]`，没有体场退化超过 1%，但 p10 为 `-0.32%`，最差 `family OOD` 域为 `-0.13%`。

所以 optimizer protocol 自身就能制造约 `0.7%–2.1%` 的平均排名差，并且尾部与 family OOD 可能反向。以后若自有模型只赢 1% 左右，不能直接解释成架构创新。

## 两个不同的闸门

### A. FNO 固定 epoch 基线闸门：通过

为了不无限等待 plateau，现在冻结以下开发合同：

1. FNO 开发基线使用 `carry continuation-Adam + restart cosine`，最多 240 attempted epochs。
2. 所有候选用同一输入、train/validation split、batch seeds 与 validation-only prefix-best 规则。
3. 本轮已经报告 endpoint error、完整 learning curve 与训练 wall time；跨架构比较前必须新增参数量、FLOPs/峰值内存和 time-to-target。
4. 自有模型必须同时比较 validation 冠军和已经 plateau 的 long-cosine control。
5. 复用 dev2 只在本轮结构、超参数与 checkpoint 冻结后计算；新的 untouched blind final 继续封存。

这允许开始 **代码、接口和功能 development pilot**，还不允许做跨架构公平性结论。

### B. 跨架构算力与论文 superiority 闸门：未通过

以下任一缺失都不能申请 blind final：

- validation 最强 FNO 尚未 plateau，或候选没有在多档 epoch、wall time、FLOPs 与内存预算上形成稳定 Pareto 优势；
- geometry/data manifest 未确认真实采集几何确实跨 case 变化；
- 没有 F-Adapter、LoRA、last-block、CGLS/TV/RBF 与统一 NeRIF refinement；
- family OOD、p10、harm、`Q_audit` 或三种子出现反向；
- 没有独立 nonlinear/cone-ray forward 或真实组内审计信号。

## 现在真正值得做的自有算法实验

在何远哲确认 geometry 为 Go 后，第一版只做一个小而可证伪的机制：

`ridge field + per-view ray/mask/angle/quality -> shared FNO -> acquisition-conditioned frequency coefficients`

必须同时运行：

- carry-continuation-Adam/restart-cosine FNO；
- carry-continuation-Adam/long-cosine FNO；
- vanilla adapter、LoRA、F-Adapter 与 last-block；
- correct / shuffled / constant / static geometry；
- 60 / 120 / 180 / 240 epoch checkpoints；再映射到 wall time、FLOPs、内存与 time-to-target。

最有论文价值的结果不一定是 epoch 240 单点最低，而可能是：更快达到同一误差、在缺失相机/标定漂移下保持稳定、或以更少参数和 wall time形成 Pareto 优势。

## 给何远哲的三个判断题

1. 是否认可把 `carry continuation-Adam + restart cosine · 240 epochs` 冻结为 FNO epoch 基线，同时保留 long-cosine plateau control？
2. 是否允许在未达到确认性 plateau 前，以同算力 learning-curve/Pareto 方式开始自有模型 pilot，但继续关闭 blind final？
3. 组内 geometry 是否跨 case、缺失视角或标定漂移变化，足以让 acquisition-conditioned 方法通过 Go 条件？

## 可复跑入口

- 配置：`demo_t16_operator/configs/v3d_fno_optimizer_audit.json`
- 训练：`demo_t16_operator/run_v3d_fno_optimizer_audit.py`
- 验证：`python -m demo_t16_operator.validate_v3d_fno_optimizer_results`
- 机器报告：`demo_t16_operator/results/v3d_fno_optimizer_audit/v3d_optimizer_report.json`
- validation 表：`demo_t16_operator/results/v3d_fno_optimizer_audit/v3d_optimizer_validation_summary.csv`
- 策略配对诊断：`demo_t16_operator/results/v3d_fno_optimizer_audit/v3d_optimizer_pairwise_summary.csv`
