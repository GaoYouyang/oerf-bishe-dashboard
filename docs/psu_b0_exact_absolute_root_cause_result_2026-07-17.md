# D0 exact-|A| 根因诊断：正式结果与算法转向

> 日期：2026-07-17  
> 证据角色：Gate B 严格 NO-GO 之后的 opened synthetic 数值机制诊断  
> 正式状态：`FACTOR_MAJORIZER_CANCELLATION_MATERIAL_DESCRIPTIVE`  
> 独立复核：`PASS_INDEPENDENT_EXACT_ABSOLUTE_DIAGNOSTIC_VALIDATION`，61 项检查通过

## 1. 这轮到底问了什么

前一轮 Factor Gate B 已经证明：voxel-factor PDHG 相对 scalar PDHG 的平均 field-L2 改善只有 1.321%，没有通过预注册的 25% 机制门。因此本轮不重开 Gate B，也不提出新算法，只追问一个根因：

> factor majorizer 的保守性，是否主要来自正负贡献在构造可分上界时被拆开，导致 `M` 相对 exact `|A|` 过松？

所有 PDHG 方法仍使用同一个 signed `A` 做前向、残差和伴随更新。`M` 或 `|A|` 只构造对角步长。比较对象是：

1. formal factor-view：factor 列和 + 逐视角最大行和；
2. factor-row hybrid：factor 列和 + 逐行 factor 行和；
3. exact-abs-view：exact `|A|` 列和 + 逐视角最大 exact 行和；
4. exact-abs-row：exact `|A|` 列和 + 逐行 exact 行和；
5. graph-PCGLS：只作 nonbinding headroom 参考，因为其 full-support 合同与 reduced-support PDHG 不同。

统计单位是两个 replicate 下的八种配对解析反应场形态，即 16 个配对场；它们不是 16 个独立同分布样本。

## 2. 为什么正式版是 v3

两次先行运行都在打开性能结果之前 fail closed：

- v1 在 MPS basis streaming 审计阶段暴露异步完成边界；没有产生性能行。
- v2 加入显式 `torch.mps.synchronize()` 后，重复 MPS basis replay 仍出现不可重复的活动数漂移/NaN；同样没有打开性能结果。
- v3 把 exact `A/M`、Schur 证书和 power estimate 移到 CPU float64；求解轨迹仍保留 MPS float32。协议、样本、阈值和求解器数学均未按性能结果修改。

CPU64 审计确认：

- M-active 坐标：2322；signed-A 非零坐标：2322；M-active/A-zero：0；
- `M >= |A|` 最大逐元素违反：0；
- factor 行和 replay 相对误差约 `9.14e-16`；列和约 `2.27e-16`；
- Schur 安全证书和设备/精度合同均通过。

## 3. 冻结端点 K=128 的主要数字

| 对照 | 相对 formal factor 的 normalized residual 平均改善 | 达到至少 10% 改善的配对场 | 解释 |
|---|---:|---:|---|
| factor-row hybrid | 2.011% | 0 / 16 | 单独取消逐视角最大行聚合影响很小 |
| exact-abs-view | 64.121% | 16 / 16 | 换掉 factor majorizer 后残差下降显著加快 |
| exact-abs-row | 64.183% | 16 / 16 | 比 exact-abs-view 只多 0.063 个百分点 |

高分位 factor slack 的中位数为 0.972832。换句话说，在诊断使用的高分位位置上，factor 上界相对 exact `|A|` 留下了非常大的松弛。结合上表，可以作出的最窄解释是：

> 在这个冻结的 same-signed-A 问题上，主要静态对角差异来自 `M` 与 exact `|A|` 之间的松弛，而不是把逐行信息压成 per-view maximum。

这仍是描述性机制定位，不是唯一 cancellation 因果机制的证明：exact-abs-view 同时改变 primal 和 dual 对角项，不能把所有差异归给一个单独数学操作。

## 4. 为什么 64% 不是“算法成功”

normalized data residual 加速并没有等比例转化为真实场恢复。平均 field relative-L2 为：

| 方法 | K=32 | K=64 | K=128 |
|---|---:|---:|---:|
| formal factor-view PDHG | 0.982876 | 0.967141 | 0.959944 |
| exact-abs-view PDHG | 0.919872 | 0.914658 | 0.914945 |
| exact-abs-row PDHG | 0.918036 | **0.911423** | 0.913594 |
| graph-PCGLS（nonbinding） | 0.421283 | **0.391071** | 0.421921 |

exact-abs-row 在 K=128 的 normalized residual 已降到 0.221037，formal factor 为 0.617131；按均值之比是 64.18% 的 residual 下降，但 field-L2 只从 0.959944 降到 0.913594，即 4.83%。若先对每个场计算百分比再平均，则对应数字是 64.97% 和 4.90%；二者是不同 estimand，必须分开标注。更重要的是，exact-abs-row 的 field-L2 从 K=64 的 0.911423 恶化到 K=128 的 0.913594，gradient relative-L2 从 1.074493 上升到 1.154849，而 residual 仍继续下降。front-F1 同期从 0.2335 上升到 0.2507，因此不同结构指标并非同向变化。

这提示三个后续瓶颈：

1. 更快拟合数据不能解决 BOST 逆问题的不可辨识性；
2. 噪声、正则化和停止规则开始比静态对角步长更重要；
3. 论文不能只报告 data residual，必须同时报告 field、gradient/front 和最坏样本。

graph-PCGLS 的 K=128 field-L2 为 0.421921，exact-abs-row 与它的误差比为 2.1653，但这个比较不具有绑定性：graph 使用 full-support Sobolev 表示，变量空间和先验方向不同。它只能说明“仍有很大 headroom”，不能归因成 Krylov 机制已被严格证明。

## 5. 当前允许与禁止的结论

### 可以说

- CPU64 审计下 `M >= |A|` 与 replay 合同成立；
- exact `|A|` 静态对角度量在 opened synthetic same-signed-A 问题上显著加快 normalized residual；
- factor-row 单独替换影响很小，因此主要静态差异来自 factor majorizer 与 exact `|A|` 的松弛；
- residual 加速只带来有限 field 改善，并出现 K=64 后的早停信号；
- 下一算法需要同时处理 metric、全局/history correction 与正则化/停止，而不是只学习一个更大的对角步长。

### 不能说

- 已提出优于 DeepONet、FNO、FFNO、NeRIF 或 graph-PCGLS 的新算法；
- exact `|A|` 已解决三维 BOST 重建；
- 结果已经在真实反应流、未见几何或 held-out camera 上泛化；
- 16 个配对场等于 16 个 IID 实验；
- `0.9728` 表示平均 factor 误差或 97% 的更新无效；它是 p05 tail tightness 对应的高分位 slack；
- Gate B 被重新打开，或 FM-CG-PDNO 已获得训练授权。

此外，当前 synthetic view scaling 来自 clean-truth projection RMS。solver 递推本身不直接读 field truth，但完整合成 pipeline 不是 truth-blind；在真实迁移前必须改用独立 calibration、flow-off repeats 或观测可得的归一化。结果还同时查看了多个方法、checkpoint 和指标，因此保持 exploratory/descriptive，不从 16 行构造 p-value 或置信区间。

## 6. 对下一版自有算法的直接约束

本轮把候选模型的最低结构要求缩小为三部分：

1. **cancellation-aware geometry metric**：读取射线覆盖、相机姿态和可观测 operator statistics，预测接近 exact `|A|` 行列质量的 metric；输出必须经过 Schur-safe clipping，且能回退到解析安全上界。
2. **global/history correction**：对局部 diagonal metric 无法表达的跨视角耦合，用低秩/global residual history correction 补充；必须与同调用 Krylov 基线公平比较。
3. **event/uncertainty-aware stopping**：用 discrepancy、跨视角残差一致性、front proxy 和 uncertainty 决定停止；禁止读取 truth 指标，必须在 held-out geometry/session 上校准。

这三部分应分阶段验证。第一阶段只逼近安全 metric 并做 held-out geometry；第二阶段才加入 global correction；第三阶段在拿到连续 4D 序列后加入时序停止。任何阶段如果强解析基线已经消除主要缺口，就停止增加网络容量。

## 7. 下一项可证伪实验

### 现在可做

- 在公开/合成多几何集合上训练一个小型 geometry-to-metric 网络；比较解析 factor、exact `|A|` teacher、随机几何留出和 Schur violation rate。
- 冻结 K=32/64/128，同时用不读 truth 的 residual/history 特征预测 K=64 附近的停止；评估 field/front regret 和最坏样本。
- 以 exact-abs metric 为 base，增加低秩 residual subspace correction；严格记账 signed `A/A^T` 调用和墙钟时间。

### 必须等师兄数据合同

- H2 rotation/optical mismatch 需要相机几何 provenance、rotation-40 作者 forward、mask 和 manifest；当前协议已冻结，但状态仍是 `UNCONSTRUCTED_FROZEN_DEVELOPMENT_ONLY_H2_DIAGNOSTIC`。
- 真实论文门需要 flow-off repeats、phantom/paired renderer 或连续 timestamp/exposure/dropout 序列，并按 geometry/session 做永久留出。

## 8. 复核入口

```bash
PYTHONPATH=. .venv/bin/python site_tools/validate_psu_b0_exact_absolute_diagnostic.py \
  --config demo_t16_operator/configs/psu_b0_exact_absolute_root_cause_v3_cpu64_audit_amendment.json \
  --evidence demo_t16_operator/results/psu_b0_exact_absolute_root_cause
```

正式产物：

- `demo_t16_operator/results/psu_b0_exact_absolute_root_cause/report.json`
- `demo_t16_operator/results/psu_b0_exact_absolute_root_cause/trajectory_rows.csv`
- `demo_t16_operator/results/psu_b0_exact_absolute_root_cause/tightness_rows.csv`
- `demo_t16_operator/results/psu_b0_exact_absolute_root_cause/audit_rows.json`
- `demo_t16_operator/results/psu_b0_exact_absolute_root_cause/validation_report.json`
- `demo_t16_operator/results/psu_b0_exact_absolute_root_cause/checksums.sha256`
