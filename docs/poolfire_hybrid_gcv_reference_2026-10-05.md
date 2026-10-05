# Hybrid-GCV 参考资格审查 / Hybrid-GCV Reference Qualification

日期 / Date: 2026-10-05

## 中文结论

**独立实现一致，但固定参考资格不通过。** 重正交的 projected-GCV Hybrid LSQR 在 11 条已开封 PoolFire 训练轨迹、每条 101 帧和 5/7/9 相机上覆盖 3,333 个单元、23,331 条检查点评分。36 个分层是 33 个轨迹×相机层，加 3 个按相机合并层；它们不是 36 组独立样本，也不是新的未见数据测试。

- 128 步的 field/full-gradient/observation 绝对 p90 门分别为 0.50/0.75/0.20，**36/36 分层通过**。
- 冻结的参考稳定性要求：64 到 128 步，field/full-gradient/interior-gradient/observation 四项 p90 的绝对变化在每层均不超过 0.001。**0/36 分层通过**，四项指标各自都在全部 36 层超门。
- 两套独立算子/求解实现的全部保存状态及物理指标已核对。恢复审查 13/13 项通过；逐检查点状态最大相对差为 3.06e-11，物理指标最大绝对差为 1.84e-11，分层汇总最大差为 1.39e-12。离散判决完全一致。

| 64 到 128 步的变化 | 36 层中最大 p90 绝对变化 | 冻结上限 |
|---|---:|---:|
| 场 | 0.049488 | 0.001 |
| 全梯度 | 0.079524 | 0.001 |
| 内部梯度 | 0.083842 | 0.001 |
| 观测 | 0.005240 | 0.001 |

### 做了什么，为什么

此前未重正交的实现未通过数值资格检查。另行冻结的本次实现使用完整基的双遍重正交，以及与之对应的稠密投影问题；它通过数值资格后才进行本次场评分。Hybrid LSQR/GCV 是经典对照，不是新提出的学习算法。这里要回答的是：它能否提供一个满足冻结精度与稳定性要求的参考，供后续 warm start 公平比较。

独立运行已经完成全部 3,333 个单元，但最终汇总遗漏了外层记录的轨迹/相机标签，没有生成原完成记录。审查在新的目录中仅恢复内存中的标签视图，再核对全部原始输出和数据身份；没有修改原文件、数值、阈值或原完成记录，也没有重跑求解器。源码缺陷已在纯元数据记录上复现；没有保留下来的历史异常日志，因此不声称看到过当时的异常堆栈。

**补齐汇总是工程修复；经独立一致性核验后拒绝参考资格，才是本轮科学判断。** 本次恢复新增 0A+0A^T。原算法在 128 步的逻辑账为每样本 128A+128A^T，另有几何准备、重正交、投影 SVD/GCV 和基向量存储成本，不能把这些成本当成免费或以逻辑调用账替代实测速度。

### 对路线的影响

这关闭本次固定 Hybrid-GCV 参考尝试，不追加深度、不改阈值或正则化参数来救援。参考未合格时，较少步数的检查点只能作描述性诊断，不能据此声明 matched-accuracy 调用节省。该结论不证明全部 Hybrid 方法或 C 路线不可能，也不允许把不同历史算子合同下的数字直接合并。

下一项比较必须先有独立可解释的质量目标或合格参考、部署时可见的停止规则，以及同价或更便宜的经典控制。不能将“都通过宽泛绝对门”偷换为“四项误差匹配”。现有三维场仍能继续受控虚拟研究，找不到的旧二维实验配对不是虚拟研究的前置条件；像素级/真实 BOST 结论仍需要实际光学尺度和实验依据。

未打开 validation/test/独立外部数据；未训练新的预测器；没有 wall/RSS、外部泛化、真实 BOST、算法突破或论文成功结论。

## English Result

**Independent implementations agree, but this fixed reference fails qualification.** Reorthogonalized projected-GCV Hybrid LSQR covers 3,333 cells and 23,331 checkpoint scores from 11 already-opened PoolFire training trajectories, 101 frames each, and 5/7/9 cameras. The 36 strata comprise 33 trajectory-camera strata and three camera-pooled strata; they are not 36 independent datasets or a fresh generalization test.

At 128 steps, all **36/36 strata** pass absolute p90 limits of 0.50 for field, 0.75 for full gradient and 0.20 for observation. However, **0/36** pass the frozen reference-stability requirement: the absolute p90 change from 64 to 128 steps must not exceed 0.001 for any of field, full gradient, interior gradient or observation. Each metric exceeds that limit in every stratum. Their maximum changes are respectively **0.049488, 0.079524, 0.083842 and 0.005240**.

All stored states and physical metrics from the separate operator/solver implementations were compared. All 13 recovery checks pass. Maximum per-checkpoint state relative, physical-metric absolute and summary absolute differences are **3.06e-11, 1.84e-11 and 1.39e-12**. Every discrete decision agrees. Numerical consistency does not imply a stable reconstruction reference.

### Recovery and Cost

The earlier implementation without reorthogonalization failed numerical eligibility. This separately frozen attempt uses two full reorthogonalization passes and the corresponding dense projected problem, passing numerical eligibility before field scoring. Hybrid LSQR/GCV is a classical control, not a new learned method.

The independent run computed all 3,333 cells, but its final summarizer omitted trajectory/camera labels present in enclosing records. Recovery reconstructed only an in-memory metadata view and verified all original outputs and input identities in a separate audit. Original files, numerical values, gates and original receipts remain untouched; no solver was rerun. A metadata-only record reproduces the source defect. No historical exception log survives, so no original traceback is claimed.

Summary recovery is engineering; the independently checked reference rejection is the scientific decision. Recovery adds **0A+0A^T**. The original 128-step algorithm logically requires **128A+128A^T per sample**, excluding nonfree geometry setup, reorthogonalization, projected SVD/GCV and basis storage. No fresh wall-time or RSS measurement was performed.

### Consequence and Limits

Close this fixed reference attempt without increasing depth or changing thresholds or regularization to rescue it. With an unqualified reference, lower-step checkpoints are descriptive only, not evidence of matched-accuracy call savings. This does not rule out all Hybrid methods or the C route, and results from different historical operator contracts must not be pooled as if directly comparable.

Any next comparison needs an independently justified quality target or qualified reference, a deployment-visible stopping rule and equal-or-cheaper classical controls. Passing broad absolute gates is not four-metric accuracy matching. Existing 3D fields still support controlled virtual research; lost experimental 2D pairs are not a prerequisite for that work. Pixel-calibrated or real-BOST claims still require optical scale and experimental evidence.

No validation/test/independent external data were opened and no new predictor was trained. Algorithmic breakthrough, paper success, external generalization, resource speedup and real BOST all remain **false**.

[Redacted numerical summary](poolfire_hybrid_gcv_reference_2026-10-05_public_summary.json)
