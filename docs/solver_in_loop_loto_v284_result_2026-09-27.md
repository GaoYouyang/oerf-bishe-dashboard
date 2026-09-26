# v284: Accuracy passes; strict cost advantage fails

**2026-09-27 · Frozen five-fold complete-trajectory LOTO · Clean nine-camera PoolFire proxy**

## Verdict

`FAIL_SOLVER_IN_LOOP_LOTO_STRICT_COST`. The independently recomputed numerical and physical checks pass, and all 505 held-out frame pairs satisfy the four 1% accuracy gates. The stricter cost contract does not: the learned initialization is strictly cheaper than every required control on 453/505 pairs, but at least one same-AMG control is no worse on the other 52. Only one of five complete held-out trajectories passes every frame.

This is a valid, mixed cost result—not a stable complete-trajectory advantage, speedup, algorithm breakthrough, external generalization, or real BOST result.

## Experiment and audit

A small shared observation/geometry-conditioned map predicts one warm initial field. It is fitted separately in five leave-one-trajectory-out folds from the other four trajectories, using train-only solver-derived targets. Each fold has 404 training and 101 held-out frames. The refinement remains the original AMG-PCGLS solver with its frozen stopping certificate; held-out CFD truth is not used by the predictor or solver.

The fit reached its frozen 200-iteration limit in each fold. A separate implementation recomputed the initial and terminal full-training losses and gradients. The sealed prediction and scoring chain completed with exit code zero; all 26 numerical, physical, certificate, and accounting checks passed. A separate post-exit reducer independently reconstructed the 505-point comparisons and trajectory tails; its summaries matched exactly. The final result and its input/output closure remained unchanged.

All 505 query pairs pass field, full-gradient, interior-gradient, and observation accuracy. Across the five trajectories, the respective p90 ranges are 0.000459–0.000485, 0.000346–0.000451, 0.000661–0.000682, and 5.74e-6–7.42e-6; all are below the frozen 0.01 limits.

| Held-out fold | Strictly cheaper than every control | Complete fold passes? |
|---:|---:|:---|
| 1 | 94/101 | No |
| 2 | 97/101 | No |
| 3 | 101/101 | Yes |
| 4 | 67/101 | No |
| 5 | 94/101 | No |
| **Total** | **453/505** | **1/5** |

The mandatory same-AMG comparisons include Zero, BP, fixed-tensor, one-V-cycle, and historical dual-ridge controls. All 52 rejected pairs are caused by at least one same-AMG control; the additional favorable idealized classical lower-cost comparisons cause none of these rejections. Historical dual ridge is no worse on 42 of the 52 rejected pairs. A finer post-hoc breakdown finds dual-ridge is the sole failing control on 29 pairs, overlaps another same-AMG failure on 13, and is absent from the other 10. In its 42 no-worse cases, the candidate ties dual-ridge on both A and A^T counts on 15 pairs and uses more of both on 27. Control incidences overlap. Failures are concentrated in held-out fold 4 (34/52).

A post-hoc equal-thirds chronology check finds 14/34, 11/34, and 9/33 cost failures in the early, middle, and late portions of fold 4; the other four folds have 7, 4, 0, and 7 failures in total. This argues against describing the result as only a late-frame tail, while the strong fold concentration points to a trajectory-level generalization problem that this table alone cannot explain. The thirds were chosen after opening the result, are descriptive only, and do not change the frozen all-frame/all-trajectory gate.

## Interpretation and boundary

The result supports a narrower observation: the fitted initializer can preserve the required final accuracy and beat all listed controls on many individual frames in this opened clean nine-camera proxy. It does **not** establish a stable trajectory-level reduction in exact calls. No matched fresh-process wall-time or whole-pipeline RSS comparison was performed, so no speed or resource advantage is claimed.

Close this exact recipe. Do not tune its optimizer budget, loss, stopping rule, or model size after seeing the held-out result. Any later proposal needs a physically distinct, prospectively frozen mechanism and controls; this result alone does not authorize scaling the model or renting a GPU. Five opened trajectories in one public proxy setting are not an untouched external test or a real experimental BOST transfer.

## 中文摘要

正式判决：`FAIL_SOLVER_IN_LOOP_LOTO_STRICT_COST`。独立复算与物理检查有效，505 个留出帧对全部通过场、全梯度、内部梯度、观测四项 1% 精度门；但严格成本门要求每一帧在 A 与 A^T 两项调用上都优于全部对照。候选仅在 453/505 帧对严格更便宜，另有 52 帧至少被一个同 AMG 对照追平或反超。五条完整留出轨迹只有 1/5 全程通过。

五折按完整轨迹留一；每折 404 帧训练、101 帧留出。模型部署时只读观测和报告几何，训练目标只来自该折训练轨迹的求解器派生量；未改动的 AMG-PCGLS 与停止证书负责后续精化。独立训练损失/梯度验证、预测评分链和退出后复核均成功，26 项数值、物理、证书与调用账检查全部通过，退出后重建的摘要与封存结果完全一致。

按折统计严格优于全部对照的帧数依次为 94/101、97/101、101/101、67/101、94/101。四指标 p90 在五折上的范围分别为：场 0.000459–0.000485、全梯度 0.000346–0.000451、内部梯度 0.000661–0.000682、观测 5.74e-6–7.42e-6，均低于冻结的 0.01 精度门。

52 个失败帧对全部由同 AMG 对照造成，而不是更有利的理想化经典成本下界；其中 42 个帧对的历史 dual-ridge 已不比候选更贵。进一步拆分：29 个只被 dual-ridge 单独触发，另 13 个同时涉及其他同 AMG 对照，剩余 10 个不涉及 dual-ridge。42 个 dual-ridge 反例中，候选有 15 个在 A 与 A^T 调用数上都打平，27 个两项调用都更多；各对照触发数存在重叠。失败集中在第 4 折（52 个中 34 个）。这些是从封存逐帧表作出的事后归因，不改变冻结门槛。

结果打开后按每折帧序等分早/中/晚的探索性检查显示：第 4 折三段分别有 14/34、11/34、9/33 个成本反例；其余四折合计分别有 7、4、0、7 个。失败不能简单概括成“只在后段恶化”，但强烈的整轨迹集中也不能由这张表单独解释。该等分是事后描述，不是预注册分层，不改变冻结门槛。

因此这不是“没有学习信号”：很多单帧上有条件性的调用优势；但它没有达到预注册的稳定完整轨迹优势，当前配方应关闭。没有做匹配的 fresh-process 墙钟时间或全流程 RSS 对照，不能称为加速或资源优势；也不是外部泛化、真实 BOST、算法突破或论文成功。不得根据留出结果追加训练轮数、改损失或放大模型来追过门槛。
