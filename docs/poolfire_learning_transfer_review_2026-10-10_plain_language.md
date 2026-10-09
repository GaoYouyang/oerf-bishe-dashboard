# 2026-10-10 学习收益与精化：阶段复盘

## 中文

这次复盘把三件事分开：学习有没有作用、能否超过强同价对照、是否有论文级完整收益。

**学习有小幅最终作用。** 在11条已打开PoolFire train轨迹、三帧、5/7/9相机的99个训练内样本上，小型共享参数递归初值训练后的最终场、全梯度、内部梯度和观测误差全部比自身未学习版本更低。四项相对改善中位数分别0.2801%、0.3295%、0.2134%、2.7517%。初值四项教师损失下降26.62%，但这不等于最终误差同幅下降；所有样本都是训练角色，不是LOTO或完整序列。

**精化不是简单在抵消学习。** 另行冻结一次不训练的诊断：从未学习初值记录原CGLS31的系数，再把同一历史作用于已封存学习初值。正式与独立实现分别重建状态、物理图像及有符号损失闭包。原算法实际系数适应使全部99个平均教师终点损失比冻结历史更低；相对未学习版本的终点教师损失改善中位数2.4601%，冻结历史为0.9434%。因此目前不能把失败概括成“求解器破坏学习”。

冻结历史与实际终点的场差最大0.5963%，未达到结果前1e-7数值等同门。这只说明具体历史不是精确有限转移替代，没有检验近似梯度是否有用。保留带符号交叉项，不宣称独立因果比例。有限CG可微性研究提供问题动机，但不能直接保证本奇异BOS法向与浮点实现：[Gratton et al., 2014](https://epubs.siam.org/doi/10.1137/120889848)。

**强对照仍否决当前固定配方。** 每查询35对A/A^T的学习版本在全部99样本至少一项不如同成本冷CGLS35，四指标同参考为0/99，基本绝对质量分层33/33。停止本固定参数量、阶段、更新与精化配方，不增加训练、换阈值或扩大网络救援。有限正信号不是部署算法成功，也没有证明全路线不可能。

已有3333样本、33完整分层的经典同精度链仍保留；公平批量运行没有稳定时间或联合内存优势。训练与归因分别约169.27和49.01秒、峰值约2.005和0.688GiB，仅是研究预算，不能当成fresh部署性能。

下一投入围绕能保留到精化终点的有效逆作用，并优先检验最强同价对照。已有可交付的物理接口、完整经典基线和学习对照应收拢；停止只追初值loss、相似小变体和重复整站发布。当前没有新算法突破、资源优势、外部泛化或真实BOST结论。

## English

Separate whether learning has an effect, whether it beats a strong equal-cost control, and whether it delivers complete paper-level benefits.

**Learning has a small endpoint effect.** On 99 in-sample anchors from eleven opened PoolFire train trajectories, three frames and 5/7/9 cameras, a small shared-parameter recursive initializer improves actual field, full-gradient, interior-gradient and observation errors over its own unfitted version on every cell. Median relative gains are 0.2801%, 0.3295%, 0.2134% and 2.7517%. A 26.62% seed teacher-loss decrease is not the same decrease in endpoint error. Every anchor has a training role: no LOTO or complete-sequence claim.

**Refinement does not simply cancel learning.** A separately frozen no-fit attribution records original CGLS31 coefficients from the unfitted seed and applies that history to the sealed fitted seed. Formal and independent implementations reconstruct states, physical images and signed loss closure. Actual coefficient adaptation gives lower mean teacher endpoint loss than frozen history on all 99 cells. Median relative teacher endpoint improvement over the unfitted arm is 2.4601%, versus 0.9434% with frozen history. The present evidence therefore does not support blaming refinement for destroying learning.

The maximum field discrepancy between frozen history and the actual endpoint is 0.5963%, failing the predeclared 1e-7 numerical-identity gate. This rejects that particular exact finite-transfer substitute; it does not test approximate-gradient usefulness. Retain signed cross terms without causal percentages. Finite-CG differentiability motivates the question, not a guarantee for this singular BOS normal and floating-point implementation: [Gratton et al., 2014](https://epubs.siam.org/doi/10.1137/120889848).

**The strong control still rejects the fixed recipe.** At 35 A/A^T pairs per query, the learned arm is worse on at least one error than equal-call cold CGLS35 on all 99 cells. Four-error matches are 0/99 despite 33/33 basic absolute strata. Close the fixed parameters, stages, updates and refinement recipe without extra fitting, threshold changes or larger-network rescue. This limited signal is neither deployed success nor impossibility of the whole route.

Retain the classical matched-accuracy chain on 3333 cells and 33 complete strata, but fair batched execution has no stable time or joint memory benefit. Fitting and attribution take approximately 169.27 and 49.01 seconds, with peaks about 2.005 and 0.688GiB: research budgets, not fresh deployment performance.

Prioritize effective inverse action retained at the refined endpoint and the strongest equal-cost controls. Consolidate the deliverable physical interface, complete classical baseline and learned comparisons. Stop chasing seed loss, similar tiny variants and repetitive full-site publication. No new algorithm breakthrough, resource benefit, external generalization or real-BOST result follows.

[去隐私汇总 / Privacy-safe summary](poolfire_learning_transfer_review_2026-10-10_public_summary.json)
