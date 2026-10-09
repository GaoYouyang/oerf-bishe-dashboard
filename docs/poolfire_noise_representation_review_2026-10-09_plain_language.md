# 2026-10-09 全局复盘：表示能力、噪声与实际成本

## 中文

主目标没有变：从多相机观测与几何预测有用的初值，用精确物理算子和未改动的 CGLS/PCGLS 精化，在同精度下获得实际资源收益。经典控制的成功不能替代学习型方法的成功。

**噪声不是只要把观测拟合得更紧就能解决。** 在11条已打开训练轨迹的99个三帧哨兵、5/7/9相机上，分别施加0.1%与1%的受控相对观测噪声，共198样本。直接经典初值的四指标匹配从99/99降为2/99；两档的基础绝对精度分层仍全部通过。在1%档，88/99样本出现观测误差更低、真实场误差反而更高。这是归一化的虚拟噪声，不是实验噪声，也不是完整时间序列。

**当前观测交互初值没有稳定超过便宜对照。** 每折排除整条轨迹；拟合全部封存后才预测，预测全部封存后才用真值评分。99个留出哨兵通过33/33基础绝对精度分层，但四指标联合匹配为0/99。它与线性控制的逐样本四指标Pareto胜出数为29对62，不能称学习突破。每个候选逻辑成本为34A+33A^T，几何准备、训练和存储并非免费。这个固定配方关闭，不扩成更大的同族网络。源闭包另有事后核验：最初绑定遗漏的继承辅助文件与此前封存字节一致；原运行的遗漏不隐去，也不事后改合同。

**继续优化旧读出系数不是主要突破口。** 另一次双实现归因允许每个样本使用教师解，在原有初值特征内选最优系数。联合场/图像拟合损失的空间外占比，逐样本中位数为88.586%，平均为84.776%。因此这套特征内仅调整读出仍有主要缺口。这个目标是教师拟合平方损失，不是实际密度误差的88.6%不可恢复，更不能当作精化后CGLS误差的下界。没有新部署训练或新精化。

三项独立核验分别19/19、13/13、12/12通过。数值复算通过与科学方法成功是两回事。

**资源也要看完整工作负载。** 既有3333样本、33完整分层的经典初值仍保持同精度；但公平批量执行的总时间为89.19秒，对照88.85秒，峰值RSS约3.891对0.697GiB，没有稳定时间或联合资源优势。旧逐个右端工作负载约16%的时间改善仅保留在其原范围，不再作为最新无条件结论。

下一投入优先级：不同的有效逆表示，同时保护有效信号与控制噪声放大；先用有限、可证伪的证据检验，再决定是否训练和扩展。停止同族读出微调、重复跑已关闭配方和为小负结果逐次发布页面。当前没有算法突破、外部泛化、真实BOST或论文成功结论，也没有证明整个C路线或所有注意力方法不可能。

## English

The goal remains an observation/geometry-only learned initializer, exact physical lift and unchanged CGLS/PCGLS refinement, with actual resource gains at matched accuracy. Successful classical controls cannot substitute for learned-method success.

**Tighter observation fit is not sufficient under noise.** Apply controlled relative observation noise of 0.1% and 1% to 99 three-frame anchors from eleven opened train trajectories with 5/7/9 cameras:198 cells. Direct classical initialization falls from 99/99 to 2/99 four-error matches, despite passing all basic absolute sampled strata at both levels. At 1%, 88/99 cells have lower observation error but higher actual field error. This is globally normalized virtual noise, not measured experimental noise or a full temporal sequence.

**The current content-conditioned initializer does not stably beat its cheaper controls.** Each fit excludes an entire trajectory; all fits seal before held prediction, and all predictions seal before truth scoring. The 99 held anchors pass 33/33 basic absolute strata but 0/99 joint four-error matches. Its per-cell four-error Pareto wins against the linear control are 29 versus 62 in the reverse direction. Logical candidate cost is 34 A+33 A^T; geometry setup, fitting and storage remain nonfree. Close this fixed recipe without expanding it into a larger same-family network. A separate post-run source audit confirms that inherited helper bytes omitted from the original binding match earlier seals; disclose the omission and retain the original contract.

**Further optimization of the old readout is not the main opportunity.** An independent post-open audit allows teacher-visible optimal coefficients within the same seed features. The per-sample median fraction of joint field/image fit loss outside that span is 88.586%, and its sample mean is 84.776%. This is a teacher-fit squared-loss attribution, not a claim that 88.6% of actual density error is irrecoverable or a lower bound on refined CGLS endpoints. No new deployed fit or refinement is performed.

The three independent validations pass 19/19, 13/13 and 12/12 checks respectively. Numerical reproduction does not mean method success.

**Resources must cover the whole workload.** Existing classical initialization retains matched accuracy on 3333 cells and 33 complete strata. Fair batched execution takes 89.19 s against 88.85 s, with peak RSS about 3.891 versus 0.697 GiB: no stable wall-time or joint resource win. Retain the earlier roughly 16% scalar-workload timing result only in its original scope, not as the latest unconditional verdict.

Prioritize a genuinely different effective inverse representation that preserves useful signal while controlling noise amplification. Use bounded falsifiable evidence before authorizing fitting or expansion. Do not keep retuning the same readout, rerunning closed recipes or publishing each small negative separately. No algorithm breakthrough, external generalization, real-BOST or paper-success claim follows; neither the entire C route nor all attention methods have been ruled out.

[Public aggregate summary](poolfire_noise_representation_review_2026-10-09_public_summary.json)
