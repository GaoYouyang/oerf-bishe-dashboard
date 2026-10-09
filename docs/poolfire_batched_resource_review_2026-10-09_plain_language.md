# 2026-10-09：计算效率复盘 / Computational efficiency review

同一11条已打开train轨迹、101帧与5/7/9相机，共3333样本、33个完整分层。两边都改为小批独立右端执行，迭代、正则和物理目标不变，不是block Krylov，也没有训练新模型。每边三次fresh运行全部封存后独立核验，双方3333/3333四指标匹配、33/33完整分层保持，重复结果逐位一致。

The same eleven opened train trajectories, 101 frames and 5/7/9 cameras contain 3333 cells and 33 complete strata. Both arms use small batches of independent right-hand sides with unchanged iterations, ridge and physical objective: not block Krylov or a new model. Three fresh runs per arm seal before independent validation; both retain 3333/3333 four-error matches and 33/33 complete strata with bitwise-identical repeats.

新缓存离线工作负载中，CGLS128耗时中位数88.85秒、峰值RSS约0.697GiB；完整经典初值为89.19秒、约3.891GiB。三组时间比例1.014/1.002/0.981，没有稳定时间优势，内存约5.6倍。35对调用比128对少，但完整因子setup、应用与输出不是免费。旧逐个右端工作负载的有限加速证据保留，不能把跨运行耗时之差当作因果提速，也不是单帧在线延迟。

In this new cached offline workload, CGLS128 has median wall time 88.85 s and peak RSS about 0.697 GiB; full classical initialization has 89.19 s and 3.891 GiB. Paired wall ratios 1.014/1.002/0.981 show no stable time win; memory is about 5.6x higher. 35 call pairs are fewer than 128, but complete factor setup, application and output are nonfree. Retain earlier scalar-workload time evidence; cross-run differences are not causal speedups or single-frame latency.

另一个独立99锚点审计澄清：旧共享容量模型四项真实误差都略有改善，但场误差改善中位数只有0.27%/0.45%/0.60%，不是LOTO成功。参考误差的弱灵敏度能量下界也不证明不可恢复。后续把实际CFD质量与有限参考匹配分开，同时要求有效率的经典对照、完整时间与RSS证据；不继续堆小配方或用大网络救援。当前没有稳定学习增益、外部或真实BOST结论。

A separate independent 99-anchor audit finds small actual-error improvements in all four metrics for the old shared-capacity model, with median field gains only 0.27%/0.45%/0.60%, not LOTO success. Its weak-sensitivity certificate is not an irrecoverability proof. Keep actual CFD quality distinct from finite-reference agreement and require efficient classical controls plus complete wall/RSS evidence. Do not accumulate tiny variants or rescue them with larger networks. No stable learned gain, external or real-BOST result follows.

同精度减少调用不等于完整提速。已有物理接口、完整经典精度及旧工作负载收益继续保留；学习型完整资源胜利尚未实现。

Matched-accuracy call reduction is not complete speedup. Retain qualified physics, complete classical accuracy and old-workload benefits; a learned whole-resource win remains unachieved.
