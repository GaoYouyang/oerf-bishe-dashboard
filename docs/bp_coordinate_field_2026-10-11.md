# 小网络场初值：必要留轨迹门 / Small Neural Field Initializer

## 中文

### 最新残差校正检验

固定网络不重训，另冻经典残差校正接口，然后接未修改CGLS4。同一第一外折9个已开封锚点四指标匹配9/9、三档采样绝对门通过。场误差比无先验中位降低30.51%，比合法均值降低10.95%；场和两项梯度在全部9点优于均值。但4个观测误差增加2.58e-6至4.54e-5，均值联合不伤害仅5/9，因此预注册严格价值门仍FAIL。不能在结果后豁免该门。

独立18项及终态68项核验一致，状态/指标最大差1.08e-11/3.74e-14；共享封存权重，不是独立重训。网络6A+6AT，更便宜无先验5A+5AT且自身9/9匹配。完整几何因子仍非免费，不能把精度供应或少调用归因学习。仅保留有限场先验收益，关闭固定接口，不扩外折或调参；不是完整序列、未知几何、噪声、资源、论文或真实BOST成功。

### 原直接场初值检验

这是实际神经训练，不再是零训练的成本筛选。小型共享网络读取反投影、局部梯度、已知几何和空间坐标，给出场初值，再接未修改CGLS。部署不保留完整测量逆因子、QR或训练源场库。

训练使用其余10条轨迹、5/7/9相机的3030个合法配对。被留轨迹全部101帧、三个相机数的303配对不参与标签、归一化、均值或停止；1600次固定更新后封存。先只检验冻结第一外折的9个锚点，不是完整序列或11外折结果。

| 对照 | 四指标共同不伤害 | 场误差比中位数 |
| --- | ---: | ---: |
| 未训练反投影同价初值 | 9/9 | 0.65479 |
| 合法训练均值同价初值 | 6/9 | 0.93804 |
| CGLS35 | 9/9 | 0.72489 |
| Jacobi-PCGLS35 | 9/9 | 0.69368 |
| dual-ridge35 | 9/9 | 0.74249 |
| 强CGLS128 | 0/9 | 0.85742 |

三档采样绝对门通过，但四指标严格强参考匹配0/9。因此固定配方FAIL，在另外10折前停止，不扩宽、改loss、加深或租GPU。学习确实改善有限样本的场误差，但不能推出同精度逆求解充分；均值控制还有3个查询在联合指标上优于网络。已有更便宜、但保留完整因子的解析参考在这9点通过。

独立NumPy实现分别重建网络作用、折内统计、物理重放与原样CGLS，14项核验通过；state/metric最大差约2.81e-14/3.66e-15。权重共享，不是独立重训。每查询实际35A+35AT，几何、缓存和训练非免费，没有fresh部署wall/RSS、未知几何、噪声、外门或真实BOST证据。

## English

### Latest Residual Correction Gate

The frozen network was not refitted. A separately frozen classical residual correction and unchanged CGLS4 match all four reference metrics on 9/9 already-opened anchors and all three sampled absolute strata. Median field error falls 30.51% against no prior and 10.95% against the lawful fold mean. Field and both gradient errors improve over the mean at all nine points. However, four observation errors increase by 2.58e-6 to 4.54e-5, leaving strict joint mean-control nonharm at 5/9. The preregistered value gate remains FAIL; no retrospective exemption.

Independent 18-check action and 68-check terminal audits agree, with state/metric differences of 1.08e-11/3.74e-14 and shared frozen weights, not independent retraining. Neural cost is 6A+6AT; the cheaper no-prior control costs 5A+5AT and also matches 9/9. The full geometry factor remains nonfree. Retain finite learned field information and close this fixed interface, without refit or expanded folds. Not learned call savings, complete sequences, unseen geometry, noise robustness, resource advantage, paper or real-BOST success.

### Original Direct Field Gate

This is actual fixed neural fitting,not another zero-training cost screen. A small shared field map reads backprojection,local derivatives,reported geometry and spatial coordinates,then uses unchanged CGLS. Deployment needs no full measurement inverse,QR or source-field bank.

All101 frames at all three camera counts of one held trajectory are excluded from labels,normalization,mean controls and stopping. Training uses3030 lawful pairs from the other ten trajectories and1600 fixed updates. Only nine frozen anchors were tested,not complete sequences or all eleven outer folds.

Median field error improves27.51% versus equal-cost CGLS35 and6.20% versus the legal mean initializer. The primary is jointly nonharmful to all three strong35-step controls on9/9 queries,but only6/9 against the mean. All three sampled absolute strata pass; strict four-metric CGLS128 matches remain0/9. The necessary gate FAIL closes this literal recipe before the remaining folds,without larger networks or hyperparameter rescue. A cheaper full-factor analytic reference already passes these nine queries.

Independent NumPy learned-action and physical replay agrees across14 checks with shared frozen weights,not independently retrained optimizers. State/metric discrepancies are2.81e-14/3.66e-15. Query cost is35A+35AT plus nonfree geometry,cache,fit and network work. No fresh deployment wall/RSS,unseen-geometry,noise,external,real-BOST or paper-success claim.

## 文献边界 / Literature Boundary

经典重建后再学习场修正已有先例：[Deep Null Space Learning](https://arxiv.org/abs/1806.06137)。坐标傅里叶特征也不是新的：[Fourier Features](https://arxiv.org/abs/2006.10739)。本模型没有精确null-space投影，不继承数据一致性定理或文献中的加速/泛化保证。

Classical reconstruction followed by learned field correction and coordinate Fourier features are established approaches in the linked primary papers. This model is not an exact null-space network and inherits no consistency,speed or generalization guarantee.
