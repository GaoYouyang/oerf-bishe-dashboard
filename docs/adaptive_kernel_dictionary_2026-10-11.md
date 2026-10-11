# 固定核字典的精度与成本取舍

## 中文

**固定128点数据相关字典的严格保真门FAIL，但保留了可量化的场收益与压缩效果。** 这是已知五相机干净输入、11折完整留轨迹隔离、每折3帧共33查询的必要小门，不是完整时序。

主候选用训练观测选核字典，等宽对照用均匀字典；两者各自由正式与独立实现拟合、预测、物理重放。全部23项核验通过，最大场差2.31e-13、指标差6.88e-15。每折完全排除留出轨迹101帧，只有其他1010源参与训练。字典选择不读取场标签、时间或工况。

| 指标 | 数据选点128 | 均匀128对照 |
| --- | --- | --- |
| 四指标匹配强CGLS128 | 33/33 | 33/33 |
| 采样绝对与匹配分层 | 11/11 | 11/11 |
| 对均值/无先验四项不伤害 | 32/33、33/33 | 33/33、33/33 |
| 相对均值/无先验的场误差中位比 | 0.906719、0.796396 | 0.901844、0.793609 |
| 相对原完整核的场误差中位比 | 1.090459 | 1.095873 |

主候选比无先验/均值先验场误差中位降低20.36%/9.33%，但比原条件先验增9.05%，超过结果前冻结的5%保真限度，并有一个均值对照伤害。因此不能只用33匹配宣称整体成功；对照也未守住保真线，不能事后替换。

预测器数值存储由196.04MB降至20.85MB，为原来的10.64%。同一精确打包逆作用、lift和CGLS4下，局部八查询温热批次43.04ms对81.40ms，为原核52.88%。两项只是部署payload与局部资格：完整几何因子未从管线中消失，不是进程RSS或全准备/训练至部署计时。每预测仍为5A+5A^T，主要省调用来自经典解析控制，不能归因于学习。源标签、QR、核选择、拟合、融合和因子准备均非免费。

停止这份固定128配方，不加点、调核或扩网络救援。保留先前完整1111查询的条件先验正证据、噪声失败、Fourier失败及部署成本记录。该结果说明精度与存储存在取舍，不关闭整个C路线。没有完整序列压缩、神经逆作用、新几何/相机数泛化、外门或真实BOST成功。

## English

**The fixed128-centre data-dependent dictionary fails strict fidelity while retaining measurable field gains and compression.** This necessary gate covers33 clean queries, three anchors per each of11 complete held-trajectory folds, at one known five-camera geometry. It is not a complete temporal sequence.

The primary selects kernel sections from training observations; the equal-width control uses uniform sections. Formal and separate implementations independently fit, predict and physically replay both. All23 checks pass; maximum state and metric discrepancies are2.31e-13 and6.88e-15. Each fold excludes all101 held frames and uses only1010 legal other-source pairs. Selection reads no field labels, time or condition.

Both methods match the qualified CGLS128 reference on33/33 four-metric queries and pass11/11 sampled absolute/matched strata. Primary joint mean/zero-prior nonharm is32/33 and33/33; the uniform control is33/33 against both. Primary median field-error ratios to mean/zero are0.906719/0.796396, or9.33%/20.36% improvements. However, its ratio to the exact prior is1.090459, exceeding the preregistered1.05 fidelity allowance. The control ratio1.095873 also fails. No retrospective replacement or success from matches alone.

Predictor numerical payload falls from196.04MB to20.85MB, ratio0.106356. With identical packed inverse, exact lift and unchanged CGLS4, a warmed eight-query batch takes43.04ms versus81.40ms, ratio0.528769. These are payload/local eligibility, not fresh-process RSS or whole preparation/training-to-deployment resources. The complete geometry factor remains. Each prediction still uses5A+5A^T; classical solving explains the principal call savings, not learning. Source/QR/kernel-selection/fitting/fusion/factor preparation remains nonfree.

Close this fixed recipe without more centres, kernel tuning or larger-network rescue. Retain the earlier1111-query exact-prior positive, noise/Fourier failures and deployment outcomes. This is an accuracy-storage tradeoff, not closure of the C route, neural inverse action, complete-sequence compression, new-geometry/cardinality generalization, external or real-BOST success.

## Method Context

Data-dependent Nyström bases are established; see [Nyström versus Fourier](https://papers.neurips.cc/paper_files/paper/2012/hash/621bf66ddb7c962aa0d22ac97d69b793-Abstract.html) and [computational regularization](https://papers.nips.cc/paper_files/paper/2015/hash/03e0704b5690a2dee1861dc3ad3316c9-Abstract.html). Their statistical sampling results do not certify this deterministic BOS dictionary, supply a novelty claim or guarantee transfer.
