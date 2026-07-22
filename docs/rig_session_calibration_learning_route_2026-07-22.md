# 从“95% 区间”到独立 rig/session：给初学者的校准学习路线

这份路线只解决一个问题：为什么 v4 的 97.9% coverage 仍然失败，以及下一次怎样按真实 acquisition 切数据。读完后，你应能自己发现“把同一段高速序列随机切帧”为什么不是 fresh 验证。

## 0. 先记住六句话

1. **point error** 问中心准不准，**coverage** 问区间是否按承诺包含真值。
2. coverage 越高不一定越好；把区间放到无限大可以得到 100%。
3. conformal 的基本保证是对 exchangeable unit 的 **marginal coverage**，不是每个 rig 都精确 95%。
4. 同一 rig 的 1,000 帧共享相机、标定、背景和流动历史，不能自动算 1,000 个独立 rig。
5. 先学一个异方差尺度，再用独立 calibration split 做最后 inflation；不能在 audit 数据上一起调。
6. BOST 区间还要同时报最大半轴、field/gradient、held-out view 和下游物理误差。

## 1. coverage 到底在数什么

假设每次实验得到一个五维参数估计 `q_hat` 和协方差 `V_hat`。Wald score 可以写成：

```text
R = (q_true-q_hat)^T V_hat^-1 (q_true-q_hat)
```

给定阈值 `T`，置信椭球包含真值等价于 `R <= T`。重复独立实验 100 次，有 95 次满足，就是经验 coverage 95%。

但还必须报告椭球最大半轴：

```text
radius_max = sqrt(T * largest_eigenvalue(V_hat))
```

若 `T` 放大 100 倍，半轴放大 10 倍，coverage 会提高，但参数可能已经失去分辨力。

## 2. finite-sample order statistic 怎么手算

split conformal 常用：

```text
k = ceil((n_cal + 1) * (1-alpha))
T = calibration_scores_sorted[k]
```

v4 每个 cell 有 `n_cal=250`，目标 95%，所以：

```text
k = ceil(251 * 0.95) = 239
```

不是直接调用普通 `quantile(0.95)` 就结束。第 239 个顺序统计量还要和“谁是 exchangeable unit”一起解释。

### 手算练习

1. `n_cal=19` 时，95% finite-sample threshold 为什么会落到最大值？
2. `n_cal=20` 时，`ceil(21*0.95)=20`，为什么仍是最大值？
3. 若只有 12 个独立 rig，即使每个 rig 有 1,000 帧，能否直接说校准样本数是 12,000？

答案：若目标是对新 rig 泛化，外层独立单位仍只有 12；帧数只能帮助估计每个 rig 内部的 score 分布。

## 3. pooled、per-rig 与 global max 分别在保证什么

### pooled frame calibration

把所有帧的 score 混在一起取分位数。若每个帧真正 exchangeable，它给 observation-level marginal coverage。现实中容易让帧数多的 easy rig 主导阈值。

### equal-rig / CDF pooling

每个 rig 的经验 CDF 权重相同，再求整体分位数。它减少“大 rig 投票更多”的问题，但有限样本保证需要明确的两层模型，不能只凭经验均值称严格 95%。

### per-rig calibration

每个已知 rig 单独取阈值。它能改善已知组 conditional coverage，但新 rig 没有历史标签时不知道该用哪一个阈值；小组样本还会导致高方差。

### global worst-cell max

取所有 cell 95% 阈值的最大值。它保护 opened cells，却经常非常保守。v4 的 exact Godambe 从 nominal 93.09% 被推到 97.90%，就是这个现象。

## 4. marginal coverage 为什么不是 conditional coverage

设 easy rig coverage 100%，hard rig coverage 90%，两者各占一半，总体就是 95%。所以“总体 95%”不能推出“给定这个 hard rig 也有 95%”。

在完全无分布假设时，对每一个连续 covariate 值都要求 exact conditional coverage 一般不可能。可执行的折中是：

- 预先定义少量物理组并做 group coverage；
- 学习可解释的异方差尺度，再保留独立 conformal inflation；
- 报告 worst-rig / hard-quartile，而不是只报 pooled mean；
- 遇到 support 外 geometry/noise 时 fallback。

## 5. physics-normalized score 在学什么

不要让网络直接学“这个样本应该覆盖”。先预测 score 的正尺度：

```text
log s_hat(z) = beta_0 + beta^T z
R_normalized = R / s_hat(z)
```

其中 `z` 只能由部署时可见信息组成，例如：

- flow-off covariance 的 trace、condition number、camera-block correlation；
- profile information 最小特征值与 bread condition；
- 有效 view 数和角度覆盖；
- held-out residual 的 camera/time autocorrelation；
- 若两层 forward 都存在，straight-vs-curved discrepancy。

fit rigs 用于估 `beta`，calibration rigs 只用于最终 `R_normalized` 分位数，audit rigs 完全不参与。先比较 constant、log-ridge、isotonic/GAM，再考虑 MLP/DeepSets；低容量模型没有信号时，神经网络不应继续。

## 6. 五天保姆式练习

### Day 1：只看一列 CSV

打开 `learning_labs/results/temporal_qcal_profile_inference_v1/trial_metrics.csv`，只保留：

```text
base_seed, direction_index, amplitude_multiplier, split,
iterative_exact_godambe_stat, iterative_exact_godambe_radius_qref
```

会数 21 个 cell、每 cell 250 calibration + 250 evaluation。不要先看网络。

### Day 2：手写四种阈值

用 NumPy 实现 pooled、per-rig、per-cell、global max。复算结果导读中的 97.90% 与 post-hoc 94.50%，同时报最差 cell 和半轴。

### Day 3：理解 cluster-size bias

运行本库的 [hierarchical calibration toy](../learning_labs/rig_session_calibration_toy.py)：

```bash
.venv/bin/python -m learning_labs.rig_session_calibration_toy
```

观察 easy rig 帧多、hard rig 帧少时，frame-pooled coverage 为什么会偏向 easy rig。再看 equal-rig weighting 是否修复，以及代价是什么。打开 [四联图](../learning_labs/results/rig_session_calibration_toy_v1/rig_session_calibration_toy.png) 时，必须把 A/B 的 coverage 与 C 的半径一起读。

### Day 4：只学尺度，不学答案

在 toy 的 observable-scale 情景中拟合 log-ridge normalizer；随后切换到 hidden-scale 和 sign-flip shift。若模型在 shift 下失败，你应能解释：conformal calibration 没有把 OOD 变成 exchangeable。正式数字与算法结构见 [toy 结果导读](rig_session_calibration_toy_result_2026-07-22.md)。

### Day 5：翻译成 OERF 数据合同

画一张真实 split 图：

```text
fit sessions -> threshold-calibration sessions -> locked audit sessions
      |                    |                         |
  learn scale         choose inflation            evaluate once
```

把“session”换成师兄认可的实际单位。若师兄说只有一段序列，就把论文目标降为 sensitivity/consistency，不写 acquisition-level coverage。

## 7. 你应该能回答的十个自测题

1. 为什么 100% coverage 可能是失败？
2. v4 为什么用第 239 个而不是普通 95% quantile？
3. marginal 95% 与每 rig 95% 有什么区别？
4. 为什么奇偶帧不是独立 cross-fit？
5. 什么时候可以 pooled，什么时候必须按 cluster？
6. scale model 为什么必须与最终 calibration split 分开？
7. point-fraction coverage 为什么不是全体素 simultaneous coverage？
8. held-out reprojection 为什么不能单独证明 field truth？
9. 哪些 `z` 特征是部署可见，哪些会偷看 truth？
10. learned normalizer 输给 log-ridge 时，为什么应该停止？

## 8. 必读一级来源

1. Lei et al., [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116)：split/full conformal 与 locally varying width。
2. Angelopoulos and Bates, [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)：先建立 rank/coverage 直觉。
3. Barber et al., [The limits of distribution-free conditional predictive inference](https://arxiv.org/abs/1903.04684)：为什么 exact conditional coverage 不能白送。
4. Barber et al., [Beyond Exchangeability](https://doi.org/10.1214/23-AOS2276)：时间漂移、权重与保证损失。
5. Dunn, Wasserman, and Ramdas, [Two-Layer Hierarchical Models](https://arxiv.org/abs/1809.07441)：重复测量和组间异质。
6. Romano et al., [Conformalized Quantile Regression](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)：异方差 shape + 独立 calibration。
7. Ma et al., [Calibrated UQ for Operator Learning](https://openreview.net/forum?id=cGpegxy12T)：函数输出 coverage 的定义边界。

## 9. 学完后与师兄说什么

> 我不会把同一段视频的帧当独立实验。想请您确认：真实数据中什么算独立 shot/session，flow-off 或 target repeat 有多少；最终需要的是 q 联合区间、field/gradient，还是 PIV velocity correction 的 no-harm 门。只有独立单位足够，我才会把 pooled/physics-normalized calibration 写成 coverage 结果。

**这条学习路线只证明你理解了如何设计下一实验，不证明任何新算法或真实 BOST 结果。**
