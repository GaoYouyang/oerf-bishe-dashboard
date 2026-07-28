# PoolFire C 路线 v20：共享线性 lift innovation 子空间正式负结果

日期：2026-07-28

正式判决：
`FAIL_PREREGISTERED_LIFT_SUBSPACE_RANKS_V20`

独立复算：`PASS_INDEPENDENT_RECOMPUTATION_V20`

突破状态：`algorithm_breakthrough=false`

## 1. 这次实际检验了什么

v19 已排除小范围 per-view 平移与缩放。v20 不再猜运动参数，而是把每帧
compact CNN dual proposal 先做精确提升：

```text
z_t -> h_t = A^T z_t
```

随后对相邻偶数关键帧与奇数跳过帧构造：

```text
delta h_t = h_odd - h_previous_even
```

每次留出一整条 trajectory，只用另外四条轨迹的 200 个 lift innovation
拟合共享子空间。留出轨迹的 oracle 系数可以看见真实
`delta h_t`，但只能在冻结子空间内投影。这个 oracle 不可部署，它只回答：

> 如果连答案可见的系数都无法在共享 lift span 内守住最终重建精度，
> 是否还值得立刻训练一个只能看观测的系数网络？

## 2. 为什么改到 lift 空间

dual proposal 的欧氏误差与最终重建质量在 v19 已被证伪。v20 把表示放到
`Range(A^T)`：

```text
h_hat_t = h_previous_even + mean_fit + U_r c_t
```

`mean_fit` 和每个 `U_r` 都由 fit-only 的 `A^T` lift innovation 形成，
所以候选仍在 `Range(A^T)`。随后执行同一条未修改路径：

```text
h_hat_t
    -> observation-only alpha
    -> strict CGLS K1
```

固定公共 rank 为 `0, 16, 32, 64, 96, 128, 160, 192`。每个 PASS 必须由
同一数值 rank 在五条留出轨迹上同时成立。各 fold 维数不同的
`effective_full` 只作 span 敏感性检查，不能冒充公共 rank。

## 3. 证据链如何避免先看结果再调模型

正式执行顺序为：

```text
clean committed source
    -> private result-before-run execution release
    -> Stage 1: observations/times only
    -> seal all 20 arms x 5 folds x 50 skipped lifts
    -> separate-process Stage 1 QR+small-SVD recomputation
    -> only after PASS receipt, read gauge truth
    -> formal scoring
    -> independent solver/metric recomputation
```

Stage 1 独立预检结果：

```text
candidate max |difference| = 4.4964e-14
span metric max difference = 1.2669e-15
rank-zero control difference = 0
rank eligibility disagreement = 0
truth file bytes read = false
```

全量独立复算的 metric 最大差仍为 `1.2669e-15`，封存输入前后不变。

## 4. 最关键的逐轨迹结果

冻结门要求 all-frame 与 50 个 skipped frames 都满足：

- joint matched fraction 至少 90%；
- joint 与每项 harm 不超过 5%；
- severe harm 为 0。

| Trajectory | 最小 primary passing rank | rank 192 skipped joint | rank 192 harm | 判决 |
|---|---:|---:|---:|---|
| p14-s05 | 0 | 100% | 0% | 通过 |
| p22-s03 | 32 | 100% | 0% | 通过 |
| p33-s01 | 0 | 100% | 0% | 通过 |
| p45-s05 | 无 | 66% | 0% | 失败 |
| p58-s03 | 无 | 72% | 0% | 失败 |

p45 与 p58 随 rank 增加持续改善，但 rank 192 和 `effective_full` 都停在
66% 与 72%。因此不是 rank 网格少试了一个点，也不能用前三条轨迹的
成功平均掉后两条失败。

## 5. 这次失败意味着什么

这是可信的表示/目标负结果，不是数值发散：

- 五条 parent Full-CNN Dual-K1 全部通过；
- self-span 与 pooled-span sanity 全部通过；
- p45/p58 在 rank 192 的 joint harm 都是 0，severe harm 也是 0；
- 失败来自许多帧以小幅偏差越过严格 matched-accuracy 阈值，而不是少数
  帧灾难性崩溃；
- primary centered 与 raw sensitivity 在高 rank 几乎同样停滞。

可以守住的结论是：

> 以 lift-L2 选择系数时，一个跨工况共享的线性
> `A^T`-lift innovation span 不能在固定 50% 跳帧预算下覆盖全部五条
> PoolFire trajectory。

不能外推为：

- 所有系数选择目标都不可能；
- 条件化、局部字典、非线性或 history-aware 模型不可能；
- 真实 BOST 不可能加速。

## 6. 调用账的真实边界

如果未来有一个可部署系数预测器，51 个精确关键帧和 50 个 span 内跳过帧
的反事实调用账是：

```text
temporal lift candidate: 202 A + 152 A^T
full-CNN Dual-K1:       202 A + 202 A^T
Zero-CGLS K4:           404 A + 404 A^T
```

也就是相对 full-CNN Dual-K1 再少 50 次 `A^T`。但 v20 oracle 实际仍生成
全部 101 个 CNN proposal 与全部 101 个 exact lift，所以没有实测
`A^T`、wall、CPU 或 RSS 优势。

## 7. 为什么下一步不直接换大网络

v20 的 oracle 系数最小化 lift-space 欧氏误差。v19 已说明 proposal/L2
更小不一定让最终 field、gradient、observation 更好；v20 的 p45/p58
又表现为“无 harm 但 matched fraction 不够”，因此当前最有信息量的下一门
不是扩大 FNO/GRU，而是保持同一 frozen span，只改变 oracle 系数目标：

```text
c_A = argmin_c || A(h_base + U c) - A h_exact ||_2
```

这项 measurement-aware oracle 仍不可部署，但能区分：

1. **若 p45/p58 通过：**共享 span 有 headroom，失败主要来自 lift-L2
   错配；下一模型应预测 A-space/final-metric-aware 系数；
2. **若仍失败：**同一共享线性 span 的可观察部分也不够，再转向
   regime-conditioned/local/nonlinear representation；
3. 无论哪种结果，都不直接授权 fresh/test 或真实 BOST 结论。

## 8. 是否成功、是否突破

成功的是科学判别：

- 完成五条 trajectory、两种表示、九个 rank/敏感性和两个强 control；
- 独立 Stage 1 与全量重建复算都通过；
- 准确定位 p45/p58 为跨工况共享表示的阻断，而不是继续盲调 motion；
- 给出下一次最小、可证伪且直接改变算法设计的实验。

失败的是候选方法：

- 没有一个公共数值 rank 在五条 trajectory 同时通过；
- 没有 deployable coefficient predictor；
- 没有 native skip、wall、CPU、RSS、fresh/test 或真实 BOST 结果。

因此：

```text
algorithm_breakthrough=false
```

这不是论文成功，但它把下一项工作从“随便试一个更大的神经网络”压缩成了一个
明确问题：究竟是系数目标错了，还是共享线性 span 本身不够。
