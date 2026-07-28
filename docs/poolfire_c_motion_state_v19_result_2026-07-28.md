# PoolFire C 路线 v19：平移式 dual transport 正式负结果

日期：2026-07-28

正式判决：
`FAIL_BOUNDED_PER_VIEW_PROPOSAL_L2_WARP_DIAGNOSTIC_V19`

独立复算：`PASS_INDEPENDENT_RECOMPUTATION_V19`

最终证据封装：`PASS_FINAL_EVIDENCE_SEAL_V19`

突破状态：`algorithm_breakthrough=false`

## 1. 这次实际做了什么

v18 已证明：如果强制每隔一帧少运行一次 compact CNN，简单 hold 或
对角增量 transport 会明显损失最终重建精度。v19 检查一个更具体的
物理假设：

> 相邻时刻的变化是否主要是各 detector view 上的小范围平移与幅值
> 缩放？如果是，能否把关键帧的 dual proposal 平移到下一帧，再交给
> 精确 `A^T` 和未修改 CGLS K1 修正？

正式实验使用五条 PoolFire fit trajectory，每条 101 帧。偶数帧是
51 个精确关键帧，奇数帧是 50 个真正需要 transport 的跳过帧。

所有 deployable temporal arm 的计划 CNN 预算相同，都是 `51/101`。
每帧重建仍执行：

```text
dual proposal
    -> exact A^T lift
    -> observation-only alpha
    -> unchanged CGLS K1
```

所以每条轨迹仍是 `202A+202A^T`。相对 Zero-K4 的调用减半来自既有
Dual-K1，不是 v19 的新增贡献。

## 2. 为什么专门加入非部署 diagnostic

仅用当前观测估计运动失败时，有两种完全不同的解释：

1. 历史 dual 确实可以被平移复用，只是观测无法识别正确位移；
2. 即使看见当前完整 dual，平移与缩放本身也不是合适的状态模型。

为区分两者，v19 加入
`oracle_proposal_motion_hold_dual_k1`。它用当前帧完整 CNN proposal
选择每个 view 的整数位移与标量缩放，再把上一关键帧 proposal
transport 到当前帧。

这个 arm 必须运行全部 101 次 CNN，不能进入资源节省比较。它也不是
最终重建误差的数学上界，只是对以下固定家族的非部署诊断：

```text
每个 view 独立
整数 dy, dx in [-2, 2]
zero-fill 边界
一个 least-squares 标量缩放
以 proposal L2 选择参数
```

## 3. 结果前冻结和审计修正

第一次审计发现了四个会制造假结论的问题，正式运行前全部修复：

1. release 必须绑定 clean commit、协议、执行器、独立验证器、模型、
   几何和五条轨迹的 READY/checksum/原始数组；
2. 不能让 51 个零误差关键帧稀释 50 个跳过帧，必须同时报告
   all-frame 和 skipped-only 门；
3. oracle 实际需要 101 次 CNN，不能误记成 51 次；
4. oracle 失败只能否定这个固定 per-view shift-scale 家族，不能写成
   所有运动模型都不可能。

正式顺序是：

```text
clean source commit
    -> private result-before-data execution release
    -> formal five-trajectory run
    -> independent reimplementation and recomputation
    -> final release/report/validation evidence seal
```

正式结果前没有读取 p33 post-open development、fresh、stopping
validation 或 test。

## 4. 最关键的 skipped-only 结果

下表只统计 50 个非关键帧。冻结门要求每条轨迹 joint match 至少
90%、joint/per-metric harm 不超过 5%、severe harm 为 0。

| Trajectory | Observation motion joint | Oracle proposal-motion joint | Oracle harm | Oracle pass |
|---|---:|---:|---:|---|
| p14-s05 | 100% | 100% | 0% | 是 |
| p22-s03 | 70% | 74% | 0% | 否 |
| p33-s01 | 100% | 100% | 0% | 是 |
| p45-s05 | 40% | 40% | 2% | 否 |
| p58-s03 | 16% | 14% | 6% | 否 |

非部署 proposal-visible diagnostic 在三条轨迹上失败。由于冻结规则是
逐 trajectory 全部通过，不能用 p14 与 p33 的成功平均掉 p22、p45、
p58 的失败。

## 5. 运动估计实际看到了什么

对 50 个跳过帧，oracle 选出非零整数位移的比例为：

| Trajectory | Observation-visible nonzero shift | Proposal-visible nonzero shift |
|---|---:|---:|
| p14-s05 | 0% | 0% |
| p22-s03 | 4% | 0% |
| p33-s01 | 2% | 0% |
| p45-s05 | 0% | 0% |
| p58-s03 | 48% | 2% |

即使直接看当前完整 proposal，四条轨迹的最佳 proposal-L2 参数从未
选择非零平移，p58 也只有 2%。它主要依靠约 `0.85-1.03` 范围内的
view scale 调整，而不是位移。

这给出一个有用但不是突破的诊断：

- 当前 coarse straight-ray proxy 中，相邻 dual proposal 的主要变化
  不是这个尺度下的二维刚性平移；
- p58 的观测看起来经常发生位移，但 proposal 几乎不发生对应位移，
  说明 detector 图像运动不能直接当作 dual state 运动；
- 对反应流而言，反应源项、扩散、热膨胀、视线积分和拓扑变化都可能
  改变梯度形态与幅值，历史状态不是简单 warp 就能得到。

## 6. 强控制给出的第二个结论

fit-only 2072 维对角 observation-delta control 的 skipped proposal
`p90` 明显低于 motion hold：

| Trajectory | Diagonal delta p90 | Observation-motion p90 | Oracle-motion p90 |
|---|---:|---:|---:|
| p14-s05 | 0.128 | 0.232 | 0.231 |
| p22-s03 | 0.137 | 0.257 | 0.251 |
| p33-s01 | 0.119 | 0.212 | 0.212 |
| p45-s05 | 0.163 | 0.303 | 0.302 |
| p58-s03 | 0.163 | 0.307 | 0.294 |

但 diagonal delta 在 p22、p45、p58 的 skipped joint match 仍只有
58%、40%、8%。这说明：

> 让 proposal 的欧氏误差更小，并不自动保证经过 `A^T`、alpha 和
> CGLS K1 后的 field / gradient / observation 非劣。

下一模型如果继续做 temporal innovation，训练目标不能只优化 dual
空间 MSE；必须把最终 reconstruction compatibility 或其可信代理纳入
选择。

## 7. 独立复算

独立验证器没有导入正式 v19 motion 或 temporal helper，重新实现了：

- 三个原生尺寸 detector view 的拆分与封装；
- zero-fill 整数位移、scale、tie-break；
- fit-only diagonal 与六通道增量拟合；
- fixed hold、linear、motion hold、motion channel delta 和
  proposal-visible diagnostic；
- exact `A^T`、alpha、strict CGLS K1、Zero-K4；
- all-frame 与 skipped-only field / gradient / observation 门。

最大数值差为 `1.1102230246251565e-16`，输入文件验证前后完全一致。
最终 seal 进一步绑定了 release、report 与 independent validation。

## 8. 是否成功、是否有突破

工程与证据闭环成功，候选算法没有成功。

成功的部分：

- 真实跑完五条轨迹、七个 arm 和两个精度视角；
- 修复了关键帧稀释、oracle 调用误账和证据绑定问题；
- 独立复算与最终 seal 通过；
- 明确排除了一个直觉上合理、实际不够用的 transport 家族。

失败的部分：

- bounded per-view integer shift + scale 在三条轨迹上无法保持最终精度；
- observation-visible motion 没有形成可部署优势；
- 没有 native CNN skip、wall、CPU 或 RSS 结果；
- 没有打开 fresh/test，也没有真实 BOST 证据。

因此：

```text
algorithm_breakthrough=false
```

## 9. 接下来真正值得跑的科学门

不继续调位移半径、scale、阈值或更大 optical-flow 网络，也不直接训练
FNO/UNO/GRU。下一步先做一个更便宜的非部署表示诊断：

> 在四条 fit trajectory 上学习 dual innovation
> `delta z_t = z_t - z_key` 的低维子空间；对 held-out trajectory，
> 先允许使用真实 `delta z_t` 的 oracle 投影系数，只检验这个子空间
> 能否在固定 `51/101` keyframe 预算下通过 skipped-only 最终重建门。

判决顺序：

1. 如果 oracle low-rank innovation 仍失败，当前 PoolFire proxy 上的
   50% temporal proposal amortization 路线停止；
2. 如果 oracle 通过，再训练只看部署观测
   `(y_t, y_key, z_key)` 的小模型预测低维系数；
3. 只有 observation-only predictor 五轨迹全部通过，才实现原生
   causal skip worker 和 wall/RSS profile；
4. 真实 BOST 的同精度仍必须由组内重复测量、噪声和标定不确定度定义。

这个顺序优先回答“表示是否有 headroom”，避免在没有上限证据时浪费
算力训练大网络。

## 10. 原创性边界

关键帧推理、历史状态 warp、动态 CT motion compensation、GRU-BOST
和 4D BOST 都已有明确先例。v19 不能声称 first、SOTA、全局唯一或
新型 motion estimation。

即使未来 low-rank innovation 成功，可防御的贡献也只能是一个窄组合：

1. BOST 特定的 observation-visible dual innovation；
2. exact `A^T` lift 与未修改物理求解器；
3. trajectory-level skipped-only non-inferiority；
4. native model-call、wall、CPU、RSS 实测；
5. fresh CFD trajectory 与组内真实 BOST 迁移。

当前结果只支持：

```text
这个固定的 per-view integer shift + scale family 不够用。
其他非刚性、形态或低秩状态模型尚未被证明可行，也没有被证明不可能。
```
