# PoolFire C 路线 v18：因果 dual 时间复用正式负结果

日期：2026-07-28

正式判决：`FAIL_POST_OPEN_TEMPORAL_AMORTIZATION_GATE_V18`

独立复算：`PASS_INDEPENDENT_RECOMPUTATION_V18`

突破状态：`algorithm_breakthrough=false`

## 1. 为什么做这一步

v16 已在一条额外公开 development trajectory 上把完整调用从
`404A+404A^T` 降到 `202A+202A^T`，但 v17 的 CPU 中位仍比
Zero-K4 高 11.28%，第一次冷进程也仍有 236.82% wall harm。v18
不再继续调整线程，而是检验一个更接近流动物理的问题：

> 相邻时刻的观测若变化不大，能否只在关键帧运行 compact CNN，
> 其余帧从上一关键帧因果输运 dual proposal？

在线门只使用当前与历史观测，不使用真实三维场。无论 proposal
来自 CNN 还是时间输运，每帧仍执行精确 `A^T`、可观测 alpha 和
未修改 CGLS K1。

## 2. 冻结方法

五条 fit trajectory 用于拟合：

1. 相邻观测变化的中位阈值；
2. 一个标量 delta transport；
3. 一个 2072 维对角 delta transport。

选中的因果规则是：

```text
change_t = ||y_t - y_key|| / ||y_key||

如果 age >= 2 或 change_t > fit-only median：
    运行完整 CNN，更新 keyframe
否则：
    z_t = z_key + diag_gain * (y_t - y_key)
```

阈值和对角增益先做 trajectory-level leave-one-out。p33 只作为已经
消费过的 post-open development mechanism diagnostic，不参与拟合。

公平对照包括：

- 每帧 full CNN；
- 同一门下 previous-dual hold；
- 同一门下标量 delta transport；
- 同一门下对角 delta transport；
- 固定 stride-2 对角 transport；
- Zero-CGLS K4。

## 3. 五折 fit-only 结果

选中门在五个 held-out fit trajectory 上都通过原 v16
field / gradient / observation compatibility 与 harm 门，但执行率说明了
一个关键问题：

| Held-out trajectory | 计划 CNN 帧 | 比例 | Joint match | Harm |
|---|---:|---:|---:|---:|
| p14-s05 | 51 / 101 | 50.50% | 100.00% | 0 |
| p22-s03 | 72 / 101 | 71.29% | 93.07% | 0 |
| p33-s01 | 51 / 101 | 50.50% | 100.00% | 0 |
| p45-s05 | 101 / 101 | 100.00% | 100.00% | 0 |
| p58-s03 | 101 / 101 | 100.00% | 100.00% | 0 |

五折平均计划执行率为 74.46%，但两个变化更快的轨迹靠
`101/101` 全回退才守住精度。因此“五折全过”不能写成稳定加速。

## 4. p33 正式结果

fit-only 阈值为 `0.2388728104`。p33 的相邻观测变化更快，因果门在
101 帧中计划执行 94 次 CNN：

- 计划 CNN 减少：**6.93%**
- 冻结最低要求：**20%**
- Joint match：**100%**
- Joint harm：**0**
- Severe harm：**0**

也就是说，精度通过的原因是门几乎总选择完整 CNN，而不是成功复用。

强制固定 stride-2 后：

- 计划 CNN：51 / 101
- Joint match：**71.29%**
- Joint harm：0
- Severe harm：0
- Compatibility：**失败**

它证明简单提高复用率会越过精度边界。对角 transport 的 proposal
误差略小于标量与 hold，但没有改变这个成本结论。

## 5. 独立复算

独立验证器重新实现了：

- fit-only scalar / diagonal calibration；
- 因果 keyframe mask；
- 三种时间 transport；
- exact `A^T` lift 与 strict CGLS K1；
- Zero-CGLS K4；
- 冻结的 v16 compatibility/harm 统计。

复算结果：

| 检查 | 最大差 |
|---|---:|
| 五折结果 | 0 |
| calibration | 0 |
| p33 fields | 0 |
| p33 metrics | 0 |
| Zero-K4 fields | 0 |

最终状态为 `PASS_INDEPENDENT_RECOMPUTATION_V18`。

## 6. 必须保留的证据边界

v18 是 counterfactual proposal headroom diagnostic：

- 为了同时比较所有 temporal controls，runner 先计算了 101 帧 full
  proposal，再按冻结 mask 替换非关键帧；因此 94/101 是**计划执行数**，
  不是已经在 native 入口实测的调用数。
- p33 是已经消费过的 development trajectory，且本轮没有做新的
  observation-only predictor / score process 隔离。
- 所有 temporal arm 仍使用 `202A+202A^T`。相对 Zero-K4 的 50%
  算子减少来自既有 Dual-K1，不是 v18 的新增贡献。
- 未运行 wall、CPU 或 RSS profile；没有打开 fresh、stopping
  validation 或 test。

所以不能写成实际 CNN 加速、额外算子减少、独立泛化、真实 BOST 或论文成功。

## 7. 负结果后的策略调整

当前“观测变化阈值 + previous dual/对角 delta transport”路线停止。
不再在 p33 上继续调阈值、最大间隔或对角增益。

探索性 previous-field recycling 进一步显示：

- 只在第一帧运行 CNN、随后每帧做一次 `A/A^T` 校正，在 p33 上
  joint match 仅约 0.99%，不是可用解；
- 固定 stride-2 reset 可把账降到 `152A+152A^T`，在 p33 上达到
  94.06% joint match，但在 held-out p58 上只有 84.16%，没有形成
  跨轨迹可靠门；
- 使用 fit truth 选择更宽阈值虽可在已见 p33 上得到较低调用账，
  但 leave-one-trajectory-out 对 p58 失败，不能升级为正式候选。

这些探索只负责否定简单时间复用，不计作正式算法结果。下一候选若继续
利用时间信息，必须显式建模运动/拓扑变化，并打赢 previous-field、
Krylov recycling、周期关键帧和相同 CNN 预算对照。

## 8. 原创性红线

关键帧稀疏推理、状态输运、reuse gate、dynamic inverse recycling
都有明确先例，包括 Deep Feature Flow、Reuse Gate、DeltaCNN、
hybrid projection recycling、Learned ReSeSOp 和动态 CT/4D 重建。
因此 v18 不是单点创新。

只有未来同时闭合以下证据，才可能形成窄而可防御的贡献：

1. BOST 特定的 observation-visible refusal gate；
2. dual proposal 的因果运动/状态建模；
3. 每帧精确 `A^T` 与未修改物理求解器；
4. trajectory-level field/gradient/observation 非劣；
5. 实测 CNN、`A/A^T`、wall、CPU、RSS；
6. fresh trajectory 与组内真实 BOST 迁移。

当前明确结论仍是：

```text
algorithm_breakthrough=false
```
