# v10.8.2：完整 dual 目标五条都能过，普通线性映射一条也没过

状态：`PASS_INDEPENDENT_RECOMPUTATION_V10_8_2`

科学判决：

`PASS_FULL_VIEW_TARGET_SIMPLE_CONTROLS_FAIL`

这轮只使用五条已经属于 fit/development 的 PoolFire trajectory。没有读取
`p45-s03` fresh、`p22-s01` stopping validation 或两条 test，也没有使用真实 BOST。

## 一句话结论

固定 rank96 失败后，取消瓶颈确实找到了一个跨轨迹成立的目标：每帧完整
`K3 dual certificate` 经精确 `A^T`、observable alpha 和 restarted CGLS K1 后，
五条轨迹全部通过原来的 field / gradient / observation compatibility 门，harm 与
severe harm 都是零。

但是四个只看 observation 的便宜控制全部是 `0/5`：

| 方法 | 参数/记忆 | 通过轨迹 |
|---|---:|---:|
| Identity dual，也就是 Zero-CGLS K2 控制 | 0 | 0/5 |
| 六个 detector channel gain | 6 | 0/5 |
| 完整 2072 频率 DCT diagonal ridge | 2072 | 0/5 |
| 训练帧最近邻 dual certificate | 404 个字典样本 | 0/5 |
| 完整 K3 certificate oracle，仅作目标上限 | 非部署 oracle | **5/5** |

所以现在成立的是：

> `dual proposal -> exact A^T -> alpha -> strict K1` 外壳有跨轨迹可行目标，但普通
> 逐通道缩放、逐频率缩放和记忆检索都不能从 observation 安全地恢复这个目标。

这不是 learned predictor 成功，更不是算法加速。

## 为什么 DCT target error 不算成功

全频 DCT diagonal ridge 是最强的便宜控制。它在五条留出轨迹上的 certificate
relative-L2 p90 已经降到 `0.1775–0.3342`，而且没有 severe harm；但四条轨迹的
observation harm 仍是 `85.15%–100%`。唯一较好的 `P33-S01` joint matched 也只有
`59.41%`，低于冻结的 `90%` 门。

这说明 detector-space 的普通 L2 误差把不同方向等价处理，而 `A^T`、alpha 和 K1
会对其中某些方向高度敏感。下一模型若只优化 raw certificate MSE，很可能继续得到
“标签看起来接近、重建却不安全”的假进步。

## 一次公开纠错

v10.8 首跑错误要求：

```text
restart from x_K3 + fresh K1 == continued zero-start K4
```

这个等式不成立。restart 会重置 Krylov 共轭方向；continued K4 的第四个方向还包含
上一方向的共轭项。首跑因此被标成
`INVALID_DECISION_LOGIC_KRYLOV_RESTART_IS_NOT_CONTINUATION`，没有用于科学结论。

v10.8.1 只删除这个错误等式，仍要求五条 trajectory 全部通过原来的三类物理兼容门；
v10.8.2 又根据独立红队修正三点：

1. `z=y` 明确降为 Zero-K2 调用匹配控制，不能冒充 target prediction 成功；
2. 私有结果绑定每条 pair、几何和全部 fitted parameters；
3. 离线 K3 最小生成账更正为 `3A+3A^T`，部署候选仍是 `2A+2A^T`。

第二套实现重新生成全部 25 行结果，数值和参数最大差都为 `0.0`。

## 下一条真正有信息量的实验

现在不直接上 FNO/UNO/DeepONet。下一门先比较一个允许跨 detector、跨频率耦合的
full-view linear ridge / reduced-rank operator：

```text
normalized three-view observation
    -> full cross-view linear dual map
    -> exact A^T
    -> observable alpha
    -> strict K1
```

它回答“失败只是 diagonal 结构太弱，还是确实需要非线性/条件化”。只有这个强线性
控制仍失败，而完整 K3 target 的 5/5 上限保持成立，才有证据授权一个最小 detector
CNN；CNN 的 loss 必须直接包含 K1 后的 observation non-harm 与 field/gradient
误差，不能只拟合 raw certificate L2。

## 当前边界

```text
algorithm_breakthrough=false
deployable_predictor_success=false
fresh_trajectory_generalization=false
real_bost_validation=false
wall_time_speedup=false
paper_result=false
```

公开脱敏数字见
`docs/poolfire_c_dual_full_view_controls_v10_8_public_summary.json`。
