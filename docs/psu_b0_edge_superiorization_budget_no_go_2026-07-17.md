# TV/Huber superiorized PCGLS：机制可见，但调用效率严格 NO-GO

日期：2026-07-17
状态：`POSTOPEN_SUPPCG_BUDGET_EFFICIENCY_NO_GO`

## 一句话结论

三维 TV/Huber 非上升扰动在同 stage 下出现极小的形态相关改善，但真正的
SupPCG 必须在扰动后重建数据残差，增加一次 forward。按总
`forward + adjoint` 调用公平比较时，初始 32 个候选和唯一授权的 48 个深阶段
扩展候选全部输给普通 graph-PCGLS，因此关闭 SupPCG 性能分支，不再调步长，
fresh 与 neural proximal 继续封存。

## 1. 为什么不能把 TV 当免费后处理

[Helou et al. 的 SupPCG](https://arxiv.org/abs/1807.10151)不是“重建后做一次
平滑”。它在 PCG 更新前加入总量可求和、且不增加次级准则的扰动，然后重新
计算被扰动点的数据梯度。

本实现做了四件明确的事：

1. 3D isotropic smoothed-TV 与 isotropic Huber；
2. 每个样本使用归一化负梯度和 `gamma * a^ell` 可求和步长；
3. 每个 inner step 只有在 penalty 不增加时才接受；
4. 外边界 support 始终为零。

扰动改变体场后，旧 residual 不再正确。实现显式计算 `A s`，不把这次 forward
藏起来。`K` 个 superiorized stages 的账本为：

```text
forward calls = 2K - 1
adjoint calls = K
total calls   = 3K - 1
```

普通 PCGLS-K 则是 `K + K = 2K` 次。SupPCG-3 因而应与总调用同为 8 的
graph-PCGLS-4 比，而不是只与 graph-PCGLS-3 比。

## 2. 内核验证

- TV/Huber analytic gradient 通过中心有限差分；
- 每次 inner perturbation 的 penalty 均非上升；
- support 外体素严格保持为零；
- `gamma=0` 时，SupPCG 体场与原 fixed-SPD PCGLS 在 float64 容差内一致；
- 4 stages 的物理调用严格为 `7F + 4A`；
- 定向测试 18 项通过后才运行真实 PSU detector geometry。

这些验证只说明实现可信，不说明方法有效。

## 3. 初始尺度 smoke

固定两个已打开 replicates：0 与 8，共 16 个解析反应场。比较：

- component s3-k4；
- graph s3-k3/k4/k5/k6；
- TV/Huber；
- stages 3/4；
- inner steps 1/2；
- `gamma = 0.005/0.02/0.08/0.32`；
- decay 0.7。

总计 37 个方法，其中 32 个 SupPCG 候选，592 条逐场记录。

同 stage 下最佳 `sup_huber_k3_n1_g0p32_a0p7`：

- mean：`+0.124%`；
- p10：`-0.085%`；
- `>1%` harm：`0%`；
- worst：`-0.228%`。

这个信号很小但不是完全随机：thin-front、double-front、annular 与 vortex
更常受益，plume 和 oblique-shock 略退。

但与同总调用的 graph-PCGLS-4 比：

- mean：`-6.016%`；
- p10：`-10.299%`；
- `>1%` harm：`87.5%`；
- worst：`-15.551%`。

TV/Huber 方向的局部结构收益抵不过少做一次普通 PCGLS 更新。

## 4. 唯一授权的 tail extension

为了避免“只测浅阶段”的反驳，在第二份配置运行唯一一次扩展：

- graph-PCGLS stages 4–9；
- SupPCG stages 5/6；
- TV/Huber；
- inner steps 1/2/4；
- `gamma = 0.32/0.64/1.28/2.56`；
- 48 个 SupPCG + 7 个 classical references；
- 880 条逐场记录。

普通 graph-PCGLS 相对 component s3-k4：

| stages | mean | p10 | >1% harm | worst |
|---:|---:|---:|---:|---:|
| 4 | +1.420% | +0.108% | 0% | -0.919% |
| 5 | +6.850% | -1.697% | 18.75% | -3.065% |
| 6 | +11.224% | +0.068% | 12.5% | -2.244% |
| 7 | +14.203% | -0.074% | 6.25% | -3.653% |
| 8 | +15.157% | -3.534% | 12.5% | -8.724% |
| 9 | +17.041% | -2.395% | 12.5% | -7.774% |

这再次显示 semiconvergence：均值随深度上升，但尾部不单调。

tail screen 的最佳预算候选 `sup_huber_k6_n1_g2p56_a0p7`：

- 同 stage mean：`-2.278%`；
- 对 graph-PCGLS-8 的 mean：`-8.518%`；
- p10：`-25.463%`；
- `>1%` harm：`68.75%`；
- worst：`-26.411%`。

更大扰动不但没有保护深阶段尾部，连同 stage 机制信号也消失。

## 5. 判决与下一步

配置已预先写明：若没有候选在 graph call-budget floor 上取得正 mean，就关闭
SupPCG 分支，不再调 `gamma` 或 inner steps。该条件已触发。

这不等于“TV/Huber 无效”。更准确的结论是：

- 当前 SupPCG 的 residual rebuild 成本过高；
- 弱小的同-stage edge 信号不能抵消一次 Krylov 更新；
- 深阶段的大扰动会破坏数据与形态尾部；
- 下一强基线应直接求解 data + TV/Huber 目标，并保持每迭代一对
  `A/A^T`，优先测试 primal-dual/PDHG；
- 只有 one-pair deterministic baseline 过门，才讨论 bounded learned
  proximal map。

## 6. 证据边界

- 真实：PSU detector/ray geometry；
- 合成：反应场、flow-on noise、covariance truth；
- 已见：replicates 0 与 8；
- 未使用：真实 flow-off、实验三维真值、OERF 私有数据；
- 未做：16-replicate full opened grid、fresh confirmation、神经模型训练；
- 不宣称：TV/Huber 一般无效、算法优越或 OERF 实验有效；
- fresh：未打开。

## 7. 可复现入口

- 数值内核：
  `demo_t16_operator/psu_b0_edge_superiorization.py`
- 初始配置：
  `demo_t16_operator/configs/psu_b0_edge_superiorization_scale_smoke_v1.json`
- tail 配置：
  `demo_t16_operator/configs/psu_b0_edge_superiorization_tail_smoke_v1.json`
- runner：
  `site_tools/run_psu_b0_edge_superiorization_screen.py`
- 初始报告：
  `demo_t16_operator/results/psu_b0_edge_superiorization_scale_smoke/report.json`
- tail 报告：
  `demo_t16_operator/results/psu_b0_edge_superiorization_tail_smoke/report.json`
- 审计图：
  `demo_t16_operator/results/psu_b0_edge_superiorization_tail_smoke/psu_b0_edge_superiorization_no_go.png`
