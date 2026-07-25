# PoolFire C 路线：半收敛、Warm Start 负结果与覆盖扩充门

> 状态：`OPEN_PROXY_DEVELOPMENT_DIAGNOSIS_ONLY`<br>
> 日期：2026-07-25<br>
> 突破标记：`algorithm_breakthrough=false`<br>
> Test：两条 untouched test 均未下载、未解压、未统计、未评分<br>
> 物理边界：这是公开 PoolFire `rho` 形态上的独立 straight-ray 代理逆问题，不是真实实验 BOST

## 1. 这轮真正回答了什么

师兄确认的 C 路线是：先由算子学习给出三维初值，再用同一个物理求解器细化；在最终场精度等价时，比较完整多视角 `A/A^T` 调用、端到端时间和内存。

本轮完成了五条开放完整轨迹、505 帧、同一冻结几何上的四臂 classical 开发比较：

1. Zero-CGLS；
2. normalized BP-CGLS；
3. exact geometry-diagonal PCGLS；
4. train-only standardized dual-ridge warm start + CGLS。

同时加入了不使用随机 probe 的精确 `diag(A^T A)`，并让 K=0 初值、初始化成本和每个 checkpoint 的 `A/A^T` 都进入同一本账。

## 2. 关键发现：K=24 不是可靠的“高精度终点”

在停止规则 validation `p=22kw_size=01` 上，Zero-CGLS 出现典型半收敛：

| Zero-CGLS | K=4 | K=24 | K24 / K4 |
|---|---:|---:|---:|
| field relative-L2 p90 | 0.68190 | 1.02633 | 1.505 |
| gradient relative-L2 p90 | 0.97990 | 2.28542 | 2.332 |
| observation residual p90 | 0.34810 | 0.30430 | 0.874 |
| 完整 `A/A^T` 调用 | 8 | 48 | 6.0 |

观测残差继续下降，但三维场和梯度已经恶化。这意味着：

- “多跑迭代”不是更强的 ground truth；
- 迭代深度本身就是正则化参数；
- 若用 K=24 作为 matched-accuracy 参照，会把本来很差的终点当成目标，制造虚假的 warm-start 优势；
- 后续停止规则必须同时保护 field、gradient 和 observation，不能只盯 residual。

这是有研究价值的机制发现，但它仍来自已打开的 validation，不能冒充 confirmatory 结果。

## 3. Ridge K2 的信号为什么仍不能写成成功

结果打开后，以半收敛更合理的 Zero K4 作诊断参照，dual-ridge K2 在 `p=22kw_size=01` 上：

- 5 次完整调用，对照为 8 次，调用减少 37.5%；
- joint matched-frame fraction 为 97.03%；
- harm fraction 为 0。

但把同一个 K2/K4 规则回放到全部五条开放轨迹后，只通过 4/5：

- `p=14kw_size=01` joint pass 仅 56.44%；
- field 与 gradient 均通过，失败来自 observation residual；
- 所有轨迹的实测 wall time 都没有加速；
- 约 25 MB 的 ridge 推理开销在这个小 CPU 代理上抵消了少掉的迭代。

所以当前结论只能是：

> 已看到“减少物理调用”的 exploratory headroom，但没有稳定跨轨迹 matched-accuracy，也没有端到端 wall-time speedup。

更严格地说，这五条轨迹里三条参与了 ridge 拟合，p14 参与了超参数选择，p22
参与了 K2 诊断选择，因此独立开放评估轨迹实际为 0。`37.5%` 只能写成事后固定
比较中的理论完整算子调用计数，不能写成独立验证的跨工况加速。

## 4. 为什么没有立刻训练 FNO

### v2：线性摊销 K4 teacher 失败

v2 让模型学习从 observation 直接预测 Zero-CGLS K4 的三维场。它只在三条 fit trajectory 上训练，用 `p=14kw_size=01` 选择正则和低秩：

- 最佳检查 rank 为 256；
- joint pass fraction 为 0；
- 三个指标 pass fraction 均为 0；
- harm fraction 为 92.08%；
- 状态：`FAIL_NO_RANK_MATCHES_P14_CGLS_K4_AT_ZERO_DEPTH`。

失败后没有打开 `p=22kw_size=01`，更没有触碰 test。

### v3：PCA 先验检查证明数据覆盖不足

在训练任何神经网络前，v3 先问一个更便宜的问题：三条 fit trajectory 的低维子空间能否表示 p14 的 observation 和 K4 teacher field？

rank 256 时：

| 表征 | fit 累积能量 | p14 p90 重建误差 | p14 worst |
|---|---:|---:|---:|
| observation PCA | 99.8668% | 0.51684 | 0.55315 |
| K4 teacher field PCA | 99.9711% | 0.25314 | 0.28484 |

没有一个 rank 通过表征门，状态为 `FAIL_LATENT_OUTPUT_PCA_HEADROOM`。
训练集内部能量接近 100%，却仍不能表示新轨迹。这个结果只证明“三条 fit
trajectory 形成的线性 PCA 子空间、rank 不超过 256”没有覆盖 p14 的 K4 代理目标；
它没有排除更高 rank、非线性表示或其他架构。此时直接训练 FNO/UNO/3D U-Net
仍有较高的记忆训练轨迹风险。

## 5. 当前唯一科学门：分阶段扩充 fit 覆盖

v4 在任何新增数据下载前冻结：

1. 先接入 `p=14kw_size=05`；
2. 再接入 `p=22kw_size=03`；
3. 只在 p14 模型/表示选择 validation 上，用完全相同的 independent proxy、K4 teacher 和 PCA gate 重跑；
4. 只有 rank-256 p90 表征误差相对下降至少 20%，才继续接入其余五条 clean fit trajectory；
5. p22 停止规则 validation 不参与 acquisition 或 coverage 决策，避免职责污染；
6. 若没有实质改善，停止盲目扩数据，转而重审输入表示、几何条件和目标定义；
7. 在覆盖门通过前不训练大模型。

`p=14kw_size=03` 永久只作 development。`p=14kw_size=01` 和 `p=22kw_size=01` 的 validation 职责不变；两条 test 继续封存。

## 6. 下一代算法的可投稿假设

当前最值得保留的假设不是“FNO 一定比 DeepONet 强”，而是：

> 一个只读取部署可见 observation、显式感知 geometry/model-mismatch、以半收敛安全区为目标的轻量 warm operator，能否在不同 PoolFire trajectory 上给出可纠正初值，并以 fail-closed 回退稳定减少物理调用和端到端时间？

它至少需要以下一对一消融：

- Zero / BP / exact-diagonal PCGLS；
- dual ridge；
- 等参数 3D U-Net / FNO / UNO 或 DeepONet；
- field-only 与 K4-teacher / short-refinement loss；
- 无 gate 与 deployment-visible gate；
- 固定 K 与 validation 冻结的半收敛停止规则；
- calls、wall、memory、field、gradient、observation、p50/p90/worst、harm。

在 untouched test 与独立真实 BOST 样例都通过前，禁止写“泛化成功”“真实 BOST 加速”或“高质量论文结果已经形成”。

## 7. 独立方法学审计后的声明边界

独立只读审计确认：K24 在开放 p22 上是事后半收敛征象，但不能写成 CGLS 普遍发散；
37.5% 不是计算加速；303/101 是帧数而不是独立统计样本量；测试只能称“身份预先指定、
真值未开启”，不能称盲法测试。v4 已按审计意见删除 p22 对 acquisition 决策的授权。

代码审计还要求重算成本与读取边界。修正后：

- 外部 warm initializer 即使恰好输出全零，也必须支付一次初始 `A`；
- 既有 arm 缓存必须完整绑定协议、pair、角色、模型和全部 8 个 checkpoint；
- PCGLS 的 geometry-diagonal setup 同时进入单次使用与逐帧摊销 wall 账；
- v2 在 p14 rank gate 失败前不再读取 p22；
- 私有 PCA 包只保留 fit-only basis/statistics，旧的样本混合包已删除。

审计后重跑仍得到相同的科学结论：v2 与 v3 两道门均失败，p22/test 没有被用来
挽救失败模型。定向回归测试为 `73 passed`；这只是代码和证据链通过，不是算法成功。

## 8. 初学者怎么理解

把 CGLS 想成用观测不断修正三维答案。开始几步会补回真正看得见的大结构；继续跑太久，它会开始追逐 forward mismatch 和难以可靠恢复的细节。Warm start 的目标不是替代求解器，而是把起点放进“少走几步仍能到好答案”的区域。

这轮最重要的诚实结论是：我们找到了正确的困难，但还没有找到稳定战胜它的算法。下一步扩充训练覆盖，是为了判断这个困难能否被一个通用模型学习，而不是为了用更多数据掩盖失败。
