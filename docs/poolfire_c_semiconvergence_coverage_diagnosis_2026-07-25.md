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

## 5. 分阶段扩充 fit 覆盖：门已运行并失败

v4 在任何新增数据下载前冻结：

1. 先接入 `p=14kw_size=05`；
2. 再接入 `p=22kw_size=03`；
3. 只在 p14 模型/表示选择 validation 上，用完全相同的 independent proxy、K4 teacher 和 PCA gate 重跑；
4. 只有 rank-256 p90 表征误差相对下降至少 20%，才继续接入其余五条 clean fit trajectory；这道门不授权神经训练；
5. p22 停止规则 validation 不参与 acquisition 或 coverage 决策，避免职责污染；
6. 若没有实质改善，停止盲目扩数据，转而重审输入表示、几何条件和目标定义；
7. 在完整 clean-fit 的 trajectory-level 留一覆盖门通过前，不训练任何论文候选网络。

`p=14kw_size=03` 永久只作 development。`p=14kw_size=01` 和 `p=22kw_size=01` 的 validation 职责不变；两条 test 继续封存。

### 20% 门通过以后仍要过什么

v4 的相对改善门只回答“首批新增工况是否值得继续扩充”，不能回答
observation→field 是否可学。若通过，先完成剩余 clean-fit 接入，再按完整 trajectory
做 leave-one-trajectory-out：每个 fold 的 normalization、PCA、模型权重和 early
stopping 只能来自 fold-train trajectories，并同时报告 rank 256 与可用最大 rank。
原 v3 的绝对 output 门仍保留为 p90 `<=0.05`、worst `<=0.10`。

只有完整覆盖仍成立，才允许训练一个小型 BP-conditioned 3D U-Net sentinel，而不是
同时搜索四种架构。sentinel 至少要在 80% 的留出 fit trajectories 上达到 joint
matched-accuracy pass `>=90%`、harm `<=5%`、固定 `K<=2`；BP、推理、residual
检查和回退全部计入后，总调用必须少于 Zero K4 的 8 次，轨迹等权 wall-time
中位数不能变慢，并测 fresh-process whole-pipeline peak RSS。通过后才比较 FNO、
UNO 和 DeepONet。

### v4 实际结果

`p=14kw_size=05` 与 `p=22kw_size=03` clean-fit 接入后，rank-256 p14 K4-target
PCA p90 从 `0.2531381001` 降到 `0.2180511919`，相对改善 `13.8608%`。冻结门要求
至少改善 20%，即 p90 不高于 `0.2025104801`，所以正式状态是
`FAIL_FIRST_BATCH_MATERIAL_P14_COVERAGE_IMPROVEMENT`。

rank 256 时 fit output 能量解释率已为 `99.6543%`，但 p14 output p90 仍为
`0.218051`；p14 observation p90 为 `0.469397`。因此目前停止剩余同类 clean-fit
接入，也不授权大模型训练。下一步只允许在已经开发化的 p14 上审查：

1. rank 256 上限是否人为截断了有效方向；
2. K4 的线性齐次性是否与当前带均值的仿射 PCA 不匹配；
3. 用部署可见 observation 范数拆分“幅值”和“形状”能否降低跨工况误差；
4. 固定几何下的 view/block scaling 是否导致输入表示失配。

这轮不能调用 p22/test 救结果，也不能把线性 PCA 失败写成“非线性模型必败”。

## 6. 下一代算法的可投稿假设

当前最值得保留的假设不是“FNO 一定比 DeepONet 强”，而是：

> 一个只读取部署可见 observation、显式感知 geometry/model-mismatch、以半收敛安全区为目标的轻量 warm operator，能否在不同 PoolFire trajectory 上给出可纠正初值，并以 fail-closed 回退稳定减少物理调用和端到端时间？

它最终至少需要以下一对一消融；当前首先只允许做一个 3D U-Net sentinel：

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

代码审计还要求重算成本与读取边界。两轮修正后：

- 外部 warm initializer 即使恰好输出全零，也必须支付一次初始 `A`；
- 既有 arm 缓存必须完整绑定协议、pair、角色、模型和全部 8 个 checkpoint；
- PCGLS 的 geometry-diagonal setup 同时进入单次使用与逐帧摊销 wall 账；
- v2 在 p14 rank gate 失败前不再读取 p22；
- 私有 PCA 包只保留 fit-only basis/statistics，旧的样本混合包已删除。
- v4 的显式允许名单排除 p22，连其 pair manifest 都不读取；
- 新增 pair 同时绑定请求目录、官方 trajectory/source SHA、父协议、统一几何、
  manifest/checksums/READY，两个 first-batch source 与 pair 身份必须互不相同；
- v4 只解释 observation，不解释 pair truth 数组；私有结果原子生成并可独立复核。
- 私有结果还逐文件绑定 runner、K4 teacher、CGLS/PCGLS、几何、validator 和
  straight-ray 数值路径的 SHA，并绑定 Python/NumPy 版本；实现变化时旧缓存拒绝复用。
- runner 完成后还必须通过一个不导入 runner 的独立结果 validator；它从六条官方
  `rho` bundle 逐帧重放 606 次 forward，重算 606 个 K4 teacher、fit-only PCA、
  全部 rank 指标和 20% 数学判决；独立验证失败或状态不一致时，队列不得写
  `VERIFIED_COMPLETE`。
- 审计前旧队列曾为 shape/finite 预检解释 first-batch truth 数组；这不进入 v4
  K4/PCA 判决，但历史上不能再声称“从未打开”。修正后的验证路径明确不解释 truth。

审计后重跑得到可信 v4 结果：606 帧 source-forward、606 个 K4 teacher 和
fit-only PCA 独立重算一致；13.86% 改善没有达到 20% 门，p22/test 没有被用来挽救
失败模型。加入 v5.1 独立重算后，完整 PoolFire 与聚焦页回归为 `243 passed`。这些只是代码和证据链通过，不是
算法成功。

## 8. 初学者怎么理解

把 CGLS 想成用观测不断修正三维答案。开始几步会补回真正看得见的大结构；继续跑太久，它会开始追逐 forward mismatch 和难以可靠恢复的细节。Warm start 的目标不是替代求解器，而是把起点放进“少走几步仍能到好答案”的区域。

这轮最重要的诚实结论是：我们找到了正确的困难，但还没有找到稳定战胜它的算法。
首批训练覆盖扩充有改善但没有过门，因此下一步不再盲目加同类数据，而是检查
表示是否尊重 K4 映射的线性齐次结构，以及 rank、幅值/形状分解和 view scaling
到底哪一项限制了跨工况表示。

## 9. v5.1：rank 有 headroom，但固定全局子空间仍不够

第一版 v5 因协议审计发现 K4 线性表述、oracle projection、稳定 rank、RMS 定义和
选择规则不闭合而被降级，不发布其结果。修订后的 v5.1 在结果前冻结，并明确：

- 固定四步零初值 CGLS 一般非线性，只在 breakdown 分支不变时检查一阶齐次；
- validation 系数由 K4 target oracle 投影得到，只测子空间 containment；
- raw observation 的全部 2072 分量等权计算 RMS，floor 命中必须为 0；
- rank 同时通过 fit-only `sigma_r/sigma_1`、stable rank 与边界谱隙；
- p90 `<=0.05` 与 worst `<=0.10` 必须同时通过；
- homogeneous 必须比 best raw 至少好 2% 且 worst 不变坏，才能称 material win。

runner 对全部 505 个 fit frame 的 `0.5×/2×` 探针得到最大齐次误差 0、
breakdown mismatch 0；20 个候选行均通过数值稳定门。独立 validator 随后重新计算
606 个 K4 teacher、1010 个缩放 K4、20 个 projector、稳定 rank、passer-first
选择和 homogeneous 2% + no-harm 门，得到同一结果：

| 表示 | rank | p14 p90 | p14 worst |
|---|---:|---:|---:|
| raw centered | 256 | 0.218051 | 0.251191 |
| raw origin | 256 | 0.196516 | 0.217121 |
| raw origin | 504 | 0.148973 | 0.165437 |
| observation-RMS origin | 504 | 0.148823 | 0.165473 |

rank 扩展与过原点表示带来明显 headroom，但 RMS 只比 best raw 好 `0.1009%`，
没有达到 2% material-win 门；所有候选仍未通过绝对 headroom。独立状态为
`PASS_INDEPENDENT_POOLFIRE_C_HOMOGENEOUS_REPRESENTATION_V5_1_VALIDATION`，
科学状态为 `FAIL_DEVELOPMENT_ABSOLUTE_OUTPUT_HEADROOM`。这只能排除四个固定全局
输出子空间，不能排除 nonlinear decoder、
conditional basis、mixture-of-subspaces 或 full-field CNN，更不能声称有可部署
warm start。
