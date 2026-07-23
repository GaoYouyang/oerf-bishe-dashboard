# LGWO-A24 O1：精确方向表示筛查结果

**机器状态：** `OPENED_EXACT_NULL_HEADROOM_OBSERVED_NO_MODEL_AUTHORIZATION`

**可说的结论：** 在 6 个已经打开的 JACRU 合成 case、3 个几何簇上，如果评估器直接提供真值的精确离散零空间方向，LGWO-A24 的第一方向壳层存在稳定 field/H1 改善，同时保持 CGLS-24 的测量残差。

**不可说的结论：** 这不是可部署算法、不是训练结果、不是未见 rig 泛化、不是 OERF/真实 BOST 验证，也不是相对 NeRIF、DeepONet、FNO 或其他方法的胜利。

**突破监测：** 尚无算法突破；新增的是经过独立复算的表示层正信号。

## 1. 为什么先跑真值方向，而不是直接训练网络

LGWO-A24 只允许网络扰动第一条搜索方向：

```text
g = P_S A^T y
z = g + delta
```

随后进行一次精确线搜索和 23 个 measurement-space 共轭方向步，总预算严格为 `24F/24A^T`。在花时间训练前，先问一个更基础的问题：即使给出理想方向，这个受限接口是否有足够的 field headroom？如果理想方向都过不了门，神经网络没有训练价值。

O1 使用 dense SVD 只在评估器中把真值分为：

```text
x_truth = x_row + x_null
x_row  in range(A^T)
x_null in ker(A)
```

三个 retrospective oracle 臂分别把 `x_null`、`x_row` 或完整 `x_truth` 作为 raw correction。所有 correction 都经过同一个 `eta ||A^T y||` 范数限制，再进入完全相同的 LGWO-A24 壳层。dense SVD 的每个几何簇需要 `1001F/0A^T`，只记作评估器 setup，不能进入部署算法成本。

## 2. 运行前冻结合同

| 项目 | 冻结值 |
|---|---|
| split | 已打开的 `development`，只作机制描述 |
| case / geometry cluster | 6 / 3 |
| family | `smooth_no_interface`、`single_interface` |
| grid / camera / detector | `12^3` / 3 / `6x6` |
| oracle arms | exact-null truth、exact-row truth、full truth |
| relative radius | `0.05`、`0.10`，不选最优 eta |
| baseline | 同一壳层的 `delta=0`，数值恢复 CGLS-24 |
| algorithm budget | 每条 baseline/candidate 均为 `24F/24A^T` |
| field / H1 mean gate | `>=5% / >=3%` |
| measured / clean mean ratio | `<=1.000001 / <=1.000001` |
| tail gate | harm `<=5%`，worst field gain `>=-5%` |

配置冻结前，`seed=2113 / single_interface` 的 `eta=0.05,0.10,0.25` 已被查看。因此整个 O1 都是 opened descriptive screen，不能选择 eta、模型、split 或下一批 fresh 路线。

## 3. 三臂结果

| Oracle direction | eta | Mean field gain | Mean H1 gain | Measured / same-A clean-target ratio | Worst field gain | Gate |
|---|---:|---:|---:|---:|---:|---|
| exact null truth | 0.05 | **+6.978%** | **+6.566%** | 1.000000 / 1.000000 | **+3.741%** | 9/9 descriptive thresholds reached |
| exact null truth | 0.10 | **+13.722%** | **+13.005%** | 1.000000 / 1.000000 | **+7.262%** | 9/9 descriptive thresholds reached |
| exact row truth | 0.05 | -0.011% | -0.015% | 0.979014 / 0.998891 | -0.031% | field/H1 fail |
| exact row truth | 0.10 | -0.021% | -0.028% | 0.959528 / 0.997878 | -0.060% | field/H1 fail |
| full truth | 0.05 | +3.047% | +2.855% | 0.981168 / 0.999013 | +1.704% | field/H1 fail |
| full truth | 0.10 | +5.908% | +5.549% | 0.963540 / 0.998102 | +3.277% | 9/9 descriptive thresholds reached |

所有 36 条 candidate 的最大 measurement-direction orthogonality defect 为 `4.441e-16`，最大逐步 residual increase 为 0；每条算法调用账本都是 `24F/24A^T`。这里的 clean target 是同一离散 `A` 下去掉噪声/偏置的观测，不是独立 forward `B`，所以 exact-null 臂的 measured/clean residual 不变主要是构造保证。这些阈值也没有覆盖 held-out ray、三模型种子、OOD 或 fresh，不能称为完整 development gate。

## 4. 最窄的因果解释

1. **当前 toy 的主要重建余量位于离散不可观测部分。** exact-null direction 在 6/6 case 都改善 field，而不改变测量残差。
2. **这个方向对照不支持“只增强 row-space 就能改善 field”。** exact-row direction 虽降低 measured residual，却没有改善 field/H1；这不排除其他 row/near-null 方法。
3. **三臂不能做加性 kernel 因果分解。** 三条 raw direction 分别被归一化和 clipping，虽然同 case/radius 的 applied norm 一致，但 `delta_full != delta_null + delta_row`。full truth 在小半径不如 pure-null 只是一条描述性现象，不能据此计算 row/null 的独立贡献。
4. **“存在可学目标”不等于“观测中含有足够信息学到它”。** exact-null truth 直接读取了部署时不可用的三维真值。真正的研究难点是从 flow morphology、ray geometry、跨帧相关性和训练分布中推断一个有用的 approximate-null prior，同时在未见 rig 上拒答。

不能把 `ker(A)` 直接解释成真实光学零空间。这里的 kernel 依赖 `12^3` 网格、三相机、离散射线、support 和边界条件；提高分辨率、改变相机数或使用弯曲光线都会改变它。

当前离散算子有 1000 个 active voxels、216 个 measurement degrees of freedom，数值秩为 216，因此 nullity 为 784。这个巨大欠定性正是物理问题的核心：重投影更小或不变都不能单独证明三维场更准确，缺失部分必须来自可独立检验的流动物理先验。

## 5. 独立复算

复算 validator 没有导入正式 O1 runner。它从冻结配置重新：

1. 构造 6 个 JACRU case；
2. 为 3 个几何簇重建 dense projector；
3. 重跑 36 条 baseline/candidate 路径；
4. 对逐行指标、6 个聚合单元、SVD setup、调用账本、哈希和所有关闭的授权字段执行 1,121 项断言。

结果为：

```text
VALIDATION_PASS_OPENED_EXACT_NULL_HEADROOM_NO_AUTHORIZATION
```

复算证明结果在同一 fixture、projector、solver 与 metric 依赖下可重现；它不能发现所有共享公式错误，更不能替代独立物理 forward 或实验验证。

完整 O1 执行的 operator 账本为：dense setup `3003F/0A^T`，6 条 baseline 算法 `144F/144A^T`，36 条 candidate 算法 `864F/864A^T`，评分 `42F/0A^T`，合计 `4053F/1008A^T`，另有 3 次 `216x1000` dense SVD。当前包没有正式 wall time、peak memory 或 SVD 成本审计，所以只能说每条 solver path 的在线调用相同，不能声称完整端到端成本公平。

## 6. 下一步只允许做什么

O1 只授权我们**设计并冻结**一个小型 learnable-direction pilot；它本身不自动授权扩大模型或打开 OOD/fresh。

最小模型应满足：

- 部署输入只含 raw displacement、一次 pooled `A^T y`、ray/camera geometry 和 support；
- 参数不超过 8k，correction head 零初始化，首轮固定 `eta=0.05`；
- 训练时可用 synthetic truth，但 route selection、fallback 和正式评分不能读 truth；
- 训练诊断可以报告 correction 与 exact-null truth 的 cosine/alignment，但不得把该值输入模型或选择器；
- 同时比较 `delta=0`、简单阻尼、无 geometry、无 raw-y、FNO/DeepONet tiny budget 和直接 field warm start；
- 任一 field/H1、measured/clean、逐 case tail、调用预算或 fallback 门失败就停止，不靠换 seed、eta 或 split 补救。

## 7. 证据入口

- 冻结配置：`demo_t16_operator/configs/lgwo_a24_exact_direction_oracle_opened_v1.json`
- 正式 runner：`site_tools/run_lgwo_a24_exact_direction_oracle.py`
- 原始 36 行：`demo_t16_operator/results/lgwo_a24_exact_direction_oracle_opened_v1/rows.csv`
- 聚合指标：`demo_t16_operator/results/lgwo_a24_exact_direction_oracle_opened_v1/aggregate.csv`
- 机器摘要：`demo_t16_operator/results/lgwo_a24_exact_direction_oracle_opened_v1/summary.json`
- 独立 validator：`site_tools/validate_lgwo_a24_exact_direction_oracle.py`
- 独立 validation：`demo_t16_operator/results/lgwo_a24_exact_direction_oracle_opened_v1/independent_validation/validation.json`
- 论文式图：`demo_t16_operator/results/lgwo_a24_exact_direction_oracle_opened_v1/diagnostic_v2.pdf`

这一步最重要的真实进展不是“网络已经赢了”，而是我们第一次用严格同预算壳层和因果对照确认：**在当前有限角 BOST toy 中，值得让模型学习的是不可见结构先验，而不是继续用昂贵网络重复可见空间的最小二乘拟合。**
