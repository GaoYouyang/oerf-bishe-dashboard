# PoolFire C 路线：跨轨迹实验合同

> 冻结日期：2026-07-25
>
> 状态：`PASS_FROZEN_POOLFIRE_C_CROSS_TRAJECTORY_EXPERIMENT`
>
> 证据等级：`PREREGISTERED_PROXY_EXPERIMENT_CONTRACT_ONLY`
>
> 当前突破：`algorithm_breakthrough=false`

## 先讲人话：这一步做了什么

五条公开 PoolFire 轨迹已经准备好：

- 三条只用于拟合：`p=33kw_size=01`、`p=45kw_size=05`、
  `p=58kw_size=03`；
- 第一条 validation `p=14kw_size=01` 只选择模型与正则；
- 第二条 validation `p=22kw_size=01` 只选择 correction budget、
  固定迭代深度和回退规则；
- 两条 test `p=22kw_size=05`、`p=58kw_size=01` 仍未读取。

“数据 READY”不等于“算法有效”。这份合同的作用，是在生成跨轨迹结果前
把题目、评分规则、成本账和失败条件先写死。以后不能因为某个方法暂时不好看，
就换切分、换容差、删失败帧或偷偷多跑几步。

## 我们真正要检验的命题

给定完全相同的三视角代理观测 `b`，比较：

1. Zero 初值；
2. normalized backprojection 初值；
3. 固定几何对角预条件的 PCGLS；
4. 只看观测的 dual-ridge warm start；
5. 只有经典方法留下稳定 headroom 后，才加入最小 3D U-Net/FNO。

所有初值都接**同一个**冻结的 CGLS/PCGLS refinement。候选方法只有同时满足：

- 最终 gauge-centered field relative-L2 与参考方法等价；
- 最终 gradient relative-L2 与参考方法等价；
- 最终 observation residual relative-L2 也不劣；
- 完整三视角 `A/A^T` 调用更少；
- 端到端时间和全流程峰值内存按同一范围计量；
- 逐轨迹 p90、worst 与 harm 没有被平均数掩盖；

才可能写成“同精度降低重建成本”。

## 数据角色为什么必须分开

| 数据 | 唯一职责 | 不能做什么 |
|---|---|---|
| 3 条 fit pilot | 拟合 ridge 权重、计算训练集 normalization | 使用 validation/test 统计量 |
| `p=14kw_size=01` | 从冻结候选中选模型和正则 | 进入最终权重、选择迭代步数 |
| `p=22kw_size=01` | 冻结每个 arm 的 correction budget、停止与回退 | 修改模型、特征、权重或 normalization |
| 2 条 test | 同一个冻结包的一次确认评价 | 两条之间改方法或重试 |
| 旧 `p=14kw_size=03` | 仅保留为 development 线索 | 进入 validation/test 或“独立复现”汇总 |

如果 validation 被用于职责之外的决策，它就视为 **burned**，不能继续挂着
validation 的标签。这一点能阻止“看完 p22 的结果再回头改网络”。

## 题目如何固定

### 场与网格

- 原变量：PoolFire `rho`；
- ROI：`x=[24,56)`、`y=[24,56)`、`z=[0,64)`；
- 独立参考网格：`32×32×64`；
- 逆问题网格：`16×16×32`；
- coarsening：精确 `2×2×2` block mean；
- 每帧减空间均值，只评价可辨识的 gauge-centered 场；
- `rho` 的物理单位、cell/point 语义、`rho→Δn` 和相机标定仍未闭合。

因此当前实验只能称为 **CFD density-gradient morphology proxy**，不能称为
真实或 calibrated BOST。

### 独立正演与逆算子

- 出题器：`CompositeGaussLegendreReferenceForward`；
- 答题器：`ProjectionFirstInteriorStraightRayOperator`；
- 三个轴向视角；
- 参考求积阶数为 4，最大步长是最小参考网格间距的一半；
- 参考器没有 adjoint，也不与 inverse 共用矩阵、模块实例或缓存；
- inverse 必须先通过 `2e-12` dot-test 和 `1e-12` 常数零空间门。

独立正演比“用同一个 A 生成 b 再用 A 求解”更难，但更接近实际模型失配。

## 帧不是 505 个独立样本

每条开放轨迹使用存储顺序中的全部 101 帧，不随机把同一轨迹的帧拆进不同角色。
相邻帧高度相关，所以：

1. 先在每条完整 trajectory 内计算逐帧配对指标；
2. 再报告该 trajectory 的 p50、p90、worst 和 harm；
3. 跨 trajectory 汇总时每条轨迹等权。

论文不能把 `5×101=505` 写成 505 个独立样本。当前只有三条 fit pilot 和两条
职责不同的 validation，统计结论必须保持克制。

## Normalization 与 ridge 的防泄漏规则

- observation mean/std 只由三条 fit trajectory 计算；
- field 先逐帧去 gauge，再只用 fit trajectory 的 field mean/global RMS；
- 所有 floor、clipping、mask 和常数都要保存并绑定 SHA；
- p14 只从冻结的
  `1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100`
  中选 ridge multiplier；
- 选择顺序是 p90、worst、median；p90 使用 NumPy `higher` quantile，
  平手取更强正则；
- 选完后仍只在 fit trajectory 上拟合最终权重，禁止 `fit + validation` 重训。

旧函数 `fit_calibrated_dual_ridge()` 会把 calibration 拼回训练集，因此不能直接
拿来做正式跨轨迹 runner。现在新增的
`select_train_only_standardized_dual_ridge()` 会用 train-only statistics 和
train-only final fit；validation 只选择 lambda，不改变权重样本数。

## 同精度与成本如何判

参考是 `Zero + CGLS, K=24`。每个候选深度来自冻结集合：

```text
K = 0, 1, 2, 4, 8, 12, 16, 24
```

在 `p=22kw_size=01` 上为每个 arm 固定一个 K。逐帧容差为：

```text
field:       max(0.010, 0.05 × Zero-K24 field error)
gradient:    max(0.015, 0.05 × Zero-K24 gradient error)
observation: max(0.005, 0.10 × Zero-K24 observation residual)
```

三项必须同时通过；joint pass fraction 至少 90%，harm fraction 不超过 5%。
未达标的帧必须计入失败，不能从 wall-time 表中删掉。合格候选依次按：

1. 最少完整 `A + A^T` 调用；
2. 最短端到端 wall time；
3. 更小固定 K；

选择。test/deployment 时不允许看 field truth 决定停止。

一次完整 `A` 是三个视角各 forward 一次，一次完整 `A^T` 是三个视角各 adjoint
一次。BP、`A^T b` 特征、初始投影、PCGLS setup、normalization、网络推理和
refinement 都进入账本。setup/fit 成本与单次 deployment 成本分开报告；
单次使用和可摊销使用也分开报告。内存必须是 fresh process 的 whole-arm peak RSS。

## 为什么还不训练 FNO

神经网络不是下一步的默认动作。先跑 classical gate：

- Zero + CGLS；
- normalized BP + CGLS；
- 真正固定几何对角预条件的 PCGLS；
- train-only dual ridge + CGLS。

只有 dual ridge 在两条 validation 的分工链上，仍稳定保持“同精度、低调用、
低 harm”的余量，才允许冻结神经训练协议。首批只比较：

- `normalized A^T b + fixed sensitivity → 3D U-Net → x0`；
- `normalized A^T b + fixed sensitivity → 3D FNO → x0`。

BP 的 `A^T` 成本必须计入，不能让神经模型免费获得经典方法要付费的特征。

## 测试集怎样保持一次性

当前没有 `TEST_RELEASE.json`，所以 test 不能读取。将来 opening 前必须生成同一个
联合冻结包，同时绑定两条 test，至少包含：

- 帧清单、reference/gauge、normalization constants；
- proxy forward、inverse geometry、solver；
- 模型权重、每个 arm 的固定 K、matched tolerances；
- 全部候选、种子、排除规则和报告模板；
- 上述文件各自的 SHA。

两条 test 必须在同一个 atomic release 中授权；打开第一条后，到第二条结束前，
代码、配置、权重和报告模板都不能改变。不能先看 `p=22kw_size=05`，再修改方法去
跑 `p=58kw_size=01`。

## 现在能说什么

可以说：

- 三条 train pilot 与两条职责分开的 validation 均已通过数据完整性门；
- test truth 仍未打开；
- 跨轨迹题目、数据角色、评分、成本和 test release 条件已机器冻结；
- 下一步可以实现正式 classical-control runner。

不能说：

- 已证明跨轨迹泛化；
- ridge、FNO、DeepONet 或我们的算法已经获胜；
- 已实现真实 BOST、`rho→Δn` 或相机标定；
- 已取得 wall-time、内存或算法突破；
- 已经形成可投稿论文结果。

## 机器复现

协议文件：

```text
learning_labs/protocols/poolfire_c_cross_trajectory_experiment_v1.json
```

验证：

```bash
python site_tools/validate_poolfire_c_cross_trajectory_experiment.py \
  learning_labs/protocols/poolfire_c_cross_trajectory_experiment_v1.json

python -m pytest -q \
  site_tools/test_validate_poolfire_c_cross_trajectory_experiment.py
```

当前合同 SHA-256：

```text
bb536f8b0f25a4ced47263798b7d5cdb0efa94308d3796a6ced6e7c37fd1d76b
```

当前结果：`13 passed`。这些测试证明合同会拒绝角色混用、validation refit、
逐帧伪独立、truth-based stopping、假 PCGLS、过早神经训练、非联合 test release
和越界成功声明；它们不证明算法性能。

另外，正式 runner 的两个首要接口阻塞已移除：

- `VerifiedPoolFireRhoBundle` 现在必须按预期 trajectory/split 绑定 validation，
  并无条件拒绝 test split；
- train-only standardized dual ridge 已实现 deterministic model hash、训练集
  featurewise normalization、训练集 field scaling 和 validation-only lambda
  selection。

相关 loader、ridge 与合同定向测试合计 `28 passed`。这仍是实现代码门，不是
classical baseline 结果。

## Pair 准备进度

五条开放轨迹已经按合同生成：

- 每条 101 帧，共 505 帧；
- observation 为 `(101,2072)` float64；
- gauge truth 为 `(101,16,16,32)` float64；
- 五条轨迹共享同一个冻结几何；
- 高分辨率原场不复制进 pair bundle；
- 每个 bundle 均有 manifest、payload checksums、READY 和独立 validator；
- test pair generator 明确拒绝两条 untouched test。

聚合输入审计状态为 `PASS_FIVE_OPEN_TRAJECTORY_PAIR_AUDIT`。同一几何下，
reference-vs-inverse 投影失配的逐轨迹 p50 从 `34.0%` 到 `51.0%`，worst 从
`39.5%` 到 `55.7%`。这说明 coarse inverse 的模型失配随流场工况变化，固定
correction budget 需要严格的逐轨迹 harm 门；它不是任何算法的精度或速度结果。

## 下一步

1. 冻结 train-only normalization constants 和 ridge candidate report；
2. 写真正 geometry-only sensitivity PCGLS，不能用 identity 冒充；
3. 用 fresh process 跑四个 classical arms；
4. 在 p14 只选 ridge，在 p22 只冻 K/回退；
5. 逐 trajectory 生成 matched-accuracy、harm、calls/wall/memory 表；
6. 结果不够好就保留负结果，不开 test、不硬上 FNO。
