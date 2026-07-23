# CG-PDNO 从本科毕设到论文：预注册式执行协议

更新时间：2026-07-15
目标：把“我设计一个比 DeepONet/FNO 更好的模型”改写为可证伪、可迁移、不会因弱基线或 inverse crime 得出假结论的研究计划。

## 1. 主假设与反假设

### H1：低调用精度

在相同 `F/F^T` 调用或端到端 wall-time 前沿上，geometry/covariance-conditioned correction 相对预白化 SPG/PBB、FNO/DeepONet warm start 和 Learned Primal-Dual 提高 sparse-view BOST 的 field/front 指标。

### H2：变化条件下的尾部可靠性

优势主要出现在 camera dropout、异方差照明、leave-one-layout-out 和 thin-front，而不是所有 IID 条件；同时 p10 gain 与 `>1% harm` 不恶化。

### H0：确定性升级解释全部收益

若 prewhitening、正确伴随、强步长/停止和相同调用预算已解释 learned gain，或独立 generator/真实数据上优势消失，则停止“新算子 superiority”，转为严格 negative/mechanism study。

## 2. 三层数据与不可越界的结论

| 层 | 数据 | 允许回答 | 禁止回答 |
| --- | --- | --- | --- |
| L0 | analytic 3D phantom + exact linear operator | 代码、伴随、消融、场真值、优化机制 | 真实 BOST 泛化 |
| L1 | S3D/独立 CFD/FiReSMOKE + independent cone/curved ray | 反应流形态和 generator shift | 实验噪声/标定真实性 |
| L2 | OpenBOST + OERF | 真实 mask、标定、noise、held-out view、运行成本 | 无独立体真值时的绝对 3D L2 |

公开数据角色详见 `public_dataset_transfer_map.md`。RealPDEBench/Michigan 是二维时间算子热身，不冒充三维 BOST 真值。

## 3. 冻结切分

- `D_train`：field family、geometry、noise realization 与实验 run 全属于训练角色。
- `V_select`：选择 epoch、超参数、trust/acceptance 参数；同一 source field 的不同视角不能跨角色。
- `V_lock`：一次性验收选择规则，不再改阈值。
- `T_iid`：新 fields、新 noise realization；分布与训练相同。
- `T_family`：thin front/激波/未见火焰形态。
- `T_noise`：相机异方差、空间相关、亮度依赖、坏像素。
- `T_geometry`：leave-one-layout/rig；不能只换 mask 而共用同一 operator id。
- `T_joint`：family + noise + geometry 同时变化。
- `T_real`：OpenBOST/OERF；最终才打开。

每次方法设计后必须生成新的 blind id；已经看过的 test 自动降级为 development。

## 4. 强基线矩阵

### 传统/数值

- ridge/Tikhonov、CGLS/LSQR、SART；
- TV/Poisson regularization；
- fixed Landweber；
- projected BB、全调用记账 nonmonotone SPG；
- prewhiten-only 与 oracle/deployable discrepancy 分栏。

### 直接学习

- 同规模 3D U-Net；
- FNO 与 F-FNO；
- DeepONet；
- padding+mask 与 permutation-invariant ray-set encoder；
- GINO-style geometry control。

### 模型驱动

- Learned Primal-Dual/同规模 unrolled；
- NeRIF/NeDF；
- physics warm-start NeRIF；
- TDBOST/framewise control（进入 4D 后）。

所有方法共用 loader、split、normalization、主指标、seed 列表和 cost logger。

## 5. 候选消融

| 编号 | 版本 | 要回答的问题 |
| --- | --- | --- |
| A0 | deterministic prewhitened base | 神经网络之前能做到多少 |
| A1 | base + unconstrained correction | 纯容量收益与 OOD 爆炸 |
| A2 | + geometry ray-set | 可变相机是否真的提供可辨识信息 |
| A3 | + covariance | 异方差收益是否超出 deterministic prewhitening |
| A4 | + residual acceptance | 尾部能否下降，额外 calls 是否值得 |
| A5 | + approximate-null UQ | 少视角多解能否被校准，而非生成漂亮方差图 |
| A6 | + 4D sparse innovation | 连续数据到位后，是否保留瞬态前缘 |

每个模块只在前一版本通过 fresh lock 后加入；不一次训练“大而全”。

## 6. 指标和调用账本

- field：relative L2、H1/gradient error、mass/peak/centroid；
- front：Chamfer、normal angle、thin-front recall/precision；
- optical：active reprojection、held-out camera、per-camera whitened residual；
- reliability：p10/CVaR gain、`>1% harm`、worst layout、risk-coverage；
- compute：参数、FLOPs、峰值内存、wall time、`F/F^T/J^T` 次数；
- statistics：source field/run cluster bootstrap，至少 5 train seeds。

判决不是“某个平均数更小”，而是 candidate 是否在预注册困难域形成 Pareto 改善且尾部不失守。

## 7. 12 周执行

| 周 | 产物 | 放行门槛 |
| --- | --- | --- |
| 1 | 线代/SVD、Radon、BOST 物理链；复跑当前 toy | 能手推伴随并解释 non-uniqueness |
| 2 | 统一 `BOSTBatch`、单位/坐标/geometry manifest | 所有 loader 通过 shape/unit tests |
| 3 | CGLS/TV/PBB/SPG + call frontier | 数值强基线冻结 |
| 4 | FNO/FFNO/DeepONet/Learned-PD 小尺度 | 同数据同预算、3 seeds |
| 5 | Base-Correction CG-PDNO | zero-init=fallback、acceptance/call tests |
| 6 | 5 seeds + IID/noise/layout/family/joint | V_lock 过 mean/p10/harm |
| 7 | independent cone/curved generator | inverse crime 下降后仍有信号 |
| 8 | 一个 3D CFD snapshot pipeline | front 指标与单位审计 |
| 9 | OpenBOST 9 support + held-out views | loader/geometry/noise/report 完整 |
| 10 | OERF 最小样例 | 伴随/VJP、真实 noise、权限确认 |
| 11 | 全消融、bootstrap、成本、失败图 | 论文硬门槛 G0-G7 审计 |
| 12 | 师兄红队、题目冻结、图表和草稿 | 不通过则收束为本科毕设/负结果 |

## 8. 本机与服务器分工

本机 Mac：8³-32³ 单元/机制、数据 loader、3 seeds、短程筛选、图表与 validator。
远端 GPU：64³-128³、完整 FNO/FFNO/DeepONet/3D U-Net、5 seeds、OpenBOST 大包、独立 nonlinear renderer。

迁移到服务器前必须交付一个自包含 run manifest：commit、config hash、data split hash、命令、环境、预计显存、checkpoint/output 路径和停止规则。

## 9. 何远哲数据最小包

首包只要 1-3 帧：九视角 displacement 或 reference/flow-on、mask/坏点、完整 ray/标定、体网格与单位、现有重构、flow-off repeats、F 与 F^T/J^T 接口、权限边界。详细字段见 `he_yuanzhe_minimum_data_contract.md`。

## 10. 投稿前的一票否决

1. 只在同一 synthetic generator 内成立；
2. 只赢未预白化或未收敛基线；
3. 额外 calls/相机/训练数据未计成本；
4. 只有 mean，没有 p10/harm/worst-layout；
5. 真实实验只有 reprojection，却写成三维真值更准；
6. 已看 test 后继续调方法，却仍称 blind；
7. 新意只能描述成“把已有模块拼起来”，没有明确 BOST 物理困难和可证伪机制。
