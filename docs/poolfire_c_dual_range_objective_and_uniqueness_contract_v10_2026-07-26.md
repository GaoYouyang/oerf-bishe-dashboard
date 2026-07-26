# DualRange-K1 v10：优化目标、预期优势与独特性合同

**日期：** 2026-07-26
**机器合同：** `learning_labs/protocols/poolfire_c_dual_range_outer_contract_v10.json`
**合同 SHA-256：** `882364ee37a2665904b7538f1b8cec81fce6b587c258c5d277ab159f23b9048c`
**当前状态：** `FROZEN_OBJECTIVE_AND_EVIDENCE_CONTRACT_BEFORE_MODEL_TRAINING_PREDICTION_OR_SCORE`

## 先说结论

我们的唯一主线已经确认，不再是泛泛“做一个 FNO/DeepONet”：

> **在三维 BOST 代理逆问题中，让网络只提出观测域双变量，经一次物理伴随算子
> `A^T` 生成受约束初值，再用未修改的 CGLS K1 保持在 Zero-CGLS K4 的冻结开发
> compatibility envelope 内，同时降低完整 `A/A^T`、端到端时间且不增加峰值内存。**

当前机制的作品指纹是：

```text
BOST-specific observation-space dual proposal
+ by-construction Range(A^T) lift
+ pre-A^T deployment-visible accept/fallback gate
+ unchanged CGLS K1
+ field/gradient/observation trajectory-level non-inferiority
+ complete A/A^T, wall, RSS and tail accounting
+ mandatory real-BOST transfer
```

这是一种**组合级原创性假设**。截至 2026-07-26 的有界一级来源检索中，尚未找到
包含完整七件套的同构方法；但这不是“全球绝对唯一”或专利自由实施证明。真正严谨的
说法应是：

> 在已审阅公开一级来源与已知组内信息范围内，未发现与完整方法、部署门和证据合同
> 同构的 BOST 重建方案。

## 1. 固定优化目标

判决采用词典序，后面的收益不能抵消前面的失败：

1. 每条完整 trajectory 的 field、gradient、observation 单侧非劣；
2. 每条轨迹的 p50、p90、worst、harm 与 severe-harm；
3. 完整多视角 `A/A^T` 调用；
4. fresh-process 端到端 wall time；
5. whole-pipeline peak RSS；
6. 组内真实 BOST 迁移。

主参考是 Zero-CGLS K4：

| 方法 | 完整 A | 完整 A^T | 当前角色 |
|---|---:|---:|---|
| Zero-CGLS K4 | 4 | 4 | 精度参考 |
| DualRange-K1 接受分支 | 2 | 2 | 主候选 |

如果 DualRange-K1 真正通过同一冻结终点，接受样本的完整算子对理论上减少 **50%**。
部署目标进一步要求：

- 五条 outer trajectory 等权 wall 中位数至少下降 **10%**；
- 任一 outer 或 p14 行不得慢超过 **5%**；
- 每行 fresh-process peak RSS p90 不超过 Zero-K4 的 **1.05 倍**；
- 任一轨迹的材料性伤害都不能被平均收益掩盖。

这三个数来自已冻结的 v9.4/v9.4.1 开发协议，结果出现后不得下调。v10 共有
**17 个正式 arms、6 行完整轨迹、102 个原子 prediction**；ungated DualRange 与
pre-`A^T` gated policy 必须同时报告，不能只展示 gate 接受的简单帧。它们目前只是
公开 straight-ray proxy 的开发门，不等于真实实验的“同精度”。真实 BOST 非劣界仍
必须由相机重复测量、标定误差、噪声或空间分辨率给出物理依据。

## 2. 主算法为什么可能有优势

主候选写成：

```text
z_theta = G_theta(y, geometry-visible summaries)
h       = A^T z_theta
alpha   = argmin_[0, alpha_max] ||y - alpha A h||_2^2
x0      = alpha h
x1      = unchanged CGLS K1(x0)
```

### 结构优势

1. **网络不自由猜三维场。** 它只在 2072 维观测空间提出 `z_theta`，再由真实
   `A^T` 提升到 8192 维场空间。
2. **可观测子空间约束是构造成立的。** `x0=alpha A^T z_theta` 必然属于
   `Range(A^T)`，网络无法直接往至少 6120 维不可观测零空间里写任意纹理。
3. **经典求解器不改。** 后端仍是未修改 CGLS K1，因此可以独立核对调用、残差和
   结果，也更容易接入组内 solver。
4. **初始 measurement residual 不增加。** 解析尺度区间包含 `alpha=0`。这只约束
   初始观测残差，不能冒充 field/gradient no-harm。
5. **可以在付出候选算子调用前回退。** pre-`A^T` 风险门若拒绝，直接运行
   Zero-K4；接受率为 `p` 时，平均完整算子对是 `4-2p`。

### 证据优势

作品不只报“平均 field-L2 好了一点”，而要求：

- 五条完整 outer LOTO 轨迹逐条过门；
- 已见 p14 单独作 mandatory veto；
- 对比同预算 Zero-K2、normalized-BP、解析二维 Galerkin、固定 dual filter 和
  dual ridge；
- 对比自由三维 direct-field model 与 Cross14，证明 range 限制是否真正有用；
- 同时报 ungated 全帧候选与 gated policy，且每条轨迹接受率至少 50%；
- 同时报 `A/A^T`、非算子 FLOP、模型字节、wall、RSS、接受率和错误接受率；
- p45-s03 fresh holdout 与两条 test 在独立 release 前继续封存。

这套证据设计比“只拿 DeepONet/FNO 当弱基线”更难过门，也更有论文说服力。

## 3. 哪些东西明确不是我们的首创

| 已有方向 | 代表一级来源 | 对我们的约束 |
|---|---|---|
| BOST 神经隐式重建 | [NeRIF](https://arxiv.org/abs/2409.14722)、[NeDF](https://arxiv.org/abs/2409.19971)、[Neural Refractive Index Primitives](https://arxiv.org/abs/2605.11454) | 不能说“首次用神经网络做三维 BOST” |
| 学习 warm start 后接固定迭代器 | [Learning to Warm-Start Fixed-Point Optimization Algorithms](https://www.jmlr.org/papers/v25/23-1174.html) | warm start + 固定求解器本身不新 |
| 神经算子给经典 PDE solver 初值 | [PFEM](https://arxiv.org/abs/2601.03086)、[MD-PNOP](https://arxiv.org/abs/2509.01416) | “算子学习加速经典 solver”本身不新 |
| 神经算子与迭代器互补 | [HINTS](https://www.nature.com/articles/s42256-024-00910-x) | 网络补慢模态或低频本身不新 |
| 学习反投影或投影域滤波 | [Learned Backprojection](https://arxiv.org/abs/1908.00593)、[FNO-BP](https://arxiv.org/abs/2402.12141) | `G(y)` 后反投影本身不新 |
| 学习共轭梯度预条件器 | [CNN-driven preconditioners for CG](https://doi.org/10.1088/2632-2153/ae76ba) | learned preconditioner/Krylov acceleration 本身不新 |
| 学习 warm basis 后做 Krylov 投影 | [WB-IPM](https://arxiv.org/abs/2510.05926) | “学习信息引导 Krylov 子空间”已有高度相近先例 |
| 学习模型误差并结合经典搜索方向 | [Learned ReSeSOp](https://doi.org/10.1088/1361-6420/adef73) | 数据一致性、学习校正与子空间迭代的组合不新 |

因此，不能靠“用了算子学习”“输出观测域”“后面接 CGLS”中的任一单点宣称首创。
独特性只能来自**完整机制、BOST 场景、预算子回退和严格证据同时成立**。

当前最危险的结构近邻是 2025 预印本 WB-IPM。它让网络生成解空间 warm basis，再使用
增强 Golub-Kahan/Krylov 投影；我们的剩余差异是观测域 proposal、精确 `A^T` 提升、
未修改 CGLS K1、pre-`A^T` 回退和 `2A+2A^T` 账。这些只是**待验证差异**，不能在
outer 和真实 BOST 证据通过前写成贡献。

## 4. 预期产出分三级

### A. 机制论文素材

- 证明 `Range(A^T)`、条件常数零空间正交和初始 residual 不增加；
- 审计接受分支严格 `2A+2A^T`；
- 证明隐藏算子调用、错误 shape、非有限输出会 fail closed；
- 公开完整负结果，包括旧 GEOK 与失败候选。

这一级现在已有代码和测试，但还不是算法性能成果。

### B. 公开代理方法成果

只有以下全部成立才达到：

- 五条 outer LOTO 与 p14 veto 都过单侧非劣门；
- 主候选逐条不劣于最强同价或更便宜 control，并至少一项 p90 改善 2%；
- 接受分支保持 50% 完整算子对减少；
- wall 达到冻结的 10% 目标，RSS 无害；
- 独立 validator 复算，P0/P1 为零。

这一级可形成扎实的本科毕设核心和方法论文初稿，但仍只能叫
“公开 CFD straight-ray BOST proxy 结果”。

### C. 可投稿的 BOST 成果

在 B 的基础上还需：

- p45-s03 fresh holdout 一次性确认；
- 至少一个组内真实 BOST case；
- 真实相机、单位、reference、光流、噪声与标定误差闭合；
- 用真实不确定度定义非劣界；
- 师兄确认组内没有未发表同构方案或专利冲突。

只有到这一级，论文摘要才可以写“propose a new BOST reconstruction
acceleration method”。期刊接收仍不能被任何代码或合同保证。

## 5. “独一无二”怎样变成可防御

我们不承诺无法证明的全球唯一，而用五层指纹让作品难以与别人混淆：

1. **问题指纹：** 三维 BOST 的多视角密度梯度逆问题；
2. **表示指纹：** 学习 2072 维双空间 proposal，不自由输出 8192 维场；
3. **数学指纹：** `A^T` 提升、CGLS range 兼容、条件零空间命题；
4. **部署指纹：** pre-`A^T` 风险门与清楚的接受/回退平均成本公式；
5. **证据指纹：** 三指标、逐轨迹尾部、同预算强对照、调用/wall/RSS、真实迁移。

即使未来找到某篇包含其中两三项的论文，剩余维度仍可继续形成可区分贡献。若找到完整
同构工作，就必须诚实改题、缩小主张或停止该原创性路线。

## 6. 当前真实状态

```text
optimization_objective_frozen=true
combination_level_fingerprint_frozen=true
dual_range_mechanism_implemented=true
accepted_branch_call_ledger=2A+2AT
reference_call_ledger=4A+4AT
same_budget_strong_controls_required=true
formal_arm_count=17
formal_prediction_count=102
pre_AT_gate_implemented=false
formal_capability_isolation_proven=false
model_architecture_frozen=false
model_training_authorized=false
outer_performance_opened=false
fresh_holdout_opened=false
real_BOST_transfer=false
global_uniqueness_proven=false
algorithm_breakthrough=false
```

合同本身已经过两轮独立红队。第二轮修复后，14 项 fail-closed 定向测试会拒绝阈值、
调用账、truth 白名单、训练授权、score-token bindings、failure actions、receipt
identity 和突破声明的篡改；与机制、聚焦页和 Pages builder 的联合检查为
`117 passed`。这证明目标/证据口径被锁住，不是 102 个预测或算法结果已经产生。

下一步不是立刻开 FNO/DeepONet，而是结果前冻结**唯一一个最小 `G_theta` 架构**、
参数上限、fit-only nested trajectory 选择、K1 后损失、种子和 checkpoint 规则；
同时实现 17-arm registry、pre-`A^T` gate 状态机、逐帧调用凭据，以及只给模型进程
`y` 和白名单特征的 `execve` 能力隔离。当前 `truth_blind_attested=true` 只是声明，
不是隔离证明。完成这些并生成新的 runner/validator hash 后，才有资格生成 102 个
v10 outer predictions。

## 7. 需要师兄确认的四件事

1. 组内是否已有未发表的 observation-space filter、learned BP、Krylov initializer、
   warm start、pyramid initializer、学生论文或专利？
2. 真实流程中一次 `A`、一次 `A^T`、光流、曲光线 ray tracing 和外层优化分别多慢？
3. 真实 BOS 重复测量与标定误差是多少，能否据此给出三类非劣界？
4. 公开 proxy 过门后，哪个最小真实 case 可提供 observation、geometry、reference 与
   现有 solver 结果做迁移验证？

这四项是把“独特的公开代理作品”升级为“课题组真正需要的可投稿方法”的最后外部接口。
