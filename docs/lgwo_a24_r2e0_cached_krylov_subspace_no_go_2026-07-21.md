# R2-E0 缓存 Krylov 子空间重优化：有效 NO-GO 与下一条算法路

## 一句话结论

R2-E0 完整保留冻结 H1-20 的 `20F/20A^T`，缓存 20 个场方向与对应投影，随后用
`0F/0A^T` 在这 20 维子空间内比较数据最小二乘、简单插值、四个固定 Huber 权重和
truth-only oracle。缓存合同通过，但没有固定 Huber 候选取得可部署资格；即使把合成
truth 交给 residual-safe oracle，mean field/gradient gain 也只有
`+0.0040% / +0.1647%`，远低于预写 `5% / 5%` 门。

因此当前可写的是：**冻结 H1-20 子空间缺少数据一致的联合场/梯度表示余量；继续在同一
子空间调 Huber 权重或训练权重选择器没有依据。** 这不是“所有 Huber、Krylov 或算子学习
都无效”，更不是算法、真实重建、泛化或论文成功。

## 1. 运行边界

| 项目 | 冻结合同 |
|---|---|
| 数据 | 6 个已打开解析反应形态 proxy；公开 PSU 几何 |
| forward | truth 与 reconstruction 使用同一 QMC8 算子，属于 inverse-crime 型诊断 |
| 基线 | H1-20，`lambda_H1=1e-3` |
| 共享物理预算 | 每 case `20F/20A^T` |
| reduced 预算 | 所有系数重优化 `0F/0A^T` |
| 固定候选 | data exact、`beta={.25,.5,.75}`、Huber ratio `{.25,.5,1,2}` |
| evaluator-only | joint field/gradient truth oracle 与 residual-safe ray |
| fresh / real | 都未打开 |

记缓存场方向为 `P=[p_1,...,p_20]`，加权数据投影为 `B=A_w P`，冻结 H1 系数为
`c_H1`。任意 reduced 候选的观测残差可直接写成：

```text
r(c) = r_H1 - B (c - c_H1)
```

所以系数优化不需要再次调用物理 `A` 或 `A^T`。固定 Huber 目标为：

```text
J(c) = 0.5 ||r(c)||_2^2 + lambda_eff H_delta(Pc)
```

其中 `lambda_eff` 只由 H1 点的数据损失、Huber 势能和预写 balance ratio 定标。每一步还
必须满足 residual envelope、field trust radius、Huber nonincrease；失败就精确回退 H1。

## 2. 没有把运行错误藏起来

1. **v1 无效：** reduced least-squares 的合并 `MPS float32 -> CPU float64` 转换污染右端项，
   第一 case 即失败；结果目录只保留 `invalid_attempt.json`，禁止同配置重跑。
2. **v2 无效：** v1 修复只覆盖最小二乘，Huber reoptimizer 仍直接在 MPS 请求 float64；同样
   没有产生算法结果，并单独保留失败证据。
3. **v3 有效：** 所有 CPU-double 工作区统一为“先复制到 CPU，再升 float64”；返回时先降到
   目标 dtype，再进入 MPS。最小二乘、完整 Huber 和 residual-safe oracle 都有 MPS 定向回归。

v1/v2 不能合并到科学结果，也不能作为“失败 case”参与均值。它们只是可追溯的软件失败。

## 3. 有效 v3 数字

### 3.1 缓存与调用合同

- 6/6 case 的 cached path 与冻结 H1 reference 均过门；
- 最大 field relative difference：`3.659e-7`；
- 最大 residual relative difference：`1.666e-6`；
- 最大投影方向正交缺陷：`8.285e-6`；
- 每 case reference 与 cache 都是精确 `20F/20A^T`；
- reduced stage 全部为 `0F/0A^T`；
- 结果矩形为 `6 cases x 12 methods = 72 rows`。

运行后另写了一个不导入 runner/core 的独立 validator，从 CSV/JSON/PNG、Git blob 与几何输入摘要
复算 `873` 项，返回 `VALIDATION_PASS_POST_RUN_INTERNAL_CONSISTENCY_ONLY`。它能证明已保存证据内部
一致，但因为验证器在正式运行后实现，不能冒充运行前协议约束，也不能证明真实重建、泛化或算法优越性。

### 3.2 主要候选

| 方法 | mean field gain | mean gradient gain | residual ratio | Huber ratio | accepted cases | 判决 |
|---|---:|---:|---:|---:|---:|---|
| data exact | `+0.1184%` | `-1.6445%` | `0.8951` | `1.0505` | 6/6 | 数据更合，梯度更坏 |
| Huber .25 | `+0.000002%` | `0.0000%` | `1.0000` | `1.0000` | 1/6 | 不合格 |
| Huber .50 | `+0.000002%` | `0.0000%` | `1.0000` | `1.0000` | 2/6 | 不合格 |
| Huber 1.00 | `+0.000002%` | `0.0000%` | `1.0000` | `1.0000` | 3/6 | 不合格 |
| Huber 2.00 | `+0.000002%` | `0.0000%` | `1.0000` | `1.0000` | 4/6 | truth-free 诊断对象，但不合格 |
| truth oracle safe | `+0.0040%` | `+0.1647%` | `1.0200` | `0.9960` | evaluator-only | 不过 `5%/5%` oracle 门 |

四个 Huber 候选的非零更新只有约 `1e-10` field relative shift，本质是浮点尺度的 no-op。
`huber_ratio_200` 只是按 truth-free 规则选出的诊断对象；它的 accepted fraction 为 `4/6`，低于
预写 `0.8`，所以“被选中”不等于“合格”。

### 3.3 oracle 为什么特别重要

不受 residual 约束的 truth joint oracle 有 `+14.52%` gradient gain，但 field 平均恶化
`-2.384%`，观测 residual 变成 H1 的 `10.23x`。把它沿 H1 点到 oracle 的射线压回 `1.02`
residual envelope 后，每个 case 只允许走约 `0.50%--0.92%` 的路程，最终只剩
`+0.0040% field / +0.1647% gradient`。

这比“固定 Huber 没调好”更强：**连知道 truth 的上界也无法在当前 20 维 span 内同时保住数据并产生
足够场/梯度收益。** 当前瓶颈是方向空间，不是四个 balance ratio 的选择器。

## 4. 这轮关闭什么，不关闭什么

### 关闭

- 在同一 H1-20 span 上继续加密 Huber ratio；
- 训练 MLP/DeepONet/FNO 去预测这四个固定权重；
- 把极小 accepted step 或 mean field 微小正数写成算法信号；
- 仅凭 unrestricted truth oracle 的 gradient 提升宣称可部署 headroom。

### 没有关闭

- 产生新方向的 flexible/generalized Krylov；
- 在迭代中改变先验或预条件器、从而真正改变 span；
- 与真实 BOST 曲光线、标定误差、位移噪声相关的方向设计；
- 有严格 data-consistency 投影与拒答门的 learned direction generator。

## 5. 下一条最窄可执行路线：R2-F0 span expressivity audit

下一步不训练大模型，先问“什么新方向真的能跳出当前 span”。建议只比较三个小而明确的方向族：

1. **Flexible prior-conditioned direction：** 用固定 H1/Huber 局部预条件器作用于 `A^T r`，
   生成不属于原始 H1 Krylov span 的方向；
2. **Residual-safe edge direction：** 由当前场的 edge/Hessian proxy 产生一个方向，只支付一次对应
   投影，并沿 data envelope 做解析回退；
3. **BOST geometry/noise-conditioned direction：** 输入只能是 view mask、noise scale、residual/Ritz
   轨迹和几何摘要，输出方向必须经过投影正交化与 residual gate。

R2-F0 首先只做 representation oracle：同预算下比较 H1 span 与扩展 span 的 truth-only 上界。如果扩展
span 在 6 个已打开 case 上仍没有显著 joint headroom，就停止；如果有，再冻结 fresh seed 协议。只有
部署可见特征能预测方向或接受/拒绝，才进入小网络。

## 6. 新颖性边界：不能把常见组合当发明

- [Hybrid Projection Methods with Recycling](https://arxiv.org/abs/2007.00207) 已研究逆问题中的
  Krylov 基压缩、回收与正则化参数选择；
- [Hybrid Projection Methods survey](https://arxiv.org/abs/2105.07221) 已系统总结 Krylov 与
  variational regularization 的结合；
- [FCG-NO](https://proceedings.mlr.press/v235/rudikov24a.html) 已把 neural operator 用作 flexible CG
  的非线性预条件器；
- [Neural Krylov Iteration](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e88870ec82f2469b0ddf32c817920c68-Abstract-Conference.html)
  已学习不变子空间来加速线性求解。

因此可争取的差异不是“首次 neural + Krylov”，而应是：**BOST 特定几何/噪声/曲光线失配下，具有
可核验 data envelope、拒答和跨 rig 尾部门的方向生成。** 这仍只是候选命题，尚无真实证据。

## 7. 现在要问何远哲师兄的 8 个问题

1. 真实 reconstruction callable 的输入/输出 shape、单位和坐标约定是什么？
2. straight-ray 与 curved-ray residual 分别在哪一层计算？
3. 能否调用 `A(x)`、`A^T(y)`，以及 JVP/VJP？
4. 主要痛点是收敛慢、边缘糊、稀视角、标定误差，还是时间一致性？
5. 当前必须击败的实验室基线和预算是什么？
6. 是否有 flow-off、重复帧、标定不确定度和逐视角噪声估计？
7. 数据能否按 rig/工况/时间段切 train/validation/test，而不是随机帧切分？
8. 师兄希望本科毕设更偏 solver、neural field，还是 4D tensor reconstruction？

在这些合同确认前，R2-F0 只能是公开 PSU 上的可复现实验，不能替实验室回答真实物理问题。

## 8. 证据入口

- 冻结 v3 配置：`demo_t16_operator/configs/lgwo_a24_r2e0_cached_krylov_diagnosis_v3.json`
- v3 报告：`demo_t16_operator/results/lgwo_a24_r2e0_cached_krylov_diagnosis_v3/report.json`
- 逐 case 指标：`demo_t16_operator/results/lgwo_a24_r2e0_cached_krylov_diagnosis_v3/metric_rows.csv`
- 候选汇总：`demo_t16_operator/results/lgwo_a24_r2e0_cached_krylov_diagnosis_v3/candidate_aggregates.csv`
- 873 项独立验证：`demo_t16_operator/results/lgwo_a24_r2e0_cached_krylov_diagnosis_v3/independent_validation.json`
- 独立 validator：`site_tools/validate_lgwo_a24_r2e0_cached_krylov_diagnosis.py`
- v1/v2 无效尝试：各版本结果目录的 `invalid_attempt.json`

## 突破监测

**没有突破。** 新增的是一个执行有效、可复核的 post-open synthetic NO-GO：它排除了“在冻结 H1-20
span 内调固定 Huber 权重或学权重选择器”这条窄路，并把下一问题定位为“能否生成真正扩展 span、又满足
数据安全门的新方向”。真实 BOST、fresh seeds、unseen rig、与 DeepONet/FNO/NeRIF/TDBOST 的胜负、论文
成功和突破仍全部为 0。
