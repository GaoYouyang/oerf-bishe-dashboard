# LGWO-A24 R2-D1：边缘修正有局部信号，但替换 H1 迭代是有效 NO-GO

> 日期：2026-07-21<br>
> 证据等级：已打开的 6 个公开 PSU 几何合成 case 上的 post-open 机制诊断<br>
> 结论：`POSTOPEN_R2D1_NO_GO_NO_FRESH_VALIDATION`<br>
> 禁止解释：真实 BOST、未见 seed/rig 泛化、自有算法成功、论文成功或突破

## 1. 这轮真正问了什么

冻结 H1-20 已经是当前必须击败的强基线。R2-D1 不再只问“加一点 Huber 会不会好”，而是问一个有明确预算约束的问题：

> 在总预算固定为 `20F/20A^T` 时，把 H1 的最后 1、2 或 4 个迭代换成数据残差容差单调、support-internal Huber 不增的边缘修正，能否同时胜过 H1-20 和相同分配下的 `rho=0` 纯数据修正？

三种预算分配为：

| 分配 | H1 前缀 | 修正步 | 倾斜比 `rho` |
|---|---:|---:|---|
| `h19_c1` | 19 | 1 | 0, 1, 2, 4 |
| `h18_c2` | 18 | 2 | 0, 1, 2, 4 |
| `h16_c4` | 16 | 4 | 0, 1, 2, 4 |

每个 case 另有一个 H1-20，共 `6 x (1 + 3 x 4) = 78` 条独立轨迹。每条候选都重新运行自己的 H1 前缀，没有共享前缀后伪造调用账本。

## 2. 为什么 R2-D0 v1 不能算结果

首次 R2-D0 预跑生成 42 行后，独立审计发现四个协议缺陷：

1. R2-D0 使用 `denominator_floor=1e-20`，冻结 H1 合同是 `1e-16`；因此当时的 `h1_20` 不能称为精确冻结基线。
2. 非零 `rho` 只与 H1-20 比，没有被要求胜过 `rho=0` 纯数据控制，无法把收益归因给 edge 项。
3. 伴随恒等式缺陷只记录、不拒绝。
4. 缺少 front-F1 与 held-out clean 门。

该目录已经放置 `INVALID_PROTOCOL_NOTICE.json`，状态是 `INVALID_PROTOCOL_DRIFT_NO_SCIENTIFIC_DECISION`。其数值只能用于调试和修订溯源，既不是成功，也不是有效 NO-GO。R2-D1 是另行冻结、明确披露已观察 42 行后的 post-open 修订。

## 3. 修正方向怎样工作

从 H1 前缀得到场 `x` 与递推加权残差 `r` 后，每步计算：

1. 数据法向 `g = A_w^T r`，消耗一次 `A^T`；
2. support-internal Huber 梯度 `e = D^T psi(Dx)`，不消耗物理投影；
3. 将 `e` 对 `g` 正交化为 `e_perp`；
4. 构造 `d = g - rho ||g|| e_perp / ||e_perp||`；
5. 计算 `p = A_w d`，消耗一次 `A`；
6. 使用非负精确线搜索并施加相对场范数 trust clamp；
7. 若实际 Huber 势能上升则只在本地回溯，若伴随缺陷超门则立即中止。

正交化使 `<g,d> = ||g||^2` 在浮点容差内成立。若 `r` 确实是当前线性算子的递推残差，且最终步长 `0 <= alpha <= <r,p>/||p||^2`，则递推数据残差平方不增。这里必须严谨地称为“条件于 residual 合同的容差单调”；终点再用所有方法统一的额外 evaluator `1F` 核验 `r = y_w - A_w x`。

## 4. 主要结果

选择规则在所有非零倾斜候选中选出 `h19_c1_edge_400`，但它没有通过联合门：

| 指标 | 选择行 | 预写门 | 判定 |
|---|---:|---:|---|
| mean field gain vs H1-20 | -0.4335% | >= 0 | 失败 |
| mean gradient gain vs H1-20 | -0.1233% | >= 0 | 失败 |
| worst field gain vs H1-20 | -0.5475% | >= -1% | 通过 |
| worst gradient gain vs H1-20 | -0.2928% | >= -1% | 通过 |
| mean field gain vs matched `rho=0` | +0.0676% | >= 0 | 通过 |
| mean gradient gain vs matched `rho=0` | +0.3959% | >= 0 | 通过 |
| weighted residual ratio vs H1-20 | 1.0378 | <= 1.02 | 失败 |
| held-out clean ratio vs H1-20 | 1.0080 | <= 1.02 | 通过 |
| mean front-F1 difference vs H1-20 | +0.000102 | >= 0 | 通过 |
| edge direction use fraction | 1.0 | >= 0.25 | 通过 |
| max adjoint identity defect | 1.10e-7 | <= 5e-5 | 通过 |

三种分配都呈现一致结构：

| 分配与 `rho=4` | field vs H1 | gradient vs H1 | field vs `rho=0` | gradient vs `rho=0` | residual ratio vs H1 |
|---|---:|---:|---:|---:|---:|
| `19+1` | -0.4335% | -0.1233% | +0.0676% | +0.3959% | 1.0378 |
| `18+2` | -0.8137% | -0.1589% | +0.1767% | +0.8178% | 1.0880 |
| `16+4` | -1.4879% | -0.1917% | +0.3941% | +1.4886% | 1.2250 |

独立审稿从逐 case CSV 与 histories 另行复算出：9 个非零 edge 组合在 field 和 gradient 上都为 6/6 优于各自 `rho=0`；gradient 的 edge-specific gain 在每种预算、每个 case 上都随 `rho=0,1,2,4` 单调改善。edge 步使 Huber 势能下降约 `0.82%--6.01%`，纯数据修正则增加约 `0.28%--0.90%`。这加强了“edge 项真实参与”的机制解释，但没有改变对 H1-20 的 NO-GO。

### 独立复算与 provenance 限制

独立 validator 不调用 runner 的聚合或选择函数，而是从 78 行逐 case 指标、78 份 histories 和冻结配置重新构造 12 个 aggregate、matched `rho=0` 归因、选择行、`20F/20A^T` 账本、5/6 held-out 覆盖、front-F1、伴随缺陷与 168 个修正步。共 561 项检查通过，复算状态仍是 `POSTOPEN_R2D1_NO_GO_NO_FRESH_VALIDATION`。

该 PASS 必须带限定语：runner 当时的 `BOUND_SOURCES` 漏列了一个提供 `_gain`、`_relative_l2`、`_synchronize` 的 helper。validator 只能在运行后证明报告所记提交中的 blob 与当前工作树哈希相同，不能证明执行瞬间不存在短暂未提交改动。因此准确状态是 `PASS_WITH_POST_RUN_DEPENDENCY_CLOSURE_R2D1_POST_OPEN_NO_GO`，不是无保留的运行时 manifest 完整性证明。完整机器可读结果见 `independent_validation.json`；下一次 runner 必须在运行前直接绑定全部 import closure。

## 5. 该保留什么，关闭什么

### 可以保留的机制线索

- 在每一种预算分配中，提高 `rho` 都使候选相对自己的纯数据控制更好，尤其是 gradient 指标。
- support-internal edge 方向真实启用，伴随缺陷远低于门，逐步递推残差保持容差不增。
- 这说明 Huber 方向不是纯数值噪声；它能在给定修正外壳中改变场/梯度权衡。

### 必须关闭的主张

- 用 edge correction **替换** H1 的最后迭代不成立。少替换时差距较小，多替换时 edge-specific 增益变大，但损失的 H1 进展更大。
- 不能因为相对 `rho=0` 为正就宣布新算法。真正的对手是 H1-20，所有候选对它的 mean field 都为负。
- 不继续在这 6 个已见 case 上扫描 `rho=8/16`、trust radius 或 Huber delta；残差比已随 `rho` 恶化，继续调参会变成结果驱动搜索。
- fresh seeds、真实实验数据和论文结果均不授权。

### 还必须直说的证据限制

- H1-20 的绝对 mean field relative-L2 约为 `0.881`，gradient relative-L2 约为 `1.130`；代表切片接近低幅模糊场。当前只是弱重建状态下的机制排序，不是可用三维重建。
- truth 与 observation 使用同一个 QMC8 forward；没有曲光线、几何失配、位移估计误差或真实相关噪声。这是 inverse-crime 型机制实验。
- 只有 6 个重复使用 case，却比较了 9 个非零候选；没有置信区间门、最小实际效应门或多重比较校正。edge-specific 正信号不能当统计确认。
- 9-view case 没有 held-out camera，因此 held-out mean 实际覆盖 `n=5`，不是 6。
- 每条候选在 float32/MPS 上独立重跑 H1 前缀，历史相对漂移约 `1.2e-7`；它很小，但最弱的微小增益仍需共享 warm state 或 CPU float64 重复才能进一步解释。
- `20F/20A^T` 只保证物理 API 预算公平；Huber 局部计算、H1 重正交与 wall time 尚未随机顺序重复并给出区间。
- 独立 validator 的 561 项复算确认算术、矩形、门和 NO-GO，但漏绑 helper 只能事后闭合；不能把它写成 contemporaneous manifest 完整。

## 6. 下一条更有信息量的候选：R2-E0 缓存 Krylov 子空间重优化

R2-D1 的失败说明不应再牺牲 H1 已证明有效的最后迭代。更合理的下一步是保留完整 H1-20，同时复用已经付费得到的 Krylov 搜索方向与其投影：

1. 运行精确 H1-20，并缓存方向基 `P=[p_1,...,p_20]` 与 `A_w P`；
2. 在 20 维系数空间内做确定性重优化，不再调用物理 `A/A^T`；
3. 用缓存的 `A_w P` 精确计算数据残差，以 H1/Huber/front proxy 作为次级准则；
4. 强制 residual 不超过 H1-20 的预写 envelope，失败就精确返回原 H1 系数；
5. 先比较固定凸目标、标量阻尼和低维 Pareto 解，再考虑让小模型从可部署特征预测有界系数或 accept/fallback。

这条路线的潜在价值不是“网络直接猜三维场”，而是把学习问题压到已经由物理算子生成的低维可审计子空间。它仍需先通过三个门：缓存版本与冻结 H1 场逐位一致；不增加物理调用；固定非学习重优化先产生可重复 headroom。

独立审稿还提出一个并列但次序稍后的候选：在 H1 每步方向形成后、消耗原有 forward 前加入有界 edge 分量，使 `rho=0` 与 H1-20 逐位一致。这可以理解为“同一步内混合”，而不是 R2-D1 的“先停止 H1 再替换”。为了减少同时改动，建议先完成 R2-E0 的方向/投影缓存与逐位等价门；缓存合同一旦成立，再比较低维全局重优化和同一步内固定混合，二者都不得增加物理调用。

## 7. 仍需何远哲师兄确认

- 实验室真实 callable 是线性 straight-ray、迭代 curved-ray，还是二者分层？
- 能否输出或缓存每步方向、投影、JVP/VJP 与 residual 层级？
- 真实几何标定误差、折射率边界、遮挡和噪声中的第一主痛点是什么？
- 目前认可的 H1/TV/NeRIF/TDBOST 基线与固定算子调用预算怎样定义？
- 哪些数据可划分为开发、未见工况、未见 rig 与最终 lockbox？

在这些合同到位前，R2-E0 只能在公开 PSU 几何合成场上做算法证伪，不能替实验室生成真实性结论。
