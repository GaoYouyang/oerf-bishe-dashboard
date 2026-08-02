# v102：坐标输运保护兼容性，但 observation-only rank 4 没有击败静态 rank 0

## 一句话结论

师兄提出的“加入微分同胚原理，增强坐标系变化后的泛化”已经进入正式数值实验，而不再只是概念建议。v102 把 fold-local 公共三维物理参考场通过每套已知九视角几何的精确 forward 投影到各自观测坐标，构造 geometry-conditioned transport features，再预测 rank-4 系数并进入相同的 strict CGLS K1。

最终没有任何 eligible 方法通过“对静态 rank 0 的稳定优势门”：

```text
FAIL_NO_REFERENCE_RANK4_OBSERVATION_PREDICTOR_ADVANTAGE_V102
PASS_INDEPENDENT_RECOMPUTATION_REFERENCE_RANK4_OBSERVATION_PREDICTOR_V102
algorithm_breakthrough = false
```

但这不是空结果。正确运输的 projected-ridge、linear、RBF 都保持 `90/90`；完全不做运输的 linear control 只有 `71/90`。这说明坐标一致性确实保护了跨几何重建兼容性，只是当前 rank-4 可观测预测没有稳定优于更简单的静态物理先验。

## “微分同胚”在这里具体做了什么

若参考域到物理域的可逆光滑映射为 `phi`，密度作为标量需要 pullback，梯度需要按 Jacobian 的逆转置变换：

```text
rho_physical(phi(x)) = rho_reference(x)
grad_physical rho = J_phi^{-T} grad_reference rho
```

BOST 还多一层：相机 ray、探测器 `u/v` 基、forward 与 adjoint 也必须同步变换。v99 已用三类可逆坐标变换验证这一整条链；错误地只 warp 三维数组会产生 `6.88%–132.56%` 的偏差。

v102 没有训练形变网络，而是采用更小、更可审计的实现：

1. 每折只用 fit 时刻的相机无关三维物理真值拟合 mean 与 rank-4 基；
2. 用每套已知几何的 `A_g` 投影 mean 和四个模态，得到 `A_g mean` 与 `A_g U`；
3. 从观测 `y`、known geometry、detector-dual anchor、投影 Gram、各视角能量和旧 100 维观测特征构成固定 184 维输入；
4. 比较 projected-ridge、linear residual、RBF residual，以及 geometry-ID、no-transport、wrong-pose controls；
5. 每条预测都进入相同的 `2A+2A^T` strict K1，并检查八个 matched-accuracy 门。

因此它准确的名称是 **known-geometry physical transport diagnostic**，不是 learned diffeomorphism、任意坐标等变网络或未见相机位姿泛化。

## 真正运行了什么

- 已开封 BLASTNet H2-air Case 6；三套九视角几何、30 个物理时刻，共 `90` 个单元；
- `3` 套留出几何 × `5` 个连续六帧时间块，共 `15` 折，并删除相邻一帧；
- `7` 个方法臂、`630` 次严格 K1，没有 breakdown；
- 在线候选总账 `1260A+1260A^T`，即每条恰好 `2A+2A^T`；
- 几何运输 setup 单列 `225A`，fit-only anchor 单列 `672A^T`，没有藏入在线预算；
- held-out 真值标签在预测封存前保持不可见，两组只改变 held-out 标签的哨兵拟合产生完全相同预测。

## 最关键结果

| 方法 | 八门通过 | 是否满足对 rank 0 的优势门 |
|---|---:|---:|
| static physical rank 0 | **90 / 90** | 基线 |
| transported projected ridge | **90 / 90** | 否 |
| transported linear residual | **90 / 90** | 否 |
| transported RBF residual | **90 / 90** | 否 |
| geometry-ID linear control | **90 / 90** | 不可选 control |
| no-transport linear control | **71 / 90** | 不可选 control |
| wrong-pose projected-ridge control | **90 / 90** | 不可选 control |

### 最简单的 projected-ridge 差在哪里

它是 eligible 方法中最有解释力的一条：

- maximum-gate p50 改善 `0.00493`，低于预注册最低值 `0.01`；
- p90-higher 改善 `0.05207`，通过；
- worst 改善 `0.00243`，通过；
- 三套几何中只有 `2/3` 的 p50 不劣，F12+ 略差 `0.000478`；
- field 逐单元胜数 `59/90`，低于最低 `60/90`；
- full-gradient 胜数 `66/90`，通过；
- 八类指标中只有 `4/8` 的 p50 / p90 / worst 全部非劣。

四个主要尾部伤害来自 field-vs-Zero-K4、interior-gradient-vs-Zero-K4、observation-vs-Zero-K2 和 observation-vs-Zero-K4。它们都很小，却是预注册门明确禁止用均值收益掩盖的局部风险。

linear residual 更明显地退化；RBF residual 几乎回到 projected-ridge，没有产生额外可预测价值。结论不是“神经网络永远无用”，而是当前 fixed rank-4 target、184 维输入和已见三几何下，没有证据支持继续放大这一 family。

## control 告诉了我们什么

### 1. 坐标输运不是装饰

no-transport linear 只有 `71/90`，而带正确几何运输的三个方法都是 `90/90`。这说明把公共物理目标投影到当前相机坐标，再读取观测残差与投影 Gram，确实在保护安全性。

### 2. 当前坐标变化还不够强

wrong-pose projected-ridge 仍为 `90/90`。这不代表错姿态正确，而说明当前三套几何与宽松的兼容门无法充分区分 pose identity。因而 v102 不能宣称 unseen-pose 或 arbitrary-diffeomorphism generalization。

下一坐标实验必须引入结果前冻结的连续、可逆、Jacobian 有界形变，并保持 density、gradient、ray、detector basis、forward 和 adjoint 一致输运；错误姿态应成为能被明确击穿的 stress control。

## 独立复算与一次透明的无效尝试

正式阶段先生成 `630` 个预测并封存，再读取 held-out 系数做评分。独立 validator 没有导入正式 v102 core、runner、回归器或 scorer，而是重写 15 折、SVD、184 维特征、guard、linear/RBF、strict K1、八门与 arm selection。

第一次独立复算被 fail-closed 判为 inconclusive：独立代码把同一个 `P^T P` 计算两次再对称化，末位舍入经回归放大后，系数最大差为 `4.30e-12`，略高于冻结的 `2e-12`。这次没有生成 validation 终态，也没有用于解释结果。

随后只修正运算次序为“先计算一次 Gram，再对称化”，没有修改模型、数据、门、正式输出或 `2e-12` 容差。第二次独立复算中 coefficients、initializer、field、residual、metrics、gates、mutation 和最终 decision 的最大差全部为 `0`；正式输入与输出在验证前后逐字节不变。

## 与现有工作的边界

“把不同物理域送到公共域再学习”已有明确先例，不能作为本项目单独的原创点：

- [Diffeomorphism Neural Operator](https://www.nature.com/articles/s42005-024-01911-3) 在 generic domain 上学习不同物理域的 PDE 算子；
- [DIMON](https://www.nature.com/articles/s43588-024-00732-2) 用显式微分同胚统一不同几何上的函数空间，并给出算子近似框架；
- [Geo-FNO](https://jmlr.org/papers/v24/23-0064.html) 学习从不规则物理域到规则 latent grid 的 deformation；
- [CT-FNO](https://openreview.net/forum?id=pMD7A77k3i) 将坐标变换和物理对称性直接放进 FNO；
- [GINO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) 用图算子把不规则几何映射到规则 latent grid。

当前仍可能有价值的窄缺口是：**在 BOST 逆问题中，同时保持密度、梯度、相机射线、探测器基和精确 adjoint 的坐标一致性，用低成本 fail-closed initializer 减少后续物理迭代。** v102 只完成了这一缺口的已知离散几何开发诊断，没有证明完整算法。

## 当前路线判决

```text
physics-consistent coordinate transport = useful for compatibility
strict observation-only rank4 advantage = failed
static physical rank0 = preferred current initializer
current rank4 predictor family = closed
continuous diffeomorphic stress gate = next coordinate question
GPU rental / large FNO = not authorized
external generalization = false
real BOST = false
paper success = false
```

这次不应通过更大 FNO、UNO 或 U-Net 挽救。下一步先构造连续可逆坐标应力门，确认当前简单 rank-0 / transported-ridge 在真正变化的坐标系下谁会失效；只有出现清晰、可预测且外部门可复现的 headroom，才重新授权学习模型。
