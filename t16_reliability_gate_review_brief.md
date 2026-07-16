# T16 可靠性门控与双分支原型：师兄审核简报

更新：2026-07-10

## 一句话结论

第一版“给 physics lift 乘一个可靠性系数”没有解决少视角翻转。随后的共享主干双分支原型证明两个输出存在可用互补，但 query-trained router 几乎塌缩为 0.5 等权。目前最强的无真值基线不是更复杂的 MLP，而是用 support 相机重投影做一维闭式最小二乘融合：其五域 field oracle regret 为 **0.014%**，明显低于等权融合的 0.197% 和神经路由的 0.233%。下一版应把这个可解释物理解作为底座，只学习其无法观测的零空间/几何残差。

## 实验合同

- 数据：固定 168 个合成三维体，`8 x 16 x 16`。
- 测试域：IID、view OOD、noise OOD、joint OOD、family OOD。
- 优化种子：`20260710 / 20260711 / 20260712`。
- 每个模型最多 30 epochs，CPU；同一数据、损失预算与 FNO 主干。
- gate screen 的可靠性输入不使用 field truth、clean observation 或 held-out-view 标签；dual router 在训练时可读取从已观测相机中拆出的 noisy query views，推理时不读取。
- oracle 只用于合成数据回顾性审计，真实实验中不可用于选模型。
- gate 的两个核心统计 CSV 与 dual prototype 的三个核心统计 CSV 均已独立整轮复跑；共 6 个 smoke tests 通过。

## 六个对照

| 方法 | 参数 | IID field | IID held-out | 五域 field oracle regret | 五域 held-out oracle regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual FNO | 43,363 | 0.2187 | 0.2507 | 3.10% | **1.65%** |
| Absolute-output FNO | 43,363 | 0.2367 | 0.2677 | 3.83% | 3.75% |
| Fixed view gate | 43,363 | 0.2187 | 0.2507 | 6.64% | 8.30% |
| Learned metadata gate | 43,396 | 0.2195 | 0.2530 | **3.08%** | 2.00% |
| Residual + observed residual channel | 43,387 | 0.2502 | 0.2913 | 13.59% | 9.69% |
| Learned observable gate | 43,428 | 0.2448 | 0.2843 | 9.02% | 8.95% |

`oracle regret` 的 oracle 是每个 seed-domain 单元中 Residual 与 Absolute 的较小误差。它回答“一个自适应模型距离事后知道最佳分支还有多远”，不是可部署指标。

## 双分支原型的新结果

原型在 64 个基础训练样本上固定生成 128 个 support/query 变体，每个优化种子训练一个 43,485 参数的双输出 FNO。三视角已经出现在 view-dropout 训练中，因此新实验不再把 `test_view_ood` 宣称为“视角数量 OOD”，它只保留旧分割名称以便对照。

| 推理路径 | IID field | view field | joint field | family field | 五域 field oracle regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual head | 0.2629 | 0.3451 | 0.3808 | 0.6955 | 不适用 |
| Absolute head | 0.2612 | 0.3395 | 0.3838 | 0.6892 | 不适用 |
| 0.5 等权融合 | 0.2577 | 0.3382 | 0.3767 | 0.6908 | 0.197% |
| **support-fit 闭式融合** | **0.2571** | **0.3380** | **0.3766** | **0.6888** | **0.014%** |
| query-trained router | 0.2578 | 0.3382 | 0.3768 | 0.6909 | 0.233% |

support-fit 权重是部署时可计算的：

```text
w_S = clip(<A_S(x_res-x_abs), y_S-A_S(x_abs)> / ||A_S(x_res-x_abs)||^2, 0, 1)
x_hat = w_S x_res + (1-w_S) x_abs
```

它不读取 field truth 或 held-out camera。在 264 个样本-种子单元中，support-fit 融合有 75.8% 达到或优于两个端点专家的较小 field error；support 重投影优势与 field head 优势的 Spearman 为 0.695。相比之下，query router 的五域平均权重均在 0.499--0.503，field head 选择准确率只有 40.5%。

这不能证明 support-fit 已是新算法；它证明了一个更值得研究的分解：**先用 forward physics 解决可观测的一维融合，再用学习模型只修正 support views 无法识别的分量。**

## 第二版模型应如何改

1. 把目前几乎全共享的两输出改成独立或低共享专家，并做参数量匹配对照；否则 router 没有足够的可路由差异。
2. 以 `w_S` 作为可解释先验，让网络只预测受限幅的 `delta_w`，而不是从零预测整个权重。
3. 把 `||A_S(x_res-x_abs)||^2` 当作路由可识别度；当两专家在观测空间几乎不可区分时，输出不确定性/拒答，不强行学习伪路由。
4. query cameras 不再只生成软专家标签，而是学习 `delta_w` 或零空间体场残差；先报 query/field alignment，再决定是否保留这个 loss。
5. 最终接入 OERF 的真实 ray geometry 和 NeRIF refinement；在此之前不声称对实验 BOST 有效。

## 关键反例

1. 固定门控在三视角把 lift 权重从 1 降到 0.6，却把 view OOD field 从 0.488 提高到 0.518，joint OOD 从 0.492 提高到 0.540。
2. 可学习 metadata gate 在 joint OOD 改善到 0.470，但 view OOD 仍为 0.490；它主要响应噪声，不会自动学会“少视角时应信哪一条分支”。
3. 可观测 lift 重投影残差很有诊断力：IID 约 0.398，view OOD 约 0.846，joint OOD 约 0.850；但把这个标量平铺成整块 3D 通道会造成种子波动和整体退化。
4. family OOD 在所有方案中仍约 0.665--0.693。架构微调不能替代流态覆盖、真实 forward geometry 与物理验收。

## 为什么双分支仍值得继续

单标量乘在 lift 上，要求同一个 correction backbone 同时适应两种不同输出参数化；首轮 gate 的训练域又没有三视角样本，只能外推。双分支原型已用 view dropout 修复这个训练合同，但仍需要让专家分工与路由目标更可识别。更合理的下一版模型是：

1. 一个共享的 geometry-aware encoder / operator trunk；
2. `x_res = x_lift + h_res(z)`；
3. `x_abs = h_abs(z)`；
4. 先用 support reprojection 计算闭式 `w_S`，它是新的必须击败基线；
5. router 只读取可观测的 view coverage、角度缺口、mask、噪声估计、support reprojection、branch disagreement，并以 `w_S` 为锚点；
6. 训练时把观测视角随机拆成 support/query，以 query-camera projection consistency 学习受限修正或零空间残差；
7. 测试时没有 query camera 时，回退到 `w_S` 与不确定性输出；有额外相机时才直接做自监督修正。

## 请师兄判断

1. 组内真实任务能否在训练时人为留出 1--2 个相机作为 query view？
2. 相机几何是否固定；若变化，能否提供每视角投影矩阵或 ray table？
3. 三视角是组内真实需求，还是本项目人为设置的 OOD 压力测试？
4. 输入应该是位移场、投影梯度、raw BOS pair，还是组内已有的初始重构？
5. 是否认可把 operator output 用作 NeRIF 初始化，并比较收敛速度、失败率和最终重投影？

## 文件

- 入口：`operator_3d_innovation_lab.html`
- 完整提案：`view_reliability_dual_branch_operator_proposal.md`
- 配置：`demo_t16_operator/configs/reliability_gates.json`
- 运行器：`demo_t16_operator/run_reliability_gates.py`
- 报告：`demo_t16_operator/results/reliability_gates/gate_report.json`
- 摘要：`demo_t16_operator/results/reliability_gates/gate_summary.csv`
- oracle regret：`demo_t16_operator/results/reliability_gates/gate_oracle_regret.csv`
- dual 配置：`demo_t16_operator/configs/dual_branch_query.json`
- dual 运行器：`demo_t16_operator/run_dual_branch_query.py`
- dual 报告：`demo_t16_operator/results/dual_branch_query/dual_report.json`
- dual 特征对齐：`demo_t16_operator/results/dual_branch_query/dual_feature_alignment.csv`
