# JACRU-M2 T0 判决：场误差显著下降，但物理数据一致性失守

**实验日期：** 2026-07-17  
**状态：** `M2_T0_NO_GO_OR_REVISE`  
**证据等级：** `E1_INDEPENDENT_RENDERER_SYNTHETIC_TRAIN_DEVELOPMENT_OOD_SCREEN`  
**Fresh / final 授权：** `false`  
**允许的下一步：** 只允许在已经打开的 T0 数据上诊断数据一致性校正；不允许声称算法胜出。

## 一句话结论

四种监督残差模型都能把合成三维场误差压低，但全部严重破坏观测一致性。JACRU-M2 的
development / exploratory-OOD field-L2 相对 CGLS 分别改善 `46.16% / 32.38%`，可是逐样本
重投影误差比 CGLS 放大 `28.56x / 35.10x`。更简单、参数更少的 pooled 3D CNN 在场误差上
还略好于 JACRU。因此本轮证明的是“监督形态先验很强，但会幻觉”，不是 JACRU 优越。

## 1. 这次真正测试了什么

每个样本先运行固定 12 步 CGLS 得到底座 `x0`，再计算逐相机残差
`r_v = y_v - A_v x0` 与 lift `A_v^T r_v`。网络只能读取 `x0`、逐相机 lift、pose、mask 和
support，不能读取 truth、family label 或 interface mask。训练 truth 只进入 loss 与评分。

本轮包含四个同量级小模型：

| 方法 | 结构 | 参数量 | 3-seed 总训练时间 |
|---|---|---:|---:|
| `jacru_m2` | shared per-view encoder + permutation-invariant pooling | 6,440 | 23.77 s |
| `pooled_cnn` | 先池化相机 lift，再用 3D CNN | 3,549 | 9.71 s |
| `grid_deeponet` | fixed-grid branch/trunk low-rank operator | 8,162 | 11.94 s |
| `pooled_fno` | pooled lift + 官方 `neuraloperator` FNO | 10,211 | 16.07 s |

所有模型使用 3 个模型 seed。train 为 32 cases，development 为 12 cases，已经打开的探索性
geometry+morphology OOD 为 18 cases。整轮 Mac MPS 用时 `68.63 s`。这只是小样本 T0
筛选，不是 100 万参数、24F/24A 的未来预注册主实验。

## 2. 严格结果

下表的 gain 都相对当前同一考卷中 field 最好的经典基线 CGLS-13；reprojection ratio 是
每个 case 相对 CGLS 的比值再取均值，不是两个总体均值相除。

| 方法 | Dev field gain | Dev H1 gain | Dev reproj / CGLS | OOD field gain | OOD H1 gain | OOD reproj / CGLS | 最坏 OOD field gain | 判决 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| JACRU-M2 | **46.16%** | **50.24%** | **28.56x** | **32.38%** | **42.68%** | **35.10x** | +3.47% | NO-GO |
| pooled CNN | **47.11%** | **50.32%** | **27.91x** | **32.80%** | **41.85%** | **34.47x** | +11.07% | NO-GO |
| grid DeepONet | 6.57% | 2.19% | 9.37x | 3.74% | -1.59% | 13.63x | -6.04% | NO-GO |
| pooled FNO | 39.23% | 40.03% | 40.52x | 19.84% | 25.80% | 53.16x | -34.85% | NO-GO |

经典参照的均值是：

| Split | CGLS-13 field | CGLS-13 H1 | CGLS-13 reproj | Huber-13 field | Huber-13 H1 | Huber-13 reproj |
|---|---:|---:|---:|---:|---:|---:|
| development | 0.664678 | 1.183715 | **0.010820** | 0.666951 | **0.950721** | 0.287909 |
| exploratory OOD | 0.659926 | 1.209269 | **0.008941** | 0.668878 | **0.955004** | 0.280563 |

JACRU 与 CNN 的 field/H1 收益在 3 个模型 seed 上都为正，且没有超过 1% 的 field harm case；
这说明监督先验不是随机噪声。但它们用不符合观测的数据换取了更像训练分布真值的体场，不能
称为反演成功。DeepONet 较保守，却没有获得足够场增益；FNO 在探索性 OOD 上出现
`12.96%` harm rate 和 `-34.85%` 最坏 case。

## 3. 为什么这是一个有价值的失败

### 3.1 “三维看起来更像”不等于“解释了 BOS 观测”

监督网络学会了 smooth / interface 合成 family 的形态规律。它可以把欠定零空间中的解推向
训练分布常见形状，因此 truth-space 指标显著改善；但预测重新经过同一个物理 forward 后，
已经无法解释输入观测。论文若只画 truth slice 或 NRMSE，就会把这种形态幻觉误写成重建能力。

### 3.2 当前 JACRU 的特殊结构尚未被证明必要

pooled CNN 参数更少、训练更快，field 指标还略好。当前证据不支持“逐相机共享编码 + 集合池化
天然更强”。它未来是否有价值，只能在真正变化的 camera count、pose、mask 和实验 geometry
上通过针对性消融证明，不能从这轮同规模考卷推断。

### 3.3 不同模型暴露了不同失败模式

- CNN / JACRU：强形态先验，场指标好，但数据一致性严重失守。
- DeepONet：修正较小，数据一致性相对没那么坏，但场与 H1 收益不足且有尾部伤害。
- FNO：能拟合平均形态，但 OOD 尾部和重投影最危险。

## 4. 下一轮 M2.1：让网络提议，让物理算子裁决

下一步不扩大网络，而是在网络修正后加入确定性的 measured-data correction：

```text
x_net = x0 + delta_theta(observable residual lifts, pose, mask)
x_(k+1) = support * [x_k + tau * A^T(y - A x_k)]
```

先在已经打开的 T0 上做 `0 / 1 / 3 / 5 / 11` 步 Landweber/CGLS-style correction sweep，
同时画 field-H1-reprojection Pareto 前沿。这个诊断只回答：现有网络收益中有多少能在回到测量
流形后保留。它不能打开 fresh，也不能选择最终超参数。

如果 3--5 步校正能把 reprojection 压回 `<=1.10x` CGLS，同时保留至少 5% field gain，才值得
实现训练时 projection loss / unrolled data-consistency block；否则说明当前 correction 主要位于
错误零空间，应该停止这条监督残差路线，而不是继续加模型容量。

## 5. 当前禁止主张

1. 禁止说 JACRU、CNN、DeepONet 或 FNO 在 BOST 三维重建上优于经典方法。
2. 禁止把探索性 OOD 写成 sealed OOD、fresh、final、真实实验或 CFD 泛化。
3. 禁止声称 JACRU 发现或恢复了 shock/interface；本轮没有 front 指标。
4. 禁止声称参数公平已经完备；四个模型只是同量级，并非严格参数匹配。
5. 禁止用 field gain 隐去 reprojection 失败。

## 6. 可复现入口

- [冻结 T0 配置](../demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json)
- [机器摘要](../demo_t16_operator/results/jacru_m2_learned_residual_t0_public/summary.json)
- [逐 case 指标](../demo_t16_operator/results/jacru_m2_learned_residual_t0_public/metric_rows.csv)
- [聚合指标](../demo_t16_operator/results/jacru_m2_learned_residual_t0_public/aggregate_rows.csv)
- [训练历史](../demo_t16_operator/results/jacru_m2_learned_residual_t0_public/training_history.csv)
- [诊断图 PDF](../demo_t16_operator/results/jacru_m2_learned_residual_t0_public/diagnostic.pdf)
- [实验 runner](../site_tools/run_jacru_m2_learned_residual_gate.py)
- [M2 文献与原创性地图](jacru_m2_literature_landscape_2026-07-17.md)
- [未来严格预注册草案](jacru_m2_preregistered_gate_2026-07-17.md)

复现命令：

```bash
PYTHONPATH=. .venv/bin/python site_tools/run_jacru_m2_learned_residual_gate.py \
  --config demo_t16_operator/configs/jacru_m2_learned_residual_t0_v1.json \
  --output-dir demo_t16_operator/results/jacru_m2_learned_residual_t0_public
```

这轮最重要的研究资产不是一个“赢了”的标题，而是一条已经被实验证明必须解决的硬约束：
**任何 learned residual 必须同时改善 truth-space 场并留在 measurement-consistent 解集附近。**
