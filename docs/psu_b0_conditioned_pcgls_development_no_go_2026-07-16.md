# BOST-GC-SPD-PCGLS V1 开发判决：0/3 NO-GO

日期：2026-07-16
证据等级：post-open development only
状态：`CONDITIONED_PCGLS_DEVELOPMENT_COMPLETE_FRESH_NOT_USED`

## 一句话结论

首个几何/噪声条件化、低维、固定正定频谱预条件器没有稳定超过
Sobolev-PCGLS-4。三种训练种子全部未达到预设的 2% 开发门槛，因此
`BOST-GC-SPD-PCGLS V1` 判为 `0/3 NO-GO`，不得生成新的 independent fresh
repeat，也不得通过扩大网络宽度继续追分。

这不是“训练报错”。优化损失持续下降，投影残差也平均改善，但三维场误差
几乎不动，部分 calibration 样本反而恶化。当前最重要的机制证据是：

> 更贴合二维投影，不等于恢复出更准确的三维折射率场。

## 1. 比较合同

| 项目 | 固定合同 |
|---|---|
| 物理层 | 真实 PSU support 相机几何，解析反应流形态，合成相机噪声 |
| 训练 / 选择 / 单次确认 | `risk_train` / `risk_validation` / `risk_calibration` |
| fresh | 全程不读取 |
| 基线 | validation-selected Sobolev-PCGLS-4，strength=4，epsilon=0.05 |
| 候选 | 2,527 参数 MLP，输出 7 个频谱 basis 系数 |
| 正定约束 | 正 multiplier、有界 log correction、几何均值归一化 |
| 求解结构 | multiplier 在一次 solve 前生成，四步中固定 |
| 调用预算 | 候选和基线均为 `4F + 4AT` |
| 随机种子 | 20262841、20262842、20262843 |
| 预设主要门槛 | validation/calibration mean field gain 均不低于 2%，bootstrap 下界大于 0，`>1%` harm rate 不高于 5%，至少 2/3 种子通过 |

零初始化已通过逐值测试：候选在所有输出系数为零时与静态
Sobolev-PCGLS-4 完全一致。固定 SPD、调用账本、数据拆分和 fresh firewall
也全部通过。

## 2. 三种子结果

正数表示候选相对静态 PCGLS-4 降低了三维场 relative-L2，负数表示恶化。

| seed | split | mean field gain | bootstrap 95% interval | p10 | minimum | `>1%` harm |
|---:|---|---:|---:|---:|---:|---:|
| 20262841 | validation | +0.0545% | [-0.0246%, +0.1487%] | -0.1196% | -0.2664% | 0/24 |
| 20262841 | calibration | +0.0559% | [-0.0154%, +0.1317%] | -0.1558% | -0.3595% | 0/30 |
| 20262842 | validation | +0.0264% | [-0.2472%, +0.3050%] | -0.6595% | -1.3742% | 1/24 |
| 20262842 | calibration | -0.1649% | [-0.5174%, +0.1194%] | -0.8497% | -3.9052% | 1/30 |
| 20262843 | validation | +0.0162% | [-0.1587%, +0.1933%] | -0.4152% | -0.9934% | 0/24 |
| 20262843 | calibration | -0.1048% | [-0.3639%, +0.0907%] | -0.5001% | -2.9087% | 1/30 |

判决为 `0/3`。所有均值都远低于 2% 门槛，所有 bootstrap 下界都小于 0；
seed 42 和 43 在 calibration 上的平均场误差还出现净恶化。

## 3. 最有价值的失败机理

| seed | split | measurement residual gain | field gain |
|---:|---|---:|---:|
| 20262841 | validation | +0.8362% | +0.0545% |
| 20262841 | calibration | +0.5963% | +0.0559% |
| 20262842 | validation | +1.7091% | +0.0264% |
| 20262842 | calibration | +1.2236% | -0.1649% |
| 20262843 | validation | +1.0869% | +0.0162% |
| 20262843 | calibration | +0.6989% | -0.1048% |

模型确实改变了预条件器，validation loss 也在 30 个 epoch 内下降；但它主要
学会了更快压低观测空间 residual，没有找到能稳定改善三维不可辨识方向的
频谱结构。front top-10% F1 和 gradient-L2 的平均变化同样很小且有正有负。

因此当前数据支持以下解释：

1. 四步静态 PCGLS 已经位于这个低维 multiplier 家族的局部前沿附近；
2. 当前相机集合基本固定，几何特征的有效变化量有限；
3. 只用初始 residual/noise 汇总预测 7 个全局频谱系数，可能无法识别每个场
   的 null-space ambiguity；
4. 训练目标包含真实三维场监督，但可观测输入仍不足以把投影拟合增益转换成
   稳定体场增益。

这些是与当前证据相容的解释，不是已经证明的唯一原因。

## 4. 这次结果否掉什么

- 否掉当前实现的“低维 observation-conditioned fixed SPD multiplier V1”；
- 否掉“只要在 PCGLS 上加一个小 MLP 就能自然超过强经典基线”的乐观假设；
- 否掉在现有 opened development 上继续增宽 MLP、扫学习率或降低 baseline
  的做法；
- 不授权 fresh repeat、实验有效性或优于 DeepONet/FNO/UNO-CG 的主张。

## 5. 这次结果没有否掉什么

- 不同静态 PCGLS 参数在不同样本上的 truth-oracle headroom；
- 真实可变相机布局、有限孔径标定和 flow-off covariance 提供的条件信号；
- TV-superiorized PCGLS 对 thin front / shock 的优势；
- 只学习停止步数、置信度或回退策略；
- residual-adaptive preconditioner，但它必须进入 Flexible CG 合同；
- 使用真实 OERF 测量和独立物理真值后的迁移价值。

## 6. 下一步决策，不先扩网络

下一实验先做有限候选的 conditional-headroom audit：

1. 在 `risk_train` 上选择一个全局静态候选；
2. 按 active-view count 选择候选，再固定映射到 validation/calibration；
3. 按 view count + 声明噪声层级选择候选，明确它只是 synthetic metadata
   诊断；
4. 计算 morphology-family oracle 和 per-sample truth oracle，只作不可部署
   的上限；
5. 全部与静态 PCGLS-4 同 `4F+4AT` 比较。

若连 per-sample truth oracle 的 headroom 都很小，说明当前有限频谱家族已经
耗尽，应转向 TV、学习停止或更强先验；若 oracle 很大而可部署分层很小，说明
问题在可观测特征与条件化映射，而不是网络容量。

## 7. 运行与产物

- 三种子训练总墙钟时间：60.32 秒；
- 峰值常驻内存：约 494 MiB；
- Apple M5 / 32 GB / MPS 足够当前 32³ 开发，不需要租 GPU；
- 公开摘要：
  [psu_b0_conditioned_pcgls_development_public_summary.json](psu_b0_conditioned_pcgls_development_public_summary.json)
- 冻结配置：
  [psu_b0_conditioned_pcgls_development_v1.json](../demo_t16_operator/configs/psu_b0_conditioned_pcgls_development_v1.json)
- 模型源码：
  [psu_b0_conditioned_pcgls.py](../demo_t16_operator/psu_b0_conditioned_pcgls.py)
- 论文图：
  [PNG](../demo_t16_operator/results/psu_b0_conditioned_pcgls_development/psu_b0_conditioned_pcgls_development_figure.png) ·
  [PDF](../demo_t16_operator/results/psu_b0_conditioned_pcgls_development/psu_b0_conditioned_pcgls_development_figure.pdf)

## 8. 严格声明边界

本实验没有读取 PSU 真实测量值，没有实验三维真值，没有 CFD truth，没有
新的 independent repeat，也没有打开 fresh。解析 morphology 不是 CFD，
development 失败不能证明所有 learned preconditioner 都无效；它只足以停止
当前 V1，并决定下一次最有信息量的实验。
