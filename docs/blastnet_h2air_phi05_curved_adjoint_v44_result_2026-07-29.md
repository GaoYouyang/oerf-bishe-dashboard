# BLASTNet v44.3：单个 coarse curved-adjoint 富集方向的有界负结果

日期：2026-07-29  
数据角色：BLASTNet 预混 H2-air 槽式燃烧器 DNS，`phi=0.5`，外部门开封后的机理开发  
正式状态：`VALIDATED_POST_OPEN_BOUNDED_NEGATIVE_V44_3`

## 1. 这次真正问了什么

v43 已经说明：在四个冻结的 straight-CGLS 增量里，即使让看见真值的受约束
oracle 调权，也没有一帧同时守住 field、gradient 和 observation 的 `1.01`
兼容门。

v44 不再更换同一个四维搜索器，而是第一次改变候选空间本身：

1. 在 Direct-K3 处计算部署可见的 curved observation residual；
2. 对该 residual 做一次精确 curved VJP；
3. 投影到 `16 x 16 x 32` 粗网格；
4. 从四个旧 straight-CGLS 增量的 span 中正交化；
5. 将得到的新方向加入四方向控制，形成五方向候选；
6. 候选生成不读 field truth，密封后才用 truth 评分。

问题不是“第五个权重能不能非零”，而是：

> 一个确实来自 curved physics、且与旧 span 数值独立的新方向，能否带来可归因的
> 三指标完整门改善？

## 2. 新方向是否真实存在

存在。四个快照都通过了方向有效性检查：

| 快照 | 旧 span 外能量比例 | 与旧 span 最大余弦 | curved adjoint 恒等式相对误差 | 第五权重 |
|---|---:|---:|---:|---:|
| S1 | 0.190369 | 2.40e-15 | 1.53e-15 | 0.144414 |
| S2 | 0.243300 | 1.98e-15 | 5.64e-16 | 0.221684 |
| S3 | 0.263525 | 2.32e-15 | 5.88e-16 | 0.225157 |
| S4 | 0.314165 | 2.95e-15 | 5.60e-16 | 0.380793 |

这排除了“新方向数值塌缩”“其实仍在旧 span 里”或“VJP 实现明显错误”三种解释。
第五个方向携带了 `19.0%-31.4%` 的旧 span 外能量，优化器也确实使用了它。

## 3. 正式结果

五方向候选相对 Direct-K4 的误差比为：

| 快照 | field / K4 | gradient / K4 | observation / K4 | 完整门 |
|---|---:|---:|---:|---|
| S1 | 0.976348 | 1.015315 | 1.014640 | FAIL |
| S2 | 0.976447 | 1.031173 | 1.013825 | FAIL |
| S3 | 0.973110 | 1.026650 | 0.992946 | FAIL |
| S4 | 0.976449 | 1.014803 | 1.042355 | FAIL |

跨帧中位数为：

```text
field / K4        0.976397
gradient / K4     1.020983
observation / K4  1.014232
```

field 在四帧都改善，但 gradient 在四帧都高于冻结的 `1.01` 线；observation
只有 S3 通过。因此完整门是 `0 / 4`。

## 4. 第五方向有没有贡献

有数值贡献，但不足以形成完整门优势。相对配对的四方向控制，五方向在四帧都降低
field 和 observation 误差；gradient 的变化则有正有负。以误差比百分点表示：

| 快照 | field 改善 | gradient 改善 | observation 改善 |
|---|---:|---:|---:|
| S1 | +0.0456 pp | -0.0391 pp | +0.3270 pp |
| S2 | +0.0113 pp | +0.0643 pp | +0.5407 pp |
| S3 | +0.0317 pp | +0.0333 pp | +0.8685 pp |
| S4 | +0.0561 pp | -0.0891 pp | +1.3589 pp |

所以不能说新方向“没用”。更准确的判决是：

```text
NO_ATTRIBUTABLE_COMPLETE_GATE_ADVANTAGE_FOR_SINGLE_COARSE_ADJOINT_ENRICHMENT
```

它扩展了可达空间，也改善了两项指标，但没有解决 gradient 安全性，因而没有
可归因的 complete-gate headroom。

## 5. 独立复算

独立 validator 不导入正式 runner 或正式 validator，重新生成 Stage A、Stage B
和候选场，得到：

```text
PASS_INDEPENDENT_RECOMPUTATION_PAIRED_COARSE_ADJOINT_ENRICHMENT_V44_3

Stage-A 最大数值差            4.49e-11
Stage-B 最大数值差            2.22e-15
候选场最大绝对差              9.02e-17
正式报告最大数值差            0
```

正式和独立两条路径都确认 `0 / 4` 与相同科学判决。候选在 truth 评分前已经密封；
但这仍不是对整套物理实现完全独立的第三方复现。

## 6. 成本与适用边界

本轮正式机理诊断总账为：

```text
96 F + 292 JVP + 4 VJP
```

其中包含新方向准备、五方向臂、配对四方向控制、审计和 Stage-B 评分。这一成本远
高于可部署 warm start，不能写成 same-cost、wall-time 或内存优势。

本轮不支持以下主张：

```text
deployable_algorithm=false
matched_accuracy=false
speedup=false
broad_generalization=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

BLASTNet `phi=0.5` 在 v40 开封后已经是开发数据；本轮不增加新的外部泛化证据。
当前 forward 仍是由公开 DNS 密度构造的 BOST proxy，不是真实相机实验。

## 7. 由结果直接决定的下一步

不训练 raw 单方向的神经近似器。它已经证明“方向独立”不等于“梯度安全”，直接
学习只会逼近一个已经失败的目标。

下一项只比较一个新机制：

> 对同一个 curved-adjoint 方向施加固定的 gradient-aware / Sobolev
> 预条件，再与匹配 continuation controls 比较。

预条件必须在看结果前固定，只能使用部署可见量与冻结几何；仍需逐帧报告三指标、
harm、完整调用账，并保留 raw curved-adjoint 作为对照。只有预条件方向先在
complete gate 上显示稳定 headroom，才有资格训练小模型去预测它。

## 8. 讲人话

这次我们确实找到了一条“以前四个方向里没有”的新路，而且它在场误差和观测误差
上都有帮助。但它把梯度细节弄坏了，所以还不能当作一个可靠的起跑方向。

这不是突破，却排除了一个很诱人的错误做法：不要先训练网络去模仿 raw curved
adjoint。先把它改造成对梯度友好的方向，再判断有没有值得学习的算法目标。

公开图表：
[v44.3 单 curved-adjoint 富集诊断](../asset_viewer.html?asset=assets%2Fblastnet_h2air_phi05_curved_adjoint_v44.png)

机器可读脱敏摘要：
[v44.3 public summary](../docs/blastnet_h2air_phi05_curved_adjoint_v44_public_summary.json)
