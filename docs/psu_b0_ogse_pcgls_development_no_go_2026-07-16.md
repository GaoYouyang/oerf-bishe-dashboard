# OGSE-PCGLS V2 开发判决：风险路由变安全，但 calibration 未过 2% 门槛

更新日期：2026-07-16
证据等级：`post-open development diagnostic only`
最终判决：`NO-GO；不得宣称算法优于现有方法`

## 一句话结论

OGSE-PCGLS V2 用首个共享伴随场 `g0 = A^T W y` 提取 44 个频谱/空间特征，
再从 train-only 贪心选择的固定 SPD 专家库预测逐专家收益。修正“基线专家得分
最高时仍发生混合”的语义错误后，严格路线在 validation / calibration 相对
static PCGLS-4 的三维场误差分别改善：

- validation：`+2.423%`，95% bootstrap CI `[+1.237%, +3.676%]`；
- calibration：`+1.651%`，95% bootstrap CI `[+0.700%, +2.902%]`；
- 两个划分的 `>1% harm rate` 都为 `0%`；
- 但 calibration 均值没有达到预注册的 `+2%` 门槛，因此总门判定为
  `NO-GO`。

这不是失败得毫无价值。它把研究瓶颈从“有没有可用频谱专家”缩小为：
**怎样仅凭测量可观测量，可靠识别何时应偏离强静态 PCGLS 基线。**

## 1. 算法到底做了什么

### 1.1 固定计算预算

所有比较都使用同一个四阶段 PCGLS：

```text
4 forward calls + 4 adjoint calls
```

首个伴随场既是 PCGLS 必需计算，也是 selector 的输入，没有额外调用
`A` 或 `A^T`。selector 一旦生成预条件器，整个求解过程中保持固定正定，
所以本次实现没有借“动态改方向”绕过 PCGLS 公平预算。

### 1.2 train-only 专家库

在 105 个固定 Sobolev-SPD 候选中，仅用 `risk_train` 真值做贪心覆盖：

| 专家数 | train 逐样本 truth-oracle 平均场收益 |
|---:|---:|
| 4 | `+4.156%` |
| 6 | `+4.817%` |
| 8 | `+5.086%` |

这说明频谱候选家族不是完全没有上限，但随着专家数增加，边际收益迅速下降。

### 1.3 可部署 selector

selector 不读取形态标签、真值场或 fresh 数据。输入只有共享首伴随场的
44 个可观测特征；训练目标是每个固定专家相对 static PCGLS-4 的逐样本场收益。
通过 family-stratified 四折 OOF 屏选 ridge 正则、温度、最大混合幅度、置信阈值
和专家库大小。

最终严格候选为：

```text
bank size = 4
ridge lambda = 1
temperature = 2
maximum blend = 0.5
confidence threshold = 0.146745
OOF intervention coverage = 43.75%
```

只有当非基线专家得分最高且 margin 超阈值时才允许偏离基线；否则精确回退到
static PCGLS-4。

## 2. 为什么修正 selector 语义很重要

旧实现仅检查 top-1 / top-2 margin，没有检查 top-1 是否就是基线专家。
因此即使模型明确判断“基线最好”，softmax 混合仍可能改变预条件器。旧结果中
validation 的 `thin_front` 曾出现 `-5.208%` 场损失。

修正后：

- 严格路线 validation 最差 paired gain 为 `-0.000014%`；
- calibration 最差 paired gain 为 `-0.000014%`；
- 两个划分均无 `>1%` 受害样本；
- 代价是 calibration 平均收益从旧诊断值进一步回落，暴露出真实容量上限。

这说明安全回退不是展示层包装，而是直接改变算法行为与研究判决。

## 3. 严格路线与诊断路线

| 路线 | Validation mean | Calibration mean | 最坏 validation | 最坏 calibration | `>1%` harm |
|---|---:|---:|---:|---:|---:|
| 风险约束 OGSE | `+2.423%` | `+1.651%` | `-0.000014%` | `-0.000014%` | `0% / 0%` |
| 放宽风险诊断 | `+3.560%` | `+2.554%` | `-12.669%` | `-7.369%` | `4.17% / 6.67%` |

放宽路线说明“收益潜力”真实存在，但其灾难性负尾使它不能成为论文主结果。
严格路线说明通过可观测回退可以消除已知灾难尾部，但现有 44 维特征和线性
gain regressor 仍不足以在 calibration 同时保住 `+2%`。

## 4. 六项开发门

严格路线通过：

- validation mean `>= 2%`；
- validation bootstrap lower `> 0`；
- validation harm rate `<= 5%`；
- calibration bootstrap lower `> 0`；
- calibration harm rate `<= 5%`。

未通过：

- calibration mean `>= 2%`。

因此严格总门为 `5/6`，但规则是六项全过才允许进入下一阶段，不做四舍五入，
不把 `+1.651%` 描述为“接近成功”。

## 5. 当前最值得继续的研究问题

### P1：把“专家收益”改成带风险的条件分布

当前 ridge 只预测均值收益。下一版应同时预测：

```text
E[gain | observable]
P(gain < -1% | observable)
lower quantile of gain
```

只有下分位数也为正时才介入。可先用低容量 quantile ridge / conformal residual，
而不是立刻上大网络。

### P2：增加便宜但更物理的可观测特征

优先测试：

- 按视角分组的 whitened residual spectrum；
- 正反投影一致性 `A g0` 的角向不平衡；
- 不同视角的梯度能量与相位一致性；
- PCGLS 第一步后残差下降率和方向夹角。

这些量仍能共享求解器已有调用，或只增加一次可明确记账的 forward call。

### P3：从“混合所有专家”改成“基线到单专家的安全路径”

当前 softmax 会把多个未必互补的专家一起混合。下一版可只沿
`baseline -> predicted best expert` 做一维 line search，并用 train-only
校准的最大安全步长限制干预。这比扩大神经网络更容易解释，也更符合固定 SPD
约束。

## 6. 证据边界

本报告只允许支持：

> 在真实 PSU 几何支持、解析反应形态与合成相机噪声的 post-open 开发数据上，
> 首伴随场包含可用于安全频谱路由的形态信息；当前风险约束 OGSE 相比 static
> PCGLS-4 有稳定正均值，但未通过 calibration `+2%` 门槛。

它不支持：

- OERF 实验数据上有效；
- 解析形态等价于 CFD；
- 已优于 DeepONet、FNO、UNO-CG、NeuralIF 或 TV 方法；
- 已形成可投稿的新算法；
- 可以打开或重复使用 fresh 结果继续调参。

## 7. 复现

```bash
cd "${REPO_ROOT}"

PYTHONPATH=. .venv/bin/python site_tools/run_psu_b0_ogse_pcgls_development.py \
  --config demo_t16_operator/configs/psu_b0_ogse_pcgls_development_v2.json \
  --development-report private_library/external_datasets/psu_bost_flight_body/b0_residual_risk_development_v1/private_report.json \
  --probe-private-report private_library/external_datasets/psu_bost_flight_body/b0_observable_morphology_probe_v1/private_report.json \
  --headroom-private-report private_library/external_datasets/psu_bost_flight_body/b0_pcgls_conditional_headroom_v1/private_report.json \
  --view-root private_library/external_datasets/psu_bost_flight_body/all_view_geometry_audit \
  --device mps \
  --private-output private_library/external_datasets/psu_bost_flight_body/b0_ogse_pcgls_development_v2/private_report.json \
  --public-output docs/psu_b0_ogse_pcgls_development_public_summary.json

PYTHONPATH=. .venv/bin/python site_tools/plot_psu_b0_ogse_pcgls_development.py \
  --input docs/psu_b0_ogse_pcgls_development_public_summary.json \
  --output demo_t16_operator/results/psu_b0_ogse_pcgls_development/psu_b0_ogse_pcgls_development_figure.png
```

公开摘要不含私有样本逐行数据、绝对路径、VPN 内容或订阅论文。
