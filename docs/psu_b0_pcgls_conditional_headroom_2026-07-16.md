# PCGLS 条件化上限审计：空间存在，但不在视角数和噪声标签里

日期：2026-07-16
状态：`PCGLS_CONDITIONAL_HEADROOM_DEVELOPMENT_COMPLETE_FRESH_NOT_USED`
证据等级：post-open development diagnostic only

## 一句话结论

105 个固定正定 PCGLS 候选中，逐样本 truth oracle 在 validation/calibration
相对静态 PCGLS-4 可降低场误差 6.52% / 7.22%；说明有限频谱家族仍有明显
上限。可是 train-selected view-count 策略只有 +0.76% / -0.26%，
view-count+noise 为 -0.11% / -5.65%，不能迁移。

相反，使用**合成形态标签**在 `risk_train` 选择四类频谱专家，再固定迁移到
validation/calibration，可得到 +2.69% / +2.38%，两边 bootstrap 下界均大于
0。这不是可部署算法，因为真实测量没有 plume/shock 标签；但它明确指出：

> 下一模型应该从可观测初始伴随场中识别形态/频谱类型，而不是继续扩大只看
> 视角数、相机几何和噪声摘要的 MLP。

## 1. 有限候选与公平预算

- 5 个 Sobolev strengths：3、4、5、6、7；
- 3 个 epsilon：0.02、0.05、0.1；
- 7 个轴向模式：isotropic、x/y/z low、x/y/z high；
- 合计 105 个固定 SPD multiplier；
- 每个候选都运行 PCGLS-4，真实账本均为 `4F+4AT`；
- 基线固定为 validation-selected `strength=4, epsilon=0.05, isotropic`；
- 只重建 `risk_train/risk_validation/risk_calibration`，fresh 完全未加载。

## 2. 策略结果

正数表示相对静态 PCGLS-4 降低 field relative-L2。

| 策略 | validation mean | calibration mean | validation p10 | calibration p10 | 解释 |
|---|---:|---:|---:|---:|---|
| train global | +0.35% | -0.22% | -2.75% | -2.81% | 单个全局参数不稳定 |
| train view count | +0.76% | -0.26% | -0.73% | -1.50% | 视角数不能解释形态差异 |
| train view + noise | -0.11% | -5.65% | -6.40% | -15.55% | 小分层过拟合，严格否掉 |
| train family label | +2.69% | +2.38% | 0.00% | 0.00% | 有迁移信号，但标签不可部署 |
| split-family truth oracle | +4.15% | +3.10% | -1.10% | -2.42% | target truth 泄漏，只作上限 |
| sample combined oracle | +6.40% | +7.04% | +1.25% | +0.89% | 不可部署上限 |
| sample field oracle | +6.52% | +7.22% | +1.59% | +0.99% | 不可部署上限 |

view+noise calibration 的最坏逐场恶化达到 -46.33%。这说明 48 个 train
样本被切成 view/noise 小格后，从 105 个候选中选最小 loss 会严重过拟合；
不能把更多 metadata 分桶误写成“物理条件化”。

## 3. 形态标签选择了什么

`risk_train` 六种解析形态最终只使用四个专家：

| 合成形态 | train-selected expert |
|---|---|
| plume | strength 5，epsilon 0.05，isotropic |
| wavy front | strength 5，epsilon 0.05，isotropic |
| thin front | strength 4，epsilon 0.05，isotropic |
| double front | strength 4，epsilon 0.05，isotropic |
| annular kernel | strength 3，epsilon 0.1，x-high |
| oblique shock | strength 3，epsilon 0.02，x-high |

这组映射具有可解释性：宽尺度 plume/wavy-front 需要更强各向同性低频先验；
annular/shock 选择较弱、带轴向偏置的先验。但坐标轴含义仍绑定当前 PSU
support 几何，不能直接外推到任意相机系统。

family-label 方案在 calibration 平均改善 2.38%，bootstrap 95% 区间为
[1.15%, 3.67%]，但最坏样本仍为 -6.53%，front-F1 平均为 -0.26%。所以它
只证明“形态有选择信号”，不证明这一硬路由已经安全或适合作为论文算法。

## 4. 为什么 V1 看不到这个信号

V1 编码的是：

- active camera geometry 和 view mask；
- 每相机白化观测 RMS、分量相关性和 sigma；
- 全局 pooled mean/max 与四个标量摘要。

它没有读取首步精确伴随

\[
g_0=A^\top M^\top\Sigma^{-2}y,
\]

而 `g0` 才把相机观测映射回三维坐标，包含 plume 宽尺度、front 方向性、
annular 结构和 shock 倾角的可观测频谱痕迹。headroom 结果因此支持新的最小
假设：从 `g0` 提取低维径向/轴向 power-spectrum descriptor，再预测四个
专家的 convex weights。

## 5. 下一算法：OMSE-PCGLS

暂名 **Observable Morphology Spectral Experts PCGLS**：

```text
initial exact normal g0 = AT W y
  -> normalized 3-D power-spectrum descriptor
  -> small calibrated selector
  -> convex log-space mixture of four positive experts
  -> freeze multiplier
  -> four-step PCGLS with the same shared initial AT
```

专家混合在 log-space 完成：

\[
\log M_\theta(k)=\sum_j w_j(g_0)\log M_j(k),\qquad
w_j\ge0,\quad \sum_jw_j=1.
\]

因此 multiplier 保持严格正定；selector 只选择/混合已验证的经典专家，不
直接生成三维场，也不输出任意频谱。实现必须让 `g0` 与 PCGLS 第一阶段共享
同一次 `AT`，总预算仍是 `4F+4AT`。

最低消融：

1. 静态 PCGLS-4；
2. train-global expert；
3. hard family-label oracle，仅作不可部署上限；
4. `g0` radial spectrum only；
5. radial + axis spectrum；
6. geometry/noise only，也就是 V1；
7. hard expert selection 与 convex mixture；
8. selector confidence fallback 到静态 PCGLS-4。

## 6. 判决边界

本审计没有真实 PSU 测量值、实验三维 truth、CFD truth、flow-off covariance
或新 fresh。family label 和 per-sample oracle 均不可部署；它们只用于判断
候选家族和条件信号是否值得继续。

当前允许的结论是：

- 有限 SPD 频谱家族仍有 6%–7% 的逐样本上限；
- 简单 view/noise 条件化不能迁移；
- synthetic morphology label 能迁移约 2.4%–2.7%，支持从初始伴随场学习
  observable morphology descriptor；
- 不能宣称 OMSE-PCGLS 已成功，也不能宣称优于 FNO/DeepONet/UNO-CG。

## 7. 产物

- [严格公开 JSON](psu_b0_pcgls_conditional_headroom_public_summary.json)
- [冻结配置](../demo_t16_operator/configs/psu_b0_pcgls_conditional_headroom_v1.json)
- [审计脚本](../site_tools/run_psu_b0_pcgls_conditional_headroom.py)
- [论文图 PNG](../demo_t16_operator/results/psu_b0_pcgls_conditional_headroom/psu_b0_pcgls_conditional_headroom_figure.png)
- [论文图 PDF](../demo_t16_operator/results/psu_b0_pcgls_conditional_headroom/psu_b0_pcgls_conditional_headroom_figure.pdf)
