# v282：有限背景虚拟像素位移接口通过独立审计

2026-09-05。九个现有三维场、四个时刻与十三份相机标定生成 468 组虚拟位移图，五、七、九相机子集共 1,404 个诊断单元。固定薄层近似的位移差异中位数约 1.9%–2.1%，最坏约 6.0%。

## 这次解决什么

按师兄建议继续使用现有三维场生成二维位移，而不是等待已遗失的配对实验图。旧代理只有沿射线的横向梯度积分；本轮加入沿路径变化的背景距离权重、包含透视与畸变的像素投影导数，以及同一背景点的流动开启减关闭符号。

固定薄层对照只把分布式深度权重替换成体积中心平面的权重，其余不变；没有根据结果拟合换算系数或选择背景平面。这是经典光学的一阶实现，不主张新颖性。[相关预印本](https://arxiv.org/html/2607.15567v1)讨论射线方程、薄层与边界折射率假设；本项目另行推导并检验固定背景点的一阶像素映射。

## 独立结果

正式实现使用逐网格单元两点积分，独立实现使用三点积分、另一套三线性梯度组装和复步长像素导数。12/12 独立检查通过；最大状态差 4.37e-12、算子差 3.10e-12、像素导数差 8.30e-14，离散伴随误差不超过 4.37e-17。重新构建的相机乱序结果和子集结果一致。

判决 `PASS_LINEAR_VIRTUAL_PIXEL_INTERFACE_V282`，仅表示该线性虚拟接口可用。下面是固定薄层近似相对分布式像素位移的 L2 差异，每层 117 个模型-标定文件组合，分位数采用 higher。

| 相机数 / Cameras | t | Median | p90 | Worst |
|---:|---:|---:|---:|---:|
| 5 | 0.0 | 2.1123% | 3.3357% | 3.8651% |
| 5 | 0.25 | 2.0508% | 5.5113% | 5.8740% |
| 5 | 0.75 | 1.9757% | 3.6693% | 4.5823% |
| 5 | 1.0 | 1.9626% | 3.5153% | 4.1639% |
| 7 | 0.0 | 2.0839% | 3.6594% | 4.3275% |
| 7 | 0.25 | 2.0203% | 5.0384% | 5.9773% |
| 7 | 0.75 | 1.8789% | 3.7406% | 4.3290% |
| 7 | 1.0 | 1.8883% | 3.7240% | 4.1692% |
| 9 | 0.0 | 2.0384% | 3.8752% | 4.2721% |
| 9 | 0.25 | 2.0105% | 4.8843% | 5.7058% |
| 9 | 0.75 | 1.9219% | 3.5440% | 4.1951% |
| 9 | 1.0 | 1.8917% | 3.5221% | 4.0474% |

## 必须保留的边界

通过的是线性化光学接口与精确离散伴随，不是重建或加速。背景、折射率尺度和边界支撑均为明确虚拟设定；每相机仅用 8×8 射线。十三份文件只对应十一种不同像素算子，不能当作十三个独立实验。 这里保留预先固定的文件加权统计，不给出独立总体的置信区间。三维场仍按每场均值与 RMS 归一化，随后施加固定支撑窗使边界折射率连续接回背景；这会改变模拟场。背景距离、折射率幅度都是合成约定，不是找回了实验物理单位。

这是零扰动附近的一阶导数；未在这些给定场上验证有限幅度非线性曲线光线。先前制造场射击验证不能替代本数据上的非线性验证。最大约 0.6134 的线性虚拟像素分量依赖设定幅度，不能当作实测位移。

离线每套实现包括 26 次几何组装、468 次主方法与 468 次薄层数据 forward、65 次各种 forward probe 和 52 次 adjoint；没有求解器、学习器或在线性能测试。最坏约 6% 是此虚拟条件下的观测差别，不是重建提升，也不能倒推旧实验失败的原因。

下一步先冻结新像素接口下的合格重建参考与可识别性检查，再决定最小预测器；不沿用旧梯度积分的精度结果。v281 固定补偿仍关闭，不训练大模型，不租 GPU。 `algorithm_breakthrough=false; paper_success=false; resource_speedup=false; external_generalization=false; curved_ray_validated=false; real_bost=false`。

# v282: finite-background virtual pixel interface independently validated

Nine existing 3D sources, four times and thirteen calibration files produce 468 virtual displacement maps and 1,404 five/seven/nine-camera diagnostics. The fixed thin-plane approximation differs by about 1.9%–2.1% at the median and 6.0% at worst.

## What changed

Following the advisor's suggestion, existing 3D fields generate virtual 2D displacement without waiting for lost paired images. The old proxy integrated transverse gradients. This audit adds the distributed background lever arm, perspective/distortion pixel Jacobian, and flow-on minus flow-off displacement of the SAME background feature.

The fixed thin-plane control replaces only the depth weight with its volume-center-plane value. No conversion coefficient or plane is chosen from results. This is a first-order classical optical implementation, not a novelty claim. The [related preprint](https://arxiv.org/html/2607.15567v1) discusses ray equations and thin-object/boundary-index assumptions; this project separately derives and checks its fixed-feature pixel tangent.

## Independent result

Formal two-point cell integration is checked by three-point integration with separately constructed trilinear gradients and complex-step sensor derivatives. All 12 independent checks pass: maximum state/operator/Jacobian differences are 4.37e-12/3.10e-12/8.30e-14, and discrete adjoint error is at most 4.37e-17. Fresh camera-order reconstruction and subset checks agree.

Decision: `PASS_LINEAR_VIRTUAL_PIXEL_INTERFACE_V282`. The table above reports thin-plane versus distributed-pixel relative L2 differences, with higher quantiles over 117 model/calibration-file combinations per stratum. It is not a reconstruction score.

## Limits and next decision

This validates a linearized optical interface and exact discrete adjoint, not reconstruction or speed. Background, refractivity scale and boundary support are explicit virtual choices, with only 8×8 rays per camera. Thirteen files contain eleven distinct pixel operators, not thirteen independent experiments. Frozen file-weighted summaries are retained without independent-population confidence intervals. Each source remains mean/RMS normalized and is multiplied by a fixed support window to join continuously to the background index. This changes the simulated field. Background distance and refractivity amplitude are synthetic conventions, not recovered physical units.

Only the derivative at zero perturbation is validated. Finite-amplitude nonlinear curved rays on these supplied fields are untested; earlier manufactured shooting checks cannot substitute. The maximum linear virtual component of about 0.6134 pixel depends on the chosen synthetic amplitude, not measurements.

Per implementation, offline work includes 26 geometry builds, 468 primary and 468 thin data forwards, 65 assorted forward probes and 52 adjoints. There is no solver, learner or online performance experiment. The roughly 6% worst optical difference is not a reconstruction gain or a retrospective explanation of old proxy failures.

Next freeze an adequate reconstruction reference and identifiability audit for the new pixel interface before any small predictor. Old gradient-integral accuracy does not transfer. The v281 fixed correction remains closed; no large-model training or GPU rental. All six claim flags shown above remain false.
