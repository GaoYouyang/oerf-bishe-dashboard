# N2-CVCR-N0：参考积分未收敛，主候选保持 HOLD

日期：2026-07-18

机器判决：`HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED`

证据等级：`PREREGISTERED_SYNTHETIC_OPERATOR_INTEGRATION_ONLY`

## 1. 最先要读的结论

这轮没有得到算法成功，也还不能给主候选下正式 NO-GO。预注册要求 `12 x 48=576` 点与
`16 x 64=1024` 点 disk reference 的相对差不超过 `0.3%`；四个 rig 中三个通过，
`audit_large_aperture` 为 `0.4101%`，因此第一道 reference gate 失败。代码按预写死顺序返回 HOLD，
没有用后面的性能数字绕过参考值问题。

## 2. 描述性性能信号

主预算为每像素 32 条高保真 pupil 子射线。二折二次控制变量的 pooled operator RMSE 是
`0.0498241`，当前最强 stochastic baseline scrambled Sobol 为 `0.0229810`；描述性比值为
`2.168x`，即所谓 improvement 是 `-116.81%`。两个 audit rig 分别比各自最强基线差
`126.06%` 与 `125.67%`，normal-action error ratio 为 `2.442x / 2.311x`。

这不是“控制变量完全无效”。在同一 IID points 的严格配对中，主候选只在 `6.25%` 的 audit
重复上伤害 IID，说明二次残差确实消掉了部分普通 Monte Carlo 变化；但 antithetic、scrambled
Sobol、sunflower 和 disk product quadrature 是更强、更便宜、更应比较的经典对手。当前候选没有
达到升级 neural coefficient/sample allocator 的门槛。

## 3. 数值与成本审计

- 4 个预写死 rigs、3 个预算、32 个随机重复；
- IID、antithetic、scrambled Sobol、sunflower、disk product quadrature 与主候选统一按
  `16/32/64` 条高保真 pupil 子射线记账；
- 主候选同一个冻结矩阵同时做 forward 与 transpose，最大内部点积相对误差 `1.207e-13`；
- mean-bias 有限重复代理在两个 audit rig 都低于 `2.5` 门；
- 结果包 checksum 全部通过；
- 这里的 wall time 是小型稠密 Python surrogate 的本机描述，不是 NeRIF/NIRT 速度证据。

## 4. 物理边界

[Molnar 等](https://arxiv.org/html/2402.15954)的有限孔径 BOS 是 pupil 与光路上的 cone/frustum
积分。PSU 后续工作中的约 `8000 points/pixel` 是 frustum data-operator 积分点，不能全部写成
“8000 个孔径样本”。本轮只随机化二维均匀 pupil；路径求积固定，pixel footprint/PSF、场依赖
弯曲光线和 calibration error 都没有进入统计变量。

所以本结果只能说明：在一个 prescribed weak-deflection pupil 子积分上，当前二折全局二次
多项式没有击败强低差异积分。它不能证明真实相机、真实反应流、三维场、NeRIF 或 OERF 的任何优劣。

## 5. 新颖性碰撞

多项式近似加 residual Monte Carlo 不是新理论。至少必须正面对比：

1. [StackMC](https://arxiv.org/abs/1606.02261)：用样本拆分控制拟合过拟合；
2. [Primary-Space Adaptive Control Variates](https://arxiv.org/abs/2008.06722)：可解析分片多项式
   加 Monte Carlo residual；
3. [Neural Control Variates](https://arxiv.org/abs/2006.01524)：学习可积分控制变量与残差采样；
4. [Recursive Control Variates for Inverse Rendering](https://doi.org/10.1145/3592139)：在可微
   inverse rendering 中利用迭代冗余降低 primal/gradient 方差。

因此这条 N0 只能作为机制基线。可能的 BOST 特定贡献必须来自 joint pupil-path-pixel 测度、
straight-to-curved 多保真、forward/JVP/VJP 一致性、反应前沿/遮挡尾部 fail-closed 和真实三维
重建收益，而不是把全局二次 basis 换成一个网络。

## 6. 下一步如何不篡改本轮判决

下一步只做 post-open reference sensitivity：固定检查 `1024 -> 1600 -> 2304 -> 4096` 点的
disk product quadrature。它可以判断 HOLD 来自参考阶数不足还是目标积分本身不平滑，但永远不把
本轮重标成 confirmatory NO-GO/GO。若高阶参考稳定，另开新预注册才能检验更合适的 BOST 结构；
若仍不稳定，先改成 streaming reference 与 pupil-path 联合收敛审计，继续禁止训练。

## 7. 复现入口

```bash
git show e83027f

.venv/bin/python demo_t16_operator/run_n2_cvcr_n0_mechanism_gate.py \
  --config demo_t16_operator/configs/n2_cvcr_n0_mechanism_prereg_v1.json \
  --output-dir demo_t16_operator/results/n2_cvcr_n0_mechanism_gate_v1

cd demo_t16_operator/results/n2_cvcr_n0_mechanism_gate_v1
shasum -a 256 -c checksums.sha256
```

结果包中的 `report.json` 是机器判决，`aggregate_metrics.csv` 和 `paired_metrics.csv` 是统计证据，
诊断图只做可视化，不替代 CSV 与门禁。
