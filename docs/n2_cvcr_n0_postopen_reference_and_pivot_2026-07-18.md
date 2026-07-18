# N2-CVCR-N0：高阶参考仍未完全稳定，N0 冻结为失败基线

日期：2026-07-18

原预注册判决：`HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED`

事后参考检查：`POSTOPEN_REFERENCE_SENSITIVITY_UNRESOLVED_AT_4096`

证据上限：单个预设 weak-deflection pupil surrogate 的合成积分与参考敏感性；没有重评分、真实三维重建或 OERF 算法结论。

## 1. 现在最准确的一句话

程序完整执行且校验通过，但更高阶的确定性孔径参考到 4096 点仍未在两个困难工况上达到预设的
`0.1%` 末级稳定线。因此原来的 HOLD 不变。与此同时，已经打开的描述性结果显示二折全局二次
控制变量明显输给 scrambled Sobol / sunflower QMC；独立文献审计又确认其统计骨架已有直接先行
工作。N0 不再调参，只保留为 BOST 专用基线。

## 2. 参考阶数到底发生了什么

事后检查的阶数和点数在运行前另行提交，固定为：

| radial × angular | 孔径点数 | 用途 |
|---|---:|---|
| 16 × 64 | 1024 | 原冻结参考 |
| 20 × 80 | 1600 | 中间阶 |
| 24 × 96 | 2304 | 中间阶 |
| 32 × 128 | 4096 | 当前最高阶 |

相对 4096 点参考，原 1024 点的最大相对 L2 差为 `0.2991%`；从 2304 到 4096 点的最大末级差为
`0.1234%`。逐工况末级差如下：

| rig | 角色 | 2304 → 4096 | 描述性 0.1% 稳定线 |
|---|---|---:|---|
| development small aperture | development | 0.04869% | 通过 |
| development medium aperture | development | 0.05980% | 通过 |
| audit large aperture | audit | 0.12339% | 未通过 |
| audit boundary crossing | audit | 0.11944% | 未通过 |

这说明计算没有卡死；困难来自大孔径与穿越高梯度边界时，被积函数对 pupil 坐标更不光滑。当前
4096 点只是“更高阶的敏感性证据”，不是被证明收敛的 exact truth。

## 3. 为什么不继续给 N0 换网络

在每像素 32 条高保真 pupil 子射线下，候选 pooled operator RMSE 为 `0.0498241`，最强 pooled
baseline scrambled Sobol 为 `0.0229810`；候选高出 `116.805%`。两个 audit rig 中候选分别比
各自最强基线高 `126.065%` 和 `125.669%`。这些数字因为 reference gate 失败只能描述，不能变成
正式 NO-GO；但它们足以说明“马上神经化”没有合理的性能依据。

更重要的是，N0 的核心不是原创：

1. [StackMC](https://arxiv.org/abs/1606.02261) 已经用 out-of-sample / cross-validation 思路学习控制变量；
2. [Regression-based Monte Carlo](https://arxiv.org/abs/2211.07422) 已经把多项式最小二乘、解析积分和残差 Monte Carlo 组成完整估计器；
3. [Primary-Space Adaptive Control Variates](https://arxiv.org/abs/2008.06722) 已经给出分片多项式、解析积分和高频残差校正；
4. [Neural Control Variates](https://arxiv.org/abs/2006.01524) 与 [Automatic Integration NCV](https://arxiv.org/abs/2409.15394) 已覆盖神经控制变量和可积分网络；
5. [Recursive Control Variates](https://doi.org/10.1145/3592139) 已进入可微逆渲染的 primal/gradient 方差降低。

所以不能把“二次 basis 换成 MLP”当论文创新，也不能写“首次对孔径使用控制变量”。

## 4. 独立审计发现的四个未来修正

本轮机器判决没有被以下问题改变，但新的 confirmatory 协议必须修复：

1. **统计门：**GO 不能只看点估计；逐 rig improvement、harm 与 bias 应给 paired bootstrap 或置信区间下界/上界。
2. **伴随门：**当前 `matrix` 对自身 `matrix.T` 只证明稠密矩阵内部转置自洽，不是独立 renderer、streaming operator 或 NeRIF 的 JVP/VJP 审计。
3. **成本门：**N0 每折实际拟合逐矩阵元素的系数，不能把参数数写成 0；不同方法的 postprocess 计时也不能混成端到端速度结论。
4. **偏差代理：**重复均值标准误必须用中心化样本方差，不用包含 bias 的 RMSE 除以样本数平方根。

## 5. 下一候选：NeRIF 路径多保真，而不是普通控制变量

[NeRIF 一级来源](https://arxiv.org/html/2409.14722v2) 的网络同时输出折射率 `n(x)` 和直接梯度
`g(x)`，并让 `AD(n)` 与 `g` 保持一致；每条 ray 的路径采样数随迭代从 60 增到 200。这个双输出
结构提供了一个更贴近何远哲方向、也更有物理含义的高低保真接口：

- 低保真 `c`：直接梯度 head + straight ray + 稀疏路径点；
- 高保真 `h`：`AD(n)` 或数值微分 + 更密路径点，进一步可扩展到 curved ray / finite aperture；
- 残差：只在少量同随机状态样本上计算 `h-c`；
- 多层恒等式：`E[h] = E[c] + E[h-c]`，但两项都必须用合法、可核算的估计器；
- 训练与审计：forward、JVP 和 VJP 复用同一随机状态，并在 shock/flame-front crossing 时自动回退到高保真。

这仍只是设计假设。真正的新意不能是“用了多保真”四个字，而应同时来自：

1. `pupil × pixel footprint/PSF × path` 的联合光学测度；
2. 由 BOST 射线方程和 NeRIF 双输出定义的物理低保真模型；
3. 同随机状态的 forward/JVP/VJP 一致性或可证明误差界；
4. 对大孔径、弯曲光路、support crossing 和反应前沿的 fail-closed 回退；
5. 最终三维重建的 field-L2/H1/front 与 held-out reprojection，而不仅是单次积分 RMSE。

## 6. 新候选的最低比较合同

积分层必须包含 IID、antithetic、Owen-scrambled Sobol、sunflower QMC、deterministic disk
quadrature、StackMC/RegMC、Primary-Space ACV 与 high-order cone reference；若加入神经控制函数，还要
加入 NCV/Automatic-Integration NCV。三维层至少包含固定预算 Landweber、CGLS/Tikhonov、NeRIF
宿主和同预算 ablation。

必须逐项计数：高保真/低保真场查询、路径点、pupil 点、forward、JVP、VJP、预计算、训练时间、
峰值内存与 time-to-target。DeepONet/FNO 只有在任务真的变成“输入观测函数到输出三维场的跨实例
算子预测”时才是同层基线；它们不是 pupil 积分器的直接对手。

## 7. 复现与不可篡改边界

```bash
git show e83027f  # 原预注册
git show 478c414  # 原 HOLD 结果
git show 419235a  # 事后参考阶数在运行前冻结

.venv/bin/python demo_t16_operator/run_n2_cvcr_n0_reference_sensitivity.py \
  --config demo_t16_operator/configs/n2_cvcr_n0_mechanism_prereg_v1.json \
  --held-result demo_t16_operator/results/n2_cvcr_n0_mechanism_gate_v1/report.json \
  --output-dir demo_t16_operator/results/n2_cvcr_n0_reference_sensitivity_postopen_v1

cd demo_t16_operator/results/n2_cvcr_n0_reference_sensitivity_postopen_v1
shasum -a 256 -c checksums.sha256
```

事后阶数检查永远不能重评分原候选、改原判决或授权 learner。若要正式判定 N0 或下一候选，必须
另开预注册、修正统计/成本/伴随门，并使用未打开的新 geometry 或实验室合同允许的数据。
