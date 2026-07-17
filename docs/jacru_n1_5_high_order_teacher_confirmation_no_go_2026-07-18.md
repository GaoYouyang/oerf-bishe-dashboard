# JACRU N1.5 高阶教师校正确认：NO-GO，但形成了稳定强基线

## 一句话结论

候选 `damping_high_order_b0p75` 在冻结后一次性打开的 6 个新几何簇、12 个合成场上，场相对
L2 平均改善 **3.6323%**、H1 半范数误差平均改善 **10.3084%**，全部场均为正增益；但预先冻结
的场均增益门是 **5%**，因此总判决必须是 `SYNTHETIC_CONFIRMATION_NO_GO`。

这不是“差一点就算成功”。它说明高阶教师校正是一个稳定、可复现、比简单阻尼更强的物理
基线；同时也说明继续微调同一个插值系数，不足以支撑方法论文。

## 物理范围必须说窄

当前合成夹具把观测由连续解析梯度积分生成，把重建算子写成体素有限差分加三线性采样。因此
本轮 `epsilon = G_H - G_L` 只代表：

- 连续解析梯度渲染与体素有限差分/三线性投影之间的表示和离散化失配；
- 固定的平行光线、固定采样点和固定相机几何；
- 无传感器噪声、无 camera bias 的机制隔离实验。

它不包含有限孔径、景深、弯曲光线、光流误差、标定漂移或真实反应流。因此不能把本轮结果写成
“完整 BOST 前向模型校正”或“真实实验泛化”。

## N1.5-A：先问失配能否由可见量预测

训练输入只允许使用相机姿态、局部 detector 坐标、观测、CGLS-12 暖启动投影及其局部曲率；
truth、exact mismatch、family label 和 development target 均不进入推理接口。

| 候选 | opened-development 结果 | 判决 |
|---|---:|---|
| component damping | 相对不校正的失配增益 38.62% | 冻结为简单强对照 |
| curvature-visible ridge | 相对不校正增益 45.62%；相对 damping 再增 11.68% | 12 个场中 2 个受损，NO-GO |
| PCA-16 exact-coefficient oracle | 残余失配比 0.3343 | 只说明表示上界，不可部署、不进门 |

关键教训：把 measurement mismatch 的 L2 预测得更准，不必然让三维逆解更准。

## N1.5-B：把预测器放回同预算重建闭环

校正路径的预算为：CGLS-12 暖启动、1 次可见低阶投影、CGLS-12 暖启动细化，共
`25F/24AT`。强参考为低阶 CGLS-25 的 `25F/25AT`。高阶教师采用四阶差分，但稳定低阶算子仍
负责最终求解。

opened-development 机制筛查得到：

| 方法 | field gain vs low CGLS-25 | H1 gain | worst field gain |
|---|---:|---:|---:|
| component damping | +3.721% | +10.779% | +0.162% |
| curvature-visible correction | +1.653% | +6.979% | -2.709% |
| damping + high-order teacher, beta=0.75 | +4.799% | +10.899% | +1.655% |
| direct fourth-order CGLS-25 | -5.100% | +2.030% | -10.140% |
| exact mismatch oracle | +8.617% | +11.553% | +2.203% |

“直接高阶”明显恶化，而“高阶只做误差教师、低阶继续求解”稳定改善。这是当前最有物理意义的
结构性发现，但这些数字来自已经打开的开发集，只能生成确认假设。

## 冻结与一次性确认

候选、预算、门槛、代码和新种子在 Git 提交 `67338a0` 中先冻结，随后才打开确认任务。六个种子
由 `SHA256("jacru-n1-5-confirmation-v1:{index}")` 的固定规则生成，与原 train/development/OOD
种子不重叠。输出目录禁止覆盖。

确认结果：

| 门 | 冻结阈值 | 观察值 | 通过 |
|---|---:|---:|---|
| mean field gain vs low CGLS-25 | >= 5% | **3.6323%** | 否 |
| mean H1 gain | >= 3% | **10.3084%** | 是 |
| worst geometry-cluster field gain | >= -1% | **1.3527%** | 是 |
| field harm >1% rate | <= 0% | **0%** | 是 |
| mean field gain over damping | >= 0% | **0.6864%** | 是 |
| each-family mean field gain | >= 0% | smooth **2.1703%**, interface **5.0944%** | 是 |

几何簇 bootstrap 95% 区间为 `[2.1580%, 5.2702%]`，仅作描述，不参与门。最坏单场仍为
`+0.8979%`。这说明结果是“稳定但幅度不足”，不是由少数灾难样本导致的失败。

## 为什么不继续调 beta

确认集已经打开。此后根据确认结果改 beta、改门槛或增加同族手工特征，都只能叫 post-open
探索，不能再次复用这六个种子作独立证据。并且 exact mismatch oracle 在确认集上也只有
`+5.668%` 场增益，说明当前 solver/fixture 下剩余可兑现空间本来就有限。

## 下一代算法：学习逆问题真正看见的失配

N1.6 不再把 `||epsilon - epsilon_hat||_2` 当主训练目标，而优先研究：

```text
g(x,z) = A(z)^T [G_H(x,z) - G_L(x,z)]
```

或等价的 measurement-range / Schur-sensitive 分量。原因是低阶正规方程只通过 `A^T epsilon`
感受前向失配；位于不可观测或逆问题低敏感方向的 measurement L2 改善，可能对三维场没有价值。

建议最小候选：

1. `HO-control`：本轮 beta=0.75 高阶教师，固定为非学习强基线；
2. `AT-AEM-Mean`：按 geometry/相机条件化预测 `A^T epsilon` 均值；
3. `AT-AEM-LR`：在 fit split 固定低秩场空间，只预测有界系数；
4. `AT-AEM-Op`：小型 ray-set/operator network 只预测低秩系数，并以阻尼基线作 fail-closed 回退；
5. loss 同时报 field L2、H1、held-out projection、逐 geometry 尾部与 `A/AT` 成本，不能只报
   measurement mismatch。

## 必须问何远哲师兄的六个问题

1. 实验室当前 `G_L` 的 ray、aperture、gradient discretization 和 interpolation 分别怎样实现？
2. 能否提供成对的 `G_L/G_L^T`，并对一个冻结线性化点做伴随恒等式检查？
3. 能否离线运行更高保真的 cone-ray/curved-ray `G_H`，哪怕只覆盖少量 phantom/CFD 场？
4. 是否有 flow-off、标定 bootstrap、f-number、焦面、背景距离和 optical-flow confidence？
5. 能否永久保留整台相机、整个 session 或一档 f-number，禁止训练和选参接触？
6. 师兄更希望本科毕设先交付“稳定高阶教师基线”，还是继续做 `A^T epsilon` 的小网络原型？

## 可复核入口

- Stage A：`demo_t16_operator/results/jacru_n1_5_approximation_error_headroom_development_full1/`
- post-open reconstruction：`demo_t16_operator/results/jacru_n1_5_reconstruction_aware_postopen_full1/`
- frozen confirmation：`demo_t16_operator/results/jacru_n1_5_high_order_teacher_confirmation_once/`
- 实现：`demo_t16_operator/jacru_n1_5_high_order_correction.py`
- 确认 runner：`site_tools/run_jacru_n1_5_high_order_teacher_confirmation.py`
