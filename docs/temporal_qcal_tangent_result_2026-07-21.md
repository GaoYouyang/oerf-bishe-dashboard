# 多帧三维 BOST `q_cal` 切空间诊断：v2 严格结果

> 总状态：`POSTOPEN_TEMPORAL_TANGENT_NO_GO`
>
> 证据等级：同族 synthetic voxel forward、3 个新 development rigs、621 个 q-evaluation rows；不是 OERF 真实数据、fresh audit、三维重建或新算法成功。
>
> 最重要的结论：**多帧输运能解除结构性零秩，但当前信息太弱，无法在注册噪声下可靠恢复几何。低秩先验可以制造很强的曲率，却同时产生不可接受的场模型偏差。**

## 1. 为什么保留 v1，又为什么以 v2 为准

v1 保留在 `learning_labs/results/temporal_qcal_tangent_v1/`，其 deployable `q_hat` 与总 `NO-GO` 未被推翻。但独立代码审计发现：q-trial 循环把最后一个 reacting scene 的 `fields` 误传给所有 `teacher_*` 列，因此 v1 的 teacher CRLB、teacher q error 等列无效。

v2 做了四项明确纠正：

1. 每个模型按 scene 选择 teacher field，并用零噪声非线性余项单测锁住；
2. `C_est` 的不确定性改名为 plug-in covariance proxy，严格 CRLB 只用于 teacher Jacobian；
3. residual 使用 `m-rank(B)-5` 个自由度和双侧 99% chi-square NIS；
4. 加入参数尺度不变的广义信息谱、SNR sweep 和 95% 置信半径拒答门。

v2 使用 v1 的随机 seed namespace，保证 phantom、q direction 与标准噪声方向配对。取证信息：

| 项 | SHA-256 |
|---|---|
| source | `d57b3ea537172da5fc8d4600e2f27fb3ebed1b898fbfdd8d0933dc83b4dd2db1` |
| config | `ab3a4cd1020da0f29e33cc90b7e9cf7fa911af0fe9c58d8c98dc1d30c6b288e0` |
| protocol | `a0e4063279a8c00252184cbda77c91efa1080f0c585c2f160e075a2ffbf85b8b` |

## 2. 结构结论：哪些“4D 结构”真的没有用

15 个零控制与 3 个 known-target 正控制全部通过数值门：

| 模型 | profile 结果 | 为什么 |
|---|---:|---|
| 每帧独立自由体素 | `0/5` | 每帧 `A` 满行秩，场能吸收任意局部几何响应 |
| `X=Phi H` 且两因子都未知 | `0/5` | `delta Phi` 可以吸收 `dA Phi H`；低秩本身不是几何信息 |
| 固定时间系数、自由空间因子 | `0/5` | 仅固定 temporal basis 仍允许空间因子自由补偿 |
| 同一静态场重复六帧 | `0/5` | 重复观测增加 SNR，不改变几何/场 gauge |
| 输运 + 每帧自由 innovation | `0/5` | 一旦 innovation 恢复逐帧自由度，辨识力再次消失 |
| 已知 target | `5/5`，广义 retention 约 1 | 证明投影、有限差分和秩门能发现真正的已知场信息 |

这给毕业设计一个必须记住的定理边界：**不能把 DeepONet/FNO/Tucker/CP 的低维表示本身写成“解决了联合标定”。只有它客观地缩小 nuisance tangent，且在独立数据上保持场模型正确，才可能释放几何信息。**

## 3. 精确输运：结构上有信息，实践上不可用

在 `x_t=W_t x_0` 且 `W_t` 完全正确的最有利合成条件下：

| 指标 | 3 rigs 范围 / 中位数 |
|---|---:|
| profile rank | 每个 rig `5/5` |
| 最弱广义 retention | `9.76e-5` 至 `3.63e-4`，中位 `1.34e-4` |
| trace retention | `0.90%` 至 `1.78%`，中位 `1.36%` |
| nominal model residual | 中位 `3.01e-15`，数值零 |
| reference-noise q relative-L2 | `3.47` 至 `18.13`，中位 `9.41` |
| teacher q relative-L2 | 中位 `10.54` |
| teacher linear CRLB relative RMS | 中位 `11.05` |
| NIS | `9/9` 通过，dof-normalized RMS 中位 `1.002` |
| uncertainty-authorized | `0/9` |

这里没有矛盾。rank `5/5` 只说明五个方向原则上非零；最弱广义 retention 只有约 `1e-4`，意味着几何信息落在很弱的正交补方向。NIS 通过只说明“噪声解释正确”，不说明参数置信区间足够窄。

## 4. SNR 扫描：可用窗口在哪里

SNR sweep 固定同一 phantom、q、forward 和标准噪声方向，只缩放噪声；因此它是 post-open threshold map，不是独立重复或 coverage 证明。

| noise multiplier | plugin q error 中位 | teacher q error 中位 | teacher CRLB 中位 | 授权 |
|---:|---:|---:|---:|---:|
| `0` | `0.0215` | `0.00145` | `0` | `0/9`，零噪声不做统计更新 |
| `1/128` | `0.0895` | `0.0847` | `0.0863` | `7/9` |
| `1/64` | `0.174` | `0.167` | `0.173` | `0/9`，95% 最大半径仍过宽 |
| `1/32` | `0.354` | `0.332` | `0.345` | `0/9` |
| `1/16` | `0.698` | `0.661` | `0.691` | `0/9` |
| `1` | `9.41` | `10.54` | `11.05` | `0/9` |

当前 synthetic base noise fraction 是 `0.002`；`1/128` 对应约 `1.56e-5`。这不是现实相机噪声结论，但它说明：若只靠 iid 重复把标准误差压低 128 倍，粗略需要约 `128^2=16384` 倍独立信息量，单纯“多拍几帧”大概率不可行。更现实的路线是已知 calibration target/sentinel、更有利的视角或几何激励、外部 PIV/速度约束，或真正可信的动力学模型。

## 5. 三个很容易写错的“正结果”

### 5.1 冻结 PCA 基

pilot PCA 的最弱广义 retention 很高：rank 4/8/16 的中位数约 `0.864/0.798/0.648`。但 nominal clean model residual 中位仍为 `9.13%/5.95%/5.42%`，最差可到 `32.67%`。这是**先验缩小 nuisance 后制造的强曲率**，不是正确标定。

### 5.2 10% velocity mismatch

它的最弱广义 retention 中位反而从精确输运的 `1.34e-4` 增到 `2.10e-4`，但 nominal model residual 中位为 `0.155%`，q error 中位 `14.26`。旧 RMS 门会错接收；v2 的 NIS `0/9` 通过，因此全部拒答。

### 5.3 输运 + 共享 birth

加入共享 source 后 nominal residual 回到数值零，说明场模型更匹配；但最弱广义 retention 降到中位 `1.58e-5`，q error 中位 `20.39`。更灵活的 nuisance 修复了场，却抹掉更多几何信息。v2 将其 `research_signal` 判为 false。

## 6. 对“算子学习 + 三维重构”的直接启发

本轮没有产生可投稿算法，但把下一步从泛泛“做 FNO”缩成了三个可证伪候选：

1. **Uncertainty-gated transport-profiled variable projection**：先做严格 classical baseline；迭代更新几何，使用 teacher/plug-in/parametric-bootstrap 三层不确定度，任何失配 NIS 触发回退。
2. **Sentinel-anchored 4D reconstruction**：在普通时序 BOST 之外加入少量已知 target/reference frame 或可测 PIV/速度约束，直接抬高最弱广义特征值；研究重点是“多少 anchor 才够”，不是堆大网络。
3. **Learned tangent proposal, classical verifier**：DeepONet/FNO 只预测 transport/innovation tangent 或 warm start；最终更新必须由真实 forward JVP/VJP、held-out camera/time NIS 和置信椭球验证。网络不能自行授权几何更新。

第三条最贴合用户希望的“开发自己的算子模型”，但它必须击败：精确/失配 transport、固定 PCA、Tucker/CP、TDBOST、普通 DeepONet/FNO，以及同调用预算的迭代 variable projection。第一篇可靠成果更可能来自**安全拒答 + anchor/实验设计**，而不是宣称一个黑盒网络普遍更准。

## 7. 下一轮不能跳过的真实输入

需要何远哲师兄回答：

1. 连续 run 中多少帧共享同一套相机几何？是否有原始 timestamp、drop-frame 和同步误差？
2. curved/straight forward 是否可调用？能否给 geometry JVP/VJP 或至少固定 pose 的 matrix-free `A/A^T`？
3. 组内最关心的几何误差量级：yaw/pitch/roll 多少度、detector shift 多少 pixel/domain unit？
4. flow-off/reference 重复帧的位移噪声、相机间 covariance 和 temporal drift 大约多大？
5. 是否有 calibration target、静止已知场、PIV 速度、火焰前缘或其他 sentinel，可作为少量 anchor？
6. TDBOST 的 tensor factors、rank、time window 与 birth/extinction 失效案例能否提供？
7. 真实论文终点是 field-L2、held-out reprojection、PIV compensation、front fidelity、速度还是 4D temporal fidelity？

在这些合同到位前，下一项本机工作应是一个 matched T0 的 500-noise bootstrap + q-amplitude sweep，以及 one-step 对迭代 variable projection；它能验证 coverage 与非线性收敛，但仍不会替代真实实验。

## 8. 文件入口

- [预注册与 v2 纠错附录](temporal_qcal_tangent_protocol_2026-07-21.md)
- [v2 machine report](../learning_labs/results/temporal_qcal_tangent_v2/report.json)
- [逐模型指标](../learning_labs/results/temporal_qcal_tangent_v2/model_metrics.csv)
- [621 条 q 评估](../learning_labs/results/temporal_qcal_tangent_v2/q_trials.csv)
- [结构控制](../learning_labs/results/temporal_qcal_tangent_v2/structural_controls.csv)
- [诊断图](../learning_labs/results/temporal_qcal_tangent_v2/temporal_qcal_tangent_diagnostic.png)
- [SHA-256](../learning_labs/results/temporal_qcal_tangent_v2/checksums.sha256)

最终允许的句子只有：**在当前同族 synthetic proxy 中，多帧精确输运可产生非零五维几何信息，但注册噪声使其远未达到实践可辨识；冻结低秩基带来的强曲率与场模型偏差同时存在。**
