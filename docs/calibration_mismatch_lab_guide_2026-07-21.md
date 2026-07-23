# 三维 BOST 标定失配小实验：从 `A_true != A_est` 到可拒答校准

> 当前状态：`MECHANISM_SCREEN_COMPLETE_NO_ALGORITHM_OR_REAL_DATA_CLAIM`
>
> 证据等级：`SYNTHETIC_3D_BOST_POSE_MISMATCH_MECHANISM_ONLY`
>
> 突破监测：**没有突破；总门为 NO-GO**
>
> 适合放在学习路线：Day 3 之后、Day 8 泄漏审计之前
> 硬件：CPU 即可，不需要租 GPU

## 1. 这次只问一个很窄的问题

前一个一维算子基础实验假设几何参数已经准确。真实 BOST 更麻烦：生成观测的真实射线算子记作
`A_true`，重建程序拿到的标定算子记作 `A_est`。如果相机方位、俯仰、滚转或探测器横向位置有小偏差，二者就不相等：

```text
y = A_true x_true + noise
x_hat = Solve(A_est, y)
```

这次不训练 DeepONet、FNO 或 NeRIF，也不宣称解决一般相机自标定。问题被刻意缩成：

> 已经知道一条耦合标定误差的方向和符号，只不知道它的一维幅度。能否完全不看三维场真值，靠留一相机重投影选择修正幅度？

这是一个有利先验。若方法在这个一维门上仍不安全，直接让网络同时回归多相机完整位姿更没有依据。

## 2. 为什么与何远哲师兄的方向相连

NeRIF 把 BOST 折射率场写成连续神经隐式场，并通过观测约束重建；它说明“连续场表示 + 可微 forward”是课题组已经走通的重要主线，但公开论文不能被解读为已经证明了未知相机几何下的鲁棒性。[NeRIF arXiv 原文](https://arxiv.org/abs/2409.14722)

TDBOST 进一步把目标推向高速、四维时空重建。对本项目最重要的推论是：同一实验 session 中，很多帧通常共享相机标定，因此标定参数可能作为 session-level nuisance parameter 跨帧估计。这个推论需要师兄确认真实采集合同，不能由本合成实验替实验室下结论。[TDBOST DOI](https://doi.org/10.1145/3809488)

相邻领域已经明确研究“名义采集几何与真实几何不一致”：可微射线模型可以联合优化体场和几何；未标定成像工作也把 sensor location / projection angle 当作未知测量坐标。它们支持问题的重要性，但 CT 结论不能直接搬成 BOST 成功。[2026 differentiable ray calibration](https://arxiv.org/abs/2606.21405) · [Differentiable Uncalibrated Imaging](https://arxiv.org/abs/2211.10525)

多相机折射成像研究还表明，折射投影模型、内外参与 ray tracing 应被放在同一标定链中；普通针孔近似在折射界面下可能形成系统误差。这是方法学背景，不是 OERF 数据结论。[开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5713033/)

## 3. 实验怎样避免一个明显的 inverse crime

每个合成场有两条不同数值链：

1. **观测侧**：连续解析场及其解析梯度沿真实 ray 积分；
2. **反演侧**：`10 x 10 x 10` voxel 场，经有限差分、三线性插值、相机平面投影和 ray 求和形成离散算子。

所以观测并不是拿同一张离散矩阵乘 voxel truth 再反演。它只排除了这一种窄 inverse crime，并没有排除合成到真实、直线到曲光线、噪声模型和光流提取误差等更大差距。

固定合同如下：

| 项目 | 本轮设置 |
| --- | --- |
| rig | 6 个确定性随机基准 rig |
| 相机 | 每 rig 6 台平行射线相机 |
| detector | 每相机 `5 x 5` ray，每 ray 2 个 `u/v` 分量 |
| 体素 | `10^3 = 1000` 个未知量 |
| 场 | 3 种解析 morphology proxy，共每档 18 个 case |
| 失配档 | `0 / 0.5 / 1 / 2` 个幅度单位 |
| 单位含义 | 主导量约为角度：1 档平均 ray direction 误差约 0.97 度，并伴随小幅 roll/横向平移 |
| 噪声 | 相对观测范数的确定性高斯噪声 |
| 求解器 | 所有方法共享同一个 Tikhonov/dual ridge 求解器和同一 case 内正则强度 |
| 选择可见量 | noisy held-out-camera measurements；不读体真值，不读 clean observation |

`1000` 个未知量只对应 `6 x 25 x 2 = 300` 个观测，本来就严重欠定；这里没有 TV、时空张量或神经先验，因此 field relative-L2 约 `0.63` 并不奇怪。本轮比较的是标定策略之间的相对变化，不是追求漂亮重建图。

## 4. 七个角色不能混在一起

| 名称 | 做什么 | 部署可见？ | 角色 |
| --- | --- | --- | --- |
| reported geometry | 直接相信误报标定 | 是 | 冻结基线 |
| naive LOCO | 每个候选在 5 台相机上重建、1 台上评分，取六折平均最小 | 是 | 高功效但可能误修 |
| half-step LOCO | 把 naive LOCO 的修正幅度固定乘 `0.5` | 是 | 简单阻尼基线 |
| single-frame LOCO-LCB | 只有六台相机的配对残差改善下置信界为正才修正，否则回退 | 是 | 单场 fail-closed |
| multiframe camera-block LCB | 先在同一 rig 的三个场内按相机平均，再跨六个 camera block 做 heuristic LCB | 有条件 | 需要“多帧共享标定”合同；统计单位不是 session/rig |
| exact geometry teacher | 直接使用 `A_true` | 否 | privileged teacher，不是可部署算法 |
| truth-selected candidate ceiling | 用体真值从有限候选网格中挑 field error 最小者 | 否 | evaluator-only candidate-grid oracle，不是理论上限 |

注意：LOCO 不等于真正独立验证集。六折都来自同一个合成场，只是在每一折把一台相机移出求解。它能阻断最直接的“同一行既拟合又评分”，不能授权真实泛化。`2.015` 只作为近似单侧 `t_5` 的机制筛查常数；六个 camera block 不保证独立，所以这里的 LCB **没有正式置信覆盖含义**。

## 5. 先看几何误差到底有多大

| 失配幅度 | `||A_est-A_true||_F / ||A_true||_F` | 解释 |
| ---: | ---: | --- |
| 0 | 0 | 数值匹配 |
| 0.5 | 0.0571 | 约 5.7% 的离散算子差异 |
| 1.0 | 0.1133 | 约 11.3% 的离散算子差异 |
| 2.0 | 0.2229 | 约 22.3% 的离散算子差异 |

这只针对本实验归一化几何。不能把“1 档”直接等同实验室的 1 度标定误差阈值，也不能从算子 Frobenius 差异推导真实折射率场误差。

## 6. 结果：功效与安全没有同时过门

下表报告相对 reported geometry 的 field relative-L2 改善。正数是改善，负数是恶化。

| 失配 | naive LOCO 平均 / 最差 | half-step 平均 / 最差 | single LCB 平均 / 回退率 | multiframe camera-block LCB 平均 / 回退率 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `-0.35% / -1.71%` | `-0.11% / -0.56%` | `0% / 100%` | `0% / 100%` |
| 0.5 | `+1.93% / -0.78%` | `+1.44% / 0%` | `+0.06% / 94%` | `0% / 100%` |
| 1.0 | `+5.66% / +0.73%` | `+3.87% / +0.57%` | `+1.93% / 72%` | `+0.85% / 83%` |
| 2.0 | `+13.27% / +4.24%` | `+8.29% / +3.07%` | `+7.76% / 50%` | `+9.58% / 17%` |

### 允许写下的三条发现

1. **几何误差是真实机制风险。** 同一个 solver 从 exact geometry teacher 的平均 field error 约 `0.631`，恶化到 2 档 reported geometry 的 `0.733`；这只在当前合成合同成立。
2. **残差最小不自动等于安全。** naive LOCO 在 1/2 档对全部 18 个 case 都改善，但零失配时平均恶化 `0.35%`、最差恶化 `1.71%`；0.5 档也出现个案恶化。
3. **可拒答会损失检出功效。** single/multiframe camera-block LCB 在所有档位都保持 100% 非劣化，但对 0.5/1 档大量回退。多帧 camera-block pooling 只在 2 档明显增加授权率；1 档不同场的可观测性反而会互相稀释。

### 不能写下的结论

- 不能说发明了通用自标定算法；
- 不能说优于 DeepONet、FNO、NeRIF 或 TDBOST；本轮根本没有训练这些基线；
- 不能说真实 BOST 场误差降低；没有真实体真值、真实几何或实验噪声；
- 不能把 18 个 case 当 18 个独立实验重复；它们只来自 6 个 synthetic rig 和 3 种 morphology proxy；
- 不能把六个 camera block 叫六个独立 session，也不能说 multiframe pooling 已适合 4D 数据；真实帧间相关、动态场和共同标定必须由师兄确认。

总门继续为 `NO-GO`。冻结门槛要求：所有非零失配档的平均收益至少 `5%`、改善 case 比例至少 `75%`、最差非零档收益不低于 `-5%`，且零失配平均收益不低于 `-5%`。当前 multiframe camera-block LCB 在 0.5 档的最小平均收益和启动率均为 `0`，因此没有通过。门槛现在也显式写入 `report.json`；保留 NO-GO 比移动阈值追求 PASS 更重要。

## 7. 对“自己的新模型”最有价值的启发

当前证据不支持直接训练一个黑箱网络输出完整相机位姿。更有根据的候选是：

### 候选 A：Observability-Weighted Session Calibration Operator

让一个小网络只预测每一帧/每一相机对标定参数的**证据权重**，而不是直接预测修正量：

```text
per-frame residual features
  -> nonnegative observability weights
  -> weighted session calibration objective
  -> bounded calibration update
  -> held-out-camera LCB gate
  -> accept or fall back to reported geometry
```

这样网络负责“哪些帧有信息”，物理 solver 负责“修多少”，LCB 负责“是否允许部署”。它比端到端回归位姿更容易做消融：uniform weights、signal-norm weights、learned weights、truth-oracle weights 上界。

### 候选 B：Geometry-Latent Neural Operator with Calibration Envelope

把小维 calibration latent `z_g` 输入 FNO/DeepONet/implicit field，但输出必须经过一个有限候选包络：

```math
x(z_g), \quad z_g \in [z_min,z_max],
```

并同时报告 nominal、best residual、half-step、LCB fallback 和 exact teacher。若 learned latent 只在平均数上赢、尾部或零失配伤害不受控，就判 NO-GO。

### 候选 C：4D Shared-Geometry / Dynamic-Field Factorization

把“跨帧共享的几何 nuisance”与“逐帧变化的三维场”显式分解：

```text
session geometry code g     (slow/shared)
time-dependent field code h_t (fast/dynamic)
joint differentiable renderer F(g, h_t)
```

真正创新门不在“用了 latent code”，而在是否能证明：固定计算预算下，未见 rig、未见动态形态和噪声 shift 的 field/H1/held-out reprojection 尾部优于同容量 geometry-conditioned FNO/DeepONet、NeRIF-style per-case optimization 与 classical joint calibration。

## 8. 下一轮实验顺序

1. **先拆误差模式。** rotation-only、translation-only、roll-only、principal-point/intrinsics-only、孔径-only 分开，避免把所有误差塞进一个标量。
2. **做可观测性图。** 对每个帧、相机和 calibration parameter 计算 finite-difference/JVP sensitivity、Fisher/Gram 条件数与 residual gain，检查哪些场根本不提供校准信息。
3. **实现不用训练的权重基线。** uniform、observation norm、predicted sensitivity norm、robust median；它们必须先于神经权重。
4. **再做小网络。** 只学 bounded nonnegative weights，参数量和推理成本逐项记录；不要先做完整 FNO。
5. **加入真实 forward 接口。** 只有师兄确认 callable、ray convention、标定字段、JVP/VJP 和 split 后，才把合成 pose builder 替换成实验室接口。
6. **最后才做 DeepONet/FNO 横向。** 相同训练场、相同 view/rig split、相同输入权限、相同 forward-call 与 wall-time 预算；同时报告平均、逐 rig、p95/worst、A/B consistency 和失败案例。

## 9. 现在必须问何远哲师兄的 10 个问题

1. 当前 BOST/TDBOST 每个时刻有多少相机、每相机多少有效背景位移点？
2. 一段 4D 序列是否严格共享同一套内参、外参、背景平面和光学常数？
3. 相机是否可能在运行中发生机械漂移或热漂移；漂移时间尺度多长？
4. 标定文件包含哪些字段：`K/R/t`、畸变、背景平面、像素尺度、aperture/f-number、折射界面？
5. forward callable 是否能显式接收几何参数并重建 ray，而不是把几何烘焙进不可微 `.so`？
6. 是否能提供 geometry JVP/VJP，或至少允许对少数标定参数做 finite difference？
7. 是否有重复标定、静态标定靶、flow-off 数据或不同时间的 calibration residual，可估计真实误差范围？
8. held-out camera reprojection 在组内是否被认可为无体真值评估，还是还有更可信的物理守恒/传感器指标？
9. 最主要真实失败是场重建伪影、跨相机 residual、PIV 下游偏差、时间闪烁，还是计算速度？
10. 允许公开/本地使用哪些 synthetic、aggregate 和真实数据；哪些必须留在实验室私有环境？

这些问题已经可以与 [师兄接口沟通单](n5_d5_advisor_first_contact_2026-07-19.md) 合并使用。没有回复时，继续公开 synthetic 可观测性实验，不替实验室生成结论。

## 10. 复跑与逐文件检查

```bash
cd /path/to/oerf-bishe-dashboard

.venv/bin/python -m pytest -q learning_labs/test_calibration_mismatch_lab.py

.venv/bin/python learning_labs/calibration_mismatch_lab.py \
  --output-dir "$HOME/Desktop/calibration_mismatch_my_run"
```

默认冻结产物：

- [四联图](../learning_labs/results/calibration_mismatch_v2/calibration_mismatch_lab.png)
- [机器报告](../learning_labs/results/calibration_mismatch_v2/report.json)
- [逐 case 指标](../learning_labs/results/calibration_mismatch_v2/case_metrics.csv)
- [候选曲线](../learning_labs/results/calibration_mismatch_v2/candidate_curves.csv)
- [逐相机 fold 账本](../learning_labs/results/calibration_mismatch_v2/camera_fold_scores.csv)
- [SHA-256 清单](../learning_labs/results/calibration_mismatch_v2/checksums.sha256)
- [实验源码](../learning_labs/calibration_mismatch_lab.py)
- [定向测试](../learning_labs/test_calibration_mismatch_lab.py)

读图时按这个顺序：左下确认失配确实改变算子；右上看残差选择面是否平；左上比较角色；右下检查每个 case 的尾部，最后才看平均数。

## 11. 一句话总结

**连续三维射线标定误差在合成 BOST 中确实能显著伤害重建；残差选择有功效但不天然安全，fail-closed 能保尾部却会漏检，而跨帧共享只有在场景提供足够可观测信息时才增加功效。下一模型应学习“证据权重”，不是无约束地猜位姿。**
