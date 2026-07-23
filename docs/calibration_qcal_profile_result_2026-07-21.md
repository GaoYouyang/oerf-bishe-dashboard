# 三维 BOST `q_cal` 剖面诊断：结果、含义与研究入口

- **总判决：** `POSTOPEN_QCAL_PROFILE_DATA_ONLY_NO_GO`
- **数值门：** `PASS`
- **参考先验下的排序线索：** `PASS`，但只是 prior-conditioned post-open mechanism signal
- **数据本身几何可辨识：** `NO-GO`
- **突破监测：否。** 没有真实 BOST 数据、封存 rig、自动标定、三维重建改善或神经算子训练。

## 1. 这轮真正问了什么

相机残差一致性只能告诉我们某个 measurement witness 是否稳定，不能证明几何参数可以从同一批数据中辨识。这轮把问题写成

```text
y = A(eta) x + epsilon
```

其中 `eta` 是 yaw、pitch、roll、detector shift-u 和 shift-v 五个尺度化几何参数，`x` 是未知三维折射率场。我们分别计算：

```text
G_raw    = C^T C
S_0      = C^T (I - U_r U_r^T) C
S_lambda = C^T lambda (B B^T + lambda I)^(-1) C
```

`G_raw` 默认三维场已知且不动；`S_0` 允许自由 voxel field 随几何一起变化；`S_lambda` 在 ridge 先验下限制了场的变化。只有 `S_0` 可以被解读为当前模型中的 data-only 局部几何信息。

## 2. 先看懂一个最小反例

如果 `y = eta + x`，则改变 `eta` 确实会改变观测，所以 raw `J^T J` 大于零。但只要同时令 `delta x = -delta eta`，观测就完全不变。这就是“对几何敏感”与“几何可辨识”的区别。

当前代理问题的全相机离散算子为 `A in R^(300 x 1000)`，数据行空间满秩。自由 1000 维 voxel field 可以吸收五个局部几何方向在数据空间内的变化，所以 `S_0` 必须坍缩。

## 3. 证据合同

| 项目 | 冻结设置 |
|---|---|
| synthetic rig | 6 个 |
| 每个 rig 的独立场 | smooth plume、wrinkled interface、shock-expansion 共 3 帧 |
| 几何参数 | 5 个无量纲 mode |
| 相机预算 | 从 6 台中固定选 3 台，穷举 20 个子集 |
| 先验扫描 | `1e-6, 1e-4, 2e-3, 1e-2, 1` |
| estimated Jacobian | 用 6 相机 noisy pilot 重建的 `x_hat` 做中心差分 |
| teacher Jacobian | voxelized truth 经同一 voxel forward family 做中心差分 |
| 观测生成 | 连续 analytic renderer，故 pilot 含离散化模型失配 |

这是 **all-camera pilot-assisted future camera subset design**。它可以回答“已经拍过一次 6 相机 pilot，下次只保留 3 台时如何排序”；不能回答“只有这 3 台观测时能否自洽重建和标定”。真正的 subset-only 部署要求每个候选子集独立重建 nuisance field。

## 4. 冻结数字

| 指标 | 结果 | 判读 |
|---|---:|---|
| data-only `S_0` 相对秩 | estimated `0/5`，teacher `0/5` | 不可辨识 |
| data-only trace retention 最大值 | `7.62e-30` / `7.50e-30` | 数值零 |
| reference `alpha=0.002` teacher trace retention | `4.83%` 到 `9.42%` | 正曲率来自先验 |
| estimated-vs-teacher profile Spearman | 平均 `0.956`，最低 `0.910` | 排序结构保留 |
| estimated profile oracle D-efficiency | 中位 `0.990`，最低 `0.922` | 过 post-open 排序门 |
| estimated raw oracle D-efficiency | 中位 `0.235`，最低 `0.165` | raw 排序明显失效 |
| 最大方位角间隔基线 | 平均 D-efficiency `0.098` | 几何展开不等于 profile 信息 |
| estimated-vs-teacher Jacobian relative-L2 | 平均 `0.818` | 幅值并不准，不可宣称准确 `q_cal` |

6 个 rig 中，estimated profile 与 teacher 选中完全相同子集的有 3 个；其余 3 个虽选择不同，但仍保留 `0.922` 以上的 D-efficiency。这是一条值得继续验证的排序线索，不是 fresh 泛化结论。

## 5. 不能挑最好看的 alpha

| ridge `alpha` | estimated median trace retention | teacher median trace retention |
|---:|---:|---:|
| `1e-6` | `0.0057%` | `0.0055%` |
| `1e-4` | `0.533%` | `0.515%` |
| `2e-3` | `5.684%` | `6.063%` |
| `1e-2` | `13.624%` | `14.624%` |
| `1` | `54.446%` | `53.262%` |

从弱先验到强先验，正则 profile curvature 增加约四个数量级，每个 rig 在五个 alpha 下还会切换 2 到 3 个“最优”子集。这说明 `q_cal` 大小和最优相机都可被先验强度改变。当前 `alpha=0.002` 只是前序 ridge solver 的参考值，不是实验物理常数。

## 6. 这个负结果为什么有用

1. **关闭了 raw `J^T J` 直接当可观测性的路。** raw 相机排序与 teacher profile 排序的 Spearman 平均为负。
2. **指出了真正的创新入口。** 要让 `S_0` 大于零，必须用已知标定目标、低维物理场、多帧共享时空结构或明确神经场 tangent prior 减少 nuisance 自由度。
3. **留下一个神经模型必须击败的经典基线。** 任何学习型选视角方法都要比同一 prior-conditioned Schur/profile 基线更好，并报告额外 JVP/VJP、训练和 wall-time 成本。

## 7. 最值得发展的三条线

### A. 已知 target/sentinel 标定

用已知密度体、平面背景或可独立测得的 reference frame 构造 `C`，此时不需要让自由流场与几何争夺同一批残差。这是最稳、最容易与师兄现有系统对接的路。

### B. 4D 共享场与低秩时间切空间

若多帧共享几何，而流场只在一个低秩张量或连续时间基中变化，则 nuisance tangent 不再是每帧 1000 个自由 voxel。这与何远哲师兄的 4D BOST 张量分解主线最贴合。首个必须验证的量不是重建均值，而是降维后 `S_0` 的最小特征值和近零特征向量。

### C. 受约束的 neural tangent/operator 先验

用 NeRIF、tensor basis 或 neural operator 定义一个低维场 tangent，然后在这个 tangent 上计算 profile curvature。神经模型不直接输出“真相机”，只输出受限基、PSD 低秩更正或有界阻尼；物理 solver 仍执行几何更新。只有在封存 rig 上同时改善 geometry error、field relative-L2、尾部和成本，才可以进入论文主结果。

## 8. 下次和何远哲师兄必须确认的事

1. forward 是否把相机内外参暴露为 callable，能否得到 geometry JVP/VJP？
2. straight-ray、curved-ray、位移场 residual 和最终 image residual 分别在哪一层？
3. 现有 4D 模型共享的是几何、空间 basis、时间 coefficient，还是其他 latent？
4. 是否有独立 calibration target/reference/sentinel，能不能与主重建帧分开？
5. 真实噪声协方差、参数标定不确定度和组内认可的 geometry baseline 是什么？
6. 如果只用 3 台相机，是否要求当帧自洽重建，还是允许用先前的 6 相机 pilot？

已有可直接发送的问题表在 [给师兄的第一轮接口问题](n5_d5_advisor_first_contact_2026-07-19.md)。

## 9. 一级来源只支持到哪里

- [Golub & Pereyra, Variable Projection](https://epubs.siam.org/doi/10.1137/0710036)：支持用伪逆和正交投影消去线性 nuisance variable；不支持真实 BOST 全局可辨识。
- [Aravkin & van Leeuwen, Estimating Nuisance Parameters in Inverse Problems](https://arxiv.org/abs/1206.6532)：支持在 ML/MAP 逆问题中 profile nuisance parameter；不支持当前 alpha 是物理正确先验。
- [Triggs et al., Bundle Adjustment: A Modern Synthesis](https://link.springer.com/chapter/10.1007/3-540-44480-7_21)：支持消去三维结构后求 reduced camera system，也提醒 gauge freedom；不能把稀疏特征点直接等同于连续折射率场。
- [Differentiable Uncalibrated Imaging](https://arxiv.org/html/2211.10525)：支持联合优化未知 measurement coordinates 和场，并明确自由度过多时需要 consistency constraints；其实验主要是 CT，不是 BOST。
- [OASIS sensor arrangement](https://arxiv.org/html/2309.10698v2)：支持用 generalized Schur complement 消去 landmark 并对有效 FIM 做传感器布局；这只是 SLAM 方法学类比。
- [Geometry Calibration in Tomography with a Differentiable Ray-Based Model](https://arxiv.org/abs/2606.21405)：支持几何参数的可微 ray model 和高效 VJP；它是通用 tomography 预印本，不证明 OERF 现有 callable 已具备该接口。

上述文献没有一篇证明 Schur/Fisher `q_cal` 已在三维 BOST 中验证。本轮是数学上有据的迁移假设和 synthetic mechanism diagnostic。

## 10. 复现与核对

```bash
cd /path/to/oerf-bishe-dashboard
.venv/bin/python -m pytest -q \
  learning_labs/test_calibration_qcal_profile_lab.py
.venv/bin/python -m learning_labs.calibration_qcal_profile_lab \
  --output-dir /tmp/calibration_qcal_profile_repeat
```

正式产物在 `learning_labs/results/calibration_qcal_profile_v1/`：`report.json`、4 张 CSV ledger、四联图与 `checksums.sha256`。当前专项测试为 `13 passed`。重复运行一致只证明这个 synthetic 合同可复现，不会把 post-open 开发集变成 fresh 科学证据。
