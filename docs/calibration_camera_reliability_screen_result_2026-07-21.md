# 三维 BOST 相机可靠性权重回放：结果与下一步

- **判决：** `POSTOPEN_CAMERA_RELIABILITY_WEIGHT_REPLAY_NO_GO`
- **证据等级：** `OPENED_SYNTHETIC_LEDGER_LORO_CAMERA_RELIABILITY_ONLY`
- **突破监测：否。** 这是已打开 synthetic ledger 上的机制回放，不是真实标定、三维重建、泛化或论文成功。

## 1. 这轮到底问了什么

上一轮的标定失配实验发现：六台相机等权时，camera-block heuristic LCB 能拒绝明显危险的修正，却会漏掉不少本来有益的修正。最直接的问题是：

> 如果某台相机在其他 synthetic rig 上经常和其余相机给出相反的残差改善方向，能否降低它的权重，让拒答门更有功效？

这轮没有重新生成观测、没有重建新场，也没有训练神经网络。它只重放冻结的 `calibration_mismatch_v2` 逐相机 LOCO score，共 `3024` 条唯一记录；所有策略使用同一候选修正和同一 heuristic LCB。这样能隔离“怎样聚合相机证据”这一项机制，但不能回答真实 BOST 是否有效。

## 2. 初学者先分清三个词

### 2.1 可靠性 `q_rel`

它问的是：这台相机的残差变化是否与其他相机一致、是否像稳定的 measurement witness。本轮的 LORO consensus-correlation weight 属于这一类。

### 2.2 标定可观测性 `q_cal`

它问的是：改变某个几何参数时，观测会不会发生可区分的变化；多个位姿、焦距或平移参数是否互相耦合。真正的局部诊断至少需要残差对几何参数的 Jacobian、尺度归一化 `J^T J` 谱、近零特征方向和参数相关性。本轮没有 geometry JVP/VJP，所以**不能**把残差一致性权重称为 observability weight。

### 2.3 场信息量 `q_field`

它问的是：某台相机为三维折射率场增加了多少独立信息。一个视角对标定参数很敏感，不等于它一定能约束场的 near-null mode；反过来也一样。以后应以 view-conditioned normal operator、边际秩/谱改善或独立目标相机收益来定义，不能把它和 `q_rel` 混成一个分数。

## 3. 主候选怎样工作

对每个目标 synthetic rig：

1. 只用另外 `5` 个 rig 拟合六个相机权重；目标 rig 不参与拟合。
2. 对每台相机，计算它的候选残差改善与“其余五台相机改善中位数”的相关性。
3. 负相关截断到零，再把权重限制在 `[0.5, 2]`；归一化后最大与最小权重之比不超过 `4`。
4. 在目标 rig 上，选择器只读 noisy LOCO score、candidate label 和冻结权重。体真值、clean observation、真实几何、field gain 和失配幅度都不是模型输入。
5. weighted heuristic LCB 为正才允许非零修正，否则精确回退到 reported geometry。

这只是六个闭式标量，不是“新网络”。它应该成为未来小网络必须击败的低成本基线。

## 4. 冻结结果

| 失配档 | uniform 平均场收益 | LORO reliability 平均场收益 | 增量 | reliability 回退率 | 判读 |
|---:|---:|---:|---:|---:|---|
| 0 | 0.00% | 0.00% | 0.00 pp | 100% | 两者均回退 |
| 0.5 | 0.00% | 0.00% | 0.00 pp | 100% | 小失配仍完全漏检 |
| 1 | 0.85% | 4.05% | +3.19 pp | 16.67% | 功效明显增加，但均值仍低于 5% 门 |
| 2 | 9.58% | 11.20% | +1.62 pp | 0% | 六个 rig 均改善，但不是 fresh 证据 |

主候选在 1 档有 `5/6` 个 rig 改善，2 档为 `6/6`；但相对 uniform 的逐 rig 增量并非全正。2 档的 seed `503` 从 `11.88%` 降到 `10.89%`，差 `-0.99` 个百分点。图中保留了这个尾部，不能只汇报均值。

相机 2 的平均归一化权重约为 `0.0889`，其余相机约为 `0.166--0.195`。这说明当前六个 synthetic rig 中确实有一个稳定的 camera-role pattern；也正因为如此，它可能只是在记住相机身份，不能自动迁移到换了相机顺序或几何的实验装置。

## 5. 为什么总门仍是 NO-GO

冻结门要求所有非零失配档同时满足：平均 field gain 至少 `5%`、改善 case 比例至少 `75%`、worst gain 不低于 `-5%`。0.5 档的平均收益与改善比例均为 `0`，1 档均值只有 `4.05%`，因此主候选没有过门。

更重要的是，这里的 `6` 个 camera fold 高度相关：每个 LOCO fold 的五相机训练集大量重叠；同一 score surface 又同时用于候选准入和排序。这会产生 selection-after-inference。`2.015` 只是沿用上一轮的 t5-style 描述性常数，不具有置信覆盖或单次实验安全证书的含义。六个 rig 也已经被打开，LORO 不能把它们重新变成 fresh test。

## 6. 相同在线物理预算对照告诉了什么

这里的“相同”只指新增 forward、adjoint 和重建调用均为 0。uniform 在线读取 `3024` 个 score value；主 LORO 六折的离线拟合读取 `15120` 个训练 score value，再读取 `3024` 个目标 score value，总计 `18144`。所以主候选拥有跨 rig 离线拟合成本，不能声称端到端同计算预算；本轮也没有测 wall time 与 peak memory。

- `target_local_slope` 在 1/2 档达到 `3.38%/12.76%`，但它直接从目标 rig 的候选 score surface 构造权重，只能算同数据探索对照。
- `target_local_curvature` 在 0.5/1/2 档为 `0.48%/4.05%/10.35%`，说明局部曲线形状可能携带信息，但没有独立 sentinel 时仍存在双重使用。
- `robust_median_uniform_gate` 与 uniform 决策完全等价，说明这一小数据集里准入门而非候选排序占主导。
- 主 LORO 权重提高了中、大失配功效，却没有触及小失配。这关闭了“继续在六个已打开 rig 上调相关系数或换一个小网络就能解决”的支线。

## 7. 下一算法不应是一团网络

下一版应拆成三条可单独证伪的支路：

| 支路 | 部署输入 | 经典基线 | 只有什么过门后才学 |
|---|---|---|---|
| `q_rel` measurement reliability | 独立 sentinel 帧的 whitened residual、噪声尺度、相机健康量 | inverse variance、Huber/MAD、consensus correlation | bounded monotone calibrator |
| `q_cal` geometry observability | geometry JVP/VJP、参数尺度、scaled `J^T J` spectrum | Gauss-Newton/LM damping、SVD truncation | PSD low-rank correction 或 bounded preconditioner |
| `q_field` marginal field information | view-conditioned `A/A^T`、normal-operator probe、候选视角集合 | covariance/greedy D-optimal 或 rank gain | permutation-invariant set operator |

三者的组合器只能输出有界非负权重、阻尼或停止建议；真正的几何修正仍由物理 solver 求解。选择相机的 `select` 帧、决定是否接受的 `sentinel` 帧、最终评价的 `target` 帧必须分开。若师兄的 callable 没有 geometry JVP/VJP，就先停在 `q_rel`，不要把它改名为 `q_cal`。

## 8. 真正可投稿的验证合同

这不是当前结果，而是下一阶段最低合同：

1. `20` 个开发 rig/session、`10` 个冻结 validation、至少 `30` 个 sealed audit；同一物理序列的帧不能跨 split。
2. 相机身份随机置换或显式 geometry token，防止模型只记住 camera 2。
3. 使用连续、off-grid、未知方向的 SE(3) 与内参扰动，不把失配方向、符号或幅度直接交给模型。
4. select/sentinel/target 三角色数据隔离；candidate selection 和 acceptance 不复用同一 measurement residual。
5. 同时报告 field relative-L2、gradient/front 指标、逐 rig 尾部、harm rate、Schur/Jacobian violation、`A/A^T` 与 JVP/VJP 调用、wall time、显存/内存。
6. 必须比较 reported geometry、oracle exact geometry、简单阻尼、Gauss-Newton/LM、uniform/robust weighting、NeRIF 或组内认可的强重建基线。
7. 只有 sealed audit 上保持收益、尾部和成本，才讨论神经组合器；不能用 development mean gain 宣称泛化。

## 9. 给何远哲师兄的最短问题

1. 现有 forward 是否能以相机内外参为显式输入？能否计算 geometry JVP/VJP？
2. straight-ray、curved-ray 和最终 image residual 各在哪一层可访问？
3. 真实数据里更常见的是固定标定偏差、跨帧漂移、焦距/畸变误差，还是位移提取噪声？
4. 多帧是否真的共享同一标定；独立 split 的最小单位是 rig、session、flame condition 还是日期？
5. 哪些相机/帧可以分别充当 select、sentinel 和 target，避免 residual 双重使用？
6. 组内认可的几何修正基线、主指标和可接受计算预算是什么？

可直接把回答记入 [真实接口回复工作台](../advisor_interface_intake.html)，再映射到数据合同；不要在没有回答时替实验室补默认值。

## 10. 一级来源怎样约束这条路线

- [NeRIF 原文](https://arxiv.org/html/2409.14722v2)：支持连续折射率/梯度场、标定 ray 和观测一致性；其实验使用 8 个视角重建、另 1 个视角验证。它不证明未知几何可辨识，也不授权本轮的可靠性权重。
- [Direct BOST with RBF](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100)：使用 ray tracing 的 reprojection displacement 检查重建，并展示 BOST forward 与一般 CT 的差异；reprojection consistency 不是体真值或几何可观测性证明。
- [Differentiable Uncalibrated Imaging](https://arxiv.org/html/2211.10525)：说明 measurement coordinates 不确定时可联合优化坐标和重建，且神经场/样条都可微；它验证的是 CT，不是 BOST 唯一性或真实 OERF 成功。
- [Geometry Calibration in Tomography with a Differentiable Ray-Based Model](https://arxiv.org/abs/2606.21405)：支持对 acquisition geometry 求高效梯度并联合校准；这是 2026 年预印本和通用 tomography 邻近证据，不是组内接口已经存在的证据。
- [Bengio & Grandvalet 2004](https://www.jmlr.org/papers/v5/grandvalet04a.html)：说明 K-fold CV 方差不存在普适无偏估计，重叠 fold 的朴素独立误差条会低估方差。
- [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)：说明有限样本上优化选择准则会导致 model-selection overfitting；因此同一 camera score 不能既选候选又被当作独立安全证据。

## 11. 复跑与核对

```bash
cd /path/to/oerf-bishe-dashboard
.venv/bin/python -m pytest -q learning_labs/test_calibration_camera_reliability_screen.py
.venv/bin/python learning_labs/calibration_camera_reliability_screen.py \
  --output-dir /tmp/calibration_camera_reliability_repeat
```

runner 把冻结输入固定为 `learning_labs/results/calibration_mismatch_v2/`，启动时先核对其 checksum，不接受运行时换输入。正式产物位于 `learning_labs/results/calibration_camera_reliability_v1/`。先看四联图，再核对 `report.json`、逐 case/rig 决策 CSV、权重 CSV 和 `checksums.sha256`。重复运行一致只证明同一代码与输入可重放，不证明科学外推。
