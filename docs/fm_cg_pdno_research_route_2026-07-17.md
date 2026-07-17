# 算子学习与三维重建下一算法路线：FM-CG-PDNO

日期：2026-07-17
状态：`CLOSED AFTER PREREGISTERED GATE B NO-GO / NO NEURAL SMOKE`

## 0. Gate B 事后判决，不回写原假设

V4 formal Gate B 已在 source commit `204bbe8` 上第一次完成 factor 轨迹。独立验证器
重算 4,048 项检查并通过，但 deterministic voxel-factor PDHG 相对 scalar PDHG 的
K=32 mean field reduction 只有 1.321%，相对 view-block 只有 1.242%，与 exact-K
graph-PCGLS 的 mean error gap 为 133.439%。八项预注册门通过 5 项、失败 3 项，正式
状态为 `GATE_B_E2_MECHANISM_NO_GO`。

因此本文定义的解锁条件没有满足：不实现 learned proximal smoke，不打开 fresh，
不把 FM-CG-PDNO 当作当前论文主算法。以下结构保留为被证伪研究合同，不能再解释为
待执行承诺。下一步转向 forward-model discrepancy 的 RayKernel-DCO，或在有真实
timestamp/缺帧序列时转 TRAIL-4D；完整判决见
[Gate B NO-GO](psu_b0_factor_pdhg_gate_b_no_go_2026-07-17.md)。

## 1. 为什么选这条主线

旧 frozen-FNO adapter 在当前开发域稳定输给 continued FNO，继续堆频域 adapter 缺少
证据。Gate A 刚证明的是一个更有价值的起点：BOST forward/adjoint、covariance
whitening、signed-factor majorizer 和正的 camera/site/voxel step 可以显式审计。

因此下一候选不是“黑箱网络直接吐三维体”，而是：

> **FM-CG-PDNO：Factor-Majorized Covariance-Geometry Primal-Dual Neural Operator**

它把已知物理迭代保留为主干，只让小网络学习确定性 solver 未表达的 bounded proximal
修正。关闭学习支路时，输出必须逐元素退化为 deterministic factor-PDHG。

## 2. 候选结构

已知因子保持显式：

```text
A_b = W_b P G_c E
D   = D_+ E
M_b = |W_b| P |G_c| E >= |A_b|
N   = |D_+| E
```

由 `M/N` 计算 voxel-wise `T` 与 camera/site-wise `Sigma`，展开 3、6 或 12 层：

```text
x_det = P_C[x_k - T(A^T p + D^T q)]

delta = beta * T^(1/2) * R_theta(
    x_det,
    T A^T p,
    T D^T q,
    |D x_det|,
    support,
    geometry/covariance embedding
)

x_(k+1) = P_C[x_det + delta]
```

第一版 `R_theta` 用共享权重的 3D depthwise-CNN，输出层零初始化，`tanh` 限幅，
参数控制在约 45k--55k。`beta=0`、移除 geometry、移除 covariance、scalar metric、
非共享层和 3/6/12 层都必须成为显式消融。

## 3. 三个真实物理问题，只能按数据条件选择

| 路线 | 真正要解决的失真 | 启动条件 | 失败即关闭 |
|---|---|---|---|
| H1 GQ-NIO | 少视角、有限角、BOS 梯度 gauge 与近零空间 | 能固定 `S/Q_fit/Q_audit` 相机和完整 ray/pose/mask | 同总相机数不能比直接 `S+Q_fit` 强基线改善 3%，或 audit 改善不超过 flow-off 不确定度 |
| H2 RayKernel-DCO | thin/cone ray 与有限孔径、景深、曲线光路、标定漂移的算子失配 | 至少两档光圈/焦平面，或 paired low/high-fidelity renderer 与 phantom | forward discrepancy 不超过噪声地板 2 倍，或输给显式 cone-ray/标定优化 |
| H3 TRAIL-4D | 异步、缺帧、曝光积分、火焰新生/熄灭与拓扑变化 | 50--200 连续帧、真实 timestamp/exposure/dropout 日志 | event p95、前缘时间偏差或换 cadence 后泛化失败 |

当前默认优先顺序不是固定的：有光学标定对就选 H2，有连续序列就选 H3，只有静态
多视角且能留 audit camera 才选 H1。三条不能同时大规模训练。

## 4. Mac 先证伪的冻结 benchmark

- 网格：`8 x 16 x 16`，固定物理域。
- 160 train、40 validation。
- 五个测试域各 32 个独立场：IID、family OOD、noise OOD、geometry OOD、joint OOD。
- 28 个 geometry：16 train、4 validation、4 OOD、4 stress，整组隔离。
- 场族：plume、Gaussian/flame、thin/double front、annular、oblique shock、vortex pair、multi-plume。
- 噪声：相机异方差、detector graph correlation、坏点和整机缺失；covariance 只由独立
  noise-only calibration frames 拟合。
- 连续场先独立分配角色，再采样网格；严禁把同一场的不同视角或滑窗拆进 train/test。
- 训练 renderer 与最终 cone/curved-ray renderer 不同，避免 inverse crime。

主基线为 DeepONet、FNO、可执行 FFNO、deterministic factor-PDHG 和 graph-PCGLS。
仓库当前还没有正式 FFNO runner；补齐并做参数、输入、loss、搜索时间匹配后，才能
声称比较 DeepONet/FNO/FFNO。

## 5. 指标和最低门

主指标是五个测试域等权的 paired field relative-L2 gain，同时报告：

- H1/gradient、front Chamfer、法向、位置、宽度、precision/recall；
- used-view 与 held-out-view whitened reprojection；
- mass、centroid、peak；
- p10、CVaR10、worst、`gain < -1%` harm rate；
- 参数、训练成本、推理 p50/p90、峰值内存；
- cold setup、warm solve、摊销时间和真实 `F/F^T` 调用数。

Mac development 只负责放行 GPU：相对锁定最佳基线需 mean gain `>=3%`、95% CI
下界 `>0`、p10 `>=0`、harm `<=5%`、全部困难域均值为正、held-out reprojection
不恶化且端到端时间不超过 3 倍。达不到就停止或改写物理假设，不能靠增加 seed 把
偶然结果包装成成功。

论文级结论还需 fresh joint-OOD mean `>=5%` 且 CI 下界 `>0`、两个未见分辨率、
六个未见 geometry block、独立 renderer、反应流 CFD 家族、五种子和真实 PSU/OERF
held-out camera/session。真实数据没有体真值时，只能声明 held-out measurement 改善。

## 6. 给何远哲的六个决策问题

1. 当前主要瓶颈是有限孔径/标定失配，还是 4D 新生、熄灭、异步和缺帧？
2. NeRIF/TDBOST 能否暴露 `F`、`F^T/J^T`、ray、mask、grid 和单位？
3. 是否保存 f-number、焦距、物距、焦平面、flow-off repeats？
4. 是否有 paired low/high-fidelity simulation 或 calibration phantom？
5. 能否固定一台不参与训练和选模的 audit camera？
6. 能否先提供一段带 timestamp、曝光和缺帧信息的 50--200 帧序列？

## 7. 原冻结执行顺序与实际停止点

1. Gate A 已完成，只解锁 Gate B，不自动解锁神经网络。
2. Gate B 的 16-sample、四方法、四 checkpoint 同调用比较已完成。
3. Gate B 正式 NO-GO，原第 4 步的 FM-CG-PDNO smoke 被阻断。
4. 不再为此分支补 360-field 训练或 FFNO 排名；先用 D0 定位 majorizer 松弛来源。
5. 根据师兄可提供的数据，只选择 H2 RayKernel-DCO 或 H3 TRAIL-4D；H1 降为第三顺位。

本文件现在是保留的失败研究合同，不是算法提案或结果胜出报告。任何新路线都必须
另写物理假设、数据角色、基线与停止门，不能沿用本文件阈值后继续调参。
