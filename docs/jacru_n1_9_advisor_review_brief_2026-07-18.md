# 给何远哲师兄：N1.9 判决与下一步数据合同

日期：2026-07-18

## 先看结论

我没有训练新网络，而是先比较两个同预算 rank-6 表示是否有足够上限：

```text
Residual-Contrast: orth(d,r,C1r,C2r,Kd,Kr)
Damping-Contrast:  orth(d,r,C1d,C2d,Kd,Kr)
```

两者都只用部署可见量构造 basis，均为 `25F/24A^T`。设计和阈值先提交为 `52490e5`，再运行
同一已打开的 6 个 synthetic geometry、12 个 paired fields。正式状态是：

`N1_9_RANK6_CAMERA_GLOBAL_K_BRANCH_CLOSED`

## 最关键数字

| 表示 | field vs CGLS-24 | H1 | extra-headroom | exact retention | P A^T gain | 门 |
|---|---:|---:|---:|---:|---:|---:|
| Residual-Contrast | +6.207% | +10.672% | 51.408% | 72.917% | 28.112% | 16/17 |
| Damping-Contrast | +5.452% | +8.768% | 36.864% | 64.042% | 35.787% | 15/17 |

Residual 只差预冻结的 `extra-headroom >=60%`；Damping 还未达到 `exact retention >=70%`。
“差一点”不能代替过门，所以两者都不带入新 split，也不接 DeepONet/FNO/MLP learner。

## 我认为最有物理意义的观察

- Residual 在 12/12 case 的 H1 更低，在 6/6 single-interface case 的 field 更低；
- Damping 在 6/6 smooth case 的 field 略低，并在 8/12 case 的 data residual 更低；
- Damping 的 support-adjoint gain 更高，但仍只有 35.787%，不到 50% 机制线。

当前最合理的解释是：residual contrast 更偏界面恢复，damping contrast 更偏 measurement-space
一致性，两个目标没有被同一个低秩表示统一。这只是 opened synthetic 机制信号，不是算法结论。

## 希望师兄帮我确认的六件事

1. 组内真实三维 BOST/TDBOST pipeline 能否给 camera pose、像素/ray 坐标、mask/confidence 和标定版本？
2. 当前最主要的 mismatch 是 finite aperture、ray bending、标定漂移、位移提取，还是离散算子？
3. 是否有可调用的 `A/A^T`、JVP/VJP 或 held-out camera reprojection 接口？
4. 能否永久留出一台 camera、一个 session 或一个工况，等模型与阈值冻结后再打开？
5. 论文主指标应优先 field/H1、界面位置、held-out image、PIV compensation，还是时序稳定性？
6. 能否给一个 NeRIF/TDBOST 典型失败 case，并标注它更像标定、数据、优化或表示问题？

## 若数据合同允许，我建议的新课题表述

不是“再做一个 FNO”，而是：

> 面向多相机 BOST 的 measurement-consistent、interface-aware response representation：仅从 pose、
> ray/pixel、mask/confidence、warm residual 与有限 `A/A^T` probes 生成有界校正；在按
> geometry/session/camera 留出的 holdout 上，同时验证界面恢复、held-out reprojection、尾部伤害和成本。

先做数据适配、伴随测试和经典/NeRIF/TDBOST 基线；只有固定表示在新 split 上有 headroom，才训练
coefficient/basis generator。当前 N1.9 不支持直接启动 learner。

## 审核时请重点判断

- “界面恢复 vs measurement consistency”的分叉是否对应组内真实失败模式；
- 真实 forward 是否足以暴露上述部署可见量；
- 哪个 camera/session 可以做永久 holdout；
- 这个问题是否比继续做 synthetic rank-6 basis 更贴近近期项目需求。

完整证据：[N1.9 分支关闭报告](jacru_n1_9_global_contrast_branch_closed_2026-07-18.md)；
[机器摘要](../demo_t16_operator/results/jacru_n1_9_global_contrast_postopen_full1/summary.json)。
