# 给何远哲师兄的一页审核材料：N1.5 高阶教师到伴随一致误差学习

## 我现在真正想解决的问题

不是直接用 FNO/DeepONet 猜三维场，而是处理三维 BOST 中“廉价前向模型与更真实前向物理不一致”
造成的系统误差，并保证修正真的改善三维逆解，而不只是改善 measurement residual。

## 已完成、可复核的最小结果

- 连续解析梯度 renderer 与体素 FD/三线性算子之间的 synthetic mismatch 约为观测范数的显著比例；
- 可见曲率 ridge 能把 mismatch L2 压得更低，但进入重建后输给简单阻尼；
- 直接用四阶算子 CGLS 会恶化场误差；
- 用四阶算子只估计误差、仍让稳定二阶算子求解，表现稳定；
- 候选 `damping + 0.75 * (high-order - damping)` 在 Git 预冻结后打开 6 个新几何簇：
  field `+3.632%`、H1 `+10.308%`、最坏场 `+0.898%`、相对 damping 再 `+0.686%`；
- 由于预设 field 门为 `+5%`，正式判决是 **NO-GO**，没有把它写成算法胜利。

## 我对失败原因的判断

当前 loss 学的是完整 measurement mismatch `epsilon`，但逆问题主要通过 `A^T epsilon` 感受它。
一些 measurement-space 改善落在零空间或低敏感方向，对三维场帮助很小。确认集上 exact mismatch
oracle 的 field gain 也只有 `+5.668%`，说明当前夹具的可兑现上界并不宽。

## 希望师兄审核的下一候选

```text
g = A^T (G_H - G_L)
g_hat = U c_theta(observation, geometry, warm field)
```

- `U` 在训练 split 固定，是低秩、可审计的场空间；
- 小网络只预测有界系数 `c_theta`，不直接输出三维场；
- 本轮高阶教师固定为 strong control；
- 任一几何风险门失败就回退到 damping/low-order baseline；
- 先做 synthetic mechanism screen，再谈真实 BOST，不先对 DeepONet/FNO 宣称胜出。

## 需要师兄确认的数据/接口

1. 当前实验代码能否调用匹配的 `A` 与 `A^T`？
2. 是否有比当前 thin/straight-ray 更高保真的离线 forward，可成对生成 `G_L/G_H`？
3. 可提供哪些原始 reference/distorted images、flow-off、标定和 f-number 元数据？
4. 能否留一台 camera 或完整 session 作永久 audit？
5. 本科毕设是否接受“高阶教师基线 + 伴随一致小模型 + 严格负结果”作为完整交付？

## 请师兄重点否决或确认

- 真实实验中的主要 forward mismatch 是否确实来自 finite aperture / ray bending / calibration，而不是
  当前 synthetic discretization；
- `A^T epsilon` 是否符合组内现有代码和物理解释；
- 下一步应先做 cone-ray paired renderer，还是先在已有 BOST 数据上做 held-out camera likelihood。
