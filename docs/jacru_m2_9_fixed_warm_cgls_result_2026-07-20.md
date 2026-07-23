# JACRU-M2.9：固定 learned warm start 保留了场先验，但没有在同预算内恢复数据一致性

**正式状态：** `M2_9_FIXED_WARM_CGLS_DEVELOPMENT_NO_GO`

**证据等级：** 已打开 synthetic mechanism screen；不是 fresh、真实 BOST、跨 rig 泛化或论文性能证据。

**突破监测：** 无算法突破。

## 1. 这次到底问了什么

M2.9 不再训练新 selector，也不临场重训网络。四种架构、三个模型种子的 12 个 checkpoint 先保存成无 pickle NPZ，并在 reconstruction 评分前锁定 manifest、字节哈希、训练分区和运行时版本。评分时只加载这些固定权重。

问题只有一个：

> learned field 当作非零初值后，CGLS 能否保留它带入的离散 exact-kernel 分量，同时在与强 zero-start 基线相同的总 `A/A^T` 预算内修复可观测分量？

当前 learned proposal 的特征准备并不免费。它先运行 CGLS-12、terminal residual forward 和 grouped adjoint，总计 `13F/13A^T`。因此：

```text
k = 0: candidate = 13F / 13A^T
k > 0: candidate = (14+k)F / (13+k)A^T
```

每个 `k=0..10` 都比较：

- componentwise zero-start CGLS；
- 更强的 pair-rounded zero-start CGLS；
- pair-rounded Huber-PDHG；
- `k>0` 时另加完全相同 refinement 调用数的 affine-CGNE 机制控制。

development 只使用未参与 early stopping 的 6 cases / 3 geometry clusters。18 个 exploratory-OOD cases 不允许反选 `k`。

## 2. 正式结果

| 架构 | development 中 field gain 最大的 k | Field / H1 gain | Measured / clean reprojection ratio | 主要失败门 |
|---|---:|---:|---:|---|
| JACRU-M2 | 2 | +45.734% / +40.888% | 14.5965 / 3.2276 | measured、clean reprojection |
| pooled CNN | 2 | +44.045% / +39.427% | 14.4060 / 3.2206 | measured、clean reprojection |
| grid DeepONet | 2 | +2.143% / -11.971% | 6.8996 / 1.7343 | field、H1、reprojection、harm、worst |
| pooled FNO | 1 | +36.743% / +34.628% | 29.5586 / 7.2277 | reprojection、harm |

没有任何 architecture/checkpoint 同时通过全部 development 门，所以没有合法选择点，OOD 不得参与补救。

即使只看每种架构在 `k=0..10` 中最小的 measured-reprojection ratio，仍分别为：

```text
JACRU-M2 10.6432
pooled CNN 10.7093
grid DeepONet 4.3267
pooled FNO 13.9647
frozen gate <= 1.10
```

这不是擦线失败，而是数量级差距。

## 3. 为什么 field 很好，reprojection 却很差

每个 toy geometry 约有 150 个测量、1000 个 active voxels。这个离散 inverse 极度欠定。learned field 可以通过训练分布带入测量本身无法唯一决定的场先验，因此 field-L2 可能很好；但它的 row-space 部分仍可能与当前测量不一致。

Warm CGLS 的更新属于 `range(A^T)`：

```text
x_k = x_0 + q(A^T A) A^T (y - A x_0)
```

所以它可以逐步修正可观测部分，同时保留 `P_ker x_0`。正式审计确实看到：

- 最大 CGLS recurrence closure：`1.073e-13 <= 1e-12`；
- 最大 numerical exact-kernel drift：`1.867e-14 <= 1e-10`；
- Warm-CGLS breakdown：`0 / 3168`。

但“核分量被保留”不等于“核分量造成了 field 改善”，更不等于真实光学零空间。当前 SVD 只是 `150 x 1000`、support-restricted 的离散 toy evaluator。

真正的成本问题是：learned proposal 在开始 refinement 前已经花了 `13F/13A^T`。例如 `k=10` 时，候选总计 `24F/23A^T`，而 pair-rounded zero-start CGLS 可以把全部预算用于 24 步 data fitting。候选只剩 10 步修复一个明显不一致的 learned field，因而保留 field prior 的同时追不上基线的 measured residual。

## 4. CGLS 不是坏掉了

同一 learned field、同一 `F/A^T` refinement 次数下，warm CGLS 通常优于 affine-CGNE。以 development `k=6` 为例：

| 架构 | CGLS 相对同调用 CGNE 的 field gain | H1 gain | measured reprojection ratio |
|---|---:|---:|---:|
| JACRU-M2 | +3.398% | +3.713% | 0.5623 |
| pooled CNN | +3.297% | +3.683% | 0.5461 |
| pooled FNO | +3.146% | +2.868% | 0.5637 |

因此有限步 CGLS 多项式有小的机制优势，但远不足以抵消 13 对物理调用的 feature-preparation 成本。结论是当前 warm-start **接口和成本结构**失败，不是 CGLS 实现故障。

## 5. 下一主候选：Lean Geometry-Conditioned Warm Operator

下一步不应继续给当前 fixed warm start 扫更多 `k`。更有信息量的假设是：

> 用 raw displacement、相机/射线几何和最多一次 pooled adjoint 构造 warm field，把节省出的物理调用全部还给 CGLS；如果 field prior 真有价值，它应在相同总预算内同时保住 field 与 data consistency。

最小版本：

```text
input: y, camera/ray geometry, A^T y
learned proposal cost: 0F / 1A^T
fresh warm projection: 1F / 0A^T
refinement: CGLS-23
total: 24F / 24A^T
control: zero-start CGLS-24
```

这仍不是原创性结论。Learned warm starts、CG data consistency、null-space networks 都已有先例。潜在论文贡献只能来自：

1. BOST rig geometry 条件化，而不是输入一个固定稠密矩阵；
2. 按 `F/A^T` 和端到端时间严格计费；
3. unseen-rig / held-out-ray 的 fail-closed fallback；
4. repeated-frame BOST/TDBOST 中可摊销的 rig-specific near-null basis。

相邻一级来源：

- [NeRIF, Physics of Fluids 2025](https://doi.org/10.1063/5.0250899)
- [TDBOST, ACM TOG 2026](https://doi.org/10.1145/3809488)
- [Learning to Warm-Start Fixed-Point Optimization Algorithms, JMLR 2024](https://www.jmlr.org/papers/v25/23-1174.html)
- [MoDL, IEEE TMI 2019](https://doi.org/10.1109/TMI.2018.2865356)
- [Deep Null Space Learning, Inverse Problems 2019](https://doi.org/10.1088/1361-6420/aaf14a)

## 6. 三个下一步严格可证伪问题

### A. 低成本 warm proposal 能否闭合预算

只比较 `A^T y`、低成本 geometry-conditioned CNN/DeepONet/FNO 与 zero-start CGLS；总调用统一为 24 对。若 field/H1、measured/clean reprojection、harm 和 worst 任一过不了冻结门，关闭该接口。

### B. field gain 是否真的来自 kernel/near-null 分量

在 opened synthetic evaluator 上做 full warm、exact-null-erased warm、row-space-erased warm 和 zero-start 四臂因果消融。若 full warm 不稳定优于 null-erased warm，就不能把当前 field gain 归因于 kernel prior。

### C. fixed rig 的离线 basis 能否被多帧摊销

对同一相机 rig 用 randomized SVD/Lanczos 预计算低秩 row/near-null basis，在线只学习 bounded coefficients。必须同时报告一次性 setup、每帧成本和摊销 break-even 帧数；unseen rig 不能免费复用旧 basis。

## 7. 需要问何远哲师兄

1. 同一实验 rig 通常会连续采多少帧？离线算子分解能否跨帧复用？
2. 组内能否提供矩阵自由 `A/A^T`、相机标定、ray 与 mask，而不需要导出稠密矩阵？
3. 真正认可的 data-consistency 终点是 training-camera displacement、held-out camera/ray、independent renderer，还是 PIV 补偿误差？
4. flow-off repeats 和 optical-flow confidence 是否可用来估计噪声 floor，避免把拟合噪声写成物理一致性？
5. TDBOST 的同 rig 多时刻结构能否作为 amortized basis 的第一真实迁移目标？

## 8. 可复核工件

- 冻结协议：`docs/jacru_m2_9_fixed_warm_cgls_frozen_protocol_2026-07-20.md`
- 正式结果：`demo_t16_operator/results/jacru_m2_9_fixed_warm_cgls_postopen_public/`
- 独立验证：`demo_t16_operator/results/jacru_m2_9_fixed_warm_cgls_postopen_validation/validation.json`
- 独立 validator：`site_tools/validate_jacru_m2_9_fixed_warm_cgls.py`
- 评分 runner：`site_tools/run_jacru_m2_9_fixed_warm_cgls.py`

独立 validator 核验了 10/10 输出哈希、792 条 baseline、3168 条 candidate、2880 条 CGNE、264 个 seed-level aggregate 和 88 个 decision cells。最终 `errors=[]`，重算状态与 summary 一致。
