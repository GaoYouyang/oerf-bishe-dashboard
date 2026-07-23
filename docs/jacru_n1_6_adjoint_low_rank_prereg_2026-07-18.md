# N1.6 预注册：伴随加权的低秩模型误差修正

日期：2026-07-18

状态：`POSTOPEN_DEVELOPMENT_HYPOTHESIS_GENERATION_ONLY`

证据边界：只允许在已经打开过的 synthetic development 上生成一个未来确认假设。

## 1. 为什么不是继续预测完整 measurement mismatch

合成观测写成：

\[
y = G_H(x^\star) = A_L x^\star + \epsilon^\star,
\qquad
\epsilon^\star = G_H(x^\star)-A_Lx^\star.
\]

低阶 CGLS 的正规方程只通过下式感受模型误差：

\[
A_L^T\epsilon^\star.
\]

N1.5-A 已经显示：降低
\(\|\hat\epsilon-\epsilon^\star\|_2\) 并不保证改善三维场。
N1.6 因此不再把 measurement-L2 当成主要训练几何，而是在训练阶段为每个
fit case 求解：

\[
c_i^\star
=
\arg\min_c
\left\|
A_{L,i}^T\left(\mu + Bc-r_i\right)
\right\|_2^2
+\eta\|c\|_2^2,
\]

其中：

- \(r_i=\epsilon_i^\star-\epsilon_{\mathrm{damping},i}\)；
- \(\mu,B\) 只由 fit split 的 residual mismatch 构造；
- \(B\) 位于 measurement space；
- fresh 几何上的修正仍由当前真实低阶伴随 \(A_L^T\) 进入重建；
- 不训练一个脱离当前算子的独立“神经伴随”。

## 2. 可部署候选

小型 ridge 只预测低秩系数：

\[
\hat c = f_\theta(z),
\qquad
\hat\epsilon
=
\epsilon_{\mathrm{damping}}
+s(\mu+B\hat c).
\]

部署特征 \(z\) 只允许来自：

1. 相机几何和 detector 元数据；
2. `case.inference.observations_uv`；
3. 低阶 CGLS-12 暖启动场；
4. 暖启动的低阶 forward projection；
5. 由上述可见量计算的全局、逐相机和场粗糙度统计。

部署函数不得接收整个 `CaseRecord`。以下量禁止进入预测器：

- truth volume；
- fresh exact mismatch；
- phantom family label；
- confirmation target；
- high-order forward output；
- 最终 field/H1 分数。

## 3. fail-closed 结构

候选同时执行三层约束：

1. 每个系数裁剪到 fit coefficient envelope；
2. 特征标准分数超出 fit envelope 时回退；
3. 合成 residual RMS 超出 fit envelope 时回退。

回退方法固定为 `component_damping`。必须同时报告：

- takeover rate；
- fallback rate；
- takeover 后增益；
- 混合回退后的总体增益。

当前模型没有协方差头，因此 Schur/PSD 门为 `not applicable`，不得填成零违规。

## 4. 固定物理调用预算

所有可部署生产路径统一为：

```text
CGLS-12 warm start                 12F + 12A^T
visible warm projection             1F +  0A^T
corrected warm-start CGLS-12       12F + 12A^T
------------------------------------------------
total                              25F + 24A^T
```

匹配低阶基线固定为：

```text
CGLS-24 + one registered projection = 25F + 24A^T
```

所有模型推理、特征构造、high-order control 和 evaluator oracle 时间单独记账。
真值伴随诊断写入独立 evaluator CSV，不得混入 deployment case rows。

## 5. 数据拆分和独立单位

- fit：12 个 geometry clusters，每个 cluster 两个 paired families；
- calibration：4 个 geometry clusters；
- opened development：6 个 geometry clusters；
- 有效独立单位是 `base_seed_geometry_cluster`，不是 case、ray 或 pixel；
- 每个 cluster 必须恰好包含 `smooth_no_interface` 和 `single_interface`；
- fit/calibration/development 的 seed 和 geometry digest 必须两两不交叠；
- N1.6 development 不得复用 N1.5 frozen confirmation 的 development seeds 或 digest。

## 6. 预注册网格

- feature set：`summary`、`camera`、`camera_geometry`；
- basis rank：1、2、4、8；
- adjoint target L2：`1e-8`、`1e-4`、`1e-2`；
- ridge alpha：`0.01`、`1`、`100`；
- residual shrinkage：`0.25`、`0.5`、`0.75`、`1`。

PCA basis 和 coefficient target 只使用 fit。候选选择只使用 calibration。
development 只评估 calibration 已选定的一个候选。

## 7. 必须同时出现的对照

| 路线 | 是否可部署 | 作用 |
|---|---:|---|
| matched low CGLS-24 | 是 | `25F/24A^T` 主基线 |
| component damping | 是 | 最简单统计修正 |
| damping + train basis mean | 是 | 不使用 case 特征的低秩控制 |
| selected fail-closed ridge | 是 | N1.6 主候选 |
| selected raw unbounded | 否，仅诊断 | 测量安全门的代价 |
| high-order teacher interpolation | 否，教师控制 | 每 case 需要 1 次 high-order F |
| measurement-L2 coefficient oracle | 否 | 检查普通 PCA 系数上限 |
| adjoint-weighted coefficient oracle | 否 | 检查低秩可兑现上限 |
| exact mismatch oracle | 否 | 完整 evaluator ceiling |

## 8. calibration 选择门

候选首先必须满足：

- takeover rate 不低于 75%；
- mean field gain 相对 damping 不低于 0；
- worst-case field gain 相对 damping 不低于 -1%；
- 相对 damping 的大于 1% harm rate 为 0。

主排序量是相对 damping 的 mean field gain。真值派生的伴随 residual gain
只能作为明确标注的 calibration evaluator tie-breaker。

## 9. 进入一次冻结确认前必须同时通过

- mean field gain 相对 matched low CGLS-24 不低于 5%；
- mean H1 gain 不低于 3%；
- mean field gain 相对 component damping 不低于 0.5%；
- 不劣于 N1.5 的 high-order teacher `beta=0.75` 控制；
- worst-case field gain 相对 low 不低于 -1%；
- 相对 low 和相对 damping 的大于 1% harm rate 都为 0；
- takeover rate 不低于 75%；
- evaluator adjoint-residual gain 相对 damping 不低于 10%；
- 部署 high-order forward/adjoint 调用均为 0。

任一门失败，状态固定为：

```text
POSTOPEN_NO_GO_NO_CONFIRMATION_ROUTE
```

## 10. 结果解释边界

即使全部通过，也只能说明：

> 在连续解析梯度 renderer 与 voxel FD/三线性低阶算子之间的 synthetic
> representation mismatch 上，存在一个值得进行独立冻结确认的候选。

它仍不能支持：

- 真实 BOST 成功；
- finite-aperture、ray bending、optical flow 或 calibration drift 已解决；
- OOD 或跨实验 session 泛化；
- 已优于 FNO、DeepONet、NeRIF 或课题组真实重建管线；
- 已建立论文新颖性。

本轮首先回答“伴随加权低秩参数化是否有可兑现 headroom”，而不是直接训练大网络。
