# JACRU-M2.3--M2.8 初学者决策指南

**给谁看：** 物理本科生、第一次接触逆问题和 Krylov 迭代的人  
**数据范围：** 已打开的 M2-T0 synthetic 数据，不是真实 BOST 实验数据  
**总判决：** M2.3--M2.8 都没有授权 fresh/final，也没有证明算法在真实 BOST 上成功。它们的价值是把问题逐层定位：先证明 learned field prior 有 headroom，再证明“怎样把它和光学观测相容”是困难，最后发现困难主要已经从“求解器够不够快”转成“目标是否在追噪声、模型误差和尾部风险”。

## 0. 先把三种证据分开

### 算法

算法只能使用真实运行时会有的东西：观测 `y`、forward `A`、adjoint `A^T`、几何和预先冻结的参数。它不能读取 truth、field error、family label 或 test mask。若某个候选用到了 truth，它只能是 evaluator，不是算法。

### oracle

oracle 是为了回答“理论上有没有希望”。本轮 exact dense SVD / exact camera-block 都把小 toy 算子 `A` 组装成 dense 矩阵，所以可以看到一个普通算法暂时看不到的上界。它们排除了 setup 成本，不能进入 runtime 或效率排名。

### 真实证据

真实证据至少还需要实验室的 BOST 数据合同、噪声和相机标定、独立 forward/renderer、跨 rig 或跨实验条件测试，以及预注册的 fresh split。当前结果包是 **E1 opened synthetic evidence**：可以指导研究，不可以写成“真实流场泛化成功”。

## 1. 统一数学图景

把三维折射率场或密度场写成向量 `x`，把一组相机观测写成 `y`：

```text
A : R^n -> R^m       forward measurement operator
A^T                 与 A 配对的 Euclidean adjoint
x_net              神经网络给出的 learned proposal
x_ref              无 truth 的参考重建
```

网络擅长补充光学观测没有直接提供的三维先验，但它可能把错误的可观测分量也放进去。于是定义

```text
delta = x_net - x_ref
```

并尝试把 `delta` 的一部分删掉，使输出既保留三维先验，又不过度违背观测。理想的有限矩阵 null-space 投影是

```text
P_row = A^T (A A^T)^dagger A
P_ker = I - P_row
x_out = x_ref + P_ker delta
```

它满足 `A x_out = A x_ref`。注意：这只是**冻结离散矩阵的**观测不变，不等于真实有限孔径光学中的不可见性。

## 2. M2.3：先试 matrix-free 投影，为什么失败

### 做了什么

M2.3 不再把 `A` 存成 dense 矩阵，而是在 measurement space 解

```text
b = A (x_net - x_ref)
(A A^T + lambda I) z = b
x_k = x_ref + (x_net - x_ref) - A^T z_k
```

每个 Krylov 步大致需要一次 `A` 和一次 `A^T`，所以可以在大规模问题上运行。它是对 M2.2 exact null-space oracle 的有限调用近似。

### 真实数字与 NO-GO

在 JACRU development 上，damped `k=4` 的 field gain 约 `45.88%`，但相对同预算 CGLS 的 reprojection ratio 仍为 `14.79x`；`k=12` 时 ratio 约 `32.55x`，并开始出现 `8.33%` harm。两种 backbone 都没有 development-eligible candidate。

根本原因不是一句“CG 太慢”就能概括：这个版本固定的是 `A x_ref`，而 `x_ref` 是准备好的 CGLS-12。红队推导的 anchor ratio 在不同预算下约从 `1.599x` 增至 `25.232x`。如果目标要求保持 `A x_ref`，任何精确 null correction 都不能把它变成更好的 matched-CGLS 观测残差。继续加预条件器也不能改变这个收敛极限。

**本地证据：** [M2.3 形式化红队](jacru_m2_3_formal_red_team_2026-07-17.md)、[M2.3 结果包](../demo_t16_operator/results/jacru_m2_3_matrix_free_projection_postopen_public/README.md)。

## 3. M2.4：改成真正的 affine observation target

### 为什么要改目标

M2.3 的目标把网络校正拴在 `x_ref` 上。M2.4 直接让网络 proposal 朝观测 `y` 投影：

```text
b = A x_net - y
(A A^T + lambda I) z = b
x_k = x_net - A^T z_k
```

当 `lambda=0` 且系统完全解完时，它对应 `A x = y` 的仿射投影。这个目标在数学上更适合“把 learned 先验和观测结合起来”，所以 M2.4 是必要的修正，不是为了掩盖 M2.3。

### 结果

JACRU development 的代表数字如下：

| 投影步数 `k` | field gain | reprojection / matched CGLS | harm rate |
|---:|---:|---:|---:|
| 1 | `46.73%` | `19.99x` | `0%` |
| 2 | `47.34%` | `16.40x` | `0%` |
| 8 | `41.36%` | `20.89x` | `8.33%` |
| 12 | `39.54%` | `19.66x` | `8.33%` |
| 32 | `35.22%` | `33.38x` | `8.33%` |

exact affine dense oracle 的 residual 可以到约 `6.0e-16`，说明目标在当前 toy 中可达；但有限步 identity CG 在 learned preparation 已花掉 `13F/13Adj` 后仍远远落后于 matched classical baseline。因此这里是“目标修正正确，但有限预算未过门”，不是“神经网络没有用”。

**本地证据：** [M2.4 配置](../demo_t16_operator/configs/jacru_m2_4_affine_observation_projection_postopen_v1.json)、[M2.4 结果包](../demo_t16_operator/results/jacru_m2_4_affine_observation_projection_postopen_public/README.md)、[预条件器红队中的 M2.4 表格](jacru_m2_5_preconditioner_design_red_team_2026-07-17.md)。

## 4. M2.5：exact Jacobi 预条件器

### 它是什么

对 `H = A A^T + lambda I`，Jacobi 只取

```text
D = diag(H)
M = D
```

它相当于单独缩放每个 measurement coordinate，不处理不同相机射线之间的相关性。PCG 用 `M^-1` 帮助 Krylov 迭代更快，但不改变 `H z=b` 的解，也不改变 affine target。

### 结果和决定

JACRU 在 `k=8` 时，identity reprojection ratio 约 `20.89x`，exact-Jacobi 约 `17.66x`；`k=12` 约 `15.19x`，但 `k=12` 已不是 24-call 主预算点，而且 harm 仍为 `8.33%`。所以它有小的 RHS-specific 加速，却没有接近 `1.10x` 的门。

这个 oracle 的 diagonal 来自 dense active matrix；当前路径明确报告约 `1001 F-equivalent` setup，并排除在算法和效率主张之外。结论是 **NO-GO，停止继续调 Hutchinson diagonal、floor 或 damping 来挽救它**。

**本地证据：** [M2.5 结果包](../demo_t16_operator/results/jacru_m2_5_exact_jacobi_preconditioner_oracle_postopen_public/README.md)、[M2.5/M2.6 预条件器红队](jacru_m2_5_preconditioner_design_red_team_2026-07-17.md)。

## 5. M2.6：camera-block oracle 找到机制，但仍未过重建门

### 为什么分块

当前 toy 的观测形状是 `[75, 2]`，对应 3 个 camera blocks，每块 50 个 measurement coordinates。camera-block 预条件器保留块内耦合：对每个 camera block 解一个小的 dense SPD 子系统，再把各块拼起来。

### 结果

在 JACRU development 上，exact camera-block 的 `k=8`：

```text
reprojection ratio = 1.263x
field gain          = 40.14%
harm rate           = 8.33%
worst field gain    = -8.75%
```

`k=12` 可以到约 `0.270x`，但需要 `26F/25Adj`，超过 24-call cap；即使残差很小，harm 仍为 `8.33%`，worst gain 约 `-9.31%`。pooled CNN 的趋势相同。这个结果很有价值：它证明 camera-local correlation 是真实的加速机制，但也证明“更快逼近 `Ax=y`”不自动等于“更好的三维场”。

**判决：** `M2_6_CAMERA_BLOCK_PRECONDITIONER_ORACLE_NO_GO`。它是 mechanism headroom，不是可部署算法。

**本地证据：** [M2.6 配置](../demo_t16_operator/configs/jacru_m2_6_camera_block_preconditioner_oracle_postopen_v1.json)、[M2.6 结果包](../demo_t16_operator/results/jacru_m2_6_camera_block_preconditioner_oracle_postopen_public/README.md)、[独立 M2.5/M2.6 验证器](../site_tools/validate_jacru_m2_5_m2_6_evidence.py)。

## 6. M2.7：把预算内所有 `k=0..10` 都扫一遍

### 为什么要做 ceiling audit

不能只挑一个漂亮的 `k=8` 或 `k=12`。M2.7 在 exact camera-block oracle 下枚举所有 `k=0` 到 `10`，因为 learned preparation 的 13 次调用使 `k<=10` 才能留在 24-call cap 内。

### 结果

`k=9/10` 的平均 reprojection 已低于 `1.10x`：development 上，JACRU / pooled CNN 在 `k=9` 约为 `0.852x / 0.914x`，在 `k=10` 约为 `0.623x / 0.668x`。但是两者仍有同一个 development 单界面样本受损：`k=9` 的 harm rate 为 `8.33%`，最差 field gain 为 JACRU `-8.89%`、CNN `-11.89%`；`k=10` 也没有解除该尾部。

**这一步的学习点：** 平均数过门不代表方法可靠。对反应流和 BOST，单个 rig、遮挡、噪声或火焰界面失败都可能决定方法是否能交给师兄使用。

**本地证据：** [M2.7 配置](../demo_t16_operator/configs/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_v1.json)、[M2.7 结果包](../demo_t16_operator/results/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public/README.md)。

## 7. M2.8：简单插值和 truth-oracle calibration ceiling

### 做了什么

在 `k=9/10` 的 camera-block 输出和原网络输出之间做固定插值：

```text
x(alpha) = x_net - alpha (x_net - x_pcg)
```

`alpha=0` 是网络，`alpha=1` 是 PCG。固定 alpha 不需要额外 operator call，所以它是一个很便宜的控制实验；同时 evaluator 还允许每个样本读取 truth，寻找满足观测门的最佳 alpha，作为**不可部署上界**。

### 结果与边界

固定 alpha 形成明显 trade-off：以 JACRU、`k=10` 为例，`alpha=0.5` 的平均 reprojection 约 `197.7x`，`alpha=0.99` 约 `4.03x`，`alpha=1` 才约 `0.623x`；但 `alpha=1` 的最差 field gain 约 `-7.40%`，harm rate `8.33%`。也就是说，插值只能在两个失败倾向之间移动，不能凭空创造一个同时满足全部门的方案。

truth-oracle alpha 只回答“每个样本如果知道 truth，最多能调到哪里”。K=9 的 development 可行率只有 JACRU `83.33%`、CNN `69.44%`；K=10 两者也只有 `97.22%`。对问题界面样本，满足逐样本重投影门的 truth-optimal alpha 约为 `0.988–0.992`，六个模型种子的 field gain 仍全部为负。它不能用于训练、选择、部署或论文方法排名；如果一个方法需要它才能过门，就说明还没有 observable calibration。M2.8 最终状态是 `M2_8_INTERPOLATION_CALIBRATION_ENVELOPE_NO_GO`。

**本地证据：** [M2.8 配置](../demo_t16_operator/configs/jacru_m2_8_interpolation_calibration_ceiling_postopen_v1.json)、[M2.8 结果包](../demo_t16_operator/results/jacru_m2_8_interpolation_calibration_ceiling_postopen_public/README.md)。

## 8. 为什么现在停止堆预条件器

1. **预条件器改变路径，不改变目标。** 对同一个 affine system，identity、Jacobi 和 camera-block 只改变到达 `Ax=y` 的速度。
2. **M2.3 的 anchor 先天不合适。** 如果仍要求 `A x_out=A x_ref`，强预条件器也不能修复 CGLS-12 anchor 与 matched CGLS 的差距。
3. **M2.6/M2.7 已经显示“残差快”和“场不受损”分离。** `k=9/10` 平均 reprojection 过门仍有 `8.33%` harm。
4. **oracle setup 不能伪装成部署成本。** dense camera-block 是为了找机制，不是实验室机器上免费存在的矩阵。
5. **噪声下精确追 `y` 可能是坏事。** BOST 的观测包含相机噪声、背景纹理误差、标定误差和 forward mismatch。更强 solver 可能只是更快地拟合这些误差。

所以下一步不是“再换一个更复杂的 preconditioner”，而是研究 **noise-aware target + observable fail-closed**。

## 9. 下一步三个最小候选

这三项只应在新的数据合同、噪声模型和预注册 gate 到位后开始。它们是最小研究候选，不是已经成功的方法。

### N1：按噪声尺度的 damped affine correction

如果有测得的 noise covariance `Sigma_y`，定义 correction `d`：

```text
d_lambda = argmin_d 0.5 || Sigma_y^(-1/2) [A(x_net-d)-y] ||_2^2
                         + 0.5 lambda ||d||_2^2
x_out = x_net - d_lambda
```

`lambda` 不用 truth 选，而按预先冻结的噪声尺度、标定数据和 geometry descriptor 归一化。目标不再强迫 `Ax=y`，而是允许 residual 落在“噪声可解释”的范围内。必须同时画 field L2、H1、measured reprojection、independent clean reprojection、逐 rig worst case 和 harm rate。

### N2：discrepancy principle 的固定步数版本

不要把观测 residual 越小越好。若噪声独立且已白化，可用

```text
chi2(x) = || Sigma_y^(-1/2) (A x-y) ||_2^2
```

预先在 calibration noise 上确定允许区间，例如围绕自由度 `m` 的置信区间。对固定候选 `k, lambda`，只依据 `chi2` 是否进入区间选择或停止；超出区间就 fallback。选择规则必须只在 development calibration 上冻结，不能看 fresh truth。

### N3：observable fail-closed calibration envelope

给每个输入记录不含 truth 的 observable descriptor：camera count、有效 mask 比例、相机姿态范围、`A/A^T` adjoint closure、白化 residual、correction norm、projection closure 和调用账本。若候选落在 calibration envelope 外，或者

```text
chi2(candidate) 不在噪声区间
closure 超门
预算超门
几何描述符超训练范围
```

则返回预先指定的 baseline，并记录 `reject_reason`。这不能证明 field 一定无害，但可以把“算法不知道自己不可靠”的情况变成可审计的拒绝。必须报告 coverage、reject rate、accepted-case harm 和 fallback baseline，不能只报告 accepted mean。

## 10. 你现在应该做的三个学习练习

### 练习 A：二维小矩阵手算投影

取一个 `2 x 3` 矩阵 `A`，手算 `A A^T`，比较

```text
P_row = A^T (A A^T)^(-1) A
P_ker = I-P_row
```

验证 `A P_ker delta=0`，再给 system 加 `lambda I`，观察 `A delta_lambda` 为什么不再为零。这个练习对应 M2.2/M2.3 的核心数学。

### 练习 B：只用 callable A/A^T 写 PCG

禁止在 solver 里传 truth。记录每次 `A`、`A^T`、preconditioner 的调用次数，验证

```text
Ax_k - y = r_k + lambda z_k
```

并画 `system residual` 与真实 `reprojection residual` 两条曲线。它会直观看出“小 system residual”不等于“正确的物理观测”。

### 练习 C：加可控噪声再做 N1--N3

先用已打开 toy 生成多组已知 `Sigma_y` 的观测噪声，固定 seed 和预算；画四联图：field relative-L2、H1、白化 `chi2`、逐 case harm。最后故意放入 camera pose/mask OOD，检查 fail-closed 是否拒绝，而不是只看平均值。

## 11. 给师兄汇报时的一句话版本

> 我们已经证明 learned residual 在 toy 的 numerical null space 中有明显 field headroom，但 M2.3 的旧 anchor 不可行；M2.4 改成 affine observation target 后，camera-block oracle 能把有限步 reprojection 明显加速，却在 24-call 和逐样本 no-harm 门上失败。M2.5--M2.8 因此停止继续堆预条件器，下一步应拿到真实 BOST 的噪声/标定合同，研究 noise-aware target 与 observable fail-closed，而不是把 synthetic oracle 结果写成真实算法成功。

## 12. 当前可引用的本地入口

- [M2.2 exact null-space headroom](jacru_m2_2_exact_nullspace_headroom_2026-07-17.md)
- [M2.3 formal red-team](jacru_m2_3_formal_red_team_2026-07-17.md)
- [M2.5 preconditioner red-team](jacru_m2_5_preconditioner_design_red_team_2026-07-17.md)
- [M2.3--M2.4 独立验证器](../site_tools/validate_jacru_m2_3_m2_4_evidence.py)
- [M2.5--M2.6 独立验证器](../site_tools/validate_jacru_m2_5_m2_6_evidence.py)
- [M2.7 target/no-harm ceiling 结果](../demo_t16_operator/results/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public/summary.json)
- [M2.8 calibration ceiling 结果](../demo_t16_operator/results/jacru_m2_8_interpolation_calibration_ceiling_postopen_public/summary.json)

**最终边界：** 本指南没有把任何 M2.3--M2.8 结果写成论文成功、真实 BOST 成功或泛化成功。它的用途是帮助你理解已经证伪了什么、还剩什么值得做，以及下一次拿到实验室数据后应该先问哪些物理问题。
