# LGWO-A24：低调用数几何条件方向的结果前候选协议

**候选状态：** `O1_OPENED_EXACT_NULL_HEADROOM_OBSERVED_NO_MODEL_AUTHORIZATION`

**当前证据：** 算法安全壳、调用账本与小矩阵单元测试已完成；O1 在 6 个已打开 synthetic cases 上观察并独立复算了 exact-null truth direction 的表示余量。尚未训练模型，没有 OERF 数据，也没有 OOD/fresh/generalization/novelty 结论。

**突破监测：** 无突破。

## 0. O1 结果更新：存在表示余量，不等于模型会学

运行前冻结的 O1 用 dense SVD 评估器把真值分成 exact row/null-space 两部分，并在同一 `24F/24A^T` 壳层内比较 exact-null、exact-row 和 full-truth 三个方向。6 cases / 3 geometry clusters 的结果为：

| arm | eta | field / H1 mean gain | measured / clean ratio | worst field gain | gate |
|---|---:|---:|---:|---:|---|
| exact-null truth | 0.05 | +6.978% / +6.566% | 1.000000 / 1.000000 | +3.741% | descriptive thresholds reached |
| exact-null truth | 0.10 | +13.722% / +13.005% | 1.000000 / 1.000000 | +7.262% | descriptive thresholds reached |
| exact-row truth | 0.05 | -0.011% / -0.015% | 0.979014 / 0.998891 | -0.031% | field/H1 fail |
| exact-row truth | 0.10 | -0.021% / -0.028% | 0.959528 / 0.997878 | -0.060% | field/H1 fail |

未导入正式 runner 的 validator 重建全部 case/projector 并重跑 36 行，以 1,121 项断言复现结果。它仍共享 fixture、projector、solver 和 metric 依赖，只是确定性复算，不是独立物理验证。配置冻结前已经查看过一个 case 的三个 eta；全部 6 cases 也都曾用于更早的 M2.9 路线，因此 O1 只能支持 opened representation-headroom 描述，不能选 eta、训练路线或论文结论。完整结果见 `docs/lgwo_a24_o1_exact_direction_result_2026-07-20.md`。

## 1. 为什么 M2.9 之后不能直接做 warm-CGLS-23

M2.9 的 fixed learned warm start 先花了 `13F/13A^T` 造 proposal。它保住了明显的 field prior，却只剩少量调用修复数据一致性，最终全部 development 路线被 reprojection 门否掉。

最初的下一步草案是：

```text
A^T y + learned warm field + restarted CGLS-23
```

这里有一个必须先修正的数学问题：即使 learned correction 为零，从 `A^T y` 形成一个非零初值后再“重启” CGLS，也不严格等于 zero-start CGLS-24，因为重启丢失了原 Krylov 递推状态。这样一来，网络路线与基线之间会混入 solver-shell 差异，不能把任何收益归给模型。

LGWO-A24 改成只扰动 **第一条搜索方向**，后续在测量空间显式构造共轭方向。零扰动时，它退化为全重正交形式的 CGLS-24，而不是另一个重启算法。

## 2. 算法定义

记测量为 `y`，support projector 为 `P_S`：

```text
g = P_S A^T y
delta_theta = bounded_network(y, g, geometry, support)
z = g + delta_theta
u_0 = A z
alpha_0 = <y, u_0> / <u_0, u_0>
x_1 = alpha_0 z
r_1 = y - alpha_0 u_0
```

对后续 `j=1..23`：

```text
s_j = P_S A^T r_j
v_j = A s_j

对全部历史 (p_i, u_i=A p_i) 做两遍 modified Gram-Schmidt：
    beta_ji = <v_j, u_i> / <u_i, u_i>
    s_j <- s_j - beta_ji p_i
    v_j <- v_j - beta_ji u_i

alpha_j = <r_j, v_j> / <v_j, v_j>
x_{j+1} = P_S (x_j + alpha_j s_j)
r_{j+1} = r_j - alpha_j v_j
```

部署函数只接收：

- 原始 displacement `y`；
- 一次 pooled adjoint `A^T y`；
- 相机/射线几何张量；
- 与经典基线完全相同的 support。

它不接收 truth、family、split、case seed、geometry digest 或评价指标。proposal 执行前后会比较 operator call ledger；任何隐藏 `A/A^T` 调用直接 fail closed。

## 3. 当前可证明的安全性质

### 3.1 零扰动恢复强基线

当 `delta_theta=0` 时，第一方向是 `A^T y`。第一步精确线搜索等于 CGLS 第一步；后续全重正交方向张成同一 Krylov 空间并满足 `A^T A` 共轭。因此在精确算术下恢复 zero-start CGLS 轨迹。

当前小矩阵 CPU float64 测试在 24 步终点得到：

```text
field difference norm       1.862e-16
measurement difference norm 4.126e-16
max A-direction defect      2.220e-16
```

这只是数值单元测试，不是反应流重建结果。

### 3.2 测量残差逐步非增

每一步都沿当前 measurement direction 做一维最小二乘：

```text
alpha_j = argmin_alpha ||r_j - alpha v_j||_2^2
```

因此精确算术下 `||r_{j+1}||_2 <= ||r_j||_2`。实现记录最大 residual increase；正式运行要求它不超过 float64 容差。

这只保证相对上一步非增，不保证最终 residual 一定优于 CGLS-24。后者仍必须作为同预算主基线。

### 3.3 预算严格闭合

| 阶段 | Forward | Adjoint |
|---|---:|---:|
| pooled `A^T y` | 0 | 1 |
| bounded proposal | 0 | 0 |
| anchored projection `A z` | 1 | 0 |
| 23 个 continuation steps | 23 | 23 |
| **总计** | **24** | **24** |

重正交、网络推理、几何预处理、内存和端到端时间另行报告，不能把它们藏在“物理调用相同”后面。

## 4. 最小网络，不先堆大 backbone

算法包络只允许小于 20k 参数的方向修正器；第一个 Mac pilot 进一步限制在 8k 参数以内：

1. detector 分支：共享 2D 卷积编码 raw displacement 与 ray descriptors；相机维 masked mean/max pooling；
2. volume 分支：输入 `[A^T y, support, x, y, z]`，使用两个 width-12 depthwise-separable 3D block；
3. detector token 只通过 FiLM 调制 volume block；
4. `1x1x1` correction head 零初始化；
5. correction 强制 support mask，并投影到

```text
||delta_theta||_2 <= eta ||A^T y||_2, eta in {0.05, 0.10}
```

`eta` 只能用 early-stop split 选择。几何、位移尺度或噪声特征越出训练 calibration envelope 时，精确设置 `delta_theta=0`，回退到强基线壳层。

第一轮不使用 U-Net、FNO 或 DeepONet。只有小模型在严格门上出现稳定 headroom，才允许把 backbone 当消融加入，而不是把“模型更大”当创新。

## 5. 训练目标与数据防火墙

在训练 synthetic truth 上，可以对最终 24 步输出使用：

```text
L = field_relL2
  + 0.25 H1_rel
  + lambda_A ||A delta_theta||^2 / (||A g||^2 + eps)
  + lambda_m [log(R_measured / 1.02)]_+^2
  + lambda_c [log(R_clean / 1.02)]_+^2
  + lambda_tail CVaR_20%(field_harm)
```

`R_measured`、`R_clean` 都以同 case 的 CGLS-24 为分母。训练允许访问 synthetic truth，部署 API 和 route selection 不允许。

数据必须按完整 geometry cluster、rig、session 和时间块划分，不能随机拆相邻帧、ray、voxel 或重叠 crop。route-development 打开前，权重、归一化、`eta`、loss 权重、epoch、fallback envelope 和 gate 全部冻结。

## 6. 必做消融

1. zero-start 标准 CGLS-24；
2. 同一 anchored/reorthogonalized 壳层，`delta=0`；
3. 完整 `raw y + A^T y + geometry`；
4. 去掉 geometry；
5. 去掉 raw `y`；
6. 去掉 `A delta` penalty；
7. geometry/measurement 联合置换不变性；
8. 只错配 geometry 的敏感性；
9. opened synthetic 上的 full、exact-null-erased、row-space-erased retrospective SVD 三臂；SVD 不得进入部署算法。

第 1 与第 2 若不在 float64 容差内相同，先修 solver，禁止训练。第 9 若没有稳定 full-over-null-erased 差异，禁止把收益归因于 kernel prior。

## 7. 冻结门

### Development

- field/H1 gain `>=5% / >=3%`；
- measured、independent-clean reprojection mean ratio 均 `<=1.05`；
- 两种 reprojection 的逐 case P95 均 `<=1.10`；
- held-out-ray ratio `<=1.10`；
- `>1%` field harm rate `<=5%`，worst field gain `>=-5%`；
- 三个模型种子的 mean field gain 全为正；
- breakdown 为 0，账本恰为 `24F/24A^T`；
- `delta=0` 壳层与 CGLS-24 的 field/residual 相对差分别 `<=1e-10/1e-11`。

### OOD

只评估 development 已冻结的一条路线：

- field/H1 `>=2% / >=0%`；
- reprojection mean/P95 `<=1.10/1.15`；
- 不得重新选 `eta`、epoch、loss、fallback threshold 或 backbone。

### Fresh

至少 8 个未见 geometry clusters、24 cases；以 geometry cluster bootstrap，field gain 的 95% 下界必须大于 0。fresh synthetic 仍不能写成真实 BOST。

## 8. Mac 最小 pilot

先只回答“安全壳和信息假设是否值得训练”：

```text
grid: 12^3
cameras: 3
detector: 6x6
train / early-stop / route-dev: 16 / 4 / 6 cases
model seed: 1
network width: 8, parameters <= 8k
eta: frozen at 0.05
epochs: <= 30
training: MPS float32
formal scoring: CPU float64
OOD/fresh: closed
```

继续扩模前必须同时满足：

1. 24 对调用账本闭合；
2. `delta=0` 恢复基线；
3. field 正收益时 measured/clean reprojection 都不高于 `1.10`。

任一失败就停止该接口。三项都通过，才升级到 32 train、3 model seeds、18 OOD。

## 9. 当前实现与验证

- 算法壳：`demo_t16_operator/lgwo_a24_anchored_cdls.py`
- 单元测试：`demo_t16_operator/test_lgwo_a24_anchored_cdls.py`
- 安全壳结果：`9 passed`，Ruff 通过
- O1 机器结果：`OPENED_EXACT_NULL_HEADROOM_OBSERVED_NO_MODEL_AUTHORIZATION`
- O1 独立复算：36 rows、6 aggregate cells、1,121 assertions

已验证：

- arbitrary-`K` 和 A24 的精确 call ledger；
- `delta=0` 恢复 CGLS；
- bounded correction；
- residual 非增；
- calibration fallback；
- proposal 隐藏 operator call 的 fail-closed；
- deployable signature 不含 truth/family/split。

未验证：

- 网络能学到有用 correction；
- BOST geometry 条件化有效；
- 可学习模型的 synthetic field/reprojection gate；
- unseen rig/session；
- 真实反应流与 OERF 数据；
- 相对 DeepONet/FNO/FFNO/NeRIF/TDBOST 的性能或新颖性。

## 10. 新颖性边界

learned warm start、CG data consistency、unrolled inverse solvers、null-space networks 和 neural operators 都已有先例。当前候选不能仅凭“神经网络 + CGLS”声称原创。

可能形成论文贡献的组合只有：

1. BOST camera/ray geometry 条件化的方向扰动；
2. `delta=0` 精确基线恢复与逐步 data-residual 安全结构；
3. 同 `A/A^T`、同端到端时间、同内存下的严格比较；
4. unseen-rig / held-out-ray 的 fail-closed fallback；
5. TDBOST 固定 rig 多帧中的可摊销 near-null/row basis。

在严格 development、OOD、fresh 与真实数据门全部完成前，这仍只是可证伪候选。
