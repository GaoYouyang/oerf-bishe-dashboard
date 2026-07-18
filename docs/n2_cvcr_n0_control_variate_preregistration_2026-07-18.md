# N2-CVCR-N0：有限孔径控制变量算子预注册

日期：2026-07-18

状态：`PREREGISTERED_BEFORE_RESULTS`

证据上限：`SYNTHETIC_OPERATOR_INTEGRATION_MECHANISM_ONLY`

## 1. 真实问题与本轮边界

[Molnar 等人的 cone-ray BOS 模型](https://arxiv.org/html/2402.15954)把有限孔径测量写成孔径盘和光路上的积分；
其物理动机是打开光圈虽然提高进光量、缩短曝光，却会扩大景深模糊。PSU 后续开放 BOST 工作报告
cone-ray data loss 约需 `8000 points/pixel`；这些是 **frustum 积分点**，不能全部叫作孔径样本。
这使一个明确问题浮现出来：能否在不改变有限孔径期望、
不删除伴随、也不让网络直接猜三维场的情况下，减少每次 forward/adjoint 所需的高保真子射线？

本轮只在仓库已有的 prescribed weak-deflection 线性有限孔径沙盒中检验一个统计机制。它不是完整
Molnar 模型、不是非线性光线追迹、不是 OERF 数据，也不包含三维重建结论。

必须始终分开四层误差：pupil 有限孔径积分、pixel footprint/PSF、随未知折射率改变的弯曲光路、
以及相机标定误差。本轮只随机化第一层；路径积分仍用固定求积，`bend` 也是预设曲线而不是场依赖
ray equation。控制变量可以更快地积分既定模型，但不能把错误的 pupil、pixel、ray 或 calibration
变正确。

## 2. 候选估计器

把一个孔径点记作 `U=(u,v)`，单子射线线性算子记作 `A(U)`，目标是

```text
A_cone = E_U[A(U)],   U ~ Uniform(unit disk).
```

控制变量采用已知孔径均值的中心化二次基：

```text
phi(U) = [1, u, v, u^2-1/4, uv, v^2-1/4].
E[phi(U)] = [1, 0, 0, 0, 0, 0].
```

全部 IID 子射线被预先分成两个独立半区。第一半只负责评价由第二半拟合的 `H_2(U)=C_2 phi(U)`，
第二半只负责评价由第一半拟合的 `H_1(U)`：

```text
A_hat_k = E[H_-k(U)] + mean_{i in fold k}(A(U_i)-H_-k(U_i))
A_hat_CV = (A_hat_1 + A_hat_2) / 2.
```

因为拟合控制变量和被校正样本独立，而且残差项没有被删掉，所以每个 fold 的期望仍是 `A_cone`。
这一结构与通用的 sample-split/StackMC 控制变量思想相关，而不是凭空声称全新的蒙特卡洛理论：
[Tracey 与 Wolpert](https://arxiv.org/abs/1606.02261)。本项目要检验的是它能否成为**保留伴随的
BOST cone-ray 算子组件**，而不是宣称发明控制变量。

还有一个后续必须单独处理的陷阱：`A_hat` 无偏并不意味着
`1/2 ||A_hat x-y||^2` 或其梯度无偏，因为平方会多出估计方差项。本轮只测冻结估计算子的
forward、adjoint 与 normal action；只有机制过门后，才能预注册双独立估计器或永久冻结算子的
三维逆解实验。

## 3. 同预算对照

每个方法使用完全相同的高保真子射线数 `B in {16,32,64}`：

| 方法 | 随机性 | 高保真子射线成本 | 角色 |
|---|---:|---:|---|
| IID Monte Carlo | 32 个重复 | `B` | 最朴素基线 |
| antithetic MC | 32 个重复 | `B` | 消掉奇对称变化的强经典基线 |
| scrambled Sobol | 32 个重复 | `B` | 低差异强基线 |
| sunflower QMC | 确定性 | `B` | 仓库历史孔径盘规则 |
| disk product quadrature | 确定性 | `B` | `r^2` Gauss × 角向周期梯形强基线 |
| two-fold quadratic CV | 32 个重复 | `B` | 主候选，六维基，无额外射线 |

参考值由 `16 x 64=1024` 点的 Gauss-radial / periodic-angular disk quadrature 给出，并与
`12 x 48=576` 点结果比较；二者相对差超过 `0.3%` 时整轮只能报
`HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED`。

## 4. 四个预写死的 synthetic rigs

两组 development 只用于解释机制；最终门只看两个预声明 audit rigs。audit 同时扩大 aperture radius、
cone slope 和 bend，其中 `audit_boundary_crossing` 故意增加子射线跨越体素/support 边界的机会，用来
压力测试低阶光滑近似。结果打开后不得换 rig、换主预算、换基或调 ridge。

## 5. 主门与停止条件

主预算固定为 `B=32`。只有以下条件全部通过，才允许预注册一个“学习样本分配/基系数”的 N1 候选：

1. 两个 audit rig 的参考积分均通过 `0.3%` 收敛门；
2. 候选 pooled operator-RMSE 至少优于每个 rig 最强经典基线 `10%`；
3. 每个 audit rig 都至少优于其最强基线 `3%`；
4. 相对同一 IID 样本均值的逐重复伤害率不超过 `25%`；
5. mean-bias 的有限重复代理不超过 `2.5 x RMSE/sqrt(R)`；
6. normal-action error 不劣于最强基线；
7. 同一估计算子做 forward/adjoint，点积误差不超过 `1e-12`。

任一门失败就停止把这个二次控制变量升级成神经网络。即使全部通过，也只说明 synthetic cone-ray
积分的 cost/error 机制有 headroom；下一阶段仍需独立三维重建、真实有限孔径配对数据和 OERF 合同。

## 6. 创新性上限

二次 pupil basis 和交叉拟合本身不足以构成高质量论文创新。后续学习版本必须正面对比
[Neural Control Variates](https://arxiv.org/abs/2006.01524)与
[Recursive Control Variates for Inverse Rendering](https://doi.org/10.1145/3592139)。可能成立的差异只能来自
BOST 特定 frustum 测度、forward/JVP/VJP 一致性、反应流尖锐前沿的失效检测、跨光学几何迁移和
真实三维重建收益，而不是把通用控制变量换一个名字。

## 7. 预注册运行入口

```bash
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_aperture_control_variate.py \
  demo_t16_operator/test_finite_aperture_bost.py

.venv/bin/python demo_t16_operator/run_n2_cvcr_n0_mechanism_gate.py \
  --config demo_t16_operator/configs/n2_cvcr_n0_mechanism_prereg_v1.json \
  --output-dir demo_t16_operator/results/n2_cvcr_n0_mechanism_gate_v1
```

配置和代码必须先提交，再打开结果。运行后不得把 synthetic mean gain 写成算法、真实场、泛化或论文成功。
