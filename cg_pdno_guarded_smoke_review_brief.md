# CG-PDNO：协方差/几何条件化 BOST 逆算子的第一轮机制审计

更新时间：2026-07-15
证据等级：`engineering/development only`。没有真实 BOST 三维真值、独立生成器或 DeepONet/FNO/NeRIF 完整对照，因此不允许写“优于现有算法”。

## 一句话判断

主候选应保留为 **显式 `F/F^T`、异方差预白化、可变视角集合条件化、固定深度的低调用次数逆算子**。三种子小型 smoke 给出稳定正信号，但两个 OOD 实验同时证明：无条件 learned correction 会灾难性失效，简单噪声缩放也不是通用答案。下一版应改成“共享物理主路径 + 单次可拒绝 learned correction”，再进入独立生成器和 OpenBOST。

## 算法接口

学习的是带采集条件的逆算子：

`G_theta(y, g, M, C, S) -> n(x,y,z)`

- `y`：多视角 displacement/phase；
- `g`：每条 ray 或相机几何；
- `M`：active view / bad-pixel mask；
- `C`：由 flow-off 重复帧或置信度估计的观测协方差；
- `S`：部署时可获得的体 support；
- `n`：非负三维折射率或密度相关场。

当前每层执行：

`r_k = M (y - F_g x_k)`

`d_k = F_g^T C^{-1} r_k / L(g,C)`

`[alpha_k, beta_k] = controller(z_geometry, z_noise, z_residual, k/K)`

`x_{k+1} = Pi_{S,+}[x_k + alpha_k d_k + beta_k P_theta(x_k,d_k,delta x_k)]`

约束：`0 < alpha_k < 2`；proximal 最后一层零初始化；不学习停止；`trust=0` 时学习步长回到固定物理步，proximal 归零，数值上严格等于预白化 projected-gradient fallback。

## 三组关键实验

### E1：无门控前驱模型暴露灾难性 OOD

同为 4 次 `F + F^T` 的小型 straight-ray phantom：

| 域 | learned vs prewhitened physics mean gain | `>1%` harm |
| --- | ---: | ---: |
| IID | +72.08% | 0% |
| family OOD | +44.77% | 3.13% |
| layout OOD | +54.86% | 0% |
| noise OOD | **-72.52%** | 46.88% |
| joint OOD | **-326.51%** | 96.88% |

结论：IID 大幅获益不能掩盖 learned correction 在噪声外推时爆炸。

### E2：事后设计的噪声门在全新字段上救回前驱模型

固定 `b_noise=min(1,(0.05/q)^2)` 后，使用从未参与 v1 的 `9,000,000` seed offset：

| 域 | unguarded gain | guarded gain | guarded harm |
| --- | ---: | ---: | ---: |
| noise OOD | -78.35% | **+38.19%** | 0% |
| joint OOD | -157.41% | **+16.20%** | 0% |
| layout OOD | +48.67% | +48.67% | 6.25% |

这是“噪声信任控制有机制价值”的证据，不是可部署结论：当前 `q` 来自生成器；真实系统必须用独立 flow-off、brightness/optical-flow confidence 或 phase variance 估计。

### E3：三种子 CG-PDNO fresh 运行否掉“噪声门总有益”

新角度、新字段、新几何池，train/validation/test geometry id 无重叠：

| 方法 | validation mean gain | validation max seed harm | test mean gain | test max seed harm |
| --- | ---: | ---: | ---: | ---: |
| raw CG-PDNO | +41.14% | 11.11% | **+21.77%** | 0% |
| noise-trust CG-PDNO | +38.29% | 11.11% | +13.36% | 0% |

简单噪声门没有减少伤害，反而丢掉测试收益，因此当前不纳入主算法。验证伤害集中于 `g_0110110` 与 `g_0011110` 两个字段/布局，且噪声低；粗角度描述子和条件数都不能单独识别。

事后 residual-ratio 诊断在 validation 选出阈值 `0.9`，把 source-field harm 从 11.11% 降到 0%，mean gain 从 41.14% 升到 42.16%；对已存在 test 的诊断结果是 21.77% 降到 20.50%，harm 仍为 0。该阈值不是 blind 结论，而且当前 recurrent 结构要额外跑完整 fallback，调用成本不合格。

## 下一版结构：Base-Correction CG-PDNO

1. 先运行 `K` 次固定、预白化、可复核的物理主路径，得到 `x_base`。
2. ray-set encoder 与小型 proximal 只预测一次 `delta x`，不再让 learned state 改写后续物理梯度。
3. 计算 `x_candidate=Pi_{S,+}(x_base+b delta x)`。
4. 用一次额外 forward 比较 candidate/base 白化残差；若未通过 validation 冻结的接受条件，返回 `x_base`。
5. 总调用与 `K+1` 或 `K+2` 次传统迭代做同前沿比较；不能把额外 forward 隐去。

这个改写的研究问题是：**在保留严格物理 fallback 与完整调用记账时，BOST 特有的 geometry/covariance-conditioned correction 能否提高少视角薄前沿的低调用精度，并降低 tail harm？**

## 论文门槛

必须全部满足后才保留题目 *Covariance- and Geometry-Conditioned Data-Consistent Neural Operators for Sparse-View BOST*：

1. `F/F^T` 内积误差 `<1e-5`，真实 forward 若非线性则验证 VJP/Jacobian-adjoint。
2. 与 CGLS/TV、预白化 SPG/PBB、FNO/FFNO、DeepONet、Learned Primal-Dual、NeRIF 同数据、同相机、同调用/时间前沿比较。
3. 至少 5 个训练种子；source field/run 为统计单元；报告 mean、p10、`>1% harm`、worst-layout 和 field-cluster bootstrap。
4. train/validation/fresh-blind 的 field、geometry、noise realization、run 全隔离；test 不参与架构或阈值选择。
5. 至少一个独立 CFD/nonlinear ray generator和一个真实 OpenBOST/OERF case。
6. 没有三维真值的真实实验只报告 held-out camera、重复性、物理边界和成本，不伪造 field L2。

## 现在给何远哲师兄看的问题

> 我把主线收缩为 geometry/covariance-conditioned、显式 F/F^T 的低调用 BOST 逆算子。小型三种子实验有正信号，但也发现 learned correction 会在特定噪声/布局下失效；目前不声称优于 FNO/DeepONet。想先请你确认：组内 displacement 是否有 flow-off 重复帧或 confidence；现有 forward 是否能调用 F 与 F^T/J^T；九路几何是否跨实验变化；能否留一套真实 case 和一条 camera 只作盲验；以及主终点更看重 field、held-out view、薄前缘还是速度？我先交付 loader、伴随检查、预白化强基线和调用前沿，再决定是否训练完整模型。

## 复核命令

```bash
.venv/bin/python -m pytest -q demo_t16_operator/test_cg_pdno.py
.venv/bin/python demo_t16_operator/validate_candidate_operator_guarded_fresh.py
.venv/bin/python demo_t16_operator/validate_cg_pdno_smoke.py
.venv/bin/python demo_t16_operator/validate_cg_pdno_trust_fresh.py
```

当前：`8 passed`；guarded precursor `736` rows；CG-PDNO smoke `120` rows；trust fresh `144` rows；所有对应 validator 通过。
