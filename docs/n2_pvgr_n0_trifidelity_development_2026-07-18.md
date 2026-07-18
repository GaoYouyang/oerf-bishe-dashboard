# N2-PVGR-N0：物理有效性门控三级光线算子开发判决

## 结论先行

- 机器硬结论：`DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION`。
- 本轮诊断：`ORACLE_HEADROOM_CURRENT_PROXY_AND_IMPLEMENTATION_NO_GO`。
- 9/9 个 `case × stress` 通过数值与有限总体统计合同。
- 只有 4/9 个工况满足“轨迹残差方差不超过原混合残差 10%”的机制门；它明显依赖折射强度。
- 当前低成本风险代理 0/9 通过相对常数概率路由的 `0.90×` 方差门。
- 使用真实 `|H-M|` 的 oracle 分配 4/9 通过，说明概率分配存在有限 headroom，但现有代理没有抓住它。
- 3/3 个基础工况通过冻结路由 JVP/VJP 和双独立副本平方损失合同。
- 精确场查询合同为全高保真的 `0.621875×`，但当前 Mac 实现的证书加稀疏路由实际为 `2.48–2.51×`；实现成本门 0/3。

这不是算法成功。它是一次有用的开发 `NO-GO`：三级分解值得保留，当前风险分数和证书实现必须重做。

## 为什么从两级改成三级

上一版把两种误差混在同一个残差中：

1. 自动坐标梯度与中心差分梯度的差异；
2. 直线路径与场依赖曲线路径的差异。

单纯的曲率分数无法稳定预测这个混合残差。因此本轮冻结三条路线：

| 层级 | 路线 | 作用 |
|---|---|---|
| `L0` | straight ray + automatic gradient | 最便宜的机制参照 |
| `M` | straight ray + central difference | 去掉梯度实现差异，保留直线近似 |
| `H` | field-dependent RK4 curved ray + central difference | 当前合成高保真目标 |

随机估计器使用 `M` 作为全 ray 基线：

\[
\widehat\mu_H
=\frac1N\sum_i\left[M_i+\frac{I_i}{\pi_i}(H_i-M_i)\right],
\qquad I_i\sim\operatorname{Bernoulli}(\pi_i).
\]

条件于冻结的场、ray 状态和非零 `pi`，这是有限总体 Horvitz–Thompson difference estimator。公式本身不是创新。

## 概率分配修正

独立统计审计发现第一版“先加 floor，再按风险分剩余预算”不满足方差最优 KKT 条件。本轮已经改为等增量成本下的截断水位解：

\[
\pi_i=\operatorname{clip}_{[\pi_{\min},1]}(c q_i),
\]

其中 `q_i` 是非负残差风险，`c` 由期望 high-call 预算唯一确定。unsafe ray 精确设为 `pi=1`。若 `q_i=|H_i-M_i|`，这是 oracle；若 `q_i` 来自低路线特征，只能叫 proxy allocation。

实现同时保留两条路径：

- `horvitz_thompson_mean`：先算完整 high replay，用于核对均值与精确方差；
- `horvitz_thompson_sparse_mean`：只接收被 Bernoulli mask 选中的 high 输出，证明在线执行不需要预先支付全部 high 成本。

## 证书到底证明什么

对 `n=1+alpha f`，smoothstep 网格给出浮点评估的全局导数上界：

\[
|\partial_i f|\le1.5q_i\max|\Delta_i v|,
\]

\[
|\partial_{ii}f|\le6q_i^2\max|\Delta_i v|,
\qquad
|\partial_{ij}f|\le2.25q_iq_j\max|\Delta_i\Delta_jv|.
\]

由 `||d'||≤K` 得到：

\[
\|d(s)-d_0\|\le Ks,
\qquad
\|r(s)-r_0-sd_0\|\le\frac12Ks^2.
\]

代码进一步在直线周围的网格 cell tube 内使用局部上界，检查中心差分 stencil、归一化域和合成 frustum。

必须保留三个证据层级：

| 层级 | 当前状态 |
|---|---|
| 连续 ODE 公式上界 | 有保守公式，但 float64 不是 outward-rounded interval proof |
| RK4 与 crossing | 64→128、动量、离散 crossing 是数值 proxy |
| 真实相机与反应流 | 没有米制 ROI、背景距离、标定误差和真实 RI envelope，尚未建立 |

当前 renderer 是连续 smoothstep 场，没有 mask/occupancy 分支，所以主路由只启用 domain + synthetic-frustum 门；support crossing 证书只报告“如果未来启用 mask 会怎样”。不能把 inactive support 门写成已经解决真实前沿拓扑。

## 冻结实验

- family：`smooth_plume`、`wrinkled_density_interface`；未打开两个 reserved audit family。
- rig：smooth narrow、wrinkled wide、smooth wide。
- stress：`1× / 3× / 10×` 无量纲折射倍率。
- population：每 rig 64 条 scrambled Sobol pupil ray。
- route：`L0/M/H = 64` 步，high reference `128` 步。
- routing：`pi_min=0.1`；几何安全 ray 平均 `pi=0.25`；8192 个独立 Bernoulli replay。
- gradient：4-ray、10-step 冻结路由；4096 个 JVP replay；两个独立 4096-replica cross-loss 分支。

这些都是 development choices，不是 blind audit。

## 主要结果

### 基础尺度

| Rig | `Var(H-M)/Var(H-L0)` | proxy / uniform variance | oracle / uniform variance | strict future-mask safe | primitive cost / full H | Mac end-to-end / full H |
|---|---:|---:|---:|---:|---:|---:|
| smooth narrow | 0.0481 | 1.0971 | 0.8344 | 1/64 | 0.6219 | 2.51 |
| wrinkled wide | 0.00679 | 0.9734 | 0.9068 | 0/64 | 0.6219 | 2.51 |
| smooth wide | 0.0917 | 0.9943 | 0.6682 | 10/64 | 0.6219 | 2.48 |

解释：

1. `M` 的确把原残差方差压低到约 `0.7%–9.2%`，所以机制分解有价值。
2. 当前 risk proxy 在 smooth narrow 上反而让方差增加约 `9.7%`；另外两组也没有达到 `10%` 改善门。
3. oracle 在两组有明确余量，说明问题不是 HT 公式完全无效，而是可观测特征不足。
4. 原语合同低于 1，但 Python 证书的逐 ray/逐 interval 循环使端到端时间超过 full H 两倍。当前代码不能宣传加速。

### 强度依赖

| Stress | geometry/frustum safe | 轨迹残差现象 | 路由含义 |
|---|---|---|---|
| `1×` | 3 个 rig 都是 64/64 | 三组均低于 10% 门 | 存在三级分解 headroom |
| `3×` | 3 个 rig 都是 64/64 | 仅 wrinkled wide 仍低于 10% | `M` 不再普遍是强控制变量 |
| `10×` | 3 个 rig 都是 0/64 | 轨迹残差占 42%–90% | fail closed，全部 high，无加速主张 |

这正是需要寻找的物理窗口：曲线修正开始重要，但尚未吃完相机/视锥余量。当前无量纲场还不能证明实验里存在这个窗口。

## 导数与训练损失合同

基础尺度三组均通过：

- routed JVP 对固定 mask 有限差分相对误差：`2.9e-9–1.9e-8`；
- VJP dot relative error：`2.2e-16–9.9e-15`；
- 两独立副本 cross-loss 相对误差：低于 `8.3e-6`；
- cross-loss directional derivative 相对误差：低于 `8.3e-6`。

冻结 `pi` 与 mask 时，forward/JVP/VJP 的 HT 结构保持一致。如果 `pi=pi(theta)` 且只对 `1/pi` 普通反传，却忽略 Bernoulli 分布导数，会缺少 score-function 项。当前实现明确 detach probability。

同一副本平方会多出 `0.5 trace(Cov)`；本轮把这个正偏差作为负对照保留。双副本单次 loss 可以为负，不能据单次值 early-stop。

## 为什么当前版本判 NO-GO

1. 当前风险代理与真正 `|H-M|` 的秩相关在不同 rig 中符号不稳定。
2. proxy routing 0/9 过 `0.90×` 方差门。
3. support/front 的严格证书非常保守；启用真实 mask 时基础尺度最多只有 10/64 ray 可降采样。
4. 证书没有复用 `M` 已计算的场值与中心差分，重复工作严重。
5. float64 cell bounds 不是区间算术证明。
6. 没有真实位移提取噪声、标定误差或 held-out 物理端点。

## 新颖性边界

不能声称首创：HT/Russian roulette、MLMC/MFMC/ACV、曲光线、曲光线伴随、有限孔径 cone-ray、事件感知可微渲染、神经控制变量、一般 adaptive routing。

最接近且必须引用的一级来源：

- [Horvitz–Thompson estimator](https://doi.org/10.1080/01621459.1952.10483446)
- [Gradient Estimation Using Stochastic Computation Graphs](https://proceedings.neurips.cc/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html)
- [Adjoint Nonlinear Ray Tracing](https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/)
- [Single View Refractive Index Tomography with Neural Fields](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)
- [NeRIF](https://doi.org/10.1063/5.0250899)
- [NeDF](https://doi.org/10.1063/5.0241191)
- [finite-aperture cone-ray BOS](https://arxiv.org/html/2402.15954)
- [He et al. 4D BOST](https://doi.org/10.1145/3809488)

当前可检验的窄定位是：**Physics-Validity-Gated Multifidelity Ray Operators for Neural BOST**。贡献若成立，应来自 BOST 特有的有效性证书、正确的 forward/JVP/VJP 合同、物理价值窗口与真实成本，而不是重新命名 HT。

## 复现

```bash
.venv/bin/python demo_t16_operator/run_n2_pvgr_n0_trifidelity_development.py
.venv/bin/python -m pytest \
  demo_t16_operator/test_ray_safety_certificate.py \
  demo_t16_operator/test_topology_certified_routing.py \
  demo_t16_operator/test_run_n2_pvgr_n0_trifidelity_development.py
```

结果目录：`demo_t16_operator/results/n2_pvgr_n0_trifidelity_development_v1/`。

## 下一步停止规则

只有同时出现以下证据，才继续向论文候选升级：

1. 可观测 proxy 在每个开发 rig 上相对 constant-pi exact variance 都低于 `0.90×`；
2. 证书 + `M` + sparse `H` 的端到端 p90 低于 full `H`；
3. fresh-instance calibration 的逐 rig false-safe 单侧上界过门；
4. 真实曲线修正高于 flow-off/标定不确定度，同时仍位于校准 frustum 内；
5. 重建层的 field L2/H1/front 与 held-out reprojection 无退化。

任一项长期无法成立，应停止这条路，而不是继续修改阈值。
