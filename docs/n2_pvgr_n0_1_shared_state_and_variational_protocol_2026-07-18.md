# N2-PVGR-N0.1 / N1：共享直线路径状态与轨迹变分预测器冻结协议

> **后续审计勘误（2026-07-18）：**N1 的
> `delta d'=F0+A delta r+B delta d` 是完整动力学的一次仿射/Newton 型直线路径修正，
> 不是 `d'=epsilon F` 在 `epsilon=0` 的精确离散同伦 JVP。精确一阶同伦轨迹只保留
> `delta d'=F0`，而中央差分算子的 `A_delta/B_delta` 只进入观测积分导数。N2 已用
> forward-mode JVP 逐元素核对，并把两个 reference no-harm 失败修复到 9/9；但 Picard-1/2
> 在当前弱合成场上更快更准，因此新的结果仍只是机制桥接，不是论文算法成功。见
> [N2 算子一致同伦桥接](n2_pvgr_n2_operator_consistent_bridge_2026-07-18.md)。

## 为什么在训练路由器前多加这一轮

`N2-PVGR-N0` 已经证明三件事：三级 `L0/M/H` 分解在基础尺度有机制余量；当前
curvature/tube risk proxy 没有稳定利用余量；当前 Python 证书实现比 full high 更慢。

本协议不扩大模型容量。它先回答两个更基础的问题：

1. 能否让 medium、证书和 risk feature 读取同一个 straight-path state，消除重复查询和循环？
2. 能否用一阶轨迹变分方程直接预测有方向的 `H-M`，而不是用曲率标量猜残差大小？

两项都属于 development。未打开 reserved family，未运行真实 BOST，不能据此声称重建、
泛化、实验或论文成功。

## 新发现：旧 residual label 的收敛门不够严格

旧实验验证了完整 high output 的 64→128 步相对误差，却没有验证真正用于 routing 的
微小残差 `R=H-M`。重新用相同步数的 central-straight `M_s` 与 curved `H_s` 比较到
256 步参考后得到：

| Rig | `R64` 对 `R256` relative-L2 | `R128` 对 `R256` relative-L2 | `Var(R128)/Var(R256)` |
|---|---:|---:|---:|
| smooth narrow | 9.42% | 2.54% | 1.0013 |
| wrinkled wide | 10.31% | 2.02% | 1.0020 |
| smooth wide | 6.64% | 1.98% | 0.9997 |

`3x/10x` stress 的结论相同：64 步 residual relative-L2 约 6.5%–10.2%，128 步约
2.0%–2.6%。因此冻结：

- `M` 与执行 high 都使用 128 步；
- 256 步只作 residual reference；
- residual convergence 必须逐 rig `<=5%`；
- 不能再用“完整输出误差低于 1%”替代微小 residual 的收敛审计。

这不会反转上一轮 NO-GO。用 128 步 residual 重算后，旧 proxy/uniform 方差比仍为
`0.964–1.105`，0/6 个 1x/3x 可路由工况通过 0.90 门；oracle 仍只有 4/6 通过。

## N0.1：共享 `StraightPathState`

一份 state 必须同时包含：

```text
midpoint positions and straight direction
projection_u / projection_v
scalar field and refractive index
central-difference gradient and transverse curvature
medium projected deflection
cell ids and normalized-domain margins
explicit field-query / coordinate-VJP accounting
```

medium output、active domain/frustum certificate 和 risk feature 只能从这份 state 读取。
当前 renderer 没有 mask/occupancy 分支，所以 support/root certificate 不进入在线执行；它只保留
为未来真实 renderer 的 report-only 压力测试。

### 查询账本

128 步 central straight state 每 ray 使用 `7*128=896` 次 scalar primitive query。
128 步 curved RK4 high 按当前实现使用 `35*128=4480` 次。若安全 ray 平均 high 概率为
0.25，且 active certificate 不新增 scalar query，则期望查询比为：

\[
\frac{896+0.25\times4480}{4480}=0.45.
\]

这只是 primitive ceiling。实际结果必须把 state 构建、cell bound、概率分配、随机 mask、
sparse high、HT 聚合、Python 调度和内存分配全部计入。

后续性能审计确认，这个警告不是形式问题：旧实现中 16/64 条 sparse high 的 CPU 中位时间
仍约为 full 64-ray high 的 99%，因为 64 步 high 不论 ray 数都要经历 1,799 次顺序插值
dispatch。旧 `0.621875` 只能称 scalar point-query ratio，不能称加速。N0.1 必须先融合每个
RK4 stage 的七点查询、移除重复 grid validation，再重新测 composite closure。

### 公平时间门

- Apple M5 CPU、float64；MPS 不作为主结果，因为当前物理审计使用 float64；
- 固定 thread count、预热、至少 20 个独立 route masks；
- 比较 composite routed p90 与 full-high p10，不把分开测得的中位数相加；
- 同时报 amortization horizon `1/64/1024`，静态 camera geometry 与随场变化的 bounds 分账；
- 3/3 基础 rig 都要求 `routed p90 / full-high p10 < 0.90`；
- 任一 NaN、域外、无法认证或强度 10x 的 ray 必须 fail closed 到 high。

N0.1 若只通过成本门，只能称工程修复，不计论文算法贡献。

## N1：一阶轨迹变分残差预测器

在 straight medium path `(r0,d0)` 上定义：

\[
F(r,d)=\frac{(I-dd^T)\nabla n(r)}{n(r)}.
\]

令 curved ray 相对 straight path 的差为 `delta r, delta d`。冻结场和 ray population 后，
使用仿射变分系统：

\[
\delta r'=\delta d,
\qquad
\delta d'=F_0+A\delta r+B\delta d,
\]

其中：

\[
A=(I-d_0d_0^T)
\left(\frac{\nabla^2n}{n}-\frac{\nabla n\nabla n^T}{n^2}\right),
\]

而对任意方向扰动 `q=(I-d0 d0^T) delta d`：

\[
B\delta d=-q(d_0^T\nabla\log n)-d_0(q^T\nabla\log n).
\]

预测 residual 是沿 straight path 对 `A delta r+B delta d` 的投影积分。它有两种冻结用途：

1. `risk_i=||Delta_hat_i||`，只改变 HT probability；
2. `M1=M+Delta_hat`，作为更强 control variate，再用 HT 无偏纠正 `H-M1`。

公式中的 trajectory linearization、dynamic ray tracing 和 control variate 都可能已有先例；
当前不把它们写成新颖性。可检验价值只可能来自 BOST 特定有效性门、低成本 residual prediction、
forward/JVP/VJP 同状态合同和真实重建实证。

### N1 已实现的数学与代码合同

当前 `trajectory_variational_predictor.py` 使用 float64、冻结直线路径和 RK4 仿射变分积分；
场、路径、Hessian、预测 residual 与 risk 全部 stop-gradient。任一射线发生域外、折射率下穿、
位置或方向扰动超过线性化半径时，risk 设为 infinity 并回退，不允许静默外推。

独立测试已经覆盖：

- 常系数解析解、零梯度场、切空间约束和输入 fail-closed；
- 梯度/Hessian 与空间中心差分；
- `A/B` 对 ray RHS 的双边有限差分；
- automatic/central 两种完整曲光线 `H-M` 对照；
- 48 步弱偏折开发核上方向余弦大于 0.99999、relative-L2 小于 0.005。

这些只证明当前离散实现与当前合成核相容，不证明真实 BOST、非光滑激波、焦散或跨设备泛化。

### 正式开发筛选：强信号，但不是全过

使用 3 个 development rig、3 个 stress、64 rays，匹配执行步数 128，参考步数 256。完整
候选 closure 包含 shared central medium、automatic-Hessian predictor 和 correction；计时采用
每条路线 20 次随机交错，比较 candidate p90 与 full-high p10。

| 指标 | 9 行范围 | 开发门 | 结果 |
|---|---:|---:|---|
| `prediction` 对 matched `H128-M128` relative-L2 | 0.0464–0.0685 | <=0.10 | 9/9 |
| `Var(H-M-prediction)/Var(H-M)` | 0.0030–0.0226 | <=0.50 | 9/9 |
| 逐 ray risk Spearman | 0.9259–0.9961 | >=0.50 | 9/9 |
| candidate 对 full `H256` relative-L2 | 0.000281–0.000957 | <=0.002 | 9/9 |
| candidate reference error / full `H128` reference error | 0.9998–1.7736 | <=1.10 | 7/9 |
| candidate p90 / full-high p10 wall time | 0.0875–0.0882 | <=0.25 | 3/3 rig |

机器结论保持 `DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION`。失败项是
`wrinkled_wide_aperture` 的 3x 与 10x reference no-harm gate；不是速度、排序或 matched
residual 门。

### 新漏洞：matched residual 收敛不推出 mixed closure 收敛

`H128-M128` 与 `H256-M256` 很接近，是因为 high 与 medium 的积分误差会部分抵消。但是实际
候选 `M128+prediction128` 对 `H256` 的比较混合了两种离散层级，`M128-M256` 不再抵消。
在 residual 本身远小于完整输出时，这项数值误差可与物理轨迹 residual 同量级。

因此今后必须分开报告：

1. matched-discretization residual accuracy，用于同一 128 步 high target 的控制变量；
2. candidate/full-output 对 256 步 reference 的绝对误差；
3. candidate reference error 相对 full `H128` reference error 的 no-harm ratio；
4. 任何只在 matched residual 上通过、却恶化 reference no-harm 的场景必须 fail closed。

这也给出下一步优先级：先做离散 RK4-step JVP（包括方向归一化）和 Picard 一/二次更新强
基线，再决定是修正离散 defect，还是用 N1 risk 只做回退路由。此时不应先训练更大的网络。

### 新颖性边界

动态射线追踪、一阶光路扰动、可微非线性光线和多保真估计都有历史先例。建议从以下一级来源
核对边界：[Raffel 的 BOS 综述](https://link.springer.com/article/10.1007/s00348-015-1927-5)、
[Norton 1987](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-10-1919)、
[Lira 与 Vest 1987](https://opg.optica.org/ao/abstract.cfm?uri=ao-26-18-3919)、
[Adjoint Nonlinear Ray Tracing](https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/)、
[Zhao 等 CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)、
[cone-ray BOS](https://arxiv.org/abs/2402.15954) 和
[NeRIF](https://arxiv.org/html/2409.14722v2)。

可争取的贡献只能是“面向神经 BOST 测量算子的共享状态变分 defect correction、物理有效域
回退、离散一致性与端到端重建/成本证据”的组合。何远哲 TDBOST 已公开提到 ray-path
distortion correction；在拿到其输入、监督目标和实现细节前，不能声称首次为 OERF/BOST
预测光路畸变。

## 冻结比较矩阵

每个可路由 rig/stress 必须同时比较：

| 角色 | 方法 |
|---|---|
| 无信息基线 | constant `pi` |
| 旧零阶代理 | curvature/tube scalar proxy |
| 工程复用代理 | shared-state projected curvature proxy |
| 新解析候选 | variational residual norm |
| 修正 medium | `M1=M+Delta_hat` |
| 不可部署上界 | true `|H-M|` oracle |
| 目标 | high-only 128-step curved operator，256-step residual reference |

不允许只对比 DeepONet/FNO，也不允许把 oracle 当方法。

## N1 开发门

进入任何小型学习 router 前，3/3 基础 rig 必须同时满足：

1. `R128` 对 `R256` relative-L2 `<=5%`；
2. variational norm 对 true residual norm 的逐 rig Spearman `>=0.5`；
3. variational-proxy / constant-`pi` exact variance `<=0.90`；
4. `Var(H-M1)/Var(H-M)<=0.50`；
5. predictor 计入后的 work-variance ratio `<=0.90`；
6. active certificate false-safe count 为 0；
7. routed forward/JVP/VJP 与双独立副本 loss 合同继续通过。

任一项不过，解析 N1 判 NO-GO。不能通过调低门槛或扩大网络挽救。

## 真实迁移前仍缺的证据

- 米制 ROI、背景距离和归一化矩阵；
- 波长、气体组成、压力温度范围与 Gladstone-Dale 模型；
- 相机内外参、pixel pitch、focus、f-number、畸变和 aperture model；
- 最大偏折工况及 flow-off/重复图像的 optical-flow 与 calibration uncertainty；
- 当前 NeRIF/4D renderer 的 ray/pupil/path sample 数、mask/occupancy 分支与 forward/JVP/VJP profile；
- held-out view/time 及独立温度、密度、PLIF 或其他物理终点。

如果真实 curved correction 始终低于这些测量不确定度，即使 N0.1/N1 的合成方差和速度过门，
整条论文路线仍应停止。
