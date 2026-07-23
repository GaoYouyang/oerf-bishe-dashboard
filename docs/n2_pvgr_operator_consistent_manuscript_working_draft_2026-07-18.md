# 面向曲光线 BOST 的离散算子一致弯曲同伦导数

> **WORKING DRAFT / NOT SUBMISSION READY**
> **中文论文工作稿，不是投稿稿，不是已完成论文，也不是创新性或优越性声明。**
> 版本：`2026-07-18 / development mechanism bridge`
> 机器判定：`MECHANISM_BRIDGE_SIGNAL_ONLY_96_CELL_RECONSTRUCTION_AND_REAL_DATA_GATES_CLOSED`

## 候选标题

### 首选工作标题

**面向曲光线背景纹影层析的离散算子一致弯曲同伦导数：forward-JVP 教师、解析 OCBH 与 Picard 强基线的机制验证**

### 备选标题 A

**曲光线 BOST 测量算子的一阶离散缺陷：算子一致线性化及其强迭代基线**

### 备选标题 B

**从直线路径到曲光线路径：BOST 离散 bend-homotopy 导数的可复现开发研究**

这些标题均为内部候选。正式题名必须等待 96-cell 分组实验、三维重建、神经算子强基线、
reserved-family audit 和真实数据门全部完成后再确定。

---

## 证据等级声明

本文当前证据等级仅为：

> **Development-only mechanism bridge / synthetic forward-operator evidence**

本文所有数值只来自仓库中的下列冻结机器结果：

- `demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/result.json`
- `demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/metrics.csv`
- `demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/teacher_metrics.csv`
- `demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/reference_sentinel.csv`
- `demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/timing.csv`
- `demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/manifest.json`

`result.json` 中四项授权均为 `false`：

| 授权项 | 当前值 |
|---|---|
| development bridge authorization | `false` |
| reserved audit authorization | `false` |
| paper claim authorization | `false` |
| real data authorization | `false` |

因此，本文不得被解读为以下任一结论：

1. 不得称 OCBH 已经是论文级新算法。
2. 不得称 OCBH 优于 Picard、DeepONet、FNO、F-FNO/FFNO、NeRIF、TDBOST 或其他现有方法。
3. 不得称九个合成 development cell 证明了跨场、跨几何或真实反应流泛化。
4. 不得称当前 forward-only 结果完成了三维重建。
5. 不得把 H256 或 H512 称为实验真值。
6. 不得把当前 Mac CPU/Python/PyTorch 墙钟比例称为硬件无关复杂度优势。

本文的唯一正面证据是：在当前离散程序和九个已打开的合成 development cell 内，解析 OCBH
与完整 forward-mode 离散 JVP 在浮点精度上等价；作为一阶预测器，OCBH 修正了旧 N1 的主要
算子不一致；但在当前九格的 matched residual 与 Mac CPU worst-case 墙钟汇总上，仍被简单
Picard-1/2 强基线明显压过。这个结论不表示 Picard 在每个 no-harm 指标的每一格都逐项占优。

---

## 摘要

背景纹影层析（background-oriented schlieren tomography, BOST）通过多视角背景位移反演
三维折射率或密度场。常用弱偏折模型沿预定直线积分折射率梯度，但在较强折射率梯度、宽孔径
或薄界面附近，真实光路会依赖未知介质并发生弯曲。完整曲光线追迹可降低这一模型失配，却会
增加前向计算、微分和三维反演的成本。本工作稿研究一个范围更窄的机制问题：能否对与高保真
程序完全相同的中央差分场查询、RK4 离散和中点观测规则构造一阶弯曲缺陷，而不调用完整高保真
输出作为候选输入。

我们在离散曲光线程序中引入只控制轨迹弯曲的标量同伦参数，同时保持观测 integrand 为完整物理
曲率。完整程序的 forward-mode JVP 作为离散教师；解析候选 OCBH
（operator-consistent bend-homotopy）在直线路径上计算同一中央差分算子的坐标与方向 Jacobian，
并传播精确的一阶离散切线。九个合成 development cell 中，OCBH 对 forward-JVP 教师的最坏
输出 relative-L2 为 `2.1591563693023527e-14`，位置和方向切线最坏 relative-L2 分别为
`4.950399402606913e-15` 和 `5.379379355283756e-15`。相对 H256 evaluator，OCBH 的最坏
matched-residual relative-L2 为 `0.013371097842020507`，最坏全局 reference no-harm 比值为
`1.0644073595295993`，最坏逐射线 Q95 no-harm 比值为 `1.0561120759599356`。九个 OCBH
主候选格、九个教师等价检查格、九个 H256/H512 sentinel 和三个计时 rig 均满足当前 development 门。

然而，修正最终路径重积分的 off-by-one 后，Picard-1 与 Picard-2 在同九格上取得更低的最坏
matched-residual relative-L2，分别为 `0.0017091774596397962` 和
`0.0004982646139615191`；它们的最坏 p90/H128-p10 墙钟比也仅为
`0.025417265529224447` 和 `0.037156454198213146`，低于 OCBH 的
`0.15078973281471242`。这构成本文最重要的反面结果：当前证据不支持把 OCBH 作为最佳前向
算法。OCBH 更合理的后续角色是可解释的一阶导数、remainder/risk 特征和可微三维重建算子支架。
在 96-cell 分组实验、cone-ray、三维重建、等预算 DeepONet/FNO/F-FNO 对比、reserved audit
和真实 OERF 数据验证完成前，本文保持 **WORKING DRAFT / NOT SUBMISSION READY**。

**关键词：** 背景纹影层析；曲光线；离散同伦；Jacobian-vector product；Picard 迭代；
神经算子；三维折射率重建；负结果

---

## 1. 物理问题

### 1.1 从 BOS 位移到三维折射率场

BOS 比较无流场参考图与有流场畸变图，通过背景纹理的表观位移感知透明介质中的折射率梯度。
BOST 将多个视角的位移或图像观测联合反演为三维折射率、密度或温度相关场。对反应流而言，
火焰面、热羽流、激波和组分梯度都可能形成强空间折射率变化。

在本文的归一化合成程序中，单条射线的曲率写为

$$
F_\Delta(r,q)
=\frac{P(q)g_\Delta(r)}{n(r)},
\qquad
P(q)=I-u(q)u(q)^\mathsf{T},
\qquad
u(q)=\frac{q}{\lVert q\rVert_2},
$$

其中 $n(r)$ 是折射率，$g_\Delta(r)$ 是与高保真程序相同的中央差分折射率梯度。曲光线满足

$$
\frac{\mathrm d r}{\mathrm ds}=u(q),
\qquad
\frac{\mathrm d q}{\mathrm ds}=F_\Delta(r,q).
$$

离散观测使用每个步长两端状态的算术中点：

$$
H_s=\Pi h\sum_j
F_\Delta\!\left(
\frac{r_j+r_{j+1}}{2},
\frac{q_j+q_{j+1}}{2}
\right),
$$

其中 $s$ 是步数，$h$ 是步长，$\Pi$ 将三维偏折投影到探测器平面两轴。当前输出是归一化
synthetic angular deflection，不是经过焦距、放大率、像元尺寸和标定转换后的真实像素位移。

### 1.2 计算困境

直线路径模型 $M_s$ 计算便宜，但忽略了“未知折射率场改变光路，改变后的光路又改变观测积分”
这一反馈。完整曲光线 $H_s$ 在逐场优化或三维重建中需要反复进行 RK4 场查询；若还需对体场
参数反向传播，成本与内存压力会进一步增加。

本文不直接解决完整三维逆问题，而先隔离一个可证伪问题：

> 在不把 fresh exact-high 输出喂给候选的前提下，能否构造与完整离散曲光线程序同算子、同离散、
> 同求积规则的一阶 trajectory defect，并明确它相对简单逐次逼近的真实价值？

### 1.3 当前问题边界

当前机器实验仅包含三组 synthetic rig、两个已打开的 phantom family、单一 field seed/家族配置、
每格 64 条 ray 和归一化折射率应力扫描。它尚未包含：

- 真实背景图、光流、相机标定或像素单位；
- 同一像素内的有限孔径 cone average；
- 遮挡、焦散、间断界面和薄透镜 ray-linking；
- 多视角三维反演和独立体场真值；
- OERF 的真实相机、镜头、喷雾、火焰或高速时序数据。

---

## 2. 相关工作边界

### 2.1 曲光线与逐次逼近不是新概念

Norton 的 ray-linking 工作将固定两端点的射线路径写成隐式积分方程并使用 successive
approximations；Červený 等对动态射线追迹和近轴射线系统给出了经典框架。因此，本稿不能声称
“首次用一阶修正近似曲光线”或“首次用 Picard 迭代求折射路径”。当前 Picard 实现还是固定入口
状态的 IVP 路径更新，不是 Norton 意义下满足两端边界条件的 ray-linking 算法。

### 2.2 BOS/BOST 与有限孔径

Raffel 的 BOS 综述给出了从折射率梯度、光线偏折到背景位移的实验链条。Molnar 等进一步对
finite-aperture BOS 建立 cone-ray forward/inverse model，表明一像素对应孔径上的 ray bundle，
不能总被一条 thin ray 取代。因此，本稿的单射线 population 结果不能被称为 cone-ray BOST，
也不能声称首次考虑景深或有限孔径。

### 2.3 可微曲光线与神经折射率场

Zhao 等在 CVPR 2024 中通过神经场和可微曲光线追迹处理折射率层析。OERF 的 NeRIF 已用神经
隐式折射率场进行 BOST 体重建；TDBOST 已沿 $X-Y-Z-T$ 张量分解和轻量网络推进高速四维重建。
这些工作关闭了“首次神经折射率场”“首次可微曲光线重建”和“首次四维神经 BOST”等宽泛新颖性
表述。当前 OCBH 若有潜在贡献，只能来自严格限定的组合：BOST 特定离散算子的一致一阶导数、
可观测 remainder/fail-closed 证书、cone-ray 三维重建和公平强基线验证。

### 2.4 DeepONet、FNO 与 F-FNO 是待运行基线

DeepONet 通过 branch/trunk 结构学习函数空间之间的非线性算子；FNO 在频域参数化积分核；
F-FNO 使用可分离谱层等结构降低频域模型的计算负担。这些方法在各自论文任务中的结果不能直接
迁移成 BOST 结论。当前仓库结果没有在同一三维数据、几何切分、参数预算、训练预算、VJP 数和
墙钟预算下运行三者，因此本文不能报告任何相对优越性。

### 2.5 与何远哲方向的对齐方式

本方向与 NeRIF/TDBOST 的合理接口不是另起一个脱离实验的通用网络，而是：

1. 为 NeRIF 型折射率表示提供与实际中央差分/曲光线 renderer 一致的可微测量算子；
2. 评估直线、OCBH、Picard 与完整曲光线在三维重建中的误差和预算；
3. 将时空低秩或神经算子只用于难以由低阶物理解释的 remainder；
4. 在 TDBOST 公开代码及师兄实际 distortion module 上做重叠审计，避免重复已有实现。

---

## 3. 离散 bend-homotopy 方法

### 3.1 轨迹同伦

引入标量 $\epsilon\in[0,1]$，只控制轨迹弯曲：

$$
\frac{\mathrm d r_\epsilon}{\mathrm ds}=u(q_\epsilon),
\qquad
\frac{\mathrm d q_\epsilon}{\mathrm ds}
=\epsilon F_\Delta(r_\epsilon,q_\epsilon).
$$

观测 integrand 始终使用完整 $F_\Delta$，不再乘 $\epsilon$：

$$
Y_s(\epsilon)=\Pi h\sum_j
F_\Delta\!\left(
\frac{r_{\epsilon,j}+r_{\epsilon,j+1}}2,
\frac{q_{\epsilon,j}+q_{\epsilon,j+1}}2
\right).
$$

在同一步数和同一离散规则下，

$$
Y_s(0)=M_s,\qquad Y_s(1)=H_s.
$$

定义一阶离散轨迹缺陷

$$
D_s=\left.\frac{\mathrm dY_s(\epsilon)}{\mathrm d\epsilon}\right|_{\epsilon=0},
\qquad
C_s=M_s+D_s.
$$

$C_s$ 是从 $\epsilon=0$ 到 $\epsilon=1$ 的一阶外推，不是完整曲光线精确解。

### 3.2 轨迹切线

令

$$
\delta r=\left.\frac{\partial r_\epsilon}{\partial\epsilon}\right|_0,
\qquad
\delta q=\left.\frac{\partial q_\epsilon}{\partial\epsilon}\right|_0.
$$

由于

$$
\left.\frac{\mathrm d}{\mathrm d\epsilon}
\left[\epsilon F_\Delta(r_\epsilon,q_\epsilon)\right]
\right|_{\epsilon=0}=F_\Delta(r_0,q_0),
$$

一阶轨迹方向切线的 forcing 是 $F_0$。旧 N1 将 $A\delta r+B\delta q$ 反馈同时放入轨迹
切线，相当于另一种 affine/Newton-like 直线路径修正，不是上述 bend-homotopy 的精确一阶导数。
在实现中，切线不是先写连续方程再换求积器，而是逐项匹配实际 RK4 stage、步后归一化和中点输出
的离散程序。

### 3.3 同一中央差分算子的输出 Jacobian

虽然 $A\delta r+B\delta q$ 不应进入该同伦的一阶轨迹 forcing，它们仍出现在最终观测的路径导数。
在每个离散采样点，坐标 Jacobian 为

$$
A_\Delta
=P\left(
\frac{Dg_\Delta}{n}
-\frac{g_\Delta(\nabla n)^\mathsf T}{n^2}
\right),
$$

方向 Jacobian 为

$$
B_\Delta
=-\frac{\left[(q\cdot g_\Delta)I+qg_\Delta^\mathsf T\right]P}{n},
$$

其中 $Dg_\Delta$ 必须对高保真程序实际使用的中央差分梯度求导，分母 $n$ 的导数来自同一标量
插值原语。用当前位置的 automatic Hessian 替换 $Dg_\Delta$ 会改变被求导的离散算子。

### 3.4 fail-closed 与当前实现范围

当前 OCBH 实现检查 float64、有限值、折射率正下界、domain/stencil margin、位置和方向扰动上限。
违反声明合同的路径不应被悄悄截断或继续外推。当前结果仍缺少 cell/topology/caustic 与有限差分
remainder 的完整证书；因此“fail-closed”只是局部数值合同，不是对真实光学安全性的证明。

---

## 4. Forward-JVP teacher 与解析 OCBH

### 4.1 Forward-JVP teacher

教师实现把 dual input $(\epsilon,\dot\epsilon)=(0,1)$ 送入完整离散程序，并通过 forward-mode
JVP 对以下操作整体求导：

1. 每步四个 RK4 stage；
2. 每个 stage 的中央差分场查询；
3. 每完整步后的方向归一化；
4. endpoint arithmetic mean 定义的中点观测；
5. 始终不乘 $\epsilon$ 的物理输出 integrand。

教师不把 H128/H256 输出作为输入，但它执行完整 forward-mode 程序，定位是正确性 oracle 和诊断
工具，不是部署候选。

### 4.2 解析 OCBH

在 $\epsilon=0$ 时，primal path 为直线。解析 OCBH 将所有 RK4 endpoint 和 midpoint 合并为
batched coefficient path，使用：

- 1 个标量场 batch；
- 6 个中央差分 offset batch；
- 4 个 coordinate reverse sweep；
- 解析 RK4 tangent propagation；
- 0 次 exact-high evaluation；
- 0 次 reverse-mode field VJP。

在 64 rays、128 步的每个九格 case 中，机器账本记录：

| 路线 | logical scalar-grid point queries | 说明 |
|---|---:|---|
| OCBH | `115136` | 7 个 batched interpolation dispatch |
| forward-JVP teacher | `286720` | 1 次 forward-mode bend JVP |
| H128 | `286720` | 1 次 exact-high evaluation |

OCBH/H128 的逻辑 point-query ratio 为 `0.4015625`。这一账本没有计量 peak RSS、host scalar
synchronization 或 field JVP/VJP，也不代表硬件无关复杂度。

### 4.3 教师等价结果

九格中解析 OCBH 与 forward-JVP teacher 的最坏值如下：

| 对象 | 最坏 relative-L2 | 最坏 case |
|---|---:|---|
| detector output tangent | `2.1591563693023527e-14` | wrinkled-wide, `10x` |
| position tangent | `4.950399402606913e-15` | smooth-wide, `10x` |
| direction tangent | `5.379379355283756e-15` | smooth-wide, `3x` |

三类误差的 development 阈值均为 `2e-10`，九格教师检查均通过。该结果证明两个实现对当前离散
目标等价，不证明该离散目标等于真实实验光学。

---

## 5. Picard 强基线

### 5.1 更新规则

Picard 基线从校准直线路径 $(r^0,q^0)$ 出发。第 $k$ 次 sweep 在上一条路径的中点冻结曲率，
先更新方向，再使用新方向重建位置：

$$
q_i^{k}=\operatorname{normalize}\!\left(
q_{\mathrm{in}}+h\sum_{j<i}F_\Delta(r_{j+1/2}^{k-1},q_{j+1/2}^{k-1})
\right),
$$

$$
r_i^{k}=r_{\mathrm{in}}+h\sum_{j<i}
\operatorname{normalize}(q_j^{k}+q_{j+1}^{k}).
$$

这是一种 direction-first frozen-midpoint Picard/Gauss-Seidel 更新。它没有调用 exact RK4 high，
也没有给出焦散区收敛证明。

### 5.2 Off-by-one 反例与修复

若完成第 $k$ 次路径更新后仍返回更新前路径上的观测，则名义 Picard-1 实际仍是 straight-path
output。当前机器结果使用已修复合同：完成指定 sweep 后，在最终更新路径上再执行一次七点曲率
batch 以计算输出。

| 方法 | 路径更新 batch | 最终输出 batch | total field point queries |
|---|---:|---:|---:|
| Picard-1 | `1` | `1` | `114688` |
| Picard-2 | `2` | `1` | `172032` |

相对 H128 的 `286720` 次 logical field-point queries，对应比值分别为 `0.4` 和 `0.6`。

### 5.3 为什么它是不可绕开的基线

Picard-1/2 没有复杂 Jacobian，也不需要神经网络。在当前弱弯曲 synthetic domain 中，它们用
一次或两次路径重采样就可能吸收大部分轨迹非线性。如果新方法不能在更困难且物理有效的 domain、
三维反演、可微训练或 fail-closed 风险路由中超过 Picard，那么增加网络或高阶导数没有充分意义。

---

## 6. 九格 development protocol

### 6.1 Cell 构成

当前九格由三个 case 和三个 dimensionless stress multiplier 的笛卡尔积构成：

| case ID | phantom family | phantom seed | stress multiplier | refractivity scale |
|---|---|---:|---|---|
| `smooth_narrow_aperture` | `smooth_plume` | `1729` | `1/3/10` | `0.0003/0.0009/0.003` |
| `smooth_wide_aperture` | `smooth_plume` | `1729` | `1/3/10` | `0.0003/0.0009/0.003` |
| `wrinkled_wide_aperture` | `wrinkled_density_interface` | `2718` | `1/3/10` | `0.0003/0.0009/0.003` |

每格使用 64 rays；候选与 H128 execution 均为 128 步，H256 evaluator 为 256 步，H512 只用于
development convergence sentinel。`oblique_compression_sheet` 和 `shock_expansion_pair` 两个
reserved family 没有打开。

### 6.2 方法与角色

| 方法 | 角色 | 是否拥有 OCBH primary gate |
|---|---|---|
| continuous affine N1 | 历史诊断基线 | 否 |
| OCBH | 当前主候选 | 是 |
| Picard-1 | 强逐次逼近基线 | 否，仅描述性比较 |
| Picard-2 | 强逐次逼近基线 | 否，仅描述性比较 |
| H128 | execution high route | 不适用 |
| H256 | synthetic evaluator | 不适用 |
| H512 | reference sentinel | 不适用 |

Picard 行在 `metrics.csv` 中没有 OCBH gate 字段；这不表示 Picard 失败，也不能把 Picard 写成
“9/9 过 OCBH 门”。

### 6.3 冻结门

OCBH 当前关键阈值为：

| 指标 | development threshold |
|---|---:|
| teacher output/position/direction relative-L2 | `<= 2e-10` |
| base output relative-L2 | `<= 2e-10` |
| matched residual prediction relative-L2 | `<= 0.02` |
| corrected residual variance ratio | `<= 0.01` |
| per-ray risk Spearman | `>= 0.99` |
| valid ray fraction | `>= 1.0` |
| candidate-to-H256 relative-L2 | `<= 0.002` |
| global reference no-harm ratio | `<= 1.1` |
| Q95 reference no-harm ratio | `<= 1.1` |
| H256-to-H512 output relative-L2 | `<= 0.0001` |
| H256-to-H512 matched-residual relative-L2 | `<= 0.01` |
| candidate p90 / H128 p10 wall-time ratio | `<= 0.25` |
| candidate / H128 logical point-query ratio | `<= 0.45` |

这些阈值是本项目的 development engineering gates，不是文献定理，也没有经过独立审稿确认。

### 6.4 统计单位

当前 64 rays 是同一合成场与同一 rig 中的 ray population，不是 64 个独立物理重复。当前每个
phantom family 只有一个已使用 seed，因此不能把 ray-level 分布当作 field-level 泛化统计。
下一阶段必须以 field seed 为重复单位，并按 field family 分组汇报尾部。

---

## 7. 结果

### 7.1 最坏格汇总

下表对每个方法在九格中取最坏值。matched residual、global no-harm、Q95 no-harm 和墙钟比越低
越好；risk Spearman 越高越好。no-harm 比值的分母是 H128 相对 H256 的误差，`1` 表示与 H128
持平。

| 方法 | matched residual rel-L2 | global no-harm | Q95 no-harm | risk Spearman 最小值 | 最坏 p90/H128-p10 |
|---|---:|---:|---:|---:|---:|
| continuous affine N1 | `0.06854056237083514` | `1.7736041210188718` | `1.687675173974484` | `0.9259157509157508` | `0.09557508928224155` |
| OCBH | `0.013371097842020507` | `1.0644073595295993` | `1.0561120759599356` | `0.9986721611721611` | `0.15078973281471242` |
| Picard-1 | `0.0017091774596397962` | `1.0010147173920803` | `1.0009439748033466` | `0.9998168498168497` | `0.025417265529224447` |
| Picard-2 | `0.0004982646139615191` | `1.000986085720339` | `1.0009083760036501` | `0.9999084249084248` | `0.037156454198213146` |

Picard 的 risk 定义是 $\lVert P-M\rVert$ 的描述性排序，不是已验证的可观测 router。不同方法的
risk Spearman 不应被解释为完全相同的安全证书。

### 7.2 OCBH 主门与机器判定

OCBH 的主候选格为 `9/9`，教师等价检查格为 `9/9`，H256/H512 sentinel 为 `9/9`，逻辑 query
检查为 `9/9`，计时 rig 为 `3/3`。机器仍然给出：

```text
MECHANISM_BRIDGE_SIGNAL_ONLY_96_CELL_RECONSTRUCTION_AND_REAL_DATA_GATES_CLOSED
```

这表示“机制桥接信号存在，但下一阶段门保持关闭”，不是 audit authorization。

### 7.3 Reference sentinel

H256 对 H512 的最坏 development sentinel 为：

| 指标 | 最坏值 | 最坏 case |
|---|---:|---|
| full output relative-L2 | `4.3648222131298665e-05` | wrinkled-wide, `3x` |
| matched residual relative-L2 | `0.004257434544404077` | wrinkled-wide, `1x` |

两项均满足当前 sentinel 阈值，因此 H256 可继续作为这九格的 synthetic evaluator。该收敛检查
只比较两个数值离散层级，不能把 H512 升格为真实物理真值。

### 7.4 墙钟结果

三个 stress=`1x` rig 各交错测量 20 次；候选取 p90，H128 取同 rig 的 p10。各方法最坏比值及
对应候选 p90 为：

| 方法 | 最坏 p90/H128-p10 | 对应 p90 seconds |
|---|---:|---:|
| continuous affine N1 | `0.09557508928224155` | `0.044062550600000014` |
| OCBH | `0.15078973281471242` | `0.0695463453` |
| Picard-1 | `0.025417265529224447` | `0.0117228003` |
| Picard-2 | `0.037156454198213146` | `0.017137079200000002` |
| H128 | `1.0168323759533031` | `0.46897739130000005` |

环境为 local Mac CPU、PyTorch float64、单进程共享候选与参考。当前未测 peak RSS，未计 field
JVP/VJP，未显式统计 host scalar synchronization。该表只能描述当前实现。

---

## 8. 反面结果

### 8.1 旧 N1 的“更完整线性化”反而不是目标导数

旧 N1 在轨迹切线中加入 $A\delta r+B\delta q$，看似比 $\delta q'=F_0$ 更完整，但它求解的
不是 bend-homotopy 在 $\epsilon=0$ 的导数。其最坏 matched residual 为
`0.06854056237083514`，最坏 global/Q95 no-harm 为 `1.7736041210188718` 和
`1.687675173974484`。这说明线性化项更多不等于离散目标更一致。

### 8.2 OCBH 修正机制错误，但没有成为最佳算法

OCBH 将最坏 matched residual 降至 `0.013371097842020507`，并通过当前九格主门；但 Picard-1/2
进一步降至 `0.0017091774596397962` 和 `0.0004982646139615191`，墙钟也明显更低。因而不能写
“OCBH 击败强基线”。当前更合理的假设是：弱弯曲域里，一两次路径重采样已经足够，一阶 Jacobian
的额外结构尚未转化为前向精度或速度优势。

### 8.3 机器精度等价不是物理正确性

`2.1591563693023527e-14` 证明解析 OCBH 与 forward-JVP teacher 对同一程序一致。若两者共享
同一个错误的场插值、边界条件或相机模型，它们仍会共同错误。因此教师等价只能关闭实现层面的
一条疑问，不能关闭模型失配。

### 8.4 当前 headroom 可能过小

Picard 的 global no-harm 最坏值已接近 `1`。在当前九格继续训练复杂 residual network，可能只会
学习数值微差或 field seed 特征。进入神经学习前，必须先在更强但仍满足 domain/topology 合同的
开发域证明 $H-P1$ 或 $P2-P1$ 存在稳定、可预测且有现实成本价值的信号。

### 8.5 当前“wide aperture”仍不是 cone-ray 测量

九格中的 ray population 没有按像素聚合成有限孔径观测，也没有真实 f-number、共焦背景点、
遮挡和图像域 optical flow。任何单射线结果都不能替代 cone-level 误差和 pupil 尾部检查。

---

## 9. 局限性

1. **合成域过窄。** 仅 3 个 case、2 个已打开 family 和每 family 一个 field seed。
2. **无独立 audit。** reserved families 明确未打开，candidate 与 evaluator 仍共享进程。
3. **无三维反演。** 当前只评价 measurement-operator output，没有 field relative-L2、H1、界面或
   held-out-view reconstruction 指标。
4. **无真实数据。** 没有 OERF 相机标定、flow-off/on 重复、真实噪声、体场真值或独立物理终点。
5. **非 cone-ray。** 当前没有 per-pixel aperture integration、薄透镜焦点和 image formation。
6. **仅一阶近似。** OCBH 没有二阶 remainder 上界，也没有焦散和路径拓扑改变证书。
7. **当前实现 detached。** 尚未完成对三维 field parameter 的 JVP/VJP dot test 和端到端反传。
8. **计时不完整。** 未测 peak RSS、host synchronization、编译/预处理和训练期 VJP 成本。
9. **神经强基线缺失。** DeepONet、FNO、F-FNO/FFNO 尚未在相同任务和预算下运行。
10. **无统计泛化结论。** rays 不是独立 field repeats；当前不能给置信区间或跨分布成功率。
11. **新颖性未授权。** 一阶曲光线、dynamic ray tracing、可微曲光线、神经 RI 场、cone-ray 和
    四维张量 BOST 都已有直接文献先例。
12. **物理量未标定。** 当前 normalized output 不能换写成真实像素、密度、温度或实验误差。

---

## 10. 下一阶段硬门

### 10.1 96-cell grouped-by-field development screen

在打开 reserved family 前，按现有报告冻结：

```text
2 field families
x 2 rig orientation packages
x 2 aperture packages
x 3 stress levels
x 4 field seeds
= 96 physical cells
```

field seed 是统计重复单位。每格至少比较 N1、OCBH、Picard-1、Picard-2、H128 和 H256，并继续
报告 matched residual、absolute error、global/Q95 no-harm、valid fraction、domain/stencil margin、
cost 与 false-safe。结果必须按 family/rig/stress 分组展示，不能只报总体均值。

**进入下一门的条件：** OCBH 的潜在角色必须在 96-cell 中稳定。如果 Picard 继续同时支配精度、
尾部和成本，则停止把 OCBH 当前向候选，只保留教师、风险特征或可微支架角色。

### 10.2 Cell/topology/caustic 与 remainder certificate

需要显式检测：

- 中央差分 stencil 是否跨越不连续单元或支持边界；
- 路径是否发生拓扑变化、ray crossing 或 caustic；
- $\epsilon=1$ 是否超出一阶外推的可信域；
- $\lVert P1-(M+D)\rVert$、$\lVert P2-P1\rVert$ 是否能构成可观测 remainder feature；
- 无法证明安全时是否能 fail closed 到 Picard-2 或 H128。

### 10.3 三维 reconstruction gate

必须先完成 field-level JVP/VJP dot test，再将 OCBH/Picard/curved-high 接到同一个重建目标。最低
协议应包含：

- 6 个训练视角与 2 个 held-out 视角；
- 相同初始化、相同正则项、相同停止规则和相同 forward/VJP 预算；
- field relative-L2、gradient/H1 error、held-out reprojection、tail error 和墙钟；
- 直线、Picard-1、Picard-2、OCBH 与 exact-curved renderer 的逐项消融；
- 不把训练视角数据一致性等同于体场正确性。

若 OCBH 仅在 forward prediction 有效、但 VJP 不一致或重建不稳定，则该路线不能进入论文主方法。

### 10.4 DeepONet/FNO/F-FNO(FFNO) 公平比较门

神经算子只在 96-cell 和可微重建合同稳定后启动。建议比较两类任务：

1. **Forward residual operator：** 学习 $H-P1$ 或 $P2-P1$，而不是从零预测完整 $H$。
2. **Inverse/reconstruction operator：** 从多视角观测与几何条件到三维场，作为迭代重建的对照或
   warm start，而不是跳过真实 forward residual 检查。

公平性至少冻结：

- 完全相同的 train/validation/fresh geometry-OOD 切分；
- 相同可用物理输入和相同真值泄漏边界；
- 参数量、优化步数、训练 wall time、peak memory 和 forward/VJP 调用；
- DeepONet 的 branch sensor 与 trunk query 合同；
- FNO/F-FNO 的网格、mode、padding 和 resolution transfer 合同；
- field、held-out reprojection、尾部和校准指标；
- 简单 MLP、线性/低秩 residual、Picard-1/2 与 exact-curved 强基线。

在这些基线实际运行前，标题、摘要和结论中不得出现“优于 DeepONet/FNO/FFNO”。

### 10.5 Cone-ray gate

必须区分 `pinhole-straight`、`pinhole-curved`、`cone-straight` 与 `cone-curved` 四个象限。最小
cone-ray 基线需要：共同背景焦点、固定 Sobol pupil prefix、per-subray validity、孔径积分收敛、
遮挡权重、cone mean 与 pupil Q95 同时报表。只有 flow-off 共焦和 image/deflection formation 合同
明确后，才能使用“finite-aperture BOST”表述。

### 10.6 Reserved-family audit gate

96-cell 的方法、阈值、代码 hash、绘图模板和停止规则冻结后，才允许打开
`oblique_compression_sheet` 与 `shock_expansion_pair`。打开后不得调参再回写 development 门；若失败，
应报告失败并重新启动新的版本化协议，而不是覆盖原结果。

### 10.7 Real-data gate

进入 OERF 真实数据前，需要何远哲师兄确认并冻结：

1. 相机内外参、镜头、孔径、背景距离、像元尺寸和坐标单位；
2. reference/flow-on 图像、光流或 distortion module 的定义；
3. flow-off 重复、噪声、坏点、遮挡、mask 和同步误差；
4. train/validation/fresh session 与 held-out camera 的切分；
5. 是否存在 CFD、PIV、压力、热电偶或其他独立物理终点；
6. NeRIF/TDBOST 已有 renderer、loss、时空因子和 distortion correction 的重叠边界；
7. 数据使用、论文署名、代码开放和派生物共享权限。

若真实实验没有体场真值，只能报告 held-out image/reprojection consistency、跨 session 稳定性和独立
物理量一致性，不能报告实验 field relative-L2。

---

## 11. 暂定研究假设与停止规则

### 11.1 可继续检验的假设

**H1：** 算子一致 bend-homotopy 导数可作为 Picard remainder 的可解释风险特征。
**H2：** 在更强但无焦散的域中，$H-P1$ 比完整 $H$ 更适合由小型 neural operator 学习。
**H3：** 在三维重建中，OCBH 的同算子 VJP 可能比 detached 前向近似更有价值。
**H4：** cone-ray averaging 会改变单射线误差排序，因此需要 cone-level router 而不是照搬单射线风险。

这些是假设，不是结果。

### 11.2 明确停止规则

出现以下任一情况，应收缩或停止 OCBH 主线：

1. 96-cell 中 Picard-1/2 在精度、尾部、有效率和成本上持续支配 OCBH；
2. OCBH 的风险特征不能预测 $P1$ 失败或不能降低 false-safe；
3. field JVP/VJP dot test 失败，或三维重建梯度与 exact-curved 不一致；
4. cone-ray、界面或真实几何下一阶外推频繁越界且无法可靠 fail closed；
5. 与 NeRIF/TDBOST 的代码和方法审计显示候选贡献已被覆盖；
6. 等预算 neural/operator-free baseline 已达到同等结果，复杂模型没有现实收益。

负结果仍应保留并报告，因为它能防止把计算资源投入到无 headroom 的模型堆叠。

---

## 12. 数据与代码可用性

### 12.1 当前可用内容

当前 development 结果由仓库内合成场 generator 产生，不依赖受限实验 PDF 或未获授权的 OERF
数据。可复现入口为：

- 配置：`demo_t16_operator/configs/n2_pvgr_n2_operator_consistent_bridge_v1.json`
- 正式 runner：`demo_t16_operator/run_n2_pvgr_n2_operator_consistent_bridge.py`
- OCBH：`demo_t16_operator/operator_consistent_homotopy_predictor.py`
- forward-JVP teacher：`demo_t16_operator/discrete_rk4_jvp_predictor.py`
- Picard：`demo_t16_operator/picard_curved_ray_baseline.py`
- 完整机器结果：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/`
- 哈希账本：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/manifest.json`

`manifest.json` 为配置、源配置、前序结果、runner、三种方法实现、结果表和图记录 SHA-256 与字节数。
在正式复现实验前，应先验证 manifest，再在隔离环境重跑，而不是只复用当前生成图。

### 12.2 当前不可用内容

- 没有用于本文结果的真实 OERF 原始数据；
- 没有公开或再分发任何受限订阅论文 PDF；
- 没有真实相机标定、体场真值或独立物理终点；
- 没有训练好的 DeepONet/FNO/F-FNO checkpoint；
- 没有 reserved-family audit 结果；
- 没有投稿级匿名代码包、环境锁文件或独立机器复现。

正式投稿的数据与代码可用性声明必须在权限、匿名审稿和组内政策确定后重写。

---

## 13. 伦理与可复现性

1. **诚实标注证据。** synthetic、development、reserved、real 四级证据不得合并；九格过门不能写成
   论文成功。
2. **保留反面结果。** Picard 优于 OCBH、旧 N1 失效和 off-by-one 修复都属于核心结果，不得从
   投稿图表中删除。
3. **防止数据泄漏。** 96-cell development、reserved audit 与真实 fresh session 必须使用不同
   field/session 单位；不能把同一场中的 rays 随机拆成 train/test 来声称泛化。
4. **预注册后开盲集。** reserved family 只在方法、阈值、代码和图表模板冻结后打开。
5. **报告完整成本。** 除 wall time 外，还应报告场查询、dispatch、JVP/VJP、peak RSS、host sync、
   训练成本与失败回退成本。
6. **禁止结果选择。** 每个 case/seed/rig 都进入逐格表，不能只展示最好看的 flame 或 shock 切片。
7. **权限边界。** 未公开组内数据、标定和论文材料仅在获批环境使用；没有明确授权时不得上传到
   GitHub Pages 或公开仓库。
8. **作者与贡献。** 最终作者顺序、数据贡献、软件贡献和导师责任由课题组按实际贡献确认；本工作稿
   不预设作者名单。
9. **独立复核。** 投稿前至少需要另一台机器重跑、哈希核对、数值容差审计、公式到代码逐项映射和
   导师/师兄方法重叠审查。
10. **无虚构。** 本文不得填入尚未运行的 96-cell、三维重建、神经算子或真实数据数字。

---

## 14. 当前可写结论

在当前九个 float64 合成 development cell 中，解析 OCBH 与完整 forward-mode 离散 JVP 在约
`2.16e-14` 的最坏输出 relative-L2 内一致。这说明离散 bend-homotopy 的 trajectory-only tangent
及同一中央差分算子的输出 Jacobian 已得到实现层验证。相较旧 continuous affine N1，OCBH 明显
降低当前九格的 residual 与 reference-tail 失配。

但 Picard-1/2 在同九格的 matched residual 与当前 CPU worst-case 墙钟汇总上都优于 OCBH，
直接否定了“OCBH 已是最佳前向算法”的叙事；这不等于 Picard 在每个 no-harm 指标的每一格都占优。当前最有价值
的研究问题不再是继续放大 OCBH，而是检验它能否作为 Picard remainder 的可解释风险特征、
算子一致的 field VJP 支架或 cone-ray 三维重建中的 fail-closed 路由组件。

在 96-cell、三维 reconstruction、DeepONet/FNO/F-FNO、cone-ray、reserved family 和真实数据门
关闭期间，本稿保持：

> **WORKING DRAFT / NOT SUBMISSION READY / NO PAPER-LEVEL SUCCESS CLAIM**

---

## 15. 一级来源

以下仅列与当前物理、方法边界和下一阶段基线直接相关的一级来源。链接指向出版社、正式会议论文页、
作者公开原稿或作者代码。

1. M. Raffel, “Background-oriented schlieren (BOS) techniques,” *Experiments in Fluids*, 2015.
   [Springer 正式页](https://link.springer.com/article/10.1007/s00348-015-1927-5)；
   [DOI](https://doi.org/10.1007/s00348-015-1927-5)

2. S. J. Norton, “Computing ray trajectories between two points: a solution to the ray-linking problem,”
   *Journal of the Optical Society of America A*, 1987.
   [Optica 正式页](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-10-1919)

3. V. Červený, L. Klimeš, and I. Pšenčík, “Paraxial ray approximations in the computation of
   seismic wavefields in inhomogeneous media,” *Geophysical Journal International*, 1984.
   [Oxford Academic 正式页](https://academic.oup.com/gji/article/79/1/89/601880)；
   [DOI](https://doi.org/10.1111/j.1365-246X.1984.tb02843.x)

4. J. P. Molnar et al., “Forward and inverse modeling of depth-of-field effects in
   background-oriented schlieren,” *AIAA Journal*.
   [AIAA DOI](https://doi.org/10.2514/1.J064095)；
   [作者公开原稿](https://arxiv.org/html/2402.15954)

5. B. Zhao et al., “Single View Refractive Index Tomography with Neural Fields,” *CVPR 2024*.
   [CVF 正式开放页](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)

6. Y. He et al., “Neural refractive index field: Unlocking the potential of background-oriented
   schlieren tomography in volumetric flow visualization,” *Physics of Fluids*, 2025.
   [AIP 正式页](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)；
   [DOI](https://doi.org/10.1063/5.0250899)；
   [作者公开原稿](https://arxiv.org/html/2409.14722v2)

7. Y. Zheng et al., “Instantaneous refractive index compensation on the velocity measurement using
   simultaneous PIV-BOST,” *Experiments in Fluids*, 2025.
   [Springer 正式页](https://link.springer.com/article/10.1007/s00348-025-04093-y)；
   [DOI](https://doi.org/10.1007/s00348-025-04093-y)

8. Y. He et al., “Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren
   Tomography for High-Speed, High-Fidelity Flow Field Reconstruction,” *ACM Transactions on
   Graphics*, 2026.
   [ACM DOI](https://doi.org/10.1145/3809488)；
   [作者代码](https://github.com/Hyz617/TDBOST)

9. L. Lu et al., “Learning nonlinear operators via DeepONet based on the universal approximation
   theorem of operators,” *Nature Machine Intelligence*, 2021.
   [Nature DOI](https://doi.org/10.1038/s42256-021-00302-5)；
   [作者公开原稿](https://arxiv.org/abs/1910.03193)

10. Z. Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations,”
    *ICLR 2021*.
    [OpenReview 正式论文](https://openreview.net/forum?id=c8P9NQVtmnO)；
    [作者公开原稿](https://arxiv.org/abs/2010.08895)

11. A. Tran et al., “Factorized Fourier Neural Operators,” 2021.
    [作者公开原稿](https://arxiv.org/abs/2111.13802)

12. S. Lunz et al., “On Learned Operator Correction in Inverse Problems,”
    *SIAM Journal on Imaging Sciences*, 2021.
    [SIAM 正式页](https://epubs.siam.org/doi/10.1137/20M1338460)；
    [DOI](https://doi.org/10.1137/20M1338460)

13. S. Grauer et al., “Instantaneous 3D Flame Imaging by Background-Oriented Schlieren
    Tomography,” *Combustion and Flame*, 2018.
    [Elsevier 正式页](https://www.sciencedirect.com/science/article/pii/S0010218018302694)；
    [DOI](https://doi.org/10.1016/j.combustflame.2018.06.022)

14. R. Molinaro et al., “Neural Inverse Operators for Solving PDE Inverse Problems,” *ICML 2023*.
    [PMLR 正式论文页](https://proceedings.mlr.press/v202/molinaro23a.html)

---

## 附录 A. 本工作稿的数字核对规则

1. 方法最坏值从 `result.json.method_rows` 按 `method_id` 分组后取 max/min。
2. 教师等价误差从 `teacher_metrics.csv` 与 `result.json.teacher_rows` 交叉核对。
3. H256/H512 收敛值从 `reference_sentinel.csv` 与 `result.json.reference_sentinel_rows` 交叉核对。
4. 墙钟值从 `timing.csv` 读取；候选使用 p90，分母使用同 rig H128 p10。
5. OCBH 的 `9/9` 仅指其冻结 primary gates；Picard 没有套用该 gate schema。
6. 所有正文结果均为 development synthetic evidence，不从图像手工抄数。
7. 后续任何新实验必须新建版本化 result directory，不得覆盖本稿所引用的 v1 结果。

## 附录 B. 投稿前必须删除或替换的工作稿内容

- 删除页首 `WORKING DRAFT` 的前提是 paper claim authorization 真正变为 `true`；
- 将候选标题替换为导师确认且数据库级新颖性检索支持的正式标题；
- 用 96-cell、三维重建、神经强基线、reserved audit 和真实数据结果替换“下一阶段”占位；
- 增加正式作者、单位、基金、利益冲突、数据许可和贡献声明；
- 增加完整实验设置、超参数、硬件、统计检验、消融、失败案例和图表；
- 将所有内部路径改为匿名、可归档、带版本和 DOI/commit hash 的公开材料；
- 经何远哲师兄与蔡伟伟老师核对 TDBOST/NeRIF 重叠边界后再写贡献点。
