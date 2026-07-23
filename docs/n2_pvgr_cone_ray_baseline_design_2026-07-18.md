# N2-PVGR cone-ray / BOS 强基线一级来源审计与最小实现设计

> 日期：2026-07-18
> 状态：设计与一级来源审计，不是算法结果
> 适用主线：N2-PVGR 的有限孔径、曲光线、离散 JVP 与 Picard 对照
> 证据边界：只使用出版社页面、官方机构页面、作者公开原稿和作者公开代码进行事实核验；没有下载、复制或发布受限 PDF 正文。

## 0. 一页结论

1. **cone-ray 和曲光线不是同一个问题。** cone-ray 解决“一个像素通过有限孔径接收一束射线”的孔径积分；曲光线解决“一条射线在非均匀折射率场内怎样弯曲”。二者构成两个独立维度，至少要比较 `pinhole-straight`、`pinhole-curved`、`cone-straight`、`cone-curved` 四个象限。
2. **Molnar 等已经提出并验证了 BOS finite-aperture cone-ray forward/inverse。** 因而不能声称“首次将 cone-ray 用于 BOS/BOST”“首次考虑 BOS 景深”或“首次用神经场做 cone-ray BOST”。
3. **Norton 1987 解决的是固定两端点 ray-linking。** 该文给出隐式积分方程、逐次逼近和一阶折射率扰动路径；它可作为 Picard/一阶路径基线的理论祖先，但不是有限孔径平均，也不能把当前初值 RK4 直接称为 Norton 算法。
4. **当前 synthetic rig 已有 pupil 状态、圆盘采样、直线/曲线单射线和 128/256 步积分，却没有显式的“像素分组后孔径平均”输出，也没有薄透镜焦点、背景图像、光流、遮挡和真实 f-number。** 因而可以立即实现一个诚实的 `synthetic aperture-bundle baseline`，但只有在通过 flow-off 共焦门后，才能把它称为 Molnar 意义下的 cone-ray baseline。
5. **最小强基线 CRB-v0 应是外层孔径积分包装器，而不是新网络。** 对同一组 Sobol pupil 样本，分别包裹 straight、full curved RK4、N1 variational、discrete RK4-JVP、Picard-1 和 Picard-2；所有方法输出同一个 per-pixel cone average，并与同一高精度 curved-cone reference 比较。
6. **当前最值得检验的科学问题不是“cone-ray 会不会平均得更平滑”，而是：** 宽孔径、皱褶前沿下，孔径平均会放大还是掩盖 N1 的两项 reference no-harm 失败；单射线的误差排序能否可靠预测 cone-level 误差；离散 JVP/Picard 能否以更少的高保真路径调用保持前沿和尾部安全。

机器可执行前的硬结论：

> `DESIGN_READY / IMPLEMENTATION_AND_CONVERGENCE_NOT_YET_RUN / NO NOVELTY OR REAL-BOST CLAIM`

## 1. 名词先拆开

### 1.1 BOS、BOST 与观测量

BOS 用参考图和流场存在时的畸变图估计背景纹理位移。Raffel 的开放综述把小角度、近轴条件下的偏折写成折射率横向梯度沿视线的积分；BOST 再把多个视角的位移或图像差异反演成三维折射率、密度或温度场。

几何光学中的单射线满足

$$
\frac{\mathrm d}{\mathrm ds}\bigl(n\mathbf d\bigr)=\nabla n,
\qquad \|\mathbf d\|_2=1,
$$

等价地，当前仓库的曲光线内核使用

$$
\mathbf r'=\mathbf d,
\qquad
\mathbf d'=\frac{(\mathbf I-\mathbf d\mathbf d^\mathsf T)\nabla n(\mathbf r)}{n(\mathbf r)}.
$$

相机平面上的两分量偏折可抽象为

$$
\boldsymbol\delta_{p,a}
=\mathbf\Pi_p\int_{\gamma_{p,a}}
\frac{(\mathbf I-\mathbf d\mathbf d^\mathsf T)\nabla n}{n}\,\mathrm ds,
$$

其中 `p` 是像素或 chief-ray 索引，`a` 是孔径位置，`Pi_p` 投影到像素的两个横向方向。这个式子描述的是**一条**给定射线。

### 1.2 单中心射线

单中心射线，也称 chief ray 或 pinhole/thin-ray 近似，只取孔径中心 `a=0`：

$$
\boldsymbol\delta_p^{\mathrm{thin}}=\boldsymbol\delta_{p,0}.
$$

它没有表达同一像素通过镜头不同位置所接收的多条光路。即使这条中心射线使用全非线性 RK4 弯曲积分，它仍然是 `pinhole-curved`，不是 cone-ray。

### 1.3 cone-ray

真实有限孔径对二维 pupil 区域 `A_p` 积分。最简单的 deflection-domain 模型为

$$
\overline{\boldsymbol\delta}_p
=\frac{\int_{A_p}w_p(a)\,\boldsymbol\delta_{p,a}\,\mathrm da}
{\int_{A_p}w_p(a)\,\mathrm da},
$$

其中 `w_p(a)` 可包含 pupil transmission、vignetting、occlusion 和辐射权重。均匀圆孔、无衰减时，可用 Sobol 点 `u_k,v_k in [0,1]^2` 映射到圆盘：

$$
a_k=R\sqrt{u_k}
\begin{bmatrix}\cos(2\pi v_k)\\\sin(2\pi v_k)\end{bmatrix},
\qquad
\widehat{\boldsymbol\delta}_p^{(K)}
=\frac1K\sum_{k=1}^{K}\boldsymbol\delta_{p,a_k}.
$$

`sqrt(u)` 不能省略，否则会在圆心过采样。当前 `joint_state_geometry` 已使用这一圆盘映射。

### 1.4 deflection average 还不等于完整 BOS 图像形成

Molnar 作者稿明确提醒：有限孔径的图像形成是背景强度经过不同光路后在孔径上的积分，非线性图像 warp 与孔径积分一般不交换。更完整的表达是

$$
I_p^{\mathrm{on}}
=\int_{A_p}w_p(a)\,I_{\mathrm{bg}}\bigl(q_{p,a}[n]\bigr)\,\mathrm da,
$$

再由 `I_off` 和 `I_on` 做 optical flow 或直接进入 UBOST。通常

$$
\operatorname{OF}\!\left(\int_A I_a\,\mathrm da\right)
\ne
\int_A\operatorname{OF}(I_a)\,\mathrm da.
$$

所以本设计分两级：

- **CRB-v0D：deflection-domain cone average。** 当前 synthetic rig 可实现，用来隔离孔径采样与路径弯曲；不能声称复现了完整相机图像。
- **CRB-v1I：image-domain cone BOS。** 需要背景图、薄透镜/标定模型、像素响应、遮挡、flow-off/on 图和 optical-flow/UBOST 接口；当前 synthetic rig 不具备。

### 1.5 四个物理象限

| 孔径 | 路径 | 简写 | 说明 |
|---|---|---|---|
| 单中心 | 直线 | `PS` | 经典 pinhole + paraxial/弱偏折底座 |
| 单中心 | 曲线 | `PC` | 只检查 ray bending，不检查有限孔径 |
| 有限孔径 | 每条直线 | `CS` | Molnar 型 cone-ray 的最小近轴版本，检查 depth-of-field |
| 有限孔径 | 每条曲线 | `CC` | 同时检查 pupil integration 与 field-dependent bending；是当前 N2 的高保真开发教师，不自动等于真实相机 |

必须禁止三个术语误用：

- `cone-ray` 不是 cone-beam CT；
- 文献中的 `CBOS` 通常是 Colored Background-Oriented Schlieren，不是 cone-BOS；
- `wide aperture` 不是 `strong ray bending` 的同义词。

## 2. Norton 1987 与 Picard 的准确位置

[Norton 1987 的 Optica 正式页](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-10-1919)核验了三个事实：

1. 问题是已知折射率分布和两端点 `a,b`，但未知从 `a` 出发、恰好到达 `b` 的初始方向，即 ray-linking 边值问题。
2. 作者把射线方程改写成满足边界条件的隐式积分方程，用 successive approximations 求真路径。
3. 作者还给出对折射率扰动一阶正确的显式路径，可作为逐次逼近初值。

这与当前 N2 的关系是：

- N1 的 frozen-straight variational predictor 与 Norton 的一阶路径都利用“小折射率扰动下直线路径附近的一阶修正”，但方程、边界条件和离散目标不同，不能说 N1 实现了 Norton。
- 当前 `trace_field_dependent_rays` 是给定入口位置和方向的**初值问题** RK4；它不强制曲线终点落在指定背景点，因此也不是 Norton ray-linking。
- 未来若只对当前 IVP 路径反复更新采样点，应命名为 `Picard-style IVP path iteration`。
- 只有显式固定 pupil 点和背景焦点、每轮检查 endpoint residual 并收敛，才可命名为 `Norton-style two-point ray-linking baseline`。

公平实验应同时保留两个版本，不能混名：

| 基线 | 边界合同 | 主要用途 |
|---|---|---|
| `P1/P2-IVP` | 固定入口状态，1/2 次路径重采样 | 与当前 RK4、discrete JVP 和 N1 直接比较 |
| `Norton-BVP` | 固定 pupil 点与背景点，检查 endpoint residual | 检查真实 focused bundle 的 ray-linking 误差 |

## 3. Molnar cone-ray 论文真正证明了什么

[AIAA Journal 正式 DOI 页面](https://doi.org/10.2514/1.J064095)和[作者开放原稿](https://arxiv.org/html/2402.15954)共同支持以下事实：

- 有限孔径让一个像素接收二维 aperture 上的一束光线，pinhole 模型把它错误压成一条 thin ray。
- 作者把孔径上的 ray bundle 积分称为 cone-ray model，并在 buoyancy-driven turbulence 与高超声速球绕流上做 forward/inverse 分析。
- 高超声速 forward 对每像素抽取 1000 个 pupil 样本，用非线性 ray tracing 后平均偏折；逆解每像素在 frustum 中采样 10,000 个点，并报告该采样的 forward 标准差处于长程值的 2% 内。
- 实验覆盖 `f/22, f/16, f/11, f/8, f/5.6, f/4`。作者报告孔径打开时 shock 偏折被抹平，cone-ray 逆解相对 pinhole 更能保留尖锐 shock；同时明确指出孔径越大，逆问题越模糊、噪声放大越严重，cone-ray 并非免费改进。
- 对球体遮挡，作者将被实体挡住的 rays 置零；遮挡会使 pupil 权重不再对称，不能简单删除无效 ray 后偷偷重新归一化。
- 作者同时比较 deflection-based 与 UBOS/image-based inversion，并指出后者避免独立 optical-flow 阶段带来的部分误差。

该论文**不能替本项目证明**：

- OERF 的镜头、焦距、背景距离和折射率幅值也处于同样的 finite-aperture 误差区间；
- 当前 dimensionless `aperture_radius=0.035/0.075` 可以换算为 `f/22...f/4`；
- 当前 64 条 synthetic pupil rays 已经收敛；
- 对 per-ray 有效的 N1/JVP/Picard 在孔径平均后仍然 no-harm；
- cone-ray 会改善所有流场，或可以忽略 optical-flow、遮挡与相机标定。

## 4. Raffel 综述给出的最低 BOS 合同

[Raffel 2015 开放综述](https://link.springer.com/article/10.1007/s00348-015-1927-5)确认：

- BOS 通过相关/光流比较背景点阵的参考图与畸变图；观测与折射率或密度横向梯度的视线积分相关。
- 小角度和近轴公式包含背景放大倍率、流场到背景距离等系统常数；只报告无量纲 `gradient integral` 还不是像素位移。
- BOST 是多视角 line-of-sight inverse problem；从位移到三维密度仍需要几何标定、反演和正则化。
- 小孔径可减少 image blur、提高空间分辨率，但需要更多照明；孔径选择是光量、景深、运动模糊和空间分辨率之间的实验折中。

因此，当前 N2 的 forward-only synthetic 结果最多支持“measurement-operator mechanism”。没有 `f`、像元尺寸、背景距离、放大率、flow-off/on 图和标定误差时，不能报告真实像素偏移、真实 f-number 或实验精度。

## 5. 当前仓库能做什么，不能做什么

### 5.1 已核对的现有合同

当前 N2-PVGR development 配置和实现提供：

| 项目 | 当前值/能力 |
|---|---|
| 场表示 | `17 x 17 x 17` float64 smoothstep scalar grid |
| 场家族 | smooth plume、wrinkled density interface；shock families 仍保留关闭 |
| 折射率应力 | base scale `3e-4`，乘数 `1, 3, 10` |
| pupil population | 每个 case 64 个二维 Sobol 状态 |
| 圆盘映射 | `r=sqrt(u)`、`theta=2*pi*v`，均匀面积采样 |
| aperture radius | narrow `0.035`；wide `0.075`，均为 synthetic dimensionless 值 |
| path | `path_half_length=0.62` |
| 差分 | central difference step `0.002` |
| 执行/参考步数 | 128 / 256 |
| 单射线 forward | frozen straight、field-dependent curved RK4、path-integrated deflection |
| 近似器 | N1 frozen-path variational defect predictor |
| 当前 N1 development 结果 | 7/9；wrinkled-wide 的 3x/10x reference no-harm 失败 |

### 5.2 现有 64 rays 的真实含义

当前 runner 把 64 个 pupil 状态当作 64 条独立样本，用于 per-ray residual、风险排序和方差。它没有把这些 rays 按像素聚合成一个 cone measurement。因此：

- 已有 7/9 结论是**单射线 population 的 measurement-operator 证据**；
- 它不是 64-ray cone reconstruction 的结果；
- 不能从 per-ray variance reduction 推出 cone-average error，因为正负误差可能在平均中抵消；
- 也不能只看 cone mean，因为平均可能掩盖 pupil 尾部失效、遮挡或 caustic。

### 5.3 当前几何还缺一项关键门：flow-off 共焦

物理 cone-ray 中，同一像素对应的 flow-off rays 应从 aperture 不同位置指向同一个背景焦点。当前 `joint_state_geometry` 用 `aperture_radius`、`cone_u` 和 `cone_z` 生成一个参数化 bundle，但没有显式镜头平面、背景平面、焦距或共同背景点。

所以分两步：

1. **立刻可做：`AB-D0` synthetic aperture-bundle average。** 直接对现有 64 条 rays 聚合，检验算法对“bundle averaging”的数值行为；必须标成 synthetic bundle，不能标 Molnar-equivalent cone-ray。
2. **强基线必须做：`CRB-v0D` focused-pupil cone。** 新增一个纯几何构造，使每个 pixel 的 pupil points 在 flow-off 条件下交于同一 background point，并检查交点残差；随后才运行 straight/curved/JVP/Picard。

最低共焦门建议为：

$$
e_{\mathrm{focus},p}
=\max_k\|\mathbf r_{p,k}(s_{\mathrm{bg}})-\mathbf b_p\|_2
\le 10^{-10}
$$

用于 float64 synthetic geometry。若将来换成真实标定，应改用像素或物理长度容差，并由标定不确定度决定，不能沿用 `1e-10`。

## 6. CRB-v0：可在当前 synthetic rig 上实现的最小强算法

### 6.1 输入

`CRB-v0` 的显式输入合同应包含：

| 输入 | 形状/类型 | 说明 |
|---|---|---|
| `values_zyx` | `[Z,Y,X]`, float64 | 当前折射率 primitive 的标量场；`n=1+scale*field` |
| `pixel_centres` | `[P,2]`, float64 | normalized detector/background transverse coordinates；单像素 rehearsal 可用 `P=1` |
| `rig` | frozen dataclass | view angle、path range、synthetic aperture radius 与横向基 |
| `pupil_states` | `[K,2]`, float64 | 固定 Sobol prefix；所有方法共用同一状态与 seed |
| `path_steps` | integer | 同一比较内固定 `S`；execution/reference 分开冻结 |
| `path_mode` | enum | `straight`, `curved-rk4`, `n1`, `discrete-jvp`, `picard-1`, `picard-2` |
| `gradient_mode` | enum | 当前公平主线固定 central difference；自动梯度另作消融 |
| `visibility` | `[P,K]` mask/weight | synthetic v0 默认全 1；未来遮挡不能静默删除 |
| `weight` | `[P,K]` | v0 为均匀面积权重；未来可加入 pupil transmission/vignetting |

真实 f-number 需要 `D_ap=f/N`。当前 rig 没有焦距和物理孔径，故 `aperture_radius` 只能叫 synthetic radius；禁止把 `0.035/0.075` 改写成某个 f-stop。

### 6.2 输出

算法必须同时返回聚合量和未聚合诊断：

| 输出 | 形状 | 用途 |
|---|---|---|
| `subray_deflection_uv` | `[P,K,2]` | 防止 cone mean 掩盖尾部错误 |
| `cone_deflection_uv` | `[P,2]` | 与其他方法比较的共同主输出 |
| `subray_valid_mask` | `[P,K]` | domain、stencil、frustum、topology、endpoint 门 |
| `focus_residual` | `[P,K]` | 证明 flow-off bundle 共焦 |
| `quadrature_se/cv` | `[P,2]` | pupil integration 收敛，不等于模型误差 |
| `path_diagnostics` | structured | domain margin、direction norm、support crossings、endpoint residual |
| `cost_ledger` | structured | point queries、dispatches、JVP/VJP、host sync、wall time、peak memory |

如果任一 subray 无效，默认应 fail closed：整个 pixel 标为 invalid。若另报“删除遮挡 ray 后重新归一化”的物理版本，必须同时保留原始 blocked fraction 和未归一化通量，不能把删除行为藏在平均中。

### 6.3 最小伪代码

```python
def cone_forward(field, pixels, rig, pupil_states, mode, steps):
    # 1. 一个冻结 Sobol prefix 供所有候选共享
    pupil_xy = disk_map_sqrt_radius(pupil_states)

    # 2. 每个 pixel 的 aperture points 指向同一个 flow-off background focus
    rays = focused_pupil_geometry(pixels, pupil_xy, rig)
    assert max_focus_residual(rays) <= focus_tolerance

    # 3. 内层路径算法可以替换，外层 pupil quadrature 完全相同
    delta_uv, diagnostics = trace_and_project(
        field, rays, mode=mode, steps=steps
    )

    # 4. fail closed；不静默 clamp、不静默删除无效 ray
    valid = physical_validity_gate(diagnostics)
    if not valid.all():
        return invalid_pixel_packet(delta_uv, diagnostics)

    # 5. 返回平均值，也保留每条 ray 和积分误差诊断
    cone_uv = weighted_mean(delta_uv, pupil_weights, axis="pupil")
    return cone_uv, delta_uv, diagnostics, exact_cost_ledger()
```

### 6.4 两阶段实现顺序

#### 阶段 A：一个 pixel-cone，复用当前三个 cases

- 每个 case 视为一个 pixel-cone，`K in {4,8,16,32,64,128,256}`。
- 先运行 `PS/PC/CS/CC` 四象限，不做逆解。
- `K_ref=1024` 只作为初始建议；必须以收敛门决定，不得因为成本高就宣称 64 足够。
- `S in {64,128,256,512}` 独立检查 path convergence。
- 先保留 development families；shock/expansion reserved families 不因实现 baseline 而自动打开。

#### 阶段 B：小型多 pixel rig

- 在横向平面建立固定 `8 x 8` pixel centres；每 pixel 共用同一 Sobol pupil prefix。
- 报 spatial map、front-adjacent pixel 与 smooth-control pixel，不能只报全局均值。
- 只有这一阶段才能开始讨论一个小型 deflection field；仍不是 BOST reconstruction。

## 7. 成本合同

### 7.1 当前实现可审计的 point-query 下界

令 `P` 为 pixel 数，`K` 为每 pixel subray 数，`S` 为 path steps。当前 central-difference gradient 每个位置需要 6 个偏移场值，加 1 个中心折射率值。

| 路线 | 当前内核的主要场点查询 | 说明 |
|---|---:|---|
| straight central cone | `7 P K S` | 1 个 `n` + 6 个 central-difference 点；现有 shared state 已按该单位记账 |
| full curved RK4 + midpoint observation | 约 `35 P K S` | 每步 4 次 RK4 RHS，加 1 次 midpoint observation；每次 7 点 |
| Picard-1/2 | 实测记账 | 取决于每轮是否重算 `n/grad n`、是否缓存 straight state；不得只用迭代次数估算 |
| N1 variational | 实测记账 | 包含 straight medium、gradient/Hessian 系数与变分积分；Hessian autograd 不能折算成普通 point query |
| discrete RK4-JVP | 实测记账 | 需单独报告 primal RHS、JVP/HVP/VJP 与保存状态；不能只报 forward 次数 |

`35PKS` 是当前代码路径的 point-query 结构，不是 wall-time 预测。批处理可把大量点塞入少数 interpolation dispatch；autograd、Python 循环、内存流量和 host synchronization 会使 point count 与时间不同。必须同时报告：

- scalar field point queries；
- vectorized interpolation dispatches；
- coordinate JVP/VJP/HVP 次数；
- Python/CUDA host sync 次数；
- p10/p50/p90 wall time；
- peak resident memory / accelerator memory；
- 输出精度与 invalid fraction。

### 7.2 当前 N1 时间只能怎样引用

本仓库当前 development 证据显示，在 `K=64` 个独立 rays、`S=128`、CPU float64 条件下，N1 candidate 的 `p90/full-high p10` 约为 `0.0875--0.0882`。这说明当前 forward closure 有开发成本优势，但：

- 它尚未包含 JVP/VJP 训练成本；
- 它尚未聚合成 cone measurement；
- 它没有真实 image formation；
- 7/9 且 wrinkled-wide 两项 no-harm 失败。

因此该数字只能作为 CRB-v0 的本机 timing anchor，不能作为 cone-ray 加速、真实速度或可部署结论。

## 8. 与 discrete JVP / Picard 的公平比较

### 8.1 所有方法必须解决同一个外层问题

所有候选都必须先对相同的 `P x K` subrays 产生两分量输出，再用完全相同的 pupil weights 聚合：

$$
\widehat{A}_{K,S}^{m}[n]
=\sum_{k=1}^{K}w_k A_S^{m}[n; a_k],
\qquad
m\in\{\mathrm{straight,N1,JVP,P1,P2,curved}\}.
$$

不得用单中心 N1 对比 K-ray curved cone 后宣称“更快”；也不得给 JVP/Picard 更少的 pupil samples，却不画 accuracy-cost frontier。

### 8.2 冻结高精度目标

development reference 建议定义为

$$
H_{\mathrm{cone}}^{\mathrm{ref}}
=\widehat{A}_{K_{\mathrm{ref}},S_{\mathrm{ref}}}^{\mathrm{curved\ RK4}}[n],
$$

其中 `K_ref` 和 `S_ref` 必须分别通过 pupil 与 path 收敛审计。不要用 `K=64,S=256` 自动当真值。

三类误差必须拆开：

$$
E_{\mathrm{path}}^{m}
=\frac{\|\widehat A_{K,S}^{m}-\widehat A_{K,S}^{\mathrm{curved}}\|_2}
{\|\widehat A_{K,S}^{\mathrm{curved}}\|_2},
$$

$$
E_{\mathrm{pupil}}
=\frac{\|\widehat A_{K,S}^{\mathrm{curved}}-
\widehat A_{K_{\mathrm{ref}},S}^{\mathrm{curved}}\|_2}
{\|\widehat A_{K_{\mathrm{ref}},S}^{\mathrm{curved}}\|_2},
$$

$$
E_{\mathrm{step}}
=\frac{\|\widehat A_{K_{\mathrm{ref}},S}^{\mathrm{curved}}-
\widehat A_{K_{\mathrm{ref}},S_{\mathrm{ref}}}^{\mathrm{curved}}\|_2}
{\|H_{\mathrm{cone}}^{\mathrm{ref}}\|_2}.
$$

这可以防止路径误差、孔径采样误差和时间步误差互相抵消。

### 8.3 必须同场比较的候选

| ID | 方法 | 必须保持的合同 |
|---|---|---|
| `PS` | center straight | 同一 chief ray、同一 central difference、同一 `S` |
| `CS` | straight cone | 同一 `K`、same focused pupil bundle；有限孔径强经典底座 |
| `N1-C` | cone-wrapped N1 variational | 每条 subray 独立预测，再聚合；保留 per-ray validity |
| `DJVP-C` | cone-wrapped discrete RK4-JVP | 线性化**同一个离散 RK4 map**，做 FD directional check |
| `P1-C` | one Picard-style IVP update | 同一入口状态、同一 cached state、一次重采样 |
| `P2-C` | two Picard-style IVP updates | 两次重采样；检查是否真收敛而非偶然抵消 |
| `CC-128` | full curved cone, execution | `K,S=128` 等冻结执行预算 |
| `CC-ref` | full curved cone, reference | `K_ref,S_ref` 收敛教师；不参与速度优越性宣传 |
| `damped/interp` | 简单阻尼或 `straight + alpha*(curved-straight)` | 防止复杂方法只胜过空白对照 |

若实现 Norton-BVP，应单列，不能代替 `P1/P2-IVP`，因为边界值问题不同。

### 8.4 公平的主指标

forward-only development 至少报告：

1. cone mean relative-L2；
2. per-pixel absolute error 与 p50/p90/p99；
3. per-subray residual relative-L2 与 p90/p99，防平均抵消；
4. pupil quadrature SE/CV 与 `K -> K_ref` 收敛；
5. reference no-harm ratio，相对同 `K,S` full curved execution；
6. invalid-ray fraction、minimum domain/stencil/frustum margin；
7. support crossing、cell-event、focus residual 与 endpoint residual；
8. point queries、dispatch、JVP/VJP/HVP、wall quantiles、peak memory。

将来进入 inverse/reconstruction 后，另加：

- field relative-L2、SSIM/PSNR；
- wrinkled/shock front 的 ASSD、HD95、1-voxel F1；
- held-out camera reprojection；
- smooth-field no-harm 与 false-front rate；
- matched image-domain 和 independent audit-renderer 两套证据。

### 8.5 建议的 development 门，不是论文阈值

在打开任何保留族前，可预注册：

- `focus residual` 全部过门；
- `K_ref` 翻倍后 cone output relative change `<=0.5%`；
- `S_ref` 翻倍后 cone output relative change `<=0.2%`；
- candidate 对 `CC-ref` cone relative-L2 `<=0.2%`；
- candidate/reference error 不高于同预算 `CC-128` 的 `1.10x`；
- per-subray p99 不高于同预算基线的 `1.10x`；
- candidate `p90 wall / CC-128 p10 wall <=0.25`；
- 所有 domain/topology/focus 门 100% 通过。

这些阈值只是与现有 N1 development 门衔接的工程协议。只有在实际误差尺度、真实噪声地板和师兄数据合同到位后，才能重新冻结论文门。

## 9. 与 OERF / 何远哲方向的关系

### 9.1 NeRIF

[NeRIF 的 AIP 正式页](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)和[作者原稿](https://arxiv.org/html/2409.14722)显示：

- 网络以三维坐标为输入，输出折射率 `n` 与其梯度；
- 沿 back-traced ray 随机采样约 60--200 个 path points；
- 同时用数值差分、自动微分与直接梯度 head 构造一致性和位移损失；
- 做了 turbulent Bunsen flame 的九路内窥 BOST 实验。

公开方法段落描述的是沿“the ray”的积分，没有给出二维 pupil integration、f-number sweep 或 finite-aperture quadrature 的证据。作者在结论中把 nonlinear ray tracing 作为可进一步整合的方向。因此本基线与 NeRIF 的关系应写成：

> 在保持 neural refractive-index representation 不变时，审计 straight/thin-ray、field-dependent bending 与 finite-aperture pupil integration 三种 forward fidelity 的误差和成本。

不能写成“NeRIF 没有考虑任何真实光学”或“cone-ray 一定能改进 NeRIF”；公开证据只支持“finite aperture 未在该方法描述中被显式验证”。

### 9.2 同步 PIV-BOST

[Zheng、He 等 2025 的 Springer 正式页](https://link.springer.com/article/10.1007/s00348-025-04093-y)确认：

- 一台相机与一分九内窥系统提供 turbulent non-piloted Bunsen flame 的九个 BOST views；
- neural reconstruction 给出三维折射率场；
- 该场被用于估计并补偿 PIV 成像平面的热致折射位移，摘要报告约 `+/-2%` 的瞬时速度误差量级。

这说明 forward fidelity 不只影响漂亮的三维图，还会传递到下游速度测量。但该论文没有替本项目证明 finite aperture 是组内主导误差；需要师兄提供相机、镜头、焦平面和 flow-off 数据后才能排序。

### 9.3 TDBOST

[ACM 正式 DOI](https://doi.org/10.1145/3809488)、[何远哲公开代码仓库](https://github.com/Hyz617/TDBOST)和[蔡伟伟教授 SJTU 官方主页](https://me.sjtu.edu.cn/teacher_directory1/caiweiwei)共同确认：

- TDBOST 是组内已接受的 4D BOST / tensor decomposition 主线；
- 公开代码以 per-pixel rays、path samples、`u/v` deflection map 和 tensor model 组织计算；
- 公开仓库 HEAD 在本次核验时为 `3393ca7`，数据生成器核心 `proj` 以编译二进制发布，公开 Python 代码不足以独立核对其 distortion correction 是否等价于 ray bending、finite aperture 或 image-domain BOS。

因此最重要的师兄审核问题不是“能否在 TDBOST 后再接一个 cone module”，而是：

1. TDBOST 的训练 target 是图像差、光流位移还是偏折角？
2. `proj` 中的 distortion correction 修正的是背景板交点、折射路径、相机畸变，还是其他量？
3. 一条 pixel ray 还是一个 pupil ray bundle？是否存在 f-number/focus 参数？
4. forward/JVP/VJP 是否同一离散目标，能否做 dot test？

### 9.4 与课题组需求的最窄接口

SJTU 官方主页将课题组方向列为流体光学诊断、线性/非线性层析、机器学习流动显示，并列出 BOST 软件和 TDBOST。最贴合的工作不是独立发明一套通用 ray tracer，而是给现有 BOST/NeRIF/TDBOST 提供一个可插拔、可审计的 forward-fidelity ladder：

```text
thin straight
    -> focused straight cone
    -> per-ray first-order / discrete JVP / Picard
    -> full curved cone
    -> image-domain cone BOS / UBOST
```

每一级都有相同输入输出、误差分解、调用成本和 fail-closed 条件，师兄可以据真实 rig 选择最低但足够准确的一层。

## 10. 能与不能声称的创新

### 10.1 现在绝对不能声称

- 首次提出 BOS/BOST cone-ray、finite-aperture 或 depth-of-field 模型；Molnar 等已直接完成。
- 首次用神经隐式场做 cone-ray BOS inverse；Molnar 已把 cone-ray 嵌入 NIRT。
- 首次做 BOST 三维/四维神经重建；Grauer、NeDF、NeRIF、GRU-BOST、TDBOST 等均是直接先例。
- 首次做固定端点 ray-linking、逐次逼近或一阶折射率扰动路径；Norton 已给出。
- 首次用 RK4、Picard、JVP/VJP 或自动微分处理 GRIN ray tracing。
- 当前 7/9 synthetic N1 已优于 NeRIF、TDBOST、DeepONet、FNO 或 cone-ray strong baseline。
- 当前 synthetic aperture radius 对应真实 f-number，或有限孔径是 OERF rig 的主导 mismatch。
- 只凭 cone mean 变好就声称前沿、真实图像或三维重建改善。

### 10.2 尚可检验、但未证明的新意候选

以下只能写成 hypotheses：

1. **Pupil-shared discrete defect operator。** 对同一像素的多条 pupil rays 共享场缓存、局部 Hessian-vector products 或低秩路径响应，以低于逐 ray full RK4 的成本逼近 curved-cone forward，同时保持 objective-consistent JVP/VJP。
2. **双轴 fail-closed fidelity router。** 分别估计 pupil quadrature risk 与 trajectory-linearization risk，仅在两者都过门时使用便宜路径；否则回退 Picard/full curved。贡献必须来自 observable certificate，而不是 truth/oracle routing。
3. **Cone-integrated no-harm correction。** 学习/解析修正直接以 cone measurement 为目标，并约束 per-subray tail，避免正负 residual 在孔径平均中制造虚假成功。
4. **跨 aperture / camera geometry 的条件算子。** 输入 unordered `(camera, pixel, pupil, reliability)` 集合，对未见 f-number、focus 和 view count 保持校准；必须超过简单坐标拼接、VIDON/Deep Sets 和 parameter-matched post-corrector。
5. **BOST forward-error budget。** 在同一真实 rig 上系统拆分 optical flow、finite aperture、ray bending、camera calibration 与 discretization 的贡献；若能证明直接前作尚未覆盖这一联合协议，并取得严格成对数据和独立 renderer，可能比再造一个大网络更有论文价值。

即使文献检索没有找到完全相同的组合，也只能写：

> 在本次核验的直接前作中，尚未发现“BOST focused-pupil cone integration + objective-consistent discrete trajectory correction + observable fail-closed routing + per-subray tail audit”这一完整组合。该观察只形成待证伪假设，不构成 novelty proof。

## 11. 需要向何远哲确认的数据合同

按优先级询问：

1. 当前 NeRIF/TDBOST forward 是 straight、prescribed-bend、nonlinear curved ray，还是 finite-aperture ray bundle？
2. `distortion correction` 的精确定义、输入、target、loss 和推理位置是什么？
3. 每个像素的 camera origin/direction、background intersection、near/far、mask 是否可导出？
4. 镜头焦距、f-number、focus distance、背景距离、像元尺寸和 circle of confusion 是否记录？
5. 是否有同一相机、同一流场条件下的 paired f-number 或 focus sweep？没有则不能因果识别 finite aperture。
6. 是否保存 flow-off/on 原图、光流位移、可靠度、occlusion 和 invalid mask？
7. 当前每 ray/path 的 sample 数、采样策略、forward profile 和峰值显存是多少？
8. 能否调用同一 forward 的 JVP/VJP；是否已有有限差分 directional test 与 adjoint dot test？
9. 可否先提供一个小合同：`16--64` 个 pixel rays、一个小折射率场、相机参数，以及 straight/high forward 配对输出？
10. 组内认为主导 mismatch 的排序是什么：finite aperture、ray bending、camera calibration、optical-flow displacement，还是 field discretization？依据是哪组实验？
11. PIV-BOST 九路内窥镜的 f-number/focus 是否一致，bundle 各通道是否有独立 PSF/畸变标定？
12. TDBOST 公开 `proj` 二进制能否提供公式、接口文档或可核验的纯 Python 小例，而不需要公开受限实验数据？

没有这些回答前，CRB-v0 只能是 synthetic mechanism baseline；不能把缺失信息交给网络隐式“学掉”。

## 12. 一级来源核验表

以下均为可点击的出版社页、官方机构页、作者原稿或作者代码。摘要与正文要点均为转述，没有复制受限全文。

| # | 一级来源 | 核验到的作用 | 不能替本项目证明 |
|---:|---|---|---|
| 1 | Stephen J. Norton, [Computing ray trajectories between two points: a solution to the ray-linking problem](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-10-1919), JOSA A 4, 1919--1922 (1987) | 固定端点 ray-linking、隐式积分方程、successive approximation、一阶路径 | finite-aperture 平均、BOS image formation、当前 IVP RK4 等价性 |
| 2 | Markus Raffel, [Background-oriented schlieren (BOS) techniques](https://link.springer.com/article/10.1007/s00348-015-1927-5), Experiments in Fluids 56, 60 (2015), open access | BOS 物理、位移-偏折-折射率梯度链、多视角 tomography 与孔径/照明折中 | 任意具体 rig 的误差量级或 cone-ray 收敛 |
| 3 | Molnar et al., [AIAA Journal 正式页](https://doi.org/10.2514/1.J064095), 62, 4316--4329 (2024) | 同行评审的 finite-aperture cone-ray BOS forward/inverse 直接先例 | OERF 系统也有相同收益 |
| 4 | Molnar et al., [作者开放原稿](https://arxiv.org/html/2402.15954) | aperture integral、1000 rays/pixel forward、10,000 frustum points/pixel inverse、f/22--f/4、occlusion 与 UBOS 细节 | 当前 64-ray synthetic rig 已复现论文 |
| 5 | Nicolas et al., [A direct approach for instantaneous 3D density field reconstruction from BOS measurements](https://link.springer.com/article/10.1007/s00348-015-2100-x), Experiments in Fluids 57, 13 (2016) | 直接从 deviation field 重建 3D density、正则化与实验 bench 的经典强基线 | finite aperture、neural field 或 curved-ray 校正 |
| 6 | Grauer et al., [Instantaneous 3D flame imaging by BOST](https://www.sciencedirect.com/science/article/pii/S0010218018302694), Combustion and Flame 196, 284--299 (2018) | 多相机 turbulent flame BOST、Tikhonov/TV、wrinkled flame front 与 23-camera 实验 | cone-ray、4D tensor 或当前 OERF rig |
| 7 | Grauer and Steinberg, [Fast and robust volumetric refractive index measurement by UBOST](https://link.springer.com/article/10.1007/s00348-020-2912-1), Experiments in Fluids 61, 80 (2020) | 将 optical flow 方程与 tomography 合并，说明图像域/位移域是不同观测合同 | finite-aperture 或曲光线自动解决 |
| 8 | Cai et al., [Direct BOST using radial basis functions](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100), Optics Express 30, 19100--19118 (2022), open access | 连续 RBF 直接折射率重建、projection matrix 与 ray discretization 的经典连续场对手 | neural/cone/curved path 新意 |
| 9 | Li et al., [Neural deflection field for sparse-view TBOS](https://doi.org/10.1063/5.0241191), Physics of Fluids 36, 121701 (2024) | sparse/limited views、神经 gradient field、high-fidelity nonlinear ray-traced synthetic BOS | finite aperture、真实跨 rig 泛化 |
| 10 | He et al., [NeRIF AIP 正式页](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the), Physics of Fluids 37, 017143 (2025) | 何远哲主线的 neural refractive-index representation 与实验 BOST | cone-ray 已验证或 ray bending 已解决 |
| 11 | He et al., [NeRIF 作者原稿](https://arxiv.org/html/2409.14722) | 可核对 `n/grad n` heads、60--200 path samples、AD/ND loss、九路 Bunsen 实验 | finite-pupil/image-domain 模型存在 |
| 12 | Zheng et al., [Instantaneous refractive-index compensation using simultaneous PIV-BOST](https://link.springer.com/article/10.1007/s00348-025-04093-y), Experiments in Fluids 66, 164 (2025) | 何远哲参与的九路内窥 BOST、PIV downstream compensation 与真实 reacting-flow 价值 | finite aperture 是误差主因 |
| 13 | He et al., [TDBOST ACM 正式 DOI](https://doi.org/10.1145/3809488), ACM TOG (accepted/publisher record) | 4D tensor-decomposition BOST 直接课题组前作 | 公开页面不足以辨认 cone/ray-bending 细节 |
| 14 | He Yuanzhe, [TDBOST 作者公开代码](https://github.com/Hyz617/TDBOST) | per-pixel rays、path sampling、u/v deflection maps 与 tensor reconstruction 的公开接口 | 编译 `proj` 内核的物理公式、finite aperture 或完整可复现性 |
| 15 | 上海交通大学机械与动力工程学院, [蔡伟伟教授官方主页](https://me.sjtu.edu.cn/teacher_directory1/caiweiwei) | OERF 的非线性层析、机器学习流动显示、BOST/TDBOST 与相关项目关系 | 任一候选算法已获课题组认可 |
| 16 | W. H. Southwell, [Ray tracing in gradient-index media](https://opg.optica.org/abstract.cfm?uri=josa-72-7-908), JOSA 72, 908--911 (1982) | GRIN ray-tracing 数值方法与 Euler/高阶方法精度-成本问题的经典先例 | 当前 RK4/Picard/JVP 的具体排序 |

## 13. 实施后的判决树

```text
flow-off focused-pupil gate 失败
    -> 只能叫 synthetic aperture bundle；停止 cone-ray 物理主张

focused-pupil gate 通过
    -> 独立做 K-convergence 与 S-convergence

K 或 S 不收敛
    -> 提高 reference；禁止比较候选

收敛 reference 建立
    -> 比 PS / CS / PC / CC，先判断 aperture 与 bending 谁主导

CS 接近 CC，PS 远离 CC
    -> finite aperture 主导；优先 pupil quadrature/cache，不必复杂轨迹模型

PC 接近 CC，PS 远离 CC
    -> bending 主导；优先 discrete JVP/Picard/N1

CS 与 PC 都远离 CC
    -> aperture x bending 交互显著；候选必须 cone-wrapped

N1/JVP/Picard cone mean 过门但 per-ray p99 失败
    -> 判 cancellation-only；不得写算法成功

synthetic 全过但无真实 paired aperture/geometry
    -> 只授权小数据合同；不得写真实 BOST 或泛化结论
```

## 14. 最小交付清单

真正开始实现 CRB-v0 时，至少应新增以下独立产物；本次审计没有创建它们：

- focused-pupil geometry 与 flow-off closure 单元测试；
- `P=1` 的 `PS/PC/CS/CC` 四象限 runner；
- 独立 `K` 与 `S` 收敛审计；
- N1、discrete JVP、Picard-1/2 的统一 cone wrapper；
- point-query/dispatch/JVP/VJP/host-sync/wall/memory 成本账本；
- cone mean、per-ray tail、focus/topology/invalid diagnostics 图；
- frozen JSON config、CSV、result、manifest 与 machine decision；
- 师兄审核后的真实最小数据合同。

在这些产物全部通过前，论文中最诚实的写法是：

> We design a source-audited finite-aperture baseline for the N2-PVGR synthetic rig and preregister a comparison between cone-wrapped variational, discrete-JVP, Picard, and full curved-ray operators. No cone-ray implementation, real-data validation, or novelty claim is reported at this stage.
