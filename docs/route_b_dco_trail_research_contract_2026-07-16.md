# 路线 B 研究合同：DCO-BOST 与 TRAIL-4D

日期：2026-07-16

状态：碰撞审计完成；尚未预注册新鲜实验，尚无 superiority 结果
适用边界：反应流 BOST 三维/四维重建；与何远哲 NeRIF/TDBOST 主线对齐

## 1. 为什么必须换问题

V5N-V5R 已依次否掉四种容易但不够扎实的叙事：

1. 纯 shared-field 网络不是强重建器：相对 PBB-32 总体差 15.83%。
2. 网络 prior 没有成为通用低预算加速器：v5p 新鲜门禁三项全部失败，本机慢 5.14 倍。
3. In-sample source residual 不能形成安全门：selected-field harm 22.77%。
4. Reserved camera 也不能形成稳定符号门：虽然总体 +3.405%，但只有 12/18 cells 为正，selected-field harm 33.71%，rig D 的排序相关仅 0.016。

因此下一方法不能继续学习“从观测直接猜场”或“事后判断网络是否会赢”。学习模块必须对应一个独立可测的物理缺口。

## 2. 两个真实物理缺口

### 2.1 DCO-BOST：forward model mismatch

低保真重建算子 $A_0(g)$ 通常采用 thin ray、名义孔径、理想标定或低采样 cone ray；真实测量算子 $A_\star(g)$ 还受到有限孔径、景深、曲线光路、镜头畸变、标定漂移和空间相关噪声影响。即使优化器完全收敛，错误的 forward/adjoint 仍会把解推向错误场。

工作候选不是自由 CNN，而是几何条件化的显式低秩修正：

\[
A_{\mathrm{corr}}(g)=A_0(g)+U\,C_\phi(g)\,V^\top,
\]

其中 $g$ 包含视角、孔径、cone 参数、焦距/物距、bend 或标定摘要；$U,V$ 是共享测量/体素基，轻量网络或正则回归只预测小矩阵 $C_\phi(g)$。它的伴随不是另训一个网络，而是按结构严格给出：

\[
A_{\mathrm{corr}}(g)^\top=A_0(g)^\top+V\,C_\phi(g)^\top U^\top.
\]

临时名称：**GC-BiLOC**（geometry-conditioned bidirectional low-rank operator correction）。名称只用于代码沟通，尚未做商标/论文标题检索。

### 2.2 TRAIL-4D：时空低秩假设在突变处失效

TDBOST 将 $X-Y-Z-T$ 张量拆成三组二维耦合分量并用轻量网络提高效率。真实反应流还存在火焰新生、熄灭、局部破裂、涡卷吸、间歇热点、缺帧和相机异步；固定低秩容易把这些稀疏创新抹平。

TRAIL-4D 的工作模型是：

\[
f_t=\mathcal{W}(f_{t-1},u_t)+s_t,
\]

其中 $\mathcal{W}$ 是 transport/warp，$u_t$ 是低维运动场，$s_t$ 是稀疏创新。低秩分支解释被输运的主体，创新分支解释新生、熄灭和拓扑突变。该路线最贴近师兄，但没有真实 timestamp、缺帧模式和连续序列时不得启动 superiority 训练。

## 3. 已有工作碰撞矩阵

以下入口已于 2026-07-16 通过作者稿、出版社或课题组主页核对。

| 已有工作 | 已经占据的贡献 | 本项目不能再声称 | 需要留下的差异 |
|---|---|---|---|
| [NeRIF](https://arxiv.org/abs/2409.14722) | 连续神经折射率场、BOST 数值与火焰实验 | 首个 neural field BOST | 跨 rig 的 operator mismatch，而非逐实例隐式场 |
| [TDBOST](https://doi.org/10.1145/3809488) | 4D 张量分解 + 轻量网络 | 首个 4D BOST 或张量神经重建 | transport、sparse innovation、异步与突变边界 |
| [Cone-ray BOS](https://arxiv.org/abs/2402.15954) | 有限孔径 depth-of-field forward/inverse model；f/22 到 f/4 | 首次处理有限孔径 | 用少量校准探针逼近昂贵高保真算子，并审计逆问题稳定性 |
| [NeDF](https://arxiv.org/abs/2409.19971) | 稀疏视角、神经 deflection field、高保真 nonlinear ray tracing | 首次 sparse-view/nonlinear-ray neural BOST | 显式修正算子与严格共轭伴随，不是逐场坐标网络 |
| [Neural refractive-index primitives](https://arxiv.org/abs/2605.11454) | hash encoding、自动离散梯度、3D mask | 仅靠更强 positional/hash encoding 就有新意 | 物理失配与跨 rig 泛化证据 |
| [Learned Operator Correction](https://arxiv.org/abs/2005.07069) | forward-adjoint correction 与收敛条件 | 首次校正 forward/adjoint | BOST 光学参数条件化、少探针、显式低秩共轭结构 |
| [Inverse Problems with Learned Forward Operators](https://arxiv.org/abs/2311.12528) | learned restriction 与 simplified-physics correction 综述 | “学便宜 forward”本身是新方向 | 具体 BOST 数据合同和强失配实验 |
| [Learned ReSeSOp](https://arxiv.org/abs/2410.23061) | inexact operator、在线误差水平、动态成像 | 首次在迭代中处理未知模型误差 | 与其正面对比；不能只打普通 PBB/FNO |
| [Reference Neural Operators](https://proceedings.mlr.press/v235/cheng24c.html) | 几何变形下的数据高效 operator learning | 几何条件化本身是新意 | 相机/ray 光学几何与双向低秩校正 |
| [Operator SVD](https://proceedings.mlr.press/v235/ryu24b.html) | 用神经网络做嵌套低秩 operator SVD | 神经低秩分解本身是新意 | 低保真到高保真差分、校准查询和 inverse impact |
| [Structured matrix learning](https://proceedings.mlr.press/v336/amsel26a.html) | 用 matvec/adjoint queries 学低秩等结构矩阵 | 少量查询学习低秩矩阵本身是新意 | BOST 几何条件与真实标定探针；需引用并比较查询效率 |

结论：GC-BiLOC 目前只能作为**领域化组合假设**，不是已确认原创算法。论文价值必须来自三者同时成立：真实 BOST mismatch、严格 forward/adjoint 结构、跨 rig/光学参数的新鲜证据。

## 4. GC-BiLOC 的最小可证伪实验

### 4.1 数据单元

每个 `rig_id` 对应一对算子：

- $A_0$：便宜 nominal thin/cone-ray、低 aperture/path samples；
- $A_\star$：高 aperture/path samples、真实孔径/curve/cone 参数；
- $g$：七视角角度、model/truth radius、cone-u、cone-z、bend、焦距/物距摘要；
- calibration probes $x_j$：反应流 phantom、随机平滑场、thin-front 场；
- 双向查询：$A_\star x_j$ 与 $A_\star^\top z_j$。

训练不能直接把完整 $A_\star-A_0$ 矩阵交给网络后再声称 query-efficient。完整矩阵只用于 synthetic 评分；训练应限制 calibration probes 数量，并把 forward/adjoint queries 精确记账。

### 4.2 Split

先生成 manifest 再构造算子：

- train：基础角度、孔径、cone/bend；
- development：新参数组合，只选 rank、ridge 与特征；
- fresh-A：新角度布局；
- fresh-B：新孔径/depth-of-field；
- fresh-C：joint OOD；
- design lock：在 fresh 全部通过前不构造。

同一 rig 的所有 probes、fields、views 必须留在同一 split。不能把 probe 当独立 rig 扩大样本量。

### 4.3 必须比较的基线

1. $A_0$：不修正 nominal operator。
2. 高保真 $A_\star$：accuracy upper bound，同时报告构造、内存和 matvec 时间。
3. Global mean discrepancy。
4. Nearest-geometry discrepancy。
5. Polynomial/ridge geometry-to-discrepancy。
6. Shared HOSVD/Tucker correction，不用神经网络。
7. Learned forward-adjoint correction 的等容量实现。
8. Learned ReSeSOp 或最接近的公开实现/可复现替代。
9. 若规模允许，Reference Neural Operator/GNOT 类几何基线。

DeepONet、FNO、F-FNO 可以保留，但不能作为唯一强对手。

### 4.4 五层指标

**Operator 层：**

- held-out probe relative-L2；
- discrepancy capture (1-\|A_{\rm corr}-A_\star\|_F/\|A_0-A_\star\|_F)；
- singular spectrum 与最坏方向误差；
- forward/adjoint query 数、参数、内存和 wall-clock。

**Adjoint/gradient 层：**

- dot-product test 必须到浮点误差；
- $\nabla_x\frac12\|A_{\rm corr}x-y\|^2$ 与高保真 gradient 的角度和相对误差；
- 不能只报告 forward PSNR。

**Inverse 层：**

- PBB/CGLS 使用 $A_0$、$A_{\rm corr}$、$A_\star$；
- 同停止准则、support、regularization、calls 与 wall-clock；
- field relative-L2、gradient/front error、held-out view residual。

**鲁棒性层：**

- 分开 rig、角度、孔径、bend、noise、topology；
- field cluster、positive-cell fraction、worst cell、harm fraction；
- 不用 rows 数制造窄置信区间。

**真实性层：**

- synthetic 高保真结果不等于实验成功；
- 至少需要一个真实 calibration phantom 或已知温度/密度场；
- 无真实 truth 时只报告 held-out optical consistency、重复性与物理积分量。

## 5. 首轮 Go/No-Go 建议

2026-07-16 红队后，本节由单一硬阈值改为顺序门禁；完整定义见 [V6B-1.0 少查询预注册协议](v6b_limited_query_preregistration_2026-07-16.md)。旧的 gradient cosine 实现没有计算真实残差梯度，旧的 `max degradation <= 5%` 又会被小分母单 rig 主导，两者不得继续使用。

| Gate | 冻结门槛 | 失败后的动作 |
|---|---:|---|
| Physics gap | nominal mismatch RMS ≥ 独立 noise floor 的 2 倍 | 标为物理缺口未解析，不评价算法 |
| Numerical validity | float64 dot defect ≤ 1e-10；finite-difference 通过 | `INVALID`，停止 |
| Operator unlock | vs 固定 B* ≥ 10%，cluster CI 排除 0；36/48 rigs；各层 ≥ 8/16 | 不解封 inverse |
| Strong operator GO | ≥ 25%，且 A/B/C 各 ≥ 12/16 | 才允许强 operator 主张 |
| Paired safety | p90 degradation ≤ 5%；max ≤ 20%；各层聚合非负 | 停止或回退 |
| True residual gradient | median cosine ≥ 0.99；p05 ≥ 0.95；无负值；median rel-error ≤ 0.25 | 不进入 inverse |
| Fresh inverse gain | vs $A_0$-PBB ≥ 3%，CI 排除 0，并胜 B* 或 2% 非劣 + runtime 优势 | 不宣称 reconstruction value |
| Runtime | 目标分辨率下 matvec ≤ 0.5 × direct high-fidelity，另报 100 次重建摊销 | 关闭 surrogate-efficiency 主张 |

即使全部通过，也只允许打开下一套 design lock；不能直接写“优于现有方法”。

## 6. Mac 上的实现顺序

1. `6×6×4` 网格，40-60 rigs，先缓存 $A_0/A_\star$ 和 manifest。
2. 非学习 HOSVD：检查 discrepancy 在 rig/measurement/voxel 三个模态是否真低秩。
3. Geometry ridge：预测 core $C(g)$，建立最强便宜基线。
4. 小 MLP：只有 ridge 在 development 留下系统残差时才加入。
5. Forward/adjoint probe audit；失败则停止。
6. 仅对通过的 operator correction 跑 PBB/CGLS inverse。
7. `16³-32³` 只用于确认缩放趋势；高分辨率与真实 ray tracer 再迁服务器。

第一晚训练不应该从 3D CNN 开始，而应从 HOSVD 谱图开始。若 discrepancy 不低秩或不随 $g$ 平滑，GC-BiLOC 的结构假设当场失败。

## 7. TRAIL-4D 的启动条件

必须从何远哲确认：

- 每帧真实 timestamp 与相机同步误差；
- 连续 run 长度、帧率、缺帧与曝光；
- TDBOST 当前输入/输出、分解 rank、loss、训练/推理时间；
- 是否存在新生、熄灭、拓扑断裂的标注或可计算 proxy；
- 能否封存按 session 分割的新序列。

最低基线：TDBOST、逐帧 PBB/NeRIF、固定低秩 Tucker/CP、3D+time FNO、transport-only、innovation-only。主指标除全场误差外，必须有 event-conditioned error：新生/熄灭窗口、thin-front、temporal derivative、缺帧恢复和最坏 session。

## 8. 现在发给师兄的决策问题

1. 组内当前最痛的误差是 ray/aperture/标定失配，还是 4D 中的新生、熄灭、异步和缺帧？
2. NeRIF/TDBOST 是否能提供可调用的 $F$ 与 $F^\top/J^\top$，以及 ray、mask、grid、unit？
3. 是否有 paired low/high-fidelity simulation、标定 phantom 或多档 f-number 数据，可用于 operator discrepancy？
4. 若优先 4D，能否先给一小段带 timestamp、缺帧和同步信息的序列，而不是整理完整数据集？

## 9. 当前最诚实的投稿判断

单靠当前 synthetic 小网格和组合算法，不足以支持高水平论文。GC-BiLOC 达到论文候选至少还缺：独立真实 mismatch、强 inexact-operator 基线、跨 rig 新鲜证据、成本优势和一项不是现有 operator correction 直接替换的 BOST 特有发现。TRAIL-4D 达到论文候选则至少还缺真实高速序列、事件条件指标、TDBOST 复现和对异步/拓扑突变的明确增量。

本科毕设的安全交付可以是：完整 negative-evidence chain + 一个通过严格 synthetic design gate 的机制原型 + 真实数据接口与小样例。论文是额外上限，不应倒逼伪造“已胜出”的结果。

## 10. V5S-V6A 执行结果：GC-BiLOC 已否，RayKernel-DCO 留作数据决策

这部分是合同执行后的结果，不是首开前预注册。所有 V5T-V6A 都是在 V5S 结果打开后形成的 synthetic development/post-open 诊断；源码哈希用于可复现，不冒充公开预注册时间戳。

| 阶段 | 被检验的结构 | 主要数值 | 判决 |
|---|---|---:|---|
| V5S | 全局 measurement×voxel 共享低秩 GC-BiLOC | error 0.8285；full-matrix ridge 0.5217；相对 -58.81% | `GC_BILOC_DEVELOPMENT_NO_GO` |
| V5T | truth-parameter 局部 tangent/secant oracle | tangent 0.5050；secant 0.2607；parameter-gap error 0.5607 | 可表示性有上限，但不可部署，NO-GO |
| V5U | truth geometry 对齐后的全局 CAL-HOSVD | 只消掉 8.39% 原始失配；0.8094 vs ridge 0.4762 | 全局低秩再次 NO-GO |
| V5V | 每相机固定 5×5 measurement kernel | oracle 0.9043；预测 0.9171 | 相机内并非平移不变，NO-GO |
| V5W | 仅有限孔径因素的固定 left/right kernel | 最好预测 0.8074 vs ridge 0.8143；仅 +0.854% | 固定核 NO-GO |
| V5X | 每条 ray 的 3×3×3 voxel kernel | oracle 0.3587；两阶段预测 0.8160；1113× 压缩 | 表示近门槛，kernel-target 学习失败 |
| V5Z | 端到端线性 ray-conditioned kernel | 0.7707；相对 ridge +5.359%；worst ratio 0.777 | 稳定正信号，未过 10% |
| V6A | `33→64→64→27` ray-kernel hypernetwork，3 seeds | 0.7485；+8.080%；6/12 rigs；aggregate worst ratio 0.761；max paired degradation +13.69% | `NO_GO_STOP_CAPACITY_ESCALATION` |

V5Y 还保留了一个重要的工程失败：原始大学习率使 error 达 1.9769；V5Z 稳定化后才得到 0.7707。因此不能把 V5Y 当结构反例，也不能隐藏它只展示稳定版本。

### 10.1 结构漏斗给出的物理解释

1. 原始多因素失配不是全局低秩。
2. 只做 truth geometry 对齐仍留下 91.61% discrepancy norm，renderer fidelity 是主缺口之一。
3. 每相机固定核失败，说明 aperture/DOF 影响随探测器位置、ray 和深度变化。
4. 每 ray 局部 3D kernel oracle 已到 0.3587，证明表示比全局矩阵低秩更对题。
5. 两阶段核回归失败而端到端训练改善，说明中间 kernel 系数不唯一；应直接优化可观测 operator action。
6. 小 hypernetwork 比线性模型再改善约 2.7 个百分点，但 rig 胜率只有 50%，不能继续在 opened 集上堆容量。

### 10.2 当前工作名与数学形式

下一候选临时称 **RayKernel-DCO**。对 measurement ray $r$，用几何摘要 $g_r$ 生成局部体素核：

\[
k_r=h_\phi(g_r),\qquad
(A_{\mathrm{corr}}x)_r=(A_0 K_r(k_r)x)_r.
\]

伴随必须由同一 materialized operator 或同一核的转置组合给出，不能另训 $A^\top$：

\[
A_{\mathrm{corr}}^\top z=K(k)^\top A_0^\top z.
\]

当前实现只在小型 linearized finite-aperture surrogate 上测试；公式不代表已覆盖真实 curved-ray nonlinear BOST。

## 11. V6B fresh gate：少查询适配，而不是继承 V6A 权重

V6A 已打开，禁止继续用它扫宽度、层数、核半径和阈值。静态代码审计还发现：V6A 训练直接读取完整 $A_\star$ rows 与 truth geometry，因此 V6B 只能继承 **ray-local kernel 结构假设**，不能继承其权重或把它称为 zero-shot。

V6B-1.0 冻结为：在 query-limited train cache 上从头训练一个 `33 -> 64 -> 64 -> 27` 单模型；新 rig 上只用 `K` 个预冻结 calibration probes 拟合 27 个 kernel-channel gates，不微调 8,091 个网络参数。高保真 oracle 只暴露 `measure(x)`，第 `K+1` 次调用报错；完整 operator、truth geometry 与 truth adjoint 只存在于独立 evaluator。

- primary synthetic `K=32`；`K=4/8/16` 只做 opened-development 剂量曲线；
- fresh-A：16 个未见 aperture/f-number/focus rigs；
- fresh-B：16 个未见 layout/angles rigs；
- fresh-C：16 个 aperture × layout joint-OOD rigs；
- 每 rig 封存 32 个 operator probes 与 12 个 inverse fields；rig 是统计单位；
- 同预算基线必须包含 minimum-norm secant、ray-local ridge + graph Laplacian、geometry polynomial/B-spline ridge；
- operator 先过第 5 节的 10% 解锁门，才允许构造 inverse；25% 是强 GO，不是进入 inverse 的唯一阈值；
- 真实 BOST 无 3D truth 时不报告 field-L2，改用 held-out view、phantom、重复性和物理积分量。

V6A 报告里的 `worst-rig ratio` 是 `max(candidate error) / max(baseline error)`，只能描述两条总体最坏尾部，不能证明每个 rig 都安全。V6B 改用 paired rig 的 p90 + catastrophic max guard；`max <= 5%` 只作为额外 `ZERO_OBSERVED_HARM` 标签。

V5X 的原始 oracle 门槛为 ≤0.35，结果 0.3587，没有通过。V5Y-V6A 使用的 ≤0.4 是看过 V5X 后的 post-open 诊断前提，只用于决定是否继续机制分析；它不属于 V6B fresh gate，也不得追溯性改判 V5X。

公开资源审计确认：PSU 70-view 数据可做固定 rig 的真实 reconstruction/reprojection，`photon` 可生成独立合成光圈/布局；两者都不能单独提供完整 V6B 真实证据。详见 [公开 BOST 外部基准审计](public_external_bost_benchmark_audit_2026-07-16.md) 与 [V6B-1.0 协议](v6b_limited_query_preregistration_2026-07-16.md)。在 OERF 数据合同未满足前，V6B 保持 **`UNCONSTRUCTED`**。

2026-07-16 已完成 query-budget、27-gate forward/adjoint、dot/finite-difference、pre-score hash 和 in-class/misspecified toy 的协议自检；模型外 K=32 中同预算 secant 仍反超 27-gate，因而该结果只判 `PASS_PROTOCOL_CONFORMANCE_ONLY`，不改变上述 `UNCONSTRUCTED` 状态。逐项边界见 [学习日志第 22 节](v5h_v5m_共享场逆算子研究日志_2026-07-16.md)。

## 12. 更新后的师兄决策问题

1. OERF 当前 finite-aperture/DOF 误差是否真实存在并影响 NeRIF/TDBOST，还是 synthetic renderer 放大了一个次要因素？
2. 现有代码是否能对一个 calibration phantom 返回 $F x$ 与 $F^\top z/J^\top z$？若没有精确伴随，能否提供自动微分 VJP？
3. 是否保存 f-number、焦距、物距、焦平面、相机姿态和 reference/flow-off repeats？
4. 是否有多档光圈、移动焦平面或 paired thin/cone-ray simulation，可构造独立 aperture mismatch？
5. 若没有这些数据，TDBOST 连续序列是否包含 timestamp、缺帧、同步误差、新生/熄灭事件？
6. 在 RayKernel-DCO 与 TRAIL-4D 之间，哪一个对应组内目前真正会影响结果或投稿的痛点？
