# JACRU-M2 文献版图与原创性边界（2026-07-17）

> **结论先行：JACRU-M2 尚未证明新颖。**
>
> 截至 2026-07-17，本次有界检索只能支持一个待验证假设：`有限孔径 BOST 的经典逆解底座 + 离散目标一致的锥射线 forward/VJP + 显式界面状态 + 变相机集合条件化的 learned residual unrolling` 这一**联合实现**可能仍有窄缺口。它不支持“首次提出经典重建加神经校正”“首次把物理放入展开网络”“首次用神经算子做逆问题”“首次处理可变传感器/几何”“首次界面感知重建”“首次神经 BOST”或“首次有限孔径神经 BOST”等宽泛声明。

## 1. 审计范围与证据规则

- 主题：`classical inverse base + learned residual operator`、learned correction、physics-informed unrolling、inverse neural operator、geometry-conditioned set operator、interface-aware reconstruction，以及它们在 BOS/BOST/CBOS 中的交叉。
- 来源：只采用原始论文页面、出版社/会议官方页面、arXiv 和作者官方代码仓库。未把综述、博客、聚合站或二手解读作为结论依据。
- 访问边界：没有下载、上传或转存受限 PDF；付费论文只核验出版社元数据/摘要或作者公开的 arXiv 版本。
- 检索性质：这是围绕 JACRU-M2 的**有界原创性审计**，不是系统综述，也不是对全球文献“查无此文”的证明。后续新论文、不同术语和未被索引的工作都可能改变判断。
- 当前实现边界：仓库中的 M0/M1 证据只允许继续探索 learned residual operator；尚没有一个完成训练、跨样本验证并击败强基线的 M2。因此，算法新颖性、性能优越性和界面收益目前都不能成立。

### 1.1 “learned correction”必须拆成三类

三类先例不能被混写成一个新概念：

1. **解空间/输出校正**：先算经典逆解 `x_0 = B_G(y)`，再学习 `x = x_0 + C_θ(x_0)`；Deep Null Space Learning 是直接先例。
2. **前向算子校正**：学习修正不准确的 forward/adjoint，使优化使用更可信的物理算子；Learned Operator Correction 是直接先例。
3. **迭代更新校正**：在每个展开阶段把 `A_G`、`A_G^*`、残差或数据一致性项交给可学习模块；Learned Primal-Dual、MoDL 是直接先例。

JACRU-M2 若采用第三类，应明确其新增对象究竟是**残差到更新的算子**、**经典解到增量的后处理器**，还是**forward/VJP 的模型误差修正器**。这三者的训练目标、数据一致性保证和对照基线不同。

### 1.2 待审计的 M2 形式

一个可审计而非口号式的 M2 可写为：

```text
x0 = B_G(y)                                      # 冻结的经典逆解底座
rk = A_G(xk) - y                                 # 与训练目标一致的锥射线残差
gk = VJP[A_G](xk, rk)                            # 同一离散 forward 的精确 VJP
Δxk = Rθ(xk, rk, gk, interface_k, SetEnc(G, q)) # 学习残差更新
x{k+1} = Project(xk + αk Δxk)                    # 物理/范围约束
```

其中 `G` 至少含相机内外参、孔径/射线束几何、mask；`q` 可含可靠度或观测质量。若实现没有冻结且可复现的 `B_G`、逐阶段数据一致性、同一离散目标的 VJP、显式界面状态和无序相机集合输入，就不能使用完整的“JACRU-M2”联合贡献表述。

## 2. 相关性分层

| 方法族 | 层级 | 与 JACRU-M2 的真实关系 | 已占据的思想 | 尚未覆盖的部分 |
|---|---|---|---|---|
| NeRIF、NeDF、Neural Refractive Index Primitives | **A：任务直接** | 都直接从稀疏 BOS/BOST 位移重建连续场 | 神经隐式场、射线采样、梯度/折射率一致性、真实火焰验证 | 跨实例摊销的残差算子、经典底座展开、显式无序相机集合通常不是核心 |
| 有限孔径 cone-ray BOS | **A：物理直接** | 与 JACRU 的 cone-ray/finite-aperture 核心最接近 | 已把锥射线模型嵌入神经重建，并展示冲击界面恢复 | 未必是经典底座上的跨样本 learned residual unrolling |
| BOST/CBOS、GRU-BOST、TDBOST | **A：任务直接** | 给出传统、深度时序和 4D BOST 先例 | BOST 逆解、稀疏/时序学习、4D 张量分解均已有 | 不自动包含 JACRU 的五项联合机制 |
| Learned Primal-Dual、MoDL | **A-：机制直接** | 最强的“物理算子进入可学习迭代”碰撞 | forward/adjoint 或数据一致性嵌入展开网络早已有之 | 未针对有限孔径 BOST、界面块和可变相机集合 |
| Neural Inverse Operators、PI-DION | **B：逆算子近邻** | 直接学习观测到未知函数的逆映射 | operator-valued observation、逆问题函数空间映射、无标签物理训练已有 | 不等同于离散 objective-consistent 的 BOST block unrolling |
| VIDON | **B：集合机制近邻** | 对数量和位置变化的传感器输入进行置换不变聚合 | “可变相机/传感器集合编码”本身不新 | 未提供 BOST cone-ray 数据一致性或界面更新 |
| GINO | **B：几何机制近邻** | 用几何信息和不规则点云连接潜在规则网格 | geometry-conditioned operator 本身不新 | 主要是变几何 PDE 前向算子，不是相机集合条件的 BOST 逆展开 |
| IONet、IANO、phi-DeepONet、level-set PINN | **B：界面机制近邻** | 显式使用子域、界面位置/拓扑或 level set | 界面感知编码、损失、反传和 discontinuity latent 均已有 | 未形成所述 BOST 经典底座 + 锥射线 VJP + 集合条件联合结构 |
| DeepONet、FNO | **C：基础架构** | 提供 operator learning 的标准参照 | branch/trunk、Fourier kernel、函数到函数映射不是新意 | 原始形式并不证明适合可变相机、有限孔径逆问题或锐界面 |

**术语警告：**相关光学文献中的 `CBOS` 通常指 **Colored Background Oriented Schlieren**，不是 cone-beam 或 cone-ray。JACRU 文稿应写全称，不能借缩写制造错误的技术谱系。

## 3. 已存在的思想

### 3.1 经典逆解后接学习校正已经存在

Deep Null Space Learning 已把经典重建器与学习到的校正组合，并用算子零空间约束数据一致性。因而，`classical reconstruction + neural residual` 不能作为 JACRU-M2 的新颖点。真正需要回答的是：BOST 特有的锥射线残差、界面状态和相机集合是否产生了**不可由通用后处理器替代**的可测机制收益。

### 3.2 learned operator correction 已经存在

Learned Operator Correction 针对不准确 forward model，分别修正数据空间和解空间中的 forward/adjoint。若 JACRU-M2 的“residual correction”实际是在补偿 thin-ray 与 finite-aperture forward mismatch，则必须与该路线正面对照，不能把“校正物理模型”写成首次。

### 3.3 physics-informed unrolling 已经成熟

Learned Primal-Dual 把 forward/adjoint 放入固定阶段的原始-对偶网络；MoDL 把 CNN 先验与显式数据一致性/数值求解块交替。`A`、`A^*`、残差或数据一致性进入网络不是新颖性来源。JACRU-M2 最多可争取的是**BOST 特定离散目标、界面块和相机集合条件的联合设计**。

### 3.4 inverse neural operator 与可变传感器编码已有先例

Neural Inverse Operators 已研究从算子值观测到未知函数的逆映射；PI-DION 已研究用物理损失、无需目标标签训练深度逆算子。VIDON 已明确处理传感器数量/位置变化及置换不变输入。因此，不能把“inverse operator”“physics-informed inverse operator”或“variable camera set”单独列为首创。

### 3.5 geometry-conditioned operator 已存在

GINO 使用几何表征、点云图算子和 FNO 潜在网格处理变化几何。它不是 BOST 逆算法，但足以封堵“首个 geometry-conditioned neural operator”声明。JACRU-M2 必须证明相机集合条件化不是普通坐标拼接或 Deep Sets 外壳，而是改变了跨几何泛化或物理一致性。

### 3.6 锐界面表示与界面感知算子已有多条路线

- 参数 level set 与 phase-field tomography 已把二值/分段场和界面周长先验用于逆问题。
- IONet、IANO 和 phi-DeepONet 已分别使用子域分支、界面信息编码、几何位置编码及不连续 latent 表示。
- 2026 年 level-set PINN 预印本已在域逆问题中提出锐界面 level set、interface-aware backpropagation 和界面附近自适应采样。

因此，“显式界面”“界面感知反传”“神经算子处理不连续场”都不能单独声称新颖。

### 3.7 神经 BOST、有限孔径和 4D 重建均已有直接先例

- 传统 BOST/CBOS、TV/Tikhonov 重建早已存在；TV 用于火焰前缘等突变也已有直接示范。
- GRU-BOST、NeDF、NeRIF 与 Neural Refractive Index Primitives 已覆盖深度时序模型、神经偏折场和神经隐式折射率场。
- finite-aperture 工作已经提出 cone-ray forward，并把它嵌入神经重建，恢复冲击界面。
- TDBOST 已覆盖高时空分辨率 4D BOST 的张量分解路线。

所以 JACRU-M2 不能使用“首个神经 BOST”“首个稀疏视角神经 BOST”“首个有限孔径神经重建”“首个 4D BOST”等表述。

## 4. 仍可能存在的窄缺口

以下仅是**尚待证伪的联合缺口**。本次检索没有发现完全相同的端到端组合，不等于证明不存在：

1. **经典底座上的跨实例残差展开**：冻结一个强、可复现的 finite-aperture BOST 逆解 `B_G`，学习的是逐阶段增量，而不是从坐标随机初始化对单场拟合，也不是一次性后处理。
2. **离散目标一致的 cone-ray forward/VJP**：训练、推理和审计都调用同一离散 `A_G`；VJP 通过伴随测试和有限差分检查，不用另一个近似投影器制造虚假收益。
3. **非冗余的显式界面块**：界面变量对更新路径有可识别作用，并在参数量匹配的无界面模型、TV/Huber/level-set/phase-field 基线之外带来稳定收益。
4. **无序、可变数量相机集合条件化**：模型接受 `(geometry, displacement, mask, reliability)` 集合，在相机置换、缺失视角和未见几何上保持性能；必须超过 VIDON 式集合编码和普通坐标拼接。
5. **有限孔径与相机 nuisance 联合鲁棒性**：孔径、标定扰动、遮挡和可靠度不是训练集标签泄漏，而是进入 forward 或条件编码，并通过独立重投影审计。
6. **no-harm 的界面选择性**：在无跳跃/平滑场上，界面模块不制造伪表面；在锐界面场上才提供边界定位收益。这个“选择性收益”比平均 RMSE 小幅下降更可能构成可发表机制。
7. **同一模型跨 view-count/geometry 泛化**：不是为每组相机重新训练一个模型；否则“set operator”贡献只剩输入包装。

更稳妥的候选贡献句式是：

> We evaluate a BOST-specific residual-unrolling architecture that combines a frozen finite-aperture classical inverse, objective-consistent cone-ray VJPs, explicit interface-state updates, and permutation-invariant conditioning on variable camera sets.

这句话仍然只是**被评估的方法描述**，不是“首次”或“优于现有方法”的结论。

## 5. 不可声称

在完整 M2 实现、强基线和独立审计完成前，不得声称：

- `JACRU-M2 is novel`、`first`、`首次`、`填补空白`或“此前没有类似方法”。
- 首次提出 classical inverse + neural residual/correction。
- 首次把 forward/adjoint、物理损失或数据一致性用于 learned unrolling。
- 首次提出 neural inverse operator、physics-informed inverse operator、geometry-conditioned operator 或 variable-sensor set operator。
- 首次进行 interface-aware、level-set、phase-field 或 discontinuity-capturing neural reconstruction。
- 首次对 BOS/BOST 使用神经网络、神经隐式场、稀疏视角、有限孔径 cone-ray 或 4D 重建。
- M0/M1 已证明 interface gain、JACRU superiority 或可以打开 final split。
- 在不匹配的相机划分、forward fidelity、迭代次数、算子调用数、参数量或 wall-clock 预算下宣称优越。
- 只凭训练/验证 forward residual 下降，宣称真实场恢复或界面恢复更准确。

允许的保守表述只有：

> 在本次已核验的 25 篇核心原始工作中，各组成思想均有先例；尚未发现与所述五项联合条件完全相同的实现。该观察只形成实验假设，JACRU-M2 的新颖性仍未证明。

## 6. 最小可发表贡献门槛

下面是 JACRU-M2 的**项目级 go/no-go 门槛**，不是对所有期刊的通用规则。

### Gate A：方法身份必须清楚

- 一页算法或伪代码明确 `B_G`、`A_G`、VJP、learned residual、interface state、set encoder 和投影/约束的位置。
- 说明哪些权重跨阶段共享、哪些量冻结、训练标签是什么、是否跨样本摊销。
- 用伴随内积测试和有限差分方向导数验证 cone-ray VJP；测试容差、dtype 和随机种子固定。

### Gate B：必须击败正确的强基线

至少包括：

1. 同一 finite-aperture forward 下的 CGLS/LSQR、Tikhonov、TV、Huber。
2. level-set 或 phase-field 界面逆解。
3. `classical base + parameter-matched CNN/FNO post-corrector`。
4. 使用同一 `A_G/A_G^*`、相同阶段数和调用预算的 Learned Primal-Dual 或 MoDL 型展开。
5. 一个直接神经隐式 BOST 基线，如 NeRIF/NeDF/Neural Refractive Index Primitives 的可比实现。
6. `VIDON-style set encoder` 与简单 camera-coordinate pooling。

所有基线必须使用相同 train/validation/audit 场、相机划分、噪声、孔径模型、可见 mask 和预处理。若 forward fidelity 或算子调用预算不一致，结果只能标为诊断，不能支撑优越性。

### Gate C：联合机制必须被消融识别

至少做六个独立消融：去掉 classical base、去掉 learned residual、cone-ray 改 thin-ray、VJP 改不一致近似、去掉 interface state、set encoder 改定长拼接。另做：

- 参数量/计算量匹配的界面与非界面模型。
- 相机输入随机置换不变性测试。
- view dropout、未见相机数量、未见几何和 aperture shift。
- 平滑/无界面负对照，统计伪界面率。
- forward-only、inverse-only 和联合误差分解，避免把标定误差归因于 learned residual。

### Gate D：证据协议必须阻止数据泄漏

- 训练、调参、audit、final 四层数据/场景固定且内容哈希可追踪；final 在 Gate A-C 通过前保持关闭。
- 至少 5 个随机种子、3 种相机几何、3 种 view count，并报告均值、方差与配对置信区间。
- 合成数据同时用 matched forward 和独立高保真 audit forward；真实数据使用 held-out camera reprojection，不以训练相机重投影代替外部证据。
- 记录参数量、峰值显存、训练时间、推理时间、每次重建的 forward/VJP 次数。

### Gate E：预注册的最低量化信号

建议在打开 final split 前预注册以下内部阈值：

- 相对最强**匹配预算**基线，体场主指标平均相对改善至少 `5%`，且配对置信区间不跨 0。
- 在真正含界面的测试上，ASSD 或 HD95 至少改善 `10%`，并同时提高容差为 1 voxel 的界面 F1 至少 `0.03`。
- 在无界面/平滑控制上，主指标退化不超过 `1%`，伪界面体素率不显著高于最强基线。
- held-out camera reprojection error 不高于最强基线的 `1.05×`；否则场误差改善可能来自不可观测 hallucination。
- 在未见 geometry/view-count/aperture 上，至少保留一半的 in-distribution 相对收益。

若只改善 RMSE 而界面指标、held-out reprojection 或平滑 no-harm 失败，应报告为负结果或工程折中，不能包装为“jump-aware mechanism 已验证”。

### Gate F：达到什么才算最小论文贡献

最小可发表单位不是一个新模块名，而是同时满足：

1. 可复现的 BOST 特定联合算法；
2. 对最接近先例的公平比较；
3. 证明界面块与 set conditioning 各自不可替代的消融；
4. 在合成高保真 audit 和真实 held-out reprojection 上都不破坏数据一致性；
5. 对失败范围、孔径/视角边界和计算成本作完整披露。

满足这些条件后，仍建议把贡献限定为“a BOST-specific joint realization and empirical validation”，而不是未经更广泛检索支持的“the first”。

## 7. 最相关的 25 篇原始论文

### 7.1 经典底座、校正与展开

| # | 原始来源 | 为什么读 | 必须抽取的内容 |
|---|---|---|---|
| 1 | Schwab, Antholzer, Haltmeier, [Deep Null Space Learning for Inverse Problems](https://arxiv.org/abs/1806.06137) (2018) | 与“经典重建 + learned correction”最直接碰撞 | 组合形式、null-space data consistency、收敛条件；据此定义 JACRU 后处理基线 |
| 2 | Lunz et al., [Learned Operator Correction for Inverse Problems](https://epubs.siam.org/doi/10.1137/20M1338460) (SIAM J. Imaging Sciences, 2021) | 区分“校正解”与“校正错误 forward/adjoint” | 数据空间/解空间校正、迭代耦合方式、对模型失配的假设 |
| 3 | Adler & Öktem, [Learned Primal-Dual Reconstruction](https://arxiv.org/abs/1707.06474) (2018) | 物理算子进入学习展开的标准强基线 | 每阶段 forward/adjoint 调用、状态通道、阶段数、非线性 forward 支持 |
| 4 | Aggarwal, Mani, Jacob, [MoDL: Model-Based Deep Learning Architecture for Inverse Problems](https://arxiv.org/abs/1712.02862) (2017/2019) | CNN 先验与显式数据一致性交替的强基线 | 共享权重、CG/data-consistency block、训练损失、计算预算 |

### 7.2 逆神经算子、集合输入与几何条件

| # | 原始来源 | 为什么读 | 必须抽取的内容 |
|---|---|---|---|
| 5 | Lu et al., [Learning nonlinear operators via DeepONet](https://www.nature.com/articles/s42256-021-00302-5) (Nature Machine Intelligence, 2021; [official code](https://github.com/lululxvi/deeponet)) | operator learning 的 branch/trunk 基础 | 固定传感器假设、输入/查询表示、误差与泛化设置；不能把原始 DeepONet 当可变相机证明 |
| 6 | Li et al., [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895) (2020; [official library](https://github.com/neuraloperator/neuraloperator)) | FNO 是最常见残差算子/后处理器参照 | 规则网格、Fourier modes、分辨率迁移、参数和 FLOPs 匹配方式 |
| 7 | Molinaro et al., [Neural Inverse Operators for Solving PDE Inverse Problems](https://proceedings.mlr.press/v202/molinaro23a.html) (ICML, 2023) | 与“观测算子到未知函数”的 M2 目标最接近 | operator-valued observations、DeepONet+FNO 组合、训练分布、逆问题稳定性与 OOD 设置 |
| 8 | Cho & Son, [Physics-Informed Deep Inverse Operator Networks](https://openreview.net/forum?id=0FxnSZJPmh) (ICLR, 2025; [arXiv](https://arxiv.org/abs/2412.03161)) | 直接封堵“首个 physics-informed inverse operator” | 无目标标签训练、物理残差、partial measurements、稳定性结果与适用 PDE 类别 |
| 9 | Prasthofer, De Ryck, Mishra, [Variable-Input Deep Operator Networks](https://arxiv.org/abs/2205.11404) (2022) | 可变数量/位置传感器和置换不变编码的直接先例 | set aggregation、传感器位置编码、universality、缺失/变数量测试协议 |
| 10 | Li et al., [Geometry-Informed Neural Operator for Large-Scale 3D PDEs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html) (NeurIPS, 2023; [official code](https://github.com/neuraloperator/neuraloperator)) | geometry-conditioned operator 的直接机制近邻 | SDF/点云几何输入、GNO-to-grid/FNO 流程、变几何划分与尺度 |
| 11 | Lanthaler et al., [Nonlinear reconstruction for operator learning of PDEs with discontinuities](https://arxiv.org/abs/2210.01074) (2022) | 解释线性重建型 operator 在不连续场上的结构性困难 | lower bound 条件、FNO/shift-DeepONet 的非线性重建优势、哪些结论不能外推到 BOST |

### 7.3 level set、phase field 与界面感知神经算子

| # | 原始来源 | 为什么读 | 必须抽取的内容 |
|---|---|---|---|
| 12 | Kadu, van Leeuwen, Batenburg, [A parametric level-set method for partially discrete tomography](https://arxiv.org/abs/1704.00568) (2017) | 逆断层中显式几何/分段场的强经典基线 | level-set 参数化、背景/异常假设、双层或交替优化、界面自由度 |
| 13 | Dunbar & Elliott, [Binary recovery via phase field regularization for first traveltime tomography](https://arxiv.org/abs/1811.02865) (2018) | phase-field + 数据失配 + 周长松弛的直接逆问题先例 | double-obstacle 能量、Gamma 收敛、离散化与不可微 forward 处理 |
| 14 | Wu et al., [IONet: Learning to Solve PDEs with Interfaces](https://arxiv.org/abs/2308.14537) (2023) | 子域分支和 interface-aware physics loss 的先例 | 多 branch/trunk、界面条件损失、跨子域耦合；确认其主要是 PDE forward 而非 BOST inverse |
| 15 | Roy et al., [phi-DeepONet: A Discontinuity Capturing Neural Operator](https://arxiv.org/abs/2604.08076) (2026 preprint) | 不连续输入/输出和界面 latent 的最新直接碰撞 | 多 branch、one-hot domain decomposition、非线性 interface embedding、physics/interface loss |
| 16 | Wang et al., [Cross-Field Interface-Aware Neural Operators for Multiphase Flow Simulation](https://ojs.aaai.org/index.php/AAAI/article/view/39887) (AAAI, 2026; [official code](https://github.com/genshinzx/aaai26-iano-code)) | 界面位置/拓扑作为显式辅助几何的先例 | interface-aware function encoding、geometry-aware positional encoding、低数据/噪声协议 |
| 17 | Yao, Li, Qian, [Level-set physics-informed neural networks for domain inverse problems of gravimetry](https://arxiv.org/abs/2607.03772) (2026 preprint) | “锐界面 + 域逆问题 + interface-aware backprop”最近邻 | level-set 网络、修改的界面演化导数、梯度支持区、自适应界面采样；与精确离散 VJP 的区别 |

### 7.4 BOS/BOST/CBOS 直接先例

| # | 原始来源 | 为什么读 | 必须抽取的内容 |
|---|---|---|---|
| 18 | Sourgen, Leopold, Klatt, [Reconstruction of the density field using the Colored Background Oriented Schlieren Technique (CBOS)](https://doi.org/10.1016/j.optlaseng.2011.07.012) (Optics and Lasers in Engineering, 2012) | 澄清 CBOS 的真实含义和经典密度场重建链 | 彩色编码、观测形成、反演与标定假设；禁止把 CBOS 误写成 cone-ray 缩写 |
| 19 | Grauer et al., [Instantaneous 3D flame imaging by background-oriented schlieren tomography](https://doi.org/10.1016/j.combustflame.2018.06.022) (Combustion and Flame, 2018) | 多相机火焰 BOST 与锐前缘正则化的经典直接基线 | 23-camera geometry、Tikhonov/TV、火焰前缘、评价和实验不确定性 |
| 20 | Bo et al., [Background-oriented Schlieren tomography using gated recurrent unit](https://doi.org/10.1364/OE.505992) (Optics Express, 2023) | 直接封堵“首个 learned BOST” | GRU 如何利用角度/空间相关性、输入输出组织、视角设置、传统基线与泛化限制 |
| 21 | Molnar et al., [Forward and inverse modeling of depth-of-field effects in background-oriented schlieren](https://doi.org/10.2514/1.J064095) (AIAA Journal, 2024; [arXiv](https://arxiv.org/abs/2402.15954)) | 与 JACRU cone-ray/finite-aperture 声明最直接碰撞 | cone-ray forward、thin-ray mismatch、aperture sweep、神经重建、TV/Euler 约束、shock interface 指标 |
| 22 | Li et al., [Neural deflection field for sparse-view tomographic background oriented Schlieren](https://doi.org/10.1063/5.0241191) (Physics of Fluids, 2024; [arXiv](https://arxiv.org/abs/2409.19971)) | 稀疏视角 BOST 的直接神经场基线 | density-gradient/deflection 表示、位置编码、分层采样、limited-angle 和噪声协议 |
| 23 | He et al., [NeRIF: Neural refractive index field for BOS tomography](https://doi.org/10.1063/5.0250899) (Physics of Fluids, 2025; [arXiv](https://arxiv.org/abs/2409.14722)) | 当前最直接的神经隐式折射率场近邻 | `n` 与 `grad n` 输出、AD/ND 一致性、射线/位移损失、8-view 和 held-out reprojection；确认其 per-instance 性质 |
| 24 | Lu et al., [Neural Refractive Index Primitives for Flame Field Reconstruction Using Background-Oriented Schlieren](https://doi.org/10.1016/j.combustflame.2026.115082) (Combustion and Flame, 2026; [arXiv](https://arxiv.org/abs/2605.11454)) | 最新直接 neural-implicit BOST 竞争者 | compact MLP、multiresolution hash、automatic-discrete gradient loss、3D mask、真实火焰和速度/内存 |
| 25 | He et al., [Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction](https://doi.org/10.1145/3809488) (ACM TOG, 2026; [official code](https://github.com/Hyz617/TDBOST)) | 4D BOST 与高效时空表示的直接先例 | 张量分解形式、时间共享、forward/data term、内存/速度、相机与场景划分；静态 M2 不能借 4D 结果泛化 |

## 8. 阅读与抽取优先级

### P0：先决定 M2 是否值得实现

优先精读 #1、#2、#3、#7、#9、#17、#21、#22、#23、#24、#25。每篇统一抽取：

- 输入/输出的数学空间；是 per-instance 优化还是跨实例训练。
- forward、adjoint/VJP 和数据一致性出现在哪一层。
- 相机/传感器数量与位置是否可变，是否保证置换不变。
- 界面是显式状态、辅助输入、loss 还是仅由网络隐式吸收。
- train/validation/test 的场景、几何、视角和噪声划分。
- 与经典逆解的关系、算子调用次数、参数量、wall time 和 memory。
- 是否有 held-out view、真实数据、独立 forward 或 no-interface 负对照。

### P1：再决定“jump-aware”是否有机制含量

精读 #11--#17，建立统一界面指标：体场相对误差、ASSD、HD95、容差 F1、界面厚度偏差、伪界面率。若 JACRU 的界面状态不能在这些指标上超过 TV、level set、phase field 和参数匹配非界面网络，就应删除“jump-aware”贡献声明。

## 9. 最终原创性判定

| 判定项 | 当前状态 | 原因 |
|---|---|---|
| 组成模块新颖 | **RED** | 经典校正、算子校正、展开、逆算子、集合编码、几何条件、界面表示和神经 BOST 均有先例 |
| 五项联合实现可能有缺口 | **AMBER** | 本次有界检索未发现完全相同组合，但没有 absence proof |
| M2 已实现并可复现 | **RED** | 当前没有完成训练和跨样本审计的 learned residual operator 证据 |
| M2 优于强基线 | **RED** | 尚无匹配 forward、预算和划分的完整比较 |
| 可提出“首次/新颖”声明 | **RED** | 文献与实验两条证据链都未闭合 |
| 可继续受控实验 | **AMBER** | 只适合按 Gate A-E 实现、消融和证伪，不应先写结论 |

**最终结论：JACRU-M2 尚未证明新颖。**当前最诚实、也最有科研价值的下一步，是把它当作一个可能失败的联合机制假设：先完成 objective-consistent M2、强基线和 no-harm/held-out 审计，再决定是否保留 JACRU 这一方法名与原创性叙述。
