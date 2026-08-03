# C 路线一级来源文献红队：高度同构近邻与可主张边界

日期：2026-08-04  
范围：sparse-view 3D BOST / Schlieren tomography、learned warm start、neural operator / FNO / PINO、geometry-conditioned / diffeomorphism、exact-operator refinement  
证据状态：`PRIMARY_SOURCE_RED_TEAM_COMPLETE`  
科学状态：`algorithm_breakthrough=false`、`paper_success=false`、`global_uniqueness_proven=false`

## 硬结论

在截至 2026-08-04 本轮核对的论文、期刊和 arXiv 官方页面中，**没有找到一篇同时覆盖下列七个组件、且与当前 C 路线没有材料性差别的高度同构工作**：

1. 稀疏视角三维 BOST；
2. 部署时只读 observation 与显式已知 geometry 的摊销式 warm initializer；
3. geometry-conditioned / diffeomorphic 公共参考坐标处理；
4. 学习 proposal 经冻结离散 BOST 算子的精确伴随 `A_g^T` 提升回物理场；
5. 后端保持未修改的 CGLS / PCGLS / Krylov refinement；
6. 逐重建单元同时检查 field、full-gradient、interior-gradient、observation 的 non-inferiority / no-harm；
7. 精确 `A/A^T` 调用账之后，再检查 fresh-process wall time 与 whole-pipeline peak RSS。

但这个结论必须和另一半同时写：**七个组件中的每一个，以及多个二元或三元组合，都已经有明确在先工作。** 因此，当前没有“神经算子组件创新”“微分同胚组件创新”“warm start 组件创新”或“Krylov 组件创新”；最多只剩一个 **BOST-specific 组合与严格证据合同** 的待验证贡献。未发现同构论文不是全球唯一证明，更不是专利清查或对组内未发表工作的排除。

精确状态可写为：

> `NO_HIGHLY_ISOMORPHIC_PAPER_FOUND_IN_SCOPED_PRIMARY_SOURCE_REVIEW`  
> `COMPONENT_LEVEL_NOVELTY_REJECTED`  
> `NARROW_COMPOSITION_HYPOTHESIS_REMAINS`  
> `GLOBAL_UNIQUENESS_NOT_PROVEN`

## 七项同构判据

下表使用 `✓` 表示论文明确覆盖，`△` 表示有结构类比但对象或保证不同，`—` 表示没有。它是面向论文 claim 的技术判断，不是法律意义的专利意见。

| 一级来源 | A 稀疏 3D BOST | B observation + geometry initializer | C 微分同胚 / 参考域 | D 精确伴随 / 物理 lift | E 未修改 Krylov | F 逐单元多指标 no-harm | G 调用 + wall/RSS | 红队判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| [WB-IPM：FMT warm basis + iterative projection](https://arxiv.org/abs/2510.05926) | — | ✓ | — | △ | — | — | △ | **最危险的组装级近邻。** 三维光学逆问题、网络由测量生成 warm basis、`A/A^T` 与 Golub-Kahan/Krylov 都存在；但它重构并增强求解空间，后端不是未修改 CGLS，也没有 BOST 几何或当前逐单元成本合同。 |
| [70-view / nine-view 实验 BOS 数据与 NIRT](https://arxiv.org/abs/2508.17120) | ✓ | — | △ | △ | — | — | — | **最危险的任务级近邻。** 已在真实 Mach 4.8 流场做 nine-view neural-implicit BOS reconstruction；网络是逐实例优化的隐式场，不是摊销式 warm initializer。 |
| [NeDF](https://arxiv.org/abs/2409.19971)、[NeRIF](https://arxiv.org/abs/2409.14722)、[NRIP](https://arxiv.org/abs/2605.11454) | ✓ | — | △ | △ | — | — | △ | 已覆盖稀疏视角或三维 BOST 神经隐式重建、数值与真实火焰。它们直接阻断“首个神经三维 BOST / 首个 sparse-view BOST”表述。 |
| [BOST-GRU](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-23-39182)、[Pyramid BOST](https://link.springer.com/article/10.1007/s00348-025-04153-3)、[Hybrid refinement](https://doi.org/10.1063/5.0190778) | ✓ | △ | — | △ | — | — | △ | 直接网络重建、coarse-to-fine、projection / matrix correction 和初值影响均已有 BOST 先例；“网络加速 BOST”或“先粗后精”不是贡献。 |
| [Inverse acoustic NN warm start](https://arxiv.org/abs/2212.08736) | — | ✓ | △ | △ | △ | — | △ | **最危险的流程级近邻。** 测量进入网络得到初值，再交给经典 Gauss-Newton，并检验噪声和 limited aperture；差别主要是非 BOST、无显式 `A_g^T` lift 与当前成本合同。 |
| [NOWS](https://arxiv.org/abs/2511.02481) 与 [Learning to Warm-Start Fixed-Point Algorithms](https://jmlr.org/papers/v25/23-1174.html) | — | △ | — | △ | ✓ | — | ✓ | **最危险的 solver-shell 近邻。** neural operator / NN 产生初值、经典迭代器保持不变、迭代数与 wall time 同时核算已经成立。 |
| [EUV PINO warm start](https://arxiv.org/abs/2607.25330) 与 [neural-operator super-fidelity](https://arxiv.org/abs/2312.11842) | — | △ | △ | △ | △ | △ | ✓ | 已覆盖三维光学 PINO warm start，以及低保真到高保真初值再由传统 solver 收敛；阻断“首个三维光学 neural-operator warm start”。 |
| [Learned Primal-Dual](https://arxiv.org/abs/1707.06474) 与 [Learned ReSeSOp](https://arxiv.org/abs/2410.23061) | — | ✓ | △ | ✓ | — | △ | △ | learned inverse reconstruction 结合 forward / adjoint、数据一致性与模型误差已有成熟先例；当前差别只能是“只学初值、后端不改”和 BOST-specific 成本门。 |
| [FCG-NO](https://proceedings.mlr.press/v235/rudikov24a.html) | — | △ | — | △ | — | — | ✓ | neural operator 已作为 nonlinear preconditioner 进入 flexible CG，并有跨分辨率实验；当前固定网格 CNN 不能借用其 neural-operator 或 discretization-invariant 结论。 |
| [Geo-FNO](https://jmlr.org/papers/v24/23-0064.html)、[DIMON](https://www.nature.com/articles/s43588-024-00732-2)、[DNO](https://www.nature.com/articles/s42005-024-01911-3)、[DAFNO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html) | — | — | ✓ | — | — | — | △ | 学习或给定坐标变换、公共参考域、显式几何编码与跨几何 operator learning 已充分覆盖；微分同胚只能作为采用的机制，不是新颖性来源。 |
| [Spectrally Safe warm starts](https://arxiv.org/abs/2606.21828)、[dynamic-tomography adaptive restart](https://proceedings.mlr.press/v172/knopp22a.html)、[neural dual warm start with fallback](https://arxiv.org/abs/2605.09382) | — | △ | — | △ | △ | △ | ✓ | 局部物理安全、warm/cold restart、learned dual + exact solver + fallback 都已有先例；当前 truth-based cellwise gate 只是离线评价合同，不是部署时安全机制。 |

没有任何一行达到七项全覆盖。最接近的也不是一篇，而是四条谱系的交集：**WB-IPM 的光学逆问题组装、nine-view NIRT 的真实 BOST 任务、NOWS 的未修改 Krylov 壳、DIMON / Geo-FNO 的几何参考域。** 这正是当前组合仍可能形成论文边界、却绝不能把任一组件单独包装成创新的原因。

## 最危险近邻：按“最可能让审稿人拒绝新颖性”的顺序

### 1. WB-IPM：最危险的完整结构近邻

[WB-IPM](https://arxiv.org/abs/2510.05926) 已经在三维荧光分子层析中让 Attention U-Net 从测量生成 warm basis，再把它嵌入带精确 `A/A^T` 的 augmented flexible Golub-Kahan / hybrid projection。论文还明确指出：直接把网络输出当作普通初值可能恶化结果，因此必须设计学习与迭代的接口。

它会击穿以下宽泛说法：

- 首次把深度学习先验与三维光学层析迭代相结合；
- 首次把 learned warm information 放入 Krylov / Golub-Kahan；
- 首次发现普通神经初值可能伤害迭代并需要结构化接口。

当前可区分点只有：BOST 观测与相机几何、精确 `A_g^T` lift、**不修改** CGLS / PCGLS、以及逐单元 matched-accuracy 与完整成本门。若后端为了网络改成 flexible / augmented Krylov，这个关键差异会显著缩小。

### 2. Nine-view NIRT：最危险的 BOST 任务近邻

[开放 BOS tomography 数据工作](https://arxiv.org/abs/2508.17120) 提供 70 个实验视角、标定和代码，并用其中九个视角完成真实高速流 NIRT 重建与 validation-deflection 检查。它意味着“nine-view”“实验 3D BOS”“neural-implicit sparse-view reconstruction”均已有公开一级来源。

当前方法不能用“视角只有九个”本身支撑新颖性。真正差别必须是**摊销式 observation / geometry-only 初值 + exact-operator Krylov refinement + 成本与伤害合同**，并最终在真实 BOST 上与 NIRT / NeRIF 类完整管线公平比较。

### 3. Inverse acoustic warm start：最危险的工作流近邻

[Inverse acoustic NN warm start](https://arxiv.org/abs/2212.08736) 已经实现“传感器测量 -> 神经网络初值 -> 经典 Gauss-Newton”，并检验噪声与有限孔径。它不是体积 BOST，也没有当前的伴随提升和逐单元成本合同，但已经吃掉“measurement-only learned initializer followed by classical inverse refinement”这一上层流程。

### 4. NOWS：最危险的未修改求解器近邻

[NOWS](https://arxiv.org/abs/2511.02481) 明确让 neural operator 产生 CG / GMRES 初值，同时保持离散与 solver infrastructure 不变，并报告迭代数和端到端时间。因此“neural operator + unchanged Krylov”不是新颖性。当前必须证明的是 BOST 特定的 observation / geometry 输入、`A_g^T` lift、逐单元 no-harm 评价与真实资源优势。

### 5. NeDF / NeRIF / NRIP：最危险的组内主题近邻

[NeDF](https://arxiv.org/abs/2409.19971) 针对 sparse-view TBOS，用隐式 density-gradient field、位置编码和分层采样做逐实例重建；[NeRIF](https://arxiv.org/abs/2409.14722) 与 [NRIP](https://arxiv.org/abs/2605.11454) 则将神经隐式折射率表示推进到数值与真实火焰。它们并非摊销式 warm start，却已经覆盖“神经表示改善 BOST 空间分辨率、噪声与计算成本”的主叙事。

### 6. EUV PINO 与 Spectrally Safe：堵住两个容易误写的卖点

[EUV PINO](https://arxiv.org/abs/2607.25330) 已把因子化 FNO / PINO 用作三维光学 PSFD solver 的 warm start；[Spectrally Safe Neural Operator Warm-Starts](https://arxiv.org/abs/2606.21828) 已经说明低平均误差仍可能造成局部物理缺陷，并对 Newton / Krylov 后端加入安全导向修复。于是“首个光学 PINO warm start”和“首次关注 neural warm start 的局部安全”都不可写。

## 不能再声称什么

以下句子应从题名、摘要、引言贡献点和答辩幻灯片中删除：

1. **first neural / AI reconstruction for 3D BOST**；GRU、NeDF、NeRIF、NRIP 与 NIRT 已覆盖。
2. **first sparse-view or nine-view neural BOST**；NeDF 与 nine-view 实验 NIRT 已覆盖。
3. **first learned warm start for inverse imaging / optical tomography**；inverse acoustic warm start 与 WB-IPM 已覆盖。
4. **first neural-operator warm start with an unchanged Krylov solver**；NOWS 已覆盖。
5. **first 3D optical FNO / PINO warm start**；EUV PINO 已覆盖。
6. **first learned reconstruction using exact forward / adjoint**；Learned Primal-Dual、Learned ReSeSOp 与 WB-IPM 已覆盖相关范式。
7. **first geometry-conditioned / diffeomorphic neural operator**；Geo-FNO、DIMON、DNO、DAFNO 等已覆盖。
8. **diffeomorphism equivariant**；仅做公共坐标 canonicalization 不等于证明群作用下的交换律或等变性。没有形式命题和跨变换实验时，只能写 `coordinate-conditioned` 或 `diffeomorphic transport`。
9. **neural operator / discretization invariant**；当前固定 `32 x 16 x 16` CNN 没有跨离散共享参数与收敛证据，只能称 learned warm initializer。FNO 架构本身也不能自动替代任务级跨分辨率验证；[FNO](https://openreview.net/pdf?id=c8P9NQVtmnO) 与 [PINO](https://arxiv.org/abs/2111.03794) 的函数空间主张不能直接借用。
10. **exact physics**；当前最多是“对冻结离散 straight-ray proxy 使用 exact `A_g/A_g^T`”。[BOS deflection error analysis](https://arxiv.org/abs/2607.15567) 明确展示 thin-object、boundary-index、paraxial 与 perpendicularity 等近似层级；它不等于真实曲折光线、相机和实验物理全精确。
11. **cellwise no-harm is a deployable safety guarantee**；当前 gate 读取真值，只能作离线 non-inferiority 评价。除非另有 observation-visible abstention / fallback 并在外部数据验证，否则不能称运行时安全。
12. **fewer `A/A^T` calls means faster or lower-memory**；必须等 fresh-process wall 和 whole-pipeline RSS 实测。
13. **global uniqueness、first、SOTA、broad generalization、real-BOST acceleration**；本轮检索和当前代理证据都不支持。

## 仍可能成立的贡献边界

当前最稳妥、也最有含金量的贡献假设不是一个新网络层，而是下面这个窄组合：

> We investigate whether an amortized, deployment-visible, geometry-conditioned initializer for sparse-view 3D BOST can be lifted through the exact adjoint of a frozen discrete BOST operator and passed to an unmodified Krylov refinement, while satisfying preregistered per-cell multi-metric non-inferiority and full operator/resource accounting.

中文：

> 研究一个只使用部署可见观测与已知几何的摊销式三维 BOST 初值，能否通过冻结离散 BOST 算子的精确伴随提升进入物理场，再交给未修改 Krylov 精化，并同时满足结果前冻结的逐单元多指标非劣与完整算子/资源成本账。

它最多包含四类可防御贡献：

1. **BOST-specific solver interface。** observation / geometry-only proposal、exact adjoint lift 与 unchanged Krylov 的具体接口，而不是 warm start 概念本身。
2. **严格评价合同。** 轨迹与几何留出、逐单元 field / full-gradient / interior-gradient / observation 非劣、harm tails、相同后端与 exact-call ledger。它是实验设计贡献，不是安全定理。
3. **可证伪的资源结论。** 只有外部工况和真实 BOST 在 matched accuracy 下同时减少 exact calls、fresh wall 与 whole-pipeline RSS，才形成有意义的工程贡献。
4. **清晰失败边界。** 如果跨几何、噪声、曲折光线或真实标定后失效，严格定位何时 learned initializer 不应被接受，也可能构成有价值的负结果。

微分同胚、FNO / PINO、CNN、`A^T`、CGLS 与 no-harm 思想均只能作为采用的已有组件。论文贡献必须落在**它们如何为 BOST 组合、如何公平验证、以及组合是否真的在外部和真实数据上成立**。

## 审稿人最可能的五次攻击

| 审稿攻击 | 当前诚实回答 | 要让回答有说服力，最终证据必须是什么 |
|---|---|---|
| “这就是 NOWS / WB-IPM 换到 BOST。” | 上层范式确实相同；材料差别是 BOST observation/geometry-only 输入、exact adjoint lift、unchanged CGLS 与逐单元成本合同。 | 与 NOWS-style initializer、WB-IPM-style learned basis 或可实现的强近邻公平比较；清楚披露为何某个近邻不能同预算复现。 |
| “nine-view BOST 已经被 NIRT 做过。” | 是；nine-view 不是创新。 | 在同一真实数据与标定上比较完整 NIRT / NeRIF 轨道和 warm-start + Krylov 轨道，分别报告精度、调用、wall、RSS 与训练/逐实例优化成本。 |
| “微分同胚就是 DIMON / Geo-FNO。” | 是；公共参考域不是创新，只是几何条件机制。 | 有/无 transport、错误 transport、geometry token、相机条件和未见几何组合的消融；不使用 first/equivariant 表述。 |
| “你的 3D CNN 不是 neural operator。” | 当前确实不是。 | 在固定网格阶段称 learned initializer；只有跨分辨率、参数共享和离散收敛证据成立后才升级术语。 |
| “no-harm 用了真值，部署时怎么知道？” | 当前是离线评价门，不是在线安全门。 | 若要声称安全，另建只看 observation residual / uncertainty 的 abstention 或 fallback，并在未见工况验证风险覆盖与回退成本。 |

## 对当前论文定位的直接裁决

- **没有找到高度同构单篇工作：是。** 这是本轮有价值的 claim 边界进展。
- **组件级原创性：否。** 每个组件和多个组合都有在先工作。
- **可投稿的方法贡献已经成立：否。** 还缺独立公开工况、真实资源和真实 BOST 迁移。
- **算法突破：否。** `algorithm_breakthrough=false`。
- **最合理的论文定位：** BOST-specific hybrid reconstruction 与严格 empirical audit，而不是“提出一种全新的 neural operator / diffeomorphic solver”。
- **必须向何远哲师兄补做的非公开核对：** 组内是否已有未发表的 observation-conditioned initializer、learned backprojection、warm-start CGLS、相关专利或在投稿工作。本轮公开文献检索不能替代这一项。

## English executive red-team summary

No single primary-source paper located in this review, through 2026-08-04, is highly isomorphic to the complete seven-part target: sparse-view 3D BOST; an amortized observation-and-explicit-geometry-only initializer; diffeomorphic/reference-chart conditioning; an exact adjoint lift through the frozen discrete BOST operator; unmodified Krylov refinement; per-cell field/gradient/observation non-inferiority; and exact-call plus fresh wall/RSS accounting.

This is only a scoped negative search result. Component-level novelty is rejected. WB-IPM is the most dangerous assembled optical-inverse neighbor; the open nine-view NIRT study is the closest BOST-task neighbor; the inverse-acoustic warm start is the closest workflow neighbor; NOWS is the closest unchanged-Krylov neighbor; and DIMON / Geo-FNO / DNO are the closest geometry-reference-domain neighbors. The defensible contribution, if external and real-BOST gates eventually pass, is a narrow BOST-specific composition and falsifiable evaluation contract. It is not a new neural-operator, diffeomorphism, Krylov, adjoint, or safety primitive.

Recommended claim status: `no fully isomorphic work found in the scoped review`; never `first`, `globally unique`, or `state of the art`.
