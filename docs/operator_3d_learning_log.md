# 3D 逆问题学习持续日志

日期：2026-07-16

这份日志只记录我在读懂和复核这条实验线时真正学到的东西。重点不是把结果写成“模型越来越强”，而是把每次尝试的前提、数字、失败原因和下一步验证条件留下来。

## 先把证据等级说清楚

- **L0：真实实验/论文证据。** 目前没有。这里没有 OpenBOS/OERF 真实测量，也没有论文级 superiority 结果。
- **L1：预注册的 held-out synthetic development。** 有固定配置、固定 checkpoint 和首开前冻结的门禁，但数据仍是 synthetic proxy；只有训练/校准/部署特征合同一致时才能支持或否定一个开发假设，不能直接证明真实装置有效。
- **L2：合成数据上的 post-open 诊断。** 已经看过结果后才提出规则、挑 ensemble 或分析机制，只能生成下一轮假设，不能倒写成预注册成功。
- **L3：实现/数据合同检查。** 例如哈希、调用次数、梯度方向、样本归属和字段完整性。它说明实验做得是否可审计，不等于方法效果。

下面每段都会标明主要证据等级。所有“增益”都是相对于该段明确写出的基线；正数只表示数值指标变好，不自动表示方法成功。

## 1. 先问对问题：网络到底要解决什么

原来的问题是：只用 source 相机观测，学习一个三维修正场，再让不同 target 相机通过各自前向算子解码。最初的直觉是，网络也许能直接猜出 source 没解释完的 target residual。

但第一轮很快暴露出两个问题。第一，`zero_correction` 这个极简单的基线不能省略：它就是不再声称 residual 可以迁移。第二，同一个物理场应该产生一个共享的三维修正，而不是因为换了一台 target 相机就产生另一个场。

因此学习目标逐渐从“网络单独重建”改成了更谨慎的两种可能：

1. 共享场 prior 作为经典 PBB/CG 的起点或低预算辅助；
2. 学习真正独立的物理缺口，例如低保真 forward model 与高保真算子之间的失配，或者 4D 流场中的时间突变。

这不是措辞变化，而是由后面的数字逼出来的研究定位。共享场网络相对 PBB-32 总体差 **15.83%**，不能再包装成通用重建器。

## 2. V5P：低预算 hybrid 看起来有一点收益，但没有通过门禁

**问题。** V5O 的事后预算曲线在代理预算 `B=11` 附近出现过约 `+3.38%`，所以要验证一个更窄的假设：固定 `anchor=0.1`、8 步 prior-anchored PBB，能否在较少 source operator calls 下接近或超过 PBB-9。

**做了什么。** 在打开新 target labels 前冻结了配置和三种子 checkpoint。候选每个 field 精确执行 `8F/9A`，主基线 PBB-9 执行 `9F/9A`；使用 6 个新 rigs、3 个 topology families、360 个 field、720 个 target rows、18 个 `rig × family` cells。门槛提前写死为：总体 gain 至少 3%、正 cell 至少 75%、最坏退化不超过 5%。没有构造 design-lock rows，也没有用 fresh labels 选参数。

**结果。** 候选相对 PBB-9 的 cluster-mean target standardized RMSE 为 **+2.575%**，没有达到 3%；只有 **8/18 = 44.44%** cells 为正，没有达到 75%；最坏 cell 退化 **6.643%**，超过 5%。绝对 RMSE 为：PBB-9 **3.0336**，候选 **2.9555**，PBB-11 **2.9938**，PBB-32 **2.8495**。也就是说候选确实比 PBB-9 略好，但没有稳定地赢，而且充分迭代的 PBB-32 更好。

成本也没有形成加速证据：本机单次候选总时间 **0.2608 s**，PBB-9 为 **0.0507 s**，约慢 **5.14 倍**。候选少一次 source forward 的调用账本，不能抵消三种子 CNN 的时间。

**为什么失败。** 失败不是因为平均数完全为负，而是因为收益不够大、不够普遍、尾部有伤害，而且真实 wall-clock 方向相反。尤其 `tilted_flame_brush` 在 6/6 rigs 为正，但 `triple_jet_merger` 在 0/6 为正，说明 prior 对形态有偏好；不能把一个 family 的好处平均到所有形态上。

**学到什么。** “少一次算子调用”不等于“整体更快”；“比 PBB-9 好一点”不等于“低预算方法成立”；field truth 诊断也不能替代 target residual 主门禁。更重要的是，失败发生在首开前已冻结规则的 synthetic development 上，所以可以诚实地说这条低预算假设没有过门，而不是继续给同一 prior 加结构。

**证据等级。** **L1 + L3**。原始报告字段是 `preregistered_fresh_synthetic_development` / `FRESH_DEVELOPMENT_NO_GO`；本文统一称 held-out synthetic development，且它仍只是合成弱偏折 proxy，不是论文成果或 OERF 结果。

## 3. V5Q/V5R 留下的教训：能排序不等于能安全路由

V5P 打开数据后，V5Q 检查 source residual 是否能预测候选什么时候会赢；V5R 又模拟了一台不参加重建的 reserved camera，检查跨视角 residual 是否能做安全门。

V5Q 的 source residual gain 与 target gain 的 Spearman 相关在 field/cell 层是 **0.554/0.802**，6/6 rigs 方向一致，看起来有信号。但自然的“source gain 为正才采用候选”规则，整体只有 **+1.543%**，正 cells **55.56%**，被选 field 仍有 **22.77%** 受伤。相关性因此只能说明“值得在新数据验证”，不能授权在旧数据上扫阈值。

V5R 的 reserved camera 规则把整体 gain 提到 **+3.405%**，最坏 cell 退化压到 **0.986%**，但正 cells 只有 **12/18 = 66.67%**，selected harm 反而是 **33.71%**。其中一个 rig 的 reserved-to-target 相关只有 **0.016**，说明这个安全门并不跨 rig 稳定。

**学到什么。** 平均数、相关系数和最坏 cell 必须一起看；一个规则可以提高平均 gain，同时让被选中的场仍然大量受伤。source-only 或 reserved-view 的事后规则都没有资格被写成“可靠性 gate”。

**证据等级。** **L2**。V5Q/V5R 是 post-open mechanism diagnosis，生成了下一轮实验问题，但不能改判 V5P，也不能算新鲜验证。

## 4. 路线 B：把学习模块移到 forward model mismatch

前面的失败说明继续学习“观测到场”的 prior 很容易和 PBB/CG 重叠，也容易变成事后挑路由。路线 B 改问一个更具体的问题：如果便宜的 nominal forward operator `A0` 本身和真实/高保真算子 `A*` 不一致，能否用少量校准 probe 学一个结构化修正？

候选 GC-BiLOC 是：

```text
A_corr(g) = A0(g) + U C_phi(g) V^T
A_corr(g)^T = A0(g)^T + V C_phi(g)^T U^T
```

这里 `g` 是视角、孔径、cone、bend、焦距/物距等几何摘要；学习器只预测小矩阵 `C_phi(g)`，并且 forward 与 adjoint 强制成对。这样学习的是算子误差，不是直接偷看 target 去猜三维场。

路线 B 的最低要求也被写清楚了：先验证 operator discrepancy，再验证 adjoint/gradient，最后才跑 inverse。必须比较 `A0`、高保真 `A*`、global/nearest geometry、非神经 HOSVD/ridge，以及公开的 learned-operator 类基线；不能只和一个弱 baseline 比。

## 5. V5S：GC-BiLOC 的第一轮结构筛选没有打过最强便宜基线

**问题。** 低秩结构是否真的能用几何参数预测，且比一个直接使用完整 discrepancy 的便宜 ridge 更好？如果连算子层都没有优势，就不该进入 PBB/CGLS inverse。

**做了什么。** 在 12 个 development rigs 上，每个 rig 有 168 个 measurements 和 9 个 probe fields；共有 144 个 voxels。选出的结构参数是 measurement rank `24`、voxel rank `24`、relative ridge `1.0`。报告使用高保真 truth matrices 做评分和共享子空间诊断，但没有把完整矩阵交给 geometry predictor；没有构造 design-lock rigs，也没有跑 inverse reconstruction。

**结果。** 最强便宜基线是 `full_matrix_geometry_ridge`：平均 probe forward relative error **0.03013**，平均 relative operator error **0.08902**，平均 discrepancy error **0.52170**，最坏 discrepancy error **0.67231**，最坏 gradient cosine **0.99692**。

GC-BiLOC ridge 的对应数字是 **0.05794**、**0.14155**、**0.82849**、**0.86895**、**0.9960209366**。报告用 mean relative discrepancy error 计算相对改善，得到 **-58.81%**；它不是 relative operator error 的改善值。预设至少要 **+10%**，因此决定为 **`GC_BILOC_DEVELOPMENT_NO_GO`**。

有一个容易误读的地方：GC-BiLOC 的 gradient cosine 仍然很高，平均 **0.99774**，并不代表它成功。梯度方向大体对，不等于它把 operator discrepancy 的幅度和结构学准了；在 inverse 中还要看残差、收敛、最坏方向和成本。

**为什么失败。** 当前 hand-designed synthetic mismatch 在 development rigs 上并没有表现出“低秩几何条件化模型优于完整 discrepancy ridge”的证据。更具体地说，低秩压缩丢掉了足够多的 discrepancy 信息；而且参数和结构是在 development rigs 上选择的，没有 fresh rigs 来证明泛化。

**学到什么。** “有物理形式的低秩模型”不自动比简单回归更好；先做 HOSVD/奇异谱检查是必要的。operator 层没有通过时，继续跑 inverse 只会把一个未验证的算子误差放大成更难解释的重建结果。V5S 的失败反而帮忙收窄了下一步：要么找到 discrepancy 真正低秩且随 `g` 平滑的条件，要么保留更简单的 ridge/HOSVD，不强行上 GC-BiLOC。

**证据等级。** **L2 的合成开发筛选 + L3 的算子审计**。V5S 没有首开前的公开预注册时间戳；源码哈希只说明本次运行用了什么代码。报告将证据标签严格写为 `synthetic_operator_structure_development_only`。它足以暂停当前候选，不能支持 inverse superiority、design lock、实验或论文结论。

## 6. V5T-V5V：先把“失配到底长什么样”拆开

V5S 失败后，最差的做法是直接把 MLP 加深。这里先连续做了三个结构诊断，分别问：参数变化是否能用局部导数表示、几何校准后剩余误差是否低秩、相机级卷积核是否足够。

### V5T：真参数都给你，局部切线仍然不够

V5T 是一个故意很宽松的 oracle 诊断：它直接使用 truth-side 参数偏移，不训练参数估计器。换句话说，它不是可部署算法，只问“如果参数全知道，这个表示能不能装下失配”。

- 高保真 renderer 使用名义参数时，总 discrepancy error 是 **0.4672**。
- 一阶局部 tangent 是 **0.5050**，没有比高保真名义模型更好。
- 对角二阶近似爆到 **5.2666**，说明简单逐参数二阶项会严重失真。
- additive secant oracle 降到 **0.2607**，12/12 rigs 都优于 V5S 的 full-matrix ridge；但它对参数缺口本身的相对误差仍是 **0.5607**，超过预设 0.35。

通俗地说：相机参数的影响不是在名义点附近“沿几根直线轻轻移动”。大范围变化时，射线和有限孔径效应会耦合；一阶导数装不下，盲目加对角二阶项还会炸。secant 的正信号只说明“连接两个真实参数点”比局部泰勒展开合适，不说明我们能从标定图里估出这些参数。

**判决：**`CAMERA_LOCAL_TANGENT_REPRESENTATION_NO_GO_POSTOPEN`。证据等级 **L2**。

### V5U：把几何对齐后，剩余 renderer 误差仍不低秩

V5U 给低保真和高保真 renderer 使用同一套 truth geometry，只保留 path/aperture fidelity 差异。这个理想校准只消掉原始 discrepancy norm 的 **8.39%**；剩余误差仍有 **91.61%**。

校准后 full-matrix geometry ridge 的 error 是 **0.4762**，而 CAL-HOSVD 是 **0.8094**，相对差 **69.99%**。measurement/voxel 前 16 个奇异方向只解释 **34.28% / 38.37%** 能量。即使 oracle shared subspace 也只有 **0.7938**。

这说明当前 synthetic mismatch 不是“先标定，再用一个全局小低秩补丁”就能解决。全局 HOSVD 失败不是网络训练不够久，而是结构假设与位置相关的光学效应不匹配。

**判决：**`CALIBRATED_RENDERER_LOW_RANK_NO_GO_POSTOPEN`。证据等级 **L2 + L3**。

### V5V：每台相机一个 5×5 探测器核也装不下

V5V 测试 `A_corr,v = K_v A_low,v`：每个 view 用一个半径 2 的 5×5 measurement-space kernel。它非常紧凑，每 rig 只有 175 个核系数，几何 predictor 375 个系数；伴随缺陷为 **4.4e-16**，所以代码里的 forward/adjoint 是严格成对的。

但表示本身失败：oracle camera-local kernel 的 error **0.9043**，预测核 **0.9171**，而 full-matrix ridge 是 **0.4762**。这意味着有限孔径不是整台相机共享的平移不变 blur；同一视角内，不同探测器位置和射线深度也在改变核。

**判决：**`CAMERA_LOCAL_KERNEL_REPRESENTATION_NO_GO_POSTOPEN`。证据等级 **L2 + L3**。

## 7. V5W-V6A：逐射线局部核留下一个窄但真实的结构信号

### V5W：先隔离有限孔径一个因素

前面把 geometry、bend、cone、aperture 一起变，解释太混乱。V5W 固定 truth angles、cone、bend 和 path sampling，只比较 `radius=0` 与 truth finite aperture。这样回答的是一个干净问题：有限孔径能否近似为 measurement-side 或 voxel-side 的固定局部核？

- full-matrix geometry ridge：**0.8143**，worst **1.4181**。
- 最好预测的 measurement-side 核：**0.8074**，只改善 **0.854%**，worst **0.9724**。
- 最好 oracle voxel-side 核：**0.7058**，仍远高于 0.35 门槛。

固定核有一点改善尾部，却仍装不下主要误差。物理解释是：有限孔径的 point-spread/averaging 随 ray、depth 和视场位置变化，不是全图共享卷积。

**判决与等级：**`APERTURE_KERNEL_REPRESENTATION_NO_GO_POSTOPEN`，**L2 + L3**。固定核的尾部改善只生成逐 ray 假设，不是 gate pass。

### V5X：每条 ray 一个 3×3×3 体素核，oracle 已接近门槛

V5X 让每条 measurement ray 拥有自己的 27 系数局部 voxel kernel，再用 33 维射线/几何特征回归这些核。

- 完整 oracle row-wise kernel：mean **0.3587**、worst **0.4084**，已经接近预设 0.35。
- 但“两阶段先拟合核、再回归核”的预测器是 **0.8160**，比 full-matrix ridge 还差 **0.209%**。
- 参数只有 **891** 个，对比 full-matrix predictor 的 **991,872** 个，压缩约 **1113×**；worst rig 是 full-matrix 的 **76.55%**。

这一步第一次把“表示不行”和“学习不行”分开了：局部核表示有很大 oracle headroom，但单个 kernel target 不唯一；两阶段监督迫使网络拟合一个不稳定的中间答案。

**判决与等级：**`RAY_CONDITIONED_KERNEL_DEVELOPMENT_NO_GO`，**L2 + L3**。V5X 的预设 oracle 门槛是 **≤0.35**，实际 **0.3587**，因此“接近”不能改写成通过。

### V5Y/V5Z：端到端线性模型有效，但原优化器先炸了一次

V5Y 不再监督中间 kernel，直接让 891 个参数从 operator rows 端到端学习高低保真差异。原始 `lr=0.03`、batch 256 的训练发散，development error 达 **1.9769**，worst **4.1598**。这不是结构失败的干净证据，而是优化失败。

V5Z 将学习率降到 0.003、batch 提到 1024，加入 gradient clipping 和 cosine decay。稳定后 error 降到 **0.7707**，相对 full-matrix ridge 改善 **5.359%**，worst rig 比例 **0.7770**，精确伴随 dot-product defect 为 0。它没有达到 10% 门槛，但证明“直接训练可识别算子行”优于“两阶段拟合不唯一核”。

V5Y 之后把 V5X oracle 的诊断前提从 **≤0.35** 放宽为 **≤0.4**。这是看过 0.3587 后写出的 post-open eligibility rule，只允许继续检查端到端优化是否值得研究；它不能追溯性地把 V5X 改判为通过，也不是未来 fresh gate。

**判决与等级：**V5Y/V5Z 均为 `DIRECT_RAY_KERNEL_DEVELOPMENT_NO_GO`，**L2 + L3**。V5Z 是稳定化机制信号，不是 fresh improvement。

### V6A：小型超网络达到 8.08%，然后按规则停止扩容

V6A 使用 `33→64→64→27` 的小 MLP，根据每条 ray 的几何特征生成局部核；三个固定种子在 24 个内部 fit rigs 上训练、6 个 selection rigs 选步数，再在全部 30 个 development rigs 上按固定步数 refit。

- 三种子 ensemble error：**0.7485**。
- 相对 full-matrix geometry ridge 改善：**8.080%**，低于预设 10%。
- 正 rig：**6/12 = 50%**，低于预设 75%。
- worst rig：**1.0791**，是 full-matrix worst 的 **76.09%**。
- 但逐 rig 配对后，最大相对退化仍是 **13.69%**，伤害 rig 比例 **6/12 = 50%**；aggregate worst ratio 不能充当安全门。
- 单模型 **8,091** 参数，约比 full-matrix predictor 小 **122.6×**；三模型 ensemble 共 24,273 参数。
- 三个单种子 error 为 **0.7559 / 0.7526 / 0.7589**，说明不是某个幸运种子制造的数字。

这是目前最值得带给师兄看的算法信号：模型很小、伴随可精确构造，aggregate 平均和两组最大误差之比都改善。但逐 rig 仍有 13.69% 退化，只赢一半 rigs；所有训练都使用完整 synthetic operator rows 和 truth-calibrated geometry。因此不能叫“新算法已成功”，也不能继续在同一 opened development 集上加层、扫宽度直到超过 10%。

**判决：**`RAY_KERNEL_HYPERNET_DEVELOPMENT_NO_GO_STOP_CAPACITY_ESCALATION`。证据等级 **L2 + L3**。

## 8. 现在真正形成的研究假设

当前可以带着证据向何远哲提出的假设是：

> 在有限孔径 BOST 中，高低保真算子差异具有相机内、射线位置相关的局部体素核结构；用光学几何条件化的小型 hypernetwork 生成核，并由同一核严格构造 forward/adjoint，可能比逐 rig 完整矩阵回归更省参数、更稳健，但真实装置上的可辨识性和跨 rig 泛化尚未验证。

这条假设比“用 FNO 做三维重建”具体，因为它指出了：

1. **真实物理缺口：**有限孔径/景深造成位置相关而非全局平移不变的 averaging；
2. **算法结构：**ray-conditioned local 3D kernel hypernetwork；
3. **逆问题约束：**forward 与 adjoint 共用同一核，不能各学一个黑盒；
4. **当前边界：**只在 factor-isolated synthetic operator-level development 上有 8.08% 近信号；
5. **必须验证：**少量 calibration probes 能否辨识、fresh aperture/angle OOD、真实 held-out view、PBB/CGLS inverse impact 和 wall-clock。

这也和已有工作发生直接碰撞：cone-ray BOS 已显式处理有限孔径，Learned Operator Correction 已研究 forward/adjoint 修正，2026 年 differentiable geometry calibration 已做联合几何标定与重建。因此可能的新意不是“首次处理孔径”或“首次校正算子”，而只能是 **BOST 特定的逐射线局部核、query-efficient calibration、严格伴随和跨 rig 失败边界的联合证据**。

## 9. 当前总判断与停止规则

1. 纯 shared-field 网络和低预算 prior-anchored PBB 已被强 PBB 基线否掉。
2. source/reserved residual 有排序信号，却不能安全路由。
3. 全局低秩、局部 tangent、相机级固定 kernel 都不适合当前 finite-aperture discrepancy。
4. 逐射线、位置相关 3D kernel 有 oracle headroom；端到端线性/非线性学习分别改善 5.36%/8.08%，并降低 worst tail。
5. V6A 未过 10% 与 75% rig 门槛，必须停止在 opened synthetic development 上继续堆容量。
6. 下一步只能二选一：拿真实/独立 BOST calibration evidence 验证 ray-kernel，或拿何远哲连续序列启动 TRAIL-4D；没有新数据就先做基础学习和接口，不制造 fresh claim。

当前总状态是 **`NO_DESIGN_LOCK_OPEN`**、**`NO_INVERSE_SUPERIORITY_CLAIM`**、**`NO_REAL_BOST_EVIDENCE`**。

## 10. 下一步可执行实验

### 路线 A：RayKernel-DCO 的 fresh 验证

1. 在构造数据前冻结 v6b：模型宽度、核半径、训练步数和 seeds；相对最佳便宜非神经基线的 discrepancy 改善至少 **25%**、正 rigs 至少 **75%**、逐 rig 最大相对退化不超过 **5%**，且校正 matvec 时间不超过直接高保真 matvec 的 **50%**。
2. fresh-A 留新 aperture/f-number；fresh-B 留新 view layout；fresh-C 联合 OOD。rig 不能跨 split。
3. 训练输入从完整 operator rows 降为有限 forward/adjoint calibration probes，并记录 query 数；否则不能声称 query-efficient。V6A 报告中的 `max(candidate error) / max(baseline error)` 只作 aggregate tail 描述，不能替代逐 rig 退化门槛。
4. 加入非神经局部多项式、full-matrix ridge、cone-ray high-fidelity、Learned Operator Correction 和 learned ReSeSOp 对手。
5. operator gate 通过后才跑 PBB/CGLS；同 support、正则选择、停止规则、calls、内存和 wall-clock。
6. 真实数据无 3D truth 时，主报 held-out camera residual、重复性、标定 phantom 和物理积分量。

### 路线 B：TRAIL-4D 的最小启动包

1. 先拿一段连续原始序列和真实 timestamp，不要求完整数据集。
2. 复现 TDBOST 的输入、rank、loss 和推理成本。
3. 构造 transport-only、innovation-only、固定低秩、逐帧 PBB/NeRIF 和普通 FNO 基线。
4. 除全场误差外，专门报告新生、熄灭、拓扑断裂、缺帧和相机异步窗口。

### 现在请师兄回答的六个问题

1. 组内最痛的是有限孔径/景深、几何标定、曲线光路，还是 4D 突变/异步？
2. 现有 NeRIF/TDBOST 能否暴露 `F` 和 `Fᵀ/Jᵀ`，以及 ray、mask、grid、unit？
3. 是否有多档 f-number、焦平面或 paired low/high-fidelity simulation？
4. 能否给 1 个小 calibration phantom 或 flow-off/reference repeat，而不是先整理全库？
5. 若做 4D，能否给 50-200 帧带 timestamp、缺帧与同步信息的最小连续 run？
6. 师兄更愿意先审核 RayKernel-DCO 的有限孔径假设，还是 TRAIL-4D 的事件条件指标？

## 11. 不能写进论文摘要的句子

- 不能把旧的 `+0.035%` 写成 V0 的有效提升；加入 zero baseline 后正确口径是 **-4.083% / 0-of-4**。
- 不能把 V5L 的 post-open ensemble `+6.329%` 写成预注册成功。
- 不能把 V5P 的 `+2.575%` 写成 gate pass，或把 `8F/9A` 写成实际加速。
- 不能把 V5T 的 truth-parameter secant oracle 写成可部署标定。
- 不能把 V5X 的 row-wise oracle 写成模型结果。
- 不能把 V6A 的 `+8.080%` 写成 fresh、真实 BOST、inverse 或 superiority；它是 opened synthetic operator-level development near-signal。
- 不能把 synthetic weak-deflection proxy、truth-calibrated geometry 或完整 operator rows 写成 OERF 实验条件。
- 不能把候选结构自动升级成论文创新；论文价值仍需要真实 mismatch、强邻近基线、fresh rig/session、成本优势和 BOST-specific finding。

## 12. 发布前红队：把“本机能跑”升级成“干净克隆能核”

红队发现 V5P-V5R 依赖三份被 `.gitignore` 隐藏的 synthetic checkpoint；如果只提交 report，别人克隆仓库后无法重放冻结预测。现在只发布 3101/3102/3103 三份约 84 KB 的自生成权重，并把它们写入 V5P report、V5Q/V5R provenance 和顶层 release checksum。它们不含真实实验数据或论文内容。

V5R 也补上了与 V5Q 相同的防火墙：在读取原始 target labels 前，必须重建 V5P 的六组冻结预测并匹配同一 SHA-256。V5Y、V5Z、V6A 则明确记录 MPS 环境；跨设备只要求 validator 用容差核对存档聚合，不声称 bitwise deterministic。

**证据等级：L3 实现/产物审计。** 独立 validator 通过只说明内部一致，不能把 V5P 或 V6A 的 NO-GO 改写为算法成功。复现边界见 [V5P-V6A 发布复现说明](v5p_v6a_release_reproducibility.md)。

## 13. 本日志使用的直接材料

- [V5H-V5R 共享场逆算子研究日志](v5h_v5m_共享场逆算子研究日志_2026-07-16.md)：前序问题与 V5P-V5R 数字。
- [路线 B 研究合同](route_b_dco_trail_research_contract_2026-07-16.md)：算法碰撞、门槛和真实数据合同。
- [V5P report](../demo_t16_operator/results/v5p_fresh_budget_gate/report.json)：首开低预算门禁。
- [V5S report](../demo_t16_operator/results/v5s_dco_low_rank_screening/report.json)：全局低秩筛选。
- [V5T report](../demo_t16_operator/results/v5t_camera_local_tangent_diagnosis/report.json)：局部切线和 secant oracle。
- [V5U report](../demo_t16_operator/results/v5u_calibrated_renderer_residual_screening/report.json)：校准后 renderer residual。
- [V5V report](../demo_t16_operator/results/v5v_camera_local_kernel_correction/report.json)：相机级固定核。
- [V5W report](../demo_t16_operator/results/v5w_clean_aperture_kernel_screening/report.json)：有限孔径因素隔离。
- [V5X report](../demo_t16_operator/results/v5x_ray_conditioned_voxel_kernel/report.json)：逐射线局部核 oracle 与两阶段预测。
- [V5Y report](../demo_t16_operator/results/v5y_direct_ray_conditioned_kernel/report.json)：原始优化失败。
- [V5Z report](../demo_t16_operator/results/v5z_stabilized_direct_ray_kernel/report.json)：稳定线性模型。
- [V6A report](../demo_t16_operator/results/v6a_ray_kernel_hypernetwork_development/report.json)：三种子超网络与停止扩容判决。
- [结构漏斗图](../demo_t16_operator/results/operator_structure_funnel_v5s_v6a.png)：从全局低秩到逐射线超网络的统一可视化。

## 14. V6B：先造一扇真的“只能问 K 次”的门

V6A 用完整 operator rows 训练，所以它不能回答“到了新装置，只给少量标定，能不能适配”。V6B 新增 `BudgetedForwardOracle`：外部只能调用 `measure(x)`，第 `K+1` 次直接报错，也拿不到真值矩阵或真值伴随。输入维数是 64，主预算 `K=32`，因此校准不可能偷偷看完一组完整输入基。

toy 正控制中，真值本来就在 27-gate 家族内，gate 能恢复到数值精度；故意加入家族外残差后，gate error 是 `0.1188`，反而输给同预算最小范数校准的 `0.1080`。这不是坏消息：它证明查询防火墙会暴露模型错配，不会自动把候选包装成成功。

**学到什么。** 查询协议本身也是研究产物。`K forward + 0 truth-adjoint`、第 `K+1` 次拒绝、hidden scoring 前哈希和同预算基线，决定后续数字能不能相信。V6B 的判决只叫 `PASS_PROTOCOL_CONFORMANCE_ONLY`；真正的 fresh 数据还没有构造。

## 15. V6C/V6D：一个补丁怎样失败，又怎样被红队修正

V6C 在 27-gate 后面加 rank 不超过 K 的 residual update。它在手工 misspecified toy 中把 error 降到 `0.0838`，但在本来已经属于 gate 家族的 in-class 层把噪声也当成信号，误差约放大 12 倍。因此 always-on SRCO 明确失败。

V6D 的 post-open 假设是：先估 gate residual 中超过噪声地板的比例，再决定低秩补丁开多少。第一次红队发现 ridge residual 不能一般写成 `n-tr(H)`；第二次又发现 toy 的噪声按 probe 列异方差，不能拿总噪声能量做同方差平均。最终实现直接计算

\[
\operatorname{tr}[(I-H)\Sigma_{diag}(I-H)^T],
\]

并用显式 hat matrix 单元测试、probe-block 顺序测试和两次完整确定性重跑核对。修正后 K=32 的 in-class / misspecified 中位 error 是 `0.00017827 / 0.08767794`，但这些数字仍来自 generator-known covariance 和人工低秩失配。

**学到什么。** 数字没怎么变不代表旧公式没问题；只有公式、噪声生成器和测试描述同一个统计模型，结果才可复核。DF-SRCO 现在只是 `POST-OPEN TOY ONLY`。低秩更新、multisecant、多保真 residual 和 active acquisition 都已有文献；可能保留的新意只能是 BOST 的结构保持 probe、严格查询预算、真实 flow-off covariance 和 inverse/adjoint 闭环。

## 16. PSU 真实数据：终于拿到 5 GB，但还没有“跑出结果”

Penn State 的 9-view 核心 ZIP 已完整下载，size、SHA-256 和 ZIP CRC 都通过。解压后的 `HSOF_9CAM_RT.mat` 是 5.228 GB 的压缩 MATLAB v5 文件，含 MCOS subsystem；SciPy `whosmat` 会在它上面异常，因此新增了流式 v5 header scanner，只读每个变量开头并在 subsystem offset 前停止。

真实审计得到：97 个命名变量；`X/Y/Z` 都是 `400 x 350 x 350`；11 个关键 ray/deflection 字段的宽度统一为 `49,766,400`；26 个作者 loader 所需字段都存在。本轮补完数值门禁与公开汇总防泄漏测试后全量测试为 `381 passed`，发布 validator 仍须单独解释为 `PASS_INTERNAL_CONSISTENCY_ONLY`。

**这还不是什么。** `SCHEMA_CONFORMANT` 只说明“箱子完整、标签和形状对得上”，不说明单位正确、坐标方向正确，更不说明 NIRT 重建成功。下一步依次是数值范围/单位抽查、作者 loader、随机 ray 方向检查、9-view NIRT、冻结后的 held-out reprojection；没有完成这些步骤前，网页不会展示三维性能数字。

新增入口：[V6B 协议](v6b_limited_query_preregistration_2026-07-16.md) · [DF-SRCO 工作稿](df_srco_manuscript_working_draft_2026-07-16.md) · [PSU 外部审计](public_external_bost_benchmark_audit_2026-07-16.md) · [V6D report](../demo_t16_operator/results/v6d_df_gated_srco_postopen/report.json)

## 17. PSU 数值 loader：真实数据过门，官方 NIRT 仍然 NO-GO

上一节只证明 MAT 的变量名和形状存在。这一轮实现了按变量选择的 MATLAB v5 流式 reader：小变量完整读取；大变量完整解压并核验数值 payload 哈希，但只保留几何地标或成组 measurement rows。这样不用把 5.23 GB 文件和 3.92 亿字节网格数组一次塞进内存。

真实读值得到 `siz=[2160,2560,9]`，乘积正好是 `49,766,400`。`X/Y/Z` 的 cell-centered 域分别反推出 0.150/0.130/0.130 m，与官方脚本的 150/130/130 mm 一致。19 个 `v` 样本的单位范数最大误差约 `2.32e-8`；`c` 样本恰好解析出 9 个不同 camera/view centers。13 项 loader 数值契约检查全部通过。

随后对官方 NIRT 做了不执行重数据的 preflight。11/11 Python 文件能通过 AST，但当前 Python 没有 TensorFlow，默认入口是预测并寻找不存在的 checkpoint，代码还强制 `/GPU`、写死 Windows CUDA XLA 路径，并有 6 个静态 blocker。只计算 `cam_data`、`b_data`、`X/Y/Z` 的常驻下界就约 9.25 GiB，未计任何副本、临时量或 TensorFlow/XLA。

**学到什么。**“真实数据可读”和“作者算法可复现”是两个独立门。现在前者从 L3 header audit 升级为 L3 numeric loader conformance；后者仍是 `FULL_AUTHOR_NIRT_NO_GO_CURRENT_ENVIRONMENT`。这次 NO-GO 不是失败拖延，而是明确告诉后续先做 tiny fixture、流式 loader 和 CPU/MPS smoke，不能直接运行默认 `NIRT.py` 再用 OOM 或缺 checkpoint 当研究结果。

完整复核见 [PSU 9-view 数值 loader 门禁](psu_9view_numeric_loader_gate_2026-07-16.md)；网页数字对应的 aggregate-only [机器可读汇总](psu_9view_numeric_loader_summary.json) 不含样本值、作者源码、本机路径或私有目录。仍未解锁 NIRT 重建、held-out reprojection、3D truth、算法胜出或 OERF 声称。

## 18. PSU 九视角几何：真正的问题先出在“积分域”

这一轮没有训练模型，而是把 49,766,400 条真实射线逐条送进作者 box/cone 公式。先确认一处明确的接口问题：MATLAB `find()` 产生 1-based mask，作者 Python/TensorFlow 直接 gather，没有减 1；真实 inactive mask 的最大值恰好等于测量总数，作为 Python 索引会越界。因此本地适配器只做显式 `index - 1`，不改作者源码，也不把 active/inactive 的物理标签当成已确认。

九视角结果不是“都没问题”：0、3、6 号视角出现相同结构。作者只要 cone 长度非零就使用双锥区间，却没有再与外层 reconstruction box 相交；全九视角 cone 总路径中 **184,128.681 m** 位于 box 外，pooled 比例 **9.8976%**。250,597 条射线没有完整 box 段，其中 182,023 条仍被非零 cone 区间救回，最终还有 68,574 条零长度射线。

active 中心线掩膜没有命中当前坏几何标记，inactive/boundary 掩膜只在 0、3、6 号视角命中约 1.10%–1.35%。这说明最直接风险更像边界 loss 与域合同；它还不能证明 active 测量或三维密度已经被破坏。有限孔径采样会偏离中心线，而作者 `oob_mat` 恒为 1，所以“active 中心线安全”也不能升级为“完整光束安全”。

**判决：**`ALL_VIEW_GEOMETRY_AUDIT_NO_GO`。执行完整，科学判决 NO-GO；算法胜出锁定。完整讲解见 [PSU 几何域合同门禁](psu_geometry_domain_contract_gate_2026-07-16.md)，公开图由 JSON/CSV 自动生成，不手填数字。

## 19. A1 裁剪能修机械合同，但它不是最终物理基线

为了只隔离“域外 cone 段”这一件事，A1 保留作者双锥根和 `cone miss -> box` 回退，只把所有区间限制到前向射线并与 box 求交。全九视角中，A1 改变 1,879,113 条射线，移除作者混合域总路径的 **2.4282%**；0、3、6 号视角分别移除 **6.8969% / 7.6520% / 7.1707%**。所有 A1 正长度端点都回到前向 box 内，但 789,416 条射线变成显式零长度，需要 geometry-safe mask 过滤。

最重要的红队结论是：A1 仍继承无界双锥和 miss 回退，因此只能叫 `AUTHOR_COMPATIBILITY_ABLATION_ONLY`。下一步必须另建 B0 前向盒与 B1 `box ∩ 单叶锥` 固定域；B2 再对每个有限孔径样本乘域指示函数，B3 丢弃空域/跨域 ray。只有 held-out camera 重投影也改善，才允许进入逆解和神经算子比较。

**学到什么。** 研究创新不一定先来自更大的网络。一个可发表方向可以从真实数据里发现稳定的 forward-domain failure，再提出严格的 fixed-domain operator、有限孔径采样合同、强基线和 fresh held-out 验证。反过来，如果 B0/B1/B2 只让几何更规整却不改善 held-out，论文就应诚实停在工程诊断，而不是继续调网络把局部数字刷高。

## 20. B0/B1 第一次真实答卷：公式对了，不等于这个域选对了

这一轮我们把作者的混合域放到一边，独立写了两个最朴素的解析几何：

- **B0：**每条射线只在前向 reconstruction box 里积分；
- **B1：**每条射线只在 `box ∩ 单叶 cone` 里积分，cone miss 就是 miss，不再偷偷换成 box。

先用人工几何和 20,000 条随机射线查公式，再对真实九视角全量跑了 49,766,400 条中心线。结果中没有出现端点跑出 box、B1 跑到锥的反向一叶、B1 比 B0 更长，或 B0 miss 而 B1 命中这类自相矛盾。从编程与数学合同看，它们过关了。

但数据随即给了一个更重要的警告：B1 只保留 B0 总路径的 **15.1880%**。这不是说 B1 一定错，而是说“25 度单叶锥是真正物理支持域”是一个非常强的假设，必须请师兄说清 axis、vertex、angle 从哪里来。

0 号视角更具体：1,013,446 条 active 中心线里，有 **1,350 条**完全不命中 B1。它们不是 NaN 或代码崩了，而是作者 cone 函数也认为 miss，然后原程序把它们回退成长约 0.231 m 的 box 积分。B1 不回退后，这批真实有位移信号的 active 测量就会被删掉。

**用人话说：**我们已经造出了两把刻度准确的尺，但还没证明第二把尺量的是正确物理边界。因此当前默认主基线应是更保守的 B0；B1 是待审核的 sampling-hull 消融；B2 再检查有限孔径整束光是否越界。在 held-out camera 还没改善前，这些都不是三维重建成功，更不是算子学习胜出。

## 21. B2/B3：一小块孔径越界，不能草率删掉整条测量

这一轮把每条中心线周围的有限孔径光束也检查了。我们没有沿用每次都不同的随机点，而是分别用 8、16、32 个固定低差异点，让别人可以完全复算。域外点贡献置零，但分母仍是原来的样本数；这样“少了多少光束权重”不会被幸存样本重新放大。

active B1 的总权重保留率从 99.99465%、99.99198% 到 99.96442%。用人话说，即便 32 点检查更细，绝大多数 active 有限孔径积分质量仍在声明域内，损失只有很小一部分。

但是另一个数字变化很大：只要一条 ray 有一个点越界，就给它贴上 any-OOD 标签。这个标签在 8、16、32 点时分别命中 2,660、7,689、99,617 条 active rays。原因不神秘：检查点越多，碰到边界的机会越高，而且三组点不是彼此包含的嵌套设计。

所以我们专门实现了 B3，而不是凭感觉删数据：

- `indicator_keep`：中心线命中就保留，越界小点由 B2 置零；
- `drop_empty`：只有整束孔径都没有域内支持才丢；
- 87.5% / 93.75% floor：预先声明至少保留多少离散支持；
- `drop_any_out`：只要一个点越界就丢整条 ray。

32 点下，87.5% floor 只排除 1,773 条 active B1，93.75% floor 排除 4,405 条，strict `drop_any_out` 却排除 99,617 条。这里最重要的不是宣布 87.5% 胜出，而是发现“整条删除”会把很小的局部支持差异放大成强烈的数据选择。

**学到什么。** B3 不是一个无害的数据清洗开关，而是前向物理模型的一部分。当前最保守的参考应是 B0 + fixed-denominator indicator；B1 和两档 floor 都只作消融。必须用 held-out camera 和 flow-off 噪声判断哪个政策更接近真实光学，不能在同一份 opened 数据上挑最漂亮的阈值。

本轮加入 B2/B3 导出、政策原语与绘图测试后，全量测试为 `563 passed`。这个数字只证明当前代码契约与回归检查通过，不替代物理验证。

公开入口：[B2 摘要](psu_aperture_sensitivity_public_summary.json) · [B3 摘要](psu_b3_policy_public_summary.json) · [四联图](../demo_t16_operator/results/psu_b3_policy_audit/psu_b3_policy_sensitivity_figure.png)

## 22. B1 参数到底有多敏感，以及怎样避免“拿答案出题”

这一轮先把 12 个变体写死，再看真实九视角结果。包括公开 25 度参考、axis 反号、15/20/30/35 度和 vertex 六个方向各 5 mm 的粗移动。这样做的目的不是找最漂亮参数，而是先知道 B1 这把“空间剪刀”有多锋利。

结果最直观的一条是：axis 一反号，公开参考原来命中的 10,627,472 条 active 中心线全部没了。用人话说，axis 正负不是代码里随便统一一下的符号，它决定锥朝哪边开。

angle 也不是温和旋钮。15 度只留下 48.78% active hits，20 度留下 84.84%；30 和 35 度看起来几乎都命中，但它们和 25 度参考的区间重合仍只有 73.14% 和 57.43%。所以“hit 都在”不代表每条光实际积分的空间差不多。

vertex 移 5 mm 没有把系统完全打碎，但 active support IoU 只剩 89.31% 到 93.06%。其中 z 负向移动会丢掉 127,855 条参考 active hits。5 mm 是粗应力测试，不是说真实标定就有 5 mm 误差；它只告诉我们：没有 CAD、标定或 held-out 证据时，vertex 不能默认正确。

**学到什么。** 当前不能从这份 opened 敏感性结果里挑 30 度、35 度或某个 vertex。最保守参考仍是 B0 + fixed-denominator indicator。B1 只能作为冻结消融，参数必须让师兄用物理来源确认，或只用唯一 development rotation 40 决定。

为了防止后面训练时“拿答案出题”，70 个视角也已经提前分好：

- 9 个作者 support views 用于重建；
- rotation 40 的 7 个视角是唯一 development run；
- 18 个同相机未见旋转视角是主审计；
- 另外 12 个未见相机和 24 个联合未见视角只做泛化压力测试。

最终不是把数百万像素当独立样本，而是把 10/20/30/60/70/80 六次旋转当六个实验块。候选要六块全部低于 B0，单侧 exact sign probability 才是 1/64。还必须同时超过 flow-off 重复性地板、守住 p95、环境区不增大并通过标定扰动。

评分器已经先写好并通过 synthetic 测试。它会检查 18 个视角是否完整、是否重复、文件哈希是否改变、mask 是否重叠，以及 front-band 是否真的属于 active。即使全部 image-space gate 通过，它仍不会输出 field-L2 或“唯一三维真值”，因为 PSU 没有独立三维密度 ground truth。

**算力判断。** 这一阶段 Mac 足够，GPU 不会替我们回答 cone 的物理语义。只有 development 给出超过重复性地板的正信号，且 32³ profile 证明需要扩到 64³ 以上多模型多种子时，才租 CUDA。

完整说明：[B1 参数敏感性与 70 视角协议](psu_b1_parameter_sensitivity_and_heldout_protocol_2026-07-16.md) · [参数图](../demo_t16_operator/results/psu_b1_parameter_sensitivity/psu_b1_parameter_sensitivity_figure.png) · [留出协议](psu_heldout_camera_protocol_public_summary.json)

## 23. B0 重建接口：投影拟合很好，三维场仍可能差很多

这一轮终于从“几何审计”迈到了“可逆解接口”，但仍然没有碰 rotation 40 或 final audit。新接口把标量扰动场依次送进三维有限差分、三线性插值、真实 `Ru/Rv` 投影和 `L·Csys/M` 缩放；域外有限孔径样本置零，但固定分母不变。伴随不是另训网络，而是把这条链逐项转置。

先在 12³ 合成场做闭环。`A/Aᵀ` 内积误差是 **6.78e-15**；固定 60 次 Landweber 后，measurement relative L2 从 1.0 降到 **0.005028**。但 field relative L2 仍有 **0.4504**。

**用人话说：**相机上几乎重投影对了，不代表三维里面就恢复对了。BOS 只看梯度，本身有常数零空间，少视角还会留下更大的不可辨识子空间。这个反例以后必须放在论文结果里，防止把漂亮的 held-out 图误写成“真实三维场已恢复”。

然后用真实九视角几何做接口审计：每个 support view 取 256 条不依赖位移大小的 active 分位射线，共 2,304 条、36,864 个 QMC-16 样本。16³/32³ 的 CPU float64 dot defect 为 **4.97e-16 / 1.78e-16**，MPS float32 为 **7.28e-8 / 9.89e-8**，全部过冻结阈值。

这个子集单次 forward/adjoint 只有毫秒级，但不能线性外推到全量一千万级 active rays。下一门是流式遍历全部 support rays 的 16³ Landweber/PBB/CGLS，而不是立即上 128³ 网络。

官方 rotation 30–90 archive 也已经完整下载，大小 **4,095,655,393 bytes**，SHA-256 与 ZIP CRC 已本地记录。rotation 40 仍只允许在真实 support inverse 和停止接口冻结后打开；final rotations 继续封存。

**算力判决。** 当前 Mac 继续做 16³/32³ baseline，不租 GPU。只有全量 support profile、development repeatability 和候选结构都给出必要性后，才把 64³/128³ 多模型多种子迁移到 CUDA。

完整入口：[B0 重建接口门禁](psu_b0_reconstruction_interface_gate_2026-07-16.md) · [公开摘要](psu_b0_reconstruction_interface_public_summary.json) · [四联图](../demo_t16_operator/results/psu_b0_interface_audit/psu_b0_interface_audit_figure.png)

## 24. 一千万条真实射线终于跑进逆解：32³ 明显优于 16³

这轮把九个 support views 的 `10,628,822` 条 active rays 全部接进了流式 `A/Aᵀ`。每次完整调用包含 329 个内部 chunks、每条 16 个有限孔径样本，总共约 1.70 亿 sample points。chunk 只是内存实现细节，一次完整遍历才记一个 operator call。

先出现了一个很重要的负结果：小子集 float32 dot-test 通过，但全量 float32 用真实 observation 做 dual 时 defect 变成 `8.49e-4`；换确定性随机 dual 仍为 `2.04e-5`，刚好高于冻结的 `2e-5`。没有把门槛放宽，而是改用 float64。全量 float64 defect 为 `3.46e-15`，完整 `F+Aᵀ` 约 53.4 秒，RSS 约 5.34 GiB。

**用人话说：**一千万条射线叠加时，很多很小的 float32 舍入会一起出现。小测试过了不代表大任务也过。好消息是，这台 Mac 跑 float64 只慢一点点，所以当前没必要租服务器。

16³ 固定 4 步 CGLS 用 `4F+5Aᵀ`，把 support relative L2 从 1 降到 `0.78771`。直接重新 forward 和递推 residual 只差 `1.74e-16`，所以数值账本可信，但拟合还不够。

随后在看结果前写死 32³ 仍然只跑 4 步，且只有 residual 绝对下降至少 0.02 才算分辨率信号。32³ 最终是 `0.62713`，比 16³ 绝对下降 `0.16058`、相对改善 `20.39%`，九个视角全部改善。pair 时间 50.5 秒、RSS 仍约 5.35 GiB。

**学到什么。**

1. 32³ 应取代 16³ 成为后续低分辨率 reference；
2. 当前成本主要在一亿七千万有限孔径采样，不在 3D voxel array；
3. 值得优化的是 stencil cache、ray batching 和伴随安全混合精度；
4. residual 仍有 0.627，不能靠“分辨率提高有效”就宣布模型正确；
5. rotation 40 必须检验 32³ 的改善是否迁移，而不是继续在 support 上加迭代挑最小 residual；
6. learned model 最合理的角色是 preconditioner/correction，并始终经过真实 `A/Aᵀ` 数据一致性。

完整入口：[全 support CGLS 与分辨率门禁](psu_b0_full_support_cgls_and_resolution_gate_2026-07-16.md) · [对照 JSON](psu_b0_streaming_resolution_public_summary.json) · [分辨率图](../demo_t16_operator/results/psu_b0_streaming_resolution/psu_b0_streaming_resolution_figure.png)

## 25. 为什么不再“多开几个反演”，而是先把每次反演变快

这一轮先查了电脑到底慢在哪里。结果不是网速：数据已经在本地，下载吞吐也有约 310 Mbps。真正的问题是每次 forward 或 adjoint 都重新算一遍 1.70 亿个有限孔径样本的位置，再重新生成三线性插值八角点。它们加起来占单个 chunk 约 82% 时间。

所以没有盲目同时开很多完整反演。那样只会让几个任务抢同一颗 CPU、同一块内存和 SSD。我们改成：

1. 固定几何只算一次；
2. 把 lower corner、局部分数、mask、投影和 scale 存到私有 cache；
3. 反演串行读 cache；
4. 测试、网页、文档和绘图在旁边并行。

完整 cache 是 5.017 GB，14.94 秒建完。严格对照里，缓存前后 forward 和 adjoint 的相对差都是 0，说明没有为了快偷偷换算子。

同一会话下，完整 `F+Aᵀ` 从 37.92 秒降到 17.04 秒，是 2.23 倍加速。更重要的是，把原来 32³、固定 4 步 CGLS 完整重跑后：

- residual 一模一样，都是 `0.6271324683999563`；
- 重建体相对差只有 `1.17e-16`；
- 优化时间从 218.03 秒降到 74.95 秒，是 2.91 倍。

**用人话说：**我们没有让答案变“更好看”，而是让完全同一个答案更快得到。这是后面做新算法的地基。现在可以在本机认真比较 Tikhonov、TV、不同 Krylov 预条件器，或让小网络只负责提出搜索方向；每一步仍由真实 `A/Aᵀ` 检查，不需要一上来就租服务器。

但 cache 本身不是论文创新。真正可能写进方法论文的，是在这个快速、精确的物理层上解决：有限孔径失配、少视角零空间、薄反应前沿、几何不确定，以及 learned preconditioner 能否在相同 calls 下稳定胜过 CGLS/PBB。

完整入口：[紧凑缓存与快速参考门禁](psu_b0_compact_cache_and_fast_reference_gate_2026-07-16.md) · [缓存 benchmark](psu_b0_compact_cache_public_summary.json) · [CGLS 对照](psu_b0_cached_reference_public_summary.json)

## 26. 第一个真实几何上的 learned preconditioner：普通情况有信号，联合越界必须否掉

这一轮第一次把“小网络只提搜索方向”的想法接到真实 PSU 九视角 support 几何上。输入三维场仍是解析 plume / flame-front 代理，所以不是实验三维真值；但每条观测射线、相机布局、有限孔径和 `A/Aᵀ` 都来自前面冻结的 B0 接口。

先补了一个容易被忽略的强对手。BOS 观测的是折射率或密度扰动的空间梯度，普通四步 CGLS 和 identity steepest descent 会严重压低标量场低频。验证集在预先写死的 `p=0,...,6` 中选择 inverse-Sobolev 谱方向，`p=5` 的 combined loss 是 `0.44419`，远好于 `p=0` 的 `1.21360`。所以 learned model 不能只打弱 CGLS，必须从 `p=5` 精确零初始化后再证明增量。

候选只有 2,227 个参数。它读取逐视角白化 residual、噪声尺度、相机 mask 和迭代阶段，输出一个有界、严格为正的 Fourier multiplier；每一步仍先算精确 `AᵀWr`，再做解析线搜索。网络不能直接生成三维场，Sobolev 和 learned 都严格使用 `4F+4Aᵀ`。

三种子在 IID 上相对 Sobolev 提升 `+4.36% / +4.62% / +4.26%`，噪声单独越界仍约 `+4.28%` 到 `+4.46%`，4–5 视角单独越界也有 `+1.41%` 到 `+1.77%`。这说明模型学到了重复的分布内各向异性，不是某个幸运种子。

但联合 OOD 同时换成 thin/double front、4–5 views、8%–12.5% 噪声和 QMC-32→QMC-8 算子失配后，三种子均值变成 `-0.432% / -0.368% / -0.199%`。p10 约 `-4.5%`，每个种子的 `>1% harm` 都是 `33.3%`；candidate measurement residual 也约 `0.404–0.410`，差于 Sobolev 的 `0.355`。预注册要求至少两个种子联合 OOD 不退化，实际为 `0/3`。

**用人话说：**网络在熟悉范围内会把 Sobolev 方向修得更合适，但当形态、噪声和相机数量一起变化时，它不知道自己已经离开训练范围，仍然自信地修正。每一步数据项下降，只能证明沿自己的方向在下降，不能证明这条轨迹比 Sobolev 更好，更不能证明三维场更真。

**正式判决：**`SPECTRAL_PRECONDITIONER_PILOT_CANDIDATE_NO_GO_OR_INCOMPLETE`。这是带真实几何的 L1/L3 合成开发证据，不是 FNO/DeepONet superiority，也没有打开 rotation 40 或 final audit。

下一代只允许做 **Support-Enveloped Spectral Correction**：

\[
P_{\theta,\tau}
=P_{\mathrm{Sobolev}}
+\tau(z)\left(P_\theta-P_{\mathrm{Sobolev}}\right),
\qquad 0\le\tau\le1.
\]

它必须在超出声明支持域时精确退回 Sobolev，并在训练内加入 camera dropout、相关噪声和尖锐前沿压力；loss 还要惩罚相对 Sobolev 的 residual 风险。当前六个 audit split 已经打开，从现在起只能算 development；下一次判决必须使用新形态、新噪声和新种子。

完整入口：[首轮 NO-GO 说明](psu_b0_spectral_preconditioner_no_go_2026-07-16.md) · [严格公开摘要](psu_b0_spectral_preconditioner_pilot_public_summary.json) · [四联图](../demo_t16_operator/results/psu_b0_spectral_preconditioner_pilot/psu_b0_spectral_preconditioner_pilot_figure.png)

## 27. 视角回退确实能止损，但也证明“只看视角数”不够

首轮 joint OOD 的一个明显特征是 active views 从训练的 6–9 个掉到 4–5 个。为了不在 opened 数据上重新训练和扫阈值，这轮只包了一层固定规则：

```text
6–9 views: 使用原 learned spectral direction
其他情况: 逐值使用 p=5 Sobolev direction
```

实现上没有用 `fallback + τ(candidate-fallback)`，而是用布尔选择。原因是 MPS float32 即使 `τ=0/1` 也可能留下约 `1e-7` 舍入，进而让 top-10% front threshold 的一个边界体素换组。连续指标冻结容差 `1e-6`，离散 front F1 容差 `5e-4`；方向本身另有逐值单元测试。

结果非常干净：

- view OOD 和 joint OOD 的 learned coverage 都变成 0；
- joint OOD 三种子 `>1% harm` 从 33.3% 变成 0，均值约等于 Sobolev 的 0% gain；
- IID、noise OOD 与 exact control 保留原来的约 4% 信号；
- family OOD 仍处于 6–8 views，所以规则完全不触发，`harm` 仍为 20.8%–25%，p10 仍为负。

**用人话说：**我们找到了一种可靠的“这时别用网络”信号，但没有找到“网络在新形态上也可靠”的证据。joint OOD 变安全，是因为模型完全没出手，不是它突然学会了联合泛化。

所以这一轮只能叫 `POSTOPEN_SUPPORT_ENVELOPE_DIAGNOSIS_COMPLETE_NOT_FRESH`。它通过实现门，不通过方法门。下一代 `τ(z)` 必须除了 view-count margin，还读取白化 residual 的均值、最大值、跨相机离散度、相对 Sobolev 的 residual-risk proxy 和 correction magnitude；并在训练内加入 camera dropout、相关噪声和 thin-front stress。

下一次 fresh gate 还要防一个“虚假安全”策略：不能靠把 coverage 全降到 0 获得 harm=0。必须同时报告 coverage、accepted gain、p10、harm、wall time 和相同 `F/Aᵀ` calls。

完整入口：[视角支持域回退诊断](psu_b0_support_envelope_postopen_diagnosis_2026-07-16.md) · [严格公开摘要](psu_b0_support_envelope_postopen_public_summary.json) · [四联图](../demo_t16_operator/results/psu_b0_support_envelope_postopen/psu_b0_support_envelope_postopen_figure.png)

## 28. OCRRG 曾被记录为 synthetic gate pass，后续合同审计撤回该解释

这一轮没有继续扩大谱网络。我们冻结了一个更小的问题：只看部署时能拿到的
residual、精确伴随梯度、视角 mask 和候选方向，能不能判断“这次该不该让
学习器接管”。

方法暂称 OCRRG。它用 16 个无真值特征预测 learned preconditioner 相对
inverse-Sobolev 的 field-gain，再减去 split-conformal 的保守误差分位数。
只有预测下界、特征距离和 6 至 9 视角硬支持同时通过，才运行 learned
四步求解；否则整条路径精确回到 Sobolev。判断本身不需要把两种重建都跑完，
所以仍是相同的 `4F+4Aᵀ`。

当时的 held-out synthetic 协议先在提交 `cd5d4a0` 中冻结，再打开七组各 24 个
新场。按后来发现不一致的 feature-order contract，三种模型种子曾被记录为通过
候选门：

- support IID：coverage 36.1%，平均 gain +1.38%，harm 2.78%；
- 未见形态：coverage 26.4%，平均 gain +1.04%，harm 0；
- 强相关噪声：coverage 43.1%，平均 gain +1.31%，harm 2.78%；
- 未见形态 + 强噪声：coverage 27.8%，平均 gain +1.41%，harm 0；
- 3 至 5 视角两组：coverage 0，逐值回退 Sobolev。

**用人话说：**这张历史表格描述了风险门把“见什么都出手”改成“有把握才
出手”，并在该批 synthetic rows 上压低坏尾部。但后续发现 calibration 与 deployment
使用了不同的 feature order，所以这些数字只能保留为 post-open 描述，不能再写成
gate pass、conformal 保证或“真实进步”。

独立 validator 找到 4 条被接受后仍恶化超过 1% 的记录，只来自两个源样本：
一个 6-view plume 在两个种子上退化约 2.6%，一个强相关噪声的 6-view
oblique shock 在两个种子上退化 4.5% 至 5.7%。这说明 pooled risk model
对最低支持视角数和特定物理形态仍不够保守。

下一步不是在该批已打开数据上扫阈值，而是先统一 canonical feature function，再
换全新 seeds 做独立重复，并把风险
校准改成按 view count、形态族和噪声强度分组。真实迁移前还必须用师兄提供
的 flow-off repeats 替换合成 covariance。没有这一步，不能宣称逐样本安全、
任意 OOD conformal 保证或优于 FNO/DeepONet。

完整入口：[历史判决与后续修正](psu_b0_residual_risk_fresh_result_2026-07-16.md) · [公开 JSON](psu_b0_residual_risk_fresh_public_summary.json) · [诊断图](../demo_t16_operator/results/psu_b0_residual_risk_fresh/psu_b0_residual_risk_fresh_figure.png)

## 29. 我把“3/3 过门”重新拆开，发现 conformal 契约其实没闭合

这次最重要的工作不是再训练一个网络，而是把旧 fresh 的第一步特征逐值
重算。504 条部署特征与冻结报告的 prediction 最大只差 `8.24e-5` 个百分点，
所以复现链是闭合的。

但代码里藏着一个顺序差异：

- 训练和 calibration 先把方向乘 support mask，再计算方向范数、修正量等特征；
- 真正 deployment 先计算这些特征，solver 后面才乘 support mask。

把两种顺序放在同一批 504 rows 上比较，有 7 条 accept/fallback 决策改变，
prediction 最多移动 0.826 个百分点。当前 4 条 harmful rows 恰好没有因此
改变，所以原 fresh 的经验表格还是真实的；但 conformal calibration 要求
校准和测试使用同一个 score function，这个前提没有满足。

**用人话说：**原来那次“3/3”不是造假的，但它只能叫“预设经验指标通过”，
不能继续叫“已经校准好的 conformal 风险门”。这正是严肃研究里必须主动
降级结论的地方。

同一轮还验证了两件事：

1. harmful rows 全在 6-view，但 6-view 的 80 条 accepted rows 中大部分有
   正收益，不能简单全部关掉；
2. exact-view quantile 仍抓不到 4 条 harmful rows，而且 8/9-view calibration
   只有 1/0 个字段，说明按视角硬分桶也不够。

下一版会先写唯一的 canonical feature function 并做逐值测试，再加两个
物理可解释否决器：

- 低频梯度 + 大 correction / gain span 的 spectral stress；
- 高 component correlation + 低 residual/gradient scale 的 camera stress。

这些组合是 opened fresh 启发的，所以旧 fresh 永久只作诊断。只有平衡
6/7/8/9-view development、leave-one-family-out 选择、阈值冻结和全新 seeds
independent repeat 都完成后，才能重新谈一个 conformal 或 risk-controlled
候选。

完整入口：[特征契约诊断](psu_b0_residual_risk_postopen_diagnosis_2026-07-16.md) · [公开 JSON](psu_b0_residual_risk_postopen_diagnosis_public_summary.json) · [四联图](../demo_t16_operator/results/psu_b0_residual_risk_postopen_diagnosis/psu_b0_residual_risk_postopen_diagnosis_figure.png)

## 30. 双否决器第一次筛选：相机支路有信号，plume 支路还不行

我没有用 fresh 去选阈值。先在旧 development 上把 spectral threshold、
camera threshold 和 6-view extra margin 组成 294 个有限候选，再按 validation
coverage、overall harm 和各视角 harm 选一个。

development 最终选择了比较宽松的两个 stress threshold，而且 6-view extra
margin 选成 0%。这本身已经说明：旧 development 没有提供足够证据，支持
“所有 6-view 应更保守”。

把这个 development-selected 版本放到已经打开的 fresh 上只作诊断：

- 两条 correlated-noise shock harm 被 camera stress veto 拒掉；
- 两条 low-frequency plume harm 仍然被接受；
- support IID coverage 从 36.1% 降到 27.8%，但 harm 没变；
- strong correlated noise coverage 从 45.8% 降到 27.8%，harm 归零。

这不是“成功一半”的宣传结果。它告诉我下一批 development 必须主动加入
低频、宽尺度、弱梯度 plume stress；否则 spectral veto 只是根据 opened
反例写出的漂亮公式，没有可重复证据。

还有一个很容易踩的坑：看过 plume 的 lower bound 后，事后给 6-view margin
加约 0.6% 可能刚好把它们挡住。但这就是 fresh leakage，所以我没有这么做。
下一次阈值必须从新 development 自己长出来。

完整入口：[Multi-Veto 开发筛选](psu_b0_multiveto_development_screen_2026-07-16.md) · [公开 JSON](psu_b0_multiveto_development_screen_public_summary.json)

## 31. 强基线把当前学习方向推翻了：这是一次有价值的 no-go

这一轮先补了 L2/H1 Tikhonov 和普通 CGLS。它们在四步预算内虽然把投影残差降得更快，但三维场误差反而比固定 Sobolev 差约 35%–56%。这说明 BOST 少视角问题真正需要的是频谱先验，不能拿裸 CGLS 当“强基线”。

随后实现了 Sobolev 预条件 CGLS（PCGLS）、各向异性 Sobolev 和分阶段 Sobolev。最重要的结果是：

- PCGLS-4 只在 `risk_validation` 选择 `strength=4, epsilon=0.05`；
- 固定四步重建最后不需要计算未使用的 \(A^\top r_4\)，所以真实预算是 `4F + 4AT`；
- 它在 `risk_validation` 比三种子 learned 均值降低约 5.00% 场误差，在未用于选参的 `risk_calibration` 降低约 4.94%；
- 七个已经打开的 stress split 都有正的平均改善，逐场至少赢 20/24；
- 168 个打开诊断场 pooled field relative L2：PCGLS-4 为 0.6246，learned 为 0.6711。

讲人话：旧模型学到的方向比“固定平滑梯度”好，但没有比经典共轭梯度会利用历史搜索方向。这个差距不是靠再调风险门能救回来的，因此当前 learned steepest direction 正式判为 no-go。

下一版不能再从“设计一个更好的单步方向”出发，而应从 PCGLS 出发：

1. 首选：先根据相机几何、视角和噪声生成一个正定频谱预条件器，然后在四步 PCGLS 中固定使用；
2. 低风险：只学习 PCGLS 在第几步停止或何时回退；
3. 高风险：若预条件器随残差变化，改用 flexible CG，并显式处理方向正交化。

完整审计见 `docs/psu_b0_pcgls_no_go_2026-07-16.md`。

## 32. 第一个 SPD-PCGLS 小网络也没有过线：先查上限，不扩宽度

我把上一节提出的最小模型真正写出来并训练了。它有 2,527 个参数，读取相机
几何、视角 mask、噪声和初始 residual 的摘要，只输出 7 个低维频谱系数。
输出始终为正、有界并做几何均值归一化，而且在四步 PCGLS 内完全固定。零
初始化时，它逐值等于强基线 Sobolev-PCGLS-4。

三种随机种子在 Apple M5 上总共训练约一分钟，程序和优化都正常，但科学
结果是 `0/3 NO-GO`：

- validation 的平均场误差改善只有 +0.016% 至 +0.054%；
- calibration 是 -0.165% 至 +0.056%；
- 所有 bootstrap 下界都小于 0，远低于预先写死的 2% 门槛；
- seed 42/43 各有一个 calibration 场恶化超过 1%。

最值得记住的现象是：网络把 measurement residual 平均改善了约
0.60%–1.71%，三维 field gain 却几乎为零，甚至变负。

**讲人话：**从不同角度拍到的二维偏折图可以被拟合得更漂亮，但少视角
BOST 的三维空缺信息并不会凭空回来。当前小网络更像在调整“怎样贴合已经
看到的投影”，没有找到“怎样判断看不到的三维部分”。

这次不能靠把 MLP 从 24 hidden 改成 128 hidden 续命。下一步先做一个不训练
网络的 conditional-headroom audit：让有限个静态 PCGLS 候选分别接受
全局选择、按视角数选择、按视角数+噪声选择、按形态 oracle 选择和逐样本
truth oracle 选择。

- 如果逐样本 oracle 也没有明显空间，说明这个频谱家族已经接近上限，应转
  TV、学习停止或真实数据；
- 如果 oracle 很大、按可观测条件选择却很小，说明缺的是能识别 null-space
  风险的输入和映射，不是网络宽度；
- 如果按视角/噪声就能稳定改善，才值得重新训练一个更小、更可解释的
  selector。

完整判决见
`docs/psu_b0_conditioned_pcgls_development_no_go_2026-07-16.md`。

## 33. 105 个固定 PCGLS 候选告诉我：上限存在，但“按视角数选”没用

这一步没有训练网络。我先把五档 Sobolev strength、三档 epsilon 和七种
轴向频谱模式组成 105 个固定 SPD 候选，所有候选都用同一个四步 PCGLS，
预算严格保持 `4F+4AT`。

结果把问题切得很清楚：

- 训练集只选一个全局候选，validation +0.35%，calibration -0.22%；
- 按 active view count 选，validation +0.76%，calibration -0.26%；
- 按 view count + noise 选，validation -0.11%，calibration -5.65%；
- 用不可部署的真实形态标签选，validation +2.69%，calibration +2.38%；
- 每个样本直接看三维真值再选，validation +6.52%，calibration +7.22%，
  且没有负尾。

**讲人话：**同一套频谱工具箱里确实有更合适的扳手，但“拍了几台相机、
噪声多大”不足以告诉我们该拿哪一把。真正决定频谱选择的是场的形态，
而形态不能在部署时从标签读取。

这也排除了一个很诱人的错误方向：继续把 geometry/noise MLP 加宽。逐样本
oracle 很大、简单可观测分层很小，说明短板是“怎样从测量中识别三维形态与
null-space 风险”，不是频谱 basis 数量不够。

完整入口：

- [conditional headroom 判决](psu_b0_pcgls_conditional_headroom_2026-07-16.md)
- [公开摘要](psu_b0_pcgls_conditional_headroom_public_summary.json)
- [四联图](../demo_t16_operator/results/psu_b0_pcgls_conditional_headroom/psu_b0_pcgls_conditional_headroom_figure.png)

## 34. 首伴随场里确实藏着形态信息，但不能把合成标签带到部署

PCGLS 本来就要先计算

\[
g_0=A^\top W y.
\]

所以我从这个共享首伴随场提取了 44 个不增加 `A/A^T` 调用的特征，包括
低/中/高频能量、轴向频谱不平衡、空间矩、稀疏度和梯度统计。然后做了三层
审计：

1. 元数据特征：只看视角、噪声和几何摘要；
2. 首伴随场特征：只看部署可获得的 `g0`；
3. 形态标签：只作不可部署上限。

首伴随场的 hard selector 在 validation / calibration 分别给出约
+2.22% / +1.72%，说明它真的读到了一部分形态；元数据路线没有这个信号。
但严格要求 train OOF accepted harm 不超过 5% 时，没有任何候选可冻结。

**讲人话：**二维测量反投影回来之后，确实会留下“这个场更像细前沿还是宽
羽流”的痕迹；问题是我们现在只会预测一个最可能的类别，还不会判断
“这次判断错了会不会把三维重建毁掉”。

因此第一版 OMSE 用四个固定专家做形态分类，只能作为中间实验。修正基线
回退语义后，它在 validation +2.03%，calibration +1.29%，而 calibration
仍有 6.67% 样本恶化超过 1%。它比直接 MLP 有信息，但不是可用算法。

## 35. OGSE 把分类改成收益回归：负尾清零，但总门仍是 NO-GO

第二版不再让 selector 猜合成形态标签，而是用 train-only 真值监督每个
固定专家相对 static PCGLS-4 的逐样本收益。专家库也不手挑，而是在
`risk_train` 上贪心覆盖：

- 4 专家 oracle headroom +4.16%；
- 6 专家 +4.82%；
- 8 专家 +5.09%。

这版叫 OGSE-PCGLS。它从首伴随场预测每个专家的收益分数，然后在 log-space
里生成一个固定正定 multiplier。

审计过程中还抓到一个关键实现错误：旧混合器只检查 top-1 / top-2 margin，
即使 top-1 就是基线专家也会发生 softmax 混合。修正为“只有非基线专家
top-1 且 margin 过阈值才介入”后，严格路线的灾难负尾消失：

- validation +2.423%，95% CI [+1.237%, +3.676%]；
- calibration +1.651%，95% CI [+0.700%, +2.902%]；
- 两个 split 的 `>1% harm` 都为 0；
- calibration 没达到预注册 +2%，所以总门仍是 NO-GO。

放宽风险路线能达到 +3.56% / +2.55%，但最坏样本分别退化 -12.67% /
-7.37%。这说明收益潜力不是幻觉，真正缺口是**风险条件分布**。

下一版不先上大网络，而是让 selector 同时估计：

```text
mean gain
lower quantile of gain
P(gain < -1%)
```

并增加按视角分组的 residual spectrum、`A g0` 角向不平衡、第一步残差下降率
与方向夹角等物理可观测量。只有预测下分位数为正时才沿
`baseline -> single expert` 做有限幅度介入。

完整入口：

- [OGSE V2 严格判决](psu_b0_ogse_pcgls_development_no_go_2026-07-16.md)
- [公开摘要](psu_b0_ogse_pcgls_development_public_summary.json)
- [论文四联图](../demo_t16_operator/results/psu_b0_ogse_pcgls_development/psu_b0_ogse_pcgls_development_figure.png)

## 36. RQ-OGSE 第一次过了 field 主门，但我主动没有把它叫成功

我把 OGSE 的“把所有专家 softmax 混起来”改成了一件更容易解释的事：

```text
不确定 -> 原样用 static PCGLS-4
确定 -> 只沿 baseline 到一个专家走固定距离
```

为了不让几百组阈值反复跑三维重建，我先算好 13 个有限动作。后面的
648 个 selector 只查这些动作在每个训练样本上的真实结果。这个改动把整轮
RQ 实验压到约 12 秒，而且没有少算任何最终候选的 `F/AT`。

最亮眼的一条 mean-only 路线是：

- validation field gain `+3.321%`；
- calibration field gain `+2.907%`；
- 两层 bootstrap 下界都大于 0；
- validation 没有 `>1%` field harm；
- calibration 只有 1/30 个 `>1%` field harm。

按最初 field-L2 的八项门，它真的全过了。但我继续看 front-F1 后发现：

- calibration front 均值 `-0.261%`；
- correlated-noise oblique shock 最坏下降 `-30.876%`；
- 另一个 validation shock 最坏下降 `-27.404%`。

**讲人话：**三维体素整体平均更接近真值，不代表火焰边界或激波面也更准。
模型可能把大面积平滑区域修好了，却把最重要的尖锐结构磨坏。

所以这次不能写成“8/8 GO”。准确说法是：

> field utility signal 通过；reacting-front safety 没有通过；总判 HOLD。

## 37. 分位数和 front-risk 头为什么没有白做

我又分别比较了 mean、quantile、quantile+harm 和
mean+quantile+harm 四种路由。

联合风险头把 validation/calibration 的 field harm 都清零了，但 field
平均收益只剩 `+1.979% / +1.777%`，没有达到双 2% 门槛。

然后我给 front-F1 绝对下降也增加 lower-quantile 和 harm-probability 头。
严格多目标路线：

- validation field `+1.192%`，front mean `+0.375%`；
- calibration field `+1.382%`，front mean `-0.060%`；
- 两层 field harm 都为 0。

它更谨慎，却仍不能同时保住收益和 front。这说明问题不只是阈值：

```text
当前 44 个特征 = 所有相机反投影求和后的全局摘要
```

求和以后，看不到“哪一台相机和其他相机打架”，也看不到相关噪声只污染了
哪些视角。下一版要把每个相机的 residual 和 adjoint contribution 分开，再用
对相机顺序不敏感的 set encoder 聚合。

还有一个我修正了的时序错误：first-step residual contraction 只有跑完第一步
才知道，不能拿它来决定第一步之前的固定 preconditioner。要用它，就必须
baseline 先走一步后 restart/FCG，或者增加 probe calls 并如实记账。

完整入口：

- [RQ-OGSE HOLD 判决](psu_b0_rq_ogse_pcgls_development_hold_2026-07-17.md)
- [论文工作草稿](rq_ogse_manuscript_working_draft_2026-07-17.md)
- [RQ 公开摘要](psu_b0_rq_ogse_pcgls_development_public_summary.json)
- [多目标公开摘要](psu_b0_mo_rq_ogse_pcgls_development_public_summary.json)
- [RQ/front 四联图](../demo_t16_operator/results/psu_b0_rq_ogse_pcgls_development/psu_b0_rq_ogse_pcgls_development_figure.png)

## 38. VD0-A：逐视角伴随分解做对了，但 18 个冲突统计还不够

RQ 的下一假设是 pooled `g0` 把相机之间的冲突抹掉了。于是我先实现
`adjoint_by_view`：每条射线只生成一次散射贡献，再按相机槽累加；九个相机
输出求和必须回到原 pooled 伴随。

接口门通过：

- grouped sum 最大相对误差 `1.78e-7`；
- 射线 scatter 只遍历一次；
- 定向实现和特征测试通过；
- 但保留九份体场会增加内存与逐视角有限差分伴随，不能把“一次调用”写成
  “与 pooled 完全同 FLOP”。

然后我从每个相机的伴随场提取 18 个显式统计：范数份额、熵、求和抵消、
两两 cosine、负相关比例和每个视角与 pooled 方向的一致性。没有训练大网络。

整轮复用了旧 RQ 的 16 个训练动作缓存，没有重新跑 train reconstruction，
只用了 `5.97 s / 436 MB`。结果是：

- pooled 严格路线仍复现 `+3.321% / +2.907%`；
- view-conflict 单独没有严格路线，放宽后虽约 `+1.1% / +1.1%`，但 field
  harm 为 `25.0% / 13.3%`，front 均值为 `-1.41% / -3.40%`；
- pooled + view strict 为 `+2.258% / +1.604%`，calibration harm
  `6.67%`，front mean `-1.376%`。

留一形态时，拼接路线把 pooled 的 `-0.999%` 提到 `+0.199%`；但留一噪声
仍是 `-0.347%`。这说明逐视角信息不是完全没用，却没有形成可迁移的安全
表示。

**讲人话：**我现在能把每台相机“各自怎么把二维位移推回三维场”拆出来，
但只比较这些三维方向互相像不像，还看不见二维图里的尖锐 front，也不知道
每台相机从什么方向观察。所以不能因为接口完成了就马上上 DeepSets。

下一步只允许补 VD0-B：

1. 每台相机白化位移图的高频、ridge/梯度集中度和方向各向异性；
2. 与该视角成对的 camera pose / projection basis；
3. 再做 leave-one-family 和 leave-one-noise。

如果这一步仍不能同时减少 field/front harm，就停止 set encoder；不靠加深
网络把 post-open 数据调到好看。

完整判决：

- [VD0-A 判决](psu_b0_view_decomposed_probe_no_go_2026-07-17.md)
- [公开摘要](psu_b0_view_decomposed_probe_public_summary.json)
- [四联图](../demo_t16_operator/results/psu_b0_view_decomposed_probe/psu_b0_view_decomposed_probe_figure.png)

## 39. VD0-B 恢复真实 detector 邻域，但仍不能安全路由

VD0-A 之后，下一步被严格限制为“二维 front proxy + camera pose”，不能直接训练 DeepSets。实现时先发现一个容易制造假结果的问题：每台相机的 256 条射线是从一百多万 active pixels 按分位点抽出的，抽样顺序不能排成 `16 x 16` 当作真实图像。于是本轮先从 MATLAB 列主序线性索引恢复 detector row/column，在真实不规则 pixel 坐标上建 8 邻域图，再用局部加权最小二乘估计位移 Jacobian。

新增 30 个 observable descriptors：邻域对比、Jacobian、front top-10% 能量集中、structure anisotropy、divergence/curl balance，以及把二维主方向经 `Ruvecs/Rvvecs` 投到世界坐标后的无符号方向一致性。它们不使用三维 truth、形态标签、重建场、迭代后 residual 或 PSU 实测 deflection。

结果有一条真实但不足以继续扩容的信号：

- pooled 的 leave-one-family / leave-one-noise 为 **-0.999% / -2.217%**；
- detector-only 为 **+1.051% / +1.118%**；
- pooled+detector 为 **+0.907% / +0.947%**。

但真正的 validation/calibration 转移失败：

- pooled+detector field gain **+2.805% / +2.901%**，没有双 split 击败 pooled 的 **+3.321% / +2.907%**；
- field harm **12.5% / 10.0%**，超过 5%；
- front mean **-0.077% / -0.778%**；
- front p10 **-7.873% / -7.242%**。

因此判决是 **`VD0B_DETECTOR_POSE_NOT_TRANSFER_SUPPORTED_STOP_SET_ENCODER`**。这不是说 detector geometry 没用，而是说当前特征能看到开发压力，却不能决定哪个 PCGLS 专家对 front 安全。尤其 legacy `camera_correlated` noise 仍是在伪方格上生成的，不能把 leave-one-noise 的正数当作 measured camera covariance 证据。

下一步不再训练网络，先做两件更接近实验的问题：

1. 用 PSU 公开 `epsu/epsv` 做真实-vs-synthetic feature distribution audit，只检查工作域，不训练；
2. 向师兄要 flow-off/reference repeats，估计真实 detector graph covariance、view bias 和 temporal drift。

若真实位移特征大部分落在 synthetic 95% 区间之外，当前路由研究只保留为接口和负结果；必须先改数据生成器，而不是增加模型容量。

**证据等级。** **L2 + L3**。真实 PSU support/detector geometry，analytic morphology，synthetic noise，post-open mechanism probe；没有 real measurement training、fresh repeat、experimental field truth 或 superiority。

## 40. 真实 PSU 位移证明当前 synthetic detector 工作域不够

VD0-B 判 NO-GO 后，立刻执行了原定的 P0 measurement-distribution audit。读取公开 PSU `epsu/epsv`，对真实和 synthetic 都做 per-view RMS normalization，只比较 detector 邻域空间结构，不把 RMS 当作 measured noise sigma。

PSU 当前本地 9-view 数据只有 **1 个真实物理流场**。枚举 6-9 active views 得到 130 个 camera subsets，但它们明确不是 130 个独立样本。

结果：

- validation/calibration 到 train 中心的稳健距离中位数 **1.646 / 1.350**；
- real subsets 为 **3.076**；
- real 到最近 train row 的中位距离 **1.873**，validation/calibration 为 **0.883 / 0.755**；
- **130/130** real subsets 至少一个 informative feature 超出 train 95% 包络；
- 平均 feature outside fraction 为 **23.99%**，validation/calibration 只有 **2.72% / 7.44%**。

超界最明显的是 neighbor contrast 和 local Jacobian：真实 mean contrast **1.544**，synthetic 97.5% 上界 **0.897**；真实 mean log-Jacobian **3.392**，synthetic 上界 **2.851**。

这不能证明差异一定来自 shock，因为 optical-flow noise、registration、mask boundary、camera bias、finite aperture 和真实高频 front 都可能贡献。但它足以证明当前 synthetic generator 没有覆盖真实输入工作域。因此 set encoder 继续封存，下一步改成 flow-off covariance、graph-correlated noise 和 held-out camera/front endpoint。

**证据等级。** **L0 输入值 + L3 工作域审计**。使用真实公开 deflection values，但没有实验 3D truth、独立 flow fields、reconstruction 或训练；只能证明 descriptor mismatch。

## 41. 先回答“要多少张 flow-off”，再谈真实 covariance

公开 PSU 论文说每次测试原本拍了 2000 张 flow-off 和 2000 张 flow-on，但公开
ZIP 索引只给每个 camera-rotation condition 一张平均 flow-on TIFF 和复合
reference/deflected 产物。98 张 `withoutCylinder` TIFF 是不同标定靶角度，不是
同一条件下的时间重复。因此公开包可以给 detector geometry，不能用来估时间
covariance。

这次在真实 PSU 九相机 detector 坐标上做了一个采集规划实验。每台相机的
256 条 detector rays 构成 8 邻域图，模型从简单到复杂依次是：

1. `u/v` 两分量 IID；
2. graph-heat 空间相关；
3. graph + 每个 detector node 的平滑异方差；
4. 在白化坐标中再加一个 rank-1 低频同步漂移。

每次只用 75% repeats 拟合，25% repeats 选择是否启用复杂模型；最后在 160 张
完全封闭的合成测试帧上算 likelihood、coverage 和 harm。三类压力族、8 个
随机种子、9 台相机共得到 7776 条 trial rows。

**结果：**

- 4/8/12/20/32 张都没有通过全部门；
- 32 张最坏 coverage p90 误差仍是 12.44 个百分点；
- 50 张降到 5.625 个百分点，第一次通过 8 个百分点门；
- 50 张时 graph truth 的 NLL gain 中位数是 0.03448 nat/dim；
- IID false activation 为 0，p90 harm 为 0；
- nonstationary truth 的 rank-1 drift 启用率达到 90.28%，IID 为 0。

**讲人话：**20 张足够“看出图相关性可能存在”，但不够把不确定度校准得稳。
32 张已经接近，却仍有坏尾部。当前给师兄的请求应明确写成每台相机至少
50 张，并把约 13 张永久留出，不能先平均、不能拿去挑模型。

这个结论仍只是采集规划，不是重建成功。图 covariance、graph Matérn 和低秩
漂移都不是空白创新。真正可能形成论文的部分，是把真实 flow-off 标定接入
BOST detector graph、held-out camera、whitened PCGLS 与 front reconstruction，
并证明它改变了真实三维反演的可靠性。

完整说明：

- [DG-CovGate 技术说明](psu_b0_detector_graph_covariance_acquisition_gate_2026-07-17.md)
- [公开归档 repeat 审计](psu_flowoff_repeat_inventory_public_summary.json)
- [结果图](../demo_t16_operator/results/psu_b0_detector_graph_covariance_gate/psu_b0_detector_graph_covariance_gate_figure.png)

**证据等级。** **L2 geometry + L3 synthetic acquisition planning**。使用真实
detector graph，未使用真实 temporal repeats，未做三维 reconstruction，不宣称
算法优越。

## 42. 正确 covariance 确实帮助平均重建，但坏尾部仍然否决它

DG-CovGate 回答了“50 张 flow-off 才够稳”，这次继续问更关键的一步：
把正确 covariance 接进三维 inverse 后，field 和 front 是否真的改善？

我先实现了一个线性 detector whitening wrapper：

```text
B(x) = L A(x)
B^T(r) = A^T L^T(r)
```

它支持 component IID、diagonal、graph heat、node amplitude 和低秩 drift，
并通过 detector-level 与完整 BOST adjoint identity。包装后固定 K 步 PCGLS
仍然只有 K 次 forward 和 K 次 adjoint，没有把 whitening 当成“免费多跑一次
物理算子”。

单种子 smoke 看起来很好：graph-correlated noise 下 DG-CovGate 的 field gain
中位数是 `+2.567%`，oracle 为 `+2.611%`，而 IID truth 下 gate 自动退回
component-IID。这个结果只用了 3.38 秒，所以我没有继续庆祝，而是立即冻结
16 个全新 calibration/field/noise seed。

多种子结果更真实：

- mean field gain `+1.178%`；
- 16-replicate Student-t 95% CI `[+0.786%, +1.571%]`；
- gradient mean gain `+0.932%`；
- front-F1 mean gain `+0.01225`；
- 但 field p10 `-1.029%`；
- `>1%` harm rate `10.94%`。

预注册要求 p10 至少 `-0.5%`、harm 不超过 `10%`，所以判 **NO-GO**。16 个
replicate 中只有 6 个单次 smoke 过门，10 个不过。

最重要的诊断是 DG-CovGate 与 oracle covariance 几乎重合。annular kernel
平均约 `-2.04%`，thin front 也有坏尾部；oracle 同样如此。这说明问题不是
“50 张还没把 covariance 拟合准”，而是 whitening 改变了 normal operator 的
谱以后，继续使用 IID objective 下选定的 Sobolev strength=5 和固定四步
early stopping，会产生 morphology-dependent bias/variance tradeoff。

**讲人话：**给每种噪声正确的权重，平均上确实更准；但重建算法的“方向盘”
还是按旧路面调的，遇到环状薄结构会偶尔偏得更多。正确噪声模型是必要条件，
不是自动成功按钮。

下一步先在已经打开的 16 种子上做 post-open 诊断：

1. 扫固定 Sobolev strength，检查预条件器是否必须随 covariance 联动；
2. 扫 partial whitening/precision tempering，寻找 mean 与 p10 的 Pareto；
3. 若能把 annular/thin 尾部压住，再冻结全新种子；
4. deterministic 路线过门后，才允许小型 operator/controller 学习 selector。

学习模型必须击败“正确 whitening + 重新条件化的经典 PCGLS”，不能把
deterministic GLS 的收益归功于网络。

完整入口：

- [严格 NO-GO 说明](psu_b0_dg_wpcgls_multiseed_no_go_2026-07-17.md)
- [冻结配置](../demo_t16_operator/configs/psu_b0_dg_wpcgls_multiseed_v1.json)
- [四联图](../demo_t16_operator/results/psu_b0_dg_wpcgls_multiseed/psu_b0_dg_wpcgls_multiseed_figure.png)
- [结果 JSON](../demo_t16_operator/results/psu_b0_dg_wpcgls_multiseed/report.json)

**证据等级。** **L2 real detector geometry + L3 fresh synthetic
reconstruction pilot**。没有真实 flow-off repeats、实验三维真值或 neural
operator comparison。

## 43. 重新条件化很强，但 pooled 早停规则再次证明“平均赢”不够

DG-WPCGLS 的坏尾部出现后，我没有直接训练 controller，而是先做了 120 个
低自由度候选：5 个 spatial tempering、5 个 Sobolev strength、4 个 stages，
再加 full graph anchor。为了避免重复从头求 2/3/4/5 步，我增加了 trajectory
checkpoint 复用，并用逐值测试证明它等于独立求解。逻辑调用仍是 6,784 对，
物理调用降到 2,464 对，运行只用了 41 秒。

原始选择规则挑出 `full_graph_s3_k4`，前后两半相对旧 `component_s5_k4`
都约提高 24%–25%。但这里不能庆祝：`component_s3_k4` 自己已经贡献约
24%，大部分收益只是经典 Sobolev 预条件器重调。

换成公平的同 strength、同 stage 基线后，graph covariance 的真实 pooled
增量为：

- mean `+1.406%`；
- cluster 95% CI `[+1.235%, +1.578%]`；
- p10 `+0.166%`；
- harm `2.34%`；
- worst `-7.920%`。

也就是说，大多数场受益，但极少数 annular/oblique 场会严重回退。为了看
能否只用部署可见量保护尾部，我又保存了 stage 2–5 的 whitened residual、
residual reduction、alpha、beta、relative update 和 gradient/field norm。

正式规则审计有 348 条：

- 单阈值 stage-4/5：0 条通过选择门；
- rollback/continue：5 条通过选择门；
- 最佳规则在 selection 为 `+3.765%`、worst `-1.775%`；
- 到 opened diagnostic 仍有 `+3.340%` mean，但 p10 `-1.746%`、harm
  `12.5%`、worst `-17.532%`。

因此严格判 **`OBSERVABLE_POOLED_STOPPING_RULE_NO_GO`**，fresh 不打开。

**讲人话：**六个总量就像只看汽车的平均转速和油耗，不能告诉我们是哪一个
轮子在打滑。继续加深 MLP 只会更擅长记住 64 个选择场，不会自动获得逐相机
的物理信息。

下一步优先补一个 deterministic TV/Huber-superiorized PCGLS，先看
edge-preserving regularization 能不能天然压住 annular/shock 尾部。若它成立，
再让小型 operator 学 bounded proximal map；另一支线才是保留逐相机 detector
graph、pose 和 covariance spectrum 的 set controller。

完整说明：

- [严格 NO-GO](psu_b0_covariance_conditioning_stopping_no_go_2026-07-17.md)
- [结果四联图](../demo_t16_operator/results/psu_b0_covariance_stopping_rule_audit/psu_b0_covariance_conditioning_audit.png)

**证据等级。** **L2 real detector geometry + L3 post-open synthetic
mechanism audit**。没有真实 flow-off、实验三维真值或 fresh confirmation。

## 44. TV/Huber 方向做对了，但它的额外 forward 不划算

上一节决定先补 TV/Huber 强基线。这次没有把 PCGLS 结果拿去随手平滑，而是
按 SupPCG 的定义实现：

1. 每轮先沿 TV 或 Huber 的负梯度做不增加 penalty 的小扰动；
2. 步长按 `gamma * a^ell` 递减，保证扰动总量可控；
3. 扰动后重建 measurement residual；
4. 再做 fixed-SPD PCG 更新。

关键成本是第 3 步。普通 PCGLS-K 用 `K F + K A^T`，SupPCG-K 用
`(2K-1) F + K A^T`。所以 SupPCG-3 必须与总调用相同的 graph-PCGLS-4
比较，不能只和同 stage 的 graph-PCGLS-3 比。

两个已打开 replicate 的初始 scale smoke 中，最佳 Huber-3 在同 stage 仅有
`+0.124%` mean，小于 1% 的坏尾为 0；说明 front/annular 的确有一点结构信号。
但换成同总调用的 graph-PCGLS-4 后：

- mean `-6.016%`；
- p10 `-10.299%`；
- harm `87.5%`；
- worst `-15.551%`。

唯一授权的深阶段扩展也失败。48 个候选中最佳 Huber-6 对调用预算下界
graph-PCGLS-8 为：

- mean `-8.518%`；
- p10 `-25.463%`；
- harm `68.75%`；
- worst `-26.411%`。

因此按预先写入配置的停止规则，关闭 SupPCG 性能分支，不再调 gamma。

**讲人话：**TV/Huber 的小修正确实偶尔能让边缘更好，但每修一次都要重新拍
一遍“虚拟投影”。同样的计算钱拿去多做一轮普通 PCGLS，整体更划算。继续
调步长只会在两个已见 replicate 上过拟合。

下一步换成每迭代只用一对 `A/A^T` 的 primal-dual/PDHG，直接求解
data + TV/Huber 目标。只有它能同时改善 mean 和坏尾，才考虑让小网络学习
bounded proximal 参数。

完整说明：

- [严格 NO-GO](psu_b0_edge_superiorization_budget_no_go_2026-07-17.md)
- [结果四联图](../demo_t16_operator/results/psu_b0_edge_superiorization_tail_smoke/psu_b0_edge_superiorization_no_go.png)

**证据等级。** **L2 real detector geometry + L3 two-replicate post-open
scale/tail smoke**。没有 full opened grid、fresh、真实 flow-off 或实验真值。

## 45. One-pair PDHG 跑完了：问题不是慢一点，而是几乎没离开零场

这轮先发生了一次必须如实保留的基础设施失败。v1 的 12 条 stress trajectory
各自完成 32 轮后，审计代码尝试让 MPS tensor 在一次操作里同时搬到 CPU 并转
`float64`，PyTorch MPS 不支持，于是得到 `PDHG_PREFLIGHT_INVALID`。它没有产生
任何性能行，所以不能说算法成功或失败。

我没有覆盖这次失败。原始 JSON 留在本地私有审计库，公开仓库只保留脱敏摘要；
随后冻结 v2，只允许把导出改成“先搬 CPU，再在 CPU 转 float64”，其他数据、
步长、候选、门槛和 MPS float32 求解全部不变。新增回归测试后，E1 116/116 tests
和全仓 875 项测试都通过。

v2 完整跑了：

- 12/12 stress trajectories 通过；
- 32 个 PDHG 候选 + 17 个 controls，共 49 methods；
- 784 条 paired metric rows，0 个 invalid candidate；
- 判决为 **`POSTOPEN_PDHG_SCALE_NO_GO`**。

排名第一是 `pdhg_huber_a1of256_k4`，但“第一”只表示它在 32 个失败候选中最不
差。相对同预算 graph-PCGLS：

- mean field gain `-68.432%`；
- p10 `-120.638%`；
- 16/16 个场都超过 1% harm；
- worst `-140.923%`；
- gradient mean `-31.464%`；
- front mean `-0.2201`；
- 两个 replicate mean 都为负；
- 只有 wall-time ratio `1.207 < 3` 通过。

最关键的不是“TV 没用”，而是 data-only PDHG 自己也几乎没动：

| K | data-only PDHG field-L2 | graph-PCGLS field-L2 |
|---:|---:|---:|
| 4 | 0.999644 | 0.628707 |
| 8 | 0.999121 | 0.549110 |
| 16 | 0.998029 | 0.463761 |
| 32 | 0.995881 | 0.421089 |

零场的 relative error 就约等于 1。PDHG 做 32 轮仍是 0.9959，说明体场还没有
走到 TV/Huber 能发挥作用的位置。32 个正则候选相对各自 data-only 的最好收益
也是微小负数。

原因线索很强：两个 replicate 的 spatial-gradient norm squared 约 78,600，而
data block 只有 2.11–2.78，相差约 2.8 万–3.7 万倍。一个统一 scalar step 被
空间梯度块的最坏尺度压住，data-fitting 每步推进极小。

**讲人话：**我们让一个人同时推轻箱子和一块巨石，又规定两只手每次只能移动
同样短的距离。为了不让推巨石的手失稳，推轻箱子的手也被限制得几乎不动。
下一步不是继续换 TV 的 alpha，而是给 data、空间梯度和不同 voxel/camera
分配各自安全的步长。

下一候选是 covariance-aware signed factor-majorized block-diagonal PDHG。先做
tiny dense majorizer、零耦合、伴随和 diagonal-metric 安全检查；然后只跑
data-only Gate B。若 K=32 不能比 scalar PDHG 至少降低 25% field error，就直接
停止，不加 TV、warm start、nullspace 或网络。

只有 block data-only 真正离开零场以后，才依次解锁：

1. 两个冻结尺度的 TV/Huber activation；
2. 把 graph-PCGLS warm-start calls 计入同总预算的混合方法；
3. geometry-only near-nullspace penalty；
4. 最后才是 bounded learned metric / selector。

完整入口：

- [v2 公开 NO-GO 审计](../demo_t16_operator/results/psu_b0_pdhg_scale_smoke_v2_public/README.md)
- [下一轮 block-diagonal gate](psu_b0_scalar_pdhg_no_go_and_block_diagonal_gate_2026-07-17.md)
- [signed factor majorizer 设计](covariance_majorized_pdhg_design_2026-07-17.md)

**证据等级。** 两个已见 replicate 的 **E2 oracle-scale mechanism diagnostic**。
没有 fresh seed、held-out camera/session、真实 flow-off scale 或 OERF 实验真值；
神经训练继续封存。

## 46. 并行没有拿来同时抢 MPS，而是提前做 Gate A0

为了缩短等待，我把工作拆成三条互不争用的支线：网页与证据只读审计、PDHG
一手文献与创新边界、CPU-only block metric 原型。正式 MPS 仍串行，因为多个
训练/逆解进程会争同一块统一内存，也会破坏 wall-time 的公平比较。

Gate A0 新增了一个不接正式 runner 的 signed factor block-norm 原型和 10 项 CPU
测试。它能检查：

1. 正负 factor 在 forward/adjoint 中保留符号；
2. majorizer 只用 factor coefficient 的绝对值；
3. 空 primal/dual block 和非正步长 fail-closed；
4. 用声明的 factor norm bound 构造后，tiny dense 真正的 normalized `K` 范数小于 1；
5. power iteration 只标记为未认证估计，默认不能进入更新；若只做诊断，必须在
   构造和执行两处分别显式 opt-in。

**讲人话：**现在搭好的只是“安全带扣能不能扣上”的小样机，还不是装到真实
BOST 算子上的赛车。它没有逐元素构造 `|W|P|G_c|E`，没有 MPS 正式 runner，
没有 Gate B 性能，更没有创新优势。下一步仍是把真实 factor 的行列 majorizer
接进来，并先在 tiny dense oracle 上逐项对齐；只要 Gate A 有一项不满足，就不
打开 field truth 做性能比较。

## 47. 网速够用，真正卡住的是依赖顺序和逐元素证书

先把“是不是网络太慢”排除掉。本机实测下行 `75.8 Mbps`、上行 `37.5 Mbps`，
下载代码、论文元数据和中小文件已经够用；`284 ms` latency 表示一次请求来回要
等约 0.284 秒，所以逐个打开很多小网页会显得拖沓，但可以靠批量请求和缓存减轻。
当前耗时更大的部分是本地测试、矩阵算子核对和一次只能跑一个的 MPS 数值任务，
因此继续换网络并不会解除当前的主要阻塞。

执行方式也从“前一件做完才开始下一件”改成了依赖图：互不依赖的文献核验、
因子接口和 CPU 测试可以并行；只有上游证书齐全后，才把它们汇入 Gate A；正式
MPS 仍保留一个串行任务，避免争用统一内存并破坏计时。固定 4-worker 的最新
记录是 `958` 项并行测试通过，再单独跑 `1` 项 MPS parity，包含 151 项
fast 合同和旧 artifact 链接审计的 medium 总时长为 `14.29 s`。这里提速的是反馈循环，不是
科学门槛。

A0 红队同时纠正了一个关键概念：norm-bound prototype 只说明“按一个整体范数
上界缩小步长后，tiny 矩阵没有越过稳定边界”；它不等于逐元素 factor
certificate。正式证书必须知道每一行、每一列收到多少绝对系数，并处理严格为零
的行列。只给一个整体 norm，就像只知道整栋楼的总承重，不能据此断言每根梁都
分配正确。

这一轮已经把证书需要的部件分别做出来并用 tiny dense 或伴随恒等式核对：活动
坐标的 `E/E^T`、三线性插值的 `P/P^T`、中心差分的 `|G_c|/|G_c|^T`、先组合再
取绝对值的 `|HRQ|/|HRQ|^T`、前向 Neumann 正则项的 `|D_+|/|D_+|^T`，以及删除
严格零耦合行列后仍保留目标函数常数项的 zero constant ledger。signed chain
也已经与原来的物理 forward/adjoint 组合逐值对齐。

为什么 `|HRQ|` 不能写成 `|H||RQ|`？取一个最小例子：

```text
H  = [1, 1]，RQ = [1, -1]^T
HRQ = 1*1 + 1*(-1) = 0，所以 |HRQ| = 0
|H||RQ| = [1, 1] [1, 1]^T = 2
```

真实组合里两个带符号通道会相消；如果每一层先取绝对值，相消信息就被抹掉，
得到的是另一个更松的上界 `2`，不是组合矩阵该位置的逐元素绝对值 `0`。所以要
先把 `H`、`R`、`Q` 的带符号系数组合完，再对组合结果取绝对值。

边界仍要说清：目前只在单一冻结 scale、view-local covariance fixture 上把分段接口、
端到端 signed chain、ones-pass 和 production 6-step Huber recurrence 与 site-major
dense oracle 对齐；但冻结 fingerprint、
clean-commit CPU/MPS attestation 与独立 validator 未通过，Gate B 的同预算性能比较
更没有打开。

下一步只有四项：

1. 冻结配置、输入、测试节点和代码 fingerprint，补齐 setup/solver/scorer 及
   signed/absolute 调用账本。
2. 在同一冻结 fixture 上完成
   CPU/MPS Gate A attestation；任一项不符就停在 Gate A。
3. 只有 Gate A 全部通过后，才串行运行 Gate B，对 scalar、block、factor 与
   graph-PCGLS 做同调用预算比较。

## 48. 红队真的拦住了四个“看起来能跑”的错误

第一次单 fixture 因子链组装后没有直接宣布 Gate A 通过，而是交给独立红队找反例。
它抓到了四个会制造假安全的问题：TV 三分量的展平顺序与 site-major 数学
合同不一致；三线性插值的 `-1` 索引会静默读取最后一个 voxel；fast 门原来
没有跑新因子测试；Pages 会把含 truth/weights 的 `.npz` 复制进公开产物。

四项现在都已 fail closed：TV 在进入 dense oracle 前显式转成
`(z,y,x,component)`；索引、shape、dtype、valid/weight 一致性均在构造阶段检查；
fast 当前直接运行 170 项合同测试；Pages 默认拒绝 `.npz/.npy/.mat` 与 checkpoint/key。
另外补上了 Huber 分段目标和孤立终端 TV site 反例。

修复后，production matrix-free 路径用同一个 target 跑 6 步 Huber PDHG，每步的
primal、extrapolated primal、data dual 和 TV dual 都与独立 dense oracle 对齐；目标值
还显式加回了删除零行的常数项。最新快速门为 `170 passed`，四进程源码测试
`977 passed`，串行 MPS parity `1 passed`，完整 medium matrix `16.81 s`。

**这一次学到的东西：**快不是少做审计，而是把审计放进两三秒内必跑的反馈环。
当前只能标为 `GATE_A_PRE_ATTESTATION_MECHANICS_ONLY_VIEW_LOCAL_SINGLE_FROZEN_SCALE`；在 fingerprint、clean commit 和独立
attestation 完成前，不得称 `GATE_A_PASS`，更不得说新算法已经更好。

## 49. 提速了验证反馈，没有越过科学门槛

实测表明，当前主瓶颈不是下载网速，而是本地计算和证据依赖顺序：CPU 源码
测试固定用 4 个 worker 并行，MPS parity 和正式数值任务仍串行，避免争用统一
内存和污染计时。统一 medium 反馈由上一轮串行的 `28.91 s` 降到已验证的
`16.81 s`，在测试数增加后仍缩短约 `41.9%`；这只说明测试周转更快，不是算法性能结论。

当前狭口径仍是 `GATE_A_PRE_ATTESTATION_MECHANICS_ONLY_VIEW_LOCAL_SINGLE_FROZEN_SCALE`，正式 Gate A attestation **未通过**。几条
看似琐碎的边界是为了防止“能跑”变成假证据：exact-zero 只允许删除严格零
耦合，不得用近似零偷换问题；view-local 索引防止把全局射线编号错当某一视角内
编号；single-instance 限制防止把一个样本的 calibration scale 或 metric 广播给其他
样本；call ledger 必须把 setup、solver、scorer 和绝对值因子调用分开记账，否则
同预算比较会虚假便宜；deleted-constant 必须加回删除零行留下的目标函数常数，
否则缩约前后的目标值不再可比。

要到可发布的声明，还需冻结 config、input、test-node 和 code fingerprint，在 clean
commit 上完成固定 fixture 的 CPU/MPS attestation 与独立 validator；Gate A 全过后
才能打开 Gate B 的同调用预算比较。即使 Gate B 有信号，仍需独立 flow-off/calibration
scale、held-out camera/session 和真实实验证据，才能超出“机制实验”的窄结论。

## 50. 红队用底层写入绕过冻结，我把它继续封住了

第二轮红队不是重复跑测试，而是专门扮演“不守规矩的调用者”。它先构造互相矛盾的
whitening metadata，又让 measurement/TV 子类实际算两次却伪报一次；这说明只相信
公开 `call_report()` 不够。现在 pre-attestation 只接受 sealed exact 实现，view-local、
single-scale 和 cross-view support 字段必须彼此一致，物理账本直接读取底层计数器。

随后红队又用 `tensor.data[...]` 改写 kernel。普通 `add_()` 会增加 PyTorch `_version`，
但这种 storage 写入不会，所以第一版冻结令牌仍会放行，删除零行的目标常数也随之失真。
修复后，令牌还包含所有 setup-critical tensor 内容的 SHA-256；普通写入和 storage 写入
都会在 solver/scorer 前被拒绝。

这项严格检查会把 tensor 同步到 CPU 做 hash，所以只能用于 tiny mechanics fixture，
不能拿它测新算法速度。未来 Gate B 要另做不可变执行副本，只在计时前后核验 hash。
当前状态仍是 `GATE_A_PRE_ATTESTATION_MECHANICS_ONLY_VIEW_LOCAL_SINGLE_FROZEN_SCALE`；
它没有“PASS”字样，也不授权性能、fresh、真实重建或论文胜出结论。

## 51. 这次 Gate A 真的通过了，但只通过了 mechanics

这次没有把“测试绿了”直接写成通过。先把源码冻结在 clean commit，再由正式 runner
生成报告；随后另一套 NumPy dense oracle 不导入 production solver，从 JSON 原语重建
所有矩阵和六步 recurrence。第一次独立验证通过后，我又跑了第二次 `--no-write`。

第二次复核真的抓到一个问题：科学数值没有变，但发布前后目录多出两个文件，目录
安全预检的计数被误混进 core checks，导致 validation JSON 不能逐字复现。修复后重新
提交、重新生成、重新验证，最终稳定为：13/13 E1 PASS，20 个 selector 展开 34 个
case、零跳过，独立 validator 333 项 core checks；NumPy 六步最大状态误差约
`4.13e-16`，MPS 最大状态差约 `1.04e-7`。

讲人话就是：这个很小的 frozen mechanics 题上，公式、代码、伴随、步长、删除零项
和调用次数终于对得上了，而且别人不必相信报告里的 PASS 字样，可以自己重算。

边界同样重要：Gate B 没跑，fresh 没开，真实 OERF 没跑，没有任何模型胜出。执行
环境虽然哈希了完整 Torch、NumPy、pytest 安装树，仍是同一台 Mac，不是假装成独立
容器证明。

## 52. 下一算法不再是“再堆一个 FNO”，而是可关闭的学习 proximal

并行研究支线把下一候选收敛成 FM-CG-PDNO：保留显式 BOST forward/adjoint、
covariance whitening 和 factor metric，只让一个小型共享 3D 网络输出受限 proximal
修正。网络输出层零初始化，`beta=0` 时必须逐元素退化回 deterministic factor-PDHG。

这样每个贡献都能单独问责：是 whitening 有用、factor step 有用，还是 learned
proximal 有用。若关掉学习器后不等于经典算法，或收益只来自更多 calls，这条路线直接
失败。Mac 先用 360 个小场、28 个整组隔离 geometry 做证伪；只有 Gate B 的经典
factor metric 已有稳定正信号，才启动神经 smoke。

物理问题仍需师兄选边：有光圈/phantom/高低保真算子对就做 RayKernel-DCO；有真实
timestamp、曝光和缺帧日志就做 TRAIL-4D；只有静态多视角且能永久留一台 audit camera
才重启 GQ-NIO。三条不能一起大训练。完整结构、指标、失败门和六个数据问题见
`docs/fm_cg_pdno_research_route_2026-07-17.md`。

## 53. Gate B 真跑完了：factor 有一点信号，但远远不够

这次不是测试绿了，也不是又做一个小 toy。正式 V4 在 clean source commit
`204bbe8` 上跑了 16 个场、四种算法和 `K=4/8/16/32` 四档预算，共 256 条方法行。
独立验证器没有相信 runner 自报结论，而是重算了 4,048 项 checksum、调用账本、
配对关系和八项门禁；最后确认结果有效，但判决是 **NO-GO**。

讲人话：voxel-factor 像是给每个体素配一只不同大小的鞋，希望在病态地形里走得
更快。它确实比所有体素穿同一双鞋的 scalar PDHG 稍快：15/16 个场有正改善，两次
replicate 的均值都约 1.32%。但预先要求的是至少 25%，实际只有 1.321%；相对只按
相机分块的 view-block 也只有 1.242%，没到 3%。同样 32 次 forward/adjoint，
graph-PCGLS 的 field-L2 已到 0.421，factor-PDHG 还在 0.983，差距 133.4%。

更要紧的是 front-F1：graph-PCGLS 为 0.744，scalar/view-block 约 0.36，factor
反而只有 0.137。也就是说，它在总体 L2 上挪动了一点，却没有保护薄前缘和激波。
对于反应流三维重建，这比“均值改善不够”更危险，因为好看的体渲染可能掩盖真正
关心的结构已经坏了。

这轮还澄清了活动域。support 内有 2,744 个 voxels，但真正被 A-only 数据耦合的
只有 2,322 个，另有 422 个属于测量零空间。不能给这 422 个位置加 epsilon 就说
“可重建”；它们必须靠明确空间先验、时间演化、多模态或额外相机补信息。

因此现在明确停止：

1. 不实现原 FM-CG-PDNO learned proximal smoke；它的经典退化基线没有过 Gate B。
2. 不继续扫 factor exponent、eta、K 或阈值，把 1.3% 调成一次偶然成功。
3. 不加 TV、warm start 后把收益算给 factor；那已经是另一个目标和调用预算。
4. 不打开 fresh seed 去救 development gate 已失败的机制。

接下来 D0 只做根因诊断：在 tiny/streaming opened 数据上比较 exact `|A|` 与 factor
majorizer 的松紧，并看长时轨迹，回答“上界太松”还是“局部对角尺度本来就不是主
矛盾”。它不是新的胜负实验。

真正的论文路线要回到物理问题。有两档光圈、焦平面、phantom 或 paired renderer，
优先做 RayKernel-DCO，让算子学习修正有限孔径/景深/曲线光路的 forward mismatch；
有连续高速序列、timestamp、曝光和 dropout 日志，则优先做 TRAIL-4D。两条都保留
显式光学 forward 与强 graph-PCGLS/NeRIF 对照，不再让网络掩盖一个失败的 solver。

公开四联图、八项门和复核命令见
`demo_t16_operator/results/psu_b0_factor_pdhg_gate_b_public/README.md`。

## 54. D0 把问题问清了：残差快很多，不等于三维场准很多

这轮推进慢的主要原因不是网速。D0 的正式运行、独立重算和 Metric-A 小模型审计都在
本机完成，耗时主要来自 CPU/MPS 数值计算、重复性检查和证据门禁；真正限制下一阶段的
是实验室数据合同还没到位，而不是论文网页下载不够快。

### D0 到底问了什么

Gate B 已经说明 factor-PDHG 的提升太小。D0 没有继续换网络，而是追问一个更基础的
问题：factor majorizer 用一个容易计算但偏松的上界近似 `|A|`，是不是这个上界中的
符号抵消被忽略，导致行、列尺度过于保守？于是 D0 保持 signed `A/A^T`、初值、支持域、
迭代次数和数据完全不变，只把用于对角步长的 factor mass 换成 exact-`|A|` mass。
因此它是根因诊断，不是一个新重建算法的胜负赛。

结果支持“factor 上界过松”这个机制解释。到 `K=128`，exact-abs-row 相对 formal
factor-view 的 normalized residual 改善为 `64.183%`；但 field relative-L2 只从
`0.959944` 降到 `0.913594`，改善 `4.828%`。所以最重要的一句话是：**64% 是数据残差
口径，不是三维场重建精度提高 64%；场误差的对应改善约为 4.83%。**

这里还有两个看起来很像、其实回答不同问题的平均数：

- `ratio-of-means` 是先分别求两种方法的平均误差，再计算两组均值的相对差；D0 的正式
  口径是 residual `64.183%`、field `4.828%`。
- `paired mean` 是先对每个配对场计算改善百分比，再平均这些百分比；对应 residual
  `64.971%`、field `4.905%`。

两种算法都没有错，但它们不是同一个 estimand，不能挑较大的数字混写成一个结论。

### 为什么还不能据此发“更准的重建算法”

exact-abs-row 的场误差在六个预设检查点中，描述性均值最低的是 `K=64`：`0.911423`；
到 `K=128` 反而变为 `0.913594`。逐行看，16 行里有 10 行在 `K=128` 比 `K=64`
更差，而数据残差仍在继续下降。这提醒我们“拟合观测更好”可能不等于“恢复真实场更好”。
但目前只比较了六个离散检查点，也同时看了多个指标；front-F1 甚至没有同向恶化。
因此这里只能说 **K64 是六个检查点中的描述性最低均值**，不能宣布已经发现普适的
semi-convergence 规律，更不能把 K64 直接写成通用早停规则。

样本量也不能写成“16 个独立实验”。这 16 行来自 `2 replicate clusters x 8` 个共享
morphology；同一种形态在两个 replicate 中有关联，所以不是 16 个 IID 样本。当前不据此
构造 p-value、置信区间或广泛泛化结论。

另一个容易忽略的混淆是 synthetic view scaling。当前每个视角的缩放使用 clean-truth
projection RMS。求解递推本身不读取三维 field truth，但完整合成流程仍不是 truth-blind。
真实部署必须改用 flow-off/reference repeats、独立 calibration 或其他观测可得尺度；
否则“尺度估计”和“重建能力”会被混在一起。

公开分析器没有只相信正式报告里的 PASS 字样。它重新读取轨迹和 tightness 数据，分开
重算 ratio-of-means 与 paired mean，核对 16 行的分组结构、K64/K128 关系、Gate B 仍
关闭、signed-`A` 递推边界和 truth-scaling 标记；独立 validator 共通过 61 项检查。
公开包还固定文件清单和 SHA-256，意外多出的旧文件、被改动的正式决策或算术都应当
fail closed。这提高的是结果的可审计性，不会把 synthetic diagnostic 升级成真实实验。

### 接下来的 A、B、C 三条路线

1. **Metric-A：抵消感知的几何条件化对角度量。** 从可部署的几何/算子特征估计
   exact-`|A|` 行列 mass，目标是用更低构造成本接近 exact metric，同时保留 Schur 安全
   审计。它是当前本科主线，但必须先证明不是只学到额外阻尼。
2. **Metric-B：低秩全局残差校正和有限历史。** 在相同 reduced support、相同 signed
   physics 和相同调用预算内，检验少量全局方向能否补足静态对角尺度看不到的耦合。
   A 没过门前不扩大 B。
3. **Metric-C：事件/不确定度感知的停止与正则。** 只有拿到真实连续 4D 序列、时间戳、
   曝光、缺帧和重复测量后才启动；静态 D0 不能外推为 4D 成功。

Metric-A 初版 smoke 已得到一个有用的**负结果**。独立审计发现，预测 metric 后又用
held-out rig 的 exact mass 做逐元素裁剪，这仍依赖 exact oracle；它更像“exact metric
再加学习型阻尼”，还没有实现真正便宜、可部署的替代。两个所谓 OOD rig 的平均结果也
没有胜过 exact：独立补算 `K=32` 时，learned 与 exact 的平均 field relative-L2 分别约
为 `0.40398` 和 `0.36928`，learned 更差约 `9.40%`；平均数据残差也更差。更重要的是，
当前几何特征由 rig index 沿一条一维轨迹生成，换 seed 主要改变 jitter 和噪声，不等于
真正的新几何 OOD。

因此 Metric-A 目前不授权“算法替代”或“优于 exact”的声明。下一门禁必须：把部署输入
类型与 truth/exact mass 完全隔离；独立采样 train、safety-calibration 和 fresh
geometry-OOD；加入 factor、exact oracle、简单标量阻尼、unclipped learned 与
calibrated envelope 五组；冻结 field relative-L2 为主指标，同时报告 residual、Schur
violation、setup 成本和 `A/A^T` 调用。oracle-free learned 若不能在 fresh geometry 上
稳定击败 factor 和简单阻尼，就停止扩大网络，而不是靠加层救结果。

H2 rotation/optical mismatch 仍停在冻结但未构造状态。要启动它，师兄需要提供真实数据
合同：相机几何与 provenance、rotation-40 forward/adjoint、mask、单位、manifest，最好
再有 flow-off/reference repeats、有限孔径或高低保真 paired operator。没有这些输入，
当前最诚实的成果是 D0 的机制诊断、Metric-A 的负审计和清晰的下一实验门，而不是一篇
已经成功的高水平论文。

## 55. Metric-A v2：修好随机种子后，表面上的胜利消失了

初版 smoke 的问题是测试时偷用了 exact mass 做 `max` clipping。v2 先把这个漏洞从接口
上堵住：训练使用 8 个完整 rig，另外 3 个 rig 只用于 safety calibration，最后 4 个
fresh geometry-OOD rig 一次性评分。几何参数由独立随机量生成，noise seed 与 geometry
seed 分开；推理对象 `InferenceRigFeatures` 只携带 row/column 的部署可见特征，不携带
signed `A`、exact mass、truth 或 target。

这次比较的不是“一个网络对一个弱基线”，而是六个标签、五种不同结果：

| 方法 | 4 个 fresh rig 的平均 field relative-L2 | 不安全 rig |
|---|---:|---:|
| exact oracle | 0.703056 | 0/4 |
| calibration envelope | 151737.302297 | 4/4 |
| train-selected `0.5 x factor` | 0.862560 | 4/4 |
| factor majorizer | 0.988963 | 0/4 |
| raw oracle-free learned | 2.180e26 | 4/4 |

`exact-factor interpolation` 在训练集最后选择 `alpha=1.0`，数值上完全等于 exact oracle，
所以不能冒充第六个独立证据点。raw learned 在 OOD 上真正发散，不是图表显示问题。

这里保留一条很重要的研究教训：旧版结果曾因 rig seed 随配置顺序变化而显示“平均值有
信号”。把 seed 改成稳定的 `SHA256(base_seed, rig_id, split_role)` 后，fresh geometry
真正改变，旧数字必须全部作废。新结果逐 rig 展开是：

| fresh rig | envelope | scalar baseline | Schur violations |
|---|---:|---:|---:|
| ood-00 | 0.795978 | 0.950396 | 11 |
| ood-01 | 606945.292717 | 1.063883 | 18 |
| ood-02 | 1.114938 | 1.194957 | 1 |
| ood-03 | 2.005558 | 0.241005 | 9 |

它只在 `2/4` 个 rig 同时胜过 factor 和简单 baseline；`ood-01` 是灾难性 OOD 发散，
`ood-03` 也明显输给简单 baseline。更关键的是四个 rig 全部不安全，共 `39` 次
row/column/spectral violation。raw learned 有 `68` 次，`0.5 x factor` 也有 `28` 次；
只有 factor 与 exact oracle 为零。因此 `metric_substitution_authorized=false`、
`research_claim_authorized=false` 不是保守过头，而是被逐 rig 安全门直接否决。

首轮独立审计发现的工程缺口已经修好：learned/calibrated 路线的 factor 特征构造已计入
访问与成本账本；seed 不再依赖配置顺序；候选设置阶段由运行时 guard 实测 exact 调用为
零；布尔配置测试也不再把字符串 `"0"` 当成真。15 个聚焦测试已通过。当前结果仍来自
未提交源码快照，所以还要做一次独立重算、首次提交、clean-snapshot 重跑和 checksum
核对。修复不会把 NO-GO 变成成功，只会让这个失败结论能够被别人准确复现。

### 下一条更有希望的算法思想

直接预测一个比 factor 更小的 mass 很难给确定性安全保证。更合适的 v3 思路是学习选择
**安全分组**，而不是学习质量数值。若 signed operator 可写成 primitive contributions

```text
A = C1 + C2 + ... + CL,
```

对任意 partition `P`，都定义

```text
M_P = sum over groups G in P of abs(sum over l in G of C_l).
```

三角不等式自动给出 `M_P >= abs(A)`：每个 contribution 单独一组就是旧 factor；全部
合成一组就是 exact oracle；中间分组提供“构造成本 - tightness”折中。网络只选择一个
预定义 partition，任何选择都仍安全；真正要证明的是它能否在 fresh geometry 上比固定
partition 更好，而不是重新证明网络输出数值的上界。

## 56. v3：安全问题解决了，稳定选对的问题还没解决

v2 的失败不是简单的“网络太小”。它暴露了一个硬矛盾：想比 factor 快，预测的 mass
就必须更小；但只要某些位置低估，Schur 安全条件就可能被破坏。v3 因此不再让网络猜
连续数值，而是先手工构造一组一定安全的 partitions，再让小模型只选 partition 编号。

数学上，若固定线性化算子可以拆成

```text
A = C1 + C2 + ... + CL,
```

那么每个分组方案 `P` 都使用

```text
M_P = sum over groups G of abs(sum of C_l inside G).
```

三角不等式保证 `M_P >= abs(A)`。这次 26 个 synthetic rigs、5 种 partitions，一共
130 次审计都是零违反。换句话说，在这个小型生成器里，“选错 partition 会不会把算法
弄得数学不安全”已经被结构性消掉了。

但准确度没有一起解决：

| 方法 | 8 个 fresh rigs 的平均 field relative-L2 |
|---|---:|
| 训练选出的最佳固定分组 `paired_cross` | 0.489638 |
| geometry-conditioned selector | 0.437171 |
| all-in-one exact comparator | 0.316393 |

selector 的平均数比固定分组好 `10.7155%`，看起来是目前最像“算法信号”的一行；但它
只在 `4/8` 个 fresh rigs 胜出，最坏样本反而增加 `0.414402` 的 field-L2。exact 还比
selector 低 `27.6271%`。所以最终仍是 `research_claim_authorized=false`。

**讲人话：**我们已经造出一组不会越过护栏的挡位，但模型只在一半路况下选对挡位。
平均数变好了，不代表可以放心部署。下一步不该把决策树换成大 Transformer，而是加
一个只读可观测量的风险门：没有足够把握时回退到训练选出的固定分组，并用完全独立的
risk-calibration split 冻结接管阈值。

独立审计也抓到了三条必须公开的限制：当前 6 个所谓 safety rigs 参与了最终 stump 选择，
只能算 model-selection，不是独立验证；toy 里所有方法都拿到完整 primitives，所以
“all-in-one exact 很贵”尚未被真实内存/流式接口证明；成本数字只是解析 proxy，不是
wall time。当前源码还没被 commit 锚定，validator 的性能与成本重算正在补强。因此这
轮最准确的身份是：**有严格安全构造、但选择性能没有过门的合格负结果。**

真实迁移的第一问也已经变得非常具体：师兄的 BOST/NeRIF forward 在固定线性化点，
能否导出 `J = sum(C_l)` 的 signed primitives？如果不能，就立即停止 v3；如果可以，再
判断自然分组究竟是 view、折射率梯度分量、aperture/quadrature sample 还是 ray segment。
完整接口地图见 [v3 real-BOST interface map](v3_real_bost_interface_map_2026-07-17.md)。

这一步仍不需要租服务器。16^3/tiny dense 的结构证伪在 Mac 上很快；只有风险回退在
全新 fresh geometry 上同时过 selection-conditional harm、coverage、field/front 与
真实构造成本门，才值得扩到 32^3 或真实 BOST decomposition。

## 57. v4：不是继续堆网络，而是先学会什么时候不该接管

v3 留下的矛盾很清楚：所有候选 partition 都有数学安全证书，但 selector 在 8 个 fresh
rigs 中只赢 4 个，最坏样本还恶化 `0.414402`。这说明“不会把迭代步长弄得数学不安全”
和“会为当前几何选到更好的重建路线”是两件事。把树模型换成更大的 FNO 或 Transformer
不会自动消掉这个尾部风险。

v4 暂名 **RCCF（Risk-Calibrated Certified Fallback）**。它保留 v3 的确定性 majorizer，
但把学习器的权力缩小成“是否接管”：

```text
风险证据充足且几何仍在校准支持域内 -> 使用 selector 选择的安全 partition
否则                                -> 回退冻结的 paired_cross
任何数学证书失败                    -> 中止，不允许用回退掩盖
```

数据必须按完整 rig 分成四路，而且各自只有一种职责：

| split | 可以做什么 | 绝对不能做什么 |
|---|---|---|
| train | 拟合候选 selector 与风险分数 | 决定最终阈值 |
| model selection | 冻结特征、模型族、阈值网格和 fallback | 冒充独立风险验证 |
| risk calibration | 给冻结策略计算风险上界与 coverage 下界 | 继续换模型、换特征 |
| fresh test | 一次性报告最终 harm、coverage、field/front 与成本 | 回流调参 |

### 初学者最该理解的统计事实

假设风险校准中有 `n` 个真正被 selector 接管的独立样本，并且一次危险伤害都没看到。
95% 单侧 Clopper-Pearson 上界仍不是 0，而是

```text
1 - 0.05^(1/n).
```

要让这个上界不超过 5%，`n` 约需 59。若还从多个阈值里挑最好的阈值，需要多重比较
修正，样本通常更多。因此当前十几个或几十个 synthetic rigs 只能证明代码没有泄漏、
回退逻辑可重放，不能写成“已证明真实伤害率小于 5%”。这是证据规模问题，不是模型速度
问题。

### 首轮 micro-smoke 真正要回答的问题

1. fresh 推理是否完全不读取 truth、target、primitives、signed matrix 或未来轨迹？
2. train、model selection、risk calibration、fresh 的 rig 是否互不重叠？
3. 修改 fresh truth 或 target 后，选择结果是否保持不变？
4. 修改 geometry feature 后，选择能否按冻结规则变化？
5. 篡改风险阈值、fallback 标志、feature hash 或 split role，validator 是否拒绝？
6. 所有候选和 fallback 的确定性证书是否仍为零违反？

即使六项全过，结论也只是 **interface/protocol pass**，不是“RCCF 已优于 FNO、DeepONet
或 NeRIF”。下一证据域依次是：可控 ASTRA/TIGRE 层析、公开 flight-body BOS、多物理
PDEBench，最后才是 OERF 的固定线性化 Jacobian 和独立 session。

### 现在最需要师兄回答的一句话

> 现有 BOST/NeRIF forward 能否在同一固定线性化点、同一 mask 与同一 ray sampling 下，
> 暴露满足 `Jv = sum_l C_l(v)` 且 `J^T q = sum_l C_l^T(q)` 的 signed primitives？

若答案是否定的，RCCF 不应该继续包装成真实 BOST 算法；我们会保留方法学负结果，并把
主力转向不依赖 primitive 的有限历史校正或真正有连续 metadata 的 4D 路线。完整预注册
见 [v4 RCCF protocol](v4_risk_calibrated_certified_fallback_protocol_2026-07-17.md)。

## 58. v4 首次红队：回退机制真的工作了，但 selector 暂时没有可用价值

最高模型先完成了一版 RCCF micro-smoke，但独立复核没有直接放行。第一版虽然报告
`0/3` 校准伤害和 `0.415` 风险上界，却漏了有限阈值搜索的多重比较修正，也把经验
coverage 当成置信下界；fresh 风险还曾用全部 fresh rigs 作分母，而不是只用真正接管的
样本。这些数字已经作废，不能进入论文或公开结论。

修正后的 v1.4 做了六件关键事情：

1. 三个冻结阈值使用 Bonferroni；风险和 coverage 各用 `0.025` family budget，联合置信
   下界至少 95%。
2. 同时计算风险 CP 上界与 coverage CP 下界，任一不过就全局回退。
3. 校准对象绑定 rule、阈值网格、feature schema 和支持包络的 SHA-256；跨 rule 或 off-grid
   阈值会拒绝。
4. `float32`、二维特征和支持域外几何都只能 fallback；fresh 选择接口仍不接收 truth、
   target、primitives、signed matrix 或 solver trajectory。
5. 在任何离线轨迹前，逐 rig 验证求解器实际使用的 `A` 真等于 `sum(C_l)`；validator 既
   全量重放，也用 SciPy 独立重算 CP 分位数和 selection-conditional 分母。
6. 第二轮红队发现，仅绑定 rule 不够：攻击者可把 `Rmax=0.5` 放宽并重算一个内部自洽
   hash。v1.4 因此让运行器从冻结 config **独立计算预期政策指纹**，同时绑定 alpha 分配、
   risk/coverage 门、joint harm endpoints 与容差；“放宽门并重算内部 hash”的攻击现在也会
   被拒绝。当前四个相关测试文件共 `46 passed`。

提交前开发重放的结果是：

| 证据 | 修正后的结果 | 能说明什么 |
|---|---:|---|
| 完整 rig split | 8 / 6 / 12 / 8 | train、选模、风险校准、fresh 已分开 |
| partition / decomposition 审计 | 136 次，0 违反、0 mismatch | tiny generator 上的数学/接口一致性成立 |
| calibration 诊断接管 | 1 / 12 | 支持包络和规则只覆盖极少校准样本 |
| 风险上界 | 0.991667 | 远高于开发门 0.5，更不可能达到论文门 0.05 |
| coverage 下界 | 0.000697 | 远低于开发门 0.25 |
| 真正获准 coverage | 0 | 校准失败后没有偷偷保留接管 |
| fresh 接管 | 0 / 8 | 八个 OOD rig 全部回退 `paired_cross` |
| worst takeover harm | 未定义 | 没有接管样本，不能写成 0 harm |

因此 `synthetic_micro_interface_gate_passed=false`，所有真实 BOST、泛化和优越性主张继续为
false。这里的好消息不是“模型赢了”，而是失败模式已经变得可解释：数学证书和回退逻辑
工作；当前 observable rule 加轴对齐支持包络几乎没有覆盖，selector 没有实用价值。

下一步不能事后放宽包络来追 fresh 分数。合法路线只有两条：

1. 在 train / model-selection 内预先比较轴对齐包络、正则化 Mahalanobis 距离和小型 kNN
   support score，再用全新 risk calibration 冻结一种；
2. 增加真正独立的 geometry clusters，并迁移到 ASTRA/TIGRE 空间层析，让 joint harm 加入
   front 指标后重新校准。

如果扩大独立校准样本后仍无法同时得到风险上界、coverage 下界和端到端成本优势，就应
停止 RCCF selector，把“认证分组可构造、可靠选择不可得”作为负结果，并转向有限历史或
真实 4D 时序路线。最高模型可以加速实现与审查，但不能把 12 个校准 rig 变成 94 个独立
接管证据。

## 59. PSU-S16：先换掉共享离散链，再讨论网络是否更强

这一轮没有继续在旧 toy rig 上堆 selector，而是把真实空间基准缺的三块基础设施补齐：

1. `spatial_support_gate.py`：axis envelope、robust diagonal、shrinkage Mahalanobis、kNN 四类 truth-free support score；split-conformal 只负责给冻结分数校阈值。feature order、单位、grid、renderer 与 sampling context 全部绑定哈希；fit 中常数特征变成 exact-match，不匹配直接 fallback。
2. `analytic_bost_phantoms.py`：四类连续反应流形态代理直接返回 `q(x)` 与手写解析 `grad q(x)`，QMC renderer 不导入体素差分、三线性插值或逆算子。
3. `spatial_reconstruction_metrics.py`：新增 field/NRMSE/bias、解析 H1、ASSD、HD95、容差 surface-F1 和法向角；旧 top-10% exact-overlap F1 降级为辅助诊断。

六个相关测试文件现场运行 **55 passed**。随后在真实 PSU 九视角 support 几何上跑了第一条独立 renderer 闭环：解析 QMC-64 出题、QMC-8 体素算子答题、16 cubed、四种 morphology、1% view-RMS 噪声、固定 CGLS-12。独立 validator 与 5 个 checksum 文件全部通过。

结果并不好看，但比一个虚假的成功有价值：QMC-32/64 最大相对差只有 **0.1185%**，而平均 field-L2 仍为 **0.7087**，平均 H1 为 **1.4290**；support reprojection 却只有约 **0.062-0.070**。也就是说，积分精度和观测拟合都不能替代三维可辨识性。

同一 opened development 域上的快速 headroom probe 进一步筛掉了一条太小的想法：逐样本选择 Sobolev 标量强度的 truth oracle 只比最佳固定强度好 **1.711%**；加入各向异性轴权重也只有 **2.690%**。最佳专家确实按 plume、褶皱界面和压缩层分化，但这点上限不足以把“预测一个强度”包装成高质量论文。

**讲人话：**现在我们有了一张更难作弊的考卷。CGLS 很会把投影对上，却还原不好场；强 Sobolev 能把场误差压低，却牺牲重投影。下一算法真正要解决的是这个 Pareto 矛盾，同时守住 front 和 OOD 尾部。旧 free/nullspace corrector、positive spectral direction、简单 residual router 都已有 NO-GO，不能换名字重做。

当前准确状态：**E1 independent-renderer interface PASS；算法胜出未测试。** 完整数字、图、代码与复现命令见 [PSU-S16 独立解析 renderer 说明](psu_s16_analytic_renderer_smoke_2026-07-17.md)。

## 60. rotation-40：第一次真正打开未见风洞观测，但先停在正确的门前

这一轮按预注册只打开 PSU flight-body 数据集的 `rotation 40` 开发文件，rotation 30、
60、70、80 仍然封存。archive SHA、member bytes、CRC 和抽取后 SHA 全部匹配；公开仓库
仍不包含作者原始数组。

文件不是三维真值，而是七台相机各自的 `u_new / v_new` 位移和两类 mask。相机 2、3、4
已按作者脚本的符号与 mask 规则生成私有 shard。它们的 active vector RMS 分别为
`0.2462 / 0.3143 / 0.3257 px`，而 ambient RMS 已有
`0.1756 / 0.1812 / 0.2092 px`。这提醒我们：真实测量中的背景偏差并不小，不能把合成实验
里简单的 1% 白噪声当成完整现实。

但我们没有立刻画一张“真实泛化成功”图。这个 MAT 缺 rotation-40 的 camera extrinsics、
background extrinsics、逐像素 ray directions 和同一行绑定的 camera constants。没有这些
量，预测和像素行可能错位，任何 residual 都可能是伪数字。对应的官方背景标定成员正在
单独取回；几何通过作者脚本行级复核之前，`reprojection_scored=false`、
`algorithm_superiority=false`。

另一个小而关键的补丁是把前沿指标从“正好一个界面”扩展到 `0–2` 个界面：现在使用
Hungarian 一一匹配，同时报告 ASSD、HD95、F1@1dx/2dx、法向角、漏检和假阳性；无界面
场景预测出界面也会被罚。新旧指标回归共 **16 passed**。这让下一条“平滑背景 + 显式
phase/interface”候选可以在 plume、单激波和双界面三种情形下接受同一套严格审计，而不
会靠只挑最好看的那一道面取巧。

**讲人话：**真实考卷已经拆开到观测这一页，但坐标和题号还没核对完。现在不抢答，
先把行映射校准好；这比得到一个无法复现的漂亮分数更接近高质量论文。

完整边界见 [rotation-40 开发集说明](psu_rotation40_development_open_2026-07-17.md)。

## 61. 真实未见视角第一次闭环：基线不是“差一点”，而是几乎解释不了

rotation-40 的官方 calibration member 已单独下载并通过 ZIP CRC、字节数和 SHA 核验。
它给出的 `Arotcam` 是绕 x 轴 40° 的旋转；相机 2、3、4 全部 5,529,600 条官方 ray 与
support 0° 行旋转后的最大误差只有 `2.5e-8–3.2e-8`。support 自己的 0/50/90 已知旋转
关系也在 `6.2e-8` 内，`Dfvec / Csys / Rap` 则逐行 exact invariant。

更关键的是，接线时发现了一个会让所有结果作废的 bug：MATLAB `epsu(:)'` 是列优先，
旧 observation shard 却用了 NumPy 默认行优先。现在已改成显式 Fortran order，并用非
对称数组测试锁死；旧 shard 没有参与正式评分。

冻结合同随后在 3,847,050 条真实 active rays 上执行一次 full forward：

| 范围 | relative-L2 | measured vector RMS | predicted vector RMS |
|---|---:|---:|---:|
| Camera 2 | 0.8242 | 0.2462 px | 0.0950 px |
| Camera 3 | 0.9829 | 0.3143 px | 0.0719 px |
| Camera 4 | 0.9856 | 0.3257 px | 0.0855 px |
| pooled | **0.9596** | **0.3027 px** | **0.0826 px** |

所有射线命中 B0，预测有限；6.98 秒和约 1.28 GB 峰值内存说明 Mac 当前完全够用。真正
的问题不是算得慢，而是九视角 support 场几乎不能解释未见 40° 观测，尤其 cameras 3/4
的预测幅值远小于实测。

**讲人话：**以前我们只知道“三维反演可能不泛化”，现在第一次有一把真实尺子量出缺口。
下一算法不能只在 support 上刷 loss；它必须处理可变相机集合、camera-wise 系统偏差、
几何 OOD 和界面先验，并在 synthetic truth 上守住 field/front，在真实数据上守住 held-out
reprojection。这正是“集合条件 INR + 显式数据一致性展开 + phase/interface 表示”比单独
DeepONet/FNO 更有研究价值的原因。

完整方法、结果图和可写/不可写边界见
[rotation-40 真实重投影基线](psu_rotation40_real_reprojection_baseline_2026-07-17.md)。

## 62. 原创性红队：level set 不是创新，分裂更新机制才可能是

最高模型的只读红队把最危险的自我欺骗先划掉了：smooth background + level set 在 2017
年前后已有层析先例；phase-field/perimeter 也有成熟逆问题文献；DeepONet 对移动间断的
线性 reconstruction 下界、FNO/shift-DeepONet 的 nonlinear 对手都已发表；NeRIF 已经占据
BOST 坐标网络与梯度一致性；finite-aperture forward 和 TDBOST 也分别有明确先例。

所以“给 FNO 多一个 `phi` 通道”很难成为论文。现在暂称 **JACRU** 的候选只保留一个可能
有价值的机制：利用

```text
grad n = smooth-side terms + [n] delta(phi) grad phi
```

把 smooth fields、interface geometry、jump amplitude 和 camera bias 分开更新，每层都经过
exact cone-ray data consistency；set encoder 只处理可变相机集合，FNO 只做 smooth proximal。

这条路线也被严格限制为先做“单激波 + 已知上游状态”。接触面、火焰面和爆轰反应区的
跳跃条件不同，不能为了数据量把它们混成一个标签。真正的机制对照必须包含 phase-only
optimization；若 JACRU 只赢 CGLS、却赢不了同参数化的非神经 phase baseline，就没有资格
声称神经更新机制有贡献。

完整一手文献、强基线、失败门和给师兄的问题见
[JACRU 原创性红队](jump_aware_cone_ray_unrolling_novelty_gate_2026-07-17.md)。

## 63. JACRU-M0：算法输了，而且“漂亮的界面分数”原来是初始化送的

这一轮终于没有停在算法草图，而是把一个最小 JACRU 写成了能运行、能被强基线推翻的程序。
观测不是由逆算子自己生成：出题端沿射线积分连续解析梯度，答题端用有限差分加三线性插值的
体素算子。两个 seed、平滑场和单界面场、1% 噪声、2% camera bias；CGLS、Huber-PDHG、
phase-only 和两个 JACRU 版本全部限制为 24 次 forward 加 24 次 reverse/adjoint。

结果非常明确。Huber-PDHG 平均 field-L2 是 `0.4801`，CGLS 是 `0.4989`，带 bias 的
JACRU 是 `1.9878`，差了三倍以上。更严重的是，自动结果里看似很好的界面指标不能相信：
固定 `x` 平面在算法读取任何观测之前，已经对两个单界面样本得到 `F1@1dx = 1.0`；最终
优化后反而降到 `0.974`。生成器的界面方向和初始化方向碰巧对齐，这就是一种答案泄漏。

**讲人话：**好比考试前把一道题的图形轮廓印在草稿纸上。即使程序没有读取 truth 文件，
初始化本身也可能携带答案。今后所有界面模型都要先给“空白初始化”打分，报告最终相对初始
改善，并在无界面场惩罚假阳性。

## 64. M0.1 和 M1：修 bug 不能变成反复调参，真正留下的是职责分工

M0.1 只在已经打开的四个开发 case 上做诊断：按算子范数缩放伴随初始化、随机化平面、让
gate 从阈值以下开始，并降低学习率。field-L2 从 `1.9878` 降到 `0.7690`，修复幅度
61.31%，说明尺度问题确实存在；但它仍比 Huber-PDHG 差 60.18%，所以还是 NO-GO。

M1 换了一个更本质的结构：总预算仍是 24 对物理调用，先把 18 对交给 CGLS，冻结所得主场，
只把余下 6 对交给 jump/interface 残差。平均 field-L2 降到 `0.4950`，比 CGLS 好
`0.78%`，却仍比 Huber-PDHG 差 `3.11%`，H1 差 `15.74%`。只有重投影门通过，界面
gate 根本没有被激活。

这里不能挑 `0.78%` 当成功故事。真正有用的发现是：从零联合优化全场会把有限预算浪费在
经典求解器早已会做的事情上；“物理底座 + 小残差”明显更稳定。跨报告 validator 已确认三轮
每行都使用 24F/24R，授权结论只有“继续测试 learned residual operator”，没有方法胜出、
界面增益或打开 fresh 的权限。

完整数字见 [M0-M1 负证据判决](jacru_m0_m1_negative_evidence_2026-07-17.md)。

## 65. M2：真正的算子学习不是逐样本拟合，而是跨样本学会“经典方法错在哪里”

下一候选暂称 M2。每个样本先跑固定预算 CGLS 得到 `x0`，再计算逐相机数据残差
`r_v = y_v - A_v x0` 和其伴随 lift `A_v^T r_v`。一个共享权重、相机顺序无关的 set
encoder 读取这些 lift、pose 和 active mask，只输出一个受 support 与 gate 限制的小修正：

```text
x_hat = x0 + support * gate * ResidualOperator({A_v^T r_v, pose_v}, x0)
```

它和 M1 的关键区别是：M1 在每个测试样本上重新用 Adam 拟合参数，M2 要在 train 场上学
一个函数到函数映射，再原样迁移到未见 morphology、camera count、pose、noise 和 bias。
因此它才有资格与 DeepONet、FNO、3D CNN 比较“算子学习”能力。

第一道保险是最后一层全零初始化：训练前输出必须逐位等于 CGLS；第二道是 correction bound，
OOD 时不能任意覆盖物理解；第三道是可观测 fallback，风险信号不足就返回底座。训练真值只
用于 train loss，模型 forward API 不能接收 truth、family label 或 interface mask。

这一阶段的目标不是尽快画出赢图，而是用三到五天回答一个小而硬的问题：在相同重建调用预算
下，一个跨几何学习的残差算子能否同时赢 Huber field/H1、守住 CGLS reprojection，并在
多 seed 与 OOD 上不出现尾部伤害？答不上来就继续淘汰，不打开 fresh。

## 66. M2-T0：四个小模型第一次在同一张算子学习考卷上正面对比

M2 已经从框图变成了可运行代码。每个样本先做 12 步 CGLS，再把逐相机
`A_v^T(y_v-A_v x0)`、相机 pose、mask、support 和 `x0` 交给网络。网络不接触 truth、场族
标签或界面 mask；truth 只用于训练 loss 和最后评分。最后一层从全零开始，所以未经训练时
模型逐位退回 CGLS。

这次没有只跑“自己的模型”。同一套 32 个 train、12 个 development、18 个探索性 OOD
case 上，同时训练了 JACRU-M2、pooled 3D CNN、fixed-grid DeepONet 和官方 neuraloperator
FNO；每种方法 3 个模型 seed。参数量从 3,549 到 10,211，都属于 Mac 可以快速证伪的 T0
规模，整轮 MPS 用了 68.63 秒。

**讲人话：**以前只是问“这个想法能不能写出来”，现在开始问更严格的问题：“它比简单 CNN
到底多学到了什么？”如果自己的结构连更简单的模型都赢不了，就没有必要先租大卡放大它。

## 67. 场误差降了四成，但重投影坏了几十倍：这叫形态幻觉，不叫重建成功

结果第一眼很诱人。JACRU-M2 相对 CGLS 的 field-L2 在 development 改善 `46.16%`，探索性
OOD 改善 `32.38%`；H1 也分别改善 `50.24%` 和 `42.68%`。三个模型 seed 都为正，没有
field harm case。

可是同一个预测重新经过物理 forward 后，重投影相对 CGLS 放大到 `28.56x / 35.10x`。
pooled CNN 的 field gain 还略高：`47.11% / 32.80%`，重投影同样坏到
`27.91x / 34.47x`。DeepONet 更保守，却只有 `6.57% / 3.74%` field gain；FNO 在 OOD
出现 `12.96%` harm rate，最坏样本退化 `34.85%`，重投影更达到 `53.16x`。

这说明网络确实学到了合成训练场“通常长什么样”，却把输入观测当成了弱提示。它把欠定逆问题
推向训练分布常见的形状，因此 truth-space 切片更漂亮；但这些形状不再解释相机实际测到的
位移。如果论文只报告 NRMSE 或挑几张 slice，这个失败很容易被误包装成成功。

**讲人话：**像是模型根据往年答案写出一篇很像标准答案的作文，却没有回答这次题目。三维图
更漂亮不够，投回每台相机后还必须对得上原始观测。

完整表格与禁止主张见 [M2-T0 负证据判决](jacru_m2_t0_supervised_residual_no_go_2026-07-17.md)。

## 68. M2.1：下一步不是加宽网络，而是把每次提议拉回测量流形

下一轮先不改网络结构，只在已经打开的 T0 上给四类预测追加确定性数据一致性校正：

```text
x_net = x0 + learned_correction
x_(k+1) = support * (x_k + tau * A^T(y - A x_k))
```

会固定扫描 `0 / 1 / 3 / 5 / 11` 步，画出 field、H1 和 reprojection 的 Pareto 轨迹。这个
post-open 诊断不产生新鲜证据，只回答一个决定路线生死的问题：网络得到的场收益，有多少能在
重新满足观测后留下？

判断标准也先写清楚：若 3--5 步校正能把 reprojection 压回 CGLS 的 `1.10x` 内，同时保留
至少 `5%` field gain，才值得把 exact data-consistency block 写进训练图并进入更大预注册；
若一拉回观测收益就消失，说明当前 correction 主要是错误零空间先验，应该停止，而不是靠增大
参数量硬拟合。未来真正有论文价值的贡献会是“可变几何 residual proposal + 可证明的物理校正
+ OOD 风险回退”的完整机制，不是一张更低 NRMSE 的孤立图。

## 69. M2.1 第一次运行为什么被我自己作废：多用 11 步就必须给经典方法 11 步

第一版数据一致性诊断写完后，红队指出了一个很容易漏掉的公平性问题。learned 路径本来用了
`CGLS-12 + 1 feature pair`；再追加 11 步 Landweber 后，总预算已经是 `24F/24A`。如果还只
和 CGLS-13 比，任何重投影改善都可能只是“多算了 11 步”，不是网络贡献。

所以第一版结果没有进入网页结论，而是原样留作错误记录。v1.1 在重新运行前加入三套逐预算
对照：`CGLS-(13+k)`、`Huber-(13+k)`，以及 CGLS-12 后追加 `(k+1)` 步纯 Landweber。
后者和 learned 路径的总 forward/adjoint 数完全相同，专门拆掉“额外迭代伪成功”。

同时，代码接口新增了 `tau < 2/||A||²` 的硬检查；所谓 nullspace filter 也改成了更准确的
near-null spectral filter。有限步只是 `(I-tau A^T A)^k`，不能写成精确投影。

**讲人话：**如果我比别人多做 11 道演算，不能回头说是神经网络更聪明。先把计算额度拉平，
才知道模型贡献还剩多少。

## 70. 匹配到 24F/24A 后，场收益是真的，重投影失败也是真的

v1.1 共评分 1,620 行 learned 轨迹和 450 行匹配基线；零步结果逐位复现 T0，最大 field 和
reprojection 差都是 0。JACRU 加 11 步 measured pullback 后，development field-L2 为
`0.3424`，exploratory OOD 为 `0.3982`；相对同预算最强经典场基线仍改善
`45.34% / 35.68%`，相对 base-only Landweber 也改善 `49.44% / 39.98%`。

这说明网络确实提供了额外的 truth-space 信息，不能简单归因于多跑物理迭代。但同预算
CGLS-24 的 measured reprojection 已降到 `0.000813 / 0.000904`，JACRU 仍是
`0.03180 / 0.03480`；逐 case 比值达到 `43.12x / 41.95x`。所有 field/H1/harm 门通过，
唯一但决定性的 reprojection 门失败，零个点获准进入 fresh。

near-null 路径也没有接近零空间：11 步后 JACRU 的
`||A delta_k|| / ||y-Ax0||` 仍是 `2.282 / 3.189`，而未来门槛是 `<=0.10`。它不是差一点，
而是固定步 Landweber 在强病态算子上衰减大奇异值分量仍太慢。

**讲人话：**模型带来的三维形状信息可能是真的，但当前“验算器”来不及在有限预算里把错误
成分筛掉。好内容和坏内容黏在一起，这就是下一算法要拆开的东西。

完整判决见 [M2.1 匹配预算 NO-GO](jacru_m2_1_matched_data_consistency_no_go_2026-07-17.md)。

## 71. M2.2 不先造新网络，先问一个更基础的问题：好修正能不能落在允许零空间里

下一步先在 12³ toy 上做 exact SVD headroom oracle：取同预算经典参考 `x_ref`，把网络修正
投到 approximate inverse operator 的精确零空间，得到

```text
x_oracle = x_ref + P_ker(A) (x_net - x_ref)
```

它不是可部署算法，只回答“场收益和内部投影一致性在数学上能否共存”。如果 exact oracle
都保不住至少 25% 的原始 field gain，learned residual 路线应立即停止；如果 oracle 能保留，
再实现 matrix-free Krylov/LSQR 近似，并用相同总调用预算与 base-only Krylov 对照。

即使这一步成功，也不能把“零空间网络”本身写成原创。Deep Null Space Learning、Learned
Primal-Dual、MoDL 和 data-proximal null-space methods 都已有先例。可能的贡献只能来自更窄、
更真实的组合：有限孔径 BOST、可变相机集合、独立 renderer mismatch、matrix-free affine
projection，以及对真实 held-out image consistency 的双域审计。

这里还有一个必须记住的限制：`ker(A_inverse)` 只是体素有限差分近似算子的零空间，不一定是
连续光学 forward 的零空间。未来即便内部 reprojection 变漂亮，也要把预测送回独立解析
renderer 或真实观测验一次，否则仍可能只是服从了错误的物理近似。

## 72. M2.2 exact oracle：终于把“场收益”和“投影一致性”同时放进一个解里

M2.1 的失败留下一个悬而未决的问题：普通 Landweber 太慢，到底是算法路线不可能，还是我们
没有用对投影工具？M2.2 在 12³ toy 上直接组装 dense `A`，对每个几何只做一次 float64 SVD，
把网络 correction 精确分成 row-space 和 numerical-null-space 两部分。

结果给出了第一条真正的正 headroom。所有 12 个几何都是 150 个 measurement 对 1,000 个
active voxel，数值 rank 都为 150，因此至少有 850 维 numerical null space。JACRU correction
的 null norm fraction 在 development / OOD 为 `0.913 / 0.903`；精确删除 row 分量后，
reprojection 与 CGLS-24 一致到约 `1e-14`，field gain 仍有 `45.28% / 37.54%`，H1 gain
为 `43.75% / 40.19%`。

pooled CNN 也得到几乎相同结果：field gain `44.24% / 37.38%`。所以这次授权的是
“通用 learned residual + affine projection”方向，不是 JACRU 结构赢了。

**讲人话：**之前像一桶好水里混了泥，普通滤网 11 次还滤不干净。SVD oracle 证明泥和水在
数学上确实能分开，而且滤完后好内容大多还在；接下来要做的是设计一个不靠昂贵 SVD 的快速
滤法。

完整证据见 [M2.2 exact-null headroom](jacru_m2_2_exact_nullspace_headroom_2026-07-17.md)。

## 73. 为什么这仍然不能叫算法成功

这个 oracle 故意不参与 runtime 或调用预算排名。真实三维 BOST 不可能把百万级算子组装成
dense matrix 再做 SVD。它还只约束 approximate voxel operator：一个 correction 对这个
`A` 不可见，不代表对独立连续 renderer、有限孔径光学或真实相机不可见。

另外，850 维零空间本身就是一把双刃剑。它让网络有地方放入有用的 morphology prior，也让
网络可以把训练集模板藏进观测完全看不到的方向。当前 positive headroom 依赖 synthetic truth
训练和 opened split，不能证明真实 shock、density 或 refractive-index 恢复。

因此状态写作 `HEADROOM_FOUND_ORACLE_ONLY`，不是 `GO`。网页上可以展示它，因为它精确回答了
一个科学问题；论文里若没有 matrix-free 近似、独立 forward 和新数据门，这张图只能作为方法
动机或 oracle 上界。

## 74. M2.3：下一段真正要写的算法是 measurement-space row removal

exact projector 可以写成：

```text
P_row delta = A^T (A A^T)^dagger A delta
```

这提示比体素 Landweber 更直接的 matrix-free 算法。先算 `b=A delta`，再用固定 k 步 PCG
解 `(A A^T + lambda I)z=b`，最后输出 `x_ref + delta - A^T z`。每次 measurement-space
矩阵乘法只调用一次 `A^T` 和一次 `A`；算 `b` 与最后回投各多一对，所以 k 步总计
`(k+1)F/(k+1)A`。

下一轮首先比较 unpreconditioned CG、Jacobi 和固定 low-rank preconditioner。只有普通方法在
有限 k 下明显够不到 oracle，才有理由让网络学习 geometry-conditioned preconditioner 或停止
规则。这样“算子学习”负责加速一个明确的线性代数瓶颈，而不是直接生成无法核验的三维场。

门槛也很清楚：固定 k、同总调用 CGLS/Huber/base-only CG；保留至少 50% exact oracle gain；
reprojection 回到 matched CGLS 的 `1.10x / 1.15x`；再做 camera-count/pose/mask OOD 和独立
renderer。过不了就停在 oracle 动机，不打开 fresh。

## 75. M2.3：公式写对了一半，目标却被旧底座锁住

M2.3 用 PCG 解 `(AA^T+lambda I)z=A(x_net-x_ref)`，把 learned correction 的可见分量删掉。
实现合同通过了，但 exact limit 只能满足 `Ax=Ax_ref`。这里的 `x_ref` 是 CGLS-12；同预算 CGLS
已经继续迭代到更低 residual，所以预条件器再快也不能改变弱 anchor。最好 development
reprojection 仍约为 matched CGLS 的 `14.79x`，正式 NO-GO。

**讲人话：**我们造了一辆更快的车，却把终点设在旧位置。加速器没有办法把终点搬走。

## 76. M2.4–M2.5：目标改成观测仿射集，逐点 Jacobi 仍不够

M2.4 改解 `A x_net-y`，exact affine oracle 能把 residual 压到约 `6e-16`，说明目标集合确实
可达；identity CG 在有限预算下仍慢。M2.5 使用 dense `A` 精确构造 `diag(AA^T)`，最好也只有
约 `15.19x` matched-CGLS reprojection。它关闭了 Hutchinson diagonal 路线：没有必要用随机 probes
去便宜估计一个已经被 exact 版本证伪的结构。

## 77. M2.6：相机分块找到了真实谱结构，但均值不能覆盖受害样本

exact camera-block 把每台相机内部的 50 个 measurement coordinates 联合求逆。K=12 时 JACRU
development field gain `39.01%`、reprojection `0.270x`；CNN 也相近。闭合恒等式误差约 `1e-15`，
所以不是实现假象。

但两种网络都有 `8.33%` harm，最差 field gain 为 `-9.31% / -12.31%`。受害行全部来自同一个
`single_interface / base_seed 2113`，跨六个模型种子稳定出现。exact block 还使用 `1001F-equiv`
dense setup，K=12 超过 24-call 主预算。因此只能写“camera-local coupling 是强机制”，不能写算法成功。

## 78. M2.7：K=9 已经够快，真正失败的是 target/no-harm 联合门

补齐 K=0–10 后，JACRU 在 K=9、总预算 23F 时的 mean reprojection 已为 `0.852x`，CNN 为
`0.914x`；两者都优于 matched CGLS 的平均 residual 门。可是 harm 仍是 `8.33%`，最差为
`-8.89% / -11.89%`。K=10 没有改善尾部。

这一步很关键：不能再说“只要 learned preconditioner 更强就会成功”。solver 已在预算内达到目标，
目标本身仍会伤害含噪界面场。

## 79. M2.8：连看真值的插值 oracle 都救不了简单校准

我们测试 `x(alpha)=x_net-alpha(x_net-x_pcg)`。固定全局 alpha 没有通过点。随后 evaluator 获得一个
不可能部署的特权：对每个样本看真值，并在满足逐样本 `1.1x` CGLS reprojection 门的连续 alpha
区间中选择 field error 最小值。

K=10 时两种网络的可行率仍只有 `97.22%`；问题界面样本即使选择约 `0.99` 的最优 alpha，六个
模型种子的 field gain 仍全部为负。这个上界失败后，不能再训练一个 alpha-MLP 然后声称问题已解。

## 80. 主线转向：噪声感知目标与 fail-closed，而不是继续堆预条件器

下一轮先比较经典 discrepancy stopping、covariance-whitened PCGLS、Huber/Student-t data fidelity，
并要求 held-out camera 或 independent renderer 决定是否接管。只有固定方法先出现 field/H1、
held-out reprojection、harm/worst 与总成本的联合可行区，才允许学习 stopping 或 regularization operator。

完整判决见 [M2.3–M2.8 opened evidence](jacru_m2_3_to_m2_8_opened_evidence_2026-07-17.md)。

## 81. N1.0 先不造网络：只问“看残差决定什么时候停”够不够

M2.8 已经说明，把网络结果和 K=9/K=10 投影结果做固定插值，甚至让 evaluator 看真值逐样本挑
最优 alpha，都不能把 measurement fit 和界面场尾部同时救回来。最自然的下一个问题不是立刻训练
stopping network，而是先测试最简单、最容易解释的规则：每一步只看 measured residual、相对
CGLS-12 的 residual，或 measurement-space system residual，第一次低于阈值就停止。

本轮冻结了 37 个 specs：26 个 residual threshold 候选和 11 个固定 K 对照。它们复用 M2.7 的
K=0--10 轨迹，没有重训、没有打开 fresh。选择器不能看 field truth、clean renderer 或 case
family；阈值未命中就返回 prepared CGLS-12，并仍然支付完整尝试预算。

**讲人话：**这一步是在测试“只看验算分数，能不能知道哪一刻该停笔”。如果连一整组透明、
可解释的规则都找不到安全时刻，就不该马上做一个更黑箱的 MLP 来猜。

## 82. 结果不是所有规则都差，而是出现了两个没有交点的安全区

JACRU 的 26 个可观测候选里，6 个保护了 field tail，11 个保护了 independent clean renderer，
联合安全数是 0。tail 最好的代表 `base_residual_x4` 平均在 K=1.89 停，harm 只有 `2.78%`，
worst 是 `-1.98%`；但 clean renderer residual 平均是 base 的 `1.639x`，最坏 `3.160x`。

renderer-safe 的代表 `base_residual_x1.5` 平均在 K=3.97 停，clean ratio 已降到
`1.096x / 1.298x`，但 harm 回到 `8.33%`，worst 为 `-7.55%`。pooled CNN 更直接：26 个
候选中没有任何 tail-safe 规则；最好的 renderer-safe 规则仍有 `8.33%` harm 和 `-11.78%`
worst。

**讲人话：**早点停，三维形状不容易被噪声破坏，但相机端还解释不好；晚点停，相机端变漂亮，
某类尖锐界面却被错误测量拉坏了。阈值像一扇只能左右移动的门，而我们需要同时照顾门两边的
人，当前没有一个位置两边都安全。

## 83. 这还不是真正的 discrepancy principle，N1.1 必须先有 flow-off 标定

N1.0 的 synthetic noise scale 来自 simulator 配置，不是实验测得的噪声协方差。exact
camera-block 求逆的是 `(AA^T)_camera`，它描述 forward geometry 的谱结构，也不是 detector
noise covariance。更关键的是，在当前 full-row-rank 的欠定算子下，camera bias 可以被某个
三维场修正精确解释；只看一帧 `y`，算法没有信息判断它到底是物理场还是偏差。

所以真正的 N1.1 需要每台相机同一条件至少 50 帧未经平均的 flow-off repeats，按时间块严格
拆成 covariance fit、calibration、selection 和 lock audit，并永久留一台 camera 或一组 rays。
先用这些数据拟合低参数 whitener `W`，比较固定 covariance-PCGLS、Huber/Student-t 和
TV/H1；只有经典方法先得到 joint-safe 区，才学习 beta、proximal step 或 bounded stopping
operator。

完整判决见 [N1.0 observable stopping NO-GO](jacru_n1_0_observable_stopping_no_go_2026-07-18.md)。

## 84. N1.1：先用 flow-off 标定，再做 anchored Tikhonov 上界

这次没有训练新的 operator。我们先把每个 evaluation rig 的 synthetic flow-off 分成三份：64 帧
拟合均值/covariance、64 帧定 threshold、64 帧只做 audit。paired 模式模拟同一 session，允许
flow-off 估计目标帧共享的 camera/component offset；unpaired 模式只学习偏置分布。

然后以网络输出 `x0` 为中心，求 covariance-weighted anchored Tikhonov：

```text
x(lambda) = x0 - A^T (A A^T + lambda C)^-1 (A x0 - y)
```

选满足 calibration discrepancy 的最大 lambda，也就是“观测允许时尽量少改网络”。dense
`AA^T` 只作 toy ceiling：12 个几何合计 12,012F-equivalent，不能说可部署。

**讲人话：**我们先用静止背景测相机自己会抖多少，再决定网络结果最多应该被观测拉动多少。

## 85. 平均增益超过 40%，为什么仍然必须写 NO-GO

paired structured 的 JACRU development field gain 为 `+42.133%`，OOD 为 `+34.420%`；clean
target ratio 也降到 `0.619x / 0.668x`。数字很漂亮，但 `base_seed=2113 / single_interface`
仍在三个模型种子上受害，development harm `8.33%`、worst `-6.054%`。pooled CNN 同一 case
worst 为 `-8.891%`。

更关键的是，直接给 evaluator 精确 bias mean 和精确 IID covariance，受害样本仍未消失。这把
问题从“covariance 估不准”推进到了“bias 与物理场在当前欠定算子里不可辨识、模型失配和 robust
data term 不足”。14 个候选-模型组合全部 NO-GO，0 个 oracle 通过全部门。

**讲人话：**平均分很高，但同一道薄界面题每次都答错。毕业设计要的是能解释并保护这种反例，
不是把它藏在平均数里。

## 86. 打开结果后才发现：还必须保护 raw network center

正式门比较的是 strongest matched classical baseline。打开结果后才补查 correction 相对它自己的
raw learned proposal 是否安全，因此这项只能标为 post-open diagnostic，不能修改正式判决。

paired structured 相对 raw center：JACRU development mean 只有 `+0.716%`，但 27.78% 样本
受害、worst `-22.662%`；pooled CNN mean 为 `-2.394%`，harm 38.89%、worst `-23.229%`。
所有不读 truth、不读 exact nuisance 的候选在双 split raw-safety 六项门下仍是 0 pass。

下一协议必须同时比较 strongest classical 和 raw center。只赢一个参照，不能叫安全改进。

## 87. 红队把 N1.2 的修正顺序定清楚了

N1.1 的 NO-GO 有价值，但协议还有十个不能忽略的缺口：64 样本普通 95th quantile 的新点覆盖
实际约 93.85%；flow-off 噪声尺度仍按目标 clean RMS 条件化；oracle coverage 借用了 estimated
gate；clean target 使用同一个 voxel `A`，不是独立 renderer；scratch/formal CLI 和传递依赖哈希
也没有完整写入产物。

所以 N1.2 的顺序已经冻结为：session-level calibration -> finite-sample conformal 第 62 个次序
统计量 -> candidate-specific audit -> global/per-camera/lower 三门 -> raw/classical 双参考 ->
model-mismatch floor -> matrix-free multi-shift Lanczos。经典 IID/structured GLS、whitened CGLS、
Huber/Student-t 全部过门后，才允许学习 bounded lambda 或 robust weight。

完整复盘见 [N1.1 flow-off covariance proximal NO-GO](jacru_n1_1_flowoff_covariance_proximal_no_go_2026-07-18.md)。

## 88. N1.2：把尺子校准了，仍然没有可安全放行的候选

N1.2 先修协议，不急着造模型：同一 session 的 flow-off frames 不再假装成独立实验；64 个
calibration score 的 95% 门改用第 62 个次序统计量；global、per-camera 和 lower gate 分开；
strongest classical 与 raw network center 同时保护；sensor covariance 与 forward mismatch 分账。

post-audit pilot 覆盖 3 个 session、5 个 case、8 个候选和 80 条 metric rows，所有 checksum
通过，但 16 个 candidate-method decision、dense ceiling 和 evaluator-only oracle ceiling 的通过数
都是 0。五个 case 的 voxel-versus-continuous mismatch 已有 `15.73%–27.79%`，且明确不属于
sensor noise，也不能让部署 selector 读取。

**讲人话：**以前的问题不只是算法跑不好，尺子的刻度也混了。现在尺子分清了“相机噪声”和
“物理模型不准”，结果仍然告诉我们不能放行。这不是白做，而是阻止后面用错误噪声模型包装成功。

严格复盘见 [N1.2 post-audit protocol NO-GO](jacru_n1_2_postaudit_protocol_no_go_2026-07-18.md)。

## 89. N1.3：真正的 Huber 数据项只有约 0.85% 独立贡献

N1.3 实现了 measurement-domain Huber-PDHG，并完整展开
`mean x whitening x quadratic/Huber x spatial lambda`。6 个 session、128 个候选、3,072 条 metric
和 192 条 direct contrast 最终 0 pass。

平均最强的 diagonal candidate 有 `+16.91%` field gain，却伴随 `8.33%` harm、`-50.91%`
worst 和 `1.656x/3.432x` clean residual。更重要的是，在完全相同 mean、whitening、lambda 下，
Huber 相对 quadratic 最多只贡献 `+0.852%` nominal 和 `+0.849%` outlier field gain；加入 2%、
8 sigma sparse outliers 后没有额外 dose response。

**讲人话：**Huber 确实有一点用，但不是救命药。漂亮平均值主要来自“怎样减均值、怎样白化、
怎样平滑”的组合，而同一个薄界面仍可能被严重伤害，所以现在训练网络去自动挑 Huber 参数只会
把一个不成立的底座变黑箱。

严格复盘见 [N1.3 robust-data factorial NO-GO](jacru_n1_3_robust_data_whitening_factorial_no_go_2026-07-18.md)。

## 90. N1.4：warm start 能救一个薄界面，却会伤害更多别的场

N1.4 用 CGLS-12 的粗场梯度生成 edge weights，再用 Huber-PDHG-12 细化。审计发现第一版只有
`lambda=0.1` 的 zero-start control，无法拆开 warm start 与 lambda。v1.1 因此给
`0.05/0.1/0.2` 全部补齐 matching zero-start，增加 seed-family 集合和分段调用 fail-closed
检查，再完整重跑 33 个候选、792 行结果。

最佳平均值是 zero-start `lambda=0.2`：field `+28.81%`、H1 `+21.85%`，但已知
`2113/single-interface` 仍是 `-15.01%`，clean worst `3.00x`。uniform warm 把这个特例改善
约 `10.66%`，却让全体同 lambda 平均 field 相对 zero-start 恶化 `19.17%`。27 个 adaptive
edge 候选又全部输给 matching uniform；最好一组仍平均落后 `0.944%`。

**讲人话：**warm start 像偏科补习，确实救回一道一直错的薄界面题，却让更多普通题失分。
adaptive edge 也没有证明自己比普通均匀正则好。问题更像“观测模型把不同物理形态解释错了”，
而不是“边缘平滑力度没调好”。

严格复盘见 [N1.4 adaptive-edge warm NO-GO](jacru_n1_4_adaptive_edge_warm_robust_no_go_2026-07-18.md)。

## 91. N1.5：下一算法改学 forward mismatch，不直接猜三维场

新候选把便宜模型记作 `G_L`，把包含 finite aperture、必要时 curved rays 和 calibration
perturbation 的高保真模型记作 `G_H`，专门学习或估计：

```text
epsilon(x,z) = G_H(x,z) - G_L(x,z)
```

第一步只做条件均值；第二步做 fixed low-rank covariance；前两步在 locked development 有
headroom 后，第三步才允许小网络根据 f-number、view、pixel、geometry uncertainty 等部署可见量
预测低秩系数。网络不输出三维场，也不能接收 test truth、family label 或 audit-camera residual。

这条路线的物理依据比继续调正则更直接：NeRIF 明确处理 voxel discretization 与连续表示；
cone-ray BOS 已证明有限孔径会让 thin-ray reconstruction 随 f-number 失稳；Bayesian
approximation-error 文献则给出 accurate/coarse forward pairs 的统计补偿方法。

**讲人话：**如果地图本身画错了，再聪明的导航也会走偏。N1.5 先学习“便宜地图和真实道路差
在哪里”，再让经典重建或 NeRIF 使用这份误差说明书。它仍可能失败，但失败会回答一个真实光学
问题，也更贴近师兄能审核和实验室能验证的方向。

算法、泄漏红线、十个师兄问题和一级来源见
[N1.5 conditional approximation-error protocol](jacru_n1_5_conditional_approximation_error_protocol_2026-07-18.md)。

## 92. N1.5-A：前向误差预测得更准，不等于三维重建更准

第一轮把目标定成连续 renderer 与体素 FD/三线性算子之间的 normalized mismatch。fit/calibration/
development 按 12/4/6 个 geometry seed 分开；两种 phantom family 共用同一 geometry，因此没有把
ray 或 field 行数伪装成独立样本数。

最简单的 component damping 已把 mismatch L2 改善 `38.62%`。加入观测局部曲率、相机姿态和
CGLS-12 暖启动残差后，ridge 在 opened development 平均改善 `45.62%`，相对 damping 再好
`11.68%`；但 12 个场中有 2 个变差，触发 NO-GO。PCA-16 exact-coefficient oracle 的残余比
只有 `0.3343`，说明失配有低秩表示空间，却没有证明这些系数能由部署可见量安全推断。

**讲人话：**我们能把“地图哪里画错了”猜得更像，但这份猜测里有些部分根本不会影响导航，
还有些小错会被逆问题放大。因此 measurement residual 不能单独当论文主指标。

## 93. N1.5-B：高阶算子适合当老师，不适合直接接管求解

四阶差分算子通过了约 `3e-16` 的伴随恒等式检查。直接用它做 CGLS-25，opened development 的
field 反而平均恶化 `5.10%`；说明“离散阶数更高”不自动等于逆解更稳。

把四阶算子只用于估计暖启动场上的 `G_HO-G_L`，再让稳定低阶算子做 12 步暖启动细化，则
beta=0.75 在 opened development 得到 field `+4.799%`、H1 `+10.899%`、worst `+1.655%`，
而 component damping 只有 field `+3.721%`。候选只在 calibration 上选 beta，并明确标成
post-open hypothesis，不能当确认成功。

**讲人话：**更敏锐的老师可以指出低阶模型哪里可能错，但让这个老师亲自驾驶反而不稳；当前
最好结构是“高阶负责诊断，低阶负责求解”。

## 94. 冻结确认：所有场都变好，但平均幅度没有过 5% 门

候选、六个 SHA-256 派生的新 geometry seeds、预算和门槛先写入 Git 提交 `67338a0`，再一次性
打开不可覆盖的 confirmation。12 个场全部为正增益：mean field `+3.6323%`、mean H1
`+10.3084%`、worst field `+0.8979%`、worst geometry cluster `+1.3527%`；相对 component
damping 再好 `+0.6864%`。smooth/interface family 分别为 `+2.1703%/+5.0944%`。

唯一失败项是冻结的 field mean `>=5%` 门，所以正式状态为 `SYNTHETIC_CONFIRMATION_NO_GO`。
这条结果稳定，却不够大。以后不能再用这六个种子调 beta。

下一算法改学正规方程真正感受的 `A^T epsilon` 或 measurement-range 分量，并把本轮高阶教师
作为固定强基线。完整数字、物理边界和师兄问题见
[N1.5 confirmation NO-GO](jacru_n1_5_high_order_teacher_confirmation_no_go_2026-07-18.md)。

## 95. N1.6：不是“网络太小”，而是固定地图和导航员一起出了问题

N1.6 按预注册把完整 measurement mismatch 分成两步：先用 fit split 学一个跨几何共享的
PCA basis，再让 ridge 根据 measured observation、camera summary 和 CGLS-12 暖启动状态预测
四个系数。预测结果留在 measurement space，统一经过当前几何的 `A^T`，部署时不读三维真值、
不调用高阶 forward，也不自造一个和 forward 脱节的 adjoint。

唯一一次 opened development 的可部署结果是 field `+3.539%`、H1 `+8.242%`、worst
`+0.167%`。表面上全部场没有超过 1% 的伤害，但它有一半 case 触发 fail-closed，且相对简单
component damping 反而差 `0.184%`，所以 5 项冻结门失败，confirmation 继续关闭。

Oracle 把失败拆得很清楚：

- exact mismatch 能带来 field `+8.616%`，说明物理校正仍有空间；
- rank-4 adjoint oracle 只剩 `+4.985%`，固定共享 basis 丢掉约一半可用幅度，并仍有一个
  相对 damping 的受害 case；
- raw ridge 的伴随残差相对 damping 恶化 `25.357%`，说明系数预测方向也没有迁移；
- fail-closed 把 raw 错误挡住了，但挡错不等于学对。

**讲人话：**我们先画了一张所有相机几何共用的“四条路线地图”，再让一个小导航员选择走哪条。
真实路网会随相机和射线旋转，所以地图本身不够；导航员到了新几何又把方向猜错。继续把 ridge
换成更大的 MLP，最多只是在错误地图上训练更复杂的导航员。

下一步暂名 N1.7 KCRC：不再使用静态 PCA。它从当前 residual、damping 和低阶 `AA^T` 生成每个
geometry 自己的 Krylov basis；先检查这个可部署 basis 的 oracle 上限，再决定是否训练有界
hypernetwork。两次 `AA^T` probe 配合 10 步 refine，仍严格匹配 `25F/24A^T`。训练目标也从
measurement L2 改为穿过有限步 CGLS 后的 field/H1 response。

完整数字、一级来源、师兄问题和复现命令见
[N1.6 adjoint low-rank NO-GO](jacru_n1_6_adjoint_low_rank_no_go_2026-07-18.md)。

## 96. N1.7：换成每个几何自己的四维地图，还是不够

这次没有训练新网络。我们先问一个更便宜也更诚实的问题：如果给每个相机几何现场画一张
自己的四维 correction 地图，它本身有没有足够容量？地图由 damping、warm residual 和两次
带 support 的 `A P A^T` probe 生成；整个候选仍是 25F/24A^T。

主 measurement oracle 得到 field `+4.828%`、H1 `+11.076%`，所有 geometry 和两类场都为正；
但 field 没到 5%，只保留 exact oracle `56.717%` 的 headroom，support-adjoint gain 也只有
`16.281%`。所以 17 项门过 14 项，仍必须写 NO-GO，并在 learner 之前停止。

**讲人话：**旧方案给所有城市共用一张四路线地图；新方案终于给每个城市单独画图，确实好了一点，
但大部分真实道路仍没画进去。更聪明的导航员无法补回地图里根本不存在的路，所以现在不该训练
DeepONet/FNO/MLP 去猜四个系数。

还有一个重要细节：12/12 个系数都撞到预先冻结的安全半径。也就是说，失败可能来自“地图只有
四条路”，也可能来自“规定最多只能走这么远”。我们不能看到结果后放宽半径并改判；下一步只能
把 unbounded span 和 bounded span 分开做只读诊断，再用新数据预注册 camera-block 表示。

finite-K 真值搜索找到了 `+5.560%`，但它额外用了 33,780F/33,780A^T，且 36 个起点只有 5 个
在预算内收敛。这个数字告诉我们“或许还有 solver-aware 方向”，不代表算法已经会自己找到它。

完整账本见 [N1.7 geometry-Krylov NO-GO](jacru_n1_7_geometry_krylov_no_go_2026-07-18.md)。

## 97. N1.7-D：把安全绳放长四倍，地图仍没有完全画对

独立审计提醒我，N1.7 的 12/12 个系数都撞到安全半径，直接说“四维地图不行”会过头。因此我把
半径三个可见系数统一放大四倍，并给 Powell 更多收敛预算。这个实验是在看过结果后做的，只能
解释原因，不能改判或当新算法成功。

放宽后，measurement projection 的 12 个 case 都不再触边：field 从 `+4.828%` 升到
`+5.556%`，说明原来的安全边界确实压住了收益；但它仍只保留 exact headroom 的 `65.264%`，
support-adjoint gain 只有 `28.364%`，所以还是没过完整门。

更昂贵的 truth-conditioned finite-K 找到 field `+6.186%`、retention `72.669%`，17 项过 16 项，
只剩 adjoint 门失败。问题是它偷看了真实三维场，并在 development 上额外跑了
`74,010F/74,010A^T`，所以它是“这里可能有路”的探测器，不是会自己找路的算法。

**讲人话：**原来既有“安全绳太短”，也有“地图方向不全”。把绳子放长后能走得更远，但四条路
仍没有同时对准真实物理误差和最终重建目标。N1.8 不该直接训练这四个系数，而要先把相机编号、
射线坐标和每个几何自己的 Krylov 方向组合成新地图，再去新数据上检验。

完整审计见 [N1.7-D 四倍半径敏感性](jacru_n1_7_radius_sensitivity_audit_2026-07-18.md)。

## 98. N1.8：相机分块几乎过了重建门，但它可能画的是“捷径”而不是物理误差

这次先把五种地图写死，再复用已经看过的 6 个 geometry 做设计筛选。所有地图都花同样的
`25F/24A^T`：Krylov-4、fit-PCA + Krylov、按相机分块、按相机角度做 Fourier 调制，以及按
detector 横纵坐标做一阶调制。没有训练网络，也没有打开新数据。

Camera-Block-6 最好：field `+6.343%`、H1 `+13.203%`，12 个 case 都没有超过 1% 的伤害；
它保留了 exact oracle 总收益的 `74.518%`。但我们运行前已经把更严格的“阻尼之外还能拿回多少”
门设成 60%，它只有 `57.071%`，所以 17 项重建门只过 16 项，不能看完结果再把门改成 57%。

更值得警惕的是，它对 `P A^T` 看到的 forward mismatch 只改善 `9.474%`，远低于 50%。

**讲人话：**按相机把道路分开以后，重建车确实开得更快、更稳；但这张地图可能利用了当前
求解器的捷径，并没有真实画出“光学前向模型错在哪里”。如果现在直接让 DeepONet/FNO 学这六个
系数，可能得到一个 synthetic 上好看的导航员，却无法解释为什么能迁移到真实 BOST。

所以机器状态是 `NO_N1_8_CONFIRMATION_AUTHORIZATION`。这不是毕业设计停止，而是关闭“直接训练
这五种 basis”的分支。下一步先把 Camera-Block 的 field-friendly 方向与 Fit-PCA/Krylov 的
adjoint-friendly 方向组成一个 post-hoc union ceiling，问低秩空间里是否同时存在两种性质；只有
上限存在，才设计 geometry-conditioned、finite-step response-aware basis learner。新 geometry、
fresh、OOD 和真实数据仍不打开。

完整数字与给师兄的问题见
[N1.8 相机/射线混合表示 NO-AUTH](jacru_n1_8_hybrid_design_no_auth_2026-07-18.md)。

补充一次代码审计：原选择器在“17 个重建门全过、但 `P A^T` gain 为负”时仍可能把方法叫作
solver-aware 并授权下一步。本次没有候选全过 17 门，所以结果没有被这个漏洞改变；但未来可能
fail open。修正后负 gain 必须 NO-GO，每个候选必须达到设计 rank，并先核对 N1.7/N1.8 的
case 与 geometry digest 相同。修正版重放的 168 条科学指标逐项不变，机器状态仍是 NO-AUTH。

下一次只比较两个 rank-6 结构：`{d,r,C1r,C2r,Kd,Kr}` 和
`{d,r,C1d,C2d,Kd,Kr}`。它们分别问“按相机拆 residual”与“按相机拆 damping”哪一个贡献了
Camera-Block 的额外收益；如果两个都失败，就关闭这条 rank-6 camera/global-K 分支，而不是继续
枚举更多网络。

## 99. N1.9：界面恢复和观测一致性各赢一边，低秩拼接路线正式关闭

这次严格按上一节只比较两个候选。设计、16 项上游 source hash、17 个重建门、两项本机成本门、
精确 rank 6 和停止规则先提交为 `52490e5`，再运行完整 6 个已打开 geometry、12 个 paired fields。
smoke 子集被代码强制标成 non-decisive，不能提前授权或关闭分支。

Residual-Contrast 的结果是 field `+6.207%`、H1 `+10.672%`、相对 damping field `+2.672%`，
exact retention `72.917%`；但真正衡量“阻尼以外还拿回多少”的 extra-headroom 只有
`51.408% < 60%`，所以只能过 `16/17`。Damping-Contrast 为 field `+5.452%`、H1 `+8.768%`，
exact retention `64.042%`、extra-headroom `36.864%`，过 `15/17`。两者的 support-adjoint gain
分别为 `28.112%` 与 `35.787%`，都没有达到 50% 的 forward-correction 机制线。

逐 case 出现一个很整齐、但只能作为新问题来源的分叉：Residual 在 12/12 个 case 的 H1 更低，
在 6/6 个 single-interface case 的 field 更低；Damping 在 6/6 个 smooth case 略好，并在 8/12
个 case 的 data residual 更低。

**讲人话：**把每台相机的差异放进 residual，比较会保护火焰/密度界面；放进 damping，投影回观测
更像原数据。两张地图各自照顾了一半目标，却都没有同时画对“最后三维场”和“真实前向误差”。
继续在同一批旧题上增加第七、第八条路线，很容易变成看答案调地图。

因此机器状态是 `N1_9_RANK6_CAMERA_GLOBAL_K_BRANCH_CLOSED`。关闭的是这两个预冻结、三相机、
rank-6 synthetic 候选在旧 development 上继续堆 basis/learner；不是宣判所有 camera-aware 或
global-K 方法无效。两项本机 solver-path 成本门虽然通过，但计时排除了 evaluator oracle 系数投影，
每 case 也只测一次，不能写成部署速度优势。Schur 对当前无 covariance/majorizer 的候选不适用，
不能伪填零违反。

下一主线转成 N2：先和师兄确认真实 camera/ray/mask/calibration/held-out reprojection 合同，再按
geometry/session/camera 留出不可回看的 split。新问题是“怎样同时保护界面恢复和 measurement
consistency”，而不是“再换一个 DeepONet/FNO 名字”。没有真实数据合同时先做 adapter、伴随测试、
基线和预注册；固定表示在新 split 上有 headroom 后，才允许训练 generator。

完整证据见 [N1.9 分支关闭报告](jacru_n1_9_global_contrast_branch_closed_2026-07-18.md)，给师兄的
短稿见 [N1.9 审核 brief](jacru_n1_9_advisor_review_brief_2026-07-18.md)。

## 100. N2 第一步：把“等师兄给数据”改成七个机器可检查的门

N1.9 之后不能再在同一批 synthetic case 上换 basis。真正的问题是：实验室的主要误差到底来自
有限孔径、光线弯曲、标定漂移、位移提取还是离散化？它们共享图像、几何、mask、forward 和 split，
但需要的额外对照数据、forward fidelity 与论文终点不同，所以我先没有写新网络。

这次做了一个 JSON 数据合同和 fail-closed 验证器。合同会检查七件事：case/来源/单位/support、观测和
相机几何、线性 A/Aᵀ 或非线性 JVP/VJP、唯一主失配、独立 split、合法论文终点、存储与公开权限。它还直接拒绝
train/audit 重叠、`../` 路径、无许可公开 raw data、拿重投影冒充唯一三维真值，以及 audit 参与选
模型或早停。

当前我们手里还没有 OERF 最小 case，所以空白 intake 的**资料齐备度**就是 `0/7`，状态
`N2_WAITING_FOR_LAB_INPUT`。它不授权预注册、不授权训练、不打开 audit，也不允许写成功。测试里另有
一个纯合同 fixture 能过 7/7，但代码强制把它标为 `CONTRACT_TEST_FIXTURE_VALIDATED_NOT_REAL_DATA`，
不能冒充实验数据。

**讲人话：**以前“师兄给我点数据”像要一箱没有标签的零件，拿到后才发现单位、相机、mask 或权限
不齐。现在先给每个零件贴标签，并把最后一箱 audit 上锁。标签都齐只代表可以开始做实验，不代表
机器已经造好，更不代表论文成功。

独立红队随后发现，第一版门禁虽然报告谨慎，代码仍有能被绕过的地方：非法 schema 没有真正执行、
`NaN` 能骗过数值比较、未授权合同仍返回成功退出码、session split 可能把 audit view 藏进 training，
声明的 f-number 也没有和真实 sensor/condition 绑定。这些都已修正。现在验证器真正执行 JSON Schema
2020-12；逐固定条件读取 flow-off manifest；复算 split digest；在 view/sensor/run/session/condition/
geometry 任一拆分单位上强制 audit 角色一致；并要求真实记录有来源 manifest，synthetic fixture 不能靠
改两个字符串冒充实验数据。专项回归为 `28/28` 通过。

科学红队还把两种容易混淆的证据拆开：同一背景的 flow-off repeats 用来估时间噪声与慢漂移，多个
独立背景才用来识别 pattern-dependent bias；PSU 公开数据说明标定状态应完整记录，但没有直接证明
calibration drift。网页因此不再把资料缺失画成红色 `FAIL`，而用中性的“待实验室提供/待确认”。

最重要的新决策是只让师兄先选一个 primary mismatch：若有多 f-number 和 cone forward，做有限孔径；
若有多次标定/session，做标定漂移；若只有 raw image pairs，先做位移不确定度；若只有处理后位移和
单一真实场，就先交 loader、adjoint 与强基线，继续关闭算法主张。

完整合同见 [N2 真实物理失配与数据合同](oerf_n2_physical_mismatch_data_contract_2026-07-18.md)，一页
提问稿见 [N2 师兄确认单](oerf_n2_advisor_intake_brief_2026-07-18.md)。

## 101. 公开 PSU 是接口考场，不是有限孔径算法成绩单

这次没有重跑 v5y/v6a，也没有训练新网络。我把 PSU 70-view 开放 BOST 的论文、压缩包清单、
rotation-40 观测、几何审计、九视角 B0 `A/A^T` 和永久留出协议逐字段塞进 N2 rehearsal。
机器只允许“公开支持、公开负证据、需本地核验、缺失、禁止推断”五种标签；遇到不知道的字段不会
用默认值补齐。

结果有 16 个字段组：6 个公开支持、2 个公开负证据、3 个需要本地绑定、2 个缺失、3 个禁止推断。
七个正式 N2 门仍全部为 false，所有训练、audit、成功和 raw-data 授权也是 false。B0 operator 自己的
接口审计确实过了：CPU64 最大点积误差低于 `5e-16`、MPS32 低于 `1e-7`；但“一个 operator 会跑”
不等于“每个真实 view、condition 和 calibration 已经绑定成 N2 数据记录”。

一级来源给了一个很重要的矛盾。论文说每次试验采集了 2000 张 flow-off 和 2000 张 flow-on；但当前
公开压缩包 inventory 只看到每个 camera-rotation 的平均产物或复合容器，没有可逐帧核验的独立时间
重复。所以“实验中采过 2000 张”不能写成“我们当前拥有 2000 个 repeats”，70 个旋转视角更不能
拿来替代时间重复。

论文里的 `f/22` 与 `f/32` 也不是干净孔径对照：85、105、200 mm 镜头、相机位置和 optical channel
同时变化。它能提醒我们 finite aperture 重要，却不能证明 residual 差异就是 aperture 单独造成的。
要做师兄方向的真实孔径论文，仍要同一光路、同一 geometry 下只改 f-number 或 focus。

**讲人话：**公开 PSU 可以检查零件能不能装上、齿轮会不会转；但它没有给我们一台只换孔径、其余
都不变的对照机器，也没有独立三维尺子。因此不能拿接口通过当“新算法恢复了真实流场”。

这次还找到一个可继续深挖、但尚未授权训练的真实成本问题：论文的 cone-ray data operator 报告
`8.5%` coefficient of variation，需要约 `8000` points per pixel。下一候选不再用网络直接替掉
operator，而考虑“可解析低阶 control variate + 独立高保真 residual correction”：learner 只分配
样本或预测 control-variate 系数，最终 estimator 保持无偏并保留误差条。这样与已失败的 v6a 容量
升级不是同一实验，但仍必须先预注册 fresh geometry、逐 rig tail、`A/A^T` 和端到端成本门。

完整字段表、师兄材料清单和复现命令见
[PSU 到 N2 的接口演习](psu_n2_public_rehearsal_2026-07-18.md)。

## 102. 第一个孔径控制变量没有过关，但它把下一步照亮了

N2 的第一条小候选不是大网络，而是一个容易审计的二折二次控制变量。它想做的事情很直观：
先用便宜的二次曲面近似“孔径里不同子射线的贡献”，再只对近似没解释掉的残差做高保真采样。
为了不找一个太弱的对手自我安慰，预注册同时放进 IID、反向配对、scrambled Sobol、sunflower QMC
和确定性 disk quadrature，并按相同高保真子射线数比较。

程序正常跑完，但最先失败的是“尺子”。原先的 576 点和 1024 点参考在大孔径审计工况上还差
`0.4101%`，超过预设 `0.3%`，所以机器按规则给 `HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED`。
后面的性能仍可帮助决定研究方向：每像素 32 条高保真子射线时，候选 pooled RMSE 是
`0.0498241`，scrambled Sobol 是 `0.0229810`，候选反而高 `116.805%`。这不能改写成正式失败，
更不能写成成功；它只说明当前 N0 不值得马上换成更大网络。

我随后把参考阶数单独冻结为 1024、1600、2304、4096 点再跑。两个普通工况的 2304→4096 差异
降到 `0.04869% / 0.05980%`，但大孔径和穿越边界仍为 `0.12339% / 0.11944%`，略高于事先写下的
`0.1%` 描述线。因此原 HOLD 不变，也不重评分候选。

**讲人话：**这不是电脑卡住。像用尺子量头发丝，普通位置已经比较稳，大孔径和火焰前沿附近还会
随着尺子刻度变化。我们不能拿一把没完全校准的尺子宣布谁赢；但当前候选已经比强低差异基线差很多，
继续给它堆网络也没有科学理由。

文献红队又发现更重要的边界：StackMC、Regression-based Monte Carlo、Primary-Space Adaptive
Control Variates、Neural Control Variates 都已经覆盖“拟合一个可积分近似，再校正残差”的统计骨架。
所以 N0 冻结为失败基线，不能包装成“首次神经孔径控制变量”。

真正贴近何远哲方向的下一条路来自 NeRIF 自己。NeRIF 同时输出折射率 `n(x)` 和直接梯度 `g(x)`，
并用 `AD(n)` 检查两者一致；每条 ray 会随机取 60–200 个路径点。我们可以把“直接梯度 + straight
ray + 稀疏路径点”当低保真，把“AD/数值梯度 + 密路径点，进一步加 curved ray/finite aperture”
当高保真，只在少量同随机状态样本上计算两者残差。目标不是又造一个普通控制变量，而是研究
`pupil × pixel footprint × path` 联合积分、forward/JVP/VJP 一致性和遇到火焰前沿时自动回退。

下一轮先写新的机制合同，不直接训练：

1. 证明或数值审计多层估计器没有偷偷引入 bias；
2. forward、JVP、VJP 使用同一随机状态，并用独立实现做点积/有限差分；
3. 与 QMC、RegMC/StackMC、Primary-Space ACV、NCV 和高阶 cone reference 同层比较；
4. 同时报告积分误差、三维 field/H1/front、held-out reprojection 和完整调用成本；
5. 大孔径、boundary crossing、curved ray 任一尾部失控就回退，不让平均值掩盖失败。

完整数字、先行工作碰撞和下一候选公式见
[N2-CVCR-N0 事后参考与研究转向](n2_cvcr_n0_postopen_reference_and_pivot_2026-07-18.md)。

## 103. 自动梯度加离散梯度不是新算法，但两级残差机制值得进入盲审计设计

上一节提出“直接梯度/自动梯度做高低保真”后，我先查到了一个会改变选题边界的 2026 年论文：
*Neural Refractive Index Primitives for Flame Field Reconstruction Using Background-Oriented Schlieren*。
它已经用单一折射率 primitive 比较 automatic、central-discrete 和 hybrid gradient，并加入 smoothstep
hash、3D mask 与 occupancy/hierarchical path sampling。所以“我把自动梯度和离散梯度组合起来”
不能再当创新点。

但这里仍有一个很具体的成本问题：automatic gradient 需要一次场查询和一次坐标 VJP；三维中心差分
需要六次场查询。高分辨率、有限孔径、多路径点训练时，这个差别会反复出现。于是我写了一个完全独立
的 clean-room 小模型，不复制 2026 作者仓库：用 smoothstep 三维网格模拟可二阶求导的 refractive-index
primitive，低保真走 automatic gradient + straight path，高保真走 central difference + 规定的 high
path，再用

`mean(low_B) + mean(high_D - low_D)`

估计高保真均值。B 和 D 独立有放回抽样，只有 residual 里面的 high/low 共用同一个 pupil/path state。

四个开发场景都出现了描述性 matched-cost 收益：约 `1.36x-1.78x`。更关键的是 residual/high 方差
只有 `0.0042-0.0266`，说明两条路线在这个小模型里高度相关。固定状态 JVP 相对误差在
`7.3e-10-5.0e-9`，VJP dot 误差不超过 `2.0e-15`。这说明程序里的导数合同是自洽的。

但是机器仍然只给 `DEVELOPMENT_ONLY_NO_AUDIT_AUTHORIZATION`。原因不是保守过头，而是两个
wrinkled 场的半量到全量参考敏感度为 `0.122%` 和 `0.779%`，超过预设 `0.1%`。四个场只有两个
通过。smooth+bend 的保守计时收益也最薄，大约只有 `1.10x`，不能宣传成普遍的大幅加速。

**讲人话：**我们发现“便宜路线先算大部分、贵路线只修正差别”在平滑题目上像是有用的做法；但
火焰皱褶附近，连用来评分的参考平均值还会随着采样数明显变化。现在最多能说“值得出一张更严格的
新试卷”，不能说“新算法已经赢了”。

这次还修了两个很容易写错进论文的地方。

第一，原整数预算分配只是把连续最优比例取整，审计代理找到一个明确反例。我把它改成严格枚举预算
边界，测试里永久保留这个反例。第二，forward estimator 无偏不代表平方损失无偏。随机 forward
直接平方会多出 covariance，训练梯度也一般有偏。下一版必须用两个独立完整 estimator 的对称
cross-product gradient，或显式扣除方差，不能看到 forward identity 就写“unbiased training”。

现在最值得继续的创新点已经被压得很窄：不是 automatic/discrete，不是普通 multi-fidelity，也不是
再起一个网络名，而是

1. 对 `pupil x pixel footprint/PSF x path` 联合光学测度做多层或 multi-index 分配；
2. forward、JVP、VJP 完全复用并记录随机状态；
3. field-dependent ray ODE 的 VJP 包含 trajectory sensitivity；
4. mask、frustum、support 或 flame-front crossing 时 fail closed；
5. 最后在 unseen 形态、三维 field/H1/front、held-out reprojection 和端到端成本上过门。

我现在的学习顺序也变清楚了：先读 2026 论文 2.2.1-2.2.3，弄懂 smoothstep、automatic/discrete
gradient 和 loss；再手算两级方差与成本分配；接着读 JVP/VJP 和平方损失双样本；最后才进入 ray ODE、
伴随轨迹灵敏度和联合 pupil/path multi-index。暂时不要把精力花在加 MLP 层数或直接跑 FNO 上。

开发协议见 [N2-ADRC-N1 development protocol](n2_adrc_n1_development_protocol_2026-07-18.md)，
一级来源与禁止主张见
[Neural refractive primitive source audit](n2_neural_refractive_primitive_source_audit_2026-07-18.md)。

## 104. 曲光线导数写对了，但基础尺度下它可能小到没有实验意义

上一节最关键的缺口是 `bend` 由配置写死，路径并不会随着折射率场改变。我这次按
`d(n d)/ds = grad(n)` 自己写了一个 RK4 曲光线内核，把每一步的位置和方向留在计算图里；再用
完全相同的名义路径做一个 detach 对照。两者数值输出相同，导数不同，所以可以单独量出
trajectory sensitivity，而不把它混在普通的场梯度里。

最初用 16 和 32 步检查时，三个开发 rig 全部失败，参考差异还有 `1.6%-7.3%`。我没有把 `1%`
门槛放宽，而是把主计算提高到 64 和 128 步。之后三个开发 rig 都过了数值合法性门：参考差异降到
`0.26%-0.32%`，出口方向与曲率积分差低于 `0.057%`，端点动量守恒差低于 `0.215%`；完整轨迹
JVP 对有限差分的误差约 `4e-9-1.1e-8`，VJP 点积误差低于 `8.6e-15`。

但最值得重视的不是“3/3 通过”，而是基础折射尺度下 trajectory JVP 只占完整 JVP 的
`0.021%-0.083%`。这可能比真实相机的光流噪声、标定漂移还小。此时两级估计器算出上百倍的理论
效率上限，并不表示算法伟大，只表示便宜直线模型和昂贵曲线模型几乎完全相同。

于是我把同一开发场做了 `1/3/10/30/100x` 的无量纲压力测试。到 30x，两个 rig 首次出现超过
`1%` 的轨迹导数份额或低高模型失配；到 100x，三个 rig 都越出预设视锥，低高输出差达到
`1.87%-4.42%`。这给出了一个比“再加网络层”更真实的问题：能否在轨迹效应开始重要、但光线还
没有改变拓扑或离开标定范围的窄区间里，自动决定哪些 ray 必须算高保真？

文献红队同时提醒：曲光线、曲光线伴随、有限孔径 cone ray、两级残差、神经控制变量和一般的
不连续处理都已有先例，不能单独包装成创新。现在唯一值得继续证伪的窄假说是“BOST 拓扑安全证书
+ 随机高保真纠偏”：安全 ray 以概率 `pi` 算 high，并用 `L + I/pi*(H-L)` 保持条件无偏；不安全
ray 强制 `pi=1`。如果 `pi` 随网络参数变化，训练梯度还要额外处理，不能直接穿过离散路由反传。

**讲人话：**我现在能证明的是曲光线程序和导数尺子基本对得上，不能证明实验里值得用它。下一步
最重要的不是训练，而是向师兄拿到米制 ROI、波长/气体组成、相机/背景标定、最大偏折工况和重复
图像噪声。只有真实轨迹修正大于测量不确定度，这条论文线才值得继续。

完整排练见 [场依赖曲光线排练](n2_adrc_n1_curved_ray_rehearsal_2026-07-18.md)，下一假说及已有工作
边界见 [拓扑认证随机保真路由](n2_topology_certified_routing_hypothesis_2026-07-18.md)。

## 105. 三级路线找到了机制余量，但第一版路由和实现都应该判失败

上一节的随机公式看起来很漂亮，但真正接到程序以后发现，原来的 `H-L` 混了两件不同的事：
自动梯度和中心差分不一样，直线和曲线也不一样。只用“曲率有多大”去预测这个混合差别，就像只看
路有多弯来猜汽车油耗，却把发动机型号差异也混在里面，当然不稳定。

我把路线拆成三级：`L0=直线+自动梯度`，`M=直线+中心差分`，`H=曲线+中心差分`。这样 `H-M`
才主要表示轨迹修正。基础尺度下，`H-M` 的方差只剩原混合残差的 `0.7%-9.2%`，这个机制分解是真的
有用；但到 3x 强度，三组里只有一组仍低于 10%，到 10x 已经占 `42%-90%`，同时视锥证书把所有
ray 强制回 high。也就是说它只可能在一个有限的弱到中等折射窗口里工作。

统计代理又抓出一个公式错误：第一版分配概率的方法并不满足真正的 KKT 最优解。我已经改成
`pi=clip(c*risk, pi_min, 1)`，并加了一个能明确区分错误算法和正确算法的反例测试。程序现在有两条
路径：完整 high replay 只用来核对无偏和精确方差；在线 sparse executor 只计算 Bernoulli 选中的
high ray，二者在同一个 mask 下相对误差为零。

完整实验没有给好消息。当前风险分数相对 constant-pi 路由的精确方差比是 `0.97-1.10`，没有一组
达到预设 `0.90` 门；oracle 用真实残差分配时有 4/9 个工况达到门，说明“更好的可观测残差预测器”
可能有空间，但现在这一个没有。更严重的是，虽然场查询原语合同是 full high 的 `0.621875`，证书
没有复用直线渲染结果，而且有很多 Python 循环，Mac 实测端到端反而是 full high 的 `2.48-2.51x`。
所以机器诊断是 `ORACLE_HEADROOM_CURRENT_PROXY_AND_IMPLEMENTATION_NO_GO`，不是成功。

导数部分是通过的：冻结路由 JVP 对有限差分约 `3e-9-2e-8`，VJP dot 低于 `1e-14`；两个独立
Bernoulli 副本构造的平方损失和方向导数相对误差低于 `9e-6`。这只证明随机估计器的数学接线正确，
不能抵消风险代理和运行成本的失败。

**讲人话：**现在我们有一套不会自欺的试验台，也知道“直线中心差分”是一个不错的中间层；但是
决定哪些 ray 少算 high 的评分器很差，检查安全的程序还比直接全算更慢。下一轮最实际的两件事是：
先复用 `M` 已经计算的路径数据并把证书向量化；再写一个沿直线路径求解的小型轨迹变分方程，直接
预测 `H-M` 的方向和大小。两件事都过不了门，就停止这条路。

完整判决见 [N2-PVGR-N0 三级路线开发结果](n2_pvgr_n0_trifidelity_development_2026-07-18.md)，下一版
候选和要问师兄的问题见 [N2-PVGR 后续算法候选](n2_pvgr_next_algorithm_candidates_2026-07-18.md)。

## 106. 变分预测第一次真正超过旧代理，但 7/9 不能写成成功

上一节说要直接预测 `H-M`，这次把它实现了。新程序不是一个大网络，而是沿直线
medium path 求解一个小的变分方程：折射率梯度先把光线推离直线，Hessian 描述
偏离后看到的梯度如何变化，方向项再描述光线已经转向后的几何反馈。最后得到一个有方向的
`H-M` 修正，也可以把它的模当作哪些 ray 更危险的排序。

我先补了一个之前没有的“小残差自身收敛门”。完整 high 输出看起来收敛，不代表比它小几个数量级的
`H-M` 也收敛。64 步的 residual 对 256 步还有约 `6.5%-10.3%` 差异，128 步才降到
`2.0%-2.6%`，所以执行合同改成 128 步，256 步作参考。

变分预测器在三个开发 rig 与 `1/3/10x` 应力上给出了很强的 matched 信号：

- 对 `H128-M128` 的 relative-L2 是 `4.64%-6.85%`；
- 逐 ray risk 的 Spearman 是 `0.926-0.996`；
- 修正后的 residual 方差只剩原来的 `0.30%-2.26%`；
- 把 shared medium、Hessian predictor 和 correction 全部算进去，candidate p90 / full-high p10 是
  `0.0875-0.0882`。

这些数字已经明显好于上一节的曲率标量代理。但程序中间也暴露了一个很值钱的错误：
第一版参考指标误用了“两个 residual 之间的 relative-L2”，导致机器给 `0/9`。我没有人工改结果，
而是加了一个专门单测，把指标改成真正的剩余范数比后重跑。

修正后仍然只有 `7/9`。失败的是 wrinkled-wide 的 `3x` 和 `10x`：candidate 对 `H256`
的误差分别是 full `H128` 对 `H256` 误差的 `1.143x` 和 `1.774x`，超过冻结的 `1.10`
no-harm 门。原因是 `H128-M128` 里 high 和 medium 的积分误差会部分抵消，但
`M128+prediction128` 对 `H256` 的 mixed closure 不保证有同样的抵消。

**讲人话：**我们现在有一个会在“同一把尺”下很准地预测曲光线修正的小模型，而且它比全程
追踪光线便宜很多。但把它放到更细的参考尺子上时，两个高应力皱褶场会恶化。所以现在可以
说“终于找到一个强候选”，不能说“算法已成功”。

下一步不是加大 MLP，而是三件事：直接线性化离散 RK4 step（包括方向归一化），与 Norton/Picard
一次和两次更新对比，再为高应力皱褶场做不看 truth 的 fail-closed 回退。当前 forward 速度也不能替代
JVP/VJP 和三维重建成本门。

入门学习见 [N1 变分缺陷预测学习指南](n2_pvgr_n1_variational_learning_guide_2026-07-18.md)，数学合同、九行数据、
失败门与先行工作边界见
[N0.1/N1 共享状态与变分预测冻结协议](n2_pvgr_n0_1_shared_state_and_variational_protocol_2026-07-18.md)。

## 107. 精确离散 JVP 修掉了 7/9，但 Picard 又把我们打醒了

上一节留下了两个失败：皱褶宽孔径场的 `3x` 和 `10x` 在更细参考解下变差。最开始很容易把
原因归咎于“应力太强”或者“还缺一个更大的网络”。这次往下查了一层，发现首先该修的是我们自己
对一阶导数的定义。

旧 N1 把 `A delta r + B delta d` 放进轨迹切线方程。它可以理解为沿直线路径对完整动力学做一次
仿射修正，但它不是弯曲同伦 `d'=epsilon F` 在 `epsilon=0` 的精确导数。因为对
`epsilon F` 求导时，`epsilon` 本身已经贡献了 `F0`，而 `F` 随路径变化的反馈还会再乘一个
`epsilon`，属于二阶。精确的一阶轨迹切线只有 `delta d'=F0`；`A/B` 应在最后的观测积分求导时
进入。另一个错误更隐蔽：高保真路线用的是中央差分梯度，所以 Jacobian 也必须对同一个中央差分
程序求导，不能偷偷换成当前位置的 automatic Hessian。

我写了两个互相核对的实现：一个把完整 RK4 程序送进 PyTorch forward-mode JVP，作为很慢但直接的
教师；另一个解析传播同样的离散切线，叫 OCBH。九个开发格里，两者最坏 relative-L2 只有
`2.16e-14`，说明解析程序确实在算同一个离散导数。OCBH 的 matched residual 最坏误差降到
`1.34%`，原来两个 reference no-harm 失败降到 `1.007` 和 `1.064`，九格都过了当前机制门。
其最坏 p90/H128 p10 约为 `0.151`，逻辑场查询比为 `0.4015625`。

但真正重要的结果不是“终于 9/9”。我同时实现了历史上更朴素的 Picard 路径更新，并修掉了第一版
返回旧路径观测的 off-by-one。修正后 Picard-1/2 在同九格上都比 OCBH 更快、更准：

- Picard-1 最坏 matched residual relative-L2 为 `0.171%`，成本比约 `0.0254x`；
- Picard-2 最坏 matched residual relative-L2 为 `0.0498%`，成本比约 `0.0372x`；
- 两者最坏 reference no-harm 约为 `1.001`，也优于 OCBH 的 `1.064`。

**讲人话：**我们把数学公式修对了，也证明 OCBH 是一个精确、便宜、可解释的一阶特征；但在当前
弱合成场里，经典 Picard 更新更简单也更强。所以不能把 OCBH 包装成“自有算法已经胜出”。它更可能
成为风险证书、可微 renderer 的导数骨架，或 `Picard-1 + learned residual` 的输入，而不是最终前向
输出本身。

下一轮会把问题从九个小格扩大到按 field seed 分组的 96 个物理格，避免把同一体场上的很多 ray
误当成独立证据。只有在更强但仍无焦散的场中，`H-P1` 或 `P2-P1` 留下稳定、可学习且超过噪声的
headroom，才值得训练小型算子网络。之后还必须进入三维重建、等 VJP/等墙钟 DeepONet/FNO/FFNO
比较、有限孔径 cone-ray 和 OERF 真实几何。当前没有打开 reserved family，没有真实数据，也没有
论文或泛化授权。

完整推导与九格证据见
[N2 算子一致同伦桥接](n2_pvgr_n2_operator_consistent_bridge_2026-07-18.md)，有限孔径强基线与要向
何远哲师兄索取的 12 项数据合同见
[cone-ray 强基线设计](n2_pvgr_cone_ray_baseline_design_2026-07-18.md)。

## 108. 96 条件跑完了：Picard-1 是更强起点，但现在还不能宣布赢

上一节说要从九格扩到按 field seed 分组的 96 个条件，这轮真正做完了。
开跑之前先把两个场家族、每家族四个 seed、两个视向、两档孔径、三档应力、
256 条共同 Sobol rays、128/256/512 步参考、阈值、图表和停止规则提交到 Git，然后才看结果。
所以独立证据仍只有 8 个 field units，96 个条件是每个场里的重复物理压力测试，
不能写成 96 个独立样本。

第一次运行把 96/96 个格和计时都算完后，在最终汇总遇到了一个
`KeyError`：OCBH 账本用 `logical_scalar_grid_point_queries`，Picard 数据类用
`total_field_point_queries`。两者这里表示同一种“一个坐标上的标量网格求值”，但字段名不同。
我没有直接改 runner 再跑，也没有先打开数字；而是把 96 个 checkpoint 当作 opaque bytes 做
Merkle 封存，先提交只允许这一个字段映射的盲态恢复协议，再解析结果。这个 crash 和恢复
必须保留在将来的稿件里，不能为了好看删掉。

总判决是 `GROUPED_FACTORIAL_FAIL_NO_FORWARD_AUTHORIZATION`，原因很具体：

- OCBH primary 只过 `73/96`；
- forward-JVP teacher 是 `96/96`，说明它仍然在算对的离散导数；
- H256/H512 sentinel 只过 `80/96`，16 格的 evaluator 不足；
- OCBH 四组 timing 是 `0/4`，p90/H128-p10 为 `0.318-0.390`，高于 0.25 门；
- query 门为 `96/96`，所以问题不是账本丢失，而是精度、参考和实测成本。

Picard-1 给了强信号：8/8 field units 的 12-condition 几何平均 matched error 都比 OCBH 低，
grouped ratio 为 `0.198 [0.151, 0.264]`；最坏墙钟只是 OCBH 的 `0.315`，logical query 为
`0.996`。但它仍然不能说赢：六个 absolute-reference 失败都与 wrinkled-3163/orientation-22
的 evaluator 失败重合；另外在一个 sentinel 已过的条件里，Picard-1 的 Q95 比 OCBH 差 `1.819%`，
超过预注册的 1% 尾部门。

**讲人话：**当前不该再花时间证明 OCBH 是最佳 forward。它降级为离散机制 teacher，Picard-1
变成三维重建的第一强物理基线。但在训练网络之前，要先用 H1024 把 16 个参考失败格审清，
再做同一 curved operator 的 field JVP/VJP dot/FD 门和 6-train/2-held-out 八视角重建。只有
`H-P1` 稳定高于数值误差与师兄数据的实验噪声底，才训练小型 residual operator。

完整数字、失败格、盲态恢复和下一步见
[N3 96 条件结果审计](n2_pvgr_n3_grouped_factorial_result_audit_2026-07-18.md)；可微三维接口的入口见
[field JVP/VJP 到重建的最小设计](n2_pvgr_field_jvp_vjp_reconstruction_interface_design_2026-07-18.md)。

## 109. H1024/H2048 把问题缩到两个小残差格：先别训练网络

N3 留下 16 个 reference sentinel 失败格。这轮没有把 96 格全重跑，而是为每个失败格配一个
同 field seed、同 stress、只改变一个 geometry factor 的 matched control，共 32 格。先冻结
H256/H512/H1024、收缩率、finite/domain/topology、查询成本和条件 H2048，再正式运行。

第一版 N4 在第二格需要 H2048 时暴露控制流错误：程序先调用最终 decision 问“是否升级”，最终
decision 又要求 H2048 已存在。我保留 6 个 checkpoint 和堆栈，另开 N4.1；它不改任何样本或阈值，
只先算完整 H1024 gates，再决定是否加载 H2048，而且不复用 N4 的 checkpoint。

N4.1 真正算完 32 格以后又在画柱状图时退出：Matplotlib 不接受把整个 counts dict 当 category。
这一次 105 个数值 checkpoint 已经完整。我先对文件路径和字节做 Merkle 封存，再做 artifact recovery；
恢复只把 x 输入改为 key 列表，所有数值 level 都从已封存 checkpoint 读取。两个 validator 最后都通过，
图也做了非空检查。

最终机器判决仍是 `FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED`：

- H1024 全门通过 `23/32`；
- 9 格按规则升级 H2048；
- 7 格升级后通过，最终 reference 为 `30/32`；
- 2 格仍失败，都是 `smooth-s1871 / orientation_58 / narrow` 的 stress 1 和 3 controls。

这两个失败不能简单说成“曲线射线没收敛”。32/32 的完整 detector output、finite、domain、stencil、
direction 和 topology 都通过。两个格的 output H1024-H2048 relative-L2 都约 `6.686e-7`。真正没过的是
matched residual relative-L2：`0.1647%` 和 `0.1392%`，略高于冻结的 `0.125%`。

为什么这么敏感？stress 1 格的 H2048 matched residual norm 只有完整 output 的 `7.37e-5`；
H1024-H2048 residual absolute difference 是 `3.01e-10`，相对完整 output 只有 `1.21e-7`。
也就是说我们在拿两个很接近的完整量相减，再用一个极小残差当分母。wide aperture 对照残差更大，
同一门就能通过。这提示“相消 + 小分母”可能是主因，但目前只是机理推断。

**讲人话：**尺子的大刻度已经稳定，卡住的是两格很小的尾差。不能因为绝对差看起来小就事后改门，
也不该马上训练 FNO 去拟合一个可能低于实验噪声的信号。下一步 N5 先比较 H4096/H8192、共享节点的
direct paired residual quadrature、Richardson 和 compensated summation，再拿何远哲师兄的 flow-off
repeats 把 synthetic units 映射到真实 pixel/noise units。只有 fresh reference gate 清除两格，才开放
tiny field JVP/VJP；神经 residual operator 还在更后面。

完整数字与禁止主张见
[N4.1 评估器收敛结果审计](n2_pvgr_n4_1_evaluator_convergence_result_audit_2026-07-18.md)，下一轮四种
reference 候选与 Go/No-Go 见
[N5 cancellation-aware reference 路线](n2_pvgr_n5_cancellation_aware_reference_plan_2026-07-18.md)。

## 110. 不是“加法算错了”：D1 排除相消假说，D2 在 H8192 找到二阶尾部

N4.1 留下的两个失败格很容易让人产生一个直觉：curved 和 straight 两个完整积分很接近，最后
相减时是不是发生了浮点相消？如果是，换成先逐节点相减、pairwise sum 或 Neumaier compensated
sum，也许不用继续提高 H 就能过门。

这次没有边试边改。我先写了共享节点的 paired-residual 内核，冻结四格、H1024/H2048、五种累加、
toy 物理门、与 N4 route 的等价门和 1%/10% 判决，再做一次性 Git 证明。D1 的结果很干脆：两个
失败格上，最强的非 raw 改动只占真实 H-refinement 差的 `1.27e-9` 和 `5.19e-10`。换句话说，
加法顺序的影响比“能解释 floor”的 1% 门低了约七个数量级。独立 validator 从 `256x2` 数组重算后
仍是 `D1_ACCUMULATION_ORDER_TOO_SMALL_TO_EXPLAIN_N4_FLOOR`。

排除这个机制后，我才另开 D2，结果前冻结 H4096/H8192、final `6.25e-4` 门、`0.5` 收缩门、
1% raw/paired 门和全部几何诊断。四格都过了：最坏 H4096-H8192 relative-L2 是 `1.183e-4`，
最坏收缩比 `0.2199`，观测阶在 `2.19-2.54`。这符合 midpoint 积分进入约二阶尾部；H8192
raw/paired 浮点差最坏只占 final refinement 的 `1.70e-8`。本机完成 5.28 亿次逻辑场查询约用
216 秒，说明这一层 reference 审计不需要 GPU。

**讲人话：**前面卡住的不是“电脑不会把小数加好”，而是 H2048 还没完全走进尾部。现在这四个
已选 synthetic cells 的数值尺子稳了，但这仍不是自有算法胜利，更不是高质量论文结果。它是以后
比较 Picard-1、DeepONet、FNO/FFNO 前必须补齐的一块地基。

下一步先把 N4.1 的 23 个 H1024、7 个 H2048 和 D2 的 2 个 H8192 残差做成 32 格 adaptive
reference pack，并逐数组哈希；然后才做 field JVP/VJP dot/FD 双门和 6+2 view 最小三维重建。
真实 flow-off repeats、observable 单位和 covariance 仍需何远哲师兄提供。在这些门完成前，网络训练
继续锁定。

完整合同、逐格数字、图和禁止主张见
[N5-D1/D2 结果审计](n2_pvgr_n5_d1_d2_result_audit_2026-07-18.md)。

## 111. 32 格参考包组好了，但它诚实地叫“混合包”

D1/D2 结束时，下一步是把 N4.1 的 30 个已授权数组和 D2 的两个 H8192
数组组成一把真正能被代码读取的尺子。这次 D3 没有再跑 forward，而是先冻结
32 格顺序、源文件、哈希、步数和 `23/7/2` 映射，再作一次零 field-query 组装。

最终包是 `32 x 256 x 2` float64，23 格来自 H1024 raw subtraction，7 格来自
H2048 raw subtraction，2 格来自 H8192 paired-Neumaier。整包数组哈希是
`[private digest omitted]`，独立 validator
重建了 105 个 N4 checkpoint 的 Merkle root、每格身份、数组哈希和 5.835 亿 source-query
成本账本，最后判决 `D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY`。

**讲人话：**32 格现在已经装进同一只箱子，并且每件东西都有条码。但箱子里
有 30 件是旧的 raw 算法，两件是 paired-Neumaier；D1 只在四格上验证过两种路由
等价，所以不能假装 32 格都是统一 paired 算法。这不会妨碍下一个小规模导数实验，
但必须在论文边界里说清楚。

下一步不是开始训练 FNO。D4 先把 detector output 和 curved-straight residual 的导数
分开，用同一 tensor forward 做 JVP/VJP dot test、多 `h` 中心有限差分以及
`VJP_residual = VJP_curved - VJP_straight` 结构核对。这一关真通过后，才有资格
进入 6+2 view 三维重建。

完整映射、哈希、成本和禁止主张见
[N5-D3 结果审计](n2_pvgr_n5_d3_result_audit_2026-07-18.md)。

## 112. D4：这次通过的是“梯度发动机”，不是三维重建

D3 把 32 格 reference 装好以后，最容易犯的错误是马上训练 FNO。可真正的下一步应该先确认：
曲光线 forward 对三维场的导数到底能不能信。如果导数图在 RK4 中途断掉，loss 仍可能下降，
但优化方向并不是原来物理 forward 的方向。

这轮先在结果前固定四个小单元、每格四条光线、两种场扰动和七个有限差分步长。四种 map 分开测：
完整曲光线 detector、直光线 detector、raw curved-straight residual，以及 paired-Neumaier residual。
每个 map 都要同时过 JVP/VJP dot identity、三个指定 `h` 与 best-`h` 的有限差分、非退化信号、
重复输出和 ordered topology。任何一格都不允许被平均掉。

正式运行用了 42.997 秒，做了 1,573,152 次逻辑场查询，没有重试。32/32 map、16/16 结构门和
8/8 topology contexts 全部通过。最坏 dot defect 是 `2.845e-11`，低于 `1e-10` 门；最坏 best-`h`
FD 是 `3.062e-8`，低于 `1e-6`；三个强制步长中的最坏值是 `1.485e-7`，低于 `1e-5`。
独立 validator 没有导入 D4 runner 或 gate helper，重新生成输入并重算全部导数后仍判定 valid。

**讲人话：**现在能说“这四个选定 synthetic contexts 里，网格场到 detector 的正反导数基本是同一台
机器”。还不能说“三维能重建”，因为每格只有四条光线；也不能说“NeRIF 已可训练”，因为这里测的是
`field -> detector`，还没测 `MLP parameters -> field -> detector` 的链式导数。四格还共用同一个
`smooth-s1871` 场，所以不能把它写成跨流场泛化。

下一步先做结果前预注册的 D4b 32-cell expansion，再给一个小 decoder 加链式 dot/FD 门。只有这两关
仍稳定，才进入 6-train-view / 2-held-out-view 的 deterministic 三维重建；真实 observable 单位和
flow-off covariance 仍要向何远哲师兄确认。DeepONet、FNO/FFNO 与自有 residual operator 继续锁定。

完整数字、最坏上下文、成本和禁止主张见
[N5-D4 场导数结果审计](n2_pvgr_n5_d4_tiny_field_derivative_result_audit_2026-07-18.md)。

## 113. D4b 没有通过：它帮我们看见了两种不能交给大网络掩盖的问题

D4 在四个 selected cells 上把 grid-field JVP/VJP 跑通以后，这轮按结果前协议扩到 N4/D3 的完整
32-cell 开发总体。32 格组成 16 对、只有 5 个 field units，所以没有把方向、map 或同场 stress
冒充成独立样本。输入、两组新随机方向、cotangent、七个 h、阈值和 12558336-query 账本都在
正式结果前冻结。

最终不是 PASS：256 个 map context 过了 254 个，128 个结构控制全过，64 个 ordered topology
context 只有 58 个稳定。机器判决是 `D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`。独立
validator 没有导入 runner 或 gate helper，重新生成全部输入、导数数组和 960 个拓扑签名后，仍得到
完全相同的数字和判决。12 项结果合同测试也全部通过。

两个 map 失败都在 `wrinkled-s3163 / orientation_22 / wide / stress 1` 的同一个平滑方向，且只影响
raw residual 与 paired residual。它们的 finite difference 很好，required-h 最坏约 `1.27e-7`；
但 dot relative defect 是 `1.84e-10` 和 `1.53e-10`，超过冻结的 `1e-10` 门。absolute defect 只有
约 `1e-19`，这提示小残差 contraction 可能是问题，但我们不能看见结果后改用 absolute gate 把它救回。

六个 topology 失败都发生在最大的 `h=0.01`。插值 cell 与 frustum sign 没变，support bit 变了；
`h<=0.003` 时签名稳定。讲人话就是：最大的场扰动让少量采样点跨过了 hard support 边界，正负两边
不再跑同一个离散程序分支。它不自动等于真实光学不连续，也不允许我们随手加一个 sigmoid。

这次结果把下一步从“给 decoder 接上 autograd”改成两个更扎实的问题：第一，support 的物理语义到底
是视场/光阑/计算域还是 mask，并能否给每个方向计算 local stability radius；第二，residual map 的 dot
失败能否由 pairwise/Neumaier/extended-precision contraction 解释。两项都先做只读 post-open 诊断，
不能改变 D4b 的历史判决。只有新的 topology-certified 合同结果前冻结并通过，才重新考虑 decoder-chain。

**讲人话：**我们没有“差一点就成功”，而是在扩大样本后及时踩住刹车。好消息是问题没有散成一团：
有限差分主体和结构接线很稳，失败集中在 hard support 切换与小残差点积。它们本身可以长成一个更有
研究价值的方向：带拓扑证书和拒答机制的可微 BOST renderer。当前仍没有三维重建、模型训练、真实数据、
泛化或论文授权。

完整逐格数字、机理边界、下一协议和要问师兄的问题见
[N5-D4b 32-cell 场导数普查结果审计](n2_pvgr_n5_d4b_population_field_derivative_result_audit_2026-07-19.md)。

## 114. D4b 失败拆开了：不是求和顺序，support 也不是当前 forward 的 hard mask

D4b 留下两个问号：`p14` 的 residual dot failure 会不会只是最后一次浮点求和不够准；6 个 topology
failure 又到底是哪几个采样点变了。这轮只读已保存数组与冻结输入，没有重跑 forward/JVP/VJP，也没有
改正式判决。

第一个答案是否定的。`torch.sum`、`np.sum`、`np.dot`、`math.fsum`、Neumaier 与精确二进制有理数
contraction 的结果几乎重合。精确值仍为 `1.84168e-10` 与 `1.53431e-10`，高于 `1e-10` 门。真正明显
的是尺度：curved 和 straight 的 dot signal 各约 `1.0866e-5`，相减后 residual 只有 `7.5114e-10`，
缩小 `14,467` 倍；绝对缺陷却仍保留在 `1e-19` 数量级。讲人话就是：不是“加法器算错”，而是两个
大而接近的量相减后，原 relative denominator 变得特别苛刻。以后可以研究 mixed-scale/normwise
伴随证书，但必须在新数据上先冻结规则，不能照着 `p14` 调阈值。

第二个问号也被逐位打开。6 个 context 的 90 个 signature replay 与冻结 hash 全部一致，9 个
`h=0.01` 扰动共翻了 21 位：12 个 `0→1`、9 个 `1→0`；16 位在 RK4 stage，主要集中于入口
step 0/1 和出口附近 step 14，只涉及 ray 0/2。`h<=0.003` 两侧稳定，cell/frustum 一直没变。
更重要的修正是：当前 forward 是连续 smoothstep renderer，support threshold 用于安全/拓扑诊断，
并不是把 field 清零的 active mask。因此上一节“不同离散程序分支”的说法对现有 forward 过强；
更准确的是“协议定义的 support-set signature 改变”。历史 gate 仍照合同 fail-closed，但下一协议应检验
它是不是过度保守。

一个很有用的旁证是：这 6 个 topology-changed context 中 24/24 map gate 都通过，required-h FD
最大只有 `3.77e-7`，远低于 `1e-5`。这不能事后删除 topology gate，却形成了新的可证伪方向：允许
simple、非 grazing 的 support 等值面随场平滑移动，用 transversality/interval-root certificate 区分
“正常边界位移”和“根生成、消失、切触等真实拓扑事件”。若师兄的真实 renderer 有 hard mask、occupancy
或 ray termination，这可能成为可信可微 BOST renderer 的核心算法；若没有，它只应是解释性证书。

完整逐位表、误差分解、候选算法与要问师兄的 8 个问题见
[N5-D4b post-open 失败取证](n2_pvgr_n5_d4b_postopen_forensics_2026-07-19.md)。当前仍没有 decoder、
三维重建、算子训练、真实数据、泛化或论文授权。

## 115. D4c压力测试：新指标能救假失败，但一个 dot test 绝对不够

> **后验语义更正。** 本节记录的是已经冻结的 D4c-v1 历史运行。红队随后发现：v1 没有
> 真正执行 `F(x±hv)`，branch change 是人工标签，structure error 使用隐藏正确矩阵；其
> validator 只证明文件完整和既定逻辑一致。因此本节中的“FD/branch/structure 检出率”与
> `74.72%` pooled classification 全部撤回，不作为算法或论文证据。仍可保留的只有两个
> explicit-matrix 反例：低双线性信号会让 relative-dot 失真，单 tangent 存在 VJP 盲区。
> 修正版见下一节 D4c-v2。

这轮先在 commit `38f091f` 把 seed、24 个 trial、1/2/4/8/16 probes、10 个 gamma threshold、
4 档故障强度和 11 类反例全部固定，然后才正式运行。结果有 3,600 条 base rows 和
36,000 条 threshold/probe evaluations，没有选一个“最好看”的阈值。

第一个反例是正确线性算子，但把 cotangent 故意投影到首个 JVP 的近正交方向。
此时 VJP 完全正确，但旧 relative-dot 门把 24/24 全部拒绝；gamma-scaled normwise score
的最大值只有 `1.40e-4`。这说明小 scalar signal 不等于错梯度。但正确处理也不是直接
改判 PASS，而是标成 `LOW_SIGNAL_UNRESOLVED`，继续查多 probe、FD、structure 和 branch。

第二个反例只改 VJP，并让错误向量与第一个 tangent 正交。一个 probe 对所有强度都是
0 检出。在 threshold `2` 这个只作剖面、不作选参的位置，`1e-10` 错误用 2/4/8/16
probes 的检出是 12/24、20/24、24/24、24/24；`1e-12` 即使 16 probes 也是 0/24。
讲人话就是：多问几个方向能减少盲区，但不能证明 4913 维梯度的每个分量都对。

第三个反例更严格：用同一个错矩阵同时生成 JVP 和 VJP。它们彼此是转置，所以所有
adjoint identity 都可以过，但它们一起偏离真实 forward。本轮只有 FD 能抓它，且当前
`1e-8` 门只稳定抓到 `1e-8/1e-6`，对 `1e-12/1e-10` 没有分辨力。因此任何只报 dot test
的方法都不能单独证明梯度对真实 forward 正确。

还有一个对 BOST 很直接的负结果：如果先用 float64 造出两个很接近的 component matrix，
再做 `C-S`，那么即使后面使用 paired JVP/VJP，`delta=1e-8` 仍是 24/24 被 FD 拒绝。
所以“抗相消”不能只在最后换求和器，必须在同一 ray sample、interpolation query 和投影基上
先形成 curved-straight integrand residual，再累计。

当前不选 gamma threshold，因为预注册网格里最高总体分类率也只有 `74.72%`：clean acceptance
`83.33%`，fault detection `72.57%`。这个数足以证明新指标值得继续，不足以开 fresh
derivative gate。下一步要在全新 BOST field/rig development population 上确定三态规则、多 probe 成本和
residual-native 实现，然后才能冻结 untouched audit。

完整历史输出见
[N5-D4c-v1 开发屏](n2_pvgr_n5_d4c_msra_development_2026-07-19.md)。v1 的独立 validator
只能解释为 integrity/logic `valid=true`，不能解释为 semantic valid。field derivative、decoder、
三维重建、真实数据、泛化和论文授权仍全部为 false。

## 116. D4c-v2：真实调用 forward 以后，哪些结论才站得住

**为什么重做。** v1 最重要的教训是：文件哈希正确、表格行数正确、布尔门也按预期执行，仍然
不等于实验语义正确。如果 FD 没有调用 `F(x±hv)`，branch 是人工标签，structure 又偷看正确
矩阵，那么它只能证明一套自洽的模拟逻辑，不能证明我们想检查的 forward/JVP/VJP。

**这次具体改了什么。** 我先提交 `09a50d1`，冻结 24 trials、720 个 case、3 个 `h`、最多
16 个 tangents、10 个只作描述的 side-weighted thresholds、4 档故障和 11 类场景，然后才运行。
每一条 FD 都保存真实 plus/minus forward 输出、输入哈希、forward 返回的 branch/diagnostic state；
三路径 case 分别调用 curved、straight、direct 的 output/JVP/VJP。最终得到 34,560 组 FD pairs、
1,536 条结构证据和 36,000 条不跨场景混合的状态记录。

**最直观的结果。** 正确 low-signal cases 仍有 24/24 被旧 relative gate 拒绝，但 v2 只把它们
标成 `LOW_SIGNAL_UNRESOLVED`。在描述 threshold 2 下，`1e-10` 首探针盲向 VJP fault 用
1/2/4/8/16 probes 的检出为 0/7/19/22/24；`1e-12` 到 16 probes 仍是 0/24。多问方向能缩小
盲区，但不能证明 4913 维梯度每个方向都对。

**三种门各自负责什么。** 同一个错误矩阵同时生成 JVP/VJP 时，adjoint identity 完全可以通过；
actual FD 在当前 `1e-9` 门下只稳定拒绝 `1e-8/1e-6`，弱两档仍会漏。direct residual path 对
自己做 FD 也可以全部通过，但当它不等于 curved-straight 时，三路径 structure 门会在
`1e-8/1e-6` 两档 48/48 拒绝。也就是说，adjoint、FD、structure 不能互相替代。

**相消机制终于用了真实 FD。** separate arithmetic 在三个 component difference scales 下的
16-probe 最坏 FD error 中位数分别是 `1.58e-3`、`1.46e-5`、`1.46e-7`，全部超过 `1e-9`；
直接形成 residual primitive 的 paired path 三档都约 `1e-11`。这只是在 explicit-matrix toy 上
证明“先算两个大量再相减”会污染中心差分，是否对应真实 BOST 必须拿实验室 renderer 测。

**branch 也不再靠手填。** diagnostic-only case 是 24/24 diagnostic state flip、0/24 branch
flip；piecewise forward 是 24/24 plus/minus branch crossing，并优先判 `FAIL_BRANCH`。这给下一步
一个非常具体的接口要求：实验室 forward 必须返回真正影响控制流的 active state，support/frustum
之类只用于报告的量不能混进去。

**现在能说什么。** v2 修掉了 v1 的三类语义漏洞，并量化了每种门的检测地板。它仍是
synthetic explicit-matrix certificate characterization，不是 BOST、NeRIF、三维重建或算子学习
结果。`PASS_STRONG_SIGNAL` 也只表示有限义务未失败；弱的 injected fault 仍可能拿到这个状态。

**下一步不再堆 toy。** 向师兄要一个匿名最小包：4--16 rays、一个 field/decoder vector、两个
`Jv`、一个 `J^Tq`、curved/straight/direct callable（若存在）、precision、sampling/interpolation/
termination 规则，以及 forward 返回的 branch/diagnostic state。先接 recorder 和 h-sweep，再做
residual-native 对照；真实接口过门后才接 decoder chain 与 6+2 view inverse。

完整公式、逐档表、复现命令和给师兄的七个问题见
[D4c semantic-v2 审计](n2_pvgr_n5_d4c_msra_semantic_v2_2026-07-19.md)。独立 validator 已从
seed 重建所有输入、路径、指标与状态；结果为 `valid=true`。它没有导入 runner 或 certificate
helper，四类篡改测试也全部 fail-closed；但它仍是同一 Python/NumPy 栈，不是跨语言复现，更不
授权真实 BOST、三维重建、泛化或算法优越性。

## 117. D5：终于把“向师兄要一个最小接口”写成了能直接运行的合同

> **后续边界更正。** 本节记录冻结 synthetic 三路径协议。这里的“两个 Jv、一个 Jᵀq”是公开
> 测试使用的 probe 数，不是让师兄交付三份预计算数组；真实后端必须接受任意运行时 `v/q`。
> `1,370` 应理解为同一次确定性协议回放中的 validator 断言，不是 1,370 个独立物理实验。
> “拿到接口后先跑 53 次”也已被 118/119 的分阶段授权取代：L1 static → L2-A → 单独 2 次
> describe → 36/53 primary → 独立 validator + private challenge。

D4c-v2 结束时，网页上一直写“向师兄要 4--16 rays、两个 Jv、一个 J^Tq”。当时想表达测试
probe 数，但这个说法容易被误解成预计算数组，而且还不够执行：师兄给什么文件、每条路径调用几次、branch 从哪里来、单位怎么记、结果由谁判，
都没有机器合同。D5 把这些空白补成 JSON Schema、JSONL adapter、runner 和完全独立的 validator。

这轮先提交 `a8d8849` 冻结规则；公开审计发现结果会留下本机 Python 绝对路径，于是在入库前用
`ee792fd` 增加命令脱敏和 validator 门，再重新正式运行合成参考适配器。每个 bundle 固定 53 个请求：2 次 describe、
42 次 forward、6 次 JVP、3 次 VJP。三条路径 curved、straight、direct-residual 都要独立报告
path/callable identity；forward 自己返回实际 branch state，输出分箱等 diagnostic 另记；每个响应还要
带累计成本账本。runner 只保存原始 output/state/ledger，FD、伴随缺陷、三路径误差和结论由审计端推导。

正式 synthetic bundle 的全部三个 h 都被消费，最大 FD 相对误差 `2.07e-8`，三路径结构误差
`1.20e-15`，伴随 normwise defect `9.28e-17`。独立 validator 使用新 nonce 重启同一 adapter，
没有导入 runner、共享协议 helper 或 adapter，重算后完成 1,370 项 validator 断言。output、branch、metric、
decision 和 stored request 五种篡改即使刷新 manifest 也会失败。

**讲人话：**我们造好了一只标准插头，并用一台透明的合成机器证明插头、计数器和验电笔都能工作。
这不代表实验室的真实机器已经插上，更不代表三维重建或新算法成功。机器判决故意叫
`SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION`，真实 BOST、物理正确、导数证明、重建、优越性、
泛化和论文七类授权仍全是 false。

下一步只需要师兄提供一个匿名小适配器，不必先交整套火焰数据。优先确认真实 residual 是在同一
ray sample/integrand 层形成，还是两张 detector map 最后相减；再确认 hard mask、occupancy pruning、
dynamic sampling 或 termination 是否真的存在。当时写成“拿到接口后先跑固定 53 次”，现已由
L1/L2-A、单独 describe、primary、validator 的分阶段授权取代；真实 failure 仍根据 FAIL_BRANCH、
FAIL_STRUCTURE、FAIL_FD/ADJOINT 或 low-signal unresolved 选择物理上真实的算法问题，仍不直接开 FNO。

完整合同、复现命令、72 小时接线路线和可直接发给师兄的消息见
[N5-D5 最小真实接口桥](n5_d5_minimum_real_interface_bridge_2026-07-19.md)。

## 118. D5-L1：真实代码还没到，但现在不会因为“私有”与“可验证”互相打架

> **进度说明。** 本节的六项 L2 清单中，静态 provenance、物理合同、依赖 inventory、闭世界输出、
> 禁 public summary 与私有 probe 机制已在 119/L2-A 实现；隔离执行与真实运行观察仍未实现。

D5 的合成插头做好以后，我继续往真实实验室接口走了一步，马上撞到一个不能糊弄过去的矛盾：
师兄的 adapter、匿名输入和 raw trace 必须放在 `private_library/`，不能进公开 Git；但已经冻结的
synthetic runner 又要求 config 和 adapter source 都被同一个公开 commit 跟踪，独立 validator 还会
从这个 commit 里重新读取源码。一个文件不可能既“永远不公开”又“必须存在于公开 commit”。

这不是删掉 `.gitignore` 就能解决的。Git ignore 只是防误操作，不是权限系统，而且一旦把实验室代码
写进公开历史，后来删除也不等于没泄露。正确做法是把证据拆成公开协议 provenance、私有实现
provenance 和私有结果 provenance 三层。原来的 synthetic D5 证据保持冻结，不为了迁就真实代码而
悄悄改判据。

这轮先完成 L1 静态预检器。它不会 import adapter，也不会调用一次 renderer，只检查私有文件是否
真的留在私有目录、是否被 Git 忽略且未跟踪、是否有 symlink/hardlink、Schema 与 hash 是否一致、
`.npy` 的 size/dtype/finite 是否正确，以及源码里是否还有 placeholder、明显网络 import、凭据或
绝对路径。12 个反例测试已经全部通过；拿公开 placeholder 去跑时会按预期以退出码 2 拒绝，
`ready=false`、`formal=false`，七类 claim 仍是 0 个打开。

这里最容易误会的是“以后静态绿灯”也只叫
`STATIC_PRIVATE_INTAKE_READY_FORMAL_REPLAY_LOCKED`。讲人话就是：文件在门口的证件和包装暂时没发现
问题，可以让人审源码、准备两次 describe；它不说明里面的 forward 是真实折射光学，更不允许直接跑
36/53 primary 或训练 FNO。正式回放前还缺双 provenance、物理阈值审核、完整依赖 hash、闭世界 manifest、
禁止 public summary 的硬门，以及 validator 临时生成的私有未知 probes。

师兄需要接的最小骨架现在也缩成六个函数：描述、forward、JVP、VJP、标准输入向量和源码审阅说明。
其中 forward 必须把真正改变控制流的 branch 与只用于观察的 diagnostic 分开。如果实验室没有原生
direct residual，就诚实写没有，不能在 wrapper 末端相减两张 map 后冒充 residual-native 算子。

完整目录结构、命令、状态翻译、L2 清单和可直接发给何远哲师兄的消息见
[N5-D5-L1 私有真实适配器接线](n5_d5_private_adapter_handoff_2026-07-19.md)。当前真实 adapter 仍未收到，
因此没有绿色实验室报告，也没有三维重建、算法优越、泛化或论文授权。

## 119. L2-A：把“53 次不是总成本”这件事正式写进了机器合同

这一轮没有训练模型，也没有碰实验室数据，先修正了一个会影响后续所有结论的预算问题。原来网页常把
“53 requests”说成真实接口的一轮验证，但 53 只是一轮三路径 primary：2 describe、42 forward、
6 JVP、3 VJP。真正的独立 validator 还要再执行一次基础协议；为了防固定公开向量查表，还要在
attestation 后生成新的 tangent、cotangent 和秘密 h。

按当前冻结的 2 个私有 tangent、2 个 cotangent、3 个 h 来算，每条路径要多用
`2 JVP + 2 VJP + 2×2×3 forward = 16` 个请求。三路径是 48，所以完整计划是
`2 + 53 + 53 + 48 = 156`。如果师兄没有原生 direct residual，诚实的双路径计划是
`2 + 36 + 36 + 32 = 106`，不能在 wrapper 末端减两张 detector map，假装第三条 residual-native
路径已经存在。

L2-A 现在把这套计算写成了代码，不靠手填总数。它从 config 读取 path、公开 probe 与 h 数量，
从 L2 plan 读取私有 probe 与 h 区间，自动推导 primary、validator 和总预算；任何一项少算都会
`AUTHORIZATION_BUDGET_EXACT` fail closed。当前 frozen L1 只接受三路径，所以计划声明 direct
unavailable 时会明确指向 `BUILD_DUAL_PATH_L1_V2`，而不是诱导绕过合同。

同时新增了两份私有说明：environment lock 绑定依赖版本与 hash；physical contract 绑定参数化、
shape/spacing、axis/units、坐标手性、geometry/calibration hash、波长、sampling/interpolation/
boundary/termination、backend/wire dtype、decoder checkpoint、动态 ray/sample 账本和噪声下限。
物理审阅摘要必须等于这份合同的实际 hash，随便填一个 64 位字符串不能关门。

私有 probe 也不再只换 nonce。系统 CSPRNG 在 attestation 后生成正交 `v/q`，并从三个预注册
数量级区间各抽一个 log-uniform h；启动前只保存 seed+context commitment，不落盘 seed、向量或 h，
adapter 退出后才写 private reveal。这降低固定向量查表风险，但有限随机探针仍不证明整个高维 Jacobian。

L1+L2 targeted suite 现在是 `37 passed`。预算漂移、能力冲突、噪声为零、单位不一致、审阅 hash
伪造、h 区间倒置、secret/绝对路径、extra file、symlink、hardlink 和结果篡改都会拒绝。要诚实强调：
这些是机制测试，不是实验室结果。工具仍然没有 import 或执行 private adapter；isolated describe runner、
OS 无网络、独立成本 observer、签名事件链、dual-path L1-v2 都还没完成。

**我现在学到的关键区别：**hash 正确只说明文件没换，physical contract 齐全只说明解释被固定，
L2 局部通过只说明已覆盖的离散义务没失败。真实 BOST 物理、三维场重建和算子模型是否有效，仍要在
真实几何、标定、rig/session split、field relative-L2、逐 rig tail、Schur violation 与端到端成本上
另行验证。完整推导与下一步见
[N5-D5-L2-A 私有回放基础](n5_d5_l2_private_replay_foundation_2026-07-19.md)。

## 120. L2-B 与双路径 v2：能演练“只问两次”，当前 Mac 仍不准真实执行

119 节结束时，网页里还有两个明显空白：没有原生 direct 时只会提示“去建 dual L1”，真正的双路径
Schema 还不存在；两次 describe 也只有流程图，没有可以拒绝第三次请求、输出洪泛和 token 重放的 runner。
这一轮把这两件事补成了代码，但没有借机执行任何真实 renderer。

双路径没有去改旧三路径 v1。新 `dual-v2` 只接受 curved 和 straight，逐项复用 v1 的 identity、field、
observation、probe、state、tolerance、privacy 和 claims 合同，同时把成本锁成 2 describe、28 forward、
4 JVP、2 VJP，共 36。AST 还会抓直接或先赋值再做的 curved-straight endpoint subtraction，以及
`np.subtract/operator.sub` 和 `direct_residual` callable marker。成本也不再只和 36 这个常量对表，而是从
2 条路径、2 个 tangent、1 个 cotangent 和 3 个 h 独立推导。这样“没有第三条路”不会被末端相减悄悄改写。24 项 dual 测试通过，
其中一项把合法 dual L1 接入 L2-A，机器重算得到 `2+36+36+32=106`；所有 formal authorization 仍为 false。

L2-B 的权限比 primary 小得多。授权文件只能写 `[describe]` 和 request count 2，不能 auto-chain；父进程
发出的两条 request 绑定 authorization hash 与 one-time nonce hash，response 必须逐字等于事前 descriptor，
累计 ledger 只能从 describe 1 走到 2。nonce marker 在已打开账本 inode 内用 `O_EXCL` 消费，所以同一 inode 内复制授权文件也不能再跑一次；这不等于全局账本根已经防同 UID 替换。

第一次独立红队没有客气，指出了四个实质问题：路径 hash 后重开有 TOCTOU；nonce 没参与全局去重；
descriptor 可以塞一个假的 `physical_correctness_authorized=true`；沙箱的读取面和进程树结论写得太强。
修正后，authorization/plan/foundation 用同一 FD 完成 fstat、bounded read 和 hash，L2-A 重算后再比 inode/
bytes；输出目录通过父目录 FD 原子创建并持有 inode；descriptor 改成禁止额外字段的严格 Schema；attestation
明确写 `sandbox_read_surface_closed_world=false`、`process_exec_replacement_denied=false`、
`detached_descendant_absence_globally_proven=false`。

第二次红队继续发现：plan 虽已 snapshot，L2-A 仍按路径再读一次；nonce marker 不是 directory-FD 相对写；
output FD 关闭后又按路径复验；L1 claims 缺键可能被空字典蒙混；更关键的是，允许 `process-exec` 后，恶意
adapter 可以 exec 到另一个同样会输出合法两行 JSON 的程序。前三个竞态现已改成 snapshot bytes、ledger FD 和
同一 output FD 上的闭世界验证，claims 也必须七键完整且逐个严格为 false。

本机 fixture 观察到写文件、TCP、fork/subprocess、宿主环境、继承 FD、sibling temp read 均被拒；hang、
stdout/stderr flood、第三行、forward label、duplicate key、NaN 和深嵌套都会 fail closed。但精确比对 response
无法证明同一 PID 没有 exec 换程序，所以这次没有再靠一句 limitation 放行：默认生产入口会在读取授权、消费
nonce 和创建输出前返回 blocker。测试只能 monkeypatch 内部 capability provider，状态固定为
`L2B_DEVELOPMENT_TEST_DOUBLE_PASS_NO_SCIENCE_AUTHORIZATION`；公开调用签名没有 fixture、bypass、override、unsafe
或 insecure 参数。macOS `sandbox-exec` 已 deprecated，系统读取又使用 broad allow 加前缀排除，因此当前 Mac
没有生产 L2-B pass。

第三次红队又指出：若把 development bypass 暴露成公开参数，复制 runner 就可能绕门；持有 ledger FD 只能保证
当前 inode 内唯一，不能声称全局唯一；同样，输出在持有 FD 上验证也不能证明验证后不受同 UID 外部改写；AST
还可能漏掉 lambda direct marker 和 `from operator import sub as minus`。现在公开 bypass 已移除，生产 capability
必须同时证明 `process_exec_replacement_denied`、`durable_nonce_ledger_root_protected` 和
`output_root_external_mutation_denied`。当时三项均不满足，因此 blocker 分别是
`POST_LAUNCH_EXEC_REPLACEMENT_NOT_DENIED`、`DURABLE_NONCE_LEDGER_ROOT_NOT_PROTECTED` 和
`OUTPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED`；报告诚实写 `global_nonce_uniqueness_proven=false`、
`nonce_uniqueness_scope=CURRENT_OPEN_LEDGER_INODE_ONLY`。AST 的两个漏项也已补测试。

第四次独立审计没有发现 P1，但找出四类 P2：元组解包和字典下标可绕过 endpoint subtraction heuristic；
导入别名和 `setattr` 可绕过 direct marker；私有输入中间目录仍有同 UID 替换窗口；进程内 capability provider
本身不是生产安全边界。前两类已加入 AST 和反例测试，limitations 改成“当前 heuristic 未检出”，不再写“callable
不存在”。生产门则增加 `private_input_root_external_mutation_denied` 与
`backend_capability_attestation_externally_verified`，当前五项能力都不满足；development monkeypatch 只被标为
`PYTEST_MONKEYPATCHED_DEVELOPMENT_TEST_DOUBLE`。

聚合测试现在是 81：旧 L1/L2-A 39、L2-B test double/host gate 18、dual-v2 24。新增测试明确证明 L2-A snapshot 模式不再按路径读取 plan 内容，正式流程在持有的 output FD 上完成闭世界验证，不会关闭后按路径重开；直接运行 CLI 也只返回结构化 host blocker，不读取授权或喷 traceback。这个数字不含一个真实 BOST 物理实验。
真实 adapter、匿名 field、geometry/calibration、动态 ray/sample cost、primary、validator、decoder、三维 inverse、
DeepONet/FNO/FFNO 训练仍全未发生。下一步不是继续堆 toy，而是把 dual/native-direct 问题和轻量 describe
entrypoint 发给师兄审核，同时另行实现并红队验证能禁止 post-launch exec replacement、保护私有输入根/持久账本根/输出根并具备外部 capability attestation 的 backend。即使师兄先给
出私有 callable，当前 Mac 也不创建或消费真实 describe 授权。

完整威胁模型、36/106 推导、测试表、限制和师兄问题见
[L2-B 与 dual-path v2 机制说明](n5_d5_l2b_dual_v2_mechanism_2026-07-19.md)。
## 121. N5-D5-L2-C：把“谁说的”与“是不是真的”分开

这一轮继续沿真实 adapter 接入主线推进，没有训练模型，也没有重复冻结的 L2-B/dual-v2 门。

新增了一个只负责验证、不持有私钥的 L2-C 外部见证器。它要求两个不同 key role 分别签 capability payload 与 event/cost payload；同时把 authorization、plan、foundation、adapter、runner、challenge commitment、trust policy 和 output manifest 都绑定到同一次运行。红队发现若 policy digest 仍由调用者传入，攻击者可换自己的 trust root，因此该参数已删除：公开 verifier 只读固定 registry，而当前 registry 故意没有生产 anchor，会在读取真实 bundle 前 fail closed。

事件顺序被固定成 14 步哈希链。删除、交换或修改事件会失败；subject 与 evidence 摘要还必须从实际文件重算。但页面也明确解释：哈希链只能发现“记录被改”，不能保证观察者没有漏记现实事件。两个不同 key 也不能自动证明两个操作者或进程真正独立。

describe-only 成本被严格限制为两次 describe、零 forward/JVP/VJP，ray/sample/kernel 工作量必须写 `null`。这与未来论文需要的 `A/A^T`、ray/sample、完整 pipeline wall time、失败重试和 rig/session split 成本彻底分开。

当前定向结果：`21 passed`。真实外部签名、真实 adapter、三维重建、模型训练和论文性能结论仍为 0。下一步是独立红队、Linux/实验室宿主能力设计，以及向师兄索取匿名 callable 和成本账本合同。

最终聚合结果是旧 L1/L2-A/L2-B/dual-v2 81 加 L2-C 21，共 `102 passed`；聚焦页面 69 项通过，快速矩阵加入本轮合同后为 `226 passed`。medium 四进程首次暴露 macOS sandbox 进程组清理的并发 `EPERM`，因此矩阵把 18 项 L2-B containment 测试移到串行队列，串行结果全部通过；重新运行后并行层只剩 3 个早已冻结的 N2/D4c 失败，得到 `2211 passed, 3 failed`，另 3 项 MPS 串行通过。不能把 medium 写成全绿。

最终独立红队确认普通数据攻击下没有剩余 P1。仍有两个明确 P2：没有受保护 replay ledger，所以 `one_time_acceptance_proven=false`；两个不同 key 不证明两个操作者或 signer service 独立，所以 `role_operational_independence_proven=false`。这两项在接入生产授权前都必须解决。

## 122. L2-D0：终于能检查“签名前缀里只出现一次”，但还不能说“全局只运行一次”

L2-C 结束时留下两个 P2：没有受保护 replay ledger；两把不同 key 不证明两方真的独立。本轮先做了一个
离线 D0 verifier，把问题推进到机器可拒绝的程度，但没有为了把状态写绿而假装在线服务已经存在。

它从 index 0 重算调用者声明的全前缀。叶子和内部节点使用 RFC 9162 式 `0x00/0x01` 域分离；上一 checkpoint
必须等于 registry-pinned 静态 floor，新 checkpoint 必须由同一前缀重算。这只能说 floor 匹配，不能说 anti-rollback 已证明。

独立安全审计指出，旧版把 `nonce_commitment` 文件的 SHA-256 当成 nonce，改一个序列化就可以换摘要。现在 verifier 必须用 L2-B 原 schema 解析目标 authorization，直接读取 `one_time_nonce`，再计算带固定 domain separation 的语义摘要。前缀中所有 acceptance ID、authorization ID、authorization 摘要和账本自报 nonce 摘要值都要全前缀唯一；即使重复发生在两条非目标历史记录之间也会拒绝。但历史记录没有附各自 authorization/issuer proof，所以只能称“自报摘要值唯一”；不能说它们的 raw nonce 语义全部已验证。

三种角色现在是 sequencer、monitor A 和 monitor B。policy 要求三把不同 key、不同 operator-domain label 和
不同 service identity，三份私有 evidence 文件也从真实 bytes 重算。三类签名现在共同绑定 registry/policy 摘要、log epoch、challenge、subject 和 checkpoint；bundle 有效期必须完全落在 policy 窗口内，实际 acceptance-to-checkpoint 时差必须不大于 checkpoint 自己签署的 MMD，该 MMD 又不得超 policy 上限。但“标签不同”仍不是组织事实证明，
所以报告继续写 `role_operational_independence_proven=false`。同理，调用者前缀不能排除日志对另一个客户端
展示另一条分支；没有在线共享状态、原子 consume 和 gossip，`one_time_acceptance_proven=false` 也继续保留。

31 项测试已通过。新增反例覆盖：伪造 L2-C 成功 `status` 仍只被当作未认证 bytes；同一自报 nonce 摘要在非目标历史记录间重复；同一静态 floor 可分出两个各自通过的分支；bundle 越出 policy 窗口；实际延迟超过 policy 上限；checkpoint 自报 MMD=1s 但实际 10s 也必须拒绝；monitor 签名跨 policy context 重放；全零 enrollment review。这些测试不是让状态更好看，而是证明两个重要边界：`l2c_report_authenticity_proven=false`，`anti_rollback_protection_proven=false`。

第三个部署边界也被明写：同 UID 可以一起替换 Python 源码、registry 和摘要常量，所以 `same_uid_trust_root_replacement_excluded=false`、`verifier_binary_integrity_proven=false`。真正生产判定必须搬到不同 UID 的 root-owned 只读安装或远程 verifier，本地 JSON 不能作权威授权凭证。

公开 registry 仍故意没有生产 anchor，CLI 会在读取不存在的私有 bundle 前先返回
`NO_L2D_PRODUCTION_TRUST_ANCHOR_ENROLLED`。真实外部 ledger、真实 monitor、gossip、adapter、三维重建与训练
仍全部是 0。

安全审阅同时给了一个很重要的方向纠偏：这些机制不能成为第一次和师兄沟通的主角。第一次只应问真实
forward 入口、field 还是 decoder 参数化、residual 在 ray/sample 层还是 detector map 末端形成、能否做任意
方向 JVP/VJP、是否有 hard branch、最小合法 batch、运行环境和组内真正痛点。为此新增了一页自然中文消息，
不再让师兄第一次就评审 nonce、Ed25519 或 Landlock。

完整证明边界、一级来源和后续在线状态机见
[L2-D0 离线前缀与角色证据](n5_d5_l2d_offline_prefix_and_role_evidence_2026-07-19.md)；可直接发给师兄的版本见
[N5-D5 师兄首次沟通单](n5_d5_advisor_first_contact_2026-07-19.md)。

本轮最终验证数字已重跑：L2-D0 `31 passed`，L1/L2 聚焦核心 `133 passed`，聚焦页面 `69 passed`，fast matrix `257 passed`。medium 四进程层为 `2242 passed, 3 failed, 55 warnings`；三条失败仍是已冻结的 D4c/N2 证据状态，与本轮 L2-D0 无关。按矩阵设计拆出的 18 项 macOS containment 和 3 项 MPS 串行层另外 `21 passed`。因此可以说本轮增量没有引入新回归，但不能说全仓 medium 全绿。

## 123. 更细网格在练习题上赢了，换一个旋转角却输了

这一轮终于不再继续堆安全合同，而是回到一个真实三维 BOST 科学问题：`32³` 在九个 support views 上把
relative-L2 从 `16³` 的 `0.787711` 降到 `0.627132`，这 20.4% 的 support 改善会不会迁移到没参与重建的
rotation-40？这里必须先说清：camera 仍是 2、3、4，未见的是 rotation run，不是新相机。

为了不看结果改规则，我先把配置、runner、测试、forward、metric 和说明提交成 protocol commit
`ba77a17f...`，确认结果目录不存在；再单独生成 attestation，绑定两个场、生成报告、support split、
rotation-40 payload/geometry 与全部受监控代码的 hash。正式运行前还逐文件检查了三台相机的 `.npy` SHA、
shape、dtype 和 manifest 交叉绑定。使用全部 `3,847,050` 条 active rays，每个候选只做一次完整 forward，
公开包只写聚合数值和图。

结果很清楚：16³/32³ 在 rotation-40 的 pooled rel-L2 是 `0.843263 / 0.959591`，也就是更细网格反而退化
`0.116328`。camera 2、3、4 分别退化 `0.061519 / 0.110005 / 0.145689`，不是某一台相机拖累 pooled。
equal-camera macro 也从 `0.825173` 退化到 `0.930910`。预注册要求至少改善 `0.01` 且三相机都不伤害，
所以机器判决是明确 NO-GO。

还有一个很直观的线索：rotation-40 实测位移 RMS 是 `0.302716 px`，16³ 预测只有 `0.143223 px`，32³ 更低到
`0.082605 px`。我没有在 rotation-40 上补一个尺度因子，因为用同一数据拟合再评分就是泄漏。现在只能说
support-fit 和 held-out reprojection 发生了反转；不能说 16³ 的真实三维场更准，也不能把三台相机冒充三次独立重复。

这次 NO-GO 真正改变了算法方向：下一步不再把“更细网格”本身当创新，而是研究 coarse reconstruction 加
受约束 fine correction。fine branch 必须保留数据一致性、满足 coarse restriction，并按整组 rotation 做留出；
只要 correction 伤害任一 rotation/camera tail，就退回 coarse。工作名暂定 RTG-MRC，但在完成原创性检索前不称
新算法。先补 32³ 的 early-stopping、H1/TV 和 coarse-to-fine 强基线，再考虑让网络只学稳定 correction。

完整数字、边界、下一算法方程、第一阶段成功门与给师兄的五个问题见
[rotation-40 分辨率迁移 NO-GO](psu_rotation40_resolution_transfer_result_2026-07-19.md)。

两个独立审计随后把表述又收紧了一步。第一，机器里的 pooled 指标其实是把全部 ray 拼起来算一次 global
norm ratio，不是三个相机 relative-L2 的 ray-count 加权平均；这个标签写错不改变数值或 NO-GO，因为
equal-camera macro 与三项 camera delta 也全部退化。第二，本轮严格否证的是冻结的 `32³+CGLS4` package，
不能把网格和固定四步 CGLS 的谱滤波/收敛阶段拆开归因。

复现审计还找到了 attestation 漏掉的四个传递依赖和 requirements。它们本次与 protocol commit 完全一致，
审计代理绕过 runner 重算全部 384 万 rays，所有指标与 JSON 最大差 `1.01e-14`；但预结果机制仍不能叫完整
fail-closed。页面现在把这条 P1、四个依赖 hash、单侧预注册性质、`N=1 rotation block`、环境指纹和独立
clone replay 命令全部公开。另加的公开包 validator 会限制任意 list/对象/数值预算并交叉核 JSON、CSV、PNG、
PDF 与 checksum；它保护当前公开结果，不能倒推修复原 protocol。

这让我学到：一个数值可以是真的，证据链仍可能不完整；一个 NO-GO 可以很有用，原因归因仍必须克制。
下一版先补依赖闭包、正确 pooled 名称、support view identity 和 rotation-group baseline，再设计 RTG-MRC。

## 124. 我试着问“到底是网格坏了还是第四步算坏了”，答案是：现在还拆不干净

上一轮只知道 `16³+CGLS4` 在 rotation-40 胜过 `32³+CGLS4`，但这个比较一次改变了太多东西。
这次我先把五个对照和阈值写进 commit `48e32d7...`，连 runner 的传递 import 依赖一起锁住，推到 GitHub
后才运行。final rotations 继续没打开，公开目录也仍然只有汇总值。

最关键的对照是把 16³ 场用端点对齐三线性方法放到 32³，再用 32³ forward 计算。原来的
`A16 x16` relative-L2 是 `0.843263`，`A32 U(x16)` 是 `0.856804`，光换表示和 forward 就差了
`0.013541`，超过事前写的 `0.01` 数值屏。两个预测本身的差还是实测向量范数的 `0.111838`，所以不能把
forward 离散变化当成可以忽略的小数误差。

再看从 `U(x16)` 走到 `x32` 的冻结场修正，pooled 上它与当前残差的 cosine 是 `-0.052812`；但拆到相机后，
camera 2 是 `+0.276931`，camera 3/4 是 `-0.137083/-0.162454`。也就是说，同一个 fine correction 对一台
相机有帮助，对另外两台方向相反。固定 alpha 曲线也完全复现这个冲突：camera 2 在 alpha `0.25-0.50`
改善，camera 3/4 随 alpha 增大一直变差。不能把 camera 2 的 post-open 最优 alpha 拿来当算法结果。

机器最终没有给出“找到原因”，而是
`OPENED_BLOCK_FORWARD_GRID_CHANGE_MATERIAL_MECHANISM_UNRESOLVED`。独立代码审计又提醒：机器门只用两个
残差范数的差，理论上可能漏掉预测方向差；这次碰巧报告里的预测差也不小，但下一版必须把它正式纳入门。
另外，16 与 32 不是嵌套节点，普通 trilinear `U/D` 不是正交 multigrid restriction/prolongation，不能把
`x32-U(D(x32))` 直接叫严格高频。

所以现在最真实的新发现不是“某个新模型赢了”，而是：**一个 pooled 全局 gate 不足以保护真实多相机 BOST
修正，离散表示和求解轨迹还必须在 support rotation 内部分开验证。** 下一步只用 support rotations
`0/50/90` 做整组 leave-one-rotation-out，重放 CGLS `k=1,2,3,4,6,8,12`，使用质量加权 coarse 子空间投影，
再做可加误差归因。三折不能同向复现，就停止归因，不拿 rotation-40 继续调参数。

完整数字、为什么不能叫高频过拟合、五篇最相关一级来源和下一算法骨架见
[多分辨率机制诊断结果](psu_rotation40_multiresolution_diagnosis_result_2026-07-19.md)。

结果提交 `87f5e79...` 推送后，独立代理又从私有输入重算全部 384 万 rays：JSON/CSV 最大差为 0，
alpha 曲线最大差约 `1e-14`，图和隐私扫描也通过。它留下的术语勘误很重要：冻结图里的
`fine-field correction` 只能读作 `x32-Ux16` 两个完整重建场之差，不能读成严格高频；跨网格 raw
field norm 也没有乘体素体积，不能冒充连续物理能量。完整清单见
[独立结果审计](psu_rotation40_multiresolution_diagnosis_independent_audit_2026-07-19.md)。

## 125. 先确认每张“卷子”是谁的，再开始三折考试

上一节决定在 support rotations `0/50/90` 内做整组 leave-one-rotation-out，但我发现还有一个不能凭印象跳过的问题：九个 support view 在机器文件里只有 `0..8`，如果 rotation/camera 身份没有钉死，所谓“整组留一 rotation”可能只是我们自己贴的标签。

这次没有训练，也没有跑任何 LORO 分数。我先回到数据作者的 MATLAB 源程序。`AEDC_pprocess_auto.m` 按 `0,50,90` 的外层 rotation 循环拼接三个文件；每个文件又由 `AEDC_pprocess.m` 按 camera `2:4` 的内层循环拼接。因此 view 0/1/2 是 rotation 0 的 camera 2/3/4，view 3/4/5 是 rotation 50，view 6/7/8 是 rotation 90。九个 bundle manifest 还逐段证明每个 view 是 `HSOF_9CAM_RT.mat` 中连续的 5,529,600 rows，不是看残差后猜出来的顺序。

为了让 16³ 和 32³ 后面跑三折时少做重复工作，我新建了私有 16³ compact cache。它和已有 32³ cache 都有 10,628,822 条 corrected active rays、每条 16 个 aperture samples、329 个 chunks，各约 5.02 GB。审计器用 `verify_hashes=True` 逐数组重开：observations、camera projection、ray scale 和 valid mask 的哈希完全相同；只有网格相关的 lower-corner index 与 trilinear fraction 不同。为了不只靠共同数组猜测，审计器还把 170,061,152 个有效 aperture sample、510,183,456 个坐标分量分 chunk 反算成归一化三维位置；最大差 `1.11e-16`，通过 `1e-12` 门。这才足以说明两级离散确实在看同一批物理观测与射线参数，而不是两个偷偷换过输入的数据集。

机器状态是 `SUPPORT_ROTATION_LORO_PREFLIGHT_PASS`，但它只代表身份/cache 前置门通过。这里的身份是作者脚本、连续 block manifest 与本次 private report 的跨文件绑定；cache manifest 自己只有 view ID，并没有 camera/rotation 字段。全 detector geometry audit 的既有 NO-GO 也没有因此消失，本次“可用”只指 corrected-active-ray B0 pipeline。公开摘要不含路径、测量值、私有哈希或重建体，两个约 5 GB cache 和 private report 都不上传。现在仍然没有 field truth、没有 LORO score、没有跨流态泛化，更没有神经算子成功。

下一步正式协议会固定三折：50/90 训练留 0，0/90 训练留 50，0/50 训练留 90；每折只消费 `k=0,1,2,3,4,6,8,12`，同时记录 train 与 held-out rotation、三台 camera tail、normal residual、A/Aᵀ 调用和 wall time。只有三折方向稳定，才继续做 coarse-subspace 投影与误差归因。

完整映射、隐私边界和重放命令见 [support LORO preflight](psu_support_rotation_loro_preflight_2026-07-19.md)。

## 126. 三折考试跑完了：更细没有稳稳赢，但我终于看见了真正的问题

这次不是再做准备。Mac 用了约 48 分钟，把 rotation 0、50、90 轮流整组留出；每一次都只用另外两个
rotation 重建，再看没参与求解的三台相机。16³ 和 32³ 都保存了 `k=0,1,2,3,4,6,8,12`，一共 48 个
CGLS 场。正式结果先说坏消息：固定 `k=4` 时，32³ 没有稳定赢 16³。三折等权的 camera-macro 改善量
`16-32` 是 `-0.008178`，九个 camera 有的改善、有的变差，所有“每折、每 camera、每个尾部都不伤害”
的门都没过。所以我不能说更细网格更准，更不能说已经有新算法。

但我看到了比“谁赢一点”更有用的东西。六个组合，也就是 3 个留出 rotation × 2 个网格，全都发生同一件事：
从第四步继续算到第十二步，训练观测越来越合，没见过的 rotation 却至少有一项 pooled、camera-macro、
worst-camera 或 p95 变差。最明显的是留出 50° 的 16³，训练 macro 又下降 `0.0499`，留出 macro 却上升
`0.1287`。这就是逆问题里的半收敛：继续“解题”不等于继续接近可迁移的物理信息，后面可能只是在吸收
当前视角最容易拟合的不稳定方向。

我还把 32³ 相对 16³ 的差拆成 coarse-range disagreement 和 fine-grid orthogonal complement。结果也没有
便宜答案：留出 0° 时两项互相抵消、净伤害很小；50° 时两项都伤害；90° 时两项都改善。因此不能说
“把高频删掉就好”，也不能说“只让网络学细节就好”。它们是不是有用取决于 rotation/camera geometry，
未来的 correction 必须能按观测支持判断，判断不了就退回 coarse。

下一步 E72 不需要等师兄给新数据，也不是马上堆 FNO。我只需再算 rotation 0、50、90 各自单独训练的
16³/32³ 轨迹，共 6 条。对任意一个 outer heldout，只让另外两个 rotations 互相做 inner validation 来选停止步；
outer 的分数绝不能参与选 `k`。然后把这个 nested rotation-aware stopping 和 fixed `k=4`、只看训练 residual、
L-curve/GCV/discrepancy、H1/TV、heldout oracle 上界放在一张表里。它若不能在三折、九个 camera 和 p95 上
都 no-harm，就老实失败；它若通过，才有资格成为后面神经 correction 的稳定基座。

独立 validator 已经重新推导 fold 身份、调用预算、macro/worst、九个 camera 差、六个半收敛 screen 和
Shapley closure；9 项篡改测试通过。公开摘要没有本地路径、测量值、重建体或私有报告 hash。完整数字、
E72 方程、通过门和给师兄的五个问题见 [support LORO 正式结果](psu_support_rotation_loro_result_2026-07-19.md)。

## 127. 内层选早了，第三个角度的一台相机还是受伤：E72 老实失败

E72 不是又打开一个网络，而是先问一个基础问题：既然 E71 已经看到六条轨迹的半收敛，能不能用另外两个 rotation 互相验证，为第三个 rotation 选一个更安全的停止步？为了不让答案影响规则，我先把 checkpoint、指标、容差、回退和 outer 门全部提交。

第一次启动就被预检拦住了。我把 `50°` 三台相机的 ray count 抄错，而且三个错数的总和巧合地和正确总和一样。如果只查“一共有多少 rays”，这个错误就会混过去。运行器逐 view 和两份 cache manifest 比较，在 adjoint check、CGLS 和 selection 之前停了。所以我能透明修正三个整数，并记录修订前的轨迹数和已选 checkpoint 数都是 `0`，不需要假装这个错误从没发生。

正式运行是 `3 rotations x 2 grids = 6` 条唯一 CGLS 轨迹。每条都从零场开始，保存 `k=0,1,2,3,4,6,8,12`，一共 48 个私有 float64 场。本机用了 `1104.6 s`，峰值内存约 `7.11 GB`，完整证据预算是 `122 A / 86 A^T`。所有 solver 调用都只看它声明的 train rotation；选择器本身是 `0 A / 0 A^T`。

inner 结果其实给了一点希望。要预测 outer `50°` 时，它只用 `0° -> 90°` 和 `90° -> 0°`。`16³` 的 `k=3` 通过所有 inner no-harm，最坏主风险是 `0.998017`；`32³` 的 `k=2` 是 `0.998749`。其他四个 outer-grid 单元找不到严格安全改善，所以回退 `k=4`。按 outer `0°/50°/90°` 且每个角度先 16³ 后 32³ 排列，密封选择是 `[4, 4, 3, 2, 4, 4]`。这六个选择先单独提交，后面才允许程序读 E71 outer。

考试却没过。`16³/k3` 在 outer `50°` 的 macro 的确改善了 `0.001254`，但 group p95 上升，view 5 的 relative-L2 与 p95 分别变差约 `1.69% / 2.19%`。`32³/k2` 更明显：macro 自身就变差 `0.006310`，view 5 L2/p95 变差约 `3.03% / 3.83%`。所以即使 `16³` 三 outer 平均 macro 是正改善，也不能用平均把一台相机和尾部的伤害藏起来。两个网格都是 NO-GO。

我现在不应该继续在同一份 outer 上改阈值，因为那只是把考试题变成练习题。E72 最有用的结论是：两个 rotation 间看似安全的 stopping，不会自动保护第三个 rotation 的相机尾部。下一代方法必须把 geometry、camera group 和 tail risk 显式放进合同，证明不了就回退经典解。而且先要在新 flow 上补 H1/TV、真 pyramid BOST 和有噪声尺度的 discrepancy 强基线，再让网络学可拒答 correction。

完整数字、四联图、独立 validator、下一代三个算法候选和给师兄的问题见 [E72 正式结果](psu_nested_rotation_stopping_result_2026-07-19.md)。

## 128. 不再只猜一个 k：所有候选和所有相机尾部一起受审

E72 已经证明，只看平均分会被骗。`16³/k3` 的三折平均 macro 有一点改善，但 outer 50° 的 view 5 和 p95
同时受伤。于是这一轮没有继续在同一份结果上调阈值，也没有马上训练 FNO。我先写了一个更严格的 E73-B
证书核心。

可以把它想成一次开卷考试。候选不只有 `k2`，而是提前写死的一组 `k1/k2/k3/k6/k8/k12`；卷面也不只有
一个总分，而是 macro、worst camera、group p95 和每台相机的 L2/p95。每个独立 flow/session 最后只交一个
“最坏扣分”：把所有候选、所有指标里最危险的那一项取最大。校准得到的上界因此一次包住整个表，后面从
这组已冻结候选中选哪个 `k`，不能把不利相机悄悄删掉。

另一个关键改动是“样本不够就真的没答案”。秩用
`ceil((n+1)*(1-alpha))`。90% 覆盖至少要 9 个独立 calibration units，95% 至少 19 个，97.5% 至少 39 个，
99% 至少 99 个；少于这个数量时，程序返回无穷上界并使用 `k4` fallback，不会把秩硬截到已有最大值。
相机、ray、pixel 都不能拿来凑这个 `n`，因为它们共享同一个三维场和实验误差。

代码还会检查三件容易接错的东西：特征名称与顺序、特征合同 SHA-256、预测器产物 SHA-256。support box 外
直接回退。不过独立审稿提醒得很对：调用方同时递交“预测矩阵和正确哈希字符串”，不等于证明矩阵真由该
模型产生。现在这层只是接口防呆；正式 runner 必须自己按哈希加载模型并计算预测。axis-aligned box 也会接受
训练中没联合出现过的角点，所以它只能缩小明显外推，不能声称 OOD 安全或“被接管样本条件下 90% 安全”。

独立代码审稿随后抓到两个真问题。第一，旧实现把“严格改善”写成 `upper < tolerance - margin`；如果允许
0.1 的伤害、要求改善 0.05，一个仍然伤害 0.04 的候选竟可能被叫成改善。现在固定为 `upper < -margin`。
第二，旧 dataclass 可以被直接构造，绕开校准函数的正尺度检查。现在构造器会重新验证正尺度、rank、quantile、
fit/calibration/scale unit IDs 与哈希；prediction table 也把 action/metric 名称和顺序一起绑定。部署 unit 不能复用
fit/calibration ID。尺度来源和单次未来单位边界也都进入机器合同。

第二轮复审又发现一个更隐蔽的 Python 坑：NumPy 数组即使先设成只读，只要底层内存仍归数组自己所有，调用方
还可以把 `WRITEABLE` 标志重新打开。这样就可能把绑定时为正的伤害改成负数。现在核心数组改成以不可变
`bytes` 为底层的视图，重新打开写权限会直接失败；prediction 和 scale 还各自绑定 canonical float64 数值哈希，
每次计算上界前再验一次。攻击测试已经覆盖“`+0.04` 被强改成 `-1.0` 也不能获准接管”和“任意正尺度替换也
因数值哈希不符而拒绝”。

当前是 `35` 项 E73 聚焦测试、`0` 个新 flow 分数、`0` 个已训练 predictor。第一轮 Mac pilot 会先用 16 个解析
phantom 检查 schema、成本和强制回退；随后才考虑至少 64 个 family-aware instances，建议拆成
`24 model-fit + 20 calibration + 20 sealed audit`。即使解析形态通过，也不能冒充 CFD 或真实反应流场。

这一步的现实意义是给后续神经 correction 装一个“不会就退回经典解”的门，而不是把 conformal 包装器本身
说成新三维网络。要成为论文贡献，还必须补 H1/TV/Pyramid BOST 强基线、多个独立真实 flow/session 和一次
prospective audit。完整方程、样本量表、一级来源、Go/No-Go 与给师兄的问题见
[E73-B 联合几何与相机尾部证书协议](e73_joint_geometry_tail_certificate_protocol_2026-07-19.md)。

## 129. 模型自己算预测了，但不可信 feature 仍然不能接管

上一轮留下一个很具体的洞：调用方可以同时递交 predicted harm 和“匹配的 predictor hash”。这次我先写了
artifact-owned runner，让程序只读固定 JSON ridge 公式并亲自算预测。第一版虽然通过了 digest、shape、symlink
和原子输出测试，独立安全审稿还是判它不能上线，原因很直接：feature vector 仍由调用方填写，模型来源正确并
不等于输入来源正确。

所以我没有把测试通过写成“正式 runner 完成”，而是把权限继续收紧。静态 predictor/scale/support/certificate/
policy bundle 与每次 deployment 现在使用两个独立哈希；命令行只能访问 `private_library/e73_jgtce`，只收小写
ID，不收任意路径，也不能 `--replace`。结果用独占 hard-link 发布，同一个声明物理单元先写消费 marker；改
deployment 文件名或结果名不能重复用。JSON 会拒绝重复 key、NaN/Infinity、bool/数字字符串、过深结构，文件
会拒绝路径穿越、父目录或最终 symlink、hard-link 和 FIFO。runner/core 源码也进入静态 bundle 的哈希绑定。

最重要的行为是：precomputed feature 即使内部 diagnostic 本来会选 `k1`，正式输出仍固定为 `k4`，不会泄露或
授权那个 candidate。当前状态因此叫
`DEVELOPMENT_ONLY_CALLER_FEATURES_UNTRUSTED_NO_CANDIDATE_AUTHORIZATION`。这不是算法成功，甚至还不是可部署
selector；它只证明模型工件来源和 fail-closed 发布比上一版更完整。

数据审稿同时发现 E73 草案写了 `shear_layer/multi_lobe`，但真正的生成器是 `vortex_pair/multi_plume`，已经纠正。
Phase-0 现在冻结 16 个 analytic proxy：八个真实支持 family 各两个 seed；九个 PSU view slot 分别计 L2/p95，
不再把重复 camera ID 合并。一次 `k0,k1,k2,k3,k4,k6,k8,k12` 轨迹的完整预算是每 unit `20 A + 13 A^T`，
16 个加共享 dot test 为 `321 A + 209 A^T`。

但现在仍禁止启动计分。现有 compact cache 构造器会顺带打开真实 `observations_uv.npy`，而 synthetic schema pilot
只该借 geometry；必须先写 geometry-only adapter，并用 open-file ledger 证明 observation 打开次数为零。29 维
feature 公式也还没有代码与 hash，fallback 下游数组还没有和保存的 `k4` 做逐字节一致性门。因此 Phase-0 状态是
`PHASE0_PREFLIGHT_NO_GO_CONTRACT_INCOMPLETE`，不是算法 NO-GO。

完整的 16-unit manifest、29 维 feature、24 个指标、攻击面与四个启动阻断项见
[E73 runner 与 Phase-0 前置审计](e73_formal_runner_and_phase0_preflight_2026-07-19.md)。

## 130. “回退”也不能由配置随便起名：第二次安全复审又拦下一次假安全

上一节说正式输出固定为 `k4`，但独立复审没有只看说明文字。它沿代码发现 policy parser 实际只检查
“fallback 不在候选列表中”。如果有人把 policy 改成 `fallback_action=cgls_k5`，再把 policy 和 manifest 的
SHA-256 全部重新算一遍，旧 runner 仍会把 `k5` 当作合法回退发布。候选授权标志虽然仍是 false，但“不会就
回到同一个经典基线”的物理合同已经被偷偷换掉，所以这是合同级 P0，而不是文案小问题。

现在 runner 代码本身新增唯一常量 `FORCED_FALLBACK_ACTION="cgls_k4"`。policy 不是 `k4`，或者候选列表
反过来包含 `k4`，都必须拒绝。攻击测试不是只改文件让旧哈希失配，而是把恶意 policy 和 manifest 哈希一并
重算，确认“内部完全自洽的假 bundle”仍然过不了。runner 与 certificate core 的两个源码哈希也分别补测，
不再用一个测试名只覆盖其中一个字段。

复审还指出，bundle、deployment 和结果发布原先会三次按路径重新打开 private root，严格说中间存在目录被
替换的窗口。现在一次 `run` 只打开根目录一次，后续读取、消费账本和结果发布都沿同一个 file descriptor
完成；新测试计数确认整个事务只调用一次根目录打开函数。E73 的代码/协议合同测试因此从 55 项增加到 58 项，
另有 4 项聚焦网页映射测试。

修复后同一位独立审稿者只读复查这三个点，判定 `3 CLOSED / 0 PARTIAL / 0 OPEN`，没有发现新的 P0/P1；
我本地实际运行的 62 项 E73 聚焦检查也全部通过。这个数字只覆盖本轮代码与合同，不是 62 个物理实验。

这里仍不能写“安全 runner 完成”。两个 CLI 哈希依赖可信操作者，不是签名启动链；同一真实 flow 人为改名
仍可能绕过声明账本；进程在 marker 后崩溃会烧掉该单位；最重要的 raw-BOST feature 来源和下游 `k4` 数组
逐字节一致性仍未闭合。当前结论只是：任意动作已不能冒充 fallback，根目录替换窗口更小，而 candidate
authorization 继续保持关闭。

## 131. 这次真的没有偷看 observation：先把 feature 的物理含义钉准

Phase-0 原来卡在一个很朴素的问题：16³ compact cache 虽然主要是 ray geometry，但旧 store 一构造就会顺带
打开 `observations_uv.npy`。我们想做的是 `y=A(x_proxy)` 的纯 synthetic schema test；程序哪怕只是“顺手打开”
真实测量、嘴上说没使用，后面也很难向审稿人证明没有泄漏。

我新写的 geometry-only store 只认识五类东西：插值 lower-corner、fraction、valid mask、投影向量和 ray scale。
它没有 observation 属性，chunk 中 observation 固定为 `None`，任何人调用 `load_observations()` 都会得到
`PermissionError`。manifest 若把 `ray_scale` 等角色偷偷指向 `observations_uv.npy`，也会在打开前拒绝。小型
测试甚至直接删掉 observation 文件，geometry forward/adjoint 仍与完整 cache 逐值相同。

然后我没有只相信程序自己的“我没打开”日志，而是用 Python 的 host audit hook 监听真实 cache 根下的文件打开。
在 10,628,822 条 rays、9 个 view slots 上，它实际看见 manifest 和五类 geometry 数组，observation open count
严格为 `0`。这没有产生任何重建分数，但把 Phase-0 最危险的数据偷看入口关掉了一层。

29 维 feature 也从文字变成了代码。这里顺手纠正了一个物理命名：底层 `valid[ray,sample]` 表示 aperture sample
是否落在网格内，所以它是 `global_valid_aperture_sample_fraction`，不是“有效 ray 比例”。前 23 维仍是固定
geometry 摘要，后 6 维是 `k0..k4` measurement residual 和 `k4` normal residual。同一 geometry 上前 23 维
完全不变，因此这批 16 个 phantom 依然不能证明 geometry learning。

我还另写了一个不导入 producer 的 witness，用 Python `math` 独立重算全部 29 个值。攻击测试修改一个 feature
并重算 feature hash，或者修改 input record 并重算 input hash，都会因为独立公式对不上被拒绝。当前总计新增
16 项 geometry/feature 测试，E73 代码与协议合同从 58 项增加到 74 项。

现在仍不能启动 16-unit 计分。缺的是正式 data runner：它必须自己从 synthetic observation 和 CGLS 早期状态
构造 sufficient statistics，不能让调用方填写；拒答后的三维输出还必须与保存的 `k4` 数组逐字节相同。本轮
代码与修订 config 也要先提交，之后才允许看 metric/harm。完整边界见
[geometry-only 与 29 维 feature 说明](e73_geometry_feature_boundary_2026-07-19.md)。

## 132. “文件名没看到 observation”还不够：把审计从门口走到算子里面

上一节写“真的没有偷看”之后，独立审稿没有照单全收。它指出两个我必须承认的问题。第一，旧审计只按程序
请求的 basename 计数；如果 geometry 文件是指向 observation 的符号链接，日志仍可能只写 geometry 名。第二，
审计只构造了 adapter，并没有真的走 `iter_chunks()`、forward 和 adjoint。真实 cache 当时确实没有这些攻击，
但“这次没出事”和“合同能拒绝这类事”不是同一个结论。

修订后，可信 manifest digest 变成必填项。cache root 用目录 FD 持有，每个 geometry 文件通过 `O_NOFOLLOW` 只打开
一次；NPY header、SHA-256 校验和 copy-on-write mmap 都沿同一个 FD 完成，不再“先检查路径、再让 NumPy 重开”。
程序也在读 payload 前检查 device/inode，不能用 hard-link 冒充；manifest、
source selection、view row、chunk、array record 和 claim boundary 都是严格字段白名单，不能在不起眼的嵌套字段里
塞额外值。真实审计也不再现场读取 manifest 后自己相信自己，而是从 Git 忽略的私有 attestation 取得预先保存的
digest。公开 JSON 只写“已绑定”，不公开私有 digest 或路径。

这次真实复跑用了 10,628,822 条 rays，遍历 329 个 chunks，并实际执行一次 matrix-free forward 与 adjoint，耗时约
26 秒。四条路径结束后，host hook 仍只看到 manifest 与五类 geometry 数组，observation open count 为 `0`。
这是比上一节更强的证据，但范围仍只到这次受信 manifest、这份 adapter 和已执行的四条路径，不代表所有未来
进程都被神奇地证明安全。现在五个 geometry 文件各只出现 1 次 open，而不是旧实现中 mmap/hash 分开重开的 4 次。

审稿还纠正了我对 feature witness 的一句过头表述。witness 能抓到“改 feature 后只重算 feature hash”或“改 input
后仍拿旧 feature”的不一致；如果攻击者把 input、29 个 feature 和全部 hash 一起按公开公式重做，它当然会通过。
所以它叫独立公式见证，不叫来源认证。现在 artifact 与摘要都明确写着
`INTERNAL_CONSISTENCY_ONLY_NOT_SOURCE_AUTHENTICITY`，并把 sufficient-stat provenance 保持为 false。真正的来源门
只能由下一步 data runner 内部构造 residual 统计并绑定 runner/config hash 来关闭。

这轮也补了非立方 `shape_zyx=[4,5,6]` 的轴顺序测试，防止立方 `16³` 把 x/z 交换错误藏起来。这里仍没有打开
16-unit 的任何 field error 或 harm；补上“可信 digest 不可省略”“audit 拒绝空 trust root”和“整份自洽重生成确实
会通过 witness”的正向边界测试，以及“恶意相对 FD 打开 observation 必须被 host hook 抓住”的测试后，E73 代码
与协议合同现为 83 项。新增的是更可信的数据边界，不是算法胜利。

## 133. 同一条 CGLS 里长出 feature，回退文件也必须真的是保存的 k4

上一节留下的不是“再想一个网络”，而是一条具体的软件债：29 个 feature 虽然有 producer 和独立 witness，
但 residual 还是可以由调用方填；说明文字虽然说失败就回 `k4`，真正落盘的三维数组也没有和保存 checkpoint
逐字节对上。这次我先把这条数据链写出来，只在很小的内存夹具上跑，不打开 16 个 phantom 的任何分数。

运行器现在先用 analytic proxy 自己算 `y=A(x)`，然后只跑一条普通 CGLS 到第 12 步。`k0/1/2/3/4/6/8/12`
都从这条递推克隆，不会为了比较不同停止步重复跑前缀。feature 需要的 `k0..k4` measurement residual 和
`A^T r4` 也直接取自同一递推，不再让外面的人递一张“我保证这些数是真的”的表。小夹具上保存的每个非零
checkpoint 都和原来的 `cgls_solve` 完全相同；基础层调用顺序固定为一次 truth forward、一次 initial adjoint，
再做 12 对 forward/adjoint，总计 `13 A + 13 A^T`。后面七次 scoring forward 还没有做。

独立审稿第一次没有让它过关。审稿者做了一个会在每次 `iter_chunks()` 里偷读 observation、却一直谎报 open
count 为 0 的假 store；旧通用入口居然还能返回“valid”。现在通用入口被明确降级为 `FIXTURE_ONLY`，即使这个
恶意夹具暗读了 26 次，也永远不能得到生产状态。私有生产入口只接受精确的 geometry-only adapter 和 streaming
operator 类型，manifest digest 也不再由调用方提供，只能从 Git 忽略的 private attestation 读取。

审稿者还指出，内存里拿两份一样的 bytes 比一遍不算最终回退证明。现在 checkpoint payload map 是不可变的；
finalizer 没有“请提交 fallback bytes”这个参数，只能把密封 `k4` 原样写成 `cgls_k4.npy` 和
`forced_fallback.npy`。两个文件写完后会重新打开，比较完整 bytes 和 SHA；其余成员也全部重读，最后才写
`finalization.json`。同一个 run ID 不能覆盖，文件改一个字节后 verifier 会失败。

故障注入还覆盖了中途 denominator breakdown、adjoint NaN、伪造调用顺序、uint8 valid mask 和重复发布。
当前 E73 聚焦链是 99 项合同，相关数据链 53 项通过，bounded fast matrix 是 418 passed；旧站点 46,771 个
本地目标仍然 0 missing。它们都是软件检查，不是 94 个真实 flow。

这里仍然不能启动 16-unit 分数。下一步必须先写九个 view slot 的专用 metric scorer：七个非零 checkpoint
各一次 forward，24 个 metric 全部有限，harm 只准写成 `metric(candidate)-metric(k4)`，并且只落到私有目录。
然后提交本轮代码，让生产入口在干净 Git commit 上重验 source snapshot、attestation、dot product、调用预算和
空 score ledger。完整边界和给师兄的四个问题见
[Phase-0 数据基础层说明](e73_phase0_data_foundation_2026-07-19.md)。

## 134. 校验和能一起重算，所以还要把数组、历史和运行身份互相钉住

第三轮独立审稿又做了一个更强的攻击：把保存的 `k1` 换成零场，再把 `finalization.json` 里的 SHA-256 一起
重算。旧 verifier 只看“文件和登记的 hash 是否一致”，因此会放过这个自洽但假的 bundle。它还指出，16 个目录
只有 run 名，没有把每个目录钉到冻结的 phantom 行；同一个 unit 完全可以换一个 run 名再跑一遍。

现在每个保存 checkpoint 都会在 reopening 时重新计算 `L2` 和 `max-abs`，与同一步 solver history 对齐；history
本身还要满足相对残差公式、measurement residual 非增、`beta_k=||s_k||²/||s_{k-1}||²` 和
`alpha_k=||s_{k-1}||²/||Ap_{k-1}||²`。因此 history 多存一个 projected-direction norm，不再只看 `alpha/beta`
是否为正。攻击测试真的把 `k1` 改成零场并重算最终 checksum，验证器会在“checkpoint 与 history 不一致”这一层
拒绝；把正的 alpha 放大 1000 倍、把正 beta 放大两倍也都会失败。

每个私有目录也新增了 `unit_manifest.json`，里面同时绑定 run ID、unit ID、A/B block、family、uint32 seed、冻结
phantom 整行 hash、config hash、cache alias 和私有 source-manifest hash。finalizer 对输出根持有排他锁，先逐个
重开旧 manifest 并检查旧目录之间也没有重复；随后在 bundle 外用 `.unit-claims/<unit_id>/` 的原子目录占位，保存
原 manifest，完成后再保存 finalization hash。这样同一个 unit 换 run 名、把 u01 bundle 合法重标成 u02，或者把
`k1` 整体取负并重算 bundle 内 checksum，都会与外部 local anchor 冲突。private attestation 的每层父路径都用
`O_NOFOLLOW` 打开，即时父目录必须只有本人可访问，文件必须是本人拥有的 `0600` 单链接常规文件，读取前后还比较
inode、size、mtime 和 ctime；重复 JSON key 也不再静默采用最后一个值。

这里必须讲清一个不漂亮但诚实的边界：Python 模块里的下划线和 capability 不是密码学安全。当前合同明确要求
本地 Python 进程可信，`same-process malicious-code resistance=false`，也没有数字签名。它防正常接线错误、普通
落盘篡改和本协议覆盖的重哈希攻击；如果有人能在同一解释器中任意执行恶意代码，或者把 bundle、history、manifest
全部按公开规则一起重造，仍需要只读归档、独立复算或以后引入签名来解决。

本轮 data runner 故障注入为 23/23，相关数据链 59/59，E73 聚焦合同 107/107，bounded fast matrix
为 429/429。这些总数都来自实际重跑，不是按新测试数简单相加。真实 private unit、24-metric scorer、harm 和
16-unit 分数仍然都是 0。下一步先等独立审稿确认这个边界内没有 P0/P1，再提交源码；提交后只做不计分 preflight，
不会因为基础设施更结实就偷偷把“可以算分”当成“算法已经成功”。

## 135. 发布包还没上线，隐私扫描先找到了 8 处 cache 指纹

新数据链的暂存差异没有带出私有 manifest digest，但是我把整个 Pages 预发布包重建后，又用本机
attestation 里的真实值反向扫了一遍。结果发现 16³/32³ 两个私有 cache manifest 指纹在 4 个本来会被
打包的历史 source/config/summary 文件里共出现 8 次。它们不是 VPN 密码，但是私有数据指纹，与网页上
“不发布 private cache digest”的承诺相冲，所以不能因为 Pages 当时关着就忽略。

build runner 现在从已跟踪的机器合同严格读出两个 cache digest，对所有准备复制的跟踪文件做精确字节
扫描。命中文本时只在发布副本里换成 `PRIVATE_CACHE_MANIFEST_SHA256_REDACTED`，不改历史私有
证据；如果同一指纹出现在不可解码二进制文件里，构建直接失败，不会默默复制。新增 3 个构建回归后，
Pages builder 测试为 13/13。

预发布构建报告明确记录 `4 files / 8 occurrences redacted`；用两个真实私有 digest 再扫新 artifact，
命中数是 `0`。245 个 HTML 共 46,819 个本地目标仍是 `0 missing`。GitHub Pages 继续关闭；这一步是
发布卫生修复，不是研究结果。

## 136. 断电恢复和发布安全也要 fail closed，但这仍然不是算法突破

第四轮独立复审没有找到新的算法结果，却找到了两类真实工程风险。第一类在 private finalizer：原子 claim 写完后如果
进程退出，旧版会留下一个没有 bundle 的占位，之后整个输出根都因 claim/run 轴不一致而阻塞。第二类在 Pages
builder：旧实现从工作树读内容却把 manifest 标成旧 `HEAD`，只识别当前配置里的小写摘要，而且其他目录的 PDF、
凭据形状文本和绝对路径默认会进入发布包。这些都没有造成当前线上泄漏，因为仓库保持 private、Pages 仍关闭，
但不能等真的公开后再修。

private runner 现在先写不可变 `PREPARING_UNPUBLISHED`。只有 claim 没有 anchor、run 没有 finalization、成员轴是
允许子集且每个现有成员与本次密封 bytes 完全相同，才会在根锁内把 residue 移到
`.unit-claims/.orphan-quarantine/` 并留下恢复记录。claim 后、run 目录后、首成员后故障都可重试；manifest 冲突、
state 缺失、成员篡改都会拒绝。若 `finalization.json` 已存在而 anchor 没写成，程序不会猜测它是否成功，继续人工
NO-GO。两个 FD 泄漏也已关闭，output root 只沿 `O_NOFOLLOW` 父链创建最后一级，符号链接目标侧不会被顺手建目录。

Pages builder 则改成只读固定 clean `HEAD` blobs，构建前后都绑定 commit + tree；tracked worktree 或 index 脏时直接
拒绝。所有 PDF、Python、tests/config、checkpoint、数组、私钥和未知二进制默认不发布；私有 cache digest 从整个
HEAD 的 cache/manifest 语义中发现，再按大小写混排、raw bytes、UTF-16LE/BE 统一检查。纯文本十六进制只在发布
副本中替换，二进制编码、伪装 PDF、真实 HOME、PHY 账号形状、PEM 私钥和赋值型凭据任一命中都删除不完整产物并
失败。定向攻击回归为 builder 22/22、runner 31/31；E73 聚焦 115/115，相关数据链 67/67。

**突破监测：当前仍是“无算法突破”。** 这轮是可复现性与发布安全里程碑。只有新方法在未见 rig/session 上、同
`A/A^T` 调用和端到端成本下，面对 fixed/discrepancy/hybrid CGLS、TV/Huber、NeRIF/NeDF 与神经算子强基线，
同时改善 field relative-L2、逐相机尾部和 harm，并有重复、区间与消融，才标记为突破。合成均值提升、测试数增加、
或者把 DeepONet/FNO 接上现有重建都不够。

## 137. 半搬迁不能算隔离完成，发布包也不能沿一条后来被换掉的路径写

上一轮的 orphan quarantine 仍有一个断电窗口：先把 run 搬进隔离目录、再搬 claim，如果进程恰好死在两次 rename
之间，下一次会新建第二个隔离条目，证据被拆开；如果 claim 已搬走但 complete 还没写，旧流程甚至可能把它当成“没有
claim”而继续。现在每个隔离条目先写不可变 `ISOLATION_PREPARED` 记录，逐个绑定现有 run 成员的 SHA-256；run 与
claim 各自只能在 source 或 destination 一边出现，搬迁后分别 fsync 两个父目录。只有第二个不可变文件
`isolation_complete.json` 与 prepared 记录完全对得上，事务才算关闭。run rename 前后、claim rename 前后、complete
落盘后五个故障点都实际打断并重试过，最终仍只有一个隔离目录；其他 unit 遇到未完成事务会被阻断，旧版记录和被改过的
completion 只能人工审计。

Pages 的风险更像“门牌没变，但门后面的房子被换了”：旧代码检查 `build/` 不是 symlink 后，又按普通路径逐文件写和
清理，检查与使用之间仍有竞态。新构建全程持有 repo/build/staging 的目录 FD，每一级都用 `O_NOFOLLOW`，发布前复核
`build` inode，扫描通过后才在同一目录 rename；失败安装会恢复上一份完整 artifact，清理也只认本事务记录的 inode。
内容扫描不再只看开头 1 KB 或原始文本：公开文本必须是 UTF-8，再生成有界的 NFKC、HTML、URL、JSON/JS 转义视图，
Base64/data-URI PDF、内嵌 Bearer、数据库 DSN、大小写/转义用户目录和编码后的私有摘要都会 fail closed。目录置换测试
确认仓库外 marker 没有被写入或删除。

本轮实际回归为：Pages builder `51/51`，data-foundation runner `39/39`，三组定向合同合计 `98/98`，E73 聚焦
`123/123`，相关链 `76/76`，bounded fast matrix `483/483`；旧 artifact 的 245 个 HTML、46,771 个本地目标仍为
`0 missing`。这些数字是软件证据，不是 flow 数量。

**突破监测：仍然没有算法突破。** 新增的是“证据不会在崩溃或发布时说谎”的工程里程碑。真实 private unit、metric
scorer、harm、16-unit score、未见 rig/session field-L2 与逐相机尾部都还没有产生；因此不能把 `483 passed` 写成
三维重建优于 DeepONet/FNO/FFNO，更不能写成论文性能结果。

## 138. 半文件不能当记录，整树扫描也不能当发布封印

第五轮审计继续沿着“崩溃恰好发生在最差的一行”来查。旧的 recovery record 和 completion 都用最终文件名
直接 `O_EXCL` 写。这在正常退出时没问题，但断在创建和写完之间，就会留下空目录、零字节或半个 JSON。
程序会拒绝它，却会让所有 unit 永久等人工处理。这不是数据伪造，但是真实的可用性与可恢复性缺口。

现在 recovery record 先在 `<entry>.preparing` 目录内完整写入、同步，再在 quarantine 这一个父目录中原子改名。
只有这个 prepared entry 可见后，run/claim 才能开始搬迁。若断在空 stage 或半个 record，因为源还没动，可安全
丢弃并重建；若 record 已密封，就发布原条目续作。completion 也改成 `isolation_complete.json.preparing`，重读精确
等于期望 bytes 后才改名。回归真的模拟了 record/completion 半写、stage 同步、entry 改名、run/claim 改名、目标父目录
已同步和两个父目录都同步后的中断。发布 record 之前还会同步源 run/claim 的成员轴。

这里没有假装 POSIX 突然有了跨父目录事务。run/claim 移动仍需两个父目录分别 `fsync`；现在顺序是先目标、后源，
优先避免 neither/data-loss 窗口，但断电后仍可能需要用 source/destination XOR 把 both/neither 关闭为人工 NO-GO。要得到更强保证，
需要 copy-verify 和各自父目录内的 sibling tombstone，不是多写一句“原子”。

Pages 端的新问题是：扫描通过后到 rename 前，staging 还可能被晚注入一个文件。现在最终扫描会产生全树
`(relative path, size, SHA-256)` 封印；同目录换入后，仍沿原 staging FD 重算，不等就回滚旧 artifact。敏感扫描也不再
只做一遍 URL decode：三层 percent、HTML/JSON/JS、NFKC、Base64 中的私有摘要、换行 Base64 PDF、编码后的账号/密码，
以及藏在公开文件名里的账号和私有摘要都有确定性反例。

最后一轮反例还要求正式输出名在 seal 后和父目录 `fsync` 后各重绑一次；seal 计算本身异常、输出名被换绑或整树不等，都进入同一回滚路径，上一份完整 artifact 不会被异常新树占位。最终扫描也重新检查路径名，长/嵌套 Base64 和全大写真凭据不再被当成占位符。

超过 8192 个 Base64 字符或超过三层可检测解码深度的内容现在不再被略过，而是直接让发布包 fail closed。

本轮实际回归为 Pages builder `68/68`、data-foundation runner `53/53`、三组定向合同 `129/129`、E73 聚焦
`137/137`、相关链 `90/90`、bounded fast matrix `514/514`。这些仍是软件合同和崩溃模型内的证据，不是 137 个 flow。

**突破监测：尚无算法突破。** 这是一个值得标注的工程里程碑：它让未来的真实分数更难被半文件、拆分事务或
发布竞态污染。但真实 private unit、24-metric scorer、harm、16-unit score、fresh rig/session、field truth 和任何方法相对
DeepONet/FNO/FFNO/强经典基线的优越性仍然全部为 0。

## 139. 先把“怎样算输赢”写成代码，再谈让模型上场

上一节留下的下一步很具体：七个非零 CGLS checkpoint 怎样变成 24 个不可随意改口径的
指标。现在 fixture scorer 已经写出来了。它只接受刚刚由 data-foundation runner 在内存中产生的
`y=A(x_proxy)` 和同一条 CGLS 轨迹，按 `k1/k2/k3/k4/k6/k8/k12` 的顺序每个只做一次 forward。
进入 scorer 时 operator 计数必须正好是 `13 A + 13 Aᵀ`，出口必须正好是 `20 A + 13 Aᵀ`，
任何多跑或少跑都拒绝发布。

24 个指标没有再藏在一句“看重投影误差”里。现在是四个全局投影量、九个 view slot 各两个
L2/p95 尾部量，再加 field 和 gradient relative-L2。view id 必须严格是 `0..8`，各 slot ray count
必须正好覆盖整个 operator，所以不能把不好看的一台相机并进 pooled mean 里。所有 harm 只有一个
符号：`candidate metric - cgls_k4 metric`，正数就是 candidate 更差。

真值、观测与每个 checkpoint 现在都有内部 bytes/hash 绑定，并在任何评分 forward 前检查；分母 floor 与 p95 的线性插值也已冻结，不能通过调用参数换尺子。我特意用一个“把 k12 替换成精确 truth，同时更新对应 fixture payload”的单元测试检查符号：这时 24 个误差都必须为 0，
它相对非零 k4 的 field/gradient harm 必须为负。这只是“尺子没拿反”的测试，不是说 k12
或任何新模型已经胜出。非有限 checkpoint、真值/观测/checkpoint 绑定漂移、额外 operator call、view 顺序漂移、非 fixture 状态和非冻结分母
都已有 fail-closed 回归。

fixture metric/harm 还能发布到一个由当前用户拥有的 `0700` 目录。它从根目录逐级 no-follow 打开，先写完 `0600` staging、`fsync`，再用 no-replace hard link 发布固定名称，最后按同一 inode 重开对比完整 bytes、owner、mode 和 link count。异常清理只会删除与本次 staging inode 一致的成员。已有 bundle 不能覆盖，符号链祖先也不能用。当前它明确是 fixture-only，尚未与真实 private finalizer
集成。

独立审查还找到三个会让“失败得不够早”的问题：后置 checkpoint 形状错时已消耗前几次 forward，同样 `13/13` 计数的另一 operator 轨迹可被误配，首次 staging `fsync` 失败会留下 preparing 文件。现在所有 checkpoint 的形状和 payload 在第一次评分 forward 前全部通过，foundation 必须持有同一 operator 实例，staging 在写入前就记下 inode 以便故障清理。另外加了不依赖 scorer 内部公式的 NumPy oracle，独立重算 pooled/per-view L2、p95/RMS、field 和 gradient 四类指标。

最后一次复核又把窗口压缩到 staging 刚创建、还没有记下 inode 的瞬间。对初始 `fstat` 失败和 `umask 0777` 各做一个故障注入后，现在两条路径也不会留下空 `.preparing` 成员。

定向 scorer 测试是 `19/19`，runner + scorer + Phase-0 contract 是 `80/80`，E73 聚焦集 `156/156`，相关链 `109/109`，bounded fast matrix `533/533`。这些数字均在文件冻结后实际重跑，没有按测试数做加法。

**突破监测：仍无算法突破。** 这一步只让未来的模型无法靠合并差相机、改 harm 符号或漏算 `A/Aᵀ`
来制造胜利。下一步是把 scorer 与 private foundation 绑到同一受控进程，然后做不计分 preflight。
真实 16-unit score、predictor、fresh rig/session、DeepONet/FNO/FFNO 同门比较和论文性能主张仍然全部为 0。

## 140. GCT-KMix：知道答案后随便混也不够好

这次我没有先训练一个新网络，而是先问了一个更狠、也更省时间的问题：假设我们已经知道每个
synthetic case 的三维真值，能不能在同一条零初值 CGLS 轨迹的
`k={1,2,3,4,6,8,12}` 七个 checkpoint 之间，事后挑出最好的凸组合？如果连这个
“作弊版上限”都不够好，就没有理由再花几天训练一个只能猜这些权重的 MLP。

原来的 GCT-KMix 还要求每条 ray 的二维残差模不能超过 k4 安全包络。它是一个凸二阶锥问题，
但不能只看 solver 写了 `success` 就当成全局答案，所以我固定使用会同时报告 primal/dual 的
Clarabel。第一次 18-case 运行有一个 case 是 `AlmostSolved`，gap 为 `2.60e-8`，超过冻结的
`1e-8`。我没有把它四舍五入成成功。第二次把每条 ray 约束两边同时除以正 limit，数学可行域
不变，只改善数值尺度；结果仍有两个 `AlmostSolved`，而且 projection closure 也超门。因此两个
attempt 都原样保留为 `GCT_KMIX_FAIL_CLOSED_SOLVER_OR_CONTRACT_FAILURE`。第二次不是“修好后算法
输了”，而是“仍没拿到可发布的 tail-safe solver 证书”。

真正让路线可以停止的是一个更简单的集合关系。记只要求权重非负且和为 1 的集合为
`F_simplex`，再加逐 ray 安全约束的集合为 `F_tail`，显然：

`F_tail subset F_simplex`。

删掉安全约束只会让选择更多，所以 unconstrained truth oracle 的最小 field error 一定不大于
tail-safe oracle；换成 gain 说，它是任何 tail-safe 混合都不可能超过的乐观上界。这一支没有
退化：18/18 个 unconstrained oracle 都是 `Solved`，最大 relative primal-dual gap 只有
`7.25e-11`，field metric 与 conic objective 的最大核对误差为 `3.89e-16`。

逐 case 先算相对 GCT-KSelect 的 gain，再在 split 内取均值，得到：

- development：乐观上界 `2.3334%`；
- public exploratory OOD：乐观上界 `1.9649%`；
- 结果前冻结门：两个 split 都必须至少 `5%`。

因此即使 tail-safe solver 完美收敛，它也不可能通过 field 门。总 gate 是合取，field 门已经足够
否决当前表示，所以结论是
`ZERO_START_FIELD_HEADROOM_UPPER_BOUND_NO_GO_LEARNER_NOT_AUTHORIZED`。这没有修改两个原始
fail-closed 包，也没有拿它们的近似权重当证书。

这件事的物理含义比“模型没调好”更重要。零初值 CGLS checkpoint 都来自当前观测算子的
row-space/Krylov 轨迹；在它们之间混合，只是在调谱滤波和停止位置，不能凭空创造有限视角下不可观
的 near-null 成分。M2.2 的 exact oracle 已提示 null-space 有明显三维 field headroom，所以现在
应该改变信息来源，而不是继续换 selector 名字：

1. 冻结一个已有 learned field 作 warm start，并明确支付它的初始 projection 和推理成本；
2. 在同一总 `A/Aᵀ` 预算下生成 warm-start CGLS 轨迹，先算新的 convex-hull truth ceiling；
3. 只有 ceiling 同时过 field、H1 和逐 ray/camera 门，才训练 observable weight predictor；
4. 再用 matrix-free measurement-space row-removal 检查 correction 是否真留在 near-null 方向；
5. 最终仍需要 fresh geometry/rig-session、独立 renderer 和 OERF 数据，synthetic gain 不能升级为突破。

这里的 JACRU 只是“连续解析梯度造观测、体素有限差分/三线性插值做 inverse”的窄义
inverse-crime barrier。18 个 case 只有 6 个 geometry clusters，每个 geometry 下三种 morphology
共享相机，不能当成 18 个独立 rig。它足够帮我们淘汰一个低上限表示，却不能证明真实 BOST、
DeepONet/FNO/FFNO、NeRIF/TDBOST 或所有算子学习都失败。

**突破监测：仍无算法突破。** 新增的是一张可信的“此路不值得训练”路标。下一次只有 fixed
learned warm start 真正把 zero-start 没有的信息带进来，并在未见几何、强经典/神经基线、逐相机
尾部与完整成本上共同过门，才可能出现算法级信号。

## 141. Warm start 的场看起来很好，但它没来得及把测量对上

上一节说要换信息来源，所以这次真的把四类已经训练好的模型当成固定 warm start 来测了。为了不让
结果出来后再改模型，我先把 JACRU-M2、pooled CNN、grid DeepONet 和 pooled FNO 的三个随机种子，
共 12 个 checkpoint 存成不允许 pickle 的 NPZ。每个文件、权重语义、训练分区、运行时版本和 FNO
额外 metadata 都有哈希。正式评分器里没有 optimizer、backward 或 `_train_one`，只准加载这些字节。

这里有一个容易被忽略的成本：模型不是看一眼原始位移就直接给答案。它先用 CGLS-12 做 base field，
再算 terminal residual 和每相机 adjoint lift，总共已经花掉 `13F/13Aᵀ`。所以 warm start 的 `k=2`
不是“只算两步”，而是总计 `16F/15Aᵀ`；同预算的强 CGLS 可以从零开始算 16 步。所有比较都按这个
总账走，没有把网络输入当免费午餐。

结果第一眼很诱人。JACRU-M2 在 development 的 `k=2` 相对每个 case 三种经典法中 field 更好的那一个，
平均 field-L2 改善 `45.734%`，H1 改善 `40.888%`；pooled CNN 也有 `44.045%/39.427%`。但同一行的
measured/independent-clean reprojection ratio 是 `14.5965/3.2276` 和 `14.4060/3.2206`，门是
`<=1.10`。也就是说，三维场先验猜得像，却还没有把当前相机真正测到的位移对上。

把 k 加到 10 后，JACRU 的 clean ratio 已降到 `1.096`，field/H1 仍有 `38.87%/33.87%`；但 measured
ratio 还是 `13.509`，18 个 model-seed × case 单元里有 `16.7%` 出现超过 1% 的 field harm，worst
也降到 `-5.8%`。这不是再挑一个 k 就能解决的擦线问题。四种架构、11 个 k 一共 88 个 decision cell，
没有一个 development 全门通过，所以状态是：

`M2_9_FIXED_WARM_CGLS_DEVELOPMENT_NO_GO`。

为什么场和测量会分家？这个 toy 里每个 geometry 约 150 个测量，却有 1000 个 active voxel。网络能从
训练分布带入测量本身无法唯一决定的场结构；CGLS 更新只改 `range(Aᵀ)` 部分，离散 exact-kernel 分量会
被保留。数值审计确实看到最大 kernel drift 只有 `1.867e-14`，递推 residual 与重新 forward 的最大误差
`1.073e-13`，所以不是代码悄悄改了核分量，也不是 CGLS 算错了。但“保留了核分量”只说明线性代数按
预期工作，不说明这个核分量就是 field gain 的原因，更不等于真实光学零空间。

同调用 CGNE 还帮忙排除了另一种误解。两者从同一个 learned field 出发、花同样 refinement 调用；在
development `k=6`，warm CGLS 相对 CGNE 的 field/H1 大约再好 `3.4%/3.7%`，measured residual 约为
CGNE 的 `0.56`。所以 CGLS 的有限步选择是有用的，只是它弥补不了前面 13 对调用留下的巨大 data gap。

真正应该改的是 warm-start 接口，不是继续扫 k。下一候选先压缩输入成本：直接用 raw displacement、
相机/ray geometry 和最多一次 pooled `Aᵀy`，不再先跑 CGLS-12。如果 proposal 只花 `0F/1Aᵀ`，再付
一次 `Ax0`，同样 24 对预算里就能给 CGLS 留 23 步，而不是现在的 10 步。最小比较是：

```text
lean learned warm: 0F/1Aᵀ feature + 1F projection + CGLS-23 = 24F/24Aᵀ
strong control: zero-start CGLS-24 = 24F/24Aᵀ
```

如果同一 rig 有很多连续帧，还可以把 row/near-null basis 的一次性计算按帧摊销；但 setup、每帧成本和
break-even 帧数都要报告，换 rig 不能免费沿用旧 basis。再下一步才是仅用 held-out ray、noise floor 和
geometry feature 做可拒答 gate，失败就回退强 CGLS。

这次独立 validator 没有调用正式 runner 的聚合函数。它从 CSV 重算 792 条 baseline、3168 条 candidate、
2880 条 CGNE、264 个 seed aggregate 和 88 个 decision cells，10/10 输出哈希通过，`errors=[]`；另一个
只读审计也得到同一 NO-GO。

**突破监测：仍无算法突破。** 但这张负结果很值钱：它把下一模型的创新问题从“换哪个 backbone”缩成了
“怎样以极低 `A/Aᵀ` 成本带入 BOST 场先验，并在相同总预算下把测量一致性和坏尾一起闭合”。这才是后续
低成本 geometry-conditioned operator、rig-amortized basis 和 observable fallback 应该共同回答的问题。

## 142. 先别训练：我发现“零 correction 等于 CGLS-24”原来没那么简单

上一节的草案是：一次 `Aᵀy` 交给小网络，再把网络输出当 warm start 跑 CGLS-23。账面上看正好是
`24F/24Aᵀ`。但独立审计提醒了一个很要命的小坑：如果先形成一个非零初值再重启 CGLS，哪怕网络输出
的 correction 恰好为零，也不严格等于从零开始连续跑 24 步 CGLS。因为“重启”把原来 Krylov 递推的
方向状态丢掉了。这样以后如果模型路线赢一点，我们分不清是网络有用，还是 solver 换了。

所以这次没有急着训练。我先写了一个叫 `LGWO-A24` 的安全壳。网络不直接给完整重建场，只允许对第一条
方向 `Aᵀy` 加一个很小、有范数上限的扰动。第一步沿这个方向做精确线搜索；后面 23 步每次花一对
`A/Aᵀ`，并在测量空间把新方向对历史方向做两遍重正交。这样 correction 为零时，它就退化到全重正交
形式的 CGLS-24，而不是一个“看起来差不多”的重启算法。

当前安全壳已经有 9 个测试通过：

1. 总账严格是 `24F/24Aᵀ`；
2. 零 correction 与 CGLS-24 的终点差只有约 `1e-16`；
3. 每一步 measurement residual 不增加；
4. correction 会被限制在 `eta ||Aᵀy||` 内；
5. 输入越出 calibration envelope 时精确回退到零 correction；
6. proposal 如果偷偷调用一次 operator，立即报错停止；
7. API 没有 truth、family 或 split 参数。

这仍然不是算法成果。它只说明以后训练出来的任何差异都有更干净的归因：零 correction 是强基线，网络
只能靠一个受限方向扰动创造增量，不能靠换 solver 或偷物理调用。

数据路线也重新分级了。公开世界里暂时找不到同时有真实 3D/4D BOST、多视角标定、可重算 `A/Aᵀ`、
三维真值和明确许可证的完整数据集。下一步不能拿普通 CT 假装真实 BOST，而是四层推进：STEMPO 小型动态
3D CT 只检查三维时空算子；MILD CH4/H2 DNS 提供 `133x83x66` 的密度/温度/组分，构造
`physics-derived synthetic BOST`；TU Graz HBOS 负责真实背景、光照和位移前端；最后才接 OERF 的真实
geometry、flow-off repeat 和连续帧。

**突破监测：仍无算法突破。** 真正新增的是一条干净、可被证伪的算法接口，以及一条不会混淆证据等级的
数据桥。下一次只有小模型在同 `24F/24Aᵀ` 下同时过 field、H1、measured/clean/held-out reprojection、
逐 rig 尾部和 harm，才把状态从 `PROPOSED_UNRUN_CANDIDATE_NO_CLAIM` 往前移动。

## 143. 理想的零空间方向确实有用，但网络还没有学会任何东西

安全壳写好后，我没有马上训练，而是先做了一个“答案上限”测试：假如评估器直接告诉我们真值里哪一部分
完全不会改变当前相机测量，把这部分当作第一方向的小扰动，LGWO-A24 到底有没有改善三维场的空间？如果连
知道答案都没用，那训练网络只是在烧时间。

这次用了 6 个已经打开的合成 case、3 套相机几何。每个 case 都从真值里精确拆出 row-space 和 null-space
方向，再分别塞进同一个 `24F/24Aᵀ` 壳层。所有方向都有相同范数上限，不允许某一臂拿更大的 correction。

结果很清楚：给 exact-null truth direction 时，`eta=0.05` 的平均 field/H1 改善是
`+6.978%/+6.566%`，6 个 case 里最差也有 `+3.741%`；`eta=0.10` 是
`+13.722%/+13.005%`，最差 `+7.262%`。因为方向在离散 kernel 里，measured 和同一 `A` 下的 clean-target residual ratio
都保持在数值意义的 1。

对照组更重要。只给 exact-row truth direction，measurement residual 确实下降了，但 field/H1 平均反而
略差，大约 `-0.01%` 到 `-0.03%`。不过三条方向是各自归一化后再 clipping，完整方向并不等于另外两条
applied direction 的和，所以不能拿 full-row 计算 null 的“独立因果贡献”。最窄的解释只是：当前 norm-matched
方向屏支持继续研究测量看不见的场结构，不支持只堆 row-space 拟合，也没有证明一般 kernel 因果。

我又写了一个不导入正式 runner 的 validator。它重新造 6 个 case、3 个 dense projector，重跑 36 条路径，
对每行指标、聚合门、调用账本、哈希和关闭的授权字段做了 1,121 项断言，全部通过。

它和正式运行仍共享底层 fixture、projector、solver 和 metric，所以证明的是“同一数学实现可确定复现”，
不是第二套光学模型的独立验证。完整实验还花了 dense setup、评分和三次 SVD，总账是 `4053F/1008Aᵀ`；
`24F/24Aᵀ` 只是一条 solver path 的部署预算，不能冒充整个 oracle 实验的端到端成本。

但这里有三层刹车：第一，exact-null direction 直接偷看了三维真值，部署时根本拿不到；第二，配置冻结前已经
看过其中一个 case 的几个 eta，所以这是 opened screen；第三，`12^3`、三相机和特定 support 的离散 kernel
不等于真实 OERF 光学系统的零空间。

**突破监测：仍无算法突破，但有可复现的表示层进展。** 现在终于知道小网络应该努力学什么：利用反应流形态、
相机/ray geometry 和跨帧相关性，猜一个 approximate-null prior；同时必须在未见 rig 上知道何时猜不出来并
精确回退。下一步只做不超过 8k 参数的小 pilot，先证明“能从可部署输入学到一点这个方向”，而不是直接堆 FNO。

## 144. L1 不是“开训按钮”：先把 B 射线、三随机种子和坏尾锁死

O1 给了一个值得追的信号，但它还是偷看真值的答案上限。为了不把“有上限”直接写成“模型能学会”，这次把
下一步改成了机器可验证的 `LGWO-A24-L1` 协议。模型只有 2,729 个参数，输入只有 A 侧的 noisy displacement、
一次 `A^T y`、A geometry 和 support。fit、early-stop、route 分别固定为 24、6、24 个 cases；每个 geometry
cluster 都含 smooth plume、single interface、shock-expansion pair 三种形态，并固定三个 model seeds，不能
训练完只展示最好的一次。

我一开始把 B 设计成另一种 camera pose，但独立审计指出：如果 fit 时用 development-B，而 route-A 也是
development pose family，就等于提前把 route 的相机布局放进训练。现在改为：fit/early 的 A 和 B 都属于
train pose family，但 A/B 使用两套独立 SHA256 geometry seeds；route 才使用 development-A 与 OOD-B。B
只是 evaluator：fit-B 可算辅助损失，early-B 只选 checkpoint，route-B 在所有模型冻结后才生成。模型 API
没有任何 B 字段，也没有 `proposal_kwargs` 可以从 B payload 调模型。

数据量增加并不自动等于证据更强，所以统计单位也先锁死。三个 families 和三个 model seeds 先在同一个
geometry cluster 内平均，真正的独立单位只有 8 个 route clusters。主门要求平均 field gain 至少 5%，三个
seed 各自至少 2%，50,000 次 cluster bootstrap 的 95% 下界大于 0，至少 7/8 clusters 为正；同时 H1 至少
3%，A measured、A clean、B clean residual 均值不超过 1.05，harm 不超过 5%，worst 不低于 -5%。这比只看
72 个 case-seed 平均值更难过，但不会把相关样本冒充独立重复。

代码审计还抓到几个不能带进训练的坑：FP16 的 `1e-20` floor 会下溢并产生 NaN gradient；training proposal
原来能原地修改 live anchor；tiny anchor 的限幅分支和 deployment 不一致；A24 甚至允许传入 checkpoint 25。
现在 training state 只准 float32/float64，proposal 收到的是 clone，tiny branch 与 deployment 对齐，超过 24
直接报错，bool reorthogonalization 也拒绝。异构 batch、恶意原地修改、tiny anchor、三条网络分支的非零梯度
都加入了测试。

零 head 还有一个很朴素的问题：它能完美证明 baseline recovery，但从严格零范数分支开始训练时梯度也会是
零。因此 runner 先用 zero head 做 parity gate，再按当前固定 model seed 对 correction head 做一次且仅一次
`Normal(0,1e-4)` 初始化，bias 保持零；尺度、次数和 seed 都写进 JSON，不允许看到结果后重来。用已经永久
排除的 JACRU train seeds 做工程探针时，真实 `12^3`、三 families、完整 K=24 forward/backward 在当前 Mac
上约 0.08 秒，账本为 `24F/24A^T`，34/34 参数张量都有有限非零梯度。这个数字只是未冻结的工程测量，正式
summary 还要在预数据 commit 后由独立脚本重跑。

**突破监测：仍无算法突破。** 当前真正完成的是“训练前的科学防火墙”：28 个既有 geometry seeds 永久排除，
A/B seed 规则、三模型种子、route 门、权限边界、CPU float64 和完整成本字段都由 validator 锁定。下一步先
提交这个冻结点，再只在排除数据上重跑 implementation gates；通过后才生成 fit，route 继续封存。

## 145. Implementation gate 终于通过，但中间两次失败比 PASS 更重要

预数据 commit 后，正式脚本第一次运行立刻停在 fixture constructor：JSON 里的 `schema` 是协议元数据，不能
直接当成数据类参数。这个错误没有碰到模型，也没有生成科学结果。修复时我没有只加一行 `pop`，而是补了一个
真正执行完整 `12^3`、三 family、K=24 forward/backward 的测试，防止以后只测 helper、主入口又坏掉。

第二次确实走完了梯度路径，却暴露出更隐蔽的问题：同一个 model seed 两次运行的完整 state SHA256 不同。
原因是当时只用 seed 初始化 correction head，前面的卷积层仍使用进程全局 RNG。也就是说结果表面“固定 seed”，
实际上不能跨进程重放。此时 fit、early、route、fresh 都还没有生成，所以做了训练前协议修订 1.1：完整模型在
`torch.random.fork_rng` 内按当前 seed 构造，CPU 上转 float64，且不推进全局 RNG。旧 config 哈希和修订原因都
保存在 JSON 里，不能把它假装成原来就设计好了。

第三次在 source commit `5230a5f` 上通过。A solver 账本严格为 `24F/24A^T`，B evaluator 为 `1F/0A^T`，
34/34 参数张量都有有限非零梯度；完整模型 state SHA256 是 `8d1629b1...17028`。另起进程重跑后，除
wall/CPU/RSS 外所有字段逐项相同。独立 validator 不导入 runner，重新构造模型 state，复核配置哈希、来源 commit、
loss 算术、调用账本、route/fresh 文件系统扫描和所有 claim flags，共通过 57 项断言；checksum 也已封存。

状态名特意没有写“fit 已授权”，而是
`PASS_IMPLEMENTATION_GATES_FIT_RUNNER_IMPLEMENTATION_AUTHORIZED_ROUTE_SEALED`：现在只可以继续写 fit-only runner。
正式 loss 的 H1 stencil、epsilon、normalization 维度、五个 trainable arms 和 checkpoint/resume 合同还没全部实现，
所以科学 partition 仍保持 0 个 materialized cases。

**突破监测：无算法突破。** 这次的价值是把三种很容易伪装成“训练已经开始”的问题提前抓住：入口没跑通、
随机种子不完整、工程 smoke loss 冒充正式 loss。下一步先把 fit runner 的合同和测试补齐，再生成 24 个 fit cases；
route 继续关着。

## 146. 训练栈终于能互相对账，但这仍不是“模型跑起来了”

这一轮没有立刻生成 24 个 fit cases。我先把上一节剩下的四块拼成了一条可以互相审计的链：五个训练臂、正式
loss、三层缓存和 checkpoint。开始时各模块自己的测试都能过，但主审一接接口就发现两个真问题：`E2` 忘了乘
support mask，意味着 support 外的误差会混进主指标；proposal cache 的字段虽然没有 truth，却可以把一个允许的
引用字符串偷偷指向 `truth.npy`。这两处都在任何科学数据生成前修掉了。

现在五个 arm 是精确 registry：`full=2729`、`fixed_direction=1000`、`g_only=681`、
`no_raw_observation=2585`、`no_geometry=2225`。fixed-direction 不再用平值 `topk` 随机决定参数对应哪个 voxel，
而是要求 binary support 内恰好 1000 个 active voxels，按 flattened z-y-x 顺序固定映射；不同 batch 的 index map
不一致就直接停。它仍是容量不同的消融，不能拿来宣称结构优越性。

正式 loss 也不再接受“空数组等于零损失”。field-L2 只看 `support>0.5`，H1 只看两个端点都 active 的物理
spacing forward edges；A measured、A clean、B clean、`A delta` 和 `Ag` 必须 nonempty 且 ray shape 一致。
少付一次 B forward 或把 projection 留空，loss 会 hard fail，而不是悄悄给一个好看的零惩罚。

缓存现在有 proposal、training label、heldout-B evaluator 三个 resolve 后互不重叠的根。科学 proposal 只能从
typed `JACRUInferencePayload` 入口生成，手工数组 writer 只允许 engineering artifact；每个 fit triplet 外绑真实
source commit、config hash、fit manifest hash、entry hash 和 order。NPY/JSON 的 dtype、shape、hash、symlink、
path traversal、pickle/object array、未知文件和覆盖都会独立重读检查。它不能从哲学上证明 observation 的数值绝不
可能被恶意编码成 truth，但已经把正常 runner 的科学写入能力限制在 truth-free inference API 内。

checkpoint 则固定为 epoch `0..30` 共 31 个节点。epoch 0 是 optimizer 尚未产生状态的零成本初始化；随后每个
epoch 必须精确记录 8 个 cluster batches，即 batched API `192F/192A^T + 8B-F + 8 optimizer steps`。完整 AdamW
参数、arm registry、parameter/buffer state schema、metric-history parent chain、route/early seal、wall/RSS 和
nonfinite/fallback 都是硬字段。一个独立 chain validator 会重算 30 个 epoch 的累计 `5760F/5760A^T + 240B-F`
和 240 optimizer steps，不能删失败 epoch、插入第 31 个 epoch 或改成 best-seed selection。

协议因此在 scientific case 仍为 0 时修订到 `1.2`。这一修订没有增加任何成功 claim，只把代码里已经实现的
arm、loss、cache、optimizer、checkpoint 和成本规则写进 canonical JSON。聚焦回归目前是 `123 passed`；旧的
implementation evidence 因为 critical source 已改变，会主动失效，必须等新代码提交后在新 commit 上重跑，不能
拿 `5230a5f` 的 PASS 替当前训练栈背书。

文献边界也更清楚了。NPN 已经在做 learned null-space projection，Neural Correction Operator 已经研究 learned
inverse correction，FCG-NO 已把 neural operator 放进 flexible Krylov 迭代；所以“网络学一个零空间/修正方向”本身
不能再当创新。LGWO 仍可能成立的窄命题是：在严格不增加 `24F/24A^T` 部署预算的前提下，只修正首方向，并在
未见 rig 上用 observable envelope fail closed，同时把坏尾、B-ray consistency 和总成本一起过门。

**突破监测：仍无算法突破，也还没有训练结果。** 真正的进展是训练栈现在更难自欺：旧证据会因代码漂移失效，
空监督不会被当成零损失，错误 arm 不能冒充 2729 参数 full，缓存也不能靠一个漂亮 descriptor 掩盖文件层问题。
下一步只有在新 commit 上完成独立 pre-fit evidence 后，才允许第一次 materialize fit；route 和 fresh 继续为 0。

## 147. 现在把“准备训练数据”和“真的开始训练”分成了两把钥匙

上一节记录的是 protocol 1.2 的训练栈快照。这一轮独立复查又发现：即使每个模块单独能过测试，runner
仍可能在同一个目录里一边写数据、一边给自己签“可以训练”的许可；成本表也可能只报一次 batch API，
却不说这个 batch 实际包含三个 case。为了让这种自我授权和少报成本都过不了，协议在 0 个科学 case、
0 个 optimizer step 的时候修订到了 1.3。旧的 implementation evidence 因为源码已经变化而失效，
必须在新的干净 commit 上重做，不能拿以前的绿灯继续通行。

先把一个容易说错的数字讲清楚。体网格是 `12 x 12 x 12 = 1728` 个 voxel，但最外一圈被固定为边界，
真正参与重建的是中央 `10 x 10 x 10 = 1000` 个 voxel。现在 A 相机和 B 相机的 operator 都必须在
第一次调用前绑到这一个 exact support；另一个同样有 1000 个点、但位置挪过的 mask 也不算。固定方向
消融的 1000 个参数按全局 z-y-x 顺序逐个对应这 1000 个位置，不能随 case 重新编号。

第一把钥匙叫 Stage 1，只准准备 24 个 fit cases。它要在仓库外绑定干净 source commit、protocol report、
pre-fit evidence、24-entry manifest、独立第三次 GO 审计和唯一输出目录。拿到它以后，materializer 只能生成
24 组 proposal、label 和 heldout-B cache，再从 proposal 侧的 24 组可观测量计算 normalization 和 25%
calibration envelope；它不能构造 optimizer，也不能生成 early、route 或 fresh 数据。

这里又补了一道很实在的防伪门：pre-fit evidence 不能只收一张我们自己写的“349 个测试都过了”汇总表。
正式 clean-commit 回归必须保留 pytest 直接生成的 raw JUnit XML。packer 会给 XML 原始字节做 hash，
逐条读取 `classname::name`，拒绝 failure、error、skipped、重名或漏掉的 negative-gate testcase，再把
排序后的 testcase identity list 单独做 hash。Stage 1 会连同 protocol report、test manifest、test results、
raw XML、implementation-gate report 和独立 validation 共六份输入一起绑定，所以事后改 JSON 数字、
沿用旧 implementation 快照或删掉一条失败测试都对不上。

第二把钥匙叫 Stage 2，只在这些 fit 文件已经冻结后签发。它会逐个检查 72 个 cache manifests 和 216 个
数组的双重 hash，重新绑定 normalization、calibration、fit manifest 与 operator-ledger schema，并确认
optimizer 仍然一步都没走。以后每个 checkpoint 还要带上 Stage-2 root 的 hash；checkpoint 目录外另存
一个 node anchor，避免一个目录同时篡改模型、指标和账本后仍然“自洽”。

成本现在也要用两种口径同时说。一个 cluster batch 含三个 family，因此每个 epoch 的真实 batched API
是 `192 A-F + 192 A-A^T + 8 B-F`，对应的 case-equivalent 工作量是
`576 A-F + 576 A-A^T + 24 B-F`。真实 autograd 探针进一步测得反向图是 552 个 A-F、576 个 A-A^T
和 24 个 B-A^T case-equivalents，而不是手填成对称的 576/576；另有 8 个
optimizer steps 和 392 条显式 ledger events。两张表、底层 operator counter 和 checkpoint metrics
必须互相对得上；少报其中任何一张都不能过门。

checkpoint 也从 1.0 升到 1.1。每条 arm/seed 仍是 epoch 0 到 30 的 31 个连续节点，但现在它不只检查
文件 hash，还会按 arm 名和 model seed 重新构造 exact registered model，核对参数顺序、AdamW slots、
双成本账本、Stage-2 授权和外部 node anchor。换 arm、换 seed、删掉失败 epoch 或拿另一条轨迹的 optimizer
state 接着跑，都会直接停。

本次真实复核分成两层。独立 protocol validator 对当前 config 做了 180 项断言，全部通过；它确认
canonical config SHA256 已在私有证据中绑定，
scientific case 为 0，L1 route/fresh 文件命中为 0，所有成功 claim 都是 false。随后跑 18 个文件的
聚焦 pytest，当前结果是 `349 passed`，fit runner、Stage-1/materializer、授权根、evidence packer、implementation gate、checkpoint、
ledger、normalization、cache、loss、arm 和 protocol 测试都已全绿。

所以现在最诚实的状态是：接口红灯已经修清，科学数据也没有被污染，训练没有偷偷开始；但仍不能把
“349 passed”直接理解为可以开训。刚才的交互式回归是当前工作树检查，正式 Stage 1 还需要先形成干净
commit，再用 pytest 输出 raw JUnit，重建 raw-input-bound implementation evidence 和 checksums，取得
独立 GO，然后依次签 Stage 1、物化 24 个 fit cases、签 Stage 2。到正式运行时 testcase 数若变化，必须
以新 JUnit 为准，不能抄用这里的 349。

## 148. 加速不是跳门：把实现证据也纳入 Stage 1 原始证据

这次“火力拉满”没有直接启动训练，因为并行审计发现一个会让后续结果失效的接口空洞：pre-fit packer 已经
生成 implementation-gate 的工程报告和独立 validation，但 Stage 1 授权根仍只绑定 protocol、test manifest、
test results 和 raw JUnit 四类输入。换句话说，旧工程报告有机会被当成新提交的证据。

现在 protocol 1.3 明确冻结六项输入清单，并要求 implementation report 使用 1.1 schema、独立 validation
使用 1.0 schema、至少 57 项断言，而且必须在授权所指向的干净 commit 上重跑。canonical hash 因此更新为
该私有 config SHA256；协议仍为 180 项断言，完整聚焦回归为
`349 passed`。fit runner 的测试还故意把真实 `AdamW.step()` 设为硬失败，所以这些数字只说明调度、
授权、调用账本、checkpoint 和负向门可工作，不说明模型已经训练。

并行的新颖性审计又核对了 DCDM、FCG-NO、Deep Null Space Learning、NPN、NeRIF、NeDF 与 TDBOST 等一级来源。
“神经 Krylov”“学习零空间”“AI-BOST”都不能作为宽泛新颖性。当前唯一值得实证的窄问题，是 observable-only、
geometry-conditioned、norm-bounded 的 first-direction correction，能否在固定 24F/24A^T 壳内改善未见 rig
尾部且不破坏 A/B consistency。这个组合目前只能叫 `possibly differentiating, not yet evidenced`。

**突破监测：仍无算法突破，0 scientific cases，0 optimizer steps。** 当前下一门是干净源码提交后的
第三次独立 GO；只有它通过，Stage 1 才能物化 24 个 fit cases。

**突破监测：无算法突破，0 scientific cases，0 optimizer steps。** 这次增加的是更严格的授权边界、
1000-voxel 物理 support、fit-only normalization、双成本账本和可外部追溯的 checkpoint 链。它们不能替代
模型性能，但能让将来真的出现 gain 时，我们更有把握知道它来自算法，而不是数据泄漏、少算成本或事后换规则。

## 149. 这次真的加速了什么：把“少算一次”从论文账本里抓出来

这轮没有多造一个网络名字，而是先检查训练时到底调用了多少次物理算子。原账本把反向传播想当然地写成
对称的 24/24。真实 PyTorch autograd 探针显示：第一个 `A^T y` 只由固定观测生成，不需要向前追梯度；后面
23 个 `A^T r_k` 才依赖模型输出。因此每个三 case cluster 的反向图实际是 23 次 A-forward 等价、24 次
A-adjoint 等价，再加 held-out B loss 的 1 次 B-adjoint 等价。扩成每个 epoch 后是 `552 / 576 / 24`。
这个差别不大，却很重要：以后和 DeepONet、FNO、NeRIF 或任何自有模型比总成本时，不能靠“看起来对称”填表。

第二个修复是把 Stage 2 从“相信 materializer 说已经完成”改成“runner 自己重新验货”。现在它会重新遍历
完整物化树，核对 summary、checksums、72 个 manifests、216 个 arrays、24 条有序记录、normalization、
calibration、Stage-1 source inventory，以及前后 optimizer step 都为 0。summary 即使被重新计算 hash，
只要其中一个 artifact、triplet 或源码来源和 cache 对不上，授权和 runner 都会停。

第三个修复是证据链。protocol report、test manifest、test results、raw JUnit、implementation report、
independent validation 六份原件都会进入包；validation 同时绑定 report 文件字节 hash 与规范 JSON hash。
现在 18 文件聚焦回归是 `349 passed`，协议是 180 项，规范配置 hash 是
同一份私有 config SHA256。

下一次真正有科学信息的动作不是继续扩建门禁，而是把
`docs/n5_d5_advisor_first_contact_2026-07-19.md` 发给何远哲师兄，确认真实 forward callable、curved/straight
residual 在哪一层形成、JVP/VJP 是否可用、标定与单位、数据 split、现有基线和组内最痛的失败。没有这些
答案，synthetic LGWO 只能保持候选算法，不得冒充 OERF 结果。

**突破监测：没有算法突破。当前增量是可信成本与证据闭环；scientific cases = 0，optimizer steps = 0。**

## 150. 第三次审计踩住刹车：绿灯测试不等于断电后还能续跑

这一轮独立审计专门问了一个很朴素的问题：训练刚更新完参数，电脑就在写文件时断电，第二天到底能从哪一步
继续？答案是当前还不能可靠回答。runner 先执行 `optimizer.step()`，随后分别写 operator ledger、checkpoint
和 external anchor。checkpoint 自己已经会 staging、`fsync` 和原子 rename，但这三份证据合在一起还不是
一个事务。

所以可能出现四种诚实但麻烦的状态：参数在内存里更新过却没有磁盘记录；ledger 比 checkpoint 多一个 epoch；
checkpoint 已经提交但 anchor 还没有；或者最终 ledger/anchor 文件只写了一半。现有验证会拒绝这些不一致，
不会假装训练成功，这叫 fail-closed；但它也不能自动恢复，这不叫 crash-safe。

这件事被标为 P0，意味着第一次正式 scientific optimizer step 仍禁止。不过我们不马上再造一套庞大的安全
系统。下一动作仍是把 `docs/n5_d5_advisor_first_contact_2026-07-19.md` 发给何远哲师兄，先确认组内真实
callable、curved/straight residual、JVP/VJP、标定、数据和强基线。只有师兄确认这条 LGWO 路线值得真实
训练，才实现最小 epoch staging transaction、独立 anchor、原子 TIP 和四个故障注入点。

完整审计和恢复门见
[LGWO-A24-L1 epoch 事务与恢复独立审计](lgwo_a24_l1_epoch_transaction_audit_2026-07-20.md)。

**突破监测：没有算法突破。新增的是一个会阻止伪成功的 P0 风险证据；scientific cases = 0，optimizer steps = 0。**

## 151. 349 项全绿之后，复审仍成功伪造了“240 步”

另一位独立审计者没有继续看页面，而是直接尝试欺骗 checkpoint 链。他发现现有测试只真的执行了一次
`optimizer.step()`，然后把同一份 AdamW state 写进 epoch 1 到 30；因为验证器只要求 slot 的 step 是正整数，
这条链仍可能自称跑满 240 步。恢复代码又根据 epoch 号推断已经完成的步数，于是“文件彼此一致”不等于
“训练确实执行过这么多步”。

第二个漏洞更直接：独立 chain validator 只检查 checkpoint 里填写的 operator-ledger hash 互不重复，
没有打开真实 ledger 文件。测试可以凭空写 30 个 64 位十六进制字符串，就让 11,760 条事件和 240 步看起来
存在。第三个 P0 仍是上一节的 epoch 跨事务崩溃窗口。

这也解释了为什么工程测试数量不能当论文结果。当前只立即修两个边界清晰的洞：要求 AdamW slot
`step == epoch * 8`，并让 chain validator 从明确的 ledger root 逐个读取和复算真实文件。epoch 事务、
external anchor、runtime lock 和 TOCTOU 暂不继续扩建，先由师兄判断真实接口和物理问题是否值得启动这条 fit。

完整 P0/P1/P2 表见
[LGWO-A24-L1 第三轮独立只读复审](lgwo_a24_l1_third_readonly_audit_2026-07-20.md)。

**突破监测：没有算法突破。新发现是证据漏洞，不是性能提升；scientific cases = 0，optimizer steps = 0。**

## 152. 两个“纸面训练”漏洞已关，剩下一个真正的断电 P0

上一节发现的问题现在有了可运行修复。checkpoint 在保存和加载时都要求每个 AdamW slot 的
`step == epoch * 8`；epoch 1 必须是 8，epoch 30 必须是 240，一次 step 冒充完整轨迹会被拒绝。旧集成测试
也因此先失败了一次：它真的只更新一次却保存 epoch 1。把测试改成真实的八个 cluster steps 后才恢复通过。

独立 chain validator 现在还必须收到明确的 operator-ledger root。它只接受 `epoch_01.json` 到
`epoch_30.json`，拒绝缺失、额外文件和 symlink，逐文件复算原始 SHA256，并读取全部 11,760 条事件，核对
run/arm/seed/epoch、sequence、cluster、role、operation、purpose、batch size、API 与 case-equivalent 双成本，
再把每 epoch 推导出的八个逻辑 steps 与 checkpoint 对上。只改 checkpoint 自报 hash 或清空 events 都过不了。

18 文件聚焦回归从 349 增加到 `361 passed`，protocol validator 仍为 180 项，规范配置 hash 仍为
同一份私有 config SHA256。这仍只是当前工作树回归，正式
Stage 1 还要在干净 commit 上重建 raw JUnit 与证据包。

剩下的 P0 是 epoch 跨事务恢复：参数更新、ledger、checkpoint、anchor 还不是一次可恢复提交。现在先把桌面
的“请发给何远哲师兄_真实接口确认_2026-07-20.txt”发给师兄；只有真实接口值得继续，才实现最小事务与故障
注入，不再无边界造基础设施。

**突破监测：没有算法突破。2 个证据 P0 已关闭，1 个事务 P0 仍开放；scientific cases = 0，optimizer steps = 0。**

## 153. 先把问题和失败判据写进论文，而不是先写胜利摘要

这轮没有继续堆模型，也没有启动训练。我们先做了两份以后每天都能真正使用的地图：一份是
[LGWO-A24 论文工作稿](lgwo_a24_registered_manuscript_working_draft_2026-07-20.md)，另一份是
[14 天保姆式学习路线](lgwo_a24_14_day_caretaker_route_2026-07-20.md)。

论文稿不是提前宣布“我们比 FNO 好”，而是提前规定：LGWO 如果要被认为有用，必须在相同数据、相同
`24F/24A^T` 外壳和相同端到端成本下，正面比较 CGLS、简单阻尼/插值、learned warm-start、DCDM/FCG-NO
风格方法以及直接 DeepONet/FNO/iFNO；不能只报平均场误差，还要报逐 rig 尾部、A/B reprojection、harm、
wall time、内存和训练成本。planned figure 只写“这张图要回答什么”，结果格保持空白。这样将来即使方法失败，
失败也会留下可审计的科学结论，而不是临时换指标。

14 天路线则把门槛拆小：先看懂 `Ax=y` 和为什么 `A^T` 不是“把图像倒着算”；再手验 dot-product identity、
跑公开 PSU 的 matrix-free forward/adjoint 与小规模 CGLS；然后读 LGWO 的零 correction 控制、固定预算和
geometry split；最后根据师兄是否能给 JVP/VJP、native residual 和匿名最小几何，选择导数稳定、有限视角
或同精度降成本中的一条真实支线。每天都有口述检查、产物和降级路线，不需要第一天就啃完整 NeRIF 代码。

现在最重要的外部动作仍然很朴素：把
[第一次真实接口沟通单](n5_d5_advisor_first_contact_2026-07-19.md) 发给何远哲师兄。没有 callable、residual
层级、标定、split 与组内强基线，租 GPU 只会更快地产生无法解释的 synthetic 数字。

**突破监测：没有算法突破。新增的是可证伪论文骨架和 14 天执行路径；scientific cases = 0，optimizer steps = 0。**

## 154. 先问第一方向值不值得学：PSU-C1 给出了诚实的 NO-GO

我们本来准备让一个小网络修改 CGLS 的第一条搜索方向。直觉上这很省：网络只给一个小偏转，后面仍由物理算子
做 23 步。但“模型小”不等于“问题值得学”。最便宜的检查是先把三维真值交给一个只用于评价的 oracle，看看它
在固定 5% 修正半径里，最多能给 24 步终点带来多少好处。

第一次运行虽然完成了，却没有直接拿来写结论。独立复审发现六个问题：留出视角可能被静默丢掉；oracle 多算的
一次伴随没记账；图片用真值挑了最好案例；本地几何没有和公开 PSU manifest 锁死；离线 fit 的 96 个
case-equivalent 伴随没报告；失败时还会删掉临时目录。我们把第一次结果标成无效，并且不改方法、随机种子、split、
半径和门槛，只补审计再重跑。

审计版共 1,296 行：6 个描述性 split、每个 24 cases、9 种方法。独立 validator 重新算了 54 个聚合单元，结果
为 `VALID`。两个最重要的数是：truth-only oracle 在 IID 只有 +1.1288% field gain，在 family-OOD 只有
+1.2465%；预先写好的门是每个 split 至少 +5%。所以第一方向不是当前值得扩成神经网络的自由度。linear
observable 虽然有约 +1.03%/+1.05%，但同样没过 2% 均值门，也没有比 fixed direction 多 1 个百分点。

真正让路线发生变化的是 inverse-Sobolev 对照。它在六个 split 上把解析场误差改善了约 42%--53%，held-out B
也明显更好；但 active measured residual 经常更差，family-OOD 达到 baseline 的 4.53 倍。这不是“Sobolev
算法胜利”，而是在告诉我们：24 步未正则化 CGLS 可能正在追噪声或模型误差，控制误差的关键更像是正则化强度和
停止点，不是第一步往哪个方向偏 5%。

下一主候选因此改成一个更低维的 observable regularization/stopping policy。它只看 geometry、noise、残差轨迹、
`A^T r` 和少量 Ritz/Lanczos 标量，只输出受限的 `lambda_k`、固定谱基凸组合或 stop/continue，并且必须通过 active
residual envelope、held-out proxy 和 deterministic fallback。先用 ridge/logistic 证明信号，再考虑小网络；若简单
policy 没有过 fixed-strength、discrepancy、TV/Huber 和 FCG-NO-style 强基线，就停止。

完整数字、图、可复现入口和给何远哲师兄的九个问题见
[PSU-C1 NO-GO 与下一算法路线](lgwo_a24_psu_c1_simple_controls_no_go_2026-07-20.md)。

**突破监测：无算法突破；有真实路线收缩。first-direction fit 未获授权，scientific L1 cases = 0，optimizer steps = 0。**

## 155. 不再“搜一个更大的网络”：把下一题拆成五级可证伪实验

PSU-C1 之后最危险的反应，是看到 inverse-Sobolev 的大 field gain 就立刻训练一个网络预测正则项。文献核对表明，
“从 observation 学正则参数”、hybrid Krylov、flexible regularization、learned CG direction、neural
preconditioner、unrolled data consistency 和 null-space learning 都已有直接先例。只写一个 MLP 输出 `lambda`
既不新，也没有回答 BOST 中最真实的几何、噪声和一致性问题。

因此新建了
[正则化与停止策略文献地图](psu_c1_regularization_stopping_literature_map_2026-07-20.md)。它没有把论文名堆成清单，
而是给每篇标了“在本项目里扮演什么角色、要提取什么、不能借它声称什么”。初学顺序先从 Hansen 的不适定
逆问题和 Chung--Gazzola 的 hybrid projection 开始，再读学习正则参数、warm-start、DCDM、FCG-NO、MoDL
和 Learned Primal-Dual，最后回到 NeRIF、TDBOST、公开 PSU 与 2026 Neural RI Primitives 的组内/新颖性边界。

下一候选暂用工作名 OARS-BOST，但这只是整理实验的标签，不是算法成果。它保留 `A/A^T` 与经典 Krylov 主体，
只允许控制器读取 geometry、noise、残差轨迹、`A^T r`、少量 Ritz/Lanczos 标量和独立 B consistency；输出被限制为
有界正则强度、固定正则基的凸权重、stop/continue/reject。越出校准范围、破坏 residual envelope 或出现异常时，
必须回退到 deterministic baseline。

实验被拆成五级：S0 先比较 fixed-K、discrepancy/GCV 和 hybrid stopping；S1 只用 ridge/logistic 验证可观测信号；
S2 才允许两层小 MLP；S3 才看多步轨迹；S4 必须等真实连续 run 和 TDBOST baseline。任何一级没有独立 headroom，
就停止升级。第一项真正要跑的新实验也很朴素：保存每个 PSU case 从第 1 到 `K_max` 步的 field、A、B、H1、
Ritz 与成本轨迹，先证明“最佳停止步是否真的随 case/geometry/noise 改变”。

这一步把“开发自己的算法”变成了一个可以失败、可以复算、也能向师兄解释的研究问题。现在仍没有训练新模型，
也没有真实 BOST 成绩；下一科学门是师兄确认真实 callable、主误差与强 baseline，然后再冻结 R0/R1 协议。

**突破监测：没有算法突破。新增的是一条经过一级来源碰撞检查的模型阶梯和最低发表门；当前下一步为 R0 半收敛轨迹，不是神经训练。**

## 156. 先问早停有没有标签：答案是没有，但看见了更真实的矛盾

R0 真的把 168 个解析反应场 case 都从 `k=1` 跑到 `k=24`，共保存 4,032 个 checkpoint。原问题很简单：
每个样本的最佳停止步会不会不同？如果会，而且差很多，我们才值得让一个小模型根据 residual、noise、geometry
去猜 `k`。

结果是 test-IID 和 family-OOD 的 truth-field oracle 在 48/48 个 case 上都选 `k=24`。这意味着把真值都交给
oracle，它仍然只会回答“继续跑到最后”。这种标签不能支撑神经早停：网络学得再漂亮，本质上也只是常数 24。
noise discrepancy 在高噪声和 joint-OOD 中确实选出了很多不同的 `k`，但相对 `k=24`，field 分别差约 0.35%
和 0.32%，held-out B 也更差。标签有变化，却不是有用的变化。

更值得继续的是另一组数。IID 从 `k=1` 到 `k=24`，field relative-L2 平均改善 6.41%，但 gradient
relative-L2 平均恶化 30.96%；family-OOD 是改善 7.07%、恶化 23.80%。两个主分区的每一条轨迹都这样走，
而且 24 个平均 checkpoint 都处在 field/gradient 的 Pareto 曲线上：多跑一步，field 更好一点，gradient
就更坏一点。只挑停止点无法跳出这条曲线。

这里还有一个初学者很容易混淆的地方：front top-10% F1 大多在上升。它只问“最强的一小部分边缘位置有没有
重叠”，gradient L2 则比较整个三维梯度场的幅值和方向。主要前沿位置可以逐渐找准，同时体内出现更多弱振铃和
高频误差，所以两个指标并不矛盾，也不能互相替代。

下一步不训练停止网络。先用同一 `24F/24A^T` 预算画完整的 H1/Sobolev、Tikhonov、TV/Huber、hybrid
projection 和 edge-superiorization 路径，看看能否造出支配旧 Pareto 曲线的新 checkpoint。只有经典固定方法
无法解释、truth-only 新路径确有 headroom、observable 线性规则能保留信号，而且师兄确认真实 callable、主指标
和组内基线后，才允许用小模型输出有界正则强度或固定滤波器的凸权重。

完整报告、图和复现入口见
[R0 早停 NO-GO 与正则化冲突](lgwo_a24_r0_semiconvergence_no_go_2026-07-20.md)。

**突破监测：没有算法突破。R0 关闭了当前 24 步路径上的 learned stopping，打开的是“怎样改变路径以兼顾场值与梯度/前沿”的下一实验问题。**

## 157. H1 锁箱过门了：我们找到了强基线，还没找到新算法

R0 告诉我们，无正则 CGLS 跑得越久，体场误差越低，全局梯度误差却越高。这一轮没有马上训练
网络，而是先问一个更基础的问题：经典 H1 正则能不能同时救 field 和 gradient？如果能，以后的新方法就不能只拿 CGLS
当弱对手。

首次正式运行之前，两位独立审计者又找到了几个容易让数字显得比实际更好的漏洞：四个主指标原本没有共用同一个 bootstrap
采样 mask；个别灾难性 case 可能被平均数盖住；程序没有二次确认自己真的在预注册 commit 上运行；9 个 active views 还会让 9 视角
几何的 held-out set 变空。我们在打开锁箱前把这些都修了，把 active views 改成 6--8，为四个终点统一用 50,000 次 cluster
max-T bootstrap，并加上最差 case、最差 cluster、harm、active residual 和 held-out-B 硬门。

正式运行只打开了一次，共 336 cases、672 条求解路径、3,360 条 checkpoint 记录。固定 H1 相对同预算 CGLS 的结果是：

- IID：field `+3.6412%`，gradient `+7.8708%`，最差 case 仍为 `+1.3181% / +5.0631%`。
- family-OOD：field `+4.0791%`，gradient `+8.1137%`，最差 case 为 `+1.8017% / +4.6598%`。
- joint-OOD：field `+3.7626%`，gradient `+7.8189%`，最差 case 为 `+1.5587% / +2.5232%`。
- 无失配 exact-operator control：field `+3.7152%`，gradient `+8.2085%`。
- 四个分区的逐 case 和逐 cluster `>2% harm rate` 都是 0。

一个不导入正式 runner 统计函数的独立 validator 又重算了 43,932 项，结果是 `VALID`。如果偷改 decision CSV 中的一个数，
它会返回 `INVALID`。这一步很重要，因为“同一份代码说自己算对了”不等于独立复算。

该怎样解读？好消息是，R0 的 field-gradient 冲突在这个合成体系中不是无法跳出的，改正则路径是对的。更重要的现实是：
H1 是经典方法，不是我们的新算法。它现在变成了一个更强的“守门员”。后续 TV/Huber、H1-TV 混合或神经控制器，如果只赢 CGLS 却赢不了 H1，
就没有充分的算法贡献。

下一步仍不是训练大网络。先在完全相同的预算、split 和尾部门下比较固定 TV/Huber；只有它们和 H1 各自显示不可互换的优势，才实现
H1-TV/Huber 固定凸组合或 hybrid path。只有不同 case 的最优组合真的不同，且 geometry/noise/residual/Ritz 能在不看 truth 的情况下预测它，才给一个小模型
授权。

完整预注册、表格、图、成本与师兄问题见
[R2-A H1 合成锁箱结果](lgwo_a24_r2a_h1_lockbox_result_2026-07-20.md)。

**突破监测：没有算法突破；有重要证据里程碑。H1 合成锁箱经独立复算通过，现在是下一阶段必须击败的强基线；新算法、真实 BOST、未见实验 rig、神经模型与论文成功仍为 0。**

## 158. TV 还没开跑，先把“步子能迈多大”算诚实了

H1 锁箱之后原计划直接比较 TV/Huber，但旧 scalar-PDHG 已经暴露过一个问题：统一步长太小，几十步内几乎停在零场。继续扫正则权重只会把“走不动”误诊成“TV 没用”。所以这一轮先解决更基础的问题：怎样给每个相机缺失模式的加权 BOST 算子算一个绝不会低估的谱范数上界。

旧办法把射线投影 `M` 和三维差分 `D` 拆开，各自取最坏行列和再相乘。它严格，却非常松。在公开 PSU 九视角、QMC8、`32^3` 几何上，旧上界是 40 步幂迭代估计的约 56 到 85 倍。拿它做 PDHG 步长当然安全，但会慢得没有公平比较价值。

新办法直接使用比较矩阵 `C = W|M||D|P`。代码逐 chunk 算 `C` 的行和与列和，不展开巨型矩阵；因为真实矩阵的每个绝对值都不超过 `C`，`||C||_1||C||_infinity` 仍是严格上界。小算例用显式 SVD 反查，流式和单块实现、不同 chunk 切法、CPU/MPS 路径都对上。

正式 post-open 诊断固定了 10 个缺失视角模式，覆盖 4、6、8、9 个活动相机。新证书相对 power-40 estimate 为 `5.87--11.50x`，旧证书为 `55.86--84.76x`；等于把保守度再压低 `7.30--10.31x`。10/10 都选中新证书，建立证书的物理调用仍是 `0F/0A^T`。独立 validator 重算 155 项，返回 `VALID`。

下一步也因此更清楚：新标量上界仍偏松，不能马上宣称 TV 会赢。最值得做的是从同一个 `C` 构造逐体素对角 majorizer `q_j = sum_i C_ij sum_k C_ik`，先在小矩阵上证明 `diag(q)-A^T A` 半正定，再接到 diagonal/block PDHG。只有 data-only 收敛先改善，才加入 TV/Huber；只有同预算击败封存 H1，才进入 H1-to-Huber hybrid 和有界学习选择器。

完整推导、图和复现入口见 [R2-B0 范数证书诊断](lgwo_a24_r2b_norm_bound_diagnostic_2026-07-20.md)。

**突破监测：阶段性基础设施突破，但没有算法或论文突破。严格证书约收紧一个数量级；TV/Huber 重建、field/H1/front 收益、真实 BOST 与泛化证据仍为 0。**

## 159. 逐体素安全步长算出来了，也暴露了不能装作看不见的零覆盖

上一轮只得到一把“全场共用的尺”：不管体素被多少射线穿过，PDHG 都只能用同一个最坏步长。这轮把比较矩阵的每行质量反传回体素，得到 `q=C^T(C1)`。直观地说，`q_j` 是体素 `j` 在当前相机、噪声权重和射线离散下被数据项“拉住”的严格强度上界。

先没有相信自己的推导。我们在 `5 x 6 x 7` 网格上把整个加权物理矩阵逐列展开，直接检查 `diag(q)-A^T A` 的最小特征值，并额外试了 16 个随机二次型。streaming/单块、不同 chunk、support、全零权重、错误输入和 MPS 转 CPU float64 也都分别检查。聚焦回归 `29 passed`，产物 validator 重算 `333` 项后返回 `VALID`。

公开 PSU 的 10 个缺失视角模式上，`max(q)` 比上轮 composed 上界再紧 `1.11--1.70x`，相对冻结 power-40 estimate 为 `4.57--6.84x`。这说明它仍保守，但比全局行列最坏乘积更精细。

真正需要停下来想的是另一个数：support 内只有 `54.2%--68.6%` 的体素得到正 `q`，10 个 mask 累计有 `102,715` 个 support 内零质量。这不是说流场在那里为零。当前合同每视角只有 256 条射线、每条 QMC8，support 却是整个 `30^3` 内部立方体；零 `q` 只说明这个离散数据项没有触达该坐标。曲光线、更密射线或组内真实几何都可能改变它。

所以不能直接把 `1/q` 塞进 TV/Huber 然后开跑。data-only 健康检查必须冻结零 `q` 坐标并保留 coverage ledger；完整求解要先给 forward-Neumann 正则梯度构造 `q_G=|GP|^T(|GP|1)`，再用 `q_total=sigma_A q_A+sigma_G q_G` 证明组合 metric 的 Schur 安全性。TV 可以把邻域先验传到数据未触达的坐标，但必须明说那是正则先验，不是相机突然提供了新信息。

完整推导、图、NPZ 体素场、复现入口和下一道算法门见 [R2-B0D 逐体素 majorizer](lgwo_a24_r2b_diagonal_majorizer_2026-07-21.md)。

**突破监测：没有算法或论文突破。有可验证的数值基础设施进展，并新发现了当前公开离散合同的稀疏覆盖风险；重建收益、真实 BOST、未见 rig 和论文主张仍为 0。**

## 160. 第一个完整对角求解器跑了，结果是应该留下的 NO-GO

这一轮把上节的计划真正接到了求解器。数据项用 `q_A`，forward-Neumann 正则梯度另外构造
`q_G=|GP|^T(|GP|1)`，固定 dual steps 后得到 `q_total=sigma_A q_A+sigma_G q_G`。小网格上不仅检查
`diag(q_G)-G_P^T G_P` 半正定，还把真实 PSU 小算子和梯度矩阵堆起来，直接算归一化谱范数。CPU、MPS、内容摘要、
零质量冻结和每轮 `1F/1A^T` 共 `57 passed`。

首次开发运行在提交过的固定配置上完成：6 个解析反应场 proxy、4--9 视角、固定噪声，scalar 和 diagonal
都是 `20F/20A^T`。data-only 分支明确失败：field 平均恶化 `95.20%`，gradient 恶化 `338.59%`；它虽然
把残差压得更低，却在 `q_A` 只覆盖 support `49.3%--68.6%` 时用极大局部步长追数据，所以不能当质量候选。

加上 `q_G` 后 6/6 case 都补齐了 support，不再灾难性失败。但固定 Huber-gradient 下，field 只平均改善
`+0.1469%`，gradient 在 6/6 case 全部恶化，平均 `-0.8203%`。因此预写开发门是 NO-GO。独立 validator
重建 metric、重算汇总、核对调用与图像，`376 checks` 通过。这是算法候选的真失败，不是突破。

下一步不盲扫 lambda。scalar 证书和 diagonal 证书都使各自的矩阵差半正定，所以用一个 case-level
`beta in [0,1]` 做 `q_beta=(1-beta)q_scalar+beta q_diagonal` 仍是严格安全的标量凸组合。它可以在保留 scalar
保底的同时逐步开放对角加速。但不能直接让网络输出逐体素 `beta_j`：那时已不是两个 PSD 矩阵的标量凸组合，
安全证明会丢。先在已打开开发集画固定 beta 路径，若中间点确实同时救 field/gradient，再冻结一个 beta 到全新种子验证。

完整数字、图、证明和下一候选见 [R2-C 对角 PDHG NO-GO](lgwo_a24_r2c_diagonal_pdhg_no_go_2026-07-21.md)。

**突破监测：没有突破。新增的是一个经 376 项独立复算的开发 NO-GO，以及下一个可证明安全的 scalar--diagonal 插值候选；新算法、真实 BOST、泛化与论文成功仍为 0。**

## 161. beta 路径全部跑完：安全不等于有用，这条支线正式关闭

上一节留下的问题是：完全对角化太激进，那么只开放一部分对角步长会不会同时救 field 和
gradient？这次用一个全局标量 `beta` 把 scalar 与 diagonal 证书做凸组合。因为两个端点都是同一联合
正规矩阵的上界，它们的标量凸组合仍然安全。五个 beta 点都在显式小矩阵上过了 Schur PSD 检查，
而且 beta=0 与独立 scalar solver 的最大体场差只有 `4.93e-8`。

但质量结果没有给任何可供挑选的中间点。beta 从 0 走到 1 时，mean field gain 从 0 严格单调增到
`+0.1469%`，mean gradient gain 却从 0 严格单调降到 `-0.8203%`。最佳可选内点 beta=0.25 也只有
field `+0.0351%`，gradient `-0.1879%`，6/6 case 都伤 gradient。它没过预写双指标门，所以没有资格
换新 seed 验证。

这个单调趋势比“0.25 稍微好一点”更重要：预条件器只改变走向同一个优化目标的速度和路径，不会改变
充分收敛后的解。在当前 20 步欠收敛区间，步长越大，残差和 field L2 稍好，高频梯度误差就越坏。
继续加密 beta 网格只会变成看结果调参，训练一个 beta 预测网络则根本没有合格标签。

下一条线因此必须改变正则路径本身：用封存 H1 强基线保住低频体场，再在同一总 `F/A^T` 预算中加有残差
回退的 Huber/TV 或 edge-superiorization 局部修正。先用固定确定性路径去打冻结 H1；只有不同 case 的最优控制
确实不同，且 geometry/noise/residual/Ritz 可以不看 truth 保留这些 headroom，才允许一个可拒答、可精确回退 H1
的小控制器。

运行共 36 条求解路径、`720F/720A^T` 和 36 次 evaluator forward；独立 validator 从 NPZ 重建 30 个证书、
复算汇总、趋势、哈希、图像和结论边界，`709 checks` 通过。完整报告见
[R2-C2 beta 插值 NO-GO](lgwo_a24_r2c2_beta_interpolation_no_go_2026-07-21.md)。

**突破监测：没有突破。有效增量是一条经 709 项独立复算的单调反证，它正式关闭“只调全局 beta / 学 beta”支线，并把下一问题收窄到能否用改变目标路径的固定 hybrid 击败 H1。**

## 162. 第一次 edge 预跑被审计否决，第二次才得到能成立的 NO-GO

这一轮最重要的事情不是某个百分比，而是没有把做错的第一次运行藏起来。R2-D0 v1 已经跑出 42 行后，
独立审计发现它把冻结 H1 的 `denominator_floor=1e-16` 写成了 `1e-20`；同时虽有 `rho=0` 纯数据控制，
联合门却没有要求非零 edge 候选必须胜过它，伴随缺陷也只是记录，没有作为拒绝门。于是 v1 被明确标成
`INVALID_PROTOCOL_DRIFT_NO_SCIENTIFIC_DECISION`。这些数既不能叫成功，也不能叫算法失败，只能保留为调试溯源。

修订后的 R2-D1 没有假装自己还是第一次。配置明确写入“已观察 42 行”，恢复精确 H1 合同，并增加 matched
`rho=0`、front-F1、held-out clean、伴随恒等式和 edge 实际正步长门。它比较了 `19+1`、`18+2`、`16+4`
三种预算分配，每种测试 `rho=0,1,2,4`，另有 H1-20；6 个已见 case 共 78 条独立轨迹，每条都是
`20F/20A^T`，没有共享 H1 前缀后只在表里补调用数。

这一次得到的是有效 NO-GO。最接近的 `19+1, rho=4` 相对 H1-20 的 mean field/gradient gain 仍是
`-0.4335%/-0.1233%`，加权 residual ratio 为 `1.0378`，超过预写 `1.02` 门。它相对自己的纯数据控制却有
`+0.0676% field / +0.3959% gradient`，说明 edge 方向确实有局部作用。更激进的 `16+4, rho=4` 对控制的
gradient gain 达 `+1.4886%`，但相对 H1-20 仍为 `-0.1917%`，residual ratio 恶化到 `1.2250`。

讲人话就是：最后几步里，Huber 方向比普通数据梯度更会保护边缘；可是把 H1 的有效迭代拿走来换它，损失更大。
所以这条支线关闭的是“替换 H1 末步”，不是武断地说所有 edge prior 都无效。也不继续在同六个 case 上扫
`rho=8/16` 或 trust radius，因为 residual 已随 rho 系统恶化，继续调会变成看答案找参数。

最后又做了一次真正独立的复算：validator 没调用 runner 的聚合/选择函数，而是从 78 行 CSV、78 份 histories
和冻结配置重建 12 个汇总、选择行、matched `rho=0` 归因、20F/20A^T、5/6 held-out、front、伴随与 168 个
修正步，共过 561 项检查。它也抓住一个不能抹掉的小缺口：runner 当时漏绑了一个 imported helper。提交 blob 与
当前文件哈希一致，所以可以事后闭合依赖并保留 NO-GO；但这不等于执行瞬间的 manifest 完整。页面会明确写
`PASS_WITH_POST_RUN_DEPENDENCY_CLOSURE`，下一次 runner 必须在运行前绑定完整 import closure。

下一候选改成 R2-E0：完整保留 H1-20，缓存已经付费得到的 Krylov 搜索方向和 `A p_k`，再在 20 维系数空间内
做不增加物理投影的固定重优化。先证明缓存版与冻结 H1 逐位一致、数据 residual 可由缓存精确重算、固定非学习
目标能产生 headroom；只有这些成立，才考虑让小模型预测有界系数或 accept/fallback。

完整协议、数字和下一步见
[R2-D1 edge 预算分配 NO-GO](lgwo_a24_r2d1_edge_budget_allocation_no_go_2026-07-21.md)。

**突破监测：没有突破。有效增量是一次公开保留的无效预跑、一个经 561 项独立复算且明确标注事后依赖闭合的 78 路同预算 NO-GO，以及“edge 有局部信号但不能替换 H1”的机制定位。真实 BOST、fresh seeds、泛化与论文成功仍为 0。**

## 163. 把 H1 的 20 个方向全部留下再调权重，还是没有可用余量

上一节留下的 R2-E0 已经真正跑完。思路很朴素：H1-20 的每一步都要付一次 forward 和一次
adjoint，那就把 20 个搜索方向 `P` 和它们的投影 `A_w P` 全部缓存下来。H1 结束后，不再调用物理
算子，只在 20 个系数上做小优化。这样既不牺牲 H1 的迭代，也不会偷偷增加昂贵的光线追迹预算。

这次先遇到了两个很具体的软件问题，而且都没有藏。v1 在把 MPS 张量送到 CPU float64 做最小二乘时
污染了右端项；v2 修了最小二乘，却漏掉 Huber 优化器里的同类转换。两个目录都只留下
`invalid_attempt.json`，明确写 `result_valid=false`，没有把报错混进算法均值。v3 才把所有
CPU-double 入口和 MPS 返回统一起来，并增加最小二乘、完整 Huber、residual-safe oracle 三个 MPS
回归测试。

讲人话解释这 20 维小问题：原来 H1 找到了一间只有 20 个方向的房间。R2-E0 允许在房间里任意挪动，
但不允许打破墙。结果 data-only 最小二乘能把观测残差平均压到 H1 的 `0.8951`，却让 gradient error
恶化 `1.6445%`；四个 Huber 权重几乎一步都走不动，最大场变化只有约 `1e-10`。truth-free 规则最后把
`huber_ratio_200` 当诊断对象，但它只有 4/6 case 接受过步长，低于 0.8 门，而且质量等于 H1 到数值
噪声尺度。

最有价值的是 oracle。不给 residual 限制时，知道 truth 的 oracle 确实能把 gradient 改善
`14.52%`，但 field 反而差 `2.384%`，观测 residual 变成 `10.23x`。把它压回预写 `1.02` 数据门后，
只剩 `+0.0040% field / +0.1647% gradient`。也就是说，不是四个 Huber 权重没猜中，而是当前 20 个
方向里没有足够的“既符合数据、又修好场和梯度”的移动空间。

所以现在不能训练一个网络来预测 Huber ratio。那等于让网络在四个都不合格的标签里挑一个，看起来像
算子学习，实际没有科学问题。下一条最窄路线应先扩展 span：比较 fixed flexible/preconditioned Krylov、
一个有 residual 回退的 edge 方向，以及 BOST geometry/noise-conditioned 方向。先用 truth-only
representation oracle 判断新方向是否真的增加联合余量；有余量以后，才讨论用部署可见特征预测方向或
接受/拒绝。

正式运行之后又单独写了一个不导入 runner/core 的 validator。它从 72 行 CSV、候选汇总、缓存和
等价性账本、108 个几何输入摘要、Git blob、两次无效尝试与 PNG 重新核对，873 项全部通过，NO-GO
判决不变。但这项审计是在运行后实现的，所以准确含义只是“落盘证据内部一致”，不能倒过来说正式运行
事前就受它约束，更不能把它当成真实 BOST 或泛化验证。

这里也要防止把常见组合写成创新。hybrid/recycled Krylov、flexible Krylov、FCG-NO 和 Neural Krylov
都已有工作。我们若有差异，只能来自 BOST 的曲光线/标定/噪声困境、严格 data envelope、拒答和跨 rig
尾部验证。没有何远哲师兄的真实 callable、JVP/VJP、残差层级和数据 split，就还不能替实验室决定这条
差异是否真实存在。

完整数字、数学式、失败记录、文献边界和给师兄的 8 个问题见
[R2-E0 缓存 Krylov 子空间 NO-GO](lgwo_a24_r2e0_cached_krylov_subspace_no_go_2026-07-21.md)。

**突破监测：没有突破。有效进展是关闭了“同一 H1-20 span 内调 Huber 权重 / 学权重选择器”支线，并把下一问题定位到真正扩展 span 的方向生成。真实 BOST、fresh、泛化和论文成功仍为 0。**

## 164. R2-F0 还没开箱：先把“新方向真的有用”这句话算公平

今天的真实增量不是跑出了更好的重建，而是把 R2-F0 的判题方式补得更严格。R2-E0 已经说明：只在 H1 的
20 个旧方向里重新配系数，几乎没有同时改善 field 和 gradient 的空间。R2-F0 因此准备加入 residual
backprojection、Huber 先验和 edge-gated backprojection 三类固定方向。但在打开六个复用 case 前，必须先
排除一种很容易出现的假胜利：候选因为得到了更强的 truth-only 系数优化而赢，并不等于新方向真的扩展了表示空间。

这就是为什么主对手必须是 **matched H1 span oracle**。候选 C5 在 `P20 + k` 个新方向组成的空间里，用
synthetic truth 重配全部系数，再经过同一个 residual 与 field-trust 安全 ray。若只拿它和普通 H1-(20+k)
迭代终点 C2 比，候选同时占了“新方向”和“truth 重配系数”两种便宜，无法知道收益来自哪里。C3 则把
H1-(20+k) 的旧经典空间也交给完全相同的 joint objective、系数求解和安全 ray。只有 C5 稳定胜过 C3，才可说
新增方向带来了 H1 继续迭代不能解释的表示余量；这仍只是离线 oracle 上界，不是可部署算法。

C5 相对旧空间 C1 的增量也必须用**百分点**，不能用相对百分比。举个简单例子：假设 H1-20 的误差是
`1.00`，C1 降到 `0.20`，它的改善是 `80%`；C5 再降到 `0.19`，改善是 `81%`。新增方向只多贡献了
`81%-80%=1` 个百分点，而不是拿 `0.01/0.20` 算出的 `5%`。当前门要求 field 和 gradient 都至少增加
`3` 个百分点。这样可以防止旧 oracle 已经很强、剩余误差很小时，把一个很小的绝对变化放大成看似漂亮的相对收益。

rank floor 的敏感性也不能只换一个阈值标签。`1e-5`、`1e-6`、`1e-7` 会改变 SVD 认为哪些分量已经属于
旧增广空间 `G=[A_w;sqrt(lambda)D]`，因此必须在每一档重新做 projector、重新生成投影后的方向、重新组成 span，
再重新求 oracle 系数、安全 ray 和最终指标。若三档共用主阈值得到的同一批方向，只在最后重算 rank，那只能说明
记账表稳定，不能说明科学判决稳定。R2-F0 要检查的是从“方向生成”到“oracle 终点”的整条链是否对 rank floor 稳定。

`DirectionPacket` 是这条链的 truth firewall。讲人话就是：先把只靠实验时可见信息能生成的东西装进一个封包，
例如 `x20`、缓存 residual、H1 基底、support、几何与权重摘要、冻结常数以及实测 `F/A^T` 调用账本；封包先落盘、
计算哈希并封存，然后才允许另一个 `OraclePacket` 打开 synthetic truth。前一个封包不能收到 truth、clean field、
field/gradient 指标、带 plume 或 shock 语义的 case 名称，也不能根据 oracle 结果更换 family。它防的是方向公式、参数或
候选集合偷偷看答案。需要特别强调：C5 的系数和安全 ray 终点仍然看了 truth，所以 firewall 只证明“原始方向没有看
truth”，不会把 C5 变成部署方法。

数值计算还要把 MPS float32 与 CPU float64 分工。大体量物理投影可以继续在 Mac 的 MPS float32 上跑；但小型
projector、SVD、截断最小二乘、oracle 系数和安全 ray 对微小奇异值很敏感，统一转到 CPU float64 计算，再把需要继续
走物理路径的张量送回 MPS。这样保护的是 `1e-5` 量级的 rank、near-null 和 ray 判定不被 float32 舍入噪声伪造，
不是在证明物理模型正确，也不是在证明真实 BOST 泛化。

当前可复核的软件证据如下：方向核心与 runner 的聚焦单元测试为 `23 passed`；从仓库根目录显式使用
`PYTHONPATH=.` 启动的独立 validator 测试为 `13 passed`，合并运行是 `36 passed`。synthetic validator 的
伪造证据包包含 40 个方法、240 条主指标、144 条 rank-floor 指标、6 条 projector sensitivity 和 6 个 NPZ，
独立重算 `20,185` 项，并对指标、case 删除、方法互换、调用账本、rank floor、方向哈希、NPZ、truth flag、
artifact hash、敏感性指标和 projector 状态共 11 类篡改 fail-closed。这证明单元公式和 synthetic 验证器合同能抓住
这些错误；它没有运行正式 runner，没有重新生成真实 `A/A^T`，也没有证明正式产物已经与 validator 完整兼容。

因此正式 R2-F0 继续保持 `HOLD_R2F0_PROTOCOL_NOT_READY_TO_FREEZE`。配置中的源码绑定仍为空，正式目录与
`.incomplete` 目录都不存在，**R2-F0 打开的正式科学 case 数仍为 0**。下一道门是：逐条闭合冻结前红队的 P0/P1
问题，尤其是 end-to-end rank-floor、DirectionPacket 先封存后开 oracle、实测调用账本和原子证据清单；随后把
runner、core、config、validator、既有证据与 import closure 绑定到同一个 HEAD，让 runner 产生的 synthetic
dry-run 包通过独立 validator，再做第二轮红队复审。只有这些全部通过，才允许冻结配置并对六个已打开过的 mechanism
case 做一次正式 R2-F0；即使过门，也只得到 `REPRESENTATION_SIGNAL_ONLY_NO_AUTHORIZATION`，不能直接训练网络或
声称真实重建、泛化与论文成功。

来源边界与冻结问题见 [R2-F0 一级来源边界](lgwo_a24_r2f0_primary_source_boundary_2026-07-21.md) 和
[R2-F0 冻结前红队](lgwo_a24_r2f0_protocol_red_team_2026-07-21.md)。

**突破监测：没有突破。今天新增的是更公平、可证伪的 R2-F0 判题合同，以及通过单元与 synthetic 篡改测试的验证器骨架；算法收益、正式科学结果、真实 BOST、fresh、泛化、学习器授权与论文成功仍为 0。**

## 165. R2-F0 的软件护栏更严了，但科学盒子仍然没打开

上一节说“先把方向封存，再看 synthetic truth”。第二轮红队继续追问：即使顺序没错，判题公式本身有没有可能让一个普通方向看起来像新方向？答案是有，而且这次确实找到了几处需要修的地方。

最重要的一处是三方向联合 `RHE` 的对手。它不能只和 `R`、`H`、`E` 三个单方向比，还必须和 `RH`、`RE`、`EH` 三个双方向比。否则一个有效信息其实已经由两个方向提供，第三个方向只添了很小噪声，联合体仍可能被误报成“三者协同”。现在 `RHE` 必须超过六个真子空间；单测专门构造了“超过所有单方向、但输给一个双方向”的反例，结果必须 fail。

第二处是“空间新颖”不能只看几根向量两两不平行。想象旧 `P20` 是一张二十维薄纸，新方向看起来彼此夹角很大，却可能整组仍几乎躺在旧纸面里。现在 runner 和独立 validator 分别用 PyTorch float64 与 NumPy 计算 canonical correlation、最小主角和实际秩增量。只有整个新子空间确实从旧空间里伸出来，才有资格进入表示门。

第三处是数值精度。Mac 的 MPS 很适合跑大投影，但 float32 不适合在 `1e-5` 附近决定一个奇异方向该保留还是删除。现在小型线性代数、oracle 系数、安全 ray、field/gradient 指标和最终判门都留在 CPU float64；只有明确送进物理算子的 endpoint 才转换设备 dtype。若任何连续指标离门槛太近，协议直接给出 numerical ambiguity，不允许靠最后几位舍入赢。

独立 validator 也不再只核对 runner 写出的数字。正式结构要求保存 PSU 离散射线采样的索引、三线性权重、投影坐标和 ray scale；validator 只用 NumPy 重建一套 `A/A^T`，重新计算 40 个主方法投影、三档 floor 的 24 个投影、伴随探针和自己的内积恒等式。小型正例能通过，修改投影或伴随后会 fail。这说明将来正式包里的公开 PSU 物理核可以被另一套实现复核，但不说明真实 OERF 曲光线、标定或实验噪声已经正确。

当前聚焦套件为 `77 passed`，Ruff、字节码编译、配置状态和差异检查也通过。这里的 77 是软件测试数，不是实验 case 数。正式配置仍是 `HOLD_R2F0_PROTOCOL_NOT_READY_TO_FREEZE`，源码绑定为空，正式目录与 `.incomplete` 都不存在，**打开的 R2-F0 科学 case 仍然是 0**。

接下来先等第二轮只读红队给出最终 P0/P1 清单，再决定能否冻结公开 PSU reused-case audit。更重要的外部门没有变化：需要用 [给何远哲师兄的首次接口清单](n5_d5_advisor_first_contact_2026-07-19.md) 确认真实 callable、straight/curved residual 层级、JVP/VJP、几何标定、主要失败模式、认可基线、数据 split 和宿主合同。没有这些信息，可以继续把公开机制实验做严谨，但不能替实验室发明真实物理困难。

**突破监测：没有突破。新增的是更难被假阳性骗过的协议、独立离散物理核复算和 77 项软件证据；正式表示收益、可部署学习器、真实 BOST、fresh、泛化与论文成功仍为 0。**

## 166. 不再继续造协议：先把师兄的九个回答变成研究路线

R2-F0 的软件护栏已经足够多，继续增加签名、账本或 synthetic 角色不会告诉我们实验室真正卡在哪里。当前最有价值的动作，是让何远哲师兄确认真实 forward、数据和主痛点。仓库原本已有七门 N2 数据合同、空白 JSON 和 validator，但它们适合机器检查，不适合边聊微信或开会边记录。因此新增了一个独立的 [真实接口回复工作台](../advisor_interface_intake.html)。

工作台只问九类事实：主痛点、场参数化、可调用的 `A/Aᵀ` 或 forward/JVP/VJP、straight/curved/direct residual 的形成层级、hard branch、独立 split 单位、最小匿名资料、组内强基线与主指标、保存和论文权限。未知值保持“待确认”，不会用默认选项替师兄补答案。

这些问题不是通用 AI 问卷。NeRIF 的一级来源明确给出九路投影、实验 8+1 留出、DeepFlow 位移、按标定 ray 反向采样；数值数据生成附录使用 RK4 ray tracing，而结论把 nonlinear ray tracing 作为可继续集成的能力。因此我们必须向师兄确认真实反演代码到底处在哪一层，不能从论文措辞直接断言。同步 PIV-BOST 又表明，三维折射率重建会进入真实速度测量补偿链，小火焰条件下报告的瞬时速度误差量级约为 ±2%；所以最终指标不能只剩一个三维 field L2，还要问同步、像面、梯度与最终物理量误差。

工作台目前预写八条条件路线：导数一致性、straight-to-curved discrepancy、有限孔径、标定漂移、位移提取、有限视角、端到端成本和 4D 序列。每条只给出第一项可失败实验、必须比较的强基线与停止条件。例如只有当主痛点选 ray bending、存在成对路径且 callable 支持导数时，才显示 `ROUTE_PAIRED_RAY_MISMATCH_READY`；缺任何条件就返回 `NEEDS_*`，不会自动命名新算法。

页面的真实交互测试已经完成：选择 ray bending、implicit field、forward+JVP+VJP、独立 curved/straight、无已知 hard branch、session split，加上最小 callable/geometry/high-fidelity forward、组内基线和本机保存权限后，进度为 `9/9`，路线正确变为 paired-ray mismatch；保存、刷新恢复和清空均通过。测试草稿已经清空。桌面与 390px 移动端没有文本溢出，静态页面测试为 `3 passed`。

导出的 `ADVISOR_REPLY_DRAFT` 仍不是 N2 数据合同。收到师兄确认后，要人工核对事实并映射到 `data_templates/oerf_n2_lab_intake.placeholder.json`，再运行七门 validator。工作台中的所有 claim 位默认都是 false；它不授权训练、audit 开封、私有数据上传、真实 BOST 改善、泛化或论文成功。

**突破监测：没有突破。新增的是把“等师兄回复”变成可操作、可分流、不会偷填事实的本地工作台；真实 callable、真实数据、正式重建结果和论文贡献仍为 0。**

## 167. 先用一个小实验把五个基础概念串起来，再碰大网络

前面的路线已经把伴随、gauge、CGLS、几何漂移和算子学习分别讲过，但对初学者来说，它们仍像五门不相干的课。这次新增了一个 CPU-only 的一维线性小实验，把它们接在同一条因果链里：64 维 synthetic field 经过 24 个均值为零的导数核形成 measurement；核的位置由一个标量几何参数控制。这个参数只是 operator shift 教具，不是相机标定或真实曲光线模型。

实验先过三项结构门。随机向量上的伴随内积相对误差为 `2.41e-16`；常数场响应比为 `1.84e-17`；把任意场整体加 `0.7` 后，measurement 相对变化为 `2.63e-16`。它们说明离散转置和预设 gauge 结构按代码工作，不说明 forward 是正确 BOST 物理。

随后比较四条路线。fixed ridge 只看 `y`，geometry-conditioned ridge 看 `[y,g*y]`；两者使用完全相同的 240 个训练 case、噪声、truth 与 ridge alpha。nominal Tikhonov 永远用 `A0`；exact Tikhonov 每个 case 都得到正确 `Ag`，所以只是 privileged teacher。在 `g=0/0.02/0.06` 三点，conditioned 相对 fixed 的平均 field error 降低 `19.68%/39.27%/52.47%`。但外推时 exact teacher 的 `0.04766` 仍优于 conditioned 的 `0.06382`，提示“把物理算子弄对”仍有余量。

最值得记住的是 CGLS 半收敛。第 7 步 field error 最低，为 `0.11065`，measurement residual 为 `0.03412`；继续到第 36 步，residual 降到 `0.01340`，field error 却爆到 `5.31624`。所以实验室以后即使给出漂亮 reprojection，也不能把它自动当作三维场正确，必须有 stopping、正则化、held-out view 和独立物理指标。

独立审计补充了三个必须当面写出的限制。第一，conditioned ridge 有 3,072 个系数，fixed ridge 只有 1,536 个，而且前者多拿了几何侧信息；所以这是 information/capacity ablation，不是同容量架构竞赛。第二，第 7 步由 synthetic truth 事后选出，只是 oracle diagnostic，不能当部署 stopping rule。第三，clean-measurement residual 用 evaluator-only 无噪声投影计算，不是模型收到的 noisy residual。

代码、测试、JSON、两张 CSV 与四联图已放入 `learning_labs/`。定向测试增至 `6 passed`，其中一项实际运行两次并比较完整 report 与数组；报告还记录 Python、NumPy、Matplotlib、平台和源码 SHA-256。默认结果与第二次独立运行的四个产物逐字节一致。这个确定性只证明同一环境能重放，不证明真实物理。完整讲解见 [算子基础小实验中文导读](operator_foundations_lab_guide_2026-07-21.md)。

**突破监测：没有突破。当前新增证据严格属于 `EDUCATIONAL_SYNTHETIC_LINEAR_PROXY_ONLY`；真实 BOST、三维重建、新算法、DeepONet/FNO 优越性、跨 rig 泛化和论文成功仍为 0。下一有效门仍是师兄确认 callable、residual 层级、JVP/VJP、几何、split、基线与权限。**

## 168. 第一次明确让 `A_true` 和 `A_est` 分家：标定修正有功效，但安全门会漏检

一维小实验默认算法知道正确几何，这次把更接近真实 BOST 的麻烦单独拿出来：连续解析场和真实 ray 生成观测 `y=A_true x+noise`，重建却只拿到带方位、俯仰、滚转和横向平移偏差的 `A_est`。观测侧用解析梯度积分，反演侧用 `10^3` voxel 的有限差分加三线性采样，避免直接用同一离散矩阵造数据再求逆。它仍是直线平行射线 synthetic proxy，不是 OERF 真实相机或曲光线。

实验冻结 6 个 rig、每 rig 6 台相机、3 种 morphology proxy，共每档 18 个场；同一 case 内所有方法共享观测、Tikhonov solver 和正则强度。标定误差幅度从 0 增至 0.5/1/2 时，离散算子的相对 Frobenius 失配约为 `0/0.0571/0.1133/0.2229`。1 档对应本教具中平均 ray direction 误差约 0.97 度，但不能把它当实验室阈值。

第一种 naive LOCO 完全不看体真值：每个候选在五台相机上重建，在第六台 noisy measurement 上评分，六折平均最小者被选中。它在 1/2 档对全部 18 个 case 改善 field error，平均收益为 `5.66%/13.27%`；但零失配时平均反而恶化 `0.35%`、最差恶化 `1.71%`，0.5 档也有个案恶化 `0.78%`。这直接证伪了“held-out reprojection 最小就天然安全”。

固定半步阻尼把修正幅度乘 0.5，在 0.5/1/2 档分别平均改善 `1.44%/3.87%/8.29%`，这三档的 18 个 case 都非劣化；但零失配仍平均恶化 `0.11%`，所以简单阻尼也不能当授权门。

第三种 single-frame LOCO-LCB 只在六台相机的配对 residual 改善下置信界为正时才修正，否则退回 reported geometry。它在四档都保持 `100%` 非劣化，却在 0.5/1/2 档分别回退 `94%/72%/50%`。这不是成功，而是安全与功效的明确冲突。

考虑 TDBOST/4D 可能让多帧共享一套标定，又加入 multiframe camera-block LCB：同一 rig 的三个场先在每台相机内平均证据，再跨六个 camera block 算 heuristic LCB。它在 2 档把回退率从 50% 降到 17%，平均收益从 `7.76%` 增至 `9.58%`；但 1 档反而回退 83%、只平均改善 `0.85%`。原因不能直接写成定论，当前可见现象是不同场对标定参数的可观测性会互相增强，也会互相稀释。这里的统计单位是 camera block，不是 session/rig；`2.015` 也只是近似单侧 `t_5` 的机制筛查常数，没有正式置信覆盖主张。

所以总门保留 `NO-GO`，没有移动阈值追求 PASS。更值得继续的问题也因此收窄：网络不应直接回归完整位姿，而应先判断每帧/每相机证据是否可靠。不过这里不能提前把残差权重叫作 observability weight；下一轮必须先用不用训练的可靠性对照检验功效，再看 geometry JVP/VJP 是否足以单独定义真正的标定可观测性。

独立审计还要求把“argmin 没用 truth”与“整条函数没有 truth 能力”分开。修订后，per-field 与 multiframe selector 只接收删去 truth/clean 的 deployment record；field/clean evaluator 指标在选择之后附加，并额外冻结逐相机 fold score CSV，才能从结果包独立复算 pooling。完整物理解释、角色表、结果表、三条候选模型、复跑命令与给师兄的 10 个问题见 [三维 BOST 标定失配小实验导读](calibration_mismatch_lab_guide_2026-07-21.md)。

**突破监测：没有突破。新增的是 `SYNTHETIC_3D_BOST_POSE_MISMATCH_MECHANISM_ONLY` 证据和一个明确 NO-GO：naive residual selection 不安全，严格 fail-closed 功效不足。真实相机标定、曲光线、实验三维真值、DeepONet/FNO/NeRIF 对比、跨 rig 泛化和论文成功仍为 0。**

## 169. 可靠性权重不是可观测性：中等失配有线索，小失配仍失败

上一节最后留下一个容易说错的词：如果某台相机的残差变化更稳定，就给它更高“可观测性权重”。一级来源和独立审计都指出，这个命名不成立。残差一致性最多说明 measurement reliability；真正的 calibration observability 要看残差对位姿、焦距或畸变参数的 Jacobian、尺度化 `J^T J` 谱、近零特征方向和参数耦合。当前 frozen ledger 没有 geometry JVP/VJP，所以这轮统一改称 camera reliability screen。

实验没有重新调用 forward、adjoint 或重建器，也没有训练网络。它只重放上一轮冻结的 3,024 条逐相机 LOCO score。对每个目标 synthetic rig，主候选留下它不看，只用其他 5 个 rig 估计六台相机的闭式权重：若某相机的候选残差改善和其余相机改善的中位数长期同向，权重较高；负相关截到零，随后把原始权重限制在 `[0.5,2]`，归一化后的最大最小比不超过 4。

第二轮审计把“代码没用真值”加固成结构隔离。deployment loader 现在只验证并打开 camera-score CSV，从它自身推导 rig、family、档位和候选；它不解析、也不携带包含 field/oracle 汇总的上游 report。决策冻结后 evaluator 才验证完整 checksum 并加载真值指标。poison test 把 report 替换成伪造 truth summary 后，部署决策仍逐项不变，而 evaluator 必须因 checksum 不匹配拒绝。这比只检查 selector 函数签名更强，但仍只是本地软件隔离，不是外部安全证明。

先做了一个必要的旧基线检查。uniform replay 与 v2 的 24 个 rig-severity 决策逐项相同，mismatch 为 0。之后比较主 LORO reliability 权重：1 档平均 field gain 从 uniform 的 `0.85%` 增至 `4.05%`，增加 `3.19` 个百分点，回退率从 `83.33%` 降到 `16.67%`；2 档从 `9.58%` 增至 `11.20%`，增加 `1.62` 个百分点，六个 rig 都得到正 field gain。但 2 档 seed 503 相对 uniform 少了 `0.99` 个百分点，说明逐 rig 尾部并非全赢。

真正决定 NO-GO 的是 0.5 档：uniform 和主候选都 100% 回退，平均收益、改善 case 比例都为 0。冻结门要求所有非零档平均收益至少 5%、改善比例至少 75%；主候选在这里没有功效，1 档均值也只有 4.05%。因此状态是 `POSTOPEN_CAMERA_RELIABILITY_WEIGHT_REPLAY_NO_GO`，没有调低 `2.015` 或事后移动门槛追 PASS。

“同预算”也被审计收窄为相同在线物理预算。uniform 在线选择读取 3,024 个 score value；主 LORO 六折要额外读取 15,120 个训练 score value，再读取 3,024 个目标 score value，总计 18,144。两者新增 forward、adjoint 和重建调用都为 0，但端到端计算量不同；wall time 与 peak memory 本轮没有测，不能写成成本相同。

六台相机的平均 LORO 权重也暴露出一个风险：camera 2 约为 `0.0889`，其余约为 `0.166--0.195`。这可以解释当前六个 synthetic rig 的改善，却可能只是记住了固定 camera identity。如果真实装置换了相机顺序、数量或几何，这个模式未必存在。六个 LOCO fold 的五相机训练集还高度重叠，同一 score surface 同时用于候选准入与排序；所以 `2.015` 仍只是描述性 t5-style heuristic，不能解释成置信覆盖、安全证书或显著性。

下一模型因此拆成三条线，而不是把所有信息塞进一个网络：`q_rel` 读取独立 sentinel 帧的 whitened residual 和噪声尺度；`q_cal` 只在拿到 geometry JVP/VJP 后，从 scaled `J^T J`、近零方向和耦合构造；`q_field` 用 view-conditioned normal operator 或边际谱/秩增益衡量对三维场的独立信息。三条经典基线分别过门后，才允许一个有界组合器输出权重、阻尼或停止建议，物理 solver 仍负责几何更新。

完整数字、初学者解释、同预算对照、六篇一级来源、30-rig sealed audit 合同与给师兄的问题见 [相机可靠性权重回放结果](calibration_camera_reliability_screen_result_2026-07-21.md)。

**突破监测：没有突破。新增的真实价值是把“残差可靠性”和“几何可观测性”分开，并证实有界 LORO 权重只提高中、大失配的回放功效，未解决小失配。新 forward、新重建、真实数据、fresh rig、神经算子、泛化和论文成功仍为 0。**

## 170. `q_cal` 第一次真正消去未知场：raw 敏感性有值，data-only 辨识力是零

这一轮没有继续调相机残差权重，而是直接计算几何 Jacobian。局部模型写成 `y=A(eta)x+noise`，`eta` 包含 yaw、pitch、roll、shift-u 和 shift-v 五个无量纲 mode，`x` 是 1000 维 voxel field。对比三个量：known-field raw `C^T C`、消去自由场的 data-only `S0`、加 ridge 先验后的 `S_lambda`。

最重要的结果是一个结构性 NO-GO。六相机算子是 `300 x 1000`且满行秩，自由场的数据切空间已填满 300 维观测空间。因此几何变化在数据中造成的局部变化都能被某个 voxel perturbation 吸收。六个 rig 的 estimated/teacher `S0` 相对秩全是 `0/5`，trace retention 最大只有约 `7.6e-30`。raw `J^T J` 即使很大，也不等于 joint reconstruction 中的几何可辨识性。

第一轮独立审计抓住了两个容易造假阳性的问题。其一，原型用连续 analytic renderer 算 teacher Jacobian，却用离散 voxel operator 消去 nuisance field，两者不属于同一 likelihood。正式版已改成 voxelized truth 经同一 forward family 生成 teacher；连续 renderer 只生成 noisy pilot observation。其二，三相机排序使用了六相机 pilot 重建的 `x_hat`，所以合法含义是“全相机 pilot 辅助的下一次相机布置”，不是“只靠这三台就能当帧自洽重建”。

在这个修正后的同模型 teacher 中，参考 `alpha=0.002` 的 prior-conditioned 排序出现了一条值得追踪的线索：estimated-vs-teacher profile 排序 Spearman 平均 `0.956`、最低 `0.910`，选中子集的 oracle D-efficiency 中位 `0.990`、最低 `0.922`。相比之下，estimated raw 的 D-efficiency 中位只有 `0.235`。但 estimated Jacobian 相对 teacher 的平均 L2 误差仍有 `0.818`，所以只能说排序结构在这个 post-open proxy 中部分保留，不能说 `q_cal` 数值已被准确预测。

alpha 扫描进一步说明正曲率是先验制造的。teacher 的 median trace retention 从 `alpha=1e-6` 的 `0.0055%` 增到 `alpha=1` 的 `53.26%`，同一 rig 的最优子集随 alpha 切换 2 到 3 次。因此不能挑最好看的 alpha 宣称成功，必须说明场先验、噪声白化和参数尺度。

对毕设最有用的结论不是“这条路不行”，而是创新问题被定位了：要让 data-only `S0` 真正出现非零方向，必须引入已知 calibration target、低维物理场、4D 共享张量/时间基，或明确受约束的 neural-field tangent。其中 4D 共享低秩场与何远哲师兄的 TDBOST 主线最直接。下一个有效机制实验应先问：缩小 nuisance tangent 后 `S0` 的最小特征值是否真的抬起；然后才训练任何 DeepONet/FNO/NeRIF 组件。

完整数字、入门反例、三条研究入口、一级来源、复跑命令与给师兄的问题见 [`q_cal` 剖面结果导读](calibration_qcal_profile_result_2026-07-21.md)。专项测试为 `13 passed`，正式产物的 report、四张 CSV、四联图和 checksum 已固定。

**突破监测：没有突破。新增的是一个经过两轮数学/代码审计的结构性 data-only NO-GO，以及一条只在参考先验、同模型、post-open synthetic proxy 下过门的相机排序线索。真实 BOST、subset-only 部署、fresh rig、自动标定、神经算子、重建改善、泛化和论文成功仍为 0。**

## 171. 多帧确实把 0/5 抬成 5/5，但噪声让它仍然不能用

上一节留下的问题是：如果六帧共享一套相机几何，并且场不再逐帧自由变化，几何信息会不会从 nuisance tangent 里露出来？这次没有直接训练 FNO，而是先把最常见的“4D 结构”逐个做成可证伪控制。

结果先关闭了三个看似聪明、实际无效的说法。每帧自由 voxel 仍是 `0/5`；把序列写成 `X=Phi H`、但 `Phi/H` 都允许变化，仍是 `0/5`；只固定时间系数、让空间因子自由变化，也仍是 `0/5`。原因很直接：当前 `A` 满行秩，几何导数造成的变化可以由 `delta Phi` 吸收。低秩、Tucker、CP 或神经隐式表示本身，不会自动创造联合标定信息。

真正抬秩的是已知输运 `x_t=W_t x0`。一个共同初场必须同时解释六帧，因此精确输运的 profile rank 在三个新 rig 上都是 `5/5`。但最弱广义 retention 只有 `9.76e-5` 到 `3.63e-4`，中位 `1.34e-4`；trace retention 中位也只有 `1.36%`。这说明五个方向原则上都非零，却有非常薄的最弱方向。

v1 在注册噪声下给出 q relative-L2 中位 `9.41`，但旧门仍会接受，因为 residual 只有约 `0.74 sigma`。独立审计指出，这个归一化漏掉了 nuisance 和五个几何参数消耗的自由度：`m=1152`、`rank(B)=512` 时，正确剩余自由度是 `635`，纯噪声的旧 RMS 期望正好约 `sqrt(635/1152)=0.742`。所以残差小根本不代表参数可信。

审计还找到了一个真实代码错误：q-trial 循环把最后一个 reacting scene 的 field 误传给所有 `teacher_*` 列。它不影响 deployable `q_hat` 或 v1 的 NO-GO，但 v1 的 teacher CRLB 和 teacher q error 全部作废。v2 保留相同随机 seed namespace，按 model scene 重新选 teacher field，并加零噪声余项单测。现在无噪声 teacher q error 中位为 `0.00145`，说明堆叠、导数符号和局部线性链路是对的；注册噪声下 teacher q error 中位 `10.54`、严格 teacher CRLB 中位 `11.05`，失败主要来自 practical SNR。

SNR sweep 又把门槛量化出来。把当前 synthetic base sigma 降到 `1/128` 时，plugin q error 中位 `0.0895`、teacher CRLB `0.0863`，9 个案例中 7 个通过新不确定度门；到 `1/64`，q error 中位仍有 `0.174`，但 95% 最大半径过宽，0 个授权；注册 sigma 下 0 个授权。这个 sweep 固定同一噪声方向只改幅度，因此只是 post-open threshold map，不是 coverage 或泛化证明。

冻结 PCA 基展示了相反的危险。rank 4/8/16 的最弱广义 retention 中位约 `0.864/0.798/0.648`，看起来远强于输运；但 clean model residual 中位仍约 `9.13%/5.95%/5.42%`，最差到 `32.67%`。这不是“低秩效果好”，而是先验把 nuisance 空间压小后制造强曲率，同时把真实场塞错了。

反应流 proxy 则给出更细的折中。仅输运无法解释 birth，NIS 9/9 拒绝；加入一个共享 source 后 nominal residual 回到数值零，但最弱 retention 降到 `1.58e-5`、q error 中位升到 `20.39`。越真实的 nuisance 会保护场拟合，也会抹掉更多几何信息。把 innovation 放回每帧自由后，结构又完全回到 `0/5`。

v2 因此不再用“rank 满 + residual <2”授权。它要求 99% chi-square NIS、plug-in 95% 最大半径不超过 `0.25 q_ref`、局部参数包络不越过 `0.1`、更新显著且 profile 满秩。reference noise 下 exact 和所有 mismatch 的授权数都是 0；10% velocity mismatch、错误时间顺序和未建模 birth 的旧 false accept 被关闭。这个“0 false accept”不是算法成功，因为正确精确输运也全部拒答；它只证明门现在知道自己没把握。

对算子学习的直接启发是：网络不能再被设计成一个直接输出相机位姿的黑盒。更合理的结构是让 DeepONet/FNO 预测 transport/innovation tangent、warm start 或 anchor 权重，再由真实 forward JVP/VJP、held-out camera/time NIS 和置信椭球决定是否更新。经典底座必须先实现迭代 variable projection、q-amplitude sweep 和 500-noise bootstrap。完整结果见 [多帧 q_cal v2](temporal_qcal_tangent_result_2026-07-21.md)，文献路线见 [动态算子一级来源](temporal_operator_primary_sources_2026-07-21.md)。

**突破监测：没有突破。新增的是“多帧精确输运可结构性抬秩，但当前 SNR 仍实践不可辨”的严格 NO-GO、对 v1 teacher 泄漏的公开纠错，以及一个能关闭旧 false accept 的不确定度门。真实 BOST、真实 4D reconstruction、神经算子优越性、fresh audit、泛化和论文成功仍为 0。**

## 172. 500 次独立噪声后，真正坏掉的是 plug-in 覆盖率

v2 的 `1/128` 低噪声窗口只是固定噪声方向的 SNR 地图，不能证明 95% 置信域真能覆盖 95%。这次对 3 个 rig、3 个方向、6 个 `q` 幅度和 5 个噪声档分别生成 500 个独立高斯复本，总计评估 270,000 次 teacher/plug-in 估计；另外跑了 864 个 one-step 与稠密 iterative variable projection 的配对 trial。

数值实现门是过的：三个 rig 的 full profile Jacobian 中心差分相对误差在 `1.98e-6` 到 `5.35e-6`，864 个 iterative trial 无数值失败，objective 全部单调。但预注册主门仍是 NO-GO：`q=q_ref, noise=1/128` 的 teacher coverage 只有 8/9 cell 过门，plug-in 只有 5/9。plug-in pooled relative-L2 中位 0.0743、p90 0.1390 看上去都不大，但这不能弥补置信域欠覆盖。

最有用的定位出现在 `q=2 q_ref` 的低噪声格。teacher 九个 cell 的平均 coverage 仍约 94.9%，plug-in 却只有 48.0%，两个 cell 甚至为 0。同时 projected nonlinear remainder 中位只是线性响应的 0.54%。这说明主因不是局部 affine forward 已完全失效，而是 nominal `B(0)` 场拟合把部分几何信号吸收进 `x_hat`，随后的 plug-in covariance 又没有包含场误差、同数据相关性和 Jacobian 变化。

这个机制直接把下一步收窄为两个不用大网络的基线：一是对 nuisance field 一阶正交的 profile score 加 sandwich covariance；二是 frame/view cross-fitting，用不重叠数据估场和构造 geometry score。后者只能减少同噪声耦合，不会自动消除场估计误差，所以必须与前者分开对照。

经典 iterative variable projection 给出了弱而混合的改善：q 误差中位从 0.1094 降到 0.0996，下降 8.91%，没达到冻结的 10% 门；field 和六帧 sequence 中位只改善 1.33% 和 1.90%。`q=2` 子组改善较强，`q=0.5` 一个子格还略差，不能挑子组写成稳定优越。总体 86% trial 触发 trust bound，也超过预注册的 5% 上限；本轮不回头改门。

完整入门解释、主格表、三个下一算法形状、复现命令和禁止主张见 [500 噪声 + variable projection 结果导读](temporal_qcal_bootstrap_varpro_result_2026-07-22.md)。

**突破监测：没有突破。新增的是一个通过独立噪声覆盖审计定位的 plug-in 欠覆盖机制，以及一个未过门的经典 iterative reference。新算法、神经算子、真实 BOST/4D 重建、fresh 方向泛化、论文成功和突破仍为 0。**

## 173. 97.9% 覆盖并不比 95% 更好：这次失败在过度保守

上一节发现 plug-in 置信域会漏掉真值，这一轮把“点估计中心”和“区间宽度”分开查。数据仍是已经打开过的 3 个 synthetic rig 和 3 个旧方向；每个 rig 另加一个 `q=0`，再对 `q/q_ref=1,2` 生成新噪声。21 个 cell 各 500 次，共 10,500 条观测，前 250 次只校准，后 250 次只评估。

先看点估计。one-step plug-in 的 `||q_hat-q||/q_ref` 中位是 `0.08610`，iterative full-profile 降到 `0.07192`，改善 `16.46%`；field 和 sequence 中位也分别改善 `2.87%`、`3.09%`。但只有 `64.44%` 的配对观测 q 误差更好，还不是逐例稳定胜出。完整 profile 的第一步中位误差反而是 `0.50545`，说明从零初值出发的一次大公式并不能代替通常 5 次 profile evaluation 的迭代。

再看不确定度。plug-in 原生 pooled coverage 只有 `72.93%`；同一个 iterative 终点上的 GN sandwich 和 exact-score Godambe 都是 `93.09%`，21 个 cell 中 19 个达到 90%。这说明完整 profile 已经修掉了主要问题。exact bread 与 GN bread 的结果又几乎重合：逐观测统计量相对差中位约 `0.124%`，所以当前低噪声局部 proxy 里，复杂 residual-curvature 项不是主要矛盾。

预注册协议没有直接接受 93.09%，而是在每个 calibration cell 取有限样本 95% 顺序统计量，再用 21 个阈值的最大值统一校准。这样评估覆盖率升到 `97.90%`，21/21 cell 都超过 90%，最大半轴中位只有 `0.2562 q_ref`。第一眼很像成功，但覆盖率的 95% Clopper--Pearson 区间是 `97.48%--98.27%`，整个区间都高于目标 95%。区间可以靠保守放大得到，所以冻结门故意要求 95% 必须落在这个区间内；本轮因此严格是 `POSTOPEN_DEVELOPMENT_FORENSICS_NO_GO`。

奇偶帧 cross-fit 也被真正证伪。它原生 coverage 只有 `62.61%`；若用 worst-cell 包络硬补，coverage 会到 `99.81%`，但半轴中位膨胀到 `3.149 q_ref`。同一段序列的奇偶帧共享初场、输运、相机和模型误差，不是独立 acquisition，切帧不能凭空完成去偏。

结果打开后做了一项明确标成 post-hoc 的校准粒度复算。把所有 calibration score 合并后取 pooled 95% 阈值，评估 coverage 是 `94.50%`，95% 区间 `93.84%--95.10%`，21/21 cell 仍超过 90%，半轴中位还降到 `0.2345 q_ref`。这个数不能改写本轮判决，因为阈值方案是在看过结果后换的，而且还是同三个旧 rig；但它给下一轮一个很具体的预注册候选：在全新 rig/session 上比较 pooled、分层收缩和 global worst-cell 三种校准，必须同时守住总体 95%、逐 rig 尾部与区间功效。

对算子学习的启发也更清楚了。若真实 full-profile 太贵，网络不应直接宣布 `q`，而应预测 warm start、低秩 nuisance/transport tangent、预条件或有界校准修正；随后由真实 forward/JVP/VJP 做少量 correction，并以 profile score、held-out 物理指标和 fail-closed 半径接受或回退。要把中心偏差继续往下压，则需要 flow-off/known-target 或独立 acquisition 支持 physical-target orthogonal score，而不是继续在三个 synthetic rig 上调 sandwich。

完整数字、数学边界、事后探索表和给师兄的问题见 [联合剖面推断 v4 结果](temporal_qcal_profile_inference_result_2026-07-22.md)。

**突破监测：没有突破。新增的是一个通过 10,500 条配对观测确认的 full-profile development signal、一个因过度覆盖而严格保留的 NO-GO，以及一个只能用于冻结 fresh 协议的 pooled-calibration 线索。新算法、真实 BOST/4D 重建、跨 rig 泛化、论文成功和突破仍为 0。**

## 174. 模型能学会尺度，也能在关系翻转时把 hard rig 覆盖打到 11%

v4 留下一个很具体的问题：最坏 cell 包络太宽，pooled frame calibration 又可能让 frame 多的 easy rig 占更多票。真实接口还没到位，所以这轮没有碰休眠 fresh BOST 盒子，而是先做一个带正例、负对照和反例的层级校准教学实验。

三个情景都严格按独立 rig 切成 120 个 fit、120 个 calibration 和 400 个 evaluation；每个 rig 只有 30--240 帧，难 rig 故意更少。逐 frame pooled 在可观测情景的 observation-weighted coverage 看起来有 `92.39%`，但按 rig 等权只有 `87.34%`，hard quartile 更只有 `56.79%`。同一个结果已经说明，不能把同一段 sequence 的大量 frame 当作大量新实验。

低容量 log-ridge 只从 fit rig 学部署可见特征与 score scale 的关系，再在 calibration rig 上冻结 inflation。在可观测情景中，它的 hard coverage 达到 `93.76%`，相对 frame pooled 增加 `36.97` 个百分点；中位半径又比 equal-rig 小 `24.04%`。这证明 toy 里确实有可利用的尺度信息，但不证明真实 BOST 也有。

负对照更重要。难度完全隐藏时，log-ridge 的 fit R2 只有 `1.39%`，hard coverage 与 equal-rig 只差 `0.19` 个百分点；模型没有凭空创造信息。关系翻转时，fit R2 仍高达 `95.21%`、学到正 slope `0.8838`，evaluation 真 slope 却是 `-0.8887`。此时 rig mean coverage 降到 `69.96%`，hard quartile 只剩 `11.23%`，p10 只有 `2.77%`。训练拟合很好，部署仍可以非常错。

所以当前候选不再是“用 FNO 直接预测 q”。更可信的结构是：物理摘要驱动低容量 scale/tangent proposal，support/relationship gate 判断是否在域内；域内仍做独立 session calibration 和 1--2 次 exact profile correction，域外退回保守 A0；最后用 held-out view、field/gradient 与 PIV velocity endpoint 接受或拒绝。网络只有在同预算下超过 log-ridge、equal-rig、full-profile fixed warm start 后才有资格进入。

完整结果、算法框图、六个师兄问题和禁止主张见 [rig/session 层级校准 toy 结果](rig_session_calibration_toy_result_2026-07-22.md)，入门练习见 [校准学习路线](rig_session_calibration_learning_route_2026-07-22.md)。定向测试 `7 passed`，结果散列 5/5 通过，严格 JSON 也可解析。

**突破监测：没有突破。新增的是一个可复现的 cluster-size bias 机制、一个 learned-scale 适用窗口和一个非常强的 OOD 失败反例。真实 acquisition、真实 callable、三维/4D 重建、神经算子优越性、跨 rig 泛化和论文成功仍为 0。**

## 176. 高频模型在离散投影上更好，却在连续导数下 7/7 更差

公开 NIR-BOS 代码审计之后，我没有直接照搬 Fourier/hash 网络，而是先问一个更基础的问题：如果训练 renderer 用固定步长中心差分，网络会不会利用这个差分算子的频率盲区？为避免同一个离散链出题和答题，观测由连续解析场梯度与 96 点积分生成，逆端只看到 32 点 ray samples 和独立参数化；6 个 train、2 个 development、2 个 test 角完全分离。

第一轮 4 个解析反应形态、两档噪声、两个优化 seed 的预检没有授权 GCS selector。高频 `[1,2,4,8,16]` 模型的 central held-out projection 平均略好，field relative-L2 却在 5/8 单元明显更差；场与 central-test 损害方向的一致率只有 0.25。这个结果说明共享 central renderer 可能遮住场问题，但还不能定因。

随后冻结 14 个全新 dense angles，并用 `FD(h)`、`FD(h/2)`、`FD(h/4)` 与 automatic derivative 重渲染同一已训练模型。排除唯一事后观察单元后，高频模型的 dense-AD 在 7/7 单元比低频差；中位高频减低频为 `+0.48161`，高频 AD 减自身 `FD(h)` 为 `+0.54144`。development GCS 与 dense-AD 损害的 Spearman 为 `0.82143`，原 central test 重放漂移为 0。

数学原因可以直接手推。对 `sin(pi f x)`，中心差分导数与连续导数的振幅比为 `sinc(pi f h)`。当前 `h=2/15` 时，`f=4` 只保留 0.594，`f=8` 变成 -0.062，`f=16` 只有 0.061。网络可以在离散 renderer 的近盲频带放入很强结构，central projection 看不明显，AD 会把它完整暴露。

这个机制与 2026 *Neural Refractive Index Primitives* 报告的 Fourier + AD 梯度噪声一致，但不能声称首次发现：该论文已经比较 automatic/discrete/hybrid，ReNO 已经定义 operator aliasing，mip-NeRF 也已处理尺度相关采样混叠。我们新增的只是一个 BOST 梯度投影 clean-room 反例和可复现审计。

完整学习路线见 [梯度混叠零基础导图](gcs_gradient_aliasing_learning_route_2026-07-22.md)，冻结审计见 [连续 renderer 配置](gcs_fourier_continuous_audit_freeze_2026-07-22.md)。

**突破监测：没有突破。新增的是通过 7/7 单元、独立角度和四档 renderer 支持的 synthetic 连续/离散混叠机制；新算法、真实 BOST、算子学习、泛化、论文成功和突破仍为 0。**

## 177. 多尺度护栏改善了连续投影，但没有可靠改善三维场

确认机制后，我先在唯一已打开的 `wrinkled / 8% noise / seed 101` 上筛候选。AD-only、固定 25% AD hybrid 和四 renderer 等权高频模型的 field relative-L2 分别是 `0.23420 / 0.14303 / 0.13538`，都没有超过低频基线 `0.13340`。因此没有把普通 hybrid 包装成新想法。

下一候选 MGRS 使用低频稳健基座和零输出高频残差。残差同时拟合 AD 与 `FD(h), FD(h/2), FD(h/4)`；每个 development checkpoint 必须四项逐一不劣于基座，平均至少改善 0.5%，否则精确恢复零残差。配置、门和两阶段 split 在新结果产生前以 commit `d3ae73a` 冻结，runner 再以 commit `687e22f` 提交。

Stage A 的正式结果是 NO-GO。`MGRS-56` 场改善 0/3，中位场差 `+0.001859`；`MGRS-6816` 场改善 2/3，dense-AD 中位改善 `-0.027056`，但场中位只改善 `-0.001082`，未达到预写 `-0.002`。12 条 seed-level 路径中 MGRS-56/MGRS-6816 分别只有 2/6 与 4/6 残差获准。Stage B 的四个 oblique/shock 单元因此保持未运行。

这次失败很有解释力：四个 renderer 都在相同有限角集合上看投影。它们能抑制只适配某一差分步长的高频，却不能阻止残差进入相机投影的近零空间。下一候选不应继续堆 renderer，而应先做带 `L2 + H1` 最小残差的经典基线，再与 TV/Huber 对照；只有学习式频带 gate 在同预算下超过这些基线，才有理由进入神经算子。

完整数字、师兄五问和复现命令见 [MGRS Stage A 结果](gcs_mgrs_stage_gate_result_2026-07-22.md)。

**突破监测：没有突破。新增的是一个会精确退回基座的可运行算法候选，以及一个保持 Stage B 密封的严格 Stage A NO-GO。新算法优越性、算子学习、真实 OERF、跨 rig 泛化、论文成功和突破仍为 0。**

## 175. 公开代码能看，不等于能跑；默认 test 也不等于独立测试

这轮找到了一份很贴近我们方向的公开实现：2026 年 *Neural Refractive Index Primitives* 的作者仓库。它有 Phantom 1、MATLAB 生成链、Python 神经隐式训练和 CUDA ray marcher，表面上看像是终于可以直接训练了。我没有立刻改代码开跑，而是先把仓库固定在 commit `a385cce...`，逐项检查许可、数据身份、split、路径、设备和依赖。

先说能跑的部分。作者的 Fourier 编码是一个相对独立的 PyTorch 数学核。我在 Apple MPS 上给它两个三维点，输出形状是 `2 x 39`；前向、一阶导和二阶导都为有限值。这个绿灯很有用，说明本机可以先写 clean-room 的 Fourier/小 MLP/指标实验，不必所有事情都等服务器。

但完整入口还是红灯。`main_BOS.py` 会自行覆盖命令行并强制 `--fp16 --cuda_ray`，设备选择只有 CUDA 或 CPU；renderer 顶层无条件导入 CUDA raymarcher，hash 还把一个 tensor 写死到 CUDA。环境文件锁定 Windows、MSYS2 和 CUDA 11.8，数据 JSON 又用 Windows 反斜杠。64 个文件引用在 Mac 上原样一个都找不到，替换分隔符后才是 64/64。删掉一个参数远远不够。

真正改变 benchmark 设计的是 split 审计。仓库写着 12 train、2 validation、2 test，但 validation 的两个位姿和 test 的两个位姿都分别复用了训练集前两个位姿；validation 和 test 的两对位姿又完全相同。再把 image、mask、img-mask 与 RI integral 解码成像素数组比较，validation/test 的 8/8 对全部相同。文件哈希略有区别只是 PNG 编码层差别，像素内容没有独立性。

这意味着默认 `test` 不能承担 unseen-camera 证据。它最多帮我们检查保存、绘图或数值流程。下一步必须重新生成独立角度，最好冻结连续角区留出；否则“test error”这个名字会让人误以为已经检验视角泛化。

另外，MATLAB 和 Python 各有一个 Phantom 1 目录，但逐相对路径、文件大小和 SHA-256 完全相同：都是 71 个文件、92,846,449 bytes、同一个 tree hash。因此仍然只有一个独立三维场。把它加很多噪声、切很多 rays 或复制很多视角，都不能把一个函数变成 operator learning 的多 field 样本。

算法路线也因此收窄。2026 论文已经做过 Fourier/hash、automatic/discrete/hybrid gradient、mask 和层级采样，“换编码”不能写成创新。更值得先测的是一个 `GCS-Hash` 诊断：看固定审计 rays 上离散梯度与 AD 梯度的失配，能否提前预测 hash 的噪声过拟合或边界饱和。它若没有预测力，就关闭自适应解冻，不再堆网络。

如果诊断有跨 field 预测力，再做主候选：Fourier 稳定基座加有界 hash residual，gate 只看部署可见的光流置信度、跨视角 residual、噪声和 geometry；support 外回退 Fourier。它必须同时超过 Fourier、hash 和固定 50:50 混合，并对参数量、ray samples、wall-clock、公平输入和最坏场 harm 分账。

正式指标也做了纠正。折射率本身接近 1，直接对完整 `n` 算 relative-L2 会把误差稀释；主 field 指标应对 `delta n = n - n0` 计算，并同时报告 gradient/front、新相机 displacement、边界饱和、逐场尾部和成本。单个 held-out projection 不能代替三维真值。

完整机器报告、图、许可边界、三个算法候选和服务器迁移顺序见 [公开 NIR-BOS 复现门禁](open_nir_bos_release_readiness_audit_2026-07-22.md) 与 [三维 benchmark 合同](open_nir_bos_benchmark_contract_2026-07-22.md)。

**突破监测：没有突破。新增的是一个真实 MPS 组件绿灯、完整入口的可复核红灯、默认 split 泄漏和单 field 身份的机器证据。作者训练、三维重建、真实火焰、算法优越、跨场泛化和论文成功仍为 0。**

## 178. 残差更平滑了，总场仍然可能往错的方向走

MGRS Stage A 失败后，这一轮没有继续堆 renderer，而是先把经典正则对照补齐。低频基场冻结，`[6,8,16]` 高频残差仍从严格零输出开始。我在固定 `7^3` 内点上用 `h/4` 差分构造归一化 `L2+H1` 和 `L2+Huber-gradient`，各扫三个强度，再加无正则 MGRS control。所有候选共用 240 步、四 renderer、两个 seed 和精确回退规则。

42 次 MPS 拟合用时 232.57 秒，低频基场与 MGRS control 对旧证据的最大重放差都是 0。结果仍是 NO-GO：最好的 `L2+Huber 0.003` 的 field 中位差为 `-0.001482`，相对 MGRS 只多改善 `0.000400`，未达 `0.001` 增量门；H1 中位差仍为 `+0.001275`。六个正则候选没有一个过全门，Stage B 保持密封。

最重要的学习不是“H1 没用”。设基场误差为 `e0`，残差为 `d`，总 H1 误差的变化是 `2<grad(e0),grad(d)> + ||grad(d)||^2`。我们惩罚的是残差自身的二次项，但不知道它和基场真误差的交叉项是正还是负。所以“残差更平滑”不能保证“总场更正确”。

已开路径也支持这个定位：28 条获准残差的 dense-AD 全部改善，但只有 14 条同时改善 field/H1。wrinkled 单元的 7/7 获准路径改善，smooth 只有 7/21，说明可能存在形态依赖；但这只是已开开发线索，不是泛化证据。

下一候选应该改问题：先比较正则总场 `n0+d` 的 H1/TV/Huber，再用 held-out residual、残差粗糙度、geometry 或时间一致性建立可拒答的 correction-alignment gate。不再在相同 residual-only 目标上继续扫 lambda。

完整方法、数字、公式、复现命令与师兄五问见 [残差正则 Stage A NO-GO](gcs_regularized_residual_stage_a_result_2026-07-22.md)。

**突破监测：没有突破。新增的是 42 次可复现经典正则对照、一个保持 Stage B 密封的 NO-GO，以及“残差范数无法控制总误差交叉项”的下一步定位。新算法、真实 BOST、算子学习优越、跨 rig 泛化、论文成功和突破仍为 0。**

## 179. 总场能量也不是真值，但它可能帮我们拒绝坏修正

上一轮 residual-only H1 无法控制基场误差与残差的交叉项，所以这次把正则直接放到总场 `n0+d`。为了不把“场整体缩小”误认为梯度先验成功，同时加了纯总场 L2 对照。代码、9 条路径、双对照增量门和 Stage B 密封规则先固化在 commit `9553fcd`，再运行 54 次 MPS 拟合。

结果仍是 NO-GO。最好 `total_h1_0p01` 的 field 中位差为 `-0.001565`，相对 MGRS 多改善 `0.000483`，相对最好 residual-only 只多 `0.000083`；truth-H1 中位仍恶化 `+0.001434`。纯 L2 0.003 已经做到 `-0.001494`，说明那一点 field 收益大部分可能是保守收缩，不是梯度结构被正确恢复。

更直接的反例是：`total_h1_0p01` 确实让候选总场的归一化 H1 能量中位降了 `0.73%`，但它到真值的 H1 误差却更大。这说明“更平滑”仍不等于“更真”，尤其对 wrinkled interface 和 shock。

但 36 条获准路径的事后诊断给出了新线索。这些路径的 dense-AD 全部改善，只有 18 条同时改善 field/H1。若再要求总场 L2 不增，可保留全部 18 条改善，但仍错放 5 条有害修正；总场 Huber 不增则保留 10 条改善、错放 0，但漏掉 8 条改善。这已经是一条很清楚的 safety-recall Pareto，但只有 3 个独立物理单元，9 个候选重复不能当成 9 倍样本。

下一候选改为 **Observable Energy-Alignment Gate**：不再改重建网络，而是用总场 L2/H1/Huber 有符号变化、残差粗糙度、四 renderer margin、噪声与 geometry 判断是否接受修正，不确定就精确回退。先在 smooth/wrinkled 同 family 里扩 phantom seeds、噪声和角度缺失，按 seed/geometry 分组留一；不打开 oblique/shock Stage B。

完整数字、混淆矩阵、新候选输入和师兄五问见 [总场正则 NO-GO 与能量门线索](gcs_total_field_regularization_result_2026-07-22.md)。

**突破监测：没有突破。新增的是 54 次总场经典对照 NO-GO、对“平滑不等于真实”的直接反例，以及一个只能进入扩样验证的事后能量门假设。可部署 gate、新算法、真实 BOST、算子学习优越、跨 rig 泛化、论文成功和突破仍为 0。**

## 180. 零误放为什么还是失败：它只敢接一道题

上一节的能量门只在 3 个物理单元上看过，所以这次真正扩到了 12 个新 phantom。每个 phantom 都经历两档噪声和三种相机/ray 压力，共 72 个条件单元；每个单元又跑两个网络重复，共 144 条路径。两个网络 seed 只是重复测量，不能装成两个新火焰。

还有一个很重要的分母问题。144 条路径里只有 74 条被原修正器准入，70 条回退。我们没有把回退路径从统计中删掉。如果只看“算法愿意出手”的时候，一个非常保守的算法很容易看起来漂亮；真实系统却必须为所有帧负责。

真值评分后，72 个单元只有 7 个是 field 和 H1 都足够变好，40 个在灰区，25 个明显有害。简单 L2 非增门确实留住了 7/7 个好修正，但也放进了 14 个不好修正。H1/Huber 更严，只留住 1/7，还是放错 1 个。所以“能量降了”仍然不是“离真值近了”。

低容量 ridge gate 的数字最容易骗人：`1 TP / 0 FP / 6 FN`，precision 是 100%。但它在 72 道题里只敢答 1 道，其余全部交白卷。我们只观察了 1 个被接受的 phantom group，就算它没出错，零事件的单侧 95% 风险上界仍是 0.95。讲人话：下一次出错的真实概率仍可能很高，我们只是样本太少。

我本来还希望一个完全分开的 holdout camera 能当审判。结果也不行：有些三维场真的变好时，这个 holdout 投影反而变差；有些投影变好，三维场却进了近零空间。一张新 X 光片不能唯一确定一个三维人体，同样，一个新 BOST 投影也不是三维场真值。

两次完整 MPS 运行比较了 40,937 个数，最大差是 0。这个结果很有价值：它说明我们不是因为偶然 seed 才得到 NO-GO。但它只是“失败可重放”，不是“算法成功”。

下一步不再扫能量阈值。我们要问每个相机分别在说什么：拿掉某一个相机后，重建修正方向会不会突然翻转？是六个视角一起支持它，还是一个坏 camera 拉着所有人跑？这就是下一个 working hypothesis：View-Influence Selective Residual Operator。先用不训练的 exact leave-one-view 基线看信息是否存在，没信号就停；有信号才进 JVP/VJP 近似、ridge 和小 set encoder。

完整表格、重放散列和师兄六问见 [同 family 能量对齐 NO-GO](gcs_energy_alignment_same_family_result_2026-07-23.md)，文献边界见 [Observable gate 一级来源](observable_gate_primary_sources_2026-07-23.md)。

**突破监测：没有突破。新增的是 12 个独立 phantom、72 个条件单元、144 条完整路径和两次数值一致重放支持的 energy-gate NO-GO，以及一个需要先做信息上界的逐视角影响候选。新算法、真实 BOST/PIV-BOST、算子学习优越、跨形态/跨 rig 泛化、论文成功和突破仍为 0。**

## 181. 为什么先花 28 道题检查尺子，而不是直接跑 912 道题

逐视角影响的直觉很简单：六台相机一起训练出一个修正，把其中一台拿走重新训练，如果三维修正突然反向，那一台相机可能在独自拉动近零空间。可真正做起来不能只把一列 residual 临时遮住，因为训练最优点也会跟着变。exact leave-one-view 要把 base 和 residual 都用相同 seed 从头重训。

完整账单是 144 条全视角重放，再加 768 条去单相机重训，共 912 条 base+residual 拟合。直接跑有两个风险：相机 ray block 切错，或者所谓 full replay 已经和上一轮不一致。那样两小时计算只是在很认真地测一把弯尺子。

所以先做 28 路径 pilot：两个已经打开的 phantom、两档噪声、nominal six-view、一个网络重复。4 条 full 用来对冻结结果，24 条 LOO 用来检查切视角和 feature。pilot 不接真值标签，不算 AUC，不挑算法。它只回答代码能不能稳定执行。

逐视角 feature 也只看网络自己：修正变化多大、方向余弦、沿原方向投影、norm ratio。full correction 是零就直接拒答。真值必须等 observable CSV 落盘后才能连接，避免算法一边做 feature 一边偷看答案。

如果 pilot 过关，才允许运行 912 路径的已开数据机制面板。完整面板仍不是论文结果；它只决定要不要申请一批新 calibration phantom。若 grouped AUC 没过预写门，exact LOO 支线关闭，不训练 set encoder。

完整公式、成本和 V0--V5 顺序见 [逐视角影响可执行协议](gcs_view_influence_mechanism_protocol_2026-07-23.md)。

**突破监测：没有突破。新增的是逐视角切片/特征代码、协议验证器和 28 路径 pilot runner；它们尚未运行，逐视角信息是否存在仍未知。**

## 182. 第一次 pilot 真的拦住了错误，第二次才开始算

第一次启动在训练前就停了：我把 correction 评估网格写到 `half_width=1.0`，现有边界窗函数要求严格小于 1。没有放宽旧函数，而是把协议改到 0.95，并把“网格必须严格在边界内”加入 validator。这个失败说明 28 路径检查不是装饰。

修复并重新提交后，pilot 跑了 240.77 秒。4 条 full replay 和冻结源的最大 observable difference 是 0，24 条 exact LOO 全完成，四组 influence feature 全部有限，状态是 `VIEW_INFLUENCE_PILOT_INVARIANTS_PASS_NOT_A_MECHANISM_RESULT`。

还有一个不需要真值就能看到的结构：24 次去视角重训里 18 次 residual admitted，6 次回退；删 90°/120° 时各只有 2/4 次 admitted。这个现象不能写成物理发现，但 full panel 应显式记录 LOO admission fraction，不能让零 correction 悄悄混在 cosine 里。

完整面板也必须回答“逐视角特征比旧能量特征多了什么”。所以在任何真值 join 前，协议新增固定三路对照：source observable、view influence only、source + view influence；组合 grouped AUC 必须至少比 source control 高 0.05。否则即使组合 AUC 好看，也可能只是重复上一轮的旧信号。

**突破监测：没有突破。pilot 只授权实现和运行完整 post-open 信息面板；mechanism signal、新 calibration、fresh audit、Stage B、新算法与论文结论仍未授权。**

## 183. AUC 都有 0.91 了，为什么还是失败

912 次拟合已经全部完成：144 条全视角重放，768 条去单视角重训。144 条 full replay 与上一轮冻结观测的最大差是 0，所以这次没有拿一个已经漂移的模型当尺子。

结果第一眼很容易被误读。source 能量/投影特征的 AUC 是 0.9626，逐视角特征自己是 0.7297，两者合起来仍有 0.9099。如果只问“0.91 高不高”，似乎已经可以写一个网络。但我们在看结果前问的问题是：“新特征有没有比旧特征多带来至少 0.05 AUC？”真实答案是 `-0.05275`，不仅没增加，反而降低了。

讲人话：旧学生单独做 455 道正负排序题，做对了 438 道；加上新同学的意见后，只做对 414 道。新同学不是完全不会，但他提供的信息和旧答案重复、不稳定，或在小样本下互相干扰。我们不能在看完答案后才选几句有用的话重新考试，因为协议已经写明不允许事后特征选择和 threshold tuning。

另一个问题是 coverage。144 条网络路径里只有 74 条能得到完整特征；合并两个网络重复后，72 个条件只有 27 个完全可观测。combined 在 `p>=0.8` 时确实是 1 TP / 0 FP，但它只敢接受 72 道题中的 1 道，漏掉 6/7 个好修正。这不是广泛安全，而是近乎全拒绝。

运行末尾还出了一个工程事故。fail-closed 行没有 LOO feature 列，可观测行有；旧 CSV writer 用第一行字段当全部表头，遇到后面的新列就报错。好消息是 144 条 JSONL 已经全部落盘。恢复器把拟合函数替换成“一调就报错”，只做汇总；恢复前后记录 SHA-256 完全相同，新拟合数是 0。独立 validator 又用显式正负样本对重算 AUC，16 项检查全部通过。

所以这条支线在 V1 停下。不做 JVP/VJP 近似，不训练 set encoder，不为了赚回已花的两小时而继续调参。下一步应当把师兄的真实 callable、geometry、curved/straight residual 层级和真实物理 endpoint 拿到手，再决定算子是学 warm start、preconditioner、bounded correction 还是 uncertainty。

完整数字、恢复证据、风险表和下一步三件事见 [逐视角影响 912-path 正式判决](gcs_view_influence_panel_result_2026-07-23.md)。

**突破监测：没有突破。新增的是一个 912-fit、独立 pairwise 复算支持的严格 NO-GO：逐视角影响对已开 synthetic 有部分信息，但无增量价值，支线关闭。新算法、真实 BOST/PIV-BOST、算子学习优越、跨 rig 泛化、论文成功和突破仍为 0。**

## 184. 先把一条光线接对，才有资格训练网络

这一轮没有先写 Fourier MLP，也没有先拿 FNO 跑排行榜。我们先问一个更朴素的问题：如果把作者公开 Phantom 1 的真实三维折射率场放进我们自己写的直线光线积分器，能不能重放由 detector `u/v` 经 `uvtoeps` 转换后的三分量 XYZ projection？

第一版 v0 诚实失败了。6,144 条 ray 中只有 6,021 条穿过声明的 ROI，相交率 97.998%，没有过原先 99% 的门；发布 CGLS-TV 场从 64 个积分点加到 128 个积分点时，输出还变化 5.260%，也没有过 2% 的数值收敛门。单元测试还抓到平行 ray 碰 AABB 时的判断错误。于是 v0 状态固定为 `D0_FORWARD_IDENTITY_NO_GO_FIX_GEOMETRY_OR_UNITS`，不能因为后来修好了就删除。

看过 v0 后，我们只做定位。未穿过 ROI 的 123 条 ray 仍有观测，但 RMS 只有穿过 ray 的 `0.003834`；它们更像“经过零场区域的合法 ray”，不该从总分母删除。另一方面，128 到 256 积分点的变化已经降到 `0.009622`。所以 v1 在运行前写死两条修复：ROI 外预测为零但保留在 6,144 条分母中；主积分改成 256 点，128 点只作收敛检查。这叫 post-open repair，不叫 fresh test。

v1 的 7 个机器门全部为真，验证器完成了 281 项检查。它从充分统计量独立重算全局指标并核对哈希；相机对齐、ROI 外 RMS 和 quadrature 原始量来自运行摘要，验证器只重算是否过阈值，没有重新读取外部体数据追光。ground truth 重放的三分量平均 Pearson 是 `0.988137`，一个全局尺度后的 relative-L2 是 `0.146214`；发布 CGLS-TV 分别是 `0.980959` 和 `0.203872`。

这里最容易误读的是“PASS”。独立审计指出，冻结协议没有给 ground-truth Pearson/L2 设置门，所以 0.988/0.146 是 post-open 描述，不是触发 PASS 的条件；v0 也只有初始开发合同，不能叫预注册。机器状态保留，但人工结论必须更窄。

审计还指出更实际的问题：当前 forward 是 NumPy `gradient` + SciPy `map_coordinates`，没有 autograd、伴随、JVP/VJP 或梯度一致性测试。也就是说，它能做值重放，不能直接接网络训练。下一步先做 D0.5：Torch/JAX 值一致性、有限差分方向导数、伴随或 JVP/VJP 点积测试。新 GT 门只能在独立 phantom 上冻结，不能看完这一个 phantom 再补线。

这里还有一个机器可读的坑：旧 summary 已经落盘，里面仍写“下一步做 overfit smoke”。我们没有改历史 JSON，而是新增审计覆盖层，把当前有效决定写成 `training_authorized=false` 和 `d0_5_required=true`。以后脚本或人读取这轮结果，必须先看覆盖层，不能只摘旧 summary 的一句话。

还有一个负结果值得保留。ground truth 的重放误差比发布 CGLS-TV 低 `0.057658`，所以“CGLS-TV 通过反演吸收了有限孔径/光流失配，反而更贴观测”这个想法没有得到支持。14.6% 的余差可能来自有限孔径、图像扭曲与光流、离散化、单 ray 近似或 resize 差异，但本轮没有能力判定是哪一个。

D0.5 通过后才允许优化烟测：发布 CGLS-TV 只作固定锚点；低分辨率 voxel、Fourier MLP 和 B2 base + 有界 residual 使用同一值语义。正式排序只能预先选择一个 primary resource budget，并完整报告 steps、ray calls、wall、参数和内存；不能假装这些预算可以同时严格匹配。它仍只用于发现优化和失败模式，只有一个 phantom，不能写优越或泛化。

完整表格、六个师兄接口问题和复现入口见 [公开 Phantom 1 值重放诊断报告](open_nir_bos_d0_forward_identity_result_2026-07-23.md)。

**突破监测：没有突破。新增的是一个先失败、再按冻结语义修复的 D0 值重放诊断，以及独立审计发现的 GT 冻结门和可微实现缺口；新算法、训练、真实重建、优越性、泛化与论文成功仍为 0。**

## 185. 尺子终于可以反向传播，但还没有开始解题

上一轮 D0 的 NumPy/SciPy forward 能重放值，却不能把梯度传给网络。这一轮先把 D0.5 的 CPU64、MPS32、有限差分、JVP/VJP 和 ROI 外零梯度阈值写进 JSON，协议 SHA-256 固定为 `80df1e59...d71b3`，然后才第一次运行。

核心实现没有搬作者 CUDA 代码。它显式写了 `2h/N` 的 cell-centred 网格、`np.gradient(edge_order=2)` 的边界公式、三线性 border 插值、ray--AABB 和 midpoint quadrature。这样做比调用一个黑盒 `grid_sample` 麻烦，但 XYZ 轴序、边界和无效 ray 的每一步都能和 SciPy 对照。

CPU64 的合成场值误差是 `4.58e-16`，有限差分/JVP 是 `2.45e-10`，JVP/VJP 对偶缺陷是 `1.97e-15`。公开 Phantom 用 12 个视角、768 条固定 ray、128 个积分点复跑发布 CGLS-TV 场，Torch/SciPy relative-L2 是 `3.06e-16`。这些数字的含义是“两把数值尺子一致”，不是“重建误差只有 1e-16”。

第一次 MPS v0 没有假装成功。审计统计器在 Apple GPU 上直接请求 float64，后端报错；错误结果目录保留。修复为先搬到 CPU 再转 float64 后，v1 的 MPS 值误差 `9.16e-8`、有限差分/JVP `1.86e-4`、对偶缺陷 `4.98e-7`，都过了事前阈值。14 个本地测试通过，证据包复核器又重跑公开射线、不同形状随机场和 autograd gradcheck，88/88 通过。它直接导入被审 forward，所以这是包一致性与异构复跑，不是第二套独立实现。

这张绿灯只授权一件很窄的事：先在 CPU64 上比较同一个公开 Phantom 的 low-resolution voxel、Fourier MLP 和 Fourier base + bounded residual matched-budget 优化烟测。MPS 当前只是小型合成兼容性 PASS；公开场 mini-batch 前向、反向、有限性和内存门没过以前，不授权 MPS Phantom 训练。它也不授权 FNO/DeepONet 排行榜，因为这里只有一个独立三维函数；不授权 geometry 或 curved-ray 梯度，因为当前导数只对固定 rays、改变 voxel field 成立。

审计还留下三个要正面写出的口子：正式 benchmark 前应补完整 6,144 rays 与逐视角尾部；CPU32 需要单独选择有限差分步长，不能照搬 CPU64 的 `1e-6`；Fourier MLP 参数梯度还要在 B1 用多个方向复查。完整数字和下一组三臂设计见 [D0.5 可微前向门禁结果](open_nir_bos_d0_5_torch_forward_result_2026-07-23.md)。

**突破监测：没有突破。新增的是一个 CPU 主门通过、MPS 合成兼容门通过的可微薄射线实现、保留的第一次 MPS 失败、公开值重放和 88 项包一致性/异构复跑。训练、三维重建、算子泛化、真实 OERF、算法优越和论文成功仍为 0。**

## 186. 小尺子换成了公开大体场，但仍只准做一次短烟测

D0.5 在很小的合成场上证明 Apple MPS 能反向传播，却没有回答 `140 x 294 x 140` 的公开体场会不会爆内存、三线性 gather 的反向累加会不会漂、以及公开观测形成的 loss 能不能给出与 CPU 一致的三维场梯度。M0 就只补这道设备桥，不训练网络，也不调学习率。

正式运行前先冻结 12 个视角、每视角 8 条射线、128 点 midpoint quadrature、chunk 24、三条无效射线、三次完整 MPS 重复和 25 个数值门。协议 SHA-256 是 `43fef428...55ea8`，事前 commit 是 `6ea5ba7`。第一次 v0 在方向导数的审计标量转换处报错；修一次后，v1 又在主 loss 的同类转换处报错。两个失败包都保留，并且都没有打开任何训练授权。第二次修复只是把审计转换统一成“先搬到 CPU，再升到 float64”，没有改协议、射线、阈值或物理 forward。

v2 才完成正式判决，25/25 通过。CPU32 与 MPS32 的 prediction relative-L2 是 `9.60e-8`，MPS32 与 CPU64 的 field-gradient relative-L2 是 `1.08e-4`，MPS 方向导数误差是 `2.76e-5`。三次完整重复的 prediction 最大漂移为 `0`，gradient 最大漂移为 `9.92e-10`；三条无效射线的输出和场梯度都严格为零。四个同步采样点看到的最大 driver allocation 增量约 `1.016 GiB`，占事前 2 GiB 门的 `50.8%`，清理后 current allocation 增量为零。这里的内存证据不是连续 profiler 峰值，不能扩写成“长训练内存稳定”。

保存包又通过 152/152 项一致性检查：协议和源码绑定、v0/v1/v2 精确文件集、25 行阈值表、12 个射线选择哈希、17 个公开输入哈希和越权 claim 都被复核。验证器没有实现第二套物理 forward，也没有重跑 MPS；结果包没有保存 576 万体素的完整梯度，所以这仍是包一致性审计，不是独立物理复现。

现在唯一新增的机器授权是 `mps_single_public_phantom_voxel_smoke=true`：只能对同一个 opened Phantom、同一组冻结 96 条射线、固定 straight rays、ROI、128 点积分和 chunk 24 做很短的 voxel-field optimizer/failure-mode smoke。任意 batch、Fourier MLP、bounded residual、10 步内存稳定、完整训练、三维重建、operator learning、跨 field/geometry 泛化、真实 OERF、优于 DeepONet/FNO/NeRIF/NIRP、论文成功和突破仍全部为 false。

下一步分两条线：MPS M0.1 只做这 96 条射线上的短程 voxel 优化并用 CPU 复核 checkpoint；CPU D0.6 则事前冻结 `S0_VOXEL / S1_FOURIER / S2_BOUNDED_RESIDUAL` 的 matched-budget 筛选。B1/B2 要进入 MPS，必须另过神经参数多方向梯度和连续 optimizer-step 内存门。完整数字见 [M0 公开大体场 MPS 门报告](open_nir_bos_m0_public_mps_result_2026-07-23.md)。

**突破监测：没有突破。新增的是公开大体场上 CPU64/CPU32/MPS32 的值与 voxel-field 梯度桥、两个保留的工程失败和 152 项保存包审计；优化收敛、重建、神经参数梯度、算子学习、泛化和论文成功仍为 0。**

## 187. 三个模型还没开跑，先把它们关进同一间考场

M0 过门后，最容易犯的错误是立刻训练 Fourier MLP，然后拿一个看起来更小的误差说“神经表示更好”。D0.6 先把三条 arm 的考场锁死：`S0_VOXEL` 有 31,875 个参数，`S1_FOURIER` 有 31,873 个，`S2_BOUNDED_RESIDUAL` 有 31,970 个，最大差只有 97，也就是 0.3043%。它们都必须先输出同一个 `27 x 53 x 27` 栅格，再走同一个 128 点直线 ray forward，不能给神经场额外连续采样优势。

公开 Phantom 的 12 x 512 条射线继续复用，但每个视角用确定性 hash 分成 400 fit、56 dev 和 56 audit。这里发现并堵住了一个泄漏口：如果 TRAINER 能打开整张公开观测图，它理论上就能读到 audit 像素。因此现在由独立 `SPLIT_BROKER` 一次性读取 manifest、二维 mask 和 12 张图，写出互不重叠且带哈希的 fit/dev/audit 私有 shard 后退出；TRAINER 只能挂载 fit/dev，九个 checkpoint 封存后 AUDITOR 才能挂载 audit。`n_GroundTruth.mat`、`flowcglsTV.mat` 和 `3Dmask.mat` 仍只能由最后的 GT scorer 打开。14 个 broker 输入、3 个 postseal 体数据、12 组 selection/split hash 和 10 个固定 batch hash 已经单独冻结。

公平账本不再只数“训练步数”。每个 arm/seed 都有 4 个学习率候选、4 步短筛选、4 次完整 dev forward 和 110 次正式更新，合计 130 次 forward、126 次 VJP、123,648 次 ray evaluation；乘 128 个积分点后，主预算严格为 `15,826,944 RQWU`。最终 fit/dev/audit、GT、CGLS-TV anchor 和 256 点积分诊断都另计成本，不能假装免费。

`S2` 也不能暗中多学：前 80 步残差分支关闭，必须和同 seed 的 `S1` 共享逐字节相同的 trunk/base 初值与轨迹；第 80 步才冻结 base，用 `rho = 0.25 * max(P95(|e*base|), 1e-3)` 固定残差幅度，最后 30 步只训练 97 参数 residual head。prefix hash 不同就直接失败，不解释模型效果。

机器协议 SHA-256 已冻结为 `1ef2aa6f...8b50d8`，输入身份 SHA-256 是 `495c8f5e...ffff8e`。预提交机械验证已经检查 132 项，其中 129 项通过；三个预期失败分别是协议、输入身份和验证器源码还没有进入当前 Git HEAD。这是预期的先锁设计状态；把三者和测试提交后，才允许重跑 preflight。runner、dry-run 和正式训练授权此刻仍全部为 false。

完整门包括 S0 至少把零场 fit MSE 降低 20%；S2 相对 S1 的 field 中位配对改善至少 5%，gradient/front 至少一项改善 2% 且其余不退化，audit median/p90/worst 分别守 2%/2%/5%，至少两个 seed 同向且任何 seed 不得伤害超过 5%。任何一条没过都记录 NO-GO，再按事前规则回退 S1 或 S0。

**突破监测：没有突破。新增的是一个真正可证伪、参数与 RQWU 都对齐、GT 与 audit 分角色隔离的单场筛选协议；模型尚未训练，三维重建、算子学习、跨场/跨几何泛化、真实 OERF、算法优越和论文成功仍为 0。**

## 188. 审计真的抓到了漏账，所以先修考场，不急着开跑

上一节记录的是第一版冻结状态，不是最终绿灯。独立审计发现，LR 规则明明要求在 step 4 checkpoint 上用四个 fit batch 的 1,920 条射线统一复评，旧预算却只算了四次训练 forward 和完整 dev forward。四个训练 loss 来自四个不同 checkpoint，不能拿平均值冒充 step-4 fit-union loss；如果直接训练，账面和真实工作量就不一致。第一版 commit `01ce64d` 继续保留，没有覆盖，也没有产生任何训练结果。

v1.1 把每个 LR 候选额外的一次 1,920-ray forward 写进协议。每个 `arm x seed` 因此改为 134 次 forward、126 次 VJP、131,328 次 ray evaluation，128 点积分对应 `16,809,984 RQWU`。这个量准确叫 matched projection-work budget，不是端到端 FLOPs；未来仍要单列参数计算、wall time、RSS 和 postseal 评分成本。

公平性也补了三道锁。第一，三条 arm 都乘同一个解析边界 envelope，不再让 S0 和 S1/S2 带不同边界先验。第二，S1/S2 不只比较 step 0/80 哈希，而是绑定每个 LR trial 和前 80 步的 batch、loss、梯度、参数与 Adam state；第 81 步给残差头新建 step-0 optimizer。第三，G0 逐 arm 判有效，G3 只有 field、gradient/front 和 audit median/p90/worst 全部不伤害时才回退选 S1。

audit 泄漏也从一句话变成四角色合同：broker 才能看 14 个原始公开输入，trainer 只收 fit/dev shard，九个 checkpoint 封存后 auditor 才看 audit，最后 GT scorer 才看真值、CGLS-TV 和三维 support。不过这仍只是合同，真实进程挂载隔离和负向测试还没实现，所以不能开训练。

修复版协议 SHA 是 `28025859...270dd`，输入身份 SHA 是 `df7806ab...57688`，提交为 `f721eca`。提交前预检 133/136，唯一三项失败恰好是协议、身份和验证器还没进入 HEAD；提交后同一验证器 136/136 通过，并绑定三者源码哈希与外部公开 release `a385cce...f5604`。七个定向测试和 mutation tests 也全部通过。

这次最有价值的结果不是一个更低的 loss，而是避免用错误预算跑出一个看似公平的模型比较。下一步先实现 fail-closed split broker、runner、参数梯度和 budget tests，再做不产出科学结论的 dry-run。正式训练授权仍为 false。

**突破监测：没有突破。新增的是训练前被抓住并修复的预算漏洞，以及 136/136 的设计/输入/语义预检；split broker、runner、训练、三维重建、算子学习、泛化、真实 OERF、算法优势和论文成功仍为 0。**

## 189. 数字全绿也不能直接上线：隐私门又挡住了一次

第一次 136/136 保存包在本地逻辑上是有效的，但 Pages 过滤构建拒绝发布。原因不是算法或协议，而是 `validator_uses_project_venv` 的 detail 直接保存了本机 `.venv` 的绝对路径，其中带有 `/Users/...` 用户目录。它没有密码，也不是外部数据，但仍属于不该出现在公开证据里的本机身份信息。

这次没有放宽 Pages 规则，也没有手工删掉 validation 的一行后假装原包可发布。v1 整包移到私有隔离区保留；验证器改为只输出相对标识 `.venv`，如果环境不符则输出固定的 `OUTSIDE_REQUIRED_PROJECT_VENV`，不回显真实路径。这个修复提交为 `1f0136c`，没有改变协议 SHA `28025859...270dd`、输入 SHA `df7806ab...57688`、136 个检查、预算、拆分或任何判决门。

重新生成的 publish-safe v2 仍是 136/136，validation SHA 为 `9693b18c...2450c`，三项 source-binding 与验证器 SHA 都指向 `1f0136c`。全文搜索确认不含 `/Users/`、用户名、VPN 账号、密码或本地绝对路径。接下来 Pages 构建必须从包含 v2 的干净 HEAD 重做，manifest commit 对不上就不得部署。

**突破监测：没有突破。新增的是一次被保留的发布隐私失败和一个可公开的 136/136 v2；训练、重建、算子学习、泛化、真实 OERF、算法优势和论文成功仍为 0。**

## 190. 门真的锁上了，但模型一行都还没训练

上一轮只是把“TRAINER 不能看 audit 和真值”写进协议。这一轮第一次把这句话变成真实进程限制。split broker 分两个角色：stage 只能读取冻结的 transforms、mask 和 12 张观测图，共 14 个文件；seal 只能读 stage 生成的最小 capability tree，整个外部 release 对它都是禁止读取的。两个角色也都不能联网。

第一次真实 stage 没有过。原因不是输入 hash，而是 Seatbelt 规则太严：连路径元数据也被禁止，worker 在做 canonical path 时就收到系统拒绝。这个失败发生在任何 shard 创建之前。我们没有关掉沙箱，而是把权限改细：目录和文件元数据可以读，文件内容仍默认拒绝，只为 14 个精确路径开放。随后又加了真实网络负向探针，只有文件和网络两种阻断都成立，worker 才能启动。

成功的 v2 从同一个公开 Phantom 生成 4,800 条 fit、672 条 dev 和 672 条 audit。每个条目带 view id、flat pixel id、三分量 observation、ray origin/direction 和相机中心对齐量。三组身份没有交集，并集严格等于冻结的 6,144 条 rays。audit NPZ 与 fit/dev 物理分开，三个文件都只在本机 `private_library`，网页只公开 count 和 hash。

为了检查“同样的数据是不是每次真的写成同样的字节”，seal 又独立跑了一次。fit、dev、audit 三个 SHA-256 分别稳定为 `bced0252...4c895`、`e8ecfd3a...3520d` 和 `506583d6...524b`，全部 byte-identical。文件先 fsync，目录再 fsync，最后一次原子 rename；目标目录存在就失败，不能覆盖第一次不利结果。

公开验证器把源码 commit、协议、输入身份、私有 manifests、两次 shard hash、Seatbelt 负测、断网和 claim closure 重新串起来，43/43 通过。这里的 PASS 只属于 broker：optimizer steps 是 0，checkpoint 是 0，field/reprojection metric 都没有计算。

**突破监测：没有突破。新增的是一个真实、可重复、会在越权时停止的数据隔离层；模型、runner、三维重建、算子学习、泛化、算法优势和论文成功仍为 0。**

## 191. 审计又拦住了“马上训练”：两个词没定义清楚

broker 通过以后，最诱人的动作是直接开三条 arm。但独立审计在 runner 开写前又找到两个会改变判决的歧义。

第一个是 quantile。旧协议明确 view p90 用 NumPy `linear`，却没有说 `q=P95(|e*base|)`、S2 correction p90 和 front Hausdorff95 用哪种插值。样本数量有限时，`linear`、`nearest` 或 Torch 默认实现会给出不同的 rho 和门槛值。现在统一冻结为 `numpy.quantile(..., method="linear")`，输入必须有限，结果用 float64 hex 保存。

第二个是“九个 checkpoint”。旧文字要求九个 checkpoint 都封存后才开 audit，但 G0 又允许某个 arm 无效后 fallback。失败的 arm 可能没有 checkpoint，这两条规则同时满足不了。现在改成九个不可变 terminal receipts：每个 arm × seed 要么有 `SEALED_CHECKPOINT`，要么有 `SEALED_FAILURE_TOMBSTONE`。失败 tombstone 保存失败码、最后事件、下一个预期事件、账本前缀和源码/协议 hash；一旦封存，不能后来用重跑 checkpoint 替换。

runner 还额外冻结了 CPU 单线程、ray chunk 64、Adam 的 `foreach/fused/amsgrad` 等实现 flags、S2 residual 使用 selected LR、第 81 步必须新建空 Adam，以及每个成功 arm/seed 恰好 260 个预算事件。S1/S2 必须是两个真实模型和两个真实 optimizer 锁步运行，不能只跑一次后复制一份漂亮 trace。

17 项定向与 mutation 测试已经通过，但这只证明新 overlay 自洽。下一步才是实现三种参数化、AuditedD05Projector、expected-event ledger、step-81 receipt 和失败注入 dry-run。正式训练继续为 false。

**突破监测：没有突破。新增的是在训练前消除 quantile 和失败开封矛盾，并把 runner 的可证伪规则锁死；算法效果仍完全未知。**

## 192. 审计说门还有缝，所以重锁一次

上一节说 broker 43/43，但独立审计没有被这个数字说服。它发现了两个真问题：隐藏 worker 可以不经过 Seatbelt launcher 直接调用，而且 `close_fds=True` 不会关闭 fd 0。如果有人先打开禁止文件，再把它当作 stdin 交给 worker，旧代码仍可读到内容。

这不能反推上一次已经偷读了 GT，但它说明“只能读 14 个文件”的声明超过了当时的证据。我们没有把审计意见改成一句小字，而是暂停部署，保留旧 attestation，然后修底层机制。

现在 worker 必须亲自试读一个获准文件和一个禁止 GT 文件，并亲自试网络。只有禁止读取和网络都返回系统权限拒绝时才能继续。launcher 把 stdin 固定为 `/dev/null`；worker 在读任何数据前盘点 fd 3 以上的描述符，发现一个就失败。把禁止文件当 stdin 的原始反例现在会在入口被拦住。

审计还找到了三个工程缝隙。目录级 `os.replace` 会替换一个已有空目录，所以改成 macOS `renamex_np(RENAME_EXCL)`；private verifier 以前会忽略额外子目录，现在要求完整节点集精确相等；输入路径以前只拒绝最后一层 symlink，现在每个中间组件都要检查。

修复提交是 `0705adbe`，13 个定向测试通过。然后没有复用旧目录，而是建立新 v3 stage，再独立 seal 两次。fit/dev/audit 仍是 4,800/672/672，三个 shard 的 hash 与两次间都逐字节相同。新证据记录 stdin 为 `/dev/null`、额外继承 FD 为 0，而 optimizer 步数仍为 0。

这次最重要的进步不是“PASS 数从 43 变成 53”，而是允许独立审计推翻我们自己的过强结论，再用可复现反例修机制。这比多一个漂亮数字更接近可发表研究的做法。

**突破监测：没有突破。新增的是一次被保留的证据失效、两个 blocker 和五个 major finding 的修复、以及新 v3 确定性 seal；runner、训练、三维重建、算子学习、泛化和论文结果仍为 0。**

## 193. 开跑之前，先证明不会换模型、换光线或重复更新

上一节 broker 已经把 fit/dev/audit 拆开，但还缺一条从模型到梯度再到 optimizer 的可追责链。如果只有一个可调用的 forward，运行时仍可能换了参数对象、改了学习率、把同一次梯度更新两次，或者保留相机/像素编号却替换真实射线坐标。

现在三个参数化都必须通过同一个 `AuditedD05Projector`。VJP 完成时会绑定真正参与反传的参数名、对象和事件学习率；Adam 不能换一组同名对象，也不能临时改 LR。optimizer 已经改参数、但 `OPTIMIZER_COMMIT` 落盘失败时，该 run 会永久中毒；第二次调用不能再更新一次。

射线也不再只锁 view/pixel ID。batch contract 现在同时锁 origins、directions、view IDs 和 flat pixel IDs 的数值内容哈希。回归测试专门伪造了“编号不变、光线起点改变”的 batch，注册表必须拒绝。但 broker 还没给正式 batch 签发这个 content hash，所以正式注册表当前会主动 fail closed，这是正确行为。

定向测试是 `64 passed`。随后新建 v3：S0/S1/S2 各 3 个 seed，共 9 个独立子进程，每个只跑一对真实 D0.5 `FORWARD/VJP/Adam`，也就是 2/260 个事件。9/9 终止票据和 terminal package 复核通过，但终态故意是 `FAILED_SEALED`，audit unlock 仍为 false。独立证据验证器又复核 11 个源文件、9 个正式日程族、2 个复合批次和 v3 保存包，`failure_count=0`。

这次 PASS 只说明一对机械链能干净地完成并干净地停止。正式 260-event trainer、4 个 LR 筛选、S1/S2 的 80 步 lockstep、S2 的 step-81 optimizer 重建、checkpoint 内容审计和科学评分都还没有。下一步不是把 2 事件粗暴扩成 260，而是先让何远哲师兄确认真实 callable、straight/curved residual 层级、JVP/VJP、坐标/单位、标定版本和认可的强基线。

完整证据、五个入门概念和师兄问题见 [D0.6 runner v3 机械验证报告](open_nir_bos_d0_6_runner_v3_result_2026-07-23.md)。

**突破监测：没有突破。新增的是参数、学习率、射线几何、VJP、optimizer 更新和失败票据的可证伪机械链；算法效果、重建、泛化和论文结果仍完全未知。**

## 194. terminal 端点终于不再只看序号

上一节的 v3 已经能证明“一对 forward/VJP/Adam 事件跑完并封存”，但第三轮独立审计问了一个很尖锐的问题：terminal 说自己停在第 5 个事件，系统到底只检查了数字 `5`，还是知道第 5 个事件应该是什么？

旧 v7 的答案还不够好。它会检查 sequence 连续，却可能接受一个序号正确、内容荒唐的伪事件，例如把训练 step 写成 `999`。如果最后一页账本能这样伪造，那么“下一个事件是什么”也可能来自另一条学习率分支。另一个小问题是 batch identity 缺字段时会漏出原始 `KeyError`，而不是给出稳定、可审计的合同错误。

v9 给 9 个 `arm x seed`、每个 4 条学习率分支、每分支 260 个事件建立了精确索引。索引不只保存序号，还保存事件类型、阶段、step、batch、学习率和事件哈希。terminal 的最后事件与下一事件必须逐字段命中同一条分支，不能把一个 LR 的末尾接到另一个 LR 的开头。坏 identity、越过 260 的序号、任意 checkpoint LR、缺失或被改写的 manifest 都会稳定失败。

独立复查随后又找出三个实现缝隙：写状态前没有每次重读 manifest；checkpoint LR 只要求大于零，没有要求属于冻结四候选；超出 260 的 sequence 会泄漏 `IndexError`。三处都修完后重新复查，定向问题全部关闭。v8 其实已经补上核心端点绑定，但 manifest 被改时命令行仍打印 traceback，而不是规范的 `FAIL` JSON，所以 v8 保留为历史，重新生成 v9。

当前机器证据是 `103 passed`、16 个合同字段、9/9 个合成 worker、每个 2/260 个事件，终态仍是 `FAILED_SEALED`。这句话容易被误读，所以拆开说：

- `FAILED_SEALED` 不是模型效果失败，而是本轮故意不创建 checkpoint、不开放 audit，并把“尚未训练”封存成不可冒充成功的终态。
- “端点属于冻结 schedule”不等于“前面 1 到 N 的全部日志都由 state 层独立重放”。当前 prefix hash 仍来自 worker receipt；完整 journal snapshot 与逐事件重放是下一道门。
- Python 内的私有属性和 capability 是防误用的正确性保护，不是安全沙箱，也没有外部签名、跨进程锁或第三方透明日志。
- 合成射线内容已经绑定；正式实验射线还缺 broker 签发的 geometry content hash，所以真实 batch 会主动拒绝 seal。

现在最值钱的下一步不是盲目把 2 个事件扩成 260 个，而是拿到何远哲师兄的 tiny 真实 callable：确认 straight/curved forward 的层级、JVP/VJP、坐标单位、几何标定、当前强基线和最痛的物理失配。拿到这些后，先在真实 fixture 上重做一对事件，再补可重放 journal、四 LR reset、S1/S2 前 80 步 lockstep、S2 step-81 optimizer 重建和 checkpoint 内容审计。

完整审计演化、证据上限和需要问师兄的问题见 [D0.6 runner v9 机械验证报告](open_nir_bos_d0_6_runner_v9_result_2026-07-23.md)。

**突破监测：没有突破。新增的是 terminal 端点与冻结日程逐字段绑定，以及审计发现后修复的三处状态机缝隙；formal trainer、重建指标、算法优势、算子泛化、真实 OERF 和论文结果仍为 0。**

## 195. 师兄终于替我们砍掉三条岔路：只做 C，但 C 还没有跑出结果

前面做了很多反问题、导数和证据门，是为了防止把一个漂亮的网络输出误写成三维重建成功。但一直有个更大的问题没解决：本科毕设到底应该主攻有限视角、forward 失配、4D 时序，还是计算成本？这次何远哲师兄直接选择了 C：

> 算子学习做 warm start，在最终精度相同的前提下降低 BOST 三维重建成本。

这句话把项目变得具体了。网络不再负责“一步替代物理重建”，而是根据多视角 BOS 观测和几何给出一个更好的三维初值。后面仍接同一个 CGLS/PCGLS 或组内认可的物理迭代器。我们真正比较的是在终点误差等价时需要多少次 forward/adjoint、多少 wall time 和多少内存，而不是只比较网络单次输出的 relative-L2。

师兄还纠正了一个会浪费几个月的误区：不用从头学习并运行完整三维 CFD。网格、燃烧模型、边界条件、收敛与算力可能让一个可信算例就耗掉数月。当前应使用现成公开 CFD 轨迹，例如 PoolFire，把其中的三维密度场当作数字真值；我们只补够用的数据常识，包括 `rho`、坐标轴、spacing、时间、单位、裁剪、插值和物理范围。

独立审计随后补上了一个容易被忽略的物理坑：BOS 主要看折射率梯度，对整个场加一个常数往往不会改变偏折。以后不直接把绝对 `rho/n` 当重建目标，而是先用实验可得的环境、flow-off 或已知边界冻结 reference，重建 `Δrho/Δn`。若没有物理 reference，就让所有方法共享同一个零均值或边界 gauge，并报告梯度指标。绝不能只让网络从功率/工况标签猜背景均值，再用绝对场 L2 获得经典方法不可能得到的优势。

师兄给的本地 BOS 模拟工具已经做了私有原件/工作副本备份，没有进入 Git。接口初审显示，它会从三维场和九视角相机生成参考光线、偏折量与曲线路径；它更像“从 CFD 场生成 BOS 偏折观测”，还不能自动等同于完整的点阵图渲染和可用于反演的 forward/adjoint。当前必须请师兄确认四件事：

1. 折射率场函数应返回密度、折射率还是折射率增量；
2. 空间步长和两个比例常数的物理意义与单位；
3. 保存偏折量是像素、归一化坐标还是偏折角；
4. 最终 BOS 渲染、forward、adjoint 和经典重建基线是否另有代码。

公开 PoolFire 试验轨迹已经用可断点续传的后台任务下载，目标文件约 6.43 GB，完成后还要同时通过精确文件大小和 SHA-256，才允许进入读取与转换。低内存检查器已经能在不把 6 GB 数组全部装进内存的情况下读取 NPZ 元数据和数组头，并且默认只在报告中保存文件名，不回显本机绝对路径。REALM 原任务是“用当前 CFD 场预测未来 CFD 场”，我们的任务则是“用 CFD 密度场生成 BOS 观测，再从观测重建三维场”，两者不能混写。

第一版算法也被刻意压小。C0 暂定为 Adjoint-Residual Warm Start：冻结直线/线性阶段只复用一次反投影 `A^T y`，把它连同坐标或几何输入小型 FNO/3D U-Net，输出三维初值，再接固定物理迭代器；若正式链路使用随场弯曲的非线性光线，则必须改用参考态 `J^T[y-F(x_ref)]`，并统计 forward/JVP/VJP。旧方案曾为每个样本额外消耗 13 次 forward 和 13 次 adjoint 来造特征，特征成本已经可能吃掉全部加速收益，所以不再沿用。

强基线顺序冻结为 zero、`A^T y`、PCGLS/CGLS、简单阻尼或插值，以及容量匹配的 FNO/DeepONet 初值。评价必须拆成两张账：部署主表只能用 validation 冻结的迭代数、measurement discrepancy 或保留相机等可见量停止，再事后检查终点是否等价；PoolFire 有真值时可以另外画“第一次达到 field 阈值”的 oracle time-to-target，但它只估计 headroom，不能冒充在线停止。两张账都报告 median、p90、worst 与 harm rate。只有 C0 在未见 PoolFire 轨迹上既不损最终精度、又稳定降低部署主账总成本，才升级到 C1 的可观测 Krylov 子空间或 C2 的短程迭代轨迹损失。

学习主页和三分钟汇报页也已经从“四选一”改成唯一 C 路线。桌面与 375 像素移动端都做了真实渲染检查，没有横向溢出或控制台错误；复制给师兄和蔡老师的两段文字也可直接使用。JMLR 的 fixed-point warm start、NOWS、super-fidelity 和逆声散射 warm start 被加入第 10 周核心阅读，用来提醒我们：**warm start 本身已经不是创新点，BOST 物理、几何条件化、可观测子空间、严格成本账本和独立迁移证据才可能形成贡献。**

**突破监测：没有算法突破。真正新增的是师兄锁定 C、公开数据与私有模拟工具的数据链、可恢复下载、低内存检查器、唯一 C 学习路线和两周最小闭环。速度提升、优于 FNO/DeepONet、跨工况泛化、真实 OERF 与论文成功仍全部未证明。**

## 196. 6.43 GB 压缩包其实会展开到 9.31 GB，所以先修数据桥

师兄说“网上找点数据就行”，并不代表下载一个 NPZ 后直接 `np.load` 就完成了数据准备。PoolFire 首条 train trajectory 的公开压缩文件约 6.43 GB；从已经验证连续性的下载前缀读取 ZIP local header，再从 deflate 流起点解出 NPY header，得到真实数组：

```text
member = data.npy
shape = (101, 9, 80, 80, 200)
dtype = float64
order = C
numeric payload = 9,308,160,000 bytes
```

metadata 同时确认 101 个时间点、9 个变量和 11/2/2 条 train/val/test trajectory，`rho` 是 channel 5。旧 case YAML 仍写 21 个时间步，而数据目录里的另一份 101-step YAML 又给九个变量几乎相同的异常大统计量，所以两份都不能直接拿来归一化。

本机虽然有 32 GiB 内存，但当时可用内存只有约 6 GiB。即使强行加载 9.31 GB 数组成功，类型转换、网络输入、物理算子和求解器工作区也会继续复制数据。因此新增流式 extractor：先核对完整 archive SHA，再按 `(time, channel)` 顺序解压；每次只保留一个三维场，只把 rho 写入 float32 memmap。默认 stride `(2,2,4)` 会产生 `(101,40,40,50)` 的 rho bundle，数值 payload 约 32.32 MB。

这个 extractor 默认拒绝 test trajectory，检查 metadata split、shape、dtype、C-order 和唯一 `data.npy`，读到 member EOF 触发 ZIP CRC，再写出 rho、coords、times、manifest 和 checksums。目标目录只有所有步骤成功后才原子出现；SHA mismatch、shape mismatch 或旧输出存在都会停止。两套 Python 环境合计 13 项定向测试通过。

后台 watcher 也已启动，但它只等下载状态变为 `complete`。下载脚本必须先同时通过精确文件大小和公开 SHA-256；失败状态不会绕过。rho bundle 的 manifest 会明确写“绝对 CFD 密度、reference/gauge 尚未应用、单位未确认”。因此下一步仍是核对 rho 数值与单位，向师兄确认 `rho/n/n-1/Δn`、比例常数和偏折单位，再做常数场/线性场 smoke 与 adjoint dot test。

网页发布也完成了结构修复。当前 GitHub 方案不支持从私有仓库更新 Pages，因此私有源仓与公开静态发布仓已经拆开：公开仓只有经过 fail-closed 构建器过滤的单提交静态产物，没有源码历史、私有工具、PDF、VPN 内容或本机路径；原分享 URL 保持不变。公开页面与本地产物逐文件 SHA 一致。

**突破监测：没有算法突破。新增的是首个真实 PoolFire 数组头、9.31 GB 内存风险的定量确认、13 项通过的流式 rho 数据桥，以及源码私有/网页公开的可持续发布结构。完整 archive SHA/CRC、rho 数值、BOS forward、重建、warm-start 提速、泛化和论文成功仍未证明。**

## 197. 独立审计把数据桥退回重修：不让“能解压”冒充“物理数据已合格”

上一版数据桥通过了 13 项测试，但两名独立审计者仍找到了会制造假证据的问题。最严重的是：脚本先按路径计算 SHA，随后又按同一路径重新打开；如果文件恰好在两步之间被替换，manifest 可能记录旧哈希、实际却读取新内容。metadata 也只记录了文件名，没有把具体版本的 SHA 绑定到输出。损坏 ZIP 在某些读取位置还可能抛出原始 traceback，把本机路径写进日志。

这轮没有把审计意见记成“以后再修”。提取器改为只打开一次 trajectory，用同一个文件描述符完成前置 SHA、ZIP/NPY 读取和末尾 SHA 复验；metadata 必须给出预期 SHA，并从已哈希的同一份字节解析。ZIP CRC、float32 overflow、Fortran-order、路径替换、并发输出、metadata mismatch 和错误路径脱敏都新增了反例测试。输出目录采用独占创建，`READY.json` 最后提交；存在目录或没有 READY 都不得被后续任务读取。

物理审计又指出一个更隐蔽的问题：BOS 依赖密度梯度，直接用 stride `(2,2,4)` 抽点会改变火焰前沿强度。如果合成观测和反演都沿用这个失真版本，仍可能得到自洽但没有现实意义的漂亮结果。因此原始桥改为保存完整 `(101,80,80,200)` float32 rho，数值 payload 约 517.12 MB；低分辨率副本以后必须由单独冻结的抗混叠或体积平均算子生成。提取时会检查全部 101 帧、全部 full-resolution rho，而不是只检查最终抽中的点。

测试现在在两套 Python 环境均为 20/20。审计还确认 metadata 坐标三个轴均降序，数值跨度约 `1.2 x 1.2 x 3.0`，与 README 的 `3 x 3 x 3 m³` 描述冲突；两个 PoolFire test case 也只是功率与尺寸的组合留出，不能写成“未见功率 OOD”或“未见尺度 OOD”。这些都进入了下一道 G0 门：单位、轴方向、cell center/edge、参考态、光学参数和独立 forward 必须冻结后，才允许生成训练对。

**突破监测：没有算法突破。新增的是一轮真正改变实现的红队审计、20 项通过的输入完整性与发布门、full-resolution rho 合同，以及对坐标冲突、组合留出和 inverse crime 的明确限制。完整 archive SHA/CRC、真实 rho 统计、可靠 BOST 观测、C0 训练、同精度提速、组内迁移和论文结论仍为 0。**

## 198. 第一条公开 PoolFire 轨迹终于过了完整数据门

下载最终到达公开声明的 `6,428,997,975 bytes`，独立复算 SHA-256 得到 `6080ddcc...81383c`，与公开值逐字一致。脚本随后用同一个已经哈希的文件描述符完成 ZIP/NPY 全量读取和末尾复验；`data.npy` 的 CRC、metadata SHA、派生数组 checksums 与 `READY.json` 全部通过。下载与抽取的后台任务随后停止，避免 keepalive 在成功后继续重复校验 6.43 GB 文件。

full-resolution 输出是 `(101,80,80,200)` 的 float32 rho，共 `129,280,000` 个值。独立 mmap 扫描没有直接相信 manifest，而是重新检查全部体素：finite 与正值比例都是 100%，min/max/mean/std 分别为 `0.1889829934 / 1.1793500185 / 1.1608747931 / 0.0605809878`。每帧均值只在 `1.1604899379–1.1611630479` 之间变化；这说明背景占据了大多数体积，也提醒后续不能只报全场 relative-L2，否则“输出接近常数背景”可能得到虚假的好分数。

时间轴是 30 到 32 的 101 个点，步长约 0.02，但单位仍未知。x/y 轴从 `0.5925` 降到 `-0.5925`，z 轴从 `2.9925` 降到 `0.0075`，三个 spacing 都约为 `-0.015`。它与 README 的 `3 x 3 x 3 m³` 仍有冲突，所以现在只能说“公开 CFD raw bridge 完成”，不能说“物理 BOST 数据完成”。下一步必须冻结坐标语义、rho/time 单位、参考态、Gladstone-Dale 条件与独立光学 forward。

**突破监测：这是数据工程门的真实通过，不是算法突破。新增的可靠事实是完整 source SHA/CRC、129,280,000 个 rho 的全量有限性/正值、四文件 checksum 和 READY 均通过。BOST 观测、forward/adjoint、经典三维重建、C0 warm start、matched-accuracy 加速、组内迁移和论文结论仍为 0。**

## 199. 守恒不等于看得清：红队把 `(2,2,4)` 从主方案撤了下来

full-resolution rho 通过后，第一版低分辨率草稿采用 `(2,2,4)` 块平均，得到 `40×40×50`。它确实把均匀网格离散和保留到 `1.90×10^-10` 的相对误差，但独立审计指出：原网格三个方向的数值 spacing 都约为 `0.015`，这一选择会人为制造 `0.03×0.03×0.06` 的各向异性网格，额外抹平竖直火焰前缘。

因此实现当场改为默认 `(2,2,2)`，得到 `101×40×40×100`、约 64.64 MB 的等距候选。三个降序坐标轴和 rho 沿相同维度一起反序，输出轴全部升序；帧 0/50/100 又从 full-resolution rho 独立重算，和候选逐点完全一致。脚本也不再把离散和等价写成“质量守恒”：rho 单位和 cell-center 语义没有权威证明，所以只能称 uniform-grid discrete integral。

这仍不足以放行训练。新增代理审计分别在 full/coarse grid 上求二阶有限差分梯度，再沿 x/y/z 积分横向 rho 梯度，并把 full-resolution detector plane 限制到 coarse plane。它不是相机标定后的 BOST，只用来检查“守恒是否掩盖了光学相关结构损失”。三帧结果是：

| 候选 | 梯度 RMS 保留 | 正交 LOS 代理最大 relative-L2 |
|---|---:|---:|
| `(2,2,2)` | `74.45%–75.44%` | `25.06%–26.57%` |
| `(2,2,4)` | `69.22%–70.67%` | `30.43%–35.36%` |

所以 `(2,2,2)` 只是当前最小主候选，`(2,2,4)` 降为审计对照，C0 继续关闭。派生器和代理审计器在两套 Python 环境各通过 10 项定向测试；结果 JSON、图和完整边界已进入 [PoolFire 低分辨率代理证据](poolfire_preprocessing_proxy_evidence_2026-07-23.md)。

一级来源审计又锁定了 G0 的另一半：反应流不能默认 `n-1=K_air rho`。更完整的稀薄气体混合式是 `n-1=rho K_mix(lambda,Y)`，因此 `grad n = K_mix grad rho + rho grad K_mix`。PoolFire 四个组分通道的质量/摩尔/分密度语义和缺失物种闭合尚未确认，固定常数会删掉组分梯度项。当前判决于是拆成：

- `G0-SMOKE = GO`：固定且明确标注的空气 K 可用于常数场、线性场、符号和步长调试；
- `G0-PHYSICS = HOLD`：单位、组分、波长、reference、背景端点/像素语义、straight/curved 和独立反演 forward 未闭合前，不生成论文训练标签；
- warm-start 模型优先输出 `Delta n_0`，而不是直接预测绝对 rho。

完整公式、Tier-A/Tier-B forward 和可直接发给师兄的十二个接口问题见 [PoolFire G0 光学合同](poolfire_optical_contract_g0_2026-07-23.md)。

**突破监测：没有算法突破。新增的是一次改变默认实现的红队否决、首个可复现等距低分辨率候选、首个梯度/LOS 代理负证据和 `G0-SMOKE GO / G0-PHYSICS HOLD` 光学合同。可靠 BOST 观测、经典重建、C0 warm start、同精度提速、跨轨迹泛化和论文结论仍为 0。**

## 200. 第一层 forward/adjoint 终于能过，但它还碰不到 PoolFire

这轮没有训练网络。先把最小问题缩到不能再缩：假设已经有一个节点上的三维 `Delta n` 场，光线沿 x、y 或 z 正方向直线穿过，程序能不能正确算出两个横向偏折角，并且给出真正对应的离散 adjoint？

新算子把容易混淆的东西都写死了：

- 数组顺序是 `[x,y,z]`；
- 坐标和积分权重用 metre；
- `Delta n` 无量纲，输出只叫 small-angle deflection，语义记作 rad；
- LOS 用节点 trapezoid，横向梯度内部二阶中心、边界二阶单边；
- adjoint 是普通 Euclidean 数组内积下的精确转置；
- 不包含相机、像素、背景距离、曲光线、Gladstone-Dale、渲染或光流。

红队特别提醒：错误的 A 和同样错误的 A^T 也能一起通过 dot test。所以验收没有只做一次随机内积，而是同时做了：

1. 非零常数场，检查 gauge；
2. 三个 LOS 轴的线性场，独立解析符号和尺度；
3. metre 改写成 millimetre 后的一致重参数化；
4. 三轴共 60 个 dot cases，其中每轴包含 8 个角点脉冲；
5. JVP/VJP 从 `10^-2` 到 `10^-7` 的中心差分步长扫描；
6. 连续正弦解析场在 9/17/33/65 网格上的收敛。

14 项机器门全部通过。最坏三轴线性尺度 relative-L2 是 `2.39e-15`；60 个 dot cases 最大归一化差是 `2.03e-17`；JVP/VJP 最佳差分别为 `1.19e-14` 和 `2.74e-15`；网格收敛阶是 `2.04 / 2.11 / 2.08`。这说明声明的节点离散在 float64 下按预期工作。

但这一轮同时发现了新的接口阻塞：PoolFire 的 rho 是 cell-centred block mean，本算子接收 node field。两者不能靠 reshape 或默认插值直接连接。下一步要把 cell-centre conservative LOS 与显式 cell-to-node 两条路线并排实现，在独立解析场上比较边界、偏折和 adjoint，再进入任意相机与 curved/straight 门。

完整结果、图和可复现命令见 [Tier-A 直线 forward/adjoint 证据](poolfire_g0_tier_a_straight_evidence_2026-07-23.md)。

**突破监测：没有算法突破。新增的是首个带单位、边界、三轴解析、60 个 adjoint cases 和二阶收敛的 `PASS_TIER_A_STRAIGHT_CODE_SMOKE_ONLY`。G0-PHYSICS 仍为 HOLD，training_authorized=false；PoolFire 光学模型、真实 BOST、经典三维重建、C0 warm start、同精度提速、泛化和论文结论仍为 0。**

## 201. 一个“很自然”的 cell-to-node 接法被解析证据否掉了

上一节留下的接口问题是：PoolFire 是 cell-centred，Tier-A 是 node-field。最顺手的做法看起来是先把 cells 平均/外推到 nodes，调用已经通过的 node operator，再把 detector nodes 平均回 cells。这轮把它和原生 cell-centred、projection-first interior 两条路线一起写成了显式线性算子，并给每一级都配了精确 Euclidean transpose。

三条路线都能通过线性、单位、三轴尺度和 adjoint dot test。但把 cell-to-node composite matrix 真正展开后，问题出现了：LOS 等效权重变成

```text
[1.25, 0.75, 1, ..., 1, 0.75, 1.25] * h
```

总和仍是 `Nh`，所以只检查积分长度完全看不出来。横向非导数方向又暗中加入 `[0.25,0.50,0.25]` 低通，第一格导数退化成 `(f1-f0)/h`。这说明错误或不合适的 forward 与它自己的精确 transpose 可以一起通过点积测试。

为了不依赖同源实现，新增了点采样与精确 cell-average 两套 manufactured solutions。平滑 cell-average 场从 `9^3` 收敛到 `65^3`：native 最低阶 `1.947`、projection-first interior 最低阶 `1.972`，cell-to-node 只有 `1.734`；最细网格上 cell-to-node relative-L2 为 `0.006130`，是 native `0.001673` 的 `3.664` 倍。

火焰前缘代理用 `tanh` 控制厚度。在 `33^3` 网格、10%-90% 厚度约 `8.79 cells` 的 resolved case，native / interior 误差约为 `1.445%`，cell-to-node 是 `3.817%`，高 `2.642` 倍。前缘只有约 `1.10 cells` 厚时，native 误差仍为 `35.05%`，所以被强制标为 unresolved，不得拿来证明网络“恢复了真实细节”。

当前处置因此很明确：

- native cell-centred 进入独立 forward 验证，但继续审计完整边界；
- cell-to-node 退出 truth forward，只保留为离散敏感性反例；
- projection-first interior 暂作第一版 Zero/BP/CGLS/PCGLS 基线候选，统一裁 detector 四周各一格，不虚构边界外场值。

tiny `4^3` 显式矩阵也再次提醒：native 单视角 rank `15`、nullity `49`；interior rank `8`、nullity `56`。精确 adjoint 不等于可辨识，更不等于三维重建成功。

完整公式、两种场语义、前缘分辨率扫描、SVD 与复现命令见 [PoolFire cell-centred 接口判别证据](poolfire_g0_cell_center_evidence_2026-07-23.md)。

**突破监测：这是关键数值发现，不是算法突破。新增的是一条被机器证据否掉的 cell-to-node truth 路线，以及更可信的 interior 基线候选；PoolFire 单位、`rho -> Delta n`、相机、独立 forward、经典重建、C0 warm start、同精度提速、泛化和论文结论仍为 0。**

## 202. 参考正演终于不再偷偷等于 inverse，但还不是 PoolFire 光学真值

上一节确定 projection-first interior 可以做第一版 inverse 基线候选，但如果拿它自己生成观测，再用它自己重建，网络和经典 solver 都会面对一个过分干净的封闭世界。这种情况下即使 warm start 明显更快，也可能只是学会了同一个离散矩阵，而不是学到真实 BOST 问题。

这轮新增了一条故意不提供 `adjoint()` 的参考正演。它接受任意 orthographic 或 pinhole 相机，用单位射线和每条射线自己的正交 `u/v/t` 基，在 metre 坐标中做 forward half-ray AABB clipping，然后直接对连续 `grad(Delta n)` 做复合 Gauss-Legendre 积分。输出只叫两分量 small-angle deflection，同时保存 hit mask、`s_in/s_out`、路径长度、分段数和梯度调用账。像素倍率、背景板、曲线光线、组分折射率和光流都没有偷偷塞进去。

独立红队要求不能只测几条直线。现在 63 条斜视角线性场、289 条二次场和 289 条余弦场都与各自闭式答案对上，relative-L2 分别为 `1.54e-16 / 2.08e-16 / 1.64e-15`。斜视角 Gaussian 的二点 Gauss-Legendre 在渐近区观测阶为 `3.83 / 3.94`，最细步长相对十二点参考误差为 `2.60e-11`。把全部坐标和相机放大 `7.3` 倍、同时正确缩放梯度后，偏折变化只有 `5.47e-16`，说明射线参数确实是物理弧长，不是随意的 near/far 数字。

更关键的是，两条路线在同一个连续解析场上做了显式非同构比较：

| 网格 | inverse vs independent reference |
|---:|---:|
| `9^3` | `7.927%` |
| `17^3` | `2.261%` |
| `33^3` | `0.603%` |
| `65^3` | `0.156%` |

误差约二阶下降，说明它们在有限网格上不是同一个数值映射，但会收敛到同一个连续问题。参考模块的 AST 依赖审计也确认，对 node/cell inverse 模块及其导数矩阵的 import 数是 `0`。这比“给同一矩阵加点 Gaussian noise”更接近真正的 inverse-crime 控制。

不过当前 continuous gradient 仍来自解析 manufactured field，不是 PoolFire，也不是曲线光线生成器。`PASS_ARBITRARY_RAY_REFERENCE_CODE_GATE_ONLY` 只允许我们继续接师兄确认后的相机与 CFD 语义；`G0_PHYSICS_HOLD` 和 `training_authorized=false` 没有改变。下一步可以搭 Zero/BP/CGLS/PCGLS/Direct Operator 的统一接口和合成解析测试，但正式 C0 训练仍要等 `rho/T/Yk -> Delta n`、domain edges、相机和 solver 输出语义闭合。

完整公式、图、解析 oracle 和复现命令见 [任意视角参考正演证据](poolfire_g0_reference_forward_evidence_2026-07-23.md)。

**突破监测：没有算法突破。新增的是一条与 inverse primitive 零依赖、能处理任意直线视角并通过 641 条解析斜射线检查的参考正演代码门，以及有限网格非同构、连续极限一致的证据。PoolFire 光学真值、曲线光线、经典三维重建、C0 warm start、同精度提速、泛化和论文结论仍为 0。**

## 203. C 路线终于有了不会藏成本的统一比赛场

这一轮没有切换方向，只完成师兄确认的 C 路线底座：让 observation-only
Direct/Operator 初值进入同一个 CGLS/PCGLS refinement，再问它能不能以更少的完整
`A/A^T` 调用达到 Zero 强基线的最终精度。

最初版本很快通过了 7 个单元测试，但只读红队找到了几个会把论文结果做假的漏洞：
裸 `cached_projection` 可以伪造零残差；任意 callable 不能被直接称作固定 SPD
预条件器；外部已经算好的 field 可能隐藏 truth 或算子调用；默认去均值也会在
`A=I` 这类均值可观测算子上制造零误差。我们没有带着这些漏洞跑“优势曲线”，而是
逐项修掉：

- projection cache 的 token 只保留 opaque ID；field SHA-256 与 projection 留在当前
  operator 私有注册表，合法 token 后续被加属性也不能改变缓存内容；scale 和求解器
  都执行 one-shot consume，批量运行后注册表必须归零；
- PCGLS 只接受精确 `FixedDiagonalSPD` 类型，并在求解器内直接乘不可写 diagonal，
  不允许子类覆盖成时变 callable；
- Direct initializer 由审计层只传只读 observation 并计时，同时诚实标成
  `CONTROLLED_INPUT_SELF_ATTESTED`，不冒充沙箱证明；
- evaluator 默认不做 gauge；去均值必须带与同一 audited inverse 绑定、额外花费
  `2 A` 的 opaque 数值证书，伪造或换 wrapper 都拒绝；
- independent reference 由实现类型、模块、实例、无 adjoint 和不共享离散矩阵五类
  机器检查共同判定，不接受结果脚本手写布尔值；
- `1...24` 每一步都记录累计 `A/A^T`、推理和墙钟，稀疏点只用于画图，不用于判定
  首次达标。

最终定向测试为 `14 passed`。三视角 stacked inverse 的 12 个 dot cases 最大相对差
为 `5.60e-15`，常数场输出范数 `4.83e-15`，显式 identity-PCGLS 和 CGLS 逐
checkpoint 完全一致。

制造数据没有复用 inverse：连续 Gaussian 梯度经独立 Gauss-Legendre ray integral
生成观测，`9 x 9 x 11` 粗网格 projection-first operator 负责反演。reference 和
inverse 对测试 truth 的投影本来就差 `15.09% / 19.37%`，因此这次明确包含 model
mismatch。

toy ridge Direct 在同族留出系数 case 上 direct-only field error 约 `8.39e-10`，
但一次 coarse CGLS correction 就变成 `9.69e-2`；在留出新模式 case 上，它又会从
`0.2179` 改善到约 `0.1901`。同一个 residual correction 有时伤害、有时帮助，说明
后续真正值得证伪的不是“网络初值能不能好看”，而是能否仅根据部署可见证据限制
correction budget，并在 forward mismatch 下 fail closed。

完整账本、图和复现命令见
[C 路线统一强基线与成本合同证据](poolfire_c_baseline_contract_evidence_2026-07-23.md)。

**突破监测：没有算法突破。新增的是首个 truth-blind、逐 checkpoint 计费、能拒绝伪造 cache/时变预条件器的 C 路线统一求解底座，以及一条 model-mismatch 会改变 refinement 正负作用的可证伪线索。真实 PoolFire/BOST、神经算子、跨轨迹/工况/几何泛化、GPU 端到端提速、峰值内存和论文结论仍为 0。**

## 204. 第一条真实 PoolFire CFD 轨迹进入了 C 路线，但只能叫形态代理

这轮不再用 Gaussian 场出题。经过完整 SHA-256 复核的公开 PoolFire
`p=14kw_size=03` 轨迹实际进入了统一 warm-start/CGLS 账本。高分辨率
`32 x 32 x 64` `rho` ROI 通过连续三线性梯度与 composite Gauss-Legendre
生成三视角数值观测；inverse 只使用严格 `2 x 2 x 2` block mean 得到的
`16 x 16 x 32` 场和另一个 projection-first 离散模块。

runner 现在还把 trajectory、source/metadata SHA、四个 payload SHA、shape、dtype
和时间点数量共同锁成 `realm-poolfire-p14kw-size03-rho-v1`。这不是只核对“某个
checksums 文件存在”；任何身份字段变化都会在 pair generation 前拒绝。

34 个使用帧被按时间顺序拆成四段：25 帧 train、2 帧 ridge selection、2 帧
refinement-depth validation、5 帧 later-time evaluation。相邻角色之间至少空五帧，
没有随机抽帧。最终 ridge 只用前两段共 27 帧拟合；固定 refinement depth 只在第三段
选择，得到 `K=2`。

把 `K=2` 应用到后期五帧时，平均 field relative-L2 为：

- Zero：`0.60835`，成本 `2A + 2A^T`；
- normalized BP：`0.51445`，成本 `3A + 3A^T`；
- ridge Direct warm：`0.41486`，成本 `3A + 2A^T`。

Direct warm 在五帧都优于 Zero 和 BP。更重要的是，它继续迭代到 `K=24` 后平均误差
反而恶化到 `0.48620`，虽然 data residual 继续下降。这把 toy 门里的线索推进到了
真实 CFD 形态：强 forward mismatch 下，coarse solver 会先修正初值，再逐渐把它拉向
错误的 coarse data-consistent 解。

这仍然不能写成算法胜利。full-resolution reference 与 coarse inverse 的平均投影失配
高达 `35.011%`，粗网格只保留 `73.911%` 的 gradient RMS；`rho` 单位、cell 语义、
`rho -> Delta n`、真实相机和 pixel displacement 都没有闭合。所有帧还来自同一条
trajectory，而且五个 later-time frames 已在 v0 开发中被打开，只能算 exploratory，
不能冒充 fresh confirmatory test。

还有一个之前容易说过头的边界：solver callable 的参数中没有 truth，post-hoc scorer
也和求解器分开；但 Direct initializer 仍在同一个 Python 进程中执行。因此当前只能标
`CONTROLLED_INPUT_SELF_ATTESTED` 和
`independent_noninterference_proven=false`。进入真正 fresh test 前，必须先把冻结
initializer 放进只读模型参数与 observation 的独立进程。

正式 runner、独立 validator 和相关测试分别得到
`PASS_REAL_CFD_MORPHOLOGY_PROXY_CONTRACT_ONLY`、
`PASS_INDEPENDENT_ARTIFACT_VALIDATION` 和 `39 passed`。完整数据合同、逐帧表、
版本冲突与复现命令见
[PoolFire 真实 CFD 形态代理与 Warm-Start 第一闭环](poolfire_cfd_morphology_proxy_evidence_2026-07-23.md)。

**突破监测：没有算法突破。新增的是首条真实公开 CFD 轨迹上的四段隔离闭环、固定 `K=2` 的明确数值 headroom，以及“少量 correction 有益、过度 correction 有害”的主线机制证据。下一步先隔离 initializer，再用新增 trajectory 做 fresh confirmatory，并把研究重点放在 calibration-aware correction budget，而不是立刻把 ridge 换成更大的网络。**

## 205. initializer 已经搬进独立进程，但小算子上没有 wall-time 加速

上一节最后一个软件阻塞是 Direct initializer 与 truth、inverse 和 evaluator 仍共享
Python 进程。这轮没有换模型、没有重新调后期五帧，只把 evaluation inference 改成
固定数据协议的 fresh-exec worker，并把完整开销写进原来的求解账本。

父进程只把冻结 dual-ridge 的四组数组、metadata 和一帧 observation 编码到 stdin。
worker 用 `python -I -B` 启动，request schema 不接受 truth、inverse、projection
cache 或 Python callable。macOS Seatbelt 每次都实际探测并拒绝声明 CFD bundle 的
读取、canary 读写和网络访问；除 stdin/stdout/stderr 外没有继承文件描述符。worker
源码 SHA 也绑定到冻结常量，executor 创建后再替换文件会 fail closed。

第一轮红队指出五个会夸大结果的问题：输入 noninterference 证明过头、请求与回包
序列化漏计时、RSS 在 response 生成前采样、worker hash 循环自证、重复 ZIP member
与无限 stdout 未 fail closed。修复后，计时从 request 编码前开始，到 stdout 有界
读取、NPZ 解码、dtype/shape/model/output SHA 和 receipt 全部核验后结束；child
退出时由父进程用 `wait4` 读取 max RSS。第二轮红队只剩 worker hash 的 TOCTOU，
改为运行时也传固定 hash 后关闭。对应负向测试会拒绝 worker 替换、伪造模型、
重复 member、超限 stdout、伪装 whole-pipeline RSS 和虚假 truth noninterference。

正式 v1 共运行 7 个 fresh worker：2 个 refinement-validation、5 个 evaluation。
固定 `K=2` 的数值与 v0 一致：

- Zero：`0.60835`，`2A + 2A^T`；
- normalized BP：`0.51445`，`3A + 3A^T`；
- ridge Direct：`0.41486`，`3A + 2A^T`。

冻结 target 为 `0.64945` 时，Direct 首次达标平均支付 3 次完整 `A/A^T`，Zero 为
4 次，BP 为 6 次。它是调用数 headroom，但 target 较宽松，不能代替新增 trajectory
的 matched-accuracy 主表。

成本上的负结果同样重要。每次 request 约 `2.32 MB`，fresh-exec 平均约 `75 ms`，
child max RSS 最坏约 `44.2 MiB`。当前 `16 x 16 x 32` CPU inverse 极便宜，Zero
两步平均不到 `1 ms`，所以 Direct 的端到端 wall time 明显更慢。只有真实 BOST
forward/JVP/VJP 足够昂贵时，少一次或更多物理调用才可能赚回推理成本；现在不能写
速度成功。

还有两条边界不能抹掉。父进程在 post-hoc scorer 打开 truth 前构造 request，但没有
独立外部证明 observation 本身完全不依赖 truth，所以
`evaluation_truth_noninterference_proven=false`。Seatbelt 只拒绝声明的 bundle 根，
不是整个文件系统的无数据副本证明，所以
`filesystem_wide_noninterference_proven=false`。child RSS 也不是训练、pair
generation、solver 与 worker 合并后的全流程峰值。

正式状态为
`PASS_REAL_CFD_MORPHOLOGY_PROXY_WITH_ISOLATED_INITIALIZER_CONTRACT_ONLY`，
独立 validator 为 `PASS_INDEPENDENT_ISOLATED_ARTIFACT_VALIDATION`。完整协议、成本表、
图和复现命令见
[PoolFire C 路线独立进程 Warm-Start 成本门](poolfire_c_isolated_initializer_evidence_2026-07-23.md)。

**突破监测：没有算法突破。新增的是主线结果第一次经过 data-only fresh-exec 推理、完整序列化计时、child RSS 与负向变异审计；同时得到一个必须公开的负结果：当前小代理算子上调用数有 headroom，但 wall time 没有加速。下一步不再扩建隔离基础设施，只进入新增 PoolFire trajectory 的预注册 fresh 比较。**

## 206. 不再随机切帧：官方 15 条 PoolFire 轨迹已经固定角色

师兄确认的 C 路线没有变化：让算子学习给三维反演一个更好的初值，并在最终精度
相同的条件下降低完整 `A/A^T` 调用和端到端成本。这轮没有去碰旧算法分支，也没有
急着训练 FNO；先把决定结果是否可信的数据边界做实。

官方 PoolFire 一共有 15 条完整轨迹。现在机器协议固定为 11 train、2 validation、
2 untouched test。`p=14kw_size=01` 只负责模型/正则选择，
`p=22kw_size=01` 只负责 correction budget 与停止规则，两条 test
`p=22kw_size=05`、`p=58kw_size=01` 在模型、阈值、种子、指标和报告模板全部冻结前
不解压。此前已经看过的 `p=14kw_size=03` 后期五帧永久保留为 development，不能
以后换个名字当 fresh test。

真实 `data.npz` 首次跑验证时抓到一个摘要级错误：官方变量标签不是简写 `rho`、
`T`，而是 `rho.npy`、`T.npy`。修正后，元数据 SHA、11/2/2 组成员、九个变量顺序、
`80 x 80 x 200` 网格和 101 个时间标签全部通过。这说明“先让机器核对原始对象”
比把人工笔记当真更可靠。

新 acquisition 工具会断点续传 6.4–6.7 GB 轨迹，先检查精确字节数与 SHA，再流式
提取 full-resolution `rho`；READY 完成后默认删除大原始缓存。test 必须显式
`--seal-test-only`，文件名写成 `*.sealed.npz`，并且工具无条件拒绝 test
`--extract`。这只是工具级 fail closed，不冒充操作系统级不可读保险箱。

第一条新增训练轨迹 `p=33kw_size=01` 已进入独立下载/提取队列。第一版直接使用
单个 `curl --retry` 进程，真实网络发生 HTTP/2 CANCEL 和 SSL 中断后，curl 内部
retry 会把已增长的 `.part` 从头改写，约 1 GB 进度因此没有保住。这个实现已停用，
不能把“命令还活着”误报成稳定续传。

修复版改为 Python 外层重试：每次重新读取当前 `.part` 长度，再启动一个新的
`curl --http1.1 --continue-at -`；前后大小必须单调不减，连续五次无字节进展或文件
缩小立即 fail closed。保留下来的 partial 会从现有长度继续。在 receipt 与 READY
出现前，它仍只能标为 acquisition in progress，不能算轨迹接入成功。

完整表、声明边界和复现命令见
[PoolFire 多轨迹协议与首条新增数据接入](poolfire_trajectory_protocol_evidence_2026-07-23.md)。
协议、acquisition 和 extractor 的定向测试当前为 `29 passed`。即使 curl
返回零退出码，partial 没有达到协议冻结字节数也不能提前视为下载完成。

**突破监测：没有算法突破。新增的是后续所有模型都必须遵守的完整 trajectory 级 11/2/2 协议、测试集锁门和可续传数据桥。下一道科学门是新增轨迹上的 Zero/BP/CGLS/PCGLS/dual-ridge classical control；只有未参与拟合的完整轨迹仍显示 headroom，才启动最小神经算子。**

## 207. 第一条新增完整轨迹真的接入了，不再停在“下载中”

`p=33kw_size=01` 现已完成，而不是继续卡在页面上的 acquisition in progress。
原始公开 archive 为 `6,522,109,719 bytes`；外层 HTTP/1.1 续传完成后，工具先核对
官方字节数和 SHA，再流式检查 ZIP/NPY 并提取 full-resolution `rho`。最终 bundle
是 `(101,80,80,200)` float32，101 个时间标签从 30 到 32；所有 `rho` 都有限且
严格为正。

完成后又做了一次独立复核：receipt 的协议/source 身份、manifest、READY 绑定和
`rho.npy`、coords、times、manifest 四个 checksum 全部一致。原始 6.52 GB 文件和
partial 已删除，只保留约 517 MB 的 `rho` bundle、receipt 与下载日志。正式状态是
`PASS_FIRST_ADDITIONAL_TRAIN_TRAJECTORY_READY`。

为什么页面看起来很久没动：一条 trajectory 不只是下载 6.5 GB，还要顺序读取原始
九通道大数组、验证完整流、抽出 `rho`、写 checksum，公开页面又没有在后台每秒同步
本机私有队列。此前页面停在“下载中”是状态更新滞后，不是算法一直原地训练。

现在串行进入第二条 train `p=45kw_size=05`，后面依次是
`p=58kw_size=03` 和两条 validation。只有至少三条新增 train 与两条 validation
完成后，才冻结统一 observation generator 并开始逐 trajectory 的
Zero/BP/CGLS/PCGLS/dual-ridge 表；提前训练 FNO 只会把数据偶然性学进去。

**突破监测：没有算法突破。真实增量是第一条额外完整训练轨迹通过身份、内容、数组和 READY 全链验证。下一步是完成其余 train/validation 接入，而不是重复单轨迹数值。**

## 208. 第二条新增训练轨迹完成，第三条开始

`p=45kw_size=05` 在下载到约 2.52 GB 时曾连续五次遇到 Hugging Face SSL
连接失败。下载器按约定停止并保留 partial；恢复后从同一字节继续，没有把已下载
内容重写。最终官方 `6,634,789,365 bytes` 对象通过字节数与 SHA，随后完成
ZIP/NPY 流式检查和 full-resolution `rho` 提取。

独立复核再次检查协议/source 身份、manifest、READY、四文件 checksum、坐标、
时间和数组内容。得到 `(101,80,80,200)` float32 `rho`，全部有限且严格为正；
原始 archive 与 partial 已删除，`test_truth_opened=false`。正式状态提升为
`PASS_SECOND_ADDITIONAL_TRAIN_TRAJECTORY_READY`。

第三条 train `p=58kw_size=03` 已按同一串行流程启动。三条 train 完成后仍不会
立即宣布“可以训练论文模型”：还需要两条职责分开的 validation，才能冻结模型选择、
correction budget 与停止规则，避免在训练轨迹上边看结果边改规则。

**突破监测：没有算法突破。真实进展是第二条跨工况训练轨迹完成全链数据接入，断点恢复也经受了一次真实 SSL 故障；下一门仍是第三条 train 与两条 validation。**

## 209. 三条训练 pilot 闭合，第一条 validation 启动

`p=58kw_size=03` 下载到 `6,428,472,735 / 6,611,053,939 bytes` 后连续五次
无法连接 Hugging Face。下载器按约定 fail closed，partial 没有缩小。恢复任务
从同一字节继续，只补齐剩余约 183 MB，没有重下前面的 6 GB。

完成后重新做了独立 14 项复核：协议 hash、官方 source 字节数/SHA、manifest
hash、checksums 文件 hash、四文件 checksum、`rho` shape/dtype、有限性、
严格正值、坐标 shape、101 个时间点、READY、原始缓存删除和
`test_truth_opened=false` 全部通过。`rho` 为 `(101,80,80,200)` float32，
范围约 `0.19206` 到 `1.17954`。正式状态为
`PASS_THIRD_ADDITIONAL_TRAIN_TRAJECTORY_READY`。

这意味着三条新增 train pilot 的数据门闭合，但还不能训练并挑选论文模型。
第一条 validation `p=14kw_size=01` 已启动，只允许决定模型与正则；第二条
`p=22kw_size=01` 只允许冻结 correction budget 和停止规则。两条职责不能
混用，也不能把 untouched test 提前拿来救结果。

**突破监测：没有算法突破。真实增量是三条 train pilot 全部通过独立数据链复核；下一有效门是两条 validation READY 后冻结跨轨迹实验合同。**

## 210. 第一条 validation READY，第二条启动

模型选择 validation `p=14kw_size=01` 已完成下载、官方 source 校验、
full-resolution `rho` 提取和原始缓存清理。随后重新计算四文件 checksum，并做
独立 14 项复核：协议/source/manifest/READY 绑定、shape、dtype、有限性、
严格正值、坐标、时间轴、缓存删除和 test 未打开全部通过。

`rho` 为 `(101,80,80,200)` float32，范围约 `0.18310` 到 `1.18569`。
正式状态为 `PASS_FIRST_VALIDATION_TRAJECTORY_READY`。这条轨迹只允许
模型与正则选择，不能用来决定 correction budget 或停止规则。

第二条 validation `p=22kw_size=01` 已启动，并且 partial 正常增长。它的唯一
职责是冻结 correction budget 与停止规则。两条 validation 都 READY 后，下一步
不是立刻看 test，而是先把抽帧、reference、normalization、proxy observation、
经典基线、matched-accuracy tolerance、指标和成本账冻结成跨轨迹实验合同。

**突破监测：没有算法突破。真实增量是第一条职责受限的 validation 完成独立复核；下一门是第二条 validation 与跨轨迹实验合同。**

## 211. 第二条 validation READY，跨轨迹规则在看结果前锁死

停止规则 validation `p=22kw_size=01` 已完成公开 source 下载、SHA/字节数、
ZIP/NPY 流式检查、full-resolution `rho` 提取和原始缓存清理。四文件 checksum
和独立 14 项复核全部通过；`rho` 为 `(101,80,80,200)` float32，范围约
`0.18372` 到 `1.19238`，所有值有限且严格为正，test 仍未打开。

现在五条开放数据是 `3 / 3 train pilot + 2 / 2 validation READY`。这仍只是
数据完整性，不是重建或算法结果。为避免进入“先看曲线再改规则”的循环，本轮没有
直接开 FNO，也没有生成跨轨迹结果，而是先冻结
`poolfire_c_cross_trajectory_experiment_v1.json`。

合同把三条 fit pilot、p14 模型选择、p22 固定 correction budget 和两条 untouched
test 分开。它使用全部 101 帧但明确 frame 不是独立样本；固定 ROI、32×32×64
独立 reference、16×16×32 inverse、gauge 和 exact block mean；normalization 与
最终 ridge 权重只能使用 fit trajectories，validation refit 被禁止。经典门先比较
Zero、normalized BP、真正固定几何对角 PCGLS 和 dual ridge。

同精度不再只看一个 field 数字，而要同时通过 field、gradient 和 observation
residual；再报告逐轨迹 p50/p90/worst、harm、完整 A/A^T、端到端 wall 和 fresh
process whole-arm peak RSS。两条 test 必须由一个联合冻结包同时授权，不能看完
第一条后改模型再开第二条。

合同 validator 和 13 个 fail-closed 测试均通过，状态为
`PASS_FROZEN_POOLFIRE_C_CROSS_TRAJECTORY_EXPERIMENT`。独立审计还指出旧
`fit_calibrated_dual_ridge()` 会把 calibration 拼回最终拟合，因此正式 runner
必须另写 train-only final fit，不能直接复用旧单轨迹流程。

**突破监测：没有算法突破。真实增量是等待阶段结束、五条开放轨迹完整、跨轨迹评分与防泄漏规则事前冻结。下一门是统一 pair generator 和四个 classical arms；只有 ridge 在两条 validation 的职责链上仍有稳定 headroom，才准训练最小 3D U-Net/FNO。**

## 212. 正式 runner 的两个隐蔽泄漏点先被拆掉

代码审计发现，旧 `VerifiedPoolFireRhoBundle` 把 manifest split 写死成
`train`，所以它会拒绝合法 validation；旧 `fit_calibrated_dual_ridge()`
选完 lambda 后又用 `train + calibration` 重拟合，若直接套到 p14，就会让
validation 进入最终权重。

loader 现在新增显式 `expected_trajectory` 与 `expected_split` 绑定，只接受
train/val；即使调用者主动请求 `expected_split=test` 也会 fail closed。旧 p14s03
流程仍默认 train，原有结果合同不变。

跨轨迹 ridge 另写为 `select_train_only_standardized_dual_ridge()`：observation
featurewise mean/std、field mean/global RMS 和最终 dual weights 全部只来自 fit
samples；p14 只从七个冻结 multiplier 中选择 lambda。选择顺序是 p90
(`higher` quantile)、worst、median，完全平手时取更强正则。最终
`fit_sample_count` 不会随 validation 样本数增加。

validation loader、投毒/身份错配/test 拒绝、train-only normalization、确定性
hash、平手规则和合同 validator 合计 `28 passed`。这只证明正式 runner 的两个
基础接口不再沿用已知泄漏路径，尚未生成任何跨轨迹 observation、重建或成本结果。

**突破监测：没有算法突破。下一门仍是流式 pair generator、geometry-only PCGLS 与四臂 classical 表；不能把 28 个代码测试写成 ridge 已经更准或更快。**

## 213. 五条轨迹 505 帧统一出题完成，发现工况依赖的模型失配

正式 pair generator 只接受合同注册的三条 fit 和两条 validation，CLI 没有
`--test` 或 `--allow-test`。每条输出 101 个三视角 observation、gauge-centered
`16×16×32` truth、时间标签和逐帧失配审计；full-resolution field 不复制进 pair
库。写入采用临时目录，全部 payload checksum 完成后才原子生成 READY。

五条轨迹共 505 帧已全部生成，并由另一个 validator 逐条检查 contract/role/split、
shape/dtype、finite、非零范数、逐帧 zero-mean gauge、frame order、payload SHA 和
READY。五条共享一个 geometry binding，test pair 数仍为 0。

输入审计出现一个真实但不是算法的信号：独立 reference 与 coarse inverse truth
projection 的相对失配，逐轨迹 p50 为：

- p33s01：`51.0%`，p90 `53.9%`，worst `55.7%`；
- p45s05：`39.6%`，p90 `44.5%`，worst `45.7%`；
- p58s03：`34.0%`，p90 `37.0%`，worst `39.5%`；
- p14s01 validation：`47.5%`，p90 `49.5%`，worst `52.5%`；
- p22s01 validation：`47.5%`，p90 `52.3%`，worst `53.9%`。

几何完全相同，差别来自流场形态与有限网格近似对梯度的不同响应。这给论文主线一个
更具体的困难：warm start 不仅要在一个误差水平上快，还要在 `34%–56%` 的
condition-dependent model mismatch 下不伤害尾部。因此 p22 的 fixed-K/harm
门不是形式主义，而是决定方法能否迁移的核心。

公开网页只放聚合图和无路径 summary；私有 pair、派生 hashes 和本机位置均未上传。

**突破监测：没有算法突破。新增的是首个 5 trajectory / 505 frame / one-geometry 的统一输入闭环，以及工况依赖模型失配证据。下一门是 geometry-only PCGLS、train-only ridge 选择和四臂 matched-accuracy 表。**

## 214. 四臂经典表完成：先发现 K=24 这个“参考答案”本身错了

五条开放轨迹、505 帧都跑完了 Zero-CGLS、normalized BP-CGLS、真正的
geometry-diagonal PCGLS 和 train-only dual-ridge warm start。PCGLS 的对角项
不是随机估出来的，而是从 LOS 权重、差分矩阵和 detector selection 精确得到
`diag(A^T A)`；K=0 初值和每个 checkpoint 的完整 `A/A^T` 也进入同一本账。

结果先否掉了原计划中的一个关键假设：在 `p=22kw_size=01` 上，Zero-CGLS
从 K=4 继续跑到 K=24 时，observation residual p90 从 `0.34810` 降到
`0.30430`，但 field p90 从 `0.68190` 恶化到 `1.02633`，gradient p90
从 `0.97990` 恶化到 `2.28542`。也就是说，求解器更会“解释观测”了，却离真实
三维场更远。这就是含模型失配逆问题里的半收敛。

按原先预注册的 K24 参照，所有候选都只能选到 K=8 左右，没有 warm-start
调用优势。结果打开后，用更合理但只能算 post-hoc 的 Zero K4 作诊断参照，
dual-ridge K2 在 p22 用 5 次完整调用达到 97.03% joint pass，而 Zero K4
要 8 次，调用数少 37.5%。可它在 p14 只通过 56.44% 帧，且五条轨迹上
wall time 全部更慢；约 25 MB 的 ridge 推理开销吃掉了少跑两步的收益。

**讲人话：**原来的终点像是“把同一道题反复改到卷面更工整，却把答案改错了”。
我们发现了这个坑，也看见少跑两步可能省调用，但现在还不能说算法成功，因为换到
p14 就失守，而且真实时间没有变快。

**突破监测：没有算法突破。新增的是完整跨轨迹经典表、精确几何 PCGLS 和一个必须写进论文的方法学发现：K24 在模型失配下严重半收敛，不能继续当高精度参照。**

## 215. v2 线性摊销器失败：它连 p14 的 K4 teacher 都学不住

既然 Zero K4 比 K24 更像合理的物理终点，我设计了 v2：只从 observation
预测 Zero-CGLS K4 field，相当于把四步物理求解摊销进一个低秩线性算子。它用
三条 fit trajectory 的 303 帧拟合，只在 p14 选择正则和 rank；由于设计来自已经
打开的 v1 结果，它明确标为 adaptive development。

结果是 `FAIL_NO_RANK_MATCHES_P14_CGLS_K4_AT_ZERO_DEPTH`。即使 rank 256，
field、gradient、observation 三项 pass fraction 仍全是 0，harm 为 92.08%。
失败后运行器没有打开 p22，更没有碰 test。

**讲人话：**不是把矩阵做得更大就能把四步 CGLS 直接背下来。三条训练轨迹和 p14
长得差得太远，线性模型在训练条件外几乎不会做这道题。

**突破监测：没有算法突破。v2 是有用的负结果：当前数据覆盖下，直接用低秩线性算子摊销 K4 不可行。**

## 216. v3 PCA 先替神经网络踩刹车：真正瓶颈是跨轨迹覆盖

我没有立刻把线性层换成 FNO，而是先做了一个更便宜的表示 headroom 审计。若三条
fit trajectory 连一个 256 维 PCA 子空间都不能覆盖 p14，那么更大的网络很可能只是
把训练轨迹记得更牢。

rank 256 已解释 fit observation 的 99.8668% 能量、fit K4 field 的 99.9711%
能量；但在 p14 上，observation PCA p90 重建误差仍为 `0.51684`，K4 field PCA
p90 仍为 `0.25314`。没有一个 rank 达到预设的 output p90 `<=0.05`、worst
`<=0.10` 门，状态是 `FAIL_LATENT_OUTPUT_PCA_HEADROOM`。这轮只读 fit 和 p14，
p22/test 都没打开。

**讲人话：**训练集内部看起来几乎“什么都解释了”，一换完整工况就解释不了。这不是
模型层数的问题，是训练样本覆盖的问题。现在直接训练 FNO，容易得到漂亮训练曲线和
难看的真实迁移。

**突破监测：没有算法突破。新增的是在烧神经网络训练时间前就定位出的 representation coverage failure。**

## 217. v4 只扩两条邻近工况，再决定要不要继续下载

下一步没有无边界扩大数据集。v4 先冻结一个分阶段门：只接
`p=14kw_size=05` 和 `p=22kw_size=03`，重新生成完全相同的 independent proxy
pair、K4 teacher 和 PCA audit。只有 rank-256 p90 表征误差相对下降至少 20%，
才继续接剩余五条 clean fit trajectory；没有改善就停止下载，重审输入表示和目标。

第一条 `p=14kw_size=05` 已在项目外私有队列中单实例断点下载。它完成后仍要过
官方字节数/SHA、ZIP/NPY、full-resolution rho、finite/positive、manifest、
checksums、READY 和缓存清理，不能把 partial 或下载进程写成数据接入成功。

正式 test 继续封存，p14/p22 的 validation 职责不变。当前 Mac 已有可用的 PyTorch
MPS 环境，但覆盖门没有改善前不会训练 3D U-Net/FNO/UNO。

**讲人话：**现在不是缺一张更强显卡，而是先确认“增加两种相邻火焰工况，能不能让模型真正见过更完整的变化”。这个问题没答对，换服务器只会更快地过拟合。

**突破监测：没有算法突破。当前唯一有效门是两条新增 clean-fit 轨迹能否显著修复 p14/p22 的表征覆盖；通过后才进入最小神经算子。**

## 218. 独立审计后重算：结论没变，但证据链更干净了

两名独立审计代理分别检查了方法学和代码。它们没有发现 P0 级错误，但抓到五个
会污染成本或证据边界的问题：旧 v2 在 rank gate 失败前提前载入 p22；已有 arm
缓存只按文件存在判断，未严格核对协议、pair、模型与全部 checkpoint；外部返回
全零初值时漏记一次必要的前向投影；PCGLS 几何对角的 setup 没有进入单次使用与
摊销 wall 账；PCA 私有包混放了 basis 和 fit/validation 样本。

这些问题现在全部按 fail-closed 修正：

- v2 只先读取三条 fit 与 p14，rank gate 通过后才有权读取 p22；
- 缓存必须同时绑定 experiment、geometry、pair READY、trajectory role、ridge
  model 和 8 个完整 checkpoint，否则拒绝聚合；
- 只有求解器内部可信的 `initializer=None` 可以省略初始前向，外部初值即使数值
  恰好全零也要支付一次 `A`；
- PCGLS 同时报告 setup、单次全轨迹 wall 和 101 帧摊销 setup，不再只展示迭代段；
- v3/v4 的私有 NPZ 只保留 fit-only basis/statistics，不保留任何 fit 或 validation
  样本，并且强制写到仓库外。

审计后的 targeted suite 为 `73 passed`。严格缓存验证接受了五条既有 arm 结果；
v1 重新聚合后仍是开放代理 development。v2 重跑仍为
`FAIL_NO_RANK_MATCHES_P14_CGLS_K4_AT_ZERO_DEPTH`，并明确
`stopping_validation_opened_by_v2=false`；v3 重跑仍为
`FAIL_LATENT_OUTPUT_PCA_HEADROOM`。rank-256 的 p14 K4-target PCA p90 仍是
`0.253138`，所以 v4 的 20% 改善门仍冻结为 `<=0.202510`。

**讲人话：**这轮没有把失败结果“修成成功”。它做的是把计账漏洞和验证集读取顺序
收紧后再算一遍，确认当前真正的困难仍然是训练工况覆盖不足。

**突破监测：没有算法突破。真实增量是审计后的 v1→v2→v3 证据链可重复，p22/test
没有被拿来救模型；下一门仍是两条 clean-fit 轨迹的 p14-only coverage 复查。**

## 219. v4 再收紧：20% 门只决定要不要继续拿数据

第二轮独立审计发现，原 v4 代码虽然不读取 p22 的数组，却会在统一几何检查时顺手
读取 p22 pair manifest；新增 pair 也没有强制要求“目录名、请求轨迹、官方 source、
父协议、几何、manifest 和 READY”全部一致。这样不会立刻改变 PCA 数值，却会让
“p22 完全没有参与此次决策”和“两个新增工况确实是两个不同 source”说得不够硬。

现在这两个问题已按 fail-closed 修复：

- v4 的允许名单只有 3 条既有 fit、2 条 first-batch clean fit 和 p14 模型选择
  validation；p22 停止验证连 manifest 都不读取；
- 新增 pair 必须绑定官方 PoolFire trajectory protocol、source SHA、统一 geometry、
  请求轨迹、目录名、manifest、checksums 与 READY，复制或换名会被拒绝；
- v4 只解释 observation 并由 observation 生成 K4 teacher，不解释任何 pair
  `gauge_truth.npy`；truth 文件仍受 checksum 保护；
- 私有 PCA basis、五条 fit pair 的身份绑定、协议绑定和公开 summary 先在临时目录
  完整写好，再原子生成 READY；已有结果只有全部 hash 一致才可复用；
- 私有续跑队列使用唯一锁和独立 run log，不会重复启动同一下载或并发覆盖结果。

31 个定向测试和完整 PoolFire 回归 `200 passed`。已有 3 条 fit 与 p14 pair 还在
真实私有文件上通过了 observation-only 身份复核。第一条新增数据 p14s05 仍在单实例
断点获取中；partial 不是 READY，也不是算法结果。

更重要的方法学修正是：即使 v4 的 rank-256 p90 从 `0.253138` 降到
`<=0.202510`，它也只说明固定线性子空间覆盖有至少 20% 改善，只授权继续接剩余
clean-fit trajectories。全部 clean-fit 完成后，必须按完整 trajectory 做
leave-one-trajectory-out，比较 rank 256 与可用最大 rank；绝对门仍保留
p90 `<=0.05`、worst `<=0.10`。

只有这道完整覆盖门通过，才训练一个预注册的小型 BP-conditioned 3D U-Net
sentinel。至少 80% 的留出轨迹要达到 joint pass `>=90%`、harm `<=5%`、
固定 `K<=2`，总调用严格少于 Zero K4 的 8 次，轨迹等权 wall-time 中位数不变慢，
并测 fresh-process 全流程 peak RSS。sentinel 通过后，才允许公平比较 FNO、UNO
和 DeepONet。

**讲人话：**20% 门只是在问“再拿数据有没有用”，不是在问“神经网络赢没赢”。
先证明训练工况覆盖得住，再用一个很小的模型试水；小模型都过不了，就不烧大模型。

**突破监测：没有算法突破。真实增量是 v4 的角色隔离、轨迹身份和原子结果证据链
闭合，并把神经训练授权从一次 PCA 相对改善后移到完整轨迹留一与 sentinel 门。**

## 220. v4 缓存再加一道锁：代码变了，旧结果必须失效

独立代码复审又发现一个容易被忽略的问题：v4 旧缓存虽然绑定了数据、协议、几何和
pair，却没有绑定“到底是哪一版代码算出来的”。如果以后修改 K4 teacher、CGLS/PCGLS、
几何构造或 validator，旧目录仍可能被当成当前结果复用。

现在私有 manifest 会逐文件绑定 v4 runner、K4 teacher、classical geometry、
pair validator、CGLS/PCGLS、cross-trajectory geometry、warm model、CFD proxy
和两份 straight-ray operator 的 SHA，同时绑定 Python 与 NumPy 版本；READY
再绑定这份实现指纹。任一数值路径文件或运行时版本变化，旧结果都会 fail closed，
必须重新计算，不能悄悄沿用。

现场还发现一个早于唯一锁修改时间启动的旧续跑器。它已被单独终止，正在增长的
p14s05 下载器没有被打断；后续只允许带原子锁、唯一 run log 和状态文件的新队列
接管。实现指纹修复后的定向测试为 `31 passed`，完整 PoolFire 回归为
`200 passed`。

**讲人话：**同一个数据，用不同版本的算法算，不能假装是同一次实验。现在每个 v4
结果都带着一张“代码身份证”；代码哪怕改一处，旧结果就不能蒙混过关。

**突破监测：没有算法突破。这里修复的是可重复性和缓存可信度；科学门仍是
p14s05/p22s03 接入后运行 p14-only v4 coverage gate。**

## 221. v4 不再自己给自己发合格证

实现指纹闭合后还剩一个方法学问题：v4 runner 会写私有结果，也会在同一模块里检查
既有缓存。即使单元测试通过，这仍不算真正独立的结果复核。

现在新增了一个不导入 v4 runner 的独立 validator。它会重新完成以下检查：

- 只枚举 3 条既有 fit、2 条 first-batch clean fit 和 p14 模型选择轨迹；
  p22 stopping validation 不在允许名单；
- 对六条 pair 重新运行 observation-only validator，核对官方 source、父协议、
  geometry、manifest、checksums 与 READY，不解释 truth 数组；
- 回到六条官方 full-resolution `rho` bundle，逐帧重放 606 次独立 forward，要求每个
  observation 与 pair 内保存值逐元素完全相同；
- 用底层 PCGLS 原语重算 606 个 K4 teacher，再只用五条 fit 轨迹重拟合 PCA，逐 rank
  重算 p50/p90/worst、能量、20% 改善和 PASS/FAIL，不能只信 runner 写出的数值；
- 重新核对私有 manifest、READY、协议绑定和十一个数值路径文件的实现指纹；
- 打开私有 NPZ 时只允许七个 basis/statistics 数组，并检查 shape、float64、
  finite、正 observation scale 和非负 singular values；若混入 fit/validation
  样本立即失败；
- 再次确认 `neural_training_authorized=false`、`test_truth_opened=false`、
  `algorithm_breakthrough=false`。

独立审计曾抓到四个 P1：结果行仍可能自证、pair 可能作内部自洽的跨轨迹拼接、
implementation binding 漏了直接 forward 依赖，以及旧队列的 pair 预检曾为
shape/finite 检查解释过 truth 数组。前三项已由 full source replay、独立 K4/PCA
重算和依赖闭包修复；最后一项不能抹去历史，因此只能诚实记录：旧预检碰过 truth，
但 v4 的 K4/PCA 判决没有使用 truth。新队列已显式传入
`--skip-truth-array-inspection`，且只有独立验证报告与 public status 一致才写
`VERIFIED_COMPLETE`。完整 PoolFire 回归为 `208 passed`。

**讲人话：**以前是“做题的人顺便批自己的卷子”；现在换了一套没有调用原做题程序的
批卷逻辑，重新算关键答案。两边都一致，结果才进入下一步。

**突破监测：没有算法突破。新增的是更强的独立结果复核。**

## 222. 首批覆盖扩充真实改善 13.86%，但冻结的 20% 门没有通过

`p=14kw_size=05` 和 `p=22kw_size=03` 已完成 clean-fit 接入。修正后的 runner
生成新结果后，独立 validator 从官方 `rho` 重放全部 606 帧 forward、重算 606 个
K4 teacher 和 fit-only PCA，得到同一个判决：

| 项目 | 冻结基线 / 门槛 | 实际结果 |
|---|---:|---:|
| rank-256 p14 K4-target PCA p90 | 基线 `0.253138` | `0.218051` |
| 相对下降 | 至少 `20%` | `13.8608%` |
| 最大允许 p90 | `0.202510` | 未达到 |
| 是否继续拿剩余 clean-fit | 只有 PASS 才允许 | `false` |
| 是否授权神经训练 | 本门本来就不授权 | `false` |

这不是“完全没进展”：两条新 fit 轨迹确实让 p90 降了约 `0.0351`。但事前冻结的是
20%，实际只到 13.86%，所以必须判 FAIL，不能临时降低标准。rank 256 时 fit K4
target 已解释 `99.6543%` 能量，p14 output p90 仍为 `0.218051`；observation p90
更高达 `0.469397`。这更像是跨工况表示/对齐问题，而不是简单增加同类样本后就会
自动消失的欠采样。

**讲人话：**新加的两本题库让答案更接近了，但仍没有接近到事前规定的合格线。
继续盲目下载更多同类题目不划算。下一步要检查“答案表示方法”本身：K4 是固定
线性算子作用于 observation，当前带均值 PCA 可能把幅值和形状混在一起，也可能受
rank 256 上限约束。先比较更高 rank、过原点子空间以及只使用 observation 可见幅值
的齐次归一化，仍只做开发诊断，不碰 p22 stopping validation 与两条 test。

**突破监测：没有算法突破。可信的新事实是首批覆盖扩充只改善 13.86%，正式
v4 状态为 `FAIL_FIRST_BATCH_MATERIAL_P14_COVERAGE_IMPROVEMENT`。**

## 223. 第一版 v5 被协议审计主动降级，没有把漂亮数字直接发布

在 v4 失败后，第一版 v5 比较了 rank 256–504、带均值/过原点以及 observation-RMS
齐次表示。runner 一度得到 rank-504 p90 `0.148823`。但独立协议审计随后发现六个
P1，最关键的是：

- 固定四步、零初值 CGLS 一般不是线性映射；只有在 breakdown 分支不变且有限精度
  可忽略时，才对整体标量缩放呈一阶齐次；
- validation target 在子空间里的投影系数来自完整 K4 target，是 oracle containment
  检查，不是 observation→coefficient 的部署模型；
- rank 504 需要 fit-only 奇异值、稳定数值秩和边界谱隙门，不能只因理论最大 rank
  是 504 就直接接受；
- RMS 必须绑定 raw `observations.npy` 的全部 2072 分量、固定顺序、等权、无 mask、
  无标准化，floor 命中必须为 0；
- 必须先排除数值不稳定行，再同时检查 p90/worst；“齐次表示获胜”还要预先规定
  相对 best raw 的最小改善和 no-harm 条件；
- 失败最多排除这四种固定全局子空间，不能顺手排除 nonlinear decoder、
  conditional basis、mixture-of-subspaces 或 full-field CNN。

因此第一版 v5 私有结果和图被移入 provisional 归档，公开候选目录删除。它不能作为
机制结论、模型结论或论文结果。

**讲人话：**不是数字看起来变小就收下。先问“这个小数字是不是依赖一个部署时拿不到
的 oracle 投影”“最后几维是不是数值噪声”。答案没闭合，就把结果降级重做。

**突破监测：没有算法突破。这里的进展是审计在发布前成功阻止了过度解释。**

## 224. v5.1 把齐次性、稳定 rank 和 oracle 边界补齐

v5.1 在任何新结果前单独提交冻结，新增：

1. 对全部 505 个 fit frame 分别计算 `K4(0.5y)`、`K4(2y)`，与
   `0.5 K4(y)`、`2 K4(y)` 比较，并要求四步 breakdown flags 完全一致；
2. rank 必须同时通过 `sigma_r/sigma_1 >= 1e-8`、fit-only stable rank 和边界谱隙
   `>=1.001`；
3. 明写 `oracle_target_projection=true`、`deployable model=false`；
4. centered 候选把 mean field 的存储量计入 decoder bytes；
5. homogeneous 只有相对 best raw 的 p90 至少改善 2%，且 worst 不变坏，才算
   material win。

runner 完成后，独立 validator 不导入 v5.1 runner 或 representation helper，重新
计算 606 个 K4 teacher、1010 个缩放 K4、20 个 projector、稳定 rank、passer-first
选择和 homogeneous 2% + no-harm 门，得到完全一致的结果：

| 检查 | runner 结果 |
|---|---:|
| 505 帧 × 2 个 scale 的最大齐次误差 | `0` |
| breakdown path mismatch | `0` |
| stable eligible rows | `20/20` |
| v4 raw-centered rank-256 p90 | `0.218051` |
| raw-origin rank-256 p90 | `0.196516` |
| best raw-origin rank-504 p90 | `0.148973` |
| best RMS-origin rank-504 p90 | `0.148823` |
| RMS 相对 best raw 改善 | `0.1009%`，低于冻结的 `2%` |
| 最佳 p90 / worst | `0.148823 / 0.165473` |
| 绝对门 | `p90<=0.05` 且 `worst<=0.10`，FAIL |

这说明 provisional 信号主要来自 rank 256→504 和去掉仿射均值：仅在 rank 256
改成过原点就降低约 `9.88%`，扩到 rank 504 后比 v4 降约 `31.68%`；RMS 幅值拆分
几乎没有额外贡献。即使如此，固定全局子空间离绝对门仍有大距离。

**讲人话：**我们排除了“只要除以一个强度就解决跨工况”的简单故事。现有困难更像
空间形态、位置或条件依赖的基底变化。下一步若独立复核通过，应优先检查对齐、
conditional basis / mixture-of-subspaces 或不受固定 PCA 输出瓶颈限制的 full-field
decoder，而不是把 RMS 包装成创新。

独立状态为
`PASS_INDEPENDENT_POOLFIRE_C_HOMOGENEOUS_REPRESENTATION_V5_1_VALIDATION`，
科学状态仍是 `FAIL_DEVELOPMENT_ABSOLUTE_OUTPUT_HEADROOM`。

**突破监测：仍没有算法突破。这是一条可信的机制性负结果；没有部署模型、调用减少、
wall-time、内存、泛化或真实 BOST 证据。**

## 225. 输运确实存在，但原始 BP 不能稳定告诉我们该往哪里平移

v5.1 之后最自然的猜想是：固定全局 PCA 之所以 rank 很高，可能是在为同一团结构的
空间移动反复付费。因此这一轮没有训练网络，而是先冻结一个两阶段机制门。

第一阶段只看部署时拿得到的 observation。六条已开放轨迹各 101 帧，每帧只做一次
`BP=A^T y`，总账是 `0 A + 606 A^T`。BP 的平方被当作 energy，用来计算三维质心、
协方差、fit-only canonical center 和 half-away-from-zero 整数 shift。这个阶段的
API 没有 target、teacher 或 truth 参数，p14 也不能反过来修改 canonical center。
独立验证器没有调用正式 runner，重新生成 606 个 BP 和全部统计，逐数组最大绝对差
都是 0。

第二阶段才在私有评估中生成固定四步零初值 CGLS teacher，共
`2424 A + 2424 A^T`。teacher 的全局 q10-q90 最大质心跨度是 `2.596` 个体素，
64.52% 帧的 oracle 整数 shift 非零，所以“结构位置会变”这个物理前提成立。
问题出在 raw BP proxy：

| 轨迹 | 质心 L∞ p90（体素） | 整数 shift 完全一致 | 误差不超过 1 体素 |
|---|---:|---:|---:|
| p14 模型验证 | `0.304` | `70.30%` | `100%` |
| p14-s05 | `0.962` | `31.68%` | `99.01%` |
| p22-s03 | `0.359` | `85.15%` | `100%` |
| p33-s01 | `0.516` | `37.62%` | `100%` |
| p45-s05 | `1.847` | `14.85%` | `63.37%` |
| p58-s03 | `0.628` | `57.43%` | `100%` |

冻结门要求每条轨迹都满足 p50≤0.5、p90≤1、worst≤2、exact shift≥75%、
within-one≥95%，不能用 pooled mean 抹掉坏工况。`p=45kw_size=05` 的
p50/p90/worst 是 `1.181/1.847/2.108`，所以正式状态是
`FAIL_T0_BP_INTEGER_TRANSLATION_PROXY`。其他多条轨迹也没有通过 exact-shift 门。

这个失败不是随机乱跳。p45 三轴质心相关仍约为 `0.793/0.841/0.848`，说明 raw BP
大致跟随 teacher 运动，却有强工况相关的系统偏差。更像的物理解释是几何灵敏度：
`A^T y` 在射线覆盖更密或 `diag(A^T A)` 更大的位置天然更亮，能量质心因而不一定
等于待重建场的位置。

协议在 T0 失败后立即停止，SVD、aligned basis 和 T1 都没有运行。没有 target-oracle
shift 救场，没有训练 3D U-Net/FNO/UNO/DeepONet，也没有打开 p22 stopping
validation 或两条 test。

独立 Stage-2 validator 不导入正式 runner、transport stage、transport audit 或
alignment helper。它重新读取冻结 observation、重建几何并生成 606 个 K4 teacher，
六类 teacher 数组的最大绝对差全部为 0；独立调用账同样是
`2424 A + 2424 A^T`。验证前后 Stage-2 文件集合与字节身份不变，T1 主数组和
SVD/basis 输出计数都是 0。正式独立状态为
`PASS_INDEPENDENT_VALIDATION_OF_FAIL_T0_POOLFIRE_C_BP_TRANSPORT_STAGE2_V6`。

本轮定向 v6、聚焦页与图表共 `113 passed`；完整 PoolFire 相关回归
`293 passed`；Pages builder `68 passed`；当前工作树公开链接审计
`47368` 项、`missing=0`。这些数字只说明实现和公开证据链闭合，不是算法性能。

下一条候选不是自由拟合的校正网络，而是固定几何 Jacobi 均衡：

```text
BP_raw = A^T y
BP_equalized = D^{-1} BP_raw
D ≈ diag(A^T A)
```

它仍只花一次完整 `A^T`，在线多出来的是逐体素乘法。下一协议必须在结果前冻结
floor、归一化、全部逐轨迹门和独立复算；只有 equalized BP 在所有轨迹上可靠定位，
才重新授权整数对齐 T1。若它也失败，再考虑 fit-only、observation-visible 的轻量
校准映射，仍不直接训练大网络。

**讲人话：**火焰结构真的在移动，但普通反投影像一张亮度不均的地图：它能看出大致
往哪边走，却会被相机几何“照亮”的区域拉偏。现在先把地图亮度校平，再判断能不能
用它导航；不能导航就停止这条 shifted-POD 路线。

**突破监测：没有算法突破。新增的是一条经过 fail-closed 机制门的可信负结果，它
排除了 raw-BP 单一全局整数平移，不排除几何均衡 BP、条件基底或全场解码器。**

## 226. 固定几何均衡改善了坏工况，但没有关闭跨轨迹定位缺口

旧 v7 的唯一一次 sealed 运行先暴露了一个实施错误：PoolFire 三个原始坐标轴按降序
保存，而手工 coordinate-only 几何重建没有执行 v6 数据桥已有的反转与端点均匀化。
straight-ray operator 在读取 606 帧 observation 前正确拒绝了降序 x 轴。没有结果
目录，也没有科学判决，所以不能把这次失败写成均衡 BP 的正面或负面证据。

我随后单独提交了 v7.1 修复附录。它只允许恢复同一坐标规范化，`BP_eq=D^{-1}A^T y`
公式、`diag(A^T A)`、`1e-6` relative floor、六条轨迹、T0 阈值、raw 对照和失败动作
全部冻结不变。正式 runner 与独立 validator 分别实现规范化，三个轴的最大修正分别是
`5.43e-8/5.43e-8/1.80e-7`，均低于既有 `3e-7` 容差，并复现六条 observation 绑定的
同一几何身份。

v7.1 的 sealed 单次运行完成 606 帧并通过独立复算。所有候选数组与独立结果的最大
绝对差为 0，raw BP 最大差也为 0。科学状态却仍是
`FAIL_T0_GEOMETRY_EQUALIZED_BP_PROXY`：

| 轨迹 | raw p90 | equalized p90 | raw exact | equalized exact |
|---|---:|---:|---:|---:|
| p33-s01 | 0.516 | 0.501 | 37.62% | 36.63% |
| p45-s05 | 1.847 | 1.286 | 14.85% | 13.86% |
| p58-s03 | 0.628 | 0.625 | 57.43% | 61.39% |
| p14-s05 | 0.962 | 0.748 | 31.68% | 53.47% |
| p22-s03 | 0.359 | 0.499 | 85.15% | 86.14% |
| p14-s01 validation | 0.304 | 0.317 | 70.30% | 69.31% |

几何均衡确实把 p45 的 p90 降了约 30%，也明显提高 p14-s05 的 exact，这说明固定几何
灵敏度是偏差机制之一。但五条轨迹 exact 仍低于 75%；p45 还失败 p50、p90 与
within-one；p22 的 p90 从 0.359 恶化到 0.499，触发材料性 harm。固定对角 Jacobi
只能纠正跨样本不变的逐体素尺度，无法解释剩余的视角能量不平衡、形态变化和
observation-dependent 偏差。

因此 Jacobi 定位路线按协议停止。下一候选只能是 fit-only、deployment-visible 的
低容量校准映射：输入 raw/equalized BP 质心、逐视角能量、view balance 和低阶谱矩，
先预测校准质心，并在未参与拟合的 p14-s01 上过同一 T0 与 harm 门。它仍不输出三维
场，也不授权 FNO/UNO/DeepONet。若这一步失败，就停止 shifted-POD 定位支线，回到
observation 到 full-field warm initializer。

**突破监测：没有算法突破。** 新增的是一条可信机制负结果：几何灵敏度是问题的一
部分，但固定 `D^{-1}` 不足以成为通用定位器。没有 A/A^T 减少、wall-time 加速、内存
下降、真实 BOST、泛化或论文成功证据。

## 227. 低容量质心校准只过 3/6：停止 shifted-POD，直接学习完整三维初值

v7.1 之后，我们给“先定位、再平移”的思路最后一次低容量机会。v8 没有上大网络，
而是只用部署时已经拿得到的 8 个数字：

1. raw `BP=A^T y` 的三个归一化质心；
2. equalized BP 相对 raw BP 的三个质心改变量；
3. 三组视角能量中 x/z 与 y/z 的两个 log-ratio。

模型按 x、y、z 三个轴分开，每轴只有 5 个系数，总计 15 个 float64 参数。五条 fit
轨迹采用外层 leave-one-trajectory-out；每个外层内部仍按完整轨迹选择岭回归正则，
没有把同一条轨迹的相邻帧拆到训练和验证两边。fit worker 只能收到训练特征与训练
teacher，predict worker 只能收到冻结模型、heldout 特征和 heldout raw 质心；
heldout teacher 要等预测原子发布后才允许用于评分。

正式结果不是“平均有提升”，而是逐轨迹判决：

| 轨迹 | candidate p90 | candidate exact | 判决 |
|---|---:|---:|---|
| p33-s01 | `0.310` | `98.02%` | PASS |
| p45-s05 | `1.219` | `18.81%` | FAIL |
| p58-s03 | `0.601` | `59.41%` | FAIL |
| p14-s05 | `0.962` | `30.69%` | FAIL |
| p22-s03 | `0.334` | `80.20%` | PASS |
| p14-s01，已见开发验证 | `0.233` | `76.24%` | PASS |

独立 outer-crossfit 实际只过 `2/5`；表中的 p14-s01 是已经看过的开发验证，贡献
另外 `1/1`。所以合计虽是 3/6，正式状态仍是
`FAIL_T0_OBSERVATION_CENTROID_CALIBRATION_PROXY`，不是“3 条成功、3 条再调一调”。
三个失败还不是同一种原因：

- p45 的 p90 和 exact 都有材料性改善，但离冻结门仍很远；
- p58 的连续质心误差略降，整数 exact 却比 equalized control 更差；
- p14-s05 的 8 特征训练范围覆盖率高达 `93.07%`，模型却几乎退回 raw BP，并触发
  p90/exact harm。

这也否掉了一个看似方便的事后解释：特征是否落在训练 min/max 内，不能直接当
fallback。p33 的联合覆盖率是 `0%` 却通过，p14-s05 是 `93.07%` 却失败。覆盖率可以
帮助理解分布差异，不能替代真实的逐轨迹场精度与伤害门。

独立 validator 没有导入正式 calibrator、runner 或评分 helper，重新计算 8 个特征、
331 次岭求解、6 个模型、6 条预测和全部判决。43/43 个正式数组的最大绝对差为
`0.0`，验证前后正式结果树逐字节不变。runner 的 `1.288 s` 与约
`69.9/39.8 MB` parent/child RSS 只是复用既有 observation、BP 和 teacher 后的校准
审计成本，不是完整 observation→`A^T`→三维初值的端到端成本，不能写速度或内存成功。

**讲人话：**我们试图只看火焰“中心往哪里走”，再把一个固定模板搬过去。但三维场
不只有中心位置，还有形状、边界、局部梯度、多个团块和不同视角下的模糊。15 个参数
可以修正部分系统偏差，却不能把这些被丢掉的信息变回来。继续围绕质心调特征和阈值，
很可能只是在修一个不够用的代理目标。

所以下一步不再估整数 shift，也不再做 shifted-POD。新的结果前协议要直接研究
observation/BP 到完整 `16×16×32` warm field 的低容量空间映射，按顺序比较：

1. identity BP；
2. geometry-equalized BP；
3. 局部 separable 3D convolution/ridge；
4. 低模频域 transfer；
5. 前四项在完整 trajectory 上确有 headroom 后，才训练一个小型 BP-conditioned
   3D U-Net sentinel。

新门直接看 full-field、gradient、observation、逐轨迹 harm 和同一 CGLS/PCGLS
refinement 后的 matched accuracy。FNO、UNO、DeepONet 仍在 sentinel 之后；p22
在 v8 本轮没有读取，但它早期已用于 classical depth/半收敛诊断，因此不能再算
fresh stopping validation；两条 untouched test 继续关闭。

**突破监测：没有算法突破。** 这次新增的是可信的路线淘汰证据和更直接的下一实验
问题。`neural_training_authorized=false`，`algorithm_breakthrough=false`，
`paper_success=false`。

## 228. v9 先锁问题再写模型：Cross14 完整三维 residual warm start

v8 失败后，这一轮没有马上训练 3D U-Net。我们先把“完整三维初值”到底指什么、和谁
比、怎样才算有用写成机器协议，并让两名独立审计代理专门找漏洞。

第一轮科学审计指出，原先暂定的 8 参数二阶差分模型虽然很小，但把正负方向响应绑在
了一起，也没有一个足够清楚的强正则回退。最终主候选改成两通道、每通道 7 个十字
邻域位置的 residual ridge：

```text
q = G(A^T y)
e = G(W q)
target = (CGLS_K4(0,y) - e) / sr
x0 = G(e + sr * Ctheta(q/sq, e/se))
```

`Ctheta` 只有 `2×7=14` 个 float64 权重，没有 bias。七个位置是
`center, -x, +x, -y, +y, -z, +z`；权重跨全部体素和帧共享。`sq/se/sr` 只能由
当前 fold 的训练轨迹计算。`lambda` 趋于无穷时，修正项归零，严格退回 equalized
BP。这样模型如果学不到跨工况规律，会回到一个明确的基线，而不是产生任意平均场。

五条 fit 仍做 nested leave-one-complete-trajectory-out；p14 只是已经见过的
mandatory veto。主候选固定做 K2 refinement，完整调用账是：

| 阶段 | A | A^T |
|---|---:|---:|
| raw/equalized BP | 0 | 1 |
| 非零初值 residual | 1 | 0 |
| 两步 CGLS | 2 | 2 |
| 总计 | 3 | 3 |

所以 Cross14-K2 是 6 次完整调用，Zero-K4 是 8 次；但 Zero-K3 同样只有 6 次，
因此 Cross14 还必须逐轨迹在 field/gradient p90 上不差于 Zero-K3，并至少一项改善
2%。此外还冻结了 raw/equalized identity、两种 observable line search、44 参数
DCT low-mode residual、dual ridge、geometry-PCGLS 和 normalized-BP 对照。不能
看完结果后把某个 control 改名为主算法。

审计还抓到一个更重要的证据历史错误：`p=22kw_size=01` 在 v8 没有读取，不代表它
从未被读取。早期 classical runner 已用它做 depth 和半收敛诊断，所以它不能再叫
fresh stopping validation。v9 明确禁止再用 p22 选择模型、K、阈值或救失败结果。

我们在只看官方 metadata 的条件下预指定 `p=45kw_size=03` 为一次性 proxy holdout。
当前项目受管的 `private_data/PoolFire` 和 `private_results` 范围内，对
`45kw_size=03/p45s03` 的名称与文本审计均为零命中，没有 receipt、raw/partial、
derived rho、pair 或数字结果。但是这只能支持“项目受管范围内、实现前选定的 fresh
proxy holdout”，不能证明整台机器历史上绝对没有手工下载或删除过副本，也不能称
密码学盲测、untouched test、unseen-power 或 unseen-size。

为此新增了单独的 metadata-only selection receipt，并继续禁止获取 p45-s03。
只有 development 模型、scaler、lambda、exact K2、solver、指标、阈值、controls、
runtime harness 和报告模板全部锁定，才允许再写一个一次性 no-replace release
receipt。

协议与实现顺序也不再靠一句布尔声明：

- 协议提交：`10415cf`；
- holdout selection 提交：`db0dbba`；
- validator 实现提交：`78ae519`；
- validator 用真实 `git merge-base --is-ancestor` 检查协议提交是当前实现提交的
  严格祖先，并从历史提交重新读取协议核对 SHA；
- canonical bytes 和递归 key/type schema 同时冻结；
- duplicate key、NaN/Infinity、extra/missing、parent 改写、历史 p22 洗白、
  fresh holdout 提前打开、truth 泄漏、inverse crime、容量偷增、DCT cutoff 搜索、
  lambda 扩网格、调用漏账、pooled-frame 伪重复和突破声明抢跑都会 fail closed。

定向测试现在是 `21 passed`，无需额外 `PYTHONPATH`；CLI 状态为
`PASS_FROZEN_POOLFIRE_C_FULL_FIELD_LOW_CAPACITY_PROTOCOL_V9`。这只证明协议和
证据顺序可执行，不是模型、重建、速度或内存结果。

**讲人话：**这轮不是“又写了一堆规则”，而是终于把下一次试验变成一个很难自欺的
问题：14 个局部权重能不能让同样 6 次物理调用比 Zero-K3 更准，并达到 Zero-K4 的
终点？如果不能，就停；如果能，还要过 wall、RSS 和一次未打开工况，才有资格训练
第一个小型 3D U-Net。

**突破监测：没有算法突破。** 当前只允许进入不读取 p45-s03 的 core/unit-test 和
fit/deployment/score 三进程实现。`fresh_v9_holdout_opened=false`，
`neural_training_authorized=false`，`algorithm_breakthrough=false`。

## 229. Cross14 的“零件”写完并审过了，但还没有开始报性能分数

v9 协议冻结后，这一轮只实现不需要读取轨迹的数值核心。现在已有一个明确的
`Cross14`：输入 raw BP 和 geometry-equalized BP，每个通道只取当前体素与六个相邻
体素，总共学习 14 个共享权重。它输出完整 `16×16×32` 初值，再接固定两步 CGLS。

代码没有把 606 帧全部摊成一个巨大的设计矩阵，而是逐帧累计一个 `14×14` Gram
矩阵。每条训练轨迹先单独求均方，再做轨迹等权平均，避免“帧数多的轨迹多投票”。
正式模型只有五个事前冻结的 lambda；canonical JSON 和 digest 会拒绝改权重、加字段、
NaN 或非规范文件。

边界测试专门核对了 `center, ±x, ±y, ±z` 在三个轴上的 reflect 规则，结果与独立
`numpy.pad(mode="reflect")` 全数组一致。还检查了常数场、gauge、正比例齐次、退化
Gram、EVD 与直接法、充分统计量目标、模型序列化和强正则回退。

独立审计抓到一个值得修的口子：最初 runner 虽然不再接受任意函数，却仍能收到一个
手工拼出的 equalizer 对象。现在 runner 会重新计算冻结的
`median / max(sensitivity, floor)` 公式并核对 geometry-only 报告，伪造 multiplier、
floor 或 truth-access 标记都会拒绝。不过“这个 sensitivity 是否真的来自本次正式
operator”还必须由下一层 geometry digest 和 manifest 证明，不能只靠 Python 对象
类型。

完整调用账也被代码断言：

| 阶段 | A | A^T |
|---|---:|---:|
| raw/equalized BP | 0 | 1 |
| 非零初值 residual | 1 | 0 |
| CGLS K2 | 2 | 2 |
| 总计 | 3 | 3 |

核心单测 `31 passed`；与 baseline、v9 protocol validator 联合是 `69 passed`；
Ruff、编译和 diff 检查通过。实现提交为 `fc97cd7`。独立审计没有 P0，允许的正式
状态只有：

`PASS_NO_DATA_CROSS14_CORE_CODE_GATE_ONLY`

**讲人话：**我们把发动机零件尺寸、装配方式和油耗表上的计算规则都验了一遍，但车
还没上赛道。下一步要把 fit、deployment、score 拆成互相看不到不该看数据的进程，
再在完整轨迹上和同成本 Zero-K3、目标终点 Zero-K4 公平比赛。

本轮没有读取任何 PoolFire trajectory、`p45-s03`、历史 p22 或两条 test。还没有
跨轨迹精度、wall、RSS 或算法优势。当前：

```text
equalizer_provenance_bound=false
process_truth_free_proven=false
independent_noninterference_proven=false
trajectory_split_proven=false
neural_training_authorized=false
algorithm_breakthrough=false
paper_success=false
```

## 230. 同一个审计脚本为什么先失败，拆成两个运行时后才真正可信

正式五个 outer fit 和一个 p14 development fit 已经完成，但我们没有马上看它们在
heldout 轨迹上表现怎样。先要证明两件事：训练数组真的是由冻结物理代理生成的，六个
模型也真的是由这些数组按 nested LOTO 和 one-SE 规则拟合的。

第一版 v9.2 想在一个进程里同时逐位证明两件事，结果正确地 fail closed。进一步取证
发现，旧 source artifact 是 Python 3.13.9 / NumPy 2.3.5 生成，正式 fit 是
Python 3.11.5 / NumPy 2.4.6 生成。旧运行时能够让 606 个 K4 teacher 与保存结果逐位
相同，新运行时能够让六个 ridge model 与保存结果逐位相同；交换运行时会出现最后几位
浮点差异。

这里最容易犯的错，是看见差异只有 `1e-15` 左右，就临时加一个 tolerance 让测试绿。
我们没有这样做。提交 `9764ce3` 先冻结双运行时协议，明确两个角色各自能看什么、能做
什么、两份 receipt 怎样绑定；之后提交 `fddb40b` 才实现 validator。

Source role 的正式结果是：

```text
PASS_RUNTIME_BOUND_SOURCE_TO_REQUEST_V9_3
```

它重算六条轨迹 606 帧的 raw BP、sensitivity、multiplier、equalized BP 和 K4
teacher，五类最大绝对差都是 `0.0`。调用账也是 `0A+606A^T` 与
`2424A+2424A^T`。它没有读取 fit outputs，也没有运行 nested fit。

Fit role 的正式结果是：

```text
PASS_RUNTIME_BOUND_SOURCE_AND_NESTED_FIT_BATCH_V9_3
```

它先确认 Source receipt 仍绑定当前未变化的 request tree，再独立重算六个模型。
五个 outer heldout 恰好各覆盖一次，p14 单独报告；六行都选择 `lambda=1e-4`，全部
selection 和最终 14 参数模型逐元素一致。26 项测试还会拒绝错误运行时、过期 receipt、
软链接、错误 commit、伪造 PASS 和抢跑 breakthrough。

**讲人话：**以前我们只知道“厨房交出了六盘菜”。现在分别由食材检验员证明原料来自
正确供应链，再由另一个厨师按同一菜谱重做六盘且完全一致。我们仍然没有让评委打分，
所以不知道菜好不好吃；但至少不再怀疑端上来的东西是不是换了原料或改了配方。

这次是**证据链突破**，不是算法突破。它只授权下一步在看任何 outer 数字前冻结评分
协议。当前仍是：

```text
outer_scoring_authorized=false
matched_accuracy_proven=false
wall_time_speedup=false
algorithm_breakthrough=false
paper_success=false
```

## 231. “独一无二”不能靠改名字：Cross14 降级为哨兵，GEOK-Warm 才是方法假设

这一轮把我们的优化目标重新写成固定顺序：

1. 先让最终 field、gradient、observation 精度逐轨迹等价；
2. 再比较完整 `A/A^T`；
3. 然后实测 fresh-process wall；
4. 最后看 whole-pipeline peak RSS；
5. 任一坏轨迹的 harm 不能被平均值遮住。

这也澄清了 Cross14 的身份。它只有 14 个局部共享权重，很适合检查“自由局部三维
residual 有没有跨轨迹 headroom”，也适合当可解释 control。但 BOST 神经重建、
learned warm start、neural operator + Krylov、对角 sensitivity equalization 都有
明确先例，所以 Cross14 不能撑起“新算法”。

更关键的物理问题是零空间。任意 3D CNN/FNO 输出的初值可能包含
`Null(A)` 分量。后续 CGLS 的修正位于 `Range(A^T)`，无法消除这部分错误；网络图像
可能看起来平滑、measurement residual 也可能不错，但不可观测幻觉会一直留在最终场。

因此新的论文级假设暂命名为 GEOK-Warm：

```text
q0 = P A^T y
e0 = P D^-1 q0
q1 = P A^T A q0
(c0,c1) = G_theta(deployment-visible summaries)
h = c0 q0 + c1 q1
```

`e0` 只提供 geometry sensitivity 条件；真正的 warm field `h` 被限制在
`K2(A^T A,A^T y)`，因此位于 `Range(A^T)`。再用 `A h` 做解析 measurement-residual
尺度校准，最后交给不变的 CGLS/PCGLS。网络不直接吐最终重建，也不在每次迭代中替代
真实 forward。

它仍不能保证成功。我们只能说：在已经核对的 NeRIF、NeDF、Neural Refractive Index
Primitives、JMLR learned warm starts、NOWS、FCG-NO、2026 年 6 月的 Spectrally
Safe Neural Operator Warm-Starts 和七项高风险专利中，尚未找到与“BOST
geometry-equalized observable + 可观测 Krylov 受限初值 + 未修改 solver +
matched-accuracy 成本门 + deployment-visible fallback”完整同构的单项。每个组成
部分都有近邻，所以独特性来自完整结构、理论命题和真实效果，不来自名字。最新近邻
也说明“solver-safe warm start”不能再泛称原创；我们必须证明的是 BOST 中
`Null(A)` 不可纠正风险与 `Range(A^T)` 受限初值的特定机制。

**讲人话：**不能保证全世界从来没人有过相似念头。能保证的是，我们不会把别人做过
的零件重新命名；我们会逐项写清来源，把初值限制在物理可纠正空间，并用最强近邻和
真实成本去打。如果最后赢了，差异清楚、证据完整；如果没赢，也会留下一个可信、可
复现、真正属于自己的负结果。

下一步仍按顺序：

1. 结果前冻结 Cross14 outer prediction/score 与全部强 controls；
2. 先看自由局部 residual 是否有 headroom；
3. 有 headroom 才单独冻结 GEOK-Warm，不用大模型救失败；
4. 通过完整 trajectory、fresh holdout、wall/RSS 后，再申请真实 BOST 迁移；
5. 投稿前重跑文献/专利 claim chart，并询问师兄组内是否有未发表近邻。

当前：

```text
Cross14 = sentinel
GEOK-Warm = unvalidated method hypothesis
global_uniqueness_proven = false
defensible_novelty_space_identified = true
algorithm_breakthrough = false
```

下一门是绑定 protocol、代码提交、trajectory 角色、geometry/equalizer、solver、
runtime 和报告模板的三进程 manifest。五条 outer LOTO 与 p14 veto 全部过门前，
继续不获取 `p45-s03`。

## 232. 补记：三个目录角色分家了，但审计禁止我们把它叫成正式实验

上一节说 Cross14 的数值零件已经写完，缺的是证据角色。这一轮把流程真正拆成了三个
独立命令：

1. `fit` 只收到训练轨迹的 raw/equalized BP 和 K4 teacher；
2. `deployment` 只收到冻结模型、heldout observation 和冻结 geometry；
3. `score` 必须等 initializer 与 K2 candidate 原子发布完成，才收到 teacher/truth
   做离线评价。

可以把它理解成考试：出题人先把复习材料交给训练进程；考生进考场时只拿题目和已经
封好的模型；交卷且封存以后，阅卷进程才拿到标准答案。这样比在一个 Python 脚本里
写一句“这里不读取 truth”更可信。

每个请求目录都有精确白名单。多一个 `.DS_Store`、truth 文件、目录、软链接或硬链接，
都会在运行前失败。请求还绑定了外层 digest、全部关键源码、Python/NumPy/BLAS、
trajectory 角色、geometry 坐标、从同一个 operator 重建的 equalizer、模型 payload
和上一阶段 READY/checksum。输出先写临时目录，逐文件 fsync，最后原子发布且拒绝覆盖。

这次的数值核对不是“又跑了一遍同一个函数”。validator 没有导入正式 Cross14、
三角色 worker、baseline CGLS 或评分 helper，而是重新写 14 维特征、Gram/EVD ridge、
geometry/equalizer、K2 和三种 metric。合成 fixture 上 fit、deployment、score 的
逐数组最大绝对差都是 `0.0`。但它仍复用了正式 straight-ray operator primitive，
所以准确说法是“上层数值路径独立重写”，不是完整 operator 独立验证。

更重要的是，后实现审计没有因为测试全绿就放行。它抓到三个 P0：

1. 原 worker 接受调用者手选 lambda，甚至只给一条 fit 轨迹，也可能写出 formal PASS，
   绕过协议冻结的 inner LOTO 和 one-standard-error rule；
2. fit 的三个数组只有字节 SHA，没有 trajectory-axis、pair registry、geometry 和
   teacher generation 的语义凭据；
3. score 没有证明它看到的 observation 与 deployment 相同，也没有证明 teacher 是
   由同一 observation/geometry 的 Zero-K4 产生。

我们没有把这些问题改名成“残余风险”后继续跑数。当前 worker 已被严格降级为
synthetic-only：即使请求 101 帧 formal run 也会失败。deployment receipt 新增
observation SHA；score 必须逐字节匹配，并重新运行同一 observation/geometry 下的
Zero-K4 与外部 teacher 核对；truth、teacher、candidate、initializer 都必须满足
`1e-12` gauge 门。claim 字段也改为与冻结协议逐字一致。

部署主账仍是每帧 `3A+3A^T`。score 旁账现在是 `6A+4A^T`：其中 `4A+4A^T` 用于
同源 K4 复算，另外 `2A` 投影 candidate 和 teacher；这不能混进部署成本。gradient
metric 使用真实网格间距的三个方向 forward difference 拼接。

新增负向测试会拒绝 formal 101 帧、部署后 observation 换包、伪造 teacher 和非零
gauge truth。源码提交为 `12ea0d5`；三角色定向测试现在是 `19 passed`，联合回归
`112 passed`。修复后的第二轮只读复审结论是当前 synthetic-only 声明范围内
`P0=0 / P1=0`。

**讲人话：**我们先搭了三个房间，但审计发现“谁决定参赛模型”和“标准答案从哪来”
还没有正式门禁。因此现在宁可锁死正式模式，也不拿两帧合成演示冒充比赛。下一步先
写外层裁判：机器固定五个 outer fold、每个 inner LOTO、全部 lambda、one-SE、训练
数组来源和 clean commit；然后才跑五条 101 帧、p14 veto、强基线、wall 与整流程 RSS。

当前严格状态：

```text
PASS_SYNTHETIC_V9_THREE_ROLE_CODE_GATE_ONLY
PASS_SYNTHETIC_V9_THREE_ROLE_NUMERICAL_RECOMPUTATION_WITH_SHARED_OPERATOR_PRIMITIVE_ONLY
formal_v9_scientific_gate_implemented=false
formal_101_frame_run_completed=false
independent_full_protocol_validation_proven=false
development_LOTO_completed=false
wall_time_speedup=false
algorithm_breakthrough=false
paper_success=false
```

## 233. 作品身份再次收紧：网络必须先打赢“解析二维投影”

这一轮没有为了“独一无二”再造一个更花哨的模型名，而是把作品压缩成六项不可拆开的
指纹：BOST 特定逆问题、只读部署可见量、受限初值、同一物理求解器收尾、可见量决定
回退，以及 trajectory 尾部和完整成本共同裁决。

新核对的近邻让边界更严格了：

- Deep Null Space Learning 和 Deep Decomposition Learning 已经讨论神经网络如何利用
  range/null-space 并保持数据一致；
- Bayes Meets Krylov 已经用先验和右预条件器改变 CGLS 的 Krylov 子空间；
- Neural Preconditioning via Krylov Subspace Geometry 已经用主角度损失和可微
  FGMRES 训练神经预条件器。

所以“用了零空间”“用了 Krylov 几何”“网络帮助 CGLS”都不能算我们的原创点。

更关键的是，GEOK 暂定的 `q0,q1` 本来就是一个二维 Krylov basis。只用观测 `y`
就可以在这个二维空间里做 exact projected least-squares，解析求出 measurement
residual 最优系数。若神经网络连这个几乎零参数的 control 都打不过，它只是在更复杂地
重新发明 CGLS。

未来 GEOK 的预注册因此必须增加：

```text
exact 1D line search
exact 2D Krylov/Galerkin projection
zero-start call-matched CGLS
fit-only fixed coefficients
observation-conditioned coefficients
```

只有最后一项在相同调用预算下改善最终 field/gradient、p90/worst 与 harm，同时真实
wall/RSS 不更差，才能说明它学到的是观测条件化的场先验。

工程上，本轮还补了两块“以后不能偷账”的地基：

1. 提交 `98b2f94` 分开记录 trainable 参数、常驻数值工件、非 `A/A^T` MAC 与完整
   算子调用，11 个 outer arms 都有显式公式；
2. 提交 `a06070c` 实现每个 row-arm 五次全新 PID 的 wall/wait4 RSS harness，但正式
   入口继续 fail-closed，直到可信 prediction release 真正绑定 worker 与 66 个输出。

联合检查分别为 `108 passed` 与 `56 passed`。这些只提高成本证据可信度，不说明方法
更准或更快。

**讲人话：**现在不是给作品贴“原创”标签，而是主动找一个最省、最聪明的经典对手来
打。连它都能赢，而且每一笔物理调用和内存都算清，作品的独立性才站得住。

当前：

```text
global_uniqueness_proven=false
defensible_method_fingerprint_frozen=true
exact_2d_krylov_control_required=true
formal_outer_runtime_authorized=false
algorithm_breakthrough=false
```

## 234. “独特”不是没人用过这些零件，而是整套组合经得住最强反例

又完成了一轮只查公开一级来源的近邻审计。结论比“我们很新”更有用，也更严格：

- 神经网络给迭代器初值，别人做过；
- 神经算子帮助 CG/GMRES，别人做过；
- 神经网络重建 BOST，别人做过；
- BOST 里 coarse-to-fine、低分辨率结果给高分辨率 CGLS 当初值，也已经有
  Pyramid-BOST。

所以不能把 `Cross14`、`FNO + CGLS` 或“learned warm start”本身写成创新。真正值得
保留的研究命题只有这一整套组合：

```text
BOST-specific frozen geometry
+ deployment-visible observation/BP only
+ one-shot observable-Krylov constrained initializer
+ unchanged CGLS/PCGLS final solver
+ matched field/gradient/observation endpoint
+ complete A/A^T, wall and RSS accounting
+ truth-free acceptance/fallback
```

截至 2026-07-26 核对的 20 项核心公开近邻中，没有发现一项把这些条件全部同时做到。
这叫“限定检索范围内未发现同构组合”，不叫“全球首创已经证明”。专利、学位论文、
未索引稿件和组内未公开方案仍需继续核对。

这轮新增了六个最危险的阅读入口：Pyramid-BOST、UBOST、Direct-RBF BOST、Hybrid
refinement、HINTS 和 NeurKItt。它们也变成正式 controls：

```text
pyramid-style coarse initialization
RBF / reduced basis
classical deflation
exact 2D Krylov / Galerkin projection
call-matched zero-start CGLS
learned subspace / neural warm-start control
```

**讲人话：**我们不靠给常见零件换名字来“确保独一无二”。我们先主动找到所有最像的
工作，再把最便宜、最强的办法都放到同一赛道。若 GEOK-Warm 仍能在未见完整轨迹上保持
同样终点精度、降低物理调用、真实 wall 和内存，而且拒绝坏样本时不偷看真值，这套作品
才会有清晰、难混淆的个人指纹。

当前边界：

```text
public_near_neighbor_core_set=20
no_exact_full_combination_found_within_reviewed_sources=true
global_uniqueness_proven=false
group_unpublished_ip_checked=false
formal_outer_result_opened=false
algorithm_breakthrough=false
```

## 235. 红队否掉“看起来新但不省调用”的旧方案，主候选改成 DualRange-K1

这轮先做了一件比继续堆模型更重要的事：重新把旧 GEOK v0 的每一次完整正演和伴随
都算了一遍。

```text
q0=A^T y                    0A + 1A^T
q1=A^T A q0                1A + 1A^T
line search 的 A h          1A + 0A^T
CGLS K2                    2A + 2A^T
总计                        4A + 4A^T
```

这与 Zero-CGLS K4 完全相同。也就是说，旧方案也许能改变初值表示，却没有完成师兄要求
的“同精度下降低重建成本”。它现在被明确降级为失败候选，不再靠一个好听的名字留在
主线上。

修订后的最小候选叫 DualRange-K1。网络不直接猜三维场，只输出与观测同形状的
`z_theta(y)`：

```text
h = A^T z_theta(y)
x0 = alpha h
then unchanged CGLS K1
```

`alpha` 用只看观测的解析线搜索求，区间包含 0。这样有三条硬性质：

1. `x0` 天然属于 `Range(A^T)`；
2. 初始 measurement residual 不会比零初值更坏；
3. 接受分支完整账严格是 `2A+2A^T`，相对 Zero-K4 理论减少 50%。

代码与原 baseline 联合测试为 `22 passed`。随后又在冻结的 `16×16×32` 三维场、
2072 维三视角几何上做了不读 rho、不读真值的随机观测烟测：

```text
alpha=0.061941614953514523
initial_residual_ratio=0.80221651960594
initializer_field_mean=9.215718466126788e-19
total_calls=2A+2AT
```

**讲人话：**现在至少不是“在纸上省调用”。同一个正式几何接口已经真的跑通，而且账
对得上。但它还没有学到任何东西，也没有打开 outer 性能结果，所以不能说算法成功。

近邻审计也补到了 25 项一级来源。学习反投影、sinogram filter、learned warm start、
neural operator + Krylov 和 BOST 神经重建都有人做过。当前可能形成个人作品指纹的，
只能是下面这整套东西一起成立：

```text
BOST-specific observation-space proposal
+ by-construction Range(A^T) lift
+ pre-A^T deployment-visible risk gate
+ unchanged CGLS K1
+ field/gradient/observation non-inferiority
+ complete A/A^T, wall, RSS and tail-harm evidence
+ real BOST transfer
```

下一步不是马上训练大模型，而是结果前冻结 DualRange-K1 的 outer contract，并把
`z=y` normalized BP、exact 1D/2D Galerkin、dual ridge、call-matched CGLS 和 learned
backprojection 全部放进同一张赛表。

当前边界：

```text
old_geok_v0_call_reduction_claim=rejected
dual_range_k1_mechanism_gate=passed
formal_outer_performance_opened=false
field_or_gradient_no_harm_proven=false
global_uniqueness_proven=false
algorithm_breakthrough=false
```

## 236. 把“我要做得独一无二”改写成不能随结果变的 v10 合同

这一轮没有训练网络，也没有偷看 outer 分数。先把真正的优化目标锁死：

```text
先逐轨迹满足 field / gradient / observation 单侧非劣
-> 再看 p50 / p90 / worst / harm
-> 再把完整 A/A^T 从 4+4 降到 2+2
-> 再要求五轨迹等权 wall 中位数至少快 10%
-> 再检查整流程 peak RSS
-> 最后才进入真实 BOST
```

机器合同是
`learning_labs/protocols/poolfire_c_dual_range_outer_contract_v10.json`。它继承 v9.4
的六行完整轨迹、三类误差和开发容差，也继承 v9.4.1 的全局 prediction barrier、
fresh-exec wall/RSS 与资源账；但把主方法改成 DualRange-K1，并锁定 17 个正式 arms。

最重要的同预算对照包括：

```text
Zero-CGLS K2
z=y 的 normalized-BP + CGLS K1
exact 2D projected least-squares / Galerkin
fit-only fixed dual filter
dual ridge + CGLS K1
不受 Range(A^T) 约束的 direct-field model + CGLS K1
```

如果 DualRange 只比 Zero-K4 便宜，却打不过这些同样 `2A+2A^T` 的简单方法，就不能说
网络学到了有用的 BOST 先验，也不允许靠换大 FNO/UNO/DeepONet 救场。

这轮又补查了六个很危险的近邻：

- JMLR 2024 *Learning to Warm-Start Fixed-Point Optimization Algorithms*；
- 2026 *Pretrain Finite Element Method*；
- 2025/2026 *MD-PNOP*；
- 2026 *Convolutional neural network-driven preconditioners for conjugate
  gradients*；
- 2025 *A Warm-basis Method for Bridging Learning and Iteration*（WB-IPM）；
- 2025 *Learned ReSeSOp for solving inverse problems with inexact forward
  operator*。

它们进一步说明，learned warm start、神经初值后接经典 solver、保持最终物理解和
学习 Krylov 加速都不是我们的单点创新。WB-IPM 是目前最危险的结构近邻：它已经让
网络生成 warm basis，再进入增强 Golub-Kahan/Krylov 投影。我们的观测域 proposal、
精确 `A^T` 提升、原样 CGLS K1 和 pre-`A^T` 回退仍是差异，但在 outer 与真实 BOST
证据通过前还不能叫贡献。现在只保留七件套的组合级作品指纹：

```text
BOST-specific dual proposal
+ Range(A^T) lift
+ pre-A^T gate
+ unchanged CGLS K1
+ three-metric trajectory non-inferiority
+ full call/wall/RSS/tail accounting
+ real-BOST transfer
```

**讲人话：**我不能替全世界、专利库和组内未发表工作担保“绝对没有第二个人想到”。
能做的是把作品的七个维度同时做实，让它即便每个零件都能找到近邻，完整方法仍有清楚、
可防御、很难混淆的个人指纹。如果未来找到完整撞题，合同要求改题或停止首创主张，而
不是装作没看见。

v10 现在明确不授权训练。下一步只允许先冻结一个最小 `G_theta` 架构、参数上限、
fit-only nested trajectory 选择、K1 后损失、种子/checkpoint 和全局 prediction
barrier；完成后才允许生成性能输出。

独立实验红队随后发现第一稿验证器仍可能被篡改阈值、篡改 control 调用数、把 truth
塞进 gate 白名单，而且旧 v9.4.1 的 66-output roster 与新方法不相容。已经按这个
反例把 v10 改成：

```text
17 arms x 6 rows = 102 atomic predictions
ungated DualRange + gated policy 必须同时报告
direct-field K1 与 PCGLS K2 进入最强同价或更便宜 control 集
逐帧 A/A^T receipt
17 次 fresh exec 循环平衡 arm 顺序
固定 contract SHA + freeze receipt + report template
```

验证器现在会拒绝非劣门改成 0、wall 门改成永远通过、Zero-K2 写成 999 次调用、gate
允许 `field_truth`、训练前置条件被清空、负调用伪装成总和 `2+2`，以及删除 WB-IPM
碰撞边界。第二轮复审又补上 freeze receipt 的精确 identity/authorization/claim
校验，并锁住 score-token bindings、failure actions、fresh guard 和 wall 生命周期。
v10 合同篡改定向检查为 `14 passed`，联合页面/机制/Pages builder 为 `117 passed`。
仍未关闭的是未来执行实现：pre-`A^T` gate、模型/gate 的 `execve`
能力隔离、全部 controls、正式 runner/validator 和 102-output batch seal。

当前边界：

```text
optimization_objective_frozen=true
combination_level_fingerprint_frozen=true
reviewed_primary_source_count_minimum=31
closest_structural_collision=WB-IPM
formal_arm_count=17
formal_prediction_count=102
accepted_branch_target=2A+2AT
reference=Zero-CGLS-K4@4A+4AT
pre_AT_gate_implemented=false
formal_capability_isolation_proven=false
model_training_authorized=false
outer_performance_opened=false
global_uniqueness_proven=false
algorithm_breakthrough=false
```

## 237. 最小 G_theta 已冻结，但独立审计把正式训练挡在正确的位置

这轮没有训练网络，也没有打开 outer 分数。先把最小候选写成了不能随结果变的结构：

```text
2072 维三视角 observation
-> 固定 4x4 DCT，96 个系数
-> 3 个视角能量 + 1 个投影能量
-> 100 -> 32 -> 32 -> 96 MLP
-> 有界 DCT residual
-> z_theta(y)
```

奇对称化会让输出 bias 永远相消，所以把它删掉了。最终是 7,360 个 float64 参数，
占 58,880 bytes。模型保证 `G(0)=0`、实数缩放齐次和
`||G(y)-y|| <= 0.5||y||`，但这些都不是精度或速度结果。

为了避免把 DCT/RMS 的收益误写成神经收益，又实现了三种同表示对照：

```text
clipped identity DualRange      0 参数
diagonal DCT dual filter       96 参数
full linear DCT dual map     9216 参数
```

它们和 MLP 使用同一 RMS、DCT、修正上限、`A^T`、alpha 与 CGLS K1。训练损失也改为
在同一个 K1 之后比较 K4 teacher，并按完整 trajectory 计算 mean + worst-11 tail；
不能随机拆帧。

更关键的是，独立代码审计找到两个必须先修的 P0：

1. 旧 Python callback 可以绕过 `AuditedLinearOperator`，偷偷多调用底层 `A^T`；
2. 旧通用 solver 在 denominator breakdown 时会设 `alpha=0` 后继续，不符合 v10
   的 fail-closed 规则。

我把两者都做成了可重复的负面测试。第一个反例里真实底层是 `1A+2A^T`，wrapper
却只记 `1A+1A^T`；第二个执行门因此也不能靠一句“使用同一 CGLS”带过。identity
DualRange 与旧 normalized BP 的命名混淆也被反例纠正：当 `A=0.5I` 时，旧 BP
scale 为 4，而 DualRange alpha 被截到 2。

当前定向联合检查：

```text
61 passed
minimal_model_inference_implemented=true
matched_linear_control_inference_implemented=true
strict_v10_solver_implemented=false
callback_free_proposal_receipt_implemented=false
model_training_authorized=false
outer_performance_opened=false
algorithm_breakthrough=false
```

**讲人话：**模型的样子终于定清楚了，但现在最重要的不是立刻让 Mac 开始训练，而是
先保证模型不能偷用物理算子、每次 `A/A^T` 都真的记得住、所有 arm 遇到数值 breakdown
时用同一种规则停。下一轮先写 strict solver 和“数值 proposal + 一次性收据”，再做
隔离 worker；这比跑出一个账不可信的好数字更接近论文。

## 238. strict solver 代码门通过，但 2A+2A^T 仍没有正式落地

这一轮继续没有训练网络，也没有打开 outer、fresh 或 test。先把旧求解器不符合 v10
的地方单独改成了一条新执行路径。

旧 CGLS/PCGLS 遇到退化 denominator 时会把 `alpha=0` 后继续。新 strict 路径固定为：

```text
检查 gamma
-> 调用 A p 并记账
-> 检查 raw dtype / shape / finite
-> 检查 denominator > 1e-30
-> 在临时数组算 candidate field / residual
-> 检查 finite
-> 最后才提交状态
```

正常满秩问题上，新旧 CGLS/PCGLS 的 K1/K2/K4 逐 checkpoint 数值一致；零算子、秩
退化、有限小分母、NaN/Inf、overflow、错误 dtype/shape 和被篡改 SPD 都会在错误更新
前失败。

第一次独立审计仍找到了五个 P1：初始化调用可能自报漏账、旧 wrapper 隐式转
float64、SPD 数组可重新打开、异常分类太宽、失败回执里 residual 状态不够诚实。修完
后第二次审计又发现公开 raw handle、回执没覆盖 preparation，以及外部 NumPy
`over=raise` 会绕开统一回执。最终处理为：

```text
strict wrapper 改为组合，不公开 raw operator
初始化与求解绑定同一 wrapper 完整生命周期
raw output 在 cast 前检查
PCGLS diagonal 入口重验并私有复制
contract / execution / numerical breakdown 分型
内部固定 np.errstate，再由 finite 门统一裁决
失败回执绑定准备账、求解账、累计账和提交前后摘要
成功回执绑定所有 checkpoint 摘要
```

最终独立复审在当前 same-process code-gate 边界内得到 `P0=0 / P1=0`：

```text
strict focused=26 passed
v10.1 joint=99 passed
all PoolFire C related regression=376 passed
strict_v10_numerical_solver_code_gate_implemented=true
capability_isolated_worker_proven=false
outer_evaluation_authorized=false
algorithm_breakthrough=false
```

**讲人话：**求解器现在不会在坏分母上偷偷走一步，准备阶段和迭代阶段的账也能连起来
看。但网络还没有被关进一个拿不到 `A/A^T` 的独立进程，物理进程也还不能只靠一次性
数值收据接收 proposal。所以“接受分支 2A+2A^T”目前仍是数学目标，不是已经跑通的
正式事实。

完整证据页：
`docs/poolfire_c_dual_strict_solver_code_gate_v10_1_2026-07-26.md`。

下一步只做 callback-free proposal artifact 与一次性 receipt，再接 sibling
inference/physics worker；训练授权继续关闭。

## 239. 数值 proposal 已能一次性接入 CGLS K1，但还没有独立 worker

这一轮仍没有训练网络，也没有打开 outer、fresh 或 test。完成的是 v10.2 数据交接
代码门：物理路径不再接收 Python 模型 callback，只接收一个固定字节格式的
observation-space proposal。

讲人话，它现在像一张只能验一次的数值快递单：

```text
y 的摘要 + 模型/几何/预处理摘要
-> 2072 个 float64 的 z
-> broker 验单并烧掉 request
-> 物理端烧掉 verified token
-> A^T z、A(A^T z)、可观测 alpha
-> 烧掉 initializer authorization
-> strict CGLS K1
```

request、verified token 和 initializer authorization 都是第一次尝试即消费；无效后
不能换一个 payload 重试。header 增减字段、尾随字节、错误 shape/dtype、`NaN/Inf`、
model/geometry/observation 替换、并发双消费和复制状态字符串都会拒绝。

第一次独立红队抓到一个真 P1：当时虽然 solver 不收普通伪造 initializer，但代码仍有
一个内部 issuer，调用者可以先手工做 `1A+1A^T`，再配一个假的 artifact SHA 取得授权。
现在这个 issuer 已删除。broker 消费、`A^T z`、`A h`、line search、cache 和授权签发
被合并到 strict operator 的单一入口，不能再把自制 `InitializerPreparation` 塞进去。

复审又找到一个 PCGLS 的孤立 cache 旁路。主方法本来就冻结为 CGLS K1，所以最终把
verified authorization 在消费前锁死为：

```text
require_identity = true
checkpoints = (1,)
```

误送 PCGLS 或 K2 会先拒绝但不烧授权，随后仍能走正确 CGLS K1。observation 错配或
授权后额外调用 operator 则会烧授权并清掉 cache。

当前成功路径的 strict-wrapper 代码账确实是：

```text
range lift + alpha   1A + 1A^T
CGLS K1              1A + 1A^T
total                2A + 2A^T
```

可复现测试：

```text
artifact focused=54 passed
artifact + strict focused=80 passed
v10.2 joint=141 passed
tracked PoolFire C suite=775 passed
final same-process red team=P0=0 / P1=0 / P2=0
```

但这仍不是“算法加速成功”。当前制品只验证字节和绑定，不能证明实际 worker 没读
truth、没偷调底层 operator；wall/RSS 也还没有由隔离父进程测出来。因此必须继续写：

```text
worker_authenticity_proven=false
capability_isolated_worker_proven=false
formal_callback_free_physics_worker_implemented=false
formal_accepted_branch_call_reduction_proven=false
model_training_authorized=false
outer_performance_opened=false
algorithm_breakthrough=false
```

完整证据页：
`docs/poolfire_c_dual_proposal_artifact_code_gate_v10_2_2026-07-27.md`。

下一步只做 sibling `execve` inference / physics worker、父进程 wall / process-tree RSS
测量和 pre-`A^T` accept/fallback；前置门不通过就不训练。

## 240. 三个模型都真的训练了，但 96 个低频修正模态仍装不下 K4

这一轮把“先判断算法值不值得，再扩建正式 worker”真正执行完了。p14 结果打开前先
冻结 v10.3 / v10.3.1：五条 fit 轨迹做完整 trajectory 内层留一，逐 slot 对轨迹
等权，checkpoint 只看完整轨迹 mean-all + worst-11，Diagonal 和 Full linear 从零
初始化，唯一 teacher 是此前封存的 Zero-CGLS K4。

先跑的便宜控制已经说明两对调用很难补回第四步信息：

```text
Zero K4: field/gradient/observation p90 = 0.6332 / 1.2180 / 0.3330
Zero K3:                                  0.6329 / 1.0661 / 0.3713
Zero K2:                                  0.6911 / 0.9503 / 0.4383
```

Exact 2D projected LS 与 Zero K2 的逐帧三指标最大差约 `2.6e-15`。它解析地求两个
系数，但没有制造新的 Krylov 信息。

三种 observation-only proposal 都按冻结配置在 Mac CPU float64 上完成：

| 模型 | 参数 | 选中 epoch | field p90 | gradient p90 | observation p90 |
|---|---:|---:|---:|---:|---:|
| Diagonal DCT | 96 | 20 | 0.6435 | 0.8595 | 0.4385 |
| Full linear DCT | 9,216 | 20 | 0.6389 | 0.8607 | 0.4389 |
| Odd DCT-MLP | 7,360 | 20 | 0.6736 | 0.8677 | 0.4534 |

Diagonal 和 Full linear 的 field/gradient harm 都是 0，但 observation harm 是
101/101。MLP 更差，field harm 为 10.89%，observation harm 仍是 100%；相对更好
的线性控制，field/gradient 所谓改善分别为 -5.43% / -0.96%，没有通过 2% 的冻结
非线性门。

三个 checkpoint 又由不导入正式 runner/torch operator/scoring helper 的 NumPy
validator 重做 proposal、DCT、radial cap、`A/A^T`、alpha、CGLS K1 和全部指标。
三种模型最大逐指标差均约 `1e-14`，封存输入未改变。

**讲人话：**这不是“神经网络没跑起来”。三个模型都跑完了，而且全线性模型已经比
对角模型多两个数量级参数。它们共同保留原 observation，只允许在 96 个低频 DCT
模态里做最多 50% 的修正。field 和 gradient 能变好，observation 却一起停在 K2
附近，说明应该先怀疑输出表示和修正半径，而不是继续把 MLP 加宽。

所以正式状态是：

```text
FAIL_NO_DUALRANGE_V10_3_MODEL_PASSES_DEVELOPMENT_COMPATIBILITY
formal_outer_evaluation_authorized=false
fresh_holdout_opened=false
untouched_test_opened=false
algorithm_breakthrough=false
```

下一步不建 sibling worker，也不打开 p45-s03。先另行冻结一个 post-open 表示上限
诊断：逐帧求当前 96 模态/50% cap 内的 oracle proposal，并分解 K4 所需 dual
correction 的模态内外能量。oracle 也失败就关闭 `B_96`，转向预注册的多分辨率
detector-graph / spectral residual；oracle 通过才查 fit 目标与优化。

完整图表和证据见
`docs/poolfire_c_dual_development_screen_v10_3_result_2026-07-27.md`。

## 241. 完整 K3 重启能过，但 96 个低频系数仍装不下

v10.4.2 已经完整跑完并通过独立复算。先说最重要的：它没有发现算法突破，但把失败
位置向前推进了一大步。

在同一个已打开的 p14 开发轨迹上，保留完整 K3 dual certificate，再做一次未修改
CGLS K1，三项冻结门全部通过：

```text
Full K3 restart:
field / gradient / observation p90 = 0.6205 / 1.0923 / 0.3466
joint matched = 100%
joint harm = 0%

Zero K4 reference:
field / gradient / observation p90 = 0.6332 / 1.2180 / 0.3330
```

这说明 `2A+2A^T` 的 restart 外壳并非先天地没有能力。真正失败的是当前表示：

```text
best capped B96:
field / gradient / observation p90 = 0.6663 / 0.8725 / 0.4601
joint matched = 0%
joint harm = 100%

best uncapped B96:
field / gradient / observation p90 = 0.6355 / 0.9072 / 0.4245
joint matched = 0%
joint harm = 100%
```

三个 capped、三个 uncapped 搜索各跑 800 步；独立 NumPy 方向导数检查全部通过。
所以这次不能用“优化器没跑”轻易解释，但仍只能写“有限搜索没有找到 headroom”，
不能写 B96 数学不可能。

更直观的原因是，逐通道 `4x4`、总 rank 96 只保留 K3 dual correction 能量的：

```text
minimum = 37.11%
p10    = 38.52%
p50    = 42.12%
```

这里还纠正了一个统计命名问题：旧机器 summary 把 capture 的最大值放在名为
`worst` 的字段里。capture 越高越好，真正的坏尾部应该是 minimum 和低侧 p10。
修正后结论没有翻转，反而更清楚地说明 B96 覆盖不足。

六个 view-channel 的中位能量份额约为：

```text
8.35%, 0.64%, 8.95%, 0.65%, 42.04%, 39.29%
```

因此下一步不是增加 MLP 参数，而是严格比较：

1. fit-only 联合选择频率与通道计数；
2. 同秩逐通道 `4x4 / 6x6 / 8x8`；
3. rank `96 / 216 / 384`；
4. projection support 与 teacher-oracle headroom 分开判决；
5. rank 2072 必须完整复现 K3 restart 的 proposal、`A^T`、alpha、field、metrics
   和 gate。

红队第一次审查仍给出 `P0=0 / P1=6`，所以 v10.5 没有启动。已经先关闭两个漏洞：

- 工具现在只接受协议中五条 fit 轨迹和固定 rank 梯；
- 五条 fit 合并后的唯一 basis 可以原子、不可覆盖地 seal，p14 阶段不能悄悄换。

当前还缺 fit-heldout 的同壳 teacher deficiency、完整 rank-2072 恒等控制、两阶段
p14 truth 隔离、正式 runner 和独立 validator。它们没闭合前继续写：

```text
v10_5_execution_authorized=false
fresh_holdout_opened=false
untouched_test_opened=false
algorithm_breakthrough=false
```

结果页：
`docs/poolfire_c_dual_representation_ceiling_v10_4_2_result_2026-07-27.md`。

## 242. selected 不是低秩捷径：10 次输给简单均匀 DCT

这一轮先把 v10.5 的比较链补完整，再做了一个不读取 p14 truth 的五轨迹留一检查。
比较很公平：selected 和 uniform 使用相同 rank、相同 `A^T`、相同 observable alpha
和相同 strict CGLS K1，参考都只是 K4 teacher。

结果没有迎合“自适应选择一定更聪明”的直觉：

```text
rank 96:  uniform 5/5 胜，selected 0/5
rank 216: uniform 5/5 胜，selected 0/5
rank 384: selected 5/5 胜，uniform 0/5
```

15 个“轨迹×秩”组合里，均匀控制赢 10 个。selected 的优势只在 rank 384 出现，
而且幅度远小于它在 rank 96/216 上的劣势。讲人话：有限容量很小时，优先保留规则
低频比“按 fit 能量挑频率”更稳；容量升到 384 后，fit 信息才开始带来一点帮助。
所以后续保留两个 family，但不再把 selected 当低秩默认创新点。

第一轮红队还抓到 token 复制重放、final writer 可绕过验证、validator 复用同一 DCT
等问题。修复后：

```text
fixed-ledger token copy replay: rejected
final validation/decision tamper: rejected
validator DCT: independent SciPy implementation
runtime identity: exact 16-file closure + coordinates-derived physics identity
targeted suite: 66 passed
second red team: P0=0 / P1=0 / P2=0
```

这里仍然不能写“算法成功”。fit-only 数字尚未在 clean committed HEAD 下生成正式
不可覆盖 bundle；p14 Stage A 也没有运行，p22 stopping、fresh 和 test 都继续封存。
当前下一步只有：提交审计过的闭包，生成私有 release，再执行一次正式 p14 12-arm
矩阵。`same_UID_filesystem_wide_noninterference_proven=false`，
`algorithm_breakthrough=false`。

## 243. rank 96 真的装得下，但我们还不会从照片里找到那 96 个系数

v10.5 的正式 Stage A 和一次性 p14 独立评分已经完成。和前几轮最重要的区别是：
这次没有再问“某个网络平均误差有没有变小”，而是把两个问题彻底拆开：

```text
projection support:
直接把 K3 correction 投影进 basis，能不能过门？

teacher-oracle headroom:
如果允许在同一个 basis 里逐帧找最佳系数，是否存在能过门的答案？
```

12 臂全部由同一个 clean HEAD、同一几何和同一 K1 外壳生成，候选先封存；一次性
score token 被消费后，独立 validator 才加载 p14 truth。五类执行门全通过，
hard gate failure 为 0。

结果很有方向性：

```text
projection selected 96/216/384: FAIL / FAIL / FAIL
projection uniform  96/216/384: FAIL / FAIL / FAIL
oracle selected     96/216/384: PASS / PASS / PASS
oracle uniform      96/216/384: FAIL / FAIL / FAIL
```

selected rank96 oracle 的 joint matched 为 `93.07%`，joint harm 为 `0%`，
severe harm 为 `0`；rank216/384 的 joint matched 都是 `100%`。反过来，简单
projection 六臂的 joint matched 全为 `0%`、joint harm 全为 `100%`。

**讲人话：**我们已经找到一个 96 维“房间”，答案确实在里面；旧方法失败，不是房间
太小，而是它从照片出发走错了位置。uniform 低频房间即使让 oracle 自己找也没过门，
说明 fit-selected basis 不是装饰。下一步不该继续无边界加 rank，也不该换一个更大
网络碰运气，而是只研究一件事：怎样从 observation 预测 selected basis 里的 oracle
系数。

这也修正了 v10.3 的解释。v10.3 的 96 参数、9,216 参数和小 MLP 共同使用 uniform
低频 `B96` 与 50% cap；现在不能把它们的失败外推成“任何 rank96 detector proposal
都不行”。真正需要做的是 fit-only 的 oracle-coefficient distillation，并在完整
trajectory 留一上先打赢 ridge、全线性和时间持续性控制。

当前边界仍然很硬：

```text
teacher_oracle_headroom=true
observation_to_coefficient_predictability=unknown
learned_predictor_success=false
fresh_holdout_opened=false
test_opened=false
algorithm_breakthrough=false
```

完整脱敏表、解释与下一门见
`docs/poolfire_c_dual_spectral_v10_5_p14_result_2026-07-27.md`。

## 244. rank96 不是跨轨迹房间：五条只过一条，系数蒸馏停止

v10.7 没有直接训练网络，而是先问一个更便宜也更关键的问题：每一折只用另外四条
trajectory 选 rank96 basis，在第五条上允许 oracle 自己找最有利的系数，这个表示
到底装不装得下合格 warm start？

五折 target 都在读 proxy truth 前完成独立逐数组复算。全部封存后一次性评分：

```text
p14-s05: FAIL, joint matched 0%, harm 100%, severe 24
p22-s03: FAIL, joint matched 0%, harm 100%, severe 21
p33-s01: PASS, joint matched 100%, harm 0%, severe 0
p45-s05: FAIL, joint matched 0%, harm 100%, severe 0
p58-s03: FAIL, joint matched 0%, harm 100%, severe 45
```

正式判决是 `1/5`，远低于事前要求的 `4/5`，总 severe harm 为 `90`。我又完全重写
了一遍逐帧四步 CGLS 和三类指标，没有调用正式 compatibility helper；复算结果逐项
差 `0.0`，仍是 `1/5`。第一次独立脚本因为底层算子不接受 batch，在计算指标前就
停止，没有产生可用数字。

最扎眼的是四条失败轨迹的 observation harm 全是 `100%`。gradient 虽然都没有触发
harm，但它不能抵消数据一致性和 field 的材料性伤害。因此不能再说“rank96 已经装得
下，只差网络找系数”。p14 单点结论只对当时用全部 fit 选 basis 的开发条件成立。

结果后做的覆盖诊断显示，五折 basis 的 Jaccard 中位为 `0.8286`、并集仅 `115`
个 atom；失败不是 basis 每折乱跳。唯一通过的 p33 对 held-out K3 correction 的
rank96 能量捕获 p50 为 `85.47%`，四条失败轨迹只有 `68.28%–78.86%`。探索性
rank384 捕获升到 `88.22%–96.95%`，但这还不是 oracle 或模型成功。

所以按冻结规则立即停止：

```text
T1_training_target_generation_authorized=false
raw_rank96_coefficient_distillation_stopped=true
larger_MLP_rescue_authorized=false
algorithm_breakthrough=false
```

下一研究对象改为 coverage-adaptive / full-view detector dual proposal：保留精确
`A^T`、observable alpha 和 strict K1，但取消已被证伪的固定 rank96 瓶颈，并把
训练目标直接放在 K1 后的物理误差上。完整结果见
`docs/poolfire_c_dual_coefficient_attainability_v10_7_result_2026-07-27.md`。

## 245. full-view 目标真的存在，但普通线性图一条轨迹也没过

v10.8 把 rank96 瓶颈整个拿掉，不先训练神经网络，而是问两个更基础的问题：

1. 完整 detector-space `K3 dual certificate` 经 `A^T -> alpha -> restarted K1`
   后，能不能在五条 fit trajectory 上达到 K4 的兼容精度？
2. 如果能，最简单的 observation-only 映射能不能预测它？

第一问的答案是 **能**：full-K3 oracle 五条全部通过，joint harm 和 severe harm
都是零。第二问的答案是 **暂时不能**：

```text
Identity / Zero-K2 control: 0/5
six channel gains:          0/5
full DCT diagonal ridge:    0/5
nearest observation:        0/5
full K3 certificate oracle: 5/5
```

中间还抓到了一处自己的数学错误。首跑错误要求“从 `x_K3` restart 一步必须等于
continued K4”；但 restart 会清空原 Krylov 共轭方向，两者不应相等。首跑因此作废，
没有拿来改阈值。v10.8.1 删除这个错误等式，只保留原来的三类 compatibility 门；
v10.8.2 又按独立红队把 `z=y` 明确降为 Zero-K2 控制、补齐私有输入与参数绑定，并
更正离线最小调用账。第二套实现重算 25 行，数值和参数最大差都是 `0.0`。

最有价值的失败来自完整 2072 频率 DCT diagonal ridge：certificate relative-L2
p90 已经是 `0.1775–0.3342`，但四条轨迹 observation harm 仍达
`85.15%–100%`。这说明 raw target L2 不区分经过 `A^T` 和 K1 后危险的方向。

所以当前不是“网络已经成功”，而是把下一条模型的职责说清楚了：先跑允许跨 detector
与跨频率耦合的 full-view linear/reduced-rank operator；它仍失败后，才训练一个最小
detector CNN，而且 loss 必须直接约束 K1 后 observation non-harm 与
field/gradient，而不是只拟合 certificate MSE。

```text
full_view_target_viable=true
simple_cross_trajectory_candidate=false
independent_recomputation_max_difference=0.0
fresh_holdout_opened=false
algorithm_breakthrough=false
```

完整结果见
`docs/poolfire_c_dual_full_view_controls_v10_8_result_2026-07-27.md`。

## 246. 全视角线性 KRR 也没过：自由度更多，跨工况反而更差

v10.9 执行了 v10.8 留下的下一道门：不再让每个 DCT 频率各自缩放，而是允许完整
2072 维输入和输出任意线性耦合。每个 outer fold 用另外四条 trajectory 拟合，
内部再按完整 trajectory 留一选正则。同时保留一个可以偷看 outer 结果选 lambda 的
非部署 oracle，用来区分“选择器选错”与“这个有限模型族本身没有 headroom”。

结果三个臂全部 `0/5`：

```text
nested target-selected: 0/5
nested safety-selected: 0/5
outer lambda oracle:    0/5
```

最宽松 outer oracle 的 certificate p90 为 `0.4001–0.6084`，五条 joint match
都是 `0%`；P14/P22/P45/P58 的 observation harm 分别为 `91.09% / 86.14% /
100% / 100%`，P45 还有 15 个 severe frame。它甚至比 v10.8 更受约束的 DCT
diagonal target p90 `0.1775–0.3342` 更差。讲人话：只有 505 帧时，让所有 detector
坐标自由耦合并没有学到更通用的物理关系，反而丢掉了局部结构并跨工况过拟合。

第二套 NumPy 实现重写了 CGLS certificate、kernel ridge、`A^T -> alpha -> K1`
和三类指标；全部数值叶子最大差 `4.44e-16`。红队审计是 `P0=0 / P1=0`，没有发现
LOTO 泄漏。准确结论只限于“冻结九点、无截距、RMS 归一化 full-linear KRR 失败”，
不能夸成所有线性算子数学上不可能。

这条负结果满足了最小非线性 sentinel 的授权门。现在正在本机训练一个 77,020 参数、
奇对称、多视角 detector CNN；它仍输出 dual proposal，仍走 exact `A^T` 与原 K1，
loss 直接看 K1 后 field / gradient / observation deficiency。五折跑完前仍然是：

```text
algorithm_breakthrough=false
fresh_opened=false
test_opened=false
```

完整结果见
`docs/poolfire_c_dual_full_linear_krr_v10_9_result_2026-07-27.md`。

## 247. 不是随便换 CNN：K3 dual 天生 odd-homogeneous，但不是线性的

v10.9 失败后，我没有马上堆更大模型，而是先检查目标映射本身的对称性。CGLS 每一步
的 `alpha` 和 `beta` 都是二次型比值：输入 observation 乘 `a` 时，分子分母同时
乘 `a^2`，所以 K3 dual certificate 严格满足 `G(ay)=aG(y)`；但这些比值会随输入
的谱方向变化，所以通常不满足可加性。

冻结 PoolFire 算子上抽取 40 对 fit observation 后：

```text
scale homogeneity worst: 3.21e-16
odd symmetry worst:      0
non-additivity p50:      4.15%
non-additivity p90:      6.77%
non-additivity worst:    9.33%
```

这不是浮点噪声。它说明 full-linear KRR 与目标结构确实不完全匹配，也解释了 v11
为什么同时使用 RMS normalize/denormalize、odd symmetrization 和非线性多尺度
detector convolution。网络仍不能绕开物理算子：proposal 后必须 exact `A^T`，
再走原 alpha 与 K1。

完整推导见
`docs/poolfire_c_cgls_dual_homogeneity_note_v11_2026-07-27.md`。这个推导是设计依据，
不是算法突破；最终仍看五折和独立 checkpoint replay。

## 248. 第一次 5/5：最小非线性 dual CNN 在 fit-only 五折过门

这次不是“又写了一个网络”，而是真把 v10.9 留下的问题跑完了。

v10.9 的全线性 KRR 已经允许 2072 维 detector 坐标任意耦合，连偷看 outer 结果选
正则的 oracle 都是 `0/5`。随后我先证明 K3 dual target 对 observation 是奇对称、
尺度齐次但通常不可加，再按这个结构做了一个 77,020 参数的多视角 detector CNN。
它只输出 dual proposal，后面仍然必须经过精确 `A^T`、可观测 alpha 和原始 CGLS K1。

正式 v11.2 每折只用四条完整 trajectory 训练，第五条 101 帧全部留出。五折结果是：

```text
P14-S05: PASS, joint match 100%, harm 0%, severe 0
P22-S03: PASS, joint match 100%, harm 0%, severe 0
P33-S01: PASS, joint match 100%, harm 0%, severe 0
P45-S05: PASS, joint match 100%, harm 0%, severe 0
P58-S03: PASS, joint match 100%, harm 0%, severe 0
```

候选每帧实测是 `2A + 2A^T`，Zero-CGLS K4 是 `4A + 4A^T`。505 帧总账为
`1010 A + 1010 A^T`。独立 validator 重新做 NumPy 推理、物理链和全部指标，最大
科学数值差只有 `2.22e-16`。

**讲人话：**在这五条已经用于开发方向判断的公开代理轨迹里，网络用一半完整算子调用，
到达了 K4 的兼容精度范围，而且没有一帧触发材料性伤害。这是目前最强的正面结果。
它说明“保留 detector 局部结构的非线性 dual proposal”值得继续，旧的无结构线性路线
可以停止。

但我没有把它写成算法突破。v11.0 虽然也跑出 `5/5`，证据边界不够严；v11.1 在第三个
checkpoint 后被红队中止；只有重跑的 v11.2 关闭了 checkpoint-before-truth、
bytes-used 输入、坐标几何和独立源码绑定。更重要的是，这五条 trajectory 都属于
已经打开过的 fit pool，不是真正 fresh；墙钟、整管线内存和真实 BOST 也没测。

```text
fit_only_strong_candidate=true
fresh_generalization_proven=false
wall_time_speedup=false
whole_pipeline_rss_speedup=false
real_bost=false
algorithm_breakthrough=false
```

完整表格、为什么这样设计以及下一道 fresh 门见
`docs/poolfire_c_dual_detector_cnn_v11_result_2026-07-27.md`。

## 249. fresh 不是全胜：精度和调用过了，wall / RSS 没过

这次真正打开了锁模前选好的 `p45-s03`，而且没有只挑好看的部分汇报。

唯一 full-fit checkpoint 先用五条 fit trajectory 的 505 帧 observation-only 输入
训练，再由第二套实现独立复算。随后才冻结一次性 release、下载公开 PoolFire 文件、
只生成 observations，并让 Candidate 与 Zero-K4 各跑 17 个全新进程。truth 在
34 份输出、逐帧调用账和独立 replay 全部封口之后才打开一次。

结果的正面部分很扎实：

```text
101/101 joint matched
joint harm = 0
field / gradient / observation harm = 0
severe harm = 0
Candidate = 2A + 2A^T / frame
Zero-K4 = 4A + 4A^T / frame
```

但资源结果没有过：

```text
wall median: 1.1232 s vs 1.0562 s, 慢 6.35%
peak RSS p90: 343.82 MB vs 293.47 MB, 高 17.16%
```

独立 score replay 的数值差为 0。正式判决因此是
`PASS_FRESH_PROXY_ACCURACY_CALLS_RESOURCE_GATE_FAILED`，不是突破。

继续拆时间后发现，101 帧 CNN proposal 约占 0.116 s，两对物理算子只比四对省约
0.071 s。讲人话：网络确实少算了物理算子，但网络自己更贵，在这个便宜 straight-ray
CPU proxy 上得不偿失。

我没有回头用 fresh truth 调原模型，而是冻结一个 compact capacity ladder：
先跑 10,548 参数的 `w16d2` 完整五折；只有 5/5 才测 runtime，失败才运行 33,336
参数的 `w24d3`。两条 untouched test 继续封存。

完整结果见
`docs/poolfire_c_dual_detector_cnn_fresh_v11_3_result_2026-07-27.md`。

## 250. 参数缩小 7.30 倍，为什么还是没有真正加速

这轮没有继续加大网络，而是按事前容量阶梯只跑最小的 `w16d2`：

```text
77,020 params -> 10,548 params
fit-only LOTO = 5/5
每条 joint match = 100%
每条 harm = 0
每条 severe = 0
```

独立 validator 重新解析五个 checkpoint、重跑 505 帧物理链，最大科学差
`4.44e-16`。因为第一档已经 5/5，33,336 参数的第二档没有运行。

随后用五条 fit observation-only 输入训练唯一 full-fit checkpoint，耗时 112.22 秒；
原 77,020 参数版本约 290.86 秒。第二套 checkpoint 解析与 solver/metric replay 的差
为 `1.39e-17`。审计也纠正了措辞：模型类仍共享，所以状态写成
`PASS_INDEPENDENT_METRIC_REPLAY_WITH_SHARED_MODEL_V12_1`，不能冒充完全独立实现。

`p45-s03` 已经被 v11.3 烧掉，因此本轮明确只作 post-open development profile。
17 个 Candidate 和 17 个 Zero-K4 冷进程结果是：

```text
joint matched = 101/101
joint harm = 0
Candidate = 2A + 2A^T / frame
Zero-K4 = 4A + 4A^T / frame

wall: 1.0814 s vs 1.0845 s，只快 0.28%（门槛 10%，FAIL）
RSS p90: 353.71 MB vs 295.53 MB，高 19.69%（FAIL）
```

第一次 profile 在 34 个 worker 都跑完后，被只允许五条 fit 的 loader 拒绝，没有产生
判决。修复后新 run 把每条外层 wall 原子写入 progress，再用专门 pair validator
评分。第三套 validator 又独立核对 34 条 roster、3,434 条逐帧 receipt、全部统计和
compatibility，数值差为 0。

**讲人话：**网络变小是真的，训练变快也是真的，但这个 proxy 的物理算子太便宜，
Python/Torch 冷启动、模型初始化、几何加载、序列化和激活内存占了大头。删参数不能
自动消掉这些共同成本。batch 调小会省内存但增加 proposal 时间，batch 调大会略快但
扩大激活内存。

因此这不是算法突破。当前可守住的结论只有：

```text
compact_fit_loto_5_of_5=true
postopen_p45_compatibility_101_of_101=true
complete_operator_calls_reduced_50_percent=true
cold_process_wall_speedup=false
whole_worker_rss_benefit=false
untouched_test_opened=false
real_bost=false
algorithm_breakthrough=false
```

完整表格和边界见
`docs/poolfire_c_dual_detector_compact_v12_1_result_2026-07-27.md`。

## 251. 调用减半不是假的，关键在程序是不是常驻

v12.1 的冷进程对比只快 0.28%，但每次都要重新启动 Python、导入 Torch、加载模型、
几何和 observation。这轮不改模型、不改 batch、不改阈值，只把程序保持在内存里：

```text
5 个独立 session / arm
每个 session 先预热 1 条完整轨迹
随后计时 17 条 101 帧完整轨迹
每个 arm 共 85 个 measured pass
```

结果通过独立 validator 重算：

```text
Candidate = 112.46 ms / 101 frames
Zero-K4   = 128.74 ms / 101 frames
steady-state wall 快 12.64%（PASS）

Candidate RSS p90 = 398.13 MB
Zero-K4 RSS p90   = 344.06 MB
RSS 高 15.71%（FAIL）
```

为什么能快？Candidate 的 112.46 ms 可以拆成：

```text
compact model proposal = 49.06 ms
2A + 2A^T solver       = 63.11 ms
```

Zero-K4 的 `4A + 4A^T` solver 是 128.74 ms。少掉两对算子省约 65.63 ms，
足够覆盖 49.06 ms 的模型推理，还剩约 16.28 ms 净收益。五个 session 全部更快，
最弱一组也快 9.63%。

**讲人话：**之前“调用减半却没加速”不是调用账造假，而是每次启动程序都花掉太多
固定成本。连续处理很多帧时，这些固定成本只付一次，算法计算本身确实出现了速度
余量。

但现在仍不能叫突破：

- p45 已烧掉，只是 development，不是第二次 fresh；
- wall 只在常驻内核口径通过，冷启动没有通过；
- RSS 仍高 15.71%；
- 两条 untouched test 和真实 BOST 都没打开。

所以正式状态是：

```text
PASS_POSTOPEN_PERSISTENT_WALL_HEADROOM_RSS_FAILED
persistent_kernel_wall_headroom=true
algorithm_breakthrough=false
```

完整结果见
`docs/poolfire_c_dual_detector_compact_persistent_v12_2_result_2026-07-27.md`。

## 252. FP32 不是内存解法：五条都更快，RSS 仍只有 3/5

v12.2 已经证明常驻进程里模型成本小于省下的两对算子成本，但 RSS 高 15.71%。
这轮没有重训，也没有改 exact `A^T`、alpha 或 K1，只把 proposal 网络改为 FP32，
网络输出立即转回 float64。

结果是五条已经开放的 fit trajectory 各跑 5 个 session、每个 session 预热 1 条再
计时 17 条；每个臂合计 425 条完整 101 帧 pass：

```text
五条 joint match = 100%
五条 harm = 0
五条 wall reduction = 12.62% - 14.93%（全部通过）

RSS ratio:
p14 1.0042 PASS
p22 1.0433 PASS
p33 1.0696 FAIL
p45 1.0262 PASS
p58 1.0668 FAIL
```

FP32 与原 FP64 的 proposal worst 相对差只有 `2.91e-7`，最终场 worst 只有
`1.75e-7`，所以失败不是精度问题。独立 validator 自己重写 K1、K4、兼容性和资源
统计，candidate/reference 场与全部科学数字最大差都是 0。

红队随后发现一个必须公开的 P1：Zero-K4 worker 也提前导入了 Torch，因此 reference
RSS 被抬高，口径反而偏袒 Candidate。即使这样仍有两条超过 1.05，所以全局失败只会
更稳，不可能因为修正基线变成成功。另两个 P1 是 parity 合同只冻结 p90，以及 FP32
本来就是受已烧掉 p45 资源结果启发的 post-open development；实际 worst 远低于门，
且本轮没有运行 p45 replay。

所以没有临时放宽门槛，也没有拿 3/5 写成功：

```text
FAIL_FIT_ONLY_FP32_PROPOSAL_ALL_GATES
postopen_p45_replay_authorized=false
algorithm_breakthrough=false
```

讲人话：FP32 保住了精度和 13%-15% 常驻速度优势，但不能稳定消掉运行时内存。下一
个有效实验必须让 Zero-K4 真正走无 Torch 的纯物理基线，并让候选使用轻量推理后端；
继续围绕同一污染口径微调 batch 或精度没有论文价值。

完整结果见
`docs/poolfire_c_dual_detector_compact_mixed_precision_v12_3_result_2026-07-27.md`。

## 253. 干净速度是真的，但“更快且不增内存”还没有成立

上一轮发现 Zero-K4 被 Torch 污染后，这轮没有继续在旧口径上做文章，而是把
Candidate 和 reference 拆成真正独立的新进程。每种实现仍然跑 5 个 session，
每个 session 预热 1 条，再计时 17 条完整 101 帧轨迹；每臂一共 85 次。

先试 MLX GPU：

```text
Candidate = 74.00 ms
Zero-K4   = 131.12 ms
wall 快 43.56%             PASS
RSS ratio = 1.3367         FAIL
```

MLX 很快，但框架内存太高。于是我把同一个 10,548 参数网络写成最小原生 C 前向，
不加载 Torch 或 MLX，网络输出后仍走完全相同的 float64 `A^T + alpha + K1`：

```text
native batch 16:
Candidate = 109.96 ms
Zero-K4   = 130.07 ms
wall 快 15.46%             PASS
RSS ratio = 1.0745         FAIL

native batch 8:
Candidate = 110.94 ms
Zero-K4   = 130.94 ms
wall 快 15.27%             PASS
RSS ratio = 1.1395         FAIL
```

三轮全部是 101/101 joint match、0 harm、0 severe，调用账也一直是 Candidate
`2A+2A^T` 对 Zero-K4 `4A+4A^T`。原生 C 与冻结 Torch 的 proposal / 最终场
worst relative-L2 只有约 `3.19e-7 / 2.04e-7`，所以失败不是模型精度，也不是
调用账造假。

v12.5 和 v12.6 都由独立只读代理重新汇总 85 次计时、5 个 session、10 个 worker、
逐帧 receipt、框架隔离和 RSS。v12.5 只差约 3.59 MiB 就到 1.05 门，但 5 个
session 中仍有 4 个失败；v12.6 减小 batch 后 RSS 反而更差。因此不能继续在同一条
已经看过结果的 p14 上试 batch 4、线程数或编译参数，直到碰巧过门。

失败后又用 5 组全新进程做了“1 次预热 + 1 次测量”的 post-hoc 诊断。Candidate /
Zero-K4 RSS p90 是 `143.15 / 127.09 MB`，ratio 仍为 `1.1263`。所以问题不只是
17 次循环把 allocator 高水位越堆越高；Candidate 本身仍有结构性内存开销。这个诊断
不参与正式判决，但排除了“把循环写得更漂亮就足够”的简单解释。

**讲人话：**这个模型确实能用一次神经预测换掉两对昂贵的物理算子，所以常驻处理
101 帧时稳定更快；但 Candidate 进程仍需要额外内存。我们现在有“调用减半”和
“单条开发轨迹干净加速”的扎实证据，没有“同精度、更快、内存不增”的完整证据。

正式判决：

```text
FAIL_CLEAN_RUNTIME_DEPLOYMENT_RESOURCE_GATE
remaining_fit_expansion_authorized=false
fresh_or_validation_or_test_opened=false
algorithm_breakthrough=false
paper_success=false
```

完整解释与统一图见
`docs/poolfire_c_dual_detector_clean_native_v12_4_to_v12_6_result_2026-07-27.md`。

## 254. v13 把时间问题解决了，但 68 kB 也不能假装过门

v12.5 的单次 fresh-process 诊断说明 Candidate 至少还要少约 9.70 MB，才可能达到
`RSS ratio <= 1.05`。所以这轮没有再换 batch 或编译参数，而是把运行方式真正改成
流式：

- 观测转 FP32、RMS、正负奇对称和恢复尺度全部放进 C；
- 四个原生线程常驻，每个线程处理完整样本；
- 每个线程只留三份单样本 feature scratch，不留 full-batch feature；
- 下一块 8 帧 proposal 与当前块的 `A^T + alpha + K1` 重叠；
- Candidate 和 Zero-K4 都逐帧写 NPY，不把整条字段留在 RAM。

先用临时随机权重把 15-worker 队列跑通。串行版是内存通过、wall 失败；Python
预取版是 wall 通过、RSS 多约 0.53 MB；把协调线程下沉到 64 KiB C 线程后，仍差约
0.51 MB。一次“卷积里现算 SiLU”的省内存尝试把 wall 拖慢到 0.41 s，立即丢弃。
最后改成样本并行 scratch，同一份卷积算术不变，临时权重演练才同时过门。

这些只是正式前工程排错，不算科学证据。锁定提交后，正式合成预检才加载冻结训练
checkpoint；仍然只生成 seed 固定的 101×2072 合成观测，没有读任何 fit observation、
field truth、fresh、validation 或 test。

正式数值：

```text
proposal relative-L2 p90/worst = 1.381e-7 / 1.427e-7
field relative-L2 p90/worst    = 7.668e-8 / 8.429e-8

v13 Dual-K1 calls = 202A + 202A^T
Zero-K4 calls     = 404A + 404A^T

v13 wall median      = 0.065039 s
Zero-K4 wall median  = 0.113459 s
wall reduction       = 42.676% PASS

v12.5 RSS p90        = 82,984,960 bytes
v13 RSS p90          = 60,571,648 bytes
Zero-K4 RSS p90      = 57,622,528 bytes
v13 / Zero-K4        = 1.05117998 FAIL
frozen cap           = 1.05000000
excess               = 67,993.6 bytes
```

五个配对 session 里 v13 全部更快，最弱也快 39.36%；相对 v12.5，RSS 已少
22.41 MB。可是预注册门不是“差不多”，多 68 kB 也仍然失败。没有重跑到运气好，
没有把门改成 1.052，也没有打开 p14 或其他数据救结果。

独立只读审计没有发现数据、算术或调用账错误，但指出一个必须讲清楚的 P1：validator
用的是 `method="higher"`，5 个 session 时 p90 就等于最大值；协议正文没有显式写
插值方法。事后算 linear-p90 会得到 ratio `1.048859`，但这不能用来改判。正确说法
是“按冻结 validator 严格失败，RSS 接近阈值且统计稳健性有限”，不是“方法已被证明
内存更差”。同样，42.68% 只覆盖预热后的 proposal + solver + 输出写入，不含启动、
模型加载和 native context 创建。

**讲人话：**我们证明了网络推理与物理迭代可以真正并行，调用减半可以转成 42.7%
的 wall 优势；还把旧运行时的额外内存基本压掉了。但在同样流式的公平参考面前，
Candidate 仍比允许值高一点点，所以完整的“同调用、更快、内存不增”主张还没成立。

这条后端支线到此冻结。继续为 68 kB 调 allocator 不会让论文更有科学价值，下一轮
只做能改变跨 trajectory compatibility、harm 或真实迁移结论的模型工作。

```text
FAIL_SYNTHETIC_PREFIT_FUSED_STREAMING_V13
formal_wall_gate_passed=true
formal_peak_RSS_gate_passed=false
RSS_statistical_robustness_limited=true
fit_execution_authorized=false
algorithm_breakthrough=false
paper_success=false
```

完整结果与图见
`docs/poolfire_c_dual_detector_fused_streaming_v13_result_2026-07-28.md`。

## 255. v14：2,912 参数不是免费午餐，P45 只差一帧仍然 FAIL

v13 已把调用减半变成 42.68% 的稳态 wall 优势，但严格 RSS 门仍差约 68 kB。
这轮没有继续调 allocator，而是只冻结一个更小候选 `w8d2`：

```text
width = 8
dilations = (1, 2)
parameters = 2,912
epochs = 120
extra seeds = 0
```

它比已经通过五折的 `w16d2` 少 72.39% 参数，但 packing、奇对称、全局 context、
post-K1 loss、exact A^T、alpha 和 strict K1 都保持不变。

五条完整 trajectory LOTO 真正跑完后：

```text
P14-S05  101/101 matched  PASS
P22-S03  101/101 matched  PASS
P33-S01  101/101 matched  PASS
P45-S05   90/101 matched  FAIL
P58-S03  101/101 matched  PASS
```

P45 的门要求至少 91/101，实际只有 90/101。11 个失败帧全部只是 observation
超界，field failure=0、gradient failure=0、harm=0、severe=0；最坏 observation
margin 只有 0.002712。讲人话：模型没有崩，只是压得太小以后，三个时间片段的
测量一致性少了一点余量。

同样 11 帧上，`w16d2` 全部仍在门内，最靠近门的 margin 也是 -0.003525。
因此这是可重复的容量下界信号，不是把阈值画在哪里都一样。

第二套 NumPy validator 重新加载五个 checkpoint，重算 505 帧
`A^T -> alpha -> K1`、三类指标和 `1010A + 1010A^T` 调用账；最大科学数值差为
`3.33e-16`。fresh、historical validation 和两条 test 都没有打开。

红队还抓到一个容易误读的旧标签：继承 v11 的 report 把“4/5 且无 harm”叫
`PASS_FIT_LOTO_DETECTOR_CNN_SENTINEL`。v14 的结果前合同只接受 5/5，所以这个旧
标签不是权威结论。权威 gate 与最终独立验证都是：

```text
FAIL_INDEPENDENT_STRICT_W8D2_CAPACITY_GATE_V14
full_fit_authorized=false
synthetic_resource_gate_authorized=false
algorithm_breakthrough=false
paper_success=false
```

所以没有训练 full-fit、没有跑资源门，也没有追加 width/seed 救结果。当前结论是：
`w16d2` 仍是最小的五轨迹 5/5 候选；继续纯粹删宽度已经越过稳健容量下限。完整结果
见 `docs/poolfire_c_dual_detector_w8d2_v14_result_2026-07-28.md`。

## 256. v15.1：P45 的残差特征不是一个跨工况通用修补点

v14 已经知道 `w8d2` 在 P45 少一帧过门，而且 11 个 miss 全部来自 observation。
这轮没有立刻再造网络，而是先把最终 residual：

```text
r = A x_K1 - y
```

拆成三视角、六分量和 low/mid/high 共 18 个模式，比较冻结的 `w8d2` 与
`w16d2`。五条 fit trajectory 共 505 帧，raw pair truth 没有请求；但为了绑定旧
模型身份读取了 truth-derived 历史报告，所以不能夸成 filesystem-wide truth-free。

第一次预运行被红队判无效：零能量并列也会拿到 top-3 票，跨轨迹一致和 top-3
还可能来自不同轨迹，负结果状态甚至以 `PASS` 开头。源码、旧 checkpoint 和状态
语义全部修完，协议升级到 v15.1 后才重新运行。

权威结果把两个看起来相似的现象拆开了：

```text
view_1_component_1_mid:
  5/5 轨迹 median 都变差
  3 条同轨迹 support
  0 strong reversal
  但 P45 fail w8/w16 = 1.019
  P45 fail/matched   = 0.940
  -> 跨轨迹容量缺口，不解释 P45 miss

view_1_component_0_low:
  P45 fail w8/w16 = 1.322
  P45 fail/matched = 1.526
  P45 failure top-3 = true
  但同轨迹 support 只有 1，P22 强反转
  -> 能解释 P45，但不能跨工况复用
```

第三视角两个低频模式也分别在 P22 或 P58 反转，或者没过 P45 富集门。最终 18 个
模式全部失败：

```text
VALIDATED_NEGATIVE_NO_SHARED_MODE_V15_1
shared_mode_ids=[]
single_correction_preregistration_authorized=false
algorithm_breakthrough=false
```

独立 validator 从十个 checkpoint 重新生成两套 505 帧结果，重写 FFT 和 gate：

```text
maximum array difference   = 0
maximum summary difference = 0
maximum Parseval error     = 8.33e-17
```

红队复审为 `P0=0 / P1=0 / P2=1`，当前定向测试 `8 passed`。剩余 P2 是未来还可
增加更多 seal/anchor mutation 回归测试，不改变本轮负判。

**讲人话：**小模型确实有一个比较普遍的中频短板，但 P45 真正出问题的是另一个
低频特征；这个特征到了别的工况甚至方向相反。此时训练“通用修补器”很可能只是把
P45 特例背下来。我们因此停止 proxy-only 架构救援，保留 `w16d2`，下一项论文级
证据必须来自独立轨迹或真实 BOST 迁移，而不是继续在这五条数据上调网络。

完整结果见
`docs/poolfire_c_observation_residual_v15_1_result_2026-07-28.md`。

## 257. v16/v16.1：调用真的减半了，但当前实现还不是低成本加速

这次终于不再在五条 fit trajectory 里循环解释模型，而是拿一条此前未参与拟合的
公开 PoolFire development trajectory 做完整 101 帧评分。冻结的 `w16d2 Dual-K1`
先过了最重要的精度与伤害门：

```text
joint match = 101 / 101
joint harm = 0
candidate = 202 A + 202 A^T
Zero-K4   = 404 A + 404 A^T
```

这说明 warm start 的核心作用不是假的：在这条额外公开轨迹上，它用一半完整算子对
到达了冻结 compatibility envelope。native streaming 实现相对正式 candidate 的
field relative-L2 p90/worst 只有 `9.24e-8 / 1.04e-7`，reference 则逐值一致。

但我随后把最容易夸大的地方单独做成 v16.1 资源门。两个 arm 各跑 18 个 fresh
process，交替先后、不做 warmup，parent 在 child 退出后用 `wait4` 记录 wall、RSS
和 CPU。结果出现了一个很清楚的分裂：

```text
典型 wall:
  candidate median = 0.201 s
  Zero-K4 median   = 0.258 s
  paired median reduction = 22.02%
  candidate faster = 16 / 18

稳定性:
  first candidate = 1.165 s
  paired reference = 0.259 s
  worst harm = 350.30%
  allowed = 5%

CPU:
  paired ratio median = 1.231
  -> candidate 多用 23.13% 总 CPU

RSS:
  p90 ratio = 1.0581
  allowed = 1.05
```

所以典型 wall 变快是真的，但它依赖更多并发 CPU，而且第一次加载 checkpoint、
native library 和 context 时出现了很重的冷启动。candidate peak RSS 也仍略高。
三门全部失败：

```text
FAIL_INDEPENDENT_PUBLIC_DEVELOPMENT_RESOURCE_GATE_V16_1
wall_gate_passed=false
peak_RSS_gate_passed=false
CPU_time_gate_passed=false
algorithm_breakthrough=false
```

独立审计复核了 36 条 records、36 份 worker report、`36×101` 条逐帧 receipt 和
全部配对统计，结论 `P0=0 / P1=0`。即使事后忽略第一个冷启动，CPU 与 RSS 仍然
失败，所以不能把它解释成“只有一个倒霉异常值”。

**讲人话：**我们现在确实有一辆少踩两次油门也能到终点的车，但发动时更费劲，
运行时还调用了更多工人，占的空间也略大。它证明了算法调用数方向有价值，却还没
证明部署成本更低。下一版必须在新的轨迹结果产生前先冻结更轻的运行机制，同时压低
冷启动、CPU 和 RSS；不能回到这条已经看过结果的数据上调到过门。

完整结果见
`docs/poolfire_c_independent_resource_v16_1_result_2026-07-28.md`。

## 258. v17：单持久线程真的修好了内存，但还没有修好完整成本

上一轮不是笼统地“性能不好”，而是三个可定位的问题：四份网络 scratch、四个持久
工作线程、每批再临时创建一个协调线程。于是这次我没有改网络、checkpoint、物理算子
或 CGLS，只改了运行时：

```text
旧实现：main + 4 persistent workers + per-batch coordinator
v17：   main + 1 persistent proposal worker
```

`begin()` 只把下一批观测交给这个持久线程，主线程继续处理当前批的 `A^T`、`A` 和
K1；`finish()` 等待结果，不再每批创建和回收线程。网络逐样本算术顺序不变，四份
feature scratch 也降成一份。

结果前把批大小 8、18×2 fresh process、第一次运行必须保留、同一 CPU/RSS/wall 门
全部锁死。数值检查先确认它没有偷换算法：

```text
candidate vs v16 field rel-L2 p90/worst = 9.24e-8 / 1.04e-7
reference vs v16 = 0 / 0
candidate ledger = 202 A + 202 A^T
reference ledger = 404 A + 404 A^T
```

资源结果比 v16.1 明显进了一步：

```text
wall median:
  candidate = 0.2280 s
  Zero-K4   = 0.2638 s
  paired reduction = 11.06%        PASS
  faster pairs = 16 / 18           PASS

peak RSS p90 ratio = 0.99488        PASS

CPU median ratio = 1.11277          FAIL
first-pair wall harm = 236.82%      FAIL
```

也就是说，单持久线程把旧的 RSS ratio `1.05805` 真正修到了 `0.99488`，典型 wall
也第一次和 RSS 同时过门；但网络推理仍让总 CPU 多 `11.28%`，第一次冷加载仍然远超
5% no-harm 门。独立 validator 重算全部 36 条 fresh-process records 后给出的正式
判决是：

```text
PASS_INDEPENDENT_RECOMPUTATION_V17
FAIL_POST_OPEN_PERSISTENT_SERIAL_RESOURCE_GATE_V17
algorithm_breakthrough=false
```

**讲人话：**这次不是白做。我们已经证明“内存高”主要是旧线程结构造成的，并把它
修掉了；现在真正剩下的是网络本身多做的 CPU 计算，以及第一次加载模型和动态库的
长尾。因此继续调线程、allocator 或 batch 已经不值。下一项值得验证的机制是只在
观测发生重要变化的关键帧运行网络，其他帧复用或传输 dual proposal；但一级来源红队
也提醒，关键帧、时间复用和 warm start 各自都不是新原语，必须胜过 previous-dual
hold、运动传输与 Krylov recycling，才能形成论文贡献。

完整结果见
`docs/poolfire_c_persistent_serial_v17_result_2026-07-28.md`。

## 259. v18：时间复用守住了精度，却几乎没有真正复用

v17 剩下的是 CNN 的 CPU 和冷启动，于是这一轮不再调线程，而是问一个更直接的
问题：相邻帧能不能少跑几次 CNN？

我先在五条 fit trajectory 上做 trajectory-level leave-one-out。门只看当前观测与
上一关键帧观测；不触发 CNN 时，用 fit-only 对角增益把上一 dual proposal 搬到当前
帧。每帧后面的 exact `A^T`、alpha 和 CGLS K1 都不变。

五折表面上全过了 compatibility，但细看计划执行率：

```text
p14-s05: 51 / 101 CNN
p22-s03: 72 / 101 CNN
p33-s01: 51 / 101 CNN
p45-s05: 101 / 101 CNN
p58-s03: 101 / 101 CNN
```

后两条变化快的轨迹没有学会“安全复用”，而是每帧都回退到 full CNN。五折平均计划
执行率为 74.46%，不能只写“五折精度通过”。

把 fit-only 阈值原样拿到已经消费过的 p33，结果更清楚：

```text
event gate:
  scheduled CNN = 94 / 101
  planned reduction = 6.93%
  required reduction = 20%
  joint match = 100%
  harm = 0

fixed stride-2:
  scheduled CNN = 51 / 101
  joint match = 71.29%
  compatibility = FAIL
```

所以温和门的精度来自“几乎总跑 CNN”，强制少跑又会失去精度。独立 validator
重写 calibration、transport、CGLS 与统计后，五组最大差都是 0：

```text
PASS_INDEPENDENT_RECOMPUTATION_V18
FAIL_POST_OPEN_TEMPORAL_AMORTIZATION_GATE_V18
algorithm_breakthrough=false
```

还有一个必须说清的工程边界：为了同批比较多个反事实 control，v18 先生成了完整
101 帧 proposal，再按 mask 替换非关键帧。因此 `94/101` 是计划执行数，不是 native
入口已经少跑了 7 次；本轮也没有做 wall/CPU/RSS 计时。正式结果既然已经失败，就
没有理由再扩建运行时。

我还做了一个不升级证据等级的 previous-field recycling 探索。只在第一帧跑 CNN，
以后每帧用缓存场做一次 `A/A^T` 校正，p33 joint match 只有约 0.99%；每两帧 reset
一次可在 p33 达到 94.06%，但 held-out p58 只有 84.16%。也就是说，简单时间连续
假设在当前 PoolFire 快变轨迹上不够稳，不能靠调一个阈值包装成泛化结果。

**讲人话：**流场确实连续，但这批数据每 0.02 s 的变化并不“小”。我们的门为了不
犯错，只好几乎每次都叫 CNN；强行省计算就会漏掉重要变化。这项负结果帮我们关掉了
“上一帧直接搬过来”这条看起来便宜、实际上不稳的路。

完整结果见
`docs/poolfire_c_temporal_dual_v18_result_2026-07-28.md`。

## 260. v19：我让“答案可见的运动诊断”先试，平移路线仍然失败

v18 失败后，一个自然解释是：上一帧不能直接 hold，但如果先估计流动位移，再把
dual proposal 平移到当前帧，也许能复用。为了不先花几天训练 optical-flow 网络，
这一轮先问更便宜、更有判别力的问题：

> 即使诊断器能看见当前完整 proposal，它能否用每个 view 的小范围整数平移与缩放，
> 从上一关键帧恢复当前 proposal，并守住最终重建精度？

如果这个“答案可见”的固定家族自己都失败，就没有理由立刻训练一个只能看观测的
更难模型。

正式运行前，独立审计指出了四个问题：执行对象没有完整绑定、51 个精确关键帧会
稀释 50 个跳过帧、oracle 被误记成 51 次 CNN、以及“上界”措辞过强。我先全部
修复，再冻结 private execution release。正式表同时检查全部 101 帧和仅 50 个
skipped frames；oracle 正确记为 101 次 CNN，只是非部署 diagnostic。

最关键的 skipped-only joint match：

```text
trajectory   observation motion   proposal-visible diagnostic
p14-s05             100%                    100%
p22-s03              70%                     74%
p33-s01             100%                    100%
p45-s05              40%                     40%
p58-s03              16%                     14%
required              90%                     90%
```

p22、p45、p58 三条都失败。更关键的是，proposal-visible diagnostic 在 p14、
p22、p33、p45 的 50 个跳过帧中从未选择非零位移，p58 也只有 2%。也就是说，
即使直接看当前 proposal，最佳 proposal-L2 参数也主要是幅值缩放，不是平移。

还有一个反直觉结果：2072 维 diagonal delta 的 proposal p90 比 motion hold
低很多，但 p22、p45、p58 的最终 skipped joint match 仍只有 58%、40%、8%。
因此“dual MSE 变小”与“最终 field / gradient / observation 非劣”不是一回事。
下一模型不能只优化 proposal MSE。

独立 validator 没有导入正式 motion/temporal helper，重新实现所有 transport、
exact `A^T`、alpha、CGLS K1、Zero-K4 与双重精度门。最大差为：

```text
1.1102230246251565e-16
```

最终状态：

```text
PASS_INDEPENDENT_RECOMPUTATION_V19
PASS_FINAL_EVIDENCE_SEAL_V19
FAIL_BOUNDED_PER_VIEW_PROPOSAL_L2_WARP_DIAGNOSTIC_V19
algorithm_breakthrough=false
```

**讲人话：**这次我们真的让“知道当前答案的裁判”先试了一遍。它在三个工况上仍然
搬不准，说明这里不是把上一帧左右挪两格就能解决。反应、扩散、热膨胀、视线积分和
火焰形态变化都可能改变 dual；观测里看起来像移动，也不一定对应 dual 在移动。

这次没有算法突破，但得到了一条扎实的停止结论：不再调位移半径、缩放、阈值，也不
直接训练 optical-flow/FNO/GRU。下一步先做 dual innovation 的低维子空间
diagnostic。只有 oracle 低秩系数能在五条 held-out trajectory 的 50 个 skipped
frames 全部过门，才训练 observation-only 系数预测器；否则当前 proxy 上的 50%
时间摊销路线停止。

完整结果见
`docs/poolfire_c_motion_state_v19_result_2026-07-28.md`。

## 261. v20：共享线性 lift 子空间在高功率工况仍过不了门

v19 证明小范围平移不够以后，我没有直接训练 GRU/FNO，而是先让一个更强的
非部署 oracle 试：把完整 CNN proposal 做精确 `A^T` lift，再用另外四条
trajectory 的 lift innovation 学共享低秩子空间。留出轨迹的系数可以直接看见
真实 lift innovation；如果这种“答案可见”系数都失败，观测网络只会更难。

这次先修完两个会让证据作废的问题再正式跑：Stage 1 不能在 seal 前哈希读取
truth 文件，Stage 2 必须重新只读加载封存候选。独立预检改用
QR + small SVD，不复制 formal SVD；rank 边界还要过谱隙门。

Stage 1 独立结果：

```text
candidate max difference = 4.4964e-14
span max difference = 1.2669e-15
rank-zero controls = exact
truth bytes read before receipt = false
```

正式五折结果的 skipped-only joint match：

```text
trajectory   smallest passing rank   rank-192 joint
p14-s05               0                  100%
p22-s03              32                  100%
p33-s01               0                  100%
p45-s05              none                 66%
p58-s03              none                 72%
required                                   90%
```

p45 和 p58 随 rank 增加在改善，但 `192` 与各 fold 的完整有效 span 都没有
过门。它们的 harm 和 severe harm 都是 0，所以不是重建崩溃，而是很多跳过帧
以小幅偏差越过严格非劣阈值。独立全量复算的 metric 最大差为
`1.2669e-15`，正式判决：

```text
PASS_INDEPENDENT_RECOMPUTATION_V20
FAIL_PREREGISTERED_LIFT_SUBSPACE_RANKS_V20
algorithm_breakthrough=false
```

**讲人话：**低功率和部分中功率工况很好压缩，但 p45/p58 的时间变化不在其他
四条轨迹学到的共享线性方向里。继续把 rank 从 192 调到 193 没意义。不过这次
oracle 用的是 lift-L2 系数，v19 已证明 L2 小不等于最终重建过门。因此下一次不先
换大网络，而是在同一 span 内按 `A h` 的观测空间误差选 oracle 系数：若能修复
p45/p58，说明训练目标错了；若仍失败，才转向工况条件化或非线性表示。

完整结果见
`docs/poolfire_c_temporal_lift_v20_result_2026-07-28.md`。

## 262. v21：把系数目标换到观测空间，p45/p58 仍然没有过门

v20 留下了一个必须先回答的小问题：共享线性 lift span 失败，到底是 span
不够，还是 lift-L2 这个系数目标与最终观测不一致？

这一轮没有换 keyframe、basis、rank、solver 或评分门。每条留出 trajectory
仍有 51 个 exact keyframes 和 50 个 skipped frames，span 仍只由另外四条
trajectory 构成。唯一变化是 oracle 系数从三维 lift 投影改为：

```text
argmin_c ||A(h_base + Uc) - A h_exact||_2
```

为了公平，同批还逐 rank 重放了 v20 lift-L2 control。独立验证重新计算候选、
最终场和所有指标，得到：

```text
PASS_INDEPENDENT_RECOMPUTATION_V21
candidate max difference = 4.496403249731884e-14
metric max difference = 3.219646771412954e-15
sealed inputs unchanged = true
```

固定 primary centered family、rank 192 的 skipped-only joint match 是：

```text
trajectory   A-space   paired lift-L2   required
p14-s05       100%          100%           90%
p22-s03       100%          100%           90%
p33-s01       100%          100%           90%
p45-s05        66%           66%           90%
p58-s03        70%           72%           90%
```

p14、p22、p33 存在通过的数值 rank；p45 的最佳结果仍是 66%，p58 的最佳
结果仍是配对 lift-L2 的 72%。p45/p58 的 joint harm 都是 0，所以不是算法
崩溃，而是大量帧离严格非劣门差一点。没有任何公共数值 rank 能让五条轨迹
逐条通过，正式判决为：

```text
FAIL_PREREGISTERED_A_SPACE_LIFT_RANKS_V21
algorithm_breakthrough=false
```

**讲人话：**我们给了 oracle 一个更懂相机观测的目标，但它在四条轨迹上没有
改变，在 p58 还从 72% 变成 70%。这说明 v20 的问题不只是“loss 写错了”；
当前跨工况共享的一套线性变化方向，确实装不下 p45/p58 的时间演化。

本轮为了算 A-space oracle 额外用了 2500 次 `A`，这是诊断成本。它不能被藏进
未来的 `202A+152A^T` 反事实部署账，也不能写成 native CNN skip、wall 或内存
加速。fresh、untouched test、真实 BOST 和端到端泛化都没有打开。

现在应当停止的，是同一个全局线性 span 上继续堆 rank、换相似 loss 或直接加大
FNO/GRU。没有被否定的，是结果前冻结的工况条件化/局部字典、非线性或
history-aware 表示，以及更保守的跳帧预算与 deployment-visible fallback。
这些是下一轮可检验假设，不是 v21 已经证明的优势。

完整结果见
`docs/poolfire_c_temporal_lift_measurement_v21_result_2026-07-28.md`。

## 263. v24：三条公开 CFD 外部代理复现通过

这次没有继续在已经打开的轨迹上调模型。我先把方法、rank、ridge、K1、兼容门、
成本账和三条轨迹顺序一起冻结，再去获取三条此前不在项目数据根目录的公开
PoolFire trajectory：

```text
p=33kw_size=05
p=45kw_size=01
p=58kw_size=05
```

它们是新的功率/尺寸组合，但功率值和尺寸值都在 fit 数据出现过，所以不能叫
unseen-power、unseen-size 或 geometry OOD。三条全部完成预测后才统一评分，
并且规定一条失败就整体失败，不能靠平均值冲过去。

固定 Reduced Warm K1 的在线成本是：

```text
Primary:    202A + 152A^T = 354
Full parent:202A + 202A^T = 404
Zero-K4:    404A + 404A^T = 808
```

正式结果：

```text
trajectory   all match   skipped match   harm   wall reduction
p33-s05       101/101        50/50          0       13.48%
p45-s01       101/101        50/50          0       11.81%
p58-s05        99/101        48/50          0       12.80%
```

p58 的两处 miss 只在 observation 指标，frame 77 和 79 超出严格 match 线
`0.000846 / 0.004517`；field 与 gradient 仍匹配，也没有越过 harm 线。
把它们公开保留下来，是为了不让“3/3 通过”把最靠近失败的位置藏掉。

不同数值路径重新生成 package、候选、K1、Zero-K4、指标和 benchmark：

```text
candidate max difference <= 2.20e-15
metric max difference <= 2.22e-16
benchmark statistic difference = 0
PASS_INDEPENDENT_EXTERNAL_HOLDOUT_RECOMPUTATION_V24
PASS_EXTERNAL_PROXY_REPLICATION
```

**讲人话：**我们现在不再只有“在开发数据上看起来快”。一个固定的方法换到三条
新组合的公开 CFD 轨迹后，精度门没有塌，三条都少算 50 次精确 `A^T`，而且实测
wall 都快了 10% 以上。这个信号比 v23 扎实得多。

但它仍不是算法突破。有效 official untouched test 是 0，数据仍来自同一个
PoolFire 数据集和同一个 straight-ray proxy，真实相机、标定误差、flow-off
重复测量和组内 forward 都没有进入。rank-199 package 还有
`200A+200A^T=400` 的离线成本；按等成本要复用 8 条 101 帧序列才摊平，不能藏。

下一步不再无边界增加同类 CFD 工况。真正能改变论文结论的是把冻结方法接到组内
真实 BOST，用重复采集与标定不确定度定义“同精度”，再量完整 A/A^T、端到端
wall 和全流程内存。

当前判决：

```text
important_proxy_replication_milestone=true
algorithm_breakthrough=false
real_BOST=false
paper_success=false
```

完整结果见
`docs/poolfire_c_observable_external_v24_result_2026-07-28.md`。

## 264. v25：曲折光线物理压力没有击穿固定 Warm K1

v24 仍有一个很大的解释漏洞：训练、观测和反演都使用直线射线，方法可能只是
吃到了 forward 完全一致的便宜。为直接检验这个漏洞，我没有重训模型，也没有
调整兼容门，而是把观测正演换成场依赖 eikonal 曲折光线：

```text
dr/ds=t
dt/ds=(I-tt^T)grad(n)/n
n=1+beta*(rho-mean(rho))
```

师兄提供的 BOS notebook 只用于确认物理方程与真实工作流背景；正式代码完全
重新实现，私有工具、路径和数据没有进入公开仓库。

协议在结果前固定三条 post-open fit-morphology 轨迹、每条 101 帧、五个 beta、
192 步正式积分、96 步收敛对照和八个方法。最高 `beta=0.002` 时，曲率相对
线性极限的 observation p90 变化为：

```text
p14-s05   3.113%   convergence worst 0.080%
p33-s01  14.188%   convergence worst 0.372%
p58-s03   5.705%   convergence worst 0.187%
```

p33 的最坏观测变化达到 17.707%，所以这不是接近零的装饰性扰动；三条 96/192
步差又都低于冻结的 0.5% 数值门。

最高曲率档上的共同结果：

```text
Normalized BP          202 calls  FAIL
Zero CGLS K1           202 calls  FAIL
Reduced Warm K1        354 calls  PASS
Full parent Warm K1    404 calls  PASS
Zero CGLS K2           404 calls  FAIL
Geometry PCGLS K2      404 calls  FAIL
Zero CGLS K3           606 calls  FAIL
Zero CGLS K4           808 calls  PASS
```

主方法三条全 101 帧和奇数 50 帧都是 100% joint match、0 harm。独立判决器
重新计算 `3×5×8=120` 个方法判决、调用账、最便宜兼容臂、严格支配关系和最高
通过 beta，状态为：

```text
PASS_INDEPENDENT_DECISION_RECOMPUTATION_CURVED_RAY_STRESS_V25
PASS_CONTROLLED_CURVED_RAY_PROXY_STRESS_V25
```

**讲人话：**这次确实排除了一个重要失败解释：warm start 的优势没有在光路从
直线改成场依赖曲线后立刻消失，而且 p33 的 forward mismatch 已经很明显。
不过 beta 仍是 synthetic 强度，没有真实 `rho->n`、相机、背景图、位移提取、
噪声与重复测量；三条轨迹也都是已打开的同一 PoolFire 数据集。因此这是积极的
物理鲁棒性增量，不是算法突破。

```text
algorithm_breakthrough=false
real_BOST=false
paper_success=false
```

完整结果见
`docs/poolfire_c_curved_ray_v25_result_2026-07-28.md`。

## 265. v26.3：外部组合和曲折 forward 同时出现，固定 Warm K1 仍通过

v25 的正结果还有一个缺口：使用的三条轨迹已经参与过 fit-morphology 开发。
这次没有再训练模型，而是把 v24 的三条 external-to-fit 组合直接接到同一套
field-dependent eikonal 曲折光线 forward：

```text
p33-s05
p45-s01
p58-s05
```

实验并没有一次就放行。v26.1 因“曲率压力帧”和“兼容性评分帧”未绑定，在 truth
解码前停止。v26.2 跑完后没有查看结果值，独立审计又指出两点：不能把离散 beta
写得像连续区间，且 96/192 收敛不能只抽最高曲率的 5 帧。v26.3 因此冻结为：

- 每个离散 beta 独立判断；
- robustness 与 cost dominance 分开；
- 曲率差至少 1% 的每一帧都必须做 96/192 对照；
- 每条轨迹至少 10 个非平凡帧，才允许形成压力结论；
- 三条轨迹一条不过就不能报告该 beta；
- 不重训、不换 rank、不调门。

低 beta 的真实情况是：

```text
beta       p33 nontrivial   p45   p58   all-three stress gate
0.0001          0            0     0             NO
0.0005          0           92     2             NO
0.001          69          101    91             YES
0.002         101          101   101             YES
```

所以 `0.0001`、`0.0005` 不是算法失败，而是压力太弱，不能拿来制造“全 beta
通过”。真正可报告的离散档只有 `0.001`、`0.002`，两档都通过 robustness 与
economic dominance。

最高档的 observation 曲率差与数值收敛：

```text
trajectory   curvature p90   curvature worst   96/192 worst
p33-s05          3.215%          3.718%            0.143%
p45-s01         12.335%         16.360%            0.450%
p58-s05          4.494%          5.234%            0.186%
```

p45 的压力不是小扰动；三条最坏收敛差都低于 0.5%，曲率 p90 至少是数值差的
22 倍。

最高档的重建结果：

```text
trajectory   all match   odd match   nontrivial match   harm
p33-s05       101/101      50/50          101/101          0
p45-s01       101/101      50/50          101/101          0
p58-s05        99/101      48/50           99/101          0
```

p58 是最薄弱边界，不应写成完美 100%；但三套集合都仍在冻结兼容包络内。八方法
共同判决保持：

```text
Normalized BP          202 calls  FAIL
Zero CGLS K1           202 calls  FAIL
Reduced Warm K1        354 calls  PASS
Full parent Warm K1    404 calls  PASS
Zero CGLS K2           404 calls  FAIL
Geometry PCGLS K2      404 calls  FAIL
Zero CGLS K3           606 calls  FAIL
Zero CGLS K4           808 calls  PASS
```

独立 validator 从原始 rho 重算了 `1515` 个 192-step 正式 observation、`749`
个 96-step 对照和 `120` 个方法判决。正式 observation、收敛值的最大差都是
`0`：

```text
PASS_INDEPENDENT_FULL_ARM_RECOMPUTATION_EXTERNAL_CURVED_RAY_STRESS_V26_3
PASS_EXTERNAL_TO_FIT_CURVED_RAY_STRESS_V26_3
```

**讲人话：**现在可以有理有据地说，固定 warm initializer 的 proxy 优势没有在
“新功率/尺寸组合 + 曲折光线 forward mismatch”同时出现时消失。这个证据比
v25 强，而且结果不是通过平均值或近零扰动凑出来的。

但这仍是同一个 PoolFire 数据集上的 post-open physics stress。没有真实
`rho->n`、相机、背景图、位移提取、噪声、标定误差、official test，也没有在
v26 重测 wall/RSS。因此正式边界仍是：

```text
important_reproducible_physics_robustness_increment=true
algorithm_breakthrough=false
real_BOST=false
paper_success=false
```

完整结果见
`docs/poolfire_c_external_curved_v26_result_2026-07-28.md`。

## 266. v27：调用减少没有自动变成稳定部署加速

v26.3 已经确认固定 Reduced Warm K1 在三条 external-to-fit 轨迹、两个有效
曲折压力档上保持重建兼容。v27 没有重训或改阈值，只问一个资源问题：

> `354` 次完整调用相对 Full Parent 的 `404` 和 Zero-K4 的 `808`，是否真的
> 换成 wall、CPU 与内存收益？

我们没有只跑几次取最好值。每个 `trajectory × beta` 单元执行 102 个配对
triad，三臂六种顺序各 17 次，并保留 6 个 warmup triad。三条独立轨迹、两个
beta 共运行：

```text
1836 次正式 fresh process
 108 次 warmup
1944 次 fresh process
3888 份 worker + parent 回执
```

独立 validator 从回执重算全部场、调用账、配对统计和 50000 次 circular
moving-block bootstrap，最大统计差为 0：

```text
PASS_INDEPENDENT_RECOMPUTATION_EXTERNAL_CURVED_RESOURCE_V27
```

相对 Full Parent，六个单元的 wall 中位都快 `10.33%-11.16%`，CPU 中位 ratio
为 `0.8859-0.8951`。这证明少算 50 次 `A^T` 的时间机制是真实的，不只是调用账
好看。

但是相对最重要的 Zero-K4，wall 中位只快 `2.19%-2.89%`，保守 95% 降幅只有
`1.40%-2.45%`，远低于 10% 门；RSS p90 又高 `10.99%-12.69%`。相对父方法的
RSS p90 也高 `6.86%-11.03%`。六个单元全部失败，正式判决是：

```text
FAIL_EXTERNAL_CURVED_RECONSTRUCTION_RESOURCE_V27
algorithm_breakthrough=false
```

**讲人话：**方法确实少算、也能重建，但这台 Mac 上的 `16x16x32` 线性 proxy
算子太便宜。模型/package、几何、Python fresh-process 和内存开销吃掉了绝大
多数调用优势。它相对完整父模型快约 10%，相对不用模型的 Zero-K4 却只快约
2%-3%，而且更占内存，所以不能写成稳定部署加速。

这个负结果直接停止两件低价值工作：不再靠增加重复次数挽救点估计，也不再在同一
fresh worker 上用更大网络堆性能。下一次真正能改变论文判断的门，必须把冻结方法
接到昂贵的 nonlinear curved forward/JVP/VJP 或组内真实 BOST；同时保留 fresh
与模型常驻两种口径。若昂贵物理算子下仍不能超过 Zero-K4，就关闭“速度贡献”
主张，而不是继续包装调用数。

完整结果见
`docs/poolfire_c_external_curved_resource_v27_result_2026-07-28.md`。

## 267. v28：真正进入曲线光路非线性逆问题，首次看到稳定 matched-budget headroom

v27 的负结果把问题说得很清楚：在 `16x16x32` 的廉价直线矩阵代理里，即使少算
一半以上 `A/A^T`，模型和进程固定开销也可能把 wall 优势吃掉。因此这轮没有继续
优化同一个小矩阵 worker，而是把昂贵物理本身放进逆问题。

我实现了一个 PyTorch 可微版本的 v25 曲线光路前向：

```text
current 3D field x
  -> curved ray integration F(x)
  -> exact-program J(x)v
  -> exact-program J(x)^T w
  -> matrix-free Gauss-Newton-CGLS
```

它先通过独立 NumPy forward 一致性、JVP 中心差分、VJP 内积恒等式、非线性残差
下降和三线性重采样测试。PoolFire 观测由 192 步 NumPy 生成器产生，逆模型只用
96 步 PyTorch 程序，因此不是把完全相同的离散代码互相验证。

第一次预算是：

```text
Zero:          2 outer x 2 inner
Reduced Warm:  1 outer x 2 inner
```

这里出现了一个必须保留的失败：7 条轨迹里有 6 条能少一次 outer，但 p45-s05 的
observation residual 还没有达到 Zero 两次 outer 的终点。更重要的是，旧 runner
曾把“同样跑到最后也能达到”宽松标成 PASS。这个标签被修正为只有严格少一次
outer 才通过。

没有换大模型。只把候选在同一次线性化内的 CGLS 内步从 2 增到 3：

```text
Zero:          2 outer x 2 inner = 11F + 4JVP + 6VJP = 21
Reduced Warm:  1 outer x 3 inner =  6F + 3JVP + 4VJP = 13
```

这相当于用一个便宜 inner Krylov step 换掉一次昂贵的 ray retracing 和 Jacobian
重线性化。p45-s05 的 observation residual 从 `0.04465` 降到 `0.02004`，越过
Zero 的 `0.02903`，field 和 gradient 也继续优于 Zero。

随后从干净源码提交运行了 7 条已开放轨迹、3 种 arm 顺序，共 21 次 matched-budget
执行。独立聚合器不相信 runner 的总标签，逐文件检查源码绑定、轨迹角色、预算、
调用账和三项不等式。结果是：

```text
21 / 21 三项 matched accuracy PASS
非线性逻辑调用 21 -> 13，减少 38.10%
21 / 21 wall 都下降至少 15%
最差单次 wall 降幅 24.17%
逐轨迹三顺序 wall 中位降幅 28.33%-31.16%
```

7 条轨迹的 Reduced / Zero 终点比范围：

```text
field       0.4496 - 0.6716
gradient    0.8203 - 0.9079
observation 0.0845 - 0.6902
```

**讲人话：**这次少算的不再是一个便宜小矩阵，而是随当前三维场变化的曲线光路
forward 和 Jacobian。固定 warm initializer 让求解器从更接近真值的区域出发，
所以一次重线性化加三个内部 Krylov 步，就超过零初值两次重线性化的终点；而且这个
信号在 7 条轨迹和三种执行顺序中没有消失。

但仍然必须把边界写在结果旁边：

```text
post_open_development_only=true
fresh_process_repetition_gate_completed=false
whole_pipeline_peak_rss_gate_completed=false
external_holdout_used=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

所以当前判断是：

```text
显著阶段性进展 = true
突破性进展 = false
```

独立近邻审计还指出一个必须补的强基线：让相同规模网络直接从 observation 输出
field 初值，再接同一个未修改 GN-CGLS。只有 dual proposal + 物理 lift 能在相同
训练预算和终点门下优于这个 Direct-Field WS-GN-CGLS，才能证明贡献不只是“学习
初值有效”。

完整结果见
`docs/poolfire_c_differentiable_curved_matched_v28_result_2026-07-28.md`。

## 268. v28 独立审计：原 PASS 撤回，问题不是 Krylov 步数不够

上一节记录的是 v28 首轮聚合时看到的 Reduced-vs-Zero 信号。随后独立代码审计
发现，那个 PASS 的范围写大了，必须立刻降级。

审计确认核心自动微分本身没有明显错误：生产尺寸只读检查中，JVP 有限差分相对
误差约 `1.45e-7`，VJP 内积误差约 `3.51e-16`。21 个结果也确实完整，Reduced
相对 Zero 的三项终点和 `13` 对 `21` 调用账没有算错。

真正的问题有三项：

1. 外部模型、几何、数据和 frame payload 没有被完整绑定。
2. validator 只独立聚合 runner 写出的标量，没有从重建场独立复算科学指标。
3. 原主门忽略了 Full Parent；Reduced 的 gradient error 在 7/7 个单帧样本上
   都比 Full Parent 更高，差值范围为 `+0.000531` 到 `+0.022493`。

因此正式状态从：

```text
PASS_POST_OPEN_DEVELOPMENT_CURVED_MATCHED_BUDGET_V28
```

降为：

```text
HOLD_FULL_PARENT_AND_RECOMPUTATION_GATES_V28
algorithm_breakthrough=false
```

我没有停在审计文字上，而是实际运行了最直接的反证实验：把 Reduced 从
`1 outer x 3 inner` 加深到 `1 outer x 4 inner`，在同一 7 个已开放样本上与
原冻结的 Full Parent `1x3` 比较。

结果：

```text
observation 优于 Parent：7 / 7
gradient 劣于 Parent：7 / 7
三项同时不劣于 Parent：0 / 7
```

p33-s01 的 gradient 差值只从 `+0.022493` 变成 `+0.022371`，但 observation
差值已经改善到 `-0.007724`。讲人话：更多 Krylov 步继续把“看得见的观测残差”
压低了，却几乎没有找回 reduced representation 丢掉的“观测近零空间三维结构”。

所以现在停止用更多 inner steps 挽救 Reduced。下一项真正有科学价值的实现是同
训练预算的 `Direct-Field WS-GN-CGLS`：网络从相同 observation / geometry
直接输出 field 初值，再接同一个未修改 GN-CGLS。它将检验收益究竟来自一般的
learned warm start，还是来自我们想主张的 dual/reduced + physical lift 结构。

## 269. v29：Direct-Field 真实训练后五折通过，但“少算”还没有变成“更快”

这轮没有继续给 v28 的 reduced 表示打补丁，而是把独立审计要求的强对照真正做了
出来：一个 10,524 参数的因果三维 CNN，输入相邻两帧 geometry-equalized BP 和
固定物理 base，直接输出三维场初值；随后只做观测可计算的标量校正，再接完全未改的
CGLS K1。

最初的 p33 pilot 说明，Direct proposal 在 field 和 gradient 上已经有明显
headroom，但 observation 还没有达到 Zero-K4。继续堆 observation loss、K3 teacher
或低频 residual basis 都没有形成严格 K4 三项通过。这里没有挑好看的结果写成功，
而是回头检查 baseline 本身：Zero-K4 虽然继续压低 observation，却开始明显恶化
gradient；Zero-K3 才是这组 proxy 上更平衡的 early-stopped 解。

因此冻结比较对象为 Zero-K3，并实际重训五个 leave-one-trajectory-out fold。每个
fold 的 held-out trajectory 都不参与 alpha、cap 或网络训练。五折 500 帧的聚合
p90 为：

```text
                   Direct K1    Zero K3    relative reduction
field              0.572526     0.679131       15.70%
gradient           0.806682     0.916389       11.97%
observation        0.370540     0.380479        2.61%
```

五条轨迹的三项 p90 全部通过。p45-s05 仍有 12/100 帧 observation harm，所以不能
写成每一帧都支配；其余四折三项逐帧 harm 都是 0。

随后在五条 fit 上训练一个最终 checkpoint，再一次性评分已经打开但不参加本次训练的
p14-s01。p14 的三项 p90 为：

```text
field        0.484033 vs 0.633080
gradient     0.703886 vs 0.835100
observation  0.353301 vs 0.371543
harm         0 / 100 on all three metrics
```

为了避免 runner 自己验证自己，又写了另一条独立程序：它不导入正式 runner 的数据
准备、deployment、Zero-CGLS 或 metric 函数，重新计算五条 fit 的 alpha/cap，
重载 checkpoint，再重做 p14 的 100 帧输出和指标。最大指标差只有
`9.54e-9`，调用账和判决完全一致：

```text
PASS_INDEPENDENT_RECOMPUTATION_OPENED_VALIDATION_V29_3
```

truth-free 部署代码的实际回执是：

```text
Direct K1 = 200A + 201A^T = 401 complete calls
Zero K3   = 300A + 300A^T = 600 complete calls
reduction = 33.17%
```

这次调用优势是真执行，不是公式估算。但 fresh-process benchmark 给出了必须正视的
负结果：CPU compute wall 慢 `2.59-2.91x`，fresh wall 慢 `1.26-1.38x`，
peak RSS 高 `1.28-1.34x`；MPS 也没有通过。当前直线 `16x16x32` NumPy 算子太
便宜，CNN 和张量转换的固定开销更大。

**讲人话：**现在终于有一条候选方法，能在五折和一个开放验证轨迹上用约三分之二
的完整算子调用，得到比 Zero-K3 更好的场、梯度和观测精度。这是实际算法进展。
但在当前 Mac 的廉价直线代理上它反而更慢、更占内存，所以还不是“加速算法突破”。

正式边界：

```text
stage_level_key_progress=true
algorithm_breakthrough=false
untouched_or_fresh_validation=false
curved_inverse_evaluated_for_v29=false
real_BOST=false
wall_or_rss_advantage_proven=false
paper_success=false
```

完整结果见
`docs/poolfire_c_direct_field_v29_result_2026-07-28.md`。

## 270. v35-v37：一步修正失败，四步低保真缺陷修正同时通过精度与 fresh 资源门

这轮没有继续扩大网络，也没有绕开 Full Parent。真正的问题是：能不能少做一次
昂贵曲线重线性化，同时把缺掉的三维结构补回来。

先跑了最便宜的 SARC-K3-M1：

```text
learned Direct -> curved GN-CGLS K3
-> 1 次 straight-adjoint residual correction
-> 1 次 curved F safety
```

它在 12/12 轨迹上优于 Zero，但有 5 条 observation 仍劣于 Full Parent，最差
ratio 为 `1.041992`。正式与独立判决都是失败。这证明“随便加一个反投影”不够。

随后冻结 SARC-K3-M4。差别只在 residual subproblem 固定做恰好 4 步 straight
CGLS；不看 truth、不按轨迹调步数。结果是 12/12 轨迹的 field、gradient 和
observation 全部同时不劣于 Full Parent 与 Zero：

```text
worst ratio vs Full Parent
field        0.895933
gradient     0.952248
observation  0.975982
```

非线性账从 Zero 的 `21` 降到 `14`，理论减少 `33.33%`；straight 总账是
`6A+7A^T`。另一条程序独立重建 12 个场，field 最大差为 0，曲线 prediction 最大
差 `3.55e-15`，判决一致。

随后不是停在调用账，而是执行了 144 个 fresh child：

```text
12 trajectories x 2 arms x (1 warmup + 5 measured)
```

资源门结果：

```text
trajectory-equal median external wall ratio  0.745827
worst trajectory external wall ratio         0.754330
trajectory-equal median RSS ratio             1.006705
worst trajectory RSS ratio                    1.023578
```

三个冻结门全部通过。独立 validator 又逐项检查 144 receipts、144 fields、顺序、
输入、调用账和资源算术，最终状态：

```text
PASS_INDEPENDENT_VALIDATION_SARC_K3_M4_RESOURCE_V37
```

独立 validator 自己前两次 fail-closed：一次误读 v36 嵌套 ledger，一次对 fresh
中间 residual history 使用过严 `1e-9` 容差。两个失败回执都保留；修复只针对
validator 结构和浮点比较，资源门与算法结果没有改。

**讲人话：**这次候选不只是少算了 33.3% 的昂贵物理调用，而且本机 12 条轨迹的
fresh external wall 全部实测下降约 24.6%-27.7%，RSS 最差只增加约 2.36%。
这是目前最强的算法证据。

但 Full Parent 只用 13 次 nonlinear call，而候选用 14 次；候选更准但略贵。因此
它是新的 Pareto 工作点，不是对所有方法的无条件支配。并且 12 条科学轨迹都已为
开发打开，untouched test 和真实 BOST 仍未运行：

```text
key_positive_result=true
breakthrough_candidate=true
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/poolfire_c_sarc_k3_m4_v37_result_2026-07-29.md`。

## 271. v38-v39.1：两条额外轨迹保住精度，并用系统原始回执补齐资源证据

v37 之后没有继续堆网络，而是直接追两个会改变论文判断的问题：方法能不能迁到
另外两条轨迹，以及资源优势是不是 runner 自己报出来的“自证”。

先做 v38.1。候选结构和所有门都不改，只在两条未参与 v35-v37 方法开发的轨迹上
运行。两条都通过相对 Full Parent 与 Zero 的 field、gradient、observation 门：

```text
P22-S05 candidate = 0.377172 / 0.750618 / 0.010339
P58-S01 candidate = 0.394918 / 0.694652 / 0.029653
```

相对 Direct-K4，P22 的三项比值是
`0.999344 / 0.999342 / 0.968067`；P58 是
`1.003505 / 1.003326 / 0.952726`。也就是说，P58 的 field/gradient 略差约
0.35%/0.33%，但 observation 好约 4.73%。它是兼容的精度-成本折中，不是所有
指标无条件支配。

独立程序重建场后通过；v38.2 又从封存数组重新算了 16 组指标，数值信号仍在。
但所有官方 PoolFire test stream 在历史流程中都已经打开，v38 只能叫
post-open method transfer，不能叫 fresh generalization。

随后发现 v39 第一版有一个真正的证据缺口：它丢掉了 `/usr/bin/time -l` 的原始
stderr，独立 validator 只能复核 runner 已声明的 wall/RSS。这个结果没有被硬说
成论文证据，而是重写为 v39.1：

```text
2 trajectories x 4 arms x (1 warmup + 8 measured)
= 72 fresh child processes
```

每个 child 都单独保留原始 macOS wall/RSS 回执；每个 arm 在四个顺序位置各出现
两次。正式资源结果：

```text
trajectory-equal median wall / Zero       0.744260
worst trajectory wall / Zero              0.745148
trajectory-equal median wall / Direct-K4  0.825418
worst trajectory wall / Direct-K4         0.827567
worst RSS / Zero                           1.012848
worst RSS / Direct-K4                      1.015228
```

72 份 worker receipt、72 个场和 72 份原始 time receipt 又由不导入正式
benchmark/worker/solver 的程序独立重算：

```text
PASS_INDEPENDENT_VALIDATION_SARC_POSTOPEN_RESOURCE_V39_1
maximum field absolute difference = 3.793420197406583e-08
maximum summary numeric difference = 0.0
```

**讲人话：**现在可以有根据地说，在这两条额外但已历史打开的 PoolFire 轨迹上，
SARC-K3-M4 不仅保住精度，而且相对 Zero-K4 快约 25.6%，相对 Direct-K4 快约
17.5%，内存增加不超过冻结容差。相对 Direct-K3 它反而平均慢约 1.8%，所以绝对
不能写成“全局最快”。

这是真实的关键正结果，但突破标签仍不变：

```text
key_positive_result=true
post_open_transfer_proven=true
fresh_process_resource_replication_proven=true
all_official_poolfire_test_streams_historically_opened=true
generalization_proven=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/poolfire_c_sarc_postopen_v39_1_result_2026-07-29.md`。

## 272. v40.2：第一次跨出 PoolFire，外部精度门 0 / 4

v39.1 以后，继续在 PoolFire 上跑更多帧已经不能回答泛化。于是这轮没有改模型，
而是把冻结的 SARC-K3-M4 零适配放到 BLASTNet 的预混 H2-air 槽式燃烧器 DNS。
它是另一套燃烧物理与数据族，官方网格为 `651 x 401 x 201`，提供五个时刻的
`RHO_kgm-3`。

为了先排除“下错数据”，三份坐标和五份密度文件从公开发布端重新获取一次，
8 / 8 与第一份副本逐字节一致。随后按结果前冻结的裁剪、粗网格、
straight/curved forward、checkpoint 和 1.01 倍精度门，一次运行四个目标时刻。

正式结果是：

```text
snapshot  field/K4  gradient/K4  observation/K4
1         0.982258  1.003059     1.039317
2         0.981588  1.015216     1.037703
3         0.979353  1.011266     1.021523
4         0.980637  1.005087     1.065418
```

四帧的 SARC 三项都优于 Zero，也都没有被 Direct-K3 Pareto 支配；但 0 / 4 同时
通过相对 Direct-K4 的 field、gradient、observation 门。正式状态是：

```text
FAIL_EXTERNAL_BLASTNET_ACCURACY_GATE_SARC_K3_M4_V40_2
```

独立 validator 没有导入正式 runner 的指标或判门函数，重新跑了 16 次 curved
forward。最大指标差 `8.88e-16`，最大 observation receipt 差 `2.22e-16`，
封存场与 prediction barrier 都没变。因此失败不是汇总脚本造成的。

精度失败后，资源门按协议没有运行。否则得到的只是“不等精度情况下更快”，与师兄
确定的“同精度下降低重建成本”无关。

外部门已经开封后，又做了一次只用于解释的幅度扫描。沿原 correction 方向，
四帧的 observation 最优 alpha 分别是：

```text
1.4, 1.4, 1.4, 1.2
```

原 correction 的确偏弱，但调到最优幅度后仍是 0 / 4 同时过三门；三帧
observation 仍失败，四帧 gradient 都越界。也就是说，不是把修正乘大就能解决，
原 correction 的一维方向本身不够。

**讲人话：**这次算法没有成功，但我们真正淘汰了两个错误想法：

1. PoolFire 上的 SARC 可以不适配直接搬到另一类燃烧场；
2. 外部失败只需要学一个更大的全局 correction 增益。

下一版只有在另一个开发数据上学习“多个修正方向 + 物理尺度/残差频谱条件”才有
继续计算的意义。BLASTNet `phi=0.5` 已经开封，只能用于机理开发，不能再充当
正式外部测试。

```text
external_zero_adaptation_executed=true
external_accuracy_gate_pass=false
post_open_amplitude_only_repair=false
resource_gate_run=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_sarc_external_v40_result_2026-07-29.md`。

## 273. v41：拆开四个 Krylov 增量逐帧调权，仍然 0 / 4

v40.2 已经证明“原 correction 只乘一个更大的数”修不好。于是这轮没有马上训练
大网络，而是做更便宜、也更能改变判断的机制实验：把 straight-ray CGLS 的四步
correction 拆成四个 increments，让每个 BLASTNet 开封快照单独用 observation
residual 选择四个权重。

每帧跑 3 个固定起点的有界 L-BFGS-B，权重范围是 `[-2, 3]`。优化器不读取
truth；truth 只在权重选完后评分。结果：

```text
snapshot  field/K4  gradient/K4  observation/K4
1         0.976484  1.016197     1.019140
2         0.977307  1.031789     1.019967
3         0.974200  1.027001     1.004856
4         0.977260  1.014603     1.056837
```

四帧 field 都进一步改善，但四帧 gradient 都过不了冻结的 1.01 线，三帧
observation 也过不了，最后仍是 `0 / 4`。跨帧中位数也显示同一冲突：

```text
                         field       gradient    observation
four-direction         0.955901     0.989316     0.310794
Direct-K4              0.979420     0.971054     0.301608
```

独立程序重新构造四步 basis、重跑 12 个优化起点、重建候选场并重算完整门：

```text
PASS_INDEPENDENT_RECOMPUTATION_KRYLOV_INCREMENT_SPAN_V41
maximum metric difference       = 3.58e-15
maximum optimizer weight diff   = 0
maximum candidate field diff    = 0
```

所有最终被选中的最佳起点都报告收敛，但 S3 有一个未被选中的起点失败，因此不能
把它升级为“四方向空间已被数学证明不够”。最精确的说法是：

```text
NO_PASSING_CANDIDATE_FOUND_UNDER_OBSERVATION_OBJECTIVE
```

还有一个重要成本纠正：`4A + 4A^T` 只表示构造 straight basis 的成本。逐帧
L-BFGS-B 总共用了 `1425` 次 curved forward，所以 v41 是后验机理诊断，不是
可部署算法，更不是加速结果。

近邻文献审计又排除了“四个可学习 Krylov 权重就是创新”的表述。RAM 已有
Krylov Subspace Module，CASSI 已有复杂光学 forward 下的 CG unrolling，
FCG-NO、DCDM 和 NeurKItt 也都覆盖了神经方法辅助 Krylov 求解。我们仍可检验的
窄差异只能是 BOST 的 straight-to-curved cross-fidelity correction、固定昂贵
调用预算和 observable fail-closed 回退。

**讲人话：**四个旋钮比一个旋钮灵活，但这次仍没把三项指标一起调回合格区。
下一步先做 truth-aware constrained oracle：如果 oracle 在同一四方向空间里都找
不到合格点，就关闭这条 basis；只有 oracle 能通过，才值得训练 observation-only
系数网络。

```text
post_open=true
new_external_generalization_evidence=false
same_cost_or_speed_claim_authorized=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_krylov_increment_span_v41_result_2026-07-29.md`。

## 274. v42/v43：oracle 也没救回固定四方向，停止训练系数网络

v41 只按 observation 目标调四个权重，可能因为没有把 field 和 gradient 门直接
放进优化器而错过可行点。这一轮真正执行了两种 truth-aware constrained oracle，
回答“这个四方向空间里到底有没有值得学习的目标”。

v42 先穷举每帧 323 个确定性 screen 点，再直接对精确 curved observation 做
SLSQP。screen 是 0 / 4 通过，但后面的第一个优化起点就把每帧预算耗尽：合计
2800 次 curved forward 和 1508 次 reverse VJP，四次优化都没有形成可判定解。
独立重放最大差为 0，所以正确结论只能是“搜索器预算耗尽、结果 inconclusive”，
不能说 basis 已经失败。

随后没有继续堆同类 SLSQP，而是冻结 v43：在每个 trust-region 中心计算一次精确
curved prediction 和四个精确 JVP，用局部 affine observation 模型求四维 QCQP，
候选再回到精确 curved forward 上验收。场与梯度门、四个方向、系数盒都没有改变。

正式 v43 约 121 秒完成，独立程序约 119 秒完整重放。四帧最佳 ratio 是：

```text
snapshot  field/K4  gradient/K4  observation/K4
1         0.979208  1.010000     1.020448
2         0.984410  1.010000     1.055484
3         0.980775  1.010000     1.022709
4         0.978627  1.010000     1.057592
```

四帧的场误差都更低，但梯度被推到 1.01 边界，observation 仍全部超过 1.01。
完整门仍是 0 / 4。正式调用账是：

```text
31 F + 112 JVP + 0 VJP
```

独立 validator 重新构造 basis、重跑 curved forward/JVP 轨迹、重做每次
接受/拒绝并重算完整门；逐行、联合指标和选中场的最大数值差都为 0：

```text
PASS_INDEPENDENT_RECOMPUTATION_KRYLOV_JVP_TRUST_REGION_ORACLE_V43
```

**讲人话：**这次不是“网络没调好”，而是连能看 truth 的 oracle 在冻结搜索范围
内也没找到四个系数同时守住三项指标。继续给同一个固定四方向 basis 训练
MLP、FNO 或 DeepONet 没有依据，所以这条系数预测支线现在关闭。下一条真正有
科学价值的工作必须改变方向/basis 如何产生，让新方向携带 curved-observation
信息，而不是继续放大同一个搜索器。

这是经过验证、能节省后续大量训练成本的负结果，但不是数学不可行性证明，也不是
可部署算法、加速、真实 BOST 或论文成功：

```text
validated_bounded_negative=true
selector_pilot_authorized=false
same_cost_or_speed_claim_authorized=false
new_external_generalization_evidence=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_krylov_trust_region_v43_result_2026-07-29.md`。

## 275. v44.3：新方向是真的，但 raw curved-adjoint 仍不够安全

v43 已经把“继续在固定四方向里换优化器”这条路关掉。这一轮不再调旧权重，而是
真正改变 basis：在 Direct-K3 处用部署可见的 curved observation residual 做一次
精确 VJP，把结果投到粗网格，再从四个 straight-CGLS 增量中正交化，得到第五个
curved-adjoint 方向。

先确认它不是假新方向。四帧的旧 span 外能量比例是：

```text
S1  19.04%
S2  24.33%
S3  26.35%
S4  31.42%
```

它与旧 span 的最大余弦只有 `2.95e-15`，curved adjoint 恒等式误差也在
`1.53e-15` 以下；第五权重为 `0.144-0.381`。因此新方向确实被优化器使用，也
确实携带旧四方向没有的信息。

正式五方向结果相对 Direct-K4 为：

```text
snapshot  field/K4  gradient/K4  observation/K4
1         0.976348  1.015315     1.014640
2         0.976447  1.031173     1.013825
3         0.973110  1.026650     0.992946
4         0.976449  1.014803     1.042355
```

四帧 field 都更好，但四帧 gradient 都越过冻结的 `1.01` 门，observation 也只有
一帧通过，所以完整门仍是 `0 / 4`。与配对四方向控制相比，第五方向在四帧都改善
field 和 observation，说明它不是完全无效；问题是它没有修复 gradient 安全性。

独立程序不导入正式 runner/validator，重新生成两阶段数据和候选场：

```text
PASS_INDEPENDENT_RECOMPUTATION_PAIRED_COARSE_ADJOINT_ENRICHMENT_V44_3
Stage-A 最大数值差     4.49e-11
Stage-B 最大数值差     2.22e-15
候选场最大绝对差       9.02e-17
正式报告最大数值差     0
```

正式机理账本是 `96 F + 292 JVP + 4 VJP`，因此这只是昂贵的开封后诊断，不能写成
部署加速。

**讲人话：**我们造出了一条真正不同的新路，它确实让“大轮廓”和“相机观测”更
接近答案，但把密度梯度细节弄坏了。现在最重要的决策不是马上训练网络，而是停止
模仿这个 raw 方向，只做固定的 gradient-aware / Sobolev 预条件方向，并与匹配
controls 比较。只有预条件后的方向先通过完整门，才值得让神经网络学习。

```text
validated_bounded_negative=true
raw_curved_adjoint_training_authorized=false
same_cost_or_speed_claim_authorized=false
new_external_generalization_evidence=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_curved_adjoint_v44_result_2026-07-29.md`。

## 276. v45：把新方向磨平，还是没有得到可学目标

v44 最明显的问题是 gradient 变差，所以先试最便宜的解释：也许 curved-adjoint
方向高频太强，只要做固定 Sobolev 平滑就能保住细节。

这一轮没有训练网络，只比较 `lambda=0.25/1/4/16`、raw、四方向和
straight-continuation controls。结果不论用固定 lambda、只看 observation 选 lambda，
还是事后让 truth-oracle 在整个 family 中挑答案，四个快照都没有完整通过点。

逐帧最佳 fixed-Sobolev 的最差指标比为：

```text
S1  1.0155
S2  1.0302
S3  1.0252
S4  1.0427
```

**讲人话：**把同一条路磨得更平滑，并没有把它变成安全路线。v45 是 scratch
diagnostic，没有独立重放，所以它只负责关闭“立刻正式化 Sobolev family”这一步，
不能写成普遍的平滑方法无效。

```text
formal_v45_authorized=false
algorithm_breakthrough=false
```

## 277. v46：整体调幅也没有网格见证

接着检验第二个便宜解释：方向本身也许没错，只是整条 correction 走多或走少了。
冻结 v44 场后扫描

```text
x(alpha) = Direct-K3 + alpha * (x_v44 - Direct-K3)
alpha = 0, 0.025, ..., 2
```

四帧各 81 点，总账 `332 F + 0 JVP + 0 VJP`。逐帧最小 minimax ratio 与 alpha 为：

```text
S1  1.014653  alpha=0.975
S2  1.020830  alpha=0.825
S3  1.011276  alpha=0.725
S4  1.042263  alpha=0.975
```

四帧都没有 passing grid point，也没有固定 alpha 同时过四帧。独立程序重新构场和
重跑 curved forward，最大 metric / ratio 差为 `3.33e-15 / 2.33e-15`。

这里必须守住一句边界：81 点网格没有 witness，不等于连续 alpha 域无解。v46 只关闭
“用这个冻结网格设计幅度 selector”，不能写成数学不可能。

```text
amplitude_only_selector_authorized=false
continuous_path_impossibility_proven=false
algorithm_breakthrough=false
```

## 278. v47：五个旋钮分别调，gradient 与 observation 仍在打架

v46 失败后，不再把五个分量绑成同一个 alpha。v47 保留四个 straight-CGLS 增量和
一个 curved-adjoint 方向，让五个权重独立变化；field 和 gradient 必须满足冻结真值
约束，objective 只减精确 curved observation residual。

两个起点分别是 v43 可行四维点加 `w5=0`，以及从零点径向收缩到可行域的 v44 权重。
每起点最多 8 次 JVP trust-region 外循环。实际精确评分 48 个候选：

```text
snapshot  candidates  passes  field/K4  gradient/K4  observation/K4
S1        12          0       0.978542  1.010000     1.016493
S2         7          0       0.983386  1.010000     1.045775
S3        17          0       0.979858  1.010000     1.011126
S4        12          0       0.978027  1.010000     1.043375
```

账本是 `16 A + 16 A^T + 52 F + 310 JVP + 0 VJP`。独立重算器不导入 v47 runner，
重新构五个方向、全部候选场、curved prediction、三指标与调用账：

```text
PASS_INDEPENDENT_RECOMPUTATION_POST_OPEN_FIVE_DIRECTION_CONSTRAINED_V47
metric 最大差       3.33e-15
ratio 最大差        2.33e-15
joint metric 最大差 1.11e-15
```

最有价值的现象不是“又 0/4”，而是四个最优点都把 gradient 推到允许的 `1.01`
边界，observation 仍超标。field 已有余量，所以当前局部冲突明确落在 gradient 与
curved-observation 之间。

**讲人话：**现在没有理由训练五系数网络，因为我们还没有证明这五个旋钮能调出合格
答案。但两个相关起点和 7-17 个候选也不足以宣布整个五维空间无解。下一步只花一笔
封顶预算做确定性五维全局反例搜索；若任一快照仍失败，工程上关闭五系数路线，再新增
一个真正改变 span 的二阶 curved Krylov 方向。

```text
five_coefficient_learning_target_authorized=false
global_five_space_impossibility_proven=false
formal_algorithm_authorized=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_curved_followups_v47_result_2026-07-29.md`。

## 279. v49-v50：先拒绝一次不可重放结果，再得到可信的固定表负结果

v47 只看了五维空间中 48 个局部候选，所以这一轮用封顶预算检查较远处是否还有
完整门 witness。

v49 先做 1,664 次全局搜索，再跑四个 Powell 局部搜索；runner 记录了 2,304 个
唯一候选和 0 个通过点。但独立 validator 在第 48 个局部请求发现请求几何不一致。
小于 `1e-10` 的浮点差异改变了 Powell 分支，四个局部搜索也都没有报告收敛。
因此 v49 不是负结果，而是：

```text
v49_scientific_decision=INCONCLUSIVE
```

v50 保留同一全局搜索，把不可稳定重放的 Powell 换成目标函数无关的固定局部候选
表。runner 和 validator 分别生成候选表，validator 再对全部 2,304 个唯一候选做
精确 curved forward 和完整门重算：

```text
independent_validation  PASS
unique candidates       2,304
complete-gate passes    0
best field / K4         0.983386
best gradient / K4      1.010000
best observation / K4   1.045775
```

**讲人话：**这次真正可靠的不是“我们证明五维无解”，而是：在一张事先固定、
可以被第二套程序逐项重放的 2,304 候选表里，没有合格答案。最好的候选已经把
gradient 安全额度全部用完，observation 仍高出门槛约 3.58 个百分点。继续在原五个
方向上细调，信息价值已经很低。

下一步不是训练五系数网络，也不是继续堆搜索点，而是构造第二个固定线性化
residual-adjoint 增广方向：先消去第五方向解释的 residual，再对剩余 residual 做
VJP，并从现有五方向中正交化。它会真正改变 span，但不能夸大成完整 Hessian 或
二阶求解器。

```text
fixed_candidate_roster_negative=true
five_space_engineering_route_closed=false
mathematical_nonexistence_proven=false
stage_2_authorized=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_fixed_roster_v50_result_2026-07-29.md`。

## 280. v51：第六方向是真的，也真的被用了，但梯度安全仍没过

v50 告诉我们原五个方向继续细调价值很低，所以这一轮没有再加搜索点，而是真正换了
可达空间。具体做法是：先让现有第五方向解释一部分 curved observation residual，
再把剩余 residual 通过冻结在 Direct-K3 的精确伴随 `J^T` 送回三维场，投到粗网格
并从旧五方向中正交化，得到第六方向。

它不是完整 Hessian 或二阶优化器，准确名称是“固定线性化的第二残差伴随增广方向”。

独立流程重放器没有导入正式 runner，而是重新生成方向、S5/S6 搜索和所有指标；
但两者共享冻结的 v44 curved forward/JVP/VJP、几何和 metric/gate 内核，所以不是
外部独立物理实现。方向审计得到：

```text
粗网格旧 span 外能量       30.84%
观测旧 span 外能量         38.54%
观测切空间秩               5 -> 6
选中 w6                    0.244372
Stage-A 数组最大差         0
最终 score 最大差          2.44e-15
```

这说明第六方向在当前固定线性化和参数域内既不是旧方向的局部数值重复，也不是
“算出来但没使用”的装饰项；它不证明全局可辨识性或抗噪秩。

相同搜索账 `9 F + 48 JVP` 下，S5 与 S6 相对 Direct-K4 的结果是：

```text
              field      gradient    observation
S5            0.976447   1.031173    1.013825
S6            0.975858   1.030041    1.008749
w6=0 ablation 0.976052   1.030495    1.015399
```

**讲人话：**新方向有用，它把 observation 从 1.01 门外推进了门内；但 gradient
仍为 `1.030041`，比门线高约 2.0 个百分点，所以完整门还是失败。这不是突破，但它
把问题进一步钉死：只沿 observation residual 继续加伴随方向，能补观测，却没有
消掉梯度安全瓶颈。

S5 的四次额外 F 是 no-op padding，并没有增加搜索机会；第六方向准备的
`1 F + 1 JVP + 2 VJP` 也另记。因此这里的“相同预算”只是名义搜索调用账匹配，不是
端到端总成本相同。`w6=0` 消融也没有重新优化前五个权重，不能拿它证明五维连续空间
不可能。

下一项不训练大网络，也不机械加第三条同类方向。先在冻结六维 span 内做一次有界
truth-feasible headroom 诊断，区分“六维空间本身没有合格点”和“部署可见 selector
没有找到合格点”。这两种失败需要完全不同的后续算法。

```text
direction_is_real_and_used=true
observation_gate_crossed=true
complete_gate_pass=false
six_space_impossibility_proven=false
neural_training_authorized=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_second_residual_adjoint_v51_result_2026-07-29.md`。

## 281. v52：冻结六维有限候选表 0 / 2,312

v51 留下了一个必须回答的问题：第六方向已经新增了秩，为什么完整门还失败？可能是
observation-only selector 选错了系数，也可能是固定六方向空间本身没有合格点。

这轮没有训练网络，而是在同一个 S2 快照的六维空间里先构造 field/gradient 真值
可行域，再把候选表完全冻结：

```text
global Sobol rays    2,048 requests
local S6 neighborhood  256 requests
historical anchors        8 requests
total                  2,312 requests
```

候选表在第一次 curved forward 前密封，不去重、不早停。v47/v50 锚点重复，所以
唯一权重是 2,311，但 runner 和 validator 都必须完整执行 2,312 次。

独立红队在正式计算前发现了两个会伤害证据的 P1：`NaN` 能绕过旧差异比较，以及长跑
后未重新哈希输入。修复后，runner 与不导入 runner 的 validator 分别跑完
`2,312 F`，重新生成的 roster 和全部 score 最大差都为 0，source/input 前后身份
也一致。

正式结果：

```text
complete-gate passes               0 / 2,312
all three no worse than Zero       1,867
observation no worse than Direct-K3  778
all three within 1.01 of Direct-K4     0

best field / K4                    0.985041
best gradient / K4                 1.010000
best observation / K4              1.038412
best minimum gate margin          -0.028412
```

**讲人话：**最好的点已经把 gradient 安全额度用到边界，field 也够好，但
observation 仍差 2.84 个百分点。v51 原始 S6 能把 observation 压进门内，是因为
gradient 退到了 1.030041；一旦要求 field/gradient 安全，observation 又回到门外。

这不是“连续六维无解”的证明。在连续约束 oracle 完成前，先暂停训练六系数
selector：目前没有一批合格标签，继续堆 FNO/DeepONet/MLP 只会学习一个尚未
证实可达的目标。

下一项只保留一次六维连续约束 oracle，用精确 curved VJP、固定多起点和独立请求
轨迹重放检查是否存在很窄的可行口袋。若仍失败，正式停止 residual-only 六方向，
改造表示本身，让新方向直接处理梯度安全；若找到 witness，才回到部署可见
gradient-aware selector。

```text
finite_roster_negative=true
continuous_six_space_nonexistence_proven=false
six_coefficient_selector_training_authorized=false
algorithm_breakthrough=false
paper_success=false
```

完整结果见
`docs/blastnet_h2air_phi05_six_space_truth_feasible_roster_v52_result_2026-07-29.md`。

## 282. v53：连续 oracle 无结论，但找到了会污染 smooth KKT 的体素边界切换

v52 的 2,312 个固定候选全部失败后，这轮真正运行了事前冻结的连续六维约束 oracle：
12 个起点、每起点最多 256 次 exact curved request。runner 和独立 validator
分别重走请求、终点、KKT 和三指标，正式账为 `3,088 F + 3,070 reverse-equivalent`。

结果仍是 0 个完整门 witness。正式最佳点相对 Direct-K4 为：

```text
field        0.982639
gradient     1.010000
observation  1.033843
```

但 11 个起点耗尽预算，唯一 SciPy success 也没过 KKT stationarity，所有终点没有
同时满足有界负结论条件。因此正式判决不是“六维无解”，而是：

```text
INCONCLUSIVE_CONTINUOUS_SIX_SPACE_ORACLE_S2_V53
```

随后没有继续盲目加 optimizer budget，而是把 gradient 约束椭球精确映到球面并做
双侧导数检查。真正抓到的问题是：`N=96` 终点的 `198,912` 个采样位置里，正向
`1e-8` 扰动只有一个 midpoint 跨了体素 lower cell，负向没有：

```text
view/step/stage/ray       0 / 77 / midpoint / 259
cell                      [26,19,41] -> [26,19,40]
positive objective jump   4.07e-5
negative objective change 1.18e-11
AD derivative             -1.176e-4
central derivative         2.036e2
```

同一个终点改用 N=128 时该处没有切换；但 N=128 重新优化后的终点又在另一个
midpoint 出现一个单侧切换。这说明 fixed-step + 三线性 field-gradient forward 是
piecewise smooth，换步数只移动切换面，不能让 smooth KKT 自动变成可靠裁判。

为了仍然寻找正见证，12 起点 Powell 共执行 2,973 次 exact F，最佳 observation /
Direct-K4 为 `1.033484`；固定 seed 的 1,024 点 Sobol 球面探测有 434 个可行 exact
F，最佳为 `1.107221`。两者都是 0 pass，但都不构成不存在性证明。

**讲人话：**旧六方向空间目前既没有合格答案，也没有被数学证明无解；更关键的是
优化器所在的数值地面有细小台阶。继续往同一个 smooth optimizer 里投算力已经不值。
下一步改表示：用最小 observation-only dual proposal，经精确 A^T、可观测 alpha
和未修改 CGLS K1 做 warm start；旧六系数网络正式停止。

```text
formal_v53_independently_replayed=true
postopen_cell_switch_diagnostic_independently_validated=false
continuous_six_space_nonexistence_proven=false
six_coefficient_selector_training_authorized=false
new_initializer_hypothesis_may_be_preregistered=true
matched_accuracy=false
speedup=false
real_bost=false
algorithm_breakthrough=false
```

完整结果见
`docs/blastnet_h2air_phi05_six_space_continuous_oracle_v53_result_2026-07-30.md`。

## 283. v54-v55：旧 dual CNN 有一点跨数据族信号，但还达不到同精度少调用

v53 决定停止旧六方向后，我没有立刻训练一个更大的网络。先补了一个更便宜、也更
能改变路线判断的控制：把 PoolFire 五条 fit 轨迹上已经训练完成的 `w16d2`
detector CNN 原封不动放到 BLASTNet H2-air 上，权重、输入排布、`A^T` lift 和
observable alpha 全部不改。

v54 每帧只用 `2A+2A^T`，与 Zero-K2 同成本。四个时刻的中位数是：

```text
                    field      gradient    observation
w16d2 Dual-K1      0.963449    0.976921    0.800665
Zero-K2            0.948967    1.020615    0.754943
Zero-K4            0.887832    1.128520    0.554804
```

它比 Zero-K2 的 gradient 好 4.28%，但 field 差 1.53%、observation 差 6.06%。
所以模型不是完全没有迁移，它更像带来了一点高频正则化，却没有给出更好的完整解。

根据这个 trade-off，v55 只允许增加一件事：把同一 CGLS recurrence 再走一步，
总成本变为 `3A+3A^T`，与 Zero-K3 同成本。结果：

```text
                    field      gradient    observation
w16d2 Dual-K2      0.908949    1.111039    0.645687
Zero-K3            0.912750    1.093312    0.653189
Zero-K4            0.887832    1.128520    0.554804
```

相对同成本 Zero-K3，field 和 observation 分别改善约 0.42% 与 1.15%，但
gradient 恶化约 1.62%。相对 Zero-K4，field 仍差约 2.38%，observation 仍差约
16.38%。四个目标时刻都没有同时进入 Zero-K4 的 1.01 三指标 envelope。

runner 之外的另一套实现重新装载 checkpoint、重算网络、CGLS、三指标和全部调用账；
v54 与 v55 的最大指标差都为 0。因此当前可信判决是：

```text
measurable_cross_domain_signal=true
matched_accuracy_call_reduction=false
larger_same_family_cnn_authorized=false
fresh_external_generalization=false
algorithm_breakthrough=false
```

**讲人话：**这次不是“一无所获”。旧模型确实比同成本迭代在两项上略好，说明
warm-start 思路没有完全断；但它在第三项上还债，而且追不上多一步 CGLS。现在最
合理的动作不是把 CNN 做大，而是换坐标：先让冻结几何 `A` 产生 measurement-range
basis，再训练一个只有数百参数的小 dual gate。这个实验若失败，就能把问题明确
归因到低秩 measurement-range 表示，而不是继续猜网络容量。

完整结果见
`docs/blastnet_h2air_phi05_w16d2_transfer_controls_v55_result_2026-07-30.md`。

## 2026-07-30：v56-v58 找到第一条“精确 + 少调用 + fresh 加速 + 内存不伤”的路线

这次先没有继续堆网络。v56 问了一个更基础的问题：能不能把测量空间压成一个小的
全局 basis，再在那个 basis 里跑？答案是否定的。rank 64 只覆盖约 `15.29%` 的
几何能量，随机可实现观测投影回去的残差中位数高达 `0.920678`。这条低秩路线当场
停止，没有拿它继续训练。

随后换了一个不丢信息的表示。固定线性几何下令 `B=A A^T`，把 zero-start CGLS
K4 改写成 detector-space conjugate-residual 递推，四步后只做一次 `A^T`。它不是
近似：另一套独立程序重新构造几何和 CSR、重新跑五条轨迹全部 `505` 帧，最大 field
差 `4.34e-16`、最大 residual 差 `1.44e-15`。

真正意外的是当前几何的 `B` 只有 `2.58%` 非零。CSR 内存从 `32.75 MiB` 降到
`1.27 MiB`，少 `96.11%`。于是每帧的昂贵调用可以从

```text
4 A + 4 A^T
```

改成

```text
4 sparse B + 1 A^T
```

为了确认这不是 Python 同进程计时幻觉，又完整运行了
`5 trajectories × 2 arms × 17 repeats = 170` 个新进程。最终轨迹等权端到端
wall ratio 为 `0.824652`，即典型快约 `17.5%`；RSS p90 ratio 为 `1.000865`，
基本不变。五条轨迹各自的 wall 中位数都改善。

**讲人话：**今天确实有可喜可贺的进展。这是目前少数把“结果相同、物理调用更少、
程序真的更快、内存没明显变坏”同时做实的一条路线。但它现在仍是固定几何 classical
control，不是神经网络创新；离线黑盒构造 `B` 的完整调用 break-even 约 592 帧，
本次 505 帧还没越过；也没有换相机几何、真实 BOST 或曲折光线。因此我把它标成
“重大代理机理正结果”，不写顶刊突破：

```text
major_proxy_mechanism_result=true
external_geometry_replay=false
real_BOST=false
operator_learning_result=false
global_novelty_proven=false
algorithm_breakthrough=false
paper_success=false
```

下一步不再回到无边界 CNN 海选。先用 BLASTNet 的不同几何从头构造新的 `B`，检查
稀疏性、机器精度等价和 fresh 加速是否一起保住。若它成立，再把 exact sparse B
作为物理核心，只学习几何变化、曲折光线或模型失配引起的小修正。

完整结果见
`docs/poolfire_c_sparse_detector_replay_v58_result_2026-07-30.md`。

## 2026-07-30：v59 外部坐标代数保住了，五帧部署资源没有保住

这次没有复用 PoolFire 的 `B` 或稀疏位置，而是从 BLASTNet H2-air 外部坐标重新
构造 `A`、`B=A A^T` 和 CSR。外部坐标的尺度和长宽比明显不同，但仍是同样的三轴
投影拓扑。

独立程序没有导入正式 replay core，重新构造全部矩阵并重算 5 个 observation。最大
field / residual 相对差为 `4.30e-15 / 6.03e-15`，CSR indices 与 indptr 完全
一致，非零比例仍为 `2.5782%`。这说明坐标尺度和长宽比变化没有破坏代数重放。

但正式资源门严格失败：

```text
17 repeats × 2 arms = 34 fresh processes
core compute ratio       0.809684
outer wall ratio         1.023894
peak RSS p90 ratio       1.153122
```

**讲人话：**真正算那五帧时，稀疏重放快约 19%；可是加载稀疏 `B` 的固定成本比
五帧节省的计算还多，内存也多用了约 15%。所以程序从启动到结束反而慢约 2.4%。
这不是“差一点通过”，而是 wall 和 RSS 两道冻结门都失败。

中间还抓到一个很重要的数学命名问题。子代理建议把 detector-space 递推改叫
CGNE，但重新手推并用随机矩阵逐步对照后发现，原来的 CR 才是对的：CR lift 与
CGLS 最大差 `3.0e-16`，标准 CGNE 与同一 CGLS 迭代差 `0.379`。因此撤回了错误
改名，并在文档中明确两套系数不同。

当前真实边界是：

```text
external_coordinate_algebra_transfer=true
fresh_resource_transfer=false
arbitrary_camera_geometry_transfer=false
operator_learning_result=false
real_BOST=false
algorithm_breakthrough=false
paper_success=false
```

下一步只做两件直接改变论文判断的事：先冻结 batch-length 摊销曲线，确定短序列
到长序列的真实资源转折；再换真正不同的视角拓扑，检查稀疏结构是不是三轴代理的
特例。exact sparse `B` 若继续成立，才把它作为固定物理核心，只学习几何变化或
曲折光线的小修正。

完整结果见
`docs/blastnet_sparse_detector_replay_v59_result_2026-07-30.md`。

## 2026-07-30：v60.1 不再存 B，101 次调用下端到端真正过门

v59 的失败很明确：稀疏 `B=A A^T` 在真正计算时快，但为了五帧任务把它从磁盘
加载进来，固定成本和内存开销反而把收益吃掉了。这次没有继续压缩 CSR，而是重新
拆了三轴投影算子。

每个视角都能写成：

```text
A_s = G_s P_s
```

`P_s` 是沿视线把三维场压成二维图，`G_s` 是在探测器平面求两个偏折分量。
因此任意源视角到目标视角的 `A_t A_s^T` 都可以先做二维梯度伴随，再做一次
LOS 权重收缩或外积，最后做二维梯度 forward。当前三正交轴结构下，整个过程
不需要保存 dense/CSR `B`，也不需要在每次 `B` 乘法中生成三维场。

独立程序没有导入正式 core 或 runner，重新做了全部计算：

```text
17 个随机 detector vector 的最大 B 作用差     2.92e-16
5 个 observation 的最大 K4 field 差           1.09e-15
5 个 observation 的最大 K4 residual 差        1.28e-15
persistent geometry factors                    45,568 bytes
```

每帧的完整物理算子账从：

```text
4A + 4A^T
```

变成：

```text
4 次二维 factorized-B + 1A^T
```

完整 `A/A^T` 从 8 次降到 1 次。正式 fresh 资源实验使用 17 次重复、两种算法和
5/101 两种负载，共 68 个新进程。结果很能说明问题：

```text
5 calls:
  core compute ratio   0.739354
  outer wall ratio     0.983849

101 calls:
  core compute ratio   0.716687
  outer wall p50       0.841184
  outer wall p90       0.876558
  outer worst          0.912475
  worker-self RSS      0.967493
```

**讲人话：**五帧仍然太短，程序启动和读文件几乎把计算收益吃完；连续处理一条
PoolFire 长度的 101 帧后，固定成本被摊薄，端到端典型时间真实下降约 15.9%，
而且 worker 自身内存没有变坏。101 次只是按固定顺序循环五个 observation 来测
资源，不是 101 个新样本，所以不能冒充泛化。

这是今天真正可喜可贺的进展：第一次在外部坐标代理上同时得到“不存 `B`、机器
精度等价、完整调用少 87.5%、fresh outer wall 通过、RSS 通过”。但它依赖
`x/y/z` 三正交轴拓扑，不是九视角相机、曲折光线、learned warm start 或真实
BOST：

```text
important_proxy_structure_result=true
arbitrary_camera_geometry_transfer=false
curved_ray_transfer=false
operator_learning_result=false
real_bost=false
global_novelty_proven=false
algorithm_breakthrough=false
paper_success=false
```

下一步不会回去无边界换网络。先把何远哲师兄 NeRIF 里更接近真实实验的九视角、
相机投影和校准射线放进同一资源合同，判断二维因子化能保留多少；只有剩余误差
确实来自相机/曲折光线并且 deployment-visible，才训练一个最小 correction
operator。这样网络负责的会是明确的物理缺口，而不是重新学一遍整个逆问题。

完整结果见
`docs/blastnet_factorized_detector_v60_1_result_2026-07-30.md`。

## 2026-07-30：v61 把精确结构从三轴扩到了九个平行视角

v60.1 的资源正结果仍可能只是 `x/y/z` 三正交轴的特殊现象。为避免把这个特例
写成一般算法，这次把视角改成九个覆盖 `0°–170°` 的平行投影视角，其中八个都
不是坐标轴方向。

正式实验先冻结角度、网格、detector、采样数、精度门和独立复算要求，再在
`8×8×8` 的小审计问题上显式构造 `A` 与 `B=A A^T`。这里允许显式矩阵只是为了
检验结构，不是部署实现。结果为：

```text
view/component blocks                         324
maximum sigma2 / sigma1                       3.20e-16
maximum rank-one block reconstruction error   7.54e-15
maximum factorized B action error             1.57e-15
maximum K4 field / residual error              1.98e-15 / 2.37e-15
independent validator maximum report drift     0
```

为什么会这样？这些相机只在 `xy` 平面内转动，detector 的竖直方向始终沿 `z`。
每个分量的算子都能拆成一维竖直算子与二维水平投影算子的 Kronecker 积：

```text
A_(theta,c) = Z_c ⊗ H_(theta,c)

B_((t,ct),(s,cs))
  = (Z_ct Z_cs^T) ⊗ (H_(t,ct) H_(s,cs)^T)
```

所以每个目标/源视角和 u/v 分量 block，在竖直×水平重排后必然只有一个非零
Kronecker 奇异值。独立程序没有导入正式 core 或 runner，重新构造了几何、
全部 324 个谱、17 个随机 `Bv` 和 5 个 K4，得到完全相同的报告。

**讲人话：**此前我们知道三台沿坐标轴看的理想相机可以省掉昂贵的三维往返；
现在知道九台从不同水平角度看的平行相机也保留同一个精确结构。这比“换个网络
再训练”更像一条可解释的方法主线，因为可精确计算的物理部分不需要交给网络猜。

但现在还不能说算法完成。小审计里的 factors 是先造出完整 `B`，再逐 block 做
SVD 提取的；这在真实尺寸上不可接受。`32× smaller` 只是小矩阵表示数字，不是
正式内存结论。针孔相机、roll/elevation、逐射线标定和曲折光线也可能破坏同一个
竖直因子：

```text
nine_view_parallel_algebra_transfer=true
scalable_factor_construction=false
online_resource_result=false
pinhole_camera_transfer=false
calibrated_camera_transfer=false
operator_learning_result=false
real_bost=false
algorithm_breakthrough=false
paper_success=false
```

下一步只做会改变论文判断的门：从一维竖直和二维水平 primitives 直接生成 factors，
在 `16×16×32` 上完全不构造 dense `A/B`，再用 101 次 fresh 调用公平比较
Zero-CGLS K4、matrix-free detector CR 和 analytic Kronecker replay。只有精度、
端到端时间与内存一起通过，才进入真实相机失配与最小 learned correction。

完整结果见
`docs/nine_view_parallel_detector_kronecker_v61_result_2026-07-30.md`。

## 2026-07-30：v62.2 把九视角结构变成了真正更快的可扩展数值核

v61 只在 `8³` 小问题上显式构造 `A/B`，证明九视角 detector-normal block 是
Kronecker rank one。那时还不知道能不能在正式粗网格尺寸上直接构造，也不知道
实际程序是否更快。

v62.2 已在 `16×16×32` 场上从一维竖直和二维水平 primitives 直接生成 factors，
全程没有形成 dense 三维 `A` 或 `B`。三条正式算法臂是：

```text
Zero-CGLS K4                  4A + 4A^T
matrix-free detector CR K4    4A + 5A^T
analytic-factor detector CR   4 analytic-B + 1A^T
```

先用三个 fresh correctness process 保存并比较五个完整三维场和 residual，不再
只比 norm：

```text
analytic maximum field difference       4.149e-15
analytic maximum residual difference    8.674e-15
matrix-free maximum field difference    4.364e-15
matrix-free maximum residual difference 7.818e-15
frozen threshold                        1.000e-10
```

然后运行 `2 workloads × 3 arms × 17 repeats = 102` 个串行 fresh timing
process。每个区组内三条臂相邻、顺序随机。101 次负载的关键结果是：

```text
analytic / Zero compute p50             0.172307
analytic / Zero outer p50               0.202526
analytic / Zero outer p90               0.205030
analytic / Zero outer worst             0.209606
analytic / Zero process-tree RSS p90    0.988063
matrix-free / Zero outer p50            1.053712
```

**讲人话：**只是把 CGLS 改写成 detector-space 递推并不会变快，matrix-free
control 反而慢约 5.4%。真正的收益来自解析 Kronecker factors 删除四轮昂贵的
三维 forward/adjoint 往返。连续 101 次处理时，fresh outer-wall 中位数从
`23.901 s` 降到 `4.850 s`，下降约 79.7%；RSS 只下降约 1%，因此速度结果很强，
内存只能写“不恶化”，不能写突破。

首轮也完成了 102 个 timing worker，但 controller 的相邻区间检查写错，在
controller 专有 wall/RSS 账本落盘前异常退出。因为这些外层计时无法从 worker
输出恢复，首轮被永久标为 invalid，102 个 worker 的资源数值全部禁止复用。修复
和回归测试后，从头重跑了第二轮 3 + 102 个进程。本文只使用第二轮。

独立 validator 没有导入正式 runner/core，重新聚合全部 worker、完整向量和 68
个配对行；batch、correctness、gate 和 CSV 的最大报告差都是 0，校验和全部通过。
但它没有再跑一套独立的 102-worker timing，所以这是独立聚合审计，不是第二台
机器的资源复现实验。

这是今天最重要的真实正结果，但证据边界仍然很硬：

```text
scalable_analytic_factor_construction=true
fresh_proxy_resource_gate=true
parallel_camera_transfer=true
pinhole_camera_transfer=false
calibrated_camera_transfer=false
curved_ray_transfer=false
operator_learning_result=false
real_bost=false
broad_generalization=false
algorithm_breakthrough=false
paper_success=false
```

101 次负载只是循环五个 seeded synthetic proxy fields，不是 101 个独立物理
样本。下一步最值钱的实验不再是重复调这个平行几何，也不是立刻堆大网络，而是
加入 pinhole、elevation、roll 和逐射线标定扰动，测量精确 analytic core 的失配
是否低秩、稳定、能由部署可见相机参数解释。只有这个答案为正，最小 learned
correction 才有清楚的物理对象。

完整结果、脱敏机器摘要和图表分别见：

- `docs/nine_view_analytic_factor_resource_v62_result_2026-07-30.md`
- `docs/nine_view_analytic_factor_resource_v62_public_summary.json`
- `assets/nine_view_analytic_factor_resource_v62.png`

## 2026-07-30：v63 先失败，v65 才找到低秩几何近似真正该做的工作

这次没有因为 v62.2 在平行相机上很快，就直接把“低秩修补”写成新算法。
先做的 v63 问得很严格：当相机出现有限源距、elevation、roll、焦距、主点和
目标偏移以后，能不能把平行核心 `A0` 加一个很小的 residual，直接当成真实
已知几何 `Ag` 来完成 K4？

答案是不能。18 个场景、22 个 truth、两族近似和 `q=1/2/4/8` 一共 13,032
个原子；`q=1/2/4` 全部没有通过，而且 `A0+residual` 比同 q standalone 更差。
clean-room NumPy 程序没有导入正式射线、三线性、梯度、分解或 runner，仍把
九张表和 FAIL 独立复现，最大差约 `1.03e-12`。

**讲人话：**一个近似相机算子不够准，就不能假装它是真相。继续加 rank 或堆
网络只会把问题藏起来，所以“近似算子直接替代真实物理”这条路线正式关闭。

v64 随后换了一个更合理的角色：近似算子只在 detector space 里找一个便宜起点，
然后必须经过一次真实 `Ag^T` 提升和一到两步未修改的真实 CGLS：

```text
y
 -> cheap detector CR with standalone Atilde
 -> dual proposal z
 -> x0 = Ag^T z
 -> exact Ag-CGLS K1 or K2
```

已开封场景出现全 cell 正信号后，没有继续在同一批数据上调。v65 在看到任何新
结果前固定了新的九视角角度、11 个几何 pattern、19 个随机 truth、5 个结构 truth、
两个候选和六项逐 cell 门。第一次正式运行和独立复算得到：

```text
formal atoms                                  1,848 / 1,848
q8 + exact Ag^T + K1                         264 / 264 PASS
q4 + exact Ag^T + K2                         264 / 264 PASS

q8-K1 exact cost                             2A + 2A^T
q8-K1 worst field/gradient/observation harm  0.985320 / 0.984208 / 0.952726
q8-K1 worst ratio vs Zero-K2                 0.855251 / 0.881628 / 0.827317

q4-K2 exact cost                             3A + 3A^T
q4-K2 worst field/gradient/observation harm  0.959885 / 0.955884 / 0.915029
q4-K2 worst ratio vs Zero-K3                 0.884484 / 0.896546 / 0.843994

independent maximum numeric difference       3.38e-13
```

这次所有比值都小于 1，不是平均值变好却藏着坏样本。q8-K1 在每个新 cell 上
都不劣于 Zero-K4，同时精确算子对从 4 个降到 2 个；q4-K2 从 4 个降到 3 个，
精度余量更大。两者还都逐 cell 击败相同精确调用数的 Zero-K2/Zero-K3。

**真正的新认识：**低秩几何近似不适合当最终物理模型，但可以提供普通
Zero-CGLS 前几步找不到的有用方向；真实 `Ag^T` 和真实 CGLS 再负责把结果拉回
物理一致空间。这比“训练一个网络直接猜密度场”更可解释，也更接近师兄说的
warm start。

但这还不是突破。当前是无噪声 `8×8×8` 已知 straight-ray 小预言机；cheap
factor 动作、factor setup、目标尺寸 wall 和峰值 RSS 都没有计入，也没有曲折
光线、真实相机标定、组内位移图或算子学习：

```text
fresh_known_geometry_small_oracle=true
target_scale_resource_result=false
wall_time_speedup=false
whole_pipeline_peak_memory_result=false
curved_ray_transfer=false
operator_learning_result=false
real_bost=false
algorithm_breakthrough=false
paper_success=false
```

下一步已经收窄：只把 q8-K1 和 q4-K2 扩到 `16×16×32`，逐相机/分量构造 factors，
把 setup、cheap actions、精确调用、fresh wall 和整条进程峰值内存一起算清；
再放到 PoolFire 和另一个公开反应流族。目标尺寸和公开外部族都过门之后，才研究
最小 geometry/observation-conditioned rank 或 coefficient predictor。

完整结果、脱敏摘要和图表：

- `docs/nine_view_geometry_warm_refinement_v65_result_2026-07-30.md`
- `docs/nine_view_geometry_warm_refinement_v65_public_summary.json`
- `assets/nine_view_geometry_warm_refinement_v65.png`

## 2026-07-30：v66.1 把小预言机正信号带到了目标尺寸，但还没有证明更快

这一轮没有换模型，也没有为了追求更好看的平均数重新挑帧。v65 固定下来的两条
方法原样扩到 `32×16×16` 九视角几何：

```text
q8: cheap detector CR -> exact Ag^T -> exact CGLS K1
q4: cheap detector CR -> exact Ag^T -> exact CGLS K2
```

数据是五条已经开封的公开 PoolFire 形态轨迹，每条固定取
`0 / 25 / 50 / 75 / 100` 帧，再乘三档相机几何扰动。每条候选共有 75 个单元，
加上五条 controls，一共 525 个正式原子。

第一版 runner 的数值虽然是正的，但连续红队审计找到了五类足以撤销正式资格的
问题：正式 commit/source closure 没锁死、真值没有从原始 `rho` 重建、调用账
仍可依赖声明值、存在绕过完整复算的发布入口、validation/READY 的原子发布次序
不够严格。旧授权已经在公开前撤销并归档，不能用于下面的结论。

v66.1 修复这五类问题，还加入了 A0 control 的全局支配性检查、轨迹等权汇总，
以及 q8/q4 分开的 factor 质量诊断。最后一次正式运行得到：

```text
q8 + exact lift + K1     75 / 75 PASS
exact cost               2A + 2A^T
worst harm vs Zero-K4    0.995624 / 0.999003 / 0.925687
worst vs Zero-K2         0.902464 / 0.984080 / 0.634459

q4 + exact lift + K2     75 / 75 PASS
exact cost               3A + 3A^T
worst harm vs Zero-K4    0.970006 / 0.991862 / 0.797811
worst vs Zero-K3         0.927389 / 0.985671 / 0.679828
```

**讲人话：**近似得并不非常准的 q4/q8 算子，仍能比普通 Zero-CGLS 更早找到
有用方向；但它不能单独完成重建，必须由冻结的 noise-free straight-ray proxy
下的精确 `Ag^T` 和未修改 CGLS 接管。

修正后的独立程序从已提交、detached、tracked-clean 的 checkout 运行，并恢复
正式 runner 的 Python/NumPy/PyTorch 数值环境。它没有导入正式 runner、v66
factor core、v63 geometry core 或 v62 analytic core，而是从相机参数和随机种子
重新搭建数值路径；同时从五条原始 `rho` 重新做轴翻转、固定 ROI、`2×2×2`
block-mean 和全局均值规范化。五条轨迹合计 505 帧、4,136,960 个真值数与 pair
真值逐值完全相同，最大差 0；随后复算 525 个原子，6,150 个行级数字最大差
`8.88e-14`，1,062 个 factor 数字最大差 `1.85e-14`，所有 mismatch 为 0。
五条 raw/pair 输入和正式结果在复算前后也没有变化，validation/test truth 未打开。
这套程序仍共享冻结的体素梯度算子和三线性 stencil，所以它是独立重算，不是
端到端物理独立。

这次真正推进了论文判断：**公开 CFD straight-ray 形态代理上的目标尺寸 Stage A
已经通过，五条轨迹合计 505 帧的 Stage B 可以开始。**但它仍没有回答 setup
加进去以后是否更快、整条进程是否更省内存，也
没有曲折光线、真实相机、组内位移图或外部未开数据：

```text
full_trajectory_stage_b_authorized=true
fresh_resource_stage_c_authorized=false
operator_learning_result=false
real_bost=false
algorithm_breakthrough=false
paper_success=false
```

完整结果、脱敏机器摘要和图表：

- `docs/nine_view_poolfire_target_scale_warm_v66_result_2026-07-30.md`
- `docs/nine_view_poolfire_target_scale_warm_v66_public_summary.json`
- `assets/nine_view_poolfire_target_scale_warm_v66.png`

## 2026-07-30：v67.1 全 505 帧通过，终于排除了“只挑到五个好帧”

v66.1 只在每条轨迹的五个冻结时间点上通过。它是重要正信号，但仍可能有一个
很朴素的解释：方法恰好避开了中间更难的帧。因此这轮没有换模型，而是把同一
q8-K1 / q4-K2 原样扩到五条轨迹全部 505 帧和三档已知九视角几何。

正式运行前放入了 16 个对照：Zero K1-K4、scaled exact BP、Jacobi-PCGLS、两种
A0 proposal，以及分别选择 lambda 的 leave-one-trajectory-out dual/direct ridge。
第一版长任务刚启动、尚未产生任何结果时，独立红队发现四个会影响科学判决的
问题：三种几何的 lambda 尾部被 pooled、direct-field 沿用 dual lambda、validator
没有从实际调用点独立计数、`equal-or-cheaper` 用词超过了 exact-call 证据。
我立即停止了空结果任务，没有读取任何数字，再把协议修成 v67.1：

```text
lambda selection  = max over geometry × metric × reference tails
dual lambda       = selected independently
direct lambda     = selected independently
validator ledger  = counted at actual forward / adjoint call sites
cost claim        = exact A/A^T budget only
```

修正版第二轮只读红队得到 P0=0、P1=0，随后唯一正式任务运行约 31.6 分钟，
原子写出 18 臂 × 1,515 单元 = 27,270 个结果。正式 runner 自报 PASS，但当时
仍没有把它当结论；另一个 detached clean checkout 又从 raw rho 重建并重跑全部
arms，最终独立裁决也是 PASS：

```text
q8 + exact A^T + CGLS K1     1,515 / 1,515 PASS
q4 + exact A^T + CGLS K2     1,515 / 1,515 PASS

q8 exact cost                2A + 2A^T
q8 worst harm / Zero-K4
  field                      1.002008
  gradient                   0.999111
  interior gradient          1.001399
  observation                0.929281

q8 worst / equal-call Zero-K2
  field                      0.910587
  gradient                   0.986495
  interior gradient          0.953952
  observation                0.644986
```

**讲人话：**q8 在个别帧的 field 和内部梯度比跑满四步的 Zero-K4 略差
0.20% / 0.14%，但没有超过事前允许的 1% 等价带；它对同样只用两对精确算子的
Zero-K2，则每一帧四项都更好。更强的是，16 个 controls 中没有任何一个在任何
单元同时压过 q8 的四项指标。不是平均数好看，也不是靠某条轨迹撑起来：15 个
trajectory-by-geometry 层各自都是 101/101 PASS。

独立程序比较了 534,795 个逐行数字，最大差 `6.43e-10`；14,136 个汇总和判决
数字最大差 `2.55e-12`；selection 最大差 `2.27e-13`；实际调用总账差为 0。
正式 `rows.jsonl` 另做外部完整性核对，恰好 27,270 行、27,270 个唯一 cell ID、
重复 0。raw/pair 输入和正式 payload 在验证前后均未变化，两条 test 没有打开。

这次是真正值得高兴的进展，因为它排除了“挑帧”“多调用”“简单 ridge 足够”
和“便宜 control 全局支配”四个解释，并把 q8-K1 推进到了 fresh 资源门。但还
不能说算法突破：

```text
full_trajectory_result=true
fresh_resource_stage_c_authorized=true
fresh_wall_speedup=false
whole_pipeline_peak_memory_result=false
independent_public_family_transfer=false
neural_operator_result=false
curved_ray_transfer=false
real_bost=false
algorithm_breakthrough=false
paper_success=false
```

q8 目前是固定 geometry-compressed factor，不是神经网络。下一步只做一件事：
把 factor setup 和八次 cheap factor actions 都放进 fresh worker，用至少 11 个
随机相邻完整区组公平比较 q8-K1 与 Zero-K4。只有 outer wall、逐轨迹不变慢和
whole-pipeline RSS 一起通过，才把“50% 少 exact calls”升级成“真实资源收益”。

完整结果、脱敏机器摘要和图表：

- `docs/nine_view_poolfire_full_trajectory_v67_result_2026-07-30.md`
- `docs/nine_view_poolfire_full_trajectory_v67_public_summary.json`
- `assets/nine_view_poolfire_full_trajectory_v67.png`

## 2026-07-31：v68.3 真的快了约 34.5%，但内存把整道资源门否决了

v67.1 已经回答了“精度能不能守住”：固定 q8-K1 在五条 PoolFire 轨迹全部
505 帧与三档已知九视角几何上通过，精确调用账从 Zero-K4 的
`4A+4A^T` 降为 `2A+2A^T`。这次不再调 rank、不挑帧，也不训练新网络，只把
两条方法放进全新的进程里公平计时和测内存。

正式批次包括：

```text
reference workers           30
timing workers              330
random adjacent pairs       165
trajectories × geometries   5 × 3
```

结果很清楚，而且是一个“半边成功、整体失败”的结论：

```text
q8 / Zero outer wall
  p50                       0.655236   PASS
  p90                       0.661113   PASS
  worst                     0.671715

q8 / Zero RSS p90
  worker self               1.384474   FAIL
  sampled worker tree       1.376802   FAIL
  sampled pipeline          1.311116   FAIL
```

**讲人话：**q8 每次重建大约从 15.67 秒降到 10.27 秒，五条轨迹都稳定快，
所以“少一半精确物理调用能不能真的省时间”这件事已经得到正答案。但 q8 为了
这份速度，factor 构造阶段出现了很高的瞬时工作区；worker 自身的 p90 RSS
大约多了 38.4%，连 controller+worker 的采样下界也多了 31.1%。合同要求
时间和内存同时过门，所以不能挑速度宣布成功：

```text
fresh_wall_gate_pass=true
rss_gate_pass=false
stage_c_pass=false
algorithm_breakthrough=false
paper_success=false
```

独立程序重新播放了 30 个 reference、3,030 帧，核对 30/30 canonical digest
和实际调用账；又重算 165 个配对区组与 903 个聚合数字，最大差全部是 0。
validator 中有一条计时边界相差约 `6.0e-11 s`，小于系统时钟约
`4.17e-8 s` 的分辨率。修复只允许这一类时钟量化差，没有改正式回执、结果
数字或科学阈值；随后只读审计确认原子发布有效。

我又检查了 165 个正式 candidate 回执。最终保留的 q8 factors 只有
`5,899,392 bytes`（约 5.63 MiB），最大显式 rearranged block 是
`33,554,432 bytes`（32 MiB）；它们都远小于约 167.6 MiB 的 worker-self p90
增量。高水位还包含第二份 ray bundle、block 构造和 randomized-SVD 临时数组。
所以准确瓶颈不是“常驻 factors 太大”，而是 **factor setup 与瞬时工作区**。

这不是突破，但比继续盲目训练模型更有价值：问题已经从模糊的“算法到底行不行”
收窄到一个具体瓶颈：

> 用 tiled / streamed 构造、工作区复用、ray-bundle 共享，或低精度存储加
> 高精度累积，尽量去掉 factor setup 的约 168 MiB 高水位差，同时保住
> v67.1 精度、50% 精确调用减少和约 34.5% wall 收益。

已有 tiled 小网格原型只能证明代数可行，synthetic 整进程内存曾下降，但 factor
构造和 outer wall 反而变慢；它还不是科学结果。下一版不会直接把它包装成修复，
而是先比较 compact、chunked、streamed 和 mixed-precision 几种表示的
memory-time Pareto，再让唯一候选重新通过完整 v67.1 Stage B 与同一 v68.3
fresh 门。大网络继续不授权。

完整结果、脱敏机器摘要和图表：

- `docs/nine_view_poolfire_fresh_resource_v68_result_2026-07-31.md`
- `docs/nine_view_poolfire_fresh_resource_v68_public_summary.json`
- `assets/nine_view_poolfire_fresh_resource_v68.png`

## 2026-07-31：v70 把内存候选救回了精度门，1,515/1,515 通过

v68.3 已经把问题说得很具体：旧 q8-K1 真能快约 34.5%，但 setup 期间的重复
ray bundle、32 MiB block 和 randomized-SVD 工作区让 RSS 多 31%-38%。
这次没有继续训练网络，而是直接改 factor 构造：

```text
旧：exact stencil + 第二份 ray bundle + 完整 block + p2 SVD
新：复用 exact stencil + detector-v 四行 tile + p0 SVD
```

先做了三个随机完整区组的开发筛选。新 tiled p0 相对旧 dense q8 的
worker-tree RSS 中位比是 `0.80093`，约少 19.9%；wall 中位比是 `1.05095`，
只慢约 5.1%。这还不是正式资源结果，但它是唯一有机会同时修复 RSS、又保留旧
q8 对 Zero-K4 大幅 wall headroom 的候选，所以在看精度结果前把参数固定下来。

随后重跑与 v67.1 完全相同的全轨迹精度门：

```text
5 trajectories × 3 geometries × 101 frames = 1515 cells
candidate pass                                1515 / 1515
exact budget                                  2A + 2A^T
failed cells                                  0
```

最差的 Zero-K4 harm 是 interior-gradient 的 `1.00281`，仍低于冻结的
`1.01`；相对同调用 Zero-K2，四项 worst 全部低于 1。独立程序没有复用 tiled
builder，而是用旧 dense p0 重新构造 factors，再重算全部 1,515 个单元；最大
指标差只有 `5.68e-14`，通过判决完全一致。

**讲人话：**低内存版本没有为了省内存把重建精度弄坏。这是今天最值得高兴的
真实进展，也确实改变了下一步：v70 已获得重新跑 fresh wall/RSS 的资格。

但它现在仍不是突破：

```text
stage_b_pass=true
fresh_resource_stage_authorized=true
fresh_resource_result=false
algorithm_breakthrough=false
paper_success=false
```

下一步不再改 rank、tile 或门槛，只按 v68.3 同一口径重跑 30 个 reference、
330 个 timing worker、165 个随机相邻区组。wall 与三种 RSS 必须一起过门，
才能说旧 q8 的“快但吃内存”被真正修好。

完整结果、脱敏机器摘要和图表：

- `docs/nine_view_stencil_factor_stageb_v70_result_2026-07-31.md`
- `docs/nine_view_stencil_factor_stageb_v70_public_summary.json`
- `assets/nine_view_stencil_factor_stageb_v70.png`

## 2026-07-31：v70.1 速度仍过门，但内存高尾让 Stage C 再次失败

v70 的低内存构造已经在 1,515/1,515 单元守住精度。这次把它和 Zero-K4 放进
与 v68.3 同口径的正式 fresh 批次：30 个 reference、330 个 timing worker、
165 个随机相邻区组，factor setup 与全部计算都包含在 worker 内。

```text
tiled p0 q8 / Zero-K4

outer wall ratio
  p50 / p90 / worst       0.670558 / 0.677727 / 0.701055   PASS

worker-self RSS ratio
  p50 / p90 / worst       1.086393 / 1.134775 / 1.171482   FAIL
worker-tree RSS p90       1.132203                          FAIL
pipeline RSS p90          1.064684                          FAIL
```

讲人话：时间收益是真的。candidate 的全局中位时间约 10.91 秒，Zero 约
16.28 秒，五条轨迹都快约 33%。但正式合同不是“只要快就行”；三类内存高尾
都超过 1.05，所以 Stage C 整体仍是 FAIL。

独立程序从 raw pair 重建 1,515 个 observation，重放 30 个 reference 共
3,030 帧，再重算 165 个区组和 903 个聚合数字；最大差全部为 0。它也核对了
v70 父结果仍是 1,515/1,515 PASS。准确边界：

```text
PASS_INDEPENDENT_VALIDATION_TILED_EXACT_P0_FRESH_RESOURCE_V70_1
FAIL_TILED_EXACT_P0_FRESH_RESOURCE_STAGE_C_V70_1
algorithm_breakthrough=false
paper_success=false
```

我没有立即再烧一批正式 worker，而是先问“工作区再减半是否真有资源
headroom”。tile4 改为 tile2 后，三档几何的 factor/forward/adjoint 相对差都在
约 `1.6e-13` 以内，声明工作区从 12 MiB 降到 6 MiB。随后冻结一条 101 帧
observation stratum、六个 fresh 配对区组：

```text
tile2 / Zero-K4 outer wall p50 / p90   0.661271 / 0.678067   PASS
tile2 / Zero-K4 RSS p50 / p90          1.040516 / 1.060203   FAIL
```

独立重算与正式摘要最大差为 0。内存中位已经过门，但保守高尾仍超出约 1.02 个
百分点；按事前规则不降门、不扩到 330 worker，tile-size tuning 到此关闭。

下一步不再抠 tile。真实 BOST 装置的几何在标定后固定，应把一次性 calibration
factor build 与在线 observation-stream reconstruction 分成两张账，同时报告
cold cost、online wall/RSS、artifact 大小和摊销临界点。只有在线账独立通过，
才能谨慎说“固定标定几何下的在线重建资源收益”；离线成本不会被隐藏。

完整证据与脱敏摘要：

- `docs/nine_view_stencil_factor_fresh_resource_v70_1_result_2026-07-31.md`
- `docs/nine_view_stencil_factor_fresh_resource_v70_1_public_summary.json`
- `assets/nine_view_stencil_factor_fresh_resource_v70_1.png`

## 2026-07-31：v72 第一次把在线速度和内存两扇门同时推开

v70.1 / v71 已经把问题定位得很具体：q8-K1 的精度和调用减少都成立，时间也
稳定更快，但每个 fresh worker 重新构造 factor 会留下过高的内存高水位。继续
缩 tile 已经没有足够余量，所以这次没有再改算法，而是换成更符合固定装置的
执行方式：

```text
冻结的已知合成几何
-> 离线编译 5.63 MiB q8 factor artifact
-> 在线 fresh worker 校验并只读映射 artifact
-> exact A^T lift
-> unchanged CGLS K1
```

先做了硬等价检查。18 个 view-component block 写入、重开以后，逐数组最大差、
随机 forward 相对差、adjoint 相对差全部是 `0`。完整离线 fresh process 用时
`2.2163 s`，peak RSS `446.20 MiB`，没有读取 observation、truth 或 raw rho。

然后在一条已经开封的 101 帧 observation stream 上，按结果前写死的顺序跑六个
相邻 fresh 配对区组：

```text
loaded q8-K1 / Zero-K4

outer wall p50 / p90 / worst
  0.566428 / 0.583830 / 0.583830        PASS

worker-self RSS p50 / p90 / worst
  1.002156 / 1.029576 / 1.029576        PASS
```

候选在线完整时间 p50 为 `8.6864 s`，Zero-K4 为 `15.3683 s`；候选依旧只用
`2A+2A^T`，对照为 `4A+4A^T`。完整离线成本没有被藏起来：按当前 stream 的
中位节省量，`2.2163 s` 在一条 101 帧序列内即可摊销。

独立程序没有导入正式 probe/controller，重新哈希 54 个数组文件，核对 12 个
worker 的实际调用账和读取边界，再从原子回执重组六个区组并独立算分位数与
break-even。正式与独立结果最大差为 `0`：

```text
PASS_INDEPENDENT_RECOMPUTATION_CALIBRATED_FACTOR_ONLINE_V72
PASS_CALIBRATED_FACTOR_ONLINE_HEADROOM_PROBE_V72
```

**讲人话：**这是真正值得高兴的阶段性正结果。此前我们一直是“速度过门、内存
不过门”，现在第一次在固定几何在线口径下把两扇门同时推开了，而且不是靠放宽
阈值或隐藏离线成本。

但它还不能叫突破。这里只是一条已打开轨迹和一个已知合成几何；没有标定图像、
相机参数估计或漂移测试，所以不是“完成相机标定”。loaded artifact 还必须重新
跑三档几何、五条轨迹、全部 1,515 个精度单元：

```text
formal_loaded_artifact_stage_b=false
formal_online_stage_c=false
camera_calibration_result=false
algorithm_breakthrough=false
paper_success=false
```

下一步只做完整 loaded-artifact Stage B。只有 `1515/1515` 再次通过，才允许
把同一路径放进多轨迹正式 fresh resource Stage C；不会提前训练大网络，也不会
把这次单 stream 开发门包装成论文成功。

完整证据、脱敏摘要和图表：

- `docs/nine_view_calibrated_factor_online_v72_result_2026-07-31.md`
- `docs/nine_view_calibrated_factor_online_v72_public_summary.json`
- `assets/nine_view_calibrated_factor_online_v72.png`

## 2026-07-31：v73 把 loaded artifact 的完整精度风险关掉了

v72 的结果很鼓舞人：固定几何的 factor 不再每个 worker 现做，而是离线做好、
在线只读加载以后，一条 101 帧 stream 的时间和内存第一次同时过门。但这还缺一
块关键证据：加载和序列化会不会在其他几何、其他轨迹上悄悄改变数值？

所以这次没有继续改模型，也没有换门槛。正式实验只做一件事：把三档几何各自
编译成只读 q8 工件，再用真正的加载路径跑完五条 PoolFire fit trajectory 的
全部 101 帧。

```text
3 geometries × 5 trajectories × 101 frames = 1515 cells

compatibility PASS                    1515 / 1515
loaded reproduces canonical v70       1515 / 1515
three artifacts                       162 arrays
maximum array difference              0
maximum metric difference             0
exact online budget                   2A + 2A^T
```

相对四步 Zero-CGLS 的 field / gradient / interior-gradient / observation
最坏误差比是 `1.00252 / 0.99949 / 1.00281 / 0.93356`，全部低于事先固定的
`1.01` 门。相对同样只花 `2A+2A^T` 的 Zero-CGLS K2，四项最坏比值也都低于
1，所以这不是用更多精确算子换来的结果。

我还用一套不导入正式 runner 和 loader 的程序重新验了一遍。它自己解析 NPY
头和数据区，拒绝 object、Fortran-order、尾随字节和文件集合漂移，再重算全部
1,515 个单元。正式与独立指标最大差仍然是 `0`；相对独立 dense p0 reference
的 forward / adjoint 最大误差只有 `1.23e-15 / 1.24e-15`。

**讲人话：**v72 回答的是“单条开发序列看起来能更快且不更吃内存”；v73 回答
的是“把方法真正做成可加载工件以后，完整五条轨迹的精度没有坏”。这两个答案
现在能接起来了，下一步终于可以正当地跑多轨迹 fresh 资源门。

这是一个真实、扎实的阶段性进展，但还不是突破：

```text
loaded_artifact_full_trajectory_stage_b_pass=true
formal_multi_trajectory_online_stage_c_authorized=true
fresh_wall_speedup=false
whole_pipeline_rss_advantage=false
external_family_transfer=false
real_bost_result=false
algorithm_breakthrough=false
paper_success=false
```

下一步只比较 loaded q8-K1 与 Zero-K4 的多轨迹 fresh wall 和三层 RSS，把工件
加载、校验、worker 启动和整条 101 帧序列都算进去。wall 与 RSS 必须一起过，
否则就老实记录失败，不会用更大网络或临时降低门槛挽救。

完整证据、脱敏摘要和图表：

- `docs/nine_view_loaded_artifact_stageb_v73_result_2026-07-31.md`
- `docs/nine_view_loaded_artifact_stageb_v73_public_summary.json`
- `assets/nine_view_loaded_artifact_stageb_v73.png`

## 2026-07-31：v74.1 把完整多轨迹时间与内存资源门一起推开了

v73 证明了加载工件以后精度不坏，但还不能回答一个更现实的问题：把 Python
进程启动、工件加载和校验、101 帧完整重建都算进去，它是否真的比 Zero-CGLS
K4 更快，并且不会用内存高尾换速度？

第一次正式运行 v74.0 完成了全部 worker，却在独立验证时发现一份回执的浮点
耗时比同源整数纳秒耗时短 `0.056 ns`。旧合同要求它不能更短，所以整批被判
invalid。没有挑数字，也没有复用旧 worker；v74.1 先把时钟口径修成“同源差异
超过 1 微秒即拒绝，计分取两者最大值”，然后从零重新运行：

```text
reference workers        30
fresh timing workers     330
paired blocks            165
trajectories             5
known geometries         3
frames per worker        101
```

独立 validator 重建了 1,515 帧 observation，重放 30 个 reference 共 3,030
帧，检查 360 份回执与 165 个随机相邻完整区组。所有 reference digest 和实际
调用账都一致，norm-sum 最大差为 `0`。

正式结果是：

```text
candidate exact budget        2A + 2A^T
Zero-K4 exact budget          4A + 4A^T

fresh wall ratio
  p50 / p90 / worst           0.56507 / 0.57440 / 0.63788

RSS ratio p90
  worker-self                 1.03456
  sampled worker-tree         1.01971
  sampled pipeline            1.01409
```

候选的 fresh wall 中位数是 `8.9251 s`，Zero-K4 是 `15.7548 s`，也就是中位
约下降 `43.49%`。五条轨迹与三档几何的 p50 wall 和 RSS 资源门全部通过。
这次不只是“理论少算两步”，而是完整 fresh worker 的真实计时正结果。

**讲人话：**此前我们知道候选“答案没坏、精确算子少一半”，现在第一次又确认
“把加载和启动都算进去，确实稳定更快，而且预注册的内存高尾门也过了”。这把
PoolFire 代理内的精度、调用和资源三层证据接起来了。

但不能把好消息说大：

- worker-self 和 worker-tree 的单次 worst ratio 仍为 `1.08731 / 1.07281`；
  通过的是预先固定的全局 p90 与分层 p50 门，不是每一次都更省内存；
- RSS 只保存 coverage summary 与峰值回执，没有保存 raw sampling trace，
  所以独立程序不能重新生成操作系统采样轨迹；
- 当前仍是已开封 PoolFire fit family、已知几何、无噪声 straight-ray 代理；
- 没有独立公开反应流族、curved ray、相机标定、真实 BOS 位移图或组内重复测量；
- q8 factor 是固定几何压缩工件，不是训练得到的 neural operator。

所以准确状态是：

```text
loaded_artifact_fresh_resource_stage_c_pass=true
exact_operator_pair_reduction_fraction=0.50
median_fresh_wall_reduction_fraction=0.4349
public_proxy_sampled_rss_gate_pass=true
external_family_transfer=false
curved_ray_result=false
real_bost_result=false
operator_learning_result=false
algorithm_breakthrough=false
paper_success=false
```

下一步只做结果前冻结的独立公开反应流族外门。外部数据不能参与当前 factor、
门槛或候选选择；它必须重新通过 matched accuracy、精确调用、fresh wall 与
三层 RSS，之后才有资格申请组内真实位移图与标定数据。

完整证据、脱敏摘要和图表：

- `docs/nine_view_loaded_artifact_fresh_resource_v74_1_result_2026-07-31.md`
- `docs/nine_view_loaded_artifact_fresh_resource_v74_1_public_summary.json`
- `assets/nine_view_loaded_artifact_fresh_resource_v74_1.png`

## 2026-07-31：v75 第一次真正跨反应流族，完整合同只有 5/75

v74.1 把 PoolFire 内部的精度、调用、时间和内存接成了完整正结果。最危险的
问题也随之变得很明确：这个固定几何 q8 factor 会不会只适合 PoolFire 形态？

所以 v75 没有再看一条 PoolFire，也没有先看 BLASTNet 图像再挑容易的工况。
它在读取任何网格或密度数值前，按“至少 25 个有序快照、rho 字节数最小、
case id 最小”选定 vitiated H2-air Case 3。正式一次性开封后得到：

```text
25 frames x 3 geometries = 75 cells

complete frozen contract       5 / 75
all four Zero-K4 harm gates   74 / 75
resource stage                not authorized
```

真正的失败点不是 field 或 observation：

```text
gate pass counts                vs Zero-K4    vs equal-call Zero-K2
field                              75/75              75/75
full gradient                       75/75              46/75
interior gradient                   74/75               5/75
observation                         75/75              75/75
```

相对同样只用 `2A+2A^T` 的 Zero-K2，候选 field p50 为 `0.96946`，
observation p50 为 `0.62754`，都明显更好；但 interior-gradient p50 / p90 /
worst 是 `1.01501 / 1.02677 / 1.04685`。也就是说，固定 factor 很会保低频场
和观测残差，却没有保住外部反应流的局部梯度。

boundary-shell gradient error / full-gradient error 的中位达到 `0.94879`，
所以只看 full-gradient 很容易被边界壳掩盖；独立 interior-gradient 门确实
抓到了 aggregate 指标看不见的失败。5 个完整通过单元也只出现在最早的 frame
index 0 和 2，后续 22 帧全部没有通过。

独立 validator 重新哈希 29 个原始文件，不导入正式预处理、runner 或高层
solver，自行重建 25 个真值、75 个 observation 和三条臂。正式与独立指标
最大绝对/相对差为 `2.13e-14 / 4.71e-16`，所以这不是偶然的脚本漂移。

**讲人话：**这是一次有价值的负结果。PoolFire 上的 `1515/1515` 和资源 PASS
都是真的，但“固定 loaded factor 原样跨族”不成立。不能拿 `74/75` 的 K4
近似信号改写预注册的 `5/75 FAIL`，也不能换 Case 4/6 重跑同一个候选。

下一步不直接上大网络。先在已经开封的 Case 3 上检查同样 `2A+2A^T` 预算内，
当前 warm-CGLS 已经产生的二维可观测子空间是否存在 truth-aware oracle
headroom。没有就关线；有才训练最小 observation-only 系数预测器，并把另一个
未打开工况留给新算法，而不是替换 v75 负结果。

```text
external_accuracy_pass=false
resource_stage_authorized=false
unchanged_factor_cross_family_transfer=false
adaptive_same_budget_initializer_impossible=false
algorithm_breakthrough=false
paper_success=false
```

完整证据、脱敏摘要和图表：

- `docs/nine_view_vitiated_h2air_external_accuracy_v75_result_2026-07-31.md`
- `docs/nine_view_vitiated_h2air_external_accuracy_v75_public_summary.json`
- `assets/nine_view_vitiated_h2air_external_accuracy_v75.png`

## 2026-07-31：v76 证明“把两个系数交给更强网络”也救不回完整合同

v75 失败后，最容易产生的错觉是：当前 `q8-K1` 只是系数选得不好，换一个
MLP、FNO 或 DeepONet 预测两个更聪明的系数，也许就能把 `5/75` 救回来。

v76 没有直接训练网络，而是先让一个能看见真值的 oracle 搜索这两个系数的
全部可能性。表示保持同一个精确调用预算：

```text
h = A^T z
n = A^T (y - A h)
x(c) = c1 h + c2 n

exact budget = 2A + 2A^T
coefficient search = 0 additional exact calls
```

每个单元都要同时满足相对 Zero-K4 的四项 `1.01` no-harm 门，以及相对同调用
Zero-K2 的四项 `1.00` 门。结果是：

```text
strict feasible witnesses          58 / 75
exact dual infeasibility proofs    17 / 75
numerical inconclusive              0 / 75
```

**讲人话：**58 个单元说明“调系数”确实比当前方法强很多；但 17 个单元不是
“优化器没搜到”，而是数学证书证明这两个方向形成的二维平面与合格区域没有
交点。既然答案根本不在平面里，再大的系数网络也不可能把它预测出来。

每档几何都有失败：F12+/F15+/F30+ 分别为 `5/7/5` 个严格不可行。oracle 在
全部 75 个单元守住 field，且相对 K4 的 full/interior gradient 也全过；真正
冲突是为了改善同调用 K2 gradient，必须牺牲 K4 observation。17 个不可行
单元的 minimax 点全部违反 K4 observation，其中 12 个同时违反 K2
interior-gradient，8 个同时违反 K2 full-gradient。

正式执行后，第一版 validator 因为要求独立重建后的 `Fraction` 分子分母逐字
相同而 fail closed。不同浮点运算顺序在末位有约 `1e-15` 差，转成精确分数后
文本当然不同。v76.1 没有改任何结果、门槛或单元，只把验证条件修成真正需要
的东西：正式路径和独立路径都要用同一整数 simplex 权重，各自重建精确正定
二次式，并在不可行单元上各自得到严格正的 exact lower。

```text
formal/independent max difference       4.7962e-14
exact lower rebuild difference          6.2970e-16
determinant relative rebuild difference 2.7871e-15
```

所以这次的失败是可信的，而且直接节省了后续训练成本：

```text
span_h_n_closed=true
larger_network_on_same_span_authorized=false
case4_or_case6_opened=false
resource_stage_authorized=false
algorithm_breakthrough=false
paper_success=false
```

下一步不能再把算力投给“两系数预测器”。必须先让表示跳出这个二维平面，例如
由只看部署可见输入的小型三维网络产生空间变化 correction：

```text
x0 = h + u_theta(h, geometry-visible features)
r0 = y - A x0
x1 = one unchanged CGLS step from x0
```

这样仍可能保持 `2A+2A^T`，但 `u_theta` 可以直接修局部梯度，而不是用两个
全局标量在 observation 与 gradient 之间来回拉扯。新表示仍要先过
truth-aware capacity gate；Case 3 从此只作 development，另一个未打开工况
继续保留为一次性外部门。

完整证据、脱敏摘要和图表：

- `docs/nine_view_obs2d_oracle_v76_result_2026-07-31.md`
- `docs/nine_view_obs2d_oracle_v76_public_summary.json`
- `assets/nine_view_obs2d_oracle_v76.png`

## 2026-07-31：v77 第一次真的做了空间修正，但 GSLB8 只有 7/75

v76 关掉的是两个全局系数。它告诉我们：不管系数网络多强，只要最后答案仍
被限制在 `span{h,n}`，17 个单元就严格没有解。所以 v77 没有再训练一个更大
MLP，而是先换表示。

这次把三维场内部放了一套很粗的 spline 控制格，并用边界置零与零均值约束
去掉不合理自由度。每档冻结几何再按“同样的内部梯度能量，谁最不容易被
observation 看见”排序，取前 8 个空间模态：

```text
x0 = h + U_g,8 a
x1 = one unchanged CGLS step from x0
```

这里的 `a` 仍由看得见真值的 oracle 选择。原因很简单：如果 oracle 都找不到
合格系数，就不值得先训练一个只看 observation 的网络。

正式结果：

```text
完整八门见证       7 / 75
冻结搜索 negative 68 / 75
数值不确定          0 / 75
```

真正有信息的不是只有“7/75”，而是各门的拆解：

```text
field / K4, K2             75 / 75, 75 / 75
observation / K4, K2       75 / 75, 75 / 75
full-gradient / K4, K2     56 / 75, 27 / 75
interior-gradient / K4, K2 74 / 75,  7 / 75
```

所以空间变化 correction 的方向不是完全错。它把 observation residual 压得
很明显，field 也全部优于同调用 K2；但 8 个模态没有足够容量恢复局部梯度。
对 BOST 来说，梯度不能被当成次要指标，因为背景点位移正是由折射率梯度驱动。

这次也要准确区分 v76 与 v77：

- v76 的 17 个失败有数学上的 exact infeasibility certificate；
- v77 是非凸搜索，68 个 negative 只表示冻结的 12 个起点和条件重启没有找到
  见证，不证明 GSLB8 在数学上绝对无解；
- 但结果前合同要求 75/75 才允许训练，所以 7/75 已足以停止 GSLB8 predictor。

独立 validator 没有导入正式 runner、optimizer helper 或 spline/mode helper，
重新构造三档模态并重跑全部声明起点与条件重启：

```text
maximum metric difference       1.7764e-14
maximum gate difference         1.2023e-11
stable projector distance       2.0165e-13
formal/raw payload unchanged    true
```

中间暴露了三处数值/验证问题：一次有限非成功 endpoint 的确定性重启、一个
过严的 provenance gate 比较，以及 `sqrt(1-sigma^2)` 在子空间几乎相同时的
消减误差。每次只修明确缺陷，数据、表示、rank、系数球、起点、八门、controls
和调用账都没动；最后从头重跑 formal 与独立验证，没有复用旧行。

现在的结论不是“空间修正失败”，而是：

```text
GSLB8 closed under frozen search
GSLB32 capacity stage authorized
neural training not authorized
resource stage not authorized
algorithm_breakthrough=false
paper_success=false
```

下一步只把同一套嵌套模态从 8 增到预注册的 32，仍保持相同 75 个 development
单元、相同八门和 `2A+2A^T`。如果 32 模态仍不能 75/75，就继续按合同判断
表示容量，而不是靠更大网络掩盖。

完整证据、脱敏摘要和图表：

- `docs/nine_view_geometry_spline_low_observability_v77_result_2026-07-31.md`
- `docs/nine_view_geometry_spline_low_observability_v77_public_summary.json`
- `assets/nine_view_geometry_spline_low_observability_v77.png`

## 2026-07-31：v78 让 GSLB32 的 75/75 表示容量门真正成立

v77 只有 7/75，不是因为空间修正这个大方向完全错了，而是前 8 个
low-observability modes 没有足够的局部梯度容量。v78 没有换数据、放宽门槛或增加
在线精确调用，只按之前冻结的嵌套顺序把表示扩到 32 模态：

```text
z  = loaded-q8 detector CR4(y)
h  = A_g^T z
x0 = h + U_g,32 a
x1 = one unchanged exact CGLS step
```

正式结果先给出 75/75 待验证信号。第一次独立 validator 在任何结果输出前停住：
一个起点缩进系数球后，因为 float64 舍入仍高出半径一个 ULP。没有重跑正式搜索，
也没有改半径或容差；只让独立验证器把缩放因子逐 ULP 往球内移动，最多 8 步。
实际只有 1 次投影需要 2 步，修复后的起点与封存正式起点最大差只有
`3.66e-12`，原门仍是 `1e-9`。

两轮独立红队先后发现并关闭了 formal evidence anchor 与 payload substitution 路径。
最终审计为 `P0=0 / P1=0`，唯一 repaired validator 才被启动。它重新构造三档几何、
32 个模态、75 个 observation、K2/K4 controls，并为每个单元重跑全部 21 个起点：

```text
pass / negative / inconclusive          75 / 0 / 0
F12+ / F15+ / F30+                      25 / 25 / 25
maximum metric recomputation difference 1.3323e-14
maximum gate recomputation difference   5.3722e-11
maximum mode-column difference          1.5451e-13
maximum projector distance              1.3019e-13
```

相对同调用 Zero-K2，四项最坏误差比全部不超过 1：

```text
field / full-gradient / interior-gradient / observation
0.975803 / 0.999036 / 0.999036 / 0.760340
```

所以科学判断发生了真实变化：

```text
GSLB32 representation headroom proven on opened Case3 cells
separate observation-only predictor protocol may be frozen
neural training not yet authorized
resource stage not authorized
algorithm_breakthrough=false
paper_success=false
```

这里最容易被误写的是“调用减半”。在线确实从 Zero-K4 的 `4A+4A^T` 降到
`2A+2A^T`，但 32 模态冷构造每套几何还要 `160A+32A^T`。当前每档几何只有
25 帧：

```text
GSLB32 including setup = 192 + 25*4 = 292
Zero-K4               =       25*8 = 200
```

满接受率也至少要 48 帧才打平，所以 v78 不是总加速结果。后验模型尺寸诊断还显示，
第 25-32 模态承载约 59% 系数能量，不能把物理表示又截回 8/16；但 75 个物理修正场
的 90%/95% 统计变化约落在 10/12 个主方向。这个低秩只能作为下一轮候选，PCA、decoder、
scaler 和 rank 必须在 frame-grouped outer fold 内重新拟合，不能把全 75 cells 的后验
统计冒充泛化。

下一步先让最简单的方法挑战学习模型：analytic U32 observation-ridge、
`A^T A U32` normal-image control、full-32 ridge 与 grouped-fold reduced-rank linear head。
同一 truth 帧的三套几何必须同折，禁止随机 cell split 和时间索引。若 full-32
observation-only control 都无法预测逐单元八门见证，就停止放大网络，转向增加独立
fit 数据或改变 observation-adaptive 表示。

完整证据、脱敏摘要和图表：

- `docs/nine_view_geometry_spline_capacity_v78_result_2026-07-31.md`
- `docs/nine_view_geometry_spline_capacity_v78_public_summary.json`
- `assets/nine_view_geometry_spline_capacity_v78.png`

## 2026-07-31：v79 说明“观测拟合得好”仍可能把三维梯度做错

v78 的 75/75 很容易让人误以为网络训练已经万事俱备。其实 v78 的系数是看着真值
选出来的，它只证明 32 模态空间里存在好答案，没有证明部署时能从 observation 找到它。

所以 v79 先让两个最便宜的解析方法挑战学习模型：

```text
方法一：用 r_h = y - A h 直接解 U32 系数
方法二：用 n_h = A^T(y - A h) 直接解 U32 系数
两者：x0 = h + U32 a，再跑一轮不变的 exact CGLS
在线账：2A + 2A^T
```

结果非常清楚：

```text
                                  observation residual   normal residual
完整八门通过                           9 / 75              0 / 75
field / K4, K2                       73 / 75, 75 / 75    49 / 75, 74 / 75
full-gradient / K4, K2               62 / 75, 42 / 75    43 / 75, 16 / 75
interior-gradient / K4, K2           55 / 75, 12 / 75    51 / 75,  2 / 75
observation / K4, K2                 75 / 75, 75 / 75    75 / 75, 75 / 75
```

也就是说，两条方法都能把测量空间的答案做得很好，但这并不保证三维内部的梯度正确。
对 BOST 而言，这不是一个可以忽略的指标问题：背景位移正是由折射率或密度梯度驱动，
只报 observation loss 会把一个物理上错误的场包装成成功。

v79.2 独立 validator 没有导入正式的 metric/pass helper，重新写了 gradient、误差和八门
逻辑，并复算 1,200 条 arm rows 与 300 条 outer rows。三轮同一红队最终为
`P0=0 / P1=0`，协调器的标准库 seal 复核也通过。主要 scaled 差不超过 `1.07e-11`。

启动时还出现过一次纯工程错误：默认 Python 没有 SciPy，程序在 import 阶段就退出，
没有读取 formal/raw 数据，也没有写结果。随后使用已绑定依赖的项目环境继续同一冻结验证。
这个错误保留在日志中，但不算一次科学运行。

现在应当准确写成：

```text
frozen observation-residual control closed
frozen normal-residual control closed
all observation-only predictors closed = false
neural training authorized = false
algorithm_breakthrough = false
paper_success = false
```

这不是“又失败了一次”这么简单。它把后续模型真正需要解决的问题定位得更窄：模型不能
只追求 coefficient MSE 或 observation residual，而必须从部署可见输入中恢复能守住局部
三维梯度的系数。下一份最小 predictor 协议只有在结果前冻结输入、同帧三几何同折、
fold-local preprocessing、逐单元八门和 fail-closed 动作后，才可能获得训练授权。

完整证据、脱敏摘要和图表：

- `docs/nine_view_gslb32_analytic_controls_v79_result_2026-07-31.md`
- `docs/nine_view_gslb32_analytic_controls_v79_public_summary.json`
- `assets/nine_view_gslb32_analytic_controls_v79.png`

## 2026-07-31：v80 说明“能学到一部分”还不等于可以部署

v79 只测试了两个解析 residual 映射，所以不能据此说 observation-only 学习没有希望。
v80 把这个问题变成了一个更严格但仍很小的确定性学习实验：同一 25 帧 Case 3、三档几何，
每一帧的三套几何保持在同一个连续时间折里，并留一帧 embargo；所有 normalization、模型与
超参数选择都只使用当前 fit 帧。

部署输入不看真值，也不看第一次 exact forward 后才能得到的 residual。它只有：

```text
U^T h                        32 维
(AU)^T y                     32 维
(AU)^T r_q8                  32 维
四个可见量的 log1p norm       4 维
几何专属模型                  100 维
共享模型再加 geometry one-hot  103 维
```

我没有直接上神经网络，而是按难度依次跑 mean、linear ridge、RBF KRR。结果是：

```text
                         mean    linear    RBF
shared + geometry ID     50/75   51/75    58/75
geometry-specific        50/75   49/75    58/75
```

RBF 确实比 v79 的 9/75 有明显进步，说明这些部署可见特征里不是完全没有信息。但合同要求
75/75，因为任一工况帧伤害三维场或观测都不能被平均分掩盖。共享 RBF 的 17 个失败里，9 个
只失败在 observation 相对 Zero-K4 不伤害，4 个只失败在同调用 interior-gradient；其余 4 个
含 full-gradient 或多门联合失败。最难的 F30+ 仍只有 16/25。

独立程序重新做了嵌套选择、预测、投影、物理指标和八门判断。450 份 raw prediction、投影后
prediction 和 metrics 的最大差都是 0，gate 最大差是 `5.55e-16`。把 held-out 系数标签改掉
以后，冻结 predictor API 输出变化也是 0。这里证明的是 API 级标签不干扰，不冒充整个进程从未
读取相应文件。

现在的准确结论是：

```text
constant / linear ridge / RBF KRR under the frozen v80 contract = closed
all observation-only predictors closed = false
larger neural model on the same target authorized = false
online exact calls = 2A + 2A^T
algorithm_breakthrough = false
paper_success = false
```

这一步最重要的不是 58 这个数字，而是策略发生了变化。v78 的 oracle witness 可能不是唯一、
连续或最容易从观测识别的一组系数。下一步先检查“相近观测特征是否对应跳变系数”，再尝试定义
更规范的可行监督目标，或让表示本身随 observation 调整。只有一个小型确定性 sentinel 能先
稳定全覆盖，才值得训练神经模型；否则堆容量只会掩盖目标歧义。

完整证据、脱敏摘要和图表：

- `docs/nine_view_gslb32_strict_observation_krr_v80_result_2026-07-31.md`
- `docs/nine_view_gslb32_strict_observation_krr_v80_public_summary.json`
- `assets/nine_view_gslb32_strict_observation_krr_v80.png`

## 2026-08-01：v81 排除了一个很像主因、其实量级不够的解释

v80 的 RBF 只有 58/75。最先该怀疑的不是“网络还不够大”，而是它学的 32 维标签
本身会不会乱跳：v78 每个单元都从 21 个不同起点做优化，如果这些起点落到差异很大的
合格解，同一个观测就可能没有唯一、平滑的监督答案。

我没有重新训练，也没有再跑一轮优化，而是把 v78 已封存的全部端点逐单元比较：

```text
cells                              75
starts per cell                    21
passing endpoints                 1,575 / 1,575
endpoint spread / v80 miss p50    0.00662%
endpoint spread / v80 miss p90    0.01197%
endpoint spread / v80 miss worst  0.01898%
retrospective limit               0.1%
```

最坏单元的起点漂移也只占 v80 预测误差的约五千分之一。物理修正场的相对漂移最坏
是 `6.36e-5`，21 个端点最大门值的范围最坏是 `8.42e-11`。所以至少在 v78 原来的
minimax 目标和既定 21 个起点里，优化器不是把标签弄乱的那个人。

独立程序没有导入正式 runner，而是用 Gram 矩阵公式重新算两两距离。两套公式的单元与
汇总指标最大差都是 `5.55e-11`，科学判决相同。这个差来自对极小距离的浮点消减，远小于
`1e-3` 科学判据，不会改变结论。

这次没有算法突破，但它真正节省了后续研究时间：不再优先为同一个 v78 目标做多起点
canonicalization，也不靠更大 RBF 或神经网络挽救。下一步应改变表示本身，让修正方向随
部署时可见的 observation 和 geometry 自适应；仍保持同一八门、时间分组和在线调用账。

边界必须讲清：v81 没测试另一种 truth-aware 目标是否有概念多解，也没有外部工况、资源、
曲线光线或真实 BOST 结果。当前仍是：

```text
numerical start instability under the frozen v78 objective = closed
alternative-objective nonuniqueness = untested
next route = observation-adaptive representation diagnostic
neural training authorized = false
algorithm_breakthrough = false
paper_success = false
```

完整证据、脱敏摘要和图表：

- `docs/nine_view_gslb32_target_stability_v81_result_2026-08-01.md`
- `docs/nine_view_gslb32_target_stability_v81_public_summary.json`
- `assets/nine_view_gslb32_target_stability_v81.png`

## 2026-08-01：v82 用四个空间调制方向补齐了 75/75 表示容量

v81 排除了“随机优化起点把标签搞乱”这个主因后，我没有继续扩大
RBF，也没有直接上 FNO 或 U-Net。先问一个更便宜、能直接改变科学判断的问题：
**固定 GSLB32 空间会不会少了随 observation 位置改变的局部形变自由度？**

我把它做成了两个匹配对照。两者都只有四个参数，都用同一个
`[-1,1]^4` 参数盒、同一物理预算、同一 13 起点搜索、同一八门，都放进
`2A+2A^T` 的 strict CGLS K1 壳：

```text
coefficient-band4   在固定 U32 内分四段修正系数
spatial-mask4       用 z / y / x / 径向掩膜调制 observation-conditioned 基础修正
```

结果是：

```text
                                     总通过    救回 v80 失败    F12 / F15 / F30
coefficient-band4                     73/75          15/17            25 / 25 / 23
spatial-mask4                         75/75          17/17            25 / 25 / 25

coefficient-band4 worst maximum gate   +0.00821
spatial-mask4 worst maximum gate       -0.04192
```

maximum gate 必须小于等于 0 才表示八门同时通过。固定空间内的对照剩下两个
F30 失败；空间调制不仅把它们补上，而且全局最坏单元仍有明确负裕量。
它也没有偷偷多调用物理算子：空间调制的额外 exact 账是 `0A+0A^T`。

独立验证中有一个值得保留的失败记录。原 validator 把 32 维 base coefficient 送进了
只接受 4 维 beta 的检查路径，在写出任何验证结果前 fail-closed 退出。我没有重跑或
改动正式结果，修复只删除这个无效调用；每个 arm 的 beta=0 基线、exact endpoint replay、
八门、原始优化约束与调用账都保留。

修复后的独立程序自己重建方向、重跑搜索和八门：

```text
formal optimizer terminals complete       1950 / 1950
independent terminals complete            1950 / 1950
maximum gate difference                   4.44e-16
maximum metric difference                 0
exact best-field replay difference        4.51e-17
direction change after truth mutation     0
```

这一轮是实质性的机制正结果：**空间自适应表示现在有了值得学的 75/75 容量**。
但四个系数仍由真值可见的 oracle 选择，所以它不是部署算法，也没有证明外部泛化、
wall/RSS 优势、curved ray 或真实 BOST。准确状态仍是：

```text
spatial-mask4 family-specific capacity = pass on opened Case 3
observation-only coefficient predictor = untested
external generalization = false
algorithm_breakthrough = false
paper_success = false
```

下一步只做一件事：冻结 spatial-mask4，用同一五个连续时间外折和一帧 embargo，
训练最小的 observation-only `features -> 4 coefficients` predictor。若不能 75/75，就关闭该表示，
不用更大网络挽救；只有它过门后，才值得打开一个之前未见的公开反应流工况。

完整证据、脱敏摘要和图表：

- `docs/nine_view_observation_adaptive_mask_capacity_v82_result_2026-08-01.md`
- `docs/nine_view_observation_adaptive_mask_capacity_v82_public_summary.json`
- `assets/nine_view_observation_adaptive_mask_capacity_v82.png`

## 2026-08-01：v83.1 证明“表示里有解”还没有变成“观测能找到解”

v82 的 spatial-mask4 在 75 个已开封单元里都能找到八门通过的四维 beta。今天没有把它直接包装成
算法成功，而是把最容易漏掉的那层数据隔离补齐：每个外折都从本折训练数据重新生成上游基础系数、
beta 目标和超参数，不能复用全局 OOF 结果。

正式结果与独立复算一致：

```text
fit targets with a direct passing witness      276 / 276
held-out representation capacity                75 / 75
beta-zero                                        58 / 75
constant mean                                    66 / 75
compact52 linear / RBF                           66 / 75, 65 / 75
enriched148 linear / RBF                         63 / 75, 68 / 75
formal vs independent maximum differences        0
```

所以失败不是“这一折根本没有可行 beta”，而是只看部署可见 observation 的冻结模型没能稳定预测它。
最佳 enriched RBF 仍差 7 个单元。更重要的是，这七个失败不是杂乱地坏在不同物理量上：field、全梯度、
内部梯度和同调用 Zero-K2 对照全部通过，只有最终 observation 相对 Zero-K4 的 1% no-harm 门越线。

这给出了比“换更大网络”更具体的下一步。measurement residual 在部署时能直接算，所以先研究一个
可观测的安全回退或追加一轮 CGLS：大多数安全单元停在 K1，危险单元再花一对 A/A^T。只有它能逐单元
守住八门且平均调用仍少于 Zero-K4，才值得继续。不能用同一批已看过的失败帧反复调阈值后冒充外部验证。

这是一条可信负结果，也是一条有用的路线收缩：

```text
spatial-mask4 representation capacity = 75 / 75
best strict observation-only point predictor = 68 / 75
direct point-regression roster = closed
larger neural model authorized = false
next mechanism = observable safety fallback / one-extra-iteration refinement
algorithm_breakthrough = false
paper_success = false
```

完整证据、脱敏摘要和图表：

- `docs/nine_view_spatial_mask_predictor_v83_result_2026-08-01.md`
- `docs/nine_view_spatial_mask_predictor_v83_public_summary.json`
- `assets/nine_view_spatial_mask_predictor_v83.png`

## 2026-08-01：v84.2 第一次把“学到的起点”与“可观测安全回退”闭成 75/75

v83.1 留下的七个失败有一个很重要的共同点：三维场、全梯度、内部梯度和同调用 Zero-K2 都没坏，
只有最终 measurement residual 相对 Zero-K4 略差。这个量部署时可以直接算，所以今天没有继续堆模型，
而是问一个更实在的问题：**能不能让安全单元早点停，危险单元只多做一步？**

做法很小：先运行 enriched148-RBF warm initializer 到 K1，再计算相对 residual。阈值只从每个外折
fit 部分的 OOF residual 得到，而且同一帧三档几何先取最坏值；held-out 单元的 truth、八门和参照指标
不参与分支。低于阈值就停在 K1，高于阈值就沿原来的 CGLS recurrence 继续到 K2。

正式结果和不导入正式 runner 的独立复算完全一致：

```text
always learned K1                         67 / 75, 2.0000 A + 2.0000 A^T
residual-gated K1/K2 continuation         75 / 75, 2.3467 A + 2.3467 A^T
accepted at K1 / continued to K2          49 / 26
F12 / F15 / F30 joint pass                25 / 25 / 25
exact-call reduction versus Zero-K4       41.3333%
formal versus independent declared diffs  all 0
```

这次可以叫“开发集机制正结果”，因为它同时守住了逐单元八门和调用预算，不再只是容量 oracle 或平均改善。
但不能叫突破：Case 3 已经被反复用于机制开发；Case 4/6 仍封存；`41.3%` 只是 exact 调用账，不是 wall
或内存实测；同一进程从模拟真值生成 observation，因此 process-level never-read 也没有证明。

还修正了一个容易误解的参照问题。Zero-K4 只必然在相对自己的四个 no-harm 比值上等于 1，另外四门
是在和不同的 Zero-K2 比；半收敛时它完全可能不通过。Zero-K2 同理。正式与独立程序分别验证两个参照
的本征比值最大偏差都为 0，没有降低候选的八门标准。

下一步不是在 Case 3 上继续调阈值。先把全量 Case 3 的 predictor、预处理、阈值、调用账和报告模板
冻结成单一 deployment artifact，再按早已预注册的 Case 4、Case 6 顺序做外部门。外部 matched-accuracy
通过后，才值得烧 fresh-process wall 与 whole-pipeline RSS。

```text
Case3 development mechanism headroom = true
external generalization = false
resource advantage = false
algorithm_breakthrough = false
paper_success = false
real BOST = false
```

完整证据、脱敏摘要和图表：

- `docs/nine_view_observable_safety_gate_v84_result_2026-08-01.md`
- `docs/nine_view_observable_safety_gate_v84_public_summary.json`
- `assets/nine_view_observable_safety_gate_v84.png`

## 2026-08-01：v85 第一次完成两个零适配外部门，Case 4 全过，Case 6 卡在内部梯度

今天不再继续在 Case 3 上调模型，而是把已经冻结的 predictor、归一化、九视角几何、residual 阈值和
CGLS 回退原封不动地放到两个外部反应流工况。顺序也提前固定为 Case 4 再 Case 6，前一个结果不能用来
更新后一个。这是当前路线第一次真正回答“换一个反应流以后还能不能守住严格八门”。

结果不是单一的好消息，也不是模糊的失败：

```text
Case 4                  84 / 84 PASS
Case 4 K1 / K2          41 / 43
Case 4 mean exact calls 2.5119 A + 2.5119 A^T  (-37.2% vs Zero-K4)

Case 6                  78 / 90 FAIL
Case 6 K1 / K2          90 / 0
Case 6 failures         12, all interior-gradient only
joint external gate     FAIL (required 174 / 174)
```

Case 4 是可信的零适配外部正结果。Case 6 则说明当前 scalar residual gate 还不够：它把 90 个单元全部判断
为“可以在 K1 停止”，但其中 12 个单元的内部局部梯度不安全。field、整体 gradient 和 observation 对照门
都通过，因此不是整体场完全失效。12 个失败中，4 个违反相对 Zero-K4 的 no-harm 门，8 个违反相对同调用
Zero-K2 的优势门。

失败单元的 residual score 为 `0.24797-0.35614`，通过单元为 `0.21709-0.36313`，两类高度重叠。事后若仍
坚持单标量阈值并要求零漏判，最多只能安全接受 `11/90`。这说明仅靠“全局残差有多小”看不见“局部内部
梯度是否危险”。这个 11/90 只用于解释失效，不能拿来更新阈值后补考 Case 6。

两个外部门都由独立程序重新生成预测、残差、八门和实际调用回执；声明的 prediction field、residual、score
和调用账最大差都为 0。Case 6 observation 的独立物理重算也逐数组相等。证据可信，但联合科学结论仍然是
FAIL，所以没有启动 wall/RSS 资源测试。

下一项直接机理诊断是在已经开封的 Case 6 上，把同一个 warm K1 强制继续一步真实 CGLS K2：如果 12 个
失败全部被救回，主因是门控器识别不了局部风险；如果 K2 仍失败，说明 warm 表示本身也缺少内部梯度容量。
它只能改变机制判断，不能重新算一次外部通过。任何新选择器都必须留到另一个真正未打开的工况一次性验证。

```text
Case 4 zero-adaptation external pass = true
Case 4 + Case 6 joint external pass = false
resource gate authorized = false
external generalization = false
algorithm_breakthrough = false
paper_success = false
real BOST = false
```

完整证据、脱敏摘要和图表：

- `docs/nine_view_v84_external_cases_v85_result_2026-08-01.md`
- `docs/nine_view_v84_external_cases_v85_public_summary.json`
- `assets/nine_view_v84_external_cases_v85.png`

## 2026-08-02：v86 证明“统一多走一步”也不是 Case 6 的答案

今天没有换网络，也没有给 Case 6 重新调阈值。我把 v85 留下的问题拆成最小实验：同一个 observation-only
warm K1，所有 90 个单元都强制沿未修改 CGLS recurrence 再走一步。模型、归一化、几何、候选场和两组
对照都不变；唯一新增成本是每单元 `1A+1A^T`，总账从 `2A+2A^T` 变成 `3A+3A^T`。

结果不是“12 个失败全部救回”，而是总体更差：

```text
Warm K1 pass             78 / 90
Forced K2 pass           71 / 90
pass -> pass             66
fail -> pass              5
pass -> fail             12
fail -> fail              7
```

最值得记住的不是 71 这个数字，而是它为什么发生。forced K2 在 90 个单元上都让 field、整体 gradient 和
observation relative-L2 下降，说明全局拟合确实继续变好；但 interior-gradient 不是单调下降。K1 时只有
4 个相对 Zero-K4 的内部梯度 no-harm 失败，K2 后增到 16 个。新增失败集中在早期瞬态：frame 0-4 的
15 个 geometry-cell 中，11 个由通过变失败，剩下 4 个继续失败；frame 15-29 的 45 个单元则全部通过。

这给了一个非常具体的物理/算法判断：全局观测残差看不到局部火焰梯度风险，而标准 CGLS 的下一步也只保证
它自己的全局最小二乘目标继续下降，不保证局部 interior-gradient 不受伤。所以旧 scalar gate 要关闭，
“所有单元统一停 K1”和“所有单元统一走 K2”也都要关闭。

独立程序没有导入正式 v86 runner。它重新建三档九视角几何、重新跑模型、独立实现 K1/K2 recurrence、
重新预处理真值并计算全部八门；fields、residuals、90 行结果和摘要最大差都是 0。正式科学判决是：

`FAIL_FORCED_K2_DOES_NOT_RESCUE_ALL_CASE6_CELLS_V86`

下一步不是扩大 FNO/UNO/U-Net，而是先审计 K1 与 K2 之间的连续线段：

`x(t)=x_K1+t(x_K2-x_K1),  t in [0,1]`

如果每个单元都有非空的八门可行区间，才说明“阻尼/深度选择”有表示容量，之后再研究部署可见的选择器；
如果有单元整段都无解，就关闭这条方向，重新设计 warm correction 表示。因为 Case 6 已经开封，这仍只能是
机理诊断，任何新策略都必须在另一个未打开工况重做一次性外部门。

```text
post-open mechanism diagnostic = true
external validation = false
resource advantage = false
algorithm_breakthrough = false
paper_success = false
real BOST = false
```

完整证据、脱敏摘要和图表：

- `docs/nine_view_v85_case6_forced_k2_v86_result_2026-08-02.md`
- `docs/nine_view_v85_case6_forced_k2_v86_public_summary.json`
- `assets/nine_view_v85_case6_forced_k2_v86.png`
