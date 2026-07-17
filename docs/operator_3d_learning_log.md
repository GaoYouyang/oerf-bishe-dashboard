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
