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
`8d2bba156028e4b14385f5a563d4d7c18817bb17a70dc0856bfeb240e8e765ed`，独立 validator
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
