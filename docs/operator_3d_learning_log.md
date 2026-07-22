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
canonical config SHA256 是 `7de695ebc419ad2e7a7edba0c10d45b9ff31027a0252c41c2481faf56f4eb085`，
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
`7de695ebc419ad2e7a7edba0c10d45b9ff31027a0252c41c2481faf56f4eb085`；协议仍为 180 项断言，完整聚焦回归为
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
`7de695ebc419ad2e7a7edba0c10d45b9ff31027a0252c41c2481faf56f4eb085`。

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
`7de695ebc419ad2e7a7edba0c10d45b9ff31027a0252c41c2481faf56f4eb085`。这仍只是当前工作树回归，正式
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
