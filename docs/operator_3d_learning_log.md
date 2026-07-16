# 3D 逆问题学习持续日志

日期：2026-07-16

这份日志只记录我在读懂和复核这条实验线时真正学到的东西。重点不是把结果写成“模型越来越强”，而是把每次尝试的前提、数字、失败原因和下一步验证条件留下来。

## 先把证据等级说清楚

- **L0：真实实验/论文证据。** 目前没有。这里没有 OpenBOS/OERF 真实测量，也没有论文级 superiority 结果。
- **L1：预注册的新鲜合成开发证据。** 有固定配置、固定 checkpoint 和首开前冻结的门禁，但数据仍是 synthetic proxy；可以否定或支持一个开发假设，不能直接证明真实装置有效。
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

**证据等级。** **L1 + L3**。这是 `preregistered_fresh_synthetic_development`，报告明确写着 `FRESH_DEVELOPMENT_NO_GO`，但仍然只是合成弱偏折 proxy，不是论文成果或 OERF 结果。

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

## 28. 风险门第一次通过 fresh，但仍抓漏了两个危险病例

这一轮没有继续扩大谱网络。我们冻结了一个更小的问题：只看部署时能拿到的
residual、精确伴随梯度、视角 mask 和候选方向，能不能判断“这次该不该让
学习器接管”。

方法暂称 OCRRG。它用 16 个无真值特征预测 learned preconditioner 相对
inverse-Sobolev 的 field-gain，再减去 split-conformal 的保守误差分位数。
只有预测下界、特征距离和 6 至 9 视角硬支持同时通过，才运行 learned
四步求解；否则整条路径精确回到 Sobolev。判断本身不需要把两种重建都跑完，
所以仍是相同的 `4F+4Aᵀ`。

fresh 协议先在提交 `cd5d4a0` 中冻结，再打开七组各 24 个新场。三种模型
种子都通过了预注册候选门：

- support IID：coverage 36.1%，平均 gain +1.38%，harm 2.78%；
- 未见形态：coverage 26.4%，平均 gain +1.04%，harm 0；
- 强相关噪声：coverage 43.1%，平均 gain +1.31%，harm 2.78%；
- 未见形态 + 强噪声：coverage 27.8%，平均 gain +1.41%，harm 0；
- 3 至 5 视角两组：coverage 0，逐值回退 Sobolev。

**用人话说：**风险门把原学习器“见什么都出手”改成了“有把握才出手”。
它确实大幅压低了坏尾部，同时没有在所有支持域样本上装死。这是一个真实
进步，但还不是成功算法。

独立 validator 找到 4 条被接受后仍恶化超过 1% 的记录，只来自两个源样本：
一个 6-view plume 在两个种子上退化约 2.6%，一个强相关噪声的 6-view
oblique shock 在两个种子上退化 4.5% 至 5.7%。这说明 pooled risk model
对最低支持视角数和特定物理形态仍不够保守。

下一步不是在当前 fresh 上扫阈值，而是换全新 seeds 做独立重复，并把风险
校准改成按 view count、形态族和噪声强度分组。真实迁移前还必须用师兄提供
的 flow-off repeats 替换合成 covariance。没有这一步，不能宣称逐样本安全、
任意 OOD conformal 保证或优于 FNO/DeepONet。

完整入口：[fresh 判决](psu_b0_residual_risk_fresh_result_2026-07-16.md) · [公开 JSON](psu_b0_residual_risk_fresh_public_summary.json) · [论文图](../demo_t16_operator/results/psu_b0_residual_risk_fresh/psu_b0_residual_risk_fresh_figure.png)

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
