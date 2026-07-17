# Covariance-weighted 3D PDHG/TV-Huber 红队清单

日期：2026-07-17
状态：`PREREGISTRATION_CHECKLIST_ONLY`
适用对象：PSU B0 固定域上的 covariance-weighted 3D PDHG/TV-Huber
确定性基线

> 这不是结果报告，也不预告 PDHG 会赢。它是一份开跑前合同：先把数学问题、
> 数值实现、计算账本和开封顺序钉死，再允许看重建误差。任何一项没有留下可
> 复核证据，都按“未通过”处理，不能用平均指标好看来补票。

## 0. 先冻结我们究竟在解什么

建议把基线写成一个没有歧义的鞍点问题。令

\[
B=s_dWA,\qquad y=s_dWb,
\]

其中 `A` 是固定三维场到 detector displacement 的线性算子，`W` 满足
`W^T W = C^{-1}`，`C` 是测量噪声协方差；`s_d` 是预先冻结的数据项尺度。
目标函数为

\[
\min_x\;
\frac12\lVert Bx-y\rVert_2^2
+\lambda\sum_v\phi_\delta\!\left(\lVert(Dx)_v\rVert_2\right)
+I_{\mathcal X}(x).
\]

`D` 是带物理网格间距的三维梯度，`I_X` 表示固定 support 以及明确声明的物理
约束。TV 使用 `phi_0(t)=t`；Huber 使用

\[
\phi_\delta(t)=
\begin{cases}
t^2/(2\delta),&t\le\delta,\\
t-\delta/2,&t>\delta.
\end{cases}
\]

最直接的 one-pair PDHG 拆分是 `Kx=(Bx,Dx)`：

\[
\begin{aligned}
p^{k+1}&=\frac{p^k+\sigma_B(B\bar x^k-y)}{1+\sigma_B},\\
q^{k+1}&=\operatorname{prox}_{\sigma_D(\lambda R)^*}
             (q^k+\sigma_DD\bar x^k),\\
x^{k+1}&=\Pi_{\mathcal X}\!\left[x^k-\tau(B^Tp^{k+1}+D^Tq^{k+1})\right],\\
\bar x^{k+1}&=x^{k+1}+\theta(x^{k+1}-x^k).
\end{aligned}
\]

这样每轮只需一次 `B` 和一次 `B^T`；`D/D^T` 仍要计时，但不能冒充昂贵的
投影算子调用。

## 1. 目标函数与尺度门禁

- [ ] **OBJ-1 写死数据项归一化。** 明确 `s_d` 是 `1`、`1/sqrt(m)`，还是由
  物理单位得到的固定尺度。若从 `||WAx-Wb||^2/2` 改为除以测量数 `m`，必须
  同步换算 `lambda`；不能把同一个数值 `lambda` 跨两种目标函数直接比较。
- [ ] **OBJ-2 写死场的单位。** 说明 `x` 是折射率、折射率扰动、密度，还是
  无量纲代理场；记录坐标单位、voxel spacing 和 detector displacement 单位。
  改单位后，`lambda` 与 Huber `delta` 都必须按定义重标度。
- [ ] **OBJ-3 协方差只做一次定义。** 记录 `C` 的来源、均值去除、shrinkage、
  特征值截断和分块方式，并验证 `B^T=A^TW^Ts_d`。不能一边把残差白化，
  一边漏掉伴随端的 `W^T`。
- [ ] **OBJ-4 区分 oracle 与可部署协方差。** 由 synthetic truth/noise generator
  得到的 `C` 只能叫 oracle mechanism audit；只有独立 flow-off repeats 估计的
  `C` 才能进入真实迁移结论。
- [ ] **OBJ-5 固定 TV/Huber 定义。** 冻结 isotropic/anisotropic、Huber 公式、
  `delta`、是否乘 voxel volume、是否按 active voxels 归一化。论文、代码和配置
  必须使用同一公式。
- [ ] **OBJ-6 不偷换变量。** 若引入 Sobolev 或其他变换 `x=Su`，目标必须写成
  `R(Su)` 并最终评估物理场 `x`。在 `u` 上做 TV 后称为“物理场 TV”不成立。
- [ ] **OBJ-7 物理约束有来源。** support mask 可以继承冻结 B0 域；非负约束
  只有在目标物理量必为非负时才允许。若重建的是可正可负的扰动场，默认不得
  用非负投影获取无说明的收益。
- [ ] **OBJ-8 配置可复述目标。** 单个机器可读配置应足以重建 `A/W/D/X`、
  `lambda/delta`、步长、初始化、精度与停止规则，并保存配置哈希。

**讲人话：**先规定“尺子”和“单位”。否则调 `lambda` 得到的提升，很可能只
是把数据项悄悄缩小了，而不是算法真的更会重建。

## 2. Operator norm 与步长门禁

- [ ] **NORM-1 对完整 `K` 定标。** 标量步长必须满足
  `tau * sigma * ||K||_2^2 < 1`，这里 `K=[B;D]`，不是只估 `A`。block step
  则验证 `||Sigma^(1/2) K T^(1/2)||_2^2 < 1`。
- [ ] **NORM-2 白化后重估。** `W`、数据尺度、mask、voxel spacing 或 resolution
  任一变化都可能改变范数；不得沿用旧 `A` 的步长。
- [ ] **NORM-3 tiny dense oracle。** 在可显式组装的小问题上用 SVD 得到真范数，
  对照 matrix-free power/Lanczos 实现及其伴随调用。
- [ ] **NORM-4 不把 power estimate 叫上界。** 普通 power iteration 通常给出
  近似值而非严格上界。若没有可证明界，就冻结安全裕量并使用有理论依据的
  backtracking/residual 检查；不能仅凭“训练没炸”声称满足收敛条件。
- [ ] **NORM-5 解析检查 `D`。** 对所选边界条件，核对三维 forward-difference
  梯度的解析界；各向异性网格不能把 `hx=hy=hz=1` 的界直接搬来。
- [ ] **NORM-6 做步长压力测试。** 在不看 truth 排名的数值测试中检查冻结步长、
  更保守步长和故意越界步长；记录 fixed-point residual、有限值与停止原因。
- [ ] **NORM-7 范数估计也记账。** 单次装置校准成本和每次重建成本分别报告；
  若只给 PDHG 预计算范数，经典基线也应获得等价的预计算机会。

**红线：**出现 NaN、对偶量持续发散，或只能靠删掉失败 case 保持稳定时，先
判实现/步长失败，不进入算法准确率排名。

## 3. 边界条件与离散伴随门禁

- [ ] **BC-1 冻结外边界。** 明确 `D` 在体盒边界采用 Neumann/no-flux、周期，
  还是向盒外零值的 Dirichlet；图示一维端点公式，避免不同人实现不同问题。
- [ ] **BC-2 冻结 mask 内边界。** “mask 外 `x=0`”与“不跨 mask 边缘计算差分”
  分别相当于边界拉向零和 no-flux，物理含义不同，必须二选一并做消融。
- [ ] **BC-3 使用真实 spacing。** `Dx,Dy,Dz` 分别除以 `hx,hy,hz`；若内积含
  voxel volume，`D^T` 也必须在同一离散内积下构造。
- [ ] **BC-4 做逐维手算。** 用常数场、单 voxel impulse、线性 ramp 和触边
  thin-front 检查方向、端点与 mask 行为。
- [ ] **BC-5 通过 dot-product test。** 对多组固定随机向量验证
  `<Dx,q>=<x,D^Tq>`；对 `B/B^T` 另做相同测试。容差应在运行前按 dtype 与
  算子尺度规定。
- [ ] **BC-6 不误解符号。** 若代码把 `D^T` 命名为 divergence，要明确它是
  梯度的伴随还是负伴随，PDHG 更新式只能使用与测试一致的符号。

## 4. Prox 正确性门禁

- [ ] **PROX-1 数据对偶 prox 有解析对照。** 对
  `F(z)=||z-y||^2/2`，检查
  `prox_(sigma F*)(p)=(p-sigma*y)/(1+sigma)`，包括广播、dtype 和 batch 维。
- [ ] **PROX-2 TV 对偶 prox 是逐 voxel 的球投影。** isotropic TV 应检查
  `q <- q / max(1, ||q||_2/lambda)`；不得误写成逐分量 clip 后仍称 isotropic。
- [ ] **PROX-3 Huber 对偶 prox 与定义匹配。** 对本文 `phi_delta`，先做
  `q <- q/(1+sigma_D*delta/lambda)`，再投影到半径 `lambda` 的逐 voxel
  二范数球；显式处理 `lambda=0`，不得产生除零或残留平滑。
- [ ] **PROX-4 约束 prox 单独测试。** support、非负或 box projection 必须幂等，
  且不得改动合法点；多个约束的交投影若非一次 clip 可得，不能假装其组合仍
  是精确 prox。
- [ ] **PROX-5 做解析和数值双重 oracle。** 标量、三向量、`A=I` 与 tiny 3D
  case 使用手算/独立凸优化解对照；再用 Moreau identity 检查 primal/dual prox。
- [ ] **PROX-6 检查极限。** 覆盖 `lambda=0`、很大 `delta`、`delta` 趋近 TV、
  零梯度、极大梯度、空 support 与单 voxel support。
- [ ] **PROX-7 不要求目标逐步单调。** 标准 PDHG 的单步 primal objective 可以
  波动；正确门禁是 saddle/fixed-point residual、可计算的 primal-dual gap 或
  与独立最优解的一致性，不能因为某一步 objective 上升就擅自改算法。

## 5. 零正则 sanity gate

- [ ] **ZERO-1 `lambda=0` 真正关闭正则。** TV/Huber 对偶变量应被固定到零，
  `delta`、`D` 与上一次运行状态不得继续影响解。
- [ ] **ZERO-2 关闭额外约束后对齐最小二乘。** tiny full-rank case 与直接解
  对齐；欠定且从零初始化时，说明所期望的最小范数解及所用内积。
- [ ] **ZERO-3 约束不关闭时换正确 oracle。** support/nonnegative PDHG 应与
  同一个约束最小二乘问题比较，不能拿 unconstrained PCGLS 当真值。
- [ ] **ZERO-4 不强求逐迭代等价。** PDHG 与 PCGLS 轨迹本来不同；应比较同一
  目标的收敛解、残差和 KKT/fixed-point 条件，而不是要求第 `k` 步体场相等。
- [ ] **ZERO-5 做尺度与精确数据检查。** `A=I`、`b=Ax_true`、`b=0` 及同时缩放
  `A,b` 的 case 应符合推导；任何异常先归因实现，不得进入物理解释。
- [ ] **ZERO-6 warm start 清零。** 每个独立样本的 primal、dual、extrapolated
  state 都按配置初始化，防止上一个场泄漏到下一个场。

## 6. 调用预算与运行代价公平门禁

- [ ] **CALL-1 分开记录 `F` 与 `A^T/J^T`。** 一轮本基线应为一次 `B` 和一次
  `B^T`，同时记录 `W/W^T`、`D/D^T`、prox 和 projection 的 wall time。
- [ ] **CALL-2 初始化和输出都入账。** 初始 residual、额外计算 `Ax^k`、选
  checkpoint、算 observable stopping feature 所需的投影都不能藏在 runner 外。
- [ ] **CALL-3 norm/covariance 成本分层。** 报告 acquisition、covariance fit、
  whitening materialization、norm estimate、单次重建和多次摊销成本。
- [ ] **CALL-4 同时做三种比较。** 至少提供 same `F/F^T` calls、same wall time
  和 converged/固定 tolerance 三张表；任何一种赢都不能代替另外两种。
- [ ] **CALL-5 基线同权。** graph-PCGLS、component-PCGLS 及其他确定性邻近
  基线使用同一 `A/W`、support、precision、hardware、warm-up 与可用调参预算。
- [ ] **CALL-6 缓存必须对称。** 共享 geometry/covariance cache 可以复用；
  只为候选缓存投影而让基线重算，或反过来，都不公平。
- [ ] **CALL-7 truth 指标不算部署动作。** truth-L2/front 指标可离线评分，但
  不能进入停止、回退或模型选择；若算法运行时需要它，就已经不可部署。
- [ ] **CALL-8 记录失败运行。** OOM、NaN、超时和 early abort 都保留在总表，
  不得从分母删除后再报告成功率。

**讲人话：**PDHG 的卖点是“一轮只拍一次虚拟投影、反传一次”。如果为了挑
最好的一轮又偷偷多投影，或者不算范数估计成本，这个卖点就没有被证明。

## 7. 半收敛与停止规则门禁

- [ ] **STOP-1 保存完整轨迹。** 每个冻结 checkpoint 保存 whitened residual、
  data term、TV/Huber、primal/dual fixed-point residual、field 指标和形态指标。
- [ ] **STOP-2 区分优化收敛与重建最优。** 目标函数收敛只说明解对了所写问题；
  truth-L2 先好后坏说明目标/正则与物理指标不完全一致，不能靠继续迭代修辞化解。
- [ ] **STOP-3 oracle iteration 只作诊断上界。** 每个样本按 truth 选最佳轮次
  不可部署，只能标成 oracle envelope，不能作为方法主结果。
- [ ] **STOP-4 停止规则只用可观测量。** 固定迭代、预注册 residual/KKT 阈值
  或无需 truth 的规则才有资格部署；规则输入和阈值必须在测试数据开封前冻结。
- [ ] **STOP-5 与经典基线共同调停止深度。** 不能给 PDHG 搜完整轨迹，却只给
  PCGLS 一个历史 stage；每个方法得到等价的开发预算和独立冻结规则。
- [ ] **STOP-6 报告尾部随迭代的曲线。** mean 上升而 p10/worst 恶化就是明确
  的半收敛风险；不得只截取平均值峰顶。
- [ ] **STOP-7 检查 dual restart 与 extrapolation。** 若修改 `theta`、restart
  或 averaging，视为新候选并重新记账，不得在看过 truth 后悄悄打开。

## 8. 形态尾部与统计门禁

- [ ] **TAIL-1 逐场配对。** 候选与基线必须使用同一 geometry、场、噪声和
  replicate；先算逐场 gain，再汇总，不能比较两个不配对的均值。
- [ ] **TAIL-2 预先冻结形态组。** 至少保留 thin/double front、annular、
  oblique shock、multi-plume、vortex 等已有反应场族；看完结果后新增分组只能
  标为探索性。
- [ ] **TAIL-3 不让大体积平滑区淹没 front。** 同时报 field relative-L2、
  front F1/位置/宽度或已冻结的 edge 指标，以及逐相机 whitened residual。
- [ ] **TAIL-4 强制报告风险列。** mean/median 之外必须给 p10、超过预注册伤害
  阈值的比例、worst case 和每个 morphology 的方向一致性。
- [ ] **TAIL-5 最高独立单位是 replicate/session。** voxel、ray 和同一场的
  多个 checkpoint 不是独立样本；置信区间/重采样应在 replicate 或 session
  层聚类。
- [ ] **TAIL-6 小分母单列。** 相对增益在基线误差接近零时会爆炸；同时给绝对
  差、基线分母和预注册稳定化规则，不得临时删点。
- [ ] **TAIL-7 最坏 case 要可视化。** 固定色标展示 truth、baseline、candidate、
  error 和 front slice；不能只展示最好看的三维体渲染。
- [ ] **TAIL-8 mean 赢、尾部输不算通过。** 是否允许小幅最坏退化必须在运行前
  写入门禁；若未预注册，红队默认要求坏尾不能被平均收益掩盖。

## 9. Post-open 选择偏差门禁

- [ ] **OPEN-1 给每个数据单元贴状态。** 明确 `implementation-only`、
  `opened-development`、`opened-diagnostic`、`fresh-sealed` 和 `real-external`；
  已经看过 truth 的数据永远不能重新命名为 fresh validation。
- [ ] **OPEN-2 新候选自动继承 post-open 标签。** PDHG 是在已有 PCGLS/SupPCG
  结果之后提出的，因此在同一批已见场上的任何优势都只是开发证据。
- [ ] **OPEN-3 首次运行前冻结搜索空间。** 保存 `lambda/delta`、步长策略、
  iteration checkpoints、边界条件、评价指标、聚合方式和通过/停止条件的配置
  及哈希。
- [ ] **OPEN-4 所有候选进入 ledger。** 包括失败配置、临时扩展和人工查看次数；
  不得只保留最终赢家来缩小多重选择表面。
- [ ] **OPEN-5 一次开封只回答一个问题。** scale smoke 只能定数量级；若门禁
  未预先授权 tail extension，看到失败后无限扩网格就是追逐噪声。
- [ ] **OPEN-6 选择与诊断不循环使用。** diagnostic 一旦反馈到结构、损失或
  阈值修改，它随即成为 development，下一次确认必须换新的独立单位。
- [ ] **OPEN-7 morphology 标签不得成为隐藏 truth。** 除非部署时存在独立、
  预验证的形态识别器，否则不能按真实 morphology 为每个样本挑 `lambda`。
- [ ] **OPEN-8 fresh 只跑冻结 finalist。** fresh 开封前应锁定少量候选、单一
  主比较和判决代码；开封后不再修 bug。若发现实质 bug，作废该次确认并另取
  新鲜单位，不能修完重跑后仍称首次确认。
- [ ] **OPEN-9 论文措辞跟证据走。** opened synthetic 可以支持机制发现或
  NO-GO；没有 fresh/real external 就不能写“泛化”“实验有效”或“优于现有方法”。

## 10. 真实 OERF 迁移限制门禁

- [ ] **REAL-1 用独立 flow-off repeats 估计 `C`。** 处理相机固定图样、空间
  相关、跨相机相关和时间漂移；重复数是否足够由 effective rank、谱稳定性和
  held-out likelihood/whiteness 检查决定，不能沿用 synthetic oracle。
- [ ] **REAL-2 covariance fit 与测试 session 隔离。** shrinkage 和截断参数只
  在 calibration/development session 选择；报告跨日期、曝光和相机状态的漂移。
- [ ] **REAL-3 验证白化是否真的白。** 在未参与拟合的 flow-off 上检查零均值、
  单位尺度、空间/跨相机/时间剩余相关；“矩阵可逆”不等于噪声模型正确。
- [ ] **REAL-4 校准真实 forward/adjoint。** 相机 pose、有限孔径、ray support、
  BOS displacement extraction 与单位换算必须进入 `A`；若是 nonlinear bending，
  则明确切换为局部 `J/J^T`，线性 B0 结论不能直接外推。
- [ ] **REAL-5 区分 forward mismatch 与正则化失败。** 用 held-out camera/
  detector reprojection、flow-off residual map 和标定 phantom 定位误差；更强 TV
  不能替代错误几何。
- [ ] **REAL-6 没有三维真值时降低主张。** held-out reprojection、跨 session
  稳定性和与独立诊断/PIV 的一致性是间接证据，不等同于真实 volumetric truth。
- [ ] **REAL-7 检查 resolution transfer。** `lambda/delta` 随 voxel spacing、
  active volume、view 数和噪声尺度如何换算必须由目标离散化给出；16^3 成功
  不能直接声称可运行或可泛化到实验分辨率。
- [ ] **REAL-8 检查反应场偏差。** synthetic morphology 应力测试需覆盖实验中
  的 thin fronts、强梯度、遮挡、稀疏视角和时变结构；未覆盖的现象列为外推风险。
- [ ] **REAL-9 时间问题不冒充静态问题。** 若迁移到 TDBOST/4D，时间正则、
  motion 与 frame covariance 是新的目标函数；逐帧 3D PDHG 只能作为基线。
- [ ] **REAL-10 私有数据不进入公开产物。** 网站只发布聚合指标、许可允许的图
  和可重建 synthetic 配置；真实数组、VPN 论文、凭据及本机路径保留在私有库。

## 11. 何时才允许 learned proximal

下面任一项未通过，都继续做确定性 PDHG/物理诊断，不训练 learned proximal。

- [ ] **LEARN-1 确定性内核已过门。** objective、adjoint、norm、prox、边界、
  zero-regularization、调用账本和 tiny oracle 全部有自动测试与独立复核。
- [ ] **LEARN-2 强经典基线已冻结。** TV 与 Huber 的 `lambda/delta`、步长、停止
  规则和公平 PCGLS 邻居均完成开发；网络不能靠击败一个未调好的 PDHG 获胜。
- [ ] **LEARN-3 先指出可学习的真实缺口。** 例如固定 Huber 在某类 front 尾部
  存在重复、可观测的偏差；“想试神经网络”不是科学假设。
- [ ] **LEARN-4 输入在部署时可见。** 网络只能使用 `x,p,q`、逐相机白化残差、
  geometry/covariance 元数据等部署可得量，不能输入 truth error 或真实形态标签。
- [ ] **LEARN-5 输出受到约束。** 优先只学习 bounded step、`lambda/delta` map
  或受限 residual correction；保留显式 data-consistency `B/B^T` 步。若模块不
  是某个凸函数的 prox，就称 learned denoiser/unrolled update，不虚称 proximal。
- [ ] **LEARN-6 稳定性单独审计。** 检查经验 Lipschitz/Jacobian 范数、输入扰动、
  covariance/geometry shift、极端幅度和迭代外推；出现发散必须有冻结 fallback。
- [ ] **LEARN-7 数据划分按最高层单位。** train/selection/test 至少按 replicate、
  morphology、camera 或 session 隔离；相邻帧和同一场的噪声副本不得跨 split。
- [ ] **LEARN-8 比较预算完整。** 同 `F/F^T` calls、wall time、显存/内存、参数量、
  训练能耗和调参次数；预训练成本与单次/多次重建摊销分开报告。
- [ ] **LEARN-9 做身份与移除消融。** learned block 置 identity/zero、固定为经典
  prox、移除 covariance/geometry 输入，并对参数量匹配网络比较，确认收益来自
  声称的机制。
- [ ] **LEARN-10 独立确认先于 superiority。** 只有冻结网络在 fresh synthetic
  和独立真实 session 同时通过 mean、尾部、物理一致性与预算门禁，才允许讨论
  优于经典基线；否则保留为探索性结果或 NO-GO。

**允许顺序：**

```text
数学合同冻结
  -> tiny oracle / adjoint / prox / zero-lambda 通过
  -> opened-development 上调强确定性 PDHG
  -> finalist 与停止规则冻结
  -> fresh synthetic 确认
  -> 独立 flow-off covariance + real session 迁移
  -> 找到可重复、可观测的剩余缺口
  -> 才允许 bounded learned proximal
```

## 12. 红队签字页

每次正式 screen 前复制下表并填写证据路径。`口头确认`、`代码看起来正确`、
`平均值提高`都不算证据。

| 门禁 | 必需证据 | 判决 |
|---|---|---|
| G0 数学合同 | 公式、单位、配置哈希、数据状态表 | `PASS / FAIL / MISSING` |
| G1 数值内核 | dense oracle、dot-product、prox、zero-lambda 测试 | `PASS / FAIL / MISSING` |
| G2 稳定性 | norm 记录、步长依据、fixed-point/gap 轨迹 | `PASS / FAIL / MISSING` |
| G3 公平预算 | `F/F^T` 账本、wall time、预计算与内存表 | `PASS / FAIL / MISSING` |
| G4 选择纪律 | 冻结配置、候选 ledger、开封记录、停止条件 | `PASS / FAIL / MISSING` |
| G5 结果风险 | 逐场配对、形态尾部、worst 可视化、失败运行 | `PASS / FAIL / MISSING` |
| G6 fresh 确认 | 未见单位、单次冻结判决、不可回写日志 | `PASS / FAIL / MISSING` |
| G7 真实迁移 | flow-off covariance、held-out session、forward audit | `PASS / FAIL / MISSING` |
| G8 learned 解锁 | `LEARN-1` 至 `LEARN-10` 全部有证据 | `PASS / FAIL / LOCKED` |

最终判决只能取以下之一：

- `IMPLEMENTATION_NOT_READY`：G0-G2 任一未通过；
- `COMPARISON_NOT_AUDITABLE`：G3-G5 任一未通过；
- `DETERMINISTIC_PDHG_NO_GO`：实现可信，但预注册性能门失败；
- `SYNTHETIC_FINALIST_ONLY`：fresh synthetic 通过，尚无真实迁移；
- `REAL_TRANSFER_CANDIDATE`：真实外部证据通过，但 learned 仍需单独解锁；
- `LEARNED_PROXIMAL_ALLOWED`：G8 全部通过，只表示允许实验，不表示算法已赢。

这份清单最重要的作用不是让 PDHG 更容易“成功”，而是让失败发生在可解释的
位置：是目标写错、prox 写错、预算吃亏、形态尾部受损，还是 synthetic 到真实
装置的迁移断裂。只有能区分这些原因，后续算子学习才有值得学习的对象。
