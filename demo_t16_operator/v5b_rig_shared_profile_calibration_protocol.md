# v5b 设计协议：机架共享、元数据锚定的 Profile-Schur Cone Calibration

状态：`M1_DEVELOPMENT_AFTER_FORWARD_GATE_NOT_PREREGISTERED_NOT_A_LOCK`。这是由 v5a 失败导出的
候选协议。首次 M0 把 soft support 直接转成 bool，导致 320/320 体素全部激活；旧输出已封存为
缺陷轨迹。M1 使用阈值 0.05 的显式 96/320 支撑域，并且只允许使用通过 Stage 0 收敛门、源代码
哈希相符的 renderer setting。M1 已运行并通过文件校验，但其接受门在独立 audit view 上明显
不安全，因此仍只是开发期反例，不是确认性 lock。工作名暂用 **RigCal-BOST**；名字不构成
创新声明。

## 1. 要解决的真实困境

有限孔径把针孔单光线变成光线束，景深与焦平面又让不同深度的贡献发生变化。真实观测为

\[
y_s=A_{\rm truth}(\theta_*)x_s+\epsilon_s,
\]

其中 `x_s` 是第 `s` 个未知三维折射率/密度场，`θ` 是同一相机机架或采集块共享的入口瞳、
焦平面偏移及少量 pupil correction。v5a 错把 `θ` 当成每幅流场可以独立从 residual 猜出的
量，结果主要识别了 morphology。

更完整的观测模型还应写成

\[
y_{s,k}=g_k A_k(\theta)x_s+b_{s,k}+\epsilon_{s,k},
\]

其中 `g_k` 是标定后残余系统增益，而不是可随意吸收所有曝光、位移尺度、几何灵敏度和
Gladstone--Dale 转换误差的自由常数。若没有已知密度/温度目标、可信自由流边界或其他外部
幅值锚，`(g,x)→(g/c,cx)` 给出严格全局尺度 gauge；此时只能报告归一化场或复合量，不能声称
绝对场幅值已经由 BOS 数据辨识。

v5b 的核心改动是：**共享低维光学参数，分离每场高维 nuisance，再拒绝不可辨识的更新。**

## 2. 不是创新的已有组件

- 有限孔径/cone-ray BOS：Molnar 等，
  [Forward and inverse modeling of depth-of-field effects in BOS](https://arxiv.org/abs/2402.15954)。
- 多未知信号共享校准参数：blind calibration 已有长期研究，例见
  [Gribonval et al.](https://arxiv.org/abs/1111.7248)。
- profile likelihood、variable projection、Fisher information 和 Schur complement：标准统计/
  逆问题工具。
- 拒绝选项与 risk-coverage：见
  [El-Yaniv & Wiener](https://jmlr.org/papers/v11/el-yaniv10a.html)、
  [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)。
- 有限样本风险校准：见
  [Learn then Test](https://arxiv.org/abs/2110.01052)与
  [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg)。
- learned forward/adjoint correction：见
  [Lunz et al.](https://arxiv.org/abs/2005.07069)与
  [Arridge et al.](https://arxiv.org/abs/2311.12528)。

因此，不得把“cone ray + shared calibration + reject”本身写成首创。候选贡献只能是：

1. BOST 特定的低维光学参数化和 reference/flow-off 锚定；
2. 把体场 nuisance 投影掉后的可辨识性证据；
3. 嵌套留相机、条件尾部风险和真实 f-stop sweep 的完整协议；
4. 在 NeRIF/TDBOST differentiable forward 中用 JVP/VJP 实现，而非显式大矩阵。

## 3. Stage 0：先校准“尺子”，再运行联合反演

旧 M0 运行暴露了一个更早的失败：高密 truth renderer 与低密 reconstruction renderer
分别生成算子时，已知真场的 residual 只在 `3/6` blocks 命中最近孔径，operator matrix
distance 更只有 `1/6`。因此 Stage A 之前必须先过确定性的前向模型收敛门：

1. **exact-node sanity：**把真孔径放入候选库，同 renderer、无噪声时必须精确恢复；否则先查
   实现错误，不讨论可辨识性。
2. **off-grid same-renderer sanity：**同积分规则下，离网格真孔径必须由物理最近候选点排序第一；
   这只是 inverse-crime mechanics check，不是泛化证据。
3. **heterogeneous-renderer convergence：**固定一个更密 truth renderer，扫描 reconstruction 的
   path/aperture quadrature；分别报告 native normalization、共享物理尺度以及剖面掉一个全局
   scalar gain 后的 operator distance。
4. **连续有效半径：**若离散候选的最优半径长期偏离物理半径，计算
   `r_eff=argmin_r D(A_recon(r),A_truth(r*))`。若只有 `r_eff` 可稳定恢复，后续参数必须叫
   effective aperture，不能包装成真实入口瞳半径。
5. **冻结通过条件：**至少一个 reconstruction renderer setting 在所有 development rig-radius
   blocks、三种距离口径下都把最近物理半径排第一，且排名 margin 不贴近数值误差；未通过就
   禁止调 Stage A 的 ridge、prior 或 accept threshold。

Stage 0 只检查离散前向映射是否可作为尺子。它不证明未知体场下可辨识，也不证明有限孔径
算子会改善三维场 L2。

### 3.1 Stage 0 与 M1 的实际结果

Stage 0 扫描 `2 rigs × 3 truth radii × 12 reconstruction renderer settings = 72` 行，在 native
normalization、shared physical scale 和 scalar-gain-profiled 三种距离下共同检查最近候选排序。
只有 `(path samples, aperture samples)=(29,41)` 与 `(35,41)` 两个 setting 同时通过全部 6 个
rig-radius blocks；M1 冻结使用 `(35,41)`。该 setting 的三个口径都为 `6/6` 命中，最小排序
margin 分别为 `0.001990`、`0.001932` 和 `0.000564`。最后一个 margin 很小，说明允许自由标量
增益会削弱孔径区分；它不是“已解决 gain gauge”的证据。

在 M1 的 96/320 支撑域上：

- closest operator matrix、clean truth-field residual 和 noisy truth-field residual 均为 `6/6`；
- rig-shared data-SSE-only 为 `5/6`，固定 ridge 的 regularized profile 为 `4/6`；
- 事后扫描中，6 个 inner views、`lambda=1` 可得到 `6/6`，但这不是预注册结果，不能回填 M1；
- M1 当前接受 11/48，accepted inner-field L2 平均改善 `9.36%`，但 audit view 有 `5/11`
  变差，最坏 audit RMS 增加 `32.11%`。

所以 M1 只支持一个更窄的机制判断：**离散前向尺子和显式支撑错误已闭合；固定正则下的
profile 排序仍依赖视角数/正则口径，当前 outer acceptance 不能预测独立相机安全性。** 下一版
必须先比较尺度无关或按外视角选择的正则，再设计不读取 audit view 的风险门。

## 4. Stage A：共享参数的 profile objective

同一 acquisition block 有多个未知场 `x_1,...,x_S`，共享 `θ`：

\[
\hat\theta
=\arg\min_{\theta\in\mathcal C}\left[
\sum_{s=1}^{S}\min_{x_s\ge0}
\left\{\tfrac12\|W_s^{1/2}(y_s-A(\theta)x_s)\|_2^2
+\tfrac{\lambda}{2}R(x_s)\right\}
+\tfrac{\gamma}{2}\|\theta-\theta_{meta}\|_{\Sigma_{meta}^{-1}}^2
\right].
\]

- `θ_meta` 来自 f-number、焦距、焦平面、物距或 reference 图案 PSF；
- `C` 是元数据和标定误差给出的物理可行集合；
- `R` 首先用固定非负/TV 或同一 FISTA 正则，不用神经网络掩盖 identifiability；
- soft support 必须用冻结阈值显式二值化，并记录 active/total voxel；禁止用 `astype(bool)`
  把所有极小正数都激活。每个场的 measurement/unknown count 与数值秩必须进报告；
- `θ` 必须在同一机架/采集块共享，禁止每幅未知流场各自撞向网格端点；
- pilot 先用 1D effective aperture，只有它通过后才增加焦平面等第二参数。
- metadata prior 明确位于场级求和之外，只加一次；不得因场数 `S` 或相机数 `K` 改变而
  偷偷改变 prior 与数据项的口径。
- 若加入相机增益，使用 `g_k=exp(η_k)` 并由独立标定不确定度给窄先验；固定一台参考相机
  或冻结 `Ση_k=0` 只是在数值上消除 gauge，不会凭空恢复绝对物理尺度。
- 曝光/照明增益、位移估计尺度、几何灵敏度和 Gladstone--Dale 转换必须分栏命名；禁止让
  一个无约束标量吸收全部模型失配。

Stage A 的对手不是“完全不调参”而已，还包括共享 grid search、nominal metadata cone-ray、
v5a per-scene residual grid 和 oracle true operator。

## 5. Stage B：regularized profile Gauss–Newton/Schur curvature

对场 `s`，令

\[
J_{x,s}=A(\hat\theta),\qquad
J_{\theta,s}=\frac{\partial A(\theta)}{\partial\theta}
\bigg|_{\hat\theta}\hat x_s.
\]

平滑、无约束小模型中的局部 regularized profile Gauss–Newton/Schur curvature：

\[
I_{\theta\mid x,s}
=J_{\theta,s}^{\top}W_s
\left[I-J_{x,s}(J_{x,s}^{\top}W_sJ_{x,s}+\lambda H_R)^{-1}
J_{x,s}^{\top}W_s\right]J_{\theta,s}.
\]

直观解释：先减掉“仅改变三维场也能解释”的 measurement direction。若剩余能量很小，
当前相机/场形态下的孔径不可辨识；算法必须输出 `ABSTAIN_UNIDENTIFIABLE`，不能硬给半径。

术语和保证边界必须冻结：

- 加入 `λH_R` 后，这不是经典 data Fisher；metadata prior 也不能算成“数据辨识能力”。
- `λ>0` 时方括号是 regularized residual maker，不是幂等正交投影。
- 非线性 residual 的完整 Hessian 还有二阶项；这里首先只是 Gauss–Newton 近似。
- 正曲率只排除当前位置附近的连续平坦方向，不能排除多峰、边界、gauge 或全局等价。
- TV 非光滑、非负约束激活时普通 `H_R` 未必存在；首轮只用 smooth ridge 透明参考，后续
  必须基于 active-set/KKT tangent space 或平滑 TV 重写。
- 必须分栏报告 data-only、metadata-prior 与 regularized total curvature，并画完整 profile
  curve。若只有强正则/先验制造曲率，就降级为 metadata-constrained refinement。

实现顺序：

1. 8×8×5 小矩阵用显式 SVD/线性求解，验证公式和数值秩；
2. 32³ 用 HVP/CG 近似 Schur complement；
3. NeRIF/TDBOST 只使用 forward、JVP、VJP，不形成完整 Jacobian；
4. `dA/dθ` 用 float64 的 `h` 与 `h/2` 两个中心差分复核；不稳定或撞边界就拒绝；
5. 显式版与 matrix-free 版在小问题上误差必须小于预注册容差。

## 6. Stage C：嵌套 cross-view 安全层

不能用选孔径的同一批相机再证明它安全。

- inner views：拟合 `x_s, θ`；
- outer held-out views：只比较 candidate 与 fallback 的白化重投影；
- final audit camera：从始至终不可见，只用于首开报告。

每个 outer fold 定义

\[
d_{s,f}=\chi^2_{H_f}(\hat x_s,\hat\theta)
-\chi^2_{H_f}(x_{fallback},\theta_{fallback}).
\]

下面的复合风险分数仅保留为远期候选，当前不能直接使用：

\[
q_s=\max_f \operatorname{UCB}_{95}(d_{s,f})
+\alpha\,\operatorname{CIwidth}(\hat\theta)
+\beta\,\operatorname{MAD}_f(\hat\theta_f)
+M\,\mathbf 1[\text{boundary/meta conflict}].
\]

`χ²`、参数 CI 宽度与 MAD 的单位不同，`α/β/M` 尚无冻结的归一化与校准方式。首轮只用
可审计的分项判决，不把它们任意相加。只有同时满足以下条件才采用 candidate：

1. `q_s < -δ`；
2. 所有 outer fold 的 `d_{s,f}` 同号且优于 fallback；
3. data-only regularized profile-Schur 标量信息量超过冻结阈值；
4. `θ` 不撞物理边界且不与可信元数据冲突；
5. condition-risk calibration 允许当前阈值。

否则先回退 nominal metadata cone-ray；若它也未在 outer views 胜过 pinhole，则回退等预算
pinhole FISTA。最少视角若无法同时提供 inner/outer/audit，直接报告“不支持该安全层”，
不能复用视角伪造独立证据。

M1 的实际结果否掉了当前门：11 个 accepted 样本虽然在内层 field L2 上平均改善 `9.36%`，
但 `5/11` 的 audit RMS 增加，最大增幅 `32.11%`；按“样本百分比先平均”和“均值 RMS 再算
百分比”两种口径，audit 总体变化也分别为 `-1.61%` 与 `-2.50%`。因此下一版不能沿用 M1
阈值，只能把这些数字当作新门的失败需求：outer score 必须先在独立 development 数据上证明
能排序 audit harm，不能直接因为 outer residual 变小就接受。

## 7. 数据设计：必须是配对析因，不再只做边际平衡

v5a 分别平衡 family、radius 和 noise 后独立打乱，未形成完整 `family × radius × K` 析因，
因此锁后分组只可描述、不可作因果归因。

v5b pilot 对同一个 latent field、同一个 mask 和配对噪声实现，完整扫过 true aperture：

- `2 rigs × 3 fixed f-stops × 2 unseen families × 5 fields = 60` 个场级单元；
- 每个场级单元再配对 `K=3/4/5`，但统计 bootstrap 以 latent field/rig 为 cluster；
- truth forward 使用比 reconstruction 更密、且代码路径独立的 Monte-Carlo/cone-ray renderer；
- true aperture 使用 off-grid 连续值，禁止总落在候选节点；
- 每个 acquisition block 的 `θ` 固定，场形态变化但硬件参数不变；
- development、threshold-calibration、confirmatory lock 的 family/rig/seed 全部互斥。

独立性口径：`aperture × K` 是同一 latent field/rig 的配对重复测量，不是新增独立样本。
上述 60 行若复用 latent field，只有 `2 rigs × 2 families × 5 fields = 20` 个 field/rig
bundles，且共享 `θ` 的最高随机层只有 6 个 rig×f-stop blocks。统计层级必须在预注册时固定，
不能结果出来后在 field 与 block bootstrap 之间选择。

当前 v5a lock 已经“烧掉”，只能用于设计动机和失败诊断，不能进入 v5b 的阈值、权重或
确认性数字。

## 8. 基线与公平合同

### 小模型机制阶段

1. pinhole FISTA / PBB；
2. nominal metadata cone-ray FISTA；
3. v5a per-scene full-residual hard grid；
4. per-scene cross-view grid；
5. soft operator marginalization；
6. rig-shared grid/profile calibration，无 identifiability gate；
7. RigCal-BOST 完整三阶段；
8. oracle nearest bank 与 oracle true operator，只作 headroom。

### 放大和公开数据阶段

在输入/输出、分辨率、split 与训练预算能严格匹配后，再加入 TV/CGLS、NeRIF/NIRT、
DeepONet、FNO、F-FNO/FFNO、GINO/VIDON 或 Learned Primal-Dual。必须沿用各论文原指标，
同时补充共同指标；不同任务或数据上的表格不得直接宣称谁优于谁。

所有方法共同记录：

- forward、adjoint/JVP/VJP 调用；
- exact/iterative linear solves；
- wall time、peak memory、参数量、训练时间；
- 相同相机、相同噪声 realization、相同 support 和停止预算；
- baseline 调参只用 development，不看 confirmatory lock。

## 9. 指标

### 三维场

- relative L2、gradient relative L2、front F1；
- mass relative error、centroid error；
- 薄前缘位置/厚度误差（有定义和真值时）。

### 光学与校准

- aperture/entrance-pupil MAE、置信区间覆盖、boundary-hit rate；
- inner、outer、final-audit whitened reprojection；
- mean of per-sample percent changes 与 percent change of mean RMS；
- accepted-only audit increase rate 和最大增幅；
- estimated `θ` 对 family 的泄漏，可用 leave-one-family-out classifier/AUC 作诊断。

### 选择性风险

- coverage；
- overall mean/p10/harm；
- accepted-only mean、p10、`P(gain<-1% | accept)`；
- risk-coverage curve 与 AURC；
- 每个 family/rig/noise/K 的 worst-group 指标。

## 10. 统计和样本量

- 主比较为同 latent field/rig/noise 的 paired difference；
- 置信区间的最高 cluster 层必须按 estimand 固定：共享校准误差以 acquisition block 为层，
  field outcome 以预先定义的 field/rig bundle 为层；不能把配对 aperture/K 重复当独立样本；
- family/rig/noise 是预注册分层，不在结果出来后挑最好子组；
- harm 用 exact Clopper-Pearson 单侧 95% 上界，不只报点估计；
- 若观察到 0 个 harm，要让上界低于 5%，至少需要 59 个 accepted 样本；零 accepted 必须
  判 `INCONCLUSIVE`，不能写成 0 harm；
- coverage 为 20% 时，`N=295` 只是期望得到 59 个 accepted，实际达到至少 59 个的概率约
  52.3%；90%/95% 把握约需 340/354 个独立 primary units；
- 一个更稳妥的确认设计候选为 864 个冻结 primary field bundles。若 coverage 主张也要求
  单侧 95% CP 下界至少 20%，需至少 193 个 accepted；在 `m=193` 时允许至多 4 个 harm，
  风险的单侧 95% CP 上界约 4.68%。`m=173,h<=3` 虽能把风险上界压到约 4.42%，却不能同时
  支持 20% coverage 的置信下界。最终数字仍须按冻结 deployment mixture 复算；
- 固定因子配额不等于 iid mixture。若采用分层固定配额，不能同时把普通 pooled CP 称为
  exact，需预注册分层/Poisson-binomial 或 conformal risk-control 口径；
- 60 场 pilot 只能快速证伪，不能证明安全。

## 11. 分阶段 Go / No-Go

### Gate A：identifiability mechanism

- Stage 0 的 exact-node、same-renderer 和 heterogeneous-renderer convergence 全部闭合；
- 同一场的 paired aperture 排序明显优于 v5a residual margin；
- data-only profile-Schur curvature 与 aperture MAE、raw gain、harm 有稳定单调关系，且该
  关系不能由增大正则强度单独制造；
- estimated `θ` 不再被 family 分类器轻易识别；
- 不通过则停止 blind/profile calibration，转向已知元数据 cone-ray baseline。

### Gate B：pilot selective safety

- coverage ≥20%；
- overall mean gain ≥2%；
- accepted-only p10 ≥0；
- accepted conditional harm 点估计 ≤5%；
- outer/final audit 的两种平均口径均不上升；
- 最坏 family/rig 平均 gain 不为负；
- 仅作 pilot 信号，不宣称统计安全。

### Gate C：confirmatory synthetic lock

- coverage 的单侧 95% 下界 ≥20%；
- accepted harms 的单侧 95% 上界 ≤5%；
- 完整 Gate B 复现；
- source/config/split/threshold 预注册并外部提交后首开；
- independent validator、tamper test、样本级 CSV 和 checksum 全部通过。

### Gate D：真实 f-stop sweep

- 同一装置至少 3 个固定 f-stop，reference/flow-off 与未知流场都保留；
- 不使用体真值时，主终点为 outer/audit displacement、物理可行性和 downstream 指标；
- 无真实 sweep 时，论文结论只能是 synthetic surrogate，不得写成解决真实 BOST。

## 12. 现在需要何远哲确认

1. 是否保存每次实验的 f-number、焦距、物距、焦平面和相机型号？
2. reference/flow-off 原图能否用于估计 background-dot PSF/blur？
3. 组内 forward 是 thin ray、cone ray、bilinear ray tracing 还是经验 distortion correction？
4. NeRIF/TDBOST 是否能提供对体场和光学参数的 JVP/VJP？
5. 同一机架能否在多个流场/时间帧之间共享校准参数？硬件是否会漂移？
6. 能否做一次 3 个 f-stop 的小型控制实验，并封存一条 audit camera？
7. 是否有已知密度/温度、自由流边界或其他绝对幅值锚？相机间的相对增益怎样标定？
8. 组内所谓孔径参数代表机械光圈、有效入口瞳还是通过 PSF 拟合得到的 effective aperture？

## 13. 实现顺序

1. `[done]` 通过 Stage 0 renderer convergence，冻结 M1 的算子单位、quadrature 和有效半径语义；
2. `[done]` 写 M1 paired counterfactual dataset、显式 96/320 支撑域和 cluster manifest；
3. `[done]` 在 8×8×5 显式矩阵上实现 rig-shared profile grid、regularized profile-Schur、outer
   folds 与三层 oracle；
4. `[next]` 把 SSE 与 penalty 分开，以 held-out inner/outer view 或尺度规范选择 `lambda`，并检验
   结果是否跨 3/4/5/6 views 稳定；
5. `[next]` 在完全不读取 audit view 的 development 上拟合 risk score，再冻结新 family/rig/seed；
6. 首开新 lock，失败则保留；
7. 只有 Gate A/B 有信号，才实现 matrix-free JVP/VJP 和 NeRIF 接口；
8. 只有 synthetic confirmatory gate 通过，才申请/使用组内真实数据做 f-stop sweep。

确认性实现还必须拆成两条进程边界：fitter 只能看到 `VisibleBlock`（不含 family、true θ、
truth field、truth operator 和 final pixels）；先写并外部提交 `decision_commit.json`，再由独立
`open_final_audit.py` 校验 hash 后打开 final camera。开发期同进程读取 truth 只能标为 M0
development first-open/debug，不能称 confirmatory first-open。

## 14. 直接碰撞的 prior art 边界

- [Golub & Pereyra 1973](https://epubs.siam.org/doi/10.1137/0710036)：variable projection
  消去线性变量；“profile 掉体场”不是新意。
- [Pathuri-Bhuvana et al. 2020](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0001931.pdf)：
  约束 VarPro 联合层析重建和换能器位置校准；“低维 calibration + tomography”不是新意。
- [AIDA, Hom et al. 2007](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-24-6-1580)：
  官方摘要支持 multi-frame/3-D myopic deconvolution，以及 maximum-likelihood 与
  regularization 的自动平衡；在取得正文精确页码前，本项目不再用它支持“多个变化对象共享
  时不变 PSF / reference-PSF statistics”这一更强陈述。
- [Robinson & Milanfar 2006](https://doi.org/10.1109/TIP.2006.871079)：成像中用 Schur
  information complement 去 nuisance；Schur 工具不是新意。
- [Raue et al. 2009](https://academic.oup.com/bioinformatics/article/25/15/1923/213246)：
  完整 profile likelihood 区分结构/实际不可辨识，并说明局部 Hessian 可能误导。
- [Molnar et al. 2024](https://doi.org/10.2514/1.J064095)：BOS 有限孔径 cone-ray forward/
  inverse 与真实 f-number 已有；不能声称首次引入 cone ray 或 f-stop 验证。
- [Gribonval et al. 2012](https://arxiv.org/abs/1111.7248)与
  [Kech & Krahmer 2017](https://doi.org/10.1137/16M1067469)：多信号盲标定仍需结构先验和
  gauge 固定；尺度歧义不是本项目发现的新现象。
- [Brynjarsdóttir & O'Hagan 2014](https://doi.org/10.1088/0266-5611/30/11/114007)：
  模型失配可与物理校准参数混淆；仅提高拟合不能证明参数物理正确。

因此最窄、仍待证伪的贡献是：在固定 acquisition rig 的有限孔径多视角 BOST 中，绘制
“流场 nuisance 与有效孔径何时可分”的可辨识性地图，并验证一个约束感知的 profile-curvature
拒绝分数是否比 residual margin 更能预测校准收益与失败；最终证据必须来自独立场、独立 rig、
真实 f-stop 和最终保留相机。

RigCal-BOST 目前只是一个可证伪假设。论文价值不来自名字，而来自它是否能在形态、机架、
孔径、噪声和真实成像变化下减少 catastrophic tail，并且每一步都能由外部证据复核。
