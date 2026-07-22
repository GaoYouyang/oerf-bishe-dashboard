# 多帧三维 BOST `q_cal` 切空间辨识力预注册

> 状态：`PREREGISTERED_POSTOPEN_SYNTHETIC_PROTOCOL`
>
> 证据边界：只验证多帧 nuisance tangent 的线性代数机制；不使用 OERF 私有数据，不声称真实标定、真实重建、算法突破或论文成功。
>
> 上游事实：单帧自由体素场的 field operator 在当前离散化中满行秩，因此 data-only profile Gram 为零。正则化得到的曲率必须称为 prior-conditioned。

## 1. 本轮只回答什么

共享相机几何的多帧模型写成

```text
y_t = A(eta) x_t + epsilon_t,        t = 0,...,T-1
C_t[:,j] = d(A(eta)x_t)/d eta_j
```

把所有帧堆叠后，令 `B` 为允许场模型的局部 nuisance Jacobian，`C` 为五维几何 Jacobian。纯数据局部几何信息为

```text
S = C^T (I - P_col(B)) C.
```

本轮问题不是“多帧是否一般更好”，而是：**哪一种明确的跨帧场约束，能让几何响应离开 nuisance tangent；这种信息是否伴随可接受的场模型残差和局部参数恢复？**

## 2. 五类必须同时出现的模型

| 编号 | 场模型 | nuisance 参数 | 预期含义 |
|---|---|---|---|
| C0 | 每帧独立自由体素 `x_t` | `T*n` | 若每帧 `A` 满行秩，必须保持 rank `0/5` |
| C1 | 未知低秩分解 `X=Phi H` | `Phi` 与 `H` 都可变 | 低秩因子化本身不等于几何信息；需做切空间反例 |
| C2 | 固定时间系数 `H`、自由空间因子 `Phi` | `Phi` 可变 | 在 `A` 满行秩时，几何响应仍可被 `delta Phi` 吸收 |
| C3 | 同一静态场重复 `T` 次 | 一个共享自由场 | 防止把重复帧数或独立噪声误写成动态信息 |
| K0 | 已知固定标定 target | 无场 nuisance | 正控制；必须恢复 raw geometry 的五维秩 |
| P0 | 从独立 fit sessions 冻结空间基 `Phi_fit` | 每帧系数 | 只能称 training-prior-conditioned；同时报告表示误差和 fit/audit 分离 |
| T0 | 已知输运 `x_t=W_t x_0` | 共同初场 `x_0` | 只有单个共同状态要解释全部帧，可能产生条件性信息 |
| T1 | 输运 + 共享新生项 `x_t=W_t x_0+a_t s` | `x_0`、共享 source `s` | 反应流的最小 birth/heat-release proxy；检验模型灵活性与信息的折中 |
| T2 | 输运 + 每帧自由 innovation | `x_0` 和每帧自由场 | 必须作为“灵活性重新杀死辨识力”的负控制 |

`P0/T0/T1` 即使通过，也不叫 data-only universal observability：它们分别条件于冻结训练基或冻结动力学结构。

## 3. 合成场和真实物理对应关系

- 体素网格、六相机和五个 scaled pose modes 与上游 `q_cal` 诊断同一 forward family；不混用连续 teacher 与离散 nuisance operator。
- 基础场来自同仓库 analytic BOST phantom，并在体素网格上采样。
- `W_t` 使用零边界、三线性半拉格朗日平移，代表已知平均对流的最小模型，不代表真实反应流求解器。除完全冻结速度外，还要把一个全局 velocity-scale tangent 加入 nuisance，检查“速度也未知”后信息是否仍在。
- reacting stress 在序列后半段加入局部共享 source，代表新生/热释放导致的非输运成分；它不是火焰化学真值。
- 冻结 PCA 基必须由与 audit seed 不同的 fit sessions 得到；同时保留 voxel-teacher PCA ceiling 与 reported-geometry noisy pilot reconstruction PCA。

## 4. 几何导数和局部恢复

先对整个 dense voxel operator 做中心差分，而不是为每个场挑不同步长：

```text
D_j = (A(+h_j e_j)-A(-h_j e_j))/(2 h_j)
C_t[:,j] = D_j x_t.
```

步长 ladder 与上游一致。选择第一个连续两档相对变化不高于 `5e-3`、且差分信号高于 `100 eps` cancellation floor 的细档。没有平台则整 rig fail closed。

局部参数恢复使用小的组合 pose perturbation，观测由真实 perturbed voxel operator 产生；候选模型只看 reported-geometry operator、独立 fit prior 和 noisy observation。对每个模型：

1. 在 `B` 上拟合 nuisance；
2. 用拟合场构造 `C_est`；
3. 在 `col(B)` 的正交补上最小二乘求五维 `eta`；
4. 报告 `eta` relative-L2、物理单位误差、profile rank、condition number、nominal model residual 和校正后 whitened residual。

teacher `C` 只作 evaluator ceiling，不能参与候选决策。

## 5. 预冻结数值门

- operator finite-difference 最大相对变化 `<=5e-3`；
- nuisance SVD 正交缺陷 `<=1e-10`；
- profile symmetry/PSD 缺陷相对 raw 最大特征值 `<=1e-10`；
- `B^T(I-P_B)C` 相对正交缺陷 `<=1e-10`；
- 所有 CSV/JSON 数值有限；rank 使用 raw Gram 最大特征值乘 `rcond`，禁止按近零 profile 自归一化。

## 6. 预冻结机制门

### 6.1 结构负控制

跨所有 rig：

- C0、C1、C2 的 profile relative rank 必须为 `0/5`；
- C3 的 profile relative rank 必须为 `0/5`，K0 必须为 `5/5`；
- trace retention 与 max-eigenvalue retention 必须都 `<=1e-10`。

任一失败，不能解释为“低秩带来可辨识性”，先审计切空间实现。

### 6.2 精确输运条件信号

T0 仅在以下条件同时满足时记为 `CONDITIONAL_SIGNAL`：

- 每个 rig 的 teacher profile rank 为 `5/5`；
- 最差 trace retention `>=1e-4`；
- nominal clean model residual `<=1e-10`；
- deployable `C_est` 的参数 relative-L2 中位数 `<=0.25`、最大值 `<=0.75`；
- 校正后 whitened RMS residual 中位数 `<=2.0 sigma`。

这仍只证明“在精确已知输运的合成局部模型中有信号”。

### 6.3 失配与 abstention

- velocity mismatch、未建模 birth 或 OOD PCA 若校正后 residual `>2 sigma`，必须拒答；
- 错误 transport 的帧序/时间方向必须作为 negative control；若仍被授权且参数错误，计入 false accept；
- 若 residual 看似通过但参数 relative-L2 `>0.75`，记为 `UNSAFE_FALSE_ACCEPT`；出现任何 false accept，总体不得进入算法候选；
- T1 只有同时降低 reacting residual、保持 profile rank `5/5`，且参数尾部不差于 T0-on-reacting，才算有后续研究价值；
- T2 必须恢复 rank `0/5`，防止把无限制 innovation 写成改进。

## 7. 固定输出

- `report.json`：合同、门、调用成本、允许/禁止结论；
- `model_metrics.csv`：逐 rig、scene、model 的 profile 与表示误差；
- `q_trials.csv`：逐局部 perturbation 的恢复误差、residual 与 abstention；
- `fd_metrics.csv`：逐 rig、mode 的 operator-level 差分平台；
- `temporal_qcal_tangent_diagnostic.png`：信息、模型残差、参数恢复和创新灵活性四联图；
- `checksums.sha256`：同轮产物摘要。

## 8. 无论结果如何都禁止的句子

- “多帧自动解决了 BOST 标定”；
- “低秩分解使几何可辨”；
- “已优于 DeepONet/FNO/TDBOST/NeRIF”；
- “已在真实反应流、真实 OERF 数据或跨 rig 上泛化”；
- “这就是毕业论文算法”或“可以投稿”。

下一道真实门仍是何远哲师兄确认：真实序列的共享几何、timestamp、curved/straight callable、JVP/VJP、场表示、相机标定协方差、birth/extinction 频率、主基线和允许的 fit/audit split。

## 9. v1 打开后的纠错附录

> 分类：`POSTOPEN_CORRECTIVE_AMENDMENT`。以下门是在 v1 结果打开后形成，不能反写成 v1 的预注册门；v2 仍是 development diagnostic，不是 fresh confirmation。

v1 保留在 `learning_labs/results/temporal_qcal_tangent_v1/`。它给出三个重要事实：

1. 精确输运把 profile rank 提到 `5/5`，但 reference noise 下参数 relative-L2 中位数为 `9.41`；
2. 旧 residual 归一化除以 `sqrt(m)`，没有扣除 nuisance 与五个几何参数，自由度错误；
3. 独立审计发现 q-trial 循环把最后一个 reacting scene 的 `fields` 误传给所有 `teacher_*` 列。该作用域错误不影响 deployable `q_hat` 和 v1 `NO-GO`，但 v1 所有 `teacher_*` 列作废。

v2 必须执行以下纠正：

- teacher field 按每个 model 的 scene 显式选择，并加零噪声局部余项回归测试；
- v2 的 artifact schema 可更新，但随机实验继续使用 v1 seed namespace，保证 q direction、phantom 与噪声方向配对；不得因版本号变化偷换案例；
- `C_est` 由同一带噪 observation 拟合得到，只能报告 `plug-in covariance proxy`，严格 CRLB 只允许 teacher `B/C/Sigma`；
- residual 使用 `dof=m-rank(B)-5`，报告 `NIS=||r||^2/sigma^2` 与双侧 99% chi-square 区间；
- 使用 `E=G_raw^(-1/2) S G_raw^(-1/2)` 的广义特征值，避免 trace retention 掩盖最弱几何方向；
- reference-noise 更新只有在 NIS 通过、plug-in 95% 最大半径不超过 `0.25 q_ref`、`||q_hat||+r95<=0.1` 且 Wald 更新显著时才授权；这些是 post-open synthetic screen，不是相机工程容差；
- SNR multiplier 扫描固定同一场、同一 q、同一标准噪声方向，只缩放 `Sigma=rho^2 Sigma_0`。它只定位 practical-identifiability 区间，不可称独立重复、coverage 或泛化；
- v2 的 `shared_birth_research_signal` 必须同时过参数尾部与不确定度门，不能再由“rank + nominal residual”单独通过；
- 错误 transport、10% velocity mismatch 与未建模 birth 在 reference noise 下只要有任何 uncertainty-authorized case，就判 mismatch gate 失败。

v2 仍不完成以下工作：每格 500 次独立噪声 bootstrap、60 个 fresh audit rigs、held-out camera/frame NIS、迭代 nonlinear variable projection、真实 transport/innovation 误差和真实 BOST covariance。它们是进入 estimator-audit 阶段前的下一组门。
