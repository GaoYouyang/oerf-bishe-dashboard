# DG-WPCGLS 多种子重建试验：均值信号成立，尾部安全门失败

## 一句话结论

在真实 PSU 九相机 detector geometry、8 类解析反应场和合成 graph-correlated
noise 上，正确协方差白化的 PCGLS-4 相对 component-IID PCGLS-4 获得
`+1.178%` 平均三维场增益，按 16 个独立 replicate 聚类的 95% Student-t
区间为 `[+0.786%, +1.571%]`。但逐场 `p10=-1.029%`，`>1%` harm rate 为
`10.94%`，超过预注册的 `-0.5% / 10%` 两条安全门。因此本轮判为
**NO-GO**，不能宣称新算法优越。

## 为什么做这一步

前一轮 DG-CovGate 只回答了“每台相机至少要多少张 flow-off repeats 才能把
covariance 校准稳”，没有回答“正确 covariance 是否真的改变三维重建”。
本轮把 detector-domain whitening 组成线性算子 `L`：

```text
B(x) = L A(x)
B^T(r) = A^T L^T(r)
```

然后复用现有固定深度 PCGLS。所有方法严格使用 `4F+4A^T`，detector whitening
只在二维测量域计算，不偷加 BOST forward/adjoint calls。

## 实验冻结与可复现性

- whitening/smoke 代码在打开 smoke 结果前提交：`911ba51`；
- 16 种子协议在打开新种子前提交：`f061acb`；
- smoke 配置 SHA-256：
  `1cd60ba7aee1f01bfec5eae4cf798a0141704355e76e830f51d43937d35d48a1`；
- multi-seed report SHA-256：
  `332fa713f453087b45169acfbf49ff963c5ce5ac32f1df263e496ace757c95bf`；
- 16 replicates、每个 replicate 8 个 reaction morphologies、2 个 noise
  families、6 个方法，共 1536 条逐场 method rows；
- Apple MPS 总墙钟约 `14.50 s`，所有 solver calls 对账通过；
- MPS 独立重跑存在不超过约 `1.8e-6` 的末位浮点差异，但判决、所有 gate 和
  front-F1 完全一致。后续 confirmatory run 应使用容差比较，或在 CPU 上补一条
  deterministic audit。

## 方法对照

1. `scalar_iid`：一个全局噪声方差；
2. `component_iid`：每相机估计 u/v 分量 covariance，无空间相关；
3. `diagonal_shrinkage`：逐 detector node/分量方差收缩；
4. `dg_covgate`：held-out flow-off repeats 决定是否启用 graph covariance；
5. `oracle_covariance`：使用 synthetic generator 的真实 covariance 参数；
6. `wrong_graph_assumption`：故意把 graph covariance 分配到错误 detector
   邻域，检查“任意重新加权都能变好”的反例。

## 主要结果

| graph-correlated noise | mean gain | 95% CI | p10 | >1% harm | front-F1 gain |
|---|---:|---:|---:|---:|---:|
| DG-CovGate | +1.178% | [0.786%, 1.571%] | -1.029% | 10.94% | +0.01225 |
| Oracle covariance | +1.146% | [0.712%, 1.579%] | -1.043% | 10.94% | +0.01206 |
| Diagonal shrinkage | -0.345% | [-1.206%, 0.516%] | -7.013% | 39.06% | +0.00028 |
| Wrong graph | -1.857% | [-2.833%, -0.882%] | -9.283% | 56.25% | +0.03158 |

`dg_covgate` 与 oracle 几乎重合，说明当前失败主要不是 R=50 covariance 拟合
不准，而是“正确 GLS data metric + 固定四步 PCGLS + 原 IID 条件下选定的
Sobolev preconditioner”本身产生形态依赖的 bias/variance 交换。

逐形态看：

- plume、wavy front、double front、oblique shock、vortex pair 的平均增益为正；
- annular kernel 平均约 `-2.04%`，16 个 replicate 中 43.75% 超过 1% 伤害；
- thin front 平均为正，但 p10 约 `-1.77%`；
- oracle covariance 在 annular/thin 上出现几乎同样的尾部，进一步排除
  “只是 covariance estimator 过拟合”的解释。

## 物理与数值解释

相关噪声白化会改变 data-space 内积，也就是把法方程从
`A^T A x = A^T y` 改成 `A^T C^{-1} A x = A^T C^{-1} y`。这不是一个无害的
预处理：它改变了各 detector graph modes 对重建的权重，也改变了 normal
operator 的谱。当前 Sobolev strength=5 是在旧 IID objective 下选出的，把它
原样套到新的 whitened normal operator 上，可能对环状或极薄 front 造成
early-stopping bias。

这只是当前证据支持的**机制推断**，还不是定理。必须用 post-open
preconditioner/tempering sweep 和全新种子验证。

## 下一算法候选

工作名暂定为 **covariance-conditioned risk-controlled PCGLS**，分两层证伪：

1. 在已打开的 16 种子上扫描固定 Sobolev strength、whitening tempering 和
   stage count，确认尾部是否来自 metric/preconditioner mismatch；
2. 若存在 Pareto 改善，冻结一个低自由度规则，再用完全新的 calibration、
   field 和 flow-on noise seeds 验证 mean、p10、harm、front 与错误 covariance
   对照；
3. 只有 deterministic covariance-conditioned PCGLS 已闭合，才允许训练小型
   operator/controller。学习模型必须击败“纯白化 + 重新选定预条件器”，不能把
   deterministic GLS 的收益记到神经网络名下。

部署可见 selector 只能使用 flow-off calibration、camera/ray geometry、初始
observation 或已支付步骤的 residual；不能使用 reaction family、三维 truth 或
truth-best iteration。

## 与何远哲方向的贴合点

- NeRIF、NRIP 和 TDBOST 都依赖 measurement loss；真实 detector covariance
  可以直接进入这些 loss，而不局限于体素 PCGLS；
- 本轮证明“把 MSE 换成 covariance-weighted MSE”并不会自动保证 front-safe，
  因此有真实研究问题：如何联合 measurement model、inverse conditioning 与
  reacting-front reliability；
- 组内最关键数据仍是每相机至少 50 张未经平均的 flow-off repeats、时间顺序、
  confidence/bad-pixel mask，以及一条 held-out camera/session。

## 文献边界

- Rajendran et al. 已研究 BOS displacement uncertainty 及其经 Poisson
  integration 的传播，说明局部 sharp gradient 可把不确定度传播到全场：
  <https://doi.org/10.1088/1361-6501/ab60c8>
- Generalized least-squares CT 已明确研究 detector blur 与 correlated noise，
  所以“相关噪声 + GLS”本身不是新意：
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4201055/>
- Direct BOST-RBF 已明确指出小位移使重建易受噪声影响：
  <https://doi.org/10.1364/OE.457429>
- cone-ray BOS 证明 forward-model accuracy 会显著改变 inverse fidelity：
  <https://arxiv.org/abs/2402.15954>
- NeRIF 是本项目最直接的 neural implicit reconstruction 邻居：
  <https://arxiv.org/abs/2409.14722>

因此可辩护的新意不能写成“首次对白化噪声做 BOST”。它必须来自真实
flow-off acquisition、PSU/OERF detector geometry、同调用 inverse、
front/tail-risk contract，以及 deterministic/learned components 的严格拆账。

## 当前禁止表述

- 不说“已经优于 DeepONet、FNO、NeRIF 或 TDBOST”；
- 不说“真实 PSU/OERF 重建提升 1.18%”；
- 不把 analytic reaction morphology 写成 CFD 或实验真值；
- 不把正均值和正 CI 单独拿出来，必须同时报告 `p10=-1.029%` 与
  `harm=10.94%`；
- 不因错误 graph 的 front-F1 偶尔更高就把 front 指标当作 selector，错误
  graph 的 field tail 明显失败。

## 产物

- 配置：
  `demo_t16_operator/configs/psu_b0_dg_wpcgls_multiseed_v1.json`
- whitening：
  `demo_t16_operator/detector_covariance_whitening.py`
- runner：
  `site_tools/run_psu_b0_dg_wpcgls_multiseed.py`
- report：
  `demo_t16_operator/results/psu_b0_dg_wpcgls_multiseed/report.json`
- 逐场 rows：
  `demo_t16_operator/results/psu_b0_dg_wpcgls_multiseed/metric_rows.csv`
- 聚类汇总：
  `demo_t16_operator/results/psu_b0_dg_wpcgls_multiseed/replicate_summaries.csv`
- 四联图：
  `demo_t16_operator/results/psu_b0_dg_wpcgls_multiseed/psu_b0_dg_wpcgls_multiseed_figure.png`
