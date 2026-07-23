# JACRU-M2.3 至 M2.8：measurement-space 投影主线的打开数据判决

**日期：** 2026-07-17  
**总状态：** `OPENED_SYNTHETIC_MECHANISM_AUDIT_NO_METHOD_AUTHORIZATION`  
**证据边界：** 12³ synthetic T0；development 与 exploratory OOD 均已打开；exact SVD、exact Jacobi、exact camera-block 都是不可部署 oracle；fresh/final 与真实 BOST 均未打开。

## 1. 先说结论

这条支线没有得到“新算法已经成功”的结论，但把真正的瓶颈从三个错误方向中分离了出来：

1. M2.3 的 `Ax=Ax_ref` 目标本身被 CGLS-12 弱 anchor 锁死，预条件器无法改变其极限。
2. M2.4 改成正确的 `Ax=y` 仿射目标后，场收益可以保留；identity CG 的有限预算收敛太慢。
3. M2.5 证明逐 measurement Jacobi 不是答案；M2.6 证明 camera-local coupling 才是主要谱结构。
4. M2.7 证明 exact camera-block 在 23–24 次总 forward 预算内已经足够快，但同一个 single-interface case 在六个网络种子上稳定受害。
5. M2.8 进一步证明：固定全局插值救不了；即使 evaluator 看真值并逐样本选择最佳连续插值，也无法同时满足逐样本重投影与尾部场误差。

因此当前阻断项已经不是“PCG 还不够快”，而是**把含噪、带 camera bias 的测量值当作必须精确拟合的仿射目标，会在部分界面场上伤害三维恢复**。下一步应研究 noise-aware target、discrepancy stopping、robust/covariance data fidelity 与 fail-closed fallback；停止继续堆 Jacobi、Nyström 或 learned SPD。

## 2. 六轮证据放在一张表里

| 阶段 | 问题 | 关键结果 | 严格判决 |
|---|---|---|---|
| M2.3 | 固定步 measurement-space CG 能否逼近 M2.2？ | 最好 development reprojection 仍约为 matched CGLS 的 `14.79x`；且收敛极限只能回到 `Ax_ref` | `NO-GO`；target 错，不是单纯 solver 慢 |
| M2.4 | 改为 `Ax=y` 后是否可行？ | exact affine oracle residual 约 `6e-16`；identity CG 在 K=4 仍约 `15.93x` | `NO-GO`；目标可达，有限预算太慢 |
| M2.5 | exact `diag(AAᵀ)` 能否加速？ | JACRU 最好约 `15.19x`；谱条件数平均反而恶化 | `NO-GO`；关闭 Hutchinson diagonal |
| M2.6 | exact camera-block 能否利用相机内耦合？ | K=12 的 JACRU field gain `39.01%`、reprojection `0.270x`，但 harm `8.33%`、worst `-9.31%`；CNN worst `-12.31%` | 强机制信号、整体 `NO-GO`；setup 为 `1001F-equiv` oracle，K=12 还超 24-call 主预算 |
| M2.7 | 在 K≤10、总预算≤24 时是否有联合可行点？ | K=9 已过 mean reprojection：JACRU `0.852x`、CNN `0.914x`；但 harm 仍 `8.33%`，worst `-8.89% / -11.89%` | `NO-GO`；速度问题已解决，target/no-harm 冲突未解 |
| M2.8 | 固定插值或逐样本最优校准能否救尾部？ | 固定 alpha 无通过点；truth-oracle 在 K=10 仍只有 `97.22%` dev rows 存在可行 alpha，问题界面 case 的最优 alpha 约 `0.99` 仍为负增益 | `NO-GO`；简单 interpolation/calibration family 封口 |

## 3. 数学上到底发生了什么

M2.4 以后使用

```text
b = A x_net - y
(A A^T) z = b
x_k = x_net - A^T z_k
```

exact solve 把网络输出投到 `{x: Ax=y}`。camera-block preconditioner 明显改善了 `AAᵀ` 的有限步条件，因此 M2.7 在 K=9 就通过平均重投影门。这说明 M2.6 的谱诊断是真信号。

但 measurement target 含有 synthetic noise 与 camera bias。越接近 `Ax=y`，越可能把这些误差写回欠定的三维场。受害样本固定为 development 的 `single_interface / base_seed 2113`；JACRU 与 pooled CNN 的三个模型种子都失败，不能归因于一次训练偶然性。

M2.8 定义简单插值

```text
x(alpha) = x_net - alpha * (x_net - x_pcg),  0 <= alpha <= 1.
```

对每个样本，measurement residual 与 field squared error 都是 alpha 的凸二次函数。evaluator 先精确求出满足

```text
||A x(alpha)-y|| <= 1.1 * ||A x_CGLS-y||
```

的 alpha 区间，再使用真值选择其中 field error 最小的 alpha。这比任何可部署校准器都强。如果这个上界仍失败，固定 alpha、只看 residual 的 selector 或更复杂的 alpha 网络都不能被当前证据授权。

结果是 K=9 的可行率只有 JACRU `83.33%`、CNN `69.44%`；K=10 都是 `97.22%`。即使对可行的受害 case，truth-optimal alpha 仍约为 `0.988–0.992`，field gain 为负。故简单混合不能解除冲突。

## 4. 现在停止什么

- 停止 Hutchinson diagonal：它只是在较低 setup 成本下估计已经失败的 exact diagonal。
- 暂停 block-plus-low-rank、Nyström 与 learned SPD：它们只会更快到达同一个有尾部伤害的 `Ax=y` target。
- 不把 M2.6 的均值收益写成算法成功：exact block setup 为 `1001F-equiv`，不在效率预算中。
- 不打开 fresh/final：所有 target、K、alpha 与失败 case 都已在 opened T0 上看过。
- 不宣称 JACRU 优于 CNN：两者表现接近，当前信号来自通用 affine solver，而不是 set encoder。

## 5. 下一步三个最小候选

### N1：噪声感知 discrepancy target

不要要求 residual 逼近近零，而是要求 whitened residual 到达 flow-off noise 与 calibration uncertainty 能解释的范围：

```text
||C_noise^(-1/2)(Ax-y)||^2 approximately measurement degrees of freedom.
```

先比较经典 early-stopped PCGLS、Tikhonov 与 Huber data fidelity；只有固定方法出现联合可行区，才允许学习 stopping/regularization operator。当前没有真实 flow-off covariance，因此 N1 只能先做 synthetic stress，不能称 OERF 实验验证。

### N2：独立观测 fail-closed fallback

保留一台 audit camera、一个未用于 reconstruction 的 ray subset 或一个独立 renderer。候选只有在 held-out evidence 改善时接管，否则返回 matched CGLS/robust baseline。选择器不得读取 truth、family label 或最终评分。

第一版不要训练大网络，只测试固定阈值：whitened residual、camera-wise residual concentration、correction norm、held-out reprojection 与 uncertainty interval。若这些可观测量连受害界面 case 都识别不了，就停止 selector 路线。

### N3：robust/covariance affine proximal

把 raw least-squares target 换成显式稳健数据项，例如 camera-block covariance + Huber/Student-t residual，并把 learned proposal 只作为 proximal center：

```text
argmin_x  rho(C_noise^(-1/2)(Ax-y)) + beta ||x-x_net||^2.
```

先用固定 beta、固定 robust scale 和经典 solver 建立 Pareto ceiling。只有 ceiling 同时通过 field/H1、clean/held-out reprojection、harm、worst 与总成本，才讨论 geometry-conditioned beta 或 neural operator。

## 6. 现在必须向师兄确认的数据合同

1. 是否有同一 camera/geometry 下的 flow-off temporal repeats，可估计 detector covariance？
2. 是否能永久留出一台 audit camera，或至少留出不参与重建的像素/ray block？
3. 何远哲当前 BOST forward 是否提供可调用的 `A`、`A^T/J^T`、ray、mask、camera pose 与单位？
4. 实验中主要误差来自背景图像噪声、标定漂移、有限孔径，还是 BOS optical model mismatch？
5. 是否有 phantom、CFD/PIV/压力或其他独立三维/二维物理证据，而不是只用训练相机重投影？
6. 同一 geometry 会复用多少帧？这决定 camera-block/covariance setup 能否摊销。

## 7. 可复现入口

- M2.3/M2.4 独立 validator：`site_tools/validate_jacru_m2_3_m2_4_evidence.py`
- M2.5/M2.6 独立 validator：`site_tools/validate_jacru_m2_5_m2_6_evidence.py`
- M2.7/M2.8 独立 validator：`site_tools/validate_jacru_m2_7_m2_8_evidence.py`
- M2.7 packet：`demo_t16_operator/results/jacru_m2_7_target_no_harm_pareto_ceiling_postopen_public/`
- M2.8 runner：`site_tools/run_jacru_m2_8_interpolation_calibration_ceiling.py`
- M2.8 packet：`demo_t16_operator/results/jacru_m2_8_interpolation_calibration_ceiling_postopen_public/`
- 预条件器红队：`docs/jacru_m2_5_preconditioner_design_red_team_2026-07-17.md`
- 一级来源原创性审计：`docs/jacru_m2_3_primary_literature_novelty_audit_2026-07-17.md`

## 8. 论文表述边界

当前可写：

> 在打开的有限孔径 BOST synthetic fixture 上，exact camera-block measurement-space PCG 显著缩短 affine data-consistency 收敛，但在严格端到端调用预算下，测量拟合与单界面场尾部误差仍存在联合冲突；固定插值与 truth-oracle calibration ceiling 均未解除该冲突。

当前不可写：新算法优于 DeepONet/FNO、真实 BOST 泛化成功、可部署加速、首次提出 affine projection、可发表高水平论文已完成。下一篇可辩护工作的核心应是**噪声/失配感知的 BOST data-consistency target 与可核验回退**，而不是再给 PCG 加一个网络。
