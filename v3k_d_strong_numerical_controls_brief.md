# v3k-D 强数值对照：先证明“需要学习”，再训练新模型

日期：2026-07-15

性质：`8x16x16` synthetic development、冻结 FNO、validation-only 选参、zero-new-learning numerical controls。不是 fresh blind、真实 BOST、论文 superiority 或原创算法结论。

## 一句话判决

FNO 起点的 **quadratic-step-then-projection** 是当前 validation 绝对精度冠军：source-field mean relative L2 为 `0.143855`，相对 feasible FNO 平均改善 `45.5258%`，40/40 validation fields 为正。

但它不是同预算冠军：

- geometry-normalized Landweber：`64 A + 64 A^T = 128` calls，误差 `0.147083`；
- quadratic candidate：`129 A + 64 A^T = 193` calls，误差 `0.143855`；
- quadratic 相对 geometry-normalized 的 validation 平均收益只有 `+1.2443% [95% field-cluster CI +0.5282%, +1.9583%]`，positive-field fraction `70%`，`>1%` harm rate `17.5%`。

因此，v3k-D 完成的是 **强数值基线与创新门槛**，不是新算法。learned scalar 仍明确关闭；下一步先补 projected BB1/BB2 和 matched-call stopping curve。

## 这轮回答了什么问题

v3k-C 已证明经代数校验的伴随残差方向值得保留，但只比较过 geometry-normalized Landweber 与 Jacobi。v3k-D 补齐四个会直接挑战“学习步长有价值”的对手：

1. 所有布局共用一个绝对步长的 global Landweber；
2. 每个布局用 `beta/L_m` 的 geometry-normalized Landweber；
3. 沿当前未投影二次直线使用闭式步长、随后再做硬投影；
4. 按 operator-only 谱常数分区、再用 validation truth 选表项的非神经 lookup；
5. 另从 locked FNO 与 validation-tuned ridge 两个起点重复全部对照。

## 数学合同

对布局 `m` 的 used-view 数据项：

`F_m(x) = 1/2 ||M_m(Ax-y)||_2^2`。

负梯度/伴随残差：

`r_m(x) = A^T M_m(y-Ax)`。

geometry-normalized Landweber：

`x_(t+1) = P_C[x_t + (beta/L_m) r_m(x_t)]`，`L_m=||M_m A||_2^2`。

global control：

`x_(t+1) = P_C[x_t + (beta/L_global) r_m(x_t)]`，`L_global=max_m L_m`。

投影前二次闭式候选：

`alpha_t = ||r_t||_2^2 / ||M_m A r_t||_2^2`。

必须强调：这个 `alpha_t` 只对未投影二次直线精确。`P_C` 会改变路径，所以不能称为 constrained/projected exact line search。正式措辞是 **quadratic closed-form step before projection**。

## 选择协议

### validation 之前冻结

- 同一个 FNO checkpoint 与 28 个 geometry masks；
- 非负二值硬支撑投影；
- ridge grid：`1e-6 ... 1e-1`；
- `beta={0.25,...,1.9}`、`T={1,2,4,8,16,32,64}`；
- global `L` 定义、谱区边界规则、tie-break 与统计单位；
- 每场先折叠 4 个布局，再以 source field 为独立单位。

### selection 只允许读取

- validation field relative L2；
- operator mask 与精确谱常数；
- FNO/ridge 起点和对应观测。

selection screen/commit 不计算 Q_audit、used-view reprojection 或任何 test metric。写入 `v3k_d_selection_commit.json` 后，才构造四个后置 development-domain 指标。

现有四个 test domains 已被 v3k-A/B/C 等实验查看，因此不是 fresh blind confirmation。

## validation 结果

| 方法 | 起点 | field L2 | A + A^T calls | 角色 |
| --- | --- | ---: | ---: | --- |
| Locked FNO raw | FNO | 0.257826 | 0 | 未投影网络对照 |
| Feasible FNO | FNO | 0.247739 | 0 | 0-call 前沿 |
| Ridge raw | λ=0.01 | 0.579540 | 0 | 无软窗 ridge；lambda 单独选择 |
| Feasible ridge | λ=1e-6 | 0.378348 | 0 | ridge warm start |
| Geometry Landweber | FNO | 0.147083 | 128 | 128-call 前沿 |
| Global Landweber | FNO | 0.148836 | 128 | 统一绝对步长对照 |
| Spectral lookup | FNO | 0.147083 | 128 | 退化为 geometry control |
| Geometry Landweber | ridge | 0.147581 | 128 | 双起点归因 |
| Global Landweber | ridge | 0.149248 | 128 | 双起点全局对照 |
| Spectral lookup | ridge | 0.147581 | 128 | 同样退化 |
| **Quadratic + projection** | **FNO** | **0.143855** | **193** | 绝对精度冠军 |
| Quadratic + projection | ridge | 0.144231 | 193 | 与 FNO 版本接近 |

所有固定步长方法都由 validation 选中 `beta=1.9,T=64`；lookup 的低/高谱区也都选回同一组参数。因此两区查表没有产生条件化收益。

## 精度冠军跨域表现

相对 feasible FNO：

| 域 | mean field gain | 95% field-cluster CI | positive fields | Q_audit gain |
| --- | ---: | ---: | ---: | ---: |
| validation | +45.5258% | [+41.3505%, +49.6840%] | 100% | +44.6714% |
| geometry-held-out IID | +45.3039% | [+41.0632%, +49.3932%] | 100% | +44.1693% |
| noise OOD | +25.3430% | [+22.6843%, +28.0229%] | 100% | +31.9084% |
| family OOD | +25.7371% | [+24.9711%, +26.5752%] | 100% | +26.8956% |
| joint OOD | +24.9805% | [+24.2388%, +25.7223%] | 100% | +26.4933% |

这些是相对 weak warm start 的结果。它们不能替代 strong-to-strong 比较。

## strong-to-strong 比较暴露了什么

quadratic FNO 相对 geometry Landweber：

| 域 | mean field gain | 95% CI | >1% harm rate |
| --- | ---: | ---: | ---: |
| validation | +1.2443% | [+0.5282%, +1.9583%] | 17.5% |
| geometry-held-out IID | +1.1327% | [+0.4947%, +1.8149%] | 12.5% |
| noise OOD | **-0.9128%** | [-1.7172%, -0.1066%] | 53.125% |
| family OOD | +3.3199% | [+3.1612%, +3.4811%] | 0% |
| joint OOD | +3.3346% | [+3.0839%, +3.5829%] | 0% |

这说明投影前二次步长不是“所有条件下都更好”：它在 high-noise 域平均劣于更便宜的 geometry Landweber。

FNO quadratic 相对 ridge quadratic 的 validation 平均优势只有 `+0.7441%`，95% CI `[-0.0677%, +1.5613%]`；在 family/joint OOD 中，ridge quadratic 反而略好。长迭代后起点差异明显缩小，初始化价值应转到 **短预算 time-to-target** 而不是只看 64-step endpoint。

## 投影前二次步长审计

- 10 个 split × start cells 全部没有 projected data objective increase；
- validation 的 `alpha L_m` 中位数约 `2.86-2.87`，最大约 `4.58-4.94`；
- 这不违反 fixed Landweber 的 `beta<2` 说法，因为 exact steepest-descent step 是当前 gradient Rayleigh quotient 的倒数，不是固定 `beta/L_m`；
- 投影相对 proposal 的 validation median change 约 `1.31%`，说明“投影前精确”与“投影后路径”确实不同，不能省略措辞边界。

## 尾部风险

精度冠军相对 feasible FNO 的最差 layout 仍是 noise OOD `g_011011101`：

- 平均 field gain `+19.6356%`；
- positive fraction `90.625%`；
- `>1%` harm rate `9.375%`；
- 最差单场 `-16.9406%`。

所以未来候选的主要研究目标应是降低高噪声尾部，而不是只让总体均值再改善零点几百分点。

## 条件化是否仍有研究空间

validation truth oracle 在 12 个部署规则中逐场选最优：

- 相对单一冠军平均 headroom `1.5866%`；
- 中位 headroom `0.8516%`；
- 67.5% fields 的最好规则不是 FNO quadratic；
- 最常成为 field winner 的是 ridge quadratic（14/40）、FNO quadratic（13/40）、FNO global（9/40）与 ridge global（4/40）。

这只是 **不可部署的验证真值诊断**。它说明可能存在“起点 × 噪声 × 迭代阶段”的规则选择信号，但不能证明部署特征能预测它。

粗 spectral lookup 没有利用到这部分空间。下一步必须先测试不需要网络的迭代历史规则。

## 下一道门：projected Barzilai-Borwein

令 `g_k=grad F(x_k)`、`s_k=x_k-x_(k-1)`、`y_k=g_k-g_(k-1)`：

- `alpha_BB1 = (s^T s)/(s^T y)`；
- `alpha_BB2 = (s^T y)/(y^T y)`。

第一版只做确定性 baseline：

1. BB1、BB2 与预先定义的交替 BB；
2. 分母非正/过小时回退到 geometry-normalized step；
3. 按 operator spectral bound 做 clip；
4. 每步同一个 hard projection；
5. 记录 objective、field error、Q_audit、tail 与 `A/A^T` calls；
6. validation 选择 variant/clip/stopping，之后一次性读 development domains。

BB 只用已算出的相邻迭代与梯度，不应额外调用 A/A^T。若它在相同预算下达到或超过 learned scalar，后者不值得开发。

## learned scalar 的 Go/No-Go

只有以下条件全部满足，才允许训练小型有界 scalar/stopping model：

1. projected BB1/BB2 与 matched-call stopping curve 已完成；
2. 输入只含部署可得的 residual、下降率、operator/mask/geometry、iteration 与实测 confidence；
3. 新 fields/layouts/noise realizations 与分析脚本训练前哈希封存；
4. fresh lock 上相对同调用最强数值 baseline：mean gain `>=1%`；
5. 95% field-cluster CI 下界 `>0`，positive fields `>=90%`；
6. `>1%` harm rate `<=5%`，最差单场不低于 `-10%`；
7. 每个 development domain 的 field/Q_audit 平均伤害不低于 `-0.25%`。

不过门就停止 learned step，保留“BOST 强数值重建基线与条件化失败边界”作为严谨本科成果，不继续堆 voxel/set/attention。

## 七天学习与执行

### Day 1：梯度与伴随

- 手推 `1/2||M(Ax-y)||^2`；
- 独立写 `4x6` 随机矩阵 inner-product 与 finite-difference check；
- 解释 `A^T y` 与 `A^T(y-Ax)` 的差别。

### Day 2：独立复现 v3k-D

- 运行 19 项测试、runner、validator 与 checksums；
- 从 474 行 screen 手动找回 ridge lambda、四类 solver 和两区 lookup winner；
- 从 ledger 重画 0/128/193 calls 的 validation frontier。

### Day 3：推导 BB

- 从 secant equation `H s≈y` 推 BB1/BB2；
- 构造 `s^T y<=0`、分母接近零与 clip 生效案例；
- 读原始 BB 论文时只提取“二点近似与成本”，不包装成创新。

### Day 4-5：实现 projected BB

- 先写 toy 单测，再接 v3k 数据；
- 明确 gradient 符号、fallback 与 call ledger；
- selection 函数保持 field-only。

### Day 6：尾部诊断

- 单独画 `g_011011101` 的 32 个 field gains；
- 检查 noise level、residual history、active-set change 与起点差异；
- 不删负样本。

### Day 7：给何远哲审核

只带一页结果和四个问题：

1. 真实 BOST forward 是否已有 VJP/adjoint？
2. hard support 是否在真实数据可用？
3. 目标优先级是短预算 time-to-target、64-step endpoint 还是 worst-layout tail？
4. 能否封存一套新 fields/layouts/noise 作为 fresh lock？

## 角色化文献

1. [Landweber 1951](https://doi.org/10.2307/2372313)：经典迭代与步长条件；固定梯度迭代不是创新。
2. [Piana & Bertero 1997](https://doi.org/10.1088/0266-5611/13/2/016)：projected Landweber 与 preconditioning；每步投影已有成熟先例。
3. [Barzilai & Borwein 1988](https://doi.org/10.1093/imanum/8.1.141)：two-point spectral step；相邻迭代估计曲率不是学习创新。
4. [Yuan 2010](https://doi.org/10.1007/s10107-009-0267-8)：严格凸二次上的 exact line search；用于约束“精确”措辞。
5. [Learned Primal-Dual](https://doi.org/10.1109/TMI.2018.2799231)：forward/adjoint 嵌入可训练展开已有先例；只学步长不能单独构成高质量创新。

## 可复现资产

- 配置：`demo_t16_operator/configs/v3k_d_strong_numerical_controls.json`
- 数值核心：`demo_t16_operator/adjoint_landweber.py`
- runner：`demo_t16_operator/run_v3k_d_strong_numerical_controls.py`
- 独立 validator：`demo_t16_operator/validate_v3k_d_strong_controls_results.py`
- 工程测试：`demo_t16_operator/test_v3k_d_numerical_controls.py`
- 公开结果：`demo_t16_operator/results/v3k_d_strong_numerical_controls/`
- 学习/决策主页：`strong_numerical_controls_lab.html`

独立 validator 已复算：474 validation screen rows、672 pair rows、8,064 sample metric rows、60 absolute summaries、135 pairwise comparisons、20 layout cells、10 quadratic audits、60 call-ledger rows、12 checksums；base checkpoint drift 为 0。

公开结果目录不含 checkpoint、NPZ、VPN 论文或本机绝对路径。
