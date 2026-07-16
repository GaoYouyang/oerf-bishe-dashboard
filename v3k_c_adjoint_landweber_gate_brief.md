# v3k-C 伴随方向闸门：先把 Landweber 做成必须击败的基线

日期：2026-07-15

性质：`8x16x16` synthetic development、冻结 FNO warm start、zero-new-learning numerical refinement。不是完整 zero-learning reconstruction，不是 blind final、真实 BOST、superiority 或原创性结论。

## 一句话判决

离散 forward/adjoint 数学合同通过；在修复“反复乘软 support 窗”的混杂并从 validation selection 中移除 Q_audit 后，几何归一化 projected Landweber 仍把 validation source-field mean relative L2 从 `0.247739` 降到 `0.147083`。按 40 个独立场先折叠四个布局后计算，相对可行域 FNO 的平均收益为 **+44.6024%，95% field-cluster CI [+40.1023%, +49.1099%]**。

因此，伴随残差是下一阶段值得保留的物理方向；但 Landweber、Jacobi、support projection 和全局步长都只是强基线。当前只允许开发“受谱界约束的样本条件化步长”原型，不允许声称已有新算法、论文 superiority，或直接扩大 voxel attention。

## 先读懂四行数学

对布局 `m`，定义 used-view 数据项：

`F_m(x) = 1/2 ||M_m(Ax-y)||_2^2`。

它对三维场的梯度是：

`grad F_m(x) = A^T M_m(Ax-y)`。

因此负梯度/伴随残差为：

`r_m(x) = A^T M_m(y-Ax)`。

本轮标准更新是：

`x_(t+1) = P_C[x_t + (beta/L_m) r_m(x_t)]`，其中 `L_m = ||M_m A||_2^2`、`0 < beta < 2`。

`P_C` 只做非负与二值硬支撑域投影，满足 `P_C(P_C(x))=P_C(x)`。数据生成器中的余弦 soft window 只用于定义 phantom，不再在每轮迭代中重复相乘。

必须会解释的三件事：

- `A^T y` 是把观测反投影回场空间；`A^T(y-Ax_0)` 才描述当前预测在声明 forward 下的数据不一致方向。
- 若 `A=[A_1;...;A_K]`，则 `sum_i A_i^T(y_i-A_i x)=A^T(y-Ax)`；“pooled residual”与“per-view residual 简单求和”数学等价，不能当作两项创新。
- 这里的 `beta/L_m` 是每个布局按精确谱常数归一化的步长，不是所有布局共用一个绝对 `alpha`。

## 数值合同与选择协议

### Forward 的真实含义

- `A` 的 shape 是 `[9,16,256]`；每个深度切片独立把 `16x16` xy 场投影成 9 个视角、每视角 16 个 detector bins。
- 8 个 z 切片复用同一个二维线性算子；当前 toy forward 没有跨 z 耦合、曲线光路或真实相机标定。
- 每个布局启用 6 个视角，并永久排除 `Q_audit=60 deg`；合法布局共有 `C(8,6)=28` 个。
- 因此本轮只验证离散线性方向，不能外推成真实 BOST 的三维 ray tracing 结论。

### 选择顺序

1. 锁定冻结 FNO checkpoint、28 个 mask、硬支撑规则、`beta x iteration` 网格和 tie-break。
2. 只在 40 个 validation source fields 上筛选；每场先平均 4 个布局。
3. 比较 standard/Jacobi、8 个 `beta`、7 个迭代数，共 112 行。
4. selection 函数只计算 field relative L2；screen/commit 中没有 Q_audit、used-view residual 或 test 指标。
5. 写入 `v3k_c_selection_commit.json` 后，才计算 Q_audit 与四个 development test domains。

现有 test domains 已被 v3k-A/B 等前序实验查看，因此仍是 development audit，不是 fresh blind confirmation。

## 先修掉的两个错误

### 错误 1：soft support 伪装成 projection

旧实现是 `P(x)=S*ReLU(x)`，其中 `S` 是 `[0,1]` 余弦窗。它不满足幂等性，每轮都会额外收缩软边界，导致候选比 comparator 多得到几十次 oracle taper。

正式结果已改为 `P_C(x)=1[S>0]*ReLU(x)`，并加入幂等性单测。旧的 `+29.71%` 结果作废，不进入网页结论。

### 错误 2：Q_audit 虽未排序，却在 commit 前被计算

旧 screen 曾把 Q_audit 诊断值与 field selection 值一起计算。这不算进入 argmin，但也不能称“封存”。正式版本把 selection 函数缩成 field-only，commit 的 selected record 只有 split、method、`beta`、iterations 与 field statistics。

这两次修正说明：实验协议不是写在报告里的装饰，而是会实质改变能否相信结果的代码合同。

## 正式 development 结果

选中 standard Landweber：`beta=1.9`、`T=64`。Jacobi 也选中 `beta=1.9`、`T=64`。

| 域 | Feasible FNO field L2 | Standard Landweber field L2 | 场级平均收益 | 95% field-cluster CI | Q_audit 平均收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.247739 | 0.147083 | +44.6024% | [+40.1023%, +49.1099%] | +42.6682% |
| geometry-held-out IID | 0.238686 | 0.140174 | +44.5911% | [+40.1206%, +48.8455%] | +42.7708% |
| noise OOD | 0.247791 | 0.184623 | +25.9735% | [+23.3531%, +28.6485%] | +31.5899% |
| family OOD | 0.590727 | 0.453605 | +23.1896% | [+22.4064%, +23.9862%] | +23.6925% |
| joint OOD | 0.595574 | 0.462064 | +22.4001% | [+21.7531%, +23.0546%] | +23.0994% |

场级统计先在同一个 source field 的四个布局内平均，再以 source field 为独立单位。validation 上 40/40 场为正；四个 development test domains 在布局折叠后也都是 32/32 场为正。

### 免费投影与 Landweber 必须分开

把 raw FNO 单独投影到非负硬支撑域，在 validation 上已有 `+4.4745% [3.8158%, 5.1883%]` 的场级收益。因此以后所有学习方法必须同时报告：

1. locked FNO raw；
2. FNO + 相同可行域投影；
3. tuned numerical refinement；
4. learned candidate。

不这样分组，就会把一个免费的后处理误写成网络创新。

### Jacobi 没形成稳定优势

standard 相对 Jacobi 在 validation 只领先 `+0.2446% [0.0396%, 0.4476%]`；在四个 test domains 上符号不稳定，Jacobi 的绝对误差有时反而略低。当前证据不支持继续做 learned Jacobi/voxel preconditioner。

### 最坏布局仍有尾部伤害

20 个 split-layout cells 中，平均收益最差的是 noise OOD 的 `g_011011101`：

- 32 个独立 source fields；
- 平均 field gain `+19.4737%`；
- positive fraction `90.625%`；
- `>1%` harm rate `9.375%`；
- 最坏单场 gain `-17.5799%`。

所以不能写“每个样本都改善”。下一模型必须降低这个尾部风险，而不仅是让全局均值再好看一点。

## 这轮究竟证明了什么

### 可以说

- 28 个布局的离散 adjoint identity 最大相对误差为 `1.94e-15`。
- 中心有限差分梯度检查最大相对误差为 `2.44e-8`。
- 在当前 synthetic depth-separable linear forward、冻结 FNO 起点和已知硬支撑下，几何归一化伴随迭代同时改善场误差、used-view 数据一致性和未用于选择的 Q_audit。
- 此后任何 learned correction 都必须击败 tuned projected Landweber，而不是只击败 raw FNO。

### 不能说

- 不能说伴随残差保证真实三维场误差下降；它只保证在合适步长下沿声明数据项的下降方向移动。
- 不能说提出了 Landweber、Jacobi、unrolling、learned step 或 adjoint-residual 新方法。
- 不能说结果适用于曲线光路/非线性 BOST。真实情形应写 `J_i(x)^T W_i[y_i-F_i(x)]`，其中 `J_i` 是当前状态的 Jacobian/VJP。
- 不能说现有 Q_audit 是完全独立真实相机证据；synthetic noise scale 仍由全视角 clean RMS 生成，存在弱生成耦合。
- 不能修补冻结数据的坐标通道后继续沿用旧 checkpoint。v3i 继承了 x/z 通道命名对调，必须在下一套数据合同中重生成并重训。

## 下一模型：先学条件化步长，不学大网络

工作标签可暂叫 `Residual-Conditioned Spectral Step`，只用于组织实验，不代表原创性成立。

### 候选形式

`beta_t = 2*sigmoid(h_theta(q_t))`

`x_(t+1) = P_C[x_t + beta_t/L_m * A^T M_m(y-Ax_t)]`

`q_t` 只能使用部署时可得量，例如：

- normalized residual norm 与其下降率；
- active camera count、mask gap、`L_m` 与 condition proxy；
- displacement noise/confidence summary；
- iteration index 与 correction/field norm ratio。

禁止把 ground truth field error、Q_audit 或 test-domain label 放进 `q_t`。

### 先补齐四个强 control

1. **Global absolute step**：所有布局共享 `alpha=c/L_global`，其中 `L_global=max_m L_m`。
2. **Geometry-normalized step**：本轮 `beta/L_m`。
3. **Closed-form line search**：沿 `p=A^TMr` 用部署可得的数据项选择步长。
4. **Lookup control**：只按 K/noise bin/mask condition 查表，不训练神经网络。

若条件化 scalar 不能击败这四个对手，就停止 learned step。一个全局 learned scalar 与 validation 选择常数没有本质区别，不算学习创新。

### 公平比较方式

- 若候选只做 1 次 `A/A^T`，只能与 1-step controls 比；不能拿一次 learned update 对比本轮 64-step 的精度后暗示更优。
- 论文级比较同时画 error vs forward/adjoint calls、wall time 和 peak memory。
- 三个以上训练 seeds；先按 source field 折叠布局，再做 paired field bootstrap。
- 必须报告 worst-layout、p10、`>1%` harm rate 和 Q_audit，不只报告 mean。
- 开一套新生成并封存的 fields/layouts；当前 development domains 只用于建模与排错。

### 继续/停止树

1. Global step、line search、ridge-start control 未实现：不训练 scalar。
2. Scalar 未在 fresh lock 上击败所有强 controls：停止 learned step，保留数值基线成果。
3. Scalar 通过但 voxel gate 未击败 scalar、静态 voxel lookup 与同参数 local CNN：停止 voxel claim。
4. 只有真实数据确实有跨视角 confidence/标定差异时，才做 per-view reliability set。
5. Set 模型未击败 inverse-variance weighting、mask-only、shuffled calibration 与 matched VIDON/FNO：停止 set aggregation。

## 四周保姆式学习与执行

### 第 1 周：会算、会验

- 手推 `1/2||Ax-y||^2` 的梯度，解释 residual、gradient、negative gradient 三个符号。
- 用 `4x6` 随机矩阵数值验证 `<Ax,z>=<x,A^Tz>`。
- 用中心差分与 PyTorch autograd 双重检查梯度。
- 做 `P(Px)=Px`，并亲手构造 soft-window 反例。

交付：一页公式笔记、一个不依赖仓库的 `toy_adjoint.py`、三张误差曲线。

### 第 2 周：完整复现 v3k-C

- 运行单测、runner、独立 validator 和 checksum。
- 从 `v3k_c_validation_screen.csv` 画 `beta x T` 曲线。
- 从 sample metrics 复算 source-field collapse 与 worst-layout。
- 逐项解释为什么 Q_audit 没进入 selection artifact。

交付：五分钟口头讲解 + 一页复现实验记录；不能只截图 PASS。

### 第 3 周：补强数值对手

- 实现 global absolute step；比较其与 `beta/L_m`。
- 实现 steepest-descent closed-form line search。
- 从 ridge 与 frozen FNO 两个起点运行，分开 warm-start 收益与迭代器收益。
- 画 error vs `A/A^T` calls；先不训练任何网络。

交付：强基线表、失败样本切片、计算成本图和冻结配置。

### 第 4 周：有界 scalar prototype

- 先做查表 control，再做小 MLP `h_theta(q)`；输出用 sigmoid 保证 `0<beta<2`。
- 固定 correction direction，只学习步长，避免同时改变方向和幅度导致不可归因。
- 用 validation 选 checkpoint；development OOD 只验收一次。
- 若未过 mean/CI/tail/Q 四道门，写负结果并停止。

交付：给何远哲审核的 3 页 brief，不先写“新算法”标题。

## 精读材料：每篇只抽取一个角色

1. [Landweber 1951](https://doi.org/10.2307/2372313)：看经典迭代形式与步长约束；不要把它包装成项目贡献。
2. [Computed Tomography: Algorithms, Insight, and Just Enough Theory](https://doi.org/10.1137/1.9781611976670)：读 algebraic iterative reconstruction 与 regularization 章节，补 SIRT/CGLS/停止规则。
3. [Solving ill-posed inverse problems using iterative deep neural networks](https://arxiv.org/abs/1704.04058)：抽取“forward/data gradient 进入 learned update”的接口，确认这一思想已有先例。
4. [Learned Primal-Dual Reconstruction](https://arxiv.org/abs/1707.06474)：只追每轮 forward/adjoint 调用与 primal/dual state；理解为什么它远大于本科学期的一步 scalar。
5. [MoDL](https://arxiv.org/abs/1712.02862)：看数据一致性模块、共享权重与 conjugate-gradient control；它提醒我们 learned model 必须和强数值 solver 比。
6. [On Learned Operator Correction in Inverse Problems](https://doi.org/10.1137/20M1338460)：看 forward/adjoint correction 与错误物理模型的边界，为真实 BOST 标定误差预留接口。
7. [The Troublesome Kernel](https://doi.org/10.1137/23M1568739)：理解少视角 inverse 的 hallucination、稳定性和 kernel 风险；对应本轮必须报告的尾部伤害。

来源复核日期：2026-07-15。以上优先链接原论文、出版社或作者公开预印本，不上传受限全文。

## 需要何远哲明确回答的 9 个问题

1. 组内真实 BOST forward 是直线投影、曲线 ray tracing，还是可微数值模拟？
2. 是否已有与 forward 匹配的 adjoint/VJP，能否做 inner-product/finite-difference check？
3. 每个视角是否有 displacement confidence、坏点 mask、标定协方差或同步质量？
4. 真实 support/ROI 在推理时已知吗？是硬 mask、软先验，还是完全未知？
5. 真实目标是折射率、密度、密度梯度还是归一化标量？单位与坐标方向是什么？
6. 能否按独立 experiment case/geometry 分组，另留一套从未查看的 blind cases？
7. 哪个指标最接近组内科学任务：3D GT、held-out camera、质量/积分量、边界位置还是时序一致性？
8. 允许公开哪些 synthetic 代码、配置和派生指标？真实数据只做私有 case study 是否可接受？
9. 师兄更希望本科成果是“可靠数值基线 + 失败边界”，还是必须包含一个小型 learned component？

## 可复现资产

- 配置：`demo_t16_operator/configs/v3k_c_adjoint_landweber_gate.json`
- 数值核心：`demo_t16_operator/adjoint_landweber.py`
- runner：`demo_t16_operator/run_v3k_c_adjoint_landweber_gate.py`
- 独立 validator：`demo_t16_operator/validate_v3k_c_adjoint_landweber_results.py`
- 10 项工程测试：`demo_t16_operator/test_v3k_c_adjoint_landweber.py`
- 公开结果：`demo_t16_operator/results/v3k_c_adjoint_landweber_gate/`
- 私有 dataset/checkpoint：仍由 `.gitignore` 排除，不进入 GitHub Pages。

独立 validator 复算 28 个布局检查、112 个 validation grid rows、672 个 field-layout pairs、2,688 个 method-sample rows、20 个 split summaries、20 个 pairwise summaries、20 个 layout summaries和 11 个公开资产 checksum；冻结 FNO checkpoint drift 为 0。superiority、confirmatory scalar、voxel preconditioner、per-view set 与 blind final 全部保持关闭。
