# T16 算子学习 x BOST 三维重建实验规格

更新日期：2026-07-11

## 0. 当前实测 checkpoint

`demo_t16_operator/` 已完成第一版可运行 linear-synthetic closure test，不再只是实验提案：

- 168 个 `8 x 16 x 16` synthetic volumes；64 train、16 val、88 个分层测试样本；
- Gaussian/flame 训练族与完整 thin-front family OOD；
- physics lift、3D U-Net、官方 NeuralOperator 2.0 residual 3D FNO；
- 一次 train-only global affine calibration，不做 per-sample truth alignment；
- field、gradient、observed/held-out reprojection、mass、centroid 和运行成本；
- 30 epoch CPU 运行，独立复跑的 sample-level CSV 字节级一致。

IID field relative L2：physics lift `0.4573`、U-Net `0.2706`、FNO `0.2321`。FNO 在五个 split 的 field mean 均低于 U-Net，但三视角 OOD 的 held-out reprojection 只从 U-Net `0.5466` 降到 FNO `0.5420`，配对差异 `-0.0046 +/- 0.0218` 的 95% 区间跨零；joint OOD 也没有稳定 held-out 优势。thin-front family OOD 的 FNO field error 仍为 `0.6789`。

因此下一阶段不是继续把 toy leaderboard 做大，而是验证：

1. FNO 的改善是否来自 physics lift、spectral architecture 或容量/优化差异；
2. 如何消除少视角下 field 与 held-out physics 的分叉；
3. 独立高分辨率生成器上是否真的有 resolution transfer；
4. 能否接入组内 geometry/data，或作为 NeRIF initialization。

详细证据见 `t16_operator_smoke_review_brief.md`，代码说明见 `demo_t16_operator/README.md`。

2026-07-11 v2d 更新：已锁定 Q_fit=80°、Q_audit=60°，完成同重建预算 direct/correction/numerical audit。fixed learned correction 相对 `S∪Q direct` 在 K=4/6/8 为 -20.17%/-13.07%/-15.11%，当前 checkpoint 路径 0/3 通过并停止扩模型。因训练 mask 不匹配、88 fields 已参与开发且 forward 仍为 linear synthetic stack，状态为 `PILOT_ONLY_CURRENT_CHECKPOINT_PATH_FAILS`。详见 `operator_fair_budget_dashboard.html`。

## 1. 研究问题

目标不是为某一个观测样本重新优化一套网络参数，而是学习一个可跨样本复用的逆算子：

```text
G_theta: multi-view BOS displacement + geometry + mask -> 3D refractive-index field
```

对新观测，模型应一次前向推理得到三维场；BOST forward model 再把预测场投影回观测空间做物理验收。

这和 NeRIF 的关系是互补而非替代：NeRIF-style coordinate field 主要针对单个场做 per-instance optimization；T16 学习跨 phantom、噪声、视角和分辨率可复用的 amortized inverse。后续可以让算子输出成为 NeRIF 初值，再用重投影损失细化。

## 2. 推荐的本科主模型

### 2.1 Adjoint-lift residual FNO

先把不规则的多视角观测通过已知几何提升到规则三维网格，再使用 3D FNO：

```text
d, geometry, mask
  -> z0 = A_g^* d or a fixed-budget SIRT reconstruction
  -> input channels = [z0, sampling-density, valid-mask, noise/view metadata]
  -> delta_n = FNO_theta(input channels)
  -> n_hat = z0 + delta_n
  -> d_hat = A_g(n_hat)
```

推荐理由：

- FNO 擅长规则网格上的场到场映射；反投影先解决多视角观测和三维网格之间的形状不匹配。
- residual learning 让模型只学习传统重建的系统性残差，降低从零生成三维场的难度。
- forward operator 保留在网络外，便于做 held-out view 和重投影验证。
- 现有 M1/M2 BOST synthetic pipeline 可以提供 `A_g`、phantom、噪声和指标。

### 2.2 必做对照

1. 固定预算的 Landweber/SIRT 或现有 M1 baseline。
2. 3D U-Net/CNN：检验收益来自“算子设计”还是普通卷积容量。
3. FNO without reprojection loss：检验 BOST 物理约束的贡献。
4. FNO direct field prediction vs residual correction：检验 physics lift 的价值。
5. 小规模 NeRIF-style per-instance optimization：比较一次推理速度与单样本优化精度。
6. 仅当进入 4D 扩展：framewise reconstruction、M3B fixed-rank temporal SVD、held-out/奇异谱驱动的无真值 rank selector。M3B temporal rank 不等同 FNO Fourier modes 或网络宽度。

## 3. 第二模型：DeepONet

DeepONet 更接近严格的 measurement-to-query operator：

- branch net 编码多视角位移场、相机几何和 mask。
- trunk net 输入查询坐标 `(x, y, z)`。
- branch/trunk 特征内积输出 `n(x, y, z)`。

它天然支持任意查询坐标和连续输出，但标准 branch net 往往要求固定数量、固定顺序的传感器。若真实 BOST 有可变视角或缺失视角，需要增加 view mask、DeepSets/attention encoder，或先固定 3/5/7/9-view 子集。DeepONet 适合作为第二架构，不建议第一周就同时实现全部可变视角机制。

## 4. 损失函数

建议从四项开始：

```text
L = lambda_field * L_field
  + lambda_grad  * L_grad
  + lambda_proj  * L_reprojection
  + lambda_bc    * L_boundary
```

- `L_field`: synthetic truth 上的 global relative L2 或 H1/Lp loss。
- `L_grad`: 折射率梯度误差，避免只拟合低频平均场。
- `L_reprojection`: `A_g(n_hat)` 与输入 displacement 的误差；必须同时报告训练视角与 held-out 视角。
- `L_boundary`: VOI 边界折射率或扰动归零条件，约束常数项和漂移。

第一版不建议直接加入 Navier-Stokes PDE loss。BOST forward projection 本身就是最贴题、最可验证的物理约束；只有获得速度、压力、边界条件或 CFD 数据后，才考虑 PINO/PINN 型流体方程约束。

## 5. 数据集设计

### 5.1 样本内容

每个样本至少包含：

```text
field_gt:        [D, H, W]
deflection:      [V, 2, H_img, W_img] or equivalent projection stack
geometry:        view angles / ray origins / ray directions / scale
valid_mask:      [V, H_img, W_img]
lifted_volume:   [C, D, H, W]
metadata:        phantom family, noise, view count, resolution, seed
```

如果进入 temporal operator，再增加 `dynamics family / frame rate / exposure / event label`；静态 T16 不为了凑 4D 而伪造时间标签。

### 5.2 Phantom family

- smooth Gaussian / ellipsoid mixtures：用于 sanity check。
- flame-like elongated structures：更贴近折射率场。
- multiscale blobs / thin fronts：检验高频和空间分辨率。
- transient/dynamic fields：只在静态算子跑通后进入 4D 扩展。

### 5.3 划分原则

不能只随机打散同一生成器参数。至少做四种测试：

1. unseen random seeds；
2. unseen phantom parameters；
3. unseen phantom family；
4. unseen noise/view/resolution condition。

如果训练和测试只是同一 phantom family 的邻近参数，不能声称跨工况泛化。

### 5.4 M3B-informed condition split

M3B 的 80 环境交叉实验不是 T16 的训练结果，但它已经说明 view/noise 联动足以改变先验的收益符号。T16 第一版建议把工况格显式写入 manifest：

表中的 `0.07/0.14/0.28/0.42` 是 M3B 合成位移噪声乘子，不是像素、偏折角或 OERF 实验噪声单位。接真实数据前必须用 background-displacement residual、重复测量或组内噪声模型重新标定，不能照搬数值。

| Split | View / noise 条件 | 回答的问题 |
| --- | --- | --- |
| IID | 5/7 views；noise 0.07/0.14/0.28；phantom 与 seed 不重叠 | 同一工作域内是否真正学到跨样本 inverse operator |
| View OOD | 整格保留 3 views 与 9 views | 严重欠定与较充分视角能否迁移 |
| Noise OOD | 整格保留 noise 0 与 0.42 | 无噪极限和高噪极限是否失效 |
| Joint OOD | 3/9 views × noise 0/0.42 | view 与 noise 同时偏移时是否仍可信 |
| Family OOD | 完整保留一种 thin-front/flame-like family | 避免只对同一生成器参数插值 |

这是一份启动 split，不是唯一正确划分。至少再做一次条件对调 sanity check，确认结论不是某一格被指定为 test 的偶然结果。不能把同一 phantom 的相邻参数或同一时序的相邻帧分到 train/test 两边。

M3B 的直接启示应写成可证伪目标：

1. 不只报告全局平均收益，必须报告每个 view×noise cell 的 gain 与失败格数量；
2. field L2、held-out reprojection 和 mass/centroid 等积分量必须并列，不能用其中一个替代全部；
3. 真实数据没有 field truth 时，模型选择只能依赖 held-out view、forward residual、边界和可获得的物理 trace；
4. temporal operator 只能在静态 T16 跑稳后进入，并与 M3B fixed-rank SVD 公平比较。

## 6. 三阶段规模

以下数字是启动预算，不是论文结论：

| 阶段 | 输出规模 | 建议样本量 | 目标 |
| --- | --- | --- | --- |
| O0 | 2D 64x64 | 500-2,000 | 跑通 DeepONet/FNO、数据 loader 和 baseline |
| O1 | 3D 24^3 或 32^3 | 1,000-5,000 | adjoint-lift residual FNO 主实验 |
| O2 | 3D 48^3/64^3 或真实样例 | 视显存与数据决定 | 分辨率迁移、真实数据或 NeRIF refinement |

3D 数据量、显存和 I/O 会迅速增长。先做小体积并记录显存/吞吐，再决定是否提高分辨率；不要先承诺 200^3。

## 7. 评价矩阵

### 场空间

- global relative L2 与 paper-style squared L2；两者分开命名。
- CC、SSIM 或 slice-SSIM。
- gradient L2/H1 error。
- centroid、mass/integral trace 或等价物理代理量。

### 观测空间

- train-view reprojection error。
- held-out-view reprojection error。
- 不同噪声、视角、mask 和 geometry perturbation 下的误差。
- view×noise condition cell 的平均、置信区间、正/负收益格和最坏格；不能只给 pooled mean。

### 算子属性

- unseen phantom family generalization。
- resolution transfer：训练 24^3，测试 32^3/48^3；只能报告实测结果，不能因为 FNO 论文有 zero-shot super-resolution 就默认成立。
- inference time、训练成本、参数量、峰值显存。
- calibration curve 或 ensemble uncertainty，可作为后续增强。

## 8. 12 周执行

1. 第 1 周：读 DeepONet、FNO 和 JMLR neural operator 综述；跑官方 1D/2D tutorial。
2. 第 2 周：用 2D Radon/BOS toy 构造 500 个样本；跑 CNN、FNO、DeepONet 最小对照。
3. 第 3-4 周：把 M1 的 forward/backprojection 扩成批量数据生成器；保存 manifest 和 train/val/test split。
4. 第 5 周：实现 3D U-Net/CNN baseline 和 adjoint-lift residual FNO。
5. 第 6 周：统一训练预算，画 field/reprojection/runtime 三组图。
6. 第 7 周：加入 held-out view 和 reprojection loss，做 loss ablation。
7. 第 8 周：按 5.4 的 condition split 做 noise/view/mask 扫描，画 gain map、失败格和 held-out/物理量分叉。
8. 第 9 周：做 unseen phantom family 和 resolution transfer。
9. 第 10 周：实现最小 DeepONet，比较固定视角输入与 FNO lift。
10. 第 11 周：若有组内样例，做 zero/few-shot transfer；否则补系统偏差 failure signature。
11. 第 12 周：整理 8 张论文图、模型边界和何远哲问题清单。

## 9. 预期论文图

1. BOST measurement -> adjoint lift -> residual FNO -> forward validation pipeline。
2. GT / SIRT / U-Net / FNO / FNO+projection loss 三维切片。
3. view count × noise 分层 gain heatmap、失败格计数与最坏格切片。
4. seen vs unseen phantom family 泛化图。
5. train/test resolution transfer 表。
6. field error vs held-out reprojection scatter，并标出 mass/centroid 变差象限。
7. loss/model ablation。
8. inference time / peak memory / accuracy Pareto 图。

## 10. 停止条件与降级方案

- 若 3D FNO 显存超预算：回到 2D/3D-stack，或使用 tensorized FNO/patch training。
- 若 synthetic dataset 生成太慢：预计算 geometry operator，降低体素和视角数，先保证 500-1,000 个样本。
- 若 FNO 不优于 3D U-Net：把研究问题改为“operator inductive bias 与分辨率迁移是否成立”，保留负结果。
- 若 reprojection loss 降低但 field error 变差：作为病态逆问题和 non-uniqueness 结果，不继续只调 loss 权重。
- 若真实数据域差太大：不声称真实泛化，转为 synthetic-to-real failure audit 或 NeRIF initialization 工具。

## 11. 找何远哲确认

1. 师兄说的“算子学习”具体是 projection-to-volume inverse operator，还是 3D/4D flow evolution operator？
2. 目标物理量是 refractive index、density、temperature、chemiluminescence，还是 velocity correction？
3. 输入应是 raw BOS image、displacement field、sinogram/projection，还是传统重建初值？
4. 组内是否已有批量 paired data，样本量、视角数、分辨率和数据权限如何？
5. 更希望 DeepONet、FNO、operator transformer，还是只要一个可靠的 operator baseline？
6. 真实验收最看 field error、test-view PSNR/reprojection、速度、显存还是跨工况泛化？
7. 是否接受 adjoint/SIRT lift + residual FNO 作为本科主模型？
8. 算子输出是否应作为 NeRIF/TDBOST 初始化，而不是直接替代现有重建？
9. M3B 已显示 field 与 mass trace 会分叉；真实 BOST 中必须保真的积分量、频率或事件量是什么？
10. 是否接受把 3/9 views 与 noise 0/0.42 整格留作 condition OOD，还是组内有更真实的视角/噪声分布？

## 12. 真实建议

“算子学习相对容易”成立的前提是把任务收缩为模型设计、合成 paired dataset 和清晰 benchmark。网络本身可以较快搭起来，真正困难的是训练分布、真实数据偏移和物理验收。最稳的题目不是泛泛地“用 FNO 做三维重建”，而是：

> 面向少视角 BOST 的物理提升神经算子：比较 CNN、Residual/Absolute/可靠度门控 FNO 在跨视角、跨噪声和跨分辨率三维折射率重建中的泛化边界。

M3B 使这个题目多了一条明确底线：所谓“泛化边界”必须落实为预声明的 condition cells、失败格和多物理指标，而不是一张总体平均误差表。

## 13. 已运行三种子因果消融与规格修订

固定 `t16_smoke_v1` 的 168 个 volumes，只改变三个优化种子：

| 模型 | 参数 | IID field mean | IID held-out mean |
| --- | ---: | ---: | ---: |
| Residual FNO | 43,363 | 0.2187 | 0.2507 |
| Absolute-output FNO | 43,363 | 0.2367 | 0.2677 |
| Residual FNO, no reprojection loss | 43,363 | 0.2375 | 0.2698 |
| Matched U-Net | 49,111 | 0.3000 | 0.3488 |

这轮对规格的修订不是简单把 Residual FNO 定为赢家：

1. Residual FNO 在五个测试域的 field/held-out 共 `30/30` 个 seed-domain 单元优于 matched U-Net，支持保留 spectral architecture；
2. 带 reprojection loss 的 field mean 在五域都更低，但只有三个种子，Student-t 区间仍宽；
3. Residual 在 IID/noise OOD 的三个种子全部更好，Absolute 在 3-view/joint OOD 的三个种子全部更好；
4. Absolute FNO 仍接收 physics lift channel，只消融 skip，不代表 raw measurement direct operator；
5. family OOD 仍约 `0.67`，架构小改动不能替代训练分布与真实几何。

因此下一版模型规格改为：

```text
z0 = physics_lift(d, g)
q  = observable_reliability(view_coverage, geometry, residual_statistics)
alpha = gate(q)
n_hat = alpha * z0 + correction_head(x)
```

最低对照应包括 fixed gate、learned gate、Residual FNO、Absolute FNO 与 matched U-Net。gate 不能使用 field truth 或 test label；真实部署只允许 view geometry、位移残差、标定质量、train/held-out camera consistency 等可观测输入。

## 14. v2a-v2c 结果对实验规格的再次修订

### 已完成的因果链

1. 独立 Residual/Absolute experts 提高分支差异并给出更强 support-fit，但参数约翻倍；
2. truth-oracle null projection 证明独立 support-fit 仍有 38.626% 平均结构性 headroom；
3. matched free/null learned corrector 显示 hard constraint 保持 support consistency，却没有稳定学到修正方向；
4. adaptive query 的闭式 alpha 把 full learned null 的 -0.126% 总体结果转为 one-query +0.746%、all-query +0.956%；
5. 15 个 seed-domain 汇总均值为正，但底层是 88 个独立测试体场 × 3 个 model seeds，不是 264 个独立物理样本；
6. field-truth 最优 alpha 对当前 learned direction 也只提高 +1.212%，方向质量而非 alpha 复杂度是当前主瓶颈；
7. family OOD 绝对 field error 仍约 0.69。

### 新的主实验因子

| 轴 | 最小水平 | 目的 |
| --- | --- | --- |
| expert capacity | shared dual / independent dual / equal-wide single / independent ensemble | 拆开结构与参数量收益 |
| correction | none / free / hard-null / query-calibrated hard-null | 验证 data consistency 与方向/幅度分工 |
| query count | 0 / 1 / 2 / all withheld | 量化额外观测价值 |
| total camera budget | S-only / S∪Q direct / S+Q correction | 检验相机分工是否超过直接加视角 |
| query split | Q_fit / Q_audit | 将选角与 alpha 拟合与最终 held-out 验收完全隔离 |
| query placement | fixed / random / largest gap / max predicted response / design-optimal | 分开可部署固定相机和逐样本 adaptive 收益 |
| query corruption | noise / calibration offset / timing offset / blur | 检查验证视角是否可靠 |
| correction direction | random / PCA / branch difference / non-learning solve / learned / truth oracle | 检验网络是否真正学到更好方向 |
| forward mismatch | matched A / independent high-resolution A / cone-ray / calibration mismatch | 消除 inverse crime |
| geometry | fixed / rotated / limited arc / clustered / jittered / calibrated hardware | 与 M3B LOGO 对齐 |
| resolution | independently generated 16 / 24 / 32 grid | 验证真正的 discretization transfer |

### 新的主图

1. support-range/null + query alpha 模型图；
2. full null 与 query-calibrated 五域 improvement 图；
3. query count/angle/noise 的 value-of-information Pareto；
4. support leakage 与 held-out improvement 散点；
5. equal-parameter architecture/capacity controls；
6. independent resolution + geometry transfer；
7. operator warm-start NeRIF 的总耗时/最终 residual/失败率；
8. 真实 BOST held-out camera 与失败案例。

### 新停止条件

- 若同总相机数下 S+Q correction 不能超过 S∪Q direct 强基线，删除相机分工主贡献。
- 若预先固定一台 query 不能保留至少 80% adaptive 收益，不将 one-query 写成可部署方法。
- 若 learned direction 的 field-oracle gain 仍低于 `max(5%, 2×真实重复性 CV)`，停止扩大 corrector。
- 若独立 Q_audit 与 field/physics improvement 不对齐，删除“query-calibrated”主贡献，回退 support-fit。
- 若 equal-wide single operator 追平 independent dual，不能写 dual expert superiority。
- 若 approximate null projection 在真实 ray geometry 下开销或 leakage 不可接受，改局部线性化/Krylov projection。
- 若 query camera 的收益不足以抵消硬件/同步成本，把它定位为 audit sensor，而非部署必需组件。
