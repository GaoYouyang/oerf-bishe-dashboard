# N1.6 伴随加权低秩校正：opened-development NO-GO

- 日期：2026-07-18
- 状态：`POSTOPEN_NO_GO_NO_CONFIRMATION_ROUTE`
- 证据等级：E1 synthetic opened-development 机制证据
- 预注册代码提交：`0f4ac73`
- 真实 BOST / OOD / fresh / final：均未打开

## 一句话判决

N1.6 证明了“完整 forward mismatch 仍有可利用重建 headroom”，但同时否掉了当前
`共享 measurement PCA 基 + 统计特征 ridge 系数预测`。可部署候选相对 matched low CGLS-24
的 field relative-L2 平均改善 `3.5391%`，低于预注册 `5%` 门，而且相对更简单的 component
damping 反而差 `0.1841%`。因此不运行 confirmation，也不在已经打开的 development 上重调
同一网格。

## 1. 这次到底测试了什么

设便宜的低阶离散 forward 为 `G_L`，高保真/连续 renderer 为 `G_H`，二者在同一场和几何上的
measurement mismatch 为：

```text
epsilon = G_H(x, z) - G_L(x, z)
```

N1.6 不让网络直接输出三维场，也不在部署时调用 `G_H`。算法先做 CGLS-12 暖启动，再从以下
部署可见量预测一个 measurement-space correction：

- 当前 geometry metadata；
- 实际 measured observation，而不是 clean observation；
- CGLS-12 warm field；
- `G_L(warm field)` 与 warm residual。

fit split 上先对 `epsilon - component_damping` 做 measurement PCA；再拟合小型标准化 ridge，
预测使当前低阶伴随残差最小的低秩系数。最终 correction 仍经过当前几何的 `A^T` 进入求解，
没有学习一个与 forward 脱节的独立伴随。

## 2. 预算和泄漏合同

| 路线 | warm | 可见投影 / basis probe | refine | 总低阶调用 | 高阶调用 |
|---|---:|---:|---:|---:|---:|
| matched low CGLS-24 | 24 | final projection 1F | 0 | 25F / 24A^T | 0 |
| N1.6 deployable | 12 | 1F | 12 | 25F / 24A^T | 0 |
| high-order teacher control | 12 | 1F | 12 | 25F / 24A^T | 1 high-order F |

部署行没有 `truth_volume`、fresh exact mismatch、family label、confirmation target 或 high-order
output。exact mismatch、伴随残差和 truth 指标只存在于独立 evaluator 行。两种 phantom family
共用一个 geometry，因此独立统计单位始终是 `base_seed_geometry_cluster`，不是 12 个独立 rigs。

## 3. 一次性打开的结果

calibration 从冻结网格中选择：

```text
adjlr_camera_r4_tl21em02_a1e00_s1p0
feature set = camera
rank = 4
adjoint target L2 = 0.01
ridge alpha = 1.0
residual shrinkage = 1.0
```

opened development 含 6 个 geometry clusters、每个 geometry 两个 paired field families，共 12
个 case。

| 指标 | N1.6 fail-closed | 冻结门 | 判决 |
|---|---:|---:|---|
| field gain vs matched low CGLS-24 | `+3.5391%` | `>= +5%` | fail |
| H1 gain vs matched low | `+8.2421%` | `>= +3%` | pass |
| field gain vs component damping | `-0.1841%` | `>= +0.5%` | fail |
| field gain vs high-order teacher beta=.75 | `-1.3976%` | `>= 0%` | fail |
| worst case field gain vs low | `+0.1667%` | `>= -1%` | pass |
| worst geometry gain vs low | `+0.7897%` | 描述项 | positive |
| >1% harm vs low / damping | `0 / 0` | 都必须为 0 | pass |
| takeover / fallback | `50% / 50%` | takeover `>=75%` | fail |
| evaluator adjoint-residual gain vs damping | `+1.1276%` | `>= +10%` | fail |
| deployment high-order F / A^T | `0 / 0` | `0 / 0` | pass |

结论不是“差一点通过”。11 项 future-confirmation checks 中有 5 项失败，而且它没有超过最简单
的阻尼基线。

## 4. Oracle 分解：失败发生在哪里

| 路线 | field gain vs low | 额外 gain vs damping | evaluator adjoint-residual gain vs damping | 尾部 / 角色 |
|---|---:|---:|---:|---|
| component damping | `+3.7201%` | `0` | `0` | deployable strong baseline |
| damping + fit-basis mean | `+3.9294%` | `+0.2071%` | `+10.1082%` | deployable control |
| selected fail-closed predictor | `+3.5391%` | `-0.1841%` | `+1.1276%` | takeover 50% |
| raw predictor diagnostic | `+3.4590%` | `-0.2219%` | `-25.3569%` | 无边界时预测方向错误 |
| adjoint rank-4 oracle | `+4.9852%` | `+1.4402%` | `+48.3092%` | worst vs damping `-1.1793%` |
| measurement-L2 rank-4 oracle | `+5.2454%` | `+1.7106%` | `+47.2958%` | evaluator only |
| high-order teacher beta=.75 | `+4.7980%` | `+1.1734%` | `+17.7682%` | 1 high-order F |
| exact mismatch oracle | `+8.6164%` | `+5.3059%` | `+100%` | worst vs low `+2.2031%` |

这张表给出两个独立瓶颈：

1. **表示瓶颈。** exact mismatch oracle 有 `8.62%`，共享 rank-4 adjoint oracle 只剩
   `4.99%`，并且仍伤害一个相对 damping 的 case。固定全局 PCA 基不能随几何旋转。
2. **预测瓶颈。** 即使 rank-4 oracle 有小 headroom，raw ridge 的伴随残差比 damping 更差
   `25.36%`；fail-closed 只是阻止伤害，没有制造优势。

fit split 上 rank 1/2/4/8 的平均 adjoint-target residual ratio 约为
`0.793/0.705/0.450/0.297`，说明增加 rank 能拟合训练 mismatch；但 calibration 上更高 rank
并没有给出更好的稳定 field 决策。这不是“把 rank 调到更大”就能解决的问题。

## 5. 为什么这仍是有价值的负结果

- 它把“measurement mismatch 好看”与“有限步三维重建真正变好”分开。
- 它证明 exact mismatch 的有效 headroom 明显高于 damping，不应放弃 model correction 主线。
- 它证明一个跨几何共享的静态低秩基不够；下一候选必须由当前 `A/A^T` 生成几何条件化表示。
- 它证明 fail-closed 路径确实在工作：raw predictor 的错误方向没有被包装成部署成功。

## 6. 下一假设：几何条件化的有限步 Krylov correction

暂名 **N1.7 KCRC（Krylov-Conditioned Residual Correction）**。这只是待预注册假设，不是已验证
创新。核心变化不是换更大的 MLP，而是换表示空间和训练目标。

### 6.1 表示

从当前 measured residual、component damping 和低阶 normal operator 生成每个 geometry 自己的
measurement basis，例如：

```text
r0 = y - A x_warm
b0 = component_damping
b1 = r0
b2 = A A^T b0
b3 = A A^T r0
delta_y = b0 + sum_j c_j * normalized(b_j)
```

两次 `AA^T` probe 各需 1F+1A^T；把 refine 从 12 步降到 10 步后，候选仍恰好是
`25F/24A^T`，与 matched CGLS-24 等预算。输出仍在 measurement space，随后统一经过当前 `A^T`，
所以不学习脱离几何的独立 adjoint。

### 6.2 先过 representation gate，再训练网络

1. 只用 fit/calibration 测试 per-geometry Krylov span 的 adjoint-optimal oracle。
2. 若 oracle 不能同时超过 damping、高阶教师和逐 geometry tail 门，立即停止，不训练 learner。
3. oracle 过门后，小 hypernetwork 才读取 basis Gram matrix、相机/射线几何、warm residual 统计，
   输出有界系数 `c_j`。
4. 训练损失直接穿过固定 10 步 CGLS refine，优化 finite-iteration field/H1 response；
   measurement L2 只作辅助项。
5. calibration envelope、ensemble disagreement 或独立 camera evidence 不过门时回退 damping。

### 6.3 这条路线对应的一级来源

- Lunz et al., *On Learned Operator Correction in Inverse Problems*：明确指出 forward-only correction
  的困难，并给出 forward-adjoint correction 框架。官方开放页：
  <https://epubs.siam.org/doi/10.1137/20M1338460>
- Arridge, Hauptmann & Korolev, *Inverse Problems with Learned Forward Operators*：综述 learned
  forward restriction 与 model correction，并强调 adjoint 信息的重要性：
  <https://arxiv.org/abs/2311.12528>
- He et al., *Neural refractive index field*：用连续隐式场和随机 ray sampling 降低 voxel
  离散误差，并用 held-out view reprojection 验证真实火焰：
  <https://arxiv.org/html/2409.14722v2>
- Molnar et al., *Forward and Inverse Modeling of Depth-of-Field Effects in BOS*：有限孔径 cone-ray
  模型在 f/22 到 f/4 下显著提高 shock reconstruction 稳健性，给出真实物理 mismatch 靶点：
  <https://doi.org/10.2514/1.J064095>
- Cai et al., *Direct BOST using radial basis functions*：直接重建折射率并用未参与重建的相机
  做 reprojection；也明确报告 projection-matrix 构造的计算成本：
  <https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-11-19100>

这些文献支持“为什么研究 model mismatch / adjoint / geometry-conditioned representation”，不证明
KCRC 已有新颖性。正式 novelty claim 还需更完整检索和真实有限孔径/弯曲光线实验。

## 7. 现在问何远哲师兄的六个关键问题

1. 当前 BOST/NeRIF forward 能否分别暴露 matrix-free `A(v)` 和严格配对 `A^T(u)`？
2. 真实系统的 f-number、焦面、焦距、背景距离和 aperture sampling 是否可得？
3. 师兄认为主导 mismatch 更可能来自 finite aperture、ray bending、camera calibration，还是
   optical-flow bias？请按优先级排序。
4. 是否能生成同一场/同一几何的 `G_L/G_H` paired projections，且记录每次 high-fidelity query？
5. 是否能永久留出一台相机、一个 session 和一个 f-number，只用于最终物理验证？
6. 能否提供一个最小真实 packet：原始 reference/distorted images、相机参数、mask/confidence、
   reconstructed field 和当前代码的调用入口？

## 8. 复现与完整性

```bash
cd oerf-bishe-dashboard

.venv/bin/python site_tools/run_jacru_n1_6_adjoint_low_rank.py \
  --config demo_t16_operator/configs/jacru_n1_6_adjoint_low_rank_development_v1.json \
  --output-dir demo_t16_operator/results/jacru_n1_6_adjoint_low_rank_development_full1

cd demo_t16_operator/results/jacru_n1_6_adjoint_low_rank_development_full1
shasum -a 256 -c checksums.sha256
```

本次 SHA-256 检查 15 个被登记文件全部 `OK`。结果包含机器摘要、逐 case 指标、单独的 evaluator
adjoint diagnostics、fit targets、manifest、配置快照、provenance、PNG/PDF 图和 checksums。

## 9. 允许和禁止的表述

**允许：**

- “在当前 opened synthetic JACRU surrogate 上，完整 mismatch oracle 显示约 8.62% field
  headroom。”
- “共享低秩表示和当前 ridge predictor 均未通过门；下一步测试 geometry-conditioned Krylov
  representation。”
- “可部署候选不调用高阶 forward/adjoint，并与 CGLS-24 匹配 25F/24A^T。”

**禁止：**

- “N1.6 已优于 FNO、DeepONet、NeRIF 或 TDBOST。”
- “3.54% synthetic mean gain 是真实 BOST 或泛化成功。”
- “exact mismatch / adjoint oracle 是可部署算法。”
- “fail-closed 无伤害等于算法可靠”，因为它有 50% case 直接回退，而且尚无真实/OOD 证据。
