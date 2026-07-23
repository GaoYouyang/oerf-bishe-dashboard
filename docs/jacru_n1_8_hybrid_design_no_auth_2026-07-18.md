# N1.8 相机/射线混合表示：旧开发集设计屏 NO-AUTH

日期：2026-07-18

机器状态：`NO_N1_8_CONFIRMATION_AUTHORIZATION`

证据等级：`E1_SYNTHETIC_OPENED_DEVELOPMENT_HYPOTHESIS_DESIGN_ONLY`

## 1. 一句话结论

五个部署可见、同为 `25F/24A^T` 的低秩表示都在 6 个已打开 geometry、12 个 paired fields
上完成了无 learner oracle 比较。Camera-Block-6 最接近重建门，但 incremental extra-headroom
只有 `57.071% < 60%`，且 support-adjoint gain 只有 `9.474% < 50%`。因此不授权新确认集、
不训练系数网络，也不能声称已经得到新算法。

这不是整个研究目标停止。它关闭的是“直接在这五个 basis 上训练 learner”这条具体分支。

## 2. 运行前冻结与完整性

- 代码、配置、测试和设计边界先提交为 `9f8e030d8b0b913e080f4e9fa88e6f7349a240b4`，再运行完整设计屏。
- 独立审计随后发现一个没有触发本次判决的潜在 fail-open：若未来 17 个重建门全过但 `P A^T`
  gain 为负，旧分类仍可能授权 solver-aware。修正版强制负 gain NO-GO、候选精确 rank，并在引用
  N1.7 均值前验证 12 个 development case/digest 完全相同。
- 审计修正提交为 `d1c1b24cc8eaa48c9aa4e893d875c8e629e7d271`；修正版重放的 168 条逐 case
  科学指标和五个 gate 数字与原包完全相同，状态仍为 NO-AUTH。
- 修正版结果包 provenance 指向 `d1c1b24`；11 个受校验文件全部通过 SHA-256。
- 只复用已经打开的 development；没有构造新 geometry，没有打开 fresh/OOD/final/真实 BOST。
- basis 构造不读当前 case 的 truth 或 exact mismatch；没有 learner，也没有 finite-K field search。
- 统计单位是 6 个 geometry clusters；两个 phantom family 不是额外独立样本。
- 五个候选均为 25F/24A^T、0 次 high-order 调用；Camera-Block-6 用 0F/0A^T setup 加 12 步 refine，
  其余候选用 2F/2A^T setup 加 10 步 refine。

## 3. 五候选结果

| 表示 | field vs CGLS-24 | H1 vs CGLS-24 | field vs damping | extra-headroom | exact retention | P A^T gain | 重建门 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Krylov-4 total | +5.319% | +9.695% | +1.734% | 33.628% | 62.488% | 27.025% | 14/17 |
| Fit-PCA2 + Krylov-6 | +5.635% | +9.095% | +2.090% | 40.517% | 66.202% | 38.947% | 15/17 |
| Camera-Block-6 | **+6.343%** | +13.203% | **+2.874%** | **57.071%** | **74.518%** | 9.474% | **16/17** |
| Pose-Fourier-K6 | +6.148% | **+13.369%** | +2.645% | 52.179% | 72.223% | 8.417% | 16/17 |
| Detector-Moment-K6 | +5.481% | +12.518% | +1.943% | 38.831% | 64.389% | 8.169% | 15/17 |

五个候选的逐 case field harm `>1%` 比例均为 0，最坏 geometry gain 也均为正。这个安全信号值得
保留，但不能覆盖冻结门失败。Camera-Block-6 的 exact retention 已超过 70%，仍因更严格、真正
增量的 extra-headroom 只有 57.071% 而失败。

## 4. 为什么不能把 60% 改成 57%

运行前独立审计发现旧 retention 把 component damping 已经获得的收益也算进分子。N1.8 因此先冻结

```text
R_extra = sum_i(E_damping_i - E_candidate_i)
          / sum_i(E_damping_i - E_exact_i)
```

并把门设为 60%，明确高于已经看过的 N1.7 事后 ceiling。现在把门降到 57% 会是标准的看结果调门。
更重要的是，Camera-Block-6 的 `P A^T` gain 只有 9.474%；即使放宽 extra 门，它也不能被解释为
forward-correction representation。

## 5. 这组数字真正提示了什么

1. **相机分块对有限步 field 很有用。** Camera-Block-6 不做 normal probes，仍比 Krylov-4 高
   `1.024` 个 field 百分点，说明 per-camera amplitude 是当前 surrogate 中真实存在的求解坐标。
2. **重建坐标与 forward mismatch 坐标明显错位。** Camera/Pose 表示的 field/H1 最强，
   `P A^T` gain 却只有约 8%–9%；Fit-PCA 的 adjoint 较高，却少拿了约 16.6 个百分点 extra-headroom。
3. **简单拼 rank 不够。** Fit-PCA、Camera-Block、Pose-Fourier 和 Detector-Moment 都是 rank-6，
   结果差异不能只归因于维数；但正式新 split 仍需 camera-label shuffle 和 rank-matched control。
4. **当前 synthetic 物理仍很窄。** 它只覆盖 analytic renderer 与 voxel/trilinear inverse 的差异，
   不覆盖 finite aperture、ray bending、optical-flow bias、标定漂移或真实火焰数据。

## 6. 下一步：两个 rank-6 contrast 证伪，不立即开新 split

令 `C1/C2` 为三个相机上的两个正交中心化 contrast mask，`K=A P A^T`。只冻结两个候选：

1. **Residual-Contrast Global-K6：**`orth(d,r,C1r,C2r,Kd,Kr)`。
2. **Damping-Contrast Global-K6：**`orth(d,r,C1d,C2d,Kd,Kr)`。

两者均用 2F/2A^T setup + 10 refine，保持 25F/24A^T；完整 correction radius、向量顺序、
精确 rank 6、原 17 门和 `P A^T >= 0` 下限必须在旧 development 重放前冻结。第一项检验
Pose 的 residual 局部性与全局 normal response 能否互补；第二项隔离 Camera-Block 的收益是否
主要来自按相机拆 damping。若两者都失败，关闭 rank-6 camera/global-K 分支，不继续堆网络。

### 后续才是 Geometry-conditioned response basis

只有上述 contrast 表示存在联合 headroom，才设计真正的算子学习候选：从 camera pose、detector coordinate、mask、
warm residual 与 `A/A^T` probes 生成低秩 basis；训练目标穿过冻结的有限步 CGLS，同时约束 field、
H1、`P A^T` 和 correction norm。它应被称为 response-aware solver representation，除非 50%
adjoint 机制门也通过。DeepONet/FNO 可作 coefficient/basis generator 对照，不能成为创新点本身。

## 7. 给何远哲师兄确认的五个问题

1. 真实 BOST 接口是否提供每条 ray 的 camera ID、像素坐标、pose、mask/confidence 和单位？
2. 组内最主要的 mismatch 是 finite aperture、ray bending、标定漂移、位移提取，还是离散算子？
3. 是否能永久留出一台 camera 或一个 session，只用于 held-out reprojection？
4. 真实项目更认可 field、flame-front、held-out image、PIV compensation 还是其他主终点？
5. 能否给一个 NeRIF/TDBOST 典型失败 case，以及可调用的 `A/A^T` 或 Jacobian-vector 接口？

## 8. 复现入口

- [运行前冻结](jacru_n1_8_hybrid_design_freeze_2026-07-18.md)
- [机器摘要](../demo_t16_operator/results/jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1/summary.json)
- [逐 case 指标](../demo_t16_operator/results/jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1/case_metrics.csv)
- [basis 诊断](../demo_t16_operator/results/jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1/basis_diagnostics.csv)
- [SHA-256 清单](../demo_t16_operator/results/jacru_n1_8_hybrid_design_screen_postopen_audit_amended_full1/checksums.sha256)

当前允许的论文表述是：在已打开的内部 synthetic surrogate 上，相机分块表示改善了有限步重建，
但未通过预先冻结的增量 headroom 与 solver-adjoint 联合要求，因此没有授权 learner 或确认实验。
