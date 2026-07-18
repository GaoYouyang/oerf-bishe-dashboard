# 给何远哲师兄：N1.8 相机/射线表示判决与下一步请教

日期：2026-07-18

## 想请您先判断的核心方向

我现在没有直接训练 DeepONet/FNO 去预测三维场，而是先问一个更窄的问题：在固定的
`25F/24A^T` 重建预算下，加入 camera/ray 条件的低秩 correction basis，是否能同时改善有限步
三维重建，并解释当前 forward-model mismatch 对求解器真正可见的部分？

当前仍是内部 `12^3` synthetic surrogate：连续 analytic renderer 生成观测，voxel finite-difference
+ trilinear operator 做反演。它不代表有限孔径、光线弯曲、位移提取或真实火焰已经验证。

## 已完成的五个同预算表示

| 表示 | field vs CGLS-24 | H1 | extra-headroom | P A^T gain | 判决 |
|---|---:|---:|---:|---:|---|
| Krylov-4 | +5.319% | +9.695% | 33.628% | 27.025% | 14/17 |
| Fit-PCA2 + Krylov-6 | +5.635% | +9.095% | 40.517% | 38.947% | 15/17 |
| Camera-Block-6 | **+6.343%** | +13.203% | **57.071%** | 9.474% | **16/17** |
| Pose-Fourier-K6 | +6.148% | **+13.369%** | 52.179% | 8.417% | 16/17 |
| Detector-Moment-K6 | +5.481% | +12.518% | 38.831% | 8.169% | 15/17 |

6 个 geometry、12 个 paired fields 的 field/H1 尾部均为正，五候选都没有超过 1% 的逐 case
伤害。Camera-Block-6 最接近，但运行前冻结的 incremental extra-headroom 门是 60%，实际只有
57.071%；而且 `P A^T` gain 只有 9.474%，不能解释成忠实 forward correction。

所以机器状态是 `NO_N1_8_CONFIRMATION_AUTHORIZATION`：不降门、不打开 fresh、不训练 learner。
独立审计还修复了一个未来可能把负 adjoint gain 错标 solver-aware 的 fail-open；修正版重放的
168 条科学指标逐项不变。

## 我建议下一步只证伪两个结构

令 `d` 为 component damping，`r` 为 warm residual，`C1/C2` 为三个相机的两个正交中心化
contrast mask，`K=A P A^T`。

1. **Residual-Contrast Global-K6**：`orth(d,r,C1r,C2r,Kd,Kr)`。
2. **Damping-Contrast Global-K6**：`orth(d,r,C1d,C2d,Kd,Kr)`。

两者都保持 2F/2A^T setup + 10 refine，即总 25F/24A^T。前者检验 residual 的相机局部性，
后者检验 Camera-Block 的收益是否主要来自 damping 分块。若两者都失败，我会关闭 rank-6
camera/global-K 分支，不继续堆网络；若其中存在联合 field/adjoint headroom，再研究由 camera pose、
detector coordinate、mask、residual 和 A/A^T probes 生成 basis 的 geometry-conditioned learner。

## 想请您确认的五件事

1. 真实 BOST 数据是否能给每条 ray 的 camera ID、像素坐标、pose、mask/confidence 和单位？
2. 组内目前最主要的 mismatch 更接近 finite aperture、ray bending、标定漂移、位移提取，还是离散算子？
3. 能否永久留出一台 camera 或一个 session，只用于 held-out reprojection，不参与训练和选模？
4. 组内最认可的主终点是三维 field、flame-front、held-out image、PIV 补偿，还是其他量？
5. 能否给一个 NeRIF/TDBOST 的典型失败 case，以及可调用的 `A/A^T` 或 Jacobian-vector 接口？

完整机器证据、口径和复现命令见
[N1.8 NO-AUTH 报告](jacru_n1_8_hybrid_design_no_auth_2026-07-18.md)。
