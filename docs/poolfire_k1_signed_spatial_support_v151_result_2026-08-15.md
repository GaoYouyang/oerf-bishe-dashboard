# v151：有符号空间状态没有关闭跨轨迹支持缺口

## 这次真正问了什么

v149.1 的小预测器没有通过完整轨迹外折。v150 随后发现，原来的 305 维 deployment-visible 汇总特征在部分留出工况上几乎没有跨轨迹邻域支持；但 v150 的一个连续误差数组超出预注册独立容差 `4.68e-11`，所以它的总状态必须保持 `INCONCLUSIVE`，不能事后放宽门槛。

v151 不再训练另一个模型，也不读取 CFD 真值或 Krylov 系数目标。它只检查一个物理上更丰富的可观测状态能否让五条已开封 PoolFire 轨迹在特征空间互相覆盖：

- 保留 v149 的 61 维局部汇总；
- 对每个相机/分量的 K1 residual 与 exact-K1 dual 保留有符号 `4x4` 低频 DCT 相位；
- 把其他相机的 residual 用报告的 right/up 轴提升到世界坐标、求平均，再投回目标相机坐标；
- 使用相机置换等变的 active-set mean/std/min/max 聚合；
- 五条完整轨迹 leave-one-trajectory-out，标准化和支持阈值只由四条 fit 轨迹决定。

本轮共有 `60,654` 个 active camera/component group rows。没有 target model、物理 replay、额外精确 `A/A^T` 调用或 GPU 训练。

## 实际结果

原 v149 汇总状态的全局支持率为 `84.43%`，40 个 trajectory-camera-component 分层中 `15/40` 通过，20 个 trajectory-camera 分层中 `7/20` 通过；最差分层只有 `0.216%`。

加入有符号空间与跨相机对齐后：

- 全局支持率降为 `67.42%`；
- component 分层为 `16/40`；
- camera 分层为 `8/20`；
- 最差 component 分层提高到 `4.98%`，但仍远低于冻结的 `90%` 门；
- p14-s05 / p22-s03 的轨迹支持率达到 `99.20% / 99.62%`；
- p33-s01 / p45-s05 / p58-s03 只有 `48.25% / 43.24% / 46.73%`。

所以正式科学判决为：

`FAIL_SIGNED_SPATIAL_CROSS_TRAJECTORY_SUPPORT_V151`

这不是“特征越多越差”的简单结论。新状态修复了最极端的局部缺口，却同时把 p33/p45/p58 与 fit 工况之间更深的分布差异暴露出来。当前问题更像工况覆盖和跨工况归一化，而不是模型容量不足。

## 独立复算

第二实现没有导入正式 v151 数值模块。它使用 SciPy 的 orthonormal `dctn`、独立世界坐标 peer alignment 和 `cdist` 邻居搜索，从原始 deployment-visible 数组重建全部状态与支持判决。

- signed feature 最大差：`5.33e-15`；
- fold normalization 最大差：`2.88e-14`；
- distance / threshold 最大差：`8.66e-15`；
- nearest indices、support flags 与离散判决全部一致；
- 所有独立检查通过，正式结果树和输入未改变。

因此这次负结果不是容差、网络、断网或实现分叉造成的。

## 路线如何调整

关闭当前 signed-spatial peer state，不在它上面训练更大的 CNN、FNO、UNO 或 DeepONet，也不租 GPU。下一门改为数据覆盖问题：利用尚未进入当前五轨迹审计、但已经属于公开 train split 的 PoolFire 工况，扩展功率与尺寸组合，再在结果前冻结一个只用已知工况和可观测量的归一化/支持审计。

这一步仍不能打开 validation/test，也不能用 held-out 轨迹身份或真值做调参。只有扩展后的 fit-only 支持门先通过，才有理由重新冻结一个最小预测器。

当前边界：

- `signed_spatial_peer_state_closed=true`；
- `predictor_training_authorized=false`；
- `physical_replay_authorized=false`；
- `gpu_rental_authorized=false`；
- `algorithm_breakthrough=false`；
- `resource_speedup=false`；
- `external_generalization=false`；
- `curved_ray_validated=false`；
- `real_bost=false`；
- `paper_success=false`。

公开证据：

- `docs/poolfire_k1_signed_spatial_support_v151_public_summary.json`
- `assets/figures/poolfire_k1_signed_spatial_support_v151.png`

## English checkpoint

v151 asks whether a richer deployment-visible state closes the cross-trajectory support gap behind the failed v149.1 group-coordinate predictors. It fits no target model and reads no CFD truth or Krylov-coordinate label.

The state retains the v149 local summary and adds signed low-frequency `4x4` DCT phase coefficients for each camera/component K1 residual and exact-K1 dual. Peer-camera residual vectors are aligned through reported right/up axes in world coordinates and reprojected into the target camera frame. Complete-trajectory leave-one-out normalization and support thresholds use fit trajectories only.

Across `60,654` active group rows, the scalar-summary baseline has `84.43%` global support. The signed spatial peer state falls to `67.42%`. It improves the worst component stratum from `0.216%` to `4.98%`, but only `16/40` component strata and `8/20` camera strata pass the frozen `90%` support gate. Trajectory support is `99.20% / 99.62% / 48.25% / 43.24% / 46.73%` for p14, p22, p33, p45, and p58 respectively.

An independent implementation uses SciPy `dctn`, an independently written world-frame peer alignment, and `cdist`. The maximum feature, normalization, and distance differences are `5.33e-15`, `2.88e-14`, and `8.66e-15`; all discrete support decisions match and every independent check passes.

The scientific decision is `FAIL_SIGNED_SPATIAL_CROSS_TRAJECTORY_SUPPORT_V151`. This closes the current signed-spatial peer state before predictor training. It does not prove that warm-start prediction is mathematically impossible, and it provides no reconstruction, exact-call, wall/RSS, external-transfer, curved-ray, real-BOST, or paper-success result. The next gate expands public training-condition coverage and freezes a target-free condition-normalization support audit before any new predictor is fitted.
