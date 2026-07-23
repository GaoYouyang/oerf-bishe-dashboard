# MGRS 两阶段候选门冻结说明

冻结时间：2026-07-22

## 1. 为什么不再继续调普通 hybrid

连续/离散审计已经显示，高频 Fourier MLP 可以在训练所用的中心差分渲染器上取得更低投影误差，却在连续自动导数和独立角度上出现明显恶化。2026 年的 *Neural Refractive Index Primitives* 又已经系统比较 automatic、discrete 和 hybrid gradients，并明确报告 Fourier 高频基在 automatic differentiation 下会放大梯度噪声。因此，“再换一个固定 AD/FD 比例”既没有在代表单元上超过低频基线，也不足以形成新颖性。

代表单元的 post-open 扫描结果同样是负面的：AD-only、25% AD hybrid 和四渲染器等权高频模型的 field relative-L2 分别为 `0.23420`、`0.14303`、`0.13538`，低频基线为 `0.13340`。这些控制不会进入未见单元。

## 2. 候选算法形状

暂称 **MGRS：Multi-scale Gradient-consensus Residual Safeguard**。

1. 先按既有离散中心差分损失训练低频基座 `[1,2,4]`；
2. 冻结基座；
3. 加一个最终输出严格为零的高频残差分支；
4. 残差同时拟合 automatic、`FD(h)`、`FD(h/2)`、`FD(h/4)` 四个投影损失；
5. 每次 checkpoint 都在独立 development 角上逐渲染器比较；
6. 只有四项都不劣于基座、且归一化平均至少改善 0.5%，才保存残差；否则恢复严格零输出，最终预测与基座逐值相同。

这不是数学上的真实场误差保证。它只是一个部署可见的多尺度一致性护栏，不能观察 BOST 零空间中的场误差，也不能替代独立相机、真实几何、有限孔径或实验验证。

## 3. 为什么分两阶段

唯一已经用于调参的单元是 `wrinkled_density_interface / 8% noise / seed 101`。其余新算法输出在冻结时尚未运行，但原低/高频基线输出已经公开，因此本轮不能称盲法确认。

- Stage A：`smooth_plume` 两档噪声，加 `wrinkled_density_interface / 2%`；同时比较 `MGRS-56` 与 `MGRS-6816`。
- Stage B：只运行 Stage A 按冻结规则机械选出的一个候选；四个单元全部来自未用于候选筛选的 `oblique_compression_sheet` 与 `shock_expansion_pair`。

模型 seed 先在每个物理单元内取中位数，绝不把两个优化 seed 当成两个独立物理样本。Stage A 的主终点是候选减基座的 field relative-L2 中位数；Stage B 除 field 外还要求 dense-angle AD 与原 test-angle AD 的中位数都不变坏。

## 4. 允许和禁止的结论

即使 Stage B 通过，也只允许写成：在固定连续解析 teacher、稀疏平行直线和两种 Gaussian 噪声下，一个零初始化、多渲染器准入的残差候选获得了需要继续外部验证的 synthetic signal。

不得写成：新算子学习算法、优于 NeRIF/DeepONet/FNO、真实 BOST 重建、OERF 数据成功、跨 rig 泛化、论文完成或突破。

## 5. 一级来源边界

- [NeRIF](https://arxiv.org/html/2409.14722v2)：已经使用随机 ray sampling、automatic/numerical differentiation consistency 和投影损失；MGRS 不能把“首次混合 AD/ND”当贡献。
- [Neural Refractive Index Primitives](https://arxiv.org/html/2605.11454)：已经比较 automatic/discrete/hybrid gradients，报告 Fourier 高频 AD 噪声，并加入 jitter 与 hash encoding；固定 hybrid、换 hash 或加 jitter 都不是本候选的新意。
- [Representation Equivalent Neural Operators](https://arxiv.org/abs/2305.19913)：已经定义离散表示导致的 operator aliasing；本轮只能说在 BOST 梯度投影链中测试一个具体机制，不能声称首次发现 aliasing。
- [mip-NeRF](https://arxiv.org/abs/2103.13415)：已经用连续尺度与积分位置编码处理成像采样混叠；若后续加入 footprint-aware encoding，必须作为直接相关基线和思想来源。

完整数值合同位于 `demo_t16_operator/configs/gcs_mgrs_stage_gate_v1.json`。配置必须先提交，再打开 Stage A。
