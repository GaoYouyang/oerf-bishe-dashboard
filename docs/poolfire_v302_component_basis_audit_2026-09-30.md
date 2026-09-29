# v302: Shared 2-D Component-Basis Audit / 共享二维分量基底诊断

**Date / 日期:** 2026-09-30
**Evidence grade / 证据等级:** independently recomputed post-open diagnostic / 独立复算的已开封诊断

## 中文

### 观测量说明

本诊断所用的 V299–V302 前向分支输出两个横向小偏折角，单位为弧度；它没有把角度映射到背景面物理位移或图像像素，也没有使用经验证的混合物密度—折射率关系。项目中的 V282 是另一条独立的虚拟有限背景像素前向，在明确假设下生成像素位移并验证线性化算子/伴随；它不是实测 BOS，也不能自动替代本诊断的角度观测。两条链均不能据此声称已验证真实实验位移。

### 问题与方法

V300 的事后分析显示，两个图像位移分量需要不同的标量增益；V301 则显示相机间差异温和。V302 检查一个更窄的可能性：是否存在跨相机共享的二维分量混合（例如存储分量基底不一致）。只使用相同五条已开封 PoolFire 轨迹的 25 行、9 个视角及原有效射线掩码。每折留一整条轨迹，用其余四条拟合，再在留出的整条轨迹上比较恒等映射、对角缩放和无截距的完整 2×2 线性映射。离散投影由已开封真值经冻结算子计算，因此这是回顾性归因，不是部署输入。

### 结果

- 对角缩放在 5/5 留出轨迹上都降低了归一化观测残差；恒等映射残差范围为 `0.21547–0.29395`，对角缩放后为 `0.13942–0.20892`。
- 完整 2×2 映射仅在 1/5 折略优于对角缩放，最大差约 `2.56×10⁻⁵`；其余 4 折略差，最大差约 `3.47×10⁻⁵`。拟合的非对角项绝对值最大约 `0.00470`。
- formal 与独立数值路径最大差为 `2.66×10⁻¹⁵`，上游封存输入复核未变。

### 解释边界

结果削弱了“共享的分量旋转/交叉混合是主要错配来源”这一特定解释；它与分量尺度差异相容，但不能确认物理来源。相机局部坐标是否可由同一矩阵变换也尚未由真实标定验证。本结果不构成相机标定、因果识别、可部署修正、前瞻泛化、matched-accuracy 调用节省、wall/RSS 收益、真实 BOST 或算法突破；v284 的整轨迹严格成本失败不变。

## English

### Observable clarification

The V299–V302 forward branch used here outputs two transverse small-angle deflections in radians. It does not map them to physical background-plane or image-pixel displacement, and it does not use a validated mixture density-to-refractivity relation. V282 is a separate virtual finite-background pixel forward under explicit assumptions; its linearized operator/adjoint check is not measured BOS and does not automatically replace the angular observations in this diagnostic. Neither branch establishes validation against measured experimental displacement.

### Question and method

V300 found different post-hoc scalar gains for the two image-displacement components, while V301 found modest camera-wise variation. V302 tests one narrower possibility: a shared 2-D component mixing, such as a stored component-basis mismatch. It reuses only 25 rows from the same five already-open PoolFire trajectories, nine views, and the original valid-ray masks. Each fold holds out one complete trajectory; identity, diagonal scaling, and an unconstrained zero-intercept 2×2 map are fitted on the other four and scored on the held-out trajectory. The discrete projection is formed by applying the frozen operator to already-open truth, so this is retrospective attribution, not a deployment input.

### Results

- Diagonal scaling lowers normalized observation residual on all 5/5 held-out trajectories; identity residuals range from `0.21547–0.29395`, versus `0.13942–0.20892` after diagonal scaling.
- The full 2×2 map is only slightly better than diagonal scaling in 1/5 folds (largest improvement about `2.56×10⁻⁵`) and slightly worse in the other 4/5 (largest increase about `3.47×10⁻⁵`). The largest absolute fitted off-diagonal term is about `0.00470`.
- The formal and independent numerical paths differ by at most `2.66×10⁻¹⁵`; sealed upstream inputs were rechecked unchanged.

### Interpretation limits

This weakens the specific explanation that shared cross-component rotation/mixing is the main source of mismatch. It is compatible with component-dependent scaling but does not identify its physical cause. Whether local camera coordinates admit one shared transform is not verified by real calibration. This is not camera calibration, causal identification, a deployable correction, prospective generalization, matched-accuracy call savings, wall/RSS benefit, real BOST, or an algorithmic breakthrough. V284's strict full-trajectory cost failure remains unchanged.

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.
