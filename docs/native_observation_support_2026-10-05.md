# 原生观测支持与完整视野

## 中文

本次更新只发布私有三维标量代理的定性接口检查，不公开原始场、相机几何、数值数组、图表或组内源码。物理语义与单位仍未确认，不能称真实像素 BOS 位移或真实 BOST 迁移。

内部独立第二实现分别生成几何、重建单元积分与伴随，并验证相机乱序。旧粗探测器存在零列节点；结果前冻结的几何配置使用已知坐标包围盒确定完整视野，并按原生网格维度密采样，消除了零列。保持旧视野、使用相同密度的对照也解除支持下界否决，因此不能将效果解释为独特算法、学习模型或完整视野公式的独占优势。

对零列集合 U，零初值且不混合未观测空间节点的解法，其 gauge-centered 场误差至少为 `||rho_U-mean(rho_U)|| / ||rho-mean(rho)||`。本次只检查这个必要下界。消除零列不代表算子满列秩，也不代表四项物理精度或参考稳定性通过。

更密的探测器改变每次 A/A^T 的成本，不能与旧问题的调用次数直接相比宣称提速。旧 PoolFire 参考与 GCV 判决保持冻结，不因这个不同观测家族的检查而改写。下一科学门仍是另行冻结的经典参考解及其数值和物理精度核对；warm initializer、较少调用、fresh wall/RSS、外部门与真实 BOST 均未由本次检查证明。

这不是公开外部基准或第三方复现。页面不能提供私有实验的完整可复算证据；详细输入和结果只在本地保留，供获授权的组内审查。

### 后续物理歧义与先验排序

未正则化原生参考未通过冻结预算内的收敛证书和独立一致性要求，保留不可判定，不能称已收敛最小范数解或精确零空间下界。随后只对同一个已封存固定场做新鲜物理重放，独立实现确认三维场差别很大时二维观测仍可接近。这是后开封有限场对的可识别性诊断，不是所有算法误差的普适下界，也没有假设实验噪声大小。

真值可见的固定场对检查进一步确认：各向同性 TV 偏好正确场，平方梯度惩罚反而偏好错误场。这个排序没有拟合参数或新逆解，不能证明 TV 可行解唯一或 TV 重建通过。TV 在 BOS 火焰层析中已有[研究先例](https://www.sciencedirect.com/science/article/pii/S0010218018302694)，不属于本项目的首次方法贡献。

随后单独冻结的经典干净观测约束 TV 求解完成。成熟求解库与单独递推、几何和梯度实现的有限步输出通过内部独立一致性核对，首帧三维精度改善；但新鲜可行性与固定点收敛证书没有通过，权威参考判决仍不可判定。不能把绝对精度门通过替代参考认证，也不能把昂贵经典求解当成便宜 initializer。没有开启完整时序扩展，不授权学习、提速、外部泛化或真实 BOST 结论；不事后加深迭代、放宽证书或挑选更容易的场。

## English

This update publishes only a qualitative interface check on a private 3D scalar proxy. Raw fields, camera geometry, numerical arrays, figures and group source code are not released. Physical semantics and units remain unconfirmed, so this is neither calibrated pixel BOS displacement nor real-BOST transfer.

An internal independent second implementation separately generates geometry, rebuilds cell integration and its adjoint, and checks camera reordering. The old coarse detector contains zero-column nodes. A preregistered geometry configuration derives a complete field of view from the known coordinate box and samples densely at native-grid dimensions, removing these columns. The equally dense old-view control also clears the support-bound veto. The effect cannot be attributed to a unique algorithm, learned model or exclusive advantage of the complete-view formula.

For a zero-column set U, zero-start methods that do not mix unobserved spatial nodes have a gauge-centered field-error lower bound of `||rho_U-mean(rho_U)|| / ||rho-mean(rho)||`. Only this necessary bound is checked. Removing zero columns does not establish full column rank, four-metric physical accuracy or reference stability.

Denser detectors change the cost of each A/A^T call, preventing a direct call-count speed comparison with the old problem. Frozen PoolFire reference and GCV decisions are not revised by this different observation family. The next scientific gate remains a separately frozen classical reference with numerical and physical-accuracy checks. This interface check establishes no warm-initializer benefit, reduced calls, fresh wall/RSS advantage, external gate or real BOST.

This is not an open external benchmark or third-party reproduction. The page does not provide complete reproducible evidence for the private experiment; detailed inputs and results remain local for authorized group review.

### Later ambiguity and prior ranking

The unregularized native reference fails its frozen-budget convergence certificate and independent consistency gates, remaining inconclusive rather than a certified minimum-norm solution or exact nullspace floor. A later fresh physical replay of an identical sealed fixed field independently confirms that large 3D differences can coexist with nearly identical 2D observations. This is a post-open finite-pair identifiability diagnosis, not a universal error bound for every algorithm, and no experimental noise level is assumed.

A truth-visible fixed-pair audit also confirms that isotropic TV favors the true field while a squared-gradient penalty favors the wrong field. This ranking fits no parameters and computes no new inverse solution; it proves neither uniqueness nor successful TV reconstruction. TV already has [BOS flame-tomography precedents](https://www.sciencedirect.com/science/article/pii/S0010218018302694) and is not a first-method contribution here.

The separately frozen classical TV solve under clean observation constraints is now complete. Finite iterates from an established solver library and a separate recurrence, geometry and gradient implementation pass internal independent consistency checks and improve first-frame 3D accuracy. Fresh feasibility and fixed-point convergence certificates nevertheless fail, leaving the authoritative reference verdict inconclusive. Absolute accuracy is not reference certification, and an expensive classical solve is not a cheap initializer. No full-sequence escalation, learning, speed, external generalization or real-BOST claim is authorized; iteration depth, certificates and target fields are not revised after results.
