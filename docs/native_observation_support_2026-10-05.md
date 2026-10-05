# 原生观测支持与完整视野

## 中文

本次更新只发布私有三维标量代理的定性接口检查，不公开原始场、相机几何、数值数组、图表或组内源码。物理语义与单位仍未确认，不能称真实像素 BOS 位移或真实 BOST 迁移。

内部独立第二实现分别生成几何、重建单元积分与伴随，并验证相机乱序。旧粗探测器存在零列节点；结果前冻结的几何配置使用已知坐标包围盒确定完整视野，并按原生网格维度密采样，消除了零列。保持旧视野、使用相同密度的对照也解除支持下界否决，因此不能将效果解释为独特算法、学习模型或完整视野公式的独占优势。

对零列集合 U，零初值且不混合未观测空间节点的解法，其 gauge-centered 场误差至少为 `||rho_U-mean(rho_U)|| / ||rho-mean(rho)||`。本次只检查这个必要下界。消除零列不代表算子满列秩，也不代表四项物理精度或参考稳定性通过。

更密的探测器改变每次 A/A^T 的成本，不能与旧问题的调用次数直接相比宣称提速。旧 PoolFire 参考与 GCV 判决保持冻结，不因这个不同观测家族的检查而改写。下一科学门仍是另行冻结的经典参考解及其数值和物理精度核对；warm initializer、较少调用、fresh wall/RSS、外部门与真实 BOST 均未由本次检查证明。

这不是公开外部基准或第三方复现。页面不能提供私有实验的完整可复算证据；详细输入和结果只在本地保留，供获授权的组内审查。

## English

This update publishes only a qualitative interface check on a private 3D scalar proxy. Raw fields, camera geometry, numerical arrays, figures and group source code are not released. Physical semantics and units remain unconfirmed, so this is neither calibrated pixel BOS displacement nor real-BOST transfer.

An internal independent second implementation separately generates geometry, rebuilds cell integration and its adjoint, and checks camera reordering. The old coarse detector contains zero-column nodes. A preregistered geometry configuration derives a complete field of view from the known coordinate box and samples densely at native-grid dimensions, removing these columns. The equally dense old-view control also clears the support-bound veto. The effect cannot be attributed to a unique algorithm, learned model or exclusive advantage of the complete-view formula.

For a zero-column set U, zero-start methods that do not mix unobserved spatial nodes have a gauge-centered field-error lower bound of `||rho_U-mean(rho_U)|| / ||rho-mean(rho)||`. Only this necessary bound is checked. Removing zero columns does not establish full column rank, four-metric physical accuracy or reference stability.

Denser detectors change the cost of each A/A^T call, preventing a direct call-count speed comparison with the old problem. Frozen PoolFire reference and GCV decisions are not revised by this different observation family. The next scientific gate remains a separately frozen classical reference with numerical and physical-accuracy checks. This interface check establishes no warm-initializer benefit, reduced calls, fresh wall/RSS advantage, external gate or real BOST.

This is not an open external benchmark or third-party reproduction. The page does not provide complete reproducible evidence for the private experiment; detailed inputs and results remain local for authorized group review.
