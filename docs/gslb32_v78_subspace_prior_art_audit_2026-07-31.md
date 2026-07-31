# GSLB32 v78 子空间在先技术审计

> 日期：2026-07-31  
> 角色：结果无关的原创性边界；在 v78 正式结果完成前冻结。  
> 当前状态：`algorithm_breakthrough=false`。

## 1. 本轮为什么检索

v78 检验的是一个很窄的问题：固定九视角几何下，把先验校正限制在 32 维几何样条
子空间，是否能在相同的 `2A+2A^T` 在线预算内，让 75 个已经开封的 Case3 cell 全部
通过 field、gradient、observation 与 harm 门。

即使这个数值门通过，也不能自动声称“首次使用低维子空间加速迭代重建”。子空间增广、
Krylov 回收、reduced basis 和 learned warm start 都已有成熟先例。本审计用于把未来可写
贡献限制在 BOST 特定的完整组合，而不是某个已经存在的零件。

## 2. 新增的一级来源

| 来源 | 已覆盖的思想 | 对本项目的直接约束 |
|---|---|---|
| [A Framework for Deflated and Augmented Krylov Subspace Methods](https://epubs.siam.org/doi/10.1137/110820713) | 用固定子空间增广 Krylov 空间，并讨论显式增广与投影后修正的等价框架 | 不能把“固定低维空间 + Krylov refinement”本身写成创新 |
| [Hybrid Projection Methods with Recycling for Inverse Problems](https://epubs.siam.org/doi/10.1137/20M1349515) | 对离散逆问题压缩并回收解空间 basis，在保持近似精度的同时降低存储或计算 | 不能泛称首次把跨样本信息压缩成 basis 来加速逆问题 |
| [Subspace Recycling-Based Regularization Methods](https://epubs.siam.org/doi/10.1137/20M1379617) | 证明合适条件下的子空间增广仍可保持正则化性质，并在成像逆问题中验证 | 若未来讨论正则化或稳定性，必须与该理论区分，不能由经验门替代理论 |
| [Reduced Krylov Basis Methods for Parametric Partial Differential Equations](https://epubs.siam.org/doi/10.1137/24M1661236) | 从一个高保真参数实例生成 reduced Krylov basis，再低成本求解参数问题族 | 不能把“离线昂贵 basis、在线低维解”写成首次；必须突出 BOST observation-only 选择与完整成本门 |
| [Deep learning initialized compressed sensing in volumetric spatio-temporal subspace reconstruction](https://link.springer.com/article/10.1007/s10334-024-01222-2) | 以深度学习结果 warm-start 固定迭代重建，并报告重建时间下降 | 不能声称首次用学习初值降低体重建时间；最终必须比较完整 wall/RSS，而非只报迭代数 |
| [A Neural Network Warm-Start Approach for the Inverse Acoustic Obstacle Scattering Problem](https://arxiv.org/abs/2212.08736) | 从散射场观测预测星形边界的 `2M+1` 个 Fourier 系数，再由标准 Gauss-Newton 精修 | 不能声称首次实现 observation-only 到低维几何系数再接传统物理迭代；差异必须落在 BOST、精确 `A^T` lift、固定 CGLS/PCGLS 与成本门 |
| [On Learned Operator Correction in Inverse Problems](https://epubs.siam.org/doi/10.1137/20M1338460) | 学习 forward/adjoint correction，并给出修正算子进入变分逆问题的条件 | 如果后续从 straight-ray proxy 迁移 curved/real BOST，必须区分“初值校正”和“算子校正” |

## 3. v78 通过后仍然不能写什么

- 不能写 first reduced-basis reconstruction。
- 不能写 first augmented/recycled Krylov inverse solver。
- 不能写 first learned warm start for volumetric reconstruction。
- 不能写 first observation-to-low-dimensional-coefficients warm start。
- 不能写 first offline-expensive / online-cheap inverse method。
- 不能把 75/75 的 truth-aware oracle feasibility 写成可部署模型、外部泛化或真实 BOST。

## 4. 尚可能保留的窄贡献

只有以下完整结构在后续证据中共同成立，才可能形成可防御的方法贡献：

1. 针对 BOST 多视角偏折观测和 reference/gauge 语义；
2. 低维字典由冻结几何构造，但在线系数只能由部署可见 observation 决定；
3. 初值经精确 `A^T` lift，随后使用未修改的 CGLS/PCGLS；
4. 与 Zero/BP/PCGLS/dual-ridge、解析投影、recycling/reduced-basis controls 公平比较；
5. 在 matched field/gradient/observation accuracy 下减少完整 `A/A^T`；
6. fresh-process wall 和 whole-pipeline peak RSS 也真实下降；
7. 独立公开反应流外门成立，最终再迁移组内真实 BOST。

因此，v78 的科学作用只是回答“32 维父空间是否有足够表示 headroom”。它不是论文算法
本身。若 v78 失败，应关闭该容量，并按预注册规则决定是否只做一次 127 维父空间上界；
若 v78 通过，也只能授权训练最小 observation-only 系数预测器。

## 5. 当前判决

截至 2026-07-31，本轮新增来源进一步确认：**组件级原创性不成立**。限定公开来源范围内，
仍未发现完全同构覆盖“BOST 特定 observation-only 几何样条系数、精确 lift、未修改
CGLS/PCGLS、matched-accuracy 成本账、外部门与真实 BOST 迁移”的方法，但这只是
`not found in reviewed sources`，不是全球唯一证明。
