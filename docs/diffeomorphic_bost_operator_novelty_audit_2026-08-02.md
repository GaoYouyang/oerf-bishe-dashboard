# 微分同胚原理怎样真正进入三维 BOST warm start：近邻工作、创新边界与必做对照

更新日期：2026-08-02

## 先给结论

师兄的建议是正确的，但“使用微分同胚”本身不是创新点。DNO、Geo-FNO 与 DIMON 已经系统研究了把不同物理域映射到公共参考域后再学习算子。

当前项目仍有研究价值的窄缺口是：

```text
折射率/密度标量 pullback
+ 协梯度 J^{-T}
+ 标定 camera ray / detector basis
+ BOST forward / adjoint 可交换
+ observation-only warm initializer
+ matched-accuracy 下减少 exact A/A^T
```

截至本次限定在官方期刊、作者 arXiv 与作者代码的审计，没有发现一项公开近邻同时闭合以上全部环节。这只能写成“在限定来源范围内未发现完全同构方法”，不能写成全球首个、唯一或 SOTA。

## 六个最接近的方法

| 方法 | 它真正解决什么 | 与本项目的重合 | 仍缺什么 |
|---|---|---|---|
| [Diffeomorphism Neural Operator](https://www.nature.com/articles/s42005-024-01911-3) | 将不同形状 PDE 域通过微分同胚映射到 generic domain，在公共域学习参数函数到解函数；正式发表于 2025-01-08 | 与 v101 的相机无关物理参考域最接近 | 未处理 BOST ray、detector basis 与物理 adjoint |
| [Geo-FNO](https://www.jmlr.org/papers/v24/23-0064.html) | 学习物理域到规则潜在域的 deformation，再用 FFT/FNO 处理不规则几何 | 可作为未来 learned deformation 基线 | inverse Fourier transform 不是逆问题的 exact `A^T` |
| [DIMON](https://www.nature.com/articles/s43588-024-00732-2) | 用显式微分同胚把几何依赖 PDE 算子送到统一模板域，并给出近似理论 | 是 v99 到 v101 “先输运、后学习”的最强理论近邻 | 未落地多相机 BOST 测量链 |
| [Adaptive Coordinate Transforms](https://arxiv.org/abs/2605.06203) | 在神经算子层间学习数据自适应坐标重采样，追踪移动结构 | 可作 observation-adaptive warp 对照 | 不保证全局可逆，不能自动称微分同胚 |
| [Diffeomorphism-Equivariant Neural Networks](https://arxiv.org/abs/2602.06695) | 通过可微配准把图像规范化到群轨道的 canonical representative | 对应“先规范化观测、再预测 initializer” | 当前是图像识别/分割，不是 BOST 逆问题 |
| [Diffeomorphic Neural Operator Learning](https://arxiv.org/abs/2508.06690) | 用网络预测微分同胚并以映射复合推进输运型演化 | 可启发用可逆形变表达流动结构 | 学的是时间演化，不是多视角重建 |

## 物理接口矩阵

| 方法 | 标量 pullback | 梯度 Jacobian 变换 | ray / detector basis | BOST forward / adjoint |
|---|---|---|---|---|
| DNO | 有函数复合 | 未明确实现 BOST 协梯度 | 无 | 无 |
| Geo-FNO | 有 | 含几何变换与测度，但不是 BOST 梯度链 | 无 | 无 |
| DIMON | 有 | 理论允许按物理量类型输运 | 无 | 无 |
| ACT | 隐特征重采样 | 不是显式物理梯度通道 | 无 | 无 |
| DiffeoNN | 图像复合 | 无 | 无 | 无 |
| Diffeomorphic NO | 标量或守恒密度群作用 | 未处理 BOST 梯度投影 | 无 | 无 |
| 本项目 v99 | 有 | 有，`J^{-T}` | 有 | 有，含 forward/adjoint 交换与 dot-product test |

这里最容易混淆的一点是：模型训练中的反向传播不能自动算作经过物理验证的 BOST adjoint。我们需要的是测量算子 `A_phi` 与精确伴随 `A_phi^T` 在坐标输运下共同成立。

## v99、v100、v101 分别回答什么

### v99：物理接口

v99 在两个三维格点旋转和一个显式光滑剪切上验证标量场、协梯度、ray、detector basis、forward 与 adjoint 的一致输运，11 个门全部被独立复算。它证明实现原则正确，不证明任意微分同胚或未见机位泛化。

### v100：错误目标被排除

v100 完整留出相机几何与连续时间块。direct target + K1 为 `90/90`，但 geometry-specific reconstruction endpoint 的共享 rank 0 至 40 场基最高只有 `2/90`。这说明不同几何下的优化器终点不是合适的公共物理目标，继续放大其 PCA/FNO predictor 没有依据。

### v101：公共参考域容量

v101 改用相机无关的物理密度真值，只从每折唯一物理时刻建立参考域基，再与部署可见 detector-dual anchor 按固定 beta 混合。它当前只回答 truth-aware 容量是否存在；正式与独立结果出来前，不是 observation-only 模型、泛化或算法成功。

## 论文必须保留的对照

1. 完整输运、scalar-only、去掉 `J^{-T}`、只变 ray、固定 detector basis、wrong warp。
2. `A_phi T rho` 的 forward 交换、adjoint 交换和独立 dot-product test。
3. `det J` 最小值、folding 数、逆映射往返误差与边界保持。
4. “场与相机共同输运”和“固定物理相机只重参数化场”必须分开测试。
5. geometry ID、continuous pose/ray、Geo-FNO deformation、DIMON 式参考域、ACT 与 no-warp。
6. v100 endpoint target 与 v101 physical truth target，保持相同 K1 外壳和成本账。
7. 完整留出相机 geometry，并按形变幅度、Jacobian condition number 和训练域距离分层报告。
8. field、full-gradient、interior-gradient、observation 八门全部 matched-accuracy，再报告逐工况 harm、exact `A/A^T`、fresh wall 与 whole-pipeline RSS。
9. 最终使用组内位移图、完整标定、重复测量噪声与师兄认可基线重跑真实 BOST 迁移。

## 可以向师兄怎样准确表述

> 我们已经先把微分同胚原理落实到 BOST 的物理坐标输运，而不是只 warp 三维数组。v99 验证了标量、梯度、ray、detector basis、forward 和 adjoint 的共同变换；v100 又发现不同几何的重建终点不能直接组成公共输出空间。现在 v101 正在改用相机无关真值参考域做容量门。只有它通过，才训练 observation 与连续相机描述到参考域系数的小模型。

当前证据状态：

```text
coordinate_transport_interface = passed
geometry_specific_endpoint_common_space = failed
reference_truth_capacity = running
strict_observation_only_unseen_geometry = not run
algorithm_breakthrough = false
paper_success = false
real_BOST = false
```
