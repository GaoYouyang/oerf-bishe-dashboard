# PoolFire C 路线：输运对齐与条件基底阅读路线

## 先说结论

v5.1 的固定全局子空间诊断即使把 rank 从 256 提到 504，p14 K4-target
oracle-projection p90 仍约为 0.149，离 0.05 绝对门很远。这个现象与
transport-dominated model reduction 的经典困难相似：一个局部结构只要在空间中
移动，未对齐 snapshot 的线性子空间就可能需要很多 mode。

这只是一个待验证类比，不等于 PoolFire 失败已经被证明来自平移。下一步必须先在
当前代理上测结构位置、尺度和形状，而不是直接套 shifted POD 或训练大网络。

**当前状态：T0 假设设计，不授权神经训练。** 在线坐标只能来自 observation 或
`BP=A^T y`，不能用 K4 target 决定 shift；所有 zero-fill 越界都要单列
cropped energy，不能把丢掉难重建区域当成表示改善。

## 五篇最相关的一级来源

### 1. Shifted Proper Orthogonal Decomposition

J. Reiss, P. Schulze, J. Sesterhenn, V. Mehrmann,
*The Shifted Proper Orthogonal Decomposition: A Mode Decomposition for
Multiple Transport Phenomena*, SIAM Journal on Scientific Computing 40(3),
A1322-A1344, 2018.

- 官方页：https://doi.org/10.1137/17M1140571
- 开放预印本：https://arxiv.org/abs/1512.01985
- 要提取：为什么 snapshot shift 能让奇异值更快衰减；多输运速度如何分离；
  shift 估计错误会怎样。
- 与本项目的关系：检查 K4 target 的高 rank 是否主要在“为同一结构的不同位置付费”。
- 不能直接照搬：论文中的 shift 可由完整状态估计；我们的在线 shift 必须来自
  deployment-visible observation 或 `BP=A^T y`，不能用未知 target。

### 2. Transport Reversal

D. Rim, S. Moe, R. J. LeVeque,
*Transport Reversal for Model Reduction of Hyperbolic Partial Differential
Equations*, SIAM/ASA Journal on Uncertainty Quantification 6(1), 118-150,
2018.

- 开放正文：https://arxiv.org/abs/1701.07529
- 要提取：snapshot matrix 为何出现慢奇异值衰减；pivot、shift、cut-off 和多个传播
  profile；插值 shift 引入的 numerical diffusion。
- 与本项目的关系：如果 PoolFire 中存在多个移动/变形结构，单一质心 shift 可能不够；
  必须报告裁剪能量和插值误差，不能让对齐本身偷偷降低误差。

### 3. Transported Subspaces

D. Rim, B. Peherstorfer, K. T. Mandli,
*Manifold Approximations via Transported Subspaces: Model Reduction for
Transport-Dominated Problems*, SIAM Journal on Scientific Computing 43(4),
A2563-A2586, 2021.

- 官方页：https://doi.org/10.1137/20M1316998
- 开放预印本：https://arxiv.org/abs/1912.13024
- 要提取：全局 transport 与局部线性 approximation space 如何组合；为什么这构成
  nonlinear manifold；offline/online 成本怎样分开。
- 与本项目的关系：比“一个固定 PCA basis”更自然的候选是
  `observable transport/alignment + local basis + short physical refinement`。

### 4. Registration-Based Model Reduction

T. Taddei,
*A Registration Method for Model Order Reduction: Data Compression and
Geometry Reduction*, SIAM Journal on Scientific Computing 42(2),
A997-A1027, 2020.

- 官方页：https://doi.org/10.1137/19M1271270
- 要提取：registration map 如何与 equation-independent compression 结合；
  deformation map 的可逆性、regularization 与几何误差。
- 与本项目的关系：相机/视角 geometry 固定不代表重建结构固定。若简单平移不够，
  可考虑小形变，但必须由 observation-visible feature 驱动并限制 Jacobian。

### 5. Deep Convolutional Autoencoder Nonlinear Manifold

K. Lee, K. T. Carlberg,
*Model Reduction of Dynamical Systems on Nonlinear Manifolds Using Deep
Convolutional Autoencoders*, Journal of Computational Physics 404, 108973,
2020.

- DOI：https://doi.org/10.1016/j.jcp.2019.108973
- 开放全文：https://www.osti.gov/servlets/purl/1574441
- 要提取：为什么 nonlinear trial manifold 可超过 optimal linear-subspace ROM；
  decoder Jacobian、minimum-residual projection与实际在线成本。
- 与本项目的关系：如果对齐/条件基底仍失败，full-field 3D convolutional decoder
  是合理下一步；但必须从 `BP` 或 observation 预测 field，不能继续使用 target-oracle
  projection。

补充：B. Peherstorfer,
*Model Reduction for Transport-Dominated Problems via Online Adaptive Bases
and Adaptive Sampling*, SIAM Journal on Scientific Computing 42(5),
A2803-A2836, 2020，官方页 https://doi.org/10.1137/19M1257275 。
它说明局部传播结构可以驱动在线 basis update；对我们而言，任何 update 成本都必须
折算为额外 A/A^T、wall time 和 memory。

## 下一步实验，不直接训练网络

### Gate T0：位置与形状是否真的发生跨工况偏移

只用五条 fit 与已开发化的 p14：

1. 对每帧 observation 计算 `BP=A^T y`，支付并记录一次 adjoint；
2. 用 `BP^2` 定义三维 energy centroid 和 covariance；
3. 对 K4 target 只做事后机制核对：比较 BP centroid 与 target energy centroid；
4. 报告逐 trajectory 的 centroid p50/p90/range、spread 和相关系数；
5. 若 BP 不能可靠指示 target 位置，停止 deployment-visible alignment。

这里不能用 target centroid 决定在线 shift；它只能回答 BP centroid 有没有资格成为
可部署 proxy。

### Gate T1：BP 可见的整数平移能否降低 oracle containment rank

在结果前冻结：

- canonical center 只能由 fit trajectory 决定；
- shift 来自 BP centroid，不能来自 K4 target；
- 先只允许整数 voxel shift；
- 平移后越界部分置零，禁止 periodic wrap；
- 单独报告被裁剪的 field energy；
- aligned basis 只用 fit；
- validation 先按 BP shift 对齐，oracle 投影后再逆 shift；
- 与 raw-origin 同 rank、公平 decoder bytes 比较。

若 p90/worst 没有实质改善，说明单一平移不是主因，停止继续雕刻 shifted-POD。

### Gate T2：条件基底还是 full-field decoder

只有 T1 有信号才比较：

- `single aligned basis`；
- `BP-feature selected mixture-of-subspaces`；
- 同总 decoder bytes 的全局 basis；
- 一个小型 BP→K4 3D residual U-Net sentinel。

五条 fit 必须做 trajectory-level leave-one-out。p14 已经参与多轮开发，只能作
development sanity check；p22 stopping validation 与两条 test 继续封存。

## 对毕业设计算法的启发

当前可保留的算法假设是：

> 用一次物理 adjoint 得到 BP，并从 BP 提取部署可见的输运/形态坐标；在该坐标下选择
> 或移动一个轻量局部表示，再由同一 CGLS/PCGLS 做少量纠正。只有 matched-accuracy
> 等价且完整 A/A^T、wall、memory 和 harm 都改善时，才称 warm-start 加速。

它比“再做一个 FNO”更贴合当前失败机制，也更贴近三维 BOST 的 physics-computation
接口。但目前它只是研究假设，不是创新成果。

## 初学者最小练习

1. 用一维高斯脉冲生成 100 个平移 snapshot，画未对齐与对齐后的 singular-value
   decay。
2. 解释为什么一个只有一个形状参数的平移族，线性 POD rank 仍可能很高。
3. 手算一次整数 shift 的 zero-fill 与 periodic roll 差别。
4. 写出 `BP=A^T y` 的调用成本，并解释为什么 target centroid 不能部署。
5. 区分三件事：target 能被子空间表示、observation 能预测 coefficient、warm start
   经 refinement 后真的更快。
