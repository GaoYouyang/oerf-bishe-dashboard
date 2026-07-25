# PoolFire C 路线：输运对齐与条件基底阅读路线

## 先说结论

v5.1 的固定全局子空间诊断即使把 rank 从 256 提到 504，p14 K4-target
oracle-projection p90 仍约为 0.149，离 0.05 绝对门很远。这个现象与
transport-dominated model reduction 的经典困难相似：一个局部结构只要在空间中
移动，未对齐 snapshot 的线性子空间就可能需要很多 mode。

这只是一个待验证类比，不等于 PoolFire 失败已经被证明来自平移。下一步必须先在
当前代理上测结构位置、尺度和形状，而不是直接套 shifted POD 或训练大网络。

**当前状态：T0 已按冻结协议失败；不授权 T1，也不授权神经训练。** 在线坐标只能来自
observation 或 `BP=A^T y`，不能用 K4 target 决定 shift；所有 zero-fill 越界都要
单列 cropped energy，不能把丢掉难重建区域当成表示改善。真实计算已经确认
K4 teacher 的位置在时间与工况间发生移动，但原始 BP 的质心存在明显工况相关偏差，
因此还没有资格驱动统一的整数平移。

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

### T0 已得到什么结果

Stage 1 只读六条轨迹的 observation，每帧恰好形成一次 `BP=A^T y`，总账为
`0 A + 606 A^T`。独立验证器重新生成全部 BP、质心、协方差、fit-only canonical
center 和整数 shift，逐数组最大绝对差均为 0。

Stage 2 再用同一冻结 observation 生成 606 个 K4 teacher，总账为
`2424 A + 2424 A^T`。T0 的主要结果是：

| 轨迹 | BP-target 质心 L∞ p50 / p90 / worst（体素） | 整数 shift 完全一致 | 误差不超过 1 体素 |
|---|---:|---:|---:|
| `p=14kw_size=01` | 0.185 / 0.304 / 0.361 | 70.30% | 100% |
| `p=14kw_size=05` | 0.793 / 0.962 / 1.126 | 31.68% | 99.01% |
| `p=22kw_size=03` | 0.224 / 0.359 / 0.453 | 85.15% | 100% |
| `p=33kw_size=01` | 0.320 / 0.516 / 0.651 | 37.62% | 100% |
| `p=45kw_size=05` | 1.181 / 1.847 / 2.108 | 14.85% | 63.37% |
| `p=58kw_size=03` | 0.371 / 0.628 / 0.753 | 57.43% | 100% |

K4 teacher 的全局 q10-q90 最大质心跨度为 2.596 体素，非零 oracle shift 帧占
64.52%，所以“结构会移动”成立。可是冻结协议要求每条轨迹的 BP proxy 都过门，
`p=45kw_size=05` 明显失败，其他多条轨迹的 exact-shift fraction 也未达 75%。
因此正式状态是 `FAIL_T0_BP_INTEGER_TRANSLATION_PROXY`。

这个结果不是“没有输运”，而是“原始反投影不能稳定定位输运”。尤其值得注意的是，
`p=45kw_size=05` 的三个质心分量相关系数仍约为 0.793、0.841、0.848，说明 BP
大致跟随移动，却带有显著系统偏差。协议在 SVD 之前停止，T1 没有运行；没有
aligned-basis 数字、神经模型、调用减少或速度结论。

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

由于 T0 已失败，T1 当前不运行。不能绕过这个失败，用 target-oracle shift 生成一张
更好看的 aligned-POD 曲线。

### Gate T2：条件基底还是 full-field decoder

只有后续新的、结果前冻结的 deployment-visible 定位机制先通过独立门，才比较：

- `single aligned basis`；
- `BP-feature selected mixture-of-subspaces`；
- 同总 decoder bytes 的全局 basis；
- 一个小型 BP→K4 3D residual U-Net sentinel。

五条 fit 必须做 trajectory-level leave-one-out。p14 已经参与多轮开发，只能作
development sanity check；p22 stopping validation 与两条 test 继续封存。

## 对毕业设计算法的启发

原始 BP 质心假设已被否掉；当前可保留的更窄算法假设是：

> 用一次物理 adjoint 得到 BP 后，先用固定、只依赖几何的
> `D^{-1} A^T y` 灵敏度均衡，消除不同体素被射线覆盖程度不同带来的位置偏差；只有
> 均衡后的部署可见质心在全部轨迹过门，才允许移动轻量局部表示，再由同一
> CGLS/PCGLS 做少量纠正。

这里的 `D` 来自固定几何的 `diag(A^T A)`，在线额外成本只是逐点乘法，仍保持每帧
一次 `A^T`。这比事后拟合一个自由 affine correction 更容易解释，也直接针对
`p=45kw_size=05` 的“相关但有系统偏差”。但它目前只是下一条待预注册假设，不是
创新成果。如果均衡 BP 仍失败，再考虑 fit-only、observation-visible 的校准映射；
仍不能直接跳到 FNO。

## 初学者最小练习

1. 用一维高斯脉冲生成 100 个平移 snapshot，画未对齐与对齐后的 singular-value
   decay。
2. 解释为什么一个只有一个形状参数的平移族，线性 POD rank 仍可能很高。
3. 手算一次整数 shift 的 zero-fill 与 periodic roll 差别。
4. 写出 `BP=A^T y` 的调用成本，并解释为什么 target centroid 不能部署。
5. 区分三件事：target 能被子空间表示、observation 能预测 coefficient、warm start
   经 refinement 后真的更快。
6. 解释 `diag(A^T A)` 为什么描述几何灵敏度，并比较 raw BP 与
   `D^{-1}A^T y` 在不增加完整 forward 调用时的在线成本。
