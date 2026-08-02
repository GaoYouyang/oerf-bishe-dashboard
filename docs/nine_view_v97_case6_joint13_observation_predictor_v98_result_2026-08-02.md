# v98：联合 13 维小模型仍未通过严格 observation-only 门

## 结论

v98 得到一个经过独立复算的**关键负结果**：v96 已经证明联合 `13` 维表示在真值辅助下具有 `90/90` 容量，但在五折时间隔离和一帧 embargo 下，只看部署可见观测与已知几何的小模型仍不能稳定预测这 `13` 个联合坐标。

三个冻结候选的完整八门通过数为：

- geometry-conditioned scenario mean：`75/90`
- nested linear ridge：`75/90`
- nested RBF KRR：`71/90`

没有候选达到要求的 `90/90`，也没有候选在每个 outer fold 都达到 `18/18`。因此当前 mean / linear ridge / RBF KRR 直接联合坐标回归路线关闭，不能把 v96 的表示容量写成可部署算法成功。

```text
scientific_status = FAIL_NO_STRICT_OBSERVATION_ONLY_JOINT13_PREDICTOR_V98
algorithm_breakthrough = false
paper_success = false
gpu_rental_recommended_now = false
```

## 为什么必须先做这个小模型门

v97 已经排除了“固定旧九维、只预测四个新增系数”的结构。下一步最便宜且可证伪的问题是：若允许旧九维与新四维一起变化，普通的小型 observation-only predictor 能否学会 v96 的联合 witness？

如果 mean、线性 ridge 和 RBF KRR 已经能在严格外折中达到 `90/90`，才有理由比较更强网络；若它们失败，则应先看失败结构，判断缺的是模型非线性、局部空间特征、几何表示还是目标定义，而不是直接租 GPU 堆参数。

## 固定实验合同

- 数据角色：已经打开的公开 BLASTNet Case 6 开发工况。
- 样本：`30` 帧、`3` 档已知九视角几何，共 `90` 个单元。
- 外折：五个连续六帧时间折；同一物理帧的三档几何保持同一角色；边界一帧 embargo。
- 目标：独立验证后的 v96 联合 `13` 维 truth-aware witness。
- 输入：观测投影、残差、范数、物理 Gram 谱、物理球中心、parent K1 坐标、既有 observation features 和几何身份。
- 候选：scenario mean、nested linear ridge、nested RBF KRR；模型选择只用 outer-fit 内的联合坐标误差。
- 物理回放：预测先封存，随后才读取 held-out truth 并运行精确物理回放。
- 精度门：field、full-gradient、interior-gradient、observation 相对 Zero-K2 与 Zero-K4 的八门全部不越线。
- 在线精确算子账：每个候选严格为 `2A + 2A^T`。
- 反泄漏哨兵：修改 held-out target 后，模型选择与预测必须逐值不变。

## 正式结果

| observation-only 候选 | 完整通过 | 失败 | F12+ | F15+ | F30+ | 精确账 |
|---|---:|---:|---:|---:|---:|---:|
| scenario mean | 75/90 | 15 | 29/30 | 24/30 | 22/30 | 2A + 2A^T |
| nested linear ridge | **75/90** | 15 | 29/30 | 24/30 | 22/30 | 2A + 2A^T |
| nested RBF KRR | 71/90 | 19 | 28/30 | 23/30 | 20/30 | 2A + 2A^T |

线性 ridge 的逐 outer-fold 通过数为：

| 连续时间折 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| 通过数 / 18 | 11 | 12 | 16 | 18 | 18 |

scenario mean 为 `8 / 15 / 16 / 18 / 18`，RBF KRR 为 `8 / 13 / 15 / 17 / 18`。后三折明显更容易，前两折集中失败；这是已开封开发工况中的时间分布结构，不是外部泛化证据，也不能直接解释成某个确定物理阶段。

## 失败结构

复杂模型没有超过简单均值：linear 与 mean 同为 `75/90`，RBF 反而降为 `71/90`。这说明当前瓶颈不能简单归因于“回归器不够非线性”。

- scenario mean 的越线来自 full-gradient / Zero-K2 和 observation / Zero-K2；
- linear ridge 的失败同时涉及 field、full-gradient、interior-gradient 与 observation 门；
- RBF KRR 同样出现多种门越线；
- F30+ 通过数最低，表明较强几何条件下的尾部风险更大；
- Fold 3 与 Fold 4 的 mean/linear 已达到 `18/18`，但早期折没有稳定迁移。

因此，下一步不应在同一输入上无门槛扩大 MLP/FNO。更有价值的问题是：当前输入是否把相机几何仅当作类别标签，而没有表达坐标变化下的物理共变关系。

## 独立复算

独立 validator 没有导入正式 v98 模型或 runner，并重新实现特征缩放、mean / ridge / RBF、五折、`66` 个候选、联合坐标投影和 inner selection。它随后重新执行全部 `270` 个精确物理回放并重算八门。

复算结果：

- 三个 arm 的最终通过数与正式判决完全一致；
- held-out target mutation 对选择与预测的最大影响均为 `0`；
- 调用 receipt 失败数为 `0`；
- predicted-q 最大差：`8.54e-10`；
- field 最大差：`2.88e-9`；
- gate 最大差：`4.52e-9`；
- metric 最大差：`1.84e-7`，低于独立审计的 `1e-6` 数值容差；
- `algorithm_breakthrough=false`、`external_generalization=false`、`resource_advantage=false`。

两条路径仍共享 pre-v98 的物理和数据 kernel，因此这不是端到端独立物理实现证明；process-level never-read 也没有被证明。

## 师兄提出的几何原理边界

当前模型**没有**把每台相机的内外参或位姿矩阵作为连续输入。它使用的是：

1. 三档已知几何的身份编码；
2. 由固定 forward operator 派生的投影、Gram 谱与物理球特征。

这能做固定已知几何下的模型/求解器优化，但没有学习相机排列不变性、`SE(3)` 等变性或一般坐标变换，也没有验证未见机位泛化。

师兄建议的微分同胚路线因此是有根据的下一门。Geo-FNO、DNO 与 DIMON 的共同思想是：把不同物理域通过平滑可逆映射变换到统一参考域，再学习参考域上的算子。但 BOST 的相机变化首先改变测量算子，而不一定改变物理域；这里必须先建立自己的物理一致性，不能只把三维数组做图像形变。

## 下一条有效门：坐标变换可交换性

在训练任何新网络前，先冻结一个最小的 geometry/diffeomorphism feasibility gate：

1. 定义只依赖已知几何的平滑可逆映射 `phi_g`，并验证正 Jacobian、逆映射误差和边界行为。
2. 将密度按标量 pullback 变换；将密度梯度按 `J_phi^{-T}` 变换。
3. 同步变换相机射线、探测器坐标、forward 与 adjoint，检查变换前后的观测和内积恒等式是否可交换。
4. 分开比较：仅 geometry ID、显式连续位姿/射线编码、参考域 canonicalization、坐标等变约束。
5. 使用 leave-one-geometry-out，而不是只在三档几何内做时间折；未见机位必须在结果前封存。

只有物理可交换性先通过，才授权一个小型 set/ray-conditioned sentinel。若这一步失败，就说明“微分同胚增强”不能直接套到当前 BOST proxy，应回到测量算子的几何建模，而不是扩大网络。

## 证据边界

本轮不是：

- 相机位姿泛化成功；
- 微分同胚或群等变模型成功；
- 外部工况泛化；
- wall time 或内存加速；
- curved-ray 或真实 BOST；
- 论文完成或顶刊结果。

v96 的联合表示容量证据保留，但截至 v98：

```text
representation_capacity_breakthrough = true
strict_observation_only_joint_predictor = false
algorithm_breakthrough = false
external_generalization = false
resource_advantage = false
real_BOST = false
```

目前仍不建议租 GPU。

## 相关一级来源

- [Geo-FNO: Fourier Neural Operator with Learned Deformations for PDEs on General Geometries](https://www.jmlr.org/papers/v24/23-0064.html)
- [Diffeomorphism Neural Operator for various domains and parameters of partial differential equations](https://www.nature.com/articles/s42005-024-01911-3)
- [DIMON: A scalable framework for learning geometry-dependent solution operators](https://www.nature.com/articles/s43588-024-00732-2)

