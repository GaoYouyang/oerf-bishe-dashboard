# v105-v105.1：8× 微分同胚输运显著改善，但四级单调收敛门失败

## 一句话结论

高分辨率不是无效方向，但当前 `32→64→128→256` 节点数翻倍方案不能被称为收敛算法：`8×` 终点的场、内部梯度和九视角观测 worst 全部进入 v104 原有可信区间，但 `2×` 层三项都比 `1×` 更差，结果前冻结的尾部单调门为 `0/3`。

```text
v105 = INCONCLUSIVE_INVALID_HIGH_RESOLUTION_TRANSPORT_EXECUTION_V105
v105.1 = FAIL_NO_HIGH_RESOLUTION_DIFFEOMORPHIC_TRANSPORT_CONVERGENCE_V105_1
v105.1 independent = PASS_INDEPENDENT_RECOMPUTATION_CASE6_DIFFEOMORPHIC_TRANSPORT_CONVERGENCE_V105_1
algorithm_breakthrough = false
```

所以当前准确结论是：**真实源分辨率证明了细网格 endpoint headroom，但非嵌套网格序列没有通过严格离散收敛。** 不能只展示 `8×` 好结果并省略 `2×` 反弹，也不能据此开始大模型训练。

## 为什么必须做这一步

v103-v104 已证明六类三维微分同胚的 Jacobian、逆映射以及 `A_phi/A_phi^T` 离散伴随都正确，但在 `32×16×16` 上直接做两次三线性 warp 时，场、内部梯度和观测 worst 为：

```text
0.105339 / 0.291055 / 0.130877
```

它们分别超过 `0.08 / 0.25 / 0.12` 的冻结上限。v105 因此不再调整网络或形变幅值，而是直接读取已开封 BLASTNet Case 6 的公开 `1408×128×128` 原始 CFD 密度，在更细的物理网格完成正向和逆向坐标输运，最后只 restriction 一次到粗逆问题网格。

这一步回答的是离散物理前置问题，不是 warm-start 性能问题。

## 正式实验合同

- 物理样本：同一 5 帧已开封 Case 6 密度；
- 形变：`x/y/z` 三轴正负共 6 类，幅值固定 `|a|=0.08`；
- 相机：同一 3 套已知九视角几何；
- 分辨率：`32×16×16`、`64×32×32`、`128×64×64`、`256×128×128`；
- 每层：`5×6×3=90` 行，总计 360 行；
- 正式插值：分块 `scipy.ndimage.map_coordinates(order=1)`；
- 独立插值：手写分块八角点 gather，逆映射使用 48 步二分法；
- 没有训练、没有 warm initializer、没有打开新 validation/test、没有 GPU。

对每个分辨率 `H`：

```text
f_H -> inverse coordinate warp -> support/gauge
    -> forward coordinate warp -> support/gauge
    -> one coarse restriction
```

场和内部梯度对规约后的 roundtrip 评分；观测严格复用 v104 定义，对第二次零均值规约前的 pullback 施加 `A_ref`。

## v105 为什么 invalid

v105 首次正式批次和独立复算都完成了 360 行，但合同把 `A_ref` 施加在第二次零均值规约后的 roundtrip 上，而 v104 施加在规约前的 pullback 上。结果是：

- `1×` 场和内部梯度与 v104 一致；
- `1×` 观测指标最大差 `7.23e-5`；
- 原合同要求最大差不超过 `2e-10`。

因此 v105 必须记为 `INCONCLUSIVE_INVALID`。这些曲线没有直接参与最终科学裁决。

v105.1 只修复观测状态定义，四个分辨率、全部阈值、形变、样本和单调门均未改变。修复后 `1×` 全部指标与 v104 最大差为 `9.44e-16`。

## v105.1 正式结果

### 1. 三项 worst 随分辨率的实际轨迹

| 源信息网格 | 场 roundtrip | 内部梯度 roundtrip | 观测 equivariance |
|---|---:|---:|---:|
| `1× 32×16×16` | 0.105339 | 0.291055 | 0.130877 |
| `2× 64×32×32` | **0.130155** | **0.457941** | **0.144313** |
| `4× 128×64×64` | 0.059296 | 0.209123 | 0.064429 |
| `8× 256×128×128` | **0.034155** | **0.124688** | **0.031399** |
| 冻结上限 | 0.08 | 0.25 | 0.12 |

`8×` 终点三项全部进入可信区间，相对 `1×` worst 的比例为：

```text
field = 0.3242
interior gradient = 0.4284
observation = 0.2399
```

这是真实的数值 headroom，但它不是完整收敛证明。

### 2. 为什么正式状态仍然是 FAIL

结果前冻结的门要求：

1. 每项 `8×` worst 进入原 v104 上限；
2. p90-higher 和 worst 在四级网格上逐级不增；
3. `8×/1×` worst 不超过 `0.8`；
4. 至少 95% 单元的 `8×` 不劣于 `1×`；
5. 至少 80% 单元在四级网格上逐级不增。

实际结果：

| 判据 | 场 | 内部梯度 | 观测 |
|---|---:|---:|---:|
| `8×` 绝对门 | PASS | PASS | PASS |
| `8×/1×` ratio | PASS | PASS | PASS |
| p90 逐级不增 | FAIL | FAIL | FAIL |
| worst 逐级不增 | FAIL | FAIL | FAIL |
| `8×` cellwise noninferior | 100.0% | 93.3% | 93.3% |
| 四级 cellwise monotone | 33.3% | 20.0% | 26.7% |

`2×` 反弹不是少数异常点，而是三类尾部共同出现。因此预注册结论只能是 `FAIL_NO_HIGH_RESOLUTION_DIFFEOMORPHIC_TRANSPORT_CONVERGENCE_V105_1`。

## 独立复算

独立程序没有导入 v105/v105.1 正式 core 或 runner。它重新读取 5 个原始密度文件，用手写八角点 gather、48 步二分逆映射和独立 restriction 重算全部 360 行。

| 项目 | 最大差 |
|---|---:|
| 逐行三个指标 | `4.48e-15` |
| 摘要与收敛报告 | `2.33e-14` |
| 判据布尔不一致 | `0` |
| 正式输入输出漂移 | `false` |

独立路径仍共享冻结的网格验证和九视角 physics kernel，因此 `end_to_end_physics_independence_proven=false`；但三线性输运与逆映射本身是独立实现。

## 物理与数值解释

`32→64→128→256` 看似每次翻倍，实际节点并不嵌套。例如 `32` 个节点有 `31` 个区间，而 `64` 个节点有 `63` 个区间；粗网格节点并不是细网格节点的严格子集。restriction 因此又进行一次错位插值，可能造成 `2×` 的系统性反弹。

下一门不是把 `2×` 删除，也不是只报 `4×/8×`，而是使用区间严格嵌套的网格：

```text
32×16×16
63×31×31
125×61×61
249×121×121
```

这样 coarse node 在每个细网格上都是精确节点，restriction 可以直接 stride，不再混入额外插值。如果嵌套网格仍不单调，才进一步检验“一体素 support 的物理厚度随分辨率变化”是否是主要原因。

## 与已有坐标神经算子的关系

坐标变换并不是空白方向。Geo-FNO 学习物理域到规则潜空间的 deformation；GINO 用图神经算子在不规则几何与规则潜网格之间投影；CT-FNO 显式使用 canonical coordinate transforms 处理域形状和对称性。它们都说明师兄提出的方向有文献基础，但也意味着“加入坐标变换”本身不能作为创新点。

- [Geo-FNO: Fourier Neural Operator with Learned Deformations for PDEs on General Geometries](https://arxiv.org/abs/2207.05209)
- [GINO: Geometry-Informed Neural Operator for Large-Scale 3D PDEs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)
- [CT-FNO: Coordinate Transform Fourier Neural Operators](https://openreview.net/forum?id=pMD7A77k3i)
- [CORAL: Operator Learning with Neural Fields on General Geometries](https://openreview.net/forum?id=4jEjq5nhg1)

当前可能形成论文差异的仍是窄组合：**BOST 逆问题、坐标一致的 forward/adjoint、固定昂贵调用预算、matched-accuracy refinement 与可观测回退**。这需要后续真实结果支持，不能先写成首创。

## 成功、失败与突破状态

### 成功

- 第一次使用公开原始 `1408×128×128` CFD，而不是对粗网格上采样；
- `8×` 终点三项 worst 全部进入原 v104 可信区间；
- 正式与独立插值路径闭合到约 `1e-14`；
- 定位出非嵌套节点序列这一可检验的数值混杂因素。

### 失败

- 四级 p90 和 worst 单调门为 `0/3`；
- 内部梯度与观测的 cellwise final-noninferior 只有 `93.3%`，低于 95%；
- 没有 warm-start、matched-accuracy、A/A^T、wall/RSS 或真实 BOST 结果。

### 当前状态

```text
8x absolute numerical fidelity = passed
four-level monotone convergence = failed
warm initializer evaluated = false
algorithm_breakthrough = false
external_generalization = false
paper_success = false
real_BOST = false
GPU training = not authorized
```

