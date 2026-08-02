# v106-v106.1：微分同胚输运在共同规范与严格嵌套网格下通过四级收敛门

## 一句话结论

师兄提出的微分同胚方向获得了第一个严格、可独立复算的数值机制正结果：在同一 5 帧三维反应流、6 类平滑可逆形变和 3 套九视角几何上，使用区间严格嵌套网格、精确 stride restriction，并让场、梯度与观测在同一个粗网格零均值规范下评分后，三项 p90、worst、逐单元单调性和最终绝对门全部通过。

```text
v106 = INCONCLUSIVE_INVALID_INTERVAL_NESTED_TRANSPORT_EXECUTION_V106
v106.1 = PASS_COMMON_GAUGE_INTERVAL_NESTED_TRANSPORT_CONVERGENCE_HEADROOM_V106_1
independent = PASS_INDEPENDENT_RECOMPUTATION_CASE6_COMMON_GAUGE_INTERVAL_NESTED_V106_1
numerical mechanism headroom = breakthrough
algorithm_breakthrough = false
```

这证明“坐标输运可以在当前 BOST 代理离散上形成可信收敛链”，但尚未证明 learned initializer、未见坐标泛化、真实速度优势或真实 BOST 成功。

## 为什么没有直接训练网络

微分同胚不是给网络加一个坐标参数那么简单。NeRIF 的 BOST 方程观测的是折射率梯度沿光路的积分，而不是绝对折射率常数；坐标变化后，标量场、协梯度、相机射线、探测器基、forward 与 adjoint 必须一致变化。若这一层离散数值本身不稳定，后续模型可能只是在拟合网格误差。

因此 v103-v106.1 依次回答：

1. 物理变换与离散伴随有没有写对；
2. 粗网格往返输运是否足够准确；
3. 高分辨率端点改善是否来自真实收敛；
4. 观测评分是否和场、梯度使用同一个物理规范。

## v106 为什么被判无效

v106 已经换成严格嵌套的节点序列：

```text
32 x 16 x 16
63 x 31 x 31
125 x 61 x 61
249 x 121 x 121
```

粗节点在每个更细网格上都是精确节点，因此 restriction 不再插值。但完整性检查发现：未施加任何形变时，场和内部梯度在四级网格上都保持约 `1e-16` 恒等，观测却从 `0` 漂到 `0.006034 / 0.008739 / 0.010092`。

原因不是形变，而是三类指标没有在同一规范状态上评分。对 BOST 来说，常数折射率偏置理论上不产生体内梯度观测；当前离散 support 边缘的有限差分却把这个偏置变成了假梯度。v106 因而 fail-closed，所有漂亮的收敛曲线都没有被拿来下科学结论。

## v106.1 只修了什么

v106.1 没有改帧、形变、几何、网格、插值、阈值或单调门，只做一处修复：

```text
field / gradient / observation
均对同一个 coarse support-zero-mean roundtrip tensor 评分
```

修复后，未形变参考态在四级网格上的场、内部梯度和观测恒等误差都不超过约 `1.66e-15`。

## 正式结果

### worst 误差

| 嵌套网格 | 场 roundtrip | 内部梯度 roundtrip | 观测 equivariance |
|---|---:|---:|---:|
| `1x 32x16x16` | 0.105339 | 0.291055 | 0.130879 |
| `2x 63x31x31` | 0.073452 | 0.252901 | 0.087492 |
| `4x 125x61x61` | 0.052644 | 0.187780 | 0.053826 |
| `8x 249x121x121` | **0.026397** | **0.095576** | **0.023116** |
| 冻结绝对上限 | 0.08 | 0.25 | 0.12 |

### p90-higher 误差

| 嵌套网格 | 场 | 内部梯度 | 观测 |
|---|---:|---:|---:|
| `1x` | 0.102160 | 0.281994 | 0.124446 |
| `2x` | 0.070447 | 0.202338 | 0.074355 |
| `4x` | 0.039936 | 0.149843 | 0.045286 |
| `8x` | **0.016726** | **0.052058** | **0.015018** |

三项的 p90 和 worst 都逐级下降。`8x/1x` worst 比为：

```text
field = 0.2506
interior gradient = 0.3284
observation = 0.1766
```

90 个物理单元中，三项的 `8x` 最终不劣比例均为 `100%`；三项逐单元四级单调比例也均为 `100%`。经验阶的中位数约为 `1.56 / 1.53 / 1.57`，仅作描述，不作为通过条件。

## 独立复算

独立程序没有导入正式 runner 或正式插值实现。它用手写八角点 gather、局部 stride restriction 和独立汇总逻辑重算全部 360 行，并重新散列已开封输入。

| 审计项 | 结果 |
|---|---:|
| 正式行 / 独立行 | `360 / 360` |
| 逐行指标最大差 | `1.58e-14` |
| 汇总最大差 | `1.18e-13` |
| 判据布尔不一致 | `0` |
| 正式输入输出漂移 | `false` |

独立路径仍共享冻结的九视角 physics kernel，因此不能声称端到端物理独立；但坐标插值、restriction、汇总和判决已经独立重写。

## 这项结果意味着什么

### 已经成功的部分

- 把师兄的建议从概念推进到三维反应流上的可执行坐标输运；
- 排除了非嵌套节点错位和规范不一致两个离散混杂因素；
- 三项尾部、逐单元单调和绝对门全部通过；
- 正式与独立实现闭合到约 `1e-13`。

这是一个真实的**数值机制门突破**：后续再研究坐标条件 warm start，已经不再建立在明显失真的 warp 上。

### 仍未成功的部分

- 没有训练 observation-only 模型；
- 没有在结果前封存的未见形变或未见相机位姿上测试；
- 没有证明 matched-accuracy 下减少 exact `A/A^T`；
- 没有 fresh wall、峰值内存、curved-ray 或真实 BOST 结果；
- 当前五帧和六类形变都属于已开封机制开发集。

所以正式状态仍是：

```text
algorithm_breakthrough = false
external_generalization = false
paper_success = false
real_BOST = false
GPU training = not authorized
```

## 与公开文献的边界

[Can neural operators always be continuously discretized?](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b31f6d65f2584b3c4347148db36fe07f-Abstract-Conference.html)指出：无限维空间的微分同胚不必能被有限维微分同胚连续逼近，表示本身也必须随离散收敛。v105.1 到 v106.1 的网格实验正是在检查这一前置条件，而不是把“连续理论正确”直接等同于“离散实现正确”。

[Radon Neural Operator](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e66233a208ef32f56df6312263239fa0-Abstract-Conference.html)在 sinogram 域讨论了微分同胚下的双 Lipschitz 强单调性质，说明投影域表示与坐标泛化具有直接联系；但它解决 PDE operator learning，不是 BOST 逆问题的精确伴随 warm start。

[NeRIF](https://arxiv.org/html/2409.14722v2)明确把 BOST 写成折射率梯度沿光路积分，并用连续坐标网络降低体素离散误差。它是本项目最直接的实验室物理父工作；本项目尝试补的是另一层：在保留 exact forward/adjoint 与迭代 refinement 的前提下，用坐标一致的低成本初始化减少昂贵求解调用。

一般的 DNO、Geo-FNO 和 DIMON 已覆盖“把不同几何拉回公共参考域学习”的思想，因此微分同胚本身不能写成原创。当前可能形成差异的窄组合仍是：

```text
BOST gradient/ray physics
+ diffeomorphism-consistent transport
+ exact adjoint lift
+ observation-only warm initialization
+ matched-accuracy exact-call and resource gate
```

## 下一步只做一件事

当前网格每一级都保留一格 support 边界，因此它的物理厚度会随分辨率变化。下一门先固定物理 support 厚度，保持其他设置不变，确认当前收敛不是边界层偶然变薄造成的。只有这一门继续通过，才把坐标描述接入最小 warm initializer，并在未见坐标变化上执行结果前冻结的 matched-accuracy 测试。
