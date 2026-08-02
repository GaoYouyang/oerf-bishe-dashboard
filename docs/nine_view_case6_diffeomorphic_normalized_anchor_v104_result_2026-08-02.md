# v103-v104：微分同胚数学链闭合，但粗网格坐标输运与固定目标 warm start 未通过

## 一句话结论

师兄提出的“加入微分同胚原理，提高坐标系变化后的泛化”已经被实现成两轮正式、可证伪的三维 BOST 代理实验，而不是只在网络名称里加入一个变换模块。

两轮最终状态是：

```text
v103 = INCONCLUSIVE_INVALID_NUMERICAL_TRANSPORT_V103
v104 = INCONCLUSIVE_INVALID_NUMERICAL_TRANSPORT_V104
v104 independent = PASS_INDEPENDENT_RECOMPUTATION_CASE6_DIFFEOMORPHIC_NORMALIZED_ANCHOR_V104
algorithm_breakthrough = false
```

数学接口本身通过：六类坐标变换都保持正 Jacobian、可逆、插值权重和为一，离散 forward/adjoint 的伴随误差最坏为 `5.60e-14`。但在 `32×16×16` 粗网格上，用三线性插值做形变和逆形变仍造成不可忽略的场与梯度损失；同时，固定 rank-0 物理均值目标即使经过可观测 BP 归一化，在恒等坐标下也是 `0/15` 通过，在六类小形变下是 `0/90` 通过。

因此当前准确结论不是“微分同胚没有用”，而是：**不能把粗网格三线性 warp 当成连续坐标泛化的可信实验载体，也不能用固定 rank-0 目标 + normalized BP + K1 形成安全 warm start。**

## 微分同胚怎样进入当前 BOST 代理

设参考坐标到物理坐标的光滑可逆映射为 `phi`，冻结的一族形变为：

```text
u' = u + a (1-u^2) sin(pi v / 2) cos(pi w / 2)
```

其中被移动的轴依次为 `x/y/z`，正负方向各一条。v103 使用 `|a|=0.18`；v104 只允许依据 v103 的无效结果做一次预注册修复，把幅值减为 `|a|=0.08`，没有扫参数挑结果。

离散算子不是只 warp 三维数组，而是：

```text
A_phi x   = A_ref P_phi S x
A_phi^T y = S P_phi^T A_ref^T y
```

- `S` 是固定的一体素物理边界投影；
- `P_phi` 是同一组三线性 gather 权重；
- `P_phi^T` 是这些权重的精确 scatter 转置；
- 物理真值和 fold-fit 目标用逆映射输运，并重新施加 support 与零均值 gauge；
- 正确搬运、完全不搬运、相反符号搬运三臂具有相同在线调用成本。

这只是一条 coordinate-conjugated straight-ray proxy，不是非线性折射光线或真实 BOST。

## v103 为什么无效

v103 首先测试较明显的 `|a|=0.18` 坐标变化。解析 Jacobian、逆映射、插值行和与离散伴随全部通过，但三项结果前冻结的数值保真门都失败：

| 粗网格坐标往返量 | p50 | p90-higher | worst | 允许 worst |
|---|---:|---:|---:|---:|
| 场 relative-L2 | 0.0792 | 0.1198 | **0.1468** | 0.08 |
| 内部梯度 relative-L2 | 0.1581 | 0.2893 | **0.2996** | 0.25 |
| 参考观测等变误差 | 0.0470 | 0.1203 | **0.1307** | 0.12 |

同时，未经归一化的 `A_phi^T y` 量级失控，正确搬运臂的 maximum-gate p50/p90/worst 达到约 `156.61 / 197.25 / 228.08`。所以 v103 只能定位两个机制问题：粗网格插值失真和反投影尺度失控；它不能用来判断微分同胚是否改善泛化。

## v104 只修了哪两件事

v104 在看任何新结果前只允许两项修复：

1. 把唯一幅值从 `0.18` 降为 `0.08`；
2. 对反投影做只看观测的最小残差标量归一化：

```text
h = A_phi^T y
s = <A_phi h, y> / ||A_phi h||^2
x0 = 0.75 s h + 0.25 t
```

随后从 `x0` 做一轮未修改 CGLS K1。每个候选实际为 `3A+2A^T`，与 `3A+3A^T` 的 Zero-K3 比较，并额外用 Zero-K4 检查 no-harm。它确实少一次精确 adjoint，但没有 wall/RSS 主张。

## v104 正式结果

### 1. 归一化修掉了量级爆炸，但没有修掉精度缺口

v104 的 maximum-gate 从 v103 的百量级降到约 `0.43–0.52`。这证明可观测标量归一化修复了明显的尺度病态；但 maximum-gate 定义为“实际比值减去冻结阈值”，所以正数仍然表示失败。

| v104 候选 | 完整八门通过 | maximum-gate p50 | p90-higher | worst |
|---|---:|---:|---:|---:|
| 恒等坐标 rank-0 normalized-BP K1 | **0 / 15** | 0.4540 | 0.5205 | 0.5279 |
| 正确微分同胚搬运 | **0 / 90** | 0.4579 | 0.5100 | 0.5213 |
| 不搬运 control | **0 / 90** | 0.4572 | 0.5096 | 0.5204 |
| 相反符号 control | **0 / 90** | 0.4568 | 0.5133 | 0.5222 |

正确搬运对“不搬运”的 p50/p90 改善为 `-0.000675 / -0.000357`；对相反符号搬运为 `-0.001093 / +0.003298`。这些值不但远低于冻结的 `0.01 / 0.02` 最低优势，而且方向不稳定。由于数值输运门先失败，这些对照只能用于定位，不能包装成正式算法优劣。

### 2. 更小形变仍没有通过粗网格保真门

| 粗网格坐标往返量 | p50 | p90-higher | worst | 允许 worst |
|---|---:|---:|---:|---:|
| 场 relative-L2 | 0.0502 | 0.1022 | **0.1053** | 0.08 |
| 内部梯度 relative-L2 | 0.1299 | 0.2820 | **0.2911** | 0.25 |
| 参考观测等变误差 | 0.0371 | 0.1244 | **0.1309** | 0.12 |

减小幅值改善了场的中位与最坏误差，却没有让梯度和观测最坏误差进入可信区间。继续把幅值调得更小只会让 stress 退化成接近恒等映射，无法回答坐标泛化问题，因此这条调参路线停止。

### 3. 数学与调用账没有问题

- 六类 warp 的解析 Jacobian determinant 为 `0.84–1.16`；
- 最坏条件数 `1.19048`；
- 逆映射残差最坏 `1.11e-16`；
- transformed operator 的伴随误差最坏 `5.60e-14`；
- 在线总账 `1590A+1305A^T`，离线观测生成 `105A`，伴随探针 `54A+54A^T`；
- 没有 solver breakdown。

这说明失败来自表示与离散精度，不是 forward/adjoint 写反或偷了调用预算。

## 独立复算

独立 validator 没有导入 v104 正式 core 或 runner。它用固定迭代二分法重建所有逆映射，用独立稀疏 gather/scatter 重建 `P/P^T`，重跑 495 条重建，再核对正式封存、输入绑定、摘要、对照统计和全部调用账。

最终差异：

| 独立复算项目 | 最大绝对差 |
|---|---:|
| 六个场/初值/残差数组 | `1.17e-15` |
| metrics 与八门 | `1.11e-15` |
| 数值诊断 | `7.63e-15` |
| 报告摘要 | `1.11e-15` |
| joint-pass 布尔不一致 | `0` |

正式输入与输出在验证前后保持不变。前两次验证尝试分别因显式三维 shape 检查发现扁平数组输入而停止，没有生成终态，也没有用于科学解释；修复后整套正式实验从新提交重新运行。

## 成功、失败和突破判断

### 成功

- 把师兄的建议转成了可证伪的三维坐标压力实验；
- 证明 `A_phi/A_phi^T` 的离散共轭与精确伴随可以实现到舍入误差；
- 找到并修复 raw BP 的尺度病态；
- 用正确、无、反向搬运 controls 证明当前结果没有可辨识的坐标搬运收益。

### 失败

- `32×16×16` 上三线性 warp 不足以承载可信的连续微分同胚压力门；
- 固定 rank-0 目标 + normalized BP + K1 连恒等坐标预条件都没有通过；
- 没有 matched-accuracy、外部坐标泛化、wall/RSS 或真实 BOST 结果。

### 当前突破状态

```text
diffeomorphic discrete adjoint = verified
coarse-grid transport fidelity = failed
fixed rank-0 normalized anchor = not viable
algorithm_breakthrough = false
external_generalization = false
paper_success = false
real_BOST = false
GPU training = not authorized
```

## 路线如何收缩

不再做第三轮幅值调参，也不在这套固定目标上训练 FNO、UNO、U-Net 或 DeepONet。下一科学门只问一个更基础的问题：**若在高分辨率/连续物理域先做坐标输运，再统一 restriction 到粗逆问题网格，场、梯度和观测误差是否随分辨率收敛？**

只有高分辨率 reference transport 通过收敛门，才重新设计 observation-adaptive、pose-conditioned initializer；否则微分同胚只保留为几何接口约束，不作为当前算法主干。这个决策把算力从无效的粗网格网络训练转回真正的物理瓶颈。

