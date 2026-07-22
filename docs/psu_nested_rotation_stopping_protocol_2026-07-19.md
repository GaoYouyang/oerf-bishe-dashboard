# E72 PSU 嵌套旋转停止规则：冻结协议

状态：`FROZEN_BEFORE_ANY_SINGLE_ROTATION_TRAJECTORY_OR_NESTED_SELECTION`

## 0. 冻结后预检修订记录

初始协议提交 `5d47a80bef7091611dd4e424f8eb1f26b0ccb50c` 启动正式 inner 阶段时，逐视角预检发现配置中 `50°` 三台相机的射线数是转录错误。错误三数与真实三数总和恰好相同，因而总 ray-count 门未能识别；两份冻结 cache manifest 和 E71 证据均一致给出正确逐视角数量：

```text
view 3 / camera 2:  988006
view 4 / camera 3: 1363792
view 5 / camera 4: 1311324
```

运行器在 shared adjoint preflight、任何 CGLS 轨迹和任何 nested selection 之前 fail closed，所以修订时已计算轨迹数为 `0`、已选 checkpoint 数为 `0`。本修订只更正三个元数据整数；cache hash、观测数组、指标、checkpoint、容差、tie-break、回退和判决规则全部不变。此后的新提交才是正式 E72 protocol commit。

## 1. 这一步究竟回答什么

E71 已在同一个 PSU 真实流场上完成三个支撑旋转块的 CGLS 轨迹，并发现：从 `k=4` 继续迭代到 `k=12` 时，训练残差继续下降，但每个“旋转 × 网格”单元至少有一项留出投影指标恶化。这是**留出旋转重投影上的经验半收敛现象**，不是三维真值误差已经上升的证明。

E72 只问一个更窄、可证伪的问题：

> 不读取某个外层旋转的分数，仅用另外两个旋转之间的双向迁移，能否为双旋转 CGLS 重建选择一个停止步 `k*`，并在该外层旋转上相对固定 `k=4` 同时改善平均 camera-macro 指标且不伤害任何预先声明的组级、相机级和尾部指标？

必须保留一个不舒服但真实的限制：E71 的三个外层轨迹在 E72 设计前已经存在并被研究者看过。因此 E72 不是全新未见测试，只能称为：

**同一物理场上的 post-hoc programmatically withheld cross-fit reanalysis（事后设计、仅由程序暂不读取外层分数的交叉拟合再分析）。**

它不能支撑独立重复、统计显著性、三维场精度、跨流态泛化、通用停止定理或算法普遍优越性。

## 2. 数据角色

三个支撑旋转为 `0° / 50° / 90°`，每个旋转包含三台物理相机。相机行不是独立重复。

| 外层旋转 | 内层方向 1 | 内层方向 2 | E71 最终双旋转重建 |
|---|---|---|---|
| `0°` | train `50°` → validate `90°` | train `90°` → validate `50°` | train `50°+90°` → score `0°` |
| `50°` | train `0°` → validate `90°` | train `90°` → validate `0°` | train `0°+90°` → score `50°` |
| `90°` | train `0°` → validate `50°` | train `50°` → validate `0°` | train `0°+50°` → score `90°` |

对 `16³` 和 `32³` 分别选择、分别判决；禁止用外层结果选择网格。

## 3. 为什么只有六条拟合轨迹

唯一拟合键为：

```text
(train_rotation, grid, cache_manifest_hash, solver_config_hash)
```

外层 fold ID 禁止进入拟合键。每个网格只需训练 `0° / 50° / 90°` 三个单旋转模型；每个 checkpoint 对完整九视角做一次前向，再切成 train 旋转和另外两个 target 旋转。总计：

```text
3 train rotations × 2 grids = 6 CGLS trajectories
```

不是按三个 outer fold 重复得到十二条轨迹。

## 4. 冻结的 CGLS 路径

```text
K = {0, 1, 2, 3, 4, 6, 8, 12}
reference = k4
start = zero field
dtype = float64
gauge = zero outer boundary nodes
runtime = CPython 3.11.5 / NumPy 2.4.6 / PyTorch 2.13.0
```

正式命令固定使用仓库 `.venv/bin/python`。较旧的 PyTorch 2.2.2 不能直接处理本缓存的 `uint16` memmap 索引，因此运行器必须在打开正式缓存前核对上述版本；换环境不是无声兼容，而是需要新的环境验证与协议修订。

每个唯一拟合只跑一次十二步 CGLS，并克隆八个 checkpoint。`k=0` 是精确零场，不调用前向算子。

单个唯一拟合的逻辑调用预算：

| 部分 | `A` | `A^T` |
|---|---:|---:|
| train 子算子伴随点积检查 | 1 | 1 |
| 十二步 CGLS | 12 | 13 |
| 七个非零 checkpoint 的完整九视角评分 | 7 | 0 |
| 合计 | 20 | 14 |

每个网格再做一次共享完整算子伴随检查，因此两网格总预算固定为 `122 A / 86 A^T`。选择和 E71 连接均必须是 `0 A / 0 A^T`。

这里的 `122/86` 是六个完整 fit 与两个完整 shared-preflight **证据单元的 artifact budget**。shared preflight 也必须持久化并验哈希，恢复时不得无记录重跑。报告同时分开给出：本次 invocation 新执行的完整单元调用、复用单元数、发现的未完成 `.building` 次数，以及这些中断尝试可能增加的调用上界。只要存在中断单元，就禁止把 artifact budget 冒充进程生命周期的精确累计调用数。

## 5. 内层选择规则

对内层方向 `d`、指标 `m` 和 checkpoint `k`，定义相对 `k=4` 的比值：

```text
q[d,m](k) = E[d,m](k) / max(E[d,m](4), 1e-15)
```

主风险只包含三项：

```text
camera-macro relative-L2
worst-camera relative-L2
group p95 ray error / group signal RMS
```

```text
R(k) = max over two inner directions and three objective metrics of q[d,m](k)
A(k) = mean over two inner directions of q[d,macro](k)
```

但候选可行性比主风险更严格。候选必须在两个内层方向上同时满足：

- pooled、camera-macro、worst-camera relative-L2 不高于各自 `k=4 + 1e-8`；
- group p95 相对 `k=4` 的比值不高于 `1 + 1e-8`；
- 每台相机的 relative-L2 不高于各自 `k=4 + 1e-8`；
- 每台相机的 p95 比值不高于 `1 + 1e-8`。

`1e-8` 只是数值容差，不是重复性下限或物理显著性阈值。

选择顺序：

1. 非 `k=4` 候选必须可行且满足 `R(k) < 1 - 1e-8`。
2. 若没有这样的候选，强制回退 `k*=4`。
3. 否则最小化 `R(k)`。
4. `R` 在 `1e-8` 内并列时最小化 `A(k)`。
5. 再并列时选择更早的 iteration。
6. `k=0` 保留；若它获胜，只能解释为“零场在本屏幕中最安全”，不能解释为重建成功。

## 6. 两阶段封存

### 阶段 I：inner

输入只有冻结配置和两个私有 compact cache。该阶段：

1. 生成并逐单元校验六条唯一拟合；
2. 保存 48 个私有 float64 checkpoint 数组；
3. 公开输出仅含聚合分数、调用账本、哈希和选择；
4. 写入 `sealed_selection.json`；
5. 完全不读取 E71 outer summary。

inner 输出和 selection 必须先提交 Git，形成 `selection_commit`。

### 阶段 II：outer join

只有在 selection 已提交后，outer 命令才允许：

1. 验证 `selection_commit` 是当前 HEAD 的祖先；
2. 验证 inner summary、selection 和 finalization 与该提交逐字节一致；
3. 验证 E71 summary 和 finalization 的冻结 SHA-256；
4. 用已选 `k*` 查 E71 双旋转轨迹；
5. 不运行新的 `A/A^T`。

这种程序隔离防止代码路径偷看外层，但不能抹去研究者此前已经看过 E71 结果这一事实。因此结果状态名也必须使用 `POST_HOC_PROGRAMMATICALLY_WITHHELD`，避免“outer blind”被误读为研究者真正未见。

## 7. 外层判决

对每个网格分别计算：

```text
Delta_macro = mean over three outer rotations of
              [E_macro(k4) - E_macro(k*)]
```

通过必须同时满足：

- `Delta_macro > 1e-8`；
- 每个 outer fold 的 pooled、macro、worst、group-p95 均相对 `k=4` no-harm；
- 九个 held-out camera 单元的 relative-L2 和 p95 均相对 `k=4` no-harm。

任一条件失败，该网格判为 `NO_GO`。禁止看完 outer 后调整指标、容差、回退或 tie-break，再把同一数据称为新验证。

## 8. 冻结对照

- `k=4`：唯一 primary baseline。
- `k=0`：零场 sanity control。
- 最小训练 checkpoint residual：训练侧诊断，预计常偏向晚停。
- sparse-checkpoint L-curve：只用训练残差和解范数的离散 Menger 曲率；仅作诊断。
- GCV：当前禁用，因为没有经过验证的 influence trace 和 grouped noise model。
- discrepancy principle：当前禁用，因为没有独立重复性噪声尺度和模型偏差界。
- outer oracle checkpoint：不可部署的诊断上界，绝不参与选择。

## 9. 数值、恢复与隐私门

- cache manifest 和声明数组打开时全部验 SHA-256；
- train/full 算子伴随相对误差不超过 `1e-10`；
- 每个 solver call record 只含声明 train rotation；
- 每个评分 forward record 必须含完整九视角；
- 六个 fit key 唯一且不含 outer fold；
- checkpoint 形状、float64、有限性、SHA-256 全部验证；
- 六个 `k=0` 数组必须逐元素精确为零；
- 完整单元可恢复复用；未 finalization 的 `.building` 单元才允许重建；
- 两个 shared full-operator preflight 也作为独立持久单元恢复；
- 已 finalization 单元哈希不符时 fail closed；
- 任何恢复返回前仍须重新验证 Git protocol commit 与 dirty-worktree 门；
- public 输出拒绝 tensor、私有绝对路径、cache 文件名、测量数组和重建体素。
- public JSON 上限为 `10 MiB`；任一纯数值嵌套序列超过 `256` 个叶节点即拒绝，并对 raw field/voxel/measurement/checkpoint 键执行 deny-list。

## 10. 理论定位

- [Nemirovskii (1986)](https://doi.org/10.1016/0041-5553%2886%2990002-9) 支持把迭代次数视作病态问题中的正则化参数，但不证明当前 CGLS、几何误差或投影 holdout 满足其全部条件。
- [Hucker & Reiß (2025)](https://link.springer.com/article/10.1007/s00211-025-01469-4) 区分 prediction risk 与 reconstruction risk；E72 只能观察前者的一种 image-space proxy。
- [Morozov discrepancy principle](https://doi.org/10.1016/0041-5553%2866%2990046-2) 需要独立误差尺度，当前数据合同不具备。
- [Hansen & O'Leary (1993)](https://epubs.siam.org/doi/abs/10.1137/0914086) 给出 L-curve 背景；本协议的稀疏 checkpoint 实现只能作经典诊断对照。
- [Golub, Heath & Wahba (1979)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1979.10489751) 的 GCV 需要有效复杂度信息；不能把强相关像素或相机组直接当普通 leave-one-out 样本。
- [He et al., NeRIF](https://doi.org/10.1063/5.0250899) 说明连续折射率场与留出重投影在 BOST 中具有直接相关性，但其留出投影也不等于实验三维真值。
- [Hu et al., Pyramid BOST](https://link.springer.com/article/10.1007/s00348-025-04153-3) 是后续 coarse-to-fine 强基线；简单上采样 `16³` 场不等于该方法。

## 11. 决策后的下一步

- 若两个网格都 `NO_GO`：停止把“旋转选择停止步”当算法主贡献，转向 H1/TV、真正 pyramid/projection correction 或具备外部噪声合同的 discrepancy rule。
- 若仅一个网格通过：仍不能选它作为赢家；只能报告网格异质性，并在新物理场上预注册复验。
- 若两个网格均通过：只获得同一物理场的描述性 cross-fit 候选。下一道门仍是**新的流态/几何/重复实验**，之后才轮到神经修正或论文主张。
