# PoolFire cell-centred 接口判别证据

> 日期：2026-07-23
>
> 主线：C 路线，神经算子 warm start，在同精度下降低三维 BOST 重建成本
>
> 证据等级：`TIER_A_CELL_CENTER_BRIDGE_CODE_GATE_ONLY`
>
> 机器判决：`PASS_CELL_CENTER_ROUTE_DISCRIMINATION_CODE_GATE_ONLY`
>
> 物理门：`G0_PHYSICS_HOLD`
>
> 训练授权：`false`

## 1. 这轮到底解决了什么

上一轮 Tier-A 算子接收的是 Cartesian **节点场** `Delta n(x_i,y_j,z_k)`，但 PoolFire 的 `rho` 来自 CFD 网格，当前只能确认它是 **cell-centred 数组**。节点值、单元中心点值和有限体积单元平均不是同一个数学对象。

这轮没有训练 FNO、DeepONet 或 3D U-Net，而是先回答一个更基础的问题：

> cell-centred 场应该怎样接入直线 BOST forward，才能保留清楚的积分、边界、伴随和误差语义？

我们实现并比较了三条路线：

| 路线 | 做法 | 当前处置 |
|---|---|---|
| Native cell-centred | 在 cell centres 上求横向导数，沿 LOS 用 midpoint 权重 `h` 积分 | 进入独立 forward 验证，边界继续审计 |
| Cell-to-node composite | 先把 cell 值外推/平均到 nodes，再调用 Tier-A node operator，最后把 detector nodes 平均回 cells | **不能作为 truth forward**；只保留为离散敏感性反例 |
| Projection-first interior | 先沿 LOS 积分，再只在 detector 内部做中心差分，四周各裁一格 | 当前最稳妥的第一版经典基线候选 |

这里的 PASS 不是“某个算法效果好”，而是：

1. 三条实现的线性、单位、尺度和离散转置都能被复核；
2. 测试确实识别并拒绝了一条表面合理、但会暗中改动物理离散的适配路线；
3. unresolved 薄前缘被明确标成失败区，而不是包装成成功。

## 2. 三条路线的数学含义

设 LOS 为 `l`，两个横向方向为 `u,v`，cell-centred 场为 `f`，参考折射率为 `n_ref`。

### 2.1 原生 cell-centred

横向导数内部使用二阶中心差分，完整视场边界使用二阶单边差分：

```text
epsilon_u = (1 / n_ref) sum_l h_l D_u^c f
epsilon_v = (1 / n_ref) sum_l h_l D_v^c f
```

uniform cell centres 对应的物理 domain edges 取最外层中心再向外半个 spacing，因此 LOS 长度是 `N_l h_l`，不是 `(N_l-1)h_l`。

它的 Euclidean adjoint 必须使用：

```text
A^T y =
    (D_u^c)^T P_l^T y_u / n_ref
  + (D_v^c)^T P_l^T y_v / n_ref
```

边界存在时，`D^T` 绝不可以偷换成 `-D`。

### 2.2 cell-to-node composite

一维 cell-to-node 矩阵采用：

```text
左边界 node:  1.5 f_0 - 0.5 f_1
内部 node:    0.5 f_{j-1} + 0.5 f_j
右边界 node: -0.5 f_{N-2} + 1.5 f_{N-1}
```

随后调用 node-field Tier-A operator，detector 上再用相邻 nodes 的平均得到 cell-centred 输出。完整转置顺序必须反过来：

```text
detector Q^T -> node operator A_node^T -> cell-to-node T^T
```

这条路线可以把 dot test 做到机器精度，但这并不代表它是正确的 truth forward。

### 2.3 projection-first interior

先沿 LOS 做 midpoint projection：

```text
p(u,v) = sum_l h_l f(l,u,v)
```

然后只在 detector 内部 cell centres 上做中心差分：

```text
epsilon_u = M D_u^c p / n_ref
epsilon_v = M D_v^c p / n_ref
```

`M` 会裁掉横向四周各一格。它不假设网格外存在什么场，也不引入 cell-to-node 的端部外推。

代价是 detector ROI 变小，而且单视角 nullspace 依然很大。因此它只是第一版强基线候选，不是完整 BOST 相机模型。

## 3. 关键发现：cell-to-node 会暗中改变 forward

独立展开 composite matrix 后，cell-to-node 路线的 LOS 等效权重不是 midpoint 的全 1，而是：

```text
[1.25, 0.75, 1, ..., 1, 0.75, 1.25] * h
```

总权重仍然等于 `N h`，所以只检查“权重和”发现不了问题。但入口和出口附近的 cell 被分别增权或减权 25%。如果火焰前缘靠近 LOS 端部，观测会被系统性改变。

在非导数的 detector 方向，复合矩阵内部变成：

```text
[0.25, 0.50, 0.25]
```

这相当于额外做了一次低通滤波。边界却突然变成 identity，处理并不平移不变。

在导数方向的第一格，复合矩阵退化为：

```text
(f_1 - f_0) / h
```

也就是一阶前向差分。它能通过精确 adjoint dot test，却在解析收敛中显示出较低的全局阶数。这正是“错误 A 和对应的 A^T 可以一起通过转置测试”的实例。

## 4. 解析证据怎样设计

为了避免把 CFD block mean 当成点值，我们对每个解析场做了两套输入与 oracle：

1. `point_sample`：输入是解析函数在 cell centre 的点值，oracle 是 detector cell centre 的解析偏折；
2. `cell_average`：输入是每个三维 cell 的精确体积平均，oracle 是 detector cell 的精确面积平均偏折。

### 4.1 平滑周期场

使用一个 LOS envelope 与两个横向周期模式：

```text
Delta n =
  [1 + 0.2 cos(2 pi x / Lx)]
  [Ay sin(2 pi y / Ly) + Az cos(2 pi z / Lz)]
```

LOS envelope 的解析积分恰好是 `Lx`。网格使用每轴 `9 / 17 / 33 / 65` 个 cells。

### 4.2 火焰前缘代理

使用：

```text
Delta n =
  [1 + 0.2 cos(2 pi x / Lx)]
  A tanh((y-y0)/w) exp[-z^2/(2 sigma^2)]
```

`tanh` 用来模拟陡峭反应前缘。既做固定物理宽度的网格收敛，也在 `33^3` 网格上把 `w/h` 从 `0.5` 扫到 `8`。

`tanh` 的 10%-90% 厚度约为 `2.197 w`。因此：

- `w/h <= 1` 被强制标为 `UNRESOLVED_DO_NOT_USE_AS_SUCCESS`；
- `w/h = 2` 只算 marginal；
- `w/h >= 4` 才进入 resolved analytic case。

## 5. 核心结果

### 5.1 平滑 cell-average 场

| 路线 | 最低观测收敛阶 | 65³ relative-L2 | 处置 |
|---|---:|---:|---|
| Native cell-centred | `1.94697` | `0.00167277` | 通过二阶候选门 |
| Cell-to-node composite | `1.73380` | `0.00612963` | 拒绝作为 truth forward |
| Projection-first interior | `1.97240` | `0.00155661` | 通过二阶候选门 |

在最细平滑网格上，cell-to-node 的误差是 native 的 `3.664x`。

### 5.2 resolved 前缘

在 `33^3` 网格、`w/h=4` 时，10%-90% 前缘厚度约为 `8.79 cells`：

| 路线 | relative-L2 |
|---|---:|
| Native cell-centred | `0.0144489` |
| Cell-to-node composite | `0.0381709` |
| Projection-first interior | `0.0144570` |

cell-to-node 的误差是 native 的 `2.642x`，并且超过预注册的 2% resolved-case 门。

### 5.3 unresolved 前缘

在 `w/h=0.5` 时，native 的 cell-average relative-L2 仍为 `0.3505`。这不是“模型还需要调参”，而是当前网格根本没有解析这个前缘。

后续训练集必须保存每个样本的前缘厚度/网格 spacing 诊断。未解析样本不能被拿来证明某个 warm start 更准确；最多只能用于鲁棒性或 fail-closed 测试。

## 6. exact adjoint 不等于可辨识

三条路线都通过：

- 三个 LOS 方向；
- 常数 gauge；
- 线性符号与尺度；
- metre/mm 一致重参数化；
- `n_ref` 反比例缩放；
- 每条路线 36 个随机/角点 dot cases；
- composite 每一级转置；
- JVP/VJP 中心差分。

但 tiny `4^3` 单视角显式矩阵仍显示：

| 路线 | 矩阵 shape | rank | nullity |
|---|---:|---:|---:|
| Native | `32 x 64` | `15` | `49` |
| Cell-to-node | `32 x 64` | `15` | `49` |
| Interior | `8 x 64` | `8` | `56` |

所以：

> exact adjoint 只说明优化方向与声明的离散 forward 一致，不说明三维场能够被单视角唯一重建。

正式 C 路线仍必须使用多视角、统一 gauge/reference、强正则/预条件基线，并报告 nullspace 与不可观测分量。

## 7. 对 C 路线的直接影响

### 7.1 现在保留什么

第一版经典 inverse baseline 优先使用：

1. projection-first interior operator；
2. 明确 ROI 与 detector crop；
3. Zero / backprojection / CGLS / PCGLS；
4. 所有方法使用同一个 forward、adjoint、停止规则和 ROI；
5. warm start 只输出初值，再由相同 solver 到达匹配终点。

native full-detector 路线作为边界敏感性对照继续保留。

### 7.2 现在否掉什么

以下做法不能进入 truth label 生成：

- 把 cell-centred rho 直接 reshape 成 node field；
- 默认用线性 cell-to-node 外推，然后把 dot-test PASS 当物理正确；
- 在 underresolved 前缘上比较网络和 PCGLS 后宣称“网络恢复了真实细节”；
- 用同一离散 forward 同时生成观测和反演，再把自洽结果写成真实 BOST；
- 在 PoolFire 单位、cell-average 语义和 `rho -> Delta n` 未确认时启动正式 C0。

## 8. 仍需师兄确认的五件事

可以直接问：

1. PoolFire 或组内 CFD 数组是中心点采样，还是 finite-volume cell average？
2. 三个坐标的单位和真实 domain edges 是什么？当前 metadata 数值跨度与 README 描述有冲突。
3. `rho/T/Y_k` 到 `Delta n` 使用什么关系、波长和环境 reference？
4. 正式相机的 LOS、detector crop、像素映射与边界 ROI 怎样定义？
5. 经典重建是保留完整 detector 边界，还是允许统一裁一格后比较？

## 9. 下一道门

这轮之后，主线顺序是：

1. 师兄确认 cell semantics、单位、`rho -> Delta n` 和 ROI；
2. 把 axis-aligned LOS 升级为任意相机 ray direction、正交 detector basis、domain clipping 与独立 quadrature；
3. 用与 inverse 离散不同的 high-fidelity forward 生成观测；
4. 建立 Zero / BP / CGLS / PCGLS 强基线；
5. 只有基线稳定后才证伪 C0 Adjoint-Residual Warm Start；
6. 比较 matched final accuracy 下的 `A/A^T` 调用、端到端时间、内存、median/p90/worst/harm rate。

## 10. 复现

```bash
.venv/bin/python site_tools/run_poolfire_g0_cell_center_gate.py
.venv/bin/python site_tools/render_poolfire_g0_cell_center_figure.py \
  learning_labs/results/poolfire_g0_cell_center_gate_v0/result.json \
  assets/poolfire_g0_cell_center_gate_v0.png
.venv/bin/pytest -q \
  learning_labs/test_poolfire_g0_cell_center_operator.py \
  site_tools/test_run_poolfire_g0_cell_center_gate.py
```

主要文件：

- `learning_labs/poolfire_g0_cell_center_operator.py`
- `learning_labs/test_poolfire_g0_cell_center_operator.py`
- `site_tools/run_poolfire_g0_cell_center_gate.py`
- `site_tools/test_run_poolfire_g0_cell_center_gate.py`
- `learning_labs/results/poolfire_g0_cell_center_gate_v0/result.json`
- `assets/poolfire_g0_cell_center_gate_v0.png`

## 11. 证据边界

**关键发现，但不是算法突破。**

本轮真实发现了一条会通过 adjoint dot test、却会改动 LOS 权重、额外低通并降低边界阶数的适配路线，并用解析证据将它从 truth forward 中剔除。这减少了后续 inverse crime 和假加速的风险。

仍未证明：

- PoolFire 的物理单位与 cell-average 语义；
- 组分相关 Gladstone-Dale / `Delta n`；
- 任意相机、像素位移、背景图、光流或曲光线；
- 可靠多视角 BOST 观测；
- CGLS/PCGLS 重建；
- C0 warm start 提速；
- 跨轨迹、跨工况或跨几何泛化；
- 优于 FNO/DeepONet；
- 论文结果。
