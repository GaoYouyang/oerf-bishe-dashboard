# N2-PVGR N5-D4 四单元场导数门禁预注册

**状态：** 结果前预注册，未运行正式数值实验。
**唯一目的：** 在极小的 CPU float64 闭包上，判断现有曲光线 BOST 前向程序对三维栅格场的 JVP/VJP 是否自洽。
**不回答：** 真实相机是否准确、三维场能否重建、神经算子是否优于 DeepONet/FNO，或是否已得到可发表结果。

## 1. 为什么先做这一门

毕设主线的最终问题是从多视角 BOST 观测反演三维折射率/密度场，再学习从观测函数到场函数的算子。无论使用一阶优化、Gauss-Newton、隐式神经场还是神经算子，都需要同一个前向算子

\[
F(\theta)\in\mathbb{R}^{R\times 2}
\]

对场参数 \(\theta\) 给出可信的

\[
J(\theta)v,\qquad J(\theta)^Tq.
\]

如果轨迹对折射率场的依赖被静默 `detach`，或曲光线与直光线残差不是同一离散观测，后面即使 loss 下降也可能只是错误梯度的局部现象。因此 D4 是“实现证书”，不是新算法证书。

## 2. 冻结对象

### 2.1 四个单元

| 序号 | 单元 | 配对 | 角色 | 压力倍数 |
|---:|---|---|---|---:|
| 1 | `smooth-s1871-orientation_58-wide__stress_1` | p04 | N3 failure | 1 |
| 2 | `smooth-s1871-orientation_58-narrow__stress_1` | p04 | matched control | 1 |
| 3 | `smooth-s1871-orientation_58-wide__stress_3` | p05 | N3 failure | 3 |
| 4 | `smooth-s1871-orientation_58-narrow__stress_3` | p05 | matched control | 3 |

这四个单元全部共用 `smooth-s1871` 场和 58° 视角，只覆盖宽/窄孔径与 1/3 倍无量纲折射强度。它们是 N4 失败单元及其配对对照，不是独立抽样。D4 若通过，也只能证明这些选定上下文中的程序导数合同。

### 2.2 光线、网格与积分

- 网格：`17 x 17 x 17`，CPU float64。
- 光线：从 N4/D3 的 256 条公共 Sobol 序列取前 4 条，不重抽。
- 曲光线：显式 RK4，16 步，中点曲率积分。
- 空间梯度：冻结的中心差分，`difference_step=0.002`。
- 观测：每条光线两个探测器分量 `(u,v)`，合计 `[4,2]`。
- 单位：合成内部偏折量，尚未标定成 pixel 或 angle。

16 步不是物理收敛参考，而是导数程序的小型 smoke gate。收敛的 forward 参考属于 D3，两者不得混合。

## 3. 四个冻结闭包

1. `curved_detector`：完整场依赖的曲光线轨迹与中点积分。
2. `straight_detector`：冻结直路径上的 Born-style 中点积分。
3. `raw_curved_minus_straight`：先分别完成两个探测器输出，再相减。
4. `paired_neumaier_residual`：在同一组中点上先减弯/直积分被积函数，再做 Neumaier 补偿求和。

每个单元冻结 2 个场方向 \(v\) 与 2 个对应余切 \(q\)。方向 0 是 `3x3x3` 平均池化后的低频扰动，更近似物理场的平滑变化；方向 1 保留 raw Gaussian 高频扰动，用于压力测试参数空间导数。两者的外两层体素置零后做 L2 归一化。余切使用等幅 Rademacher 符号并做 L2 归一化，因此每条光线的 `u/v` 分量都是非零的，不允许用近零余切隐藏某一行。同一 `(cell,direction)` 的四个闭包必须共用完全相同的 \(v,q\)。一次性证明不只记录 RNG seed，还会把实际 field/ray/direction/cotangent 数组存成提交的 float64 little-endian NPZ，并对每个数组和整个归档做 SHA-256 绑定。正式 runner 消费这些精确字节，不依赖库版本下的 RNG 重放。

## 4. 三类互相独立的导数证据

### 4.1 JVP/VJP 内积对账

JVP 与 VJP 必须来自同一 tensor-to-tensor closure：

\[
e_{dot}=\frac{|\langle Jv,q\rangle-\langle v,J^Tq\rangle|}
{\max(|\langle Jv,q\rangle|,|\langle v,J^Tq\rangle|,10^{-30})}.
\]

每个闭包、单元和方向均需 `e_dot <= 1e-10`，且内积信号至少为 `1e-16`。信号过小时不能用巨大分母 floor 伪造“零误差”，而是 fail closed。

### 4.2 固定多步长中心有限差分

\[
D_hF(\theta)[v]=\frac{F(\theta+h v)-F(\theta-h v)}{2h},
\]

\[
e_{FD}(h)=\frac{\|D_hF-Jv\|_2}
{\max(\|D_hF\|_2,\|Jv\|_2,10^{-30})}.
\]

步长在结果前冻结为

`[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]`。

每个闭包必须同时满足：

- 最好的 `e_FD <= 1e-6`；
- 三个预先指定步长 `1e-2, 3e-3, 1e-3` 必须各自满足 `e_FD <= 1e-5`；
- 全部七个步长都必须 finite，且 `||F+ - F-||` 大于冻结的 `1024 eps64 output_scale` 信号底。

因此不能只挑一个偶然好看的 \(h\)。

### 4.3 结构恒等式

直接残差必须与两条路径的差一致：

\[
F_r=F_c-F_s,
\quad J_rv=J_cv-J_sv,
\quad J_r^Tq=J_c^Tq-J_s^Tq.
\]

同时对账 raw 与 paired-Neumaier 的 output/JVP/VJP。这一对账检查累加顺序是否改变了所声称的离散观测，不把更好消除舍入当成更好的物理模型。

`raw_curved_minus_straight` 的定义本身就是两闭包相减，因此这个恒等式只记为 wiring invariant，不重复计为一份独立物理导数证据。paired-Neumaier 则作为独立累加顺序对照。

## 5. 导数上下文不变门

对每个 \(\theta\pm hv\) 的曲光线轨迹重算：

- 支撑集进出次数；
- 合成视锥越界布尔标记；
- domain/stencil 最小余量；
- 方向向量范数误差。

不只比较“穿越了几次”。实现将按光线、RK4 `k1-k4`、曲/直中点与七个 central-stencil offset 保存有序的插值 cell ID、support bit 和 frustum-margin sign，再对序列哈希。有序签名与基点不同时，机器决策必须是 `D4_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`，不得换方向或换步长。

## 6. 查询与成本预算

设光线数 \(R=4\)，RK4 步数 \(S=16\)。一次闭包的逻辑场点查询为：

| 闭包 | 一次调用 |
|---|---:|
| curved detector | `35 R S` |
| straight detector | `7 R S` |
| raw residual | `42 R S` |
| paired residual | `42 R S` |

每个 map/direction 有 1 次 JVP、1 次 VJP 和 14 次有限差分 forward，共 16 次闭包。四单元乘两方向的 derivative-map 账本为 `1,032,192` 个逻辑场点查询。

每个 topology 签名包含 `28RS` 的诊断 RK4 重放和 `42RS` 的有序 stage/midpoint/stencil support 采样，合计 `70RS`。每个 cell/direction 需基点加 14 个扰动签名，八个方向上共 `537,600` 个逻辑查询。选定上下文预算为 `1,569,792`。另加一个高弯曲强度的 full-path vs detached-path JVP 安全对照，记 `3,360` 次逻辑查询，总账本为 `1,573,152`。JVP/VJP 即使没有额外 scalar query，也必须记为导数 sweep，不得写成零成本。当前 API 的 host sync 与 saved-tensor bytes 尚未仪器化，正式结果只能报 wall time 和 peak RSS，不得伪造显存/内存细分。

同一账本另按 interpolation dispatch 统计：derivative maps `175,744`、topology `61,680`、强轨迹对照 `686`，总计 `238,110` 次。field-point query 和 dispatch 是两种不同成本，不得相互替代。

## 7. 机器决策与开门边界

只有每一个 `(cell,direction,map)` 的 finite、dot、multi-h FD、repeat-output、domain/stencil 和结构恒等式全部通过，才能得到：

`D4_TINY_SELECTED_FIELD_DERIVATIVE_CONTRACT_VALID`

任一普通门失败：

`D4_TINY_FIELD_DERIVATIVE_FAIL_CLOSED`

任一扰动改变 topology 上下文：

`D4_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`

通过时只将 `tiny_selected_field_derivative_contract` 置为 true，并允许撰写一份新的、结果前预注册的 32-cell 导数扩展门。下列结论仍必须是 false：

- population-wide field JVP/VJP；
- 6+2 三维重建；
- DeepONet/FNO 训练或优越性；
- 真实 BOST 数据、泛化或可发表算法。

此外，D4 的 primitive 是直接的 `field[17,17,17] -> detector[4,2]`，尚未包含 `decoder(theta) -> field` 的神经参数化。即使通过，后续仍需独立验证 decoder 的链式 JVP/VJP，不能把网格场核心等同于神经算子已可训练。

正式 runner 先写入预注册的 `_work` 目录，所有结果与 manifest 完整后才原子重命名为正式目录。如果中途崩溃，`_work` 作为失败证据保留；不得删除后在同一预注册下重试，必须先建立恢复协议。

结果前证明和冻结 NPZ 也采用 fail-closed 发布：先在同目录排他创建隐藏临时文件，写完后 `flush + fsync`，再以排他硬链接原子发布完整字节，并同步目录。占用检查使用 `lexists`，因此悬空符号链接也视为已占用。任何正式文件、临时文件或并发同名文件已存在都会拒绝覆盖；若两件证明之间崩溃，已发布文件或临时文件保留为失败证据，同一协议不得静默重来。正式结果目录在开始和最终重命名前也重复检查 `lexists`；这降低误覆盖风险，但仍不声称抵御不遵守协议的外部进程在最后一条检查后的竞争写入。

独立 validator 不导入 D4 runner、D4 gate、冻结输入构造器、程序签名器，也不复用 N4/N0 runner 的单元展开、种子或 rig helper。它保留一份独立实现，并从提交的上游配置重新生成 32 单元后再选定四个上下文；同时固定复核 `p01` 至 `p16` 的顺序、同 field-unit/同 stress、恰好一个几何因素变化，以及失败门只能来自 `matched_residual_convergence_gate_met` 或 `output_convergence_gate_met`。这是在实现层面降低共同错误，不代表独立软件栈或独立研究者复现。

validator 第一次成功后会排他、原子写入确定性的 `validation_report.json`；它是验证后副产物，故明确排除在 runner 预先生成的 manifest 文件哈希集合之外。manifest 必须声明这一排除，validator 会复核该声明；相同字节可幂等复验，不同字节、符号链接或残留 staging 文件一律拒绝覆盖。

## 8. 与一级文献的关系

- He et al. 的 NeRIF 将神经隐式折射率场与 BOST 体重建直接连接，是本毕设的最近实验学语境：<https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the>
- Zhao et al., CVPR 2024 展示了单视图折射率层析与 neural field 表示，但不代替本项目的多视角 BOST 观测合同：<https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html>
- Adjoint Nonlinear Ray Tracing 是曲光线反问题中伴随方法与梯度验证的直接参考：<https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/>
- Pearlmutter 给出了不显式构造 Hessian/Jacobian 的向量积思路，是后续 matrix-free Gauss-Newton 的基础：<https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf>

上述文献说明“可微曲光线 + 神经场/反演”有充分先例。D4 不声称新颖性；它的价值是为后续新模型建立一个不能因错梯度而假胜的最小证据门。
