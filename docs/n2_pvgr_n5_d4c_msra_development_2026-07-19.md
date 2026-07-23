# N5-D4c-v1：混合尺度随机伴随开发屏（语义主张已撤回）

> **2026-07-19 红队更正：本页 v1 只能作为负证据。** v1 的文件、哈希、行数和既定
> 布尔逻辑通过了完整性 validator，但它没有实际调用 `F(x+h v)` 与 `F(x-h v)`；所谓
> `fd_relative_error` 是候选 JVP 与参考 JVP 的差。hard-branch 布尔值由场景直接注入，
> structure 指标也使用了隐藏的正确矩阵，而不是独立 curved/straight/direct 三路径。
> 因此“独立 FD”“真实分支检出”“三路径结构检出”三个语义主张全部撤回。
>
> v1 的 `74.72%` 还混合了不同场景、故障强度和任务角色，不具有算法性能含义，不再用于
> 任何阈值、候选排序或论文论证。后续请以
> [D4c semantic-v2 审计](n2_pvgr_n5_d4c_msra_semantic_v2_2026-07-19.md)为准。

> 证据等级：预注册后的 synthetic fault-injection development。
>
> 机器判决：`D4C_MSRA_DEVELOPMENT_CHARACTERIZATION_ONLY_NO_AUTHORIZATION`
>
> 本页不是 D4b 改判，不是 fresh BOST 导数通过，也不是三维重建或神经算子结果。

## v1 历史输出：哪些还能保留，哪些必须撤回

这一轮得到的不是一个已经可用的新阈值，而是一份更清楚的算法边界。

1. **可保留：低双线性信号会让 signal-relative 指标失真。** explicit-matrix toy 中，24/24 个正确近正交样本被旧 relative-dot 门拒绝；这是反例，不是 BOST 导数通过。
2. **可保留：一个随机 tangent 有可构造的 VJP 盲区。** 与首个 tangent 正交的错误在一个 probe 下不可见；v1 的多 probe 数字仅描述其冻结 toy。
3. **可保留：彼此转置不等于对真实 forward 正确。** 同一错误矩阵生成的 JVP/VJP 会通过 adjoint identity；但 v1 没有用实际 forward 中心差分证明后续检出率。
4. **仅作启发：separate 与 paired arithmetic 的 toy 趋势。** v1 的所谓 FD 是参考导数差，不能写成实际 `F(x±hv)` 证据。
5. **必须撤回：真实 branch/structure 检出。** v1 使用人工 branch 标签和隐藏矩阵 oracle，只证明预设布尔逻辑按代码执行。
6. **必须撤回：74.72% 总分类率。** 它跨任务混合且没有科学目标函数；v2 禁止这种 pooled accuracy，也继续禁止选阈值。

## 1. 真实物理困难是什么

BOST 不是直接测三维折射率。相机看到背景图案的位移，位移与光线沿途的折射率梯度及相机投影有关。NeRIF 进一步把折射率场写成连续神经表示，再沿 ray 采样、积分并与位移观测比较。

当前 synthetic curved-ray forward 中，有两类容易混在一起的问题：

- **物理 residual 小：**弯曲光路与直线路径的两个 component 都可能不小，但它们的差很小。D4b 的 p14 中，component contraction signal 比 residual signal 大约 `14,466.67` 倍。
- **数值实现也在相减：**如果先分别计算大 component，再在有限精度中相减，真实小 residual 会混入 component 级舍入误差。

这会直接影响后续三维重建和算子学习：

1. inverse solver 依赖 `F` 与 `F^T` 或 nonlinear `Jv/J^Tq`；
2. 错误梯度可能让单场 NeRIF 优化得到看似下降、实际不对应 forward 的结果；
3. 用这些结果生成 neural-operator 训练对，会把数值错误固化为监督信号；
4. 因此先证明导数接口的检测能力，比先换 DeepONet、FNO 或 FFNO 更靠近真实瓶颈。

最近的 OERF 物理语境仍是 [He et al. NeRIF 正式 DOI](https://doi.org/10.1063/5.0250899) 与[公开作者稿](https://arxiv.org/html/2409.14722v2)。本轮没有使用其未公开数据，也没有声称复现论文实验。

## 2. 新指标到底算什么

令：

\[
x=Jv,\qquad y=J^Tq,
\]

\[
a=\langle x,q\rangle,\qquad b=\langle v,y\rangle,
\qquad e=|a-b|.
\]

### 2.1 旧 signal-relative 指标

\[
\rho_{\mathrm{signal}}
=\frac{e}{\max(|a|,|b|,s_0)}.
\]

它回答的是：两个标量 contraction 相对当前标量 signal 是否一致。若 `a` 与 `b` 本来接近 0，正确算子的舍入误差也会被小分母放大。

### 2.2 normwise 指标

\[
D_{\mathrm{action}}=
\max\left(\|Jv\|_2\|q\|_2,
\|v\|_2\|J^Tq\|_2\right),
\]

\[
\rho_{\mathrm{norm}}=
\frac{e}{\max(D_{\mathrm{action}},s_0)}.
\]

它把误差与导数作用量比较，不会因为一个近正交 bilinear probe 自动爆炸。

### 2.3 gamma-scaled score

binary64 的 unit roundoff 为 `u=2^-53`。长度为 `n` 的标准 dot-product 浮点模型常写成：

\[
\gamma_n=\frac{nu}{1-nu}.
\]

本轮报告：

\[
S_\gamma=
\frac{\rho_{\mathrm{norm}}}
{\max(\gamma_{n_{in}},\gamma_{n_{out}})}.
\]

同时保留 contraction-only envelope：

\[
E_{fp}=\gamma_{n_{out}}\sum_i |x_iq_i|
+\gamma_{n_{in}}\sum_j |v_jy_j|.
\]

必须注意：`gamma_n` 只直接约束最终 dot products。ray marching、插值、JVP/VJP transform 和残差构造的误差传播没有被这个式子完整证明。因此 `S_gamma` 是候选诊断，不是现成的整程序定理。数值误差背景可读 [Higham, Accuracy and Stability of Numerical Algorithms](https://epubs.siam.org/doi/book/10.1137/1.9780898718027)，自动微分与导数验证可读 [Griewank and Walther, Evaluating Derivatives](https://epubs.siam.org/doi/book/10.1137/1.9780898717761)。

## 3. 为什么一定需要多个 probe

单个 dot test 只检查一个标量：

\[
\langle Jv,q\rangle=\langle v,J^Tq\rangle.
\]

若错误向量 `e=\widetilde{J^Tq}-J^Tq` 恰好满足：

\[
\langle v,e\rangle=0,
\]

那么错误 VJP 仍会通过这个 probe。输入空间是 `17^3=4913` 维，一个 tangent 只看一个方向。

本轮把 probe 数冻结为 `1/2/4/8/16`。对每个固定 cotangent，复用一次 VJP，再增加独立 tangent 与 JVP。结果说明：

| VJP blind fault | 1 probe | 2 probes | 4 probes | 8 probes | 16 probes |
|---|---:|---:|---:|---:|---:|
| `1e-12` | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 |
| `1e-10` | 0/24 | 12/24 | 20/24 | 24/24 | 24/24 |
| `1e-8` | 0/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| `1e-6` | 0/24 | 24/24 | 24/24 | 24/24 | 24/24 |

表中采用 gamma threshold `2` 只是固定网格中的一个剖面，**不是本轮选出的阈值**。

## 4. 预注册与实验矩阵

规则在结果前提交为 commit `38f091f`：

- CPU float64；
- 输入维数 `4913`，输出维数 `8`；
- 24 个固定 trial；
- 5 个 probe counts；
- 10 个 gamma thresholds，完整报告，不选 winner；
- fault magnitudes：`1e-12/1e-10/1e-8/1e-6`；
- cancellation scales：`1e-8/1e-6/1e-4`；
- FD 与 structure thresholds 均为 `1e-8`；
- 11 类 scenario；
- D4b 只作 post-open 描述，禁止阈值选择或改判。

最终产生：

- `3,600` 条 synthetic base rows；
- `36,000` 条 threshold/probe gate evaluations；
- 10,483,847-byte 完整 gate CSV；
- 预注册图表、result、summary、manifest；
- 独立 validator 从 CSV 重算全部门并确认 `valid=true`。

## 5. 11 类反例各自在防什么

| Scenario | 为什么需要 | 哪一门应主要负责 |
|---|---|---|
| clean general | 正确普通线性算子 | 所有门通过 |
| clean low bilinear signal | 正确但 scalar signal 近零 | mixed-scale 防止旧门假拒绝，同时状态应保守 |
| separate cancellation residual | 大 component 分别计算再相减 | gamma/FD 暴露数值不稳定 |
| paired cancellation residual | residual 先形成再作用 | 应比 separate 更稳，但仍检查 residual 构造精度 |
| VJP aligned fault | 错误沿首 tangent 可见 | adjoint gate |
| VJP first-probe blind fault | 单方向正交盲区 | 多 probe |
| JVP aligned fault | forward derivative 错 | adjoint + FD |
| self-consistent wrong derivative | 错误 JVP/VJP 彼此转置 | FD/oracle，adjoint无法负责 |
| structure mismatch | direct residual 与 components 不一致 | structure gate |
| diagnostic-only support flip | 签名变化但 forward 不变 | 只报告，不误叫 hard branch |
| hard branch crossing | occupancy 分支真实改变 forward | branch + FD fail-closed |

## 6. 关键结果，不只报好看的部分

### 6.1 旧门的假拒绝被明确复现

- low-signal clean：传统门 `0/24` 通过；
- 同一批 `S_gamma` 最大值：`1.3994707864e-4`；
- 说明 scalar relative defect 大，不足以证明 VJP 错。

但正确动作不是“normwise 小就自动 PASS”。更严谨的三态应是：

1. `PASS_STRONG_SIGNAL`：signal-relative、normwise、多 probe、FD、structure、branch 全部通过；
2. `LOW_SIGNAL_UNRESOLVED`：relative 失败而 normwise 小，只能进入额外 probe/FD/oracle；
3. `FAIL`：任一真实错误门触发。

### 6.2 多 probe 有用，但有检测地板

在 threshold `2` 的说明剖面中：

- aligned JVP/VJP fault：`1e-10` 及以上 24/24 检出；
- blind VJP fault：8 probes 才在 `1e-10` 达到 24/24；
- `1e-12` blind fault 即使 16 probes 仍 0/24 检出。

因此不能写“多 probe 证明了完整 VJP”。它只把盲区从确定性单方向问题变成可量化的概率覆盖问题。

### 6.3 adjoint identity 不能抓自洽错误

self-consistent wrong derivative 中，JVP 与 VJP 来自同一错误矩阵，因而 adjoint identity 仍成立：

- `1e-12/1e-10`：当前 FD threshold 下未检出；
- `1e-8/1e-6`：FD 24/24 检出。

这是本轮最重要的反例之一：任何只展示 dot test 的论文，都没有单独证明导数相对真实 forward 正确。

### 6.4 paired residual 不是一句“换求和器”

paired residual 的 gamma score 三个尺度都很低，说明 JVP/VJP 内部闭合；但 `delta=1e-8` 时 FD 仍 0/24 通过，最大 FD relative error 约 `3.07e-8`。

原因是：先在 float64 中构造 `C=A+delta B/2` 和 `S=A-delta B/2`，再形成 `C-S`，小 residual 已经在矩阵形成阶段受损。真正值得迁移到 BOST 的实现是：

- 在同一 ray sample、同一 interpolation query、同一投影基上形成 curved/straight integrand difference；
- 再做 paired或补偿累加；
- 不能先得到两个完整大 detector maps 再相减。

这与当前 `cancellation_aware_residual.py` 的 paired-integrand 方向一致，但仍需在 fresh BOST fields 上验证。

### 6.5 最高分类率仍不够

预注册网格中最高的总体正确分类率是 `74.72%`：

- clean acceptance：`83.33%`；
- fault detection：`72.57%`；
- gamma threshold：`0.25`；
- probes：`4/8/16` 均出现同一总体值。

这不是“最佳模型”。因为：

1. 配置明确禁止选阈值；
2. 这里混合了 `1e-12` 到 `1e-6` 的不同检测难度；
3. synthetic explicit matrix 不等于 BOST renderer；
4. 83.33% clean acceptance 本身不能作为 field derivative authorization。

## 7. Forward Branch Ledger

| 事件 | 当前 renderer 是否用于 forward | D4c 语义 |
|---|---:|---|
| smoothstep interpolation cell | 是 | 改变地址选择；attested smoothstep 通常 C1、非一般 C2，cell change 需 FD 报告，不自动证明一阶导数不存在 |
| support threshold bit | 否 | diagnostic-only，report-only |
| frustum sign | 否 | diagnostic-only，report-only |
| domain/stencil exception | 是 | base 或任一扰动有效性变化即 fail-closed |
| refractive-index validity | 是 | 有效性变化即 fail-closed |
| hard mask/occupancy/termination | 当前不存在 | 若实验室 renderer 存在，active set/termination/sample count 变化先 fail-closed |

`h(t)=3t^2-2t^3` 在 cell 两端的一阶导数为 0，因此相邻共节点的 smoothstep interpolation 通常 C1；二阶导数一般不连续。D4b 的 exact-bit gate 是历史预注册的保守合同，不能事后删除；D4c 则必须先按真实 forward 语义重新冻结。

真实 hard mask 或可见性边界不能仅靠 naive autodiff 处理。相关一级来源包括 [Bangaru et al., Systematically differentiating parametric discontinuities](https://dl.acm.org/doi/10.1145/3450626.3459775) 与 [Li et al., Differentiable Monte Carlo ray tracing through edge sampling](https://dl.acm.org/doi/10.1145/3272127.3275109)。它们只说明不连续边界需要专门处理，不证明当前 BOST renderer 已经存在同类机制。

## 8. 目前最值得保留的算法候选

暂名：**Residual-native MSRA certificate**。

它不是一个神经网络层，而是进入三维 inverse/operator learning 前的导数可信度接口：

1. **Residual-native forward：**在共同 sample/query 上形成 curved-straight residual，再累计；
2. **Mixed-scale report：**同时保留 signal-relative、normwise、gamma-scaled、absolute envelope 和 cancellation proxy；
3. **Randomized coverage：**固定 cotangent 下使用预注册的多 tangent probes；
4. **Independent primal check：**多个固定 `h` 的 FD 只检查 JVP，不用 dot identity 替代；
5. **Structural check：**direct residual 必须与 component difference 在 output/JVP/VJP 三层一致；
6. **Branch-semantic fail-closed：**只对真实改变 forward 的事件硬拒绝，diagnostic-only bit 只报告；
7. **三态输出：**strong pass、low-signal unresolved、fail，不把 low-signal 自动改成成功。

真正可能形成论文贡献的条件是：

- 在 fresh BOST field/rig population 上减少旧 relative gate 的 false reject；
- 同时保持对已知 VJP/JVP/structure/hard-branch fault 的检出；
- 证书状态能预测真实 inverse optimizer 的梯度检查、收敛稳定性或 held-out reprojection；
- 在相同 `F/Jv/J^Tq` 调用预算下，附带成本可接受；
- 最终迁移到何远哲实际 renderer 和 decoder chain，而不是只停在显式矩阵 toy。

## 9. 下一 fresh BOST 开门前要冻结什么

下一步不是直接把 threshold `2` 写进正式协议。建议先做独立 BOST development，再开 untouched audit：

### Development-BOST

- 新 phantom seeds、rig orientations 和 stress，不复用 p14；
- 至少 8 个 field/rig units，避免 32 cells 共享少数 field 的伪重复；
- 每个 context 预注册 8 tangents、2 cotangents；
- residual-native 与 separate residual 并排；
- 固定 `h` 网格，不按单个 case 选最佳 `h`；
- 注入 aligned、blind、self-consistent 和 branch faults；
- 记录逐 field/rig tail，而不是 pooled mean；
- 同时记录 JVP/VJP/FD 调用数与 wall time。

### Untouched audit

- development 结束后才冻结一个阈值与 probe budget；
- 新 fields/rigs 只开一次；
- 每个 context 必须过，不能平均；
- low-signal unresolved 单列，不得记成 pass；
- 只有 derivative interface 过门，才允许 decoder-chain；
- decoder-chain 过门后，才允许 6+2 view inverse；
- 三维重建过门后，才允许与 DeepONet/FNO/FFNO 同预算比较。

## 10. 一周补课路线

### Day 1：伴随恒等式

- 手算 `J=I_2` 的 `<Jv,q>=<v,J^Tq>`；
- 构造一个错误 VJP，但让误差与 `v` 正交；
- 产出：解释为什么单 probe 不充分。

### Day 2：浮点与 gamma_n

- 读 Higham 的 floating-point model 与 dot product error；
- 用 `n=8` 和 `n=4913` 手算 `gamma_n`；
- 产出：区分 contraction error 与整条 autodiff program error。

### Day 3：相消与 residual-native arithmetic

- 跑 `separate` 和 `paired` matrix toy；
- 把 `delta` 从 `1e-4` 改到 `1e-8`；
- 产出：一张误差随 component difference scale 的图。

### Day 4：随机 probe

- 给固定 VJP fault 做 1/2/4/8/16 tangent 检测；
- 产出：检出率与 JVP 调用数曲线。

### Day 5：FD 与自洽错误

- 构造同一错误矩阵产生 JVP/VJP；
- 观察所有 adjoint tests 通过、FD 失败；
- 产出：说明每一门的不可替代性。

### Day 6：branch semantics

- 在源码中标出 support、frustum、domain exception、refractive-index guard；
- 产出：一张 forward-control-flow ledger。

### Day 7：给师兄讲明白

- 用三句话说清问题、负结果与需要的数据；
- 不说“算法已经成功”，说“检测框架找到了旧门的假拒绝与自身盲区”。

## 11. 现在问何远哲最有价值的问题

1. 真实 renderer 是否存在 hard mask、occupancy pruning、invalid-ray removal、dynamic sample count 或 ray termination？
2. curved 与 straight contribution 是在 sample/integrand 层成对形成，还是先形成两个 detector maps 再相减？
3. 训练或优化实际使用 float32、float64 还是 mixed precision？
4. 能否给一个匿名最小 case：field/decoder parameters、4 rays、`F`、2 个 `Jv`、1 个 `J^Tq`？
5. decoder 是 voxel、MLP、hash encoding 还是 tensor decomposition？参数到 field 的 JVP/VJP 接口在哪里？
6. 实际可接受的 gradient-check 误差与 optimizer noise floor 是多少？
7. 能否批量计算多个 JVP，或用 `vmap/jacfwd` 降低 8-probe 成本？
8. 有无 flow-off repeats、位移单位、mask/confidence、相机 calibration uncertainty 可用于 real-data gate？

## 12. 可复现入口

```bash
cd /path/to/oerf-bishe-dashboard

# 预注册 commit
git show --stat 38f091f

# 指标和 runner 测试
.venv/bin/python -m pytest \
  demo_t16_operator/test_mixed_scale_adjoint_certificate.py \
  demo_t16_operator/test_run_n2_pvgr_n5_d4c_msra_development.py -q

# 独立验证器测试
.venv/bin/python -m pytest \
  site_tools/test_validate_n2_pvgr_n5_d4c_msra_development.py -q

# 正式结果只可在新输出路径复现；现有正式目录拒绝覆盖
.venv/bin/python demo_t16_operator/run_n2_pvgr_n5_d4c_msra_development.py
```

结果目录：

```text
demo_t16_operator/results/n2_pvgr_n5_d4c_msra_development_v1/
```

独立验证：

```text
valid = true
synthetic_base_rows = 3600
threshold_probe_rows = 36000
threshold_selected = null
historical_d4b_decision_retained = true
```

## 最终边界

本轮可以说：

- 旧 relative-dot 门会在正确低信号情形产生 24/24 假拒绝；
- 多 probe 能减少单方向 VJP 盲区，并量化检测成本；
- adjoint identity 无法发现自洽错误，FD 与 structure 门不可删；
- 当前 renderer 的 support/frustum 是 diagnostic-only；
- paired residual 必须进入共同积分原语，而不只是最后换一个求和器。

本轮不能说：

- threshold `2` 已被选中；
- D4b 已改判；
- field derivative interface 已通过；
- NeRIF 或 TDBOST 已被改进；
- 三维重建、真实数据或泛化成功；
- 新方法优于 DeepONet、FNO、FFNO；
- 已经形成可投稿的算法结论。

下一步要把这份 fault-injection 证据迁移到 fresh BOST fields/rigs，并让证书状态真正预测 inverse optimization 的可靠性。
