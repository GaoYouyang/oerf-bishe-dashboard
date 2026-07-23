# N5-D4c semantic-v2：真实前向调用的导数证书语义审计

> 证据等级：预注册后的 **synthetic explicit-matrix semantic characterization**。
>
> 协议提交：`09a50d1f2eb7cd187fc5e016b848dd882971059f`
>
> 机器判决：`D4C_MSRA_SEMANTIC_V2_CHARACTERIZED_NO_BOST_AUTHORIZATION`
>
> 独立重放：`valid=true`。验证器未导入 runner 或 certificate helper，从 seed 独立重建全部
> case、实际 forward、输入哈希、三路径、指标和状态。它仍使用同一 Python/NumPy 数值栈，
> 不是跨语言外部复现。

## 先讲人话

D4c-v1 最大的问题不是公式算错，而是“考试题”和“参考答案”来自同一套替代逻辑：它没有真的
调用 `F(x+h v)`、`F(x-h v)`，branch 是人工标签，structure 又偷看了隐藏正确矩阵。文件可以
完整、哈希可以一致，但语义仍然不成立。

v2 把这三件事改成了可追溯的实际操作：

1. 对每个 case、每个 tangent、每个 `h` 都真正调用两次 forward；
2. forward 自己返回 branch state 与 diagnostic state，不接受外部注入的 `changed=true`；
3. curved、straight、direct 三条路径分别计算 output、JVP 和 VJP，再检查
   `direct - (curved - straight)`；
4. 输出不再只有 pass/fail，而是 `PASS_STRONG_SIGNAL`、`LOW_SIGNAL_UNRESOLVED` 或带首个失败义务的
   `FAIL_*`；
5. 禁止把不同场景和故障强度混成一个“总体准确率”，也禁止从 development 网格选最好阈值。

这仍然不是新 BOST 算法，更不是论文成功。它是一套在进入三维重建和算子学习前，检查
forward/JVP/VJP 是否真的描述同一个离散程序的实验规程。

## 1. 冻结了什么

| 项目 | 冻结值 |
|---|---:|
| trials | 24 |
| 输入维数 | 4913，即 `17^3` |
| 输出维数 | 8 |
| tangent probes | 1 / 2 / 4 / 8 / 16 |
| 中心差分步长 | `1e-4` / `1e-3` / `1e-2` |
| side-weighted 描述阈值 | 0.25 到 128，共 10 档 |
| 故障相对强度 | `1e-12` / `1e-10` / `1e-8` / `1e-6` |
| cancellation scale | `1e-8` / `1e-6` / `1e-4` |
| FD 门 | `1e-9`，所有 `h` 与 probe prefix 取最坏值 |
| structure 门 | `1e-9` |
| threshold selection | 禁止 |

正式运行保存：

- 720 个唯一 case；
- 11,520 条逐 probe 伴随证据；
- 34,560 组实际 `F(x+h v)` / `F(x-h v)`；
- 1,536 条三路径逐 probe 结构证据；
- 3,600 条 probe-prefix evidence；
- 36,000 条按 scenario、magnitude、probe count、threshold 分层的状态记录。

每个 FD row 还保存 base/plus/minus 输入 SHA-256、`h.float_hex()`、8 维原始输出、候选 JVP、
中心差分估计，以及 forward 返回的 branch/diagnostic state。reported error 不是 validator 的真值；
验证器必须独立重建输入和 forward 后再计算。

## 2. side-weighted 指标是什么

令候选 JVP 为 `Jv`、候选 VJP 为 `J^T q`。伴随缺陷是

\[
e=|\langle Jv,q\rangle-\langle v,J^Tq\rangle|.
\]

v1 用一个共享的最大 `gamma_n` 乘单一 normwise scale。v2 把两个浮点收缩的维数分开：

\[
s_\gamma=gamma_{8}\|Jv\|\|q\|
 +\gamma_{4913}\|v\|\|J^Tq\|,
\qquad
r_\gamma=\frac{e}{s_\gamma}.
\]

这只是 **已算出 JVP/VJP 后的 dot-contraction 诊断**。它没有覆盖 renderer 内部插值、ray marching、
矩阵向量积或自动微分程序的全部舍入误差，所以本轮绝不从 `r_gamma` 反推一个“数学保证阈值”。

## 3. runner 报告了什么

### 3.1 低双线性信号必须叫 unresolved

24/24 个正确 explicit-matrix case 被旧 `1e-10` signal-relative 门拒绝；side-weighted score 最大为
`4.5541e-5`。v2 没有把它们洗成 pass，而是在其他义务通过时标为
`LOW_SIGNAL_UNRESOLVED`。

原因很简单：两个接近零的 scalar dot 很容易让相对误差爆大；但 normwise 小也不能证明 4913 维
VJP 每个方向都正确。正确动作是增加独立 probes、FD 与结构检查，而不是只换分母。

### 3.2 多 probe 缩小盲区，但没有完整证明

下面只固定描述网格中的 threshold `2`，**不是选定阈值**。对于与首个 tangent 严格正交的 VJP
故障：

| 相对故障 | 1 probe | 2 probes | 4 probes | 8 probes | 16 probes |
|---:|---:|---:|---:|---:|---:|
| `1e-12` | 0/24 | 0/24 | 0/24 | 0/24 | 0/24 |
| `1e-10` | 0/24 | 7/24 | 19/24 | 22/24 | 24/24 |
| `1e-8` | 0/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| `1e-6` | 0/24 | 24/24 | 24/24 | 24/24 | 24/24 |

这张表支持“多个随机方向比一个方向更难被构造性躲过”，不支持“16 probes 证明完整 VJP 正确”。
真实 renderer 还要同时改变 cotangent，并把每个 `Jv/J^Tq` 的实际成本纳入预算。

### 3.3 adjoint identity 抓不到彼此自洽的错误

`self_consistent_wrong_derivative` 用同一错误线性算子生成 JVP 与 VJP，因此它们严格互为转置。
actual-forward FD 的 16-probe 最坏误差中位数为：

| 相对故障 | FD error 中位数 | FD 门结果 |
|---:|---:|---:|
| `1e-12` | `1.15e-11` | 0/24 拒绝 |
| `1e-10` | `2.52e-10` | 0/24 拒绝 |
| `1e-8` | `2.23e-8` | 24/24 拒绝 |
| `1e-6` | `2.63e-6` | 24/24 拒绝 |

结论不是 FD 万能，而是每种门有明确责任：adjoint 检查 JVP/VJP 是否彼此转置，FD 检查 JVP 是否
符合实际 forward。弱于当前噪声地板的错误仍会漏掉，必须诚实报告分辨率。

### 3.4 direct FD 自洽也可能破坏三路径恒等式

`three_path_structure_mismatch` 的 direct path 自己是完全线性的，所以 actual FD 误差始终约
`1e-11`；但它故意不等于 `curved - straight`。三路径结构误差的中位数为：

| 注入强度 | structure error 中位数 | structure 门拒绝 | direct FD 拒绝 |
|---:|---:|---:|---:|
| `1e-12` | `2.41e-12` | 0/24 | 0/24 |
| `1e-10` | `2.55e-10` | 0/24 | 0/24 |
| `1e-8` | `2.98e-8` | 24/24 | 0/24 |
| `1e-6` | `2.36e-6` | 24/24 | 0/24 |

这正是 residual BOST 实现需要单独结构门的理由：一个 direct residual operator 可以对自己
可微，却并不等于代码中两条 component path 的差。

### 3.5 先算两个大量再相减，会污染实际 FD

separate path 先计算 `C(x)` 和 `S(x)`，再在 float64 中做差；paired path 直接计算冻结的
residual primitive。16 probes、所有 `h` 取最坏值后的 FD error 中位数：

| component difference scale | separate | paired |
|---:|---:|---:|
| `1e-8` | `1.58e-3` | `1.03e-11` |
| `1e-6` | `1.46e-5` | `1.14e-11` |
| `1e-4` | `1.46e-7` | `1.15e-11` |

三档 separate cases 全部被 `1e-9` FD 门拒绝，三档 paired cases 全部通过。这里的价值是一个
数值机制反例：如果 curved/straight contribution 很大、差很小，必须尽可能在同一 ray sample、
同一 interpolation query 和同一投影基上先形成 residual，再累计。

它仍不证明真实 OERF renderer 必然存在同样的尺度比；需要师兄提供匿名最小路径才能测量。

### 3.6 diagnostic 与真正 branch 已经分账

- `diagnostic_only_support_flip`：24/24 diagnostic state 发生变化，0/24 branch state 变化，
  forward 仍是同一个线性映射；
- `actual_piecewise_branch_crossing`：24/24 plus/minus 调用返回不同 branch state，24/24 actual FD
  也超过门，优先判 `FAIL_BRANCH`。

因此 support/frustum 统计量如果不参与 forward，就不能冒充“数学不可微”；反过来，如果真实
renderer 有 hard mask、occupancy pruning、dynamic sample count 或 ray termination，就必须记录
实际 active state，并在 crossing 时 fail-closed 或使用专门的边界导数。

## 4. 怎样读 `PASS_STRONG_SIGNAL`

这个名字只表示“当前这一组有限义务在强 dot signal 下没有失败”，不表示候选导数已被证明正确。
最直接的反例是：`1e-10` self-consistent wrong derivative 和 `1e-10` three-path mismatch 低于冻结门，
可能仍得到 `PASS_STRONG_SIGNAL`。这不是程序 bug，而是检测分辨率的公开边界。

所以论文或网页不得写：

- “v2 证明了导数正确”；
- “side-weighted 指标优于现有算法”；
- “16 probes 足以验证 4913 维 VJP”；
- “在 BOST/NeRIF 上成功”；
- “已经可以训练 DeepONet/FNO/FFNO”。

目前可以写的只有：v2 在 explicit-matrix fault injection 上，修复了 v1 的三类语义漏洞，并量化了
各门的互补责任和检测地板。

## 5. 与何远哲 / NeRIF 主线怎样接上

NeRIF 用连续神经场表示折射率，再通过光线传播与位移观测优化场参数。无论最终使用普通 MLP、
hash encoding、DeepONet 或 FNO，训练都依赖一个真实 forward 与它的 JVP/VJP。若梯度只在
dot identity 内自洽，却不对应实际 ray renderer，那么：

1. 单场优化的 loss 可以下降，但物理 forward 不一致；
2. 生成的“重建真值”会把数值错误带进 neural-operator 监督对；
3. operator learner 看似拟合得很好，迁移到真实 rig 时会出现不可解释的系统偏差；
4. 继续扩大网络或 GPU 规模只会放大错误合同。

D4c-v2 因此更像 **训练数据与梯度接口的质检层**。只有它接入真实 renderer 后确实抓到一种
BOST-specific failure，而且 residual-native 修复在同预算下改善三维重建，才可能上升为论文方法的一部分。

### 5.1 四条最相关的一级来源

1. [Higham, *Accuracy and Stability of Numerical Algorithms*](https://doi.org/10.1137/1.9780898718027)：
   先读第 2 章浮点算术与误差模型。它解释 `gamma_n` 从哪里来，也提醒我们误差界必须对应实际
   计算路径，不能把一个 dot-product 界外推到整个 renderer。
2. [Griewank & Walther, *Evaluating Derivatives*](https://doi.org/10.1137/1.9780898717761)：
   先读第 2、3、14 章。重点是 forward/reverse 的程序语义、seed directions，以及控制流或
   非光滑程序不能只按普通 straight-line AD 理解。
3. [Bangaru et al., *Systematically Differentiating Parametric Discontinuities*](https://doi.org/10.1145/3450626.3459775)：
   参数化不连续需要显式积分语义与边界项；这给真实 renderer 中 hard mask/occupancy/termination
   的后续处理提供方法背景，但不意味着可以直接套到 BOST。
4. [Li et al., *Differentiable Monte Carlo Ray Tracing through Edge Sampling*](https://doi.org/10.1145/3272127.3275109)：
   可微光线追踪中的可见性边界会产生传统采样漏掉的分布项。对我们最有用的不是照搬 edge
   sampling，而是理解“branch crossing 不能只靠局部 AD 假装不存在”。

OERF 物理宿主仍是 [He et al., Neural Refractive Index Field](https://doi.org/10.1063/5.0250899)。
上面四条解决的是数值与程序语义前置问题，不能替代 NeRIF/BOST 的真实位移、光线、密度场和
重建证据。

## 6. 下一步：D4d 真实 renderer adapter

在没有组内数据时，不再扩大 explicit-matrix toy。下一步最小接口应只要求一个匿名小包：

1. 一个 field 或 decoder parameter vector；
2. 4--16 条真实 ray；
3. curved、straight、direct residual 三个 callable，或明确说明只存在其中哪些；
4. 两个 `Jv`、一个 `J^Tq`；
5. forward 实际使用的 precision、采样步长、interpolation 和 termination 规则；
6. 每次调用返回 branch state 与 diagnostic state；
7. 一组不会公开的最小 observation，仅用于检查尺度与 gradient noise floor。

拿到接口后按顺序做：

1. 先只接 recorder，不改 renderer；
2. 在 smooth field、thin front、shock-like field 上量 h-sweep；
3. 独立比较 separate 与 residual-native 路径；
4. 冻结 probe/cotangent 数与成本；
5. untouched fields/rigs 只开一次；
6. 通过后才接 decoder chain 和 6+2 view 最小三维 inverse；
7. 最后才与同调用预算的 DeepONet/FNO/FFNO 比较。

## 7. 给师兄的精确问题

1. 真实 renderer 中 curved/straight contribution 是同一 sample 层成对形成，还是先生成两个
   detector maps 再相减？
2. support threshold、frustum、domain/stencil validity、refractive-index validity 哪些真的改变 forward？
3. 是否存在 hard mask、occupancy pruning、dynamic sample count 或 ray termination？
4. JVP/VJP 来自 PyTorch autodiff、自写伴随，还是外部代码？实际 precision 是 float32 还是 float64？
5. 组内 gradient check 目前使用哪些 `h`，典型 noise floor 是多少？
6. 能否提供 4 条匿名 ray、2 个 `Jv`、1 个 `J^Tq` 和三个路径的最小接口，不需要公开完整数据？
7. NeRIF/TDBOST 最常见失败集中在 shock、thin front、低频 plume、camera calibration drift，还是
   其他结构？

## 8. 复现入口

```bash
/opt/anaconda3/bin/python -m pytest -q \
  demo_t16_operator/test_side_weighted_adjoint_certificate.py \
  demo_t16_operator/test_run_n2_pvgr_n5_d4c_msra_semantic_v2.py

/opt/anaconda3/bin/python \
  demo_t16_operator/run_n2_pvgr_n5_d4c_msra_semantic_v2.py

/opt/anaconda3/bin/python \
  site_tools/validate_n2_pvgr_n5_d4c_msra_semantic_v2.py
```

正式 runner 拒绝未提交源码、拒绝覆盖已有输出；验证器也必须独立实现，不得导入 runner 或
side-weighted 模块。结果目录中的 `case_specs.jsonl`、6 张 CSV、`result.json`、图、manifest 和
validation report 构成完整审计包。正式 validator 重建了 720 cases、11,520 probes、34,560 组
FD 与 36,000 条 decision；四类篡改测试在刷新 manifest 后仍能拦截 input hash、branch state、
reported metric 和 reported status 的修改。

## 9. 仍然关闭的授权

- field derivative interface：false；
- decoder parameter chain：false；
- three-dimensional reconstruction：false；
- neural-operator training/superiority：false；
- real data/generalization：false；
- paper claim：false。

这不是悲观，而是把“程序通过测试”和“真实物理算法成立”之间的距离写清楚。当前最有价值的结果
不是一个漂亮排行榜，而是我们已经知道下一次向师兄要什么最小接口、每个门负责抓什么错误，以及
哪些数字绝不能写进论文摘要。
