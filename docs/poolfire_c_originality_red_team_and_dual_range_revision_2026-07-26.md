# C 路线原创性红队与 DualRange-K1 修订

**日期：** 2026-07-26

**作用：** 在打开 outer performance 结果前修正算法机制与论文声明

**当前状态：** `algorithm_breakthrough=false`

## 1. 优化目标不变

唯一主问题仍是何远哲师兄确认的 C 路线：

```text
PoolFire CFD rho
-> independent straight-ray density-gradient proxy
-> 16x16x32 inverse problem
-> deployment-visible learned warm initializer
-> unchanged CGLS/PCGLS refinement
```

判决采用不可交换的词典序：

1. 先满足冻结的 field / gradient / observation **单侧非劣门**；
2. 再比较逐 trajectory 的 p50 / p90 / worst 与 harm；
3. 再要求完整多视角 `A/A^T` 调用减少；
4. 再实测 fresh-process wall time；
5. 最后检查 whole-pipeline peak RSS。

这里不能把开发阶段临时容差称为“同精度”。论文中的非劣界必须由噪声、离散误差、
实验分辨率或重复测量不确定度给出物理依据。

## 2. 红队否决旧 GEOK v0

旧候选使用：

```text
q0 = A^T y
q1 = A^T A q0
h  = c0 q0 + c1 q1
x0 = alpha h
then CGLS K2
```

把所有物理调用完整入账：

| 步骤 | A | A^T |
|---|---:|---:|
| `q0=A^T y` | 0 | 1 |
| `q1=A^T A q0` | 1 | 1 |
| 解析尺度需要 `A h` | 1 | 0 |
| 缓存 `A h` 后 CGLS K2 | 2 | 2 |
| **总计** | **4** | **4** |

Zero-CGLS K4 同样是 `4A+4A^T`。因此旧 GEOK v0 可能是表示实验，却不是降调用算法，
不能承担“同精度下降低重建成本”的论文主张。它保留为已否决候选和消融，不再作为
当前主方法。

红队同时修正三个数学边界：

- `D^-1A^T y` 通常不属于 `Range(A^T)`，所以 geometry-equalized BP 不能在没有额外
  投影证明时被称为 range-safe；
- 对角 PCGLS 的迭代空间通常是 `M^-1 Range(A^T)`，CGLS 的 range 命题不能直接复制
  到 PCGLS；
- 仅看 measurement residual 的 fallback 无法保证 field / gradient no-harm，因为
  当前 `n=8192, m=2072`，至少存在 `6120` 维不可观测零空间。

## 3. 修订候选：DualRange-K1

网络不再自由输出三维场，也不在线构造第二个 Krylov 向量。它只输出与观测同形状的
双空间 proposal：

```text
z_theta = G_theta(y, geometry-visible summaries)
h       = A^T z_theta
alpha   = argmin_[0, alpha_max] ||y - alpha A h||_2^2
x0      = alpha h
then unchanged CGLS K1
```

### 可证明结构

1. `x0 in Range(A^T)` 是代数恒等式，不依赖训练成功；
2. 若常数场属于 `Null(A)`，则 `<1,x0>=<A1,z_theta>=0`，不需要可能破坏 range
   约束的显式减均值；
3. 解析区间包含 `alpha=0`，所以初始 measurement residual 不劣于零初值；
4. 网络错误仍可能伤害最终 field / gradient，不能从第 3 条推出 no-harm。

### 接受分支调用账

| 步骤 | A | A^T |
|---|---:|---:|
| `h=A^T z_theta` | 0 | 1 |
| `A h` 与解析尺度，投影进入一次性缓存 | 1 | 0 |
| unchanged CGLS K1 | 1 | 1 |
| **总计** | **2** | **2** |

若它达到 Zero-K4 的预注册终点，这才是理论上 50% 的完整调用减少。它仍不自动代表
wall-time 或内存加速。

### 回退账必须分开

- **pre-`A^T` gate：** 只用 `y`、几何摘要和模型不确定度。拒绝后直接 Zero-K4，
  拒绝分支仍为 `4A+4A^T`；若接受率为 `p`，平均调用为 `4-2p` 对。
- **post-`A h` gate：** 已支付 warm 的 `1A+1A^T`。拒绝后再跑 Zero-K4，最坏为
  `5A+5A^T`；平均调用 `5-3p`，只有 `p>1/3` 才在均值上优于 Zero-K4。

所以首选研究的是结果前冻结的 pre-`A^T` 可见量风险门，而不是事后用真值挑好样本。

## 4. 独特性假设，而不是全球唯一承诺

截至 2026-07-26 的 25 项一级来源有界检索，没有发现一项同时覆盖：

```text
BOST-specific inverse problem
+ observation-space learned dual proposal
+ by-construction Range(A^T) lift
+ pre-A^T deployment-visible accept/fallback
+ unchanged one-step CGLS correction
+ preregistered field/gradient/observation non-inferiority
+ full A/A^T, wall, RSS and trajectory-tail accounting
+ real BOST transfer
```

这只能支持“在已审阅公开来源中未发现同构组合”。学习反投影、sinogram filter、
learned warm start、neural operator + Krylov、BOST 神经重建和 solver safety 均已有
先例，任何一个零件都不能单独主张首创。专利、学位论文、未索引稿件和组内未发表工作
仍需继续核对。

可防御的作品指纹暂定为：

> **BOST-specific dual proposal + range-restricted lift + pre-operator risk
> gate + unchanged CGLS K1 + non-inferiority and full-cost evidence.**

最终方法名待 outer 结果和师兄的组内知识产权核对后再冻结，避免先命名、后找贡献。

### 补充的 learned-backprojection 近邻

原 20 项核心 claim chart 之外，本轮又把五项容易与“双空间 proposal”混淆的一级来源
纳入近邻集：

- [Learned backprojection](https://arxiv.org/abs/1908.00593)；
- [LInFBP](https://arxiv.org/abs/2505.01768)；
- [Noise2Filter](https://arxiv.org/abs/2007.01636)；
- [FNO-BP](https://arxiv.org/abs/2402.12141)；
- [Data-consistent reconstruction networks](https://arxiv.org/abs/2003.11253)。

因此不能把“在观测域学习再反投影”写成创新。待证明的差异只能是 BOST 特定的
range-restricted lift、低调用 CGLS K1 接口、pre-`A^T` 风险门和完整非劣/成本证据
共同成立。

## 5. 已完成的最小机制证据

代码入口：

- `learning_labs/poolfire_c_dual_range_warm.py`
- `site_tools/test_poolfire_c_dual_range_warm.py`

联合测试覆盖：

1. `z=y` 与 normalized BP 数值一致；
2. 已知常数零空间下，`A^Tz` 与常数场正交；
3. 负方向会被解析尺度压到 `alpha=0`，初始 residual 不增加；
4. 接受分支与 CGLS K1 的总账严格为 `2A+2A^T`；
5. dual callable 隐藏调用 `A/A^T` 或返回错误 shape 时 fail-closed。

新增测试与原 baseline 联合结果为 `22 passed`。随后在已冻结的六轨迹共同几何上使用
固定随机观测做了不读真值的机制烟测：

```text
field_shape=(16,16,32)
observation_shape=(2072,)
alpha=0.061941614953514523
initial_residual_ratio=0.80221651960594
initializer_field_mean=9.215718466126788e-19
total_calls=2A+2AT
algorithm_breakthrough=false
```

这不是训练结果，也没有任何 field / gradient 真值指标。它只证明同一机制与调用账在
正式冻结几何上可执行，而非只在玩具矩阵上成立。

## 6. 必须打赢的对照

### 同调用或更便宜

- Zero-CGLS K2 / K3 / K4；
- `z=y`，即 normalized BP + CGLS K1；
- exact 1D line search；
- exact 2D projected least-squares / Galerkin；
- dual ridge / Tikhonov；
- fit-only fixed dual filter；
- classical DCT/PCA/RBF reduced basis。

### 学习近邻

- learned sinogram filter / learned backprojection；
- Cross14 自由三维 sentinel；
- direct 3D U-Net / FNO / UNO / DeepONet；
- learned subspace / deflation；
- 最接近的一次性 neural warm-start control。

全部方法共享 trajectory split、算子、reference/gauge、终点、非劣界、停止规则、种子
和成本账。不能只对比 DeepONet/FNO，也不能只报平均 field-L2。

## 7. 论文放行条件

公开 proxy 只能在以下条件全部成立后支持方法结果：

- outer LOTO 未见整轨迹通过预注册单侧非劣门；
- 每条轨迹 p90 / worst / harm 无材料性退化；
- DualRange-K1 比 Zero-K4 和最强便宜 control 少完整 `A/A^T`；
- fresh-process wall 的 trajectory 等权中位数不慢；
- whole-pipeline peak RSS 没有被模型和缓存抵消；
- pre-`A^T` gate 的拒绝、接受和错误接受率全部报告。

真实 BOST 主张还必须增加组内最小 case、真实相机/单位/reference、现用 solver、
测量噪声、标定误差和至少一次真实迁移复现。

当前严格状态：

```text
old_geok_v0_call_reduction_claim=rejected
dual_range_k1_mechanism_candidate=true
range_restriction_proven_by_construction=true
initial_measurement_residual_nonincrease_proven=true
accepted_branch_full_calls=2A+2AT
field_or_gradient_no_harm_proven=false
outer_performance_opened=false
real_bost_transfer_completed=false
global_uniqueness_proven=false
algorithm_breakthrough=false
```

## 8. 立即向师兄确认

1. 组内是否已有未发表的 observation-space filter、learned BP、Krylov initializer、
   warm start、pyramid initialization 或相关专利？
2. 真实流程最慢的是一次 `A`、一次 `A^T`、光流、非线性 ray tracing，还是外层优化？
3. 真实验收的噪声/重复性误差是多少，能否据此定义 field / gradient / observation
   的非劣界？
4. 公开 proxy 过门后，哪一个最小真实 BOST case 可用于迁移检查？

这四问决定“理论调用减少”能否转化为实验室真正有用、也能写进论文的贡献。
