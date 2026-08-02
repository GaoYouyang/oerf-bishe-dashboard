# v96：四个观测自适应频谱方向把 Case 6 表示容量补到 90/90

## 一句话结论

v96 得到了当前 C 路线第一个经过独立复算的 **Case 6 全单元表示容量正结果**：在旧九维 physical-ball 表示之外，加入四个只由部署可见观测残差和已知几何构造的频谱带方向后，truth-aware 系数搜索在 `30 帧 × 3 档几何 = 90` 个单元上全部通过 field、full-gradient、interior-gradient、observation 及相对 Zero-K2 / Zero-K4 的八个门。

这是一项**表示容量突破**，但还不是可部署算法突破。当前系数仍由真值辅助寻找；下一步必须证明仅看 observation 与 geometry 的小模型能够稳定预测这些系数。

## 为什么做这一步

v95.1 已经把问题定位清楚：

- 父 K1 严格通过 `78/90`；
- 最佳 observation-only 线性策略为 `81/90`；
- 即使用真值在父 K1、mean、linear、RBF 候选中挑选，候选并集上限也只有 `83/90`。

所以继续调置信阈值或扩大同一个九维 predictor 不可能补齐七个共同失败。真正缺失的是**候选表示本身**。v96 因此先问一个更基础的问题：能否在不增加在线 exact `A/A^T` 调用的前提下，用观测残差和已知几何生成少量新方向，让严格可行解重新存在？

## 方法：从残差的可观测频谱里生成四个方向

对封存的九视角几何，记旧 GSLB32 基为 `U`、其观测投影为 `P = AU`，父初始化残差为 `r`。v96 对 `G = P^T P` 做特征分解，并按降序谱划成四个固定的八模态频带。第 `b` 个频带的系数方向为：

```text
c_b = V_b diag(1 / (lambda_b + 1e-8 lambda_max)) V_b^T P^T r
```

再把 `U c_b` 对旧九维方向做物理内积正交化与白化，得到四个 observation-adaptive 补方向。它有五个关键性质：

1. **部署可见**：方向生成只使用 observation residual 与 known geometry；
2. **精确嵌套**：四个新系数取零时，严格退化为旧九维 family；
3. **物理度量一致**：13 维搜索仍使用基于 correction Gram 的基不变 physical ball；
4. **成本不增加**：几何相关量可预计算，候选在线 exact 账仍为 `2A + 2A^T`；
5. **先验可解释**：四个方向对应残差在低到高谱段中的反投影修正，而不是黑箱自由场。

## 正式结果

### 严格通过数

| 几何 | 单元数 | 通过数 | maximum-gate p50 | p90-higher | worst |
|---|---:|---:|---:|---:|---:|
| F30+ | 30 | 30 | -0.04663 | -0.00770 | -0.00444 |
| F15+ | 30 | 30 | -0.08420 | -0.02747 | -0.00606 |
| F12+ | 30 | 30 | -0.07721 | -0.03891 | -0.02064 |
| **合计** | **90** | **90** | - | - | - |

所有 worst 都小于零，意味着每个单元的八个门都留有严格余量。旧九维 family 唯一未解决的 `F30+ / frame 12` 被新表示修复。

### 结构与数值检查

- 旧九维 fallback 的 field / residual 嵌套差为 `0 / 0`；
- 与 v91 旧证据重放的最大 field / residual 差约为 `4.04e-15 / 2.80e-14`；
- reduced 与 exact witness 的最大 field / residual 差为 `5.55e-17 / 8.88e-16`；
- physical-ball 往返系数误差为 `4.44e-16`，能量相对误差为 `2.34e-16`；
- 四方向 complement 最大条件数为 `11.22`，13 维物理 Gram 最大条件数约为 `1.0`；
- 新几何预计算没有产生在线 exact 调用；每个 witness 的在线账保持 `2A + 2A^T`。

## 独立复算

独立 validator 没有导入正式 v96 core 或 runner，而是重新实现频谱方向公式、物理补空间和 13 维 physical ball，并对全部 90 个单元重新构造 field、residual、metrics、gates 与调用 receipt。

复算结果：

- `90/90` 严格通过，失败列表为空；
- field、residual、metrics、gates、频带范数、ridge floor 与 complement condition 的正式/独立最大差全部为 `0`；
- 对最难的 `F30+ / frame 12`，独立 differential-evolution + SLSQP 搜索得到 maximum gate `-0.0113983`；
- 独立搜索收敛，调用 receipt 无失败。

上游仍共享 pre-v96 的物理、数据与门函数，因此这不是端到端物理实现独立性证明；但它已经独立复算了本轮新增的表示、搜索与判决链。

## 成功了什么，尚未成功什么

### 已成功

- 证明旧路线的最后容量缺口不是不可修复的物理矛盾；
- 找到一个只依赖部署可见残差与几何、可解释、低维且保持在线 exact 成本的补表示；
- 在已开封 Case 6 的 90 个单元上，把 truth-aware 严格容量从旧九维的 `89/90` 补到 `90/90`；
- 正式运行与独立实现给出一致判决。

### 尚未成功

- 13 维系数还不是 observation-only predictor 的输出；
- 没有证明在新的未开封公开反应流工况上泛化；
- 没有运行 fresh wall time 或 whole-pipeline peak RSS；
- 仍是 known-geometry、noise-free、straight-ray 代理，不是曲折光线或真实 BOST；
- 没有形成论文完成或算法整体突破。

因此当前准确状态是：

```text
representation_capacity_breakthrough = true
algorithm_breakthrough = false
paper_success = false
external_generalization = false
resource_advantage = false
real_BOST = false
```

## 下一科学门

下一步不再扩大方向数量，也不直接租 GPU。使用同一已开封 Case 6，冻结一个最小的 observation-only 系数预测器：

- 输入只含 `y`、known geometry、父 K1 residual 的谱带能量与投影特征；
- 输出只预测四个新增频谱方向系数，并保留旧九维父预测 / fail-closed fallback；
- 保持五个连续时间外折、一帧 embargo、同一八门和 `2A + 2A^T` 在线账；
- 与零新增系数、解析频谱滤波、线性 ridge、RBF-KRR 等便宜控制公平比较；
- 必须达到 `90/90`，并通过 held-out-label mutation noninterference，才允许冻结一个此前未打开的独立公开反应流外门。

只有外部门和真实资源门也通过，才进入组内真实 BOST 迁移与论文主张阶段。
