# v135 固定 2x2 局部空间-频率容量诊断

> 公开日期：2026-08-10  
> 正式状态：`FAIL_V135_LOCAL_SPACE_FREQUENCY_CAPACITY`  
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_LOCAL_SPACE_FREQUENCY_V135_1`  
> 结论边界：`algorithm_breakthrough=false`，未训练新模型，未运行资源门、外部门或真实 BOST。

## 一句话结论

把每个相机、每个位移分量的四个 detector-DCT 频带乘上四个平滑 2x2 局部窗口后，严格通过从 v134 的 **2591/3700** 提高到 **3162/3700**，救回 571 个单元；但仍有 **538** 个 observation-only 失败、完整轨迹仍为 **0/5**。因此固定 2x2 局部表示有真实部分容量，却不足以授权学习器。

## 为什么做这一步

v133 已把三类场与梯度门全部守住，但 observation 尾部仍失败；v134 在同一全局频谱 span 内改变目标权重，只从 2353 提高到 2591。问题因此更像“空间形态没有被全局频带表达”，而不只是目标函数权重不对。

v135 保持数据、物理 forward、K1 correction 壳、四项指标和 1.05 逐单元门全部不变，只改变表示：

1. 每个活跃相机与位移分量仍使用四个 detector-DCT 频带；
2. 每个频带乘上左上、右上、左下、右下四个平滑非负窗口；
3. 四窗口逐像素和为 1，因此把四份系数设成父系数即可严格重建 v134；
4. 5/7/9/12 相机对应 160/224/288/384 个方向，所有相机使用同一规则，保持相机换序等变；
5. 候选仍只走一次 exact lift、一次 exact projection 和未修改 CGLS K1，总账仍为 `2A+2A^T`；参考仍是 Zero-CGLS K4 的 `4A+4A^T`。

## 运行了什么

在五条已开封 PoolFire CFD 轨迹、5/7/9/12 相机、clean 以及噪声、旋转、平移、内参、全位姿和 combined 扰动上，共评估 **3700** 个条件单元。

- v134 已通过的 2591 个单元原样保留，防止新表示伤害父结果；
- 只对 v134 的 1109 个失败单元运行七个 truth-aware 局部候选；
- 所有单元另跑一个只看部署可见残差的局部 ridge-LS 便宜控制；
- 只有 3700/3700 且 5/5 完整轨迹过门，才允许训练最小预测器。

这里的 truth-aware 选择只回答“表示里有没有可行解”，不是部署算法。

## 结果

| 证据 | v133 | v134 | v135 |
|---|---:|---:|---:|
| 四指标严格通过 | 2353/3700 | 2591/3700 | **3162/3700** |
| 相对上一阶段新增 | - | +238 | **+571** |
| 剩余失败 | 1347 | 1109 | **538** |
| 完整轨迹 | 0/5 | 0/5 | **0/5** |

538 个失败全部只超过 observation 门：field、full-gradient 与 interior-gradient 的失败数都为 **0**。剩余失败的 observation/K4 比值为 p50 **1.0822**、p90-higher **1.1831**、worst **1.2387**。

### 相机数暴露了真正瓶颈

| 相机数 | v134 通过 | v135 通过 | v135 新救回 | v135 剩余失败 |
|---:|---:|---:|---:|---:|
| 5 | 310/925 | **483/925** | 173 | **442** |
| 7 | 625/925 | **862/925** | 237 | 63 |
| 9 | 833/925 | **916/925** | 83 | 9 |
| 12 | 823/925 | **901/925** | 78 | 24 |

**442/538** 个剩余失败来自 5 相机。最差组合是 5 相机 + combined stress，仅 26/75 通过。相比之下，9 相机已达到 916/925。这说明下一步应专门处理稀疏视角下的局部 observation 残差，而不是平均扩大所有工况的模型。

### 轨迹尾部

| 轨迹 | 通过 | 剩余失败 | observation p90-higher |
|---|---:|---:|---:|
| p14-s05 | 735/740 | 5 | 1.0371 |
| p22-s03 | 702/740 | 38 | 1.0471 |
| p33-s01 | 671/740 | 69 | 1.0495 |
| p45-s05 | 493/740 | 247 | 1.0912 |
| p58-s03 | 561/740 | 179 | 1.1526 |

p45 与 p58 继续主导尾部，说明固定象限窗口不能充分跟随不同火焰形态的局部残差位置。

### 便宜控制没有解释收益

只看 K1 residual 的局部 ridge-LS 控制通过 **0/3700**。这不证明学习方法一定有效，但排除了“同一局部基上一次简单残差拟合就能得到 v135 容量”的解释。

## 独立复算与 v135.1 修复

第一版独立验证器完整重算后拒绝发布，因为它把同一个 `5e-10` 绝对容差同时用于 order-one 物理指标和最高约 `1.47e5` 的条件数诊断。差异探针证明：

- 状态、3162 通过数、候选索引、逐单元布尔数组全部完全一致；
- selected/local metric 最大差 `2.11e-15`；
- 系数最大差 `7.77e-12`；
- summary 最大差 `6.66e-16`；
- 唯一触发拒绝的是条件数类诊断的 `4.59e-8` 绝对差，约为 `3.1e-13` 相对差。

v135.1 在第二次重算前单独冻结尺度感知比较：物理指标和系数仍守 `5e-10` 绝对门，离散数组仍要求逐位一致；只有诊断数组使用 `5e-10 + 1e-11|reference|`。完整重算的最大缩放差为 **0.1701 < 1**，正式结果树在验证前后不变。这个修复只改变验证器的数值尺度处理，不改变正式结果或科学门。

## 决策

1. **关闭固定 2x2 局部表示**：它有部分容量，但没有达到 3700/3700 与 5/5。
2. **不训练 CNN/FNO/UNO/DeepONet**：容量门未过时扩大模型没有依据。
3. **下一门改为残差自适应局部窗口**：窗口中心与尺度只由部署可见 K1 residual 和已知几何生成，维度与 exact-call 账保持受控。
4. **优先检验 5 相机稀疏视角**：先问自适应表示能否修复 442 个稀疏视角失败，再看 p45/p58 形态尾部。
5. **实验数据继续等待**：当前已覆盖模拟观测噪声、相机旋转/平移、内参与 combined 扰动；真实实验到位后仍需按真实噪声和标定误差重设门限。

---

# English: v135 Fixed 2x2 Local Space-Frequency Capacity Diagnostic

## Bottom line

Multiplying each per-camera, per-component four-band detector-DCT direction by four smooth 2x2 windows raises strict passes from **2591/3700** in v134 to **3162/3700** in v135, rescuing 571 cells. However, **538** observation-only failures remain and complete trajectories stay at **0/5**. The fixed 2x2 representation therefore provides genuine partial capacity but does not authorize a learned predictor.

## What was held fixed

The five opened PoolFire trajectories, 5/7/9/12-camera subsets, 37 clean/noise/pose/intrinsic/combined conditions, straight-ray forward model, K1 correction shell, four metrics, per-cell limit, and `2A+2A^T` candidate ledger are unchanged. The four windows are nonnegative and sum to one, so the v134 parent span is contained exactly. The same construction is applied to every active camera, preserving camera-permutation equivariance.

## Main evidence

- Strict joint passes: **3162/3700**, a gain of **571** over v134.
- Remaining failures: **538**, all observation-only; no field or gradient cell fails.
- Five-camera conditions account for **442/538** remaining failures.
- By trajectory, p45-s05 and p58-s03 account for 247 and 179 failures.
- The deployment-visible local residual ridge-LS control passes **0/3700**.
- No complete trajectory passes, so no predictor or resource gate is authorized.

The first independent validator correctly failed closed because a uniform absolute tolerance was inappropriate for both order-one metrics and large condition-number diagnostics. A separately frozen v135.1 amendment retained strict absolute checks for physical metrics and coefficients, exact equality for discrete decisions, and used a scale-aware check only for diagnostics. Full recomputation then matched every scientific decision; the maximum physical-metric difference was `2.11e-15`, with zero exact-array failures.

## Next decision gate

The fixed 2x2 representation is closed. The next experiment will generate adaptive local windows solely from deployment-visible K1 residuals and known geometry, targeting the sparse five-camera observation tail without adding exact operator calls. It must first pass a truth-aware capacity test and cheap deterministic controls. CNN/FNO/UNO/DeepONet training remains unauthorized.

This is a post-open proxy mechanism result, not an algorithmic breakthrough, resource-speedup result, external-generalization result, or real-BOST validation.
