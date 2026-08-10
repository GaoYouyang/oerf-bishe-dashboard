# v136 残差自适应局部窗口容量诊断

> 公开日期：2026-08-10
> 正式状态：`FAIL_V136_RESIDUAL_ADAPTIVE_LOCAL_WINDOW_CAPACITY`
> 独立复算：`PASS_INDEPENDENT_RECOMPUTATION_RESIDUAL_ADAPTIVE_LOCAL_WINDOW_V136_1`
> 结论边界：`algorithm_breakthrough=false`，未训练新模型，未启动资源门、独立外门或真实 BOST。

## 一句话结论

v136 用部署可见的 K1 residual 能量质心和宽度移动、缩放每路相机的 2x2 局部窗口，在同一 **3700** 个单元和同一 `2A+2A^T` 壳中，把严格通过从 v135 的 **3162/3700** 提高到 **3215/3700**，仅新增救回 **53** 个单元。仍有 **485** 个 observation-only 失败、完整轨迹仍为 **0/5**，因此这条“质心 + 宽度”自适应机制被关闭，不授权训练预测器或更大的神经网络。

## 为什么做这一步

v135 已证明固定局部窗口比全局频带更有表达力，但它把 detector 永久切成同样的四块，无法跟随不同火焰形态和相机条件下残差区域的移动。v136 因此先检验一个最小、可证伪而且部署可用的假设：

> 如果真正缺失的是局部区域的位置和尺度，那么仅从当前 K1 residual 估计每路相机的残差能量质心与宽度，移动和缩放四个平滑窗口，应当能明显补回剩余 538 个 observation 尾部。

这个试验不使用神经网络，也不增加 exact forward / adjoint 调用。真值只用于已开封开发集上的容量判定，不能进入部署输入。

## 固定了什么

数据、物理壳和科学门均与 v135 保持一致：

1. 五条已开封 PoolFire CFD 轨迹；
2. `5/7/9/12` 相机集合；
3. clean、观测噪声、旋转、平移、内参、全位姿和 combined 扰动；
4. `32x16x16` 三维粗网格与 frozen straight-ray forward；
5. 候选 `2A+2A^T`、Zero-CGLS K4 参考 `4A+4A^T`；
6. field、full-gradient、interior-gradient、observation 四类逐单元指标及轨迹 p90/worst 门；
7. 相机换序等变和不读取 validation/test truth 的边界。

## v136 的自适应窗口

对每个活跃相机、每个位移分量：

1. 从部署可见 K1 residual 的平方能量计算 detector 平面的质心；
2. 从同一能量分布计算水平和竖直标准差；
3. 质心被固定裁剪到 detector 内部，半宽由 `1.5 x std` 决定并裁剪到冻结范围；
4. 围绕该中心生成四个非负平滑窗口，逐像素和严格为 1；
5. 每个窗口仍乘上四个 DCT 频带，形成每相机 32 个方向；
6. 把四个局部系数设为同一父系数时可精确复现 v135，因此新表示严格包含父表示。

实际质心范围为 **3.0322-13.5**，半宽范围为 **3.5598-8.0**，没有触发 fallback；partition-of-unity 最大误差为 `2.22e-16`。

## 运行结果

| 证据 | v133 | v134 | v135 | v136 |
|---|---:|---:|---:|---:|
| 四指标严格通过 | 2353/3700 | 2591/3700 | 3162/3700 | **3215/3700** |
| 相对上一阶段新增 | - | +238 | +571 | **+53** |
| 剩余失败 | 1347 | 1109 | 538 | **485** |
| 完整轨迹 | 0/5 | 0/5 | 0/5 | **0/5** |

三类场与梯度仍全部为 **3700/3700**，485 个失败全部只超过 observation 门。剩余失败的 observation/K4 比值为 p50 **1.07846**、p90-higher **1.18139**、worst **1.23361**。

### 稀疏视角仍是主要瓶颈

| 相机数 | v135 通过 | v136 通过 | 新救回 | 剩余失败 |
|---:|---:|---:|---:|---:|
| 5 | 483/925 | **516/925** | 33 | **409** |
| 7 | 862/925 | **875/925** | 13 | 50 |
| 9 | 916/925 | **920/925** | 4 | 5 |
| 12 | 901/925 | **904/925** | 3 | 21 |

5 相机仍贡献 **409/485** 个剩余失败。v136 的 53 个新增通过中有 33 个来自 5 相机，但这个增量不足以改变整轨迹判决。

### p45 与 p58 仍主导尾部

| 轨迹 | v135 通过 | v136 通过 | 剩余 | observation p90 / worst |
|---|---:|---:|---:|---:|
| p14-s05 | 735 | 737 | 3 | 1.03707 / 1.07443 |
| p22-s03 | 702 | 717 | 23 | 1.04709 / 1.07478 |
| p33-s01 | 671 | 690 | 50 | 1.04817 / 1.07506 |
| p45-s05 | 493 | 507 | **233** | **1.08258 / 1.11047** |
| p58-s03 | 561 | 564 | **176** | **1.14969 / 1.23361** |

仅 p45 和 p58 就占 **409/485** 个失败。这说明窗口位置和尺度不是这两个高功率形态尾部的主缺失自由度。

### 师兄要求的误差分解已保留

| 因子 | v136 剩余失败 |
|---|---:|
| clean | 12 |
| observation noise | 75 |
| rotation | 75 |
| translation | 79 |
| intrinsics | 73 |
| full pose | 81 |
| combined | 90 |

误差并没有只集中在某一种扰动。后续仍会保留 clean baseline，并分别检查观测噪声、旋转、平移、焦距/主点和 combined 条件；相机增删、顺序和数量变化将作为独立的结果前冻结鲁棒性门。

## 为什么判定机制失败

对 v135 的 538 个失败单元，v136 在 **447** 个单元上降低了 observation 误差，但中位改善因子仅 **1.00283**，p90 为 **1.01069**，最大为 **1.02031**；只有 53 个跨过严格门，91 个完全没有改善。

更关键的是，自适应统计没有区分力：

| 统计 | 被救回 53 个 | 未解决 485 个 |
|---|---:|---:|
| residual 中心 p50 | 7.876 | 7.624 |
| residual 半宽 p50 | 6.113 | 6.332 |

两组的中心和宽度高度重叠。也就是说，剩余误差并不是简单的“窗口中心放错”或“窗口太宽/太窄”，继续调裁剪值、窗口平滑度或再加几个尺度没有足够物理依据。

部署可见的 adaptive projection-only 便宜控制仍为 **0/3700**，进一步说明仅靠一次局部投影不能得到严格容量。

## 独立复算

独立程序没有导入正式选择器，重新生成 residual moments、窗口、局部候选、四类指标和所有判决：

- 质心最大差：`0`；
- 半宽最大差：`0`；
- 便宜控制指标最大差：`3.33e-16`；
- 系数最大差：`4.41e-12`；
- 局部候选指标最大差：`1.22e-15`；
- 最终选择指标最大差：`9.99e-16`；
- summary 最大差：`1.11e-15`；
- 离散数组不一致：`0`；
- 缩放诊断差：`0.08459 < 1`；
- 正式结果与父证据在验证前后均未变化。

因此 **3215/3700、+53、485 个 observation-only 失败和 0/5 完整轨迹** 是可信的科学负结果。

## 决策与下一步

1. **关闭 residual 质心 + 宽度的 2x2 自适应窗口**，不再通过调阈值或更多窗口尺度挽救。
2. **不训练系数预测器、CNN、FNO、UNO 或 DeepONet**，当前表示的容量门没有通过，也不需要租 GPU。
3. **下一载体必须物理上不同**：仍只读取部署可见 observation / residual 与已知几何，但要显式表达 residual 的正负相位、局部符号结构和跨相机几何耦合，而不是只看能量包络。
4. **仍用同一成本壳和严格门**：先做便宜确定性 control 和 truth-aware capacity；只有全部 3700 单元及五条完整轨迹过门，才允许最小 observation-only predictor。
5. **继续执行师兄的鲁棒性建议**：公开 CFD 阶段分别加入观测噪声、相机旋转/平移、焦距/主点误差、相机增删/换序/数量变化；实验数据到位后，再依据真实噪声和标定文件重设实验门。

---

# English: v136 Residual-Adaptive Local-Window Capacity Diagnostic

## Bottom line

v136 moves and scales each camera's 2x2 local windows using the energy centroid and width of the deployment-visible CGLS-K1 residual. On the same **3,700** cells and the same `2A+2A^T` shell, strict passes rise from **3162/3700** to **3215/3700**, a gain of only **53**. There are still **485** observation-only failures and complete trajectories remain **0/5**. The centroid-and-width adaptation is therefore closed; it does not authorize a predictor or larger neural model.

## What was tested

The data, five opened PoolFire trajectories, `5/7/9/12` camera subsets, clean/noise/rotation/translation/intrinsic/full-pose/combined factors, `32x16x16` inverse grid, frozen straight-ray physics, `2A+2A^T` candidate shell, Zero-K4 reference, and all field/gradient/observation gates are unchanged from v135.

For every active camera and displacement component, v136 computes a detector-space residual-energy centroid and width, freezes clipped centers and half-widths, and constructs four smooth nonnegative windows that sum to one. The construction strictly contains v135 and is equivariant to camera reordering. Truth is used only to ask whether the opened development representation contains a feasible candidate.

## Main evidence

- Strict joint passes: **3215/3700**, only **53** above v135.
- Remaining failures: **485**, all observation-only; every field and gradient cell passes.
- Five-camera cases still contribute **409/485** remaining failures.
- p45-s05 and p58-s03 contribute **233** and **176** failures.
- The deployment-visible adaptive projection-only control passes **0/3700**.
- All five complete trajectories still fail.
- Independent recomputation matches the selected metrics to `9.99e-16`, with zero discrete-array failures.

Among the 538 v135 failures, observation error decreases in 447 cells, but the median improvement factor is only **1.00283**. The residual centroid and width distributions for rescued and unresolved cells substantially overlap, so these two energy-envelope statistics do not identify the remaining tail.

## Decision

The residual-centroid/width window family is closed. The next minimal falsifiable carrier must be physically different and remain deployment-visible: it must encode residual sign/phase, local signed structure, and cross-camera geometric coupling while preserving the same exact-call shell and strict gates. No CNN, FNO, UNO, DeepONet, GPU rental, resource gate, or external gate is authorized before capacity passes.

The advisor's robustness guidance remains part of the long-term route: retain a clean CFD baseline; separately test observation noise, rotation, translation, focal/principal-point error, and combined error; then test camera addition/removal/order/count changes. Real experimental BOST data will be incorporated later with its actual calibration and noise model.

This is an independently recomputed post-open proxy negative result, not an algorithmic breakthrough, external-generalization result, resource-speedup result, paper success, or real-BOST validation.
