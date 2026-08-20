# v170：五相机并非没有容量，失败的是当前几何选择目标

更新：2026-08-21

## 先说结论

v169 已经证明：只按低频几何可观测性选择相机，会让四个五相机梯度尾部全部变差。但它只重建了每套标定中被该目标选中的一个子集，尚不能回答一个更根本的问题：**五相机条件本身没有足够信息，还是我们选子集的目标错了？**

v170 在读取结果前固定同一 DCT1024、各向同性 H1、固定 multiplier `0.03` 和原六项 field / gradient / observation 门，对 13 套标定分别穷举九选五的全部 `126` 个子集。总计重建 `1,638` 个算子设置和 `58,968` 个候选 cell。

正式程序用二进制 MILP 判断有限可行性；完全独立的第二实现用支配剪枝整数动态规划重新判断。两者一致得到：

`PASS_GEOMETRY_ONLY_SHARED_FIVE_CAMERA_SUBSET_CAPACITY_V170`

最重要的容量层级要求每套标定只选一个五相机子集，并在该标定的全部 9 个三维场和 4 个时间上共用。这个 calibration-shared 层级通过 `4/4` 个时间分层：

| 归一化时间 | field p90 / worst | gradient p90 / worst | observation p90 / worst | 判决 |
| ---: | :--- | :--- | :--- | :--- |
| 0.00 | 0.383423 / 0.406818 | 0.733335 / 0.931543 | 0.129987 / 0.148662 | PASS |
| 0.25 | 0.379728 / 0.449621 | 0.744963 / 0.942562 | 0.124331 / 0.150721 | PASS |
| 0.75 | 0.365320 / 0.389353 | 0.748953 / 0.866080 | 0.121018 / 0.144211 | PASS |
| 1.00 | 0.365310 / 0.393213 | 0.730538 / 0.843256 | 0.126488 / 0.144232 | PASS |

这改变了科学判断：**当前受控代理中的五相机有限家族确实有容量。** v169 的失败不能再解释成“所有五相机子集都不行”，而应精确归因于“当前低频几何可观测性目标没有选中满足重建门的子集”。

但通过很窄。最紧的是 `t=0.75` 的 gradient p90：`0.748953`，只比冻结的 `0.750000` 门低 `0.001047`。这不是可以随意挥霍的余量。

## 容量不是部署方法

v170 的子集见证使用已经开封的三维真值做离线可行性选择。它回答“这样的子集存在吗”，不回答“部署时如何仅靠几何或观测找到它”。逐 cell 可变子集的上限也通过 `4/4`，但它比 calibration-shared 更依赖真值，只作为更宽的容量上界。

结果后的稳健性核查没有改变主判决，只检查它是否由唯一偶然见证支撑。正式与独立候选数组的阈值分类完全一致：13 套标定中，每套都有至少 `12` 个、最多 `81` 个在本标定全部 9 个场和 4 个时间上零越线的五相机子集，中位数为 `64`，总计 `744` 个。说明容量不是单个脆弱子集偶然擦线，但部署选择目标仍未建立。

v170 的 calibration-shared 见证与 v169 的几何启发式名单在 `13/13` 套标定中都不同。四个时间的 gradient p90 从 v169 的 `0.895479 / 0.883457 / 0.895914 / 0.860270` 变为 `0.733335 / 0.744963 / 0.748953 / 0.730538`。这证明了选择目标的重要性，不证明真值不可见时已经能选对。

## 独立复算

第二实现从候选算子和指标开始重新构建全部 `58,968` 个候选 cell，并用与正式 MILP 不同的动态规划判定两个容量层级。`23/23` 项独立检查全部通过。

候选指标与汇总指标的最大绝对差为 `3.49e-11 / 1.87e-12`；direct-forward 与 direct-residual 哨兵最大差为 `4.09e-14 / 6.30e-13`；stationarity 最大差为 `1.03e-15`。所有 p90 / worst 离散分类、cellwise 可行性、calibration-shared 可行性和最终判决完全一致。

## 成本与证据边界

单个已经选定的 H1 重建逻辑在线账仍是 `1A+1A^T`。但这次穷举搜索本身使用 `468` 次离线完整观测、`1,638` 次 direct sentinel，并继承 `13,299` 个 forward-equivalent 几何基投影；它们绝不能被写成部署成本或速度优势。

本结果仍是由可执行三维密度场与相机几何生成的直线射线受控代理，没有逐工况配对实验二维位移。没有 observation-only selector、预测器、fresh wall/RSS、未见外部工况或真实 BOST 结果。

因此当前准确边界是：五相机有限家族存在 objective-design headroom；下一门才是把几何开发条件与未见几何严格分开，只让部署选择器读取报告几何，并和便宜确定性 control 公平比较。这个门是 CPU 规模，不租 GPU。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`。

---

# v170: five cameras have capacity; the current selection objective is the failure

Updated: 2026-08-21

v169 established that selecting cameras by low-frequency geometry-only observability worsens all four five-camera gradient tails. It evaluated only the single subset chosen by that objective, however, leaving a more fundamental question unresolved: does the five-camera family itself lack adequate information, or is the subset objective wrong?

v170 freezes the same DCT1024 basis, isotropic H1 penalty, multiplier `0.03`, and six field / gradient / observation gates, then enumerates all `126` five-of-nine subsets for each of 13 calibrations. This produces `1,638` operator setups and `58,968` candidate cells.

The formal implementation uses binary mixed-integer feasibility. A fully independent second implementation uses dominance-pruned integer dynamic programming. Both return:

`PASS_GEOMETRY_ONLY_SHARED_FIVE_CAMERA_SUBSET_CAPACITY_V170`

The primary capacity level assigns one subset to each calibration and shares it across all nine 3D fields and four times. It passes all `4/4` time strata. Gradient p90 / worst values are `0.733335 / 0.931543`, `0.744963 / 0.942562`, `0.748953 / 0.866080`, and `0.730538 / 0.843256`. Field and observation gates also pass throughout.

This changes the scientific judgment: the finite five-camera family has capacity in the controlled proxy. The v169 failure cannot be explained as every five-camera subset being inadequate. It is instead attributable to the exact low-frequency geometry objective selecting inadequate subsets.

The margin is narrow. At `t=0.75`, gradient p90 is `0.748953`, only `0.001047` below the frozen `0.750000` gate.

This remains a truth-aware capacity result, not a deployment method. 3D truth is used offline to find feasible witnesses. A post-open robustness audit, used only for interpretation, finds that every calibration has between `12` and `81` subsets with no local p90 or worst-threshold violation across all nine fields and four times; the median is `64`, and formal and independent classifications agree exactly. Capacity is therefore not supported by a single accidental subset, but no result-blind selector has yet been established.

The v170 witness differs from the v169 geometry heuristic in all `13/13` calibrations. Five-camera gradient p90 values change from `0.895479 / 0.883457 / 0.895914 / 0.860270` to `0.733335 / 0.744963 / 0.748953 / 0.730538`. This establishes the importance of the objective; it does not establish that the correct subset can be found without truth.

All `23/23` independent checks pass. Maximum candidate-metric and summary differences are `3.49e-11 / 1.87e-12`; direct-forward and direct-residual sentinel differences are `4.09e-14 / 6.30e-13`; maximum stationarity difference is `1.03e-15`. Threshold classifications, both feasibility levels, and the final verdict agree exactly.

One already selected H1 reconstruction has a logical online ledger of `1A+1A^T`. The exhaustive search itself consumes substantial offline work and is not deployment cost or speed evidence. Observations remain a controlled straight-ray proxy generated from executable 3D density fields and camera geometry rather than condition-matched experimental BOST.

The precise outcome is objective-design headroom for the finite five-camera family. The next gate must separate geometry-development and held-out geometry conditions, let a result-blind selector read reported geometry only, and compare it with cheap deterministic controls. This remains CPU-scale; GPU rental is not authorized.

`algorithm_breakthrough=false`, `paper_success=false`, `resource_speedup=false`, `real_bost=false`.
