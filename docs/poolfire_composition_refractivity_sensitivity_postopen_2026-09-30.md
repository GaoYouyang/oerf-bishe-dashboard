# Composition-Dependent Refractivity Sensitivity / 组分相关折射率敏感性

**Date / 日期:** 2026-09-30
**Evidence grade / 证据等级:** post-open conditional physical diagnostic, independently recomputed on two opened training trajectories / 已开封条件性物理诊断，在两条已开封训练轨迹上独立复算

## 中文

### 问题与方法

在已开封 PoolFire 训练轨迹上，组分相关的 Gladstone-Dale 折射率增量会否改变密度-only 表示的物理梯度？主诊断使用 p22-size03 的 101 帧，并比较 `g=rho*Kmix` 与固定系数代理 `g_fixed=rho*Kbar`；p22 另用同一固定三视角几何比较二者的离散投影。随后在不同的 p14-size03 官方训练轨迹全部 101 帧上，按结果前冻结的相同系数、闭合和 `Kbar` 独立复算折射率场与梯度敏感性；该复现没有重新计算二维投影。

`Kmix` 采用公开可见光 BOS 文献中的物种系数。计算条件假设为导出 `CH4/CO2/H2O/O2` 是质量分数，且未导出部分只有 `N2`。不裁剪、不重归一化。此假设尚未从该数据的反应机制/导出配置中确认，因此所有结果都须按条件性敏感性解释。

### 结果

- 101 帧上，`rho*Kmix` 相对固定系数密度代理的场差 p50/p90/worst 为 `0.128%/0.131%/0.133%`。
- 空间梯度差为 `3.40%/3.50%/3.64%`，梯度方向余弦约 `0.9996`。
- 经相同三视角直线投影后，观测相对差 p50/p90/worst 为 `2.33%/2.42%/2.48%`。
- p14 的 101 帧复现得到场差 p50/p90/worst `0.0925%/0.0936%/0.0953%`，梯度差 `2.5816%/2.6189%/2.6406%`，梯度方向余弦 p90 `0.9998616`。这条复现只核对三维场与梯度，没有重新做二维投影。
- p22 与 p14 的 formal 输入分别由独立路径重建；p14 的最大绝对差为 `2.75e-12`。两条轨迹都只是已开封训练工况，不构成留出验证。
- 对 p22 的全部 101 帧、每帧 1.28 million voxels，另按质量分数与摩尔分数两种条件假设逐体素计算理想气体压力。跨帧中位的空间 p10/p50/p90 分别为 `101330.74/101346.33/101363.30 Pa` 与 `100996.27/101022.67/101037.70 Pa`。相对标准大气压 `101325 Pa`（定义见 [NIST SI Guide](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors)），质量分数假设的 p50 偏差约 `+21 Pa`，摩尔分数假设约 `-302 Pa`。两套独立实现对所有逐帧分位数最大差 `2.91e-11 Pa`。

### 数据与解释边界

公开 REALM 论文将 `Y_k` 定义为物种质量分数，描述 PoolFire 为单步甲烷-空气机制，并给出 `0.015 m` 网格间距；官方 PoolFire 字段统计列出 9 个变量。论文正文对快照通道的概述包含压力，而字段统计及当前本地导出列出的 9 个通道未列压力。需进一步确认导出版本与精确通道映射，不能只由变量名推断缺失组分必定为 `N2`。

两条已开封轨迹都显示，在同一条件组分闭合和系数假设下，梯度相对变化处于约 `2.6%-3.5%` 的量级，明显大于三维 refractivity 场本身约 `0.09%-0.13%` 的差异。p22 的投影差只适用于其已计算的三视角离散代理，不能外推为 p14 投影结果。本诊断提示密度到折射率的物理映射可能影响梯度目标，但没有确认数据组分语义或实际气体光学模型。它不是实验标定、像素位移、三维重建评分、求解器/学习器收益、外部泛化或真实 BOST。未访问 validation/test；没有拟合或应用校正系数。

附加的理想气体一致性检查使质量分数解释相对更可信：在假定当前 `rho` 为质量密度、`T` 为 K、四种已导出物种为质量分数且遗漏部分只有 N2 时，隐含压力的时空分布中心约 `1 atm`，并比摩尔分数假设更靠近标准大气压。不过没有压力字段、边界压力或版本级导出配置，因此两种基底仍未被数据直接辨别；该检查不能确认真实单位或物种闭合。

### 下一步要核实的物理输入

优先确认数据导出映射、每个通道的单位、四个已列物种是否为质量分数以及缺失组分；随后确认实际 `n(rho,T,Y)` 关系、使用波长、`XYdeflection` 两分量定义、`level` 的物理单位和从物理位移到像素的光路/相机尺度。已有三维场和相机标定不必重发，丢失的二维配对也不必为继续虚拟前向研究专门寻找。

## English

### Question and method

On an already-open PoolFire training trajectory, can composition-dependent Gladstone-Dale refractivity change a density-only straight-ray projection? The diagnostic uses only 101 frames from one opened training condition and one fixed three-view geometry. It compares `g=rho*Kmix` with a fixed-coefficient proxy `g_fixed=rho*Kbar`, then sends both through the same discrete projection operator.

`Kmix` uses species coefficients from a public visible-light BOS reference. The conditional assumptions are that the exported `CH4/CO2/H2O/O2` channels are mass fractions and that the only unexported complement is `N2`. No clipping or renormalization is applied. These assumptions have not been confirmed against this dataset's mechanism/export configuration, so the result is only a conditional sensitivity diagnostic.

### Results

- Across 101 frames, the field difference between `rho*Kmix` and the fixed-coefficient density proxy has p50/p90/worst values of `0.128%/0.131%/0.133%`.
- The spatial-gradient difference is `3.40%/3.50%/3.64%`, with gradient-direction cosine about `0.9996`.
- After the same three-view straight-ray projection, the relative observation difference is `2.33%/2.42%/2.48%` at p50/p90/worst.
- An independent arithmetic implementation rebuilt all 101 frames. Maximum absolute differences in refractivity, gradient, and cosine metrics are `5.42e-18`, `2.78e-17`, and `5.55e-16`. An independent ray projection reproduces the summary statistics within `6.25e-17`.

### Data and interpretation limits

The public REALM paper defines `Y_k` as species mass fractions, describes PoolFire as a single-step methane-air mechanism, and gives a `0.015 m` grid spacing. The official PoolFire field-statistics file lists nine variables. The paper's prose description of a snapshot includes pressure, while the field-statistics list and current local export list nine channels without pressure. The export version and exact channel map need confirmation; omitted species cannot be inferred to be only `N2` from channel names alone.

Across two already-open training trajectories, the same conditional closure/coefficient assumptions produce gradient differences of roughly `2.6%-3.5%`, larger than the `0.09%-0.13%` refractivity-field differences. The projection difference was computed only for the p22 three-view discrete proxy; no p14 projection difference was recomputed. This suggests that the density-to-refractivity mapping may affect the gradient target, but it does not confirm species semantics or the actual gas-optics law. It is not experimental calibration, pixel displacement, reconstruction accuracy, solver/learner gain, external generalization, or real BOST. Validation/test were not accessed; no correction coefficient was fitted or applied.

An additional all-voxel ideal-gas plausibility check is more consistent with the mass-fraction interpretation: assuming `rho` is mass density, `T` is kelvin, the four exported species are mass fractions, and the omitted remainder is only N2, the implied pressure is centered near `1 atm` and closer to standard atmospheric pressure than under the mole-fraction interpretation. There is no pressure field, boundary-pressure reference, or export-version record, so this does not identify the actual basis, units, or species closure.

### Physical inputs to confirm next

First confirm the export mapping, units for each channel, whether the four listed species are mass fractions, and which species are omitted. Then confirm the actual `n(rho,T,Y)` relation, wavelength, the two `XYdeflection` components, the physical unit of `level`, and the optical/camera scale from physical displacement to pixels. Existing 3-D fields and camera calibrations need not be resent; a missing paired 2-D experimental file need not be searched for just to continue virtual forward studies.

## Sources

- [REALM paper, dataset conventions and PoolFire generation](https://arxiv.org/html/2512.18595)
- [Official REALM PoolFire field statistics](https://github.com/deepflame-ai/REALM/blob/main/datasets/cases/PoolFire/PoolFire_stats.yaml)
- [Visible-light BOS Gladstone-Dale species coefficients](https://doi.org/10.6108/JPNE.2026.6.1.087)
- [Raffel, Background-oriented schlieren (BOS) techniques](https://doi.org/10.1007/s00348-015-1927-5)
- [NIST Guide to the SI: standard atmosphere is exactly 101325 Pa](https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors)

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.
