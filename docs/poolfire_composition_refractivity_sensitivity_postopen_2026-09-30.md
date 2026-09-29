# Composition-Dependent Refractivity Sensitivity / 组分相关折射率敏感性

**Date / 日期:** 2026-09-30
**Evidence grade / 证据等级:** post-open conditional physical diagnostic with independent recomputation / 已开封条件性物理诊断，独立复算通过

## 中文

### 问题与方法

在已开封 PoolFire 训练轨迹上，组分相关的 Gladstone-Dale 折射率增量是否会改变密度-only 直线投影？只使用一个已开封训练工况的 101 帧与同一固定三视角几何。诊断比较 `g=rho*Kmix` 与固定系数代理 `g_fixed=rho*Kbar`，并把二者送入同一离散投影算子。

`Kmix` 采用公开可见光 BOS 文献中的物种系数。计算条件假设为导出 `CH4/CO2/H2O/O2` 是质量分数，且未导出部分只有 `N2`。不裁剪、不重归一化。此假设尚未从该数据的反应机制/导出配置中确认，因此所有结果都须按条件性敏感性解释。

### 结果

- 101 帧上，`rho*Kmix` 相对固定系数密度代理的场差 p50/p90/worst 为 `0.128%/0.131%/0.133%`。
- 空间梯度差为 `3.40%/3.50%/3.64%`，梯度方向余弦约 `0.9996`。
- 经相同三视角直线投影后，观测相对差 p50/p90/worst 为 `2.33%/2.42%/2.48%`。
- 独立算术复算重建全部 101 帧；折射率差、梯度差和方向余弦的最大绝对差分别为 `5.42e-18`、`2.78e-17`、`5.55e-16`。独立射线投影重建的最大统计差不超过 `6.25e-17`。

### 数据与解释边界

公开 REALM 论文将 `Y_k` 定义为物种质量分数，描述 PoolFire 为单步甲烷-空气机制，并给出 `0.015 m` 网格间距；官方 PoolFire 字段统计列出 9 个变量。论文正文对快照通道的概述包含压力，而字段统计及当前本地导出列出的 9 个通道未列压力。需进一步确认导出版本与精确通道映射，不能只由变量名推断缺失组分必定为 `N2`。

本结果只说明在上述组分闭合和系数假设下，组成扰动在该离散投影中仍约为观测 RMS 的 2.3%-2.5%，且梯度相对变化大于场本身的变化。它不是实验标定、像素位移、三维重建评分、求解器/学习器收益、外部泛化或真实 BOST。未访问 validation/test；没有拟合或应用校正系数。

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

This only shows that, under the stated closure and coefficient assumptions, composition perturbations remain about 2.3%-2.5% of observation RMS in this discrete projection, with a larger relative change in gradients than in the field. It is not experimental calibration, pixel displacement, a reconstruction score, solver/learner gain, external generalization, or real BOST. Validation/test were not accessed, and no correction coefficient was fitted or applied.

### Physical inputs to confirm next

First confirm the export mapping, units for each channel, whether the four listed species are mass fractions, and which species are omitted. Then confirm the actual `n(rho,T,Y)` relation, wavelength, the two `XYdeflection` components, the physical unit of `level`, and the optical/camera scale from physical displacement to pixels. Existing 3-D fields and camera calibrations need not be resent; a missing paired 2-D experimental file need not be searched for just to continue virtual forward studies.

## Sources

- [REALM paper, dataset conventions and PoolFire generation](https://arxiv.org/html/2512.18595)
- [Official REALM PoolFire field statistics](https://github.com/deepflame-ai/REALM/blob/main/datasets/cases/PoolFire/PoolFire_stats.yaml)
- [Visible-light BOS Gladstone-Dale species coefficients](https://doi.org/10.6108/JPNE.2026.6.1.087)
- [Raffel, Background-oriented schlieren (BOS) techniques](https://doi.org/10.1007/s00348-015-1927-5)

`algorithm_breakthrough=false`; `paper_success=false`; `external_generalization=false`; `resource_speedup=false`; `real_bost=false`.
