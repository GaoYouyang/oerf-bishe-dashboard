# v28 审计修正：Zero 对照有信号，Full Parent 保真门未通过

更新时间：2026-07-28

## 一句话结论

我把原先只在直线代理逆问题中工作的 `Observable Reduced Warm`，接到一个真正随
当前三维场改变光线路径的可微曲线前向 `F(x)` 上，并用精确程序微分得到
`J(x)v` 与局部物理伴随 `J(x)^* w`。在 7 个已开放 PoolFire 轨迹标签下各取 frame 49、
3 种执行顺序共 21 次 matched-budget 实验中：

- `Reduced Warm + Gauss-Newton-CGLS 1 outer x 3 inner` 每次都同时不劣于
  `Zero + 2 outer x 2 inner` 的 field、gradient、observation 三项终点；
- 非线性逻辑调用从 `11F + 4JVP + 6VJP = 21` 降到
  `6F + 3JVP + 4VJP = 13`，减少 **38.10%**；
- 把模型、低秩包和 warm field 构造时间计入后，21 次同进程执行的 wall 全部下降，
  最小降幅 **24.17%**；逐轨迹三顺序中位降幅为 **28.33%-31.16%**。
- 但相对更强的 `Full Parent Warm 1x3`，Reduced 的 gradient error 在 **7/7**
  个单帧样本上都更差，绝对差为 `0.00053-0.02249`。

因此，v28 证明的是“相对 Zero 的非线性开发 headroom”，没有证明 reduced
近似保持 Full Parent 品质。独立审计还发现输入身份绑定与科学指标独立复算不完整，
所以正式标签已经从 PASS 降为：

```text
HOLD_FULL_PARENT_AND_RECOMPUTATION_GATES_V28
algorithm_breakthrough=false
```

## 为什么要做这一步

v27 已经证明，在廉价的 `16x16x32` 直线矩阵代理中，减少 `A/A^T` 调用并不等于
端到端加速：模型与进程固定开销吞掉了大部分收益，RSS 还变差。继续微调同一个小
矩阵后端没有科学价值。

v28 因此改变了真正昂贵的对象：

1. 高分辨率未知场为 `32x32x64`。
2. 光线按照当前场的梯度逐步弯曲，前向不再是固定矩阵。
3. NumPy 生成器使用 192 个积分步；逆问题使用独立 PyTorch 实现和 96 个积分步，
   避免把完全相同的离散程序当成“真实模型”与“逆模型”互相验证。
4. Gauss-Newton 每次外层都重新计算当前场的 `F/JVP/VJP`，因此少一次 outer
   表示真的少一次非线性重线性化，不是少记了一次便宜矩阵乘法。

## 创新边界与必须击败的基线

一级来源核对后，下面这些宽泛表述都不能当作创新：

- 场依赖曲线光路与可微追迹已经出现在
  [Single View Refractive Index Tomography with Neural Fields](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)；
- “网络给初值，再由 Gauss-Newton 精修”已经出现在
  [A neural network warm-start approach for the inverse acoustic obstacle scattering problem](https://doi.org/10.1016/j.jcp.2023.112341)；
- BOST 的多尺度非线性投影修正已经出现在
  [A pyramid approach for background-oriented schlieren tomography](https://doi.org/10.1007/s00348-025-04153-3)。

截至 2026-07-28 的有界检索没有找到与“曲线 BOST `F(x)`、精确程序
`J(x)v/J(x)^*w`、observation/dual proposal、局部物理伴随提升、未修改
GN-CGLS、matched-cost 三项非劣门”完全同构的公开方法，但这不是全球唯一性证明。

因此必须增加 `Direct-Field WS-GN-CGLS`：用相同 observation、geometry、
训练样本、参数量和训练预算，直接预测 field 初值，再接同一个未修改 GN-CGLS。
如果 dual/reduced 方法不能优于它，只能说明“learned warm start 有效”，不能
证明 dual-space restriction 与物理 lift 有独立价值。

## 数值实现检查

核心实现先通过以下检查，才进入 PoolFire：

- PyTorch curved forward 与独立 NumPy v25 forward 在测试网格上一致；
- JVP 通过中心有限差分检查；
- VJP 通过 `<Jv,w> = <v,J^T w>` 内积恒等式；
- Gauss-Newton-CGLS 在小型非线性算例中确实降低观测残差；
- 粗到细三线性重采样精确保留仿射场。

聚焦测试结果为 `6 passed`。这些检查证明离散程序和导数接口自洽，不证明物理
相机、Gladstone-Dale 常数或真实实验噪声已经闭合。

## 先出现过一次真实失败

最初比较使用：

```text
Zero:          2 outer x 2 inner
Reduced Warm:  1 outer x 2 inner
```

严格重算后，7 个单帧样本中只有 6 个能少一次 outer。`p45-s05` 的 field 和 gradient
已经明显优于 Zero，但 observation residual 仍是 `0.04465`，没有达到 Zero
两次 outer 的 `0.02903`。

旧 runner 还曾把“最终能达到”宽松标成 PASS。我修正了状态机：只有 **严格少一次
outer** 才算 iteration saving。然后只增加同一次线性化内部的 CGLS 深度：

```text
Reduced Warm: 1 outer x 3 inner
```

`p45-s05` 的 observation residual 降到 `0.02004`，同时 field 和 gradient
继续优于 Zero。它的账从 Zero 的 `21` 次非线性逻辑调用降为 `13` 次，wall 从约
`11.31 s` 降为约 `7.17 s`；warm 构造只需约 `0.02 s`。

这次修正不是事后换大模型，而是把一个昂贵的 outer relinearization 换成一个较便宜
的 inner Krylov step。随后同一预算被冻结，并在 7 个轨迹标签下的单帧样本、
3 个顺序上重跑。

## 七个轨迹标签下的单帧结果

下表中的误差值都是：

```text
Reduced Warm 1x3 final / Zero 2x2 final
```

小于等于 1 才通过。wall 区间来自同一轨迹的三种 arm 执行顺序，并包含保守计入的
模型、package 和 reduced warm field 构造。

| 轨迹 | 角色 | field | gradient | observation | wall 中位降幅 | 三顺序范围 |
|---|---|---:|---:|---:|---:|---:|
| p14-s01 | model-selection val | 0.5169 | 0.8203 | 0.0913 | 28.33% | 25.09%-34.41% |
| p14-s05 | fit expansion | 0.5243 | 0.8349 | 0.4569 | 29.39% | 28.53%-34.42% |
| p22-s01 | stopping val | 0.4932 | 0.8213 | 0.0845 | 30.61% | 27.80%-34.90% |
| p22-s03 | fit expansion | 0.4496 | 0.8373 | 0.2226 | 30.86% | 24.17%-33.43% |
| p33-s01 | fit | 0.5468 | 0.9079 | 0.1570 | 31.16% | 25.97%-33.91% |
| p45-s05 | fit | 0.6716 | 0.9018 | 0.6902 | 29.67% | 26.90%-31.82% |
| p58-s03 | fit | 0.5410 | 0.8555 | 0.3548 | 31.10% | 24.35%-35.03% |

全部 21 次执行都满足 **相对 Zero** 的三项终点不劣式，且 wall 都比各自的
Zero 对照低至少 15%。执行顺序改变不影响数值指标；独立聚合器逐文件检查了源码
绑定、轨迹角色、预算、调用账、指标不等式和 wall 统计。这里的“独立”仅指聚合
代码独立，不能写成科学指标已经独立复算。

## 独立审计为什么把 PASS 降为 HOLD

审计发现三项会直接改变论文判断的问题：

1. Runner 没有完整封存模型、几何、PoolFire bundle、frame payload 的身份；
   validator 也没有逐项核对这些外部输入。
2. Runner 只保存标量误差，validator 因而只能重新聚合，不能从重建场、真值和
   预测观测独立复算 field / gradient / observation 指标。
3. 原主门只比较 Reduced 与 Zero，没有要求 Reduced 保持 Full Parent 的品质。

第三项不是形式问题。重新汇总 21 个结果后，Reduced 相对 Full Parent 的
gradient error 在 7/7 个样本上都更高：

| 样本 | Reduced - Full Parent gradient relative-L2 |
|---|---:|
| p14-s01 | +0.009279 |
| p14-s05 | +0.002657 |
| p22-s01 | +0.011972 |
| p22-s03 | +0.003829 |
| p33-s01 | +0.022493 |
| p45-s05 | +0.000531 |
| p58-s03 | +0.000787 |

这说明当前 Reduced 方法利用更少信息时，确实丢失了部分梯度结构；Zero 很弱，
所以“胜过 Zero”不足以证明 reduced approximation 本身成立。

## 我实际尝试了最直接的补救

为了判断这只是 CGLS 没迭代够，还是表示本身缺信息，我在同一 7 个已开放样本上
把 Reduced 从 `1 outer x 3 inner` 加深为 `1 outer x 4 inner`，再与原冻结的
`Full Parent 1x3` 比较。

结果：

```text
observation 更好：7 / 7
gradient 仍更差：7 / 7
field + gradient + observation 同时通过 Parent 门：0 / 7
```

其中最差的 p33-s01，gradient 差值只从 `+0.022493` 变为 `+0.022371`；几乎
没有被额外 Krylov 步修复，而 observation 差值改善到 `-0.007724`。

**解释：**继续做同一 Jacobian 下的观测残差最小化，只能进一步拟合可见观测；
它无法恢复 reduced warm start 在观测弱约束/近零空间方向丢失的三维梯度结构。
因此不再通过堆 inner steps 挽救这条表示。

## 当前可以说什么

当前可以说：

> 在已经开放的公开 PoolFire CFD 曲线光路代理单帧上，固定的观测条件化 reduced
> warm field 相对 Zero 能以更少 `F/JVP/VJP` 调用达到三项不劣终点，并表现出
> 同进程 wall headroom；但它未保持 Full Parent 的 gradient 品质，正式状态为 HOLD。

这个结果比 v27 更有意义，因为 v28 的主要成本确实来自随场变化的非线性
`F/JVP/VJP`，不是人为把一个固定小矩阵重复很多次。

## 当前不能说什么

仍然不能说：

- **外部泛化。** 7 个轨迹标签在这轮之前已经打开；其中包含 fit 与两个 validation
  角色，没有新的 sealed external trajectory。
- **正式资源优势。** 目前是同进程三顺序开发计时，没有 fresh-process 重复、
  进程树峰值 RSS、CPU 和冷启动尾部。
- **优于或等价于 Full Parent。** Reduced 的 gradient error 在 7/7 个样本上更差；
  多加一个 inner Krylov 步没有修复。
- **完整独立复算。** 当前只完成独立聚合；外部输入身份和重建数组尚未形成可复算
  receipt。
- **真实 BOST。** 当前是公开 CFD density 的曲线光路代理，没有相机标定、背景图像、
  重复测量噪声、折射率量纲闭合或组内真实 reconstruction callable。
- **算法突破。** warm start、Gauss-Newton、Krylov refinement、可微 ray tracing
  各自都有公开近邻；组合差异和真实 BOST 价值还需一级来源审计与师兄核对组内 IP。
- **论文成功。** 单帧 frame 49、单个 `beta=0.002` 的跨轨迹结果还不能替代时间尾部、
  噪声、几何误差和真实实验迁移。

## 突破判断

**突破性进展：否。**

**显著阶段性进展：是，但原 PASS 已被审计降级。**

理由不是“某个平均数变好”，而是：

1. 真实进入了场依赖曲线 forward 的非线性逆问题；
2. 7 个轨迹标签的单帧样本逐条守住相对 Zero 的三项同精度；
3. 精确调用账从 21 降到 13；
4. 21 个顺序执行全部出现 wall 正收益；
5. 一条初始失败被保留并解释，修正后再统一冻结重跑；
6. 独立审计发现 Parent 门与复算缺口后，公开结论主动从 PASS 降为 HOLD；
7. `1x4` 追加诊断证实问题是表示的信息损失，不是 inner depth 不够。

当前不进入 fresh/RSS 或外部测试。下一道科学门是增加同训练预算的
`Direct-Field WS-GN-CGLS` 强基线，并让下一版 runner 完整绑定输入、保存可独立
复算的重建 receipt。只有 dual/reduced 机制能够守住 Parent 品质并优于 Direct
Field，才值得继续做资源门和外部确认。
