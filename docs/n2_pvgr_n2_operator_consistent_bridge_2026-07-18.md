# N2-PVGR-N2：算子一致弯曲同伦与 Picard 强基线

> 状态：`development mechanism bridge`。
>
> 本文没有开放 reserved phantom，没有真实 OERF 数据，没有三维重建，没有 DeepONet/FNO
> 对比，也没有论文或创新授权。所有数字只描述当前 CPU float64 合成程序。

## 1. 一句话结论

旧 N1 的 7/9 失败不是“网络还不够大”，而是数学目标混了两种线性化：它把完整动力学的
`A delta r + B delta d` 反馈放进了轨迹切线，同时用 automatic Hessian 近似中央差分高保真
算子的坐标 Jacobian。把目标改成真正的离散弯曲同伦，并对同一个中央差分算子求导后：

- 解析 OCBH 与完整 forward-mode RK4 JVP 的最坏 relative-L2 为 `2.16e-14`；
- matched residual 最坏误差从 N1 的 `6.85%` 降到 `1.34%`；
- 原失败的 wrinkled-wide `3x/10x` reference no-harm 从 `1.143/1.774`
  降到 `1.007/1.064`；
- 9 个主候选格、9 个教师等价格、9 个 H256/H512 sentinel、3 个计时 rig 全部过门；
- OCBH p90/H128 p10 为 `0.147-0.151`，逻辑 point-query ratio 为 `0.4015625`。

但更重要的反面结论是：修正 off-by-one 后，Picard-1/2 在这九个弱合成格的
**matched residual 与当前 Mac CPU worst-case 墙钟汇总上都优于 OCBH**。因此 OCBH 目前只能
作为可解释的离散导数、风险特征和未来可微算子宿主，不能声称是最佳前向算法。这个限定很重要：
Picard 并非在每个 no-harm 指标的每一格都逐项占优。

机器结果：

```text
MECHANISM_BRIDGE_SIGNAL_ONLY_96_CELL_RECONSTRUCTION_AND_REAL_DATA_GATES_CLOSED
```

## 2. 真实问题：我们到底在对什么求导

中央差分曲光线 RHS 写为

\[
F_\Delta(r,q)=\frac{[I-u(q)u(q)^T]g_\Delta(r)}{n(r)},
\qquad u(q)=\frac{q}{\|q\|}.
\]

引入只控制轨迹弯曲的同伦参数 `epsilon`：

\[
r'=u(q),\qquad q'=\epsilon F_\Delta(r,q).
\]

观测积分始终使用完整物理曲率，不再乘 `epsilon`：

\[
Y_s(\epsilon)=\Pi h\sum_j
F_\Delta\!\left(\frac{r_j+r_{j+1}}2,
                  \frac{q_j+q_{j+1}}2\right).
\]

于是同一步数 `s` 下：

\[
Y_s(0)=M_s,\qquad Y_s(1)=H_s.
\]

`Y'_s(0)` 才是离散、同算子、同求积规则的一阶 trajectory defect。

## 3. N1 的概念错误为什么会产生 7/9

N1 求解的是

\[
\delta r'=P\delta d,
\qquad
\delta d'=F_0+A\delta r+B\delta d.
\]

这可以作为沿直线路径对完整物理动力学作一次仿射/Newton 型修正，但不是上述同伦导数。
因为

\[
\left.\frac{d}{d\epsilon}
[\epsilon F(r_\epsilon,d_\epsilon)]\right|_{\epsilon=0}=F_0,
\]

所以精确一阶轨迹切线应为

\[
\delta d'=F_0,
\]

`A delta r + B delta d` 在轨迹方程中属于更高阶项。`A/B` 仍然需要，但只用于最终观测
integrand 的路径导数。

第二个问题是算子不一致。高保真 `H` 使用中央差分梯度 `g_delta`，所以坐标 Jacobian 是

\[
A_\Delta=P\left(
\frac{Dg_\Delta}{n}
-\frac{g_\Delta(\nabla n)^T}{n^2}
\right),
\]

其中分母 `n` 的导数来自标量插值原语的 automatic gradient；分子来自中央差分梯度本身的
Jacobian。它不等于当前位置的 automatic Hessian。

raw-direction Jacobian 为

\[
B_\Delta=-\frac{[(d\cdot g_\Delta)I+d g_\Delta^T]P}{n}.
\]

这里不能再随意在左侧乘 `P`；有限 cone 下每条 ray 的投影轴也未必与共同参考方向严格正交。

## 4. 两个实现为什么在 float64 容差内一致

### 4.1 慢教师：完整 forward-mode JVP

`discrete_rk4_jvp_predictor.py` 把 scalar dual `epsilon=(0,1)` 送进完整程序，包含：

1. 每步四个 RK4 stage；
2. 每个 stage 的中央差分场查询；
3. 每完整步后的方向归一化；
4. endpoint arithmetic mean 的 midpoint output；
5. 输出 integrand 不乘 `epsilon`。

一次开发观察中，它在 128 步、64 rays 上约需 `5.4-6.0 s`，比正式计时表中完整 H128 的
约 `0.46-0.47 s` 更慢，所以只能作为教师与诊断，不是部署候选。前一个区间尚未写入
`timing.csv`，不能作为正式性能证据引用。

### 4.2 快候选：解析 OCBH

`operator_consistent_homotopy_predictor.py` 利用 `epsilon=0` 的直线路径，把所有 RK4 endpoint
和 midpoint 合并成一个 batched coefficient path：

- 1 次标量场 batch；
- 6 次中央差分 offset batch；
- 1 次标量坐标 gradient reverse sweep；
- 3 次中央差分梯度 Jacobian reverse sweep；
- 解析传播 RK4 tangent；
- exact high 调用数为 0。

对每个 ray，逻辑 point query 为 `7(2s+1)`；H128 为 `35s`，在 `s=128` 时比值是
`0.4015625`。九格中解析 OCBH 对 forward-mode 教师的最坏误差：

| 对象 | 最坏 relative-L2 |
|---|---:|
| detector output tangent | `2.159e-14` |
| position tangent | `4.950e-15` |
| direction tangent | `5.379e-15` |

这证明解析实现与当前离散教师等价，不证明教师代表真实实验。

## 5. 九格结果

三个原 development rig、两个已打开 phantom family 和 `1/3/10x` stress 共 9 格。以下均取
各方法最坏格：

| 方法 | matched residual rel-L2 | H256 no-harm | per-ray Q95 no-harm | risk Spearman 最小值 |
|---|---:|---:|---:|---:|
| N1 continuous affine | `0.06854` | `1.77360` | `1.68768` | `0.92592` |
| OCBH | `0.01337` | `1.06441` | `1.05611` | `0.99867` |
| Picard-1 | `0.001709` | `1.001015` | `1.000944` | `0.99982` |
| Picard-2 | `0.000498` | `1.000986` | `1.000908` | `0.99991` |

这里 Picard 的 risk 只是 `||Picard-M||` 的描述性排序，不是一个已经验证的可观测路由器。

## 6. Picard off-by-one 是怎么修掉的

第一版 Picard 代码做完第 `q` 次路径更新后，返回的观测仍使用更新前路径的曲率。这样名义上的
Picard-1 实际还是 straight-path output，Picard-2 才等价于一次弯曲路径重积分。

修正后，每完成指定次数的路径 sweep，还要在最终更新后的路径上额外做一次七点 curvature batch。
因此真实成本是：

| 方法 | 路径更新 batch | 最终输出 batch | 总七点 batch |
|---|---:|---:|---:|
| Picard-1 | 1 | 1 | 2 |
| Picard-2 | 2 | 1 | 3 |

这次额外输出查询已经计入 `total_field_point_queries` 和墙钟时间。修正后的当前 Python 实现中，
Picard-1/2 仍明显快于 OCBH。

## 7. 墙钟与参考解

三个 rig 各交错 20 次，候选使用 p90，完整 H128 使用 p10：

| 方法 | 最坏 p90/H128 p10 |
|---|---:|
| N1 continuous affine | `0.0956` |
| OCBH | `0.1508` |
| Picard-1 | `0.0254` |
| Picard-2 | `0.0372` |
| H128 | `1.0168` |

这只是当前 Mac CPU/Python/PyTorch 实现速度。H128 有大量顺序 dispatch 和 host scalar sync，
候选进行了批处理，所以不能把该比例写成硬件无关的算法复杂度优势。

H256 对 H512 development sentinel 的最坏值：

- 完整输出 relative-L2：`4.365e-5`；
- matched residual relative-L2：`0.004257`。

因此 H256 在当前九格可以继续作为 synthetic evaluator。它仍不是实验真值。

## 8. 对“自有算法”的真实含义

当前证据推翻了一个诱人的故事：不能说“我们设计的 OCBH 比所有已有方法更好”。Picard 是有明确
历史谱系的强基线，而且在当前问题上更好。真正值得研究的组合可能是：

1. **OCBH 作为可解释风险特征。**用 `||Y'(0)||`、Picard-1 与 OCBH 的 remainder、域余量和
   aperture statistics 决定 P1/P2/H128，而不是直接把 OCBH 当最终输出。
2. **Picard teacher + neural residual operator。**只学习 `H-P1` 或 `P2-P1`，而不是从场到完整
   `H` 黑箱回归；但当前弱场 headroom 太小，必须在更强且仍无焦散的开发域证明有可学信号。
3. **算子一致的可微 renderer。**把 OCBH 的 central-operator Jacobian 接入 NeRIF/三维重建，
   再比较等 VJP 和等 wall-time 的 DeepONet/FNO/FFNO；当前实现是 detached，尚未过此门。
4. **cone-ray 扩展。**有限孔径下 per-subray 曲线和像素平均可能比单中心 ray 修正更重要，不能
   用当前单 ray 结果替代。

## 9. 下一轮 96 格冻结设计

在打开 reserved family 前，先用已开放 generator 做分组 factorial：

```text
2 field families
x 2 rig orientation packages
x 2 aperture packages
x 3 stress levels
x 4 field seeds
= 96 physical cells
```

field seed 是重复单位，不能把同一体场的 256 条 ray 当 256 个独立样本。每格比较 N1、OCBH、
Picard-1/2、H128/H256，并继续冻结 matched、absolute、global no-harm、Q95 tail、domain/caustic、
cost 和 false-safe 门。

随后才进入：

- field JVP/VJP dot test；
- 6 train views + 2 held-out views 的三维重建；
- 等 VJP 与等 wall-time 的 DeepONet/FNO/FFNO 对比；
- peak RSS、host sync 与独立进程参考隔离；
- cone-ray aperture baseline；
- reserved phantom；
- OERF 真实几何、噪声和独立物理终点。

## 10. 不能声称什么

- 不能声称首次一阶曲光线修正；Norton 和 dynamic/paraxial ray tracing 已有长期先例。
- 不能声称首次可微曲光线或神经折射率重建；CVPR 2024 curved neural RI tomography 与
  OERF NeRIF 已存在。
- 不能声称优于 DeepONet/FNO/FFNO；它们尚未在同数据、同预算、同三维重建任务上运行。
- 不能声称 Picard 或 OCBH 在真实反应流、焦散、间断界面或有限孔径下泛化。
- 不能把当前 p90 比值写成硬件无关加速。
- 不能把 9/9 机制桥接写成论文成功。

可能的论文新意只能来自组合：**BOST 离散算子一致的 trajectory-only homotopy derivative，
加 observable remainder/cell/topology fail-closed 证书，再嵌入 cone-ray 三维神经重建并通过真实数据
和强基线。**这一表述仍需与何远哲师兄的 TDBOST distortion module 逐项核对。

## 11. 一级来源

1. [Norton 1987, OSA/Optica](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-4-10-1919)
2. [Červený et al. 1984, GJI](https://academic.oup.com/gji/article/79/1/89/601880)
3. [Raffel 2015 BOS review](https://link.springer.com/article/10.1007/s00348-015-1927-5)
4. [Zhao et al. CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)
5. [He et al. NeRIF, Physics of Fluids](https://pubs.aip.org/aip/pof/article/37/1/017143/3331552/Neural-refractive-index-field-Unlocking-the)
6. [He et al. PIV-BOST, Experiments in Fluids](https://link.springer.com/article/10.1007/s00348-025-04093-y)
7. [TDBOST DOI](https://doi.org/10.1145/3809488)
8. [TDBOST 作者代码](https://github.com/Hyz617/TDBOST)

cone-ray 的完整一级来源、最小实现和 12 项师兄数据合同见
[cone-ray 强基线设计](n2_pvgr_cone_ray_baseline_design_2026-07-18.md)。

## 12. 可复现入口

- 配置：`demo_t16_operator/configs/n2_pvgr_n2_operator_consistent_bridge_v1.json`
- 正式 runner：`demo_t16_operator/run_n2_pvgr_n2_operator_consistent_bridge.py`
- OCBH：`demo_t16_operator/operator_consistent_homotopy_predictor.py`
- forward-mode 教师：`demo_t16_operator/discrete_rk4_jvp_predictor.py`
- Picard：`demo_t16_operator/picard_curved_ray_baseline.py`
- 机器结果：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/result.json`
- 逐方法表：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/metrics.csv`
- 教师表：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/teacher_metrics.csv`
- 参考 sentinel：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/reference_sentinel.csv`
- 计时表：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/timing.csv`
- 哈希清单：`demo_t16_operator/results/n2_pvgr_n2_operator_consistent_bridge_v1/manifest.json`

复现：

```bash
.venv/bin/python demo_t16_operator/run_n2_pvgr_n2_operator_consistent_bridge.py
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_discrete_rk4_jvp_predictor.py \
  demo_t16_operator/test_operator_consistent_homotopy_predictor.py \
  demo_t16_operator/test_picard_curved_ray_baseline.py \
  demo_t16_operator/test_run_n2_pvgr_n2_operator_consistent_bridge.py
```
