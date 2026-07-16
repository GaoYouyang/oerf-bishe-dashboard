# PSU 9-view 几何域合同门禁：从作者 A0 到固定域 B0/B1

日期：2026-07-16
状态：A0-A1、B0-B3 和 B1 参数敏感性真实审计完成；70 视角留出协议与评分接口冻结；无重建和算法胜出结论

## 1. 这轮到底解决了什么

之前只知道 5.23 GB 的 PSU BOST 数据能被低内存 reader 读出，不能回答作者前处理给每条射线选出的积分区间是否真的落在同一个重建域里。本轮没有先训练网络，而是先把 49,766,400 条真实射线逐条过一遍：

1. 将 MATLAB v5 的 11 个核心测量变量按视角流式转为随机访问 shard；
2. 对 MATLAB `find()` 产生的掩膜显式执行 `index_python = index_matlab - 1`；
3. 原样抽取作者 `rayBoxIntersection` 与 `rayConeIntersection` 的 NumPy 几何公式；
4. 对九个视角分别构造作者 `b_data/cam_data/ipf/epf` 合同并记录异常位；
5. 用 SHA-256、shape、dtype、字节数和上下游 manifest 重新核验全部视角；
6. 生成只含聚合数值的公开 JSON 和论文级四面板图。

当前公开图和 JSON 只支持一个结论：**作者前处理在 0、3、6 号视角存在可重复的计算几何域不一致。** 它不证明重建误差、不证明密度场错误，也不证明任何神经网络更优。

## 2. 数据与来源绑定

- MAT：`HSOF_9CAM_RT.mat`
- MAT SHA-256：`622852cb06faa90271479b78ffe98e6d02bc50217ebde067226f727569f8c788`
- 作者几何源码：`meas.py`
- 几何源码 SHA-256：`abc3e6dd3c7afb6e9f266d6654890a28c015fa6cf242cac91ecf4ea6e6107caf`
- 当前验证合同 SHA-256：以 [公开九视角汇总](psu_all_view_geometry_audit_summary.json) 的 `run_contract_sha256` 为准

一个必须保留的诚实边界是：最初生成 9.8 GiB 数值中间产物时，生成器源码哈希没有写入清单。本轮已经逐文件重新核验所有产物并绑定当前验证器，但不能倒过来说“当前生成器源码就是最初产物的原始生成证明”。公开 JSON 将这一点写为 `NOT_RECORDED_NUMERIC_ARTIFACTS_REVALIDATED_BY_FILE_HASH`。

## 3. 第一处确认问题：MATLAB 掩膜基数

作者 MATLAB 前处理使用 `find()` 生成 `amask_all/imask_all`。MATLAB 索引从 1 开始，Python/TensorFlow `gather` 从 0 开始；作者 Python 入口直接载入并 gather，没有减 1。

真实数值证据中：

- active mask：10,628,822 个索引，最小 864,201，最大 49,130,390；
- inactive mask：29,430,836 个索引，最小 1,955，最大恰好 49,766,400；
- 两者都不含 0；inactive 的最大值作为 Python 零基索引越界 1 位。

因此本地适配器的唯一显式规则是：

```text
python_zero_based_global_index = matlab_find_index - 1
view_local_index = python_zero_based_global_index - view_start
```

这是一项机械修复，不代表 active/inactive 的物理语义已经确认。原作者源码保持只读，不做静默篡改。

## 4. A0：作者原始混合域为什么 NO-GO

作者流程先算外层 box，再算 25 度 cone。只要 cone 返回非零长度，就直接使用 cone 两根；cone 长度为 0 时才退回 box。这个规则有三个问题：

1. box 函数算的是无限直线与盒的交，而不是显式的前向射线 `t >= 0`；
2. cone 函数描述无界双锥，不区分单叶方向，且一正一负根、切线、退化和 miss 都可能被压成长度 0；
3. 非零 cone 段没有再与外层重建 box 相交，因而同一批射线可能在不同空间域积分。

九视角真实普查得到：

- 0、3、6 号视角为 `NO-GO`，其余六个视角通过当前机械合同；
- 250,597 条射线的完整 box 长度为 0；
- 其中 182,023 条仍被非零 cone “救回”；
- 最终仍有 68,574 条零长度射线；
- 全九视角 cone 段总长 1,860,338.081 m，其中 184,128.681 m 落在 box 外；
- pooled 路径长度加权域外比例为 **9.8976%**；
- 0、3、6 号视角各自约有 25%–26% 的 cone 总路径长度在 box 外。

active 中心线掩膜在九视角均未命中当前 `final-zero/cone-outside` 标记；inactive/boundary 掩膜只在 0、3、6 号视角命中，比例约 1.10%–1.35%。这意味着当前最直接的污染风险更接近边界损失和域合同，而不是已经证明 active 测量损失错误。

还要注意：`active unsafe = 0` 只检查中心线。有限孔径会把采样点横向移出中心线，作者 `oob_mat` 当前恒为 1，因此 B2 仍必须做逐孔径样本域指示；不能把中心线安全写成完整光束安全。

## 5. A1：作者兼容裁剪消融做了什么

A1 保留作者双锥根和 `cone miss -> box` 回退，只增加两条机械规则：

1. 所有区间先裁成前向射线 `t >= 0`；
2. 非零 cone 段必须再与前向 box 相交；无交时显式变成零长度，不允许域外采样。

这不是最终物理域，只用于隔离“裁剪”本身的影响。真实九视角结果：

- 1,879,113 / 49,766,400 条射线区间改变；
- 作者混合域总路径长度移除 **2.4282%**；
- 0、3、6 号视角分别移除 **6.8969% / 7.6520% / 7.1707%**；
- 所有正长度 A1 端点都位于前向 box 内；
- 789,416 条射线在 A1 下成为显式零长度，因此训练前必须过滤；
- active 中心线掩膜仍为 0 条改变；inactive 掩膜改变约 1.10%–1.35%。

A1 的科学判决只能写成 `AUTHOR_COMPATIBILITY_ABLATION_ONLY`。它没有建立固定空间域，没有运行 TensorFlow、LoS loss、三维重建或 held-out camera，更没有算法 superiority。

## 6. 接下来六个层级必须分开

| 层级 | 定义 | 作用 | 能否作为最终基线 |
|---|---|---|---|
| A0 `author_exact` | 作者双锥；非零 cone 优先；miss 回 box | 复现原始行为和问题 | 只能作对照 |
| A1 `clipped_hybrid` | A0 再裁到前向 box；保留回退 | 隔离裁剪的机械影响 | 只能作兼容消融 |
| B0 `forward_box` | 归一化方向；显式平行 slab；`t >= 0` | 无形状先验的固定域 | 是保守主基线 |
| B1 `one_nappe_cone_box` | `box ∩ 单叶锥`；cone miss 不回 box | 有预声明 sampling hull 的固定域 | 是形状域基线 |
| B2 `aperture_indicator` | 对每个 aperture/path sample 乘 `1_D(x_m)` | 检查完整有限孔径光束，而非中心线 | 必须消融 |
| B3 `geometry_safe_mask` | 丢弃 domain-empty / aperture-crossing ray | 防止零长度或域外行进入 loss | 必须消融 |

B1 的 25 度 cone 当前只应称为**计算采样 hull**。没有独立物理证据时，不能称为激波角、马赫角或真实流场边界。

## 7. B0/B1 的数学合同

先归一化方向：

```text
d_hat = d / ||d||
r(l) = o + l d_hat,  l >= 0
```

B0 对每个轴做 slab intersection。平行且原点在 slab 外则 miss；平行且在 slab 内则该轴贡献 `(-inf,+inf)`。最终：

```text
l_enter = max(0, max_i near_i)
l_exit  = min_i far_i
hit iff l_exit > l_enter + tolerance
```

B1 定义单叶锥：

```text
alpha(l) = V dot (r(l)-vertex)
K = {x: alpha >= 0 and ||q_perp|| <= alpha tan(theta)}
D = box intersect K
```

必须先限制 `alpha >= 0`，再解二次或线性不等式 `G(l) <= 0`，最后与 B0 区间求交。切线是零测度，不得触发 box 回退。

## 8. B2 的有限孔径合同

对路径位置 `l_m`：

```text
R(l_m) = Rap * (1 - l_m / Df)
x_m = r(l_m) + R(l_m) sqrt(u_m)
      [cos(phi_m) Rx + sin(phi_m) Ry]
```

预测位移必须使用：

```text
eps_u_hat = Csys * L / M * sum_m 1_D(x_m) [Ru dot grad(rho)(x_m)]
```

`v` 分量同理。不能按“仍在域内的样本数”重新归一化，否则会改变原本有限孔径积分的物理幅值。

## 9. 下一轮可证伪实验

1. B0/B1 已过解析 box、平行 slab、单叶方向、apex、tangent、`A≈0`、无 box overlap 与 20,000 条随机直接成员审计；
2. 九视角 B0/B1 的 49,766,400 条真实中心线已普查，解析不变量通过，但 B1 物理语义未通过；
3. B2 已用 8/16/32 点确定性低差异设计比较 centerline-only 与 sample-level indicator，并保留原固定分母；
4. B3 已分别比较 keep、drop-empty、87.5%/93.75% 预声明支持阈值和 drop-any-OOD；不得把坏 ray 改标签继续用；
5. 只有几何和 LoS smoke 都过门，才比较 held-out camera RMSE/MAE；
6. held-out 无改善时，结论只是“计算域更干净”，不能进入论文算法主张；
7. held-out 改善后才允许进入 PBB/CGLS/NeRIF 小规模逆解，并与相同 calls、正则、停止和 Monte Carlo 预算比较。

## 10. 本机与服务器边界

当前 Mac 为 Apple M5、10 核 CPU、10 核 GPU、32 GB 统一内存。它已经能稳定完成：

- 5.23 GB MAT 流式读取；
- 九视角 9.8 GiB shard/几何审计；
- A0/A1/B0/B1 全 49,766,400 射线普查；
- B2 有限孔径小批量 LoS、B3 mask、低分辨率 MPS 模型与消融。

暂不应在本机承诺：完整 400×350×350 体网格、多视角全量 TensorFlow NIRT、64³–128³ 多模型五种子训练或高采样 finite-aperture renderer。是否租 GPU 由实测门禁决定：先记录一个固定 16³/32³ case 的峰值统一内存、单步时间和吞吐，再按目标分辨率外推；没有这个 profile 不提前租卡。

## 11. 现在最需要何远哲确认的问题

1. OERF/NeRIF/TDBOST 的 MATLAB `find()` 掩膜进入 Python 前是否统一减 1？
2. 组内 cone/visual hull 是否与外层 reconstruction box 相交？cone miss 时是否回退为 box？
3. 组内的 cone 是单叶还是双锥；axis、vertex、angle 是数据集固定常量还是逐装置标定？
4. `oob_mat` 恒为 1 是有意设计、尚未实现，还是已有其他域外处理？
5. inactive mask 是纯边界条件、自由流标签、环境背景，还是只用于采样？
6. 是否存在 flow-off repeats、calibration phantom 或一条 held-out camera，可验证几何改动而不需要 3D truth？
7. 组内 forward 能否导出 `F` 与 `Fᵀ/Jᵀ`，并明确 ray、mask、grid 和 unit？
8. 如果有限孔径不是当前痛点，能否给 50–200 帧带 timestamp/同步误差的连续序列，转向 TDBOST 的 4D 突变问题？

## 12. 论文主张边界

现在允许写：

- 真实九视角前处理存在可重复的计算域不一致；
- MATLAB/Python 掩膜基数需要显式适配；
- A1 可机械保证正长度中心线段落在前向 box，并量化移除路径；
- B0/B1 的解析不变量在九视角全量中心线上通过；
- B1 只保留 B0 总路径的 15.1880%，且排除 0 号视角 1,350 条 active 中心线，因此它仍是待物理审核的 sampling-hull 消融；
- B2/B3 的离散支持和政策敏感性已经完成；下一门是 held-out camera 和 flow-off 不确定度。

现在不允许写：

- 0/3/6 是三个独立物理实验；
- 25 度是激波角或马赫角；
- A1 改善了密度场、温度场或 held-out camera；
- active 中心线安全等于完整有限孔径光束安全；
- 已经优于 DeepONet、FNO、NeRIF、NIRT 或传统层析；
- 当前结果足以支持高水平论文摘要。

## 13. 公开复核入口

- [九视角 aggregate-only JSON](psu_all_view_geometry_audit_summary.json)
- [A1 兼容裁剪公开 JSON](psu_clipped_hybrid_public_summary.json)
- [九视角论文级 PNG](../demo_t16_operator/results/psu_all_view_geometry_audit/psu_all_view_geometry_audit_figure.png)
- [SVG 矢量图](../demo_t16_operator/results/psu_all_view_geometry_audit/psu_all_view_geometry_audit_figure.svg)
- [PDF 图](../demo_t16_operator/results/psu_all_view_geometry_audit/psu_all_view_geometry_audit_figure.pdf)
- [图构建清单](../demo_t16_operator/results/psu_all_view_geometry_audit/psu_all_view_geometry_audit_figure_manifest.json)

私有 MAT、原始/订阅论文、逐射线 shard、掩膜索引和运行日志均不进入 GitHub Pages。

## 14. B0/B1 真实九视角结果：数学通过，物理仍未过门

B0/B1 使用与作者源码无依赖的 NumPy float64 解析实现，每条方向先归一化，再以 65,536 条为块流式普查九视角。真实运行处理 49,766,400 条中心线，没有发现非有限输出、端点越 box、B1 脱离单叶、B1 在 B0 miss 时命中，或 B1 长度超过 B0 的情况。这只说明解析合同自洽。

全体路径账本是：

- B0 前向 box 命中 49,515,803 条，总路径 11,030,217.512 m；
- B1 单叶 `cone ∩ box` 命中 22,924,319 条，总路径 1,675,274.584 m；
- B1 仅保留 B0 总路径的 **15.1880%**，并从 B0 命中集排除 26,591,484 条中心线；
- 作者 A0/A1 的混合域与 B0 不是同一个积分问题，因此 B0 总路径比作者选择路径长 45.4591% 不应被写成“改善”或“损失”；
- B1 总路径与作者混合域总路径的比值是 22.0924%，同样只是域定义差异。

最关键的反例在 0 号视角：active mask 中有 **1,350 / 1,013,446** 条中心线在 B1 下为 miss。这 1,350 条在作者程序中全部属于 `cone length = 0 -> box fallback`，没有非有限值；作者最终为它们使用约 0.2314–0.2318 m 的 box 路径。因此：

1. B1 的“不回退”在数学上干净，但会删掉一小批真实 active 测量；
2. 在 axis、vertex、angle 和 active mask 物理语义未由师兄确认前，B1 不能成为默认训练域；
3. B0 是当前更保守的主基线，B1 应作为 visual-hull/sampling-hull 消融；
4. B2 必须复现完整孔径光束的域内/域外支持，并保留原固定样本数的归一化；
5. 只有 held-out camera 也受益，才能说某个域合同对重建有效。

本轮科学判决是 `MECHANICAL_PASS_PHYSICAL_CONE_SEMANTICS_AND_FINITE_APERTURE_UNCONFIRMED`，不是训练 GO。

## 15. B2/B3 真实九视角结果：权重损失小，不代表可以整条删 ray

B2 使用与作者有限孔径随机变量边缘分布一致的确定性配对低差异点。每个中心线命中分别取 8、16、32 个 path+disk 样本，域外样本乘 `1_D(x_m)`，分母始终保留原固定样本数。三组设计不是嵌套序列，因此结果是离散设计敏感性，不是统计置信区间或严格单调收敛。

active B1 的固定分母支持保留率为：

| 每条中心线样本数 | 支持保留率 | any-OOD active rays | empty-support active rays |
|---:|---:|---:|---:|
| 8 | 99.99465% | 2,660 | 0 |
| 16 | 99.99198% | 7,689 | 0 |
| 32 | 99.96442% | 99,617 | 0 |

这两个指标回答不同问题：

- 支持保留率衡量原固定分母下仍留在声明域内的孔径权重；
- any-OOD 计数只问一条 ray 是否至少有一个离散样本越界。

因此，active B1 总积分权重最多只少约 0.0356%，但 32 点的 `drop_any_out` 会整条删除约 0.9374% 的中心线命中。整射线政策把很小的局部支持损失放大成明显的数据删减。

B3 在看结果前固定五种政策。active B1 的排除数为：

| 政策 | QMC-8 | QMC-16 | QMC-32 |
|---|---:|---:|---:|
| `indicator_keep` | 0 | 0 | 0 |
| `drop_empty` | 0 | 0 | 0 |
| support floor 87.5% | 1,290 | 1,223 | 1,773 |
| support floor 93.75% | 2,660 | 2,719 | 4,405 |
| `drop_any_out` | 2,660 | 7,689 | 99,617 |

当前不从这些数字里挑“最佳阈值”。最小假设参考是 B0 + fixed-denominator indicator；`drop_empty` 只用于数值清理。B1、支持阈值和 strict drop 都保留为 held-out 消融。判决是 `MASK_POLICY_SENSITIVITY_ONLY_NO_RECONSTRUCTION_OR_POLICY_SUPERIORITY`。

公开复核：

- [B2 严格公开摘要](psu_aperture_sensitivity_public_summary.json)
- [B3 严格公开摘要](psu_b3_policy_public_summary.json)
- [B2/B3 四联图](../demo_t16_operator/results/psu_b3_policy_audit/psu_b3_policy_sensitivity_figure.png)
- [图构建清单](../demo_t16_operator/results/psu_b3_policy_audit/psu_b3_policy_sensitivity_figure_manifest.json)

## 16. B1 参数敏感性：axis 不是无害符号，25 度也不是已确认真值

在读取真实参数结果前，冻结了公开参考、axis 反号、15/20/30/35 度和六个 vertex 5 mm 位移，共 12 个变体。全九视角机械不变量全部通过，因此比较的是不同声明域，而不是数值崩溃。

最强反例是 axis 反号：公开参考命中的 10,627,472 条 active 中心线全部丢失，active support IoU 为 0。15 度只保留 48.7807% active hits，20 度保留 84.8418%；30/35 度虽几乎命中全部 active，active support IoU 仍只有 73.1381% / 57.4330%，说明 hit 接近不等于积分区间接近。

六个 vertex 5 mm 应力测试的 active support IoU 为 89.3126% 到 93.0639%，all-ray support IoU 为 87.5155% 到 90.8837%。它们不是标定协方差，只说明 B1 对 vertex 有持续且可量化的依赖。z 负向 5 mm 丢失 127,855 条参考 active hits，是六个 vertex 变体中最强 hit 变化。

当前判决是 `B1_PARAMETER_DEPENDENCE_QUANTIFIED_PHYSICAL_SELECTION_REQUIRED`。这些数字不能用于挑 angle、vertex 或 axis；B0 + fixed-denominator indicator 仍是主参考。完整表和边界见 [B1 参数敏感性与 70 视角协议](psu_b1_parameter_sensitivity_and_heldout_protocol_2026-07-16.md)。

## 17. 70 视角留出：最终判决按六个旋转 run，不按数百万像素

公开数据的 7 台相机乘 10 次旋转被完整划分为：

- 9 个 support views：camera 2/3/4 × rotation 0/50/90；
- 7 个 development views：完整 rotation 40；
- 18 个 primary audit views：同三台相机 × rotation 10/20/30/60/70/80；
- 12 个未见相机 audit；
- 24 个未见相机与未见旋转 joint audit。

70/70 视角恰好出现一次。final audit 禁止参与模型选择、停止或 B1 参数选择。主终点在每个 rotation block 内汇总三台相机 active vector relative L2；候选必须六块全胜，对应预声明单侧 exact sign probability `1/64 = 0.015625`。

即使六块全胜，还必须同时超过 flow-off repeatability floor、守住 p95 tail、不得增加 ambient RMS，并通过 calibration perturbation。缺任何一项，只能写 image-space diagnostic，不能写实验优越性或 field-L2。

预测包模板和评分器已冻结并通过 synthetic conformance 测试；真实 audit payload 仍未打开。公开入口：

- [70 视角协议摘要](psu_heldout_camera_protocol_public_summary.json)
- [预测包模板](../data_templates/psu_heldout_reprojection_bundle_template.json)
- [最终评分器](../site_tools/score_psu_heldout_reprojection.py)
