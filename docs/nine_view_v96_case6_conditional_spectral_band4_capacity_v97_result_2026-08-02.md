# v97：固定旧九维后，只学习四个新频谱系数仍不足

## 结论

v97 得到一个经过独立复算的**关键负结果**：把 v95 已冻结的 observation-only 九维系数固定后，即使允许真值辅助寻找四个 v96 新频谱系数，最佳基线也只能从 `81/90` 提升到 `86/90`，没有任何一个冻结基线达到严格 `90/90`。

因此，不能训练原计划的“四系数预测器”。v96 的 `13` 维联合表示容量 `90/90` 仍然成立，但旧九维与新四维必须一起调整；把旧九维锁死，会把剩余可行解排除在外。

```text
conditional_four_coefficient_training_authorized = false
joint_old_and_new_adaptation_required = true
algorithm_breakthrough = false
paper_success = false
```

## 为什么先做这一步

v96 的四个新方向本身只由观测残差与已知几何构造，但 v96 的全部 `13` 个系数仍由真值辅助搜索。若直接把 v95 的九维预测固定、只训练四个新系数，必须先证明：对每个单元，四维条件子空间中至少存在一个通过八门的解。

若这个容量门不成立，再大的四系数网络也不可能稳定达到 `90/90`。v97 因而先测试“模型是否有解”，避免把训练算力花在一个结构上限不足的任务上。

## 固定实验合同

- 数据角色：已经打开的公开 BLASTNet Case 6 开发工况。
- 样本：`30` 帧、`3` 档已知九视角几何，共 `90` 个单元。
- 旧系数：分别固定为 v95 selected linear、v95 ungated linear、v85 parent K1 三个部署可见基线。
- 自由变量：仅四个 v96 观测自适应频谱系数。
- 物理约束：在原 `13` 维物理球中精确切出每个旧基线对应的四维条件球。
- 精度门：field、full-gradient、interior-gradient、observation 相对 Zero-K2 与 Zero-K4 的八门全部不越线。
- 在线精确算子账：每个候选仍为 `2A + 2A^T`。
- 真值角色：只用于条件容量搜索和最终评分，不用于旧九维基线、方向生成或未来部署输入。

## 正式结果

| 冻结旧九维基线 | 四系数搜索前 | 四系数真值搜索后 | 仍失败 | 判决 |
|---|---:|---:|---:|---|
| selected linear v95 | 81/90 | **86/90** | 4 | 不足 |
| ungated linear v95 | 80/90 | 85/90 | 5 | 不足 |
| parent K1 v85 | 78/90 | 85/90 | 5 | 不足 |

最佳 selected 基线的逐几何结果为：

| 几何 | 通过数 | 总数 |
|---|---:|---:|
| F30+ | 29 | 30 |
| F15+ | 27 | 30 |
| F12+ | 30 | 30 |

其 maximum-gate 为：

- `p50 = -0.0654552`
- `p90-higher = -0.00618890`
- `worst = +0.0320668`

门值小于等于零才通过，因此 worst 仍明确越线。

## 失败结构

selected 基线剩余四个失败全部只落在 **interior-gradient** 门：

- 两个单元相对同调用 Zero-K2 的 interior-gradient 误差越线；
- 两个单元相对 Zero-K4 的 interior-gradient harm 越线；
- 这些单元的 field、full-gradient 与 observation 门均已通过。

这不是“观测拟合完全失败”，而是固定旧九维后，四个新方向无法独立重分配局部梯度所需的物理预算。下一步必须让旧空间与新频谱补空间联合调整。

## 独立复算

独立 validator 没有导入正式 v97 core 或 runner，并重新实现四维条件物理球。它重新构造三个基线在全部 `90` 个单元上的 `270` 次精确回放，重新计算场、残差、指标、八门与调用 receipt。

复算结果：

- field 最大正式/独立差：`0`
- residual 最大差：`0`
- metrics 最大差：`0`
- gates 最大差：`0`
- zero-new maximum-gate 最大差：`0`
- 条件坐标最大差：`1.39e-17`
- 调用 receipt 失败：`0`

独立程序还对最难失败单元重新执行全局搜索，仍未找到通过候选。准确表述是“在冻结搜索下未找到全单元可行基线”，不是数学上证明四维条件子空间绝对不可能。

上游仍共享 pre-v97 的物理、数据和门函数，因此这不是端到端物理实现独立性证明。

## 科学意义

1. **保住了 v96 的真实价值。** 联合 `13` 维表示仍有 `90/90` 容量，四个频谱方向确实带来新能力。
2. **关闭了错误训练目标。** 固定旧九维、只预测四维的结构上限只有 `86/90`，不应租 GPU 或扩大网络。
3. **定位了耦合需求。** 失败集中在 interior-gradient，说明不是简单追加系数，而是旧空间与新空间之间需要联合能量分配。
4. **缩短下一轮证据链。** 下一门只需比较小型联合 `13` 维预测与低秩耦合修正，不需要先上 FNO、UNO 或大 U-Net。

## 证据边界

本轮不是：

- 可部署 observation-only 算法成功；
- 外部工况泛化；
- wall time 或内存加速；
- curved-ray 或真实 BOST；
- 论文完成或顶刊结果。

v96 的表示容量突破保留，但截至 v97：

```text
representation_capacity_breakthrough = true
algorithm_breakthrough = false
external_generalization = false
resource_advantage = false
real_BOST = false
```

## 下一条有效门

冻结一个小型 observation-only **联合旧九维 + 新四维**预测门，或一个明确耦合的低秩修正头；保持相同五折隔离、一帧 embargo、`90` 单元、八门和 `2A+2A^T` 账。只有它严格达到 `90/90`，才有理由打开新的公开外部工况和测 fresh wall/RSS。

目前仍不建议租 GPU。
