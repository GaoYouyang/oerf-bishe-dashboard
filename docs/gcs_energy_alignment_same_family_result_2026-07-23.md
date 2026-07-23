# Observable Energy-Alignment 同 family 面板：可重放 NO-GO，下一步必须增加观测信息

## 0. 一页结论

正式状态：`ENERGY_ALIGNMENT_SAME_FAMILY_NO_GO_STAGE_B_SEALED`。

上一轮的 3 个已开 synthetic 物理单元中，总场 L2/H1/Huber 的有符号变化看起来可能区分好修正和坏修正。本轮把这个事后线索放进一个结果前冻结的开发面板：12 个新 phantom seed、2 档噪声、3 种 rig 压力和 2 个网络重复，共 72 个条件单元、144 条尝试路径。

结果否定了当前能量门：

- 72 个条件单元中只有 7 个真改善，40 个中性，25 个有害；
- 总场 L2 非增门保留 7/7 个改善，但同时错放 14 个非改善单元，precision 只有 `0.3333`；
- H1/Huber/多数票门只保留 1/7 个改善，仍错放 1 个；
- leave-one-phantom-seed-out ridge logistic 只接受 1 个单元，观察到 `1 TP / 0 FP / 6 FN`。零误放是用近乎全拒绝换来的，不是安全证明；1 个接受 group 下零事件的单侧 95% 风险上界仍为 `0.95`；
- 8 种门全部未达到预写 candidate-signal 合同，Stage B 保持密封。

第二次完整 Apple MPS 重放比较了 `40,937` 个数值，最大绝对差为 `0`，分类不匹配为 `0`。这证明该负结果可重放，不证明算法成功。

**突破监测：没有突破。** 本轮得到的是一个更坚固的负结论：标量能量和单个留出投影都不足以判断三维修正方向。下一候选必须增加逐视角影响或跨模态物理信息。

## 1. 结果前冻结的层级

协议与代码在结果开启前固化于 commit `fc4e506c839300c9864aeb9964756ddde80df364`。

| 层级 | 数量 | 统计角色 |
|---|---:|---|
| morphology family | 2 | 仍是 smooth plume / wrinkled interface 同 family 开发，不是跨形态确认 |
| phantom seed | 12 | 独立 group，是 bootstrap、sign test 和留一分割的单位 |
| noise label | 2 | 2% / 8%；绝对 sigma 来自旧开发场的固定标尺，不读新真值 RMS |
| rig stress | 3 | nominal 6-view、sparse 4-view、coarse-ray 6-view |
| condition unit | 72 | `phantom x noise x rig`，主混淆矩阵单位 |
| network repeat | 2 | nuisance repeat，不当成新物理样本 |
| attempted path | 144 | 完整分母，拒绝路径不从统计里消失 |

视角角色也是分开的：train 只进 optimizer，development 只选 checkpoint，test 只给可观测 gate holdout，dense audit 只在门决定后评分。四类角度互不重叠。

真值只在拟合后生成三类标签：

- `beneficial`：两次网络重复的 field 和 H1 差都 `<= -0.0005`；
- `harmful`：任一重复的 field 或 H1 差 `>= +0.0005`；
- `neutral`：其余情况。运行时无法知道 neutral，所以 gate 接受 neutral 也记为 false accept。

## 2. 执行完整性

| 项目 | run 1 | run 2 |
|---|---:|---:|
| Apple MPS 时间 | 1032.315 s | 1032.085 s |
| model rows | 144 | 144 |
| history rows | 1728 | 1728 |
| admitted paths | 74 | 74 |
| rejected/fallback paths | 70 | 70 |
| fully admitted condition units | 27 | 27 |
| fallback condition units | 45 | 45 |
| finite outputs | 144/144 | 144/144 |

每条 attempted path 的结构账本是固定的：240 次 base train renderer、26 次 base checkpoint renderer、960 次 residual train renderer、8 次 residual baseline renderer、48 次 residual checkpoint renderer、8 次 disjoint gate holdout renderer 和 6 次 post-fit test/dense renderer。这里是非线性 neural BOST renderer，不能伪写成线性 `A/A^T` 调用。

## 3. 八种 gate 的冻结结果

| gate | TP | FP | FN | sensitivity | precision | 独立 TP groups | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| 接受所有 admitted | 7 | 20 | 0 | 1.000 | 0.259 | 3 | NO-GO |
| total L2 `<= 0` | 7 | 14 | 0 | 1.000 | 0.333 | 3 | NO-GO，误放太多 |
| total H1 `<= 0` | 1 | 1 | 6 | 0.143 | 0.500 | 1 | NO-GO |
| total Huber `<= 0` | 1 | 1 | 6 | 0.143 | 0.500 | 1 | NO-GO |
| L2 AND Huber | 1 | 1 | 6 | 0.143 | 0.500 | 1 | NO-GO |
| L2 AND H1 | 1 | 1 | 6 | 0.143 | 0.500 | 1 | NO-GO |
| energy majority | 1 | 1 | 6 | 0.143 | 0.500 | 1 | NO-GO |
| grouped ridge, `p>=0.8` | 1 | 0 | 6 | 0.143 | 1.000 | 1 | NO-GO，近全拒绝 |

预写合同要求 0 false-positive condition、至少 8 TP condition、至少 6 TP phantom groups、每个 family 至少 2 个 TP group，且每个 rig 至少 1 个 TP。八种 gate 都没有通过。

ridge 的 `precision=1` 尤其容易误导。它只在 12 个 phantom group 中接受了 1 个 group 的 1 个条件。对 1 个观测 group 零 false event，Clopper--Pearson 单侧 95% 上界是 `0.95`。所以现在最多只能说“观察到零误放”，不能说“证明安全”。

## 4. 为什么失败

### 4.1 改善集中在少数 phantom

7 个 beneficial condition 只来自 3/12 个 phantom seed：`3301:2`、`3306:1`、`3401:4`。最大单 group 占 4/7，即 `57.14%`。这说明 candidate benefit 不是均匀的 family-level 规律。

| 分组 | beneficial | neutral | harmful |
|---|---:|---:|---:|
| smooth | 3 | 18 | 15 |
| wrinkled | 4 | 22 | 10 |
| nominal 6-view | 3 | 11 | 10 |
| sparse 4-view | 2 | 13 | 9 |
| coarse-ray 6-view | 2 | 16 | 6 |
| 2% noise | 5 | 20 | 11 |
| 8% noise | 2 | 20 | 14 |

### 4.2 L2 有重叠，留出投影也不是真值单调替身

当然，beneficial 的 total-L2 change 中位数更负（`-0.04648`），harmful 的中位数为 `-0.00538`，但两类区间大量重叠。total-L2 change 与 joint truth margin 的 Spearman 只有 `+0.14883`。

更值得注意的是 disjoint holdout renderer。我们本来预期“holdout 改善越大，三维真值越好”；实际它与 joint truth margin 的 Spearman 为 `+0.39417`，方向并不支持这个单调假设。beneficial 单元的 worst-repeat holdout improvement 中位数反而是 `-0.14828`。

这不奇怪：少数新视角仍然只约束投影空间，不能唯一确定三维场。一个修正可以伤害某个 holdout 投影却减小全场误差，也可以对上 holdout 而把误差藏进其他视角的近零空间。

## 5. 完整重放证据

两次运行的 canonical summary SHA-256 都是：

`8348d27a3f2ed36f56f3b10255d0169097644bd7d1f79e094aaf8fc279d54cd0`

canonical manifest SHA-256 都是：

`3e5640d984941ac81727848ce60274b119e0bd1021ed8188e4d0ef37bb43c7d4`

repeat comparator 对 6 张 CSV 的 2,032 行逐项比较，共检查 `40,937` 个数值，最大绝对差 `0`，超过 `1e-6` 容差的数值 `0`，分类字段不匹配 `0`。runtime 和图像作为 volatile 产物不进 canonical 比较。

## 6. 下一候选：View-Influence Selective Residual Operator

这个名字目前只是 working hypothesis，不是已发明算法。它要回答的问题是：

> 一个 residual correction 被加入后，各个相机对该修正的支持是一致的，还是某个视角在独自拉动一个近零空间方向？

最小输入不再是一个总 residual，而是一个带相机身份的集合：

1. 每视角 train/dev/holdout residual 变化向量；
2. leave-one-view reconstruction influence：移除某视角后，修正方向的幅度、方向和稳定性如何变化；
3. camera pose、ray density、estimated noise、support overlap 和标定不确定度；
4. 总场/残差 L2/H1/Huber，只作辅助特征；
5. 真实接口到位后，加入 PIV velocity compensation error、质量/动量/能量 residual 或时序输运一致性。

候选必须从低容量到高容量依次比较：

| 层 | 模型 | 作用 |
|---|---|---|
| V0 | no gate + 已失败 energy gates | 固定下界，不允许丢掉 |
| V1 | exact leave-one-view classical influence | 不学习的信息上界与成本基线 |
| V2 | first-order JVP/VJP influence approximation | 测试是否能用少量真 forward 近似 V1 |
| V3 | grouped ridge/logistic on view statistics | 最低容量可拒答基线 |
| V4 | small permutation-invariant set encoder | 只有 V3 显示非线性 headroom 才开 |
| V5 | operator correction proposal + exact fallback | 只有前四层的信息和成本门都通过才有资格训练 |

真正的 operator learning 任务不是直接从 displacement 猜完整场，而是学一个跨视角数、几何和噪声的集合函数：输入各视角影响签名，输出有界修正幅度或拒答概率。

## 7. 下一轮的最小可证伪协议

1. 先从当前 12 个已开 phantom 只做 post-open mechanism engineering，用 exact leave-one-view 确认特征是否有信号；不把它写成 final result。
2. 若 V1 对 beneficial/nonbeneficial 无任何 grouped 分离，立即关闭该支线，不训练 set encoder。
3. 若有信号，新建一批不同 phantom seed 作 calibration，按 phantom 整组分割；所有阈值在 final audit 前冻结。
4. final audit 必须是又一批新 phantom，且至少包含新形态或真实 rig/session；不能再从同 12 个 seed 里切行。
5. 主门同时报 field relative-L2、truth-H1/front、逐 rig 最坏值、false acceptance、retained benefit、coverage、renderer/JVP/VJP 调用和端到端时间。
6. 0 false acceptance 必须伴随 group-level 上界；若上界过宽或 coverage 近 0，必须 NO-GO。
7. oblique/shock Stage B 继续密封；本文不授权打开。

## 8. 与一级来源的准确关系

- NeRIF 在实验中用 8 个投影重建、留 1 个投影做 reprojection validation。它支持“留视角是重要外部检查”，但不支持“单个 reprojection 指标就是三维真值风险”。
- SelectiveNet 强调 risk--coverage trade-off；它提醒我们不能只报 accepted 子集的 precision。
- Conformal Risk Control 控制的是预先定义、对 threshold 单调的 loss 的期望风险。当前 12-group 开发面板既没有 final exchangeable audit，也没有足够窄的 group risk bound，不能借用 conformal 名字。
- PIV-BOST 把三维折射率场连到速度补偿；它说明未来最有价值的 cross-modal endpoint 不是再加一个平滑度，而是补偿后速度误差。

完整条目和阅读提取见 [Observable gate 一级来源边界](observable_gate_primary_sources_2026-07-23.md)。

## 9. 现在需要何远哲师兄确认的六件事

1. 真实 NeRIF/BOST callable 能否返回逐 camera residual，以及去掉一个 camera 后的少量 correction？
2. 是否有对场参数或 voxel/implicit field 的 JVP/VJP？若没有，exact leave-one-view 的可接受成本是多少？
3. 真实主痛点是少视角、坏相机、光线弯曲、有限孔径、背景位移失配，还是时间不同步？
4. held-out camera 是否真的可从重建中留出，还是九视角都必须用于稳定重建？
5. 能否给一小段 PIV-BOST 同步帧，使 gate 最终对 velocity compensation 而不是 synthetic field truth 负责？
6. 可接受的每帧时间和一次性 rig calibration 时间各是多少？4D BOST 可以摊销哪些几何/低秩计算？

## 10. 复现与机器产物

```bash
cd /path/to/oerf-bishe-dashboard

PYTHONPATH=. .venv/bin/python -m pytest -q \
  learning_labs/test_gcs_energy_alignment_gate.py \
  learning_labs/test_run_gcs_energy_alignment_panel.py \
  learning_labs/test_compare_gcs_energy_alignment_repeats.py \
  learning_labs/test_analyze_gcs_energy_alignment_mechanism.py

shasum -a 256 -c \
  learning_labs/results/gcs_energy_alignment_panel_v1_run1/checksums.sha256
shasum -a 256 -c \
  learning_labs/results/gcs_energy_alignment_panel_v1_run2/checksums.sha256
shasum -a 256 -c \
  learning_labs/results/gcs_energy_alignment_repeat_audit_v1/checksums.sha256
shasum -a 256 -c \
  learning_labs/results/gcs_energy_alignment_mechanism_v1/checksums.sha256
```

主产物：

- `learning_labs/results/gcs_energy_alignment_panel_v1_run1/summary.json`
- `learning_labs/results/gcs_energy_alignment_panel_v1_run1/gate_summaries.csv`
- `learning_labs/results/gcs_energy_alignment_repeat_audit_v1/repeat_comparison.json`
- `learning_labs/results/gcs_energy_alignment_mechanism_v1/mechanism.json`
- `learning_labs/results/gcs_energy_alignment_mechanism_v1/energy_alignment_mechanism.png`

## 11. 证据等级与禁止主张

当前只授权三句话：

1. 结果前冻结的 12-phantom 同 family synthetic 面板中，七个固定能量门和一个低容量 grouped ridge gate 均未达到预写 candidate-signal 合同。
2. 零观察误放来自只接受一个 group，当前不构成安全或泛化证据。
3. 两次 Apple MPS 完整运行的 canonical 数值一致，该 NO-GO 可重放。

不允许声称：已发明安全 gate、已得到新算法、已优于 NeRIF/DeepONet/FNO、已验证真实 BOST/PIV-BOST、已证明跨形态或跨 rig 泛化、论文成功或研究突破。
