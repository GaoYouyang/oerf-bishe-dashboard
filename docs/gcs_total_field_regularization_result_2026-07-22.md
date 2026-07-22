# 总场 L2 / H1 / Huber Stage A 结果：NO-GO，但出现可观测能量门线索

## 0. 结论

正式状态：`TOTAL_FIELD_REGULARIZATION_STAGE_A_NO_GO_STAGE_B_SEALED`

上一轮证明 residual-only `L2+H1 / L2+Huber` 不能稳定把投影改善转成三维场改善。这一轮把经典正则改为直接作用于总场 `n0+d`，使目标显式包含冻结基场与残差的交叉项。另加纯总场 L2 对照，用于识别改善是否只来自整体收缩。

54 次拟合后，8 个总场正则候选仍无一通过全门。最好的 `total_h1_0p01` 只比最佳 residual-only 对照多改善 field `0.000083`，真实 H1 误差中位仍恶化 `+0.001434`。Stage B 没有运行。

但事后诊断出现了一个值得独立扩样的可观测线索：**候选总场能量是否相对基场不增**。这现在只是假设，不是已验证 gate。

## 1. 对照设计

共比较 9 条路径：

1. 无输出正则的 `MGRS-6816` 精确重放；
2. 两个总场 L2 强度；
3. 三个总场 `L2+H1` 强度；
4. 三个总场 `L2+Huber-gradient` 强度。

总场惩罚在固定 `7^3` 内点网格上计算，梯度用 `h/4` 中心差分。目标在冻结基场处中心化，因此零残差时正则增量为 0，但对残差参数的一阶梯度不为 0。这个梯度就包含基场-残差交叉项。

候选 checkpoint 仍只由部署可见信息准入：AD、`FD(h)`、`FD(h/2)`、`FD(h/4)` 四个 development renderer 必须逐一不劣于基场，平均至少改善 0.5%，否则精确回退基场。三维 truth 不进入训练或 checkpoint 准入。

## 2. 执行完整性

| 项目 | 数值 |
|---|---:|
| 已开 family-noise 单元 | 3 |
| 每单元 model seed | 2 |
| 候选 | 9 |
| 拟合路径 | 54 |
| checkpoint 历史 | 648 |
| Apple MPS 总时间 | 300.65 s |
| 低频基场对旧结果最大差 | 0 |
| MGRS control 对旧结果最大差 | 0 |
| finite 输出 | 54/54 |

运行前代码、候选、权重、门和 Stage B 密封规则已固化在 commit `9553fcd`。结果绑定该 commit，且 summary 同时存储 config、runner、总场正则模块、MGRS、residual-only 对照和 pretest 的 SHA-256。

## 3. 主结果

差值都是候选减冻结低频基场，负数较好。

| 候选 | field 中位差 | truth-H1 中位差 | dense-AD 中位差 | 总场 H1 能量变化 | 相对 MGRS field 增量 | 相对最佳 residual-only 增量 |
|---|---:|---:|---:|---:|---:|---:|
| `MGRS control` | -0.001082 | +0.001323 | -0.027056 | +0.037487 | 0 | -0.000400 |
| `total L2 0.003` | -0.001494 | +0.001171 | -0.027279 | +0.033614 | +0.000412 | +0.000012 |
| `total L2 0.01` | -0.001429 | +0.001307 | -0.027128 | +0.034457 | +0.000347 | -0.000053 |
| `total H1 0.001` | -0.001507 | +0.001289 | -0.027147 | +0.034173 | +0.000425 | +0.000025 |
| `total H1 0.003` | -0.001203 | +0.001284 | -0.027136 | +0.024241 | +0.000121 | -0.000279 |
| `total H1 0.01` | **-0.001565** | +0.001434 | -0.027176 | **-0.007309** | **+0.000483** | **+0.000083** |
| `total Huber 0.001` | -0.001069 | +0.001294 | -0.027135 | +0.036590 | -0.000013 | -0.000413 |
| `total Huber 0.003` | -0.001140 | +0.001318 | -0.027193 | +0.033049 | +0.000058 | -0.000342 |
| `total Huber 0.01` | -0.001410 | +0.001353 | -0.027096 | +0.021766 | +0.000328 | -0.000072 |

预写门要求：

- 至少 2/3 单元 field 改善；
- field 中位差不高于 `-0.002`；
- truth-H1 中位差不高于 0；
- dense-AD 不劣化；
- 相对 MGRS 至少多改善 `0.001`；
- 相对最佳 residual-only 至少多改善 `0.0005`。

8/8 总场正则候选均未达到 field 幅度、truth-H1 和双对照增量门。这不是可以通过降阈值解决的边缘成功。

## 4. 物理/数学解释

### 4.1 总场平滑仍不是真值对齐

`total_h1_0p01` 是唯一个让总场 H1 能量中位下降的候选，但 truth-H1 误差仍上升。原因是先验 `||grad n||^2` 假定真场更平滑；对平滑 plume 可能有利，对 wrinkled interface / shock 可能消掉真实高频。“能量低”不是物理真值的同义词。

### 4.2 纯 L2 已经解释了大部分小收益

`total_l2_0p003` 的 field 中位差为 `-0.001494`，与最好总场 H1 的 `-0.001565` 很接近。因此不能把这个小改善归因为梯度结构被正确建模；它大部分可能只是更保守的场幅。

### 4.3 同一投影门仍无法区分修正方向

54 条路径中 36 条通过原四 renderer 准入。这 36 条的 dense-AD 投影全部改善，但只有 18 条同时改善 field 和 truth-H1。这是一个对称的 18/18 分裂：投影一致性对三维修正方向几乎没有判别力。

## 5. 事后能量门诊断

所有门都只使用部署可计算的候选场与基场能量，但门的选择与评分发生在已开 Stage A 上，因此只能生成假设。

| 可观测门 | 保留的真改善 | 错放的有害路径 | 漏掉的改善 | sensitivity | specificity | precision |
|---|---:|---:|---:|---:|---:|---:|
| 总场 L2 非增 | 18 | 5 | 0 | 1.000 | 0.722 | 0.783 |
| 总场 H1 非增 | 4 | 0 | 14 | 0.222 | 1.000 | 1.000 |
| 总场 Huber 非增 | 10 | 0 | 8 | 0.556 | 1.000 | 1.000 |

只看原 MGRS control 的 4 条获准路径，总场 L2 非增恰好保留 2 条改善并拒绝 2 条有害路径。这个 `2/2 + 2/2` 是很好的思路来源，但样本只有 4，而且 9 个候选共享同三个物理单元，不能把 36 条路径写成 36 个独立实验。

## 6. 下一算法候选：Observable Energy-Alignment Gate

新候选不再改重建场本身，而是判断一个 MGRS 修正是否值得接受。

### 6.1 固定输入

- 总场 L2/H1/Huber 相对基场的有符号变化；
- 残差 L2/H1 与粗糙度 `H1/L2`；
- 四 renderer 的 development improvement 与 worst ratio；
- 噪声估计、相机角度/数量、ray sample 步长；
- 若有时序，加入相邻帧的输运/低秩一致性。

这些输入都不需要真实三维场。

### 6.2 发展顺序

1. 先扩充已开 development panel：在 smooth/wrinkled 同 family 内增加 phantom seeds、噪声、角度缺失与 ray sampling，不碰 oblique/shock Stage B。
2. 先比固定门：`L2<=0`、`Huber<=0`、两者串/并联，报告 safety-recall Pareto。
3. 再比低容量 logistic/ridge/tree gate，严格按 phantom seed 或 geometry 分组留一，不按候选路径随机分。
4. 主标签必须是 `field<base AND truth-H1<base`，并同时报告 retained field gain、worst harm、precision、sensitivity 和拒答率。
5. 只有固定门或低容量 gate 在留一单元上仍安全，才在新 commit 里冻结 oblique/shock Stage B 协议。

算子学习在这条路线里的合理位置，是学一个跨 geometry/noise 的有界准入算子，不是继续用更大网络拟合同一组投影。

## 7. 现在需要师兄确认什么

1. 真实 NeRIF/BOST 重建中，折射率增量的 L2/H1/TV 能量是否有可比的物理尺度与 support mask？
2. 平滑 plume、wrinkled flame、shock/expansion 在师兄数据里是分开训练，还是要求单一模型通用？
3. 有无连续帧，可以用时间一致性作为新观测？
4. 可否给出一个最小 callable：输入候选场或参数，输出 train/dev camera displacement 与所需 JVP/VJP？
5. 最终评分更看重 field/density/temperature，还是 PIV-BOST 补偿后 velocity？能量 gate 必须服务真实终点。

## 8. 复现

```bash
cd /path/to/oerf-bishe-dashboard

PYTHONPATH=. .venv/bin/python -m pytest -q \
  learning_labs/test_gcs_total_field_regularization.py \
  learning_labs/test_run_gcs_total_field_regularization_stage_a.py \
  learning_labs/test_analyze_gcs_total_field_energy_gate.py

PYTHONPATH=. .venv/bin/python \
  learning_labs/run_gcs_total_field_regularization_stage_a.py \
  --config demo_t16_operator/configs/gcs_total_field_regularization_stage_a_v1.json \
  --output-dir /tmp/gcs_total_field_regularization_replay

PYTHONPATH=. .venv/bin/python \
  learning_labs/analyze_gcs_total_field_energy_gate.py \
  --source-dir learning_labs/results/gcs_total_field_regularization_stage_a_v1 \
  --output-dir /tmp/gcs_total_field_energy_gate_diagnostic
```

## 9. 证据等级与禁止主张

当前只允许两句话：

1. 已开 synthetic Stage A 上，总场 L2/H1/Huber 经典对照没有达到预写算法门。
2. 已开路径中，总场有符号能量变化对修正利弊有事后判别信号，需要更大独立开发面板。

不允许声称：已发明能量 gate、已证明安全准入、已超越 NeRIF/FNO/DeepONet、已做真实 BOST 重建、已证明跨形态/跨 rig 泛化、论文成功或突破。
