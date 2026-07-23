# L2+H1 / L2+Huber 残差正则 Stage A 结果：NO-GO

## 0. 先看结论

正式状态：`REGULARIZED_RESIDUAL_STAGE_A_NO_GO_STAGE_B_SEALED`

这一轮不是新算法，而是对 MGRS 后续问题的经典正则化对照。七个候选在完全相同的低频基场、高频残差网络、四个投影 renderer、240 步预算和两个优化 seed 下比较：

- 无输出正则的 `MGRS-6816` 重放对照；
- 三个强度的归一化 `L2 + H1` 残差惩罚；
- 三个强度的归一化 `L2 + Huber-gradient` 残差惩罚。

结果是：**六个正则化候选没有一个同时过 field、H1、dense-angle AD 和相对 MGRS 增量门**。四个 oblique/shock Stage B 单元仍未运行。

## 1. 这一轮真正在问什么

MGRS 已经证明：让 AD、`FD(h)`、`FD(h/2)` 和 `FD(h/4)` 同时同意，可以改善新角度的连续投影，但不能保证三维场改善。这一轮问：

> 如果把高频残差限制为小范数、小梯度或边缘保留的小梯度，投影改善能否稳定转化为三维折射率场改善？

正则项只使用网络场与固定内点网格，不读 synthetic truth，因此未来可在真实重建时计算。三维 truth 只在训练结束后做已开启 Stage A 的诊断。

## 2. 目标函数与可部署边界

设冻结低频基场为 `n0`，零输出高频残差为 `d`，候选场为 `n0+d`。投影项仍是四个 renderer 的平均 whitened MSE。正则项在固定 `7 x 7 x 7` 内点网格上计算，梯度用 `h/4` 中心差分：

```text
J_H1 = L_data
     + L_base * [lambda_0 E(d^2) / E(n0^2)
               + lambda_1 E(|grad_h/4 d|^2) / E(|grad_h/4 n0|^2)]
```

Huber 对照把第二项换成归一化梯度幅值的 Huber 函数，`delta=0.25`。`L_base` 是冻结基场在 train rays 上的四 renderer 平均损失，用来把正则权重变成无量纲数。

准入规则没改：每个 checkpoint 的四个 development renderer 必须逐一不劣于低频基场，平均至少改善 0.5%；否则恢复严格零残差。

## 3. 执行和复现性

| 项目 | 数值 |
|---|---:|
| Stage A 形态/噪声单元 | 3 |
| 优化 seed | 2 |
| 候选 | 7 |
| 残差拟合 | 42 |
| seed-level 记录 | 42 |
| unit-level 记录 | 21 |
| checkpoint 历史 | 504 |
| Apple MPS 总时间 | 232.57 s |
| 低频基场对旧证据最大差 | 0 |
| MGRS control 对旧证据最大差 | 0 |

每个残差拟合结构性费用为 960 次 train projection-renderer evaluation、48 次 checkpoint projection-renderer evaluation、8 次 baseline evaluation，以及 252 次正则计算。每次正则计算在 343 点上做 7 次 field forward。

## 4. 主结果

下表的差都是候选减冻结低频基场，负数较好。

| 候选 | field 改善单元 | field 中位差 | H1 中位差 | dense-AD 中位差 | 相对 MGRS field 增量 | 判决 |
|---|---:|---:|---:|---:|---:|---|
| `MGRS control` | 2/3 | -0.001082 | +0.001323 | -0.027056 | 0 | NO-GO |
| `L2+H1 0.003` | 2/3 | -0.000833 | +0.001304 | -0.027191 | -0.000250 | NO-GO |
| `L2+H1 0.01` | 2/3 | -0.001118 | +0.001166 | -0.027188 | +0.000036 | NO-GO |
| `L2+H1 0.03` | 2/3 | -0.000598 | +0.000750 | -0.027154 | -0.000484 | NO-GO |
| `L2+Huber 0.003` | 2/3 | -0.001482 | +0.001275 | -0.027142 | +0.000400 | NO-GO |
| `L2+Huber 0.01` | 2/3 | -0.001264 | +0.001286 | -0.027201 | +0.000182 | NO-GO |
| `L2+Huber 0.03` | 2/3 | -0.001084 | +0.001402 | -0.027008 | +0.000002 | NO-GO |

预写门要求 field 中位差不高于 `-0.002`、H1 中位差不高于 0、dense-AD 不劣化，且相对 MGRS 至少多改善 `0.001`。六个正则候选全部同时失败 field 幅度、H1 方向和相对 MGRS 增量门。

## 5. 为什么“残差更平滑”不等于“总场更正确”

这个 NO-GO 不是偶然的调参不足，而是暴露了一个结构问题。设基场误差 `e0 = n0 - n*`，加入残差 `d` 后，真实 H1 误差的变化为：

```text
||grad(e0 + d)||^2 - ||grad(e0)||^2
= 2 <grad(e0), grad(d)> + ||grad(d)||^2
```

惩罚 `||grad(d)||^2` 只能压住最后的非负二次项，不知道交叉项 `2<grad(e0),grad(d)>` 是在修正基场误差，还是和它同向累加。真实场不可见时，单纯限制残差自身的范数无法控制这个交叉项。

数据也符合这个解释：

- 42 条路径中 28 条残差获准；28/28 的 dense-AD 投影都改善；
- 但只有 14/28 同时改善 field 和 H1，说明投影一致性仍然不识别三维近零空间方向；
- 已开数据里出现明显形态分叉：wrinkled 单元的 7/7 获准路径同时改善 field/H1；smooth 路径只有 7/21。这只是一个待验证假设，不能当成形态泛化证据；
- 获准路径中，改善 field 与损害 field 两组的残差粗糙度 `normalized H1 / normalized L2` 中位数约为 16.74 与 29.21。这可以作为下一轮可观测 gate 的候选特征，但阈值是在已开 Stage A 上看到的，未经留出验证。

## 6. 下一步如何改算法问题

### 立即停止

1. 不再对相同的 residual-only `L2/H1/Huber` 目标继续密集扫 lambda。
2. 不用 Stage A 的 field truth 训练一个“准入分类器”后再声称可部署。
3. 不打开 Stage B，除非一份新协议先写清候选、可观测 gate、阈值、对照和失败回退。

### 最值得的三个候选

1. **总场 H1/TV/Huber 强基线**：直接正则 `n0+d` 而不是只正则 `d`，让目标显式包含与基场的交叉项。这仍是经典对照，必须先做。
2. **可拒答的 correction-alignment gate**：输入只用 held-out camera residual、残差粗糙度、噪声、geometry 和时间一致性；只在能预测修正方向与真误差反向时放行，否则精确回退低频基场。
3. **时空/4D 对齐约束**：如果师兄数据有连续帧，用 advection/低秩张量/时间光流提供有别于同角度投影的新信息。这比继续增加同角度 renderer 更可能改变零空间。

算子学习应该放在第二或第三项里：学“哪种先验/修正值得信任”，而不是只学一个更大的投影拟合网络。

## 7. 需要师兄确认的最小数据合同

1. 真实训练中的 scalar field 是 refractive-index increment、density 还是其他变量？是否有背景值/支撑 mask？
2. 最终主终点是三维 field、held-out displacement、density/temperature，还是 PIV-BOST 补偿后的 velocity？
3. 现有 renderer 训练时使用 AD、ND 还是 hybrid？差分步长、ray samples、有限孔径与 curved ray 分别是什么？
4. 数据是单帧还是连续时序？有无用于独立评分的 held-out cameras / calibration target / CFD truth？
5. 师兄目前最常见的失败是平滑区假结构、激波过平滑、边界偏差、少视角零空间，还是运行成本？

## 8. 复现

```bash
cd /path/to/oerf-bishe-dashboard
PYTHONPATH=. .venv/bin/python -m pytest -q \
  learning_labs/test_gcs_multiscale_residual.py \
  learning_labs/test_gcs_regularized_residual.py \
  learning_labs/test_run_gcs_regularized_residual_stage_a.py

PYTHONPATH=. .venv/bin/python \
  learning_labs/run_gcs_regularized_residual_stage_a.py \
  --config demo_t16_operator/configs/gcs_regularized_residual_stage_a_v1.json \
  --output-dir /tmp/gcs_regularized_residual_stage_a_replay

cd learning_labs/results/gcs_regularized_residual_stage_a_v1
shasum -a 256 -c checksums.sha256
```

## 9. 严格主张边界

当前允许：已开 synthetic Stage A 上的 residual-only 正则化 NO-GO，以及对“残差范数不控制总误差交叉项”的数学解释。

当前不允许：新算法、算法优越、算子学习成功、真实 BOST 重建、跨形态/跨 rig 泛化、论文成功或突破性进展。
