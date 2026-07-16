# v3k-F：BOST 投影 BB 的可部署噪声停止审计

> 状态：`development only`。这是一项确定性控制实验，不是新神经网络，也不是论文优越性结论。

## 一句话判断

用当前观测幅值估计噪声尺度、再执行 Morozov discrepancy first-crossing，可以在 T16 小型合成 BOST 上显著减少平均 `A/A^T` 调用，并修复 fixed PBB 的平均 noise-OOD 退化；但联合 OOD 没有可靠改善，样本尾部伤害仍在，而且没有真实相机噪声协方差，因此 **不放行 learned stopping**。

## 协议

- 初值、PBB 步长规则、64 次上限和 checkpoint 全部继承并冻结自 v3k-E。
- 验证场按 source field 分成 `V_tune=24` 和 `V_lock=16`；同一场的四种布局不得跨角色。
- 只在 `V_tune` 选择 `tau`、camera max factor 和 NCP multiplier；选择提交文件明确记录 `test_dataset_constructed=false`。
- 可部署噪声估计只使用 active observation 和已知相对噪声参数 `q`：

  `sigma_hat = q * RMS(y_active) / sqrt(1 + q^2)`。
- 真值最优迭代和 generator clean-observation noise scale 只作 oracle，不得参加部署结论。
- 若在第 `k<64` 次残差检查后停止，账本计为 `(k+1)A + kA^T`；跑满上限计为 `64A + 64A^T`。

## 主要结果

与 fixed Landweber 64 次相比，discrepancy stopping 的 source-field 平均相对增益为：

| 域 | 平均 field gain | 95% field-cluster CI | `>1%` 伤害率 | 平均总算子调用 |
|---|---:|---:|---:|---:|
| `V_lock` | +4.34% | [+1.96%, +6.78%] | 18.75% | 55.61 |
| IID | +5.16% | [+3.27%, +7.19%] | 9.38% | 60.65 |
| noise OOD | +7.10% | [+4.70%, +9.59%] | 12.50% | 41.90 |
| family OOD | +14.83% | [+14.04%, +15.66%] | 0% | 128.00 |
| joint OOD | +11.95% | [+11.04%, +12.86%] | 0% | 124.82 |

更严格地与同一 PBB 轨迹的 fixed-64 endpoint 比较：

- IID：平均 +2.75%，CI [+0.05%, +5.71%]，但 `>1%` 伤害率 37.5%。
- noise OOD：平均 +15.09%，CI [+9.82%, +20.25%]，伤害率 18.75%。
- family OOD：全部跑满 64 次，和 fixed PBB 完全相同。
- joint OOD：平均 -0.38%，CI [-0.94%, -0.03%]；这里不能宣称改善。

## 被否掉的想法

1. **NCP residual whiteness 不能单独做停止器。** 薄火焰前缘未被解析时，结构化 signal residual 会被误判为有色噪声，导致 morphology OOD 提前停止。
2. **camera-max 条件没有形成独立机制。** `V_tune` 选出的 camera max factor 在当前数据上不约束，hybrid 退化成普通 pooled discrepancy。
3. **现在训练 risk/stop head 没有科学授权。** 当前 test 域已经用于开发审计，且真实相机协方差、曲线光路和 fresh blind 都缺失。

## 下一项最小实验

先做 `v3k-G`：每相机异方差/相关噪声的预白化 discrepancy。

`r_white = C_camera^{-1/2} M (F(x)-y)`

它必须包含：

- 独立标定或独立生成的相机协方差，不得从 test truth 估计；
- pooled 与 worst-camera 两个判据；
- 新 fields、新噪声 realization、新 layouts 和冻结 sensor model；
- 同时报告 mean、p10、`>1%` harm、front error、held-out reprojection 和完整 `A/A^T` 账本。

只有确定性规则在 fresh lock 上仍留下稳定 headroom，才允许训练小型 stopping/risk head。

## 复核命令

```bash
.venv/bin/python -m pytest -q \
  demo_t16_operator/test_v3k_e_projected_bb.py \
  demo_t16_operator/test_v3k_f_noise_stopping.py
.venv/bin/python demo_t16_operator/validate_v3k_f_noise_stopping_results.py
```

当前独立复核：`24 passed`；`6,048` method rows；`4,704` stopping rows；private path scan 通过。
