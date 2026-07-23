# 多帧 BOST `q_cal` 联合剖面推断 v4：为什么 97.9% 覆盖仍然是 NO-GO

> 日期：2026-07-22  
> 机器状态：`POSTOPEN_DEVELOPMENT_FORENSICS_NO_GO`  
> 证据等级：`POSTOPEN_SAME_RIG_PROFILE_INFERENCE_FORENSICS`  
> 突破监测：**否**  
> 一句话：完整 variable projection 明显改善了点估计，GN sandwich 与 exact-score Godambe 也把原生覆盖率从 plug-in 的 72.93% 提到 93.09%；但冻结的 worst-cell 校准包络又把覆盖推到 97.90%，其 95% 区间完全高于目标 95%，所以不能把“更保守”写成校准成功。

## 1. 这轮究竟问了什么

v3 已经证明：先在名义几何下估场，再把场固定去估 `q_cal`，会出现严重欠覆盖。v4 不再泛泛问“能不能换个网络”，而是把问题拆成三层：

1. **点估计中心是否有偏？** 比较 one-step plug-in、完整 profile 的第一步、迭代 full-profile 和两折 cross-fit 对照。
2. **协方差公式是否太乐观？** 在同一个迭代终点上比较 GN sandwich、exact-score Godambe 和 penalized profile contrast。
3. **校准尺度能否迁移？** 21 个已打开 cell 各用 250 个噪声复本校准，再把同一个冻结阈值用于另外 250 个复本。

固定数据是 3 个已经在 v3 打开的 synthetic rig、每个 rig 的 3 个旧方向、`q/q_ref=1,2`，另加每个 rig 一个 `q=0` cell，共 21 个 cell、10,500 条观测。校准半与评估半只更换高斯噪声，不是 10,500 次独立物理实验，也不是 fresh-rig 验证。

## 2. 冻结主门：看似全绿，仍应判 FAIL

| 主门 | 实际 | 冻结要求 | 判决 |
|---|---:|---:|---|
| exact Godambe 评估覆盖率 | 97.90% | `>=92.5%` 且 95% CP 区间包含 95% | **FAIL** |
| 覆盖率 95% CP 区间 | 97.48%--98.27% | 包含 95% | **FAIL，过度覆盖** |
| 逐 cell 覆盖 | 21/21 个 `>=90%` | 21/21 | PASS |
| 校准最大半轴中位 / p90 | `0.2562 / 0.2782 q_ref` | `<=0.5 / <=1.0 q_ref` | PASS |
| field 中位变化 vs plug-in | 改善 2.87% | 不得恶化超过 2% | PASS |
| sequence 中位变化 vs plug-in | 改善 3.09% | 不得恶化超过 2% | PASS |
| 数值与差分检查 | 全过 | 全过 | PASS |

为什么 97.9% 不是“比 95% 更好”？因为覆盖率可以靠把置信域无限放大得到。冻结门要求目标 95% 必须落进实际覆盖率的二项置信区间，就是为了拒绝这种假胜利。本轮区间的下界已经是 97.48%，所以明确是过度保守，而不是统计误差范围内的 95%。

## 3. 点估计：迭代 full-profile 有真实 development signal

在 5,250 条评估观测上：

| 点估计方法 | `||q_hat-q||/q_ref` 中位 | p90 | 非零 `q` relative-L2 中位 | field-L2 中位 |
|---|---:|---:|---:|---:|
| one-step plug-in | 0.08610 | 0.18054 | 0.06463 | 0.01713 |
| full-profile 第一步 | 0.50545 | 1.50295 | 0.64296 | 0.07853 |
| iterative full-profile | **0.07192** | **0.13385** | **0.04988** | **0.01664** |
| corrected two-fold cross-fit | 0.20736 | 0.51109 | 0.16612 | 未定义为场重建终点 |

iterative full-profile 相对 plug-in 的 `q_ref` 归一化中位误差降低 16.46%，64.44% 的配对观测更好；field 和 sequence 中位误差分别降低 2.87% 和 3.09%。这比 v3 的 8.91% 更强，但只能叫 **opened-rig development signal**：方向、rig、噪声模型和 forward family 都已打开。

full-profile 第一步反而很差并不矛盾。它从 `q=0` 出发，第一步被局部 trust radius 限制；完整迭代通常需要 5 次 profile evaluation 才到终点。这个对照说明“把一个 full Jacobian 步骤接在网络后面”并不足够，若要提速，学习器应优化 warm start、预条件或低维搜索方向，而不能假设一步修正等于迭代解。

这份稠密 CPU 参考实现的 iterative wall time 中位约为 43.77 ms/observation，第一步约 8.87 ms；向量化批处理的 plug-in 约为 0.047 ms/observation。两者一个逐观测迭代、一个批量线性代数，不能直接写成公平加速比，报告也明确关闭成本优越性 claim。但这个差距说明，真实大算子上最合理的学习目标是减少 profile/JVP 求解次数，同时保持 exact correction，而不是只追求一次黑盒前向。

## 4. 协方差：主要改善来自完整 profile，不是复杂 bread

| 不确定度方法 | 原生 pooled coverage | 原生 cell `>=90%` | 冻结包络后 coverage | 包络 / `chi2_5` | 校准半轴中位 |
|---|---:|---:|---:|---:|---:|
| plug-in native | 72.93% | 10/21 | 99.75% | 9.327x | `0.7054 q_ref` |
| iterative GN sandwich | 93.09% | 19/21 | 97.90% | 1.277x | `0.2561 q_ref` |
| iterative exact Godambe | 93.09% | 19/21 | 97.90% | 1.279x | `0.2562 q_ref` |
| penalized profile contrast | 93.52% | 20/21 | 97.77% | 1.249x | 不解释为 Wald 半轴 |
| corrected cross-fit Wald | 62.61% | 9/21 | 99.81% | 43.677x | `3.1491 q_ref` |

两个关键判断：

1. GN sandwich 与 exact-score Godambe 几乎重合。两者统计量的逐观测相对差中位约 0.124%，半轴相对差中位约 0.087%。在这个低噪声局部 proxy 中，保留 residual-curvature 项没有改变科学结论。
2. 两折 cross-fit 是明确的负对照。它原生欠覆盖更严重，若用 worst-cell 包络补到约 100% 覆盖，半轴中位会扩到 `3.15 q_ref`。切奇偶帧没有创造独立 acquisition，也没有解决共享场与输运造成的相关性。

因此，当前最值得保留的经典底座是 **iterative full-profile + GN/exact score**，而不是 cross-fit，也不是单纯给 plug-in covariance 乘一个很大的数。

## 5. 事后探索：下一轮应冻结 pooled 校准，但本轮不能改判

看到冻结结果后，我从同一 CSV 重新计算了四种校准粒度。它们全部是 post-hoc forensic，只能生成下一协议，不能回头替换本轮主门。

| 事后阈值方案 | 评估 coverage | 95% CP 区间 | 最差 cell | 半轴中位 / p90 |
|---|---:|---:|---:|---:|
| 冻结 global worst-cell max | 97.90% | 97.48%--98.27% | 96.4% | 0.2562 / 0.2782 |
| pooled calibration | **94.50%** | **93.84%--95.10%** | **91.2%** | **0.2345 / 0.2546** |
| 每 rig 的 worst-cell max | 96.23% | 95.68%--96.73% | 92.8% | 0.2343 / 0.2782 |
| 每 cell 自己校准 | 95.41% | 94.81%--95.96% | 92.0% | 0.2294 / 0.2711 |

pooled threshold 为 11.856，冻结 global max 为 14.154。pooled 方案在同一批已打开 rig 的新噪声半上恰好表现最好：总体 95% 目标落在 CP 区间内，21/21 cell 仍高于 90%，区间还更窄。但它没有 conditional coverage 保证，也没有跨新 rig 验证；每-cell 方案更是直接使用了 cell 身份，不能作为未知实验的部署方法。

当前可以提出、但还不能声称成立的假设是：full-profile 之后剩余尺度异质性已经较小，global worst-cell max 对一个 rig 的尾部过度反应。下一轮应该**事前冻结 pooled 或分层收缩校准**，并用完全新 rig/session 检验总体 95%、逐 rig 尾部和半轴功效是否同时成立。

## 6. 对算子学习与三维重建的直接启发

### 候选 A：经典 full-profile + fresh pooled calibration

这是必须先打败的最强非学习基线。新协议固定新 rig、五个参数的正/负单轴、`q=0` 和 operator-only anchor weak direction；校准单位按独立 rig/session 切分，而不是随机切同一序列的帧。若这条经典线已经足够，神经网络没有必要接管 `q`。

### 候选 B：物理目标正交 score

当前 sandwich 主要修正标准误，不能保证 ridge 后的 pseudo-field 中心等于真实物理场。下一研究式应显式处理中心偏差：

```text
psi_q^perp = psi_q - A_qx A_xx^-1 psi_x + bias_transport_correction
```

最后一项必须来自独立 flow-off/target、可信输运或重复 acquisition，不能读 synthetic truth。它需要何远哲师兄确认真实 callable、JVP/VJP、噪声 covariance 与独立重复单位。

### 候选 C：可验证的 learned tangent / warm start

若完整 profile 在真实算子上太贵，DeepONet/FNO/轻量时空网络只预测：

- `q` 的保守 warm start；
- nuisance/transport tangent 的低秩基；
- profile solver 的预条件或阻尼 proposal；
- pooled 校准的有界修正量。

随后仍由真实 forward/JVP/VJP 做 1--2 次 correction，并以 profile score、held-out view/time、field/gradient 终点和 fail-closed 半径决定接受或回退。创新点若最终成立，应是“学习提议 + 物理求解器验证 + rig-level 校准”这条组合，而不是宣称一个黑盒直接替代 TDBOST/NeRIF。

## 7. 下一轮冻结前必须回答的 8 个问题

已整理成 [给何远哲师兄的最小确认单](temporal_qcal_profile_inference_advisor_questions_2026-07-22.md)。最关键的是：

1. 实验室真正要联合估计的 `q` 是几何、曲光线、时间对齐还是输运？
2. straight/curved forward 是否有 callable，能否给 JVP/VJP？
3. 独立统计单位是 shot、session、rig 还是只有同一段高速序列？
4. 是否存在 flow-off、已知 target 或重复标定，可估计真实 covariance 与中心偏差？
5. 组内最强基线、hard case 和最终物理指标是什么？

没有这些回答，继续在三个 synthetic rig 上调校准器只会把 development set 越用越薄。

## 8. 复现与证据边界

```bash
cd /path/to/oerf-bishe-dashboard
.venv/bin/python -m learning_labs.temporal_qcal_profile_inference_lab \
  --output-dir learning_labs/results/temporal_qcal_profile_inference_v1
cd learning_labs/results/temporal_qcal_profile_inference_v1
shasum -a 256 -c checksums.sha256
```

正式运行用时 634.70 s，生成 10,500 行 trial、105 行 uncertainty cell、84 行 point cell、3 行导数检查、21 行成本账本和一张四联图。三个 rig 的 bread/score 中心差分相对误差最大分别为 `6.66e-6` 与 `4.77e-6`，数值门通过。

## 9. 不能说的话

- 不能把 97.90% 覆盖写成优于 95%；
- 不能用 post-hoc pooled 方案翻转冻结 NO-GO；
- 不能把 penalized profile contrast 叫严格 likelihood-ratio test；
- 不能把奇偶帧 cross-fit 叫独立实验；
- 不能声称已经优于 DeepONet、FNO、FFNO、NeRIF 或 TDBOST；
- 不能声称新算法、真实三维/4D 重建、跨 rig 泛化、论文成功或突破。

**本轮最扎实的增量是：完整 profile 已把 plug-in 的主要点估计与方差问题收窄到一个可检验的校准粒度问题，同时证明奇偶帧 cross-fit 不是答案。下一步有了明确算法形状，但还没有论文结论。**
