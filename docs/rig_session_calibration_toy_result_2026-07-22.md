# rig/session 层级校准 toy 结果：能学到尺度，也能在关系翻转时惨败

> 日期：2026-07-22
> 状态：`EDUCATIONAL_MECHANISM_SIGNAL_COMPLETE`
> 证据等级：`EDUCATIONAL_SYNTHETIC_CLUSTER_CALIBRATION_MECHANISM_ONLY`
> 突破监测：**否**

## 一句话结论

当部署可见特征确实解释 score 异方差时，低容量 log-ridge normalizer 能显著修复 hard-rig coverage 并缩短区间；当难度隐藏时它与 equal-rig 基线几乎相同；当 fit/calibration 与 evaluation 的条件关系反转时，它会在训练拟合看似很好的同时严重失效。因此下一候选必须是“物理可解释 normalizer + 独立 acquisition 校准 + support/relationship gate + 保守 fallback”，不能是直接输出答案的黑盒网络。

这只是一个有正反例的统计教学实验，不是 BOST forward、三维重建、新算法或跨 rig 泛化证据。

## 1. 先看最重要的四个数字

| 冻结检查 | 结果 | 判定 |
|---|---:|---|
| observable：log-ridge 相对 frame-pooled 的 hard coverage 增量 | `+36.97` 个百分点 | 教学正例通过 |
| observable：log-ridge hard coverage | `93.76%` | 通过预注册 `>=90%` |
| observable：log-ridge / equal-rig 中位半径比 | `0.7596`，即缩短 `24.04%` | 通过 |
| sign-flip：log-ridge hard coverage | `11.23%` | 反例按预期失败 |

oracle 在三个情景的 rig-mean coverage 为 `95.07% / 95.01% / 94.71%`，说明 score 生成、真尺度正规化和 Monte Carlo 规模没有出现明显实现异常。正式产物 checksum 全部通过。

## 2. 为什么逐 frame pooled 会骗人

observable 情景故意让 easy rig 拥有更多 frame。逐 frame pooling 得到：

- observation-weighted coverage：`92.39%`；
- equal-rig mean coverage：`87.34%`；
- hard-quartile coverage：`56.79%`；
- rig coverage p10：`55.99%`。

这四个数描述的是同一方法。若只报所有 frame 的 pooled mean，会把最重要的 hard-rig 失败抬高约 35.6 个百分点。真实高速 BOST 中，同一 sequence 的 frame 共享 camera、geometry、background 和 flow history，所以正式评估必须以师兄确认的 acquisition/session 为外层单位。

equal-rig CDF 把 observable 的 hard coverage 提到 `72.98%`，但仍未达到 95%；worst-rig 阈值则把 rig mean 推到 `99.85%`，代价是中位半径从 `4.60` 膨胀到 `9.49`。这再次说明 coverage 和 width 必须一起看。

## 3. 学尺度什么时候有用

在 `observable_scale` 中，真尺度主要由部署可见 `z` 决定。fit-only log-ridge 的：

- slope：`0.9133`；
- fit `R^2`：`0.9692`；
- evaluation 真关系 slope：`0.9052`。

随后用独立 calibration rigs 对 normalized score 取 equal-rig threshold。最终得到：

| 方法 | rig mean | hard quartile | rig p10 | rigs `>=90%` | 中位半径 |
|---|---:|---:|---:|---:|---:|
| frame pooled | 87.34% | 56.79% | 55.99% | 68.25% | 3.9086 |
| equal rig | 92.77% | 72.98% | 73.22% | 81.00% | 4.5982 |
| worst rig | 99.85% | 99.42% | 100.00% | 99.75% | 9.4855 |
| log-ridge normalized | **94.87%** | **93.76%** | 89.56% | 89.25% | **3.4928** |
| oracle normalized | 95.07% | 95.01% | 92.63% | 99.75% | 3.4057 |

这里可以说“正确可见特征携带了有用尺度信息”。不能说 log-ridge 是新算法：异方差正规化、locally varying interval、group/hierarchical conformal 都有大量先例；toy 的尺度律还是我们人为写进去的。

## 4. 没信息时，模型没有凭空变聪明

`hidden_scale` 中真难度只由不可见 `u` 决定。log-ridge 的 fit `R^2` 只有 `1.39%`，slope 为 `-0.1132`。它的 hard coverage 为 `68.70%`，equal-rig 为 `68.88%`，只差 `0.19` 个百分点；中位半径也几乎相同：`4.3196` 对 `4.3137`。

这条负对照很重要：如果在真实数据中，geometry/noise/信息谱摘要不能预测 score scale，就应停止加深网络。MLP、DeepSets 或 FNO 不会凭空创造不存在的部署信息。

## 5. 最危险的情景：训练拟合很好，部署仍崩溃

`sign_flip_shift` 的 fit/calibration 关系 slope 约为正 `0.9`，evaluation 则为负 `0.9`。log-ridge 在 fit 上仍有 `R^2=95.21%`，学到 slope `+0.8838`；evaluation 真 slope 却是 `-0.8887`。

结果是：

- rig-mean coverage：`69.96%`；
- hard-quartile coverage：`11.23%`；
- rig p10：`2.77%`；
- 只有 `54.50%` 的 rig 达到 90% coverage。

这不是模型容量不足，而是条件关系变了。独立 calibration 只能保护与 calibration exchangeable 的单位，不能把 OOD 自动变成 IID。一个在训练集 `R^2` 很高的网络，同样可能给出很窄而错误的置信域。

## 6. 由 toy 推出的候选算法形状

工作名暂称 **guarded acquisition-conditioned profile operator**，只表示待验证结构，不主张新颖性：

```text
geometry / noise / information summaries z_g
                |
                v
    low-capacity scale + tangent proposal
                |
        +-------+----------------+
        | support/relationship gate |
        +-------+----------------+
          inside |          outside
                 v             v
  normalized calibration   conservative A0 fallback
                 |             |
                 +------v------+
                        |
          1--2 exact profile corrections
                        |
   held-out view + field/gradient + PIV endpoint
```

### 经典部分，必须先固定

1. full-profile / exact-score reference；
2. frame-pooled、equal-rig、worst-rig 和简单 shrinkage；
3. log-ridge / isotonic / GAM scale model；
4. 明确的 acquisition-level split 和 fallback；
5. A/A-transpose 或 forward/JVP/VJP 调用、wall time、显存账本。

### 真正值得让算子学习做的部分

1. 从多视角 residual、geometry 和 temporal context 预测低秩 nuisance/transport tangent；
2. 给 iterative profile 提供 warm start 或预条件，不直接替代 solver；
3. 预测 heteroscedastic scale，但最终 inflation 必须由独立 session calibration 决定；
4. 给出 support score / shift alarm，门外自动退回 A0；
5. 用 held-out camera/time、gradient/front 和 PIV velocity correction 共同决定接受。

### 论文创新必须超过的门槛

- 同一调用预算下优于 full-profile fixed warm start，而不只是优于弱 DeepONet/FNO；
- 在 unseen acquisition 的 rig-mean、p10、hard quartile 都成立；
- 对 curved/straight mismatch、geometry drift、covariance shift 和 view dropout 分别消融；
- normalizer 输给 log-ridge 时停止，不用更深网络挽救叙事；
- support 外 fallback 的 coverage/width 与误拒率都单独报告；
- 至少一个真实物理终点，不把 held-out reprojection 当三维 truth。

## 7. 现在最值得问何远哲师兄的六件事

1. 什么是可认为独立的 shot/session/acquisition？每种工况大约有多少个？
2. 部署时可见的 geometry、noise covariance、timestamp、view-angle 和 flow-off 摘要有哪些？
3. 是否有 straight 与 curved 两层 callable，以及 JVP/VJP 或至少可差分参数？
4. 当前最认可的 baseline 和 hard case 是什么？按什么命令得到结果？
5. 主终点是 field、gradient/front、held-out view，还是 PIV velocity correction？
6. 遇到新 burner、相机重标定或新折射强度时，实验上怎样识别 support 外样本？

在这六项闭合前，不能启动休眠 fresh BOST audit，也不应租大卡训练 FNO。

## 8. 复现与证据文件

```bash
.venv/bin/python -m pytest -q learning_labs/test_rig_session_calibration_toy.py
.venv/bin/python -m learning_labs.rig_session_calibration_toy
cd learning_labs/results/rig_session_calibration_toy_v1
shasum -a 256 -c checksums.sha256
```

- [冻结协议](rig_session_calibration_toy_protocol_2026-07-22.md)
- [基础学习路线](rig_session_calibration_learning_route_2026-07-22.md)
- [fresh 激活缺口审计](temporal_qcal_fresh_activation_audit_2026-07-22.md)
- [机器报告](../learning_labs/results/rig_session_calibration_toy_v1/report.json)
- [15 行方法表](../learning_labs/results/rig_session_calibration_toy_v1/method_summary.csv)
- [6,000 行逐 rig 表](../learning_labs/results/rig_session_calibration_toy_v1/rig_metrics.csv)
- [四联图](../learning_labs/results/rig_session_calibration_toy_v1/rig_session_calibration_toy.png)
- [源码](../learning_labs/rig_session_calibration_toy.py)
- [7 项定向测试](../learning_labs/test_rig_session_calibration_toy.py)

## 9. 一级来源边界

- Lei et al., [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116)：marginal coverage 与 locally varying width。
- Dunn, Wasserman, and Ramdas, [Distribution-Free Prediction Sets for Two-Layer Hierarchical Models](https://arxiv.org/abs/1809.07441)：两层独立单位与组内重复。
- Barber et al., [The limits of distribution-free conditional predictive inference](https://arxiv.org/abs/1903.04684)：一般条件下 exact conditional coverage 的限制。
- Barber et al., [Conformal Prediction Beyond Exchangeability](https://doi.org/10.1214/23-AOS2276)：交换性破坏后的加权与保证边界。
- Romano et al., [Conformalized Quantile Regression](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)：异方差 shape 与独立 calibration。
- Moya et al., [Conformalized-DeepONet](https://doi.org/10.1016/j.physd.2024.134418)：operator 输出的 conformal UQ 邻近基线，不提供 BOST 物理正确性。

**最高允许表述：层级校准 toy 成功复现了一个适用机制、一个无信息负对照和一个关系漂移反例，并把下一候选收窄为带 support gate 与 fallback 的 acquisition-conditioned profile operator。新算法、真实 BOST、三维/4D 重建、跨 rig 泛化、论文成功和突破仍为 0。**
