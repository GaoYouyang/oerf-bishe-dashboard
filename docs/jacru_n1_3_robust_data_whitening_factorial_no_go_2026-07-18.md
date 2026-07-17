# JACRU N1.3 robust measurement fidelity factorial: NO-GO

- 日期：2026-07-18
- 状态：`N1_3_ROBUST_DATA_WHITENING_DEVELOPMENT_NO_GO`
- 证据级别：`E0_OPENED_SYNTHETIC_MECHANISM_SCREEN_NO_OOD`
- 结论边界：opened synthetic development；无 independent confirmation、无 OOD、无真实 BOST。

## 1. 真正测试的问题

N1.3 没有再造一个神经网络，而是实现了测量域 Huber-PDHG：

```text
min_x  sum_i rho_delta([W(Ax-y+mu)]_i)
     + lambda sum_j phi_kappa((grad x)_j)
     + ridge/2 ||x||^2
```

这里的数据 Huber 使用单位尾部斜率约定：小残差为 `r^2/(2 delta)`，大残差为
`|r|-delta/2`。它与常见尾部斜率为 `delta` 的 Huber 只差尺度，但正则参数不能直接照搬。

完整因子包括：

- mean policy：zero / estimated；
- whitening：unwhitened / diagonal / isotropic / structured；
- data loss：quadratic / Huber，`delta=0.75/1.345/2.0`；
- edge weight：`0/0.001/0.01/0.1`；
- nominal 与每相机 2%、8 sigma 的 deterministic sparse flow-on outlier stress。

## 2. 规模与判决

| 项目 | 数量 |
|---|---:|
| independent synthetic session | 6 |
| nominal case | 12 |
| nominal + outlier case | 24 |
| candidate | 128 |
| metric row | 3,072 |
| direct Huber-versus-quadratic contrast | 192 |
| passed candidate | 0 |

## 3. 为什么漂亮均值仍是 NO-GO

平均 field 最强的是 `estimated mean + diagonal whitening + Huber delta=2 + lambda=0.1`：

| 指标 | nominal |
|---|---:|
| field gain vs fixed Huber-24 | `+16.913%` |
| H1 gain | `+9.134%` |
| harm rate | `8.33%` |
| worst field gain | `-50.915%` |
| clean reprojection mean / worst vs CGLS | `1.656x / 3.432x` |

这不是接近成功：一个候选可以在 smooth family 上大幅变好，同时在薄界面样本上崩溃。均值不能
覆盖 `-50.9%` 的受害尾部，也不能覆盖 clean renderer 一致性失败。

## 4. Huber 本身只贡献约 0.85%

最好的严格配对 Huber-versus-quadratic 对照是 isotropic whitening、`delta=2`、`lambda=0.1`：

| mean policy | nominal field | outlier field | nominal H1 | clean ratio Huber/quadratic |
|---|---:|---:|---:|---:|
| zero | `+0.852%` | `+0.849%` | `+0.750%` | `0.952x` |
| estimated | `+0.847%` | `+0.848%` | `+0.728%` | `0.942x` |

outlier stress 下的增益几乎没有增加，说明这组 synthetic sparse contamination 没有产生预期的
robust-loss dose response。主要平均信号来自 mean/whitening/regularization 组合，而不是 Huber
本身。

## 5. 方法学限制

1. full whitening 后逐坐标 Huber 依赖 whitening basis；旋转白化坐标可能改变哪些 residual 被裁剪。
2. dense SVD norm setup 不可部署，也没有进入 24F/24AT 效率预算。
3. 只有 6 个独立 session；24 个 field/stress 行不能假装成 24 个独立统计样本。
4. synthetic outlier 不是实测 BOS optical-flow failure。
5. forward-model mismatch 仍需与 sensor covariance 分开。

## 6. 判决

N1.3 只支持一句话：在当前 opened synthetic screen 中，测量域 Huber 相对匹配 quadratic
control 有小幅、稳定但不足的平均收益，无法闭合薄界面尾部与 clean renderer 门。不得训练
geometry-guarded neural correction，不得打开 OOD。

## 7. 重放入口

- runner：[`run_jacru_n1_3_robust_data_whitening.py`](../site_tools/run_jacru_n1_3_robust_data_whitening.py)
- config：[`jacru_n1_3_robust_data_whitening_development_v1.json`](../demo_t16_operator/configs/jacru_n1_3_robust_data_whitening_development_v1.json)
- summary：[`summary.json`](../demo_t16_operator/results/jacru_n1_3_robust_data_whitening_factorial_full2/summary.json)
- direct contrasts：[`factorial_contrast_rows.csv`](../demo_t16_operator/results/jacru_n1_3_robust_data_whitening_factorial_full2/factorial_contrast_rows.csv)
- checksums：[`checksums.sha256`](../demo_t16_operator/results/jacru_n1_3_robust_data_whitening_factorial_full2/checksums.sha256)
