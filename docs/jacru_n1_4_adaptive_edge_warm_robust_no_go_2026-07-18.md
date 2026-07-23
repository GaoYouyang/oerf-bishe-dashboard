# JACRU N1.4 adaptive-edge warm robust reconstruction: NO-GO

- 日期：2026-07-18
- 状态：`N1_4_ADAPTIVE_EDGE_WARM_ROBUST_DEVELOPMENT_NO_GO`
- 证据级别：`E0_OPENED_SYNTHETIC_MECHANISM_SCREEN_NO_OOD`
- 正式使用结果包：`factorial_full2`，不是缺少 matched zero-start controls 的历史 `full1`。

## 1. 假设与公平预算

假设是：先用 12 步 structured-whitened CGLS 获得粗场，根据其梯度生成固定 edge weight，再用
12 步 Huber-data PDHG 细化，或许能保护薄反应前沿。

```text
w(r) = w_min + (1-w_min) / (1 + (|grad x_warm(r)| / t_q)^p)
```

所有候选严格使用 24F/24AT：

- adaptive / uniform warm：CGLS-12 + PDHG-12；
- zero-start：PDHG-24；
- `lambda=0.05/0.1/0.2` 均有 matching uniform warm 与 matching zero-start control；
- edge map 只读同 case warm field，不读 truth、family 或 clean projection。

truth 与 clean projection **没有进入 reconstruction API**，但确实用于 opened-development diagnostic
gate selection；本包没有 independent confirmation 或 OOD。

## 2. v1.1 完整规模

| 项目 | 数量 |
|---|---:|
| independent synthetic session | 6 |
| nominal case | 12 |
| nominal + deterministic outlier case | 24 |
| adaptive candidate | 27 |
| uniform warm control | 3 |
| zero-start control | 3 |
| metric row | 792 |
| passed candidate | 0 |

全部 13 个结果文件 checksum 通过；seed-family 全集和分段调用预算均由 runner fail closed 校验。

## 3. 三个代表点

| 候选 | field gain | H1 gain | harm | worst | clean mean / worst |
|---|---:|---:|---:|---:|---:|
| zero-start 24, `lambda=0.2` | `+28.811%` | `+21.847%` | `8.33%` | `-15.009%` | `1.532x / 3.002x` |
| uniform warm, `lambda=0.2` | `+17.928%` | `+11.772%` | `8.33%` | `-2.755%` | `1.775x / 4.645x` |
| best adaptive, `lambda=0.2,q=.95,wmin=.5` | `+15.645%` | `+9.517%` | `8.33%` | `-4.573%` | `1.749x / 4.600x` |

三者都失败：前者平均最好却严重伤害已知薄界面，后两者保护部分尾部却把 independent-clean
一致性进一步拉坏。

## 4. 自适应边缘没有机制增益

27 个 adaptive candidate 与相同 `lambda` 的 uniform warm control 逐 case 配对。最好的 adaptive
仍然：

- mean field 相对 uniform：`-0.944%`；
- worst paired field：`-1.705%`；
- mean H1：`-0.859%`。

也就是说，在该 screen 中没有任何一组 adaptive 参数优于 matching uniform。独立审计对全部
adaptive 的总体比较同样得到 field/H1 均值和最坏值均为负。

## 5. warm start 显示的是冲突，不是胜利

在 `lambda=0.2` 下，warm 相对 zero-start：

- 全体 nominal mean field：`-19.170%`；
- worst paired field：`-39.187%`；
- mean H1：`-17.057%`；
- 但 `base_seed=2113/single_interface` 局部 field 改善 `+10.655%`。

所以 warm start 的确改变了薄界面偏置，却以大范围平均性能和 clean consistency 为代价。它不是
一个可泛化的修复，而是进一步证明：同一低保真观测模型对 smooth 与 sharp-interface family 的
解释发生冲突。

## 6. 为什么主线转向 forward mismatch

所有 33 个候选都同时失败于 harm rate、clean mean、clean worst 和 known-interface safety。
继续调 `lambda/q/wmin` 只是在同一冲突曲线上移动。下一候选应问：

> 低保真 thin-ray/voxel forward 是否把有限孔径、弯曲光路、标定误差或离散误差系统性地解释成
> 三维场结构？

这就是 N1.5 的条件化 approximation-error 路线。它先修观测模型，再考虑把小型网络用于预测
低秩 discrepancy statistics；不再直接让网络猜三维场。

## 7. 重放入口

- runner：[`run_jacru_n1_4_adaptive_edge_warm_robust.py`](../site_tools/run_jacru_n1_4_adaptive_edge_warm_robust.py)
- v1.1 config：[`jacru_n1_4_adaptive_edge_warm_robust_development_v1_1.json`](../demo_t16_operator/configs/jacru_n1_4_adaptive_edge_warm_robust_development_v1_1.json)
- summary：[`summary.json`](../demo_t16_operator/results/jacru_n1_4_adaptive_edge_warm_robust_factorial_full2/summary.json)
- diagnostic：[`diagnostic.png`](../demo_t16_operator/results/jacru_n1_4_adaptive_edge_warm_robust_factorial_full2/diagnostic.png)
- checksums：[`checksums.sha256`](../demo_t16_operator/results/jacru_n1_4_adaptive_edge_warm_robust_factorial_full2/checksums.sha256)
