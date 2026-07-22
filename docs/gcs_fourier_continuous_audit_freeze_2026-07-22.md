# GCS-Fourier 连续 renderer 审计冻结

日期：2026-07-22  
状态：**一个 cell 已观察；其余 7 个 paired units 与 14 个 dense angles 尚未运行**

## 为什么必须追加这一步

第一轮用中心差分 renderer 同时训练和测试，因此 test-view 指标仍可能看不见中心差分自己的近零空间。冻结代表 cell 的 post-open 探针出现了明确分离：

| 模型 | test central relative-L2 | test AD relative-L2 |
|---|---:|---:|
| Fourier-low | 0.104907 | 0.155631 |
| Fourier-high | 0.104886 | 0.802766 |

高频模型在离散 renderer 里几乎不输，却在同一连续网络场的自动微分导数下明显失真。这支持一个机制假设：**训练算子能够被亚网格振荡欺骗；held-out angle 仍使用同一离散梯度时，并不是真正独立的物理检查。**

但该 cell 已经看过，不能计入后续主门。

## 尚未打开的审计

在 0 到 180 度之间每 7.5 度取一角，剔除原 train/development/test 的十个角度，留下 14 个全新 dense-audit angles：

```text
7.5, 22.5, 37.5, 52.5, 67.5, 75, 82.5,
97.5, 112.5, 127.5, 142.5, 157.5, 165, 172.5
```

每个冻结网络用四种梯度 renderer 评估：`FD(h)`、`FD(h/2)`、`FD(h/4)` 与 AD。主 endpoint 是 dense unseen angles 上 AD clean relative-L2 的 `high - low`。

## 七单元门

排除 `wrinkled_density_interface / noise 0.08` 整个 paired unit 后，其余 7 个单位必须同时满足：

1. 重新训练所得原 test-central 指标与首轮记录最大绝对差不超过 `1e-5`；
2. high 的 dense-AD 在 7/7 单元都差于 low；
3. dense-AD `high-low` 中位数至少 `0.05`；
4. development GCS 与 dense-AD delta 的 Spearman rho 至少 `0.5`；
5. high 的 dense-AD 在 7/7 单元都差于自己的 dense-FD(h)；
6. 上述 `AD-FD(h)` 差的中位数至少 `0.05`。

通过只说明“离散投影一致性是值得进入正则化算法设计的真实 synthetic 机制”，不授权算法成功；失败则关闭 GCS/一致性主线。

冻结配置：[gcs_fourier_continuous_audit_v1.json](../demo_t16_operator/configs/gcs_fourier_continuous_audit_v1.json)
