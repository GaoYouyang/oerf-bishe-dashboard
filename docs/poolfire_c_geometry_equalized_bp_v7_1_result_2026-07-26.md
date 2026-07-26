# PoolFire C 路线 v7.1：固定几何均衡仍不足以稳定定位

## 一句话结论

`BP_eq = D^{-1}A^T y` 修正了部分几何偏置，尤其把 p45-s05 的质心
L∞ p90 从 `1.847` 降到 `1.286` 体素，但六条轨迹没有全部通过冻结 T0 门。
五条轨迹的整数 shift 完全一致率仍低于 `75%`，p22-s03 的质心 p90 还出现材料性
恶化。因此正式状态是 `FAIL_T0_GEOMETRY_EQUALIZED_BP_PROXY`，Jacobi 均衡路线停止。

## 先说明 v7 为什么没有结果

旧 v7 的唯一一次 sealed 运行在读取 606 帧 observation 前停止。PoolFire 原始坐标轴
按降序保存，v7 手工重建几何时遗漏了 v6 已有的“反转并均匀化为升序坐标”步骤。
这属于实施失败，不是科学结果。

v7.1 在结果前单独冻结了修复附录，只允许恢复同一坐标规范化；均衡公式、relative
floor、六条轨迹、T0 阈值、raw 对照、成本账和失败动作均不得改变。正式 runner 与
独立 validator 分别实现坐标规范化，并都复现了六条轨迹绑定的同一几何身份。

## 逐轨迹结果

| 轨迹 | raw p90 | equalized p90 | raw exact | equalized exact | equalized within-one |
|---|---:|---:|---:|---:|---:|
| p33-s01 | `0.516` | `0.501` | `37.62%` | `36.63%` | `100%` |
| p45-s05 | `1.847` | `1.286` | `14.85%` | `13.86%` | `92.08%` |
| p58-s03 | `0.628` | `0.625` | `57.43%` | `61.39%` | `100%` |
| p14-s05 | `0.962` | `0.748` | `31.68%` | `53.47%` | `99.01%` |
| p22-s03 | `0.359` | `0.499` | `85.15%` | `86.14%` | `100%` |
| p14-s01 validation | `0.304` | `0.317` | `70.30%` | `69.31%` | `100%` |

冻结门要求每条轨迹同时满足 p50≤0.5、p90≤1、worst≤2、exact≥75%、
within-one≥95%、误差大于两体素的帧数为 0。p45-s05 同时失败 p50、p90、exact
和 within-one；p33、p58、p14-s05、p14-s01 失败 exact。p22-s03 虽通过绝对门，
但 equalized p90 相对 raw 从 `0.359` 升到 `0.499`，触发材料性 harm 标记。

## 独立验证与成本边界

- 独立 validator 不导入正式 equalizer 或 runner。
- 它从 606 帧 observation 重新生成 raw BP，再独立解析计算 `diag(A^T A)` 与
  equalized BP。
- 所有候选数组最大绝对差为 `0`，raw BP 最大绝对差也为 `0`。
- 几何审计额外成本为 `2A + 1A^T`，只属于离线证据，不属于部署。
- 单帧部署仍是一次 `A^T` 加 `8192` 次逐体素乘法。
- 本轮没有证明 A/A^T 减少、wall-time 加速或内存下降。

## 为什么这个负结果有用

固定 `D^{-1}` 只能校正与几何覆盖相关、且跨样本不变的逐体素尺度偏置。它能明显
改善 p45 和 p14-s05，说明几何灵敏度确实是偏差来源之一；但 exact shift 仍普遍
失败，说明剩余误差不是单一固定对角缩放能解释的。可能还包含视角间能量不平衡、
场形态变化、边界截断和 observation-dependent 偏置。

这排除了“固定 Jacobi 均衡后直接做统一整数平移”的路线，但没有排除全场 warm
initializer。

## 下一有效门

下一候选必须在结果前预注册，并且只用 fit trajectories 学习、部署时只看
observation-visible 特征。最小方案是一个低容量仿射质心校准：

```text
features = [raw/equalized BP centroid, per-view energy, view balance,
            low-order spectral moments]
calibrated_centroid = B features + b
```

它首先只回答“能否在未参与拟合的 p14-s01 上稳定定位”，不输出三维场，也不训练
FNO/UNO/DeepONet。若留出轨迹仍不能通过同一 T0 与 harm 门，就停止 shifted-POD
定位支线，回到 observation→full-field warm initializer。

**突破监测：没有算法突破。** 新增的是经过 sealed 单次运行和独立复算的可信负结果。
`neural_training_authorized=false`，两条 untouched test 和 p22 stopping validation
仍未打开，当前也不是真实 BOST。
