# 真实 PSU 位移与合成 detector-feature 工作域审计

日期：2026-07-17

## 结论

VD0-B 失败后，没有继续增加网络容量，而是直接读取 PSU 公开数据中的真实 `epsu/epsv`，检查真实 detector structure 是否被当前 synthetic development 覆盖。结果是否定的：

- validation 到 risk-train 中心的稳健距离中位数：**1.646**；
- calibration：**1.350**；
- 真实 PSU camera subsets：**3.076**；
- 真实到最近 train row 的距离中位数：**1.873**，validation/calibration 为 **0.883 / 0.755**；
- 130/130 真实 camera subsets 至少有一个 informative feature 落在 train 95% 包络外；
- 平均每个 informative feature 有 **23.99%** 的真实 subset 落在包络外，validation/calibration 只有 **2.72% / 7.44%**。

当前决定：

```text
FLOW_OFF_REFERENCE_REPEATS_OR_GRAPH_CORRELATED_NOISE_MODEL
BEFORE ANY SET ENCODER
```

## 1. 审计如何避免伪样本

本地 9-view PSU 数据对应 **1 个真实物理流场**。为了检查 active-camera 组合敏感性，枚举了 6-9 个 active views 的全部组合，共 130 个 masks：

```text
C(9,6) + C(9,7) + C(9,8) + C(9,9) = 130
```

这 130 个 rows 不是 130 个独立实验，报告和 JSON 都明确保留：

```text
real_physical_field_count = 1
camera_subsets_are_independent_physical_fields = false
```

因此不能据此计算论文意义上的实验置信区间或泛化误差。

## 2. 为什么使用 per-view RMS normalization

真实数据当前没有 flow-off repeats，不能把某个估计量冒充 measured noise sigma。为比较空间结构，本轮对真实和 synthetic 都做：

```text
each view displacement / each view vector RMS
```

这只消除每台相机的整体幅度，使邻域对比、局部 Jacobian、front concentration、anisotropy 和方向结构可比。它不是 noise whitening，也不能替代真实协方差。

## 3. 差异主要在哪里

真实 subset 在以下六项中几乎全部超出 synthetic train 95% 范围：

| feature | synthetic q2.5-q97.5 | real subset median | outside fraction |
|---|---:|---:|---:|
| neighbor contrast max | 0.338-0.994 | 1.688 | 100% |
| neighbor contrast mean | 0.299-0.897 | 1.544 | 100% |
| neighbor contrast min | 0.238-0.783 | 1.412 | 100% |
| log local Jacobian min | 1.633-2.610 | 3.257 | 100% |
| log local Jacobian mean | 1.869-2.851 | 3.392 | 100% |
| log local Jacobian max | 2.068-3.034 | 3.637 | 100% |

最直接的解释是：真实 PSU 位移在 detector 邻域上的变化远比当前 analytic morphology + sparse-ray synthetic observation 更尖锐或更不规则。

但这里不能唯一归因于真实 shock，因为还混合了：

- optical-flow measurement noise；
- dot/background registration error；
- mask boundary；
- camera-specific bias；
- real shock/front 高频结构；
- finite aperture 和 depth-of-field；
- 当前 synthetic ray sampling 与真实 dense image processing 的差异。

所以本轮能证明“工作域不匹配”，不能证明“某一个物理机制就是原因”。

## 4. 对算法路线的影响

VD0-B 的 train stress 正信号现在必须降级解释。它更可能表示 route 学会了 synthetic generator 内部的结构，而不是学会了真实 BOST camera risk。

在工作域明显不匹配时：

- DeepSets 会更有能力拟合错误分布；
- FNO 不会自动创造缺失的真实 detector statistics；
- 增加 feature width 或 attention 不能修复 noise/forward mismatch；
- 即使 validation/calibration 都来自同一 generator，漂亮数字也不构成真实迁移。

因此当前最有论文价值的方向不是“再设计一个更深的 router”，而是：

> 构建 detector-geometry-consistent、由 flow-off covariance 约束的 BOST synthetic-to-real interface，并研究工作域偏移如何影响 front reconstruction 和 solver selection。

这比泛泛的 domain adaptation 更贴近何远哲方向，因为它直接作用于 NeRIF/TDBOST 的输入位移、camera projection 和 front fidelity。

## 5. 现在需要师兄提供的最小数据

不需要立刻提供完整实验库。最小包：

1. 同一设置的 20-50 帧 flow-off/reference repeats；
2. 每台相机的 ROI/mask 与 pixel coordinates；
3. exposure、f-number、focus/background distance；
4. 若已经计算 optical flow，保留 raw `u/v`，不要先全局平滑；
5. 一帧 flow-on + 对应 reference；
6. 一台不参与重建的 held-out camera。

有这些数据后可以先完成：

- graph covariance；
- row/column or low-frequency bias；
- `u-v` cross-correlation；
- temporal drift；
- camera-specific spectrum；
- real front-feature envelope。

仍不需要租 GPU。

## 6. 下一轮门槛

只有新的 graph-correlated synthetic generator 同时满足下面条件，才重新打开小型 set encoder：

1. validation/calibration 仍在 train 工作域内；
2. 至少一个真实 flow-off/session 的大部分 scale-free descriptors 被覆盖；
3. real flow-on 不再在 neighbor contrast/Jacobian 上系统性超界；
4. 路由主报 held-out camera/front risk，而不是只报 synthetic field mean；
5. 同一动作库、同一 calls、同一拒答规则。

在此之前，`route_training_authorized = false`。

## 7. 可复现入口

- 审计脚本：`site_tools/audit_psu_b0_real_detector_feature_domain.py`
- 单元测试：`site_tools/test_audit_psu_b0_real_detector_feature_domain.py`
- 公开 JSON：`docs/psu_b0_real_detector_feature_domain_public_summary.json`
- 图：`demo_t16_operator/results/psu_b0_real_detector_feature_domain/psu_b0_real_detector_feature_domain_figure.png`

本机：

```text
wall time: 1.15 s
peak RSS: 397 MB
real physical fields: 1
deterministic camera subsets: 130
reconstruction: none
training: none
```
