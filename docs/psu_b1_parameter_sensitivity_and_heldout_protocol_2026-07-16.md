# PSU B1 参数敏感性与 70 视角留出协议

日期：2026-07-16

当前状态：

- B1 的 axis、angle、vertex 12 组预声明真实九视角敏感性已经完成；
- 70 个公开视角已被互斥划分为 support、development 和三类 final audit；
- 最终 18 视角同相机未见旋转评分接口、指标和成功门槛已经写死；
- audit payload 仍封存，没有运行三维重建，也没有算法优越性结论。

## 1. 为什么这一轮不是“调 cone 参数”

B1 的定义是 `forward box ∩ one-nappe cone`。此前已经证明解析公式可以稳定执行，但不能证明公开参数

```text
vertex = (0.06, 0.015, 0) m
axis = (-1, -0.1, 0)
half angle = 25 deg
```

就是正确物理支持域。

因此本轮在看真实敏感性结果前冻结 12 个变体，只回答三个问题：

1. axis 的符号是不是纯记号，还是会改变实际积分域？
2. 15、20、25、30、35 度会怎样改变 active 测量支持？
3. vertex 沿 axis、横向法向和 z 方向移动 5 mm 时，结果是否近似不变？

这些变体不是参数搜索。5 mm 约为当前最大 box 跨度的 2.27%，是粗应力测试，不代表标定误差分布。当前数据不得用于选择“最优” axis、angle 或 vertex。

预注册配置：

- 状态：`FROZEN_BEFORE_REAL_SCORING`
- 冻结时间：`2026-07-16T08:21:40Z`
- 选择政策：`NO_PARAMETER_SELECTION_FROM_THIS_AUDIT_PHYSICAL_SELECTION_AND_ALGORITHM_SUPERIORITY_LOCKED`
- 公开摘要 SHA-256：`b401dbfc321de9cf5193b769e2a40930ab87f64d713a9a2b8d341d2a55daa97a`

## 2. 九视角真实结果

下表中的 active hit 指当前 active 中心线仍命中候选域的比例；active support IoU 比较候选和 25 度公开参考在每条 active ray 上的区间重合；all path/B0 是候选总路径占保守 B0 前向 box 总路径的比例。

| 变体 | active hit | active support IoU | all path / B0 | active hit 变化 |
|---|---:|---:|---:|---:|
| 公开参考 25° | 99.9873% | 100.0000% | 15.1880% | 基准 |
| axis 反号 | 0.0000% | 0.0000% | 0.0085% | 丢失 10,627,472 |
| 15° | 48.7807% | 25.2749% | 5.0098% | 丢失 5,442,654 |
| 20° | 84.8418% | 58.0930% | 9.2568% | 丢失 1,609,785 |
| 30° | 99.9997% | 73.1381% | 22.6182% | 新增 1,321 |
| 35° | 100.0000% | 57.4330% | 31.1432% | 新增 1,350 |
| vertex 沿 axis +5 mm | 99.7887% | 92.1015% | 13.7492% | 丢失 21,110 |
| vertex 沿 axis -5 mm | 100.0000% | 93.0639% | 16.7115% | 新增 1,350 |
| vertex 横向 +5 mm | 99.2949% | 90.0553% | 15.3168% | 丢失 73,593 |
| vertex 横向 -5 mm | 99.6775% | 89.7649% | 15.0279% | 丢失 34,279，新增 1,350 |
| vertex z +5 mm | 99.0820% | 89.5837% | 15.2043% | 丢失 97,571，新增 1,350 |
| vertex z -5 mm | 98.7844% | 89.3126% | 15.1356% | 丢失 127,855 |

## 3. 这组数字真正说明什么

### 3.1 axis 符号具有实质语义

axis 反号后，公开参考命中的 10,627,472 条 active 中心线全部丢失，active support IoU 为 0。它不是可以随手统一符号的无害约定。若组内代码、CAD 或相机坐标系对 axis 正方向解释不同，B1 会选择几乎完全相反的空间。

### 3.2 angle 不是温和超参数

15 度只保留 48.78% active 中心线，20 度保留 84.84%。30、35 度虽然几乎保留全部 active hit，但 support IoU 仍只有 73.14% 和 57.43%，总路径分别增至 B0 的 22.62% 和 31.14%。因此“active hit 接近 100%”不能证明积分区间接近。

### 3.3 vertex 粗扰动会持续改变空间支持

六个 5 mm 应力测试的 active support IoU 为 89.31% 到 93.06%，all-ray support IoU 为 87.52% 到 90.88%。这说明 B1 对 vertex 不是完全脆弱，但也绝不是可忽略。z 负向 5 mm 会丢失 127,855 条公开参考 active hits，是当前六个 vertex 变体中最强的 hit 变化。

### 3.4 当前不能从这些结果选参数

公开参考并没有因为“命中更多”就自动成为物理真值；扩大 angle 也不能因为 active hit 更高而获胜。选择必须来自以下至少一项外部证据：

- CAD、标定靶或 visual hull 对 axis/vertex/angle 的独立约束；
- development rotation 的未见视角重投影；
- flow-off repeats 给出的实际测量重复性；
- calibration covariance 下的稳健性。

因此当前默认仍是最小假设的 **B0 + fixed-denominator aperture indicator**。B1 及其参数只作预声明消融。

## 4. 70 个视角怎样被完整封存

[PSU 数据论文](https://arxiv.org/abs/2508.17120)和作者公开脚本共同表明，完整数据是 7 台相机乘 10 次模型旋转，共 70 个视角。作者九视角支持集是 camera 2、3、4 在 rotation 0、50、90 的笛卡尔积。

冻结协议将 70 个视角恰好覆盖一次：

| split | cameras | rotations | 视角数 | 用途 |
|---|---|---|---:|---|
| support reconstruction | 2,3,4 | 0,50,90 | 9 | 重建、优化和 support residual |
| development rotation 40 | 1-7 | 40 | 7 | 唯一允许模型选择和停止的外部 run |
| primary audit | 2,3,4 | 10,20,30,60,70,80 | 18 | 六个未见旋转独立块 |
| camera audit | 1,5,6,7 | 0,50,90 | 12 | 同运行、未见相机布局 |
| joint audit | 1,5,6,7 | 10,20,30,60,70,80 | 24 | 相机和旋转同时未见 |

验证器已确认：

- 70/70 视角被覆盖且无重叠；
- support 精确复现作者九视角；
- development 是完整 rotation 40 run；
- primary audit 是六个未见旋转块；
- final audit 禁止参与模型选择、停止和参数选择。

协议 SHA-256：`0c77d7e427a15eec1a5a6b87b1c31ca875d0c93fd0a084191c9d3180f3b1b774`

## 5. 最终主终点为什么按 rotation block 判

同一张图里的数百万像素不是数百万个独立实验。相机共享同一次旋转、同一流动条件、同一标定和处理链，因此最终实验单位固定为 `MODEL_ROTATION_RUN_BLOCK`。

主终点对每个未见 rotation 汇总 camera 2、3、4 的 active displacement：

```text
relative L2 =
sqrt(sum_active((u_hat-u)^2 + (v_hat-v)^2))
/
sqrt(sum_active(u^2 + v^2))
```

候选必须在 10、20、30、60、70、80 度六个块中全部低于冻结 B0 reference。预声明单侧 exact sign probability 为

```text
p = 1 / 2^6 = 1/64 = 0.015625
```

不能 bootstrap 像素来制造很小的 p 值。

## 6. 五道成功门

即使六个块全部降低 residual，也还不能直接叫成功：

1. **primary gate**：六个 rotation block 的 active relative L2 全部降低；
2. **practical gate**：六块 active RMSE 降幅中位数必须大于 development flow-off repeatability floor；
3. **tail gate**：任何 primary block 的 p95 都不能比 B0 恶化超过 repeatability floor；
4. **ambient gate**：环境区预测位移 RMS 不能增加；
5. **calibration gate**：冻结的标定扰动审计仍须通过。

若缺 flow-off repeats 或 calibration covariance，最多只能报告 held-out image-space consistency，不能报告实验优越性。

## 7. 已冻结的预测包接口

每个 final audit 视角使用一个只含数值数组的 NPZ：

```text
measured_uv_px   (N,2)
candidate_uv_px  (N,2)
baseline_uv_px   (N,2)
active_mask      (N,)
ambient_mask     (N,)
front_band_mask  (N,)
```

manifest 必须在开 final audit 前绑定：

- protocol SHA-256；
- candidate/baseline checkpoint SHA-256；
- candidate/baseline config SHA-256；
- 18 个 NPZ 的逐文件 SHA-256；
- development repeatability floor；
- calibration perturbation gate。

评分器拒绝：

- 路径逃逸、文件哈希漂移或重复视角；
- 不完整的 18 视角网格；
- 非布尔或重叠 mask；
- front-band 不属于 active；
- NaN/Inf；
- 缺失 repeatability 或 calibration 时的优越性主张。

即使所有 image-space gate 通过，报告仍明确：

```text
field_l2_available = false
unique_3d_truth_established = false
authorized = HELD_OUT_IMAGE_SPACE_CONSISTENCY_ONLY
```

这是因为真实 PSU 数据没有独立三维密度真值。更低重投影误差不等于唯一恢复了真实三维场。

## 8. 对算法设计的直接启发

### 稳妥候选：B0-reference DC-NeRIF

- B0 + fixed denominator 作为主 forward；
- B1 25 度、20/30 度和 vertex stress 只作冻结消融；
- 同样的 support、ray samples、regularization budget 和 stopping；
- development rotation 40 只开一次做模型选择；
- final audit 只比较一个冻结候选和一个冻结 B0 reference。

这条线最适合作为本科毕设主干，因为失败时仍能给出完整的物理和评测结论。

### 高风险候选：Domain-conditioned RayKernel-DCO

只有当 development 证明 B0 的 residual 仍显著高于 repeatability floor，才学习位置相关的 operator correction。输入至少包括：

- ray origin/direction；
- camera 与 rotation；
- aperture basis/radius；
- B0/B1 support length 和 boundary distance；
- calibration uncertainty summary。

它必须和同预算 Broyden、multisecant、TSVD、full-matrix ridge、Learned ReSeSOp 及 solver+correction 对比，并共享 forward/adjoint call 账本。

### 不建议的捷径

- 直接在 9 个 support views 上比较 FNO、DeepONet 和自有网络；
- 用 final audit 挑 angle、mask threshold 或 early stopping；
- 把 70 个视角随机按像素或 ray 打散；
- 只报 pooled residual，不报六个 rotation block；
- 把更低 residual 写成 field-L2 或真实密度优越性。

## 9. Mac 与服务器边界

当前 Mac 已经完成 49,766,400 条中心线的 B0/B1/B2/B3 及 12 组 B1 参数敏感性，说明几何、评分器、16³/32³ smoke 和数据预处理不需要租 GPU。

租 CUDA 服务器只在以下条件同时满足后触发：

1. development rotation 40 给出高于 repeatability floor 的正信号；
2. 32³ 模型已经完成内存和单步时间 profile；
3. 候选和强基线的 checkpoint/config/split 已冻结；
4. 目标是 64³ 或更高、多模型、多随机种子公平比较。

在此之前租卡只会加速一个尚未被物理证据授权的问题。

## 10. 现在最值得问何远哲的问题

1. B1 axis 的负方向来自何种坐标约定，是否能由 CAD 或 visual hull 独立复核？
2. 25 度是固定经验裁剪、视场几何还是物理流场先验？
3. vertex 是否逐装置标定；5 mm 应力测试相对真实标定协方差是大还是小？
4. flow-off repeats 能否按相同 optical-flow/BOS 处理生成逐相机 repeatability floor？
5. 组内最终希望把 rotation/session 还是 timestamp block 当独立实验单位？
6. 是否接受先以 held-out image-space consistency 作为毕设主终点，再把三维 truth 留给 synthetic/CFD？

## 11. 可复核入口

- [B1 参数敏感性公开摘要](psu_b1_parameter_sensitivity_public_summary.json)
- [B1 参数敏感性 PNG](../demo_t16_operator/results/psu_b1_parameter_sensitivity/psu_b1_parameter_sensitivity_figure.png)
- [B1 参数敏感性 PDF](../demo_t16_operator/results/psu_b1_parameter_sensitivity/psu_b1_parameter_sensitivity_figure.pdf)
- [70 视角协议公开摘要](psu_heldout_camera_protocol_public_summary.json)
- [冻结协议配置](../demo_t16_operator/configs/psu_heldout_camera_protocol_v1.json)
- [预测包模板](../data_templates/psu_heldout_reprojection_bundle_template.json)
- [最终评分器](../site_tools/score_psu_heldout_reprojection.py)
- [PSU 官方数据索引](https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/)

公开文件不含原始 deflection、逐像素数组、本机路径、VPN 信息或私有派生产物。
