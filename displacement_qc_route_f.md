# 选题 F 详细执行页：BOS/PIV-BOST 位移估计与质量控制

用途：如果何远哲认为本科阶段最有价值的工作不是立刻复现完整 NeRIF / 4D BOST，而是先把真实 BOST / PIV-BOST 图像前处理、位移估计、坏视角筛查和误差传播做扎实，这页就是 F 路线的执行说明。

最后更新：2026-07-09。

---

## 1. 一句话定位

把 BOST / PIV-BOST 的输入从“几张图和一个位移场”变成“可审计的位移数据产品”：

```text
reference/deformed image
  -> displacement estimation
  -> confidence / bad-region mask
  -> view/frame health score
  -> BOST/NeRIF reconstruction or PIV-BOST compensation
  -> error propagation report
```

这条线不替代 A 的 NeRIF/BOST 鲁棒性主线，也不替代 B 的 PIV-BOST 补偿主线。它的价值是给 A/B/E 提供共同入口：先判断输入是否可信，再讨论三维重构或速度补偿。

---

## 2. 为什么 OERF 可能需要它

真实 BOST/PIV-BOST 不是从干净张量开始，而是从图像开始。图像层会有这些问题：

- 背景图案模糊、对比度不足、speckle/dot pattern 不均匀。
- 火焰或强密度梯度造成局部 wavefront aberration、blur、multiple deflection 或遮挡。
- 外界 atmospheric turbulence、光窗、反射、振动带来额外位移。
- 多视角数据里某些 view/frame 质量差，但不一定一眼能看出来。
- PIV 粒子图像还会叠加 particle density、particle image diameter、window size、outlier replacement 等处理参数。

如果这些前处理误差不被记录，后面的 NeRIF/4D BOST 误差会混在一起，论文里也很难说明“是重构方法失败，还是输入位移场不可信”。

---

## 3. 直接对齐的文献

| 类别 | 文献 | 它给 F 路线什么 |
| --- | --- | --- |
| He 主线 | Zheng/He et al., simultaneous PIV-BOST, DOI `10.1007/s00348-025-04093-y` | PIV-BOST 为什么需要把折射率场和速度误差接起来。 |
| 实时位移 | Zheng et al., real-time GPU-accelerated BOS, DOI `10.1007/s00348-026-04277-0` | 光流前处理速度、实时/离线 trade-off、工程化入口。 |
| 系统误差 | Wolf et al., wavefront aberration BOS, DOI `10.1016/j.measen.2024.101509` | wavefront aberration 可作为 quality stress test。 |
| 系统误差 | Liu et al., atmospheric turbulence shock detection, DOI `10.1364/OE.567012` | 外界扰动和 shock/strong-gradient 区域会污染 BOS 判断。 |
| PIV 工具 | OpenPIV, PIVlab, PIV operating rules, PIV uncertainty, RAFT-PIV | 粒子图像互相关、深度光流和 uncertainty 的对照工具链。 |
| BOS 位移基础 | Raffel 2015 BOS review; Rajendran dot tracking / UQ | 位移估计、dot blur、amplification、confidence weighting 和 density uncertainty。 |

---

## 4. 最小可毕业问题

不要问“哪个 optical flow 方法最强”。本科题应该问：

> 对 BOST/NeRIF/PIV-BOST 来说，怎样的图像位移场足够可信？如何自动标出不可用区域，并量化这些输入误差会怎样传到重构或速度补偿？

最小可毕业版本只需要完成四件事：

1. 生成 synthetic background image pair 和 synthetic particle image pair。
2. 比较 2-4 个位移估计方法。
3. 输出 bad-region mask、confidence、error map、runtime。
4. 把位移误差传到 M3A velocity compensation 或 M3C reconstruction-quality report。

---

## 5. 数据层设计

建议每一组 view/frame 都转成一个 manifest，而不是散落的文件名。

```json
{
  "case_id": "synthetic_flame_001",
  "view_id": "cam_03",
  "frame_id": 128,
  "reference_image": "ref_cam03_0128.png",
  "deformed_image": "def_cam03_0128.png",
  "displacement_u": "u_cam03_0128.npy",
  "displacement_v": "v_cam03_0128.npy",
  "confidence": "conf_cam03_0128.npy",
  "bad_region_mask": "mask_bad_cam03_0128.png",
  "method": "farneback",
  "quality_score": 0.82,
  "notes": "synthetic wavefront blur + 2 percent Gaussian noise"
}
```

真实数据到来前，先用 synthetic manifest；真实数据到来后，只换路径和相机几何字段。

---

## 6. 位移方法选择

| 方法 | 为什么放进本科 benchmark | 不要过度承诺 |
| --- | --- | --- |
| OpenPIV 互相关 | PIV-BOST 粒子图像的传统可解释 baseline。 | 不一定适合随机背景图案的所有 BOS 场景。 |
| Farneback optical flow | OpenCV 容易跑，适合 dense displacement baseline。 | 不是高精度 SOTA，但适合作为稳定参考。 |
| DIS / TV-L1 optical flow | 速度快，适合前处理和质量筛查。 | 参数敏感，必须记录 runtime 和失败区域。 |
| RAFT / RAFT-PIV | 深度光流/神经 PIV 的代表邻居。 | 不要把本科题变成训练大模型；最多做预训练模型或小样例对照。 |
| Dot tracking / DIC | 与 BOS 背景图案和局部相关方法相邻。 | 若无成熟实现，先作为 related work 或后续扩展。 |

---

## 7. 质量指标

最少报告这些指标：

| 指标 | 作用 | 适用数据 |
| --- | --- | --- |
| EPE / displacement RMSE | 有 ground truth 时衡量位移误差。 | synthetic background / synthetic particle。 |
| Bad-pixel ratio | 统计误差超过阈值的区域比例。 | synthetic 或有 reference 的真实数据。 |
| Photometric residual | 用估计位移 warp 回去后，图像还剩多少残差。 | 无 ground truth 的真实图像。 |
| Local correlation / PIV validation flag | 标出互相关峰值弱、outlier 或不可信窗口。 | PIV image pair。 |
| Smoothness / gradient anomaly | 检查位移场是否出现非物理尖峰。 | BOS / PIV 都可用。 |
| Runtime | 判断能否服务批量筛帧或实时前处理。 | 所有方法。 |
| Downstream error | 位移误差传到 NeRIF/M3A/M3C 后造成多大影响。 | 最关键，决定是否贴毕设主线。 |

---

## 8. Synthetic stress test

至少做四类扰动：

1. **Noise**：高斯噪声、shot noise、JPEG-like compression。
2. **Blur**：全局 defocus、局部 heat-haze blur、motion blur。
3. **Texture failure**：背景图案过稀、过密、周期纹理、低对比度。
4. **Flow-like distortion**：vortex-like displacement、shock-like sharp gradient、wavefront-like aberration、atmospheric-turbulence-like random phase。

每类扰动都输出同一张表：

| 扰动强度 | 方法 | EPE | bad-pixel | photometric residual | runtime | 是否建议进入 BOST |
| --- | --- | ---: | ---: | ---: | ---: | --- |

---

## 9. 与 A/B/E 的接口

### 接 A：NeRIF/BOST 鲁棒性

把 F 的位移误差当作输入噪声，不再只加 Gaussian noise：

```text
clean displacement
  + method-specific displacement error
  + bad-region mask
  -> NeRIF / voxel baseline
  -> reconstruction error and reprojection residual
```

### 接 B：PIV-BOST 折射补偿

比较三层补偿：

- raw particle image dewarping；
- correlation/displacement correction；
- exported velocity-field correction。

F 路线至少能给 B 路线提供第一层和第二层的误差边界。

### 接 E：系统误差和投影补全

把 bad-view / bad-region mask 输入到 M3C：

- 如果一个 view 局部坏掉，是否该 mask 掉？
- 如果一个 view 整体质量差，是否该丢掉？
- 投影补全是否会在坏视角上“编出看似合理但错误”的信息？

---

## 10. 12 周执行表

| 周 | 任务 | 可验收产物 |
| --- | --- | --- |
| 1 | 读 Raffel 2015、OpenPIV/PIVlab 入门、realtime GPU BOS 摘要和方法图。 | 位移估计方法族一页图。 |
| 2 | 写 synthetic background generator。 | ref/def background pair + ground-truth displacement。 |
| 3 | 跑 Farneback / DIS / TV-L1 baseline。 | EPE/rmse/runtime 表。 |
| 4 | 写 synthetic particle image pair 或接 OpenPIV 示例。 | PIV displacement field + validation mask。 |
| 5 | 加 noise/blur/texture stress test。 | bad-region mask 和 failure gallery。 |
| 6 | 读 wavefront aberration / atmospheric turbulence 文献。 | wavefront-like / turbulence-like stress test。 |
| 7 | 接 M3A：位移误差到 velocity-field error。 | 补偿前后速度误差图。 |
| 8 | 接 M3C：bad-view/bad-region 对重构质量影响。 | 视角质量和重构误差热图。 |
| 9 | 写 manifest 和自动报告脚本。 | 单个 case 的 HTML/Markdown quality report。 |
| 10 | 如果有真实数据，做 loader；没有则用开源/合成数据完善。 | 数据请求清单 + loader smoke test。 |
| 11 | 整理论文图：pipeline、method comparison、failure cases。 | 开题/中期汇报图 4-6 张。 |
| 12 | 写成可交给师兄的技术 brief。 | F 路线 4 页 memo + 是否转主线判断。 |

---

## 11. 第一轮要问何远哲

1. 组内 BOST/PIV-BOST 当前位移提取用什么方法或软件？
2. 有没有 per-view/per-frame 的质量记录，还是靠人工看图筛？
3. 最常见失败是背景图案、火焰强梯度、相机同步、标定、光窗，还是 PIV 粒子图像质量？
4. 师兄希望我输出位移场、mask、confidence、quality score、重投影误差，还是一个完整 HTML report？
5. 如果只能给一小段样例数据，优先给 background image pair、PIV raw image pair、已有 displacement field，还是 reconstruction result？

---

## 12. 风险边界

- 不要把 F 写成“我提出新的 optical flow 算法”。本科更稳的说法是“面向 BOST/PIV-BOST 的位移估计质量控制与误差传播”。
- 不要把 RAFT/深度光流作为必须项；没有 GPU 或训练数据时，只保留为对照或 related work。
- 不要脱离 BOST/NeRIF/PIV-BOST，只做图像算法排行榜。每个指标最后都要回答“这会不会影响重构或速度补偿”。
- 如果师兄给了真实数据，优先做 loader、mask、quality report；如果没有真实数据，优先做 synthetic stress test 和 M3A/M3C 接口。

---

## 13. 最小论文标题备选

1. 面向背景纹影层析的图像位移估计质量控制与误差传播分析。
2. 面向 PIV-BOST 折射补偿的粒子图像位移估计与不确定度评估。
3. 面向神经折射率场重构的 BOS 位移场置信度建模与坏视角筛查。

最建议标题是第 1 个，因为它最稳、最不依赖真实 PIV 数据，也最容易接回 A/E 主线。
