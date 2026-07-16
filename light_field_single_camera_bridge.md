# Light-field / Single-camera Tomography Bridge

用途：这份文件把 OERF 的 light-field / plenoptic / single-camera endoscopic tomography 方向，翻译成 Gao Youyang 毕设可用的判断。它不是要把主线从何远哲的 BOST/NeRIF 拉走，而是解释：OERF 为什么长期关心“少相机、有限光学通道、硬件编码 + 计算重构”，以及这些经验怎样反过来支撑 BOST/NeRIF 的真实数据接口和 M3C 系统误差路线。

最后核验：2026-07-07。主要依据：蔡伟伟交大主页、OERF 公开方向、Optica / Springer / ScienceDirect 官方页面和本工作台 `references.bib`、`source_audit.md`、`paper_library/papers.json`。

---

## 1. 总判断

这条线的关键词是：

- light-field / plenoptic imaging：一个相机记录空间和角度信息。
- single-camera endoscopic tomography：通过内窥/光纤/多输入端把多视角投影编码到一台相机上。
- calibration / view mapping：多视角不是免费得到的，必须解决几何标定、畸变、分辨率和视角数取舍。
- high-speed volumetric imaging：减少相机数量和同步复杂度，服务 kHz / 10 kHz 级三维反应流诊断。

对你的毕设结论：

1. 它是 OERF Computational Imaging 的关键背景。
2. 它和何远哲 BOST/NeRIF 的共同母题是“有限观测下恢复三维/四维流场”。
3. 它不应替代 NeRIF/BOST 主线，因为你当前最贴师兄的是折射率场重构、PIV-BOST 补偿和 4D BOST 子问题。
4. 它最适合转化为 M3C 问题：视角数、标定、光路映射、空间分辨率、硬件编码误差和真实数据接口。

---

## 2. 核心论文地图

| 论文 | DOI / 入口 | 这篇解决什么 | 对 BOST/NeRIF 的启发 |
| --- | --- | --- | --- |
| Assessment of plenoptic imaging for reconstruction of 3D discrete and continuous luminous fields | `10.1364/JOSAA.36.000149`；Optica 官方页 | 用 plenoptic imaging 进行三维离散/连续 luminous field 重构评估；官方摘要说明 PI 可用单相机同时记录空间和角度信息 | 说明“少相机不是白拿信息”，光场采样、标定和算法选择会影响三维重构可信度 |
| kHz-rate volumetric flame imaging using a single camera | `10.1016/j.optcom.2018.12.036`；ScienceDirect 官方页 | 单相机高速体火焰成像 | 和 BOST 的共同点是把硬件复杂度转化为计算反问题；可作为 single-camera/kHz 背景 |
| Parametric study on single-camera endoscopic tomography | `10.1364/JOSAB.379793`；Optica 官方页 | 优化 single-camera tomography 系统参数，例如 fiber bundle 输入端数和镜头焦距；官方摘要指出九投影注册到单相机表现最好 | 对 M3C 很有用：不要只问网络结构，也要问 view count、fiber/camera geometry 和 focal length |
| Improved calibration model for single-camera endoscopic tomographic systems | `10.1364/JOSAB.396415`；Optica 官方页 | 单相机内窥层析的标定模型 | 对真实 NeRIF/BOST 数据接口很关键：相机/光纤/视角映射错误可能比网络误差更大 |
| Numerical and experimental validation of a single-camera 3D velocimetry based on endoscopic tomography | `10.1364/AO.58.001363`；Optica 官方页 | 单相机内窥三维速度测量 | 和 PIV-BOST 速度补偿线相邻，可帮助提出“速度场接口”问题 |
| Three-dimensional concentration field imaging in a swirling flame via endoscopic volumetric laser-induced fluorescence at 10-kHz-rate | `10.1007/s11431-019-1574-4`；Springer 官方页 | 10 kHz 内窥体 LIF 浓度场成像 | 说明 OERF 的三维诊断对象不仅有折射率，也包括浓度/反应标量；但本科不必同时做 |
| Reconstruction of kHz-rate 3-D flame image sequences from a low-rate 2-D recording via a data-driven approach | `10.1364/JOSAB.398009`；Optica 官方页 | 低帧率二维记录重建 kHz 三维火焰序列 | 和 4D BOST 的时序补全/低维先验相邻，适合 related work，不适合开局复现 |

---

## 3. 和何远哲方向的关系

### 共同点

| 共同问题 | Light-field / single-camera 线 | He / BOST-NeRIF 线 |
| --- | --- | --- |
| 观测有限 | 一台相机、有限角度、有限 micro-lens / fiber views | 九视角 BOST、少视角、limited-view 重构 |
| 硬件编码 | 光场相机、内窥/光纤束、单相机多投影 | 背景纹影、多视角光路、可能的内窥/多输入系统 |
| 反问题 | 从编码图像恢复 3D luminous/concentration/velocity field | 从背景位移恢复 3D refractive-index field |
| 真实误差 | 标定、畸变、空间分辨率、视角映射 | 标定、mask、bad view、景深、位移置信度、forward model mismatch |
| 可做本科子题 | 参数扫描、calibration checklist、view-count tradeoff | M2/M3C view/noise/geometry sanity check |

### 关键差异

- Light-field / endoscopic tomography 多数重构对象是发光场、浓度场或速度场；NeRIF/BOST 主线重构的是折射率场或其梯度相关观测。
- Light-field 依赖 microlens array 的 4D light-ray 采样；BOST 依赖背景图案 apparent displacement 与折射率梯度。
- Single-camera endoscopic 系统强调把多投影注册到同一传感器；何远哲 NeRIF 论文更强调神经隐式折射率场对 BOST 反问题的表示能力。

所以写开题时不要说“我要做光场相机”。更稳的说法是：

> OERF 在 light-field 与 single-camera endoscopic tomography 中已经长期研究“有限光学通道下的三维反演”。本课题选择 BOST/NeRIF，是因为它直接对齐何远哲当前工作；但会借鉴这些前史中的视角数、标定、空间分辨率和硬件编码误差分析。

---

## 4. 可转化成 M3C 的本科小实验

| 小实验 | 输入 | 输出 | 为什么有价值 |
| --- | --- | --- | --- |
| view-count tradeoff | 5/7/9/13/21/70 view subsets 或合成九视角 | error vs view count 曲线 | 呼应 single-camera endoscopic tomography 中“九投影”系统设计问题 |
| view mapping error | 对每个 view 加小角度偏差或像素 shift | 重构误差、重投影误差、bad-view 排名 | 对应真实标定误差，比调网络层数更贴实验 |
| spatial-resolution proxy | 用 edge/blob phantom | edge spread、FWHM、resolution score | 呼应 light-field / endoscopic 系统的空间分辨率损失 |
| encoded-observation comparison | parallel camera views vs grouped/single-camera-like views | 同视角数下的误差对照 | 说明硬件编码方式会改变反问题条件数 |
| calibration checklist | 不跑模型，只整理字段 | `camera_matrix`、`view_angle`、`fiber_mapping`、`mask`、`MTF`、`background_distance` | 能直接拿去问何远哲真实数据接口 |

---

## 5. 读法和取舍

优先读：

1. `10.1364/JOSAA.36.000149`：抓住 plenoptic imaging 的单相机空间+角度采样、校准和算法评估。
2. `10.1364/JOSAB.379793`：抓住 single-camera endoscopic tomography 的参数扫描和九投影设计结论。
3. `10.1364/JOSAB.396415`：抓住 calibration model 对真实重构的影响。

只作背景：

- `10.1007/s11431-019-1574-4`：10 kHz endoscopic LIF 很强，但物理量从折射率切到了浓度场。
- `10.1364/JOSAB.398009`：data-driven kHz 3D flame sequence 很适合 4D related work，但不要开题阶段承诺复现。

暂不深挖：

- 一般 light-field camera review、light-field microscopy、非流场方向的 plenoptic imaging。它们能补概念，但会分散 BOST/NeRIF 主线。

---

## 6. 给何远哲的问题

1. 组内 BOST/NeRIF 数据是否来自多相机、单相机多输入端、内窥/光纤束，还是另一种光路？
2. 如果是九视角系统，九个 view 的 mapping / calibration 是如何给到算法的？
3. 是否存在某些 view 长期质量差、遮挡、畸变或空间分辨率不足？
4. 真实数据里有没有 MTF、PSF、background distance、measurement volume distance、f-number、aperture 或 depth-of-field 记录？
5. 本科阶段做一个 view mapping / calibration / spatial resolution sanity-check report，对组里是否有实际价值？

---

## 7. 最终建议

把这条线放在论文和汇报中的位置应当是：

- 第 1 章：OERF computational imaging 背景。
- 第 2 章：有限视角、硬件编码和真实几何误差的反问题来源。
- 第 4 章或附录：M3C view/geometry sanity check 的理论依据。
- 不放在题目标题中，除非何远哲明确让你转向 single-camera/endoscopic tomography。

一句话结论：

> Light-field / single-camera tomography 是理解 OERF 的重要桥，但你的毕设主线仍应锁定 BOST/NeRIF；它最该提供的是视角数、标定和硬件编码误差意识。
