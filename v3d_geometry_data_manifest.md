# v3d 组内 Geometry / Data 确认清单

用途：发给何远哲师兄确认。它不是索要整套私有数据，而是先判断 **acquisition-geometry-conditioned F-Adapter 是否有可识别的研究问题**。

更新时间：2026-07-11

## 第一闸门：geometry 是否真的跨样本变化

请先勾选真实情况：

- [ ] 不同实验的相机数量 `K` 会变化。
- [ ] 相机角度、位置或基线会变化。
- [ ] 会出现缺失相机、遮挡区域或坏视角。
- [ ] 标定矩阵会随实验重新估计，存在可量化漂移。
- [ ] 不同相机的噪声、分辨率、曝光或置信度明显不同。
- [ ] 以上均不变化，所有样本使用完全固定的 canonical geometry。

**No-Go 条件：**若最后一项成立，并且组内也不关心缺失视角、不同 K、标定漂移或跨装置迁移，geometry-conditioned 方法没有可识别收益。此时应停止 v3d，把课题改为 saturated FNO / F-Adapter benchmark、NeRIF warm-start 或真实数据误差边界。

## 最小数据合同

| 数据项 | 最小形状或格式 | 单位/坐标系 | 每样本变化？ | 当前能否提供 |
|---|---|---|---|---|
| BOS displacement | `K x H x W x 2` | pixel 或物理位移 | 是 | 待确认 |
| 有效 mask | `K x H x W` | bool / confidence | 是 | 待确认 |
| camera intrinsics | `K x 3 x 3` 或等价参数 | pixel convention | 通常否 | 待确认 |
| camera extrinsics | `K x 4 x 4` 或 ray table | world/camera convention | 可能 | 待确认 |
| ray coverage / sensitivity | volume grid 或可生成接口 | 与重建网格一致 | 可能 | 待确认 |
| calibration uncertainty | 每相机标量/协方差/重复标定差 | 明确含义 | 可能 | 待确认 |
| noise / confidence | 每相机或每像素 | 方差、SNR 或经验分数 | 是 | 待确认 |
| reconstruction target | `D x H x W` | refractive index / density | 是 | 待确认 |
| timestamps | 每视角采集时刻 | s / frame index | 时序任务必需 | 待确认 |
| held-out audit view | observation + geometry | 不进入训练/选型 | 是 | 待确认 |

## 必须写清的坐标约定

1. 世界坐标轴方向、原点和长度单位。
2. volume array 的轴顺序，例如 `z, y, x`。
3. 图像坐标是 `row, col` 还是 `x, y`，原点位于哪里。
4. extrinsic 是 `world_to_camera` 还是 `camera_to_world`。
5. displacement 正方向、像素缩放和 background plane 距离。
6. 折射率、密度或温度之间使用的物性关系及适用范围。

这些信息只要有一项含糊，网络就可能把坐标错误学成“规律”。

## 数据角色不能混用

| 角色 | 允许用途 | 禁止用途 |
|---|---|---|
| train | 拟合模型参数 | 充当最终展示 |
| validation | 选 epoch、rank、频带和正则 | 声称确认性泛化 |
| dev2 | 诊断 OOD、geometry 消融和停止路线 | 继续反复调参后当 blind |
| blind final | 方法、阈值和脚本冻结后一次开启 | 回头改模型 |
| real audit | 检查 held-out view、重复性和物理 trace | 没有 GT 时声称绝对 3D 准确 |

## v3d 需要的 geometry groups

至少应能定义三类真正不同的采集条件，例如：

1. 不同 `K` 或角度布局。
2. canonical layout 与 missing/bad-camera layout。
3. nominal calibration 与独立重标定/扰动 layout。

划分必须按实验 case / geometry group 隔离，不能把同一次采集切成相邻帧后随机分到 train 和 test。

## 给师兄的六个最短问题

1. 组内“算子学习”最终想做 projection-to-volume inverse operator，还是 3D/4D evolution operator？
2. 当前真实相机布局是否跨 case 变化？若不变，未来是否需要 missing-camera 或跨装置泛化？
3. 能否先提供一个脱敏样例的 displacement、mask、geometry 和单位说明，而不是整套数据？
4. 组内最强传统与神经基线分别是什么，F-Adapter-style control 是否认可为必须项？
5. 哪个信号可以完全锁定为独立审计：额外相机、PIV correction、PLIF/front、积分量还是重复实验？
6. NeRIF 的 forward、loss、stop rule 能否作为统一后处理接口，而不是只给自有模型使用？

## Go / No-Go 判决

### Go

- geometry 或视角可靠性确实跨样本变化；
- 每个样本能绑定正确的 geometry/mask/confidence；
- 至少有一个不进入训练和选择的 audit signal；
- 师兄认可 F-Adapter、continued/full FNO 与 geometry shuffle 是强制对照。

### No-Go

- geometry 完全固定，且研究目标也不包含缺失视角或跨装置；
- displacement 与 geometry 无法逐样本对应；
- 只有漂亮切片，没有 held-out observation 或外部物理 trace；
- 只能给最终结果图，无法获得 forward/loss 或最小 baseline 接口。

No-Go 不是毕设失败。保底成果可以转为：**validation-saturated FNO 与 F-Adapter 的公平 benchmark + v3c 机制性负结果 + NeRIF warm-start 可行性与真实数据边界。**
