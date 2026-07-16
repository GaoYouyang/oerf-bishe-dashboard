# 发给何远哲师兄的最小数据合同

更新时间：2026-07-15
用途：把“能不能给我一些数据”改成一次可以回答、可以交付、不会反复补字段的请求。先拿最小包，不要求师兄一次整理完整论文数据。

## 可以直接发的短消息

> 师兄，我准备先做“几何与噪声条件化的低调用次数 BOST 三维重建算子”，核心仍显式调用现有 forward/adjoint，不改实验物理。为了让公开数据上的代码以后能直接迁移，能否先给我一个最小样例包：同一工况 1-3 帧的九视角 reference/flow-on 图或处理后 displacement，九路视角的 mask 与标定/射线参数，一份现有重构结果，以及每个字段的单位和公开边界？如果只能给 displacement 也可以。另请确认组内 forward 是否已有 `F` 和 `F^T/J^T`，以及 flow-off 重复帧能否用于估计逐相机/逐像素噪声。我先只交付 loader、伴随检查、预白化强基线和 held-out-view 报告，不先动完整数据。

## P0：没有这些就不能迁移

| 类别 | 最小字段 | 为什么必须 |
| --- | --- | --- |
| 观测 | 每视角 displacement/phase 的两个分量，或 reference + flow-on 原图 | 定义模型输入；原图可后补，位移先行也能跑 |
| 有效区 | view id、mask、坏点/遮挡、被删除的相机 | 避免把无效像素当零位移 |
| 几何 | 每视角 projection/ray origin-direction，或可生成 ray 的内外参 | 同一模型跨布局的关键，不应只给角度标签 |
| 网格 | 重建体范围、`D×H×W`、坐标轴方向、单位 | 防止转置、镜像和毫米/米错误 |
| 前向 | `F(x)` 的脚本/矩阵/函数接口 | 训练和真实 data consistency 必需 |
| 伴随 | `F^T(r)`、`J^T r` 或允许我实现后由师兄核对 | unrolled 算法和调用预算必需 |
| 参考结果 | 同工况现有 voxel/NeRIF/TDBOST 输出，哪怕只有切片 | 只作 sanity check，不冒充真值 |
| 权限 | 可否本机保存、组会展示、毕业论文使用、公开代码/图 | 数据进仓库前先锁边界 |

## P1：决定论文创新是否成立

1. **噪声标定**：每路相机至少 20-50 张 flow-off 重复帧；若已有光流置信度、相关峰值、亮度或 phase variance，一并给出。
2. **几何变化**：装置是否只有固定九路，还是不同实验日、裁剪、光纤映射或相机微移会变化？
3. **support 来源**：support/mask 是独立 CAD/标定得到，还是看了重构结果后手工画的？部署时能否获得？
4. **留出视角**：是否允许固定一条相机只作 `Q_audit`，不参加训练和停止？
5. **实验 run id**：至少给日期/工况/序列编号，保证按 run 切分，不能随机拆帧造成泄漏。
6. **独立物理参考**：是否有热电偶、Rayleigh/PLIF、CTC、PIV 或已知 phantom，可用于验证某一截面或统计量？
7. **模型失配**：当前用直线、cone ray 还是曲线 ray？窗口、有限孔径和二次成像是否已进入 forward？

## P2：若要接何远哲 4D / PIV-BOST 主线

- 原始时间戳、曝光、帧率；九路是否严格同步，若不同步要保留每路时间。
- 连续序列的 run 边界；至少 3 个独立 run，而不是同一序列随机切帧。
- TDBOST 的张量切分、rank、时间窗口、显存与当前最慢环节。
- PIV 图像对、`Δt`、激光片位置、未补偿/补偿速度场和同步映射。
- 师兄当前最关心的失败案例：薄前缘抹平、拓扑变化、坏视角、强折射、速度偏差或计算时间，只选一个作主终点。

## 建议的数据目录

```text
case_YYYYMMDD_condition/
  manifest.json
  raw/reference/view_*.tif
  raw/flow_on/view_*.tif
  displacement/view_*_{u,v}.npy
  masks/view_*.npy
  geometry/cameras.json
  geometry/rays_or_forward.*
  noise/flow_off_repeat_*.tif
  reference_reconstruction/volume.*
  permissions.txt
```

`manifest.json` 至少包含：

```json
{
  "case_id": "...",
  "run_id": "...",
  "field_units": "refractive_index|density_kg_m3|normalized",
  "observation_units": "pixel|phase|metric_deflection",
  "volume_shape": [0, 0, 0],
  "volume_bounds_m": [[0, 0], [0, 0], [0, 0]],
  "view_ids": [],
  "timestamps_s": [],
  "forward_model": "linear_straight|cone|curved|unknown",
  "support_available_at_inference": true,
  "permissions": {
    "local_training": false,
    "group_meeting": false,
    "thesis_figures": false,
    "public_release": false
  }
}
```

## 第一次拿到数据后的 48 小时交付

1. loader 和单位/坐标审计，不训练网络。
2. `⟨Fx,r⟩` 与 `⟨x,F^Tr⟩` 的伴随误差。
3. 九路亮度、位移、噪声、mask 面积和坏点表。
4. ridge/CGLS、预白化 Landweber、PBB/SPG 的 error-vs-call 或 held-out-view 曲线。
5. 明确列出哪些指标有真值、哪些只有重投影，避免漂亮图代替测量证据。

这份最小合同的核心不是多要数据，而是先得到可部署时真的可用的信息：几何、噪声、mask、run 边界和 forward/adjoint。
