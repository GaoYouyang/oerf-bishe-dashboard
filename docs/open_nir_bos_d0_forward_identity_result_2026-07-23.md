# 公开 Phantom 1 clean-room 值重放诊断：v0 失败、v1 机器门通过与 D0.5 缺口

日期：2026-07-23

当前状态：`D0_FORWARD_IDENTITY_PASS_IMPLEMENTATION_DIAGNOSTIC_ONLY`

最高证据等级：`E1_PUBLIC_SINGLE_PHANTOM_POSTOPEN_IMPLEMENTATION_DIAGNOSTIC_ONLY`

突破监测：**没有突破**

## 1. 一句话结论

我们第一次把公开 Phantom 1 的相机、直线光线、三维折射率梯度和公开 XYZ projection 接进了一个不依赖作者 CUDA 代码的 clean-room **值重放**实现。v1 的机器合同门全部为真，但 ground-truth 误差不是冻结 PASS 条件，当前 NumPy/SciPy 实现也没有 autograd；因此它只授权先做 D0.5 可微镜像与梯度一致性检查，**还不直接授权训练，更不等于三维重建、新算法或泛化成功**。

## 2. 为什么这一步必须先做

算子学习不是把体数据喂给一个更大的网络。BOS/BOST 的学习问题至少包含一条物理链：

```text
三维折射率场 n(x) -> 空间梯度 -> 沿相机光线积分 -> XYZ deflection/projection
探测器 u/v 光流 -> uvtoeps -> 公开 XYZ projection 表示
```

只要相机坐标、XYZ 轴序、长度单位、ROI、光线方向或积分规则有一个接错，网络仍可能把训练误差压低，却是在补偿接口错误。这样的“成功”不能迁移到何远哲师兄的真实 geometry，也不能拿来与 DeepONet、FNO 或神经隐式场公平比较。

本轮使用的一级来源是 [Neural Refractive Index Primitives 论文](https://arxiv.org/html/2605.11454) 和其[作者公开仓库](https://github.com/Weihu22/Neural-Implicit-Reconstruction-for-BOS)，固定 commit 为 `a385cce83d88df24ed05dccfd6fde20e124f5604`。论文给出的合成观测链包含有限孔径多光线积分、图像模糊、背景扭曲与光流位移提取；公开 `img*.png` 保存的是由 detector `u/v` 经 `uvtoeps` 转换后的 XYZ projection。clean-room 实现只用单条直线光线和 midpoint quadrature，因此目标是检查**值级相容性与接线错误**，不是声称复现完整成像链或证明算子同一性。

## 3. 我们实际实现和保留了什么

- 精确读取公开数据中的 16-bit RGB PNG；避免 Pillow 默认路径静默降为 8-bit。
- 从相机矩阵重建 ray origin/direction，并显式处理平行于 AABB 面的光线。
- 对三维折射率场求 XYZ 梯度，以 midpoint quadrature 做线积分。
- 固定 12 个 train views、每视角 512 条 ray、总计 6,144 条；v0/v1 使用相同像素选择哈希。
- 同时重放 ground truth、support-masked ground truth 和发布的 CGLS-TV 场。
- 结果仓库只保留聚合指标、充分统计量、输入哈希和图，不复制作者图像、mask、MAT 场或源码。
- 验证器从充分统计量独立重算全局指标，并核对输入边界与 SHA-256；相机、ROI 外 RMS 和 128/256 quadrature 数值来自运行摘要，只重算阈值布尔值，没有重新读取外部体数据或重新追光。

## 4. v0 为什么必须判失败

v0 的初始开发合同采用 128 个积分步、64 步作收敛比较，并把未穿过 ROI 的 ray 排除。v0 没有独立的事前协议文件，不能称为预注册。结果如下：

| 门 | v0 结果 | 初始开发要求 | 判定 |
|---|---:|---:|---|
| ray 与 ROI 相交率 | 6,021 / 6,144 = 0.97998 | >= 0.99 | 失败 |
| CGLS-TV 64 -> 128 quadrature 差 | 0.05260 | <= 0.02 | 失败 |
| 相机中心方向 | 通过 | 最小 cosine >= 0.99 | 通过 |
| CGLS 重放与轴序 | 通过 | 相关/误差/identity 门 | 通过 |

因此 v0 的正式状态永久保留为：

```text
D0_FORWARD_IDENTITY_NO_GO_FIX_GEOMETRY_OR_UNITS
```

单元测试还抓到了第一版平行 ray/AABB 判断错误。修复代码错误并不会抹掉 v0；v0 仍记录“原协议不能通过”。

## 5. 为什么允许 v1 修复，而不是偷偷降门槛

v0 打开后只做了定位，不换 ray、不训练网络：

- 123 条未相交 ray 的观测 RMS 为 `0.0144852`，相交 ray 为 `3.7785555`，比值 `0.00383353`；它们更符合“ROI 外零场贡献”，而不是“无效样本”。
- ground truth 的 128 -> 256 quadrature 差为 `0.005695`，CGLS-TV 为 `0.009622`；说明 256 步可作主值，128 步可作收敛比较。

所以 v1 在运行前冻结了两项**语义修复**：

1. ROI 外 ray 的预测固定为零，但仍保留在 6,144 条总分母中；同时要求 `RMS(outside)/RMS(inside) <= 0.01`。
2. 主积分用 256 步，比较器用 128 步，差仍必须 `<= 0.02`。

冻结协议 SHA-256 为 `1833632d4e5cce6b75cf095fcf5799488f4bc4dc055f8a7127afa4c091bd3e94`。这不是 fresh test；证据等级明确写着 `POSTOPEN`。

## 6. v1 的机器结果与验证边界

验证状态为 `VALID_PASS_D0_IMPLEMENTATION_ONLY`，共 281 项检查。指标算术、CSV 充分统计、输入清单、公开边界和 SHA-256 得到了独立核对；几何与 quadrature 原始量没有由验证器重新生成。

| 重放场 | mean component Pearson | 一个全局尺度 | 缩放后 relative-L2 | 直接 relative-L2 |
|---|---:|---:|---:|---:|
| ground truth | 0.9881369 | 0.9489756 | 0.1462135 | 0.1555878 |
| support-masked ground truth | 0.9881394 | 0.9484864 | 0.1461789 | 0.1557401 |
| released CGLS-TV | 0.9809594 | 1.0043039 | 0.2038715 | 0.2039147 |

几何与数值门：

| 诊断 | v1 结果 | 门 |
|---|---:|---:|
| 总 ray 分母 | 6,144 / 6,144 | 必须精确相等 |
| 最小相机中心 alignment cosine | 0.9944110 | >= 0.99 |
| 128 -> 256 quadrature 差 | 0.00962175 | <= 0.02 |
| ROI 外/内观测 RMS 比 | 0.00383353 | <= 0.01 |
| 最优轴序/符号 | XYZ / `+++` | 必须与直接解释一致 |

### 冻结 PASS 没有包含什么

v1 冻结协议对相机方向、ray 记账、quadrature、轴序和发布 CGLS-TV 重放设置了门，但**没有为 ground-truth Pearson、ground-truth relative-L2 或逐视角尾部设置接受阈值**。因此 `0.9881369 / 0.1462135` 是有价值的 post-open 描述，不是触发 PASS 的预先门槛。机器状态原样保留，人工解释必须比状态名更窄。

## 7. 独立审计新增的 D0.5 前置门

当前实现调用 NumPy `gradient` 和 SciPy `map_coordinates`，没有 PyTorch/JAX autograd、伴随、JVP/VJP 或梯度一致性测试。算法训练前必须先完成：

1. **Torch/JAX 值一致性：**固定同一场、相机、ray 与 128/256 积分规则，对 NumPy reference 做逐分量绝对/相对误差门。
2. **有限差分方向导数：**对 voxel 场和 Fourier MLP 参数各做中心差分 vs autograd；覆盖平滑场、边界场和 ROI 外 ray。
3. **伴随或 JVP/VJP 点积：**若实现线性离散壳，检验 `<A dx, y> = <dx, A^T y>`；若直接对网络参数求导，检验 JVP/VJP 双线性一致。
4. **GT 新门只在独立 phantom 冻结：**现有 Phantom 1 已经打开，不能事后给 0.988/0.146 附近设阈值再称验证。至少需要新的 synthetic field/camera split。

D0.5 全过之前，“可微算子”“可训练接口”和“公平烟测已就绪”都不能写。

冻结 artifact 不能在看过结果和审计后篡改，因此原 `summary.json.next_authorized_action` 仍保留。当前有效机器决定由[事后审计覆盖层](open_nir_bos_d0_forward_identity_v1_audit_overlay.json)给出：`training_authorized=false`、`d0_5_required=true`。覆盖层只否决训练授权，不把 post-open 审计冒充 fresh science。

## 8. 一个重要的负结论

ground-truth 重放误差比发布 CGLS-TV 重放误差低 `0.057658`，而不是更高。因此本轮**不支持**“CGLS-TV 场通过反演吸收了有限孔径、光流或预处理失配，所以更能重放观测”这一猜想。

ground-truth 仍有约 14.6% 的缩放后相对残差。它与有限孔径、图像扭曲/光流、离散化、单 ray 近似和 SciPy/MATLAB resize 差异都相容，但这次实验不能把残差唯一归因于其中任何一个。把“可能原因”写成“已证明机制”会越过证据。

## 9. D0.5 以后才允许做优化烟测

D0.5 通过后，下一轮仍不是论文 benchmark，只检查几种参数化能否优化、成本是否可控、会在哪里失败：

| 代号 | 参数化 | 它回答的问题 |
|---|---|---|
| B0 | 发布 CGLS-TV 重放 | 固定外部参考，不参与 optimizer-step 或参数量匹配 |
| B1 | 低分辨率 voxel | 最简单可训练场能否下降，暴露梯度/单位错误 |
| B2 | Fourier MLP | 连续隐式场在 Mac 上的基本表达与频谱偏置 |
| H1 | B2 base + 有界 residual / NASH-Mix-lite | 与 parameter-matched B2-wide、call-matched B2-long 两个对照比较 |

所有可训练臂使用相同 12 views、相同 6,144 ray、相同可微 forward 值语义和积分规则。正式排序前只选择一个 primary resource budget，例如总 `ray samples x quadrature samples`；optimizer step、wall time、参数量和峰值内存作为完整账本报告，不能假装四者能同时严格匹配。还须冻结损失、初始化、seed 数、调参预算、全局尺度政策和逐视角尾部指标。B0 只作固定物理锚点。只有一个已打开 phantom 时，结果只能叫 optimization smoke，不能叫方法优越或泛化。

停止条件：

- NumPy/Torch 值一致性、有限差分或伴随/JVP/VJP 任一失败；
- 输入哈希或相机/轴序语义漂移；
- 改善只来自更多 ray、更多步骤或更多参数；
- field 指标改善但 projection/tail 明显恶化；
- 候选只击败故意欠训练的 baseline；
- 需要把无再分发许可的作者数据或源码复制进项目仓库。

## 10. 需要何远哲师兄确认的六个接口问题

1. 组内 callable 的输出是全局 XYZ deflection，还是每台相机 detector `u/v` 位移？
2. 不穿过 ROI 的 ray 应该返回零、跳过、mask，还是在边界终止？
3. 相机/场的坐标轴、长度单位、ROI 支撑和归一化的精确定义是什么？
4. 真实主残差使用 straight ray 还是 curved ray；两层是否都可调用？
5. 是否能提供 JVP/VJP，或至少稳定的 `A(x)` 与 `A^T(y)` 接口和调用成本？
6. 独立单位是什么：新 field、shot、session、rig 还是 view；论文验收的主物理 endpoint 是 density/RI、reprojection、PIV velocity，还是其他量？

这些答案决定我们的算子学习应该学场、warm start、预条件器、有界 correction，还是只做 uncertainty/gating。没有接口合同，不应先把网络结构定死。

## 11. 可复现入口

- [v1 机器摘要](../learning_labs/results/open_nir_bos_d0_forward_identity_v1/summary.json)
- [v1 独立验证](../learning_labs/results/open_nir_bos_d0_forward_identity_v1_validation/validation.json)
- [v1 充分统计量](../learning_labs/results/open_nir_bos_d0_forward_identity_v1/sufficient_statistics.csv)
- [v1 输入哈希清单](../learning_labs/results/open_nir_bos_d0_forward_identity_v1/input_manifest.csv)
- [v1 冻结协议](../learning_labs/open_nir_bos_d0_forward_identity_v1_protocol.json)
- [v1 事后审计覆盖层](open_nir_bos_d0_forward_identity_v1_audit_overlay.json)
- [v0 NO-GO 摘要](../learning_labs/results/open_nir_bos_d0_forward_identity_v0/summary.json)
- [v0 -> v1 修复协议](open_nir_bos_d0_forward_identity_repair_protocol_2026-07-23.md)

**当前最诚实的里程碑不是“模型赢了”，而是“公开直线 ray 的值重放链已经可审计，且独立审计明确找出了 GT 冻结门与可微实现这两个下一步缺口”。**
