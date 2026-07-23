# PSU rotation-40 多分辨率机制诊断：冻结开发协议

状态：`FROZEN_POSTOPEN_DIAGNOSTIC_BEFORE_DERIVED_FORWARD_EVALUATION`

这不是新的盲测或预注册确认实验。rotation-40 已在 E68 正式审计中打开，32³+CGLS4
相对 16³+CGLS4 的退化已经知道。本协议只在看见**派生对照的 forward 结果之前**冻结，目的是把
“换了 forward 网格”与“重建场本身变了”尽量拆开，为下一步算法设计排雷。

## 1. 只回答一个问题

已有比较同时改变了两件事：

1. forward/gradient/interpolation 使用 16³ 或 32³ 离散网格；
2. CGLS 从零出发在不同离散系统中走 4 步，得到不同的重建场。

因此不能从 `x16` 优于 `x32` 直接推断“细网格有害”或“高频过拟合”。本轮固定以下对照：

| 记号 | 固定计算 | 作用 |
|---|---|---|
| `A16 x16` | 原 16³ 场在原 16³ forward 上评分 | E68 端点复核 |
| `A32 Ux16` | 把 16³ 场三线性延拓到 32³，再用 32³ forward | 只换 forward 离散化 |
| `A16 Dx32` | 把 32³ 场限制到 16³，再用 16³ forward | 看 32³ 场的粗尺度投影 |
| `A32 UDx32` | 32³ 场先下采样再上采样，用 32³ forward | 隔离 32³ 场的细尺度剩余 |
| `A32 x32` | 原 32³ 场在原 32³ forward 上评分 | E68 端点复核 |

`U` 和 `D` 都固定为 PyTorch 三线性 resize，`align_corners=True`。这让两个网格端点对应同一
物理边界；每次 resize 后重新施加一层零外边界 gauge。数组布局固定为
`[batch, channel, z, y, x]`。

## 2. 细尺度修正诊断

固定直线：

\[
x(\alpha)=U x_{16}+\alpha\left(x_{32}-U x_{16}\right),
\qquad \alpha\in\{0,0.25,0.5,0.75,1\}.
\]

只报告这五个预先列出的点，不搜索最优参数。由于 forward 是线性的，直线预测由两个端点精确
组合；仍额外直接 forward `alpha=0.5`，要求与线性组合最大绝对差不超过 `1e-10`。

在 pooled 与每台相机上计算：

\[
r=y-A_{32}Ux_{16},\qquad d=A_{32}(x_{32}-Ux_{16}),
\]

以及 `cos(r,d)`。负值表示冻结的 32³ 场修正沿该真实观测残差的反方向移动。还报告
`alpha*=<r,d>/||d||²`，但它只是**看完 rotation-40 后的解释量**，不能成为候选、不能用于 final
rotation，也不能写成“学到的 gate”。

## 3. 预先固定的机制屏

只有同时满足以下三个数值条件，机器才输出
`OPENED_BLOCK_FIELD_CORRECTION_ANTI_ALIGNED_GRID_FORWARD_GAP_SMALL`：

- `|relL2(A16 x16)-relL2(A32 Ux16)| <= 0.01`；
- `relL2(A32 x32)-relL2(A32 Ux16) >= 0.01`；
- camera 2/3/4 的 `cos(r,d)` 全部小于 0。

这些 `0.01` 只是冻结的机制筛查尺度，不是统计、物理或工程显著性界限。条件成立也只能说明：
在这个已经打开的 rotation block 上，forward 网格替换造成的端点分数差较小，而冻结重建场之间
的修正与残差反向。它不能证明唯一因果机制，更不能把问题单独归因于高频、CGLS、噪声或网格。

## 4. 完整性与隐私边界

- 协议提交必须先于输出目录；runner 检查输出在该 commit 中不存在。
- runner、测试、配置和全部直接/传递 operator 依赖都必须逐字节匹配协议 commit。
- E68 source config、source result、两个 frozen volume、private reports、geometry/payload manifests
  和每个 `.npy` 均复核哈希、shape 与 dtype。
- 只允许使用 rotation-40 已打开的 camera 2/3/4 active rows；final rotations 继续封存。
- 公开结果只含汇总指标、CSV、图和 checksum，不含测量、预测、几何、场数组或本地路径。

## 5. 明确不授权

- 不授权三维场精度或 field relative-L2；
- 不授权跨旋转、跨 session、跨设备或跨实验泛化；
- 不授权相机独立重复，三台相机共享一个 rotation run；
- 不授权算法优越性、RTG-MRC 有效性、神经算子训练成功或论文主张；
- 不允许从五个 alpha 中挑一个，随后把同一 rotation-40 当测试集。

下一步如果机制屏成立，优先在 support views 内做 leave-one-rotation-out：比较 CGLS 步数、H1/TV、
coarse-to-fine correction 与 fail-closed fallback。只有开发规则冻结后，才可一次性打开新的 final
rotation 作为确认审计。
