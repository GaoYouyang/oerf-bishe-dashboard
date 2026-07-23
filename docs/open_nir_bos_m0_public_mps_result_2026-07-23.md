# 公开 Phantom M0：MPS 大体场前向/反向门

> 日期：2026-07-23
>
> 主状态：`M0_PUBLIC_MPS_FORWARD_BACKWARD_PASS_IMPLEMENTATION_ONLY`
>
> 事前门：25 / 25
>
> 包一致性审计：`VALID_M0_PACKAGE_CONSISTENCY_ONLY`，152 / 152
>
> 突破监测：**没有突破**

## 先说人话

D0.5 只在一个很小的合成场上证明 Apple MPS 能反向传播。M0 把问题推进到真实公开数据尺寸：使用作者发布的 `140 x 294 x 140` CGLS-TV 体场、12 个相机视角、每视角 8 条事前固定射线，在 CPU `float64`、CPU `float32` 和 Apple MPS `float32` 上计算同一个 forward、同一个有效射线 MSE 和对整个三维场的梯度。

结果表明：在**这一个公开 Phantom、这 96 条冻结射线、固定直线几何和 voxel-field 参数化**下，MPS 的值和场梯度与 CPU 锚点一致，三次重复稳定，公开大体场方向导数通过，无效射线给出严格零输出和零场梯度，离散内存采样没有越门。因此可以开始一个很短的 voxel 优化烟测。

这不代表已经完成三维重建，更不代表 Fourier MLP、DeepONet、FNO、NeRIF 或新算法已经在 MPS 上得到验证。

## 为什么不能沿用 D0.5 的合成 MPS 绿灯

D0.5 的 MPS 场很小，无法回答三个工程问题：

1. `140 x 294 x 140` 场在反向传播时是否会爆内存；
2. MPS 三线性 gather 的反向累加在大场上是否稳定；
3. 真实公开场和真实公开观测构成的 loss，是否仍能产生有限、非零且与 CPU 一致的场梯度。

M0 专门补这道设备桥，不训练模型，不调学习率，也不看 ground truth 选参数。

## 先锁协议，再看结果

正式执行前，协议、runner 和测试已提交为私有 commit `6ea5ba7`。协议 SHA-256：

`43fef428307f7c9932615b69110bc0932d6ddfc4aa7dafd1ded21f24c5555ea8`

冻结内容包括：

- 12 个视角，每视角 8 条固定射线，共 96 条；
- 128 点 midpoint quadrature，ray chunk 为 24；
- 有效 AABB 射线上的 MSE；
- CPU64 锚点、CPU32 与 MPS32 三角比较；
- 三次完整 MPS 前向/反向重复；
- 公开大体场上的中心有限差分方向导数；
- 三条事前写定的无效射线；
- `PYTORCH_ENABLE_MPS_FALLBACK=0`；
- driver allocation 相对 baseline 的 2 GiB 门和清理后 64 MiB 门；
- 25 个数值门和全部 claim boundary。

## 两次失败没有删除

### v0

第一次正式运行在方向导数统计阶段停止。原因不是 forward 或 backward 数值失败，而是统计代码直接请求把 MPS 标量转换为 CPU `float64`，Apple 后端拒绝。v0 状态：

`M0_PUBLIC_MPS_EXECUTION_ERROR_PRESERVED`

### v1

第一次修复只覆盖方向导数标量，没有覆盖主 loss 的同类转换，因此 v1 在第二个统计位置再次停止。v1 同样完整保留，授权仍为 false。

### v2

第二次修复把所有 CPU64 审计转换集中成“先搬到 CPU，再升到 float64”的单一 helper，并增加 MPS 回归测试与源码搜索测试。协议、射线、阈值和物理 forward 都没有变化。v2 才进入 25 项正式数值判决。

## v2 的关键结果

| 检查 | 实测 | 事前门 | 判定 |
|---|---:|---:|---|
| CPU32 vs MPS32 prediction relative-L2 | `9.5994e-8` | `5e-5` | PASS |
| CPU32 vs CPU64 prediction relative-L2 | `1.1079e-5` | `5e-5` | PASS |
| MPS32 vs CPU64 prediction relative-L2 | `1.1064e-5` | `5e-5` | PASS |
| 最坏单视角 prediction relative-L2 | `3.5952e-5` | `2e-4` | PASS |
| CPU32 vs MPS32 field-gradient relative-L2 | `4.9161e-7` | `5e-4` | PASS |
| CPU32 vs CPU64 field-gradient relative-L2 | `1.0833e-4` | `1e-3` | PASS |
| MPS32 vs CPU64 field-gradient relative-L2 | `1.0829e-4` | `1e-3` | PASS |
| field-gradient cosine defect | `5.8670e-9` | `1e-4` | PASS |
| MPS 方向导数 relative error | `2.7573e-5` | `5e-3` | PASS |
| 三次 MPS prediction 最大漂移 | `0` | `1e-6` | PASS |
| 三次 MPS gradient 最大漂移 | `9.9230e-10` | `1e-5` | PASS |
| 无效射线输出 / 场梯度最大值 | `0 / 0` | `0 / 0` | PASS |
| NaN/Inf | `0` | `0` | PASS |
| sampled driver allocation delta | `1,090,813,952 B` | `2,147,483,648 B` | PASS |
| 清理后 current allocation delta | `0 B` | `67,108,864 B` | PASS |

25 项中最接近阈值的是 sampled driver allocation，使用了约 `50.8%` 的 2 GiB 上限。这个值来自四个同步采样点，不是连续 profiler 得到的真实峰值。

## 152 项包一致性审计

验证器没有重跑 MPS，也没有实现第二套物理 forward。它完成的是保存证据的一致性审计：

- 协议 SHA 和锁文件；
- v0、v1、v2 精确文件集合与 checksums；
- v2 执行 commit `c8f35a1`、tree 和五个源码哈希；
- 25 行 CSV 的顺序、值、阈值和 `value <= threshold` 判决；
- CPU/MPS 内存 delta 和方向导数误差重算；
- 12 个射线选择哈希和 17 个公开输入文件哈希；
- 两个失败包仍关闭授权；
- 不含原始 `.mat/.npy/.pt`、本机绝对路径、VPN 账号或密码；
- 所有越权 claim 仍为 false。

准确名称是 `VALID_M0_PACKAGE_CONSISTENCY_ONLY`，152 / 152。它不能独立重算完整跨设备梯度，因为结果包没有导出 576 万体素的梯度数组。

## 现在到底授权了什么

唯一新增的正授权：

`mps_single_public_phantom_voxel_smoke=true`

具体限制：

- 只能是同一个 opened public Phantom；
- 只能使用结果里 12 个 selection hash 对应的 96 条射线；
- 固定 straight rays、ROI、128 点积分和 chunk 24；
- 只对 voxel field 做短程 optimizer/failure-mode smoke；
- 最多能说“这条机器路径在冻结 batch 上具备实现兼容性”。

仍为 false：

- arbitrary batch 和 full-ray MPS；
- Fourier MLP 或 bounded residual 的 MPS 参数梯度；
- 10 步内存稳定和完整训练；
- 三维重建成功；
- operator learning；
- 跨 field / 跨 geometry 泛化；
- 真实 OERF；
- 优于 DeepONet、FNO、NeRIF 或 NIRP；
- 论文成功和突破。

## 下一步怎么走

两条线可以并行，但不能混写结论：

1. **MPS M0.1：**只在这 96 条冻结射线上做 voxel-field 短程优化，检查连续 optimizer steps、loss 有限性、内存稳定和同一 checkpoint 的 CPU 复核。
2. **CPU D0.6：**冻结 `S0_VOXEL / S1_FOURIER / S2_BOUNDED_RESIDUAL` 单公开 field matched-budget 协议，用它筛参数化和失败模式。

B1/B2 若要转到 MPS，必须另过神经参数多方向梯度和至少 10 步内存门。单 Phantom 的三臂比较仍是 neural implicit instance fitting，不是 DeepONet/FNO 意义下的 operator learning。

## 证据入口

- 冻结协议：`learning_labs/open_nir_bos_m0_public_mps_protocol.json`
- v0 错误包：`learning_labs/results/open_nir_bos_m0_public_mps_v0/summary.json`
- v1 错误包：`learning_labs/results/open_nir_bos_m0_public_mps_v1/summary.json`
- v2 正式摘要：`learning_labs/results/open_nir_bos_m0_public_mps_v2/summary.json`
- 25 项表：`learning_labs/results/open_nir_bos_m0_public_mps_v2/metrics.csv`
- 图：`learning_labs/results/open_nir_bos_m0_public_mps_v2/m0_public_mps_gate.png`
- 152 项包审计：`learning_labs/results/open_nir_bos_m0_public_mps_v2_validation/validation.json`
- 当前授权覆盖层：`docs/open_nir_bos_m0_public_mps_authorization_overlay.json`

## 最终证据边界

**真实增量：公开大体场上的 CPU64/CPU32/MPS32 前向与 voxel-field 反向桥，通过 25 个事前门；两个工程错误包保留；保存证据通过 152 项包一致性审计。**

**没有得到：优化收敛、三维重建、神经参数梯度、算子学习、真实 OERF、泛化、模型优越、论文成功或突破。**
