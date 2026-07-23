# PSU B0 三维重建接口门禁

> 状态：`REAL_SUPPORT_GEOMETRY_INTERFACE_PASS_NO_RECONSTRUCTION`
>
> 证据边界：真实 PSU 九视角几何 + 合成逆解；没有读取 rotation 40 development 数组，没有打开 final audit，没有真实三维重建或算法胜出结论。

## 1. 这一轮解决了什么

前面的 B0-B3 审计回答了“每条光线应该在哪个空间域积分”。这一轮进一步回答：

1. 如何把一个规则网格上的标量折射率/密度扰动场变成相机 `u/v` 位移；
2. 如何用完全相同的离散原语构造精确伴随 `Aᵀ`；
3. 如何把有限孔径域外样本保留在固定分母中，而不是对幸存样本重新归一化；
4. 如何显式处理 BOS 只测梯度导致的加性常数零空间；
5. 当前 Mac 是否足以继续做 16³/32³ 基线；
6. 在打开 rotation 40 development 前，哪些接口和评估规则必须冻结。

它把“先做一个低分辨率 B0/PBB 或 NeRIF”从一句建议推进成了可测试代码合同。

## 2. 离散前向模型

未知量是规则网格上的标量扰动场：

\[
x \in \mathbb{R}^{N_z\times N_y\times N_x}.
\]

每个有限孔径样本点 \(p_{r,m}\) 的预测使用同一条链：

\[
\hat{\mathbf d}_r
=
\frac{L_r C_r}{M}
\sum_{m=1}^{M}
\mathbf P_r
\mathcal I_{p_{r,m}}
D x.
\]

其中：

- \(D\) 是声明的三维有限差分梯度，输出顺序为 `dx,dy,dz`；
- \(\mathcal I\) 是规则网格上的三线性插值；
- \(\mathbf P_r\) 用真实 `Ruvecs/Rvvecs` 把三维梯度投影到相机 `u/v`；
- \(L_r\) 是 B0 前向 box 内路径长度；
- \(C_r\) 是真实 `Csys_all`；
- \(M\) 是原始固定样本数。

样本点若落在 B0 外，其八个插值权重全部置零，但 \(M\) 不变。这就是 `fixed-denominator B0 indicator`，禁止 survivor renormalization。

## 3. 为什么必须有 gauge

BOS 对标量场只通过梯度敏感：

\[
D(x+c)=Dx.
\]

因此任何常数 \(c\) 都位于观测零空间。若不固定 gauge：

- 两个预测完全相同的场可能只因常数偏移而得到不同 field-L2；
- 优化器可能在不可观测方向漂移；
- “三维场更准”会混入任意参考密度选择。

当前低分辨率接口把未知量定义为相对自由来流的标量扰动，并冻结一层外边界为零。它是数值 gauge，不是对真实激波外全部流场精确为零的物理声称。后续可把它与已知边界密度、ambient 区均值或 CFD reference 做消融，但 final audit 前必须固定。

## 4. 精确伴随不是“再训练一个反向网络”

前向离散链是：

```text
support × scalar field
  → finite-difference gradient
  → trilinear gather
  → Ru/Rv projection
  → L·Csys/M scaling
```

伴随严格反向执行：

```text
u/v residual
  → Ru/Rv transpose
  → trilinear scatter-add
  → finite-difference transpose
  → support transpose
```

没有独立 learned adjoint。若未来学习 operator correction，forward 与 adjoint也必须共享同一核或通过同一可微 forward 的 VJP 构造，不能各训一个黑盒。

## 5. 代码门禁

当前新增测试覆盖：

1. 有限差分 `D/Dᵀ` 内积恒等式；
2. 完整 BOST `A/Aᵀ` dot-product；
3. matrix-free forward 与 tiny materialized matrix 完全一致；
4. 一半样本越 B0 时固定分母只保留一半贡献；
5. 逻辑 forward/adjoint 调用记账；
6. Dirichlet gauge 与 support；
7. 合成固定迭代逆解；
8. 私有真实几何摘要到公开摘要的字段隔离；
9. 16³/32³ CPU/MPS 图表生成。

核心入口：

- [B0 重建接口](../demo_t16_operator/psu_b0_reconstruction_interface.py)
- [冻结配置](../demo_t16_operator/configs/psu_b0_reconstruction_interface_v1.json)
- [合成闭环 runner](../site_tools/run_psu_b0_interface_fixture.py)
- [真实几何接口审计](../site_tools/run_psu_b0_real_interface_audit.py)
- [公开机器摘要](psu_b0_reconstruction_interface_public_summary.json)

## 6. 合成闭环结果

合成 fixture 使用 12³ 标量场、3 个正交视角方向、147 条射线、每条 16 个有限孔径样本和固定 60 次 Landweber：

| 指标 | 结果 | 正确解释 |
|---|---:|---|
| `A/Aᵀ` relative dot defect | `6.78e-15` | 离散伴随闭合 |
| 初始 measurement relative L2 | `1.0` | 零场起点 |
| 60 次后 measurement relative L2 | `0.005028` | 同一合成 forward 下拟合闭合 |
| fixture field relative L2 | `0.4504` | 仍有显著三维场误差 |
| optimization calls | `61 F / 60 Aᵀ` | 含一次最终评估 forward |

最重要的不是残差降到 0.5%，而是它和 45.0% field-L2 同时出现。这是一个直接反例：

> 很好的投影一致性不等于唯一、准确的三维场。

因此 PSU final audit 即使通过，也只能先称 held-out image-space consistency；实验数据没有独立三维真值，不能输出 field-L2。

## 7. 真实九视角几何结果

为了在不打开 development/final audit 的前提下验证真实接口，从九个 support views 各取 256 条 active 射线。选择只依据排序后的 active mask 分位，不读取位移大小，不按模型残差筛选：

- 总射线：`2,304`；
- QMC-16 总样本：`36,864`；
- 所有选中 active 中心线均命中 B0；
- 这一子集的有限孔径样本全部在 B0；
- 原始方向范数范围为 `0.9999999614` 到 `1.0000000393`。

伴随结果：

| 网格 | CPU float64 dot defect | MPS float32 dot defect |
|---|---:|---:|
| 16³ | `4.97e-16` | `7.28e-8` |
| 32³ | `1.78e-16` | `9.89e-8` |

两种设备均通过预先冻结的 `1e-11` / `2e-5` 门槛。

## 8. 本机资源画像

以下只是固定 2,304-ray 子集的单调用中位数，不是全量九视角训练速度：

| 网格 | CPU F | CPU Aᵀ | MPS F | MPS Aᵀ | MPS current allocation |
|---|---:|---:|---:|---:|---:|
| 16³ | 1.091 ms | 0.703 ms | 0.944 ms | 1.577 ms | 3.54 MiB |
| 32³ | 1.274 ms | 0.991 ms | 0.903 ms | 1.394 ms | 7.78 MiB |

这里 MPS 不比 CPU 全面快，是因为任务太小，设备调度开销占比高。不能据此说 CPU 更适合最终训练，也不能把时间线性外推到 1,000 万 active rays。

当前硬件判决仍是：

- 16³/32³ 接口、dot-test、小批量 PBB/Landweber、轻量网络：Mac 足够；
- 全 support streaming baseline：先实测；
- 64³/128³、多模型、多种子：只有 development 出现超过重复性地板的正信号后再租 CUDA。

## 9. 已校验但仍封存的 archive

官方 `2025-05` archive 已完整下载并通过 CRC：

- 大小：`4,095,655,393` bytes；
- 包含 rotation 30、40、50、60、70、80、90 的 processed deflection MAT；
- rotation 40 是唯一 development run；
- rotation 30、60、70、80 含 final-audit 数据，仍不得提取用于看图、选模或调停止。

CRC 通过只证明文件完整，不授权访问审计答案。

## 10. 下一步

接口已冻结，但还缺真正的真实 support inverse：

1. 构造可流式遍历九视角 active rays 的逻辑 `F/F*`；
2. 用 16³ 跑固定预算 Landweber、PBB 和 CGLS；
3. 记录完整 support traversal 的 wall time、峰值内存和 calls；
4. 检查常数零空间、边界 gauge 和正则化对重投影的影响；
5. 冻结 B0 baseline checkpoint/config/prediction hash；
6. 只有此后才提取 rotation 40，并且只用于一次 development 选择；
7. 若 B0 residual 已接近 flow-off floor，不继续堆神经算子；
8. 若 residual 存在稳定结构，再进入 DC-NeRIF、DCO-ReSeSOp 或 RayKernel-DCO。

当前不需要用户租 GPU，也不需要在桌面创建执行文件。

## 11. 图

- [接口门禁四联图 PNG](../demo_t16_operator/results/psu_b0_interface_audit/psu_b0_interface_audit_figure.png)
- [接口门禁四联图 PDF](../demo_t16_operator/results/psu_b0_interface_audit/psu_b0_interface_audit_figure.pdf)
- [合成真值/重建/误差图](../demo_t16_operator/results/psu_b0_interface_fixture/psu_b0_interface_fixture.png)

这些图明确把真实几何 dot-test、子集资源画像和合成逆解分开；没有任何真实 PSU reconstruction slice。

