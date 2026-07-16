# PSU B0 全 support CGLS 与分辨率门禁

> 状态：`SAME_CALL_RESOLUTION_SIGNAL_PASS_NO_FIELD_TRUTH`
>
> 证据边界：真实 PSU 九个 support views、全部 corrected active rays、QMC-16、B0、float64、固定 4 步 CGLS。rotation 40 development 与 final audit 均未打开；没有实验三维真值，没有算法胜出结论。

## 1. 这一轮真正推进了什么

此前的 B0 接口只在每视角 256 条确定性 active 射线、共 2,304 条射线上检查了前向、伴随和资源。它证明离散链正确，但不能回答全量真实反演是否可运行。

本轮实现了：

1. 九视角 `10,628,822` 条 corrected active rays 的确定性流式数据源；
2. 每次完整遍历 329 个 chunks 的逻辑 `A/Aᵀ`；
3. 梯度只计算一次、每个 chunk 重建有限孔径 stencil、伴随累加后再做有限差分转置；
4. 全量 dot-product、调用账本、墙钟时间和进程 RSS；
5. 固定 4 步 CGLS，不从 support curve 或 development 选迭代数；
6. `16³` 与 `32³` 在相同 rays、QMC、dtype 和 calls 下的分辨率对照；
7. 私有重建体、公开聚合 JSON 和论文级图。

逻辑调用定义没有改变：

> 一次 `A` 或 `Aᵀ` 必须完整遍历全部 10,628,822 条选中 support rays；329 个内部 chunks 不能被伪装成 329 次独立 operator calls。

## 2. 全规模 float32 为什么被判 NO-GO

小子集上的 float32 dot-test 很好，不代表一千万射线的 scatter accumulation 仍然足够精确。

| 精度与对偶向量 | relative inner-product defect | 冻结阈值 | 判决 |
|---|---:|---:|---|
| float32 + 真实 observation dual | `8.493e-4` | `2e-5` | FAIL |
| float32 + deterministic random dual | `2.041e-5` | `2e-5` | FAIL |
| float64 + deterministic random dual | `3.458e-15` | `1e-11` | PASS |

随机 float32 的 norm-product defect 只有 `6.07e-9`，说明结构没有断裂，传统相对内积比值还受到高维近正交影响。但它仍略高于预注册阈值，因此没有放宽门槛，而是把正式低分辨率基线切换为 float64。

float64 的代价很小：

- float32 完整 `F+Aᵀ`：约 `48.29 s`；
- float64 完整 `F+Aᵀ`：约 `53.43 s`；
- float64 进程最大 RSS：约 `5.34 GiB`。

这条负结果产生了一个值得继续研究的算法问题：

> 能否用块内 fp32、块间补偿或 float64 reduction、共享 forward/adjoint 核，在接近 fp32 成本下保持全规模伴随一致性？

它可以形成 `Adjoint-Safe Compressed Streaming Operator` 候选，但必须和纯 fp32、纯 float64 在 dot defect、CGLS residual、wall time、内存和 GPU 可迁移性上比较，不能只给它起名字。

## 3. 16³ 全 support 固定 CGLS

配置：

- 九个 support views；
- 全部 `10,628,822` corrected active rays；
- 每条 QMC-16，共约 1.70 亿有限孔径 samples / logical call；
- B0 fixed-denominator；
- float64 CPU；
- 零一层 outer-boundary gauge；
- zero start；
- 固定 4 步 CGLS；
- optimization budget 为 `4F + 5Aᵀ`；
- 最终直接重算另记 `1F`，不混入优化预算。

结果：

| 指标 | 数值 |
|---|---:|
| full-stream dot defect | `3.458e-15` |
| 4 步 optimization wall time | `238.22 s` |
| direct support relative measurement L2 | `0.7877107` |
| direct-vs-recurrence residual difference | `1.739e-16` |
| process max RSS | `5,738,184,704 bytes` |
| outer-boundary max abs | `0` |

九个视角 residual 为 `0.7335` 到 `0.8467`。全部低于 zero-start 的 1，但视角间仍有明显差异。

正确解释：

- 全量接口和 CGLS 数值闭合；
- Mac 能运行真实低分辨率 classical baseline；
- 16³、4 steps 只能解释约一部分 support displacement energy；
- residual 仍高，不是算法成功或失败的单一答案。

它可能同时包含：

1. 16³ 分辨率不足；
2. 固定 4 步过早停止；
3. 缺少 Tikhonov/TV/front-aware regularization；
4. B0 与真实有效积分域不完全一致；
5. 有限孔径、标定、mask、flow-off 噪声或 outlier；
6. 标量线性化模型和真实折射过程之间的失配；
7. 少视角三维不可辨识。

## 4. 32³ 同 calls 分辨率门禁

在看 32³ 结果前冻结：

- rays、views、QMC16、float64、CPU、gauge 全部不变；
- 仍为固定 4 步、`4F + 5Aᵀ`；
- 只有绝对 residual 至少下降 `0.02`，才称为 material resolution signal；
- 不允许用 32³ curve 反选迭代数。

结果：

| 网格 | support residual | pair wall time | max RSS |
|---|---:|---:|---:|
| 16³ | `0.7877107` | `53.43 s` | `5.344 GiB` |
| 32³ | `0.6271325` | `50.55 s` | `5.349 GiB` |

绝对下降：

\[
0.7877107 - 0.6271325 = 0.1605782.
\]

相对改善为 `20.385%`，远高于冻结的 `0.02` 绝对门槛；九个 support views 全部改善。因此：

> 分辨率是当前 support fit 的实质瓶颈，32³ 取代 16³ 成为正式低分辨率参考。

但它仍不是实验三维真值。32³ residual `0.6271` 仍然很高，而且更细网格带来更多可拟合自由度；support improvement 可能包含对测量噪声和模型误差的吸收。只有 rotation 40 和 flow-off/calibration 门能判断它是否迁移。

## 5. 这对“算子学习 + 三维重建”选题意味着什么

### 最稳妥主线

`32³ B0 float64 CGLS` 作为不可绕开的强 classical reference：

1. 固定 support checkpoint/config/hash；
2. 在 rotation 40 上只做一次 development reprojection；
3. 比较 fixed CGLS、Tikhonov-CGLS、TV/primal-dual 和低分辨率 DC-NeRIF；
4. 若 development residual 没超过 repeatability floor，不学习 correction；
5. 若存在跨视角稳定结构残差，再训练 learned preconditioner 或 ray-conditioned correction。

### 更有创新潜力的算法组合

候选不应是“再做一个 FNO”，而应拆成三个可单独证伪的模块：

1. **Adjoint-Safe Streaming Physics Layer**
   - 压缩或缓存不随迭代变化的 sample/stencil；
   - 块内低精度、块间稳定累积；
   - exact/shared forward-adjoint contract；
   - 目标是减少每次约 50 s 的全遍历成本。

2. **Learned Krylov Preconditioner**
   - 输入 `Aᵀb`、当前 residual statistics、view geometry 和 gauge；
   - 输出搜索方向或预条件器，不直接替代数据一致性；
   - 每一步仍通过真实 `A/Aᵀ`；
   - 与 CGLS、Jacobi、Tikhonov 和 PBB 按相同 calls 比较。

3. **Held-out Residual Correction**
   - 只有 rotation 40 显示可重复、跨相机结构残差时才激活；
   - correction 必须在六个 primary rotation blocks 全部同向；
   - flow-off、p95、ambient 和 calibration 任一失败就回退 classical reference。

这条组合比直接比较 DeepONet/FNO 更贴合何远哲方向：核心仍是 BOST 三维场、有限孔径、真实相机几何和可验证 forward model，学习只负责经典逆解确实解决不了的部分。

## 6. 本机与服务器结论

当前不需要租服务器：

- 16³/32³ 完整 pair 都约 51–53 s；
- 4 步 CGLS 约 3.6–4.0 min；
- RSS 约 5.35 GiB；
- 32³ 相比 16³ 没有实质增加内存或时间。

本机不适合的是：

- 对全部 rays 做数千到数万次完整神经训练遍历；
- 64³/128³ 多模型、多种子、大 batch；
- 大型 NeRIF/FNO/DeepONet 系统性超参数搜索。

服务器触发条件仍然是：rotation 40 出现超过 repeatability floor 的正信号，算法、calls、split 和停止规则先冻结，再根据 64³/128³ profile 租 CUDA。当前不创建桌面协助文件。

## 7. 打开 rotation 40 前还要冻结什么

1. 32³ volume、config、public report 和 private report hash；
2. fixed 4-step support checkpoint；
3. rotation 40 的 view mapping、mask 和 renderer contract；
4. development 只允许比较的候选数量；
5. 不使用 development truth 不存在的 field-L2；
6. support residual、active/front-band/p95/ambient 指标；
7. flow-off/repeatability 尚缺时，任何 practical margin 只能标 `PENDING`。

## 8. 给师兄确认的四个问题

1. `epsu/epsv` 的单位、符号和 `Csys` 是否与当前标量密度扰动定义完全一致？
2. 32³ 场应使用自由来流密度边界、ambient 均值 gauge，还是当前零扰动 outer boundary？
3. 是否有 flow-off/reference pairs，用于估计每个相机和 rotation 的 repeatability floor？
4. rotation 40 的七个视角是否允许作为唯一 development run，其他六个 rotation run 保持一次性 final audit？

## 9. 公开复核入口

- [16³ 全 support 摘要](psu_b0_streaming_baseline_public_summary.json)
- [32³ 全 support 摘要](psu_b0_streaming_32cubed_public_summary.json)
- [16³→32³ 严格对照摘要](psu_b0_streaming_resolution_public_summary.json)
- [全规模精度筛选](psu_b0_streaming_precision_public_summary.json)
- [16³ CGLS 四联图](../demo_t16_operator/results/psu_b0_streaming_baseline/psu_b0_streaming_baseline_figure.png)
- [16³/32³ 分辨率图](../demo_t16_operator/results/psu_b0_streaming_resolution/psu_b0_streaming_resolution_figure.png)

公开文件不含本地路径、原始 measurement values、逐射线坐标、私有 volume voxels 或 final-audit 内容。

## 10. 更新：紧凑缓存把同一 32³ reference 加速近三倍

性能 profile 表明约 82% 时间花在每次重复生成 aperture geometry 与 trilinear stencil。现在已把固定 lower-corner index、fraction、valid、projection 和 ray scale 写入 5.017 GB 私有 cache。

冻结的 same-session benchmark 中：

- direct/cached forward relative difference：`0.0`；
- direct/cached adjoint relative difference：`0.0`；
- pair median：`37.92 s → 17.04 s`；
- pair speedup：`2.225×`。

随后完整回放原冻结 4-step CGLS：

- residual 仍为 `0.6271324683999563`，绝对差 `0.0`；
- volume relative difference `1.17e-16`；
- optimization `218.03 s → 74.95 s`；
- speedup `2.909×`。

因此 cached 32³ operator 成为后续快速参考，但它只优化执行，不增加物理正确性或 held-out 证据。

完整入口：[紧凑缓存与快速参考门禁](psu_b0_compact_cache_and_fast_reference_gate_2026-07-16.md) · [缓存 benchmark](psu_b0_compact_cache_public_summary.json) · [CGLS 对照](psu_b0_cached_reference_public_summary.json)
