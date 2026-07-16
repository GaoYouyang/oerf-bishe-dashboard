# PSU B0 紧凑缓存与快速参考门禁

## 状态与证据边界

- **判决：**`CACHED_32_CUBED_REFERENCE_REPLAY_PASS`
- **使用范围：**九个 support views、全部 corrected active rays、B0、QMC-16、32³、float64。
- **仍然封存：**rotation 40 development 与 final audit 均未打开。
- **私有边界：**缓存和重建体只留在 `private_library/`，不上传 GitHub Pages。

## 1. 为什么先优化算子，而不是同时跑更多模型

此前 16³ 和 32³ 的完整 `F+Aᵀ` 都约 51–53 秒，而体素数增加 8 倍后时间几乎没变。chunk profile 进一步显示：

- aperture geometry 生成约占单 chunk `32%`；
- trilinear stencil 构建约占 `50%`；
- 真正 forward 数值计算约占 `17%`；
- 几何与 stencil 合计约占 `82%`。

这些量在每次 CGLS、Tikhonov、TV 或 learned preconditioner 迭代中都不会改变，却被重复生成。因此当前最有效的并行不是同时开多个完整反演，而是：

1. 几何与插值坐标只构建一次；
2. 重型 inverse 串行复用缓存；
3. 测试、网页、文档和绘图并行；
4. 将来有多张 GPU 时，再让不同模型或 seed 真正并行。

## 2. 缓存保存了什么

每个固定有限孔径样本不再保存完整八角点 indices/weights，而保存：

- lower-corner linear index；
- 三个局部分数坐标；
- 有效域 mask；
- 每条 ray 的 `Ru/Rv` 投影；
- 已冻结的 `L·Csys/M` scale；
- support observation；
- 329 个原始 chunk 的连续边界与 view id。

32³/QMC-16 全 support cache：

| 项目 | 数值 |
|---|---:|
| rays | `10,628,822` |
| aperture samples | `170,061,152` |
| chunks | `329` |
| fraction dtype | `float64` |
| cache bytes | `5,016,803,984` |
| build wall time | `14.94 s` |

缓存 manifest、shape、dtype 和 SHA-256 均在本地验证。loader 会拒绝未完成状态、缺失数组、网格不匹配、chunk 不连续或 checksum 错误。

## 3. 数值等价门禁

在跑完整 benchmark 前冻结：

- 同一 32³ grid；
- 同一 10,628,822 rays；
- 同一 QMC-16；
- 同一 float64；
- 同一随机 volume / dual；
- 同一 8 threads；
- direct 与 cached 各两次完整 pair；
- forward/adjoint relative difference 上限 `1e-14`；
- cached dot defect 上限 `1e-11`。

结果：

| 指标 | 结果 | 阈值 | 判决 |
|---|---:|---:|---|
| forward relative difference | `0.0` | `1e-14` | PASS |
| adjoint relative difference | `0.0` | `1e-14` | PASS |
| cached relative dot defect | `1.97e-14` | `1e-11` | PASS |

因此紧凑缓存没有改变声明的离散算子。

## 4. same-session 性能门禁

执行顺序在看结果前冻结为：

`direct → cached → cached → direct`

| 模式 | pair times | median |
|---|---|---:|
| direct geometry rebuild | `37.25 / 38.59 s` | `37.92 s` |
| compact cache | `17.05 / 17.03 s` | `17.04 s` |

中位加速：

\[
\frac{37.9204}{17.0396}=2.2254\times.
\]

冻结门槛是 `1.5×`，因此通过。benchmark 进程同时保留 direct/cached operator、随机 dual 和比较输出，最大 RSS 为 `10,297,163,776 bytes`；这不是单独缓存 inverse 的常驻内存。

不能把该结果写成通用硬件速度。它只证明在这台 Mac、同一会话、同一数据和线程设置下，重复几何构建是主瓶颈，缓存能实质缩短完整调用。

## 5. 固定 4 步 CGLS 回放

缓存基准通过后，才用原来的冻结预算完整回放：

- zero start；
- 4 steps；
- optimization `4F + 5Aᵀ`；
- 最终直接评估另记 `1F`；
- 不从缓存曲线重选迭代数。

| 指标 | direct reference | cached replay |
|---|---:|---:|
| support relative measurement L2 | `0.6271324683999563` | `0.6271324683999563` |
| optimization wall time | `218.03 s` | `74.95 s` |
| direct-vs-recurrence residual | `2.25e-16` 量级 | `2.25e-16` |

对私有重建体逐元素比较：

- volume relative difference：`1.17e-16`；
- max absolute difference：`1.39e-17`；
- support residual absolute difference：`0.0`；
- optimization speedup：`2.909×`。

因此缓存版本正式取代“每次重建 geometry/stencil”的实现，成为 32³ support 快速参考。原 direct 版本仍保留作独立复核。

## 6. 这对算法开发有什么实际意义

现在一次完整 forward 或 adjoint 约 `8 s`，而不是约 `19–27 s`。这会改变可执行范围：

- 4 步 CGLS 可在约 75 秒内完成；
- 约 100 次完整 operator calls 的消融可在十几分钟量级完成；
- Tikhonov-CGLS、TV/primal-dual 和固定迭代 Krylov 消融可以在本机严肃运行；
- 小型 learned preconditioner 可以先在低 calls、少 seeds 下证伪；
- 数千次全量调用仍可能需要数小时到更久，64³/128³ 多模型多种子仍需后续 profile。

下一批算法优先级：

1. **Classical neighbors**：CGLS、Tikhonov-CGLS、TV/primal-dual、Jacobi/PBB，全部按同 calls 比较；
2. **Learned Krylov preconditioner**：网络只提出预条件方向，每一步仍通过 exact cached `A/Aᵀ`；
3. **Adjoint-safe mixed precision**：比较 float32 fractions、补偿累积与 float64 reference 的 dot/residual/time/cache size；
4. **Held-out residual correction**：只有 rotation 40 出现超过 flow-off floor 的稳定结构残差才启动；
5. **Multiresolution route**：低分辨率 warm start 是否能在相同 calls 下加速 32³，而不是只增加总预算。

缓存本身是可靠研究基础设施，不是足够的论文创新。论文创新仍必须解决真实物理困境，例如有限孔径失配、几何不确定、少视角零空间、反应流薄前沿或 held-out 相机残差。

## 7. 网速和服务器判决

当前不是网速问题：

- PSU support 数据和 cache 都在本地；
- `A/Aᵀ` 与 CGLS 不访问互联网；
- 实测下载吞吐约 `310 Mbps`，仅 RTT 偏高；
- 2.23×/2.91× 加速来自移除重复计算，不来自网络调整。

当前不租服务器。触发条件仍是 rotation 40 的改善超过 repeatability floor、候选和停止规则冻结，并且 64³/128³ 多模型多种子 profile 证明本机不够。

## 8. 公开复核入口

- [缓存 benchmark JSON](psu_b0_compact_cache_public_summary.json)
- [缓存 CGLS 运行摘要](psu_b0_cached_reference_run_public_summary.json)
- [direct/cached reference 对照](psu_b0_cached_reference_public_summary.json)
- [缓存门禁四联图](../demo_t16_operator/results/psu_b0_compact_cache/psu_b0_compact_cache_figure.png)
- [全 support CGLS 与分辨率门禁](psu_b0_full_support_cgls_and_resolution_gate_2026-07-16.md)
- [缓存 builder/reader](../site_tools/psu_b0_compact_cache.py)
- [缓存 benchmark runner](../site_tools/benchmark_psu_b0_compact_cache.py)
- [缓存 reference comparator](../site_tools/compare_psu_b0_cached_reference.py)

公开文件不含本地路径、support measurement values、逐射线坐标、缓存数组、重建 voxels 或私有文件哈希。
