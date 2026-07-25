# PoolFire rho 数据桥：首轨迹头审计与流式抽取合同

> 状态：`PUBLIC_TRAJECTORY_SHA_CRC_RHO_READY / PHYSICAL_UNITS_PENDING`
>
> 证据等级：`PUBLIC_CFD_RAW_BRIDGE_ONLY`
>
> 这份报告证明一条公开 CFD train trajectory 已通过完整 size/SHA、ZIP/NPY、full-resolution rho 与派生 checksum 门。它不证明单位、BOST forward、三维重建、warm start 或泛化。

## 1. 为什么先做这一层

师兄已经确认不需要从零运行三维 CFD，可以直接使用公开数据。当前选择 REALM PoolFire 的第一条 train trajectory：

- case：`p=14kw_size=03`
- 公开压缩文件大小：`6,428,997,975 bytes`
- 公开 SHA-256：`6080ddcc595f38296bff5bfabbad098c2c48cf402ca68bee112a16875781383c`
- 来源：[REALM PoolFire on Hugging Face](https://huggingface.co/datasets/realm-bench/realm-bench-PoolFire/tree/main/data/train)

下载任务使用固定 HTTP Range、精确 `Content-Range` 和分块字节数校验。2026-07-23，最终文件大小为 `6,428,997,975 bytes`，独立复算 SHA-256 与公开值逐字一致；随后才启动抽取。

## 2. 已经从文件前缀确认的事实

下载完成前，先从已验证连续性的本地前缀读取 ZIP local header，并从 deflate 流起点解出 NPY header；下载完成后，这些信息又由完整 ZIP/CRC 读取复核。当前得到：

| 项目 | 已确认值 |
|---|---|
| ZIP member | `data.npy` |
| ZIP compression | deflate，ZIP64 size sentinel |
| NPY version | `1.0` |
| shape | `(101, 9, 80, 80, 200)` |
| memory order | C-order |
| dtype | little-endian `float64` |
| NPY 数值 payload | `9,308,160,000 bytes` |

对应 metadata：

- metadata SHA-256：`8ecb161c1de3b038a8e46d35320a85f28054e4b3d7f785afe09fd000a3060738`；
- `times.shape == (101,)`，字符串时间从 30 到 32；
- `variables == [CH4, CO2, H2O, O2, T, rho, Ux, Uy, Uz]`；
- `rho` 是 channel 5；
- train/val/test 是 11/2/2 条完整 trajectory；
- `coords.shape == (3, 80, 80, 200)`。

坐标目前还有一条必须显式解决的冲突：三个轴都按降序保存，数值跨度约为 `1.2 × 1.2 × 3.0`，而本地 REALM README 写的是 `3 × 3 × 3 m³`。在确认坐标单位、轴方向、cell center/edge 和数据版本前，不能据此计算物理梯度尺度。

这也解释了为什么不能直接把整个 trajectory `np.load` 到内存：压缩包约 6.43 GB，单个数组解压后仅数值 payload 就约 9.31 GB，还没有计入复制、类型转换、网络输入和求解器工作区。

## 3. 统计文件暂不可信

本机两份 PoolFire YAML 互相矛盾：

- 旧 case 配置写 21 个时间步，字段统计看起来具有物理量级；
- 数据目录副本写 101 个时间步，但九个变量的均值、标准差和极值呈现异常相似的大数。

因此当前不使用 YAML 的 mean/std 做归一化。必须从通过 SHA 的真实 trajectory 流式重算 `rho` 的 min/max/mean/std、NaN/Inf，再决定训练归一化。

## 4. 流式抽取器

新增的 `extract_poolfire_rho.py` 按以下顺序 fail closed：

1. 对 trajectory 与 metadata 分别要求预期 SHA-256；
2. trajectory 只打开一次：同一个文件描述符先验 SHA、读取 ZIP/NPY、末尾再验 SHA，路径被替换或原文件中途变化都会被拒绝；
3. metadata 从已哈希的同一份内存字节解析，并把 SHA 写入 manifest；
4. 检查 trajectory basename 在 metadata 中只属于一个 split，默认拒绝 test split；
5. 检查 ZIP 只有 `data.npy`，shape、实数 dtype、C-order、time、variable 和 spatial metadata 一致；
6. 逐个 `(time, channel)` 解压，每次最多保留一个三维场；
7. 对 **101 帧全部 full-resolution rho** 检查 finite、严格正值并稳定累计统计，不允许 stride 把坏点漏掉；
8. 将 `rho` 转成 `float32` 后再次检查 finite，再写入 memmap；
9. 读到 member EOF，触发 ZIP CRC 检查；
10. 写出数组、manifest、checksums 和 `READY.json`；目标目录用独占 `mkdir` 保留，`READY.json` 最后提交。没有 READY 的中间目录必须视为失败，旧结果绝不覆盖。

第一份数据桥保留完整空间分辨率，若完整 101 帧都通过，输出：

```text
rho.shape = (101, 80, 80, 200)
rho.dtype = float32
rho payload = 517,120,000 bytes
```

不再在原始数据桥直接用 `(2,2,4)` 抽点。BOS 依赖密度梯度，直接抽点会改变前沿强度；低分辨率训练副本必须在下一层用冻结的抗混叠或体积平均算子生成，并与 full-resolution truth 一起保留。

这仍是**绝对 CFD 密度**，manifest 会明确写：

```text
absolute rho copied from public CFD; reference/gauge not applied
unit_status = unverified
```

它不会提前转换成 `Δrho` 或 `Δn`，因为单位、环境 reference 和 Gladstone-Dale 条件尚未由师兄确认。

## 5. 当前机器验证

- `test_extract_poolfire_rho.py` 与 NPZ inspector 合计：`20 passed`；
- 第二套 Python 环境：`Ran 20 tests ... OK`；
- 覆盖 source/metadata SHA、路径替换竞态、ZIP 损坏与路径脱敏、shape/C-order、float32 overflow、未采样坏点、test split、并发输出保护、真实 checksum 与 READY 绑定；
- 完整 source SHA：`6080ddcc595f38296bff5bfabbad098c2c48cf402ca68bee112a16875781383c`；
- `rho.npy`、`coords.npy`、`times.npy`、`manifest.json` 四个 checksum 均通过，`READY.json` 又绑定 manifest 与 checksum 文件的 SHA；
- 完成后已停止下载与抽取后台任务，避免成功任务被 keepalive 重复运行。

独立 mmap 复核没有复用 manifest 的统计结论：

| 项目 | 结果 |
|---|---|
| rho shape / dtype | `(101, 80, 80, 200)` / `float32` |
| finite / positive | `129,280,000 / 129,280,000`，全部严格正值 |
| rho min / max | `0.1889829934 / 1.1793500185` |
| rho mean / std | `1.1608747931 / 0.0605809878` |
| 每帧 mean 范围 | `1.1604899379 – 1.1611630479` |
| time | `30.00 – 32.00`，101 帧，步长约 `0.02`，单位未确认 |
| x / y | `0.5925 → -0.5925`，80 点，步长约 `-0.015` |
| z | `2.9925 → 0.0075`，200 点，步长约 `-0.015` |

这些数值与常见 SI 密度量级相容，但“看起来合理”不是单位证据；在数据说明或师兄确认前，不能把 rho 单位正式写成 `kg/m³`。

## 6. 下一道物理门

rho bundle 通过后仍不能直接训练 C0。还要依次完成：

1. 重算真实 rho 范围并核对单位；
2. 解决坐标降序、物理单位、domain 尺度和 cell center/edge 冲突；
3. 冻结由部署条件可得的 `rho_ref`，禁止读取 test truth 体均值；
4. 确认师兄工具输入的是 `rho`、`n`、`n-1` 还是 `Δn`；
5. 用常数场检查偏折接近零，用线性场检查方向、正负号和尺度；
6. 确认九视角输出是角度、像素位移还是归一化坐标；
7. 得到与 forward 匹配的 adjoint/VJP，并通过 dot test；
8. 用不同分辨率或扰动参数的 forward 生成观测、用独立较粗算子反演，避免 inverse crime；
9. 先跑 Zero、BP、CGLS/PCGLS，再允许训练 C0。

PoolFire 的两个 test case 只是在功率与尺寸的**组合**上留出；每个功率值和尺寸值都分别在训练中出现。后续只能称 combination holdout，不能写成“未见功率 OOD”或“未见尺度 OOD”。101 帧也不是 101 个独立样本，统计单位必须是 trajectory，时间帧只能做块 bootstrap。

## 7. 允许与禁止的表述

当前允许：

> PoolFire 一条 train trajectory 已通过完整 size/SHA、ZIP CRC、full-resolution rho、派生 checksum 和 READY 门；它现在是可复现的公开 CFD 原始桥，但单位、光学 forward 和算法效果仍未闭合。

当前禁止：

- “PoolFire 已经完成 BOST 接入”；
- “已经生成可靠 BOS 数据”；
- “已经完成三维重建”；
- “warm start 已经提速”；
- “算法可泛化或可投稿”。
