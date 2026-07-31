# v75 结果前协议：先用一个真正不同的反应流工况做外部门

## 这次为什么不是继续跑 PoolFire

v73 已经在五条 PoolFire 轨迹、三档九视角几何的全部 `1,515/1,515`
个重建单元守住精度，v74.1 又在完整 fresh process 中证明：

```text
loaded q8-K1      2A + 2A^T
Zero-CGLS K4      4A + 4A^T
fresh wall p50    0.56507
fresh wall p90    0.57440
RSS p90           三层全部 <= 1.05
```

但这仍可能只是 PoolFire 这一族形态上的成功。继续增加同类 PoolFire 工程量，
不会回答“这套固定几何低秩算子是否真的能迁移”。

v75 因此只做一件会改变论文判断的事：把**完全不参与候选设计的公开氢气反应流
DNS**作为一次结果前冻结的外部门。

## 为什么选择 BLASTNet Case 3

官方 BLASTNet 页面给出了 22 个 vitiated H2-air freely propagating flame
参数工况。冻结前只读取了：

- 官方页面和 `info.json`；
- Kaggle 数据集版本、许可、文件名和字节数；
- 官方教程的读取方式；
- 没有读取任何网格或密度数值，也没有看任何派生统计或图像。

选择规则在结果前固定为：

1. 必须属于官方 22 个工况；
2. 必须公开提供 `RHO_kgm-3` 和 X/Y/Z 网格；
3. 至少有 25 个有序快照，让 p90 至少覆盖三个尾部帧；
4. 按“所需密度字节数、case id”升序取第一项。

满足条件的工况按密度成本从低到高起始为：

| Case | 快照 | 网格 | Ka_u | Uin | rho-only |
|---:|---:|---:|---:|---:|---:|
| **3** | **25** | **1152x128x128** | **6.8** | **36 m/s** | **1.758 GiB** |
| 4 | 28 | 1152x128x128 | 13 | 36 m/s | 1.969 GiB |
| 6 | 30 | 1408x128x128 | 6.8 | 54 m/s | 2.578 GiB |
| 5 | 42 | 1408x128x128 | 2.4 | 54 m/s | 3.609 GiB |
| 2 | 55 | 1152x128x128 | 2.4 | 36 m/s | 3.867 GiB |

因此选择 Case 3 不是因为它的结果更容易，而是因为它是在能形成时间尾部统计的
候选中获取成本最低。25 个 rho 加三份完整网格和 `info.json` 共
`2,113,947,316` bytes，远小于下载整个 `28.54 GB` 数据集。

如果 Case 3 全门通过，下一批工况现在就固定为 Case 4 和 Case 6；不能看完
Case 3 后再挑一个“更容易”的工况。如果 Case 3 失败，这条候选的首次外部
迁移就记负结果，Case 4/6 不会被用来替换它。

## 数据怎样变成同一个三维逆问题

Case 3 原始数组是 `(source x, source y, source z)=(1152,128,128)`。
官方边界条件说明 source x 是入口/出口和火焰传播方向，所以结果前固定：

```text
source x -> reconstruction z (32 cells)
source y -> reconstruction y (16 cells)
source z -> reconstruction x (16 cells)
```

每帧只做以下确定性处理：

1. 以 little-endian float32、C-order 读取三份完整网格和 rho；
2. 验证 X/Y/Z 网格确实可分离、有限且严格单调；
3. 在完整物理包围盒内做 float64 三线性插值到 `32x16x16`；
4. 使用与 v66-v74 相同的一体素外边界 support；
5. 只在 support 内部减去空间均值，固定 gauge；
6. 不裁剪火焰、不追踪火焰面、不配准、不做幅值归一化。

随后仍使用 v73/v74.1 的三档**同一归一化九视角几何**生成
`y=A_g x`。Case 3 不参与 factor 构造、rank 选择、阈值选择或任何训练。

## 精度门：先做 75 个原子，不合格就停

```text
25 frames x 3 geometries = 75 cells
```

固定候选与对照：

```text
candidate       loaded q8 detector CR -> exact A^T lift -> CGLS K1
equal-cost      Zero-CGLS K2
accuracy ref    Zero-CGLS K4
```

每个 cell 必须同时满足：

- field / full-gradient / interior-gradient / observation 相对 Zero-K4 的
  harm ratio 都不超过 `1.01`；
- 同四项误差都不大于同调用预算的 Zero-K2；
- 实际调用账严格为候选 `2A+2A^T`、参考 `4A+4A^T`。

`75/75` 缺一不可。均值或中位数不能掩盖一帧失败，失败帧的身份必须公开。
独立 validator 不导入正式预处理或 runner，重新读取 29 个原始文件、重建
25 个真值和 75 个 observation，再独立重跑三条臂和所有判据。

## 资源门：只有精度全过才运行

如果且仅如果独立精度验证为 `75/75 PASS`，才运行：

```text
6 reference workers
66 fresh timing workers
33 randomized adjacent complete blocks
25 frames per worker
```

冻结门沿用 v74.1：

- global wall p50 `<=0.90`；
- global wall p90 `<=1.05`；
- 每档 geometry wall p50 `<=1.05`；
- worker-self、worker-tree、pipeline RSS p90 都 `<=1.05`；
- 每档 geometry 的三类 RSS p50 都 `<=1.05`。

下载、协议或测试通过都不是科学结果。Case 3 的精度和资源门都完成独立验证
以前，准确状态始终是：

```text
external_accuracy_result=false
external_resource_result=false
algorithm_breakthrough=false
paper_success=false
```

## 即使通过，能说到哪里

Case 3 全部通过时，只能说：

> 固定的 geometry-compressed q8 warm initializer 在一个未参与设计的公开
> vitiated H2-air DNS 工况上保持同一代理精度，并在当前 Mac 环境保留资源收益。

仍然不能说：

- 已经在整个 BLASTNet 或所有反应流中泛化；
- 已经处理未知相机几何、噪声、曲线光线或折射非线性；
- 这是 neural operator；
- 已经在真实 BOST 实验数据上成功；
- 已经达到论文突破或顶刊结论。

Case 4/6 的预注册扩展、独立公开族尾部和组内真实位移图仍是后续不同证据门。
