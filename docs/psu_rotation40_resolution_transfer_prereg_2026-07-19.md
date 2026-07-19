# PSU rotation-40：16³→32³ 网格分辨率迁移审计预注册

> 状态：`FROZEN_BEFORE_16_CUBED_ROTATION40_SCORE`
>
> 留出单位：`ROTATION_RUN_NOT_CAMERA`
>
> 证据范围：已经打开的 rotation-40 development；真实 BOS 位移；无实验三维真值；不打开 final rotations。

## 1. 这一步只回答一个窄问题

九个 support views 上，在固定四步 CGLS、QMC-16、float64 和相同逻辑调用数下，`32³` 的 measurement relative-L2 是 `0.6271324684`，`16³` 是 `0.7877106877`。这两个数以及两个场的哈希都在本协议冻结前已知。support 拟合改善约 20.4%，但它不能证明更细离散网格恢复了更真实的三维场：额外自由度也可能吸收 support 几何特有的噪声、标定误差或 forward mismatch。

rotation-40 已按既有 development 协议打开，冻结 `32³` 场曾得到 pooled vector relative-L2 `0.959591`；`16³` 尚未在同一批全量 active rows 上评分。因此本审计只问：

> support 上的固定调用数网格分辨率收益，是否方向一致地迁移到**同一组三台物理相机在未参与重建的 rotation-40 运行**？

support 与评分都使用 camera 2、3、4。support rotations 是 0°、50°、90°，评分 rotation 是 40°；所以留出的是一次 rotation run，不是新相机。三台相机的行共享同一次物理运行，不能冒充三个独立重复。本次独立 rotation block 数是 `1`。

## 2. 一级来源与物理边界

PSU 数据论文公开了 7 台相机 × 10 次模型旋转共 70 views、九视角 limited-data reconstruction、validation deflections、标定与 mask，以及 NIRT/TV 示例；它也把 limited views、calibration 与 deflection sensing 列为潜在伪影来源。数据 DOI 为 [10.26208/1VE2-5C19](https://doi.org/10.26208/1VE2-5C19)，开放论文见 [arXiv:2508.17120v2](https://arxiv.org/html/2508.17120) 与 [Experiments in Fluids 正式版](https://doi.org/10.1007/s00348-026-04189-z)。

这些来源支持“用未参与重建的旋转 view 做 image-space validation”，不提供 rotation-40 的独立三维真值，也不授权把重投影误差称为 field error。

## 3. 两个且仅两个冻结对象

| 对象 | 名义网格值 | outer-zero 后自由内点 | 网格节点间距 | 已知 support rel-L2 |
|---|---:|---:|---:|---:|
| `support_cgls4_16cubed` | 4,096 | 2,744 | 14.667 mm | 0.7877106877 |
| `support_cgls4_32cubed` | 32,768 | 27,000 | 7.097 mm | 0.6271324684 |

两个对象都来自 camera 2/3/4 × rotation 0/50/90 的九个全 active support views；使用普通未加权 vector-L2、零场初始化、四步 CGLS、`4F + 5A^T`、QMC-16、无 positivity 和一层 outer grid-node zero gauge。场、私有生成报告、公开摘要和 support 分辨率对照摘要均以 SHA-256 冻结。

`32³` 的名义网格值是 `16³` 的 8 倍，自由内点约 9.84 倍。相同 solver call 数不等于相同参数量、内存或计算成本；因此这只是同一重建流程的**网格敏感性与 image-space 迁移检查**，不是算法公平比较，也不能宣称 `32³` “击败” `16³`。

不改变迭代数，不重新重建，不从 rotation-40 拟合 amplitude scale，不删除 ray，不按结果选择相机、metric 或阈值，不加入第三个候选。

## 4. 开结果前的不可变绑定

正式运行前采用双提交：

1. 先提交配置、runner、测试、active-ray store、forward operator、metric 实现和本说明；正式结果目录必须不存在。
2. 在该 protocol commit 上一次性生成 attestation，记录 protocol SHA、全部受监控代码哈希、候选场与来源报告哈希、payload/geometry 哈希、support split 以及 `ROTATION_RUN_NOT_CAMERA` 身份；再单独提交 attestation。
3. runner 仅在 protocol commit 是当前 HEAD 祖先、所有受监控文件仍与该 commit 一致、attestation 已跟踪且无未提交改动、正式输出仍不存在时运行。

运行前还会逐文件复核 camera 2/3/4 的 payload shard、geometry manifest 与全部 `.npy` 的 SHA-256、shape 和 dtype，并验证 payload manifest 到 geometry manifest 的交叉绑定。公开包不出现本地路径或逐射线数组。

## 5. 冻结指标与机器判决

主指标是全 active rows 合并后的 ray-count-weighted pooled 值：

```text
Delta_R40 = pooled_relL2_16 - pooled_relL2_32
```

结果前冻结的数值筛查线是 `Delta_R40 >= 0.01`。`0.01` 不是物理、工程或实际显著性门槛；在 flow-off 重复测量和不确定度估计到位前，只能称为 **predeclared numerical screen**。同时要求 camera 2、3、4 的 `32³` relative-L2 各自不高于 `16³`，容差为 `0`。

| 条件 | 机器判决 | 允许解释 |
|---|---|---|
| pooled 过数值线且三相机均不退化 | `RESOLUTION_TRANSFER_SIGNAL_PASS_NO_FIELD_TRUTH` | 本次已打开旋转上出现 image-space 迁移信号 |
| pooled 过线但任一相机退化 | `POOLED_TRANSFER_WITH_CAMERA_HARM_NO_GO` | pooled 掩盖相机异质性，不授权继续扩大分辨率 |
| pooled 改善 < 0.01 | `SUPPORT_RESOLUTION_GAIN_DID_NOT_CLEAR_NUMERICAL_TRANSFER_GATE_NO_GO` | support 收益未通过预注册的数值迁移筛查 |

除 pooled 外，必须同时公开 equal-camera macro-average、逐相机 delta 与 worst-camera delta。macro 只给三台相机等权，仍不是三个独立物理重复。

无论哪个判决，都不授权：实验 field-L2、算法优越性、计算公平性、实际显著性、跨 session/rig 泛化、神经算子成功、final rotations 或论文主结论。

## 6. 成本、诊断与公开边界

每个候选保存 pooled 与逐相机的 vector relative-L2、component RMSE/MAE、residual magnitude p95、measured/predicted vector RMS；同时记录名义网格值、自由内点、节点间距、forward wall time、候选端到端时间和调用后 peak RSS。成本只用于披露，不进入主判决。

公开结果包只含聚合 JSON/CSV、PNG/PDF 图、README 和 checksum。逐射线 measurement、prediction、geometry、私有 volume、原始 payload 与 final rotations 不进入 Git 或 GitHub Pages；runner 在写盘前递归执行敏感 key 与私有路径检查。

## 7. 结果怎样约束下一算法

- **若迁移通过：**更细离散仍有 image-space 收益，但若 `32³` residual 仍高，下一步应在同分辨率比较 TV/H1/NeRIF 表示与真实 held-out view，不能把“更大模型”当作原因。
- **若 pooled 通过而相机受损：**优先研究 camera-conditioned calibration、噪声或 geometry mismatch；保留逐相机 tail，不能只优化 pooled loss。
- **若数值迁移失败：**停止用更细网格吸收 support residual；优先检查 B0/cone-ray、位移偏差、标定、mask 与 rotation/session mismatch。

这一步不是新算法成果，而是决定后续算子学习究竟应补表示能力，还是应显式建模观测算子失配。
