# PSU rotation-40 多分辨率机制诊断：独立结果审计

## 审计对象

- 协议 commit：`48e32d780e66926dfb493f28e0375fb83c4057f9`
- 正式结果 commit：`87f5e79539f64de8172720b80ebb2efaef7871b3`
- 机器状态：`OPENED_BLOCK_FORWARD_GRID_CHANGE_MATERIAL_MECHANISM_UNRESOLVED`
- 数据范围：已打开的 rotation-40 camera 2/3/4 active rows；没有读取 final rotations

结果提交已经推送到 `origin/main`。因此审计开始时“结果目录尚未 Git 跟踪”的 P1 已由独立 result
commit 解决。Git 与 checksum 仍不是第三方签名或不可删除公共归档，但现在至少形成了协议先于结果、结果字节
可定位的双提交顺序。

## 1. 通过项

| 审计项 | 结果 |
|---|---|
| 协议先于输出 | 通过；协议 commit 与 push 均先于正式目录创建，协议树中不存在输出目录 |
| 本地 import 闭包 | 通过；AST 识别的 8 个运行时本地模块都包含在 12 个冻结文件中 |
| 协议文件字节 | 通过；当前文件、protocol commit 与 summary 中 SHA-256 一致 |
| source evidence | 通过；E68 source commit 为祖先，source config/result 哈希一致 |
| 私有输入预检 | 通过；rotation-40 manifest/数组与两个 frozen volume 的 SHA、shape、dtype 一致 |
| package checksum | 通过；README、CSV、PNG、PDF、summary 五个文件全部匹配 |
| JSON ↔ CSV | 通过；40/40 行逐值一致，最大差为 0 |
| global metric | 通过；独立重组误差不超过 `6.9e-15` |
| cosine / alpha | 通过；公式重算误差为 0 |
| 固定 alpha 曲线 | 通过；独立重算最大差约 `1.0e-14` |
| midpoint 线性 | 通过；direct 与端点线性组合最大绝对差 `1.665e-16` |
| 全量独立 forward | 通过；全部 `3,847,050` rays 的 rel-L2/RMSE/MAE/p95 等与 summary 差为 0 |
| 图表 | 当前图通过；主柱与 cosine 柱的像素位置未发现图文冲突 |
| 隐私 | 当前包通过；未发现私有路径、凭据、测量/预测/几何/场数组或 final rotation 内容 |

公开 validator 还会限制目录文件集、checksum、JSON 树深、list/object 大小、427 个数值叶子预算、
protocol commit 的 12 文件闭包、40 行 CSV、固定 alpha 二次曲线与 claim firewall。它返回
`PUBLIC_AGGREGATE_PACKAGE_VALID`。

## 2. 机器状态为什么与冻结门一致

冻结输入为：

```text
|relL2(A16 x16) - relL2(A32 Ux16)| = 0.013540556 > 0.01
relL2(A32 x32) - relL2(A32 Ux16)   = 0.102787545 >= 0.01
all three camera correction cosines negative = false
camera 2 cosine = +0.276931
```

因此代码先进入 `grid_gap > 0.01` 分支，严格输出当前状态。`MATERIAL` 只表示一个人为数值机制屏被
触发；它不是 p-value、置信区间、物理效应量、工程显著性或因果判决。页面和结果说明应把主语写成
**representation/forward package gap**，不能单独写成 forward grid 的因果影响。

## 3. 仍存在的 P2

### P2-1：`U/D` 不是严格 multigrid 算子

16 与 32 的节点不是嵌套网格。当前 `U/D` 只是 `align_corners=True` 的端点对齐三线性重采样：

```text
||DU-I|| / ||I||           ≈ 0.471
||(UD)^2-UD|| / ||UD||     ≈ 0.344
```

并且 `D` 不是 `U` 的欧氏伴随。因此冻结协议和图中的 `fine-field correction` 是历史术语，必须解释为
`x32-Ux16` 这两个冻结重建之间的**完整场差**；它不是已识别的高频真值、coarse-orthogonal mode、
严格频带或 multigrid fine component。协议与图属于冻结结果，不能在看完结果后重画成另一个实验；本审计
负责公开勘误。

### P2-2：跨网格 raw field norm 不是连续物理能量

runner 的 field norm 是未乘体素体积的向量范数。`||Ux16||/||x16||=2.118`，但加入均匀体积权重后的
近似比例约为 `0.713`。所以 summary 中跨 16/32 网格的 raw norm 只能作数组规模描述，不能说能量增加或
减少。同一 32³ 网格上的场差比值仍可作离散诊断，但不是 field truth。

### P2-3：机器门没有正式使用预测方向差

冻结门只比较 `A16 x16` 与 `A32 Ux16` 两个 residual norm ratio。一般而言，两个预测可以方向不同却具有
相同残差范数。本轮 post-open 另算得到：

```text
||A32 Ux16 - A16 x16|| / ||y||       = 0.111837918
||A32 Ux16 - A16 x16|| / ||A16 x16|| = 0.236381173
```

这次预测差也不小，但它没有进入预先冻结的门，不能倒写成预注册成功。下一协议必须同时约束 residual gap
与 prediction gap。

### P2-4：没有重放 CGLS iteration path

本轮只加载两个冻结的第 4 步终场，没有重跑 `k=1,2,3,4,6,8,12`。因此仍不能区分表示空间、不同离散
系统中的 Krylov 谱滤波轨迹与“第 4 步落在不同早停位置”。这正是下一项 support-only LORO 的任务。

### P2-5：环境不是冻结闭包的一部分

协议绑定了完整本地 Python import 闭包，但没有绑定 Python、PyTorch、NumPy、Matplotlib、OS/CPU 或
container image。当前运行环境已在结果后记录：CPython 3.11.5、torch 2.13.0、NumPy 2.4.6、
Matplotlib 3.11.0、Darwin 25.5 arm64、CPU float64、8 torch threads。它是 post-open provenance，
不能倒称预结果 attestation，也不能证明跨主机 bitwise reproducibility。

### P2-6：公开 validator 的图与隐私证明有限

validator 会检查 PNG 尺寸、PDF header、JSON/CSV、checksum 与敏感 key/path 预算，但不从图像像素重建
所有数据点；`private_payload_detected=false` 也是当前 schema 检查的结论，不是信息论证明。独立代理另行检查
了当前图中主柱/cosine 位置并扫描当前包，结果通过；未来改包仍需重做审计。

## 4. 独立审计授权的最窄结论

> 当前公开包在一个已经打开的 rotation-40 block 上具有内部算术一致性，冻结门严格导出
> `MECHANISM_UNRESOLVED`。本审计不确认三维场精度、因果机制、高频过拟合、CGLS 失效、跨旋转泛化或
> 算法优越性。`U/D` 仅为端点对齐三线性重采样，不构成严格 prolongation/restriction、正交频带分解或
> coarse-space projection；`x32-Ux16` 是两个冻结重建之间的完整场差，而非已识别的细尺度真值。三台
> 相机共享一个 rotation run，不能视为三个独立重复。任何 alpha 都是 post-open 解释量，不得成为部署参数
> 或确认结果。

## 5. 下一门

进入网络前先完成 `SUPPORT-LORO-MECH-1`：

1. support rotations `0/50/90` 整组 leave-one-rotation-out；
2. 重放 CGLS `k=1,2,3,4,6,8,12`，32³ cache 显式 `verify_hashes=True`；
3. 用质量加权 coarse-space projector，不再用普通 resize 冒充投影；
4. 同时冻结 residual gap、prediction gap、equal-rotation、worst-camera、p95 与调用成本门；
5. 三折不能一致归因就输出 `ROTATION40_RUN_SPECIFIC_OR_UNRESOLVED`；
6. rotation-40 只在全部规则和 candidate hashes 冻结后做已打开一致性回放，final rotations 继续封存。

## 6. 入口

- [完整结果与算法转向](psu_rotation40_multiresolution_diagnosis_result_2026-07-19.md)
- [冻结开发协议](psu_rotation40_multiresolution_diagnosis_protocol_2026-07-19.md)
- [post-open 环境记录](psu_rotation40_multiresolution_diagnosis_environment_2026-07-19.json)
- [机器 summary](../demo_t16_operator/results/psu_rotation40_multiresolution_diagnosis_public_v1/summary.json)
- [公开包 validator](../site_tools/validate_psu_rotation40_multiresolution_diagnosis_public.py)
