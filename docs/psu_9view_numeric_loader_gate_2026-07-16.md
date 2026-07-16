# PSU 9-view 真实数据数值 loader 门禁

更新时间：2026-07-16

结论标签：`LOADER_NUMERIC_CONTRACT_CONFORMANT`

证据边界：`SELECTED_NUMERIC_LOADER_CONTRACT_ONLY_NO_NIRT_RECONSTRUCTION`

## 1. 这次到底完成了什么

此前只知道 5.23 GB 的 `HSOF_9CAM_RT.mat` 包含作者 loader 需要的变量名和 shape。本轮增加了一个低内存 MATLAB v5 reader，对选中的压缩变量做三件事：

1. 在 MCOS subsystem 前建立变量索引，不反序列化相机对象；
2. 完整解压被选变量的 zlib stream，并计算数值 payload SHA-256；
3. 小变量完整读取，大变量只保留确定性几何地标或成组测量向量，不在内存中物化整块数组。

因此现在不只是“字段像是对的”，还验证了官方 9-view 快照的核心数值几何、存储顺序和单位契约。原始 ZIP、MAT、作者源码和私有审计 JSON 仍只在被 Git 忽略的本地资料库中；GitHub 只发布我们自己写的 reader、测试和不含原始数据的结论。

以 392,000,000-byte 的 `X` 数值 payload 做一次本机完整流校验，`/usr/bin/time -l` 记录 maximum resident set size 为 `52,838,400 bytes`，约 50.4 MiB；真实耗时 0.42 s 受文件缓存影响，只用于说明内存有界，不作为速度 benchmark。

## 2. 真实数据读出的配置

`siz` 被完整解码为：

```text
[2160, 2560, 9]
```

即每个视图为 2160 × 2560，9 个视图共有：

```text
2160 × 2560 × 9 = 49,766,400 measurements
```

这与 `c`、`v`、`Ruvecs`、`Rvvecs`、`Rapvec` 等字段的 measurement width 完全一致。官方 readme 与 MATLAB 预处理脚本说明，这 9 个视图来自 cameras 2-4 与 body rotations 0°、50°、90° 的 3 × 3 组合；不能把全部 70 个采集视图误写成一次重建的输入数。

自由来流/归一化参数的完整标量读取为：

| 字段 | 本地数值审计读值 | 使用边界 |
|---|---:|---|
| `rho_inf` | 0.0584 | 作者重建输入参数，不单独宣称为新实验测量 |
| `D_inf` | 0.082 | 与 82 mm 参考长度一致 |
| `c_inf` | 152.06 | 作者代码中的参考声速 |
| `gamma_inf` | 1.4 | 无量纲参考参数 |
| `T_inf` | 57.5466490791 | 作者重建输入参数 |

这些值证明本地 MAT 与作者脚本约定一致，不证明算法重建正确。

## 3. 三维网格与单位

`X/Y/Z` 均为 `400 × 350 × 350` double。reader 使用作者 Python loader 的 C-order 访问语义选择 15 个角点、中心和逐轴地标，同时按 MATLAB F-order 映射回 payload。三个 392,000,000-byte 数值 payload 均完成全流校验。

| 轴 | 首/末 cell center | spacing | 由 cell centers 反推的总域 |
|---|---:|---:|---:|
| X | -0.0748125 / 0.0748125 m | 0.000375 m | 0.150 m |
| Y | -0.0648142857 / 0.0648142857 m | 0.0003714286 m | 0.130 m |
| Z | -0.0648142857 / 0.0648142857 m | 0.0003714286 m | 0.130 m |

地标检查同时满足：同一 X index 的 X 值不随 Y/Z 变化，Y、Z 同理；上下边界关于 0 对称。它与官方脚本的 150 × 130 × 130 mm、400 × 350 × 350 cell-centered 域一致，说明 MAT 坐标以米存储。

## 4. 光线与视图契约

对 `c`、`v`、`Ruvecs`、`Rvvecs`、`Rapvec` 统一抽取 19 个 measurement rows，并强制三个向量分量成组读取：

- 19 个 `v` 样本的范数范围为 0.9999999799 至 1.0000000232；最大绝对误差约 `2.32e-8`；
- `c` 样本按 1e-6 精度聚类后恰好得到 9 个 camera/view centers；
- 五个字段使用相同 measurement indices，所有样本有限；
- `Rapvec` 样本范围为 0.001640625 至 0.0019318182，均为正。

这里没有强行要求 `Ruvecs/Rvvecs` 与 `v` 构成正交单位基。它们在作者 cone/aperture 模型中是投影/采样方向，代码没有给出足以支持该额外假设的合同；错误加入“看起来合理”的正交约束反而会制造假失败。

## 5. 为什么还不能运行完整官方 NIRT

preflight 对 11 个官方 Python 文件做了 AST、依赖、路径、静态入口和内存下界检查。结果是：

- 11/11 文件语法可解析；仓库 Python 3.11.5 环境中的 NumPy 2.4.6、SciPy 1.17.1 可导入；当前环境没有 TensorFlow；
- 默认 `pred_flag=1` 是预测，不是训练，并会寻找当前不存在的默认 checkpoint；
- 默认数据路径与本地官方包布局不匹配；
- 入口强制 `tf.device('/GPU')`，并写死 Windows CUDA XLA 路径；
- 静态检查还发现 `tf.ones(3,1)`、缺少 `auto_w` 合同和 Tensor slice `.assign()` 等 6 个 blocker；
- 仅 `cam_data`、`b_data`、`X/Y/Z` 三组已知常驻数组的显式下界约为 9.25 GiB，占 32 GiB 主机内存 28.9%；这还没有计算源数组副本、临时几何量、TensorFlow tensors、模型和 XLA。

所以当前判决是：

```text
FULL_AUTHOR_NIRT_NO_GO_CURRENT_ENVIRONMENT
```

这不是说 9-view NIRT 永远不可复现，也不是下载或 VPN 问题。它只禁止在当前未修改源码和环境上直接执行 `python NIRT.py`。安全的下一门是：小型 synthetic geometry fixture、流式 loader adapter、CPU/MPS 网络构造测试，然后才是受控的真实 9-view 短程运行。

## 6. 可复核命令

下面的 `<LOCAL_MAT_PATH>` 与 `<PRIVATE_OUTPUT_DIR>` 只指向本机私有资料库，不应提交到 Git：

```bash
PYTHONPATH=. python3 -m site_tools.psu_bost_mat_sample \
  <LOCAL_MAT_PATH> \
  --variables siz rho_inf D_inf c_inf gamma_inf T_inf \
  --output <PRIVATE_OUTPUT_DIR>/mat_numeric_scalar_audit.json

PYTHONPATH=. python3 -m site_tools.psu_bost_mat_sample \
  <LOCAL_MAT_PATH> \
  --variables X Y Z --order author_c --strategy grid_landmarks \
  --full-threshold-bytes 0 \
  --output <PRIVATE_OUTPUT_DIR>/mat_numeric_xyz_audit.json

PYTHONPATH=. python3 -m pytest -q \
  site_tools/test_psu_bost_mat_sample.py \
  site_tools/test_psu_bost_loader_conformance.py \
  site_tools/test_official_nirt_preflight.py \
  site_tools/test_build_psu_public_summary.py
```

实现入口：

- [低内存 MAT reader](../site_tools/psu_bost_mat_sample.py)
- [loader 数值契约判定](../site_tools/psu_bost_loader_conformance.py)
- [官方 NIRT preflight](../site_tools/official_nirt_preflight.py)
- [公开机器可读汇总](psu_9view_numeric_loader_summary.json)

## 7. 下一步门槛

1. 用小型、可公开生成的 MAT fixture 复刻作者 `setup.schlieren()` 的字段和 shape，不读 5 GB 全量文件；
2. 对作者 `rayBoxIntersection` / `rayConeIntersection` 做 3-10 条解析几何 smoke，并核对进入/离开点；
3. 设计按 view/pixel 分块的 streaming loader，避免构造完整 `N × 18` double `cam_data`；
4. 冻结 1 个 training smoke 配置和 1 个 prediction smoke 配置，先在 CPU/MPS 上过网络/损失合同；
5. 只有 smoke、内存测量和路径修复全部通过，才启动 9-view 真实短程 NIRT；
6. 重建冻结后再打开 held-out deflection，报告逐 camera/rotation residual。没有 3D truth 时，不能把 reprojection 低误差写成场真值正确。

## 8. 仍然关闭的论文主张

- 没有 9-view 三维重建结果；
- 没有 held-out reprojection；
- 没有 NeRIF、NIRT、DeepONet、FNO 或自有算法横向胜出；
- 没有真实 flow-off covariance、真实有限孔径 mismatch 或 OERF 组内证据；
- 没有对作者源码做功能修复后的独立复现。

这次增量的价值是把“数据文件存在”推进为“核心 loader 数值契约可验证”，同时及时阻止了一次很可能以 OOM、缺 checkpoint 或 TensorFlow 源码错误结束的盲跑。
