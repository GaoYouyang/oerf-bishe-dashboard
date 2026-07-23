# PSU support rotation LORO：视角身份与双网格缓存 preflight

## 先说结论

这一步只闭合后续实验的“输入是谁”与“两个网格是不是在看同一批物理射线”。它**没有运行 LORO CGLS、没有产生候选分数，也没有接触 rotation 40 或最终审计 rotations**。

preflight 通过后，九个匿名 support view 的机械映射为：

| rotation | camera 2 | camera 3 | camera 4 |
|---:|---:|---:|---:|
| 0° | view 0 | view 1 | view 2 |
| 50° | view 3 | view 4 | view 5 |
| 90° | view 6 | view 7 | view 8 |

这个次序不是根据残差反推的。官方 `AEDC_pprocess_auto.m` 先按 `0, 50, 90` 外层循环拼接 rotation；每个 rotation 内，`AEDC_pprocess.m` 按 camera `2:4` 的内层循环拼接字段。九个 bundle manifest 又把 `HSOF_9CAM_RT.mat` 固定为九段连续、等长的 5,529,600-row block，因此本次 preflight 可跨文件闭合 view ID 与 rotation/camera 的对应关系。必须保留一个边界：compact-cache manifest 自己只有 view ID，没有内嵌 camera/rotation 字段；不能把“跨文件绑定”写成“cache 单文件自证”。

## 为什么要先做这一步

上一轮 E69 已经说明：在 rotation 40 上，`16³` 与 `32³` 的 frozen field、离散表示和 forward package 同时变化，不能把差异简单叫作“高频过拟合”。下一步要在 support 内做整组 leave-one-rotation-out：

1. 用两个 rotation 的六台相机重建；
2. 把第三个 rotation 的三台相机整体留出；
3. 对 `0°/50°/90°` 三折完整轮换；
4. 同时比较 `16³`、`32³`、迭代停止点和 coarse-subspace 约束。

若 view 身份没有机械绑定，所谓“整组留一 rotation”就只是标签故事，不能成为论文证据。

## 新增的私有计算基础设施

本机现在有两份同源 compact cache：

| grid | rays | aperture samples/ray | chunks | cache bytes |
|---:|---:|---:|---:|---:|
| 16³ | 10,628,822 | 16 | 329 | 5,016,803,984 |
| 32³ | 10,628,822 | 16 | 329 | 5,016,803,984 |

审计器以 `verify_hashes=True` 打开两份 cache，逐个验证 manifest 声明的数组 SHA-256。跨网格比较得到：

- observations、camera projection、ray scale、valid sample mask 的哈希完全相同；
- 只有与离散网格有关的 lower-corner indices 和 trilinear fractions 不同；
- 审计器对 `170,061,152` 个有效 aperture samples（`510,183,456` 个坐标分量）分 chunk 反算归一化三维坐标；16³/32³ 最大绝对差为 `1.1102230246251565e-16`、RMS 为 `2.2520622693914397e-17`，均通过冻结的 `1e-12` 门；
- 两份 cache 都只含 support rotations，rotation 40 与 final audit 均未进入。

cache 和 private report 保留在 `private_library`，不会进入 GitHub Pages。公开 JSON 只保留映射、计数、布尔门和 cache 字节数，不含测量值、重建 voxel、本机路径或私有文件哈希。

## 可重复运行

```bash
.venv/bin/python -m site_tools.audit_psu_support_rotation_loro_preflight \
  --script-root private_library/external_datasets/psu_bost_flight_body/scripts \
  --view-root private_library/external_datasets/psu_bost_flight_body/all_view_geometry_audit \
  --cache16-root private_library/external_datasets/psu_bost_flight_body/b0_compact_cache_16cubed_qmc16_v1 \
  --cache32-root private_library/external_datasets/psu_bost_flight_body/b0_compact_cache_32cubed_qmc16_v1 \
  --private-output private_library/external_datasets/psu_bost_flight_body/support_rotation_loro_preflight_v1/private_report.json \
  --public-output docs/psu_support_rotation_loro_preflight_public_summary.json
```

## 还没有证明什么

- 没有 field truth，所以不能报告 field relative-L2；
- support LORO 是同一次实验流场的未见相机布局/rotation 重投影，不等于跨流态泛化；
- cache 完整性不等于物理 forward model 正确；
- 全视角 geometry audit 对 detector 全域仍有既有 NO-GO；本次可用性只指 corrected-active-ray B0 pipeline；
- 后续若 `32³` 在 held-out rotation 上更差，仍要把 representation、forward discretization 与 solver trajectory 分开归因；
- 任何候选都必须在协议冻结后运行，不能用 rotation 40 或 final rotations 选择门槛。

机器可读公开摘要：[psu_support_rotation_loro_preflight_public_summary.json](psu_support_rotation_loro_preflight_public_summary.json)
