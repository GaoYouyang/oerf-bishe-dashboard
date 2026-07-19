# PSU rotation-40 分辨率迁移：独立科学、复现与隐私审计

> 总判定：**数值 NO-GO 可信，未发现实际篡改或私有数据泄露；证据链仍有明确工程缺口，不能称完整 fail-closed 或论文结论。**

## 1. 独立复算确认了什么

审计不使用 runner 已写出的比较字段，重新执行两个冻结场在全部 `3,847,050` 条 rotation-40 rays 上的 forward，并重新计算 pooled、逐相机和辅助指标：

- 16³ concatenated-all-ray vector relative-L2：`0.8432631430215097`；
- 32³ concatenated-all-ray vector relative-L2：`0.9595912440290707`；
- `16³ - 32³ = -0.11632810100756097`；
- camera 2/3/4 改善量：`-0.061519002957733604 / -0.11000484612567829 / -0.14568911539028828`；
- 独立复算与 `summary.json` 全指标最大绝对差：`1.01e-14`；
- 冻结判决要求 `>= +0.01` 且三相机不伤害，因此 NO-GO 不是边缘判断。

JSON 与 CSV 的 8 行逐字段一致；PNG 的柱高/幅值点与 JSON 像素拟合误差不超过 `0.30 / 0.44 px`；PDF 结论一致。五个公开资产的 checksum 全通过。公开包未发现绝对本地路径、私有文件名、账号信息、measurement/prediction/ray/volume 数组。

结果 commit 前只做过一次非科学 packaging 规范化：把 CSV 的 CRLF 行尾改为 LF，数值与字段未变，随后更新 checksum，并重新运行独立 public-package validator。该规范化不改变任何指标、图或判决。

## 2. 两个必须保留的 P1 边界

### P1-A：预结果 attestation 没覆盖完整传递依赖闭包

冻结清单包含顶层 runner、active store、forward operator 与 metric，但以下实际参与计算的本地模块没有进入 attestation：

| 漏绑定依赖 | 本次文件 SHA-256 |
|---|---|
| `demo_t16_operator/psu_b0_reconstruction_interface.py` | `0aa1695dacf16f95ea32aeb8d0b019451721a48e65238101549447f68e467ba8` |
| `site_tools/psu_bost_aperture_domain.py` | `9b1481fa8b6ccc88f78cecbb6ebc3993e524fbe586162671209035035ccd280d` |
| `site_tools/psu_bost_forward_geometry.py` | `88e7c02cb6186d9b71d011725fa9ea7fc935440db47499482ff97a4cb1d841d8` |
| `site_tools/psu_b0_real_support_store.py` | `752da89461ab6270835762a76540b44122c475644fea2bdfe290a111ad3a7c42` |

审计确认这四个文件在运行后仍与 protocol commit `ba77a17f...` 完全一致，所以没有发现本次计算被改写；但机制上，attestation 不能阻止“运行前临时改传递依赖、运行后恢复”。下一版协议必须递归解析本地 import 闭包，并绑定 requirements/lockfile、Python 与关键数值库版本。

### P1-B：正式结果必须由第三个 result commit 固定

审计发生时 HEAD 仍是 attestation commit `7cc9c3c`，结果包只在 staging area。目录内 checksum 可以与文件一起被整体替换，不能代替 Git 结果锚。处置要求：summary、CSV、PNG、PDF、README、checksums、结果说明、独立审计与网页必须进入一个单独 result commit 并推送；最终以该 commit 的 Git 历史为准。

## 3. 不改变判决、但必须公开的 P2 勘误

### P2-A：pooled weighting 标签不精确

冻结 metadata 写 `RAY_COUNT_WEIGHTED_OVER_ALL_SELECTED_ROWS`。实际公式是全部 rows 拼接后的 global norm ratio：

```text
sqrt( sum_c ||prediction_c - measurement_c||² / sum_c ||measurement_c||² )
```

它不是逐相机 relative-L2 的 ray-count-weighted arithmetic mean。真正的 ray-count mean 在 16³/32³ 上约为 `0.834763 / 0.944219`，而正式 global norm ratio 是 `0.843263 / 0.959591`。三台相机、equal-camera macro 和两种 pooled 汇总都同向，因此 NO-GO 不变。下一版字段应改名 `GLOBAL_VECTOR_L2_RATIO_OVER_CONCATENATED_ROWS`；本轮不静默改写冻结配置与机器包，只通过 post-open erratum 更正解释。

### P2-B：这是单侧预注册，不是完全盲试验

协议冻结前，32³ 的 rotation-40 分数 `0.959591` 已由旧基线打开；16³ 正式分数尚未知。因此准确名称是“在 16³ scoring 前预注册的 development comparison”。它仍防止看到 16³ 后改阈值、相机或 metric，但不能冒充双方都未见的 confirmatory test。

### P2-C：网格与固定四步 CGLS 求解轨迹混杂

32³ 自由内点约是 16³ 的 9.84 倍；相同四步 CGLS 不代表相同谱滤波、正则强度或收敛阶段。本结果严格否证的是冻结 `32³+CGLS4` package 的 rotation-transfer，不单独证明“细网格过拟合”或“高频 null-space correction”是原因。下一步先比较 `U x16`、原生 `x32`、1/2/3/4 步轨迹、H1/TV 和 coarse-to-fine continuation。

### P2-D：support view identity 未端到端机械绑定

配置冻结 camera 2/3/4 × rotation 0/50/90，公开 PSU 论文与既有 held-out protocol 也支持这个映射；候选 volume、报告和九个匿名 view ID 已哈希绑定。但 runner 没有把记录匿名 ID 到 camera/rotation 映射的 source script/held-out protocol 纳入 attestation。same-camera rotation holdout 的语义可信，机械闭包不完整。

### P2-E：公开隐私检查应从黑名单升级到结构白名单

当前公开包最长列表只有 3 项，审计确认没有实际数组泄露；但 runner 的递归检查可以被任意名称的大型 Python numeric list 绕过。结果打开后已补独立公开包 validator：限制顶层 schema、对象规模、递归深度、任意列表长度和数值标量预算，并交叉核 JSON/CSV/PNG/PDF/checksum；这能保护当前公开包，但不能倒推修复预结果 runner。下一协议仍应冻结正式 JSON Schema 白名单。

## 4. P3 成本与统计解释

- `N=1` 个独立 rotation block，三台相机和 384 万 rays 都不是独立物理重复。
- rotation 40 同时可能混合新观察几何、独立运行条件、标定、模型变形和 data-driven mask，不是“只改变一个纯角度变量”的理想干预。
- peak RSS 是整个进程的累计 high-water mark；第二候选的数值不是独立候选峰值，不应用于严肃内存优越性比较。
- 两个 protocol/attestation commit 未签名，`formal_results_absent_at_creation` 是本机 Git 自证，不能理论排除更早试跑后删除；当前只确认仓库记录中未见结果后改规则。

## 5. 独立审计后的准确表述

可以说：

> 在 16³ 正式评分前冻结的 PSU rotation-40 development 比较中，冻结的 32³+CGLS4 package 虽在九个 support views 上有更低 measurement residual，却在同一 camera 2/3/4 的一个 rotation-40 运行上得到更高 concatenated-all-ray、equal-camera macro 与三项 camera-wise relative-L2，因此按预注册数值线判 NO-GO。

不能说：

- 细网格本身已被证明过拟合；
- 16³ 有更低真实三维 field error；
- 三台相机或每条 ray 是统计重复；
- 当前结果已达到完整 fail-closed、跨 rotation 泛化或论文优越性证据。

## 6. 下一版必须补齐

1. 递归本地 import 依赖闭包、requirements/lockfile 与环境 fingerprint 的预结果 attestation。
2. held-out protocol、source script 与匿名 support view identity 的机械绑定。
3. `GLOBAL_VECTOR_L2_RATIO_OVER_CONCATENATED_ROWS` 正确命名，加 equal-camera macro、cosine、尺度 oracle 仅诊断、空间 residual spectrum 与 front/shock ROI。
4. 公开 report JSON Schema 白名单、列表长度上限、独立 package validator 与正式端到端 smoke。
5. support 内 leave-one-rotation-out 选择停止/正则；最终方法冻结后一次性打开 sealed rotation blocks，不在 final 上继续选模。

运行环境见 [环境指纹 JSON](psu_rotation40_resolution_transfer_environment_2026-07-19.json)，从独立 clone 重放见 [完整 replay 命令](psu_rotation40_resolution_transfer_replay_2026-07-19.md)。
