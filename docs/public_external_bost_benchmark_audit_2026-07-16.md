# 公开 BOST 外部基准审计：V6B 现在能做什么

审计日期：2026-07-16

判定规则：只把官方论文、作者仓库、数据 DOI 与仓库许可当作可执行依据。论文开放阅读不等于数据/代码可以再分发。

## 1. 先给结论

目前没有一个公开资源同时提供：独立多光圈、独立相机布局、二者联合 OOD、受限的高保真 `F x` 查询、可核验的 `F^T z`/VJP，以及实验三维真值。因此完整 V6B 仍是 **`UNCONSTRUCTED`**。

但已经形成两条不浪费时间的外部证据路线：

1. **PSU flight-body BOS 数据**：做固定真实装置的 9 视角重建、留出视角重投影、真实 mask/噪声/标定 loader 和 NIRT 强基线。
2. **`photon` renderer**：在 NVIDIA/CUDA 环境生成新光圈、新相机角度和已知密度场，做跨 renderer 的 V6B synthetic smoke test。

两者合起来仍不等于 OERF 实验结论。PSU 缺受控光圈 sweep；`photon` 是合成数据且没有现成伴随。

## 2. 资源分级

| 资源 | 现在可取得的内容 | 适合的任务 | 不能支持的结论 | 判定 |
|---|---|---|---|---|
| [PSU 70-view flight-body BOS](https://doi.org/10.26208/1VE2-5C19) | 7 相机 × 10 模型转角、标定、flow-off/on、mask、三类 deflection、NIRT 与结果 | 固定 rig 的真实 reconstruction/reprojection；外部 loader | 跨光圈 operator learning；实验 field-L2 | 本地可用 |
| [`photon`](https://github.com/lalitkrajendran/photon) | GPL-3.0 源码；f-number、焦距、物距、相机角度、NRRD 密度场与 ray 输出可配置 | fresh-A/B/C 合成 renderer；外部 forward smoke test | 真实光学；Mac 原生训练；精确 `F^T` | CUDA 服务器可用 |
| [finite-aperture cone-ray BOS](https://arxiv.org/abs/2402.15954) | 论文包含 f/22 至 f/4 的真实实验和 cone-ray 证据 | 最直接的多光圈物理依据 | 数据/代码未公开，不能跑 benchmark | 联系作者 |
| [NeRIF](https://arxiv.org/abs/2409.14722) | 论文、9 视角合成/实验设计 | OERF 最近邻重建协议与 held-out view | Data Availability 为合理请求；无公开训练包 | 联系何远哲/蔡老师 |
| [NeDF](https://arxiv.org/abs/2409.19971) | 论文中的 5-180 视角和 limited-angle 设计 | 视角数量/布局对照设计 | 数据、代码、LES GT 未公开 | 联系作者 |
| [TDBOST](https://github.com/Hyz617/TDBOST) | 代码树与部分公开 Drive 文件 | 读数据格式、4D 结构和缺口 | 无根许可证；投影器/完整数据不齐；不能合法重发 | 联系何远哲 |
| [Neural Implicit Reconstruction for BOS](https://github.com/Weihu22/Neural-Implicit-Reconstruction-for-BOS) | Phantom 1 合成数据与 NIR-BOS 代码可查看 | 方法复现、hash/implicit baseline | 仓库无许可证；不能默认改作或再分发 | 只读审计 |
| [HBOS reactive-flow data](https://doi.org/10.3217/nzz9b-rn487) | 约 725 MB 的 flame time-signal、shadowgraphy 与 ray-tracing files | 2D 反应流时间/事件预训练 | 不是 3D tomography/operator corpus | TRAIL 热身 |

## 3. PSU 数据的可执行事实

### 3.1 体量与内容

- 12 个 ZIP 的 HTTP `Content-Length` 合计 **51,664,057,334 bytes = 51.664 GB = 48.116 GiB**。
- 论文和 readme 明确为 70 views；公开流程把相机 2-4 在模型旋转 0°、50°、90° 组成 9-view reconstruction，其余观测可用于验证。
- 第 6 包只含预处理后的 `data/HSOF_9CAM_RT.mat`：ZIP 为 **5,117,966,684 bytes**，文件解压后约 **5.23 GB**。这是复跑官方 NIRT 的最小重数据入口。
- 第 10 包含完整 NIRT Python 源码、说明 PDF、重建 checkpoint/曲线和图；第 12 包包含 MATLAB 预处理/可视化脚本与工具。
- NIRT 代码读取 `c, v, epsu_all, epsv_all, masks, Ru/Rv, Rx/Ry, Rap, Df`，并可在 pinhole 与 volumetric/cone sampling 间切换。

### 3.2 光学字段的边界

PSU 数据确实包含 cone/aperture 所需的方向、入口/出口半径与距离字段，但不是受控的“同一相机只改变 f-number”实验。论文中的 f/32 与 f/22 对应不同相机/镜头，不应伪装成 aperture sweep。它适合检验：

1. loader、坐标系、单位和 mask 是否正确；
2. 9-view NIRT 是否能重现 held-out deflection；
3. nominal thin-ray、公开 cone-ray 与我们的 corrected operator 在同一固定 rig 上的 measurement consistency；
4. 真实 deflection sensing 算法 CC / WOF / HS-OF 的敏感性。

它不能单独检验跨 f-number 的 RayKernel-DCO 泛化。

### 3.3 许可与本地边界

元数据写明 `Access_Constraints: None`，但 `Use_Constraints` 不是标准 OSI/Creative Commons 许可；它允许在本机转换和增值，并要求每年通知 University 自定义/增值工作。DataCite `rightsList` 为空。因此当前策略是：

- 数据、源码副本和派生产物只放 `private_library/`；
- GitHub Pages 只放来源链接、我们自己写的审计、接口和指标；
- 不重新分发原始 ZIP、作者源码、checkpoint 或派生数据；
- 若论文工作实际使用并改造该包，投稿前向 Penn State 数据联系人确认通知方式。

## 4. 本地已完成的下载、结构与数值 loader 审计

忽略 Git 的本机私有存储现已包含：

- 官方 7 页 readme PDF、12 个 NIRT Python 文件；
- 顶层 MATLAB 预处理/可视化脚本、关键工具、重建 Summary SVG 与 deflection cuts；
- part 06 的完整 9-view ZIP 与单独解压的 `HSOF_9CAM_RT.mat`；
- 每个最小文件、ZIP 和 MAT 的 SHA-256 记录。

完整 ZIP 为 `5,117,966,684` bytes，SHA-256 为 `4a4b2cbfebfb4e9f413a339502c8988f18140148daf3f1cf6f386621d30d6497`；解压 MAT 为 `5,227,925,615` bytes，SHA-256 为 `622852cb06faa90271479b78ffe98e6d02bc50217ebde067226f727569f8c788`。ZIP 已通过 `unzip -t` CRC 检查。

MAT 是带 MCOS subsystem 的压缩 MATLAB v5 文件，SciPy `whosmat` 在该文件上异常退出。因此新增的流式 v5 scanner 只解压每个变量的 header prefix，并在 subsystem offset 前停止，不读取 3D 数值 payload。审计结果为 `SCHEMA_CONFORMANT`：

- 97 个命名变量，26 个 loader 必需字段均存在；
- `X/Y/Z` 均为 `400 x 350 x 350` double；
- `c/v` 与 `Ru/Rv/Rx/Ry`、`Csys_all`、`epsu/epsv`、`Rap/Df` 的 measurement width 均为 `49,766,400`；
- `siz` 含 3 个元素；变量名无重复；
- `Itn_all/Iu_all/Iv_all` 在该预处理包中是 `0 x 0`，仅确认与公开文件一致，不能把它们当作已加载图像。

header 判定只证明 archive、变量名、shape 与作者 loader 约定对得上。随后新增按变量选择的流式数值 reader，对被选变量完整验证 zlib stream 与数值 payload SHA-256，同时只保留小变量全值、网格几何地标或成组 measurement rows。真实数值合同得到：

- `siz=[2160,2560,9]`，乘积为 `49,766,400`，与 ray 字段宽度一致；
- `X/Y/Z` 是中心对称 cell-centered 网格，反推域为 0.150 × 0.130 × 0.130 m；
- 19 个 `v` 样本的单位范数最大误差约 `2.32e-8`；
- `c` 样本解析出 9 个不同 camera/view centers；
- 13 项机器可读 loader 数值契约全部通过，状态为 `LOADER_NUMERIC_CONTRACT_CONFORMANT`。

这仍没有验证 deflection 全分布、mask 数值语义、held-out reprojection、NIRT 重建或伴随。官方 NIRT preflight 进一步发现当前环境缺 TensorFlow、默认 checkpoint/运行路径不存在、6 个静态 blocker；已知常驻数组内存下界约 9.25 GiB。因此未修改官方入口的当前判决是 `FULL_AUTHOR_NIRT_NO_GO_CURRENT_ENVIRONMENT`，安全下一步是 tiny fixture 与 streaming loader，而不是直接启动 5 GB 全量训练。

当前私有论文/资料库共 **27 个 PDF、约 185 MB PDF 体积**；新增的是数据说明，不是订阅论文。GitHub 中仍为 0 个 VPN/订阅 PDF。

复核小清单：

```bash
.venv/bin/python site_tools/psu_bost_selective_fetch.py --refresh-index
.venv/bin/python site_tools/psu_bost_selective_fetch.py --fetch-minimal
PYTHONPATH=. .venv/bin/python -m pytest -q site_tools/test_psu_bost_selective_fetch.py
```

下载 9-view 核心包时使用可续传脚本；若完整归档已存在，脚本只做 size/SHA-256/CRC 复核，不再次联网：

```bash
zsh site_tools/download_psu_bost_core.sh

# 对本地 MAT 做 header/shape/hash 审计，不运行 NIRT
.venv/bin/python site_tools/psu_bost_mat_audit.py <LOCAL_MAT_PATH> \
  --output <LOCAL_AUDIT_JSON> --sha256

# 数值抽样只写入被 Git 忽略的私有输出目录
PYTHONPATH=. python3 -m site_tools.psu_bost_mat_sample <LOCAL_MAT_PATH> \
  --variables X Y Z --order author_c --strategy grid_landmarks \
  --full-threshold-bytes 0 --output <PRIVATE_OUTPUT_DIR>/mat_numeric_xyz_audit.json
```

详细结果与边界见 [PSU 9-view 数值 loader 门禁](psu_9view_numeric_loader_gate_2026-07-16.md)；公开页面使用的聚合数字另有 [机器可读汇总](psu_9view_numeric_loader_summary.json)，不包含原始数组、标量样本、作者源码副本或私有路径。

## 5. 外部 benchmark 的四级门

### E0 - Loader conformance

- MAT 字段、shape、dtype、单位、坐标手性、相机/旋转 id 和 mask 全部生成机器可读 manifest；
- 随机抽 20 条 ray 与 MATLAB/作者图进行投影方向检查；
- flow-off 区域平均 deflection 应接近作者 readme 报告的 `<0.01 px` 数量级。

### E1 - 作者基线复现

- 只用公开 9 views 训练/重建；
- held-out views 只在模型冻结后打开；
- 报告每 camera/rotation 的 deflection relative-L2、角度误差、shock-edge 距离与 runtime；
- 不能用作者重建结果作为训练标签再称“外部复现”。

### E2 - 固定 rig 算子比较

- 同一 NIRT field/phantom 下比较 thin、公开 cone 与 corrected forward；
- forward/adjoint、query、MC samples 和 wall-clock 分账；
- 没有真实 `F^T` 时，只能写 forward/reprojection benchmark，不能报告 gradient fidelity。

### E3 - 跨 renderer synthetic smoke

- 用 `photon` 冻结 train/development/fresh-A/B/C manifest；
- 主变量：f-number、focus/object distance、camera angle/layout；
- 密度场必须包含平滑 plume、thin front、shock-like discontinuity 和独立反应流切片；
- 通过 E3 只说明跨 renderer synthetic 可迁移，不能升级成真实 OERF 证据。

## 6. 现在需要向何远哲索取的最小东西

1. 一套 OERF flow-off/reference repeats 与相机标定，不必先给完整实验；
2. 每台相机的 f-number、焦距、物距、焦平面、传感器像元和裁剪；
3. NeRIF/TDBOST 中真正调用的 forward、VJP/adjoint 和 ray 数据结构；
4. 最好有两档光圈或移动焦平面；若没有，给 calibration phantom 或 paired thin/cone simulation；
5. 数据若不可外传，只需由师兄在组内运行我们的 loader/探针脚本并返回无敏感内容的指标表。

## 7. 对论文路线的影响

当前更稳妥的论文叙事不是“我们已经比 DeepONet/FNO 好”，而是：

> 先用公开真实 flight-body BOS 建立固定 rig 的外部 reconstruction/reprojection 基线，再用独立 `photon` renderer 检验少查询 ray-local correction 的跨光圈/布局机制，最后只在 OERF 标定证据到位后打开 apparatus-specific claim。

这条证据链把真实数据、可控真值和组内应用分开，允许负结果，也避免把一个内部合成 generator 上的 8.08% 近信号包装成论文胜出。
