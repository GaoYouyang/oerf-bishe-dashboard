# 算子学习 + 三维重建：公开基准迁移梯

日期：2026-07-17
状态：`PLAN_ONLY / NO_LEVEL_EXECUTED / NO_FRESH_SESSION_CONSTRUCTED`

## 0. 结论与证据纪律

本路线不是一张“数据集排行榜”，而是一条逐级增加物理接近度、同时逐级减少可用真值的迁移梯：

`Walnut CT -> LoDoPaB-CT（可选） -> helium BOS -> TRON-BOS -> photon -> PSU frozen held-out -> OERF`

必须始终分开三类证据：

1. **CT 数值 oracle**：可验证线性逆问题、稀疏角度退化、数据一致性、伴随和算子学习机制；**不是 BOST 证据**。
2. **公开真实 BOS 测量**：若完成相应门禁，可检查真实背景纹理、位移提取、mask、观测到的时间变异和测量域重投影；单 tare 数据不能分离流动变化与测量噪声，没有独立三维真值时也不能报告实验 field-L2。
3. **独立 renderer 与 OERF**：`photon` 可提供受控合成真值和跨 renderer 检验；只有独立 OERF session 才能支持装置特定结论。

本文不预设任何现有算法胜出。每一级只决定“是否值得进入下一级”；负结果、无法复现和许可不清都是有效结论。

## 1. 总表

| 级别 | 资源与一手来源 | 许可 | 公开体量与本机可行性 | 若按门禁完成，最多可支持 | 不能证明 | 进入下一级的最低条件 |
|---|---|---|---|---|---|---|
| L0 | [FIPS Walnut CT 数据页](https://fips.fi/open-datasets/x-ray-tomographic-datasets/tomographic-x-ray-data-of-a-walnut/)、[Zenodo DOI](https://doi.org/10.5281/zenodo.1254206)、[数据论文](https://arxiv.org/abs/1502.04064) | 数据页明确 `CC BY 4.0` | 单个 2D slice，含三档分辨率 sinogram/显式测量矩阵、120-view 原始测量及 1200-view FBP reference reconstruction；应先查 Zenodo file manifest 再选最小矩阵。适合本机 CPU | 显式 `A`/`A^T`、dot test、稀疏角度、正则化路径、operator surrogate 的 oracle 误差 | BOS 光学、折射率到位移、相机标定、独立物理真值、3D BOST | 数值合同全过，且盲测不劣于预注册基线；否则停在 L0 修算子/协议 |
| L1 | [LoDoPaB-CT Zenodo](https://doi.org/10.5281/zenodo.3384092)、[Scientific Data 论文](https://www.nature.com/articles/s41597-021-00893-z)、[官方生成代码](https://github.com/jleuschn/lodopab_tech_ref)、[DIVal](https://github.com/jleuschn/dival) | Zenodo 当前记录为 `CC BY 4.0`；论文早期文本曾写 ODC-By，执行时以下载版本元数据并保留快照 | Zenodo 当前总计约 **55.0 GB**；35,820/3,522/3,553 train/validation/test slices，362 x 362 reference reconstruction，模拟低剂量投影。全量不适合作为首轮本机下载 | 大样本、噪声、分布切分、学习型逆问题的训练稳定性和 calibration | BOST；真实 BOS 测量；独立物理真值；有限孔径光学 | 仅在 L0 机制通过且有明确大规模问题时进入；先 metadata/代码，未经预算批准不拉全量 |
| L2a | [helium BOS Zenodo](https://doi.org/10.5281/zenodo.6136052)、[论文](https://arxiv.org/abs/2202.04122) | `CC BY 4.0` | 全部约 **3.8 GB**；两个 BOS ZIP 分别约 **906.1 MB**、**854.6 MB**，每组是单视角时间序列，且每组只有一个 tare（首帧）。单个 BOS 包本机可行，但不是三维数据 | 真实 BOS 背景、observed temporal variability、单 tare 下的位移/伪 schlieren 稳健性；不能从该数据单独估计 flow-off noise | 3D tomography、独立密度真值、跨视角重建 | 先用预览/readme；只有需要真实图像前端压力测试时才下载一个 BOS ZIP |
| L2b | [TRON-BOS 官方实验室页](https://ecs.syr.edu/faculty/zigunov/tomobos.html)、[样例数据 DOI](https://doi.org/10.5281/zenodo.18118110)、[独立 GitHub 仓库](https://github.com/3dfernando/tron-bos) | Zenodo record 及其列出的文件标记 `CC BY 4.0`，执行时保留 metadata snapshot 并逐文件复核；独立 GitHub 仓库根许可证 `unknown`，仅查 metadata，不 clone/use/redistribute | 样例约 **16.314777708 GB**；官方页称单相机旋转喷嘴可形成数千视角。作者 MAT/VTK 是作者处理/重建产物，不是独立 truth | 真实 BOS/FCD 图像域、旋转角序列、位移到投影/密度重建链的工程适配 | 独立 3D truth；多相机固定 rig；OERF 泛化；算法优胜 | 能冻结角度、tare/reference、mask、标定与单位 manifest，并完成留角重投影；若流动漂移门失败则停止 tomography |
| L3 | [`lalitkrajendran/photon`](https://github.com/lalitkrajendran/photon)、[方法论文](https://arxiv.org/abs/1812.05902)、[出版 DOI](https://doi.org/10.1088/1361-6501/ab1ca8) | 代码 `GPL-3.0`；第三方依赖及生成数据另审计 | 仓库含 sample data；需要编译 CUDA ray-tracing shared library，当前 Apple Silicon Mac 不作为原生运行环境，需 NVIDIA/CUDA 服务器 | 独立合成 renderer；truth 来自运行前冻结的用户输入密度/折射率场；可控 f-number/焦距/物距/相机姿态 | 真实光学；现成精确 `F^T`/VJP；PSU/OERF 实验结论 | CUDA 环境可复现且输入场、单位、Gladstone-Dale 关系和边界条件可核验；否则不报 field-L2 |
| L4 | [PSU 70-view flight-body BOS DOI](https://doi.org/10.26208/1VE2-5C19)、[作者论文](https://arxiv.org/abs/2508.17120) | **record-level unknown pending rights confirmation**；记录页页脚虽指向 `CC BY 4.0`，仍不足以确认该 record/files 的授权。确认前不公开再分发原包、源码副本、checkpoint 或派生数据 | 12 包合计约 **51.664 GB**；7 相机与 10 模型转角的组合关系需按 manifest 逐项核验，不把“70 views”简化为已确认的独立视角。9-view MAT **5.23 GB**、常驻数组下界 **9.25 GiB** 均为 local audit measurement；本机只宜 selective/streaming audit | 最接近 BOST 的真实固定 rig：loader、标定、mask、9-view reconstruction、冻结后的 held-out-view reprojection | 独立实验 3D truth；作者重建也不得作为 truth；同一相机受控 aperture sweep；field-L2；OERF 装置有效 | E0 loader 后才开 E1；打开 PSU held-out 前必须冻结 `photon` 模型族、阈值、报告模板与 hash；内存/依赖不满足就停 |
| L5 | 组内冻结的 OERF 数据/标定/算子合同；公开邻近协议仅以 [NeRIF 论文](https://arxiv.org/abs/2409.14722) 和 [TDBOST 作者代码](https://github.com/Hyz617/TDBOST) 为入口 | 未公开内容默认权利未确认、限制在获批环境内；由数据所有者书面确认用途、共享和派生物边界 | 未形成可公开下载的冻结 corpus；大小与本机可行性在 manifest 到位前均为 unknown | 仅在独立 session 下检验 apparatus-specific reconstruction/reprojection、跨日期/装置漂移和最终研究问题 | 未设盲测、无真值或无独立验证时，仍不能宣称场重建准确或方法胜出 | 数据合同、盲测、防泄漏、基线与停止规则全部预注册后才运行 |

## 2. 每一级的任务、指标与停止条件

### L0：Walnut CT，小矩阵/稀疏角度 oracle

**最小任务**

- 只取最低分辨率显式矩阵、对应 sinogram、1200-view FBP reference reconstruction 或能支持同一分辨率比较的最小文件；该 FBP 不是独立物理真值。
- 固定 120/60/30/20 views 子采样规则；角度索引在运行前写入 manifest。
- 比较 FBP/LSQR 或 Tikhonov 与待测算子方法；同一数据、同一角度、同一停止预算。

**迁移指标**

- `dot_rel_error = |<Ax,y>-<x,A^Ty>| / max(|<Ax,y>|,|<x,A^Ty>|,eps)`。
- measurement relative-L2、reference relative-L2/PSNR/SSIM、迭代数、wall time、峰值内存。
- 20-view 相对 120-view 的退化曲线；正则参数只用 development angles 选择。
- 若学习 `A` 或修正项：另报 unseen-vector action error 和 residual whiteness，不得只报 reconstruction 图像指标。

**停止条件**

- shape、vectorization order、角度索引或 `A^T` dot test 任一失败，立即停。
- 待测方法若只在调参集改善、盲测不改善，记录负结果，不进入 LoDoPaB。
- 即使全部通过，结论也只能是“CT 逆问题机制通过”。

### L1：LoDoPaB-CT，大规模低剂量逆问题，仅机制

**最小任务**

- 首轮只获取 Zenodo metadata、官方技术参考代码和 patient split CSV；不下载 55 GB 全量。
- 若确需训练，先估算单个官方 archive 的磁盘、解压和训练 I/O；最小成对 observation/reference-reconstruction 分片必须保持 patient-level split。LoDoPaB 的所谓 truth 在本文中统一称 reference reconstruction，不视作独立物理真值。
- 保持官方 35,820/3,522/3,553 划分，禁止按 slice 重新随机拆分导致患者泄漏。

**迁移指标**

- patient-level test PSNR/SSIM、measurement consistency、低剂量噪声分层误差、校准误差。
- 训练/验证曲线、跨随机种子均值和离散度、吞吐、峰值 RAM/VRAM、磁盘占用。
- 与解析/迭代基线使用同一 forward operator；报告性能-算力 Pareto，不只报单点最好值。

**停止条件**

- 没有 L0 通过的机制假设，或问题不需要大样本统计，跳过 LoDoPaB。
- 本机剩余磁盘不足“下载 + 解压 + checkpoint + 20% 余量”，不下载。
- LoDoPaB 结果不得迁移表述为 BOS/BOST 成功。

### L2：helium BOS / TRON-BOS，真实 BOS 测量域

两者作用不同：TRON-BOS 更接近旋转 tomography；helium BOS 是可许可、可控体量的真实图像前端压力测试。它们都没有与每帧/每角度独立配准的三维真值。

**最小任务**

- helium BOS：只在需要时下载 `20210419-Run1.zip` 或 `20210420-Run1.zip` 之一；每包是单视角时间序列，且只有一个 tare，即按官方说明使用的首帧；后续帧不得混入 reference 估计。
- TRON-BOS：样例固定为 DOI `10.5281/zenodo.18118110`（Zenodo record 及其列出的文件标记 `CC BY 4.0`，约 `16.314777708 GB`，需保存 metadata snapshot 并逐文件复核）；独立 GitHub 仓库许可单独记为 `unknown`，仅查 metadata。核验角度表、reference/tare、像素尺寸和光学几何；作者 MAT/VTK 仅作格式/复现参考，不作独立 truth。
- 统一输出 measurement manifest：source hash、frame/view id、reference id、mask、坐标方向、单位、裁剪和预处理版本。

**迁移指标**

- 图像配准残差、有效像素率、outlier rate 与 observed temporal variability。只有另有独立 repeated flow-off frames 时才报告 flow-off displacement 均值/方差；helium 单 tare 数据中该项标为 `UNAVAILABLE`。
- 位移场的空间频谱、时间稳定性、正反向一致性和 reference 敏感性。
- TRON-BOS 若具角度序列：冻结后留角 reprojection relative-L2、梯度方向误差、边缘位置误差。
- TRON-BOS 必须预注册流动漂移停止门：用重复角度、tare/reference 稳定性或可审计的时间代理量检验采集期间漂移；超过冻结阈值即停止 tomography，只保留逐帧/图像域审计。

**停止条件**

- 角度、tare、标定或单位不能冻结，停止 tomography，只保留图像域审计。
- TRON-BOS Zenodo record 及其列出的文件按记录所示 `CC BY 4.0` 条款处理并保留 metadata/file 快照；独立 GitHub 仓库在根许可证确认前保持 `unknown`，仅做 metadata 检查，不 clone/use/redistribute。
- 没有独立 3D truth 时，禁止 field-L2、3D PSNR/SSIM 和“真实密度恢复准确”表述。

### L3：`photon`，独立合成 renderer

**最小任务**

- 固定一个由用户明确提供的 density/refractive-index field corpus，至少含 smooth plume、thin front、shock-like discontinuity；train/development/fresh-A/B/C 生成前冻结输入文件、单位、边界条件与 hash。
- 扫描 f-number、焦距/物距、camera angle/layout；renderer 配置、Gladstone-Dale 关系与随机种子进入 manifest。
- 将 `photon` 只作为独立 forward oracle；其 truth 仅来自上述已冻结用户输入场。若单位、density-to-refractive-index/Gladstone-Dale 关系或边界条件任一无法核验，不报告 field-L2。

**迁移指标**

- 对可核验输入场报告 field relative-L2、forward relative-L2、unseen-ray action error。
- aperture/layout/joint-OOD 分层误差，结构边缘距离，查询数、CUDA wall time、显存。
- 与本仓库 generator 的 cross-renderer gap；任何方法都按相同 query/训练预算比较。

**停止条件**

- CUDA build、依赖许可或配置复现失败，停止并保留构建日志，不用本仓库 renderer 替代后仍称“独立”。
- 输入场来源、单位、Gladstone-Dale 关系或边界条件不能核验时，仍可报告 forward/图像域诊断，但禁止 field-L2。
- `photon` 通过只能写“跨 renderer synthetic 迁移通过”。

### L4：PSU 9/70-view，冻结后的 held-out 固定真实 rig

**最小任务**

- 先复用 DOI 元数据、readme、代码/文件 manifest 和小文件；不重复拉取 51.664 GB 全集。许可保持 `record-level unknown pending rights confirmation`，确认前不公开再分发任何原包或派生数据。
- E0：字段、shape、dtype、单位、坐标手性、相机/转角、mask 与 hash；谨慎核对 7 相机、10 模型转角与“70 views”的实际组合，不预设为 70 个独立同时视角。
- E1：只用官方 9 views 完成作者基线或接口兼容性复现；作者提供或作者流程生成的重建不得作 truth。
- E2：打开其余 held-out 前，先冻结 `photon` 阶段选定的模型族、阈值、超参数、报告模板与 hash；随后只做一次 held-out forward/reprojection 主分析。

**迁移指标**

- 每 camera/rotation 的 held-out deflection relative-L2、角度误差、shock-edge distance、mask 分层误差。
- train-view 与 held-out-view gap、runtime、峰值内存、forward query 数、失败视角清单。
- 若比较 thin/cone/corrected operator：固定 field、rays、mask 与查询预算，逐模型分账。

**停止条件**

- loader 数值合同未过，不运行重建。
- 当前环境缺依赖、checkpoint/路径或预计内存越界，判 `NO-GO_CURRENT_ENVIRONMENT`，改做 streaming/tiny fixture。
- `photon` 模型族/阈值未冻结，或 held-out 视角已被用于调参，则该轮作废并重新冻结协议。
- PSU 作者重建不得作为 truth；PSU 通过最多证明“真实固定 rig 的 measurement consistency”，不能替代独立 3D truth 或 aperture 泛化。

### L5：OERF，最后一关

**运行前数据合同**

- 独立 flow-off/reference repeats、完整相机内外参、像元/裁剪、焦距、f-number、物距/焦平面、同步与缺帧标记。
- train/development/fresh session 按采集日期或实验 run 冻结；fresh 在模型、阈值和报告模板 hash 后才解封。
- 明确是否有 calibration phantom、数值仿真 truth、独立传感器或仅有 held-out camera；不同 truth 等级分账。
- 数据所有者书面确认本机使用、组内共享、公开指标、派生物和论文发布许可。

**迁移指标**

- 无独立 3D truth：只报 held-out-camera reprojection、flow-off residual、重复实验稳定性、拒答率和 failure slices。
- 有独立 truth：才增加 field relative-L2、结构边缘距离、积分量误差及不确定性覆盖率。
- 跨日期/相机/光圈分别报告，不把 pooled mean 掩盖最差装置；同时报告算力、查询和人工标定成本。

**停止条件**

- 数据合同任一关键项缺失，保持 `UNCONSTRUCTED`，不启动正式比较。
- fresh session 泄漏、事后改阈值或选择性丢弃失败 run，该轮作废。
- 只有重投影改善而无独立 truth，不得宣称三维场更准确；只有均值改善而预注册安全/复现门未过，不得宣称胜出。

## 3. 统一比较合同

每一级都生成一份不可变 manifest，至少包含：来源 URL/DOI、版本/commit、许可证快照、文件名/字节数/hash、split、几何、单位、预处理、随机种子、基线、预算、指标、停止阈值和解封时间。

统一规则：

- baseline 与候选方法使用相同观测、mask、split、forward 定义和算力/查询预算。
- development 只选超参数；test/fresh 只运行一次主分析。补充诊断标为 post-open，不追溯改判。
- 报告每样本/每视角分布、最差组、失败数与置信区间，不只报告平均值。
- 允许结论为 `PASS_MECHANISM_ONLY`、`PASS_MEASUREMENT_DOMAIN_ONLY`、`NO-GO`、`UNKNOWN_LICENSE` 或 `UNCONSTRUCTED`。
- “方法胜出”至少要求：预注册主指标通过、关键次指标不显著恶化、跨随机种子稳定、预算公平、独立 test/fresh 未泄漏；本文不声称这些条件已经满足。

## 4. 最小下载顺序

严格按以下顺序执行；每一步都允许停止：

1. **只读网页/API metadata**：记录所有 DOI、许可、file manifest、字节数、checksum 和版本；不下载 PDF 或数据包。
2. **代码与小文档**：Walnut/LoDoPaB 技术代码、TRON-BOS/`photon` 仓库 metadata；根许可证不明确时只做远端 metadata 检查，不 clone/use。论文优先 HTML/arXiv 摘要页，不抓受限 PDF。
3. **Walnut 最小矩阵包**：从 Zenodo manifest 选择最低分辨率 `A + sinogram + reference`，先完成 L0。
4. **LoDoPaB metadata + split CSV**：只有 L0 通过且确需大样本时，才规划一个成对分片；默认不下载 55 GB。
5. **helium BOS 单包**：只为真实图像前端选择一个约 0.85--0.91 GB BOS ZIP，不下载两组 schlieren 包；按单视角时间序列和单个首帧 tare 处理。
6. **TRON-BOS 样例**：从 DOI `10.5281/zenodo.18118110` 按 record/file metadata 所示 `CC BY 4.0` 获取约 16.314777708 GB 内容并保留快照；独立 GitHub 仓库许可仍为 `unknown`，不 clone/use；先过流动漂移停止门。
7. **`photon` sample + CUDA 构建**：仅在 NVIDIA 服务器、依赖和许可证审计完成后运行；冻结用户输入场、单位、Gladstone-Dale 关系、边界、模型族和阈值。
8. **PSU frozen held-out**：readme/manifest/小代码优先；5.23 GB 与 9.25 GiB 仅作为 local audit measurement。权利确认前不公开再分发；只有 `photon` 模型族/阈值冻结后才打开 held-out。
9. **OERF**：最后接入组内冻结 session；在许可与数据合同完成前不复制、不同步或公开任何原始数据。

## 5. 链接与许可核验方法

每次正式运行前重新核验，避免仅凭本文日期的快照：

```bash
# 1) DOI/网页只检查响应链与最终 URL，不下载正文或数据
curl -sSIL --max-redirs 5 'https://doi.org/10.5281/zenodo.1254206'
curl -sSIL --max-redirs 5 'https://doi.org/10.5281/zenodo.3384092'
curl -sSIL --max-redirs 5 'https://doi.org/10.5281/zenodo.6136052'
curl -sSIL --max-redirs 5 'https://doi.org/10.5281/zenodo.18118110'
curl -sSIL --max-redirs 5 'https://doi.org/10.26208/1VE2-5C19'

# 2) Zenodo JSON 只取 metadata/file manifest；保存 license、size、checksum、version
curl -sS 'https://zenodo.org/api/records/1254206'
curl -sS 'https://zenodo.org/api/records/3384092'
curl -sS 'https://zenodo.org/api/records/6136052'
curl -sS 'https://zenodo.org/api/records/18118110'

# 3) GitHub 许可证与 commit；不要把仓库“Public”误判为有许可证
git ls-remote https://github.com/3dfernando/tron-bos.git HEAD
git ls-remote https://github.com/lalitkrajendran/photon.git HEAD
curl -sSIL https://github.com/3dfernando/tron-bos/blob/main/LICENSE
curl -sSIL https://github.com/lalitkrajendran/photon/blob/master/LICENSE.txt
```

核验记录必须保存“检查日期、最终 URL、HTTP 状态、版本/commit、许可证标识、文件字节数与 checksum”。`HEAD 200` 只证明链接可达，不证明内容许可；Zenodo `Open` 或页面通用页脚也不自动授权具体 record/files。TRON-BOS 的 Zenodo record 及其 listed files 按记录所示 `CC BY 4.0` 并保留快照，独立 GitHub 仓库记 `unknown`；PSU 记 **`record-level unknown pending rights confirmation`**，确认前不公开再分发。

## 6. 当前决策

当前最小可执行路线是：**Walnut L0 先建立数值 oracle；LoDoPaB 仅作可选机制扩展；helium 单视角时间序列先做真实图像前端；TRON-BOS 过流动漂移门后做旋转测量审计；`photon` 用冻结的用户输入场做独立合成 renderer；冻结 `photon` 模型族与阈值后才打开 PSU held-out；若未来构建、批准并冻结 OERF independent session，才可能打开该层。**

在 OERF 数据合同和独立验证未到位前，最终状态保持 `UNCONSTRUCTED`。任何 CT 改善、真实 BOS 重投影改善或 synthetic renderer 改善，都不能单独升级为“BOST 三维重建已被证明”或“现有算法胜出”。
