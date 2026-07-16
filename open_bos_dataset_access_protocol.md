# Open BOS Tomography Dataset 访问与本科预演协议

生成日期：2026-07-07；DataCite / Crossref 复核：2026-07-09

用途：把 Molnar/Grauer 等人的公开 70 视角 BOS/TBOS 数据集转成可执行的本科毕设预案。它是没有 OERF 真实数据时最贴近 BOST/NeRIF 的公开 benchmark，但物理对象是高速流飞行体，不是火焰。

## 1. 已核验公开入口

| 类型 | 入口 | 用途 |
| --- | --- | --- |
| 正式论文 | `https://doi.org/10.1007/s00348-026-04189-z` | Experiments in Fluids 67, article 44 (2026)，正式引用入口 |
| arXiv 可读版 | `https://arxiv.org/abs/2508.17120` | 开放阅读 PDF，论文库已缓存 |
| Data Commons 目录 | `https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/` | 12 个 zip 和官方 README |
| DataCite 元数据 | `https://doi.org/10.26208/1VE2-5C19` -> Penn State `Dataset=6500` | 已核为 Dataset；creators 包含 Molnar、Singh、Clifford、Thayer、Peltier、Jones、Grauer；`rightsList` 为空，因此代码/数据可作公开 benchmark 入口，但代码复用仍需谨慎 |
| 本地目录索引 | `data_templates/open_bos_zip_file_content.txt` | 已缓存 255 KB 的 zip 内容清单，不含大体量数据 |
| 本地结构摘要 | `data_templates/open_bos_index_summary.json` / `data_templates/open_bos_view_plan.md` | 从官方清单自动解析出的目录规模、70 视角矩阵和本科预演任务 |
| 本地 70 行 manifest | `data_templates/open_bos_view_manifest.csv` | 每行对应一个 `ROT_*** + cam_**`，列出 REF/DEF、CC、HSOF、WOF40、mask 路径提示 |
| 本地 limited-view 预案 | `data_templates/open_bos_subset_plans.json` / `data_templates/open_bos_subset_plans.md` | 5/7/9/13/21/70 views 的确定性子采样入口 |
| 视角矩阵图 | `figures/open_bos_view_grid.svg` | 10 个 rotation group x 7 台相机的可用性图 |

论文关键信息：

- 数据：70 views of high-speed flow over a flight body。当前官方清单解析显示，这 70 views 对应 `def_data/REF_IMGS` 与 `def_data/DEF_IMGS` 下 10 个 `ROT_***` 旋转组 x 7 台相机的 REF/DEF image pairs。
- calibration：`cal_data/Angle_120_deg` 到 `Angle_240_deg` 共 13 个角度 x 7 台相机，合计 91 个 `.mat` 标定文件；它和 70 views 不是同一个计数体系，ROT-to-Angle 映射需要从论文/脚本继续确认。
- 内容：reference / deflected images、calibration data、deflection fields、sample tomographic reconstructions。
- 方法：NIRT with total variation regularization；另有 3D compressible Euler data assimilation。
- 对毕设价值：可做 70 -> 21 / 13 / 9 / 7 / 5 views 子采样，预演 NeRIF/BOST 的 loader、mask、deflection、reprojection、uncertainty 和 report。

## 2. Data Commons 体量

| 文件 | 大小 |
| --- | --- |
| `000-readme-zip-file-content.txt` | 254,998 bytes |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-01.zip` | 3.41 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-02.zip` | 4.23 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-03.zip` | 4.24 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-04.zip` | 4.28 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-05.zip` | 3.81 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-06.zip` | 4.77 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-07.zip` | 4.53 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-08.zip` | 4.18 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-09.zip` | 4.55 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-10.zip` | 4.52 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-11.zip` | 4.29 GiB |
| `molnar-et-al-open-source-bos-tomography-dataset-2025-12.zip` | 1.31 GiB |

总量约 51.66 GB，约 48.12 GiB。不要把 zip 数据放进网页仓库；只把 manifest、目录索引、下载脚本和小图表放进仓库。

## 3. README 结构摘要

从 `data_templates/open_bos_zip_file_content.txt` 提取到：

| 项目 | 摘要 |
| --- | --- |
| 文件条目 | 1838 个文件条目 |
| 顶层目录 | `batch/`、`cad/`、`cal_data/`、`cal_pkg/`、`data/`、`def_data/`、`pyscripts/`、`results/`、`scripts/`、`tools/` |
| calibration 角度 | `Angle_120_deg` 到 `Angle_240_deg`，共 13 个角度 x 7 台相机 = 91 个 `.mat` 标定文件 |
| flow views | `ROT_000` 到 `ROT_090`，共 10 个旋转组 x 7 台相机 = 70 个 REF/DEF image-pair views |
| 相机 | `cam_01` 到 `cam_07`，共 7 台相机 |
| 典型文件 | `.mat` calibration/deflection、`.tiff`/`.jpg`/`.png` 图像、`.stl`/`.step` CAD、MATLAB `.m` scripts、Python `.py` scripts、results 可视化 |
| 重点子目录 | `cal_data/Angle_*_deg/`、`def_data/REF_IMGS/`、`def_data/DEF_IMGS/`、`results/R076/`、`results/Deflections/`、`data/DEF_PROC/`、`data/MASK_PROC/` |

这说明它不仅是论文附件，而是能练习完整数据链路的公开数据包：CAD -> calibration -> raw/reference images -> deflection processing -> reconstruction/results。

### 不要混淆的相邻论文

`Convection Over Coffee: Revisiting a Steamy Background-Oriented Schlieren Dataset` 已通过 Crossref 核到 AIAA SCITECH 2026 DOI `10.2514/6.2026-1499`，但本轮没有发现独立 DataCite/Zenodo/OSF/PSU Data Commons 下载入口。它目前应作为 BOS dataset / 教学热对流案例的论文线索，而不是能马上跑的公开 benchmark。真正可以规划 loader、manifest 和 limited-view 实验的是 `10.26208/1VE2-5C19` 这个 70-view Open BOS tomography dataset。

### 代码目录补充

2026-07-08 复核 arXiv HTML 与 DataCite：论文末尾把数据和代码统一指向 DOI `10.26208/1VE2-5C19`，DataCite 摘要明确写有 full tomography codebase，但 `rightsList` 当前为空。因此它可以作为公开 benchmark 和代码结构参考，不应直接把代码复制进你的毕业仓库。

从官方 zip 内容清单可见，代码/脚本相关条目至少包括：

| 目录 | 条目数 | 未压缩字节 | 对你最有用的线索 |
| --- | ---: | ---: | --- |
| `pyscripts/` | 12 | 82,823 | `NIRT.py`、`train.py`、`network.py`、`sample.py`、`reg.py`、`meas.py`，对应 NIRT 训练和正则化结构 |
| `scripts/` | 379 | 273,161,473 | MATLAB preprocessing / visualization / reconstruction scripts，以及大量论文结果图 |
| `tools/` | 177 | 1,114,642 | `HornSchunck.m`、camera/background helpers、mesh voxelisation 工具和文件级 license 线索 |
| `batch/` | 1 | 521 | `BATCH_R076.cmd` 批处理入口 |

本科阶段建议只先读 `pyscripts/README.txt`、`NIRT.py`、`train.py`、`network.py` 的接口思想，再把变量名迁移到自己的轻量 PyTorch / NumPy demo。是否可直接运行或复用原代码，需要先确认 Data Commons / 作者许可。

## 4. 本科预演路线

### Day 1: 只读目录索引

- 不下载 51.66 GB 全量数据。
- 阅读 `open_bos_zip_file_content.txt`，标出 `cal_data`、`def_data`、`results` 和 `scripts` 的路径。
- 填 `data_templates/open_bos_manifest.json` 中的 `fields_to_extract`。
- 运行 `python3 data_templates/summarize_open_bos_index.py`，生成 `open_bos_index_summary.json`、`open_bos_view_manifest.csv`、`open_bos_view_plan.md` 和 `open_bos_view_grid.svg`。
- 运行 `python3 data_templates/plan_open_bos_subsets.py`，生成 5/7/9/13/21/70 views 的 deterministic subset 预案。

### Day 2: 下载最小子集

下载策略先保守：

1. 优先下载 `000-readme-zip-file-content.txt`。
2. 只下载一个 zip 或请求师兄确认是否需要 full dataset。
3. 若磁盘允许，先处理 calibration / masks / deflection results，不急着处理所有 raw images。

建议本地目录：

```text
data/open_bos/
  raw_archive/
    000-readme-zip-file-content.txt
    molnar-et-al-open-source-bos-tomography-dataset-2025-XX.zip
  extracted_minimal/
    cal_data/
    def_data/
    results/
  reports/
```

### Day 3-4: 写 loader

最小 loader 只需要输出：

| 字段 | 说明 |
| --- | --- |
| `case_id` | 例如 `R076` |
| `angle_deg` | 120-240 deg |
| `cam_id` | 1-7 |
| `calibration_path` | 对应 `.mat` |
| `reference_image_path` | flow-off / reference |
| `deflected_image_path` | flow-on / disturbed |
| `deflection_path` | 处理后的 deflection field |
| `mask_path` | 若有 mask |
| `can_use_in_public_report` | open benchmark 为 true；OERF 内部数据另设权限 |

当前 `data_templates/open_bos_view_manifest.csv` 已经按这个思路生成 70 行轻量清单。它先把文件路径组织起来，不下载图像；真正做 reconstruction 前，需要确认 `ROT_***` 与 `cal_data/Angle_*_deg` 的几何映射。

当前 `data_templates/open_bos_subset_plans.md` 已经给出 9-view 预案，可作为第一次 limited-view BOST 讨论入口。注意：这个方案只是等间隔抽样 baseline，不是论文级 optimal view selection。

### Day 5-7: 做三个图

1. `view_grid.png/svg`：10 个 `ROT_***` x 7 相机的 70 view 可用性矩阵。
2. `nine_view_subset.png`：从 70 views 中选 9 views 的规则和覆盖角度。
3. `deflection_quality_report.png`：位移范围、mask 覆盖、缺失视角、异常值。

## 5. 和 OERF/何远哲真实数据的映射

| Open BOS 字段 | 迁移到 OERF BOST 时替换什么 |
| --- | --- |
| `Angle_*_deg` + `cam_*` | OERF 九视角或内窥视角编号 |
| `cal_data/*.mat` | 组内 camera matrix / ray model / view geometry |
| `REF_IMGS` / `DEF_IMGS` | flow-off / flow-on 背景图或 BOS pair |
| `DEF_PROC` / `results/Deflections` | 组内位移场或 optical-flow output |
| `MASK_PROC` | ROI / flame mask / valid reconstruction region |
| CAD files | burner / chamber / optical window geometry |
| NIRT reconstructions | OERF voxel / NeRIF / BOST reference result |

不要把高速飞行体的物理结论写成火焰结论。它的用途是验证数据接口、少视角选择、重投影误差和报告工具。

## 6. 开题可写边界

> 若暂时无法获取 OERF 内部 BOST 数据，本课题将使用公开 Open-source BOS tomography dataset of high-speed flow over a flight body 作为外部 benchmark，预演多视角数据组织、有限视角子采样、位移场读取、重投影验证和自动报告流程。该数据集物理对象不同于火焰，但其多视角 BOS/TBOS 观测形式与 BOST/NeRIF 数据接口高度相邻，可作为真实数据迁移前的工程验证。

## 7. 给何远哲的确认问题

1. 如果组内数据暂时不能给，是否认可用这个 open BOS dataset 做 loader 和 report 预演？
2. 组内九视角数据是否也能抽象成 `angle/view/camera/mask/deflection` 这类 manifest？
3. 您更想先看到 9-view subset 的重构结果，还是 data-quality report？
4. 真实数据里是否有类似 `REF_IMGS` / `DEF_IMGS` 的原始图，还是只给位移场？
5. OERF 的 calibration 是 `.mat`、`.json`、还是组内自定义格式？
6. 论文里公开展示 open BOS 结果，真实 OERF 数据只做内部验证，这样是否合适？
