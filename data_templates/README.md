# OERF BOST / PIV-BOST 真实数据接入模板

用途：当何远哲给你一小份真实 BOST、PIV-BOST 或 4D BOST 数据时，用统一 manifest 管理文件、单位、几何、公开边界和缺失项。

## 为什么需要 manifest

真实实验数据最容易卡在这些地方：

- 有图片，但没有 view id 和裁剪规则。
- 有位移场，但不知道单位是 pixel 还是 mm。
- 有标定文件，但不知道坐标轴方向。
- 有重构切片，但不知道是否能写入本科论文。
- 有 PIV 图像，但不知道和哪一帧 BOST 同步。

manifest 的作用是把这些信息写在同一个文件里，避免数据一多就混乱。

## 推荐目录

```text
data_real/
  oerf_bost_sample_YYYYMMDD/
    manifest.json
    raw/
      flow_off/
      flow_on/
      masks/
    displacement/
      view_01_uv.npy
      view_02_uv.npy
    calibration/
      camera_intrinsics.json
      view_geometry.json
      volume_bounds.json
    reference/
      voxel_reconstruction.npy
      nerif_slice_xy.png
    notes/
      readme_from_he.md

  oerf_piv_bost_sample_YYYYMMDD/
    manifest.json
    piv/
      image_pairs/
      velocity_fields/
    bost/
      manifest.json
    synchronization/
      timestamps.csv
```

## 文件说明

- `bost_sample_manifest.json`：九视角 BOST 样例 manifest。
- `piv_bost_manifest.json`：同步 PIV-BOST 样例 manifest。
- `open_bos_manifest.json`：公开 BOS/TBOS benchmark manifest，用于无组内数据时的 open-source pipeline 预演。
- `open_bos_zip_file_content.txt`：Penn State Data Commons 官方 zip 内容清单，已缓存为轻量索引；不要把 12 个大 zip 放进网页仓库。
- `summarize_open_bos_index.py`：解析官方 zip 内容清单，生成视角矩阵、JSON 摘要和 70 行 view manifest。
- `open_bos_index_summary.json`：机器可读索引摘要，区分 13 x 7 calibration 文件和 10 x 7 = 70 个 REF/DEF flow views。
- `open_bos_view_manifest.csv`：70 行视角清单，每行对应一个 `ROT_*** + cam_**`，含 REF/DEF、CC、HSOF、WOF40、mask 路径提示。
- `open_bos_view_plan.md`：给人看的 Open BOS 视角预演页，可直接发给师兄讨论。
- `plan_open_bos_subsets.py`：从 70 行 view manifest 生成 5/7/9/13/21/70 views 的 deterministic limited-view 子采样方案。
- `open_bos_subset_plans.json`：机器可读子采样方案，含每个 preset 的 view ids、rotation coverage 和 camera coverage。
- `open_bos_subset_plans.md`：给人看的 limited-view 子采样说明，适合拿 9-view 方案先问何远哲。
- `../figures/open_bos_view_grid.svg`：70 视角可用性矩阵图。
- `validate_manifest.py`：检查 manifest 必要字段，并可检查文件路径是否存在。

## 使用方式

复制一个 manifest 到真实数据目录：

```bash
cp data_templates/bost_sample_manifest.json data_real/oerf_bost_sample_YYYYMMDD/manifest.json
```

先只检查字段：

```bash
python3 data_templates/validate_manifest.py data_real/oerf_bost_sample_YYYYMMDD/manifest.json --allow-missing
```

等真实文件都放好后，检查字段和文件路径：

```bash
python3 data_templates/validate_manifest.py data_real/oerf_bost_sample_YYYYMMDD/manifest.json
```

重新生成 Open BOS 轻量索引：

```bash
python3 data_templates/summarize_open_bos_index.py
```

这一步不下载 51.66 GB 全量数据，只读取已经缓存的 `open_bos_zip_file_content.txt`。当前解析结果显示：`cal_data/Angle_*_deg` 是 13 个角度 x 7 台相机的 calibration 文件；论文 reported 70 views 对应的是 `def_data/REF_IMGS` 和 `def_data/DEF_IMGS` 下的 10 个 `ROT_***` 旋转组 x 7 台相机。ROT 到具体 calibration angle 的映射需要从论文或脚本继续确认，不能只凭目录名推断。

生成 limited-view 子采样方案：

```bash
python3 data_templates/plan_open_bos_subsets.py
```

这一步会输出 5/7/9/13/21/70 views preset。它只是等间隔抽样的第一版实验入口，不是最优视角选择算法；真正进入论文实验前，需要和何远哲确认几何、视角编号和 OERF 数据目标。

## 和何远哲沟通时的说法

师兄，如果您方便给一小份 BOST 样例，我可以按这个 manifest 整理：原始图、mask、位移场、标定、参考重构和可公开边界。这样我第一周能先交付数据读取、可视化和重投影检查，不会一上来改您的原始流程。

## 最低可用字段

一份 BOST 数据最少需要：

- `dataset_id`
- `modality`
- `views`
- `raw.flow_off`
- `raw.flow_on`
- `geometry.volume_bounds`
- `displacement.fields`
- `permissions`

一份 PIV-BOST 数据还需要：

- `piv.image_pairs`
- `piv.delta_t`
- `synchronization`
- `piv.velocity_units`
