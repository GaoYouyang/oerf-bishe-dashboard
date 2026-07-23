# Open BOS limited-view subset plans

生成日期：2026-07-07

用途：从 `open_bos_view_manifest.csv` 的 70 个 REF/DEF image-pair views 中，生成 5/7/9/13/21/70 views 的确定性子采样方案。它不是最优视角选择算法，只是本科预演时的第一版可复现实验入口。

## 使用边界

- 这些 subset 只按 `rotation_id` 和 `cam_id` 排序后等间隔抽样，目的是快速形成 baseline。
- 真正写论文前，需要从 Open BOS 论文/脚本确认 `ROT_***` 到 calibration angle / geometry 的映射。
- 迁移到 OERF 时，应把 `rotation_id` / `cam_id` 换成组内真实 view id、camera matrix 和 ray model。

## 子采样摘要

| preset | rotations covered | cameras covered | first view | last view |
| --- | ---: | ---: | --- | --- |
| `5_views` | 5 | 3 | `ROT_000_cam_01` | `ROT_090_cam_07` |
| `7_views` | 7 | 5 | `ROT_000_cam_01` | `ROT_090_cam_07` |
| `9_views` | 9 | 7 | `ROT_000_cam_01` | `ROT_090_cam_07` |
| `13_views` | 10 | 7 | `ROT_000_cam_01` | `ROT_090_cam_07` |
| `21_views` | 10 | 3 | `ROT_000_cam_01` | `ROT_090_cam_07` |
| `70_views` | 10 | 7 | `ROT_000_cam_01` | `ROT_090_cam_07` |

## 9-view 预案

这是最适合先和何远哲讨论的版本：视角数足够少，能模拟 limited-view BOST，又比 5-view 稳一点。

| # | view_id | rotation | camera | key paths |
| ---: | --- | --- | --- | --- |
| 1 | `ROT_000_cam_01` | `ROT_000` | `cam_01` | REF/DEF + HSOF + MASK listed in CSV |
| 2 | `ROT_010_cam_03` | `ROT_010` | `cam_03` | REF/DEF + HSOF + MASK listed in CSV |
| 3 | `ROT_020_cam_04` | `ROT_020` | `cam_04` | REF/DEF + HSOF + MASK listed in CSV |
| 4 | `ROT_030_cam_06` | `ROT_030` | `cam_06` | REF/DEF + HSOF + MASK listed in CSV |
| 5 | `ROT_040_cam_07` | `ROT_040` | `cam_07` | REF/DEF + HSOF + MASK listed in CSV |
| 6 | `ROT_060_cam_02` | `ROT_060` | `cam_02` | REF/DEF + HSOF + MASK listed in CSV |
| 7 | `ROT_070_cam_04` | `ROT_070` | `cam_04` | REF/DEF + HSOF + MASK listed in CSV |
| 8 | `ROT_080_cam_05` | `ROT_080` | `cam_05` | REF/DEF + HSOF + MASK listed in CSV |
| 9 | `ROT_090_cam_07` | `ROT_090` | `cam_07` | REF/DEF + HSOF + MASK listed in CSV |

## 建议实验命名

| 实验名 | 目的 |
| --- | --- |
| `open_bos_05view_loader_smoke` | 检查 loader 能否读最小 subset。 |
| `open_bos_09view_baseline_reproj` | 做第一版 limited-view reprojection baseline。 |
| `open_bos_13view_vs_21view` | 比较视角数对重投影误差和运行时间的影响。 |
| `open_bos_70view_reference_readonly` | 只做全量索引/报告，不一开始跑完整重构。 |

## 给何远哲的一句话

> 我已经把公开 Open BOS 的 70 个 REF/DEF views 做成 manifest，并生成 5/7/9/13/21/70 views 的 deterministic subset。这个不是声称视角最优，而是为了先跑 loader、mask、deflection 和 report。如果组内九视角 BOST 数据可以给到类似字段，我可以把同一套接口迁移过去。
