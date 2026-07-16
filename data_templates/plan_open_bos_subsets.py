#!/usr/bin/env python3
"""Create deterministic limited-view subset plans from the Open BOS view manifest."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_templates"
VIEW_MANIFEST_CSV = DATA_DIR / "open_bos_view_manifest.csv"
SUBSET_JSON = DATA_DIR / "open_bos_subset_plans.json"
SUBSET_MD = DATA_DIR / "open_bos_subset_plans.md"

PRESETS = [5, 7, 9, 13, 21, 70]


def read_views() -> list[dict[str, str]]:
    with VIEW_MANIFEST_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["rotation_id"], r["cam_id"]))
    return rows


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def compact_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "view_id": row["view_id"],
        "rotation_id": row["rotation_id"],
        "cam_id": row["cam_id"],
        "ref_image_path": row["ref_image_path"],
        "def_image_path": row["def_image_path"],
        "hsof_paths": row["hsof_paths"],
        "mask_path": row["mask_path"],
        "calibration_path_hint": row["calibration_path_hint"],
        "calibration_mapping_status": row["calibration_mapping_status"],
    }


def make_plans(rows: list[dict[str, str]]) -> dict:
    plans = {}
    total = len(rows)
    for count in PRESETS:
        selected_indices = evenly_spaced_indices(total, count)
        selected = [compact_row(rows[i]) for i in selected_indices]
        rotations = sorted({row["rotation_id"] for row in selected})
        cams = sorted({row["cam_id"] for row in selected})
        plans[f"{count}_views"] = {
            "count": count,
            "selection_rule": f"Evenly spaced over the sorted {total}-view manifest by rotation_id then cam_id.",
            "indices": selected_indices,
            "rotation_coverage": rotations,
            "camera_coverage": cams,
            "views": selected,
        }
    return {
        "generated_date": date.today().isoformat(),
        "source_manifest": str(VIEW_MANIFEST_CSV.relative_to(ROOT)),
        "total_views": total,
        "presets": PRESETS,
        "warning": "These are deterministic planning subsets, not experimentally optimized view-selection results. Confirm geometry, angle mapping, and OERF priorities with He Yuanzhe before using them as final experiments.",
        "plans": plans,
    }


def write_markdown(data: dict) -> None:
    lines = [
        "# Open BOS limited-view subset plans",
        "",
        f"生成日期：{data['generated_date']}",
        "",
        "用途：从 `open_bos_view_manifest.csv` 的 70 个 REF/DEF image-pair views 中，生成 5/7/9/13/21/70 views 的确定性子采样方案。它不是最优视角选择算法，只是本科预演时的第一版可复现实验入口。",
        "",
        "## 使用边界",
        "",
        "- 这些 subset 只按 `rotation_id` 和 `cam_id` 排序后等间隔抽样，目的是快速形成 baseline。",
        "- 真正写论文前，需要从 Open BOS 论文/脚本确认 `ROT_***` 到 calibration angle / geometry 的映射。",
        "- 迁移到 OERF 时，应把 `rotation_id` / `cam_id` 换成组内真实 view id、camera matrix 和 ray model。",
        "",
        "## 子采样摘要",
        "",
        "| preset | rotations covered | cameras covered | first view | last view |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for name, plan in data["plans"].items():
        views = plan["views"]
        lines.append(
            f"| `{name}` | {len(plan['rotation_coverage'])} | {len(plan['camera_coverage'])} | `{views[0]['view_id']}` | `{views[-1]['view_id']}` |"
        )

    lines.extend([
        "",
        "## 9-view 预案",
        "",
        "这是最适合先和何远哲讨论的版本：视角数足够少，能模拟 limited-view BOST，又比 5-view 稳一点。",
        "",
        "| # | view_id | rotation | camera | key paths |",
        "| ---: | --- | --- | --- | --- |",
    ])
    for i, row in enumerate(data["plans"]["9_views"]["views"], start=1):
        lines.append(
            f"| {i} | `{row['view_id']}` | `{row['rotation_id']}` | `{row['cam_id']}` | REF/DEF + HSOF + MASK listed in CSV |"
        )

    lines.extend([
        "",
        "## 建议实验命名",
        "",
        "| 实验名 | 目的 |",
        "| --- | --- |",
        "| `open_bos_05view_loader_smoke` | 检查 loader 能否读最小 subset。 |",
        "| `open_bos_09view_baseline_reproj` | 做第一版 limited-view reprojection baseline。 |",
        "| `open_bos_13view_vs_21view` | 比较视角数对重投影误差和运行时间的影响。 |",
        "| `open_bos_70view_reference_readonly` | 只做全量索引/报告，不一开始跑完整重构。 |",
        "",
        "## 给何远哲的一句话",
        "",
        "> 我已经把公开 Open BOS 的 70 个 REF/DEF views 做成 manifest，并生成 5/7/9/13/21/70 views 的 deterministic subset。这个不是声称视角最优，而是为了先跑 loader、mask、deflection 和 report。如果组内九视角 BOST 数据可以给到类似字段，我可以把同一套接口迁移过去。",
    ])
    SUBSET_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    rows = read_views()
    data = make_plans(rows)
    SUBSET_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    write_markdown(data)
    print(f"Wrote {SUBSET_JSON.relative_to(ROOT)}")
    print(f"Wrote {SUBSET_MD.relative_to(ROOT)}")
    print(f"Generated presets: {', '.join(str(p) for p in PRESETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
