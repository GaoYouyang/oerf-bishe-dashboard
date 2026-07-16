#!/usr/bin/env python3
"""Summarize the cached Open BOS Data Commons zip-content index."""

from __future__ import annotations

import json
import re
import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_templates"
INDEX_PATH = DATA_DIR / "open_bos_zip_file_content.txt"
SUMMARY_JSON = DATA_DIR / "open_bos_index_summary.json"
SUMMARY_MD = DATA_DIR / "open_bos_view_plan.md"
VIEW_MANIFEST_CSV = DATA_DIR / "open_bos_view_manifest.csv"
VIEW_GRID_SVG = ROOT / "figures" / "open_bos_view_grid.svg"

DATASET_PREFIX = "molnar-et-al-open-source-bos-tomography-dataset-2025/"
ENTRY_RE = re.compile(r"^\s*(\d+)\s+\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s+(.+?)\s*$")
ARCHIVE_RE = re.compile(r"Content of ZIP archive\s+(\S+\.zip)")
ARCHIVE_LINE_RE = re.compile(r"Archive:\s+(\S+\.zip)")
CAL_RE = re.compile(r"^cal_data/Angle_(\d+)_deg/ang_\d+_deg_cam_(\d+)\.mat$")
BG_CAL_RE = re.compile(r"^cal_data/Background/cam_(\d+)\.mat$")
REF_DEF_RE = re.compile(
    r"^def_data/(REF_IMGS|DEF_IMGS)/(REF_ROT|DEF_ROT)_(\d+)/"
    r"(?:REF_ROT|DEF_ROT)_\d+_CAM_(\d+)\.tiff$"
)
CC_RE = re.compile(r"^def_data/DEF_IMGS/DEF_ROT_(\d+)/CC_DEF_ROT_\d+\.mat$")
HSOF_RE = re.compile(r"^(?:def_data/DEF_IMGS/DEF_ROT_(\d+)/|data/DEF_PROC/)HSOF_DEF_ROT_(\d+)\.mat$")
WOF_RE = re.compile(r"^def_data/DEF_IMGS/DEF_ROT_(\d+)/WOF40_DEF_ROT_\d+\.mat$")
MASK_RE = re.compile(r"^data/MASK_PROC/MASKS_ROT_(\d+)\.mat$")


def strip_dataset_prefix(path: str) -> str:
    if path.startswith(DATASET_PREFIX):
        return path[len(DATASET_PREFIX):]
    return path


def parse_entries() -> tuple[list[dict], list[dict], list[str]]:
    files: list[dict] = []
    dirs: list[dict] = []
    archives: list[str] = []
    current_archive = "unknown"

    for line in INDEX_PATH.read_text(errors="replace").splitlines():
        archive_match = ARCHIVE_RE.search(line) or ARCHIVE_LINE_RE.search(line)
        if archive_match:
            current_archive = archive_match.group(1)
            if current_archive not in archives:
                archives.append(current_archive)
            continue

        entry_match = ENTRY_RE.match(line)
        if not entry_match:
            continue

        size = int(entry_match.group(1))
        raw_path = entry_match.group(2)
        path = strip_dataset_prefix(raw_path)
        if not path:
            continue

        entry = {
            "archive": current_archive,
            "path": path,
            "bytes": size,
        }
        if raw_path.endswith("/"):
            dirs.append(entry)
        else:
            files.append(entry)

    return files, dirs, archives


def extension(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return suffix if suffix else "[none]"


def collect_summary(files: list[dict], dirs: list[dict], archives: list[str]) -> dict:
    top_level_counts: Counter[str] = Counter()
    top_level_bytes: Counter[str] = Counter()
    ext_counts: Counter[str] = Counter()
    ext_bytes: Counter[str] = Counter()
    archive_counts: dict[str, Counter[str]] = defaultdict(Counter)
    archive_bytes: dict[str, Counter[str]] = defaultdict(Counter)

    cal_grid: dict[int, set[int]] = defaultdict(set)
    background_cal_cams: set[int] = set()
    ref_grid: dict[str, set[int]] = defaultdict(set)
    def_grid: dict[str, set[int]] = defaultdict(set)
    ref_paths: dict[tuple[str, int], str] = {}
    def_paths: dict[tuple[str, int], str] = {}
    cc_rotations: set[str] = set()
    hsof_rotations: set[str] = set()
    wof40_rotations: set[str] = set()
    mask_rotations: set[str] = set()
    cc_paths: dict[str, str] = {}
    hsof_paths: dict[str, list[str]] = defaultdict(list)
    wof40_paths: dict[str, str] = {}
    mask_paths: dict[str, str] = {}
    sample_paths: dict[str, list[str]] = defaultdict(list)
    results_dirs: Counter[str] = Counter()

    for item in files:
        path = item["path"]
        top = path.split("/", 1)[0]
        top_level_counts[top] += 1
        top_level_bytes[top] += item["bytes"]
        ext = extension(path)
        ext_counts[ext] += 1
        ext_bytes[ext] += item["bytes"]
        archive_counts[item["archive"]][top] += 1
        archive_bytes[item["archive"]][top] += item["bytes"]

        if len(sample_paths[top]) < 6:
            sample_paths[top].append(path)

        if match := CAL_RE.match(path):
            cal_grid[int(match.group(1))].add(int(match.group(2)))
        if match := BG_CAL_RE.match(path):
            background_cal_cams.add(int(match.group(1)))
        if match := REF_DEF_RE.match(path):
            root, _, rotation, cam = match.groups()
            cam_id = int(cam)
            if root == "REF_IMGS":
                ref_grid[rotation].add(cam_id)
                ref_paths[(rotation, cam_id)] = path
            else:
                def_grid[rotation].add(cam_id)
                def_paths[(rotation, cam_id)] = path
        if match := CC_RE.match(path):
            cc_rotations.add(match.group(1))
            cc_paths[match.group(1)] = path
        if match := HSOF_RE.match(path):
            rotation = match.group(1) or match.group(2)
            hsof_rotations.add(rotation)
            hsof_paths[rotation].append(path)
        if match := WOF_RE.match(path):
            wof40_rotations.add(match.group(1))
            wof40_paths[match.group(1)] = path
        if match := MASK_RE.match(path):
            mask_rotations.add(match.group(1))
            mask_paths[match.group(1)] = path
        if path.startswith("results/"):
            parts = path.split("/")
            if len(parts) > 1:
                results_dirs[parts[1]] += 1

    all_rotations = sorted(set(ref_grid) | set(def_grid), key=int)
    all_cams = sorted(set().union(*ref_grid.values(), *def_grid.values()) if all_rotations else [])
    view_matrix = []
    view_manifest_rows = []
    complete_ref_def_pairs = 0
    for rotation in all_rotations:
        ref_cams = sorted(ref_grid.get(rotation, set()))
        def_cams = sorted(def_grid.get(rotation, set()))
        both_cams = sorted(set(ref_cams) & set(def_cams))
        complete_ref_def_pairs += len(both_cams)
        view_matrix.append({
            "rotation_id": rotation,
            "ref_cams": [f"cam_{cam:02d}" for cam in ref_cams],
            "def_cams": [f"cam_{cam:02d}" for cam in def_cams],
            "ref_def_pair_cams": [f"cam_{cam:02d}" for cam in both_cams],
            "has_cc_mat": rotation in cc_rotations,
            "has_hsof_mat": rotation in hsof_rotations,
            "has_wof40_mat": rotation in wof40_rotations,
            "has_mask_proc": rotation in mask_rotations,
        })
        for cam in all_cams:
            cam_name = f"cam_{cam:02d}"
            if cam in set(ref_cams) | set(def_cams):
                view_manifest_rows.append({
                    "view_id": f"ROT_{rotation}_{cam_name}",
                    "rotation_id": f"ROT_{rotation}",
                    "cam_id": cam_name,
                    "ref_image_path": ref_paths.get((rotation, cam), ""),
                    "def_image_path": def_paths.get((rotation, cam), ""),
                    "cc_path": cc_paths.get(rotation, ""),
                    "hsof_paths": ";".join(sorted(hsof_paths.get(rotation, []))),
                    "wof40_path": wof40_paths.get(rotation, ""),
                    "mask_path": mask_paths.get(rotation, ""),
                    "calibration_path_hint": f"cal_data/Angle_*_deg/ang_*_deg_cam_{cam:02d}.mat",
                    "calibration_mapping_status": "ROT-to-Angle mapping must be confirmed from scripts or paper before reconstruction",
                })

    archive_overview = []
    for archive in archives:
        archive_overview.append({
            "archive": archive,
            "file_count": sum(archive_counts[archive].values()),
            "uncompressed_bytes": sum(archive_bytes[archive].values()),
            "top_level_file_counts": dict(sorted(archive_counts[archive].items())),
        })

    missing_cal_pairs = []
    cal_angles = sorted(cal_grid)
    cal_cams = sorted(set().union(*cal_grid.values()) if cal_angles else [])
    for angle in cal_angles:
        for cam in cal_cams:
            if cam not in cal_grid[angle]:
                missing_cal_pairs.append({"angle_deg": angle, "cam": f"cam_{cam:02d}"})

    return {
        "generated_date": date.today().isoformat(),
        "index_file": str(INDEX_PATH.relative_to(ROOT)),
        "source_index_bytes": INDEX_PATH.stat().st_size,
        "archives": archives,
        "archive_count": len(archives),
        "file_entries": len(files),
        "directory_entries": len(dirs),
        "uncompressed_file_bytes_in_index": sum(item["bytes"] for item in files),
        "top_level": {
            "file_counts": dict(sorted(top_level_counts.items())),
            "uncompressed_bytes": dict(sorted(top_level_bytes.items())),
        },
        "extensions": {
            "file_counts": dict(ext_counts.most_common()),
            "uncompressed_bytes": dict(ext_bytes.most_common()),
        },
        "archive_overview": archive_overview,
        "calibration": {
            "angles_deg": cal_angles,
            "camera_ids": [f"cam_{cam:02d}" for cam in cal_cams],
            "angle_camera_pair_count": sum(len(cams) for cams in cal_grid.values()),
            "missing_angle_camera_pairs": missing_cal_pairs,
            "background_camera_ids": [f"cam_{cam:02d}" for cam in sorted(background_cal_cams)],
        },
        "flow_views": {
            "rotation_ids": all_rotations,
            "camera_ids": [f"cam_{cam:02d}" for cam in all_cams],
            "ref_def_pair_count": complete_ref_def_pairs,
            "paper_reported_view_count": 70,
            "inferred_view_count_from_ref_def_pairs": complete_ref_def_pairs,
            "interpretation": "The cached index shows 10 DEF/REF rotation groups x 7 cameras = 70 image-pair views; calibration files span 13 angle folders x 7 cameras.",
            "view_matrix": view_matrix,
        },
        "view_manifest": {
            "csv": str(VIEW_MANIFEST_CSV.relative_to(ROOT)),
            "row_count": len(view_manifest_rows),
            "rows": view_manifest_rows,
        },
        "processed_fields": {
            "cc_rotations": sorted(cc_rotations, key=int),
            "hsof_rotations": sorted(hsof_rotations, key=int),
            "wof40_rotations": sorted(wof40_rotations, key=int),
            "mask_rotations": sorted(mask_rotations, key=int),
        },
        "results_dirs": dict(results_dirs.most_common(20)),
        "sample_paths": dict(sorted(sample_paths.items())),
        "next_actions": [
            "Use the 10 x 7 flow-view matrix to build a loader skeleton without downloading the full 51.66 GB archive.",
            "Ask He Yuanzhe whether OERF nine-view BOST data should be represented as rotation/view groups or as physical camera IDs.",
            "If disk space allows, download only the zip slices that contain the first rotation group's REF/DEF images and processed HSOF/mask files.",
            "Keep the public benchmark separate from OERF internal data and replace geometry/unit fields before using lab data.",
        ],
    }


def camera_mark(row: dict, cam: str) -> str:
    has_ref_def = cam in row["ref_def_pair_cams"]
    if has_ref_def and row["has_hsof_mat"] and row["has_mask_proc"]:
        return "REF+DEF+PROC"
    if has_ref_def:
        return "REF+DEF"
    if cam in row["ref_cams"] or cam in row["def_cams"]:
        return "PARTIAL"
    return "-"


def write_markdown(summary: dict) -> None:
    flow = summary["flow_views"]
    cal = summary["calibration"]
    top_counts = summary["top_level"]["file_counts"]
    top_bytes = summary["top_level"]["uncompressed_bytes"]

    lines = [
        "# Open BOS 索引摘要与视角预演",
        "",
        f"生成日期：{summary['generated_date']}",
        "",
        "用途：把 Penn State Data Commons 的 255 KB 官方 zip 内容清单转成可讨论、可执行的数据入口。这里不包含 51.66 GB 全量数据。",
        "",
        "## 关键结论",
        "",
        f"- 官方清单共解析出 {summary['archive_count']} 个 zip、{summary['file_entries']} 个文件条目和 {summary['directory_entries']} 个目录条目。",
        f"- calibration 是 {len(cal['angles_deg'])} 个角度 x {len(cal['camera_ids'])} 台相机，共 {cal['angle_camera_pair_count']} 个 `.mat` 标定文件；这不是论文中 70 views 的直接计数。",
        f"- flow image-pair 视角是 {len(flow['rotation_ids'])} 个 `ROT_***` 旋转组 x {len(flow['camera_ids'])} 台相机，共 {flow['ref_def_pair_count']} 个 REF/DEF image-pair views，和论文 reported 70 views 对齐。",
        "- 对你的毕设最有用的抽象是：`rotation_id` + `cam_id` + `REF/DEF image pair` + `HSOF/CC/WOF deflection` + `mask` + `calibration`。",
        "",
        f"![Open BOS view grid](../figures/{VIEW_GRID_SVG.name})",
        "",
        "## 顶层目录规模",
        "",
        "| 目录 | 文件数 | 未压缩字节 | 为什么重要 |",
        "| --- | ---: | ---: | --- |",
    ]

    reasons = {
        "cal_data": "相机/视角标定，迁移到 OERF 时最需要替换",
        "def_data": "REF/DEF 原始图、CC/HSOF/WOF 位移结果，是 loader 的核心",
        "data": "处理后的 deflection/mask/volume 数据，适合先做小报告",
        "results": "论文结果和可视化，可用于 sanity check",
        "scripts": "MATLAB pipeline 线索，用来读变量名和处理流程",
        "tools": "辅助工具，适合只读结构",
        "cad": "飞行体几何，只能作为 open benchmark 几何",
        "cal_pkg": "标定包/图片，辅助理解几何",
        "pyscripts": "Python 工具线索",
        "batch": "批处理入口，帮助定位主流程",
        "readme.pdf": "官方说明文档，下载全量前先读",
    }
    for key, count in sorted(top_counts.items()):
        label = f"`{key}/`" if key in reasons and key != "readme.pdf" else f"`{key}`"
        lines.append(f"| {label} | {count} | {top_bytes.get(key, 0):,} | {reasons.get(key, '辅助文件')} |")

    lines.extend([
        "",
        "## 70 视角矩阵",
        "",
        "| ROT group | cam_01 | cam_02 | cam_03 | cam_04 | cam_05 | cam_06 | cam_07 | processed fields |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in flow["view_matrix"]:
        fields = []
        if row["has_cc_mat"]:
            fields.append("CC")
        if row["has_hsof_mat"]:
            fields.append("HSOF")
        if row["has_wof40_mat"]:
            fields.append("WOF40")
        if row["has_mask_proc"]:
            fields.append("MASK")
        marks = [camera_mark(row, f"cam_{cam:02d}") for cam in range(1, 8)]
        lines.append(f"| `ROT_{row['rotation_id']}` | " + " | ".join(marks) + f" | {', '.join(fields) or '-'} |")

    lines.extend([
        "",
        "## 建议的本科预演任务",
        "",
        "1. 只读 `open_bos_index_summary.json`，先生成一个 70-view manifest，不下载全量数据。",
        "2. 按 `ROT_000`、`ROT_010`、`ROT_020` 选 3 个旋转组，模拟 21-view 或 9-view 子采样。",
        "3. 给每个 view 记录 `ref_image_path`、`def_image_path`、`hsof_path`、`mask_path` 和 `calibration_path`，字段缺失就显式写 `missing`。",
        "4. 做第一版 `view_grid.png/svg` 和 `view_manifest.csv`，先证明数据组织没错。",
        "5. 再进入重投影误差、NeRIF-style neural field 或 PIV-BOST 补偿，不一开始就追完整 NIRT 复现。",
        "",
        "## 给何远哲的具体问题",
        "",
        "1. OERF 九视角 BOST 更接近 `ROT group x cam_id`，还是固定物理相机编号？",
        "2. 组内数据是否也能给到 REF/DEF 原始图，还是只有处理后的 displacement/deflection field？",
        "3. 如果先交付 data loader + view-quality report，是否比直接跑 NeRIF 更符合组内近期需求？",
        "4. 真实数据里是否存在与 `HSOF`、`CC`、`WOF40` 类似的多算法位移结果可做 baseline 对照？",
        "5. 标定文件能否公开字段名和单位，即使图像数据暂时不能公开？",
        "",
        "## 迁移边界",
        "",
        "- Open BOS 的物理对象是高速飞行体，不是火焰；可以迁移数据结构、少视角选择、重投影和报告工具，不能迁移物理结论。",
        "- `cal_data/Angle_*_deg` 是公开 benchmark 标定角度；迁移到 OERF 时要替换为组内 camera matrix / ray model。",
        "- 网页仓库只提交官方索引、摘要和小图，不提交 12 个 zip 数据包。",
    ])

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def write_manifest_csv(summary: dict) -> None:
    rows = summary["view_manifest"]["rows"]
    fieldnames = [
        "view_id",
        "rotation_id",
        "cam_id",
        "ref_image_path",
        "def_image_path",
        "cc_path",
        "hsof_paths",
        "wof40_path",
        "mask_path",
        "calibration_path_hint",
        "calibration_mapping_status",
    ]
    with VIEW_MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_svg(summary: dict) -> None:
    rows = summary["flow_views"]["view_matrix"]
    cams = summary["flow_views"]["camera_ids"]
    cell_w = 78
    cell_h = 32
    left = 128
    top = 88
    width = left + cell_w * len(cams) + 92
    height = top + cell_h * len(rows) + 116

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Open BOS 70-view grid</title>",
        "<desc id=\"desc\">Ten rotation groups by seven cameras with reference, deflected, processed field, and mask availability.</desc>",
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#1f2a2e}",
        ".title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#59676d}.hdr{font-size:12px;font-weight:700}.row{font-size:12px;font-weight:650}.tiny{font-size:11px;fill:#59676d}.celltxt{font-size:10px;font-weight:750;fill:#fff}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#f7faf9"/>',
        '<text x="32" y="38" class="title">Open BOS flow-view grid</text>',
        '<text x="32" y="60" class="sub">10 ROT groups x 7 cameras = 70 REF/DEF image-pair views; processed fields and masks are rotation-level assets.</text>',
    ]

    for c, cam in enumerate(cams):
        x = left + c * cell_w
        parts.append(f'<text x="{x + cell_w / 2}" y="{top - 18}" class="hdr" text-anchor="middle">{cam}</text>')

    for r, row in enumerate(rows):
        y = top + r * cell_h
        parts.append(f'<text x="{left - 18}" y="{y + 21}" class="row" text-anchor="end">ROT_{row["rotation_id"]}</text>')
        for c, cam in enumerate(cams):
            x = left + c * cell_w
            status = camera_mark(row, cam)
            if status == "REF+DEF+PROC":
                fill = "#2f9e73"
                text = "R+D"
            elif status == "REF+DEF":
                fill = "#6cbf8f"
                text = "R+D"
            elif status == "PARTIAL":
                fill = "#f1b565"
                text = "partial"
            else:
                fill = "#d7dee2"
                text = "-"
            parts.append(f'<rect x="{x + 4}" y="{y + 4}" width="{cell_w - 8}" height="{cell_h - 8}" rx="6" fill="{fill}"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 23}" class="celltxt" text-anchor="middle">{text}</text>')

    legend_y = top + cell_h * len(rows) + 34
    legend = [
        ("#2f9e73", "REF+DEF + processed field/mask"),
        ("#6cbf8f", "REF+DEF image pair"),
        ("#f1b565", "partial"),
    ]
    for i, (color, label) in enumerate(legend):
        x = 32
        y = legend_y + i * 22
        parts.append(f'<rect x="{x}" y="{y - 13}" width="18" height="18" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 26}" y="{y + 1}" class="tiny">{label}</text>')

    parts.append("</svg>")
    VIEW_GRID_SVG.write_text("\n".join(parts) + "\n")


def main() -> int:
    files, dirs, archives = parse_entries()
    summary = collect_summary(files, dirs, archives)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_manifest_csv(summary)
    write_markdown(summary)
    write_svg(summary)
    print(f"Wrote {SUMMARY_JSON.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_MD.relative_to(ROOT)}")
    print(f"Wrote {VIEW_MANIFEST_CSV.relative_to(ROOT)}")
    print(f"Wrote {VIEW_GRID_SVG.relative_to(ROOT)}")
    print(f"views: {summary['flow_views']['ref_def_pair_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
