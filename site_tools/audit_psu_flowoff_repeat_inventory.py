#!/usr/bin/env python3
"""Audit whether the public PSU BOS archive exposes temporal repeat frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


ARCHIVE_PATTERN = re.compile(r"^Archive:\s+(\S+)")
FILE_PATTERN = re.compile(
    r"^\s*\d+\s+\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s+(.+)$"
)
DEFLECTED_PATTERN = re.compile(
    r"/def_data/DEF_IMGS/DEF_ROT_(\d+)/"
    r"DEF_ROT_\1_CAM_(\d+)\.tiff$",
    re.IGNORECASE,
)
RAW_CALIBRATION_PATTERN = re.compile(
    r"/cal_pkg/Data/raw_images/withoutCylinder/Cam(\d+)/(.+\.tiff)$",
    re.IGNORECASE,
)
PROCESSED_CALIBRATION_PATTERN = re.compile(
    r"/cal_pkg/Data/processed_images/withoutCylinder/Cam(\d+)/(.+\.jpg)$",
    re.IGNORECASE,
)
PROCESSED_DEFLECTION_PATTERN = re.compile(
    r"/data/DEF_PROC/HSOF_DEF_ROT_(\d+)\.mat$",
    re.IGNORECASE,
)


def summarize_archive_index(text: str) -> dict[str, Any]:
    """Summarize distinct image conditions without treating angles as repeats."""

    archive: str | None = None
    archives: set[str] = set()
    file_count = 0
    deflected_conditions: dict[tuple[str, str], list[str]] = {}
    raw_calibration_conditions: dict[tuple[str, str], list[str]] = {}
    processed_calibration_conditions: dict[
        tuple[str, str], list[str]
    ] = {}
    processed_deflections: list[str] = []
    cam_angles_all_files: list[str] = []
    for line in str(text).splitlines():
        archive_match = ARCHIVE_PATTERN.match(line)
        if archive_match:
            archive = archive_match.group(1)
            archives.add(archive)
            continue
        file_match = FILE_PATTERN.match(line)
        if not file_match:
            continue
        path = file_match.group(1)
        if path.endswith("/"):
            continue
        file_count += 1
        deflected = DEFLECTED_PATTERN.search(path)
        if deflected:
            key = (deflected.group(1), deflected.group(2))
            deflected_conditions.setdefault(key, []).append(path)
        raw = RAW_CALIBRATION_PATTERN.search(path)
        if raw:
            key = (raw.group(1), raw.group(2))
            raw_calibration_conditions.setdefault(key, []).append(path)
        processed = PROCESSED_CALIBRATION_PATTERN.search(path)
        if processed:
            key = (processed.group(1), processed.group(2))
            processed_calibration_conditions.setdefault(key, []).append(path)
        if PROCESSED_DEFLECTION_PATTERN.search(path):
            processed_deflections.append(path)
        if path.endswith("/def_data/CamAnglesAll.mat"):
            cam_angles_all_files.append(path)

    def duplicate_count(
        conditions: dict[tuple[str, str], list[str]],
    ) -> int:
        return sum(len(paths) > 1 for paths in conditions.values())

    def maximum_count(
        conditions: dict[tuple[str, str], list[str]],
    ) -> int:
        return max((len(paths) for paths in conditions.values()), default=0)

    return {
        "schema_version": "psu-flowoff-repeat-inventory-1.0",
        "archive_count": len(archives),
        "listed_file_count": file_count,
        "deflected_flow_on_tiff": {
            "file_count": sum(map(len, deflected_conditions.values())),
            "unique_camera_rotation_conditions": len(deflected_conditions),
            "conditions_with_more_than_one_file": duplicate_count(
                deflected_conditions
            ),
            "maximum_files_per_camera_rotation_condition": maximum_count(
                deflected_conditions
            ),
        },
        "calibration_without_cylinder": {
            "raw_tiff_file_count": sum(
                map(len, raw_calibration_conditions.values())
            ),
            "raw_unique_named_conditions": len(raw_calibration_conditions),
            "raw_conditions_with_more_than_one_file": duplicate_count(
                raw_calibration_conditions
            ),
            "processed_jpg_file_count": sum(
                map(len, processed_calibration_conditions.values())
            ),
            "processed_unique_named_conditions": len(
                processed_calibration_conditions
            ),
        },
        "processed_hsof_rotation_files": len(processed_deflections),
        "cam_angles_all_container_files": len(cam_angles_all_files),
        "temporal_repeat_assessment": {
            "independent_temporal_flowoff_frames_available_per_condition": 0,
            "independent_temporal_flowon_frames_available_per_condition": 0,
            "averaged_flowon_images_listed_per_camera_rotation": 1,
            "public_archive_authorizes_temporal_covariance_estimation": False,
            "reason": (
                "The index exposes one averaged flow-on TIFF per camera and "
                "rotation plus composite reference/deflected containers. "
                "Calibration target angles are distinct geometries, not "
                "repeated flow-off samples at one fixed condition."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-index",
        type=Path,
        default=Path("data_templates/open_bos_zip_file_content.txt"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_archive_index(
        args.archive_index.read_text(encoding="utf-8", errors="replace")
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
