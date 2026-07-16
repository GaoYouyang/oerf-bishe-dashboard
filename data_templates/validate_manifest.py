#!/usr/bin/env python3
"""Validate OERF BOST/PIV-BOST manifest templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "BOST": ["dataset_id", "modality", "owner", "status", "experiment", "views", "raw", "geometry", "displacement", "permissions"],
    "PIV-BOST": ["dataset_id", "modality", "owner", "status", "bost", "piv", "synchronization", "permissions"],
    "BOS/TBOS": ["dataset_id", "modality", "source", "purpose", "expected_public_structure", "local_paths", "view_plan", "fields_to_extract", "metrics", "migration_to_oerf", "permissions"],
}


PATH_KEYS = {
    "mask",
    "camera_intrinsics",
    "view_geometry",
    "voxel_reconstruction",
    "nerif_reconstruction",
    "manifest",
    "refractive_index_field",
    "gradient_field",
    "frame_a",
    "frame_b",
    "raw_velocity_field",
    "corrected_velocity_field",
    "timestamps"
}


def walk(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from walk(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def is_path_field(path: str) -> bool:
    last = path.replace("]", "").split(".")[-1]
    return last in PATH_KEYS or last.endswith("_fields") or last in {"flow_off", "flow_on", "fields", "reference_slices", "image_pairs"}


def collect_paths(obj: Any) -> list[str]:
    paths: list[str] = []
    for key_path, value in walk(obj):
        if value is None:
            continue
        if isinstance(value, str) and ("/" in value or value.endswith((".json", ".npy", ".png", ".csv", ".tif", ".tiff"))):
            paths.append(value)
    return paths


def validate(manifest_path: Path, allow_missing: bool) -> int:
    data = json.loads(manifest_path.read_text())
    modality = data.get("modality")
    if modality not in REQUIRED_TOP_LEVEL:
        print(f"ERROR: unsupported modality {modality!r}")
        return 2

    errors = []
    warnings = []
    for key in REQUIRED_TOP_LEVEL[modality]:
        if key not in data:
            errors.append(f"missing top-level key: {key}")

    for key_path, value in walk(data):
        if value == "unknown" or value is None:
            warnings.append(f"unfilled field: {key_path}")

    if not allow_missing:
        base = manifest_path.parent
        for rel in collect_paths(data):
            if rel.startswith("unknown"):
                continue
            if not (base / rel).exists():
                errors.append(f"missing file path: {rel}")

    if warnings:
        print("WARNINGS")
        for item in warnings:
            print(f"- {item}")

    if errors:
        print("ERRORS")
        for item in errors:
            print(f"- {item}")
        return 1

    print(f"OK: {manifest_path} ({modality})")
    if allow_missing and warnings:
        print("Template fields are still unfilled, but --allow-missing accepted them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-missing", action="store_true", help="do not fail on missing file paths; useful for templates")
    args = parser.parse_args()
    return validate(args.manifest, args.allow_missing)


if __name__ == "__main__":
    raise SystemExit(main())
