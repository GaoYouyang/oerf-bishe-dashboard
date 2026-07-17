#!/usr/bin/env python3
"""Export the minimal tracked parent-metric snapshot required by Gate B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPLICATES = (0, 8)
FAMILIES = (
    "plume",
    "wavy_front",
    "thin_front",
    "double_front",
    "annular_kernel",
    "oblique_shock",
    "vortex_pair",
    "multi_plume",
)
CHECKPOINTS = (4, 8, 16, 32)
CANDIDATE_PREFIXES = ("pdhg_data_only_k", "graph_s3_k")
FIELDS = (
    "replicate",
    "sample_index",
    "reaction_family",
    "candidate_id",
    "field_relative_l2",
    "gradient_relative_l2",
    "front_top10_f1",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_snapshot_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    selected: dict[tuple[int, int, str, int], dict[str, str]] = {}
    for source in rows:
        identifier = str(source["candidate_id"])
        prefix = next(
            (candidate for candidate in CANDIDATE_PREFIXES if identifier.startswith(candidate)),
            None,
        )
        if prefix is None:
            continue
        suffix = identifier.removeprefix(prefix)
        if not suffix.isdigit() or int(suffix) not in CHECKPOINTS:
            continue
        replicate = int(source["replicate"])
        sample = int(source["sample_index"])
        iteration = int(suffix)
        key = (replicate, sample, prefix, iteration)
        if key in selected:
            raise ValueError(f"duplicate parent snapshot row: {key}")
        if replicate not in REPLICATES or not 0 <= sample < len(FAMILIES):
            raise ValueError(f"unexpected parent snapshot coordinate: {key}")
        if str(source["reaction_family"]) != FAMILIES[sample]:
            raise ValueError("parent snapshot reaction-family label changed")
        output = {field: str(source[field]) for field in FIELDS}
        for metric in ("field_relative_l2", "gradient_relative_l2", "front_top10_f1"):
            value = float(output[metric])
            if not 0.0 <= value < float("inf"):
                raise ValueError(f"parent snapshot metric is invalid: {metric}")
        selected[key] = output
    expected = {
        (replicate, sample, prefix, iteration)
        for replicate in REPLICATES
        for sample in range(len(FAMILIES))
        for prefix in CANDIDATE_PREFIXES
        for iteration in CHECKPOINTS
    }
    if set(selected) != expected:
        raise ValueError("parent snapshot coverage is incomplete or unexpected")
    return [selected[key] for key in sorted(selected)]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_snapshot(
    *,
    source_rows: Path,
    public_summary: Path,
    output: Path,
) -> dict[str, Any]:
    public = json.loads(public_summary.read_text(encoding="utf-8"))
    source_hash = file_sha256(source_rows)
    declared = public["integrity"]["private_artifact_sha256_by_opaque_role"][
        "metric_rows"
    ]
    if source_hash != declared:
        raise ValueError("private parent rows do not match the tracked public audit hash")
    with source_rows.open(newline="", encoding="utf-8") as handle:
        rows = extract_snapshot_rows(list(csv.DictReader(handle)))
    output.mkdir(parents=True, exist_ok=True)
    metric_path = output / "metric_rows.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "psu-b0-gate-b-parent-metric-snapshot-1.0",
        "status": "TRACKED_MINIMAL_SYNTHETIC_PARENT_SNAPSHOT",
        "row_count": len(rows),
        "columns": list(FIELDS),
        "replicates": list(REPLICATES),
        "reaction_families": list(FAMILIES),
        "checkpoints": list(CHECKPOINTS),
        "candidate_prefixes": list(CANDIDATE_PREFIXES),
        "source_private_metric_rows_sha256": source_hash,
        "source_public_summary_repository_path": public_summary.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "source_public_summary_sha256": file_sha256(public_summary),
        "snapshot_metric_rows_sha256": file_sha256(metric_path),
        "contains_experimental_flow_truth": False,
        "contains_credentials_or_private_paths": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
    (output / "checksums.sha256").write_text(
        f"{file_sha256(manifest_path)}  manifest.json\n"
        f"{file_sha256(metric_path)}  metric_rows.csv\n",
        encoding="ascii",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument(
        "--public-summary",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/results/psu_b0_pdhg_scale_smoke_v2_public/summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "demo_t16_operator/results/psu_b0_factor_pdhg_gate_b_parent_snapshot",
    )
    args = parser.parse_args()
    manifest = build_snapshot(
        source_rows=args.source_rows.resolve(),
        public_summary=args.public_summary.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
