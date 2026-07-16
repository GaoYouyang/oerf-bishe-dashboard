#!/usr/bin/env python3
"""Export an aggregate-only PSU loader summary safe for the public site."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def build_public_summary(
    loader: dict[str, Any], preflight: dict[str, Any], *, source_sha256: str
) -> dict[str, Any]:
    if loader.get("status") != "LOADER_NUMERIC_CONTRACT_CONFORMANT":
        raise ValueError("loader contract is not conformant")
    if preflight.get("status") != "FULL_AUTHOR_NIRT_NO_GO_CURRENT_ENVIRONMENT":
        raise ValueError("preflight status changed; review before public export")
    if not SHA256_PATTERN.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be a lowercase 64-character digest")
    if not all(loader.get("checks", {}).values()):
        raise ValueError("not all loader checks passed")

    coordinates = [
        {
            key: item[key]
            for key in (
                "field",
                "axis",
                "dimension",
                "lower_cell_center_m",
                "upper_cell_center_m",
                "spacing_m",
                "inferred_cell_centered_extent_m",
                "separable_on_landmarks",
                "centered",
                "extent_matches_official",
            )
        }
        for item in loader["coordinate_contract"]
    ]
    dependencies = {
        name: {
            "importable": bool(status["importable"]),
            **({"version": status["version"]} if status.get("importable") else {}),
        }
        for name, status in preflight["dependencies_available"].items()
    }
    return {
        "schema_version": "psu-hsof-public-loader-summary-1.0",
        "status": "LOADER_NUMERIC_CONTRACT_CONFORMANT",
        "evidence_scope": "AGGREGATE_SELECTED_NUMERIC_LOADER_CONTRACT_NO_RECONSTRUCTION",
        "source": {
            "dataset_doi": "https://doi.org/10.26208/1VE2-5C19",
            "file_label": loader["source_snapshot"]["file"],
            "file_size_bytes": loader["source_snapshot"]["file_size_bytes"],
            "sha256": source_sha256,
        },
        "configuration": loader["configuration"],
        "numeric_contract_checks": loader["checks"],
        "coordinate_contract": coordinates,
        "ray_contract": loader["ray_contract"],
        "official_nirt_preflight": {
            "status": preflight["status"],
            "dependencies": dependencies,
            "blocker_count": preflight["blocker_count"],
            "blocker_codes": [
                item["code"]
                for item in preflight["static_hazards"]
                if item["severity"] == "blocker"
            ],
            "known_persistent_floor_bytes": preflight["memory_floor"][
                "known_persistent_floor_bytes"
            ],
            "known_persistent_floor_gib": preflight["memory_floor"][
                "known_persistent_floor_gib"
            ],
            "host_physical_memory_bytes": preflight["memory_floor"].get(
                "host_physical_memory_bytes"
            ),
            "safe_next_gate": preflight["decision"]["safe_next_gate"],
        },
        "reader_memory_probe": {
            "variable": "X",
            "numeric_payload_bytes": 392000000,
            "maximum_resident_set_size_bytes": 52838400,
            "scope": "ONE_LOCAL_CACHED_RUN_MEMORY_EVIDENCE_NOT_A_SPEED_BENCHMARK",
        },
        "public_export_policy": {
            "contains_raw_arrays": False,
            "contains_scalar_samples": False,
            "contains_private_paths": False,
            "contains_author_source_copy": False,
        },
        "limitations": [
            "deterministic landmarks and measurement rows are aggregate loader evidence, not full-array distribution statistics",
            "deflection arrays and masks are not yet numerically audited",
            "the official NIRT entrypoint was not executed",
            "there is no held-out reprojection, 3-D truth or algorithm superiority result",
        ],
        "generated_by": [
            "site_tools/psu_bost_mat_sample.py",
            "site_tools/psu_bost_loader_conformance.py",
            "site_tools/official_nirt_preflight.py",
            "site_tools/build_psu_public_summary.py",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader-contract", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loader = json.loads(args.loader_contract.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    report = build_public_summary(
        loader, preflight, source_sha256=args.source_sha256
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote public aggregate summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
