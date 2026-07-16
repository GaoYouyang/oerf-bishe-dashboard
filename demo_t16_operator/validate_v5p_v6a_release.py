#!/usr/bin/env python3
"""Independently validate the committed V5P-V6A evidence release.

Passing this validator means that files, provenance links, decisions, and
reported V6A aggregates are internally consistent. It does not turn a failed
development gate into a scientific success claim.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from release_provenance import file_sha256


OPERATOR_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OPERATOR_ROOT.parent
MANIFEST_PATH = OPERATOR_ROOT / "results" / "v5p_v6a_release_checksums.sha256"
RESULT_DIRECTORIES = (
    "v5p_fresh_budget_gate",
    "v5q_postopen_topology_diagnosis",
    "v5r_reserved_view_reliability_diagnosis",
    "v5s_dco_low_rank_screening",
    "v5t_camera_local_tangent_diagnosis",
    "v5u_calibrated_renderer_residual_screening",
    "v5v_camera_local_kernel_correction",
    "v5w_clean_aperture_kernel_screening",
    "v5x_ray_conditioned_voxel_kernel",
    "v5y_direct_ray_conditioned_kernel",
    "v5z_stabilized_direct_ray_kernel",
    "v6a_ray_kernel_hypernetwork_development",
)
CHECKPOINT_PATHS = {
    str(seed): OPERATOR_ROOT
    / "results"
    / "v5k_shared_field_work"
    / "sf_rio_adjoint_only"
    / str(seed)
    / "best.pt"
    for seed in (3101, 3102, 3103)
}
SHA256 = re.compile(r"[0-9a-f]{64}")


class ValidationError(AssertionError):
    """Raised when release evidence is missing or inconsistent."""


class Validator:
    def __init__(self) -> None:
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise ValidationError(message)

    def close(
        self,
        actual: Any,
        expected: Any,
        message: str,
        *,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-11,
    ) -> None:
        left = float(actual)
        right = float(expected)
        self.require(
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol),
            f"{message}: {left} != {right}",
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValidationError("cannot average empty values")
    return math.fsum(items) / len(items)


def validate_root_manifest(validator: Validator) -> set[str]:
    validator.require(MANIFEST_PATH.is_file(), "missing release checksum manifest")
    recorded: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        validator.require(len(parts) == 2, "malformed release checksum line")
        digest, raw_name = parts
        path_token = PurePosixPath(raw_name)
        validator.require(bool(SHA256.fullmatch(digest)), f"invalid digest: {raw_name}")
        validator.require(not path_token.is_absolute(), f"absolute release path: {raw_name}")
        validator.require(".." not in path_token.parts, f"unsafe release path: {raw_name}")
        validator.require(raw_name not in recorded, f"duplicate release path: {raw_name}")
        lowered = {part.lower() for part in path_token.parts}
        validator.require(
            not ({"private_library", "tmp_downloads"} & lowered),
            f"private path in public release: {raw_name}",
        )
        validator.require(path_token.suffix.lower() != ".pdf", f"PDF in public release: {raw_name}")
        path = REPOSITORY_ROOT / path_token
        validator.require(path.is_file(), f"missing release file: {raw_name}")
        validator.require(file_sha256(path) == digest, f"release hash mismatch: {raw_name}")
        recorded[raw_name] = digest

    required = {
        "general_operator_research_lab.html",
        "demo_t16_operator/validate_v5p_v6a_release.py",
        "demo_t16_operator/results/operator_structure_funnel_v5s_v6a.png",
        *{
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in CHECKPOINT_PATHS.values()
        },
    }
    validator.require(required <= set(recorded), "release manifest misses a core file")
    return set(recorded)


def validate_directory_checksums(validator: Validator) -> None:
    for directory_name in RESULT_DIRECTORIES:
        directory = OPERATOR_ROOT / "results" / directory_name
        checksum_path = directory / "checksums.sha256"
        validator.require(checksum_path.is_file(), f"missing checksum: {directory_name}")
        for line in checksum_path.read_text(encoding="ascii").splitlines():
            digest, name = line.split("  ", 1)
            validator.require(Path(name).name == name, f"unsafe local checksum: {name}")
            target = directory / name
            validator.require(target.is_file(), f"missing local artifact: {target}")
            validator.require(file_sha256(target) == digest, f"local hash mismatch: {target}")


def validate_provenance(validator: Validator) -> None:
    for directory_name in RESULT_DIRECTORIES:
        report_path = OPERATOR_ROOT / "results" / directory_name / "report.json"
        report = load_json(report_path)
        provenance = report.get("source_provenance", {})
        direct = provenance.get("direct_dependency_sha256")
        if directory_name in {
            "v5p_fresh_budget_gate",
            "v5q_postopen_topology_diagnosis",
            "v5r_reserved_view_reliability_diagnosis",
            "v5y_direct_ray_conditioned_kernel",
            "v5z_stabilized_direct_ray_kernel",
            "v6a_ray_kernel_hypernetwork_development",
        }:
            validator.require(isinstance(direct, dict) and direct, f"missing provenance: {directory_name}")
        if not isinstance(direct, dict):
            continue
        for raw_name, digest in direct.items():
            validator.require(bool(SHA256.fullmatch(str(digest))), f"invalid provenance digest: {raw_name}")
            path = OPERATOR_ROOT / PurePosixPath(raw_name)
            validator.require(path.is_file(), f"missing provenance file: {raw_name}")
            validator.require(file_sha256(path) == digest, f"provenance mismatch: {raw_name}")


def validate_v5p_replays(validator: Validator) -> None:
    reports = {
        name: load_json(OPERATOR_ROOT / "results" / name / "report.json")
        for name in RESULT_DIRECTORIES[:3]
    }
    v5p = reports["v5p_fresh_budget_gate"]
    validator.require(v5p["decision"] == "FRESH_DEVELOPMENT_NO_GO", "V5P decision drift")
    for seed, path in CHECKPOINT_PATHS.items():
        validator.require(file_sha256(path) == v5p["checkpoint_hashes"][seed], f"checkpoint drift: {seed}")
    frozen_hash = v5p["prediction_sha256_before_scoring"]
    for name in ("v5q_postopen_topology_diagnosis", "v5r_reserved_view_reliability_diagnosis"):
        report = reports[name]
        validator.require(report["v5p_prediction_sha256"] == frozen_hash, f"V5P link drift: {name}")
        validator.require(report["reproduced_prediction_sha256"] == frozen_hash, f"V5P replay drift: {name}")


def validate_v6a_aggregates(validator: Validator) -> None:
    directory = OPERATOR_ROOT / "results" / "v6a_ray_kernel_hypernetwork_development"
    report = load_json(directory / "report.json")
    rows = load_csv(directory / "development_rig_metrics.csv")
    by_method: dict[str, dict[str, float]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["rig_id"]] = float(
            row["relative_aperture_residual_error"]
        )
    baseline = by_method["full_matrix_geometry_ridge"]
    candidate = by_method["hypernetwork_ensemble"]
    validator.require(set(candidate) == set(baseline), "V6A paired rig set drift")
    ratios = [candidate[rig] / baseline[rig] for rig in sorted(baseline)]
    improvement = 1.0 - mean(candidate.values()) / mean(baseline.values())
    validator.close(improvement, report["ensemble_improvement_over_full_matrix_ridge"], "V6A improvement")
    validator.close(max(ratios) - 1.0, report["ensemble_maximum_paired_rig_degradation_vs_full_matrix_ridge"], "V6A max paired degradation")
    validator.close(mean(ratio > 1.0 for ratio in ratios), report["ensemble_paired_rig_harm_fraction_vs_full_matrix_ridge"], "V6A harm fraction")
    validator.close(mean(ratio < 1.0 for ratio in ratios), report["ensemble_positive_rig_fraction_vs_full_matrix_ridge"], "V6A positive fraction")
    validator.require(
        report["decision"]
        == "RAY_KERNEL_HYPERNET_DEVELOPMENT_NO_GO_STOP_CAPACITY_ESCALATION",
        "V6A decision must remain NO-GO",
    )


def main() -> None:
    validator = Validator()
    files = validate_root_manifest(validator)
    validate_directory_checksums(validator)
    validate_provenance(validator)
    validate_v5p_replays(validator)
    validate_v6a_aggregates(validator)
    print(
        json.dumps(
            {
                "status": "PASS_INTERNAL_CONSISTENCY_ONLY",
                "checks": validator.checks,
                "release_files": len(files),
                "scientific_gate": "V6A_NO_GO",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
