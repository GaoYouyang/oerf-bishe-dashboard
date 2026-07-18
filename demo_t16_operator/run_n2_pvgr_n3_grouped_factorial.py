#!/usr/bin/env python3
"""Run the preregistered 96-cell N2-PVGR grouped development factorial.

The field seed, not a ray or geometry cell, is the statistical repeat. This
runner never opens a reserved phantom and never authorizes reconstruction,
real-data, neural-operator, novelty, or paper claims.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import run_n2_pvgr_n2_operator_consistent_bridge as bridge
except ImportError:
    import run_n2_pvgr_n2_operator_consistent_bridge as bridge


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n3_grouped_factorial_preregistered_v1.json"
)

METHODS = (
    "continuous_affine_n1",
    "operator_consistent_homotopy",
    "picard_1",
    "picard_2",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
    )


def _git_text(*args: str) -> str:
    return _git(*args).stdout.decode("utf-8").strip()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_staging_dir(work_dir: Path) -> Path:
    staging = work_dir / "final_artifacts_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def _geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(float(value) for value in values), dtype=float)
    if array.size == 0 or np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("geometric mean requires finite nonnegative values")
    return float(np.exp(np.mean(np.log(np.maximum(array, 1e-30)))))


def expand_factorial_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    factorial = config["factorial"]
    cases: list[dict[str, Any]] = []
    for family in factorial["field_families"]:
        for seed in family["field_seeds"]:
            field_unit_id = f"{family['id']}-s{int(seed)}"
            for orientation in factorial["rig_orientation_packages"]:
                for aperture in factorial["aperture_packages"]:
                    case_id = f"{field_unit_id}-{orientation['id']}-{aperture['id']}"
                    rig = {
                        key: value for key, value in orientation.items() if key != "id"
                    }
                    rig.update(
                        {key: value for key, value in aperture.items() if key != "id"}
                    )
                    cases.append(
                        {
                            "id": case_id,
                            "field_unit_id": field_unit_id,
                            "family_id": str(family["id"]),
                            "phantom_family": str(family["phantom_family"]),
                            "phantom_seed": int(seed),
                            "orientation_id": str(orientation["id"]),
                            "aperture_id": str(aperture["id"]),
                            "rig": rig,
                        }
                    )
    return cases


def _validate_contract(config: dict[str, Any], source: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n3-grouped-factorial-preregistered-1.0":
        raise ValueError("schema drifted from the preregistration")
    if config.get("candidate_id") != "N2-PVGR-N3-GF96":
        raise ValueError("candidate identifier drifted from the preregistration")
    if config.get("status") != (
        "preregistered_development_only_no_audit_authorization"
    ):
        raise ValueError("status is not the frozen preregistered development status")
    if config.get("device") != "cpu" or config.get("dtype") != "float64":
        raise ValueError("formal grouped factorial is frozen to CPU float64")
    if source.get("device") != "cpu" or source.get("dtype") != "float64":
        raise ValueError("source config is not CPU float64")
    if int(config["population_count"]) != 256:
        raise ValueError("formal grouped factorial requires 256 rays per cell")
    steps = (
        int(config["execution_step_count"]),
        int(config["reference_step_count"]),
        int(config["reference_sentinel_step_count"]),
    )
    if steps != (128, 256, 512):
        raise ValueError("formal grouped factorial requires 128/256/512 steps")
    if tuple(int(value) for value in config["picard_sweep_counts"]) != (1, 2):
        raise ValueError("formal grouped factorial requires Picard-1/2")
    if config.get("teacher_scope") != "all_96_physical_cells":
        raise ValueError("teacher scope drifted")
    if config.get("reference_sentinel_scope") != "all_96_physical_cells":
        raise ValueError("reference-sentinel scope drifted")
    if config.get("resume_policy") != (
        "only_hash_validated_cell_checkpoints_from_the_same_preregistration"
    ):
        raise ValueError("resume policy drifted")

    factorial = config["factorial"]
    expected_families = (
        ("smooth", "smooth_plume", (1729, 1871, 1987, 2131)),
        (
            "wrinkled",
            "wrinkled_density_interface",
            (2718, 2851, 3001, 3163),
        ),
    )
    actual_families = tuple(
        (
            str(item["id"]),
            str(item["phantom_family"]),
            tuple(int(seed) for seed in item["field_seeds"]),
        )
        for item in factorial["field_families"]
    )
    if actual_families != expected_families:
        raise ValueError("field-family or field-seed contract drifted")
    expected_orientations = (
        ("orientation_22", 22.0, 0.05, -0.04, 0.62, 0.0),
        ("orientation_58", 58.0, -0.07, 0.06, 0.62, 0.0),
    )
    actual_orientations = tuple(
        (
            str(item["id"]),
            float(item["view_angle_degrees"]),
            float(item["detector_u"]),
            float(item["detector_z"]),
            float(item["path_half_length"]),
            float(item["bend"]),
        )
        for item in factorial["rig_orientation_packages"]
    )
    if actual_orientations != expected_orientations:
        raise ValueError("orientation package contract drifted")
    expected_apertures = (
        ("narrow", 0.035, 0.04, 0.03),
        ("wide", 0.075, 0.07, 0.05),
    )
    actual_apertures = tuple(
        (
            str(item["id"]),
            float(item["aperture_radius"]),
            float(item["cone_u"]),
            float(item["cone_z"]),
        )
        for item in factorial["aperture_packages"]
    )
    if actual_apertures != expected_apertures:
        raise ValueError("aperture package contract drifted")
    if factorial.get("field_unit_definition") != "phantom_family_x_field_seed":
        raise ValueError("field-unit definition drifted")
    if factorial.get("ray_is_not_an_independent_repeat") is not True:
        raise ValueError("ray independence declaration drifted")
    if factorial.get("common_sobol_prefix_by_field_unit") is not True:
        raise ValueError("common Sobol prefix contract drifted")

    reserved = tuple(
        str(value) for value in config["reserved_audit_families_not_opened"]
    )
    if reserved != tuple(source["reserved_audit_families_not_opened"]):
        raise ValueError("reserved family contract drifted")
    opened = {
        str(item["phantom_family"]) for item in config["factorial"]["field_families"]
    }
    if opened.intersection(reserved):
        raise ValueError("a reserved family was opened in development")
    if opened != {"smooth_plume", "wrinkled_density_interface"}:
        raise ValueError("opened family set drifted from the preregistration")

    cases = expand_factorial_cases(config)
    expected_base = int(factorial["expected_base_case_count"])
    stresses = tuple(factorial["dimensionless_stress_scale_multipliers"])
    expected_cells = int(factorial["expected_physical_cell_count"])
    field_units = {str(case["field_unit_id"]) for case in cases}
    if len(cases) != expected_base or expected_base != 32:
        raise ValueError("base-case count is not exactly 32")
    if len(cases) * len(stresses) != expected_cells or expected_cells != 96:
        raise ValueError("physical-cell count is not exactly 96")
    if len(field_units) != int(factorial["expected_field_unit_count"]):
        raise ValueError("field-unit count drifted")
    if len(field_units) != 8 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("field-unit or case identifiers are not unique")
    if tuple(float(value) for value in stresses) != (1.0, 3.0, 10.0):
        raise ValueError("stress levels drifted")
    if not all(value is False for value in config["claim_authorizations"].values()):
        raise ValueError("all broad claim authorizations must remain false")
    if int(config["grouped_decision"]["bootstrap_replicates"]) != 20000:
        raise ValueError("bootstrap replicate count drifted")
    expected_unimplemented = (
        "ray_crossing_or_caustic_certificate",
        "discontinuous_cell_topology_certificate",
        "peak_rss_by_method",
        "host_scalar_synchronization_count",
        "independent_process_evaluator_isolation",
    )
    if tuple(config["known_unimplemented_integrity_gates"]) != expected_unimplemented:
        raise ValueError("known-unimplemented integrity gate list drifted")


def _validate_preregistration(
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    if not attestation_path.is_file():
        raise FileNotFoundError("committed preregistration attestation is missing")
    attestation = _read_json(attestation_path)
    if attestation.get("formal_results_absent_at_creation") is not True:
        raise ValueError("attestation does not prove formal output was absent")
    if attestation.get("formal_work_output_absent_at_creation") is not True:
        raise ValueError("attestation does not prove formal checkpoints were absent")
    if attestation.get("formal_output") != config["formal_output"]:
        raise ValueError("attested formal output path drifted")
    if attestation.get("config_sha256") != _sha256(config_path):
        raise ValueError("current config does not match the attestation")

    protocol_commit = str(attestation["protocol_commit"])
    ancestry = _git("merge-base", "--is-ancestor", protocol_commit, "HEAD", check=False)
    if ancestry.returncode != 0:
        raise ValueError("protocol commit is not an ancestor of the run HEAD")
    if set(attestation["attested_files"]) != set(config["attested_files"]):
        raise ValueError("attested file key set drifted")
    for key, entry in attestation["attested_files"].items():
        expected_path = str(config["attested_files"][key])
        if entry["path"] != expected_path:
            raise ValueError(f"attested path drifted for {key}")
        current_path = _resolve(expected_path)
        if _sha256(current_path) != entry["sha256"]:
            raise ValueError(f"current file hash drifted for {key}")
        frozen = _git("show", f"{protocol_commit}:{expected_path}").stdout
        if hashlib.sha256(frozen).hexdigest() != entry["sha256"]:
            raise ValueError(f"protocol commit hash drifted for {key}")

    tracked = _git(
        "ls-files", "--error-unmatch", _relative(attestation_path), check=False
    )
    if tracked.returncode != 0:
        raise ValueError("attestation is not committed")
    relevant_paths = [
        _relative(attestation_path),
        *(str(value) for value in config["attested_files"].values()),
    ]
    status = _git("status", "--porcelain", "--", *relevant_paths).stdout
    if status.strip():
        raise ValueError("preregistered files or attestation have uncommitted changes")
    return attestation


def _bridge_config(config: dict[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(config)
    adapted["timing_stress_multiplier"] = float(config["timing"]["stress_multiplier"])
    return adapted


def _source_for_run(
    source: dict[str, Any],
    config: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    adapted = copy.deepcopy(source)
    adapted["population_count"] = int(config["population_count"])
    adapted["development_cases"] = cases
    adapted["dimensionless_stress_scale_multipliers"] = list(
        config["factorial"]["dimensionless_stress_scale_multipliers"]
    )
    return adapted


def _tensor_sha256(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _execution_case(case: dict[str, Any]) -> dict[str, Any]:
    """Reuse one Sobol prefix across the twelve conditions of a field unit."""

    adapted = copy.deepcopy(case)
    adapted["id"] = str(case["field_unit_id"])
    return adapted


def _field_unit_ledger(
    cases: list[dict[str, Any]], source: dict[str, Any]
) -> list[dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    for case in cases:
        representatives.setdefault(str(case["field_unit_id"]), case)
    rows = []
    for field_unit, case in sorted(representatives.items()):
        execution_case = _execution_case(case)
        values, states, _ = bridge._build_case_context(execution_case, source)
        rows.append(
            {
                "field_unit_id": field_unit,
                "family_id": str(case["family_id"]),
                "phantom_family": str(case["phantom_family"]),
                "phantom_seed": int(case["phantom_seed"]),
                "field_sha256": _tensor_sha256(values),
                "common_sobol_prefix_sha256": _tensor_sha256(states),
                "ray_count": len(states),
                "condition_reuse_count": 12,
            }
        )
    if len(rows) != 8 or any(int(row["ray_count"]) != 256 for row in rows):
        raise ValueError("field-unit ledger is not the frozen 8 x 256 contract")
    return rows


def _cell_metadata(case: dict[str, Any], stress: float) -> dict[str, Any]:
    return {
        "cell_id": f"{case['id']}__stress_{stress:g}",
        "case_id": str(case["id"]),
        "field_unit_id": str(case["field_unit_id"]),
        "ray_prefix_id": str(case["field_unit_id"]),
        "family_id": str(case["family_id"]),
        "orientation_id": str(case["orientation_id"]),
        "aperture_id": str(case["aperture_id"]),
        "dimensionless_stress_multiplier": float(stress),
    }


def _symmetric_method_gates(
    metrics: dict[str, float], screens: dict[str, Any]
) -> dict[str, bool]:
    absolute_gate = float(metrics["candidate_to_high_reference_relative_l2"]) <= float(
        screens["maximum_candidate_to_high_reference_relative_l2"]
    )
    global_scoreable = float(
        metrics["high_execution_to_high_reference_relative_l2"]
    ) >= float(screens["minimum_scoreable_global_reference_relative_l2"])
    q95_scoreable = float(metrics["high_execution_q95_reference_error"]) >= float(
        screens["minimum_scoreable_q95_reference_error"]
    )
    return {
        "matched_residual_gate_met": float(
            metrics["matched_residual_prediction_relative_l2"]
        )
        <= float(screens["maximum_matched_residual_prediction_relative_l2"]),
        "residual_variance_gate_met": float(
            metrics["corrected_residual_variance_ratio"]
        )
        <= float(screens["maximum_corrected_residual_variance_ratio"]),
        "risk_spearman_gate_met": float(metrics["per_ray_risk_spearman"])
        >= float(screens["minimum_per_ray_risk_spearman"]),
        "valid_fraction_gate_met": float(metrics["valid_ray_fraction"])
        >= float(screens["minimum_valid_ray_fraction"]),
        "absolute_reference_gate_met": absolute_gate,
        "global_no_harm_gate_met": (
            float(
                metrics[
                    "candidate_reference_error_to_high_execution_reference_error_ratio"
                ]
            )
            <= float(
                screens[
                    "maximum_candidate_reference_error_to_high_execution_reference_error_ratio"
                ]
            )
            if global_scoreable
            else absolute_gate
        ),
        "q95_no_harm_gate_met": (
            float(metrics["candidate_q95_reference_error_to_high_execution_q95_ratio"])
            <= float(
                screens[
                    "maximum_candidate_q95_reference_error_to_high_execution_q95_ratio"
                ]
            )
            if q95_scoreable
            else float(metrics["candidate_q95_reference_error"])
            <= float(screens["maximum_unscoreable_candidate_q95_reference_error"])
        ),
    }


def _reference_false_safe(row: dict[str, Any]) -> bool:
    gates = row["symmetric_gates"]
    return bool(gates["valid_fraction_gate_met"]) and not all(
        bool(gates[key])
        for key in (
            "absolute_reference_gate_met",
            "global_no_harm_gate_met",
            "q95_no_harm_gate_met",
        )
    )


def _flatten_method_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        key: value
        for key, value in row.items()
        if key not in {"metrics", "gates", "symmetric_gates", "ratio_scoreability"}
    }
    flat.update(row["metrics"])
    flat.update({f"ocbh_gate_{key}": value for key, value in row["gates"].items()})
    flat.update(
        {
            f"symmetric_gate_{key}": value
            for key, value in row["symmetric_gates"].items()
        }
    )
    flat.update(
        {f"ratio_{key}": value for key, value in row["ratio_scoreability"].items()}
    )
    return flat


def _compact_bundle(
    bundle: dict[str, Any],
    case: dict[str, Any],
    stress: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata = _cell_metadata(case, stress)
    method_rows = []
    for raw in bundle["method_rows"]:
        row = copy.deepcopy(raw)
        row.update(metadata)
        row["symmetric_gates"] = _symmetric_method_gates(
            row["metrics"], config["development_screens"]
        )
        row["ratio_scoreability"] = {
            "global_scoreable": float(
                row["metrics"]["high_execution_to_high_reference_relative_l2"]
            )
            >= float(
                config["development_screens"][
                    "minimum_scoreable_global_reference_relative_l2"
                ]
            ),
            "q95_scoreable": float(row["metrics"]["high_execution_q95_reference_error"])
            >= float(
                config["development_screens"]["minimum_scoreable_q95_reference_error"]
            ),
        }
        row["all_symmetric_gates_pass"] = all(row["symmetric_gates"].values())
        method_rows.append(row)
    teacher = copy.deepcopy(bundle["teacher_row"])
    teacher.update(metadata)
    sentinel = copy.deepcopy(bundle["sentinel_row"])
    sentinel.update(metadata)
    return {
        "metadata": metadata,
        "method_rows": method_rows,
        "teacher_row": teacher,
        "sentinel_row": sentinel,
        "query_accounting": copy.deepcopy(bundle["query_accounting"]),
    }


def _checkpoint_path(work_dir: Path, cell_id: str) -> Path:
    return work_dir / "cells" / f"{cell_id}.json"


def _load_or_run_cell(
    case: dict[str, Any],
    source: dict[str, Any],
    bridge_config: dict[str, Any],
    *,
    stress: float,
    work_dir: Path,
    preregistration_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    metadata = _cell_metadata(case, stress)
    path = _checkpoint_path(work_dir, metadata["cell_id"])
    if path.is_file():
        if not resume:
            raise FileExistsError(f"checkpoint exists but resume is disabled: {path}")
        payload = _read_json(path)
        if payload.get("preregistration_sha256") != preregistration_sha256:
            raise ValueError(f"checkpoint preregistration drift: {path}")
        if payload.get("metadata") != metadata:
            raise ValueError(f"checkpoint cell metadata drift: {path}")
        return payload

    execution_case = _execution_case(case)
    bundle = bridge._case_scale_bundle(
        execution_case,
        source,
        bridge_config,
        stress=float(stress),
    )
    payload = _compact_bundle(bundle, case, float(stress), bridge_config)
    payload["preregistration_sha256"] = preregistration_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)
    return payload


def _timing_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        case
        for case in cases
        if case["family_id"] == "smooth" and int(case["phantom_seed"]) == 1729
    ]
    if len(selected) != 4:
        raise ValueError("timing package selection must contain exactly four cases")
    return selected


def _timing_rows(
    cases: list[dict[str, Any]],
    source: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in _timing_cases(cases):
        execution_case = _execution_case(case)
        for raw in bridge._timing_bundle(execution_case, source, config):
            row = copy.deepcopy(raw)
            row.update(
                {
                    "case_id": str(case["id"]),
                    "ray_prefix_id": str(case["field_unit_id"]),
                    "orientation_id": str(case["orientation_id"]),
                    "aperture_id": str(case["aperture_id"]),
                    "timing_package_id": (
                        f"{case['orientation_id']}__{case['aperture_id']}"
                    ),
                }
            )
            rows.append(row)
    return rows


def _field_unit_summaries(
    method_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        grouped[(str(row["field_unit_id"]), str(row["method_id"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (field_unit, method), values in sorted(grouped.items()):
        if len(values) != 12:
            raise ValueError(
                f"field-unit/method group is not 12 cells: {field_unit}/{method}"
            )
        metrics = [row["metrics"] for row in values]
        false_safe = sum(int(_reference_false_safe(value)) for value in values)
        rows.append(
            {
                "field_unit_id": field_unit,
                "family_id": str(values[0]["family_id"]),
                "phantom_family": str(values[0]["phantom_family"]),
                "phantom_seed": int(values[0]["phantom_seed"]),
                "method_id": method,
                "physical_cell_count": len(values),
                "geometric_mean_matched_residual_relative_l2": _geometric_mean(
                    item["matched_residual_prediction_relative_l2"] for item in metrics
                ),
                "worst_matched_residual_relative_l2": max(
                    float(item["matched_residual_prediction_relative_l2"])
                    for item in metrics
                ),
                "worst_global_no_harm_ratio": max(
                    float(
                        item[
                            "candidate_reference_error_to_high_execution_reference_error_ratio"
                        ]
                    )
                    for item in metrics
                ),
                "worst_q95_no_harm_ratio": max(
                    float(
                        item[
                            "candidate_q95_reference_error_to_high_execution_q95_ratio"
                        ]
                    )
                    for item in metrics
                ),
                "minimum_valid_ray_fraction": min(
                    float(item["valid_ray_fraction"]) for item in metrics
                ),
                "false_safe_cell_count": false_safe,
                "raw_invalid_cell_count": sum(
                    int(float(item["valid_ray_fraction"]) < 1.0) for item in metrics
                ),
                "symmetric_gate_pass_cell_count": sum(
                    int(value["all_symmetric_gates_pass"]) for value in values
                ),
            }
        )
    if len(rows) != 8 * len(METHODS):
        raise ValueError("field-unit summary does not contain 8 x 4 rows")
    return rows


def _blocked_bootstrap_ci(
    paired_rows: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    by_family: dict[str, np.ndarray] = {}
    for family in sorted({str(row["family_id"]) for row in paired_rows}):
        values = np.asarray(
            [
                float(row["matched_ratio_vs_ocbh"])
                for row in paired_rows
                if row["family_id"] == family
            ],
            dtype=float,
        )
        if values.shape != (4,):
            raise ValueError("blocked bootstrap requires four field units per family")
        by_family[family] = values
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        sampled = np.concatenate(
            [values[rng.integers(0, 4, size=4)] for values in by_family.values()]
        )
        bootstrap[index] = _geometric_mean(sampled)
    alpha = (1.0 - float(confidence_level)) / 2.0
    return (
        _geometric_mean(row["matched_ratio_vs_ocbh"] for row in paired_rows),
        float(np.quantile(bootstrap, alpha)),
        float(np.quantile(bootstrap, 1.0 - alpha)),
    )


def _query_ratio(query_rows: list[dict[str, Any]], method: str) -> float:
    return max(
        float(row["query_accounting"][method]["logical_scalar_grid_point_queries"])
        / float(row["query_accounting"]["high128"]["logical_scalar_grid_point_queries"])
        for row in query_rows
    )


def _query_ratio_vs_ocbh(query_rows: list[dict[str, Any]], method: str) -> float:
    return max(
        float(row["query_accounting"][method]["logical_scalar_grid_point_queries"])
        / float(
            row["query_accounting"]["operator_consistent_homotopy"][
                "logical_scalar_grid_point_queries"
            ]
        )
        for row in query_rows
    )


def _timing_ratio_vs_ocbh(timing_rows: list[dict[str, Any]], method: str) -> float:
    by_package: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in timing_rows:
        by_package[str(row["timing_package_id"])][str(row["method_id"])] = row
    ratios = []
    for methods in by_package.values():
        ratios.append(
            float(methods[method]["p90_seconds"])
            / max(float(methods["operator_consistent_homotopy"]["p90_seconds"]), 1e-30)
        )
    if len(ratios) != 4:
        raise ValueError("timing dominance requires four packages")
    return max(ratios)


def _floor_stabilized_ratio(
    candidate: float,
    baseline: float,
    *,
    scoreability_floor: float,
) -> tuple[float, bool]:
    """Compare to a baseline without turning machine-zero denominators into evidence."""

    scoreable = float(baseline) >= float(scoreability_floor)
    denominator = float(baseline) if scoreable else float(scoreability_floor)
    return float(candidate) / max(denominator, 1e-30), scoreable


def _dominance_rows(
    method_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["field_unit_id"]), str(row["method_id"])): row for row in field_rows
    }
    by_condition = {
        (
            str(row["field_unit_id"]),
            str(row["orientation_id"]),
            str(row["aperture_id"]),
            float(row["dimensionless_stress_multiplier"]),
            str(row["method_id"]),
        ): row
        for row in method_rows
    }
    decision = config["grouped_decision"]
    gates = decision["picard_forward_dominance"]
    output = []
    for method in gates["eligible_methods"]:
        paired = []
        method_cells = [row for row in method_rows if row["method_id"] == str(method)]
        if len(method_cells) != 96:
            raise ValueError("dominance method does not contain exactly 96 cells")
        for field_unit in sorted({key[0] for key in by_key}):
            candidate = by_key[(field_unit, str(method))]
            conditions = sorted(
                {
                    (key[1], key[2], key[3])
                    for key in by_condition
                    if key[0] == field_unit and key[4] == str(method)
                }
            )
            if len(conditions) != 12:
                raise ValueError("dominance pairing requires 12 conditions per field")
            matched_ratios = []
            global_ratios = []
            q95_ratios = []
            valid_differences = []
            global_scoreable_conditions = 0
            q95_scoreable_conditions = 0
            for orientation, aperture, stress in conditions:
                candidate_cell = by_condition[
                    (field_unit, orientation, aperture, stress, str(method))
                ]
                ocbh_cell = by_condition[
                    (
                        field_unit,
                        orientation,
                        aperture,
                        stress,
                        "operator_consistent_homotopy",
                    )
                ]
                candidate_metrics = candidate_cell["metrics"]
                ocbh_metrics = ocbh_cell["metrics"]
                matched_ratios.append(
                    float(candidate_metrics["matched_residual_prediction_relative_l2"])
                    / max(
                        float(ocbh_metrics["matched_residual_prediction_relative_l2"]),
                        1e-30,
                    )
                )
                global_ratio, global_scoreable = _floor_stabilized_ratio(
                    float(candidate_metrics["candidate_to_high_reference_relative_l2"]),
                    float(ocbh_metrics["candidate_to_high_reference_relative_l2"]),
                    scoreability_floor=float(
                        config["development_screens"][
                            "minimum_scoreable_global_reference_relative_l2"
                        ]
                    ),
                )
                q95_ratio, q95_scoreable = _floor_stabilized_ratio(
                    float(candidate_metrics["candidate_q95_reference_error"]),
                    float(ocbh_metrics["candidate_q95_reference_error"]),
                    scoreability_floor=float(
                        config["development_screens"][
                            "minimum_scoreable_q95_reference_error"
                        ]
                    ),
                )
                global_ratios.append(global_ratio)
                q95_ratios.append(q95_ratio)
                global_scoreable_conditions += int(global_scoreable)
                q95_scoreable_conditions += int(q95_scoreable)
                valid_differences.append(
                    float(candidate_metrics["valid_ray_fraction"])
                    - float(ocbh_metrics["valid_ray_fraction"])
                )
            paired.append(
                {
                    "field_unit_id": field_unit,
                    "family_id": str(candidate["family_id"]),
                    "matched_ratio_vs_ocbh": _geometric_mean(matched_ratios),
                    "worst_condition_matched_ratio_vs_ocbh": max(matched_ratios),
                    "worst_condition_global_no_harm_ratio_vs_ocbh": max(global_ratios),
                    "worst_condition_q95_no_harm_ratio_vs_ocbh": max(q95_ratios),
                    "valid_fraction_difference_vs_ocbh": min(valid_differences),
                    "global_scoreable_condition_count": global_scoreable_conditions,
                    "q95_scoreable_condition_count": q95_scoreable_conditions,
                }
            )
        point, lower, upper = _blocked_bootstrap_ci(
            paired,
            seed=int(decision["bootstrap_seed"]),
            replicates=int(decision["bootstrap_replicates"]),
            confidence_level=float(decision["confidence_level"]),
        )
        row = {
            "method_id": str(method),
            "field_units_with_lower_geometric_mean_matched_error": sum(
                int(item["matched_ratio_vs_ocbh"] < 1.0) for item in paired
            ),
            "grouped_geometric_mean_matched_ratio_vs_ocbh": point,
            "grouped_geometric_mean_matched_ratio_ci_lower": lower,
            "grouped_geometric_mean_matched_ratio_ci_upper": upper,
            "worst_condition_matched_ratio_vs_ocbh": max(
                item["worst_condition_matched_ratio_vs_ocbh"] for item in paired
            ),
            "worst_condition_global_no_harm_ratio_vs_ocbh": max(
                item["worst_condition_global_no_harm_ratio_vs_ocbh"] for item in paired
            ),
            "worst_condition_q95_no_harm_ratio_vs_ocbh": max(
                item["worst_condition_q95_no_harm_ratio_vs_ocbh"] for item in paired
            ),
            "minimum_field_unit_valid_fraction_difference_vs_ocbh": min(
                item["valid_fraction_difference_vs_ocbh"] for item in paired
            ),
            "global_scoreable_condition_count": sum(
                int(item["global_scoreable_condition_count"]) for item in paired
            ),
            "q95_scoreable_condition_count": sum(
                int(item["q95_scoreable_condition_count"]) for item in paired
            ),
            "worst_package_wall_p90_ratio_vs_ocbh": _timing_ratio_vs_ocbh(
                timing_rows, str(method)
            ),
            "maximum_logical_point_query_ratio_to_high128": _query_ratio(
                query_rows, str(method)
            ),
            "maximum_logical_point_query_ratio_vs_ocbh": _query_ratio_vs_ocbh(
                query_rows, str(method)
            ),
            "symmetric_gate_pass_cell_count": sum(
                int(cell["all_symmetric_gates_pass"]) for cell in method_cells
            ),
            "raw_invalid_cell_count": sum(
                int(float(cell["metrics"]["valid_ray_fraction"]) < 1.0)
                for cell in method_cells
            ),
            "false_safe_cell_count": sum(
                int(_reference_false_safe(cell)) for cell in method_cells
            ),
            "paired_field_units": paired,
        }
        row["dominance_gates"] = {
            "all_field_units_matched_better": row[
                "field_units_with_lower_geometric_mean_matched_error"
            ]
            >= int(
                gates["minimum_field_units_with_lower_geometric_mean_matched_error"]
            ),
            "matched_ratio_upper_ci_gate": row[
                "grouped_geometric_mean_matched_ratio_ci_upper"
            ]
            <= float(
                gates[
                    "maximum_upper_confidence_bound_of_grouped_geometric_mean_matched_ratio"
                ]
            ),
            "worst_condition_matched_gate": row["worst_condition_matched_ratio_vs_ocbh"]
            <= float(gates["maximum_worst_condition_matched_ratio_vs_ocbh"]),
            "global_no_harm_gate": row["worst_condition_global_no_harm_ratio_vs_ocbh"]
            <= float(gates["maximum_worst_condition_global_no_harm_ratio_vs_ocbh"]),
            "q95_no_harm_gate": row["worst_condition_q95_no_harm_ratio_vs_ocbh"]
            <= float(gates["maximum_worst_condition_q95_no_harm_ratio_vs_ocbh"]),
            "validity_gate": row["minimum_field_unit_valid_fraction_difference_vs_ocbh"]
            >= float(gates["minimum_field_unit_valid_fraction_difference_vs_ocbh"]),
            "wall_time_gate": row["worst_package_wall_p90_ratio_vs_ocbh"]
            <= float(gates["maximum_package_wall_p90_ratio_vs_ocbh"]),
            "query_gate": row["maximum_logical_point_query_ratio_vs_ocbh"]
            <= float(gates["maximum_logical_point_query_ratio_vs_ocbh"]),
            "symmetric_per_cell_gate": row["symmetric_gate_pass_cell_count"] == 96,
            "raw_validity_gate": row["raw_invalid_cell_count"] == 0,
            "false_safe_gate": row["false_safe_cell_count"]
            <= int(config["development_screens"]["maximum_false_safe_cell_count"]),
        }
        row["dominates_ocbh_forward_role"] = all(row["dominance_gates"].values())
        output.append(row)
    return output


def _flatten_query_rows(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in query_rows:
        metadata = {
            key: value for key, value in row.items() if key != "query_accounting"
        }
        for method, accounting in row["query_accounting"].items():
            flat.append({**metadata, "method_id": method, **accounting})
    return flat


def _ocbh_gate_summary(
    method_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    sentinel_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    primary = [
        row for row in method_rows if row["method_id"] == "operator_consistent_homotopy"
    ]
    timing = [
        row for row in timing_rows if row["method_id"] == "operator_consistent_homotopy"
    ]
    query_limit = float(
        config["development_screens"][
            "maximum_candidate_to_high128_logical_point_query_ratio"
        ]
    )
    query_pass = sum(
        int(_query_ratio([row], "operator_consistent_homotopy") <= query_limit)
        for row in query_rows
    )
    summary = {
        "primary_pass_count": sum(
            int(
                row["all_symmetric_gates_pass"]
                and row["gates"].get("base_output_consistency_gate_met", False)
            )
            for row in primary
        ),
        "primary_required_count": len(primary),
        "teacher_pass_count": sum(int(row["all_gates_pass"]) for row in teacher_rows),
        "teacher_required_count": len(teacher_rows),
        "sentinel_pass_count": sum(int(row["all_gates_pass"]) for row in sentinel_rows),
        "sentinel_required_count": len(sentinel_rows),
        "timing_pass_count": sum(
            int(row.get("timing_gate_met", False)) for row in timing
        ),
        "timing_required_count": len(timing),
        "query_pass_count": query_pass,
        "query_required_count": len(query_rows),
    }
    summary["all_inherited_gates_pass"] = all(
        summary[key.replace("required", "pass")] == value
        for key, value in summary.items()
        if key.endswith("required_count")
    )
    return summary


def _write_figure(
    path: Path,
    method_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    dominance_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    gate_summary: dict[str, Any],
    decision: str,
) -> None:
    colors = ["#9a5b4a", "#286f69", "#426a9c", "#7d6098"]
    labels = ["N1 affine", "OCBH", "Picard-1", "Picard-2"]
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    worst_matched = [
        max(
            float(row["metrics"]["matched_residual_prediction_relative_l2"])
            for row in method_rows
            if row["method_id"] == method
        )
        for method in METHODS
    ]
    axes[0, 0].bar(labels, worst_matched, color=colors)
    axes[0, 0].set_yscale("log")
    axes[0, 0].tick_params(axis="x", rotation=18)
    axes[0, 0].set_title("96-cell worst matched residual")

    worst_no_harm = [
        max(
            float(
                row["metrics"][
                    "candidate_reference_error_to_high_execution_reference_error_ratio"
                ]
            )
            for row in method_rows
            if row["method_id"] == method
        )
        for method in METHODS
    ]
    axes[0, 1].bar(labels, worst_no_harm, color=colors)
    axes[0, 1].axhline(1.1, color="#b44b3f", linestyle="--")
    axes[0, 1].tick_params(axis="x", rotation=18)
    axes[0, 1].set_title("96-cell worst H256 no-harm")

    for index, row in enumerate(dominance_rows):
        point = float(row["grouped_geometric_mean_matched_ratio_vs_ocbh"])
        lower = float(row["grouped_geometric_mean_matched_ratio_ci_lower"])
        upper = float(row["grouped_geometric_mean_matched_ratio_ci_upper"])
        axes[0, 2].errorbar(
            index,
            point,
            yerr=[[point - lower], [upper - point]],
            fmt="o",
            capsize=5,
            color=colors[index + 2],
        )
    axes[0, 2].axhline(1.0, color="#50565c", linestyle="--")
    axes[0, 2].axhline(0.75, color="#b44b3f", linestyle=":")
    axes[0, 2].set_xticks([0, 1], ["Picard-1", "Picard-2"])
    axes[0, 2].set_title("field-blocked matched ratio / OCBH")

    for method, color in zip(METHODS, colors, strict=True):
        values = [
            float(row["geometric_mean_matched_residual_relative_l2"])
            for row in field_rows
            if row["method_id"] == method
        ]
        axes[1, 0].plot(range(8), values, marker="o", label=method, color=color)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("field unit (4 smooth + 4 wrinkled)")
    axes[1, 0].set_title("field-unit geometric matched error")
    axes[1, 0].legend(fontsize=7)

    timing_max = []
    for method in METHODS + ("high128",):
        timing_max.append(
            max(
                float(row["candidate_p90_to_high128_p10_wall_time_ratio"])
                for row in timing_rows
                if row["method_id"] == method
            )
        )
    axes[1, 1].bar(labels + ["H128"], timing_max, color=colors + ["#50565c"])
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].set_title("four-package worst p90 / H128 p10")

    gate_labels = ["primary", "teacher", "sentinel", "timing", "query"]
    gate_values = [
        gate_summary[f"{name}_pass_count"] / gate_summary[f"{name}_required_count"]
        for name in gate_labels
    ]
    axes[1, 2].bar(gate_labels, gate_values, color="#286f69")
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].set_title("OCBH inherited gate fractions")
    axes[1, 2].text(
        0.02,
        0.04,
        decision.replace("_", " "),
        transform=axes[1, 2].transAxes,
        fontsize=8,
        wrap=True,
    )
    figure.suptitle(
        "N2-PVGR-N3 preregistered grouped factorial: development only",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# N2-PVGR-N3 grouped 96-cell development factorial",
        "",
        f"- Machine decision: `{result['machine_decision']}`",
        f"- Physical cells: `{result['physical_cell_count']}`",
        f"- Independent field units: `{result['field_unit_count']}`",
        (
            "- OCBH inherited gates: "
            f"`{result['ocbh_gate_summary']['all_inherited_gates_pass']}`"
        ),
    ]
    for row in result["picard_forward_dominance_rows"]:
        lines.append(
            f"- {row['method_id']} dominates OCBH forward role: "
            f"`{row['dominates_ocbh_forward_role']}`"
        )
    lines.extend(
        [
            "- Reserved families remain unopened.",
            "- Reconstruction, real-data, neural superiority, and paper claims remain unauthorized.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    config_path: Path,
    output_dir: Path,
    work_dir: Path,
    *,
    resume: bool = True,
    enforce_formal_output: bool = True,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    source_path = _resolve(str(config["source_config"]))
    source = _read_json(source_path)
    _validate_contract(config, source)
    parent = _read_json(_resolve(str(config["parent_mechanism_result"])))
    if parent.get("machine_decision") != (
        "MECHANISM_BRIDGE_SIGNAL_ONLY_96_CELL_RECONSTRUCTION_AND_REAL_DATA_GATES_CLOSED"
    ):
        raise ValueError("parent mechanism result is not the frozen N2 bridge")
    attestation = _validate_preregistration(config, config_path)
    attestation_path = _resolve(str(config["pre_registration_attestation"]))
    preregistration_sha256 = _sha256(attestation_path)

    formal_output = _resolve(str(config["formal_output"])).resolve()
    formal_work = _resolve(str(config["formal_work_output"])).resolve()
    if enforce_formal_output and output_dir.resolve() != formal_output:
        raise ValueError("formal run output path drifted from preregistration")
    if enforce_formal_output and work_dir.resolve() != formal_work:
        raise ValueError("formal work path drifted from preregistration")
    if output_dir.exists():
        raise FileExistsError("formal output already exists; results are immutable")
    if work_dir.exists() and any(work_dir.iterdir()) and not resume:
        raise FileExistsError("work checkpoints exist but resume is disabled")
    work_dir.mkdir(parents=True, exist_ok=True)

    cases = expand_factorial_cases(config)
    run_source = _source_for_run(source, config, cases)
    adapted_config = _bridge_config(config)
    field_unit_ledger = _field_unit_ledger(cases, run_source)
    method_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    sentinel_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    stresses = [
        float(value)
        for value in config["factorial"]["dimensionless_stress_scale_multipliers"]
    ]
    total = len(cases) * len(stresses)
    completed = 0
    for case in cases:
        for stress in stresses:
            payload = _load_or_run_cell(
                case,
                run_source,
                adapted_config,
                stress=stress,
                work_dir=work_dir,
                preregistration_sha256=preregistration_sha256,
                resume=resume,
            )
            method_rows.extend(payload["method_rows"])
            teacher_rows.append(payload["teacher_row"])
            sentinel_rows.append(payload["sentinel_row"])
            query_rows.append(
                {
                    **payload["metadata"],
                    "query_accounting": payload["query_accounting"],
                }
            )
            completed += 1
            print(
                f"[{completed:03d}/{total:03d}] {payload['metadata']['cell_id']}",
                flush=True,
            )

    timing_rows = _timing_rows(cases, run_source, adapted_config)
    field_rows = _field_unit_summaries(method_rows, config)
    dominance_rows = _dominance_rows(
        method_rows,
        field_rows,
        timing_rows,
        query_rows,
        config,
    )
    gate_summary = _ocbh_gate_summary(
        method_rows,
        teacher_rows,
        sentinel_rows,
        timing_rows,
        query_rows,
        config,
    )
    any_picard_dominates = any(
        bool(row["dominates_ocbh_forward_role"]) for row in dominance_rows
    )
    if not gate_summary["all_inherited_gates_pass"]:
        decision = "GROUPED_FACTORIAL_FAIL_NO_FORWARD_AUTHORIZATION"
    elif any_picard_dominates:
        decision = "PICARD_DOMINATES_OCBH_FORWARD_ROLE_CLOSED_FIELD_VJP_GATE_NEXT"
    else:
        decision = "OCBH_NOT_DOMINATED_CONDITIONAL_FIELD_VJP_GATE_NEXT"

    result = {
        "schema": str(config["schema"]),
        "candidate_id": str(config["candidate_id"]),
        "machine_decision": decision,
        "formal_preregistration_validated": True,
        "protocol_commit": str(attestation["protocol_commit"]),
        "run_head_commit": _git_text("rev-parse", "HEAD"),
        "preregistration_attestation_sha256": preregistration_sha256,
        "physical_cell_count": len(query_rows),
        "field_unit_count": len({row["field_unit_id"] for row in method_rows}),
        "field_unit_ledger": field_unit_ledger,
        "ray_count_per_cell": int(config["population_count"]),
        "ray_is_independent_repeat": False,
        "ocbh_gate_summary": gate_summary,
        "picard_forward_dominance_rows": dominance_rows,
        "method_rows": method_rows,
        "field_unit_summary_rows": field_rows,
        "teacher_rows": teacher_rows,
        "reference_sentinel_rows": sentinel_rows,
        "timing_rows": timing_rows,
        "query_accounting_rows": query_rows,
        "claim_authorizations": copy.deepcopy(config["claim_authorizations"]),
        "reserved_audit_families_not_opened": list(
            config["reserved_audit_families_not_opened"]
        ),
        "claim_boundary": (
            "Preregistered synthetic development evidence only. It does not prove "
            "three-dimensional reconstruction, neural-operator superiority, real "
            "OERF validity, cross-family audit generalization, novelty, or a paper claim."
        ),
        "mandatory_next_gate": (
            "field JVP/VJP dot tests and equal-budget three-dimensional reconstruction"
        ),
        "figure": "n2_pvgr_n3_grouped_factorial.png",
    }

    artifact_dir = _prepare_staging_dir(work_dir)
    result_path = artifact_dir / "result.json"
    _atomic_json(result_path, result)
    bridge._write_csv(
        artifact_dir / "metrics.csv",
        [_flatten_method_row(row) for row in method_rows],
    )
    bridge._write_csv(artifact_dir / "field_unit_summary.csv", field_rows)
    bridge._write_csv(artifact_dir / "field_unit_ledger.csv", field_unit_ledger)
    bridge._write_csv(
        artifact_dir / "picard_forward_dominance.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"paired_field_units", "dominance_gates"}
            }
            | {f"gate_{key}": value for key, value in row["dominance_gates"].items()}
            for row in dominance_rows
        ],
    )
    bridge._write_csv(
        artifact_dir / "teacher_metrics.csv",
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"metrics", "gates"}
                },
                **row["metrics"],
                **row["gates"],
            }
            for row in teacher_rows
        ],
    )
    bridge._write_csv(
        artifact_dir / "reference_sentinel.csv",
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"metrics", "gates"}
                },
                **row["metrics"],
                **row["gates"],
            }
            for row in sentinel_rows
        ],
    )
    bridge._write_csv(artifact_dir / "timing.csv", timing_rows)
    bridge._write_csv(
        artifact_dir / "query_accounting.csv", _flatten_query_rows(query_rows)
    )
    figure_path = artifact_dir / str(result["figure"])
    _write_figure(
        figure_path,
        method_rows,
        field_rows,
        dominance_rows,
        timing_rows,
        gate_summary,
        decision,
    )
    _write_summary(artifact_dir / "summary.md", result)
    (artifact_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_inputs: dict[str, tuple[Path, Path]] = {
        "config": (config_path, config_path),
        "attestation": (attestation_path, attestation_path),
        "source_config": (source_path, source_path),
        "parent_result": (
            _resolve(str(config["parent_mechanism_result"])),
            _resolve(str(config["parent_mechanism_result"])),
        ),
        "result": (result_path, output_dir / "result.json"),
        "metrics": (artifact_dir / "metrics.csv", output_dir / "metrics.csv"),
        "field_unit_summary": (
            artifact_dir / "field_unit_summary.csv",
            output_dir / "field_unit_summary.csv",
        ),
        "field_unit_ledger": (
            artifact_dir / "field_unit_ledger.csv",
            output_dir / "field_unit_ledger.csv",
        ),
        "picard_forward_dominance": (
            artifact_dir / "picard_forward_dominance.csv",
            output_dir / "picard_forward_dominance.csv",
        ),
        "teacher_metrics": (
            artifact_dir / "teacher_metrics.csv",
            output_dir / "teacher_metrics.csv",
        ),
        "reference_sentinel": (
            artifact_dir / "reference_sentinel.csv",
            output_dir / "reference_sentinel.csv",
        ),
        "timing": (artifact_dir / "timing.csv", output_dir / "timing.csv"),
        "query_accounting": (
            artifact_dir / "query_accounting.csv",
            output_dir / "query_accounting.csv",
        ),
        "figure": (figure_path, output_dir / str(result["figure"])),
        "summary": (artifact_dir / "summary.md", output_dir / "summary.md"),
        "config_snapshot": (
            artifact_dir / "config_snapshot.json",
            output_dir / "config_snapshot.json",
        ),
    }
    for key, relative in config["attested_files"].items():
        path = _resolve(str(relative))
        manifest_inputs[f"attested_{key}"] = (path, path)
    manifest = {
        "schema": "n2-pvgr-n3-grouped-factorial-manifest-1.0",
        "protocol_commit": str(attestation["protocol_commit"]),
        "run_head_commit": result["run_head_commit"],
        "files": {
            key: {
                "path": _relative(declared_path),
                "sha256": _sha256(source_path_for_hash),
                "bytes": source_path_for_hash.stat().st_size,
            }
            for key, (source_path_for_hash, declared_path) in manifest_inputs.items()
        },
    }
    _atomic_json(artifact_dir / "manifest.json", manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(artifact_dir, output_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-output", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = _read_json(config_path)
    source = _read_json(_resolve(str(config["source_config"])))
    _validate_contract(config, source)
    attestation = _validate_preregistration(config, config_path)
    cases = expand_factorial_cases(config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "protocol_commit": attestation["protocol_commit"],
                    "field_units": len({case["field_unit_id"] for case in cases}),
                    "base_cases": len(cases),
                    "physical_cells": len(cases)
                    * len(
                        config["factorial"]["dimensionless_stress_scale_multipliers"]
                    ),
                    "reserved_families_opened": False,
                    "formal_output_exists": _resolve(
                        str(config["formal_output"])
                    ).exists(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    output = (args.output or _resolve(str(config["formal_output"]))).resolve()
    work = (args.work_output or _resolve(str(config["formal_work_output"]))).resolve()
    result = run(
        config_path,
        output,
        work,
        resume=not args.no_resume,
        enforce_formal_output=True,
    )
    print(json.dumps({"machine_decision": result["machine_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
