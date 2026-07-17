#!/usr/bin/env python3
"""Build a frozen, checksum-verified public slice of the Metric-A v2 smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "demo_t16_operator/results/cancellation_aware_metric_surrogate_smoke"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "demo_t16_operator/results/cancellation_aware_metric_surrogate_smoke_public"
)
INPUT_SCHEMA = "cancellation-aware-metric-surrogate-smoke-report-2.0"
INTERFACE_SCHEMA = "cancellation-aware-metric-surrogate-smoke-2.0"
PUBLIC_SCHEMA = "cancellation-aware-metric-surrogate-smoke-public-1.0"
EVIDENCE_SCOPE = "SYNTHETIC_TINY_SIGNED_MATRIX_THREE_WAY_GEOMETRY_SMOKE_ONLY"
STATUS = "DEVELOPMENT_ONLY_SYNTHETIC_INTERFACE_SMOKE"
FRESH_RIGS = ("ood-00", "ood-01", "ood-02", "ood-03")
METHODS = (
    "factor",
    "exact_oracle",
    "scalar_factor_train_selected",
    "exact_factor_interpolation_oracle",
    "learned_oracle_free",
    "calibrated_envelope",
)
METHOD_LABELS = {
    "factor": "Factor",
    "exact_oracle": "Exact oracle",
    "scalar_factor_train_selected": "Scalar x factor",
    "exact_factor_interpolation_oracle": "Exact duplicate (alpha=1)",
    "learned_oracle_free": "Raw learned",
    "calibrated_envelope": "Calibrated",
}
SOURCE_FILES = {
    "report.json",
    "geometry_manifest.csv",
    "metric_rows.csv",
    "trajectory_rows.csv",
    "predictions.csv",
    "model_parameters.csv",
}
EXPECTED_SOURCE_SHA256 = {
    "geometry_manifest.csv": "49e7e9e6ca32b90bcb294f71ca2af947f6369b228b7d6ad6e1d2ed3659b2292a",
    "metric_rows.csv": "c185084aa54e8a8e0124fb9381c795498e7f6f24add6876d80b2f80516f1dd3b",
    "model_parameters.csv": "656fe544cedcfa99ceb26d8578a9fb43a707f5a5e63989e9cdd604b2c77ce243",
    "predictions.csv": "4811b0cc49b25d93e7c57d9196f06d1d121f529e580abe678e57b62ea12086b8",
    "report.json": "57dd60515afe49ac9a0448e0eaa2a1e839a2ca5e5e3efcdcd7579dcb9dd2af8a",
    "trajectory_rows.csv": "56fcbb8d7e34066a810da7991e5c3af6e00f1d9bf34516b99ba9dd762f1b45dc",
}
PUBLIC_GENERATED_FILES = {
    "README.md",
    "summary.json",
    "method_summary.csv",
    "fresh_rig_comparison.csv",
    "decision_gates.csv",
    "diagnostic.png",
    "diagnostic.pdf",
}
PUBLIC_FILES = PUBLIC_GENERATED_FILES | {"checksums.sha256"}
EXPECTED_CLAIM_BOUNDARY = {
    "generalization_claimed": False,
    "new_algorithm_claimed": False,
    "real_data_used": False,
    "superiority_claimed": False,
}
EXPECTED_METHOD_CONTRACTS = {
    "calibrated_envelope": "DEPLOYABLE_INPUT_ONLY_AFTER_EXACT_CALIBRATION_COST_NO_FRESH_EXACT",
    "exact_factor_interpolation_oracle": "NONDEPLOYABLE_TRAIN_SELECTED_EXACT_FACTOR_ORACLE",
    "exact_oracle": "NONDEPLOYABLE_EXACT_ABS_A_ORACLE",
    "factor": "DEPLOYABLE_FACTOR_MAJORIZER_BASELINE",
    "learned_oracle_free": "DEPLOYABLE_INPUT_ONLY_UNCLIPPED_ESTIMATOR",
    "scalar_factor_train_selected": "DEPLOYABLE_TRAIN_SELECTED_SCALAR_TIMES_FACTOR_BASELINE",
}
METRIC_FIELDS = {
    "rig_id", "split_role", "method", "row_relative_l2_vs_exact",
    "column_relative_l2_vs_exact", "row_p95_relative_entry_error",
    "column_p95_relative_entry_error", "final_normalized_residual_l2",
    "final_field_relative_l2", "row_violation_count", "column_violation_count",
    "spectral_violation_count", "total_violation_count", "maximum_row_product",
    "maximum_column_product", "dense_normalized_spectral_norm_squared",
    "schur_squared_upper_bound", "exact_masses_recomputed_from_signed_a",
}
TRAJECTORY_FIELDS = {
    "rig_id", "split_role", "method", "iteration", "normalized_residual_l2",
    "field_relative_l2", "solution_l2",
}
PREDICTION_FIELDS = {"rig_id", "method", "axis", "index", "mass"}
MODEL_FIELDS = {"tensor", "index", "value"}
GEOMETRY_FIELDS = {
    "rig_id", "split_role", "geometry_seed", "noise_seed", "angle", "aperture",
    "shear", "cancellation", "geometry_parameters_sha256",
}


class ValidationError(ValueError):
    """Raised when the frozen source or generated public bundle drifts."""


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"missing or unsafe input: {path.name}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    result = _number(value, name)
    if not result.is_integer():
        raise ValidationError(f"{name} must be an integer")
    return int(result)


def _close(left: Any, right: Any, name: str, *, rtol: float = 1e-10) -> None:
    a, b = _number(left, name), _number(right, name)
    if not math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12):
        raise ValidationError(f"source arithmetic drift: {name}: {a} != {b}")


def _load_csv(path: Path, expected_fields: set[str]) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"missing or unsafe input: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValidationError(f"CSV schema drift: {path.name}")
        rows = list(reader)
    if not rows or any(None in row or set(row) != expected_fields for row in rows):
        raise ValidationError(f"malformed CSV: {path.name}")
    return rows


def _validate_manifest(root: Path, *, enforce_frozen_hashes: bool) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValidationError("source root is missing or unsafe")
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    if observed_files != SOURCE_FILES | {"checksums.sha256"}:
        raise ValidationError("source file set drift")
    if any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise ValidationError("source contains an unsafe entry")
    manifest_path = root / "checksums.sha256"
    manifest: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        token = PurePosixPath(name)
        if (
            not separator
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or token.is_absolute()
            or len(token.parts) != 1
            or name in manifest
        ):
            raise ValidationError("malformed source checksum manifest")
        manifest[name] = digest
    if set(manifest) != SOURCE_FILES:
        raise ValidationError("source checksum file set drift")
    for name, digest in manifest.items():
        actual = _sha256(root / name)
        if actual != digest:
            raise ValidationError(f"source checksum mismatch: {name}")
    if enforce_frozen_hashes and manifest != EXPECTED_SOURCE_SHA256:
        raise ValidationError("frozen source hash mismatch")
    return manifest


def _validate_metric_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {(rig, method) for rig in FRESH_RIGS for method in METHODS}
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    integer_fields = {
        "row_violation_count", "column_violation_count", "spectral_violation_count",
        "total_violation_count", "exact_masses_recomputed_from_signed_a",
    }
    numeric_fields = METRIC_FIELDS - {"rig_id", "split_role", "method"} - integer_fields
    for row in rows:
        key = (row["rig_id"], row["method"])
        if key in parsed or key not in expected or row["split_role"] != "fresh_geometry_ood":
            raise ValidationError(f"unexpected metric identity: {key}")
        item: dict[str, Any] = dict(row)
        for name in numeric_fields:
            item[name] = _number(row[name], name)
            if item[name] < 0.0:
                raise ValidationError(f"negative metric: {name}")
        for name in integer_fields:
            item[name] = _integer(row[name], name)
            if item[name] < 0:
                raise ValidationError(f"negative count: {name}")
        if item["exact_masses_recomputed_from_signed_a"] != 1:
            raise ValidationError("Schur audit did not recompute exact masses from signed A")
        summed = (
            item["row_violation_count"]
            + item["column_violation_count"]
            + item["spectral_violation_count"]
        )
        if item["total_violation_count"] != summed:
            raise ValidationError("violation count arithmetic drift")
        parsed[key] = item
    if set(parsed) != expected:
        raise ValidationError("metric row coverage drift")
    return parsed


def _validate_trajectory(
    rows: list[dict[str, str]], metrics: Mapping[tuple[str, str], Mapping[str, Any]]
) -> None:
    checkpoints = (0, 1, 2, 4, 8, 16, 32)
    expected = {
        (rig, method, iteration)
        for rig in FRESH_RIGS
        for method in METHODS
        for iteration in checkpoints
    }
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (row["rig_id"], row["method"], _integer(row["iteration"], "iteration"))
        if key in seen or key not in expected or row["split_role"] != "fresh_geometry_ood":
            raise ValidationError(f"unexpected trajectory identity: {key}")
        seen.add(key)
        residual = _number(row["normalized_residual_l2"], "normalized_residual_l2")
        field = _number(row["field_relative_l2"], "field_relative_l2")
        solution = _number(row["solution_l2"], "solution_l2")
        if min(residual, field, solution) < 0.0:
            raise ValidationError("negative trajectory metric")
        if key[-1] == 32:
            metric = metrics[(key[0], key[1])]
            _close(residual, metric["final_normalized_residual_l2"], "final residual")
            _close(field, metric["final_field_relative_l2"], "final field")
    if seen != expected:
        raise ValidationError("trajectory coverage drift")


def _validate_predictions(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], float]:
    expected = {
        (rig, method, axis, index)
        for rig in FRESH_RIGS
        for method in METHODS
        for axis, size in (("row", 14), ("column", 11))
        for index in range(size)
    }
    parsed: dict[tuple[str, str, str, int], float] = {}
    for row in rows:
        key = (row["rig_id"], row["method"], row["axis"], _integer(row["index"], "index"))
        mass = _number(row["mass"], "mass")
        if key in parsed or key not in expected or mass <= 0.0:
            raise ValidationError(f"unexpected prediction identity or mass: {key}")
        parsed[key] = mass
    if set(parsed) != expected:
        raise ValidationError("prediction coverage drift")
    return parsed


def _validate_model(rows: list[dict[str, str]], report: Mapping[str, Any]) -> None:
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["tensor"], _integer(row["index"], "model index"))
        if key in seen or key[1] < 0:
            raise ValidationError("model parameter identity drift")
        seen.add(key)
        _number(row["value"], "model value")
    if len(rows) != _integer(report["training"]["parameter_count"], "parameter_count") + 32:
        raise ValidationError("model parameter row count drift")


def _validate_geometry(rows: list[dict[str, str]], report: Mapping[str, Any]) -> None:
    assignments = report["config"]["rigs"]["assignments"]
    if not isinstance(assignments, dict):
        raise ValidationError("geometry assignment schema drift")
    expected = set(assignments.items())
    seen: set[tuple[str, str]] = set()
    geometry_hashes: set[str] = set()
    for row in rows:
        key = (row["rig_id"], row["split_role"])
        digest = row["geometry_parameters_sha256"]
        if key in seen or key not in expected or len(digest) != 64:
            raise ValidationError("geometry manifest identity drift")
        seen.add(key)
        geometry_hashes.add(digest)
        _integer(row["geometry_seed"], "geometry_seed")
        _integer(row["noise_seed"], "noise_seed")
        for name in ("angle", "aperture", "shear", "cancellation"):
            _number(row[name], name)
    if seen != expected or len(rows) != 15 or len(geometry_hashes) != 15:
        raise ValidationError("geometry manifest coverage or independence drift")


def _validate_report_arithmetic(
    report: Mapping[str, Any], metrics: Mapping[tuple[str, str], Mapping[str, Any]]
) -> None:
    aggregate = report.get("aggregate_fresh_ood")
    if not isinstance(aggregate, dict) or set(aggregate) != set(METHODS):
        raise ValidationError("aggregate method set drift")
    for method in METHODS:
        rows = [metrics[(rig, method)] for rig in FRESH_RIGS]
        expected = aggregate[method]
        if set(expected) != {
            "fresh_rig_count", "fresh_rigs_with_any_schur_violation",
            "mean_final_field_relative_l2", "mean_final_normalized_residual_l2",
        }:
            raise ValidationError("aggregate schema drift")
        if _integer(expected["fresh_rig_count"], "fresh_rig_count") != 4:
            raise ValidationError("fresh rig count drift")
        unsafe = sum(row["total_violation_count"] > 0 for row in rows)
        if _integer(expected["fresh_rigs_with_any_schur_violation"], "unsafe rigs") != unsafe:
            raise ValidationError("aggregate unsafe-rig arithmetic drift")
        _close(
            expected["mean_final_field_relative_l2"],
            np.mean([row["final_field_relative_l2"] for row in rows]),
            f"{method} field mean",
        )
        _close(
            expected["mean_final_normalized_residual_l2"],
            np.mean([row["final_normalized_residual_l2"] for row in rows]),
            f"{method} residual mean",
        )
    decision = report.get("decision")
    if not isinstance(decision, dict):
        raise ValidationError("missing decision")
    if decision.get("metric_substitution_authorized") is not False:
        raise ValidationError("metric substitution decision drift")
    if decision.get("research_claim_authorized") is not False:
        raise ValidationError("research claim decision drift")
    if decision.get("calibrated_envelope_all_fresh_schur_safe") is not False:
        raise ValidationError("calibrated safety decision drift")
    if decision.get("calibrated_envelope_beats_factor_and_simple_on_each_fresh_rig") is not False:
        raise ValidationError("calibrated stable-win decision drift")
    total = sum(row["total_violation_count"] for row in metrics.values())
    if _integer(decision.get("all_methods_total_schur_violation_count"), "all violations") != total:
        raise ValidationError("decision violation arithmetic drift")
    flags = {
        rig: (
            metrics[(rig, "calibrated_envelope")]["final_field_relative_l2"]
            < metrics[(rig, "factor")]["final_field_relative_l2"]
            and metrics[(rig, "calibrated_envelope")]["final_field_relative_l2"]
            < metrics[(rig, "scalar_factor_train_selected")]["final_field_relative_l2"]
        )
        for rig in FRESH_RIGS
    }
    if decision.get("per_fresh_rig_stable_win_flags") != flags:
        raise ValidationError("per-rig stable-win arithmetic drift")
    if sum(flags.values()) != 2:
        raise ValidationError("frozen 2/4 stable-win result drift")
    selection = report.get("simple_control_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("exact_factor_interpolation_is_oracle") is not True
        or selection.get("selected_exact_factor_duplicate_of_exact_oracle") is not True
    ):
        raise ValidationError("exact-factor oracle boundary drift")
    _close(selection.get("selected_exact_factor_alpha"), 1.0, "selected exact-factor alpha")
    for rig in FRESH_RIGS:
        exact = metrics[(rig, "exact_oracle")]
        duplicate = metrics[(rig, "exact_factor_interpolation_oracle")]
        for name in METRIC_FIELDS - {"rig_id", "split_role", "method"}:
            _close(exact[name], duplicate[name], f"{rig} exact duplicate {name}")


def load_release(
    root: Path, *, enforce_frozen_hashes: bool = True
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str, int], float],
    dict[str, str],
]:
    manifest = _validate_manifest(root, enforce_frozen_hashes=enforce_frozen_hashes)
    report = _load_json(root / "report.json")
    if not isinstance(report, dict):
        raise ValidationError("report is not an object")
    if report.get("schema_version") != INPUT_SCHEMA or report.get("interface_schema_version") != INTERFACE_SCHEMA:
        raise ValidationError("report schema drift")
    if report.get("evidence_scope") != EVIDENCE_SCOPE or report.get("status") != STATUS:
        raise ValidationError("evidence scope or status drift")
    if report.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        raise ValidationError("claim boundary drift")
    if report.get("method_contracts") != EXPECTED_METHOD_CONTRACTS:
        raise ValidationError("method contract drift")
    split = report.get("split_contract")
    if (
        not isinstance(split, dict)
        or tuple(split.get("fresh_geometry_ood_rig_ids", ())) != FRESH_RIGS
        or split.get("split_unit") != "COMPLETE_RIG"
        or split.get("geometry_parameters_independently_sampled") is not True
        or split.get("noise_seed_is_not_geometry_seed") is not True
        or split.get("random_ray_split_used") is not False
    ):
        raise ValidationError("fresh split contract drift")
    calibration = report.get("calibration_envelope")
    if (
        not isinstance(calibration, dict)
        or calibration.get("fresh_exact_access") is not False
        or calibration.get("factor_mass_vector_accesses") != 6
        or calibration.get("factor_feature_construction_calls") != 3
    ):
        raise ValidationError("fresh exact-access boundary drift")
    feature_cost = report.get("feature_cost_contract")
    if (
        not isinstance(feature_cost, dict)
        or feature_cost.get("learned_features_require_factor_row_and_column_mass") is not True
        or feature_cost.get("end_to_end_cost_reduction_claimed") is not False
    ):
        raise ValidationError("feature cost contract drift")
    evidence_counting = report.get("evidence_counting")
    if (
        not isinstance(evidence_counting, dict)
        or evidence_counting.get("raw_method_count") != 6
        or evidence_counting.get("independent_method_count") != 5
        or evidence_counting.get("duplicate_methods") != {
            "exact_factor_interpolation_oracle": {
                "duplicate_of_exact_oracle": True,
                "reason": "TRAIN_SELECTED_ALPHA_EQUALS_1.0",
            }
        }
    ):
        raise ValidationError("evidence counting drift")
    provenance = report.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("source_snapshot_status")
        != "COMMITTED_CLEAN_REPRODUCIBLE_FROM_COMMIT"
        or provenance.get("source_tree_clean") is not True
        or provenance.get("clean_rerun_required_after_commit") is not False
        or provenance.get("geometry_manifest_sha256")
        != EXPECTED_SOURCE_SHA256["geometry_manifest.csv"]
        or provenance.get("fresh_predictions_sha256")
        != EXPECTED_SOURCE_SHA256["predictions.csv"]
        or provenance.get("model_parameters_sha256")
        != EXPECTED_SOURCE_SHA256["model_parameters.csv"]
    ):
        raise ValidationError("source provenance boundary drift")
    timing = report.get("timing")
    if not isinstance(timing, dict) or timing.get("role") != "MEASURED_SINGLE_RUN_NONCOMPARATIVE":
        raise ValidationError("timing evidence boundary drift")
    fresh_ledgers = report.get("call_ledger", {}).get("fresh_by_method", {})
    fresh_timing = timing.get("fresh_method_seconds", {})
    if set(fresh_ledgers) != set(METHODS) or set(fresh_timing) != set(METHODS):
        raise ValidationError("fresh cost ledger method set drift")
    for method in ("learned_oracle_free", "calibrated_envelope"):
        if (
            fresh_ledgers[method].get("factor_mass_vector_accesses") != 8
            or fresh_ledgers[method].get("factor_feature_construction_calls") != 4
            or fresh_timing[method].get("factor_feature_construction_is_setup_subcomponent") is not True
        ):
            raise ValidationError("learned factor-feature cost ledger drift")
    instrumentation = report.get("fresh_exact_access_instrumentation")
    if (
        not isinstance(instrumentation, dict)
        or instrumentation.get("fresh_candidate_exact_access") is not False
        or instrumentation.get("fresh_candidate_exact_mass_access_count") != 0
        or instrumentation.get("blocked_exact_mass_access_count") != 0
    ):
        raise ValidationError("fresh exact-access instrumentation drift")

    geometry_rows = _load_csv(root / "geometry_manifest.csv", GEOMETRY_FIELDS)
    metric_rows = _load_csv(root / "metric_rows.csv", METRIC_FIELDS)
    trajectory_rows = _load_csv(root / "trajectory_rows.csv", TRAJECTORY_FIELDS)
    prediction_rows = _load_csv(root / "predictions.csv", PREDICTION_FIELDS)
    model_rows = _load_csv(root / "model_parameters.csv", MODEL_FIELDS)
    metrics = _validate_metric_rows(metric_rows)
    _validate_geometry(geometry_rows, report)
    _validate_trajectory(trajectory_rows, metrics)
    predictions = _validate_predictions(prediction_rows)
    for rig in FRESH_RIGS:
        for axis, size in (("row", 14), ("column", 11)):
            for index in range(size):
                _close(
                    predictions[(rig, "exact_oracle", axis, index)],
                    predictions[(rig, "exact_factor_interpolation_oracle", axis, index)],
                    f"{rig} exact duplicate prediction",
                )
    _validate_model(model_rows, report)
    _validate_report_arithmetic(report, metrics)
    return report, metrics, predictions, manifest


def _gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def derive_public_tables(
    report: Mapping[str, Any],
    metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    predictions: Mapping[tuple[str, str, str, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mass_stats: dict[str, dict[str, float]] = {}
    for method in METHODS:
        ratios: list[float] = []
        for rig in FRESH_RIGS:
            for axis, size in (("row", 14), ("column", 11)):
                for index in range(size):
                    exact = predictions[(rig, "exact_oracle", axis, index)]
                    ratios.append(predictions[(rig, method, axis, index)] / exact)
        array = np.asarray(ratios, dtype=float)
        mass_stats[method] = {
            "mass_entry_count": int(array.size),
            "mass_coverage_fraction": float(np.mean(array >= 1.0 - 1e-12)),
            "mass_ratio_median_vs_exact": float(np.median(array)),
            "mass_ratio_p95_vs_exact": float(np.percentile(array, 95)),
            "mass_ratio_mean_vs_exact": float(np.mean(array)),
        }

    timings = report["timing"]["fresh_method_seconds"]
    ledgers = report["call_ledger"]["fresh_by_method"]
    method_summary: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [metrics[(rig, method)] for rig in FRESH_RIGS]
        method_summary.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "method_contract": report["method_contracts"][method],
            "duplicate_of": "exact_oracle" if method == "exact_factor_interpolation_oracle" else "",
            "counts_as_independent_evidence": method != "exact_factor_interpolation_oracle",
            "fresh_rig_count": 4,
            "mean_final_normalized_residual_l2": float(np.mean([row["final_normalized_residual_l2"] for row in rows])),
            "mean_final_field_relative_l2": float(np.mean([row["final_field_relative_l2"] for row in rows])),
            "unsafe_fresh_rig_count": sum(row["total_violation_count"] > 0 for row in rows),
            "row_violation_count": sum(row["row_violation_count"] for row in rows),
            "column_violation_count": sum(row["column_violation_count"] for row in rows),
            "spectral_violation_count": sum(row["spectral_violation_count"] for row in rows),
            "total_violation_count": sum(row["total_violation_count"] for row in rows),
            **mass_stats[method],
            "setup_seconds": timings[method]["setup_seconds"],
            "iteration_seconds": timings[method]["iteration_seconds"],
            "audit_seconds": timings[method]["audit_seconds"],
            "total_seconds": timings[method]["total_seconds"],
            "signed_forward_solver_calls": ledgers[method]["signed_forward_solver_calls"],
            "signed_transpose_solver_calls": ledgers[method]["signed_transpose_solver_calls"],
            "setup_exact_mass_materializations": ledgers[method]["setup_exact_mass_materializations"],
            "factor_mass_vector_accesses": ledgers[method]["factor_mass_vector_accesses"],
            "factor_feature_construction_calls": ledgers[method]["factor_feature_construction_calls"],
            "factor_feature_construction_seconds": timings[method]["factor_feature_construction_seconds"],
            "estimator_head_forward_calls": ledgers[method]["estimator_head_forward_calls"],
            "timing_is_single_run_noncomparative": True,
        })

    fresh_rows: list[dict[str, Any]] = []
    for rig in FRESH_RIGS:
        field_order = sorted(METHODS, key=lambda method: metrics[(rig, method)]["final_field_relative_l2"])
        residual_order = sorted(METHODS, key=lambda method: metrics[(rig, method)]["final_normalized_residual_l2"])
        calibrated = metrics[(rig, "calibrated_envelope")]
        factor = metrics[(rig, "factor")]
        scalar = metrics[(rig, "scalar_factor_train_selected")]
        stable_win = calibrated["final_field_relative_l2"] < min(
            factor["final_field_relative_l2"], scalar["final_field_relative_l2"]
        )
        for method in METHODS:
            row = metrics[(rig, method)]
            fresh_rows.append({
                "rig_id": rig,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "duplicate_of": "exact_oracle" if method == "exact_factor_interpolation_oracle" else "",
                "counts_as_independent_evidence": method != "exact_factor_interpolation_oracle",
                "field_rank_lower_is_better": field_order.index(method) + 1,
                "residual_rank_lower_is_better": residual_order.index(method) + 1,
                "final_field_relative_l2": row["final_field_relative_l2"],
                "final_normalized_residual_l2": row["final_normalized_residual_l2"],
                "total_violation_count": row["total_violation_count"],
                "schur_safe": row["total_violation_count"] == 0,
                "calibrated_vs_factor_field_gain_percent": _gain(
                    factor["final_field_relative_l2"], calibrated["final_field_relative_l2"]
                ) if method == "calibrated_envelope" else "",
                "calibrated_vs_scalar_field_gain_percent": _gain(
                    scalar["final_field_relative_l2"], calibrated["final_field_relative_l2"]
                ) if method == "calibrated_envelope" else "",
                "calibrated_beats_factor_and_scalar_on_field": stable_win if method == "calibrated_envelope" else "",
            })

    by_method = {row["method"]: row for row in method_summary}
    calibrated = by_method["calibrated_envelope"]
    factor = by_method["factor"]
    scalar = by_method["scalar_factor_train_selected"]
    stable_wins = sum(
        metrics[(rig, "calibrated_envelope")]["final_field_relative_l2"]
        < min(
            metrics[(rig, "factor")]["final_field_relative_l2"],
            metrics[(rig, "scalar_factor_train_selected")]["final_field_relative_l2"],
        )
        for rig in FRESH_RIGS
    )
    aggregate = {
        "calibrated_vs_factor_mean_field_gain_percent": _gain(
            factor["mean_final_field_relative_l2"], calibrated["mean_final_field_relative_l2"]
        ),
        "calibrated_vs_scalar_mean_field_gain_percent": _gain(
            scalar["mean_final_field_relative_l2"], calibrated["mean_final_field_relative_l2"]
        ),
        "calibrated_stable_field_win_rig_count": stable_wins,
        "calibrated_stable_field_win_denominator": 4,
        "calibrated_unsafe_rig_count": calibrated["unsafe_fresh_rig_count"],
        "calibrated_total_violation_count": calibrated["total_violation_count"],
        "calibrated_field_gain_vs_scalar_percent_by_rig": {
            rig: _gain(
                metrics[(rig, "scalar_factor_train_selected")]["final_field_relative_l2"],
                metrics[(rig, "calibrated_envelope")]["final_field_relative_l2"],
            )
            for rig in FRESH_RIGS
        },
        "raw_learned_mean_field_relative_l2": by_method["learned_oracle_free"]["mean_final_field_relative_l2"],
        "all_methods_total_violation_count": sum(row["total_violation_count"] for row in method_summary),
    }
    gates = [
        {"gate": "source metric substitution authorization", "expected": "false", "observed": "false", "passed": True, "meaning": "V2 NO AUTH"},
        {"gate": "source research claim authorization", "expected": "false", "observed": "false", "passed": True, "meaning": "no paper claim opened"},
        {"gate": "calibrated envelope safe on every fresh rig", "expected": "true", "observed": "false", "passed": False, "meaning": "4/4 fresh rigs unsafe; 39 violations [11,18,1,9]"},
        {"gate": "calibrated envelope beats factor and scalar on every fresh rig", "expected": "true", "observed": "false", "passed": False, "meaning": "2/4 field wins: ood-00 and ood-02"},
        {"gate": "real BOST data used", "expected": "true before deployment claim", "observed": "false", "passed": False, "meaning": "synthetic smoke only"},
        {"gate": "statistical generalization evidence", "expected": "independent repeated study", "observed": "none", "passed": False, "meaning": "no IID claim or statistics"},
        {"gate": "exact-factor alpha=1 duplicate disclosed", "expected": "duplicate of exact oracle", "observed": "byte-level masses and metric rows identical", "passed": True, "meaning": "not an independent evidence unit"},
    ]
    return method_summary, fresh_rows, gates, aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValidationError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    output: Path,
    method_summary: list[dict[str, Any]],
    fresh_rows: list[dict[str, Any]],
) -> None:
    by_method = {row["method"]: row for row in method_summary}
    by_pair = {(row["rig_id"], row["method"]): row for row in fresh_rows}
    colors = {
        "factor": "#4C78A8", "exact_oracle": "#59A14F",
        "scalar_factor_train_selected": "#F28E2B",
        "exact_factor_interpolation_oracle": "#8CD17D",
        "learned_oracle_free": "#E15759", "calibrated_envelope": "#B07AA1",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    fig.suptitle(
        "Metric-A v2 synthetic smoke: four fresh geometry rigs (V2 NO AUTH)\n"
        "metric substitution = false | research claim = false",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    divergent = ("learned_oracle_free", "calibrated_envelope")
    bounded = [method for method in METHODS if method not in divergent]
    x = np.arange(len(bounded))
    means = [by_method[method]["mean_final_field_relative_l2"] for method in bounded]
    ax.bar(x, means, color=[colors[method] for method in bounded], alpha=0.82)
    offsets = np.linspace(-0.14, 0.14, len(FRESH_RIGS))
    for rig_index, rig in enumerate(FRESH_RIGS):
        values = [by_pair[(rig, method)]["final_field_relative_l2"] for method in bounded]
        ax.scatter(x + offsets[rig_index], values, color="#17202A", s=24, alpha=0.72, zorder=3)
    ax.set_xticks(x, [METHOD_LABELS[method] for method in bounded], rotation=18, ha="right")
    ax.set_ylabel("Field relative L2 (lower is better)")
    ax.set_title("A. Field mean + per-rig points; divergent metrics in log inset")
    ax.grid(axis="y", alpha=0.22)
    harm_01 = -_gain(
        by_pair[("ood-01", "scalar_factor_train_selected")]["final_field_relative_l2"],
        by_pair[("ood-01", "calibrated_envelope")]["final_field_relative_l2"],
    )
    harm_03 = -_gain(
        by_pair[("ood-03", "scalar_factor_train_selected")]["final_field_relative_l2"],
        by_pair[("ood-03", "calibrated_envelope")]["final_field_relative_l2"],
    )
    ax.text(
        0.02, 0.78,
        f"Wins vs factor+scalar: ood-00, ood-02\nHarms vs scalar: ood-01 +{harm_01:.2e}% | ood-03 +{harm_03:.1f}%\nMeans: calibrated 1.517e5 | raw learned 2.180e26",
        transform=ax.transAxes, va="top", fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": "#B03A2E", "alpha": 0.92, "pad": 3},
    )
    inset = ax.inset_axes([0.70, 0.38, 0.28, 0.46])
    positions = np.arange(4)
    width = 0.36
    learned_values = [by_pair[(rig, "learned_oracle_free")]["final_field_relative_l2"] for rig in FRESH_RIGS]
    calibrated_values = [by_pair[(rig, "calibrated_envelope")]["final_field_relative_l2"] for rig in FRESH_RIGS]
    inset.bar(positions - width / 2, learned_values, width=width, color=colors["learned_oracle_free"], label="raw")
    inset.bar(positions + width / 2, calibrated_values, width=width, color=colors["calibrated_envelope"], label="cal")
    inset.set_yscale("log")
    inset.set_xticks(positions, [rig[-2:] for rig in FRESH_RIGS])
    inset.set_title("Divergent learned metrics\n(log scale)", fontsize=9)
    inset.legend(frameon=False, fontsize=6, ncol=2)
    inset.tick_params(labelsize=7)
    inset.grid(axis="y", alpha=0.18)

    ax = axes[0, 1]
    x = np.arange(len(METHODS))
    row_v = np.asarray([by_method[method]["row_violation_count"] for method in METHODS])
    col_v = np.asarray([by_method[method]["column_violation_count"] for method in METHODS])
    spec_v = np.asarray([by_method[method]["spectral_violation_count"] for method in METHODS])
    ax.bar(x, row_v, label="row", color="#4C78A8")
    ax.bar(x, col_v, bottom=row_v, label="column", color="#F28E2B")
    ax.bar(x, spec_v, bottom=row_v + col_v, label="spectral", color="#E15759")
    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Violation count across four rigs")
    ax.set_title("B. Signed-A Schur safety audit")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    ax.text(
        0.02, 0.72, "Calibrated: 39 violations [11,18,1,9]; 4/4 fresh rigs unsafe",
        transform=ax.transAxes, fontsize=9, color="#922B21",
        bbox={"facecolor": "white", "edgecolor": "#B03A2E", "alpha": 0.92, "pad": 3},
    )

    ax = axes[1, 0]
    coverage = np.asarray([100.0 * by_method[method]["mass_coverage_fraction"] for method in METHODS])
    median_ratio = np.asarray([by_method[method]["mass_ratio_median_vs_exact"] for method in METHODS])
    ax.bar(x, coverage, color=[colors[method] for method in METHODS], alpha=0.8)
    ax.axhline(100.0, color="#222222", linewidth=1, linestyle="--")
    ax.set_ylabel("Entries >= exact mass (%)")
    ax.set_ylim(0, 112)
    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=18, ha="right")
    twin = ax.twinx()
    twin.plot(x, median_ratio, color="#111111", marker="o", linewidth=1.5, label="median mass / exact")
    twin.axhline(1.0, color="#666666", linewidth=1, linestyle=":")
    twin.set_yscale("log")
    twin.set_ylabel("Median mass / exact (log scale)")
    ax.set_title("C. Mass coverage and conservatism")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1, 1]
    setup = 1000 * np.asarray([by_method[method]["setup_seconds"] for method in METHODS])
    iteration = 1000 * np.asarray([by_method[method]["iteration_seconds"] for method in METHODS])
    audit = 1000 * np.asarray([by_method[method]["audit_seconds"] for method in METHODS])
    ax.bar(x, setup, label="setup", color="#76B7B2")
    ax.bar(x, iteration, bottom=setup, label="32 iterations", color="#4E79A7")
    ax.bar(x, audit, bottom=setup + iteration, label="audit", color="#BAB0AC")
    for index, method in enumerate(METHODS):
        row = by_method[method]
        annotation = (
            f"A/AT {row['signed_forward_solver_calls']}/{row['signed_transpose_solver_calls']}\n"
            f"E/F/H {row['setup_exact_mass_materializations']}/{row['factor_mass_vector_accesses']}/{row['estimator_head_forward_calls']}"
        )
        ax.text(index, setup[index] + iteration[index] + audit[index] + 0.025, annotation, ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Measured time (ms)")
    ax.set_title("D. Cost/call ledger (single-run timing; not speed evidence)")
    ax.set_ylim(0, 1.28 * float(np.max(setup + iteration + audit)))
    ax.text(
        0.99, 0.02,
        "E/F/H = exact setups / factor-vector accesses / estimator heads; shared factor majorizer mats = 15",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.22)

    fig.text(
        0.5,
        0.002,
        "Synthetic interface smoke; 4 designed fresh rigs; no IID claim, no statistics, no real-data/generalization/superiority claim; not a new algorithm.",
        ha="center",
        fontsize=9.5,
        color="#7B241C",
    )
    png_path = output / "diagnostic.png"
    fig.savefig(
        png_path, dpi=180, bbox_inches="tight",
        facecolor="white", transparent=False,
    )
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path)
    fig.savefig(
        output / "diagnostic.pdf",
        bbox_inches="tight", facecolor="white", transparent=False,
        metadata={"Creator": "OERF Metric-A public analyzer", "CreationDate": None},
    )
    plt.close(fig)


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = summary["aggregate_result"]
    method_rows = summary["methods"]
    lines = [
        "# Metric-A v2 public analysis slice",
        "",
        "**V2 NO AUTH.** This is a checksum-frozen analysis of a tiny synthetic signed-matrix interface smoke. It is not a new algorithm, a real BOST result, a generalization result, or a superiority claim.",
        "",
        "## What was recomputed",
        "",
        "- Four complete fresh geometry-OOD rigs and six fixed methods were read from the frozen source CSV files.",
        "- Residual means, field-relative-L2 means, per-rig ranks, unsafe-rig counts, Schur violation counts, mass coverage, mass conservatism, and call/cost ledgers were recomputed.",
        "- Mean summaries and per-rig outcomes are separate. The four rigs are not treated as IID samples; no confidence interval, p-value, or significance claim is produced.",
        "- Provenance is `COMMITTED_CLEAN_REPRODUCIBLE_FROM_COMMIT`; the runner captured Git state before writing tracked outputs.",
        "",
        "## Central result",
        "",
        f"- Calibrated envelope mean field gain vs factor: **{aggregate['calibrated_vs_factor_mean_field_gain_percent']:.3e}%** (negative means harm).",
        f"- Calibrated envelope mean field gain vs train-selected scalar-factor control: **{aggregate['calibrated_vs_scalar_mean_field_gain_percent']:.3e}%** (negative means harm).",
        f"- Per-rig stable field wins against both controls: **{aggregate['calibrated_stable_field_win_rig_count']}/4**.",
        f"- Calibrated envelope unsafe rigs: **{aggregate['calibrated_unsafe_rig_count']}/4**, with **{aggregate['calibrated_total_violation_count']}** total Schur violations.",
        f"- Calibrated wins versus factor and scalar on **ood-00** and **ood-02**, but harms the scalar control on **ood-01 by {-aggregate['calibrated_field_gain_vs_scalar_percent_by_rig']['ood-01']:.3e}%** and **ood-03 by {-aggregate['calibrated_field_gain_vs_scalar_percent_by_rig']['ood-03']:.2f}%** field error.",
        "- Both source authorization decisions remain false.",
        "- The selected exact-factor interpolation has `alpha=1` and is exactly identical to the exact oracle; it is a disclosed duplicate control, not an independent evidence unit.",
        "- Learned and calibrated inference each consume the factor row/column features: 8 factor-vector accesses and 4 feature-construction calls over four fresh rigs. The shared synthetic bundle records 15 factor-majorizer materializations; no end-to-end cost reduction is claimed.",
        "",
        "## Method means and safety",
        "",
        "| Method | Mean residual | Mean field L2 | Unsafe rigs | Violations | Mass coverage | Median mass/exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            "| {method_label} | {mean_final_normalized_residual_l2:.6g} | {mean_final_field_relative_l2:.6g} | {unsafe_fresh_rig_count}/4 | {total_violation_count} | {coverage:.1f}% | {ratio:.4g} |".format(
                coverage=100 * row["mass_coverage_fraction"],
                ratio=row["mass_ratio_median_vs_exact"],
                **row,
            )
        )
    lines += [
        "",
        "## Files",
        "",
        "- `summary.json`: machine-readable evidence boundary and recomputed aggregate.",
        "- `method_summary.csv`: method means, safety, mass, timing, and call accounting.",
        "- `fresh_rig_comparison.csv`: long-form per-rig values and rankings.",
        "- `decision_gates.csv`: passed and failed scientific gates.",
        "- `diagnostic.png` / `diagnostic.pdf`: four-panel visual audit.",
        "- `checksums.sha256`: hashes of every generated public artifact except the manifest itself.",
        "",
        "## Interpretation boundary",
        "",
        "The raw learned estimator diverges strongly and the calibrated envelope has one extreme fresh-rig failure, so both are isolated in a log-scale inset. The calibrated envelope is unsafe on every fresh rig, with per-rig violations [11, 18, 1, 9], and beats both deployable controls on only two of four rigs. It is a strict negative result, not authorization to substitute the metric.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_checksums(output: Path) -> None:
    lines = [f"{_sha256(output / name)}  {name}\n" for name in sorted(PUBLIC_GENERATED_FILES)]
    (output / "checksums.sha256").write_text("".join(lines), encoding="ascii")


def _validate_public_bundle(output: Path) -> None:
    observed = {path.name for path in output.iterdir() if path.is_file()}
    if observed != PUBLIC_FILES or any(path.is_symlink() for path in output.iterdir()):
        raise ValidationError("generated public file set drift")
    manifest: dict[str, str] = {}
    for line in (output / "checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in manifest:
            raise ValidationError("malformed public checksum manifest")
        manifest[name] = digest
    if set(manifest) != PUBLIC_GENERATED_FILES:
        raise ValidationError("public checksum file set drift")
    for name, digest in manifest.items():
        if _sha256(output / name) != digest:
            raise ValidationError(f"public checksum mismatch: {name}")


def run(input_root: Path = DEFAULT_INPUT, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report, metrics, predictions, source_manifest = load_release(input_root)
    if output_root.is_symlink():
        raise ValidationError("unsafe public output root")
    output_root.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_root.iterdir()}
    if not existing.issubset(PUBLIC_FILES):
        raise ValidationError("unexpected stale public files")
    if any(path.is_symlink() or not path.is_file() for path in output_root.iterdir()):
        raise ValidationError("unsafe existing public entry")

    method_summary, fresh_rows, gates, aggregate = derive_public_tables(
        report, metrics, predictions
    )
    summary: dict[str, Any] = {
        "schema_version": PUBLIC_SCHEMA,
        "title": "Metric-A v2 cancellation-aware metric surrogate synthetic smoke",
        "status": "V2_NO_AUTH",
        "source_evidence_scope": EVIDENCE_SCOPE,
        "source_status": STATUS,
        "source_sha256": source_manifest,
        "source_provenance": {
            "source_commit": report["provenance"]["source_commit"],
            "source_snapshot_status": report["provenance"]["source_snapshot_status"],
            "source_tree_clean": True,
            "clean_rerun_required_after_commit": False,
        },
        "source_decision": {
            "metric_substitution_authorized": False,
            "research_claim_authorized": False,
        },
        "cost_contract": {
            "learned_features_require_factor_row_and_column_mass": True,
            "shared_factor_majorizer_materializations": report["call_ledger"]["data_generation"]["factor_majorizer_materializations"],
            "learned_factor_mass_vector_accesses": report["call_ledger"]["fresh_by_method"]["learned_oracle_free"]["factor_mass_vector_accesses"],
            "calibrated_factor_mass_vector_accesses": report["call_ledger"]["fresh_by_method"]["calibrated_envelope"]["factor_mass_vector_accesses"],
            "learned_factor_feature_construction_calls": report["call_ledger"]["fresh_by_method"]["learned_oracle_free"]["factor_feature_construction_calls"],
            "calibrated_factor_feature_construction_calls": report["call_ledger"]["fresh_by_method"]["calibrated_envelope"]["factor_feature_construction_calls"],
            "end_to_end_cost_reduction_claimed": False,
            "timing_role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE",
        },
        "claim_boundary": {
            **EXPECTED_CLAIM_BOUNDARY,
            "synthetic_smoke_only": True,
            "fresh_geometry_rig_count": 4,
            "iid_sample_claimed": False,
            "statistical_inference_performed": False,
            "metric_substitution_authorized": False,
            "research_claim_authorized": False,
        },
        "statistical_contract": {
            "unit": "four designed complete fresh geometry rigs",
            "mean_reported": True,
            "per_rig_reported": True,
            "iid_claimed": False,
            "confidence_intervals_reported": False,
            "p_values_reported": False,
            "significance_claimed": False,
        },
        "aggregate_result": aggregate,
        "methods": method_summary,
        "figure_contract": {
            "field_mean_and_per_rig_are_both_visible": True,
            "raw_and_calibrated_divergence_isolated_on_log_scale": True,
            "safety_violations_visible": True,
            "mass_coverage_and_conservatism_visible": True,
            "timing_role": "MEASURED_SINGLE_RUN_NONCOMPARATIVE",
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_root / "method_summary.csv", method_summary)
    _write_csv(output_root / "fresh_rig_comparison.csv", fresh_rows)
    _write_csv(output_root / "decision_gates.csv", gates)
    _plot(output_root, method_summary, fresh_rows)
    _write_readme(output_root / "README.md", summary)
    _write_checksums(output_root)
    _validate_public_bundle(output_root)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run(args.input, args.output)
    result = summary["aggregate_result"]
    print(
        "PASS_METRIC_A_V2_PUBLIC_ANALYSIS "
        f"factor_gain={result['calibrated_vs_factor_mean_field_gain_percent']:.6f}% "
        f"scalar_gain={result['calibrated_vs_scalar_mean_field_gain_percent']:.6f}% "
        f"stable_wins={result['calibrated_stable_field_win_rig_count']}/4 "
        f"unsafe={result['calibrated_unsafe_rig_count']}/4 V2_NO_AUTH"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
