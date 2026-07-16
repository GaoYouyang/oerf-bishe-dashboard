#!/usr/bin/env python3
"""Independently validate the v5a first-open calibration artifacts.

Validation success means that the on-disk evidence is internally consistent.
The scientific claim gate may legitimately be failed, provided the report says
so and its five gate decisions agree with independently recomputed metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5a_blind_aperture_calibration.json"
DEFAULT_RESULT = ROOT / "results" / "v5a_blind_aperture_calibration"
SPLITS = ("independent_select", "independent_lock")
BASELINES = (
    "pinhole_pbb_4",
    "pinhole_pbb_equal_calls",
    "pinhole_fista_equal_calls",
)
METRICS = (
    "relative_l2",
    "gradient_relative_l2",
    "front_f1",
    "mass_relative_error",
    "centroid_error",
    "nominal_pinhole_residual_rms",
    "audit_true_operator_residual_rms",
)
SUMMARY_FIELDS = (
    "candidate_mean_relative_l2",
    "baseline_mean_relative_l2",
    "mean_gain_percent",
    "p10_gain_percent",
    "harm_rate_over_1_percent",
    "coverage",
    "candidate_mean_gradient_relative_l2",
    "baseline_mean_gradient_relative_l2",
    "candidate_mean_front_f1",
    "baseline_mean_front_f1",
    "candidate_mean_mass_relative_error",
    "baseline_mean_mass_relative_error",
    "candidate_mean_centroid_error",
    "baseline_mean_centroid_error",
    "candidate_mean_audit_true_residual_rms",
    "baseline_mean_audit_true_residual_rms",
    "audit_reprojection_change_percent",
)
EXPECTED_ASSETS = {
    "config_snapshot.json",
    "selection_commit.json",
    "selection_calibration.csv",
    "sample_metrics.csv",
    "summary.csv",
    "operator_manifest.json",
    "v5a_blind_aperture_calibration.png",
    "report.json",
}
ADJOINT_ERROR_LIMIT = 1e-5
PASSED_CLAIM_STATUS = (
    "SYNTHETIC_WEAK_BOST_BLIND_APERTURE_GATE_PASSED_"
    "REAL_OPTICS_AND_LEARNED_BASELINES_REQUIRED"
)
FAILED_CLAIM_STATUS = "SYNTHETIC_WEAK_BOST_BLIND_APERTURE_GATE_FAILED_OR_INCOMPLETE"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ValidationError(AssertionError):
    """Raised when an evidence consistency check fails."""


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
        rel_tol: float = 3e-6,
        abs_tol: float = 3e-7,
    ) -> None:
        try:
            actual_value = float(actual)
            expected_value = float(expected)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{message}: non-numeric value") from exc
        self.require(
            math.isfinite(actual_value) and math.isfinite(expected_value),
            f"{message}: non-finite value",
        )
        self.require(
            math.isclose(
                actual_value,
                expected_value,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ),
            f"{message}: {actual_value} != {expected_value}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(validator: Validator, row: dict[str, str], key: str) -> float:
    validator.require(key in row, f"missing numeric CSV field: {key}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid numeric CSV field: {key}") from exc
    validator.require(math.isfinite(value), f"non-finite CSV field: {key}")
    return value


def boolean(validator: Validator, value: Any, message: str) -> bool:
    validator.require(
        value in {"True", "False"}, f"{message}: invalid boolean {value!r}"
    )
    return value == "True"


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValidationError("cannot average an empty sequence")
    return math.fsum(items) / len(items)


def quantile(values: Iterable[float], q: float) -> float:
    """Match numpy's default one-dimensional linear quantile."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValidationError("cannot quantile an empty sequence")
    position = (len(ordered) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def threshold_candidates(confidence: list[float], count: int) -> list[float]:
    values = sorted(
        set(quantile(confidence, index / (count - 1)) for index in range(count))
    )
    minimum, maximum = min(confidence), max(confidence)
    lower = minimum - max(1e-9, abs(minimum) * 1e-6)
    upper = maximum + max(1e-9, abs(maximum) * 1e-6)
    return [lower, *values, upper]


def expected_sample_header(baseline: str, method: str) -> list[str]:
    fields = [
        "split",
        "sample_index",
        "geometry_id",
        "family",
        "calibration_id",
        "true_aperture_radius",
        "active_reconstruction_views",
        "accepted",
        "selected_method",
        "selected_baseline",
        "confidence",
        "estimated_aperture_radius",
    ]
    names = list(
        dict.fromkeys((baseline, method, "selected_candidate", "oracle_true_operator"))
    )
    fields.extend(f"{name}_{metric}" for name in names for metric in METRICS)
    fields.append("selected_gain_percent")
    return fields


def validate_checksums(validator: Validator, result_dir: Path) -> None:
    checksum_path = result_dir / "checksums.sha256"
    lines = [
        line for line in checksum_path.read_text(encoding="utf-8").splitlines() if line
    ]
    validator.require(len(lines) == len(EXPECTED_ASSETS), "checksum target count")
    recorded: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        validator.require(len(parts) == 2, "malformed checksum line")
        expected, name = parts
        validator.require(
            bool(SHA256_PATTERN.fullmatch(expected)), f"invalid digest: {name}"
        )
        validator.require(Path(name).name == name, f"unsafe checksum target: {name}")
        validator.require(name not in recorded, f"duplicate checksum target: {name}")
        path = result_dir / name
        validator.require(path.is_file(), f"missing checksum target: {name}")
        validator.require(sha256(path) == expected, f"checksum mismatch: {name}")
        recorded[name] = expected
    validator.require(set(recorded) == EXPECTED_ASSETS, "checksum asset set mismatch")
    actual_files = {path.name for path in result_dir.iterdir() if path.is_file()}
    validator.require(
        EXPECTED_ASSETS | {"checksums.sha256"} <= actual_files,
        "missing first-open result files",
    )


def validate_hash_chain(
    validator: Validator,
    config_path: Path,
    config: dict[str, Any],
    result_dir: Path,
    snapshot: dict[str, Any],
    commit: dict[str, Any],
    report: dict[str, Any],
) -> None:
    validator.require(snapshot == config, "config snapshot differs from current config")
    validator.require(report["config"] == snapshot, "report embeds a different config")
    validator.require(
        report["evidence_label"] == config["evidence_label"],
        "report evidence label differs from config",
    )
    validator.require(
        commit["config_sha256"] == sha256(result_dir / "config_snapshot.json"),
        "selection commit config hash mismatch",
    )
    validator.require(
        report["selection_commit_sha256"]
        == sha256(result_dir / "selection_commit.json"),
        "report selection commit hash mismatch",
    )
    validator.require(
        report["selection"] == commit["selected_operator_method"],
        "report selection differs from durable commit",
    )
    validator.require(
        report["selected_baseline"] == commit["selected_baseline"],
        "report baseline differs from durable commit",
    )
    validator.require(
        commit["created_before_independent_lock"] is True,
        "selection commit is not marked pre-lock",
    )
    validator.require(
        report["lock_status"].startswith("FIRST_OPEN;"),
        "report is not marked FIRST_OPEN",
    )

    sources = {
        "runner": ROOT / "run_v5a_blind_aperture_calibration.py",
        "finite_aperture_generator": ROOT / "finite_aperture_bost.py",
        "measurement_contract": ROOT / "measurement_contract.py",
    }
    validator.require(
        set(report["source_sha256"]) == set(sources), "source hash key set"
    )
    for label, source in sources.items():
        validator.require(source.is_file(), f"missing source for hash check: {label}")
        validator.require(
            report["source_sha256"][label] == sha256(source),
            f"source hash mismatch: {label}",
        )
    validator.require(config_path.is_file(), "requested config path is missing")


def validate_operator_manifests(
    validator: Validator,
    config: dict[str, Any],
    commit: dict[str, Any],
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> None:
    validator.require(
        report["operator_audit"]["select"] == manifest["select"],
        "report/select operator manifest mismatch",
    )
    validator.require(
        report["operator_audit"]["lock"] == manifest["lock"],
        "report/lock operator manifest mismatch",
    )
    validator.require(
        commit["select_operator_manifest"] == manifest["select"],
        "commit/select operator manifest mismatch",
    )

    reconstruction_sets: dict[str, set[str]] = {}
    truth_sets: dict[str, set[str]] = {}
    for split, short in (
        ("independent_select", "select"),
        ("independent_lock", "lock"),
    ):
        item = manifest[short]
        rig = config["independent_rigs"][split]
        validator.require(item["split"] == split, f"operator split label: {split}")
        validator.require(
            item["candidate_aperture_radii"] == config["candidate_aperture_radii"],
            f"candidate aperture radii: {split}",
        )
        validator.require(
            item["true_aperture_radii"] == rig["true_aperture_radii"],
            f"true aperture radii: {split}",
        )
        reconstruction = item["reconstruction_operator_sha256"]
        truth = item["truth_operator_sha256"]
        validator.require(
            len(reconstruction) == len(config["candidate_aperture_radii"]),
            f"reconstruction operator count: {split}",
        )
        validator.require(
            len(truth) == len(rig["true_aperture_radii"]),
            f"truth operator count: {split}",
        )
        for digest in [*reconstruction, *truth]:
            validator.require(
                isinstance(digest, str) and bool(SHA256_PATTERN.fullmatch(digest)),
                f"invalid operator digest: {split}",
            )
        reconstruction_set, truth_set = set(reconstruction), set(truth)
        validator.require(
            len(reconstruction_set) == len(reconstruction),
            f"duplicate reconstruction operator hash: {split}",
        )
        validator.require(
            len(truth_set) == len(truth),
            f"duplicate truth operator hash: {split}",
        )
        overlap = sorted(reconstruction_set & truth_set)
        validator.require(
            not overlap, f"truth/reconstruction operator overlap: {split}"
        )
        validator.require(
            item["truth_reconstruction_hash_overlap"] == overlap,
            f"reported truth/reconstruction overlap: {split}",
        )
        validator.require(
            item["audit_camera_index"] == rig["audit_camera_index"],
            f"audit camera index: {split}",
        )
        validator.require(
            item["audit_camera_excluded_from_noise_scale"] is True,
            f"audit camera noise-scale declaration: {split}",
        )
        reconstruction_sets[split] = reconstruction_set
        truth_sets[split] = truth_set

    all_reconstruction = set().union(*reconstruction_sets.values())
    all_truth = set().union(*truth_sets.values())
    validator.require(
        all_reconstruction.isdisjoint(all_truth),
        "global truth/reconstruction operator hash overlap",
    )
    validator.require(
        reconstruction_sets["independent_select"].isdisjoint(
            reconstruction_sets["independent_lock"]
        ),
        "select/lock reconstruction operator hash overlap",
    )
    validator.require(
        truth_sets["independent_select"].isdisjoint(truth_sets["independent_lock"]),
        "select/lock truth operator hash overlap",
    )
    validator.require(
        report["operator_audit"]["truth_reconstruction_operator_equal"] is False,
        "report marks truth and reconstruction operators equal",
    )


def validate_sample_provenance(
    validator: Validator,
    config: dict[str, Any],
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
    commit: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    validator.require(
        len(rows) == sum(int(config["counts"][split]) for split in SPLITS),
        "sample row count",
    )
    validator.require({row["split"] for row in rows} == set(SPLITS), "sample split set")
    by_split = {
        split: [row for row in rows if row["split"] == split] for split in SPLITS
    }
    geometry_sets: dict[str, set[str]] = {}
    calibration_sets: dict[str, set[str]] = {}
    observed_families: dict[str, set[str]] = {}
    observed_radii: dict[str, set[float]] = {}

    for split, short in (
        ("independent_select", "select"),
        ("independent_lock", "lock"),
    ):
        split_rows = by_split[split]
        expected_count = int(config["counts"][split])
        rig = config["independent_rigs"][split]
        audit_camera = int(rig["audit_camera_index"])
        view_count = len(rig["angles_degrees"])
        validator.require(len(split_rows) == expected_count, f"sample count: {split}")
        validator.require(
            len(config["camera_noise_factors"]) == view_count,
            f"camera noise factor count: {split}",
        )
        validator.require(
            0 <= audit_camera < view_count,
            f"audit camera outside rig: {split}",
        )
        indices = [int(number(validator, row, "sample_index")) for row in split_rows]
        validator.require(
            indices == list(range(expected_count)), f"sample index order: {split}"
        )

        geometry_ids: list[str] = []
        calibration_ids: list[str] = []
        masks: list[str] = []
        families: list[str] = []
        radii: list[float] = []
        allowed_radii = {float(value) for value in rig["true_aperture_radii"]}
        for row in split_rows:
            match = re.fullmatch(
                rf"{re.escape(split)}:m([01]+):field=(\d{{3}})", row["geometry_id"]
            )
            validator.require(
                match is not None, f"malformed geometry id: {row['geometry_id']}"
            )
            assert match is not None
            mask, field_index = match.groups()
            validator.require(
                len(mask) == view_count, f"view mask width: {row['geometry_id']}"
            )
            validator.require(
                mask[audit_camera] == "0",
                f"audit camera is active: {row['geometry_id']}",
            )
            active = int(number(validator, row, "active_reconstruction_views"))
            validator.require(
                active == mask.count("1"), f"active view count: {row['geometry_id']}"
            )
            validator.require(
                active in {int(value) for value in config["active_views"][split]},
                f"active view budget outside config: {row['geometry_id']}",
            )
            validator.require(
                int(field_index) == int(number(validator, row, "sample_index")),
                f"geometry/sample index mismatch: {row['geometry_id']}",
            )
            radius = number(validator, row, "true_aperture_radius")
            validator.require(
                radius in allowed_radii, f"true aperture outside config: {split}"
            )
            expected_calibration = f"{split}:aperture={radius:.5f}"
            validator.require(
                row["calibration_id"] == expected_calibration,
                f"calibration id/radius mismatch: {row['geometry_id']}",
            )
            validator.require(
                row["family"] in set(config["families"][split]),
                f"family outside config: {row['family']}",
            )
            geometry_ids.append(row["geometry_id"])
            calibration_ids.append(row["calibration_id"])
            masks.append(mask)
            families.append(row["family"])
            radii.append(radius)

        validator.require(
            len(set(geometry_ids)) == expected_count, f"duplicate geometry id: {split}"
        )
        validator.require(
            len(set(masks)) == expected_count, f"duplicate reconstruction mask: {split}"
        )
        validator.require(
            set(families) == set(config["families"][split]),
            f"observed family set: {split}",
        )
        validator.require(
            set(radii) == allowed_radii, f"observed calibration radii: {split}"
        )
        geometry_sets[split] = set(geometry_ids)
        calibration_sets[split] = set(calibration_ids)
        observed_families[split] = set(families)
        observed_radii[split] = set(radii)

    validator.require(
        observed_families["independent_select"].isdisjoint(
            observed_families["independent_lock"]
        ),
        "select/lock field family overlap",
    )
    validator.require(
        observed_radii["independent_select"].isdisjoint(
            observed_radii["independent_lock"]
        ),
        "select/lock true calibration-radius overlap",
    )
    geometry_overlap = sorted(
        geometry_sets["independent_select"] & geometry_sets["independent_lock"]
    )
    calibration_overlap = sorted(
        calibration_sets["independent_select"] & calibration_sets["independent_lock"]
    )
    validator.require(not geometry_overlap, "select/lock geometry overlap")
    validator.require(not calibration_overlap, "select/lock calibration id overlap")
    validator.require(
        manifest["select_lock_geometry_id_overlap"] == geometry_overlap,
        "manifest geometry-overlap declaration",
    )
    validator.require(
        manifest["select_lock_calibration_id_overlap"] == calibration_overlap,
        "manifest calibration-overlap declaration",
    )

    audit = report["operator_audit"]
    validator.require(
        commit["audit_camera_used_by_reconstruction_or_selection"] is False,
        "commit says audit camera was used by reconstruction/selection",
    )
    validator.require(
        audit["audit_camera_used_by_reconstruction_or_selection"] is False,
        "report says audit camera was used by reconstruction/selection",
    )
    validator.require(
        commit["audit_camera_excluded_from_noise_scale"] is True,
        "commit does not exclude audit camera from noise scaling",
    )
    validator.require(
        audit["audit_camera_excluded_from_noise_scale"] is True,
        "report does not exclude audit camera from noise scaling",
    )
    return by_split


def validate_baseline(
    validator: Validator,
    commit: dict[str, Any],
    report: dict[str, Any],
    select_rows: list[dict[str, str]],
) -> str:
    means = commit["baseline_mean_relative_l2"]
    validator.require(set(means) == set(BASELINES), "baseline mean key set")
    for name in BASELINES:
        validator.require(
            math.isfinite(float(means[name])) and float(means[name]) >= 0.0,
            f"invalid baseline mean: {name}",
        )
    selected = min(
        BASELINES, key=lambda name: (float(means[name]), BASELINES.index(name))
    )
    validator.require(
        commit["selected_baseline"] == selected, "strongest select baseline"
    )
    validator.require(
        report["selected_baseline"] == selected, "report selected baseline"
    )
    recomputed = mean(
        number(validator, row, f"{selected}_relative_l2") for row in select_rows
    )
    validator.close(means[selected], recomputed, "selected baseline mean")
    return selected


def validate_calibration(
    validator: Validator,
    config: dict[str, Any],
    rows: list[dict[str, str]],
    select_rows: list[dict[str, str]],
    selection: dict[str, Any],
    baseline: str,
) -> None:
    expected_methods = ["cv_hard", "full_residual_hard", "uniform_operator_mean"]
    expected_methods.extend(
        f"cv_soft_t{float(value):g}"
        for value in config["algorithm"]["soft_temperatures"]
    )
    validator.require(bool(rows), "empty selection calibration CSV")
    validator.require(
        {row["method"] for row in rows} == set(expected_methods),
        "calibration method set",
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    observed_order: list[str] = []
    for row in rows:
        if row["method"] not in grouped:
            observed_order.append(row["method"])
        grouped[row["method"]].append(row)
    validator.require(observed_order == expected_methods, "calibration method order")

    constraints = config["selection_gate"]
    feasible_rows: list[dict[str, str]] = []
    for method in expected_methods:
        method_rows = grouped[method]
        thresholds = [number(validator, row, "threshold") for row in method_rows]
        validator.require(
            thresholds == sorted(thresholds), f"threshold order: {method}"
        )
        validator.require(
            3 <= len(method_rows) <= int(constraints["quantile_count"]) + 2,
            f"threshold count: {method}",
        )
        for row in method_rows:
            coverage = number(validator, row, "coverage")
            p10 = number(validator, row, "p10_gain_percent")
            harm = number(validator, row, "harm_rate_over_1_percent")
            number(validator, row, "mean_gain_percent")
            validator.require(0.0 <= coverage <= 1.0, f"coverage range: {method}")
            validator.require(0.0 <= harm <= 1.0, f"harm range: {method}")
            expected_feasible = (
                coverage >= float(constraints["minimum_coverage"])
                and p10 >= float(constraints["minimum_p10_gain_percent"])
                and harm <= float(constraints["maximum_harm_rate_over_1_percent"])
            )
            stored_feasible = boolean(
                validator, row["feasible"], f"feasible flag: {method}"
            )
            validator.require(
                stored_feasible is expected_feasible, f"feasibility formula: {method}"
            )
            if stored_feasible:
                feasible_rows.append(row)

    method = str(selection["method"])
    validator.require(
        method in grouped, "selected method is absent from calibration CSV"
    )
    confidence = [number(validator, row, "confidence") for row in select_rows]
    expected_thresholds = threshold_candidates(
        confidence, int(constraints["quantile_count"])
    )
    selected_method_rows = grouped[method]
    validator.require(
        len(selected_method_rows) == len(expected_thresholds),
        "selected-method threshold candidate count",
    )
    for calibration_row, expected_threshold in zip(
        selected_method_rows, expected_thresholds
    ):
        threshold = number(validator, calibration_row, "threshold")
        validator.close(
            threshold, expected_threshold, "selected-method threshold candidate"
        )
        accepted = [value >= threshold for value in confidence]
        gains = []
        for sample, is_accepted in zip(select_rows, accepted):
            baseline_error = number(validator, sample, f"{baseline}_relative_l2")
            raw_error = number(validator, sample, f"{method}_relative_l2")
            candidate_error = raw_error if is_accepted else baseline_error
            gains.append(
                100.0 * (baseline_error - candidate_error) / max(baseline_error, 1e-12)
            )
        expected_values = {
            "coverage": mean(float(value) for value in accepted),
            "mean_gain_percent": mean(gains),
            "p10_gain_percent": quantile(gains, 0.10),
            "harm_rate_over_1_percent": mean(float(value < -1.0) for value in gains),
        }
        for key, value in expected_values.items():
            validator.close(
                calibration_row[key], value, f"selected-method calibration {key}"
            )

    if feasible_rows:
        chosen = max(
            feasible_rows,
            key=lambda row: (
                float(row["mean_gain_percent"]),
                float(row["coverage"]),
                -float(row["harm_rate_over_1_percent"]),
            ),
        )
        expected_reason = (
            "best_select_gain_subject_to_predeclared_coverage_and_tail_constraints"
        )
        validator.require(
            selection["method"] == chosen["method"], "selected calibration method"
        )
        for key in (
            "threshold",
            "coverage",
            "mean_gain_percent",
            "p10_gain_percent",
            "harm_rate_over_1_percent",
        ):
            validator.close(selection[key], chosen[key], f"committed selection {key}")
        validator.require(
            selection["feasible"] is True, "committed feasible selection flag"
        )
    else:
        expected_reason = "no_feasible_operator_selector_abstain_all"
        validator.require(selection["feasible"] is False, "committed abstention flag")
        validator.close(selection["coverage"], 0.0, "committed abstention coverage")
    validator.require(
        selection["selection_reason"] == expected_reason, "selection reason"
    )


def validate_sample_selection(
    validator: Validator,
    config: dict[str, Any],
    by_split: dict[str, list[dict[str, str]]],
    selection: dict[str, Any],
    baseline: str,
) -> None:
    method = str(selection["method"])
    threshold = float(selection["threshold"])
    candidate_radii = [float(value) for value in config["candidate_aperture_radii"]]
    for split, rows in by_split.items():
        for row in rows:
            validator.require(
                row["selected_method"] == method, f"sample method: {split}"
            )
            validator.require(
                row["selected_baseline"] == baseline, f"sample baseline: {split}"
            )
            confidence = number(validator, row, "confidence")
            expected_accepted = confidence >= threshold
            accepted = boolean(
                validator, row["accepted"], f"accepted flag: {row['geometry_id']}"
            )
            validator.require(
                accepted is expected_accepted,
                f"threshold application: {row['geometry_id']}",
            )
            estimated_radius = number(validator, row, "estimated_aperture_radius")
            validator.require(
                min(candidate_radii) <= estimated_radius <= max(candidate_radii),
                f"estimated aperture outside candidate span: {row['geometry_id']}",
            )
            if method in {"cv_hard", "full_residual_hard"}:
                validator.require(
                    estimated_radius in candidate_radii,
                    f"hard selector radius outside bank: {row['geometry_id']}",
                )
            for metric in METRICS:
                baseline_value = number(validator, row, f"{baseline}_{metric}")
                raw_value = number(validator, row, f"{method}_{metric}")
                selected_value = number(validator, row, f"selected_candidate_{metric}")
                validator.close(
                    selected_value,
                    raw_value if accepted else baseline_value,
                    f"raw-or-fallback {metric}: {row['geometry_id']}",
                )
            baseline_error = number(validator, row, f"{baseline}_relative_l2")
            candidate_error = number(validator, row, "selected_candidate_relative_l2")
            expected_gain = (
                100.0 * (baseline_error - candidate_error) / max(baseline_error, 1e-12)
            )
            validator.close(
                row["selected_gain_percent"],
                expected_gain,
                f"selected gain: {row['geometry_id']}",
            )


def recompute_summary(
    validator: Validator,
    rows: list[dict[str, str]],
    baseline: str,
) -> dict[str, float]:
    candidate_errors = [
        number(validator, row, "selected_candidate_relative_l2") for row in rows
    ]
    baseline_errors = [
        number(validator, row, f"{baseline}_relative_l2") for row in rows
    ]
    gains = [
        100.0 * (base - candidate) / max(base, 1e-12)
        for candidate, base in zip(candidate_errors, baseline_errors)
    ]
    candidate_audit = [
        number(validator, row, "selected_candidate_audit_true_operator_residual_rms")
        for row in rows
    ]
    baseline_audit = [
        number(validator, row, f"{baseline}_audit_true_operator_residual_rms")
        for row in rows
    ]
    audit_change = [
        100.0 * (candidate - base) / max(base, 1e-12)
        for candidate, base in zip(candidate_audit, baseline_audit)
    ]
    return {
        "candidate_mean_relative_l2": mean(candidate_errors),
        "baseline_mean_relative_l2": mean(baseline_errors),
        "mean_gain_percent": mean(gains),
        "p10_gain_percent": quantile(gains, 0.10),
        "harm_rate_over_1_percent": mean(float(value < -1.0) for value in gains),
        "coverage": mean(
            boolean(validator, row["accepted"], "summary accepted flag") for row in rows
        ),
        "candidate_mean_gradient_relative_l2": mean(
            number(validator, row, "selected_candidate_gradient_relative_l2")
            for row in rows
        ),
        "baseline_mean_gradient_relative_l2": mean(
            number(validator, row, f"{baseline}_gradient_relative_l2") for row in rows
        ),
        "candidate_mean_front_f1": mean(
            number(validator, row, "selected_candidate_front_f1") for row in rows
        ),
        "baseline_mean_front_f1": mean(
            number(validator, row, f"{baseline}_front_f1") for row in rows
        ),
        "candidate_mean_mass_relative_error": mean(
            number(validator, row, "selected_candidate_mass_relative_error")
            for row in rows
        ),
        "baseline_mean_mass_relative_error": mean(
            number(validator, row, f"{baseline}_mass_relative_error") for row in rows
        ),
        "candidate_mean_centroid_error": mean(
            number(validator, row, "selected_candidate_centroid_error") for row in rows
        ),
        "baseline_mean_centroid_error": mean(
            number(validator, row, f"{baseline}_centroid_error") for row in rows
        ),
        "candidate_mean_audit_true_residual_rms": mean(candidate_audit),
        "baseline_mean_audit_true_residual_rms": mean(baseline_audit),
        "audit_reprojection_change_percent": mean(audit_change),
    }


def validate_summaries(
    validator: Validator,
    by_split: dict[str, list[dict[str, str]]],
    summary_rows: list[dict[str, str]],
    report: dict[str, Any],
    selection: dict[str, Any],
    baseline: str,
) -> dict[str, dict[str, float]]:
    validator.require(len(summary_rows) == 2, "summary row count")
    validator.require(
        {row["split"] for row in summary_rows} == set(SPLITS), "summary split set"
    )
    stored = {row["split"]: row for row in summary_rows}
    recomputed: dict[str, dict[str, float]] = {}
    for split, short in (
        ("independent_select", "select"),
        ("independent_lock", "lock"),
    ):
        row = stored[split]
        validator.require(row["baseline"] == baseline, f"summary baseline: {split}")
        values = recompute_summary(validator, by_split[split], baseline)
        recomputed[split] = values
        for key in SUMMARY_FIELDS:
            validator.close(row[key], values[key], f"CSV summary {split}/{key}")
            validator.close(
                report[f"{short}_summary"][key],
                values[key],
                f"report summary {split}/{key}",
            )

    select_summary = recomputed["independent_select"]
    for key in (
        "coverage",
        "mean_gain_percent",
        "p10_gain_percent",
        "harm_rate_over_1_percent",
    ):
        validator.close(
            selection[key], select_summary[key], f"selection/select summary {key}"
        )
    return recomputed


def validate_budget(
    validator: Validator,
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    algorithm = config["algorithm"]
    calls = report["call_accounting"]
    bank_size = len(config["candidate_aperture_radii"])
    folds = int(algorithm["cross_view_folds"])
    probe_stages = int(algorithm["probe_stages"])
    total = int(algorithm["total_call_budget"])
    diagnostic = bank_size * (folds + 1) * probe_stages
    final = total - diagnostic
    validator.require(total == 60, "v5a total call budget is not 60")
    validator.require(bank_size == 5, "v5a candidate bank size is not five")
    validator.require(final >= 2, "diagnostic calls leave fewer than two final stages")
    validator.require(
        calls["candidate_bank_size"] == bank_size, "reported candidate bank size"
    )
    validator.require(calls["cross_view_folds"] == folds, "reported cross-view folds")
    validator.require(
        calls["probe_stages_per_operator"] == probe_stages, "reported probe stages"
    )
    validator.require(calls["total_call_budget"] == total, "reported total call budget")
    for direction in ("forward", "adjoint"):
        validator.require(
            calls[f"diagnostic_{direction}"] == diagnostic,
            f"diagnostic {direction} budget",
        )
        validator.require(
            calls[f"final_selected_operator_{direction}"] == final,
            f"final selected-operator {direction} budget",
        )
        validator.require(
            calls[f"blind_candidate_{direction}"] == diagnostic + final == total,
            f"blind candidate {direction} total",
        )
        validator.require(
            calls[f"pinhole_equal_call_baselines_{direction}"] == total,
            f"equal-call pinhole {direction} budget",
        )
    expected_spectral = int(config["counts"]["independent_lock"]) * bank_size * (
        folds + 1
    ) + int(config["counts"]["independent_lock"])
    validator.require(
        calls["lock_exact_spectral_decompositions"] == expected_spectral,
        "lock exact spectral-decomposition accounting",
    )
    validator.require(
        calls["truth_only_audit_forward_per_method"] == 1,
        "truth-only audit forward accounting",
    )
    validator.require(
        calls["oracle_true_operator_is_not_deployable"] is True,
        "truth oracle is not marked non-deployable",
    )


def validate_gates_and_adjoint(
    validator: Validator,
    config: dict[str, Any],
    commit: dict[str, Any],
    report: dict[str, Any],
    summaries: dict[str, dict[str, float]],
) -> tuple[dict[str, bool], bool]:
    validator.close(
        commit["select_mismatch_penalty_percent"],
        report["select_mismatch_penalty_percent"],
        "select mismatch penalty commit/report",
    )
    select_mismatch = float(report["select_mismatch_penalty_percent"])
    lock_mismatch = float(report["lock_mismatch_penalty_percent"])
    validator.require(
        math.isfinite(select_mismatch) and math.isfinite(lock_mismatch),
        "non-finite mismatch penalty",
    )
    lock = summaries["independent_lock"]
    gate = config["claim_gate"]
    expected = {
        "mismatch_is_nontrivial": lock_mismatch
        >= float(gate["minimum_mismatch_penalty_percent"]),
        "mean_gain": lock["mean_gain_percent"]
        >= float(gate["minimum_mean_gain_percent"]),
        "p10_gain": lock["p10_gain_percent"] >= float(gate["minimum_p10_gain_percent"]),
        "harm_rate": lock["harm_rate_over_1_percent"]
        <= float(gate["maximum_harm_rate_over_1_percent"]),
        "audit_reprojection": lock["audit_reprojection_change_percent"]
        <= float(gate["maximum_audit_reprojection_increase_percent"]),
    }
    validator.require(set(report["gate_checks"]) == set(expected), "five-gate key set")
    validator.require(report["gate_checks"] == expected, "five gate decisions")
    gate_passed = all(expected.values())
    expected_status = PASSED_CLAIM_STATUS if gate_passed else FAILED_CLAIM_STATUS
    validator.require(
        report["claim_status"] == expected_status, "claim status/gate result"
    )

    operator_audit = report["operator_audit"]
    for split, key in (
        ("independent_select", "select_nominal_adjoint_relative_error"),
        ("independent_lock", "lock_nominal_adjoint_relative_error"),
    ):
        error = float(operator_audit[key])
        validator.require(
            math.isfinite(error) and error >= 0.0, f"adjoint error value: {split}"
        )
        validator.require(
            error <= ADJOINT_ERROR_LIMIT,
            f"adjoint error exceeds {ADJOINT_ERROR_LIMIT:g}: {split}",
        )
    return expected, gate_passed


def main() -> int:
    options = parse_args()
    validator = Validator()
    try:
        config = load_json(options.config)
        result_dir = options.result_dir
        validator.require(result_dir.is_dir(), "result directory is missing")
        validate_checksums(validator, result_dir)

        snapshot = load_json(result_dir / "config_snapshot.json")
        commit = load_json(result_dir / "selection_commit.json")
        manifest = load_json(result_dir / "operator_manifest.json")
        report = load_json(result_dir / "report.json")
        calibration_header, calibration_rows = load_csv(
            result_dir / "selection_calibration.csv"
        )
        sample_header, sample_rows = load_csv(result_dir / "sample_metrics.csv")
        summary_header, summary_rows = load_csv(result_dir / "summary.csv")

        validator.require(
            calibration_header
            == [
                "method",
                "threshold",
                "coverage",
                "mean_gain_percent",
                "p10_gain_percent",
                "harm_rate_over_1_percent",
                "feasible",
            ],
            "selection calibration CSV schema",
        )
        validator.require(
            summary_header == ["split", "baseline", *SUMMARY_FIELDS],
            "summary CSV schema",
        )
        validate_hash_chain(
            validator,
            options.config,
            config,
            result_dir,
            snapshot,
            commit,
            report,
        )
        validate_operator_manifests(validator, config, commit, manifest, report)
        by_split = validate_sample_provenance(
            validator, config, sample_rows, manifest, commit, report
        )
        baseline = validate_baseline(
            validator, commit, report, by_split["independent_select"]
        )
        selection = commit["selected_operator_method"]
        validator.require(
            sample_header == expected_sample_header(baseline, str(selection["method"])),
            "sample metrics CSV schema",
        )
        validate_calibration(
            validator,
            config,
            calibration_rows,
            by_split["independent_select"],
            selection,
            baseline,
        )
        validate_sample_selection(validator, config, by_split, selection, baseline)
        summaries = validate_summaries(
            validator,
            by_split,
            summary_rows,
            report,
            selection,
            baseline,
        )
        validate_budget(validator, config, report)
        gate_checks, gate_passed = validate_gates_and_adjoint(
            validator, config, commit, report, summaries
        )

        print(
            "PASS: checksums; config/selection/source hashes; pre-lock provenance; "
            "operator separation"
        )
        print(
            "PASS: audit-camera masks; 60-call budget; threshold application; "
            "sample/summary aggregation"
        )
        print(
            "PASS: five gate decisions and claim status are truthful; "
            f"adjoint errors <= {ADJOINT_ERROR_LIMIT:g}"
        )
        print(
            json.dumps(
                {
                    "validator_status": "PASS",
                    "checks": validator.checks,
                    "sample_rows": len(sample_rows),
                    "calibration_rows": len(calibration_rows),
                    "selected_baseline": baseline,
                    "selected_method": selection["method"],
                    "algorithm_gate": "PASSED" if gate_passed else "FAILED",
                    "gate_checks": gate_checks,
                    "claim_status": report["claim_status"],
                    "adjoint_relative_error": {
                        "select": report["operator_audit"][
                            "select_nominal_adjoint_relative_error"
                        ],
                        "lock": report["operator_audit"][
                            "lock_nominal_adjoint_relative_error"
                        ],
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        ValidationError,
        OSError,
        csv.Error,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"FAIL after {validator.checks} checks: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
