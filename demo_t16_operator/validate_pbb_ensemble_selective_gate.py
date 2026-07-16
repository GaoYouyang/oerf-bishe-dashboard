#!/usr/bin/env python3
"""Independently validate the PBB selective-ensemble gate artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "pbb_ensemble_selective_gate.json"
RESULT_FILES = (
    "config_snapshot.json",
    "selection_commit.json",
    "history.csv",
    "threshold_calibration.csv",
    "sample_metrics.csv",
    "summary.csv",
    "pbb_ensemble_selective_gate.png",
    "report.json",
)
ADJOINT_ERROR_LIMIT = 1e-5
BASELINES = ("fixed_pg", "projected_bb", "fista")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result-dir", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: Any, expected: Any, message: str) -> None:
    a, e = float(actual), float(expected)
    require(math.isfinite(a) and math.isfinite(e), f"{message}: non-finite value")
    # CSV values were emitted from float32 tensors; aggregate recomputation is
    # therefore allowed the corresponding small serialization/accumulation drift.
    require(math.isclose(a, e, rel_tol=2e-6, abs_tol=2e-7), f"{message}: {a} != {e}")


def quantile(values: list[float], q: float) -> float:
    """Match numpy's default linear quantile without making validation depend on numpy."""
    ordered = sorted(values)
    require(bool(ordered), "cannot quantile an empty sequence")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def numeric(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"invalid numeric field {key!r}") from exc


def boolean(value: str) -> bool:
    require(value in {"True", "False"}, f"invalid boolean CSV value: {value}")
    return value == "True"


def validate_checksums(result: Path) -> None:
    manifest = result / "checksums.sha256"
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    require(set(name for _, name in (line.split("  ", 1) for line in lines)) == set(RESULT_FILES),
            "checksum manifest does not cover exactly the eight public assets")
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, "malformed checksum manifest line")
        expected, name = parts
        path = result / name
        require(path.is_file(), f"missing checksum target: {name}")
        require(digest(path) == expected, f"checksum mismatch: {name}")


def validate_hashes(config_path: Path, config: dict, result: Path, commit: dict, report: dict) -> None:
    snapshot_path = result / "config_snapshot.json"
    snapshot = load_json(snapshot_path)
    require(snapshot == config, "config snapshot differs from requested config")
    config_hash = digest(snapshot_path)
    selection_hash = digest(result / "selection_commit.json")
    require(commit["config_sha256"] == config_hash, "selection config hash mismatch")
    require(report["selection_commit_sha256"] == selection_hash, "report selection hash mismatch")
    require(commit["select_operator_sha256"] == report["operator_audit"]["select_sha256"],
            "select operator hash is not carried consistently")
    require(commit["train_operator_sha256"] == report["operator_audit"]["train_sha256"],
            "train operator hash is not carried consistently")
    expected_sources = {
        "runner": ROOT / "run_pbb_ensemble_selective_gate.py",
        "cg_pdno": ROOT / "cg_pdno.py",
        "independent_generator": ROOT / "independent_reaction_bost.py",
        "measurement_contract": ROOT / "measurement_contract.py",
    }
    for label, source in expected_sources.items():
        require(report["source_sha256"][label] == digest(source), f"source hash mismatch: {label}")


def validate_geometry_and_lock_order(samples: list[dict[str, str]], report: dict, commit: dict) -> None:
    by_split = {split: [row for row in samples if row["split"] == split]
                for split in ("independent_select", "independent_lock")}
    require(len(by_split["independent_select"]) == 36, "unexpected independent-select row count")
    require(len(by_split["independent_lock"]) == 40, "unexpected independent-lock row count")
    select_ids = {row["geometry_id"] for row in by_split["independent_select"]}
    lock_ids = {row["geometry_id"] for row in by_split["independent_lock"]}
    require(len(select_ids) == 36 and len(lock_ids) == 40, "duplicate geometry id in sample metrics")
    require(not select_ids & lock_ids, "select/lock geometry overlap")
    overlap = report["geometry_overlap"]
    require(overlap["select_lock"] == [], "report does not mark select/lock overlap empty")
    require(overlap["train_validation"] == [], "report does not mark train/validation overlap empty")
    require(commit["created_before_independent_lock"] is True, "selection commit was not marked pre-lock")
    require(report["lock_status"].startswith("FIRST_OPEN"), "lock is not marked first-open")


def validate_baseline_selection(select_rows: list[dict[str, str]], commit: dict, report: dict) -> str:
    means = {name: sum(numeric(row, f"{name}_relative_l2") for row in select_rows) / len(select_rows)
             for name in BASELINES}
    selected = min(BASELINES, key=lambda name: (means[name], BASELINES.index(name)))
    require(commit["selected_deterministic_baseline"] == selected,
            "selected baseline is not the strongest deterministic select baseline")
    require(report["selected_deterministic_baseline"] == selected, "report baseline mismatch")
    for name in BASELINES:
        close(commit["baseline_mean_relative_l2"][name], means[name], f"{name} baseline mean")
    return selected


def gain(row: dict[str, str], baseline: str, candidate_key: str = "candidate") -> float:
    base = numeric(row, f"{baseline}_relative_l2")
    candidate = numeric(row, f"{candidate_key}_relative_l2")
    return 100.0 * (base - candidate) / max(base, 1e-12)


def validate_threshold(select_rows: list[dict[str, str]], calibration: list[dict[str, str]],
                       commit: dict, config: dict, baseline: str) -> None:
    uncertainties = [numeric(row, "uncertainty_score") for row in select_rows]
    quantile_count = int(config["selection_gate"]["quantile_count"])
    candidates: list[float] = [-1.0]
    candidates.extend(
        sorted(
            set(
                quantile(uncertainties, i / (quantile_count - 1))
                for i in range(quantile_count)
            )
        )
    )
    upper = max(uncertainties) + max(1e-9, 1e-6 * max(uncertainties))
    candidates.append(upper)
    require(len(calibration) == len(candidates), "unexpected threshold calibration row count")
    constraints = config["selection_gate"]
    calculated: list[dict[str, Any]] = []
    for expected_threshold, row in zip(candidates, calibration):
        threshold = numeric(row, "threshold")
        close(threshold, expected_threshold, "threshold candidate")
        accepted = [numeric(item, "uncertainty_score") <= threshold for item in select_rows]
        gains = [gain(item, baseline, "raw_ensemble") if ok else gain(item, baseline, "pbb_fallback")
                 for item, ok in zip(select_rows, accepted)]
        values = {
            "coverage": sum(accepted) / len(accepted),
            "mean_gain_percent": sum(gains) / len(gains),
            "p10_gain_percent": quantile(gains, 0.10),
            "harm_rate_over_1_percent": sum(value < -1.0 for value in gains) / len(gains),
        }
        feasible = (values["coverage"] >= float(constraints["minimum_coverage"])
                    and values["p10_gain_percent"] >= float(constraints["minimum_p10_gain_percent"])
                    and values["harm_rate_over_1_percent"] <= float(constraints["maximum_harm_rate_over_1_percent"]))
        for key, value in values.items():
            close(row[key], value, f"threshold {key}")
        require(boolean(row["feasible"]) is feasible, "threshold feasibility mismatch")
        calculated.append({**values, "threshold": threshold, "feasible": feasible})
    feasible = [row for row in calculated if row["feasible"]]
    chosen = (max(feasible, key=lambda row: (row["mean_gain_percent"], row["coverage"], -row["threshold"]))
              if feasible else calculated[0])
    selection = commit["threshold_selection"]
    for key in ("threshold", "coverage", "mean_gain_percent", "p10_gain_percent", "harm_rate_over_1_percent"):
        close(selection[key], chosen[key], f"committed threshold {key}")
    require(selection["feasible"] is chosen["feasible"], "committed threshold feasibility mismatch")
    expected_reason = ("best_select_mean_gain_subject_to_predeclared_tail_and_coverage_constraints"
                       if feasible else "no_feasible_select_threshold_abstain_all")
    require(selection["selection_reason"] == expected_reason, "threshold selection reason mismatch")


def validate_summary(samples: list[dict[str, str]], summary: list[dict[str, str]], report: dict,
                     baseline: str) -> None:
    require(len(summary) == 2, "summary must contain select and lock rows")
    for split in ("independent_select", "independent_lock"):
        rows = [row for row in samples if row["split"] == split]
        item = next((row for row in summary if row["split"] == split), None)
        require(item is not None, f"missing summary row: {split}")
        require(item["baseline"] == baseline, f"summary baseline mismatch: {split}")
        gains = [gain(row, baseline) for row in rows]
        expected = {
            "candidate_mean_relative_l2": sum(numeric(row, "candidate_relative_l2") for row in rows) / len(rows),
            "baseline_mean_relative_l2": sum(numeric(row, f"{baseline}_relative_l2") for row in rows) / len(rows),
            "mean_gain_percent": sum(gains) / len(gains),
            "p10_gain_percent": quantile(gains, 0.10),
            "harm_rate_over_1_percent": sum(value < -1.0 for value in gains) / len(gains),
            "coverage": sum(boolean(row["accepted"]) for row in rows) / len(rows),
            "certificate_violation_rate": sum(boolean(row["certificate_violation"]) for row in rows) / len(rows),
            "candidate_mean_gradient_relative_l2": sum(numeric(row, "candidate_gradient_relative_l2") for row in rows) / len(rows),
            "baseline_mean_gradient_relative_l2": sum(numeric(row, f"{baseline}_gradient_relative_l2") for row in rows) / len(rows),
            "candidate_mean_front_f1": sum(numeric(row, "candidate_front_f1") for row in rows) / len(rows),
            "baseline_mean_front_f1": sum(numeric(row, f"{baseline}_front_f1") for row in rows) / len(rows),
        }
        for key, value in expected.items():
            close(item[key], value, f"{split} summary {key}")
            close(report["select_summary" if split.endswith("select") else "lock_summary"][key], value,
                  f"{split} report {key}")


def validate_sample_semantics(samples: list[dict[str, str]], commit: dict, config: dict, baseline: str) -> None:
    member_count = len(config["training"]["seeds"])
    threshold = float(commit["threshold_selection"]["threshold"])
    for row in samples:
        accepted = numeric(row, "uncertainty_score") <= threshold
        require(boolean(row["accepted"]) is accepted, "sample accepted flag disagrees with committed threshold")
        candidate = numeric(row, "candidate_relative_l2")
        raw = numeric(row, "raw_ensemble_relative_l2")
        fallback = numeric(row, "pbb_fallback_relative_l2")
        close(candidate, raw if accepted else fallback, "candidate is not raw-or-fallback selection")
        require(row["selected_baseline"] == baseline, "sample selected baseline mismatch")
        require(0.0 <= numeric(row, "member_acceptance_rate") <= 1.0, "member acceptance rate out of range")
        require(member_count == 5, "unexpected ensemble member count")
        close(numeric(row, "gain_vs_" + baseline + "_percent"), gain(row, baseline), "stored selected-baseline gain")


def validate_gate_and_budget(config: dict, report: dict, samples: list[dict[str, str]], baseline: str) -> None:
    lock = report["lock_summary"]
    gate = config["claim_gate"]
    expected_gate = {
        "mean_gain": float(lock["mean_gain_percent"]) >= float(gate["minimum_mean_gain_percent"]),
        "p10_gain": float(lock["p10_gain_percent"]) >= float(gate["minimum_p10_gain_percent"]),
        "harm_rate": float(lock["harm_rate_over_1_percent"]) <= float(gate["maximum_harm_rate_over_1_percent"]),
        "certificate": float(lock["certificate_violation_rate"]) <= float(gate["maximum_certificate_violation_rate"]),
    }
    require(report["gate_checks"] == expected_gate, "report gate_checks disagree with config and lock summary")
    require(config["lipschitz_method"] == "exact_small_matrix", "config does not mark exact_small_matrix")
    require(report["call_accounting"]["lipschitz_method"] == "exact_small_matrix", "report loses exact_small_matrix marker")
    calls = report["call_accounting"]
    stages = int(config["model"]["stages"])
    for method in ("candidate_shared", "fixed_pg", "projected_bb", "fista"):
        require(calls[f"{method}_forward"] == stages, f"{method} forward budget mismatch")
        require(calls[f"{method}_adjoint"] == stages, f"{method} adjoint budget mismatch")
    require(calls["correction_head_passes"] == len(config["training"]["seeds"]), "head count mismatch")
    require(calls["physical_trajectory_repeated_per_head"] is False, "physical calls repeated per head")
    require(calls["metric_only_forward_per_method"] == 1, "metric-only forward accounting mismatch")
    require(calls["lipschitz_power_iterations_precomputation"] == 0,
            "exact spectral path must not report power-iteration calls")
    expected_decompositions = (
        int(config["counts"]["train"]) * int(config["depth"])
        + int(config["counts"]["validation"]) * int(config["depth"])
        + int(config["counts"]["independent_select"])
        + int(config["counts"]["independent_lock"])
    )
    require(calls["exact_spectral_decompositions"] == expected_decompositions,
            "exact spectral-decomposition count mismatch")
    select_families = set(config["families"]["independent_select"])
    lock_families = set(config["families"]["independent_lock"])
    require(not select_families & lock_families, "select and lock field families overlap")
    audit = report["operator_audit"]
    require(float(audit["select_adjoint_relative_error"]) <= ADJOINT_ERROR_LIMIT, "select adjoint error exceeds threshold")
    require(float(audit["lock_adjoint_relative_error"]) <= ADJOINT_ERROR_LIMIT, "lock adjoint error exceeds threshold")
    require(audit["select_lock_equal"] is False, "select and lock operators unexpectedly equal")
    require(all(row["selected_baseline"] == baseline for row in samples), "mixed selected baselines")


def main() -> int:
    options = args()
    config = load_json(options.config)
    result = options.result_dir or ROOT / "results" / "pbb_ensemble_selective_gate"
    try:
        report = load_json(result / "report.json")
        commit = load_json(result / "selection_commit.json")
        samples = load_csv(result / "sample_metrics.csv")
        summary = load_csv(result / "summary.csv")
        calibration = load_csv(result / "threshold_calibration.csv")
        validate_checksums(result)
        validate_hashes(options.config, config, result, commit, report)
        validate_geometry_and_lock_order(samples, report, commit)
        select_rows = [row for row in samples if row["split"] == "independent_select"]
        baseline = validate_baseline_selection(select_rows, commit, report)
        validate_threshold(select_rows, calibration, commit, config, baseline)
        validate_sample_semantics(samples, commit, config, baseline)
        validate_summary(samples, summary, report, baseline)
        validate_gate_and_budget(config, report, samples, baseline)
        print("PASS: checksums; config/selection hashes; geometry disjointness; pre-lock commit")
        print("PASS: strongest select baseline; threshold calibration; sample selection; summary metrics")
        print("PASS: config gate checks; exact_small_matrix; physical call budget; adjoint error <= 1e-5")
        print(json.dumps({"status": "PASS", "select_rows": len(select_rows), "lock_rows": 40,
                          "selected_baseline": baseline, "claim_status": report["claim_status"]},
                         ensure_ascii=False))
        return 0
    except (AssertionError, KeyError, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
