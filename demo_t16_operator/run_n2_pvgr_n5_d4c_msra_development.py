#!/usr/bin/env python3
"""Run the preregistered N5-D4c MSRA certificate development screen.

The screen is intentionally synthetic and development-only.  It compares a
traditional signal-relative adjoint test with a gamma-scaled normwise metric,
then exposes failure modes that neither metric can resolve without multiple
probes, finite differences, structural controls, and actual branch semantics.

The frozen D4b result is read only for a labelled post-open panel.  It is never
used to select a threshold or to change the historical fail-closed decision.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_t16_operator.mixed_scale_adjoint_certificate import (  # noqa: E402
    evaluate_mixed_scale_adjoint,
    gamma_n,
    summarize_probe_set,
)


DEFAULT_CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n2_pvgr_n5_d4c_msra_development_preregistered_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "demo_t16_operator/results/n2_pvgr_n5_d4c_msra_development_v1"
)
SCHEMA = "n2-pvgr-n5-d4c-msra-development-result-1.0"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"external-test-input/{resolved.name}"


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=check, capture_output=True
    )


def _assert_committed(paths: Iterable[Path]) -> str:
    commit = _git("rev-parse", "HEAD").stdout.decode().strip()
    for path in paths:
        relative = _relative(path)
        frozen = _git("show", f"{commit}:{relative}", check=False)
        if frozen.returncode != 0:
            raise ValueError(f"D4c source is not committed: {relative}")
        if hashlib.sha256(frozen.stdout).hexdigest() != _sha256(path):
            raise ValueError(f"D4c committed source drifted: {relative}")
    return commit


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize a degenerate vector")
    return array / norm


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate_array - reference_array)
        / max(np.linalg.norm(candidate_array), np.linalg.norm(reference_array), 1e-300)
    )


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "n2-pvgr-n5-d4c-msra-development-preregistered-1.0":
        raise ValueError("unexpected N5-D4c config schema")
    if config.get("status") != (
        "preregistered_development_only_no_derivative_or_reconstruction_authorization"
    ):
        raise ValueError("N5-D4c config is not development-only")
    if config.get("device") != "cpu" or config.get("dtype") != "float64":
        raise ValueError("N5-D4c must use CPU float64")
    if int(config["input_dimension"]) != 17**3 or int(config["output_dimension"]) != 8:
        raise ValueError("N5-D4c dimensions drifted from the D4b field/output shapes")
    if int(config["trial_count"]) < 4:
        raise ValueError("N5-D4c needs at least four frozen trials")
    probes = [int(value) for value in config["probe_counts"]]
    if probes != sorted(set(probes)) or probes[0] != 1:
        raise ValueError("probe_counts must be sorted, unique, and begin at one")
    thresholds = [float(value) for value in config["gamma_threshold_grid"]]
    if thresholds != sorted(set(thresholds)) or any(value <= 0 for value in thresholds):
        raise ValueError("gamma threshold grid must be sorted, unique, and positive")
    for key in ("fault_relative_magnitudes", "cancellation_deltas"):
        values = [float(value) for value in config[key]]
        if values != sorted(set(values)) or any(value <= 0 for value in values):
            raise ValueError(f"{key} must be sorted, unique, and positive")
    if config["decision_contract"].get("threshold_selection_is_forbidden") is not True:
        raise ValueError("D4c must forbid threshold selection")
    if any(bool(value) for value in config["claim_authorizations"].values()):
        raise ValueError("D4c development cannot pre-authorize any claim")


def _matrix(rng: np.random.Generator, output_dim: int, input_dim: int) -> np.ndarray:
    return rng.normal(size=(output_dim, input_dim)).astype(np.float64) / math.sqrt(
        input_dim
    )


def _probe_bank(
    rng: np.random.Generator, count: int, input_dim: int
) -> np.ndarray:
    return np.stack(
        [_normalize(rng.normal(size=input_dim)) for _ in range(count)], axis=0
    )


def _probe_summaries(
    jvps: np.ndarray,
    cotangent: np.ndarray,
    tangents: np.ndarray,
    vjp: np.ndarray,
    probe_counts: list[int],
) -> dict[int, dict[str, float | int | bool]]:
    evidence = [
        evaluate_mixed_scale_adjoint(jvp, cotangent, tangent, vjp)
        for jvp, tangent in zip(jvps, tangents, strict=True)
    ]
    return {
        count: summarize_probe_set(evidence[:count]).to_dict()
        for count in probe_counts
    }


def _scenario_rows(
    *,
    scenario: str,
    expected_role: str,
    trial: int,
    probe_summaries: dict[int, dict[str, float | int | bool]],
    traditional_threshold: float,
    fd_relative_error: float,
    structure_relative_error: float,
    actual_forward_branch_changed: bool,
    diagnostic_support_changed: bool,
    parameter_name: str = "none",
    parameter_value: float = 0.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe_count, summary in probe_summaries.items():
        rows.append(
            {
                "scenario": scenario,
                "expected_role": expected_role,
                "trial": trial,
                "parameter_name": parameter_name,
                "parameter_value": float(parameter_value),
                "probe_count": int(probe_count),
                "maximum_dot_relative_defect": float(
                    summary["maximum_dot_relative_defect"]
                ),
                "maximum_gamma_scaled_normwise_score": float(
                    summary["maximum_gamma_scaled_normwise_score"]
                ),
                "median_gamma_scaled_normwise_score": float(
                    summary["median_gamma_scaled_normwise_score"]
                ),
                "minimum_dot_signal": float(summary["minimum_dot_signal"]),
                "maximum_dot_condition_proxy": float(
                    summary["maximum_dot_condition_proxy"]
                ),
                "traditional_dot_gate": bool(
                    summary["all_finite"]
                    and float(summary["maximum_dot_relative_defect"])
                    <= traditional_threshold
                ),
                "fd_relative_error": float(fd_relative_error),
                "structure_relative_error": float(structure_relative_error),
                "actual_forward_branch_changed": bool(actual_forward_branch_changed),
                "diagnostic_support_changed": bool(diagnostic_support_changed),
            }
        )
    return rows


def _hard_branch_fd_error(
    positive: np.ndarray,
    negative: np.ndarray,
    tangents: np.ndarray,
) -> float:
    errors: list[float] = []
    for tangent in tangents:
        branch_local = positive @ tangent
        if tangent[0] >= 0.0:
            central = 0.5 * ((positive @ tangent) + (negative @ tangent))
        else:
            central = 0.5 * ((negative @ tangent) + (positive @ tangent))
        errors.append(_relative_l2(central, branch_local))
    return max(errors)


def _build_synthetic_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_dim = int(config["input_dimension"])
    output_dim = int(config["output_dimension"])
    trial_count = int(config["trial_count"])
    maximum_probes = max(int(value) for value in config["probe_counts"])
    probe_counts = [int(value) for value in config["probe_counts"]]
    traditional = float(config["traditional_dot_relative_threshold"])
    faults = [float(value) for value in config["fault_relative_magnitudes"]]
    cancellation = [float(value) for value in config["cancellation_deltas"]]
    rows: list[dict[str, Any]] = []

    for trial in range(trial_count):
        rng = np.random.default_rng(int(config["seed_base"]) + trial)
        operator = _matrix(rng, output_dim, input_dim)
        alternative = _matrix(rng, output_dim, input_dim)
        tangents = _probe_bank(rng, maximum_probes, input_dim)
        cotangent = _normalize(rng.normal(size=output_dim))
        jvps = tangents @ operator.T
        vjp = operator.T @ cotangent

        rows.extend(
            _scenario_rows(
                scenario="clean_general",
                expected_role="clean_should_pass",
                trial=trial,
                probe_summaries=_probe_summaries(
                    jvps, cotangent, tangents, vjp, probe_counts
                ),
                traditional_threshold=traditional,
                fd_relative_error=0.0,
                structure_relative_error=0.0,
                actual_forward_branch_changed=False,
                diagnostic_support_changed=False,
            )
        )

        first_jvp = jvps[0]
        low_cotangent = rng.normal(size=output_dim)
        low_cotangent = low_cotangent - first_jvp * (
            float(low_cotangent @ first_jvp) / float(first_jvp @ first_jvp)
        )
        low_cotangent = _normalize(low_cotangent)
        low_vjp = operator.T @ low_cotangent
        rows.extend(
            _scenario_rows(
                scenario="clean_low_bilinear_signal",
                expected_role="clean_should_pass_but_signal_relative_can_false_reject",
                trial=trial,
                probe_summaries=_probe_summaries(
                    jvps, low_cotangent, tangents, low_vjp, probe_counts
                ),
                traditional_threshold=traditional,
                fd_relative_error=0.0,
                structure_relative_error=0.0,
                actual_forward_branch_changed=False,
                diagnostic_support_changed=False,
            )
        )

        rows.extend(
            _scenario_rows(
                scenario="diagnostic_only_support_flip",
                expected_role="clean_current_forward_report_only",
                trial=trial,
                probe_summaries=_probe_summaries(
                    jvps, cotangent, tangents, vjp, probe_counts
                ),
                traditional_threshold=traditional,
                fd_relative_error=0.0,
                structure_relative_error=0.0,
                actual_forward_branch_changed=False,
                diagnostic_support_changed=True,
            )
        )

        blind = rng.normal(size=input_dim)
        blind = blind - tangents[0] * float(blind @ tangents[0])
        blind = _normalize(blind)
        aligned_output = cotangent
        vjp_norm = max(float(np.linalg.norm(vjp)), 1e-300)
        for magnitude in faults:
            aligned_vjp = vjp + magnitude * vjp_norm * tangents[0]
            rows.extend(
                _scenario_rows(
                    scenario="vjp_aligned_fault",
                    expected_role="fault_should_reject_by_adjoint",
                    trial=trial,
                    parameter_name="relative_fault_magnitude",
                    parameter_value=magnitude,
                    probe_summaries=_probe_summaries(
                        jvps, cotangent, tangents, aligned_vjp, probe_counts
                    ),
                    traditional_threshold=traditional,
                    fd_relative_error=0.0,
                    structure_relative_error=0.0,
                    actual_forward_branch_changed=False,
                    diagnostic_support_changed=False,
                )
            )

            blind_vjp = vjp + magnitude * vjp_norm * blind
            rows.extend(
                _scenario_rows(
                    scenario="vjp_first_probe_blind_fault",
                    expected_role="fault_needs_multiple_random_tangents",
                    trial=trial,
                    parameter_name="relative_fault_magnitude",
                    parameter_value=magnitude,
                    probe_summaries=_probe_summaries(
                        jvps, cotangent, tangents, blind_vjp, probe_counts
                    ),
                    traditional_threshold=traditional,
                    fd_relative_error=0.0,
                    structure_relative_error=0.0,
                    actual_forward_branch_changed=False,
                    diagnostic_support_changed=False,
                )
            )

            corrupted_jvps = jvps.copy()
            for index in range(len(corrupted_jvps)):
                corrupted_jvps[index] = corrupted_jvps[index] + (
                    magnitude
                    * max(float(np.linalg.norm(corrupted_jvps[index])), 1e-300)
                    * aligned_output
                )
            fd_error = max(
                _relative_l2(candidate, reference)
                for candidate, reference in zip(corrupted_jvps, jvps, strict=True)
            )
            rows.extend(
                _scenario_rows(
                    scenario="jvp_aligned_fault",
                    expected_role="fault_should_reject_by_adjoint_and_fd",
                    trial=trial,
                    parameter_name="relative_fault_magnitude",
                    parameter_value=magnitude,
                    probe_summaries=_probe_summaries(
                        corrupted_jvps, cotangent, tangents, vjp, probe_counts
                    ),
                    traditional_threshold=traditional,
                    fd_relative_error=fd_error,
                    structure_relative_error=0.0,
                    actual_forward_branch_changed=False,
                    diagnostic_support_changed=False,
                )
            )

            output_direction = _normalize(rng.normal(size=output_dim))
            input_direction = _normalize(rng.normal(size=input_dim))
            rank_one = np.outer(output_direction, input_direction)
            rank_one *= float(np.linalg.norm(operator))
            wrong = operator + magnitude * rank_one
            wrong_jvps = tangents @ wrong.T
            wrong_vjp = wrong.T @ cotangent
            wrong_fd_error = max(
                _relative_l2(candidate, reference)
                for candidate, reference in zip(wrong_jvps, jvps, strict=True)
            )
            rows.extend(
                _scenario_rows(
                    scenario="self_consistent_wrong_derivative",
                    expected_role="fault_adjoint_blind_fd_must_reject",
                    trial=trial,
                    parameter_name="relative_fault_magnitude",
                    parameter_value=magnitude,
                    probe_summaries=_probe_summaries(
                        wrong_jvps, cotangent, tangents, wrong_vjp, probe_counts
                    ),
                    traditional_threshold=traditional,
                    fd_relative_error=wrong_fd_error,
                    structure_relative_error=0.0,
                    actual_forward_branch_changed=False,
                    diagnostic_support_changed=False,
                )
            )

            direct = operator + magnitude * rank_one
            direct_jvps = tangents @ direct.T
            direct_vjp = direct.T @ cotangent
            rows.extend(
                _scenario_rows(
                    scenario="structure_mismatch",
                    expected_role="fault_adjoint_and_fd_blind_structure_must_reject",
                    trial=trial,
                    parameter_name="relative_fault_magnitude",
                    parameter_value=magnitude,
                    probe_summaries=_probe_summaries(
                        direct_jvps, cotangent, tangents, direct_vjp, probe_counts
                    ),
                    traditional_threshold=traditional,
                    fd_relative_error=0.0,
                    structure_relative_error=_relative_l2(direct, operator),
                    actual_forward_branch_changed=False,
                    diagnostic_support_changed=False,
                )
            )

        for delta in cancellation:
            curved = operator + 0.5 * delta * alternative
            straight = operator - 0.5 * delta * alternative
            residual = curved - straight
            separate_jvps = (tangents @ curved.T) - (tangents @ straight.T)
            separate_vjp = (curved.T @ cotangent) - (straight.T @ cotangent)
            paired_jvps = tangents @ residual.T
            paired_vjp = residual.T @ cotangent
            ideal_jvps = delta * (tangents @ alternative.T)
            for scenario, candidate_jvps, candidate_vjp, role in (
                (
                    "separate_cancellation_residual",
                    separate_jvps,
                    separate_vjp,
                    "numerically_unstable_correct_formula_needs_rejection_or_rewrite",
                ),
                (
                    "paired_cancellation_residual",
                    paired_jvps,
                    paired_vjp,
                    "clean_should_pass",
                ),
            ):
                rows.extend(
                    _scenario_rows(
                        scenario=scenario,
                        expected_role=role,
                        trial=trial,
                        parameter_name="component_difference_scale",
                        parameter_value=delta,
                        probe_summaries=_probe_summaries(
                            candidate_jvps,
                            cotangent,
                            tangents,
                            candidate_vjp,
                            probe_counts,
                        ),
                        traditional_threshold=traditional,
                        fd_relative_error=max(
                            _relative_l2(candidate, reference)
                            for candidate, reference in zip(
                                candidate_jvps, ideal_jvps, strict=True
                            )
                        ),
                        structure_relative_error=0.0,
                        actual_forward_branch_changed=False,
                        diagnostic_support_changed=False,
                    )
                )

        hard_jvps = jvps
        hard_vjp = vjp
        rows.extend(
            _scenario_rows(
                scenario="hard_branch_crossing",
                expected_role="fault_branch_must_reject_even_if_local_adjoint_passes",
                trial=trial,
                probe_summaries=_probe_summaries(
                    hard_jvps, cotangent, tangents, hard_vjp, probe_counts
                ),
                traditional_threshold=traditional,
                fd_relative_error=_hard_branch_fd_error(
                    operator, alternative, tangents
                ),
                structure_relative_error=0.0,
                actual_forward_branch_changed=True,
                diagnostic_support_changed=False,
            )
        )
    return rows


def _gate_rows(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fd_threshold = float(config["finite_difference_relative_threshold"])
    structure_threshold = float(config["structure_relative_threshold"])
    clean_roles = {
        "clean_should_pass",
        "clean_should_pass_but_signal_relative_can_false_reject",
        "clean_current_forward_report_only",
    }
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        for threshold in config["gamma_threshold_grid"]:
            gamma_gate = bool(
                row["maximum_gamma_scaled_normwise_score"] <= float(threshold)
            )
            fd_gate = bool(row["fd_relative_error"] <= fd_threshold)
            structure_gate = bool(
                row["structure_relative_error"] <= structure_threshold
            )
            branch_gate = not bool(row["actual_forward_branch_changed"])
            combined = bool(gamma_gate and fd_gate and structure_gate and branch_gate)
            expected_clean = row["expected_role"] in clean_roles
            evaluated.append(
                row
                | {
                    "gamma_threshold": float(threshold),
                    "gamma_gate": gamma_gate,
                    "fd_gate": fd_gate,
                    "structure_gate": structure_gate,
                    "branch_gate": branch_gate,
                    "combined_gate": combined,
                    "expected_clean": expected_clean,
                    "correct_classification": combined if expected_clean else not combined,
                }
            )
    return evaluated


def _aggregate(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_roles = {
        "clean_should_pass",
        "clean_should_pass_but_signal_relative_can_false_reject",
        "clean_current_forward_report_only",
    }
    summary: list[dict[str, Any]] = []
    for threshold in config["gamma_threshold_grid"]:
        for probe_count in config["probe_counts"]:
            members = [
                row
                for row in rows
                if row["gamma_threshold"] == float(threshold)
                and row["probe_count"] == int(probe_count)
            ]
            clean = [row for row in members if row["expected_role"] in clean_roles]
            faults = [row for row in members if row["expected_role"] not in clean_roles]
            summary.append(
                {
                    "gamma_threshold": float(threshold),
                    "probe_count": int(probe_count),
                    "clean_count": len(clean),
                    "clean_acceptance_rate": sum(row["combined_gate"] for row in clean)
                    / len(clean),
                    "fault_count": len(faults),
                    "fault_detection_rate": sum(not row["combined_gate"] for row in faults)
                    / len(faults),
                    "overall_correct_classification_rate": sum(
                        row["correct_classification"] for row in members
                    )
                    / len(members),
                }
            )

    scenario_summary: list[dict[str, Any]] = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for probe_count in config["probe_counts"]:
            for threshold in config["gamma_threshold_grid"]:
                members = [
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["probe_count"] == int(probe_count)
                    and row["gamma_threshold"] == float(threshold)
                ]
                scenario_summary.append(
                    {
                        "scenario": scenario,
                        "probe_count": int(probe_count),
                        "gamma_threshold": float(threshold),
                        "count": len(members),
                        "combined_acceptance_rate": sum(
                            row["combined_gate"] for row in members
                        )
                        / len(members),
                        "traditional_dot_acceptance_rate": sum(
                            row["traditional_dot_gate"] for row in members
                        )
                        / len(members),
                        "gamma_gate_acceptance_rate": sum(
                            row["gamma_gate"] for row in members
                        )
                        / len(members),
                        "fd_gate_acceptance_rate": sum(
                            row["fd_gate"] for row in members
                        )
                        / len(members),
                        "structure_gate_acceptance_rate": sum(
                            row["structure_gate"] for row in members
                        )
                        / len(members),
                        "branch_gate_acceptance_rate": sum(
                            row["branch_gate"] for row in members
                        )
                        / len(members),
                    }
                )
    return {"threshold_probe_summary": summary, "scenario_summary": scenario_summary}


def _retrospective_d4b(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        key: _resolve(value)
        for key, value in config["retrospective_d4b"].items()
        if key not in {"role"}
    }
    result = _read_json(paths["result"])
    rows = _read_json(paths["rows"])
    forensics = _read_json(paths["forensics"])
    if result.get("machine_decision") != "D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED":
        raise ValueError("D4b historical decision drifted")
    if any(bool(value) for value in result["authorizations"].values()):
        raise ValueError("D4b historical authorizations drifted")
    maximum_gamma = gamma_n(int(config["input_dimension"]))
    map_rows: list[dict[str, Any]] = []
    for row in rows:
        for map_id, payload in row["maps"].items():
            action_scale = max(float(payload["jvp_norm"]), float(payload["vjp_norm"]))
            normwise = float(payload["dot_absolute_defect"]) / max(
                action_scale, 1e-300
            )
            map_rows.append(
                {
                    "cell_index": int(row["cell_index"]),
                    "pair_id": str(row["pair_id"]),
                    "role": str(row["role"]),
                    "direction_index": int(row["direction_index"]),
                    "map_id": map_id,
                    "dot_relative_defect": float(payload["dot_relative_defect"]),
                    "dot_condition_proxy": action_scale
                    / max(
                        abs(float(payload["dot_jvp_cotangent"])),
                        abs(float(payload["dot_tangent_vjp"])),
                        1e-300,
                    ),
                    "normwise_defect": normwise,
                    "gamma_scaled_normwise_score": normwise / maximum_gamma,
                    "historical_map_gate": bool(payload["map_gate"]),
                    "historical_topology_gate": bool(row["topology_gate"]),
                }
            )
    threshold_counts = [
        {
            "gamma_threshold": float(threshold),
            "passing_map_count": sum(
                item["gamma_scaled_normwise_score"] <= float(threshold)
                for item in map_rows
            ),
            "map_count": len(map_rows),
        }
        for threshold in config["gamma_threshold_grid"]
    ]
    return {
        "role": config["retrospective_d4b"]["role"],
        "historical_machine_decision": result["machine_decision"],
        "historical_decision_changed": False,
        "source_hashes": {key: _sha256(path) for key, path in paths.items()},
        "map_rows": map_rows,
        "threshold_counts": threshold_counts,
        "historical_counts": result["counts"],
        "postopen_support_flip_count": int(
            forensics["topology_summary"]["support_flip_count"]
        ),
        "interpretation": "post-open descriptive comparison only; no gamma threshold is selected and no D4b gate or authorization changes",
    }


def _plot(
    config: dict[str, Any],
    base_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    retrospective: dict[str, Any],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    low = [
        row
        for row in base_rows
        if row["scenario"] == "clean_low_bilinear_signal" and row["probe_count"] == 1
    ]
    axes[0, 0].scatter(
        [row["maximum_dot_condition_proxy"] for row in low],
        [row["maximum_dot_relative_defect"] for row in low],
        label="traditional relative",
        color="#d1495b",
    )
    axes[0, 0].scatter(
        [row["maximum_dot_condition_proxy"] for row in low],
        [row["maximum_gamma_scaled_normwise_score"] for row in low],
        label="gamma-scaled normwise",
        color="#00798c",
    )
    axes[0, 0].axhline(
        float(config["traditional_dot_relative_threshold"]),
        color="#d1495b",
        linestyle="--",
        alpha=0.7,
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("dot condition proxy")
    axes[0, 0].set_ylabel("reported score (different scales)")
    axes[0, 0].set_title("Correct low-signal probes expose relative-dot false rejects")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)

    thresholds = [float(value) for value in config["gamma_threshold_grid"]]
    probes = [int(value) for value in config["probe_counts"]]
    matrix = np.zeros((len(thresholds), len(probes)), dtype=np.float64)
    for item in aggregate["threshold_probe_summary"]:
        i = thresholds.index(float(item["gamma_threshold"]))
        j = probes.index(int(item["probe_count"]))
        matrix[i, j] = float(item["fault_detection_rate"])
    image = axes[0, 1].imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 1].set_xticks(range(len(probes)), probes)
    axes[0, 1].set_yticks(range(len(thresholds)), [f"{value:g}" for value in thresholds])
    axes[0, 1].set_xlabel("frozen random tangent probes")
    axes[0, 1].set_ylabel("gamma threshold (no value selected)")
    axes[0, 1].set_title("Combined fault detection rate")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

    for scenario, color, marker in (
        ("separate_cancellation_residual", "#d1495b", "o"),
        ("paired_cancellation_residual", "#2a9d8f", "s"),
    ):
        members = [
            row
            for row in base_rows
            if row["scenario"] == scenario and row["probe_count"] == 1
        ]
        grouped: dict[float, list[float]] = {}
        for row in members:
            grouped.setdefault(float(row["parameter_value"]), []).append(
                float(row["maximum_gamma_scaled_normwise_score"])
            )
        axes[1, 0].plot(
            sorted(grouped),
            [float(np.median(grouped[key])) for key in sorted(grouped)],
            marker=marker,
            color=color,
            label=scenario.replace("_", " "),
        )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_xlabel("component difference scale")
    axes[1, 0].set_ylabel("median gamma-scaled normwise score")
    axes[1, 0].set_title("Paired arithmetic suppresses cancellation amplification")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.2)

    colors = {
        "curved_detector": "#264653",
        "straight_detector": "#2a9d8f",
        "raw_curved_minus_straight": "#e76f51",
        "paired_neumaier_residual": "#f4a261",
    }
    for map_id, color in colors.items():
        members = [
            row for row in retrospective["map_rows"] if row["map_id"] == map_id
        ]
        axes[1, 1].scatter(
            [row["dot_condition_proxy"] for row in members],
            [row["gamma_scaled_normwise_score"] for row in members],
            s=24,
            alpha=0.75,
            label=map_id.replace("_", " "),
            color=color,
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("D4b dot condition proxy")
    axes[1, 1].set_ylabel("D4b gamma-scaled normwise score")
    axes[1, 1].set_title("Frozen D4b shown only as post-open context")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(
        "N5-D4c MSRA development: metric family and blind-spot stress test",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary(result: dict[str, Any]) -> str:
    headline = result["headline_diagnostics"]
    return "\n".join(
        (
            "# N5-D4c MSRA development screen",
            "",
            f"- Decision: `{result['machine_decision']}`",
            f"- Synthetic base rows: `{result['counts']['synthetic_base_rows']}`",
            f"- Threshold/probe evaluations: `{result['counts']['threshold_probe_evaluations']}`",
            f"- Correct low-signal probes rejected by the frozen traditional relative gate: `{headline['low_signal_traditional_reject_count']}/{headline['low_signal_count']}`",
            f"- Self-consistent wrong derivative accepted by adjoint identity but rejected by FD: `{headline['self_consistent_adjoint_blind_fd_reject_count']}/{headline['self_consistent_count']}`",
            f"- Diagnostic-only support flips leave current forward branch gate open: `{headline['diagnostic_support_report_only_count']}/{headline['diagnostic_support_count']}`",
            f"- Actual hard-branch crossings rejected by the branch gate: `{headline['hard_branch_reject_count']}/{headline['hard_branch_count']}`",
            f"- Historical D4b decision retained: `{result['retrospective_d4b']['historical_machine_decision']}`",
            "- No gamma threshold was selected; this development artifact authorizes no derivative, reconstruction, model, real-data, generalization, or paper claim.",
            "",
        )
    )


def run(
    config_path: Path,
    output: Path,
    *,
    require_committed_source: bool = True,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output = output.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    expected_output = _resolve(config["output"])
    if output != expected_output and require_committed_source:
        raise ValueError("formal D4c output path drifted from preregistration")
    if output.exists() or os.path.lexists(output):
        raise FileExistsError(f"refusing to replace existing D4c output: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or os.path.lexists(temporary):
        raise FileExistsError(f"D4c temporary path already exists: {temporary}")

    source_paths = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "demo_t16_operator/mixed_scale_adjoint_certificate.py",
        ROOT / "demo_t16_operator/test_mixed_scale_adjoint_certificate.py",
        ROOT / "demo_t16_operator/test_run_n2_pvgr_n5_d4c_msra_development.py",
    ]
    protocol_commit = (
        _assert_committed(source_paths)
        if require_committed_source
        else "TEST_UNCOMMITTED_SOURCE_ALLOWED"
    )
    source_hashes_before = {_relative(path): _sha256(path) for path in source_paths}

    base_rows = _build_synthetic_rows(config)
    evaluated_rows = _gate_rows(config, base_rows)
    aggregate = _aggregate(config, evaluated_rows)
    retrospective = _retrospective_d4b(config)
    low_signal = [
        row
        for row in base_rows
        if row["scenario"] == "clean_low_bilinear_signal" and row["probe_count"] == 1
    ]
    self_consistent = [
        row
        for row in evaluated_rows
        if row["scenario"] == "self_consistent_wrong_derivative"
        and row["probe_count"] == 1
        and row["gamma_threshold"] == float(config["gamma_threshold_grid"][0])
    ]
    support = [
        row
        for row in evaluated_rows
        if row["scenario"] == "diagnostic_only_support_flip"
        and row["probe_count"] == 1
        and row["gamma_threshold"] == float(config["gamma_threshold_grid"][0])
    ]
    hard = [
        row
        for row in evaluated_rows
        if row["scenario"] == "hard_branch_crossing"
        and row["probe_count"] == 1
        and row["gamma_threshold"] == float(config["gamma_threshold_grid"][0])
    ]
    source_hashes_after = {_relative(path): _sha256(path) for path in source_paths}
    if source_hashes_before != source_hashes_after:
        raise ValueError("a D4c source changed during execution")

    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": config["candidate_id"],
        "status": "development_characterization_only_no_authorization",
        "machine_decision": config["decision_contract"]["decision"],
        "protocol_commit": protocol_commit,
        "config_sha256": _sha256(config_path),
        "source_hashes": source_hashes_before,
        "threshold_selected": None,
        "threshold_selection_forbidden": True,
        "probe_counts": config["probe_counts"],
        "gamma_threshold_grid": config["gamma_threshold_grid"],
        "counts": {
            "synthetic_base_rows": len(base_rows),
            "threshold_probe_evaluations": len(evaluated_rows),
            "trial_count": int(config["trial_count"]),
            "scenario_count": len({row["scenario"] for row in base_rows}),
        },
        "headline_diagnostics": {
            "low_signal_count": len(low_signal),
            "low_signal_traditional_reject_count": sum(
                not row["traditional_dot_gate"] for row in low_signal
            ),
            "low_signal_maximum_gamma_score": max(
                row["maximum_gamma_scaled_normwise_score"] for row in low_signal
            ),
            "self_consistent_count": len(self_consistent),
            "self_consistent_adjoint_blind_fd_reject_count": sum(
                row["gamma_gate"] and not row["fd_gate"] for row in self_consistent
            ),
            "diagnostic_support_count": len(support),
            "diagnostic_support_report_only_count": sum(
                row["diagnostic_support_changed"]
                and row["branch_gate"]
                and row["combined_gate"]
                for row in support
            ),
            "hard_branch_count": len(hard),
            "hard_branch_reject_count": sum(not row["branch_gate"] for row in hard),
        },
        "branch_semantic_ledger": config["branch_semantic_ledger"],
        "aggregate": aggregate,
        "retrospective_d4b": retrospective,
        "claim_authorizations": dict(config["claim_authorizations"]),
        "forbidden_call_counts": {
            "bost_forward": 0,
            "bost_jvp": 0,
            "bost_vjp": 0,
            "decoder": 0,
            "reconstruction": 0,
            "training": 0,
        },
        "limitations": [
            "synthetic explicit matrices are fault-injection controls, not BOST physics validation",
            "gamma_n bounds only the final contractions unless a full autodiff rounding analysis is added",
            "random probes provide probabilistic coverage and cannot prove a full 4913-dimensional VJP",
            "self-consistent wrong derivatives can satisfy every adjoint identity and therefore require independent FD or oracle evidence",
            "support/frustum are diagnostic-only in the current renderer but the laboratory renderer semantics remain unknown",
            "retrospective D4b rows are post-open context and cannot select a threshold or change the historical decision",
        ],
        "next_legal_step": "independently audit this development artifact, then freeze a fresh BOST field/rig population and a threshold/probe rule before any derivative authorization",
    }

    temporary.mkdir(parents=True)
    _write_json(temporary / "result.json", result)
    _write_csv(temporary / "synthetic_rows.csv", base_rows)
    _write_csv(temporary / "threshold_probe_rows.csv", evaluated_rows)
    _plot(
        config,
        base_rows,
        aggregate,
        retrospective,
        temporary / "n2_pvgr_n5_d4c_msra_development.png",
    )
    (temporary / "summary.md").write_text(_summary(result), encoding="utf-8")
    manifest = {
        "schema": "n2-pvgr-n5-d4c-msra-development-manifest-1.0",
        "protocol_commit": protocol_commit,
        "config_sha256": result["config_sha256"],
        "source_hashes": source_hashes_before,
        "artifacts": {
            name: {
                "bytes": (temporary / name).stat().st_size,
                "sha256": _sha256(temporary / name),
            }
            for name in (
                "result.json",
                "synthetic_rows.csv",
                "threshold_probe_rows.csv",
                "n2_pvgr_n5_d4c_msra_development.png",
                "summary.md",
            )
        },
    }
    _write_json(temporary / "manifest.json", manifest)
    os.replace(temporary, output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.config, args.output)
    print(
        json.dumps(
            {
                "machine_decision": result["machine_decision"],
                "protocol_commit": result["protocol_commit"],
                "counts": result["counts"],
                "headline_diagnostics": result["headline_diagnostics"],
                "threshold_selected": result["threshold_selected"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
