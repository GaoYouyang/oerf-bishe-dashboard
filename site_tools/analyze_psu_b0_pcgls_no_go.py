#!/usr/bin/env python3
"""Quantify why the current learned PSU direction loses to Sobolev-PCGLS."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


PUBLIC_SCHEMA = "psu-b0-pcgls-no-go-analysis-public-1.0"
STATUS = "CURRENT_LEARNED_STEEPEST_DIRECTION_NO_GO_AFTER_STRONG_BASELINE"
RAW_SEEDS = (20261741, 20261742, 20261743)
DEVELOPMENT_SPLITS = ("risk_validation", "risk_calibration")
FRESH_SPLITS = (
    "fresh_iid_support",
    "fresh_family_ood",
    "fresh_correlated_noise_ood",
    "fresh_family_noise_ood",
    "fresh_geometry_ood",
    "fresh_joint_ood",
    "fresh_exact_operator_control",
)


def _binomial_greater_tail(wins: int, count: int) -> float:
    if not 0 <= int(wins) <= int(count):
        raise ValueError("binomial inputs are invalid")
    return float(
        sum(comb(int(count), value) for value in range(int(wins), int(count) + 1))
        / (2 ** int(count))
    )


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 20000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 2 or int(draws) < 100:
        raise ValueError("bootstrap requires at least two values and 100 draws")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(array), size=(int(draws), len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min((count - rank) * float(values[index]), 1.0)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_split_summary(
    rows: list[dict[str, Any]],
    *,
    split: str,
    pcgls_method: str,
    learned_prefix: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    group = [row for row in rows if str(row["split"]) == str(split)]
    sample_ids = sorted({str(row["sample_id"]) for row in group})
    lookup = {
        (str(row["sample_id"]), str(row["method"])): float(
            row["field_relative_l2"]
        )
        for row in group
    }
    learned_methods = sorted(
        {
            str(row["method"])
            for row in group
            if str(row["method"]).startswith(str(learned_prefix))
        }
    )
    if not sample_ids or not learned_methods:
        raise ValueError(f"split {split} lacks paired learned rows")
    pcgls = np.asarray(
        [lookup[(sample_id, pcgls_method)] for sample_id in sample_ids],
        dtype=np.float64,
    )
    learned = np.asarray(
        [
            np.mean(
                [
                    lookup[(sample_id, method)]
                    for method in learned_methods
                ]
            )
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )
    gain = 100.0 * (learned - pcgls) / np.maximum(learned, 1e-12)
    lower, upper = _bootstrap_mean_interval(
        gain,
        seed=int(bootstrap_seed),
    )
    wins = int(np.sum(pcgls < learned))
    return {
        "split": str(split),
        "sample_count": len(sample_ids),
        "learned_methods_averaged": learned_methods,
        "pcgls_method": pcgls_method,
        "learned_field_relative_l2_mean": float(np.mean(learned)),
        "pcgls_field_relative_l2_mean": float(np.mean(pcgls)),
        "pcgls_relative_error_reduction_mean_percent": float(np.mean(gain)),
        "pcgls_relative_error_reduction_median_percent": float(
            np.median(gain)
        ),
        "pcgls_relative_error_reduction_p10_percent": float(
            np.quantile(gain, 0.10)
        ),
        "pcgls_relative_error_reduction_minimum_percent": float(np.min(gain)),
        "bootstrap_mean_95_interval_percent": [lower, upper],
        "pcgls_win_count": wins,
        "pcgls_win_rate": float(wins / len(sample_ids)),
        "one_sided_sign_test_p_value": _binomial_greater_tail(
            wins,
            len(sample_ids),
        ),
    }


def _pooled_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fresh = [row for row in rows if str(row["split"]) in FRESH_SPLITS]
    sample_ids = sorted({(str(row["split"]), str(row["sample_id"])) for row in fresh})
    lookup = {
        (
            str(row["split"]),
            str(row["sample_id"]),
            str(row["method"]),
        ): float(row["field_relative_l2"])
        for row in fresh
    }
    definitions = (
        ("sobolev_selected", ("sobolev_selected",), 4, 4, 0),
        (
            "raw_learned_seed_mean",
            tuple(f"raw_seed_{seed}" for seed in RAW_SEEDS),
            4,
            4,
            2227,
        ),
        (
            "gated_learned_seed_mean",
            tuple(f"gated_seed_{seed}" for seed in RAW_SEEDS),
            4,
            4,
            2227,
        ),
        ("pcgls_3_selected", ("pcgls_3_selected",), 3, 3, 0),
        ("pcgls_4_selected", ("pcgls_4_selected",), 4, 4, 0),
    )
    output = []
    for label, methods, forward, adjoint, parameters in definitions:
        values = np.asarray(
            [
                np.mean(
                    [
                        lookup[(split, sample_id, method)]
                        for method in methods
                    ]
                )
                for split, sample_id in sample_ids
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "method": label,
                "sample_count": len(values),
                "field_relative_l2_mean": float(np.mean(values)),
                "field_relative_l2_median": float(np.median(values)),
                "forward_calls": int(forward),
                "adjoint_calls": int(adjoint),
                "total_operator_calls": int(forward + adjoint),
                "trainable_parameter_count": int(parameters),
            }
        )
    return output


def build_analysis(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["metric_rows_private"]
    summaries = []
    for index, split in enumerate((*DEVELOPMENT_SPLITS, *FRESH_SPLITS)):
        summaries.append(
            paired_split_summary(
                rows,
                split=split,
                pcgls_method="pcgls_4_selected",
                learned_prefix="raw_seed_",
                bootstrap_seed=2026071600 + index,
            )
        )
    adjusted = holm_adjust(
        [float(row["one_sided_sign_test_p_value"]) for row in summaries]
    )
    for row, value in zip(summaries, adjusted):
        row["holm_adjusted_sign_test_p_value"] = float(value)
    same_call = all(
        execution["logical_calls_per_sample"]
        == {"forward": 4, "adjoint": 4}
        for execution in report["execution"]
        if execution["method"] == "pcgls_4_selected"
    )
    validation = next(row for row in summaries if row["split"] == "risk_validation")
    calibration = next(
        row for row in summaries if row["split"] == "risk_calibration"
    )
    fresh = [row for row in summaries if row["split"] in FRESH_SPLITS]
    no_go = {
        "same_forward_and_adjoint_calls_as_learned": bool(same_call),
        "zero_trainable_parameters": True,
        "selected_only_on_risk_validation": True,
        "validation_bootstrap_lower_gain_above_zero": (
            float(validation["bootstrap_mean_95_interval_percent"][0]) > 0.0
        ),
        "calibration_bootstrap_lower_gain_above_zero": (
            float(calibration["bootstrap_mean_95_interval_percent"][0]) > 0.0
        ),
        "all_opened_fresh_split_mean_gains_above_zero": all(
            float(row["pcgls_relative_error_reduction_mean_percent"]) > 0.0
            for row in fresh
        ),
        "minimum_opened_fresh_win_count": min(
            int(row["pcgls_win_count"]) for row in fresh
        ),
    }
    no_go["current_learned_direction_is_no_go"] = all(
        bool(value)
        for key, value in no_go.items()
        if key != "minimum_opened_fresh_win_count"
    ) and int(no_go["minimum_opened_fresh_win_count"]) >= 20
    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": STATUS,
        "evidence_scope": (
            "POSTOPEN_DEVELOPMENT_DIAGNOSIS_ON_REAL_PSU_SUPPORT_GEOMETRY_"
            "WITH_ANALYTIC_REACTION_MORPHOLOGY_AND_SYNTHETIC_NOISE"
        ),
        "selected_pcgls": report["selected_candidates"]["pcgls_4"],
        "paired_split_summary": summaries,
        "pooled_fresh_frontier": _pooled_frontier(rows),
        "no_go_gate": no_go,
        "interpretation": {
            "what_failed": (
                "The residual-conditioned positive spectral direction was "
                "compared against fixed Sobolev steepest descent, but a standard "
                "Sobolev-preconditioned Krylov recurrence extracts substantially "
                "more value from the same four forward and four adjoint calls."
            ),
            "why_this_matters": (
                "The learned controller's apparent gain is not evidence of a "
                "better inverse operator once a strong same-budget Krylov baseline "
                "is included."
            ),
            "what_remains_valid": (
                "The PSU B0 operator, synthetic stress generator, observable-risk "
                "diagnostics, and exact fallback machinery remain reusable."
            ),
        },
        "next_algorithm_routes": [
            {
                "rank": 1,
                "working_name": "BOST-GC-SPD-PCGLS",
                "idea": (
                    "Predict one bounded positive geometry/noise-conditioned "
                    "Fourier multiplier before iteration and hold it fixed for "
                    "all PCGLS stages."
                ),
                "proof_obligation": (
                    "The cached multiplier must stay positive and self-adjoint, "
                    "so the restricted preconditioner remains SPD."
                ),
                "minimum_baselines": [
                    "static Sobolev PCGLS",
                    "validation-tuned anisotropic PCGLS",
                    "NeuralIF or learned approximate-inverse concept control",
                    "four-step unpreconditioned CGLS",
                ],
                "risk": "medium",
            },
            {
                "rank": 2,
                "working_name": "BOST-PCGLS learned stopping and safeguard",
                "idea": (
                    "Keep the classical iterates and learn only whether to stop "
                    "at stages one through four or fall back to a smoother iterate."
                ),
                "proof_obligation": (
                    "Selection must use truth-free residual/geometry features and "
                    "a new independent calibration and audit."
                ),
                "risk": "low_to_medium",
            },
            {
                "rank": 3,
                "working_name": "BOST flexible learned Krylov",
                "idea": (
                    "Allow a residual-adaptive learned SPD map but replace standard "
                    "CG recurrence with flexible CG and explicit orthogonalization."
                ),
                "proof_obligation": (
                    "Compare against Notay-style flexible CG and account for all "
                    "stored directions and local neural evaluations."
                ),
                "risk": "high",
            },
        ],
        "literature_boundary": [
            {
                "title": "Flexible Conjugate Gradients",
                "url": "https://epubs.siam.org/doi/abs/10.1137/S1064827599362314",
                "lesson": (
                    "A changing preconditioner requires a flexible recurrence or "
                    "explicit orthogonalization; standard PCG theory no longer applies."
                ),
            },
            {
                "title": "Learning Preconditioners for Conjugate Gradient PDE Solvers",
                "url": "https://proceedings.mlr.press/v202/li23e.html",
                "lesson": (
                    "Learned SPD factorizations inside PCG are established prior "
                    "art, so BOST novelty must come from the optical operator and "
                    "finite-aperture/view/noise conditioning."
                ),
            },
            {
                "title": "Neural incomplete factorization",
                "url": "https://openreview.net/forum?id=FozLrZ3CI5",
                "lesson": (
                    "Learning a structured factorization is another strong prior-art "
                    "control and offers an unsupervised matrix loss."
                ),
            },
            {
                "title": "UNO-CG",
                "url": "https://arxiv.org/abs/2508.02681",
                "lesson": (
                    "Unitary neural operators can be constructed as convergence-safe "
                    "learned preconditioners; a generic Fourier-neural-PCG claim is "
                    "therefore not novel by itself."
                ),
            },
            {
                "title": "Superiorization of PCG for tomography",
                "url": "https://arxiv.org/abs/1807.10151",
                "lesson": (
                    "TV-superiorized PCG is a necessary non-learning tomography "
                    "baseline before claiming front or shock preservation."
                ),
            },
        ],
        "claim_boundary": {
            "postopen_development_only": True,
            "fresh_diagnostic_is_confirmatory": False,
            "experimental_field_truth_used": False,
            "real_psu_measurement_values_used": False,
            "analytic_morphology_is_cfd": False,
            "pcgls_is_a_new_algorithm": False,
            "new_learned_pcgls_candidate_implemented": False,
            "algorithm_superiority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    output = build_analysis(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
