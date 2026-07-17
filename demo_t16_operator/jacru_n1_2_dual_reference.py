"""Truth-side dual-reference gates for the opened N1.2 synthetic screen.

This module is deliberately evaluator-only.  A reconstruction selector must
not import it because every row contains field errors against withheld truth.
It turns per-case metrics into one immutable comparison contract: a candidate
must improve one pre-registered matched classical reconstruction while also
not damaging the raw learned proposal that supplied its centre.  A per-case
best-classical envelope may still be reported elsewhere, but it cannot be the
primary gate because selecting that reference reads evaluator-side field truth.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence


def _finite(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _positive(value: Any, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(materialized) / len(materialized)


def add_dual_reference_metrics(
    row: Mapping[str, Any],
    *,
    harm_threshold_fraction: float,
) -> dict[str, Any]:
    """Add gains against raw and strongest matched classical references."""

    threshold = _finite(harm_threshold_fraction, "harm_threshold_fraction")
    if threshold < 0.0:
        raise ValueError("harm_threshold_fraction must be nonnegative")
    required = (
        "candidate_field_relative_l2",
        "candidate_h1_relative_error",
        "raw_field_relative_l2",
        "raw_h1_relative_error",
        "registered_classical_field_relative_l2",
        "registered_classical_h1_relative_error",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"dual-reference row is missing fields: {missing}")
    candidate_field = _positive(row[required[0]], required[0])
    candidate_h1 = _positive(row[required[1]], required[1])
    raw_field = _positive(row[required[2]], required[2])
    raw_h1 = _positive(row[required[3]], required[3])
    classical_field = _positive(row[required[4]], required[4])
    classical_h1 = _positive(row[required[5]], required[5])
    field_gain_raw = (raw_field - candidate_field) / raw_field
    h1_gain_raw = (raw_h1 - candidate_h1) / raw_h1
    field_gain_classical = (classical_field - candidate_field) / classical_field
    h1_gain_classical = (classical_h1 - candidate_h1) / classical_h1
    return {
        **dict(row),
        "field_gain_to_raw": field_gain_raw,
        "h1_gain_to_raw": h1_gain_raw,
        "field_gain_to_registered_classical": field_gain_classical,
        "h1_gain_to_registered_classical": h1_gain_classical,
        "field_harm_vs_raw": field_gain_raw < -threshold,
        "field_harm_vs_registered_classical": field_gain_classical < -threshold,
    }


def _minimum_group_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    metric: str,
) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(_finite(row[metric], metric))
    if not grouped:
        raise ValueError(f"no groups for {group_key}")
    means = {key: _mean(values) for key, values in sorted(grouped.items())}
    return min(means.values()), means


def aggregate_dual_reference_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate case rows without allowing seed/session/family tails to vanish."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["candidate_id"]), str(row["method"]), str(row["split"]))
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (candidate_id, method, split), group in sorted(groups.items()):
        seed_raw: dict[int, list[float]] = defaultdict(list)
        seed_classical: dict[int, list[float]] = defaultdict(list)
        for row in group:
            seed = int(row["model_seed"])
            seed_raw[seed].append(_finite(row["field_gain_to_raw"], "field_gain_to_raw"))
            seed_classical[seed].append(
                _finite(
                    row["field_gain_to_registered_classical"],
                    "field_gain_to_registered_classical",
                )
            )
        session_raw_min, session_raw = _minimum_group_mean(
            group, group_key="session_id", metric="field_gain_to_raw"
        )
        session_classical_min, session_classical = _minimum_group_mean(
            group, group_key="session_id", metric="field_gain_to_registered_classical"
        )
        family_raw_min, family_raw = _minimum_group_mean(
            group, group_key="family", metric="field_gain_to_raw"
        )
        family_classical_min, family_classical = _minimum_group_mean(
            group, group_key="family", metric="field_gain_to_registered_classical"
        )
        output.append(
            {
                "candidate_id": candidate_id,
                "method": method,
                "split": split,
                "row_count": len(group),
                "field_gain_to_raw_mean": _mean(row["field_gain_to_raw"] for row in group),
                "h1_gain_to_raw_mean": _mean(row["h1_gain_to_raw"] for row in group),
                "field_gain_to_registered_classical_mean": _mean(
                    row["field_gain_to_registered_classical"] for row in group
                ),
                "h1_gain_to_registered_classical_mean": _mean(
                    row["h1_gain_to_registered_classical"] for row in group
                ),
                "field_harm_rate_vs_raw": _mean(bool(row["field_harm_vs_raw"]) for row in group),
                "field_harm_rate_vs_registered_classical": _mean(
                    bool(row["field_harm_vs_registered_classical"]) for row in group
                ),
                "worst_field_gain_to_raw": min(float(row["field_gain_to_raw"]) for row in group),
                "worst_field_gain_to_registered_classical": min(
                    float(row["field_gain_to_registered_classical"]) for row in group
                ),
                "minimum_session_mean_field_gain_to_raw": session_raw_min,
                "minimum_session_mean_field_gain_to_registered_classical": session_classical_min,
                "minimum_family_mean_field_gain_to_raw": family_raw_min,
                "minimum_family_mean_field_gain_to_registered_classical": family_classical_min,
                "session_mean_field_gains_to_raw": session_raw,
                "session_mean_field_gains_to_registered_classical": session_classical,
                "family_mean_field_gains_to_raw": family_raw,
                "family_mean_field_gains_to_registered_classical": family_classical,
                "per_model_seed_field_gain_means_to_raw": {
                    str(seed): _mean(values) for seed, values in sorted(seed_raw.items())
                },
                "per_model_seed_field_gain_means_to_registered_classical": {
                    str(seed): _mean(values) for seed, values in sorted(seed_classical.items())
                },
                "all_model_seed_means_to_raw_nonnegative": all(
                    _mean(values) >= 0.0 for values in seed_raw.values()
                ),
                "all_model_seed_means_to_registered_classical_positive": all(
                    _mean(values) > 0.0 for values in seed_classical.values()
                ),
                "clean_reprojection_ratio_to_base_mean": _mean(
                    _finite(row["clean_reprojection_ratio_to_base"], "clean ratio")
                    for row in group
                ),
                "clean_reprojection_ratio_to_base_maximum": max(
                    _finite(row["clean_reprojection_ratio_to_base"], "clean ratio")
                    for row in group
                ),
                "selector_valid_rate": _mean(bool(row["selector_valid"]) for row in group),
                "residual_closure_relative_error_maximum": max(
                    _finite(row["residual_closure_relative_error"], "closure error")
                    for row in group
                ),
            }
        )
    return output


def dual_reference_decisions(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    candidate_metadata: Mapping[str, Mapping[str, Any]],
    candidate_calibration_sanity_passed: Mapping[str, bool],
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply one frozen gate set to development and OOD aggregates."""

    lookup = {
        (str(row["candidate_id"]), str(row["method"]), str(row["split"])): row
        for row in aggregates
    }
    pairs = sorted({(key[0], key[1]) for key in lookup})
    decisions: list[dict[str, Any]] = []
    for candidate_id, method in pairs:
        if candidate_id not in candidate_metadata:
            raise ValueError(f"missing metadata for candidate {candidate_id}")
        try:
            development = lookup[(candidate_id, method, "development")]
            ood = lookup[(candidate_id, method, "ood")]
        except KeyError as error:
            raise ValueError(f"candidate {candidate_id}/{method} lacks both splits") from error

        checks: dict[str, bool] = {
            "candidate_specific_calibration_sanity": bool(
                candidate_calibration_sanity_passed.get(candidate_id, False)
            ),
            "development_classical_field_mean": float(
                development["field_gain_to_registered_classical_mean"]
            ) >= float(gates["development_classical_field_gain_mean_minimum"]),
            "development_classical_h1_mean": float(
                development["h1_gain_to_registered_classical_mean"]
            ) >= float(gates["development_classical_h1_gain_mean_minimum"]),
            "ood_classical_field_mean": float(ood["field_gain_to_registered_classical_mean"])
            >= float(gates["ood_classical_field_gain_mean_minimum"]),
            "ood_classical_h1_mean": float(ood["h1_gain_to_registered_classical_mean"])
            >= float(gates["ood_classical_h1_gain_mean_minimum"]),
            "development_raw_field_mean": float(development["field_gain_to_raw_mean"])
            >= float(gates["raw_field_gain_mean_minimum"]),
            "ood_raw_field_mean": float(ood["field_gain_to_raw_mean"])
            >= float(gates["raw_field_gain_mean_minimum"]),
            "development_raw_harm": float(development["field_harm_rate_vs_raw"])
            <= float(gates["field_harm_rate_maximum"]),
            "ood_raw_harm": float(ood["field_harm_rate_vs_raw"])
            <= float(gates["field_harm_rate_maximum"]),
            "development_classical_harm": float(
                development["field_harm_rate_vs_registered_classical"]
            ) <= float(gates["field_harm_rate_maximum"]),
            "ood_classical_harm": float(ood["field_harm_rate_vs_registered_classical"])
            <= float(gates["field_harm_rate_maximum"]),
            "development_raw_worst": float(development["worst_field_gain_to_raw"])
            >= float(gates["worst_field_gain_minimum"]),
            "ood_raw_worst": float(ood["worst_field_gain_to_raw"])
            >= float(gates["worst_field_gain_minimum"]),
            "development_classical_worst": float(
                development["worst_field_gain_to_registered_classical"]
            ) >= float(gates["worst_field_gain_minimum"]),
            "ood_classical_worst": float(ood["worst_field_gain_to_registered_classical"])
            >= float(gates["worst_field_gain_minimum"]),
            "session_tail_raw": min(
                float(development["minimum_session_mean_field_gain_to_raw"]),
                float(ood["minimum_session_mean_field_gain_to_raw"]),
            ) >= float(gates["minimum_session_mean_field_gain"]),
            "session_tail_classical": min(
                float(development["minimum_session_mean_field_gain_to_registered_classical"]),
                float(ood["minimum_session_mean_field_gain_to_registered_classical"]),
            ) >= float(gates["minimum_session_mean_field_gain"]),
            "family_tail_raw": min(
                float(development["minimum_family_mean_field_gain_to_raw"]),
                float(ood["minimum_family_mean_field_gain_to_raw"]),
            ) >= float(gates["minimum_family_mean_field_gain"]),
            "family_tail_classical": min(
                float(development["minimum_family_mean_field_gain_to_registered_classical"]),
                float(ood["minimum_family_mean_field_gain_to_registered_classical"]),
            ) >= float(gates["minimum_family_mean_field_gain"]),
            "all_seed_means": bool(development["all_model_seed_means_to_raw_nonnegative"])
            and bool(ood["all_model_seed_means_to_raw_nonnegative"])
            and bool(development["all_model_seed_means_to_registered_classical_positive"])
            and bool(ood["all_model_seed_means_to_registered_classical_positive"]),
            "development_clean_mean": float(
                development["clean_reprojection_ratio_to_base_mean"]
            ) <= float(gates["development_clean_ratio_mean_maximum"]),
            "development_clean_worst": float(
                development["clean_reprojection_ratio_to_base_maximum"]
            ) <= float(gates["development_clean_ratio_worst_maximum"]),
            "ood_clean_mean": float(ood["clean_reprojection_ratio_to_base_mean"])
            <= float(gates["ood_clean_ratio_mean_maximum"]),
            "ood_clean_worst": float(ood["clean_reprojection_ratio_to_base_maximum"])
            <= float(gates["ood_clean_ratio_worst_maximum"]),
            "selector_valid_rate": min(
                float(development["selector_valid_rate"]),
                float(ood["selector_valid_rate"]),
            ) >= float(gates["selector_valid_rate_minimum"]),
            "residual_closure": max(
                float(development["residual_closure_relative_error_maximum"]),
                float(ood["residual_closure_relative_error_maximum"]),
            ) <= float(gates["residual_closure_relative_error_maximum"]),
        }
        metadata = dict(candidate_metadata[candidate_id])
        decisions.append(
            {
                "candidate_id": candidate_id,
                "method": method,
                "uses_truth": bool(metadata.get("uses_truth", False)),
                "uses_exact_nuisance": bool(metadata.get("uses_exact_nuisance", False)),
                "selector_family": str(metadata.get("selector_family", "unknown")),
                "checks": checks,
                "development": dict(development),
                "ood": dict(ood),
                "passed": all(checks.values()),
            }
        )
    return decisions


__all__ = [
    "add_dual_reference_metrics",
    "aggregate_dual_reference_rows",
    "dual_reference_decisions",
]
