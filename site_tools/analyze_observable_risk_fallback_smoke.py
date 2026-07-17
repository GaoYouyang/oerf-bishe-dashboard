#!/usr/bin/env python3
"""Export a bounded public diagnostic from a strictly validated RCCF v4 bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from site_tools.run_observable_risk_fallback_smoke import RISK_FIELDS
from site_tools.validate_observable_risk_fallback_smoke import validate_result_bundle


PUBLIC_SCHEMA = "observable-risk-fallback-smoke-public-1.0"
PUBLIC_STATUS = "FAILED_NO_AUTHORITY_DEVELOPMENT_ONLY_NEGATIVE_RESULT"
PUBLIC_FILES = frozenset(
    {
        "README.md",
        "summary.json",
        "calibration_inspection.csv",
        "fresh_inspection.csv",
        "diagnostic.png",
    }
)
CALIBRATION_FIELDS = (
    "rig_id",
    "candidate_partition",
    "risk_score",
    "acceptance_threshold",
    "support_gate_passed",
    "harm_failure",
    "fallback_used",
)
FRESH_FIELDS = (
    "rig_id",
    "candidate_partition",
    "risk_score",
    "acceptance_threshold",
    "support_gate_passed",
    "fallback_used",
    "observed_field_harm_vs_fallback",
    "observed_residual_harm_vs_fallback",
    "harm_failure",
)
PUBLIC_CLAIM_BOUNDARY = {
    "research_claim_authorized": False,
    "real_bost_claim_authorized": False,
    "generalization_claim_authorized": False,
    "superiority_claim_authorized": False,
    "paper_superiority_claim_authorized": False,
    "algorithm_superiority_claimed": False,
    "deeponet_fno_nerif_superiority_claimed": False,
    "timing_advantage_claimed": False,
}
SOURCE_AUTHORITY_FLAGS = (
    "research_claim_authorized",
    "real_bost_claim_authorized",
    "generalization_claim_authorized",
    "paper_superiority_claim_authorized",
)


class PublicExportError(ValueError):
    """Raised when a source or public-output boundary is violated."""


def _reject_constant(raw: str) -> None:
    raise PublicExportError(f"non-finite JSON constant is forbidden: {raw}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PublicExportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise PublicExportError(f"{path.name} must contain a JSON object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicExportError(f"validated report has no {name} object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PublicExportError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PublicExportError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise PublicExportError(f"{name} must be finite")
    return number


def _count(value: Any, name: str) -> int:
    number = _finite(value, name)
    if not number.is_integer() or number < 0:
        raise PublicExportError(f"{name} must be a nonnegative integer")
    return int(number)


def _read_risk_rows(input_dir: Path) -> list[dict[str, str]]:
    path = input_dir / "risk_rows.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RISK_FIELDS:
            raise PublicExportError("risk_rows.csv schema drift")
        rows = list(reader)
    if not rows or any(None in row or tuple(row) != RISK_FIELDS for row in rows):
        raise PublicExportError("risk_rows.csv contains malformed rows")
    return rows


def _reduced_rows(
    rows: Sequence[Mapping[str, str]], split_role: str, fields: Sequence[str]
) -> list[dict[str, str]]:
    return [
        {field: row[field] for field in fields}
        for row in rows
        if row["split_role"] == split_role
    ]


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output(input_dir: Path, output_dir: Path) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise PublicExportError("public output must differ from the full result directory")
    if output_dir.is_symlink():
        raise PublicExportError("public output cannot be a symbolic link")
    if output_dir.exists() and not output_dir.is_dir():
        raise PublicExportError("public output exists and is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = list(output_dir.iterdir())
    unexpected = {path.name for path in entries}.difference(PUBLIC_FILES)
    if unexpected or any(path.is_symlink() or not path.is_file() for path in entries):
        raise PublicExportError("public output contains unexpected or unsafe entries")
    for path in entries:
        path.unlink()


def _assert_negative_no_authority(report: Mapping[str, Any]) -> None:
    gates = _mapping(report.get("gates"), "gates")
    risk = _mapping(report.get("risk_calibration"), "risk_calibration")
    claims = _mapping(report.get("claim_boundary"), "claim_boundary")
    if gates.get("synthetic_micro_interface_gate_passed") is not False:
        raise PublicExportError("publicizer is bounded to the failed RCCF v4 result")
    if risk.get("development_gate_passed") is not False:
        raise PublicExportError("RCCF v4 development gate must remain failed")
    if any(gates.get(name) is not False for name in SOURCE_AUTHORITY_FLAGS):
        raise PublicExportError("validated source unexpectedly grants claim authority")
    if any(value is not False for value in claims.values()):
        raise PublicExportError("validated source claim boundary is not entirely false")


def _build_summary(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    aggregate = _mapping(report.get("aggregate"), "aggregate")
    risk = _mapping(report.get("risk_calibration"), "risk_calibration")
    development = _mapping(config.get("development_gate"), "development_gate")

    calibration_count = _count(aggregate.get("calibration_count"), "calibration_count")
    fresh_count = _count(aggregate.get("fresh_count"), "fresh_count")
    takeover_count = _count(
        aggregate.get("fresh_takeover_count"), "fresh_takeover_count"
    )
    if takeover_count > fresh_count:
        raise PublicExportError("fresh takeover count exceeds fresh count")

    calibration = {
        "count": calibration_count,
        "accepted_count": _count(
            aggregate.get("calibration_accepted_count"),
            "calibration_accepted_count",
        ),
        "failure_count": _count(
            aggregate.get("calibration_failure_count"),
            "calibration_failure_count",
        ),
        "risk_upper_bound": _finite(
            risk.get("risk_upper_bound"), "risk_upper_bound"
        ),
        "maximum_risk_upper_gate": _finite(
            development.get("maximum_risk_upper"), "maximum_risk_upper"
        ),
        "diagnostic_takeover_coverage": _finite(
            risk.get("takeover_coverage"), "takeover_coverage"
        ),
        "diagnostic_takeover_coverage_lower_bound": _finite(
            risk.get("takeover_coverage_lower_bound"),
            "takeover_coverage_lower_bound",
        ),
        "authorized_takeover_coverage": _finite(
            risk.get("authorized_takeover_coverage"),
            "authorized_takeover_coverage",
        ),
        "authorized_takeover_coverage_lower_bound": _finite(
            risk.get("authorized_takeover_coverage_lower_bound"),
            "authorized_takeover_coverage_lower_bound",
        ),
        "minimum_takeover_coverage_gate": _finite(
            development.get("minimum_takeover_coverage"),
            "minimum_takeover_coverage",
        ),
        "development_gate_passed": False,
    }
    if calibration["authorized_takeover_coverage"] != 0.0:
        raise PublicExportError("failed development gate must authorize zero coverage")

    fresh = {
        "count": fresh_count,
        "takeover_count": takeover_count,
        "fallback_count": fresh_count - takeover_count,
        "takeover_coverage": _finite(
            aggregate.get("fresh_takeover_coverage"), "fresh_takeover_coverage"
        ),
        "fallback_rate": _finite(
            aggregate.get("fresh_fallback_rate"), "fresh_fallback_rate"
        ),
        "harm_count": _count(aggregate.get("fresh_harm_count"), "fresh_harm_count"),
        "selection_conditional_harm_rate": aggregate.get(
            "fresh_selection_conditional_harm_rate"
        ),
        "worst_takeover_field_harm": aggregate.get(
            "fresh_worst_takeover_field_harm"
        ),
        "worst_takeover_residual_harm": aggregate.get(
            "fresh_worst_takeover_residual_harm"
        ),
    }
    for name in (
        "selection_conditional_harm_rate",
        "worst_takeover_field_harm",
        "worst_takeover_residual_harm",
    ):
        if fresh[name] is not None:
            fresh[name] = _finite(fresh[name], name)

    certificates = {
        "partition_audit_count": _count(
            aggregate.get("partition_audit_count"), "partition_audit_count"
        ),
        "partition_audit_violation_count": _count(
            aggregate.get("partition_audit_violation_count"),
            "partition_audit_violation_count",
        ),
        "operator_decomposition_mismatch_count": _count(
            aggregate.get("operator_decomposition_mismatch_count"),
            "operator_decomposition_mismatch_count",
        ),
    }
    certificates["deterministic_certificate_checks_passed"] = (
        certificates["partition_audit_violation_count"] == 0
    )
    certificates["operator_decomposition_checks_passed"] = (
        certificates["operator_decomposition_mismatch_count"] == 0
    )

    return {
        "schema_version": PUBLIC_SCHEMA,
        "status": PUBLIC_STATUS,
        "result_label": "FAILED / NO AUTHORITY / DEVELOPMENT ONLY / NEGATIVE RESULT",
        "source_result": {
            "schema_version": report.get("schema_version"),
            "status": report.get("status"),
            "evidence_scope": report.get("evidence_scope"),
        },
        "calibration": calibration,
        "fresh": fresh,
        "certificates_and_decomposition": certificates,
        "claim_boundary": dict(PUBLIC_CLAIM_BOUNDARY),
        "timing_statement": "No timing comparison or timing advantage claim is made.",
        "public_csv_schema": {
            "calibration_inspection.csv": list(CALIBRATION_FIELDS),
            "fresh_inspection.csv": list(FRESH_FIELDS),
        },
    }


def _format_optional(value: Any) -> str:
    return "not estimable (no takeover)" if value is None else f"{float(value):.6f}"


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    calibration = _mapping(summary["calibration"], "calibration")
    fresh = _mapping(summary["fresh"], "fresh")
    certificates = _mapping(
        summary["certificates_and_decomposition"],
        "certificates_and_decomposition",
    )
    text = f"""# RCCF v4 bounded public diagnostic

**FAILED / NO AUTHORITY / DEVELOPMENT ONLY / NEGATIVE RESULT**

This public slice was emitted only after strict clean-source validation of the full result bundle. It is a small synthetic interface diagnostic, not research evidence for real BOST, generalization, or method superiority.

## Decision

- Calibration risk upper bound: {calibration['risk_upper_bound']:.6f}; development gate: <= {calibration['maximum_risk_upper_gate']:.6f}. The development gate failed.
- Diagnostic calibration coverage: {calibration['diagnostic_takeover_coverage']:.6f}; authorized coverage after the failed gate: {calibration['authorized_takeover_coverage']:.6f}.
- Fresh policy: {fresh['takeover_count']} takeover and {fresh['fallback_count']} fallback among {fresh['count']} rigs.
- Fresh conditional harm: {_format_optional(fresh['selection_conditional_harm_rate'])}.
- Deterministic audits: {certificates['partition_audit_count']} partition checks, {certificates['partition_audit_violation_count']} certificate violations, and {certificates['operator_decomposition_mismatch_count']} operator-decomposition mismatches.

Passing deterministic certificate/decomposition checks does not override the failed statistical development gate. Every research, real-BOST, generalization, and superiority authorization remains false.

## Public files

- `summary.json`: bounded aggregate decision data and explicit false claim flags.
- `diagnostic.png`: calibration risk, diagnostic versus authorized coverage, fresh fallback/takeover, and certificate/decomposition counts.
- `calibration_inspection.csv`: reduced risk-calibration rows only.
- `fresh_inspection.csv`: reduced fresh-policy rows only.

No geometry seeds, source paths, provenance internals, solver trajectories, cost proxies, or timing measurements are copied. No timing comparison or timing advantage claim is made.
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def _bar_labels(ax: Any, bars: Any, *, percent: bool = False) -> None:
    labels = [f"{bar.get_height():.1%}" if percent else f"{bar.get_height():g}" for bar in bars]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)


def _plot(path: Path, summary: Mapping[str, Any]) -> None:
    calibration = _mapping(summary["calibration"], "calibration")
    fresh = _mapping(summary["fresh"], "fresh")
    certificates = _mapping(
        summary["certificates_and_decomposition"],
        "certificates_and_decomposition",
    )
    red, charcoal, teal, amber = "#b23a32", "#4d5963", "#17807e", "#d18b26"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        "RCCF v4 | FAILED | DEVELOPMENT ONLY | NO AUTHORITY",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    risk_upper = float(calibration["risk_upper_bound"])
    risk_gate = float(calibration["maximum_risk_upper_gate"])
    bars = ax.bar(["calibration\nrisk upper"], [risk_upper], color=red, width=0.55)
    ax.axhline(risk_gate, color=charcoal, linestyle="--", linewidth=2, label=f"gate <= {risk_gate:.1%}")
    ax.set_ylim(0.0, max(1.0, risk_upper * 1.16))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Calibration risk upper vs gate: FAIL")
    ax.set_ylabel("selection-conditional risk")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    _bar_labels(ax, bars, percent=True)

    ax = axes[0, 1]
    coverage_values = [
        float(calibration["diagnostic_takeover_coverage"]),
        float(calibration["diagnostic_takeover_coverage_lower_bound"]),
        float(calibration["authorized_takeover_coverage"]),
        float(calibration["authorized_takeover_coverage_lower_bound"]),
    ]
    bars = ax.bar(
        ["diagnostic\nobserved", "diagnostic\nlower", "authorized\nobserved", "authorized\nlower"],
        coverage_values,
        color=[teal, "#73b7b4", red, "#d98c86"],
    )
    coverage_gate = float(calibration["minimum_takeover_coverage_gate"])
    ax.axhline(coverage_gate, color=charcoal, linestyle="--", linewidth=2, label=f"minimum gate {coverage_gate:.1%}")
    ax.set_ylim(0.0, max(1.0, max(coverage_values, default=0.0) * 1.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Diagnostic coverage is not authorized coverage")
    ax.set_ylabel("calibration coverage")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    _bar_labels(ax, bars, percent=True)

    ax = axes[1, 0]
    bars = ax.bar(
        ["fallback", "takeover"],
        [int(fresh["fallback_count"]), int(fresh["takeover_count"])],
        color=[amber, teal],
    )
    ax.set_ylim(0.0, max(1.0, float(fresh["count"]) * 1.18))
    ax.set_title("Fresh policy action counts")
    ax.set_ylabel("fresh rigs")
    ax.grid(axis="y", alpha=0.2)
    _bar_labels(ax, bars)

    ax = axes[1, 1]
    count_values = [
        int(certificates["partition_audit_count"]),
        int(certificates["partition_audit_violation_count"]),
        int(certificates["operator_decomposition_mismatch_count"]),
    ]
    bars = ax.bar(
        ["partition\naudits", "certificate\nviolations", "decomposition\nmismatches"],
        count_values,
        color=[charcoal, red, red],
    )
    ax.set_ylim(0.0, max(1.0, max(count_values) * 1.18))
    ax.set_title("Deterministic certificate/decomposition counts")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.2)
    _bar_labels(ax, bars)

    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def run(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    report = validate_result_bundle(input_dir, require_clean_source=True)
    if not isinstance(report, Mapping):
        raise PublicExportError("validator did not return a report object")
    _assert_negative_no_authority(report)

    config = _load_json(input_dir / "config_snapshot.json")
    risk_rows = _read_risk_rows(input_dir)
    calibration_rows = _reduced_rows(
        risk_rows, "risk_calibration", CALIBRATION_FIELDS
    )
    fresh_rows = _reduced_rows(risk_rows, "fresh_geometry_ood", FRESH_FIELDS)
    summary = _build_summary(report, config)
    if len(calibration_rows) != summary["calibration"]["count"]:
        raise PublicExportError("calibration CSV row count differs from validated report")
    if len(fresh_rows) != summary["fresh"]["count"]:
        raise PublicExportError("fresh CSV row count differs from validated report")

    _prepare_output(input_dir, output_dir)
    _write_csv(
        output_dir / "calibration_inspection.csv",
        calibration_rows,
        CALIBRATION_FIELDS,
    )
    _write_csv(output_dir / "fresh_inspection.csv", fresh_rows, FRESH_FIELDS)
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_readme(output_dir / "README.md", summary)
    _plot(output_dir / "diagnostic.png", summary)
    if {path.name for path in output_dir.iterdir()} != PUBLIC_FILES:
        raise PublicExportError("public output file set drift")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a bounded public visualizer from a full RCCF v4 result directory."
    )
    parser.add_argument("input_dir", type=Path, help="full validated RCCF result directory")
    parser.add_argument("output_dir", type=Path, help="bounded public output directory")
    args = parser.parse_args()
    summary = run(args.input_dir, args.output_dir)
    print(json.dumps({"status": summary["status"], "files": sorted(PUBLIC_FILES)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
