#!/usr/bin/env python3
"""Publish a checksum-verified descriptive view of the completed PSU-B0 D0 release."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = "psu-b0-exact-absolute-root-cause-report-1.0"
PUBLIC_SCHEMA = "psu-b0-exact-absolute-root-cause-public-1.0"
CHECKPOINTS = (4, 8, 16, 32, 64, 128)
REPLICATES = (0, 8)
FAMILIES = (
    "plume", "wavy_front", "thin_front", "double_front", "annular_kernel",
    "oblique_shock", "vortex_pair", "multi_plume",
)
PDHG_METHODS = (
    "scalar_a_only_pdhg", "formal_factor_view_a_only_pdhg",
    "factor_row_hybrid_a_only_pdhg", "exact_abs_view_a_only_pdhg",
    "exact_abs_row_a_only_pdhg",
)
VARIANTS = (
    ("factor_row_hybrid", "factor_row_hybrid_a_only_pdhg", "Factor row"),
    ("exact_abs_view", "exact_abs_view_a_only_pdhg", "Exact |A| view"),
    ("exact_abs_row", "exact_abs_row_a_only_pdhg", "Exact |A| row"),
)
SOURCE_FILES = {
    "report.json", "trajectory_rows.csv", "tightness_rows.csv", "audit_rows.json",
}
PUBLIC_GENERATED_FILES = {
    "README.md", "decision_gates.csv", "diagnostic.pdf", "diagnostic.png",
    "method_frontier.csv", "paired_k128_gains.csv", "summary.json",
    "tightness_summary.csv",
}
PUBLIC_FILES = PUBLIC_GENERATED_FILES | {"checksums.sha256"}
EXPECTED_CLAIM_BOUNDARY = {
    "algorithm_superiority_claimed": False,
    "allowed_claim": "opened synthetic numerical root-cause diagnostic only",
    "experimental_flow_truth_used": False,
    "formal_gate_b_reopened": False,
    "generalization_claimed": False,
    "graph_comparison_binding": False,
    "metric_only_uses_absolute_operator": True,
    "neural_operator_training_authorized": False,
    "new_algorithm_claimed": False,
    "real_flowoff_repeats_used": False,
    "solver_recurrence_operator": "SIGNED_A",
    "support_null_prior_difference_disclosed": True,
}
TRAJECTORY_FIELDS = {
    "replicate", "sample_index", "reaction_family", "method", "iterations",
    "forward_calls", "adjoint_calls", "field_relative_l2", "gradient_relative_l2",
    "front_top10_f1", "data_coupled_relative_l2", "data_null_support_relative_l2",
    "data_coupled_error_energy", "data_null_support_error_energy",
    "data_null_support_reconstruction_energy", "normalized_data_residual_l2",
    "trajectory_elapsed_seconds",
}
TIGHTNESS_FIELDS = {
    "replicate", "sample_index", "reaction_family", "row_ratio_minimum",
    "row_ratio_p05", "row_ratio_median", "row_ratio_mean", "column_ratio_minimum",
    "column_ratio_p05", "column_ratio_median", "column_ratio_mean",
    "global_exact_to_factor_mass_ratio", "global_slack_mass", "exact_zero_row_count",
    "exact_zero_column_count", "factor_only_nonzero_count", "exact_only_nonzero_count",
    "factor_majorizer_active_coordinate_count", "signed_A_nonzero_coordinate_count",
    "M_active_A_zero_coordinate_count", "nullspace_dimension_claimed",
    "dominance_violation_maximum", "dominance_relative_violation_maximum",
    "setup_factor_row_relative_error", "setup_factor_column_relative_error",
    "audit_content_sha256", "mps_repeat_content_sha256",
    "mps_repeat_required_for_this_row", "solver_mps_setup_call_ledger",
    "audit_cpu64_setup_call_ledger", "exact_streaming_replay_call_ledger",
}


class ValidationError(ValueError):
    """Raised when the frozen input release is not exactly the expected release."""


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
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                      object_pairs_hook=_unique_object)


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


def _validate_source_manifest(root: Path) -> dict[str, str]:
    path = root / "checksums.sha256"
    if path.is_symlink() or not path.is_file():
        raise ValidationError("missing source checksum manifest")
    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        token = PurePosixPath(name)
        if (not separator or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or token.is_absolute() or len(token.parts) != 1 or name in manifest):
            raise ValidationError("malformed source checksum manifest")
        manifest[name] = digest
    if set(manifest) != SOURCE_FILES:
        raise ValidationError("source checksum file set drift")
    for name, digest in manifest.items():
        if _sha256(root / name) != digest:
            raise ValidationError(f"source checksum mismatch: {name}")
    return manifest


def _validate_trajectory(rows: list[dict[str, str]]) -> None:
    expected = {(replicate, index, family, method, checkpoint)
                for replicate in REPLICATES for index, family in enumerate(FAMILIES)
                for method in (*PDHG_METHODS, "graph_pcgls") for checkpoint in CHECKPOINTS}
    seen = set()
    for row in rows:
        key = (_integer(row["replicate"], "replicate"), _integer(row["sample_index"], "sample_index"),
               row["reaction_family"], row["method"], _integer(row["iterations"], "iterations"))
        if key in seen or key not in expected:
            raise ValidationError(f"unexpected trajectory identity: {key}")
        seen.add(key)
        if _integer(row["forward_calls"], "forward_calls") != key[-1] or _integer(row["adjoint_calls"], "adjoint_calls") != key[-1]:
            raise ValidationError("trajectory calls are not exact-K")
        for name in TRAJECTORY_FIELDS - {"replicate", "sample_index", "reaction_family", "method", "iterations", "forward_calls", "adjoint_calls", "normalized_data_residual_l2"}:
            if _number(row[name], name) < 0.0:
                raise ValidationError(f"negative trajectory value: {name}")
        residual = row["normalized_data_residual_l2"]
        if key[3] == "graph_pcgls":
            if residual not in {"", None}:
                raise ValidationError("graph residual must stay outside PDHG arithmetic")
        elif _number(residual, "normalized_data_residual_l2") < 0.0:
            raise ValidationError("negative residual")
    if seen != expected:
        raise ValidationError("trajectory coverage drift")


def _validate_tightness(rows: list[dict[str, str]]) -> None:
    expected = {(replicate, index, family) for replicate in REPLICATES for index, family in enumerate(FAMILIES)}
    seen = set()
    for row in rows:
        key = (_integer(row["replicate"], "replicate"), _integer(row["sample_index"], "sample_index"), row["reaction_family"])
        if key in seen or key not in expected:
            raise ValidationError(f"unexpected tightness identity: {key}")
        seen.add(key)
        for name in ("row_ratio_minimum", "row_ratio_p05", "row_ratio_median", "row_ratio_mean", "column_ratio_minimum", "column_ratio_p05", "column_ratio_median", "column_ratio_mean", "global_exact_to_factor_mass_ratio", "global_slack_mass", "dominance_violation_maximum", "dominance_relative_violation_maximum"):
            value = _number(row[name], name)
            if value < 0.0 or ("ratio" in name and value > 1.0 + 1e-12):
                raise ValidationError(f"invalid tightness value: {name}")
        if _integer(row["exact_only_nonzero_count"], "exact_only_nonzero_count") != 0:
            raise ValidationError("|A| support escaped factor majorizer")
    if seen != expected:
        raise ValidationError("tightness coverage drift")


def load_release(root: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    manifest = _validate_source_manifest(root)
    report = _load_json(root / "report.json")
    audit = _load_json(root / "audit_rows.json")
    if not isinstance(report, dict) or report.get("schema_version") != INPUT_SCHEMA:
        raise ValidationError("report schema drift")
    if report.get("status") != "FACTOR_MAJORIZER_CANCELLATION_MATERIAL_DESCRIPTIVE":
        raise ValidationError("release status drift")
    boundary = report.get("claim_boundary")
    if boundary != EXPECTED_CLAIM_BOUNDARY:
        raise ValidationError("claim boundary drift")
    decision = report.get("decision")
    if (
        not isinstance(decision, dict)
        or decision.get("formal_gate_b_reopened") is not False
        or decision.get("graph_comparison_binding") is not False
        or decision.get("graph_support_contract_matches_pdhg") is not False
        or decision.get("causal_krylov_explanation_claimed") is not False
        or decision.get("claim")
        != "OPENED_SYNTHETIC_SAME_SIGNED_A_DIAGONAL_DIAGNOSTIC_ONLY_NO_NEW_ALGORITHM_NO_EXPERIMENTAL_OR_GENERALIZATION_CLAIM"
    ):
        raise ValidationError("decision claim boundary drift")
    audit_shapes = {
        frozenset({"eta_squared", "mode", "normalized_norm_squared_power_estimate", "power_call_ledger", "power_estimate_device", "power_estimate_dtype", "power_value_is_upper_bound", "replicate", "sample_index", "schur_certificate_is_theorem_backed", "schur_certificate_squared_upper_bound", "solver_metric_device", "solver_metric_dtype"}): 48,
        frozenset({"elapsed_seconds", "method", "replicate", "sample_index", "scorer_ledger", "solver_ledger"}): 80,
        frozenset({"elapsed_seconds", "ledger", "method", "replicate", "sample_index"}): 16,
    }
    observed_shapes: dict[frozenset[str], int] = defaultdict(int)
    if not isinstance(audit, list):
        raise ValidationError("audit schema or coverage drift")
    for row in audit:
        if not isinstance(row, dict):
            raise ValidationError("audit row is not an object")
        observed_shapes[frozenset(row)] += 1
    if dict(observed_shapes) != audit_shapes:
        raise ValidationError("audit schema or coverage drift")
    trajectory = _load_csv(root / "trajectory_rows.csv", TRAJECTORY_FIELDS)
    tightness = _load_csv(root / "tightness_rows.csv", TIGHTNESS_FIELDS)
    _validate_trajectory(trajectory)
    _validate_tightness(tightness)
    if report.get("metric_row_count") != len(trajectory) or report.get("tightness_row_count") != len(tightness):
        raise ValidationError("report row count drift")
    return report, trajectory, tightness, manifest


def _mean(rows: Iterable[dict[str, str]], name: str) -> float:
    return float(np.mean([_number(row[name], name) for row in rows]))


def derive_public_tables(
    trajectory: list[dict[str, str]],
    tightness: list[dict[str, str]],
    formal_decision: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indexed = {(int(row["replicate"]), int(row["sample_index"]), row["method"], int(row["iterations"])): row for row in trajectory}
    frontier = []
    for method in (*PDHG_METHODS, "graph_pcgls"):
        for checkpoint in CHECKPOINTS:
            group = [indexed[(r, s, method, checkpoint)] for r in REPLICATES for s in range(len(FAMILIES))]
            binding = method != "graph_pcgls"
            frontier.append({"method": method, "iterations": checkpoint, "opened_row_count": len(group),
                             "comparison_binding": binding,
                             "comparison_role": ("same-signed-A reduced-support diagnostic" if binding else "nonbinding headroom only"),
                             "support_contract": ("2322-coordinate data-coupled reduced support" if binding else "full support"),
                             "prior_contract": ("data-only zero-gauge recurrence" if binding else "Sobolev graph prior"),
                             "mean_normalized_data_residual_l2": (None if method == "graph_pcgls" else _mean(group, "normalized_data_residual_l2")),
                             "mean_field_relative_l2": _mean(group, "field_relative_l2"),
                             "mean_gradient_relative_l2": _mean(group, "gradient_relative_l2"),
                             "mean_front_top10_f1": _mean(group, "front_top10_f1")})
    gains = []
    for replicate in REPLICATES:
        for sample_index, family in enumerate(FAMILIES):
            baseline = indexed[(replicate, sample_index, "formal_factor_view_a_only_pdhg", 128)]
            for key, method, label in VARIANTS:
                candidate = indexed[(replicate, sample_index, method, 128)]
                residual_base, residual_value = _number(baseline["normalized_data_residual_l2"], "baseline residual"), _number(candidate["normalized_data_residual_l2"], "candidate residual")
                field_base, field_value = _number(baseline["field_relative_l2"], "baseline field"), _number(candidate["field_relative_l2"], "candidate field")
                gains.append({"replicate": replicate, "sample_index": sample_index, "reaction_family": family, "variant": key, "variant_label": label,
                              "baseline_normalized_data_residual_l2": residual_base, "variant_normalized_data_residual_l2": residual_value,
                              "residual_gain_percent": 100.0 * (residual_base - residual_value) / residual_base,
                              "baseline_field_relative_l2": field_base, "variant_field_relative_l2": field_value,
                              "field_gain_percent": 100.0 * (field_base - field_value) / field_base})
    tightness_summary = []
    for row in tightness:
        tightness_summary.append({name: row[name] for name in ("replicate", "sample_index", "reaction_family", "row_ratio_p05", "row_ratio_median", "column_ratio_p05", "column_ratio_median", "global_exact_to_factor_mass_ratio", "global_slack_mass")})
    exact = [row for row in gains if row["variant"] == "exact_abs_row"]
    exact_frontier = [row for row in frontier if row["method"] == "exact_abs_row_a_only_pdhg"]
    baseline_frontier = next(row for row in frontier if row["method"] == "formal_factor_view_a_only_pdhg" and row["iterations"] == 128)
    exact_endpoint = next(row for row in exact_frontier if row["iterations"] == 128)
    ratio_of_means_residual_gain = 100.0 * (
        baseline_frontier["mean_normalized_data_residual_l2"]
        - exact_endpoint["mean_normalized_data_residual_l2"]
    ) / baseline_frontier["mean_normalized_data_residual_l2"]
    ratio_of_means_field_gain = 100.0 * (
        baseline_frontier["mean_field_relative_l2"]
        - exact_endpoint["mean_field_relative_l2"]
    ) / baseline_frontier["mean_field_relative_l2"]
    k64 = next(row for row in exact_frontier if row["iterations"] == 64)
    k128_rises_count = sum(
        _number(indexed[(replicate, sample, "exact_abs_row_a_only_pdhg", 128)]["field_relative_l2"], "K128 field")
        > _number(indexed[(replicate, sample, "exact_abs_row_a_only_pdhg", 64)]["field_relative_l2"], "K64 field")
        for replicate in REPLICATES
        for sample in range(len(FAMILIES))
    )
    median_row_p05_slack = float(np.median([1.0 - float(row["row_ratio_p05"]) for row in tightness_summary]))
    if formal_decision is not None:
        formal_gain = formal_decision.get("mean_normalized_residual_gain_percent")
        if (
            not isinstance(formal_gain, dict)
            or not math.isclose(
                _number(formal_gain.get("exact_abs_row"), "formal exact-row residual gain"),
                ratio_of_means_residual_gain,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _number(formal_decision.get("median_high_quantile_factor_slack"), "formal tail slack"),
                median_row_p05_slack,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValidationError("formal decision arithmetic drift")
    decision = {"exact_abs_row_paired_mean_residual_gain_percent": float(np.mean([row["residual_gain_percent"] for row in exact])),
                "exact_abs_row_paired_mean_field_gain_percent": float(np.mean([row["field_gain_percent"] for row in exact])),
                "exact_abs_row_ratio_of_means_residual_gain_percent": ratio_of_means_residual_gain,
                "exact_abs_row_ratio_of_means_field_gain_percent": ratio_of_means_field_gain,
                "formal_factor_k128_mean_residual": baseline_frontier["mean_normalized_data_residual_l2"],
                "exact_abs_row_k128_mean_residual": exact_endpoint["mean_normalized_data_residual_l2"],
                "formal_factor_k128_mean_field_error": baseline_frontier["mean_field_relative_l2"],
                "exact_abs_row_k128_mean_field_error": exact_endpoint["mean_field_relative_l2"],
                "formal_factor_k128_mean_gradient_error": baseline_frontier["mean_gradient_relative_l2"],
                "exact_abs_row_k128_mean_gradient_error": exact_endpoint["mean_gradient_relative_l2"],
                "formal_factor_k128_mean_front_f1": baseline_frontier["mean_front_top10_f1"],
                "exact_abs_row_k128_mean_front_f1": exact_endpoint["mean_front_top10_f1"],
                "exact_abs_row_descriptive_mean_minimum_evaluated_checkpoint": min(exact_frontier, key=lambda row: row["mean_field_relative_l2"])["iterations"],
                "exact_abs_row_mean_k128_gt_k64_descriptive": exact_frontier[-1]["mean_field_relative_l2"] > k64["mean_field_relative_l2"],
                "exact_abs_row_k128_gt_k64_opened_row_count": k128_rises_count,
                "median_row_p05_slack": median_row_p05_slack,
                "opened_field_count": 16,
                "replicate_cluster_count": 2,
                "iid_field_count_claimed": False}
    return frontier, gains, tightness_summary, decision


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_figure(
    frontier: list[dict[str, Any]],
    gains: list[dict[str, Any]],
    tightness: list[dict[str, Any]],
    decision: Mapping[str, Any],
    output: Path,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    colors = {"scalar_a_only_pdhg": "#7f8c8d", "formal_factor_view_a_only_pdhg": "#3d6d9c", "factor_row_hybrid_a_only_pdhg": "#8c6d31", "exact_abs_view_a_only_pdhg": "#14756a", "exact_abs_row_a_only_pdhg": "#c24b3b", "graph_pcgls": "#6f5b95"}
    labels = {key: label for key, _, label in VARIANTS} | {"scalar_a_only_pdhg": "Scalar", "formal_factor_view_a_only_pdhg": "Formal factor view", "factor_row_hybrid_a_only_pdhg": "Factor row", "exact_abs_view_a_only_pdhg": "Exact |A| view", "exact_abs_row_a_only_pdhg": "Exact |A| row", "graph_pcgls": "Graph-PCGLS"}
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)
    for method in PDHG_METHODS:
        rows = [r for r in frontier if r["method"] == method]
        axes[0, 0].plot([r["iterations"] for r in rows], [r["mean_normalized_data_residual_l2"] for r in rows], marker="o", linewidth=2, color=colors[method], label=labels[method])
        axes[0, 1].plot([r["iterations"] for r in rows], [r["mean_field_relative_l2"] for r in rows], marker="o", linewidth=2, color=colors[method], label=labels[method])
    graph_rows = [r for r in frontier if r["method"] == "graph_pcgls"]
    # Graph is deliberately rendered only as a nonbinding comparator.
    axes[0, 1].plot([r["iterations"] for r in graph_rows], [r["mean_field_relative_l2"] for r in graph_rows], marker="s", linestyle="--", linewidth=1.7, color=colors["graph_pcgls"], label="Graph-PCGLS (nonbinding comparator)")
    for axis, title, ylabel in ((axes[0, 0], "A  Mean normalized data residual", "mean normalized residual"), (axes[0, 1], "B  Mean field error", "mean field relative L2")):
        axis.set_xscale("log", base=2); axis.set_xticks(CHECKPOINTS); axis.set_xticklabels(CHECKPOINTS); axis.set_xlabel("K (exact forward/adjoint calls)"); axis.set_ylabel(ylabel); axis.set_title(title, loc="left"); axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper right")
    axes[0, 1].text(0.02, 0.04, "Graph comparator is nonbinding; support contract differs.", transform=axes[0, 1].transAxes, fontsize=7.5, color="#4e3e68")
    axes[0, 1].text(0.02, 0.14, "Among 6 evaluated K, descriptive mean is lowest at K64; no general stop rule.", transform=axes[0, 1].transAxes, fontsize=7.2, color="#8c3026")
    variant_keys = [key for key, _, _ in VARIANTS]
    values = [[row["residual_gain_percent"] for row in gains if row["variant"] == key] for key in variant_keys]
    boxes = axes[1, 0].boxplot(values, tick_labels=[labels[key] for key in variant_keys], patch_artist=True, medianprops={"color": "#222222"})
    for patch, (_, method, _) in zip(boxes["boxes"], VARIANTS, strict=True): patch.set_facecolor(colors[method]); patch.set_alpha(0.7)
    axes[1, 0].axhline(0, color="#666666", linewidth=0.8); axes[1, 0].set_ylabel("K128 residual gain vs formal factor view (%)"); axes[1, 0].set_title("C  2 replicate clusters x 8 shared morphologies (not IID)", loc="left"); axes[1, 0].grid(axis="y", alpha=0.22)
    axes[1, 0].text(0.98, 0.95, f"Exact |A| row paired mean: {decision['exact_abs_row_paired_mean_residual_gain_percent']:.1f}%\nRatio of means: {decision['exact_abs_row_ratio_of_means_residual_gain_percent']:.1f}%", transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=8, color="#8c3026")
    ratios = [[float(row[name]) for row in tightness] for name in ("row_ratio_p05", "column_ratio_p05", "global_exact_to_factor_mass_ratio")]
    ratios.append([float(row["global_slack_mass"]) for row in tightness])
    ratios.append([1.0 - float(row["row_ratio_p05"]) for row in tightness])
    box = axes[1, 1].boxplot(ratios, tick_labels=["row p05\ntightness", "column p05\ntightness", "global mass\ntightness", "global mass\nslack", "row p05\ntail slack"], patch_artist=True, medianprops={"color": "#222222"})
    for patch, color in zip(box["boxes"], ("#e0ab55", "#6b9ac4", "#55a092", "#c24b3b", "#a53f74"), strict=True): patch.set_facecolor(color); patch.set_alpha(0.75)
    rng = np.random.default_rng(0)
    for index, values in enumerate(ratios, start=1):
        axes[1, 1].scatter(index + rng.normal(0.0, 0.035, len(values)), values, s=12, color="#293b42", alpha=0.55, zorder=3)
    axes[1, 1].set_ylim(0, 1.04); axes[1, 1].set_ylabel("unitless tightness or complementary slack"); axes[1, 1].set_title("D  Exact-to-factor tightness and explicitly labeled slack", loc="left"); axes[1, 1].grid(axis="y", alpha=0.22)
    axes[1, 1].text(0.02, 0.04, "Slack = 1 - matching tightness; tail slack is not mean factor error.", transform=axes[1, 1].transAxes, fontsize=7.5, color="#3f4c52")
    figure.suptitle("PSU B0 D0 exact-|A| diagnostic: post-NO-GO opened synthetic evidence only\nGate B remains closed; no new algorithm, real-data, or generalization claim", fontsize=13, fontweight="bold")
    for suffix, options in ((".png", {"dpi": 220}), (".pdf", {})):
        figure.savefig(output.with_suffix(suffix), bbox_inches="tight", facecolor="white", **options)
    plt.close(figure)


def run(input_root: Path, output_root: Path) -> dict[str, Any]:
    report, trajectory, tightness, manifest = load_release(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output_root.iterdir()} - PUBLIC_FILES
    if unexpected:
        raise ValidationError(f"unexpected stale public files: {sorted(unexpected)}")
    frontier, gains, tightness_summary, decision = derive_public_tables(
        trajectory,
        tightness,
        report["decision"],
    )
    _write_csv(output_root / "method_frontier.csv", frontier)
    _write_csv(output_root / "paired_k128_gains.csv", gains)
    _write_csv(output_root / "tightness_summary.csv", tightness_summary)
    gates = [{"gate": "input_checksums_verified", "passed": True, "detail": "four frozen release files match checksums.sha256"}, {"gate": "input_schema_and_coverage", "passed": True, "detail": "576 trajectory rows and 16 paired tightness rows"}, {"gate": "formal_gate_b_remains_closed", "passed": True, "detail": "post-NO-GO diagnostic only; Gate B was not reopened"}, {"gate": "graph_comparison_binding", "passed": False, "detail": "Graph-PCGLS remains nonbinding because support and prior contracts differ"}, {"gate": "new_algorithm_claimed", "passed": False, "detail": "not claimed"}, {"gate": "experimental_flow_truth_used", "passed": False, "detail": "not used"}, {"gate": "generalization_claimed", "passed": False, "detail": "not claimed"}]
    _write_csv(output_root / "decision_gates.csv", gates)
    boundary = report["claim_boundary"]
    summary = {"schema_version": PUBLIC_SCHEMA, "status": "PUBLIC_DESCRIPTIVE_D0_DIAGNOSTIC_COMPLETE", "evidence_scope": "POST_NO_GO_OPENED_SYNTHETIC_SAME_SIGNED_A_DIAGONAL_DIAGNOSTIC_ONLY", "source_release": {"schema_version": report["schema_version"], "status": report["status"], "verified_file_count": len(manifest)}, "headline": {"ratio_of_means_exact_abs_row_residual_gain_percent": decision["exact_abs_row_ratio_of_means_residual_gain_percent"], "paired_mean_exact_abs_row_residual_gain_percent": decision["exact_abs_row_paired_mean_residual_gain_percent"], "ratio_of_means_exact_abs_row_field_gain_percent": decision["exact_abs_row_ratio_of_means_field_gain_percent"], "paired_mean_exact_abs_row_field_gain_percent": decision["exact_abs_row_paired_mean_field_gain_percent"], "descriptive_mean_minimum_evaluated_checkpoint": decision["exact_abs_row_descriptive_mean_minimum_evaluated_checkpoint"], "mean_k128_gt_k64_descriptive": decision["exact_abs_row_mean_k128_gt_k64_descriptive"], "k128_gt_k64_opened_row_count": decision["exact_abs_row_k128_gt_k64_opened_row_count"], "general_early_stopping_rule_validated": False, "gradient_error_worsens_at_k128_vs_formal_factor": decision["exact_abs_row_k128_mean_gradient_error"] > decision["formal_factor_k128_mean_gradient_error"]}, "statistical_contract": {"replicate_clusters": 2, "paired_morphologies_per_cluster": 8, "opened_rows": 16, "iid_rows_claimed": False}, "known_confounders": {"synthetic_view_scaling_uses_clean_truth": True, "multiplicity_adjusted_hypothesis_test_claimed": False}, "claim_boundary": {"formal_gate_b_reopened": boundary["formal_gate_b_reopened"], "post_no_go_diagnostic_only": True, "new_algorithm_claimed": boundary["new_algorithm_claimed"], "algorithm_superiority_claimed": boundary["algorithm_superiority_claimed"], "experimental_flow_truth_used": boundary["experimental_flow_truth_used"], "generalization_claimed": boundary["generalization_claimed"], "graph_comparison_binding": boundary["graph_comparison_binding"], "solver_recurrence_operator": boundary["solver_recurrence_operator"], "real_flowoff_repeats_used": boundary["real_flowoff_repeats_used"], "causal_cancellation_mechanism_proved": False}, "public_export_policy": {"contains_local_paths": False, "contains_raw_measurements": False, "contains_reconstructions": False, "contains_private_source_hashes": False}}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_figure(frontier, gains, tightness_summary, decision, output_root / "diagnostic")
    readme = "# PSU B0 D0 exact-|A| public diagnostic\n\nThis is a checksum-verified post-NO-GO descriptive export from the completed opened synthetic D0 release. **Gate B remains closed.** At K128, exact-|A| row diagonalization lowers the mean normalized residual from {:.4f} to {:.4f}: a {:.2f}% ratio-of-means reduction. The mean field error changes from {:.4f} to {:.4f}, only a {:.2f}% ratio-of-means gain. The paired-percent means are {:.2f}% and {:.2f}%, respectively; they are reported separately because they are different estimands. Among the six evaluated checkpoints, descriptive mean field error is smallest at K{}; K128 is higher in {}/16 opened rows, so no general early-stopping rule is claimed. Gradient error is worse than formal factor at K128.\n\nThe 16 rows are two replicate clusters containing the same eight morphology families, not 16 IID experiments. Synthetic view scaling uses clean-truth projections, so the complete pipeline is not truth-blind. Tail slack is an operator-level p05 diagnostic, not average factor error. Graph-PCGLS is only a nonbinding headroom comparator because its support and prior contracts differ.\n\nNo unique causal mechanism, new algorithm, experimental-flow truth, or generalization claim is made. The solver recurrence remains `{}`. Run `PYTHONPATH=. .venv/bin/python site_tools/analyze_psu_b0_exact_absolute_diagnostic.py` from the repository root to regenerate. `checksums.sha256` covers exactly the declared public file set except itself.\n".format(decision["formal_factor_k128_mean_residual"], decision["exact_abs_row_k128_mean_residual"], decision["exact_abs_row_ratio_of_means_residual_gain_percent"], decision["formal_factor_k128_mean_field_error"], decision["exact_abs_row_k128_mean_field_error"], decision["exact_abs_row_ratio_of_means_field_gain_percent"], decision["exact_abs_row_paired_mean_residual_gain_percent"], decision["exact_abs_row_paired_mean_field_gain_percent"], decision["exact_abs_row_descriptive_mean_minimum_evaluated_checkpoint"], decision["exact_abs_row_k128_gt_k64_opened_row_count"], boundary["solver_recurrence_operator"])
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    generated = sorted(output_root / name for name in PUBLIC_GENERATED_FILES)
    if any(path.is_symlink() or not path.is_file() for path in generated):
        raise ValidationError("public generated file set is incomplete or unsafe")
    (output_root / "checksums.sha256").write_text("".join(f"{_sha256(path)}  {path.name}\n" for path in generated), encoding="ascii")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=REPOSITORY_ROOT / "demo_t16_operator/results/psu_b0_exact_absolute_root_cause")
    parser.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "demo_t16_operator/results/psu_b0_exact_absolute_root_cause_public")
    args = parser.parse_args()
    print(json.dumps(run(args.input_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
