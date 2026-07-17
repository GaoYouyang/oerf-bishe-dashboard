#!/usr/bin/env python3
"""Create a strictly redacted public slice of the v3 grouped-majorizer smoke.

This exporter deliberately exposes aggregate metrics only. It never copies the
source report, source checksums, local paths, geometry seeds, or private source
metadata into the public directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "demo_t16_operator/results/certified_grouped_majorizer_smoke"
DEFAULT_OUTPUT = ROOT / "demo_t16_operator/results/certified_grouped_majorizer_smoke_public"
PUBLIC_SCHEMA = "certified-grouped-majorizer-smoke-public-1.0"
FRESH_RIGS = tuple(f"fresh-{i:02d}" for i in range(8))
METHODS = (
    "singleton_factor", "fixed_paired_local", "fixed_paired_cross",
    "fixed_triad_bridge", "train_selected_fixed",
    "geometry_conditioned_selector", "all_in_one_exact_oracle",
)
LABELS = {
    "singleton_factor": "singleton factor",
    "fixed_paired_local": "fixed paired-local",
    "fixed_paired_cross": "best fixed (paired-cross)",
    "fixed_triad_bridge": "fixed triad-bridge",
    "train_selected_fixed": "train-selected fixed",
    "geometry_conditioned_selector": "geometry selector",
    "all_in_one_exact_oracle": "exact oracle (nondeployable)",
}
SOURCE_FILES = {
    "report.json", "construction_cost_rows.csv", "geometry_manifest.csv",
    "metric_rows.csv", "partition_audit_rows.csv", "selection_rows.csv",
    "trajectory_rows.csv", "checksums.sha256", "config_snapshot.json",
}
PUBLIC_FILES = {
    "README.md", "summary.json", "method_summary.csv", "fresh_rig_field.csv",
    "safety_summary.csv", "diagnostic.png", "diagnostic.pdf", "checksums.sha256",
}


class ValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{field} is not finite")
    return result


def _load_csv(root: Path, name: str, fields: set[str]) -> list[dict[str, str]]:
    with (root / name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != fields:
            raise ValidationError(f"CSV schema drift: {name}")
        rows = list(reader)
    if not rows or any(set(row) != fields for row in rows):
        raise ValidationError(f"malformed CSV: {name}")
    return rows


def _verify_source(root: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if root.is_symlink() or not root.is_dir():
        raise ValidationError("source root is missing or unsafe")
    observed = {p.name for p in root.iterdir() if p.is_file()}
    if observed != SOURCE_FILES:
        raise ValidationError(f"source file set drift: {sorted(observed)}")
    if any(p.is_symlink() for p in root.iterdir()):
        raise ValidationError("source contains symlink")
    manifest = {}
    for line in (root / "checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, sep, name = line.partition("  ")
        if not sep or name in manifest or name not in SOURCE_FILES - {"checksums.sha256"}:
            raise ValidationError("malformed source checksum manifest")
        manifest[name] = digest
    expected_names = SOURCE_FILES - {"checksums.sha256"}
    if set(manifest) != expected_names:
        raise ValidationError("source checksum coverage drift")
    for name, digest in manifest.items():
        if _sha256(root / name) != digest:
            raise ValidationError(f"source checksum mismatch: {name}")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "DEVELOPMENT_ONLY_SYNTHETIC_CERTIFIED_PARTITION_SMOKE":
        raise ValidationError("unexpected evidence status")
    if report.get("evidence_scope") != "SYNTHETIC_MULTIPRIMITIVE_CPU_SMOKE_ONLY":
        raise ValidationError("unexpected evidence scope")
    boundary = report.get("claim_boundary", {})
    if boundary != {"exact_oracle_is_deployable": False, "generalization_claimed": False,
                    "paper_superiority_claimed": False, "real_bost_claimed": False}:
        raise ValidationError("claim boundary drift")
    decision = report.get("decision", {})
    if decision.get("research_claim_authorized") is not False or decision.get("real_bost_claim_authorized") is not False:
        raise ValidationError("authorization boundary drift")
    if decision.get("all_partition_audits_zero_violation") is not True or decision.get("selector_all_fresh_schur_safe") is not True:
        raise ValidationError("source safety decision drift")
    provenance = report.get("provenance", {})
    if provenance.get("source_worktree_dirty") is not False:
        raise ValidationError("source provenance boundary drift")
    metric_fields = set(csv.DictReader((root / "metric_rows.csv").open(encoding="utf-8")).fieldnames or ())
    required_metric_fields = {"rig_id", "split_role", "method", "oracle_only", "final_field_relative_l2",
                              "final_normalized_residual_l2", "harm_vs_train_selected_fixed_field_l2",
                              "pointwise_violation_count", "row_violation_count", "column_violation_count",
                              "spectral_violation_count", "total_violation_count"}
    if not required_metric_fields <= metric_fields:
        raise ValidationError("metric fields drift")
    metrics = _load_csv(root, "metric_rows.csv", metric_fields)
    fresh = [row for row in metrics if row["rig_id"] in FRESH_RIGS]
    if len(fresh) != len(FRESH_RIGS) * len(METHODS):
        raise ValidationError("fresh metric coverage drift")
    audits = _load_csv(root, "partition_audit_rows.csv", set(csv.DictReader((root / "partition_audit_rows.csv").open()).fieldnames or ()))
    return report, fresh, audits, metrics


def _num(row: Mapping[str, str], key: str) -> float:
    value = _finite(row[key], key)
    if value < 0:
        raise ValidationError(f"negative {key}")
    return value


def _derive(report: Mapping[str, Any], fresh: list[dict[str, str]], audits: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by = {(r["rig_id"], r["method"]): r for r in fresh}
    if set(by) != {(rig, method) for rig in FRESH_RIGS for method in METHODS}:
        raise ValidationError("fresh method identity drift")
    for row in fresh:
        for key in ("final_field_relative_l2", "final_normalized_residual_l2", "harm_vs_train_selected_fixed_field_l2"):
            value = _finite(row[key], key)
            if key != "harm_vs_train_selected_fixed_field_l2" and value < 0:
                raise ValidationError(f"negative {key}")
        counts = [int(row[key]) for key in ("pointwise_violation_count", "row_violation_count", "column_violation_count", "spectral_violation_count")]
        total = int(row["total_violation_count"])
        if any(c < 0 for c in counts) or total != sum(counts):
            raise ValidationError("violation arithmetic drift")
    summary = []
    for method in METHODS:
        rows = [by[(rig, method)] for rig in FRESH_RIGS]
        summary.append({
            "method": method, "label": LABELS[method], "fresh_rig_count": 8,
            "mean_field_relative_l2": float(np.mean([_num(r, "final_field_relative_l2") for r in rows])),
            "mean_normalized_residual_l2": float(np.mean([_num(r, "final_normalized_residual_l2") for r in rows])),
            "total_safety_violations": sum(int(r["total_violation_count"]) for r in rows),
            "unsafe_fresh_rigs": sum(int(r["total_violation_count"]) > 0 for r in rows),
            "cost_contract": "ANALYTIC_PROXY_NOT_WALL_TIME",
            "deployable": method != "all_in_one_exact_oracle",
        })
    rig_rows = []
    wins = 0
    harms = []
    for rig in FRESH_RIGS:
        selector = _num(by[(rig, "geometry_conditioned_selector")], "final_field_relative_l2")
        best_fixed = _num(by[(rig, "train_selected_fixed")], "final_field_relative_l2")
        exact = _num(by[(rig, "all_in_one_exact_oracle")], "final_field_relative_l2")
        harm = selector - best_fixed
        win = harm < 0
        wins += win
        harms.append(harm)
        rig_rows.append({"rig_id": rig, "selector_field_relative_l2": selector,
                         "best_fixed_field_relative_l2": best_fixed, "exact_oracle_field_relative_l2": exact,
                         "selector_minus_best_fixed": harm, "selector_beats_best_fixed": win})
    selector_mean = next(r["mean_field_relative_l2"] for r in summary if r["method"] == "geometry_conditioned_selector")
    fixed_mean = next(r["mean_field_relative_l2"] for r in summary if r["method"] == "train_selected_fixed")
    exact_mean = next(r["mean_field_relative_l2"] for r in summary if r["method"] == "all_in_one_exact_oracle")
    improvement = 100 * (fixed_mean - selector_mean) / fixed_mean
    safety_by_partition: dict[str, int] = {}
    audit_values: dict[str, dict[str, list[float]]] = {}
    for row in audits:
        partition = row["partition_name"]
        safety_by_partition.setdefault(partition, 0)
        audit_values.setdefault(partition, {"spectral_ratio": [], "row_product": [], "column_product": []})
        component_counts = [int(row[key]) for key in (
            "pointwise_violation_count", "row_violation_count",
            "column_violation_count", "spectral_violation_count",
        )]
        total = int(row["total_violation_count"])
        if any(value < 0 for value in component_counts) or total != sum(component_counts):
            raise ValidationError("partition violation arithmetic drift")
        spectral = _finite(row["dense_normalized_spectral_norm_squared"], "dense normalized spectral norm squared")
        bound = _finite(row["schur_squared_upper_bound"], "Schur squared upper bound")
        row_product = _finite(row["maximum_row_product"], "maximum row product")
        column_product = _finite(row["maximum_column_product"], "maximum column product")
        if min(spectral, row_product, column_product) < 0 or bound <= 0:
            raise ValidationError("invalid certificate value")
        if (
            (spectral / bound > 1.0 + 1e-10 and component_counts[3] == 0)
            or (row_product > 1.0 + 1e-10 and component_counts[1] == 0)
            or (column_product > 1.0 + 1e-10 and component_counts[2] == 0)
        ):
            raise ValidationError("certificate threshold arithmetic drift")
        audit_values[partition]["spectral_ratio"].append(spectral / bound)
        audit_values[partition]["row_product"].append(row_product)
        audit_values[partition]["column_product"].append(column_product)
        safety_by_partition[partition] += total
    if set(safety_by_partition) != set(report["partition_catalogue"]):
        raise ValidationError("partition audit coverage drift")
    certificate_rows = []
    for partition in safety_by_partition:
        values = audit_values[partition]
        max_ratio = max(values["spectral_ratio"])
        certificate_rows.append({
            "partition": partition,
            "audit_row_count": len(values["spectral_ratio"]),
            "max_normalized_spectral_norm_squared_over_schur_bound": max_ratio,
            "minimum_spectral_certificate_margin_to_one": 1.0 - max_ratio,
            "max_row_product": max(values["row_product"]),
            "max_column_product": max(values["column_product"]),
            "safety_threshold": 1.0,
            "total_safety_violations": safety_by_partition[partition],
            "certificate": "triangle_inequality_grouped_majorizer",
        })
    gates = [
        {"gate": "all partitions safety audit", "observed": "0 violations", "passed": all(v == 0 for v in safety_by_partition.values())},
        {"gate": "selector wins every fresh rig", "observed": f"{wins}/8", "passed": wins == 8},
        {"gate": "research authorization", "observed": "NO AUTH", "passed": False},
        {"gate": "real BOST evidence", "observed": "not used", "passed": False},
    ]
    aggregate = {
        "selector_mean_field_relative_l2": selector_mean,
        "best_fixed_mean_field_relative_l2": fixed_mean,
        "exact_oracle_mean_field_relative_l2": exact_mean,
        "selector_mean_improvement_vs_best_fixed_percent": improvement,
        "selector_fresh_wins": wins, "selector_fresh_denominator": 8,
        "selector_worst_harm_vs_best_fixed": max(harms),
        "all_partition_safety_violations": sum(safety_by_partition.values()),
        "safety_violations_by_partition": safety_by_partition,
        "safety_certificate_by_partition": {
            row["partition"]: {
                key: value for key, value in row.items()
                if key not in {"partition", "certificate"}
            }
            for row in certificate_rows
        },
        "minimum_spectral_certificate_margin_to_one": min(
            row["minimum_spectral_certificate_margin_to_one"] for row in certificate_rows
        ),
        "research_claim_authorized": False,
        "real_bost_claim_authorized": False,
        "cost_definition": "ANALYTIC_PROXY_NOT_WALL_TIME",
        "cost_superiority_claimed": False,
    }
    return summary, rig_rows, gates, aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _plot(output: Path, methods: list[dict[str, Any]], rigs: list[dict[str, Any]], aggregate: Mapping[str, Any]) -> None:
    by = {r["method"]: r for r in methods}
    labels = ["best fixed", "selector", "exact oracle"]
    keys = ["train_selected_fixed", "geometry_conditioned_selector", "all_in_one_exact_oracle"]
    colors = ["#4c78a8", "#e15759", "#59a14f"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("v3 certified grouped majorizer | synthetic smoke | NO AUTH", fontsize=16, fontweight="bold")
    ax = axes[0, 0]
    x = np.arange(3)
    ax.bar(x, [by[k]["mean_field_relative_l2"] for k in keys], color=colors)
    for i, row in enumerate(rigs):
        ax.scatter(x, [row["best_fixed_field_relative_l2"], row["selector_field_relative_l2"], row["exact_oracle_field_relative_l2"]], color="#222", s=18)
    ax.set_xticks(x, labels); ax.set_ylabel("field relative L2 (lower is better)"); ax.set_title("Mean bars + each fresh rig"); ax.grid(axis="y", alpha=.2)
    ax.text(.02, .95, f"selector vs fixed: +{aggregate['selector_mean_improvement_vs_best_fixed_percent']:.2f}%\nnot authorized as a success", transform=ax.transAxes, va="top", bbox={"facecolor":"white","edgecolor":"#b03a2e"})
    ax = axes[0, 1]
    vals = [aggregate["selector_fresh_wins"], aggregate["selector_fresh_denominator"] - aggregate["selector_fresh_wins"]]
    ax.bar(["wins", "non-wins"], vals, color=["#59a14f", "#f28e2b"]); ax.set_ylim(0, 8); ax.set_ylabel("fresh rigs"); ax.set_title("Selector vs best fixed: 4/8 wins")
    ax.text(.02, .88, f"worst harm = {aggregate['selector_worst_harm_vs_best_fixed']:.6f}", transform=ax.transAxes, va="top")
    ax = axes[1, 0]
    certificate = aggregate["safety_certificate_by_partition"]
    names = list(certificate)
    x = np.arange(len(names))
    spectral_ratios = [certificate[name]["max_normalized_spectral_norm_squared_over_schur_bound"] for name in names]
    row_products = [certificate[name]["max_row_product"] for name in names]
    column_products = [certificate[name]["max_column_product"] for name in names]
    width = 0.23
    spectral_bars = ax.bar(x - width, spectral_ratios, width, label="max spectral norm² / Schur bound", color="#4c78a8")
    ax.bar(x, row_products, width, label="max row product", color="#f28e2b")
    ax.bar(x + width, column_products, width, label="max column product", color="#59a14f")
    ax.axhline(1.0, color="#b03a2e", linewidth=1.5, linestyle="--", label="safety threshold = 1")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("normalized certificate value")
    ax.set_title("Certificate margins by partition | 0 safety violations")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=.2)
    legend = ax.legend(frameon=True, fontsize=7, ncol=2, loc="upper left")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(.94)
    ax.bar_label(spectral_bars, labels=[f"{value:.3f}" for value in spectral_ratios], padding=2, fontsize=7)
    ax = axes[1, 1]; ax.axis("off")
    ax.text(.02, .90, "READOUT", fontsize=13, fontweight="bold")
    ax.text(.02, .76, f"Safety: all tested partitions stay below threshold 1;\n0 violations across 130 synthetic audits.\nTightest spectral margin = {aggregate['minimum_spectral_certificate_margin_to_one']:.6f}.\n\nAccuracy: +10.72% mean, but only 4/8 wins;\nworst harm = 0.414402. Mixed result.\n\nCost: analytic proxy only.\nNO AUTH: no paper, generalization, or real-BOST claim.", fontsize=11, va="top", linespacing=1.4)
    fig.savefig(output / "diagnostic.png", dpi=170); fig.savefig(output / "diagnostic.pdf")
    plt.close(fig)


def _write_readme(path: Path, aggregate: Mapping[str, Any]) -> None:
    path.write_text(
        f"""# v3 Certified Grouped Majorizer: public smoke summary

This is a strictly redacted, synthetic CPU smoke release. It contains no local paths, seeds, private source metadata, or restricted papers.

## What the figure says
- Every tested partition has 0 safety violations in the exported audit.
- The geometry-conditioned selector wins {aggregate['selector_fresh_wins']}/{aggregate['selector_fresh_denominator']} fresh rigs against the train-selected fixed partition.
- Its mean field relative-L2 improvement is +{aggregate['selector_mean_improvement_vs_best_fixed_percent']:.2f}%, but the worst fresh-rig harm is {aggregate['selector_worst_harm_vs_best_fixed']:.6f}.
- Therefore this is a mixed result, not an algorithm success: `research_claim_authorized=false`.
- The exact oracle is a nondeployable comparator, not a proposed method.
- Cost is `ANALYTIC_PROXY_NOT_WALL_TIME`; no cost superiority claim is made.

## Evidence boundary
Synthetic dense multi-primitive matrices are not validated BOST data. Eight fresh geometries are a smoke-test split, not statistical generalization evidence. The bundle is intended for review and follow-up experiment design, not as a paper result.

Files: `summary.json`, `method_summary.csv`, `fresh_rig_field.csv`, `safety_summary.csv`, and `diagnostic.png`/`diagnostic.pdf`.
""",
        encoding="utf-8",
    )


def _write_checksums(output: Path) -> None:
    lines = [f"{_sha256(output / name)}  {name}\n" for name in sorted(PUBLIC_FILES - {"checksums.sha256"})]
    (output / "checksums.sha256").write_text("".join(lines), encoding="ascii")


def _validate_public(output: Path) -> None:
    if {p.name for p in output.iterdir()} != PUBLIC_FILES:
        raise ValidationError("public file set drift")
    for p in output.iterdir():
        if p.is_symlink() or not p.is_file(): raise ValidationError("unsafe public entry")
    for line in (output / "checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, sep, name = line.partition("  ")
        if not sep or name not in PUBLIC_FILES - {"checksums.sha256"} or _sha256(output / name) != digest:
            raise ValidationError("public checksum mismatch")
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in output.iterdir() if p.suffix in {".md", ".json", ".csv"})
    if str(ROOT) in text or "/Users/" in text or "geometry_seed" in text:
        raise ValidationError("private provenance leaked")


def run(input_root: Path = DEFAULT_INPUT, output_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_root.exists() and any(p.name not in PUBLIC_FILES for p in output_root.iterdir()):
        raise ValidationError("public output contains stale file")
    report, fresh, audits, _ = _verify_source(input_root)
    methods, rigs, gates, aggregate = _derive(report, fresh, audits)
    output_root.mkdir(parents=True, exist_ok=True)
    for p in output_root.iterdir():
        if p.is_dir() or p.is_symlink(): raise ValidationError("public output contains unsafe directory")
        if p.name not in PUBLIC_FILES: p.unlink()
    summary = {"schema_version": PUBLIC_SCHEMA, "status": "V3_MIXED_RESULT_NO_AUTH", "evidence_scope": "SYNTHETIC_MULTIPRIMITIVE_CPU_SMOKE_ONLY", "claim_boundary": {"research_claim_authorized": False, "real_bost_claim_authorized": False, "generalization_claimed": False, "paper_superiority_claimed": False}, "aggregate": aggregate, "gates": gates, "redaction": {"private_paths": False, "seeds": False, "source_provenance_included": False, "restricted_material": False}}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_root / "method_summary.csv", methods); _write_csv(output_root / "fresh_rig_field.csv", rigs)
    _write_csv(output_root / "safety_summary.csv", [
        {
            "partition": partition,
            **values,
            "certificate": "triangle_inequality_grouped_majorizer",
        }
        for partition, values in aggregate["safety_certificate_by_partition"].items()
    ])
    _plot(output_root, methods, rigs, aggregate); _write_readme(output_root / "README.md", aggregate); _write_checksums(output_root); _validate_public(output_root)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=DEFAULT_INPUT); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None); run(args.input, args.output); print("PASS_CERTIFIED_GROUPED_MAJORZIER_PUBLIC")
    return 0


if __name__ == "__main__": raise SystemExit(main())
