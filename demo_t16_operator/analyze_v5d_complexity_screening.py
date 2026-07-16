#!/usr/bin/env python3
"""Audit the opened-v5c v5d screening without changing its selections.

The Phase-A screen deliberately used opened truth for method development.  This
post-open audit records target reachability, noise-model assumptions, effective
independent units, and complete source hashes.  It cannot promote a development
variant into confirmatory evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "results" / "v5d_decoupled_complexity_screening"
DEFAULT_OUTPUT = ROOT / "results" / "v5d_decoupled_complexity_postopen_diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def target_path_diagnostic(
    values: Iterable[float], target: float
) -> dict[str, float | bool]:
    """Describe whether a continuous target is bracketed by a sampled path."""

    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("target path must contain finite values")
    level = float(target)
    if not np.isfinite(level):
        raise ValueError("target must be finite")
    lower = float(np.min(array))
    upper = float(np.max(array))
    return {
        "path_min": lower,
        "path_max": upper,
        "target": level,
        "target_bracketed": bool(lower <= level <= upper),
        "nearest_grid_gap": float(np.min(np.abs(array - level))),
    }


def build_target_rows(
    surface_rows: Sequence[dict[str, str]],
    selection_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    """Audit Morozov and equal-DF targets along selected-radius fold paths."""

    output: list[dict[str, Any]] = []
    for selection in selection_rows:
        method = selection["method"]
        if method not in {"morozov", "equal_df"}:
            continue
        target = float(
            selection[
                "discrepancy_target" if method == "morozov" else "effective_df_target"
            ]
        )
        metric = (
            "mean_whitened_discrepancy"
            if method == "morozov"
            else "mean_effective_df_fraction"
        )
        selected_radius = float(selection["selected_radius"])
        matching = [
            row
            for row in surface_rows
            if row["rig_id"] == selection["rig_id"]
            and row["block_id"] == selection["block_id"]
            and np.isclose(float(row["candidate_aperture_radius"]), selected_radius)
        ]
        validation_views = sorted(
            {int(row["validation_camera_index"]) for row in matching}
        )
        if not validation_views:
            raise ValueError("selected radius has no saved surface path")
        for view in validation_views:
            path = [
                float(row[metric])
                for row in matching
                if int(row["validation_camera_index"]) == view
            ]
            diagnostic = target_path_diagnostic(path, target)
            output.append(
                {
                    "variant_id": selection["variant_id"],
                    "method": method,
                    "rig_id": selection["rig_id"],
                    "block_id": selection["block_id"],
                    "validation_camera_index": view,
                    "selected_radius": selected_radius,
                    "target_metric": metric,
                    **diagnostic,
                }
            )
    return output


def summarize_targets(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant_id in sorted({str(row["variant_id"]) for row in rows}):
        selected = [row for row in rows if row["variant_id"] == variant_id]
        summaries.append(
            {
                "variant_id": variant_id,
                "method": selected[0]["method"],
                "fold_path_count": len(selected),
                "target_bracketed_count": int(
                    sum(bool(row["target_bracketed"]) for row in selected)
                ),
                "target_bracketed_fraction": float(
                    np.mean([bool(row["target_bracketed"]) for row in selected])
                ),
                "mean_nearest_grid_gap": float(
                    np.mean([float(row["nearest_grid_gap"]) for row in selected])
                ),
                "maximum_nearest_grid_gap": float(
                    np.max([float(row["nearest_grid_gap"]) for row in selected])
                ),
            }
        )
    return summaries


def write_figure(
    path: Path,
    variants: Sequence[dict[str, str]],
    target_summaries: Sequence[dict[str, Any]],
) -> None:
    order = [row["variant_id"] for row in variants]
    match = {row["variant_id"]: int(row["nearest_bank_match_count"]) for row in variants}
    boundary = {
        row["variant_id"]: int(row["selected_radius_boundary_count"])
        for row in variants
    }
    summary_by_id = {str(row["variant_id"]): row for row in target_summaries}
    morozov = [row for row in variants if row["method"] == "morozov"]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].barh(order, [match[item] for item in order], color="tab:green")
    axes[0, 0].set(
        title="Opened-truth matches (development only)",
        xlabel="nearest-bank matches out of 6",
        xlim=(0, 6.2),
    )

    targets = [float(row["variant_id"].split("_")[-1].replace("p", ".")) for row in morozov]
    achieved = [float(row["mean_selected_whitened_discrepancy"]) for row in morozov]
    colors = [int(row["nearest_bank_match_count"]) for row in morozov]
    axes[0, 1].scatter(targets, achieved, c=colors, cmap="viridis", s=110)
    lower = min(targets + achieved) - 0.08
    upper = max(targets + achieved) + 0.08
    axes[0, 1].plot([lower, upper], [lower, upper], "--", color="tab:gray")
    for row, x_value, y_value in zip(morozov, targets, achieved, strict=True):
        axes[0, 1].annotate(row["variant_id"], (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0, 1].set(
        title="Morozov target vs selected diagonal discrepancy",
        xlabel="configured target",
        ylabel="mean selected discrepancy",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )

    axes[1, 0].barh(order, [boundary[item] for item in order], color="tab:red")
    axes[1, 0].set(
        title="Radius-bank boundary selections",
        xlabel="blocks at a radius boundary out of 6",
        xlim=(0, 6.2),
    )

    axes[1, 1].axis("off")
    morozov_one = next(row for row in variants if row["variant_id"] == "morozov_1p00")
    morozov_half = next(row for row in variants if row["variant_id"] == "morozov_0p50")
    bracket_half = summary_by_id["morozov_0p50"]
    bracket_one = summary_by_id["morozov_1p00"]
    lines = [
        "Why Phase A cannot freeze a winner",
        "",
        "1. Truth labels ranked 10 variants on opened v5c.",
        "2. sigma and support are synthetic truth-derived oracles.",
        "3. Generator noise is camera-correlated; whitening is diagonal.",
        "4. Six blocks reuse fields within only two rigs/sessions.",
        "",
        f"Morozov 0.50: {morozov_half['nearest_bank_match_count']}/6 matches; ",
        f"target bracketed {bracket_half['target_bracketed_count']}/{bracket_half['fold_path_count']} fold paths.",
        f"Morozov 1.00: {morozov_one['nearest_bank_match_count']}/6 matches; ",
        f"target bracketed {bracket_one['target_bracketed_count']}/{bracket_one['fold_path_count']} fold paths.",
        "",
        "Verdict: useful mechanism diagnosis, zero confirmatory superiority.",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.45,
        family="monospace",
    )
    figure.suptitle(
        "v5d Phase-A post-open assumption audit - no method freeze",
        fontsize=15,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"diagnosis output already exists: {output}")
    selection_rows = read_csv(source / "selection_rows.csv")
    surface_rows = read_csv(source / "complexity_surface.csv")
    variants = read_csv(source / "variant_summary.csv")
    target_rows = build_target_rows(surface_rows, selection_rows)
    target_summaries = summarize_targets(target_rows)
    config = json.loads((source / "config_snapshot.json").read_text(encoding="utf-8"))
    source_config_path = (ROOT / str(config["source_config"])).resolve()
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))

    output.mkdir(parents=True, exist_ok=False)
    target_path = output / "target_reachability.csv"
    target_summary_path = output / "target_reachability_summary.csv"
    write_csv(target_path, target_rows)
    write_csv(target_summary_path, target_summaries)
    figure_path = output / "v5d_phase_a_assumption_audit.png"
    write_figure(figure_path, variants, target_summaries)

    dependencies = [
        Path(__file__).resolve(),
        (ROOT / "decoupled_complexity.py").resolve(),
        (ROOT / "run_v5d_decoupled_complexity_screening.py").resolve(),
        (ROOT / "run_v5b_rig_shared_profile_pilot.py").resolve(),
        (ROOT / "rig_shared_profile.py").resolve(),
        (ROOT / "nested_crossview.py").resolve(),
        (ROOT / "finite_aperture_bost.py").resolve(),
        (ROOT / "independent_reaction_bost.py").resolve(),
        (ROOT / "v5c_nested_crossview_protocol.md").resolve(),
        source_config_path,
    ]
    report = {
        "claim_status": "V5D_PHASE_A_POSTOPEN_DIAGNOSIS_NO_METHOD_FREEZE",
        "scientific_verdict": "NO_METHOD_FREEZE",
        "phase_a_is_confirmatory": False,
        "nearest_bank_matches": {
            row["variant_id"]: int(row["nearest_bank_match_count"])
            for row in variants
        },
        "key_contrast": {
            "truth_tuned_morozov_0p50_matches": int(
                next(row for row in variants if row["variant_id"] == "morozov_0p50")[
                    "nearest_bank_match_count"
                ]
            ),
            "nominal_unit_discrepancy_morozov_1p00_matches": int(
                next(row for row in variants if row["variant_id"] == "morozov_1p00")[
                    "nearest_bank_match_count"
                ]
            ),
            "gcv_matches": int(
                next(row for row in variants if row["variant_id"] == "gcv")[
                    "nearest_bank_match_count"
                ]
            ),
        },
        "target_reachability": target_summaries,
        "assumption_audit": {
            "opened_truth_used_for_variant_ranking": True,
            "noise_scale_derived_from_clean_truth": True,
            "support_derived_from_synthetic_generator": True,
            "noise_generator_has_camera_internal_correlation": bool(
                float(source_config["correlation_fraction"]) > 0.0
            ),
            "phase_a_whitening_uses_only_diagonal_sigma": True,
            "upre_unbiased_interpretation_authorized": False,
            "standard_gcv_iid_interpretation_authorized": False,
            "morozov_unit_discrepancy_interpretation_authorized": False,
            "independent_rig_session_count": len(source_config["rigs"]),
            "blocks_reuse_fields_within_rig": True,
        },
        "required_before_new_lock": [
            "covariance-aware or empirically prewhitened residual diagnostics",
            "explicit unreachable-target and common-complexity handling",
            "frozen method and target selected without new truth labels",
            "outer camera/session evaluation not reused for radius selection",
            "new families, seeds, rigs, and eventually real f-stop data",
        ],
        "source_artifact_hashes": {
            path.name: sha256(path)
            for path in sorted(source.iterdir())
            if path.is_file()
        },
        "complete_dependency_hashes": {
            str(path.relative_to(ROOT.parent)): sha256(path) for path in dependencies
        },
    }
    report_path = output / "diagnosis.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        "# v5d Phase-A post-open assumption audit\n\n"
        "**Verdict: `NO_METHOD_FREEZE`.** The 5/6 result for `morozov_0p50` "
        "was selected with opened truth across ten variants. Its sigma and support "
        "are synthetic oracles, while the generated camera noise is correlated and "
        "the screen used diagonal whitening. The physically nominal unit-discrepancy "
        "variant recovered 0/6 and selected a radius-bank boundary in 4/6 blocks.\n\n"
        "This package preserves target-path reachability and complete dependency "
        "hashes. It is a mechanism diagnosis, not evidence that Morozov, GCV, or an "
        "OERF-ready algorithm is superior.\n",
        encoding="utf-8",
    )
    checksum_paths = [target_path, target_summary_path, figure_path, report_path, readme_path]
    (output / "checksums.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in checksum_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
