#!/usr/bin/env python3
"""Screen v5d complexity-first selectors on the already opened v5c data.

This runner intentionally uses v5c truth labels for development ranking.  It
cannot create confirmatory evidence; its output only selects methods that may
later be frozen and tested on new field families, seeds, rigs, and real f-stop
data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .decoupled_complexity import (
        METHODS,
        DecoupledSelection,
        DecoupledSurface,
        build_decoupled_surface,
        select_radius_from_surface,
    )
    from .run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        read_json,
        support_mask_from_config,
    )
except ImportError:
    from decoupled_complexity import (
        METHODS,
        DecoupledSelection,
        DecoupledSurface,
        build_decoupled_surface,
        select_radius_from_surface,
    )
    from run_v5b_rig_shared_profile_pilot import (
        DevelopmentBlock,
        build_development_blocks,
        nearest_index,
        read_json,
        support_mask_from_config,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5d_decoupled_complexity_screening.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5d_decoupled_complexity_screening"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validated_variants(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    variants = tuple(dict(item) for item in config.get("variants", []))
    if not variants:
        raise ValueError("at least one screening variant is required")
    identifiers = [str(item.get("id", "")) for item in variants]
    if any(not identifier for identifier in identifiers):
        raise ValueError("every variant needs a nonempty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("variant ids must be unique")
    for item in variants:
        if item.get("method") not in METHODS[:-1]:
            raise ValueError(
                "phase-A screening supports non-nested complexity methods only"
            )
    return variants


def surface_rows(
    block: DevelopmentBlock, surface: DecoupledSurface
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for radius, fold_paths in zip(surface.radii, surface.paths, strict=True):
        for validation_view, points in zip(
            surface.validation_views, fold_paths, strict=True
        ):
            for point in points:
                rows.append(
                    {
                        "rig_id": block.rig_id,
                        "block_id": block.block_id,
                        "true_aperture_radius": block.true_radius,
                        "metadata_aperture_radius": block.metadata_radius,
                        "candidate_aperture_radius": radius,
                        "validation_camera_index": validation_view,
                        "kappa": point.kappa,
                        "mean_gcv": point.mean_generalized_cross_validation,
                        "mean_upre": point.mean_unbiased_predictive_risk,
                        "mean_effective_df_fraction": (
                            point.mean_effective_degrees_of_freedom_fraction
                        ),
                        "mean_whitened_discrepancy": (
                            point.mean_whitened_discrepancy
                        ),
                        "mean_df_corrected_discrepancy": (
                            point.mean_degrees_of_freedom_corrected_discrepancy
                        ),
                        "outer_validation_mse": point.outer_validation_mse,
                    }
                )
    return rows


def selection_row(
    block: DevelopmentBlock,
    selection: DecoupledSelection,
    variant: dict[str, Any],
    radii: np.ndarray,
    kappa_bounds: tuple[float, float],
) -> dict[str, Any]:
    selected = selection.selected
    nearest = float(radii[nearest_index(radii, block.true_radius)])
    selected_kappas = np.asarray(selected.fold_selected_kappas, dtype=float)
    boundary = np.isclose(selected_kappas, kappa_bounds[0]) | np.isclose(
        selected_kappas, kappa_bounds[1]
    )
    return {
        "variant_id": str(variant["id"]),
        "method": selection.method,
        "discrepancy_target": float(variant.get("discrepancy_target", 1.0)),
        "effective_df_target": float(
            variant.get("effective_degrees_of_freedom_target", 0.5)
        ),
        "rig_id": block.rig_id,
        "block_id": block.block_id,
        "true_aperture_radius": block.true_radius,
        "nearest_bank_radius": nearest,
        "metadata_aperture_radius": block.metadata_radius,
        "selected_radius": selected.radius,
        "nearest_bank_match": bool(np.isclose(selected.radius, nearest)),
        "selected_radius_boundary": selected.radius_index in (0, len(radii) - 1),
        "mean_outer_validation_mse": selected.mean_validation_mse,
        "relative_radius_margin": selection.relative_radius_margin,
        "fold_score_deletion_radius_stability_fraction": (
            selection.fold_score_deletion_radius_stability_fraction
        ),
        "fold_selected_kappas": "|".join(
            f"{value:.12g}" for value in selected.fold_selected_kappas
        ),
        "fold_kappa_boundary_count": int(np.sum(boundary)),
        "mean_selected_effective_df_fraction": float(
            np.mean(selected.fold_effective_degrees_of_freedom_fractions)
        ),
        "mean_selected_whitened_discrepancy": float(
            np.mean(selected.fold_whitened_discrepancies)
        ),
        "mean_selected_df_corrected_discrepancy": float(
            np.mean(
                selected.fold_degrees_of_freedom_corrected_discrepancies
            )
        ),
    }


def summarize_screening(
    rows: Sequence[dict[str, Any]], variant_order: Sequence[str]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant_id in variant_order:
        selected = [row for row in rows if row["variant_id"] == variant_id]
        if not selected:
            raise ValueError(f"variant has no rows: {variant_id}")
        summaries.append(
            {
                "variant_id": variant_id,
                "method": selected[0]["method"],
                "block_count": len(selected),
                "nearest_bank_match_count": int(
                    sum(bool(row["nearest_bank_match"]) for row in selected)
                ),
                "selected_radius_boundary_count": int(
                    sum(bool(row["selected_radius_boundary"]) for row in selected)
                ),
                "fold_kappa_boundary_count": int(
                    sum(int(row["fold_kappa_boundary_count"]) for row in selected)
                ),
                "mean_fold_score_deletion_radius_stability_fraction": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "fold_score_deletion_radius_stability_fraction"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_outer_validation_mse": float(
                    np.mean(
                        [float(row["mean_outer_validation_mse"]) for row in selected]
                    )
                ),
                "mean_selected_effective_df_fraction": float(
                    np.mean(
                        [
                            float(row["mean_selected_effective_df_fraction"])
                            for row in selected
                        ]
                    )
                ),
                "mean_selected_whitened_discrepancy": float(
                    np.mean(
                        [
                            float(row["mean_selected_whitened_discrepancy"])
                            for row in selected
                        ]
                    )
                ),
                "mean_selected_df_corrected_discrepancy": float(
                    np.mean(
                        [
                            float(
                                row[
                                    "mean_selected_df_corrected_discrepancy"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            -int(row["nearest_bank_match_count"]),
            int(row["fold_kappa_boundary_count"]),
            -float(row["mean_fold_score_deletion_radius_stability_fraction"]),
            float(row["mean_outer_validation_mse"]),
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["development_rank"] = rank
    return summaries


def write_figure(
    path: Path,
    selection_rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
) -> None:
    variant_ids = [str(row["variant_id"]) for row in summaries]
    blocks = sorted({str(row["block_id"]) for row in selection_rows})
    error = np.zeros((len(variant_ids), len(blocks)), dtype=float)
    kappa_boundary = np.zeros_like(error)
    for row in selection_rows:
        i = variant_ids.index(str(row["variant_id"]))
        j = blocks.index(str(row["block_id"]))
        error[i, j] = float(row["selected_radius"]) - float(
            row["nearest_bank_radius"]
        )
        kappa_boundary[i, j] = float(row["fold_kappa_boundary_count"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    image = axes[0, 0].imshow(error, cmap="coolwarm", vmin=-0.16, vmax=0.16)
    axes[0, 0].set(
        title="Selected radius minus nearest truth-bank radius",
        xlabel="opened v5c block",
        ylabel="development variant",
        xticks=np.arange(len(blocks)),
        yticks=np.arange(len(variant_ids)),
        xticklabels=[str(index + 1) for index in range(len(blocks))],
        yticklabels=variant_ids,
    )
    figure.colorbar(image, ax=axes[0, 0], label="radius error")

    axes[0, 1].barh(
        variant_ids,
        [int(row["nearest_bank_match_count"]) for row in summaries],
        color="tab:green",
    )
    axes[0, 1].set(
        title="Nearest-bank matches on opened v5c",
        xlabel="matched blocks out of 6",
        xlim=(0, 6.2),
    )

    image = axes[1, 0].imshow(kappa_boundary, cmap="magma", vmin=0, vmax=4)
    axes[1, 0].set(
        title="Fold-local kappa boundary selections",
        xlabel="opened v5c block",
        ylabel="development variant",
        xticks=np.arange(len(blocks)),
        yticks=np.arange(len(variant_ids)),
        xticklabels=[str(index + 1) for index in range(len(blocks))],
        yticklabels=variant_ids,
    )
    figure.colorbar(image, ax=axes[1, 0], label="boundary folds out of 4")

    axes[1, 1].scatter(
        [float(row["mean_selected_effective_df_fraction"]) for row in summaries],
        [float(row["mean_selected_whitened_discrepancy"]) for row in summaries],
        c=[int(row["nearest_bank_match_count"]) for row in summaries],
        cmap="viridis",
        s=90,
    )
    for row in summaries:
        axes[1, 1].annotate(
            str(row["variant_id"]),
            (
                float(row["mean_selected_effective_df_fraction"]),
                float(row["mean_selected_whitened_discrepancy"]),
            ),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1, 1].set(
        title="Selected complexity operating points",
        xlabel="mean effective-DF fraction",
        ylabel="mean whitened discrepancy",
    )
    figure.suptitle(
        "v5d complexity-first screening on opened v5c data - not confirmatory",
        fontsize=14,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def write_checksums(output: Path, paths: Sequence[Path]) -> None:
    lines = [f"{sha256(path)}  {path.name}" for path in paths]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"screening output already exists: {output}")
    screening = read_json(config_path)
    variants = validated_variants(screening)
    source_config_path = (ROOT / str(screening["source_config"])).resolve()
    source_report_path = (ROOT / str(screening["source_first_open_report"])).resolve()
    source_config = read_json(source_config_path)
    source_report = read_json(source_report_path)
    kappas = tuple(float(value) for value in screening["kappas"])
    if len(kappas) < 3 or np.any(np.diff(kappas) <= 0.0):
        raise ValueError("screening kappas must be strictly increasing")
    radii = np.asarray(source_config["candidate_aperture_radii"], dtype=float)
    support = support_mask_from_config(source_config)
    blocks, _ = build_development_blocks(source_config)

    surface_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    for block in blocks:
        surface = build_decoupled_surface(
            block.reconstruction_bank,
            radii,
            block.observations,
            block.noise_std,
            block.inner_views,
            support,
            kappas,
            include_nested_cross_validation=False,
        )
        surface_records.extend(surface_rows(block, surface))
        for variant in variants:
            selection = select_radius_from_surface(
                str(variant["method"]),
                surface,
                discrepancy_target=float(variant.get("discrepancy_target", 1.0)),
                effective_degrees_of_freedom_target=float(
                    variant.get("effective_degrees_of_freedom_target", 0.5)
                ),
            )
            selection_records.append(
                selection_row(
                    block,
                    selection,
                    variant,
                    radii,
                    (kappas[0], kappas[-1]),
                )
            )

    summaries = summarize_screening(
        selection_records, [str(item["id"]) for item in variants]
    )
    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(screening, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    surface_path = output / "complexity_surface.csv"
    selection_path = output / "selection_rows.csv"
    summary_path = output / "variant_summary.csv"
    write_csv(surface_path, surface_records)
    write_csv(selection_path, selection_records)
    write_csv(summary_path, summaries)
    figure_path = output / "v5d_decoupled_complexity_screening.png"
    write_figure(figure_path, selection_records, summaries)

    report = {
        "claim_status": screening["claim_status"],
        "claim_boundary": (
            "post-open method development on the same deterministic v5c blocks; "
            "truth labels are used for ranking; no new family, seed, rig, real BOS, "
            "or neural-operator superiority evidence"
        ),
        "source_first_open_commit": source_report.get("preopen_git_commit"),
        "source_first_open_claim_status": source_report.get("claim_status"),
        "source_first_open_report_sha256": sha256(source_report_path),
        "surface_row_count": len(surface_records),
        "selection_row_count": len(selection_records),
        "variant_summary": summaries,
        "development_ranking_uses_opened_truth": True,
        "nested_cv_in_phase_a": False,
        "next_step": (
            "run true camera-deletion and nested-CV/refit audits only for a small "
            "predeclared finalist set, then freeze a new-family/new-seed/new-rig lock"
        ),
        "source_hashes": {
            "runner": sha256(Path(__file__).resolve()),
            "decoupled_complexity": sha256(
                (ROOT / "decoupled_complexity.py").resolve()
            ),
            "profile_module": sha256((ROOT / "rig_shared_profile.py").resolve()),
            "screening_config": sha256(config_path),
            "source_v5c_config": sha256(source_config_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        "# v5d decoupled-complexity screening\n\n"
        "This package is post-open development on v5c, not a new lock. Truth labels "
        "are used to rank variants. The complete radius/fold/kappa surface is saved "
        "so later analyses do not silently rebuild or retune it.\n\n"
        f"Top development variant: `{summaries[0]['variant_id']}` with "
        f"{summaries[0]['nearest_bank_match_count']}/6 nearest-bank matches. "
        "That number is not confirmatory and cannot authorize a paper claim.\n",
        encoding="utf-8",
    )
    write_checksums(
        output,
        [
            config_snapshot,
            surface_path,
            selection_path,
            summary_path,
            figure_path,
            report_path,
            readme_path,
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
