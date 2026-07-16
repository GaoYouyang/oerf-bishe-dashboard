#!/usr/bin/env python3
"""Compare diagonal and exact-covariance complexity rules on opened v5c.

The exact covariance is reconstructed from the clean synthetic observation and
the known generator coefficients.  It is an oracle mechanism diagnostic, not a
deployable estimator and not confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .covariance_decoupled_complexity import (
        build_covariance_decoupled_surface,
    )
    from .covariance_whitening import camera_noise_covariance
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
        read_json,
        support_mask_from_config,
    )
    from .run_v5d_decoupled_complexity_screening import (
        selection_row,
        sha256,
        summarize_screening,
        surface_rows,
        write_checksums,
        write_csv,
    )
except ImportError:
    from covariance_decoupled_complexity import (
        build_covariance_decoupled_surface,
    )
    from covariance_whitening import camera_noise_covariance
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
        read_json,
        support_mask_from_config,
    )
    from run_v5d_decoupled_complexity_screening import (
        selection_row,
        sha256,
        summarize_screening,
        surface_rows,
        write_checksums,
        write_csv,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "v5e_covariance_whitening_diagnostic.json"
DEFAULT_OUTPUT = ROOT / "results" / "v5e_covariance_whitening_diagnostic"
WHITENING_MODES = ("diagonal", "exact_covariance_oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validated_variants(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    variants = tuple(dict(item) for item in config.get("variants", []))
    if not variants:
        raise ValueError("at least one covariance diagnostic variant is required")
    identifiers = [str(item.get("id", "")) for item in variants]
    if any(not identifier for identifier in identifiers):
        raise ValueError("every variant needs a nonempty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("variant ids must be unique")
    for item in variants:
        if item.get("whitening") not in WHITENING_MODES:
            raise ValueError("unknown whitening mode")
        if item.get("method") not in METHODS[:-1]:
            raise ValueError("v5e diagnostic excludes nested_cv")
    modes_by_method: dict[str, list[str]] = {}
    for item in variants:
        modes_by_method.setdefault(str(item["method"]), []).append(
            str(item["whitening"])
        )
    expected_modes = sorted(WHITENING_MODES)
    for method, modes in modes_by_method.items():
        if sorted(modes) != expected_modes:
            raise ValueError(
                f"method {method} requires one variant for each whitening mode"
            )
    return variants


def exact_covariances(
    block: DevelopmentBlock, source_config: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    return tuple(
        camera_noise_covariance(
            clean,
            sigma,
            correlation_fraction=float(source_config["correlation_fraction"]),
            signal_fraction=float(source_config["signal_fraction"]),
        )
        for clean, sigma in zip(
            block.clean_observations, block.noise_std, strict=True
        )
    )


def tagged_surface_rows(
    block: DevelopmentBlock,
    surface: DecoupledSurface,
    whitening: str,
) -> list[dict[str, Any]]:
    rows = surface_rows(block, surface)
    for row in rows:
        row["whitening"] = whitening
    return rows


def tagged_selection_row(
    block: DevelopmentBlock,
    selection: DecoupledSelection,
    variant: dict[str, Any],
    radii: np.ndarray,
    kappa_bounds: tuple[float, float],
) -> dict[str, Any]:
    row = selection_row(block, selection, variant, radii, kappa_bounds)
    row["whitening"] = str(variant["whitening"])
    row["covariance_uses_clean_synthetic_truth"] = (
        variant["whitening"] == "exact_covariance_oracle"
    )
    return row


def paired_summary(
    summaries: Sequence[dict[str, Any]], variants: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["variant_id"]): row for row in summaries}
    output: list[dict[str, Any]] = []
    methods = []
    for variant in variants:
        method = str(variant["method"])
        if method not in methods:
            methods.append(method)
    for method in methods:
        diagonal_id = next(
            str(item["id"])
            for item in variants
            if item["method"] == method and item["whitening"] == "diagonal"
        )
        exact_id = next(
            str(item["id"])
            for item in variants
            if item["method"] == method
            and item["whitening"] == "exact_covariance_oracle"
        )
        diagonal = by_id[diagonal_id]
        exact = by_id[exact_id]
        output.append(
            {
                "method": method,
                "diagonal_variant_id": diagonal_id,
                "exact_variant_id": exact_id,
                "diagonal_nearest_bank_matches": int(
                    diagonal["nearest_bank_match_count"]
                ),
                "exact_nearest_bank_matches": int(
                    exact["nearest_bank_match_count"]
                ),
                "match_delta_exact_minus_diagonal": int(
                    exact["nearest_bank_match_count"]
                    - diagonal["nearest_bank_match_count"]
                ),
                "diagonal_mean_raw_discrepancy": float(
                    diagonal["mean_selected_whitened_discrepancy"]
                ),
                "exact_mean_raw_discrepancy": float(
                    exact["mean_selected_whitened_discrepancy"]
                ),
                "diagonal_mean_df_corrected_discrepancy": float(
                    diagonal["mean_selected_df_corrected_discrepancy"]
                ),
                "exact_mean_df_corrected_discrepancy": float(
                    exact["mean_selected_df_corrected_discrepancy"]
                ),
            }
        )
    return output


def write_figure(
    path: Path,
    selection_rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
) -> None:
    variant_ids = [str(row["variant_id"]) for row in summaries]
    blocks = sorted({str(row["block_id"]) for row in selection_rows})
    errors = np.zeros((len(variant_ids), len(blocks)), dtype=float)
    for row in selection_rows:
        i = variant_ids.index(str(row["variant_id"]))
        j = blocks.index(str(row["block_id"]))
        errors[i, j] = float(row["selected_radius"]) - float(
            row["nearest_bank_radius"]
        )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    image = axes[0, 0].imshow(errors, cmap="coolwarm", vmin=-0.16, vmax=0.16)
    axes[0, 0].set(
        title="Radius error on opened v5c",
        xlabel="block",
        ylabel="variant",
        xticks=np.arange(len(blocks)),
        yticks=np.arange(len(variant_ids)),
        xticklabels=[str(index + 1) for index in range(len(blocks))],
        yticklabels=variant_ids,
    )
    figure.colorbar(image, ax=axes[0, 0], label="selected minus nearest truth bank")

    axes[0, 1].barh(
        variant_ids,
        [int(row["nearest_bank_match_count"]) for row in summaries],
        color=[
            "tab:blue" if str(row["variant_id"]).startswith("diagonal") else "tab:green"
            for row in summaries
        ],
    )
    axes[0, 1].set(
        title="Opened-truth matches (diagnostic only)",
        xlabel="matches out of 6",
        xlim=(0, 6.2),
    )

    methods = [str(row["method"]) for row in pairs]
    locations = np.arange(len(methods))
    width = 0.36
    axes[1, 0].bar(
        locations - width / 2,
        [float(row["diagonal_mean_raw_discrepancy"]) for row in pairs],
        width,
        label="diagonal raw RSS/m",
        color="tab:blue",
    )
    axes[1, 0].bar(
        locations + width / 2,
        [float(row["exact_mean_raw_discrepancy"]) for row in pairs],
        width,
        label="exact covariance raw RSS/m",
        color="tab:green",
    )
    axes[1, 0].axhline(1.0, color="tab:gray", linestyle="--")
    axes[1, 0].set(
        title="Selected raw discrepancy",
        ylabel="mean RSS / measurement count",
        xticks=locations,
        xticklabels=methods,
    )
    axes[1, 0].legend()

    axes[1, 1].bar(
        locations - width / 2,
        [
            float(row["diagonal_mean_df_corrected_discrepancy"])
            for row in pairs
        ],
        width,
        label="diagonal",
        color="tab:blue",
    )
    axes[1, 1].bar(
        locations + width / 2,
        [
            float(row["exact_mean_df_corrected_discrepancy"])
            for row in pairs
        ],
        width,
        label="exact covariance oracle",
        color="tab:green",
    )
    axes[1, 1].axhline(1.0, color="tab:gray", linestyle="--")
    axes[1, 1].set(
        title="Residual-DOF corrected discrepancy",
        ylabel="RSS / (m - 2 tr(H) + tr(H^2))",
        xticks=locations,
        xticklabels=methods,
    )
    axes[1, 1].legend()
    figure.suptitle(
        "v5e exact covariance oracle diagnostic on opened v5c - not confirmatory",
        fontsize=15,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    config = read_json(config_path)
    variants = validated_variants(config)
    source_config_path = (ROOT / str(config["source_config"])).resolve()
    source_report_path = (ROOT / str(config["source_phase_a_report"])).resolve()
    source_config = read_json(source_config_path)
    kappas = tuple(float(value) for value in config["kappas"])
    if len(kappas) < 3 or np.any(np.diff(kappas) <= 0.0):
        raise ValueError("kappas must be strictly increasing")
    radii = np.asarray(source_config["candidate_aperture_radii"], dtype=float)
    support = support_mask_from_config(source_config)
    blocks, _ = build_development_blocks(source_config)

    surface_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    for block in blocks:
        diagonal_surface = build_decoupled_surface(
            block.reconstruction_bank,
            radii,
            block.observations,
            block.noise_std,
            block.inner_views,
            support,
            kappas,
            include_nested_cross_validation=False,
        )
        covariance_surface = build_covariance_decoupled_surface(
            block.reconstruction_bank,
            radii,
            block.observations,
            exact_covariances(block, source_config),
            block.inner_views,
            support,
            kappas,
            include_nested_cross_validation=False,
        )
        surfaces = {
            "diagonal": diagonal_surface,
            "exact_covariance_oracle": covariance_surface,
        }
        for whitening, surface in surfaces.items():
            surface_records.extend(tagged_surface_rows(block, surface, whitening))
        for variant in variants:
            selection = select_radius_from_surface(
                str(variant["method"]),
                surfaces[str(variant["whitening"])],
                discrepancy_target=float(variant.get("discrepancy_target", 1.0)),
                effective_degrees_of_freedom_target=float(
                    variant.get("effective_degrees_of_freedom_target", 0.5)
                ),
            )
            selection_records.append(
                tagged_selection_row(
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
    pairs = paired_summary(summaries, variants)
    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = output / "config_snapshot.json"
    config_snapshot.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    surface_path = output / "whitening_surfaces.csv"
    selection_path = output / "selection_rows.csv"
    summary_path = output / "variant_summary.csv"
    paired_path = output / "paired_whitening_summary.csv"
    write_csv(surface_path, surface_records)
    write_csv(selection_path, selection_records)
    write_csv(summary_path, summaries)
    write_csv(paired_path, pairs)
    figure_path = output / "v5e_covariance_whitening_diagnostic.png"
    write_figure(figure_path, selection_records, summaries, pairs)

    report = {
        "claim_status": config["claim_status"],
        "scientific_boundary": (
            "opened-v5c post-open mechanism diagnostic; exact covariance uses clean "
            "synthetic truth; no deployable covariance estimate, new data, real BOS, "
            "or confirmatory superiority evidence"
        ),
        "variant_summary": summaries,
        "paired_whitening_summary": pairs,
        "surface_row_count": len(surface_records),
        "selection_row_count": len(selection_records),
        "exact_covariance_uses_clean_synthetic_truth": True,
        "residual_df_formula": "m - 2*trace(H) + trace(H^2)",
        "next_step": (
            "use the result to decide whether empirical flow-off covariance and "
            "continuous target solving are worth a session-level outer evaluation"
        ),
        "source_hashes": {
            "config": sha256(config_path),
            "source_config": sha256(source_config_path),
            "source_phase_a_report": sha256(source_report_path),
            "runner": sha256(Path(__file__).resolve()),
            "covariance_whitening": sha256(
                (ROOT / "covariance_whitening.py").resolve()
            ),
            "covariance_surface": sha256(
                (ROOT / "covariance_decoupled_complexity.py").resolve()
            ),
            "decoupled_surface": sha256(
                (ROOT / "decoupled_complexity.py").resolve()
            ),
            "noise_generator": sha256(
                (ROOT / "independent_reaction_bost.py").resolve()
            ),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme_path = output / "README.md"
    readme_path.write_text(
        "# v5e exact-covariance whitening diagnostic\n\n"
        "This experiment compares diagonal and exact per-camera covariance whitening "
        "on the already opened v5c blocks. Exact covariance is reconstructed from the "
        "clean synthetic observation, so it is an oracle mechanism check, not a "
        "deployable result. See `paired_whitening_summary.csv` before interpreting any "
        "truth-match count.\n",
        encoding="utf-8",
    )
    write_checksums(
        output,
        [
            config_snapshot,
            surface_path,
            selection_path,
            summary_path,
            paired_path,
            figure_path,
            report_path,
            readme_path,
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
