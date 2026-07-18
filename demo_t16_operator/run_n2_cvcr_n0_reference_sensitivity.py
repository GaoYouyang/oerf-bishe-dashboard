#!/usr/bin/env python3
"""Post-open high-order reference sensitivity for the held CVCR-N0 gate.

This diagnostic never changes the preregistered HOLD decision or re-scores a
candidate.  It only asks whether increasingly expensive deterministic disk
quadratures approach a stable operator in the prescribed synthetic renderer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .aperture_control_variate import (
        disk_product_quadrature,
        weighted_operator_mean,
    )
    from .finite_aperture_bost import finite_aperture_reference_scale
    from .run_n2_cvcr_n0_mechanism_gate import build_bank, read_json, relative_l2
except ImportError:
    from aperture_control_variate import disk_product_quadrature, weighted_operator_mean
    from finite_aperture_bost import finite_aperture_reference_scale
    from run_n2_cvcr_n0_mechanism_gate import build_bank, read_json, relative_l2


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "n2_cvcr_n0_mechanism_prereg_v1.json"
DEFAULT_HELD_RESULT = ROOT / "results" / "n2_cvcr_n0_mechanism_gate_v1" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "n2_cvcr_n0_reference_sensitivity_postopen_v1"
ORDER_LADDER = (
    (16, 64),
    (20, 80),
    (24, 96),
    (32, 128),
)
DESCRIPTIVE_LAST_STEP_THRESHOLD = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--held-result", type=Path, default=DEFAULT_HELD_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_sensitivity(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rig_summaries = []
    for rig in config["rigs"]:
        rig_id = str(rig["id"])
        angles = np.asarray(rig["angles_degrees"], dtype=float)
        scale = finite_aperture_reference_scale(
            int(config["grid_size"]),
            int(config["depth"]),
            angles,
            aperture_samples=1,
            path_samples=int(rig["path_samples"]),
            cone_u=float(rig["cone_u"]),
            cone_z=float(rig["cone_z"]),
            bend=float(rig["bend"]),
        )
        references = []
        build_seconds = []
        for radial_order, angular_order in ORDER_LADDER:
            points, weights = disk_product_quadrature(radial_order, angular_order)
            start = time.perf_counter()
            bank = build_bank(config, rig, points, scale=scale)
            references.append(weighted_operator_mean(bank, weights))
            build_seconds.append(time.perf_counter() - start)
            del bank
        final_reference = references[-1]
        consecutive = [None]
        for previous, current in zip(references[:-1], references[1:], strict=True):
            consecutive.append(relative_l2(previous, current))
        for index, ((radial_order, angular_order), reference) in enumerate(
            zip(ORDER_LADDER, references, strict=True)
        ):
            rows.append(
                {
                    "rig_id": rig_id,
                    "rig_role": str(rig["role"]),
                    "radial_order": radial_order,
                    "angular_order": angular_order,
                    "point_count": radial_order * angular_order,
                    "consecutive_relative_l2": ""
                    if consecutive[index] is None
                    else float(consecutive[index]),
                    "relative_l2_to_4096_point_reference": relative_l2(
                        reference, final_reference
                    ),
                    "build_seconds": build_seconds[index],
                    "normalization_scale": scale,
                }
            )
        last_step = float(consecutive[-1])
        original_to_final = relative_l2(references[0], final_reference)
        rig_summaries.append(
            {
                "rig_id": rig_id,
                "rig_role": str(rig["role"]),
                "original_1024_to_4096_relative_l2": original_to_final,
                "last_2304_to_4096_relative_l2": last_step,
                "descriptive_last_step_stable": last_step
                <= DESCRIPTIVE_LAST_STEP_THRESHOLD,
            }
        )
    maximum_last_step = max(
        row["last_2304_to_4096_relative_l2"] for row in rig_summaries
    )
    status = (
        "POSTOPEN_REFERENCE_SENSITIVITY_STABLE_AT_4096_DESCRIPTIVE_ONLY"
        if maximum_last_step <= DESCRIPTIVE_LAST_STEP_THRESHOLD
        else "POSTOPEN_REFERENCE_SENSITIVITY_UNRESOLVED_AT_4096"
    )
    summary = {
        "schema": "n2-cvcr-n0-postopen-reference-sensitivity-1",
        "status": status,
        "original_preregistered_decision": "HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED",
        "original_decision_unchanged": True,
        "order_ladder": [
            {
                "radial_order": radial,
                "angular_order": angular,
                "point_count": radial * angular,
            }
            for radial, angular in ORDER_LADDER
        ],
        "descriptive_last_step_threshold": DESCRIPTIVE_LAST_STEP_THRESHOLD,
        "maximum_last_step_relative_l2": maximum_last_step,
        "maximum_original_1024_to_4096_relative_l2": max(
            row["original_1024_to_4096_relative_l2"] for row in rig_summaries
        ),
        "rig_summaries": rig_summaries,
        "authorizations": {
            "rescore_original_candidate": False,
            "change_original_decision": False,
            "train_learned_control_variate": False,
            "three_dimensional_reconstruction_claim": False,
            "experimental_or_oerf_claim": False,
            "generalization_claim": False,
        },
        "claim_ceiling": (
            "Post-open deterministic reference sensitivity in one prescribed "
            "weak-deflection pupil surrogate; no candidate re-scoring."
        ),
    }
    return rows, summary


def write_figure(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    rig_ids = [row["rig_id"] for row in summary["rig_summaries"]]
    colors = ("#315f93", "#146f66", "#8a651b", "#a34f43")
    for rig_id, color in zip(rig_ids, colors, strict=True):
        group = [row for row in rows if row["rig_id"] == rig_id]
        axes[0].plot(
            [row["point_count"] for row in group],
            [max(float(row["relative_l2_to_4096_point_reference"]), 1e-8) for row in group],
            marker="o",
            linewidth=2,
            color=color,
            label=rig_id,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Deterministic pupil points")
    axes[0].set_ylabel("Relative L2 to 4096-point reference")
    axes[0].set_title("Reference ladder sensitivity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    summaries = summary["rig_summaries"]
    positions = np.arange(len(summaries))
    width = 0.36
    axes[1].barh(
        positions - width / 2,
        [100.0 * row["original_1024_to_4096_relative_l2"] for row in summaries],
        height=width,
        color="#315f93",
        label="1024 to 4096",
    )
    axes[1].barh(
        positions + width / 2,
        [100.0 * row["last_2304_to_4096_relative_l2"] for row in summaries],
        height=width,
        color="#146f66",
        label="2304 to 4096",
    )
    axes[1].axvline(
        100.0 * DESCRIPTIVE_LAST_STEP_THRESHOLD,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="descriptive 0.1% line",
    )
    axes[1].set_yticks(positions, labels=rig_ids)
    axes[1].set_xlabel("Relative operator difference (%)")
    axes[1].set_title("Original and final-step sensitivity")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].axis("off")
    text = [
        summary["status"],
        "",
        "Original decision remains:",
        summary["original_preregistered_decision"],
        "",
        f"max 2304 to 4096: {100.0 * summary['maximum_last_step_relative_l2']:.4f}%",
        f"max 1024 to 4096: {100.0 * summary['maximum_original_1024_to_4096_relative_l2']:.4f}%",
        "",
        "No candidate re-scoring.",
        "No learner authorization.",
        "Synthetic pupil sensitivity only.",
    ]
    axes[2].text(
        0.02,
        0.98,
        "\n".join(text),
        transform=axes[2].transAxes,
        va="top",
        family="monospace",
        fontsize=9.5,
        color="#17252b",
    )
    figure.suptitle("N2-CVCR-N0 post-open reference sensitivity", fontsize=15)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_checksums(output: Path) -> None:
    targets = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in targets) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    held_result = read_json(args.held_result)
    if held_result["gate_report"]["decision"] != "HOLD_REFERENCE_QUADRATURE_NOT_CONVERGED":
        raise ValueError("the source result is not the preregistered reference HOLD")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = run_sensitivity(config)
    summary["source_config_sha256"] = sha256(args.config)
    summary["source_held_report_sha256"] = sha256(args.held_result)
    write_csv(output / "reference_ladder.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_figure(output / "reference_sensitivity.png", rows, summary)
    write_checksums(output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
