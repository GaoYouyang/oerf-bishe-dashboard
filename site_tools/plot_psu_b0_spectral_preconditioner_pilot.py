#!/usr/bin/env python3
"""Render the PSU positive-spectral preconditioner no-go pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_SCHEMA = "psu-b0-spectral-preconditioner-pilot-figure-1.0"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def _rows_by_key(
    summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["split"]), str(row["method"])): row
        for row in summary["aggregates"]
    }


def render_figure(
    summary: dict[str, Any],
    output_stem: Path,
) -> dict[str, Any]:
    if summary["status"] != "SPECTRAL_PRECONDITIONER_PILOT_CANDIDATE_NO_GO_OR_INCOMPLETE":
        raise ValueError("this figure expects the frozen no-go pilot")
    rows = _rows_by_key(summary)
    seeds = [int(row["seed"]) for row in summary["training"]]
    learned_methods = [f"learned_seed_{seed}" for seed in seeds]
    splits = (
        "test_iid",
        "test_family_ood",
        "test_noise_ood",
        "test_geometry_ood",
        "test_joint_ood",
        "test_exact_operator_control",
    )
    split_labels = (
        "IID",
        "Family\nOOD",
        "Noise\nOOD",
        "View\nOOD",
        "Joint\nOOD",
        "Exact-op\ncontrol",
    )
    colors = ("#00798c", "#3066be", "#6b5ca5")

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    figure.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.12,
        top=0.89,
        hspace=0.42,
        wspace=0.28,
    )

    selection = summary["sobolev_selection"]
    strength = np.asarray([row["strength"] for row in selection["grid"]])
    validation = np.asarray(
        [row["validation_combined_loss"] for row in selection["grid"]]
    )
    axes[0, 0].plot(
        strength,
        validation,
        marker="o",
        linewidth=2.3,
        color="#15616d",
    )
    chosen = float(selection["selected_strength"])
    chosen_value = float(selection["selected_validation_combined_loss"])
    axes[0, 0].scatter([chosen], [chosen_value], s=90, color="#d1495b", zorder=3)
    axes[0, 0].annotate(
        f"selected p={chosen:g}",
        (chosen, chosen_value),
        xytext=(-72, 18),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#d1495b"},
    )
    axes[0, 0].set_title("A  Strong deterministic baseline selection", loc="left")
    axes[0, 0].set_xlabel("Inverse-Sobolev spectral power")
    axes[0, 0].set_ylabel("Validation field + gradient loss")
    axes[0, 0].grid(alpha=0.25)

    x = np.arange(len(splits))
    width = 0.22
    for index, method in enumerate(learned_methods):
        gain = [rows[(split, method)]["field_gain_vs_sobolev_mean_percent"] for split in splits]
        axes[0, 1].bar(
            x + (index - 1) * width,
            gain,
            width=width,
            color=colors[index],
            label=f"seed {seeds[index]}",
        )
    axes[0, 1].axhline(0.0, color="#272727", linewidth=1)
    axes[0, 1].axvspan(3.55, 4.45, color="#d1495b", alpha=0.10)
    axes[0, 1].set_xticks(x, split_labels)
    axes[0, 1].set_ylabel("Field L2 gain vs selected Sobolev (%)")
    axes[0, 1].set_title("B  Mean gain survives alone, not jointly", loc="left")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=1)
    axes[0, 1].grid(axis="y", alpha=0.25)

    methods = ("identity", "cgls", "sobolev_selected")
    method_labels = ("Identity SD", "CGLS", "Sobolev")
    method_colors = ("#9aa0a6", "#ef8354", "#0b6e4f")
    learned_mean = []
    for split in splits:
        learned_mean.append(
            np.mean(
                [rows[(split, method)]["field_relative_l2_mean"] for method in learned_methods]
            )
        )
    for index, (method, label, color) in enumerate(
        zip(methods, method_labels, method_colors, strict=True)
    ):
        axes[1, 0].plot(
            x,
            [rows[(split, method)]["field_relative_l2_mean"] for split in splits],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[1, 0].plot(
        x,
        learned_mean,
        marker="o",
        linewidth=2.5,
        color="#3066be",
        label="Learned mean",
    )
    axes[1, 0].set_xticks(x, split_labels)
    axes[1, 0].set_ylabel("Field relative L2 (lower is better)")
    axes[1, 0].set_title("C  Physics structure dominates generic Krylov baselines", loc="left")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 0].grid(alpha=0.25)

    p10 = np.asarray(
        [
            np.mean(
                [
                    rows[(split, method)]["field_gain_vs_sobolev_p10_percent"]
                    for method in learned_methods
                ]
            )
            for split in splits
        ]
    )
    harm = np.asarray(
        [
            np.mean(
                [
                    rows[(split, method)]["field_harm_over_one_percent_rate"]
                    for method in learned_methods
                ]
            )
            for split in splits
        ]
    )
    axes[1, 1].bar(x, p10, color="#6b5ca5", width=0.62, label="p10 gain")
    axes[1, 1].axhline(0.0, color="#272727", linewidth=1)
    axes[1, 1].set_xticks(x, split_labels)
    axes[1, 1].set_ylabel("p10 paired field gain (%)")
    axes[1, 1].set_title("D  Tail reliability triggers the no-go", loc="left")
    risk_axis = axes[1, 1].twinx()
    risk_axis.plot(
        x,
        100.0 * harm,
        color="#d1495b",
        marker="D",
        linewidth=2.0,
        label=">1% harm rate",
    )
    risk_axis.set_ylabel(">1% harm rate (%)", color="#d1495b")
    risk_axis.tick_params(axis="y", colors="#d1495b")
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle(
        "PSU finite-aperture spectral preconditioner pilot: IID signal, joint-OOD no-go",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.035,
        (
            "Real PSU support geometry + analytic morphology truth; 4F + 4A^T for "
            "Sobolev and learned methods.\n"
            "No experimental 3D truth, rotation-40, final audit, FNO/DeepONet "
            "comparison, or superiority claim."
        ),
        fontsize=8.5,
        color="#4f5d66",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, options in (
        (".png", {"dpi": 220}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, facecolor="white", **options)
        if suffix == ".svg":
            _normalize_svg(path)
        outputs[path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    plt.close(figure)
    return {
        "schema_version": FIGURE_SCHEMA,
        "status": "FIGURE_COMPLETE_FROZEN_SYNTHETIC_NO_GO",
        "source_status": summary["status"],
        "output_files": outputs,
        "claim_boundary": {
            "real_psu_geometry_used": True,
            "experimental_field_truth_available": False,
            "algorithm_superiority": False,
            "joint_ood_gate_passed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = render_figure(
        _load_json(args.summary),
        args.output_stem,
    )
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
