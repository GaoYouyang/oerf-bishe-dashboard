#!/usr/bin/env python3
"""Render the streamed PSU B0 CGLS gate without implying field truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


FIGURE_SCHEMA = "psu-b0-streaming-baseline-figure-1.0"


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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary must be a JSON object")
    return value


def _slice_panel(
    axis: plt.Axes,
    volume: np.ndarray,
) -> None:
    nz, ny, nx = volume.shape
    slices = (
        ("z mid", volume[nz // 2]),
        ("y mid", volume[:, ny // 2, :]),
        ("x mid", volume[:, :, nx // 2]),
    )
    scale = float(np.max(np.abs(volume)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    axis.set_axis_off()
    lefts = (0.0, 0.34, 0.68)
    image = None
    for left, (title, values) in zip(lefts, slices, strict=True):
        inset = axis.inset_axes([left, 0.17, 0.29, 0.70])
        image = inset.imshow(
            values,
            origin="lower",
            cmap="coolwarm",
            norm=norm,
            interpolation="nearest",
        )
        inset.set_title(title, fontsize=9)
        inset.set_xticks([])
        inset.set_yticks([])
    colorbar_axis = axis.inset_axes([0.08, 0.03, 0.84, 0.055])
    colorbar = axis.figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("reconstructed scalar perturbation (a.u.)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    axis.text(
        0.0,
        0.98,
        "A  16³ support-fit slices",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        0.0,
        0.90,
        "No experimental 3D truth; zero outer-boundary gauge",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5b6470",
    )


def render_figure(
    summary: dict[str, Any],
    volume: np.ndarray,
    output_stem: Path,
) -> dict[str, Any]:
    values = np.asarray(volume)
    if values.ndim == 5:
        values = values[0, 0]
    if values.ndim != 3 or not np.all(np.isfinite(values)):
        raise ValueError("volume must contain one finite three-dimensional field")
    history = summary["optimization"]["history"]
    iterations = np.asarray([row["iteration"] for row in history], dtype=int)
    residual = np.asarray(
        [row["relative_measurement_l2"] for row in history],
        dtype=float,
    )
    normal = np.asarray(
        [row["relative_normal_residual_l2"] for row in history],
        dtype=float,
    )
    view_rows = summary["evaluation"][
        "per_view_support_relative_measurement_l2"
    ]
    view_ids = np.asarray(
        [row["view_id_zero_based"] for row in view_rows],
        dtype=int,
    )
    view_residual = np.asarray(
        [row["relative_measurement_l2"] for row in view_rows],
        dtype=float,
    )
    calls = (
        summary["interface_profile"]["logical_calls"]
        + summary["optimization"]["logical_calls"]
        + summary["evaluation"]["evaluation_logical_calls"]
    )
    forward_seconds = [
        float(row["wall_seconds"])
        for row in calls
        if row["operation"] == "forward"
    ]
    adjoint_seconds = [
        float(row["wall_seconds"])
        for row in calls
        if row["operation"] == "adjoint"
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.4, 7.4),
        constrained_layout=True,
    )
    _slice_panel(axes[0, 0], values)

    axis = axes[0, 1]
    axis.plot(
        iterations,
        residual,
        marker="o",
        color="#136f63",
        linewidth=2,
        label="measurement residual",
    )
    axis.plot(
        iterations,
        normal,
        marker="s",
        color="#c44e52",
        linewidth=1.8,
        label="normal residual",
    )
    axis.set_yscale("log")
    axis.set_xticks(iterations)
    axis.set_xlabel("fixed CGLS iteration")
    axis.set_ylabel("relative L2")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("B  Fixed-budget convergence", loc="left")

    axis = axes[1, 0]
    colors = [
        "#3d6d9c" if value <= 1.0 else "#b94a48"
        for value in view_residual
    ]
    axis.bar(view_ids, view_residual, color=colors, width=0.72)
    axis.axhline(1.0, color="#4d5662", linestyle="--", linewidth=1)
    axis.set_xticks(view_ids)
    axis.set_xlabel("support view id")
    axis.set_ylabel("direct relative measurement L2")
    axis.set_title("C  Per-view support reprojection", loc="left")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    groups = ("forward", "adjoint")
    medians = (
        float(np.median(forward_seconds)),
        float(np.median(adjoint_seconds)),
    )
    minima = (
        float(np.min(forward_seconds)),
        float(np.min(adjoint_seconds)),
    )
    axis.bar(
        np.arange(2),
        medians,
        color=("#5b8e7d", "#d17b49"),
        width=0.58,
        label="median",
    )
    axis.scatter(
        np.arange(2),
        minima,
        color="#1e252b",
        marker="_",
        s=180,
        linewidths=2,
        label="minimum",
        zorder=3,
    )
    axis.set_xticks(np.arange(2), groups)
    axis.set_ylabel("wall seconds per full 10.63M-ray traversal")
    axis.set_title("D  Complete logical-call cost", loc="left")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    gate = summary["resource_gate"]
    axis.text(
        0.98,
        0.78,
        (
            f"pair: {gate['full_forward_adjoint_pair_wall_seconds']:.1f} s\n"
            f"peak RSS: {gate['process_max_rss_bytes'] / 1024**3:.2f} GiB\n"
            f"server needed: "
            f"{'yes' if gate['server_required_for_current_16_cubed_gate'] else 'no'}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#4d5662",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d7dfdc",
            "alpha": 0.92,
        },
    )

    figure.suptitle(
        "PSU B0 streamed CGLS gate: real support measurements, no 3D ground truth",
        fontsize=13,
        fontweight="bold",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, options in (
        (".png", {"dpi": 220}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        if suffix == ".svg":
            _normalize_svg(path)
        outputs[path.name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    plt.close(figure)
    return {
        "schema_version": FIGURE_SCHEMA,
        "status": "FIGURE_COMPLETE_SUPPORT_FIT_NO_FIELD_TRUTH",
        "source_status": summary["status"],
        "source_ray_count": summary["selection"]["total_ray_count"],
        "output_files": outputs,
        "claim_boundary": {
            "slices_are_experimental_ground_truth": False,
            "support_reprojection_is_heldout_generalization": False,
            "algorithm_superiority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    summary = _load_json(args.summary)
    volume = np.load(args.volume, allow_pickle=False)
    manifest = render_figure(summary, volume, args.output_stem)
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
