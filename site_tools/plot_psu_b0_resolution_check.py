#!/usr/bin/env python3
"""Render the same-rays, same-calls PSU 16³-to-32³ resolution check."""

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


FIGURE_SCHEMA = "psu-b0-streaming-resolution-figure-1.0"


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


def _volume(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 5:
        array = array[0, 0]
    if array.ndim != 3 or not np.all(np.isfinite(array)):
        raise ValueError("volume must contain one finite 3D field")
    return array


def _triptych(
    axis: plt.Axes,
    volume: np.ndarray,
    *,
    title: str,
    scale: float,
) -> None:
    nz, ny, nx = volume.shape
    slices = (
        volume[nz // 2],
        volume[:, ny // 2, :],
        volume[:, :, nx // 2],
    )
    labels = ("z mid", "y mid", "x mid")
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    axis.set_axis_off()
    image = None
    for left, label, values in zip(
        (0.0, 0.34, 0.68),
        labels,
        slices,
        strict=True,
    ):
        inset = axis.inset_axes([left, 0.13, 0.29, 0.72])
        image = inset.imshow(
            values,
            origin="lower",
            cmap="coolwarm",
            norm=norm,
            interpolation="nearest",
        )
        inset.set_title(label, fontsize=8)
        inset.set_xticks([])
        inset.set_yticks([])
    axis.text(
        0.0,
        0.99,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    colorbar_axis = axis.inset_axes([0.12, 0.01, 0.76, 0.045])
    colorbar = axis.figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.ax.tick_params(labelsize=7)


def render_figure(
    low_summary: dict[str, Any],
    high_summary: dict[str, Any],
    comparison: dict[str, Any],
    low_volume: np.ndarray,
    high_volume: np.ndarray,
    output_stem: Path,
) -> dict[str, Any]:
    low_values = _volume(low_volume)
    high_values = _volume(high_volume)
    scale = float(
        max(np.max(np.abs(low_values)), np.max(np.abs(high_values)), 1e-12)
    )
    low_history = low_summary["optimization"]["history"]
    high_history = high_summary["optimization"]["history"]
    iterations = np.asarray(
        [row["iteration"] for row in low_history],
        dtype=int,
    )
    low_residual = np.asarray(
        [row["relative_measurement_l2"] for row in low_history],
        dtype=float,
    )
    high_residual = np.asarray(
        [row["relative_measurement_l2"] for row in high_history],
        dtype=float,
    )
    rows = comparison["per_view"]
    view_ids = np.asarray([row["view_id_zero_based"] for row in rows], dtype=int)
    view_low = np.asarray([row["residual_16_cubed"] for row in rows], dtype=float)
    view_high = np.asarray([row["residual_32_cubed"] for row in rows], dtype=float)
    resources = comparison["resources"]

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
        figsize=(11.4, 7.3),
        constrained_layout=True,
    )
    _triptych(
        axes[0, 0],
        low_values,
        title="A  16³ support-fit field",
        scale=scale,
    )
    _triptych(
        axes[0, 1],
        high_values,
        title="B  32³ support-fit field",
        scale=scale,
    )

    axis = axes[1, 0]
    axis.plot(
        iterations,
        low_residual,
        marker="o",
        linewidth=2,
        color="#3d6d9c",
        label="16³",
    )
    axis.plot(
        iterations,
        high_residual,
        marker="s",
        linewidth=2,
        color="#14756a",
        label="32³",
    )
    axis.set_xticks(iterations)
    axis.set_xlabel("fixed CGLS iteration")
    axis.set_ylabel("relative measurement L2")
    axis.set_title("C  Same rays, same calls", loc="left")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)

    axis = axes[1, 1]
    width = 0.36
    axis.bar(
        view_ids - width / 2,
        view_low,
        width=width,
        color="#6b8fb5",
        label="16³",
    )
    axis.bar(
        view_ids + width / 2,
        view_high,
        width=width,
        color="#55a092",
        label="32³",
    )
    axis.set_xticks(view_ids)
    axis.set_xlabel("support view id")
    axis.set_ylabel("direct relative measurement L2")
    axis.set_title("D  All nine support views improve", loc="left")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    aggregate = comparison["aggregate"]
    axis.text(
        0.98,
        0.96,
        (
            f"absolute drop: {aggregate['absolute_drop']:.4f}\n"
            f"relative improvement: "
            f"{100 * aggregate['relative_improvement_fraction']:.2f}%\n"
            f"pair time: "
            f"{resources['pair_wall_seconds_16_cubed']:.1f} / "
            f"{resources['pair_wall_seconds_32_cubed']:.1f} s\n"
            f"server needed: "
            f"{'yes' if resources['server_required_for_32_cubed_gate'] else 'no'}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#3f4c52",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d7dfdc",
            "alpha": 0.94,
        },
    )
    figure.suptitle(
        (
            "PSU B0 resolution gate: full support, QMC16, float64, "
            "fixed 4-step CGLS\n"
            "Support reprojection only; no experimental 3D ground truth"
        ),
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
        "status": "FIGURE_COMPLETE_RESOLUTION_SIGNAL_NO_FIELD_TRUTH",
        "comparison_status": comparison["status"],
        "output_files": outputs,
        "claim_boundary": {
            "support_fit_is_field_truth": False,
            "resolution_signal_is_algorithm_superiority": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-summary", type=Path, required=True)
    parser.add_argument("--high-summary", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--low-volume", type=Path, required=True)
    parser.add_argument("--high-volume", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = render_figure(
        _load_json(args.low_summary),
        _load_json(args.high_summary),
        _load_json(args.comparison),
        np.load(args.low_volume, allow_pickle=False),
        np.load(args.high_volume, allow_pickle=False),
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
