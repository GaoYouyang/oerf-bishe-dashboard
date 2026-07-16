#!/usr/bin/env python3
"""Render the PSU compact-cache numerical and performance gate."""

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


FIGURE_SCHEMA = "psu-b0-compact-cache-figure-1.0"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def render_figure(
    benchmark: dict[str, Any],
    replay: dict[str, Any],
    output_stem: Path,
) -> dict[str, Any]:
    if not benchmark["status"].endswith("PASS"):
        raise ValueError("benchmark must pass before rendering")
    if not replay["status"].endswith("PASS"):
        raise ValueError("cached replay must pass before rendering")
    performance = benchmark["performance"]
    numerical = benchmark["numerical_equivalence"]
    replay_numerical = replay["numerical_equivalence"]
    replay_performance = replay["performance"]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 8.2),
        constrained_layout=True,
    )
    colors = {"direct": "#577590", "cached": "#43aa8b"}

    axis = axes[0, 0]
    direct = np.asarray(performance["direct_pair_seconds"], dtype=np.float64)
    cached = np.asarray(performance["cached_pair_seconds"], dtype=np.float64)
    for index, (name, values) in enumerate((("direct", direct), ("cached", cached))):
        jitter = np.linspace(-0.06, 0.06, len(values))
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=62,
            color=colors[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.hlines(
            np.median(values),
            index - 0.2,
            index + 0.2,
            color="#222222",
            linewidth=2,
        )
    axis.set_xticks((0, 1), ("rebuild geometry", "compact cache"))
    axis.set_ylabel("seconds per complete F + Aᵀ")
    axis.set_title("A  Same-session full-pair timing", loc="left")
    axis.grid(axis="y", alpha=0.22)
    axis.text(
        0.98,
        0.94,
        f"median speedup\n{performance['median_pair_speedup']:.3f}×",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d7dfdc",
            "alpha": 0.94,
        },
    )

    axis = axes[0, 1]
    optimization = np.asarray(
        [
            replay_performance["reference_optimization_wall_seconds"],
            replay_performance["cached_optimization_wall_seconds"],
        ],
        dtype=np.float64,
    )
    bars = axis.bar(
        (0, 1),
        optimization,
        color=(colors["direct"], colors["cached"]),
        width=0.58,
    )
    axis.bar_label(bars, fmt="%.1f s", padding=4, fontsize=9)
    axis.set_xticks((0, 1), ("direct reference", "cached replay"))
    axis.set_ylabel("fixed 4-step CGLS wall seconds")
    axis.set_title("B  Frozen 4F + 5Aᵀ optimization", loc="left")
    axis.grid(axis="y", alpha=0.22)
    axis.text(
        0.98,
        0.94,
        f"speedup\n{replay_performance['optimization_speedup']:.3f}×",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d7dfdc",
            "alpha": 0.94,
        },
    )

    axis = axes[1, 0]
    labels = ("forward", "adjoint", "volume", "support residual")
    raw_values = np.asarray(
        [
            numerical["forward_relative_difference"],
            numerical["adjoint_relative_difference"],
            replay_numerical["volume_relative_difference"],
            replay_numerical["support_residual_absolute_difference"],
        ],
        dtype=np.float64,
    )
    plotted = np.maximum(raw_values, 1e-18)
    bars = axis.bar(
        np.arange(len(labels)),
        plotted,
        color=("#277da1", "#4d908e", "#90be6d", "#f9c74f"),
        width=0.62,
    )
    axis.set_yscale("log")
    axis.axhline(1e-14, color="#c8553d", linestyle="--", linewidth=1.2)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("relative or absolute difference")
    axis.set_title("C  Direct and cached numerical equivalence", loc="left")
    axis.grid(axis="y", which="both", alpha=0.2)
    for bar, raw in zip(bars, raw_values, strict=True):
        label = "0" if raw == 0 else f"{raw:.2e}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.35,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis = axes[1, 1]
    axis.axis("off")
    cache = benchmark["cache"]
    configuration = benchmark["configuration"]
    lines = (
        ("10,628,822", "support rays"),
        (f"{configuration['sample_count']} per ray", "QMC aperture samples"),
        (f"{cache['total_cache_bytes'] / 1e9:.3f} GB", "private cache"),
        (f"{cache['build_wall_seconds']:.2f} s", "one-time cache build"),
        (
            f"{replay_numerical['cached_support_relative_measurement_l2']:.6f}",
            "same support residual",
        ),
    )
    axis.set_title("D  What the cache changes", loc="left")
    y = 0.9
    for value, label in lines:
        axis.text(0.03, y, value, fontsize=17, fontweight="bold", color="#25313c")
        axis.text(0.46, y + 0.01, label, fontsize=10, color="#59636d")
        y -= 0.145
    axis.text(
        0.03,
        0.08,
        (
            "Execution is faster; the discrete operator, call budget, support "
            "fit and evidence level are unchanged.\n"
            "Rotation-40 development and final audit remain sealed. "
            "No experimental 3D ground truth."
        ),
        fontsize=9,
        color="#4f5963",
        va="bottom",
        wrap=True,
    )

    figure.suptitle(
        "PSU B0 compact stencil cache: exact replay with lower operator cost",
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
        "status": "FIGURE_COMPLETE_EXACT_CACHE_REPLAY_NO_FIELD_TRUTH",
        "benchmark_status": benchmark["status"],
        "replay_status": replay["status"],
        "output_files": outputs,
        "claim_boundary": {
            "cache_is_algorithm_superiority": False,
            "support_replay_is_heldout_generalization": False,
            "experimental_field_truth_available": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = render_figure(
        _load_json(args.benchmark),
        _load_json(args.replay),
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
