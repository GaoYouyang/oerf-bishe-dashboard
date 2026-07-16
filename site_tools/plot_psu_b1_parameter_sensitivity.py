#!/usr/bin/env python3
"""Plot the public PSU B1 axis, angle, and vertex sensitivity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


SCHEMA = "psu-b1-parameter-sensitivity-public-summary-1.0"
STATUS = "B1_PARAMETER_DEPENDENCE_QUANTIFIED_PHYSICAL_SELECTION_REQUIRED"
MANIFEST_SCHEMA = "psu-b1-parameter-sensitivity-figure-1.0"
MANIFEST_STATUS = "FIGURE_COMPLETE_PARAMETER_SELECTION_AND_SUPERIORITY_LOCKED"
DEFAULT_STEM = "psu_b1_parameter_sensitivity_figure"
CAPTION = (
    "Predeclared real nine-view computational-support sensitivity. Axis sign, "
    "angle, and 5 mm vertex stress tests are not parameter optimization or "
    "calibration uncertainty. No reconstruction or superiority claim."
)
COLORS = {
    "reference": "#202124",
    "axis_semantics": "#D55E00",
    "angle": "#0072B2",
    "vertex": "#009E73",
    "grid": "#D9DDE1",
    "muted": "#5F6368",
}
LABELS = {
    "released_reference": "01  released 25 deg",
    "axis_sign_flip": "02  axis sign flip",
    "angle_minus_10deg": "03  angle 15 deg",
    "angle_minus_5deg": "04  angle 20 deg",
    "angle_plus_5deg": "05  angle 30 deg",
    "angle_plus_10deg": "06  angle 35 deg",
    "vertex_axis_plus_5mm": "07  vertex axis +5 mm",
    "vertex_axis_minus_5mm": "08  vertex axis -5 mm",
    "vertex_xy_normal_plus_5mm": "09  vertex normal +5 mm",
    "vertex_xy_normal_minus_5mm": "10  vertex normal -5 mm",
    "vertex_z_plus_5mm": "11  vertex z +5 mm",
    "vertex_z_minus_5mm": "12  vertex z -5 mm",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _fraction(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be a fraction")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be a fraction") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be in [0, 1]")
    return result


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid public summary: {path}") from exc
    summary = _mapping(value, "summary")
    if summary.get("schema_version") != SCHEMA:
        raise ValueError("unsupported public summary schema")
    if summary.get("status") != STATUS:
        raise ValueError("public summary is not a complete locked audit")
    return summary


def _extract(summary: Mapping[str, Any]) -> dict[str, Any]:
    variants = [
        _mapping(value, f"aggregate_variants[{index}]")
        for index, value in enumerate(
            _array(summary.get("aggregate_variants"), "aggregate_variants")
        )
    ]
    if len(variants) != 12:
        raise ValueError("expected the 12 frozen sensitivity variants")
    ids = [str(variant.get("id", "")) for variant in variants]
    if set(ids) != set(LABELS):
        raise ValueError("variant ids do not match the frozen plotting contract")
    families = [str(variant.get("family", "")) for variant in variants]
    active_hit = []
    active_iou = []
    all_path = []
    for index, variant in enumerate(variants):
        scopes = _mapping(variant.get("scopes"), f"variant[{index}].scopes")
        active = _mapping(scopes.get("active"), f"variant[{index}].active")
        all_scope = _mapping(scopes.get("all"), f"variant[{index}].all")
        active_hit.append(
            _fraction(
                active.get("candidate_hit_fraction"),
                f"variant[{index}].active_hit",
            )
        )
        active_iou.append(
            _fraction(
                active.get("ray_support_length_iou"),
                f"variant[{index}].active_iou",
            )
        )
        all_path.append(
            _fraction(
                all_scope.get("candidate_path_fraction_of_b0"),
                f"variant[{index}].all_path",
            )
        )

    views = [
        _mapping(value, f"per_view[{index}]")
        for index, value in enumerate(_array(summary.get("per_view"), "per_view"))
    ]
    if len(views) != 9:
        raise ValueError("expected nine real views")
    heatmap = []
    for view_index, view in enumerate(views):
        if int(view.get("view_id_zero_based", -1)) != view_index:
            raise ValueError("view ids must be ordered and contiguous")
        per_view_variants = [
            _mapping(value, f"per_view[{view_index}].variants[{variant_index}]")
            for variant_index, value in enumerate(
                _array(view.get("variants"), f"per_view[{view_index}].variants")
            )
        ]
        if [str(value.get("id", "")) for value in per_view_variants] != ids:
            raise ValueError("per-view variant order differs from aggregate")
        heatmap.append(
            [
                100.0
                * _fraction(
                    _mapping(value.get("active"), "active").get(
                        "candidate_hit_fraction"
                    ),
                    "per-view active hit",
                )
                for value in per_view_variants
            ]
        )
    return {
        "ids": ids,
        "families": families,
        "active_hit": active_hit,
        "active_iou": active_iou,
        "all_path": all_path,
        "heatmap": heatmap,
    }


def _style_axis(axis, panel: str, title: str) -> None:
    axis.set_title(f"{panel}  {title}", loc="left", fontsize=11, fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    axis.tick_params(labelsize=8.2)


def _bar_panel(axis, values, colors, labels, panel, title, ylabel) -> None:
    x = list(range(len(values)))
    axis.bar(x, [100.0 * value for value in values], color=colors, width=0.78)
    axis.set_xticks(x, labels)
    axis.set_xlabel("frozen variant index (matches panel D)")
    axis.set_ylabel(ylabel)
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    _style_axis(axis, panel, title)


def plot_public_summary(
    summary_path: Path,
    output_dir: Path,
    *,
    output_stem: str = DEFAULT_STEM,
) -> dict[str, Any]:
    summary = _read(summary_path)
    values = _extract(summary)
    labels = [f"{index:02d}" for index in range(1, len(values["ids"]) + 1)]
    heatmap_labels = [LABELS[variant_id] for variant_id in values["ids"]]
    colors = [COLORS[family] for family in values["families"]]

    figure, axes = plt.subplots(2, 2, figsize=(15.2, 9.4), constrained_layout=True)
    figure.suptitle(
        "PSU B1 one-nappe computational-support parameter sensitivity",
        fontsize=15,
        fontweight="bold",
        color=COLORS["reference"],
    )
    _bar_panel(
        axes[0, 0],
        values["active_hit"],
        colors,
        labels,
        "A",
        "Active centerlines retained",
        "active hit fraction",
    )
    _bar_panel(
        axes[0, 1],
        values["active_iou"],
        colors,
        labels,
        "B",
        "Active ray-support overlap with released reference",
        "support-length IoU",
    )
    _bar_panel(
        axes[1, 0],
        values["all_path"],
        colors,
        labels,
        "C",
        "All-ray candidate path relative to B0",
        "candidate path / B0 path",
    )

    axis = axes[1, 1]
    cmap = LinearSegmentedColormap.from_list(
        "support",
        ["#D55E00", "#F5C04A", "#F7F7F7", "#56B4E9", "#0072B2"],
    )
    image = axis.imshow(
        list(map(list, zip(*values["heatmap"]))),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=100.0,
    )
    axis.set_xticks(range(9), [str(value) for value in range(9)])
    axis.set_yticks(range(len(heatmap_labels)), heatmap_labels)
    axis.set_xlabel("real view id")
    axis.set_title(
        "D  Active hit fraction by view and variant",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    axis.tick_params(labelsize=8.0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    colorbar.set_label("active hit fraction (%)")

    figure.text(
        0.01,
        0.002,
        "Released cone parameters are a computational-domain hypothesis. "
        "Black: reference; orange: axis falsifier; blue: angle; green: vertex. "
        "The 5 mm shifts are coarse stress tests.",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        temporary_paths = {
            extension: temporary_dir / f"{output_stem}.{extension}"
            for extension in ("png", "pdf", "svg")
        }
        figure.savefig(
            temporary_paths["png"],
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            temporary_paths["pdf"],
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "OERF PSU B1 sensitivity public audit"},
        )
        figure.savefig(
            temporary_paths["svg"],
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "OERF PSU B1 sensitivity public audit"},
        )
        outputs = {}
        for extension, temporary_path in temporary_paths.items():
            final_path = output_dir / temporary_path.name
            os.replace(temporary_path, final_path)
            outputs[extension] = {
                "filename": final_path.name,
                "sha256": _sha256(final_path),
                "bytes": final_path.stat().st_size,
            }
    plt.close(figure)

    headline = _mapping(summary.get("headline_metrics"), "headline_metrics")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "summary_schema_version": SCHEMA,
        "summary_sha256": _sha256(summary_path),
        "caption": CAPTION,
        "panels": {
            "A": "active centerline hit fraction",
            "B": "active ray-support length IoU versus released reference",
            "C": "candidate path length as a fraction of B0 path",
            "D": "active hit fraction by real view and frozen variant",
        },
        "claim_boundary": {
            "parameter_optimization": False,
            "calibration_uncertainty": False,
            "physical_cone_validation": False,
            "reconstruction_result": False,
            "held_out_validation": False,
            "algorithm_superiority": "LOCKED",
        },
        "headline_metrics": dict(headline),
        "outputs": outputs,
    }
    manifest_path = output_dir / f"{output_stem}_manifest.json"
    partial = manifest_path.with_name(f".{manifest_path.name}.partial")
    partial.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output-stem", default=DEFAULT_STEM)
    args = parser.parse_args()
    manifest = plot_public_summary(
        args.summary,
        args.output_dir,
        output_stem=args.output_stem,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
