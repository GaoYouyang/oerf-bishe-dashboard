#!/usr/bin/env python3
"""Plot the public PSU B2 support and B3 mask-policy sensitivity audit."""

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
from matplotlib.ticker import FuncFormatter  # noqa: E402


SCHEMA = "psu-bost-b3-policy-public-summary-1.0"
STATUS = "B3_POLICY_SENSITIVITY_COMPLETE_HELD_OUT_SELECTION_REQUIRED"
MANIFEST_SCHEMA = "psu-bost-b3-policy-figure-1.0"
DEFAULT_STEM = "psu_b3_policy_sensitivity_figure"
CAPTION = (
    "Deterministic discrete aperture designs (8/16/32 points) are not nested; "
    "no confidence intervals apply. Support and row-selection diagnostics only: "
    "no reconstruction, held-out validation, or algorithm superiority."
)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#5F6368",
    "ink": "#202124",
    "grid": "#D9DDE1",
}


def _sha256_file(path: Path) -> str:
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


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be numeric") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must be a fraction in [0, 1]")
    return result


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _read_summary(path: Path) -> Mapping[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid summary JSON: {path}") from exc
    summary = _mapping(summary, "summary")
    if summary.get("schema_version") != SCHEMA:
        raise ValueError("summary has an unsupported schema")
    if summary.get("status") != STATUS:
        raise ValueError("summary is not a complete B3 policy audit")
    return summary


def _category(
    item: Mapping[str, Any],
    domain: str,
    category: str,
    location: str,
) -> Mapping[str, Any]:
    domains = _mapping(item.get("domains"), f"{location}.domains")
    domain_value = _mapping(domains.get(domain), f"{location}.domains.{domain}")
    return _mapping(
        domain_value.get(category),
        f"{location}.domains.{domain}.{category}",
    )


def _policy(
    category: Mapping[str, Any],
    name: str,
    location: str,
) -> Mapping[str, Any]:
    policies = _mapping(category.get("policies"), f"{location}.policies")
    return _mapping(policies.get(name), f"{location}.policies.{name}")


def _validate_and_extract(summary: Mapping[str, Any]) -> dict[str, Any]:
    items_raw = _array(summary.get("sensitivity"), "summary.sensitivity")
    if len(items_raw) != 3:
        raise ValueError("summary.sensitivity must contain QMC 8/16/32")
    items = [
        _mapping(value, f"summary.sensitivity[{index}]")
        for index, value in enumerate(items_raw)
    ]
    sample_counts = [
        _integer(
            item.get("sample_count_per_centerline_hit"),
            f"summary.sensitivity[{index}].sample_count",
        )
        for index, item in enumerate(items)
    ]
    if sample_counts != [8, 16, 32]:
        raise ValueError("sample counts must be exactly [8, 16, 32]")

    b2_active_b0 = []
    b2_active_b1 = []
    active_excluded = {
        "support_floor_0.875": [],
        "support_floor_0.9375": [],
        "drop_any_out": [],
    }
    inactive_kept = {
        "indicator_keep": [],
        "support_floor_0.875": [],
        "support_floor_0.9375": [],
        "drop_any_out": [],
    }
    for index, item in enumerate(items):
        location = f"summary.sensitivity[{index}]"
        active_b0 = _category(item, "B0", "active", location)
        active_b1 = _category(item, "B1", "active", location)
        inactive_b1 = _category(item, "B1", "inactive", location)
        b2_active_b0.append(
            _number(
                active_b0.get("b2_fixed_denominator_retained_sample_fraction"),
                f"{location}.B0.active.weight",
            )
        )
        b2_active_b1.append(
            _number(
                active_b1.get("b2_fixed_denominator_retained_sample_fraction"),
                f"{location}.B1.active.weight",
            )
        )
        active_hits = _integer(
            active_b1.get("centerline_hit_count"),
            f"{location}.B1.active.centerline_hit_count",
        )
        for policy_name in active_excluded:
            policy = _policy(active_b1, policy_name, f"{location}.B1.active")
            kept = _integer(
                policy.get("kept_count"),
                f"{location}.B1.active.{policy_name}.kept_count",
            )
            if kept > active_hits:
                raise ValueError("active B1 kept count exceeds centerline hits")
            active_excluded[policy_name].append(active_hits - kept)
        for policy_name in inactive_kept:
            policy = _policy(inactive_b1, policy_name, f"{location}.B1.inactive")
            inactive_kept[policy_name].append(
                100.0
                * _number(
                    policy.get("kept_fraction_of_centerline_hits"),
                    f"{location}.B1.inactive.{policy_name}.kept_fraction",
                )
            )

    qmc32 = items[-1]
    view_records = _array(
        qmc32.get("active_b1_per_view"),
        "summary.sensitivity[2].active_b1_per_view",
    )
    view_ids = []
    qmc32_view_excluded = {
        "support_floor_0.9375": [],
        "drop_any_out": [],
    }
    for index, raw_view in enumerate(view_records):
        view = _mapping(raw_view, f"active_b1_per_view[{index}]")
        view_id = _integer(
            view.get("view_id_zero_based"),
            f"active_b1_per_view[{index}].view_id",
        )
        view_ids.append(view_id)
        hits = _integer(
            view.get("centerline_hit_count"),
            f"active_b1_per_view[{index}].centerline_hit_count",
        )
        for policy_name in qmc32_view_excluded:
            kept = _integer(
                _policy(
                    view,
                    policy_name,
                    f"active_b1_per_view[{index}]",
                ).get("kept_count"),
                f"active_b1_per_view[{index}].{policy_name}.kept_count",
            )
            if kept > hits:
                raise ValueError("per-view kept count exceeds centerline hits")
            qmc32_view_excluded[policy_name].append(hits - kept)
    if view_ids != list(range(len(view_ids))):
        raise ValueError("QMC32 view ids must be ordered and contiguous")

    return {
        "sample_counts": sample_counts,
        "b2_active_b0": b2_active_b0,
        "b2_active_b1": b2_active_b1,
        "active_excluded": active_excluded,
        "inactive_kept": inactive_kept,
        "view_ids": view_ids,
        "qmc32_view_excluded": qmc32_view_excluded,
    }


def _style_axis(axis, label: str, title: str) -> None:
    axis.set_title(f"{label}  {title}", loc="left", fontsize=11, fontweight="bold")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.5)


def _save_figure(
    figure,
    *,
    output_dir: Path,
    output_stem: str,
    summary_path: Path,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
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
            metadata={"Creator": "OERF B3 public audit"},
        )
        figure.savefig(
            temporary_paths["svg"],
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "OERF B3 public audit"},
        )
        outputs = {}
        for extension, temporary_path in temporary_paths.items():
            final_path = output_dir / temporary_path.name
            os.replace(temporary_path, final_path)
            outputs[extension] = {
                "filename": final_path.name,
                "sha256": _sha256_file(final_path),
                "bytes": final_path.stat().st_size,
            }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "summary_schema_version": SCHEMA,
        "summary_sha256": _sha256_file(summary_path),
        "caption": CAPTION,
        "panels": {
            "A": "active B0/B1 fixed-denominator aperture weight retained",
            "B": "active B1 rows excluded by predeclared B3 policies",
            "C": "inactive B1 centerline hits retained by B3 policies",
            "D": "QMC32 active B1 exclusions by view",
        },
        "claim_boundary": {
            "confidence_intervals": False,
            "qmc_designs_nested": False,
            "reconstruction_result": False,
            "held_out_validation": False,
            "algorithm_superiority": "LOCKED",
        },
        "headline_metrics": metrics,
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


def plot_public_summary(
    summary_path: Path,
    output_dir: Path,
    *,
    output_stem: str = DEFAULT_STEM,
) -> dict[str, Any]:
    summary = _read_summary(summary_path)
    values = _validate_and_extract(summary)
    samples = values["sample_counts"]

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    figure.suptitle(
        "PSU finite-aperture support and B3 mask-policy sensitivity",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )

    axis = axes[0, 0]
    axis.plot(
        samples,
        [100.0 * value for value in values["b2_active_b0"]],
        marker="o",
        color=COLORS["blue"],
        label="B0 active",
    )
    axis.plot(
        samples,
        [100.0 * value for value in values["b2_active_b1"]],
        marker="s",
        color=COLORS["orange"],
        label="B1 active",
    )
    _style_axis(axis, "A", "Fixed-denominator aperture weight")
    axis.set_xticks(samples)
    axis.set_xlabel("samples per centerline hit")
    axis.set_ylabel("retained weight (%)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
    axis.legend(frameon=False, fontsize=8.5, loc="lower left")

    axis = axes[0, 1]
    active_styles = (
        ("support_floor_0.875", "floor 87.5%", COLORS["green"], "o"),
        ("support_floor_0.9375", "floor 93.75%", COLORS["purple"], "s"),
        ("drop_any_out", "drop any OOD", COLORS["vermillion"], "^"),
    )
    for policy, label, color, marker in active_styles:
        axis.plot(
            samples,
            values["active_excluded"][policy],
            marker=marker,
            color=color,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(samples)
    axis.set_xlabel("samples per centerline hit")
    axis.set_ylabel("excluded active B1 rays (log)")
    _style_axis(axis, "B", "Whole-ray policy amplification")
    axis.legend(frameon=False, fontsize=8.2, loc="upper left")
    axis.text(
        0.98,
        0.06,
        "indicator_keep / drop_empty: 0 active exclusions",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COLORS["gray"],
    )

    axis = axes[1, 0]
    inactive_styles = (
        ("indicator_keep", "indicator keep", COLORS["blue"], "o"),
        ("support_floor_0.875", "floor 87.5%", COLORS["green"], "s"),
        ("support_floor_0.9375", "floor 93.75%", COLORS["purple"], "^"),
        ("drop_any_out", "drop any OOD", COLORS["vermillion"], "D"),
    )
    for policy, label, color, marker in inactive_styles:
        axis.plot(
            samples,
            values["inactive_kept"][policy],
            marker=marker,
            color=color,
            label=label,
        )
    axis.set_xticks(samples)
    axis.set_xlabel("samples per centerline hit")
    axis.set_ylabel("inactive B1 hits kept (%)")
    axis.set_ylim(60, 101)
    _style_axis(axis, "C", "Inactive support is policy-sensitive")
    axis.legend(frameon=False, fontsize=7.8, loc="lower left", ncol=2)

    axis = axes[1, 1]
    view_ids = values["view_ids"]
    x = list(range(len(view_ids)))
    width = 0.38
    floor_values = values["qmc32_view_excluded"]["support_floor_0.9375"]
    strict_values = values["qmc32_view_excluded"]["drop_any_out"]
    axis.bar(
        [value - width / 2 for value in x],
        floor_values,
        width,
        color=COLORS["purple"],
        label="floor 93.75%",
    )
    axis.bar(
        [value + width / 2 for value in x],
        strict_values,
        width,
        color=COLORS["vermillion"],
        label="drop any OOD",
    )
    axis.set_xticks(x, [str(value) for value in view_ids])
    axis.set_xlabel("view id (zero-based)")
    axis.set_ylabel("excluded active B1 rays")
    _style_axis(axis, "D", "QMC32 exclusions localize by view")
    axis.legend(frameon=False, fontsize=8.2, loc="upper left")

    figure.text(
        0.5,
        0.002,
        CAPTION,
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLORS["gray"],
    )
    manifest = _save_figure(
        figure,
        output_dir=output_dir,
        output_stem=output_stem,
        summary_path=summary_path,
        metrics=_mapping(summary.get("headline_metrics"), "headline_metrics"),
    )
    plt.close(figure)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default=DEFAULT_STEM)
    args = parser.parse_args()
    manifest = plot_public_summary(
        args.summary_json,
        args.output_dir,
        output_stem=args.output_stem,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
