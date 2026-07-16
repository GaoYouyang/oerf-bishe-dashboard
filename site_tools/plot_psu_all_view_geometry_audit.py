#!/usr/bin/env python3
"""Build the publication figure for the PSU all-view geometry audit."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402


REPORT_SCHEMA = "psu-bost-all-view-geometry-audit-1.0"
MANIFEST_SCHEMA = "psu-bost-all-view-geometry-figure-1.0"
DEFAULT_OUTPUT_STEM = "psu_all_view_geometry_audit_figure"
FIGURE_SIZE_INCHES = (11.0, 7.4)
PNG_DPI = 300
_OUTPUT_STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PLOT_METRICS = (
    "cone_length_weighted_outside_box_fraction",
    "full_box_zero_fraction",
    "box_miss_but_cone_nonzero_fraction",
    "final_zero_length_fraction",
    "active_rms_magnitude_pixels",
    "inactive_rms_magnitude_pixels",
    "active_unsafe_geometry_fraction",
    "inactive_unsafe_geometry_fraction",
)
FRACTION_METRICS = frozenset(
    {
        "cone_length_weighted_outside_box_fraction",
        "full_box_zero_fraction",
        "box_miss_but_cone_nonzero_fraction",
        "final_zero_length_fraction",
        "active_unsafe_geometry_fraction",
        "inactive_unsafe_geometry_fraction",
    }
)
RMS_METRICS = frozenset(
    {"active_rms_magnitude_pixels", "inactive_rms_magnitude_pixels"}
)

# Okabe-Ito colors, paired with markers or hatches so color is never the only cue.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "gray": "#5F6368",
    "ink": "#202124",
    "grid": "#D9DDE1",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read report JSON {path}: {exc}") from exc
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"report JSON is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError("report JSON schema error: root must be an object")
    return report, payload


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read metrics CSV {path}: {exc}") from exc
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"metrics CSV is not valid UTF-8: {path}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("metrics CSV schema error: header row is required")
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("metrics CSV schema error: duplicate column names")
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"metrics CSV parse error: {exc}") from exc
    return rows, fieldnames, payload


def _require_numeric(
    value: Any, *, location: str, metric: str, fraction: bool = False
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{location}.{metric} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}.{metric} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{location}.{metric} must be a finite number")
    if fraction and not 0.0 <= result <= 1.0:
        raise ValueError(f"{location}.{metric} must be within [0, 1]")
    if metric in RMS_METRICS and result < 0.0:
        raise ValueError(f"{location}.{metric} must be non-negative")
    return result


def _require_view_id(value: Any, *, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{location}.view_id_zero_based must be a non-negative integer"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{location}.view_id_zero_based must be a non-negative integer"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(
            f"{location}.view_id_zero_based must be a non-negative integer"
        )
    return int(numeric)


def _normalise_record(record: Any, *, location: str) -> dict[str, float | int]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{location} must be an object")
    required = ("view_id_zero_based", *PLOT_METRICS)
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f"{location} missing required field(s): {', '.join(sorted(missing))}"
        )
    normalised: dict[str, float | int] = {
        "view_id_zero_based": _require_view_id(
            record["view_id_zero_based"], location=location
        )
    }
    for metric in PLOT_METRICS:
        normalised[metric] = _require_numeric(
            record[metric],
            location=location,
            metric=metric,
            fraction=metric in FRACTION_METRICS,
        )
    return normalised


def load_plot_records(
    report_json: Path, metrics_csv: Path
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Load, validate, and cross-check the JSON and CSV figure inputs."""

    report_json = Path(report_json)
    metrics_csv = Path(metrics_csv)
    report, report_bytes = _read_json(report_json)
    rows, fieldnames, csv_bytes = _read_csv(metrics_csv)

    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(
            f"report JSON schema error: schema_version must be {REPORT_SCHEMA!r}"
        )
    view_count = report.get("view_count")
    if (
        isinstance(view_count, bool)
        or not isinstance(view_count, int)
        or view_count < 1
    ):
        raise ValueError(
            "report JSON schema error: view_count must be a positive integer"
        )
    json_views = report.get("views")
    if not isinstance(json_views, list):
        raise ValueError("report JSON schema error: views must be an array")
    if len(json_views) != view_count:
        raise ValueError(
            "report JSON schema error: view_count does not match the views array"
        )

    required_csv_fields = {"view_id_zero_based", *PLOT_METRICS}
    missing_csv_fields = sorted(required_csv_fields.difference(fieldnames))
    if missing_csv_fields:
        raise ValueError(
            "metrics CSV schema error: missing required column(s): "
            + ", ".join(missing_csv_fields)
        )
    if len(rows) != view_count:
        raise ValueError(
            "metrics CSV schema error: row count does not match JSON view_count"
        )

    json_records = [
        _normalise_record(record, location=f"report.views[{index}]")
        for index, record in enumerate(json_views)
    ]
    csv_records = [
        _normalise_record(record, location=f"metrics_csv.row[{index + 2}]")
        for index, record in enumerate(rows)
    ]

    def by_id(
        records: Sequence[dict[str, float | int]], *, source: str
    ) -> dict[int, dict[str, float | int]]:
        indexed: dict[int, dict[str, float | int]] = {}
        for record in records:
            view_id = int(record["view_id_zero_based"])
            if view_id in indexed:
                raise ValueError(f"{source} schema error: duplicate view id {view_id}")
            indexed[view_id] = record
        return indexed

    json_by_id = by_id(json_records, source="report JSON")
    csv_by_id = by_id(csv_records, source="metrics CSV")
    if set(json_by_id) != set(csv_by_id):
        raise ValueError("JSON/CSV mismatch: view id sets differ")

    for view_id in sorted(json_by_id):
        for metric in PLOT_METRICS:
            json_value = float(json_by_id[view_id][metric])
            csv_value = float(csv_by_id[view_id][metric])
            if not math.isclose(json_value, csv_value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "JSON/CSV mismatch: "
                    f"view {view_id} field {metric} differs "
                    f"({json_value!r} != {csv_value!r})"
                )

    records = [csv_by_id[view_id] for view_id in sorted(csv_by_id)]
    source_sha256 = _sha256_bytes(report_bytes + b"\0" + csv_bytes)
    provenance = {
        "source_sha256": source_sha256,
        "source_sha256_definition": "sha256(report_json_bytes + NUL + metrics_csv_bytes)",
        "sources": {
            "report_json": {
                "filename": report_json.name,
                "sha256": _sha256_bytes(report_bytes),
            },
            "metrics_csv": {
                "filename": metrics_csv.name,
                "sha256": _sha256_bytes(csv_bytes),
            },
        },
    }
    return records, provenance


def _tick_positions(view_count: int, maximum_ticks: int = 12) -> list[int]:
    if view_count <= maximum_ticks:
        return list(range(view_count))
    positions = {
        round(index * (view_count - 1) / (maximum_ticks - 1))
        for index in range(maximum_ticks)
    }
    return sorted(positions)


def _nice_upper(maximum: float, *, fallback: float) -> float:
    if maximum <= 0.0:
        return fallback
    target = maximum * 1.12
    magnitude = 10.0 ** math.floor(math.log10(target))
    for multiplier in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        candidate = multiplier * magnitude
        if candidate >= target:
            return candidate
    raise AssertionError("nice-axis bound search failed")


def _trimmed_decimal(value: float, decimals: int) -> str:
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _percentage_formatter(upper: float) -> FuncFormatter:
    if upper <= 0.1:
        decimals = 3
    elif upper <= 1.0:
        decimals = 2
    elif upper <= 10.0:
        decimals = 1
    else:
        decimals = 0
    return FuncFormatter(lambda value, _position: _trimmed_decimal(value, decimals))


def _style_axis(ax: Axes, view_ids: Sequence[int], *, y_grid: bool = True) -> None:
    tick_positions = _tick_positions(len(view_ids))
    ax.set_xticks(tick_positions, [str(view_ids[index]) for index in tick_positions])
    ax.set_xlim(-0.55, len(view_ids) - 0.45)
    ax.set_xlabel("View (zero-based)")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.7, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    if y_grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)


def _panel_heading(ax: Axes, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )
    ax.text(
        0.062,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def _set_percentage_axis(ax: Axes, series: Sequence[Sequence[float]]) -> None:
    maximum = max((max(values, default=0.0) for values in series), default=0.0)
    upper = _nice_upper(maximum, fallback=1.0)
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=6, min_n_ticks=4, steps=[1.0, 2.0, 2.5, 5.0, 10.0])
    )
    ax.yaxis.set_major_formatter(_percentage_formatter(upper))


def _render_figure(records: Sequence[dict[str, float | int]]) -> Figure:
    view_ids = [int(record["view_id_zero_based"]) for record in records]
    x = list(range(len(records)))

    percentages = {
        metric: [100.0 * float(record[metric]) for record in records]
        for metric in FRACTION_METRICS
    }
    rms = {
        metric: [float(record[metric]) for record in records] for metric in RMS_METRICS
    }

    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCHES, squeeze=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(
        left=0.078,
        right=0.985,
        bottom=0.105,
        top=0.95,
        wspace=0.235,
        hspace=0.36,
    )
    no_go_positions = [
        index
        for index, record in enumerate(records)
        if any(
            float(record[metric]) > 0
            for metric in (
                "cone_length_weighted_outside_box_fraction",
                "full_box_zero_fraction",
                "final_zero_length_fraction",
                "inactive_unsafe_geometry_fraction",
            )
        )
    ]
    for axis in (ax_a, ax_b, ax_c, ax_d):
        for position in no_go_positions:
            axis.axvspan(
                position - 0.48,
                position + 0.48,
                color="#ECEFF1",
                alpha=0.58,
                linewidth=0,
                zorder=0,
            )

    a_values = percentages["cone_length_weighted_outside_box_fraction"]
    ax_a.bar(
        x,
        a_values,
        width=0.68,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.45,
    )
    _style_axis(ax_a, view_ids)
    _set_percentage_axis(ax_a, [a_values])
    ax_a.set_ylabel("Cone length outside box (%)")
    _panel_heading(ax_a, "A", "Outside-box cone length fraction")

    b_specs = (
        ("full_box_zero_fraction", "Full-box zero", COLORS["blue"], "///"),
        (
            "box_miss_but_cone_nonzero_fraction",
            "Box miss, cone nonzero",
            COLORS["orange"],
            "\\\\",
        ),
        ("final_zero_length_fraction", "Final zero", COLORS["green"], ".."),
    )
    bar_width = 0.24
    b_series: list[list[float]] = []
    for offset, (metric, label, color, hatch) in zip(
        (-bar_width, 0.0, bar_width), b_specs
    ):
        values = percentages[metric]
        b_series.append(values)
        ax_b.bar(
            [position + offset for position in x],
            values,
            width=bar_width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            hatch=hatch,
        )
    _style_axis(ax_b, view_ids)
    _set_percentage_axis(ax_b, b_series)
    ax_b.set_ylabel("Rays (%)")
    ax_b.legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.45)
    _panel_heading(ax_b, "B", "Zero-length ray diagnostics")

    c_specs = (
        ("active_rms_magnitude_pixels", "Active", COLORS["vermillion"], "o", "-"),
        ("inactive_rms_magnitude_pixels", "Inactive", COLORS["blue"], "s", "--"),
    )
    c_series: list[list[float]] = []
    for metric, label, color, marker, line_style in c_specs:
        values = rms[metric]
        c_series.append(values)
        ax_c.plot(
            x,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=line_style,
            linewidth=1.8,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
        )
    _style_axis(ax_c, view_ids)
    c_upper = _nice_upper(
        max((max(values, default=0.0) for values in c_series), default=0.0),
        fallback=1.0,
    )
    ax_c.set_ylim(0.0, c_upper)
    ax_c.set_ylabel("RMS displacement (pixels)")
    ax_c.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.2)
    _panel_heading(ax_c, "C", "Active and inactive RMS displacement")

    d_specs = (
        (
            "active_unsafe_geometry_fraction",
            "Active",
            COLORS["vermillion"],
            "///",
        ),
        (
            "inactive_unsafe_geometry_fraction",
            "Inactive",
            COLORS["blue"],
            "\\\\",
        ),
    )
    d_width = 0.34
    d_series: list[list[float]] = []
    for offset, (metric, label, color, hatch) in zip(
        (-d_width / 2.0, d_width / 2.0), d_specs
    ):
        values = percentages[metric]
        d_series.append(values)
        ax_d.bar(
            [position + offset for position in x],
            values,
            width=d_width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            hatch=hatch,
        )
    _style_axis(ax_d, view_ids)
    _set_percentage_axis(ax_d, d_series)
    ax_d.set_ylabel("Unsafe mask samples (%)")
    ax_d.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.8)
    if all(value == 0.0 for value in d_series[0]):
        ax_d.text(
            0.985,
            0.965,
            "Active = 0% for all views",
            transform=ax_d.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color=COLORS["vermillion"],
        )
    _panel_heading(ax_d, "D", "Active and inactive unsafe geometry")

    highlighted = ", ".join(str(view_ids[index]) for index in no_go_positions)
    figure.text(
        0.5,
        0.02,
        f"Gray bands: geometry-contract NO-GO views ({highlighted}). "
        "Deterministic ray census; no statistical error bars.",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color=COLORS["gray"],
    )

    return figure


def _save_staged_outputs(figure: Figure, stage_dir: Path, stem: str) -> dict[str, Path]:
    output_paths = {
        "png": stage_dir / f"{stem}.png",
        "svg": stage_dir / f"{stem}.svg",
        "pdf": stage_dir / f"{stem}.pdf",
    }
    fixed_time = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    common_title = "PSU all-view geometry audit"
    figure.savefig(
        output_paths["png"],
        format="png",
        dpi=PNG_DPI,
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": common_title,
            "Author": "OERF PSU BOST audit",
            "Software": "site_tools/plot_psu_all_view_geometry_audit.py",
        },
    )
    figure.savefig(
        output_paths["svg"],
        format="svg",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": common_title,
            "Creator": "site_tools/plot_psu_all_view_geometry_audit.py",
            "Description": "Four-panel all-view geometry audit figure",
            "Date": "1970-01-01T00:00:00Z",
        },
    )
    figure.savefig(
        output_paths["pdf"],
        format="pdf",
        facecolor="white",
        edgecolor="none",
        metadata={
            "Title": common_title,
            "Author": "OERF PSU BOST audit",
            "Subject": "Four-panel all-view geometry audit figure",
            "Creator": "site_tools/plot_psu_all_view_geometry_audit.py",
            "Producer": f"Matplotlib {matplotlib.__version__}",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    return output_paths


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def build_all_view_geometry_figure(
    report_json: Path,
    metrics_csv: Path,
    output_dir: Path,
    *,
    output_stem: str = DEFAULT_OUTPUT_STEM,
) -> dict[str, Any]:
    """Build PNG, SVG, PDF, and a hash manifest from the all-view report pair."""

    if not _OUTPUT_STEM_PATTERN.fullmatch(output_stem):
        raise ValueError(
            "output_stem must contain only ASCII letters, digits, dot, underscore, or hyphen"
        )
    records, provenance = load_plot_records(Path(report_json), Path(metrics_csv))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rc = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.2,
        "axes.labelsize": 9.2,
        "axes.labelcolor": COLORS["ink"],
        "axes.edgecolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.1,
        "hatch.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": MANIFEST_SCHEMA,
    }

    manifest_name = f"{output_stem}_manifest.json"
    with tempfile.TemporaryDirectory(
        prefix=f".{output_stem}.", dir=output_dir
    ) as temporary_directory:
        stage_dir = Path(temporary_directory)
        with matplotlib.rc_context(rc):
            figure = _render_figure(records)
            try:
                staged_outputs = _save_staged_outputs(figure, stage_dir, output_stem)
            finally:
                plt.close(figure)

        expected_width = round(FIGURE_SIZE_INCHES[0] * PNG_DPI)
        expected_height = round(FIGURE_SIZE_INCHES[1] * PNG_DPI)
        outputs: dict[str, dict[str, Any]] = {}
        for output_format, path in staged_outputs.items():
            _fsync_file(path)
            details: dict[str, Any] = {
                "filename": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if output_format == "png":
                details.update(
                    {
                        "dpi": PNG_DPI,
                        "width_pixels": expected_width,
                        "height_pixels": expected_height,
                    }
                )
            outputs[output_format] = details

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "FIGURE_BUILD_COMPLETE",
            **provenance,
            "data_contract": {
                "view_count": len(records),
                "view_ids_zero_based": [
                    int(record["view_id_zero_based"]) for record in records
                ],
                "plotted_metrics": list(PLOT_METRICS),
                "numeric_source": "metrics_csv_cross_checked_against_report_json_views",
            },
            "rendering": {
                "figure_size_inches": list(FIGURE_SIZE_INCHES),
                "png_dpi": PNG_DPI,
                "background": "#FFFFFF",
                "font_family": "DejaVu Sans",
                "color_palette": {
                    name: COLORS[name]
                    for name in ("blue", "orange", "green", "vermillion", "gray")
                },
                "matplotlib_version": matplotlib.__version__,
            },
            "panels": {
                "A": ["cone_length_weighted_outside_box_fraction"],
                "B": [
                    "full_box_zero_fraction",
                    "box_miss_but_cone_nonzero_fraction",
                    "final_zero_length_fraction",
                ],
                "C": [
                    "active_rms_magnitude_pixels",
                    "inactive_rms_magnitude_pixels",
                ],
                "D": [
                    "active_unsafe_geometry_fraction",
                    "inactive_unsafe_geometry_fraction",
                ],
            },
            "outputs": outputs,
        }
        manifest_path = stage_dir / manifest_name
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(manifest_path)

        for output_format in ("png", "svg", "pdf"):
            staged = staged_outputs[output_format]
            os.replace(staged, output_dir / staged.name)
        os.replace(manifest_path, output_dir / manifest_name)

    return manifest


# Small aliases keep the import surface obvious for callers that think in terms of
# either plotting or figure building.
build_figure = build_all_view_geometry_figure
plot_psu_all_view_geometry_audit = build_all_view_geometry_figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json", "--audit-json", dest="report_json", type=Path, required=True
    )
    parser.add_argument(
        "--metrics-csv", "--audit-csv", dest="metrics_csv", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", "--output-prefix", default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()
    manifest = build_all_view_geometry_figure(
        args.report_json,
        args.metrics_csv,
        args.output_dir,
        output_stem=args.output_stem,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
