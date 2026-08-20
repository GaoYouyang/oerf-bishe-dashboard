#!/usr/bin/env python3
"""Build the public v172 whole-field/time isolation verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "docs/real_bost_whole_field_time_isolated_selector_v172_public_summary.json"
)
OUTPUT = ROOT / "assets/figures/real_bost_whole_field_time_isolated_selector_v172.png"
WIDTH = 2520
HEIGHT = 1320


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    *,
    size: int,
    fill: str,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=_font(size, bold=bold), fill=fill, anchor=anchor)


def _bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    rows: list[tuple[str, int, int, str]],
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#d0d9de", width=2)
    _text(draw, ((left + right) / 2, top + 48), title, size=27, fill="#20313d", bold=True, anchor="mm")
    chart_left = left + 95
    chart_right = right - 55
    y = top + 150
    for label, value, total, color in rows:
        _text(draw, (chart_left, y), label, size=20, fill="#42545f", bold=True, anchor="lm")
        track_y = y + 35
        draw.rectangle((chart_left, track_y, chart_right, track_y + 48), fill="#e7ecef")
        width = (chart_right - chart_left) * value / total
        draw.rectangle((chart_left, track_y, chart_left + width, track_y + 48), fill=color)
        _text(
            draw,
            (chart_right, y),
            f"{value}/{total}",
            size=22,
            fill=color,
            bold=True,
            anchor="rm",
        )
        y += 155


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    controls = payload["controls"]
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f3f6f5")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 60),
        "v172 whole-field and time isolation",
        size=46,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 116),
        "held-out calibration x complete field x time | reported geometry only | independent recomputation",
        size=23,
        fill="#596772",
        anchor="mm",
    )

    _bar_panel(
        draw,
        (80, 175, 790, 1025),
        "A. Strict-safe held-out cells",
        [
            ("Triple-isolated ridge", 468, 468, "#257a68"),
            ("Fit-static", controls["fit_static"]["strict_cell_safe_count"], 468, "#d28b2f"),
            (
                "v169 geometry",
                controls["v169_fixed_geometry"]["strict_cell_safe_count"],
                468,
                "#c45142",
            ),
        ],
    )
    _bar_panel(
        draw,
        (850, 175, 1510, 1025),
        "B. Complete fields passed",
        [
            ("Triple-isolated ridge", 9, 9, "#257a68"),
            ("Fit-static", controls["fit_static"]["whole_field_models_safe_count"], 9, "#d28b2f"),
            (
                "v169 geometry",
                controls["v169_fixed_geometry"]["whole_field_models_safe_count"],
                9,
                "#c45142",
            ),
        ],
    )

    right_box = (1570, 175, 2440, 1025)
    draw.rounded_rectangle(right_box, radius=8, fill="#ffffff", outline="#d0d9de", width=2)
    _text(draw, (2005, 223), "C. Gradient p90 by time", size=27, fill="#20313d", bold=True, anchor="mm")
    chart_left, chart_top, chart_width, chart_height = 1645, 330, 720, 500
    gate = payload["comparison"]["gradient_p90_gate"]
    gate_y = chart_top + chart_height * (1 - gate)
    draw.line((chart_left, gate_y, chart_left + chart_width, gate_y), fill="#a83e35", width=4)
    _text(
        draw,
        (chart_left + chart_width - 6, gate_y - 22),
        "gate 0.750",
        size=17,
        fill="#a83e35",
        bold=True,
        anchor="rs",
    )
    series = [
        ("Primary", payload["comparison"]["primary_gradient_p90_by_time"], "#257a68"),
        ("Fit-static", payload["comparison"]["fit_static_gradient_p90_by_time"], "#d28b2f"),
        ("v169", payload["comparison"]["v169_fixed_geometry_gradient_p90_by_time"], "#c45142"),
    ]
    times = [row["time"] for row in payload["primary_strata"]]
    group_width = 165
    bar_width = 36
    gap = 12
    for time_index, time_value in enumerate(times):
        group_x = chart_left + 35 + time_index * (group_width + 12)
        for series_index, (_label, values, color) in enumerate(series):
            value = values[time_index]
            x = group_x + series_index * (bar_width + gap)
            y = chart_top + chart_height * (1 - value)
            draw.rectangle((x, y, x + bar_width, chart_top + chart_height), fill=color)
        _text(
            draw,
            (group_x + 72, chart_top + chart_height + 34),
            f"t={time_value:.2f}",
            size=17,
            fill="#4a5c68",
            anchor="mm",
        )
        _text(
            draw,
            (group_x + 72, chart_top - 24),
            f"{series[0][1][time_index]:.3f}",
            size=17,
            fill="#257a68",
            bold=True,
            anchor="mm",
        )
    legend_x = 1655
    for label, _values, color in series:
        draw.rectangle((legend_x, 915, legend_x + 28, 941), fill=color)
        _text(draw, (legend_x + 39, 928), label, size=17, fill="#4a5c68", anchor="lm")
        legend_x += 235

    draw.rounded_rectangle((80, 1080, 2440, 1250), radius=8, fill="#ffffff", outline="#d0d9de", width=2)
    _text(
        draw,
        (1260, 1120),
        "Independent verdict: 468/468 cells | 13/13 calibrations | 9/9 fields | 4/4 times | 22/22 checks",
        size=26,
        fill="#236b58",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1175),
        "Post-open controlled-proxy headroom. Camera selection only; full warm start, fresh wall/RSS, external data, and real BOST remain unproven.",
        size=20,
        fill="#5d6870",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1218),
        "algorithm_breakthrough=false",
        size=21,
        fill="#a83e35",
        bold=True,
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
