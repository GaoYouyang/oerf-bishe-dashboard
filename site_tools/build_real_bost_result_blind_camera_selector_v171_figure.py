#!/usr/bin/env python3
"""Build the public v171 result-blind camera-selector verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_result_blind_camera_selector_v171_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_result_blind_camera_selector_v171.png"
WIDTH = 2520
HEIGHT = 1320


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
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


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_strata"]
    controls = payload["controls"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f7f6")
    draw = ImageDraw.Draw(image)
    _text(draw, (WIDTH / 2, 62), "v171 result-blind geometry selector", size=45, fill="#17232d", bold=True, anchor="mm")
    _text(
        draw,
        (WIDTH / 2, 116),
        "leave-one-complete-calibration-out | geometry only | 357 parameters | independent recomputation",
        size=24,
        fill="#596772",
        anchor="mm",
    )

    left_box = (90, 180, 910, 1035)
    right_box = (970, 180, 2430, 1035)
    draw.rounded_rectangle(left_box, radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    draw.rounded_rectangle(right_box, radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    _text(draw, (500, 225), "A. Strict-safe held-out calibrations", size=28, fill="#20313d", bold=True, anchor="mm")
    _text(draw, (1700, 225), "B. Five-camera gradient p90", size=28, fill="#20313d", bold=True, anchor="mm")

    bars = [
        ("Gram-ridge", payload["selector"]["strict_local_safe_count"], "#257a68"),
        ("Fit-static", controls["fit_static"]["strict_local_safe_count"], "#d28b2f"),
        ("v169 geometry", controls["v169_fixed_geometry"]["strict_local_safe_count"], "#c45142"),
    ]
    x0, chart_bottom, max_height = 185, 875, 505
    bar_w, gap = 170, 70
    for index, (label, value, color) in enumerate(bars):
        x = x0 + index * (bar_w + gap)
        height = max_height * value / 13
        y = chart_bottom - height
        draw.rectangle((x, y, x + bar_w, chart_bottom), fill=color)
        _text(draw, (x + bar_w / 2, y - 26), f"{value}/13", size=30, fill=color, bold=True, anchor="ms")
        _text(draw, (x + bar_w / 2, chart_bottom + 46), label, size=18, fill="#42545f", bold=True, anchor="mm")
    _text(draw, (500, 965), "Complete gate: primary PASS | both controls FAIL", size=20, fill="#53636d", anchor="mm")

    chart_left, chart_top = 1090, 315
    chart_width, chart_height = 1200, 570
    gate = 0.75
    y_max = 1.0
    gate_y = chart_top + chart_height * (1 - gate / y_max)
    draw.line((chart_left, gate_y, chart_left + chart_width, gate_y), fill="#a83e35", width=4)
    _text(draw, (1700, 270), "red line: frozen gradient p90 gate 0.750", size=18, fill="#a83e35", bold=True, anchor="mm")

    times = [row["time"] for row in primary]
    series = [
        ("Gram-ridge", payload["comparison"]["gram_ridge_gradient_p90_by_time"], "#257a68"),
        ("Fit-static", payload["comparison"]["fit_static_gradient_p90_by_time"], "#d28b2f"),
        ("v169 geometry", payload["comparison"]["v169_fixed_geometry_gradient_p90_by_time"], "#c45142"),
    ]
    group_w, bar_w, bar_gap = 250, 55, 17
    for group_index, time_value in enumerate(times):
        base_x = chart_left + 85 + group_index * (group_w + 38)
        for series_index, (_label, values, color) in enumerate(series):
            value = values[group_index]
            x = base_x + series_index * (bar_w + bar_gap)
            y = chart_top + chart_height * (1 - value / y_max)
            draw.rectangle((x, y, x + bar_w, chart_top + chart_height), fill=color)
            _text(draw, (x + bar_w / 2, y - 14), f"{value:.3f}", size=15, fill=color, bold=True, anchor="ms")
        _text(draw, (base_x + 3 * bar_w / 2 + bar_gap, chart_top + chart_height + 36), f"t={time_value:.2f}", size=18, fill="#4a5c68", anchor="mm")

    legend_x = 1160
    for label, _values, color in series:
        draw.rectangle((legend_x, 942, legend_x + 34, 972), fill=color)
        _text(draw, (legend_x + 48, 957), label, size=17, fill="#4a5c68", anchor="lm")
        legend_x += 360

    draw.rounded_rectangle((90, 1085, 2430, 1250), radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    _text(
        draw,
        (1260, 1122),
        "Independent verdict: result-blind geometry selection passes 13/13 held-out calibrations and 4/4 time strata.",
        size=25,
        fill="#236b58",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1168),
        "t=0.75 gradient p90 / worst: 0.630384 / 0.692196 | independent checks: 21/21",
        size=21,
        fill="#5d6b75",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1213),
        "Post-open development evidence | fresh/external=false | real_bost=false | algorithm_breakthrough=false",
        size=20,
        fill="#5d6b75",
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
