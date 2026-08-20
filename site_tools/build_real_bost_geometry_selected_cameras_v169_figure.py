#!/usr/bin/env python3
"""Build the public v169 geometry-only camera-selection verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_geometry_selected_cameras_v169_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_geometry_selected_cameras_v169.png"
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
    rows = {(row["time"], row["camera_count"]): row for row in payload["primary"]["strata"]}
    h1 = payload["controls"]["frozen_isotropic_h1_on_previous_fixed_subset"][
        "five_camera_gradient_p90_by_time"
    ]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f5f7f8")
    draw = ImageDraw.Draw(image)
    _text(draw, (WIDTH / 2, 62), "v169 geometry-only camera selection", size=45, fill="#17232d", bold=True, anchor="mm")
    _text(
        draw,
        (WIDTH / 2, 116),
        "low-frequency H1-whitened observability | independently recomputed verdict",
        size=25,
        fill="#596772",
        anchor="mm",
    )

    matrix_box = (90, 180, 1300, 1035)
    chart_box = (1360, 180, 2430, 1035)
    draw.rounded_rectangle(matrix_box, radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    draw.rounded_rectangle(chart_box, radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    _text(draw, (695, 225), "A. Frozen strata verdict", size=29, fill="#20313d", bold=True, anchor="mm")
    _text(draw, (1895, 225), "B. Five-camera gradient p90", size=29, fill="#20313d", bold=True, anchor="mm")

    left, top = 245, 315
    cell_w, cell_h = 315, 132
    gap_x, gap_y = 25, 31
    for column, camera_count in enumerate([5, 7, 9]):
        _text(
            draw,
            (left + column * (cell_w + gap_x) + cell_w / 2, top - 38),
            f"{camera_count} cameras",
            size=22,
            fill="#3d505e",
            bold=True,
            anchor="mm",
        )
    for row_index, time_value in enumerate([0.0, 0.25, 0.75, 1.0]):
        y0 = top + row_index * (cell_h + gap_y)
        _text(draw, (left - 26, y0 + cell_h / 2), f"t={time_value:.2f}", size=21, fill="#3d505e", bold=True, anchor="rm")
        for column, camera_count in enumerate([5, 7, 9]):
            item = rows[(time_value, camera_count)]
            x0 = left + column * (cell_w + gap_x)
            passed = item["passed"]
            fill = "#e8f5ef" if passed else "#fff0e3"
            outline = "#2f8069" if passed else "#c85332"
            draw.rounded_rectangle((x0, y0, x0 + cell_w, y0 + cell_h), radius=8, fill=fill, outline=outline, width=3)
            _text(draw, (x0 + 20, y0 + 18), "PASS" if passed else "FAIL", size=19, fill=outline, bold=True)
            _text(draw, (x0 + cell_w / 2, y0 + 71), f"{item['gradient_p90']:.3f}", size=34, fill="#20313d", bold=True, anchor="mm")
            _text(draw, (x0 + cell_w / 2, y0 + 108), f"worst {item['gradient_worst']:.3f}", size=17, fill="#65737d", anchor="mm")

    chart_left, chart_top = 1490, 315
    chart_width, chart_height = 820, 570
    gate = 0.75
    y_max = 1.0
    gate_y = chart_top + chart_height * (1 - gate / y_max)
    draw.line((chart_left, gate_y, chart_left + chart_width, gate_y), fill="#b34535", width=4)
    _text(draw, (chart_left + chart_width, gate_y - 12), "gate 0.750", size=18, fill="#b34535", bold=True, anchor="rs")
    times = [0.0, 0.25, 0.75, 1.0]
    bar_w, pair_gap, group_gap = 64, 22, 88
    x = chart_left + 58
    for idx, time_value in enumerate(times):
        selected = rows[(time_value, 5)]["gradient_p90"]
        for value, color, label_offset in [(selected, "#c85332", 0), (h1[idx], "#3174a8", bar_w + pair_gap)]:
            x0 = x + label_offset
            y0 = chart_top + chart_height * (1 - value / y_max)
            draw.rectangle((x0, y0, x0 + bar_w, chart_top + chart_height), fill=color)
            _text(draw, (x0 + bar_w / 2, y0 - 18), f"{value:.3f}", size=17, fill=color, bold=True, anchor="ms")
        _text(draw, (x + bar_w + pair_gap / 2, chart_top + chart_height + 34), f"t={time_value:.2f}", size=18, fill="#4a5c68", anchor="mm")
        x += 2 * bar_w + pair_gap + group_gap
    draw.rectangle((1540, 938, 1574, 968), fill="#c85332")
    _text(draw, (1590, 953), "geometry-selected", size=19, fill="#4a5c68", anchor="lm")
    draw.rectangle((1910, 938, 1944, 968), fill="#3174a8")
    _text(draw, (1960, 953), "frozen H1 roster", size=19, fill="#4a5c68", anchor="lm")

    draw.rounded_rectangle((90, 1085, 2430, 1250), radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    _text(
        draw,
        (1260, 1124),
        "Independent verdict: 8/12 pass; every five-camera gradient-p90 stratum fails.",
        size=27,
        fill="#7b3f2d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1170),
        "Selector changed the physical five-camera roster in 13/13 geometries | 27/27 checks | selection +0A/+0AT after cache",
        size=21,
        fill="#5d6b75",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1213),
        "Controlled virtual proxy only | predictor=false | real_bost=false | algorithm_breakthrough=false",
        size=20,
        fill="#5d6b75",
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
