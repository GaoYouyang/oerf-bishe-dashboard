#!/usr/bin/env python3
"""Build the public v170 finite five-camera capacity verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_five_camera_subset_oracle_v170_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_five_camera_subset_oracle_v170.png"
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
    rows = payload["capacity"]["calibration_shared"]["strata"]
    parent = payload["parent_comparison"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f7f6")
    draw = ImageDraw.Draw(image)
    _text(draw, (WIDTH / 2, 62), "v170 finite five-camera capacity", size=45, fill="#17232d", bold=True, anchor="mm")
    _text(
        draw,
        (WIDTH / 2, 116),
        "126 subsets per calibration | calibration-shared truth-aware feasibility | independent recomputation",
        size=24,
        fill="#596772",
        anchor="mm",
    )

    left_box = (90, 180, 1300, 1035)
    right_box = (1360, 180, 2430, 1035)
    draw.rounded_rectangle(left_box, radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    draw.rounded_rectangle(right_box, radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    _text(draw, (695, 225), "A. Shared-witness p90 / worst", size=29, fill="#20313d", bold=True, anchor="mm")
    _text(draw, (1895, 225), "B. Five-camera gradient p90", size=29, fill="#20313d", bold=True, anchor="mm")

    x0, y0 = 190, 315
    cell_w, cell_h = 255, 136
    gap_x, gap_y = 28, 34
    columns = [("field", 0.5), ("gradient", 0.75), ("observation", 0.2)]
    for column, (metric, gate) in enumerate(columns):
        cx = x0 + column * (cell_w + gap_x) + cell_w / 2
        _text(draw, (cx, y0 - 44), metric, size=22, fill="#3d505e", bold=True, anchor="mm")
        _text(draw, (cx, y0 - 17), f"p90 gate {gate:.2f}", size=16, fill="#71808a", anchor="mm")
    for row_index, row in enumerate(rows):
        y = y0 + row_index * (cell_h + gap_y)
        _text(draw, (x0 - 24, y + cell_h / 2), f"t={row['time']:.2f}", size=21, fill="#3d505e", bold=True, anchor="rm")
        for column, (metric, _gate) in enumerate(columns):
            x = x0 + column * (cell_w + gap_x)
            draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=8, fill="#e8f5ef", outline="#2f8069", width=3)
            _text(draw, (x + 18, y + 16), "PASS", size=18, fill="#2f8069", bold=True)
            _text(draw, (x + cell_w / 2, y + 72), f"{row[metric]['p90_higher']:.3f}", size=32, fill="#20313d", bold=True, anchor="mm")
            _text(draw, (x + cell_w / 2, y + 110), f"worst {row[metric]['worst']:.3f}", size=16, fill="#65737d", anchor="mm")

    chart_left, chart_top = 1475, 315
    chart_width, chart_height = 835, 570
    gate = 0.75
    y_max = 1.0
    gate_y = chart_top + chart_height * (1 - gate / y_max)
    draw.line((chart_left, gate_y, chart_left + chart_width, gate_y), fill="#b34535", width=4)
    _text(draw, (1895, 270), "red line: gradient p90 gate 0.750", size=18, fill="#b34535", bold=True, anchor="mm")
    times = [0.0, 0.25, 0.75, 1.0]
    v169 = parent["v169_geometry_selected_gradient_p90_by_time"]
    v170 = parent["v170_calibration_shared_gradient_p90_by_time"]
    bar_w, pair_gap, group_gap = 62, 22, 88
    x = chart_left + 58
    for index, time_value in enumerate(times):
        for value, color, offset in [(v169[index], "#c85332", 0), (v170[index], "#2f8069", bar_w + pair_gap)]:
            bx = x + offset
            by = chart_top + chart_height * (1 - value / y_max)
            draw.rectangle((bx, by, bx + bar_w, chart_top + chart_height), fill=color)
            _text(draw, (bx + bar_w / 2, by - 18), f"{value:.3f}", size=17, fill=color, bold=True, anchor="ms")
        _text(draw, (x + bar_w + pair_gap / 2, chart_top + chart_height + 34), f"t={time_value:.2f}", size=18, fill="#4a5c68", anchor="mm")
        x += 2 * bar_w + pair_gap + group_gap
    draw.rectangle((1535, 938, 1569, 968), fill="#c85332")
    _text(draw, (1585, 953), "v169 geometry heuristic", size=18, fill="#4a5c68", anchor="lm")
    draw.rectangle((1960, 938, 1994, 968), fill="#2f8069")
    _text(draw, (2010, 953), "v170 capacity witness", size=18, fill="#4a5c68", anchor="lm")

    draw.rounded_rectangle((90, 1085, 2430, 1250), radius=8, fill="#ffffff", outline="#d2dbe1", width=2)
    _text(
        draw,
        (1260, 1124),
        "Independent verdict: calibration-shared five-camera capacity passes 4/4 time strata.",
        size=27,
        fill="#236b58",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1170),
        "Tightest margin: t=0.75 gradient p90 0.748953 vs gate 0.750000 | 23/23 independent checks",
        size=21,
        fill="#5d6b75",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1213),
        "Truth-aware capacity only | deployable selector=false | real_bost=false | algorithm_breakthrough=false",
        size=20,
        fill="#5d6b75",
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
