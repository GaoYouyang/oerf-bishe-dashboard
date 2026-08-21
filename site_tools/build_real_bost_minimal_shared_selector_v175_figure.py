#!/usr/bin/env python3
"""Build the public v175 minimal shared selector figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_minimal_shared_selector_v175_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_minimal_shared_selector_v175.png"
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


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#ced8dc", width=2)
    _text(
        draw,
        ((box[0] + box[2]) / 2, box[1] + 48),
        title,
        size=27,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    policies = payload["policies"]
    order = ["minimal_shared_gram_ridge", "ray_axis_maximin", "fit_static", "v169_low_mode_d_opt"]
    labels = {
        "minimal_shared_gram_ridge": "Minimal shared selector",
        "ray_axis_maximin": "Ray-axis maximin",
        "fit_static": "Fit-static",
        "v169_low_mode_d_opt": "v169 low-mode D-opt",
    }
    colors = {
        "minimal_shared_gram_ridge": "#267a68",
        "ray_axis_maximin": "#2d6fa3",
        "fit_static": "#d28b2f",
        "v169_low_mode_d_opt": "#9b6cab",
    }

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 58),
        "v175 minimal shared CPU camera selector",
        size=46,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 114),
        "117 calibration x field outer folds | <=357 parameters | one subset across four times",
        size=23,
        fill="#596772",
        anchor="mm",
    )

    left_box = (70, 175, 835, 1025)
    mid_box = (875, 175, 1630, 1025)
    right_box = (1670, 175, 2450, 1025)
    _panel(draw, left_box, "A. Strict-safe cells")
    _panel(draw, mid_box, "B. Complete axes passed")
    _panel(draw, right_box, "C. Global p90 relative error")

    chart_left = left_box[0] + 70
    chart_right = left_box[2] - 45
    for index, key in enumerate(order):
        item = policies[key]
        y = 305 + index * 165
        color = colors[key]
        _text(draw, (chart_left, y), labels[key], size=19, fill="#42545f", bold=True, anchor="lm")
        _text(
            draw,
            (chart_right, y),
            f'{item["strict_safe_cells"]}/468',
            size=21,
            fill=color,
            bold=True,
            anchor="rm",
        )
        draw.rectangle((chart_left, y + 30, chart_right, y + 70), fill="#e7ecef")
        width = (chart_right - chart_left) * item["strict_safe_cells"] / 468
        draw.rectangle((chart_left, y + 30, chart_left + width, y + 70), fill=color)

    axis_specs = [
        ("Calibrations", "whole_calibrations_passed", 13),
        ("3D fields", "whole_field_models_passed", 9),
        ("Times", "time_strata_passed", 4),
    ]
    chart_left = mid_box[0] + 70
    chart_right = mid_box[2] - 45
    y = 275
    for axis_label, field, total in axis_specs:
        _text(draw, (chart_left, y), axis_label, size=21, fill="#344853", bold=True, anchor="lm")
        y += 38
        for key in order:
            item = policies[key]
            color = colors[key]
            value = item[field]
            _text(draw, (chart_left, y + 17), labels[key], size=16, fill=color, bold=True, anchor="lm")
            bar_left = chart_left + 205
            draw.rectangle((bar_left, y, chart_right, y + 34), fill="#e7ecef")
            width = (chart_right - bar_left) * value / total
            draw.rectangle((bar_left, y, bar_left + width, y + 34), fill=color)
            _text(
                draw,
                (chart_right - 8, y + 17),
                f"{value}/{total}",
                size=16,
                fill="#ffffff" if width > 75 else color,
                bold=True,
                anchor="rm",
            )
            y += 40
        y += 25

    metrics = [
        ("Field", "field_p90_higher", 0.50),
        ("Gradient", "gradient_p90_higher", 0.75),
        ("Observation", "observation_p90_higher", 0.20),
    ]
    chart_left = right_box[0] + 75
    chart_right = right_box[2] - 45
    y = 270
    for metric_label, field, gate in metrics:
        _text(draw, (chart_left, y), metric_label, size=20, fill="#344853", bold=True, anchor="lm")
        _text(draw, (chart_right, y), f"gate {gate:.2f}", size=17, fill="#a83e35", bold=True, anchor="rm")
        y += 38
        for key in order:
            value = policies[key]["global_metrics"][field]
            color = colors[key]
            _text(draw, (chart_left, y + 15), labels[key], size=15, fill=color, bold=True, anchor="lm")
            bar_left = chart_left + 188
            draw.rectangle((bar_left, y, chart_right, y + 30), fill="#e7ecef")
            width = (chart_right - bar_left) * min(value, 1.0)
            draw.rectangle((bar_left, y, bar_left + width, y + 30), fill=color)
            gate_x = bar_left + (chart_right - bar_left) * gate
            draw.line((gate_x, y - 3, gate_x, y + 33), fill="#a83e35", width=3)
            _text(draw, (chart_right - 7, y + 15), f"{value:.3f}", size=15, fill="#20313d", bold=True, anchor="rm")
            y += 42
        y += 34

    draw.rounded_rectangle((70, 1080, 2450, 1250), radius=8, fill="#ffffff", outline="#ced8dc", width=2)
    _text(
        draw,
        (1260, 1120),
        "Independent verdict: the shared geometry-risk model is the only complete equal-cost policy",
        size=25,
        fill="#245f83",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1173),
        "468/468 cells | 13/13 calibrations | 9/9 fields | 4/4 times | 31/31 independent checks",
        size=21,
        fill="#4f5f68",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1218),
        "PASS_MINIMAL_SHARED_SELECTOR_HEADROOM_V175 | algorithm_breakthrough=false",
        size=20,
        fill="#267a68",
        bold=True,
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
