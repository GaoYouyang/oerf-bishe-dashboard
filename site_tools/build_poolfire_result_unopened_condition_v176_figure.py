#!/usr/bin/env python3
"""Build the public v176 result-unopened condition figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_result_unopened_condition_v176_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_result_unopened_condition_v176.png"
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
    primary = payload["primary"]
    reference = payload["reference_adequacy_diagnostic"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 58),
        "v176 result-unopened PoolFire condition",
        size=46,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 114),
        "Frozen v175 fit | 13 calibrations x 4 frames | no refit or retuning",
        size=23,
        fill="#596772",
        anchor="mm",
    )

    left_box = (70, 175, 800, 1025)
    mid_box = (840, 175, 1695, 1025)
    right_box = (1735, 175, 2450, 1025)
    _panel(draw, left_box, "A. Frozen-gate outcome")
    _panel(draw, mid_box, "B. Global p90 relative error")
    _panel(draw, right_box, "C. Failure attribution")

    labels = [
        ("Strict-safe cells", 0, 52, "#b64d43"),
        ("Complete calibrations", 0, 13, "#b64d43"),
        ("Frame strata", 0, 4, "#b64d43"),
        ("Joint harm", 52, 52, "#d37c2f"),
        ("Severe harm", 50, 52, "#9b4f68"),
    ]
    chart_left = left_box[0] + 70
    chart_right = left_box[2] - 45
    for index, (label, value, total, color) in enumerate(labels):
        y = 300 + index * 130
        _text(draw, (chart_left, y), label, size=20, fill="#42545f", bold=True, anchor="lm")
        _text(draw, (chart_right, y), f"{value}/{total}", size=22, fill=color, bold=True, anchor="rm")
        draw.rectangle((chart_left, y + 32, chart_right, y + 70), fill="#e7ecef")
        width = (chart_right - chart_left) * value / total
        if width > 0:
            draw.rectangle((chart_left, y + 32, chart_left + width, y + 70), fill=color)

    metrics = [
        ("Field", "field_relative_l2", 0.50),
        ("Gradient", "gradient_relative_l2", 0.75),
        ("Observation", "observation_relative_l2", 0.20),
    ]
    candidate_color = "#b64d43"
    reference_color = "#2d6fa3"
    chart_left = mid_box[0] + 75
    chart_right = mid_box[2] - 50
    for index, (label, field, gate) in enumerate(metrics):
        y = 300 + index * 220
        candidate_value = primary["candidate_global_p50_p90_worst"][field][1]
        reference_value = reference["primary_k4_global_p50_p90_worst"][field][1]
        _text(draw, (chart_left, y), label, size=22, fill="#344853", bold=True, anchor="lm")
        _text(draw, (chart_right, y), f"gate {gate:.2f}", size=18, fill="#a83e35", bold=True, anchor="rm")
        for row, (row_label, value, color) in enumerate(
            [("Frozen selector", candidate_value, candidate_color), ("Same-subset K4", reference_value, reference_color)]
        ):
            bar_y = y + 48 + row * 58
            _text(draw, (chart_left, bar_y + 17), row_label, size=17, fill=color, bold=True, anchor="lm")
            bar_left = chart_left + 205
            draw.rectangle((bar_left, bar_y, chart_right, bar_y + 34), fill="#e7ecef")
            width = (chart_right - bar_left) * min(value / 1.2, 1.0)
            draw.rectangle((bar_left, bar_y, bar_left + width, bar_y + 34), fill=color)
            gate_x = bar_left + (chart_right - bar_left) * gate / 1.2
            draw.line((gate_x, bar_y - 4, gate_x, bar_y + 38), fill="#a83e35", width=3)
            _text(draw, (chart_right - 8, bar_y + 17), f"{value:.3f}", size=16, fill="#20313d", bold=True, anchor="rm")

    box_left = right_box[0] + 55
    box_right = right_box[2] - 45
    y = 285
    steps = [
        ("v175 opened development", "468/468 safe", "#267a68"),
        ("v176 frozen condition", "0/52 safe", "#b64d43"),
        ("Primary same-subset K4", "0/52 safe", "#2d6fa3"),
        ("All four K4 references", "0 strict-safe each", "#9b6cab"),
    ]
    for index, (label, value, color) in enumerate(steps):
        top = y + index * 150
        draw.rounded_rectangle((box_left, top, box_right, top + 105), radius=7, fill="#f7f9f9", outline=color, width=3)
        _text(draw, ((box_left + box_right) / 2, top + 35), label, size=19, fill="#344853", bold=True, anchor="mm")
        _text(draw, ((box_left + box_right) / 2, top + 73), value, size=23, fill=color, bold=True, anchor="mm")
        if index < len(steps) - 1:
            x = (box_left + box_right) / 2
            draw.line((x, top + 108, x, top + 142), fill="#75838b", width=3)
    _text(
        draw,
        ((box_left + box_right) / 2, 920),
        "Broader five-camera reference / representation mismatch",
        size=17,
        fill="#7e403b",
        bold=True,
        anchor="mm",
    )

    draw.rounded_rectangle((70, 1080, 2450, 1250), radius=8, fill="#ffffff", outline="#ced8dc", width=2)
    _text(
        draw,
        (1260, 1120),
        "Independent verdict: close the current minimal shared-selector transfer",
        size=25,
        fill="#7e403b",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1173),
        "0/52 cells | 0/13 calibrations | 0/4 frames | 35/35 independent checks",
        size=21,
        fill="#4f5f68",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1218),
        "FAIL_RESULT_UNOPENED_POOLFIRE_CONDITION_PARITY_V176 | algorithm_breakthrough=false",
        size=19,
        fill="#b64d43",
        bold=True,
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
