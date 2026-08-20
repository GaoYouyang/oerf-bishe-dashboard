#!/usr/bin/env python3
"""Build the public v173 selected-camera warm-refinement verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_camera_selected_warm_refinement_v173_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_camera_selected_warm_refinement_v173.png"
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


def _horizontal_rows(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[tuple[str, float, float, str, str]],
    *,
    start_y: int,
    step: int,
) -> None:
    left, _top, right, _bottom = box
    chart_left = left + 78
    chart_right = right - 55
    for index, (label, value, total, color, display) in enumerate(rows):
        y = start_y + index * step
        _text(draw, (chart_left, y), label, size=19, fill="#42545f", bold=True, anchor="lm")
        _text(draw, (chart_right, y), display, size=21, fill=color, bold=True, anchor="rm")
        track_y = y + 28
        draw.rectangle((chart_left, track_y, chart_right, track_y + 36), fill="#e7ecef")
        width = (chart_right - chart_left) * min(max(value / total, 0.0), 1.0)
        draw.rectangle((chart_left, track_y, chart_left + width, track_y + 36), fill=color)


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary_h1_k1"]
    k0 = payload["blocking_h1_k0"]
    controls = payload["controls"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 58),
        "v173 selected-camera warm refinement",
        size=46,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 114),
        "same five-camera subset | H1-K1 versus cheaper H1-K0 | independent recomputation",
        size=23,
        fill="#596772",
        anchor="mm",
    )

    left_box = (70, 175, 835, 1025)
    mid_box = (875, 175, 1560, 1025)
    right_box = (1600, 175, 2450, 1025)
    _panel(draw, left_box, "A. Strict-safe cells")
    _panel(draw, mid_box, "B. Exact forward + adjoint calls")
    _panel(draw, right_box, "C. Global p90 relative error")

    _horizontal_rows(
        draw,
        left_box,
        [
            ("Selected H1-K1", 468, 468, "#267a68", "468/468"),
            ("Selected H1-K0", 468, 468, "#2d6fa3", "468/468"),
            (
                "Fit-static H1-K1",
                controls["fit_static_h1_k1"]["strict_safe_cells"],
                468,
                "#d28b2f",
                "334/468",
            ),
            (
                "v169 fixed H1-K1",
                controls["v169_fixed_h1_k1"]["strict_safe_cells"],
                468,
                "#9b6cab",
                "222/468",
            ),
            ("Zero-K4 reference", 0, 468, "#c45142", "0/468"),
        ],
        start_y=305,
        step=132,
    )

    primary_calls = primary["exact_forward_calls"] + primary["exact_adjoint_calls"]
    k0_calls = k0["exact_forward_calls"] + k0["exact_adjoint_calls"]
    reference_calls = (
        controls["zero_k4"]["exact_forward_calls"]
        + controls["zero_k4"]["exact_adjoint_calls"]
    )
    _horizontal_rows(
        draw,
        mid_box,
        [
            ("Selected H1-K1", primary_calls, 8, "#267a68", "2A + 2A^T"),
            ("Selected H1-K0", k0_calls, 8, "#2d6fa3", "1A + 1A^T"),
            ("Zero-K4", reference_calls, 8, "#c45142", "4A + 4A^T"),
        ],
        start_y=350,
        step=205,
    )
    _text(
        draw,
        ((mid_box[0] + mid_box[2]) / 2, 930),
        "H1-K0 passes every gate with half the primary exact-call ledger.",
        size=18,
        fill="#4f5f68",
        anchor="mm",
    )

    metric_rows = [
        ("Field", "field_p90_higher", 0.50),
        ("Gradient", "gradient_p90_higher", 0.75),
        ("Observation", "observation_p90_higher", 0.20),
    ]
    chart_left = right_box[0] + 90
    chart_right = right_box[2] - 60
    y = 315
    for label, key, gate in metric_rows:
        primary_value = primary["global_metrics"][key]
        k0_value = k0["global_metrics"][key]
        _text(draw, (chart_left, y), label, size=21, fill="#344853", bold=True, anchor="lm")
        _text(
            draw,
            (chart_right, y),
            f"gate {gate:.2f}",
            size=18,
            fill="#a83e35",
            bold=True,
            anchor="rm",
        )
        for offset, method, value, color in [
            (42, "K1", primary_value, "#267a68"),
            (100, "K0", k0_value, "#2d6fa3"),
        ]:
            bar_y = y + offset
            _text(draw, (chart_left, bar_y + 17), method, size=17, fill=color, bold=True, anchor="lm")
            bar_left = chart_left + 58
            draw.rectangle((bar_left, bar_y, chart_right, bar_y + 34), fill="#e7ecef")
            width = (chart_right - bar_left) * min(value, 1.0)
            draw.rectangle((bar_left, bar_y, bar_left + width, bar_y + 34), fill=color)
            gate_x = bar_left + (chart_right - bar_left) * gate
            draw.line((gate_x, bar_y - 5, gate_x, bar_y + 39), fill="#a83e35", width=3)
            _text(
                draw,
                (chart_right - 8, bar_y + 17),
                f"{value:.3f}",
                size=17,
                fill="#ffffff" if width > 95 else color,
                bold=True,
                anchor="rm",
            )
        y += 220

    draw.rounded_rectangle((70, 1080, 2450, 1250), radius=8, fill="#ffffff", outline="#ced8dc", width=2)
    _text(
        draw,
        (1260, 1120),
        "Independent verdict: both H1-K1 and H1-K0 pass 468/468, 13/13 calibrations, 9/9 fields, and 4/4 times",
        size=25,
        fill="#245f83",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1173),
        "The extra CGLS K1 step has no established advantage; the next gate isolates camera-selector value at identical H1-K0 cost.",
        size=21,
        fill="#4f5f68",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1218),
        "FAIL_CLASSICAL_CONTROL_EXPLAINS_CAMERA_SELECTED_WARM_V173 | algorithm_breakthrough=false",
        size=20,
        fill="#a83e35",
        bold=True,
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
