#!/usr/bin/env python3
"""Build the public v181 geometry-conditioned rank-16 verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_geometry_conditioned_rank16_inverse_v181_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_geometry_conditioned_rank16_inverse_v181.png"
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
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#c7d2d7", width=2)
    _text(draw, ((box[0] + box[2]) / 2, box[1] + 48), title, size=27, fill="#20313d", bold=True, anchor="mm")


def _coverage_row(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    index: int,
    label: str,
    value: str,
    passed: bool,
) -> None:
    y = box[1] + 128 + index * 145
    color = "#267a68" if passed else "#b64d43"
    draw.rounded_rectangle((box[0] + 42, y, box[2] - 42, y + 98), radius=7, fill="#f7f9f9", outline="#d6dfe2", width=2)
    _text(draw, (box[0] + 64, y + 30), label, size=19, fill="#42545f", bold=True, anchor="lm")
    _text(draw, (box[2] - 64, y + 30), value, size=26, fill=color, bold=True, anchor="rm")
    _text(draw, (box[0] + 64, y + 70), "PASS" if passed else "FAIL", size=16, fill=color, bold=True, anchor="lm")


def _metric_row(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    index: int,
    label: str,
    five_value: float,
    nine_value: float,
    gate: float,
) -> None:
    left = box[0] + 54
    right = box[2] - 46
    bar_left = left + 190
    y = box[1] + 132 + index * 212
    scale_max = 0.9
    _text(draw, (left, y + 22), label, size=20, fill="#435660", bold=True, anchor="lm")
    for row, (name, value, color) in enumerate((("five", five_value, "#315f91"), ("nine", nine_value, "#87631b"))):
        row_y = y + row * 62
        draw.rectangle((bar_left, row_y, right, row_y + 34), fill="#e7ecef")
        width = (right - bar_left) * min(value / scale_max, 1.0)
        draw.rectangle((bar_left, row_y, bar_left + width, row_y + 34), fill=color)
        gate_x = bar_left + (right - bar_left) * gate / scale_max
        draw.line((gate_x, row_y - 4, gate_x, row_y + 38), fill="#793f3a", width=3)
        _text(draw, (bar_left - 12, row_y + 17), name, size=16, fill="#64747c", anchor="rm")
        _text(draw, (right - 8, row_y + 17), f"{value:.3f}", size=17, fill="#20313d", bold=True, anchor="rm")
    _text(draw, (bar_left, y + 144), f"frozen p90 gate {gate:.2f}", size=15, fill="#6f7d84", anchor="lm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    five = payload["five_camera_primary_k1"]["global_p90"]
    nine = payload["all_nine_primary_k1"]["global_p90"]
    gates = payload["absolute_gate"]
    diagnostic = payload["mechanism_diagnostics"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(draw, (WIDTH / 2, 58), "v181 Explicit geometry conditioning: fixed rank 16 still fails", size=40, fill="#17232d", bold=True, anchor="mm")
    _text(draw, (WIDTH / 2, 112), "Geometry-specific Jacobi whitening + 16 spectral corrections + unchanged CGLS K1", size=22, fill="#596772", anchor="mm")

    left_box = (70, 175, 795, 1040)
    middle_box = (835, 175, 1710, 1040)
    right_box = (1750, 175, 2450, 1040)
    _panel(draw, left_box, "A. Complete-gate coverage")
    _panel(draw, middle_box, "B. Primary K1 p90")
    _panel(draw, right_box, "C. Mechanism diagnosis")

    _coverage_row(draw, box=left_box, index=0, label="v179 exact inverse, five", value="52/52", passed=True)
    _coverage_row(draw, box=left_box, index=1, label="v180 shared rank 16, five", value="4/52", passed=False)
    _coverage_row(draw, box=left_box, index=2, label="v181 geometry rank 16, five", value="0/52", passed=False)
    _coverage_row(draw, box=left_box, index=3, label="v181 geometry rank 16, nine", value="0/52", passed=False)
    _coverage_row(draw, box=left_box, index=4, label="v181 complete frames", value="0/4", passed=False)

    _metric_row(
        draw,
        box=middle_box,
        index=0,
        label="Field",
        five_value=five["field_relative_l2"],
        nine_value=nine["field_relative_l2"],
        gate=gates["field_p90_max"],
    )
    _metric_row(
        draw,
        box=middle_box,
        index=1,
        label="Gradient",
        five_value=five["gradient_relative_l2"],
        nine_value=nine["gradient_relative_l2"],
        gate=gates["gradient_p90_max"],
    )
    _metric_row(
        draw,
        box=middle_box,
        index=2,
        label="Observation",
        five_value=five["observation_relative_l2"],
        nine_value=nine["observation_relative_l2"],
        gate=gates["observation_p90_max"],
    )

    items = [
        ("Observable dimension", "1,009 coordinates"),
        ("Geometry factors", "13 calibrations x 2 sensors"),
        ("Whitened spectrum", f"{diagnostic['whitened_eigenvalue_minimum']:.3f} to {diagnostic['whitened_eigenvalue_maximum']:.3f}"),
        ("Jacobi residual p90", f"{diagnostic['jacobi_inverse_residual_p90']:.5f}"),
        ("Rank-16 residual p90", f"{diagnostic['rank16_inverse_residual_p90']:.5f}"),
        ("Relative reduction", f"{100 * diagnostic['relative_p90_reduction']:.2f}%"),
        ("Independent checks", "48/48"),
        ("Interpretation", "Broad inverse mismatch"),
    ]
    for index, (label, value) in enumerate(items):
        y = right_box[1] + 112 + index * 90
        _text(draw, (right_box[0] + 48, y), label, size=17, fill="#64747c", bold=True)
        _text(draw, (right_box[0] + 48, y + 32), value, size=20, fill="#20313d")

    draw.rounded_rectangle((70, 1085, 2450, 1255), radius=8, fill="#fff8ed", outline="#d6b66f", width=2)
    _text(draw, (WIDTH / 2, 1135), "Scientific decision: FAIL_GEOMETRY_CONDITIONED_RANK16_INVERSE_V181", size=27, fill="#8a493f", bold=True, anchor="mm")
    _text(draw, (WIDTH / 2, 1192), "Reported geometry is included, but sixteen corrected modes remove too little of the broad inverse mismatch.", size=22, fill="#5b4d45", anchor="mm")
    _text(draw, (WIDTH / 2, 1230), "Post-open proxy diagnostic | no deployment, resource, external, real-BOST, or breakthrough claim", size=18, fill="#6f625a", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
