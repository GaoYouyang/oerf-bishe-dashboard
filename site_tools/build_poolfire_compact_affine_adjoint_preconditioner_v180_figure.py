#!/usr/bin/env python3
"""Build the public v180 compact-adjoint-preconditioner figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_compact_affine_adjoint_preconditioner_v180_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_compact_affine_adjoint_preconditioner_v180.png"
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
    _text(
        draw,
        ((box[0] + box[2]) / 2, box[1] + 48),
        title,
        size=27,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def _comparison_row(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    index: int,
    label: str,
    count: str,
    passed: bool,
) -> None:
    y = box[1] + 120 + index * 154
    color = "#267a68" if passed else "#b64d43"
    draw.rounded_rectangle(
        (box[0] + 44, y, box[2] - 44, y + 104),
        radius=7,
        fill="#f7f9f9",
        outline="#d6dfe2",
        width=2,
    )
    _text(draw, (box[0] + 68, y + 34), label, size=20, fill="#42545f", bold=True, anchor="lm")
    _text(draw, (box[2] - 68, y + 34), count, size=26, fill=color, bold=True, anchor="rm")
    _text(draw, (box[0] + 68, y + 77), "PASS" if passed else "FAIL", size=17, fill=color, bold=True, anchor="lm")


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
    bar_left = left + 180
    y = box[1] + 130 + index * 212
    scale_max = 0.8
    _text(draw, (left, y + 22), label, size=20, fill="#435660", bold=True, anchor="lm")
    for row, (name, value, color) in enumerate(
        (("five", five_value, "#315f91"), ("nine", nine_value, "#87631b"))
    ):
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
    five = payload["five_camera_primary_k1"]
    nine = payload["all_nine_primary_k1"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 58),
        "v180 Compact shared inverse: observability is not compact predictability",
        size=40,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 112),
        "v179 exact geometry-specific inverse passes | v180 shared diagonal + rank-16 map fails",
        size=22,
        fill="#596772",
        anchor="mm",
    )

    left_box = (70, 175, 795, 1040)
    middle_box = (835, 175, 1710, 1040)
    right_box = (1750, 175, 2450, 1040)
    _panel(draw, left_box, "A. Complete-gate coverage")
    _panel(draw, middle_box, "B. Shared-map K1 p90")
    _panel(draw, right_box, "C. What the failure means")

    _comparison_row(draw, box=left_box, index=0, label="v179 exact inverse, five", count="52/52", passed=True)
    _comparison_row(draw, box=left_box, index=1, label="v180 shared map, five", count="4/52", passed=False)
    _comparison_row(draw, box=left_box, index=2, label="v180 shared map, nine", count="7/52", passed=False)
    _comparison_row(draw, box=left_box, index=3, label="v180 time strata, five", count="0/4", passed=False)
    _comparison_row(draw, box=left_box, index=4, label="v180 time strata, nine", count="0/4", passed=False)

    five_p90 = five["global_p90"]
    nine_p90 = nine["global_p90"]
    gates = payload["absolute_gate"]
    _metric_row(
        draw,
        box=middle_box,
        index=0,
        label="Field",
        five_value=five_p90["field_relative_l2"],
        nine_value=nine_p90["field_relative_l2"],
        gate=gates["field_p90_max"],
    )
    _metric_row(
        draw,
        box=middle_box,
        index=1,
        label="Gradient",
        five_value=five_p90["gradient_relative_l2"],
        nine_value=nine_p90["gradient_relative_l2"],
        gate=gates["gradient_p90_max"],
    )
    _metric_row(
        draw,
        box=middle_box,
        index=2,
        label="Observation",
        five_value=five_p90["observation_relative_l2"],
        nine_value=nine_p90["observation_relative_l2"],
        gate=gates["observation_p90_max"],
    )

    items = [
        ("Observable", "Exact inverse remains 52/52"),
        ("Shared model", "34,322 fitted coefficients"),
        ("Online K1", "2A + 2A^T"),
        ("Failure mode", "Observation tail"),
        ("Independent audit", "24/24 corrected checks"),
        ("Closed family", "Fixed shared linear map"),
        ("GPU", "Not authorized"),
    ]
    for index, (label, value) in enumerate(items):
        y = right_box[1] + 118 + index * 99
        _text(draw, (right_box[0] + 48, y), label, size=18, fill="#64747c", bold=True)
        _text(draw, (right_box[0] + 48, y + 35), value, size=21, fill="#20313d")

    draw.rounded_rectangle((70, 1085, 2450, 1255), radius=8, fill="#fff8ed", outline="#d6b66f", width=2)
    _text(
        draw,
        (WIDTH / 2, 1135),
        "Scientific decision: FAIL_SHARED_COMPACT_ADJOINT_PRECONDITIONER_V180",
        size=27,
        fill="#8a493f",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 1192),
        "Information exists, but this fixed shared low-rank linear map misses geometry-dependent inverse structure.",
        size=22,
        fill="#5b4d45",
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 1230),
        "Post-open proxy diagnostic | no resource, external, curved-ray, real-BOST, or breakthrough claim",
        size=18,
        fill="#6f625a",
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
