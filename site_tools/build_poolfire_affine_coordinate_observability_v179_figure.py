#!/usr/bin/env python3
"""Build the public v179 affine-coordinate observability figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_affine_coordinate_observability_v179_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_affine_coordinate_observability_v179.png"
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


def _result_row(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    index: int,
    label: str,
    count: str,
    passed: bool,
) -> None:
    y = box[1] + 120 + index * 140
    color = "#267a68" if passed else "#b64d43"
    draw.rounded_rectangle(
        (box[0] + 44, y, box[2] - 44, y + 92),
        radius=7,
        fill="#f7f9f9",
        outline="#d6dfe2",
        width=2,
    )
    _text(draw, (box[0] + 68, y + 31), label, size=20, fill="#42545f", bold=True, anchor="lm")
    _text(draw, (box[2] - 68, y + 31), count, size=25, fill=color, bold=True, anchor="rm")
    _text(draw, (box[0] + 68, y + 66), "PASS" if passed else "FAIL", size=17, fill=color, bold=True, anchor="lm")


def _metric_bar(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    index: int,
    label: str,
    value: float,
    gate: float,
    color: str,
) -> None:
    left = box[0] + 52
    right = box[2] - 46
    bar_left = left + 170
    y = box[1] + 132 + index * 165
    scale_max = 1.0
    _text(draw, (left, y + 21), label, size=20, fill="#435660", bold=True, anchor="lm")
    draw.rectangle((bar_left, y, right, y + 42), fill="#e7ecef")
    width = (right - bar_left) * min(value / scale_max, 1.0)
    draw.rectangle((bar_left, y, bar_left + width, y + 42), fill=color)
    gate_x = bar_left + (right - bar_left) * gate / scale_max
    draw.line((gate_x, y - 5, gate_x, y + 47), fill="#793f3a", width=3)
    _text(draw, (right - 8, y + 21), f"{value:.3f}", size=18, fill="#20313d", bold=True, anchor="rm")
    _text(draw, (bar_left, y + 68), f"frozen gate {gate:.2f}", size=16, fill="#6f7d84", anchor="lm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    k0 = payload["five_camera_measurement_pseudoinverse_k0"]
    k1 = payload["five_camera_measurement_pseudoinverse_k1"]
    controls = payload["five_camera_cheap_controls"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 58),
        "v179 Five-camera affine-coordinate observability",
        size=44,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 112),
        "1,009/1,009 measurement rank | observation + reported geometry only | post-open mechanism diagnostic",
        size=22,
        fill="#596772",
        anchor="mm",
    )

    left_box = (70, 175, 800, 1015)
    middle_box = (840, 175, 1700, 1015)
    right_box = (1740, 175, 2450, 1015)
    _panel(draw, left_box, "A. Frozen five-camera gates")
    _panel(draw, middle_box, "B. Exact inverse K0 p90")
    _panel(draw, right_box, "C. Evidence boundary")

    _result_row(draw, box=left_box, index=0, label="Exact inverse K0", count="52/52", passed=k0["passed"])
    _result_row(draw, box=left_box, index=1, label="Exact inverse K1", count="52/52", passed=k1["passed"])
    _result_row(
        draw,
        box=left_box,
        index=2,
        label="One-step coord. K0",
        count="0/52",
        passed=controls["one_step_coordinate_k0"]["passed"],
    )
    _result_row(
        draw,
        box=left_box,
        index=3,
        label="One-step coord. K1",
        count="0/52",
        passed=controls["one_step_coordinate_k1"]["passed"],
    )
    _result_row(draw, box=left_box, index=4, label="Static mean K0/K1", count="0/52", passed=False)

    p90 = k0["global_p90"]
    _metric_bar(
        draw,
        box=middle_box,
        index=0,
        label="Field",
        value=p90["field_relative_l2"],
        gate=0.5,
        color="#267a68",
    )
    _metric_bar(
        draw,
        box=middle_box,
        index=1,
        label="Gradient",
        value=p90["gradient_relative_l2"],
        gate=0.75,
        color="#315f91",
    )
    _metric_bar(
        draw,
        box=middle_box,
        index=2,
        label="Observation",
        value=p90["observation_relative_l2"],
        gate=0.2,
        color="#87631b",
    )
    _text(draw, (middle_box[0] + 52, middle_box[1] + 690), "After unchanged CGLS K1", size=19, fill="#42545f", bold=True)
    _text(
        draw,
        (middle_box[0] + 52, middle_box[1] + 742),
        "p90 = 0.250 / 0.397 / 0.067",
        size=24,
        fill="#267a68",
        bold=True,
    )

    items = [
        ("Observable", "Full 1,009-dimensional coordinates"),
        ("Independent", "36/36 checks pass"),
        ("Permutation", "Coordinate error <= 1.22e-14"),
        ("Cache cost", "26,260 forward-equivalent projections"),
        ("Trainable", "0 parameters"),
        ("Resource claim", "None"),
    ]
    for index, (label, value) in enumerate(items):
        y = right_box[1] + 120 + index * 112
        _text(draw, (right_box[0] + 48, y), label, size=18, fill="#6a7981", bold=True)
        _text(draw, (right_box[0] + 48, y + 38), value, size=21, fill="#20313d", bold=True)

    draw.rounded_rectangle((70, 1060, 2450, 1250), radius=8, fill="#ffffff", outline="#c7d2d7", width=2)
    _text(
        draw,
        (1260, 1103),
        "The five-camera data contain the coordinates; compact low-cost approximation remains unsolved",
        size=27,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1155),
        "Exact cached pseudoinverse passes, but cheap controls fail and setup cost is large",
        size=21,
        fill="#4f5f68",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1207),
        "PASS_AFFINE_MEASUREMENT_INVERSE_HEADROOM_V179 | 36/36 independent checks | breakthrough=false",
        size=18,
        fill="#267a68",
        bold=True,
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
