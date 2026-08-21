#!/usr/bin/env python3
"""Build the public v178 affine-span capacity figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "docs/poolfire_train_field_affine_span_capacity_v178_public_summary.json"
)
OUTPUT = ROOT / "assets/figures/poolfire_train_field_affine_span_capacity_v178.png"
WIDTH = 2520
HEIGHT = 1320


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def _panel(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str
) -> None:
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


def _metric_bars(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    rows: list[tuple[str, float, float, str]],
) -> None:
    left = box[0] + 62
    right = box[2] - 48
    bar_left = left + 175
    scale_max = 1.0
    for index, (label, value, gate, color) in enumerate(rows):
        y = box[1] + 140 + index * 155
        _text(
            draw, (left, y + 21), label, size=20, fill="#435660", bold=True, anchor="lm"
        )
        draw.rectangle((bar_left, y, right, y + 42), fill="#e7ecef")
        width = (right - bar_left) * min(value / scale_max, 1.0)
        draw.rectangle((bar_left, y, bar_left + width, y + 42), fill=color)
        gate_x = bar_left + (right - bar_left) * gate / scale_max
        draw.line((gate_x, y - 5, gate_x, y + 47), fill="#793f3a", width=3)
        _text(
            draw,
            (right - 8, y + 21),
            f"{value:.3f}",
            size=18,
            fill="#20313d",
            bold=True,
            anchor="rm",
        )
        _text(
            draw,
            (bar_left, y + 68),
            f"frozen gate {gate:.2f}",
            size=16,
            fill="#6f7d84",
            anchor="lm",
        )


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    projection = payload["five_camera_affine_projection_k1"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 58),
        "v178 PoolFire training-field affine-span capacity",
        size=44,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 112),
        "1,010 opened training fields | truth-aware post-open witness | five-camera and all-nine controls",
        size=22,
        fill="#596772",
        anchor="mm",
    )

    left_box = (70, 175, 790, 1015)
    mid_box = (830, 175, 1690, 1015)
    right_box = (1730, 175, 2450, 1015)
    _panel(draw, left_box, "A. Capacity and controls")
    _panel(draw, mid_box, "B. Five-camera K1 p90")
    _panel(draw, right_box, "C. What the result means")

    capacity_rows = [
        ("Affine projection K0", "52/52", "PASS", "#267a68"),
        ("Affine projection K1", "52/52", "PASS", "#267a68"),
        ("Static mean K0", "0/52", "FAIL", "#b64d43"),
        ("Static mean K1", "0/52", "FAIL", "#b64d43"),
    ]
    for index, (label, count, verdict, color) in enumerate(capacity_rows):
        y = 285 + index * 145
        draw.rounded_rectangle(
            (left_box[0] + 52, y, left_box[2] - 45, y + 96),
            radius=7,
            fill="#f7f9f9",
            outline="#d6dfe2",
            width=2,
        )
        _text(
            draw,
            (left_box[0] + 76, y + 32),
            label,
            size=20,
            fill="#42545f",
            bold=True,
            anchor="lm",
        )
        _text(
            draw,
            (left_box[2] - 72, y + 32),
            count,
            size=24,
            fill=color,
            bold=True,
            anchor="rm",
        )
        _text(
            draw,
            (left_box[0] + 76, y + 69),
            verdict,
            size=17,
            fill=color,
            bold=True,
            anchor="lm",
        )

    _metric_bars(
        draw,
        box=mid_box,
        rows=[
            ("Field", projection["global_p90"]["field_relative_l2"], 0.50, "#2d6fa3"),
            (
                "Gradient",
                projection["global_p90"]["gradient_relative_l2"],
                0.75,
                "#267a68",
            ),
            (
                "Observation",
                projection["global_p90"]["observation_relative_l2"],
                0.20,
                "#b27a32",
            ),
        ],
    )
    _text(
        draw,
        ((mid_box[0] + mid_box[2]) / 2, 890),
        "All 13 calibrations and all 4 frames pass",
        size=21,
        fill="#267a68",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        ((mid_box[0] + mid_box[2]) / 2, 936),
        "unchanged CGLS K1 | logical ledger 2A + 1A^T",
        size=17,
        fill="#5b6b73",
        anchor="mm",
    )

    right_left = right_box[0] + 58
    _text(
        draw,
        (right_left, 292),
        "Positive evidence",
        size=23,
        fill="#267a68",
        bold=True,
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 342),
        "The opened training family contains",
        size=19,
        fill="#42545f",
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 380),
        "a passing linear field witness.",
        size=19,
        fill="#42545f",
        bold=True,
        anchor="lm",
    )
    draw.line((right_left, 425, right_box[2] - 55, 425), fill="#d6dfe2", width=2)
    _text(
        draw,
        (right_left, 480),
        "Critical limitation",
        size=23,
        fill="#a76a28",
        bold=True,
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 540),
        "Stable affine rank",
        size=18,
        fill="#596772",
        anchor="lm",
    )
    _text(
        draw,
        (right_box[2] - 65, 540),
        "1009 / 1010",
        size=34,
        fill="#a76a28",
        bold=True,
        anchor="rm",
    )
    _text(
        draw,
        (right_left, 604),
        "Almost full sample rank",
        size=20,
        fill="#42545f",
        bold=True,
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 648),
        "No compact latent space established",
        size=18,
        fill="#596772",
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 690),
        "No observation-only coordinate predictor",
        size=18,
        fill="#596772",
        anchor="lm",
    )
    draw.line((right_left, 742, right_box[2] - 55, 742), fill="#d6dfe2", width=2)
    _text(
        draw,
        (right_left, 795),
        "Next falsifiable gate",
        size=23,
        fill="#2d6fa3",
        bold=True,
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 846),
        "Can observation + geometry predict",
        size=18,
        fill="#42545f",
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 884),
        "the affine coordinates under",
        size=18,
        fill="#42545f",
        anchor="lm",
    )
    _text(
        draw,
        (right_left, 922),
        "complete-trajectory isolation?",
        size=18,
        fill="#42545f",
        bold=True,
        anchor="lm",
    )

    draw.rounded_rectangle(
        (70, 1060, 2450, 1250), radius=8, fill="#ffffff", outline="#c7d2d7", width=2
    )
    _text(
        draw,
        (1260, 1103),
        "Linear field-span capacity is present; deployment-visible predictability is not yet established",
        size=27,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1155),
        "Static mean controls fail 0/52, so the positive result is not a fixed-prior effect",
        size=21,
        fill="#4f5f68",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1207),
        "PASS_TRAIN_FIELD_AFFINE_SPAN_HEADROOM_V178 | 26/26 independent checks | breakthrough=false",
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
