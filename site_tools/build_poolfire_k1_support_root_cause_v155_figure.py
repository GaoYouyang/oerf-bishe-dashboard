#!/usr/bin/env python3
"""Build the public v155 mixed-support root-cause figure with Pillow."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_k1_support_root_cause_v155_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_k1_support_root_cause_v155.png"
WIDTH = 2520
HEIGHT = 980


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
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


def _axis(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], ylabel: str) -> None:
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill="#8996a3", width=2)
    draw.line((left, bottom, right, bottom), fill="#8996a3", width=2)
    for value in range(0, 101, 20):
        y = bottom - (bottom - top) * value / 100
        draw.line((left, y, right, y), fill="#d9e2ea", width=1)
        _text(draw, (left - 14, y), str(value), size=22, fill="#425466", anchor="rm")
    _text(draw, (left + 4, top - 13), ylabel, size=19, fill="#425466", anchor="lb")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    keys = ["p45_size05", "p58_size03", "p58_size05"]
    labels = ["p45-s05", "p58-s03", "p58-s05"]
    blocks = ["observation", "k1_residual", "k1_dual", "reported_geometry"]
    block_labels = ["Observation", "K1 residual", "K1 dual", "Reported geometry"]
    colors = ["#4c78a8", "#2f9e8f", "#756bb1", "#e0a12f"]
    shares = payload["unsupported_aggregate_block_share"]
    frame_support = payload["selected_frame_support"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f7fafc")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 60),
        "v155 support-gap attribution: state and geometry both matter",
        size=42,
        fill="#17212b",
        bold=True,
        anchor="mm",
    )

    left_box = (145, 205, 1190, 765)
    right_box = (1390, 205, 2420, 765)
    _axis(draw, left_box, "Unsupported distance share (%)")
    _axis(draw, right_box, "Supported camera rows (%)")
    _text(draw, ((left_box[0] + left_box[2]) / 2, 155), "Frozen feature-block decomposition", size=30, fill="#26323d", bold=True, anchor="mm")
    _text(draw, ((right_box[0] + right_box[2]) / 2, 155), "Frame variation is descriptive, not a temporal proof", size=30, fill="#26323d", bold=True, anchor="mm")

    bar_width = 185
    bar_centers = [390, 665, 940]
    top = left_box[1]
    bottom = left_box[3]
    scale = (bottom - top) / 100
    for key, label, center in zip(keys, labels, bar_centers, strict=True):
        y_bottom = bottom
        for block, color in zip(blocks, colors, strict=True):
            value = 100 * shares[key][block]
            y_top = y_bottom - value * scale
            draw.rectangle((center - bar_width / 2, y_top, center + bar_width / 2, y_bottom), fill=color)
            _text(draw, (center, (y_top + y_bottom) / 2), f"{value:.1f}%", size=23, fill="white", bold=True, anchor="mm")
            y_bottom = y_top
        _text(draw, (center, bottom + 42), label, size=27, fill="#26323d", bold=True, anchor="mm")

    legend_y = 820
    legend_x = 185
    for label, color in zip(block_labels, colors, strict=True):
        draw.rectangle((legend_x, legend_y, legend_x + 28, legend_y + 28), fill=color)
        _text(draw, (legend_x + 40, legend_y + 14), label, size=22, fill="#425466", anchor="lm")
        legend_x += 235

    gate_y = right_box[3] - 90 * (right_box[3] - right_box[1]) / 100
    draw.line((right_box[0], gate_y, right_box[2], gate_y), fill="#2f855a", width=4)
    _text(draw, (right_box[2] - 6, gate_y + 31), "90% gate", size=20, fill="#2f855a", bold=True, anchor="rt")

    frame_values = [0, 25, 50, 75, 100]
    line_colors = ["#d45d4c", "#e0a12f", "#756bb1"]
    for key, label, color in zip(keys, labels, line_colors, strict=True):
        points = []
        for frame in frame_values:
            x = right_box[0] + (right_box[2] - right_box[0]) * frame / 100
            value = 100 * frame_support[key][str(frame)]
            y = right_box[3] - (right_box[3] - right_box[1]) * value / 100
            points.append((x, y))
        draw.line(points, fill=color, width=5, joint="curve")
        for (x, y), frame in zip(points, frame_values, strict=True):
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)
            value = 100 * frame_support[key][str(frame)]
            _text(draw, (x, y - 18), f"{value:.1f}", size=19, fill=color, bold=True, anchor="mb")
    for frame in frame_values:
        x = right_box[0] + (right_box[2] - right_box[0]) * frame / 100
        _text(draw, (x, right_box[3] + 40), str(frame), size=24, fill="#425466", anchor="mm")
    _text(draw, ((right_box[0] + right_box[2]) / 2, right_box[3] + 83), "Selected frame", size=24, fill="#26323d", anchor="mm")
    right_legend_y = 880
    right_legend_x = 1510
    for label, color in zip(labels, line_colors, strict=True):
        draw.line((right_legend_x, right_legend_y + 13, right_legend_x + 46, right_legend_y + 13), fill=color, width=5)
        draw.ellipse((right_legend_x + 17, right_legend_y + 4, right_legend_x + 35, right_legend_y + 22), fill=color, outline="white", width=2)
        _text(draw, (right_legend_x + 58, right_legend_y + 13), label, size=22, fill="#425466", anchor="lm")
        right_legend_x += 250

    _text(
        draw,
        (WIDTH / 2, 932),
        "ROOT_CAUSE_MIXED_SUPPORT_GAP_V155  |  no predictor or GPU rescue  |  algorithm_breakthrough=false",
        size=25,
        fill="#5b6773",
        anchor="mm",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
