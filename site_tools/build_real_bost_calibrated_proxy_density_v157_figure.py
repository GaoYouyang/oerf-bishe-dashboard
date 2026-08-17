#!/usr/bin/env python3
"""Build the public v157 calibrated-proxy reference figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_calibrated_proxy_density_v157_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_calibrated_proxy_density_v157.png"
WIDTH = 2520
HEIGHT = 1040


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


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    values: list[float],
    threshold: float,
) -> None:
    left, top, right, bottom = box
    ceiling = max(max(values) * 1.13, threshold * 1.25)
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#d8e0e7", width=2)
    _text(draw, ((left + right) / 2, top + 42), title, size=28, fill="#24313c", bold=True, anchor="mm")
    plot_left, plot_right = left + 76, right - 42
    plot_top, plot_bottom = top + 92, bottom - 84
    for step in range(5):
        value = ceiling * step / 4
        y = plot_bottom - (plot_bottom - plot_top) * value / ceiling
        draw.line((plot_left, y, plot_right, y), fill="#e4eaf0", width=1)
        _text(draw, (plot_left - 12, y), f"{value:.2f}", size=18, fill="#65717c", anchor="rm")
    gate_y = plot_bottom - (plot_bottom - plot_top) * threshold / ceiling
    draw.line((plot_left, gate_y, plot_right, gate_y), fill="#2f855a", width=4)
    _text(draw, (plot_right - 4, gate_y - 8), f"gate {threshold:.2f}", size=18, fill="#2f855a", bold=True, anchor="rb")
    colors = ["#d45d4c", "#e0a12f", "#2f9e8f"]
    labels = ["5 cameras", "7 cameras", "9 cameras"]
    centers = [plot_left + (plot_right - plot_left) * ratio for ratio in (0.2, 0.5, 0.8)]
    bar_width = 105
    for center, value, color, label in zip(centers, values, colors, labels, strict=True):
        y = plot_bottom - (plot_bottom - plot_top) * value / ceiling
        draw.rounded_rectangle((center - bar_width / 2, y, center + bar_width / 2, plot_bottom), radius=5, fill=color)
        _text(draw, (center, y - 13), f"{value:.3f}", size=22, fill=color, bold=True, anchor="mb")
        _text(draw, (center, plot_bottom + 35), label, size=18, fill="#425466", anchor="mm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = payload["primary"]["by_camera_count"]
    counts = ["5", "7", "9"]
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f7f9")
    draw = ImageDraw.Draw(image)

    _text(draw, (WIDTH / 2, 56), "v157 calibrated-proxy reference gate", size=43, fill="#17212b", bold=True, anchor="mm")
    _text(
        draw,
        (WIDTH / 2, 110),
        "24x24 samples per camera | DCT1024-CGLS K16 | lower is better",
        size=25,
        fill="#55626d",
        anchor="mm",
    )

    _panel(draw, (70, 165, 820, 765), "Field relative error p90", [rows[c]["field_p90"] for c in counts], 0.50)
    _panel(draw, (885, 165, 1635, 765), "Gradient relative error p90", [rows[c]["gradient_p90"] for c in counts], 0.75)
    _panel(draw, (1700, 165, 2450, 765), "Observation relative error p90", [rows[c]["observation_p90"] for c in counts], 0.20)

    draw.rounded_rectangle((115, 820, 2405, 955), radius=8, fill="#eaf4f1", outline="#b8d8ce", width=2)
    _text(draw, (155, 852), "Verified interpretation", size=24, fill="#24594c", bold=True)
    _text(
        draw,
        (155, 900),
        "Nine cameras pass all frozen tails. Five and seven cameras fit observations but still miss field/gradient tails.",
        size=25,
        fill="#2f4a43",
    )
    _text(
        draw,
        (WIDTH / 2, 1003),
        "controlled proxy, not paired real BOST | predictor and GPU not authorized | algorithm_breakthrough=false",
        size=23,
        fill="#65717c",
        anchor="mm",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
