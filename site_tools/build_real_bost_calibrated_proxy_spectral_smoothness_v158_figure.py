#!/usr/bin/env python3
"""Build the public v158 spectral-smoothness decision figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_calibrated_proxy_spectral_smoothness_v158_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_calibrated_proxy_spectral_smoothness_v158.png"
WIDTH = 2520
HEIGHT = 1040


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


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    parent: list[float],
    primary: list[float],
    threshold: float,
) -> None:
    left, top, right, bottom = box
    ceiling = max(max(parent + primary) * 1.12, threshold * 1.22)
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#d7e0e7", width=2)
    _text(draw, ((left + right) / 2, top + 40), title, size=27, fill="#1f2d38", bold=True, anchor="mm")
    plot_left, plot_right = left + 78, right - 42
    plot_top, plot_bottom = top + 92, bottom - 96
    for step in range(5):
        value = ceiling * step / 4
        y = plot_bottom - (plot_bottom - plot_top) * value / ceiling
        draw.line((plot_left, y, plot_right, y), fill="#e6ebef", width=1)
        _text(draw, (plot_left - 12, y), f"{value:.2f}", size=17, fill="#66737e", anchor="rm")
    gate_y = plot_bottom - (plot_bottom - plot_top) * threshold / ceiling
    draw.line((plot_left, gate_y, plot_right, gate_y), fill="#247a63", width=4)
    _text(draw, (plot_right - 4, gate_y - 8), f"gate {threshold:.2f}", size=17, fill="#247a63", bold=True, anchor="rb")
    centers = [plot_left + (plot_right - plot_left) * ratio for ratio in (0.18, 0.50, 0.82)]
    for index, center in enumerate(centers):
        parent_y = plot_bottom - (plot_bottom - plot_top) * parent[index] / ceiling
        primary_y = plot_bottom - (plot_bottom - plot_top) * primary[index] / ceiling
        draw.rounded_rectangle((center - 62, parent_y, center - 5, plot_bottom), radius=4, fill="#9ca8b2")
        draw.rounded_rectangle((center + 5, primary_y, center + 62, plot_bottom), radius=4, fill="#2f9e8f")
        _text(draw, (center - 34, parent_y - 10), f"{parent[index]:.3f}", size=17, fill="#63707a", anchor="mb")
        _text(draw, (center + 34, primary_y - 10), f"{primary[index]:.3f}", size=17, fill="#247a63", bold=True, anchor="mb")
        _text(draw, (center, plot_bottom + 36), f"{(5, 7, 9)[index]} cameras", size=18, fill="#42515d", anchor="mm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    primary = payload["primary"]["by_camera_count"]
    parent = payload["parent_control"]
    counts = ["5", "7", "9"]
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f7f9")
    draw = ImageDraw.Draw(image)

    _text(draw, (WIDTH / 2, 54), "v158 observable spectral-smoothing gate", size=42, fill="#17232d", bold=True, anchor="mm")
    _text(draw, (WIDTH / 2, 106), "gray: K16 parent | green: residual-budget Tikhonov primary | lower is better", size=24, fill="#596772", anchor="mm")

    _panel(
        draw,
        (70, 160, 820, 770),
        "Field relative error p90",
        [parent["field_p90_by_camera_count"][c] for c in counts],
        [primary[c]["field_p90"] for c in counts],
        0.50,
    )
    _panel(
        draw,
        (885, 160, 1635, 770),
        "Gradient relative error p90",
        [parent["gradient_p90_by_camera_count"][c] for c in counts],
        [primary[c]["gradient_p90"] for c in counts],
        0.75,
    )
    _panel(
        draw,
        (1700, 160, 2450, 770),
        "Observation relative error p90",
        [parent["observation_p90_by_camera_count"][c] for c in counts],
        [primary[c]["observation_p90"] for c in counts],
        0.20,
    )

    draw.rounded_rectangle((115, 820, 2405, 955), radius=8, fill="#fff3e6", outline="#e5bd8d", width=2)
    _text(draw, (155, 850), "Independently verified decision", size=24, fill="#7b4c19", bold=True)
    _text(draw, (155, 897), "7 and 9 cameras pass. The 5-camera field p90 is 0.630 > 0.500, so the preregistered primary fails.", size=25, fill="#5f4932")
    _text(draw, (155, 934), "Fixed 0.03/0.1 rows are diagnostic-only and cannot replace the primary after results.", size=22, fill="#75614c")
    _text(draw, (WIDTH / 2, 1004), "controlled proxy | no paired experimental 2D displacement | predictor and GPU not authorized", size=22, fill="#66737e", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
