#!/usr/bin/env python3
"""Build the public v183 observation-block Galerkin verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_observation_block_galerkin_v183_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_observation_block_galerkin_v183.png"
WIDTH = 2400
HEIGHT = 1240


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    choices = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for choice in choices:
        try:
            return ImageFont.truetype(choice, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, *, size: int, fill: str, bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill, anchor=anchor)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#c3ced2", width=2)
    text(draw, ((box[0] + box[2]) / 2, box[1] + 48), title, size=28, fill="#20313d", bold=True, anchor="mm")


def comparison(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], y: int, label: str, parent: float, block: float) -> None:
    left = box[0] + 52
    bar_left = left + 260
    right = box[2] - 52
    scale = 0.30
    text(draw, (left, y - 40), label, size=22, fill="#2a3d46", bold=True, anchor="lm")
    for row, (name, value, color) in enumerate((("v182 global", parent, "#687880"), ("v183 blocks", block, "#267a68"))):
        yy = y + row * 68
        draw.rounded_rectangle((bar_left, yy, right, yy + 42), radius=4, fill="#e8edef")
        width = (right - bar_left) * value / scale
        draw.rounded_rectangle((bar_left, yy, bar_left + width, yy + 42), radius=4, fill=color)
        gate_x = bar_left + (right - bar_left) * 0.20 / scale
        draw.line((gate_x, yy - 6, gate_x, yy + 48), fill="#b34840", width=4)
        text(draw, (bar_left - 14, yy + 21), name, size=17, fill="#65747b", anchor="rm")
        text(draw, (right - 8, yy + 21), f"{value:.3f}", size=19, fill="#17232d", bold=True, anchor="rm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    five = payload["five_camera_primary_k1"]
    nine = payload["all_nine_primary_k1"]
    parent = payload["v182_parent_comparison"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    text(draw, (WIDTH / 2, 58), "v183 Camera-component block Galerkin: useful structure, incomplete gate", size=40, fill="#17232d", bold=True, anchor="mm")
    text(draw, (WIDTH / 2, 112), "Joint observable solve + unchanged CGLS K1 | no truth, tuning, ridge, damping, or fallback", size=22, fill="#596772", anchor="mm")

    left = (70, 170, 1170, 940)
    middle = (1210, 170, 1820, 940)
    right = (1860, 170, 2330, 940)
    panel(draw, left, "A. Observation p90 versus v182")
    panel(draw, middle, "B. Strict-safe cells after K1")
    panel(draw, right, "C. Scientific boundary")

    comparison(draw, left, 310, "Five cameras", parent["five_camera_observation_p90"], five["global_p90"]["observation_relative_l2"])
    comparison(draw, left, 610, "All nine", parent["all_nine_observation_p90"], nine["global_p90"]["observation_relative_l2"])
    text(draw, (left[0] + 52, 862), "Red marker: frozen observation p90 gate = 0.20", size=19, fill="#7a4a46")

    for idx, (name, safe, color) in enumerate((("Five cameras", five["strict_cells_safe"], "#315f91"), ("All nine", nine["strict_cells_safe"], "#946d22"))):
        y = 340 + idx * 260
        text(draw, ((middle[0] + middle[2]) / 2, y), name, size=24, fill="#334750", bold=True, anchor="mm")
        text(draw, ((middle[0] + middle[2]) / 2, y + 84), f"{safe} / 52", size=62, fill=color, bold=True, anchor="mm")
        text(draw, ((middle[0] + middle[2]) / 2, y + 150), "complete gate: FAIL", size=21, fill="#a6453e", bold=True, anchor="mm")

    text(draw, ((right[0] + right[2]) / 2, 330), "FAIL", size=72, fill="#a6453e", bold=True, anchor="mm")
    bullets = [
        "field p90: pass in both arms",
        "gradient p90: pass in both arms",
        "observation p90: fail in both arms",
        "independent checks: 46 / 46",
        "matched accuracy: not established",
        "algorithm breakthrough: false",
    ]
    for idx, item in enumerate(bullets):
        yy = 450 + idx * 70
        draw.ellipse((right[0] + 52, yy - 7, right[0] + 66, yy + 7), fill="#267a68" if idx < 2 or idx == 3 else "#a6453e")
        text(draw, (right[0] + 82, yy), item, size=18, fill="#31434c", anchor="lm")

    draw.rounded_rectangle((70, 995, 2330, 1170), radius=8, fill="#20313d")
    text(draw, (1200, 1042), "Conclusion", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    text(draw, (1200, 1090), "Camera-component heterogeneity is real and useful, but one fixed coefficient per block is still insufficient.", size=24, fill="#ffffff", bold=True, anchor="mm")
    text(draw, (1200, 1134), "Close this exact block-Galerkin family; do not tune it post hoc or claim call / wall / RSS gains.", size=20, fill="#d8e2e5", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
