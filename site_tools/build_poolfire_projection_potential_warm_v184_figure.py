#!/usr/bin/env python3
"""Build the public v184 projection-potential verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_projection_potential_warm_v184_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_projection_potential_warm_v184.png"
WIDTH = 2400
HEIGHT = 1240


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    choices = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for choice in choices:
        try:
            return ImageFont.truetype(choice, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    *,
    size: int,
    fill: str,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill, anchor=anchor)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#c3ced2", width=2)
    text(
        draw,
        ((box[0] + box[2]) / 2, box[1] + 48),
        title,
        size=28,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def metric_rows(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    parent: dict[str, float],
    current: dict[str, float],
) -> None:
    labels = (
        ("Field", "field_p90", "field_relative_l2", 0.50),
        ("Gradient", "gradient_p90", "gradient_relative_l2", 0.75),
        ("Observation", "observation_p90", "observation_relative_l2", 0.20),
    )
    left = box[0] + 52
    bar_left = left + 190
    right = box[2] - 48
    scale = 1.0
    for metric_index, (label, parent_key, current_key, gate) in enumerate(labels):
        base = box[1] + 150 + metric_index * 205
        text(draw, (left, base + 42), label, size=21, fill="#2a3d46", bold=True, anchor="lm")
        for row, (name, value, color) in enumerate(
            (
                ("v183", parent[parent_key], "#267a68"),
                ("v184", current[current_key], "#b34840"),
            )
        ):
            yy = base + row * 64
            draw.rounded_rectangle((bar_left, yy, right, yy + 38), radius=4, fill="#e8edef")
            width = (right - bar_left) * min(value / scale, 1.0)
            draw.rounded_rectangle((bar_left, yy, bar_left + width, yy + 38), radius=4, fill=color)
            gate_x = bar_left + (right - bar_left) * gate / scale
            draw.line((gate_x, yy - 5, gate_x, yy + 43), fill="#805da8", width=4)
            text(draw, (bar_left - 12, yy + 19), name, size=17, fill="#65747b", anchor="rm")
            text(draw, (right - 8, yy + 19), f"{value:.3f}", size=18, fill="#17232d", bold=True, anchor="rm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    five = payload["five_camera_primary_k1"]
    nine = payload["all_nine_primary_k1"]
    parent = payload["parent_v183_primary_k1"]
    diagnostic = payload["mechanism_diagnostics"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    text(
        draw,
        (WIDTH / 2, 58),
        "v184 Detector potential: integrable residual, unusable 3D inverse lift",
        size=40,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    text(
        draw,
        (WIDTH / 2, 112),
        "Zero-mean potential + scalar-ray Jacobi lift + unchanged CGLS K1 | 50/50 independent checks",
        size=22,
        fill="#596772",
        anchor="mm",
    )

    five_box = (70, 170, 890, 940)
    nine_box = (930, 170, 1750, 940)
    verdict_box = (1790, 170, 2330, 940)
    panel(draw, five_box, "A. Five-camera K1 p90")
    panel(draw, nine_box, "B. All-nine K1 p90")
    panel(draw, verdict_box, "C. Mechanism verdict")
    metric_rows(draw, five_box, parent["five_camera"], five["global_p90"])
    metric_rows(draw, nine_box, parent["all_nine"], nine["global_p90"])
    text(draw, (five_box[0] + 52, 890), "Purple marker: frozen metric gate", size=18, fill="#6d4d8b")
    text(draw, (nine_box[0] + 52, 890), "Green: v183 parent | red: v184", size=18, fill="#50636b")

    text(draw, ((verdict_box[0] + verdict_box[2]) / 2, 310), "FAIL", size=72, fill="#a6453e", bold=True, anchor="mm")
    text(draw, ((verdict_box[0] + verdict_box[2]) / 2, 405), "strict-safe cells", size=20, fill="#65747b", anchor="mm")
    text(draw, ((verdict_box[0] + verdict_box[2]) / 2, 470), "0 / 52  |  0 / 52", size=40, fill="#a6453e", bold=True, anchor="mm")
    bullets = [
        f"residual energy explained >= {100 * diagnostic['minimum_detector_gradient_explained_energy']:.1f}%",
        "field p90: fail in both arms",
        "gradient p90: fail in both arms",
        "observation p90: fail in both arms",
        "independent checks: 50 / 50",
        "algorithm breakthrough: false",
    ]
    for index, item in enumerate(bullets):
        yy = 570 + index * 57
        color = "#267a68" if index in (0, 4) else "#a6453e"
        draw.ellipse((verdict_box[0] + 42, yy - 7, verdict_box[0] + 56, yy + 7), fill=color)
        text(draw, (verdict_box[0] + 72, yy), item, size=17, fill="#31434c", anchor="lm")

    draw.rounded_rectangle((70, 995, 2330, 1170), radius=8, fill="#20313d")
    text(draw, (1200, 1042), "Conclusion", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    text(
        draw,
        (1200, 1090),
        "Detector-plane integrability is real, but it does not identify a field-compatible 3D inverse direction.",
        size=24,
        fill="#ffffff",
        bold=True,
        anchor="mm",
    )
    text(
        draw,
        (1200, 1134),
        "Close this exact scalar-potential Jacobi lift; do not tune or rescue it with a larger model or GPU.",
        size=20,
        fill="#d8e2e5",
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
