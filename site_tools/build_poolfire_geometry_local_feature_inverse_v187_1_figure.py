#!/usr/bin/env python3
"""Build the public v187.1 geometry-local feature-capacity verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_geometry_local_feature_inverse_v187_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_geometry_local_feature_inverse_v187_1.png"
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


def label(
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
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#bcc9cd", width=2)
    label(
        draw,
        ((box[0] + box[2]) / 2, box[1] + 48),
        title,
        size=28,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def strict_safe_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    k0: dict,
    k1: dict,
) -> None:
    rows = (
        ("Five cameras K0", k0["five_camera"]["strict_cells_safe"], "#a9b6bb"),
        ("All nine K0", k0["all_nine"]["strict_cells_safe"], "#a9b6bb"),
        ("Five cameras K1", k1["five_camera"]["strict_cells_safe"], "#287f72"),
        ("All nine K1", k1["all_nine"]["strict_cells_safe"], "#3d6d9a"),
    )
    left = box[0] + 42
    bar_left = left + 220
    right = box[2] - 42
    for index, (name, value, color) in enumerate(rows):
        top = box[1] + 132 + index * 136
        label(draw, (left, top + 25), name, size=18, fill="#2f4149", bold=True, anchor="lm")
        draw.rounded_rectangle((bar_left, top, right, top + 50), radius=5, fill="#e8edef")
        value_x = bar_left + (right - bar_left) * value / 52.0
        if value:
            draw.rounded_rectangle((bar_left, top, value_x, top + 50), radius=5, fill=color)
        label(draw, (right - 7, top + 25), f"{value} / 52", size=19, fill="#17232d", bold=True, anchor="rm")
    label(draw, (left, box[3] - 78), "Frozen requirement: 52 / 52", size=18, fill="#a84e45", bold=True)
    label(draw, (left, box[3] - 42), "Complete time strata: 0 / 4 in every arm", size=17, fill="#5e6b72")


def ratio_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], k1: dict) -> None:
    metrics = (
        ("Field", "field_p90", 0.5),
        ("Gradient", "gradient_p90", 0.75),
        ("Observation", "observation_p90", 0.2),
    )
    plot_left = box[0] + 178
    plot_right = box[2] - 44
    scale_max = 9.2
    gate_x = plot_left + (plot_right - plot_left) / scale_max
    draw.line((gate_x, box[1] + 110, gate_x, box[3] - 105), fill="#a84e45", width=5)
    label(draw, (gate_x + 8, box[1] + 104), "gate = 1x", size=15, fill="#a84e45", bold=True)

    for index, (name, key, gate) in enumerate(metrics):
        top = box[1] + 155 + index * 185
        label(draw, (box[0] + 34, top + 45), name, size=19, fill="#2f4149", bold=True, anchor="lm")
        for offset, arm, color, arm_label in (
            (0, "five_camera", "#287f72", "five"),
            (64, "all_nine", "#3d6d9a", "nine"),
        ):
            value = k1[arm][key]
            ratio = value / gate
            yy = top + offset
            draw.rounded_rectangle((plot_left, yy, plot_right, yy + 42), radius=5, fill="#e8edef")
            xx = plot_left + (plot_right - plot_left) * min(ratio, scale_max) / scale_max
            draw.rounded_rectangle((plot_left, yy, xx, yy + 42), radius=5, fill=color)
            label(draw, (plot_left - 12, yy + 21), arm_label, size=15, fill=color, bold=True, anchor="rm")
            label(draw, (plot_right - 6, yy + 21), f"{value:.3f}  ({ratio:.2f}x)", size=15, fill="#17232d", bold=True, anchor="rm")
    label(draw, (box[0] + 34, box[3] - 58), "All-nine fails every K1 metric by a wide margin.", size=17, fill="#a84e45", bold=True)


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    k0 = payload["primary_k0"]
    k1 = payload["primary_k1"]
    conditioning = payload["conditioning_diagnostics"]
    independent = payload["independent_recomputation"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    label(draw, (WIDTH / 2, 58), "v187.1 Geometry-local inverse still fails", size=42, fill="#17232d", bold=True, anchor="mm")
    label(draw, (WIDTH / 2, 112), "Removing shared regression does not rescue the pooled DCT12 + Plucker feature map", size=22, fill="#596772", anchor="mm")

    strict_box = (70, 170, 980, 930)
    ratio_box = (1020, 170, 1880, 930)
    verdict_box = (1920, 170, 2330, 930)
    panel(draw, strict_box, "A. Strict-safe heldout cells")
    panel(draw, ratio_box, "B. K1 p90 relative to frozen gates")
    panel(draw, verdict_box, "C. Attribution")
    strict_safe_panel(draw, strict_box, k0, k1)
    ratio_panel(draw, ratio_box, k1)

    center = (verdict_box[0] + verdict_box[2]) / 2
    label(draw, (center, 278), "FAIL", size=70, fill="#a84e45", bold=True, anchor="mm")
    label(draw, (center, 350), "feature capacity", size=20, fill="#65747b", anchor="mm")
    bullets = [
        "shared fit removed",
        "rank: 715-1001",
        f"condition max: {conditioning['condition_maximum']:.2e}",
        f"independent: {independent['checks_passed']}/{independent['checks_total']}",
        "camera permutation: exact",
        "dense v185 remains",
        "GPU authorized: false",
        "algorithm breakthrough: false",
    ]
    for index, item in enumerate(bullets):
        yy = 438 + index * 54
        color = "#287f72" if index in (3, 4, 5) else "#a84e45" if index in (1, 2) else "#a66d2c"
        draw.ellipse((verdict_box[0] + 28, yy - 7, verdict_box[0] + 42, yy + 7), fill=color)
        label(draw, (verdict_box[0] + 56, yy), item, size=15, fill="#31434c", bold=index in (1, 2, 3), anchor="lm")

    draw.rounded_rectangle((70, 985, 2330, 1170), radius=8, fill="#20313d")
    label(draw, (1200, 1032), "Scientific interpretation", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    label(draw, (1200, 1082), "The pooled feature map, not only the shared regressor, loses or ill-conditions required information.", size=23, fill="#ffffff", bold=True, anchor="mm")
    label(draw, (1200, 1130), "Close this pooled representation; preserve dense v185 capacity and next separate pooling loss from DCT12 truncation.", size=19, fill="#d8e2e5", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
