#!/usr/bin/env python3
"""Build the public v188 camera-resolved DCT12 attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_camera_resolved_dct12_capacity_v188_public_summary.json"
PARENT = ROOT / "docs/poolfire_geometry_local_feature_inverse_v187_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_camera_resolved_dct12_capacity_v188.png"
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
        size=27,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def metric_comparison_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    parent: dict,
    current: dict,
) -> None:
    metrics = (
        ("Field", "field_p90", 0.5),
        ("Gradient", "gradient_p90", 0.75),
        ("Observation", "observation_p90", 0.2),
    )
    plot_left = box[0] + 178
    plot_right = box[2] - 42
    scale_max = 9.5
    gate_x = plot_left + (plot_right - plot_left) / scale_max
    draw.line((gate_x, box[1] + 110, gate_x, box[3] - 90), fill="#a84e45", width=5)
    label(draw, (gate_x + 8, box[1] + 106), "gate = 1x", size=15, fill="#a84e45", bold=True)
    for index, (name, key, gate) in enumerate(metrics):
        top = box[1] + 155 + index * 188
        label(draw, (box[0] + 32, top + 46), name, size=18, fill="#2f4149", bold=True, anchor="lm")
        for offset, payload, color, name_label in (
            (0, parent, "#a9b6bb", "pooled"),
            (66, current, "#3d6d9a", "resolved"),
        ):
            value = payload[key]
            ratio = value / gate
            yy = top + offset
            draw.rounded_rectangle((plot_left, yy, plot_right, yy + 42), radius=5, fill="#e8edef")
            xx = plot_left + (plot_right - plot_left) * min(ratio, scale_max) / scale_max
            draw.rounded_rectangle((plot_left, yy, xx, yy + 42), radius=5, fill=color)
            label(draw, (plot_left - 12, yy + 21), name_label, size=14, fill=color, bold=True, anchor="rm")
            label(draw, (plot_right - 6, yy + 21), f"{value:.3f} ({ratio:.2f}x)", size=15, fill="#17232d", bold=True, anchor="rm")
    label(draw, (box[0] + 32, box[3] - 52), "All-nine improves sharply, but all three p90 gates still fail.", size=17, fill="#a84e45", bold=True)


def five_camera_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    parent: dict,
    current: dict,
) -> None:
    label(draw, ((box[0] + box[2]) / 2, box[1] + 162), "2 / 52", size=70, fill="#a84e45", bold=True, anchor="mm")
    label(draw, ((box[0] + box[2]) / 2, box[1] + 224), "strict-safe K1 cells", size=18, fill="#65747b", anchor="mm")
    rows = (
        ("Field p90", "field_p90", 0.5),
        ("Gradient p90", "gradient_p90", 0.75),
        ("Observation p90", "observation_p90", 0.2),
    )
    for index, (name, key, gate) in enumerate(rows):
        yy = box[1] + 330 + index * 114
        before = parent[key]
        after = current[key]
        label(draw, (box[0] + 34, yy), name, size=16, fill="#45565e", bold=True, anchor="lm")
        label(draw, (box[2] - 34, yy), f"{after:.6f}", size=18, fill="#17232d", bold=True, anchor="rm")
        label(draw, (box[0] + 34, yy + 38), f"pooled {before:.6f}  |  gate {gate:.2f}", size=14, fill="#6e7c82", anchor="lm")
    label(draw, ((box[0] + box[2]) / 2, box[3] - 70), "No material change from v187.1", size=20, fill="#a66d2c", bold=True, anchor="mm")


def main() -> int:
    current = json.loads(SUMMARY.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    current_k1 = current["primary_k1"]
    parent_k1 = parent["primary_k1"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    label(draw, (WIDTH / 2, 58), "v188 Camera-resolved DCT12 still fails", size=42, fill="#17232d", bold=True, anchor="mm")
    label(draw, (WIDTH / 2, 112), "Pooling explains much of the all-nine instability, but not the remaining capacity gap", size=22, fill="#596772", anchor="mm")

    five_box = (70, 170, 720, 930)
    nine_box = (760, 170, 1760, 930)
    verdict_box = (1800, 170, 2330, 930)
    panel(draw, five_box, "A. Five cameras")
    panel(draw, nine_box, "B. All-nine: pooled vs resolved")
    panel(draw, verdict_box, "C. Attribution")
    five_camera_panel(draw, five_box, parent_k1["five_camera"], current_k1["five_camera"])
    metric_comparison_panel(draw, nine_box, parent_k1["all_nine"], current_k1["all_nine"])

    center = (verdict_box[0] + verdict_box[2]) / 2
    label(draw, (center, 278), "FAIL", size=70, fill="#a84e45", bold=True, anchor="mm")
    label(draw, (center, 350), "DCT12 capacity", size=20, fill="#65747b", anchor="mm")
    bullets = [
        "five: unchanged",
        "nine p90: -66% to -70%",
        "nine: 0 / 52 strict-safe",
        "rank: 715 / 1009",
        "condition max: 4.33e4",
        "independent: 44 / 44",
        "camera permutation: exact",
        "GPU authorized: false",
        "algorithm breakthrough: false",
    ]
    for index, item in enumerate(bullets):
        yy = 430 + index * 49
        color = "#287f72" if index in (1, 3, 4, 5, 6) else "#a84e45" if index in (0, 2) else "#a66d2c"
        draw.ellipse((verdict_box[0] + 30, yy - 7, verdict_box[0] + 44, yy + 7), fill=color)
        label(draw, (verdict_box[0] + 58, yy), item, size=15, fill="#31434c", bold=index in (0, 1, 2, 5), anchor="lm")

    draw.rounded_rectangle((70, 985, 2330, 1170), radius=8, fill="#20313d")
    label(draw, (1200, 1032), "Scientific interpretation", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    label(draw, (1200, 1082), "Cross-camera pooling was a major all-nine penalty, but removing it is not sufficient.", size=23, fill="#ffffff", bold=True, anchor="mm")
    label(draw, (1200, 1130), "Close pooled and camera-resolved DCT12; dense per-camera potential capacity remains the next reference question.", size=19, fill="#d8e2e5", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
