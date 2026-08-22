#!/usr/bin/env python3
"""Build the public v189 dense camera-resolved DCT attribution figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_dense_camera_resolved_dct_capacity_v189_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_dense_camera_resolved_dct_capacity_v189.png"
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
        ((box[0] + box[2]) / 2, box[1] + 50),
        title,
        size=27,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def before_after_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    rows = (("Five cameras", 2, 52), ("All nine", 0, 52))
    left = box[0] + 52
    right = box[2] - 52
    for index, (name, before, after) in enumerate(rows):
        top = box[1] + 145 + index * 248
        label(draw, (left, top), name, size=21, fill="#31434c", bold=True, anchor="lm")
        label(draw, (left, top + 62), f"DCT12   {before}/52", size=25, fill="#a84e45", bold=True, anchor="lm")
        label(draw, (right, top + 62), f"Full DCT   {after}/52", size=25, fill="#287f72", bold=True, anchor="rm")
        line_y = top + 122
        draw.line((left, line_y, right, line_y), fill="#d6dfe1", width=18)
        draw.line((left, line_y, right, line_y), fill="#287f72", width=18)
        draw.ellipse((left - 13, line_y - 13, left + 13, line_y + 13), fill="#a84e45")
        draw.ellipse((right - 13, line_y - 13, right + 13, line_y + 13), fill="#287f72")
    label(draw, ((left + right) / 2, box[3] - 82), "Unchanged physical K1", size=19, fill="#65747b", bold=True, anchor="mm")


def gate_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], data: dict) -> None:
    metrics = (
        ("Field", "field_p90", 0.5, "#3d6d9a"),
        ("Gradient", "gradient_p90", 0.75, "#287f72"),
        ("Observation", "observation_p90", 0.2, "#a66d2c"),
    )
    arms = (("Five", data["five_camera"]), ("Nine", data["all_nine"]))
    plot_left = box[0] + 160
    plot_right = box[2] - 42
    for metric_index, (name, key, gate, color) in enumerate(metrics):
        top = box[1] + 132 + metric_index * 174
        label(draw, (box[0] + 32, top + 48), name, size=18, fill="#31434c", bold=True, anchor="lm")
        for arm_index, (arm_name, payload) in enumerate(arms):
            yy = top + arm_index * 62
            ratio = payload[key] / gate
            draw.rounded_rectangle((plot_left, yy, plot_right, yy + 38), radius=5, fill="#e8edef")
            xx = plot_left + (plot_right - plot_left) * min(ratio, 1.0)
            draw.rounded_rectangle((plot_left, yy, xx, yy + 38), radius=5, fill=color)
            label(draw, (plot_left - 12, yy + 19), arm_name, size=14, fill="#65747b", bold=True, anchor="rm")
            label(draw, (plot_right - 8, yy + 19), f"{payload[key]:.3f}  ({ratio:.2f}x gate)", size=14, fill="#17232d", bold=True, anchor="rm")
    label(draw, ((box[0] + box[2]) / 2, box[3] - 82), "52/52 cells · 13/13 calibrations · 4/4 times", size=18, fill="#287f72", bold=True, anchor="mm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    label(draw, (WIDTH / 2, 58), "v189 Full per-camera DCT restores dense capacity", size=42, fill="#17232d", bold=True, anchor="mm")
    label(draw, (WIDTH / 2, 112), "The frozen v188 failure is attributable to DCT12 spectral truncation", size=22, fill="#596772", anchor="mm")

    change_box = (70, 170, 820, 930)
    gate_box = (860, 170, 1760, 930)
    verdict_box = (1800, 170, 2330, 930)
    panel(draw, change_box, "A. K1 strict-safe cells")
    panel(draw, gate_box, "B. Full-DCT K1 p90 / gate")
    panel(draw, verdict_box, "C. Attribution")
    before_after_panel(draw, change_box)
    gate_panel(draw, gate_box, payload["primary_k1"])

    center = (verdict_box[0] + verdict_box[2]) / 2
    label(draw, (center, 278), "PASS", size=70, fill="#287f72", bold=True, anchor="mm")
    label(draw, (center, 350), "root-cause attribution", size=20, fill="#65747b", anchor="mm")
    bullets = [
        "575 modes / camera",
        "rank: 1009 / 1009",
        "condition: 48.93–187.41",
        "v185 field match: 1.62e-14",
        "independent: 50 / 50",
        "camera permutation: exact",
        "compact algorithm: false",
        "GPU authorized: false",
        "algorithm breakthrough: false",
    ]
    for index, item in enumerate(bullets):
        yy = 430 + index * 49
        color = "#287f72" if index < 6 else "#a66d2c"
        draw.ellipse((verdict_box[0] + 30, yy - 7, verdict_box[0] + 44, yy + 7), fill=color)
        label(draw, (verdict_box[0] + 58, yy), item, size=15, fill="#31434c", bold=index in (0, 1, 3, 4), anchor="lm")

    draw.rounded_rectangle((70, 985, 2330, 1170), radius=8, fill="#20313d")
    label(draw, (1200, 1032), "Scientific interpretation", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    label(draw, (1200, 1082), "Restoring omitted detector frequencies reproduces v185 and closes the v188 ambiguity.", size=22, fill="#ffffff", bold=True, anchor="mm")
    label(draw, (1200, 1130), "This is a full-basis capacity reference, not a compact predictor, call reduction, or deployment result.", size=19, fill="#d8e2e5", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
