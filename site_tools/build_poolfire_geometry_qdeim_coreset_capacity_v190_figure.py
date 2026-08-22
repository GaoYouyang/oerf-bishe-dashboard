#!/usr/bin/env python3
"""Build the public v190 geometry-QDEIM coreset capacity figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_geometry_qdeim_coreset_capacity_v190_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_geometry_qdeim_coreset_capacity_v190.png"
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
        size=26,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def strict_cells_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], data: dict) -> None:
    rows = (
        ("Five cameras", data["v189_five_camera_k1_strict_safe"], data["v190_five_camera_k1_strict_safe"]),
        ("All nine", data["v189_all_nine_k1_strict_safe"], data["v190_all_nine_k1_strict_safe"]),
    )
    left = box[0] + 56
    right = box[2] - 50
    for index, (name, full_count, subset_count) in enumerate(rows):
        top = box[1] + 135 + index * 250
        label(draw, (left, top), name, size=21, fill="#31434c", bold=True, anchor="lm")
        label(draw, (left, top + 62), f"Full DCT  {full_count}/52", size=22, fill="#287f72", bold=True, anchor="lm")
        label(draw, (right, top + 62), f"QDEIM1280  {subset_count}/52", size=22, fill="#a84e45", bold=True, anchor="rm")
        line_y = top + 126
        draw.rounded_rectangle((left, line_y, right, line_y + 20), radius=6, fill="#e3e9e9")
        full_x = left + (right - left) * full_count / 52
        subset_x = left + (right - left) * subset_count / 52
        draw.rounded_rectangle((left, line_y, full_x, line_y + 20), radius=6, fill="#287f72")
        draw.ellipse((subset_x - 12, line_y - 4, subset_x + 12, line_y + 24), fill="#a84e45")
    label(draw, ((left + right) / 2, box[3] - 86), "Unchanged physical CGLS K1", size=18, fill="#65747b", bold=True, anchor="mm")


def gate_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], data: dict) -> None:
    metrics = (
        ("Field", "field_p90", 0.5, "#3d6d9a"),
        ("Gradient", "gradient_p90", 0.75, "#287f72"),
        ("Observation", "observation_p90", 0.2, "#a66d2c"),
    )
    arms = (("Five", data["five_camera"]), ("Nine", data["all_nine"]))
    plot_left = box[0] + 170
    plot_right = box[2] - 42
    for metric_index, (name, key, gate, color) in enumerate(metrics):
        top = box[1] + 128 + metric_index * 174
        label(draw, (box[0] + 30, top + 47), name, size=18, fill="#31434c", bold=True, anchor="lm")
        for arm_index, (arm_name, payload) in enumerate(arms):
            yy = top + arm_index * 62
            ratio = payload[key] / gate
            draw.rounded_rectangle((plot_left, yy, plot_right, yy + 38), radius=5, fill="#e8edef")
            width = (plot_right - plot_left) * min(ratio / 1.2, 1.0)
            draw.rounded_rectangle((plot_left, yy, plot_left + width, yy + 38), radius=5, fill=color)
            gate_x = plot_left + (plot_right - plot_left) / 1.2
            draw.line((gate_x, yy - 5, gate_x, yy + 43), fill="#9e3f39", width=3)
            label(draw, (plot_left - 12, yy + 19), arm_name, size=14, fill="#65747b", bold=True, anchor="rm")
            label(draw, (plot_right - 8, yy + 19), f"{payload[key]:.3f}  ({ratio:.2f}x gate)", size=14, fill="#17232d", bold=True, anchor="rm")
    label(draw, ((box[0] + box[2]) / 2, box[3] - 86), "Five gradient and nine observation cross the frozen p90 gates", size=16, fill="#a84e45", bold=True, anchor="mm")


def diagnostics_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], payload: dict) -> None:
    center = (box[0] + box[2]) / 2
    label(draw, (center, box[1] + 148), "FAIL", size=66, fill="#a84e45", bold=True, anchor="mm")
    label(draw, (center, box[1] + 214), "fixed subset capacity", size=18, fill="#65747b", anchor="mm")
    items = [
        ("selected coordinates", "1280", "#3d6d9a"),
        ("five reduction", "55.48%", "#3d6d9a"),
        ("nine reduction", "75.27%", "#3d6d9a"),
        ("retained rank", "1009 / 1009", "#287f72"),
        ("selected condition", "262–652", "#a84e45"),
        ("full-DCT condition", "49–187", "#287f72"),
        ("independent checks", "59 / 59", "#287f72"),
        ("algorithm breakthrough", "false", "#a66d2c"),
    ]
    for index, (name, value, color) in enumerate(items):
        yy = box[1] + 292 + index * 61
        label(draw, (box[0] + 34, yy), name, size=15, fill="#53636b", anchor="lm")
        label(draw, (box[2] - 34, yy), value, size=17, fill=color, bold=True, anchor="rm")
        if index < len(items) - 1:
            draw.line((box[0] + 34, yy + 28, box[2] - 34, yy + 28), fill="#e3e9e9", width=2)


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    label(draw, (WIDTH / 2, 56), "v190 Fixed geometry QDEIM1280 does not preserve full-DCT capacity", size=39, fill="#17232d", bold=True, anchor="mm")
    label(draw, (WIDTH / 2, 108), "Algebraic rank survives; held-out physical accuracy and tails do not", size=22, fill="#596772", anchor="mm")

    cells_box = (70, 165, 815, 925)
    gates_box = (855, 165, 1760, 925)
    diag_box = (1800, 165, 2330, 925)
    panel(draw, cells_box, "A. K1 strict-safe cells")
    panel(draw, gates_box, "B. QDEIM1280 K1 p90 / gate")
    panel(draw, diag_box, "C. Compression audit")
    strict_cells_panel(draw, cells_box, payload["comparison_to_v189"])
    gate_panel(draw, gates_box, payload["primary_k1"])
    diagnostics_panel(draw, diag_box, payload)

    draw.rounded_rectangle((70, 980, 2330, 1170), radius=8, fill="#20313d")
    label(draw, (1200, 1028), "Scientific interpretation", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    label(draw, (1200, 1078), "A fixed 1280-column geometry subset loses the complete-DCT physical inverse despite preserving response rank.", size=21, fill="#ffffff", bold=True, anchor="mm")
    label(draw, (1200, 1128), "Close this fixed subset family; no budget increase, predictor, resource, GPU, or breakthrough claim.", size=18, fill="#d8e2e5", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
