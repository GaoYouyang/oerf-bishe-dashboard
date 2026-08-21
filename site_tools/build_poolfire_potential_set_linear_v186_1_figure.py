#!/usr/bin/env python3
"""Build the public v186.1 shared-linear verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_potential_set_linear_v186_1_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_potential_set_linear_v186_1.png"
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
    five_k0: int,
    nine_k0: int,
    five_k1: int,
    nine_k1: int,
) -> None:
    rows = (
        ("Five cameras K0", five_k0, "#a9b6bb"),
        ("All nine K0", nine_k0, "#a9b6bb"),
        ("Five cameras K1", five_k1, "#287f72"),
        ("All nine K1", nine_k1, "#3d6d9a"),
    )
    left = box[0] + 46
    bar_left = left + 225
    right = box[2] - 48
    for index, (name, value, color) in enumerate(rows):
        top = box[1] + 132 + index * 136
        label(draw, (left, top + 25), name, size=18, fill="#2f4149", bold=True, anchor="lm")
        draw.rounded_rectangle((bar_left, top, right, top + 50), radius=5, fill="#e8edef")
        value_x = bar_left + (right - bar_left) * value / 52.0
        if value:
            draw.rounded_rectangle((bar_left, top, value_x, top + 50), radius=5, fill=color)
        label(draw, (right - 7, top + 25), f"{value} / 52", size=19, fill="#17232d", bold=True, anchor="rm")
    label(draw, (left, box[3] - 78), "Frozen requirement: 52 / 52", size=18, fill="#a84e45", bold=True)
    label(draw, (left, box[3] - 42), "K1 improves the map, but neither sensor arm is complete.", size=17, fill="#5e6b72")


def observation_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    five: list[float],
    nine: list[float],
) -> None:
    plot = (box[0] + 86, box[1] + 120, box[2] - 54, box[3] - 116)
    ymin, ymax = 0.18, 0.27

    def point(index: int, value: float) -> tuple[float, float]:
        x = plot[0] + index * (plot[2] - plot[0]) / 3
        y = plot[3] - (value - ymin) * (plot[3] - plot[1]) / (ymax - ymin)
        return x, y

    for tick in (0.18, 0.20, 0.22, 0.24, 0.26):
        yy = point(0, tick)[1]
        draw.line((plot[0], yy, plot[2], yy), fill="#dce3e5", width=2)
        label(draw, (plot[0] - 14, yy), f"{tick:.2f}", size=15, fill="#68777e", anchor="rm")
    gate_y = point(0, 0.20)[1]
    draw.line((plot[0], gate_y, plot[2], gate_y), fill="#a84e45", width=5)
    label(draw, (plot[2], gate_y - 17), "gate 0.20", size=16, fill="#a84e45", bold=True, anchor="rm")

    times = ("0", "0.25", "0.75", "1")
    for index, time_label in enumerate(times):
        xx, _ = point(index, ymin)
        label(draw, (xx, plot[3] + 35), f"t={time_label}", size=16, fill="#4e6068", anchor="mm")

    for values, color in ((five, "#287f72"), (nine, "#3d6d9a")):
        points = [point(index, value) for index, value in enumerate(values)]
        draw.line(points, fill=color, width=6, joint="curve")
        for xx, yy in points:
            draw.ellipse((xx - 9, yy - 9, xx + 9, yy + 9), fill=color, outline="#ffffff", width=3)

    label(draw, (plot[0], box[3] - 60), "Five cameras", size=17, fill="#287f72", bold=True)
    label(draw, (plot[0] + 178, box[3] - 60), "All nine", size=17, fill="#3d6d9a", bold=True)
    label(draw, (plot[2], box[3] - 60), "Complete time strata: 0 / 4 in both arms", size=17, fill="#a84e45", bold=True, anchor="ra")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    k0 = payload["primary_k0"]
    k1 = payload["primary_k1"]
    independent = payload["independent_recomputation"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    label(draw, (WIDTH / 2, 58), "v186.1 Shared DCT12 + Plucker linear approximation fails", size=40, fill="#17232d", bold=True, anchor="mm")
    label(draw, (WIDTH / 2, 112), "Deployment-visible set features + unchanged physical CGLS K1 | independently recomputed", size=22, fill="#596772", anchor="mm")

    strict_box = (70, 170, 1020, 930)
    obs_box = (1060, 170, 1880, 930)
    verdict_box = (1920, 170, 2330, 930)
    panel(draw, strict_box, "A. Strict-safe heldout cells")
    panel(draw, obs_box, "B. K1 observation p90 by time")
    panel(draw, verdict_box, "C. Verdict")
    strict_safe_panel(
        draw,
        strict_box,
        k0["five_camera"]["strict_cells_safe"],
        k0["all_nine"]["strict_cells_safe"],
        k1["five_camera"]["strict_cells_safe"],
        k1["all_nine"]["strict_cells_safe"],
    )
    observation_panel(
        draw,
        obs_box,
        k1["five_camera"]["observation_p90_by_time"],
        k1["all_nine"]["observation_p90_by_time"],
    )

    center = (verdict_box[0] + verdict_box[2]) / 2
    label(draw, (center, 305), "FAIL", size=68, fill="#a84e45", bold=True, anchor="mm")
    label(draw, (center, 380), "complete trajectory", size=20, fill="#65747b", anchor="mm")
    bullets = [
        "K1: 39/52 | 25/52",
        "complete times: 0/4 | 0/4",
        "controls: 0/52",
        f"independent: {independent['checks_passed']}/{independent['checks_total']}",
        "camera permutation: exact",
        "GPU authorized: false",
        "algorithm breakthrough: false",
    ]
    for index, item in enumerate(bullets):
        yy = 475 + index * 55
        color = "#287f72" if index in (3, 4) else "#a84e45" if index in (0, 1, 2) else "#a66d2c"
        draw.ellipse((verdict_box[0] + 30, yy - 7, verdict_box[0] + 44, yy + 7), fill=color)
        label(draw, (verdict_box[0] + 58, yy), item, size=15, fill="#31434c", bold=index < 2, anchor="lm")

    draw.rounded_rectangle((70, 985, 2330, 1170), radius=8, fill="#20313d")
    label(draw, (1200, 1033), "Scientific interpretation", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    label(draw, (1200, 1082), "The compact map recovers useful field and gradient structure, but observation tails remain incompatible.", size=23, fill="#ffffff", bold=True, anchor="mm")
    label(draw, (1200, 1130), "Close this DCT12 + Plucker shared-linear representation; do not enlarge it or claim call, speed, or real-BOST gains.", size=19, fill="#d8e2e5", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
