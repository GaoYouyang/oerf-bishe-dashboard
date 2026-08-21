#!/usr/bin/env python3
"""Build the public v177 five-camera Krylov capacity figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_five_camera_krylov_capacity_v177_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_five_camera_krylov_capacity_v177.png"
WIDTH = 2520
HEIGHT = 1320


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
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


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#cbd6da", width=2)
    _text(
        draw,
        ((box[0] + box[2]) / 2, box[1] + 48),
        title,
        size=27,
        fill="#20313d",
        bold=True,
        anchor="mm",
    )


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    k4 = payload["five_camera_zero_cgls_k4"]
    k8 = payload["five_camera_zero_cgls_k8"]
    nine = payload["nine_camera_zero_cgls_k4"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 58),
        "v177 exhaustive Krylov reference-capacity diagnostic",
        size=44,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 112),
        "13 calibrations x 4 frames x all 126 five-camera subsets | post-open mechanism evidence",
        size=22,
        fill="#596772",
        anchor="mm",
    )

    left_box = (70, 175, 805, 1030)
    mid_box = (845, 175, 1690, 1030)
    right_box = (1730, 175, 2450, 1030)
    _panel(draw, left_box, "A. Joint strict-safe capacity")
    _panel(draw, mid_box, "B. Per-metric cellwise feasibility")
    _panel(draw, right_box, "C. Cellwise-witness p90")

    capacity_rows = [
        ("5-camera K4 candidates", k4["strict_safe_candidates"], k4["strict_safe_candidate_total"]),
        ("5-camera K4 cells", k4["cellwise_safe_cells"], k4["cellwise_safe_total"]),
        ("5-camera K8 candidates", k8["strict_safe_candidates"], k8["strict_safe_candidate_total"]),
        ("5-camera K8 cells", k8["cellwise_safe_cells"], k8["cellwise_safe_total"]),
        ("9-camera K4 cells", nine["strict_safe_cells"], nine["strict_safe_total"]),
    ]
    for index, (label, value, total) in enumerate(capacity_rows):
        y = 285 + index * 135
        draw.rounded_rectangle(
            (left_box[0] + 55, y, left_box[2] - 45, y + 92),
            radius=7,
            fill="#f7f9f9",
            outline="#d6dfe2",
            width=2,
        )
        _text(draw, (left_box[0] + 78, y + 32), label, size=19, fill="#42545f", bold=True, anchor="lm")
        _text(draw, (left_box[2] - 72, y + 32), f"{value}/{total}", size=23, fill="#b64d43", bold=True, anchor="rm")
        _text(draw, (left_box[0] + 78, y + 67), "No jointly passing witness", size=16, fill="#7d4b48", anchor="lm")

    groups = [
        ("5-camera K4", k4["per_metric_cellwise_oracle_pass"]),
        ("5-camera K8", k8["per_metric_cellwise_oracle_pass"]),
        ("9-camera K4", nine["per_metric_pass"]),
    ]
    metric_rows = [("Field", "field", "#b64d43"), ("Gradient", "gradient", "#267a68"), ("Observation", "observation", "#2d6fa3")]
    chart_left = mid_box[0] + 70
    chart_right = mid_box[2] - 50
    for group_index, (group_label, values) in enumerate(groups):
        group_y = 278 + group_index * 235
        _text(draw, (chart_left, group_y), group_label, size=22, fill="#344853", bold=True, anchor="lm")
        for metric_index, (metric_label, key, color) in enumerate(metric_rows):
            y = group_y + 43 + metric_index * 55
            _text(draw, (chart_left, y + 17), metric_label, size=17, fill="#52636d", anchor="lm")
            bar_left = chart_left + 145
            draw.rectangle((bar_left, y, chart_right, y + 34), fill="#e7ecef")
            width = (chart_right - bar_left) * values[key] / values["total"]
            if width > 0:
                draw.rectangle((bar_left, y, bar_left + width, y + 34), fill=color)
            _text(draw, (chart_right - 8, y + 17), f'{values[key]}/{values["total"]}', size=16, fill="#20313d", bold=True, anchor="rm")

    gate_rows = [
        ("Field", "field_relative_l2", 0.50, "#b64d43"),
        ("Gradient", "gradient_relative_l2", 0.75, "#267a68"),
        ("Observation", "observation_relative_l2", 0.20, "#2d6fa3"),
    ]
    sources = [("5-camera K4", k4["cellwise_witness_p90"]), ("5-camera K8", k8["cellwise_witness_p90"])]
    chart_left = right_box[0] + 58
    chart_right = right_box[2] - 45
    scale_max = 0.90
    for metric_index, (metric_label, key, gate, color) in enumerate(gate_rows):
        base_y = 285 + metric_index * 232
        _text(draw, (chart_left, base_y), metric_label, size=22, fill="#344853", bold=True, anchor="lm")
        _text(draw, (chart_right, base_y), f"gate {gate:.2f}", size=17, fill="#8d3c35", bold=True, anchor="rm")
        for source_index, (source_label, values) in enumerate(sources):
            y = base_y + 47 + source_index * 61
            value = values[key]
            _text(draw, (chart_left, y + 17), source_label, size=16, fill="#52636d", anchor="lm")
            bar_left = chart_left + 155
            draw.rectangle((bar_left, y, chart_right, y + 34), fill="#e7ecef")
            width = (chart_right - bar_left) * min(value / scale_max, 1.0)
            draw.rectangle((bar_left, y, bar_left + width, y + 34), fill=color)
            gate_x = bar_left + (chart_right - bar_left) * gate / scale_max
            draw.line((gate_x, y - 4, gate_x, y + 38), fill="#8d3c35", width=3)
            _text(draw, (chart_right - 8, y + 17), f"{value:.3f}", size=16, fill="#20313d", bold=True, anchor="rm")

    draw.rounded_rectangle((70, 1080, 2450, 1250), radius=8, fill="#ffffff", outline="#cbd6da", width=2)
    _text(
        draw,
        (1260, 1120),
        "K8 rescues gradient and observation, but field remains 0/52",
        size=27,
        fill="#7e403b",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1170),
        "The bottleneck is low-depth field-reference / representation adequacy, not subset ranking",
        size=21,
        fill="#4f5f68",
        anchor="mm",
    )
    _text(
        draw,
        (1260, 1217),
        "FAIL_BROADER_KRYLOV_REFERENCE_REPRESENTATION_V177 | 25/25 independent checks | breakthrough=false",
        size=18,
        fill="#b64d43",
        bold=True,
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
