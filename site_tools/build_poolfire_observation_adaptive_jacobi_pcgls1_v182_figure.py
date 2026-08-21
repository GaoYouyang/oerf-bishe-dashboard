#!/usr/bin/env python3
"""Build the public v182 observation-adaptive Jacobi-PCGLS1 verdict figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_observation_adaptive_jacobi_pcgls1_v182_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_observation_adaptive_jacobi_pcgls1_v182.png"
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
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#c7d2d7", width=2)
    _text(draw, ((box[0] + box[2]) / 2, box[1] + 50), title, size=27, fill="#20313d", bold=True, anchor="mm")


def _observation_bar(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    index: int,
    label: str,
    k0: float,
    k1: float,
    gate: float,
) -> None:
    left = box[0] + 48
    right = box[2] - 48
    y = box[1] + 150 + index * 300
    bar_left = left + 145
    scale_max = 0.45
    _text(draw, (left, y - 32), label, size=23, fill="#334750", bold=True)
    for row, (name, value, color) in enumerate((("K0", k0, "#727f87"), ("K1", k1, "#267a68"))):
        row_y = y + row * 78
        draw.rounded_rectangle((bar_left, row_y, right, row_y + 48), radius=4, fill="#e5ebed")
        width = (right - bar_left) * min(value / scale_max, 1.0)
        draw.rounded_rectangle((bar_left, row_y, bar_left + width, row_y + 48), radius=4, fill=color)
        gate_x = bar_left + (right - bar_left) * gate / scale_max
        draw.line((gate_x, row_y - 8, gate_x, row_y + 56), fill="#b64d43", width=4)
        _text(draw, (bar_left - 16, row_y + 24), name, size=20, fill="#5d6c73", bold=True, anchor="rm")
        _text(draw, (right - 10, row_y + 24), f"{value:.3f}", size=21, fill="#17232d", bold=True, anchor="rm")
    reduction = 100.0 * (k0 - k1) / k0
    _text(draw, (bar_left, y + 174), f"K1 reduction {reduction:.1f}% | frozen gate {gate:.2f}", size=18, fill="#66757d")


def _normalized_metric(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    index: int,
    name: str,
    five: float,
    nine: float,
    gate: float,
) -> None:
    y = box[1] + 138 + index * 205
    left = box[0] + 52
    bar_left = left + 170
    right = box[2] - 52
    _text(draw, (left, y + 30), name, size=21, fill="#334750", bold=True, anchor="lm")
    for row, (sensor, value, color) in enumerate((("five", five, "#315f91"), ("nine", nine, "#946d22"))):
        row_y = y + row * 62
        ratio = value / gate
        draw.rounded_rectangle((bar_left, row_y, right, row_y + 36), radius=3, fill="#e5ebed")
        width = (right - bar_left) * min(ratio / 1.6, 1.0)
        draw.rounded_rectangle((bar_left, row_y, bar_left + width, row_y + 36), radius=3, fill=color)
        gate_x = bar_left + (right - bar_left) / 1.6
        draw.line((gate_x, row_y - 4, gate_x, row_y + 40), fill="#b64d43", width=3)
        _text(draw, (bar_left - 12, row_y + 18), sensor, size=16, fill="#69777e", anchor="rm")
        _text(draw, (right - 8, row_y + 18), f"{value:.3f} ({ratio:.2f}x)", size=17, fill="#17232d", bold=True, anchor="rm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    five0 = payload["five_camera_k0"]["global_p90"]
    nine0 = payload["all_nine_k0"]["global_p90"]
    five1 = payload["five_camera_primary_k1"]["global_p90"]
    nine1 = payload["all_nine_primary_k1"]["global_p90"]
    gates = payload["absolute_gate"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    _text(draw, (WIDTH / 2, 58), "v182 Observable Jacobi-PCGLS1: residual improves, gate still fails", size=39, fill="#17232d", bold=True, anchor="mm")
    _text(draw, (WIDTH / 2, 112), "Exact observation line minimization + unchanged physical CGLS K1 | no tuning or target truth", size=22, fill="#596772", anchor="mm")

    left_box = (70, 175, 1110, 1045)
    middle_box = (1150, 175, 1915, 1045)
    right_box = (1955, 175, 2450, 1045)
    _panel(draw, left_box, "A. Observation p90: K0 to K1")
    _panel(draw, middle_box, "B. K1 p90 relative to frozen gates")
    _panel(draw, right_box, "C. Independent boundary")

    _observation_bar(
        draw,
        box=left_box,
        index=0,
        label="Five cameras",
        k0=five0["observation_relative_l2"],
        k1=five1["observation_relative_l2"],
        gate=gates["observation_p90_max"],
    )
    _observation_bar(
        draw,
        box=left_box,
        index=1,
        label="All nine cameras",
        k0=nine0["observation_relative_l2"],
        k1=nine1["observation_relative_l2"],
        gate=gates["observation_p90_max"],
    )

    _normalized_metric(draw, box=middle_box, index=0, name="Field", five=five1["field_relative_l2"], nine=nine1["field_relative_l2"], gate=gates["field_p90_max"])
    _normalized_metric(draw, box=middle_box, index=1, name="Gradient", five=five1["gradient_relative_l2"], nine=nine1["gradient_relative_l2"], gate=gates["gradient_p90_max"])
    _normalized_metric(draw, box=middle_box, index=2, name="Observation", five=five1["observation_relative_l2"], nine=nine1["observation_relative_l2"], gate=gates["observation_p90_max"])

    items = [
        ("Strict-safe cells", "five 0/52 | nine 0/52", "#b64d43"),
        ("Complete calibrations", "five 0/13 | nine 0/13", "#b64d43"),
        ("Complete frames", "five 0/4 | nine 0/4", "#b64d43"),
        ("Logical K1 calls", "3A + 2A^T", "#20313d"),
        ("Independent checks", "47/47", "#267a68"),
        ("Candidate max diff", "5.07e-12", "#20313d"),
        ("Metric max diff", "8.38e-13", "#20313d"),
        ("Matched accuracy", "FAIL", "#b64d43"),
    ]
    for index, (label, value, color) in enumerate(items):
        y = right_box[1] + 118 + index * 98
        _text(draw, (right_box[0] + 38, y), label, size=17, fill="#68767d", bold=True)
        _text(draw, (right_box[0] + 38, y + 34), value, size=22, fill=color, bold=True)

    draw.rounded_rectangle((70, 1088, 2450, 1255), radius=8, fill="#fff8ed", outline="#d6b66f", width=2)
    _text(draw, (WIDTH / 2, 1138), "Scientific decision: FAIL_OBSERVATION_ADAPTIVE_JACOBI_PCGLS1_V182", size=27, fill="#8a493f", bold=True, anchor="mm")
    _text(draw, (WIDTH / 2, 1192), "Field and gradient pass, but observation p90 remains above 0.20 under both sensor arms.", size=22, fill="#5b4d45", anchor="mm")
    _text(draw, (WIDTH / 2, 1230), "Close this one-step diagonal mechanism | no call-reduction, resource, external, or real-BOST claim", size=18, fill="#6f625a", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
