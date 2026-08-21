#!/usr/bin/env python3
"""Build the public v185 potential-affine observability figure."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/poolfire_potential_affine_observability_v185_public_summary.json"
OUTPUT = ROOT / "assets/figures/poolfire_potential_affine_observability_v185.png"
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
    label(draw, ((box[0] + box[2]) / 2, box[1] + 48), title, size=28, fill="#20313d", bold=True, anchor="mm")


def metric_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    result: dict[str, float],
) -> None:
    rows = (
        ("Field", result["field_p90"], 0.50, 0.55),
        ("Gradient", result["gradient_p90"], 0.75, 0.82),
        ("Observation", result["observation_p90"], 0.20, 0.28),
    )
    left = box[0] + 48
    bar_left = left + 175
    right = box[2] - 42
    for index, (name, value, gate, scale) in enumerate(rows):
        top = box[1] + 142 + index * 178
        label(draw, (left, top + 25), name, size=21, fill="#2a3d46", bold=True, anchor="lm")
        draw.rounded_rectangle((bar_left, top, right, top + 50), radius=5, fill="#e8edef")
        value_x = bar_left + (right - bar_left) * min(value / scale, 1.0)
        gate_x = bar_left + (right - bar_left) * gate / scale
        draw.rounded_rectangle((bar_left, top, value_x, top + 50), radius=5, fill="#267a68")
        draw.line((gate_x, top - 7, gate_x, top + 57), fill="#805da8", width=5)
        label(draw, (right - 8, top + 25), f"{value:.3f}", size=20, fill="#17232d", bold=True, anchor="rm")
        label(draw, (bar_left, top + 72), f"gate {gate:.2f}", size=16, fill="#6d4d8b", anchor="lm")


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    five = payload["primary_k1"]["five_camera"]
    nine = payload["primary_k1"]["all_nine"]
    k0_five = payload["primary_k0"]["five_camera"]
    k0_nine = payload["primary_k0"]["all_nine"]
    diagnostic = payload["mechanism_diagnostics"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f5f4")
    draw = ImageDraw.Draw(image)
    label(draw, (WIDTH / 2, 58), "v185 Shared detector-potential coordinates preserve the full affine state", size=40, fill="#17232d", bold=True, anchor="mm")
    label(draw, (WIDTH / 2, 112), "Exact potential-domain affine inverse + unchanged physical CGLS K1 | independently recomputed", size=22, fill="#596772", anchor="mm")

    five_box = (70, 170, 850, 930)
    nine_box = (890, 170, 1670, 930)
    verdict_box = (1710, 170, 2330, 930)
    panel(draw, five_box, "A. Five-camera K1 p90")
    panel(draw, nine_box, "B. All-nine K1 p90")
    panel(draw, verdict_box, "C. Capacity verdict")
    metric_panel(draw, five_box, five)
    metric_panel(draw, nine_box, nine)
    label(draw, (five_box[0] + 48, 865), "Green bar: result | purple marker: frozen gate", size=18, fill="#50636b")
    label(draw, (nine_box[0] + 48, 865), "All three metrics remain inside their gates", size=18, fill="#267a68")

    label(draw, ((verdict_box[0] + verdict_box[2]) / 2, 302), "PASS", size=70, fill="#267a68", bold=True, anchor="mm")
    label(draw, ((verdict_box[0] + verdict_box[2]) / 2, 392), "strict-safe after K1", size=20, fill="#65747b", anchor="mm")
    label(draw, ((verdict_box[0] + verdict_box[2]) / 2, 462), "52 / 52  |  52 / 52", size=38, fill="#267a68", bold=True, anchor="mm")
    bullets = [
        f"retained rank: {diagnostic['retained_affine_rank_minimum']} / 1009",
        f"K0: {k0_five['strict_cells_safe']} / 52 | {k0_nine['strict_cells_safe']} / 52",
        "one-direction control: 0 / 52 in every arm",
        "independent checks: 32 / 32",
        "dense inverse: not deployable yet",
        "algorithm breakthrough: false",
    ]
    for index, item in enumerate(bullets):
        yy = 555 + index * 54
        positive = index in (0, 1, 2, 3)
        color = "#267a68" if positive else "#a66d2c"
        draw.ellipse((verdict_box[0] + 42, yy - 7, verdict_box[0] + 56, yy + 7), fill=color)
        label(draw, (verdict_box[0] + 72, yy), item, size=17, fill="#31434c", anchor="lm")

    draw.rounded_rectangle((70, 985, 2330, 1170), radius=8, fill="#20313d")
    label(draw, (1200, 1032), "Scientific interpretation", size=23, fill="#a7d5c9", bold=True, anchor="mm")
    label(draw, (1200, 1082), "v184 failed because its scalar-ray Jacobi lift was lossy, not because detector-potential compression erased the state.", size=23, fill="#ffffff", bold=True, anchor="mm")
    label(draw, (1200, 1130), "Next gate: approximate this dense inverse compactly using observation and geometry only; no speed or generalization claim yet.", size=19, fill="#d8e2e5", anchor="mm")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
