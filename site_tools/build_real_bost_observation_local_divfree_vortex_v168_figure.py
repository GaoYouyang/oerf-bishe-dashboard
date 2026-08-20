#!/usr/bin/env python3
"""Build the public v168 local divergence-free vortex decision matrix."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_observation_local_divfree_vortex_v168_public_summary.json"
OUTPUT = ROOT / "assets/figures/real_bost_observation_local_divfree_vortex_v168.png"
WIDTH = 2520
HEIGHT = 1180


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
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


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = {(row["time"], row["camera_count"]): row for row in payload["primary"]["strata"]}

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f4f7f9")
    draw = ImageDraw.Draw(image)
    _text(
        draw,
        (WIDTH / 2, 62),
        "v168 local divergence-free vortex transport",
        size=44,
        fill="#17232d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 116),
        "gradient p90 / worst by normalized time and active-camera count | gates <= 0.750 / 1.000",
        size=25,
        fill="#596772",
        anchor="mm",
    )

    left, top = 320, 220
    cell_w, cell_h = 590, 175
    gap_x, gap_y = 34, 30
    for column, camera_count in enumerate([5, 7, 9]):
        x = left + column * (cell_w + gap_x) + cell_w / 2
        _text(
            draw,
            (x, top - 52),
            f"{camera_count} cameras",
            size=29,
            fill="#344653",
            bold=True,
            anchor="mm",
        )

    for row_index, time_value in enumerate([0.0, 0.25, 0.75, 1.0]):
        y0 = top + row_index * (cell_h + gap_y)
        _text(
            draw,
            (left - 48, y0 + cell_h / 2),
            f"t = {time_value:.2f}",
            size=27,
            fill="#344653",
            bold=True,
            anchor="rm",
        )
        for column, camera_count in enumerate([5, 7, 9]):
            item = rows[(time_value, camera_count)]
            x0 = left + column * (cell_w + gap_x)
            box = (x0, y0, x0 + cell_w, y0 + cell_h)
            passed = item["passed"]
            fill = "#e8f5ef" if passed else "#fff0e3"
            outline = "#2f8069" if passed else "#c85332"
            draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=4)
            _text(draw, (x0 + 34, y0 + 30), "PASS" if passed else "FAIL", size=25, fill=outline, bold=True)
            _text(
                draw,
                (x0 + cell_w / 2, y0 + 91),
                f"{item['gradient_p90']:.3f} / {item['gradient_worst']:.3f}",
                size=38,
                fill="#20313d",
                bold=True,
                anchor="mm",
            )
            _text(
                draw,
                (x0 + cell_w / 2, y0 + 140),
                f"field p90 {item['field_p90']:.3f}  |  obs p90 {item['observation_p90']:.3f}",
                size=20,
                fill="#5c6973",
                anchor="mm",
            )

    draw.rounded_rectangle((190, 1040, 2330, 1135), radius=8, fill="#ffffff", outline="#d7e0e7", width=2)
    _text(
        draw,
        (WIDTH / 2, 1069),
        "Independent verdict: 10/12 pass; the exact local divergence-free vortex family closes.",
        size=25,
        fill="#7a3f2d",
        bold=True,
        anchor="mm",
    )
    _text(
        draw,
        (WIDTH / 2, 1110),
        "t=0.75 / 5 cameras: v168 0.818 / 1.158 | v167 0.813 / 1.146 | frozen H1 0.759 / 0.836 | 60/60 checks",
        size=20,
        fill="#66737e",
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
