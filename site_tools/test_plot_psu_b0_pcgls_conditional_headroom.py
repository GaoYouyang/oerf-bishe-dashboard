from __future__ import annotations

import json
from pathlib import Path

from site_tools.plot_psu_b0_pcgls_conditional_headroom import build_figure


ROOT = Path(__file__).resolve().parents[1]


def test_build_headroom_figure(tmp_path: Path) -> None:
    report = json.loads(
        (
            ROOT
            / "docs/psu_b0_pcgls_conditional_headroom_public_summary.json"
        ).read_text(encoding="utf-8")
    )
    output = tmp_path / "headroom.png"
    build_figure(report, output)
    assert output.exists()
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".svg").exists()
