"""Tests for the PSU support-envelope post-open figure."""

from __future__ import annotations

import json
from pathlib import Path

from site_tools.plot_psu_b0_support_envelope_diagnosis import render_figure


def test_render_support_envelope_figure(tmp_path: Path) -> None:
    summary = json.loads(
        Path(
            "docs/psu_b0_support_envelope_postopen_public_summary.json"
        ).read_text(encoding="utf-8")
    )
    manifest = render_figure(summary, tmp_path / "figure")
    assert manifest["status"] == "POSTOPEN_SUPPORT_ENVELOPE_FIGURE_COMPLETE"
    assert manifest["claim_boundary"]["fresh_candidate_gate"] is False
    assert manifest["claim_boundary"]["family_risk_resolved"] is False
    assert set(manifest["output_files"]) == {
        "figure.png",
        "figure.pdf",
        "figure.svg",
    }
