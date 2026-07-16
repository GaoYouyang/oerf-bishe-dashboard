"""Tests for the PSU spectral-preconditioner no-go figure."""

from __future__ import annotations

import json
from pathlib import Path

from site_tools.plot_psu_b0_spectral_preconditioner_pilot import render_figure


def test_render_frozen_no_go_figure(tmp_path: Path) -> None:
    summary = json.loads(
        Path(
            "docs/psu_b0_spectral_preconditioner_pilot_public_summary.json"
        ).read_text(encoding="utf-8")
    )
    manifest = render_figure(summary, tmp_path / "figure")
    assert manifest["status"] == "FIGURE_COMPLETE_FROZEN_SYNTHETIC_NO_GO"
    assert manifest["claim_boundary"]["joint_ood_gate_passed"] is False
    assert set(manifest["output_files"]) == {
        "figure.png",
        "figure.pdf",
        "figure.svg",
    }
