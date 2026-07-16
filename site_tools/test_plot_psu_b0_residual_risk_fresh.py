from __future__ import annotations

import json
from pathlib import Path

from site_tools.plot_psu_b0_residual_risk_fresh import _load


def test_plot_loader_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    try:
        _load(path)
    except ValueError as error:
        assert "JSON object" in str(error)
    else:
        raise AssertionError("plot loader accepted an array")
