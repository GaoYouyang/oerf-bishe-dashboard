from __future__ import annotations

import json
from pathlib import Path

from site_tools.validate_psu_b0_residual_risk_fresh import _load


def test_validator_load_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    try:
        _load(path)
    except ValueError as error:
        assert "expected JSON object" in str(error)
    else:
        raise AssertionError("validator accepted a non-object report")
