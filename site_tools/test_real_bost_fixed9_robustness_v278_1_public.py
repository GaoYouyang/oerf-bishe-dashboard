from __future__ import annotations

import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/real_bost_fixed9_robustness_v278_1_public_summary.json"
RESULT = ROOT / "docs/real_bost_fixed9_robustness_v278_1_result_2026-08-29.md"
FIGURE = ROOT / "assets/figures/real_bost_fixed9_robustness_v278_1.png"
CURRENT = ROOT / "operator-learning/current-evidence.json"
PAGES = (ROOT / "index.html", ROOT / "operator-learning/index.html", ROOT / "operator-learning/daily-progress.html")
LOG = ROOT / "docs/operator_3d_learning_log.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def test_v278_1_reference_first_decision_is_preserved() -> None:
    data = _summary()
    assert data["coverage"]["cells"] == 2340
    assert data["coverage"]["scored_rows"] == 9360
    assert data["reference_adequacy"]["passed_strata"] == 12
    assert data["reference_adequacy"]["total_strata"] == 20
    assert data["reference_adequacy"]["passed"] is False
    assert data["diagnostic_only"]["candidate_adjudication_authorized"] is False
    assert data["scientific_decision"].startswith("INCONCLUSIVE_REFERENCE_INADEQUATE")


def test_v278_1_independent_validation_and_claim_boundaries() -> None:
    data = _summary()
    assert data["independent_validation"]["passed"] is True
    assert (data["independent_validation"]["checks_passed"], data["independent_validation"]["checks_total"]) == (21, 21)
    assert data["paired_experimental_displacement_used"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v278_1_result_is_bilingual_and_bounded() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v278.1：" in text and "# v278.1:" in text
    assert "2,340" in text and "9,360" in text and "21/21" in text
    assert "受控虚拟 BOS" in text and "controlled virtual-BOS" in text
    assert "real_bost=false" in text and "algorithm_breakthrough=false" in text


def test_v278_1_figure_is_readable() -> None:
    assert FIGURE.exists() and FIGURE.stat().st_size > 50_000
    width, height = _png_dimensions(FIGURE)
    assert width >= 2000
    assert height >= 900


def test_v278_1_is_synchronized_once() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert _summary()["scientific_decision"] == "INCONCLUSIVE_REFERENCE_INADEQUATE_FIXED9_ROBUSTNESS_V278_1"
    assert current["metrics"]["v278_1_cells"] == 2340
    assert current["current_decision"]["v278_1_reference_adequate"] is False
    for path in PAGES:
        assert "real_bost_fixed9_robustness_v278_1" in path.read_text(encoding="utf-8")
    assert "## 2026-08-29：v278.1" in LOG.read_text(encoding="utf-8")
    daily = (ROOT / "operator-learning/daily-progress.html").read_text(encoding="utf-8")
    assert daily.count('data-date="2026-08-29"') == 1


def test_v278_1_public_artifacts_exclude_private_details() -> None:
    new_text = "\n".join(path.read_text(encoding="utf-8") for path in (SUMMARY, RESULT))
    assert all(token not in new_text for token in ("/Users/", "/Volumes/", "private_results", "feafd0c7", "9283859e"))
    integrated_text = "\n".join(path.read_text(encoding="utf-8") for path in (*PAGES, LOG))
    assert "feafd0c7" not in integrated_text
    assert "9283859e" not in integrated_text
