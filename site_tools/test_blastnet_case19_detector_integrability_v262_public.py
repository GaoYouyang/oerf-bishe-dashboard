from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_detector_integrability_no_go_v262_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_detector_integrability_no_go_v262_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_detector_integrability_no_go_v262.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_detector_integrability_v262_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self,
        _tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def test_v262_summary_records_independent_no_go_gate() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 24
    assert data["projector"]["rank"] == 255
    assert data["scope"]["new_exact_calls_A"] == 0
    assert data["scope"]["new_exact_calls_At"] == 0
    for source in data["sources"].values():
        assert source["invariant_camera_blocks"] == 0
        assert source["camera_blocks"] == 117
    assert data["sources"]["residual_k14"]["removed_energy_fraction_p90_higher"] > 0.55
    assert data["adjudication"]["projected_residual_candidate_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v262_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v262：" in text and "# v262:" in text
    for token in ("24/24", "0/117", "0.74660", "0.55742"):
        assert token in text
    assert "不生成候选场" in text
    assert "creates no candidate field" in text
    assert "algorithm_breakthrough=false" in text


def test_v262_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v262_remains_preserved_after_v263_1_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    assert current["scientific_status"] == "FAIL_CASE19_TWO_COLOR_ADDITIVE_SCHWARZ_V267"
    assert current["headline_zh"].startswith("v267")
    assert current["headline_en"].startswith("v267")
    assert current["metrics"]["v262_independent_checks_passed"] == 24
    assert current["metrics"]["v262_observation_invariant_blocks"] == 0
    assert current["current_decision"]["v262_projected_residual_candidate_authorized"] is False
    assert current["current_decision"]["v262_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith(
        "blastnet_case19_two_color_additive_schwarz_v267.png"
    )
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_detector_integrability_no_go_v262" in text
        assert "24/24" in text and "0/117" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v262" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v262_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "checkpoint.pt",
        "run ID",
    )
    assert all(token not in text for token in forbidden)


def test_all_public_html_local_links_resolve_after_v262() -> None:
    missing: list[tuple[str, str]] = []
    for page in sorted(ROOT.rglob("*.html")):
        parser = _LinkParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for raw in parser.links:
            parsed = urlsplit(raw)
            if (
                parsed.scheme
                or parsed.netloc
                or raw.startswith(("#", "mailto:", "tel:", "data:", "javascript:"))
            ):
                continue
            relative = unquote(parsed.path)
            if not relative:
                continue
            target = ROOT / relative.lstrip("/") if relative.startswith("/") else page.parent / relative
            candidates = [target]
            if relative.endswith("/"):
                candidates.append(target / "index.html")
            elif not target.suffix:
                candidates.extend((target / "index.html", target.with_suffix(".html")))
            if not any(candidate.exists() for candidate in candidates):
                missing.append((str(page.relative_to(ROOT)), raw))
    assert missing == []
