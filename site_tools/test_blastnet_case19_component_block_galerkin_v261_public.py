from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_component_block_galerkin_v261_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_component_block_galerkin_v261_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_component_block_galerkin_v261.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_component_block_galerkin_v261_figure.py"
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


def test_v261_summary_records_independent_negative_gate() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    primary = data["primary"]
    controls = data["controls"]
    assert data["scope"]["cells"] == 13
    assert data["mechanism"]["logical_exact_calls_A"] == 15
    assert data["mechanism"]["logical_exact_calls_At"] == 15
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 41
    assert primary["absolute_cells"] == 3
    assert primary["matched_cells"] == 0
    assert all(value > 1.05 for value in primary["matched_ratio_p90_higher"])
    assert primary["absolute_cells"] < controls["same_cost_k15"]["absolute_cells"]
    assert primary["absolute_cells"] < controls["cheaper_v258"]["absolute_cells"]
    assert data["adjudication"]["full_sequence_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v261_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v261：" in text and "# v261:" in text
    for token in ("41/41", "3/13", "0/13", "1.33611", "15A+15A^T"):
        assert token in text
    assert "不是算法" in text
    assert "not an algorithm" in text
    assert "algorithm_breakthrough=false" in text


def test_v261_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v261_remains_preserved_after_v262_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert current["scientific_status"] == "FAIL_CASE19_DETECTOR_INTEGRABILITY_PROJECTOR_NOT_FORWARD_INVARIANT_V262"
    assert current["headline"].startswith("v262")
    assert current["headline_zh"].startswith("v262")
    assert current["headline_en"].startswith("v262")
    assert "0/117" in current["headline_zh"] and "0/117" in current["headline_en"]
    assert metrics["v261_independent_checks_passed"] == 41
    assert metrics["v261_primary_absolute_cells"] == 3
    assert metrics["v261_primary_matched_cells"] == 0
    assert decision["v261_independent_validation_passed"] is True
    assert decision["v261_primary_improved_over_same_cost_k15"] is False
    assert decision["v261_full_sequence_authorized"] is False
    assert decision["v261_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith("blastnet_case19_detector_integrability_no_go_v262.png")
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_component_block_galerkin_v261" in text
        assert "41/41" in text and "0/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v261" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v261_public_artifacts_exclude_private_execution_details() -> None:
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


def test_all_public_html_local_links_resolve_after_v261() -> None:
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
