from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_camera_weighted_complement_v260_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_camera_weighted_complement_v260_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_component_block_galerkin_v261.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_camera_weighted_complement_v260_figure.py"
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


def test_v260_summary_records_independent_negative_gate() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    primary = data["primary"]
    control = data["controls"]["unweighted_v258_equal_call"]
    assert data["scope"]["cells"] == 13
    assert data["mechanism"]["logical_exact_calls_A"] == 15
    assert data["mechanism"]["logical_exact_calls_At"] == 14
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 52
    assert primary["absolute_cells"] == 13
    assert primary["matched_cells"] == 0
    assert primary["matched_ratio_p90_higher"][3] > 1.05
    assert primary["matched_ratio_worst"][3] > 1.05
    assert primary["matched_ratio_p90_higher"][3] > control["observation_matched_ratio_p90_higher"]
    assert primary["matched_ratio_worst"][3] > control["observation_matched_ratio_worst"]
    assert data["adjudication"]["full_sequence_authorized"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v260_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v260：" in text and "# v260:" in text
    for token in ("52/52", "13/13", "0/13", "1.22811", "1.36863", "15A+14A^T"):
        assert token in text
    assert "不是算法" in text
    assert "not an algorithm" in text
    assert "algorithm_breakthrough=false" in text


def test_v260_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v260_remains_preserved_after_v261_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert current["scientific_status"] == "MIXED_OR_NEAR_FLAT_CASE19_HAAR_IRLS_NULL_LINE_V272"
    assert metrics["v260_independent_checks_passed"] == 52
    assert metrics["v260_primary_absolute_cells"] == 13
    assert metrics["v260_primary_matched_cells"] == 0
    assert decision["v260_independent_validation_passed"] is True
    assert decision["v260_camera_weighting_improved_over_unweighted_control"] is False
    assert decision["v260_full_sequence_authorized"] is False
    assert decision["v260_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith("blastnet_case19_haar_irls_null_line_attribution_v272.png")
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_camera_weighted_complement_v260" in text
        assert "52/52" in text and "0/13" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert "v260" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v260_public_artifacts_exclude_private_execution_details() -> None:
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


def test_all_public_html_local_links_resolve_after_v260() -> None:
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
