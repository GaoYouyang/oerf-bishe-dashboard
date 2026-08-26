from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs/blastnet_case19_observation_excess_attribution_v259_public_summary.json"
RESULT = ROOT / "docs/blastnet_case19_observation_excess_attribution_v259_result_2026-08-26.md"
FIGURE = ROOT / "assets/figures/blastnet_case19_observation_excess_attribution_v259.png"
BUILDER = ROOT / "site_tools/build_blastnet_case19_observation_excess_attribution_v259_figure.py"
CURRENT = ROOT / "operator-learning/current-evidence.json"
FOCUS = ROOT / "operator-learning/index.html"
HOME = ROOT / "index.html"
DAILY = ROOT / "operator-learning/daily-progress.html"
LEARNING_LOG = ROOT / "docs/operator_3d_learning_log.md"
PUBLICATION_BOUNDARY = ROOT / "publication-boundary.html"


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


def test_v259_summary_records_independent_post_open_attribution() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    validation = data["independent_validation"]
    primary = data["primary_attribution"]
    assert data["scope"]["cells"] == 13
    assert data["scope"]["new_exact_operator_calls"] == [0, 0]
    assert data["scope"]["new_candidate_or_field_generated"] is False
    assert validation["passed"] is True
    assert validation["checks_passed"] == validation["checks_total"] == 18
    assert primary["camera"]["localized_rigs"] == 10
    assert primary["component"]["localized_rigs"] == 13
    assert primary["frequency"]["localized_rigs"] == 12
    assert data["adjudication"]["route_action"].startswith("AUTHORIZE_EXACTLY_ONE")
    assert data["adjudication"]["scientific_pass_claimed"] is False
    assert all(value is False for value in data["claims_fixed_false"].values())


def test_v259_result_is_bilingual_and_preserves_claim_boundaries() -> None:
    text = RESULT.read_text(encoding="utf-8")
    assert "# v259：" in text and "# v259:" in text
    for token in ("18/18", "10/13", "13/13", "12/13", "0A+0A^T", "0.811", "0.924"):
        assert token in text
    assert "不是算法通过" in text
    assert "not an algorithmic pass" in text
    assert "algorithm_breakthrough=false" in text


def test_v259_figure_and_builder_are_public() -> None:
    assert BUILDER.is_file()
    assert FIGURE.is_file() and FIGURE.stat().st_size > 40_000
    with Image.open(FIGURE) as image:
        assert image.width >= 2800
        assert image.height >= 1100


def test_v259_remains_preserved_after_v260_on_bilingual_primary_pages() -> None:
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    metrics = current["metrics"]
    decision = current["current_decision"]
    assert current["scientific_status"] == "FAIL_CASE19_STRATIFIED_RAY_CORRECTION_FULL_SEQUENCE_V265_1"
    assert metrics["v259_independent_checks_passed"] == 18
    assert metrics["v259_camera_localized_rigs"] == 10
    assert metrics["v259_component_localized_rigs"] == 13
    assert metrics["v259_frequency_localized_rigs"] == 12
    assert decision["v259_independent_validation_passed"] is True
    assert decision["v259_exactly_one_camera_local_diagnostic_authorized"] is True
    assert decision["v259_full_sequence_authorized"] is False
    assert decision["v259_algorithm_breakthrough"] is False
    assert current["public_evidence"]["figure"].endswith("blastnet_case19_stratified_ray_correction_full_sequence_v265_1.png")
    for page in (FOCUS, HOME, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "blastnet_case19_observation_excess_attribution_v259" in text
        assert "data-i18n-zh" in text and "data-i18n-en" in text
    for page in (FOCUS, DAILY):
        text = page.read_text(encoding="utf-8")
        assert "18/18" in text and "10/13" in text
    assert "v259" in LEARNING_LOG.read_text(encoding="utf-8")


def test_v259_public_artifacts_exclude_private_execution_details() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + RESULT.read_text(encoding="utf-8")
    forbidden = (
        "/Users/",
        "/Volumes/",
        "private_results",
        "private_worktrees",
        "source_commit",
        "checkpoint.pt",
        "run ID",
        "afe015d1",
    )
    assert all(token not in text for token in forbidden)


def test_publication_boundary_is_bilingual_and_old_private_protocols_are_absent() -> None:
    text = PUBLICATION_BOUNDARY.read_text(encoding="utf-8")
    assert "这个资产没有公开" in text
    assert "This asset is not public" in text
    assert "data-i18n-zh" in text and "data-i18n-en" in text
    assert not (
        ROOT
        / "learning_labs/protocols/poolfire_c_geometry_equalized_bp_audit_v7.json"
    ).exists()
    assert not (
        ROOT
        / "learning_labs/protocols/"
        "poolfire_c_dual_representation_ceiling_clarification_v10_4_2.json"
    ).exists()


def test_all_public_html_local_links_resolve() -> None:
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
            target = (
                ROOT / relative.lstrip("/")
                if relative.startswith("/")
                else page.parent / relative
            )
            candidates = [target]
            if relative.endswith("/"):
                candidates.append(target / "index.html")
            elif not target.suffix:
                candidates.extend((target / "index.html", target.with_suffix(".html")))
            if not any(candidate.exists() for candidate in candidates):
                missing.append((str(page.relative_to(ROOT)), raw))
    assert missing == []
