import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parents[1]
NAME = "segmented_cross_ray_learning_2026-10-10"


def test_authoritative_failure_and_posthoc_gain_are_separate():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    assert data["population"]["queries"] == 3333
    assert data["population"]["complete_strata"] == 33
    assert data["primary"]["outer_folds"] == 11
    assert data["primary"]["shared_parameters"] == 3841
    assert data["primary"]["matched_cells"] == 0
    assert data["primary"]["complete_matched_strata"] == 0
    assert data["primary"]["absolute_strata"] == 33
    assert data["primary"]["standalone_actions"] == {"A": 35, "AT": 35}
    assert data["decision"].startswith("FAIL_")
    readback = data["descriptive_readback"]
    assert readback["posthoc"] and readback["not_a_new_success_gate"]
    assert readback["tolerance_absolute"] == 1e-10
    controls = readback["comparisons"]
    assert controls["CGLS35"]["all_four_nonworse_cells"] == 3333
    assert controls["JacobiPCGLS35"]["all_four_nonworse_cells"] == 684
    assert controls["DualRidgeCG35"]["all_four_nonworse_cells"] == 0
    assert all(x > 0 for x in controls["CGLS35"]["median_paired_relative_improvement"].values())
    assert data["validation"]["checks_passed"] == 23
    assert not data["validation"]["independent_retraining"]
    assert all(x is False for x in data["claims"].values())
    assert data["costs"]["not_fresh_deployment_wall_or_rss"]


class NewSection(HTMLParser):
    def __init__(self, marker):
        super().__init__()
        self.marker = marker
        self.depth = 0
        self.tag = None
        self.pairs = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id") == self.marker:
            self.tag, self.depth = tag, 1
        elif self.depth and tag == self.tag:
            self.depth += 1
        if not self.depth:
            return
        if "data-i18n-zh" in attrs:
            self.pairs.append((attrs["data-i18n-zh"], attrs.get("data-i18n-en")))
        if tag == "a":
            self.links.append(attrs["href"])
        if tag == "img":
            self.links.append(attrs["src"])
            assert attrs.get("data-i18n-alt-en") and attrs.get("data-i18n-alt-zh")

    def handle_endtag(self, tag):
        if self.depth and tag == self.tag:
            self.depth -= 1


def test_bilingual_current_sections_and_links():
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        text = (ROOT / name).read_text()
        marker = "latest" if "daily-progress" in name else "segmented-cross-ray-20261010"
        assert text.count(f'id="{marker}"') == 1
        reader = NewSection(marker)
        reader.feed(text)
        assert len(reader.pairs) == 10
        assert all(a and b for a, b in reader.pairs)
        for link in reader.links:
            parts = urlsplit(link)
            assert (ROOT / name).parent.joinpath(parts.path).resolve().is_file()
            if "doc" in parse_qs(parts.query):
                assert (ROOT / parse_qs(parts.query)["doc"][0]).is_file()
    for name in ("index.html", "operator-learning/index.html"):
        text = (ROOT / name).read_text()
        assert text.index("const graph = evidence.latest_signed_cross_ray") < text.index("const synthesis = evidence.latest_research_synthesis")


def test_new_note_has_no_private_execution_identity():
    for suffix in (".md", "_public_summary.json"):
        text = (ROOT / f"docs/{NAME}{suffix}").read_text()
        for private in ("/Users/", "/Volumes/", "private_results/", "sha256", "checkpoint", "FROZEN.json", "OPENED.json"):
            assert private not in text
    evidence = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())["latest_signed_cross_ray"]
    assert evidence["matched_queries"] == 0 and evidence["absolute_strata"] == 33
    assert "segmented-cross-ray-20261010" in evidence["note"]


def test_directional_diagnostic_does_not_overturn_the_algorithm_gate():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    audit = data["endpoint_direction_attribution"]
    assert audit["queries"] == 3333 and audit["complete_strata"] == 33
    assert audit["cells_ratio_above_one"] == 3333
    assert audit["strata_median_ratio_above_one"] == 33
    assert 34 < audit["median_rayleigh_ratio"] < 35
    assert .34 < audit["field_alignment_cosine_median"] < .36
    assert .12 < audit["teacher_visible_field_line_energy_capacity_median"] < .13
    assert audit["additional_actions"] == {"A": 0, "AT": 0}
    assert audit["own_sealed_endpoints_and_physical_replays"]
    assert "not a full eigenspectrum" in audit["limits"]
    assert data["decision"].startswith("FAIL_")
    assert data["primary"]["matched_cells"] == 0


def test_roundoff_diagnostic_is_not_algorithm_success():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    audit = data["native_roundoff_diagnostic"]
    assert audit["queries"] == 99 and audit["sampled_strata"] == 33
    assert audit["independent_checks_passed"] == 18
    assert audit["maximum_four_map_endpoint_response"] < 3e-13
    assert audit["relative_seed_perturbation"] == 1e-12
    assert audit["no_cfd_or_teacher_scoring"]
    assert not audit["new_candidate_tested"]
    assert audit["earlier_numerical_failures_unchanged"]
    assert data["decision"].startswith("FAIL_")
    assert not data["claims"]["algorithm_breakthrough"]
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        assert (ROOT / name).read_text().count('id="native-roundoff-diagnostic"') == 1


def test_controlled_noise_signal_preserves_the_stronger_gate():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    noise = data["controlled_noise_signal"]
    assert noise["anchors"] == 99 and noise["queries"] == 198
    assert noise["draws_per_anchor"] == 2 and noise["relative_l2_radius"] == .01
    assert noise["all_four_nonworse_vs_cold"] == 198
    assert noise["all_four_nonworse_vs_jacobi"] == 39
    assert noise["all_four_nonworse_vs_dual_ridge"] == 0
    assert noise["strong_reference_matches"] == 0
    assert noise["reference_absolute_sampled_strata"] == 66
    assert noise["independent_checks_passed"] == 42
    assert noise["new_fits"] == 0 and not noise["full_noisy_sequence"]
    assert noise["original_clean_failure_unchanged"]
    assert not noise["model_rescue_authorized"] and not noise["deployment_resource_claim"]
    assert data["decision"].startswith("FAIL_") and data["primary"]["matched_cells"] == 0
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        assert (ROOT / name).read_text().count('id="controlled-noise-signal"') == 1


def test_tangent_attribution_preserves_scope_and_failure():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    audit = data["geometry_tangent_attribution"]
    assert audit["queries"] == 3333 and audit["complete_strata"] == 33
    assert audit["tail_comparisons_passed"] == 176 and audit["tail_comparisons"] == 396
    assert audit["complete_retention_strata"] == 2 and audit["retention_fraction_gate"] == .9
    assert audit["only_neural_increment_linearized"] and audit["data_dependent_k1_retained"]
    assert audit["gates_depend_on_previously_learned_weights"]
    assert audit["new_fits"] == 0 and not audit["model_rescue_authorized"]
    assert audit["strong_reference_matches"] == 0 and audit["absolute_strata"] == 33
    assert audit["standalone_actions"] == {"A": 35, "AT": 35}
    assert not audit["deployment_resource_claim"]
    assert data["decision"].startswith("FAIL_")
    assert not data["claims"]["algorithm_breakthrough"]
    note = (ROOT / f"docs/{NAME}.md").read_text()
    assert "86.46%" in note and "91.90%" in note
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        assert (ROOT / name).read_text().count('id="geometry-tangent-attribution"') == 1
