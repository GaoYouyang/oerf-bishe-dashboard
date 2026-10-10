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
        assert len(reader.pairs) == 14
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


def test_classical_prior_information_is_not_neural_or_range_success():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    prior = data["fold_prior_information"]
    assert prior["queries"] == 3333 and prior["complete_strata"] == 33
    assert prior["complete_held_trajectory_excluded"] and prior["training_cfd_labels_used"]
    assert prior["prior_fits_each_mode"] == 22 and prior["neural_updates"] == 0
    assert prior["coefficients_per_prior_fold"] == 35547
    assert prior["decision"].startswith("FAIL_") and prior["literal_recipe_closed"]
    assert prior["raw_cfd_prior"]["harm_vs_cold"] == 228
    assert prior["finite_teacher_prior"]["harm_vs_cold"] == 12
    for arm in ("raw_cfd_prior", "finite_teacher_prior"):
        assert prior[arm]["strong_reference_matches"] == 0
        assert prior[arm]["absolute_strata"] == 33
    assert prior["explicit_prior_plus_adjoint_correction"]
    assert not prior["pure_range_initializer"] and not prior["native_exact_kernel_identified"]
    assert not prior["variable_cardinality_neural_learner"]
    assert not prior["existing_initializer_contract_changed"]
    assert prior["independent_checks_passed"] == 114
    assert prior["standalone_actions"] == {"A": 35, "AT": 34}
    assert prior["strong_classical_controls_cost_one_more_AT"]
    assert prior["postseal_descriptive_pairs_not_success_gate"]
    assert not prior["deployment_resource_claim"] and not prior["main_goal_complete"]
    assert data["primary"]["matched_cells"] == 0
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        assert (ROOT / name).read_text().count('id="fold-prior-information"') == 1


def test_training_mean_information_location_does_not_promote_success():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    audit = data["fold_prior_information"]["information_location"]
    assert audit["prior_pairs"] == audit["ratio_below_one"] == 33
    assert .00017 < audit["rayleigh_ratio_median"] < .00019
    assert .17 < audit["field_difference_ratio_median"] < .18
    assert .0023 < audit["image_difference_ratio_median"] < .0025
    assert audit["actions_per_mode"] == {"A": 99, "AT": 0}
    assert audit["independent_checks_passed"] == 43
    assert audit["new_fits"] == audit["new_solver_steps"] == audit["new_network_updates"] == 0
    assert audit["no_new_held_truth"] and audit["original_closed_verdicts_unchanged"]
    assert not audit["native_exact_kernel_identified"] and not audit["deployment_resource_claim"]
    assert data["primary"]["matched_cells"] == 0 and data["decision"].startswith("FAIL_")


def test_reciprocal_raw_energy_learning_keeps_control_and_failure_boundaries():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    audit = data["reciprocal_raw_prior_energy"]
    assert audit["queries"] == 3333 and audit["complete_strata"] == 33
    assert audit["outer_folds"] == 11 and audit["shared_parameters"] == 1091
    assert audit["geometry_bias_parameters"] == 481
    assert audit["training_updates_all_arms"] == 7260
    assert audit["raw_cfd_other_fold_targets"] and audit["complete_held_trajectory_excluded"]
    assert audit["native_camera_counts"] == [5, 7, 9] and audit["manufactured_12_only"]
    assert audit["primary"]["matched_cells"] == 0
    assert audit["primary"]["complete_matched_strata"] == 0
    assert audit["primary"]["absolute_strata"] == 33
    assert audit["primary"]["all_four_nonworse_vs_cold"] == 2496
    assert audit["primary"]["harm_vs_cold"] == 837
    assert audit["primary"]["all_four_nonworse_vs_bias"] == 3333
    assert audit["primary"]["all_four_nonworse_vs_untrained"] == 3333
    assert audit["primary"]["all_four_nonworse_vs_jacobi"] == 216
    assert audit["primary"]["all_four_nonworse_vs_dual_ridge"] == 0
    assert .00056 < audit["primary"]["median_paired_relative_improvement_vs_cold"]["field"] < .00057
    assert .0077 < audit["primary"]["median_paired_relative_improvement_vs_cold"]["observation"] < .0078
    assert audit["independent_checks_passed"] == 38 and not audit["independent_retraining"]
    assert audit["standalone_actions"] == {"A": 35, "AT": 35}
    assert audit["shared_three_arm_actions"] == {"A": 103, "AT": 103}
    assert audit["literal_recipe_closed"] and not audit["global_route_impossibility"]
    assert not audit["deployment_resource_claim"] and not audit["unopened_external_conditions_read"]
    assert all(v is False for v in audit["claims"].values())
    live = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert live["latest_signed_cross_ray"]["reciprocal_raw_prior_energy"] == audit
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        assert (ROOT / name).read_text().count('id="reciprocal-raw-prior-energy"') == 1


def test_conservative_invisible_prior_attribution_does_not_claim_learned_speedup():
    data = json.loads((ROOT / f"docs/{NAME}_public_summary.json").read_text())
    audit = data["prior_gain_kernel_intervention"]
    assert audit["queries"] == 1111 and audit["complete_trajectory_folds"] == 11
    assert audit["native_camera_counts"] == [5]
    assert audit["new_fits"] == audit["new_network_updates"] == 0
    gain = audit["pooled_field_gain"]
    assert abs(gain["fraction"]-gain["lost_by_removing_h"]/gain["full_raw_vs_finite_gain"]) < 1e-14
    assert .6646 < gain["fraction"] < .6647 and gain["signed_not_clipped"]
    assert audit["positive_gain_trajectories"] == 9 and audit["negative_gain_trajectories"] == 2
    assert audit["nonpositive_gain_queries"] == 159
    assert audit["negative_denominator_fraction_undefined"]
    assert audit["conservative_complement_not_full_kernel"]
    assert audit["retained_Q_space_not_guaranteed_pure_range"]
    assert audit["independent_checks_passed"] == 57
    assert audit["field_identity_relative_max"] < 1e-7 and audit["image_identity_relative_max"] < 1e-7
    assert audit["standalone_actions"] == {"A": 35, "AT": 34}
    for arm in audit["interventions"].values():
        assert arm["matched_queries"] == arm["complete_matched_strata"] == 0
        assert arm["absolute_strata"] == 11
        assert arm["non_harm_vs_cold"]+arm["harm_vs_cold"] == 1111
    assert audit["original_failures_and_contract_unchanged"]
    assert not audit["pure_range_speedup_impossibility"] and not audit["new_predictor_authorized"]
    assert not audit["unopened_external_conditions_read"] and not audit["deployment_resource_claim"]
    assert all(v is False for v in audit["claims"].values())
    assert data["decision"].startswith("FAIL_") and data["primary"]["matched_cells"] == 0
    live = json.loads((ROOT / "operator-learning/current-evidence.json").read_text())
    assert live["latest_signed_cross_ray"]["prior_gain_kernel_intervention"] == audit
    for name in ("index.html", "operator-learning/index.html", "operator-learning/daily-progress.html", "learning_log.html"):
        assert (ROOT / name).read_text().count('id="segmented-prior-kernel-intervention"') == 1
