from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from demo_t16_operator.d4b_frozen_inputs import array_sha256, build_frozen_inputs
from demo_t16_operator.run_n2_pvgr_n5_d4b_population_field_derivative import (
    DEFAULT_CONFIG,
    _path_occupied,
    _validate_contract,
    recompute_budget,
)
from site_tools import (
    create_n2_pvgr_n5_d4b_population_field_derivative_attestation as attestation_builder,
)
from site_tools import (
    validate_n2_pvgr_n5_d4b_population_field_derivative as independent_validator,
)


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _valid_staging_bundle(tmp_path: Path):
    bundle = tmp_path / "bundle"
    staging = tmp_path / ".bundle.tmp"
    attestation_path = bundle / "attestation.json"
    archive_path = bundle / "frozen_inputs.npz"
    ready_path = bundle / "READY.json"
    config = {
        "pre_registration_bundle": str(bundle),
        "pre_registration_attestation": str(attestation_path),
        "frozen_input_archive": str(archive_path),
        "pre_registration_ready_marker": str(ready_path),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    staging.mkdir()
    staged_archive = staging / archive_path.name
    staged_archive.write_bytes(b"frozen bytes")
    protocol_commit = "a" * 40
    attestation = {
        "schema": "n2-pvgr-n5-d4b-population-field-derivative-attestation-1.0",
        "protocol_commit": protocol_commit,
        "frozen_input_archive_sha256": hashlib.sha256(
            staged_archive.read_bytes()
        ).hexdigest(),
    }
    staged_attestation = staging / attestation_path.name
    staged_attestation.write_text(json.dumps(attestation) + "\n", encoding="utf-8")
    ready = {
        "schema": "n2-pvgr-n5-d4b-preregistration-ready-1.0",
        "protocol_commit": protocol_commit,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "attestation_path": str(attestation_path),
        "attestation_sha256": hashlib.sha256(
            staged_attestation.read_bytes()
        ).hexdigest(),
        "frozen_input_archive_path": str(archive_path),
        "frozen_input_archive_sha256": hashlib.sha256(
            staged_archive.read_bytes()
        ).hexdigest(),
        "publication_rule": "atomically_rename_complete_staging_directory",
    }
    (staging / ready_path.name).write_text(json.dumps(ready) + "\n", encoding="utf-8")
    return config_path, config, staging, bundle


def test_preregistered_contract_has_exact_closed_census_and_budget() -> None:
    config = _config()
    _validate_contract(config)
    independent_validator._validate_static_config(config)
    budget = recompute_budget(config)
    assert budget == {
        "derivative_map_logical_queries": 8_257_536,
        "derivative_map_interpolation_dispatches": 1_405_952,
        "topology_logical_queries": 4_300_800,
        "topology_interpolation_dispatches": 493_440,
        "d4b_total_logical_queries": 12_558_336,
        "d4b_total_interpolation_dispatches": 1_899_392,
        "map_closure_invocations": 4096,
        "jvp_sweeps": 256,
        "vjp_sweeps": 256,
        "finite_difference_forward_calls": 3584,
        "topology_signature_invocations": 960,
    }
    population = config["population_contract"]
    assert population["expected_cell_count"] == 32
    assert population["expected_pair_ids"] == [f"p{i:02d}" for i in range(1, 17)]
    assert population["closed_development_population_not_iid_sample"] is True
    assert population["required_reporting_levels"] == [
        "cell",
        "pair_cluster",
        "field_unit_cluster",
    ]
    assert all(value is False for value in config["claim_authorizations"].values())


def test_d4_thresholds_are_unchanged_but_direction_seeds_are_fresh() -> None:
    config = _config()
    parent = json.loads(
        (DEFAULT_CONFIG.parents[2] / config["parent_d4_config"]).read_text(
            encoding="utf-8"
        )
    )
    for key in (
        "maximum_dot_relative_defect",
        "minimum_dot_signal_absolute",
        "maximum_best_finite_difference_relative_l2",
        "maximum_required_h_finite_difference_relative_l2",
        "maximum_repeat_output_relative_l2",
        "maximum_raw_structural_output_relative_l2",
        "maximum_raw_structural_jvp_relative_l2",
        "maximum_raw_structural_vjp_relative_l2",
        "maximum_raw_paired_output_relative_l2",
        "maximum_raw_paired_jvp_relative_l2",
        "maximum_raw_paired_vjp_relative_l2",
        "structural_absolute_tolerance_floor",
        "required_finite_fraction",
        "minimum_domain_margin",
        "minimum_stencil_margin",
        "maximum_direction_norm_error",
    ):
        assert config["gates"][key] == parent["gates"][key]
    assert config["finite_difference"] == parent["finite_difference"]
    assert (
        config["direction_contract"]["seed_base"]
        != parent["direction_contract"]["seed_base"]
    )
    assert (
        config["direction_contract"]["cotangent_seed_base"]
        != parent["direction_contract"]["cotangent_seed_base"]
    )


def test_frozen_inputs_are_deterministic_and_cover_all_clusters() -> None:
    config = _config()
    metadata_a, arrays_a = build_frozen_inputs(config)
    metadata_b, arrays_b = build_frozen_inputs(config)
    assert metadata_a == metadata_b
    assert metadata_a["cell_count"] == 32
    assert metadata_a["pair_count"] == 16
    assert metadata_a["field_unit_count"] == 5
    assert metadata_a["family_counts"] == {"smooth": 20, "wrinkled": 12}
    assert metadata_a["stress_counts"] == {"1": 12, "10": 10, "3": 10}
    assert metadata_a["d3_reference_method_counts"] == {
        "paired_neumaier": 2,
        "raw_separate_subtraction": 30,
    }
    assert len(arrays_a) == 32 * (2 + 2 * 2)
    assert set(arrays_a) == set(arrays_b)
    assert [row["pair_id"] for row in metadata_a["contexts"][::2]] == [
        f"p{i:02d}" for i in range(1, 17)
    ]
    assert [row["role"] for row in metadata_a["contexts"][:4]] == [
        "n3_failure",
        "matched_control",
        "n3_failure",
        "matched_control",
    ]
    for key in arrays_a:
        assert arrays_a[key].dtype == np.dtype("float64")
        assert arrays_a[key].flags.c_contiguous
        assert array_sha256(arrays_a[key]) == array_sha256(arrays_b[key])


def test_every_frozen_direction_has_zero_boundary_and_unit_norm() -> None:
    metadata, arrays = build_frozen_inputs(_config())
    for context in metadata["contexts"]:
        for entry in context["directions"]:
            direction = arrays[entry["direction_key"]]
            assert np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=2e-15)
            boundary = direction.copy()
            boundary[2:-2, 2:-2, 2:-2] = 0.0
            assert np.count_nonzero(boundary) == 0
            cotangent = arrays[entry["cotangent_key"]]
            assert np.isclose(np.linalg.norm(cotangent), 1.0, rtol=0.0, atol=2e-15)
            assert np.allclose(
                np.abs(cotangent),
                1.0 / np.sqrt(cotangent.size),
                rtol=0.0,
                atol=1e-16,
            )


def test_contract_fails_closed_on_selection_threshold_or_budget_drift() -> None:
    config = _config()
    changed_population = copy.deepcopy(config)
    changed_population["population_contract"]["expected_cell_count"] = 31
    with pytest.raises(ValueError, match="32 cells"):
        _validate_contract(changed_population)
    changed_gate = copy.deepcopy(config)
    changed_gate["gates"]["maximum_dot_relative_defect"] = 2e-10
    with pytest.raises(ValueError, match="threshold"):
        _validate_contract(changed_gate)
    changed_budget = copy.deepcopy(config)
    changed_budget["budget_contract"]["expected_total_logical_queries"] += 1
    with pytest.raises(ValueError, match="budget"):
        _validate_contract(changed_budget)


def test_independent_validator_does_not_import_d4b_decision_code() -> None:
    source = independent_validator.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    tree = ast.parse(text)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden = {
        "run_n2_pvgr_n5_d4b_population_field_derivative",
        "field_jvp_vjp_gate",
        "d4b_frozen_inputs",
        "field_program_signature",
    }
    assert not any(module.split(".")[-1] in forbidden for module in modules)
    assert "__import__" not in text
    assert "import_module" not in text


def test_attestation_publication_is_exclusive_and_detects_dangling_symlink(
    tmp_path,
) -> None:
    archive = tmp_path / "frozen.npz"
    arrays = {"probe": np.arange(6, dtype=np.float64).reshape(2, 3)}
    attestation_builder._write_npz_atomic(archive, arrays)
    with np.load(archive, allow_pickle=False) as handle:
        assert np.array_equal(handle["probe"], arrays["probe"])
    with pytest.raises(FileExistsError):
        attestation_builder._write_npz_atomic(archive, arrays)

    report = tmp_path / "validation.json"
    independent_validator._write_json_exclusive(report, {"valid": True})
    assert json.loads(report.read_text(encoding="utf-8")) == {"valid": True}
    with pytest.raises(FileExistsError):
        independent_validator._write_json_exclusive(report, {"valid": False})

    occupied = tmp_path / "formal-output"
    occupied.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    assert _path_occupied(occupied) is True
    assert attestation_builder._path_occupied(occupied) is True


def test_preregistration_bundle_publishes_as_one_directory(tmp_path) -> None:
    config_path, config, staging, bundle = _valid_staging_bundle(tmp_path)
    attestation_builder._publish_staging_bundle(config_path, config, staging, bundle)
    assert bundle.is_dir()
    assert not staging.exists()
    assert {path.name for path in bundle.iterdir()} == {
        "attestation.json",
        "frozen_inputs.npz",
        "READY.json",
    }
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replacement_config, replacement_payload, replacement, _ = _valid_staging_bundle(
        replacement_root
    )
    replacement_payload["pre_registration_bundle"] = str(bundle)
    replacement_payload["pre_registration_attestation"] = str(
        bundle / "attestation.json"
    )
    replacement_payload["frozen_input_archive"] = str(bundle / "frozen_inputs.npz")
    replacement_payload["pre_registration_ready_marker"] = str(bundle / "READY.json")
    replacement_config.write_text(
        json.dumps(replacement_payload) + "\n", encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        attestation_builder._atomic_rename_bundle(replacement, bundle)


def test_bundle_rename_crash_leaves_no_partial_final(tmp_path, monkeypatch) -> None:
    config_path, config, staging, bundle = _valid_staging_bundle(tmp_path)

    def injected_failure(_source, _destination) -> None:
        raise RuntimeError("injected atomic rename failure")

    monkeypatch.setattr(attestation_builder.os, "rename", injected_failure)
    with pytest.raises(RuntimeError, match="rename failure"):
        attestation_builder._publish_staging_bundle(
            config_path, config, staging, bundle
        )
    assert not bundle.exists()
    assert staging.is_dir()
    assert {path.name for path in staging.iterdir()} == {
        "attestation.json",
        "frozen_inputs.npz",
        "READY.json",
    }


def test_bundle_publisher_rejects_unverified_extra_file(tmp_path) -> None:
    config_path, config, staging, bundle = _valid_staging_bundle(tmp_path)
    (staging / "unverified.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unverified extra"):
        attestation_builder._publish_staging_bundle(
            config_path, config, staging, bundle
        )


def test_independent_array_replay_requires_byte_exact_equality() -> None:
    stored = {"probe": np.array([1.0], dtype=np.float64)}
    recomputed = np.array([np.nextafter(1.0, 2.0)], dtype=np.float64)
    with pytest.raises(ValueError, match="not byte-exact"):
        independent_validator._verify_array(stored, set(), "probe", recomputed)
    with pytest.raises(ValueError, match="little-endian float64"):
        independent_validator._verify_array(
            {"probe": np.array([1.0], dtype=np.float32)},
            set(),
            "probe",
            np.array([1.0], dtype=np.float64),
        )


def test_attestation_and_formal_result_state_when_present() -> None:
    config = _config()
    root = DEFAULT_CONFIG.parents[2]
    attestation = root / config["pre_registration_attestation"]
    archive = root / config["frozen_input_archive"]
    formal = root / config["formal_output"]
    if not attestation.exists() or not archive.exists():
        assert not formal.exists()
        pytest.skip("N5-D4b attestation is created only after protocol commit")
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    assert payload["formal_results_absent_at_creation"] is True
    assert payload["formal_work_output_absent_at_creation"] is True
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == payload["frozen_input_archive_sha256"]
    )
    if formal.exists():
        validation = formal / "validation_report.json"
        if validation.exists():
            assert json.loads(validation.read_text(encoding="utf-8"))["valid"] is True
