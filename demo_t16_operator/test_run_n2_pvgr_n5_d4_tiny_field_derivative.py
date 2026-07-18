from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from demo_t16_operator.automatic_discrete_multifidelity import SyntheticRayRig
from demo_t16_operator.d4_frozen_inputs import array_sha256, build_frozen_inputs
from demo_t16_operator.field_dependent_ray import sample_pupil_sobol
from demo_t16_operator.field_jvp_vjp_gate import audit_tensor_closure
from demo_t16_operator.run_n2_pvgr_n5_d4_tiny_field_derivative import (
    ROOT,
    _closures,
    _path_occupied,
    _serialize_audit,
    _strong_toy_detach_control,
    _topology_rows,
    _validate_contract,
)
from site_tools import (
    create_n2_pvgr_n5_d4_tiny_field_derivative_attestation as attestation_builder,
)
from site_tools import (
    validate_n2_pvgr_n5_d4_tiny_field_derivative as independent_validator,
)
from site_tools.create_n2_pvgr_n5_d4_tiny_field_derivative_attestation import (
    _temporary_sibling,
    _write_bytes_atomic,
    _write_npz_atomic,
)
from site_tools.validate_n2_pvgr_n5_d4_tiny_field_derivative import (
    _expand_audit_cells as independently_expand_audit_cells,
)
from site_tools.validate_n2_pvgr_n5_d4_tiny_field_derivative import (
    _rig_from_case as independently_build_rig,
)
from site_tools.validate_n2_pvgr_n5_d4_tiny_field_derivative import (
    _stable_seed as independently_build_seed,
)


CONFIG = (
    ROOT
    / "demo_t16_operator/configs/n2_pvgr_n5_d4_tiny_field_derivative_preregistered_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_preregistered_contract_and_query_budget_are_exact() -> None:
    config = _config()
    _validate_contract(config)
    assert (
        config["budget_contract"]["expected_grand_total_logical_queries"] == 1_573_152
    )
    assert config["finite_difference"]["required_h_values"] == [0.01, 0.003, 0.001]
    assert config["direction_contract"]["field_direction_modes_in_order"] == [
        "smooth_avg_pool3d_kernel3_stride1_padding1",
        "raw_torch_randn",
    ]
    assert all(value is False for value in config["claim_authorizations"].values())


def test_frozen_inputs_are_byte_deterministic_and_cover_four_by_two() -> None:
    config = _config()
    metadata_a, arrays_a = build_frozen_inputs(config)
    metadata_b, arrays_b = build_frozen_inputs(config)
    assert metadata_a == metadata_b
    assert metadata_a["cell_count"] == 4
    assert metadata_a["direction_count_per_cell"] == 2
    assert len(arrays_a) == 4 * (2 + 2 * 2)
    assert set(arrays_a) == set(arrays_b)
    for key in arrays_a:
        assert arrays_a[key].dtype == np.dtype("float64")
        assert arrays_a[key].flags.c_contiguous
        assert array_sha256(arrays_a[key]) == array_sha256(arrays_b[key])
    assert len({row["field_unit_id"] for row in metadata_a["contexts"]}) == 1
    assert len({row["cell_id"] for row in metadata_a["contexts"]}) == 4


def test_independent_context_expansion_matches_frozen_metadata() -> None:
    config = _config()
    metadata, _ = build_frozen_inputs(config)
    scientific = json.loads(
        (ROOT / config["scientific_n4_config"]).read_text(encoding="utf-8")
    )
    parent = json.loads((ROOT / config["parent_n3_config"]).read_text(encoding="utf-8"))
    expanded = {
        row["cell_id"]: row
        for row in independently_expand_audit_cells(scientific, parent)
    }
    contract = config["direction_contract"]
    for context in metadata["contexts"]:
        cell = expanded[context["cell_id"]]
        assert cell["field_unit_id"] == context["field_unit_id"]
        assert cell["pair_id"] == context["pair_id"]
        assert cell["role"] == context["role"]
        assert (
            cell["dimensionless_stress_multiplier"]
            == context["dimensionless_stress_multiplier"]
        )
        assert asdict(independently_build_rig(cell)) == context["rig"]
        for direction_index, entry in enumerate(context["directions"]):
            assert entry["direction_seed"] == independently_build_seed(
                int(contract["seed_base"]), context["cell_id"], direction_index
            )
            assert entry["cotangent_seed"] == independently_build_seed(
                int(contract["cotangent_seed_base"]),
                context["cell_id"],
                direction_index,
            )

    bad_pair_id = copy.deepcopy(scientific)
    bad_pair_id["audit_pairs"][0]["id"] = "unexpected"
    with pytest.raises(ValueError, match="identifiers or order"):
        independently_expand_audit_cells(bad_pair_id, parent)
    bad_failure_gate = copy.deepcopy(scientific)
    bad_failure_gate["audit_pairs"][0]["failed_cell"]["n3_failed_gate"] = "unknown"
    with pytest.raises(ValueError, match="invalid failed N3 gate"):
        independently_expand_audit_cells(bad_failure_gate, parent)


def test_direction_boundaries_and_normalization_are_frozen() -> None:
    metadata, arrays = build_frozen_inputs(_config())
    for context in metadata["contexts"]:
        for entry in context["directions"]:
            value = arrays[entry["direction_key"]]
            assert np.isclose(np.linalg.norm(value), 1.0, rtol=0.0, atol=2e-15)
            boundary = value.copy()
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


def test_strong_toy_control_detects_trajectory_derivative() -> None:
    result = _strong_toy_detach_control(_config())
    assert result["gate"] is True
    assert result["comparison"]["relative_l2"] >= 1e-4
    assert result["logical_field_point_queries"] == 3360


def test_nonformal_toy_exercises_all_map_and_topology_serializers() -> None:
    config = _config()
    axis = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    field = -torch.exp(-((xx / 0.43) ** 2 + (yy / 0.35) ** 2 + (zz / 0.39) ** 2))
    rays = sample_pupil_sobol(4, seed=9203)
    rig = SyntheticRayRig(
        rig_id="n5-d4-nonformal-pipeline-toy",
        view_angle_degrees=31.0,
        detector_u=0.03,
        detector_z=-0.02,
        aperture_radius=0.025,
        path_half_length=0.62,
        cone_u=0.02,
        cone_z=0.015,
        bend=0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(9209)
    direction = torch.randn(field.shape, generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    cotangent = torch.randn((4, 2), generator=generator, dtype=torch.float64)
    cotangent = cotangent / torch.linalg.vector_norm(cotangent)
    closures = _closures(
        rays,
        rig,
        difference_step=0.002,
        refractivity_scale=0.003,
        step_count=16,
    )
    arrays: dict[str, np.ndarray] = {}
    h_values = config["finite_difference"]["h_values"]
    for map_id, closure in closures.items():
        audit = audit_tensor_closure(
            closure,
            field,
            direction,
            cotangent,
            h_values,
            nondegenerate_floor=1e-16,
        )
        payload, _ = _serialize_audit(
            audit,
            prefix=f"toy_{map_id}",
            arrays=arrays,
            config=config,
        )
        assert payload["finite"] is True
        assert len(payload["levels"]) == 7
    topology, _ = _topology_rows(
        field,
        direction,
        rays,
        rig,
        {
            "difference_step": 0.002,
            "refractivity_scale": 0.003,
            "support_threshold": 0.1,
            "frustum_half_width_u": 0.005,
            "frustum_half_width_v": 0.005,
        },
        config,
    )
    assert len(topology["perturbations"]) == 14
    assert topology["base"]["group_count"] == 4 * 16 + 2
    assert len(arrays) == 4 * (3 + 7 * 3)


def test_attestation_validates_when_present() -> None:
    config = _config()
    attestation = ROOT / config["pre_registration_attestation"]
    archive = ROOT / config["frozen_input_archive"]
    if not attestation.exists() or not archive.exists():
        assert not (ROOT / config["formal_output"]).exists()
        assert not (ROOT / config["formal_work_output"]).exists()
        pytest.skip(
            "N5-D4 attestation and frozen arrays are created after protocol commit"
        )
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    assert payload["formal_results_absent_at_creation"] is True
    assert payload["formal_work_output_absent_at_creation"] is True
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == payload["frozen_input_archive_sha256"]
    )


def test_independent_validator_does_not_import_d4_decision_code() -> None:
    source = (
        ROOT / "site_tools/validate_n2_pvgr_n5_d4_tiny_field_derivative.py"
    ).read_text(encoding="utf-8")
    assert "run_n2_pvgr_n5_d4_tiny_field_derivative import" not in source
    assert "field_jvp_vjp_gate import" not in source
    assert "d4_frozen_inputs import" not in source
    assert "field_program_signature import" not in source
    assert "run_n2_pvgr_n4_evaluator_convergence as n4" not in source
    assert "run_n2_pvgr_n0_trifidelity_development import" not in source
    imported_modules = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden_suffixes = {
        "run_n2_pvgr_n5_d4_tiny_field_derivative",
        "field_jvp_vjp_gate",
        "d4_frozen_inputs",
        "field_program_signature",
        "run_n2_pvgr_n4_evaluator_convergence",
        "run_n2_pvgr_n0_trifidelity_development",
    }
    assert not any(
        module.split(".")[-1] in forbidden_suffixes for module in imported_modules
    )
    assert "__import__" not in source
    assert "import_module" not in source


def test_attestation_artifacts_publish_exclusively_and_without_staging_leaks(
    tmp_path,
) -> None:
    archive = tmp_path / "frozen.npz"
    arrays = {"probe": np.arange(6, dtype=np.float64).reshape(2, 3)}
    _write_npz_atomic(archive, arrays)
    assert archive.is_file()
    assert not _temporary_sibling(archive).exists()
    with np.load(archive, allow_pickle=False) as payload:
        assert np.array_equal(payload["probe"], arrays["probe"])
    with pytest.raises(FileExistsError):
        _write_npz_atomic(archive, arrays)

    attestation = tmp_path / "attestation.json"
    _write_bytes_atomic(attestation, b'{"status":"bound"}\n')
    assert attestation.read_bytes() == b'{"status":"bound"}\n'
    assert not _temporary_sibling(attestation).exists()
    with pytest.raises(FileExistsError):
        _write_bytes_atomic(attestation, b"replacement forbidden")


def test_atomic_publication_crash_after_link_is_preserved_and_fails_closed(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "attestation.json"

    def injected_crash(_path) -> None:
        raise RuntimeError("injected crash after exclusive link")

    monkeypatch.setattr(attestation_builder, "_fsync_directory", injected_crash)
    with pytest.raises(RuntimeError, match="injected crash"):
        attestation_builder._write_bytes_atomic(destination, b"complete bytes\n")
    staging = attestation_builder._temporary_sibling(destination)
    assert destination.read_bytes() == b"complete bytes\n"
    assert staging.read_bytes() == b"complete bytes\n"
    with pytest.raises(FileExistsError):
        attestation_builder._write_bytes_atomic(destination, b"retry forbidden\n")


def test_output_occupancy_detects_dangling_symlink(tmp_path) -> None:
    occupied = tmp_path / "formal-output"
    occupied.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    assert _path_occupied(occupied) is True
    assert attestation_builder._path_occupied(occupied) is True


def test_validation_report_crash_after_link_preserves_staging_and_blocks_retry(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "validation_report.json"
    report = {"valid": True, "decision": "frozen"}

    def injected_crash(_path, _flags) -> int:
        raise RuntimeError("injected validation directory-sync crash")

    monkeypatch.setattr(independent_validator.os, "open", injected_crash)
    with pytest.raises(RuntimeError, match="directory-sync crash"):
        independent_validator._write_json(destination, report)
    staging = destination.with_name(f".{destination.name}.tmp")
    assert destination.is_file()
    assert staging.is_file()
    assert destination.read_bytes() == staging.read_bytes()
    with pytest.raises(FileExistsError, match="staging artifact exists"):
        independent_validator._write_json(destination, report)
