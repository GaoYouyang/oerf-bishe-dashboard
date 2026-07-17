from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from site_tools.psu_rotation40_forward_mismatch import run_diagnostic
from site_tools.validate_psu_rotation40_forward_mismatch import validate_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "demo_t16_operator" / "configs" / "psu_rotation40_forward_mismatch_prereg_v1.json"
VECTOR_FIELDS = ("c", "v", "Ruvecs", "Rvvecs", "Rxvecs", "Ryvecs")
SCALAR_FIELDS = ("epsu_all", "epsv_all", "Csys_all", "Rapvec", "Dfvec")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _config(tmp_path: Path) -> Path:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["bounded_execution"]["rays_per_camera"] = 4
    path = tmp_path / "config.json"
    _write_json(path, config)
    return path


def _generator_info(config: dict) -> tuple[dict[str, str], str, str]:
    hashes = dict(config["generator"]["source_sha256"])
    canonical = "".join(f"{path}\0{hashes[path]}\n" for path in sorted(hashes))
    return hashes, hashlib.sha256(canonical.encode("ascii")).hexdigest(), config["generator"]["version"]


def _write_camera_identity_and_author(root: Path, camera_id: int, field_hash: str, geometry_hash: str, generator_fingerprint: str, generator_version: str) -> None:
    camera = root / f"camera_{camera_id:02d}"
    bundle_path = camera / "bundle" / "view_bundle_manifest.json"
    mask_path = camera / "corrected_masks" / "corrected_view_masks_manifest.json"
    identity_path = camera / "psu_rotation40_view_identity.json"
    identity = {
        "status": "PSU_ROTATION40_VIEW_IDENTITY_FROZEN",
        "camera_id": camera_id,
        "rotation_degrees": 40,
        "measurement_count": 6,
        "bundle_manifest_sha256": _sha256(bundle_path),
        "mask_manifest_sha256": _sha256(mask_path),
        "provenance": {
            "geometry_sha256": geometry_hash,
            "generator_source_fingerprint": generator_fingerprint,
            "generator_version": generator_version,
            "source_split": "ROTATION_40_DEVELOPMENT",
        },
    }
    _write_json(identity_path, identity)
    author_path = camera / "author_exact_forward_32cubed.npy"
    author_manifest = {
        "status": "AUTHOR_EXACT_FORWARD_32CUBED_FROZEN_FIELD",
        "field_sha256": field_hash,
        "geometry_sha256": geometry_hash,
        "camera_identity_sha256": _sha256(identity_path),
        "bundle_manifest_sha256": _sha256(bundle_path),
        "mask_manifest_sha256": _sha256(mask_path),
        "mask_sha256": _sha256(camera / "corrected_masks" / "amask_all_zero_based.npy"),
        "author_array_sha256": _sha256(author_path),
        "author_source_filename": "author_source/author_forward.py",
        "author_source_sha256": _sha256(root / "author_source" / "author_forward.py"),
        "generator_source_fingerprint": generator_fingerprint,
        "generator_version": generator_version,
        "measurement_count": 6,
        "row_order": "MATLAB_COLUMN_MAJOR_FIND_MINUS_ONE",
    }
    _write_json(camera / "author_exact_forward_manifest.json", author_manifest)


def _rebind_mask(root: Path, camera_id: int) -> None:
    camera = root / f"camera_{camera_id:02d}"
    mask_path = camera / "corrected_masks" / "corrected_view_masks_manifest.json"
    mask = json.loads(mask_path.read_text(encoding="utf-8"))
    mask_hash = _sha256(camera / "corrected_masks" / "amask_all_zero_based.npy")
    mask["mask_shards"][0]["sha256"] = mask_hash
    mask["provenance"]["mask_sha256"] = mask_hash
    _write_json(mask_path, mask)
    identity_path = camera / "psu_rotation40_view_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["mask_manifest_sha256"] = _sha256(mask_path)
    _write_json(identity_path, identity)
    author_path = camera / "author_exact_forward_manifest.json"
    author = json.loads(author_path.read_text(encoding="utf-8"))
    author["mask_manifest_sha256"] = _sha256(mask_path)
    author["camera_identity_sha256"] = _sha256(identity_path)
    _write_json(author_path, author)


def _bundle(tmp_path: Path) -> Path:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _, generator_fingerprint, generator_version = _generator_info(config)
    root = tmp_path / "private_bundle"
    root.mkdir()

    geometry_source = root / "geometry_source" / "meas.py"
    geometry_source.parent.mkdir()
    geometry_source.write_text("rotation40 geometry source fixture\n", encoding="utf-8")
    geometry_hash = _sha256(geometry_source)
    _write_json(
        root / "geometry_provenance.json",
        {
            "status": "ROTATION40_GEOMETRY_PROVENANCE_FROZEN",
            "source_filename": "geometry_source/meas.py",
            "source_sha256": geometry_hash,
            "source_split": "ROTATION_40_DEVELOPMENT",
            "frozen_before_rotation40_access": True,
            "generator_source_fingerprint": generator_fingerprint,
            "generator_version": generator_version,
        },
    )

    field = np.broadcast_to(np.linspace(-1.0, 1.0, 32, dtype=np.float64), (32, 32, 32)).copy()
    field_path = root / "frozen_support_field_32cubed.npy"
    np.save(field_path, field)
    field_hash = _sha256(field_path)
    _write_json(
        root / "frozen_support_field_manifest.json",
        {
            "status": "FROZEN_32CUBED_SUPPORT_FIELD",
            "grid_shape_zyx": [32, 32, 32],
            "bounds_m": {"minimum": [-0.11, -0.11, -0.11], "maximum": [0.11, 0.11, 0.11]},
            "field_sha256": field_hash,
            "geometry_sha256": geometry_hash,
            "source_split": "SUPPORT_ONLY_FROZEN_BEFORE_ROTATION_40_ACCESS",
            "frozen_before_rotation40_access": True,
            "generator_source_fingerprint": generator_fingerprint,
            "generator_version": generator_version,
        },
    )
    author_source = root / "author_source" / "author_forward.py"
    author_source.parent.mkdir()
    author_source.write_text("author source fixture; not executed by H2\n", encoding="utf-8")

    for camera_id in (2, 3, 4):
        camera = root / f"camera_{camera_id:02d}"
        bundle_root, mask_root = camera / "bundle", camera / "corrected_masks"
        bundle_root.mkdir(parents=True)
        mask_root.mkdir()
        count = 6
        origins = np.tile(np.array([[-0.2, 0.0, 0.0]], dtype=np.float64), (count, 1))
        if camera_id == 2:
            origins[0, 0] = 0.2
        arrays = {
            "c": origins,
            "v": np.tile(np.array([[1.0, 0.0, 0.0]]), (count, 1)),
            "Ruvecs": np.tile(np.array([[1.0, 0.0, 0.0]]), (count, 1)),
            "Rvvecs": np.tile(np.array([[0.0, 1.0, 0.0]]), (count, 1)),
            "Rxvecs": np.tile(np.array([[0.0, 1.0, 0.0]]), (count, 1)),
            "Ryvecs": np.tile(np.array([[0.0, 0.0, 1.0]]), (count, 1)),
            "Rapvec": np.full((count, 1), 0.004),
            "Dfvec": np.ones((count, 1)),
            "Csys_all": np.ones((count, 1)),
            "epsu_all": np.full((count, 1), 0.3),
            "epsv_all": np.full((count, 1), -0.15),
        }
        variables = []
        for name, array in arrays.items():
            path = bundle_root / f"{name}.npy"
            np.save(path, array)
            variables.append({"name": name, "shard_shape": list(array.shape), "shard_dtype": array.dtype.name, "shard_sha256": _sha256(path)})
        _write_json(
            bundle_root / "view_bundle_manifest.json",
            {
                "status": "VIEW_SHARD_BUNDLE_TRANSCODED_AND_SOURCE_STREAMS_VERIFIED",
                "aggregate": {"variable_count": 11},
                "view": {"camera_id": camera_id, "rotation_degrees": 40, "measurement_count": count},
                "variables": variables,
                "provenance": {"geometry_sha256": geometry_hash, "generator_source_fingerprint": generator_fingerprint, "generator_version": generator_version, "source_split": "ROTATION_40_DEVELOPMENT"},
            },
        )
        active_path = mask_root / "amask_all_zero_based.npy"
        np.save(active_path, np.arange(count, dtype=np.int64))
        mask_hash = _sha256(active_path)
        _write_json(
            mask_root / "corrected_view_masks_manifest.json",
            {
                "status": "CORRECTED_VIEW_MASK_SHARDS_BUILT_MECHANICAL_CONTRACT_PASS",
                "view": {"camera_id": camera_id, "rotation_degrees": 40, "measurement_count": count},
                "mask_shards": [{"variable": "amask_all", "filename": "amask_all_zero_based.npy", "sha256": mask_hash, "shape": [count], "dtype": "int64"}],
                "provenance": {"geometry_sha256": geometry_hash, "bundle_manifest_sha256": _sha256(bundle_root / "view_bundle_manifest.json"), "mask_sha256": mask_hash, "generator_source_fingerprint": generator_fingerprint, "generator_version": generator_version, "source_split": "ROTATION_40_DEVELOPMENT"},
            },
        )
        author_path = camera / "author_exact_forward_32cubed.npy"
        np.save(author_path, np.tile(np.array([[0.25, -0.1]], dtype=np.float64), (count, 1)))
        _write_camera_identity_and_author(root, camera_id, field_hash, geometry_hash, generator_fingerprint, generator_version)
    return root


def test_h2_runner_reports_nonzero_b0_sensitivity_and_centerline_misses(tmp_path: Path) -> None:
    report = run_diagnostic(config_path=_config(tmp_path), private_bundle_root=_bundle(tmp_path), private_output=tmp_path / "private_report.json")
    assert report["status"].startswith("H2_FORWARD_MISMATCH_DIAGNOSTIC_COMPLETE")
    assert report["configuration"]["qmc_interpretation"] == "B0_SENSITIVITY_ONLY_AUTHOR_ARRAY_IS_SINGLE_FIXED_ARTIFACT"
    assert len(report["per_camera"]) == 9
    assert [row["qmc_sample_count"] for row in report["pooled"]] == [8, 16, 32]
    assert any(row["metrics"]["forward_mismatch_l2"] > 0.0 for row in report["per_camera"])
    assert report["per_camera"][0]["domain"]["centerline_miss_count"] == 1
    assert validate_report(report)["status"] == "VALID"


def test_h2_runner_fails_closed_when_author_forward_is_mutated(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    author_path = root / "camera_03" / "author_exact_forward_32cubed.npy"
    values = np.load(author_path)
    values[0, 0] += 1.0
    np.save(author_path, values)
    with pytest.raises(ValueError, match="author_array_sha256"):
        run_diagnostic(config_path=_config(tmp_path), private_bundle_root=root, private_output=tmp_path / "private_report.json")


def test_h2_runner_rejects_negative_active_index(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    active_path = root / "camera_03" / "corrected_masks" / "amask_all_zero_based.npy"
    active = np.load(active_path)
    active[0] = -1
    np.save(active_path, active)
    _rebind_mask(root, 3)
    with pytest.raises(ValueError, match="negative index"):
        run_diagnostic(config_path=_config(tmp_path), private_bundle_root=root, private_output=tmp_path / "private_report.json")


def test_independent_validator_rejects_inf_metric(tmp_path: Path) -> None:
    report = run_diagnostic(config_path=_config(tmp_path), private_bundle_root=_bundle(tmp_path), private_output=tmp_path / "private_report.json")
    report["pooled"][0]["forward_mismatch_l2"] = float("inf")
    verdict = validate_report(report)
    assert verdict["status"] == "INVALID"
    assert any("non-finite" in error for error in verdict["errors"])


def test_independent_validator_rejects_provenance_mutation(tmp_path: Path) -> None:
    report = run_diagnostic(config_path=_config(tmp_path), private_bundle_root=_bundle(tmp_path), private_output=tmp_path / "private_report.json")
    report["per_camera"][1]["provenance"]["geometry_sha256"] = "f" * 64
    verdict = validate_report(report)
    assert verdict["status"] == "INVALID"
    assert any("geometry_sha256 is inconsistent" in error for error in verdict["errors"])
