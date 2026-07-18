from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from site_tools.n5_d5_private_lab_readiness import (
    PUBLIC_READINESS_SOURCES,
    ROOT,
    _safe_private_output,
    build_readiness_report,
    sha256_file,
)


PLACEHOLDER = ROOT / "data_templates/n5_d5_lab_interface.placeholder.json"
SCHEMA = ROOT / "data_templates/n5_d5_minimum_bost_interface.schema.json"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


@pytest.fixture()
def private_bundle(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("private_library/\n", encoding="utf-8")
    for relative in PUBLIC_READINESS_SOURCES:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("n5_d5_minimum_bost_interface.schema.json"):
            shutil.copy2(SCHEMA, target)
        else:
            target.write_text(f"# public fixture: {relative}\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "n5-d5-test@example.invalid")
    _git(repo, "config", "user.name", "N5 D5 Test")
    _git(repo, "add", ".gitignore", *PUBLIC_READINESS_SOURCES)
    _git(repo, "commit", "-qm", "public protocol fixture")

    private = repo / "private_library" / "anonymous_case"
    private.mkdir(parents=True)
    adapter = private / "adapter.py"
    adapter.write_text(
        "from __future__ import annotations\n"
        "def renderer_entry(request):\n"
        "    return request\n",
        encoding="utf-8",
    )
    base = private / "base.npy"
    np.save(base, np.linspace(-0.1, 0.1, 6, dtype=np.float64))
    config = json.loads(PLACEHOLDER.read_text(encoding="utf-8"))
    config["identity"].update(
        {
            "bundle_id": "anon_bundle_43f2",
            "context_id": "anon_context_701a",
        }
    )
    config["adapter"].update(
        {
            "command": [
                "{python}",
                "private_library/anonymous_case/adapter.py",
            ],
            "expected_adapter_id": "private_renderer_43f2",
            "expected_adapter_version": "reviewed-v1",
            "expected_implementation_sha256": sha256_file(adapter),
            "source_files": [
                "private_library/anonymous_case/adapter.py",
            ],
        }
    )
    config["field"].update(
        {
            "input_dimension": 6,
            "units": "dimensionless_refractive_index",
            "axis_order": "anonymous_flattened_zyx",
            "base_input": {
                "source": "npy_relative",
                "seed": None,
                "relative_path": "private_library/anonymous_case/base.npy",
                "sha256": sha256_file(base),
            },
        }
    )
    config["observation"].update(
        {
            "units": "detector_displacement_pixel",
            "component_order": "ray_major_uv",
        }
    )
    for index, path in enumerate(config["paths"]):
        role = path["role"]
        path.update(
            {
                "path_id": f"private_{role}_path_v1",
                "callable_id": f"private_{role}_callable_v1",
                "semantic_digest_sha256": hashlib.sha256(
                    f"private:{index}:{role}".encode()
                ).hexdigest(),
            }
        )
    config_path = private / "config.json"
    _write_config(config_path, config)
    return {
        "repo": repo,
        "private": private,
        "adapter": adapter,
        "base": base,
        "config": config,
        "config_path": config_path,
    }


def _report(bundle: dict[str, object]) -> dict[str, object]:
    return build_readiness_report(
        bundle["config_path"],
        repo_root=bundle["repo"],
        enforce_public_committed=True,
    )


def _codes(report: dict[str, object]) -> set[str]:
    return set(report["blocker_codes"])


def test_complete_private_bundle_is_static_ready_but_formal_replay_locked(
    private_bundle: dict[str, object],
) -> None:
    report = _report(private_bundle)

    assert report["status"] == "STATIC_PRIVATE_INTAKE_READY_FORMAL_REPLAY_LOCKED"
    assert report["ready_for_private_describe_probe"] is True
    assert report["formal_53_call_replay_authorized"] is False
    assert report["blocker_count"] == 0
    assert set(report["warning_codes"]) == {
        "FORMAL_REPLAY_PRIVATE_ATTESTATION_AVAILABLE",
        "PHYSICAL_TOLERANCES_REVIEWED",
        "PRIVATE_DEPENDENCY_CLOSURE_REVIEWED",
        "FORMAL_REPLAY_CLOSED_WORLD_MANIFEST_AVAILABLE",
        "LAB_PUBLIC_SUMMARY_HARD_GUARD_AVAILABLE",
        "UNPREDICTABLE_PRIVATE_PROBES_AVAILABLE",
    }
    assert not any(report["claim_authorizations"].values())
    serialized = json.dumps(report)
    assert str(private_bundle["repo"]) not in serialized
    assert "private_library/anonymous_case" not in serialized
    assert len(report["private_inventory"]) == 3


def test_config_outside_private_root_is_blocked(
    private_bundle: dict[str, object],
) -> None:
    public_config = private_bundle["repo"] / "config.json"
    shutil.copy2(private_bundle["config_path"], public_config)

    report = build_readiness_report(
        public_config,
        repo_root=private_bundle["repo"],
        enforce_public_committed=True,
    )

    assert report["ready_for_private_describe_probe"] is False
    assert "PRIVATE_CONFIG_UNDER_PRIVATE_ROOT" in _codes(report)


def test_tracked_private_source_is_blocked(
    private_bundle: dict[str, object],
) -> None:
    repo = private_bundle["repo"]
    relative = private_bundle["adapter"].relative_to(repo).as_posix()
    _git(repo, "add", "-f", relative)

    report = _report(private_bundle)

    assert "ADAPTER_SOURCE_0_NOT_TRACKED" in _codes(report)


def test_symlinked_private_source_is_blocked(
    private_bundle: dict[str, object],
) -> None:
    adapter = private_bundle["adapter"]
    target = private_bundle["private"] / "adapter-target.py"
    adapter.rename(target)
    adapter.symlink_to(target.name)
    config = copy.deepcopy(private_bundle["config"])
    config["adapter"]["expected_implementation_sha256"] = sha256_file(target)
    _write_config(private_bundle["config_path"], config)

    report = _report(private_bundle)

    assert "ADAPTER_SOURCE_0_NO_SYMLINK_COMPONENT" in _codes(report)


def test_hardlinked_private_source_is_blocked(
    private_bundle: dict[str, object],
) -> None:
    adapter = private_bundle["adapter"]
    hardlink = private_bundle["private"] / "adapter-hardlink.py"
    hardlink.hardlink_to(adapter)

    report = _report(private_bundle)

    assert "ADAPTER_SOURCE_0_NOT_HARDLINKED" in _codes(report)


def test_implementation_hash_mismatch_is_blocked(
    private_bundle: dict[str, object],
) -> None:
    config = copy.deepcopy(private_bundle["config"])
    config["adapter"]["expected_implementation_sha256"] = "f" * 64
    _write_config(private_bundle["config_path"], config)

    report = _report(private_bundle)

    assert "IMPLEMENTATION_HASH_MATCHES_ENTRY_SCRIPT" in _codes(report)


@pytest.mark.parametrize(
    ("array", "expected_code"),
    [
        (np.zeros(5, dtype=np.float64), "BASE_INPUT_SIZE_MATCHES"),
        (np.zeros(6, dtype=np.float32), "BASE_INPUT_DTYPE_MATCHES"),
        (
            np.asarray([0, 0, 0, 0, 0, np.nan], dtype=np.float64),
            "BASE_INPUT_FINITE",
        ),
    ],
)
def test_invalid_base_array_is_blocked(
    private_bundle: dict[str, object],
    array: np.ndarray,
    expected_code: str,
) -> None:
    np.save(private_bundle["base"], array)
    config = copy.deepcopy(private_bundle["config"])
    config["field"]["base_input"]["sha256"] = sha256_file(private_bundle["base"])
    _write_config(private_bundle["config_path"], config)

    report = _report(private_bundle)

    assert expected_code in _codes(report)


def test_adapter_placeholders_network_secret_and_absolute_path_are_blocked(
    private_bundle: dict[str, object],
) -> None:
    private_bundle["adapter"].write_text(
        "import requests\n"
        "password = 'example-not-a-real-credential'\n"
        "debug_path = '/Users/example/private/input.npy'\n"
        "raise NotImplementedError('REPLACE_ME')\n",
        encoding="utf-8",
    )
    config = copy.deepcopy(private_bundle["config"])
    config["adapter"]["expected_implementation_sha256"] = sha256_file(
        private_bundle["adapter"]
    )
    _write_config(private_bundle["config_path"], config)

    report = _report(private_bundle)
    blockers = _codes(report)

    assert "ADAPTER_SOURCE_0_IMPLEMENTED" in blockers
    assert "ADAPTER_SOURCE_0_NO_NETWORK_IMPORT" in blockers
    assert "ADAPTER_SOURCE_0_NO_SECRET_LITERAL" in blockers
    assert "ADAPTER_SOURCE_0_NO_ABSOLUTE_PATH_LITERAL" in blockers


def test_dirty_public_protocol_source_is_blocked(
    private_bundle: dict[str, object],
) -> None:
    public_source = (
        private_bundle["repo"] / "site_tools/n5_d5_private_lab_readiness.py"
    )
    public_source.write_text("# changed after commit\n", encoding="utf-8")

    report = _report(private_bundle)

    assert any(code.startswith("PUBLIC_SOURCE_CLEAN_") for code in _codes(report))


def test_private_report_output_cannot_escape_or_overwrite(
    private_bundle: dict[str, object],
) -> None:
    repo = private_bundle["repo"]
    with pytest.raises(ValueError, match="private_library"):
        _safe_private_output(repo, repo / "public-report.json")

    output = private_bundle["private"] / "report.json"
    assert _safe_private_output(repo, output) == output.resolve()
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing"):
        _safe_private_output(repo, output)
