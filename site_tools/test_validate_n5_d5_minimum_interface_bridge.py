from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from site_tools import validate_n5_d5_minimum_interface_bridge as validator
from site_tools.run_n5_d5_minimum_interface_bridge import ROOT, run


CONFIG = (
    ROOT
    / "demo_t16_operator/configs/"
    "n5_d5_minimum_interface_bridge_preregistered_v1.json"
)


def _current_source_at_commit(commit: str, relative: str) -> bytes:
    del commit
    return (ROOT / relative).read_bytes()


@pytest.fixture()
def result_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(validator, "_git_show", _current_source_at_commit)
    output = tmp_path / "result"
    run(config_path=CONFIG, output_dir=output, enforce_committed=False)
    return output


def _refresh_manifest(result_dir: Path, name: str) -> None:
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = result_dir / name
    manifest["files"][name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_independent_validator_replays_every_call(result_dir: Path) -> None:
    report = validator.validate(result_dir, write_report=False)

    assert report["valid"] is True
    assert report["machine_decision_recomputed"] == (
        "SYNTHETIC_PROTOCOL_PASS_NO_LAB_AUTHORIZATION"
    )
    assert report["check_count"] > 500
    independence = report["independence_contract"]
    assert independence["runner_imported"] is False
    assert independence["shared_protocol_helper_imported"] is False
    assert independence["adapter_imported"] is False
    assert all(
        independence[key]
        for key in (
            "vectors_regenerated",
            "fresh_nonces_used",
            "adapter_relaunched",
            "all_forward_calls_replayed",
            "all_metrics_recomputed",
        )
    )
    assert not any(report["claim_boundary"].values())


def test_tampered_output_fails_after_manifest_refresh(result_dir: Path) -> None:
    path = result_dir / "responses.jsonl"
    rows = _read_jsonl(path)
    target = next(row for row in rows if row["request_id"] == "forward-base-curved-repeat-0")
    target["output"][0] += 0.125
    _write_jsonl(path, rows)
    _refresh_manifest(result_dir, "responses.jsonl")

    with pytest.raises(ValueError, match="independent replay mismatch"):
        validator.validate(result_dir, write_report=False)


def test_tampered_branch_state_fails_after_manifest_refresh(result_dir: Path) -> None:
    path = result_dir / "responses.jsonl"
    rows = _read_jsonl(path)
    target = next(row for row in rows if row["request_id"] == "forward-base-curved-repeat-0")
    target["branch_state"]["control_flow_id"] = "wrapper:tampered"
    _write_jsonl(path, rows)
    _refresh_manifest(result_dir, "responses.jsonl")

    with pytest.raises(ValueError, match="independent replay mismatch"):
        validator.validate(result_dir, write_report=False)


def test_tampered_metric_fails_after_manifest_refresh(result_dir: Path) -> None:
    path = result_dir / "metrics.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["summary"]["maximum_fd_relative_error_all_h"] = 0.25
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest(result_dir, "metrics.json")

    with pytest.raises(ValueError, match="summary.*mismatch"):
        validator.validate(result_dir, write_report=False)


def test_tampered_decision_fails_after_manifest_refresh(result_dir: Path) -> None:
    path = result_dir / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["machine_decision"] = "LAB_INTERFACE_REPLAYED_NO_PHYSICS_AUTHORIZATION"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest(result_dir, "result.json")

    with pytest.raises(ValueError, match="machine decision mismatch"):
        validator.validate(result_dir, write_report=False)


def test_validator_has_no_runner_protocol_or_adapter_imports() -> None:
    source = Path(validator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "site_tools.run_n5_d5_minimum_interface_bridge",
        "demo_t16_operator.n5_d5_adapter_protocol",
        "demo_t16_operator.n5_d5_synthetic_reference_adapter",
    }
    assert imported.isdisjoint(forbidden)


def test_validator_rejects_stored_request_lookup_attack(
    result_dir: Path,
) -> None:
    path = result_dir / "requests.jsonl"
    rows = _read_jsonl(path)
    rows[2]["x"][0] += 0.01
    _write_jsonl(path, rows)
    _refresh_manifest(result_dir, "requests.jsonl")

    with pytest.raises(ValueError, match="stored request trace mismatch"):
        validator.validate(result_dir, write_report=False)
