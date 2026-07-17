from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


MODULE_PATH = Path(__file__).with_name("run_oerf_verification_matrix.py")
SPEC = importlib.util.spec_from_file_location("run_oerf_verification_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _completed(command: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _commands(mock_run: Mock) -> list[list[str]]:
    return [call.args[0] for call in mock_run.call_args_list]


def _use_missing_fast_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(MODULE, "EXISTING_ARTIFACT", tmp_path / "missing")


def test_fast_commands_and_single_thread_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_missing_fast_artifact(monkeypatch, tmp_path)
    run = Mock(side_effect=lambda command, **_: _completed(command))
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    MODULE.run_matrix("fast")

    commands = _commands(run)
    assert commands[0][:3] == [sys.executable, "-m", "pytest"]
    assert commands[0][3:] == [
        *MODULE.FAST_CONTRACT_TESTS,
        *MODULE.GATE_A_TARGETED_TESTS,
    ]
    assert commands[1] == ["git", "diff", "--check"]
    assert len(commands) == 2
    for call in run.call_args_list:
        env = call.kwargs["env"]
        assert {name: env[name] for name in MODULE.THREAD_ENVIRONMENT} == (
            MODULE.THREAD_ENVIRONMENT
        )
        assert call.kwargs["check"] is True


def test_medium_uses_fixed_roots_then_serial_mps_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_missing_fast_artifact(monkeypatch, tmp_path)
    run = Mock(side_effect=lambda command, **_: _completed(command))
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    MODULE.run_matrix("medium")

    commands = _commands(run)
    parallel = commands[2]
    assert parallel[:3] == [sys.executable, "-m", "pytest"]
    assert parallel[3:7] == list(MODULE.SOURCE_ROOTS)
    assert parallel[7:] == [
        "-n",
        "4",
        "--dist=loadfile",
        f"--deselect={MODULE.MPS_CASE}",
    ]
    assert commands[3] == [sys.executable, "-m", "pytest", MODULE.MPS_CASE]
    assert "." not in parallel[3:]


def test_failure_stops_later_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_missing_fast_artifact(monkeypatch, tmp_path)
    failure = subprocess.CalledProcessError(2, ["git", "diff", "--check"])
    run = Mock(side_effect=[_completed([]), failure])
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        MODULE.run_matrix("full")

    assert run.call_count == 2


def test_full_outputs_are_unique_and_checks_follow_build_and_link_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_missing_fast_artifact(monkeypatch, tmp_path)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        stdout = "abc1234\n" if command[:3] == ["git", "rev-parse", "--short"] else ""
        return _completed(command, stdout)

    run = Mock(side_effect=fake_run)
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    MODULE.run_matrix("full")
    MODULE.run_matrix("full")

    commands = _commands(run)
    build_commands = [
        command
        for command in commands
        if any(token.endswith("/build_pages_artifact.py") for token in command)
    ]
    assert len(build_commands) == 2
    outputs = [command[command.index("--output") + 1] for command in build_commands]
    assert outputs[0] != outputs[1]
    assert all(output.startswith("build/pages-audit-abc1234-") for output in outputs)

    first_build_index = commands.index(build_commands[0])
    full_tail = commands[first_build_index : first_build_index + 7]
    assert any(token.endswith("/build_pages_artifact.py") for token in full_tail[0])
    assert full_tail[1] == ["git", "status", "--porcelain", "--untracked-files=all"]
    assert any(token.endswith("/audit_public_links.py") for token in full_tail[2])
    assert full_tail[3][-3:-1] == ["required-files", "--artifact"]
    assert full_tail[4][-3:-1] == ["sensitive-suffixes", "--artifact"]
    assert full_tail[5][-3:-1] == ["private-paths", "--artifact"]
    assert full_tail[6] == ["git", "status", "--porcelain", "--untracked-files=all"]


def test_full_rejects_dirty_worktree_before_sha_or_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_missing_fast_artifact(monkeypatch, tmp_path)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        stdout = " M changed.py\n" if command[:2] == ["git", "status"] else ""
        return _completed(command, stdout)

    run = Mock(side_effect=fake_run)
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="clean git worktree"):
        MODULE.run_matrix("full")
    commands = _commands(run)
    assert commands == [["git", "status", "--porcelain", "--untracked-files=all"]]
    assert not any(
        len(command) > 1 and command[1] == "site_tools/build_pages_artifact.py"
        for command in commands
    )


def test_dry_run_executes_no_subprocess_and_contains_no_scientific_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _use_missing_fast_artifact(monkeypatch, tmp_path)
    run = Mock()
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    MODULE.run_matrix("full", dry_run=True)

    run.assert_not_called()
    output = capsys.readouterr().out
    assert "run_benchmark.py" not in output
    assert "run_m3b_" not in output
    assert "run_psu_" not in output


def test_artifact_checks_required_files_and_sensitive_suffixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)
    artifact = tmp_path / "build/artifact"
    artifact.mkdir(parents=True)
    (artifact / "index.html").write_text("ok", encoding="utf-8")
    (artifact / "pages-build-manifest.json").write_text("{}", encoding="utf-8")
    relative = artifact.relative_to(tmp_path)

    MODULE.check_artifact("required-files", relative)
    MODULE.check_artifact("sensitive-suffixes", relative)
    MODULE.check_artifact("private-paths", relative)
    (artifact / "weights.ckpt").write_bytes(b"private")
    with pytest.raises(RuntimeError, match="sensitive suffixes"):
        MODULE.check_artifact("sensitive-suffixes", relative)
    (artifact / "weights.ckpt").unlink()
    (artifact / "truth_weights.npz").write_bytes(b"private")
    with pytest.raises(RuntimeError, match="sensitive suffixes"):
        MODULE.check_artifact("sensitive-suffixes", relative)
    private_path = artifact / "private_library/secret.txt"
    private_path.parent.mkdir()
    private_path.write_text("private", encoding="utf-8")
    with pytest.raises(RuntimeError, match="private_library paths"):
        MODULE.check_artifact("private-paths", relative)
