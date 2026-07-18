from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import textwrap

from jsonschema import Draft202012Validator
import pytest

import site_tools.n5_d5_l2b_describe_runner as runner


pytestmark = pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").is_file(),
    reason="L2-B mechanism tests require the local macOS sandbox",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True
    )


def _descriptor(source_sha256: str, **extra: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "n5-d5-describe-descriptor-1.0",
        "adapter_id": "anonymous_describe_fixture",
        "adapter_version": "fixture-v1",
        "implementation_sha256": source_sha256,
        "context_id": "anonymous_context_fixture",
        "capability_profile": {
            "path_roles": ["curved", "straight"],
            "native_direct_residual": False,
            "forward_supported": True,
            "jvp_supported": True,
            "vjp_supported": True,
        },
        "claim_boundary": {
            "external_describe_requests_only": True,
            "internal_operation_absence_proven": False,
            "real_bost_authorized": False,
            "physical_correctness_authorized": False,
        },
    }
    value.update(extra)
    return value


def _valid_adapter_source(extra_descriptor_source: str = "") -> str:
    template = textwrap.dedent(
        f"""
        import hashlib
        import json
        from pathlib import Path
        import sys

        descriptor = {{
            "schema": "n5-d5-describe-descriptor-1.0",
            "adapter_id": "anonymous_describe_fixture",
            "adapter_version": "fixture-v1",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "context_id": "anonymous_context_fixture",
            "capability_profile": {{
                "path_roles": ["curved", "straight"],
                "native_direct_residual": False,
                "forward_supported": True,
                "jvp_supported": True,
                "vjp_supported": True
            }},
            "claim_boundary": {{
                "external_describe_requests_only": True,
                "internal_operation_absence_proven": False,
                "real_bost_authorized": False,
                "physical_correctness_authorized": False
            }}
        }}
        __EXTRA_DESCRIPTOR_SOURCE__
        calls = 0
        for raw in sys.stdin:
            request = json.loads(raw)
            if request["schema"] != "n5-d5-describe-request-1.0":
                raise SystemExit(41)
            if request["operation"] != "describe":
                raise SystemExit(42)
            calls += 1
            response = {{
                "schema": "n5-d5-describe-response-1.0",
                "request_id": request["request_id"],
                "audit_nonce": request["audit_nonce"],
                "operation": "describe",
                "status": "ok",
                "descriptor": descriptor,
                "runtime_ledger": {{
                    "describe_calls": calls,
                    "forward_api_calls": 0,
                    "jvp_api_calls": 0,
                    "vjp_api_calls": 0
                }}
            }}
            sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\\n")
            sys.stdout.flush()
        """
    ).lstrip()
    return template.replace(
        "__EXTRA_DESCRIPTOR_SOURCE__", extra_descriptor_source.rstrip()
    )


def _limits(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "wall_timeout_seconds": 2.0,
        "cpu_seconds": 2,
        "stdout_bytes": 262_144,
        "stderr_bytes": 8_192,
        "open_files": 16,
        "processes": 1,
    }
    value.update(updates)
    return value


def _run_staged_script(
    source: str,
    *,
    payload: bytes,
    limits: dict[str, object] | None = None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="n5d5-l2b-test-", dir="/private/tmp") as temporary:
        stage = Path(temporary).resolve()
        script = stage / "adapter.py"
        script.write_text(source, encoding="utf-8")
        profile = runner.sandbox_profile(stage)
        return runner._run_bounded_process(
            [
                "/usr/bin/sandbox-exec",
                "-p",
                profile,
                str(runner.resolve_system_python()),
                "-E",
                "-S",
                "-B",
                str(script),
            ],
            cwd=stage,
            environment=runner.scrubbed_environment(stage),
            payload=payload,
            limits=limits or _limits(),
        )


def _request_payload() -> tuple[list[dict[str, object]], bytes]:
    requests = runner._describe_requests("a" * 64)
    payload = "".join(
        runner.canonical_json(value) + "\n" for value in requests
    ).encode("utf-8")
    return requests, payload


def test_authorization_schema_hard_codes_two_describe_and_zero_claims() -> None:
    schema = json.loads(
        (runner.ROOT / runner.AUTH_SCHEMA_RELATIVE).read_text(encoding="utf-8")
    )
    authorization = json.loads(
        (runner.ROOT / runner.AUTH_PLACEHOLDER_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    assert list(Draft202012Validator(schema).iter_errors(authorization)) == []
    changed = copy.deepcopy(authorization)
    changed["scope"]["operations"] = ["describe", "forward"]
    assert list(Draft202012Validator(schema).iter_errors(changed))
    changed = copy.deepcopy(authorization)
    changed["claim_authorizations"]["reconstruction"] = True
    assert list(Draft202012Validator(schema).iter_errors(changed))
    changed = copy.deepcopy(authorization)
    changed["expected_descriptor"]["physical_correctness_authorized"] = True
    assert list(Draft202012Validator(schema).iter_errors(changed))


def test_production_api_exposes_no_fixture_or_bypass_parameter() -> None:
    parameters = inspect.signature(runner.run_authorized_describe).parameters
    assert not any(
        marker in name.lower()
        for name in parameters
        for marker in ("fixture", "bypass", "override", "unsafe", "insecure")
    )


def test_valid_adapter_returns_exactly_two_preauthorized_descriptors() -> None:
    requests, payload = _request_payload()
    source = _valid_adapter_source()
    process = _run_staged_script(source, payload=payload)

    assert process["returncode"] == 0
    assert process["timed_out"] is False
    assert process["stdout_overflow"] is False
    assert process["stderr"] == b""
    assert process["process_group_cleanup_verified"] is True
    expected = _descriptor(hashlib.sha256(source.encode("utf-8")).hexdigest())
    responses = runner._parse_responses(process["stdout"], requests, expected)
    assert len(responses) == 2
    assert [item["operation"] for item in requests] == ["describe", "describe"]


def test_scrubbed_environment_and_sandbox_effects_are_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, payload = _request_payload()
    monkeypatch.setenv("N5_D5_HOST_SENTINEL", "must-not-cross")
    with tempfile.TemporaryDirectory(
        prefix="n5d5-l2b-sibling-", dir="/private/tmp"
    ) as sibling:
        secret = Path(sibling).resolve() / "secret.txt"
        secret.write_text("must-not-be-readable", encoding="utf-8")
        source_fd = os.open(secret, os.O_RDONLY)
        os.dup2(source_fd, 99, inheritable=True)
        os.close(source_fd)
        probes = textwrap.dedent(
            f"""
            import os
            import socket
            import subprocess
            probes = {{}}
            probes["host_sentinel_visible"] = os.getenv("N5_D5_HOST_SENTINEL") is not None
            try:
                os.fstat(99)
                probes["host_fd_99_inherited"] = True
            except Exception:
                probes["host_fd_99_inherited"] = False
            try:
                child = os.fork()
                if child == 0:
                    os.setsid()
                    os._exit(0)
                os.waitpid(child, 0)
                probes["fork_detach_allowed"] = True
            except Exception:
                probes["fork_detach_allowed"] = False
            try:
                Path({str(secret)!r}).read_text()
                probes["sibling_tmp_read_allowed"] = True
            except Exception:
                probes["sibling_tmp_read_allowed"] = False
            try:
                Path("sandbox-write-probe.txt").write_text("x")
                probes["file_write_allowed"] = True
            except Exception:
                probes["file_write_allowed"] = False
            sock = socket.socket()
            probes["tcp_connect_errno"] = sock.connect_ex(("127.0.0.1", 9))
            sock.close()
            try:
                subprocess.run(["/usr/bin/true"], check=True)
                probes["subprocess_allowed"] = True
            except Exception:
                probes["subprocess_allowed"] = False
            descriptor["mechanism_probes"] = probes
            """
        )
        source = _valid_adapter_source(probes)
        try:
            process = _run_staged_script(source, payload=payload)
        finally:
            os.close(99)

    assert process["returncode"] == 0
    expected = _descriptor(
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
        mechanism_probes={
            "host_sentinel_visible": False,
            "host_fd_99_inherited": False,
            "fork_detach_allowed": False,
            "sibling_tmp_read_allowed": False,
            "file_write_allowed": False,
            "tcp_connect_errno": 1,
            "subprocess_allowed": False,
        },
    )
    responses = runner._parse_responses(process["stdout"], requests, expected)
    assert responses[1]["descriptor"]["mechanism_probes"] == expected[
        "mechanism_probes"
    ]


def test_stdout_flood_is_bounded_and_process_group_is_cleaned() -> None:
    _, payload = _request_payload()
    source = "import sys, time\nsys.stdout.write('x' * 200000)\nsys.stdout.flush()\ntime.sleep(30)\n"
    process = _run_staged_script(
        source,
        payload=payload,
        limits=_limits(stdout_bytes=4096, wall_timeout_seconds=1.0),
    )
    assert process["stdout_overflow"] is True
    assert len(process["stdout"]) == 4096
    assert process["returncode"] is not None
    assert process["process_group_cleanup_verified"] is True


def test_stderr_flood_is_bounded_and_process_group_is_cleaned() -> None:
    _, payload = _request_payload()
    source = "import sys, time\nsys.stderr.write('x' * 200000)\nsys.stderr.flush()\ntime.sleep(30)\n"
    process = _run_staged_script(
        source,
        payload=payload,
        limits=_limits(stderr_bytes=1024, wall_timeout_seconds=1.0),
    )
    assert process["stderr_overflow"] is True
    assert len(process["stderr"]) == 1024
    assert process["returncode"] is not None
    assert process["process_group_cleanup_verified"] is True


def test_one_response_then_hang_is_timed_out_fail_closed() -> None:
    _, payload = _request_payload()
    source = textwrap.dedent(
        """
        import json, sys, time
        request = json.loads(sys.stdin.readline())
        response = {
            "schema": "n5-d5-describe-response-1.0",
            "request_id": request["request_id"],
            "audit_nonce": request["audit_nonce"],
            "operation": "describe",
            "status": "ok",
            "descriptor": {},
            "runtime_ledger": {"describe_calls": 1, "forward_api_calls": 0, "jvp_api_calls": 0, "vjp_api_calls": 0}
        }
        print(json.dumps(response), flush=True)
        time.sleep(30)
        """
    )
    process = _run_staged_script(
        source,
        payload=payload,
        limits=_limits(wall_timeout_seconds=0.3),
    )
    assert process["timed_out"] is True
    assert process["process_group_cleanup_verified"] is True


def test_extra_line_and_forward_label_are_rejected() -> None:
    requests, payload = _request_payload()
    source = _valid_adapter_source() + "print('{}', flush=True)\n"
    process = _run_staged_script(source, payload=payload)
    expected = _descriptor(hashlib.sha256(source.encode("utf-8")).hexdigest())
    with pytest.raises(runner.L2BError, match="exactly two"):
        runner._parse_responses(process["stdout"], requests, expected)

    valid_source = _valid_adapter_source()
    process = _run_staged_script(valid_source, payload=payload)
    changed = process["stdout"].replace(
        b'"operation":"describe"', b'"operation":"forward"', 1
    )
    expected = _descriptor(
        hashlib.sha256(valid_source.encode("utf-8")).hexdigest()
    )
    with pytest.raises(runner.L2BError, match="operation mismatch"):
        runner._parse_responses(changed, requests, expected)


def test_direct_exec_replacement_cannot_produce_a_protocol_pass() -> None:
    requests, payload = _request_payload()
    source = "import os\nos.execv('/usr/bin/true', ['/usr/bin/true'])\n"
    process = _run_staged_script(source, payload=payload)
    assert process["returncode"] == 0
    assert process["stdout"] == b""
    with pytest.raises(runner.L2BError):
        runner._parse_responses(process["stdout"], requests, {})


def test_duplicate_keys_nonfinite_and_deep_json_are_rejected() -> None:
    with pytest.raises(runner.L2BError, match="duplicate JSON key"):
        runner._load_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(runner.L2BError, match="non-finite"):
        runner._load_json_bytes(b'{"a":NaN}')
    nested = b"[" * 24 + b"0" + b"]" * 24
    with pytest.raises(runner.L2BError, match="nesting"):
        runner._load_json_bytes(nested)


@pytest.fixture()
def authorized_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("private_library/\n", encoding="utf-8")
    schema_source = runner.ROOT / runner.AUTH_SCHEMA_RELATIVE
    schema_target = repo / runner.AUTH_SCHEMA_RELATIVE
    schema_target.parent.mkdir(parents=True)
    shutil.copy2(schema_source, schema_target)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "l2b-test@example.invalid")
    _git(repo, "config", "user.name", "L2B Test")
    _git(repo, "add", ".gitignore", runner.AUTH_SCHEMA_RELATIVE)
    _git(repo, "commit", "-qm", "public L2-B fixture")

    bundle = repo / "private_library/n5_d5_bundles/anonymous_case"
    control = repo / "private_library/n5_d5_control/anonymous_case"
    runs = repo / "private_library/n5_d5_runs/anonymous_case"
    bundle.mkdir(parents=True)
    control.mkdir(parents=True)
    runs.mkdir(parents=True)
    source = _valid_adapter_source()
    adapter = bundle / "describe_adapter.py"
    adapter.write_text(source, encoding="utf-8")
    source_hash = runner.sha256_file(adapter)
    relative_adapter = adapter.relative_to(repo).as_posix()
    plan = {
        "schema_version": "fixture-private-plan",
        "private_bundle": {
            "root": bundle.relative_to(repo).as_posix(),
            "closed_world_required": True,
            "inventory": [
                {
                    "relative_path": relative_adapter,
                    "role": "adapter_source",
                    "sha256": source_hash,
                    "bytes": adapter.stat().st_size,
                }
            ],
        },
    }
    plan_path = control / "replay-plan.json"
    _write_json(plan_path, plan)
    foundation = {
        "schema_version": "fixture-l2a-foundation",
        "status": "PRIVATE_REPLAY_FOUNDATION_READY_EXECUTION_LOCKED",
        "foundation_ready": True,
        "adapter_executed": False,
        "private_describe_probe_authorized": False,
        "formal_replay_authorized": False,
        "foundation_blocker_codes": [],
        "private_plan_sha256": runner.sha256_bytes(
            runner.canonical_json(plan).encode("utf-8")
        ),
        "claim_authorizations": runner._claim_boundary(),
    }
    foundation_path = control / "l2a-foundation.json"
    _write_json(foundation_path, foundation)
    authorization = {
        "schema_version": "n5-d5-l2b-describe-authorization-1.0",
        "record_kind": "PRIVATE_DESCRIBE_ONLY_ONE_TIME_AUTHORIZATION",
        "authorization_id": "anonymous_auth_001",
        "one_time_nonce": secrets.token_hex(32),
        "human_approved": True,
        "foundation_report": {
            "relative_path": foundation_path.relative_to(repo).as_posix(),
            "sha256": runner.sha256_file(foundation_path),
        },
        "replay_plan": {
            "relative_path": plan_path.relative_to(repo).as_posix(),
            "sha256": runner.sha256_file(plan_path),
        },
        "describe_entrypoint": {
            "inventory_relative_path": relative_adapter,
            "interpreter": "/usr/bin/python3",
            "standard_library_only": True,
            "arguments": [],
        },
        "expected_descriptor": _descriptor(source_hash),
        "output_directory": (
            runs / "l2b-describe-001"
        ).relative_to(repo).as_posix(),
        "scope": {
            "operations": ["describe"],
            "external_request_count": 2,
            "auto_chain_primary": False,
            "internal_operation_absence_claimed": False,
        },
        "limits": _limits(),
        "claim_authorizations": runner._claim_boundary(),
    }
    authorization_path = control / "describe-authorization.json"
    _write_json(authorization_path, authorization)
    plan_payload = plan_path.read_bytes()
    plan_metadata = plan_path.stat()
    plan_identity = (
        int(plan_metadata.st_dev),
        int(plan_metadata.st_ino),
        int(plan_metadata.st_size),
        int(plan_metadata.st_mtime_ns),
    )

    def reproduce_foundation(*args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["plan_snapshot_payload"] == plan_payload
        assert kwargs["plan_snapshot_stat_identity"] == plan_identity
        return copy.deepcopy(foundation)

    monkeypatch.setattr(runner, "build_foundation_report", reproduce_foundation)
    return {
        "repo": repo,
        "authorization_path": authorization_path,
        "authorization": authorization,
        "foundation": foundation,
        "plan_path": plan_path,
        "output": repo / authorization["output_directory"],
        "nonce_marker": (
            repo
            / "private_library/n5_d5_control/.l2b_nonce_ledger"
            / f"{authorization['one_time_nonce']}.consumed"
        ),
    }


@pytest.fixture
def development_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "production_backend_capabilities",
        lambda: {
            "backend": "pytest_development_test_double",
            "production_ready": False,
            "development_test_double": True,
            "process_exec_replacement_denied": False,
            "private_input_root_external_mutation_denied": False,
            "durable_nonce_ledger_root_protected": False,
            "output_root_external_mutation_denied": False,
            "backend_capability_attestation_externally_verified": False,
            "blocker_codes": [
                "TEST_DOUBLE_NO_PRODUCTION_CONTAINMENT_CLAIM",
            ],
            "reason": "pytest-only test double",
        },
    )


def test_production_backend_blocks_before_authorization_consumption(
    authorized_repo: dict[str, object],
) -> None:
    with pytest.raises(runner.L2BError) as exc_info:
        runner.run_authorized_describe(
            authorized_repo["authorization_path"],
            repo_root=authorized_repo["repo"],
            enforce_public_committed=False,
        )
    for blocker in (
        "POST_LAUNCH_EXEC_REPLACEMENT_NOT_DENIED",
        "PRIVATE_INPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
        "DURABLE_NONCE_LEDGER_ROOT_NOT_PROTECTED",
        "OUTPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
        "BACKEND_CAPABILITY_ATTESTATION_NOT_EXTERNALLY_VERIFIED",
    ):
        assert blocker in str(exc_info.value)
    assert not authorized_repo["nonce_marker"].exists()
    assert not authorized_repo["output"].exists()


def test_direct_cli_reports_the_host_blocker_without_reading_authorization() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(runner.__file__).resolve()),
            "--authorization",
            "/private/tmp/n5d5-l2b-deliberately-missing.json",
        ],
        cwd=runner.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert completed.stderr == ""
    assert payload["status"] == "L2B_PRODUCTION_BACKEND_BLOCKED_FAIL_CLOSED"
    assert payload["authorization_consumed"] is False
    assert payload["adapter_executed"] is False
    for blocker in (
        "POST_LAUNCH_EXEC_REPLACEMENT_NOT_DENIED",
        "PRIVATE_INPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
        "DURABLE_NONCE_LEDGER_ROOT_NOT_PROTECTED",
        "OUTPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
        "BACKEND_CAPABILITY_ATTESTATION_NOT_EXTERNALLY_VERIFIED",
    ):
        assert blocker in payload["blocker"]


@pytest.mark.usefixtures("development_backend")
def test_development_fixture_flow_consumes_token_and_writes_private_closed_world(
    authorized_repo: dict[str, object],
) -> None:
    report = runner.run_authorized_describe(
        authorized_repo["authorization_path"],
        repo_root=authorized_repo["repo"],
        enforce_public_committed=False,
    )

    assert report["status"] == (
        "L2B_DEVELOPMENT_TEST_DOUBLE_PASS_NO_SCIENCE_AUTHORIZATION"
    )
    assert report["production_backend_ready"] is False
    assert report["production_execution_authorized"] is False
    assert report["development_test_double_used"] is True
    assert report["private_input_root_external_mutation_denied"] is False
    assert report["backend_capability_attestation_externally_verified"] is False
    assert report["backend_capability_trust_model"] == (
        "PYTEST_MONKEYPATCHED_DEVELOPMENT_TEST_DOUBLE"
    )
    assert report["global_nonce_uniqueness_proven"] is False
    assert report["nonce_uniqueness_scope"] == "CURRENT_OPEN_LEDGER_INODE_ONLY"
    assert report["external_requests_sent"] == 2
    assert report["validated_responses"] == 2
    assert report["external_operations_sent"] == ["describe", "describe"]
    assert report["forward_jvp_vjp_external_requests"] == 0
    assert report["internal_operation_absence_proven"] is False
    assert not any(report["claim_authorizations"].values())
    assert authorized_repo["nonce_marker"].is_file()
    validation = runner.validate_result_directory(
        authorized_repo["output"], repo_root=authorized_repo["repo"]
    )
    assert validation["valid"] is True
    responses = (authorized_repo["output"] / "responses.jsonl").read_text(
        encoding="utf-8"
    )
    assert responses.count("\n") == 2
    assert "forward\"" not in responses


@pytest.mark.usefixtures("development_backend")
def test_authorized_flow_validates_on_held_output_fd_without_path_reopen(
    authorized_repo: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_path_validator(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("run path must not reopen output for validation")

    monkeypatch.setattr(runner, "validate_result_directory", forbidden_path_validator)
    report = runner.run_authorized_describe(
        authorized_repo["authorization_path"],
        repo_root=authorized_repo["repo"],
        enforce_public_committed=False,
    )
    assert report["status"].startswith("L2B_DEVELOPMENT_TEST_DOUBLE_PASS")


@pytest.mark.usefixtures("development_backend")
def test_authorization_cannot_be_reused_even_if_output_is_removed(
    authorized_repo: dict[str, object],
) -> None:
    runner.run_authorized_describe(
        authorized_repo["authorization_path"],
        repo_root=authorized_repo["repo"],
        enforce_public_committed=False,
    )
    shutil.rmtree(authorized_repo["output"])
    with pytest.raises(FileExistsError):
        runner.run_authorized_describe(
            authorized_repo["authorization_path"],
            repo_root=authorized_repo["repo"],
            enforce_public_committed=False,
        )


@pytest.mark.usefixtures("development_backend")
def test_copied_authorization_cannot_reuse_the_same_nonce(
    authorized_repo: dict[str, object],
) -> None:
    runner.run_authorized_describe(
        authorized_repo["authorization_path"],
        repo_root=authorized_repo["repo"],
        enforce_public_committed=False,
    )
    copied = copy.deepcopy(authorized_repo["authorization"])
    copied["output_directory"] = (
        "private_library/n5_d5_runs/anonymous_case/l2b-describe-002"
    )
    copied_path = authorized_repo["authorization_path"].with_name(
        "copied-describe-authorization.json"
    )
    _write_json(copied_path, copied)
    with pytest.raises(FileExistsError):
        runner.run_authorized_describe(
            copied_path,
            repo_root=authorized_repo["repo"],
            enforce_public_committed=False,
        )


@pytest.mark.usefixtures("development_backend")
def test_tampered_plan_is_rejected_before_authorization_consumption(
    authorized_repo: dict[str, object],
) -> None:
    plan_path = authorized_repo["repo"] / authorized_repo["authorization"][
        "replay_plan"
    ]["relative_path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["schema_version"] = "tampered"
    _write_json(plan_path, plan)

    with pytest.raises(runner.L2BError, match="hash"):
        runner.run_authorized_describe(
            authorized_repo["authorization_path"],
            repo_root=authorized_repo["repo"],
            enforce_public_committed=False,
        )
    assert not authorized_repo["nonce_marker"].exists()


@pytest.mark.usefixtures("development_backend")
def test_plan_change_during_foundation_reproduction_is_rejected_before_consumption(
    authorized_repo: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_plan(*args: object, **kwargs: object) -> dict[str, object]:
        plan = json.loads(
            authorized_repo["plan_path"].read_text(encoding="utf-8")
        )
        plan["schema_version"] = "changed-during-reproduction"
        _write_json(authorized_repo["plan_path"], plan)
        return copy.deepcopy(authorized_repo["foundation"])

    monkeypatch.setattr(runner, "build_foundation_report", mutate_plan)
    with pytest.raises(runner.L2BError, match="changed while|hash does not match"):
        runner.run_authorized_describe(
            authorized_repo["authorization_path"],
            repo_root=authorized_repo["repo"],
            enforce_public_committed=False,
        )
    assert not authorized_repo["nonce_marker"].exists()
