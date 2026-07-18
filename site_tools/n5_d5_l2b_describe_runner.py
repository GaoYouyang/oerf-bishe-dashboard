#!/usr/bin/env python3
"""Run one authorized, private, exactly-two-describe N5-D5 L2-B probe.

This runner stages the closed-world private bundle into an ephemeral directory
and accepts exactly two descriptor responses.  It never sends forward, JVP, or
VJP requests and never chains into the primary replay.  The current macOS backend
cannot deny post-launch exec replacement, so the production entrypoint fails
closed before consuming an authorization.  An explicit temporary-repository-only
development mode exists for adversarial fixture tests and never yields a
production-pass status.

The macOS ``sandbox-exec`` facility is deprecated.  The attestation therefore
records an observed local mechanism, not a portable or absolute containment
guarantee.  In particular, two external describe requests cannot prove that an
adapter did not perform hidden internal work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, BinaryIO

from jsonschema import Draft202012Validator

try:
    from site_tools.n5_d5_private_replay_foundation import (
        build_foundation_report,
    )
except ModuleNotFoundError:  # direct script execution from the repository root
    from n5_d5_private_replay_foundation import build_foundation_report


ROOT = Path(__file__).resolve().parents[1]
AUTH_SCHEMA_RELATIVE = (
    "data_templates/n5_d5_l2b_describe_authorization.schema.json"
)
AUTH_PLACEHOLDER_RELATIVE = (
    "data_templates/n5_d5_l2b_describe_authorization.placeholder.json"
)
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SYSTEM_PYTHON = Path("/usr/bin/python3")
REQUEST_SCHEMA = "n5-d5-describe-request-1.0"
RESPONSE_SCHEMA = "n5-d5-describe-response-1.0"
RESULT_SCHEMA = "n5-d5-l2b-describe-result-manifest-1.0"
RESULT_FILES = (
    "requests.jsonl",
    "responses.jsonl",
    "process_attestation.json",
)
CLAIM_KEYS = (
    "real_bost_interface",
    "physical_forward_correctness",
    "derivative_correctness",
    "reconstruction",
    "algorithm_superiority",
    "generalization",
    "publication",
)
ALLOWED_ENVIRONMENT = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "TMPDIR",
)
PLACEHOLDER_MARKERS = (
    "REPLACE_ME",
    "replace_describe_authorization_id",
)
MAX_JSON_DEPTH = 20
MAX_JSON_STRING = 16_384
MAX_JSON_ITEMS = 512
MAX_CONTROL_FILE_BYTES = 4 * 1024 * 1024


class L2BError(RuntimeError):
    """Fail-closed protocol or containment error."""


@dataclass(frozen=True)
class PrivateFileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    stat_identity: tuple[int, int, int, int]


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_system_python() -> Path:
    completed = subprocess.run(
        [
            str(SYSTEM_PYTHON),
            "-c",
            "import os,sys;print(os.path.realpath(sys.executable))",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": "/private/tmp",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": "/private/tmp",
        },
    )
    if completed.returncode != 0:
        raise L2BError("system Python runtime could not be resolved")
    runtime = Path(completed.stdout.strip()).resolve()
    allowed_roots = (
        Path("/Library/Developer/CommandLineTools").resolve(),
        Path("/System").resolve(),
    )
    if not runtime.is_file() or not any(_under(runtime, root) for root in allowed_roots):
        raise L2BError("resolved system Python is outside an allowed Apple toolchain")
    metadata = runtime.stat()
    if metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise L2BError("resolved system Python is not root-owned and write-protected")
    return runtime


def _lexical_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.abspath(path))


def _under(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    root = parent.resolve()
    return resolved == root or root in resolved.parents


def _contains_symlink(path: Path, parent: Path) -> bool:
    try:
        relative = path.relative_to(parent)
    except ValueError:
        return True
    current = parent
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _snapshot_private_file(
    repo_root: Path,
    value: str | Path,
    *,
    expected_sha256: str | None = None,
) -> PrivateFileSnapshot:
    private_root = (repo_root / "private_library").resolve()
    path = _lexical_path(repo_root, value)
    if not (path == private_root or private_root in path.parents):
        raise L2BError("control file is not lexically under private_library")
    if not _under(path, private_root) or _contains_symlink(path, private_root):
        raise L2BError("control file crosses a private_library path boundary")
    relative = path.relative_to(repo_root).as_posix()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=repo_root,
        check=False,
        capture_output=True,
    ).returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repo_root,
        check=False,
        capture_output=True,
    ).returncode == 0
    if not ignored or tracked:
        raise L2BError("private control file is not Git-ignored and untracked")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise L2BError("control file descriptor is not a single-link regular file")
        if before.st_size > MAX_CONTROL_FILE_BYTES:
            raise L2BError("control file exceeds the bounded size")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise L2BError("control file ended before its attested size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise L2BError("control file grew while being snapshotted")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if identity != after_identity:
        raise L2BError("control file metadata changed during snapshot")
    path_after = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(path_after.st_mode) or (
        int(path_after.st_dev), int(path_after.st_ino)
    ) != identity[:2]:
        raise L2BError("control file path changed during snapshot")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise L2BError("control file hash does not match authorization")
    return PrivateFileSnapshot(path, payload, digest, identity)


def _private_regular_file(
    repo_root: Path,
    value: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    return _snapshot_private_file(
        repo_root, value, expected_sha256=expected_sha256
    ).path


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise L2BError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise L2BError(f"non-finite JSON constant is forbidden: {value}")


def _bounded_json_value(value: object, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise L2BError("JSON nesting exceeds the describe-only bound")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if len(value) > MAX_JSON_STRING:
            raise L2BError("JSON string exceeds the describe-only bound")
        return
    if isinstance(value, int):
        if abs(value) > 2**63 - 1:
            raise L2BError("JSON integer exceeds signed 64-bit range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise L2BError("JSON number is non-finite")
        return
    if isinstance(value, list):
        if len(value) > MAX_JSON_ITEMS:
            raise L2BError("JSON array exceeds the describe-only item bound")
        for item in value:
            _bounded_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > MAX_JSON_ITEMS:
            raise L2BError("JSON object exceeds the describe-only item bound")
        for key, item in value.items():
            if not isinstance(key, str):
                raise L2BError("JSON object key is not text")
            _bounded_json_value(key, depth=depth + 1)
            _bounded_json_value(item, depth=depth + 1)
        return
    raise L2BError(f"unsupported JSON value: {type(value).__name__}")


def _load_json_bytes(payload: bytes) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise L2BError("JSON payload is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, L2BError):
            raise
        raise L2BError("JSON payload is invalid") from exc
    _bounded_json_value(value)
    return value


def _load_json_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) > MAX_CONTROL_FILE_BYTES:
        raise L2BError("JSON control file exceeds the bounded size")
    value = _load_json_bytes(payload)
    if not isinstance(value, dict):
        raise L2BError("JSON control file is not an object")
    return value


def _schema_errors(schema: dict[str, Any], value: object) -> list[str]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: list(item.path),
    )
    return [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in errors[:12]
    ]


def _safe_result_directory(repo_root: Path, value: str | Path) -> Path:
    private_root = (repo_root / "private_library").resolve()
    output = _lexical_path(repo_root, value)
    if not (output == private_root or private_root in output.parents):
        raise L2BError("L2-B output is not lexically under private_library")
    if not _under(output.parent, private_root):
        raise L2BError("L2-B output parent resolves outside private_library")
    if _contains_symlink(output.parent, private_root):
        raise L2BError("L2-B output parent contains a symlink")
    if not output.parent.is_dir():
        raise L2BError("L2-B output parent must exist before authorization is consumed")
    if output.exists():
        raise FileExistsError("refusing to overwrite an L2-B result directory")
    return output


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _exclusive_write_at(directory_fd: int, name: str, payload: bytes) -> None:
    if "/" in name or name in {"", ".", ".."}:
        raise L2BError("private result filename is not a single path component")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _opened_directory_matches_path(directory_fd: int, path: Path) -> bool:
    descriptor_metadata = os.fstat(directory_fd)
    try:
        path_metadata = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(descriptor_metadata.st_mode)
        and stat.S_ISDIR(path_metadata.st_mode)
        and (int(descriptor_metadata.st_dev), int(descriptor_metadata.st_ino))
        == (int(path_metadata.st_dev), int(path_metadata.st_ino))
    )


def _create_output_directory(output: Path) -> tuple[int, tuple[int, int]]:
    parent_flags = _directory_open_flags()
    parent_fd = os.open(output.parent, parent_flags)
    try:
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        output_fd = os.open(output.name, parent_flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    metadata = os.fstat(output_fd)
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    path_metadata = os.stat(output, follow_symlinks=False)
    if not stat.S_ISDIR(path_metadata.st_mode) or (
        int(path_metadata.st_dev), int(path_metadata.st_ino)
    ) != identity:
        os.close(output_fd)
        raise L2BError("private result directory changed during atomic creation")
    return output_fd, identity


def _verify_output_identity(
    output: Path, output_fd: int, expected_identity: tuple[int, int]
) -> None:
    descriptor_metadata = os.fstat(output_fd)
    path_metadata = os.stat(output, follow_symlinks=False)
    descriptor_identity = (
        int(descriptor_metadata.st_dev),
        int(descriptor_metadata.st_ino),
    )
    path_identity = (int(path_metadata.st_dev), int(path_metadata.st_ino))
    if (
        descriptor_identity != expected_identity
        or path_identity != expected_identity
        or not stat.S_ISDIR(path_metadata.st_mode)
    ):
        raise L2BError("private result directory identity changed before commit")


def _claim_boundary() -> dict[str, bool]:
    return {key: False for key in CLAIM_KEYS}


def _consume_authorization(
    repo_root: Path,
    *,
    auth_path: Path,
    auth_sha256: str,
    one_time_nonce: str,
) -> Path:
    private_root = (repo_root / "private_library").resolve()
    control_root = private_root / "n5_d5_control"
    if not control_root.is_dir() or _contains_symlink(control_root, private_root):
        raise L2BError("private control root is unavailable or crosses a symlink")
    ledger = control_root / ".l2b_nonce_ledger"
    marker = ledger / f"{one_time_nonce}.consumed"
    marker_payload = {
        "schema_version": "n5-d5-l2b-authorization-consumption-1.0",
        "authorization_sha256": auth_sha256,
        "authorization_path_sha256": sha256_bytes(
            auth_path.relative_to(repo_root).as_posix().encode("utf-8")
        ),
        "one_time_nonce_sha256": sha256_bytes(one_time_nonce.encode("ascii")),
        "consumed_at_unix_ns": time.time_ns(),
        "reusable": False,
        "claim_authorizations": _claim_boundary(),
    }
    directory_flags = _directory_open_flags()
    private_fd = os.open(private_root, directory_flags)
    control_fd = -1
    ledger_fd = -1
    try:
        if not _opened_directory_matches_path(private_fd, private_root):
            raise L2BError("private root changed while opening nonce ledger")
        control_fd = os.open("n5_d5_control", directory_flags, dir_fd=private_fd)
        if not _opened_directory_matches_path(control_fd, control_root):
            raise L2BError("private control root changed while opening nonce ledger")
        try:
            os.mkdir(".l2b_nonce_ledger", mode=0o700, dir_fd=control_fd)
        except FileExistsError:
            pass
        ledger_fd = os.open(
            ".l2b_nonce_ledger", directory_flags, dir_fd=control_fd
        )
        ledger_metadata = os.fstat(ledger_fd)
        if (
            not stat.S_ISDIR(ledger_metadata.st_mode)
            or stat.S_IMODE(ledger_metadata.st_mode) & 0o077
            or int(ledger_metadata.st_uid) != os.getuid()
            or not _opened_directory_matches_path(ledger_fd, ledger)
        ):
            raise L2BError("one-time nonce ledger is not a private owned 0700 directory")
        _exclusive_write_at(
            ledger_fd,
            marker.name,
            (json.dumps(marker_payload, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        os.fsync(ledger_fd)
    finally:
        if ledger_fd >= 0:
            os.close(ledger_fd)
        if control_fd >= 0:
            os.close(control_fd)
        os.close(private_fd)
    return marker


def _copy_inventory_to_stage(
    repo_root: Path,
    plan: dict[str, Any],
    stage_root: Path,
) -> dict[str, Path]:
    private_root = (repo_root / "private_library").resolve()
    bundle = _lexical_path(repo_root, plan["private_bundle"]["root"])
    if not _under(bundle, private_root) or _contains_symlink(bundle, private_root):
        raise L2BError("private bundle boundary changed before staging")
    if not bundle.is_dir():
        raise L2BError("private bundle is not a directory")
    staged: dict[str, Path] = {}
    for entry in plan["private_bundle"]["inventory"]:
        source = _lexical_path(repo_root, entry["relative_path"])
        if not _under(source, bundle) or _contains_symlink(source, bundle):
            raise L2BError("inventory source crosses the private bundle boundary")
        if not source.is_file() or source.is_symlink():
            raise L2BError("inventory source is not a regular file")
        metadata = source.stat()
        if metadata.st_nlink != 1:
            raise L2BError("inventory source is hardlinked")
        if metadata.st_size != entry["bytes"]:
            raise L2BError("inventory source size changed before staging")
        relative = source.relative_to(bundle)
        target = stage_root / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        source_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            source_flags |= os.O_NOFOLLOW
        source_fd = os.open(source, source_flags)
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            target_flags |= os.O_NOFOLLOW
        target_fd = os.open(target, target_flags, 0o400)
        digest = hashlib.sha256()
        copied = 0
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                copied += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    view = view[written:]
            os.fsync(target_fd)
        finally:
            os.close(source_fd)
            os.close(target_fd)
        if copied != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise L2BError("inventory bytes changed during staging")
        staged[entry["relative_path"]] = target
    return staged


def sandbox_profile(stage_root: Path) -> str:
    stage = str(stage_root.resolve())
    if any(character in stage for character in ('"', "\n", "\r", "\x00")):
        raise L2BError("ephemeral stage path cannot be represented in sandbox profile")
    return "\n".join(
        (
            "(version 1)",
            "(deny default)",
            "(allow file-read*)",
            '(deny file-read-data (subpath "/Users") (subpath "/Volumes") '
            '(subpath "/Network"))',
            "(deny file-read-data "
            f'(require-all (subpath "/private/tmp") (require-not (subpath "{stage}"))))',
            '(deny file-read-data (subpath "/private/var/folders"))',
            "(deny process-fork)",
            "(allow process-exec)",
            "",
        )
    )


def production_backend_capabilities() -> dict[str, Any]:
    return {
        "backend": "macos_sandbox_exec_deprecated",
        "production_ready": False,
        "development_test_double": False,
        "process_exec_replacement_denied": False,
        "private_input_root_external_mutation_denied": False,
        "durable_nonce_ledger_root_protected": False,
        "output_root_external_mutation_denied": False,
        "backend_capability_attestation_externally_verified": False,
        "blocker_codes": [
            "POST_LAUNCH_EXEC_REPLACEMENT_NOT_DENIED",
            "PRIVATE_INPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
            "DURABLE_NONCE_LEDGER_ROOT_NOT_PROTECTED",
            "OUTPUT_ROOT_EXTERNAL_MUTATION_NOT_DENIED",
            "BACKEND_CAPABILITY_ATTESTATION_NOT_EXTERNALLY_VERIFIED",
        ],
        "reason": (
            "the macOS sandbox profile must allow process-exec to launch Python "
            "and the private input, ledger, and output roots are not protected "
            "from another same-UID process; capability claims also have no "
            "external attestation"
        ),
    }


def scrubbed_environment(stage_root: Path) -> dict[str, str]:
    stage = str(stage_root.resolve())
    environment = {
        "HOME": stage,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": stage,
    }
    if tuple(sorted(environment)) != tuple(sorted(ALLOWED_ENVIRONMENT)):
        raise AssertionError("L2-B environment allowlist drifted")
    return environment


def _resource_limiter(limits: dict[str, Any]):
    def apply() -> None:
        cpu = int(limits["cpu_seconds"])
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (int(limits["open_files"]), int(limits["open_files"])),
        )
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

    return apply


@dataclass
class BoundedStream:
    pipe: BinaryIO
    limit: int
    chunks: bytearray
    overflow: threading.Event
    error: str | None = None

    def read(self) -> None:
        try:
            while True:
                chunk = self.pipe.read(4096)
                if not chunk:
                    break
                remaining = self.limit - len(self.chunks)
                if remaining > 0:
                    self.chunks.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.overflow.set()
                    break
        except Exception as exc:  # pragma: no cover - OS pipe failure
            self.error = type(exc).__name__
            self.overflow.set()
        finally:
            try:
                self.pipe.close()
            except OSError:
                pass


def _kill_process_group(
    process: subprocess.Popen[bytes], process_group: int
) -> bool:
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        return False
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.01)
    return False


def _process_group_absent(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    payload: bytes,
    limits: dict[str, Any],
) -> dict[str, Any]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        close_fds=True,
        start_new_session=True,
        preexec_fn=_resource_limiter(limits),
    )
    process_group = process.pid
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = BoundedStream(
        process.stdout,
        int(limits["stdout_bytes"]),
        bytearray(),
        threading.Event(),
    )
    stderr = BoundedStream(
        process.stderr,
        int(limits["stderr_bytes"]),
        bytearray(),
        threading.Event(),
    )
    stdout_thread = threading.Thread(target=stdout.read, daemon=True)
    stderr_thread = threading.Thread(target=stderr.read, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdin_error: str | None = None
    try:
        process.stdin.write(payload)
        process.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        stdin_error = type(exc).__name__
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass

    deadline = started + float(limits["wall_timeout_seconds"])
    timed_out = False
    killed_for_overflow = False
    cleanup_verified = True
    while process.poll() is None:
        if stdout.overflow.is_set() or stderr.overflow.is_set():
            killed_for_overflow = True
            cleanup_verified = _kill_process_group(process, process_group)
            break
        if time.perf_counter() >= deadline:
            timed_out = True
            cleanup_verified = _kill_process_group(process, process_group)
            break
        time.sleep(0.005)
    try:
        returncode = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        cleanup_verified = _kill_process_group(process, process_group)
        returncode = process.poll()
    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        cleanup_verified = False
    if not _process_group_absent(process_group):
        cleanup_verified = _kill_process_group(process, process_group)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_seconds = float(time.perf_counter() - started)
    return {
        "pid": process.pid,
        "returncode": returncode,
        "exit_signal": (
            -int(returncode)
            if isinstance(returncode, int) and returncode < 0
            else None
        ),
        "wall_seconds": wall_seconds,
        "child_user_cpu_seconds_delta": max(0.0, after.ru_utime - before.ru_utime),
        "child_system_cpu_seconds_delta": max(
            0.0, after.ru_stime - before.ru_stime
        ),
        "cumulative_children_max_rss_kb": int(after.ru_maxrss),
        "rss_measurement_scope": "RUSAGE_CHILDREN_CUMULATIVE_NOT_PROCESS_LOCAL",
        "stdout": bytes(stdout.chunks),
        "stderr": bytes(stderr.chunks),
        "stdout_overflow": stdout.overflow.is_set(),
        "stderr_overflow": stderr.overflow.is_set(),
        "stdout_reader_error": stdout.error,
        "stderr_reader_error": stderr.error,
        "stdin_error": stdin_error,
        "timed_out": timed_out,
        "killed_for_overflow": killed_for_overflow,
        "process_group_cleanup_verified": cleanup_verified,
        "hard_memory_limit_claimed": False,
    }


def _describe_requests(
    authorization_sha256: str, one_time_nonce_sha256: str | None = None
) -> list[dict[str, Any]]:
    nonce_digest = one_time_nonce_sha256 or sha256_bytes(
        b"fixture-no-production-nonce"
    )
    return [
        {
            "schema": REQUEST_SCHEMA,
            "request_id": f"describe-{index}",
            "audit_nonce": secrets.token_hex(32),
            "operation": "describe",
            "authorization_sha256": authorization_sha256,
            "one_time_nonce_sha256": nonce_digest,
        }
        for index in (1, 2)
    ]


def _parse_responses(
    stdout: bytes,
    requests: list[dict[str, Any]],
    expected_descriptor: dict[str, Any],
) -> list[dict[str, Any]]:
    if not stdout.endswith(b"\n"):
        raise L2BError("describe stdout is not newline-terminated JSONL")
    lines = stdout.splitlines()
    if len(lines) != 2 or any(not line for line in lines):
        raise L2BError("adapter must return exactly two nonempty JSONL lines")
    responses: list[dict[str, Any]] = []
    expected_descriptor_json = canonical_json(expected_descriptor)
    expected_keys = {
        "schema",
        "request_id",
        "audit_nonce",
        "operation",
        "status",
        "descriptor",
        "runtime_ledger",
    }
    for index, (line, request) in enumerate(zip(lines, requests), start=1):
        value = _load_json_bytes(line)
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise L2BError("describe response key set drifted")
        expected_pairs = {
            "schema": RESPONSE_SCHEMA,
            "request_id": request["request_id"],
            "audit_nonce": request["audit_nonce"],
            "operation": "describe",
            "status": "ok",
        }
        for key, expected in expected_pairs.items():
            if value.get(key) != expected:
                raise L2BError(f"describe response {key} mismatch")
        if canonical_json(value["descriptor"]) != expected_descriptor_json:
            raise L2BError("describe descriptor differs from preauthorized bytes")
        if value["runtime_ledger"] != {
            "describe_calls": index,
            "forward_api_calls": 0,
            "jvp_api_calls": 0,
            "vjp_api_calls": 0,
        }:
            raise L2BError("describe response runtime ledger is not fail-closed")
        responses.append(value)
    if canonical_json(responses[0]["descriptor"]) != canonical_json(
        responses[1]["descriptor"]
    ):
        raise L2BError("the two describe responses are inconsistent")
    return responses


def _manifest_payload(payloads: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "files": {
            name: {
                "sha256": sha256_bytes(payloads[name]),
                "bytes": len(payloads[name]),
            }
            for name in RESULT_FILES
        },
        "public_summary_present": False,
        "raw_trace_publication_authorized": False,
        "claim_authorizations": _claim_boundary(),
    }


def _write_result(
    *,
    output_fd: int,
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    attestation: dict[str, Any],
) -> None:
    request_payload = "".join(canonical_json(item) + "\n" for item in requests)
    response_payload = "".join(canonical_json(item) + "\n" for item in responses)
    payloads = {
        "requests.jsonl": request_payload.encode("utf-8"),
        "responses.jsonl": response_payload.encode("utf-8"),
        "process_attestation.json": (
            json.dumps(attestation, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    for name in RESULT_FILES:
        _exclusive_write_at(output_fd, name, payloads[name])
    _exclusive_write_at(
        output_fd,
        "manifest.json",
        (
            json.dumps(_manifest_payload(payloads), indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int = MAX_CONTROL_FILE_BYTES,
) -> bytes:
    if "/" in name or name in {"", ".", ".."}:
        raise L2BError("private result filename is not a single path component")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
        ):
            raise L2BError("L2-B result payload is not a bounded single-link file")
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise L2BError("L2-B result payload ended before its attested size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise L2BError("L2-B result payload grew during validation")
        after = os.fstat(descriptor)
        identity_before = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        identity_after = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if identity_before != identity_after:
            raise L2BError("L2-B result payload changed during validation")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_result_fd(output_fd: int) -> dict[str, Any]:
    try:
        names = os.listdir(output_fd)
    except OSError as exc:
        raise L2BError("L2-B result directory cannot be listed by descriptor") from exc
    expected = set(RESULT_FILES) | {"manifest.json"}
    if set(names) != expected or len(names) != len(expected):
        raise L2BError("L2-B result inventory is not closed-world")
    payloads = {
        name: _read_regular_file_at(output_fd, name)
        for name in sorted(expected)
    }
    manifest_value = _load_json_bytes(payloads["manifest.json"])
    if not isinstance(manifest_value, dict):
        raise L2BError("L2-B result manifest is not an object")
    manifest = manifest_value
    if manifest.get("schema_version") != RESULT_SCHEMA:
        raise L2BError("L2-B result manifest schema drifted")
    if manifest.get("claim_authorizations") != _claim_boundary():
        raise L2BError("L2-B result manifest contains a scientific authorization")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(RESULT_FILES):
        raise L2BError("L2-B result manifest file set drifted")
    for name in RESULT_FILES:
        if files[name] != {
            "sha256": sha256_bytes(payloads[name]),
            "bytes": len(payloads[name]),
        }:
            raise L2BError("L2-B result hash or size mismatch")
    return {
        "schema_version": "n5-d5-l2b-result-validation-1.0",
        "valid": True,
        "file_count": len(expected),
        "claim_authorizations": _claim_boundary(),
    }


def validate_result_directory(
    output: Path,
    *,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    private_root = (repo_root / "private_library").resolve()
    output = _lexical_path(repo_root, output)
    if not _under(output, private_root) or _contains_symlink(output, private_root):
        raise L2BError("L2-B result directory crosses the private boundary")
    directory_fd = os.open(output, _directory_open_flags())
    try:
        if not _opened_directory_matches_path(directory_fd, output):
            raise L2BError("L2-B result directory changed while opening validation FD")
        return _validate_result_fd(directory_fd)
    finally:
        os.close(directory_fd)


def run_authorized_describe(
    authorization_path: Path,
    *,
    repo_root: Path = ROOT,
    enforce_public_committed: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    backend_capabilities = production_backend_capabilities()
    development_test_double = (
        backend_capabilities.get("development_test_double") is True
    )
    blocker_codes = [
        str(code) for code in backend_capabilities.get("blocker_codes", [])
    ]
    required_capabilities = (
        backend_capabilities.get("process_exec_replacement_denied") is True
        and backend_capabilities.get("private_input_root_external_mutation_denied")
        is True
        and backend_capabilities.get("durable_nonce_ledger_root_protected") is True
        and backend_capabilities.get("output_root_external_mutation_denied") is True
        and backend_capabilities.get(
            "backend_capability_attestation_externally_verified"
        )
        is True
    )
    if development_test_double:
        if repo_root == ROOT.resolve() or enforce_public_committed:
            raise L2BError(
                "development test-double backend is restricted to uncommitted temporary fixtures"
            )
    elif not backend_capabilities.get("production_ready") or not required_capabilities:
        if not blocker_codes:
            blocker_codes = ["PRODUCTION_BACKEND_CAPABILITY_SET_INCOMPLETE"]
        raise L2BError(
            "production L2-B is fail-closed: " + ",".join(blocker_codes)
        )
    authorization_snapshot = _snapshot_private_file(repo_root, authorization_path)
    authorization_path = authorization_snapshot.path
    authorization_sha256 = authorization_snapshot.sha256
    authorization = _load_json_bytes(authorization_snapshot.payload)
    if not isinstance(authorization, dict):
        raise L2BError("authorization snapshot is not a JSON object")
    schema = _load_json_file(repo_root / AUTH_SCHEMA_RELATIVE)
    errors = _schema_errors(schema, authorization)
    if errors:
        raise L2BError("authorization schema failed: " + "; ".join(errors))
    serialized_authorization = canonical_json(authorization)
    if any(marker in serialized_authorization for marker in PLACEHOLDER_MARKERS):
        raise L2BError("authorization still contains placeholder markers")
    if authorization["one_time_nonce"] == "0" * 64:
        raise L2BError("authorization nonce is still the all-zero placeholder")
    if any(
        reference["sha256"] == "0" * 64
        for reference in (
            authorization["foundation_report"],
            authorization["replay_plan"],
        )
    ):
        raise L2BError("authorization contains an all-zero file digest")
    if authorization["claim_authorizations"] != _claim_boundary():
        raise L2BError("authorization attempts to unlock a scientific claim")

    plan_ref = authorization["replay_plan"]
    plan_snapshot = _snapshot_private_file(
        repo_root,
        plan_ref["relative_path"],
        expected_sha256=plan_ref["sha256"],
    )
    plan_path = plan_snapshot.path
    plan = _load_json_bytes(plan_snapshot.payload)
    if not isinstance(plan, dict):
        raise L2BError("replay plan snapshot is not a JSON object")
    foundation_ref = authorization["foundation_report"]
    foundation_snapshot = _snapshot_private_file(
        repo_root,
        foundation_ref["relative_path"],
        expected_sha256=foundation_ref["sha256"],
    )
    stored_foundation = _load_json_bytes(foundation_snapshot.payload)
    if not isinstance(stored_foundation, dict):
        raise L2BError("foundation report snapshot is not a JSON object")
    computed_foundation = build_foundation_report(
        plan_path,
        repo_root=repo_root,
        enforce_public_committed=enforce_public_committed,
        plan_snapshot_payload=plan_snapshot.payload,
        plan_snapshot_stat_identity=plan_snapshot.stat_identity,
    )
    plan_after_foundation = _snapshot_private_file(
        repo_root,
        plan_ref["relative_path"],
        expected_sha256=plan_ref["sha256"],
    )
    if (
        plan_after_foundation.stat_identity != plan_snapshot.stat_identity
        or plan_after_foundation.payload != plan_snapshot.payload
    ):
        raise L2BError("replay plan changed while L2-A was being reproduced")
    if canonical_json(stored_foundation) != canonical_json(computed_foundation):
        raise L2BError("stored L2-A foundation report does not reproduce")
    if not (
        stored_foundation.get("foundation_ready") is True
        and stored_foundation.get("adapter_executed") is False
        and stored_foundation.get("private_describe_probe_authorized") is False
        and stored_foundation.get("foundation_blocker_codes") == []
        and stored_foundation.get("claim_authorizations") == _claim_boundary()
        and stored_foundation.get("formal_replay_authorized") is False
    ):
        raise L2BError("L2-A foundation is not eligible for separate describe intent")
    if stored_foundation.get("private_plan_sha256") != sha256_bytes(
        canonical_json(plan).encode("utf-8")
    ):
        raise L2BError("L2-A report is not bound to the authorized plan")

    entry_relative = authorization["describe_entrypoint"][
        "inventory_relative_path"
    ]
    inventory_matches = [
        entry
        for entry in plan["private_bundle"]["inventory"]
        if entry["relative_path"] == entry_relative
        and entry["role"] == "adapter_source"
    ]
    if len(inventory_matches) != 1:
        raise L2BError("describe entrypoint is not one unique adapter_source")
    expected_descriptor = authorization["expected_descriptor"]
    if expected_descriptor.get("implementation_sha256") != inventory_matches[0][
        "sha256"
    ]:
        raise L2BError("expected descriptor is not bound to entrypoint bytes")
    _bounded_json_value(expected_descriptor)

    output = _safe_result_directory(repo_root, authorization["output_directory"])
    one_time_nonce_sha256 = sha256_bytes(
        authorization["one_time_nonce"].encode("ascii")
    )
    _consume_authorization(
        repo_root,
        auth_path=authorization_path,
        auth_sha256=authorization_sha256,
        one_time_nonce=authorization["one_time_nonce"],
    )
    output_fd, output_identity = _create_output_directory(output)
    requests = _describe_requests(authorization_sha256, one_time_nonce_sha256)
    valid_responses: list[dict[str, Any]] = []
    state_transitions = ["NEW", "AUTH_CONSUMED"]
    process_result: dict[str, Any] | None = None
    profile_sha256: str | None = None
    python_runtime: Path | None = None
    failure_reason: str | None = None
    try:
        if not SANDBOX_EXEC.is_file() or not SYSTEM_PYTHON.is_file():
            raise L2BError("required macOS sandbox or system Python is unavailable")
        python_runtime = resolve_system_python()
        with tempfile.TemporaryDirectory(
            prefix="n5d5-l2b-", dir="/private/tmp"
        ) as temporary:
            stage_root = Path(temporary).resolve()
            os.chmod(stage_root, 0o700)
            staged = _copy_inventory_to_stage(repo_root, plan, stage_root)
            state_transitions.append("STAGED")
            entrypoint = staged.get(entry_relative)
            if entrypoint is None:
                raise L2BError("staged describe entrypoint is missing")
            profile_text = sandbox_profile(stage_root)
            profile_sha256 = sha256_bytes(profile_text.encode("utf-8"))
            command = [
                str(SANDBOX_EXEC),
                "-p",
                profile_text,
                str(python_runtime),
                "-E",
                "-S",
                "-B",
                str(entrypoint),
            ]
            payload = "".join(
                canonical_json(request) + "\n" for request in requests
            ).encode("utf-8")
            process_result = _run_bounded_process(
                command,
                cwd=stage_root,
                environment=scrubbed_environment(stage_root),
                payload=payload,
                limits=authorization["limits"],
            )
            state_transitions.append("SPAWNED")
            if process_result["timed_out"]:
                raise L2BError("describe adapter exceeded the wall-time bound")
            if process_result["stdout_overflow"]:
                raise L2BError("describe adapter exceeded the stdout byte bound")
            if process_result["stderr_overflow"]:
                raise L2BError("describe adapter exceeded the stderr byte bound")
            if process_result["stdin_error"] is not None:
                raise L2BError("describe adapter closed stdin before two requests")
            if process_result["returncode"] != 0:
                raise L2BError("describe adapter exited unsuccessfully")
            if process_result["stderr"]:
                raise L2BError("describe adapter wrote unexpected stderr")
            if not process_result["process_group_cleanup_verified"]:
                raise L2BError("describe process-group cleanup is unproven")
            valid_responses = _parse_responses(
                process_result["stdout"], requests, expected_descriptor
            )
            state_transitions.extend(["DESCRIBE_1", "DESCRIBE_2", "EXITED"])
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}:{exc}"

    passed = failure_reason is None and len(valid_responses) == 2
    status = (
        (
            "L2B_DEVELOPMENT_TEST_DOUBLE_PASS_NO_SCIENCE_AUTHORIZATION"
            if development_test_double
            else "L2B_DESCRIBE_MECHANISM_PASS_NO_SCIENCE_AUTHORIZATION"
        )
        if passed
        else "L2B_DESCRIBE_MECHANISM_FAIL_CLOSED"
    )
    if passed:
        state_transitions.append("COMMITTED")
    else:
        state_transitions.append("FAILED_TERMINAL")
    process_public = None
    if process_result is not None:
        process_public = {
            key: value
            for key, value in process_result.items()
            if key not in {"stdout", "stderr"}
        }
        process_public.update(
            {
                "stdout_bytes": len(process_result["stdout"]),
                "stderr_bytes": len(process_result["stderr"]),
                "stdout_sha256": sha256_bytes(process_result["stdout"]),
                "stderr_sha256": sha256_bytes(process_result["stderr"]),
            }
        )
    attestation = {
        "schema_version": "n5-d5-l2b-process-attestation-1.0",
        "status": status,
        "evidence_scope": (
            "DEVELOPMENT_TEST_DOUBLE_EXTERNAL_DESCRIBE_TRANSPORT_"
            "NO_PRODUCTION_CONTAINMENT_CLAIM_NO_INTERNAL_OPERATION_PROOF_"
            "NO_SCIENCE_AUTHORIZATION"
            if development_test_double
            else "PRODUCTION_EXTERNAL_DESCRIBE_TRANSPORT_MECHANISM_"
            "NO_INTERNAL_OPERATION_PROOF_NO_SCIENCE_AUTHORIZATION"
        ),
        "authorization_sha256": authorization_sha256,
        "one_time_nonce_sha256": one_time_nonce_sha256,
        "plan_sha256": plan_ref["sha256"],
        "foundation_report_sha256": foundation_ref["sha256"],
        "expected_descriptor_sha256": sha256_bytes(
            canonical_json(expected_descriptor).encode("utf-8")
        ),
        "sandbox_backend": backend_capabilities["backend"],
        "production_backend_ready": (
            backend_capabilities["production_ready"] is True
            and required_capabilities
            and not development_test_double
        ),
        "production_backend_blocker_codes": blocker_codes,
        "development_test_double_used": development_test_double,
        "production_execution_authorized": (
            backend_capabilities["production_ready"] is True
            and required_capabilities
            and not development_test_double
        ),
        "backend_capability_trust_model": (
            "PYTEST_MONKEYPATCHED_DEVELOPMENT_TEST_DOUBLE"
            if development_test_double
            else "EXTERNALLY_ATTESTED_PRODUCTION_BACKEND"
        ),
        "backend_capability_attestation_externally_verified": backend_capabilities[
            "backend_capability_attestation_externally_verified"
        ],
        "sandbox_profile_sha256": profile_sha256,
        "staged_entrypoint_sha256": inventory_matches[0]["sha256"],
        "resolved_system_python_sha256": (
            sha256_file(python_runtime) if python_runtime is not None else None
        ),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "mac_version": platform.mac_ver()[0],
        },
        "environment_variable_names": list(ALLOWED_ENVIRONMENT),
        "staged_input_inventory_closed_world": True,
        "sandbox_read_surface_closed_world": False,
        "process_exec_replacement_denied": backend_capabilities[
            "process_exec_replacement_denied"
        ],
        "private_input_root_external_mutation_denied": backend_capabilities[
            "private_input_root_external_mutation_denied"
        ],
        "durable_nonce_ledger_root_protected": backend_capabilities[
            "durable_nonce_ledger_root_protected"
        ],
        "global_nonce_uniqueness_proven": False,
        "nonce_uniqueness_scope": "CURRENT_OPEN_LEDGER_INODE_ONLY",
        "output_root_external_mutation_denied": backend_capabilities[
            "output_root_external_mutation_denied"
        ],
        "detached_descendant_absence_globally_proven": False,
        "declared_interpreter": "/usr/bin/python3",
        "actual_argv_sanitized": [
            "/usr/bin/sandbox-exec",
            "-p",
            "<profile-bound-by-sha256>",
            "<resolved-root-owned-apple-runtime>",
            "-E",
            "-S",
            "-B",
            "<closed-world-staged-entrypoint>",
        ],
        "external_requests_sent": 2,
        "external_operations_sent": ["describe", "describe"],
        "validated_responses": len(valid_responses),
        "primary_auto_chain_attempted": False,
        "forward_jvp_vjp_external_requests": 0,
        "internal_operation_absence_proven": False,
        "state_transitions": state_transitions,
        "process": process_public,
        "failure_reason": failure_reason,
        "private_raw_failure_stdout_persisted": False,
        "private_raw_failure_stderr_persisted": False,
        "claim_authorizations": _claim_boundary(),
        "limitations": [
            "sandbox-exec is deprecated and this observation is local to the recorded macOS build.",
            "The profile has a broad system read allowance and excludes selected user, mounted-volume, and sibling-temp prefixes; its total read surface is not closed-world.",
            "process-exec is allowed to launch the interpreter and can replace the same sandboxed process; therefore the default backend is production-blocked.",
            "Private input path checks and final-component O_NOFOLLOW do not close intermediate-directory replacement races against another same-UID process.",
            "O_EXCL is atomic only within the held ledger inode; global nonce uniqueness across same-UID ledger-path replacement is not proven.",
            "The held output FD prevents path reopen races inside this run, but post-validation immutability against another same-UID process is not proven.",
            "Production capability claims require an external attestation; the development test double uses an in-process monkeypatch and is not a security boundary.",
            "RLIMIT_NPROC plus the profile blocked tested fork/subprocess probes, but global absence of a detached descendant is not proven.",
            "Exactly two external describe requests do not prove the adapter performed no hidden internal forward, JVP, or VJP work.",
            "RLIMIT_RSS is not treated as a hard memory guarantee on macOS.",
            "No real BOST physics, derivative, reconstruction, training, superiority, generalization, or publication claim is authorized.",
        ],
    }
    try:
        _write_result(
            output_fd=output_fd,
            requests=requests,
            responses=valid_responses,
            attestation=attestation,
        )
        os.fsync(output_fd)
        _validate_result_fd(output_fd)
        _verify_output_identity(output, output_fd, output_identity)
    finally:
        os.close(output_fd)
    return attestation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_authorized_describe(args.authorization, repo_root=ROOT)
    except L2BError as exc:
        print(
            json.dumps(
                {
                    "status": "L2B_PRODUCTION_BACKEND_BLOCKED_FAIL_CLOSED",
                    "blocker": str(exc),
                    "authorization_consumed": False,
                    "adapter_executed": False,
                    "formal_replay_authorized": False,
                    "model_training_authorized": False,
                    "claim_authorizations": _claim_boundary(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "external_requests_sent": result["external_requests_sent"],
                "validated_responses": result["validated_responses"],
                "formal_replay_authorized": False,
                "model_training_authorized": False,
                "claim_authorizations": _claim_boundary(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"].startswith("L2B_DESCRIBE_MECHANISM_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
