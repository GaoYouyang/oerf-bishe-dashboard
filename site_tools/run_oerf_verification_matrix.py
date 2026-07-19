#!/usr/bin/env python3
"""Run the bounded OERF verification matrix without scientific experiments."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

try:
    from site_tools.build_pages_artifact import is_blocked_publication_path
except ModuleNotFoundError:  # Direct execution places site_tools on sys.path.
    from build_pages_artifact import is_blocked_publication_path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_ROOTS = (
    "demo_t16_operator",
    "site_tools",
    "demo_m3b",
    "paper_library/tools",
)
MPS_CASES = (
    (
        "demo_t16_operator/test_psu_b0_primal_dual.py::"
        "test_cpu_float32_and_mps_float32_match_on_fixed_short_run"
    ),
    (
        "site_tools/test_psu_b0_gate_a_attestation_mps.py::"
        "test_mps_factor_recurrence_matches_cpu_reference"
    ),
    (
        "site_tools/test_psu_b0_gate_a_attestation_mps.py::"
        "test_independent_mps_validator_recomputes_parity"
    ),
)
SERIAL_HOST_CONTAINMENT_TESTS = (
    "site_tools/test_n5_d5_l2b_describe_runner.py",
)
FAST_CONTRACT_TESTS = (
    "site_tools/test_build_pages_artifact.py",
    "site_tools/test_build_psu_all_view_public_summary.py",
    "site_tools/test_build_psu_aperture_sensitivity_public_summary.py",
    "site_tools/test_build_psu_b0_resolution_summary.py",
    "site_tools/test_build_psu_b1_parameter_sensitivity_public_summary.py",
    "site_tools/test_build_psu_b3_policy_public_summary.py",
    "site_tools/test_build_psu_clipped_hybrid_public_summary.py",
    "site_tools/test_build_psu_fixed_domain_public_summary.py",
    "site_tools/test_build_psu_public_summary.py",
    "site_tools/test_n5_d5_l2c_external_witness.py",
    "site_tools/test_general_operator_n5_d5_l2_foundation_page.py",
)
GATE_A_TARGETED_TESTS = (
    "demo_t16_operator/test_psu_b0_active_coordinates.py",
    "demo_t16_operator/test_psu_b0_absolute_measurement_factor.py",
    "site_tools/test_psu_b0_factor_interfaces.py",
    "demo_t16_operator/test_psu_b0_absolute_regularization_factor.py",
    "demo_t16_operator/test_psu_b0_signed_factor_majorizer.py",
    "demo_t16_operator/test_psu_b0_factor_majorizer_pipeline.py",
    "site_tools/test_psu_b0_gate_a_attestation.py",
)
EXISTING_ARTIFACT = REPO_ROOT / "build/pages-site"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class CommandRunner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.environment = os.environ.copy()
        self.environment.update(THREAD_ENVIRONMENT)

    def run(
        self,
        command: Sequence[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        rendered = shlex.join(str(part) for part in command)
        print(f"$ {rendered}", flush=True)
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, stdout="dryrun\n", stderr="")

        started = time.perf_counter()
        try:
            return subprocess.run(
                [str(part) for part in command],
                cwd=REPO_ROOT,
                env=self.environment,
                check=True,
                text=True,
                capture_output=capture_output,
            )
        finally:
            print(f"  command elapsed: {time.perf_counter() - started:.2f}s", flush=True)


def _pytest_command(*targets: str) -> list[str]:
    return [sys.executable, "-m", "pytest", *targets]


def _run_layer(name: str, action: Callable[[], None]) -> None:
    print(f"== {name} ==", flush=True)
    started = time.perf_counter()
    try:
        action()
    finally:
        print(f"{name} elapsed: {time.perf_counter() - started:.2f}s", flush=True)


def _run_fast(runner: CommandRunner) -> None:
    runner.run(_pytest_command(*FAST_CONTRACT_TESTS, *GATE_A_TARGETED_TESTS))
    runner.run(["git", "diff", "--check"])
    if EXISTING_ARTIFACT.is_dir():
        runner.run(
            [
                sys.executable,
                "site_tools/audit_public_links.py",
                str(EXISTING_ARTIFACT.relative_to(REPO_ROOT)),
            ]
        )
    else:
        print(
            "SKIP existing artifact link audit: build/pages-site is missing; "
            "fast does not build it.",
            flush=True,
        )


def _run_medium(runner: CommandRunner) -> None:
    runner.run(
        [
            *_pytest_command(*SOURCE_ROOTS),
            *(f"--ignore={target}" for target in SERIAL_HOST_CONTAINMENT_TESTS),
            "-n",
            "4",
            "--dist=loadfile",
            *(f"--deselect={case}" for case in MPS_CASES),
        ]
    )
    runner.run(_pytest_command(*SERIAL_HOST_CONTAINMENT_TESTS))
    runner.run(_pytest_command(*MPS_CASES))


def _short_sha(runner: CommandRunner) -> str:
    if runner.dry_run:
        runner.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True)
        return "dryrun"
    result = runner.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
    )
    sha = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-fA-F]+", sha):
        raise RuntimeError(f"git returned an invalid short SHA: {sha!r}")
    return sha.lower()


def _assert_clean_worktree(runner: CommandRunner) -> None:
    command = ["git", "status", "--porcelain", "--untracked-files=all"]
    if runner.dry_run:
        runner.run(command, capture_output=True)
        return
    result = runner.run(command, capture_output=True)
    if result.stdout.strip():
        raise RuntimeError("full verification requires a clean git worktree")


def _artifact_check_command(kind: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--_artifact-check",
        kind,
        "--artifact",
        str(output.relative_to(REPO_ROOT)),
    ]


def _run_full(runner: CommandRunner) -> None:
    sha = _short_sha(runner)
    output = REPO_ROOT / "build" / f"pages-audit-{sha}-{uuid.uuid4().hex[:8]}"
    relative_output = output.relative_to(REPO_ROOT)
    runner.run(
        [
            sys.executable,
            "site_tools/build_pages_artifact.py",
            "--output",
            str(relative_output),
        ]
    )
    _assert_clean_worktree(runner)
    runner.run(
        [sys.executable, "site_tools/audit_public_links.py", str(relative_output)]
    )
    runner.run(_artifact_check_command("required-files", output))
    runner.run(_artifact_check_command("sensitive-suffixes", output))
    runner.run(_artifact_check_command("private-paths", output))
    _assert_clean_worktree(runner)
    print(f"artifact: {output}", flush=True)


def run_matrix(tier: str, *, dry_run: bool = False) -> None:
    if tier not in {"fast", "medium", "full"}:
        raise ValueError(f"unknown tier: {tier}")
    runner = CommandRunner(dry_run=dry_run)
    total_started = time.perf_counter()
    if tier == "full":
        _assert_clean_worktree(runner)
    _run_layer("fast", lambda: _run_fast(runner))
    if tier in {"medium", "full"}:
        _run_layer("medium", lambda: _run_medium(runner))
        if tier == "full":
            _assert_clean_worktree(runner)
    if tier == "full":
        _run_layer("full", lambda: _run_full(runner))
    print(f"{tier} matrix elapsed: {time.perf_counter() - total_started:.2f}s")


def check_artifact(kind: str, artifact: Path) -> None:
    artifact = (REPO_ROOT / artifact).resolve()
    if REPO_ROOT not in artifact.parents:
        raise ValueError("artifact must be inside the repository")
    if kind == "required-files":
        required = (artifact / "index.html", artifact / "pages-build-manifest.json")
        missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"artifact required files are missing: {missing}")
        print("artifact required files: OK")
        return
    if kind == "sensitive-suffixes":
        offenders = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in artifact.rglob("*")
            if path.is_file()
            and is_blocked_publication_path(path.relative_to(artifact).as_posix())
        )
        if offenders:
            raise RuntimeError(f"artifact contains sensitive suffixes: {offenders}")
        print("artifact sensitive suffixes: OK")
        return
    if kind == "private-paths":
        offenders = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in artifact.rglob("*")
            if "private_library" in path.relative_to(artifact).parts
        )
        if offenders:
            raise RuntimeError(f"artifact contains private_library paths: {offenders}")
        print("artifact private paths: OK")
        return
    raise ValueError(f"unknown artifact check: {kind}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", nargs="?", choices=("fast", "medium", "full"))
    parser.add_argument("--tier", dest="tier_option", choices=("fast", "medium", "full"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--_artifact-check",
        choices=("required-files", "sensitive-suffixes", "private-paths"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.tier and args.tier_option:
        parser.error("choose either positional tier or --tier, not both")
    args.tier = args.tier_option or args.tier
    if args._artifact_check:
        if args.artifact is None:
            parser.error("--artifact is required for an internal artifact check")
    elif args.tier is None:
        parser.error("tier is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args._artifact_check:
        check_artifact(args._artifact_check, args.artifact)
    else:
        run_matrix(args.tier, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
