from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_pages_artifact.py")
SPEC = importlib.util.spec_from_file_location("build_pages_artifact", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pages_artifact_excludes_pdf_payload_and_private_files() -> None:
    assert MODULE.should_exclude("paper_library/pdfs/example.pdf")
    assert MODULE.should_exclude("private_library/example.pdf")
    assert MODULE.should_exclude(".github/workflows/pages.yml")
    assert MODULE.should_exclude("build/pages-site/index.html")
    assert MODULE.should_exclude("demo/results/seed_1/best.pt")
    assert MODULE.should_exclude("demo/results/model.PTH")
    assert MODULE.should_exclude("demo/results/training.ckpt")
    for suffix in ("npz", "npy", "mat", "pem", "key"):
        assert MODULE.should_exclude(f"demo/results/private.{suffix}")


def test_pages_artifact_keeps_public_site_and_evidence() -> None:
    assert not MODULE.should_exclude("index.html")
    assert not MODULE.should_exclude("general_operator_research_lab.html")
    assert not MODULE.should_exclude(
        "demo_t16_operator/results/example/figure.png"
    )


def test_truth_and_weights_npz_is_not_copied_into_artifact(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = repo / "build/pages-site"
    source = repo / "demo/results/truth_weights.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"truth and weights")
    output.mkdir(parents=True)

    stats = MODULE._copy_tracked_files(
        repo,
        output,
        ["demo/results/truth_weights.npz"],
    )

    assert stats.copied_files == 0
    assert not (output / "demo/results/truth_weights.npz").exists()


def test_binary_allowlist_is_exact_path_and_documents_purpose(monkeypatch) -> None:
    allowance = MODULE.PublicBinaryAllowance(
        source_path="public_data/coordinates.npy",
        content_purpose="Browser-rendered public coordinate fixture",
    )
    monkeypatch.setattr(MODULE, "PUBLIC_BINARY_ALLOWLIST", (allowance,))

    assert not MODULE.should_exclude("public_data/coordinates.npy")
    assert MODULE.should_exclude("public_data/other.npy")


@pytest.mark.parametrize("suffix", ("pt", "pth", "ckpt", "pem", "key"))
def test_permanent_secret_suffixes_cannot_be_allowlisted(
    monkeypatch, suffix: str
) -> None:
    monkeypatch.setattr(
        MODULE,
        "PUBLIC_BINARY_ALLOWLIST",
        (
            MODULE.PublicBinaryAllowance(
                source_path=f"public_data/not_public.{suffix}",
                content_purpose="Attempted exception",
            ),
        ),
    )
    assert MODULE.should_exclude(f"public_data/not_public.{suffix}")


def test_tracked_symlink_is_rejected_before_copy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = repo / "build/pages-site"
    private = repo / "private_library/secret.npy"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private truth")
    public = repo / "public/data.txt"
    public.parent.mkdir(parents=True)
    public.symlink_to(private)
    output.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="symlinks are forbidden"):
        MODULE._copy_tracked_files(repo, output, ["public/data.txt"])
    assert not (output / "public/data.txt").exists()
