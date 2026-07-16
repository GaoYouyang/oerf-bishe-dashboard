from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_pages_artifact_keeps_public_site_and_evidence() -> None:
    assert not MODULE.should_exclude("index.html")
    assert not MODULE.should_exclude("general_operator_research_lab.html")
    assert not MODULE.should_exclude(
        "demo_t16_operator/results/example/figure.png"
    )
