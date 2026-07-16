from __future__ import annotations

from pathlib import Path

import pytest

from site_tools.official_nirt_geometry_fixture import (
    load_numpy_geometry_functions,
    run_geometry_fixture,
)


PRIVATE_MEAS = Path(
    "private_library/external_datasets/psu_bost_flight_body/pyscripts/meas.py"
)


def test_extractor_ignores_tensorflow_and_top_level_side_effects(tmp_path: Path) -> None:
    source = tmp_path / "meas.py"
    source.write_text(
        """
import missing_tensorflow
raise RuntimeError('must not execute')

def rayBoxIntersection(pix_points, vecs, min_bounds, max_bounds):
    return np.asarray([1.0]), np.asarray([2.0]), np.asarray([1.0])

def rayConeIntersection(pix_points, vecs, vertex, axis, angle):
    return np.asarray([3.0]), np.asarray([4.0]), np.asarray([1.0])

def unrelated():
    raise RuntimeError('must not compile into the fixture namespace')
""",
        encoding="utf-8",
    )
    functions = load_numpy_geometry_functions(source)
    assert sorted(functions) == ["rayBoxIntersection", "rayConeIntersection"]
    assert functions["rayBoxIntersection"](None, None, None, None)[2][0] == 1.0


def test_extractor_rejects_missing_function(tmp_path: Path) -> None:
    source = tmp_path / "meas.py"
    source.write_text(
        "def rayBoxIntersection(a, b, c, d):\n    return a, b, c\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rayConeIntersection"):
        load_numpy_geometry_functions(source)


@pytest.mark.skipif(not PRIVATE_MEAS.is_file(), reason="private PSU source not present")
def test_real_author_geometry_contract_is_characterized() -> None:
    report = run_geometry_fixture(PRIVATE_MEAS)
    assert (
        report["status"]
        == "GEOMETRY_PRIMITIVE_CONTRACT_CHARACTERIZED_WITH_LIMITATIONS"
    )
    assert all(item["passed"] for item in report["checks"])
    assert report["source"]["author_source_modified"] is False
    assert report["decision"]["full_nirt_reconstruction"] == "NOT_UNLOCKED"
    assert "private_library" not in str(report)
