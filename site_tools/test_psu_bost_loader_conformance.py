from site_tools.psu_bost_loader_conformance import build_conformance_report


def _base_report(variables):
    return {
        "status": "SELECTED_NUMERIC_STREAMS_CONFORMANT",
        "source_file": "fixture.mat",
        "source_file_size_bytes": 1234,
        "subsystem_offset": 1200,
        "variables": variables,
    }


def _variable(name, shape, samples, mode="deterministic_sample"):
    return {
        "name": name,
        "shape": list(shape),
        "sample_mode": mode,
        "integrity_status": "FULL_SELECTED_STREAM_VALIDATED",
        "samples": samples,
    }


def _sample(subscripts, value, flat=0):
    return {
        "subscripts_zero_based": list(subscripts),
        "matlab_flat_index": flat,
        "value": value,
    }


def _vector_samples(vectors):
    samples = []
    for measurement, vector in vectors.items():
        for component, value in enumerate(vector):
            samples.append(_sample((component, measurement), value))
    return samples


def test_conformant_fixture_passes_contract() -> None:
    scalar = _base_report(
        [
            _variable(
                "siz",
                (1, 3),
                [_sample((0, i), value, i) for i, value in enumerate((1, 2, 2))],
                mode="full",
            )
        ]
    )
    xyz = []
    for axis, name in enumerate(("X", "Y", "Z")):
        samples = []
        for point in ((0, 0, 0), (1, 1, 1), (0, 1, 1), (1, 0, 0)):
            samples.append(_sample(point, -0.5 if point[axis] == 0 else 0.5))
        xyz.append(_variable(name, (2, 2, 2), samples))
    xyz_report = _base_report(xyz)

    measurements = {0: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)}
    c_vectors = {0: (0.0, 0.0, 1.0), 3: (0.0, 1.0, 0.0)}
    ray = _base_report(
        [
            _variable("c", (3, 4), _vector_samples(c_vectors)),
            _variable("Ruvecs", (3, 4), _vector_samples(measurements)),
            _variable("Rvvecs", (3, 4), _vector_samples(measurements)),
            _variable("Rapvec", (1, 4), _vector_samples({0: (0.1,), 3: (0.2,)})),
        ]
    )
    v = _base_report([_variable("v", (3, 4), _vector_samples(measurements))])
    result = build_conformance_report(
        scalar,
        xyz_report,
        ray,
        v,
        expected_views=2,
        expected_grid_shape=(2, 2, 2),
        expected_domain_m=(2.0, 2.0, 2.0),
    )
    assert result["status"] == "LOADER_NUMERIC_CONTRACT_CONFORMANT"
    assert all(result["checks"].values())
    assert result["configuration"]["measurement_count"] == 4


def test_non_unit_direction_requires_review() -> None:
    scalar = _base_report(
        [
            _variable(
                "siz",
                (1, 3),
                [_sample((0, i), value, i) for i, value in enumerate((1, 1, 1))],
                mode="full",
            )
        ]
    )
    xyz = _base_report(
        [
            _variable(
                name,
                (2, 2, 2),
                [_sample((0, 0, 0), -0.5), _sample((1, 1, 1), 0.5)],
            )
            for name in ("X", "Y", "Z")
        ]
    )
    vectors = {0: (1.0, 0.0, 0.0)}
    ray = _base_report(
        [
            _variable("c", (3, 1), _vector_samples({0: (0.0, 0.0, 1.0)})),
            _variable("Ruvecs", (3, 1), _vector_samples(vectors)),
            _variable("Rvvecs", (3, 1), _vector_samples(vectors)),
            _variable("Rapvec", (1, 1), _vector_samples({0: (0.1,)})),
        ]
    )
    v = _base_report([_variable("v", (3, 1), _vector_samples({0: (2.0, 0.0, 0.0)}))])
    result = build_conformance_report(
        scalar,
        xyz,
        ray,
        v,
        expected_views=1,
        expected_grid_shape=(2, 2, 2),
        expected_domain_m=(2.0, 2.0, 2.0),
    )
    assert result["status"] == "LOADER_NUMERIC_CONTRACT_REVIEW_REQUIRED"
    assert result["checks"]["direction_samples_are_unit_vectors"] is False
