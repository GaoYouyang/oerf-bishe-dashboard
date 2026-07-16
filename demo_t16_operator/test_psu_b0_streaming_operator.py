from __future__ import annotations

import numpy as np
import torch

from .psu_b0_reconstruction_interface import (
    PSUB0VoxelGradientOperator,
    build_trilinear_stencil,
)
from .psu_b0_streaming_operator import (
    PSUB0StreamingOperator,
    StreamingRayChunk,
    cgls_solve,
    zero_outer_boundary_support,
)


class _MemoryStore:
    def __init__(
        self,
        *,
        points: np.ndarray,
        projection_u: np.ndarray,
        projection_v: np.ndarray,
        line_length: np.ndarray,
        system_constant: np.ndarray,
        observation: np.ndarray,
        chunk_rays: int,
    ) -> None:
        self.points = points
        self.projection_u = projection_u
        self.projection_v = projection_v
        self.line_length = line_length
        self.system_constant = system_constant
        self.observation = observation
        self.ray_count = int(points.shape[0])
        self.sample_count = int(points.shape[1])
        self.chunk_rays = int(chunk_rays)

    def iter_chunks(self):
        for start in range(0, self.ray_count, self.chunk_rays):
            stop = min(start + self.chunk_rays, self.ray_count)
            yield StreamingRayChunk(
                start_index=start,
                stop_index=stop,
                sample_points_xyz=self.points[start:stop],
                projection_u_xyz=self.projection_u[start:stop],
                projection_v_xyz=self.projection_v[start:stop],
                line_length=self.line_length[start:stop],
                system_constant=self.system_constant[start:stop],
                observation_uv=self.observation[start:stop],
                view_id=start // self.chunk_rays,
                b0_hit_count=stop - start,
            )

    def load_observations(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.as_tensor(
            self.observation,
            dtype=dtype,
            device=device,
        )[None]


def _fixture(dtype: torch.dtype = torch.float64):
    rng = np.random.default_rng(9107)
    ray_count = 17
    sample_count = 5
    points = rng.uniform(-0.9, 0.9, size=(ray_count, sample_count, 3))
    projection_u = rng.normal(size=(ray_count, 3))
    projection_u /= np.linalg.norm(projection_u, axis=1, keepdims=True)
    projection_v = rng.normal(size=(ray_count, 3))
    projection_v -= (
        np.sum(projection_u * projection_v, axis=1, keepdims=True)
        * projection_u
    )
    projection_v /= np.linalg.norm(projection_v, axis=1, keepdims=True)
    line_length = rng.uniform(0.8, 1.7, size=ray_count)
    system_constant = rng.uniform(0.6, 1.2, size=ray_count)
    observation = rng.normal(size=(ray_count, 2))
    store = _MemoryStore(
        points=points,
        projection_u=projection_u,
        projection_v=projection_v,
        line_length=line_length,
        system_constant=system_constant,
        observation=observation,
        chunk_rays=4,
    )
    streaming = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=(5, 6, 7),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=dtype,
    )
    stencil = build_trilinear_stencil(
        points,
        grid_shape=(5, 6, 7),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=dtype,
    )
    monolithic = PSUB0VoxelGradientOperator(
        stencil=stencil,
        projection_u_xyz=projection_u,
        projection_v_xyz=projection_v,
        line_length=line_length,
        system_constant=system_constant,
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        dtype=dtype,
    )
    return store, streaming, monolithic


def test_chunked_forward_and_adjoint_match_monolithic_operator() -> None:
    _, streaming, monolithic = _fixture()
    generator = torch.Generator().manual_seed(9211)
    volume = torch.randn(
        (2, 1, *streaming.grid_shape),
        generator=generator,
        dtype=torch.float64,
    )
    residual = torch.randn(
        (2, streaming.ray_count, 2),
        generator=generator,
        dtype=torch.float64,
    )
    assert torch.allclose(
        streaming(volume),
        monolithic(volume),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        streaming.adjoint(residual),
        monolithic.adjoint(residual),
        atol=1e-12,
        rtol=1e-12,
    )
    report = streaming.call_report()
    assert report["forward_calls"] == 1
    assert report["adjoint_calls"] == 1
    assert report["records"][0]["chunk_count"] == 5
    assert report["records"][1]["chunk_count"] == 5


def test_streaming_operator_preserves_full_dot_product_identity() -> None:
    _, streaming, _ = _fixture()
    assert streaming.adjoint_relative_error(seed=9323) < 1e-12
    assert streaming.call_report()["forward_calls"] == 0
    assert streaming.call_report()["adjoint_calls"] == 0


def test_fixed_budget_cgls_reduces_measurement_residual() -> None:
    store, _, _ = _fixture()
    support = zero_outer_boundary_support(
        (5, 6, 7),
        boundary_width=1,
        dtype=torch.float64,
    )
    operator = PSUB0StreamingOperator(
        ray_store=store,
        grid_shape=(5, 6, 7),
        grid_minimum_xyz=(-1.0, -1.0, -1.0),
        grid_maximum_xyz=(1.0, 1.0, 1.0),
        support=support,
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(9431)
    truth = torch.randn(
        (1, 1, *operator.grid_shape),
        generator=generator,
        dtype=torch.float64,
    ) * support
    observation = operator.forward(truth)
    operator.reset_call_counts()
    result = cgls_solve(operator, observation, iterations=12)
    residuals = [
        float(row["relative_measurement_l2"])
        for row in result.history
        if not bool(row["breakdown"])
    ]
    assert not result.breakdown
    assert residuals[-1] < 0.05
    assert all(right <= left + 1e-12 for left, right in zip(residuals, residuals[1:]))
    assert operator.call_report()["forward_calls"] == 12
    assert operator.call_report()["adjoint_calls"] == 13
