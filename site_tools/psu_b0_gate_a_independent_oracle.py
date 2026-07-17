"""Independent NumPy oracle for the frozen PSU-B0 Gate A fixture.

This module intentionally does not import any production reconstruction,
factor, majorizer, or fixture implementation.  It reconstructs every tiny
matrix from JSON primitives so a coordinated production error cannot validate
itself through a shared helper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch


ORACLE_SCHEMA = "psu-b0-gate-a-independent-numpy-oracle-1.0"


def _centered_axis(values: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    moved = np.moveaxis(values, axis, -1)
    output = np.zeros_like(moved)
    output[..., 0] = (moved[..., 1] - moved[..., 0]) / spacing
    output[..., -1] = (moved[..., -1] - moved[..., -2]) / spacing
    if moved.shape[-1] > 2:
        output[..., 1:-1] = (
            moved[..., 2:] - moved[..., :-2]
        ) / (2.0 * spacing)
    return np.moveaxis(output, -1, axis)


def _forward_neumann_axis(
    values: np.ndarray,
    axis: int,
    spacing: float,
) -> np.ndarray:
    moved = np.moveaxis(values, axis, -1)
    output = np.zeros_like(moved)
    output[..., :-1] = (moved[..., 1:] - moved[..., :-1]) / spacing
    return np.moveaxis(output, -1, axis)


def _gradient_matrix(
    shape: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    *,
    centered: bool,
) -> np.ndarray:
    voxel_count = math.prod(shape)
    basis = np.eye(voxel_count, dtype=np.float64).reshape(voxel_count, *shape)
    difference = _centered_axis if centered else _forward_neumann_axis
    dx = difference(basis, -1, spacing_xyz[0])
    dy = difference(basis, -2, spacing_xyz[1])
    dz = difference(basis, -3, spacing_xyz[2])
    return np.stack((dx, dy, dz), axis=1).reshape(voxel_count, -1).T


def _trilinear_matrix(
    points_xyz: np.ndarray,
    *,
    shape: tuple[int, int, int],
    minimum_xyz: np.ndarray,
    maximum_xyz: np.ndarray,
) -> np.ndarray:
    ray_count, sample_count, coordinate_count = points_xyz.shape
    if coordinate_count != 3:
        raise ValueError("sample points must use xyz coordinates")
    nz, ny, nx = shape
    voxel_count = math.prod(shape)
    counts_xyz = np.asarray([nx, ny, nz], dtype=np.float64)
    scaled = (points_xyz - minimum_xyz[None, None]) / (
        maximum_xyz - minimum_xyz
    )[None, None]
    scaled = scaled * (counts_xyz - 1.0)
    scaled = np.minimum(np.maximum(scaled, 0.0), counts_xyz - 1.0)
    lower = np.floor(scaled).astype(np.int64)
    lower = np.minimum(
        lower,
        np.asarray([nx - 2, ny - 2, nz - 2], dtype=np.int64),
    )
    fraction = scaled - lower
    inside = np.all(
        (points_xyz >= minimum_xyz[None, None])
        & (points_xyz <= maximum_xyz[None, None]),
        axis=-1,
    )
    corner_offsets = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    matrix = np.zeros(
        (3 * ray_count * sample_count, 3 * voxel_count),
        dtype=np.float64,
    )
    for ray in range(ray_count):
        for sample in range(sample_count):
            if not inside[ray, sample]:
                continue
            base = lower[ray, sample]
            fractions = fraction[ray, sample]
            for offset in corner_offsets:
                xyz = base + offset
                weight = float(
                    np.prod(
                        np.where(offset > 0, fractions, 1.0 - fractions)
                    )
                )
                x, y, z = (int(value) for value in xyz)
                voxel = z * ny * nx + y * nx + x
                for component in range(3):
                    row = component * ray_count * sample_count + ray * sample_count + sample
                    column = component * voxel_count + voxel
                    matrix[row, column] = weight
    return matrix


def _measurement_matrix(
    fixture: dict[str, Any],
    *,
    ray_count: int,
    sample_count: int,
) -> np.ndarray:
    whitening = np.asarray(
        fixture["whitening_matrix_by_view"], dtype=np.float64
    )
    scale = np.asarray(fixture["scale_by_view"], dtype=np.float64)
    projection_u = np.asarray(fixture["projection_u_xyz"], dtype=np.float64)
    projection_v = np.asarray(fixture["projection_v_xyz"], dtype=np.float64)
    projection = np.stack((projection_u, projection_v), axis=1)
    view_count, detector_count, detector_count_2 = whitening.shape
    if detector_count != detector_count_2 or detector_count % 2:
        raise ValueError("whitening dimensions are not u/v detector pairs")
    rays_per_view = detector_count // 2
    if ray_count != view_count * rays_per_view:
        raise ValueError("ray count does not match view-local whitening")
    if scale.shape != (1, view_count):
        raise ValueError("independent oracle requires one frozen scale instance")
    line_length = np.asarray(fixture["line_length"], dtype=np.float64)
    system_constant = np.asarray(fixture["system_constant"], dtype=np.float64)
    ray_scale = line_length * system_constant / float(sample_count)
    measurement_scale = float(fixture["measurement_scale"])
    matrix = np.zeros(
        (view_count * detector_count, 3 * ray_count * sample_count),
        dtype=np.float64,
    )
    for view in range(view_count):
        for output in range(detector_count):
            output_row = view * detector_count + output
            for local_ray in range(rays_per_view):
                global_ray = view * rays_per_view + local_ray
                for component in range(3):
                    kernel = 0.0
                    for uv in range(2):
                        input_detector = 2 * local_ray + uv
                        kernel += (
                            whitening[view, output, input_detector]
                            * projection[global_ray, uv, component]
                        )
                    kernel *= ray_scale[global_ray] * measurement_scale
                    kernel /= scale[0, view]
                    for sample in range(sample_count):
                        column = component * ray_count * sample_count + global_ray * sample_count + sample
                        matrix[output_row, column] = kernel
    return matrix


@dataclass(frozen=True)
class IndependentGateAOracle:
    schema_version: str
    shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    E: np.ndarray
    G: np.ndarray
    P: np.ndarray
    W: np.ndarray
    D_plus: np.ndarray
    A: np.ndarray
    M: np.ndarray
    D: np.ndarray
    N: np.ndarray
    target: np.ndarray
    eta: float
    data_row_sums: np.ndarray
    data_column_sums: np.ndarray
    tv_row_sums: np.ndarray
    tv_column_sums: np.ndarray
    data_row_mask: np.ndarray
    tv_site_mask: np.ndarray
    active_primal_mask: np.ndarray
    active_primal_indices: np.ndarray
    rho_data_by_view: np.ndarray
    sigma_data_by_view: np.ndarray
    rho_tv_by_site: np.ndarray
    sigma_tv_by_site: np.ndarray
    tau: np.ndarray

    @property
    def data_sigma_rows(self) -> np.ndarray:
        view_count = self.rho_data_by_view.size
        rows_per_view = self.A.shape[0] // view_count
        values = np.repeat(self.sigma_data_by_view, rows_per_view)
        return np.where(self.data_row_mask, values, 0.0)

    @property
    def tv_sigma_rows(self) -> np.ndarray:
        shared = np.broadcast_to(
            self.sigma_tv_by_site.reshape(self.shape_zyx)[None],
            (3, *self.shape_zyx),
        ).reshape(-1)
        active = np.broadcast_to(
            self.tv_site_mask.reshape(self.shape_zyx)[None],
            (3, *self.shape_zyx),
        ).reshape(-1)
        return np.where(active, shared, 0.0)

    @property
    def scaled_norm_squared(self) -> float:
        tv_rows = np.broadcast_to(
            self.tv_site_mask.reshape(self.shape_zyx)[None],
            (3, *self.shape_zyx),
        ).reshape(-1)
        K = np.concatenate(
            (
                self.A[self.data_row_mask][:, self.active_primal_mask],
                self.D[tv_rows][:, self.active_primal_mask],
            ),
            axis=0,
        )
        sigma = np.concatenate(
            (
                self.data_sigma_rows[self.data_row_mask],
                self.tv_sigma_rows[tv_rows],
            )
        )
        scaled = np.sqrt(sigma)[:, None] * K * np.sqrt(self.tau)[None]
        singular = np.linalg.svd(scaled, compute_uv=False)
        return float(singular[0] ** 2)


def build_independent_oracle(config: dict[str, Any]) -> IndependentGateAOracle:
    fixture = config["fixture"]
    shape = tuple(int(value) for value in fixture["grid_shape_zyx"])
    minimum = np.asarray(fixture["grid_minimum_xyz"], dtype=np.float64)
    maximum = np.asarray(fixture["grid_maximum_xyz"], dtype=np.float64)
    nz, ny, nx = shape
    spacing = (
        float((maximum[0] - minimum[0]) / (nx - 1)),
        float((maximum[1] - minimum[1]) / (ny - 1)),
        float((maximum[2] - minimum[2]) / (nz - 1)),
    )
    voxel_count = math.prod(shape)
    support = np.ones(voxel_count, dtype=np.bool_)
    support[np.asarray(fixture["support_zero_flat_indices"], dtype=np.int64)] = False
    active_support = np.flatnonzero(support)
    E = np.zeros((voxel_count, active_support.size), dtype=np.float64)
    E[active_support, np.arange(active_support.size)] = 1.0
    G = _gradient_matrix(shape, spacing, centered=True)
    D_plus = _gradient_matrix(shape, spacing, centered=False)
    points = np.asarray(fixture["sample_points_xyz"], dtype=np.float64)
    ray_count, sample_count, _ = points.shape
    P = _trilinear_matrix(
        points,
        shape=shape,
        minimum_xyz=minimum,
        maximum_xyz=maximum,
    )
    W = _measurement_matrix(
        fixture,
        ray_count=ray_count,
        sample_count=sample_count,
    )
    A = W @ P @ G @ E
    M = np.abs(W) @ P @ np.abs(G) @ E
    D = D_plus @ E
    N = np.abs(D_plus) @ E
    target = np.asarray(fixture["target_view_ray_uv"], dtype=np.float64).reshape(-1)
    eta = float(fixture["eta"])
    data_rows = M.sum(axis=1)
    data_columns = M.sum(axis=0)
    tv_rows = N.sum(axis=1)
    tv_columns = N.sum(axis=0)
    data_mask = data_rows > 0.0
    tv_by_site = np.moveaxis(tv_rows.reshape(3, *shape), 0, -1)
    tv_site_mask = np.any(tv_by_site > 0.0, axis=-1).reshape(-1)
    total_columns = data_columns + tv_columns
    active_primal_mask = total_columns > 0.0
    active_primal = np.flatnonzero(active_primal_mask)
    whitening = np.asarray(fixture["whitening_matrix_by_view"])
    view_count = whitening.shape[0]
    rows_per_view = A.shape[0] // view_count
    rho_data = data_rows.reshape(view_count, rows_per_view).max(axis=1)
    sigma_data = np.divide(
        eta,
        rho_data,
        out=np.zeros_like(rho_data),
        where=rho_data > 0.0,
    )
    rho_tv = tv_by_site.max(axis=-1).reshape(-1)
    sigma_tv = np.divide(
        eta,
        rho_tv,
        out=np.zeros_like(rho_tv),
        where=rho_tv > 0.0,
    )
    tau = eta / total_columns[active_primal]
    return IndependentGateAOracle(
        schema_version=ORACLE_SCHEMA,
        shape_zyx=shape,
        spacing_xyz=spacing,
        E=E,
        G=G,
        P=P,
        W=W,
        D_plus=D_plus,
        A=A,
        M=M,
        D=D,
        N=N,
        target=target,
        eta=eta,
        data_row_sums=data_rows,
        data_column_sums=data_columns,
        tv_row_sums=tv_rows,
        tv_column_sums=tv_columns,
        data_row_mask=data_mask,
        tv_site_mask=tv_site_mask,
        active_primal_mask=active_primal_mask,
        active_primal_indices=active_primal,
        rho_data_by_view=rho_data,
        sigma_data_by_view=sigma_data,
        rho_tv_by_site=rho_tv,
        sigma_tv_by_site=sigma_tv,
        tau=tau,
    )


def run_numpy_recurrence(
    oracle: IndependentGateAOracle,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    fixture = config["fixture"]
    weight = float(fixture["regularization_weight"])
    delta = float(fixture["huber_delta"])
    theta = float(fixture["theta"])
    active = oracle.active_primal_indices
    x = np.zeros(active.size, dtype=np.float64)
    x_bar = np.zeros_like(x)
    p = np.zeros_like(oracle.target)
    q = np.zeros(oracle.D.shape[0], dtype=np.float64)
    trace: list[dict[str, Any]] = []
    sigma_data = oracle.data_sigma_rows
    sigma_tv = oracle.tv_sigma_rows
    tv_site_mask = oracle.tv_site_mask.reshape(oracle.shape_zyx)
    for _ in range(int(fixture["iterations"])):
        expanded = np.zeros(oracle.E.shape[1], dtype=np.float64)
        expanded[active] = x_bar
        candidate_p = (
            p + sigma_data * (oracle.A @ expanded - oracle.target)
        ) / (1.0 + sigma_data)
        p = np.where(oracle.data_row_mask, candidate_p, 0.0)
        candidate_q = q + sigma_tv * (oracle.D @ expanded)
        candidate_q = candidate_q / (1.0 + sigma_tv * delta / weight)
        q_field = candidate_q.reshape(3, *oracle.shape_zyx)
        norms = np.linalg.norm(q_field, axis=0)
        q_field = q_field / np.maximum(norms / weight, 1.0)[None]
        q_field = np.where(tv_site_mask[None], q_field, 0.0)
        q = q_field.reshape(-1)
        gradient = (oracle.A.T @ p + oracle.D.T @ q)[active]
        next_x = x - oracle.tau * gradient
        next_x_bar = next_x + theta * (next_x - x)
        x, x_bar = next_x, next_x_bar
        expanded_x = np.zeros(oracle.E.shape[1], dtype=np.float64)
        expanded_x[active] = x
        residual = oracle.A @ expanded_x - oracle.target
        gradient_field = (oracle.D @ expanded_x).reshape(3, *oracle.shape_zyx)
        magnitude = np.linalg.norm(gradient_field, axis=0)
        huber = np.where(
            magnitude <= delta,
            0.5 * magnitude**2 / delta,
            magnitude - 0.5 * delta,
        )
        objective = 0.5 * float(residual @ residual) + weight * float(huber.sum())
        trace.append(
            {
                "x": x.copy(),
                "x_bar": x_bar.copy(),
                "data_dual": p.reshape(
                    np.asarray(fixture["target_view_ray_uv"]).shape
                ).copy(),
                "tv_dual": q_field.copy(),
                "objective": objective,
            }
        )
    return trace


def run_torch_dense_recurrence(
    oracle: IndependentGateAOracle,
    config: dict[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> list[dict[str, torch.Tensor]]:
    fixture = config["fixture"]
    target_device = torch.device(device)
    A = torch.as_tensor(oracle.A, dtype=dtype, device=target_device)
    D = torch.as_tensor(oracle.D, dtype=dtype, device=target_device)
    target = torch.as_tensor(oracle.target, dtype=dtype, device=target_device)
    active = torch.as_tensor(
        oracle.active_primal_indices, dtype=torch.int64, device=target_device
    )
    tau = torch.as_tensor(oracle.tau, dtype=dtype, device=target_device)
    data_mask = torch.as_tensor(
        oracle.data_row_mask, dtype=torch.bool, device=target_device
    )
    sigma_data = torch.as_tensor(
        oracle.data_sigma_rows, dtype=dtype, device=target_device
    )
    sigma_tv = torch.as_tensor(
        oracle.tv_sigma_rows, dtype=dtype, device=target_device
    )
    tv_site_mask = torch.as_tensor(
        oracle.tv_site_mask.reshape(oracle.shape_zyx),
        dtype=torch.bool,
        device=target_device,
    )
    weight = float(fixture["regularization_weight"])
    delta = float(fixture["huber_delta"])
    theta = float(fixture["theta"])
    x = torch.zeros(active.numel(), dtype=dtype, device=target_device)
    x_bar = torch.zeros_like(x)
    p = torch.zeros_like(target)
    q = torch.zeros(D.shape[0], dtype=dtype, device=target_device)
    trace: list[dict[str, torch.Tensor]] = []
    for _ in range(int(fixture["iterations"])):
        expanded = torch.zeros(A.shape[1], dtype=dtype, device=target_device)
        expanded.index_copy_(0, active, x_bar)
        p_candidate = (p + sigma_data * (A @ expanded - target)) / (
            1.0 + sigma_data
        )
        p = torch.where(data_mask, p_candidate, torch.zeros_like(p_candidate))
        q_candidate = q + sigma_tv * (D @ expanded)
        q_candidate = q_candidate / (1.0 + sigma_tv * delta / weight)
        q_field = q_candidate.reshape(3, *oracle.shape_zyx)
        q_field = q_field / torch.clamp(
            torch.linalg.vector_norm(q_field, dim=0) / weight,
            min=1.0,
        )[None]
        q_field = torch.where(tv_site_mask[None], q_field, torch.zeros_like(q_field))
        q = q_field.reshape(-1)
        gradient = (A.T @ p + D.T @ q).index_select(0, active)
        next_x = x - tau * gradient
        next_x_bar = next_x + theta * (next_x - x)
        x, x_bar = next_x, next_x_bar
        trace.append(
            {
                "x": x.clone(),
                "x_bar": x_bar.clone(),
                "data_dual": p.reshape(
                    np.asarray(fixture["target_view_ray_uv"]).shape
                ).clone(),
                "tv_dual": q_field.clone(),
            }
        )
    return trace


__all__ = [
    "IndependentGateAOracle",
    "ORACLE_SCHEMA",
    "build_independent_oracle",
    "run_numpy_recurrence",
    "run_torch_dense_recurrence",
]
