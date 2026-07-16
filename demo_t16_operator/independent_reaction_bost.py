"""Independent small 3D reaction-field and curved/cone-ray BOST generator.

This module intentionally does not call ``make_phantom`` or
``build_forward_matrix``. It is a local L1 stress generator: rays can move
between z planes, field families use reaction-front geometry, and the noise
helper includes correlations that are absent from the reconstruction model.
It is still a prescribed weak-deflection linear model, not a substitute for
nonlinear ray tracing or real OpenBOST/OERF data.
"""

from __future__ import annotations

import hashlib

import numpy as np


def grid_3d(n: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, int(n), dtype=np.float64)
    z_axis = np.linspace(-1.0, 1.0, int(depth), dtype=np.float64)
    zz, yy, xx = np.meshgrid(z_axis, axis, axis, indexing="ij")
    return xx, yy, zz


def reaction_support(n: int, depth: int) -> np.ndarray:
    xx, yy, zz = grid_3d(n, depth)
    radius = np.sqrt(xx**2 + yy**2)
    radial = 0.5 * (1.0 - np.tanh((radius - 0.88) / 0.055))
    axial = 0.5 * (1.0 - np.tanh((np.abs(zz) - 0.91) / 0.055))
    return (radial * axial).astype(np.float32)


def _normalize_field(field: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = np.maximum(field, 0.0) * support
    values -= values.min()
    values /= values.max() + 1e-10
    return values.astype(np.float32)


def _burned_kernel(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    rng: np.random.Generator,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    if center is None:
        center = rng.uniform(-0.18, 0.18, size=3)
    axes = rng.uniform([0.28, 0.34, 0.34], [0.52, 0.64, 0.72])
    x = (xx - center[0]) / axes[0]
    y = (yy - center[1]) / axes[1]
    z = (zz - center[2]) / axes[2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wrinkle = (
        rng.uniform(0.035, 0.09) * np.sin(rng.uniform(3.0, 6.0) * yy + phase)
        + rng.uniform(0.025, 0.07) * np.sin(rng.uniform(2.0, 5.0) * zz - phase)
    )
    thickness = rng.uniform(0.025, 0.06)
    return 0.5 * (1.0 - np.tanh((radius - 1.0 - wrinkle) / thickness))


def make_reaction_field(
    family: str,
    n: int,
    depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a normalized density-deficit proxy with reaction-flow shapes."""

    xx, yy, zz = grid_3d(n, depth)
    support = reaction_support(n, depth)
    if family == "expanding_kernel":
        field = _burned_kernel(xx, yy, zz, rng)
        field *= 0.75 + 0.25 * np.cos(rng.uniform(1.0, 2.5) * zz + rng.uniform(0, 6.28))
    elif family == "jet_shear":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        center_x = rng.uniform(-0.12, 0.12) + rng.uniform(0.07, 0.17) * np.sin(
            rng.uniform(2.0, 4.5) * yy + phase
        )
        center_z = rng.uniform(-0.12, 0.12) + rng.uniform(0.06, 0.15) * np.sin(
            rng.uniform(2.0, 4.0) * yy - 0.7 * phase
        )
        radial = np.sqrt((xx - center_x) ** 2 + (zz - center_z) ** 2)
        radius = rng.uniform(0.13, 0.22) + rng.uniform(0.06, 0.13) * (yy + 1.0)
        thickness = rng.uniform(0.022, 0.055)
        core = 0.5 * (1.0 - np.tanh((radial - radius) / thickness))
        inlet = 0.5 * (1.0 + np.tanh((yy + rng.uniform(0.65, 0.85)) / 0.08))
        field = core * inlet * np.exp(-0.12 * (yy + 0.5) ** 2)
    elif family == "interacting_fronts":
        offset = rng.uniform(0.20, 0.34)
        first = _burned_kernel(
            xx, yy, zz, rng, center=np.array([-offset, rng.uniform(-0.1, 0.1), 0.0])
        )
        second = _burned_kernel(
            xx, yy, zz, rng, center=np.array([offset, rng.uniform(-0.1, 0.1), 0.0])
        )
        overlap = np.sqrt(np.maximum(first * second, 0.0))
        field = np.maximum(first, second) + rng.uniform(0.15, 0.35) * overlap
    elif family == "shock_cell":
        radial = np.sqrt(
            (xx - rng.uniform(-0.08, 0.08)) ** 2
            + (zz - rng.uniform(-0.08, 0.08)) ** 2
        )
        envelope = np.exp(-0.5 * (radial / rng.uniform(0.28, 0.42)) ** 4)
        frequency = rng.uniform(6.0, 10.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        cells = 0.5 + 0.5 * np.tanh(
            rng.uniform(4.0, 7.0) * np.cos(frequency * yy + phase)
        )
        oblique = 0.5 * (
            1.0
            + np.tanh(
                (yy - rng.uniform(-0.25, 0.25) - rng.uniform(0.18, 0.32) * xx)
                / rng.uniform(0.025, 0.055)
            )
        )
        field = envelope * (0.55 * cells + 0.45 * oblique)
    elif family == "helical_plume":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        frequency = rng.uniform(2.3, 4.2)
        center_x = rng.uniform(0.10, 0.22) * np.sin(frequency * (yy + 1.0) + phase)
        center_z = rng.uniform(0.10, 0.22) * np.cos(frequency * (yy + 1.0) + phase)
        radial = np.sqrt((xx - center_x) ** 2 + (zz - center_z) ** 2)
        radius = rng.uniform(0.12, 0.19) + rng.uniform(0.025, 0.07) * (yy + 1.0)
        front = 0.5 * (
            1.0 - np.tanh((radial - radius) / rng.uniform(0.02, 0.05))
        )
        inlet = 0.5 * (
            1.0 + np.tanh((yy + rng.uniform(0.72, 0.88)) / 0.06)
        )
        field = front * inlet * (
            0.78 + 0.22 * np.cos(rng.uniform(3.0, 5.5) * yy + phase)
        )
    elif family == "stratified_ignition":
        tilt_x = rng.uniform(-0.38, 0.38)
        tilt_z = rng.uniform(-0.30, 0.30)
        signed = yy - rng.uniform(-0.25, 0.18) - tilt_x * xx - tilt_z * zz
        sheet = 0.5 * (1.0 + np.tanh(signed / rng.uniform(0.025, 0.06)))
        pocket = _burned_kernel(
            xx,
            yy,
            zz,
            rng,
            center=np.array(
                [
                    rng.uniform(-0.28, 0.28),
                    rng.uniform(-0.18, 0.18),
                    rng.uniform(-0.25, 0.25),
                ]
            ),
        )
        stratification = 0.62 + 0.38 / (
            1.0
            + np.exp(
                -rng.uniform(2.0, 4.5) * (zz - rng.uniform(-0.2, 0.2))
            )
        )
        field = np.maximum(0.72 * sheet * stratification, pocket)
    elif family == "wrinkled_flame_sheet":
        phase_x = rng.uniform(0.0, 2.0 * np.pi)
        phase_z = rng.uniform(0.0, 2.0 * np.pi)
        front = (
            rng.uniform(-0.18, 0.18)
            + rng.uniform(0.10, 0.20)
            * np.sin(rng.uniform(2.5, 4.8) * xx + phase_x)
            + rng.uniform(0.08, 0.17)
            * np.sin(rng.uniform(2.2, 4.5) * zz + phase_z)
            + rng.uniform(0.03, 0.08)
            * np.sin(rng.uniform(5.5, 8.5) * (xx + zz) - phase_x)
        )
        signed = yy - front - rng.uniform(-0.20, 0.20) * xx
        burned = 0.5 * (
            1.0 + np.tanh(signed / rng.uniform(0.022, 0.052))
        )
        transverse = 0.72 + 0.28 * np.exp(
            -((xx - rng.uniform(-0.12, 0.12)) ** 2 + zz**2)
            / rng.uniform(0.25, 0.45) ** 2
        )
        field = burned * transverse
    elif family == "vortex_ring_pair":
        radial = np.sqrt(
            (xx - rng.uniform(-0.08, 0.08)) ** 2
            + (zz - rng.uniform(-0.08, 0.08)) ** 2
        )
        ring_radius = rng.uniform(0.30, 0.48)
        tube = rng.uniform(0.075, 0.14)
        offset = rng.uniform(0.25, 0.42)
        first_distance = np.sqrt((radial - ring_radius) ** 2 + (yy - offset) ** 2)
        second_distance = np.sqrt(
            (radial - rng.uniform(0.82, 1.10) * ring_radius) ** 2
            + (yy + offset) ** 2
        )
        first = np.exp(-0.5 * (first_distance / tube) ** 2)
        second = np.exp(
            -0.5 * (second_distance / rng.uniform(0.85, 1.20) / tube) ** 2
        )
        bridge = np.exp(
            -0.5
            * (
                radial / rng.uniform(0.20, 0.34)
            ) ** 4
        ) * np.exp(-0.5 * (yy / rng.uniform(0.35, 0.55)) ** 2)
        field = np.maximum(first, rng.uniform(0.72, 0.95) * second)
        field += rng.uniform(0.12, 0.28) * bridge
    elif family == "triple_jet_merger":
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        offsets = np.asarray([-0.34, 0.0, 0.34]) + rng.uniform(-0.05, 0.05, size=3)
        jets = []
        for index, offset in enumerate(offsets):
            merger = offset * (0.25 + 0.75 * np.clip(-yy, 0.0, 1.0))
            center_x = merger + rng.uniform(0.035, 0.09) * np.sin(
                rng.uniform(2.5, 4.5) * yy + phase[index]
            )
            center_z = rng.uniform(-0.08, 0.08) + rng.uniform(0.03, 0.08) * np.cos(
                rng.uniform(2.0, 4.0) * yy - phase[index]
            )
            radial = np.sqrt((xx - center_x) ** 2 + (zz - center_z) ** 2)
            radius = rng.uniform(0.10, 0.16) + rng.uniform(0.03, 0.08) * (yy + 1.0)
            jets.append(
                0.5
                * (1.0 - np.tanh((radial - radius) / rng.uniform(0.02, 0.045)))
            )
        inlet = 0.5 * (
            1.0 + np.tanh((yy + rng.uniform(0.72, 0.88)) / 0.06)
        )
        field = np.maximum.reduce(jets) * inlet
        field += rng.uniform(0.08, 0.18) * np.sqrt(
            np.maximum(jets[0] * jets[2], 0.0)
        )
    elif family == "tilted_flame_brush":
        tilt_x = rng.uniform(-0.42, 0.42)
        tilt_z = rng.uniform(-0.34, 0.34)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        front = (
            rng.uniform(-0.18, 0.18)
            + tilt_x * xx
            + tilt_z * zz
            + rng.uniform(0.06, 0.13)
            * np.sin(rng.uniform(2.2, 4.0) * xx + phase[0])
            + rng.uniform(0.05, 0.12)
            * np.sin(rng.uniform(2.0, 3.8) * zz + phase[1])
        )
        thickness = rng.uniform(0.07, 0.14)
        brush = 0.5 * (1.0 + np.tanh((yy - front) / thickness))
        intermittency = 0.76 + 0.24 * np.sin(
            rng.uniform(3.5, 6.0) * (xx - zz) + phase[2]
        ) ** 2
        pilot = _burned_kernel(
            xx,
            yy,
            zz,
            rng,
            center=np.asarray(
                [
                    rng.uniform(-0.25, 0.25),
                    rng.uniform(-0.45, -0.18),
                    rng.uniform(-0.2, 0.2),
                ]
            ),
        )
        field = np.maximum(brush * intermittency, rng.uniform(0.35, 0.55) * pilot)
    elif family == "pulsed_toroidal_plume":
        radial = np.sqrt(
            (xx - rng.uniform(-0.08, 0.08)) ** 2
            + (zz - rng.uniform(-0.08, 0.08)) ** 2
        )
        ring_center = rng.uniform(-0.05, 0.28)
        ring_radius = rng.uniform(0.28, 0.46)
        ring_distance = np.sqrt(
            (radial - ring_radius) ** 2 + (yy - ring_center) ** 2
        )
        ring = np.exp(-0.5 * (ring_distance / rng.uniform(0.07, 0.13)) ** 2)
        center_x = rng.uniform(0.05, 0.14) * np.sin(
            rng.uniform(2.2, 3.8) * yy + rng.uniform(0.0, 2.0 * np.pi)
        )
        core_radius = rng.uniform(0.11, 0.19) + rng.uniform(0.02, 0.06) * (
            yy + 1.0
        )
        core = 0.5 * (
            1.0
            - np.tanh(
                (np.sqrt((xx - center_x) ** 2 + zz**2) - core_radius)
                / rng.uniform(0.02, 0.05)
            )
        )
        pulse = 0.62 + 0.38 * np.cos(
            rng.uniform(4.0, 7.0) * yy + rng.uniform(0.0, 2.0 * np.pi)
        ) ** 2
        field = np.maximum(ring, rng.uniform(0.58, 0.82) * core * pulse)
    else:
        raise ValueError(f"unknown independent reaction family: {family}")
    return _normalize_field(field, support)


def _derivative_matrix(shape: tuple[int, int, int], axis: int) -> np.ndarray:
    depth, height, width = shape
    sizes = (depth, height, width)
    spacing = 2.0 / max(sizes[axis] - 1, 1)
    count = depth * height * width
    matrix = np.zeros((count, count), dtype=np.float64)
    for index in np.ndindex(shape):
        row = np.ravel_multi_index(index, shape)
        coordinate = index[axis]
        if sizes[axis] == 1:
            continue
        low = list(index)
        high = list(index)
        if coordinate == 0:
            high[axis] = 1
            matrix[row, row] = -1.0 / spacing
            matrix[row, np.ravel_multi_index(tuple(high), shape)] = 1.0 / spacing
        elif coordinate == sizes[axis] - 1:
            low[axis] = coordinate - 1
            matrix[row, np.ravel_multi_index(tuple(low), shape)] = -1.0 / spacing
            matrix[row, row] = 1.0 / spacing
        else:
            low[axis] = coordinate - 1
            high[axis] = coordinate + 1
            matrix[row, np.ravel_multi_index(tuple(low), shape)] = -0.5 / spacing
            matrix[row, np.ravel_multi_index(tuple(high), shape)] = 0.5 / spacing
    return matrix


def _trilinear_row(
    x: float,
    y: float,
    z: float,
    shape: tuple[int, int, int],
) -> np.ndarray:
    depth, height, width = shape
    coordinates = np.array(
        [
            (z + 1.0) * 0.5 * (depth - 1),
            (y + 1.0) * 0.5 * (height - 1),
            (x + 1.0) * 0.5 * (width - 1),
        ]
    )
    if np.any(coordinates < 0.0) or np.any(coordinates > np.array(shape) - 1.0):
        return np.zeros(depth * height * width, dtype=np.float64)
    low = np.floor(coordinates).astype(int)
    low = np.minimum(low, np.array(shape) - 2)
    fraction = coordinates - low
    row = np.zeros(depth * height * width, dtype=np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                corner = low + np.array([dz, dy, dx])
                weight = (
                    (fraction[0] if dz else 1.0 - fraction[0])
                    * (fraction[1] if dy else 1.0 - fraction[1])
                    * (fraction[2] if dx else 1.0 - fraction[2])
                )
                row[np.ravel_multi_index(tuple(corner), shape)] += weight
    return row


def build_curved_cone_operator(
    n: int,
    depth: int,
    angles_degrees: np.ndarray,
    *,
    path_samples: int = 22,
    cone_u: float = 0.07,
    cone_z: float = 0.05,
    bend: float = 0.035,
) -> np.ndarray:
    """Build ``[detector_z,view,detector_x,voxel]`` weak-BOST rows."""

    if n < 3 or depth < 2 or path_samples < 8:
        raise ValueError("grid and path sampling are too small")
    shape = (int(depth), int(n), int(n))
    dz = _derivative_matrix(shape, 0)
    dy = _derivative_matrix(shape, 1)
    dx = _derivative_matrix(shape, 2)
    detector_x = np.linspace(-0.82, 0.82, int(n))
    detector_z = np.linspace(-0.82, 0.82, int(depth))
    path = np.linspace(-1.5, 1.5, int(path_samples))
    dl = float(path[1] - path[0])
    rows = np.zeros(
        (depth, len(angles_degrees), n, depth * n * n), dtype=np.float64
    )
    for z_index, z0 in enumerate(detector_z):
        for view_index, angle in enumerate(np.asarray(angles_degrees, dtype=float)):
            theta = np.deg2rad(angle)
            line = np.array([np.cos(theta), np.sin(theta), 0.0])
            transverse = np.array([-np.sin(theta), np.cos(theta), 0.0])
            for detector_index, u0 in enumerate(detector_x):
                direction = line + float(cone_u) * u0 * transverse
                direction = direction + float(cone_z) * z0 * np.array([0.0, 0.0, 1.0])
                direction /= np.linalg.norm(direction)
                sensitivity = transverse - np.dot(transverse, direction) * direction
                sensitivity /= np.linalg.norm(sensitivity)
                interpolation = np.zeros(depth * n * n, dtype=np.float64)
                for distance in path:
                    normalized = distance / max(abs(path[0]), abs(path[-1]))
                    curve = (
                        float(bend)
                        * (1.0 - normalized * normalized)
                        * (0.35 + abs(u0))
                        * transverse
                    )
                    point = u0 * transverse + z0 * np.array([0.0, 0.0, 1.0])
                    point = point + distance * direction + curve
                    interpolation += _trilinear_row(
                        float(point[0]), float(point[1]), float(point[2]), shape
                    ) * dl
                derivative = sensitivity[0] * dx + sensitivity[1] * dy + sensitivity[2] * dz
                rows[z_index, view_index, detector_index] = interpolation @ derivative
    nonzero = np.linalg.vector_norm(rows.reshape(-1, rows.shape[-1]), axis=1)
    scale = np.median(nonzero[nonzero > 1e-10])
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("curved/cone operator contains no valid rays")
    return (rows / scale).astype(np.float32)


def correlated_camera_noise(
    clean: np.ndarray,
    camera_std: np.ndarray,
    rng: np.random.Generator,
    *,
    correlation_fraction: float = 0.35,
    signal_fraction: float = 0.15,
) -> np.ndarray:
    """Generate diagonal-plus-correlated noise while declaring only its RMS."""

    if clean.ndim != 3:
        raise ValueError("clean must have shape [detector_z,view,detector_x]")
    sigma = np.broadcast_to(camera_std[None, :, None], clean.shape)
    iid = rng.normal(size=clean.shape)
    row = rng.normal(size=(clean.shape[0], clean.shape[1], 1))
    column = rng.normal(size=(1, clean.shape[1], clean.shape[2]))
    correlated = (row + column) / np.sqrt(2.0)
    signal_scale = np.abs(clean) / (np.sqrt(np.mean(clean**2)) + 1e-8)
    heteroscedastic = rng.normal(size=clean.shape) * signal_scale
    noise = (
        np.sqrt(max(1.0 - correlation_fraction**2 - signal_fraction**2, 0.0)) * iid
        + float(correlation_fraction) * correlated
        + float(signal_fraction) * heteroscedastic
    )
    return (sigma * noise).astype(np.float32)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()
