"""Source-independent analytic forward geometry for PSU BOST rays.

The two primitives in this module operate on rays

    x(t) = origin + t * direction_unit,  t >= 0,

so every reported parameter and length is metric, regardless of the input
direction magnitude.  Misses and zero-measure contacts are reported with
``enter == exit == length == 0`` and ``hit == False``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


CONTRACT_VERSION = "psu-bost-forward-geometry-1.0"
_DEFAULT_RTOL = 64.0 * np.finfo(np.float64).eps

FORWARD_GEOMETRY_CONTRACT: dict[str, Any] = {
    "version": CONTRACT_VERSION,
    "provenance": (
        "Self-contained analytic NumPy float64 implementation. It does not "
        "import, execute, translate, or depend on external geometry source."
    ),
    "ray": "x(t) = origin + t * normalized_direction with t >= 0",
    "units": (
        "enter, exit, and length use the same metric units as origins, box "
        "bounds, and cone vertex"
    ),
    "input_layout": (
        "(N, 3), (3, N), or a single (3,) vector; auto layout rejects (3, 3) "
        "as ambiguous and an explicit layout resolves it"
    ),
    "interval": (
        "hit means a strictly positive-measure interval; misses, tangencies, "
        "and point contacts return zero enter/exit/length"
    ),
    "box": "closed axis-aligned box, clipped to the forward ray",
    "cone": (
        "K = {vertex + q: alpha = dot(q, axis) >= 0 and "
        "norm(q - alpha*axis) <= alpha*tan(theta)}"
    ),
    "dtype": "all numeric outputs are numpy.float64",
}

Layout = Literal["auto", "rows", "columns", "N3", "3N"]


def _validate_tolerances(atol: float, rtol: float) -> tuple[float, float]:
    atol = float(atol)
    rtol = float(rtol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a finite non-negative float")
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be a finite non-negative float")
    return atol, rtol


def _layout_name(layout: Layout | str) -> Literal["auto", "rows", "columns"]:
    aliases = {
        "auto": "auto",
        "row": "rows",
        "rows": "rows",
        "n3": "rows",
        "N3": "rows",
        "column": "columns",
        "columns": "columns",
        "3n": "columns",
        "3N": "columns",
    }
    try:
        return aliases[str(layout)]
    except KeyError as exc:
        raise ValueError("layout must be 'auto', 'rows'/'N3', or 'columns'/'3N'") from exc


def _as_ray_rows(
    value: Any,
    name: str,
    layout: Literal["auto", "rows", "columns"],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), (N, 3), or (3, N)")
        rows = array.reshape(1, 3)
    elif array.ndim == 2:
        if layout == "rows":
            if array.shape[1] != 3:
                raise ValueError(f"{name} must have shape (N, 3) for row layout")
            rows = array
        elif layout == "columns":
            if array.shape[0] != 3:
                raise ValueError(f"{name} must have shape (3, N) for column layout")
            rows = array.T
        elif array.shape == (3, 3):
            raise ValueError(
                f"{name} has ambiguous shape (3, 3); pass layout='rows' or "
                "layout='columns'"
            )
        elif array.shape[1] == 3:
            rows = array
        elif array.shape[0] == 3:
            rows = array.T
        else:
            raise ValueError(f"{name} must have shape (N, 3) or (3, N)")
    else:
        raise ValueError(f"{name} must have shape (3,), (N, 3), or (3, N)")

    rows = np.ascontiguousarray(rows, dtype=np.float64)
    if rows.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one ray")
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{name} must contain only finite values")
    return rows


def _canonical_rays(
    origins: Any,
    directions: Any,
    layout: Layout | str,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_layout = _layout_name(layout)
    origin_rows = _as_ray_rows(origins, "origins", normalized_layout)
    direction_rows = _as_ray_rows(directions, "directions", normalized_layout)
    if origin_rows.shape[0] != direction_rows.shape[0]:
        raise ValueError("origins and directions must contain the same number of rays")
    return origin_rows, direction_rows


def _as_vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 3:
        raise ValueError(f"{name} must contain exactly three values")
    vector = np.ascontiguousarray(array.reshape(3), dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _stable_row_norm(rows: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(rows), axis=1)
    norm = np.zeros(rows.shape[0], dtype=np.float64)
    nonzero = scale > 0.0
    scaled = np.zeros_like(rows)
    np.divide(rows, scale[:, None], out=scaled, where=nonzero[:, None])
    norm[nonzero] = scale[nonzero] * np.sqrt(
        np.sum(scaled[nonzero] * scaled[nonzero], axis=1)
    )
    return norm


def _unit_directions(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = _stable_row_norm(directions)
    zero = norms == 0.0
    if np.any(zero):
        indices = np.flatnonzero(zero).tolist()
        raise ValueError(f"zero direction vector(s) at ray indices {indices}")
    if not np.all(np.isfinite(norms)):
        raise ValueError("direction norms must be finite")
    unit = directions / norms[:, None]
    return np.ascontiguousarray(unit, dtype=np.float64), norms


def _finite_scale(*values: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    scale = np.ones(np.broadcast_shapes(*(array.shape for array in arrays)))
    for array in arrays:
        finite_absolute = np.where(np.isfinite(array), np.abs(array), 0.0)
        scale = np.maximum(scale, finite_absolute)
    return scale


def _interval_masks(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = _finite_scale(lower, upper)
    tolerance = atol + rtol * scale
    nonempty = upper >= lower - tolerance
    positive = upper > lower + tolerance
    point = nonempty & ~positive
    return nonempty, positive, point


def intersect_forward_ray_box(
    origins: Any,
    directions: Any,
    box_minimum: Any,
    box_maximum: Any,
    *,
    layout: Layout | str = "auto",
    atol: float = 0.0,
    rtol: float = _DEFAULT_RTOL,
) -> dict[str, Any]:
    """Intersect forward rays with a closed axis-aligned box.

    Inputs are converted to float64 and directions are normalized per ray.
    Parallel slab coordinates are handled by masks, so no division by zero is
    evaluated.  ``hit`` is true only for a positive-length forward interval.
    """

    atol, rtol = _validate_tolerances(atol, rtol)
    origin, direction = _canonical_rays(origins, directions, layout)
    direction_unit, direction_norm = _unit_directions(direction)
    lower = _as_vector3(box_minimum, "box_minimum")
    upper = _as_vector3(box_maximum, "box_maximum")
    if np.any(lower > upper):
        raise ValueError("box_minimum must be componentwise <= box_maximum")

    ray_count = origin.shape[0]
    coordinate_scale = np.maximum.reduce(
        (
            np.ones_like(origin),
            np.abs(origin),
            np.broadcast_to(np.abs(lower), origin.shape),
            np.broadcast_to(np.abs(upper), origin.shape),
        )
    )
    slab_tolerance = atol + rtol * coordinate_scale

    parallel_axis = direction_unit == 0.0
    parallel_outside_axis = parallel_axis & (
        (origin < lower[None, :] - slab_tolerance)
        | (origin > upper[None, :] + slab_tolerance)
    )
    parallel_on_boundary_axis = parallel_axis & (
        (np.abs(origin - lower[None, :]) <= slab_tolerance)
        | (np.abs(origin - upper[None, :]) <= slab_tolerance)
    )

    first = np.full((ray_count, 3), -np.inf, dtype=np.float64)
    second = np.full((ray_count, 3), np.inf, dtype=np.float64)
    nonparallel = ~parallel_axis
    delta_lower = lower[None, :] - origin
    delta_upper = upper[None, :] - origin
    np.divide(delta_lower, direction_unit, out=first, where=nonparallel)
    np.divide(delta_upper, direction_unit, out=second, where=nonparallel)
    slab_near = np.minimum(first, second)
    slab_far = np.maximum(first, second)

    line_enter = np.max(slab_near, axis=1)
    line_exit = np.min(slab_far, axis=1)
    parallel_outside = np.any(parallel_outside_axis, axis=1)

    line_nonempty, line_positive, line_point_touch = _interval_masks(
        line_enter, line_exit, atol=atol, rtol=rtol
    )
    line_nonempty &= ~parallel_outside
    line_positive &= ~parallel_outside
    line_point_touch &= ~parallel_outside

    forward_enter_raw = np.maximum(line_enter, 0.0)
    forward_exit_raw = line_exit
    forward_nonempty, forward_positive, forward_point_touch = _interval_masks(
        forward_enter_raw, forward_exit_raw, atol=atol, rtol=rtol
    )
    forward_nonempty &= line_nonempty
    forward_positive &= line_nonempty
    forward_point_touch &= line_nonempty

    hit = forward_positive
    enter = np.where(hit, forward_enter_raw, 0.0).astype(np.float64)
    exit_ = np.where(hit, forward_exit_raw, 0.0).astype(np.float64)
    length = np.where(hit, exit_ - enter, 0.0).astype(np.float64)

    origin_inside = np.all(
        (origin >= lower[None, :] - slab_tolerance)
        & (origin <= upper[None, :] + slab_tolerance),
        axis=1,
    )
    forward_clipped = hit & (line_enter < 0.0)
    behind_origin = line_nonempty & (line_exit < 0.0)

    return {
        "contract_version": CONTRACT_VERSION,
        "origin": origin,
        "direction_unit": direction_unit,
        "direction_norm": direction_norm.astype(np.float64),
        "enter": enter,
        "exit": exit_,
        "length": length,
        "hit": hit,
        "line_enter": line_enter.astype(np.float64),
        "line_exit": line_exit.astype(np.float64),
        "line_positive": line_positive,
        "line_point_touch": line_point_touch,
        "forward_nonempty": forward_nonempty,
        "forward_point_touch": forward_point_touch,
        "origin_inside": origin_inside,
        "forward_clipped": forward_clipped,
        "behind_origin": behind_origin,
        "parallel_axis": parallel_axis,
        "parallel_outside_axis": parallel_outside_axis,
        "parallel_outside": parallel_outside,
        "parallel_on_boundary_axis": parallel_on_boundary_axis,
        "box_zero_extent_axis": lower == upper,
    }


def _quadratic_inequality_intervals(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    coordinate_scale: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, np.ndarray]:
    """Return up to two closed intervals satisfying a*t^2 + b*t + c <= 0."""

    ray_count = a.shape[0]
    lower = np.full((ray_count, 2), np.inf, dtype=np.float64)
    upper = np.full((ray_count, 2), -np.inf, dtype=np.float64)

    a_scale = np.maximum(1.0, np.abs(a))
    a_tolerance = rtol * a_scale
    a_near_zero = np.abs(a) <= a_tolerance
    a_positive = a > a_tolerance
    a_negative = a < -a_tolerance

    b_scale = np.maximum.reduce((np.ones_like(b), np.abs(b), coordinate_scale))
    b_tolerance = atol + rtol * b_scale
    b_near_zero = np.abs(b) <= b_tolerance

    c_scale = np.maximum.reduce(
        (np.ones_like(c), np.abs(c), coordinate_scale * coordinate_scale)
    )
    c_tolerance = atol * atol + rtol * c_scale

    with np.errstate(over="ignore", invalid="ignore"):
        discriminant = b * b - 4.0 * a * c
        discriminant_scale = np.maximum.reduce(
            (
                np.ones_like(discriminant),
                b * b,
                np.abs(4.0 * a * c),
                coordinate_scale * coordinate_scale,
            )
        )
    if not np.all(np.isfinite(discriminant)):
        raise ValueError("cone coefficient scale overflowed float64")

    discriminant_tolerance = atol * atol + rtol * discriminant_scale
    discriminant_positive = discriminant > discriminant_tolerance
    discriminant_negative = discriminant < -discriminant_tolerance
    discriminant_near_zero = ~(discriminant_positive | discriminant_negative)

    root_low = np.full(ray_count, np.nan, dtype=np.float64)
    root_high = np.full(ray_count, np.nan, dtype=np.float64)
    two_root_mask = ~a_near_zero & discriminant_positive
    if np.any(two_root_mask):
        sqrt_discriminant = np.sqrt(discriminant[two_root_mask])
        selected_b = b[two_root_mask]
        selected_a = a[two_root_mask]
        selected_c = c[two_root_mask]
        q = -0.5 * (
            selected_b + np.copysign(sqrt_discriminant, selected_b)
        )
        first_root = q / selected_a
        second_root = np.empty_like(first_root)
        q_nonzero = q != 0.0
        np.divide(selected_c, q, out=second_root, where=q_nonzero)
        second_root[~q_nonzero] = (
            -selected_b[~q_nonzero] / (2.0 * selected_a[~q_nonzero])
        )
        root_low[two_root_mask] = np.minimum(first_root, second_root)
        root_high[two_root_mask] = np.maximum(first_root, second_root)

    tangent_mask = ~a_near_zero & discriminant_near_zero
    tangent_root = np.full(ray_count, np.nan, dtype=np.float64)
    tangent_root[tangent_mask] = (
        -b[tangent_mask] / (2.0 * a[tangent_mask])
    )

    upward_secant = a_positive & discriminant_positive
    lower[upward_secant, 0] = root_low[upward_secant]
    upper[upward_secant, 0] = root_high[upward_secant]

    upward_tangent = a_positive & discriminant_near_zero
    lower[upward_tangent, 0] = tangent_root[upward_tangent]
    upper[upward_tangent, 0] = tangent_root[upward_tangent]

    downward_secant = a_negative & discriminant_positive
    lower[downward_secant, 0] = -np.inf
    upper[downward_secant, 0] = root_low[downward_secant]
    lower[downward_secant, 1] = root_high[downward_secant]
    upper[downward_secant, 1] = np.inf

    downward_all = a_negative & ~discriminant_positive
    lower[downward_all, 0] = -np.inf
    upper[downward_all, 0] = np.inf

    linear = a_near_zero & ~b_near_zero
    linear_root = np.full(ray_count, np.nan, dtype=np.float64)
    linear_root[linear] = -c[linear] / b[linear]
    linear_positive = linear & (b > 0.0)
    lower[linear_positive, 0] = -np.inf
    upper[linear_positive, 0] = linear_root[linear_positive]
    linear_negative = linear & (b < 0.0)
    lower[linear_negative, 0] = linear_root[linear_negative]
    upper[linear_negative, 0] = np.inf

    constant = a_near_zero & b_near_zero
    constant_satisfied = constant & (c <= c_tolerance)
    lower[constant_satisfied, 0] = -np.inf
    upper[constant_satisfied, 0] = np.inf

    return {
        "lower": lower,
        "upper": upper,
        "a_near_zero": a_near_zero,
        "used_linear_inequality": linear,
        "constant_inequality": constant,
        "constant_satisfied": constant_satisfied,
        "discriminant": discriminant.astype(np.float64),
        "discriminant_tolerance": discriminant_tolerance.astype(np.float64),
        "discriminant_positive": discriminant_positive & ~a_near_zero,
        "discriminant_negative": discriminant_negative & ~a_near_zero,
        "discriminant_near_zero": discriminant_near_zero & ~a_near_zero,
        "discriminant_clamped_from_negative": (
            (discriminant < 0.0) & discriminant_near_zero & ~a_near_zero
        ),
        "root_low": root_low,
        "root_high": root_high,
    }


def intersect_forward_ray_box_cone(
    origins: Any,
    directions: Any,
    box_minimum: Any,
    box_maximum: Any,
    cone_vertex: Any,
    cone_axis: Any,
    theta: float,
    *,
    layout: Layout | str = "auto",
    atol: float = 0.0,
    rtol: float = _DEFAULT_RTOL,
) -> dict[str, Any]:
    """Intersect forward rays with ``box intersect one-nappe cone``.

    The cone inequality is solved analytically as a quadratic (or its linear
    limit), then intersected with ``alpha >= 0``, ``t >= 0``, and the B0 box
    interval.  Tangencies and apex-only contacts are zero-measure misses.
    """

    atol, rtol = _validate_tolerances(atol, rtol)
    box = intersect_forward_ray_box(
        origins,
        directions,
        box_minimum,
        box_maximum,
        layout=layout,
        atol=atol,
        rtol=rtol,
    )
    origin = box["origin"]
    direction_unit = box["direction_unit"]
    vertex = _as_vector3(cone_vertex, "cone_vertex")
    axis_raw = _as_vector3(cone_axis, "cone_axis")
    axis_norm = _stable_row_norm(axis_raw.reshape(1, 3))[0]
    if axis_norm == 0.0 or not np.isfinite(axis_norm):
        raise ValueError("cone_axis must have a finite nonzero norm")
    axis = axis_raw / axis_norm

    theta = float(theta)
    if not np.isfinite(theta) or not (0.0 < theta < 0.5 * np.pi):
        raise ValueError("theta must be finite and strictly between 0 and pi/2")
    tangent_theta = float(np.tan(theta))
    tangent_squared = tangent_theta * tangent_theta
    if not np.isfinite(tangent_squared):
        raise ValueError("theta produces a non-finite cone slope")

    q0 = origin - vertex[None, :]
    alpha0 = q0 @ axis
    beta = direction_unit @ axis
    q_perpendicular = q0 - alpha0[:, None] * axis[None, :]
    direction_perpendicular = (
        direction_unit - beta[:, None] * axis[None, :]
    )

    with np.errstate(over="ignore", invalid="ignore"):
        perpendicular_direction_squared = np.sum(
            direction_perpendicular * direction_perpendicular, axis=1
        )
        perpendicular_origin_squared = np.sum(
            q_perpendicular * q_perpendicular, axis=1
        )
        cross_perpendicular = np.sum(
            q_perpendicular * direction_perpendicular, axis=1
        )
        coefficient_a = (
            perpendicular_direction_squared - tangent_squared * beta * beta
        )
        coefficient_b = 2.0 * (
            cross_perpendicular - tangent_squared * alpha0 * beta
        )
        coefficient_c = (
            perpendicular_origin_squared
            - tangent_squared * alpha0 * alpha0
        )
    coefficient_arrays = (coefficient_a, coefficient_b, coefficient_c)
    if not all(np.all(np.isfinite(value)) for value in coefficient_arrays):
        raise ValueError("cone coefficients overflowed float64")

    coordinate_scale = np.maximum(
        1.0,
        np.maximum(
            _stable_row_norm(q0),
            np.max(np.abs(q0), axis=1),
        ),
    )
    quadratic = _quadratic_inequality_intervals(
        coefficient_a,
        coefficient_b,
        coefficient_c,
        coordinate_scale=coordinate_scale,
        atol=atol,
        rtol=rtol,
    )

    alpha_tolerance = atol + rtol * np.maximum(
        coordinate_scale, np.abs(alpha0)
    )
    beta_tolerance = rtol * np.maximum(1.0, np.abs(beta))
    beta_positive = beta > beta_tolerance
    beta_negative = beta < -beta_tolerance
    beta_near_zero = ~(beta_positive | beta_negative)

    nappe_lower = np.full(origin.shape[0], -np.inf, dtype=np.float64)
    nappe_upper = np.full(origin.shape[0], np.inf, dtype=np.float64)
    nappe_lower[beta_positive] = -alpha0[beta_positive] / beta[beta_positive]
    nappe_upper[beta_negative] = -alpha0[beta_negative] / beta[beta_negative]
    nappe_impossible = beta_near_zero & (alpha0 < -alpha_tolerance)
    nappe_lower[nappe_impossible] = np.inf
    nappe_upper[nappe_impossible] = -np.inf

    double_lower = np.maximum(quadratic["lower"], 0.0)
    double_upper = quadratic["upper"].copy()
    _, double_positive, double_point = _interval_masks(
        double_lower, double_upper, atol=atol, rtol=rtol
    )
    double_cone_forward_hit = np.any(double_positive, axis=1)
    double_cone_forward_point_touch = (
        ~double_cone_forward_hit & np.any(double_point, axis=1)
    )

    cone_lower = np.maximum.reduce(
        (
            quadratic["lower"],
            np.broadcast_to(nappe_lower[:, None], quadratic["lower"].shape),
            np.zeros_like(quadratic["lower"]),
        )
    )
    cone_upper = np.minimum(
        quadratic["upper"],
        np.broadcast_to(nappe_upper[:, None], quadratic["upper"].shape),
    )
    cone_nonempty_interval, cone_positive_interval, cone_point_interval = (
        _interval_masks(cone_lower, cone_upper, atol=atol, rtol=rtol)
    )
    cone_hit = np.any(cone_positive_interval, axis=1)
    cone_point_touch = ~cone_hit & np.any(cone_point_interval, axis=1)
    multiple_cone_intervals = np.count_nonzero(
        cone_positive_interval, axis=1
    ) > 1

    cone_enter_raw = np.min(
        np.where(cone_positive_interval, cone_lower, np.inf), axis=1
    )
    cone_exit_raw = np.max(
        np.where(cone_positive_interval, cone_upper, -np.inf), axis=1
    )
    cone_enter = np.where(cone_hit, cone_enter_raw, 0.0).astype(np.float64)
    cone_exit = np.where(cone_hit, cone_exit_raw, 0.0).astype(np.float64)
    cone_length = np.where(
        cone_hit, cone_exit_raw - cone_enter_raw, 0.0
    ).astype(np.float64)

    final_lower = np.maximum(
        cone_lower, np.broadcast_to(box["enter"][:, None], cone_lower.shape)
    )
    final_upper = np.minimum(
        cone_upper, np.broadcast_to(box["exit"][:, None], cone_upper.shape)
    )
    _, final_positive_interval, final_point_interval = _interval_masks(
        final_lower, final_upper, atol=atol, rtol=rtol
    )
    final_positive_interval &= box["hit"][:, None]
    final_point_interval &= box["forward_nonempty"][:, None]
    hit = np.any(final_positive_interval, axis=1)
    point_touch = ~hit & np.any(final_point_interval, axis=1)

    enter_raw = np.min(
        np.where(final_positive_interval, final_lower, np.inf), axis=1
    )
    exit_raw = np.max(
        np.where(final_positive_interval, final_upper, -np.inf), axis=1
    )
    enter = np.where(hit, enter_raw, 0.0).astype(np.float64)
    exit_ = np.where(hit, exit_raw, 0.0).astype(np.float64)
    length = np.where(hit, exit_raw - enter_raw, 0.0).astype(np.float64)

    closest_apex_parameter = -np.sum(q0 * direction_unit, axis=1)
    closest_apex_offset = (
        q0 + closest_apex_parameter[:, None] * direction_unit
    )
    closest_apex_distance = _stable_row_norm(closest_apex_offset)
    apex_tolerance = atol + rtol * coordinate_scale
    parameter_tolerance = atol + rtol * np.maximum(
        1.0, np.abs(closest_apex_parameter)
    )
    apex_on_forward_ray = (
        (closest_apex_distance <= apex_tolerance)
        & (closest_apex_parameter >= -parameter_tolerance)
    )
    apex_in_box_interval = (
        apex_on_forward_ray
        & box["forward_nonempty"]
        & (closest_apex_parameter >= box["line_enter"] - parameter_tolerance)
        & (closest_apex_parameter <= box["line_exit"] + parameter_tolerance)
    )

    coefficient_c_tolerance = (
        atol * atol
        + rtol
        * np.maximum(
            1.0,
            np.maximum(np.abs(coefficient_c), coordinate_scale**2),
        )
    )
    origin_in_cone = (
        (alpha0 >= -alpha_tolerance)
        & (coefficient_c <= coefficient_c_tolerance)
    )
    origin_on_cone_boundary = (
        (alpha0 >= -alpha_tolerance)
        & (np.abs(coefficient_c) <= coefficient_c_tolerance)
    )
    origin_at_apex = _stable_row_norm(q0) <= apex_tolerance

    box_cone_disjoint = box["hit"] & cone_hit & ~hit
    cone_no_box_overlap = cone_hit & ~hit
    nappe_rejected = double_cone_forward_hit & ~cone_hit
    discriminant_tangent = (
        quadratic["discriminant_near_zero"] & ~quadratic["a_near_zero"]
    )

    return {
        "contract_version": CONTRACT_VERSION,
        "origin": origin,
        "direction_unit": direction_unit,
        "direction_norm": box["direction_norm"],
        "enter": enter,
        "exit": exit_,
        "length": length,
        "hit": hit,
        "point_touch": point_touch,
        "box_enter": box["enter"],
        "box_exit": box["exit"],
        "box_length": box["length"],
        "box_hit": box["hit"],
        "box_point_touch": box["forward_point_touch"],
        "cone_enter": cone_enter,
        "cone_exit": cone_exit,
        "cone_length": cone_length,
        "cone_hit": cone_hit,
        "cone_point_touch": cone_point_touch,
        "cone_interval_lower": cone_lower,
        "cone_interval_upper": cone_upper,
        "cone_interval_nonempty": cone_nonempty_interval,
        "cone_interval_positive": cone_positive_interval,
        "coefficient_a": coefficient_a.astype(np.float64),
        "coefficient_b": coefficient_b.astype(np.float64),
        "coefficient_c": coefficient_c.astype(np.float64),
        "quadratic_degenerate": quadratic["a_near_zero"],
        "used_linear_inequality": quadratic["used_linear_inequality"],
        "constant_inequality": quadratic["constant_inequality"],
        "constant_inequality_satisfied": quadratic["constant_satisfied"],
        "discriminant": quadratic["discriminant"],
        "discriminant_tolerance": quadratic["discriminant_tolerance"],
        "discriminant_positive": quadratic["discriminant_positive"],
        "discriminant_negative": quadratic["discriminant_negative"],
        "discriminant_near_zero": quadratic["discriminant_near_zero"],
        "discriminant_clamped_from_negative": quadratic[
            "discriminant_clamped_from_negative"
        ],
        "discriminant_tangent": discriminant_tangent,
        "double_cone_forward_hit": double_cone_forward_hit,
        "double_cone_forward_point_touch": double_cone_forward_point_touch,
        "nappe_rejected": nappe_rejected,
        "nappe_parallel": beta_near_zero,
        "origin_in_cone": origin_in_cone,
        "origin_on_cone_boundary": origin_on_cone_boundary,
        "origin_at_apex": origin_at_apex,
        "apex_on_forward_ray": apex_on_forward_ray,
        "apex_in_box_interval": apex_in_box_interval,
        "multiple_cone_intervals": multiple_cone_intervals,
        "box_cone_disjoint": box_cone_disjoint,
        "cone_no_box_overlap": cone_no_box_overlap,
        "box_miss_with_cone_hit": ~box["hit"] & cone_hit,
        "parallel_axis": box["parallel_axis"],
        "parallel_outside": box["parallel_outside"],
        "box_zero_extent_axis": box["box_zero_extent_axis"],
        "cone_axis_unit": axis.astype(np.float64),
        "cone_axis_norm": np.float64(axis_norm),
        "cone_tan_theta": np.float64(tangent_theta),
    }


# Descriptive aliases plus the B0/B1 labels used by the validation plan.
forward_ray_box_intersection = intersect_forward_ray_box
forward_ray_box_cone_intersection = intersect_forward_ray_box_cone
b0_forward_ray_box_intersection = intersect_forward_ray_box
b1_forward_ray_box_cone_intersection = intersect_forward_ray_box_cone


__all__ = [
    "CONTRACT_VERSION",
    "FORWARD_GEOMETRY_CONTRACT",
    "intersect_forward_ray_box",
    "intersect_forward_ray_box_cone",
    "forward_ray_box_intersection",
    "forward_ray_box_cone_intersection",
    "b0_forward_ray_box_intersection",
    "b1_forward_ray_box_cone_intersection",
]
