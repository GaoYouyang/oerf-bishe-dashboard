"""Tiny CPU/float64 signed-factor majorizer and PDHG oracle.

This file is deliberately separate from the formal PSU-B0 runner.  It keeps
the factors in the design note explicit::

    A_b = W_b @ P_b @ G @ E
    D   = D_plus @ E
    M_b = abs(W_b) @ P_b @ abs(G) @ E
    N   = abs(D_plus) @ E

``E`` and ``P_b`` are checked to be non-negative; ``G``, ``W_b`` and
``D_plus`` retain their signs.  The implementation is a tiny dense oracle,
but it also exposes callable one-pass row/column sums so a matrix-free
factor implementation can be compared without changing the metric contract.
No truth, morphology, formal runner, or performance result is involved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray
Forward = Callable[[Array], Array]


def _f64(value: Array, *, name: str, ndim: int | None = None) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {result.ndim}-D")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite float64")
    return result


def _nonnegative(value: Array, *, name: str) -> Array:
    result = _f64(value, name=name, ndim=2)
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be elementwise nonnegative")
    return result


def _positive_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class SignedFactorSystem:
    """Explicit signed factor system before activity elimination.

    ``D_plus`` has rows ordered as ``(site_0/x, site_0/y, site_0/z, ...)``.
    Every covariance-connected data block gets one ``P`` and one ``W``.
    """

    E: Array
    P_blocks: tuple[Array, ...]
    G: Array
    W_blocks: tuple[Array, ...]
    D_plus: Array
    tv_site_count: int | None = None

    def __post_init__(self) -> None:
        E = _nonnegative(self.E, name="E")
        G = _f64(self.G, name="G", ndim=2)
        D = _f64(self.D_plus, name="D_plus", ndim=2)
        P_blocks = tuple(_nonnegative(item, name=f"P_blocks[{i}]") for i, item in enumerate(self.P_blocks))
        W_blocks = tuple(_f64(item, name=f"W_blocks[{i}]", ndim=2) for i, item in enumerate(self.W_blocks))
        if not P_blocks or len(P_blocks) != len(W_blocks):
            raise ValueError("P_blocks and W_blocks must be non-empty and paired")
        if G.shape[1] != E.shape[0]:
            raise ValueError("G columns must match E rows")
        if D.shape[1] != E.shape[0]:
            raise ValueError("D_plus columns must match E rows")
        for index, (P, W) in enumerate(zip(P_blocks, W_blocks)):
            if P.shape[1] != G.shape[0]:
                raise ValueError(f"P_blocks[{index}] columns must match G rows")
            if W.shape[1] != P.shape[0]:
                raise ValueError(f"W_blocks[{index}] columns must match P rows")
        if D.shape[0] % 3:
            raise ValueError("D_plus rows must be a multiple of three")
        site_count = D.shape[0] // 3
        if self.tv_site_count is not None and int(self.tv_site_count) != site_count:
            raise ValueError("tv_site_count does not match D_plus")
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "G", G)
        object.__setattr__(self, "D_plus", D)
        object.__setattr__(self, "P_blocks", P_blocks)
        object.__setattr__(self, "W_blocks", W_blocks)
        object.__setattr__(self, "tv_site_count", site_count)

    @property
    def full_primal_count(self) -> int:
        return int(self.E.shape[1])

    def signed_data(self, block: int, x: Array) -> Array:
        return self.W_blocks[block] @ self.P_blocks[block] @ self.G @ self.E @ x

    def signed_data_adjoint(self, block: int, y: Array) -> Array:
        return self.E.T @ self.G.T @ self.P_blocks[block].T @ self.W_blocks[block].T @ y

    def signed_tv(self, x: Array) -> Array:
        return (self.D_plus @ self.E @ x).reshape(int(self.tv_site_count), 3)

    def signed_tv_adjoint(self, q: Array) -> Array:
        return self.E.T @ self.D_plus.T @ np.asarray(q, dtype=np.float64).reshape(-1)

    def dense_majorizer_data(self, block: int) -> Array:
        return np.abs(self.W_blocks[block]) @ self.P_blocks[block] @ np.abs(self.G) @ self.E

    def dense_majorizer_tv(self) -> Array:
        return np.abs(self.D_plus) @ self.E


@dataclass(frozen=True)
class CallableLinearMap:
    """A shape-declared callable map used by the one-pass audit."""

    rows: int
    cols: int
    forward: Forward
    transpose: Forward


def ones_pass(
    forward: Forward,
    transpose: Forward,
    *,
    rows: int,
    cols: int,
) -> tuple[Array, Array]:
    """Return row and column sums with exactly one call in each direction."""

    rows = int(rows)
    cols = int(cols)
    if rows < 0 or cols < 0:
        raise ValueError("map dimensions must be nonnegative")
    row_sums = _f64(forward(np.ones(cols, dtype=np.float64)), name="ones forward", ndim=1)
    col_sums = _f64(transpose(np.ones(rows, dtype=np.float64)), name="ones transpose", ndim=1)
    if row_sums.shape != (rows,) or col_sums.shape != (cols,):
        raise ValueError("callable ones pass returned an unexpected shape")
    if np.any(row_sums < -1e-14) or np.any(col_sums < -1e-14):
        raise ValueError("absolute-factor ones pass must be nonnegative")
    return np.maximum(row_sums, 0.0), np.maximum(col_sums, 0.0)


def _absolute_data_maps(system: SignedFactorSystem, block: int) -> CallableLinearMap:
    P = system.P_blocks[block]
    abs_W = np.abs(system.W_blocks[block])
    abs_G = np.abs(system.G)
    E = system.E

    def forward(x: Array) -> Array:
        return abs_W @ P @ abs_G @ E @ x

    def transpose(y: Array) -> Array:
        return E.T @ abs_G.T @ P.T @ abs_W.T @ y

    return CallableLinearMap(abs_W.shape[0], E.shape[1], forward, transpose)


def _absolute_tv_map(system: SignedFactorSystem) -> CallableLinearMap:
    abs_D = np.abs(system.D_plus)
    E = system.E

    def forward(x: Array) -> Array:
        return abs_D @ E @ x

    def transpose(y: Array) -> Array:
        return E.T @ abs_D.T @ y

    return CallableLinearMap(abs_D.shape[0], E.shape[1], forward, transpose)


def dominance_gap(actual: Array, majorizer: Array) -> float:
    """Return ``min(majorizer - abs(actual))`` for a dense audit."""

    actual = _f64(actual, name="actual", ndim=2)
    majorizer = _f64(majorizer, name="majorizer", ndim=2)
    if actual.shape != majorizer.shape:
        raise ValueError("actual and majorizer shapes differ")
    return float(np.min(majorizer - np.abs(actual))) if actual.size else 0.0


def assert_dominance(actual: Array, majorizer: Array, *, tolerance: float = 1e-12) -> None:
    gap = dominance_gap(actual, majorizer)
    scale = max(1.0, float(np.max(np.abs(actual))) if actual.size else 0.0)
    if gap < -float(tolerance) * scale:
        raise ValueError(f"factor majorizer does not dominate signed operator (gap={gap})")


@dataclass(frozen=True)
class MajorizerSetup:
    """Activity-reduced metric and dense audit matrices."""

    system: SignedFactorSystem
    eta: float
    use_callable_ones_pass: bool
    A_blocks: tuple[Array, ...]
    D: Array
    M_blocks: tuple[Array, ...]
    N: Array
    omega_data_rows: tuple[Array, ...]
    omega_data_cols: tuple[Array, ...]
    omega_tv_rows: Array
    omega_tv_cols: Array
    omega_columns: Array
    rho_data: tuple[float, ...]
    rho_tv_sites: Array
    sigma_data: tuple[float | None, ...]
    sigma_tv_sites: Array
    tau: Array
    active_primal: Array
    active_data_rows: tuple[Array, ...]
    active_tv_sites: Array
    _A_active: tuple[Array, ...]
    _D_active: Array

    @property
    def active_primal_count(self) -> int:
        return int(self.active_primal.size)

    @property
    def active_tv_site_count(self) -> int:
        return int(np.count_nonzero(self.active_tv_sites))

    @property
    def K_active(self) -> Array:
        parts = [matrix for matrix in self._A_active if matrix.shape[0]]
        if self._D_active.shape[0]:
            parts.append(self._D_active.reshape(-1, self.active_primal_count))
        return np.vstack(parts) if parts else np.zeros((0, self.active_primal_count), dtype=np.float64)

    @property
    def sigma_vector(self) -> Array:
        values: list[float] = []
        for rows, sigma in zip(self.active_data_rows, self.sigma_data):
            if np.any(rows):
                if sigma is None:
                    raise AssertionError("active data block lacks sigma")
                values.extend([sigma] * int(np.count_nonzero(rows)))
        for site, active in enumerate(self.active_tv_sites):
            if active:
                values.extend([float(self.sigma_tv_sites[site])] * 3)
        return np.asarray(values, dtype=np.float64)

    def data_forward(self, x: Array) -> tuple[Array, ...]:
        x = _f64(x, name="active primal", ndim=1)
        if x.shape != (self.active_primal_count,):
            raise ValueError("active primal has the wrong size")
        return tuple(matrix @ x for matrix in self._A_active)

    def data_adjoint(self, values: Sequence[Array]) -> Array:
        if len(values) != len(self._A_active):
            raise ValueError("data block count mismatch")
        result = np.zeros(self.active_primal_count, dtype=np.float64)
        for matrix, value in zip(self._A_active, values):
            value = _f64(value, name="data dual", ndim=1)
            if value.shape != (matrix.shape[0],):
                raise ValueError("data dual shape mismatch")
            result += matrix.T @ value
        return result

    def tv_forward(self, x: Array) -> Array:
        x = _f64(x, name="active primal", ndim=1)
        return self._D_active @ x

    def tv_adjoint(self, q: Array) -> Array:
        q = _f64(q, name="TV dual", ndim=2)
        if q.shape != (self.active_tv_site_count, 3):
            raise ValueError("TV dual shape mismatch")
        return self._D_active.reshape(-1, self.active_primal_count).T @ q.reshape(-1)


def build_majorizer_setup(
    system: SignedFactorSystem,
    *,
    eta: float = 0.8,
    use_callable_ones_pass: bool = False,
    zero_tolerance: float = 0.0,
) -> MajorizerSetup:
    """Build the signed-factor metric using exact-zero activity only."""

    eta = _positive_scalar(eta, name="eta")
    if eta >= 1.0:
        raise ValueError("eta must be strictly less than one")
    zero_tolerance = float(zero_tolerance)
    if not math.isfinite(zero_tolerance) or zero_tolerance != 0.0:
        raise ValueError("zero_tolerance must be exactly zero")

    A_blocks = tuple(
        system.W_blocks[i] @ system.P_blocks[i] @ system.G @ system.E
        for i in range(len(system.P_blocks))
    )
    D = system.D_plus @ system.E
    M_blocks = tuple(system.dense_majorizer_data(i) for i in range(len(system.P_blocks)))
    N = system.dense_majorizer_tv()
    for A, M in zip(A_blocks, M_blocks):
        if np.any(M < 0.0):
            raise ValueError("M must be elementwise nonnegative")
        assert_dominance(A, M)
    if np.any(N < 0.0):
        raise ValueError("N must be elementwise nonnegative")
    assert_dominance(D, N)

    omega_data_rows: list[Array] = []
    omega_data_cols: list[Array] = []
    for block, M in enumerate(M_blocks):
        if use_callable_ones_pass:
            row, col = ones_pass(
                _absolute_data_maps(system, block).forward,
                _absolute_data_maps(system, block).transpose,
                rows=M.shape[0],
                cols=M.shape[1],
            )
        else:
            row, col = M.sum(axis=1), M.sum(axis=0)
        omega_data_rows.append(row)
        omega_data_cols.append(col)
    if use_callable_ones_pass:
        tv_map = _absolute_tv_map(system)
        omega_tv_rows, omega_tv_cols = ones_pass(
            tv_map.forward,
            tv_map.transpose,
            rows=N.shape[0],
            cols=N.shape[1],
        )
    else:
        omega_tv_rows, omega_tv_cols = N.sum(axis=1), N.sum(axis=0)

    active_data_rows = tuple(row > 0.0 for row in omega_data_rows)
    active_tv_sites = omega_tv_rows.reshape(-1, 3).max(axis=1) > 0.0
    omega_columns = sum(omega_data_cols, start=np.zeros(system.full_primal_count)) + omega_tv_cols
    active_primal = np.flatnonzero(omega_columns > 0.0)
    if active_primal.size == 0:
        raise ValueError("no active primal columns remain after zero-coupling elimination")

    rho_data = tuple(float(row[mask].max()) if np.any(mask) else 0.0 for row, mask in zip(omega_data_rows, active_data_rows))
    rho_tv_sites = omega_tv_rows.reshape(-1, 3).max(axis=1)
    sigma_data = tuple(eta / rho if rho > 0.0 else None for rho in rho_data)
    sigma_tv_sites = np.zeros(int(system.tv_site_count), dtype=np.float64)
    for site, active in enumerate(active_tv_sites):
        if active:
            sigma_tv_sites[site] = eta / float(rho_tv_sites[site])
    tau = eta / omega_columns[active_primal]

    A_active = tuple(A[mask][:, active_primal] for A, mask in zip(A_blocks, active_data_rows))
    D_sites = D.reshape(-1, 3, system.full_primal_count)
    D_active = D_sites[active_tv_sites][:, :, active_primal]
    setup = MajorizerSetup(
        system=system,
        eta=eta,
        use_callable_ones_pass=bool(use_callable_ones_pass),
        A_blocks=A_blocks,
        D=D,
        M_blocks=M_blocks,
        N=N,
        omega_data_rows=tuple(omega_data_rows),
        omega_data_cols=tuple(omega_data_cols),
        omega_tv_rows=omega_tv_rows,
        omega_tv_cols=omega_tv_cols,
        omega_columns=omega_columns,
        rho_data=rho_data,
        rho_tv_sites=rho_tv_sites,
        sigma_data=sigma_data,
        sigma_tv_sites=sigma_tv_sites,
        tau=tau,
        active_primal=active_primal,
        active_data_rows=active_data_rows,
        active_tv_sites=active_tv_sites,
        _A_active=A_active,
        _D_active=D_active,
    )
    assert_exact_svd_safety(setup)
    return setup


def exact_svd_squared_norm(
    K: Array,
    sigma: Array,
    tau: Array,
) -> float:
    """Compute the exact tiny dense ``||sqrt(Sigma) K sqrt(T)||_2^2``."""

    K = _f64(K, name="K", ndim=2)
    sigma = _f64(sigma, name="sigma", ndim=1)
    tau = _f64(tau, name="tau", ndim=1)
    if K.shape != (sigma.size, tau.size):
        raise ValueError("K and metric dimensions differ")
    if np.any(sigma <= 0.0) or np.any(tau <= 0.0):
        raise ValueError("exact SVD metric must be strictly positive on active space")
    scaled = np.sqrt(sigma)[:, None] * K * np.sqrt(tau)[None, :]
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    return float(singular_values[0] ** 2) if singular_values.size else 0.0


def assert_exact_svd_safety(
    setup: MajorizerSetup,
    *,
    sigma: Array | None = None,
    tau: Array | None = None,
    tolerance: float = 1e-12,
) -> float:
    sigma_value = setup.sigma_vector if sigma is None else _f64(sigma, name="sigma", ndim=1)
    tau_value = setup.tau if tau is None else _f64(tau, name="tau", ndim=1)
    observed = exact_svd_squared_norm(setup.K_active, sigma_value, tau_value)
    if observed > setup.eta**2 + float(tolerance):
        raise ValueError(f"exact SVD safety check failed: {observed} > {setup.eta**2}")
    return observed


def _active_targets(setup: MajorizerSetup, targets: Sequence[Array]) -> tuple[Array, ...]:
    if len(targets) != len(setup._A_active):
        raise ValueError("data target block count mismatch")
    result = []
    for target, mask in zip(targets, setup.active_data_rows):
        target = _f64(target, name="data target", ndim=1)
        if target.shape[0] != int(np.count_nonzero(mask)):
            raise ValueError("data target must already be activity-reduced")
        result.append(target)
    return tuple(result)


@dataclass(frozen=True)
class PDHGState:
    x: Array
    x_bar: Array
    data_dual: tuple[Array, ...]
    tv_dual: Array


def pdhg_step(
    setup: MajorizerSetup,
    state: PDHGState,
    targets: Sequence[Array],
    *,
    regularization_weight: float = 0.0,
    penalty: str = "tv",
    huber_delta: float = 0.4,
    theta: float = 1.0,
) -> PDHGState:
    """Perform one camera-block/site-shared CPU float64 PDHG update."""

    x = _f64(state.x, name="x", ndim=1)
    x_bar = _f64(state.x_bar, name="x_bar", ndim=1)
    if x.shape != setup.tau.shape or x_bar.shape != x.shape:
        raise ValueError("PDHG primal state shape mismatch")
    if len(state.data_dual) != len(setup._A_active):
        raise ValueError("PDHG data dual block count mismatch")
    if state.tv_dual.shape != (setup.active_tv_site_count, 3):
        raise ValueError("PDHG TV dual shape mismatch")
    if penalty not in {"tv", "huber"}:
        raise ValueError("penalty must be 'tv' or 'huber'")
    weight = float(regularization_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("regularization_weight must be finite and nonnegative")
    if penalty == "huber" and (not math.isfinite(huber_delta) or huber_delta <= 0.0):
        raise ValueError("huber_delta must be finite and positive")
    if not 0.0 <= float(theta) <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    active_targets = _active_targets(setup, targets)

    data_dual: list[Array] = []
    data_values = setup.data_forward(x_bar)
    for index, (old, value, target, mask, sigma) in enumerate(
        zip(state.data_dual, data_values, active_targets, setup.active_data_rows, setup.sigma_data)
    ):
        old = _f64(old, name=f"data_dual[{index}]", ndim=1)
        if old.shape != value.shape:
            raise ValueError("data dual shape mismatch")
        if sigma is None:
            if old.size:
                raise ValueError("inactive data block cannot carry active dual values")
            data_dual.append(np.zeros(0, dtype=np.float64))
        else:
            data_dual.append((old + sigma * (value - target)) / (1.0 + sigma))

    if weight == 0.0 or setup.active_tv_site_count == 0:
        tv_dual = np.zeros((setup.active_tv_site_count, 3), dtype=np.float64)
    else:
        old_tv = _f64(state.tv_dual, name="tv_dual", ndim=2)
        tv_values = setup.tv_forward(x_bar).reshape(setup.active_tv_site_count, 3)
        tv_dual_values = old_tv + setup.sigma_tv_sites[setup.active_tv_sites, None] * tv_values
        if penalty == "tv":
            norms = np.linalg.norm(tv_dual_values, axis=1)
            factors = np.maximum(1.0, norms / weight)
            tv_dual = tv_dual_values / factors[:, None]
        else:
            scaled = tv_dual_values / (1.0 + setup.sigma_tv_sites[setup.active_tv_sites, None] * huber_delta / weight)
            norms = np.linalg.norm(scaled, axis=1)
            factors = np.maximum(1.0, norms / weight)
            tv_dual = scaled / factors[:, None]

    gradient = setup.data_adjoint(data_dual)
    if weight != 0.0 and setup.active_tv_site_count:
        gradient += setup.tv_adjoint(tv_dual)
    next_x = x - setup.tau * gradient
    next_x_bar = next_x + float(theta) * (next_x - x)
    return PDHGState(next_x, next_x_bar, tuple(data_dual), tv_dual)


def run_pdhg(
    setup: MajorizerSetup,
    targets: Sequence[Array],
    *,
    iterations: int = 1,
    regularization_weight: float = 0.0,
    penalty: str = "tv",
    huber_delta: float = 0.4,
    theta: float = 1.0,
    initial: PDHGState | None = None,
) -> tuple[PDHGState, ...]:
    """Run a fixed one- or six-step recurrence for dense/callable comparison."""

    iterations = int(iterations)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if initial is None:
        initial = PDHGState(
            np.zeros_like(setup.tau),
            np.zeros_like(setup.tau),
            tuple(np.zeros(int(np.count_nonzero(mask)), dtype=np.float64) for mask in setup.active_data_rows),
            np.zeros((setup.active_tv_site_count, 3), dtype=np.float64),
        )
    states: list[PDHGState] = []
    current = initial
    for _ in range(iterations):
        current = pdhg_step(
            setup,
            current,
            targets,
            regularization_weight=regularization_weight,
            penalty=penalty,
            huber_delta=huber_delta,
            theta=theta,
        )
        states.append(current)
    return tuple(states)


def pdhg_objective(
    setup: MajorizerSetup,
    state: PDHGState,
    targets: Sequence[Array],
    *,
    regularization_weight: float = 0.0,
    penalty: str = "tv",
    huber_delta: float = 0.4,
) -> float:
    """Evaluate the quadratic plus isotropic TV/Huber objective.

    ``penalty="tv"`` preserves the historical API.  The Huber branch uses
    ``r**2 / (2 delta)`` for ``r <= delta`` and ``r - delta/2`` otherwise,
    matching the conjugate proximal recurrence in :func:`pdhg_step`.
    """

    if penalty not in {"tv", "huber"}:
        raise ValueError("penalty must be 'tv' or 'huber'")
    delta = float(huber_delta)
    if penalty == "huber" and (not math.isfinite(delta) or delta <= 0.0):
        raise ValueError("huber_delta must be finite and positive")

    residual_energy = 0.0
    for value, target in zip(setup.data_forward(state.x), _active_targets(setup, targets)):
        residual_energy += float(np.dot(value - target, value - target))
    value = 0.5 * residual_energy
    if regularization_weight:
        magnitudes = np.linalg.norm(
            setup.tv_forward(state.x).reshape(-1, 3), axis=1
        )
        if penalty == "huber":
            edge_values = np.where(
                magnitudes <= delta,
                0.5 * magnitudes**2 / delta,
                magnitudes - 0.5 * delta,
            )
        else:
            edge_values = magnitudes
        value += float(regularization_weight) * float(edge_values.sum())
    return value


__all__ = [
    "CallableLinearMap",
    "MajorizerSetup",
    "PDHGState",
    "SignedFactorSystem",
    "assert_dominance",
    "assert_exact_svd_safety",
    "build_majorizer_setup",
    "dominance_gap",
    "exact_svd_squared_norm",
    "ones_pass",
    "pdhg_objective",
    "pdhg_step",
    "run_pdhg",
]
