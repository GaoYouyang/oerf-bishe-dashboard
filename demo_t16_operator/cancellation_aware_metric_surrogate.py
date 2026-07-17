"""CPU-sized, audit-oriented cancellation-aware metric surrogate smoke.

This module keeps deployment-visible inference inputs separate from synthetic
teachers.  It is development evidence only: exact masses and truths may be used
for training, calibration, comparator construction, and offline evaluation, but
the inference API cannot receive them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator

import torch
from torch import nn


SCHEMA_VERSION = "cancellation-aware-metric-surrogate-smoke-2.0"
STATUS = "DEVELOPMENT_ONLY_SYNTHETIC_INTERFACE_SMOKE"
SPLIT_ROLES = ("train", "safety_calibration", "fresh_geometry_ood")


@dataclass(frozen=True)
class TinySignedRig:
    """Complete synthetic rig, including teachers used outside inference."""

    rig_id: str
    split_role: str
    geometry_seed: int
    noise_seed: int
    geometry_parameters: tuple[float, float, float, float]
    geometry_parameters_sha256: str
    geometry_features: torch.Tensor
    row_features: torch.Tensor
    column_features: torch.Tensor
    signed_matrix: torch.Tensor
    factor_majorizer: torch.Tensor
    exact_row_mass: torch.Tensor
    exact_column_mass: torch.Tensor
    factor_row_mass: torch.Tensor
    factor_column_mass: torch.Tensor
    truth: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class InferenceRigFeatures:
    """Deployment-visible features; intentionally excludes A, exact, truth, target."""

    rig_id: str
    row_features: torch.Tensor
    column_features: torch.Tensor


@dataclass
class ExactMassInstrumentation:
    """Mutable access ledger used to enforce fresh-rig phase boundaries."""

    scopes: list[dict[str, Any]]
    accesses: list[dict[str, Any]]

    @classmethod
    def empty(cls) -> "ExactMassInstrumentation":
        return cls(scopes=[], accesses=[])

    def summary(self) -> dict[str, Any]:
        candidate_phases = {
            "factor_setup",
            "scalar_factor_setup",
            "learned_prediction_setup",
            "calibrated_envelope_prediction_setup",
        }
        candidate_fresh = [
            event
            for event in self.accesses
            if event["split_role"] == "fresh_geometry_ood"
            and event["phase"] in candidate_phases
        ]
        scope_phases = sorted({scope["phase"] for scope in self.scopes})
        return {
            "guard_scope_count": len(self.scopes),
            "guard_scope_count_by_phase": {
                phase: sum(
                    int(scope["phase"] == phase) for scope in self.scopes
                )
                for phase in scope_phases
            },
            "fresh_guarded_candidate_scope_count": sum(
                int(
                    scope["split_role"] == "fresh_geometry_ood"
                    and scope["phase"] in candidate_phases
                    and not scope["fresh_access_allowed"]
                )
                for scope in self.scopes
            ),
            "exact_mass_access_count": len(self.accesses),
            "blocked_exact_mass_access_count": sum(
                int(not event["allowed"]) for event in self.accesses
            ),
            "fresh_candidate_exact_mass_access_count": len(candidate_fresh),
            "fresh_candidate_exact_access": bool(candidate_fresh),
            "allowed_access_count_by_phase": {
                phase: sum(
                    int(event["allowed"] and event["phase"] == phase)
                    for event in self.accesses
                )
                for phase in sorted({event["phase"] for event in self.accesses})
            },
        }


@dataclass(frozen=True)
class _ExactMassScope:
    instrumentation: ExactMassInstrumentation
    rig_id: str
    split_role: str
    phase: str
    fresh_access_allowed: bool


_EXACT_MASS_SCOPE: ContextVar[_ExactMassScope | None] = ContextVar(
    "metric_a_exact_mass_scope", default=None
)


@contextmanager
def exact_mass_access_scope(
    instrumentation: ExactMassInstrumentation,
    *,
    rig_id: str,
    split_role: str,
    phase: str,
    fresh_access_allowed: bool,
) -> Iterator[None]:
    """Instrument exact-mass calls and fail immediately in guarded fresh phases."""

    scope = _ExactMassScope(
        instrumentation=instrumentation,
        rig_id=str(rig_id),
        split_role=str(split_role),
        phase=str(phase),
        fresh_access_allowed=bool(fresh_access_allowed),
    )
    instrumentation.scopes.append(
        {
            "rig_id": scope.rig_id,
            "split_role": scope.split_role,
            "phase": scope.phase,
            "fresh_access_allowed": scope.fresh_access_allowed,
        }
    )
    token = _EXACT_MASS_SCOPE.set(scope)
    try:
        yield
    finally:
        _EXACT_MASS_SCOPE.reset(token)


@dataclass(frozen=True)
class CalibrationEnvelope:
    """Global multiplicative envelope fitted with exact calibration teachers."""

    row_scale: float
    column_scale: float
    calibration_rig_ids: tuple[str, ...]
    exact_mass_materializations: int
    factor_mass_vector_accesses: int
    factor_feature_construction_calls: int
    estimator_head_forward_calls: int


@dataclass(frozen=True)
class DiagonalMetric:
    """Positive diagonal primal/dual metric for one signed matrix."""

    row_mass: torch.Tensor
    column_mass: torch.Tensor
    sigma: torch.Tensor
    tau: torch.Tensor
    eta: float


@dataclass(frozen=True)
class TrajectoryResult:
    """Residual/field checkpoints and explicit signed-operator calls."""

    rows: list[dict[str, float | int]]
    ledger: dict[str, int]


def _require_finite_positive(name: str, value: torch.Tensor) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError(f"{name} must be finite")
    if not bool(torch.all(value > 0.0)):
        raise ValueError(f"{name} must be strictly positive")


def stable_rig_seed(
    *, base_seed: int, rig_id: str, split_role: str
) -> int:
    """Derive a stable CPU seed from identifiers, independent of collection order."""

    if int(base_seed) < 0 or not rig_id or split_role not in SPLIT_ROLES:
        raise ValueError("stable seed inputs are invalid")
    payload = json.dumps(
        [int(base_seed), str(rig_id), str(split_role)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def geometry_parameters_sha256(
    *, rig_id: str, split_role: str, parameters: Sequence[float]
) -> str:
    """Hash one rig's identifiers and exact float64 geometry serialization."""

    payload = json.dumps(
        {
            "rig_id": str(rig_id),
            "split_role": str(split_role),
            "parameters_hex": [float(value).hex() for value in parameters],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sample_geometry(
    *, generator: torch.Generator, split_role: str, dtype: torch.dtype
) -> tuple[float, float, float, float]:
    """Sample geometry coordinates independently, with a held-out OOD support."""

    if split_role not in SPLIT_ROLES:
        raise ValueError(f"unknown split role: {split_role}")

    def uniform(low: float, high: float) -> float:
        draw = float(torch.rand((), generator=generator, dtype=dtype))
        return low + (high - low) * draw

    if split_role == "fresh_geometry_ood":
        # Each parameter is independently sampled outside at least one central
        # training interval; this is a synthetic stress split, not real OOD proof.
        angle = uniform(0.95, 1.28) * (-1.0 if uniform(0.0, 1.0) < 0.5 else 1.0)
        aperture = uniform(0.48, 0.64) if uniform(0.0, 1.0) < 0.5 else uniform(1.02, 1.16)
        shear = uniform(0.42, 0.62) * (-1.0 if uniform(0.0, 1.0) < 0.5 else 1.0)
        cancellation = uniform(0.42, 0.58) if uniform(0.0, 1.0) < 0.5 else uniform(0.96, 1.08)
    else:
        angle = uniform(-0.82, 0.82)
        aperture = uniform(0.66, 0.98)
        shear = uniform(-0.36, 0.36)
        cancellation = uniform(0.62, 0.94)
    return angle, aperture, shear, cancellation


def generate_tiny_rigs(
    *,
    split_assignments: Mapping[str, str],
    geometry_seed: int,
    noise_seed: int,
    row_count: int,
    column_count: int,
    dtype: torch.dtype = torch.float64,
) -> list[TinySignedRig]:
    """Generate independently sampled synthetic signed rigs.

    Geometry and noise use separate generators.  ``A = P1 - P2 + P3`` and
    ``M = |P1| + |P2| + |P3|`` preserve a triangle-inequality factor bound.
    """

    if not split_assignments:
        raise ValueError("at least one split assignment is required")
    if len(split_assignments) != len(set(split_assignments)):
        raise ValueError("rig ids must be unique")
    if set(split_assignments.values()).difference(SPLIT_ROLES):
        raise ValueError("split assignments contain an unknown role")
    rows_n, columns_n = int(row_count), int(column_count)
    if rows_n < 3 or columns_n < 3:
        raise ValueError("matrix dimensions must be at least three")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("tiny rigs require a floating-point CPU dtype")

    row_coordinate = torch.linspace(-1.0, 1.0, rows_n, dtype=dtype)
    column_coordinate = torch.linspace(-1.0, 1.0, columns_n, dtype=dtype)
    row_grid = row_coordinate[:, None]
    column_grid = column_coordinate[None, :]
    output: list[TinySignedRig] = []

    for rig_id, split_role in split_assignments.items():
        per_rig_geometry_seed = stable_rig_seed(
            base_seed=int(geometry_seed),
            rig_id=rig_id,
            split_role=split_role,
        )
        per_rig_noise_seed = stable_rig_seed(
            base_seed=int(noise_seed),
            rig_id=rig_id,
            split_role=split_role,
        )
        geometry_generator = torch.Generator(device="cpu").manual_seed(
            per_rig_geometry_seed
        )
        noise_generator = torch.Generator(device="cpu").manual_seed(per_rig_noise_seed)
        angle, aperture, shear, cancellation = _sample_geometry(
            generator=geometry_generator, split_role=split_role, dtype=dtype
        )
        geometry_parameters = (angle, aperture, shear, cancellation)
        parameter_hash = geometry_parameters_sha256(
            rig_id=rig_id,
            split_role=split_role,
            parameters=geometry_parameters,
        )
        jitter = 0.025 * torch.randn(
            (rows_n, columns_n), generator=noise_generator, dtype=dtype
        )

        center_one = 0.48 * row_grid * math.cos(angle) + 0.34 * shear
        center_two = 0.44 * row_grid * math.cos(angle + 0.37) - 0.22 * shear
        width_one = 0.27 + 0.11 * aperture
        width_two = 0.23 + 0.13 * (1.0 - 0.35 * aperture)
        positive_one = aperture * torch.exp(
            -0.5 * ((column_grid - center_one) / width_one).square()
        )
        positive_one *= 1.0 + 0.12 * torch.cos(
            math.pi * (row_grid - 0.35 * column_grid + angle)
        )
        positive_two = cancellation * torch.exp(
            -0.5 * ((column_grid - center_two) / width_two).square()
        )
        positive_two *= 1.0 + 0.10 * torch.sin(
            math.pi * (0.7 * row_grid + column_grid - shear)
        )
        signed_third = (0.055 + 0.015 * aperture) * torch.sin(
            math.pi * (1.3 * row_grid - 0.9 * column_grid + angle)
        ) + jitter

        signed_matrix = positive_one - positive_two + signed_third
        factor_majorizer = positive_one.abs() + positive_two.abs() + signed_third.abs()
        exact_row_mass = torch.sum(torch.abs(signed_matrix), dim=1)
        exact_column_mass = torch.sum(torch.abs(signed_matrix), dim=0)
        factor_row_mass = torch.sum(factor_majorizer, dim=1)
        factor_column_mass = torch.sum(factor_majorizer, dim=0)
        geometry = torch.tensor(
            [math.sin(angle), math.cos(angle), aperture, shear, cancellation],
            dtype=dtype,
        )
        row_features = torch.cat(
            (
                geometry[None].expand(rows_n, -1),
                row_coordinate[:, None],
                torch.log1p(factor_row_mass)[:, None],
            ),
            dim=1,
        )
        column_features = torch.cat(
            (
                geometry[None].expand(columns_n, -1),
                column_coordinate[:, None],
                torch.log1p(factor_column_mass)[:, None],
            ),
            dim=1,
        )
        truth = (
            0.25
            + torch.exp(-((column_coordinate - 0.28 * shear) / 0.34).square())
            + 0.12 * torch.cos(math.pi * (column_coordinate + angle))
        ).clamp_min(0.02)
        noise_scale = (
            0.003
            * torch.linalg.vector_norm(signed_matrix @ truth)
            / math.sqrt(rows_n)
        )
        target = signed_matrix @ truth + noise_scale * torch.randn(
            rows_n, generator=noise_generator, dtype=dtype
        )

        if not bool(torch.all(factor_majorizer >= torch.abs(signed_matrix))):
            raise RuntimeError("factor majorizer violated triangle inequality")
        if not bool(torch.any(signed_matrix < 0.0) and torch.any(signed_matrix > 0.0)):
            raise RuntimeError("synthetic signed matrix lost cancellation")
        output.append(
            TinySignedRig(
                rig_id=rig_id,
                split_role=split_role,
                geometry_seed=per_rig_geometry_seed,
                noise_seed=per_rig_noise_seed,
                geometry_parameters=geometry_parameters,
                geometry_parameters_sha256=parameter_hash,
                geometry_features=geometry,
                row_features=row_features,
                column_features=column_features,
                signed_matrix=signed_matrix,
                factor_majorizer=factor_majorizer,
                exact_row_mass=exact_row_mass,
                exact_column_mass=exact_column_mass,
                factor_row_mass=factor_row_mass,
                factor_column_mass=factor_column_mass,
                truth=truth,
                target=target,
            )
        )
    return output


def split_rigs_three_way(
    rigs: Sequence[TinySignedRig],
) -> tuple[list[TinySignedRig], list[TinySignedRig], list[TinySignedRig], dict[str, Any]]:
    """Return complete-rig train/calibration/fresh-OOD partitions."""

    if not rigs or len({rig.rig_id for rig in rigs}) != len(rigs):
        raise ValueError("rigs must be nonempty with unique ids")
    grouped = {role: [rig for rig in rigs if rig.split_role == role] for role in SPLIT_ROLES}
    if any(not grouped[role] for role in SPLIT_ROLES):
        raise ValueError("all three split roles require at least one complete rig")
    contract = {
        "split_unit": "COMPLETE_RIG",
        "random_ray_split_used": False,
        "geometry_parameters_independently_sampled": True,
        "noise_seed_is_not_geometry_seed": True,
        "train_rig_ids": [rig.rig_id for rig in grouped["train"]],
        "safety_calibration_rig_ids": [
            rig.rig_id for rig in grouped["safety_calibration"]
        ],
        "fresh_geometry_ood_rig_ids": [
            rig.rig_id for rig in grouped["fresh_geometry_ood"]
        ],
    }
    return (
        grouped["train"],
        grouped["safety_calibration"],
        grouped["fresh_geometry_ood"],
        contract,
    )


def inference_features_from_rig(rig: TinySignedRig) -> InferenceRigFeatures:
    """Construct deployment features, explicitly consuming factor mass vectors."""

    return InferenceRigFeatures(
        rig_id=rig.rig_id,
        row_features=torch.cat(
            (
                rig.row_features[:, :-1].clone(),
                torch.log1p(rig.factor_row_mass)[:, None],
            ),
            dim=1,
        ),
        column_features=torch.cat(
            (
                rig.column_features[:, :-1].clone(),
                torch.log1p(rig.factor_column_mass)[:, None],
            ),
            dim=1,
        ),
    )


class GeometryConditionedPositiveEstimator(nn.Module):
    """Two tiny MLP heads that emit strictly positive row/column masses."""

    def __init__(
        self,
        *,
        row_input_dim: int,
        column_input_dim: int,
        hidden_dim: int,
        row_feature_mean: torch.Tensor,
        row_feature_scale: torch.Tensor,
        column_feature_mean: torch.Tensor,
        column_feature_scale: torch.Tensor,
        row_log_target_mean: torch.Tensor,
        row_log_target_scale: torch.Tensor,
        column_log_target_mean: torch.Tensor,
        column_log_target_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        if hidden < 2:
            raise ValueError("hidden_dim must be at least two")
        self.row_head = nn.Sequential(nn.Linear(row_input_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.column_head = nn.Sequential(nn.Linear(column_input_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("row_feature_mean", row_feature_mean.clone())
        self.register_buffer("row_feature_scale", row_feature_scale.clone())
        self.register_buffer("column_feature_mean", column_feature_mean.clone())
        self.register_buffer("column_feature_scale", column_feature_scale.clone())
        self.register_buffer("row_log_target_mean", row_log_target_mean.reshape(()).clone())
        self.register_buffer("row_log_target_scale", row_log_target_scale.reshape(()).clone())
        self.register_buffer("column_log_target_mean", column_log_target_mean.reshape(()).clone())
        self.register_buffer("column_log_target_scale", column_log_target_scale.reshape(()).clone())

    def _positive(
        self,
        features: torch.Tensor,
        *,
        head: nn.Module,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (features - feature_mean) / feature_scale
        standardized_log = head(normalized).squeeze(-1)
        return torch.exp(torch.clamp(target_mean + target_scale * standardized_log, -20.0, 20.0))

    def predict_rows(self, features: torch.Tensor) -> torch.Tensor:
        return self._positive(features, head=self.row_head, feature_mean=self.row_feature_mean, feature_scale=self.row_feature_scale, target_mean=self.row_log_target_mean, target_scale=self.row_log_target_scale)

    def predict_columns(self, features: torch.Tensor) -> torch.Tensor:
        return self._positive(features, head=self.column_head, feature_mean=self.column_feature_mean, feature_scale=self.column_feature_scale, target_mean=self.column_log_target_mean, target_scale=self.column_log_target_scale)


def _standardization(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.mean(value, dim=0), torch.std(value, dim=0, unbiased=False).clamp_min(1e-6)


def fit_metric_surrogate(
    train_rigs: Sequence[TinySignedRig],
    *,
    hidden_dim: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[GeometryConditionedPositiveEstimator, dict[str, float | int]]:
    """Fit exact-mass teachers from complete training rigs only."""

    if not train_rigs or any(rig.split_role != "train" for rig in train_rigs):
        raise ValueError("training requires complete train-role rigs only")
    if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("steps and learning_rate must be positive")
    row_features = torch.cat([rig.row_features for rig in train_rigs])
    column_features = torch.cat([rig.column_features for rig in train_rigs])
    row_log_target = torch.log(torch.cat([rig.exact_row_mass for rig in train_rigs]).clamp_min(1e-12))
    column_log_target = torch.log(torch.cat([rig.exact_column_mass for rig in train_rigs]).clamp_min(1e-12))
    row_mean, row_scale = _standardization(row_features)
    column_mean, column_scale = _standardization(column_features)
    row_target_mean = torch.mean(row_log_target)
    row_target_scale = torch.std(row_log_target, unbiased=False).clamp_min(0.1)
    column_target_mean = torch.mean(column_log_target)
    column_target_scale = torch.std(column_log_target, unbiased=False).clamp_min(0.1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        estimator = GeometryConditionedPositiveEstimator(
            row_input_dim=row_features.shape[1],
            column_input_dim=column_features.shape[1],
            hidden_dim=hidden_dim,
            row_feature_mean=row_mean,
            row_feature_scale=row_scale,
            column_feature_mean=column_mean,
            column_feature_scale=column_scale,
            row_log_target_mean=row_target_mean,
            row_log_target_scale=row_target_scale,
            column_log_target_mean=column_target_mean,
            column_log_target_scale=column_target_scale,
        ).to(dtype=row_features.dtype, device="cpu")
        optimizer = torch.optim.Adam(estimator.parameters(), lr=float(learning_rate))
        huber = nn.HuberLoss(delta=0.2)

        def objective() -> torch.Tensor:
            row_prediction = torch.log(estimator.predict_rows(row_features))
            column_prediction = torch.log(estimator.predict_columns(column_features))
            fit = huber(row_prediction, row_log_target) + huber(column_prediction, column_log_target)
            under = torch.relu(row_log_target - row_prediction).square().mean()
            under += torch.relu(column_log_target - column_prediction).square().mean()
            return fit + 0.15 * under

        estimator.train()
        with torch.no_grad():
            initial_loss = float(objective())
        minimum_loss = initial_loss
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            optimizer.step()
            minimum_loss = min(minimum_loss, float(loss.detach()))
        estimator.eval()
        with torch.no_grad():
            final_loss = float(objective())
        minimum_loss = min(minimum_loss, final_loss)

    return estimator, {
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "minimum_training_loss": minimum_loss,
        "parameter_count": int(sum(p.numel() for p in estimator.parameters())),
        "full_batch_head_forward_calls": 2 * (int(steps) + 2),
        "exact_teacher_rig_count": len(train_rigs),
        "seed": int(seed),
    }


def predict_metric_masses(
    estimator: GeometryConditionedPositiveEstimator,
    features: InferenceRigFeatures,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict from the deployment-only dataclass; a complete rig is rejected."""

    if not isinstance(features, InferenceRigFeatures):
        raise TypeError("prediction requires InferenceRigFeatures")
    estimator.eval()
    with torch.no_grad():
        row = estimator.predict_rows(features.row_features)
        column = estimator.predict_columns(features.column_features)
    _require_finite_positive("predicted row mass", row)
    _require_finite_positive("predicted column mass", column)
    return row, column


def fit_calibration_envelope(
    estimator: GeometryConditionedPositiveEstimator,
    calibration_rigs: Sequence[TinySignedRig],
    *,
    margin: float,
    instrumentation: ExactMassInstrumentation,
) -> CalibrationEnvelope:
    """Fit one global safety envelope using exact calibration masses only."""

    if not calibration_rigs or any(
        rig.split_role != "safety_calibration" for rig in calibration_rigs
    ):
        raise ValueError("envelope requires safety-calibration rigs only")
    if not math.isfinite(margin) or margin < 1.0:
        raise ValueError("calibration margin must be at least one")
    row_scales: list[float] = []
    column_scales: list[float] = []
    for rig in calibration_rigs:
        row, column = predict_metric_masses(estimator, inference_features_from_rig(rig))
        with exact_mass_access_scope(
            instrumentation,
            rig_id=rig.rig_id,
            split_role=rig.split_role,
            phase="safety_calibration",
            fresh_access_allowed=False,
        ):
            exact_row, exact_column = exact_masses_from_signed_matrix(rig.signed_matrix)
        row_scales.append(float(torch.amax(exact_row / row)))
        column_scales.append(float(torch.amax(exact_column / column)))
    return CalibrationEnvelope(
        row_scale=float(margin) * max(1.0, max(row_scales)),
        column_scale=float(margin) * max(1.0, max(column_scales)),
        calibration_rig_ids=tuple(rig.rig_id for rig in calibration_rigs),
        exact_mass_materializations=len(calibration_rigs),
        factor_mass_vector_accesses=2 * len(calibration_rigs),
        factor_feature_construction_calls=len(calibration_rigs),
        estimator_head_forward_calls=2 * len(calibration_rigs),
    )


def apply_calibration_envelope(
    predicted_row_mass: torch.Tensor,
    predicted_column_mass: torch.Tensor,
    envelope: CalibrationEnvelope,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a frozen global envelope without fresh-rig exact teachers."""

    _require_finite_positive("predicted row mass", predicted_row_mass)
    _require_finite_positive("predicted column mass", predicted_column_mass)
    return predicted_row_mass * envelope.row_scale, predicted_column_mass * envelope.column_scale


def build_diagonal_metric(
    row_mass: torch.Tensor, column_mass: torch.Tensor, *, eta: float
) -> DiagonalMetric:
    """Build the row/column mass metric with ``0 < eta < 1``."""

    _require_finite_positive("row mass", row_mass)
    _require_finite_positive("column mass", column_mass)
    if not math.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between zero and one")
    return DiagonalMetric(
        row_mass=row_mass,
        column_mass=column_mass,
        sigma=float(eta) / row_mass,
        tau=float(eta) / column_mass,
        eta=float(eta),
    )


def exact_masses_from_signed_matrix(
    signed_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize exact absolute masses from signed A, never caller teachers."""

    scope = _EXACT_MASS_SCOPE.get()
    if scope is not None:
        allowed = not (
            scope.split_role == "fresh_geometry_ood"
            and not scope.fresh_access_allowed
        )
        scope.instrumentation.accesses.append(
            {
                "rig_id": scope.rig_id,
                "split_role": scope.split_role,
                "phase": scope.phase,
                "allowed": allowed,
            }
        )
        if not allowed:
            raise RuntimeError(
                "fresh exact-mass access blocked during " + scope.phase
            )
    if signed_matrix.ndim != 2 or not bool(torch.all(torch.isfinite(signed_matrix))):
        raise ValueError("signed_matrix must be a finite matrix")
    return torch.sum(torch.abs(signed_matrix), dim=1), torch.sum(torch.abs(signed_matrix), dim=0)


def audit_schur_safety(
    signed_matrix: torch.Tensor, metric: DiagonalMetric, *, eta: float
) -> dict[str, float | int]:
    """Recompute exact masses and count row, column, and spectral violations."""

    if signed_matrix.shape != (metric.sigma.numel(), metric.tau.numel()):
        raise ValueError("signed_matrix and metric shapes differ")
    if float(eta) != metric.eta:
        raise ValueError("eta differs from frozen metric")
    exact_row, exact_column = exact_masses_from_signed_matrix(signed_matrix)
    row_product = metric.sigma * exact_row
    column_product = metric.tau * exact_column
    tolerance = 64.0 * torch.finfo(signed_matrix.dtype).eps
    threshold = metric.eta * (1.0 + tolerance)
    row_violations = int(torch.count_nonzero(row_product > threshold))
    column_violations = int(torch.count_nonzero(column_product > threshold))
    normalized = torch.sqrt(metric.sigma)[:, None] * signed_matrix * torch.sqrt(metric.tau)[None, :]
    spectral_squared = float(torch.linalg.matrix_norm(normalized, ord=2).square())
    spectral_bound = metric.eta * metric.eta
    spectral_violation = int(spectral_squared > spectral_bound * (1.0 + tolerance))
    return {
        "row_violation_count": row_violations,
        "column_violation_count": column_violations,
        "spectral_violation_count": spectral_violation,
        "total_violation_count": row_violations + column_violations + spectral_violation,
        "maximum_row_product": float(torch.amax(row_product)),
        "maximum_column_product": float(torch.amax(column_product)),
        "dense_normalized_spectral_norm_squared": spectral_squared,
        "schur_squared_upper_bound": spectral_bound,
        "exact_masses_recomputed_from_signed_a": 1,
    }


def run_signed_residual_trajectory(
    signed_matrix: torch.Tensor,
    target: torch.Tensor,
    truth: torch.Tensor,
    metric: DiagonalMetric,
    *,
    checkpoints: Sequence[int],
    theta: float,
) -> TrajectoryResult:
    """Replay one A-only trajectory and report the frozen field-relative-L2."""

    if signed_matrix.ndim != 2 or target.ndim != 1 or truth.ndim != 1:
        raise ValueError("matrix, target, and truth dimensions are invalid")
    if signed_matrix.shape != (target.numel(), truth.numel()):
        raise ValueError("matrix, target, and truth shapes differ")
    if signed_matrix.shape != (metric.sigma.numel(), metric.tau.numel()):
        raise ValueError("metric shape differs from signed_matrix")
    if not math.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    ordered = sorted({int(value) for value in checkpoints})
    if not ordered or ordered[0] != 0 or ordered[-1] < 1 or any(v < 0 for v in ordered):
        raise ValueError("checkpoints must include zero and a positive endpoint")

    x = torch.zeros_like(truth)
    x_bar = x.clone()
    dual = torch.zeros_like(target)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-30)
    truth_norm = torch.linalg.vector_norm(truth).clamp_min(1e-30)
    rows: list[dict[str, float | int]] = []
    forward_solver_calls = transpose_solver_calls = forward_evaluation_calls = 0

    def record(iteration: int) -> None:
        nonlocal forward_evaluation_calls
        residual = signed_matrix @ x - target
        forward_evaluation_calls += 1
        rows.append(
            {
                "iteration": iteration,
                "normalized_residual_l2": float(torch.linalg.vector_norm(residual) / target_norm),
                "field_relative_l2": float(torch.linalg.vector_norm(x - truth) / truth_norm),
                "solution_l2": float(torch.linalg.vector_norm(x)),
            }
        )

    record(0)
    checkpoint_set = set(ordered)
    for iteration in range(1, ordered[-1] + 1):
        projected = signed_matrix @ x_bar
        forward_solver_calls += 1
        dual = (dual + metric.sigma * (projected - target)) / (1.0 + metric.sigma)
        gradient = signed_matrix.T @ dual
        transpose_solver_calls += 1
        next_x = x - metric.tau * gradient
        x_bar = next_x + float(theta) * (next_x - x)
        x = next_x
        if iteration in checkpoint_set:
            record(iteration)
    if not all(math.isfinite(float(row["field_relative_l2"])) for row in rows):
        raise RuntimeError("trajectory produced non-finite field error")
    return TrajectoryResult(
        rows=rows,
        ledger={
            "signed_forward_solver_calls": forward_solver_calls,
            "signed_transpose_solver_calls": transpose_solver_calls,
            "signed_forward_evaluation_calls": forward_evaluation_calls,
            "field_error_evaluation_calls": len(rows),
            "iteration_budget": ordered[-1],
        },
    )
