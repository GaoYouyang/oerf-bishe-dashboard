"""Deployment-input residual diagnostics for T16 v3k-F stopping controls."""

from __future__ import annotations

import hashlib

import numpy as np


def active_observation_noise_scale(
    observation: np.ndarray,
    view_masks: np.ndarray,
    relative_noise: np.ndarray,
) -> np.ndarray:
    """Estimate scalar noise sigma from active observations and calibrated q.

    The approximation assumes ``rms(y)^2 ~= rms(clean)^2 + sigma^2`` and
    ``sigma = q * rms(clean)``. It uses no clean observation, truth field, or
    inactive camera. In a real experiment q must come from an independent
    sensor/background calibration.
    """
    values = np.asarray(observation, dtype=np.float64)
    masks = np.asarray(view_masks, dtype=np.float64)
    q = np.asarray(relative_noise, dtype=np.float64)
    if values.ndim != 4 or masks.ndim != 2:
        raise ValueError("observation and masks must be [batch,depth,view,n] and [batch,view]")
    if values.shape[0] != len(masks) or values.shape[2] != masks.shape[1]:
        raise ValueError("observation and masks are not aligned")
    if q.shape != (len(values),) or np.any(~np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("relative noise must contain one positive value per sample")
    weight = masks[:, None, :, None]
    count = masks.sum(axis=1) * values.shape[1] * values.shape[3]
    if np.any(count <= 0.0):
        raise ValueError("every sample must have at least one active camera")
    observed_rms = np.sqrt(
        np.sum((values * weight) ** 2, axis=(1, 2, 3)) / count
    )
    return q * observed_rms / np.sqrt(1.0 + q * q)


def generator_noise_scale(
    clean_observation: np.ndarray,
    relative_noise: np.ndarray,
) -> np.ndarray:
    """Return simulator-known sigma; this is an oracle diagnostic only."""
    clean = np.asarray(clean_observation, dtype=np.float64)
    q = np.asarray(relative_noise, dtype=np.float64)
    if clean.ndim != 4 or q.shape != (len(clean),):
        raise ValueError("clean observation and relative noise are not aligned")
    return q * np.sqrt(np.mean(clean * clean, axis=(1, 2, 3)))


def camera_ncp_ks(residual: np.ndarray, view_masks: np.ndarray) -> np.ndarray:
    """Compute detector-axis normalized-cumulative-periodogram KS distances.

    The detector mean is removed for each depth-camera line. Periodogram power
    is averaged over depth, and the zero-frequency coefficient is excluded.
    Inactive cameras are returned as NaN so all later aggregation can preserve
    equal camera weights explicitly.
    """
    values = np.asarray(residual, dtype=np.float64)
    masks = np.asarray(view_masks, dtype=np.float64)
    if values.ndim != 4 or masks.shape != (len(values), values.shape[2]):
        raise ValueError("residual and masks are not aligned")
    if values.shape[-1] < 4:
        raise ValueError("NCP requires at least four detector samples")
    centered = values - values.mean(axis=-1, keepdims=True)
    periodogram = np.abs(
        np.fft.rfft(centered, axis=-1, norm="ortho")
    ) ** 2
    periodogram = periodogram[..., 1:].mean(axis=1)
    denominator = np.sum(periodogram, axis=-1, keepdims=True)
    cumulative = np.divide(
        np.cumsum(periodogram, axis=-1),
        denominator,
        out=np.zeros_like(periodogram),
        where=denominator > 1e-30,
    )
    target = np.arange(1, periodogram.shape[-1] + 1, dtype=np.float64)
    target /= periodogram.shape[-1]
    distance = np.max(np.abs(cumulative - target[None, None]), axis=-1)
    return np.where(masks > 0.5, distance, np.nan)


def residual_statistics(
    residual_history: np.ndarray,
    view_masks: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize discrepancy and whiteness without truth or audit cameras."""
    residual = np.asarray(residual_history, dtype=np.float64)
    masks = np.asarray(view_masks, dtype=np.float64)
    scale = np.asarray(sigma, dtype=np.float64)
    if residual.ndim != 5:
        raise ValueError("residual history must be [step,batch,depth,view,n]")
    if masks.shape != (residual.shape[1], residual.shape[3]):
        raise ValueError("residual history and masks are not aligned")
    if scale.shape != (residual.shape[1],) or np.any(scale <= 0.0):
        raise ValueError("sigma must contain one positive value per sample")
    active_camera_count = masks.sum(axis=1)
    if np.any(active_camera_count <= 0.0):
        raise ValueError("every sample must have at least one active camera")

    camera_rms = np.sqrt(np.mean(residual * residual, axis=(2, 4)))
    camera_ratio = camera_rms / scale[None, :, None]
    active = masks[None] > 0.5
    pooled = np.sqrt(
        np.sum(np.where(active, camera_ratio * camera_ratio, 0.0), axis=2)
        / active_camera_count[None]
    )
    camera_max = np.max(np.where(active, camera_ratio, -np.inf), axis=2)

    ncp_rows = [camera_ncp_ks(step, masks) for step in residual]
    ncp = np.stack(ncp_rows)
    ncp_mean = np.nanmean(ncp, axis=2)
    ncp_max = np.nanmax(ncp, axis=2)
    return {
        "discrepancy_pooled": pooled,
        "discrepancy_camera_max": camera_max,
        "ncp_camera": ncp,
        "ncp_camera_mean": ncp_mean,
        "ncp_camera_max": ncp_max,
    }


def white_noise_ncp_thresholds(
    depth: int,
    detector_pixels: int,
    active_cameras: int,
    *,
    samples: int,
    quantile: float,
    seed: int,
) -> dict[str, float]:
    """Monte Carlo reference thresholds for ideal independent white noise."""
    if min(depth, detector_pixels, active_cameras, samples) < 1:
        raise ValueError("white-noise threshold dimensions must be positive")
    if detector_pixels < 4 or not 0.5 < quantile < 1.0:
        raise ValueError("invalid NCP detector size or quantile")
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(
        size=(int(samples), int(depth), int(active_cameras), int(detector_pixels))
    ).astype(np.float32)
    masks = np.ones((int(samples), int(active_cameras)), dtype=np.float64)
    distances = camera_ncp_ks(noise, masks)
    return {
        "mean": float(np.quantile(np.mean(distances, axis=1), quantile)),
        "maximum": float(np.quantile(np.max(distances, axis=1), quantile)),
    }


def first_crossing(condition: np.ndarray, maximum_iteration: int) -> np.ndarray:
    """Return first checked iterate, or x_max after a forced-cap run."""
    values = np.asarray(condition, dtype=bool)
    maximum = int(maximum_iteration)
    if values.ndim != 2 or values.shape[0] != maximum:
        raise ValueError("condition must check x_0 through x_(maximum-1)")
    selected = np.argmax(values, axis=0).astype(np.int64)
    selected[~np.any(values, axis=0)] = maximum
    return selected


def effective_operator_calls(
    stop_iteration: np.ndarray, maximum_iteration: int
) -> tuple[np.ndarray, np.ndarray]:
    """Count A/A^T calls when the residual check reuses the gradient forward."""
    stop = np.asarray(stop_iteration, dtype=np.int64)
    maximum = int(maximum_iteration)
    if stop.ndim != 1 or np.any(stop < 0) or np.any(stop > maximum):
        raise ValueError("stop iterations must lie in [0, maximum]")
    capped = stop == maximum
    a_calls = np.where(capped, maximum, stop + 1)
    at_calls = np.where(capped, maximum, stop)
    return a_calls.astype(np.int64), at_calls.astype(np.int64)


def gather_path(path: np.ndarray, stop_iteration: np.ndarray) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    stop = np.asarray(stop_iteration, dtype=np.int64)
    if values.ndim != 5 or stop.shape != (values.shape[1],):
        raise ValueError("path and stop iterations are not aligned")
    if np.any(stop < 0) or np.any(stop >= values.shape[0]):
        raise ValueError("stop iteration is outside the stored path")
    return values[stop, np.arange(len(stop))]


def grouped_validation_assignment(
    source_indices: np.ndarray,
    sample_seeds: np.ndarray,
    tune_field_count: int,
    assignment_seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Split whole source-field clusters into deterministic tune/lock groups."""
    sources = np.asarray(source_indices, dtype=np.int64)
    seeds = np.asarray(sample_seeds, dtype=np.int64)
    if sources.shape != seeds.shape or sources.ndim != 1:
        raise ValueError("source indices and sample seeds must be aligned vectors")
    unique = np.unique(sources)
    count = int(tune_field_count)
    if not 0 < count < len(unique):
        raise ValueError("tune field count must leave a nonempty lock group")
    records = []
    for source in unique:
        subset = np.flatnonzero(sources == source)
        unique_seed = np.unique(seeds[subset])
        if len(unique_seed) != 1:
            raise ValueError("one source field maps to multiple sample seeds")
        token = f"{int(assignment_seed)}:{int(unique_seed[0])}".encode("ascii")
        records.append((hashlib.sha256(token).hexdigest(), int(source), int(unique_seed[0])))
    records.sort()
    tune_sources = {source for _, source, _ in records[:count]}
    tune_mask = np.asarray([int(source) in tune_sources for source in sources])
    manifest = [
        {
            "source_index": source,
            "sample_seed": seed,
            "validation_role": "tune" if source in tune_sources else "lock",
            "assignment_hash": digest,
        }
        for digest, source, seed in records
    ]
    return tune_mask, manifest
