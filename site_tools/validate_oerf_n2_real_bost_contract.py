#!/usr/bin/env python3
"""Validate the N2 real-BOST intake contract without opening its audit split.

The public report contains counts and gate states only. Passing all gates allows
preregistration and later non-audit development; it never opens the audit set or
authorizes an algorithm, generalization, or publication claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover - exercised by the documented install path
    Draft202012Validator = None
    FormatChecker = None


CONTRACT_SCHEMA = "oerf-n2-real-bost-contract-1.0"
REPORT_SCHEMA = "oerf-n2-real-bost-contract-readiness-public-1.0"
PLACEHOLDER_STATUS = "N2_WAITING_FOR_LAB_INPUT"
NOT_READY_STATUS = "N2_NOT_AUTHORIZED_MISSING_CONTRACT_GATES"
READY_STATUS = "N2_PREREGISTRATION_READY_AUDIT_STILL_SEALED"
FIXTURE_STATUS = "CONTRACT_TEST_FIXTURE_VALIDATED_NOT_REAL_DATA"
DOT_PRODUCT_LIMIT = 1e-6
FINITE_DIFFERENCE_LIMIT = 1e-4
MIN_FLOW_OFF_REPEATS = 50
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_VALUES = {"", "REPLACE_ME", "unknown"}
ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data_templates/oerf_n2_real_bost_contract.schema.json"
SPLIT_LIST_KEYS = (
    "training_unit_ids",
    "tuning_unit_ids",
    "validation_unit_ids",
    "audit_unit_ids",
)
SPLIT_FIELD_BY_UNIT = {
    "view": "view_id",
    "sensor": "sensor_id",
    "run": "run_id",
    "session": "session_id",
    "condition": "condition_id",
    "geometry": "geometry_id",
}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"E_TYPE_OBJECT at {location}")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"E_TYPE_ARRAY at {location}")
    return value


def _strings(value: Any, location: str) -> list[str]:
    items = _array(value, location)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"E_NONEMPTY_STRING_ARRAY at {location}")
    if len(items) != len(set(items)):
        raise ValueError(f"E_DUPLICATE_ARRAY_ITEM at {location}")
    return list(items)


def _number_or_none(value: Any, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"E_TYPE_NUMBER at {location}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"E_NONFINITE_NUMBER at {location}")
    return result


def _required(mapping: Mapping[str, Any], keys: Sequence[str], location: str) -> None:
    if any(key not in mapping for key in keys):
        raise ValueError(f"E_REQUIRED_KEY at {location}")


def _unique_ids(rows: Sequence[Mapping[str, Any]], key: str, location: str) -> list[str]:
    values: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"E_NONEMPTY_ID at {location}[{index}].{key}")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError(f"E_DUPLICATE_ID at {location}.{key}")
    return values


def _safe_relative_path(value: Any, location: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"E_RELATIVE_PATH at {location}")
    if "\\" in value:
        raise ValueError(f"E_PORTABLE_PATH at {location}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"E_PATH_ESCAPES_PRIVATE_ROOT at {location}")


def _reject_json_constant(_value: str) -> None:
    raise ValueError("E_JSON_NONFINITE: NaN and Infinity are forbidden")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except ValueError as exc:
        if str(exc).startswith("E_JSON_NONFINITE"):
            raise
        raise ValueError("E_INVALID_JSON: cannot read a valid N2 contract") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("E_INVALID_JSON: cannot read a valid N2 contract") from exc
    return dict(_mapping(value, "contract"))


def _assert_finite_numbers(value: Any, location: str = "contract") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"E_NONFINITE_NUMBER at {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            safe_key = key if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_]+", key) else "field"
            _assert_finite_numbers(item, f"{location}.{safe_key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_numbers(item, f"{location}[{index}]")


def _schema_location(error: Any) -> str:
    location = "contract"
    for part in error.absolute_path:
        if isinstance(part, int):
            location += f"[{part}]"
        elif isinstance(part, str) and re.fullmatch(r"[A-Za-z0-9_]+", part):
            location += f".{part}"
        else:
            location += ".field"
    return location


def _validate_schema(contract: Mapping[str, Any]) -> None:
    if Draft202012Validator is None or FormatChecker is None:
        raise ValueError(
            "E_DEPENDENCY_MISSING: install site_tools/requirements-n2.txt"
        )
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("E_SCHEMA_UNAVAILABLE: local schema cannot be loaded") from exc
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(contract),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        keyword = str(error.validator or "unknown")
        raise ValueError(
            f"E_SCHEMA_VIOLATION at {_schema_location(error)}: keyword={keyword}"
        )


def _split_universe(split_unit: str, views: Sequence[Mapping[str, Any]]) -> set[str]:
    field = SPLIT_FIELD_BY_UNIT.get(split_unit)
    if field is None:
        return set()
    return {
        str(view[field])
        for view in views
        if isinstance(view.get(field), str) and view.get(field)
    }


def compute_split_digest(split: Mapping[str, Any]) -> str:
    """Return the canonical digest that seals split membership and audit policy."""
    payload = {
        "split_unit": split.get("split_unit"),
        **{key: sorted(list(split.get(key, []))) for key in SPLIT_LIST_KEYS},
        "audit_locked": split.get("audit_locked"),
        "audit_opened": split.get("audit_opened"),
        "frozen_at_utc": split.get("frozen_at_utc"),
        "random_frame_split_permitted": split.get("random_frame_split_permitted"),
    }
    rendered = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _check_paths(
    provenance: Mapping[str, Any],
    field: Mapping[str, Any],
    sensors: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
    mismatch: Mapping[str, Any],
    endpoints: Mapping[str, Any],
) -> None:
    for key in ("manifest_path", "source_evidence_path"):
        _safe_relative_path(provenance.get(key), f"provenance.{key}")
    for key in ("support_path", "truth_path"):
        _safe_relative_path(field.get(key), f"field_domain.{key}")
    for index, sensor in enumerate(sensors):
        for key in ("intrinsics_path", "extrinsics_path", "ray_bundle_path"):
            _safe_relative_path(sensor.get(key), f"sensors[{index}].{key}")
    for index, view in enumerate(views):
        for key in (
            "reference_path",
            "flow_on_path",
            "displacement_path",
            "mask_path",
            "confidence_path",
            "timestamps_path",
        ):
            _safe_relative_path(view.get(key), f"views[{index}].{key}")
    evidence = _mapping(mismatch.get("evidence"), "physical_mismatch.evidence")
    for index, row in enumerate(
        _array(evidence.get("condition_evidence"), "physical_mismatch.evidence.condition_evidence")
    ):
        condition = _mapping(row, f"physical_mismatch.evidence.condition_evidence[{index}]")
        _safe_relative_path(
            condition.get("flow_off_manifest_path"),
            f"physical_mismatch.evidence.condition_evidence[{index}].flow_off_manifest_path",
        )
    _safe_relative_path(
        endpoints.get("external_reference_path"),
        "endpoints.external_reference_path",
    )


def _check_record_consistency(
    record_kind: str,
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    origin = identity.get("data_origin")
    evidence_class = provenance.get("evidence_class")
    if record_kind == "LAB_INTAKE_PLACEHOLDER":
        valid = origin == "unknown" and evidence_class == "unknown"
    elif record_kind == "CONTRACT_TEST_FIXTURE":
        valid = origin == "synthetic_contract_test" and evidence_class == "synthetic_fixture"
    else:
        valid = (origin, evidence_class) in {
            ("lab_real", "lab_handoff"),
            ("public_real", "open_dataset_manifest"),
        }
    if not valid:
        raise ValueError("E_RECORD_PROVENANCE_MISMATCH at record_kind")


def _identity_and_units_gate(
    record_kind: str,
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    field: Mapping[str, Any],
) -> tuple[bool, str]:
    if record_kind == "LAB_INTAKE_PLACEHOLDER":
        return False, "仍是待师兄填写的 intake 占位表"
    required_identity = (
        "dataset_id",
        "case_id",
        "run_id",
        "session_id",
        "condition_id",
        "geometry_id",
    )
    if any(str(identity.get(key, "")) in PLACEHOLDER_VALUES for key in required_identity):
        return False, "数据、工况、run/session 或 geometry 身份仍有占位值"
    if identity.get("data_origin") == "unknown" or identity.get("independent_unit") == "unknown":
        return False, "数据来源或独立实验单位尚未定义"
    if identity.get("acquisition_time_utc") is None:
        return False, "采集时间尚未记录"
    provenance_ready = (
        provenance.get("manifest_path") is not None
        and isinstance(provenance.get("manifest_digest_sha256"), str)
        and SHA256_RE.fullmatch(str(provenance.get("manifest_digest_sha256"))) is not None
        and isinstance(provenance.get("manifest_entry_count"), int)
        and not isinstance(provenance.get("manifest_entry_count"), bool)
        and int(provenance.get("manifest_entry_count", 0)) >= 1
        and provenance.get("source_evidence_path") is not None
        and provenance.get("verified_at_utc") is not None
    )
    if not provenance_ready:
        return False, "数据 manifest、来源证明或核验时间尚未绑定"
    if field.get("parameterization") == "unknown" or field.get("units") == "unknown":
        return False, "场变量或单位尚未定义"
    if field.get("axis_order") == "unknown":
        return False, "体数据轴顺序尚未定义"
    shape = field.get("grid_shape")
    bounds = field.get("bounds_m")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 2
            for value in shape
        )
    ):
        return False, "三维 grid_shape 尚未给出"
    if not (
        isinstance(bounds, list)
        and len(bounds) == 3
        and all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in pair
            )
            and float(pair[0]) < float(pair[1])
            for pair in bounds
        )
    ):
        return False, "三维物理边界必须用米给出且每轴下界小于上界"
    if field.get("support_path") is None:
        return False, "部署时可用的 support/ROI 尚未绑定"
    return True, "来源、单位、轴顺序、网格、物理边界与 support 已定义"


def _observation_geometry_gate(
    sensors: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    reconstruction = [view for view in views if view.get("role") == "reconstruction"]
    audit = [view for view in views if view.get("role") == "audit_locked"]
    if not sensors or len(reconstruction) < 2 or not audit:
        return False, "接口样例至少要有一个 sensor、两个 reconstruction views 和一个 audit view"
    for sensor in sensors:
        geometry = sensor.get("ray_bundle_path") is not None or (
            sensor.get("intrinsics_path") is not None
            and sensor.get("extrinsics_path") is not None
        )
        image_shape = sensor.get("image_shape")
        if not geometry or not isinstance(image_shape, list) or len(image_shape) != 2:
            return False, "每个 sensor 都要有 image shape 与 ray bundle 或内外参"
        if (
            str(sensor.get("calibration_id", "")) in PLACEHOLDER_VALUES
            or str(sensor.get("calibration_version", "")) in PLACEHOLDER_VALUES
            or sensor.get("calibration_reprojection_rmse_px") is None
        ):
            return False, "每个 sensor 都要绑定标定 ID、版本和重投影误差"
    for view in views:
        has_observation = view.get("displacement_path") is not None or (
            view.get("reference_path") is not None and view.get("flow_on_path") is not None
        )
        if not has_observation or view.get("mask_path") is None:
            return False, "每个 view 都要有位移或原图对，并绑定固定 mask"
        if view.get("confidence_path") is None:
            return False, "每个 view 都要保留置信度/相关峰或显式质量图"
        if view.get("observation_units") == "unknown" or view.get("component_order") == "unknown":
            return False, "位移单位或双分量顺序仍未知"
    return True, "观测、mask/confidence、标定和 ray/camera geometry 已绑定"


def _operator_gate(
    forward: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[bool, str]:
    interface = forward.get("interface")
    linear_pair = bool(forward.get("can_apply_forward")) and bool(forward.get("can_apply_adjoint"))
    tangent_pair = bool(forward.get("can_jvp")) and bool(forward.get("can_vjp"))
    if interface in {"explicit_matrix", "forward_adjoint"} and not linear_pair:
        return False, "声明了线性接口，但 A/A^T 不能成对调用"
    if interface == "jvp_vjp" and not tangent_pair:
        return False, "声明了非线性接口，但 JVP/VJP 不能成对调用"
    if interface not in {"explicit_matrix", "forward_adjoint", "jvp_vjp"}:
        return False, "只有 reprojection 或无 operator 接口，不能做同预算求解器实验"
    if not forward.get("deterministic") or not forward.get("row_layout_documented"):
        return False, "operator 采样不确定或 measurement row layout 未记录"
    dot = _mapping(audit.get("dot_product"), "operator_audit.dot_product")
    dot_error = _number_or_none(dot.get("relative_error"), "operator_audit.dot_product.relative_error")
    if (
        dot.get("required") is not True
        or dot_error is None
        or not 0 <= dot_error <= DOT_PRODUCT_LIMIT
    ):
        return False, f"伴随点积误差尚未达到 {DOT_PRODUCT_LIMIT:.0e}"
    needs_fd = forward.get("regime") == "curved_ray" or interface == "jvp_vjp"
    finite = _mapping(audit.get("finite_difference"), "operator_audit.finite_difference")
    finite_error = _number_or_none(
        finite.get("relative_error"),
        "operator_audit.finite_difference.relative_error",
    )
    if needs_fd and (
        finite.get("required") is not True
        or finite_error is None
        or not 0 <= finite_error <= FINITE_DIFFERENCE_LIMIT
    ):
        return False, f"非线性 Jacobian 的有限差分误差尚未达到 {FINITE_DIFFERENCE_LIMIT:.0e}"
    if audit.get("unit_scale_passed") is not True or audit.get("support_mask_passed") is not True:
        return False, "单位尺度或 support/mask 审计未通过"
    return True, "线性 A/A^T 或非线性 JVP/VJP、数值与 support 审计已过门"


def _physical_mismatch_gate(
    mismatch: Mapping[str, Any],
    forward: Mapping[str, Any],
    sensors: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    primary = mismatch.get("primary")
    evidence = _mapping(mismatch.get("evidence"), "physical_mismatch.evidence")
    if primary == "unknown" or mismatch.get("frozen_before_audit") is not True:
        return False, "主失配尚未唯一冻结，或冻结发生在 audit 打开之后"
    condition_rows = [
        _mapping(value, f"physical_mismatch.evidence.condition_evidence[{index}]")
        for index, value in enumerate(
            _array(evidence.get("condition_evidence"), "physical_mismatch.evidence.condition_evidence")
        )
    ]
    if not condition_rows:
        return False, "还没有逐固定条件的 flow-off 与物理证据清单"
    _unique_ids(condition_rows, "condition_id", "physical_mismatch.evidence.condition_evidence")
    sensors_by_id = {str(sensor["sensor_id"]): sensor for sensor in sensors}
    for row in condition_rows:
        sensor = sensors_by_id.get(str(row.get("sensor_id", "")))
        if sensor is None:
            raise ValueError("E_CONDITION_SENSOR_REFERENCE at physical_mismatch.evidence")
        matching_views = [
            view
            for view in views
            if view.get("sensor_id") == row.get("sensor_id")
            and view.get("condition_id") == row.get("condition_id")
            and view.get("session_id") == row.get("session_id")
            and view.get("geometry_id") == row.get("geometry_id")
        ]
        if not matching_views:
            raise ValueError("E_CONDITION_VIEW_REFERENCE at physical_mismatch.evidence")
        row_f = _number_or_none(row.get("f_number"), "physical_mismatch.evidence.condition_evidence.f_number")
        sensor_f = _number_or_none(sensor.get("f_number"), "sensors.f_number")
        if row_f is not None and sensor_f is not None and not math.isclose(row_f, sensor_f, rel_tol=1e-9, abs_tol=0.0):
            raise ValueError("E_CONDITION_SENSOR_FNUMBER_MISMATCH at physical_mismatch.evidence")
        observed_patterns = {str(view.get("background_pattern_id")) for view in matching_views}
        declared_patterns = set(_strings(row.get("background_pattern_ids"), "condition_evidence.background_pattern_ids"))
        if declared_patterns and not declared_patterns.issubset(observed_patterns):
            raise ValueError("E_CONDITION_PATTERN_REFERENCE at physical_mismatch.evidence")
    if any(
        int(row.get("flow_off_repeat_count", 0)) < MIN_FLOW_OFF_REPEATS
        or row.get("flow_off_manifest_path") is None
        for row in condition_rows
    ):
        return False, f"每个固定条件都要有 manifest 且 flow-off repeats 至少为 {MIN_FLOW_OFF_REPEATS}"

    declared_pairs = set(_strings(evidence.get("paired_condition_ids"), "physical_mismatch.evidence.paired_condition_ids"))
    rows_by_condition = {str(row["condition_id"]): row for row in condition_rows}
    if declared_pairs and not declared_pairs.issubset(rows_by_condition):
        raise ValueError("E_PAIRED_CONDITION_REFERENCE at physical_mismatch.evidence")

    if primary == "finite_aperture":
        pair_rows = [rows_by_condition[value] for value in declared_pairs]
        actual_levels = {
            float(row["f_number"])
            for row in pair_rows
            if row.get("f_number") is not None
        }
        declared_levels = {float(value) for value in evidence.get("f_number_levels", [])}
        channels = {
            str(sensors_by_id[str(row["sensor_id"])].get("optical_channel_id"))
            for row in pair_rows
        }
        geometries = {str(row.get("geometry_id")) for row in pair_rows}
        passed = (
            len(pair_rows) >= 2
            and len(actual_levels) >= 2
            and declared_levels == actual_levels
            and len(channels) == 1
            and len(geometries) == 1
            and evidence.get("aperture_pairing_policy") == "same_optical_channel_same_geometry"
            and evidence.get("high_fidelity_forward_available") is True
            and forward.get("aperture_model") in {"cone_multi_ray", "measured_psf"}
        )
        return passed, (
            "同光学通道、同几何的至少两个真实 f-number 已与 cone/PSF forward 成对绑定"
            if passed
            else "finite-aperture 需要同光学通道、同几何的真实 f-number 对和 cone/PSF comparator"
        )
    if primary == "ray_bending":
        passed = (
            forward.get("regime") == "curved_ray"
            and evidence.get("high_fidelity_forward_available") is True
            and len(declared_pairs) >= 2
        )
        return passed, (
            "曲线 ray 与至少两个真实成对条件已定义"
            if passed
            else "ray-bending 需要曲线 ray 高保真 forward 和至少两个真实成对条件"
        )
    if primary == "calibration_drift":
        versions = set(_strings(evidence.get("calibration_versions"), "physical_mismatch.evidence.calibration_versions"))
        sessions = set(_strings(evidence.get("repeated_session_ids"), "physical_mismatch.evidence.repeated_session_ids"))
        observed_versions = {str(sensor.get("calibration_version")) for sensor in sensors}
        observed_sessions = {str(view.get("session_id")) for view in views}
        passed = (
            len(versions) >= 2
            and versions.issubset(observed_versions)
            and len(sessions) >= 2
            and sessions.issubset(observed_sessions)
        )
        return passed, (
            "至少两个可追踪标定版本和独立 session 已成对"
            if passed
            else "calibration-drift 的本项目鉴别门需要真实版本和时间分离 session"
        )
    if primary == "displacement_extraction":
        methods = set(_strings(evidence.get("displacement_methods"), "physical_mismatch.evidence.displacement_methods"))
        observed_methods = {str(view.get("displacement_method")) for view in views}
        backgrounds = set(
            _strings(
                evidence.get("independent_background_pattern_ids"),
                "physical_mismatch.evidence.independent_background_pattern_ids",
            )
        )
        observed_backgrounds = {str(view.get("background_pattern_id")) for view in views}
        raw_pairs = bool(views) and all(
            view.get("reference_path") is not None and view.get("flow_on_path") is not None
            for view in views
        )
        passed = (
            len(methods) >= 2
            and methods.issubset(observed_methods)
            and len(backgrounds) >= 2
            and backgrounds.issubset(observed_backgrounds)
            and raw_pairs
        )
        return passed, (
            "原图对、至少两种位移方法和至少两个独立背景均已绑定"
            if passed
            else "位移支路需区分同背景时间重复与多独立背景，并保留至少两种实际方法"
        )
    if primary == "discretization":
        levels = evidence.get("discretization_levels", [])
        passed = (
            isinstance(levels, list)
            and len(set(levels)) >= 2
            and evidence.get("high_fidelity_forward_available") is True
        )
        return passed, (
            "至少两个离散层级与高保真 comparator 已冻结"
            if passed
            else "discretization 控制分支需要至少两个层级和一个高保真 comparator"
        )
    if primary == "combined_predeclared":
        return False, "combined 路线不自动授权；先冻结一个 primary，其余只作预声明消融"
    return False, "不支持的主失配类型"


def _split_gate(
    split: Mapping[str, Any],
    split_lists: Mapping[str, Sequence[str]],
    views: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Any],
) -> tuple[bool, str]:
    split_unit = str(split.get("split_unit", "unknown"))
    training = list(split_lists["training_unit_ids"])
    audit_ids = list(split_lists["audit_unit_ids"])
    digest = split.get("split_digest_sha256")
    if split_unit == "unknown" or len(training) < 2 or not audit_ids:
        return False, "split unit、至少两个 training units 或永久 audit units 尚未定义"
    universe = _split_universe(split_unit, views)
    declared = {value for values in split_lists.values() for value in values}
    if not universe or declared != universe:
        return False, "每个 view 所属独立单位都必须且只能进入一个 split 列表"
    if split.get("audit_locked") is not True or split.get("audit_opened") is not False:
        return False, "audit 必须保持 locked 且未打开"
    if split.get("random_frame_split_permitted") is not False:
        return False, "真实或 4D 数据禁止随机拆帧"
    if not isinstance(split.get("frozen_at_utc"), str) or not split.get("frozen_at_utc"):
        return False, "split 冻结时间缺失"
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        return False, "split digest 必须是 64 位小写 SHA-256"
    if digest != compute_split_digest(split):
        return False, "split digest 与当前成员和 audit 策略不一致"
    if endpoints.get("heldout_mask_policy") != "fixed_before_reconstruction":
        return False, "held-out mask 必须在重建前固定"
    return True, "独立单位全覆盖、可复算 SHA-256、audit lock 与固定 mask 已生效"


def _endpoint_gate(
    identity: Mapping[str, Any],
    field: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Any],
    split: Mapping[str, Any],
) -> tuple[bool, str]:
    primary = _mapping(endpoints.get("primary"), "endpoints.primary")
    name = primary.get("name")
    truth_metrics = {"field_relative_l2", "field_h1", "interface_assd", "interface_hd95"}
    if name == "unknown":
        return False, "论文主终点尚未定义"
    if name in truth_metrics:
        passed = (
            primary.get("truth_required") is True
            and field.get("truth_available") is True
            and field.get("truth_path") is not None
            and field.get("truth_provenance") in {"independent_measurement", "validated_cfd", "analytic"}
        )
        return passed, (
            "field/interface 主终点已有独立真值来源"
            if passed
            else "field/H1/interface 指标需要独立真值，真实重投影不能替代"
        )
    if name == "heldout_reprojection_relative_l2":
        passed = (
            primary.get("truth_required") is False
            and bool(split.get("audit_unit_ids"))
            and endpoints.get("heldout_mask_policy") == "fixed_before_reconstruction"
        )
        return passed, (
            "真实主终点为固定 mask 的永久 held-out reprojection"
            if passed
            else "held-out reprojection 需要永久 audit units 和预先固定 mask"
        )
    if name == "piv_velocity_error":
        passed = (
            primary.get("truth_required") is True
            and endpoints.get("external_reference_path") is not None
            and endpoints.get("external_reference_provenance") == "independent_measurement"
        )
        return passed, (
            "PIV velocity endpoint 已绑定独立测量"
            if passed
            else "PIV velocity error 需要独立速度参考，不可由 BOST 自己生成"
        )
    if name == "temporal_stability":
        timed = identity.get("field_is_time_resolved") is True and all(
            view.get("timestamps_path") is not None for view in views
        )
        split_ok = split.get("split_unit") in {"run", "session"}
        passed = timed and split_ok
        return passed, (
            "时间戳完整且按 run/session 隔离"
            if passed
            else "temporal endpoint 需要逐 view 时间戳和 run/session split"
        )
    return False, "当前 validator 不支持该主终点"


def _permissions_gate(
    permissions: Mapping[str, Any],
    boundary: Mapping[str, Any],
    field: Mapping[str, Any],
) -> tuple[bool, str]:
    if permissions.get("public_raw_data") is True and permissions.get("redistribution_basis") not in {
        "open_license",
        "written_permission",
    }:
        raise ValueError("E_RAW_REDISTRIBUTION_PERMISSION at permissions")
    if boundary.get("real_field_truth_claim_allowed") is True and not (
        field.get("truth_available") is True
        and field.get("truth_provenance") in {"independent_measurement", "validated_cfd", "analytic"}
    ):
        raise ValueError("E_TRUTH_CLAIM_WITHOUT_REFERENCE at claim_boundary")
    if boundary.get("audit_may_select_model") is not False or boundary.get("audit_may_select_stopping") is not False:
        raise ValueError("E_AUDIT_SELECTION_FORBIDDEN at claim_boundary")
    if boundary.get("heldout_reprojection_is_unique_3d_truth") is not False:
        raise ValueError("E_REPROJECTION_NOT_UNIQUE_TRUTH at claim_boundary")
    if boundary.get("synthetic_and_experimental_metrics_kept_separate") is not True:
        raise ValueError("E_METRIC_SCOPE_MIXED at claim_boundary")
    passed = permissions.get("local_storage") is True and permissions.get("local_training") is True
    return passed, (
        "本机存储与非 audit 训练权限已明确"
        if passed
        else "尚未得到本机存储和非 audit 训练的明确许可"
    )


def _check_split_roles(
    split: Mapping[str, Any],
    split_lists: Mapping[str, Sequence[str]],
    views: Sequence[Mapping[str, Any]],
) -> None:
    field = SPLIT_FIELD_BY_UNIT.get(str(split.get("split_unit")))
    if field is None:
        return
    audit_units = set(split_lists["audit_unit_ids"])
    non_audit_units = {
        value
        for key in ("training_unit_ids", "tuning_unit_ids", "validation_unit_ids")
        for value in split_lists[key]
    }
    for view in views:
        unit = str(view.get(field, ""))
        role_is_audit = view.get("role") == "audit_locked"
        if unit in audit_units and not role_is_audit:
            raise ValueError("E_SPLIT_ROLE_MISMATCH: audit unit contains non-audit view")
        if unit in non_audit_units and role_is_audit:
            raise ValueError("E_SPLIT_ROLE_MISMATCH: non-audit unit contains audit view")
        if role_is_audit and unit not in audit_units:
            raise ValueError("E_SPLIT_ROLE_MISMATCH: audit view is outside audit units")


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    _assert_finite_numbers(contract)
    _validate_schema(contract)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("E_SCHEMA_VERSION at schema_version")
    record_kind = str(contract.get("record_kind", ""))
    top_keys = (
        "identity",
        "provenance",
        "field_domain",
        "sensors",
        "views",
        "forward_model",
        "operator_audit",
        "physical_mismatch",
        "split_contract",
        "endpoints",
        "permissions",
        "claim_boundary",
    )
    _required(contract, top_keys, "contract")
    identity = _mapping(contract["identity"], "identity")
    provenance = _mapping(contract["provenance"], "provenance")
    field = _mapping(contract["field_domain"], "field_domain")
    sensors = [
        _mapping(value, f"sensors[{index}]")
        for index, value in enumerate(_array(contract["sensors"], "sensors"))
    ]
    views = [
        _mapping(value, f"views[{index}]")
        for index, value in enumerate(_array(contract["views"], "views"))
    ]
    forward = _mapping(contract["forward_model"], "forward_model")
    operator_audit = _mapping(contract["operator_audit"], "operator_audit")
    mismatch = _mapping(contract["physical_mismatch"], "physical_mismatch")
    split = _mapping(contract["split_contract"], "split_contract")
    endpoints = _mapping(contract["endpoints"], "endpoints")
    permissions = _mapping(contract["permissions"], "permissions")
    boundary = _mapping(contract["claim_boundary"], "claim_boundary")

    _check_record_consistency(record_kind, identity, provenance)
    sensor_ids = _unique_ids(sensors, "sensor_id", "sensors")
    view_ids = _unique_ids(views, "view_id", "views")
    sensor_set = set(sensor_ids)
    for index, view in enumerate(views):
        if view.get("sensor_id") not in sensor_set:
            raise ValueError(f"E_SENSOR_REFERENCE at views[{index}].sensor_id")
    _check_paths(provenance, field, sensors, views, mismatch, endpoints)

    split_lists = {
        key: _strings(split.get(key), f"split_contract.{key}")
        for key in SPLIT_LIST_KEYS
    }
    owner: dict[str, str] = {}
    for key, values in split_lists.items():
        for value in values:
            if value in owner:
                raise ValueError("E_SPLIT_OVERLAP: one unit appears in multiple split lists")
            owner[value] = key
    _check_split_roles(split, split_lists, views)

    gate_rows = [
        ("identity_and_units",) + _identity_and_units_gate(record_kind, identity, provenance, field),
        ("observation_and_geometry",) + _observation_geometry_gate(sensors, views),
        ("operator_and_adjoint",) + _operator_gate(forward, operator_audit),
        ("physical_mismatch_evidence",) + _physical_mismatch_gate(mismatch, forward, sensors, views),
        ("independent_split_lock",) + _split_gate(split, split_lists, views, endpoints),
        ("endpoint_legality",) + _endpoint_gate(identity, field, views, endpoints, split),
        ("permissions_and_claims",) + _permissions_gate(permissions, boundary, field),
    ]
    gates = [
        {"gate": name, "passed": bool(passed), "detail": detail}
        for name, passed, detail in gate_rows
    ]
    failed = [row["gate"] for row in gates if not row["passed"]]
    all_passed = not failed
    real_record = record_kind == "DATASET_RECORD" and identity.get("data_origin") in {
        "lab_real",
        "public_real",
    }
    if record_kind == "LAB_INTAKE_PLACEHOLDER":
        status = PLACEHOLDER_STATUS
    elif record_kind == "CONTRACT_TEST_FIXTURE" and all_passed:
        status = FIXTURE_STATUS
    elif all_passed and real_record:
        status = READY_STATUS
    else:
        status = NOT_READY_STATUS

    condition_rows = _array(
        _mapping(mismatch.get("evidence"), "physical_mismatch.evidence").get("condition_evidence"),
        "physical_mismatch.evidence.condition_evidence",
    )
    repeat_counts = [
        int(_mapping(row, "condition_evidence").get("flow_off_repeat_count", 0))
        for row in condition_rows
    ]
    may_preregister = status == READY_STATUS
    return {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "evidence_scope": "CONTRACT_READINESS_ONLY_NO_RECONSTRUCTION_NO_MODEL_RESULT",
        "record_kind": record_kind,
        "source_class": str(identity.get("data_origin", "unknown")),
        "inventory": {
            "sensor_count": len(sensors),
            "view_count": len(views),
            "reconstruction_view_count": sum(view.get("role") == "reconstruction" for view in views),
            "validation_view_count": sum(view.get("role") == "validation" for view in views),
            "audit_locked_view_count": sum(view.get("role") == "audit_locked" for view in views),
            "split_unit": str(split.get("split_unit", "unknown")),
            "training_unit_count": len(split_lists["training_unit_ids"]),
            "tuning_unit_count": len(split_lists["tuning_unit_ids"]),
            "validation_unit_count": len(split_lists["validation_unit_ids"]),
            "audit_unit_count": len(split_lists["audit_unit_ids"]),
            "fixed_condition_evidence_count": len(condition_rows),
            "minimum_flow_off_repeat_count_per_fixed_condition": min(repeat_counts, default=0),
            "provenance_manifest_entry_count": int(provenance.get("manifest_entry_count", 0)),
        },
        "frozen_thresholds": {
            "dot_product_relative_error_max": DOT_PRODUCT_LIMIT,
            "finite_difference_relative_error_max": FINITE_DIFFERENCE_LIMIT,
            "minimum_flow_off_repeats_per_fixed_condition": MIN_FLOW_OFF_REPEATS,
        },
        "declared_primary_mismatch": str(mismatch.get("primary", "unknown")),
        "declared_primary_endpoint": str(
            _mapping(endpoints.get("primary"), "endpoints.primary").get("name", "unknown")
        ),
        "gates": gates,
        "passed_gate_count": sum(row["passed"] for row in gates),
        "required_gate_count": len(gates),
        "failed_gates": failed,
        "authorization": {
            "may_preregister_n2_experiment": may_preregister,
            "may_train_on_non_audit_units_after_preregistration": may_preregister,
            "may_open_locked_audit": False,
            "may_claim_algorithm_success": False,
            "may_claim_real_bost_improvement": False,
            "may_publish_private_paths_or_raw_data": False,
        },
        "privacy": {
            "source_paths_emitted": False,
            "raw_dataset_or_case_ids_emitted": False,
            "permission_values_emitted": False,
        },
        "next_actions": [row["detail"] for row in gates if not row["passed"]],
        "claim_boundary": [
            "This report validates metadata readiness only.",
            "A sealed audit split cannot be opened by this validator.",
            "A passing contract is not a reconstruction, generalization, or publication result.",
            "Synthetic field metrics and experimental reprojection metrics remain separate.",
        ],
        "private_identifiers_were_counted_but_not_emitted": bool(view_ids or sensor_ids),
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="return zero for a fully valid synthetic contract fixture",
    )
    args = parser.parse_args()
    try:
        report = validate_contract(_load(args.contract))
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is not None:
        write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] == READY_STATUS:
        return 0
    if report["status"] == FIXTURE_STATUS:
        return 0 if args.allow_fixture else 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
