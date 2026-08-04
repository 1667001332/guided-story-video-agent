"""Canonical fingerprints for VideoJob, ExecutionUnit, Plan, and Bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .execution import VideoJob
    from .execution_bundle import ExecutionBundle
    from .execution_plan import ExecutionPlan, ExecutionUnit


_RUNTIME_KEYS = {
    "created_at",
    "updated_at",
    "timestamp",
    "timestamps",
    "runtime",
    "runtime_status",
    "status",
    "retry_count",
    "last_error",
    "provider_task_id",
    "task_id",
    "provider_response",
    "remote_response",
    "download_path",
    "download_progress",
    "local_cache_path",
    "diagnostics",
}
_PLAN_ID_KEYS = {
    "execution_plan_id",
    "execution_plan_version",
    "execution_plan_fingerprint",
}
_UNIT_ID_KEYS = {"execution_unit_id", "execution_unit_fingerprint"}
_VIDEO_JOB_ID_KEYS = {"job_id", "video_job_fingerprint"}
_FORBIDDEN_KEYS = {
    "api_key",
    "authorization",
    "bearer",
    "endpoint",
    "http_body",
    "http_payload",
    "payload",
    "provider_task_id",
    "remote_response",
    "secret",
    "task_id",
}


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)
    return value


def _canonical(value: Any, *, key: str = "", excluded: set[str] | None = None) -> Any:
    excluded = excluded or set()
    lowered = key.lower()
    if lowered in excluded or lowered in _RUNTIME_KEYS:
        return None
    if isinstance(value, Mapping):
        return {
            str(child_key): _canonical(child, key=str(child_key), excluded=excluded)
            for child_key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if str(child_key).lower() not in excluded
            and str(child_key).lower() not in _RUNTIME_KEYS
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item, excluded=excluded) for item in value]
    return _plain(value)


def canonical_json(value: Any, *, excluded: set[str] | None = None) -> str:
    return json.dumps(
        _canonical(_plain(value), excluded=excluded),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_canonical(value: Any, *, excluded: set[str] | None = None) -> str:
    return hashlib.sha256(canonical_json(value, excluded=excluded).encode("utf-8")).hexdigest()


def video_job_fingerprint(video_job: "VideoJob | Mapping[str, Any]") -> str:
    """Fingerprint immutable provider input, excluding identity/runtime data."""

    return sha256_canonical(video_job, excluded=_VIDEO_JOB_ID_KEYS)


def execution_unit_fingerprint(unit: "ExecutionUnit | Mapping[str, Any]") -> str:
    return sha256_canonical(unit, excluded=_UNIT_ID_KEYS)


def execution_plan_fingerprint(plan: "ExecutionPlan | Mapping[str, Any]") -> str:
    """Fingerprint plan content while excluding plan identity and runtime data."""

    return sha256_canonical(plan, excluded=_PLAN_ID_KEYS)


def execution_bundle_fingerprint(bundle: "ExecutionBundle | Mapping[str, Any]") -> str:
    if hasattr(bundle, "execution_plan"):
        plan = bundle.execution_plan
        jobs = bundle.video_jobs
        value = {
            "execution_plan_fingerprint": plan.execution_plan_fingerprint,
            "video_jobs": [
                {"video_job_id": job.job_id, "video_job_fingerprint": job.video_job_fingerprint}
                for job in sorted(jobs, key=lambda item: item.job_id)
            ],
        }
    else:
        raw = dict(bundle)
        plan = raw.get("execution_plan", {})
        jobs = raw.get("video_jobs", [])
        value = {
            "execution_plan_fingerprint": plan.get("execution_plan_fingerprint", ""),
            "video_jobs": sorted(
                (
                    {
                        "video_job_id": item.get("job_id", item.get("video_job_id", "")),
                        "video_job_fingerprint": item.get("video_job_fingerprint", ""),
                    }
                    for item in jobs
                ),
                key=lambda item: item["video_job_id"],
            ),
        }
    return sha256_canonical(value)


def contains_forbidden_provider_data(value: Any, path: str = "") -> tuple[str, ...]:
    """Return all forbidden key paths, including nested metadata entries."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower().replace("-", "_").replace(" ", "_")
            child_path = f"{path}.{key}" if path else str(key)
            if lowered in _FORBIDDEN_KEYS:
                found.append(child_path)
            found.extend(contains_forbidden_provider_data(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(contains_forbidden_provider_data(child, f"{path}[{index}]"))
    return tuple(found)


__all__ = [
    "canonical_json",
    "contains_forbidden_provider_data",
    "execution_bundle_fingerprint",
    "execution_plan_fingerprint",
    "execution_unit_fingerprint",
    "sha256_canonical",
    "video_job_fingerprint",
]
