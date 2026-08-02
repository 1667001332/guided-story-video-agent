"""Immutable binding between an ExecutionPlan and its VideoJobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .execution import VideoJob
from .execution_fingerprint import execution_bundle_fingerprint
from .execution_plan import ExecutionPlan


@dataclass(frozen=True, slots=True)
class ExecutionBundle:
    execution_plan: ExecutionPlan
    video_jobs: tuple[VideoJob, ...]
    bundle_fingerprint: str = ""
    schema_version: str = "execution-bundle/1"

    def __post_init__(self) -> None:
        if not isinstance(self.execution_plan, ExecutionPlan):
            raise TypeError("ExecutionBundle.execution_plan must be ExecutionPlan")
        object.__setattr__(self, "video_jobs", tuple(self.video_jobs))
        if not self.bundle_fingerprint:
            object.__setattr__(self, "bundle_fingerprint", execution_bundle_fingerprint(self))

    @property
    def video_job_map(self) -> Mapping[str, VideoJob]:
        return {job.job_id: job for job in self.video_jobs}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_plan": self.execution_plan.to_dict(),
            "video_jobs": [job.to_dict() for job in self.video_jobs],
            "bundle_fingerprint": self.bundle_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionBundle":
        if not isinstance(data, Mapping):
            raise ValueError("execution_bundle must be an object")
        return cls(
            schema_version=str(data.get("schema_version", "execution-bundle/1")),
            execution_plan=ExecutionPlan.from_dict(data.get("execution_plan", {})),
            video_jobs=tuple(_video_job_from_dict(item) for item in data.get("video_jobs", [])),
            bundle_fingerprint=str(data.get("bundle_fingerprint", "")),
        )


def _video_job_from_dict(data: Mapping[str, Any]) -> VideoJob:
    if not isinstance(data, Mapping):
        raise ValueError("ExecutionBundle.video_jobs entries must be objects")
    values = dict(data)
    for key in ("references", "character_references", "continuity_references"):
        values[key] = tuple(str(item) for item in values.get(key, []))
    values["execution_units"] = tuple(dict(item) for item in values.get("execution_units", []))
    values["metadata"] = dict(values.get("metadata", {}))
    values["duration_seconds"] = float(values.get("duration_seconds", 0.0))
    values["fps"] = None if values.get("fps") is None else float(values["fps"])
    values["confirmed"] = bool(values.get("confirmed", False))
    allowed = set(VideoJob.__dataclass_fields__)
    return VideoJob(**{key: values[key] for key in values if key in allowed})


__all__ = ["ExecutionBundle"]
