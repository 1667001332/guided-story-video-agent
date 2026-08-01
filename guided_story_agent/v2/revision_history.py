"""Serializable MoviePlan version and revision history records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_FORBIDDEN_KEYS = {
    "provider", "provider_key", "provider_name", "provider_profile", "api",
    "api_key", "payload", "provider_payload", "request_payload", "video_payload",
    "api_payload", "http_payload", "endpoint", "model", "task", "task_id",
    "video_id", "submit", "poll", "download",
}
_PROMPT_STUFFING_TERMS = (
    "masterpiece", "best quality", "ultra realistic", "ultra-realistic", "8k",
)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _ensure_safe(value: Any, path: str = "revision_history") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"revision history contains forbidden field: {path}.{key}")
            _ensure_safe(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _ensure_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for term in _PROMPT_STUFFING_TERMS:
            if term in lowered:
                raise ValueError(f"revision history contains prompt stuffing at {path}: {term}")


@dataclass(frozen=True, slots=True)
class MoviePlanVersionRecord:
    movie_plan_id: str
    version: int
    source: str
    parent_movie_plan_id: str | None = None
    source_candidate_id: str | None = None
    source_decision_id: str | None = None
    created_by: str = "system"
    reason: str = ""
    snapshot: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.movie_plan_id).strip():
            raise ValueError("movie_plan_id is required")
        if isinstance(self.version, bool) or self.version < 1:
            raise ValueError("version must be a positive integer")
        if not str(self.source).strip():
            raise ValueError("source is required")
        if not isinstance(self.snapshot, dict):
            raise TypeError("snapshot must be a JSON object")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a JSON object")
        _ensure_safe(self.created_by, "movie_plan_version.created_by")
        _ensure_safe(self.reason, "movie_plan_version.reason")
        _ensure_safe(self.snapshot, "movie_plan_version.snapshot")
        _ensure_safe(self.metadata, "movie_plan_version.metadata")

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RevisionApplyRecord:
    command_id: str
    candidate_id: str
    decision_id: str | None
    previous_movie_plan_id: str
    new_movie_plan_id: str
    applied: bool
    reason: str
    invalidated_artifacts: tuple[str, ...] = ()
    revalidation_succeeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RevisionRollbackRecord:
    command_id: str
    rollback_to_movie_plan_id: str
    previous_movie_plan_id: str
    restored_movie_plan_id: str
    rolled_back: bool
    reason: str
    invalidated_artifacts: tuple[str, ...] = ()
    revalidation_succeeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


__all__ = [
    "MoviePlanVersionRecord",
    "RevisionApplyRecord",
    "RevisionRollbackRecord",
]
