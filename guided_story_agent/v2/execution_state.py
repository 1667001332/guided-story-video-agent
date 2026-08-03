"""Durable, mutable execution state kept separate from compiler artifacts.

The objects in this module are runtime records.  They deliberately do not
modify :class:`ExecutionPlan`, :class:`ExecutionUnit`, or :class:`VideoJob`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .provider_results import ProviderJobStatus
from .provider_sanitization import sanitize_response, sanitize_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


class ExecutionState(str, Enum):
    PREPARED = "PREPARED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    BLOCKED = "BLOCKED"
    CORRUPTED = "CORRUPTED"
    SUBMISSION_UNCERTAIN = "SUBMISSION_UNCERTAIN"

    @property
    def terminal(self) -> bool:
        return self in {
            self.COMPLETED,
            self.FAILED,
            self.CANCELLED,
            self.TIMED_OUT,
            self.BLOCKED,
            self.CORRUPTED,
            self.SUBMISSION_UNCERTAIN,
        }


class ExecutionRunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class RuntimeErrorRecord:
    error_id: str
    category: str
    code: str
    message: str
    retryable: bool = False
    provider_key: str | None = None
    occurred_at: str = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", sanitize_text(self.message))
        object.__setattr__(self, "metadata", _freeze(sanitize_response(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_id": self.error_id,
            "category": getattr(self.category, "value", self.category),
            "code": self.code,
            "message": self.message,
            "retryable": bool(self.retryable),
            "provider_key": self.provider_key,
            "occurred_at": self.occurred_at,
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeErrorRecord | None":
        if not data:
            return None
        return cls(
            error_id=str(data.get("error_id", "")),
            category=str(data.get("category", "")),
            code=str(data.get("code", "")),
            message=str(data.get("message", "")),
            retryable=bool(data.get("retryable", False)),
            provider_key=data.get("provider_key"),
            occurred_at=str(data.get("occurred_at", utc_now())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ProviderJob:
    """Provider-neutral remote task handle.

    ``provider_metadata`` is retained as a constructor/read compatibility
    alias for Phase 5A.  Durable serialization uses only the sanitized name.
    """

    provider_job_id: str
    provider_key: str
    remote_job_id: str
    request_id: str
    idempotency_key: str
    status: str
    source_execution_run_id: str
    source_execution_plan_id: str
    source_execution_plan_fingerprint: str
    source_execution_unit_id: str
    source_video_job_id: str
    source_video_job_fingerprint: str
    source_movie_plan_version: int
    source_movie_plan_fingerprint: str
    source_movie_plan_lineage_token: str
    schema_version: str = "provider-job/1"
    provider_profile: str = ""
    submitted_at: str = field(default_factory=utc_now)
    last_polled_at: str | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    sanitized_provider_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        status = ProviderJobStatus.from_value(self.status)
        object.__setattr__(self, "status", status)
        metadata = self.sanitized_provider_metadata
        if metadata is None:
            metadata = self.provider_metadata
        frozen = _freeze(sanitize_response(metadata))
        object.__setattr__(self, "provider_metadata", frozen)
        object.__setattr__(self, "sanitized_provider_metadata", frozen)

    @property
    def normalized_status(self) -> ProviderJobStatus:
        return ProviderJobStatus.from_value(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {key: _plain(value) for key, value in self.__dict__.items()} if not hasattr(self, "__slots__") else {
            "provider_job_id": self.provider_job_id,
            "schema_version": self.schema_version,
            "provider_key": self.provider_key,
            "provider_profile": self.provider_profile,
            "remote_job_id": self.remote_job_id,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "status": self.normalized_status.value,
            "source_execution_run_id": self.source_execution_run_id,
            "source_execution_plan_id": self.source_execution_plan_id,
            "source_execution_plan_fingerprint": self.source_execution_plan_fingerprint,
            "source_execution_unit_id": self.source_execution_unit_id,
            "source_video_job_id": self.source_video_job_id,
            "source_video_job_fingerprint": self.source_video_job_fingerprint,
            "source_movie_plan_version": self.source_movie_plan_version,
            "source_movie_plan_fingerprint": self.source_movie_plan_fingerprint,
            "source_movie_plan_lineage_token": self.source_movie_plan_lineage_token,
            "submitted_at": self.submitted_at,
            "last_polled_at": self.last_polled_at,
            "sanitized_provider_metadata": _plain(self.sanitized_provider_metadata or {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderJob":
        return cls(
            provider_job_id=str(data.get("provider_job_id", "")),
            schema_version=str(data.get("schema_version", "provider-job/1")),
            provider_key=str(data.get("provider_key", "")),
            provider_profile=str(data.get("provider_profile", "")),
            remote_job_id=str(data.get("remote_job_id", "")),
            request_id=str(data.get("request_id", "")),
            idempotency_key=str(data.get("idempotency_key", "")),
            status=ProviderJobStatus.from_value(data.get("status", "UNKNOWN")),
            source_execution_run_id=str(data.get("source_execution_run_id", "")),
            source_execution_plan_id=str(data.get("source_execution_plan_id", "")),
            source_execution_plan_fingerprint=str(data.get("source_execution_plan_fingerprint", "")),
            source_execution_unit_id=str(data.get("source_execution_unit_id", "")),
            source_video_job_id=str(data.get("source_video_job_id", "")),
            source_video_job_fingerprint=str(data.get("source_video_job_fingerprint", "")),
            source_movie_plan_version=int(data.get("source_movie_plan_version", 0)),
            source_movie_plan_fingerprint=str(data.get("source_movie_plan_fingerprint", "")),
            source_movie_plan_lineage_token=str(data.get("source_movie_plan_lineage_token", "")),
            submitted_at=str(data.get("submitted_at", utc_now())),
            last_polled_at=data.get("last_polled_at"),
            provider_metadata=dict(data.get("provider_metadata", data.get("sanitized_provider_metadata", {}))),
            sanitized_provider_metadata=dict(data.get("sanitized_provider_metadata", data.get("provider_metadata", {}))),
        )


@dataclass(frozen=True, slots=True)
class ExecutionUnitState:
    execution_unit_id: str
    video_job_id: str
    video_job_fingerprint: str
    state: ExecutionState = ExecutionState.PREPARED
    attempt: int = 0
    provider_job_id: str | None = None
    idempotency_key: str = ""
    retry_count: int = 0
    last_error: RuntimeErrorRecord | None = None
    next_retry_at: str | None = None
    artifact_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    updated_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    lease_id: str | None = None
    download_path: str | None = None

    @property
    def status(self) -> ExecutionState:
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_unit_id": self.execution_unit_id,
            "video_job_id": self.video_job_id,
            "video_job_fingerprint": self.video_job_fingerprint,
            "state": self.state.value,
            "attempt": self.attempt,
            "provider_job_id": self.provider_job_id,
            "idempotency_key": self.idempotency_key,
            "retry_count": self.retry_count,
            "last_error": self.last_error.to_dict() if self.last_error else None,
            "next_retry_at": self.next_retry_at,
            "artifact_ids": list(self.artifact_ids),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "lease_id": self.lease_id,
            "download_path": self.download_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionUnitState":
        return cls(
            execution_unit_id=str(data.get("execution_unit_id", "")),
            video_job_id=str(data.get("video_job_id", "")),
            video_job_fingerprint=str(data.get("video_job_fingerprint", "")),
            state=ExecutionState(str(data.get("state", ExecutionState.PREPARED.value))),
            attempt=int(data.get("attempt", 0)),
            provider_job_id=data.get("provider_job_id"),
            idempotency_key=str(data.get("idempotency_key", "")),
            retry_count=int(data.get("retry_count", 0)),
            last_error=RuntimeErrorRecord.from_dict(data.get("last_error")),
            next_retry_at=data.get("next_retry_at"),
            artifact_ids=tuple(str(item) for item in data.get("artifact_ids", [])),
            created_at=str(data.get("created_at", utc_now())),
            started_at=data.get("started_at"),
            updated_at=str(data.get("updated_at", utc_now())),
            completed_at=data.get("completed_at"),
            lease_id=data.get("lease_id"),
            download_path=data.get("download_path"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionRun:
    execution_run_id: str
    schema_version: str
    execution_bundle_fingerprint: str
    execution_plan_id: str
    execution_plan_version: int
    execution_plan_fingerprint: str
    source_movie_plan_id: str
    source_movie_plan_version: int
    source_movie_plan_fingerprint: str
    source_movie_plan_lineage_token: str
    status: ExecutionRunStatus = ExecutionRunStatus.CREATED
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    unit_states: Mapping[str, ExecutionUnitState] = field(default_factory=dict)
    provider_jobs: Mapping[str, ProviderJob] = field(default_factory=dict)
    submission_intents: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    retry_records: tuple[Mapping[str, Any], ...] = ()
    leases: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    artifacts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    latest_checkpoint_id: str | None = None
    last_event_id: str | None = None
    diagnostics: tuple[str, ...] = ()
    failure_reports: tuple[Mapping[str, Any], ...] = ()
    revision_requests: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_states", MappingProxyType(dict(self.unit_states)))
        object.__setattr__(self, "provider_jobs", MappingProxyType(dict(self.provider_jobs)))
        object.__setattr__(self, "submission_intents", _freeze(self.submission_intents))
        object.__setattr__(self, "retry_records", tuple(_freeze(item) for item in self.retry_records))
        object.__setattr__(self, "leases", _freeze(self.leases))
        object.__setattr__(self, "artifacts", _freeze(self.artifacts))
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))
        object.__setattr__(self, "failure_reports", tuple(_freeze(item) for item in self.failure_reports))
        object.__setattr__(self, "revision_requests", tuple(_freeze(item) for item in self.revision_requests))

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_run_id": self.execution_run_id,
            "schema_version": self.schema_version,
            "execution_bundle_fingerprint": self.execution_bundle_fingerprint,
            "execution_plan_id": self.execution_plan_id,
            "execution_plan_version": self.execution_plan_version,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "source_movie_plan_id": self.source_movie_plan_id,
            "source_movie_plan_version": self.source_movie_plan_version,
            "source_movie_plan_fingerprint": self.source_movie_plan_fingerprint,
            "source_movie_plan_lineage_token": self.source_movie_plan_lineage_token,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "unit_states": {key: value.to_dict() for key, value in self.unit_states.items()},
            "provider_jobs": {key: value.to_dict() for key, value in self.provider_jobs.items()},
            "submission_intents": _plain(self.submission_intents),
            "retry_records": _plain(self.retry_records),
            "leases": _plain(self.leases),
            "artifacts": _plain(self.artifacts),
            "latest_checkpoint_id": self.latest_checkpoint_id,
            "last_event_id": self.last_event_id,
            "diagnostics": list(self.diagnostics),
            "failure_reports": _plain(self.failure_reports),
            "revision_requests": _plain(self.revision_requests),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionRun":
        return cls(
            execution_run_id=str(data.get("execution_run_id", "")),
            schema_version=str(data.get("schema_version", "execution-run/1")),
            execution_bundle_fingerprint=str(data.get("execution_bundle_fingerprint", "")),
            execution_plan_id=str(data.get("execution_plan_id", "")),
            execution_plan_version=int(data.get("execution_plan_version", 0)),
            execution_plan_fingerprint=str(data.get("execution_plan_fingerprint", "")),
            source_movie_plan_id=str(data.get("source_movie_plan_id", "")),
            source_movie_plan_version=int(data.get("source_movie_plan_version", 0)),
            source_movie_plan_fingerprint=str(data.get("source_movie_plan_fingerprint", "")),
            source_movie_plan_lineage_token=str(data.get("source_movie_plan_lineage_token", "")),
            status=ExecutionRunStatus(str(data.get("status", ExecutionRunStatus.CREATED.value))),
            created_at=str(data.get("created_at", utc_now())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            unit_states={
                str(key): ExecutionUnitState.from_dict(value)
                for key, value in dict(data.get("unit_states", {})).items()
            },
            provider_jobs={
                str(key): ProviderJob.from_dict(value)
                for key, value in dict(data.get("provider_jobs", {})).items()
            },
            submission_intents=dict(data.get("submission_intents", {})),
            retry_records=tuple(data.get("retry_records", [])),
            leases=dict(data.get("leases", {})),
            artifacts=dict(data.get("artifacts", {})),
            latest_checkpoint_id=data.get("latest_checkpoint_id"),
            last_event_id=data.get("last_event_id"),
            diagnostics=tuple(str(item) for item in data.get("diagnostics", [])),
            failure_reports=tuple(data.get("failure_reports", [])),
            revision_requests=tuple(data.get("revision_requests", [])),
        )


__all__ = [
    "ExecutionRun",
    "ExecutionRunStatus",
    "ExecutionState",
    "ExecutionUnitState",
    "ProviderJob",
    "ProviderJobStatus",
    "RuntimeErrorRecord",
    "utc_now",
]
