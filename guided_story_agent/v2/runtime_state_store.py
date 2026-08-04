"""Durable runtime-state store implementations.

The JSON implementation intentionally stores runtime state outside Session and
outside immutable compiler artifacts.  Every write is temp-file-then-replace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .execution_state import ExecutionRun, ProviderJob, utc_now
from .provider_sanitization import sanitize_text


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True, slots=True)
class SubmissionIntent:
    submission_intent_id: str
    execution_run_id: str
    execution_unit_id: str
    video_job_id: str
    video_job_fingerprint: str
    provider_key: str
    idempotency_key: str
    status: str = "prepared"
    created_at: str = field(default_factory=utc_now)
    provider_invocation_started: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_intent_id": self.submission_intent_id,
            "execution_run_id": self.execution_run_id,
            "execution_unit_id": self.execution_unit_id,
            "video_job_id": self.video_job_id,
            "video_job_fingerprint": self.video_job_fingerprint,
            "provider_key": self.provider_key,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "created_at": self.created_at,
            "provider_invocation_started": bool(self.provider_invocation_started),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubmissionIntent":
        return cls(
            submission_intent_id=str(data.get("submission_intent_id", "")),
            execution_run_id=str(data.get("execution_run_id", "")),
            execution_unit_id=str(data.get("execution_unit_id", "")),
            video_job_id=str(data.get("video_job_id", "")),
            video_job_fingerprint=str(data.get("video_job_fingerprint", "")),
            provider_key=str(data.get("provider_key", "")),
            idempotency_key=str(data.get("idempotency_key", "")),
            status=str(data.get("status", "prepared")),
            created_at=str(data.get("created_at", utc_now())),
            provider_invocation_started=bool(data.get("provider_invocation_started", False)),
        )


@dataclass(frozen=True, slots=True)
class RetryRecord:
    retry_record_id: str
    execution_unit_id: str
    attempt: int
    error_category: str
    retryable: bool
    retry_at: str | None
    backoff_seconds: float
    reason: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retry_record_id": self.retry_record_id,
            "execution_unit_id": self.execution_unit_id,
            "attempt": self.attempt,
            "error_category": self.error_category,
            "retryable": bool(self.retryable),
            "retry_at": self.retry_at,
            "backoff_seconds": self.backoff_seconds,
            "reason": sanitize_text(self.reason),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetryRecord":
        return cls(
            retry_record_id=str(data.get("retry_record_id", "")),
            execution_unit_id=str(data.get("execution_unit_id", "")),
            attempt=int(data.get("attempt", 0)),
            error_category=str(data.get("error_category", "")),
            retryable=bool(data.get("retryable", False)),
            retry_at=data.get("retry_at"),
            backoff_seconds=float(data.get("backoff_seconds", 0.0)),
            reason=sanitize_text(str(data.get("reason", ""))),
            created_at=str(data.get("created_at", utc_now())),
        )


@dataclass(frozen=True, slots=True)
class ExecutionLease:
    lease_id: str
    execution_run_id: str
    execution_unit_id: str
    owner_id: str
    acquired_at: str
    expires_at: str

    def is_expired(self, now: str | None = None) -> bool:
        current = _parse_time(now or utc_now())
        return current >= _parse_time(self.expires_at)

    def to_dict(self) -> dict[str, str]:
        return {
            "lease_id": self.lease_id,
            "execution_run_id": self.execution_run_id,
            "execution_unit_id": self.execution_unit_id,
            "owner_id": self.owner_id,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionLease":
        return cls(
            lease_id=str(data.get("lease_id", "")),
            execution_run_id=str(data.get("execution_run_id", "")),
            execution_unit_id=str(data.get("execution_unit_id", "")),
            owner_id=str(data.get("owner_id", "")),
            acquired_at=str(data.get("acquired_at", utc_now())),
            expires_at=str(data.get("expires_at", utc_now())),
        )


class RuntimeStateStore(Protocol):
    def create_run(self, run: ExecutionRun) -> ExecutionRun: ...
    def load_run(self, execution_run_id: str) -> ExecutionRun: ...
    def save_run(self, run: ExecutionRun) -> ExecutionRun: ...
    def save_unit_state(self, run_id: str, state: Any) -> ExecutionRun: ...
    def save_provider_job(self, run_id: str, job: ProviderJob) -> ExecutionRun: ...
    def save_submission_intent(self, run_id: str, intent: SubmissionIntent) -> ExecutionRun: ...
    def save_retry_record(self, run_id: str, record: RetryRecord) -> ExecutionRun: ...
    def save_failure_report(self, run_id: str, report: Mapping[str, Any]) -> ExecutionRun: ...
    def save_revision_request(self, run_id: str, request: Mapping[str, Any]) -> ExecutionRun: ...
    def save_lease(self, run_id: str, lease: ExecutionLease) -> ExecutionRun: ...
    def remove_lease(self, run_id: str, unit_id: str) -> ExecutionRun: ...
    def save_artifact(self, run_id: str, artifact: Mapping[str, Any]) -> ExecutionRun: ...
    def list_active_runs(self) -> tuple[ExecutionRun, ...]: ...


class InMemoryRuntimeStateStore:
    def __init__(self) -> None:
        self._runs: dict[str, ExecutionRun] = {}
        self._lock = RLock()

    def create_run(self, run: ExecutionRun) -> ExecutionRun:
        with self._lock:
            if run.execution_run_id in self._runs:
                raise ValueError("execution run already exists")
            self._runs[run.execution_run_id] = run
            return run

    def load_run(self, execution_run_id: str) -> ExecutionRun:
        with self._lock:
            try:
                return self._runs[execution_run_id]
            except KeyError as exc:
                raise FileNotFoundError(f"execution run not found: {execution_run_id}") from exc

    def save_run(self, run: ExecutionRun) -> ExecutionRun:
        with self._lock:
            self._runs[run.execution_run_id] = run
            return run

    def _update(self, run_id: str, **kwargs: Any) -> ExecutionRun:
        return self.save_run(replace(self.load_run(run_id), **kwargs))

    def save_unit_state(self, run_id: str, state: Any) -> ExecutionRun:
        run = self.load_run(run_id)
        values = dict(run.unit_states)
        values[state.execution_unit_id] = state
        return self._update(run_id, unit_states=values)

    def save_provider_job(self, run_id: str, job: ProviderJob) -> ExecutionRun:
        run = self.load_run(run_id)
        values = dict(run.provider_jobs)
        values[job.provider_job_id] = job
        return self._update(run_id, provider_jobs=values)

    def save_submission_intent(self, run_id: str, intent: SubmissionIntent) -> ExecutionRun:
        run = self.load_run(run_id)
        values = dict(run.submission_intents)
        values[intent.submission_intent_id] = intent.to_dict()
        return self._update(run_id, submission_intents=values)

    def save_retry_record(self, run_id: str, record: RetryRecord) -> ExecutionRun:
        run = self.load_run(run_id)
        return self._update(run_id, retry_records=(*run.retry_records, record.to_dict()))

    def save_failure_report(self, run_id: str, report: Mapping[str, Any]) -> ExecutionRun:
        run = self.load_run(run_id)
        values = [dict(item) for item in run.failure_reports]
        failure_id = str(report.get("failure_id", ""))
        values = [item for item in values if str(item.get("failure_id", "")) != failure_id]
        values.append(dict(report))
        return self._update(run_id, failure_reports=tuple(values))

    def save_revision_request(self, run_id: str, request: Mapping[str, Any]) -> ExecutionRun:
        run = self.load_run(run_id)
        values = [dict(item) for item in run.revision_requests]
        request_id = str(request.get("request_id", ""))
        values = [item for item in values if str(item.get("request_id", "")) != request_id]
        values.append(dict(request))
        return self._update(run_id, revision_requests=tuple(values))

    def save_lease(self, run_id: str, lease: ExecutionLease) -> ExecutionRun:
        run = self.load_run(run_id)
        values = dict(run.leases)
        values[lease.execution_unit_id] = lease.to_dict()
        return self._update(run_id, leases=values)

    def remove_lease(self, run_id: str, unit_id: str) -> ExecutionRun:
        run = self.load_run(run_id)
        values = dict(run.leases)
        values.pop(unit_id, None)
        return self._update(run_id, leases=values)

    def save_artifact(self, run_id: str, artifact: Mapping[str, Any]) -> ExecutionRun:
        run = self.load_run(run_id)
        values = dict(run.artifacts)
        values[str(artifact["artifact_id"])] = dict(artifact)
        return self._update(run_id, artifacts=values)

    def list_active_runs(self) -> tuple[ExecutionRun, ...]:
        with self._lock:
            return tuple(
                run
                for run in self._runs.values()
                if run.status not in {run.status.COMPLETED, run.status.FAILED, run.status.CANCELLED, run.status.STALE}
            )


class JsonRuntimeStateStore(InMemoryRuntimeStateStore):
    """JSON files under ``root/runs`` with validated run-id path handling."""

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("execution_run_id contains unsafe path characters")
        return self.runs_root / f"{run_id}.json"

    def _atomic_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def create_run(self, run: ExecutionRun) -> ExecutionRun:
        path = self._path(run.execution_run_id)
        with self._lock:
            if path.exists():
                raise ValueError("execution run already exists")
            self._atomic_write(path, run.to_dict())
            self._runs[run.execution_run_id] = run
            return run

    def load_run(self, execution_run_id: str) -> ExecutionRun:
        path = self._path(execution_run_id)
        with self._lock:
            if path.exists():
                run = ExecutionRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self._runs[execution_run_id] = run
                return run
        return super().load_run(execution_run_id)

    def save_run(self, run: ExecutionRun) -> ExecutionRun:
        with self._lock:
            self._atomic_write(self._path(run.execution_run_id), run.to_dict())
            self._runs[run.execution_run_id] = run
            return run

    def list_active_runs(self) -> tuple[ExecutionRun, ...]:
        runs: list[ExecutionRun] = []
        for path in sorted(self.runs_root.glob("*.json")):
            runs.append(self.load_run(path.stem))
        return tuple(
            run
            for run in runs
            if run.status.value not in {"COMPLETED", "FAILED", "CANCELLED", "STALE"}
        )


__all__ = [
    "ExecutionLease",
    "InMemoryRuntimeStateStore",
    "JsonRuntimeStateStore",
    "RetryRecord",
    "RuntimeStateStore",
    "SubmissionIntent",
]
