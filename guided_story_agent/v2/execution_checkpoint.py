"""Immutable checkpoints and atomic checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .execution_state import ExecutionRun, utc_now


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class ExecutionCheckpoint:
    checkpoint_id: str
    schema_version: str
    execution_run_id: str
    execution_bundle_fingerprint: str
    execution_plan_id: str
    execution_plan_version: int
    execution_plan_fingerprint: str
    unit_states: tuple[Mapping[str, Any], ...] = ()
    provider_jobs: tuple[Mapping[str, Any], ...] = ()
    submission_intents: tuple[Mapping[str, Any], ...] = ()
    retry_records: tuple[Mapping[str, Any], ...] = ()
    artifact_references: tuple[Mapping[str, Any], ...] = ()
    active_leases: tuple[Mapping[str, Any], ...] = ()
    scheduler_cursor: str | None = None
    last_event_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    checkpoint_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip() or not self.execution_run_id.strip():
            raise ValueError("checkpoint identifiers are required")
        for name in (
            "unit_states",
            "provider_jobs",
            "submission_intents",
            "retry_records",
            "artifact_references",
            "active_leases",
        ):
            object.__setattr__(self, name, tuple(dict(item) for item in getattr(self, name)))
        if not self.checkpoint_fingerprint:
            object.__setattr__(self, "checkpoint_fingerprint", self.compute_fingerprint())

    def _fingerprint_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("checkpoint_fingerprint", None)
        return payload

    def compute_fingerprint(self) -> str:
        return hashlib.sha256(_canonical(self._fingerprint_payload()).encode("utf-8")).hexdigest()

    def validate_fingerprint(self) -> bool:
        return self.checkpoint_fingerprint == self.compute_fingerprint()

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "schema_version": self.schema_version,
            "execution_run_id": self.execution_run_id,
            "execution_bundle_fingerprint": self.execution_bundle_fingerprint,
            "execution_plan_id": self.execution_plan_id,
            "execution_plan_version": self.execution_plan_version,
            "execution_plan_fingerprint": self.execution_plan_fingerprint,
            "unit_states": [dict(item) for item in self.unit_states],
            "provider_jobs": [dict(item) for item in self.provider_jobs],
            "submission_intents": [dict(item) for item in self.submission_intents],
            "retry_records": [dict(item) for item in self.retry_records],
            "artifact_references": [dict(item) for item in self.artifact_references],
            "active_leases": [dict(item) for item in self.active_leases],
            "scheduler_cursor": self.scheduler_cursor,
            "last_event_id": self.last_event_id,
            "created_at": self.created_at,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
        }

    @classmethod
    def from_run(cls, run: ExecutionRun, *, checkpoint_id: str | None = None, scheduler_cursor: str | None = None) -> "ExecutionCheckpoint":
        return cls(
            checkpoint_id=checkpoint_id or f"checkpoint-{uuid4().hex}",
            schema_version="execution-checkpoint/1",
            execution_run_id=run.execution_run_id,
            execution_bundle_fingerprint=run.execution_bundle_fingerprint,
            execution_plan_id=run.execution_plan_id,
            execution_plan_version=run.execution_plan_version,
            execution_plan_fingerprint=run.execution_plan_fingerprint,
            unit_states=tuple(state.to_dict() for state in run.unit_states.values()),
            provider_jobs=tuple(job.to_dict() for job in run.provider_jobs.values()),
            submission_intents=tuple(dict(item) for item in run.submission_intents.values()),
            retry_records=tuple(dict(item) for item in run.retry_records),
            artifact_references=tuple(dict(item) for item in run.artifacts.values()),
            active_leases=tuple(dict(item) for item in run.leases.values()),
            scheduler_cursor=scheduler_cursor,
            last_event_id=run.last_event_id,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionCheckpoint":
        return cls(
            checkpoint_id=str(data.get("checkpoint_id", "")),
            schema_version=str(data.get("schema_version", "execution-checkpoint/1")),
            execution_run_id=str(data.get("execution_run_id", "")),
            execution_bundle_fingerprint=str(data.get("execution_bundle_fingerprint", "")),
            execution_plan_id=str(data.get("execution_plan_id", "")),
            execution_plan_version=int(data.get("execution_plan_version", 0)),
            execution_plan_fingerprint=str(data.get("execution_plan_fingerprint", "")),
            unit_states=tuple(dict(item) for item in data.get("unit_states", [])),
            provider_jobs=tuple(dict(item) for item in data.get("provider_jobs", [])),
            submission_intents=tuple(dict(item) for item in data.get("submission_intents", [])),
            retry_records=tuple(dict(item) for item in data.get("retry_records", [])),
            artifact_references=tuple(dict(item) for item in data.get("artifact_references", [])),
            active_leases=tuple(dict(item) for item in data.get("active_leases", [])),
            scheduler_cursor=data.get("scheduler_cursor"),
            last_event_id=data.get("last_event_id"),
            created_at=str(data.get("created_at", utc_now())),
            checkpoint_fingerprint=str(data.get("checkpoint_fingerprint", "")),
        )


class CheckpointStore(Protocol):
    def save(self, checkpoint: ExecutionCheckpoint) -> ExecutionCheckpoint: ...
    def load(self, checkpoint_id: str) -> ExecutionCheckpoint: ...
    def latest(self, execution_run_id: str) -> ExecutionCheckpoint | None: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, ExecutionCheckpoint] = {}

    def save(self, checkpoint: ExecutionCheckpoint) -> ExecutionCheckpoint:
        if checkpoint.checkpoint_id in self._checkpoints:
            raise ValueError("checkpoint IDs are immutable and cannot be overwritten")
        if not checkpoint.validate_fingerprint():
            raise ValueError("checkpoint fingerprint mismatch")
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    def load(self, checkpoint_id: str) -> ExecutionCheckpoint:
        try:
            return self._checkpoints[checkpoint_id]
        except KeyError as exc:
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_id}") from exc

    def latest(self, execution_run_id: str) -> ExecutionCheckpoint | None:
        values = [item for item in self._checkpoints.values() if item.execution_run_id == execution_run_id]
        return values[-1] if values else None


class JsonCheckpointStore(InMemoryCheckpointStore):
    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_id: str) -> Path:
        if not _SAFE_ID.fullmatch(checkpoint_id):
            raise ValueError("checkpoint_id contains unsafe path characters")
        return self.root / f"{checkpoint_id}.json"

    def save(self, checkpoint: ExecutionCheckpoint) -> ExecutionCheckpoint:
        if not checkpoint.validate_fingerprint():
            raise ValueError("checkpoint fingerprint mismatch")
        path = self._path(checkpoint.checkpoint_id)
        if path.exists():
            raise ValueError("checkpoint IDs are immutable and cannot be overwritten")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    def load(self, checkpoint_id: str) -> ExecutionCheckpoint:
        path = self._path(checkpoint_id)
        if not path.exists():
            return super().load(checkpoint_id)
        checkpoint = ExecutionCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if not checkpoint.validate_fingerprint():
            raise ValueError("checkpoint fingerprint mismatch")
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    def latest(self, execution_run_id: str) -> ExecutionCheckpoint | None:
        values: list[ExecutionCheckpoint] = []
        for path in sorted(self.root.glob("checkpoint-*.json"), key=lambda item: item.stat().st_mtime_ns):
            try:
                checkpoint = self.load(path.stem)
            except (ValueError, json.JSONDecodeError):
                continue
            if checkpoint.execution_run_id == execution_run_id:
                values.append(checkpoint)
        return values[-1] if values else super().latest(execution_run_id)


__all__ = [
    "CheckpointStore",
    "ExecutionCheckpoint",
    "InMemoryCheckpointStore",
    "JsonCheckpointStore",
]
