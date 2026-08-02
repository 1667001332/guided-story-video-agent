"""Append-only event journal for the offline execution runtime."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .execution_state import utc_now
from .provider_sanitization import REDACTED, SENSITIVE_KEYS, sanitize_response


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SECRET_WORDS = set(SENSITIVE_KEYS) | {"bearer"}


def _ensure_safe_payload(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (normalized in _SECRET_WORDS or normalized.endswith("_api_key")) and child != REDACTED:
                raise ValueError(f"event payload contains forbidden secret field: {path}.{key}")
            _ensure_safe_payload(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_safe_payload(child, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event_id: str
    event_type: str
    execution_run_id: str
    execution_unit_id: str | None = None
    sequence_number: int = 0
    occurred_at: str = field(default_factory=utc_now)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.event_type.strip() or not self.execution_run_id.strip():
            raise ValueError("RuntimeEvent identifiers are required")
        if self.sequence_number < 1:
            raise ValueError("RuntimeEvent.sequence_number must be positive")
        _ensure_safe_payload(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "execution_run_id": self.execution_run_id,
            "execution_unit_id": self.execution_unit_id,
            "sequence_number": self.sequence_number,
            "occurred_at": self.occurred_at,
            "payload": json.loads(json.dumps(dict(self.payload), ensure_ascii=False, default=str)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeEvent":
        return cls(
            event_id=str(data.get("event_id", "")),
            event_type=str(data.get("event_type", "")),
            execution_run_id=str(data.get("execution_run_id", "")),
            execution_unit_id=data.get("execution_unit_id"),
            sequence_number=int(data.get("sequence_number", 0)),
            occurred_at=str(data.get("occurred_at", utc_now())),
            payload=dict(data.get("payload", {})),
        )


class ExecutionEventStore(Protocol):
    def append(
        self,
        execution_run_id: str,
        event_type: str,
        *,
        execution_unit_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> RuntimeEvent: ...

    def list_events(self, execution_run_id: str) -> tuple[RuntimeEvent, ...]: ...


class InMemoryExecutionEventStore:
    def __init__(self) -> None:
        self._events: dict[str, list[RuntimeEvent]] = {}
        self._lock = RLock()

    def append(
        self,
        execution_run_id: str,
        event_type: str,
        *,
        execution_unit_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> RuntimeEvent:
        with self._lock:
            events = self._events.setdefault(execution_run_id, [])
            if event_id:
                for existing in events:
                    if existing.event_id == event_id:
                        return existing
            event = RuntimeEvent(
                event_id=event_id or f"event-{uuid4().hex}",
                event_type=event_type,
                execution_run_id=execution_run_id,
                execution_unit_id=execution_unit_id,
                sequence_number=len(events) + 1,
            payload=sanitize_response(dict(payload or {})),
            )
            events.append(event)
            return event

    def list_events(self, execution_run_id: str) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events.get(execution_run_id, ()))


class JsonExecutionEventStore:
    """UTF-8 JSONL event store with monotonic sequence numbers."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, execution_run_id: str) -> Path:
        if not _SAFE_ID.fullmatch(execution_run_id):
            raise ValueError("execution_run_id contains unsafe path characters")
        return self.root / f"{execution_run_id}.jsonl"

    def append(
        self,
        execution_run_id: str,
        event_type: str,
        *,
        execution_unit_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> RuntimeEvent:
        with self._lock:
            events = self.list_events(execution_run_id)
            if event_id:
                for existing in events:
                    if existing.event_id == event_id:
                        return existing
            event = RuntimeEvent(
                event_id=event_id or f"event-{uuid4().hex}",
                event_type=event_type,
                execution_run_id=execution_run_id,
                execution_unit_id=execution_unit_id,
                sequence_number=len(events) + 1,
                payload=sanitize_response(dict(payload or {})),
            )
            path = self._path(execution_run_id)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            return event

    def list_events(self, execution_run_id: str) -> tuple[RuntimeEvent, ...]:
        path = self._path(execution_run_id)
        if not path.exists():
            return ()
        records: list[RuntimeEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(RuntimeEvent.from_dict(json.loads(line)))
        for expected, event in enumerate(records, start=1):
            if event.sequence_number != expected:
                raise ValueError("execution event sequence is not monotonic")
        return tuple(records)


__all__ = [
    "ExecutionEventStore",
    "InMemoryExecutionEventStore",
    "JsonExecutionEventStore",
    "RuntimeEvent",
]
