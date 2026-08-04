"""DAG scheduling and centralized runtime state transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Mapping
from uuid import uuid4

from .execution_checkpoint import CheckpointStore, ExecutionCheckpoint
from .execution_events import ExecutionEventStore
from .execution_plan import ExecutionPlan
from .execution_state import (
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionState,
    ExecutionUnitState,
    RuntimeErrorRecord,
    utc_now,
)
from .runtime_state_store import ExecutionLease, RuntimeStateStore
from .provider_runtime import Clock, SystemClock


_ALLOWED: dict[ExecutionState, set[ExecutionState]] = {
    ExecutionState.PREPARED: {ExecutionState.SUBMITTING, ExecutionState.BLOCKED, ExecutionState.CANCELLED},
    ExecutionState.SUBMITTING: {
        ExecutionState.PREPARED,
        ExecutionState.SUBMITTED,
        ExecutionState.FAILED,
        ExecutionState.BLOCKED,
        ExecutionState.SUBMISSION_UNCERTAIN,
        ExecutionState.CANCELLED,
        ExecutionState.TIMED_OUT,
    },
    ExecutionState.SUBMITTED: {
        ExecutionState.QUEUED,
        ExecutionState.RUNNING,
        ExecutionState.FAILED,
        ExecutionState.BLOCKED,
        ExecutionState.CANCELLED,
        ExecutionState.TIMED_OUT,
    },
    ExecutionState.QUEUED: {
        ExecutionState.RUNNING,
        ExecutionState.FAILED,
        ExecutionState.BLOCKED,
        ExecutionState.CANCELLED,
        ExecutionState.TIMED_OUT,
    },
    ExecutionState.RUNNING: {
        ExecutionState.DOWNLOADING,
        ExecutionState.FAILED,
        ExecutionState.BLOCKED,
        ExecutionState.CANCELLED,
        ExecutionState.TIMED_OUT,
    },
    ExecutionState.DOWNLOADING: {
        ExecutionState.PREPARED,
        ExecutionState.DOWNLOADED,
        ExecutionState.FAILED,
        ExecutionState.BLOCKED,
        ExecutionState.CANCELLED,
        ExecutionState.TIMED_OUT,
    },
    ExecutionState.DOWNLOADED: {ExecutionState.VERIFYING, ExecutionState.CANCELLED},
    ExecutionState.VERIFYING: {
        ExecutionState.VERIFIED,
        ExecutionState.CORRUPTED,
        ExecutionState.FAILED,
        ExecutionState.DOWNLOADING,
        ExecutionState.BLOCKED,
        ExecutionState.CANCELLED,
    },
    ExecutionState.VERIFIED: {ExecutionState.COMPLETED, ExecutionState.CANCELLED},
    ExecutionState.SUBMISSION_UNCERTAIN: {ExecutionState.CANCELLED},
}


class TransitionRejectedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeTransition:
    event_id: str
    execution_run_id: str
    execution_unit_id: str
    from_state: ExecutionState
    to_state: ExecutionState
    event_type: str
    attempt: int
    reason: str
    metadata: Mapping[str, Any]
    occurred_at: str
    run: ExecutionRun | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "execution_run_id": self.execution_run_id,
            "execution_unit_id": self.execution_unit_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "event_type": self.event_type,
            "attempt": self.attempt,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "occurred_at": self.occurred_at,
        }


def _run_status(states: Mapping[str, ExecutionUnitState], current: ExecutionRunStatus) -> ExecutionRunStatus:
    values = [state.state for state in states.values()]
    if values and all(state == ExecutionState.COMPLETED for state in values):
        return ExecutionRunStatus.COMPLETED
    if any(state == ExecutionState.SUBMISSION_UNCERTAIN for state in values):
        return ExecutionRunStatus.BLOCKED
    if any(state == ExecutionState.BLOCKED for state in values) and not any(
        state in {ExecutionState.PREPARED, ExecutionState.SUBMITTING, ExecutionState.SUBMITTED, ExecutionState.QUEUED, ExecutionState.RUNNING, ExecutionState.DOWNLOADING, ExecutionState.DOWNLOADED, ExecutionState.VERIFYING, ExecutionState.VERIFIED}
        for state in values
    ):
        return ExecutionRunStatus.BLOCKED
    if any(state in {ExecutionState.FAILED, ExecutionState.CORRUPTED, ExecutionState.TIMED_OUT} for state in values):
        return ExecutionRunStatus.FAILED
    if values and all(state == ExecutionState.CANCELLED for state in values):
        return ExecutionRunStatus.CANCELLED
    if any(state != ExecutionState.PREPARED for state in values):
        return ExecutionRunStatus.RUNNING
    return current


class RuntimeTransitionService:
    def __init__(
        self,
        state_store: RuntimeStateStore,
        event_store: ExecutionEventStore,
        checkpoint_store: CheckpointStore | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.state_store = state_store
        self.event_store = event_store
        self.checkpoint_store = checkpoint_store
        self.clock = clock or SystemClock()

    def transition(
        self,
        run: ExecutionRun,
        execution_unit_id: str,
        to_state: ExecutionState | str,
        *,
        event_type: str = "state_transition",
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        checkpoint: bool = True,
    ) -> RuntimeTransition:
        current_run = self.state_store.load_run(run.execution_run_id)
        unit = current_run.unit_states.get(execution_unit_id)
        if unit is None:
            raise TransitionRejectedError(f"unknown execution unit: {execution_unit_id}")
        target = to_state if isinstance(to_state, ExecutionState) else ExecutionState(str(to_state))
        events = self.event_store.list_events(current_run.execution_run_id)
        if event_id:
            for existing in events:
                if existing.event_id == event_id:
                    return RuntimeTransition(
                        event_id=existing.event_id,
                        execution_run_id=current_run.execution_run_id,
                        execution_unit_id=execution_unit_id,
                        from_state=unit.state,
                        to_state=unit.state,
                        event_type=existing.event_type,
                        attempt=unit.attempt,
                        reason=reason,
                        metadata=existing.payload,
                        occurred_at=existing.occurred_at,
                        run=current_run,
                    )
        if target == unit.state:
            return RuntimeTransition(
                event_id=event_id or f"idempotent-{execution_unit_id}-{target.value}",
                execution_run_id=current_run.execution_run_id,
                execution_unit_id=execution_unit_id,
                from_state=unit.state,
                to_state=target,
                event_type=event_type,
                attempt=unit.attempt,
                reason=reason,
                metadata=dict(metadata or {}),
                occurred_at=utc_now(),
                run=current_run,
            )
        if target not in _ALLOWED.get(unit.state, set()):
            raise TransitionRejectedError(
                f"illegal execution transition: {unit.state.value} -> {target.value}"
            )
        now = self.clock.now().isoformat()
        values = dict(metadata or {})
        error = values.get("last_error")
        if isinstance(error, Mapping):
            error = RuntimeErrorRecord.from_dict(error)
        updated_unit = replace(
            unit,
            state=target,
            updated_at=now,
            started_at=unit.started_at or (now if target != ExecutionState.PREPARED else None),
            completed_at=now if target == ExecutionState.COMPLETED else unit.completed_at,
            last_error=error if error is not None else unit.last_error,
            retry_count=int(values.get("retry_count", unit.retry_count)),
            next_retry_at=values.get("next_retry_at", unit.next_retry_at),
            provider_job_id=values.get("provider_job_id", unit.provider_job_id),
            artifact_ids=tuple(values.get("artifact_ids", unit.artifact_ids)),
            download_path=values.get("download_path", unit.download_path),
            lease_id=values.get("lease_id", unit.lease_id),
        )
        updated_states = dict(current_run.unit_states)
        updated_states[execution_unit_id] = updated_unit
        status = _run_status(updated_states, current_run.status)
        updated_run = replace(
            current_run,
            status=status,
            started_at=current_run.started_at or now,
            completed_at=now if status in {ExecutionRunStatus.COMPLETED, ExecutionRunStatus.CANCELLED} else current_run.completed_at,
            unit_states=updated_states,
        )
        event = self.event_store.append(
            current_run.execution_run_id,
            event_type,
            execution_unit_id=execution_unit_id,
            payload={
                "from_state": unit.state.value,
                "to_state": target.value,
                "attempt": updated_unit.attempt,
                "reason": reason,
                **values,
            },
            event_id=event_id,
        )
        updated_run = replace(updated_run, last_event_id=event.event_id)
        self.state_store.save_run(updated_run)
        if checkpoint and self.checkpoint_store is not None:
            checkpoint_record = ExecutionCheckpoint.from_run(updated_run)
            self.checkpoint_store.save(checkpoint_record)
            updated_run = replace(updated_run, latest_checkpoint_id=checkpoint_record.checkpoint_id)
            self.state_store.save_run(updated_run)
            self.event_store.append(
                updated_run.execution_run_id,
                "checkpoint_created",
                execution_unit_id=execution_unit_id,
                payload={"checkpoint_id": checkpoint_record.checkpoint_id},
            )
        return RuntimeTransition(
            event_id=event.event_id,
            execution_run_id=current_run.execution_run_id,
            execution_unit_id=execution_unit_id,
            from_state=unit.state,
            to_state=target,
            event_type=event_type,
            attempt=updated_unit.attempt,
            reason=reason,
            metadata=values,
            occurred_at=event.occurred_at,
            run=updated_run,
        )

    def mark_run_stale(self, run: ExecutionRun, reason: str) -> ExecutionRun:
        updated = replace(run, status=ExecutionRunStatus.STALE, diagnostics=(*run.diagnostics, reason))
        self.state_store.save_run(updated)
        self.event_store.append(run.execution_run_id, "runtime_marked_stale", payload={"reason": reason})
        return updated


class DependencyResolver:
    FAILURE_STATES = {
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
        ExecutionState.TIMED_OUT,
        ExecutionState.CORRUPTED,
        ExecutionState.SUBMISSION_UNCERTAIN,
        ExecutionState.BLOCKED,
    }

    def __init__(self, plan: ExecutionPlan, *, clock: Clock | None = None) -> None:
        self.plan = plan
        self.clock = clock or SystemClock()
        self._incoming: dict[str, list[Any]] = {unit.execution_unit_id: [] for unit in plan.execution_units}
        for edge in plan.dependency_graph:
            self._incoming.setdefault(edge.to_unit_id, []).append(edge)

    def ready_units(self, run: ExecutionRun) -> tuple[str, ...]:
        now = self.clock.now().isoformat()
        values: list[str] = []
        for unit in self.plan.execution_units:
            state = run.unit_states.get(unit.execution_unit_id)
            if state is None or state.state != ExecutionState.PREPARED:
                continue
            if state.next_retry_at and state.next_retry_at > now:
                continue
            if state.lease_id or unit.execution_unit_id in run.leases:
                continue
            dependencies = self._incoming.get(unit.execution_unit_id, [])
            if any(run.unit_states.get(edge.from_unit_id, state).state != ExecutionState.COMPLETED for edge in dependencies):
                continue
            required_refs = [item for item in unit.reference_inputs if item.required]
            if any(
                not run.unit_states.get(item.source_unit_id)
                or not run.unit_states[item.source_unit_id].artifact_ids
                or any(
                    run.artifacts.get(artifact_id, {}).get("verification_status") != "verified"
                    for artifact_id in run.unit_states[item.source_unit_id].artifact_ids
                )
                for item in required_refs
                if item.source_unit_id
            ):
                continue
            values.append(unit.execution_unit_id)
        return tuple(values)

    def failed_dependency_units(self, run: ExecutionRun) -> tuple[str, ...]:
        blocked: list[str] = []
        for unit in self.plan.execution_units:
            state = run.unit_states[unit.execution_unit_id]
            if state.state != ExecutionState.PREPARED:
                continue
            for edge in self._incoming.get(unit.execution_unit_id, []):
                source = run.unit_states.get(edge.from_unit_id)
                if source is not None and source.state in self.FAILURE_STATES:
                    blocked.append(unit.execution_unit_id)
                    break
        return tuple(blocked)

    def active_count(self, run: ExecutionRun) -> int:
        return sum(
            state.state in {
                ExecutionState.SUBMITTING,
                ExecutionState.SUBMITTED,
                ExecutionState.QUEUED,
                ExecutionState.RUNNING,
                ExecutionState.DOWNLOADING,
                ExecutionState.DOWNLOADED,
                ExecutionState.VERIFYING,
                ExecutionState.VERIFIED,
            }
            for state in run.unit_states.values()
        )

    def acquire_lease(
        self,
        run: ExecutionRun,
        unit_id: str,
        *,
        owner_id: str,
        ttl_seconds: float = 300.0,
        clock: Clock | None = None,
    ) -> ExecutionLease:
        current = clock or self.clock
        existing = run.leases.get(unit_id)
        if existing and not ExecutionLease.from_dict(existing).is_expired(current.now().isoformat()):
            raise RuntimeError(f"execution unit already has a valid lease: {unit_id}")
        acquired = current.now().isoformat()
        expires = (current.now() + timedelta(seconds=ttl_seconds)).isoformat()
        lease = ExecutionLease(
            lease_id=f"lease-{uuid4().hex}",
            execution_run_id=run.execution_run_id,
            execution_unit_id=unit_id,
            owner_id=owner_id,
            acquired_at=acquired,
            expires_at=expires,
        )
        return lease


__all__ = [
    "DependencyResolver",
    "RuntimeTransition",
    "RuntimeTransitionService",
    "TransitionRejectedError",
]
