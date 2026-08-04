"""Offline durable execution runtime for an immutable ExecutionBundle.

This module is intentionally the only orchestration entrypoint for Phase 5A.
It accepts an ``ExecutionBundle`` and never recompiles creative artifacts or
constructs a real provider request.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .execution_bundle import ExecutionBundle
from .execution_checkpoint import CheckpointStore, ExecutionCheckpoint, InMemoryCheckpointStore
from .execution_events import (
    EXECUTION_ARTIFACTS_INVALIDATED,
    EXECUTION_BLOCKED_SUBMISSION_UNCERTAIN,
    EXECUTION_RETRY_SCHEDULED,
    PROVIDER_FAILURE_CLASSIFIED,
    REVISION_APPLIED,
    REVISION_REQUESTED,
    ExecutionEventStore,
    InMemoryExecutionEventStore,
)
from .execution_plan_validation import validate_execution_bundle
from .execution_scheduler import DependencyResolver, RuntimeTransitionService
from .execution_state import (
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionState,
    ExecutionUnitState,
    ProviderJob,
    RuntimeErrorRecord,
)
from .fake_artifact_verifier import ArtifactRecord, FakeArtifactVerifier
from .provider_registry import ProviderRuntimeRegistry
from .provider_capabilities import capability_snapshot_diagnostics
from .provider_errors import ProviderErrorCategory
from .failure_protocol import FailureAction, FailureProtocol
from .provider_results import (
    DownloadDestination,
    ProviderCancelResult,
    ProviderDownloadResult,
    ProviderJobStatus,
    ProviderPollResult,
    ProviderSubmitResult,
    ProviderVerificationResult,
)
from .provider_runtime import (
    Clock,
    FakeClock,
    ProviderRequestContext,
    ProviderRuntime,
    ProviderRuntimeError,
    SystemClock,
)
from .runtime_state_store import (
    ExecutionLease,
    InMemoryRuntimeStateStore,
    JsonRuntimeStateStore,
    RetryRecord,
    RuntimeStateStore,
    SubmissionIntent,
)


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


class ExecutionRuntimeError(RuntimeError):
    pass


class StaleExecutionRuntimeError(ExecutionRuntimeError):
    pass


class RuntimeNotFoundError(ExecutionRuntimeError):
    pass


def _idempotency_key(bundle: ExecutionBundle, unit_id: str, video_job_fingerprint: str, provider_key: str) -> str:
    material = "|".join((bundle.bundle_fingerprint, unit_id, video_job_fingerprint, provider_key))
    return "idem-" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _parse_iso(value: str) -> Any:
    from datetime import datetime, timezone

    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class ExecutionRuntime:
    def __init__(
        self,
        execution_bundle: ExecutionBundle | None = None,
        *,
        state_store: RuntimeStateStore | None = None,
        event_store: ExecutionEventStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        provider_registry: ProviderRuntimeRegistry | None = None,
        artifact_verifier: FakeArtifactVerifier | None = None,
        artifact_root: str | Path | None = None,
        clock: Clock | None = None,
        owner_id: str = "offline-runtime",
        failure_protocol: FailureProtocol | None = None,
    ) -> None:
        self.execution_bundle = execution_bundle
        self.clock = clock or SystemClock()
        self.owner_id = owner_id
        self.failure_protocol = failure_protocol or FailureProtocol()
        self.provider_registry = provider_registry or ProviderRuntimeRegistry()
        self.artifact_verifier = artifact_verifier or FakeArtifactVerifier()
        self.capability_diagnostics: tuple[dict[str, str], ...] = ()
        if state_store is None and artifact_root is not None:
            root = Path(artifact_root)
            state_store = JsonRuntimeStateStore(root / "state")
            event_store = event_store or __import__(
                "guided_story_agent.v2.execution_events", fromlist=["JsonExecutionEventStore"]
            ).JsonExecutionEventStore(root / "events")
            checkpoint_store = checkpoint_store or __import__(
                "guided_story_agent.v2.execution_checkpoint", fromlist=["JsonCheckpointStore"]
            ).JsonCheckpointStore(root / "checkpoints")
        self.state_store = state_store or InMemoryRuntimeStateStore()
        self.event_store = event_store or InMemoryExecutionEventStore()
        self.checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self.artifact_root = Path(artifact_root or Path.cwd() / ".offline-runtime-artifacts")
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.transition_service = RuntimeTransitionService(
            self.state_store,
            self.event_store,
            self.checkpoint_store,
            clock=self.clock,
        )

    def _bundle_or_raise(self, bundle: ExecutionBundle | None = None) -> ExecutionBundle:
        current = bundle or self.execution_bundle
        if current is None:
            raise ExecutionRuntimeError("ExecutionBundle 是 Runtime 的唯一输入，当前未提供。")
        validation = validate_execution_bundle(current)
        if not validation.valid:
            raise ExecutionRuntimeError(
                "ExecutionBundle 无效，Runtime fail-closed："
                + "；".join(item.message for item in validation.diagnostics)
            )
        return current

    def _require_provider_capabilities(self, bundle: ExecutionBundle) -> None:
        diagnostics: list[dict[str, str]] = []
        plan = bundle.execution_plan
        assignments = {item.assignment_id: item for item in plan.provider_assignments}
        for assignment in assignments.values():
            try:
                provider = self.provider_registry.require(assignment.provider_key)
            except Exception as exc:
                diagnostics.append({"code": "provider_not_registered", "message": str(exc)})
                continue
            try:
                current = provider.capabilities()
            except Exception as exc:
                diagnostics.append({"code": "provider_capability_drift", "message": f"capabilities() failed: {exc}"})
                continue
            if not hasattr(current, "provider_key") or not hasattr(current, "capability_fingerprint"):
                diagnostics.append({"code": "provider_capability_drift", "message": "ProviderRuntime.capabilities() must return standard ProviderCapabilities。"})
                continue
            codes = capability_snapshot_diagnostics(
                plan.capability_snapshot,
                current,
                provider_profile=assignment.provider_profile,
            )
            diagnostics.extend({"code": code, "message": f"Provider capability mismatch: {code}"} for code in codes)
        self.capability_diagnostics = tuple(diagnostics)
        if diagnostics:
            raise ExecutionRuntimeError("Provider capability validation failed：" + "；".join(item["code"] for item in diagnostics))

    def _context(
        self,
        run: ExecutionRun,
        unit: Any,
        assignment: Any,
        video_job: Any,
        *,
        phase: str,
    ) -> ProviderRequestContext:
        timeout = unit.timeout_policy
        return ProviderRequestContext(
            request_id=f"{run.execution_run_id}:{unit.execution_unit_id}:attempt-{run.unit_states[unit.execution_unit_id].attempt}",
            execution_run_id=run.execution_run_id,
            execution_unit_id=unit.execution_unit_id,
            idempotency_key=run.unit_states[unit.execution_unit_id].idempotency_key,
            attempt=run.unit_states[unit.execution_unit_id].attempt,
            execution_plan_id=run.execution_plan_id,
            execution_plan_fingerprint=run.execution_plan_fingerprint,
            video_job_id=unit.video_job_id,
            video_job_fingerprint=unit.video_job_fingerprint,
            source_movie_plan_version=run.source_movie_plan_version,
            source_movie_plan_fingerprint=run.source_movie_plan_fingerprint,
            source_movie_plan_lineage_token=run.source_movie_plan_lineage_token,
            submit_timeout_seconds=float(timeout.submit_timeout or timeout.timeout_seconds),
            poll_timeout_seconds=float(timeout.poll_timeout or timeout.timeout_seconds),
            download_timeout_seconds=float(timeout.timeout_seconds),
            trace_id=f"trace-{run.execution_run_id}-{unit.execution_unit_id}-{phase}",
            metadata={"phase": phase, "provider_key": assignment.provider_key, "video_job_id": getattr(video_job, "job_id", "")},
        )

    def _load_run(self, execution_run_id: str) -> ExecutionRun:
        try:
            return self.state_store.load_run(execution_run_id)
        except FileNotFoundError as exc:
            raise RuntimeNotFoundError(f"ExecutionRun 不存在：{execution_run_id}") from exc

    def _ensure_fresh(self, run: ExecutionRun, bundle: ExecutionBundle | None = None) -> ExecutionBundle:
        current = self._bundle_or_raise(bundle)
        self._require_provider_capabilities(current)
        if (
            run.execution_bundle_fingerprint != current.bundle_fingerprint
            or run.execution_plan_fingerprint != current.execution_plan.execution_plan_fingerprint
            or run.execution_plan_id != current.execution_plan.execution_plan_id
        ):
            stale = self.transition_service.mark_run_stale(run, "ExecutionBundle fingerprint or plan identity changed")
            raise StaleExecutionRuntimeError(
                f"ExecutionRun 已 stale：{stale.execution_run_id}；请重新 /build-execution-plan 后 /start-execution。"
            )
        for unit in current.execution_plan.execution_units:
            state = run.unit_states.get(unit.execution_unit_id)
            job = current.video_job_map.get(unit.video_job_id)
            if state is None or job is None or state.video_job_fingerprint != job.video_job_fingerprint:
                self.transition_service.mark_run_stale(run, "VideoJob fingerprint changed")
                raise StaleExecutionRuntimeError("ExecutionRun 的 VideoJob provenance 已 stale。")
        self.execution_bundle = current
        return current

    def create_run(self, execution_bundle: ExecutionBundle | None = None) -> ExecutionRun:
        bundle = self._bundle_or_raise(execution_bundle)
        self._require_provider_capabilities(bundle)
        plan = bundle.execution_plan
        assignments = {item.assignment_id: item for item in plan.provider_assignments}
        for unit in plan.execution_units:
            assignment = assignments.get(unit.provider_assignment_id)
            if assignment is None:
                raise ExecutionRuntimeError(f"ExecutionUnit provider assignment missing: {unit.execution_unit_id}")
            self.provider_registry.require(assignment.provider_key)
        run_id = f"execution-run-{uuid4().hex[:20]}"
        states: dict[str, ExecutionUnitState] = {}
        for unit in plan.execution_units:
            assignment = assignments[unit.provider_assignment_id]
            states[unit.execution_unit_id] = ExecutionUnitState(
                execution_unit_id=unit.execution_unit_id,
                video_job_id=unit.video_job_id,
                video_job_fingerprint=unit.video_job_fingerprint,
                idempotency_key=_idempotency_key(bundle, unit.execution_unit_id, unit.video_job_fingerprint, assignment.provider_key),
                created_at=self.clock.now().isoformat(),
                updated_at=self.clock.now().isoformat(),
            )
        run = ExecutionRun(
            execution_run_id=run_id,
            schema_version="execution-run/1",
            execution_bundle_fingerprint=bundle.bundle_fingerprint,
            execution_plan_id=plan.execution_plan_id,
            execution_plan_version=plan.execution_plan_version,
            execution_plan_fingerprint=plan.execution_plan_fingerprint,
            source_movie_plan_id=plan.source_movie_plan_id,
            source_movie_plan_version=plan.source_movie_plan_version,
            source_movie_plan_fingerprint=plan.source_movie_plan_fingerprint,
            source_movie_plan_lineage_token=plan.source_movie_plan_lineage_token,
            unit_states=states,
            created_at=self.clock.now().isoformat(),
        )
        self.state_store.create_run(run)
        event = self.event_store.append(run_id, "execution_run_created", payload={"bundle_fingerprint": bundle.bundle_fingerprint})
        run = replace(run, last_event_id=event.event_id)
        self.state_store.save_run(run)
        return self.checkpoint(run.execution_run_id)

    def checkpoint(self, execution_run_id: str) -> ExecutionRun:
        run = self._load_run(execution_run_id)
        self._ensure_fresh(run)
        checkpoint = ExecutionCheckpoint.from_run(run)
        self.checkpoint_store.save(checkpoint)
        run = replace(run, latest_checkpoint_id=checkpoint.checkpoint_id)
        self.state_store.save_run(run)
        self.event_store.append(
            run.execution_run_id,
            "checkpoint_created",
            payload={"checkpoint_id": checkpoint.checkpoint_id},
        )
        return run

    def recover(
        self,
        execution_bundle: ExecutionBundle,
        checkpoint_id: str | None = None,
        execution_run_id: str | None = None,
    ) -> ExecutionRun:
        bundle = self._bundle_or_raise(execution_bundle)
        self._require_provider_capabilities(bundle)
        if execution_run_id is None:
            if checkpoint_id is None:
                raise ExecutionRuntimeError("recover 需要 checkpoint_id 或 execution_run_id")
            checkpoint = self.checkpoint_store.load(checkpoint_id)
            execution_run_id = checkpoint.execution_run_id
        run = self._load_run(execution_run_id)
        if checkpoint_id is None:
            checkpoint = self.checkpoint_store.latest(run.execution_run_id)
        else:
            checkpoint = self.checkpoint_store.load(checkpoint_id)
        if checkpoint is None or not checkpoint.validate_fingerprint():
            raise ExecutionRuntimeError("没有可恢复的有效 Checkpoint")
        if checkpoint.execution_bundle_fingerprint != bundle.bundle_fingerprint:
            self.transition_service.mark_run_stale(run, "Checkpoint 与当前 ExecutionBundle fingerprint 不匹配")
            raise StaleExecutionRuntimeError("Checkpoint 与当前 ExecutionBundle 不匹配。")
        if checkpoint.execution_plan_fingerprint != bundle.execution_plan.execution_plan_fingerprint:
            raise StaleExecutionRuntimeError("Checkpoint 与当前 ExecutionPlan 不匹配。")
        states = {str(item["execution_unit_id"]): ExecutionUnitState.from_dict(item) for item in checkpoint.unit_states}
        jobs = {str(item["provider_job_id"]): ProviderJob.from_dict(item) for item in checkpoint.provider_jobs}
        intents = {str(item["submission_intent_id"]): dict(item) for item in checkpoint.submission_intents}
        artifacts = {str(item["artifact_id"]): dict(item) for item in checkpoint.artifact_references}
        leases: dict[str, Mapping[str, Any]] = {}
        now = self.clock.now().isoformat()
        for item in checkpoint.active_leases:
            lease = ExecutionLease.from_dict(item)
            if not lease.is_expired(now):
                leases[lease.execution_unit_id] = lease.to_dict()
        run = replace(
            run,
            unit_states=states,
            provider_jobs=jobs,
            submission_intents=intents,
            retry_records=checkpoint.retry_records,
            artifacts=artifacts,
            failure_reports=checkpoint.failure_reports,
            revision_requests=checkpoint.revision_requests,
            leases=leases,
            latest_checkpoint_id=checkpoint.checkpoint_id,
            last_event_id=checkpoint.last_event_id,
        )
        self.state_store.save_run(run)
        self.execution_bundle = bundle
        self.event_store.append(run.execution_run_id, "runtime_recovered", payload={"checkpoint_id": checkpoint.checkpoint_id})
        return run

    def resume(self, execution_run_id: str) -> ExecutionRun:
        run = self._load_run(execution_run_id)
        self._ensure_fresh(run)
        if run.status == ExecutionRunStatus.STALE:
            raise StaleExecutionRuntimeError("stale Runtime 禁止 resume。")
        if run.status == ExecutionRunStatus.BLOCKED and (
            any(item.state == ExecutionState.SUBMISSION_UNCERTAIN for item in run.unit_states.values())
            or bool(run.revision_requests)
        ):
            return run
        return self.recover(self.execution_bundle, execution_run_id=execution_run_id)

    def step(self, execution_run_id: str) -> ExecutionRun:
        run = self._load_run(execution_run_id)
        bundle = self._ensure_fresh(run)
        if run.status in {ExecutionRunStatus.COMPLETED, ExecutionRunStatus.CANCELLED, ExecutionRunStatus.STALE}:
            return run
        plan = bundle.execution_plan
        resolver = DependencyResolver(plan, clock=self.clock)
        for unit_id in resolver.failed_dependency_units(run):
            current = self.state_store.load_run(run.execution_run_id)
            if current.unit_states[unit_id].state == ExecutionState.PREPARED:
                self.transition_service.transition(
                    current,
                    unit_id,
                    ExecutionState.BLOCKED,
                    event_type="unit_blocked",
                    reason="dependency failed",
                )
        run = self._load_run(execution_run_id)
        if run.status in {ExecutionRunStatus.FAILED, ExecutionRunStatus.BLOCKED} and not resolver.ready_units(run):
            return run
        capacity = max(1, plan.runtime_policy.max_parallelism - resolver.active_count(run))
        active = [
            unit_id
            for unit_id, state in run.unit_states.items()
            if state.state in {
                ExecutionState.SUBMITTING,
                ExecutionState.SUBMITTED,
                ExecutionState.QUEUED,
                ExecutionState.RUNNING,
                ExecutionState.DOWNLOADING,
                ExecutionState.DOWNLOADED,
                ExecutionState.VERIFYING,
                ExecutionState.VERIFIED,
            }
        ]
        if active:
            self._advance_unit(run, active[0])
            return self._load_run(execution_run_id)
        ready = resolver.ready_units(run)[:capacity]
        if not ready:
            return run
        for unit_id in ready:
            current = self._load_run(execution_run_id)
            if current.unit_states[unit_id].state == ExecutionState.PREPARED:
                self._execute_prepared(current, unit_id)
        return self._load_run(execution_run_id)

    def run_until_blocked_or_complete(self, execution_run_id: str, *, max_steps: int = 100) -> ExecutionRun:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        run = self._load_run(execution_run_id)
        for _ in range(max_steps):
            before = run.to_dict()
            if run.status in {
                ExecutionRunStatus.COMPLETED,
                ExecutionRunStatus.BLOCKED,
                ExecutionRunStatus.FAILED,
                ExecutionRunStatus.CANCELLED,
                ExecutionRunStatus.STALE,
            }:
                return run
            run = self.step(execution_run_id)
            if run.to_dict() == before:
                if isinstance(self.clock, FakeClock):
                    retry_times = [
                        _parse_iso(state.next_retry_at)
                        for state in run.unit_states.values()
                        if state.next_retry_at and _parse_iso(state.next_retry_at) > self.clock.now()
                    ]
                    if retry_times:
                        self.clock.sleep((min(retry_times) - self.clock.now()).total_seconds())
                        continue
                return run
            if isinstance(self.clock, FakeClock) and run.status not in {
                ExecutionRunStatus.COMPLETED,
                ExecutionRunStatus.BLOCKED,
                ExecutionRunStatus.FAILED,
                ExecutionRunStatus.CANCELLED,
                ExecutionRunStatus.STALE,
            }:
                intervals = [
                    unit.timeout_policy.poll_interval_seconds
                    for unit in self._bundle_or_raise().execution_plan.execution_units
                ]
                self.clock.sleep(min(intervals) if intervals else 1.0)
        return run

    def _assignment(self, bundle: ExecutionBundle, unit_id: str) -> tuple[Any, Any, Any]:
        plan = bundle.execution_plan
        unit = next(item for item in plan.execution_units if item.execution_unit_id == unit_id)
        assignment = next(item for item in plan.provider_assignments if item.assignment_id == unit.provider_assignment_id)
        job = bundle.video_job_map.get(unit.video_job_id)
        if job is None:
            raise ExecutionRuntimeError(f"VideoJob missing for unit {unit_id}")
        return unit, assignment, job

    def _runtime(self, provider_key: str) -> ProviderRuntime:
        return self.provider_registry.require(provider_key)

    def _update_unit(self, run: ExecutionRun, unit_id: str, **changes: Any) -> ExecutionRun:
        current = self.state_store.load_run(run.execution_run_id)
        state = current.unit_states[unit_id]
        updated = replace(state, updated_at=self.clock.now().isoformat(), **changes)
        self.state_store.save_unit_state(current.execution_run_id, updated)
        return self._load_run(current.execution_run_id)

    def _execute_prepared(self, run: ExecutionRun, unit_id: str) -> None:
        bundle = self._bundle_or_raise()
        unit, assignment, video_job = self._assignment(bundle, unit_id)
        resolver = DependencyResolver(bundle.execution_plan, clock=self.clock)
        lease = resolver.acquire_lease(run, unit_id, owner_id=self.owner_id, clock=self.clock)
        run = self.state_store.save_lease(run.execution_run_id, lease)
        run = self._update_unit(run, unit_id, lease_id=lease.lease_id)
        state = run.unit_states[unit_id]
        attempt = state.attempt + 1
        state = self._update_unit(run, unit_id, attempt=attempt, idempotency_key=state.idempotency_key).unit_states[unit_id]
        intent = next(
            (SubmissionIntent.from_dict(item) for item in run.submission_intents.values() if item.get("execution_unit_id") == unit_id),
            None,
        )
        if intent is None:
            intent = SubmissionIntent(
                submission_intent_id=f"intent-{uuid4().hex}",
                execution_run_id=run.execution_run_id,
                execution_unit_id=unit_id,
                video_job_id=unit.video_job_id,
                video_job_fingerprint=unit.video_job_fingerprint,
                provider_key=assignment.provider_key,
                idempotency_key=state.idempotency_key,
            )
            run = self.state_store.save_submission_intent(run.execution_run_id, intent)
            self.event_store.append(run.execution_run_id, "submission_intent_written", execution_unit_id=unit_id, payload={"idempotency_key": intent.idempotency_key})
            self.checkpoint(run.execution_run_id)
        if intent.provider_invocation_started and intent.status not in {"retryable_failure", "prepared"}:
            self._handle_error(
                self._load_run(run.execution_run_id),
                unit_id,
                ProviderRuntimeError(
                    "provider invocation marker exists without a ProviderJob",
                    category="submission_uncertain",
                    code="submission_invocation_marker",
                    provider_accepted=True,
                ),
                phase="submit",
            )
            return
        run = self.transition_service.transition(
            self._load_run(run.execution_run_id),
            unit_id,
            ExecutionState.SUBMITTING,
            event_type="provider_submit_started",
            reason="submission intent durable",
        ).run
        intent = replace(intent, status="invoking", provider_invocation_started=True)
        run = self.state_store.save_submission_intent(run.execution_run_id, intent)
        self.checkpoint(run.execution_run_id)
        provider = self._runtime(assignment.provider_key)
        context = self._context(run, unit, assignment, video_job, phase="submit")
        try:
            result = provider.submit(video_job, context)
        except ProviderRuntimeError as exc:
            self._handle_error(self._load_run(run.execution_run_id), unit_id, exc, phase="submit")
            return
        if not isinstance(result, ProviderSubmitResult) or not result.accepted:
            self._handle_error(
                self._load_run(run.execution_run_id),
                unit_id,
                ProviderRuntimeError(
                    "Provider submit did not accept the request",
                    category=ProviderErrorCategory.PROVIDER_UNAVAILABLE,
                    provider_key=assignment.provider_key,
                    provider_code="submit_not_accepted",
                    retryable=True,
                ),
                phase="submit",
            )
            return
        provider_job = result.provider_job
        run = self.state_store.save_provider_job(run.execution_run_id, provider_job)
        intent = replace(intent, status="submitted")
        self.state_store.save_submission_intent(run.execution_run_id, intent)
        run = self._update_unit(run, unit_id, provider_job_id=provider_job.provider_job_id)
        self.event_store.append(
            run.execution_run_id,
            "provider_submit_succeeded",
            execution_unit_id=unit_id,
            payload={
                "provider_job_id": provider_job.provider_job_id,
                "sanitized_provider_response": dict(result.sanitized_response),
            },
        )
        self.transition_service.transition(
            run,
            unit_id,
            ExecutionState.SUBMITTED,
            event_type="provider_submit_succeeded",
            reason="ProviderRuntime returned ProviderSubmitResult",
            metadata={"provider_job_id": provider_job.provider_job_id},
        )

    def _advance_unit(self, run: ExecutionRun, unit_id: str) -> None:
        run = self._load_run(run.execution_run_id)
        state = run.unit_states[unit_id]
        bundle = self._bundle_or_raise()
        unit, assignment, video_job = self._assignment(bundle, unit_id)
        provider = self._runtime(assignment.provider_key)
        if state.state == ExecutionState.SUBMITTING:
            submit_timeout = unit.timeout_policy.submit_timeout
            if submit_timeout is not None and state.started_at:
                if (self.clock.now() - _parse_iso(state.started_at)).total_seconds() >= submit_timeout:
                    self._handle_error(
                        run,
                        unit_id,
                        ProviderRuntimeError(
                            "Provider submit timed out; acceptance must be reconciled manually",
                            category=ProviderErrorCategory.SUBMIT_TIMEOUT,
                            provider_key=assignment.provider_key,
                            provider_code="submit_timeout",
                            submission_may_have_been_accepted=True,
                        ),
                        phase="submit",
                    )
            return
        if self._unit_timed_out(state, unit):
            if state.state in {ExecutionState.DOWNLOADING, ExecutionState.VERIFYING}:
                self._handle_error(
                    run,
                    unit_id,
                    ProviderRuntimeError(
                        "Provider download timed out",
                        category=ProviderErrorCategory.DOWNLOAD_TIMEOUT,
                        provider_key=assignment.provider_key,
                        provider_code="download_timeout",
                        retryable=True,
                    ),
                    phase="verify" if state.state is ExecutionState.VERIFYING else "download",
                )
            else:
                self._timeout_unit(run, unit_id, provider, reason="unit_total_timeout")
            return
        if state.state in {ExecutionState.SUBMITTED, ExecutionState.QUEUED, ExecutionState.RUNNING}:
            if state.next_retry_at and _parse_iso(state.next_retry_at) > self.clock.now():
                return
            provider_job = run.provider_jobs.get(state.provider_job_id or "")
            if provider_job is None:
                self._handle_error(run, unit_id, ProviderRuntimeError("ProviderJob missing after submit", category="invalid_state", code="missing_provider_job"), phase="poll")
                return
            poll_timeout = unit.timeout_policy.poll_timeout
            if poll_timeout is not None:
                reference = provider_job.last_polled_at or state.started_at
                if reference and (self.clock.now() - _parse_iso(reference)).total_seconds() >= poll_timeout:
                    self._handle_error(
                        run,
                        unit_id,
                        ProviderRuntimeError(
                            "Provider poll timed out",
                            category=ProviderErrorCategory.POLL_TIMEOUT,
                            provider_key=assignment.provider_key,
                            provider_code="poll_timeout",
                            retryable=True,
                        ),
                        phase="poll",
                    )
                    return
            try:
                result = provider.poll(provider_job, self._context(run, unit, assignment, video_job, phase="poll"))
            except ProviderRuntimeError as exc:
                self._handle_error(run, unit_id, exc, phase="poll")
                return
            if not isinstance(result, ProviderPollResult):
                self._handle_error(
                    run,
                    unit_id,
                    ProviderRuntimeError(
                        "Provider poll returned an invalid result",
                        category=ProviderErrorCategory.MALFORMED_RESPONSE,
                        provider_key=assignment.provider_key,
                        provider_code="invalid_poll_result",
                    ),
                    phase="poll",
                )
                return
            run = self.state_store.save_provider_job(run.execution_run_id, result.provider_job)
            self.event_store.append(
                run.execution_run_id,
                "provider_poll_succeeded",
                execution_unit_id=unit_id,
                payload={
                    "status": ProviderJobStatus.from_value(result.status).value,
                    "progress": result.progress,
                    "sanitized_provider_response": dict(result.sanitized_response),
                },
            )
            status = ProviderJobStatus.from_value(result.status)
            if status is ProviderJobStatus.QUEUED:
                if state.state == ExecutionState.SUBMITTED:
                    self.transition_service.transition(run, unit_id, ExecutionState.QUEUED, event_type="provider_status_changed", reason="Provider reported queued")
                return
            if status is ProviderJobStatus.RUNNING:
                if state.state == ExecutionState.SUBMITTED:
                    self.transition_service.transition(run, unit_id, ExecutionState.RUNNING, event_type="provider_status_changed", reason="Provider reported running")
                elif state.state == ExecutionState.QUEUED:
                    self.transition_service.transition(run, unit_id, ExecutionState.RUNNING, event_type="provider_status_changed", reason="Provider reported running")
                return
            if status is ProviderJobStatus.FAILED:
                self._handle_error(
                    run,
                    unit_id,
                    ProviderRuntimeError(
                        "Provider reported a failed remote job",
                        category=ProviderErrorCategory.PROVIDER_UNAVAILABLE,
                        provider_key=assignment.provider_key,
                        provider_code="provider_job_failed",
                    ),
                    phase="poll",
                )
                return
            if status is ProviderJobStatus.CANCELLED:
                self.transition_service.transition(run, unit_id, ExecutionState.CANCELLED, event_type="unit_cancelled", reason="provider cancelled")
                return
            if status is ProviderJobStatus.EXPIRED:
                self._handle_error(
                    run,
                    unit_id,
                    ProviderRuntimeError(
                        "Provider task expired",
                        category=ProviderErrorCategory.TASK_EXPIRED,
                        provider_key=assignment.provider_key,
                        provider_code="task_expired",
                    ),
                    phase="poll",
                )
                return
            if status is ProviderJobStatus.UNKNOWN:
                self._handle_error(
                    run,
                    unit_id,
                    ProviderRuntimeError(
                        "Provider returned an unmappable task status",
                        category=ProviderErrorCategory.MALFORMED_RESPONSE,
                        provider_key=assignment.provider_key,
                        provider_code="unknown_provider_status",
                    ),
                    phase="poll",
                )
                return
            if status is ProviderJobStatus.SUCCEEDED:
                if state.state == ExecutionState.SUBMITTED:
                    self.transition_service.transition(run, unit_id, ExecutionState.RUNNING, event_type="provider_status_changed", reason="Provider reported success")
                elif state.state == ExecutionState.QUEUED:
                    self.transition_service.transition(run, unit_id, ExecutionState.RUNNING, event_type="provider_status_changed", reason="Provider reported success")
                run = self._load_run(run.execution_run_id)
                self.transition_service.transition(run, unit_id, ExecutionState.DOWNLOADING, event_type="download_started", reason="provider output ready")
                self._download_and_verify(self._load_run(run.execution_run_id), unit_id, provider, unit, assignment)
                return
        if state.state in {ExecutionState.DOWNLOADING, ExecutionState.VERIFYING} and state.next_retry_at:
            if _parse_iso(state.next_retry_at) > self.clock.now():
                return
        if state.state == ExecutionState.DOWNLOADING:
            self._download_and_verify(run, unit_id, provider, unit, assignment)
        elif state.state == ExecutionState.DOWNLOADED:
            self._verify_existing(run, unit_id, unit, assignment)
        elif state.state == ExecutionState.VERIFIED:
            self.transition_service.transition(run, unit_id, ExecutionState.COMPLETED, event_type="execution_run_completed", reason="verified artifact already present")

    def _download_and_verify(self, run: ExecutionRun, unit_id: str, provider: ProviderRuntime, unit: Any, assignment: Any) -> None:
        state = run.unit_states[unit_id]
        job = run.provider_jobs.get(state.provider_job_id or "")
        if job is None:
            self._handle_error(run, unit_id, ProviderRuntimeError("ProviderJob missing during download", category="invalid_state", code="missing_provider_job"), phase="download")
            return
        if not _SAFE_COMPONENT.fullmatch(run.execution_run_id) or not _SAFE_COMPONENT.fullmatch(unit_id):
            self._handle_error(
                run,
                unit_id,
                ProviderRuntimeError(
                    "runtime artifact path component is unsafe",
                    category="invalid_request",
                    code="unsafe_artifact_path",
                ),
                phase="download",
            )
            return
        destination_root = self.artifact_root / run.execution_run_id / unit_id
        destination = DownloadDestination(
            temporary_path=str(destination_root / "provider-artifact.bin.part"),
            final_path=str(destination_root / "provider-artifact.bin"),
        )
        bundle = self._bundle_or_raise()
        video_job = bundle.video_job_map[unit.video_job_id]
        try:
            result = provider.download(job, destination, self._context(run, unit, assignment, video_job, phase="download"))
        except ProviderRuntimeError as exc:
            self._handle_error(run, unit_id, exc, phase="download")
            return
        if not isinstance(result, ProviderDownloadResult):
            self._handle_error(
                run,
                unit_id,
                ProviderRuntimeError(
                    "Provider download returned an invalid result",
                    category=ProviderErrorCategory.MALFORMED_RESPONSE,
                    provider_key=assignment.provider_key,
                    provider_code="invalid_download_result",
                ),
                phase="download",
            )
            return
        try:
            provider_verification = provider.verify(
                job,
                result,
                self._context(run, unit, assignment, video_job, phase="verify"),
            )
        except ProviderRuntimeError as exc:
            self._handle_error(run, unit_id, exc, phase="verify")
            return
        if not isinstance(provider_verification, ProviderVerificationResult):
            self._handle_error(
                run,
                unit_id,
                ProviderRuntimeError(
                    "Provider verify returned an invalid result",
                    category=ProviderErrorCategory.MALFORMED_RESPONSE,
                    provider_key=assignment.provider_key,
                    provider_code="invalid_verification_result",
                ),
                phase="verify",
            )
            return
        self.event_store.append(
            run.execution_run_id,
            "download_completed",
            execution_unit_id=unit_id,
            payload={
                "path": result.storage_path,
                "sha256": result.sha256,
                "provider_verification": provider_verification.to_dict(),
                "sanitized_provider_response": dict(result.sanitized_response_metadata),
            },
        )
        run = self.transition_service.transition(run, unit_id, ExecutionState.DOWNLOADED, event_type="download_completed", reason="Fake Artifact written", metadata={"download_path": result.storage_path}).run or self._load_run(run.execution_run_id)
        self._verify_download(self._load_run(run.execution_run_id), unit_id, unit, job, assignment, result, provider_verification)

    def _verify_existing(self, run: ExecutionRun, unit_id: str, unit: Any, assignment: Any) -> None:
        state = run.unit_states[unit_id]
        bundle = self._bundle_or_raise()
        plan = bundle.execution_plan
        provider_job = run.provider_jobs.get(state.provider_job_id or "")
        for artifact_id in state.artifact_ids:
            artifact = run.artifacts.get(artifact_id)
            if artifact and self.artifact_verifier.verify(
                ArtifactRecord.from_dict(artifact),
                provider_job=provider_job,
                expected_provenance={
                    "execution_plan_id": plan.execution_plan_id,
                    "execution_plan_fingerprint": plan.execution_plan_fingerprint,
                    "execution_unit_id": unit_id,
                    "video_job_id": unit.video_job_id,
                    "video_job_fingerprint": unit.video_job_fingerprint,
                },
            ).valid:
                self.transition_service.transition(run, unit_id, ExecutionState.VERIFYING, event_type="verification_started", reason="resume verification")
                self.transition_service.transition(self._load_run(run.execution_run_id), unit_id, ExecutionState.VERIFIED, event_type="verification_succeeded", reason="artifact remains valid")
                self.transition_service.transition(self._load_run(run.execution_run_id), unit_id, ExecutionState.COMPLETED, event_type="execution_run_completed", reason="verified artifact already present")
                return
        self.transition_service.transition(run, unit_id, ExecutionState.CORRUPTED, event_type="verification_failed", reason="artifact missing or corrupted")

    def _verify_download(
        self,
        run: ExecutionRun,
        unit_id: str,
        unit: Any,
        job: ProviderJob,
        assignment: Any,
        result: ProviderDownloadResult,
        provider_verification: ProviderVerificationResult,
    ) -> None:
        self.transition_service.transition(run, unit_id, ExecutionState.VERIFYING, event_type="verification_started", reason="verify Provider result and Artifact provenance")
        verification = self.artifact_verifier.verify(result, expected_artifact_type=(unit.expected_artifacts[0].artifact_type if unit.expected_artifacts else "video"), provider_job=job)
        artifact_id = f"artifact-{job.provider_job_id}"
        bundle = self._bundle_or_raise()
        plan = bundle.execution_plan
        artifact = ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type=result.artifact_type,
            media_type=result.media_type,
            storage_path=result.storage_path,
            size_bytes=result.size_bytes,
            sha256=result.sha256,
            verification_status="verified",
            execution_plan_id=plan.execution_plan_id,
            execution_plan_fingerprint=plan.execution_plan_fingerprint,
            execution_unit_id=unit_id,
            video_job_id=unit.video_job_id,
            video_job_fingerprint=unit.video_job_fingerprint,
            provider_job_id=job.provider_job_id,
            provider_key=assignment.provider_key,
            movie_plan_version=plan.source_movie_plan_version,
            movie_plan_fingerprint=plan.source_movie_plan_fingerprint,
            movie_plan_lineage_token=plan.source_movie_plan_lineage_token,
        )
        provenance_check = self.artifact_verifier.verify(
            artifact,
            provider_job=job,
            expected_provenance={
                "execution_plan_id": plan.execution_plan_id,
                "execution_plan_fingerprint": plan.execution_plan_fingerprint,
                "execution_unit_id": unit_id,
                "video_job_id": unit.video_job_id,
                "video_job_fingerprint": unit.video_job_fingerprint,
                "movie_plan_version": plan.source_movie_plan_version,
                "movie_plan_fingerprint": plan.source_movie_plan_fingerprint,
                "movie_plan_lineage_token": plan.source_movie_plan_lineage_token,
            },
        )
        verification_errors = verification.errors + provenance_check.errors
        valid = provider_verification.valid and verification.valid and provenance_check.valid
        if not valid:
            artifact = replace(artifact, verification_status="corrupted")
        run = self.state_store.save_artifact(run.execution_run_id, artifact.to_dict())
        if not valid:
            messages = list(verification_errors)
            if not provider_verification.valid:
                messages.append(provider_verification.verification_code)
            self.transition_service.transition(run, unit_id, ExecutionState.CORRUPTED, event_type="verification_failed", reason="; ".join(messages), metadata={"artifact_ids": [artifact_id]})
            self._release_lease(self._load_run(run.execution_run_id), unit_id)
            return
        self.event_store.append(run.execution_run_id, "verification_succeeded", execution_unit_id=unit_id, payload={"artifact_id": artifact_id})
        run = self.transition_service.transition(run, unit_id, ExecutionState.VERIFIED, event_type="verification_succeeded", reason="Fake Artifact provenance and hash verified", metadata={"artifact_ids": [artifact_id]}).run or self._load_run(run.execution_run_id)
        run = self.transition_service.transition(self._load_run(run.execution_run_id), unit_id, ExecutionState.COMPLETED, event_type="execution_run_completed", reason="unit complete").run or self._load_run(run.execution_run_id)
        self._release_lease(run, unit_id)

    def _unit_timed_out(self, state: ExecutionUnitState, unit: Any) -> bool:
        if not state.started_at:
            return False
        elapsed = (self.clock.now() - _parse_iso(state.started_at)).total_seconds()
        return elapsed >= unit.timeout_policy.unit_total_timeout

    def _timeout_unit(self, run: ExecutionRun, unit_id: str, provider: ProviderRuntime, *, reason: str = "unit_total_timeout") -> None:
        state = run.unit_states[unit_id]
        job = run.provider_jobs.get(state.provider_job_id or "")
        if job is not None:
            try:
                if bool(provider.capabilities().supports_cancel):
                    bundle = self._bundle_or_raise()
                    unit, assignment, video_job = self._assignment(bundle, unit_id)
                    provider.cancel(job, self._context(run, unit, assignment, video_job, phase="cancel"))
            except ProviderRuntimeError:
                pass
        self.transition_service.transition(run, unit_id, ExecutionState.TIMED_OUT, event_type="unit_timed_out", reason=reason)
        self._release_lease(self._load_run(run.execution_run_id), unit_id)

    def _handle_error(self, run: ExecutionRun, unit_id: str, exc: ProviderRuntimeError, *, phase: str) -> None:
        state = run.unit_states[unit_id]
        unit, assignment, _ = self._assignment(self._bundle_or_raise(), unit_id)
        current = self._load_run(run.execution_run_id)
        report = self.failure_protocol.build_report(
            exc,
            execution_run_id=current.execution_run_id,
            execution_unit_id=unit_id,
            provider_job_id=state.provider_job_id,
            source_movie_plan_id=current.source_movie_plan_id,
            source_movie_plan_fingerprint=current.source_movie_plan_fingerprint,
            source_video_job_fingerprint=state.video_job_fingerprint,
        )
        global_budget = self._bundle_or_raise().execution_plan.runtime_policy.retry_budget
        resolution = self.failure_protocol.resolve(
            report,
            retry_count=state.retry_count,
            max_attempts=unit.retry_policy.max_attempts,
            retry_budget_remaining=(
                global_budget is None or len(current.retry_records) < global_budget
            ),
            retryable_error_codes=unit.retry_policy.retryable_error_codes,
            error_code=exc.code,
        )
        self.state_store.save_failure_report(current.execution_run_id, report.to_dict())
        classified_payload = self.failure_protocol.event_payload(
            report,
            resolution,
            retry_count=state.retry_count,
        )
        classified_payload["phase"] = phase
        self.event_store.append(
            current.execution_run_id,
            PROVIDER_FAILURE_CLASSIFIED,
            execution_unit_id=unit_id,
            payload=classified_payload,
        )
        error = RuntimeErrorRecord(
            error_id=report.failure_id,
            category=report.category.value,
            code=exc.code,
            message=report.message,
            retryable=report.retryable,
            provider_key=assignment.provider_key,
            metadata={"phase": phase, **dict(report.sanitized_details)},
        )
        if resolution.action is FailureAction.STOP_AND_WARN:
            self.transition_service.transition(
                current,
                unit_id,
                ExecutionState.SUBMISSION_UNCERTAIN,
                event_type="provider_submit_uncertain",
                reason=resolution.reason,
                metadata={"last_error": error.to_dict(), "action": resolution.action.value},
            )
            current = self._load_run(current.execution_run_id)
            intent_values = dict(current.submission_intents)
            for key, raw in intent_values.items():
                if raw.get("execution_unit_id") == unit_id:
                    intent_values[key] = {**raw, "status": "submission_uncertain", "provider_invocation_started": True}
            self.state_store.save_run(replace(current, submission_intents=intent_values))
            # The uncertainty marker is part of the durable recovery boundary;
            # persist it after updating the submission intent, not only in the
            # state store.
            self.checkpoint(current.execution_run_id)
            self.event_store.append(
                current.execution_run_id,
                EXECUTION_BLOCKED_SUBMISSION_UNCERTAIN,
                execution_unit_id=unit_id,
                payload={
                    **self.failure_protocol.event_payload(
                        report,
                        resolution,
                        retry_count=state.retry_count,
                    ),
                    "recommended_action": "manual provider reconciliation; do not resubmit",
                },
            )
            # Keep the legacy event for consumers that still display the old
            # lower-case runtime vocabulary.
            self.event_store.append(
                current.execution_run_id,
                "runtime_blocked",
                execution_unit_id=unit_id,
                payload={"reason": "SUBMISSION_UNCERTAIN", "recommended_action": "manual provider reconciliation; do not resubmit"},
            )
            self._release_lease(self._load_run(current.execution_run_id), unit_id)
            return
        if resolution.action is FailureAction.REQUEST_REVISION:
            revision_request = self.failure_protocol.create_revision_request(
                report,
                request_id=resolution.revision_request_id,
            )
            current = self.state_store.save_revision_request(
                current.execution_run_id,
                revision_request.to_dict(),
            )
            self.transition_service.transition(
                current,
                unit_id,
                ExecutionState.BLOCKED,
                event_type="execution_waiting_for_revision",
                reason=resolution.reason,
                metadata={
                    "last_error": error.to_dict(),
                    "action": resolution.action.value,
                    "revision_request_id": revision_request.request_id,
                },
            )
            self.event_store.append(
                current.execution_run_id,
                REVISION_REQUESTED,
                execution_unit_id=unit_id,
                payload=self.failure_protocol.event_payload(
                    report,
                    resolution,
                    retry_count=state.retry_count,
                    revision_request_id=revision_request.request_id,
                ),
            )
            self._release_lease(self._load_run(current.execution_run_id), unit_id)
            return
        if resolution.action is FailureAction.RETRY:
            retry_count = state.retry_count + 1
            backoff = (
                report.retry_after_seconds
                if report.retry_after_seconds is not None
                else unit.retry_policy.backoff_seconds
            )
            retry_at = (self.clock.now() + timedelta(seconds=backoff)).isoformat()
            record = RetryRecord(
                retry_record_id=f"retry-{uuid4().hex}",
                execution_unit_id=unit_id,
                attempt=state.attempt,
                error_category=report.category.value,
                retryable=True,
                retry_at=retry_at,
                backoff_seconds=backoff,
                reason=report.message,
            )
            current = self._update_unit(
                current,
                unit_id,
                last_error=error,
                retry_count=retry_count,
                next_retry_at=retry_at,
            )
            if phase == "submit":
                intent_values = dict(current.submission_intents)
                for key, raw in intent_values.items():
                    if raw.get("execution_unit_id") == unit_id:
                        # The adapter error explicitly proves this attempt was
                        # not accepted, so retrying with the same key is safe.
                        intent_values[key] = {
                            **raw,
                            "status": "retryable_failure",
                            "provider_invocation_started": False,
                        }
                current = self.state_store.save_run(replace(current, submission_intents=intent_values))
            self.state_store.save_retry_record(current.execution_run_id, record)
            self.event_store.append(
                current.execution_run_id,
                EXECUTION_RETRY_SCHEDULED,
                execution_unit_id=unit_id,
                payload={
                    **self.failure_protocol.event_payload(
                        report,
                        resolution,
                        retry_count=retry_count,
                    ),
                    "retry_at": retry_at,
                    "phase": phase,
                },
            )
            self.event_store.append(
                current.execution_run_id,
                "retry_scheduled",
                execution_unit_id=unit_id,
                payload={"category": report.category.value, "retry_at": retry_at, "phase": phase},
            )
            if phase == "submit":
                self.transition_service.transition(
                    self._load_run(current.execution_run_id),
                    unit_id,
                    ExecutionState.PREPARED,
                    event_type="retry_scheduled",
                    reason=resolution.reason,
                    metadata={"last_error": error.to_dict(), "retry_count": retry_count, "next_retry_at": retry_at},
                )
                self._release_lease(self._load_run(current.execution_run_id), unit_id)
            elif phase == "verify":
                self.transition_service.transition(
                    self._load_run(current.execution_run_id),
                    unit_id,
                    ExecutionState.DOWNLOADING,
                    event_type="retry_scheduled",
                    reason="retry verification through the existing Provider Job",
                    metadata={"retry_count": retry_count, "next_retry_at": retry_at},
                )
            return
        target = ExecutionState.FAILED
        self.transition_service.transition(
            current,
            unit_id,
            target,
            event_type="unit_failed",
            reason=resolution.reason,
            metadata={"last_error": error.to_dict(), "action": resolution.action.value},
        )
        self._release_lease(self._load_run(current.execution_run_id), unit_id)

    def _release_lease(self, run: ExecutionRun, unit_id: str) -> None:
        self.state_store.remove_lease(run.execution_run_id, unit_id)
        current = self._load_run(run.execution_run_id)
        if unit_id in current.unit_states:
            self._update_unit(current, unit_id, lease_id=None)
        self.event_store.append(run.execution_run_id, "lease_released", execution_unit_id=unit_id, payload={})

    def apply_recompiled_bundle(
        self,
        execution_run_id: str,
        revised_bundle: ExecutionBundle,
        *,
        revision_request_id: str,
    ) -> ExecutionRun:
        """Start a new run from an explicitly applied, freshly compiled bundle.

        This is the runtime-side adapter boundary.  It does not call a
        DirectorAgent, RevisionGuard, or compiler, and it never reuses the old
        ExecutionRun.  The caller must complete the existing
        candidate/diff/guard/apply workflow and pass its new immutable bundle.
        """

        old_run = self._load_run(execution_run_id)
        if not str(revision_request_id).strip():
            raise ExecutionRuntimeError("revision_request_id is required before recompilation")
        request = next(
            (
                item
                for item in old_run.revision_requests
                if str(item.get("request_id", "")) == revision_request_id
            ),
            None,
        )
        if request is None:
            raise ExecutionRuntimeError("revision request is not recorded on the current ExecutionRun")
        if request.get("source_movie_plan_id") != old_run.source_movie_plan_id:
            raise StaleExecutionRuntimeError("revision request source MoviePlan id does not match the run")
        if request.get("source_movie_plan_fingerprint") != old_run.source_movie_plan_fingerprint:
            raise StaleExecutionRuntimeError("revision request source MoviePlan fingerprint is stale")
        bundle = self._bundle_or_raise(revised_bundle)
        plan = bundle.execution_plan
        if bundle.bundle_fingerprint == old_run.execution_bundle_fingerprint:
            raise StaleExecutionRuntimeError("recompiled ExecutionBundle fingerprint must differ from the old run")
        if plan.source_movie_plan_fingerprint == old_run.source_movie_plan_fingerprint:
            raise StaleExecutionRuntimeError("recompiled bundle must reference the revised MoviePlan fingerprint")

        stale_reason = "old ExecutionRun invalidated after explicit revision apply"
        self.transition_service.mark_run_stale(old_run, stale_reason)
        stale_run = self._load_run(execution_run_id)
        stale_artifacts = {
            str(artifact_id): {
                **dict(artifact),
                "stale": True,
                "stale_reason": stale_reason,
                "replaced_by_revision_request_id": revision_request_id,
            }
            for artifact_id, artifact in stale_run.artifacts.items()
        }
        self.state_store.save_run(replace(stale_run, artifacts=stale_artifacts))
        event_payload = {
            "actor": "RevisionApplyPort",
            "revision_request_id": revision_request_id,
            "execution_run_id": execution_run_id,
            "source_movie_plan_id": old_run.source_movie_plan_id,
            "source_movie_plan_fingerprint": old_run.source_movie_plan_fingerprint,
            "source_video_job_fingerprint": request.get("source_video_job_fingerprint", ""),
            "old_execution_bundle_fingerprint": old_run.execution_bundle_fingerprint,
            "new_execution_bundle_fingerprint": bundle.bundle_fingerprint,
        }
        self.event_store.append(execution_run_id, REVISION_APPLIED, payload=event_payload)
        self.event_store.append(
            execution_run_id,
            EXECUTION_ARTIFACTS_INVALIDATED,
            payload={
                **event_payload,
                "invalidated_artifact_count": len(stale_artifacts),
                "reason_summary": stale_reason,
            },
        )
        self.execution_bundle = bundle
        return self.create_run(bundle)

    # Explicit alias for callers that name the operation after its port.
    recompile_after_revision = apply_recompiled_bundle

    def cancel_unit(self, execution_run_id: str, execution_unit_id: str) -> ExecutionRun:
        run = self._load_run(execution_run_id)
        self._ensure_fresh(run)
        state = run.unit_states.get(execution_unit_id)
        if state is None:
            raise ExecutionRuntimeError(f"unknown execution unit: {execution_unit_id}")
        bundle = self._bundle_or_raise()
        unit, assignment, video_job = self._assignment(bundle, execution_unit_id)
        job = run.provider_jobs.get(state.provider_job_id or "")
        if job is not None:
            try:
                result = self._runtime(assignment.provider_key).cancel(
                    job,
                    self._context(run, unit, assignment, video_job, phase="cancel"),
                )
                if isinstance(result, ProviderCancelResult) and not result.supported:
                    self.event_store.append(
                        execution_run_id,
                        "provider_cancel_unsupported",
                        execution_unit_id=execution_unit_id,
                        payload={"supported": False, "accepted": False, "sanitized_provider_response": dict(result.sanitized_response)},
                    )
            except ProviderRuntimeError:
                pass
        if state.state != ExecutionState.CANCELLED and (
            not state.state.terminal or state.state is ExecutionState.SUBMISSION_UNCERTAIN
        ):
            self.transition_service.transition(run, execution_unit_id, ExecutionState.CANCELLED, event_type="unit_cancelled", reason="user cancellation")
        self._release_lease(self._load_run(execution_run_id), execution_unit_id)
        return self._load_run(execution_run_id)

    def cancel(self, execution_run_id: str) -> ExecutionRun:
        run = self._load_run(execution_run_id)
        for unit_id, state in list(run.unit_states.items()):
            if state.state not in {
                ExecutionState.COMPLETED,
                ExecutionState.FAILED,
                ExecutionState.CORRUPTED,
                ExecutionState.TIMED_OUT,
                ExecutionState.CANCELLED,
                ExecutionState.BLOCKED,
            }:
                self.cancel_unit(execution_run_id, unit_id)
        run = self._load_run(execution_run_id)
        cancelled = replace(run, status=ExecutionRunStatus.CANCELLED, completed_at=self.clock.now().isoformat())
        self.state_store.save_run(cancelled)
        self.event_store.append(execution_run_id, "execution_run_cancelled", payload={"reason": "user cancellation"})
        return cancelled

    def resolve_uncertain_submission(self, execution_run_id: str, execution_unit_id: str, *, provider_job: ProviderJob | None = None) -> ExecutionRun:
        raise ExecutionRuntimeError("Phase 5A 不自动处理 SUBMISSION_UNCERTAIN；需要未来的显式人工 Provider reconciliation。")

    def inspect(self, execution_run_id: str) -> dict[str, Any]:
        run = self._load_run(execution_run_id)
        bundle = self._bundle_or_raise()
        try:
            self._require_provider_capabilities(bundle)
        except ExecutionRuntimeError:
            pass
        resolver = DependencyResolver(bundle.execution_plan, clock=self.clock)
        counts: dict[str, int] = {}
        for state in run.unit_states.values():
            counts[state.state.value] = counts.get(state.state.value, 0) + 1
        submit_counts: dict[str, int] = {}
        poll_counts: dict[str, int] = {}
        for provider in self.provider_registry.keys():
            runtime = self.provider_registry.get(provider)
            submit_counts[provider] = int(getattr(runtime, "submit_count", 0))
            poll_counts[provider] = int(getattr(runtime, "poll_count", 0))
        provider_jobs = {
            key: {
                "provider_key": job.provider_key,
                "remote_job_id": job.remote_job_id,
                "status": job.normalized_status.value,
                "capability_match_status": "checked",
            }
            for key, job in run.provider_jobs.items()
        }
        return {
            "execution_run_id": run.execution_run_id,
            "execution_bundle_fingerprint": run.execution_bundle_fingerprint,
            "status": run.status.value,
            "unit_state_counts": counts,
            "ready_units": list(resolver.ready_units(run)),
            "running_units": [unit_id for unit_id, state in run.unit_states.items() if state.state in {ExecutionState.SUBMITTING, ExecutionState.SUBMITTED, ExecutionState.QUEUED, ExecutionState.RUNNING, ExecutionState.DOWNLOADING, ExecutionState.DOWNLOADED, ExecutionState.VERIFYING, ExecutionState.VERIFIED}],
            "blocked_units": [unit_id for unit_id, state in run.unit_states.items() if state.state == ExecutionState.BLOCKED],
            "failed_units": [unit_id for unit_id, state in run.unit_states.items() if state.state in {ExecutionState.FAILED, ExecutionState.TIMED_OUT, ExecutionState.CORRUPTED}],
            "submission_uncertain_units": [unit_id for unit_id, state in run.unit_states.items() if state.state == ExecutionState.SUBMISSION_UNCERTAIN],
            "latest_checkpoint_id": run.latest_checkpoint_id,
            "provider_submit_counts": submit_counts,
            "provider_poll_counts": poll_counts,
            "artifact_count": len(run.artifacts),
            "provider_jobs": provider_jobs,
            "provider_capability_diagnostics": list(self.capability_diagnostics),
            "registered_provider_keys": list(self.provider_registry.list_provider_keys()),
            "diagnostics": [
                *list(run.diagnostics),
                *list(self.capability_diagnostics),
                *(
                    [{
                        "code": "submission_uncertain",
                        "message": "Provider acceptance is unknown; manual reconciliation is required and automatic resubmission is forbidden.",
                    }]
                    if any(state.state is ExecutionState.SUBMISSION_UNCERTAIN for state in run.unit_states.values())
                    else []
                ),
            ],
        }

    def events(self, execution_run_id: str, *, limit: int | None = None) -> tuple[Mapping[str, Any], ...]:
        events = tuple(event.to_dict() for event in self.event_store.list_events(execution_run_id))
        return events[-limit:] if limit else events


__all__ = [
    "ExecutionRuntime",
    "ExecutionRuntimeError",
    "RuntimeNotFoundError",
    "StaleExecutionRuntimeError",
    "FakeClock",
]
