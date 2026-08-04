from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    ExecutionRuntime,
    ExecutionState,
    FailureAction,
    FailureProtocol,
    FakeClock,
    FakeProviderScenario,
    ProviderErrorCategory,
    ProviderRuntimeError,
    ProviderRuntimeRegistry,
    RevisionCandidateFactory,
    RevisionDiffBuilder,
    RevisionGuard,
)
from guided_story_agent.v2.execution_events import (
    EXECUTION_ARTIFACTS_INVALIDATED,
    EXECUTION_BLOCKED_SUBMISSION_UNCERTAIN,
    EXECUTION_RETRY_SCHEDULED,
    PROVIDER_FAILURE_CLASSIFIED,
    REVISION_APPLIED,
    REVISION_REQUESTED,
    InMemoryExecutionEventStore,
)
from guided_story_agent.v2.execution_fingerprint import (
    execution_bundle_fingerprint,
    execution_plan_fingerprint,
    execution_unit_fingerprint,
    video_job_fingerprint,
)
from test_v2_execution_bundle import _bundle
from test_v2_contracts import make_plan


def _protocol_error(category: ProviderErrorCategory, **kwargs: object) -> ProviderRuntimeError:
    return ProviderRuntimeError(
        "provider failed Authorization: Bearer TEST_FAILURE_SECRET",
        category=category,
        provider_code=category.value.lower(),
        sanitized_details={"Authorization": "Bearer TEST_FAILURE_SECRET", **kwargs},
    )


def test_transient_failure_is_retryable_only_within_budget() -> None:
    protocol = FailureProtocol(id_factory=lambda prefix: f"{prefix}-test")
    report, resolution = protocol.classify(
        _protocol_error(ProviderErrorCategory.TRANSIENT_NETWORK),
        execution_run_id="run-1",
        execution_unit_id="unit-1",
        provider_job_id="provider-job-1",
        source_movie_plan_id="movie-plan-1",
        source_movie_plan_fingerprint="movie-fp-1",
        source_video_job_fingerprint="video-fp-1",
        retry_count=0,
        max_attempts=2,
    )
    assert resolution.action is FailureAction.RETRY
    assert resolution.retry_allowed is True
    assert "TEST_FAILURE_SECRET" not in report.message
    assert report.sanitized_details["Authorization"] == "[REDACTED]"

    exhausted = protocol.resolve(report, retry_count=1, max_attempts=2)
    assert exhausted.action is FailureAction.ABORT


def test_uncertain_and_submit_timeout_always_stop_without_retry() -> None:
    protocol = FailureProtocol()
    for category, accepted in (
        (ProviderErrorCategory.SUBMISSION_UNCERTAIN, False),
        (ProviderErrorCategory.SUBMIT_TIMEOUT, True),
    ):
        report = protocol.build_report(
            _protocol_error(category, submission_may_have_been_accepted=accepted),
            execution_run_id="run-1",
            execution_unit_id="unit-1",
            provider_job_id=None,
            source_movie_plan_id="movie-plan-1",
            source_movie_plan_fingerprint="movie-fp-1",
            source_video_job_fingerprint="video-fp-1",
        )
        resolution = protocol.resolve(report, retry_count=0, max_attempts=99)
        assert resolution.action is FailureAction.STOP_AND_WARN
        assert resolution.retry_allowed is False


def test_policy_and_capability_failures_create_safe_revision_request() -> None:
    protocol = FailureProtocol(id_factory=lambda prefix: f"{prefix}-test")
    for category in (ProviderErrorCategory.POLICY_REJECTED, ProviderErrorCategory.UNSUPPORTED_CAPABILITY):
        report = protocol.build_report(
            _protocol_error(category, provider_payload={"secret": "must-not-cross-boundary"}),
            execution_run_id="run-1",
            execution_unit_id="unit-1",
            provider_job_id="job-1",
            source_movie_plan_id="movie-plan-1",
            source_movie_plan_fingerprint="movie-fp-1",
            source_video_job_fingerprint="video-fp-1",
        )
        resolution = protocol.resolve(report)
        request = protocol.create_revision_request(report, request_id=resolution.revision_request_id)
        assert resolution.action is FailureAction.REQUEST_REVISION
        assert request.source_movie_plan_fingerprint == "movie-fp-1"
        assert request.source_video_job_fingerprint == "video-fp-1"
        assert "provider_payload" not in request.to_dict()["sanitized_context"]
        assert "automatic_apply" in request.forbidden_changes


def test_irrecoverable_failure_aborts() -> None:
    protocol = FailureProtocol()
    report = protocol.build_report(
        _protocol_error(ProviderErrorCategory.AUTHENTICATION_FAILED),
        execution_run_id="run-1",
        execution_unit_id="unit-1",
        provider_job_id=None,
        source_movie_plan_id="movie-plan-1",
        source_movie_plan_fingerprint="movie-fp-1",
        source_video_job_fingerprint="video-fp-1",
    )
    assert protocol.resolve(report).action is FailureAction.ABORT


def test_runtime_records_failure_events_and_revision_wait_state() -> None:
    clock = FakeClock()
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry.with_fake(
            provider_keys=("fake",),
            scenario=FakeProviderScenario("policy_rejected"),
            clock=clock,
        ),
        clock=clock,
        event_store=InMemoryExecutionEventStore(),
    )
    run = runtime.create_run()
    blocked = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=20)
    assert blocked.status.value == "BLOCKED"
    assert next(iter(blocked.unit_states.values())).state is ExecutionState.BLOCKED
    assert blocked.revision_requests
    event_types = {event["event_type"] for event in runtime.events(blocked.execution_run_id)}
    assert {PROVIDER_FAILURE_CLASSIFIED, REVISION_REQUESTED}.issubset(event_types)
    assert all("TEST_FAILURE_SECRET" not in repr(event) for event in runtime.events(blocked.execution_run_id))


def test_runtime_uncertain_submission_writes_dedicated_event_and_never_resubmits() -> None:
    clock = FakeClock()
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry.with_fake(
            provider_keys=("fake",),
            scenario=FakeProviderScenario("submission_uncertain"),
            clock=clock,
        ),
        clock=clock,
    )
    run = runtime.create_run()
    blocked = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=20)
    assert blocked.status.value == "BLOCKED"
    assert runtime.provider_registry.get("fake").submit_count == 1
    assert EXECUTION_BLOCKED_SUBMISSION_UNCERTAIN in {
        event["event_type"] for event in runtime.events(blocked.execution_run_id)
    }
    assert runtime.resume(blocked.execution_run_id).status.value == "BLOCKED"
    assert runtime.provider_registry.get("fake").submit_count == 1


def test_runtime_transient_poll_failure_retries_current_job_and_records_retry_event() -> None:
    clock = FakeClock()
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry.with_fake(
            provider_keys=("fake",),
            scenario=FakeProviderScenario("retryable_poll_failure"),
            clock=clock,
        ),
        clock=clock,
    )
    run = runtime.create_run()
    result = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=80)
    assert result.status.value == "COMPLETED"
    assert runtime.provider_registry.get("fake").submit_count == len(result.unit_states)
    assert EXECUTION_RETRY_SCHEDULED in {
        event["event_type"] for event in runtime.events(result.execution_run_id)
    }


def test_revision_guard_rejects_provider_fields_and_candidate_does_not_mutate_plan() -> None:
    plan = make_plan()
    request = {
        "request_id": "revision-provider-test",
        "severity": "hard",
        "target": "story",
        "instruction": "revise story",
        "avoid": ("provider_payload",),
    }
    candidate = replace(
        RevisionCandidateFactory().create_noop_candidate(plan, ()),
        metadata={"provider_payload": {"secret": "must-be-rejected"}},
    )
    decision = RevisionGuard().evaluate(
        plan,
        candidate,
        RevisionDiffBuilder().build_diff(plan, candidate, (request,)),
        requests=(request,),
    )
    assert decision.decision in {"reject", "pending_director"}
    assert candidate.revised_movie_plan == plan


def _revised_bundle(bundle):
    new_movie_fingerprint = "revised-movie-plan-fingerprint"
    jobs = []
    for job in bundle.video_jobs:
        changed = replace(job, source_movie_plan_fingerprint=new_movie_fingerprint, video_job_fingerprint="")
        jobs.append(replace(changed, video_job_fingerprint=video_job_fingerprint(changed)))
    job_by_id = {job.job_id: job for job in jobs}
    units = []
    for unit in bundle.execution_plan.execution_units:
        changed = replace(unit, video_job_fingerprint=job_by_id[unit.video_job_id].video_job_fingerprint)
        units.append(replace(changed, execution_unit_fingerprint=""))
    units = tuple(replace(unit, execution_unit_fingerprint=execution_unit_fingerprint(unit)) for unit in units)
    plan = replace(
        bundle.execution_plan,
        source_movie_plan_fingerprint=new_movie_fingerprint,
        execution_units=units,
        execution_plan_fingerprint="",
    )
    plan = replace(plan, execution_plan_fingerprint=execution_plan_fingerprint(plan))
    return replace(
        bundle,
        execution_plan=plan,
        video_jobs=tuple(jobs),
        bundle_fingerprint=execution_bundle_fingerprint(
            replace(bundle, execution_plan=plan, video_jobs=tuple(jobs), bundle_fingerprint="")
        ),
    )


def test_explicit_recompiled_bundle_marks_old_run_stale_and_creates_new_run() -> None:
    clock = FakeClock()
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry.with_fake(
            provider_keys=("fake",),
            scenario=FakeProviderScenario("policy_rejected"),
            clock=clock,
        ),
        clock=clock,
    )
    original = runtime.create_run()
    blocked = runtime.run_until_blocked_or_complete(original.execution_run_id, max_steps=20)
    request_id = blocked.revision_requests[0]["request_id"]
    revised = _revised_bundle(runtime.execution_bundle)
    new_run = runtime.apply_recompiled_bundle(blocked.execution_run_id, revised, revision_request_id=request_id)
    old_run = runtime.state_store.load_run(blocked.execution_run_id)
    assert old_run.status.value == "STALE"
    assert new_run.execution_run_id != old_run.execution_run_id
    assert new_run.execution_bundle_fingerprint != old_run.execution_bundle_fingerprint
    assert {event["event_type"] for event in runtime.events(old_run.execution_run_id)} >= {
        REVISION_APPLIED,
        EXECUTION_ARTIFACTS_INVALIDATED,
    }
