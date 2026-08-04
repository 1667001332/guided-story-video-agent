from __future__ import annotations

from guided_story_agent.v2 import (
    ExecutionRuntime,
    ExecutionState,
    FakeClock,
    ProviderRuntimeRegistry,
    TransitionRejectedError,
    RuntimeTransitionService,
    InMemoryExecutionEventStore,
    InMemoryRuntimeStateStore,
    InMemoryCheckpointStore,
)
from test_v2_execution_bundle import _bundle


def _runtime():
    clock = FakeClock()
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry.with_fake(provider_keys=("fake",), clock=clock),
        state_store=InMemoryRuntimeStateStore(),
        event_store=InMemoryExecutionEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        clock=clock,
    )
    return runtime


def test_transition_service_rejects_illegal_jump_and_is_idempotent() -> None:
    runtime = _runtime()
    run = runtime.create_run()
    unit_id = next(iter(run.unit_states))
    service = RuntimeTransitionService(
        runtime.state_store,
        runtime.event_store,
        runtime.checkpoint_store,
        clock=runtime.clock,
    )
    transition = service.transition(
        run,
        unit_id,
        ExecutionState.SUBMITTING,
        event_type="provider_submit_started",
        event_id="transition-once",
    )
    assert transition.to_state is ExecutionState.SUBMITTING
    assert service.transition(
        transition.run,
        unit_id,
        ExecutionState.SUBMITTED,
        event_id="transition-once",
    ).event_id == "transition-once"
    try:
        service.transition(transition.run, unit_id, ExecutionState.COMPLETED)
    except TransitionRejectedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("PREPARED/SUBMITTING must not jump to COMPLETED")


def test_uncertain_submission_is_terminal_and_not_resubmitted() -> None:
    runtime = _runtime()
    fake = runtime.provider_registry.get("fake")
    fake.scenario.name = "submission_uncertain"
    fake.scenario.submission_uncertain = True
    run = runtime.create_run()
    result = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=20)
    assert result.status.value == "BLOCKED"
    first = next(iter(result.unit_states.values()))
    assert first.state is ExecutionState.SUBMISSION_UNCERTAIN
    assert fake.submit_count == 1
    resumed = runtime.resume(result.execution_run_id)
    assert resumed.unit_states[first.execution_unit_id].state is ExecutionState.SUBMISSION_UNCERTAIN
    assert fake.submit_count == 1
