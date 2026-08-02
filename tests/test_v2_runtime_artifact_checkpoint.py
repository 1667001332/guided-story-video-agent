from __future__ import annotations

from guided_story_agent.v2 import (
    ExecutionRuntime,
    ExecutionState,
    FakeClock,
    FakeProviderScenario,
    ProviderRuntimeRegistry,
    FakeProviderRuntime,
    InMemoryCheckpointStore,
    InMemoryExecutionEventStore,
    InMemoryRuntimeStateStore,
)
from test_v2_execution_bundle import _bundle


def test_corrupted_fake_artifact_never_completes_unit() -> None:
    clock = FakeClock()
    fake = FakeProviderRuntime(FakeProviderScenario("corrupted_artifact"), provider_key="fake", clock=clock)
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry({"fake": fake}),
        state_store=InMemoryRuntimeStateStore(),
        event_store=InMemoryExecutionEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        clock=clock,
    )
    run = runtime.create_run()
    result = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=30)
    assert next(iter(result.unit_states.values())).state is ExecutionState.CORRUPTED
    assert result.artifacts
    assert next(iter(result.artifacts.values()))["verification_status"] == "corrupted"


def test_checkpoint_fingerprint_is_stable_and_history_is_not_overwritten() -> None:
    runtime = ExecutionRuntime(
        _bundle(),
        provider_registry=ProviderRuntimeRegistry.with_fake(provider_keys=("fake",)),
        state_store=InMemoryRuntimeStateStore(),
        event_store=InMemoryExecutionEventStore(),
        checkpoint_store=InMemoryCheckpointStore(),
    )
    run = runtime.create_run()
    first = runtime.checkpoint(run.execution_run_id)
    checkpoint = runtime.checkpoint_store.load(first.latest_checkpoint_id)
    assert checkpoint.validate_fingerprint()
    second = runtime.checkpoint(run.execution_run_id)
    assert second.latest_checkpoint_id != first.latest_checkpoint_id
