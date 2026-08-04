from __future__ import annotations

from dataclasses import replace

from guided_story_agent.v2 import (
    ExecutionBundle,
    ExecutionRuntime,
    ExecutionState,
    FakeClock,
    FakeProviderRuntime,
    FakeProviderScenario,
    ProviderRuntimeRegistry,
    RetryPolicy,
    TimeoutPolicy,
    execution_plan_fingerprint,
    execution_unit_fingerprint,
)
from test_v2_execution_bundle import _bundle


def _rewrite_bundle(bundle: ExecutionBundle, *, retry: RetryPolicy | None = None, timeout: TimeoutPolicy | None = None) -> ExecutionBundle:
    units = tuple(
        replace(
            unit,
            retry_policy=retry or unit.retry_policy,
            timeout_policy=timeout or unit.timeout_policy,
            execution_unit_fingerprint="",
        )
        for unit in bundle.execution_plan.execution_units
    )
    units = tuple(replace(unit, execution_unit_fingerprint=execution_unit_fingerprint(unit)) for unit in units)
    plan = replace(bundle.execution_plan, execution_units=units, execution_plan_fingerprint="")
    plan = replace(plan, execution_plan_fingerprint=execution_plan_fingerprint(plan))
    return replace(bundle, execution_plan=plan, bundle_fingerprint="")


def _runtime(bundle: ExecutionBundle, scenario: FakeProviderScenario):
    clock = FakeClock()
    fake = FakeProviderRuntime(scenario, provider_key="fake", clock=clock)
    return ExecutionRuntime(bundle, provider_registry=ProviderRuntimeRegistry({"fake": fake}), clock=clock), fake, clock


def test_poll_retry_does_not_submit_again_and_keeps_job_fingerprint() -> None:
    bundle = _rewrite_bundle(bundle=_bundle(), retry=RetryPolicy(max_attempts=3, retryable_error_codes=("provider_unavailable",)))
    runtime, fake, _ = _runtime(bundle, FakeProviderScenario("retryable_poll_failure"))
    run = runtime.create_run()
    result = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=60)
    assert result.status.value == "COMPLETED"
    assert fake.submit_count == len(bundle.execution_plan.execution_units)
    assert all(state.video_job_fingerprint for state in result.unit_states.values())


def test_never_complete_can_timeout_with_fake_clock() -> None:
    bundle = _rewrite_bundle(
        _bundle(),
        timeout=TimeoutPolicy(timeout_seconds=8, poll_interval_seconds=2),
    )
    runtime, _, _ = _runtime(bundle, FakeProviderScenario("never_complete"))
    run = runtime.create_run()
    result = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=20)
    first = next(iter(result.unit_states.values()))
    assert first.state is ExecutionState.TIMED_OUT
    assert result.status.value == "FAILED"
