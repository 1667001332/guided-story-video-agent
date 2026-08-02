from __future__ import annotations

from pathlib import Path
import shutil

from guided_story_agent.v2 import (
    ExecutionPlanCompiler,
    ExecutionRuntime,
    FakeClock,
    FilmIRBuilder,
    HttpResponse,
    MockHttpProviderRuntime,
    MockHttpTransport,
    MovieIRBuilder,
    ProviderCapabilities,
    ProviderRuntimeRegistry,
)
from test_v2_contracts import make_plan


def _mock_bundle():
    film = FilmIRBuilder().build(make_plan()).film_ir
    assert film is not None
    movie = MovieIRBuilder().build(film).movie_ir
    assert movie is not None
    result = ExecutionPlanCompiler().compile(
        movie,
        ProviderCapabilities(
            "mock-http",
            provider_profile="mock-http",
            supports_reference_images=True,
            supports_character_reference=True,
            supports_audio=True,
            supports_long_video=True,
        ),
    )
    assert result.bundle is not None
    return result.bundle


def test_execution_runtime_uses_same_path_for_mock_http() -> None:
    poll_counts: dict[str, int] = {}
    submit_count = 0

    def handler(request):
        nonlocal submit_count
        if request.method == "POST":
            submit_count += 1
            task_id = f"mock-task-{submit_count:03d}"
            return HttpResponse(202, json_data={"task_id": task_id, "status": "queued"})
        if request.url.startswith("mock://artifact/"):
            return HttpResponse(200, headers={"Content-Type": "application/octet-stream"}, content=b"MOCK-BINARY")
        task = request.url.rsplit("/", 1)[-1]
        count = poll_counts.get(task, 0) + 1
        poll_counts[task] = count
        if count == 1:
            body = {"task_id": task, "status": "queued"}
        elif count == 2:
            body = {"task_id": task, "status": "running", "progress": 0.5}
        else:
            body = {"task_id": task, "status": "completed", "output_url": f"mock://artifact/{task}"}
        return HttpResponse(200, json_data=body)

    transport = MockHttpTransport(handler=handler)
    provider = MockHttpProviderRuntime(transport, authorization="Bearer TEST_PROVIDER_SECRET_123")
    clock = FakeClock()
    artifact_root = Path(".mock-http-integration-artifacts")
    try:
        runtime = ExecutionRuntime(
            _mock_bundle(),
            provider_registry=ProviderRuntimeRegistry({"mock-http": provider}),
            clock=clock,
            artifact_root=artifact_root,
        )
        run = runtime.create_run()
        result = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=100)
        assert result.status.value == "COMPLETED"
        assert all(state.state.value == "COMPLETED" for state in result.unit_states.values())
        assert len(result.provider_jobs) == len(result.unit_states)
        assert all(job.remote_job_id.startswith("mock-task-") for job in result.provider_jobs.values())
        assert len(result.artifacts) == len(result.unit_states)
        assert transport.real_network_calls == 0
        assert all("TEST_PROVIDER_SECRET_123" not in repr(event) for event in runtime.events(result.execution_run_id))
        assert not list(artifact_root.rglob("*.mp4"))
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_mock_submit_disconnect_is_uncertain_and_not_resubmitted() -> None:
    def handler(request):
        if request.method == "POST":
            raise ConnectionError("mock response disconnected")
        raise AssertionError("uncertain submission must not poll or download")

    transport = MockHttpTransport(handler=handler)
    provider = MockHttpProviderRuntime(transport)
    clock = FakeClock()
    registry = ProviderRuntimeRegistry({"mock-http": provider})
    runtime = ExecutionRuntime(_mock_bundle(), provider_registry=registry, clock=clock, artifact_root=Path(".mock-http-uncertain-artifacts"))
    run = runtime.create_run()
    blocked = runtime.run_until_blocked_or_complete(run.execution_run_id, max_steps=20)
    assert blocked.status.value == "BLOCKED"
    assert any(state.state.value == "SUBMISSION_UNCERTAIN" for state in blocked.unit_states.values())
    restarted = ExecutionRuntime(
        runtime.execution_bundle,
        provider_registry=registry,
        state_store=runtime.state_store,
        event_store=runtime.event_store,
        checkpoint_store=runtime.checkpoint_store,
        clock=clock,
        artifact_root=Path(".mock-http-uncertain-artifacts"),
    )
    resumed = restarted.resume(blocked.execution_run_id)
    assert resumed.status.value == "BLOCKED"
    assert transport.request_count == 1
    shutil.rmtree(".mock-http-uncertain-artifacts", ignore_errors=True)
