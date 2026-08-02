from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import shutil

import pytest

from guided_story_agent.v2 import (
    FakeProviderRuntime,
    FakeProviderScenario,
    HttpResponse,
    MockHttpProviderRuntime,
    MockHttpTransport,
    ProviderRequestContext,
)


@pytest.fixture(params=("fake", "mock-http"))
def provider_and_transport(request):
    if request.param == "fake":
        return FakeProviderRuntime(FakeProviderScenario("success")), None
    transport = MockHttpTransport(
        [
            HttpResponse(202, json_data={"task_id": "mock-task-001", "status": "queued"}),
            HttpResponse(200, json_data={"task_id": "mock-task-001", "status": "queued"}),
            HttpResponse(200, json_data={"task_id": "mock-task-001", "status": "running", "progress": 0.5}),
            HttpResponse(200, json_data={"task_id": "mock-task-001", "status": "completed", "output_url": "mock://artifact/mock-task-001"}),
            HttpResponse(200, headers={"Content-Type": "application/octet-stream"}, content=b"MOCK-BINARY"),
        ]
    )
    return MockHttpProviderRuntime(transport, authorization="Bearer TEST_PROVIDER_SECRET_123"), transport


@pytest.fixture
def request_context() -> ProviderRequestContext:
    return ProviderRequestContext(
        request_id="request-001",
        execution_run_id="run-001",
        execution_unit_id="unit-001",
        idempotency_key="idem-001",
        attempt=1,
        execution_plan_id="plan-001",
        execution_plan_fingerprint="plan-fp",
        video_job_id="job-001",
        video_job_fingerprint="job-fp",
        source_movie_plan_version=1,
        source_movie_plan_fingerprint="movie-fp",
        source_movie_plan_lineage_token="lineage-token",
        submit_timeout_seconds=10,
        poll_timeout_seconds=10,
        download_timeout_seconds=10,
        trace_id="trace-001",
    )


@pytest.fixture
def video_job():
    return SimpleNamespace(
        job_id="job-001",
        provider_prompt="A test prompt",
        prompt="A test prompt",
        video_job_fingerprint="job-fp",
        duration_seconds=2.0,
        aspect_ratio="16:9",
    )


@pytest.fixture
def artifact_dir():
    path = Path(".provider-contract-test-artifacts")
    path.mkdir(parents=True, exist_ok=True)
    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
