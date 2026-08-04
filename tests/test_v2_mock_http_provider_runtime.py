from __future__ import annotations

import pytest
from types import SimpleNamespace

from guided_story_agent.v2 import DownloadDestination, HttpResponse, MockHttpProviderRuntime, MockHttpTransport, ProviderRequestContext, ProviderRuntimeError


def test_mock_submit_missing_task_id_is_uncertain() -> None:
    request_context = ProviderRequestContext(
        "request-001", "run-001", "unit-001", "idem-001", 1, "plan-001", "plan-fp", "job-001", "job-fp", 1, "movie-fp", "lineage", 10, 10, 10, "trace-001"
    )
    video_job = SimpleNamespace(job_id="job-001", provider_prompt="prompt", video_job_fingerprint="job-fp")
    provider = MockHttpProviderRuntime(MockHttpTransport([HttpResponse(202, json_data={"status": "queued"})]))
    with pytest.raises(ProviderRuntimeError) as error:
        provider.submit(video_job, request_context)
    assert error.value.category.value == "MALFORMED_RESPONSE"
    assert error.value.submission_may_have_been_accepted


def test_mock_download_rejects_path_traversal() -> None:
    context = ProviderRequestContext(
        "request-001", "run-001", "unit-001", "idem-001", 1, "plan-001", "plan-fp", "job-001", "job-fp", 1, "movie-fp", "lineage", 10, 10, 10, "trace-001"
    )
    video_job = SimpleNamespace(job_id="job-001", provider_prompt="prompt", video_job_fingerprint="job-fp")
    provider = MockHttpProviderRuntime(
        MockHttpTransport(
            [
                HttpResponse(202, json_data={"task_id": "task", "status": "queued"}),
                HttpResponse(200, json_data={"task_id": "task", "status": "completed", "output_url": "mock://artifact/task"}),
            ]
        )
    )
    job = provider.submit(video_job, context).provider_job
    job = provider.poll(job, context).provider_job
    with pytest.raises(ProviderRuntimeError):
        provider.download(job, DownloadDestination("..\\unsafe.part", "..\\unsafe.bin"), context)
